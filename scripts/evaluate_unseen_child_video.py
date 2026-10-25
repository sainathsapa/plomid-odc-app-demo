"""Run the actual :class:`odc.service.Service` on an independent video.

The repository is intentionally not used for this smoke evaluation.  A
minimal repository double reports the service's transient-only mode, while
the service still executes its production capture, detector, post-processing,
tracker, specialist, temporal state, and attribute code paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _TransientRepository:
    """Repository adapter that keeps this evaluation out of the database."""

    def health_check(self) -> bool:
        return False

    def upsert_camera_status(self, _health: object) -> None:
        return None

    def upsert_video_asset(self, _asset: object) -> None:
        return None

    def save_analytics_entry(self, _entry: object) -> None:
        return None


def summarize_child_candidate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure unconfirmed child-range evidence without counting it as detection."""
    summary: dict[str, Any] = {}
    for truth in ("child", "adult", "empty"):
        scored = [row for row in rows if row.get("ground_truth") == truth]
        candidate = sum(row.get("child_candidate") is True for row in scored)
        confirmed = sum(row.get("child") == "Yes" for row in scored)
        possible = sum(
            row.get("child_candidate") is True and row.get("child") != "Yes" for row in scored
        )
        summary[truth] = {
            "frames": len(scored),
            "candidate_yes": candidate,
            "candidate_no": len(scored) - candidate,
            "candidate_rate": candidate / max(len(scored), 1),
            "confirmed_child_yes": confirmed,
            "possible_unconfirmed": possible,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--layout", default="configs/seats/demo-c1-v0.yaml")
    parser.add_argument("--out", type=Path, default=Path("reports/unseen-child-evaluation.json"))
    parser.add_argument("--child-seats", nargs="*", default=[])
    parser.add_argument("--adult-seats", nargs="*", default=[])
    parser.add_argument("--empty-seats", nargs="*", default=[])
    parser.add_argument("--occupied-seats", nargs="*", default=[])
    parser.add_argument("--unoccupied-seats", nargs="*", default=[])
    parser.add_argument("--restraint-seats", nargs="*", default=[])
    parser.add_argument("--empty-restraint-seats", nargs="*", default=[])
    parser.add_argument("--belt-visible-seats", nargs="*", default=[])
    parser.add_argument("--belt-absent-assessable-seats", nargs="*", default=[])
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Optional JSON frame annotations: frame_annotations[frame][seat] fields occupant/restraint/belt/posture.",
    )
    parser.add_argument("--scenario", default="unannotated")
    parser.add_argument("--specialist-every-n", type=int, default=30)
    parser.add_argument("--age-device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--age-every-n", type=int, default=1)
    parser.add_argument(
        "--disable-age-model",
        action="store_true",
        help="Skip age/gender inference for occupancy, restraint, belt, and pose diagnostics.",
    )
    parser.add_argument("--sample-every-n", type=int, default=6)
    parser.add_argument("--video-title", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--publisher", default="")
    parser.add_argument("--license", default="")
    parser.add_argument("--retrieved-date", default="not-recorded")
    args = parser.parse_args()
    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")
    frame_annotations: dict[str, Any] = {}
    if args.annotations is not None:
        loaded_annotations = json.loads(args.annotations.read_text())
        frame_annotations = loaded_annotations.get("frame_annotations", {})
        if not isinstance(frame_annotations, dict):
            raise SystemExit("annotations JSON must contain an object named frame_annotations")
        allowed = {
            "occupant": {"child", "adult", "empty"},
            "restraint": {"present", "absent"},
            "belt": {"visible", "absent-assessable"},
            "posture": {
                "UPRIGHT",
                "LEANING_FORWARD",
                "LEANING_SIDEWAYS",
                "RECLINED",
                "BODY_PART_OCCLUDED",
                "UNKNOWN",
            },
        }
        for frame_id, frame_seats in frame_annotations.items():
            if not isinstance(frame_seats, dict):
                raise SystemExit(f"frame annotation {frame_id} must map seat IDs to labels")
            for seat_id, labels in frame_seats.items():
                if not isinstance(labels, dict):
                    raise SystemExit(f"annotation for {frame_id}/{seat_id} must be an object")
                for field, value in labels.items():
                    if field not in allowed or value not in allowed[field]:
                        raise SystemExit(
                            f"unsupported annotation {field}={value!r} at {frame_id}/{seat_id}"
                        )

    from odc.config import AgeConfig, AppConfig, CameraConfig, DetectionConfig, ProcessingConfig
    from odc.seats import load_layout
    from odc.service import Service

    cfg = AppConfig(
        camera=CameraConfig(path=str(args.video), camera_id="evaluation", realtime=False),
        detection=DetectionConfig(
            threshold=0.30,
            history_max=100,
            child_specialist_enabled=False,
            child_specialist_every_n=max(1, args.specialist_every_n),
        ),
        age=AgeConfig(
            enabled=not args.disable_age_model,
            device=args.age_device,
            every_n=args.age_every_n,
        ),
        processing=ProcessingConfig(sample_every_n=max(1, args.sample_every_n)),
    )
    layout = load_layout(args.layout)
    service = Service(cfg, _TransientRepository(), layout)  # type: ignore[arg-type]
    started_at = time.perf_counter()
    service.start()
    seen: dict[int, dict[str, Any]] = {}
    while service._thread is not None and service._thread.is_alive():
        with service.hub.lock:
            history = list(service.hub.history)
        for entry in history:
            if entry.seat_rows:
                seen[entry.seq] = {
                    "frame_seq": entry.seq,
                    "captured_at": entry.captured_at,
                    "processed_frame_seq": entry.seq,
                    "rows": [dict(row) for row in entry.seat_rows],
                    "detections": [dict(item) for item in entry.detections],
                    "specialist_detections": [dict(item) for item in entry.specialist_detections],
                    "specialist_decisions": [dict(item) for item in entry.specialist_decisions],
                    "specialist_frame_seq": entry.specialist_frame_seq,
                    "provider": entry.provider,
                    "pose": dict(entry.pose),
                    "face_diagnostics": dict(entry.face),
                }
        time.sleep(0.01)
    age_device = service._age_device_name
    age_model_loaded = age_device not in {"", "unavailable"}
    age_error = service._age_error
    service.stop()
    elapsed_seconds = time.perf_counter() - started_at
    latencies = sorted(float(value) for value in service.hub.latencies)
    p95_index = min(len(latencies) - 1, int(0.95 * len(latencies))) if latencies else 0

    digest = hashlib.sha256(args.video.read_bytes()).hexdigest()
    mivolo_weights = Path(cfg.age.model_path) / "model.safetensors"
    mivolo_digest = (
        hashlib.sha256(mivolo_weights.read_bytes()).hexdigest()
        if mivolo_weights.is_file()
        else None
    )
    import cv2

    media = cv2.VideoCapture(str(args.video))
    video_metadata: dict[str, Any] = {
        "resolution": {
            "width": int(media.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(media.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        },
        "fps": float(media.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(media.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    media.release()
    video_metadata["duration_seconds"] = (
        video_metadata["frame_count"] / video_metadata["fps"] if video_metadata["fps"] > 0 else None
    )
    frames: list[dict[str, Any]] = sorted(seen.values(), key=lambda item: int(item["frame_seq"]))
    for frame in frames:
        frame["video_frame_index"] = int(frame["frame_seq"])
        frame["timestamp_seconds"] = (
            int(frame["frame_seq"]) / video_metadata["fps"] if video_metadata["fps"] else None
        )
    row_list = [row for frame in frames for row in frame["rows"]]
    capability_summary = {
        "seat_rows": len(row_list),
        "presence": dict(Counter(str(row.get("presence", "Unknown")) for row in row_list)),
        "occupant_category": dict(
            Counter(str(row.get("occupant_category", "unknown")) for row in row_list)
        ),
        "child": dict(Counter(str(row.get("child", "Unknown")) for row in row_list)),
        "child_candidates": {
            "confirmed_yes": sum(row.get("child") == "Yes" for row in row_list),
            "possible_unconfirmed": sum(
                row.get("child_candidate") is True and row.get("child") != "Yes" for row in row_list
            ),
            "reason": dict(
                Counter(
                    str(row.get("child_candidate_reason", "unavailable"))
                    for row in row_list
                    if row.get("child_candidate") is True and row.get("child") != "Yes"
                )
            ),
        },
        "restraint": dict(Counter(str(row.get("restraint", "Unknown")) for row in row_list)),
        "belt": dict(Counter(str(row.get("belt", "Unknown")) for row in row_list)),
        "belt_observation": dict(
            Counter(str(row.get("belt_observation", "unavailable")) for row in row_list)
        ),
        "visibility": dict(Counter(str(row.get("visibility", "Unknown")) for row in row_list)),
        "rows_with_tracks": sum(bool(row.get("track_ids")) for row in row_list),
        "rows_with_pose": sum(bool(row.get("pose")) for row in row_list),
        "posture": dict(Counter(str(row.get("posture", "UNKNOWN")) for row in row_list)),
        "pose_skeletons": sum(len(frame.get("pose", {}).get("skeletons", [])) for frame in frames),
        "pose_keypoints": sum(
            int(row.get("pose") or 0) if isinstance(row.get("pose"), (int, float)) else 0
            for row in row_list
        ),
    }
    child_seats = set(args.child_seats)
    adult_seats = set(args.adult_seats)
    empty_seats = set(args.empty_seats)
    occupied_seats = set(args.occupied_seats) or child_seats | adult_seats
    unoccupied_seats = set(args.unoccupied_seats) or empty_seats
    restraint_seats = set(args.restraint_seats)
    empty_restraint_seats = set(args.empty_restraint_seats)
    belt_visible_seats = set(args.belt_visible_seats)
    belt_absent_seats = set(args.belt_absent_assessable_seats)
    truth_counts: dict[str, int] = {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "unknown": 0,
        "child_unknown": 0,
        "adult_unknown": 0,
        "empty_unknown": 0,
        "empty_false_positive": 0,
        "adult_false_positive": 0,
        "unscored_ground_truth": 0,
    }
    for frame in frames:
        for row in frame["rows"]:
            seat_id = str(row["seat_id"])
            frame_annotation = frame_annotations.get(str(frame["video_frame_index"]), {})
            seat_annotation = (
                frame_annotation.get(seat_id, {}) if isinstance(frame_annotation, dict) else {}
            )
            truth = seat_annotation.get("occupant") or (
                "child"
                if seat_id in child_seats
                else "adult"
                if seat_id in adult_seats
                else "empty"
                if seat_id in empty_seats
                else None
            )
            row["ground_truth"] = truth
            row["ground_truth_occupancy"] = (
                "occupied"
                if seat_id in occupied_seats
                else "empty"
                if seat_id in unoccupied_seats
                else None
            )
            row["ground_truth_restraint"] = (
                seat_annotation.get("restraint") or "present"
                if seat_id in restraint_seats
                else "absent"
                if seat_id in empty_restraint_seats
                else seat_annotation.get("restraint")
            )
            row["ground_truth_belt"] = (
                seat_annotation.get("belt") or "visible"
                if seat_id in belt_visible_seats
                else "absent-assessable"
                if seat_id in belt_absent_seats
                else seat_annotation.get("belt")
            )
            row["ground_truth_posture"] = seat_annotation.get("posture")
            if truth is None:
                truth_counts["unscored_ground_truth"] += 1
                row["ground_truth_review"] = "not scored: occupant age not independently clear"
            else:
                predicted = row.get("child")
                if truth == "child" and predicted == "Yes":
                    truth_counts["tp"] += 1
                elif truth == "child" and predicted == "No":
                    truth_counts["fn"] += 1
                elif truth in {"adult", "empty"} and predicted == "Yes":
                    truth_counts["fp"] += 1
                    key = "adult_false_positive" if truth == "adult" else "empty_false_positive"
                    truth_counts[key] += 1
                elif truth == "adult" and predicted == "No":
                    truth_counts["tn"] += 1
                elif truth == "empty" and predicted == "No":
                    truth_counts["tn"] += 1
                else:
                    truth_counts["unknown"] += 1
                    key = {
                        "child": "child_unknown",
                        "adult": "adult_unknown",
                        "empty": "empty_unknown",
                    }[truth]
                    truth_counts[key] += 1
    child_total = truth_counts["tp"] + truth_counts["fn"] + truth_counts["child_unknown"]
    adult_total = (
        truth_counts["adult_false_positive"]
        + len(
            [
                1
                for frame in frames
                for row in frame["rows"]
                if row.get("ground_truth") == "adult" and row.get("child") == "No"
            ]
        )
        + truth_counts["adult_unknown"]
    )
    empty_total = len(
        [1 for frame in frames for row in frame["rows"] if row.get("ground_truth") == "empty"]
    )
    precision = truth_counts["tp"] / max(1, truth_counts["tp"] + truth_counts["fp"])
    recall = truth_counts["tp"] / max(1, child_total)
    report_rates = {
        "child_precision": precision,
        "child_recall_unknown_counts_as_miss": recall,
        "child_f1_unknown_counts_as_miss": 2 * precision * recall / max(1e-12, precision + recall),
        "adult_only_false_positive_rate_unknown_in_denominator": (
            truth_counts["adult_false_positive"] / max(1, adult_total)
        ),
        "child_unknown_rate": truth_counts["child_unknown"] / max(1, child_total),
        "adult_unknown_rate": truth_counts["adult_unknown"] / max(1, adult_total),
        "empty_restraint_false_positive_rate": (
            truth_counts["empty_false_positive"] / max(1, empty_total)
        ),
        "empty_seat_unknown_rate": truth_counts["empty_unknown"] / max(1, empty_total),
    }
    occupancy_counts = {key: 0 for key in ("tp", "fp", "tn", "fn", "unknown")}
    restraint_counts = {key: 0 for key in ("tp", "fp", "tn", "fn", "unknown")}
    belt_counts = {key: 0 for key in ("tp", "fp", "tn", "fn", "unknown")}
    posture_counts: Counter[str] = Counter()
    posture_review: Counter[str] = Counter()
    for frame in frames:
        for row in frame["rows"]:
            seat_id = str(row["seat_id"])
            truth = row.get("ground_truth_occupancy")
            prediction = row.get("presence")
            if truth == "occupied":
                if prediction in {"OCCUPIED", "LIKELY_OCCUPIED"}:
                    occupancy_counts["tp"] += 1
                elif prediction == "EMPTY_CANDIDATE":
                    occupancy_counts["fn"] += 1
                else:
                    occupancy_counts["unknown"] += 1
            elif truth == "empty":
                if prediction in {"OCCUPIED", "LIKELY_OCCUPIED"}:
                    occupancy_counts["fp"] += 1
                elif prediction == "EMPTY_CANDIDATE":
                    occupancy_counts["tn"] += 1
                else:
                    occupancy_counts["unknown"] += 1
            restraint_truth = row.get("ground_truth_restraint")
            restraint_prediction = row.get("restraint")
            if restraint_truth is not None:
                if restraint_prediction == "Unknown":
                    restraint_counts["unknown"] += 1
                elif restraint_truth == "present" and restraint_prediction == "Yes":
                    restraint_counts["tp"] += 1
                elif restraint_truth == "present":
                    restraint_counts["fn"] += 1
                elif restraint_prediction == "Yes":
                    restraint_counts["fp"] += 1
                else:
                    restraint_counts["tn"] += 1
            belt_truth = row.get("ground_truth_belt")
            belt_prediction = row.get("belt")
            if belt_truth is not None:
                if belt_prediction == "Unknown":
                    belt_counts["unknown"] += 1
                elif belt_truth == "visible" and belt_prediction == "Detected":
                    belt_counts["tp"] += 1
                elif belt_truth == "visible":
                    belt_counts["fn"] += 1
                elif belt_prediction == "Detected":
                    belt_counts["fp"] += 1
                else:
                    belt_counts["tn"] += 1
            if row.get("posture"):
                posture_counts[str(row["posture"])] += 1
            posture_truth = row.get("ground_truth_posture")
            if posture_truth is not None:
                posture_prediction = str(row.get("posture", "UNKNOWN"))
                posture_review[f"{posture_truth}->{posture_prediction}"] += 1
    report_rates["occupancy"] = {
        **occupancy_counts,
        "recall_unknown_counts_as_miss": occupancy_counts["tp"]
        / max(1, occupancy_counts["tp"] + occupancy_counts["fn"] + occupancy_counts["unknown"]),
    }
    report_rates["restraint"] = restraint_counts
    report_rates["belt_visibility"] = belt_counts
    report_rates["posture"] = {
        "status_counts": dict(posture_counts),
        "frame_annotated_confusion": dict(posture_review),
    }
    age_latencies = sorted(
        float(prediction["inference_ms"])
        for frame in frames
        for row in frame["rows"]
        for prediction in row.get("age_predictions", [])
        if prediction.get("inference_ms") is not None
    )
    age_p95_index = (
        min(len(age_latencies) - 1, int(0.95 * len(age_latencies))) if age_latencies else 0
    )
    report: dict[str, Any] = {
        "video": str(args.video),
        "sha256": digest,
        "provenance": {
            "title": args.video_title,
            "source_url": args.source_url,
            "publisher": args.publisher,
            "license_or_reuse_terms": args.license,
            "retrieved_date": args.retrieved_date,
            "metadata": video_metadata,
            "extraction": (
                "OpenCV VideoCapture via production Service; "
                f"sample_every_n={cfg.processing.sample_every_n}"
            ),
        },
        "layout": args.layout,
        "config": {
            "detector_threshold": cfg.detection.threshold,
            "sample_every_n": cfg.processing.sample_every_n,
            "realtime_replay": cfg.camera.realtime,
            "target_fps": cfg.camera.target_fps,
            "pose_enabled": cfg.pose.enabled,
            "specialist_enabled": cfg.detection.child_specialist_enabled,
            "specialist_every_n": cfg.detection.child_specialist_every_n,
            "age_model_enabled": cfg.age.enabled,
            "age_model_path": cfg.age.model_path,
            "age_model_device": (age_device),
            "age_every_n": cfg.age.every_n,
            "age_model_checkpoint_sha256": mivolo_digest,
            "age_category_boundaries_years": {
                "infant_max_inclusive": cfg.age.infant_max_years,
                "child_adult_boundary": cfg.age.child_max_years,
                "uncertainty_margin": cfg.age.uncertainty_margin_years,
                "child_confirmed_max_inclusive": (
                    cfg.age.child_max_years - cfg.age.uncertainty_margin_years
                ),
                "adult_confirmed_min_exclusive": (
                    cfg.age.child_max_years + cfg.age.uncertainty_margin_years
                ),
            },
        },
        "prediction_semantics": {
            "age": "MiVOLO V2 regressed age point estimate; configured uncertainty band maps to unknown; not class probabilities",
            "gender": "MiVOLO V2 binary softmax output; uncalibrated apparent-gender estimate",
            "child": (
                "MiVOLO age point-estimate category on a seat-clipped person crop only "
                "when the original box overlap favors its assigned seat by >=0.10; "
                "otherwise use the full person crop and abstain on ambiguous multi-seat "
                "overlap. A single-observation child estimate conflicting with another "
                "adult track in the same seat is Unknown. Crop and original boxes are "
                "serialized separately; Unknown is not a negative result."
            ),
        },
        "frames_observed": len(frames),
        "performance": {
            "elapsed_seconds": elapsed_seconds,
            "processed_snapshots_per_second": len(frames) / max(elapsed_seconds, 1e-9),
            "inference_latency_ms_p50": statistics.median(latencies) if latencies else None,
            "inference_latency_ms_p95_recent_120": latencies[p95_index] if latencies else None,
            "mivolo_batch_latency_ms_median": (
                statistics.median(age_latencies) if age_latencies else None
            ),
            "mivolo_batch_latency_ms_p95": (
                age_latencies[age_p95_index] if age_latencies else None
            ),
            "providers": sorted({str(frame["provider"]) for frame in frames}),
            "hardware": {
                "machine": platform.machine(),
                "processor": platform.processor() or "unspecified",
                "age_device": (age_device),
            },
        },
        "model_health": {
            "age_loaded": age_model_loaded,
            "age_error": age_error,
            "mivolo_checkpoint_sha256": mivolo_digest,
        },
        "scenario": args.scenario,
        "ground_truth": {
            "child_seats": sorted(child_seats),
            "adult_seats": sorted(adult_seats),
            "empty_seats": sorted(empty_seats),
            "occupied_seats": sorted(occupied_seats),
            "unoccupied_seats": sorted(unoccupied_seats),
            "restraint_seats": sorted(restraint_seats),
            "empty_restraint_seats": sorted(empty_restraint_seats),
            "belt_visible_seats": sorted(belt_visible_seats),
            "belt_absent_assessable_seats": sorted(belt_absent_seats),
            "frame_annotation_file": str(args.annotations) if args.annotations else None,
            "frame_annotation_sha256": (
                hashlib.sha256(args.annotations.read_bytes()).hexdigest()
                if args.annotations is not None
                else None
            ),
            "method": "independent manual frame review; seat-level labels supplied by CLI",
        },
        "child_metrics": truth_counts,
        "child_candidate_metrics": summarize_child_candidate_metrics(row_list),
        "child_metric_rates": report_rates,
        "capability_summary": capability_summary,
        "frames": frames,
        "final_counts": service.hub.counts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"out": str(args.out), "frames_observed": len(frames), "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
