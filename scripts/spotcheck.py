"""Engineering spot-check: pipeline vs manually reviewed ground truth.

OBSERVATIONAL — a measured snapshot on 7 clips with the developer's own
frame review as GT, not an accuracy pledge. Run:
    .venv/bin/python scripts/spotcheck.py [--max-frames 60]
Writes reports/spotcheck.json and prints a per-clip table. Exit 0 always:
this measures; gates live in validation datasets (Phase 9), never here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GT: dict[str, dict[str, object]] = {
    # Manually reviewed from footage frames (2026-10-14). Seat ids match
    # configs/seats/*.yaml. Belts: None = indeterminate GT (not evaluated).
    "footage/F1_family_ride.mp4": {
        "layout": "demo-f1-v0",
        "occupied": ["near-adult-left", "mid-adult-center", "child-seat-right"],
        "children": ["child-seat-right"],
        "restraints": ["child-seat-right"],  # baby seat is rear-right
        "belts": {},
    },
    "footage/C1_baby_carseat.mp4": {
        "layout": "demo-c1-v0",
        "occupied": ["infant-carrier"],
        "children": ["infant-carrier"],
        "restraints": ["infant-carrier"],
        "belts": {},
    },
    "footage/C2_child_seat.mp4": {
        "layout": "demo-c2-v0",
        "occupied": ["child-seat", "adult-right"],
        "children": ["child-seat"],
        "restraints": ["child-seat"],
        "belts": {},
    },
    "footage/R1_rear_wide.mp4": {
        "layout": "demo-r1-v0",
        "occupied": ["driver-front", "rear-left", "rear-center", "rear-right"],
        "children": ["rear-left"],  # blond kid rear-left
        "restraints": [],
        "belts": {},
    },
    "footage/E1_empty_cabin.mp4": {
        "layout": "demo-e1-v0",
        "occupied": [],
        "children": [],
        "restraints": [],
        "belts": {},
    },
    "footage/S2_belt_up.mp4": {
        "layout": "demo-s2-v0",
        "occupied": ["driver"],
        "children": [],
        "restraints": [],
        "belts": {"driver": "Detected"},
    },
    "footage/N1_night_drive.mp4": {
        "layout": "demo-n1-v0",
        "occupied": ["driver"],
        "children": [],
        "restraints": [],
        "belts": {},
    },
}

OCCUPIED = {"OCCUPIED", "LIKELY_OCCUPIED"}


def run_clip(path: str, gt: dict[str, object], max_frames: int) -> dict[str, object]:
    from odc.attributes import build_seat_map
    from odc.belt import SeatBeltTracker, detect_belt
    from odc.capture import FileReplaySource
    from odc.config import CameraConfig
    from odc.domain import ModelManifest
    from odc.runtime import OnnxDetectorRuntime, load_manifest
    from odc.seats import load_layout
    from odc.segmentation import PER_CLASS_THRESHOLDS, postprocess_detections
    from odc.state import SeatStateEngine, StateRules
    from odc.tracking import ByteTrackStyleTracker, TrackerConfig

    layout = load_layout(f"configs/seats/{gt['layout']}.yaml")
    manifest: ModelManifest = load_manifest("rtdetrv2-s-coco")
    runtime = OnnxDetectorRuntime(threshold=0.20)
    runtime.load(manifest)
    tracker = ByteTrackStyleTracker(TrackerConfig(high_thresh=0.35, low_thresh=0.15))
    engine = SeatStateEngine("ses-spot", layout, StateRules())
    belts = SeatBeltTracker()
    source = FileReplaySource(CameraConfig(path=path, camera_id="spot"))
    gates = {label: max(floor, 0.25) for label, floor in PER_CLASS_THRESHOLDS.items()}
    rows: list[dict[str, object]] = []
    # Full-coverage sampling: late-clip states (belting up, settling in)
    # matter as much as early frames. Defaults cover the whole clip.
    import cv2 as _cv2

    probe_cap = _cv2.VideoCapture(path)
    total = int(probe_cap.get(_cv2.CAP_PROP_FRAME_COUNT)) if probe_cap.isOpened() else 0
    probe_cap.release()
    stride = max(1, total // 30) if total > 0 else 2
    try:
        source.open()
        n = 0
        seen = 0
        while True:
            frame = source.next()
            if frame is None or frame.pixels is None:
                break
            n += 1
            if n % stride != 1:
                continue
            seen += 1
            if seen > max_frames:
                break
            result = runtime.infer(frame)
            final, _ = postprocess_detections(result.detections, gates, 0.5, floor=0.25)
            persons = [d for d in final if d.class_label == "person"]
            objects = [d for d in final if d.class_label != "person"]
            states = tracker.update(frame, persons)
            out = engine.update(frame, states, layout.seats, objects)
            current = build_seat_map(list(out.seats), list(out.occupants), frame.meta.seq)
            for row in current:
                sid = str(row["seat_id"])
                occ = row["presence"] in OCCUPIED
                boxes = [
                    o.last_bbox
                    for o in out.occupants
                    if o.seat_id == sid and o.last_bbox is not None
                ]
                obs = None
                if occ and boxes:
                    usable = [
                        b
                        for b in boxes
                        if (b.x_max - b.x_min) * (b.y_max - b.y_min) <= 0.70
                    ]
                    box = max(
                        usable or boxes,
                        key=lambda b: (b.x_max - b.x_min) * (b.y_max - b.y_min),
                    )
                    try:
                        obs = detect_belt(frame.pixels, box)
                    except Exception:
                        obs = None
                bst = belts.update(sid, occ, obs, None)
                row["belt"] = bst.state
            rows = current
    finally:
        source.close()
        runtime.close()
    return {"rows": rows, "frames": n, "inferred": seen}


def score_clip(path: str, gt: dict[str, object], rows: list[dict[str, object]]) -> dict[str, object]:
    by_id = {str(r["seat_id"]): r for r in rows}
    gt_occ = set(str(s) for s in gt["occupied"])  # type: ignore[union-attr]
    gt_child = set(str(s) for s in gt["children"])  # type: ignore[union-attr]
    gt_rest = set(str(s) for s in gt["restraints"])  # type: ignore[union-attr]
    pred_occ = {sid for sid, r in by_id.items() if r["presence"] in OCCUPIED}
    pred_child = {sid for sid, r in by_id.items() if r["child"] == "Yes"}
    pred_rest = {sid for sid, r in by_id.items() if r["restraint"] == "Yes"}

    def pr(pred: set[str], truth: set[str]) -> tuple[float, float]:
        tp = len(pred & truth)
        return (tp / len(pred) if pred else 1.0, tp / len(truth) if truth else 1.0)

    occ_p, occ_r = pr(pred_occ, gt_occ)
    child_p, child_r = pr(pred_child, gt_child)
    rest_p, rest_r = pr(pred_rest, gt_rest)
    belts: dict[str, object] = {}
    for sid, want in (gt["belts"] or {}).items():  # type: ignore[union-attr]
        got = by_id.get(str(sid), {}).get("belt", "Unknown")
        belts[str(sid)] = {"want": want, "got": got, "match": want == got}
    return {
        "seats": len(by_id),
        "occupied_pred": sorted(pred_occ),
        "occupancy_P": round(occ_p, 2),
        "occupancy_R": round(occ_r, 2),
        "child_P": round(child_p, 2),
        "child_R": round(child_r, 2),
        "restraint_P": round(rest_p, 2),
        "restraint_R": round(rest_r, 2),
        "belts": belts,
    }


def main(argv: list[str]) -> int:
    max_frames = int(argv[argv.index("--max-frames") + 1]) if "--max-frames" in argv else 30
    report: dict[str, object] = {"kind": "OBSERVATIONAL spot-check, not an accuracy claim"}
    print(f"{'clip':28s} seats occ-P/R child-P/R restr-P/R belts")
    for path, gt in GT.items():
        if not Path(path).exists():
            print(f"{path:28s} MISSING FILE")
            continue
        started = time.monotonic()
        out = run_clip(path, gt, max_frames)
        scores = score_clip(path, gt, out["rows"])  # type: ignore[arg-type]
        scores["frames"] = out["frames"]
        scores["seconds"] = round(time.monotonic() - started, 1)
        report[path] = scores
        belts = ",".join(
            f"{k}={v['got']}{'' if v['match'] else '!'}"
            for k, v in scores["belts"].items()  # type: ignore[union-attr]
        )
        print(
            f"{Path(path).name:28s} {scores['seats']}     "
            f"{scores['occupancy_P']}/{scores['occupancy_R']}    "
            f"{scores['child_P']}/{scores['child_R']}      "
            f"{scores['restraint_P']}/{scores['restraint_R']}      {belts or '—'}"
        )
    out_path = Path("reports/spotcheck.json")
    out_path.write_text(json.dumps(report, indent=1))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
