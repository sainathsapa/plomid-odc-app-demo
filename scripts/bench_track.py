"""Track + association benchmark (Phase 5) — observational, no ground truth.

Runs detector → tracker → both association variants over a clip and reports
continuity signals (track life, fragments, variant agreement, ambiguity
rate) + latency. Variant comparison, not accuracy.
Usage: .venv/bin/python scripts/bench_track.py --clip footage/R1_rear_wide.mp4 --layout configs/seats/demo-r1-v0.yaml [--max-frames N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

from odc.capture import FileReplaySource
from odc.config import CameraConfig
from odc.seats import (
    associate_seat_first,
    associate_track_first,
    load_layout,
)
from odc.tracking import ByteTrackStyleTracker, TrackerConfig


def main(argv: list[str] | None = None) -> int:
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args(argv)

    manifest_path = pathlib.Path("models/manifests/rtdetrv2-s-coco.json")
    if not manifest_path.exists():
        print("no detector manifest — skipping (weights absent)")
        return 2
    from odc.domain import ModelManifest
    from odc.runtime import OnnxDetectorRuntime

    info = json.loads(manifest_path.read_text())
    manifest = ModelManifest(
        model_id=info["model_id"],
        family=info["family"],
        checkpoint_sha256=info.get("sha256", {}).get("model.safetensors", "unset"),
        source_url=f"https://huggingface.co/{info['hf_repo']}",
        code_revision=info.get("hf_revision", "unset"),
        weights_license=info.get("weights_license", "REVIEW_REQUIRED"),
        dataset_terms=info.get("dataset_terms", "REVIEW_REQUIRED"),
        approved_for_commercial_use=bool(info.get("approved_for_commercial_use", False)),
        input_profile=info.get("input_profile", ""),
        runtime_profile=info.get("runtime_profile", ""),
    )
    layout = load_layout(args.layout)
    runtime = OnnxDetectorRuntime(threshold=args.threshold)
    runtime.load(manifest)
    tracker = ByteTrackStyleTracker(TrackerConfig())
    source = FileReplaySource(CameraConfig(path=args.clip))
    source.open()

    frames = inferred = 0
    lives: dict[str, int] = {}
    assoc_a: list = []
    assoc_b: list = []
    latencies: list[float] = []
    try:
        while frames < args.max_frames:
            frame = source.next()
            if frame is None or frame.pixels is None:
                break
            frames += 1
            t = time.monotonic()
            result = runtime.infer(frame)
            persons = [d for d in result.detections if d.class_label == "person"]
            states = tracker.update(frame, persons)
            for st in states:
                lives[st.track_id] = lives.get(st.track_id, 0) + 1
            pairs = [(st.track_id, st.last_bbox) for st in states if st.last_bbox]
            assoc_a.extend(associate_track_first(pairs, layout))
            assoc_b.extend(associate_seat_first(pairs, layout))
            latencies.append((time.monotonic() - t) * 1000)
            inferred += 1
    finally:
        source.close()
        runtime.close()

    agree = sum(
        1
        for a, b in zip(assoc_a, assoc_b, strict=False)
        if a.track_id == b.track_id and a.seat_id == b.seat_id
    )
    total = max(len(assoc_a), 1)
    latencies.sort()
    report = {
        "kind": "OBSERVATIONAL — continuity signals, no ground truth",
        "clip": args.clip,
        "layout": layout.version,
        "frames": frames,
        "confirmed_tracks": len(lives),
        "avg_track_life_frames": round(sum(lives.values()) / max(len(lives), 1), 1),
        "fragments": tracker.fragments,
        "variant_agreement": round(agree / total, 3),
        "ambiguous_a": sum(1 for a in assoc_a if a.ambiguous),
        "unassigned_a": sum(1 for a in assoc_a if a.seat_id is None),
        "latency_p50_ms": latencies[len(latencies) // 2] if latencies else 0,
        "latency_p95_ms": latencies[int(len(latencies) * 0.95)] if latencies else 0,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
