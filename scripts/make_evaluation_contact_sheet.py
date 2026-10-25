"""Create review contact sheets from frame-level child evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--kind",
        choices=(
            "child-miss",
            "adult-false-positive",
            "adult-review",
            "occupancy-unknown",
            "occupancy-false-positive",
            "belt-not-detected",
            "belt-unknown",
            "belt-detected",
            "belt-raw-candidate",
            "belt-raw-unclear",
            "posture-candidate",
            "posture-unknown",
            "posture-reviewed-failure",
        ),
        required=True,
    )
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args()
    if args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    report: dict[str, Any] = json.loads(args.report.read_text())
    video_path = Path(report["video"])
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    entries: list[dict[str, Any]] = report["frames"]
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in entries:
        for row in entry["rows"]:
            truth = row.get("ground_truth")
            prediction = row.get("child")
            if args.kind == "child-miss":
                matches = truth == "child" and prediction != "Yes"
            elif args.kind in {"adult-review", "adult-false-positive"}:
                matches = truth == "adult" and (args.kind == "adult-review" or prediction == "Yes")
            elif args.kind == "occupancy-unknown":
                occupancy_truth = row.get("ground_truth_occupancy") or truth
                matches = occupancy_truth in {"occupied", "empty", "child", "adult"} and row.get(
                    "presence"
                ) not in {"OCCUPIED", "LIKELY_OCCUPIED", "EMPTY_CANDIDATE", "EMPTY"}
            elif args.kind == "belt-not-detected":
                matches = row.get("belt") == "Not detected"
            elif args.kind == "belt-unknown":
                matches = row.get("belt") == "Unknown"
            elif args.kind == "belt-detected":
                matches = row.get("belt") == "Detected"
            elif args.kind == "belt-raw-candidate":
                matches = row.get("belt_observation") == "strap_candidate"
            elif args.kind == "belt-raw-unclear":
                matches = row.get("belt_observation") == "unclear"
            elif args.kind == "posture-candidate":
                matches = row.get("posture") in {
                    "UPRIGHT",
                    "RECLINED",
                    "LEANING_FORWARD",
                    "LEANING_SIDEWAYS",
                }
            elif args.kind == "posture-unknown":
                matches = row.get("posture") in {"UNKNOWN", "BODY_PART_OCCLUDED"}
            elif args.kind == "posture-reviewed-failure":
                reviewed_posture = row.get("ground_truth_posture")
                matches = reviewed_posture is not None and row.get("posture") != reviewed_posture
            else:
                occupancy_truth = row.get("ground_truth_occupancy") or truth
                matches = occupancy_truth in {"empty"} and row.get("presence") in {
                    "OCCUPIED",
                    "LIKELY_OCCUPIED",
                }
            if matches:
                selected.append((entry, row))
    if not selected:
        raise SystemExit(f"no frames match {args.kind}")
    if len(selected) > args.max_frames:
        if args.max_frames == 1:
            selected = [selected[len(selected) // 2]]
        else:
            indices = [
                round(i * (len(selected) - 1) / (args.max_frames - 1))
                for i in range(args.max_frames)
            ]
            selected = [selected[index] for index in indices]

    tile_width, tile_height = 640, 420
    columns = 3
    rows_count = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows_count), "#161616")
    draw = ImageDraw.Draw(sheet)
    for index, (entry, row) in enumerate(selected):
        elapsed = float(entry.get("timestamp_seconds", int(entry["frame_seq"]) / fps))
        fallback_frame = round(elapsed * fps)
        frame_index = int(entry.get("video_frame_index", fallback_frame))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame).resize((tile_width, tile_height - 66))
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(image, (x, y))
        labels = [
            f"GT={row.get('ground_truth')}  child={row.get('child')}  seat={row.get('seat_id')}",
            f"frame={frame_index}  time={elapsed:.2f}s  age_reason={row.get('age_reason', 'n/a')}",
        ]
        age_predictions = row.get("age_predictions", [])
        candidate_text = ", ".join(
            f"{item.get('track_id')}:{item.get('age_years')}y/{item.get('age_category')}"
            for item in age_predictions
        )
        if not candidate_text:
            candidate_text = ", ".join(
                f"{item.get('label')}:{float(item.get('score', 0.0)):.2f}"
                for item in entry.get("detections", [])[:8]
            )
        labels.append(f"tracks={','.join(row.get('track_ids', [])) or 'none'}")
        labels.append(f"age: {candidate_text[:72] or 'no current candidate'}")
        draw.multiline_text((x + 6, y + tile_height - 64), "\n".join(labels), fill="white")
    capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": len(selected),
                "source_resolution": [width, height],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
