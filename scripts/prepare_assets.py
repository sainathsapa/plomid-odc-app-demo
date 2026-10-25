"""Download the runtime assets ODC needs outside the model manifests.

Covers the pieces that are not Hugging Face snapshots:

- MediaPipe pose + face landmarker task bundles (local weights only)
- The staged demo/eval footage clips listed in footage/README.md

Model checkpoints are fetched separately with scripts/fetch_models.py, and the
ONNX detector export is produced by scripts/export_onnx.py. Both are checked
here so a single call reports the complete readiness state.

Every download is atomic (`.part` then rename) and verified by size, then video
clips are opened with OpenCV to prove they decode. Re-running is a no-op for
files already present and valid; pass --force to refetch.

Usage: .venv/bin/python scripts/prepare_assets.py [--force] [--check]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOOTAGE = ROOT / "footage"

CLIPS: dict[str, tuple[str, list[str]]] = {
    # name: (source page for provenance, direct download candidates)
    "F1_family_ride.mp4": (
        "https://www.pexels.com/video/mom-and-dad-with-baby-inside-moving-car-6181413/",
        [
            "https://videos.pexels.com/video-files/6181413/6181413-hd_1920_1080_25fps.mp4",
            "https://www.pexels.com/download/video/6181413/",
        ],
    ),
    "C1_baby_carseat.mp4": (
        "https://www.pexels.com/video/man-strapping-his-baby-in-a-car-seat-6181408/",
        [
            "https://videos.pexels.com/video-files/6181408/6181408-hd_1920_1080_25fps.mp4",
            "https://www.pexels.com/download/video/6181408/",
        ],
    ),
    "C2_child_seat.mp4": (
        "https://www.pexels.com/video/a-child-smiling-at-the-camera-while-sitting-in-the-car-6300815/",
        [
            "https://videos.pexels.com/video-files/6300815/6300815-hd_1920_1080_25fps.mp4",
            "https://www.pexels.com/download/video/6300815/",
        ],
    ),
    "E1_empty_cabin.mp4": (
        "https://www.pexels.com/video/the-interior-of-a-parked-bmw-car-3757013/",
        [
            "https://videos.pexels.com/video-files/3757013/3757013-hd_1920_1080_24fps.mp4",
            "https://www.pexels.com/download/video/3757013/",
        ],
    ),
    "N1_night_drive.mp4": (
        "https://www.pexels.com/video/man-driving-a-car-at-night-3554563/",
        [
            "https://videos.pexels.com/video-files/3554563/3554563-hd_1920_1080_30fps.mp4",
            "https://www.pexels.com/download/video/3554563/",
        ],
    ),
    "S1_seatbelt_closeup.mp4": (
        "https://www.pexels.com/video/close-up-footage-of-a-person-wearing-seatbelt-8155681/",
        [
            "https://videos.pexels.com/video-files/8155681/8155681-hd_1920_1080_25fps.mp4",
            "https://www.pexels.com/download/video/8155681/",
        ],
    ),
    "S2_belt_up.mp4": (
        "https://www.pexels.com/video/young-woman-putting-on-seat-belt-5391318/",
        [
            "https://videos.pexels.com/video-files/5391318/5391318-hd_1920_1080_30fps.mp4",
            "https://www.pexels.com/download/video/5391318/",
        ],
    ),
    "R1_rear_wide.mp4": (
        "https://www.pexels.com/video/group-journey-inside-a-busy-taxi-van-38684176/",
        [
            "https://videos.pexels.com/video-files/38684176/38684176-hd_1920_1080_25fps.mp4",
            "https://www.pexels.com/download/video/38684176/",
        ],
    ),
}

TASK_BUNDLES: dict[str, tuple[str, str]] = {
    "mediapipe-pose-lite/pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "MediaPipe Pose Landmarker Lite (float16)",
    ),
    "mediapipe-face/face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
        "MediaPipe Face Landmarker (float16)",
    ),
}

MODEL_FILES: list[tuple[Path, str]] = [
    (ROOT / "models" / "rtdetrv2-s-coco" / "onnx" / "model.onnx", "RT-DETRv2-S ONNX detector"),
    (ROOT / "models" / "mivolo-v2" / "model.safetensors", "MiVOLO V2 age/gender checkpoint"),
]


def _verify_video(path: Path) -> str:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("OpenCV cannot open the clip")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("clip reports no decodable frames")
        return f"{width}x{height}, {frames} frames"
    finally:
        capture.release()


def _download(dest: Path, candidates: list[str], label: str) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "Mozilla/5.0 (plomid-odc asset fetch)"}
    errors: list[str] = []
    for url in candidates:
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length") or 0)
                written = 0
                with part.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                if total and written != total:
                    raise RuntimeError(f"short download {written}/{total} bytes")
                if written < 256 * 1024:
                    raise RuntimeError(f"suspiciously small download ({written} bytes)")
            part.replace(dest)
            print(f"  {label}: {written / 1e6:.1f} MB <- {url}")
            return
        except Exception as exc:  # try the next source
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            part.unlink(missing_ok=True)
    raise SystemExit(f"could not download {label}:\n  " + "\n  ".join(errors))


def prepare(force: bool = False) -> int:
    problems: list[str] = []

    print("footage clips")
    for name, (page, urls) in CLIPS.items():
        dest = FOOTAGE / name
        if dest.exists() and not force:
            try:
                detail = _verify_video(dest)
                print(f"  {name}: present ({detail})")
                continue
            except Exception as exc:
                print(f"  {name}: invalid ({exc}); refetching")
        print(f"  {name}: fetching from {page}")
        _download(dest, urls, name)
        try:
            print(f"  {name}: verified ({_verify_video(dest)})")
        except Exception as exc:
            problems.append(f"{name}: downloaded but does not decode: {exc}")

    print("mediapipe task bundles")
    for relative, (url, label) in TASK_BUNDLES.items():
        dest = ROOT / "models" / relative
        if dest.exists() and not force and dest.stat().st_size > 1024 * 1024:
            print(f"  {relative}: present ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"  {relative}: fetching {label}")
        _download(dest, [url], relative)

    print("model checkpoints")
    for path, label in MODEL_FILES:
        if path.exists() and path.stat().st_size > 1024 * 1024:
            print(f"  {path.relative_to(ROOT)}: present ({path.stat().st_size / 1e6:.1f} MB)")
        else:
            problems.append(
                f"missing {label}: {path.relative_to(ROOT)} "
                "(run scripts/fetch_models.py, then scripts/export_onnx.py)"
            )

    if problems:
        print("asset problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("all runtime assets present and verified")
    return 0


def check() -> int:
    """Report presence without downloading anything."""
    missing = 0
    for name in CLIPS:
        dest = FOOTAGE / name
        ok = dest.exists()
        print(f"{'OK  ' if ok else 'MISS'} footage/{name}")
        missing += 0 if ok else 1
    for relative in TASK_BUNDLES:
        dest = ROOT / "models" / relative
        ok = dest.exists() and dest.stat().st_size > 1024 * 1024
        print(f"{'OK  ' if ok else 'MISS'} models/{relative}")
        missing += 0 if ok else 1
    for path, label in MODEL_FILES:
        ok = path.exists() and path.stat().st_size > 1024 * 1024
        print(f"{'OK  ' if ok else 'MISS'} {path.relative_to(ROOT)} ({label})")
        missing += 0 if ok else 1
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="refetch files that already exist")
    parser.add_argument("--check", action="store_true", help="report only; download nothing")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    return prepare(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
