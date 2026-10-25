"""Phase 5 integration — real detector/tracker/pose/layout on real clips."""

from __future__ import annotations

from pathlib import Path

import pytest

from odc.capture import FileReplaySource
from odc.config import CameraConfig
from odc.domain import ModelManifest
from odc.pose import MediaPipePose, NullPose
from odc.runtime import OnnxDetectorRuntime
from odc.seats import associate_track_first, load_layout
from odc.tracking import ByteTrackStyleTracker, TrackerConfig

WEIGHTS = Path("models/rtdetrv2-s-coco/onnx/model.onnx")
POSE_BUNDLE = Path("models/mediapipe-pose-lite/pose_landmarker_lite.task")
needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="no detector weights")
needs_pose = pytest.mark.skipif(not POSE_BUNDLE.exists(), reason="no pose bundle")


def _runtime(threshold: float = 0.5) -> OnnxDetectorRuntime:
    import json as _json

    info = _json.loads(Path("models/manifests/rtdetrv2-s-coco.json").read_text())
    manifest = ModelManifest(
        model_id=info["model_id"],
        family=info["family"],
        checkpoint_sha256=info["sha256"]["model.safetensors"],
        source_url=f"https://huggingface.co/{info['hf_repo']}",
        code_revision=info["hf_revision"],
        weights_license=info["weights_license"],
        dataset_terms=info["dataset_terms"],
        approved_for_commercial_use=info["approved_for_commercial_use"],
        input_profile=info["input_profile"],
        runtime_profile=info["runtime_profile"],
    )
    runtime = OnnxDetectorRuntime(threshold=threshold)
    runtime.load(manifest)
    return runtime


@needs_weights
def test_tracks_stable_and_reset_on_f1():
    runtime = _runtime()
    try:
        tracker = ByteTrackStyleTracker(TrackerConfig())
        source = FileReplaySource(CameraConfig(path="footage/F1_family_ride.mp4"))
        source.open()
        seen: dict[str, int] = {}
        for _ in range(12):
            frame = source.next()
            assert frame is not None
            persons = [d for d in runtime.infer(frame).detections if d.class_label == "person"]
            for st in tracker.update(frame, persons):
                seen[st.track_id] = seen.get(st.track_id, 0) + 1
        source.close()
        assert len(seen) >= 2, seen  # woman + baby tracks persist
        assert max(seen.values()) >= 8, seen  # stable across frames
        tracker.reset()
        assert tracker.live_tracks == 0
    finally:
        runtime.close()


@needs_weights
def test_association_variants_agree_on_clean_detections():
    runtime = _runtime()
    try:
        from odc.seats import associate_seat_first

        layout = load_layout("configs/seats/demo-f1-v0.yaml")
        source = FileReplaySource(CameraConfig(path="footage/F1_family_ride.mp4"))
        source.open()
        for _ in range(3):
            source.next()
        frame = source.next()
        assert frame is not None
        persons = [d for d in runtime.infer(frame).detections if d.class_label == "person"]
        pairs = [(f"p{i}", d.bbox) for i, d in enumerate(persons)]
        a = {x.track_id: x.seat_id for x in associate_track_first(pairs, layout)}
        b = {x.track_id: x.seat_id for x in associate_seat_first(pairs, layout)}
        assert set(a) == set(b)
        assert "child-seat-right" in set(a.values()), a
        source.close()
    finally:
        runtime.close()


@needs_pose
def test_mediapipe_pose_on_person_crop():
    runtime = _runtime()
    pose = MediaPipePose(str(POSE_BUNDLE))
    try:
        source = FileReplaySource(CameraConfig(path="footage/F1_family_ride.mp4"))
        source.open()
        frame = source.next()
        assert frame is not None
        persons = [d for d in runtime.infer(frame).detections if d.class_label == "person"]
        assert persons
        out = pose.estimate(frame, [persons[0].bbox])
        assert pose.calls >= 1
        for _idx, kpts in out.items():
            assert len(kpts) == 33
            assert all(0.0 <= k.confidence <= 1.0 for k in kpts)
        source.close()
    finally:
        pose.close()
        runtime.close()


def test_null_pose_pipeline_safe():
    source = FileReplaySource(CameraConfig(path="footage/E1_empty_cabin.mp4"))
    source.open()
    frame = source.next()
    assert frame is not None
    assert NullPose().estimate(frame, []) == {}
    source.close()


def test_bench_script_runs():
    import subprocess
    import sys

    r = subprocess.run(
        [
            sys.executable,
            "scripts/bench_track.py",
            "--clip",
            "footage/F1_family_ride.mp4",
            "--layout",
            "configs/seats/demo-f1-v0.yaml",
            "--max-frames",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr[-500:]
    assert "confirmed_tracks" in r.stdout
