"""Detector integration — real weights, real frames, real ORT session.

Needs models/rtdetrv2-s-coco/ (+ torch for the native parity reference).
Skipped when weights are absent (weightless CI stays green).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

WEIGHTS = Path("models/rtdetrv2-s-coco/onnx/model.onnx")
needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="no detector weights")


def _frames(n: int = 3) -> list:
    cap = cv2.VideoCapture("footage/F1_family_ride.mp4")
    out = []
    for _ in range(n):
        ok, frame = cap.read()
        assert ok
        out.append(frame)
    cap.release()
    return out


def _runtime(threshold: float = 0.3):
    import json as _json

    from odc.domain import ModelManifest
    from odc.runtime import OnnxDetectorRuntime

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
def test_loader_pins_cpu_provider():
    runtime = _runtime()
    try:
        assert runtime.providers_active[0] == "CPUExecutionProvider"
    finally:
        runtime.close()


@needs_weights
def test_real_frames_real_detections_with_provenance():
    from odc import contracts
    from odc.capture import FileReplaySource
    from odc.config import CameraConfig

    runtime = _runtime()
    try:
        source = FileReplaySource(CameraConfig(path="footage/F1_family_ride.mp4"))
        source.open()
        frame = source.next()
        assert frame is not None and frame.pixels is not None
        result: contracts.InferenceResult = runtime.infer(frame)
        labels = {d.class_label for d in result.detections}
        assert "person" in labels
        assert result.model_id == "rtdetrv2-s-coco"
        assert result.provider == "CPUExecutionProvider"
        assert result.duration_ms > 0
        source.close()
    finally:
        runtime.close()


@needs_weights
def test_deterministic_across_sessions():
    from odc.capture import FileReplaySource
    from odc.config import CameraConfig

    def boxes_once() -> list:
        runtime = _runtime()
        try:
            source = FileReplaySource(CameraConfig(path="footage/F1_family_ride.mp4"))
            source.open()
            frame = source.next()
            assert frame is not None
            result = runtime.infer(frame)
            source.close()
            return [
                (
                    d.class_label,
                    round(d.score, 6),
                    tuple(round(v, 6) for v in d.bbox.__dict__.values()),
                )
                for d in result.detections
            ]
        finally:
            runtime.close()

    assert boxes_once() == boxes_once()


@needs_weights
def test_parity_vs_frozen_native_reference():
    """ORT-CPU vs frozen MPS-native reference (tests/fixtures/).

    Frozen because torch MPS is nondeterministic ACROSS processes in
    low-end logits (0.0 within a process, ~5 across) while top detections
    agree — the deployment path (ORT-CPU) is unaffected and stable.
    Regenerate only with model/checkpoint/preprocess changes.
    """

    from odc.runtime import preprocess as ort_preprocess

    ref = np.load("tests/fixtures/rtdetrv2_native_ref.npz")
    runtime = _runtime()
    try:
        from odc.capture import FileReplaySource
        from odc.config import CameraConfig

        source = FileReplaySource(CameraConfig(path="footage/F1_family_ride.mp4"))
        source.open()
        for i in range(3):
            frame = source.next()
            assert frame is not None and frame.pixels is not None
            x = ort_preprocess(frame.pixels)
            logits, boxes = runtime._session.run(["logits", "pred_boxes"], {"images": x})[:2]
            dl = float(np.abs(logits - ref[f"logits_{i}"]).max())
            db = float(np.abs(boxes - ref[f"boxes_{i}"]).max())
            assert db < 1e-3, f"frame{i} max|dbox|={db}"
            assert dl < 1e-2, f"frame{i} max|dlogit|={dl}"
        source.close()
    finally:
        runtime.close()


@needs_weights
def test_eval_report_measured():
    from odc.capture import FileReplaySource
    from odc.config import AppConfig, CameraConfig
    from odc.domain import ModelManifest
    from odc.eval import run_detector_eval

    runtime = _runtime(threshold=0.5)
    try:
        cfg = AppConfig(camera=CameraConfig(path="footage/F1_family_ride.mp4"))
        manifest = ModelManifest(
            model_id="rtdetrv2-s-coco",
            family="rt-detr-v2",
            checkpoint_sha256="x",
            source_url="u",
            code_revision="r",
            weights_license="Apache-2.0",
            dataset_terms="coco",
        )
        report = run_detector_eval(
            FileReplaySource(cfg.camera), runtime, manifest, cfg, max_frames=6
        )
        assert report["frames_seen"] == 6
        assert report["frames_inferred"] == 1  # sample_every_n=6 samples frame 0
        assert report["latency_p50_ms"] > 0 and report["latency_p95_ms"] >= report["latency_p50_ms"]
        assert report["provider"] == "CPUExecutionProvider"
    finally:
        runtime.close()
