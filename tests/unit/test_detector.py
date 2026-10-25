"""Detector unit tests — pure math, no weights, no camera."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from odc import coco_labels
from odc.eval import _percentile, run_detector_eval
from odc.runtime import decode, preprocess, sigmoid


def test_coco_map():
    assert coco_labels.label_name(0) == "person"
    assert coco_labels.label_name(56) == "chair"
    assert coco_labels.label_name(999) == "class_999"
    assert {"person", "chair", "cell phone", "bottle"} <= coco_labels.CABIN_RELEVANT


def test_onnx_detector_session_is_reused_after_service_close(tmp_path, monkeypatch):
    import sys
    import types

    import odc.runtime as runtime_module
    from odc.domain import ModelManifest

    model_id = "session-cache-test"
    model_path = tmp_path / "models" / model_id / "onnx" / "model.onnx"
    model_path.parent.mkdir(parents=True)
    model_path.touch()
    manifest_path = tmp_path / "models" / "manifests" / f"{model_id}.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"providers_preference": ["CPUExecutionProvider"], "providers_blocked": {}})
    )
    created = []

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            self.closed = False
            created.append(self)

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=FakeSession,
            SessionOptions=lambda: object(),
        ),
    )
    monkeypatch.setattr(runtime_module, "ROOT", tmp_path)
    manifest = ModelManifest(
        model_id=model_id,
        family="test",
        checkpoint_sha256="test",
        source_url="local",
        code_revision="test",
        weights_license="test",
        dataset_terms="test",
    )
    first = runtime_module.OnnxDetectorRuntime()
    first.load(manifest)
    shared_session = first._session
    first.close()
    second = runtime_module.OnnxDetectorRuntime()
    second.load(manifest)
    assert len(created) == 1
    assert second._session is shared_session


def test_sigmoid_midpoint():
    assert abs(float(sigmoid(np.array([0.0]))[0]) - 0.5) < 1e-9


def test_decode_threshold_clip_and_degenerate():
    logits = np.full((3, 80), -10.0)
    logits[0, 0] = 5.0  # person, strong
    logits[1, 56] = 0.0  # chair, 0.5 — below 0.9, above 0.3
    boxes = np.array(
        [
            [0.5, 0.5, 0.4, 0.6],  # valid
            [2.0, 2.0, 0.5, 0.5],  # fully outside → clipped degenerate
            [0.1, 0.1, 0.1, 0.1],  # weak score anyway
        ]
    )
    got = decode(logits, boxes, "m", 7, threshold=0.9)
    assert len(got) == 1
    det = got[0]
    assert det.class_label == "person" and det.frame_seq == 7
    assert det.bbox.x_min == 0.3 and det.bbox.y_max == 0.8
    assert det.model_id == "m"


def test_preprocess_shape_dtype():
    bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = preprocess(bgr)
    assert out.shape == (1, 3, 640, 640) and out.dtype == np.float32
    assert out.max() == 0.0


def test_percentile_edges():
    assert _percentile([], 95) == 0.0
    assert _percentile([10.0, 20.0, 30.0, 40.0], 50) == 30.0


def test_manifest_pins_everything():
    info = json.loads(Path("models/manifests/rtdetrv2-s-coco.json").read_text())
    assert info["hf_revision"] and info["sha256"]["model.safetensors"] not in ("", "FILL-BY-FETCH")
    assert info["providers_preference"][0] == "CUDAExecutionProvider"
    assert info["providers_preference"][-1] == "CPUExecutionProvider"
    assert "CoreMLExecutionProvider" in info["providers_blocked"]
    assert info["weights_license"] == "Apache-2.0"


def test_provider_selection_prefers_gpu_skips_blocked():
    from odc.runtime import select_providers

    available = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert select_providers(
        ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"],
        {"CoreMLExecutionProvider"},
        available,
    ) == ["CPUExecutionProvider"]
    assert select_providers(
        ["CUDAExecutionProvider", "CPUExecutionProvider"], set(), ["CPUExecutionProvider"]
    ) == ["CPUExecutionProvider"]
    assert (
        select_providers(
            ["CPUExecutionProvider"], {"CPUExecutionProvider"}, ["CPUExecutionProvider"]
        )
        == []
    )


def test_degeneracy_tripwire():
    import numpy as np

    from odc.runtime import ProviderDegenerate, _reject_degenerate

    _reject_degenerate(np.array([[[0.5, -1.2, 3.0]]]), "cpu")  # healthy spread
    with pytest.raises(ProviderDegenerate):
        _reject_degenerate(np.zeros((1, 300, 80)), "coreml")  # the measured failure
    with pytest.raises(ProviderDegenerate):
        _reject_degenerate(np.full((1, 300, 80), float("nan")), "coreml")


def test_eval_report_is_observational():
    from odc.capture import FakeSource
    from odc.config import AppConfig
    from odc.domain import ModelManifest

    class _Rt:
        _threshold = 0.5
        providers_active = ["test-cpu"]

        def infer(self, frame):  # type: ignore[no-untyped-def]
            from odc import contracts

            return contracts.InferenceResult(detections=[], model_id="m")

    manifest = ModelManifest(
        model_id="m",
        family="f",
        checkpoint_sha256="s",
        source_url="u",
        code_revision="r",
        weights_license="l",
        dataset_terms="d",
    )
    report = run_detector_eval(FakeSource(n_frames=4), _Rt(), manifest, AppConfig())
    assert report["kind"].startswith("OBSERVATIONAL")
    assert report["frames_seen"] == 4
    assert report["frames_inferred"] == 1  # sample_every_n=6 samples frame 0 only
    assert report["provider"] == "test-cpu"
