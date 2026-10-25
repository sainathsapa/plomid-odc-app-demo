"""Gender probe integration — real CLIP on a real crop (slow, one case)."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import cv2
import pytest

needs_clip = pytest.mark.skipif(
    not Path("models/rtdetrv2-s-coco/onnx/model.onnx").exists(), reason="no weights"
)


@needs_clip
def test_probe_abstains_or_calls_woman_on_s2():
    import json

    from odc.demographics import GenderProbe, classify_appearance
    from odc.domain import GenderAppearance, ModelManifest
    from odc.runtime import OnnxDetectorRuntime

    info = json.loads(Path("models/manifests/rtdetrv2-s-coco.json").read_text())
    manifest = ModelManifest(
        model_id=info["model_id"],
        family=info["family"],
        checkpoint_sha256="x",
        source_url="u",
        code_revision="r",
        weights_license="l",
        dataset_terms="d",
    )
    runtime = OnnxDetectorRuntime(threshold=0.5)
    probe = GenderProbe()
    try:
        runtime.load(manifest)
        probe.load()
        cap = cv2.VideoCapture("footage/S2_belt_up.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
        ok, frame = cap.read()
        cap.release()
        assert ok
        h, w = frame.shape[:2]
        from datetime import datetime

        from odc import contracts
        from odc.domain import FrameMeta, RunMode

        fr = contracts.Frame(
            meta=FrameMeta(
                source_id="c",
                seq=0,
                captured_at=datetime.now(UTC),
                width=w,
                height=h,
                mode=RunMode.TEST,
            ),
            pixels=frame,
        )
        persons = [d for d in runtime.infer(fr).detections if d.class_label == "person"]
        assert persons
        big = max(
            persons, key=lambda d: (d.bbox.x_max - d.bbox.x_min) * (d.bbox.y_max - d.bbox.y_min)
        )
        b = big.bbox
        crop = frame[
            max(int(b.y_min * h), 0) : int(b.y_max * h), max(int(b.x_min * w), 0) : int(b.x_max * w)
        ]
        scores = probe.score(crop)
        area = (b.x_max - b.x_min) * (b.y_max - b.y_min)
        label, conf = classify_appearance(area, scores)
        assert label in (GenderAppearance.APPARENT_WOMAN, GenderAppearance.UNKNOWN)
        assert not (label == GenderAppearance.APPARENT_MAN and conf > 0.7)
    finally:
        probe.close()
        runtime.close()
