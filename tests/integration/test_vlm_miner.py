"""Phase 7 integration — the miner runs end-to-end on a real frame.

Slow (~1-2 min CPU): loads 694MB weights once, grounds one prompt.
Skipped without the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

CKPT = Path("models/grounding-dino-swint-ogc.pth")
needs_miner = pytest.mark.skipif(not CKPT.exists(), reason="no grounding checkpoint")


@needs_miner
def test_miner_finds_baby_seat_candidate():
    from odc.vlm import cxcywh_to_frame_xyxy, ground, load_grounding_dino, preprocess_pil

    model, tokenizer = load_grounding_dino(
        "models/grounding-dino-code/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        str(CKPT),
        device="cpu",
    )
    cap = cv2.VideoCapture("footage/C1_baby_carseat.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ok, frame = cap.read()
    cap.release()
    assert ok
    tensor, _, _ = preprocess_pil(frame)
    boxes, scores, phrases = ground(model, tokenizer, tensor, "child car seat", device="cpu")
    assert len(phrases) > 0
    assert all("car seat" in p or "seat" in p for p in phrases)
    for box in boxes:
        x0, y0, x1, y1 = cxcywh_to_frame_xyxy(tuple(float(v) for v in box))
        assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
    assert float(scores.max()) > 0.35


def test_verdicts_file_complete():
    import yaml

    verdicts = yaml.safe_load(Path("eval/attribute_verdicts.yaml").read_text())
    for attr in (
        "child_seat_any",
        "belt_any",
        "small_occupant",
        "belt_misuse",
        "object_bag",
        "blanket",
        "person_via_grounding",
    ):
        assert attr in verdicts, attr
        assert verdicts[attr]["verdict"] in (
            "SUPPORTED_FOR_DEMO",
            "PROVISIONAL",
            "UNSUPPORTED",
            "UNEVALUATED",
        )
        assert verdicts[attr]["evidence"] and verdicts[attr]["consequence"]


@needs_miner
def test_service_specialist_emits_real_child_seat_candidate():
    import time

    from odc.specialist import ChildSpecialist

    cap = cv2.VideoCapture("footage/C1_baby_carseat.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ok, frame = cap.read()
    cap.release()
    assert ok
    specialist = ChildSpecialist()
    specialist.start()
    try:
        assert specialist.available
        assert specialist.submit("specialist-test", 100, frame)
        result = None
        deadline = time.monotonic() + 90.0
        while result is None and time.monotonic() < deadline:
            result = specialist.poll()
            time.sleep(0.2)
        assert result is not None, "specialist did not complete a bounded inference job"
        assert result.status == "OK", result.error
        assert any(
            candidate.label in {"child seat", "infant seat", "child", "infant"}
            for candidate in result.candidates
        )

        empty_cap = cv2.VideoCapture("footage/E1_empty_cabin.mp4")
        empty_cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
        empty_ok, empty_frame = empty_cap.read()
        empty_cap.release()
        assert empty_ok
        assert specialist.submit("specialist-test", 50, empty_frame)
        empty_result = None
        deadline = time.monotonic() + 90.0
        while empty_result is None and time.monotonic() < deadline:
            empty_result = specialist.poll()
            time.sleep(0.2)
        assert empty_result is not None
        assert empty_result.status == "OK", empty_result.error
        assert not any(
            candidate.label in {"child", "infant"}
            for candidate in empty_result.candidates
        )
    finally:
        specialist.close()
