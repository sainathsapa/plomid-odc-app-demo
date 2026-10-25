"""SAM prompted-mask tests — real weights when present, else skip."""

from __future__ import annotations

import pathlib

import pytest

WEIGHTS = pathlib.Path("models/sam-vit-base/model.safetensors")


def test_sam_masks_real_objects_when_present():
    if not WEIGHTS.exists():
        pytest.skip("sam weights absent (documented optional precise mode)")
    import cv2

    from odc.sam import SamPromptedMasks

    sam = SamPromptedMasks()
    sam.load()
    try:
        img = cv2.imread("reports/probe/C2_child_seat.jpg")
        assert img is not None
        masks = sam.enhance(
            img,
            [
                {"label": "person", "bbox": [0.22, 0.20, 0.94, 1.0]},
                {"label": "chair", "bbox": [0.08, 0.15, 0.64, 1.0]},
            ],
        )
        assert len(masks) == 2
        for mask in masks:
            assert mask["evidence_kind"] == "sam-prompted"
            assert mask["score"] > 0.5
            assert len(mask["polygon"]) >= 3
    finally:
        sam.close()


def test_sam_degrades_cleanly():
    import numpy as np

    from odc.sam import SamPromptedMasks, mask_to_polygon

    sam = SamPromptedMasks(model_dir="/nonexistent-sam-dir")
    with pytest.raises(FileNotFoundError):
        sam.load()
    assert sam.enhance(np.zeros((64, 64, 3), dtype=np.uint8), []) == []
    assert mask_to_polygon(np.zeros((32, 32), dtype=bool)) is None
    disk = np.zeros((64, 64), dtype=bool)
    disk[(np.ogrid[:64, :64][0] - 32) ** 2 + (np.ogrid[:64, :64][1] - 32) ** 2 < 400] = True
    poly = mask_to_polygon(disk)
    assert poly is not None and len(poly) >= 3
