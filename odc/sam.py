"""Prompted instance masks (Meta SAM ViT-B, Apache-2.0, inference only).

Detector boxes prompt a real segmentation model instead of refining
rectangles classically. CPU cost (~5 s/frame) forbids the live path, so
this runs ONLY in a single-slot background worker (latest frame wins,
never queues, never blocks capture/inference) when mask_mode=precise.
Results carry their frame_seq and render with a "precise · frame #N"
badge — stale masks are labelled, never passed off as live. Anything
unavailable (no weights, load failure, bad boxes) falls back to the fast
classical masks with the reason recorded, never an exception.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .logging import get_logger

log = get_logger("sam")

MODEL_DIR = "models/sam-vit-base"
MAX_BOXES = 6
MIN_BOX_SIDE = 0.03


@lru_cache(maxsize=4)
def _load_sam_assets(model_dir: str) -> tuple[Any, Any]:
    from transformers import SamModel, SamProcessor

    if not Path(model_dir, "model.safetensors").exists():
        raise FileNotFoundError(f"missing SAM weights: {model_dir}")
    model = SamModel.from_pretrained(model_dir)
    model.eval()
    return model, SamProcessor.from_pretrained(model_dir)


def mask_to_polygon(mask01: Any) -> list[tuple[float, float]] | None:
    """Largest-contour polygon from a bool mask (frame-normalized)."""
    import cv2
    import numpy as np

    bitmap = (np.asanyarray(mask01).astype(bool)).astype("uint8") * 255
    if int((bitmap > 0).sum()) < 60:
        return None
    contours, _ = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(best)) < 60.0:
        return None
    h, w = bitmap.shape[:2]
    eps = 0.006 * cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, eps, True).reshape(-1, 2)
    poly = [(min(max(float(x) / w, 0.0), 1.0), min(max(float(y) / h, 0.0), 1.0)) for x, y in approx]
    return poly[:96] if len(poly) >= 3 else None


class SamPromptedMasks:
    """Lazy SAM wrapper. Heavy deps import on load, never at module import."""

    def __init__(self, model_dir: str = MODEL_DIR, max_boxes: int = MAX_BOXES) -> None:
        self._dir = model_dir
        self._max_boxes = max_boxes
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        self._model, self._processor = _load_sam_assets(str(Path(self._dir).resolve()))
        log.info("sam prompted masks loaded (%s)", self._dir)

    def enhance(self, pixels_bgr: Any, boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Boxes [{label, bbox:[x0,y0,x1,y1] 0..1}] -> mask dicts.

        Largest boxes first, capped. Pure outputs (no hub/service here).
        Never raises: per-box failures degrade to an empty list.
        """
        import numpy as np
        import torch
        from PIL import Image

        if self._model is None or self._processor is None or not boxes:
            return []
        try:
            frame: np.ndarray = np.asanyarray(pixels_bgr)
            im_h, im_w = int(frame.shape[0]), int(frame.shape[1])
            order = sorted(
                boxes,
                key=lambda b: (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]),
                reverse=True,
            )
            picked = []
            for b in order:
                x0, y0, x1, y1 = (float(v) for v in b["bbox"])
                if min(x1 - x0, y1 - y0) < MIN_BOX_SIDE:
                    continue
                picked.append(b)
                if len(picked) >= self._max_boxes:
                    break
            if not picked:
                return []
            pil = Image.fromarray(frame[:, :, ::-1])
            input_boxes = [
                [
                    [
                        int(b["bbox"][0] * im_w),
                        int(b["bbox"][1] * im_h),
                        int(b["bbox"][2] * im_w),
                        int(b["bbox"][3] * im_h),
                    ]
                ]
                for b in picked
            ]
            out: list[dict[str, Any]] = []
            with torch.no_grad():
                for b, prompt in zip(picked, input_boxes, strict=True):
                    inputs = self._processor(pil, input_boxes=[prompt], return_tensors="pt")
                    result = self._model(**inputs)
                    masks = self._processor.image_processor.post_process_masks(
                        result.pred_masks.cpu(),
                        inputs["original_sizes"],
                        inputs["reshaped_input_sizes"],
                    )
                    scores = result.iou_scores.detach().cpu().tolist()
                    box_scores = scores[0][0] if len(scores[0]) else []
                    if not box_scores:
                        continue
                    best = int(max(range(len(box_scores)), key=lambda i: box_scores[i]))
                    bitmap = masks[0]
                    while getattr(bitmap, "dim", lambda: 0)() > 3:
                        bitmap = bitmap[0]
                    bitmap = bitmap[best]
                    poly = mask_to_polygon(bitmap.numpy() if hasattr(bitmap, "numpy") else bitmap)
                    if poly is None:
                        continue
                    out.append(
                        {
                            "label": b["label"],
                            "score": float(box_scores[best]),
                            "polygon": [[x, y] for x, y in poly],
                            "bbox": list(b["bbox"]),
                            "color": b.get("color", "#c084fc"),
                            "fallback": False,
                            "evidence_kind": "sam-prompted",
                        }
                    )
            return out
        except Exception as e:
            log.warning("sam enhance degraded: %s", type(e).__name__)
            return []

    def close(self) -> None:
        """Keep shared SAM weights resident for subsequent sessions."""
