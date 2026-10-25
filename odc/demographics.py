"""Gender-appearance probe (EXPERIMENTAL, D30 owner override of D29 refusal).

CLIP zero-shot on person crops. Containment, all enforced in code:
- disabled unless explicitly enabled in config;
- never executed on small/young-appearing occupants (box-area gate);
- confidence floor 0.7, else UNKNOWN;
- transient diagnostic output ONLY — never written to PLOMID, never in the
  dossier or exports, never a safety input.
Biometric-categorization law exposure (GDPR special-category / EU AI Act /
BIPA) is the owner's accepted risk decision (D30). First legal review can
remove this module outright; nothing else depends on it.
"""

from __future__ import annotations

from typing import Any

from .domain import GenderAppearance
from .logging import get_logger

log = get_logger("demographics")

PROMPT_MAN = "a man"
PROMPT_WOMAN = "a woman"
CONFIDENCE_FLOOR = 0.7
MIN_BOX_AREA = 0.04


class GenderProbe:
    """Lazy CLIP wrapper. Heavy deps import on load, never at module import."""

    def __init__(self) -> None:
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        import open_clip  # type: ignore[import-untyped]

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        log.info("gender probe loaded (CLIP ViT-B-32, eval-only weights)")

    def score(self, crop_bgr: Any) -> dict[str, float]:
        """Zero-shot P(man) / P(woman) over exactly two prompts."""
        import numpy as np
        import torch
        from PIL import Image

        assert self._model is not None, "probe not loaded"
        import cv2

        pixels: np.ndarray = np.asanyarray(crop_bgr)
        rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        image = self._preprocess(Image.fromarray(rgb)).unsqueeze(0)
        tokens = self._tokenizer([PROMPT_MAN, PROMPT_WOMAN])
        with torch.no_grad():
            image_features = self._model.encode_image(image)
            text_features = self._model.encode_text(tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(-1)[0]
        return {"man": float(probs[0]), "woman": float(probs[1])}

    def close(self) -> None:
        self._model = None


def classify_appearance(
    box_area: float, scores: dict[str, float]
) -> tuple[GenderAppearance, float]:
    """Pure rule: child-gate, confidence floor, else UNKNOWN. Tested."""
    if box_area < MIN_BOX_AREA:
        return GenderAppearance.UNKNOWN, 0.0
    best = max(scores, key=lambda k: scores[k])
    conf = scores[best]
    if conf < CONFIDENCE_FLOOR:
        return GenderAppearance.UNKNOWN, 0.0
    return (
        GenderAppearance.APPARENT_MAN if best == "man" else GenderAppearance.APPARENT_WOMAN
    ), conf


def gender_for_seats(
    rows: list[dict[str, Any]],
    crops: dict[str, tuple[Any, float]],
    probe: Any,
) -> dict[str, tuple[str, float | None]]:
    """Transient per-seat appearance verdicts from person crops.

    rows carry child/presence (child seats are always skipped); crops maps
    seat_id -> (bgr_pixels, box_area). Never raises: per-seat failures are
    skipped. Output labels are GenderAppearance values ("APPARENT_MAN" /
    "APPARENT_WOMAN" / "UNKNOWN") for display only — callers must never
    persist, aggregate or safety-gate on them.
    """
    out: dict[str, tuple[str, float | None]] = {}
    for row in rows:
        sid = row.get("seat_id")
        if not sid or row.get("child") == "Yes":
            continue
        item = crops.get(sid)
        if item is None:
            continue
        pixels, area = item
        try:
            scores = probe.score(pixels)
            label, conf = classify_appearance(float(area), scores)
        except Exception:
            continue
        out[sid] = (label.value, conf if conf else None)
    return out
