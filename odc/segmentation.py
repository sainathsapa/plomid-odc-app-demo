"""Real-time post-processing: NMS, cabin filtering, segmentation masks, crops.

Honesty contract (master plan S3.2, S4.2):
- The active COCO detector has NO learned mask head and NO seat-belt /
  child-seat class. Masks here are *classical bbox-refined* segmentations
  (GrabCut + contour extraction, the plan's allowed "classical geometric/ROI"
  fallback), not panoptic predictions. They show the detected shape instead
  of a bare rectangle, update every frame, and carry provenance
  ``evidence_kind="classical-mask"``.
- Child / restraint / belt attributes are NEVER invented. See
  :mod:`odc.attributes` for the explicit candidate/unknown mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domain import BBox, Detection

# Distinct visual colors per seat state / object class (task S4 + S2).
# Seat states:
STATE_COLORS: dict[str, str] = {
    "OCCUPIED": "#ef4444",  # red
    "LIKELY_OCCUPIED": "#f97316",  # orange (transitional)
    "EMPTY_CANDIDATE": "#22c55e",  # green
    "UNKNOWN": "#9ca3af",  # gray
    "STALE": "#f59e0b",  # amber
    "CHILD": "#3b82f6",  # blue overlay for child
    "RESTRAINT": "#a855f7",  # purple overlay for child safety seat
}

# Per-class overlay colors for detections / masks.
CLASS_COLORS: dict[str, str] = {
    "person": "#ef4444",
    "chair": "#a855f7",
    "couch": "#a855f7",
    "backpack": "#eab308",
    "handbag": "#eab308",
    "suitcase": "#eab308",
    "laptop": "#eab308",
    "book": "#eab308",
    "bottle": "#06b6d4",
    "cup": "#06b6d4",
    "cell phone": "#06b6d4",
    "teddy bear": "#3b82f6",
}
DEFAULT_CLASS_COLOR = "#38bdf8"

# Cabin-impossible labels: full-frame vehicle hallucinations that the COCO
# detector emits inside a cabin view (measured on E1/C2/R1: whole-image
# "car"/"truck" at 0.5-0.65). Suppressing them is legitimate post-processing,
# not hiding evidence — the suppression count is reported, and the raw
# per-class counts remain available via the observational eval path.
VEHICLE_HALLUCINATION_LABELS = frozenset(
    {"car", "truck", "bus", "motorcycle", "airplane", "train", "boat"}
)
FULL_FRAME_AREA_SUPPRESS = 0.60

# Per-class score gates. Root cause of missing children / child seats:
# a single global 0.5 threshold drops small/occluded cabin evidence
# (measured: C1 baby seat chairs at 0.30-0.34, persons at 0.16-0.25).
# Runtime decodes at the *minimum* gate, then this module applies the
# per-class floor so empty/child/restraint evidence is not lost while
# high-precision person boxes stay strict.
PER_CLASS_THRESHOLDS: dict[str, float] = {
    "person": 0.35,
    "chair": 0.25,
    "couch": 0.25,
    "backpack": 0.30,
    "handbag": 0.30,
    "suitcase": 0.30,
    "laptop": 0.30,
    "book": 0.30,
    "bottle": 0.30,
    "cup": 0.30,
    "cell phone": 0.30,
    "teddy bear": 0.30,
}
DEFAULT_THRESHOLD = 0.40


def iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a.x_min, b.x_min), max(a.y_min, b.y_min)
    ix1, iy1 = min(a.x_max, b.x_max), min(a.y_max, b.y_max)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max((a.x_max - a.x_min) * (a.y_max - a.y_min), 1e-9)
    area_b = max((b.x_max - b.x_min) * (b.y_max - b.y_min), 1e-9)
    return inter / (area_a + area_b - inter)


def class_aware_nms(detections: list[Detection], iou_thresh: float = 0.5) -> list[Detection]:
    """Greedy per-class NMS. Fixes duplicate person/chair boxes measured on
    F1/R1 (up to 11 overlapping persons in one close-up frame). Pure."""
    kept: list[Detection] = []
    by_class: dict[str, list[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.class_label, []).append(d)
    for _label, dets in by_class.items():
        ordered = sorted(dets, key=lambda d: d.score, reverse=True)
        suppressed = [False] * len(ordered)
        for i, det in enumerate(ordered):
            if suppressed[i]:
                continue
            kept.append(det)
            for j in range(i + 1, len(ordered)):
                if not suppressed[j] and iou(det.bbox, ordered[j].bbox) >= iou_thresh:
                    suppressed[j] = True
    return sorted(kept, key=lambda d: d.score, reverse=True)


def apply_per_class_threshold(
    detections: list[Detection],
    overrides: dict[str, float] | None = None,
    floor: float = 0.0,
) -> list[Detection]:
    gates = dict(PER_CLASS_THRESHOLDS)
    if overrides:
        gates.update(overrides)
    return [
        d for d in detections if d.score >= max(gates.get(d.class_label, DEFAULT_THRESHOLD), floor)
    ]


def suppress_vehicle_hallucinations(
    detections: list[Detection],
) -> tuple[list[Detection], int]:
    """Drop full-frame vehicle boxes (cabin-impossible). Returns (kept, n_suppressed)."""
    kept: list[Detection] = []
    suppressed = 0
    for d in detections:
        area = max((d.bbox.x_max - d.bbox.x_min) * (d.bbox.y_max - d.bbox.y_min), 0.0)
        if d.class_label in VEHICLE_HALLUCINATION_LABELS and area >= FULL_FRAME_AREA_SUPPRESS:
            suppressed += 1
            continue
        kept.append(d)
    return kept, suppressed


def postprocess_detections(
    detections: list[Detection],
    per_class_gates: dict[str, float] | None = None,
    nms_iou: float = 0.5,
    floor: float = 0.0,
) -> tuple[list[Detection], dict[str, int]]:
    """Full decode-side fix: per-class gate -> cabin filter -> class-aware NMS.

    Returns (final detections, stats dict with suppressed/duplicates counts).
    Pure function — safe to unit test, no model weights.
    """
    stats = {"input": len(detections), "gated": 0, "suppressed": 0, "duplicates": 0}
    gated = apply_per_class_threshold(detections, per_class_gates, floor)
    stats["gated"] = len(detections) - len(gated)
    kept, suppressed = suppress_vehicle_hallucinations(gated)
    stats["suppressed"] = suppressed
    before = len(kept)
    final = class_aware_nms(kept, nms_iou)
    stats["duplicates"] = before - len(final)
    stats["output"] = len(final)
    return final, stats


@dataclass
class MaskResult:
    """One classical segmentation mask for a detection."""

    detection_idx: int
    label: str
    score: float
    # Normalized polygon (frame 0..1 coords). Bbox rectangle fallback when
    # GrabCut/contour extraction fails — still frame-derived, never invented.
    polygon: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    area: float
    color: str
    evidence_kind: str = "classical-mask"
    fallback: bool = False


def _bbox_polygon(bbox: BBox) -> list[tuple[float, float]]:
    return [
        (bbox.x_min, bbox.y_min),
        (bbox.x_max, bbox.y_min),
        (bbox.x_max, bbox.y_max),
        (bbox.x_min, bbox.y_max),
    ]


def detection_masks(
    pixels: Any,
    detections: list[Detection],
    max_masks: int = 20,
    grabcut_iters: int = 2,
) -> list[MaskResult]:
    """Classical bbox-refined masks, one per detection (up to max_masks).

    Tries GrabCut with the bbox as init rect on a bounded thumbnail
    (<=320px), then extracts the largest external contour as the polygon.
    Any failure (tiny box, degenerate GrabCut, missing cv2) falls back to
    the bbox rectangle with fallback=True — the UI renders both distinctly.
    Never raises: segmentation must never crash the live loop.
    """
    out: list[MaskResult] = []
    for idx, det in enumerate(detections[:max_masks]):
        bbox_t = (det.bbox.x_min, det.bbox.y_min, det.bbox.x_max, det.bbox.y_max)
        color = CLASS_COLORS.get(det.class_label, DEFAULT_CLASS_COLOR)
        fallback_poly = _bbox_polygon(det.bbox)
        area = max((det.bbox.x_max - det.bbox.x_min) * (det.bbox.y_max - det.bbox.y_min), 0.0)
        try:
            poly = _grabcut_polygon(pixels, det.bbox, grabcut_iters)
            if poly is None or len(poly) < 3:
                raise ValueError("no contour")
            out.append(
                MaskResult(
                    detection_idx=idx,
                    label=det.class_label,
                    score=det.score,
                    polygon=poly,
                    bbox=bbox_t,
                    area=area,
                    color=color,
                    fallback=False,
                )
            )
        except Exception:
            out.append(
                MaskResult(
                    detection_idx=idx,
                    label=det.class_label,
                    score=det.score,
                    polygon=fallback_poly,
                    bbox=bbox_t,
                    area=area,
                    color=color,
                    fallback=True,
                )
            )
    return out


def _grabcut_polygon(pixels: Any, bbox: BBox, iters: int) -> list[tuple[float, float]] | None:
    import cv2
    import numpy as np

    frame: np.ndarray = np.asanyarray(pixels)
    if frame is None or frame.size == 0:
        return None
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h < 8 or w < 8:
        return None
    x0, y0 = int(bbox.x_min * w), int(bbox.y_min * h)
    x1, y1 = int(bbox.x_max * w), int(bbox.y_max * h)
    bw, bh = x1 - x0, y1 - y0
    if bw < 6 or bh < 6:
        return None
    # Clamp + require strictly interior rect for GrabCut.
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    # Bounded thumbnail keeps per-frame cost predictable for live loop.
    scale = min(1.0, 320.0 / max(w, h))
    if scale < 1.0:
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        sx0, sy0, sx1, sy1 = (int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale))
    else:
        small, sx0, sy0, sx1, sy1 = frame, x0, y0, x1, y1
    sh, sw = int(small.shape[0]), int(small.shape[1])
    sx0, sy0 = max(0, sx0), max(0, sy0)
    sx1, sy1 = min(sw - 1, sx1), min(sh - 1, sy1)
    if sx1 - sx0 < 4 or sy1 - sy0 < 4:
        return None
    mask = np.zeros((sh, sw), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            small, mask, (sx0, sy0, sx1 - sx0, sy1 - sy0), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT
        )
    except Exception:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Restrict to the bbox region so background leakage cannot invent shape.
    roi = np.zeros_like(fg)
    roi[sy0:sy1, sx0:sx1] = fg[sy0:sy1, sx0:sx1]
    if int((roi > 0).sum()) < 25:
        return None
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(best)) < 25:
        return None
    eps = 0.008 * cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, eps, True)
    pts = approx.reshape(-1, 2)
    inv = 1.0 / scale if scale < 1.0 else 1.0
    poly: list[tuple[float, float]] = []
    for px, py in pts:
        nx = min(max(float(px) * inv / w, 0.0), 1.0)
        ny = min(max(float(py) * inv / h, 0.0), 1.0)
        poly.append((nx, ny))
    if len(poly) < 3:
        return None
    return poly[:64]


def crop_detection(pixels: Any, bbox: BBox, max_side: int = 224) -> bytes | None:
    """Clear cropped ROI JPEG for the crops strip. Transient; None on failure."""
    try:
        import cv2
        import numpy as np

        frame: np.ndarray = np.asanyarray(pixels)
        h, w = int(frame.shape[0]), int(frame.shape[1])
        x0, y0 = max(0, int(bbox.x_min * w)), max(0, int(bbox.y_min * h))
        x1, y1 = min(w, int(bbox.x_max * w)), min(h, int(bbox.y_max * h))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        crop = frame[y0:y1, x0:x1]
        scale = min(1.0, float(max_side) / max(crop.shape[0], crop.shape[1]))
        if scale < 1.0:
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok or len(buf) > 120_000:
            return None
        return bytes(buf)
    except Exception:
        return None


def mask_svg_polygon(mask: MaskResult) -> str:
    pts = " ".join(f"{x * 1000:.1f},{y * 1000:.1f}" for x, y in mask.polygon)
    dash = ' stroke-dasharray="10 6"' if mask.fallback else ""
    return (
        f'<polygon points="{pts}" fill="{mask.color}" fill-opacity="0.35" '
        f'stroke="{mask.color}" stroke-width="3"{dash}>'
        f"<title>{mask.label} {mask.score:.2f} "
        f"({'bbox-fallback' if mask.fallback else 'classical-mask'})</title></polygon>"
    )
