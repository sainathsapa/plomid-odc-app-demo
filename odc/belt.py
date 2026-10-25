"""Camera seat-belt strap detection (geometric, candidate-grade).

Why geometry instead of a learned classifier: COCO has no belt class and
this programme forbids training, so no validated belt classifier exists.
A shoulder strap IS a long, high-contrast diagonal line pair across the
torso — the detector looks for exactly that: Canny edges + probabilistic
Hough segments inside the torso ROI, filtered to diagonal angles, merged
into parallel pairs (a strap has two edges), confirmed temporally per seat.

Contract (camera-only absence is not independently verified):
- Output is strap VISIBILITY, never fastening: "Detected" means a
  strap-like line pair is visible; no strap candidate remains "Unknown"
  because this geometric detector has known false negatives on visible belts.
- A geometric clear-torso result is diagnostic evidence, not proof that no
  belt is worn; only a validated belt detector or buckle signal can support
  a reliable negative status.
- Buckle engagement is never claimed from camera evidence alone.
- A missing strap is never reported from a single frame (K-of-M gate).
- Dark clothing, night frames and partial occlusion yield "Cannot
  Determine" — never a false "no belt".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .domain import BBox
from .logging import get_logger

log = get_logger("belt")

# Torso ROI as fractions of the person box (head skipped, legs skipped).
TORSO_X0, TORSO_X1 = 0.15, 0.85
TORSO_Y0, TORSO_Y1 = 0.15, 0.70
# Close-up boxes (driver close to the lens) show head + upper chest: the
# standard window then frames the FACE while the strap lies below it, which
# produced false "no strap" verdicts on belted drivers (measured 9519139:
# ROI full of face, strap outside). Large boxes shift the window down and
# wider so the examined region is actually torso.
CLOSEUP_MIN_AREA = 0.30
CLOSEUP_X0, CLOSEUP_X1 = 0.05, 0.95
CLOSEUP_Y0, CLOSEUP_Y1 = 0.30, 0.92
MIN_ROI_PX = 60
MIN_BOX_AREA = 0.04
MAX_BOX_AREA = 0.70
MIN_BRIGHT, MAX_BRIGHT = 35.0, 220.0
# Diagonal strap angles from horizontal (degrees). Vertical seams and
# horizontal folds are rejected by this window.
MIN_ANGLE, MAX_ANGLE = 10.0, 75.0
MIN_FRAG_FRAC = 0.10
CAND_FRAC = 0.15
ANCHOR_FRAC = 0.32
SINGLE_LEN_FRAC = 0.28
MERGE_ANGLE_TOL = 8.0
MERGE_LAT_FRAC = 0.05
MERGE_GAP_FRAC = 0.15
# Parallel-pair separation as a fraction of ROI width (the two strap edges;
# wide webbing close up needs room — measured S2 ~0.2).
MIN_SEP_FRAC, MAX_SEP_FRAC = 0.03, 0.25
ANGLE_TOL = 10.0
OVERLAP_FRAC = 0.35

WINDOW = 10
DETECT_FRAC = 0.30
DETECT_MIN_VOTES = 3
CLEAR_NEED = 6
NODETECT_FRAC = 0.15

# Geometry fallback for frontality when pose/face estimators are off (the
# default): a clear, mid-size, mid-tone torso box with a plausible cabin
# aspect ratio is treated as assessable. Rear views (backs of heads), tiny
# fragments and extreme close-ups stay indeterminate — never "no belt".
# The floor admits portrait-framed cabin occupants (measured 0.29 on a
# frontal 3/4 view); slivers below it stay Unknown.
FALLBACK_MIN_AREA = 0.04
FALLBACK_MAX_AREA = 0.55
FALLBACK_MIN_RATIO = 0.27
FALLBACK_MAX_RATIO = 1.40


def estimate_frontal_fallback(bbox: BBox, brightness_ok: bool) -> bool | None:
    """Heuristic frontality when pose/face evidence is unavailable.

    Returns True (assessable) / False (likely non-frontal) / None (unknown).
    Pure and conservative: only mid-size, plausibly-proportioned, well-lit
    boxes read True; everything else stays None so the tracker keeps
    Unknown instead of risking a false "Not detected".
    """
    try:
        w = bbox.x_max - bbox.x_min
        h = bbox.y_max - bbox.y_min
        area = w * h
        if area <= 0 or h <= 0:
            return None
        if not brightness_ok:
            return None
        if not (FALLBACK_MIN_AREA <= area <= FALLBACK_MAX_AREA):
            return None
        ratio = w / h
        if FALLBACK_MIN_RATIO <= ratio <= FALLBACK_MAX_RATIO:
            return True
        return None
    except Exception:
        return None


@dataclass
class BeltObservation:
    """One frame of strap evidence for one occupant."""

    state: str  # "strap" | "clear" | "unclear"
    score: float
    angle: float | None = None
    segments: list[tuple[float, float, float, float]] = field(default_factory=list)
    roi: tuple[float, float, float, float] | None = None
    reason: str = ""


def torso_roi(
    bbox: BBox,
    im_w: int,
    im_h: int,
    shoulders: list[tuple[float, float]] | None = None,
    hips: list[tuple[float, float]] | None = None,
) -> tuple[int, int, int, int] | None:
    """Person box (or shoulder/hip keypoints when pose is on) -> pixel ROI."""
    if shoulders and len(shoulders) >= 2 and hips and len(hips) >= 2:
        xs = [p[0] for p in shoulders + hips]
        ys = [p[1] for p in shoulders + hips]
        x0, y0 = int(min(xs) * im_w), int(min(ys) * im_h)
        x1, y1 = int(max(xs) * im_w), int(max(ys) * im_h)
        pad_x, pad_y = int(0.06 * im_w), int(0.04 * im_h)
        x0, y0, x1, y1 = x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y
    else:
        area = (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)
        if area >= CLOSEUP_MIN_AREA:
            fx0, fx1, fy0, fy1 = CLOSEUP_X0, CLOSEUP_X1, CLOSEUP_Y0, CLOSEUP_Y1
        else:
            fx0, fx1, fy0, fy1 = TORSO_X0, TORSO_X1, TORSO_Y0, TORSO_Y1
        x0 = int((bbox.x_min + fx0 * (bbox.x_max - bbox.x_min)) * im_w)
        x1 = int((bbox.x_min + fx1 * (bbox.x_max - bbox.x_min)) * im_w)
        y0 = int((bbox.y_min + fy0 * (bbox.y_max - bbox.y_min)) * im_h)
        y1 = int((bbox.y_min + fy1 * (bbox.y_max - bbox.y_min)) * im_h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(im_w, x1), min(im_h, y1)
    if x1 - x0 < MIN_ROI_PX or y1 - y0 < MIN_ROI_PX:
        return None
    return (x0, y0, x1, y1)


def _seg_angle(x0: float, y0: float, x1: float, y1: float) -> float:
    import math

    ang = abs(math.degrees(math.atan2(y1 - y0, x1 - x0))) % 180.0
    return 180.0 - ang if ang > 90.0 else ang


def _lsd_segments(gray: Any) -> list[tuple[float, float, float, float]]:
    """Raw LSD level-line segments. Fast (~10 ms on torso ROIs)."""
    import cv2
    import numpy as np

    detector = cv2.createLineSegmentDetector(scale=0.8)
    lines, *_ = detector.detect(gray)
    if lines is None:
        return []
    out: list[tuple[float, float, float, float]] = []
    for row in np.asarray(lines).reshape(-1, 4):
        out.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
    return out


def _merge_collinear(
    segs: list[tuple[float, float, float, float, float]], diag: float
) -> list[tuple[float, float, float, float, float, float]]:
    """Greedy merge of same-edge fragments into long lines.

    Returns (x0, y0, x1, y1, angle, length). Fragments of one strap edge
    join; the two strap edges stay apart (lateral gate) for pair matching.
    """
    import math

    merged: list[tuple[float, float, float, float, float, float]] = []
    for seg in sorted(segs, key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1])):
        ax, ay, bx, by, ang = seg
        dx, dy = bx - ax, by - ay
        seg_len = math.hypot(dx, dy) or 1e-9
        ux, uy = dx / seg_len, dy / seg_len
        placed = False
        for i, (cx, cy, ex, ey, cang, _clen) in enumerate(merged):
            if abs(ang - cang) > MERGE_ANGLE_TOL:
                continue
            cdx, cdy = ex - cx, ey - cy
            cl = math.hypot(cdx, cdy) or 1e-9
            ux2, uy2 = cdx / cl, cdy / cl
            if abs(ux * ux2 + uy * uy2) < 0.99:
                continue
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            lat = abs((mx - cx) * -uy2 + (my - cy) * ux2)
            if lat > MERGE_LAT_FRAC * diag:
                continue
            t_new = sorted([(ax - cx) * ux2 + (ay - cy) * uy2, (bx - cx) * ux2 + (by - cy) * uy2])
            overlap = max(0.0, min(t_new[1], cl) - max(t_new[0], 0.0))
            gap = max(0.0, max(t_new[0], 0.0) - min(t_new[1], cl)) if overlap <= 0 else 0.0
            if gap > MERGE_GAP_FRAC * diag:
                continue
            lo, hi = min(t_new[0], 0.0), max(t_new[1], cl)
            nx, ny = cx + ux2 * lo, cy + uy2 * lo
            nx2, ny2 = cx + ux2 * hi, cy + uy2 * hi
            merged[i] = (nx, ny, nx2, ny2, cang, math.hypot(nx2 - nx, ny2 - ny))
            placed = True
            break
        if not placed:
            merged.append((ax, ay, bx, by, ang, seg_len))
    return merged


# Bright-band stage: a wide light strap on dark clothing fragments under
# LSD (curvature over the shoulder splits edges below the anchor gate), so
# line pairs alone miss it (measured 9519139 driver: 2/377 segments kept).
# The band cue is complementary: a large, elongated, diagonal BRIGHT region
# (PCA orientation + extent), strict shape gates, still confirmed temporally
# by the tracker. Runs only when the line stage finds no pattern.
BAND_BRIGHT_K = 1.75
BAND_MIN_AREA_FRAC = 0.015
BAND_MIN_ASPECT = 3.0
BAND_MIN_LEN_FRAC = 0.20
BAND_MAX_BRIGHT_FRAC = 0.50
# The band must stand out from its immediate surroundings (white webbing on
# dark cloth), not just be globally bright: rejects shirt stripes and pale
# panels whose contrast is local-low even when they pass the shape gates.
BAND_MIN_MARGIN = 50.0
BAND_RING_PX = 15


def _band_evidence(
    gray: Any, diag: float
) -> tuple[bool, list[tuple[float, float, float, float]]]:
    """Bright diagonal band search. Returns (found, overlay_segments) in
    ROI pixels. Pure; never raises (returns False on any degenerate input).
    """
    import math

    import cv2
    import numpy as np

    try:
        g: np.ndarray = np.asanyarray(gray)
        if g.ndim != 2 or min(g.shape) < 8:
            return False, []
        mean, std = float(g.mean()), float(g.std())
        if std < 5.0:
            return False, []
        _, binary = cv2.threshold(g, mean + BAND_BRIGHT_K * std, 255, cv2.THRESH_BINARY)
        bright_frac = float((binary > 0).mean())
        if bright_frac <= 0.0 or bright_frac > BAND_MAX_BRIGHT_FRAC:
            return False, []
        n, labels, stats, _cent = cv2.connectedComponentsWithStats(  # type: ignore[call-overload]
            binary, 8
        )
        rh, rw = g.shape
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (BAND_RING_PX * 2 + 1, BAND_RING_PX * 2 + 1)
        )
        found = False
        overlays: list[tuple[float, float, float, float]] = []
        for label in range(1, n):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < BAND_MIN_AREA_FRAC * rw * rh:
                continue
            ys, xs = np.nonzero(labels == label)
            if len(xs) < 32:
                continue
            comp_mean = float(g[ys, xs].mean())
            mask = (labels == label).astype(np.uint8)
            ring = cv2.dilate(mask, kernel) - mask
            ring_px = g[ring > 0]
            if len(ring_px) == 0:
                continue
            if comp_mean - float(ring_px.mean()) < BAND_MIN_MARGIN:
                continue
            pts = np.stack(
                [xs.astype(np.float64), ys.astype(np.float64)], axis=1
            )
            pts -= pts.mean(axis=0)
            cov = (pts.T @ pts) / len(pts)
            vals, vecs = np.linalg.eigh(cov)
            if vals[1] <= 1e-9:
                continue
            aspect = math.sqrt(vals[1] / max(vals[0], 1e-9))
            if aspect < BAND_MIN_ASPECT:
                continue
            axis = vecs[:, 1]
            along = pts @ axis
            length = float(np.percentile(along, 99) - np.percentile(along, 1))
            if length < BAND_MIN_LEN_FRAC * diag:
                continue
            ang = abs(math.degrees(math.atan2(axis[1], axis[0]))) % 180.0
            if ang > 90.0:
                ang = 180.0 - ang
            if not (MIN_ANGLE <= ang <= MAX_ANGLE):
                continue
            lo, hi = float(np.percentile(along, 1)), float(np.percentile(along, 99))
            mean_xy = np.stack(
                [xs.astype(np.float64), ys.astype(np.float64)], axis=1
            ).mean(axis=0)
            p0 = mean_xy + axis * lo
            p1 = mean_xy + axis * hi
            overlays.append((float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])))
            found = True
        return found, overlays
    except Exception:
        return False, []


def detect_belt(
    pixels: Any,
    bbox: BBox,
    shoulders: list[tuple[float, float]] | None = None,
    hips: list[tuple[float, float]] | None = None,
) -> BeltObservation:
    """Strap visibility for one occupant box. Never raises."""
    import math

    import cv2
    import numpy as np

    try:
        frame: np.ndarray = np.asanyarray(pixels)
        im_h, im_w = int(frame.shape[0]), int(frame.shape[1])
        box_area = (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)
        if box_area < MIN_BOX_AREA:
            return BeltObservation("unclear", 0.0, reason="box-too-small")
        if box_area > MAX_BOX_AREA:
            # Full-frame "person" boxes on macro/close-up clips carry no
            # torso to judge (measured S1 buckle close-up).
            return BeltObservation("unclear", 0.0, reason="box-unreliable")
        roi = torso_roi(bbox, im_w, im_h, shoulders, hips)
        if roi is None:
            return BeltObservation("unclear", 0.0, reason="roi-too-small")
        x0, y0, x1, y1 = roi
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if not (MIN_BRIGHT <= float(gray.mean()) <= MAX_BRIGHT):
            return BeltObservation("unclear", 0.0, reason="bad-lighting")
        # NOTE: no histogram equalization — it amplifies shirt texture into
        # competing fragments (measured S2). The brightness gate above keeps
        # only mid-tone ROIs where raw contrast suffices.
        rw, rh = x1 - x0, y1 - y0
        diag = math.hypot(rw, rh)
        roi_norm = (
            x0 / im_w,
            y0 / im_h,
            x1 / im_w,
            y1 / im_h,
        )
        frags: list[tuple[float, float, float, float, float]] = []
        for sx0, sy0, sx1, sy1 in _lsd_segments(gray):
            ang = _seg_angle(sx0, sy0, sx1, sy1)
            if MIN_ANGLE <= ang <= MAX_ANGLE and math.hypot(sx1 - sx0, sy1 - sy0) >= (
                MIN_FRAG_FRAC * diag
            ):
                frags.append((sx0, sy0, sx1, sy1, ang))
        # Line stage first (most specific cue): parallel pair, else a single
        # long edge. The bright-band stage below only runs when lines fail.
        line_reason = "no-diagonals"
        if frags:
            merged = _merge_collinear(frags, diag)
            # Anchor + supporter: one strong edge plus a parallel second edge.
            # (Real straps photograph with one crisp edge and one soft edge;
            # demanding two long lines misses them.)
            cands = [m for m in merged if m[5] >= CAND_FRAC * diag]
            anchors = [m for m in merged if m[5] >= ANCHOR_FRAC * diag]
            if anchors:
                # Parallel-pair search: the two edges of one strap.
                best: tuple[float, list[tuple[float, float, float, float]]] | None = None
                for ax, ay, bx, by, ang, _ln in anchors:
                    dx1, dy1 = bx - ax, by - ay
                    len1 = math.hypot(dx1, dy1) or 1e-9
                    for cx, cy, dx, dy, ang2, _ln2 in cands:
                        if (cx, cy, dx, dy) == (ax, ay, bx, by):
                            continue
                        if abs(ang - ang2) > ANGLE_TOL:
                            continue
                        mx2, my2 = (cx + dx) / 2.0, (cy + dy) / 2.0
                        dist = abs((mx2 - ax) * -dy1 + (my2 - ay) * dx1) / len1
                        if not (MIN_SEP_FRAC * rw <= dist <= MAX_SEP_FRAC * rw):
                            continue
                        t_c = sorted(
                            [(cx - ax) * dx1 + (cy - ay) * dy1, (dx - ax) * dx1 + (dy - ay) * dy1]
                        )
                        t_c = [t / len1 for t in t_c]
                        overlap = max(0.0, min(t_c[1], len1) - max(t_c[0], 0.0))
                        if overlap / len1 < OVERLAP_FRAC:
                            continue
                        pair = [
                            (
                                (x0 + ax) / im_w,
                                (y0 + ay) / im_h,
                                (x0 + bx) / im_w,
                                (y0 + by) / im_h,
                            ),
                            (
                                (x0 + cx) / im_w,
                                (y0 + cy) / im_h,
                                (x0 + dx) / im_w,
                                (y0 + dy) / im_h,
                            ),
                        ]
                        if best is None:
                            best = (0.9, pair)
                if best is not None:
                    return BeltObservation(
                        "strap", best[0], reason="parallel-pair",
                        segments=best[1], roi=roi_norm,
                    )
                longest = max(anchors, key=lambda m: m[5])
                ax, ay, bx, by, ang, _ln = longest
                if longest[5] >= SINGLE_LEN_FRAC * diag:
                    seg = [((x0 + ax) / im_w, (y0 + ay) / im_h,
                            (x0 + bx) / im_w, (y0 + by) / im_h)]
                    return BeltObservation(
                        "strap", 0.6, angle=ang, segments=seg,
                        roi=roi_norm, reason="single-long-line",
                    )
                line_reason = "no-strap-pattern"
        # Bright-band second opinion: wide straps fragment under LSD (a smooth
        # band leaves few or no edges at all), so a strict
        # elongated-bright-region check covers what lines miss. Either cue
        # still needs temporal confirmation downstream.
        band_found, band_segs = _band_evidence(gray, diag)
        if band_found and band_segs:
            norm = [
                (
                    (x0 + q0) / im_w,
                    (y0 + q1) / im_h,
                    (x0 + q2) / im_w,
                    (y0 + q3) / im_h,
                )
                for q0, q1, q2, q3 in band_segs[:2]
            ]
            return BeltObservation(
                "strap", 0.75, segments=norm, roi=roi_norm, reason="bright-band"
            )
        return BeltObservation("clear", 0.0, roi=roi_norm, reason=line_reason)
    except Exception as e:
        log.warning("belt detect degraded: %s", type(e).__name__)
        return BeltObservation("unclear", 0.0, reason="error")


@dataclass
class SeatBeltState:
    state: str = "Unknown"  # "Detected" | "Not detected" | "Unknown"
    strap_ratio: float = 0.0
    strap_frames: int = 0
    clear_frames: int = 0
    reason: str = "no-evidence-yet"
    issue: bool = False


class SeatBeltTracker:
    """Per-seat temporal strap votes (K-of-M). Fresh instance per video."""

    def __init__(
        self,
        window: int = WINDOW,
        detect_frac: float = DETECT_FRAC,
        clear_need: int = CLEAR_NEED,
    ) -> None:
        self._window = window
        self._detect_frac = detect_frac
        self._clear_need = clear_need
        self._votes: dict[str, deque[float | None]] = {}

    def update(
        self,
        seat_id: str,
        occupied: bool,
        obs: BeltObservation | None,
        frontal: bool | None = None,
    ) -> SeatBeltState:
        """Absence is evidence only with a frontal view (face visible):
        rear views (backs of heads) and profiles without a face read
        indeterminate, never "no belt". A visible strap always counts.

        frontal=None means "viewpoint unknown" (pose/face estimators off):
        strap observations still count (a strap is a strap from any angle),
        while clear observations stay indeterminate unless the caller
        supplies an explicit True. Use belt.estimate_frontal_fallback()
        at the call site to derive an honest True for well-posed torsos.
        """
        votes = self._votes.setdefault(seat_id, deque(maxlen=self._window))
        if occupied and obs is not None:
            if obs.state == "strap":
                votes.append(1.0)
            elif obs.state == "clear" and frontal is True:
                votes.append(0.0)
            else:
                votes.append(None)
        counted = [v for v in votes if v is not None]
        straps = sum(1 for v in counted if v >= 1.0)
        clears = sum(1 for v in counted if v <= 0.0)
        ratio = (sum(counted) / len(counted)) if counted else 0.0
        if not occupied:
            return SeatBeltState("Unknown", ratio, straps, clears, "no-occupant")
        if straps >= DETECT_MIN_VOTES and ratio >= self._detect_frac:
            return SeatBeltState(
                "Detected", ratio, straps, clears,
                f"strap-visible-candidate {straps}/{len(counted)} frames",
            )
        if clears >= self._clear_need and ratio <= NODETECT_FRAC:
            return SeatBeltState(
                "Unknown", ratio, straps, clears,
                f"no-strap-candidate {clears} clear frames; absence-unverified-camera-only",
                issue=False,
            )
        return SeatBeltState(
            "Unknown", ratio, straps, clears, f"indeterminate {straps}/{len(counted)} strap"
        )
