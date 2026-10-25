"""Seat geometry + track↔seat association (Phase 5).

Coordinates are normalized (0..1) with a versioned camera profile — never
hardcoded pixels. Association preserves ambiguity: near-ties yield
seat_id None (→ UNKNOWN downstream) instead of a forced, flickering seat.
Two variants for the Phase 5 benchmark: track-first and seat-ROI-first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .domain import BBox, SeatGeometry
from .logging import get_logger

log = get_logger("seats")


def load_layout(path: str) -> SeatLayout:
    """Load + validate a versioned seat layout YAML."""
    import pathlib

    data = yaml.safe_load(pathlib.Path(path).read_text())
    layout = SeatLayout(**data)
    for seat in layout.seats:
        if seat.camera_profile != layout.camera_profile:
            raise ValueError(f"seat {seat.seat_id} profile mismatch")
    return layout


class SeatLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_profile: str
    version: str
    seats: list[SeatGeometry] = Field(default_factory=list)
    layout_type: str = "custom"
    orientation: str = "RHD"
    orientation_reason: str | None = None
    display_name: str | None = None


def apply_orientation(layout: SeatLayout, orientation: str) -> SeatLayout:
    """Apply an explicit steering-side choice without changing seat identity.

    Display order and canonical ``seat_N`` keys remain unchanged. Only the
    physical driver/front-passenger role is swapped for a left-hand-drive
    confirmation; this keeps geometry independent from vehicle orientation.
    ``UNKNOWN`` deliberately leaves the existing metadata untouched.
    """
    if orientation not in ("RHD", "LHD"):
        return layout
    seats = list(layout.seats)
    front = [
        seat
        for seat in seats
        if seat.row == 1 and seat.physical_role in ("driver", "front_passenger")
    ]
    if len(front) == 2:
        # Use column order, not list order, as the stable visual boundary.
        left, right = sorted(front, key=lambda seat: (seat.col, seat.seat_id))
        roles = {
            left.seat_id: "driver" if orientation == "RHD" else "front_passenger",
            right.seat_id: "front_passenger" if orientation == "RHD" else "driver",
        }
        seats = [
            seat.model_copy(
                update={"physical_role": roles.get(seat.seat_id, seat.physical_role)}
            )
            for seat in seats
        ]
    return layout.model_copy(update={"orientation": orientation, "seats": seats})


def seat_display_name(seat: SeatGeometry | None) -> str:
    """Return the stable human label without changing the persistence key."""
    if seat is None:
        return "Unknown seat"
    label = getattr(seat, "label", None)
    if label:
        return str(label)
    number = getattr(seat, "display_number", None)
    if number is not None:
        return f"Seat {number}"
    return str(getattr(seat, "seat_id", "Unknown seat"))


@dataclass(frozen=True)
class Association:
    track_id: str
    seat_id: str | None
    score: float
    runner_up: str | None = None
    ambiguous: bool = False


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0:
            inside = not inside
    return inside


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    """Shoelace area in normalized (0..1) units. Pure, never raises on valid input."""
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def polygon_bbox(polygon: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def clip_bbox_to_seat_region(bbox: BBox, seat: SeatGeometry) -> BBox | None:
    """Clip a track box to its assigned seat-zone bounds for per-seat models.

    This is a crop-localization step only. The original person box and track
    association remain unchanged; a non-intersecting crop yields no evidence.
    """
    x0, y0, x1, y1 = polygon_bbox(seat.polygon)
    clipped = (
        max(bbox.x_min, x0),
        max(bbox.y_min, y0),
        min(bbox.x_max, x1),
        min(bbox.y_max, y1),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return BBox(
        x_min=clipped[0],
        y_min=clipped[1],
        x_max=clipped[2],
        y_max=clipped[3],
    )


def seat_centroid(seat: SeatGeometry) -> tuple[float, float]:
    xs = [p[0] for p in seat.polygon]
    ys = [p[1] for p in seat.polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _clip_polygon(subject: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove duplicate consecutive points; keeps clipping stable."""
    out: list[tuple[float, float]] = []
    for p in subject:
        if not out or abs(out[-1][0] - p[0]) > 1e-12 or abs(out[-1][1] - p[1]) > 1e-12:
            out.append(p)
    return out


def polygon_intersection_area(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Convex-clip intersection area (Sutherland–Hodgman against each edge of b).

    Seat zones are near-convex quads; for concave inputs this is a
    conservative approximation, which is the correct bias for an overlap
    WARNING (better to flag a suspect pair than to miss it). Pure.
    """
    subj = _clip_polygon([(float(x), float(y)) for x, y in a])
    clip = _clip_polygon([(float(x), float(y)) for x, y in b])
    if len(subj) < 3 or len(clip) < 3:
        return 0.0

    def _inside(p: tuple[float, float], e0: tuple[float, float], e1: tuple[float, float]) -> bool:
        # Clip polygon is expected CCW or CW; use signed area to pick the interior side.
        return (e1[0] - e0[0]) * (p[1] - e0[1]) - (e1[1] - e0[1]) * (p[0] - e0[0]) >= -1e-12

    def _intersect(
        s: tuple[float, float], e: tuple[float, float],
        e0: tuple[float, float], e1: tuple[float, float],
    ) -> tuple[float, float]:
        dx, dy = e[0] - s[0], e[1] - s[1]
        ex, ey = e1[0] - e0[0], e1[1] - e0[1]
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-12:
            return ((s[0] + e[0]) / 2.0, (s[1] + e[1]) / 2.0)
        t = ((e0[0] - s[0]) * ey - (e0[1] - s[1]) * ex) / denom
        return (s[0] + t * dx, s[1] + t * dy)

    # Ensure clip is CCW so _inside uses the left side consistently.
    signed = sum(
        clip[i][0] * clip[(i + 1) % len(clip)][1] - clip[(i + 1) % len(clip)][0] * clip[i][1]
        for i in range(len(clip))
    )
    if signed < 0:
        clip = list(reversed(clip))
    output = subj
    for i in range(len(clip)):
        e0, e1 = clip[i], clip[(i + 1) % len(clip)]
        if not output:
            break
        nxt: list[tuple[float, float]] = []
        prev = output[-1]
        prev_in = _inside(prev, e0, e1)
        for cur in output:
            cur_in = _inside(cur, e0, e1)
            if cur_in:
                if not prev_in:
                    nxt.append(_intersect(prev, cur, e0, e1))
                nxt.append(cur)
            elif prev_in:
                nxt.append(_intersect(prev, cur, e0, e1))
            prev, prev_in = cur, cur_in
        output = nxt
    if len(output) < 3:
        return 0.0
    return polygon_area(output)


def polygon_iou(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Intersection-over-union of two normalized polygons (0..1). Pure."""
    inter = polygon_intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = max(polygon_area(a) + polygon_area(b) - inter, 1e-12)
    return min(1.0, inter / union)


def find_overlapping_seats(
    layout: SeatLayout, iou_threshold: float = 0.05
) -> list[dict[str, Any]]:
    """Detect seat pairs whose zones overlap/intersect.

    Returns [{seat_a, seat_b, iou, intersection_area}] sorted by IoU desc.
    Pure — the calibration and seat-map editors call this for live warnings,
    and layouts.validate_layout_geometry() reuses it server-side.
    """
    out: list[dict[str, Any]] = []
    seats = list(layout.seats)
    for i in range(len(seats)):
        for j in range(i + 1, len(seats)):
            a, b = seats[i], seats[j]
            try:
                iou = polygon_iou([(float(x), float(y)) for x, y in a.polygon],
                                  [(float(x), float(y)) for x, y in b.polygon])
            except Exception:
                continue
            if iou >= iou_threshold:
                try:
                    inter = polygon_intersection_area(
                        [(float(x), float(y)) for x, y in a.polygon],
                        [(float(x), float(y)) for x, y in b.polygon],
                    )
                except Exception:
                    inter = 0.0
                out.append({
                    "seat_a": a.seat_id, "seat_b": b.seat_id,
                    "iou": round(float(iou), 4), "intersection_area": round(float(inter), 6),
                })
    out.sort(key=lambda d: float(d["iou"]), reverse=True)
    return out


def order_seats_front_to_rear(layout: SeatLayout) -> list[list[str]]:
    """Group seat ids into front→rear rows, left→right within each row.

    Row threshold adapts to the median seat height (not a fixed 0.22), so
    close-up and wide cabin geometries cluster correctly. Stored row/col
    hints win when present on every seat; otherwise centroids decide.
    Pure.
    """
    seats = list(layout.seats)
    if not seats:
        return []
    if all(s.row is not None for s in seats):
        rows: dict[int, list[SeatGeometry]] = {}
        for s in seats:
            rows.setdefault(int(s.row or 0), []).append(s)
        ordered: list[list[str]] = []
        for key in sorted(rows):
            group = rows[key]
            group.sort(key=lambda s: (s.col if s.col is not None else seat_centroid(s)[0]))
            ordered.append([s.seat_id for s in group])
        return ordered
    cents = [(s.seat_id, *seat_centroid(s)) for s in seats]
    heights = []
    for s in seats:
        _, y0, _, y1 = polygon_bbox([(float(x), float(y)) for x, y in s.polygon])
        heights.append(max(y1 - y0, 1e-6))
    heights.sort()
    median_h = heights[len(heights) // 2]
    # Two seats share a row when their y-centroids differ by less than
    # half a typical seat height (clamped to a sane band).
    band = min(0.30, max(0.08, median_h * 0.5))
    cents.sort(key=lambda t: t[2])
    grouped: list[list[tuple[str, float, float]]] = []
    for item in cents:
        if grouped:
            mean_y = sum(t[2] for t in grouped[-1]) / len(grouped[-1])
            if abs(item[2] - mean_y) < band:
                grouped[-1].append(item)
                continue
        grouped.append([item])
    for row in grouped:
        row.sort(key=lambda t: t[1])
    return [[sid for sid, _, _ in row] for row in grouped]


def validate_seat_layout(layout: SeatLayout) -> list[str]:
    """Human-readable geometry warnings: overlaps, duplicates, bad polygons.

    Pure. Empty = clean. Callers surface these as warnings (never silent),
    and the editors block saving only on hard errors from layouts.py.
    """
    warnings: list[str] = []
    seen: dict[str, int] = {}
    for s in layout.seats:
        seen[s.seat_id] = seen.get(s.seat_id, 0) + 1
    for sid, count in seen.items():
        if count > 1:
            warnings.append(f"duplicate seat_id {sid!r} ({count} zones share it)")
    for s in layout.seats:
        try:
            area = polygon_area([(float(x), float(y)) for x, y in s.polygon])
        except Exception:
            warnings.append(f"{s.seat_id}: invalid polygon")
            continue
        if area < 0.005:
            warnings.append(f"{s.seat_id}: zone too small (area {area:.4f}) — likely mis-traced")
    for pair in find_overlapping_seats(layout):
        warnings.append(
            f"overlap: {pair['seat_a']} ↔ {pair['seat_b']} "
            f"(IoU {pair['iou']}) — reposition one zone"
        )
    if not layout.seats:
        warnings.append("no seats: map is uncalibrated")
    return warnings


def _bbox_overlap_area(bbox: BBox, polygon: list[tuple[float, float]]) -> float:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    ix0, iy0 = max(bbox.x_min, min(xs)), max(bbox.y_min, min(ys))
    ix1, iy1 = min(bbox.x_max, max(xs)), min(bbox.y_max, max(ys))
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def _score_bbox_seat(bbox: BBox, seat: SeatGeometry) -> float:
    cx, cy = (bbox.x_min + bbox.x_max) / 2, (bbox.y_min + bbox.y_max) / 2
    center = 1.0 if point_in_polygon(cx, cy, seat.polygon) else 0.0
    overlap = _bbox_overlap_area(bbox, seat.polygon)
    area = max((bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min), 1e-9)
    return 0.7 * center + 0.3 * min(overlap / area, 1.0)


def candidate_seats_for_bbox(
    bbox: BBox, layout: SeatLayout, ambiguity_margin: float = 0.15
) -> list[str]:
    """Return only seats whose scores compete for this box.

    This is used to scope visibility uncertainty. A person between two seat
    zones must not prevent unrelated seats elsewhere in the cabin from
    accumulating valid empty-seat evidence.
    """
    scored = sorted(
        ((seat.seat_id, _score_bbox_seat(bbox, seat)) for seat in layout.seats),
        key=lambda item: item[1],
        reverse=True,
    )
    if not scored or scored[0][1] <= 0.0:
        return []
    best_score = scored[0][1]
    return [
        seat_id
        for seat_id, score in scored
        if score > 0.0 and best_score - score < ambiguity_margin
    ]


def associate_track_first(
    tracks: list[tuple[str, BBox]],
    layout: SeatLayout,
    ambiguity_margin: float = 0.15,
) -> list[Association]:
    """Variant A: detection + track, then attach each track to a seat."""
    if not layout.seats:
        # Uncalibrated geometry: no seat can be claimed, ever.
        return [Association(track_id, None, 0.0, None, False) for track_id, _ in tracks]
    out: list[Association] = []
    for track_id, bbox in tracks:
        scored = sorted(
            ((seat.seat_id, _score_bbox_seat(bbox, seat)) for seat in layout.seats),
            key=lambda item: item[1],
            reverse=True,
        )
        best_id, best = scored[0]
        runner = scored[1][0] if len(scored) > 1 else None
        runner_score = scored[1][1] if len(scored) > 1 else 0.0
        if best <= 0.0:
            out.append(Association(track_id, None, 0.0, runner, False))
        elif best - runner_score < ambiguity_margin:
            out.append(Association(track_id, None, best, runner, True))
        else:
            out.append(Association(track_id, best_id, best, runner, False))
    return out


def associate_seat_first(
    tracks: list[tuple[str, BBox]],
    layout: SeatLayout,
    min_evidence: float = 0.3,
) -> list[Association]:
    """Variant B: per-seat evidence pools, then attach best track per seat."""
    by_seat: dict[str, list[tuple[str, float]]] = {s.seat_id: [] for s in layout.seats}
    for track_id, bbox in tracks:
        for seat in layout.seats:
            score = _score_bbox_seat(bbox, seat)
            if score >= min_evidence:
                by_seat[seat.seat_id].append((track_id, score))
    claimed: dict[str, str] = {}
    out: list[Association] = []
    for seat_id, candidates in by_seat.items():
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[1], reverse=True)
        track_id, score = candidates[0]
        if track_id in claimed:
            out.append(Association(track_id, None, score, seat_id, True))
            continue
        claimed[track_id] = seat_id
        out.append(Association(track_id, seat_id, score, None, False))
    seen = {a.track_id for a in out}
    for track_id, _ in tracks:
        if track_id not in seen:
            out.append(Association(track_id, None, 0.0, None, False))
    return out
