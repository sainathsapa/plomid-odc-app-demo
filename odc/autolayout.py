"""Per-video seat geometry proposals (analyse-then-map, candidate-grade).

A newly uploaded or unknown clip has no curated layout. Instead of running
it against generic/stale geometry, the console probes the video itself
BEFORE streaming: sample frames evenly, detect persons, and propose seat
regions from sustained person-presence hotspots. The proposal is labelled
``auto-v1`` (never a curated calibration): generic one-based seat IDs with NO
child-region hints, so occupancy/empty/counts work while child/restraint
stay conservative (size votes + proxy evidence only). Zero sustained
person evidence yields ``None`` — the session runs uncalibrated rather
than inventing seats.

Classical aggregation only (heatmap + peak picking); no training and no
weights beyond the shared detector manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import cv2

from . import contracts
from .domain import FrameMeta, RunMode, SeatGeometry
from .logging import get_logger
from .seats import SeatLayout
from .segmentation import class_aware_nms
from .steering import propose_orientation

log = get_logger("autolayout")

GRID = 12
MAX_PROBE_FRAMES = 32
MIN_PEAK_FRAC = 0.30
MAX_SEATS = 6
PROBE_PERSON_FLOOR = 0.30
MIN_REGION_SIDE = 0.12


@dataclass
class ProbeResult:
    layout: SeatLayout | None
    frames_probed: int
    persons_seen: int
    peaks: int
    reason: str
    orientation: str = "UNKNOWN"
    orientation_confidence: float = 0.0
    orientation_reason: str = (
        "not inferred: the active COCO detector has no validated steering-wheel class"
    )


def _group_peaks_into_rows(
    peaks: list[tuple[int, int]], row_gap_cells: int = 2, max_rows: int = 3
) -> list[list[tuple[int, int]]]:
    """Group hotspot peaks into front-to-rear rows deterministically.

    The vertical gap is measured between normalized heatmap cells. A close
    vertical cluster is one cabin row; a gap of at least ``row_gap_cells`` is
    a new row. This is a proposal heuristic, not a replacement for camera
    calibration. The result is capped at three rows and sorted left-to-right.
    """
    if not peaks:
        return []
    ordered = sorted(peaks, key=lambda cell: (cell[1], cell[0]))
    rows: list[list[tuple[int, int]]] = [[ordered[0]]]
    for peak in ordered[1:]:
        current = rows[-1]
        mean_y = sum(item[1] for item in current) / len(current)
        if peak[1] - mean_y >= row_gap_cells and len(rows) < max_rows:
            rows.append([peak])
        else:
            current.append(peak)
    for row in rows:
        row.sort(key=lambda cell: cell[0])
    return rows


def _capacity_constrained_rows(
    peaks: list[tuple[int, int]],
    sizes: dict[tuple[int, int], list[tuple[float, float]]],
) -> list[list[tuple[int, int]]]:
    """Keep candidate rows physically plausible without forcing two seats.

    A close-up camera can place front and rear occupants at nearly identical
    image y coordinates. In that case vertical clustering alone is
    insufficient. The smaller apparent boxes are treated as farther-row
    candidates, but the result remains an ``auto-proposal`` requiring review.
    A row containing one peak is preserved as a one-seat row, which supports
    buses, vans, limousines, and single-column camera views.
    """
    rows = _group_peaks_into_rows(peaks)
    # Enforce physical row capacities even when clustering has already
    # produced multiple vertical bands. In particular, a front row can never
    # contain three seats in the standard passenger-vehicle proposal. Excess
    # peaks are carried into the next row instead of being rendered as an
    # impossible front bench.
    capacities = (2, 3, 2)
    normalized: list[list[tuple[int, int]]] = []
    carry: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        capacity = capacities[min(index, len(capacities) - 1)]
        current = sorted(carry + row, key=lambda peak: peak[0])
        normalized.append(current[:capacity])
        carry = current[capacity:]
    if carry:
        normalized.append(sorted(carry, key=lambda peak: peak[0]))
    rows = normalized
    if len(rows) != 1 or len(peaks) <= 2:
        return rows

    def area(peak: tuple[int, int]) -> float:
        values = sizes.get(peak, [])
        return median([w * h for w, h in values]) if values else 0.0

    by_size = sorted(peaks, key=lambda peak: (area(peak), peak[0]), reverse=True)
    rows = []
    remaining = list(by_size)
    for capacity in capacities:
        if not remaining:
            break
        current = sorted(remaining[:capacity], key=lambda peak: peak[0])
        rows.append(current)
        remaining = remaining[capacity:]
    return rows


def _pick_peaks(
    hits: list[list[float]], min_frac: float = MIN_PEAK_FRAC, max_seats: int = MAX_SEATS
) -> list[tuple[int, int]]:
    """Greedy hotspot cells: highest first, Chebyshev separation >= 2."""
    peak = max((v for row in hits for v in row), default=0.0)
    if peak <= 0.0:
        return []
    cells = sorted(
        ((v, x, y) for y, row in enumerate(hits) for x, v in enumerate(row)),
        reverse=True,
    )
    taken: list[tuple[int, int]] = []
    for value, x, y in cells:
        if value < peak * min_frac or len(taken) >= max_seats:
            break
        if all(max(abs(x - tx), abs(y - ty)) >= 2 for tx, ty in taken):
            taken.append((x, y))
    return taken


def _polygon_for_peak(
    gx: int, gy: int, sizes: dict[tuple[int, int], list[tuple[float, float]]]
) -> list[tuple[float, float]]:
    """Centered rect from median contributor box size (fallback 0.28x0.55)."""
    near: list[tuple[float, float]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            near.extend(sizes.get((gx + dx, gy + dy), []))
    if near:
        half_w = max(median([w for w, _ in near]) / 2.0, MIN_REGION_SIDE / 2.0)
        half_h = max(median([h for _, h in near]) / 2.0, MIN_REGION_SIDE / 2.0)
    else:
        half_w, half_h = 0.14, 0.275
    cx, cy = (gx + 0.5) / GRID, (gy + 0.5) / GRID
    x0, x1 = max(0.0, cx - half_w), min(1.0, cx + half_w)
    y0, y1 = max(0.0, cy - half_h), min(1.0, cy + half_h)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _polygon_for_peak_partitioned(
    peak: tuple[int, int], all_peaks: list[tuple[int, int]], sizes: dict[tuple[int, int], list[tuple[float, float]]]
) -> list[tuple[float, float]]:
    """Create a candidate zone with globally non-overlapping x partitions."""
    base = _polygon_for_peak(peak[0], peak[1], sizes)
    centers = sorted((cell[0] + 0.5) / GRID for cell in all_peaks)
    cx = (peak[0] + 0.5) / GRID
    index = min(range(len(centers)), key=lambda i: abs(centers[i] - cx))
    x0 = 0.0 if index == 0 else (centers[index - 1] + cx) / 2.0
    x1 = 1.0 if index == len(centers) - 1 else (cx + centers[index + 1]) / 2.0
    y0, y1 = base[0][1], base[2][1]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def propose_seat_layout(
    path: str,
    model_id: str = "rtdetrv2-s-coco",
    max_probe_frames: int = MAX_PROBE_FRAMES,
    person_floor: float = PROBE_PERSON_FLOOR,
) -> ProbeResult:
    """Analyse a video file, then propose seat regions. Bounded and safe:
    never raises — every failure mode returns (None, reason) instead."""
    from .runtime import OnnxDetectorRuntime, load_manifest

    src = Path(path)
    if not src.exists():
        return ProbeResult(None, 0, 0, 0, "missing file")
    try:
        manifest = load_manifest(model_id)
    except Exception as e:
        return ProbeResult(None, 0, 0, 0, f"model unavailable: {type(e).__name__}")
    cap = cv2.VideoCapture(str(src))
    try:
        if not cap.isOpened():
            return ProbeResult(None, 0, 0, 0, "unreadable video")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        if width < 16 or height < 16:
            return ProbeResult(None, 0, 0, 0, "no geometry")
        stride = max(1, total // max_probe_frames) if total > 0 else 1
        runtime = OnnxDetectorRuntime(threshold=0.20)
        try:
            runtime.load(manifest)
        except Exception as e:
            return ProbeResult(None, 0, 0, 0, f"model unavailable: {type(e).__name__}")
        try:
            person_heat = [[0.0] * GRID for _ in range(GRID)]
            furniture_heat = [[0.0] * GRID for _ in range(GRID)]
            sizes: dict[tuple[int, int], list[tuple[float, float]]] = {}

            def _accumulate(
                heat: list[list[float]], bbox: Any, score: float, weight: float
            ) -> None:
                b = bbox
                gx = min(GRID - 1, int(((b.x_min + b.x_max) / 2.0) * GRID))
                gy = min(GRID - 1, int(((b.y_min + b.y_max) / 2.0) * GRID))
                heat[gy][gx] += score * weight
                bw, bh = b.x_max - b.x_min, b.y_max - b.y_min
                sizes.setdefault((gx, gy), []).append((bw, bh))

            probed = 0
            persons = 0
            idx = 0
            while probed < max_probe_frames:
                ok, pixels = cap.read()
                if not ok or pixels is None:
                    break
                if idx % stride != 0:
                    idx += 1
                    continue
                idx += 1
                frame = contracts.Frame(
                    meta=FrameMeta(
                        source_id="probe",
                        seq=idx,
                        captured_at=datetime.now(UTC),
                        width=int(pixels.shape[1]),
                        height=int(pixels.shape[0]),
                        mode=RunMode.TEST,
                    ),
                    pixels=pixels,
                )
                try:
                    result = runtime.infer(frame)
                except Exception as e:
                    log.warning("probe infer skipped: %s", type(e).__name__)
                    continue
                dets = class_aware_nms(
                    [
                        d
                        for d in result.detections
                        if d.class_label == "person" and d.score >= person_floor
                    ],
                    0.5,
                )
                probed += 1
                for det in dets:
                    persons += 1
                    _accumulate(person_heat, det.bbox, det.score, 1.0)
                # Seat furniture itself maps EMPTY seats: chairs/couches and
                # seat-sized shell labels count at reduced weight, so visible
                # seats still propose geometry instead of vanishing.
                furn = class_aware_nms(
                    [
                        d
                        for d in result.detections
                        if d.score >= person_floor
                        and (
                            d.class_label in ("chair", "couch")
                            or (
                                d.class_label in ("suitcase", "handbag", "backpack")
                                and (d.bbox.x_max - d.bbox.x_min)
                                * (d.bbox.y_max - d.bbox.y_min)
                                >= 0.05
                            )
                        )
                    ],
                    0.5,
                )
                for det in furn:
                    _accumulate(furniture_heat, det.bbox, det.score, 0.6)
        finally:
            runtime.close()
    finally:
        cap.release()
    orientation = propose_orientation(src, max_frames=min(16, max_probe_frames))
    # Person evidence owns occupied-cabin geometry. Furniture is only a
    # fallback for a cabin with no visible people; mixing both produced extra
    # chair peaks in the uploaded three-person regression clip.
    peaks = _pick_peaks(person_heat) if persons else _pick_peaks(furniture_heat)
    if not peaks:
        return ProbeResult(
            None, probed, persons, 0, "no sustained person evidence",
            orientation=orientation.orientation,
            orientation_confidence=orientation.confidence,
            orientation_reason=orientation.reason,
        )
    profile = f"auto-{src.stem[:24]}"
    seats: list[SeatGeometry] = []
    seat_number = 1
    for row_number, row in enumerate(_capacity_constrained_rows(peaks, sizes), start=1):
        for col_number, (gx, gy) in enumerate(row, start=1):
            seats.append(
                SeatGeometry(
                    seat_id=f"seat_{seat_number}",
                    polygon=_polygon_for_peak_partitioned((gx, gy), peaks, sizes),
                    camera_profile=profile,
                    version="auto-v1",
                    row=row_number,
                    col=col_number,
                    display_number=seat_number,
                )
            )
            seat_number += 1
    layout = SeatLayout(
        camera_profile=profile,
        version="auto-v1",
        seats=seats,
        layout_type="auto-proposal",
        orientation=orientation.orientation,
        orientation_reason=orientation.reason,
        display_name="Automatic candidate seat map",
    )
    return ProbeResult(
        layout,
        probed,
        persons,
        len(peaks),
        f"{len(peaks)} hotspot(s) from {probed} frames; rows inferred from vertical gaps",
        orientation=orientation.orientation,
        orientation_confidence=orientation.confidence,
        orientation_reason=orientation.reason,
    )
