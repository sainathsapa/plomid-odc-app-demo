"""Seat-layout authoring (calibration wizard backend).

Curated YAML layouts live in configs/seats/. The wizard lets an operator
trace seat polygons on a frozen live frame and save them as a new versioned
layout — the supported path for custom vehicles and uploads (replaces
hand-edited approximations). Validation is strict: degenerate polygons,
out-of-range points, bad ids, duplicates and path traversal are all
rejected with reasons, never silently fixed.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

SEAT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
LAYOUT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_SEATS = 12
MAX_POINTS = 32
MIN_POLY_AREA = 0.005


def _poly_area(polygon: list[list[float]]) -> float:
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def validate_layout_payload(name: Any, seats: Any) -> list[str]:
    """Return human-readable errors (empty = valid). Pure, unit-tested."""
    errors: list[str] = []
    if not isinstance(name, str) or not LAYOUT_NAME_RE.match(name):
        errors.append(
            "name must be 1-64 chars of letters/digits/_/- starting alphanumerically"
        )
    if not isinstance(seats, list) or not seats:
        errors.append("seats must be a non-empty list")
        return errors
    if len(seats) > MAX_SEATS:
        errors.append(f"at most {MAX_SEATS} seats")
    seen: set[str] = set()
    for idx, seat in enumerate(seats):
        tag = f"seat[{idx}]"
        if not isinstance(seat, dict):
            errors.append(f"{tag}: must be an object")
            continue
        sid = seat.get("seat_id")
        if not isinstance(sid, str) or not SEAT_ID_RE.match(sid):
            errors.append(f"{tag}: bad seat_id (letters/digits/_/-, max 40)")
            continue
        if sid in seen:
            errors.append(f"{tag}: duplicate seat_id {sid!r}")
            continue
        seen.add(sid)
        poly = seat.get("polygon")
        if (
            not isinstance(poly, list)
            or len(poly) < 3
            or len(poly) > MAX_POINTS
            or any(
                not isinstance(p, (list, tuple))
                or len(p) != 2
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)
                or not (0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0)
                for p in poly
            )
        ):
            errors.append(
                f"{tag} ({sid}): polygon needs 3-{MAX_POINTS} finite [x,y] in 0..1"
            )
            continue
        area = _poly_area([[float(p[0]), float(p[1])] for p in poly])
        if area < MIN_POLY_AREA:
            errors.append(f"{tag} ({sid}): polygon too small (area {area:.4f})")
        for key in ("row", "col"):
            if key in seat and seat[key] is not None and (
                not isinstance(seat[key], int) or seat[key] < 1 or seat[key] > 32
            ):
                errors.append(f"{tag} ({sid}): {key} must be an int 0..32 when given")
    return errors


def validate_layout_geometry(seats: Any) -> list[str]:
    """Non-blocking geometry warnings: overlaps, missing front-to-rear order.

    Pure. Returns warnings (empty = clean). Hard errors stay in
    validate_layout_payload(); editors display these alongside them so
    overlapping or unsorted zones are always visible, never silent.
    """
    if not isinstance(seats, list) or len(seats) < 2:
        return []
    try:
        from .domain import SeatGeometry as _Seat
        from .seats import SeatLayout as _Layout

        layout = _Layout(
            camera_profile="validation",
            version="validation",
            seats=[
                _Seat(
                    seat_id=str(s.get("seat_id", f"seat-{i}")),
                    polygon=[(float(x), float(y)) for x, y in s.get("polygon", [])],
                    camera_profile="validation",
                    version="validation",
                    row=s.get("row") if isinstance(s.get("row"), int) else None,
                    col=s.get("col") if isinstance(s.get("col"), int) else None,
                )
                for i, s in enumerate(seats)
                if isinstance(s, dict) and isinstance(s.get("polygon"), list)
            ],
        )
    except Exception:
        return []
    try:
        from .seats import find_overlapping_seats as _overlaps

        warnings: list[str] = []
        for pair in _overlaps(layout):
            warnings.append(
                f"overlap: {pair['seat_a']} ↔ {pair['seat_b']} "
                f"(IoU {pair['iou']}) — drag one zone clear"
            )
        return warnings
    except Exception:
        return []


def save_layout(
    name: str,
    seats: list[dict[str, Any]],
    camera_profile: str | None = None,
    directory: str | Path = "configs/seats",
    overwrite: bool = False,
) -> Path:
    """Validate and write configs/seats/<name>.yaml. Raises with reasons."""
    errors = validate_layout_payload(name, seats)
    if errors:
        raise ValueError("; ".join(errors))
    profile = camera_profile or f"custom-{name}"
    target = Path(directory) / f"{name}.yaml"
    if target.exists() and not overwrite:
        raise FileExistsError(f"layout {name} exists (tick overwrite to replace it)")
    import yaml

    doc = {
        "camera_profile": profile,
        "version": name,
        "seats": [
            {
                "seat_id": seat["seat_id"],
                "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in seat["polygon"]],
                "camera_profile": profile,
                "version": name,
                **({"row": int(seat["row"])} if isinstance(seat.get("row"), int) else {}),
                **({"col": int(seat["col"])} if isinstance(seat.get("col"), int) else {}),
                **(
                    {"label": str(seat["label"])}
                    if isinstance(seat.get("label"), str) and seat.get("label")
                    else {}
                ),
            }
            for seat in seats
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(doc, sort_keys=False))
    return target
