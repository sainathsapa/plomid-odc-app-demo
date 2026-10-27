"""ODC engineering console (Phase 8) — Python-only FastAPI app.

Server-rendered Jinja + HTMX partials + Alpine.js interactivity + vendored
ECharts. Every displayed state traces to a PLOMID SQL row (via ?session=)
or to the transient hub (labelled LIVE). No fake detections, anywhere.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..attributes import seat_display_color
from ..config import AppConfig
from ..contracts import Repository
from ..domain import HealthStatus
from ..dossier import build_dossier, render_markdown
from ..logging import get_logger
from ..seats import SeatLayout, seat_display_name
from ..segmentation import PER_CLASS_THRESHOLDS
from ..service import Service, ServiceHub

BASE = Path(__file__).resolve().parent
log = get_logger("console")
templates = Jinja2Templates(directory=str(BASE / "templates"))


@lru_cache(maxsize=12)
def _read_first_video_frame(path: str, modified_ns: int, file_size: int) -> bytes | None:
    """Decode and cache the real first frame of one local replay video."""
    del modified_ns, file_size  # Metadata participates in the bounded cache key.
    import cv2

    capture = cv2.VideoCapture(path)
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        encoded, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return payload.tobytes() if encoded else None
    finally:
        capture.release()


STATE_COLORS = {
    # Task legend: Green empty, Red occupied, Yellow uncertain (LIKELY),
    # Blue child, Purple restraint, Orange belt issue, Gray occluded/unknown.
    "OCCUPIED": "#ef4444",
    "LIKELY_OCCUPIED": "#eab308",
    "EMPTY_CANDIDATE": "#22c55e",
    "UNKNOWN": "#9ca3af",
    "STALE": "#f59e0b",
    "CHILD": "#3b82f6",
    "RESTRAINT": "#a855f7",
    "BELT_ISSUE": "#f97316",
}
STATE_LABELS = {
    "OCCUPIED": "Occupied",
    "LIKELY_OCCUPIED": "Likely occupied",
    "EMPTY_CANDIDATE": "Empty candidate",
    "UNKNOWN": "Unknown",
    "STALE": "Stale",
}

LEGEND = [
    ("#22c55e", "Empty"),
    ("#ef4444", "Occupied"),
    ("#eab308", "Uncertain"),
    ("#3b82f6", "Child detected"),
    ("#a855f7", "Child safety seat"),
    ("#f97316", "Belt issue"),
    ("#9ca3af", "Occluded / unknown"),
]


def _presence_of(state: Any) -> str:
    if state is None:
        return "UNKNOWN"
    if isinstance(state, dict):
        return str(state.get("presence", "UNKNOWN"))
    return getattr(state.presence, "value", "UNKNOWN")


def _child_of(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("child", "Unknown"))
    return "Unknown"


def _restraint_of(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("restraint", "Unknown"))
    label = getattr(state, "object_label", None)
    if label in ("chair", "couch"):
        return "Yes"
    return "Unknown"


def _belt_issue_of(state: Any) -> bool:
    if isinstance(state, dict):
        return state.get("belt_issue") is True
    return False


def _mf_of(state: Any) -> str:
    """Short M/F badge for overlays and maps ("M" / "F" / "").

    Reads the transient gender-appearance verdict on live rows; history
    SeatStates never carry one and yield "" (no badge, never a guess).
    """
    if isinstance(state, dict):
        gender = state.get("gender", "")
        if gender == "APPARENT_MAN":
            return "M"
        if gender == "APPARENT_WOMAN":
            return "F"
    return ""


def _belt_badge_of(state: Any) -> str:
    """Short belt badge for overlays ("" when nothing to show)."""
    if isinstance(state, dict):
        if state.get("belt_issue") is True:
            return "belt?"
        if state.get("belt") == "Detected":
            return "strap candidate"
    return ""


def _visibility_of(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("visibility", ""))
    return getattr(getattr(state, "visibility", None), "value", "")


def _seat_base_color(presence: str, child: str, occluded: bool = False) -> str:
    if occluded:
        return STATE_COLORS["UNKNOWN"]
    if presence in ("OCCUPIED", "LIKELY_OCCUPIED"):
        if child == "Yes":
            return STATE_COLORS["CHILD"]
        if presence == "LIKELY_OCCUPIED":
            return STATE_COLORS["LIKELY_OCCUPIED"]
        return STATE_COLORS["OCCUPIED"]
    if presence == "EMPTY_CANDIDATE":
        return STATE_COLORS["EMPTY_CANDIDATE"]
    if presence == "STALE":
        return STATE_COLORS["STALE"]
    return STATE_COLORS["UNKNOWN"]


def _ring_cx_cy(seat: Any) -> tuple[str, str]:
    xs = [x * 1000 for x, _ in seat.polygon]
    ys = [y * 1000 for _, y in seat.polygon]
    return f"{sum(xs) / len(xs):.0f}", f"{sum(ys) / len(ys):.0f}"


def _event_frame_seq(event: Any) -> int | None:
    """Best-effort frame number for timeline→video jumps.

    Alert events carry ``frame_seq=N`` in evidence_summary (written by
    Service._record_alert on the evidence clock). Presence events encode
    the session as ``{session}-{seat}-{n}`` where n is a per-seat counter,
    not a frame — those return None so the UI falls back to timestamp
    ordering instead of a wrong frame.
    """
    try:
        summary = getattr(event, "evidence_summary", "") or ""
        if isinstance(summary, str) and "frame_seq=" in summary:
            head = summary.split("frame_seq=", 1)[1]
            digits = ""
            for ch in head:
                if ch.isdigit() or (ch == "-" and not digits):
                    digits += ch
                else:
                    break
            if digits not in ("", "-"):
                value = int(digits)
                return value if value >= 0 else None
    except Exception:
        pass
    return None


def _seat_style(state: Any) -> tuple[str, str, bool, bool, bool]:
    """Shared display semantics: (fill, label-extra, restraint, belt_issue, occluded)."""
    presence = _presence_of(state)
    child = _child_of(state)
    restraint = _restraint_of(state)
    restraint_candidate = (
        bool(state.get("restraint_candidate") or state.get("child_seat_candidate"))
        if isinstance(state, dict)
        else False
    )
    belt_issue = _belt_issue_of(state)
    occluded = _visibility_of(state) in ("OCCLUDED", "CAMERA_UNHEALTHY")
    return (
        _seat_base_color(presence, child, occluded),
        presence,
        restraint == "Yes" or restraint_candidate,
        belt_issue,
        occluded,
    )


def _wrap_seat_id(seat_id: str, width: int = 10) -> list[str]:
    """Wrap a seat id onto short lines for map labels (never truncated).

    Splits on "-" / "_" first so "child-seat-right" becomes
    ["child-", "seat-", "right"]; a chunk longer than width hard-splits.
    Capped at 3 lines — ids are short by convention, wrapping is display
    only (aria-labels and titles keep the full id).
    """
    raw = str(seat_id)
    if len(raw) <= width:
        return [raw]
    parts = raw.replace("_", "-").split("-")
    lines: list[str] = []
    cur = ""
    for i, part in enumerate(parts):
        chunk = part + ("-" if i < len(parts) - 1 else "")
        while len(chunk) > width:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(chunk[:width])
            chunk = chunk[width:]
        if len(cur) + len(chunk) <= width or not cur:
            cur += chunk
        else:
            lines.append(cur)
            cur = chunk
    if cur:
        lines.append(cur)
    return [ln for ln in lines if ln][:3] or [raw]


def _vehicle_rows(layout: SeatLayout) -> list[list[tuple[str, float, float]]]:
    """Cluster seat centroids front-to-back rows, left-to-right in row.

    Delegates to seats.order_seats_front_to_rear so the schematic, the
    timeline and the editors share one ordering (stored row/col hints win;
    otherwise an adaptive band from median seat height). Falls back to the
    legacy fixed band only if the shared helper is unavailable.
    """
    try:
        from ..seats import order_seats_front_to_rear, seat_centroid

        by_id = {s.seat_id: s for s in layout.seats}
        rows: list[list[tuple[str, float, float]]] = []
        for group in order_seats_front_to_rear(layout):
            row: list[tuple[str, float, float]] = []
            for sid in group:
                seat = by_id.get(sid)
                if seat is None:
                    continue
                cx, cy = seat_centroid(seat)
                row.append((sid, cx, cy))
            row.sort(key=lambda t: t[1])
            rows.append(row)
        return rows
    except Exception:
        cents = []
        for seat in layout.seats:
            xs = [p[0] for p in seat.polygon]
            ys = [p[1] for p in seat.polygon]
            cents.append((seat.seat_id, sum(xs) / len(xs), sum(ys) / len(ys)))
        cents.sort(key=lambda t: t[2])
        rows = []
        for item in cents:
            if rows:
                mean_y = sum(t[2] for t in rows[-1]) / len(rows[-1])
                if abs(item[2] - mean_y) < 0.22:
                    rows[-1].append(item)
                    continue
            rows.append([item])
        for row in rows:
            row.sort(key=lambda t: t[1])
        return rows


def vehicle_svg(layout: SeatLayout, states: dict[str, Any]) -> str:
    """Cabin-schematic seat map: car body + windshield + seat glyphs.

    Glyph positions derive from real polygon centroids (row-clustered), so
    any seat count/arrangement renders as a vehicle instead of floating
    polygons. Colors, rings and labels follow the same contract as the
    polygon view; hover/tap still opens seat evidence.
    """
    W, H = 600, 720
    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="group" aria-label="Vehicle seat map">',
        '<rect x="70" y="30" width="460" height="660" rx="100" fill="none" '
        'style="stroke:var(--map-frame)" stroke-width="3"/>',
        '<polygon points="150,95 450,95 405,175 195,175" fill="rgba(127,127,127,0.10)" '
        'style="stroke:var(--map-frame)" stroke-width="2"/>',
        '<text x="300" y="65" style="fill:var(--t-d3)" font-size="20" text-anchor="middle" '
        'font-family="monospace" letter-spacing="4">FRONT</text>',
        '<text x="300" y="675" style="fill:var(--map-frame)" font-size="18" text-anchor="middle" '
        'font-family="monospace" letter-spacing="4">REAR</text>',
    ]
    if not layout.seats:
        parts.append(
            '<text x="300" y="380" style="fill:var(--t-d3)" font-size="22" text-anchor="middle" '
            'font-family="monospace">no calibrated geometry</text>'
        )
        return "".join(parts) + "</svg>"
    rows = _vehicle_rows(layout)
    ncols = max(len(row) for row in rows)
    # Gap-aware glyph width: n glyphs + (n-1) 36-unit gaps must fit the
    # 400-unit car interior, otherwise the outer glyphs (e.g. a 3-wide
    # row's "child-seat-right") poke outside the car outline and their
    # labels collide with neighbours.
    gw = min(150.0, (400.0 - (ncols - 1) * 36.0) / max(1, ncols))
    gw = max(56.0, gw)
    gh = gw * 1.3
    top, bottom = 250.0, 560.0
    for ri, row in enumerate(rows):
        cy = top if len(rows) == 1 else top + ri * (bottom - top) / max(1, len(rows) - 1)
        if len(rows) > 1:
            row_label_y = max(212.0, cy - (gh * 0.82) - 22.0)
            parts.append(
                f'<text class="rowlabel" x="300" y="{row_label_y:.0f}" '
                f'text-anchor="middle" font-size="13" '
                f'style="fill:var(--t-d3)" font-family="monospace" '
                f'letter-spacing="2">ROW {ri + 1}</text>'
            )
        for ji, (seat_id, _cx, _cy) in enumerate(row):
            cx = 300.0 + (ji - (len(row) - 1) / 2.0) * (gw + 36.0)
            state = states.get(seat_id)
            color, presence, restraint, belt_issue, _occ = _seat_style(state)
            restraint_candidate = (
                isinstance(state, dict)
                and bool(state.get("restraint_candidate") or state.get("child_seat_candidate"))
                and _restraint_of(state) != "Yes"
            )
            label = STATE_LABELS.get(presence, presence)
            extras = []
            if _child_of(state) == "Yes":
                extras.append("child")
            if restraint:
                extras.append("restraint-candidate" if restraint_candidate else "restraint")
            if belt_issue:
                extras.append("belt-issue")
            mf = _mf_of(state)
            if mf:
                extras.append(mf)
            extra_txt = (" +" + "+".join(extras)) if extras else ""
            seat = next((s for s in layout.seats if s.seat_id == seat_id), None)
            display_name = seat_display_name(seat) if seat is not None else seat_id
            x, y = cx - gw / 2.0, cy - gh / 2.0
            parts.append(
                f'<g class="seat" data-seat="{seat_id}" data-presence="{escape(presence)}" '
                f'data-child="{escape(str(_child_of(state) or ""))}" '
                f'tabindex="0" role="button" '
                f'aria-label="{display_name}: {label}{extra_txt}" '
                f"x-data @click=\"$dispatch('seat-pick', {{'seat': '{seat_id}'}})\">"
                f'<rect x="{x:.0f}" y="{y:.0f}" width="{gw:.0f}" height="{gh * 0.52:.0f}" '
                f'rx="16" fill="{color}" fill-opacity="0.55" stroke="#111827" stroke-width="5"/>'
                f'<rect x="{x:.0f}" y="{y + gh * 0.48:.0f}" width="{gw:.0f}" '
                f'height="{gh * 0.52:.0f}" rx="16" fill="{color}" fill-opacity="0.35" '
                f'stroke="#111827" stroke-width="5"/>'
                f'<rect x="{cx - gw * 0.2:.0f}" y="{y - 26:.0f}" width="{gw * 0.4:.0f}" '
                f'height="26" rx="8" fill="{color}" fill-opacity="0.55" '
                f'stroke="#111827" stroke-width="4"/>'
            )
            if restraint:
                dash = ' stroke-dasharray="12 8"' if restraint_candidate else ""
                parts.append(
                    f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="34" fill="none" '
                    f'stroke="{STATE_COLORS["RESTRAINT"]}" stroke-width="9"{dash}/>'
                )
            if belt_issue:
                parts.append(
                    f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="48" fill="none" '
                    f'stroke="{STATE_COLORS["BELT_ISSUE"]}" stroke-width="8"/>'
                )
            # Keep the cabin map legible at small widths: a compact seat name
            # sits inside its own glyph instead of a long, overlapping badge.
            label_width = max(6, int((gw - 18) / 9))
            label_lines = _wrap_seat_id(display_name.replace(" ", "-"), label_width)[:3]
            if mf and len(label_lines) == 1:
                label_lines.append(f"· {mf}")
            longest = max((len(line) for line in label_lines), default=1)
            font = max(9, min(15, int((gw - 18) / max(1, longest * 0.68))))
            first_y = cy - ((len(label_lines) - 1) * (font + 2)) / 2 + font * 0.35
            label_spans = "".join(
                f'<tspan x="{cx:.0f}" dy="{0 if index == 0 else font + 2}">'
                f"{escape(line.rstrip('-'))}</tspan>"
                for index, line in enumerate(label_lines)
            )
            parts.append(
                f'<text class="maplabel" x="{cx:.0f}" y="{first_y:.0f}" '
                f'font-size="{font}" text-anchor="middle" '
                f'style="paint-order:stroke;stroke:var(--map-label-halo);stroke-width:4">'
                f"{label_spans}</text>"
            )
            parts.append(f"<title>{display_name}: {label}{extra_txt}</title></g>")
    return "".join(parts) + "</svg>"


def _place_zone_label(
    text: str, cx: float, cy: float, placed: list[tuple[float, float, float, float]]
) -> tuple[float, float]:
    """Nudge a zone label down until its box clears earlier labels.

    Auto proposals often overlap heavily (seat-2 covers half the frame), so
    centroid labels would print on top of each other ("seat-1seat-2 · M…").
    Greedy downward shift, bounded, deterministic in seat order. Appends the
    final box to placed and returns the (x, y) text position.
    """
    width = max(60.0, len(text) * 20.0)  # ~0.6em monospace advance at 34px
    lx, ly = cx, cy
    for _ in range(8):
        box = (lx - width / 2.0, ly - 22.0, lx + width / 2.0, ly + 22.0)
        if not any(
            box[0] < p[2] and p[0] < box[2] and box[1] < p[3] and p[1] < box[3] for p in placed
        ):
            break
        ly += 46.0
    placed.append((lx - width / 2.0, ly - 22.0, lx + width / 2.0, ly + 22.0))
    return lx, ly


def seat_svg(layout: SeatLayout, states: dict[str, Any], view: str = "poly") -> str:
    """Seat map in two renderings of one contract (see vehicle_svg).

    view="poly": true calibrated polygons. view="car": cabin schematic
    with seat glyphs positioned from real centroids. Colors, rings and
    labels are identical; click/hover/keyboard via data-seat attrs.
    Anything else falls back to the schematic.
    """
    if view != "poly":
        return vehicle_svg(layout, states)
    # Zones view: true calibrated polygons. preserveAspectRatio="none" so the
    # zones align exactly with a frame image placed behind them (both fill
    # their box 1:1 in normalized coordinates).
    parts = [
        '<svg viewBox="0 0 1000 1000" preserveAspectRatio="none" '
        'role="group" aria-label="Calibrated seat zones">'
    ]
    _zone_label_boxes: list[tuple[float, float, float, float]] = []
    for seat in layout.seats:
        points = " ".join(f"{x * 1000:.0f},{y * 1000:.0f}" for x, y in seat.polygon)
        state = states.get(seat.seat_id)
        color, presence, restraint, belt_issue, _occ = _seat_style(state)
        restraint_candidate = (
            isinstance(state, dict)
            and bool(state.get("restraint_candidate") or state.get("child_seat_candidate"))
            and _restraint_of(state) != "Yes"
        )
        label = STATE_LABELS.get(presence, presence)
        extras = []
        if _child_of(state) == "Yes":
            extras.append("child")
        if restraint:
            extras.append("restraint-candidate" if restraint_candidate else "restraint")
        if belt_issue:
            extras.append("belt-issue")
        mf = _mf_of(state)
        if mf:
            extras.append(mf)
        extra_txt = (" +" + "+".join(extras)) if extras else ""
        cx, cy = _ring_cx_cy(seat)
        rings = ""
        if restraint:
            # Inner purple ring preserves the occupancy base color.
            dash = ' stroke-dasharray="12 8"' if restraint_candidate else ""
            rings += (
                f'<circle cx="{cx}" cy="{cy}" r="60" fill="none" '
                f'stroke="{STATE_COLORS["RESTRAINT"]}" stroke-width="14"{dash}/>'
            )
        if belt_issue:
            rings += (
                f'<circle cx="{cx}" cy="{cy}" r="88" fill="none" '
                f'stroke="{STATE_COLORS["BELT_ISSUE"]}" stroke-width="12"/>'
            )
        zone_text = f"{seat_display_name(seat)}{' · ' + mf if mf else ''}"
        # Adaptive label: long ids shrink to their polygon's width and the
        # block clamps inside the 1000-unit viewBox, so "child-seat-right"
        # never prints over a neighbour or off the map.
        xs = [float(px) for px, _ in seat.polygon]
        bbox_w = (max(xs) - min(xs)) * 1000.0 if xs else 0.0
        longest_zone = max(len(zone_text), 1)
        fit_font = int(bbox_w / (longest_zone * 0.62)) if bbox_w > 0 else 34
        zone_font = min(34, max(20, fit_font))
        est_half = longest_zone * zone_font * 0.36
        # De-collision runs on the estimated box; clamp the resolved centre
        # so the nudge cannot push the text off the map either.
        _lx, _ly = _place_zone_label(zone_text, float(cx), float(cy), _zone_label_boxes)
        _zone_label_boxes[-1] = (
            _lx - est_half,
            _ly - zone_font * 0.7,
            _lx + est_half,
            _ly + zone_font * 0.7,
        )
        lx = min(1000 - est_half - 12, max(est_half + 12, _lx))
        ly = min(984, max(30, _ly))
        zone_stroke = max(3, zone_font // 5)
        parts.append(
            f'<g class="seat" data-seat="{seat.seat_id}" data-presence="{escape(presence)}" '
            f'data-child="{escape(str(_child_of(state) or ""))}" '
            f'tabindex="0" role="button" '
            f'aria-label="{seat_display_name(seat)}: {label}{extra_txt}" '
            f"x-data @click=\"$dispatch('seat-pick', {{'seat': '{seat.seat_id}'}})\">"
            f'<polygon points="{points}" fill="{color}" fill-opacity="0.55" '
            f'stroke="#111827" stroke-width="6"/>'
            f"{rings}"
            f'<text x="{lx:.0f}" y="{ly:.0f}" font-size="{zone_font}" '
            f'text-anchor="middle" font-family="monospace" fill="#fff" '
            f'style="paint-order:stroke" stroke="#08090b" stroke-width="{zone_stroke}">'
            f"{zone_text}</text>"
            f"<title>{seat_display_name(seat)}: {label}{extra_txt}</title></g>"
        )
    return "".join(parts) + "</svg>"


def _badge(hub: ServiceHub, session_param: str | None) -> str:
    if session_param:
        return "REPLAY"
    snap = hub.snapshot
    if not snap.running and not snap.session_id:
        return "DISCONNECTED"
    if not snap.db_healthy or not snap.model_healthy:
        return "DEGRADED"
    if snap.camera_status not in (HealthStatus.HEALTHY, HealthStatus.UNKNOWN):
        return "DEGRADED"
    if not snap.running:
        return "STOPPED"
    return snap.mode.value


def _live_seats(hub: ServiceHub) -> dict[str, Any]:
    with hub.lock:
        if hub.seat_rows:
            return {r["seat_id"]: r for r in hub.seat_rows}
        return {s.seat_id: s for s in hub.seats}


def _with_display_names(layout: SeatLayout, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach labels at the UI boundary; persisted IDs are never rewritten."""
    names = {s.seat_id: seat_display_name(s) for s in layout.seats}
    return [
        dict(row, display_name=names.get(str(row.get("seat_id")), row.get("seat_id")))
        for row in rows
    ]


def _overlay_persisted_observation(
    rows: list[dict[str, Any]], observation: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Merge the latest durable frame attributes into session history rows."""
    if not observation:
        return rows
    payload = observation.get("payload")
    if not isinstance(payload, dict):
        return rows
    saved_rows = payload.get("seat_rows")
    if not isinstance(saved_rows, list):
        return rows
    by_seat = {
        str(row["seat_id"]): row
        for row in saved_rows
        if isinstance(row, dict) and row.get("seat_id") is not None
    }
    version = str(payload.get("api_schema_version", "odc.seat-observation.v1"))
    return [
        {
            **row,
            **by_seat.get(str(row.get("seat_id")), {}),
            "api_schema_version": version,
            "observation_frame_seq": observation.get("frame_seq"),
            "observation_captured_at": observation.get("captured_at"),
        }
        for row in rows
    ]


def _live_snapshot_dict(hub: ServiceHub) -> dict[str, Any]:
    snap = hub.snapshot
    return {
        "session_id": snap.session_id,
        "running": snap.running,
        "mode": snap.mode.value,
        "frames_captured": snap.frames_captured,
        "frames_processed": snap.frames_processed,
        "frames_dropped": snap.frames_dropped,
        "last_latency_ms": snap.last_latency_ms,
        "db_healthy": snap.db_healthy,
        "model_healthy": snap.model_healthy,
        "model_id": snap.model_id,
        "provider": snap.provider,
        "camera_status": snap.camera_status.value,
        "calibrated": snap.calibrated,
        "frame_seq": snap.frame_seq,
        "suppressed_hallucinations": snap.suppressed_hallucinations,
        "duplicates_removed": snap.duplicates_removed,
        "threshold": snap.threshold,
        "layout_source": snap.layout_source,
        "pose_on": snap.pose_on,
        "pose_healthy": snap.pose_healthy,
        "dms_on": snap.dms_on,
        "dms_healthy": snap.dms_healthy,
        "driver_seat_id": snap.driver_seat_id,
        "mask_mode": snap.mask_mode,
        "specialist_enabled": snap.specialist_enabled,
        "specialist_healthy": snap.specialist_healthy,
        "specialist_frame_seq": snap.specialist_frame_seq,
        "specialist_error": snap.specialist_error,
        "updated_at": snap.updated_at.isoformat() if snap.updated_at else None,
    }


def create_app(
    cfg: AppConfig,
    repo: Repository,
    layout: SeatLayout,
    hub: ServiceHub | None = None,
    autostart: bool = False,
) -> FastAPI:
    live_hub = hub or ServiceHub()
    frame_thumbnail_cache: dict[tuple[int, str], bytes] = {}
    # The console is intentionally idle until the operator chooses a source,
    # but database health is independent of whether perception is running.
    # Probe once at boot so the idle screen does not falsely display DB DOWN
    # when start/stop has simply not been requested yet. The service performs
    # its own probe again when a session starts and reconnects after failures.
    try:
        db_ready = bool(repo.health_check())
    except Exception:
        db_ready = False
    if db_ready:
        with live_hub.lock:
            live_hub.snapshot = replace(live_hub.snapshot, db_healthy=True)
    service = Service(cfg, repo, layout, hub=live_hub) if autostart else None

    def preload_models() -> None:
        """Warm configured model assets once before the console accepts runs."""
        try:
            from ..runtime import OnnxDetectorRuntime, load_manifest

            decode_threshold = min(
                cfg.detection.min_decode_threshold,
                cfg.detection.threshold,
                min(PER_CLASS_THRESHOLDS.values()),
            )
            detector = OnnxDetectorRuntime(threshold=decode_threshold)
            detector.load(load_manifest(cfg.detection.model_id))
            detector.close()
            log.info("startup model ready: %s", cfg.detection.model_id)
        except Exception as exc:
            log.warning("startup detector unavailable: %s: %s", type(exc).__name__, exc)
        if cfg.age.enabled:
            try:
                from ..age import MiVOLOV2AgeEstimator

                age_model = MiVOLOV2AgeEstimator(
                    cfg.age.model_path,
                    device=cfg.age.device,
                    infant_max_years=cfg.age.infant_max_years,
                    child_max_years=cfg.age.child_max_years,
                    uncertainty_margin_years=cfg.age.uncertainty_margin_years,
                    gender_min_score=cfg.age.gender_min_score,
                    history_max=cfg.age.history_max,
                    track_expiry_frames=cfg.age.track_expiry_frames,
                )
                age_model.load()
                age_model.close()
                log.info("startup model ready: MiVOLO V2")
            except Exception as exc:
                log.warning("startup age model unavailable: %s: %s", type(exc).__name__, exc)
        if cfg.age.enabled or cfg.dms.enabled:
            try:
                from ..dms import FaceMesh

                face_model = FaceMesh(
                    cfg.dms.model_path,
                    num_faces=max(cfg.dms.num_faces, 4 if cfg.age.enabled else 1),
                )
                face_model.load()
                log.info("startup model ready: face landmarks")
            except Exception as exc:
                log.warning("startup face model unavailable: %s: %s", type(exc).__name__, exc)
        if cfg.pose.enabled:
            try:
                from ..pose import MediaPipePose

                MediaPipePose(cfg.pose.model_path)
                log.info("startup model ready: pose")
            except Exception as exc:
                log.warning("startup pose model unavailable: %s: %s", type(exc).__name__, exc)
        if cfg.detection.child_specialist_enabled:
            try:
                from ..vlm import load_grounding_dino

                load_grounding_dino(
                    cfg.detection.child_specialist_config,
                    cfg.detection.child_specialist_checkpoint,
                    device="cpu",
                )
                log.info("startup model ready: child specialist")
            except Exception as exc:
                log.warning("startup child specialist unavailable: %s: %s", type(exc).__name__, exc)
        if cfg.detection.mask_mode == "precise":
            try:
                from ..sam import SamPromptedMasks

                SamPromptedMasks().load()
                log.info("startup model ready: SAM")
            except Exception as exc:
                log.warning("startup SAM unavailable: %s: %s", type(exc).__name__, exc)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        import anyio

        await anyio.to_thread.run_sync(preload_models)
        if service is not None:
            service.start()
        yield
        if service is not None:
            service.stop()

    app = FastAPI(title="PLOMID ODC console", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.repo = repo
    app.state.layout = layout
    app.state.hub = live_hub
    app.state.service = service
    import threading as _threading

    app.state.service_lock = _threading.Lock()
    app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

    def ctx(request: Request, session: str | None = None, **extra: Any) -> dict[str, Any]:
        badge = _badge(live_hub, session)
        active_repo = request.app.state.repo
        # Geometry follows the ACTIVE session (auto-mapped per video),
        # never the boot layout or a previous video's seats.
        active_layout = request.app.state.layout
        if session:
            try:
                seats = {s.seat_id: s for s in active_repo.read_current_seat_states(session)}
                events = active_repo.read_event_timeline(session, limit=200)
            except Exception:
                seats, events = {}, []
            snap = None
            # History mode renders the stored rows (same attribute contract
            # as live seat_rows) so a reopened session shows its seats,
            # counts and evidence instead of an empty shell.
            try:
                from ..attributes import build_seat_map as _build_rows
                from ..attributes import live_counts as _live_counts

                read_occupants = getattr(active_repo, "read_current_occupants", None)
                occupants = read_occupants(session) if read_occupants is not None else []
                seat_rows = _build_rows(list(seats.values()), occupants)
                read_observation = getattr(active_repo, "read_latest_frame_observation", None)
                observation = read_observation(session) if read_observation is not None else None
                seat_rows = _overlay_persisted_observation(seat_rows, observation)
                counts = _live_counts(seat_rows)
            except Exception:
                seat_rows = []
                counts = {}
        else:
            seats, events, snap = _live_seats(live_hub), list(live_hub.events), live_hub.snapshot
            with live_hub.lock:
                seat_rows = list(live_hub.seat_rows)
                counts = dict(live_hub.counts)
        seat_rows = _with_display_names(active_layout, seat_rows)
        if session:
            seats = {str(row["seat_id"]): row for row in seat_rows if row.get("seat_id")}
        return {
            "request": request,
            "cfg": cfg,
            "badge": badge,
            "session": session,
            "seats": seats,
            "seat_rows": seat_rows,
            "counts": counts,
            "legend": LEGEND,
            "events": events,
            "snapshot": snap,
            "layout": active_layout,
            **extra,
        }

    @app.get("/", include_in_schema=False)
    def index() -> Any:
        from fastapi.responses import RedirectResponse

        return RedirectResponse("/live")

    @app.get("/live", response_class=HTMLResponse)
    def live(request: Request, session: str | None = None) -> Any:
        c = ctx(request, session)
        c["seat_svg"] = seat_svg(c["layout"], c["seats"], "car")
        with live_hub.lock:
            c["detections"] = list(live_hub.detections)
            c["masks"] = list(live_hub.masks)
            c["has_frame"] = live_hub.frame_jpeg is not None
            c["frame_seq"] = live_hub.frame_seq
            c["frame_ts"] = live_hub.frame_ts
            c["crop_ids"] = sorted(live_hub.crops.keys())
            c["history"] = [
                {
                    "seq": h.seq,
                    "captured_at": h.captured_at,
                    "n_detections": h.n_detections,
                    "counts": h.counts,
                }
                for h in list(live_hub.history)[-20:]
            ]
        return templates.TemplateResponse(request, "live.html", c)

    @app.get("/seats", response_class=HTMLResponse)
    def seats(request: Request, session: str | None = None, view: str = "car") -> Any:
        c = ctx(request, session)
        view = view if view in ("car", "poly") else "car"
        c["seat_svg"] = seat_svg(c["layout"], c["seats"], view)
        c["seat_view"] = view
        return templates.TemplateResponse(request, "seats.html", c)

    @app.get("/calibrate", response_class=HTMLResponse)
    def calibrate(request: Request) -> Any:
        # Merged into the seat map: the seats page is the only editor
        # (frame capture, trace, rectangle, move, validate, save, apply).
        from fastapi.responses import RedirectResponse

        return RedirectResponse("/seats", status_code=302)

    @app.post("/api/layouts/save")
    async def api_layouts_save(request: Request) -> Any:
        """Store a calibration layout in PLOMID (videos stay on disk; every
        seat map and calibration lives in the database)."""
        from pydantic import BaseModel, ConfigDict

        class SaveRequest(BaseModel):
            model_config = ConfigDict(extra="forbid")

            name: str
            seats: list[dict[str, Any]]
            camera_profile: str | None = None
            overwrite: bool = False

        try:
            body = SaveRequest(**await request.json())
        except Exception as e:
            return JSONResponse({"error": f"invalid request: {e}"}, status_code=400)
        import anyio as _anyio

        from ..seat_store import save_named_layout

        active_repo = request.app.state.repo

        def _save() -> tuple[bool, str]:
            return save_named_layout(
                active_repo, body.name, body.seats, body.camera_profile, body.overwrite
            )

        ok, message = await _anyio.to_thread.run_sync(_save)
        if ok:
            return JSONResponse({"saved": True, "layout": message, "seats": len(body.seats)})
        if "exists" in message:
            return JSONResponse({"error": message}, status_code=409)
        if "unavailable" in message or "failed" in message:
            return JSONResponse({"error": message}, status_code=503)
        return JSONResponse({"error": message}, status_code=400)

    @app.post("/api/layouts/apply")
    async def api_layouts_apply(request: Request) -> Any:
        """Make a saved PLOMID layout the ACTIVE geometry right now.

        The console map, overlays, editors and timeline follow immediately.
        A running session hot-swaps without restart or reprocessing
        (carried-over seats keep temporal state); an idle console just
        switches geometry for the next Start.
        """
        from pydantic import BaseModel, ConfigDict

        from ..seat_store import load_named_layout

        class ApplyRequest(BaseModel):
            model_config = ConfigDict(extra="forbid")

            name: str

        try:
            body = ApplyRequest(**await request.json())
        except Exception as e:
            return JSONResponse({"error": f"invalid request: {e}"}, status_code=400)
        import anyio as _anyio

        active_repo = request.app.state.repo

        def _load() -> Any:
            layout, reason = load_named_layout(active_repo, body.name)
            return layout, reason

        layout, reason = await _anyio.to_thread.run_sync(_load)
        if layout is None:
            return JSONResponse({"error": reason}, status_code=404)
        request.app.state.layout = layout
        hot_swapped: dict[str, int] | None = None
        with request.app.state.service_lock:
            service = request.app.state.service
            alive = (
                service is not None and service._thread is not None and service._thread.is_alive()
            )
            if alive:
                try:
                    hot_swapped = service.adopt_layout(layout)
                except Exception:
                    hot_swapped = None
        return JSONResponse(
            {
                "applied": True,
                "layout": layout.version,
                "seats": len(layout.seats),
                "running": bool(alive),
                "hot_swapped": hot_swapped,
            }
        )

    @app.post("/api/layouts/validate")
    async def api_layouts_validate(request: Request) -> Any:
        """Geometry check for the editors: hard errors + overlap warnings.

        Pure validation, never writes. Returns
        {"errors": [...], "warnings": [...], "rows": [[seat...], ...]} so
        both calibration and seat-map screens warn about overlapping,
        missing or mis-ordered zones before saving.
        """
        from ..layouts import validate_layout_geometry, validate_layout_payload

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        name = body.get("name", "draft")
        seats = body.get("seats", [])
        errors = validate_layout_payload(name if isinstance(name, str) else "draft", seats)
        warnings = validate_layout_geometry(seats)
        rows: list[list[str]] = []
        try:
            from ..domain import SeatGeometry as _Seat
            from ..seats import SeatLayout as _Layout
            from ..seats import order_seats_front_to_rear

            layout = _Layout(
                camera_profile="validation",
                version="validation",
                seats=[
                    _Seat(
                        seat_id=str(s.get("seat_id", f"seat-{i}")),
                        polygon=[(float(x), float(y)) for x, y in s.get("polygon", [])],
                        camera_profile="validation",
                        version="validation",
                    )
                    for i, s in enumerate(seats)
                    if isinstance(s, dict) and isinstance(s.get("polygon"), list)
                ],
            )
            rows = order_seats_front_to_rear(layout)
        except Exception:
            rows = []
        return JSONResponse({"errors": errors, "warnings": warnings, "rows": rows})

    @app.get("/api/layouts/current")
    def api_layouts_current(request: Request) -> Any:
        """Active session geometry + validation, for the seat-map editor."""
        from ..seats import find_overlapping_seats, order_seats_front_to_rear, validate_seat_layout

        layout = request.app.state.layout
        try:
            warnings = validate_seat_layout(layout)
            overlaps = find_overlapping_seats(layout)
            rows = order_seats_front_to_rear(layout)
        except Exception:
            warnings, overlaps, rows = ["validation unavailable"], [], []
        return JSONResponse(
            {
                "version": layout.version,
                "camera_profile": layout.camera_profile,
                "seats": [
                    {
                        "seat_id": s.seat_id,
                        "display_name": seat_display_name(s),
                        "polygon": [[float(x), float(y)] for x, y in s.polygon],
                        "row": s.row,
                        "col": s.col,
                        "label": s.label,
                        "display_number": s.display_number,
                        "physical_role": s.physical_role,
                    }
                    for s in layout.seats
                ],
                "warnings": warnings,
                "overlaps": overlaps,
                "rows": rows,
            }
        )

    @app.get("/api/seatmap/cache-status")
    def api_seatmap_cache_status(request: Request, video: str = Query(default="")) -> Any:
        """Persistence inspector: does PLOMID hold a seat map / probe for
        this video? Videos stay on disk; everything else lives in PLOMID."""
        from ..seat_store import load_probe_cache, load_video_seatmap, video_fingerprint

        if not video:
            return JSONResponse({"error": "missing ?video=<path>"}, status_code=400)
        active_repo = request.app.state.repo
        fp = video_fingerprint(video)
        saved, saved_reason = load_video_seatmap(active_repo, video)
        _probe, probe_reason = load_probe_cache(active_repo, video)
        return JSONResponse(
            {
                "video": video,
                "fingerprint_ok": bool(fp.get("ok")),
                "fingerprint": {k: v for k, v in fp.items() if k != "sha"},
                "saved_seatmap": None
                if saved is None
                else {"seats": len(saved.seats), "version": saved.version},
                "saved_reason": saved_reason,
                "probe_reason": probe_reason,
            }
        )

    @app.get("/timeline", response_class=HTMLResponse)
    def timeline(request: Request, session: str | None = None) -> Any:
        return templates.TemplateResponse(request, "timeline.html", ctx(request, session))

    @app.get("/diagnostics", response_class=HTMLResponse)
    def diagnostics(request: Request) -> Any:
        c = ctx(request)
        with live_hub.lock:
            c["latencies"] = list(live_hub.latencies)
        return templates.TemplateResponse(request, "diagnostics.html", c)

    @app.get("/relations", response_class=HTMLResponse)
    def relations(request: Request) -> Any:
        active = request.app.state.repo
        model = None
        error = None
        try:
            read_model = getattr(active, "read_schema_model", None)
            if read_model is not None:
                model = read_model()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return templates.TemplateResponse(
            request, "relations.html", ctx(request, model=model, model_error=error)
        )

    @app.get("/api/schema")
    def api_schema(request: Request) -> Any:
        active = request.app.state.repo
        read_model = getattr(active, "read_schema_model", None)
        if read_model is None:
            return JSONResponse({"error": "schema model is unavailable"}, status_code=503)
        try:
            return JSONResponse(read_model())
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=503)

    @app.get("/validation", response_class=HTMLResponse)
    def validation(request: Request) -> Any:
        active = request.app.state.repo
        try:
            runs = active.read_validation_runs() if hasattr(active, "read_validation_runs") else []
        except Exception:
            runs = []
        return templates.TemplateResponse(request, "validation.html", ctx(request, runs=runs))

    @app.get("/config", response_class=HTMLResponse)
    def config(request: Request) -> Any:
        active = request.app.state.repo
        try:
            manifest = active.read_model_manifest(cfg.detection.model_id)
        except Exception:
            manifest = None
        try:
            versions = (
                active.list_config_versions() if hasattr(active, "list_config_versions") else []
            )
        except Exception:
            versions = []
        return templates.TemplateResponse(
            request, "config.html", ctx(request, manifest=manifest, versions=versions)
        )

    @app.get("/dossier", response_class=HTMLResponse)
    def dossier(request: Request) -> Any:
        # Keep bookmarked URLs working while removing this unused screen from
        # the product navigation. The separate machine-readable API remains.
        from fastapi.responses import RedirectResponse

        return RedirectResponse("/live", status_code=303)

    @app.get("/api/service/preview.jpeg", include_in_schema=False)
    def api_service_preview(path: str = Query(min_length=1, max_length=512)) -> Any:
        """Serve a cached first-frame preview for a video inside footage/."""
        requested = Path(path)
        footage_root = Path("footage").resolve()
        if requested.is_absolute():
            return JSONResponse(
                {"error": "video path must be relative to footage/"}, status_code=400
            )
        target = (Path.cwd() / requested).resolve()
        try:
            target.relative_to(footage_root)
        except ValueError:
            return JSONResponse({"error": "video preview is limited to footage/"}, status_code=400)
        if not target.is_file() or target.suffix.lower() not in {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
        }:
            return JSONResponse({"error": "video file not found"}, status_code=404)
        try:
            stat = target.stat()
            payload = _read_first_video_frame(str(target), stat.st_mtime_ns, stat.st_size)
        except Exception as exc:
            log.info("video preview unavailable: %s", type(exc).__name__)
            payload = None
        if payload is None:
            return JSONResponse(
                {"error": "first video frame could not be decoded"}, status_code=422
            )
        return Response(
            payload,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/frame.jpeg", include_in_schema=False)
    def frame(seq: int | None = None) -> Any:
        with live_hub.lock:
            if seq is not None:
                for h in live_hub.history:
                    if h.seq == seq:
                        return Response(content=h.jpeg, media_type="image/jpeg")
                return Response(status_code=404)
            jpeg = live_hub.frame_jpeg
        if jpeg is None:
            return Response(status_code=204)
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/frame-thumb.jpeg", include_in_schema=False)
    def frame_thumbnail(seq: int, v: str) -> Any:
        """Return a cached small preview of a real frame in the live ring."""
        with live_hub.lock:
            entry = next(
                (item for item in live_hub.history if item.seq == seq and item.captured_at == v),
                None,
            )
            if entry is None:
                return Response(status_code=404)
            key = (seq, v)
            cached = frame_thumbnail_cache.get(key)
            source = entry.jpeg
        if cached is None:
            try:
                import cv2
                import numpy as np

                decoded = cv2.imdecode(np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    return Response(status_code=422)
                height, width = decoded.shape[:2]
                target_width = min(240, width)
                target_height = max(1, round(height * target_width / width))
                thumbnail = cv2.resize(
                    decoded, (target_width, target_height), interpolation=cv2.INTER_AREA
                )
                ok, encoded = cv2.imencode(".jpg", thumbnail, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                if not ok:
                    return Response(status_code=500)
                cached = encoded.tobytes()
            except Exception:
                return Response(status_code=503)
            with live_hub.lock:
                frame_thumbnail_cache[key] = cached
                if len(frame_thumbnail_cache) > 64:
                    frame_thumbnail_cache.pop(next(iter(frame_thumbnail_cache)))
        return Response(
            content=cached,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/crop/{crop_id}.jpeg", include_in_schema=False)
    def crop(crop_id: str) -> Any:
        with live_hub.lock:
            jpeg = live_hub.crops.get(crop_id)
        if jpeg is None:
            return Response(status_code=204)
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/partials/status", response_class=HTMLResponse)
    def partial_status(request: Request, session: str | None = None) -> Any:
        return templates.TemplateResponse(
            request, "partials/_statusbar.html", ctx(request, session)
        )

    @app.get("/partials/seat-map", response_class=HTMLResponse)
    def partial_seatmap(request: Request, session: str | None = None, view: str = "car") -> Any:
        c = ctx(request, session)
        c["seat_svg"] = seat_svg(
            c["layout"], c["seats"], view if view in ("car", "poly") else "car"
        )
        return templates.TemplateResponse(request, "partials/_seatmap.html", c)

    @app.get("/partials/seat-detail", response_class=HTMLResponse)
    def partial_seat_detail(request: Request, seat: str, session: str | None = None) -> Any:
        c = ctx(request, session)
        state = c["seats"].get(seat)
        display_name = next(
            (seat_display_name(item) for item in c["layout"].seats if item.seat_id == seat),
            seat,
        )
        # Live rows are dicts with full attributes; history rows are SeatState.
        row = state if isinstance(state, dict) else None
        return templates.TemplateResponse(
            request,
            "partials/_seatdetail.html",
            {**c, "selected": seat, "selected_name": display_name, "state": state, "row": row},
        )

    @app.get("/sse/stream", include_in_schema=False)
    async def sse_stream() -> Any:
        """Continuous live event stream: version-gated updates + heartbeats.

        Never ends after N messages (the old 60-iteration cap caused stale
        frames). Emits a heartbeat comment every 15 s so proxies/LB do not
        kill idle connections; the frontend reconnects with backoff.
        """

        async def gen() -> Any:
            last = -1
            idle_ticks = 0
            while True:
                await asyncio.sleep(0.5)
                with live_hub.lock:
                    if live_hub.version == last:
                        idle_ticks += 1
                        if idle_ticks >= 30:
                            idle_ticks = 0
                            yield ": heartbeat\n\n"
                        continue
                    idle_ticks = 0
                    last = live_hub.version
                    payload = {
                        "version": last,
                        "badge": _badge(live_hub, None),
                        "frame_seq": live_hub.frame_seq,
                        "frame_ts": live_hub.frame_ts,
                        "snapshot": _live_snapshot_dict(live_hub),
                        "counts": dict(live_hub.counts),
                        "n_detections": len(live_hub.detections),
                        "n_masks": len(live_hub.masks),
                    }
                yield f"data: {json.dumps(payload)}\n\n"

        from fastapi.responses import StreamingResponse

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/live/state")
    @app.get("/api/v1/live/state")
    def api_live_state(request: Request) -> Any:
        """Single incremental snapshot: frame metadata + detections + masks
        + seat rows + counts. Polled as SSE fallback; version-gated."""
        from ..attributes import analytics_summary

        with live_hub.lock:
            rows = list(live_hub.seat_rows)
            counts = dict(live_hub.counts)
            # Active session geometry (auto-mapped per video), not boot layout.
            active = request.app.state.layout
            by_id = {s.seat_id: s for s in active.seats}
            polygons = []
            for r in rows:
                seat = by_id.get(r["seat_id"])
                if seat is None:
                    continue
                label = f"{seat_display_name(seat)} {r['presence']}"
                if r.get("child") == "Yes":
                    label += " +child"
                if r.get("restraint") == "Yes":
                    label += " +seat"
                elif r.get("restraint_candidate"):
                    label += " +seat-candidate"
                gender = r.get("gender", "")
                if gender in ("APPARENT_MAN", "APPARENT_WOMAN"):
                    label += " +M" if gender == "APPARENT_MAN" else " +F"
                if r.get("belt_issue") is True:
                    label += " +belt?"
                elif r.get("belt") == "Detected":
                    label += " +strap-candidate"
                try:
                    color = seat_display_color(r)
                except Exception:
                    color = "#9ca3af"
                polygons.append(
                    {
                        "seat_id": r["seat_id"],
                        "display_name": seat_display_name(seat),
                        "polygon": [[float(x), float(y)] for x, y in seat.polygon],
                        "color": color,
                        "label": label,
                        "child": r.get("child", "Unknown"),
                        "child_reason": r.get("child_reason", ""),
                        "child_presence": r.get("child_presence", "UNKNOWN"),
                        "restraint": r.get("restraint") == "Yes",
                        "restraint_state": r.get("restraint", "Unknown"),
                        "restraint_candidate": bool(r.get("restraint_candidate")),
                        "restraint_reason": r.get("restraint_reason", ""),
                        "restraint_confidence": r.get("restraint_confidence"),
                        "restraint_confidence_semantics": r.get("restraint_confidence_semantics"),
                        "child_seat_presence": r.get("child_seat_presence", "UNKNOWN"),
                        "child_seat_type": r.get("child_seat_type", "UNKNOWN"),
                        "child_seat_type_candidate": r.get("child_seat_type_candidate", "UNKNOWN"),
                        "child_seat_occupancy": r.get("child_seat_occupancy", "UNKNOWN"),
                        "child_seat_occupancy_reason": r.get(
                            "child_seat_occupancy_reason", "unavailable"
                        ),
                        "child_seat_candidate": bool(r.get("child_seat_candidate", False)),
                        "child_candidate": bool(r.get("child_candidate", False)),
                        "belt_issue": r.get("belt_issue") is True,
                        "belt": r.get("belt", "Unknown"),
                        "belt_confidence": r.get("belt_confidence"),
                        "belt_confidence_semantics": r.get("belt_confidence_semantics"),
                        "belt_observation": r.get("belt_observation", "unavailable"),
                        "belt_observation_reason": r.get("belt_observation_reason", "unavailable"),
                        "belt_observation_score": r.get("belt_observation_score"),
                        "belt_status_semantics": r.get("belt_status_semantics"),
                        "gender": r.get("gender", "Unknown"),
                        "gender_conf": r.get("gender_conf"),
                        "gender_confidence_semantics": r.get("gender_confidence_semantics"),
                        "age_years": next(
                            (item.get("age_years") for item in r.get("age_predictions", [])),
                            None,
                        ),
                        "age_reason": r.get("age_reason", "unavailable"),
                        "age_model_id": r.get("age_model_id"),
                    }
                )
            try:
                analytics = analytics_summary(rows, counts)
            except Exception:
                analytics = {"percentages": {}, "notes": []}
            service = getattr(request.app.state, "service", None)
            age_config = getattr(service._cfg, "age", None) if service is not None else None
            if service is not None and getattr(age_config, "enabled", False):
                if getattr(service, "_age_estimator", None) is None:
                    gender_status = {
                        "state": "unavailable",
                        "message": "MiVOLO age and apparent-gender model unavailable",
                        "detail": getattr(service, "_age_error", None)
                        or "Check local model and runtime dependencies.",
                    }
                else:
                    gender_status = {
                        "state": "available",
                        "message": "MiVOLO age and apparent-gender outputs share one inference",
                    }
            elif service is None or not service._cfg.detection.enable_gender_appearance:
                gender_status = {"state": "off", "message": "Appearance estimate: off"}
            elif service._gender_loading:
                gender_status = {
                    "state": "loading",
                    "message": "Loading local appearance model…",
                }
            elif service._gender_probe is None and service._gender_tried:
                gender_status = {
                    "state": "unavailable",
                    "message": "Appearance model unavailable",
                    "detail": service._gender_error
                    or "Model load failed; restart with the model available.",
                }
            elif service._gender_probe is None:
                gender_status = {
                    "state": "starting",
                    "message": "Appearance model starting…",
                }
            elif any(r.get("gender") in ("APPARENT_MAN", "APPARENT_WOMAN") for r in rows):
                gender_status = {
                    "state": "available",
                    "message": (
                        "Appearance estimates active · unclassified occupants remain unknown"
                    ),
                }
            else:
                gender_status = {
                    "state": "ready",
                    "message": "Model ready · waiting for usable adult occupant crops",
                }
            return JSONResponse(
                {
                    "api_schema_version": "odc.live.v1",
                    "version": live_hub.version,
                    "badge": _badge(live_hub, None),
                    "orientation": active.orientation,
                    "orientation_reason": active.orientation_reason,
                    "frame_seq": live_hub.frame_seq,
                    "frame_ts": live_hub.frame_ts,
                    "has_frame": live_hub.frame_jpeg is not None,
                    "snapshot": _live_snapshot_dict(live_hub),
                    "detections": list(live_hub.detections),
                    "masks": list(live_hub.masks),
                    "seat_rows": rows,
                    "seat_polygons": polygons,
                    "counts": counts,
                    "analytics": analytics,
                    "gender_status": gender_status,
                    "belt_overlays": list(live_hub.belt_overlays),
                    "pose": dict(live_hub.pose),
                    "face": dict(live_hub.face),
                    "alerts": list(live_hub.alerts),
                    "precise": {
                        "seq": live_hub.precise_seq,
                        "masks": list(live_hub.precise_masks),
                    },
                    "crop_ids": sorted(live_hub.crops.keys()),
                }
            )

    @app.get("/api/live/counts")
    def api_live_counts() -> Any:
        with live_hub.lock:
            return JSONResponse({"counts": dict(live_hub.counts), "version": live_hub.version})

    @app.get("/api/alerts")
    def api_alerts(request: Request) -> Any:
        """Active + recent alerts, each anchored to its seat's zone.

        seat_anchor carries the seat polygon centroid (normalized 0..1) and
        frame_seq so the UI positions the alert badge next to its seat and
        jumps to the right frame — one coordinate system, no hardcoded
        pixel positions, stable across resize/scroll/zoom/video switches.
        """
        with live_hub.lock:
            active = list(live_hub.alerts)
            recent = list(live_hub.alerts_history)
            layout = request.app.state.layout
        try:
            from ..seats import seat_centroid as _centroid

            anchors = {}
            for s in layout.seats:
                try:
                    cx, cy = _centroid(s)
                    anchors[s.seat_id] = {
                        "cx": round(float(cx), 4),
                        "cy": round(float(cy), 4),
                        "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in s.polygon],
                    }
                except Exception:
                    continue
        except Exception:
            anchors = {}

        def _anchor(item: dict[str, Any]) -> dict[str, Any]:
            out = dict(item)
            sid = item.get("seat_id")
            out["seat_anchor"] = anchors.get(sid) if sid else None
            return out

        return JSONResponse(
            {"active": [_anchor(a) for a in active], "recent": [_anchor(a) for a in recent]}
        )

    @app.get("/api/frames/history")
    def api_frames_history() -> Any:
        with live_hub.lock:
            return JSONResponse(
                {
                    "live_seq": live_hub.frame_seq,
                    "frames": [
                        {
                            "seq": h.seq,
                            "captured_at": h.captured_at,
                            "n_detections": h.n_detections,
                            "counts": h.counts,
                        }
                        for h in list(live_hub.history)
                    ],
                }
            )

    @app.get("/api/frames/live-evidence")
    def api_live_frame_evidence(seq: int) -> Any:
        """Return transient inference overlays paired with a buffered frame."""
        with live_hub.lock:
            frame_entry = next((item for item in live_hub.history if item.seq == seq), None)
            if frame_entry is None:
                return JSONResponse(
                    {"error": "frame evidence expired from the live buffer"}, status_code=404
                )
            return JSONResponse(
                {
                    "seq": frame_entry.seq,
                    "captured_at": frame_entry.captured_at,
                    "detections": frame_entry.detections,
                    "masks": frame_entry.masks,
                    "belt_overlays": frame_entry.belt_overlays,
                    "pose": frame_entry.pose,
                    "face": frame_entry.face,
                }
            )

    @app.get("/api/frames/detections")
    def api_frame_detections(request: Request, session: str, frame_seq: int | None = None) -> Any:
        """Durable per-frame boxes/scores from PLOMID, including vector bbox text."""
        reader = getattr(request.app.state.repo, "read_frame_detections", None)
        if reader is None:
            return JSONResponse(
                {"error": "frame evidence repository capability unavailable"}, status_code=503
            )
        try:
            return JSONResponse({"session_id": session, "detections": reader(session, frame_seq)})
        except Exception as exc:
            return JSONResponse(
                {"error": f"PLOMID frame evidence unavailable: {type(exc).__name__}"},
                status_code=503,
            )

    @app.post("/api/assistant/ask")
    async def api_assistant_ask(request: Request) -> Any:
        """Explicit user-triggered assistant analysis; never automatic."""
        from pydantic import BaseModel, ConfigDict, Field

        class AskRequest(BaseModel):
            model_config = ConfigDict(extra="forbid")

            prompt: str = Field(min_length=1, max_length=4000)
            seat_id: str | None = None
            frame_seq: int | None = None

        try:
            body = AskRequest(**await request.json())
        except Exception as exc:
            return JSONResponse({"error": f"invalid assistant request: {exc}"}, status_code=400)
        with live_hub.lock:
            rows = list(live_hub.seat_rows)
            occupied_rows = [
                row for row in rows if row.get("presence") in ("OCCUPIED", "LIKELY_OCCUPIED")
            ]
            child_yes = sum(1 for row in occupied_rows if row.get("child") == "Yes")
            child_no = sum(1 for row in occupied_rows if row.get("child") == "No")
            child_unknown = sum(1 for row in occupied_rows if row.get("child") == "Unknown")
            apparent_male = sum(1 for row in occupied_rows if row.get("gender") == "APPARENT_MAN")
            apparent_female = sum(
                1 for row in occupied_rows if row.get("gender") == "APPARENT_WOMAN"
            )
            gender_unknown = max(0, len(occupied_rows) - apparent_male - apparent_female)
            context = {
                "orientation": request.app.state.layout.orientation,
                "layout": request.app.state.layout.version,
                "frame_seq": live_hub.frame_seq,
                "requested_frame_seq": body.frame_seq,
                "seat_id": body.seat_id,
                "snapshot": _live_snapshot_dict(live_hub),
                "seat_rows": rows,
                "counts": dict(live_hub.counts),
                "alerts": list(live_hub.alerts),
                "detections": list(live_hub.detections),
                "assistant_facts": {
                    "occupied_seat_evidence_count": len(occupied_rows),
                    "child_candidate_seat_count": child_yes,
                    "child_negative_evidence_seat_count": child_no,
                    "child_unknown_seat_count": child_unknown,
                    "apparent_male_seat_count": apparent_male,
                    "apparent_female_seat_count": apparent_female,
                    "gender_unknown_seat_count": gender_unknown,
                    "interpretation": (
                        "Counts are seat-level evidence, not a certified person count. "
                        "APPARENT_MAN/APPARENT_WOMAN are experimental appearance labels, "
                        "not verified sex or gender. Child No means no child candidate "
                        "from current evidence; Child Unknown is insufficient evidence."
                    ),
                },
            }
            session_id = live_hub.snapshot.session_id or None
            actual_frame_seq = live_hub.frame_seq
        from ..ollama import (
            OUT_OF_SCOPE_MESSAGE,
            OllamaConfig,
            OllamaUnavailable,
            generate,
            is_in_scope,
        )

        provider = request.app.state.cfg.assistant.provider
        model = request.app.state.cfg.assistant.model
        interaction_id = uuid4().hex

        def persist_interaction(
            status: str,
            provider: str,
            model: str | None,
            answer: str | None,
            evidence_context: dict[str, Any],
        ) -> None:
            writer = getattr(request.app.state.repo, "append_assistant_interaction", None)
            if writer is None:
                log.warning("assistant persistence unavailable; interaction_id=%s", interaction_id)
                return
            try:
                writer(
                    {
                        "interaction_id": interaction_id,
                        "session_id": session_id,
                        "occurred_at": datetime.now(UTC),
                        "frame_seq": (
                            body.frame_seq if body.frame_seq is not None else actual_frame_seq
                        ),
                        "seat_id": body.seat_id,
                        "provider": provider,
                        "model": model,
                        "status": status,
                        "prompt": body.prompt,
                        "answer": answer,
                        "context": evidence_context,
                    }
                )
            except Exception as exc:
                log.warning(
                    "assistant persistence failed; interaction_id=%s kind=%s",
                    interaction_id,
                    type(exc).__name__,
                )

        if body.frame_seq is not None and body.frame_seq != actual_frame_seq:
            message = (
                f"Frame {body.frame_seq} is not the current frame ({actual_frame_seq}); "
                "assistant analysis was not run."
            )
            persist_interaction("frame_unavailable", "plomid-context-guard", None, message, {})
            return JSONResponse(
                {
                    "error": message,
                    "provider": "plomid-context-guard",
                    "model": None,
                    "interaction_id": interaction_id,
                },
                status_code=409,
            )

        if not is_in_scope(body.prompt):
            persist_interaction(
                "scope_blocked", "plomid-scope-guard", None, OUT_OF_SCOPE_MESSAGE, {}
            )
            return JSONResponse(
                {
                    "answer": OUT_OF_SCOPE_MESSAGE,
                    "provider": "plomid-scope-guard",
                    "model": None,
                    "trigger": "scope-guard",
                    "interaction_id": interaction_id,
                }
            )

        try:
            import anyio as _anyio

            answer = await _anyio.to_thread.run_sync(
                generate,
                body.prompt,
                context,
                OllamaConfig(
                    enabled=request.app.state.cfg.assistant.enabled,
                    provider=provider,
                    url=request.app.state.cfg.assistant.url,
                    model=model,
                    timeout_s=request.app.state.cfg.assistant.timeout_s,
                ),
            )
        except OllamaUnavailable as exc:
            persist_interaction("unavailable", provider, model, None, context)
            return JSONResponse(
                {
                    "error": str(exc),
                    "provider": provider,
                    "model": model,
                    "interaction_id": interaction_id,
                },
                status_code=503,
            )
        persist_interaction("completed", provider, model, answer, context)
        return JSONResponse(
            {
                "answer": answer,
                "provider": provider,
                "model": model,
                "trigger": "explicit-user-request",
                "interaction_id": interaction_id,
            }
        )

    @app.get("/api/assistant/history")
    def api_assistant_history(
        request: Request,
        session: str | None = None,
        limit: int = Query(default=50, ge=1, le=500),
    ) -> Any:
        active = request.app.state.repo
        selected_session = session or live_hub.snapshot.session_id
        if not selected_session:
            return JSONResponse(
                {
                    "api_schema_version": "odc.assistant-history.v1",
                    "session_id": None,
                    "interactions": [],
                }
            )
        reader = getattr(active, "read_assistant_interactions", None)
        if reader is None:
            return JSONResponse({"error": "PLOMID assistant history unavailable"}, status_code=503)
        try:
            interactions = reader(selected_session, limit)
        except Exception as exc:
            return JSONResponse(
                {"error": f"PLOMID assistant history unavailable: {type(exc).__name__}"},
                status_code=503,
            )
        return JSONResponse(
            jsonable_encoder(
                {
                    "api_schema_version": "odc.assistant-history.v1",
                    "session_id": selected_session,
                    "interactions": interactions,
                }
            )
        )

    @app.get("/api/seats")
    def api_seats(request: Request, session: str | None = None) -> Any:
        active = request.app.state.repo
        if session:
            try:
                seats = active.read_current_seat_states(session)
                read_observation = getattr(active, "read_latest_frame_observation", None)
                observation = read_observation(session) if read_observation is not None else None
            except Exception as e:
                return JSONResponse({"error": f"{type(e).__name__}"}, status_code=503)
            stored_rows: dict[str, dict[str, Any]] = {}
            payload = observation.get("payload") if observation else None
            if isinstance(payload, dict) and isinstance(payload.get("seat_rows"), list):
                stored_rows = {
                    str(row["seat_id"]): row
                    for row in payload["seat_rows"]
                    if isinstance(row, dict) and row.get("seat_id") is not None
                }
            return JSONResponse(
                [
                    {
                        **stored_rows.get(s.seat_id, {}),
                        "seat_id": s.seat_id,
                        "display_name": seat_display_name(
                            next(
                                (
                                    x
                                    for x in request.app.state.layout.seats
                                    if x.seat_id == s.seat_id
                                ),
                                s,
                            )
                        ),
                        "presence": s.presence.value,
                        "confidence": s.confidence,
                        "visibility": s.visibility.value,
                        "reason_codes": s.reason_codes,
                        "object_label": s.object_label,
                        "child": stored_rows.get(s.seat_id, {}).get("child", "Unknown"),
                        "restraint": stored_rows.get(s.seat_id, {}).get("restraint", "Unknown"),
                        "restraint_candidate": stored_rows.get(s.seat_id, {}).get(
                            "restraint_candidate",
                            s.object_label in ("chair", "couch", "backpack", "handbag", "suitcase"),
                        ),
                        "belt": stored_rows.get(s.seat_id, {}).get("belt", "Unknown"),
                        "api_schema_version": "odc.seat-observation.v1",
                        "observation_frame_seq": (
                            observation.get("frame_seq") if observation else None
                        ),
                        "observation_captured_at": (
                            observation.get("captured_at") if observation else None
                        ),
                        "updated_at": s.updated_at.isoformat(),
                    }
                    for s in seats
                ]
            )
        with live_hub.lock:
            rows = list(live_hub.seat_rows)
        if rows:
            return JSONResponse(rows)
        seats = list(_live_seats(live_hub).values())
        return JSONResponse(
            [
                {
                    "seat_id": s.seat_id,
                    "display_name": seat_display_name(
                        next(
                            (x for x in request.app.state.layout.seats if x.seat_id == s.seat_id), s
                        )
                    ),
                    "presence": s.presence.value,
                    "confidence": s.confidence,
                    "visibility": s.visibility.value,
                    "reason_codes": s.reason_codes,
                    "updated_at": s.updated_at.isoformat(),
                }
                for s in seats
            ]
        )

    @app.get("/api/events")
    def api_events(
        request: Request,
        session: str | None = None,
        seat: str | None = None,
        limit: int = Query(default=200, le=1000),
    ) -> Any:
        active = request.app.state.repo
        if session:
            try:
                events = active.read_event_timeline(session, seat, limit)
            except Exception as e:
                return JSONResponse({"error": f"{type(e).__name__}"}, status_code=503)
        else:
            with live_hub.lock:
                events = [e for e in live_hub.events if seat is None or e.seat_id == seat][:limit]
        return JSONResponse(
            [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "seat_id": e.seat_id,
                    "display_name": seat_display_name(
                        next(
                            (x for x in request.app.state.layout.seats if x.seat_id == e.seat_id),
                            None,
                        )
                    )
                    if e.seat_id
                    else None,
                    "track_id": e.track_id,
                    "occurred_at": e.occurred_at.isoformat(),
                    "prior_state": e.prior_state,
                    "new_state": e.new_state,
                    "reason_codes": e.reason_codes,
                    "config_version": e.config_version,
                    "model_set_version": e.model_set_version,
                    # frame_seq is carried in evidence_summary for alerts
                    # (frame_seq=N;...) and parsed here so the timeline can
                    # jump to the exact video frame for every event kind.
                    "frame_seq": _event_frame_seq(e),
                    "evidence": e.evidence_summary,
                    "severity": e.severity.value,
                }
                for e in events
            ]
        )

    @app.get("/api/health")
    def api_health() -> Any:
        snap = live_hub.snapshot
        with live_hub.lock:
            lat = list(live_hub.latencies)
            counts = dict(live_hub.counts)
            frame_seq = live_hub.frame_seq
        ordered = sorted(lat)
        return JSONResponse(
            {
                "running": snap.running,
                "mode": snap.mode.value,
                "db_healthy": snap.db_healthy,
                "model_healthy": snap.model_healthy,
                "camera_status": snap.camera_status.value,
                "frames_captured": snap.frames_captured,
                "frames_processed": snap.frames_processed,
                "frames_dropped": snap.frames_dropped,
                "frame_seq": frame_seq,
                "counts": counts,
                "latency_p50_ms": ordered[len(ordered) // 2] if ordered else None,
            }
        )

    @app.get("/api/service/status")
    def api_service_status(request: Request) -> Any:
        service = request.app.state.service
        thread = service._thread if service else None
        running = thread is not None and thread.is_alive()
        return JSONResponse(
            {
                "running": running,
                "session_id": service._session_id if service else None,
                "badge": _badge(live_hub, None),
            }
        )

    @app.post("/api/service/start")
    async def api_service_start(request: Request) -> Any:
        from pydantic import BaseModel, ConfigDict, Field

        from ..seats import SeatLayout as _SeatLayout
        from ..seats import apply_orientation

        class StartRequest(BaseModel):
            model_config = ConfigDict(extra="forbid")

            source: str = "replay"
            path: str = "footage/F1_family_ride.mp4"
            camera_id: str = "cabin0"
            layout: str | None = "auto"
            orientation: str = "UNKNOWN"
            threshold: float = Field(default=0.7, ge=0.0, le=1.0)
            replay_loop: bool = True
            restart: bool = False
            pose: bool = False
            dms: bool = False
            gender: bool = False
            mask: str = "fast"

        try:
            body = StartRequest(**await request.json())
        except Exception as e:
            return JSONResponse({"error": f"invalid request: {e}"}, status_code=400)
        if body.source not in ("replay", "live"):
            return JSONResponse({"error": "source must be replay|live"}, status_code=400)
        if body.mask not in ("fast", "precise"):
            return JSONResponse({"error": "mask must be fast|precise"}, status_code=400)
        if body.orientation not in ("RHD", "LHD", "UNKNOWN"):
            return JSONResponse({"error": "orientation must be RHD|LHD|UNKNOWN"}, status_code=400)
        with request.app.state.service_lock:
            current = request.app.state.service
            current_alive = (
                current is not None and current._thread is not None and current._thread.is_alive()
            )
            restarted = False
            if current_alive:
                if not body.restart:
                    return JSONResponse({"error": "already running — stop first"}, status_code=409)
            base: AppConfig = request.app.state.cfg
            try:
                camera = base.camera.model_copy(
                    update={
                        "source": body.source,
                        "path": body.path,
                        "camera_id": body.camera_id,
                        "replay_loop": body.replay_loop,
                    }
                )
                detection = base.detection.model_copy(
                    update={
                        "threshold": body.threshold,
                        "mask_mode": body.mask,
                        "enable_gender_appearance": body.gender,
                    }
                )
                pose = base.pose.model_copy(update={"enabled": body.pose})
                dms = base.dms.model_copy(update={"enabled": body.dms})
                from ..domain import RunMode

                run_cfg = base.model_copy(
                    update={
                        "camera": camera,
                        "detection": detection,
                        "pose": pose,
                        "dms": dms,
                        "mode": RunMode.LIVE if body.source == "live" else RunMode.REPLAY,
                    }
                )
            except Exception as e:
                return JSONResponse({"error": f"invalid config: {e}"}, status_code=400)
            run_layout: Any = None
            force_auto = body.layout == "auto"
            layout_source = (
                "auto-requested" if force_auto else ("explicit" if body.layout else "curated")
            )
            probe_info: dict[str, Any] | None = None
            # Seat geometry source of truth is PLOMID (videos stay on disk;
            # every map/calibration lives in the database). Versioned YAML
            # seeds under configs/seats/ import on first use only.
            active_repo = request.app.state.repo
            from ..seat_store import (
                load_named_layout,
                load_probe_cache,
                load_video_seatmap,
                save_probe_cache,
                save_video_seatmap,
            )

            if body.layout and not force_auto:
                run_layout, layout_reason = load_named_layout(active_repo, body.layout)
                if force_auto and run_layout is not None and len(run_layout.seats) not in (2, 5, 7):
                    # Auto hotspot proposals are candidates. A four-zone
                    # result was the source of the supplied seat_1..seat_4
                    # snapshot; the product default remains the configured
                    # five-seat identity set until an operator confirms a
                    # different calibration.
                    fallback, _fallback_reason = load_named_layout(
                        active_repo, "standard-5-seat-rhd"
                    )
                    if fallback is not None:
                        run_layout = fallback
                        layout_source = "curated-fallback-5"
                        probe_info = {
                            **(probe_info or {}),
                            "reason": (
                                "auto proposal rejected: unsupported seat count; "
                                "five-seat default retained"
                            ),
                        }
                if run_layout is None:
                    return JSONResponse(
                        {"error": f"unknown layout: {layout_reason}"}, status_code=400
                    )
                probe_info = {"reason": layout_reason}
            else:
                # 1) camera-id layout (explicit calibration).
                if not force_auto:
                    run_layout, _ = load_named_layout(active_repo, f"demo-{body.camera_id}-v0")
                # 2) clip-name inference for the bundled demo footage
                # (F1_family_ride -> demo-f1-v0, E1_empty_cabin -> demo-e1-v0,
                # R1_rear_wide -> demo-r1-v0, ...). Keeps the demo seat map
                # working without manual layout selection.
                if run_layout is None and not force_auto:
                    import re as _re
                    from pathlib import Path as _P

                    stem = _P(body.path).stem  # e.g. F1_family_ride
                    m = _re.match(r"^([A-Za-z]+\d+)", stem)
                    if m:
                        clip_key = m.group(1).lower()  # f1, e1, r1, ...
                        run_layout, _ = load_named_layout(active_repo, f"demo-{clip_key}-v0")
                # 3) Saved seat map for THIS video (PLOMID, survives restarts).
                if run_layout is None and body.source == "replay" and not force_auto:
                    saved, saved_reason = load_video_seatmap(active_repo, body.path)
                    if saved is not None and saved.seats:
                        run_layout = saved
                        layout_source = "saved"
                        probe_info = {"reason": saved_reason, "reused": True}
                # 4) Unknown/custom video: ANALYSE first, then map. Probe
                # sustained person hotspots and propose seat regions. No
                # hotspots (or a missing file) -> honestly UNCALIBRATED,
                # never guessed seats. Runs in a thread, never blocks SSE.
                # Unchanged videos reuse the PLOMID probe (no reprocessing).
                if run_layout is None and body.source == "replay":
                    import anyio as _anyio

                    from ..autolayout import propose_seat_layout

                    def _probe() -> Any:
                        if not force_auto:
                            try:
                                cached, _reason = load_probe_cache(active_repo, body.path)
                                if cached is not None:
                                    return cached
                            except Exception:
                                pass
                        return propose_seat_layout(body.path, base.detection.model_id)

                    probe = await _anyio.to_thread.run_sync(_probe)
                    probe_info = {
                        "frames": probe.frames_probed,
                        "persons": probe.persons_seen,
                        "peaks": probe.peaks,
                        "reason": probe.reason,
                        "orientation": probe.orientation,
                        "orientation_confidence": probe.orientation_confidence,
                        "orientation_reason": probe.orientation_reason,
                    }
                    if probe.layout is not None:
                        run_layout = probe.layout
                        if body.orientation == "UNKNOWN" and probe.orientation in ("RHD", "LHD"):
                            run_layout = apply_orientation(run_layout, probe.orientation)
                        elif body.orientation != "UNKNOWN":
                            run_layout = apply_orientation(run_layout, body.orientation)
                        layout_source = f"auto-{len(probe.layout.seats)}"
                        # Persist both the probe (skip next time) and the
                        # seat map (reload on reopen) for this video.
                        try:
                            save_probe_cache(active_repo, body.path, probe)
                            save_video_seatmap(
                                active_repo,
                                body.path,
                                run_layout,
                                camera_id=body.camera_id,
                                origin="auto-probe",
                            )
                        except Exception:
                            pass
                if run_layout is None:
                    # Unknown geometry: seats stay UNKNOWN, never guessed
                    # from the wrong (or a previous video's) geometry.
                    if force_auto and body.source != "replay":
                        probe_info = {
                            "reason": (
                                "automatic seat mapping currently requires a prerecorded replay; "
                                "trace live-camera zones in Seat map"
                            )
                        }
                    run_layout = _SeatLayout(
                        camera_profile="uncalibrated", version="none", seats=[]
                    )
                    layout_source = "uncalibrated"
            if run_layout.seats and body.orientation != "UNKNOWN":
                run_layout = apply_orientation(run_layout, body.orientation)
            # Reuse previous analytics when video + layout + settings are
            # unchanged (recalculated only when an input changed).
            previous_analytics: dict[str, Any] | None = None
            if body.source == "replay" and run_layout.seats:
                try:
                    from ..seat_store import config_hash_for, load_analytics_snapshot

                    cached, _why = load_analytics_snapshot(
                        active_repo, body.path, run_layout, config_hash_for(run_cfg)
                    )
                    previous_analytics = cached
                except Exception:
                    previous_analytics = None
            # Layout is validated above: only now may a restart retire the
            # running session, so a bad layout never kills a live stream.
            if current_alive:
                restarted = True
                try:
                    current.stop()
                except Exception:
                    pass
                request.app.state.service = None
            service = Service(run_cfg, active_repo, run_layout, hub=live_hub)
            request.app.state.service = service
            # Console geometry follows the active session from here on, so
            # the map, polygons and dossier describe THIS video only.
            request.app.state.layout = run_layout
            session_id = service.start()
        return JSONResponse(
            {
                "started": True,
                "session_id": session_id,
                "restarted": restarted,
                "layout": run_layout.version,
                "layout_source": layout_source,
                "orientation": run_layout.orientation,
                "orientation_reason": run_layout.orientation_reason,
                "probe": probe_info,
                "previous_analytics": previous_analytics,
                "mask_mode": run_cfg.detection.mask_mode,
            }
        )

    @app.post("/api/service/stop")
    def api_service_stop(request: Request) -> Any:
        with request.app.state.service_lock:
            service = request.app.state.service
            if service is None:
                return JSONResponse({"stopped": False, "reason": "idle"})
            service.stop()
            request.app.state.service = None
        return JSONResponse({"stopped": True})

    @app.post("/api/service/upload")
    async def api_service_upload(request: Request, filename: str = Query(default="")) -> Any:
        """Custom video upload: raw request body + ?filename=. No multipart
        dependency on purpose (stdlib-only boundary). MP4/MOV/AVI/MKV/WebM,
        max 200 MB, sanitized name under footage/uploads/ (gitignored via
        footage/.gitignore). Body streams with a size cap (nothing is
        written until complete); disk work runs in a thread so the event
        loop never blocks. The file is verified decodable before it is
        accepted; failures report 4xx and leave no partial file."""
        import re as _re
        import time as _time

        import anyio as _anyio

        def _ensure_dir(path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)

        def _write(path: Path, data: bytes) -> None:
            path.write_bytes(data)

        def _remove(path: Path) -> None:
            path.unlink(missing_ok=True)

        def _probe_video(path: str) -> bool:
            import cv2

            cap = cv2.VideoCapture(path)
            ok, _ = cap.read()
            cap.release()
            return bool(ok)

        safe = _re.sub(r"[^A-Za-z0-9_.-]", "_", Path(filename).name)[:80]
        if not safe or "." not in safe:
            return JSONResponse({"error": "missing ?filename=<name>.mp4"}, status_code=400)
        if Path(safe).suffix.lower() not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            return JSONResponse(
                {"error": "only video files (.mp4/.mov/.avi/.mkv/.webm)"}, status_code=400
            )
        updir = Path("footage/uploads")
        await _anyio.to_thread.run_sync(_ensure_dir, updir)
        stem, suffix = safe.rsplit(".", 1)
        target = updir / f"{stem}-{int(_time.time())}.{suffix}"
        max_bytes = 200 * 1024 * 1024
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > max_bytes:
                return JSONResponse({"error": "file too large (max 200 MB)"}, status_code=413)
            chunks.append(chunk)
        if size == 0:
            return JSONResponse({"error": "empty file"}, status_code=400)
        await _anyio.to_thread.run_sync(_write, target, b"".join(chunks))
        if not await _anyio.to_thread.run_sync(_probe_video, str(target)):
            await _anyio.to_thread.run_sync(_remove, target)
            return JSONResponse({"error": "not a decodable video file"}, status_code=400)
        rel = f"footage/uploads/{target.name}"
        return JSONResponse({"path": rel, "clip": target.name})

    @app.get("/api/service/sources")
    def api_service_sources(request: Request) -> Any:
        from pathlib import Path as _Path

        footage = sorted(p.name for p in _Path("footage").glob("*.mp4"))
        uploads = sorted(
            f"uploads/{p.name}"
            for p in _Path("footage/uploads").glob("*")
            if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm")
        )
        # Layout names come from PLOMID (YAML seeds union in on DB failure).
        try:
            from ..seat_store import list_named_layouts

            layouts = list_named_layouts(request.app.state.repo)
        except Exception:
            layouts = sorted(p.stem for p in _Path("configs/seats").glob("*.yaml"))
        return JSONResponse({"clips": footage, "uploads": uploads, "layouts": layouts})

    @app.get("/fleet", response_class=HTMLResponse)
    def fleet(request: Request) -> Any:
        active = request.app.state.repo
        try:
            sessions = active.list_sessions() if hasattr(active, "list_sessions") else []
            criticals = (
                active.recent_critical_events() if hasattr(active, "recent_critical_events") else []
            )
        except Exception:
            sessions, criticals = [], []
        rollup: dict[str, int] = {}
        try:
            for session in sessions[:20]:
                for seat in active.read_current_seat_states(session["session_id"]):
                    rollup[seat.presence.value] = rollup.get(seat.presence.value, 0) + 1
        except Exception:
            rollup = {}
        return templates.TemplateResponse(
            request,
            "fleet.html",
            ctx(request, sessions=sessions, criticals=criticals, rollup=rollup),
        )

    @app.get("/api/fleet")
    def api_fleet(request: Request) -> Any:
        active = request.app.state.repo
        try:
            sessions = active.list_sessions() if hasattr(active, "list_sessions") else []
            criticals = (
                active.recent_critical_events() if hasattr(active, "recent_critical_events") else []
            )
            return JSONResponse(
                {
                    "sessions": sessions,
                    "critical_events": [
                        {
                            "event_id": e.event_id,
                            "session_id": e.session_id,
                            "event_type": e.event_type.value,
                            "seat_id": e.seat_id,
                            "occurred_at": e.occurred_at.isoformat(),
                        }
                        for e in criticals
                    ],
                }
            )
        except Exception as e:
            return JSONResponse({"error": f"{type(e).__name__}"}, status_code=503)

    @app.get("/api/dossier")
    def api_dossier(request: Request, format: str = "json") -> Any:
        active = request.app.state.repo
        try:
            manifest = active.read_model_manifest(cfg.detection.model_id)
        except Exception:
            manifest = None
        try:
            runs = active.read_validation_runs() if hasattr(active, "read_validation_runs") else []
            versions = (
                active.list_config_versions() if hasattr(active, "list_config_versions") else []
            )
            n_sessions = active.count_sessions() if hasattr(active, "count_sessions") else 0
        except Exception:
            runs, versions, n_sessions = [], [], 0
        dossier = build_dossier(cfg, request.app.state.layout, manifest, runs, versions, n_sessions)
        if format == "md":
            return PlainTextResponse(render_markdown(dossier))
        return JSONResponse(dossier)

    return app
