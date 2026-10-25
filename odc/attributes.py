"""Seat attribute aggregation: occupancy + child + restraint + belt.

Honesty rules (master plan S3.2 / S5.3):
- Occupancy comes ONLY from the temporal SeatStateEngine (person evidence
  over time). "No detection" never becomes "empty" — weak evidence stays
  UNKNOWN/STALE.
- Child = specialist age evidence associated to one generic person track.
  Apparent size and seat naming never assign age. Missing or conflicting
  evidence stays Unknown; a confirmed empty seat reports No.
- Child safety seat = sustained restraint-proxy evidence in the seat ROI.
  The engine verdict travels as a `restraint-proxy:{label}` reason token
  (chair/couch always; large suitcase/handbag/backpack since COCO mislabels
  seat shells and carriers with those names — measured on F1/C1/C2).
  COCO has no child-seat class, so this stays CHILD_SEAT_CANDIDATE, never
  a certified install check. Chair hits on an EMPTY plain vehicle seat stay
  Unknown (that is usually the vehicle seat itself).
- Seat belt = UNSUPPORTED camera-only. COCO has no belt class and the plan
  forbids inferring FASTENED from a belt-like line. The field is always
  Unknown camera-side (reason "unsupported-camera-only") unless a fused
  vehicle buckle signal is attached later. It is surfaced explicitly so the
  UI can show "Unknown — not reliably supported" instead of guessing.

Multiple attributes coexist per seat: occupancy, child, restraint and belt
are preserved side by side, never overwritten by one another.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .domain import Detection, OccupantState, SeatState
from .seats import SeatLayout, associate_track_first

RESTRAINT_PROXY_PREFIX = "restraint-proxy:"

BELT_UNSUPPORTED_REASON = "unsupported-camera-only: COCO has no belt class"

LEGEND: list[dict[str, str]] = [
    {
        "color": "#22c55e",
        "label": "Empty candidate",
        "meaning": "EMPTY_CANDIDATE sustained clear-scene evidence; not verified empty",
    },
    {"color": "#ef4444", "label": "Occupied", "meaning": "OCCUPIED person evidence"},
    {"color": "#eab308", "label": "Uncertain", "meaning": "LIKELY transitional / low confidence"},
    {"color": "#3b82f6", "label": "Child detected", "meaning": "APPARENT_SMALL candidate"},
    {"color": "#a855f7", "label": "Child safety seat", "meaning": "restraint-proxy ROI candidate"},
    {"color": "#f97316", "label": "Belt issue", "meaning": "occupied, no strap in clear view"},
    {"color": "#9ca3af", "label": "Unknown / occluded", "meaning": "weak/stale/blocked evidence"},
]

def _proxy_label(reason_codes: list[str]) -> str | None:
    for reason in reason_codes:
        if reason.startswith(RESTRAINT_PROXY_PREFIX):
            return reason[len(RESTRAINT_PROXY_PREFIX):]
    return None


def _specialist_score(reason_codes: list[str]) -> float | None:
    """Read the specialist score carried in the seat provenance tokens.

    Seat confidence measures temporal occupancy confirmation.  It is not a
    child-class score and must never be copied into ``child_confidence``.
    """
    child_scores: list[float] = []
    for reason in reason_codes:
        if reason.startswith("specialist-child-occupant:") and ":score=" in reason:
            try:
                child_scores.append(float(reason.rsplit(":score=", 1)[1]))
            except ValueError:
                continue
    for score in child_scores:
        if 0.0 <= score <= 1.0:
            return max(child_scores)
    for reason in reason_codes:
        if not reason.startswith("specialist-score:"):
            continue
        try:
            score = float(reason.removeprefix("specialist-score:"))
        except ValueError:
            continue
        if 0.0 <= score <= 1.0:
            return score
    return None


def seat_attributes(
    seat: SeatState,
    occupants: list[OccupantState],
    frame_seq: int = -1,
) -> dict[str, Any]:
    """Derive the full per-seat attribute row for the live seat map.

    Keeps seat identity stable (seat_id is the key), preserves every
    attribute, and marks uncertainty explicitly.
    """
    seat_occs = [o for o in occupants if o.seat_id == seat.seat_id]
    presence = seat.presence.value
    specialist_child = [
        reason for reason in seat.reason_codes if reason.startswith("specialist-child-occupant:")
    ]
    specialist_adult = [
        reason for reason in seat.reason_codes if reason.startswith("specialist-adult-occupant:")
    ]
    specialist_child_pending = [
        reason for reason in seat.reason_codes if reason.startswith("specialist-child-pending:")
    ]
    specialist_seat = [
        reason for reason in seat.reason_codes if reason.startswith("specialist-child-seat:")
    ]
    child_tracks = {
        reason.split(":track=", 1)[1].split(":", 1)[0]
        for reason in specialist_child
        if ":track=" in reason
    }
    adult_tracks = {
        reason.split(":track=", 1)[1].split(":", 1)[0]
        for reason in specialist_adult
        if ":track=" in reason
    }
    age_conflict = bool(specialist_child and specialist_adult) and (
        not child_tracks
        or not adult_tracks
        or bool(child_tracks.intersection(adult_tracks))
    )
    specialist_child_seat_candidate = bool(specialist_seat)
    # Child/adult status comes only from specialist evidence already linked
    # to one generic person track. Size and seat location cannot infer age.
    child = "Unknown"
    child_conf: float | None = None
    child_reason = "no-age-evidence"
    if age_conflict:
        child = "Unknown"
        child_reason = "specialist-age-conflict"
        child_conf = None
    elif specialist_child:
        child = "Yes"
        child_reason = specialist_child[-1]
        child_conf = _specialist_score(seat.reason_codes)
    elif specialist_child_pending:
        child = "Unknown"
        child_reason = "specialist-child-unconfirmed"
    elif specialist_adult:
        child = "No"
        child_reason = "specialist-adult-occupant"
    elif presence == "EMPTY_CANDIDATE":
        child = "No"
        child_reason = "empty-seat-no-occupant"

    # COCO chair/luggage labels and open-vocabulary specialist phrases are
    # candidates only; neither identifies a child restraint reliably. Seat
    # names, occupant size, and seat occupancy are not restraint evidence.
    proxy = _proxy_label(seat.reason_codes)
    restraint_candidate = bool(specialist_seat) or proxy is not None
    restraint = "Unknown"
    restraint_conf: float | None = None
    restraint_reason = (
        f"unverified-restraint-candidate:{specialist_seat[-1]}"
        if specialist_seat
        else f"unverified-restraint-candidate-proxy:{proxy}"
        if proxy is not None
        else "no-validated-restraint-detector"
    )

    # Belt default: camera-only is UNSUPPORTED by contract (service
    # overwrites with temporal tracker states when belt analysis runs).
    belt = "Unknown"
    belt_reason = BELT_UNSUPPORTED_REASON
    belt_issue = False
    # Occupant category follows the same track-associated specialist age
    # evidence as child status; size and region do not assign adult or child.
    category = "unknown"
    if specialist_child_pending or (specialist_child and specialist_adult):
        category = "unknown"
    elif specialist_child:
        category = "infant" if any(":infant" in item for item in specialist_child) else "child"
    elif specialist_adult:
        category = "adult"
    elif presence == "EMPTY_CANDIDATE":
        category = "none"
    if category in ("child", "infant"):
        placement = "restraint-unconfirmed" if restraint_candidate else "unknown"
    else:
        placement = "n/a"

    return {
        "seat_id": seat.seat_id,
        "presence": seat.presence.value,
        "confidence": seat.confidence,
        "visibility": seat.visibility.value,
        "reason_codes": list(seat.reason_codes),
        "object_label": seat.object_label,
        "child": child,
        "child_confidence": child_conf,
        "child_confidence_semantics": (
            "grounding-dino-phrase-similarity-uncalibrated"
            if child_conf is not None
            else "unavailable"
        ),
        "child_reason": child_reason,
        "restraint": restraint,
        "restraint_confidence": restraint_conf,
        "restraint_confidence_semantics": "unavailable: no validated restraint classifier",
        "restraint_reason": restraint_reason,
        "restraint_candidate": restraint_candidate,
        "child_presence": "YES" if child == "Yes" else ("NO" if child == "No" else "UNKNOWN"),
        "child_seat_presence": (
            "YES" if restraint == "Yes" else ("NO" if restraint == "No" else "UNKNOWN")
        ),
        "child_seat_candidate": specialist_child_seat_candidate,
        "child_candidate": bool(specialist_child or specialist_child_pending),
        "child_seat_type": "UNKNOWN",
        "child_seat_type_candidate": (
            "INFANT" if any(":infant seat" in item for item in specialist_seat)
            else "BOOSTER" if any(":booster seat" in item for item in specialist_seat)
            else "UNKNOWN"
        ),
        "child_seat_occupancy": "UNKNOWN",
        "child_seat_occupancy_reason": "restraint-object/person-overlap-not-validated",
        "belt": belt,
        "belt_reason": belt_reason,
        "belt_confidence": None,
        "belt_confidence_semantics": "unavailable: image cue scores are not calibrated",
        "belt_issue": belt_issue,
        "posture": "UNKNOWN",
        "posture_reason": "no-calibrated-posture-classifier",
        "dms": None,
        "occupant_category": category,
        "child_placement": placement,
        "pose": None,
        "updated_at": seat.updated_at.isoformat()
        if isinstance(seat.updated_at, datetime)
        else str(seat.updated_at),
        "frame_seq": frame_seq,
        "track_ids": [o.track_id for o in seat_occs],
    }


def build_seat_map(
    seats: list[SeatState],
    occupants: list[OccupantState],
    frame_seq: int = -1,
) -> list[dict[str, Any]]:
    # Unique seat IDs only — first occurrence wins, preventing double counts.
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for seat in seats:
        if seat.seat_id in seen:
            continue
        seen.add(seat.seat_id)
        rows.append(seat_attributes(seat, occupants, frame_seq))
    return sorted(rows, key=lambda r: r["seat_id"])


def live_counts(seat_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Counters derived from the latest seat map with unique seat IDs.

    Every seat lands in exactly one of occupied / empty / unresolved, so
    ``total == occupied + empty + unknown`` always holds and no seat is
    ever double-counted. Belt fastening is never verified camera-only, so
    those keys stay 0 by contract (see belt_detected / belt_not_detected
    for strap visibility instead).
    """
    total = len(seat_rows)
    empty = sum(1 for r in seat_rows if r["presence"] == "EMPTY_CANDIDATE")
    occupied = sum(1 for r in seat_rows if r["presence"] in ("OCCUPIED", "LIKELY_OCCUPIED"))
    children = sum(1 for r in seat_rows if r["child"] == "Yes")
    possible_children = sum(
        1 for r in seat_rows if r.get("child_candidate") is True and r["child"] != "Yes"
    )
    restraints = sum(1 for r in seat_rows if r.get("restraint_candidate") is True)
    unknown = sum(1 for r in seat_rows if r["presence"] in ("UNKNOWN", "STALE"))
    adults = sum(1 for r in seat_rows if r.get("occupant_category") == "adult")
    infants = sum(1 for r in seat_rows if r.get("occupant_category") == "infant")
    restraint_occupancy_unknown = sum(
        1 for r in seat_rows if r.get("restraint_candidate") is True
    )
    belt_detected = sum(1 for r in seat_rows if r.get("belt") == "Detected")
    belt_not_detected = sum(1 for r in seat_rows if r.get("belt") == "Not detected")
    belt_unknown = sum(1 for r in seat_rows if r.get("belt", "Unknown") == "Unknown")
    belt_issue = sum(1 for r in seat_rows if r.get("belt_issue") is True)
    # Belt camera-only is always Unknown by contract — surfaced, not hidden.
    return {
        "total": total,
        "empty": empty,
        "empty_candidates": empty,
        "occupied": occupied,
        "children": children,
        "possible_children": possible_children,
        "restraints": restraints,
        "restraint_occupied": 0,
        "restraint_empty": 0,
        "restraint_occupancy_unknown": restraint_occupancy_unknown,
        "adults": adults,
        "infants": infants,
        "belt_fastened": 0,
        "belt_unfastened": 0,
        "belt_detected": belt_detected,
        "belt_not_detected": belt_not_detected,
        "belt_unknown": belt_unknown,
        "belt_issue": belt_issue,
        "unknown": unknown,
        "unresolved": unknown,
        "determined": total - unknown,
    }


def seat_display_color(row: dict[str, Any]) -> str:
    """Map state precedence for the SVG map.

    Base fill is occupancy (blue wins for child-occupied, yellow for the
    transitional LIKELY state, gray for occluded/unobservable views).
    Restraint (purple ring) and belt-issue (orange ring) stack as separate
    indicators, so no condition hides another.
    """
    presence = row.get("presence", "UNKNOWN")
    if row.get("visibility") in ("OCCLUDED", "CAMERA_UNHEALTHY"):
        return "#9ca3af"
    if presence in ("OCCUPIED", "LIKELY_OCCUPIED"):
        if row.get("child") == "Yes":
            return "#3b82f6"  # blue wins for child-occupied (legend)
        if presence == "LIKELY_OCCUPIED":
            return "#eab308"  # yellow: transitional / low confidence
        return "#ef4444"
    if presence == "EMPTY_CANDIDATE":
        return "#22c55e"
    if presence == "STALE":
        return "#f59e0b"
    return "#9ca3af"


def analytics_summary(
    seat_rows: list[dict[str, Any]], counts: dict[str, int]
) -> dict[str, Any]:
    """Dashboard analytics: percentages plus capped diagnostic notes.

    Notes name the exact seats behind every Unknown belt, unresolved
    occupancy, or missing-evidence case — the "meaningful diagnostics"
    for unreliable classifications. Deterministic row order, max 5 notes.
    """

    def _pct(part: int, whole: int) -> float:
        return round(100.0 * part / whole, 1) if whole else 0.0

    total = counts.get("total", 0)
    determined = counts.get("determined", total)
    percentages = {
        "occupied": _pct(counts.get("occupied", 0), total),
        "empty": _pct(counts.get("empty", 0), total),
        "empty_candidate": _pct(counts.get("empty_candidates", counts.get("empty", 0)), total),
        "unresolved": _pct(counts.get("unresolved", 0), total),
        "determined": _pct(determined, total),
        "children_of_occupied": _pct(counts.get("children", 0), counts.get("occupied", 0)),
        "belt_determined": _pct(
            counts.get("belt_detected", 0) + counts.get("belt_not_detected", 0), total
        ),
    }
    notes: list[str] = []
    for row in seat_rows:
        if len(notes) >= 5:
            break
        sid = row.get("seat_id", "?")
        if row.get("presence") in ("UNKNOWN", "STALE"):
            reasons = "; ".join(row.get("reason_codes", [])[-2:]) or "weak evidence"
            notes.append(f"{sid}: occupancy unresolved ({reasons})")
        elif row.get("belt") == "Unknown" and row.get("presence") in (
            "OCCUPIED",
            "LIKELY_OCCUPIED",
        ):
            notes.append(f"{sid}: belt indeterminate ({row.get('belt_reason', '')[:80]})")
    return {"percentages": percentages, "notes": notes}


def annotate_detections(
    detections: list[Detection],
    layout: SeatLayout,
    seat_rows: list[dict[str, Any]],
    ambiguity_margin: float = 0.15,
) -> list[tuple[Detection, str | None, str | None]]:
    """Attach (seat_id, tag) to each detection for overlay labels.

    Persons in a child-candidate seat get tag ``"child candidate"`` — the
    detector label stays ``person``
    (honest: COCO has no child class); the tag is seat-state evidence
    with its own reason on the seat row. Ambiguous boxes get no seat
    (never force a wrong seat). Order follows the input list.
    """
    if not detections or not layout.seats:
        return [(d, None, None) for d in detections]
    rows = {r["seat_id"]: r for r in seat_rows}
    pairs = [(f"det{i}", d.bbox) for i, d in enumerate(detections)]
    out: list[tuple[Detection, str | None, str | None]] = []
    for assoc in associate_track_first(pairs, layout, ambiguity_margin):
        det = detections[int(assoc.track_id[3:])]
        seat_id = assoc.seat_id if not assoc.ambiguous else None
        tag: str | None = None
        if seat_id is not None and det.class_label == "person":
            row = rows.get(seat_id)
            if row is not None and row.get("child") == "Yes":
                tag = "child candidate"
        out.append((det, seat_id, tag))
    return out
