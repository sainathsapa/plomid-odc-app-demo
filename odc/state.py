"""Temporal seat-state engine (Phase 6) — evidence over time, never flicker.

Core safety rule: EMPTY_CANDIDATE requires positive empty evidence (healthy
camera + visible seat + sustained zero person evidence). Missed detections,
occlusion, and stale data yield UNKNOWN/STALE — never EMPTY, never OCCUPIED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from . import contracts
from .domain import (
    AttributeState,
    BBox,
    CameraHealth,
    Detection,
    EventType,
    HealthStatus,
    OccupantState,
    PresenceState,
    SeatGeometry,
    SeatState,
    Severity,
    SizeAppearance,
    StateEvent,
    VisibilityState,
)
from .logging import get_logger
from .seats import (
    SeatLayout,
    _bbox_overlap_area,
    associate_track_first,
    candidate_seats_for_bbox,
)

log = get_logger("state")

# Runner-up seats attach the same object box when it substantially overlaps
# them: in close-up geometry one box straddles two seats (measured F1
# suitcase 0.30 of its area over the baby seat while centering on mom's).
# Without this the evidence lands on exactly one seat — the wrong one.
MULTI_ATTACH_OVERLAP_FRAC = 0.25


def _overlap_frac(bbox: BBox, seat: SeatGeometry) -> float:
    """Box area fraction covered by the seat's bounding rectangle."""
    area = max((bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min), 1e-9)
    return _bbox_overlap_area(bbox, seat.polygon) / area


def _bbox_iou(left: BBox, right: BBox) -> float:
    """Return IoU for specialist-to-person association."""
    ix0 = max(left.x_min, right.x_min)
    iy0 = max(left.y_min, right.y_min)
    ix1 = min(left.x_max, right.x_max)
    iy1 = min(left.y_max, right.y_max)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, left.x_max - left.x_min) * max(0.0, left.y_max - left.y_min)
    right_area = max(0.0, right.x_max - right.x_min) * max(0.0, right.y_max - right.y_min)
    return inter / max(left_area + right_area - inter, 1e-9)


def _best_person_track(
    candidate: BBox,
    occupants: list[OccupantState],
    min_iou: float,
) -> tuple[str | None, float]:
    """Match an age candidate to one person track, preferring box overlap.

    Center containment allows head/body candidates that are smaller than the
    generic person box. When boxes overlap multiple tracks, IoU chooses the
    most specific track instead of accepting the first matching occupant.
    """
    center_x = (candidate.x_min + candidate.x_max) / 2.0
    center_y = (candidate.y_min + candidate.y_max) / 2.0
    matches: list[tuple[int, float, str]] = []
    for occupant in occupants:
        person_box = occupant.last_bbox
        if person_box is None:
            continue
        center_inside = (
            person_box.x_min <= center_x <= person_box.x_max
            and person_box.y_min <= center_y <= person_box.y_max
        )
        overlap = _bbox_iou(candidate, person_box)
        if center_inside or overlap >= min_iou:
            matches.append((int(center_inside), overlap, occupant.track_id))
    if not matches:
        return None, 0.0
    _center_priority, overlap, track_id = max(matches)
    return track_id, overlap


@dataclass(frozen=True)
class StateRules:
    """Deterministic temporal rules, versioned. Thresholds live here or in
    config — never scattered as constants. Version bump = revalidation."""

    version: str = "state-v1"
    min_score: float = 0.3
    min_positive_frames: int = 3
    occupy_dwell_s: float = 1.0
    confirm_extra_s: float = 2.0
    release_misses_likely: int = 5
    release_misses_unknown: int = 15
    empty_dwell_frames: int = 25
    stale_after_s: float = 5.0
    cooldown_s: float = 10.0
    ambiguity_margin: float = 0.15
    size_window: int = 5
    size_small_max: float = 0.5
    size_large_min: float = 1.1
    size_min_votes: int = 3
    # Grounding-DINO child labels are open-vocabulary similarities, not
    # probabilities. Low-score child candidates are hard negatives in cabin
    # imagery and must not become occupant identity evidence.
    specialist_child_min_score: float = 0.30
    specialist_child_direct_score: float = 0.35
    specialist_restraint_max_box_area: float = 0.65
    specialist_child_confirmation_score: float = 0.351


@dataclass
class _SeatBelief:
    presence: PresenceState = PresenceState.UNKNOWN
    visibility: VisibilityState = VisibilityState.NOT_EVALUATED
    confidence: float = 0.0
    positive_runs: int = 0
    negative_runs: int = 0
    misses: int = 0
    first_positive_at: datetime | None = None
    last_evidence_at: datetime | None = None
    last_track_id: str | None = None
    reasons: list[str] = field(default_factory=list)


def _now() -> datetime:
    return datetime.now(UTC)


LUGGAGE_LABELS = frozenset({"backpack", "handbag", "suitcase", "laptop", "book"})
# Child-restraint proxy: COCO has no child-seat class. Sustained chair/couch
# evidence in a seat ROI is the honest CANDIDATE signal (never a certified
# install check). Measured on cabin footage (F1/C1/C2): the detector also
# labels child-seat shells and carriers as suitcase/handbag/backpack, so
# LARGE boxes of those labels (seat-sized, not purses) count as proxies too.
# Tracked with the same temporal gate as luggage, kept as a separate
# attribute downstream so occupancy is never overwritten.
RESTRAINT_LABELS = frozenset({"chair", "couch"})
RESTRAINT_SOFT_LABELS = frozenset({"suitcase", "handbag", "backpack"})
# Minimum absolute box area for a soft-proxy box to count as restraint
# evidence: a real purse (~0.01-0.03) must not become a "child seat", while
# seat shells/carriers (F1 suitcase 0.16, C1 handbag 0.10) do.
RESTRAINT_SOFT_MIN_AREA = 0.05
OBJECT_EVIDENCE_FRAMES = 3
# Anti-fragment guard for apparent-size voting: tiny boxes (<8% of frame,
# e.g. a 0.02-0.07 face fragment in a close-up) carry no reliable
# body-size information — a close adult face must not vote "small child".
# Such frames cast no vote (history kept, never poisoned).
SIZE_MIN_ABS_AREA = 0.08


def _seat_bbox_area(seat: SeatGeometry) -> float:
    xs = [p[0] for p in seat.polygon]
    ys = [p[1] for p in seat.polygon]
    return max((max(xs) - min(xs)) * (max(ys) - min(ys)), 1e-9)


def _classify_size(ratios: list[float], rules: StateRules) -> tuple[SizeAppearance, float]:
    """Seat-relative size vote over a temporal window. Provisional by design:
    single-frame geometry cannot separate babies from adults across camera
    geometries (measured); the window + thresholds only suppress flicker."""
    window = ratios[-rules.size_window :]
    if len(window) < rules.size_min_votes:
        return SizeAppearance.UNKNOWN, 0.0
    small = sum(1 for r in window if r < rules.size_small_max)
    large = sum(1 for r in window if r > rules.size_large_min)
    if small >= rules.size_min_votes:
        return SizeAppearance.APPARENT_SMALL_OCCUPANT, 0.0
    if large >= rules.size_min_votes:
        return SizeAppearance.APPARENT_LARGE_OCCUPANT, 0.0
    return SizeAppearance.UNKNOWN, 0.0


class SeatStateEngine(contracts.StateEngine):
    def __init__(
        self,
        session_id: str,
        layout: SeatLayout,
        rules: StateRules | None = None,
        config_version: str = "unset",
        model_set_version: str = "unset",
    ) -> None:
        self._session = session_id
        self._layout = layout
        self._rules = rules or StateRules()
        self._config_version = config_version
        self._model_version = model_set_version
        self._beliefs: dict[str, _SeatBelief] = {
            seat.seat_id: _SeatBelief() for seat in layout.seats
        }
        self._camera_ok = True
        self._observation_good = True
        self._size_hist: dict[str, list[float]] = {}
        self._relative_small_hist: dict[str, list[bool]] = {}
        self._relative_not_small_runs: dict[str, int] = {}
        self._seat_size_hist: dict[str, list[tuple[int, float]]] = {}
        self._relative_candidate_seats: set[str] = set()
        self._specialist_evidence: dict[str, list[dict[str, Any]]] = {}
        self.last_specialist_decisions: list[dict[str, Any]] = []
        self._obj_runs: dict[tuple[str, str], int] = {}
        self._obj_miss: dict[tuple[str, str], int] = {}
        self._proxy_ok: dict[tuple[str, str], bool] = {}
        self._last_frame_ts: datetime | None = None
        self._last_seq = -1
        self._cooldown_until: dict[tuple[str, str], datetime] = {}
        self._event_seq = 0
        self.out_of_order_dropped = 0

    def set_camera_health(self, health: CameraHealth) -> None:
        self._camera_ok = health.status == HealthStatus.HEALTHY

    def rekey_layout(self, layout: SeatLayout) -> dict[str, int]:
        """Adopt edited geometry mid-session without losing temporal state.

        Beliefs for carried-over seat ids are kept (no dwell reset, no
        flicker); new ids start UNKNOWN; removed ids are dropped. Returns
        {"carried": n, "reset": m} so callers can report what happened.
        Assignment replaces the dict atomically for the streaming loop.
        """
        carried = sum(1 for s in layout.seats if s.seat_id in self._beliefs)
        self._beliefs = {
            s.seat_id: self._beliefs.get(s.seat_id, _SeatBelief()) for s in layout.seats
        }
        seat_ids = {s.seat_id for s in layout.seats}
        self._seat_size_hist = {
            sid: hist for sid, hist in self._seat_size_hist.items() if sid in seat_ids
        }
        self._relative_small_hist = {
            sid: hist for sid, hist in self._relative_small_hist.items() if sid in seat_ids
        }
        self._relative_not_small_runs = {
            sid: runs for sid, runs in self._relative_not_small_runs.items() if sid in seat_ids
        }
        self._relative_candidate_seats.intersection_update(seat_ids)
        self._specialist_evidence = {
            sid: entries for sid, entries in self._specialist_evidence.items() if sid in seat_ids
        }
        self._layout = layout
        return {"carried": carried, "reset": len(layout.seats) - carried}

    def set_frame_quality(
        self, dark: bool = False, blurry: bool = False, frozen: bool = False
    ) -> None:
        """Advisory capture quality gates EMPTY evidence (never truth)."""
        self._observation_good = not (dark or blurry or frozen)

    def _new_event(
        self,
        seat_id: str,
        prior: PresenceState,
        new: PresenceState,
        occurred_at: datetime,
        reasons: list[str],
        track_id: str | None,
    ) -> StateEvent | None:
        key = (seat_id, new.value)
        recorded = _now()
        # Cooldown compares wall-clock against the last RECORDED event only.
        # Default is epoch (never suppress the first transition); frame time
        # must never gate against wall time (replay frames can postdate it).
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        if self._cooldown_until.get(key, epoch) > recorded:
            return None
        from datetime import timedelta

        self._cooldown_until[key] = recorded + timedelta(seconds=self._rules.cooldown_s)
        self._event_seq += 1
        return StateEvent(
            event_id=f"{self._session}-{seat_id}-{self._event_seq}",
            session_id=self._session,
            event_type=EventType.SEAT_PRESENCE_CHANGED,
            severity=Severity.INFO,
            seat_id=seat_id,
            track_id=track_id,
            occurred_at=occurred_at,
            recorded_at=recorded,
            prior_state=prior.value,
            new_state=new.value,
            reason_codes=list(reasons),
            evidence_summary=";".join(reasons),
            config_version=self._config_version,
            model_set_version=self._model_version,
        )

    def update(
        self,
        frame: contracts.Frame,
        occupants: list[OccupantState],
        seats: list[SeatGeometry],
        objects: list[Detection] | None = None,
        specialist_detections: list[Detection] | None = None,
    ) -> contracts.EngineOutput:
        meta = frame.meta
        if meta.seq < self._last_seq:
            # Source restart (incl. replay loop): fresh evidence stream. The
            # new stream's clock starts over, so timestamp comparison resets
            # too — otherwise looped replays read as out-of-order forever.
            log.info("source restart detected: resetting dwell counters")
            for belief in self._beliefs.values():
                belief.positive_runs = 0
                belief.negative_runs = 0
                belief.misses = 0
            self._last_frame_ts = None
            self._last_seq = -1
            self._relative_small_hist.clear()
            self._relative_not_small_runs.clear()
            self._seat_size_hist.clear()
            self._relative_candidate_seats.clear()
            self._specialist_evidence.clear()
        if self._last_frame_ts is not None and meta.captured_at < self._last_frame_ts:
            self.out_of_order_dropped += 1
            return contracts.EngineOutput(seats=[], occupants=[], events=[])
        self._last_frame_ts = meta.captured_at
        self._last_seq = meta.seq

        pairs = [
            (occ.track_id, occ.last_bbox)
            for occ in occupants
            if occ.last_bbox is not None and (occ.last_score or 0) >= self._rules.min_score
        ]
        assoc = {
            a.track_id: a
            for a in associate_track_first(pairs, self._layout, self._rules.ambiguity_margin)
        }
        ambiguous_seats: set[str] = set()
        occupants_by_track = {occupant.track_id: occupant for occupant in occupants}
        for track_id, association in assoc.items():
            if not association.ambiguous:
                continue
            occupant = occupants_by_track.get(track_id)
            if occupant is not None and occupant.last_bbox is not None:
                ambiguous_seats.update(
                    candidate_seats_for_bbox(
                        occupant.last_bbox, self._layout, self._rules.ambiguity_margin
                    )
                )
        seat_tracks: dict[str, list[str]] = {s.seat_id: [] for s in self._layout.seats}
        for track_id, a in assoc.items():
            if a.seat_id is not None and not a.ambiguous:
                seat_tracks[a.seat_id].append(track_id)

        out_seats: list[SeatState] = []
        out_events: list[StateEvent] = []
        out_occupants: list[OccupantState] = []
        seat_objects = self._update_object_evidence(objects)
        specialist_by_seat = self._update_specialist_evidence(
            specialist_detections, occupants, meta.seq, seat_objects
        )
        for seat in self._layout.seats:
            belief = self._beliefs[seat.seat_id]
            event = self._step_seat(
                seat.seat_id,
                belief,
                seat_tracks[seat.seat_id],
                seat.seat_id in ambiguous_seats,
                meta.captured_at,
            )
            if event is not None:
                out_events.append(event)
            reasons = list(belief.reasons)
            obj_info = seat_objects.get(seat.seat_id)
            object_label = obj_info[0] if obj_info is not None else None
            if obj_info is not None:
                reasons.append(f"object-evidence:{obj_info[0]}")
                if obj_info[1]:
                    # Proxy verdict travels in reasons (persisted to SQL),
                    # so attribute mapping needs no duplicate label lists.
                    reasons.append(f"restraint-proxy:{obj_info[0]}")
            reasons.extend(specialist_by_seat.get(seat.seat_id, []))
            out_seats.append(
                SeatState(
                    session_id=self._session,
                    seat_id=seat.seat_id,
                    presence=belief.presence,
                    confidence=belief.confidence,
                    visibility=belief.visibility,
                    reason_codes=reasons,
                    object_label=object_label,
                    updated_at=meta.captured_at,
                    last_positive_at=belief.first_positive_at,
                    model_set_version=self._model_version,
                    config_version=self._config_version,
                )
            )
        for occ in occupants:
            assoc_entry = assoc.get(occ.track_id)
            seat_id = (
                assoc_entry.seat_id
                if assoc_entry is not None and not assoc_entry.ambiguous
                else None
            )
            conf = min(0.55 + 0.05 * len(seat_tracks.get(seat_id or "", [])), 0.95)
            size_value, size_conf = self._track_size(occ, seat_id, seats)
            out_occupants.append(
                OccupantState(
                    session_id=self._session,
                    track_id=occ.track_id,
                    seat_id=seat_id,
                    last_bbox=occ.last_bbox,
                    last_score=occ.last_score,
                    presence=AttributeState(
                        value=self._beliefs[seat_id].presence if seat_id else PresenceState.UNKNOWN,
                        confidence=conf if seat_id else 0.0,
                        observed_at=meta.captured_at,
                        reason="seat-evidence",
                        model_set_version=self._model_version,
                        config_version=self._config_version,
                    ),
                    apparent_size=AttributeState(
                        value=size_value,
                        confidence=size_conf,
                        observed_at=meta.captured_at,
                        reason="seat-relative-bbox",
                        model_set_version=self._model_version,
                        config_version=self._config_version,
                    ),
                    updated_at=meta.captured_at,
                )
            )
        # Compare persistent seat-relative person sizes, not only boxes from
        # this frame. Occupants are intermittently missed, so ratios from
        # recently observed seats remain available for peer comparison.
        relative_candidates_before = set(self._relative_candidate_seats)
        by_seat: dict[str, tuple[str, float]] = {}
        seat_by_id = {seat.seat_id: seat for seat in seats}
        for occ in out_occupants:
            if occ.seat_id is None or occ.last_bbox is None:
                continue
            seat = seat_by_id.get(occ.seat_id)
            if seat is None:
                continue
            box = occ.last_bbox
            area = (box.x_max - box.x_min) * (box.y_max - box.y_min)
            if area < SIZE_MIN_ABS_AREA:
                continue
            ratio = area / _seat_bbox_area(seat)
            prior = by_seat.get(occ.seat_id)
            if prior is None or ratio > prior[1]:
                by_seat[occ.seat_id] = (occ.track_id, ratio)
        for seat_id, (_track_id, ratio) in by_seat.items():
            history = self._seat_size_hist.setdefault(seat_id, [])
            history.append((meta.seq, ratio))
            del history[:-self._rules.size_window]
        # Auto layouts can split a visible person into a separate hotspot, or
        # miss them on later frames. Keep each seat's recent ratio for 60
        # source frames (about 2.4 s at 25 fps), then abstain until it is seen
        # again. Comparisons only vote for a seat observed on the current
        # frame, so reusing peers does not create extra votes for a missing
        # child track.
        recent_ratios = {
            sid: [value for seq, value in history if meta.seq - seq <= 60]
            for sid, history in self._seat_size_hist.items()
        }
        recent_ratios = {sid: values for sid, values in recent_ratios.items() if values}
        if len(recent_ratios) >= 3:
            for seat_id in by_seat:
                ratio_values = recent_ratios.get(seat_id, [])
                peers = sorted(
                    values[-1]
                    for other_id, values in recent_ratios.items()
                    if other_id != seat_id
                )
                if not ratio_values or len(peers) < 2:
                    continue
                votes = self._relative_small_hist.setdefault(seat_id, [])
                # Relative image size is retained as diagnostic metadata only.
                # Perspective and crop geometry are not sufficient evidence
                # for a child classification, so it must never create a child
                # candidate event or seat state.
                is_small = False
                votes.append(is_small)
                del votes[:-self._rules.size_window]
                enough_votes = len(votes) >= self._rules.size_min_votes
                if enough_votes and sum(votes) >= self._rules.size_min_votes:
                    self._relative_candidate_seats.add(seat_id)
                if is_small:
                    self._relative_not_small_runs[seat_id] = 0
                else:
                    self._relative_not_small_runs[seat_id] = (
                        self._relative_not_small_runs.get(seat_id, 0) + 1
                    )
                    if self._relative_not_small_runs[seat_id] >= 6:
                        self._relative_candidate_seats.discard(seat_id)
        presence_by_seat = {state.seat_id: state.presence for state in out_seats}
        for seat_id, presence in presence_by_seat.items():
            if presence == PresenceState.EMPTY_CANDIDATE:
                self._relative_candidate_seats.discard(seat_id)
                self._relative_small_hist.pop(seat_id, None)
                self._relative_not_small_runs.pop(seat_id, None)
        for seat_id in sorted(self._relative_candidate_seats - relative_candidates_before):
            self._event_seq += 1
            out_events.append(
                StateEvent(
                    event_id=f"{self._session}-{seat_id}-child-{self._event_seq}",
                    session_id=self._session,
                    event_type=EventType.CHILD_CANDIDATE_DETECTED,
                    severity=Severity.INFO,
                    seat_id=seat_id,
                    occurred_at=meta.captured_at,
                    recorded_at=_now(),
                    prior_state="UNKNOWN",
                    new_state="CHILD_CANDIDATE",
                    reason_codes=["child-candidate:relative-cabin-size"],
                    evidence_summary="relative cabin-size candidate; not an age classification",
                    config_version=self._config_version,
                    model_set_version=self._model_version,
                )
            )
        for seat_id in sorted(relative_candidates_before - self._relative_candidate_seats):
            self._event_seq += 1
            out_events.append(
                StateEvent(
                    event_id=f"{self._session}-{seat_id}-child-{self._event_seq}",
                    session_id=self._session,
                    event_type=EventType.CHILD_CANDIDATE_CLEARED,
                    severity=Severity.INFO,
                    seat_id=seat_id,
                    occurred_at=meta.captured_at,
                    recorded_at=_now(),
                    prior_state="CHILD_CANDIDATE",
                    new_state="UNKNOWN",
                    reason_codes=["child-candidate-cleared"],
                    evidence_summary=(
                        "child candidate cleared by empty-seat or contradictory size evidence"
                    ),
                    config_version=self._config_version,
                    model_set_version=self._model_version,
                )
            )
        if self._relative_candidate_seats:
            out_seats = [
                seat_state.model_copy(
                    update={
                        "reason_codes": [
                            *[code for code in seat_state.reason_codes
                              if code != "child-candidate:relative-cabin-size"],
                            "child-candidate:relative-cabin-size",
                        ]
                    }
                )
                if seat_state.seat_id in self._relative_candidate_seats
                else seat_state
                for seat_state in out_seats
            ]
        return contracts.EngineOutput(seats=out_seats, occupants=out_occupants, events=out_events)

    def _update_specialist_evidence(
        self,
        detections: list[Detection] | None,
        occupants: list[OccupantState],
        frame_seq: int,
        seat_objects: dict[str, tuple[str, bool]] | None = None,
    ) -> dict[str, list[str]]:
        """Associate specialist candidates with stable seats and expire stale evidence.

        Raises:
            ValueError: Never raised for malformed candidate output; invalid
                candidates are ignored by the domain validation boundary.

        Side Effects:
            Updates session-local specialist evidence keyed by seat identity.

        Operational Boundaries:
            O(n*m) for bounded candidate count n and seat count m. Evidence
            expires after 36 source frames, preventing stale replay results
            from replacing newer observations.
        """
        self.last_specialist_decisions = []
        person_pairs = [
            (occupant.track_id, occupant.last_bbox)
            for occupant in occupants
            if occupant.last_bbox is not None
            and (occupant.last_score or 0.0) >= self._rules.min_score
        ]
        occupant_associations = associate_track_first(
            person_pairs, self._layout, self._rules.ambiguity_margin
        )
        occupied_seats = {
            item.seat_id
            for item in occupant_associations
            if item.seat_id is not None and not item.ambiguous
        }
        occupants_by_seat: dict[str, list[OccupantState]] = {}
        current_track_seat: dict[str, str] = {}
        for occupant, association in zip(occupants, occupant_associations, strict=True):
            if association.seat_id is not None and not association.ambiguous:
                occupants_by_seat.setdefault(association.seat_id, []).append(occupant)
                current_track_seat[occupant.track_id] = association.seat_id
        if detections:
            # A phrase such as "child" is not accepted on an empty seat. The
            # candidate must map to the same configured seat as a tracked
            # occupant. This tolerates a child/head box that does not overlap
            # the full person box while rejecting unrelated long objects.
            usable: list[Detection] = []
            usable_track_ids: list[str | None] = []
            for detection in detections:
                person_track_id: str | None = None
                decision: dict[str, Any] = {
                    "label": detection.class_label,
                    "score": detection.score,
                    "bbox": [
                        detection.bbox.x_min,
                        detection.bbox.y_min,
                        detection.bbox.x_max,
                        detection.bbox.y_max,
                    ],
                    "source_frame_seq": detection.frame_seq,
                    "disposition": "accepted_non_age_candidate",
                }
                if detection.class_label in {"child", "infant"}:
                    if detection.score < self._rules.specialist_child_min_score:
                        decision["disposition"] = "rejected_below_minimum_score"
                        self.last_specialist_decisions.append(decision)
                        continue
                    candidate_association = associate_track_first(
                        [("specialist", detection.bbox)],
                        self._layout,
                        self._rules.ambiguity_margin,
                    )
                    if (
                        not candidate_association
                        or candidate_association[0].ambiguous
                        or candidate_association[0].seat_id not in occupied_seats
                    ):
                        decision["disposition"] = "rejected_seat_or_occupancy_association"
                        self.last_specialist_decisions.append(decision)
                        continue
                    candidate_seat = candidate_association[0].seat_id
                    decision["seat_id"] = candidate_seat
                    same_seat_restraint = False
                    for other in detections:
                        if other.class_label not in {
                            "child seat",
                            "infant seat",
                            "booster seat",
                        }:
                            continue
                        other_area = max(
                            0.0,
                            other.bbox.x_max - other.bbox.x_min,
                        ) * max(0.0, other.bbox.y_max - other.bbox.y_min)
                        if other_area > self._rules.specialist_restraint_max_box_area:
                            continue
                        other_association = associate_track_first(
                            [("restraint", other.bbox)],
                            self._layout,
                            self._rules.ambiguity_margin,
                        )
                        if (
                            other_association
                            and not other_association[0].ambiguous
                            and other_association[0].seat_id == candidate_seat
                            and _bbox_iou(detection.bbox, other.bbox) >= 0.02
                        ):
                            same_seat_restraint = True
                            break
                    direct_evidence = detection.score >= self._rules.specialist_child_direct_score
                    restraint_context = bool(
                        seat_objects
                        and candidate_seat in seat_objects
                        and seat_objects[candidate_seat][1]
                    )
                    temporally_testable = (
                        detection.score < self._rules.specialist_child_confirmation_score
                        and restraint_context
                    )
                    if not (direct_evidence or same_seat_restraint or temporally_testable):
                        decision["disposition"] = "rejected_evidence_gate"
                        self.last_specialist_decisions.append(decision)
                        continue
                    person_track_id, person_iou = _best_person_track(
                        detection.bbox,
                        occupants_by_seat.get(candidate_seat, []),
                        0.02,
                    )
                    if person_track_id is None:
                        decision["disposition"] = "rejected_person_track_association"
                        self.last_specialist_decisions.append(decision)
                        continue
                    decision["person_track_id"] = person_track_id
                    decision["person_iou"] = person_iou
                elif detection.class_label == "adult":
                    adult_association = associate_track_first(
                        [("specialist", detection.bbox)],
                        self._layout,
                        self._rules.ambiguity_margin,
                    )
                    if (
                        not adult_association
                        or adult_association[0].ambiguous
                        or adult_association[0].seat_id not in occupied_seats
                    ):
                        decision["disposition"] = "rejected_adult_seat_or_occupancy_association"
                        self.last_specialist_decisions.append(decision)
                        continue
                    adult_seat = adult_association[0].seat_id
                    decision["seat_id"] = adult_seat
                    person_track_id, person_iou = _best_person_track(
                        detection.bbox,
                        occupants_by_seat.get(adult_seat, []),
                        0.10,
                    )
                    if person_track_id is None:
                        decision["disposition"] = "rejected_adult_person_track_association"
                        self.last_specialist_decisions.append(decision)
                        continue
                    decision["person_track_id"] = person_track_id
                    decision["person_iou"] = person_iou
                if detection.class_label in {"child", "infant", "adult"}:
                    decision["disposition"] = "accepted_associated_candidate"
                    self.last_specialist_decisions.append(decision)
                usable.append(detection)
                usable_track_ids.append(person_track_id)
            pairs = [(f"spec{i}", detection.bbox) for i, detection in enumerate(usable)]
            for association in associate_track_first(
                pairs, self._layout, self._rules.ambiguity_margin
            ):
                if association.seat_id is None or association.ambiguous:
                    continue
                index = int(association.track_id.removeprefix("spec"))
                detection = usable[index]
                person_track_id = usable_track_ids[index]
                if detection.class_label in {"child seat", "infant seat", "booster seat"}:
                    kind = "child-seat"
                elif detection.class_label in {"child", "infant"}:
                    kind = "child-occupant"
                elif detection.class_label == "adult":
                    kind = "adult-occupant"
                else:
                    continue
                entries = self._specialist_evidence.setdefault(association.seat_id, [])
                prior = next(
                    (
                        item
                        for item in entries
                        if item["kind"] == kind
                        and item.get("track_id") == person_track_id
                    ),
                    None,
                )
                if prior is None:
                    entries.append(
                        {
                            "kind": kind,
                            "label": detection.class_label,
                            "track_id": person_track_id,
                            "score": detection.score,
                            "source_frame_seq": detection.frame_seq,
                            "accepted_frame_seq": frame_seq,
                            "hits": 1,
                        }
                    )
                else:
                    prior_hits = int(prior.get("hits", 1))
                    if int(prior.get("source_frame_seq", -1)) != detection.frame_seq:
                        prior_hits += 1
                    retained_score = max(float(prior.get("score", 0.0)), detection.score)
                    prior.update(
                        {
                            "label": detection.class_label,
                            "track_id": person_track_id,
                            "score": retained_score,
                            "source_frame_seq": detection.frame_seq,
                            "accepted_frame_seq": frame_seq,
                            "hits": prior_hits,
                        }
                    )
        out: dict[str, list[str]] = {}
        for seat_id, entries in list(self._specialist_evidence.items()):
            entries[:] = [
                evidence
                for evidence in entries
                if frame_seq - int(evidence.get("accepted_frame_seq", frame_seq)) <= 36
                and (
                    evidence["kind"] not in {"child-occupant", "adult-occupant"}
                    or current_track_seat.get(str(evidence.get("track_id"))) == seat_id
                )
            ]
            if not entries:
                del self._specialist_evidence[seat_id]
                continue
            for evidence in entries:
                if (
                    evidence["kind"] == "child-occupant"
                    and float(evidence.get("score", 0.0))
                    < self._rules.specialist_child_confirmation_score
                    and int(evidence.get("hits", 1)) < 2
                ):
                    matching_decision = next(
                        (
                            item
                            for item in self.last_specialist_decisions
                            if item.get("source_frame_seq")
                            == evidence.get("source_frame_seq")
                            and item.get("label") == evidence.get("label")
                            and item.get("person_track_id") == evidence.get("track_id")
                        ),
                        None,
                    )
                    if matching_decision is not None:
                        matching_decision["temporal_hits"] = int(evidence.get("hits", 1))
                        matching_decision["temporal_confirmed"] = False
                    pending = (
                        f"specialist-child-pending:{evidence['label']}"
                        f":track={evidence['track_id']}"
                        f":score={float(evidence['score']):.3f}"
                    )
                    out.setdefault(seat_id, []).append(pending)
                    continue
                matching_decision = next(
                    (
                        item
                        for item in self.last_specialist_decisions
                        if item.get("source_frame_seq")
                        == evidence.get("source_frame_seq")
                        and item.get("label") == evidence.get("label")
                        and item.get("person_track_id") == evidence.get("track_id")
                    ),
                    None,
                )
                if matching_decision is not None:
                    matching_decision["temporal_hits"] = int(evidence.get("hits", 1))
                    matching_decision["temporal_confirmed"] = True
                reason = f"specialist-{evidence['kind']}:{evidence['label']}"
                if evidence.get("track_id") is not None:
                    reason += f":track={evidence['track_id']}:score={float(evidence['score']):.3f}"
                out.setdefault(seat_id, []).append(reason)
                out[seat_id].append(f"specialist-score:{float(evidence['score']):.3f}")
                out[seat_id].append(
                    f"specialist-frame:{int(evidence.get('source_frame_seq', -1))}"
                )
        return out

    def _update_object_evidence(
        self, objects: list[Detection] | None
    ) -> dict[str, tuple[str, bool]]:
        """Temporal object→seat evidence. Returns seat_id → (label,
        is_restraint_proxy) for seats with sustained (>=3 frames)
        unambiguous object evidence. Covers luggage (backpack/handbag/...)
        and restraint proxies (chair/couch plus LARGE suitcase/handbag/
        backpack — COCO mislabels seat shells and carriers with those
        names; small purses stay luggage-only via RESTRAINT_SOFT_MIN_AREA).
        Restraint-first ordering per seat so a child safety seat is never
        hidden by a bag; the reason token is what persists to SQL. Resets
        when the object leaves; vehicle hallucinations (car/truck/...)
        stay ignored."""
        current: dict[tuple[str, str], int] = {}
        proxy_now: dict[tuple[str, str], bool] = {}
        if objects:
            # Pair names carry the OBJECTS index (f"obj{i}"), so the lookup
            # below uses objects[idx] — never a compacted side list.
            pairs: list[tuple[str, Any]] = []
            for i, d in enumerate(objects):
                in_object = d.class_label in LUGGAGE_LABELS or d.class_label in RESTRAINT_LABELS
                if not in_object or d.score < self._rules.min_score:
                    continue
                pairs.append((f"obj{i}", d.bbox))
            for entry in associate_track_first(pairs, self._layout, self._rules.ambiguity_margin):
                d = objects[int(entry.track_id[3:])]
                if entry.seat_id is not None and not entry.ambiguous:
                    key = (entry.seat_id, d.class_label)
                    current[key] = current.get(key, 0) + 1
                    # Proxy verdict is per-frame: hard labels always qualify;
                    # soft labels (suitcase/handbag/backpack) only when
                    # seat-sized, so a purse never becomes a "child seat".
                    area = (d.bbox.x_max - d.bbox.x_min) * (d.bbox.y_max - d.bbox.y_min)
                    proxy_now[key] = d.class_label in RESTRAINT_LABELS or (
                        d.class_label in RESTRAINT_SOFT_LABELS and area >= RESTRAINT_SOFT_MIN_AREA
                    )
                    # Runner-up attach: a box straddling two seats shares its
                    # evidence instead of landing on exactly one (possibly
                    # wrong) seat. Ambiguous near-ties still attach nowhere.
                    for seat in self._layout.seats:
                        if seat.seat_id == entry.seat_id:
                            continue
                        overlap = _overlap_frac(d.bbox, seat)
                        if overlap >= MULTI_ATTACH_OVERLAP_FRAC:
                            key2 = (seat.seat_id, d.class_label)
                            current[key2] = current.get(key2, 0) + 1
                            proxy_now[key2] = proxy_now[key]
        for key in list(self._obj_runs):
            if key in current:
                self._obj_runs[key] += 1
                self._obj_miss[key] = 0
            else:
                # One dropped frame (motion, flicker) must not erase
                # sustained evidence; two consecutive misses retire it.
                self._obj_miss[key] = self._obj_miss.get(key, 0) + 1
                if self._obj_miss[key] >= 2:
                    del self._obj_runs[key]
                    self._obj_miss.pop(key, None)
                    self._proxy_ok.pop(key, None)
        for key in current:
            self._obj_runs.setdefault(key, 1)
            self._obj_miss.setdefault(key, 0)
            self._proxy_ok[key] = proxy_now.get(key, False)
        # Restraint-first ordering: proxy candidates take precedence per
        # seat so they are never overwritten by luggage evidence. A soft
        # label counts as proxy only while its box stays seat-sized.
        confirmed: dict[str, tuple[str, bool]] = {}
        ordered = sorted(
            self._obj_runs.items(),
            key=lambda kv: (0 if self._proxy_ok.get(kv[0], False) else 1, kv[0][0]),
        )
        for (seat_id, label), runs in ordered:
            if runs >= OBJECT_EVIDENCE_FRAMES and seat_id not in confirmed:
                confirmed[seat_id] = (label, self._proxy_ok.get((seat_id, label), False))
        return confirmed

    def _track_size(
        self, occ: OccupantState, seat_id: str | None, seats: list[SeatGeometry]
    ) -> tuple[SizeAppearance, float]:
        if seat_id is None or occ.last_bbox is None:
            return SizeAppearance.UNKNOWN, 0.0
        seat = next((s for s in seats if s.seat_id == seat_id), None)
        if seat is None:
            return SizeAppearance.UNKNOWN, 0.0
        box = occ.last_bbox
        box_area = (box.x_max - box.x_min) * (box.y_max - box.y_min)
        hist = self._size_hist.setdefault(occ.track_id, [])
        if box_area < SIZE_MIN_ABS_AREA:
            # Face/box fragment in a close-up (e.g. a 0.02-0.07 adult face):
            # no reliable body-size information. Cast no vote this frame —
            # history is kept, never poisoned — and report existing history.
            del hist[: max(0, len(hist) - self._rules.size_window)]
            return _classify_size(hist, self._rules)
        ratio = box_area / _seat_bbox_area(seat)
        hist.append(ratio)
        del hist[: max(0, len(hist) - self._rules.size_window)]
        if len(self._size_hist) > 200:
            oldest = next(iter(self._size_hist))
            del self._size_hist[oldest]
        return _classify_size(hist, self._rules)

    def _step_seat(
        self,
        seat_id: str,
        belief: _SeatBelief,
        tracks: list[str],
        ambiguous_nearby: bool,
        now: datetime,
    ) -> StateEvent | None:
        rules = self._rules
        belief.last_evidence_at = now
        track_id = tracks[0] if tracks else None

        if not self._camera_ok:
            return self._transition(
                seat_id,
                belief,
                PresenceState.UNKNOWN,
                VisibilityState.CAMERA_UNHEALTHY,
                0.0,
                ["camera-unhealthy"],
                now,
                None,
            )

        if tracks:
            belief.positive_runs += 1
            belief.negative_runs = 0
            belief.misses = 0
            belief.last_track_id = track_id
            if belief.first_positive_at is None:
                belief.first_positive_at = now
            dwell = (now - belief.first_positive_at).total_seconds()
            if belief.presence in (
                PresenceState.UNKNOWN,
                PresenceState.EMPTY_CANDIDATE,
                PresenceState.STALE,
            ):
                if (
                    belief.positive_runs >= rules.min_positive_frames
                    and dwell >= rules.occupy_dwell_s
                ):
                    return self._transition(
                        seat_id,
                        belief,
                        PresenceState.LIKELY_OCCUPIED,
                        VisibilityState.VISIBLE,
                        min(0.55 + 0.05 * belief.positive_runs, 0.85),
                        ["dwell-met", f"frames={belief.positive_runs}"],
                        now,
                        track_id,
                    )
            elif belief.presence == PresenceState.LIKELY_OCCUPIED:
                if dwell >= rules.occupy_dwell_s + rules.confirm_extra_s:
                    return self._transition(
                        seat_id,
                        belief,
                        PresenceState.OCCUPIED,
                        VisibilityState.VISIBLE,
                        min(0.7 + 0.03 * belief.positive_runs, 0.95),
                        ["confirm-dwell-met"],
                        now,
                        track_id,
                    )
            belief.visibility = VisibilityState.VISIBLE
            return None

        # No person evidence for this seat. EMPTY needs a GOOD observation:
        # healthy camera, clear frame, nobody ambiguous nearby. Anything less
        # only ages the belief toward UNKNOWN — never toward EMPTY.
        belief.positive_runs = 0
        if ambiguous_nearby or not self._observation_good:
            belief.negative_runs = 0
            belief.misses += 1
        else:
            belief.negative_runs += 1
            belief.misses += 1
        if belief.presence == PresenceState.OCCUPIED:
            if belief.misses >= rules.release_misses_unknown:
                return self._transition(
                    seat_id,
                    belief,
                    PresenceState.UNKNOWN,
                    VisibilityState.VISIBLE,
                    0.0,
                    ["release-misses"],
                    now,
                    None,
                )
            if belief.misses >= rules.release_misses_likely:
                return self._transition(
                    seat_id,
                    belief,
                    PresenceState.LIKELY_OCCUPIED,
                    VisibilityState.VISIBLE,
                    0.5,
                    ["release-misses"],
                    now,
                    None,
                )
        elif belief.presence == PresenceState.LIKELY_OCCUPIED:
            if belief.misses >= rules.release_misses_unknown:
                return self._transition(
                    seat_id,
                    belief,
                    PresenceState.UNKNOWN,
                    VisibilityState.VISIBLE,
                    0.0,
                    ["release-misses"],
                    now,
                    None,
                )
        elif belief.presence in (PresenceState.UNKNOWN, PresenceState.STALE):
            if belief.negative_runs >= rules.empty_dwell_frames and not ambiguous_nearby:
                return self._transition(
                    seat_id,
                    belief,
                    PresenceState.EMPTY_CANDIDATE,
                    VisibilityState.VISIBLE,
                    0.6,
                    ["empty-dwell-met"],
                    now,
                    None,
                )
        belief.visibility = VisibilityState.VISIBLE
        return None

    def _transition(
        self,
        seat_id: str,
        belief: _SeatBelief,
        new: PresenceState,
        visibility: VisibilityState,
        confidence: float,
        reasons: list[str],
        now: datetime,
        track_id: str | None,
    ) -> StateEvent | None:
        prior = belief.presence
        if prior == new:
            return None
        belief.presence = new
        belief.visibility = visibility
        belief.confidence = confidence
        belief.reasons = list(reasons)
        if new in (PresenceState.UNKNOWN, PresenceState.EMPTY_CANDIDATE, PresenceState.STALE):
            belief.first_positive_at = None
            belief.positive_runs = 0
            belief.negative_runs = 0
        return self._new_event(seat_id, prior, new, now, reasons, track_id)

    def check_stale(self, now: datetime) -> list[StateEvent]:
        """Expire seats with no evidence beyond the stale horizon."""
        events: list[StateEvent] = []
        for seat_id, belief in self._beliefs.items():
            if belief.presence == PresenceState.STALE:
                continue
            if belief.last_evidence_at is None:
                continue
            age = (now - belief.last_evidence_at).total_seconds()
            if age > self._rules.stale_after_s:
                event = self._transition(
                    seat_id,
                    belief,
                    PresenceState.STALE,
                    VisibilityState.NOT_EVALUATED,
                    0.0,
                    ["stale-timeout", f"age_s={age:.0f}"],
                    now,
                    None,
                )
                if event is not None:
                    events.append(event)
        return events


def reconcile_from_events(
    events: list[StateEvent], session_id: str
) -> dict[str, tuple[str, datetime]]:
    """Rebuild current presence per seat: last event per seat wins.

    Documented model: append-only history is the source of truth; the
    current-state table is a cache. STALE is time-derived, not stored.
    """
    current: dict[str, tuple[str, datetime]] = {}
    for event in sorted(events, key=lambda e: (e.occurred_at, e.recorded_at)):
        if event.session_id != session_id or event.seat_id is None:
            continue
        if event.new_state is not None:
            current[event.seat_id] = (event.new_state, event.occurred_at)
    return current
