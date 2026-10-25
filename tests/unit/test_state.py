"""Temporal engine tests — every Phase 6 acceptance scenario."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from odc import contracts
from odc.domain import (
    CameraHealth,
    EventType,
    HealthStatus,
    OccupantState,
    PresenceState,
    RunMode,
    SeatGeometry,
)
from odc.seats import SeatLayout
from odc.state import SeatStateEngine, StateRules, reconcile_from_events

T0 = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
A = SeatGeometry(
    seat_id="A", polygon=[(0, 0), (0.5, 0), (0.5, 1), (0, 1)], camera_profile="t", version="t"
)
B = SeatGeometry(
    seat_id="B", polygon=[(0.5, 0), (1, 0), (1, 1), (0.5, 1)], camera_profile="t", version="t"
)
LAYOUT = SeatLayout(camera_profile="t", version="t", seats=[A, B])

RULES = StateRules(
    min_positive_frames=2,
    occupy_dwell_s=0.0,
    confirm_extra_s=1000.0,
    release_misses_likely=3,
    release_misses_unknown=5,
    empty_dwell_frames=4,
    stale_after_s=60.0,
    cooldown_s=0.0,
    ambiguity_margin=0.15,
)


def _frame(seq: int, at: datetime):
    from odc.domain import FrameMeta

    return contracts.Frame(
        meta=FrameMeta(
            source_id="c", seq=seq, captured_at=at, width=100, height=100, mode=RunMode.REPLAY
        ),
        pixels=None,
    )


def _occ(track: str, bbox, score: float, at: datetime) -> OccupantState:
    return OccupantState(
        session_id="s", track_id=track, last_bbox=bbox, last_score=score, updated_at=at
    )


def _box(x0, y0, x1, y1):
    from odc.domain import BBox

    return BBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1)


IN_A = _box(0.1, 0.1, 0.4, 0.9)
STRADDLE = _box(0.35, 0.0, 0.65, 1.0)


def _engine(**kw) -> SeatStateEngine:
    rules = StateRules(**{**RULES.__dict__, **kw})
    return SeatStateEngine("s", LAYOUT, rules, "cfg-v1", "model-v1")


def _states(engine, out):
    return {s.seat_id: s.presence for s in out.seats}


def test_rekey_layout_keeps_carried_beliefs():
    from odc.domain import PresenceState

    eng = _engine()
    # Drive seat A to LIKELY_OCCUPIED with sustained person evidence.
    at = T0
    for seq in range(3):
        at += timedelta(seconds=1)
        out = eng.update(_frame(seq, at), [_occ("t1", IN_A, 0.9, at)], [A, B])
    assert _states(eng, out)["A"] == PresenceState.LIKELY_OCCUPIED
    # Nudge A's polygon (edit) and add C: A keeps state, C starts UNKNOWN.
    A2 = SeatGeometry(
        seat_id="A", polygon=[(0, 0), (0.45, 0), (0.45, 1), (0, 1)],
        camera_profile="t", version="t2",
    )
    C = SeatGeometry(
        seat_id="C", polygon=[(0.8, 0.8), (1, 0.8), (1, 1), (0.8, 1)],
        camera_profile="t", version="t2",
    )
    summary = eng.rekey_layout(SeatLayout(camera_profile="t", version="t2", seats=[A2, C]))
    assert summary == {"carried": 1, "reset": 1}
    at += timedelta(seconds=1)
    out = eng.update(_frame(9, at), [], [A2, C])
    states = _states(eng, out)
    # A was LIKELY with 0 misses so far: one miss keeps it (no dwell reset).
    assert states["A"] == PresenceState.LIKELY_OCCUPIED
    assert states["C"] == PresenceState.UNKNOWN


def test_flicker_never_confirms():
    eng = _engine()
    events = 0
    for i in range(10):
        at = T0 + timedelta(seconds=i)
        occs = [_occ("t1", IN_A, 0.9, at)] if i % 2 == 0 else []
        out = eng.update(_frame(i, at), occs, [A, B])
        events += len(out.events)
    states = _states(eng, out)
    assert states["A"] != PresenceState.OCCUPIED
    assert states["B"] == PresenceState.EMPTY_CANDIDATE  # sustained clear absence
    assert events <= 2


def test_one_frame_miss_holds_likely():
    eng = _engine()
    for i in range(3):
        at = T0 + timedelta(seconds=i)
        eng.update(_frame(i, at), [_occ("t1", IN_A, 0.9, at)], [A, B])
    assert eng._beliefs["A"].presence == PresenceState.LIKELY_OCCUPIED
    out = eng.update(_frame(3, T0 + timedelta(seconds=3)), [], [A, B])
    assert _states(eng, out)["A"] == PresenceState.LIKELY_OCCUPIED
    for i in range(4, 9):
        out = eng.update(_frame(i, T0 + timedelta(seconds=i)), [], [A, B])
    assert _states(eng, out)["A"] == PresenceState.UNKNOWN


def test_confirm_path_likely_to_occupied():
    eng = _engine(confirm_extra_s=0.0)
    for i in range(4):
        at = T0 + timedelta(seconds=i)
        out = eng.update(_frame(i, at), [_occ("t1", IN_A, 0.9, at)], [A, B])
    assert _states(eng, out)["A"] == PresenceState.OCCUPIED


def test_brief_absence_never_empties_occupied():
    eng = _engine(confirm_extra_s=0.0)
    for i in range(4):
        eng.update(
            _frame(i, T0 + timedelta(seconds=i)),
            [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=i))],
            [A, B],
        )
    for i in range(4, 7):
        out = eng.update(_frame(i, T0 + timedelta(seconds=i)), [], [A, B])
    assert _states(eng, out)["A"] in (PresenceState.OCCUPIED, PresenceState.LIKELY_OCCUPIED)


def test_poor_observation_blocks_empty_forever():
    eng = _engine()
    eng.set_frame_quality(dark=True)
    for i in range(30):
        out = eng.update(_frame(i, T0 + timedelta(seconds=i)), [], [A, B])
    assert _states(eng, out)["A"] == PresenceState.UNKNOWN
    assert _states(eng, out)["B"] == PresenceState.UNKNOWN


def test_apparent_size_needs_window_and_votes():
    from odc.state import StateRules, _classify_size

    rules = StateRules()
    assert _classify_size([0.2, 0.3], rules)[0].value == "UNKNOWN"
    assert _classify_size([0.2, 0.3, 0.1, 0.4, 0.2], rules)[0].value == ("APPARENT_SMALL_OCCUPANT")
    assert _classify_size([1.5, 1.2, 2.0], rules)[0].value == "APPARENT_LARGE_OCCUPANT"
    assert _classify_size([0.7, 0.8, 0.6], rules)[0].value == "UNKNOWN"


def test_engine_publishes_size_after_window():
    eng = _engine()
    last = None
    for i in range(6):
        at = T0 + timedelta(seconds=i)
        last = eng.update(_frame(i, at), [_occ("t1", IN_A, 0.9, at)], [A, B])
    occ = next(o for o in last.occupants if o.track_id == "t1")
    assert occ.apparent_size is not None
    # IN_A area 0.24 / seat A area 0.5 = 0.48 < 0.5 → provisional SMALL.
    assert occ.apparent_size.value.value == "APPARENT_SMALL_OCCUPANT"
    assert occ.apparent_size.confidence == 0.0


def test_empty_requires_evidence_from_unknown():
    eng = _engine()
    for i in range(6):
        out = eng.update(_frame(i, T0 + timedelta(seconds=i)), [], [A, B])
    assert _states(eng, out)["A"] == PresenceState.EMPTY_CANDIDATE
    assert _states(eng, out)["B"] == PresenceState.EMPTY_CANDIDATE


def test_stale_and_recovery():
    eng = _engine()
    for i in range(3):
        eng.update(
            _frame(i, T0 + timedelta(seconds=i)),
            [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=i))],
            [A, B],
        )
    events = eng.check_stale(T0 + timedelta(seconds=200))
    assert {e.seat_id for e in events} == {"A", "B"}
    assert eng._beliefs["A"].presence == PresenceState.STALE
    for i in range(3):
        out = eng.update(
            _frame(10 + i, T0 + timedelta(seconds=202 + i)),
            [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=202 + i))],
            [A, B],
        )
    assert _states(eng, out)["A"] == PresenceState.LIKELY_OCCUPIED


def test_track_swap_no_storm():
    eng = _engine()
    for i in range(3):
        eng.update(
            _frame(i, T0 + timedelta(seconds=i)),
            [_occ("tA", IN_A, 0.9, T0 + timedelta(seconds=i))],
            [A, B],
        )
    events = 0
    for i in range(3, 8):
        out = eng.update(
            _frame(i, T0 + timedelta(seconds=i)),
            [_occ("tB", IN_A, 0.85, T0 + timedelta(seconds=i))],
            [A, B],
        )
        events += len(
            [
                e
                for e in out.events
                if e.event_type == EventType.SEAT_PRESENCE_CHANGED and e.seat_id == "A"
            ]
        )
    assert events == 0
    assert eng._beliefs["A"].presence == PresenceState.LIKELY_OCCUPIED


def test_ambiguity_blocks_empty():
    eng = _engine(ambiguity_margin=0.95)
    for i in range(10):
        at = T0 + timedelta(seconds=i)
        out = eng.update(_frame(i, at), [_occ("tX", STRADDLE, 0.9, at)], [A, B])
    states = _states(eng, out)
    assert states["A"] != PresenceState.EMPTY_CANDIDATE
    assert states["B"] != PresenceState.EMPTY_CANDIDATE


def test_ambiguous_track_only_blocks_empty_confirmation_in_competing_seats():
    from odc.seats import candidate_seats_for_bbox

    left = SeatGeometry(
        seat_id="left", polygon=[(0, 0), (0.4, 0), (0.4, 1), (0, 1)],
        camera_profile="t", version="t",
    )
    middle = SeatGeometry(
        seat_id="middle", polygon=[(0.4, 0), (0.7, 0), (0.7, 1), (0.4, 1)],
        camera_profile="t", version="t",
    )
    right = SeatGeometry(
        seat_id="right", polygon=[(0.8, 0), (1, 0), (1, 1), (0.8, 1)],
        camera_profile="t", version="t",
    )
    layout = SeatLayout(camera_profile="t", version="t", seats=[left, middle, right])
    rules = StateRules(**{**RULES.__dict__, "ambiguity_margin": 0.95})
    engine = SeatStateEngine("s", layout, rules)
    track_box = _box(0.35, 0.1, 0.45, 0.9)
    assert set(candidate_seats_for_bbox(track_box, layout, 0.95)) == {"left", "middle"}

    output = None
    for seq in range(5):
        at = T0 + timedelta(seconds=seq)
        output = engine.update(
            _frame(seq, at), [_occ("ambiguous-track", track_box, 0.9, at)], layout.seats
        )

    assert output is not None
    states = _states(engine, output)
    assert states["left"] != PresenceState.EMPTY_CANDIDATE
    assert states["middle"] != PresenceState.EMPTY_CANDIDATE
    assert states["right"] == PresenceState.EMPTY_CANDIDATE


def test_source_restart_resets_counters_safely():
    eng = _engine()
    eng.update(_frame(0, T0), [_occ("t1", IN_A, 0.9, T0)], [A, B])
    out = eng.update(
        _frame(0, T0 + timedelta(seconds=1)),
        [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=1))],
        [A, B],
    )
    likely_events = [e for e in out.events if e.new_state == "LIKELY_OCCUPIED"]
    assert len(likely_events) <= 1


def test_replay_loop_same_timestamps_accepted():
    eng = _engine()
    for _loop in range(2):
        for i in range(4):
            out = eng.update(
                _frame(i, T0 + timedelta(seconds=i)),
                [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=i))],
                [A, B],
            )
    assert eng.out_of_order_dropped == 0
    assert out.seats, "looped replay must rebuild state, not drop"


def test_out_of_order_dropped():
    eng = _engine()
    eng.update(
        _frame(5, T0 + timedelta(seconds=5)),
        [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=5))],
        [A, B],
    )
    out = eng.update(
        _frame(6, T0 + timedelta(seconds=1)),
        [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=1))],
        [A, B],
    )
    assert eng.out_of_order_dropped == 1
    assert out.events == [] and out.seats == []


def test_stale_never_from_future_clock():
    eng = _engine()
    for i in range(3):
        eng.update(
            _frame(i, T0 + timedelta(seconds=i)),
            [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=i))],
            [A, B],
        )
    assert eng.check_stale(T0 - timedelta(seconds=10)) == []


def test_event_content_complete():
    eng = _engine()
    all_events = []
    for i in range(3):
        out = eng.update(
            _frame(i, T0 + timedelta(seconds=i)),
            [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=i))],
            [A, B],
        )
        all_events.extend(out.events)
    assert all_events
    event = all_events[0]
    assert event.session_id == "s" and event.seat_id == "A"
    assert event.track_id == "t1"
    assert event.occurred_at <= event.recorded_at
    assert event.prior_state == "UNKNOWN" and event.new_state == "LIKELY_OCCUPIED"
    assert event.reason_codes and event.config_version == "cfg-v1"
    assert event.model_set_version == "model-v1"


def test_relative_cabin_size_never_marks_child_without_specialist_evidence():
    from odc.domain import SizeAppearance

    seats = [
        SeatGeometry(seat_id=f"S{i}", polygon=poly, camera_profile="t", version="t")
        for i, poly in enumerate(
            (
                [(0, 0), (1 / 3, 0), (1 / 3, 1), (0, 1)],
                [(1 / 3, 0), (2 / 3, 0), (2 / 3, 1), (1 / 3, 1)],
                [(2 / 3, 0), (1, 0), (1, 1), (2 / 3, 1)],
            ),
            start=1,
        )
    ]
    engine = SeatStateEngine(
        "s", SeatLayout(camera_profile="t", version="t", seats=seats), RULES
    )
    boxes = [
        _box(0.02, 0.05, 0.31, 0.95),
        _box(0.39, 0.05, 0.59, 0.95),
        _box(0.69, 0.05, 0.98, 0.95),
    ]
    output = None
    for seq in range(3):
        at = T0 + timedelta(seconds=seq)
        output = engine.update(
            _frame(seq, at),
            [_occ(f"t{i}", box, 0.9, at) for i, box in enumerate(boxes, start=1)],
            seats,
        )
    assert output is not None
    child_track = next(occ for occ in output.occupants if occ.track_id == "t2")
    assert child_track.apparent_size is not None
    assert child_track.apparent_size.value == SizeAppearance.UNKNOWN
    child_seat = next(seat for seat in output.seats if seat.seat_id == "S2")
    assert "child-candidate:relative-cabin-size" not in child_seat.reason_codes
    assert not any(
        event.event_type == EventType.CHILD_CANDIDATE_DETECTED for event in output.events
    )

    same_size_boxes = [
        _box(0.02, 0.05, 0.31, 0.95),
        _box(0.35, 0.05, 0.64, 0.95),
        _box(0.69, 0.05, 0.98, 0.95),
    ]
    for seq in range(3, 8):
        at = T0 + timedelta(seconds=seq)
        output = engine.update(
            _frame(seq, at),
            [_occ(f"t{i}", box, 0.9, at) for i, box in enumerate(same_size_boxes, 1)],
            seats,
        )
    child_seat = next(seat for seat in output.seats if seat.seat_id == "S2")
    assert "child-candidate:relative-cabin-size" not in child_seat.reason_codes

    at = T0 + timedelta(seconds=8)
    output = engine.update(
        _frame(8, at),
        [_occ(f"t{i}", box, 0.9, at) for i, box in enumerate(same_size_boxes, 1)],
        seats,
    )
    child_seat = next(seat for seat in output.seats if seat.seat_id == "S2")
    assert "child-candidate:relative-cabin-size" not in child_seat.reason_codes
    assert not any(event.event_type == EventType.CHILD_CANDIDATE_CLEARED for event in output.events)


def test_specialist_child_evidence_is_seat_associated_and_serializable():
    """The specialist evidence path must survive temporal aggregation."""
    from odc.domain import Detection

    engine = _engine()
    at = T0
    child = Detection(
        class_label="child",
        score=0.417,
        bbox=_box(0.18, 0.35, 0.32, 0.70),
        model_id="grounding-dino-swint-ogc",
        frame_seq=0,
    )
    output = engine.update(
        _frame(0, at),
        [_occ("t1", IN_A, 0.9, at)],
        [A, B],
        specialist_detections=[child],
    )
    seat = next(item for item in output.seats if item.seat_id == "A")
    assert any(
        reason.startswith("specialist-child-occupant:child:")
        for reason in seat.reason_codes
    )
    assert "specialist-score:0.417" in seat.reason_codes
    assert "specialist-frame:0" in seat.reason_codes


def test_specialist_child_box_without_person_overlap_is_rejected():
    from odc.domain import Detection

    engine = _engine()
    at = T0
    unrelated = Detection(
        class_label="child",
        score=0.9,
        bbox=_box(0.42, 0.35, 0.49, 0.70),
        model_id="grounding-dino-swint-ogc",
        frame_seq=0,
    )
    output = engine.update(
        _frame(0, at),
        [_occ("t1", IN_A, 0.9, at)],
        [A, B],
        specialist_detections=[unrelated],
    )
    seat = next(item for item in output.seats if item.seat_id == "A")
    assert not any(code.startswith("specialist-child-occupant:") for code in seat.reason_codes)


def test_low_similarity_child_candidate_is_rejected_as_uncalibrated_evidence():
    from odc.domain import Detection

    engine = _engine()
    at = T0
    low_score = Detection(
        class_label="child",
        score=0.299,
        bbox=IN_A,
        model_id="grounding-dino-swint-ogc",
        frame_seq=0,
    )
    output = engine.update(
        _frame(0, at),
        [_occ("t1", IN_A, 0.9, at)],
        [A, B],
        specialist_detections=[low_score],
    )
    seat = next(item for item in output.seats if item.seat_id == "A")
    assert not any(code.startswith("specialist-child-occupant:") for code in seat.reason_codes)


def test_weak_child_match_without_independent_child_context_is_rejected():
    from odc.domain import Detection

    engine = _engine()
    output = engine.update(
        _frame(0, T0),
        [_occ("t1", IN_A, 0.9, T0)],
        [A, B],
        specialist_detections=[
            Detection(
                class_label="child",
                score=0.314,
                bbox=_box(0.0, 0.0, 0.73, 1.0),
                model_id="gd",
                frame_seq=0,
            ),
        ],
    )
    seat = next(item for item in output.seats if item.seat_id == "A")
    assert not any(code.startswith("specialist-child-occupant:") for code in seat.reason_codes)


def test_occupant_age_decision_does_not_depend_on_detection_box_area():
    from odc.domain import BBox, Detection

    full_seat = SeatGeometry(
        seat_id="whole-cabin",
        polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
        camera_profile="t",
        version="t",
    )
    layout = SeatLayout(camera_profile="t", version="t", seats=[full_seat])
    occupant = BBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)
    candidate_boxes = [
        BBox(x_min=0.35, y_min=0.35, x_max=0.55, y_max=0.55),
        BBox(x_min=0.0, y_min=0.0, x_max=0.95, y_max=0.95),
    ]
    for seq, candidate_box in enumerate(candidate_boxes):
        engine = SeatStateEngine("s", layout, RULES, "cfg-v1", "model-v1")
        output = engine.update(
            _frame(seq, T0),
            [_occ("t1", occupant, 0.9, T0)],
            [full_seat],
            specialist_detections=[
                Detection(
                    class_label="child",
                    score=0.36,
                    bbox=candidate_box,
                    model_id="gd",
                    frame_seq=seq,
                )
            ],
        )
        seat = output.seats[0]
        assert any(code.startswith("specialist-child-occupant:") for code in seat.reason_codes)


def test_repeated_associated_low_score_child_survives_temporal_confirmation():
    from odc.domain import Detection

    engine = _engine()
    child_detection = Detection(
        class_label="child",
        score=0.33,
        bbox=_box(0.28, 0.2, 0.48, 0.8),
        model_id="gd",
        frame_seq=10,
    )
    first = engine.update(
        _frame(0, T0),
        [_occ("t1", IN_A, 0.9, T0)],
        [A, B],
        specialist_detections=[
            child_detection,
            Detection(
                class_label="child seat",
                score=0.4,
                bbox=_box(0.30, 0.25, 0.48, 0.85),
                model_id="gd",
                frame_seq=10,
            ),
        ],
    )
    first_a = next(item for item in first.seats if item.seat_id == "A")
    assert not any(code.startswith("specialist-child-occupant:") for code in first_a.reason_codes)

    second = engine.update(
        _frame(1, T0 + timedelta(seconds=1)),
        [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=1))],
        [A, B],
        specialist_detections=[
            child_detection.model_copy(update={"frame_seq": 20}),
            Detection(
                class_label="child seat",
                score=0.4,
                bbox=_box(0.30, 0.25, 0.48, 0.85),
                model_id="gd",
                frame_seq=20,
            ),
        ],
    )
    second_a = next(item for item in second.seats if item.seat_id == "A")
    assert any(
        reason.startswith("specialist-child-occupant:child:")
        for reason in second_a.reason_codes
    )


def test_unconfirmed_child_candidate_is_not_overwritten_by_adult_label():
    from odc.attributes import seat_attributes
    from odc.domain import Detection

    engine = _engine()
    candidate_box = _box(0.12, 0.2, 0.38, 0.8)
    adult = Detection(
        class_label="adult",
        score=0.349,
        bbox=candidate_box,
        model_id="gd",
        frame_seq=10,
    )
    engine.update(
        _frame(0, T0),
        [_occ("t1", IN_A, 0.9, T0)],
        [A, B],
        specialist_detections=[adult],
    )
    second = engine.update(
        _frame(1, T0 + timedelta(seconds=1)),
        [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=1))],
        [A, B],
        specialist_detections=[
            Detection(
                class_label="child",
                score=0.301,
                bbox=candidate_box,
                model_id="gd",
                frame_seq=20,
            ),
            Detection(
                class_label="child seat",
                score=0.4,
                bbox=_box(0.1, 0.2, 0.4, 0.8),
                model_id="gd",
                frame_seq=20,
            ),
        ],
    )
    seat = next(item for item in second.seats if item.seat_id == "A")
    row = seat_attributes(seat, second.occupants)
    assert any(reason.startswith("specialist-child-pending:") for reason in seat.reason_codes)
    assert row["child"] == "Unknown"
    assert row["child_reason"] == "specialist-child-unconfirmed"
    assert row["occupant_category"] == "unknown"
    assert row["child_candidate"] is True


def test_distinct_child_and_adult_tracks_do_not_conflict_at_seat_level():
    from odc.domain import BBox, Detection

    engine = _engine()
    adult_box = BBox(x_min=0.05, y_min=0.05, x_max=0.45, y_max=0.90)
    child_box = BBox(x_min=0.15, y_min=0.50, x_max=0.40, y_max=0.95)
    output = engine.update(
        _frame(0, T0),
        [
            _occ("adult-track", adult_box, 0.9, T0),
            _occ("child-track", child_box, 0.9, T0),
        ],
        [A, B],
        specialist_detections=[
            Detection(
                class_label="adult",
                score=0.60,
                bbox=BBox(x_min=0.08, y_min=0.08, x_max=0.32, y_max=0.48),
                model_id="gd",
                frame_seq=0,
            ),
            Detection(
                class_label="child",
                score=0.36,
                bbox=child_box,
                model_id="gd",
                frame_seq=0,
            ),
        ],
    )
    seat = next(item for item in output.seats if item.seat_id == "A")
    assert any(
        reason.startswith("specialist-child-occupant:child:track=child-track:")
        for reason in seat.reason_codes
    )
    assert any(
        reason.startswith("specialist-adult-occupant:adult:track=adult-track:")
        for reason in seat.reason_codes
    )


def test_child_evidence_is_removed_when_associated_track_leaves_seat():
    from odc.domain import Detection

    engine = _engine()
    first = engine.update(
        _frame(0, T0),
        [_occ("t1", IN_A, 0.9, T0)],
        [A, B],
        specialist_detections=[
            Detection(
                class_label="child",
                score=0.4,
                bbox=IN_A,
                model_id="gd",
                frame_seq=0,
            )
        ],
    )
    assert any(
        reason.startswith("specialist-child-occupant:")
        for reason in next(seat for seat in first.seats if seat.seat_id == "A").reason_codes
    )
    second = engine.update(
        _frame(1, T0 + timedelta(seconds=1)),
        [_occ("t2", _box(0.6, 0.1, 0.9, 0.9), 0.9, T0 + timedelta(seconds=1))],
        [A, B],
    )
    assert not any(
        reason.startswith("specialist-child-occupant:")
        for seat in second.seats
        for reason in seat.reason_codes
    )


def test_adult_candidate_without_matching_person_track_is_rejected():
    from odc.domain import Detection

    engine = _engine()
    output = engine.update(
        _frame(0, T0),
        [_occ("t1", IN_A, 0.9, T0)],
        [A, B],
        specialist_detections=[
            Detection(
                class_label="adult",
                score=0.8,
                bbox=_box(0.01, 0.1, 0.08, 0.4),
                model_id="gd",
                frame_seq=0,
            )
        ],
    )
    seat = next(item for item in output.seats if item.seat_id == "A")
    assert not any(code.startswith("specialist-adult-occupant:") for code in seat.reason_codes)


def test_cooldown_suppresses_reentry_spam():
    eng = _engine(cooldown_s=100.0)
    likelies = 0
    for i in range(3):
        out = eng.update(
            _frame(i, T0 + timedelta(seconds=i)),
            [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=i))],
            [A, B],
        )
        likelies += sum(1 for e in out.events if e.new_state == "LIKELY_OCCUPIED")
    for i in range(3, 9):
        out = eng.update(_frame(i, T0 + timedelta(seconds=i)), [], [A, B])
    for i in range(9, 12):
        out = eng.update(
            _frame(i, T0 + timedelta(seconds=i)),
            [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=i))],
            [A, B],
        )
        likelies += sum(1 for e in out.events if e.new_state == "LIKELY_OCCUPIED")
    assert likelies == 1


def test_camera_unhealthy_distinct():
    eng = _engine()
    for i in range(3):
        eng.update(
            _frame(i, T0 + timedelta(seconds=i)),
            [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=i))],
            [A, B],
        )
    eng.set_camera_health(CameraHealth(camera_id="c", status=HealthStatus.UNHEALTHY, checked_at=T0))
    out = eng.update(
        _frame(9, T0 + timedelta(seconds=9)),
        [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=9))],
        [A, B],
    )
    seat = next(s for s in out.seats if s.seat_id == "A")
    assert seat.presence == PresenceState.UNKNOWN
    assert seat.visibility.name == "CAMERA_UNHEALTHY"
    assert out.events and out.events[0].reason_codes == ["camera-unhealthy"]


def test_reconcile_matches_engine():
    eng = _engine()
    history = []
    for i in range(6):
        out = eng.update(
            _frame(i, T0 + timedelta(seconds=i)),
            [_occ("t1", IN_A, 0.9, T0 + timedelta(seconds=i))],
            [A, B],
        )
        history.extend(out.events)
    current = reconcile_from_events(history, "s")
    assert current["A"][0] == eng._beliefs["A"].presence.value


def _obj(label, score, box):
    from odc.domain import Detection

    return Detection(class_label=label, score=score, bbox=box, model_id="m", frame_seq=0)


def test_object_evidence_after_three_frames():
    from odc.domain import BBox

    eng = _engine()
    bag = BBox(x_min=0.1, y_min=0.5, x_max=0.3, y_max=0.8)
    last = None
    for i in range(4):
        at = T0 + timedelta(seconds=i)
        last = eng.update(_frame(i, at), [], [A, B], objects=[_obj("handbag", 0.8, bag)])
    seat = next(s for s in last.seats if s.seat_id == "A")
    assert seat.object_label == "handbag"
    assert "object-evidence:handbag" in seat.reason_codes


def test_object_evidence_clears_when_leaves():
    from odc.domain import BBox

    eng = _engine()
    bag = BBox(x_min=0.1, y_min=0.5, x_max=0.3, y_max=0.8)
    for i in range(4):
        at = T0 + timedelta(seconds=i)
        eng.update(_frame(i, at), [], [A, B], objects=[_obj("handbag", 0.8, bag)])
    for i in range(4, 6):
        at = T0 + timedelta(seconds=i)
        last = eng.update(_frame(i, at), [], [A, B], objects=[])
    seat = next(s for s in last.seats if s.seat_id == "A")
    assert seat.object_label is None


def test_non_luggage_ignored_for_seats():
    from odc.domain import BBox

    eng = _engine()
    box = BBox(x_min=0.1, y_min=0.5, x_max=0.3, y_max=0.8)
    for i in range(4):
        at = T0 + timedelta(seconds=i)
        last = eng.update(_frame(i, at), [], [A, B], objects=[_obj("car", 0.9, box)])
    seat = next(s for s in last.seats if s.seat_id == "A")
    assert seat.object_label is None


def test_soft_proxy_large_suitcase_confirms_restraint():
    # COCO mislabels seat shells/carriers as suitcase: a LARGE sustained
    # suitcase box is a restraint proxy (token), not just luggage.
    from odc.domain import BBox

    eng = _engine()
    big = BBox(x_min=0.05, y_min=0.3, x_max=0.45, y_max=0.9)  # area 0.24
    last = None
    for i in range(4):
        at = T0 + timedelta(seconds=i)
        last = eng.update(_frame(i, at), [], [A, B], objects=[_obj("suitcase", 0.8, big)])
    seat = next(s for s in last.seats if s.seat_id == "A")
    assert seat.object_label == "suitcase"
    assert "restraint-proxy:suitcase" in seat.reason_codes


def test_soft_proxy_small_purse_stays_luggage_only():
    # A purse-sized handbag is tracked as object evidence but must never
    # confirm a child seat (no proxy token).
    from odc.domain import BBox

    eng = _engine()
    tiny = BBox(x_min=0.1, y_min=0.5, x_max=0.2, y_max=0.6)  # area 0.01
    last = None
    for i in range(4):
        at = T0 + timedelta(seconds=i)
        last = eng.update(_frame(i, at), [], [A, B], objects=[_obj("handbag", 0.8, tiny)])
    seat = next(s for s in last.seats if s.seat_id == "A")
    assert seat.object_label == "handbag"
    assert "object-evidence:handbag" in seat.reason_codes
    assert not [r for r in seat.reason_codes if r.startswith("restraint-proxy:")]


def test_tiny_fragment_casts_no_size_vote():
    # A 0.02 face fragment must not vote APPARENT_SMALL even repeatedly;
    # history stays empty and the occupant reads UNKNOWN, not small.
    from odc.domain import BBox

    eng = _engine()
    frag = BBox(x_min=0.6, y_min=0.3, x_max=0.75, y_max=0.45)  # area ~0.022
    last = None
    for i in range(6):
        at = T0 + timedelta(seconds=i)
        last = eng.update(_frame(i, at), [_occ("t1", frag, 0.9, at)], [A, B])
    occ = next(o for o in last.occupants if o.track_id == "t1")
    assert occ.apparent_size is not None
    assert occ.apparent_size.value.value == "UNKNOWN"


def test_mixed_objects_with_filtered_labels_no_crash():
    # Regression: a filtered label (car) ahead of a valid one must not
    # shift indices — the handbag attribution must survive the car.
    from odc.domain import BBox

    eng = _engine()
    box = BBox(x_min=0.1, y_min=0.5, x_max=0.3, y_max=0.8)
    last = None
    for i in range(4):
        at = T0 + timedelta(seconds=i)
        last = eng.update(
            _frame(i, at),
            [],
            [A, B],
            objects=[_obj("car", 0.9, box), _obj("handbag", 0.8, box)],
        )
    seat = next(s for s in last.seats if s.seat_id == "A")
    assert seat.object_label == "handbag"


def test_straddling_box_attaches_both_seats():
    # Close-up geometry: one box straddling two seats shares evidence
    # instead of landing on exactly one (possibly wrong) seat.
    from odc.domain import BBox

    eng = _engine()
    span = BBox(x_min=0.3, y_min=0.1, x_max=0.7, y_max=0.9)
    last = None
    for i in range(4):
        at = T0 + timedelta(seconds=i)
        last = eng.update(_frame(i, at), [], [A, B], objects=[_obj("chair", 0.8, span)])
    got = {s.seat_id: s.object_label for s in last.seats}
    assert got == {"A": "chair", "B": "chair"}
