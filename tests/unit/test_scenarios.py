"""Procedural GT scenarios (bandwidth-free evaluation).

Scripted person-box sequences with EXACT expectations drive the REAL tracker
+ state engine. No video, no downloads, no model — pure logic proof with
perfect ground truth (something recorded footage can never give).
Deterministic: sinusoidal drift, fixed seeds (none needed — closed form).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from odc import contracts
from odc.domain import BBox, Detection, FrameMeta, RunMode
from odc.seats import SeatGeometry, SeatLayout
from odc.state import SeatStateEngine, StateRules
from odc.tracking import ByteTrackStyleTracker, TrackerConfig

T0 = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
LAYOUT = SeatLayout(
    camera_profile="synth",
    version="synth-v1",
    seats=[
        SeatGeometry(
            seat_id="L",
            polygon=[(0, 0), (0.5, 0), (0.5, 1), (0, 1)],
            camera_profile="t",
            version="t",
        ),
        SeatGeometry(
            seat_id="R",
            polygon=[(0.5, 0), (1, 0), (1, 1), (0.5, 1)],
            camera_profile="t",
            version="t",
        ),
    ],
)
RULES = StateRules(
    min_positive_frames=2,
    occupy_dwell_s=0.0,
    confirm_extra_s=1000.0,
    release_misses_likely=3,
    release_misses_unknown=6,
    empty_dwell_frames=5,
    stale_after_s=3600.0,
    cooldown_s=0.0,
    ambiguity_margin=0.15,
)


def _box(cx: float, cy: float, w: float = 0.2, h: float = 0.6) -> BBox:
    return BBox(x_min=cx - w / 2, y_min=cy - h / 2, x_max=cx + w / 2, y_max=cy + h / 2)


def _frame(seq: int, at: datetime):
    return contracts.Frame(
        meta=FrameMeta(
            source_id="synth", seq=seq, captured_at=at, width=100, height=100, mode=RunMode.TEST
        ),
        pixels=None,
    )


def run_scenario(frames_spec, rules=None, drift: float = 0.01):
    """frames_spec: list of (n_frames, [boxes...] | 'dark' flag...).

    Each item: (count, persons) where persons = list of (cx, cy) centers or
    [] for absence; prefix 'dark:' handled via engine.set_frame_quality.
    Returns (engine, tracker, all_events, final_states).
    """
    import math

    engine = SeatStateEngine("ses-synth", LAYOUT, rules or RULES, "cfg", "model")
    tracker = ByteTrackStyleTracker(TrackerConfig(min_hits=2, max_age=10))
    events = []
    seq = 0
    out = None
    for count, spec in frames_spec:
        dark = False
        if isinstance(spec, tuple) and spec and spec[0] == "dark":
            dark, persons = True, spec[1]
        else:
            persons = spec
        engine.set_frame_quality(dark=dark)
        for _k in range(count):
            at = T0 + timedelta(seconds=seq * 0.2)
            dets = [
                Detection(
                    class_label="person",
                    score=0.99,
                    bbox=_box(cx + drift * math.sin(seq), cy),
                    model_id="synth",
                    frame_seq=seq,
                )
                for cx, cy in persons
            ]
            frame = _frame(seq, at)
            states = tracker.update(frame, dets)
            out = engine.update(frame, states, LAYOUT.seats)
            events.extend(out.events)
            seq += 1
    return engine, tracker, events, out


def states_of(out):
    return {s.seat_id: s.presence.value for s in out.seats}


def test_clean_entry_and_stable_hold():
    _, _, events, out = run_scenario(
        [
            (5, []),
            (20, [(0.25, 0.5)]),
        ]
    )
    assert states_of(out) == {"L": "LIKELY_OCCUPIED", "R": "EMPTY_CANDIDATE"}
    likelies = [e for e in events if e.new_state == "LIKELY_OCCUPIED" and e.seat_id == "L"]
    assert len(likelies) == 1


def test_brief_occlusion_holds():
    _, _, events, out = run_scenario(
        [
            (10, [(0.25, 0.5)]),
            (3, []),
            (10, [(0.25, 0.5)]),
        ]
    )
    assert states_of(out)["L"] == "LIKELY_OCCUPIED"
    assert not [
        e for e in events if e.new_state in ("UNKNOWN", "EMPTY_CANDIDATE") and e.seat_id == "L"
    ]


def test_departure_documented_path():
    _, _, events, out = run_scenario(
        [
            (10, [(0.25, 0.5)]),
            (40, []),
        ]
    )
    assert states_of(out)["L"] == "EMPTY_CANDIDATE"  # sustained clear absence
    assert [e.new_state for e in events if e.seat_id == "L"] == [
        "LIKELY_OCCUPIED",
        "UNKNOWN",
        "EMPTY_CANDIDATE",
    ]


def test_dark_gap_never_empties():
    _, _, events, out = run_scenario(
        [
            (10, [(0.25, 0.5)]),
            (40, ("dark", [])),
        ]
    )
    assert states_of(out)["L"] == "UNKNOWN"
    assert "EMPTY_CANDIDATE" not in [e.new_state for e in events if e.seat_id == "L"]


def test_straddle_converges_without_flicker():
    # A body exactly on the zone edge: tracker dynamics resolve to one side.
    # Requirement is stability (no seat flapping, bounded events) — not a
    # specific side, and never a rapid EMPTY/OCCUPIED oscillation.
    _, _, events, out = run_scenario([(15, [(0.5, 0.5)])], drift=0.0)
    occupied = [s for s in out.seats if s.presence.value in ("OCCUPIED", "LIKELY_OCCUPIED")]
    assert len(occupied) == 1
    assert len(events) <= 3


def test_twin_duplicates_absorbed():
    _, tracker, events, out = run_scenario([(15, [(0.24, 0.5), (0.26, 0.5)])])
    assert states_of(out)["L"] == "LIKELY_OCCUPIED"
    assert states_of(out)["R"] != "OCCUPIED"
