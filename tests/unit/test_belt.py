"""Belt strap detection tests — synthetic geometry, no weights, no camera."""

from __future__ import annotations

import numpy as np

from odc.belt import SeatBeltTracker, detect_belt
from odc.domain import BBox


def _box() -> BBox:
    return BBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)


def test_parallel_pair_detected_as_strap():
    img = np.full((240, 240, 3), 120, dtype=np.uint8)
    import cv2

    # Two parallel diagonal edges: the two sides of one strap.
    cv2.line(img, (40, 200), (200, 40), (230, 230, 230), 3)
    cv2.line(img, (60, 200), (220, 40), (230, 230, 230), 3)
    obs = detect_belt(img, _box())
    assert obs.state == "strap", obs.reason
    assert obs.score >= 0.8
    assert len(obs.segments) == 2
    assert obs.roi is not None


def test_long_single_diagonal_edge_is_a_strap_candidate():
    import cv2

    # A wide belt often exposes only one clean edge against a patterned shirt.
    img = np.full((240, 240, 3), 180, dtype=np.uint8)
    cv2.line(img, (70, 145), (150, 65), (100, 100, 100), 5)
    obs = detect_belt(img, _box())
    assert obs.state == "strap", obs.reason
    assert obs.reason == "single-long-line"


def test_plain_torso_is_clear_not_strap():
    img = np.full((240, 240, 3), 120, dtype=np.uint8)
    obs = detect_belt(img, _box())
    assert obs.state == "clear"


def test_vertical_seams_rejected():
    img = np.full((240, 240, 3), 120, dtype=np.uint8)
    import cv2

    for x in (60, 120, 180):
        cv2.line(img, (x, 10), (x, 230), (230, 230, 230), 3)
    assert detect_belt(img, _box()).state == "clear"


def test_horizontal_folds_rejected():
    img = np.full((240, 240, 3), 120, dtype=np.uint8)
    import cv2

    for y in (60, 120, 180):
        cv2.line(img, (10, y), (230, y), (230, 230, 230), 3)
    assert detect_belt(img, _box()).state == "clear"


def test_tiny_and_dark_are_unclear_never_negative():
    tiny = np.full((240, 240, 3), 120, dtype=np.uint8)
    small = BBox(x_min=0.4, y_min=0.4, x_max=0.5, y_max=0.5)
    assert detect_belt(tiny, small).state == "unclear"
    dark = np.full((240, 240, 3), 8, dtype=np.uint8)
    assert detect_belt(dark, _box()).state == "unclear"
    # Missing pixels never raise.
    assert detect_belt(None, _box()).state == "unclear"


def test_tracker_needs_temporal_confirmation():
    tracker = SeatBeltTracker(window=10)
    strap = detect_belt(np.full((240, 240, 3), 120, dtype=np.uint8), _box())
    assert strap.state == "clear"
    for _ in range(3):
        state = tracker.update("A", True, strap, frontal=True)
    assert state.state == "Unknown"  # only 3 clears: not enough for a verdict
    for _ in range(4):
        state = tracker.update("A", True, strap, frontal=True)
    assert state.state == "Unknown"  # camera-only clear pixels cannot verify absence
    assert "absence-unverified" in state.reason
    assert state.issue is False
    assert tracker.update("B", False, None).state == "Unknown"  # no occupant


def test_rear_view_clears_never_confirm_absence():
    # Backs of heads (R1 van): clear torsos without a visible face must
    # stay indeterminate — never a false orange "no belt".
    tracker = SeatBeltTracker(window=10)
    clear = detect_belt(np.full((240, 240, 3), 120, dtype=np.uint8), _box())
    for _ in range(10):
        state = tracker.update("A", True, clear, frontal=False)
    assert state.state == "Unknown"
    assert state.issue is False


def test_tracker_confirms_sustained_strap():
    import cv2

    tracker = SeatBeltTracker(window=10)
    img = np.full((240, 240, 3), 120, dtype=np.uint8)
    cv2.line(img, (40, 200), (200, 40), (230, 230, 230), 3)
    cv2.line(img, (60, 200), (220, 40), (230, 230, 230), 3)
    obs = detect_belt(img, _box())
    state = tracker.update("A", True, obs)
    assert state.state == "Unknown"  # single frame never confirms
    for _ in range(4):
        state = tracker.update("A", True, obs)
    assert state.state == "Detected"
    assert state.issue is False


def test_tracker_confirms_repeated_strap_with_intermittent_clear_frames():
    from odc.belt import BeltObservation

    tracker = SeatBeltTracker(window=10)
    state = None
    for index in range(10):
        observation = (
            BeltObservation("strap", 0.6)
            if index in {1, 4, 7}
            else BeltObservation("clear", 0.0)
        )
        state = tracker.update("A", True, observation, frontal=True)
    assert state is not None
    assert state.state == "Detected"
    assert state.strap_frames == 3


def test_two_strap_candidates_do_not_confirm_belt():
    from odc.belt import BeltObservation

    tracker = SeatBeltTracker(window=10)
    state = None
    for index in range(10):
        observation = (
            BeltObservation("strap", 0.6)
            if index in {1, 7}
            else BeltObservation("clear", 0.0)
        )
        state = tracker.update("A", True, observation, frontal=True)
    assert state is not None
    assert state.state == "Unknown"
