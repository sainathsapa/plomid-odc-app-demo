"""Belt frontality fallback + tracker honesty tests. No weights for fallback."""

from __future__ import annotations

import numpy as np

from odc.belt import SeatBeltTracker, detect_belt, estimate_frontal_fallback
from odc.domain import BBox


def _box(x0=0.1, y0=0.1, x1=0.9, y1=0.9) -> BBox:
    return BBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1)


def test_fallback_accepts_midsize_well_lit_box():
    # A mid-size cabin box (0.16 area, square-ish) with good light reads
    # assessable; the full-frame 0.64 box used in detect_belt unit tests is
    # a close-up with no torso to judge and stays indeterminate.
    assert estimate_frontal_fallback(_box(0.2, 0.2, 0.6, 0.6), True) is True


def test_fallback_admits_portrait_cabin_boxes():
    # Measured 9519139 rear occupant: frontal 3/4 view in a tall narrow box
    # (ratio 0.29). Below the old 0.35 floor she never confirmed absence.
    assert estimate_frontal_fallback(_box(0.24, 0.22, 0.46, 0.98), True) is True
    assert estimate_frontal_fallback(_box(0.45, 0.0, 0.55, 1.0), True) is None


def test_closeup_roi_frames_chest_not_face():
    from odc.belt import torso_roi

    big = _box(0.45, 0.05, 1.0, 0.98)  # area 0.51: close-up
    x0, y0, x1, y1 = torso_roi(big, 1000, 1000)
    assert (y0, y1) == (328, 905)  # shifted down to the chest
    assert (x0, x1) == (477, 972)  # widened to keep the strap side
    small = BBox(x_min=0.24, y_min=0.22, x_max=0.46, y_max=0.98)
    x0, y0, x1, y1 = torso_roi(small, 1000, 1000)
    assert (y0, y1) == (333, 751)  # standard window unchanged
    assert (x0, x1) == (273, 427)


def test_wide_smooth_strap_band_detected():
    import cv2

    # Smooth wide band: blurred edges still carry line pairs here, but the
    # verdict must be strap either way (measured 9519139 driver shape).
    img = np.full((240, 240, 3), 60, dtype=np.uint8)
    cv2.line(img, (200, 10), (40, 230), (235, 235, 235), 14)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    obs = detect_belt(img, _box())
    assert obs.state == "strap", obs.reason
    assert len(obs.segments) >= 1


def test_curved_band_needs_band_stage():
    import cv2

    # Curved band: LSD fragments stay short and unmergeable (angle drift),
    # so only the band stage can fire — the 9519139 driver failure mode.
    img = np.full((240, 240, 3), 60, dtype=np.uint8)
    cv2.ellipse(img, (120, 140), (70, 70), 0, 200, 260, (235, 235, 235), 14)
    obs = detect_belt(img, _box())
    assert obs.state == "strap", obs.reason
    assert obs.reason == "bright-band"


def test_low_margin_stripes_rejected_by_band_gate():
    # Diagonal but low-contrast: the band gate must reject (margin).
    # (The line stage's pinstripe response is unchanged pre-existing
    # behavior, out of scope for this fix.)
    import cv2

    from odc.belt import _band_evidence

    img = np.full((240, 240), 150, dtype=np.uint8)
    for off in range(-240, 480, 24):
        cv2.line(img, (off, 0), (off + 240, 240), 190, 8)
    found, _ = _band_evidence(img, 339.0)
    assert found is False


def test_big_pale_blob_stays_clear():
    import cv2

    img = np.full((240, 240, 3), 60, dtype=np.uint8)
    cv2.circle(img, (120, 120), 60, (235, 235, 235), -1)
    obs = detect_belt(img, _box())
    assert obs.state == "clear", obs.reason


def test_fallback_rejects_dark_tiny_and_extreme():
    assert estimate_frontal_fallback(_box(), False) is None  # dark
    tiny = BBox(x_min=0.4, y_min=0.4, x_max=0.5, y_max=0.5)
    assert estimate_frontal_fallback(tiny, True) is None
    full = BBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
    assert estimate_frontal_fallback(full, True) is None  # close-up, no torso
    sliver = BBox(x_min=0.45, y_min=0.0, x_max=0.55, y_max=1.0)
    assert estimate_frontal_fallback(sliver, True) is None


def test_tracker_unknown_viewpoint_never_fabricates_absence():
    tracker = SeatBeltTracker(window=10)
    clear = detect_belt(np.full((240, 240, 3), 120, dtype=np.uint8), _box())
    assert clear.state == "clear"
    for _ in range(10):
        state = tracker.update("A", True, clear, frontal=None)
    assert state.state == "Unknown"
    assert state.issue is False


def test_tracker_camera_only_frontal_clear_does_not_confirm_absence():
    tracker = SeatBeltTracker(window=10)
    clear = detect_belt(np.full((240, 240, 3), 120, dtype=np.uint8), _box())
    for _ in range(7):
        state = tracker.update("A", True, clear, frontal=True)
    assert state.state == "Unknown"
    assert "absence-unverified" in state.reason
    assert state.issue is False


def test_tracker_strap_counts_without_frontality():
    import cv2

    tracker = SeatBeltTracker(window=10)
    img = np.full((240, 240, 3), 120, dtype=np.uint8)
    cv2.line(img, (40, 200), (200, 40), (230, 230, 230), 3)
    cv2.line(img, (60, 200), (220, 40), (230, 230, 230), 3)
    obs = detect_belt(img, _box())
    assert obs.state == "strap"
    for _ in range(5):
        state = tracker.update("A", True, obs, frontal=None)
    assert state.state == "Detected"


def test_tracker_missing_evidence_stays_unknown():
    tracker = SeatBeltTracker(window=10)
    assert tracker.update("A", True, None, frontal=True).state == "Unknown"
    assert tracker.update("A", False, None).state == "Unknown"
    assert tracker.update("A", True, None, frontal=None).state == "Unknown"
