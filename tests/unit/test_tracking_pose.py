"""Phase 5 unit tests — tracker math, association rules, pose mapping."""

from __future__ import annotations

import numpy as np
import pytest

from odc import contracts
from odc.domain import BBox, Detection
from odc.pose import NullPose, map_landmarks
from odc.seats import (
    SeatLayout,
    associate_seat_first,
    associate_track_first,
    load_layout,
    point_in_polygon,
    polygon_intersection_area,
)
from odc.tracking import ByteTrackStyleTracker, TrackerConfig, _Kalman, iou

BOX_A = BBox(x_min=0.1, y_min=0.2, x_max=0.4, y_max=0.9)
BOX_B = BBox(x_min=0.5, y_min=0.2, x_max=0.8, y_max=0.9)


def _det(label: str, score: float, bbox: BBox, seq: int = 1) -> Detection:
    return Detection(class_label=label, score=score, bbox=bbox, model_id="m", frame_seq=seq)


def _frame(seq: int = 1):
    from odc.capture import FakeSource

    src = FakeSource(n_frames=3)
    src.open()
    for _ in range(seq):
        frame = src.next()
    assert frame is not None
    return frame


def test_iou_identity_and_disjoint():
    assert iou(BOX_A, BOX_A) == pytest.approx(1.0)
    assert iou(BOX_A, BOX_B) == 0.0


def test_kalman_converges_to_measurement():
    kf = _Kalman(np.array([0.5, 0.5, 0.3, 0.7]))
    pred = kf.predict()
    assert pred.shape == (4,)
    for _ in range(20):
        kf.update(np.array([0.6, 0.5, 0.3, 0.7]))
    assert abs(kf.x[0] - 0.6) < 0.05


def test_tracker_confirms_and_drops():
    tracker = ByteTrackStyleTracker(TrackerConfig(min_hits=2, max_age=2))
    frame = _frame()
    assert tracker.update(frame, [_det("person", 0.9, BOX_A)]) == []
    states = tracker.update(frame, [_det("person", 0.9, BOX_A)])
    assert [s.track_id for s in states] == ["t1"]
    assert states[0].last_bbox == BOX_A and states[0].last_score == 0.9
    assert tracker.update(frame, []) == []  # age 1, still alive but unconfirmed? confirmed stays
    assert tracker.live_tracks == 1
    assert tracker.update(frame, []) == []
    assert tracker.update(frame, []) == []  # age 3 > max_age 2 → dropped
    assert tracker.live_tracks == 0
    assert tracker.fragments == 1


def test_tracker_reset_new_session_ids():
    tracker = ByteTrackStyleTracker(TrackerConfig(min_hits=1))
    frame = _frame()
    assert tracker.update(frame, [_det("person", 0.9, BOX_A)])[0].track_id == "t1"
    tracker.reset()
    assert tracker.update(frame, [_det("person", 0.9, BOX_A)])[0].track_id == "t1"
    assert tracker.fragments == 0


def test_tracker_ignores_non_person_and_low():
    tracker = ByteTrackStyleTracker(TrackerConfig(min_hits=1))
    frame = _frame()
    assert tracker.update(frame, [_det("chair", 0.9, BOX_A)]) == []
    assert tracker.update(frame, [_det("person", 0.05, BOX_A)]) == []


def test_polygon_and_layout():
    square = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
    assert point_in_polygon(0.25, 0.25, square)
    assert not point_in_polygon(0.75, 0.75, square)
    layout = load_layout("configs/seats/demo-r1-v0.yaml")
    assert layout.version == "demo-r1-v1" and len(layout.seats) == 4
    assert isinstance(layout, SeatLayout)
    for index, seat_a in enumerate(layout.seats):
        for seat_b in layout.seats[index + 1 :]:
            assert polygon_intersection_area(seat_a.polygon, seat_b.polygon) < 1e-9


def test_layout_rejects_profile_mismatch(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "camera_profile: a\nversion: v\nseats:\n"
        "  - seat_id: s\n    polygon: [[0, 0], [1, 0], [1, 1]]\n"
        "    camera_profile: b\n    version: v\n"
    )
    with pytest.raises(ValueError):
        load_layout(str(bad))


def test_association_clear_and_ambiguous():
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    baby = BBox(x_min=0.6, y_min=0.4, x_max=0.95, y_max=0.95)
    (assoc,) = associate_track_first([("t3", baby)], layout)
    assert assoc.seat_id == "child-seat-right" and not assoc.ambiguous
    middle = BBox(x_min=0.3, y_min=0.0, x_max=0.6, y_max=1.0)
    (amb,) = associate_track_first([("t9", middle)], layout, ambiguity_margin=0.96)
    assert amb.seat_id is None and amb.ambiguous and amb.runner_up is not None
    void = BBox(x_min=0.9, y_min=0.0, x_max=1.0, y_max=0.1)
    (out,) = associate_track_first([("t0", void)], layout)
    assert out.seat_id is None and not out.ambiguous


def test_seat_first_claims_once():
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    baby = BBox(x_min=0.6, y_min=0.4, x_max=0.95, y_max=0.95)
    out = associate_seat_first([("t3", baby), ("t5", baby)], layout)
    owners = [a for a in out if a.seat_id == "child-seat-right" and not a.ambiguous]
    assert len(owners) == 1


def test_empty_layout_never_claims():
    from odc.seats import SeatLayout as _Layout

    empty = _Layout(camera_profile="uncalibrated", version="none", seats=[])
    out = associate_track_first([("t1", BOX_A), ("t2", BOX_B)], empty)
    assert len(out) == 2
    assert all(a.seat_id is None and not a.ambiguous for a in out)


def test_null_pose_never_blocks():
    assert NullPose().estimate(_frame(), [BOX_A]) == {}
    NullPose().close()


def test_map_landmarks_clips_and_flags():
    rel = [(0.5, 0.5, 0.9), (2.0, -1.0, 0.1)]
    kpts = map_landmarks(rel, (100, 100, 300, 300), 1000, 1000)
    assert kpts[0].x == pytest.approx(0.2) and kpts[0].visible
    assert 0.0 <= kpts[1].x <= 1.0 and not kpts[1].visible


def test_mediapipe_missing_model_fails_cleanly():
    from odc.pose import MediaPipePose

    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        MediaPipePose("/nonexistent/model.task")


def test_pose_is_optional_contract():
    assert issubclass(type(NullPose()), contracts.PoseEstimator)
