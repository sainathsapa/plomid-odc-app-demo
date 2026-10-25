from __future__ import annotations

from datetime import UTC, datetime

from odc.domain import BBox, Detection, OccupantState, SeatGeometry
from odc.pose import associate_pose_detections
from odc.seats import clip_bbox_to_seat_region


def _detection(label: str, box: BBox, score: float = 0.9) -> Detection:
    return Detection(
        class_label=label,
        score=score,
        bbox=box,
        model_id="test-detector",
        frame_seq=1,
    )


def _occupant(track_id: str, seat_id: str | None, box: BBox) -> OccupantState:
    return OccupantState(
        session_id="session",
        track_id=track_id,
        seat_id=seat_id,
        last_bbox=box,
        last_score=0.9,
        updated_at=datetime.now(UTC),
    )


def test_pose_matches_person_boxes_to_correct_seat_tracks_and_skips_objects() -> None:
    left = BBox(x_min=0.05, y_min=0.1, x_max=0.4, y_max=0.9)
    right = BBox(x_min=0.6, y_min=0.1, x_max=0.95, y_max=0.9)
    occupants = [
        _occupant("left-track", "left-seat", left),
        _occupant("right-track", "right-seat", right),
    ]
    detections = [
        _detection("chair", left),
        _detection("person", right),
        _detection("person", left),
    ]

    associations = associate_pose_detections(detections, occupants)

    assert [(d.class_label, o.track_id, o.seat_id) for d, o in associations] == [
        ("person", "left-track", "left-seat"),
        ("person", "right-track", "right-seat"),
    ]


def test_pose_rejects_unmatched_and_unseated_person_tracks() -> None:
    occupant = _occupant(
        "unseated-track",
        None,
        BBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.6),
    )
    unmatched = _detection(
        "person",
        BBox(x_min=0.7, y_min=0.1, x_max=0.9, y_max=0.6),
    )

    assert associate_pose_detections([unmatched], [occupant]) == []


def test_duplicate_boxes_for_one_track_keep_the_best_overlap() -> None:
    track_box = BBox(x_min=0.2, y_min=0.1, x_max=0.5, y_max=0.8)
    weaker = _detection(
        "person",
        BBox(x_min=0.19, y_min=0.1, x_max=0.51, y_max=0.82),
    )
    stronger = _detection("person", track_box)
    occupant = _occupant("track", "seat", track_box)

    associations = associate_pose_detections([weaker, stronger], [occupant])

    assert len(associations) == 1
    assert associations[0][0] is stronger


def test_pose_match_threshold_is_validated() -> None:
    try:
        associate_pose_detections([], [], minimum_iou=1.1)
    except ValueError as error:
        assert "minimum_iou" in str(error)
    else:
        raise AssertionError("invalid IoU threshold was accepted")


def test_pose_crop_is_clipped_to_associated_seat_region() -> None:
    seat = SeatGeometry(
        seat_id="child-seat",
        polygon=[(0.27, 0.15), (0.64, 0.15), (0.64, 1.0), (0.27, 1.0)],
        camera_profile="c2",
        version="c2-v2",
    )
    merged_person_box = BBox(
        x_min=0.21, y_min=0.21, x_max=0.92, y_max=0.99
    )

    clipped = clip_bbox_to_seat_region(merged_person_box, seat)

    assert clipped == BBox(
        x_min=0.27, y_min=0.21, x_max=0.64, y_max=0.99
    )


def test_pose_crop_without_seat_intersection_is_missing_evidence() -> None:
    seat = SeatGeometry(
        seat_id="left",
        polygon=[(0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)],
        camera_profile="c2",
        version="c2-v2",
    )
    outside = BBox(x_min=0.6, y_min=0.1, x_max=0.9, y_max=0.8)

    assert clip_bbox_to_seat_region(outside, seat) is None
