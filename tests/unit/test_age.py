import math
from dataclasses import replace
from pathlib import Path

import pytest

from odc.age import (
    AgeGenderPrediction,
    MiVOLOV2AgeEstimator,
    _seat_has_dominant_box_overlap,
    apply_age_predictions,
    classify_estimated_age,
)
from odc.domain import BBox


def _prediction(
    track_id: str, seat_id: str, category: str, age_years: float | None = None
) -> AgeGenderPrediction:
    return AgeGenderPrediction(
        track_id=track_id,
        seat_id=seat_id,
        age_years=(age_years if age_years is not None else 5.0 if category == "child" else 35.0),
        age_category=category,
        age_class_confidence=None,
        age_confidence_semantics="point estimate; no probability",
        gender_label="UNKNOWN",
        gender_confidence=None,
        gender_confidence_semantics="uncalibrated",
        face_visible=False,
        person_bbox=(0.1, 0.1, 0.4, 0.8),
        face_bbox=None,
        model_id="mivolo-v2",
        frame_seq=4,
        inference_ms=15.0,
        reason="person-only-model-input",
        age_observation_count=2,
    )


def test_estimated_age_categories_are_explicit_and_ordered():
    assert classify_estimated_age(-1) == "unknown"
    assert classify_estimated_age(math.nan) == "unknown"
    assert classify_estimated_age(1.5) == "infant"
    assert classify_estimated_age(5.0) == "child"
    assert classify_estimated_age(11.0) == "child"
    assert classify_estimated_age(11.01) == "unknown"
    assert classify_estimated_age(17.0) == "unknown"
    assert classify_estimated_age(23.0) == "unknown"
    assert classify_estimated_age(23.01) == "adult"


def test_adult_false_positive_age_estimates_fall_into_unknown_band():
    assert classify_estimated_age(11.16) == "unknown"
    assert classify_estimated_age(16.14) == "unknown"


def test_confirmed_young_child_and_adult_estimates_survive_uncertainty_band():
    assert classify_estimated_age(5.0) == "child"
    assert classify_estimated_age(30.0) == "adult"


def test_face_boxes_are_clipped_to_frame_and_degenerate_boxes_ignored():
    clipped = MiVOLOV2AgeEstimator._face_bbox((-0.05, 0.1, 0.2, 1.2))
    assert clipped is not None
    assert (clipped.x_min, clipped.y_min, clipped.x_max, clipped.y_max) == (
        0.0,
        0.1,
        0.2,
        1.0,
    )
    assert MiVOLOV2AgeEstimator._face_bbox((-0.2, 0.1, -0.1, 0.4)) is None
    assert MiVOLOV2AgeEstimator._face_bbox((0.1, 0.2, 0.3)) is None


def test_track_and_seat_associated_child_survives_application_mapping():
    rows = [{"seat_id": "rear-left", "presence": "OCCUPIED", "track_ids": ["t1"]}]
    apply_age_predictions(rows, {"t1": _prediction("t1", "rear-left", "child")})
    assert rows[0]["child"] == "Yes"
    assert rows[0]["occupant_category"] == "child"
    assert rows[0]["age_predictions"][0]["track_id"] == "t1"


def test_single_frame_child_age_with_adult_conflict_is_unknown():
    rows = [{"seat_id": "rear-left", "presence": "OCCUPIED", "track_ids": ["t1"]}]
    rows[0]["track_ids"].append("t2")
    prediction = replace(_prediction("t1", "rear-left", "child"), age_observation_count=1)

    apply_age_predictions(
        rows,
        {
            "t1": prediction,
            "t2": _prediction("t2", "rear-left", "adult"),
        },
    )

    assert rows[0]["child"] == "Unknown"
    assert rows[0]["child_candidate"] is True
    assert rows[0]["age_predictions"][0]["classification_reason"] == (
        "single-frame-child-estimate-conflicts-with-adult-track"
    )


def test_repeated_child_age_evidence_confirms_child():
    rows = [{"seat_id": "rear-left", "presence": "OCCUPIED", "track_ids": ["t1"]}]
    prediction = replace(_prediction("t1", "rear-left", "child"), age_observation_count=2)

    apply_age_predictions(rows, {"t1": prediction})

    assert rows[0]["child"] == "Yes"


def test_single_child_track_still_improves_recall_without_adult_conflict():
    rows = [{"seat_id": "rear-left", "presence": "OCCUPIED", "track_ids": ["t1"]}]
    prediction = replace(_prediction("t1", "rear-left", "child"), age_observation_count=1)

    apply_age_predictions(rows, {"t1": prediction})

    assert rows[0]["child"] == "Yes"


def test_uncertain_age_keeps_possible_child_evidence_without_confirming_child():
    rows = [{"seat_id": "rear-left", "presence": "OCCUPIED", "track_ids": ["t1"]}]
    prediction = _prediction("t1", "rear-left", "unknown", age_years=14.0)

    apply_age_predictions(rows, {"t1": prediction})

    assert rows[0]["child"] == "Unknown"
    assert rows[0]["child_candidate"] is True
    assert rows[0]["child_candidate_track_ids"] == ["t1"]
    assert rows[0]["child_candidate_reason"] == (
        "apparent-age-point-estimate-within-child-boundary-unconfirmed"
    )


def test_wrong_seat_track_prediction_is_not_leaked():
    rows = [{"seat_id": "rear-left", "presence": "OCCUPIED", "track_ids": ["t1"]}]
    apply_age_predictions(rows, {"t1": _prediction("t1", "rear-right", "child")})
    assert rows[0]["child"] == "Unknown"
    assert rows[0]["occupant_category"] == "unknown"


def test_face_in_neighbor_seat_cannot_confirm_child_for_wrong_track_seat():
    from odc.domain import SeatGeometry
    from odc.seats import SeatLayout

    adult_seat = SeatGeometry(
        seat_id="adult",
        polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
        camera_profile="test",
        version="v1",
    )
    child_seat = SeatGeometry(
        seat_id="child",
        polygon=[(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)],
        camera_profile="test",
        version="v1",
    )
    layout = SeatLayout(camera_profile="test", version="v1", seats=[adult_seat, child_seat])
    prediction = replace(
        _prediction("t1", "adult", "child"),
        face_visible=True,
        face_bbox=(0.6, 0.3, 0.7, 0.5),
    )
    rows = [
        {"seat_id": "adult", "presence": "OCCUPIED", "track_ids": ["t1"]},
        {"seat_id": "child", "presence": "OCCUPIED", "track_ids": []},
    ]
    apply_age_predictions(rows, {"t1": prediction}, layout=layout)
    assert rows[0]["child"] == "Unknown"
    assert rows[0]["age_predictions"][0]["age_category"] == "child"
    assert rows[0]["age_predictions"][0]["classification_category"] == "unknown"
    assert rows[1]["child"] == "Unknown"


def test_child_age_from_person_box_spanning_seats_is_not_assigned_confidently():
    from odc.domain import SeatGeometry
    from odc.seats import SeatLayout

    layout = SeatLayout(
        camera_profile="test",
        version="v1",
        seats=[
            SeatGeometry(
                seat_id=seat_id,
                polygon=polygon,
                camera_profile="test",
                version="v1",
            )
            for seat_id, polygon in (
                ("left", [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]),
                ("right", [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]),
            )
        ],
    )
    prediction = replace(
        _prediction("t1", "left", "child"),
        person_bbox=(0.3, 0.2, 0.7, 0.8),
    )
    rows = [{"seat_id": "left", "presence": "OCCUPIED", "track_ids": ["t1"]}]
    apply_age_predictions(rows, {"t1": prediction}, layout=layout)
    assert rows[0]["child"] == "Unknown"
    assert rows[0]["child_candidate"] is False
    assert rows[0]["age_predictions"][0]["classification_reason"] == (
        "person-box-seat-association-ambiguous"
    )


def test_dominant_seat_overlap_can_preserve_child_age_evidence():
    from odc.domain import SeatGeometry
    from odc.seats import SeatLayout

    layout = SeatLayout(
        camera_profile="test",
        version="v1",
        seats=[
            SeatGeometry(
                seat_id=seat_id,
                polygon=polygon,
                camera_profile="test",
                version="v1",
            )
            for seat_id, polygon in (
                ("left", [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]),
                ("right", [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]),
            )
        ],
    )
    prediction = replace(_prediction("t1", "right", "child"), person_bbox=(0.3, 0.2, 0.8, 0.8))
    rows = [{"seat_id": "right", "presence": "OCCUPIED", "track_ids": ["t1"]}]

    apply_age_predictions(rows, {"t1": prediction}, layout=layout)

    assert rows[0]["child"] == "Yes"
    assert rows[0]["age_predictions"][0]["classification_category"] == "child"


def test_seat_local_age_crop_requires_clear_overlap_dominance():
    from odc.domain import SeatGeometry

    seats = {
        seat.seat_id: seat.polygon
        for seat in (
            SeatGeometry(
                seat_id="adult",
                polygon=[(0.3, 0.0), (0.5, 0.0), (0.5, 1.0), (0.3, 1.0)],
                camera_profile="test",
                version="v1",
            ),
            SeatGeometry(
                seat_id="child",
                polygon=[(0.5, 0.25), (1.0, 0.25), (1.0, 1.0), (0.5, 1.0)],
                camera_profile="test",
                version="v1",
            ),
        )
    }
    child_box = BBox(x_min=0.295, y_min=0.458, x_max=0.776, y_max=0.996)
    merged_adult_box = BBox(x_min=0.203, y_min=0.479, x_max=0.699, y_max=0.996)

    assert _seat_has_dominant_box_overlap(child_box, "child", seats)
    assert not _seat_has_dominant_box_overlap(merged_adult_box, "adult", seats)


def test_missing_age_output_and_empty_restraint_do_not_create_child():
    rows = [
        {"seat_id": "occupied", "presence": "OCCUPIED", "track_ids": [], "restraint": "Yes"},
        {"seat_id": "empty", "presence": "EMPTY_CANDIDATE", "track_ids": [], "restraint": "Yes"},
    ]
    apply_age_predictions(rows, {})
    assert rows[0]["child"] == "Unknown"
    assert rows[0]["occupant_category"] == "unknown"
    assert rows[1]["child"] == "No"
    assert rows[1]["occupant_category"] == "none"


def test_confirmed_child_among_adult_tracks_is_reported_as_mixed_occupancy():
    rows = [{"seat_id": "rear", "presence": "OCCUPIED", "track_ids": ["a", "c"]}]
    apply_age_predictions(
        rows,
        {
            "a": _prediction("a", "rear", "adult"),
            "c": _prediction("c", "rear", "child"),
        },
    )
    assert rows[0]["child"] == "Yes"
    assert rows[0]["occupant_category"] == "mixed"


def test_prediction_serialization_preserves_estimate_semantics():
    serialized = _prediction("t1", "front", "child").as_dict()
    assert serialized["age_class_confidence"] is None
    assert serialized["age_confidence_semantics"] == "point estimate; no probability"
    assert serialized["gender"] == "UNKNOWN"
    assert serialized["age_observation_count"] == 2


def test_reassigned_track_seat_resets_age_history():
    estimator = MiVOLOV2AgeEstimator("models/mivolo-v2")
    estimator._record_age("t1", "front", 5.0, 1)
    category, estimate = estimator._record_age("t1", "rear", 35.0, 2)
    assert category == "adult"
    assert estimate == 35.0
    assert estimator._history["t1"].seat_id == "rear"
    assert list(estimator._history["t1"].ages) == [35.0]


def test_track_return_after_expiry_resets_age_history():
    estimator = MiVOLOV2AgeEstimator("models/mivolo-v2", track_expiry_frames=36)
    estimator._record_age("t1", "front", 5.0, 1)
    category, estimate = estimator._record_age("t1", "front", 35.0, 38)
    assert category == "adult"
    assert estimate == 35.0
    assert list(estimator._history["t1"].ages) == [35.0]


def test_missing_local_checkpoint_fails_closed(tmp_path: Path):
    estimator = MiVOLOV2AgeEstimator(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="MiVOLO V2 checkpoint missing"):
        estimator.load()
