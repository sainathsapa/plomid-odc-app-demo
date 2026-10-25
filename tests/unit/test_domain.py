"""Domain contract tests — serialization round-trips + validation gates."""

from __future__ import annotations

from datetime import UTC

import pytest
from pydantic import ValidationError

from odc import domain as d
from odc.domain import DOMAIN_VERSION


def test_enums_cover_uncertainty():
    assert d.PresenceState.UNKNOWN and d.PresenceState.STALE
    assert d.VisibilityState.OCCLUDED and d.VisibilityState.CAMERA_UNHEALTHY
    assert d.SeatbeltState.UNSUPPORTED and d.SeatbeltState.UNKNOWN
    assert d.RestraintState.UNSUPPORTED and d.RestraintState.NOT_DETECTED
    assert d.PostureState.UNKNOWN


def test_no_age_field_names():
    assert d.SizeAppearance.APPARENT_SMALL_OCCUPANT.value.startswith("APPARENT_")
    assert not hasattr(d.SizeAppearance, "AGE")


def test_bbox_rejects_out_of_range():
    with pytest.raises(ValidationError):
        d.BBox(x_min=-0.1, y_min=0, x_max=2.0, y_max=1.0)


def test_confidence_bounds(frame_meta, seat_state):
    data = seat_state.model_dump(mode="json")
    data["confidence"] = 1.5
    with pytest.raises(ValidationError):
        d.SeatState.model_validate(data)
    assert frame_meta.domain_version == DOMAIN_VERSION


def test_round_trip_detection(detection):
    raw = detection.model_dump_json()
    back = d.Detection.model_validate_json(raw)
    assert back == detection


def test_round_trip_seat_state(seat_state):
    back = d.SeatState.model_validate_json(seat_state.model_dump_json())
    assert back == seat_state
    assert back.reason_codes == ["track-overlap", "low-light"]


def test_round_trip_event(state_event):
    back = d.StateEvent.model_validate_json(state_event.model_dump_json())
    assert back == state_event
    assert back.occurred_at <= back.recorded_at


def test_attribute_state_generic():
    from datetime import datetime

    a = d.AttributeState[d.SeatbeltState](
        value=d.SeatbeltState.UNKNOWN,
        confidence=0.0,
        observed_at=datetime.now(UTC),
        reason="no-evidence-yet",
    )
    back = d.AttributeState[d.SeatbeltState].model_validate_json(a.model_dump_json())
    assert back.value == d.SeatbeltState.UNKNOWN


def test_manifest_defaults_unapproved():
    m = d.ModelManifest(
        model_id="m",
        family="rtdetrv2",
        checkpoint_sha256="unset",
        source_url="https://example.invalid",
        code_revision="unset",
        weights_license="REVIEW_REQUIRED",
        dataset_terms="REVIEW_REQUIRED",
    )
    assert m.approved_for_commercial_use is False


def test_extra_fields_forbidden(detection):
    with pytest.raises(ValidationError):
        d.Detection.model_validate({**detection.model_dump(), "vendor_logit": 1.0})
