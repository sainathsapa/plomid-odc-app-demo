"""Steering-side proposal tests; no vehicle claim is made by the cue."""

from odc.steering import aggregate_orientation_votes


def test_consistent_wheel_votes_propose_side():
    result = aggregate_orientation_votes([("RHD", 0.8), ("RHD", 0.9), ("RHD", 0.7), ("UNKNOWN", 0.0)])
    assert result.orientation == "RHD"
    assert result.confidence > 0.0
    assert result.positive_frames == 3


def test_ambiguous_wheel_votes_fail_closed():
    result = aggregate_orientation_votes([("RHD", 0.8), ("LHD", 0.9), ("RHD", 0.7)])
    assert result.orientation == "UNKNOWN"
    assert "ambiguous" in result.reason


def test_no_wheel_evidence_is_unknown():
    result = aggregate_orientation_votes([("UNKNOWN", 0.0), ("UNKNOWN", 0.0)])
    assert result.orientation == "UNKNOWN"
    assert result.positive_frames == 0
