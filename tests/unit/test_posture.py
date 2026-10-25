from __future__ import annotations

from odc.pose import classify_seated_posture


def _skeleton(
    shoulder: tuple[float, float], hip: tuple[float, float], visible: bool = True
) -> list[tuple[float, float, float]]:
    points = [(0.5, 0.5, 0.9) for _ in range(33)]
    points[11] = (shoulder[0] - 0.05, shoulder[1], 0.9 if visible else 0.1)
    points[12] = (shoulder[0] + 0.05, shoulder[1], 0.9 if visible else 0.1)
    points[23] = (hip[0] - 0.05, hip[1], 0.9)
    points[24] = (hip[0] + 0.05, hip[1], 0.9)
    return points


def test_visible_torso_does_not_claim_posture_without_seat_reference() -> None:
    status, reason, support = classify_seated_posture(
        _skeleton((0.5, 0.4), (0.5, 0.7)), 1920, 1080
    )

    assert status == "UNKNOWN"
    assert reason == "seat-relative-posture-reference-unavailable"
    assert support is True


def test_occluded_or_missing_torso_abstains() -> None:
    occluded = classify_seated_posture(
        _skeleton((0.5, 0.4), (0.5, 0.7), visible=False), 1920, 1080
    )

    assert occluded[0] == "BODY_PART_OCCLUDED"
    assert classify_seated_posture([], 1920, 1080)[0] == "UNKNOWN"
