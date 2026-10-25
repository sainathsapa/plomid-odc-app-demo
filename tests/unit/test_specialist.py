"""Deterministic specialist routing tests without model weights."""

from __future__ import annotations

from odc.domain import BBox
from odc.specialist import SpecialistCandidate, _candidate_label, deduplicate_candidates


def _candidate(label: str, score: float, x0: float = 0.1) -> SpecialistCandidate:
    return SpecialistCandidate(
        label=label,
        score=score,
        bbox=BBox(x_min=x0, y_min=0.1, x_max=x0 + 0.3, y_max=0.5),
        phrase=label,
        frame_seq=10,
    )


def test_candidate_labels_are_bounded_and_separate() -> None:
    assert _candidate_label("a baby in an infant car seat") == "infant seat"
    assert _candidate_label("a child") == "child"
    assert _candidate_label("an adult") == "adult"
    assert _candidate_label("a red bag") is None


def test_duplicate_prompts_keep_highest_candidate_without_merging_attributes() -> None:
    candidates = deduplicate_candidates(
        [_candidate("child seat", 0.42), _candidate("child seat", 0.71), _candidate("child", 0.60)]
    )
    assert [(item.label, item.score) for item in candidates] == [
        ("child seat", 0.71),
        ("child", 0.60),
    ]


def test_hyphenated_phrases_route_to_infant_seat() -> None:
    assert _candidate_label("rear - facing infant seat") == "infant seat"
    assert _candidate_label("forward - facing child") == "child"
