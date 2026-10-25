"""Live-pipeline unit tests: NMS, cabin filter, attributes, counts. No weights."""

from __future__ import annotations

from datetime import UTC, datetime

from odc.attributes import build_seat_map, live_counts, seat_display_color
from odc.domain import BBox, Detection, PresenceState, SeatState, VisibilityState
from odc.segmentation import (
    apply_per_class_threshold,
    class_aware_nms,
    postprocess_detections,
    suppress_vehicle_hallucinations,
)


def _det(label: str, score: float, box: tuple[float, float, float, float]) -> Detection:
    return Detection(
        class_label=label,
        score=score,
        bbox=BBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
        model_id="m",
        frame_seq=0,
    )


def test_nms_removes_duplicates_keeps_best():
    a = _det("person", 0.9, (0.1, 0.1, 0.5, 0.9))
    b = _det("person", 0.7, (0.12, 0.12, 0.52, 0.88))  # near-duplicate
    c = _det("person", 0.8, (0.6, 0.2, 0.95, 0.9))  # distinct
    out = class_aware_nms([a, b, c], 0.5)
    assert len(out) == 2
    assert out[0].score == 0.9
    # Cross-class boxes never suppress each other.
    chair = _det("chair", 0.85, (0.1, 0.1, 0.5, 0.9))
    assert len(class_aware_nms([a, chair], 0.5)) == 2


def test_per_class_gates_keep_chairs_drop_weak_persons():
    dets = [
        _det("person", 0.35, (0.1, 0.1, 0.4, 0.8)),  # below person floor
        _det("chair", 0.30, (0.6, 0.3, 0.95, 0.9)),  # at chair floor
        _det("person", 0.90, (0.0, 0.0, 0.5, 1.0)),
    ]
    out = apply_per_class_threshold(dets, {"person": 0.5, "chair": 0.25})
    assert {d.class_label for d in out} == {"chair", "person"}
    assert all(d.score >= 0.5 for d in out if d.class_label == "person")


def test_vehicle_hallucination_suppressed_only_when_full_frame():
    full_car = _det("car", 0.62, (0.0, 0.0, 1.0, 1.0))
    small_car = _det("car", 0.62, (0.7, 0.15, 0.85, 0.25))
    kept, n = suppress_vehicle_hallucinations([full_car, small_car])
    assert n == 1
    assert kept == [small_car]


def test_postprocess_stats_and_empty_safe():
    dets = [
        _det("person", 0.9, (0.0, 0.0, 0.5, 1.0)),
        _det("person", 0.85, (0.02, 0.02, 0.52, 0.98)),
        _det("car", 0.65, (0.0, 0.0, 1.0, 1.0)),
    ]
    final, stats = postprocess_detections(dets, {"person": 0.4}, 0.5)
    assert stats["suppressed"] == 1
    assert stats["duplicates"] == 1
    assert len(final) == 1
    final_empty, stats_empty = postprocess_detections([], {"person": 0.4}, 0.5)
    assert final_empty == [] and stats_empty["output"] == 0


PROXY_LABELS = frozenset({"chair", "couch", "suitcase", "handbag", "backpack"})


def _seat(
    seat_id: str,
    presence: PresenceState,
    label: str | None = None,
    proxy: bool | None = None,
) -> SeatState:
    """Mirror engine output: proxy verdict travels as a reason token."""
    now = datetime.now(UTC)
    reasons: list[str] = []
    if label is not None:
        reasons.append(f"object-evidence:{label}")
        is_proxy = label in PROXY_LABELS if proxy is None else proxy
        if is_proxy:
            reasons.append(f"restraint-proxy:{label}")
    return SeatState(
        session_id="s",
        seat_id=seat_id,
        presence=presence,
        confidence=0.7,
        visibility=VisibilityState.VISIBLE,
        reason_codes=reasons,
        object_label=label,
        updated_at=now,
    )


def test_attributes_preserve_all_states_no_overwrite():
    from odc.domain import AttributeState, OccupantState, SizeAppearance

    now = datetime.now(UTC)
    occ = OccupantState(
        session_id="s",
        track_id="t1",
        seat_id="A",
        last_bbox=BBox(x_min=0.1, y_min=0.1, x_max=0.4, y_max=0.8),
        last_score=0.9,
        apparent_size=AttributeState(
            value=SizeAppearance.APPARENT_SMALL_OCCUPANT,
            confidence=0.6,
            observed_at=now,
        ),
        updated_at=now,
    )
    rows = build_seat_map([_seat("A", PresenceState.OCCUPIED, "chair")], [occ], frame_seq=7)
    assert len(rows) == 1
    row = rows[0]
    # Occupancy and restraint stay separate; size alone is not child evidence.
    assert row["presence"] == "OCCUPIED"
    assert row["child"] == "Unknown"
    assert row["restraint"] == "Unknown"
    assert row["restraint_candidate"] is True
    assert row["child_seat_occupancy"] == "UNKNOWN"
    assert row["restraint_confidence"] is None
    assert "unavailable" in row["restraint_confidence_semantics"]
    assert row["belt_confidence"] is None
    assert row["belt"] == "Unknown"
    assert "unsupported" in row["belt_reason"]
    assert row["frame_seq"] == 7


def test_specialist_child_confidence_is_not_seat_occupancy_confidence():
    """The diagnostic 0.82-style presence score cannot become child score."""
    seat = _seat("A", PresenceState.OCCUPIED, "chair")
    seat = seat.model_copy(
        update={
            "confidence": 0.82,
            "reason_codes": [
                "confirm-dwell-met",
                "specialist-child-occupant:child",
                "specialist-score:0.417",
            ],
        }
    )
    row = build_seat_map([seat], [], frame_seq=139)[0]
    assert row["confidence"] == 0.82
    assert row["child"] == "Yes"
    assert row["child_confidence"] == 0.417
    assert row["child_confidence_semantics"] == "grounding-dino-phrase-similarity-uncalibrated"


def test_missing_specialist_output_does_not_fallback_to_adult_or_child():
    seat = _seat("A", PresenceState.OCCUPIED)
    row = build_seat_map([seat], [], frame_seq=139)[0]
    assert row["child"] == "Unknown"
    assert row["occupant_category"] == "unknown"


def test_conflicting_specialist_age_evidence_abstains():
    seat = _seat("A", PresenceState.OCCUPIED)
    seat = seat.model_copy(
        update={
            "reason_codes": [
                "specialist-child-occupant:child",
                "specialist-adult-occupant:adult",
                "specialist-score:0.41",
            ]
        }
    )
    row = build_seat_map([seat], [], frame_seq=139)[0]
    assert row["child"] == "Unknown"
    assert row["child_reason"] == "specialist-age-conflict"
    assert row["occupant_category"] == "unknown"


def test_child_and_adult_predictions_on_distinct_tracks_keep_child_and_score():
    seat = _seat("A", PresenceState.OCCUPIED)
    seat = seat.model_copy(
        update={
            "reason_codes": [
                "specialist-child-occupant:child:track=child-1:score=0.340",
                "specialist-adult-occupant:adult:track=adult-1:score=0.600",
                "specialist-score:0.600",
            ]
        }
    )
    row = build_seat_map([seat], [], frame_seq=139)[0]
    assert row["child"] == "Yes"
    assert row["child_confidence"] == 0.34


def test_attributes_unknown_explicit_and_unique_ids():
    rows = build_seat_map(
        [
            _seat("A", PresenceState.UNKNOWN),
            _seat("A", PresenceState.OCCUPIED),  # duplicate ID: first wins
            _seat("B", PresenceState.EMPTY_CANDIDATE),
        ],
        [],
    )
    assert [r["seat_id"] for r in rows] == ["A", "B"]
    assert rows[0]["presence"] == "UNKNOWN"
    assert rows[0]["child"] == "Unknown"
    assert rows[0]["belt"] == "Unknown"


def test_live_counts_consistent_with_rows():
    rows = build_seat_map(
        [
            _seat("child-seat", PresenceState.OCCUPIED, "chair"),
            _seat("B", PresenceState.EMPTY_CANDIDATE),
            _seat("C", PresenceState.UNKNOWN),
        ],
        [],
    )
    counts = live_counts(rows)
    assert counts["total"] == 3
    assert counts["empty"] == 1
    assert counts["empty_candidates"] == 1
    assert counts["occupied"] == 1
    assert counts["unknown"] == 1
    assert counts["restraints"] == 1
    assert counts["children"] == 0
    assert counts["possible_children"] == 0
    assert counts["belt_unknown"] == 3
    assert counts["belt_fastened"] == 0 and counts["belt_unfastened"] == 0


def test_possible_child_count_is_separate_from_confirmed_child_count():
    rows = build_seat_map([_seat("rear", PresenceState.OCCUPIED)], [])
    rows[0]["child"] = "Unknown"
    rows[0]["child_candidate"] = True

    counts = live_counts(rows)

    assert counts["children"] == 0
    assert counts["possible_children"] == 1


def test_seat_display_color_legend():
    assert seat_display_color({"presence": "EMPTY_CANDIDATE"}) == "#22c55e"
    assert seat_display_color({"presence": "OCCUPIED"}) == "#ef4444"
    assert seat_display_color({"presence": "OCCUPIED", "child": "Yes"}) == "#3b82f6"
    assert seat_display_color({"presence": "UNKNOWN"}) == "#9ca3af"


def test_general_threshold_is_minimum_for_every_class():
    from odc.segmentation import apply_per_class_threshold

    dets = [
        _det("person", 0.39, (0.1, 0.1, 0.4, 0.8)),
        _det("person", 0.65, (0.1, 0.1, 0.4, 0.8)),
        _det("person", 0.90, (0.1, 0.1, 0.4, 0.8)),
        _det("chair", 0.26, (0.6, 0.3, 0.9, 0.9)),
        _det("chair", 0.60, (0.6, 0.3, 0.9, 0.9)),
        _det("toilet", 0.65, (0.2, 0.2, 0.5, 0.5)),
    ]
    kept_at_default = apply_per_class_threshold(dets, floor=0.7)
    assert [(d.class_label, d.score) for d in kept_at_default] == [
        ("person", 0.90)
    ]
    kept = apply_per_class_threshold(dets, {"person": 0.4}, floor=0.25)
    assert {(d.class_label, d.score) for d in kept} == {
        ("person", 0.65), ("person", 0.90), ("chair", 0.26),
        ("chair", 0.60), ("toilet", 0.65),
    }


def test_service_gates_apply_general_minimum_to_cabin_sensitive_classes():
    from odc.config import AppConfig, DetectionConfig
    from odc.seats import SeatLayout
    from odc.service import Service

    layout = SeatLayout(camera_profile="t", version="t", seats=[])
    strict = Service(AppConfig(detection=DetectionConfig(threshold=0.7)), None, layout)  # type: ignore[arg-type]
    strict_gates = strict._per_class_gates()
    assert strict_gates["person"] == 0.7
    assert strict_gates["chair"] == 0.7
    soft = Service(AppConfig(detection=DetectionConfig(threshold=0.25)), None, layout)  # type: ignore[arg-type]
    gates = soft._per_class_gates()
    assert gates["person"] == 0.35  # cabin-specific floor still applies
    assert gates["chair"] == 0.25


def test_low_score_person_and_chair_evidence_is_filtered_at_general_threshold():
    from odc.config import AppConfig, DetectionConfig
    from odc.seats import SeatLayout
    from odc.service import Service

    layout = SeatLayout(camera_profile="t", version="t", seats=[])
    service = Service(
        AppConfig(detection=DetectionConfig(threshold=0.7)), None, layout  # type: ignore[arg-type]
    )
    detections = [
        _det("person", 0.39, (0.1, 0.1, 0.3, 0.6)),
        _det("chair", 0.26, (0.6, 0.3, 0.9, 0.9)),
    ]
    final, _stats = postprocess_detections(
        detections, service._per_class_gates(), floor=0.7
    )
    assert final == []


def test_annotate_detections_child_tag_and_ambiguity():
    from odc.attributes import annotate_detections
    from odc.domain import BBox, Detection
    from odc.seats import SeatGeometry, SeatLayout

    layout = SeatLayout(
        camera_profile="t",
        version="t",
        seats=[
            SeatGeometry(
                seat_id="child-seat",
                polygon=[(0, 0), (0.5, 0), (0.5, 1), (0, 1)],
                camera_profile="t",
                version="t",
            ),
            SeatGeometry(
                seat_id="adult",
                polygon=[(0.5, 0), (1, 0), (1, 1), (0.5, 1)],
                camera_profile="t",
                version="t",
            ),
        ],
    )

    def _d(label: str, box: tuple[float, float, float, float]) -> Detection:
        x0, y0, x1, y1 = box
        return Detection(
            class_label=label,
            score=0.9,
            bbox=BBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1),
            model_id="m",
            frame_seq=0,
        )

    rows = [
        {"seat_id": "child-seat", "child": "Yes"},
        {"seat_id": "adult", "child": "No"},
    ]
    in_child = _d("person", (0.05, 0.1, 0.4, 0.9))
    straddle = _d("person", (0.35, 0.0, 0.65, 1.0))
    chair = _d("chair", (0.05, 0.2, 0.3, 0.9))
    out = annotate_detections([in_child, chair], layout, rows)
    assert out[0][1] == "child-seat" and out[0][2] == "child candidate"
    assert out[1][1] == "child-seat" and out[1][2] is None  # objects: seat, no child tag
    tied = annotate_detections([straddle], layout, rows, ambiguity_margin=0.95)
    assert tied[0][1] is None and tied[0][2] is None  # never force a wrong seat
    assert annotate_detections([], layout, rows) == []


def test_seat_name_does_not_create_restraint_evidence():
    rows = build_seat_map(
        [_seat("child-seat-right", PresenceState.OCCUPIED)], [], frame_seq=3
    )
    assert rows[0]["restraint"] == "Unknown"
    assert rows[0]["restraint_candidate"] is False


def test_child_designated_occupied_does_not_report_child_without_specialist():
    # A configured child-seat region does not establish the occupant's age.
    rows = build_seat_map([_seat("child-seat-right", PresenceState.OCCUPIED)], [], frame_seq=3)
    row = rows[0]
    assert row["child"] == "Unknown"
    assert row["child_reason"] == "no-age-evidence"
    # Plain seats with the same evidence stay Unknown — no guessing.
    rows = build_seat_map([_seat("rear-left", PresenceState.OCCUPIED)], [], frame_seq=3)
    assert rows[0]["child"] == "Unknown"


def test_large_box_without_specialist_age_evidence_stays_unknown():
    from odc.domain import AttributeState, OccupantState, SizeAppearance

    now = datetime.now(UTC)
    occ = OccupantState(
        session_id="s",
        track_id="t9",
        seat_id="child-seat-right",
        last_bbox=BBox(x_min=0.0, y_min=0.0, x_max=0.9, y_max=1.0),
        last_score=0.9,
        apparent_size=AttributeState(
            value=SizeAppearance.APPARENT_LARGE_OCCUPANT,
            confidence=0.6,
            observed_at=now,
        ),
        updated_at=now,
    )
    rows = build_seat_map([_seat("child-seat-right", PresenceState.OCCUPIED)], [occ])
    assert rows[0]["child"] == "Unknown"
    assert rows[0]["child_reason"] == "no-age-evidence"


def test_restraint_proxy_matrix():
    # Generic object classes can surface review candidates, never confirmed
    # child restraints or empty/not-empty labels.
    rows = build_seat_map([_seat("mid", PresenceState.OCCUPIED, "suitcase")], [])
    assert rows[0]["restraint"] == "Unknown"
    assert rows[0]["restraint_candidate"] is True
    # Proxy + EMPTY plain seat -> Unknown (usually the vehicle seat).
    rows = build_seat_map([_seat("seat-right", PresenceState.EMPTY_CANDIDATE, "chair")], [])
    assert rows[0]["restraint"] == "Unknown"
    assert rows[0]["restraint_candidate"] is True
    # Proxy + UNKNOWN plain seat -> Unknown until occupancy resolves.
    rows = build_seat_map([_seat("seat-right", PresenceState.UNKNOWN, "chair")], [])
    assert rows[0]["restraint"] == "Unknown"
    assert "candidate" in rows[0]["restraint_reason"]
    # Seat key is never evidence, even when it contains child/restraint.
    rows = build_seat_map([_seat("child-seat", PresenceState.OCCUPIED, "chair")], [])
    assert rows[0]["restraint"] == "Unknown"
    assert rows[0]["restraint_candidate"] is True
    rows = build_seat_map([_seat("child-seat", PresenceState.EMPTY_CANDIDATE, "chair")], [])
    assert rows[0]["restraint"] == "Unknown"
    assert rows[0]["child_seat_occupancy"] == "UNKNOWN"
    # Small purses (proxy=False) never confirm, even when occupied.
    rows = build_seat_map([_seat("mid", PresenceState.OCCUPIED, "handbag", proxy=False)], [])
    assert rows[0]["restraint"] == "Unknown"
    assert rows[0]["restraint_candidate"] is False
    # No local model can establish absence, even after empty-seat evidence.
    rows = build_seat_map([_seat("driver", PresenceState.EMPTY_CANDIDATE)], [])
    assert rows[0]["restraint"] == "Unknown"
    assert rows[0]["restraint_candidate"] is False
    assert rows[0]["child"] == "No"


def test_restraint_proxy_with_small_occupant_confirms():
    from odc.domain import AttributeState, OccupantState, SizeAppearance

    now = datetime.now(UTC)
    occ = OccupantState(
        session_id="s",
        track_id="t1",
        seat_id="mid",
        last_bbox=BBox(x_min=0.1, y_min=0.1, x_max=0.4, y_max=0.8),
        last_score=0.9,
        apparent_size=AttributeState(
            value=SizeAppearance.APPARENT_SMALL_OCCUPANT,
            confidence=0.6,
            observed_at=now,
        ),
        updated_at=now,
    )
    rows = build_seat_map([_seat("mid", PresenceState.OCCUPIED, "suitcase")], [occ])
    assert rows[0]["restraint"] == "Unknown"
    assert rows[0]["restraint_candidate"] is True
    assert rows[0]["child"] == "Unknown"


def test_occupant_category_and_placement():
    from odc.domain import AttributeState, OccupantState, SizeAppearance

    now = datetime.now(UTC)

    def _occ(track, seat, size, box):
        return OccupantState(
            session_id="s",
            track_id=track,
            seat_id=seat,
            last_bbox=box,
            last_score=0.9,
            apparent_size=AttributeState(value=size, confidence=0.6, observed_at=now)
            if size is not None
            else None,
            updated_at=now,
        )

    tiny = BBox(x_min=0.6, y_min=0.3, x_max=0.75, y_max=0.45)  # area 0.0225
    big = BBox(x_min=0.0, y_min=0.0, x_max=0.9, y_max=1.0)
    rows = build_seat_map(
        [
            _seat("infant-seat", PresenceState.OCCUPIED, "chair"),
            _seat("driver", PresenceState.OCCUPIED),
            _seat("rear", PresenceState.EMPTY_CANDIDATE),
        ],
        [
            _occ("t1", "infant-seat", SizeAppearance.APPARENT_SMALL_OCCUPANT, tiny),
            _occ("t2", "driver", SizeAppearance.APPARENT_LARGE_OCCUPANT, big),
        ],
    )
    by_id = {r["seat_id"]: r for r in rows}
    # Small + restraint is not enough to infer infant/child.
    assert by_id["infant-seat"]["occupant_category"] == "unknown"
    assert by_id["infant-seat"]["child_placement"] == "n/a"
    assert by_id["infant-seat"]["restraint"] == "Unknown"
    assert by_id["infant-seat"]["restraint_candidate"] is True
    # Box size cannot prove adult age; without specialist evidence stay unknown.
    assert by_id["driver"]["occupant_category"] == "unknown"
    assert by_id["driver"]["child_placement"] == "n/a"
    assert by_id["driver"]["child"] == "Unknown"
    # Empty seat -> none.
    assert by_id["rear"]["occupant_category"] == "none"
    assert by_id["rear"]["child"] == "No"


def test_counts_partition_and_belt_buckets():
    rows = build_seat_map(
        [
            _seat("a", PresenceState.OCCUPIED),
            _seat("b", PresenceState.LIKELY_OCCUPIED),
            _seat("c", PresenceState.EMPTY_CANDIDATE),
            _seat("d", PresenceState.UNKNOWN),
        ],
        [],
    )
    rows[0]["belt"] = "Detected"
    rows[1]["belt"] = "Not detected"
    rows[1]["belt_issue"] = True
    counts = live_counts(rows)
    # Invariant: every seat counted exactly once.
    assert counts["total"] == counts["occupied"] + counts["empty"] + counts["unknown"]
    assert counts["total"] == 4 and counts["determined"] == 3
    assert counts["belt_detected"] == 1
    assert counts["belt_not_detected"] == 1
    assert counts["belt_unknown"] == 2
    assert counts["belt_issue"] == 1
    assert counts["belt_fastened"] == 0 and counts["belt_unfastened"] == 0


def test_posture_serialization_is_explicitly_unknown_without_validated_classifier():
    row = build_seat_map([_seat("driver", PresenceState.OCCUPIED)], [])[0]
    assert row["posture"] == "UNKNOWN"
    assert row["posture_reason"] == "no-calibrated-posture-classifier"


def test_analytics_percentages_and_notes():
    from odc.attributes import analytics_summary

    rows = build_seat_map(
        [
            _seat("a", PresenceState.OCCUPIED),
            _seat("b", PresenceState.EMPTY_CANDIDATE),
            _seat("c", PresenceState.UNKNOWN),
        ],
        [],
    )
    counts = live_counts(rows)
    analytics = analytics_summary(rows, counts)
    assert analytics["percentages"]["occupied"] == 33.3
    assert analytics["percentages"]["empty_candidate"] == 33.3
    assert analytics["percentages"]["determined"] == 66.7
    assert any("c: occupancy unresolved" in n for n in analytics["notes"])
    assert any("a: belt indeterminate" in n for n in analytics["notes"])
    assert len(analytics["notes"]) <= 5


def test_display_colors_uncertain_and_occluded():
    assert seat_display_color({"presence": "LIKELY_OCCUPIED"}) == "#eab308"
    assert (
        seat_display_color({"presence": "OCCUPIED", "visibility": "OCCLUDED"}) == "#9ca3af"
    )
    assert (
        seat_display_color({"presence": "OCCUPIED", "visibility": "CAMERA_UNHEALTHY"})
        == "#9ca3af"
    )


def test_saved_relative_child_candidate_has_no_fabricated_confidence():
    now = datetime.now(UTC)
    saved_state = SeatState(
        session_id="s",
        seat_id="rear",
        presence=PresenceState.OCCUPIED,
        confidence=0.73,
        visibility=VisibilityState.VISIBLE,
        reason_codes=["child-candidate:relative-cabin-size"],
        updated_at=now,
    )

    row = build_seat_map([saved_state], [], frame_seq=118)[0]

    assert row["child"] == "Unknown"
    assert row["child_reason"] == "no-age-evidence"
    assert row["child_confidence"] is None
