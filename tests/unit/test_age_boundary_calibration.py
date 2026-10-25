import json

from scripts.analyze_age_boundary_calibration import _select_development_margin, analyze


def _report(path, video, age, seat, label, track="t1", reason="person-only-model-input"):
    path.write_text(
        json.dumps(
            {
                "video": video,
                "ground_truth": {f"{label}_seats": [seat]},
                "frames": [
                    {
                        "rows": [
                            {
                                "seat_id": seat,
                                "age_predictions": [
                                    {
                                        "seat_id": seat,
                                        "track_id": track,
                                        "age_years": age,
                                        "classification_reason": reason,
                                    }
                                ],
                            }
                        ]
                    },
                    {
                        "rows": [
                            {
                                "seat_id": seat,
                                "age_predictions": [
                                    {
                                        "seat_id": seat,
                                        "track_id": track,
                                        "age_years": age,
                                        "classification_reason": reason,
                                    }
                                ],
                            }
                        ]
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_analysis_aggregates_frames_by_track_and_keeps_unknown_band(tmp_path):
    child = tmp_path / "child.json"
    adult = tmp_path / "adult.json"
    _report(child, "child clip", 5.0, "rear", "child")
    _report(adult, "adult clip", 30.0, "rear", "adult")

    result = analyze([child, adult])

    assert result["independent_video_count"] == 2
    assert result["track_counts"] == {"child": 1, "adult": 1}
    assert result["current_policy"]["child_tracks"] == 1
    assert result["current_policy"]["adult_tracks"] == 1
    assert result["current_policy"]["unknown_tracks"] == 0
    assert result["current_policy"]["child_confirmed_max_years_inclusive"] == 11.0
    assert result["current_policy"]["adult_confirmed_min_years_exclusive"] == 23.0


def test_analysis_excludes_single_observation_and_ambiguous_associations(tmp_path):
    report = tmp_path / "report.json"
    _report(report, "clip", 5.0, "rear", "child", track="single")
    data = json.loads(report.read_text(encoding="utf-8"))
    ambiguous = json.loads(json.dumps(data["frames"][0]))
    ambiguous["rows"][0]["age_predictions"][0].update(
        track_id="ambiguous", classification_reason="person-box-seat-association-ambiguous"
    )
    data["frames"] = [data["frames"][0], ambiguous]
    report.write_text(json.dumps(data), encoding="utf-8")

    result = analyze([report])

    assert result["track_counts"] == {"child": 0, "adult": 0}
    assert result["tracks"] == []


def test_development_margin_uses_extrema_and_explicit_clearance():
    tracks = [
        {"ground_truth": "child", "median_age_years": 6.84},
        {"ground_truth": "adult", "median_age_years": 24.355},
    ]

    selected = _select_development_margin(tracks, semantic_boundary_years=17.0)

    assert selected is not None
    assert selected["available_margin_before_rounding_years"] == 6.355
    assert selected["selected_margin_years"] == 6.0
    assert selected["child_confirmed_max_years_inclusive"] == 11.0
    assert selected["adult_confirmed_min_years_exclusive"] == 23.0
