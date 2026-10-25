from scripts.evaluate_unseen_child_video import summarize_child_candidate_metrics


def test_possible_child_candidate_is_reported_separately_from_confirmed_detection() -> None:
    rows = [
        {"ground_truth": "child", "child": "Yes", "child_candidate": True},
        {"ground_truth": "child", "child": "Unknown", "child_candidate": True},
        {"ground_truth": "child", "child": "Unknown", "child_candidate": False},
        {"ground_truth": "adult", "child": "No", "child_candidate": False},
        {"ground_truth": "adult", "child": "Unknown", "child_candidate": True},
    ]

    metrics = summarize_child_candidate_metrics(rows)

    assert metrics["child"] == {
        "frames": 3,
        "candidate_yes": 2,
        "candidate_no": 1,
        "candidate_rate": 2 / 3,
        "confirmed_child_yes": 1,
        "possible_unconfirmed": 1,
    }
    assert metrics["adult"]["candidate_yes"] == 1
    assert metrics["adult"]["confirmed_child_yes"] == 0
