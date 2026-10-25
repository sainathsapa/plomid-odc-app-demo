"""Validation metrics unit tests — math + determinism, no weights."""

from __future__ import annotations

import cv2
import numpy as np

from odc.validate import box_iou, compare_reports, match_boxes, prf, run_dataset, score_seat


def test_iou_identity_partial_disjoint():
    assert box_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0
    assert abs(box_iou([0, 0, 2, 2], [1, 1, 3, 3]) - 1 / 7) < 1e-9


def test_match_greedy_tp_fp_fn():
    tp, fp, fn, mp, mg = match_boxes(
        [[0, 0, 1, 1], [2, 2, 3, 3]],
        [[0, 0, 1, 1], [5, 5, 6, 6]],
        0.5,
    )
    assert (tp, fp, fn) == (1, 1, 1)
    assert mp == [0] and mg == [0]


def test_prf_zeros_safe():
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
    assert prf(2, 1, 0)["precision"] == 2 / 3


def test_three_way_never_wrong_on_abstain():
    assert score_seat("occupied", "OCCUPIED") == "correct"
    assert score_seat("occupied", "EMPTY_CANDIDATE") == "wrong"
    assert score_seat("occupied", "UNKNOWN") == "abstain"
    assert score_seat("empty", "STALE") == "abstain"
    assert score_seat("empty", "EMPTY_CANDIDATE") == "correct"


def test_compare_detects_regression():
    old = {
        "metrics": {"person_f1": 0.5, "seat_correct_rate": 0.8, "seat_unknown_rate": 0.1},
        "gates": [{"gate": "g1", "passed": True}],
    }
    new = {
        "metrics": {"person_f1": 0.4, "seat_correct_rate": 0.8, "seat_unknown_rate": 0.1},
        "gates": [{"gate": "g1", "passed": False}],
    }
    diff = compare_reports(old, new)
    assert diff["metric_deltas"]["person_f1"] == -0.1
    assert diff["regressed"] is True
    assert compare_reports(old, old)["regressed"] is False


def _tiny_clip(path: str) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 64))
    for shade in (0, 255):
        writer.write(np.full((64, 64, 3), shade, dtype=np.uint8))
    writer.release()


def test_run_dataset_deterministic(tmp_path):
    clip = str(tmp_path / "tiny.mp4")
    _tiny_clip(clip)
    dataset = {
        "clips": {"T": clip},
        "acceptance": {
            "person_precision_dev": 0,
            "person_recall_dev": 0,
            "seat_correct_dev": 0,
            "seat_unknown_rate_max": 1,
        },
        "holdout_policy": "test",
    }
    annotations = {
        "splits": {"dev": ["T"]},
        "layouts": {},
        "frames": [
            {"clip": "T", "idx": 0, "persons": [], "seats": {}},
            {"clip": "T", "idx": 1, "persons": [], "seats": {}},
        ],
    }

    def detect(bgr):
        return [("person", 0.9, [0.1, 0.1, 0.9, 0.9])] if bgr.mean() > 100 else []

    first = run_dataset(dataset, annotations, "dev", detect)
    second = run_dataset(dataset, annotations, "dev", detect)
    stable = {k: v for k, v in first["metrics"].items() if "latency" not in k}
    assert stable == {k: v for k, v in second["metrics"].items() if "latency" not in k}
    assert first["metrics"]["person_fp"] == 1  # bright frame false positive
    assert first["metrics"]["person_fn"] == 0
    assert all(g["passed"] for g in first["gates"])
