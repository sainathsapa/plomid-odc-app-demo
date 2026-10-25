"""Production-path regression for child recall and the F1 adult false positive."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "footage/F1_family_ride.mp4"
CHECKPOINT = ROOT / "models/mivolo-v2/model.safetensors"


@pytest.mark.skipif(
    not VIDEO.exists() or not CHECKPOINT.exists(),
    reason="F1 video or MiVOLO V2 checkpoint is unavailable",
)
def test_production_service_preserves_child_recall_and_abstains_on_spanning_adult(
    tmp_path: Path,
) -> None:
    """Run the live Service and prevent the known cross-seat age false positive."""
    output = tmp_path / "f1-age-evaluation.json"
    env = dict(os.environ)
    env.pop("NO_COLOR", None)
    env["PYTHONPATH"] = str(ROOT)
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl-cache")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_unseen_child_video.py"),
            str(VIDEO),
            "--layout",
            "configs/seats/demo-f1-v0.yaml",
            "--child-seats",
            "child-seat-right",
            "--adult-seats",
            "mid-adult-center",
            "near-adult-left",
            "--occupied-seats",
            "child-seat-right",
            "mid-adult-center",
            "near-adult-left",
            "--sample-every-n",
            "12",
            "--age-every-n",
            "1",
            "--scenario",
            "integration-current-F1-cross-seat-age-regression",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=240,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    rows = [row for frame in report["frames"] for row in frame["rows"]]
    child_rows = [row for row in rows if row["ground_truth"] == "child"]
    adult_rows = [row for row in rows if row["ground_truth"] == "adult"]

    assert len(child_rows) == 28
    assert sum(row["child"] == "Yes" for row in child_rows) >= 23
    assert sum(row["child"] == "Unknown" for row in child_rows) <= 5
    assert all(row["child"] != "Yes" for row in adult_rows)
    assert len(adult_rows) == 56

    spanning_adult = next(
        row
        for frame in report["frames"]
        if frame["video_frame_index"] == 325
        for row in frame["rows"]
        if row["seat_id"] == "mid-adult-center"
    )
    assert spanning_adult["ground_truth"] == "adult"
    assert spanning_adult["child"] == "Unknown"
    assert spanning_adult["age_predictions"]
    assert spanning_adult["age_predictions"][0]["age_years"] == pytest.approx(4.05, abs=0.2)
    assert (
        spanning_adult["age_predictions"][0]["classification_reason"]
        == "person-box-seat-association-ambiguous"
    )
    recovered_child = next(
        row
        for frame in report["frames"]
        if frame["video_frame_index"] == 313
        for row in frame["rows"]
        if row["seat_id"] == "child-seat-right"
    )
    assert recovered_child["ground_truth"] == "child"
    assert recovered_child["child"] == "Yes"
    assert recovered_child["age_predictions"][0]["inference_bbox"][0] == pytest.approx(0.5)
