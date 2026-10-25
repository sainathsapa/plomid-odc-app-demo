"""Production-path regression on the locally staged, reviewed C2 child clip."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "footage/C2_child_seat.mp4"
CHECKPOINT = ROOT / "models/mivolo-v2/model.safetensors"


@pytest.mark.skipif(
    not VIDEO.exists() or not CHECKPOINT.exists(),
    reason="C2 video or MiVOLO V2 checkpoint is unavailable",
)
def test_production_service_detects_child_in_c2_video(tmp_path: Path) -> None:
    """Exercise detector, tracking, MiVOLO, seat mapping and serialization."""
    output = tmp_path / "c2-evaluation.json"
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
            "configs/seats/demo-c2-v0.yaml",
            "--child-seats",
            "child-seat",
            "--scenario",
            "development-child-with-adult-handling",
            "--sample-every-n",
            "12",
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
    child_rows = [
        row for frame in report["frames"] for row in frame["rows"] if row["seat_id"] == "child-seat"
    ]
    assert child_rows
    assert any(row["child"] == "Yes" for row in child_rows)
    assert all(row["child"] in {"Yes", "Unknown", "No"} for row in child_rows)
    assert any(
        item.get("model_id") == "mivolo-v2"
        for frame in report["frames"]
        for row in frame["rows"]
        for item in row.get("age_predictions", [])
    )
    child_skeletons = [
        skeleton
        for frame in report["frames"]
        for skeleton in frame.get("pose", {}).get("skeletons", [])
        if skeleton.get("seat") == "child-seat"
    ]
    assert child_skeletons
    assert all(
        0.27 <= skeleton["inference_bbox"]["x_min"]
        and skeleton["inference_bbox"]["x_max"] <= 0.64
        for skeleton in child_skeletons
    )
    assert all(row["posture"] in {"UNKNOWN", "BODY_PART_OCCLUDED"} for row in child_rows)
