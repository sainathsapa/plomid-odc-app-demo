"""Real production-path regression for infant and adult in one ROI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "footage/C1_baby_carseat.mp4"
CHECKPOINT = ROOT / "models/mivolo-v2/model.safetensors"


@pytest.mark.skipif(
    not VIDEO.exists() or not CHECKPOINT.exists(),
    reason="C1 video or MiVOLO V2 checkpoint is unavailable",
)
def test_service_keeps_infant_when_adult_handles_child_seat(tmp_path: Path) -> None:
    output = tmp_path / "c1-evaluation.json"
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
            "configs/seats/demo-c1-v0.yaml",
            "--child-seats",
            "infant-carrier",
            "--scenario",
            "development-infant-with-adult-handling-carrier",
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
    rows = [
        row
        for frame in report["frames"]
        for row in frame["rows"]
        if row["seat_id"] == "infant-carrier"
    ]
    assert rows
    assert any(row["child"] == "Yes" for row in rows)
    assert all(row["child"] in {"Yes", "Unknown", "No"} for row in rows)
    assert any(
        item.get("model_id") == "mivolo-v2" and item.get("age_category") in {"child", "infant"}
        for row in rows
        for item in row.get("age_predictions", [])
    )
