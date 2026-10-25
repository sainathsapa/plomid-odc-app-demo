"""Real Service-path belt-visibility regression on the reviewed S2 clip."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "footage/S2_belt_up.mp4"
ANNOTATIONS = ROOT / "eval/annotations/local-s2-belt-review-2026-10-24.json"


@pytest.mark.skipif(not VIDEO.exists(), reason="S2 belt footage is unavailable")
def test_service_detects_reviewed_strap_without_false_presence_on_absent_frames(
    tmp_path: Path,
) -> None:
    output = tmp_path / "s2-belt-evaluation.json"
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
            "configs/seats/demo-s2-v0.yaml",
            "--occupied-seats",
            "driver",
            "--annotations",
            str(ANNOTATIONS),
            "--disable-age-model",
            "--sample-every-n",
            "12",
            "--scenario",
            "integration-reviewed-s2-belt-visibility",
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
    visible = [row for row in rows if row.get("ground_truth_belt") == "visible"]
    absent = [row for row in rows if row.get("ground_truth_belt") == "absent-assessable"]

    assert len(visible) == 51
    assert len(absent) == 3
    assert any(row["belt"] == "Detected" for row in visible)
    assert all(row["belt"] != "Detected" for row in absent)
