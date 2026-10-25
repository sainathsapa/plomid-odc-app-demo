"""Analyze + uncalibrated paths (unit parts need no weights/DB)."""

from __future__ import annotations

import subprocess
import sys


def test_analyze_missing_file_fails_clean():
    r = subprocess.run(
        [sys.executable, "-m", "odc.cli", "analyze", "footage/nope.mp4"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 1


def test_analyze_bad_layout_fails_clean():
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "odc.cli",
            "analyze",
            "footage/E1_empty_cabin.mp4",
            "--layout",
            "nope",
            "--max-frames",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 1


def test_build_service_uncalibrated_without_layout():
    from test_service import _StubRepo

    from odc.config import AppConfig, CameraConfig
    from odc.service import build_service

    cfg = AppConfig(
        camera=CameraConfig(camera_id="unknown-cam-xyz", path="footage/E1_empty_cabin.mp4")
    )
    service = build_service(cfg, _StubRepo())
    try:
        assert service.hub.snapshot.calibrated is False
        assert service._layout.seats == []
    finally:
        service.stop()
