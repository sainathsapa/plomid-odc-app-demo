"""Capture integration — real files, real PLOMID, deterministic replay."""

from __future__ import annotations

import subprocess
import sys

import pytest

from odc.capture import CameraHealthWriter, CaptureError, FileReplaySource
from odc.config import CameraConfig
from odc.domain import HealthStatus
from odc.persistence import PlomidRepository

F1 = "footage/F1_family_ride.mp4"
E1 = "footage/E1_empty_cabin.mp4"


def _pass(path: str, limit: int | None = None) -> list[tuple]:
    src = FileReplaySource(CameraConfig(path=path))
    src.open()
    out = []
    while (limit is None or len(out) < limit) and (frame := src.next()) is not None:
        out.append((frame.meta.seq, frame.meta.captured_at.isoformat(), frame.pixels.shape))
    src.close()
    return out


def test_replay_deterministic_first_frames():
    assert _pass(F1, 10) == _pass(F1, 10)
    seqs = [seq for seq, _, _ in _pass(F1, 10)]
    assert seqs == list(range(1, 11))


def test_replay_full_pass_counts_match():
    assert len(_pass(E1)) == len(_pass(E1)) == 389


def test_replay_missing_and_corrupt_fail_cleanly(tmp_path):
    with pytest.raises(CaptureError):
        FileReplaySource(CameraConfig(path="footage/nope.mp4")).open()
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")
    with pytest.raises(CaptureError):
        FileReplaySource(CameraConfig(path=str(bad))).open()


def test_health_writer_bounded_writes_live(plomid_dsn, camera_health, session_meta):
    repo = PlomidRepository(plomid_dsn)
    try:
        repo.ensure_schema_version(2)
        repo.start_session(session_meta)
        writer = CameraHealthWriter(repo, min_interval_s=3600)
        for _ in range(20):
            writer.update(camera_health)
        degraded = camera_health.model_copy(update={"status": HealthStatus.DEGRADED})
        writer.update(degraded)
        writer.flush()
        assert writer.sent == 2 and writer.failed == 0
    finally:
        repo.close()


def test_replay_cli_exits_clean():
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "odc.cli",
            "--config",
            "odc.example.yaml",
            "replay",
            "--max-frames",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr[-500:]
    assert '"frames": 5' in r.stdout
