"""Capture unit tests — no files, no camera, no database."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from odc import contracts
from odc.capture import (
    BoundedFrameQueue,
    CameraHealthWriter,
    CaptureError,
    FakeSource,
    LiveCameraSource,
    QualityConfig,
    assess_quality,
)
from odc.config import CameraConfig
from odc.domain import HealthStatus, RunMode


class _StubRepo(contracts.Repository):
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.writes = 0

    def health_check(self) -> bool:
        return True

    def ensure_schema_version(self, minimum: int) -> int:
        return minimum

    def start_session(self, session: Any) -> None:
        pass

    def end_session(self, session_id: str) -> None:
        pass

    def upsert_camera_status(self, health: Any) -> None:
        if self.fail:
            raise RuntimeError("db down")
        self.writes += 1

    def write_state_snapshot(self, seats: Any) -> None:
        pass

    def write_frame_observation(
        self,
        session_id: str,
        frame_seq: int,
        captured_at: Any,
        seat_rows: Any,
        pose: Any,
        specialist_decisions: Any,
    ) -> None:
        pass

    def append_event(self, event: Any) -> None:
        pass

    def read_current_seat_states(self, session_id: str) -> Any:
        return []

    def read_latest_frame_observation(self, session_id: str) -> Any:
        return None

    def append_assistant_interaction(self, row: Any) -> None:
        pass

    def read_assistant_interactions(self, session_id: str, limit: int = 100) -> Any:
        return []

    def read_event_timeline(self, *a: Any, **k: Any) -> Any:
        return []

    def write_metric_batch(self, rows: Any) -> None:
        pass

    def read_model_manifest(self, model_id: str) -> Any:
        raise AssertionError

    def write_validation_result(self, rows: Any) -> None:
        pass

    # Seat store (V003): nothing stored — reads miss, callers degrade.
    def upsert_video_asset(self, row: Any) -> None:
        pass

    def list_video_assets_by_path(self, video_path: str) -> Any:
        return []

    def upsert_seat_layout(self, name: str, doc: Any) -> None:
        pass

    def load_seat_layout(self, name: str) -> Any:
        return None

    def list_seat_layouts(self) -> Any:
        return []

    def save_video_seatmap(self, row: Any) -> None:
        pass

    def load_video_seatmap(self, video_key: str) -> Any:
        return None

    def list_video_seatmaps_by_path(self, video_path: str) -> Any:
        return []

    def save_probe_entry(self, row: Any) -> None:
        pass

    def load_probe_entry(self, video_key: str) -> Any:
        return None

    def save_analytics_entry(self, row: Any) -> None:
        pass

    def load_analytics_entry(self, cache_key: str) -> Any:
        return None


def test_fake_deterministic_and_labelled():
    first = [(f.meta.seq, f.meta.captured_at.isoformat()) for f in _drain(FakeSource())]
    second = [(f.meta.seq, f.meta.captured_at.isoformat()) for f in _drain(FakeSource())]
    assert first == second and len(first) == 5
    assert FakeSource.labelled_fake is True
    assert FakeSource().health().status == HealthStatus.HEALTHY


def _drain(src: FakeSource) -> list[Any]:
    src.open()
    out = []
    while (frame := src.next()) is not None:
        out.append(frame)
    src.close()
    return out


def test_fake_modes_and_failures():
    assert FakeSource(mode=RunMode.DEMO).open().mode == RunMode.DEMO
    src = FakeSource(n_frames=10, fail_after=3)
    assert len(_drain(src)) == 3


def test_queue_bounded_drops_oldest():
    q = BoundedFrameQueue(maxlen=2)
    src = FakeSource(n_frames=5)
    src.open()
    while (frame := src.next()) is not None:
        q.put(frame)
    assert len(q) == 2 and q.dropped == 3
    assert q.get().meta.seq == 4
    assert q.get().meta.seq == 5
    assert q.get() is None


def test_quality_signals():
    cfg = QualityConfig()
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    q, small = assess_quality(black, None, cfg)
    assert q.dark and not q.bright and q.blurry and not q.frozen
    q2, small2 = assess_quality(white, small, cfg)
    assert q2.bright and not q2.dark
    q3, _ = assess_quality(black, small2, cfg)
    assert q3.dark
    q4, _ = assess_quality(black, small, cfg)
    assert q4.frozen  # identical consecutive frames


def test_live_bogus_fails_fast_without_hang():
    cfg = CameraConfig(
        camera_id="nope",
        source="live",
        path="/nonexistent-xyz",
        reconnect_attempts=1,
        reconnect_base_s=0.01,
    )
    src = LiveCameraSource(cfg)
    with pytest.raises(CaptureError):
        src.open()
    assert src.health().status == HealthStatus.UNHEALTHY
    assert src.health().failure_code == "OPEN_FAILED"
    assert src.next() is None  # no deadlock after failed open


def test_health_writer_rate_limits_and_recovers(camera_health):
    repo = _StubRepo()
    writer = CameraHealthWriter(repo, min_interval_s=3600)
    for _ in range(20):
        writer.update(camera_health)
    assert writer.sent == 1 and writer.skipped == 19
    degraded = camera_health.model_copy(update={"status": HealthStatus.DEGRADED})
    writer.update(degraded)
    assert writer.sent == 2
    writer.flush()
    assert writer.sent == 2 and writer.failed == 0


def test_health_writer_db_failure_never_raises(camera_health):
    writer = CameraHealthWriter(_StubRepo(fail=True), min_interval_s=0)
    writer.update(camera_health)  # must not raise
    assert writer.failed == 1 and writer.sent == 0
