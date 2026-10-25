"""Capture adapters (Phase 3) — reliable frame plumbing, visible health.

FileReplaySource is deterministic: virtual timestamps derive from file fps,
so the same file+config yields identical (seq, ts, shape) every pass.
Quality signals are ADVISORY ONLY — never persisted as truth.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from . import contracts
from .config import CameraConfig
from .domain import CameraHealth, FrameMeta, HealthStatus, RunMode
from .logging import get_logger

log = get_logger("capture")


class CaptureError(Exception):
    """Raised for fail-fast open/config problems with a clear message."""


@dataclass(frozen=True)
class QualityConfig:
    dark_thresh: float = 30.0
    bright_thresh: float = 225.0
    blur_var_thresh: float = 50.0
    frozen_diff_thresh: float = 2.0


@dataclass(frozen=True)
class FrameQuality:
    brightness: float
    blur_var: float
    dark: bool
    bright: bool
    blurry: bool
    frozen: bool


def assess_quality(
    bgr: np.ndarray, prev_gray_small: np.ndarray | None, cfg: QualityConfig
) -> tuple[FrameQuality, np.ndarray]:
    """Advisory quality signals + gray thumbnail for the next comparison."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 64))
    brightness = float(gray.mean())
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    frozen = False
    if prev_gray_small is not None:
        diff = float(np.abs(small.astype(np.int16) - prev_gray_small.astype(np.int16)).mean())
        frozen = diff < cfg.frozen_diff_thresh
    return (
        FrameQuality(
            brightness=brightness,
            blur_var=blur_var,
            dark=brightness < cfg.dark_thresh,
            bright=brightness > cfg.bright_thresh,
            blurry=blur_var < cfg.blur_var_thresh,
            frozen=frozen,
        ),
        small,
    )


@dataclass
class SourceStats:
    captured: int = 0
    dropped: int = 0
    decode_errors: int = 0
    reconnects: int = 0
    started_monotonic: float = field(default_factory=time.monotonic)

    @property
    def fps_observed(self) -> float:
        elapsed = time.monotonic() - self.started_monotonic
        return self.captured / elapsed if elapsed > 0 else 0.0


class BoundedFrameQueue:
    """Overflow drops OLDEST with a counter — never grows, never blocks."""

    def __init__(self, maxlen: int = 8) -> None:
        self._buf: deque[contracts.Frame] = deque(maxlen=max(1, maxlen))
        self.dropped = 0

    def put(self, frame: contracts.Frame) -> None:
        if len(self._buf) == self._buf.maxlen:
            self.dropped += 1
        self._buf.append(frame)

    def get(self) -> contracts.Frame | None:
        return self._buf.popleft() if self._buf else None

    def __len__(self) -> int:
        return len(self._buf)


class FileReplaySource(contracts.Source):
    kind = "replay"

    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._seq = 0
        self._fps = 0.0
        self._start: datetime = datetime.now(UTC)
        self._ended = False
        self._healthy = False
        self.stats = SourceStats()

    def open(self) -> FrameMeta:
        path = self._cfg.path
        if not Path(path).exists():
            raise CaptureError(f"replay file missing: {path}")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise CaptureError(f"cannot decode replay file: {path}")
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self._seq = 0
        self._ended = False
        self._healthy = True
        self._start = self._resolve_anchor(path)
        log.info("replay open path=%s fps=%.2f", path, self._fps)
        return self._first_meta()

    def _resolve_anchor(self, path: str) -> datetime:
        spec = self._cfg.replay_anchor
        if spec == "file-mtime":
            ts = Path(path).stat().st_mtime
            return datetime.fromtimestamp(ts, tz=UTC)
        try:
            parsed = datetime.fromisoformat(spec)
        except ValueError:
            raise CaptureError(f"bad replay_anchor: {spec!r}") from None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed

    def _first_meta(self) -> FrameMeta:
        cap = self._cap
        assert cap is not None
        return FrameMeta(
            source_id=self._cfg.camera_id,
            seq=0,
            captured_at=self._start,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            mode=RunMode.REPLAY,
        )

    def _ts_for(self, seq: int) -> datetime:
        from datetime import timedelta

        return self._start + timedelta(seconds=seq / self._fps)

    def next(self) -> contracts.Frame | None:
        if self._cap is None or self._ended:
            return None
        skips = 0
        while skips < 10:
            ok, pixels = self._cap.read()
            if not ok or pixels is None:
                # Distinguish EOF (position at end) from decode error.
                pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
                total = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if total > 0 and pos >= total:
                    self._ended = True
                    return None
                self.stats.decode_errors += 1
                skips += 1
                continue
            self._seq += 1
            self.stats.captured += 1
            if self._cfg.realtime:
                time.sleep(max(0.0, (1.0 / self._cfg.target_fps) / self._cfg.speed))
            return contracts.Frame(
                meta=FrameMeta(
                    source_id=self._cfg.camera_id,
                    seq=self._seq,
                    captured_at=self._ts_for(self._seq),
                    width=int(pixels.shape[1]),
                    height=int(pixels.shape[0]),
                    mode=RunMode.REPLAY,
                ),
                pixels=pixels,
            )
        self._healthy = False
        log.warning("replay unhealthy: 10 consecutive decode errors")
        return None

    def health(self) -> CameraHealth:
        return CameraHealth(
            camera_id=self._cfg.camera_id,
            status=HealthStatus.HEALTHY if self._healthy else HealthStatus.UNHEALTHY,
            fps_observed=self._fps,
            failure_code=None if self._healthy else "DECODE_FAILED",
            checked_at=datetime.now(UTC),
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class LiveCameraSource(contracts.Source):
    kind = "live"

    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._seq = 0
        self._status = HealthStatus.UNKNOWN
        self._failure: str | None = None
        self._last_ok = time.monotonic()
        self._start_wall = datetime.now(UTC)
        self.stats = SourceStats()

    def _target(self) -> int | str:
        try:
            return int(self._cfg.path)
        except (TypeError, ValueError):
            return self._cfg.path

    def open(self) -> FrameMeta:
        delay = self._cfg.reconnect_base_s
        last_error = "unknown"
        for attempt in range(1, self._cfg.reconnect_attempts + 1):
            cap = cv2.VideoCapture(self._target())
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    self._cap = cap
                    self._status = HealthStatus.HEALTHY
                    self._failure = None
                    self._last_ok = time.monotonic()
                    log.info("camera open target=%s", self._cfg.path)
                    return FrameMeta(
                        source_id=self._cfg.camera_id,
                        seq=0,
                        captured_at=datetime.now(UTC),
                        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        mode=RunMode.LIVE,
                    )
                cap.release()
            last_error = f"attempt {attempt} failed"
            time.sleep(delay)
            delay *= 2
        self._status = HealthStatus.UNHEALTHY
        self._failure = "OPEN_FAILED"
        raise CaptureError(f"camera open failed after attempts: {last_error}")

    def next(self) -> contracts.Frame | None:
        if self._cap is None:
            return None
        ok, pixels = self._cap.read()
        if not ok or pixels is None:
            self.stats.decode_errors += 1
            self.stats.reconnects += 1
            self._status = HealthStatus.DEGRADED
            self._failure = "READ_FAILED"
            try:
                self._cap.release()
            finally:
                self._cap = None
            return None
        self._seq += 1
        self.stats.captured += 1
        self._last_ok = time.monotonic()
        self._status = HealthStatus.HEALTHY
        self._failure = None
        return contracts.Frame(
            meta=FrameMeta(
                source_id=self._cfg.camera_id,
                seq=self._seq,
                captured_at=datetime.now(UTC),
                width=int(pixels.shape[1]),
                height=int(pixels.shape[0]),
                mode=RunMode.LIVE,
            ),
            pixels=pixels,
        )

    def health(self) -> CameraHealth:
        status, failure = self._status, self._failure
        idle = time.monotonic() - self._last_ok
        if status == HealthStatus.HEALTHY and idle > self._cfg.dead_after_s:
            status, failure = HealthStatus.UNHEALTHY, "STALE"
        elif status == HealthStatus.HEALTHY and idle > self._cfg.stale_after_s:
            status, failure = HealthStatus.DEGRADED, "STALE"
        return CameraHealth(
            camera_id=self._cfg.camera_id,
            status=status,
            fps_observed=self.stats.fps_observed,
            failure_code=failure,
            checked_at=datetime.now(UTC),
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class FakeSource(contracts.Source):
    """Scripted deterministic source for tests/demo. ALWAYS labelled fake."""

    kind = "fake"
    labelled_fake = True

    def __init__(
        self,
        camera_id: str = "fake0",
        n_frames: int = 5,
        shape: tuple[int, int] = (64, 96),
        fail_after: int | None = None,
        mode: RunMode = RunMode.TEST,
    ) -> None:
        self._camera_id = camera_id
        self._n = n_frames
        self._shape = shape
        self._fail_after = fail_after
        self._mode = mode
        self._seq = 0
        self._start = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)

    def open(self) -> FrameMeta:
        self._seq = 0
        return FrameMeta(
            source_id=self._camera_id,
            seq=0,
            captured_at=self._start,
            width=self._shape[1],
            height=self._shape[0],
            mode=self._mode,
        )

    def next(self) -> contracts.Frame | None:
        if self._seq >= self._n:
            return None
        if self._fail_after is not None and self._seq >= self._fail_after:
            return None
        self._seq += 1
        gray = np.full((self._shape[0], self._shape[1], 3), self._seq * 10, dtype=np.uint8)
        from datetime import timedelta

        return contracts.Frame(
            meta=FrameMeta(
                source_id=self._camera_id,
                seq=self._seq,
                captured_at=self._start + timedelta(seconds=self._seq / 5.0),
                width=self._shape[1],
                height=self._shape[0],
                mode=self._mode,
            ),
            pixels=gray,
        )

    def health(self) -> CameraHealth:
        return CameraHealth(
            camera_id=self._camera_id,
            status=HealthStatus.HEALTHY,
            fps_observed=5.0,
            checked_at=datetime.now(UTC),
        )

    def close(self) -> None:
        pass


class CameraHealthWriter:
    """Rate-limited camera-health persistence. Resilient: DB errors are
    logged, never raised; status changes always flush immediately."""

    def __init__(self, repo: contracts.Repository, min_interval_s: float = 5.0) -> None:
        self._repo = repo
        self._min_interval = min_interval_s
        self._last_sent_status: HealthStatus | None = None
        self._last_sent_at = 0.0
        self._pending: CameraHealth | None = None
        self.sent = 0
        self.skipped = 0
        self.failed = 0

    def update(self, health: CameraHealth) -> None:
        now = time.monotonic()
        if health.status != self._last_sent_status:
            self._send(health, now)
            return
        if now - self._last_sent_at >= self._min_interval:
            self._send(health, now)
            return
        self._pending = health
        self.skipped += 1

    def _send(self, health: CameraHealth, now: float) -> None:
        try:
            self._repo.upsert_camera_status(health)
            self._last_sent_status = health.status
            self._last_sent_at = now
            self._pending = None
            self.sent += 1
        except Exception as e:
            self.failed += 1
            self._pending = health
            # Back off: record the attempt so a down DB causes at most one
            # write per min_interval (plus immediate flush on status change),
            # never a per-frame write storm.
            self._last_sent_status = health.status
            self._last_sent_at = now
            log.warning("camera health write failed: %s: %s", type(e).__name__, e)

    def flush(self) -> None:
        if self._pending is not None:
            self._send(self._pending, time.monotonic())
