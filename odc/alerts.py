"""Rule-based cabin alerts (candidate-grade, cooldown-gated).

Alerts DERIVE from seat rows, DMS flags and session context — they never
create detections. Every alert carries severity, seat, reason and frame
provenance, and every kind says *candidate* where the evidence is
camera-only. Raised/cleared transitions at WARNING+ persist as odc_event
rows (existing table, new event types); the live list is transient and
scoped to one video (hub.reset clears it).

Kinds:
- belt (WARNING): occupied seat, belt Not-detected candidate.
- drowsy / distracted / phone (WARNING): DMS flags, preview-grade.
- unattended (WARNING): occupant continuously present past the timer
  WITHOUT vehicle-bus context — always worded "verify with door/ignition
  signals (not connected)". The CRITICAL fusion-gated path stays in
  guardian.py and requires parked+locked+closed bus signals.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .logging import get_logger

log = get_logger("alerts")

UNATTENDED_AFTER_S = 600.0
COOLDOWN_S = 60.0
HISTORY_MAX = 50


@dataclass
class Alert:
    alert_id: str
    severity: str  # INFO | WARNING | CRITICAL
    kind: str
    seat_id: str | None
    message: str
    frame_seq: int
    raised_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity,
            "kind": self.kind,
            "seat_id": self.seat_id,
            "message": self.message,
            "frame_seq": self.frame_seq,
            "raised_at": self.raised_at.isoformat(),
        }


def _dms_flags(row: dict[str, Any]) -> list[str]:
    dms = row.get("dms") or {}
    flags = []
    if dms.get("drowsy"):
        flags.append("drowsy")
    if dms.get("distracted"):
        flags.append("distracted")
    if dms.get("phone"):
        flags.append("phone")
    return flags


class AlertManager:
    """Per-video alert state. Fresh instance per session."""

    def __init__(
        self,
        unattended_after_s: float = UNATTENDED_AFTER_S,
        cooldown_s: float = COOLDOWN_S,
    ) -> None:
        self._unattended_after_s = unattended_after_s
        self._cooldown_s = cooldown_s
        self._active: dict[str, Alert] = {}
        self._cleared_at: dict[str, datetime] = {}
        self._occ_since: dict[str, datetime] = {}

    def update(
        self,
        rows: list[dict[str, Any]],
        session_id: str,
        frame_seq: int,
        now: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Returns (active, raised, cleared) as JSON-ready dicts."""
        moment = now or datetime.now(UTC)
        wanted: dict[str, Alert] = {}
        for row in sorted(rows, key=lambda r: str(r.get("seat_id", ""))):
            sid = str(row.get("seat_id", "?"))
            occupied = row.get("presence") in ("OCCUPIED", "LIKELY_OCCUPIED")
            if row.get("belt_issue") is True and occupied:
                wanted[f"belt:{sid}"] = Alert(
                    f"{session_id}-belt-{sid}-{frame_seq}", "WARNING", "belt", sid,
                    f"{sid}: no seat belt in clear view (candidate) — remind occupant",
                    frame_seq, moment,
                )
            for flag in _dms_flags(row):
                wanted[f"dms-{flag}:{sid}"] = Alert(
                    f"{session_id}-dms-{flag}-{sid}-{frame_seq}", "WARNING",
                    f"dms-{flag}", sid,
                    f"{sid}: {flag} driving behavior (preview candidate) — confirm visually",
                    frame_seq, moment,
                )
            if occupied:
                self._occ_since.setdefault(sid, moment)
                held = (moment - self._occ_since[sid]).total_seconds()
                if held >= self._unattended_after_s:
                    wanted[f"unattended:{sid}"] = Alert(
                        f"{session_id}-unattended-{sid}-{frame_seq}", "WARNING",
                        "unattended", sid,
                        f"{sid}: occupant present {held:.0f}s with engine running — "
                        "verify with door/ignition signals (not connected)",
                        frame_seq, moment,
                    )
            else:
                self._occ_since.pop(sid, None)
        raised: list[Alert] = []
        for key, alert in wanted.items():
            if key not in self._active:
                cleared = self._cleared_at.get(key)
                if cleared is not None and (moment - cleared).total_seconds() < self._cooldown_s:
                    continue
                self._active[key] = alert
                raised.append(alert)
        cleared_alerts: list[Alert] = []
        for key in [k for k in self._active if k not in wanted]:
            cleared_alerts.append(self._active.pop(key))
            self._cleared_at[key] = moment
        active = [self._active[k].to_dict() for k in sorted(self._active)]
        return (
            active,
            [a.to_dict() for a in raised],
            [
                {
                    "alert_id": a.alert_id,
                    "severity": a.severity,
                    "kind": a.kind,
                    "seat_id": a.seat_id,
                    "frame_seq": frame_seq,
                }
                for a in cleared_alerts
            ],
        )


class WebhookSender:
    """Best-effort alert webhook. Never blocks the service loop: bounded
    queue (oldest dropped), daemon sender thread, short timeout, all
    failures swallowed into logs."""

    def __init__(self, url: str, timeout_s: float = 3.0) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=20)
        self._thread = threading.Thread(target=self._pump, name="odc-webhook", daemon=True)
        self.sent = 0
        self.dropped = 0
        self.failed = 0

    def start(self) -> None:
        self._thread.start()

    def send(self, payload: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                pass
            self.dropped += 1

    def _pump(self) -> None:
        import json as _json
        import urllib.request as _request

        while True:
            try:
                payload = self._queue.get()
            except Exception:
                return
            try:
                req = _request.Request(
                    self._url,
                    data=_json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with _request.urlopen(req, timeout=self._timeout_s):
                    self.sent += 1
            except Exception as e:
                self.failed += 1
                log.warning("webhook failed: %s", type(e).__name__)
