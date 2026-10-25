"""Parked-guardian logic (Phase 13) — forgotten-occupant CANDIDATES only.

Fires ONLY when ALL hold continuously for the dwell: fresh bus signals
showing parked+locked+closed AND a temporally-confirmed OCCUPIED/LIKELY
seat. Anything missing (stale signals, UNKNOWN camera state, condition
break) disarms silently — never an alert. Output is advisory; it actuates
nothing, ever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from .domain import EventType, SeatState, Severity, StateEvent
from .fusion import VehicleSignals
from .logging import get_logger

log = get_logger("guardian")


@dataclass(frozen=True)
class GuardianConfig:
    version: str = "guardian-v1"
    dwell_s: float = 60.0
    stale_after_s: float = 5.0


@dataclass
class Guardian:
    config_version: str = "unset"
    model_set_version: str = "unset"
    rules: GuardianConfig = field(default_factory=GuardianConfig)
    _armed_since: dict[str, datetime] = field(default_factory=dict)
    _alerted: set[str] = field(default_factory=set)

    def update(
        self,
        signals: VehicleSignals,
        seats: list[SeatState],
        session_id: str,
        now: datetime | None = None,
    ) -> list[StateEvent]:
        moment = now or datetime.now(UTC)
        ignition = signals.ignition_on()
        doors = signals.any_door_open()
        locked = signals.locked()
        for signal in (ignition, doors, locked):
            if signal is None or not signal.fresh(moment, self.rules.stale_after_s):
                self._armed_since.clear()  # fail-closed: no fresh context
                return []
        assert ignition is not None and doors is not None and locked is not None
        parked = not bool(ignition.value) and not bool(doors.value) and bool(locked.value)
        if not parked:
            self._armed_since.clear()
            self._alerted.clear()
            return []
        alerts: list[StateEvent] = []
        present = {s.seat_id for s in seats if s.presence.value in ("OCCUPIED", "LIKELY_OCCUPIED")}
        for seat_id in list(self._armed_since):
            if seat_id not in present:
                del self._armed_since[seat_id]
                self._alerted.discard(seat_id)
        for seat_id in present:
            self._armed_since.setdefault(seat_id, moment)
            held = (moment - self._armed_since[seat_id]).total_seconds()
            if held >= self.rules.dwell_s and seat_id not in self._alerted:
                self._alerted.add(seat_id)
                alerts.append(
                    StateEvent(
                        event_id=f"{session_id}-{seat_id}-unattended",
                        session_id=session_id,
                        event_type=EventType.UNATTENDED_OCCUPANT_CANDIDATE,
                        severity=Severity.CRITICAL,
                        seat_id=seat_id,
                        occurred_at=moment,
                        recorded_at=moment,
                        prior_state=None,
                        new_state="UNATTENDED_CANDIDATE",
                        reason_codes=[
                            "parked-locked-closed",
                            "occupied-dwell-met",
                            f"held_s={held:.0f}",
                        ],
                        evidence_summary=(
                            "fusion-gated candidate; requires authorized response path"
                        ),
                        config_version=self.config_version,
                        model_set_version=self.model_set_version,
                    )
                )
        return alerts
