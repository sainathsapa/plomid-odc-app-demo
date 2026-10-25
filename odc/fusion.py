"""Vehicle-signal fusion interface (Phase 13) — bus truth, never camera guesses.

Signals use COVESA VSS names (kuksa.val.v1/...) with a CAN-DBC gateway
fallback documented in operations. Every signal carries source + timestamp;
stale signals are explicit and BLOCK safety-relevant logic (fail-closed).
The camera never overrides the bus.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .logging import get_logger

log = get_logger("fusion")

# VSS topic map (KUKSA.val datapath style). Legacy CAN-DBC gateways translate
# frames into these same names before ODC ever sees them.
VSS_IGNITION = "Vehicle.Powertrain.CombustionEngine.IsRunning"
VSS_DOOR_PREFIX = "Vehicle.Body.Door."
VSS_DOOR_SUFFIX = ".IsOpen"
VSS_LOCKED = "Vehicle.Body.IsLocked"
VSS_BUCKLE_PREFIX = "Vehicle.Cabin.Seat."
VSS_BUCKLE_SUFFIX = ".IsBelted"


@dataclass(frozen=True)
class Signal:
    name: str
    value: Any
    observed_at: datetime
    source: str = "vehicle-bus"

    def fresh(self, now: datetime, stale_after_s: float) -> bool:
        return (now - self.observed_at).total_seconds() <= stale_after_s


class VehicleSignals(ABC):
    """Authorized bus interface. Implementations must timestamp everything."""

    @abstractmethod
    def ignition_on(self) -> Signal | None: ...

    @abstractmethod
    def any_door_open(self) -> Signal | None: ...

    @abstractmethod
    def locked(self) -> Signal | None: ...

    @abstractmethod
    def buckled(self, seat_id: str) -> Signal | None: ...

    @abstractmethod
    def close(self) -> None: ...


class FileReplaySignals(VehicleSignals):
    """Deterministic signal log replay (JSONL: {t, name, value}).

    Timestamps are seconds offset from start; replayed against a wall clock
    anchored at first read. Eval + demo without a vehicle.
    """

    def __init__(self, path: str) -> None:
        self._rows: list[dict[str, Any]] = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                self._rows.append(json.loads(line))
        self._rows.sort(key=lambda r: r["t"])
        self._start = datetime.now(UTC)
        self._latest: dict[str, Signal] = {}

    def _pump(self) -> None:
        now = datetime.now(UTC)
        elapsed = (now - self._start).total_seconds()
        for row in self._rows:
            if row["t"] <= elapsed and row["name"] not in self._latest:
                self._latest[row["name"]] = Signal(
                    row["name"], row["value"], now, source="signal-replay"
                )
            elif row["t"] <= elapsed:
                self._latest[row["name"]] = Signal(
                    row["name"], row["value"], now, source="signal-replay"
                )

    def _get(self, name: str) -> Signal | None:
        self._pump()
        return self._latest.get(name)

    def ignition_on(self) -> Signal | None:
        return self._get(VSS_IGNITION)

    def any_door_open(self) -> Signal | None:
        self._pump()
        doors = [
            s
            for n, s in self._latest.items()
            if n.startswith(VSS_DOOR_PREFIX) and n.endswith(VSS_DOOR_SUFFIX)
        ]
        if not doors:
            return None
        latest = max(doors, key=lambda s: s.observed_at)
        return Signal(
            "any_door_open", any(bool(d.value) for d in doors), latest.observed_at, latest.source
        )

    def locked(self) -> Signal | None:
        return self._get(VSS_LOCKED)

    def buckled(self, seat_id: str) -> Signal | None:
        return self._get(f"{VSS_BUCKLE_PREFIX}{seat_id}{VSS_BUCKLE_SUFFIX}")

    def close(self) -> None:
        pass


@dataclass
class MqttSignals(VehicleSignals):
    """KUKSA.val databroker over MQTT. Client factory injectable (tests use
    a stub; pilots run paho-mqtt against mosquitto — see operations).

    Any transport failure → all reads return None (fail-closed STALE).
    """

    broker_host: str = "127.0.0.1"
    broker_port: int = 1883
    stale_after_s: float = 5.0
    _client: Any = field(default=None, repr=False)
    _cache: dict[str, Signal] = field(default_factory=dict)
    _down: bool = False

    def _connect(self) -> bool:
        if self._client is not None and not self._down:
            return True
        try:
            import paho.mqtt.client as _mqtt  # type: ignore[import-not-found,import-untyped]

            client = _mqtt.Client()
            client.on_message = self._on_message
            client.connect(self.broker_host, self.broker_port, keepalive=10)
            client.subscribe("kuksa/val/v1/#")
            client.loop_start()
            self._client = client
            return True
        except Exception as e:
            log.warning("mqtt unavailable, signals STALE: %s: %s", type(e).__name__, e)
            self._down = True
            return False

    def _on_message(self, _client: Any, _userdata: Any, message: Any) -> None:
        try:
            payload = json.loads(message.payload.decode())
            name = message.topic.replace("kuksa/val/v1/", "Vehicle.", 1)
            self._cache[name] = Signal(name, payload.get("value"), datetime.now(UTC), source="mqtt")
        except Exception as e:
            log.warning("mqtt bad frame dropped: %s", type(e).__name__)

    def _get(self, name: str) -> Signal | None:
        if not self._connect():
            return None
        return self._cache.get(name)

    def ignition_on(self) -> Signal | None:
        return self._get(VSS_IGNITION)

    def any_door_open(self) -> Signal | None:
        if not self._connect():
            return None
        doors = [
            s
            for n, s in self._cache.items()
            if n.startswith(VSS_DOOR_PREFIX) and n.endswith(VSS_DOOR_SUFFIX)
        ]
        if not doors:
            return None
        latest = max(doors, key=lambda s: s.observed_at)
        return Signal(
            "any_door_open", any(bool(d.value) for d in doors), latest.observed_at, latest.source
        )

    def locked(self) -> Signal | None:
        return self._get(VSS_LOCKED)

    def buckled(self, seat_id: str) -> Signal | None:
        return self._get(f"{VSS_BUCKLE_PREFIX}{seat_id}{VSS_BUCKLE_SUFFIX}")

    def close(self) -> None:
        try:
            if self._client is not None:
                self._client.loop_stop()
        except Exception:
            pass
        finally:
            self._client = None
