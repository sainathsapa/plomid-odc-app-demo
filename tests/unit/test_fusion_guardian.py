"""Phase 13 unit tests — fusion freshness, guardian gating, fleet reads."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from odc.domain import EventType, PresenceState, SeatState, VisibilityState
from odc.fusion import VSS_LOCKED, FileReplaySignals, MqttSignals
from odc.guardian import Guardian, GuardianConfig

T0 = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)


def _seat(presence: PresenceState) -> SeatState:
    return SeatState(
        session_id="s",
        seat_id="rear-left",
        presence=presence,
        confidence=0.8,
        visibility=VisibilityState.VISIBLE,
        updated_at=T0,
        last_positive_at=T0,
    )


def _signals_toml(tmp_path, rows: str):
    path = tmp_path / "signals.jsonl"
    path.write_text(rows)
    return FileReplaySignals(str(path))


def test_file_replay_signals_parked(tmp_path):
    from odc.fusion import VSS_DOOR_PREFIX, VSS_DOOR_SUFFIX, VSS_IGNITION

    rows = (
        f'{{"t": 0, "name": "{VSS_IGNITION}", "value": false}}\n'
        f'{{"t": 0, "name": "{VSS_DOOR_PREFIX}front{VSS_DOOR_SUFFIX}", "value": false}}\n'
        f'{{"t": 0, "name": "{VSS_LOCKED}", "value": true}}\n'
    )
    signals = _signals_toml(tmp_path, rows)
    assert signals.ignition_on() is not None and signals.ignition_on().value is False
    assert signals.any_door_open() is not None and signals.any_door_open().value is False
    assert signals.locked() is not None and signals.locked().value is True
    assert signals.buckled("rear-left") is None  # unknown signal, not invented
    signals.close()


def test_guardian_alerts_only_when_all_hold(tmp_path):
    from odc.fusion import VSS_DOOR_PREFIX, VSS_DOOR_SUFFIX, VSS_IGNITION

    parked = (
        f'{{"t": 0, "name": "{VSS_IGNITION}", "value": false}}\n'
        f'{{"t": 0, "name": "{VSS_DOOR_PREFIX}front{VSS_DOOR_SUFFIX}", "value": false}}\n'
        f'{{"t": 0, "name": "{VSS_LOCKED}", "value": true}}\n'
    )
    guardian = Guardian(
        config_version="cfg", model_set_version="m", rules=GuardianConfig(dwell_s=60)
    )
    signals = _signals_toml(tmp_path, parked)
    assert guardian.update(signals, [_seat(PresenceState.OCCUPIED)], "s", now=T0) == []
    alerts = guardian.update(
        signals, [_seat(PresenceState.OCCUPIED)], "s", now=T0 + timedelta(seconds=61)
    )
    assert len(alerts) == 1
    assert alerts[0].event_type == EventType.UNATTENDED_OCCUPANT_CANDIDATE
    assert alerts[0].severity.value == "CRITICAL"
    # No repeat while still parked.
    assert (
        guardian.update(
            signals, [_seat(PresenceState.OCCUPIED)], "s", now=T0 + timedelta(seconds=120)
        )
        == []
    )


def test_guardian_silent_without_fresh_context(tmp_path):
    from odc.fusion import VSS_LOCKED

    # Only a lock signal, no ignition/door context → incomplete context.
    partial = f'{{"t": 0, "name": "{VSS_LOCKED}", "value": true}}\n'
    signals = _signals_toml(tmp_path, partial)
    guardian = Guardian(rules=GuardianConfig(dwell_s=0))
    assert guardian.update(signals, [_seat(PresenceState.OCCUPIED)], "s", now=T0) == []


def test_guardian_silent_when_unknown_or_unlocked(tmp_path):
    from odc.fusion import VSS_DOOR_PREFIX, VSS_DOOR_SUFFIX, VSS_IGNITION, VSS_LOCKED

    def rows(locked: bool) -> str:
        return (
            f'{{"t": 0, "name": "{VSS_IGNITION}", "value": false}}\n'
            f'{{"t": 0, "name": "{VSS_DOOR_PREFIX}front{VSS_DOOR_SUFFIX}", "value": false}}\n'
            f'{{"t": 0, "name": "{VSS_LOCKED}", "value": {str(locked).lower()}}}\n'
        )

    guardian = Guardian(rules=GuardianConfig(dwell_s=0))
    unlocked = _signals_toml(tmp_path, rows(False))
    assert guardian.update(unlocked, [_seat(PresenceState.OCCUPIED)], "s", now=T0) == []
    guardian2 = Guardian(rules=GuardianConfig(dwell_s=0))
    parked = _signals_toml(tmp_path, rows(True))
    assert guardian2.update(parked, [_seat(PresenceState.UNKNOWN)], "s", now=T0) == []


def test_mqtt_down_means_stale_not_invented():
    signals = MqttSignals(broker_host="127.0.0.1", broker_port=1)
    try:
        assert signals.ignition_on() is None
        assert signals.locked() is None
        assert signals.any_door_open() is None
    finally:
        signals.close()


def test_new_event_types_exist():
    assert EventType.UNATTENDED_OCCUPANT_CANDIDATE.value == "UNATTENDED_OCCUPANT_CANDIDATE"
    assert EventType.FUSION_SIGNAL_STALE.value == "FUSION_SIGNAL_STALE"
