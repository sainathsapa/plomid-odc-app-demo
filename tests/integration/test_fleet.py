"""Phase 13 integration — fleet rollup + criticals from real SQL."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from odc import domain as d
from odc.config import AppConfig, CameraConfig
from odc.console.app import create_app
from odc.persistence import PlomidRepository
from odc.seats import load_layout

TS = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)


def _seed(plomid_dsn):
    repo = PlomidRepository(plomid_dsn)
    repo.ensure_schema_version(2)
    for sid in ("ses-a", "ses-b"):
        repo.start_session(
            d.SessionMeta(
                session_id=sid,
                camera_id="cabin0",
                source_kind="replay",
                mode=d.RunMode.REPLAY,
                started_at=TS,
                build_version="0.1.0",
            )
        )
    repo.write_state_snapshot(
        [
            d.SeatState(
                session_id="ses-a",
                seat_id="driver",
                presence=d.PresenceState.OCCUPIED,
                confidence=0.9,
                visibility=d.VisibilityState.VISIBLE,
                updated_at=TS,
            )
        ]
    )
    repo.write_state_snapshot(
        [
            d.SeatState(
                session_id="ses-b",
                seat_id="driver",
                presence=d.PresenceState.UNKNOWN,
                confidence=0.0,
                visibility=d.VisibilityState.VISIBLE,
                updated_at=TS,
            )
        ]
    )
    repo.append_event(
        d.StateEvent(
            event_id="evt-crit-1",
            session_id="ses-a",
            event_type=d.EventType.UNATTENDED_OCCUPANT_CANDIDATE,
            severity=d.Severity.CRITICAL,
            seat_id="driver",
            occurred_at=TS,
            recorded_at=TS,
            new_state="UNATTENDED_CANDIDATE",
            reason_codes=["parked-locked-closed"],
            config_version="cfg",
        )
    )
    return repo


def test_fleet_page_and_api(plomid_dsn):
    repo = _seed(plomid_dsn)
    try:
        cfg = AppConfig(camera=CameraConfig(path="footage/F1_family_ride.mp4"))
        layout = load_layout("configs/seats/demo-f1-v0.yaml")
        app = create_app(cfg, repo, layout, autostart=False)
        client = TestClient(app, raise_server_exceptions=False)
        assert client.get("/fleet").status_code == 200
        body = client.get("/api/fleet").json()
        assert {s["session_id"] for s in body["sessions"]} >= {"ses-a", "ses-b"}
        assert len(body["critical_events"]) == 1
        assert body["critical_events"][0]["event_type"] == "UNATTENDED_OCCUPANT_CANDIDATE"
        assert "ses-a" in client.get("/fleet").text
    finally:
        repo.close()


def test_list_sessions_and_criticals(plomid_dsn):
    repo = _seed(plomid_dsn)
    try:
        sessions = repo.list_sessions()
        assert {s["session_id"] for s in sessions} >= {"ses-a", "ses-b"}
        criticals = repo.recent_critical_events()
        assert [e.event_id for e in criticals] == ["evt-crit-1"]
    finally:
        repo.close()
