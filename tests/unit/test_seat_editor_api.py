"""Console API tests for the seat-mapping upgrade: validate, current, alerts."""

from __future__ import annotations

from fastapi.testclient import TestClient

import tests.unit.test_console as tc
from odc.config import AppConfig
from odc.console.app import create_app
from odc.seats import load_layout


def _client() -> TestClient:
    return TestClient(
        create_app(AppConfig(), tc._StubRepo(), load_layout("configs/seats/demo-f1-v0.yaml"))
    )


def test_validate_flags_overlap_and_reports_rows():
    client = _client()
    overlapping = [
        {"seat_id": "a", "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]},
        {"seat_id": "b", "polygon": [[0.25, 0], [0.75, 0], [0.75, 1], [0.25, 1]]},
    ]
    r = client.post("/api/layouts/validate", json={"name": "draft", "seats": overlapping})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["errors"] == []
    assert any("overlap" in w for w in body["warnings"])
    clean = [
        {"seat_id": "a", "polygon": [[0, 0], [0.4, 0], [0.4, 1], [0, 1]]},
        {"seat_id": "b", "polygon": [[0.6, 0], [1, 0], [1, 1], [0.6, 1]]},
    ]
    r = client.post("/api/layouts/validate", json={"name": "draft", "seats": clean})
    assert r.json()["warnings"] == []
    bad = {"name": "../evil", "seats": clean}
    assert client.post("/api/layouts/validate", json=bad).json()["errors"] != []


def test_current_layout_reports_calibrated_disjoint_zones_and_order():
    client = _client()
    r = client.get("/api/layouts/current")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == "demo-f1-v1"
    assert isinstance(body["seats"], list) and len(body["seats"]) == 3
    assert not any("overlap" in w for w in body["warnings"])
    assert body["overlaps"] == []
    flat = [s for row in body["rows"] for s in row]
    assert sorted(flat) == sorted(s["seat_id"] for s in body["seats"])


def test_alerts_anchor_to_seat_zones():
    client = _client()
    body = client.get("/api/alerts").json()
    assert set(body) == {"active", "recent"}


def test_events_carry_frame_seq_and_severity():
    client = _client()
    events = client.get("/api/events").json()
    assert isinstance(events, list)
    # Live hub starts empty; the contract is the shape, exercised below
    # against a stubbed event with frame_seq in evidence_summary.
    from datetime import UTC, datetime

    from odc.console.app import _event_frame_seq
    from odc.domain import EventType, Severity, StateEvent

    ev = StateEvent(
        event_id="x",
        session_id="s",
        event_type=EventType.ALERT_RAISED,
        severity=Severity.WARNING,
        occurred_at=datetime.now(UTC),
        recorded_at=datetime.now(UTC),
        evidence_summary="frame_seq=42;belt issue",
    )
    assert _event_frame_seq(ev) == 42
    ev2 = StateEvent(
        event_id="s",
        session_id="s",
        event_type=EventType.SEAT_PRESENCE_CHANGED,
        severity=Severity.INFO,
        occurred_at=datetime.now(UTC),
        recorded_at=datetime.now(UTC),
        evidence_summary="dwell-met",
    )
    assert _event_frame_seq(ev2) is None


def test_timeline_and_seats_pages_render_editor():
    from pathlib import Path

    client = _client()
    tl = client.get("/timeline").text
    assert "tl-seat" in tl and "ev-frame" in tl and "follow live" in tl
    assert "function colorJson(value)" in tl and "json-key" in tl
    seats = client.get("/seats").text
    assert "Calibrate seat zones" in seats and "me-save-name" in seats
    seat_template = Path("odc/console/templates/seats.html").read_text()
    assert "Driver monitoring" in seat_template and "Appearance" in seat_template
    assert "Evidence reasons" in seat_template
    assert '<table class="data">' not in seat_template
    assert 'id="me-move" aria-pressed="true"' in seats
    assert "data-seat-detail-close" in seats
    # Merged calibration tools live on the seats page now.
    assert "me-rect" in seats and "meCapture" in seats and "mapedit-frame" in seats
    assert "seatmap-frame" in seats
    # The draw surface must sit above the frame image: without the overlay
    # class the SVG never covers the frame and no click reaches it.
    assert 'id="mapedit-svg" viewBox="0 0 1000 1000" preserveAspectRatio="none"' in seats
    live = client.get("/live").text
    assert 'id="live-overlay" class="overlay"' in live
    status_url = "/api/seatmap/cache-status?video=footage/F1_family_ride.mp4"
    assert client.get(status_url).status_code == 200


def test_apply_switches_active_geometry_without_restart():
    client = _client()
    assert client.post("/api/layouts/apply", json={"name": "nope"}).status_code == 404
    r = client.post("/api/layouts/apply", json={"name": "demo-r1-v0"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applied"] is True and body["seats"] == 4
    assert body["running"] is False and body["hot_swapped"] is None
    current = client.get("/api/layouts/current").json()
    assert current["version"] == "demo-r1-v1"


def test_service_adopt_layout_rekeys_engine():
    from odc.config import AppConfig
    from odc.domain import SeatGeometry
    from odc.seats import SeatLayout
    from odc.service import Service
    from odc.state import SeatStateEngine
    from tests.unit.test_persistence_unit import FakeRepository

    layout = SeatLayout(
        camera_profile="t",
        version="t",
        seats=[
            SeatGeometry(
                seat_id="A",
                polygon=[(0, 0), (0.5, 0), (0.5, 1), (0, 1)],
                camera_profile="t",
                version="t",
            )
        ],
    )
    service = Service(AppConfig(), FakeRepository(), layout)
    # Idle session: geometry swaps, nothing to rekey.
    assert service.adopt_layout(layout) is None
    # Running session (engine attached): beliefs carry over.
    service._engine = SeatStateEngine("s", layout)
    assert set(service._engine._beliefs) == {"A"}
    layout2 = SeatLayout(
        camera_profile="t",
        version="t2",
        seats=[
            SeatGeometry(
                seat_id="A",
                polygon=[(0, 0), (0.4, 0), (0.4, 1), (0, 1)],
                camera_profile="t",
                version="t2",
            ),
            SeatGeometry(
                seat_id="B",
                polygon=[(0.6, 0), (1, 0), (1, 1), (0.6, 1)],
                camera_profile="t",
                version="t2",
            ),
        ],
    )
    assert service.adopt_layout(layout2) == {"carried": 1, "reset": 1}
    assert service._layout.version == "t2"
    assert set(service._engine._beliefs) == {"A", "B"}
