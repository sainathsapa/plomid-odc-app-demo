"""Service control tests — idle boot, operator start/stop over HTTP."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from odc.config import AppConfig, CameraConfig
from odc.console.app import create_app
from odc.persistence import PlomidRepository
from odc.seats import load_layout


def _app(plomid_dsn):
    repo = PlomidRepository(plomid_dsn)
    repo.ensure_schema_version(2)
    cfg = AppConfig(camera=CameraConfig(path="footage/F1_family_ride.mp4"))
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    return create_app(cfg, repo, layout, autostart=False), repo


def test_boot_is_idle(plomid_dsn):
    app, repo = _app(plomid_dsn)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        status = client.get("/api/service/status").json()
        assert status["running"] is False
        assert "DISCONNECTED" in client.get("/live").text
        assert client.post("/api/service/stop").json() == {"stopped": False, "reason": "idle"}
    finally:
        repo.close()


def test_bad_start_rejected(plomid_dsn):
    app, repo = _app(plomid_dsn)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        assert client.post("/api/service/start", json={"source": "bogus"}).status_code == 400
        assert (
            client.post(
                "/api/service/start",
                json={
                    "source": "replay",
                    "path": "footage/F1_family_ride.mp4",
                    "layout": "nope",
                    "threshold": 0.5,
                },
            ).status_code
            == 400
        )
    finally:
        repo.close()


def test_start_stop_lifecycle(plomid_dsn):
    app, repo = _app(plomid_dsn)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post(
            "/api/service/start",
            json={"source": "replay", "path": "footage/F1_family_ride.mp4", "threshold": 0.9},
        )
        assert r.status_code == 200, r.text
        session_id = r.json()["session_id"]
        assert session_id
        assert client.post("/api/service/start", json={"source": "replay"}).status_code == 409
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if client.get("/api/service/status").json()["running"]:
                break
            time.sleep(0.5)
        assert client.get("/api/service/status").json()["running"] is True
        assert client.post("/api/service/stop").json() == {"stopped": True}
        assert client.get("/api/service/status").json()["running"] is False
    finally:
        repo.close()


def test_live_source_via_file_exercises_live_branch(plomid_dsn):
    app, repo = _app(plomid_dsn)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post(
            "/api/service/start",
            json={"source": "live", "path": "footage/E1_empty_cabin.mp4", "threshold": 0.9},
        )
        assert r.status_code == 200, r.text
        deadline = time.monotonic() + 60
        live_seen = False
        while time.monotonic() < deadline:
            health = client.get("/api/health").json()
            if health["running"] and health["mode"] == "LIVE":
                live_seen = True
                break
            time.sleep(0.5)
        assert live_seen is True
        assert client.post("/api/service/stop").json() == {"stopped": True}
    finally:
        repo.close()
