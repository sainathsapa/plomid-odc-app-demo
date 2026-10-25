"""Browser E2E (Phase 8) — seat-click → inspector over seeded SQL history.

Spins real uvicorn + DSN PLOMID; drives headless Chromium.
Seeds use demo-f1 layout seat ids so map clicks resolve to SQL rows.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import pytest
import uvicorn

pytest.importorskip("playwright.sync_api")

from fastapi import FastAPI

from odc import domain as d
from odc.config import AppConfig, CameraConfig
from odc.console.app import create_app
from odc.persistence import PlomidRepository
from odc.seats import load_layout

TS = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)


def _seeded_repo(dsn):
    repo = PlomidRepository(dsn)
    repo.ensure_schema_version(2)
    repo.start_session(
        d.SessionMeta(
            session_id="ses-e2e",
            camera_id="cabin0",
            source_kind="replay",
            mode=d.RunMode.REPLAY,
            started_at=TS,
            build_version="0.1.0",
            config_version="demo-f1-v0",
        )
    )
    repo.write_state_snapshot(
        [
            d.SeatState(
                session_id="ses-e2e",
                seat_id="mid-adult-center",
                presence=d.PresenceState.LIKELY_OCCUPIED,
                confidence=0.72,
                visibility=d.VisibilityState.PARTIALLY_OCCLUDED,
                reason_codes=["track-overlap", "low-light"],
                updated_at=TS,
                last_positive_at=TS,
                model_set_version="unset",
                config_version="demo-f1-v0",
            )
        ]
    )
    repo.append_event(
        d.StateEvent(
            event_id="evt-e2e-1",
            session_id="ses-e2e",
            event_type=d.EventType.SEAT_PRESENCE_CHANGED,
            severity=d.Severity.INFO,
            seat_id="mid-adult-center",
            occurred_at=TS,
            recorded_at=TS,
            prior_state="UNKNOWN",
            new_state="LIKELY_OCCUPIED",
            reason_codes=["dwell-met"],
            evidence_summary="e2e",
        )
    )
    return repo


@pytest.fixture()
def live_url(plomid_dsn):
    repo = _seeded_repo(plomid_dsn)
    cfg = AppConfig(camera=CameraConfig(path="footage/F1_family_ride.mp4"))
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    app: FastAPI = create_app(cfg, repo, layout, autostart=False)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8099, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    import httpx

    while time.monotonic() < deadline:
        try:
            httpx.get("http://127.0.0.1:8099/live?session=ses-e2e", timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    try:
        yield "http://127.0.0.1:8099"
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        repo.close()


def test_seat_click_shows_sql_evidence(live_url):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(f"{live_url}/seats?session=ses-e2e")
            page.locator('[data-seat="mid-adult-center"]').first.click()
            page.wait_for_function(
                "() => document.getElementById('inspector').textContent.includes('0.72')"
            )
            body = page.locator("#inspector").inner_text()
            assert "LIKELY_OCCUPIED" in body
            assert "track-overlap" in body
        finally:
            browser.close()


def test_timeline_renders_chart(live_url):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(f"{live_url}/timeline?session=ses-e2e")
            page.wait_for_selector("#chart canvas", timeout=15000)
        finally:
            browser.close()
