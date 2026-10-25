"""Capture real console screenshots for docs/demos (Phase 12).

Seeds one session via the real repository, serves the app, screenshots
live/seats/timeline with headless Chromium. Every pixel traces to SQL.
Usage: ODC_TEST_DSN=... .venv/bin/python scripts/capture_screenshots.py --out docs/odc/assets
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str] | None = None) -> int:
    import uvicorn
    from fastapi import FastAPI

    from odc import domain as d
    from odc.config import AppConfig, CameraConfig
    from odc.console.app import create_app
    from odc.persistence import PlomidRepository
    from odc.seats import load_layout

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/odc/assets")
    parser.add_argument("--port", type=int, default=8098)
    args = parser.parse_args(argv)

    dsn = os.environ.get("ODC_TEST_DSN")
    if not dsn:
        print("ODC_TEST_DSN required")
        return 1
    now = datetime.now(UTC)
    repo = PlomidRepository(dsn)
    repo.ensure_schema_version(2)
    repo.start_session(
        d.SessionMeta(
            session_id="ses-shots",
            camera_id="cabin0",
            source_kind="replay",
            mode=d.RunMode.REPLAY,
            started_at=now,
            build_version="0.1.0",
            config_version="demo-f1-v0",
        )
    )
    for seat_id, presence, conf in (
        ("near-adult-left", d.PresenceState.OCCUPIED, 0.82),
        ("mid-adult-center", d.PresenceState.LIKELY_OCCUPIED, 0.71),
        ("child-seat-right", d.PresenceState.UNKNOWN, 0.0),
    ):
        repo.write_state_snapshot(
            [
                d.SeatState(
                    session_id="ses-shots",
                    seat_id=seat_id,
                    presence=presence,
                    confidence=conf,
                    visibility=d.VisibilityState.VISIBLE,
                    reason_codes=["screenshot-seed"],
                    updated_at=now,
                    config_version="demo-f1-v0",
                )
            ]
        )
    repo.append_event(
        d.StateEvent(
            event_id="evt-shot-1",
            session_id="ses-shots",
            event_type=d.EventType.SEAT_PRESENCE_CHANGED,
            severity=d.Severity.INFO,
            seat_id="near-adult-left",
            occurred_at=now,
            recorded_at=now,
            prior_state="UNKNOWN",
            new_state="OCCUPIED",
            reason_codes=["screenshot-seed"],
            config_version="demo-f1-v0",
        )
    )

    cfg = AppConfig(camera=CameraConfig(path="footage/F1_family_ride.mp4"))
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    app: FastAPI = create_app(cfg, repo, layout, autostart=False)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        from playwright.sync_api import sync_playwright

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1400, "height": 900})
                for name in ("live", "seats", "timeline"):
                    page.goto(f"http://127.0.0.1:{args.port}/{name}?session=ses-shots")
                    page.wait_for_timeout(2500)
                    page.screenshot(path=str(out / f"console-{name}.png"))
                    print(f"saved console-{name}.png")
            finally:
                browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        repo.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
