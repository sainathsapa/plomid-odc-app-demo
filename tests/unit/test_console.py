"""Console unit tests — SVG, badges, dossier. No server, no weights."""

from __future__ import annotations

from datetime import UTC

from odc.config import AppConfig
from odc.console.app import _badge, seat_svg
from odc.domain import HealthStatus, RunMode
from odc.dossier import build_dossier, render_markdown
from odc.seats import load_layout
from odc.service import ServiceHub, ServiceSnapshot


def _hub(**kw) -> ServiceHub:
    hub = ServiceHub()
    kw.setdefault("db_healthy", True)
    kw.setdefault("model_healthy", True)
    hub.snapshot = ServiceSnapshot(**kw)
    return hub


def test_badge_matrix():
    assert _badge(_hub(), None) == "DISCONNECTED"
    assert (
        _badge(
            _hub(
                running=True, session_id="s", db_healthy=True, model_healthy=True, mode=RunMode.LIVE
            ),
            None,
        )
        == "LIVE"
    )
    assert _badge(_hub(running=True, session_id="s", mode=RunMode.REPLAY), None) == "REPLAY"
    assert _badge(_hub(running=True, session_id="s", db_healthy=False), None) == "DEGRADED"
    assert _badge(_hub(running=True, session_id="s", model_healthy=False), None) == "DEGRADED"
    assert (
        _badge(_hub(running=True, session_id="s", camera_status=HealthStatus.UNHEALTHY), None)
        == "DEGRADED"
    )
    assert _badge(_hub(), "ses-history") == "REPLAY"


def test_seat_svg_states_and_accessibility():
    from datetime import datetime

    from odc.domain import PresenceState, SeatState, VisibilityState

    layout = load_layout("configs/seats/demo-r1-v0.yaml")
    svg = seat_svg(layout, {})
    assert svg.count("<polygon") == 4
    assert "tabindex" in svg and "<title>" in svg
    assert "#9ca3af" in svg  # UNKNOWN gray (live legend: gray unknown)
    now = datetime.now(UTC)
    states = {
        "rear-left": SeatState(
            session_id="s",
            seat_id="rear-left",
            presence=PresenceState.OCCUPIED,
            confidence=0.9,
            visibility=VisibilityState.VISIBLE,
            updated_at=now,
        )
    }
    svg = seat_svg(layout, states)
    assert "#ef4444" in svg and 'data-seat="rear-left"' in svg  # occupied red
    # Empty is green, child-occupied is blue, restraint adds a purple ring —
    # no attribute overwrites another.
    empty_states = {
        s.seat_id: SeatState(
            session_id="s",
            seat_id=s.seat_id,
            presence=PresenceState.EMPTY_CANDIDATE,
            confidence=0.6,
            visibility=VisibilityState.VISIBLE,
            updated_at=now,
        )
        for s in layout.seats
    }
    assert "#22c55e" in seat_svg(layout, empty_states)
    child_rows = {
        s.seat_id: {
            "presence": "OCCUPIED",
            "child": "Yes",
            "restraint": "Unknown",
        }
        for s in layout.seats
    }
    assert "#3b82f6" in seat_svg(layout, child_rows)
    restraint_rows = {
        s.seat_id: {
            "presence": "OCCUPIED",
            "child": "Unknown",
            "restraint": "Yes",
        }
        for s in layout.seats
    }
    assert "#a855f7" in seat_svg(layout, restraint_rows)
    candidate_rows = {
        s.seat_id: {
            "presence": "OCCUPIED",
            "child": "Unknown",
            "restraint": "Unknown",
            "restraint_candidate": True,
        }
        for s in layout.seats
    }
    candidate_svg = seat_svg(layout, candidate_rows)
    assert "#a855f7" in candidate_svg
    assert 'stroke-dasharray="12 8"' in candidate_svg
    assert "restraint-candidate" in candidate_svg


def test_vehicle_svg_cabin_schematic():
    from datetime import datetime

    from odc.console.app import vehicle_svg
    from odc.domain import PresenceState, SeatState, VisibilityState

    layout = load_layout("configs/seats/demo-r1-v0.yaml")
    svg = vehicle_svg(layout, {})
    assert "FRONT" in svg and "REAR" in svg
    assert svg.count('class="seat"') == 4  # glyphs, not raw zone polygons
    assert 'points="150,95 450,95 405,175 195,175"' in svg  # windshield
    assert "no calibrated geometry" not in svg
    now = datetime.now(UTC)
    states = {
        "rear-left": SeatState(
            session_id="s",
            seat_id="rear-left",
            presence=PresenceState.OCCUPIED,
            confidence=0.9,
            visibility=VisibilityState.VISIBLE,
            updated_at=now,
        )
    }
    svg = vehicle_svg(layout, states)
    assert "#ef4444" in svg  # occupied red glyph
    svg = seat_svg(layout, states, view="bogus")
    assert "FRONT" in svg  # unknown view falls back to schematic
    svg = seat_svg(layout, states)  # default stays true polygons
    assert svg.count("<polygon") == 4
    empty = load_layout("configs/seats/demo-f1-v0.yaml")
    svg = vehicle_svg(type(empty)(camera_profile="u", version="none", seats=[]), {})
    assert "no calibrated geometry" in svg


def test_standard_cabin_schematics_show_distinct_one_based_rows():
    from odc.console.app import vehicle_svg

    five = load_layout("configs/seats/standard-5-seat-rhd.yaml")
    seven = load_layout("configs/seats/standard-7-seat-rhd.yaml")
    five_svg = vehicle_svg(five, {})
    seven_svg = vehicle_svg(seven, {})
    assert "ROW 1" in five_svg and "ROW 2" in five_svg
    assert all(f"Seat {n}" in five_svg for n in range(1, 6))
    assert [five_svg.count(f'data-seat="seat_{n}"') for n in range(1, 6)] == [1] * 5
    assert all(f"ROW {n}" in seven_svg for n in range(1, 4))
    assert all(f"Seat {n}" in seven_svg for n in range(1, 8))
    assert "seat_0" not in five_svg + seven_svg


def test_partials_and_seats_support_views():
    from fastapi.testclient import TestClient

    from odc.console.app import create_app

    cfg = AppConfig()
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    client = TestClient(create_app(cfg, _StubRepo(), layout, autostart=False))
    car = client.get("/partials/seat-map?view=car").text
    poly = client.get("/partials/seat-map?view=poly").text
    assert "FRONT" in car and "FRONT" not in poly
    assert client.get("/seats?view=poly").status_code == 200
    assert "Schematic" in client.get("/seats").text


def test_dossier_honest_gaps():
    cfg = AppConfig()
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    dossier = build_dossier(cfg, layout, None, [], [], 0)
    assert "NOT production-certified" in dossier["mode"]
    assert any("breathing" in gap for gap in dossier["known_gaps"])
    md = render_markdown(dossier)
    assert md.startswith("# PLOMID ODC") and "Known gaps" in md


class _StubRepo:
    """Dependency-free repository stub for console unit tests.

    Seat-store methods are faithful in-memory dicts (layouts persist,
    video maps/probes/analytics round-trip), so save/start/cache-status
    paths exercise real logic without a database.
    """

    def __init__(self) -> None:
        from odc.service import ServiceHub

        self.hub = ServiceHub()
        self.layouts: dict = {}
        self.assets: dict = {}
        self.video_seatmaps: dict = {}
        self.probes: dict = {}
        self.analytics: dict = {}
        self.assistant_interactions: list[dict] = []

    def health_check(self):
        return True

    def ensure_schema_version(self, minimum):
        return minimum

    def start_session(self, session):
        pass

    def end_session(self, session_id):
        pass

    def upsert_camera_status(self, health):
        pass

    def write_state_snapshot(self, seats):
        pass

    def append_event(self, event):
        pass

    def read_current_seat_states(self, session_id):
        return []

    def read_event_timeline(self, *a, **k):
        return []

    def append_assistant_interaction(self, row):
        self.assistant_interactions.append(dict(row))

    def read_assistant_interactions(self, session_id, limit=100):
        rows = [row for row in self.assistant_interactions if row["session_id"] == session_id]
        return sorted(rows, key=lambda row: row["occurred_at"], reverse=True)[:limit]

    def write_metric_batch(self, rows):
        pass

    def read_model_manifest(self, model_id):
        raise AssertionError("no manifest in stub")

    def write_validation_result(self, rows):
        pass

    def upsert_video_asset(self, row):
        self.assets[row["video_key"]] = dict(row)

    def list_video_assets_by_path(self, video_path):
        return [a for a in self.assets.values() if a.get("video_path") == video_path]

    def upsert_seat_layout(self, name, doc):
        self.layouts[name] = dict(doc)

    def load_seat_layout(self, name):
        return self.layouts.get(name)

    def list_seat_layouts(self):
        return [
            {
                "layout_name": n,
                "camera_profile": d.get("camera_profile", ""),
                "version": d.get("version", n),
                "origin": d.get("origin", "hand"),
            }
            for n, d in sorted(self.layouts.items())
        ]

    def save_video_seatmap(self, row):
        self.video_seatmaps[row["video_key"]] = dict(row)

    def load_video_seatmap(self, video_key):
        return self.video_seatmaps.get(video_key)

    def list_video_seatmaps_by_path(self, video_path):
        return [
            {
                "video_key": m["video_key"],
                "video_sha": m.get("video_sha", ""),
                "layout_version": m.get("layout_version", ""),
                "origin": m.get("origin", ""),
                "video_path": video_path,
            }
            for m in self.video_seatmaps.values()
            if self.assets.get(m["video_key"], {}).get("video_path") == video_path
        ]

    def save_probe_entry(self, row):
        self.probes[row["video_key"]] = dict(row)

    def load_probe_entry(self, video_key):
        return self.probes.get(video_key)

    def save_analytics_entry(self, row):
        self.analytics[row["cache_key"]] = dict(row)

    def load_analytics_entry(self, cache_key):
        return self.analytics.get(cache_key)


def test_scope_guard_assistant_interaction_is_saved_for_history():
    from fastapi.testclient import TestClient

    from odc.console.app import create_app

    cfg = AppConfig()
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    repo = _StubRepo()
    hub = ServiceHub()
    hub.snapshot = ServiceSnapshot(session_id="chat-session")
    client = TestClient(create_app(cfg, repo, layout, hub=hub, autostart=False))

    response = client.post("/api/assistant/ask", json={"prompt": "What is the capital of France?"})

    assert response.status_code == 200
    assert response.json()["provider"] == "plomid-scope-guard"
    interaction_id = response.json()["interaction_id"]
    history = client.get("/api/assistant/history?session=chat-session").json()
    assert history["api_schema_version"] == "odc.assistant-history.v1"
    assert history["interactions"][0]["interaction_id"] == interaction_id
    assert history["interactions"][0]["status"] == "scope_blocked"
    assert history["interactions"][0]["prompt"] == "What is the capital of France?"


def test_live_dashboard_single_screen_markers_and_endpoints():
    """Live page must render the one-screen dashboard shell (no-scroll grid,
    stage, map, counts, crops, history) and expose the streaming APIs."""
    from fastapi.testclient import TestClient

    from odc.console.app import create_app

    cfg = AppConfig()
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    client = TestClient(create_app(cfg, _StubRepo(), layout, autostart=False))
    html = client.get("/live").text
    for marker in (
        'class="dash"',
        'class="card session-card" open',
        'id="live-frame"',
        'id="live-overlay"',
        'id="seatmap-live"',
        'id="counts"',
        'id="analytics"',
        'id="seat-state-list"',
        'id="crops"',
        'id="history"',
        'id="evidence"',
        'id="gender-status"',
        'id="assistant-panel"',
        'id="assistant-toggle"',
        'id="return-live"',
        'aria-label="Continuous rolling frame history"',
        "frame-thumb.jpeg?seq=",
        'id="alerts-card"',
        'id="tgl-seats"',
        'id="tgl-face"',
        'id="tgl-gender" checked',
        'id="f-pose"',
        'id="f-dms"',
        'id="f-mask"',
        'id="mask-src"',
        "Video source",
        "Your clip",
        "Empty candidates",
        "Sustained clear-scene evidence; not a verified empty seat",
        "/api/live/state",
        "/sse/stream",
    ):
        assert marker in html, marker
    assert client.get("/api/alerts").json() == {"active": [], "recent": []}
    html = client.get("/live").text
    assert 'data-theme="dark"' in html  # dark default, OS preference upgrades
    assert "odc-theme-mode" in html
    assert 'class="skip-link"' in html
    assert 'class="nav-group"' in html
    assert 'aria-live="polite"' in html
    assert 'id="theme-toggle"' in html
    assert "/static/brand/plomid-lockup-copper.svg" in html
    state = client.get("/api/live/state").json()
    assert {
        "version",
        "detections",
        "masks",
        "seat_rows",
        "seat_polygons",
        "counts",
        "analytics",
        "belt_overlays",
        "pose",
        "face",
        "alerts",
        "precise",
        "crop_ids",
    } <= set(state)
    assert state["snapshot"]["threshold"] == 0.0  # idle hub default
    assert state["gender_status"]["state"] == "off"
    assert client.get("/frame.jpeg").status_code == 204  # no frame yet, not an error
    assert set(client.get("/api/frames/history").json()) == {"live_seq", "frames"}


def test_workspaces_use_contextual_inspector_and_wide_analysis_routes():
    """Seat inspection stays beside the cabin; analysis routes use that space."""
    from fastapi.testclient import TestClient

    from odc.console.app import create_app

    client = TestClient(
        create_app(
            AppConfig(),
            _StubRepo(),
            load_layout("configs/seats/demo-f1-v0.yaml"),
            autostart=False,
        )
    )

    live = client.get("/live")
    assert live.status_code == 200
    assert 'class="layout has-inspector"' in live.text
    assert 'aria-label="Seat inspector"' in live.text

    seats = client.get("/seats").text
    assert 'class="layout wide-workspace"' in seats
    assert 'aria-label="Selected seat evidence"' in seats
    assert 'id="inspector"' in seats
    assert "seat-attribute-card" in seats
    assert 'class="layout has-inspector"' not in seats

    for path in ("/timeline", "/diagnostics", "/validation", "/config", "/fleet"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert 'class="layout wide-workspace"' in response.text, path
        assert 'aria-label="Seat inspector"' not in response.text, path

    assert client.get("/timeline").text.count("Event timeline —") == 1
    assert "Show seat events" in client.get("/timeline").text
    config = client.get("/config").text
    assert 'id="config-layout"' in config
    assert "Detection threshold" in config
    assert "Edit seat geometry" in config
    assert 'href="/dossier"' not in config
    assert client.get("/dossier", follow_redirects=False).status_code == 303


def test_uncalibrated_and_auto_proposal_notices_are_contextual():
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from odc.console.app import create_app
    from odc.service import ServiceHub, ServiceSnapshot

    layout = load_layout("configs/seats/demo-f1-v0.yaml").model_copy(
        update={"layout_type": "auto-proposal"}
    )
    hub = ServiceHub()
    hub.snapshot = replace(ServiceSnapshot(), calibrated=False)
    client = TestClient(create_app(AppConfig(), _StubRepo(), layout, hub, autostart=False))
    live = client.get("/live").text
    seats = client.get("/seats").text
    config = client.get("/config").text
    for body in (live, config):
        assert "UNCALIBRATED camera geometry" not in body
        assert "AUTO CANDIDATE — REVIEW CALIBRATION" not in body
    assert "Seat geometry is uncalibrated." in seats
    assert "Review the proposed zones below." in seats
    assert "Pilot dossier" not in live + seats + config


def test_live_frame_seat_labels_use_text_halo_without_dark_fill_plate():
    from pathlib import Path

    template = Path("odc/console/templates/live.html").read_text()
    overlay = template.split("function renderOverlay", 1)[1].split("let seatView", 1)[0]
    assert 'text-anchor="middle"' in overlay
    assert 'stroke-width="3.5"' in overlay
    assert 'fill="#08090b" fill-opacity="0.85"' not in overlay


def test_first_frame_preview_is_real_cached_and_confined_to_footage():
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient

    from odc.console.app import create_app

    client = TestClient(
        create_app(
            AppConfig(), _StubRepo(), load_layout("configs/seats/demo-f1-v0.yaml"), autostart=False
        )
    )
    response = client.get("/api/service/preview.jpeg", params={"path": "footage/R1_rear_wide.mp4"})
    assert response.status_code == 200, response.text
    frame = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None and frame.size > 0
    assert (
        client.get("/api/service/preview.jpeg", params={"path": "../README.md"}).status_code == 400
    )


def test_restart_applies_new_settings_without_manual_stop():
    from fastapi.testclient import TestClient

    from odc.console.app import create_app

    cfg = AppConfig()
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    client = TestClient(create_app(cfg, _StubRepo(), layout, autostart=False))
    try:
        body = {
            "source": "replay",
            "path": "footage/E1_empty_cabin.mp4",
            "layout": "standard-5-seat-rhd",
            "threshold": 0.9,
        }
        assert client.post("/api/service/start", json=body).status_code == 200
        assert client.post("/api/service/start", json=body).status_code == 409
        body["threshold"] = 0.35
        body["restart"] = True
        r = client.post("/api/service/start", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["restarted"] is True
        assert client.get("/api/service/status").json()["running"] is True
    finally:
        client.post("/api/service/stop")


def test_upload_accepts_video_and_rejects_junk():
    from pathlib import Path

    from fastapi.testclient import TestClient

    from odc.console.app import create_app

    cfg = AppConfig()
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    client = TestClient(create_app(cfg, _StubRepo(), layout, autostart=False))
    data = Path("footage/E1_empty_cabin.mp4").read_bytes()
    r = client.post(
        "/api/service/upload?filename=my clip.mp4",
        content=data,
        headers={"Content-Type": "video/mp4"},
    )
    assert r.status_code == 200, r.text
    rel = r.json()["path"]
    assert rel.startswith("footage/uploads/")
    try:
        assert Path(rel).exists()
        sources = client.get("/api/service/sources").json()
        assert any(u.endswith(Path(rel).name) for u in sources["uploads"])
        assert client.get("/api/service/status").json()["running"] is False
    finally:
        Path(rel).unlink(missing_ok=True)
    bad = client.post("/api/service/upload?filename=x.txt", content=b"hello")
    assert bad.status_code == 400
    fake = client.post("/api/service/upload?filename=e.mp4", content=b"not a video")
    assert fake.status_code == 400
    empty = client.post("/api/service/upload?filename=e.mp4", content=b"")
    assert empty.status_code == 400


def test_start_switches_console_geometry_to_active_session():
    # The map must describe THIS video: starting a new session switches
    # the console geometry (previously the boot layout stuck, so auto
    # sessions rendered an empty/wrong map).
    from fastapi.testclient import TestClient

    from odc.console.app import create_app

    cfg = AppConfig()
    app = create_app(cfg, _StubRepo(), load_layout("configs/seats/demo-f1-v0.yaml"))
    client = TestClient(app)
    try:
        assert app.state.layout.version == "demo-f1-v1"
        r = client.post(
            "/api/service/start",
            json={"source": "replay", "path": "footage/F1_family_ride.mp4", "layout": "demo-r1-v0"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["layout"] == "demo-r1-v1"
        assert app.state.layout.version == "demo-r1-v1"
        r = client.post(
            "/api/service/start",
            json={
                "source": "replay",
                "path": "footage/E1_empty_cabin.mp4",
                "layout": "demo-e1-v0",
                "restart": True,
            },
        )
        assert r.status_code == 200, r.text
        assert app.state.layout.version == "demo-e1-v0"
    finally:
        client.post("/api/service/stop")


def test_start_unknown_clip_probes_then_maps_or_honest_fallback():
    # Unknown clip (no curated layout): the server analyses the video
    # first, then maps seats — the response says which happened.
    from fastapi.testclient import TestClient

    from odc.console.app import create_app

    cfg = AppConfig()
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    client = TestClient(create_app(cfg, _StubRepo(), layout, autostart=False))
    try:
        r = client.post(
            "/api/service/start",
            json={"source": "replay", "path": "footage/X3_dog.mp4", "threshold": 0.35},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # "saved" = a previous probe's seat map reused from PLOMID
        # (same video, unchanged) — equally valid as a fresh auto-proposal.
        assert body["layout_source"] in (
            "auto-1",
            "auto-2",
            "auto-3",
            "uncalibrated",
            "saved",
        )
        assert body["probe"] is not None and body["probe"].get("reason")
        status = client.get("/api/service/status").json()
        assert status["running"] is True
    finally:
        client.post("/api/service/stop")


def test_map_labels_show_mf_badges():
    from odc.console.app import seat_svg, vehicle_svg

    layout = load_layout("configs/seats/demo-r1-v0.yaml")
    rows = {
        s.seat_id: {
            "presence": "OCCUPIED",
            "child": "Unknown",
            "restraint": "Unknown",
            "belt": "Detected",
            "belt_issue": False,
            "gender": "APPARENT_WOMAN",
            "visibility": "VISIBLE",
        }
        for s in layout.seats
    }
    car = vehicle_svg(layout, rows)
    assert "+F" in car  # appearance is in the seat title, without a long map badge
    assert car.count('class="maplabel"') == len(layout.seats)
    poly = seat_svg(layout, rows, "poly")
    assert "· F" in poly  # zones view labels seat + M/F at the centroid
    assert "<text" in poly
    plain = {s.seat_id: {"presence": "UNKNOWN"} for s in layout.seats}
    assert "· M" not in vehicle_svg(layout, plain)
    assert "· F" not in seat_svg(layout, plain, "poly")


def test_live_page_has_seat_context_alerts_and_frame():
    from fastapi.testclient import TestClient

    from odc.console.app import create_app

    cfg = AppConfig()
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    client = TestClient(create_app(cfg, _StubRepo(), layout, autostart=False))
    html = client.get("/live").text
    assert 'id="seat-state-list"' in html
    assert 'id="alerts-recent"' in html
    assert 'id="sidemap-frame"' in html
    assert "/api/frames/live-evidence?seq=" in html
    assert 'id="inspector"' in html
    assert "refreshRecentAlerts" in html
    assert "refreshEventHistory" in html
    assert "Active alerts" in html and "Recent cabin events" in html
    assert "PLOMID is waiting for a response from local Mistral" in html
    assert "shared inference" not in html
    assert "data-seat-detail-close" in html
    assert "/api/service/preview.jpeg?path=" in html
    assert "function fitPreviewStage()" in html
    assert "image.onload = fitPreviewStage" in html
    assert "image.onload = fitStage" not in html
    assert 'id="ly-gender"' in html
    assert "MiVOLO apparent-gender estimate" in html
    assert 'value="auto" selected>Auto-detect cabin layout' in html
    assert 'standard-5-seat-rhd" selected' not in html
    assert 'id="f-threshold" min="0" max="1" step="0.05" value="0.7"' in html
    assert '<details class="card session-card" open>' in html
    assert "Auto preserves one-seat rows for buses/single-column cabins" not in html
    assert "Choose a bundled or uploaded replay." not in html
    assert 'id="toggle-session"' in html
    assert 'id="toggle-session" class="primary"' in html
    assert 'id="start-run"' not in html
    assert 'onclick="stopRun()"' not in html
    assert "function toggleSession(event)" in html
    assert "document.getElementById('f-layout').value = ''" not in html
    assert "AI evidence boundary" not in html
    assert 'seatmap-live .seat[role="button"]' in html
    assert 'value="UNKNOWN" selected>Auto</option>' in html
    assert "Benchmark card" not in html
    assert client.get("/benchmark-card").status_code == 404
    assert client.get("/api/benchmark-card").status_code == 404


def test_real_frame_thumbnail_is_small_cached_and_timestamp_bound():
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient

    from odc.console.app import create_app
    from odc.service import FrameHistoryEntry, ServiceHub

    captured_at = "2026-10-23T11:00:00+00:00"
    ok, encoded = cv2.imencode(".jpg", np.zeros((720, 1280, 3), dtype=np.uint8))
    assert ok
    hub = ServiceHub()
    detection = {"label": "person", "score": 0.91, "bbox": [0.1, 0.2, 0.4, 0.9], "seat": "seat_1"}
    hub.history.append(
        FrameHistoryEntry(17, encoded.tobytes(), captured_at, 1, {}, detections=[detection])
    )
    client = TestClient(
        create_app(AppConfig(), _StubRepo(), load_layout("configs/seats/demo-f1-v0.yaml"), hub=hub)
    )
    response = client.get("/frame-thumb.jpeg", params={"seq": 17, "v": captured_at})
    assert response.status_code == 200
    thumbnail = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert thumbnail.shape[:2] == (135, 240)
    assert response.headers["cache-control"] == "private, max-age=3600"
    evidence = client.get("/api/frames/live-evidence", params={"seq": 17})
    assert evidence.status_code == 200
    assert evidence.json()["detections"] == [detection]
    assert client.get("/api/frames/live-evidence", params={"seq": 18}).status_code == 404
    assert (
        client.get(
            "/frame-thumb.jpeg", params={"seq": 17, "v": "wrong-frame-timestamp"}
        ).status_code
        == 404
    )


def test_live_api_reports_mf_model_readiness():
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from odc.console.app import create_app

    cfg = AppConfig()
    enabled_cfg = cfg.model_copy(
        update={
            "age": cfg.age.model_copy(update={"enabled": False}),
            "detection": cfg.detection.model_copy(update={"enable_gender_appearance": True}),
        }
    )
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    app = create_app(cfg, _StubRepo(), layout, autostart=False)
    app.state.service = SimpleNamespace(
        _cfg=enabled_cfg,
        _gender_loading=False,
        _gender_probe=None,
        _gender_tried=True,
        _gender_error="weights are not available",
    )
    result = TestClient(app).get("/api/live/state").json()["gender_status"]
    assert result["state"] == "unavailable"
    assert "weights are not available" in result["detail"]


def test_versioned_live_api_exposes_additive_schema_version():
    from fastapi.testclient import TestClient

    from odc.console.app import create_app

    app = create_app(
        AppConfig(), _StubRepo(), load_layout("configs/seats/demo-f1-v0.yaml"), autostart=False
    )
    response = TestClient(app).get("/api/v1/live/state")
    assert response.status_code == 200
    assert response.json()["api_schema_version"] == "odc.live.v1"


def test_versioned_live_api_preserves_candidate_and_unknown_capability_states():
    from fastapi.testclient import TestClient

    from odc.console.app import create_app

    layout = load_layout("configs/seats/demo-e1-v0.yaml")
    app = create_app(AppConfig(), _StubRepo(), layout, autostart=False)
    row = {
        "seat_id": "driver",
        "presence": "OCCUPIED",
        "confidence": 0.8,
        "visibility": "VISIBLE",
        "reason_codes": [],
        "child": "Unknown",
        "restraint": "Unknown",
        "restraint_candidate": True,
        "child_seat_type": "UNKNOWN",
        "child_seat_type_candidate": "INFANT",
        "child_seat_occupancy": "UNKNOWN",
        "child_seat_occupancy_reason": "person-overlap-unvalidated",
        "belt": "Unknown",
        "belt_observation": "unclear",
        "belt_observation_reason": "bad-lighting",
        "belt_observation_score": 0.0,
        "belt_status_semantics": "candidate-only",
        "posture": "BODY_PART_OCCLUDED",
        "posture_reason": "shoulder-or-hip-landmark-not-visible",
        "track_ids": ["t1"],
        "updated_at": "2026-10-24T00:00:00+00:00",
        "frame_seq": 1,
    }
    app.state.hub.publish(ServiceSnapshot(), [], [], seat_rows=[row])

    payload = TestClient(app).get("/api/v1/live/state").json()
    serialized = payload["seat_rows"][0]

    assert serialized["restraint"] == "Unknown"
    assert serialized["restraint_candidate"] is True
    assert serialized["child_seat_occupancy"] == "UNKNOWN"
    assert serialized["child_seat_type"] == "UNKNOWN"
    assert serialized["child_seat_type_candidate"] == "INFANT"
    assert serialized["belt"] == "Unknown"
    assert serialized["belt_observation"] == "unclear"
    assert serialized["posture"] == "BODY_PART_OCCLUDED"
