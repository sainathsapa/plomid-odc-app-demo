"""Console app integration — TestClient against seeded real PLOMID rows.

Seeds go through the real repository (SQL), so screen/API agreement with
repository queries is asserted, not assumed.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from odc.config import AppConfig, AssistantConfig, CameraConfig
from odc.console.app import create_app
from odc.domain import RunMode
from odc.persistence import PlomidRepository
from odc.seats import load_layout
from odc.service import ServiceHub, ServiceSnapshot


def _seeded_repo(dsn, session_meta, seat_state, state_event):
    repo = PlomidRepository(dsn)
    repo.ensure_schema_version(2)
    repo.start_session(session_meta)
    repo.write_state_snapshot([seat_state])
    repo.write_frame_observation(
        session_meta.session_id,
        12,
        seat_state.updated_at,
        [
            {
                "seat_id": seat_state.seat_id,
                "presence": seat_state.presence.value,
                "confidence": seat_state.confidence,
                "visibility": seat_state.visibility.value,
                "child": "Yes",
                "child_reason": "mivolo-age-track-confirmed",
                "age_predictions": [{"track_id": "t-child", "age_years": 6.2}],
                "gender": "UNKNOWN",
                "belt": "Unknown",
                "posture": "UNKNOWN",
                "track_ids": ["t-child"],
            }
        ],
        {"frame_seq": 12, "skeletons": [{"track_id": "t-child", "visible": 13}]},
        [{"seat_id": seat_state.seat_id, "decision": "unknown", "reason": "fixture"}],
    )
    repo.append_assistant_interaction(
        {
            "interaction_id": "assistant-test-1",
            "session_id": session_meta.session_id,
            "occurred_at": seat_state.updated_at,
            "frame_seq": 12,
            "seat_id": seat_state.seat_id,
            "provider": "ollama",
            "model": "mistral",
            "status": "completed",
            "prompt": "Why is this child status confirmed?",
            "answer": "A child estimate is attached to this seat.",
            "context": {"seat_rows": [{"seat_id": seat_state.seat_id, "child": "Yes"}]},
        }
    )
    repo.append_event(state_event)
    return repo


def _app(dsn, session_meta, seat_state, state_event, hub=None):
    repo = _seeded_repo(dsn, session_meta, seat_state, state_event)
    cfg = AppConfig(camera=CameraConfig(path="footage/F1_family_ride.mp4"))
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    return create_app(cfg, repo, layout, hub=hub, autostart=False), repo


def test_all_screens_200(plomid_dsn, session_meta, seat_state, state_event):
    app, repo = _app(plomid_dsn, session_meta, seat_state, state_event)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        for path in (
            "/live",
            "/seats",
            "/timeline",
            "/diagnostics",
            "/relations",
            "/validation",
            "/config",
        ):
            assert client.get(path).status_code == 200, path
        assert client.get("/dossier", follow_redirects=False).status_code == 303
    finally:
        repo.close()


def test_schema_model_api_and_page(plomid_dsn, session_meta, seat_state, state_event):
    app, repo = _app(plomid_dsn, session_meta, seat_state, state_event)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        model = client.get("/api/schema").json()
        assert model["api_schema_version"] == "odc.schema-model.v1"
        assert model["schema_version"] >= 8
        assert model["totals"]["enforced"] == 12
        assert model["totals"]["soft"] >= 3
        assert model["totals"]["tables"] >= 17
        assert model["totals"]["rows"] >= 1

        links = {
            (row["source_table"], row["source_column"], row["target_table"], row["target_column"])
            for row in model["enforced"]
        }
        assert ("odc_probe_cache", "video_key", "odc_video_asset", "video_key") in links
        assert ("odc_inference_metric", "session_id", "odc_session", "session_id") in links
        assert ("odc_event", "session_id", "odc_session", "session_id") in links
        soft = {(row["source_table"], row["source_column"]) for row in model["soft"]}
        assert ("odc_inference_metric", "model_id") in soft

        page = client.get("/relations")
        assert page.status_code == 200
        assert "Enforced foreign keys" in page.text
        assert "probe_cache_asset_fk" in page.text
        assert "odc_probe_cache" in page.text
        assert "Documented soft relations" in page.text
    finally:
        repo.close()


def test_history_mode_shows_sql_rows(plomid_dsn, session_meta, seat_state, state_event):
    app, repo = _app(plomid_dsn, session_meta, seat_state, state_event)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        body = client.get("/live?session=ses-test").text
        assert "rear-left" in body and "LIKELY_OCCUPIED" in body
        seats = client.get("/api/seats?session=ses-test").json()
        assert [s["seat_id"] for s in seats] == ["rear-left"]
        assert seats[0]["reason_codes"] == ["track-overlap", "low-light"]
        assert seats[0]["child"] == "Yes"
        assert seats[0]["age_predictions"][0]["age_years"] == 6.2
        assert seats[0]["api_schema_version"] == "odc.seat-observation.v1"
        assert seats[0]["observation_frame_seq"] == 12
        assert "mivolo-age-track-confirmed" in client.get("/live?session=ses-test").text
        chat_history = client.get("/api/assistant/history?session=ses-test").json()
        assert chat_history["api_schema_version"] == "odc.assistant-history.v1"
        assert chat_history["interactions"][0]["model"] == "mistral"
        assert chat_history["interactions"][0]["context"]["seat_rows"][0]["child"] == "Yes"
        events = client.get("/api/events?session=ses-test").json()
        assert len(events) == 1 and events[0]["new_state"] == "LIKELY_OCCUPIED"
        detail = client.get("/partials/seat-detail?seat=rear-left&session=ses-test").text
        assert "0.72" in detail
        assert "No record" in client.get("/partials/seat-detail?seat=nope&session=ses-test").text
    finally:
        repo.close()


def test_api_agrees_with_repository(plomid_dsn, session_meta, seat_state, state_event):
    app, repo = _app(plomid_dsn, session_meta, seat_state, state_event)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        api_seats = client.get("/api/seats?session=ses-test").json()
        repo_seats = repo.read_current_seat_states("ses-test")
        assert len(api_seats) == len(repo_seats) == 1
        assert api_seats[0]["presence"] == repo_seats[0].presence.value
    finally:
        repo.close()


def test_unavailable_mistral_request_is_saved_in_plomid(
    plomid_dsn, session_meta, seat_state, state_event
):
    app, repo = _app(plomid_dsn, session_meta, seat_state, state_event)
    app.state.cfg = AppConfig(
        camera=CameraConfig(path="footage/F1_family_ride.mp4"),
        assistant=AssistantConfig(enabled=False),
    )
    from odc.service import ServiceSnapshot

    app.state.hub.snapshot = ServiceSnapshot(session_id=session_meta.session_id)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/assistant/ask",
            json={"prompt": "Why does PloMID show this seat as Unknown?", "seat_id": "rear-left"},
        )
        assert response.status_code == 503
        interaction_id = response.json()["interaction_id"]
        history = client.get("/api/assistant/history?session=ses-test").json()
        interaction = next(
            row for row in history["interactions"] if row["interaction_id"] == interaction_id
        )
        assert interaction["status"] == "unavailable"
        assert interaction["prompt"] == "Why does PloMID show this seat as Unknown?"
        assert interaction["context"]["seat_rows"] == []
    finally:
        repo.close()


def test_degraded_and_disconnected_modes(plomid_dsn, session_meta, seat_state, state_event):
    hub = ServiceHub()
    hub.snapshot = ServiceSnapshot(running=True, session_id="s", db_healthy=False)
    app, repo = _app(plomid_dsn, session_meta, seat_state, state_event, hub=hub)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        assert "DEGRADED" in client.get("/live").text
        hub.snapshot = ServiceSnapshot(
            running=True, session_id="s", db_healthy=True, model_healthy=True, mode=RunMode.DEMO
        )
        assert "DEMO" in client.get("/live").text
    finally:
        repo.close()


def test_db_outage_is_503_not_fake(plomid_dsn, session_meta, seat_state, state_event):
    app, repo = _app(plomid_dsn, session_meta, seat_state, state_event)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        dead = PlomidRepository(
            "postgresql://plomid:secret@127.0.0.1:1/plomid", connect_timeout_s=1
        )
        app.state.repo = dead
        try:
            assert client.get("/api/seats?session=ses-test").status_code == 503
            assert client.get("/seats").status_code == 200  # shell renders, data empty
        finally:
            dead.close()
    finally:
        repo.close()


def test_sse_streams_and_frame_empty(plomid_dsn, session_meta, seat_state, state_event):
    app, repo = _app(plomid_dsn, session_meta, seat_state, state_event)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        assert client.get("/frame.jpeg").status_code == 204
        with client.stream("GET", "/sse/stream") as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data:"):
                    assert "badge" in line
                    break
    finally:
        repo.close()


def test_dossier_downloads(plomid_dsn, session_meta, seat_state, state_event):
    app, repo = _app(plomid_dsn, session_meta, seat_state, state_event)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        md = client.get("/api/dossier?format=md").text
        assert md.startswith("# PLOMID ODC") and "Known gaps" in md
        body = client.get("/api/dossier?format=json").json()
        assert body["sessions_recorded"] >= 1
    finally:
        repo.close()
