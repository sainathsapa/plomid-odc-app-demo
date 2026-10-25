"""Integration tests — every row here travels SQL-over-PG-wire to real PLOMID.

Provisioning: `plomid_dsn` fixture (fresh scratch instance per test).
"""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest

from odc.domain import RepositoryError, SqlErrorKind
from odc.migrate import (
    MIGRATIONS_DIR,
    apply_pending,
    current_version,
    list_migrations,
    split_statements,
)
from odc.persistence import PlomidRepository

TS = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)


def _repo(dsn: str) -> PlomidRepository:
    return PlomidRepository(dsn)


def test_migrate_fresh_to_current(plomid_dsn):
    repo = _repo(plomid_dsn)
    try:
        latest = max(v for v, _ in list_migrations())
        assert repo.ensure_schema_version(latest) == latest
        assert repo.health_check() is True
    finally:
        repo.close()


def test_migrate_upgrade_v1_to_v2(plomid_dsn):
    import os

    if os.environ.get("ODC_DESTRUCTIVE_TESTS") != "1":
        pytest.skip("drops tables; set ODC_DESTRUCTIVE_TESTS=1 on a scratch instance")

    v1 = (MIGRATIONS_DIR / "V001__odc_foundation.sql").read_text()
    conn = psycopg.connect(plomid_dsn, autocommit=True)
    try:
        with conn.cursor() as cur:
            for table in (
                "odc_event",
                "odc_seat_state_current",
                "odc_occupant_state_current",
                "odc_inference_metric",
                "odc_validation_result",
                "odc_validation_run",
                "odc_camera_status",
                "odc_model_artifact",
                "odc_session",
                "odc_seat_layout",
                "odc_video_asset",
                "odc_video_seatmap",
                "odc_probe_cache",
                "odc_analytics_snapshot",
                "odc_schema_version",
            ):
                try:
                    cur.execute(f"DROP TABLE {table}")
                except Exception:
                    pass
            cur.execute(
                "CREATE TABLE odc_schema_version (schema_version BIGINT PRIMARY KEY, "
                "applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            for statement in split_statements(v1):
                cur.execute(statement)
            cur.execute("INSERT INTO odc_schema_version (schema_version) VALUES (1)")
        assert current_version(conn) == 1
    finally:
        conn.close()
    repo = _repo(plomid_dsn)
    try:
        latest = max(v for v, _ in list_migrations())
        assert repo.ensure_schema_version(2) == latest
        conn = psycopg.connect(plomid_dsn, autocommit=True)
        try:
            applied = apply_pending(conn)
            assert applied == []
            assert current_version(conn) == latest
        finally:
            conn.close()
    finally:
        repo.close()


def test_schema_mismatch_fails_clearly(plomid_dsn):
    repo = _repo(plomid_dsn)
    try:
        repo.ensure_schema_version(2)
        with pytest.raises(RepositoryError) as exc:
            repo.ensure_schema_version(999)
        assert exc.value.error.kind == SqlErrorKind.SCHEMA_MISMATCH
    finally:
        repo.close()


def test_session_event_roundtrip(
    plomid_dsn, session_meta, seat_state, state_event, model_manifest, camera_health
):
    repo = _repo(plomid_dsn)
    try:
        repo.ensure_schema_version(2)
        repo.start_session(session_meta)
        repo.upsert_model_artifact(model_manifest)
        assert repo.read_model_manifest(model_manifest.model_id).family == "rt-detr-v2"
        repo.upsert_camera_status(camera_health)
        repo.write_state_snapshot([seat_state])
        seats = repo.read_current_seat_states("ses-test")
        assert len(seats) == 1 and seats[0].presence.value == "LIKELY_OCCUPIED"
        repo.write_frame_observation(
            "ses-test",
            7,
            TS,
            [{"seat_id": "rear-left", "child": "Unknown", "belt": "Unknown"}],
            {"frame_seq": 7, "skeletons": []},
            [{"seat_id": "rear-left", "decision": "unknown"}],
        )
        observation = repo.read_latest_frame_observation("ses-test")
        assert observation is not None and observation["frame_seq"] == 7
        assert observation["payload"]["api_schema_version"] == "odc.seat-observation.v1"
        assert observation["payload"]["seat_rows"][0]["child"] == "Unknown"
        assert observation["payload"]["pose"]["skeletons"] == []
        repo.append_assistant_interaction(
            {
                "interaction_id": "assistant-roundtrip",
                "session_id": "ses-test",
                "occurred_at": TS,
                "frame_seq": 7,
                "seat_id": "rear-left",
                "provider": "ollama",
                "model": "mistral",
                "status": "completed",
                "prompt": "Why is the belt unknown?",
                "answer": "The strap evidence is insufficient.",
                "context": {"seat_rows": [{"seat_id": "rear-left", "belt": "Unknown"}]},
            }
        )
        history = repo.read_assistant_interactions("ses-test")
        assert history[0]["interaction_id"] == "assistant-roundtrip"
        assert history[0]["context"]["seat_rows"][0]["belt"] == "Unknown"
        repo.append_event(state_event)
        repo.append_event(state_event)  # idempotent repeat
        timeline = repo.read_event_timeline("ses-test")
        assert len(timeline) == 1
        assert repo.read_event_timeline("ses-test", seat_id="other") == []
        repo.end_session("ses-test")
    finally:
        repo.close()


def test_occupant_metrics_validation(plomid_dsn, session_meta, occupant):
    repo = _repo(plomid_dsn)
    try:
        repo.ensure_schema_version(2)
        repo.start_session(session_meta)
        repo.write_occupant_snapshot([occupant])
        repo.write_metric_batch(
            [
                {
                    "session_id": "ses-test",
                    "model_id": "m",
                    "provider": "cpu",
                    "captured_at": TS,
                    "inference_ms": 41.2,
                    "queue_ms": 3.0,
                    "dropped_frames": 1,
                    "memory_bytes": 2_000_000,
                }
            ]
        )
        repo.write_validation_run(
            {
                "run_id": "run-1",
                "build_version": "0.1.0",
                "dataset_id": "sviro-mini",
                "hardware_profile": "m2pro-cpu",
                "model_manifest": '{"a":1}',
                "started_at": TS,
                "completed_at": TS,
                "status": "DONE",
            }
        )
        repo.write_validation_result(
            [
                {
                    "run_id": "run-1",
                    "task": "occupancy",
                    "label": "person",
                    "scenario": "rear-day",
                    "precision": 0.9,
                    "recall": 0.85,
                    "f1": 0.87,
                    "fp_count": 2,
                    "fn_count": 3,
                    "latency_p50_ms": 40.0,
                    "latency_p95_ms": 65.0,
                    "notes": "smoke",
                }
            ]
        )
    finally:
        repo.close()


def test_outage_is_degraded_not_hidden():
    repo = PlomidRepository("postgresql://plomid:secret@127.0.0.1:1/plomid", connect_timeout_s=1)
    try:
        assert repo.health_check() is False
        with pytest.raises(RepositoryError) as exc:
            repo.ensure_schema_version(2)
        assert exc.value.error.kind == SqlErrorKind.CONNECTION
    finally:
        repo.close()


def test_server_timeout_default_visible(plomid_dsn):
    repo = _repo(plomid_dsn)
    try:
        assert repo.health_check() is True
    finally:
        repo.close()
    conn = psycopg.connect(plomid_dsn, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW statement_timeout")
            row = cur.fetchone()
        assert row is not None and row[0] not in ("0", "0s"), row
    finally:
        conn.close()
