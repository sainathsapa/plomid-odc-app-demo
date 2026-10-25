"""Shared deterministic fixtures — no randomness, no network, no model."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from odc import domain as d

TS = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)


# --- live-PLOMID access (integration + e2e) ---
# DSN-only over pgwire against the HOST instance. Start it, e.g.:
#   ./../plomid/target/release/plomid-server --data /tmp/plomid-test \
#     --host 127.0.0.1 --port 5432 --username plomid --password secret
#   export ODC_TEST_DSN="postgresql://plomid:secret@127.0.0.1:5432/plomid"
# Skips cleanly when unreachable. Cleanup deletes ONLY namespaced test rows
# (ses-test%, ses-chain, ses-a/b, ses-e2e, ses-shots, val-%, run-%) —
# never anything else. Never SQLite, never fakes on this path.

_TEST_SESSIONS = (
    "ses-test",
    "ses-chain",
    "ses-a",
    "ses-b",
    "ses-e2e",
    "ses-shots",
)


def _clean_test_rows(repo) -> None:
    filt = " OR ".join([f"session_id = '{s}'" for s in _TEST_SESSIONS])
    filt += " OR session_id LIKE 'ses-test%'"
    repo._run("test-clean-events", f"DELETE FROM odc_event WHERE {filt}")
    repo._run("test-clean-seats", f"DELETE FROM odc_seat_state_current WHERE {filt}")
    repo._run("test-clean-occ", f"DELETE FROM odc_occupant_state_current WHERE {filt}")
    repo._run("test-clean-frame-detections", f"DELETE FROM odc_frame_detection WHERE {filt}")
    repo._run("test-clean-frame-observations", f"DELETE FROM odc_frame_observation WHERE {filt}")
    repo._run("test-clean-assistant", f"DELETE FROM odc_assistant_interaction WHERE {filt}")
    repo._run("test-clean-metrics", f"DELETE FROM odc_inference_metric WHERE {filt}")
    repo._run(
        "test-clean-valres",
        "DELETE FROM odc_validation_result WHERE run_id LIKE 'val-%' OR run_id = 'run-1'",
    )
    repo._run(
        "test-clean-valruns",
        "DELETE FROM odc_validation_run WHERE run_id LIKE 'val-%' OR run_id = 'run-1'",
    )
    repo._run("test-clean-cam", "DELETE FROM odc_camera_status WHERE camera_id = 'cabin0'")
    repo._run("test-clean-sessions", f"DELETE FROM odc_session WHERE {filt}")
    repo._run(
        "test-clean-models",
        "DELETE FROM odc_model_artifact WHERE model_id LIKE '%@sha-unset'",
    )
    repo._run("test-clean-layouts", "DELETE FROM odc_seat_layout WHERE layout_name LIKE 'test-%'")
    repo._run(
        "test-clean-seatmaps",
        "DELETE FROM odc_video_seatmap WHERE video_key LIKE 'test-%'",
    )
    repo._run("test-clean-probes", "DELETE FROM odc_probe_cache WHERE video_key LIKE 'test-%'")
    repo._run(
        "test-clean-analytics", "DELETE FROM odc_analytics_snapshot WHERE cache_key LIKE 'test-%'"
    )
    repo._run("test-clean-assets", "DELETE FROM odc_video_asset WHERE video_key LIKE 'test-%'")


@pytest.fixture()
def plomid_dsn():
    import os

    dsn = os.environ.get("ODC_TEST_DSN", "postgresql://plomid:secret@127.0.0.1:5432/plomid")
    try:
        conn = psycopg.connect(dsn, connect_timeout=3, autocommit=True)
        conn.execute("SELECT 1")
    except Exception as e:
        pytest.skip(f"no PLOMID at DSN ({type(e).__name__}); start one and set ODC_TEST_DSN")
    from odc.persistence import PlomidRepository

    repo = PlomidRepository(dsn)
    try:
        repo.ensure_schema_version(2)
        _clean_test_rows(repo)
    finally:
        conn.close()
        repo.close()
    yield dsn
    repo = PlomidRepository(dsn)
    try:
        _clean_test_rows(repo)
    finally:
        repo.close()


@pytest.fixture()
def frame_meta() -> d.FrameMeta:
    return d.FrameMeta(
        source_id="cabin0",
        seq=42,
        captured_at=TS,
        width=1920,
        height=1080,
        mode=d.RunMode.REPLAY,
    )


@pytest.fixture()
def detection() -> d.Detection:
    return d.Detection(
        class_label="person",
        score=0.87,
        bbox=d.BBox(x_min=0.1, y_min=0.2, x_max=0.4, y_max=0.9),
        model_id="rtdetrv2-s-coco@unset",
        frame_seq=42,
    )


@pytest.fixture()
def seat_state() -> d.SeatState:
    return d.SeatState(
        session_id="ses-test",
        seat_id="rear-left",
        presence=d.PresenceState.LIKELY_OCCUPIED,
        confidence=0.72,
        visibility=d.VisibilityState.PARTIALLY_OCCLUDED,
        reason_codes=["track-overlap", "low-light"],
        updated_at=TS,
        last_positive_at=TS,
        model_set_version="unset",
        config_version="demo-v1",
    )


@pytest.fixture()
def state_event() -> d.StateEvent:
    return d.StateEvent(
        event_id="evt-1",
        session_id="ses-test",
        event_type=d.EventType.SEAT_PRESENCE_CHANGED,
        severity=d.Severity.INFO,
        seat_id="rear-left",
        occurred_at=TS,
        recorded_at=TS,
        prior_state="UNKNOWN",
        new_state="LIKELY_OCCUPIED",
        reason_codes=["dwell-met"],
        evidence_summary="3/5 frames, track t7",
    )


@pytest.fixture()
def session_meta() -> d.SessionMeta:
    return d.SessionMeta(
        session_id="ses-test",
        camera_id="cabin0",
        source_kind="replay",
        mode=d.RunMode.REPLAY,
        started_at=TS,
        build_version="0.1.0",
        config_version="demo-v1",
    )


@pytest.fixture()
def model_manifest() -> d.ModelManifest:
    return d.ModelManifest(
        model_id="rtdetrv2-s-coco@sha-unset",
        family="rt-detr-v2",
        checkpoint_sha256="pending-review",
        source_url="https://example.invalid",
        code_revision="pending-review",
        weights_license="REVIEW_REQUIRED",
        dataset_terms="REVIEW_REQUIRED",
        input_profile="640px",
        runtime_profile="onnx-cpu",
    )


@pytest.fixture()
def camera_health() -> d.CameraHealth:
    return d.CameraHealth(
        camera_id="cabin0",
        session_id="ses-test",
        status=d.HealthStatus.HEALTHY,
        fps_observed=5.0,
        checked_at=TS,
    )


@pytest.fixture()
def occupant() -> d.OccupantState:
    def attr(value: Any) -> Any:
        return d.AttributeState(value=value, confidence=0.8, observed_at=TS)

    return d.OccupantState(
        session_id="ses-test",
        track_id="t7",
        seat_id="rear-left",
        presence=attr(d.PresenceState.LIKELY_OCCUPIED),
        apparent_size=attr(d.SizeAppearance.UNKNOWN),
        posture=attr(d.PostureState.UNKNOWN),
        seatbelt=attr(d.SeatbeltState.UNSUPPORTED),
        restraint=attr(d.RestraintState.UNKNOWN),
        updated_at=TS,
    )
