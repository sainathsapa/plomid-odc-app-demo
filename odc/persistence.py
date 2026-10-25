"""PLOMID SQL repository (Phase 2) — the only durable-truth boundary.

All statements parameterized. All writes idempotent by PK / ON CONFLICT so
retries are safe. Retries cover CONNECTION/TIMEOUT only, max 3. Values are
never logged. Perception code never sees a connection.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import psycopg

from . import contracts
from .domain import (
    CameraHealth,
    ModelManifest,
    OccupantState,
    RepositoryError,
    SeatState,
    SessionMeta,
    SqlError,
    SqlErrorKind,
    StateEvent,
)
from .logging import get_logger
from .migrate import apply_pending, current_version

log = get_logger("persistence")
MAX_RETRIES = 3


def to_sql_error(exc: Exception) -> SqlError:
    if isinstance(exc, RepositoryError):
        return exc.error
    if isinstance(exc, psycopg.OperationalError):
        text = str(exc).lower()
        kind = SqlErrorKind.TIMEOUT if "timeout" in text else SqlErrorKind.CONNECTION
        return SqlError(kind=kind, message=str(exc)[:300], retryable=True)
    if isinstance(exc, psycopg.errors.UniqueViolation):
        return SqlError(kind=SqlErrorKind.CONFLICT, message=str(exc)[:300])
    if isinstance(exc, psycopg.errors.ForeignKeyViolation):
        return SqlError(kind=SqlErrorKind.CONFLICT, message=str(exc)[:300])
    if isinstance(exc, psycopg.errors.UndefinedTable | psycopg.errors.UndefinedColumn):
        return SqlError(kind=SqlErrorKind.SCHEMA_MISMATCH, message=str(exc)[:300])
    if isinstance(exc, psycopg.errors.FeatureNotSupported | psycopg.errors.SyntaxError):
        return SqlError(kind=SqlErrorKind.UNSUPPORTED, message=str(exc)[:300])
    if isinstance(exc, SqlError):
        return exc
    return SqlError(kind=SqlErrorKind.QUERY, message=f"{type(exc).__name__}: {exc}"[:300])


class PlomidRepository(contracts.Repository):
    def __init__(
        self,
        dsn: str,
        connect_timeout_s: int = 5,
        statement_timeout_ms: int = 10_000,
    ) -> None:
        self._dsn = dsn
        self._connect_timeout_s = connect_timeout_s
        self._statement_timeout_ms = statement_timeout_ms

    def _connect(self) -> psycopg.Connection:
        # One short-lived connection per operation: deterministic, no pool
        # threads, no setup-callback races. Fits PLOMID's single-writer
        # profile at ODC's cadence (snapshots ~1 Hz, events on change).
        # Server-side statement_timeout default (30 s) bounds queries;
        # per-op SET LOCAL lands with explicit transactions in Phase 6+.
        conn = psycopg.connect(self._dsn, autocommit=True, connect_timeout=self._connect_timeout_s)
        # ODC owns a dedicated namespace inside the single PLOMID database.
        # CREATE/SET are supported by the documented PLOMID SQL boundary and
        # make every repository operation route consistently without requiring
        # operators to run a separate bootstrap command.
        conn.execute("CREATE SCHEMA IF NOT EXISTS plomid_odc")
        conn.execute("SET search_path TO plomid_odc, public")
        return conn

    def close(self) -> None:
        pass

    def _run(
        self, op: str, sql: str, params: tuple[Any, ...] | None = None, fetch: str = "none"
    ) -> Any:
        last: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            start = time.monotonic()
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                        out: Any = None
                        if fetch == "all":
                            out = cur.fetchall()
                        elif fetch == "one":
                            out = cur.fetchone()
                log.info(
                    "sql ok op=%s attempt=%d ms=%.1f",
                    op,
                    attempt,
                    (time.monotonic() - start) * 1000,
                )
                return out
            except Exception as e:
                err = to_sql_error(e)
                log.warning("sql fail op=%s attempt=%d kind=%s", op, attempt, err.kind)
                last = RepositoryError(err)
                if not err.retryable:
                    break
        raise last or RepositoryError(SqlError(kind=SqlErrorKind.QUERY, message=f"{op} failed"))

    # -- lifecycle ------------------------------------------------------

    def health_check(self) -> bool:
        try:
            self._run("health_check", "SELECT 1", fetch="one")
            return True
        except RepositoryError:
            return False

    def ensure_schema_version(self, minimum: int) -> int:
        last: RepositoryError | None = None
        for _ in range(MAX_RETRIES):
            try:
                with self._connect() as conn:
                    applied = apply_pending(conn)
                    if applied:
                        log.info("migrations applied=%s", applied)
                    self._copy_legacy_public_rows(conn)
                    current = current_version(conn)
                if current < minimum:
                    raise RepositoryError(
                        SqlError(
                            kind=SqlErrorKind.SCHEMA_MISMATCH,
                            message=f"schema v{current} < required v{minimum}",
                        )
                    )
                return current
            except RepositoryError as e:
                last = e
                if not e.error.retryable:
                    break
            except Exception as e:
                last = RepositoryError(to_sql_error(e))
                if not last.error.retryable:
                    break
        raise last or RepositoryError(
            SqlError(kind=SqlErrorKind.SCHEMA_MISMATCH, message="schema check failed")
        )

    def _copy_legacy_public_rows(self, conn: psycopg.Connection) -> None:
        """Copy pre-namespace ODC rows once, without deleting legacy data."""
        tables = (
            "odc_session",
            "odc_model_artifact",
            "odc_camera_status",
            "odc_seat_state_current",
            "odc_event",
            "odc_occupant_state_current",
            "odc_inference_metric",
            "odc_validation_run",
            "odc_validation_result",
            "odc_seat_layout",
            "odc_video_asset",
            "odc_video_seatmap",
            "odc_probe_cache",
            "odc_analytics_snapshot",
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name LIKE 'odc_%'"
            )
            available = {row[0] for row in cur.fetchall()}
        for table in tables:
            if table not in available:
                continue
            try:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO plomid_odc." + table + " "
                            "SELECT * FROM public." + table + " ON CONFLICT DO NOTHING"
                        )
            except Exception as exc:
                # Legacy migration is best-effort per table; the new schema
                # remains usable and the failure is visible in logs.
                log.warning("legacy ODC copy skipped table=%s error=%s", table, type(exc).__name__)

    # -- writes (all idempotent) ----------------------------------------

    def start_session(self, session: SessionMeta) -> None:
        self._run(
            "start_session",
            """INSERT INTO odc_session
               (session_id, camera_id, source_kind, mode, started_at, ended_at,
                build_version, config_version, status)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (session_id) DO UPDATE SET
                 ended_at = EXCLUDED.ended_at, status = EXCLUDED.status""",
            (
                session.session_id,
                session.camera_id,
                session.source_kind,
                session.mode.value,
                session.started_at,
                session.ended_at,
                session.build_version,
                session.config_version,
                session.status,
            ),
        )

    def end_session(self, session_id: str) -> None:
        self._run(
            "end_session",
            "UPDATE odc_session SET status = 'CLOSED', ended_at = now() WHERE session_id = %s",
            (session_id,),
        )

    def upsert_model_artifact(self, manifest: ModelManifest) -> None:
        self._run(
            "upsert_model_artifact",
            """INSERT INTO odc_model_artifact
               (model_id, family, checkpoint_sha256, source_url, code_revision,
                weights_license, dataset_terms, approved_for_commercial_use,
                input_profile, runtime_profile)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (model_id) DO UPDATE SET
                 checkpoint_sha256 = EXCLUDED.checkpoint_sha256,
                 code_revision = EXCLUDED.code_revision,
                 approved_for_commercial_use = EXCLUDED.approved_for_commercial_use""",
            (
                manifest.model_id,
                manifest.family,
                manifest.checkpoint_sha256,
                manifest.source_url,
                manifest.code_revision,
                manifest.weights_license,
                manifest.dataset_terms,
                manifest.approved_for_commercial_use,
                manifest.input_profile,
                manifest.runtime_profile,
            ),
        )

    def upsert_camera_status(self, health: CameraHealth) -> None:
        self._run(
            "upsert_camera_status",
            """INSERT INTO odc_camera_status
               (camera_id, session_id, status, fps_observed, failure_code, checked_at)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON CONFLICT (camera_id) DO UPDATE SET
                 session_id = EXCLUDED.session_id, status = EXCLUDED.status,
                 fps_observed = EXCLUDED.fps_observed,
                 failure_code = EXCLUDED.failure_code,
                 checked_at = EXCLUDED.checked_at""",
            (
                health.camera_id,
                health.session_id,
                health.status.value,
                health.fps_observed,
                health.failure_code,
                health.checked_at,
            ),
        )

    def write_state_snapshot(self, seats: list[SeatState]) -> None:
        for seat in seats:
            self._run(
                "write_state_snapshot",
                """INSERT INTO odc_seat_state_current
                   (session_id, seat_id, presence, confidence, visibility,
                    reason_codes, updated_at, last_positive_at,
                    model_set_version, config_version)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (session_id, seat_id) DO UPDATE SET
                     presence = EXCLUDED.presence, confidence = EXCLUDED.confidence,
                     visibility = EXCLUDED.visibility,
                     reason_codes = EXCLUDED.reason_codes,
                     updated_at = EXCLUDED.updated_at,
                     last_positive_at = EXCLUDED.last_positive_at,
                     model_set_version = EXCLUDED.model_set_version,
                     config_version = EXCLUDED.config_version""",
                (
                    seat.session_id,
                    seat.seat_id,
                    seat.presence.value,
                    seat.confidence,
                    seat.visibility.value,
                    seat.reason_codes,
                    seat.updated_at,
                    seat.last_positive_at,
                    seat.model_set_version,
                    seat.config_version,
                ),
            )

    def write_occupant_snapshot(self, occupants: list[OccupantState]) -> None:
        for occ in occupants:

            def val_conf(attr: Any) -> tuple[str | None, float | None]:
                if attr is None:
                    return None, None
                return attr.value.value, attr.confidence

            presence, presence_conf = val_conf(occ.presence)
            size, size_conf = val_conf(occ.apparent_size)
            posture, posture_conf = val_conf(occ.posture)
            belt, belt_conf = val_conf(occ.seatbelt)
            restraint, restraint_conf = val_conf(occ.restraint)
            self._run(
                "write_occupant_snapshot",
                """INSERT INTO odc_occupant_state_current
                   (session_id, track_id, seat_id, presence, presence_conf,
                    apparent_size, apparent_size_conf, posture, posture_conf,
                    seatbelt, seatbelt_conf, restraint, restraint_conf, updated_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (session_id, track_id) DO UPDATE SET
                     seat_id = EXCLUDED.seat_id, presence = EXCLUDED.presence,
                     presence_conf = EXCLUDED.presence_conf,
                     apparent_size = EXCLUDED.apparent_size,
                     apparent_size_conf = EXCLUDED.apparent_size_conf,
                     posture = EXCLUDED.posture, posture_conf = EXCLUDED.posture_conf,
                     seatbelt = EXCLUDED.seatbelt, seatbelt_conf = EXCLUDED.seatbelt_conf,
                     restraint = EXCLUDED.restraint,
                     restraint_conf = EXCLUDED.restraint_conf,
                     updated_at = EXCLUDED.updated_at""",
                (
                    occ.session_id,
                    occ.track_id,
                    occ.seat_id,
                    presence,
                    presence_conf,
                    size,
                    size_conf,
                    posture,
                    posture_conf,
                    belt,
                    belt_conf,
                    restraint,
                    restraint_conf,
                    occ.updated_at,
                ),
            )

    def append_event(self, event: StateEvent) -> None:
        self._run(
            "append_event",
            """INSERT INTO odc_event
               (event_id, session_id, event_type, severity, seat_id, track_id,
                occurred_at, recorded_at, prior_state, new_state, reason_codes,
                evidence_summary, config_version, model_set_version)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (event_id) DO NOTHING""",
            (
                event.event_id,
                event.session_id,
                event.event_type.value,
                event.severity.value,
                event.seat_id,
                event.track_id,
                event.occurred_at,
                event.recorded_at,
                event.prior_state,
                event.new_state,
                event.reason_codes,
                event.evidence_summary,
                event.config_version,
                event.model_set_version,
            ),
        )

    def write_metric_batch(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self._run(
                "write_metric_batch",
                """INSERT INTO odc_inference_metric
                   (session_id, model_id, provider, captured_at, inference_ms,
                    queue_ms, dropped_frames, memory_bytes)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT DO NOTHING""",
                (
                    row["session_id"],
                    row["model_id"],
                    row.get("provider", ""),
                    row["captured_at"],
                    row["inference_ms"],
                    row.get("queue_ms", 0.0),
                    row.get("dropped_frames", 0),
                    row.get("memory_bytes"),
                ),
            )

    def write_frame_detections(self, rows: list[dict[str, Any]]) -> None:
        """Persist normalized per-frame boxes and scores for replay/R&D."""
        import json as _json

        for row in rows:
            self._run(
                "write_frame_detections",
                """INSERT INTO odc_frame_detection
                   (detection_id, session_id, frame_seq, captured_at, class_label,
                    confidence, bbox, seat_id, track_id, metadata)
                   VALUES (%s,%s,%s,%s,%s,%s,%s::vector,%s,%s,%s::jsonb)
                   ON CONFLICT (detection_id) DO UPDATE SET
                     confidence = EXCLUDED.confidence,
                     bbox = EXCLUDED.bbox,
                     seat_id = EXCLUDED.seat_id,
                     track_id = EXCLUDED.track_id,
                     metadata = EXCLUDED.metadata""",
                (
                    row["detection_id"],
                    row["session_id"],
                    row["frame_seq"],
                    row["captured_at"],
                    row["class_label"],
                    row["confidence"],
                    row["bbox"],
                    row.get("seat_id"),
                    row.get("track_id"),
                    _json.dumps(row.get("metadata", {})),
                ),
            )

    def write_frame_observation(
        self,
        session_id: str,
        frame_seq: int,
        captured_at: datetime,
        seat_rows: list[dict[str, Any]],
        pose: dict[str, Any],
        specialist_decisions: list[dict[str, Any]],
    ) -> None:
        """Persist the complete structured seat output once per processed frame."""
        import json as _json

        payload = {
            "api_schema_version": "odc.seat-observation.v1",
            "seat_rows": seat_rows,
            "pose": pose,
            "specialist_decisions": specialist_decisions,
        }
        self._run(
            "write_frame_observation",
            """INSERT INTO odc_frame_observation
               (session_id, frame_seq, captured_at, payload)
               VALUES (%s,%s,%s,%s::jsonb)
               ON CONFLICT (session_id, frame_seq) DO UPDATE SET
                 captured_at = EXCLUDED.captured_at, payload = EXCLUDED.payload""",
            (session_id, frame_seq, captured_at, _json.dumps(payload, allow_nan=False)),
        )

    def read_latest_frame_observation(self, session_id: str) -> dict[str, Any] | None:
        """Read the latest structured frame payload for a persisted session."""
        row = self._run(
            "read_latest_frame_observation",
            """SELECT frame_seq, captured_at, payload::text
               FROM odc_frame_observation
               WHERE session_id = %s
               ORDER BY frame_seq DESC
               LIMIT 1""",
            (session_id,),
            fetch="one",
        )
        if row is None:
            return None
        import json as _json

        captured_at = row[1].isoformat() if hasattr(row[1], "isoformat") else str(row[1])
        return {
            "frame_seq": row[0],
            "captured_at": captured_at,
            "payload": _json.loads(row[2]),
        }

    def append_assistant_interaction(self, row: dict[str, Any]) -> None:
        """Persist a single explicit local assistant request and its evidence context."""
        import json as _json

        self._run(
            "append_assistant_interaction",
            """INSERT INTO odc_assistant_interaction
               (interaction_id, session_id, occurred_at, frame_seq, seat_id,
                provider, model, status, prompt, answer, context)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
               ON CONFLICT (interaction_id) DO NOTHING""",
            (
                row["interaction_id"],
                row.get("session_id"),
                row["occurred_at"],
                row.get("frame_seq"),
                row.get("seat_id"),
                row["provider"],
                row.get("model"),
                row["status"],
                row["prompt"],
                row.get("answer"),
                _json.dumps(row.get("context", {}), allow_nan=False),
            ),
        )

    def read_assistant_interactions(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Read explicit assistant exchanges for a session, newest first."""
        bounded_limit = max(1, min(int(limit), 500))
        rows = (
            self._run(
                "read_assistant_interactions",
                f"""SELECT interaction_id, session_id, occurred_at, frame_seq,
                      seat_id, provider, model, status, prompt, answer, context::text
               FROM odc_assistant_interaction
               WHERE session_id = %s
               ORDER BY occurred_at DESC
               LIMIT {bounded_limit}""",
                (session_id,),
                fetch="all",
            )
            or []
        )
        import json as _json

        result = []
        for row in rows:
            result.append(
                {
                    "interaction_id": row[0],
                    "session_id": row[1],
                    "occurred_at": row[2].isoformat()
                    if hasattr(row[2], "isoformat")
                    else str(row[2]),
                    "frame_seq": row[3],
                    "seat_id": row[4],
                    "provider": row[5],
                    "model": row[6],
                    "status": row[7],
                    "prompt": row[8],
                    "answer": row[9],
                    "context": _json.loads(row[10]),
                }
            )
        return result

    def read_frame_detections(
        self, session_id: str, frame_seq: int | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        """Read persisted boxes/confidence; bbox is returned as vector text."""
        sql = """SELECT detection_id, session_id, frame_seq, captured_at,
                         class_label, confidence, bbox::text, seat_id, track_id,
                         metadata::text
                  FROM odc_frame_detection
                  WHERE session_id = %s"""
        params: tuple[Any, ...] = (session_id,)
        if frame_seq is not None:
            sql += " AND frame_seq = %s"
            params += (frame_seq,)
        rows = self._run("read_frame_detections", sql, params, fetch="all") or []
        return [
            {
                "detection_id": row[0],
                "session_id": row[1],
                "frame_seq": row[2],
                "captured_at": row[3].isoformat() if hasattr(row[3], "isoformat") else str(row[3]),
                "class_label": row[4],
                "confidence": row[5],
                "bbox": row[6],
                "seat_id": row[7],
                "track_id": row[8],
                "metadata": row[9],
            }
            for row in rows[: max(1, min(int(limit), 5000))]
        ]

    # -- reads ----------------------------------------------------------

    def read_current_seat_states(self, session_id: str) -> list[SeatState]:
        from .domain import PresenceState, VisibilityState

        rows = self._run(
            "read_current_seat_states",
            """SELECT session_id, seat_id, presence, confidence, visibility,
                      reason_codes, updated_at, last_positive_at,
                      model_set_version, config_version
               FROM odc_seat_state_current WHERE session_id = %s""",
            (session_id,),
            fetch="all",
        )
        return [
            SeatState(
                session_id=r[0],
                seat_id=r[1],
                presence=PresenceState(r[2]),
                confidence=r[3],
                visibility=VisibilityState(r[4]),
                reason_codes=list(r[5]),
                updated_at=r[6],
                last_positive_at=r[7],
                model_set_version=r[8],
                config_version=r[9],
            )
            for r in rows
        ]

    def read_current_occupants(self, session_id: str) -> list[OccupantState]:
        """Read the latest persisted occupant attributes for historical review."""
        from .domain import (
            AttributeState,
            OccupantState,
            PostureState,
            PresenceState,
            RestraintState,
            SeatbeltState,
            SizeAppearance,
        )

        rows = (
            self._run(
                "read_current_occupants",
                """SELECT session_id, track_id, seat_id, presence, presence_conf,
                      apparent_size, apparent_size_conf, posture, posture_conf,
                      seatbelt, seatbelt_conf, restraint, restraint_conf, updated_at
               FROM odc_occupant_state_current WHERE session_id = %s""",
                (session_id,),
                fetch="all",
            )
            or []
        )

        def attr(
            value: str | None,
            confidence: float | None,
            enum_type: Any,
            observed_at: datetime,
        ) -> Any:
            if value is None or confidence is None:
                return None
            return AttributeState(
                value=enum_type(value),
                confidence=confidence,
                observed_at=observed_at,
            )

        return [
            OccupantState(
                session_id=row[0],
                track_id=row[1],
                seat_id=row[2],
                presence=attr(row[3], row[4], PresenceState, row[13]),
                apparent_size=attr(row[5], row[6], SizeAppearance, row[13]),
                posture=attr(row[7], row[8], PostureState, row[13]),
                seatbelt=attr(row[9], row[10], SeatbeltState, row[13]),
                restraint=attr(row[11], row[12], RestraintState, row[13]),
                updated_at=row[13],
            )
            for row in rows
        ]

    def read_event_timeline(
        self, session_id: str, seat_id: str | None = None, limit: int = 200
    ) -> list[StateEvent]:
        from .domain import EventType, Severity

        # PLOMID requires LIMIT as a literal (params rejected) — clamp the
        # app-controlled int and inline it; all values stay parameterized.
        count = max(1, min(int(limit), 1000))
        if seat_id is None:
            rows = self._run(
                "read_event_timeline",
                f"""SELECT event_id, session_id, event_type, severity, seat_id,
                          track_id, occurred_at, recorded_at, prior_state, new_state,
                          reason_codes, evidence_summary, config_version,
                          model_set_version
                   FROM odc_event WHERE session_id = %s
                   ORDER BY occurred_at LIMIT {count}""",
                (session_id,),
                fetch="all",
            )
        else:
            rows = self._run(
                "read_event_timeline",
                f"""SELECT event_id, session_id, event_type, severity, seat_id,
                          track_id, occurred_at, recorded_at, prior_state, new_state,
                          reason_codes, evidence_summary, config_version,
                          model_set_version
                   FROM odc_event WHERE session_id = %s AND seat_id = %s
                   ORDER BY occurred_at LIMIT {count}""",
                (session_id, seat_id),
                fetch="all",
            )
        return [
            StateEvent(
                event_id=r[0],
                session_id=r[1],
                event_type=EventType(r[2]),
                severity=Severity(r[3]),
                seat_id=r[4],
                track_id=r[5],
                occurred_at=r[6],
                recorded_at=r[7],
                prior_state=r[8],
                new_state=r[9],
                reason_codes=list(r[10]),
                evidence_summary=r[11],
                config_version=r[12],
                model_set_version=r[13],
            )
            for r in rows
        ]

    def read_model_manifest(self, model_id: str) -> ModelManifest:
        row = self._run(
            "read_model_manifest",
            """SELECT model_id, family, checkpoint_sha256, source_url,
                      code_revision, weights_license, dataset_terms,
                      approved_for_commercial_use, input_profile, runtime_profile
               FROM odc_model_artifact WHERE model_id = %s""",
            (model_id,),
            fetch="one",
        )
        if row is None:
            raise RepositoryError(
                SqlError(kind=SqlErrorKind.QUERY, message=f"unknown model_id {model_id}")
            )
        return ModelManifest(
            model_id=row[0],
            family=row[1],
            checkpoint_sha256=row[2],
            source_url=row[3],
            code_revision=row[4],
            weights_license=row[5],
            dataset_terms=row[6],
            approved_for_commercial_use=row[7],
            input_profile=row[8],
            runtime_profile=row[9],
        )

    def write_validation_run(self, row: dict[str, Any]) -> None:
        self._run(
            "write_validation_run",
            """INSERT INTO odc_validation_run
               (run_id, build_version, dataset_id, hardware_profile,
                model_manifest, started_at, completed_at, status)
               VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s)
               ON CONFLICT (run_id) DO UPDATE SET
                 completed_at = EXCLUDED.completed_at,
                 status = EXCLUDED.status""",
            (
                row["run_id"],
                row["build_version"],
                row["dataset_id"],
                row.get("hardware_profile", ""),
                row.get("model_manifest", "{}"),
                row["started_at"],
                row.get("completed_at"),
                row.get("status", "RUNNING"),
            ),
        )

    def write_validation_result(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self._run(
                "write_validation_result",
                """INSERT INTO odc_validation_result
                   (run_id, task, label, scenario, precision, recall, f1,
                    fp_count, fn_count, latency_p50_ms, latency_p95_ms, notes)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (run_id, task, label, scenario) DO UPDATE SET
                     precision = EXCLUDED.precision, recall = EXCLUDED.recall,
                     f1 = EXCLUDED.f1, fp_count = EXCLUDED.fp_count,
                     fn_count = EXCLUDED.fn_count,
                     latency_p50_ms = EXCLUDED.latency_p50_ms,
                     latency_p95_ms = EXCLUDED.latency_p95_ms,
                     notes = EXCLUDED.notes""",
                (
                    row["run_id"],
                    row["task"],
                    row["label"],
                    row.get("scenario", ""),
                    row.get("precision"),
                    row.get("recall"),
                    row.get("f1"),
                    row.get("fp_count"),
                    row.get("fn_count"),
                    row.get("latency_p50_ms"),
                    row.get("latency_p95_ms"),
                    row.get("notes", ""),
                ),
            )

    # -- seat store (V003): videos stay on disk, everything else in PLOMID.
    # All writes idempotent upserts; JSONB travels as text with ::jsonb.

    def upsert_video_asset(self, row: dict[str, Any]) -> None:
        self._run(
            "upsert_video_asset",
            """INSERT INTO odc_video_asset
               (video_key, video_path, video_sha, size_bytes, mtime)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (video_key) DO UPDATE SET
                 video_path = EXCLUDED.video_path, video_sha = EXCLUDED.video_sha,
                 size_bytes = EXCLUDED.size_bytes, mtime = EXCLUDED.mtime""",
            (
                row["video_key"],
                row.get("video_path", ""),
                row.get("video_sha", ""),
                int(row.get("size_bytes", 0)),
                int(row.get("mtime", 0)),
            ),
        )

    def list_video_assets_by_path(self, video_path: str) -> list[dict[str, Any]]:
        rows = self._run(
            "list_video_assets_by_path",
            """SELECT video_key, video_path, video_sha, size_bytes, mtime
               FROM odc_video_asset WHERE video_path = %s""",
            (video_path,),
            fetch="all",
        )
        return [
            {
                "video_key": r[0],
                "video_path": r[1],
                "video_sha": r[2],
                "size_bytes": r[3],
                "mtime": r[4],
            }
            for r in rows
        ]

    def upsert_seat_layout(self, name: str, doc: dict[str, Any]) -> None:
        import json as _json

        # Column is layout_version: bare version collides with the PLOMID
        # version() builtin (V004). Python dicts keep the "version" key.
        self._run(
            "upsert_seat_layout",
            """INSERT INTO odc_seat_layout
               (layout_name, camera_profile, layout_version, origin, seats, updated_at)
               VALUES (%s,%s,%s,%s,%s::jsonb,now())
               ON CONFLICT (layout_name) DO UPDATE SET
                 camera_profile = EXCLUDED.camera_profile,
                 layout_version = EXCLUDED.layout_version,
                 origin = EXCLUDED.origin, seats = EXCLUDED.seats,
                 updated_at = now()""",
            (
                name,
                doc.get("camera_profile", ""),
                doc.get("version", name),
                doc.get("origin", "hand"),
                _json.dumps(doc.get("seats", [])),
            ),
        )

    def load_seat_layout(self, name: str) -> dict[str, Any] | None:
        row = self._run(
            "load_seat_layout",
            """SELECT layout_name, camera_profile, layout_version, origin, seats
               FROM odc_seat_layout WHERE layout_name = %s""",
            (name,),
            fetch="one",
        )
        if row is None:
            return None
        return {
            "layout_name": row[0],
            "camera_profile": row[1],
            "version": row[2],
            "origin": row[3],
            "seats": row[4] if isinstance(row[4], list) else [],
        }

    def list_seat_layouts(self) -> list[dict[str, Any]]:
        rows = self._run(
            "list_seat_layouts",
            """SELECT layout_name, camera_profile, layout_version, origin
               FROM odc_seat_layout ORDER BY layout_name""",
            None,
            fetch="all",
        )
        return [
            {"layout_name": r[0], "camera_profile": r[1], "version": r[2], "origin": r[3]}
            for r in rows
        ]

    def save_video_seatmap(self, row: dict[str, Any]) -> None:
        import json as _json

        self._run(
            "save_video_seatmap",
            """INSERT INTO odc_video_seatmap
               (video_key, video_sha, camera_id, calibration_profile,
                layout_version, origin, layout, updated_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,now())
               ON CONFLICT (video_key) DO UPDATE SET
                 video_sha = EXCLUDED.video_sha, camera_id = EXCLUDED.camera_id,
                 calibration_profile = EXCLUDED.calibration_profile,
                 layout_version = EXCLUDED.layout_version, origin = EXCLUDED.origin,
                 layout = EXCLUDED.layout, updated_at = now()""",
            (
                row["video_key"],
                row.get("video_sha", ""),
                row.get("camera_id", ""),
                row.get("calibration_profile", ""),
                row.get("layout_version", ""),
                row.get("origin", "auto"),
                _json.dumps(row.get("layout", {})),
            ),
        )

    def load_video_seatmap(self, video_key: str) -> dict[str, Any] | None:
        row = self._run(
            "load_video_seatmap",
            """SELECT s.video_key, s.video_sha, s.camera_id, s.calibration_profile,
                      s.layout_version, s.origin, s.layout, a.video_path
               FROM odc_video_seatmap s
               LEFT JOIN odc_video_asset a ON a.video_key = s.video_key
               WHERE s.video_key = %s""",
            (video_key,),
            fetch="one",
        )
        if row is None:
            return None
        return {
            "video_key": row[0],
            "video_sha": row[1],
            "camera_id": row[2],
            "calibration_profile": row[3],
            "layout_version": row[4],
            "origin": row[5],
            "layout": row[6] if isinstance(row[6], dict) else {},
            "video_path": row[7],
        }

    def list_video_seatmaps_by_path(self, video_path: str) -> list[dict[str, Any]]:
        rows = self._run(
            "list_video_seatmaps_by_path",
            """SELECT s.video_key, s.video_sha, s.layout_version, s.origin, a.video_path
               FROM odc_video_seatmap s
               JOIN odc_video_asset a ON a.video_key = s.video_key
               WHERE a.video_path = %s""",
            (video_path,),
            fetch="all",
        )
        return [
            {
                "video_key": r[0],
                "video_sha": r[1],
                "layout_version": r[2],
                "origin": r[3],
                "video_path": r[4],
            }
            for r in rows
        ]

    def save_probe_entry(self, row: dict[str, Any]) -> None:
        import json as _json

        layout = row.get("layout")
        self._run(
            "save_probe_entry",
            """INSERT INTO odc_probe_cache
               (video_key, video_sha, frames_probed, persons_seen, peaks,
                reason, layout, updated_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,now())
               ON CONFLICT (video_key) DO UPDATE SET
                 video_sha = EXCLUDED.video_sha, frames_probed = EXCLUDED.frames_probed,
                 persons_seen = EXCLUDED.persons_seen, peaks = EXCLUDED.peaks,
                 reason = EXCLUDED.reason, layout = EXCLUDED.layout,
                 updated_at = now()""",
            (
                row["video_key"],
                row.get("video_sha", ""),
                int(row.get("frames_probed", 0)),
                int(row.get("persons_seen", 0)),
                int(row.get("peaks", 0)),
                row.get("reason", ""),
                _json.dumps(layout) if layout is not None else None,
            ),
        )

    def load_probe_entry(self, video_key: str) -> dict[str, Any] | None:
        row = self._run(
            "load_probe_entry",
            """SELECT video_key, video_sha, frames_probed, persons_seen, peaks,
                      reason, layout, updated_at
               FROM odc_probe_cache WHERE video_key = %s""",
            (video_key,),
            fetch="one",
        )
        if row is None:
            return None
        return {
            "video_key": row[0],
            "video_sha": row[1],
            "frames_probed": row[2],
            "persons_seen": row[3],
            "peaks": row[4],
            "reason": row[5],
            "layout": row[6] if isinstance(row[6], dict) else None,
            "updated_at": row[7].isoformat() if row[7] else None,
        }

    def save_analytics_entry(self, row: dict[str, Any]) -> None:
        import json as _json

        self._run(
            "save_analytics_entry",
            """INSERT INTO odc_analytics_snapshot
               (cache_key, video_key, layout_hash, config_hash, payload, updated_at)
               VALUES (%s,%s,%s,%s,%s::jsonb,now())
               ON CONFLICT (cache_key) DO UPDATE SET
                 video_key = EXCLUDED.video_key, layout_hash = EXCLUDED.layout_hash,
                 config_hash = EXCLUDED.config_hash, payload = EXCLUDED.payload,
                 updated_at = now()""",
            (
                row["cache_key"],
                row.get("video_key", ""),
                row.get("layout_hash", ""),
                row.get("config_hash", ""),
                _json.dumps(row.get("payload", {})),
            ),
        )

    def load_analytics_entry(self, cache_key: str) -> dict[str, Any] | None:
        row = self._run(
            "load_analytics_entry",
            """SELECT cache_key, video_key, layout_hash, config_hash, payload
               FROM odc_analytics_snapshot WHERE cache_key = %s""",
            (cache_key,),
            fetch="one",
        )
        if row is None:
            return None
        return {
            "cache_key": row[0],
            "video_key": row[1],
            "layout_hash": row[2],
            "config_hash": row[3],
            "payload": row[4] if isinstance(row[4], dict) else {},
        }

    def read_validation_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        count = max(1, min(int(limit), 200))
        rows = self._run(
            "read_validation_runs",
            f"""SELECT run_id, build_version, dataset_id, hardware_profile,
                      status, started_at, completed_at
               FROM odc_validation_run ORDER BY started_at DESC LIMIT {count}""",
            None,
            fetch="all",
        )
        return [
            {
                "run_id": r[0],
                "build_version": r[1],
                "dataset_id": r[2],
                "hardware_profile": r[3],
                "status": r[4],
                "started_at": r[5].isoformat() if r[5] else None,
                "completed_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ]

    def list_config_versions(self) -> list[str]:
        rows = self._run(
            "list_config_versions",
            "SELECT DISTINCT config_version FROM odc_event ORDER BY config_version",
            None,
            fetch="all",
        )
        return [r[0] for r in rows]

    def count_sessions(self) -> int:
        row = self._run("count_sessions", "SELECT COUNT(*) FROM odc_session", None, fetch="one")
        return int(row[0]) if row else 0

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        count = max(1, min(int(limit), 200))
        rows = self._run(
            "list_sessions",
            f"SELECT session_id, camera_id, status, started_at FROM odc_session "
            f"ORDER BY started_at DESC LIMIT {count}",
            None,
            fetch="all",
        )
        return [
            {
                "session_id": r[0],
                "camera_id": r[1],
                "status": r[2],
                "started_at": r[3].isoformat() if r[3] else None,
            }
            for r in rows
        ]

    def recent_critical_events(self, limit: int = 50) -> list[StateEvent]:
        from .domain import EventType, Severity

        count = max(1, min(int(limit), 200))
        rows = self._run(
            "recent_critical_events",
            f"""SELECT event_id, session_id, event_type, severity, seat_id,
                      track_id, occurred_at, recorded_at, prior_state, new_state,
                      reason_codes, evidence_summary, config_version,
                      model_set_version
               FROM odc_event WHERE severity = 'CRITICAL'
               ORDER BY occurred_at DESC LIMIT {count}""",
            None,
            fetch="all",
        )
        return [
            StateEvent(
                event_id=r[0],
                session_id=r[1],
                event_type=EventType(r[2]),
                severity=Severity(r[3]),
                seat_id=r[4],
                track_id=r[5],
                occurred_at=r[6],
                recorded_at=r[7],
                prior_state=r[8],
                new_state=r[9],
                reason_codes=list(r[10]),
                evidence_summary=r[11],
                config_version=r[12],
                model_set_version=r[13],
            )
            for r in rows
        ]
