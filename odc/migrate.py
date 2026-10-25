"""ODC schema migrations (Phase 2) — versioned, ordered, fail-closed."""

from __future__ import annotations

import re
from pathlib import Path

import psycopg

from .domain import RepositoryError, SqlError, SqlErrorKind

MIGRATION_RE = re.compile(r"^V(\d+)__.+\.sql$")
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "sql" / "migrations"
ODC_TABLES = (
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
    "odc_frame_detection",
    "odc_frame_observation",
    "odc_assistant_interaction",
    "odc_schema_version",
)


def list_migrations(directory: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in sorted(directory.glob("V*.sql")):
        match = MIGRATION_RE.match(path.name)
        if not match:
            raise RepositoryError(
                SqlError(
                    kind=SqlErrorKind.SCHEMA_MISMATCH,
                    message=f"bad migration filename: {path.name}",
                )
            )
        found.append((int(match.group(1)), path))
    versions = [v for v, _ in found]
    if len(set(versions)) != len(versions):
        raise RepositoryError(
            SqlError(
                kind=SqlErrorKind.SCHEMA_MISMATCH,
                message="duplicate migration versions",
            )
        )
    return found


def applied_versions(conn: psycopg.Connection) -> set[int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'plomid_odc' AND table_name = 'odc_schema_version'"
        )
        row = cur.fetchone()
        if row is None or row[0] == 0:
            cur.execute(
                "CREATE TABLE plomid_odc.odc_schema_version (schema_version BIGINT PRIMARY KEY, "
                "applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            return set()
        cur.execute("SELECT schema_version FROM plomid_odc.odc_schema_version")
        return {r[0] for r in cur.fetchall()}


def split_statements(sql: str) -> list[str]:
    """Split a migration file into single statements.

    PLOMID executes one statement per request — multi-statement execute()
    does not apply the whole file. ODC migrations are plain DDL with no
    semicolons inside statements, so a semicolon split is exact. Leading
    `--` comment lines are stripped per chunk (never whole chunks).
    """
    statements: list[str] = []
    for part in sql.split(";"):
        lines = [line for line in part.splitlines() if not line.strip().startswith("--")]
        text = "\n".join(lines).strip().rstrip(";").strip()
        if text:
            statements.append(text)
    return statements


def apply_pending(conn: psycopg.Connection) -> list[int]:
    """Apply unapplied migrations in order, each in its own transaction."""
    done = applied_versions(conn)
    applied: list[int] = []
    for version, path in list_migrations():
        if version in done:
            continue
        with conn.transaction():
            with conn.cursor() as cur:
                # PLOMID starts a fresh transaction for each migration; set
                # the namespace inside that transaction as well as at connect
                # time so unqualified legacy migration SQL is deterministic.
                cur.execute("SET search_path TO plomid_odc, public")
                for statement in split_statements(path.read_text()):
                    # PLOMID transaction handling resets effective search path
                    # for DDL, so qualify every ODC object explicitly.
                    qualified = statement
                    for table in sorted(ODC_TABLES, key=len, reverse=True):
                        qualified = re.sub(
                            rf"(?<![A-Za-z0-9_.]){table}(?![A-Za-z0-9_])",
                            f"plomid_odc.{table}",
                            qualified,
                        )
                    for old_index, new_index in (
                        ("odc_event_session_idx", "plomid_odc_event_session_idx"),
                        ("odc_video_asset_path_idx", "plomid_odc_video_asset_path_idx"),
                        ("odc_metric_session_idx", "plomid_odc_metric_session_idx"),
                        ("odc_event_type_idx", "plomid_odc_event_type_idx"),
                        (
                            "odc_frame_observation_session_idx",
                            "plomid_odc_frame_observation_session_idx",
                        ),
                        ("odc_assistant_session_idx", "plomid_odc_assistant_session_idx"),
                    ):
                        qualified = qualified.replace(
                            "CREATE INDEX " + old_index, "CREATE INDEX " + new_index
                        )
                    cur.execute(qualified)
                cur.execute(
                    "INSERT INTO plomid_odc.odc_schema_version (schema_version) VALUES (%s)",
                    (version,),
                )
        applied.append(version)
    return applied


def current_version(conn: psycopg.Connection) -> int:
    done = applied_versions(conn)
    return max(done) if done else 0
