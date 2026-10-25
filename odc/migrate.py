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
    does not apply the whole file. This splitter respects single-quoted
    strings and `--` line comments, so semicolons inside either do not
    create fake statements. ODC migrations use no dollar-quoted strings or
    block comments.
    """
    statements: list[str] = []
    buf: list[str] = []
    in_string = False
    in_comment = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if in_comment:
            if ch == "\n":
                in_comment = False
                buf.append(ch)
            i += 1
            continue
        if in_string:
            buf.append(ch)
            if ch == "'":
                if i + 1 < len(sql) and sql[i + 1] == "'":
                    buf.append("'")
                    i += 2
                    continue
                in_string = False
            i += 1
            continue
        if ch == "'":
            in_string = True
            buf.append(ch)
            i += 1
            continue
        if ch == "-" and i + 1 < len(sql) and sql[i + 1] == "-":
            in_comment = True
            i += 2
            continue
        if ch == ";":
            text = "".join(buf).strip()
            if text:
                statements.append(text)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    text = "".join(buf).strip()
    if text:
        statements.append(text)
    return statements


def qualify_statement(statement: str) -> str:
    """Make unqualified ODC objects deterministic inside one migration.

    PLOMID resets the effective search path for DDL, so table references are
    prefixed explicitly. Index names keep their historical prefix mapping,
    and a RENAME TO target is left unqualified because it names the new
    object rather than referencing a table.
    """
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
        qualified = qualified.replace("CREATE INDEX " + old_index, "CREATE INDEX " + new_index)
    # A RENAME TO target is an object name, not a table reference. The
    # generic qualifier above would turn the target into
    # "plomid_odc.<name>", which PLOMID rejects.
    return re.sub(
        r"(?i)(RENAME\s+TO\s+)plomid_odc\.",
        r"\1",
        qualified,
    )


def apply_pending(conn: psycopg.Connection, target_version: int | None = None) -> list[int]:
    """Apply unapplied migrations in order, one statement per transaction.

    PLOMID discards DML when the same transaction also contains DDL, so a
    migration that rebuilds a table and copies rows must never run inside
    one long transaction. Each statement commits on its own; the version
    row is written only after every statement succeeds.

    The search path is set inside each statement's transaction because DDL
    resets it. This keeps unqualified index names in plomid_odc on fresh
    installs while data copies still commit.

    ``target_version`` is for tests and staged upgrades; production callers
    leave it unset and apply every pending migration.
    """
    done = applied_versions(conn)
    applied: list[int] = []
    previous_autocommit = conn.autocommit
    if not previous_autocommit:
        conn.commit()
        conn.autocommit = True
    try:
        for version, path in list_migrations():
            if target_version is not None and version > target_version:
                break
            if version in done:
                continue
            for statement in split_statements(path.read_text()):
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute("SET search_path TO plomid_odc, public")
                        cur.execute(qualify_statement(statement))
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO plomid_odc.odc_schema_version (schema_version) VALUES (%s)",
                        (version,),
                    )
            applied.append(version)
    finally:
        if not previous_autocommit:
            conn.autocommit = False
    return applied


def current_version(conn: psycopg.Connection) -> int:
    done = applied_versions(conn)
    return max(done) if done else 0
