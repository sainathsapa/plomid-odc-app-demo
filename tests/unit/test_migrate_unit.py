"""Unit tests for migration splitting and statement qualification."""

from __future__ import annotations

from odc.migrate import qualify_statement, split_statements


def test_split_ignores_semicolon_inside_comment():
    sql = "-- this comment; has a semicolon\nCREATE TABLE t (id TEXT);\n-- trailing; note\n"
    assert split_statements(sql) == ["CREATE TABLE t (id TEXT)"]


def test_split_ignores_semicolon_inside_string_literal():
    sql = "INSERT INTO t (note) VALUES ('a;b');\nINSERT INTO t (note) VALUES ('c''d;e');"
    assert split_statements(sql) == [
        "INSERT INTO t (note) VALUES ('a;b')",
        "INSERT INTO t (note) VALUES ('c''d;e')",
    ]


def test_split_skips_empty_statements():
    assert split_statements(";;  ;\nSELECT 1;;") == ["SELECT 1"]


def test_qualify_prefixes_tables_and_historical_index_names():
    assert (
        qualify_statement("CREATE INDEX odc_event_session_idx ON odc_event (session_id)")
        == "CREATE INDEX plomid_odc_event_session_idx ON plomid_odc.odc_event (session_id)"
    )


def test_qualify_does_not_rewrite_staging_suffix():
    statement = "CREATE TABLE odc_camera_status_fk (camera_id TEXT PRIMARY KEY)"
    assert qualify_statement(statement) == statement


def test_qualify_leaves_rename_target_unqualified():
    assert (
        qualify_statement("ALTER TABLE odc_camera_status RENAME TO odc_camera_status_old")
        == "ALTER TABLE plomid_odc.odc_camera_status RENAME TO odc_camera_status_old"
    )


def test_qualify_is_idempotent_for_qualified_names():
    statement = "SELECT 1 FROM plomid_odc.odc_session"
    assert qualify_statement(statement) == statement
