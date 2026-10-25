"""Phase 12 tests — selftest honesty, bootstrap provisioning, purge."""

from __future__ import annotations

import subprocess
import sys


def _run(*args: str, env: dict | None = None):
    import os

    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, "-m", "odc.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        env=merged,
    )


def _json_out(stdout: str) -> dict:
    import json as _json

    lines = stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "{")
    return _json.loads("\n".join(lines[start:]))


def test_selftest_without_dsn_skips_honestly():
    r = _run("selftest", env={"ODC_DSN": ""})
    assert r.returncode == 0
    body = _json_out(r.stdout)
    statuses = {c["check"]: c["status"] for c in body["checks"]}
    assert statuses["config"] == "pass"
    assert statuses["plomid"] == "skip"
    assert "secret" not in r.stdout.lower()


def test_selftest_bad_config_fails():
    r = _run("--config", "footage/nope.yaml", "selftest")
    assert r.returncode == 1


def test_bootstrap_needs_dsn():
    r = _run("bootstrap", env={"ODC_DSN": ""})
    assert r.returncode == 1


def test_bootstrap_provisions_live(plomid_dsn):

    from odc.migrate import list_migrations

    r = _run("bootstrap", env={"ODC_DSN": plomid_dsn})
    assert r.returncode == 0
    latest = max(v for v, _ in list_migrations())
    assert f"schema v{latest}" in r.stdout


def test_selftest_live_reports_plomid(plomid_dsn):

    r = _run("selftest", env={"ODC_DSN": plomid_dsn})
    assert r.returncode == 0
    body = _json_out(r.stdout)
    statuses = {c["check"]: c["status"] for c in body["checks"]}
    assert statuses["plomid"] == "pass"
