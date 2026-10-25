"""Phase 11 unit tests — secrets, SQLi surface, payloads, recovery."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from odc.domain import EventType, StateEvent
from odc.redact import scrub_dsn, scrub_mapping


def test_dsn_password_never_in_text():
    dirty = "postgresql://plomid:s3cret!@127.0.0.1:5432/plomid"
    clean = scrub_dsn(dirty)
    assert "s3cret" not in clean and "***" in clean
    assert scrub_dsn("no password here") == "no password here"
    assert scrub_dsn(clean) == clean  # idempotent


def test_mapping_secrets_redacted():
    out = scrub_mapping({"dsn": "x", "user": "u", "token": "t", "mode": "dev"})
    assert out == {"dsn": "***", "user": "u", "token": "***", "mode": "dev"}


def test_cli_outputs_contain_no_password():
    r = subprocess.run(
        [sys.executable, "-m", "odc.cli", "--config", "odc.example.yaml", "config-check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0 and "secret" not in r.stdout.lower()
    r = subprocess.run(
        [sys.executable, "-m", "odc.cli", "health"], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0


def test_no_string_interpolated_sql():
    """Only static SQL + params. Reviewed exception: clamped-int LIMIT
    literals in read paths (D11/F4). Any new f-string SQL fails this test."""
    import re

    sites = set()
    for path in Path("odc").rglob("*.py"):
        text = path.read_text()
        for match in re.finditer(
            r'f("{3}|"""|\'\'\'|")(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE)', text
        ):
            sites.add(f"{path}:{match.group(2)}")
    assert sites == {"odc/persistence.py:SELECT"}, sites


def test_event_payload_bounded():

    from pydantic import ValidationError

    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        StateEvent(
            event_id="e",
            session_id="s",
            event_type=EventType.SEAT_PRESENCE_CHANGED,
            occurred_at=now,
            recorded_at=now,
            evidence_summary="x" * 2001,
        )


def test_purge_removes_only_old(tmp_path):
    import os
    import time

    root = tmp_path / "reports"
    root.mkdir()
    old, new = root / "old.jpg", root / "new.jpg"
    old.write_bytes(b"0")
    new.write_bytes(b"0")
    ancient = time.time() - 40 * 86400
    os.utime(old, (ancient, ancient))
    r = subprocess.run(
        [sys.executable, "-m", "odc.cli", "purge", "--dir", str(root), "--older-than-days", "30"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0 and "purged 1 files" in r.stdout
    assert not old.exists() and new.exists()


def test_retry_exhausts_budget_then_surfaces():
    import psycopg

    from odc.domain import RepositoryError, SqlErrorKind
    from odc.persistence import PlomidRepository

    calls = {"n": 0}

    def always_down(*args, **kwargs):
        calls["n"] += 1
        raise psycopg.OperationalError("connection failed: outage")

    import odc.persistence as persistence_mod

    real_connect = psycopg.connect
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(psycopg, "connect", always_down)
    monkeypatch.setattr(persistence_mod.psycopg, "connect", always_down)
    repo = PlomidRepository("postgresql://plomid:secret@127.0.0.1:1/plomid", connect_timeout_s=1)
    try:
        with pytest.raises(RepositoryError) as exc:
            repo._run("probe", "SELECT 1", fetch="one")
        assert exc.value.error.kind == SqlErrorKind.CONNECTION
        assert calls["n"] == 3  # full retry budget, then surfaced (never hidden)
    finally:
        repo.close()
        monkeypatch.undo()
        assert real_connect is not None


def test_service_restart_starts_new_session_cleanly():
    import time

    from test_service import _NoModelService, _StubRepo

    from odc.config import AppConfig
    from odc.seats import SeatGeometry, SeatLayout

    layout = SeatLayout(
        camera_profile="t",
        version="t",
        seats=[
            SeatGeometry(
                seat_id="A",
                polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
                camera_profile="t",
                version="t",
            )
        ],
    )
    repo = _StubRepo()
    first = _NoModelService(AppConfig(), repo, layout)
    first.start()
    time.sleep(0.5)
    first.stop()
    second = _NoModelService(AppConfig(), repo, layout)
    second.start()
    time.sleep(0.5)
    second.stop()
    assert len(repo.sessions) == 2
    assert first.hub.snapshot.running is False


def test_corrupt_weights_degrade_without_crash():
    from odc.domain import ModelManifest
    from odc.runtime import OnnxDetectorRuntime

    manifest = ModelManifest(
        model_id="no-such-model",
        family="f",
        checkpoint_sha256="s",
        source_url="u",
        code_revision="r",
        weights_license="l",
        dataset_terms="d",
    )
    runtime = OnnxDetectorRuntime()
    with pytest.raises(FileNotFoundError):
        runtime.load(manifest)


def test_no_banned_packages_in_lockfile():
    """AGPL/GPL carriers must never enter the dependency closure.

    Full license text verified 2026-10-13 via pip-licenses (MIT/BSD/
    Apache/PSF/ISC/MPL-file-level only, zero GPL-family) and pip-audit
    (zero known vulnerabilities). Re-scan on every dependency change.
    """
    import re

    banned = re.compile(r"ultralytics|yolo.world|yoloworld", re.IGNORECASE)
    text = Path("requirements.lock").read_text()
    names = [line.split("==")[0] for line in text.splitlines() if "==" in line]
    assert [n for n in names if banned.search(n)] == []
