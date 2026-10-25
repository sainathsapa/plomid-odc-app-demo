"""Config + health + CLI tests."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from odc.config import AppConfig, load_config
from odc.domain import HealthStatus, RunMode
from odc.health import build_health

ODC = [sys.executable, "-m", "odc.cli"]


def test_defaults_no_secrets():
    cfg = AppConfig()
    assert cfg.plomid.dsn is None
    assert cfg.mode == RunMode.REPLAY
    assert cfg.detection.threshold == 0.7


def test_example_yaml_loads():
    cfg = load_config("odc.example.yaml")
    assert cfg.camera.camera_id == "cabin0"
    assert cfg.processing.sample_every_n == 6


def test_invalid_config_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("mode: FLY\n")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_dsn_via_env_only(monkeypatch):
    monkeypatch.setenv("ODC_DSN", "postgresql://u:p@h:1/db")
    cfg = load_config("odc.example.yaml")
    assert cfg.plomid.dsn is not None
    assert "u:p" not in cfg.model_dump_json(exclude={"plomid": {"dsn"}})


def test_dotenv_file_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "ODC_MODE=LIVE\nODC_CAMERA__TARGET_FPS=15.0\nODC_DSN=postgresql://u:pw@h:1/db\n"
    )
    monkeypatch.delenv("ODC_MODE", raising=False)
    monkeypatch.delenv("ODC_CAMERA__TARGET_FPS", raising=False)
    monkeypatch.delenv("ODC_DSN", raising=False)
    cfg = load_config(None)
    assert cfg.mode == RunMode.LIVE
    assert cfg.camera.target_fps == 15.0
    assert cfg.plomid.dsn is not None


def test_yaml_beats_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("ODC_MODE=LIVE\n")
    (tmp_path / "c.yaml").write_text("mode: REPLAY\n")
    monkeypatch.delenv("ODC_MODE", raising=False)
    assert load_config(tmp_path / "c.yaml").mode == RunMode.REPLAY


def test_health_unknown_db_degrades():
    health = build_health(AppConfig(), HealthStatus.UNKNOWN, "not probed")
    assert health.overall == HealthStatus.DEGRADED
    assert health.components[0].name == "config"


def test_cli_version_runs_clean():
    r = subprocess.run([*ODC, "version"], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and r.stdout.strip() == "0.1.0"


def test_cli_config_check_no_secrets():
    r = subprocess.run(
        [*ODC, "--config", "odc.example.yaml", "config-check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert json.loads(r.stdout)["camera"]["camera_id"] == "cabin0"


def test_cli_health_without_db():
    r = subprocess.run([*ODC, "health"], capture_output=True, text=True, check=False)
    assert r.returncode == 0  # DEGRADED, not failure
    body = json.loads(r.stdout)
    assert body["overall"] == "DEGRADED"
    assert body["mode"] == "REPLAY"
