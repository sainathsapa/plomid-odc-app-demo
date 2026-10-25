"""ODC application health model (Phase 1)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from .config import AppConfig
from .domain import HealthStatus, RunMode


class ComponentHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: HealthStatus
    detail: str = ""
    checked_at: datetime = datetime.now(UTC)


class AppHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: str
    build_version: str
    mode: RunMode
    overall: HealthStatus
    components: list[ComponentHealth]
    checked_at: datetime = datetime.now(UTC)


def build_health(cfg: AppConfig, db_status: HealthStatus, db_detail: str = "") -> AppHealth:
    components = [
        ComponentHealth(name="config", status=HealthStatus.HEALTHY, detail="validated"),
        ComponentHealth(name="plomid", status=db_status, detail=db_detail or "not probed"),
    ]
    overall = db_status if db_status != HealthStatus.HEALTHY else HealthStatus.HEALTHY
    if db_status == HealthStatus.UNKNOWN:
        overall = HealthStatus.DEGRADED
    return AppHealth(
        app=cfg.app,
        build_version=cfg.build_version,
        mode=cfg.mode,
        overall=overall,
        components=components,
    )
