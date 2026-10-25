"""Pilot dossier builder (Phase 8) — real outputs only, gaps stated plainly."""

from __future__ import annotations

from typing import Any

from .config import AppConfig
from .domain import ModelManifest
from .seats import SeatLayout


def build_dossier(
    cfg: AppConfig,
    layout: SeatLayout,
    manifest: ModelManifest | None,
    validation_runs: list[dict[str, Any]],
    config_versions: list[str],
    session_count: int,
    fusion_status: str = "VSS interface + file-replay + MQTT adapter; no OEM bus connected",
    guardian_version: str = "guardian-v1 (gated, advisory only)",
) -> dict[str, Any]:
    return {
        "product": "PLOMID ODC — Occupant & Cabin State Platform",
        "build_version": cfg.build_version,
        "mode": "engineering demonstrator and pilot platform — NOT production-certified",
        "architecture": [
            "capture → pipeline → onnxruntime → perception → tracking → seats",
            "→ temporal state engine → PLOMID SQL repository → api/console",
        ],
        "database": "PLOMID only, SQL-only boundary (see ODC_SQL_CAPABILITIES.md)",
        "model": manifest.model_dump(mode="json") if manifest else None,
        "seat_layout": {
            "profile": layout.camera_profile,
            "version": layout.version,
            "seats": [s.seat_id for s in layout.seats],
        },
        "state_rules": cfg.state.version,
        "validation_runs": validation_runs,
        "config_versions_seen": config_versions,
        "sessions_recorded": session_count,
        "oem_integration": {
            "vehicle_signals": fusion_status,
            "guardian": guardian_version,
            "fleet": "aggregate rollups + criticals only; no raw media leaves vehicles",
            "vectors": (
                "deferred with rationale (no onboard embedder; relational suffices; privacy)"
            ),
        },
        "known_gaps": [
            "child-seat / seatbelt / apparent-size: gated candidates, evidence in ODC_ONTOLOGY.md",
            "no breathing/heartbeat CPD from camera; fusion interface required",
            "no ECU latency promises from laptop numbers",
            "no ISO 26262 / ASPICE / 21434 certification evidence",
            "pose model commercial terms unverified (REVIEW_REQUIRED)",
        ],
        "recovery": "camera/model/SQL failures degrade visibly; bounded queues; STALE expiry",
        "privacy": "edge-local, metadata-only, no raw video retention, no identity",
    }


def render_markdown(dossier: dict[str, Any]) -> str:
    lines = [
        f"# {dossier['product']}",
        "",
        f"Build: {dossier['build_version']}",
        f"Status: {dossier['mode']}",
        "",
    ]
    if dossier["model"] is not None:
        lines += [
            "## Model",
            f"- {dossier['model']['model_id']} "
            f"(commercial approved: {dossier['model']['approved_for_commercial_use']})",
            "",
        ]
    lines += [
        "## Seat layout",
        f"- {dossier['seat_layout']['profile']} "
        f"{dossier['seat_layout']['version']}: " + ", ".join(dossier["seat_layout"]["seats"]),
        "",
    ]
    lines += ["## Known gaps"]
    lines += [f"- {gap}" for gap in dossier["known_gaps"]]
    return "\n".join(lines) + "\n"
