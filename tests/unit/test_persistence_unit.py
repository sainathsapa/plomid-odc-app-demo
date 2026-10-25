"""Unit tests: in-memory fake repository + migration file rules.

Fakes prove the Repository contract is implementable without a database.
No SQL runs here; live SQL is covered by tests/integration/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from odc import contracts
from odc.domain import (
    CameraHealth,
    ModelManifest,
    SeatState,
    SessionMeta,
    StateEvent,
)
from odc.migrate import MIGRATIONS_DIR, RepositoryError, list_migrations


class FakeRepository(contracts.Repository):
    def __init__(self) -> None:
        self.sessions: dict[str, SessionMeta] = {}
        self.seats: dict[tuple[str, str], SeatState] = {}
        self.events: dict[str, StateEvent] = {}
        self.frame_observations: dict[tuple[str, int], dict[str, Any]] = {}
        self.assistant_interactions: dict[str, dict[str, Any]] = {}
        self.layouts: dict[str, dict[str, Any]] = {}
        self.assets: dict[str, dict[str, Any]] = {}
        self.video_seatmaps: dict[str, dict[str, Any]] = {}
        self.probes: dict[str, dict[str, Any]] = {}
        self.analytics: dict[str, dict[str, Any]] = {}
        self.version = 2

    def health_check(self) -> bool:
        return True

    def ensure_schema_version(self, minimum: int) -> int:
        assert self.version >= minimum
        return self.version

    def start_session(self, session: SessionMeta) -> None:
        self.sessions[session.session_id] = session

    def end_session(self, session_id: str) -> None:
        self.sessions[session_id].status = "CLOSED"

    def upsert_camera_status(self, health: CameraHealth) -> None:
        assert health.camera_id

    def write_state_snapshot(self, seats: list[SeatState]) -> None:
        for seat in seats:
            self.seats[(seat.session_id, seat.seat_id)] = seat

    def write_frame_observation(
        self,
        session_id: str,
        frame_seq: int,
        captured_at: Any,
        seat_rows: list[dict[str, Any]],
        pose: dict[str, Any],
        specialist_decisions: list[dict[str, Any]],
    ) -> None:
        self.frame_observations[(session_id, frame_seq)] = {
            "frame_seq": frame_seq,
            "captured_at": captured_at.isoformat(),
            "payload": {
                "api_schema_version": "odc.seat-observation.v1",
                "seat_rows": seat_rows,
                "pose": pose,
                "specialist_decisions": specialist_decisions,
            },
        }

    def read_latest_frame_observation(self, session_id: str) -> dict[str, Any] | None:
        rows = [row for (sid, _), row in self.frame_observations.items() if sid == session_id]
        return max(rows, key=lambda row: row["frame_seq"]) if rows else None

    def append_assistant_interaction(self, row: dict[str, Any]) -> None:
        self.assistant_interactions.setdefault(row["interaction_id"], dict(row))

    def read_assistant_interactions(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self.assistant_interactions.values()
            if row.get("session_id") == session_id
        ]
        return sorted(rows, key=lambda row: row["occurred_at"], reverse=True)[:limit]

    def append_event(self, event: StateEvent) -> None:
        self.events.setdefault(event.event_id, event)

    def read_current_seat_states(self, session_id: str) -> list[SeatState]:
        return [s for (sid, _), s in self.seats.items() if sid == session_id]

    def read_event_timeline(
        self, session_id: str, seat_id: str | None = None, limit: int = 200
    ) -> list[StateEvent]:
        rows = [e for e in self.events.values() if e.session_id == session_id]
        if seat_id is not None:
            rows = [e for e in rows if e.seat_id == seat_id]
        return rows[:limit]

    def write_metric_batch(self, rows: list[dict[str, Any]]) -> None:
        assert isinstance(rows, list)

    def read_model_manifest(self, model_id: str) -> ModelManifest:
        raise AssertionError("not seeded in fake")

    def write_validation_result(self, rows: list[dict[str, Any]]) -> None:
        assert isinstance(rows, list)

    def upsert_video_asset(self, row: dict[str, Any]) -> None:
        self.assets[row["video_key"]] = dict(row)

    def list_video_assets_by_path(self, video_path: str) -> list[dict[str, Any]]:
        return [a for a in self.assets.values() if a.get("video_path") == video_path]

    def upsert_seat_layout(self, name: str, doc: dict[str, Any]) -> None:
        self.layouts[name] = dict(doc)

    def load_seat_layout(self, name: str) -> dict[str, Any] | None:
        return self.layouts.get(name)

    def list_seat_layouts(self) -> list[dict[str, Any]]:
        return [
            {
                "layout_name": name,
                "camera_profile": doc.get("camera_profile", ""),
                "version": doc.get("version", name),
                "origin": doc.get("origin", "hand"),
            }
            for name, doc in sorted(self.layouts.items())
        ]

    def save_video_seatmap(self, row: dict[str, Any]) -> None:
        self.video_seatmaps[row["video_key"]] = dict(row)

    def load_video_seatmap(self, video_key: str) -> dict[str, Any] | None:
        return self.video_seatmaps.get(video_key)

    def list_video_seatmaps_by_path(self, video_path: str) -> list[dict[str, Any]]:
        out = []
        for seatmap in self.video_seatmaps.values():
            asset = self.assets.get(seatmap["video_key"], {})
            if asset.get("video_path") == video_path:
                out.append(
                    {
                        "video_key": seatmap["video_key"],
                        "video_sha": seatmap.get("video_sha", ""),
                        "layout_version": seatmap.get("layout_version", ""),
                        "origin": seatmap.get("origin", ""),
                        "video_path": video_path,
                    }
                )
        return out

    def save_probe_entry(self, row: dict[str, Any]) -> None:
        self.probes[row["video_key"]] = dict(row)

    def load_probe_entry(self, video_key: str) -> dict[str, Any] | None:
        return self.probes.get(video_key)

    def save_analytics_entry(self, row: dict[str, Any]) -> None:
        self.analytics[row["cache_key"]] = dict(row)

    def load_analytics_entry(self, cache_key: str) -> dict[str, Any] | None:
        return self.analytics.get(cache_key)


def test_fake_implements_contract():
    assert not contracts.Repository.__abstractmethods__ - set(dir(FakeRepository))
    FakeRepository()


def test_fake_session_event_roundtrip(session_meta, seat_state, state_event):
    repo = FakeRepository()
    repo.start_session(session_meta)
    repo.write_state_snapshot([seat_state])
    repo.append_event(state_event)
    repo.append_event(state_event)  # idempotent repeat
    assert len(repo.read_current_seat_states("ses-test")) == 1
    assert len(repo.read_event_timeline("ses-test")) == 1
    assert repo.read_event_timeline("ses-test", seat_id="other") == []
    repo.end_session("ses-test")
    assert repo.sessions["ses-test"].status == "CLOSED"


def test_migrations_ordered_unique():
    found = list_migrations()
    versions = [v for v, _ in found]
    assert versions == sorted(versions) and len(versions) >= 2
    assert all(p.exists() for _, p in found)
    assert MIGRATIONS_DIR.is_dir()


def test_migrations_reject_bad_names(tmp_path: Path):
    (tmp_path / "V002__a.sql").write_text("SELECT 1")
    (tmp_path / "V002__b.sql").write_text("SELECT 1")
    with pytest.raises(RepositoryError):
        list_migrations(tmp_path)
    (tmp_path / "V002__b.sql").unlink()
    (tmp_path / "nope.sql").write_text("SELECT 1")
    found = list_migrations(tmp_path)
    assert [v for v, _ in found] == [2]
