"""Seat-store integration — every row here travels SQL-over-PG-wire to real PLOMID.

Videos stay on disk (tmp files); seat maps, probes and analytics live in
PLOMID tables (V003). Provisioning: `plomid_dsn` fixture.
"""

from __future__ import annotations

from pathlib import Path

from odc.autolayout import ProbeResult
from odc.domain import SeatGeometry
from odc.persistence import PlomidRepository
from odc.seat_store import (
    load_analytics_snapshot,
    load_named_layout,
    load_probe_cache,
    load_video_seatmap,
    save_analytics_snapshot,
    save_named_layout,
    save_probe_cache,
    save_video_seatmap,
)
from odc.seats import SeatLayout


def _video(tmp_path: Path) -> Path:
    target = tmp_path / "test-seat-store.mp4"
    target.write_bytes(b"\x00\x07seat-store-video" * 500)
    return target


def _layout() -> SeatLayout:
    return SeatLayout(
        camera_profile="t", version="v1",
        seats=[SeatGeometry(
            seat_id="driver", polygon=[(0, 0), (0.4, 0), (0.4, 1), (0, 1)],
            camera_profile="t", version="v1", row=0, col=1,
        )],
    )


def test_video_seatmap_roundtrip_pg(plomid_dsn, tmp_path):
    repo = PlomidRepository(plomid_dsn)
    try:
        repo.ensure_schema_version(3)
        video = _video(tmp_path)
        assert save_video_seatmap(repo, video, _layout(), camera_id="c0") is not None
        loaded, reason = load_video_seatmap(repo, video)
        assert loaded is not None, reason
        assert [s.seat_id for s in loaded.seats] == ["driver"]
        assert (loaded.seats[0].row, loaded.seats[0].col) == (0, 1)
        assert repo.list_video_assets_by_path(str(video))
    finally:
        repo.close()


def test_named_layout_save_and_seed_pg(plomid_dsn):
    repo = PlomidRepository(plomid_dsn)
    try:
        repo.ensure_schema_version(3)
        seats = [{"seat_id": "driver", "polygon": [[0, 0], [0.4, 0], [0.4, 1], [0, 1]]}]
        ok, _ = save_named_layout(repo, "test-pg-van", seats)
        assert ok
        ok, msg = save_named_layout(repo, "test-pg-van", seats)
        assert not ok and "exists" in msg
        layout, reason = load_named_layout(repo, "test-pg-van")
        assert layout is not None, reason
        assert layout.version == "test-pg-van"
        names = [r["layout_name"] for r in repo.list_seat_layouts()]
        assert "test-pg-van" in names
        versions = {r["layout_name"]: r["version"] for r in repo.list_seat_layouts()}
        assert versions["test-pg-van"] == "test-pg-van"
        # Seed fallback imports curated YAML on first load.
        seed, _ = load_named_layout(repo, "demo-e1-v0")
        assert seed is not None and seed.seats
    finally:
        repo.close()


def test_probe_and_analytics_reuse_pg(plomid_dsn, tmp_path):
    repo = PlomidRepository(plomid_dsn)
    try:
        repo.ensure_schema_version(3)
        video = _video(tmp_path)
        assert save_probe_cache(repo, video, ProbeResult(_layout(), 6, 9, 1, "x")) is True
        probe, reason = load_probe_cache(repo, video)
        assert probe is not None, reason
        assert probe.layout is not None and len(probe.layout.seats) == 1
        payload = {"counts": {"total": 1}}
        assert save_analytics_snapshot(repo, video, _layout(), "cfg-1", payload) is not None
        hit, _ = load_analytics_snapshot(repo, video, _layout(), "cfg-1")
        assert hit == payload
        miss, reason = load_analytics_snapshot(repo, video, _layout(), "cfg-2")
        assert miss is None and "recalculation" in reason
    finally:
        repo.close()
