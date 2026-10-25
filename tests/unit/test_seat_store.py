"""PLOMID-backed seat store tests — fingerprinting, reuse, invalidation.

Videos stay on disk; seat maps, probes and analytics live in PLOMID. The
FakeRepository from test_persistence_unit stands in for the database here
(same contract); live SQL is covered by tests/integration/test_seat_store_pg.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from odc.autolayout import ProbeResult
from odc.domain import SeatGeometry
from odc.seat_store import (
    analytics_cache_key,
    layout_fingerprint,
    layout_from_json,
    layout_to_json,
    load_analytics_snapshot,
    load_named_layout,
    load_probe_cache,
    load_video_seatmap,
    save_analytics_snapshot,
    save_named_layout,
    save_probe_cache,
    save_video_seatmap,
    video_fingerprint,
)
from odc.seats import SeatLayout
from tests.unit.test_persistence_unit import FakeRepository


@pytest.fixture
def repo() -> FakeRepository:
    return FakeRepository()


@pytest.fixture
def video(tmp_path: Path) -> Path:
    target = tmp_path / "test-clip.mp4"
    target.write_bytes(b"\x00\x01fake-video" * 1000)
    return target


def _layout() -> SeatLayout:
    return SeatLayout(
        camera_profile="t", version="v1",
        seats=[SeatGeometry(
            seat_id="driver", polygon=[(0, 0), (0.4, 0), (0.4, 1), (0, 1)],
            camera_profile="t", version="v1", row=0, col=0,
        )],
    )


def test_fingerprint_missing_file_is_honest(tmp_path: Path):
    fp = video_fingerprint(tmp_path / "nope.mp4")
    assert fp["ok"] is False and "missing" in fp["reason"]


def test_seatmap_round_trip_in_plomid(repo: FakeRepository, video: Path):
    key = save_video_seatmap(repo, video, _layout(), camera_id="c0", origin="test")
    assert key is not None
    loaded, reason = load_video_seatmap(repo, video)
    assert loaded is not None, reason
    assert [s.seat_id for s in loaded.seats] == ["driver"]
    assert loaded.seats[0].row == 0 and loaded.seats[0].col == 0
    # Video asset tracked alongside.
    assert repo.list_video_assets_by_path(str(video))


def test_seatmap_missing_is_graceful(repo: FakeRepository, video: Path):
    loaded, reason = load_video_seatmap(repo, video)
    assert loaded is None and "no saved seat map" in reason


def test_seatmap_invalidates_when_video_changes(repo: FakeRepository, video: Path):
    assert save_video_seatmap(repo, video, _layout()) is not None
    assert save_probe_cache(repo, video, ProbeResult(None, 4, 0, 0, "none")) is True
    video.write_bytes(b"completely different bytes" * 9999)
    loaded, reason = load_video_seatmap(repo, video)
    assert loaded is None and "changed" in reason
    probe, probe_reason = load_probe_cache(repo, video)
    assert probe is None and "changed" in probe_reason


def test_seatmap_corrupt_row_is_graceful(repo: FakeRepository, video: Path):
    key = save_video_seatmap(repo, video, _layout())
    assert key is not None
    repo.video_seatmaps[key]["layout"] = {"bogus": True}
    loaded, reason = load_video_seatmap(repo, video)
    assert loaded is None and "invalid" in reason


def test_probe_cache_reuse(repo: FakeRepository, video: Path):
    probe = ProbeResult(_layout(), 8, 12, 1, "1 hotspot(s) from 8 frames")
    assert save_probe_cache(repo, video, probe) is True
    cached, reason = load_probe_cache(repo, video)
    assert cached is not None, reason
    assert cached.layout is not None and len(cached.layout.seats) == 1


def test_analytics_reused_until_inputs_change(repo: FakeRepository, video: Path):
    layout = _layout()
    payload = {"counts": {"total": 1}}
    assert save_analytics_snapshot(repo, video, layout, "cfg-1", payload) is not None
    hit, _ = load_analytics_snapshot(repo, video, layout, "cfg-1")
    assert hit == payload
    other = SeatLayout(
        camera_profile="t", version="v2",
        seats=[SeatGeometry(
            seat_id="driver", polygon=[(0.5, 0), (1, 0), (1, 1), (0.5, 1)],
            camera_profile="t", version="v2",
        )],
    )
    assert layout_fingerprint(layout) != layout_fingerprint(other)
    miss, reason = load_analytics_snapshot(repo, video, other, "cfg-1")
    assert miss is None and "recalculation" in reason
    miss2, _ = load_analytics_snapshot(repo, video, layout, "cfg-2")
    assert miss2 is None
    assert analytics_cache_key(tmp_path_missing(video), layout, "cfg-1") is None


def tmp_path_missing(video: Path) -> Path:
    return video.parent / "missing.mp4"


def test_layout_fingerprint_stable():
    assert layout_fingerprint(_layout()) == layout_fingerprint(_layout())


def test_config_hash_changes_with_settings():
    from odc.config import AppConfig
    from odc.seat_store import config_hash_for

    assert config_hash_for(AppConfig()) == config_hash_for(AppConfig())
    other = AppConfig()
    other.detection.threshold = 0.9
    assert config_hash_for(AppConfig()) != config_hash_for(other)


def test_layout_json_round_trip():
    doc = layout_to_json(_layout())
    back = layout_from_json(doc)
    assert [s.seat_id for s in back.seats] == ["driver"]
    assert back.seats[0].row == 0


def test_named_layout_save_load_and_overwrite(repo: FakeRepository):
    seats = [{"seat_id": "driver", "polygon": [[0, 0], [0.4, 0], [0.4, 1], [0, 1]]}]
    ok, name = save_named_layout(repo, "test-van", seats)
    assert ok and name == "test-van"
    ok, msg = save_named_layout(repo, "test-van", seats)
    assert not ok and "exists" in msg
    ok, _ = save_named_layout(repo, "test-van", seats, overwrite=True)
    assert ok
    layout, reason = load_named_layout(repo, "test-van")
    assert layout is not None, reason
    assert [s.seat_id for s in layout.seats] == ["driver"]
    ok, msg = save_named_layout(repo, "../evil", seats)
    assert not ok


def test_named_layout_seed_fallback(repo: FakeRepository):
    # Curated YAML seeds import into PLOMID on first load.
    layout, reason = load_named_layout(repo, "demo-f1-v0")
    assert layout is not None, reason
    assert len(layout.seats) == 3
    assert repo.load_seat_layout("demo-f1-v0") is not None
    assert "demo-f1-v0" in [
        r["layout_name"] for r in repo.list_seat_layouts()
    ]
    missing, _ = load_named_layout(repo, "no-such-layout")
    assert missing is None
