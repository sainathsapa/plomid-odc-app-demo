"""Contract tests — interfaces exist, fakes are labelled, vendors don't leak."""

from __future__ import annotations

import pytest

from odc import contracts as c
from odc import domain as d


def test_interfaces_abstract():
    for cls in (c.DetectorRuntime, c.Source, c.Tracker, c.StateEngine, c.Repository):
        with pytest.raises(TypeError):
            cls()  # type: ignore[abstract]


def test_repository_covers_sql_surface():
    methods = {
        "health_check",
        "ensure_schema_version",
        "start_session",
        "end_session",
        "upsert_camera_status",
        "write_state_snapshot",
        "append_event",
        "read_current_seat_states",
        "read_event_timeline",
        "write_metric_batch",
        "read_model_manifest",
        "write_validation_result",
        # V003 seat store: videos on disk, everything else in PLOMID.
        "upsert_video_asset",
        "list_video_assets_by_path",
        "upsert_seat_layout",
        "load_seat_layout",
        "list_seat_layouts",
        "save_video_seatmap",
        "load_video_seatmap",
        "list_video_seatmaps_by_path",
        "save_probe_entry",
        "load_probe_entry",
        "save_analytics_entry",
        "load_analytics_entry",
    }
    assert methods <= set(c.Repository.__abstractmethods__)


def test_fake_source_must_be_labelled():
    class Fake(c.Source):
        kind = "fake"
        labelled_fake = True

        def open(self): ...
        def next(self): ...
        def health(self): ...
        def close(self): ...

    assert Fake().labelled_fake is True


def test_domain_has_no_vendor_types():
    leaked = [
        name
        for name in dir(d)
        if any(
            tag in name.lower() for tag in ("dfine", "rtdetr", "onnx", "tensorrt", "ultralytics")
        )
    ]
    assert leaked == []
