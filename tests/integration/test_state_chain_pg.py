"""Phase 6 integration — full chain into real PLOMID, read back from SQL.

detector → tracker → association → state engine → repository, on real
clips. Then current state + timeline are read back and reconciled.
Needs detector weights; skipped without them.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from test_detector_ort import WEIGHTS, _runtime

from odc.capture import FileReplaySource
from odc.config import AppConfig, CameraConfig
from odc.domain import PresenceState
from odc.persistence import PlomidRepository
from odc.seats import load_layout
from odc.state import SeatStateEngine, StateRules, reconcile_from_events
from odc.tracking import ByteTrackStyleTracker

needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="no detector weights")


def _chain(clip: str, layout_path: str, max_frames: int, dsn: str):
    cfg = AppConfig(camera=CameraConfig(path=clip))
    rules = StateRules(
        min_positive_frames=2,
        occupy_dwell_s=0.0,
        confirm_extra_s=1000.0,
        empty_dwell_frames=1000,
        stale_after_s=3600.0,
        cooldown_s=0.0,
    )
    layout = load_layout(layout_path)
    repo = PlomidRepository(dsn)
    repo.ensure_schema_version(2)
    runtime = _runtime(threshold=0.3)
    tracker = ByteTrackStyleTracker()
    engine = SeatStateEngine("ses-chain", layout, rules, "cfg-v1", "model-v1")
    source = FileReplaySource(cfg.camera)
    source.open()
    session_started = False
    try:
        for _ in range(max_frames):
            frame = source.next()
            if frame is None or frame.pixels is None:
                break
            if not session_started:
                from odc.domain import RunMode, SessionMeta

                repo.start_session(
                    SessionMeta(
                        session_id="ses-chain",
                        camera_id="cabin0",
                        source_kind="replay",
                        mode=RunMode.REPLAY,
                        started_at=frame.meta.captured_at,
                        build_version="0.1.0",
                        config_version="cfg-v1",
                    )
                )
                session_started = True
            persons = [d for d in runtime.infer(frame).detections if d.class_label == "person"]
            states = tracker.update(frame, persons)
            out = engine.update(frame, states, layout.seats)
            if out.seats:
                repo.write_state_snapshot(out.seats)
            for event in out.events:
                repo.append_event(event)
        return repo, engine
    finally:
        source.close()
        runtime.close()


@needs_weights
def test_chain_f1_states_and_events_in_sql(plomid_dsn):
    repo, engine = _chain(
        "footage/F1_family_ride.mp4", "configs/seats/demo-f1-v0.yaml", 30, plomid_dsn
    )
    try:
        seats = repo.read_current_seat_states("ses-chain")
        assert {s.seat_id for s in seats} == {
            "near-adult-left",
            "mid-adult-center",
            "child-seat-right",
        }
        occupied = {
            s.seat_id
            for s in seats
            if s.presence in (PresenceState.OCCUPIED, PresenceState.LIKELY_OCCUPIED)
        }
        assert len(occupied) >= 2, [(s.seat_id, s.presence.value) for s in seats]
        timeline = repo.read_event_timeline("ses-chain")
        assert any(e.new_state in ("OCCUPIED", "LIKELY_OCCUPIED") for e in timeline)
        assert all(e.session_id == "ses-chain" and e.event_id for e in timeline)
        current = reconcile_from_events(timeline, "ses-chain")
        live = {s.seat_id: s.presence.value for s in seats}
        for seat_id, (state, _) in current.items():
            assert live[seat_id] == state
    finally:
        repo.close()


@needs_weights
def test_chain_e1_brief_absence_stays_unknown(plomid_dsn):
    repo, engine = _chain(
        "footage/E1_empty_cabin.mp4", "configs/seats/demo-f1-v0.yaml", 10, plomid_dsn
    )
    try:
        seats = repo.read_current_seat_states("ses-chain")
        assert all(s.presence == PresenceState.UNKNOWN for s in seats), [
            (s.seat_id, s.presence.value) for s in seats
        ]
    finally:
        repo.close()


@needs_weights
def test_stale_events_persisted(plomid_dsn):
    repo, engine = _chain(
        "footage/F1_family_ride.mp4", "configs/seats/demo-f1-v0.yaml", 5, plomid_dsn
    )
    try:
        last_ts = engine._last_frame_ts
        assert last_ts is not None
        events = engine.check_stale(last_ts + timedelta(seconds=4000))
        assert events
        for event in events:
            repo.append_event(event)
        timeline = repo.read_event_timeline("ses-chain")
        assert any(e.new_state == "STALE" for e in timeline)
    finally:
        repo.close()
