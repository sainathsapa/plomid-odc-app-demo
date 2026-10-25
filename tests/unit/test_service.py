"""Service loop unit tests — degraded paths, evidence-clock staleness."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from odc import contracts
from odc.capture import FakeSource
from odc.config import AppConfig
from odc.domain import RunMode
from odc.seats import SeatGeometry, SeatLayout
from odc.service import Service, ServiceHub

LAYOUT = SeatLayout(
    camera_profile="t",
    version="t",
    seats=[
        SeatGeometry(
            seat_id="A", polygon=[(0, 0), (1, 0), (1, 1), (0, 1)], camera_profile="t", version="t"
        )
    ],
)


class _StubRepo(contracts.Repository):
    def __init__(self) -> None:
        self.sessions: list = []
        self.seats: list = []
        self.events: list = []
        self.frame_observations: list[dict[str, Any]] = []

    def health_check(self) -> bool:
        return True

    def ensure_schema_version(self, minimum: int) -> int:
        return minimum

    def start_session(self, session: Any) -> None:
        self.sessions.append(session)

    def end_session(self, session_id: str) -> None:
        pass

    def upsert_camera_status(self, health: Any) -> None:
        pass

    def write_state_snapshot(self, seats: Any) -> None:
        self.seats.extend(seats)

    def write_frame_observation(
        self,
        session_id: str,
        frame_seq: int,
        captured_at: Any,
        seat_rows: list[dict[str, Any]],
        pose: dict[str, Any],
        specialist_decisions: list[dict[str, Any]],
    ) -> None:
        self.frame_observations.append(
            {
                "session_id": session_id,
                "frame_seq": frame_seq,
                "seat_rows": seat_rows,
                "pose": pose,
                "specialist_decisions": specialist_decisions,
            }
        )

    def append_event(self, event: Any) -> None:
        self.events.append(event)

    def read_current_seat_states(self, session_id: str) -> Any:
        return []

    def read_latest_frame_observation(self, session_id: str) -> Any:
        return None

    def append_assistant_interaction(self, row: Any) -> None:
        pass

    def read_assistant_interactions(self, session_id: str, limit: int = 100) -> Any:
        return []

    def read_event_timeline(self, *a: Any, **k: Any) -> Any:
        return []

    def write_metric_batch(self, rows: Any) -> None:
        pass

    def read_model_manifest(self, model_id: str) -> Any:
        raise AssertionError

    def write_validation_result(self, rows: Any) -> None:
        pass

    # Seat store (V003): this stub models a database with no stored rows —
    # writes vanish, reads miss, and callers degrade gracefully.
    def upsert_video_asset(self, row: Any) -> None:
        pass

    def list_video_assets_by_path(self, video_path: str) -> Any:
        return []

    def upsert_seat_layout(self, name: str, doc: Any) -> None:
        pass

    def load_seat_layout(self, name: str) -> Any:
        return None

    def list_seat_layouts(self) -> Any:
        return []

    def save_video_seatmap(self, row: Any) -> None:
        pass

    def load_video_seatmap(self, video_key: str) -> Any:
        return None

    def list_video_seatmaps_by_path(self, video_path: str) -> Any:
        return []

    def save_probe_entry(self, row: Any) -> None:
        pass

    def load_probe_entry(self, video_key: str) -> Any:
        return None

    def save_analytics_entry(self, row: Any) -> None:
        pass

    def load_analytics_entry(self, cache_key: str) -> Any:
        return None


class _NoModelService(Service):
    def _build_source(self) -> contracts.Source:
        return FakeSource(n_frames=20, mode=RunMode.DEMO)

    def _build_runtime(self) -> None:
        return None


def test_service_runs_degraded_without_model():
    repo = _StubRepo()
    service = _NoModelService(AppConfig(), repo, LAYOUT)
    service.start()
    time.sleep(1.0)
    service.stop()
    assert repo.sessions, "session must start even degraded"
    assert repo.seats, "snapshots must persist in degraded mode"
    assert repo.frame_observations, "structured frame outputs must persist in PLOMID"
    assert repo.frame_observations[0]["seat_rows"]
    assert service.hub.snapshot.model_healthy is False
    assert service.hub.snapshot.db_healthy is True
    assert all(e.new_state != "STALE" for e in service.hub.events)


def test_service_shutdown_saves_analytics_snapshot():
    from tests.unit.test_persistence_unit import FakeRepository

    repo = FakeRepository()
    service = _NoModelService(AppConfig(), repo, LAYOUT)
    service.start()
    time.sleep(1.0)
    service.stop()
    assert repo.analytics, "shutdown must persist an analytics snapshot in PLOMID"
    payload = next(iter(repo.analytics.values()))["payload"]
    assert payload["counts"]["total"] == 1
    assert payload["layout_version"] == "t"


def test_service_db_outage_stays_transient():
    class _Dead(_StubRepo):
        def health_check(self) -> bool:
            return False

        def ensure_schema_version(self, minimum: int) -> int:
            raise AssertionError("must not be called when unhealthy")

    service = _NoModelService(AppConfig(), _Dead(), LAYOUT)
    service.start()
    time.sleep(0.8)
    service.stop()
    assert service.hub.snapshot.db_healthy is False
    assert service.hub.snapshot.running is False


def test_hub_publish_bounds_and_versions():
    hub = ServiceHub()
    v1 = hub.publish(
        __import__("odc.service", fromlist=["ServiceSnapshot"]).ServiceSnapshot(), [], []
    )
    v2 = hub.publish(
        __import__("odc.service", fromlist=["ServiceSnapshot"]).ServiceSnapshot(), [], []
    )
    assert (v1, v2) == (1, 2)


def test_hub_does_not_duplicate_cumulative_candidate_events():
    from odc.domain import EventType, Severity, StateEvent
    from odc.service import ServiceSnapshot

    hub = ServiceHub()
    now = datetime.now(UTC)
    event = StateEvent(
        event_id="session-seat_2-child-1",
        session_id="session",
        event_type=EventType.CHILD_CANDIDATE_DETECTED,
        severity=Severity.INFO,
        seat_id="seat_2",
        occurred_at=now,
        recorded_at=now,
        reason_codes=["child-candidate:relative-cabin-size"],
    )

    hub.publish(ServiceSnapshot(), [], [event])
    hub.publish(ServiceSnapshot(), [], [event])

    assert [item.event_id for item in hub.events] == [event.event_id]


def test_hub_reset_scopes_state_to_one_video():
    from odc.service import ServiceSnapshot

    hub = ServiceHub()
    hub.publish(ServiceSnapshot(session_id="old"), [], [])
    detection = {"label": "person", "bbox": [0.1, 0.2, 0.4, 0.9], "score": 0.91}
    hub.note_full_frame(
        b"jpeg",
        [detection],
        [],
        {},
        1.0,
        9,
        "ts",
        {"total": 2},
        seat_rows=[{"seat_id": "child-seat", "child": "Unknown", "frame_seq": 9}],
        specialist_detections=[{"label": "child", "score": 0.33, "frame_seq": 6}],
        specialist_decisions=[{"disposition": "accepted_associated_candidate"}],
        specialist_frame_seq=6,
    )
    assert hub.history[-1].detections == [detection]
    assert hub.history[-1].seat_rows[0]["frame_seq"] == 9
    assert hub.history[-1].specialist_detections[0]["frame_seq"] == 6
    assert hub.history[-1].specialist_decisions[0]["disposition"] == (
        "accepted_associated_candidate"
    )
    before = hub.version
    hub.reset()
    assert hub.version == before + 1
    assert hub.seats == [] and hub.seat_rows == [] and hub.counts == {}
    assert hub.detections == [] and hub.masks == [] and hub.frame_jpeg is None
    assert hub.frame_seq == -1 and list(hub.history) == [] and hub.events == []
    assert hub.snapshot.session_id == ""
