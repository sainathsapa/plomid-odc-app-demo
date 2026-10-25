"""ODC component contracts (Phase 1) — interfaces only, no logic.

Implementations land in their phases (capture: Ph3, runtime/perception:
Ph4, tracking/seats: Ph5, state: Ph6, repository: Ph2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .domain import (
    BBox,
    CameraHealth,
    Detection,
    FrameMeta,
    Keypoint,
    ModelManifest,
    OccupantState,
    SeatGeometry,
    SeatState,
    SessionMeta,
    StateEvent,
)


@dataclass(frozen=True)
class InferenceResult:
    detections: list[Detection]
    keypoints: dict[int, list[Keypoint]] = field(default_factory=dict)
    model_id: str = ""
    provider: str = ""
    duration_ms: float = 0.0


@dataclass(frozen=True)
class Frame:
    meta: FrameMeta
    pixels: Any = None  # numpy array from Phase 3 on; None in contract tests


@dataclass(frozen=True)
class EngineOutput:
    seats: list[SeatState]
    occupants: list[OccupantState]
    events: list[StateEvent]


class DetectorRuntime(ABC):
    @abstractmethod
    def load(self, manifest: ModelManifest) -> None: ...

    @abstractmethod
    def infer(self, frame: Frame) -> InferenceResult: ...

    @abstractmethod
    def close(self) -> None: ...


class Source(ABC):
    kind: str = "unknown"
    labelled_fake: bool = False

    @abstractmethod
    def open(self) -> FrameMeta: ...

    @abstractmethod
    def next(self) -> Frame | None: ...

    @abstractmethod
    def health(self) -> CameraHealth: ...

    @abstractmethod
    def close(self) -> None: ...


class Tracker(ABC):
    @abstractmethod
    def update(self, frame: Frame, detections: list[Detection]) -> list[OccupantState]: ...

    @abstractmethod
    def reset(self) -> None: ...


class PoseEstimator(ABC):
    """Optional capability: returns frame-normalized keypoints per box index.

    Absence ([], low visibility) is *missing evidence*, never absent limbs.
    Implementations must never crash the pipeline (log + degrade).
    """

    @abstractmethod
    def estimate(self, frame: Frame, boxes: list[BBox]) -> dict[int, list[Keypoint]]: ...

    @abstractmethod
    def close(self) -> None: ...


class StateEngine(ABC):
    @abstractmethod
    def update(
        self,
        frame: Frame,
        occupants: list[OccupantState],
        seats: list[SeatGeometry],
        objects: list[Detection] | None = None,
    ) -> EngineOutput: ...


class Repository(ABC):
    """PLOMID SQL boundary. Implementations use parameterized SQL only."""

    @abstractmethod
    def health_check(self) -> bool: ...

    @abstractmethod
    def ensure_schema_version(self, minimum: int) -> int: ...

    @abstractmethod
    def start_session(self, session: SessionMeta) -> None: ...

    @abstractmethod
    def end_session(self, session_id: str) -> None: ...

    @abstractmethod
    def upsert_camera_status(self, health: CameraHealth) -> None: ...

    @abstractmethod
    def write_state_snapshot(self, seats: list[SeatState]) -> None: ...

    @abstractmethod
    def write_frame_observation(
        self,
        session_id: str,
        frame_seq: int,
        captured_at: datetime,
        seat_rows: list[dict[str, Any]],
        pose: dict[str, Any],
        specialist_decisions: list[dict[str, Any]],
    ) -> None: ...

    @abstractmethod
    def append_event(self, event: StateEvent) -> None: ...

    @abstractmethod
    def read_current_seat_states(self, session_id: str) -> list[SeatState]: ...

    @abstractmethod
    def read_latest_frame_observation(self, session_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def append_assistant_interaction(self, row: dict[str, Any]) -> None: ...

    @abstractmethod
    def read_assistant_interactions(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def read_event_timeline(
        self, session_id: str, seat_id: str | None = None, limit: int = 200
    ) -> list[StateEvent]: ...

    @abstractmethod
    def write_metric_batch(self, rows: list[dict[str, Any]]) -> None: ...

    @abstractmethod
    def read_model_manifest(self, model_id: str) -> ModelManifest: ...

    @abstractmethod
    def write_validation_result(self, rows: list[dict[str, Any]]) -> None: ...

    # -- seat store (V003): seat maps, calibration, probe + analytics reuse.
    # Videos stay on disk; everything else lives in PLOMID. All writes are
    # idempotent upserts; reads return None when nothing is stored.

    @abstractmethod
    def upsert_video_asset(self, row: dict[str, Any]) -> None: ...

    @abstractmethod
    def list_video_assets_by_path(self, video_path: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def upsert_seat_layout(self, name: str, doc: dict[str, Any]) -> None: ...

    @abstractmethod
    def load_seat_layout(self, name: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def list_seat_layouts(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def save_video_seatmap(self, row: dict[str, Any]) -> None: ...

    @abstractmethod
    def load_video_seatmap(self, video_key: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def list_video_seatmaps_by_path(self, video_path: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def save_probe_entry(self, row: dict[str, Any]) -> None: ...

    @abstractmethod
    def load_probe_entry(self, video_key: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def save_analytics_entry(self, row: dict[str, Any]) -> None: ...

    @abstractmethod
    def load_analytics_entry(self, cache_key: str) -> dict[str, Any] | None: ...
