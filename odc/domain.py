"""PLOMID ODC — versioned domain contracts (Phase 1).

Model-independent: no detector/vendor type may leak into state types.
Every model carries the domain version it was validated against.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

DOMAIN_VERSION = "1.0"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RunMode(str, Enum):
    LIVE = "LIVE"
    REPLAY = "REPLAY"
    TEST = "TEST"
    DEMO = "DEMO"
    SIMULATED = "SIMULATED"


class PresenceState(str, Enum):
    OCCUPIED = "OCCUPIED"
    LIKELY_OCCUPIED = "LIKELY_OCCUPIED"
    EMPTY_CANDIDATE = "EMPTY_CANDIDATE"
    UNKNOWN = "UNKNOWN"
    STALE = "STALE"


class VisibilityState(str, Enum):
    VISIBLE = "VISIBLE"
    PARTIALLY_OCCLUDED = "PARTIALLY_OCCLUDED"
    OCCLUDED = "OCCLUDED"
    OUT_OF_VIEW = "OUT_OF_VIEW"
    CAMERA_UNHEALTHY = "CAMERA_UNHEALTHY"
    NOT_EVALUATED = "NOT_EVALUATED"


class SizeAppearance(str, Enum):
    APPARENT_SMALL_OCCUPANT = "APPARENT_SMALL_OCCUPANT"
    APPARENT_LARGE_OCCUPANT = "APPARENT_LARGE_OCCUPANT"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"


class GenderAppearance(str, Enum):
    """EXPERIMENTAL, owner-accepted risk (D30). Never safety, never identity,
    never persisted, never run on small occupants."""

    APPARENT_MAN = "APPARENT_MAN"
    APPARENT_WOMAN = "APPARENT_WOMAN"
    UNKNOWN = "UNKNOWN"
    DISABLED = "DISABLED"


class SeatbeltState(str, Enum):
    FASTENED_CANDIDATE = "FASTENED_CANDIDATE"
    UNFASTENED_CANDIDATE = "UNFASTENED_CANDIDATE"
    MISROUTED_CANDIDATE = "MISROUTED_CANDIDATE"
    OCCLUDED = "OCCLUDED"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"


class RestraintState(str, Enum):
    CHILD_SEAT_CANDIDATE = "CHILD_SEAT_CANDIDATE"
    BOOSTER_CANDIDATE = "BOOSTER_CANDIDATE"
    OTHER_OBJECT_CANDIDATE = "OTHER_OBJECT_CANDIDATE"
    NOT_DETECTED = "NOT_DETECTED"
    OCCLUDED = "OCCLUDED"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"


class PostureState(str, Enum):
    UPRIGHT = "UPRIGHT"
    LEANING_FORWARD = "LEANING_FORWARD"
    LEANING_SIDEWAYS = "LEANING_SIDEWAYS"
    RECLINED = "RECLINED"
    BODY_PART_OCCLUDED = "BODY_PART_OCCLUDED"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


class EventType(str, Enum):
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_ENDED = "SESSION_ENDED"
    SEAT_PRESENCE_CHANGED = "SEAT_PRESENCE_CHANGED"
    CHILD_CANDIDATE_DETECTED = "CHILD_CANDIDATE_DETECTED"
    CHILD_CANDIDATE_CLEARED = "CHILD_CANDIDATE_CLEARED"
    OCCUPANT_TRACK_STARTED = "OCCUPANT_TRACK_STARTED"
    OCCUPANT_TRACK_ENDED = "OCCUPANT_TRACK_ENDED"
    ATTRIBUTE_BECAME_UNKNOWN = "ATTRIBUTE_BECAME_UNKNOWN"
    ALERT_RAISED = "ALERT_RAISED"
    ALERT_CLEARED = "ALERT_CLEARED"
    CAMERA_DEGRADED = "CAMERA_DEGRADED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    DATABASE_DEGRADED = "DATABASE_DEGRADED"
    CONFIG_CHANGED = "CONFIG_CHANGED"
    UNATTENDED_OCCUPANT_CANDIDATE = "UNATTENDED_OCCUPANT_CANDIDATE"
    FUSION_SIGNAL_STALE = "FUSION_SIGNAL_STALE"


class SqlErrorKind(str, Enum):
    CONNECTION = "CONNECTION"
    TIMEOUT = "TIMEOUT"
    CONFLICT = "CONFLICT"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    UNSUPPORTED = "UNSUPPORTED"
    QUERY = "QUERY"


T = TypeVar("T")


class AttributeState(Strict, Generic[T]):
    """One attribute: value + own confidence, evidence time, reason, versions."""

    value: T
    confidence: float = Field(ge=0.0, le=1.0)
    observed_at: datetime
    reason: str = ""
    model_set_version: str = "unset"
    config_version: str = "unset"


class BBox(Strict):
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)


class FrameMeta(Strict):
    domain_version: str = DOMAIN_VERSION
    source_id: str
    seq: int = Field(ge=0)
    captured_at: datetime
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mode: RunMode
    dropped_before: int = Field(default=0, ge=0)


class Detection(Strict):
    class_label: str
    score: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    evidence_kind: str = "detector"
    model_id: str
    frame_seq: int


class Keypoint(Strict):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    visible: bool = True


class SeatGeometry(Strict):
    seat_id: str
    polygon: list[tuple[float, float]] = Field(min_length=3)
    camera_profile: str
    version: str
    # Optional cabin ordering hints (derived when absent). Defaults keep
    # every existing YAML valid; the editor and auto-layout fill these so
    # front-to-rear ordering survives reloads instead of being re-guessed.
    row: int | None = None
    col: int | None = None
    label: str | None = None
    display_number: int | None = Field(default=None, ge=1)
    physical_role: str | None = None


class SeatState(Strict):
    domain_version: str = DOMAIN_VERSION
    session_id: str
    seat_id: str
    presence: PresenceState
    confidence: float = Field(ge=0.0, le=1.0)
    visibility: VisibilityState
    reason_codes: list[str] = Field(default_factory=list)
    object_label: str | None = None
    updated_at: datetime
    last_positive_at: datetime | None = None
    model_set_version: str = "unset"
    config_version: str = "unset"


class OccupantState(Strict):
    domain_version: str = DOMAIN_VERSION
    session_id: str
    track_id: str
    seat_id: str | None = None
    last_bbox: BBox | None = None
    last_score: float | None = None
    presence: AttributeState[PresenceState] | None = None
    apparent_size: AttributeState[SizeAppearance] | None = None
    posture: AttributeState[PostureState] | None = None
    seatbelt: AttributeState[SeatbeltState] | None = None
    restraint: AttributeState[RestraintState] | None = None
    updated_at: datetime


class StateEvent(Strict):
    domain_version: str = DOMAIN_VERSION
    event_id: str
    session_id: str
    event_type: EventType
    severity: Severity = Severity.INFO
    seat_id: str | None = None
    track_id: str | None = None
    occurred_at: datetime
    recorded_at: datetime
    prior_state: str | None = None
    new_state: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    evidence_summary: str = Field(default="", max_length=2000)
    config_version: str = "unset"
    model_set_version: str = "unset"


class CameraHealth(Strict):
    camera_id: str
    session_id: str | None = None
    status: HealthStatus
    fps_observed: float | None = None
    failure_code: str | None = None
    checked_at: datetime


class ModelManifest(Strict):
    model_id: str
    family: str
    checkpoint_sha256: str
    source_url: str
    code_revision: str
    weights_license: str
    dataset_terms: str
    approved_for_commercial_use: bool = False
    input_profile: str = ""
    runtime_profile: str = ""


class SessionMeta(Strict):
    domain_version: str = DOMAIN_VERSION
    session_id: str
    camera_id: str
    source_kind: str
    mode: RunMode
    started_at: datetime
    ended_at: datetime | None = None
    build_version: str
    config_version: str = "unset"
    status: str = "OPEN"


class SqlError(Strict):
    kind: SqlErrorKind
    message: str
    retryable: bool = False


class RepositoryError(Exception):
    """Raised by the persistence boundary; carries a serializable SqlError."""

    def __init__(self, error: SqlError) -> None:
        super().__init__(error.message)
        self.error = error
