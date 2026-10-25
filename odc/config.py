"""ODC configuration (Phase 1) — validated, no secrets on disk."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .domain import RunMode

if TYPE_CHECKING:
    from .state import StateRules


class QualityThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dark_thresh: float = 30.0
    bright_thresh: float = 225.0
    blur_var_thresh: float = 50.0
    frozen_diff_thresh: float = 2.0


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str = "cabin0"
    source: str = "replay"
    path: str = "footage/F1_family_ride.mp4"
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    target_fps: float = Field(default=5.0, gt=0, le=120)
    realtime: bool = False
    speed: float = Field(default=1.0, gt=0)
    replay_anchor: str = "file-mtime"
    replay_loop: bool = False
    reconnect_attempts: int = Field(default=3, ge=1, le=10)
    reconnect_base_s: float = Field(default=0.5, gt=0)
    stale_after_s: float = Field(default=5.0, gt=0)
    dead_after_s: float = Field(default=30.0, gt=0)
    health_min_interval_s: float = Field(default=5.0, gt=0)
    quality: QualityThresholds = Field(default_factory=QualityThresholds)


class PlomidConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dsn: SecretStr | None = None
    connect_timeout_s: float = Field(default=5.0, gt=0)
    statement_timeout_ms: int = Field(default=10_000, gt=0)


DEFAULT_LOCAL_PLOMID_DSN = "postgresql://plomid:plomid@127.0.0.1:5432/plomid"


class ProcessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_every_n: int = Field(default=6, ge=1)
    queue_max: int = Field(default=8, ge=1)


class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = "rtdetrv2-s-coco"
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_gender_appearance: bool = False
    # Mask source: fast classical GrabCut refinement (live path) or precise
    # SAM-box-prompted masks (background worker, labelled with frame seq).
    mask_mode: Literal["fast", "precise"] = "fast"
    # Optional per-class score floors. The selected general threshold remains
    # a minimum for all classes; a class override may raise its minimum.
    per_class_thresholds: dict[str, float] = Field(default_factory=dict)
    nms_iou: float = Field(default=0.5, gt=0.0, lt=1.0)
    min_decode_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    enable_segmentation: bool = True
    mask_max: int = Field(default=20, ge=1, le=50)
    history_max: int = Field(default=20, ge=1, le=100)
    child_specialist_enabled: bool = False
    # Legacy Grounding DINO candidate path. It is disabled by default for age
    # classification; production age/gender comes from MiVOLO V2 below.
    child_specialist_every_n: int = Field(default=30, ge=1, le=120)
    child_specialist_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    child_specialist_checkpoint: str = "models/grounding-dino-swint-ogc.pth"
    child_specialist_config: str = (
        "models/grounding-dino-code/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )


class PoseConfig(BaseModel):
    """Body-pose tracking (MediaPipe, local weights only).

    Enabled by default for the cabin showcase. When enabled, runs
    on the largest few person boxes as supporting evidence (seat
    association, frontality for belt verdicts, skeleton overlay). Never
    drives occupancy states by itself.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_targets: int = Field(default=3, ge=1, le=6)
    every_n: int = Field(default=2, ge=1, le=10)
    model_path: str = "models/mediapipe-pose-lite/pose_landmarker_lite.task"


class AlertsConfig(BaseModel):
    """Rule-based cabin alerts (candidate-grade).

    Webhook is off unless a URL is configured (env ODC_ALERTS__WEBHOOK_URL);
    delivery is best-effort in a background thread and never blocks the
    loop. Unattended timing is camera-only and always worded to require
    door/ignition verification.
    """

    model_config = ConfigDict(extra="forbid")

    webhook_url: str = ""
    min_severity: str = "WARNING"
    unattended_after_s: float = Field(default=600.0, gt=0)
    cooldown_s: float = Field(default=60.0, gt=0)


class DmsConfig(BaseModel):
    """Driver-monitoring previews (face landmarks, local weights only).

    Disabled by default: zero additional DMS overhead. When enabled, face
    state (blink/jaw/head-pose) feeds per-seat drowsiness, distraction and
    phone-use candidates plus a face overlay. Preview-grade, never safety.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    every_n: int = Field(default=3, ge=1, le=10)
    num_faces: int = Field(default=2, ge=1, le=4)
    model_path: str = "models/mediapipe-face/face_landmarker.task"


class AgeConfig(BaseModel):
    """Local MiVOLO V2 age and apparent-gender estimates for tracked people."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model_path: str = "models/mivolo-v2"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    infant_max_years: float = Field(default=2.0, ge=0.0, le=18.0)
    child_max_years: float = Field(default=17.0, gt=0.0, le=30.0)
    uncertainty_margin_years: float = Field(default=6.0, ge=0.0, le=15.0)
    gender_min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    history_max: int = Field(default=3, ge=1, le=15)
    track_expiry_frames: int = Field(default=36, ge=1, le=1000)
    every_n: int = Field(default=1, ge=1, le=30)


class AssistantConfig(BaseModel):
    """Explicit local Ollama assistant; never part of frame processing."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    url: str = "http://127.0.0.1:11434"
    model: str = "mistral"
    timeout_s: float = Field(default=30.0, gt=0, le=120)


class StateConfig(BaseModel):
    """Versioned temporal rules — mirrors state.StateRules (bump = revalidate)."""

    model_config = ConfigDict(extra="forbid")

    version: str = "state-v1"
    min_score: float = 0.3
    min_positive_frames: int = 3
    occupy_dwell_s: float = 1.0
    confirm_extra_s: float = 2.0
    release_misses_likely: int = 5
    release_misses_unknown: int = 15
    empty_dwell_frames: int = 25
    stale_after_s: float = 5.0
    cooldown_s: float = 10.0
    ambiguity_margin: float = 0.15
    size_window: int = 5
    size_small_max: float = 0.5
    size_large_min: float = 1.1
    size_min_votes: int = 3
    specialist_child_min_score: float = Field(default=0.30, ge=0.0, le=1.0)
    specialist_child_direct_score: float = Field(default=0.35, ge=0.0, le=1.0)
    specialist_restraint_max_box_area: float = Field(default=0.65, ge=0.0, le=1.0)
    specialist_child_confirmation_score: float = Field(default=0.351, ge=0.0, le=1.0)

    def to_rules(self) -> StateRules:
        from .state import StateRules

        return StateRules(**self.model_dump())


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ODC_",
        env_nested_delimiter="__",
        extra="ignore",  # real shells carry stray ODC_* vars; YAML keys stay strict below
    )

    app: str = "plomid-odc"
    env: str = "dev"
    mode: RunMode = RunMode.REPLAY
    build_version: str = "0.1.0"
    camera: CameraConfig = Field(default_factory=CameraConfig)
    plomid: PlomidConfig = Field(default_factory=PlomidConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    dms: DmsConfig = Field(default_factory=DmsConfig)
    age: AgeConfig = Field(default_factory=AgeConfig)
    assistant: AssistantConfig = Field(default_factory=AssistantConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    seat_layout_version: str = "unset"
    model_set_version: str = "unset"


def _load_dotenv(path: str | Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE lines, # comments, no expansion).

    stdlib-only on purpose: secrets stay out of YAML, and ODC_DSN keeps one
    meaning everywhere (shell env wins over file — never the reverse).
    """
    import os

    candidate = Path(path) if path else Path(".env")
    if not candidate.exists():
        return
    for raw in candidate.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML file over defaults; env/.env may inject the DSN and knobs."""
    _load_dotenv()
    data: dict[str, Any] = {}
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
    cfg = AppConfig(**data)
    dsn = __import__("os").environ.get("ODC_DSN")
    if dsn:
        cfg.plomid.dsn = SecretStr(dsn)
    return cfg
