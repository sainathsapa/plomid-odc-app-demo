"""ODC service loop — capture → detect → track → state → PLOMID.

Real-time pipeline (task upgrade):
- Continuous capture with per-frame preview JPEGs (smooth video) and
  sampled inference (bounded latency). Sampling uses a dedicated
  frames_seen counter — the old double-increment of `processed` is fixed.
- Decode at min_gate, then per-class floors + vehicle-hallucination filter
  + class-aware NMS (fixes duplicates + missing children / child seats).
- Classical bbox-refined masks + ROI crops per frame (segmentation module).
- Rolling history ring (last N frames) for previous-frame inspection.
- Stable seat identities: hub publishes seat_rows keyed by seat_id with
  child/restraint/belt attributes preserved side by side + live counts.
- Live-camera reconnect with backoff; replay-loop restart; stale checks on
  the evidence clock; visible DEGRADED on model/DB failure.

Runs in a background thread with a stop event. Degrades visibly:
model failure → MODEL_UNAVAILABLE (states UNKNOWN); DB failure →
transient-only latest (logged, bounded), UI shows DEGRADED.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from . import contracts
from .alerts import AlertManager, WebhookSender
from .attributes import annotate_detections, build_seat_map, live_counts
from .belt import SeatBeltTracker, detect_belt
from .capture import CameraHealthWriter, FileReplaySource, LiveCameraSource
from .config import AppConfig
from .domain import (
    EventType,
    HealthStatus,
    RunMode,
    SessionMeta,
    Severity,
    StateEvent,
)
from .logging import get_logger
from .seats import SeatLayout, load_layout
from .segmentation import (
    CLASS_COLORS,
    DEFAULT_CLASS_COLOR,
    PER_CLASS_THRESHOLDS,
    crop_detection,
    detection_masks,
    postprocess_detections,
)
from .state import SeatStateEngine, StateRules

log = get_logger("service")


@dataclass
class ServiceSnapshot:
    session_id: str = ""
    running: bool = False
    mode: RunMode = RunMode.REPLAY
    frames_captured: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    last_latency_ms: float = 0.0
    db_healthy: bool = False
    model_healthy: bool = False
    model_id: str = "unset"
    provider: str = "unset"
    camera_status: HealthStatus = HealthStatus.UNKNOWN
    calibrated: bool = True
    updated_at: datetime | None = None
    # Live-sync helpers (new, defaulted so old callers keep working).
    frame_seq: int = -1
    suppressed_hallucinations: int = 0
    duplicates_removed: int = 0
    threshold: float = 0.0
    layout_source: str = ""
    pose_on: bool = False
    pose_healthy: bool = False
    dms_on: bool = False
    dms_healthy: bool = False
    driver_seat_id: str = ""
    mask_mode: str = "fast"
    specialist_enabled: bool = False
    specialist_healthy: bool = False
    specialist_frame_seq: int = -1
    specialist_error: str | None = None


@dataclass
class FrameHistoryEntry:
    seq: int
    jpeg: bytes
    captured_at: str
    n_detections: int
    counts: dict[str, int]
    detections: list[dict[str, Any]] = field(default_factory=list)
    specialist_detections: list[dict[str, Any]] = field(default_factory=list)
    specialist_decisions: list[dict[str, Any]] = field(default_factory=list)
    seat_rows: list[dict[str, Any]] = field(default_factory=list)
    specialist_frame_seq: int = -1
    provider: str = "unset"
    masks: list[dict[str, Any]] = field(default_factory=list)
    belt_overlays: list[dict[str, Any]] = field(default_factory=list)
    pose: dict[str, Any] = field(default_factory=dict)
    face: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceHub:
    """Thread-safe latest-output store for SSE/screens. Transient only."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    snapshot: ServiceSnapshot = field(default_factory=ServiceSnapshot)
    seats: list[Any] = field(default_factory=list)
    occupants: list[Any] = field(default_factory=list)
    events: list[StateEvent] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)
    specialist_detections: list[dict[str, Any]] = field(default_factory=list)
    specialist_decisions: list[dict[str, Any]] = field(default_factory=list)
    masks: list[dict[str, Any]] = field(default_factory=list)
    crops: dict[str, bytes] = field(default_factory=dict)
    belt_overlays: list[dict[str, Any]] = field(default_factory=list)
    pose: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    alerts_history: list[dict[str, Any]] = field(default_factory=list)
    seat_rows: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    precise_seq: int = -1
    precise_masks: list[dict[str, Any]] = field(default_factory=list)
    face: dict[str, Any] = field(default_factory=dict)
    frame_jpeg: bytes | None = None
    frame_seq: int = -1
    frame_ts: str = ""
    version: int = 0
    history: deque[FrameHistoryEntry] = field(default_factory=lambda: deque(maxlen=20))

    def publish(
        self,
        snapshot: ServiceSnapshot,
        seats: list[Any],
        events: list[StateEvent],
        occupants: list[Any] | None = None,
        seat_rows: list[dict[str, Any]] | None = None,
        counts: dict[str, int] | None = None,
    ) -> int:
        with self.lock:
            self.snapshot = snapshot
            self.seats = list(seats)
            if occupants is not None:
                self.occupants = list(occupants)
            if seat_rows is not None:
                self.seat_rows = list(seat_rows)
            if counts is not None:
                self.counts = dict(counts)
            seen_event_ids: set[str] = set()
            unique_events: list[StateEvent] = []
            for event in [*events, *self.events]:
                if event.event_id in seen_event_ids:
                    continue
                seen_event_ids.add(event.event_id)
                unique_events.append(event)
                if len(unique_events) == 200:
                    break
            self.events = unique_events
            self.version += 1
            return self.version

    def note_frame(
        self, jpeg: bytes | None, detections: list[dict[str, Any]], latency_ms: float
    ) -> None:
        with self.lock:
            if jpeg is not None:
                self.frame_jpeg = jpeg
            self.detections = detections[:50]
            self.latencies = (self.latencies + [latency_ms])[-120:]

    def note_specialist_detections(self, detections: list[dict[str, Any]]) -> None:
        """Retain raw specialist evidence for auditable frame diagnostics."""
        with self.lock:
            self.specialist_detections = detections[:50]

    def note_specialist_decisions(self, decisions: list[dict[str, Any]]) -> None:
        """Publish the current frame's specialist association decisions."""
        with self.lock:
            self.specialist_decisions = [dict(item) for item in decisions[:50]]

    def note_full_frame(
        self,
        jpeg: bytes | None,
        detections: list[dict[str, Any]],
        masks: list[dict[str, Any]],
        crops: dict[str, bytes],
        latency_ms: float,
        seq: int,
        captured_at: str,
        counts: dict[str, int] | None = None,
        belt_overlays: list[dict[str, Any]] | None = None,
        pose: dict[str, Any] | None = None,
        face: dict[str, Any] | None = None,
        seat_rows: list[dict[str, Any]] | None = None,
        specialist_detections: list[dict[str, Any]] | None = None,
        specialist_decisions: list[dict[str, Any]] | None = None,
        specialist_frame_seq: int = -1,
        provider: str = "unset",
    ) -> None:
        with self.lock:
            if jpeg is not None:
                self.frame_jpeg = jpeg
                self.frame_seq = seq
                self.frame_ts = captured_at
                try:
                    self.history.append(
                        FrameHistoryEntry(
                            seq=seq,
                            jpeg=jpeg,
                            captured_at=captured_at,
                            n_detections=len(detections),
                            counts=dict(counts or {}),
                            detections=[dict(item) for item in detections[:50]],
                            specialist_detections=[
                                dict(item) for item in (specialist_detections or [])[:50]
                            ],
                            specialist_decisions=[
                                dict(item) for item in (specialist_decisions or [])[:50]
                            ],
                            seat_rows=[dict(item) for item in (seat_rows or [])],
                            specialist_frame_seq=specialist_frame_seq,
                            provider=provider,
                            masks=[dict(item) for item in masks[:50]],
                            belt_overlays=[dict(item) for item in (belt_overlays or [])[:20]],
                            pose=dict(pose or {}),
                            face=dict(face or {}),
                        )
                    )
                except Exception:
                    pass
            self.detections = detections[:50]
            self.masks = masks[:50]
            self.crops = dict(list(crops.items())[:12])
            if belt_overlays is not None:
                self.belt_overlays = belt_overlays[:20]
            if pose is not None:
                self.pose = dict(pose)
            if face is not None:
                self.face = dict(face)
            self.latencies = (self.latencies + [latency_ms])[-120:]
            if counts is not None:
                self.counts = dict(counts)
            self.version += 1

    def preview_only(self, jpeg: bytes | None, seq: int, captured_at: str) -> None:
        """Lightweight per-captured-frame video update (no inference)."""
        with self.lock:
            if jpeg is not None:
                self.frame_jpeg = jpeg
                self.frame_seq = seq
                self.frame_ts = captured_at
            self.version += 1

    def note_precise(self, seq: int, masks: list[dict[str, Any]]) -> None:
        """Store background SAM masks if newer than stored (slow jobs must
        never roll the overlay backward)."""
        with self.lock:
            if seq >= self.precise_seq:
                self.precise_seq = seq
                self.precise_masks = masks[:20]
                self.version += 1

    def note_alerts(self, active: list[dict[str, Any]], raised: list[dict[str, Any]]) -> None:
        with self.lock:
            self.alerts = active[:20]
            self.alerts_history = (raised + self.alerts_history)[:50]
            if active or raised:
                self.version += 1

    def reset(self) -> None:
        """Clear all transient live state for a NEW video/session.

        Seats, counts, detections, masks, crops, history and events belong
        to exactly one video — a new Start must never inherit the previous
        video's numbers. Stops keep last states (no reset there); starts
        always reset. Bumps the version so the UI refreshes immediately.
        """
        with self.lock:
            self.snapshot = ServiceSnapshot()
            self.seats = []
            self.occupants = []
            self.events = []
            self.detections = []
            self.specialist_detections = []
            self.specialist_decisions = []
            self.masks = []
            self.crops = {}
            self.belt_overlays = []
            self.pose = {}
            self.precise_seq = -1
            self.precise_masks = []
            self.face = {}
            self.alerts = []
            self.alerts_history = []
            self.seat_rows = []
            self.counts = {}
            self.latencies = []
            self.frame_jpeg = None
            self.frame_seq = -1
            self.frame_ts = ""
            maxlen = self.history.maxlen or 20
            self.history = deque(maxlen=maxlen)
            self.version += 1


class Service:
    def __init__(
        self,
        cfg: AppConfig,
        repository: contracts.Repository,
        layout: SeatLayout,
        rules: StateRules | None = None,
        hub: ServiceHub | None = None,
    ) -> None:
        self._cfg = cfg
        self._repo = repository
        self._layout = layout
        self._rules = rules or cfg.state.to_rules()
        self.hub = hub or ServiceHub()
        try:
            self.hub.history = deque(maxlen=max(1, cfg.detection.history_max))
        except Exception:
            pass
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._session_id = f"ses-{int(time.time())}"
        self._sam: Any = None
        self._sam_busy = False
        self._sam_lock = threading.Lock()
        self._sam_last_seq = -(10**9)
        self._mask_mode = cfg.detection.mask_mode
        self._engine: Any = None
        self._gender_probe: Any = None
        self._gender_tried = False
        self._gender_loading = False
        self._gender_error: str | None = None
        self._gender_tick = 0
        self._gender_cache: dict[str, tuple[str, float | None]] = {}
        self._age_estimator: Any = None
        self._age_device_name = "unavailable"
        self._age_error: str | None = None
        self._age_cache: dict[str, Any] = {}

    def adopt_layout(self, layout: SeatLayout) -> dict[str, int] | None:
        """Hot-swap seat geometry on the RUNNING session: the console map,
        overlays and all later frames use the new zones immediately, while
        carried-over seats keep their temporal beliefs (no restart, no
        reprocessing, no dwell reset). Returns the rekey summary, or None
        when no session is running (caller should just switch geometry).
        Thread-safe for the streaming loop (atomic swaps only).
        """
        self._layout = layout
        engine = self._engine
        if engine is None:
            return None
        try:
            summary: dict[str, int] | None = engine.rekey_layout(layout)
            return summary
        except Exception as e:
            log.warning("adopt layout degraded: %s", type(e).__name__)
            return {"carried": 0, "reset": len(getattr(layout, "seats", []) or [])}

    def _build_source(self) -> contracts.Source:
        if self._cfg.camera.source == "live":
            return LiveCameraSource(self._cfg.camera)
        return FileReplaySource(self._cfg.camera)

    def _build_runtime(self) -> contracts.DetectorRuntime | None:
        try:
            from .runtime import OnnxDetectorRuntime, load_manifest

            manifest = load_manifest(self._cfg.detection.model_id)
            # Decode at the minimum gate so small/occluded cabin evidence
            # (chairs ~0.3, small persons lower) survives; per-class floors
            # are applied in post-processing.
            decode_thr = min(
                self._cfg.detection.min_decode_threshold,
                self._cfg.detection.threshold,
                min(PER_CLASS_THRESHOLDS.values()),
            )
            runtime = OnnxDetectorRuntime(threshold=decode_thr)
            runtime.load(manifest)
            return runtime
        except Exception as e:
            log.warning("model unavailable, running degraded: %s: %s", type(e).__name__, e)
            return None

    def start(self) -> str:
        if self._thread is not None:
            return self._session_id
        # New video, new truth: drop the previous video's frames, seats,
        # counts and events before the thread publishes anything.
        self.hub.reset()
        self._age_cache.clear()
        self._thread = threading.Thread(target=self._loop, name="odc-service", daemon=True)
        self._thread.start()
        return self._session_id

    def stop(self, timeout_s: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def _per_class_gates(self) -> dict[str, float]:
        # The selected threshold is the minimum score shown and used for every
        # class; per-class values may tighten a class further.
        threshold = self._cfg.detection.threshold
        gates = {
            label: max(threshold, class_floor)
            for label, class_floor in PER_CLASS_THRESHOLDS.items()
        }
        gates.update(self._cfg.detection.per_class_thresholds or {})
        return gates

    def _snapshot(
        self,
        *,
        running: bool,
        captured: int,
        processed: int,
        dropped: int,
        latency: float,
        db_healthy: bool,
        model_healthy: bool,
        runtime: Any,
        health_status: HealthStatus,
        frame_seq: int,
        suppressed: int,
        duplicates: int,
        pose_on: bool = False,
        pose_healthy: bool = False,
        dms_on: bool = False,
        dms_healthy: bool = False,
        driver_seat_id: str = "",
        specialist_enabled: bool = False,
        specialist_healthy: bool = False,
        specialist_frame_seq: int = -1,
        specialist_error: str | None = None,
    ) -> ServiceSnapshot:
        cfg = self._cfg
        version = self._layout.version
        if version == "none":
            layout_source = "uncalibrated"
        elif version.startswith("auto"):
            layout_source = "auto-proposal"
        else:
            layout_source = "curated"
        return ServiceSnapshot(
            session_id=self._session_id,
            running=running,
            mode=cfg.mode,
            frames_captured=captured,
            frames_processed=processed,
            frames_dropped=dropped,
            last_latency_ms=latency,
            db_healthy=db_healthy,
            model_healthy=model_healthy,
            model_id=cfg.detection.model_id if model_healthy else "unavailable",
            provider=getattr(runtime, "providers_active", ["unset"])[0] if runtime else "unset",
            camera_status=health_status,
            calibrated=bool(self._layout.seats) and not version.startswith("auto"),
            updated_at=datetime.now(UTC),
            frame_seq=frame_seq,
            suppressed_hallucinations=suppressed,
            duplicates_removed=duplicates,
            threshold=cfg.detection.threshold,
            layout_source=layout_source,
            pose_on=pose_on,
            pose_healthy=pose_healthy,
            dms_on=dms_on,
            dms_healthy=dms_healthy,
            driver_seat_id=driver_seat_id,
            mask_mode=self._mask_mode,
            specialist_enabled=specialist_enabled,
            specialist_healthy=specialist_healthy,
            specialist_frame_seq=specialist_frame_seq,
            specialist_error=specialist_error,
        )

    def _record_alert(self, item: dict[str, Any], raised: bool, at: datetime | None = None) -> None:
        sev = item.get("severity", "WARNING")
        severity = sev if sev in ("INFO", "WARNING", "CRITICAL") else "WARNING"
        # Alerts share the EVIDENCE clock (the frame that raised them), never
        # wall time: replay frames carry virtual timestamps, and mixing
        # clocks scattered presence and alert markers on the timeline.
        # Frame seq travels in evidence_summary so the timeline can jump to
        # the exact frame without fragile event_id parsing.
        stamp = at or datetime.now(UTC)
        try:
            frame_seq = int(item.get("frame_seq", -1))
        except Exception:
            frame_seq = -1
        self._repo.append_event(
            StateEvent(
                event_id=f"{item.get('alert_id', 'alert')}-{'raised' if raised else 'cleared'}",
                session_id=self._session_id,
                event_type=EventType.ALERT_RAISED if raised else EventType.ALERT_CLEARED,
                severity=Severity(severity),
                seat_id=item.get("seat_id"),
                occurred_at=stamp,
                recorded_at=datetime.now(UTC),
                new_state="ACTIVE" if raised else "CLEARED",
                reason_codes=[str(item.get("kind", "alert"))],
                evidence_summary=f"frame_seq={frame_seq};{str(item.get('message', ''))}"[:500],
                config_version=self._cfg.seat_layout_version,
                model_set_version=self._cfg.model_set_version,
            )
        )

    def _sam_job(self, seq: int, pixels: Any, boxes: list[dict[str, Any]]) -> None:
        """Background precise-mask job: never blocks capture/inference."""
        try:
            sam = self._sam
            masks = sam.enhance(pixels, boxes) if sam is not None else []
            if masks:
                self.hub.note_precise(seq, masks)
        except Exception as e:
            log.warning("sam job degraded: %s", type(e).__name__)
        finally:
            with self._sam_lock:
                self._sam_busy = False

    def _maybe_dispatch_sam(
        self, seq: int, pixels: Any, detections: list[Any], objects: list[Any]
    ) -> None:
        """Single-slot SAM dispatch: latest frame wins, never queues."""
        if self._sam is None:
            return
        with self._sam_lock:
            if self._sam_busy or seq - self._sam_last_seq < 40:
                return
            self._sam_busy = True
            self._sam_last_seq = seq
        boxes = [
            {
                "label": d.class_label,
                "bbox": [d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max],
                "color": CLASS_COLORS.get(d.class_label, DEFAULT_CLASS_COLOR),
            }
            for d in (detections + objects)
            if d.class_label in ("person", "chair", "couch", "suitcase", "handbag", "backpack")
        ]
        threading.Thread(
            target=self._sam_job, args=(seq, pixels, boxes), name="odc-sam", daemon=True
        ).start()

    def _loop(self) -> None:
        from .tracking import ByteTrackStyleTracker, TrackerConfig

        cfg = self._cfg
        source = self._build_source()
        try:
            first = source.open()
        except Exception as e:
            log.error("source open failed: %s: %s", type(e).__name__, e)
            self.hub.publish(
                ServiceSnapshot(
                    session_id=self._session_id,
                    mode=cfg.mode,
                    running=False,
                    model_healthy=False,
                    camera_status=HealthStatus.UNHEALTHY,
                    updated_at=datetime.now(UTC),
                ),
                [],
                [],
            )
            return
        runtime = self._build_runtime()
        model_healthy = runtime is not None
        specialist: Any = None
        specialist_result: Any = None
        specialist_applied_frame = -1
        if cfg.detection.child_specialist_enabled:
            try:
                from .specialist import ChildSpecialist

                specialist = ChildSpecialist(
                    checkpoint=cfg.detection.child_specialist_checkpoint,
                    config=cfg.detection.child_specialist_config,
                    threshold=cfg.detection.child_specialist_threshold,
                )
                specialist.start()
                if not specialist.available:
                    log.warning("child specialist unavailable; child fields remain UNKNOWN")
            except Exception as e:
                log.warning("child specialist setup degraded: %s", type(e).__name__)
                specialist = None
        age_estimator: Any = None
        if cfg.age.enabled and runtime is not None:
            try:
                from .age import MiVOLOV2AgeEstimator

                age_estimator = MiVOLOV2AgeEstimator(
                    cfg.age.model_path,
                    device=cfg.age.device,
                    infant_max_years=cfg.age.infant_max_years,
                    child_max_years=cfg.age.child_max_years,
                    uncertainty_margin_years=cfg.age.uncertainty_margin_years,
                    gender_min_score=cfg.age.gender_min_score,
                    history_max=cfg.age.history_max,
                    track_expiry_frames=cfg.age.track_expiry_frames,
                )
                age_estimator.load()
                self._age_estimator = age_estimator
                self._age_device_name = age_estimator.device
                self._age_error = None
            except Exception as e:
                self._age_error = f"{type(e).__name__}: {e}"
                log.exception("MiVOLO age/gender unavailable; classifications remain unknown")
                self._age_estimator = None
                self._age_device_name = "unavailable"
                age_estimator = None
        # Cabin-tuned association: low-score person boxes can be the only
        # visible evidence for a child in a restraint. When age inference is
        # enabled, expose a new track immediately so it can reach MiVOLO; seat
        # occupancy still requires the independent state engine's multi-frame
        # dwell. Without age inference, retain the three-hit track birth gate.
        tracker = ByteTrackStyleTracker(
            TrackerConfig(
                high_thresh=0.35,
                low_thresh=0.15,
                match_iou=0.3,
                min_hits=1 if age_estimator is not None else 3,
            )
        )
        # Fresh per video: strap votes must never leak across uploads.
        belt_tracker = SeatBeltTracker()
        from .dms import DmsTracker

        dms_tracker = DmsTracker()
        last_dms: dict[str, Any] = {}
        face_mesh = None
        dms_healthy = False
        driver_seat_id = next(
            (seat.seat_id for seat in self._layout.seats if seat.physical_role == "driver"),
            "",
        )
        if cfg.dms.enabled or age_estimator is not None:
            try:
                from .dms import FaceMesh

                face_mesh = FaceMesh(
                    cfg.dms.model_path,
                    num_faces=max(cfg.dms.num_faces, 4 if age_estimator is not None else 1),
                )
                face_mesh.load()
                dms_healthy = cfg.dms.enabled
            except Exception as e:
                log.warning("face mesh unavailable, dms off: %s", type(e).__name__)
                face_mesh = None
        alert_manager = AlertManager(
            unattended_after_s=cfg.alerts.unattended_after_s,
            cooldown_s=cfg.alerts.cooldown_s,
        )
        webhook = WebhookSender(cfg.alerts.webhook_url) if cfg.alerts.webhook_url else None
        if webhook is not None:
            webhook.start()
        pose_est = None
        pose_healthy = False
        if cfg.pose.enabled and runtime is not None:
            try:
                from .pose import MediaPipePose

                pose_est = MediaPipePose(cfg.pose.model_path)
                pose_healthy = True
            except Exception as e:
                log.warning("pose unavailable, continuing without it: %s", type(e).__name__)
        if self._mask_mode == "precise":
            try:
                from .sam import SamPromptedMasks

                sam = SamPromptedMasks()
                sam.load()
                self._sam = sam
            except Exception as e:
                log.warning("precise masks unavailable, fast fallback: %s", type(e).__name__)
                self._sam = None
                self._mask_mode = "fast"
        engine = SeatStateEngine(
            self._session_id,
            self._layout,
            self._rules,
            cfg.seat_layout_version,
            cfg.model_set_version,
        )
        self._engine = engine
        health_writer = CameraHealthWriter(self._repo, cfg.camera.health_min_interval_s)
        db_healthy = False
        session_started = False
        try:
            db_healthy = self._repo.health_check()
            if db_healthy:
                self._repo.ensure_schema_version(2)
                self._repo.start_session(
                    SessionMeta(
                        session_id=self._session_id,
                        camera_id=cfg.camera.camera_id,
                        source_kind=source.kind,
                        mode=cfg.mode,
                        started_at=first.captured_at,
                        build_version=cfg.build_version,
                        config_version=cfg.seat_layout_version,
                    )
                )
                session_started = True
        except Exception as e:
            log.warning("plomid unavailable, transient-only: %s", type(e).__name__)
            db_healthy = False

        inbox: queue.Queue[Any] = queue.Queue(maxsize=cfg.processing.queue_max)
        dropped = 0
        captured = 0
        processed = 0  # inference frames only — incremented exactly once per inference
        frames_seen = 0  # every captured frame (drives sampling, never double-counted)
        last_latency = 0.0
        last_provider = "unset"
        last_stale_check = time.monotonic()
        last_heartbeat = 0.0
        last_reconnect = 0.0
        last_db_probe = 0.0
        suppressed_total = 0
        duplicates_total = 0
        all_events: list[StateEvent] = []
        last_pose: dict[str, Any] = {"frame_seq": -1, "skeletons": []}
        gates = self._per_class_gates()
        try:
            while not self._stop.is_set():
                # Recover persistence after PostgreSQL/PLOMID restarts without
                # requiring the operator to restart the perception session.
                # Existing writes remain idempotent; the UI changes from DOWN
                # back to healthy only after a real probe succeeds.
                now = time.monotonic()
                if not db_healthy and now - last_db_probe >= 5.0:
                    last_db_probe = now
                    try:
                        if self._repo.health_check():
                            self._repo.ensure_schema_version(2)
                            if not session_started:
                                self._repo.start_session(
                                    SessionMeta(
                                        session_id=self._session_id,
                                        camera_id=cfg.camera.camera_id,
                                        source_kind=source.kind,
                                        mode=cfg.mode,
                                        started_at=first.captured_at,
                                        build_version=cfg.build_version,
                                        config_version=cfg.seat_layout_version,
                                    )
                                )
                                session_started = True
                            db_healthy = True
                            log.info("plomid persistence recovered")
                    except Exception as e:
                        log.warning("plomid recovery probe failed: %s", type(e).__name__)
                frame = source.next()
                if frame is None:
                    if isinstance(source, FileReplaySource) and source._ended:
                        if cfg.camera.replay_loop:
                            log.info("replay loop: reopening")
                            try:
                                source.close()
                            except Exception:
                                pass
                            source.open()  # seq resets; engine re-confirms
                            continue
                        log.info("replay ended cleanly")
                        break
                    if isinstance(source, LiveCameraSource):
                        # Live reconnect with backoff: the capture adapter
                        # drops its handle on read failure; reopen it here
                        # instead of spinning on None forever.
                        now = time.monotonic()
                        if now - last_reconnect > 2.0:
                            last_reconnect = now
                            try:
                                try:
                                    source.close()
                                except Exception:
                                    pass
                                source.open()
                                log.info("live camera reconnected")
                                continue
                            except Exception as e:
                                log.warning("live reconnect failed: %s", type(e).__name__)
                        time.sleep(0.2)
                        continue
                    time.sleep(0.05)
                    continue
                try:
                    inbox.put_nowait(frame)
                except queue.Full:
                    try:
                        inbox.get_nowait()
                    except queue.Empty:
                        pass
                    inbox.put_nowait(frame)
                    dropped += 1
                try:
                    work = inbox.get(timeout=0.2)
                except queue.Empty:
                    continue
                if work.pixels is None:
                    continue
                captured += 1
                frames_seen += 1
                # Continuous video: preview JPEG for EVERY captured frame so
                # the stream never freezes while inference samples.
                preview = _encode_preview(work.pixels)
                try:
                    ts_iso = work.meta.captured_at.isoformat()
                except Exception:
                    ts_iso = ""
                self.hub.preview_only(preview, work.meta.seq, ts_iso)
                if frames_seen % max(1, cfg.processing.sample_every_n) != 1:
                    continue
                detections: list[Any] = []
                objects: list[Any] = []
                specialist_detections: list[Any] = []
                suppressed = 0
                duplicates = 0
                if runtime is not None:
                    try:
                        result = runtime.infer(work)
                        final, stats = postprocess_detections(
                            result.detections,
                            per_class_gates=gates,
                            nms_iou=cfg.detection.nms_iou,
                            floor=cfg.detection.threshold,
                        )
                        suppressed = int(stats.get("suppressed", 0))
                        duplicates = int(stats.get("duplicates", 0))
                        suppressed_total += suppressed
                        duplicates_total += duplicates
                        detections = [d for d in final if d.class_label == "person"]
                        objects = [d for d in final if d.class_label != "person"]
                        last_latency = result.duration_ms
                        last_provider = result.provider or "unset"
                    except Exception as e:
                        log.warning("infer failed: %s", type(e).__name__)
                if specialist is not None:
                    try:
                        completed = specialist.poll()
                        if completed is not None:
                            specialist_result = completed
                        if (
                            specialist.available
                            and frames_seen % max(1, cfg.detection.child_specialist_every_n) == 1
                        ):
                            specialist.submit(self._session_id, work.meta.seq, work.pixels)
                        if (
                            specialist_result is not None
                            and specialist_result.session_id == self._session_id
                            and specialist_result.frame_seq != specialist_applied_frame
                        ):
                            from .specialist import specialist_as_detections

                            specialist_detections = specialist_as_detections(specialist_result)
                            specialist_applied_frame = specialist_result.frame_seq
                    except Exception as e:
                        log.warning("child specialist poll degraded: %s", type(e).__name__)
                if specialist_result is not None:
                    self.hub.note_specialist_detections(
                        [
                            {
                                "label": candidate.label,
                                "phrase": candidate.phrase,
                                "score": candidate.score,
                                "bbox": [
                                    candidate.bbox.x_min,
                                    candidate.bbox.y_min,
                                    candidate.bbox.x_max,
                                    candidate.bbox.y_max,
                                ],
                                "frame_seq": candidate.frame_seq,
                                "model_id": candidate.model_id,
                            }
                            for candidate in specialist_result.candidates
                        ]
                    )
                # Segmentation masks + crops for THIS frame (transient).
                masks_out: list[dict[str, Any]] = []
                crops_out: dict[str, bytes] = {}
                try:
                    if cfg.detection.enable_segmentation and work.pixels is not None:
                        all_dets = detections + objects
                        mask_results = detection_masks(
                            work.pixels, all_dets, max_masks=cfg.detection.mask_max
                        )
                        for m in mask_results:
                            masks_out.append(
                                {
                                    "label": m.label,
                                    "score": m.score,
                                    "polygon": [[x, y] for x, y in m.polygon],
                                    "bbox": list(m.bbox),
                                    "color": m.color,
                                    "fallback": m.fallback,
                                    "evidence_kind": m.evidence_kind,
                                }
                            )
                        for i, det in enumerate(all_dets[:12]):
                            crop = crop_detection(work.pixels, det.bbox)
                            if crop is not None:
                                crops_out[f"crop-{i}"] = crop
                except Exception as e:
                    log.warning("segmentation degraded: %s", type(e).__name__)
                try:
                    states = tracker.update(work, detections)
                    engine.set_camera_health(source.health())
                    out = engine.update(
                        work,
                        states,
                        self._layout.seats,
                        objects,
                        specialist_detections,
                    )
                except Exception as e:
                    # One bad frame must never kill the stream: keep the
                    # latest frame visible, count the drop, carry on.
                    log.exception(
                        "track/state failed on frame %d: %s", work.meta.seq, type(e).__name__
                    )
                    dropped += 1
                    continue
                processed += 1
                self.hub.note_specialist_decisions(
                    list(getattr(engine, "last_specialist_decisions", []))
                )
                # Seat rows + counts with stable unique seat IDs.
                try:
                    if out.seats:
                        rows = build_seat_map(list(out.seats), list(out.occupants), work.meta.seq)
                    else:
                        rows = list(self.hub.seat_rows)
                except Exception:
                    rows = list(self.hub.seat_rows)
                age_predictions: dict[str, Any] = {}
                age_due = processed % max(1, cfg.age.every_n) == 0
                # Body pose (optional supporting evidence): largest person
                # boxes at a gated cadence. Never drives occupancy itself;
                # skeletons feed the overlay, seat association, belt-torso
                # ROIs and frontality gating.
                pose_skeletons: list[dict[str, Any]] = []
                pose_by_track: dict[str, dict[str, Any]] = {}
                if (
                    pose_est is not None
                    and detections
                    and processed % max(1, cfg.pose.every_n) == 0
                ):
                    try:
                        from .pose import associate_pose_detections
                        from .seats import clip_bbox_to_seat_region

                        # Pose only the detector boxes matched to an active,
                        # seat-associated person track. Generic object boxes
                        # and geometry-only seat guesses are not pose evidence.
                        associated = associate_pose_detections(
                            detections,
                            out.occupants,
                        )
                        # Skip merged full-frame boxes and tiny fragments;
                        # keep the strongest usable person boxes first.
                        seat_by_id = {seat.seat_id: seat for seat in self._layout.seats}
                        usable_targets: list[tuple[Any, Any, Any]] = []
                        for detection, occupant in associated:
                            seat = seat_by_id.get(occupant.seat_id or "")
                            if seat is None:
                                continue
                            pose_box = clip_bbox_to_seat_region(detection.bbox, seat)
                            if pose_box is None:
                                continue
                            pose_area = (pose_box.x_max - pose_box.x_min) * (
                                pose_box.y_max - pose_box.y_min
                            )
                            if 0.03 <= pose_area <= 0.70:
                                usable_targets.append((detection, occupant, pose_box))
                        targets = sorted(
                            usable_targets,
                            key=lambda pair: (
                                (pair[2].x_max - pair[2].x_min) * (pair[2].y_max - pair[2].y_min)
                            ),
                            reverse=True,
                        )[: cfg.pose.max_targets]
                        kp = pose_est.estimate(work, [target[2] for target in targets])
                        for idx, points in kp.items():
                            if idx >= len(targets):
                                continue
                            _pdet, occupant, _pose_box = targets[idx]
                            pseat = occupant.seat_id
                            if pseat is None:
                                continue
                            visible = sum(1 for point in points if point.visible)
                            entry = {
                                "seat": pseat,
                                "track_id": occupant.track_id,
                                "bbox": occupant.last_bbox.model_dump()
                                if occupant.last_bbox is not None
                                else None,
                                "inference_bbox": _pose_box.model_dump(),
                                "visible": visible,
                                "points": [
                                    [point.x, point.y, 1.0 if point.visible else 0.0]
                                    for point in points
                                ],
                            }
                            pose_skeletons.append(entry)
                            pose_by_track[occupant.track_id] = entry
                        # Attempted this frame: publish even when empty (person
                        # left) so stale skeletons never linger; skipped frames
                        # keep the last pose instead of blinking.
                        last_pose = {"frame_seq": work.meta.seq, "skeletons": pose_skeletons}
                    except Exception as e:
                        log.warning("pose degraded: %s", type(e).__name__)
                # Faces (optional previews): estimated at a gated cadence;
                # matching + tracker run below after rows exist.
                face_results: list[Any] = []
                faces_estimated = False
                if (
                    face_mesh is not None
                    and (
                        (age_estimator is not None and age_due)
                        or (cfg.dms.enabled and processed % max(1, cfg.dms.every_n) == 0)
                    )
                    and out.occupants
                ):
                    try:
                        face_results = face_mesh.estimate(work)
                        faces_estimated = True
                    except Exception as e:
                        log.warning("face degraded: %s", type(e).__name__)
                if age_estimator is not None and age_due:
                    try:
                        from .age import apply_age_predictions

                        age_predictions = age_estimator.estimate(
                            work.pixels,
                            list(out.occupants),
                            face_results,
                            work.meta.seq,
                            layout=self._layout,
                        )
                        self._age_error = None
                        self._age_cache.update(age_predictions)
                        active_track_ids = {
                            occupant.track_id
                            for occupant in out.occupants
                            if occupant.seat_id is not None
                        }
                        self._age_cache = {
                            track_id: prediction
                            for track_id, prediction in self._age_cache.items()
                            if track_id in active_track_ids
                        }
                    except Exception as e:
                        self._age_error = f"{type(e).__name__}: {e}"
                        self._age_cache.clear()
                        log.exception("MiVOLO inference failed; age/gender remain unknown")
                        for row in rows:
                            row["child"] = "Unknown"
                            row["child_presence"] = "UNKNOWN"
                            row["occupant_category"] = "unknown"
                            row["gender"] = "UNKNOWN"
                            row["gender_conf"] = None
                            row["age_reason"] = "mivolo-inference-fault"
                            row["age_error"] = self._age_error
                if cfg.age.enabled:
                    from .age import apply_age_predictions

                    active_track_ids = {
                        occupant.track_id
                        for occupant in out.occupants
                        if occupant.seat_id is not None
                    }
                    current_predictions = {
                        track_id: prediction
                        for track_id, prediction in self._age_cache.items()
                        if track_id in active_track_ids
                    }
                    apply_age_predictions(
                        rows,
                        current_predictions if age_estimator is not None else {},
                        layout=self._layout,
                        child_candidate_max_years=cfg.age.child_max_years,
                    )
                    if self._age_error:
                        for row in rows:
                            if row.get("child") == "Unknown":
                                row["age_reason"] = "mivolo-unavailable-or-inference-fault"
                                row["age_error"] = self._age_error
                # DMS per seat: faces matched to occupant boxes (smallest
                # containing box wins), phone overlap in the lower body,
                # temporal tracker. Skipped frames reuse the last state so
                # the cadence never resets confirmation progress.
                from .dms import phone_near_body

                dms_faces_by_seat: dict[str, Any] = {}
                face_payload: dict[str, Any] | None = None
                try:
                    if faces_estimated:
                        phones = [d for d in objects if d.class_label == "cell phone"]
                        matched: list[tuple[Any, str | None]] = []
                        for face in face_results:
                            fx = (face.bbox[0] + face.bbox[2]) / 2.0
                            fy = (face.bbox[1] + face.bbox[3]) / 2.0
                            best_sid: str | None = None
                            best_area: float | None = None
                            for occ in out.occupants:
                                b = occ.last_bbox
                                if b is None or occ.seat_id is None:
                                    continue
                                if b.x_min <= fx <= b.x_max and b.y_min <= fy <= b.y_max:
                                    area = (b.x_max - b.x_min) * (b.y_max - b.y_min)
                                    if best_area is None or area < best_area:
                                        best_area = area
                                        best_sid = occ.seat_id
                            matched.append((face, best_sid))
                        by_seat: dict[str, Any] = {}
                        for face, sid in matched:
                            if sid is not None and sid not in by_seat:
                                by_seat[sid] = face
                        payload_faces = []
                        for row in rows:
                            sid = row["seat_id"]
                            occupied = row["presence"] in ("OCCUPIED", "LIKELY_OCCUPIED")
                            face = by_seat.get(sid)
                            if face is not None:
                                dms_faces_by_seat[sid] = face
                            boxes = [
                                o.last_bbox
                                for o in out.occupants
                                if o.seat_id == sid and o.last_bbox is not None
                            ]
                            near = phone_near_body(phones, boxes) if phones and boxes else False
                            if sid == driver_seat_id:
                                st = dms_tracker.update(sid, occupied, face, near)
                            else:
                                # Face landmarks may be visible for passengers,
                                # but driver-attention verdicts and alerts are
                                # produced only for the configured driver seat.
                                from .dms import DmsState

                                st = DmsState(
                                    face=face is not None,
                                    ear=face.ear_value if face is not None else None,
                                    jaw=face.jaw if face is not None else 0.0,
                                    yaw=face.yaw if face is not None else None,
                                    pitch=face.pitch if face is not None else None,
                                    reason="passenger-face-not-driver-evaluated",
                                )
                            row["dms"] = st.to_dict()
                            flags = [
                                code
                                for code, on in (
                                    ("DROW", st.drowsy),
                                    ("DIST", st.distracted),
                                    ("PHON", st.phone),
                                    ("YAWN", st.yawning),
                                )
                                if on
                            ]
                            row["dms_text"] = "/".join(flags) or "ok" if face is not None else "–"
                            last_dms[sid] = (row["dms"], row["dms_text"])
                            if face is not None:
                                payload_faces.append(
                                    {
                                        "seat": sid,
                                        "dots": {k: [v[0], v[1]] for k, v in face.dots.items()},
                                        "blink": round(face.blink, 3),
                                        "jaw": round(face.jaw, 3),
                                    }
                                )
                        face_payload = {
                            "frame_seq": work.meta.seq,
                            "faces": payload_faces,
                            "detected_face_boxes": [
                                {
                                    "bbox": list(face.bbox),
                                    "seat_id": seat_id,
                                }
                                for face, seat_id in matched
                            ],
                        }
                    else:
                        for row in rows:
                            sid = row["seat_id"]
                            if sid == driver_seat_id and cfg.dms.enabled and not dms_healthy:
                                cached = (
                                    {
                                        "drowsy": False,
                                        "distracted": False,
                                        "phone": False,
                                        "yawning": False,
                                        "face": False,
                                        "reason": "dms-model-unavailable",
                                    },
                                    "UNAVAILABLE",
                                )
                            else:
                                cached = last_dms.get(sid, (None, "–"))
                            row["dms"], row["dms_text"] = cached
                except Exception as e:
                    log.warning("dms degraded: %s", type(e).__name__)
                # Seat belts: geometric strap evidence per occupied seat,
                # temporal per-seat votes, frontality-gated absence. Seat
                # association reuses the engine's track→seat verdicts (never
                # nearest-box guessing), and missing/empty detections yield
                # Unknown — never a fabricated fastened/unfastened verdict.
                belt_overlays: list[dict[str, Any]] = []
                try:
                    from .belt import estimate_frontal_fallback
                    from .seats import clip_bbox_to_seat_region

                    # Whole-frame brightness once per frame (not per seat):
                    # feeds the frontality fallback below.
                    try:
                        import cv2 as _cv2
                        import numpy as _np

                        _gray = _cv2.cvtColor(_np.asanyarray(work.pixels), _cv2.COLOR_BGR2GRAY)
                        frame_bright = 35.0 <= float(_gray.mean()) <= 220.0
                    except Exception:
                        frame_bright = False
                    for row in rows:
                        sid = row["seat_id"]
                        occupied = row["presence"] in ("OCCUPIED", "LIKELY_OCCUPIED")
                        posture_skeleton: dict[str, Any] | None = None
                        seat_geometry = next(
                            (seat for seat in self._layout.seats if seat.seat_id == sid),
                            None,
                        )
                        occupant_boxes = [
                            (o.track_id, o.last_bbox)
                            for o in out.occupants
                            if o.seat_id == sid and o.last_bbox is not None
                        ]
                        if seat_geometry is not None:
                            occupant_boxes = [
                                (track_id, clipped)
                                for track_id, box in occupant_boxes
                                if (clipped := clip_bbox_to_seat_region(box, seat_geometry))
                                is not None
                            ]
                        obs = None
                        frontal: bool | None = None
                        row["belt_observation"] = "unavailable"
                        row["belt_observation_reason"] = "no-usable-person-torso"
                        row["belt_observation_score"] = None
                        row["belt_observation_roi"] = None
                        if occupied and occupant_boxes and work.pixels is not None:
                            # Largest USABLE box: giant merged person+car boxes
                            # (>0.70, measured S2) carry no torso to judge, so
                            # a tighter box on the same seat wins over them.
                            usable = [
                                (track_id, box)
                                for track_id, box in occupant_boxes
                                if (box.x_max - box.x_min) * (box.y_max - box.y_min) <= 0.70
                            ]
                            selected_track_id, box = max(
                                usable or occupant_boxes,
                                key=lambda item: (
                                    (item[1].x_max - item[1].x_min)
                                    * (item[1].y_max - item[1].y_min)
                                ),
                            )
                            skel = pose_by_track.get(selected_track_id)
                            posture_skeleton = skel
                            shoulders = hips = None
                            dms_face = dms_faces_by_seat.get(sid)
                            if dms_face is not None:
                                # A found face is frontal enough to judge.
                                frontal = True
                            elif skel is not None:
                                pts = skel["points"]
                                shoulders = [
                                    (p[0], p[1])
                                    for i, p in enumerate(pts)
                                    if i in (11, 12) and p[2] >= 0.5
                                ]
                                hips = [
                                    (p[0], p[1])
                                    for i, p in enumerate(pts)
                                    if i in (23, 24) and p[2] >= 0.5
                                ]
                                face = sum(
                                    1
                                    for i, p in enumerate(pts)
                                    if i in (0, 1, 2, 3, 4, 5, 6, 9, 10) and p[2] >= 0.5
                                )
                                frontal = face >= 2
                                if len(shoulders) < 2:
                                    shoulders = None
                                if len(hips) < 2:
                                    hips = None
                            try:
                                obs = detect_belt(work.pixels, box, shoulders, hips)
                            except Exception as e:
                                log.warning("belt degraded: %s", type(e).__name__)
                                obs = None
                            row["belt_observation"] = (
                                "unavailable"
                                if obs is None
                                else "strap_candidate"
                                if obs.state == "strap"
                                else "clear_candidate"
                                if obs.state == "clear"
                                else "unclear"
                            )
                            row["belt_observation_reason"] = (
                                obs.reason if obs is not None else "no-usable-person-torso"
                            )
                            row["belt_observation_score"] = obs.score if obs is not None else None
                            row["belt_observation_roi"] = (
                                list(obs.roi) if obs is not None and obs.roi else None
                            )
                            if frontal is None and obs is not None:
                                # Pose/face estimators off (default): fall back
                                # to geometry so belt verdicts still work on
                                # real video. Unclear/dark/tiny torsos stay
                                # None → Unknown, never a false negative.
                                frontal = estimate_frontal_fallback(box, frame_bright)
                                if frontal is True and obs.state == "clear":
                                    row["belt_frontal"] = "fallback-geometry"
                        bst = belt_tracker.update(sid, occupied, obs, frontal)
                        row["belt"] = bst.state
                        row["belt_reason"] = bst.reason
                        row["belt_issue"] = bst.issue
                        row["belt_status_semantics"] = (
                            "Detected means a repeated strap-shaped visual candidate; "
                            "belt identity and buckle fastening are not verified"
                        )
                        if bst.state == "Detected" and obs is not None and obs.segments:
                            belt_overlays.append(
                                {
                                    "seat_id": sid,
                                    "segments": [list(s) for s in obs.segments],
                                    "state": bst.state,
                                }
                            )
                        elif bst.state == "Not detected" and obs is not None and obs.roi:
                            belt_overlays.append(
                                {"seat_id": sid, "roi": list(obs.roi), "state": bst.state}
                            )
                        skel = posture_skeleton
                        row["pose"] = skel["visible"] if skel is not None else None
                        row["posture"] = "UNKNOWN"
                        row["posture_confidence"] = None
                        row["posture_landmarks_complete"] = None
                        row["posture_confidence_semantics"] = (
                            "unavailable: posture model is not calibrated"
                        )
                        if skel is not None and work.pixels is not None:
                            from .pose import classify_seated_posture

                            height, width = work.pixels.shape[:2]
                            (
                                row["posture"],
                                row["posture_reason"],
                                row["posture_landmarks_complete"],
                            ) = classify_seated_posture(skel["points"], width, height)
                        else:
                            row["posture_reason"] = "pose-not-visible"
                except Exception as e:
                    log.warning("belt/pose enrichment degraded: %s", type(e).__name__)
                # Gender appearance (EXPERIMENTAL, owner-accepted D30): opt-in
                # CLIP probe on large non-child occupant crops, throttled to
                # every 10th inference frame with per-seat cached verdicts.
                # Transient DISPLAY ONLY — never persisted (DB writes use
                # out.seats, never rows), never aggregated, never safety.
                if cfg.detection.enable_gender_appearance and age_estimator is None:
                    try:
                        tick_now = self._gender_tick
                        self._gender_tick += 1
                        if self._gender_probe is None and not self._gender_tried:
                            self._gender_tried = True
                            self._gender_loading = True
                            try:
                                from .demographics import GenderProbe

                                gp = GenderProbe()
                                gp.load()
                                self._gender_probe = gp
                                self._gender_error = None
                            except Exception as e:
                                self._gender_error = f"{type(e).__name__}: {e}"
                                log.warning("gender probe unavailable: %s", self._gender_error)
                            finally:
                                self._gender_loading = False
                        if self._gender_probe is not None and work.pixels is not None:
                            if tick_now % 10 == 0:
                                from .demographics import MIN_BOX_AREA as _GMIN
                                from .demographics import gender_for_seats as _gfs

                                crops: dict[str, Any] = {}
                                for row in rows:
                                    if row.get("presence") not in (
                                        "OCCUPIED",
                                        "LIKELY_OCCUPIED",
                                    ):
                                        continue
                                    if row.get("child") == "Yes":
                                        continue
                                    boxes = [
                                        o.last_bbox
                                        for o in out.occupants
                                        if o.seat_id == row["seat_id"] and o.last_bbox is not None
                                    ]
                                    if not boxes:
                                        continue
                                    box = max(
                                        boxes,
                                        key=lambda b: (b.x_max - b.x_min) * (b.y_max - b.y_min),
                                    )
                                    area = (box.x_max - box.x_min) * (box.y_max - box.y_min)
                                    if area < _GMIN:
                                        continue
                                    try:
                                        import numpy as _np

                                        frame_arr = _np.asanyarray(work.pixels)
                                        im_h, im_w = (
                                            int(frame_arr.shape[0]),
                                            int(frame_arr.shape[1]),
                                        )
                                        px0 = max(0, int(box.x_min * im_w))
                                        py0 = max(0, int(box.y_min * im_h))
                                        px1 = min(im_w, int(box.x_max * im_w))
                                        py1 = min(im_h, int(box.y_max * im_h))
                                        if px1 - px0 < 16 or py1 - py0 < 16:
                                            continue
                                        crops[row["seat_id"]] = (
                                            frame_arr[py0:py1, px0:px1],
                                            area,
                                        )
                                    except Exception:
                                        continue
                                if crops:
                                    for sid, verdict in _gfs(
                                        rows, crops, self._gender_probe
                                    ).items():
                                        self._gender_cache[sid] = verdict
                    except Exception as e:
                        log.warning("gender degraded: %s", type(e).__name__)
                    for row in rows:
                        cached = self._gender_cache.get(row["seat_id"])
                        if cached is not None:
                            row["gender"] = cached[0]
                            row["gender_conf"] = cached[1]
                try:
                    counts = live_counts(rows)
                except Exception:
                    counts, rows = dict(self.hub.counts), list(self.hub.seat_rows)
                # Alerts: derived, cooldown-gated; WARNING+ transitions persist.
                # The evidence clock (frame captured_at) drives both the
                # manager and the persisted events so timeline markers for
                # occupancy and alerts share one time base.
                try:
                    frame_now = work.meta.captured_at
                    active_alerts, raised, cleared = alert_manager.update(
                        rows, self._session_id, work.meta.seq, frame_now
                    )
                    self.hub.note_alerts(active_alerts, raised)
                    order = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
                    floor = order.get(cfg.alerts.min_severity, 1)
                    for item, is_raised in [(a, True) for a in raised] + [
                        (a, False) for a in cleared
                    ]:
                        if db_healthy and order.get(item.get("severity", "WARNING"), 1) >= 1:
                            try:
                                self._record_alert(item, is_raised, at=frame_now)
                            except Exception:
                                db_healthy = False
                        if (
                            webhook is not None
                            and order.get(item.get("severity", "WARNING"), 1) >= floor
                        ):
                            webhook.send(
                                {
                                    "session_id": self._session_id,
                                    "raised": is_raised,
                                    "frame_seq": work.meta.seq,
                                    **item,
                                }
                            )
                except Exception as e:
                    log.warning("alerts degraded: %s", type(e).__name__)
                # Detection overlay annotations (seat + child tag) need the
                # rows above, so det_dicts are built here — after the engine.
                try:
                    # Specialist boxes are evidence candidates and are often
                    # overlapping prompt duplicates. Keep them in seat rows,
                    # diagnostics and SQL persistence, but do not cover the
                    # live image with raw open-vocabulary labels.
                    display_detections = detections + [
                        item
                        for item in objects
                        if item.class_label not in {"chair", "couch"} and item.score >= 0.55
                    ]
                    annotated = annotate_detections(display_detections, self._layout, rows)
                    det_dicts = [
                        {
                            "label": d.class_label,
                            "score": d.score,
                            "bbox": [d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max],
                            "seat": seat_id,
                            "tag": tag,
                        }
                        for d, seat_id, tag in annotated
                    ]
                except Exception:
                    det_dicts = [
                        {
                            "label": d.class_label,
                            "score": d.score,
                            "bbox": [d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max],
                            "seat": None,
                            "tag": None,
                        }
                        for d in display_detections
                    ]
                self.hub.note_full_frame(
                    preview,
                    det_dicts,
                    masks_out,
                    crops_out,
                    last_latency,
                    work.meta.seq,
                    ts_iso,
                    counts,
                    belt_overlays,
                    last_pose,
                    face_payload,
                    rows,
                    [
                        {
                            "label": item.label,
                            "phrase": item.phrase,
                            "score": item.score,
                            "bbox": [
                                item.bbox.x_min,
                                item.bbox.y_min,
                                item.bbox.x_max,
                                item.bbox.y_max,
                            ],
                            "frame_seq": item.frame_seq,
                            "model_id": item.model_id,
                        }
                        for item in (
                            specialist_result.candidates if specialist_result is not None else ()
                        )
                    ],
                    list(getattr(engine, "last_specialist_decisions", [])),
                    specialist_result.frame_seq if specialist_result is not None else -1,
                    last_provider,
                )
                # Persist structured frame outputs and inference evidence.
                # Pixels remain outside PLOMID; structured seat rows, pose,
                # specialist decisions, and detection boxes remain queryable.
                if db_healthy:
                    try:
                        self._repo.write_frame_observation(
                            self._session_id,
                            work.meta.seq,
                            work.meta.captured_at,
                            rows,
                            last_pose,
                            list(getattr(engine, "last_specialist_decisions", [])),
                        )
                        detection_writer = getattr(self._repo, "write_frame_detections", None)
                        if detection_writer is not None:
                            all_frame_detections = detections + objects + specialist_detections
                            annotations = annotate_detections(
                                all_frame_detections, self._layout, rows
                            )
                            frame_detections = []
                            for index, det in enumerate(all_frame_detections):
                                _, seat_id, tag = annotations[index]
                                frame_detections.append(
                                    {
                                        "detection_id": (
                                            f"{self._session_id}:{work.meta.seq}:{index}"
                                        ),
                                        "session_id": self._session_id,
                                        "frame_seq": work.meta.seq,
                                        "captured_at": work.meta.captured_at,
                                        "class_label": det.class_label,
                                        "confidence": float(det.score),
                                        "bbox": (
                                            f"[{float(det.bbox.x_min)},{float(det.bbox.y_min)},"
                                            f"{float(det.bbox.x_max)},{float(det.bbox.y_max)}]"
                                        ),
                                        "seat_id": seat_id,
                                        "track_id": None,
                                        "metadata": {
                                            "tag": tag,
                                            "source": det.evidence_kind,
                                            "model_id": det.model_id,
                                            "source_frame_seq": det.frame_seq,
                                        },
                                    }
                                )
                            if frame_detections:
                                detection_writer(frame_detections)
                        self._repo.write_metric_batch(
                            [
                                {
                                    "session_id": self._session_id,
                                    "model_id": getattr(runtime, "_model_id", "unset") or "unset",
                                    "provider": last_provider,
                                    "captured_at": work.meta.captured_at,
                                    "inference_ms": last_latency,
                                    "queue_ms": 0.0,
                                    "dropped_frames": dropped,
                                }
                            ]
                        )
                    except Exception as e:
                        log.warning("metric persist failed, transient-only: %s", type(e).__name__)
                        db_healthy = False
                self._maybe_dispatch_sam(work.meta.seq, work.pixels, detections, objects)
                if not out.seats and not out.events and not out.occupants:
                    # Empty output (uncalibrated geometry or no evidence):
                    # heartbeat at most 1/s so the console shows a scanning
                    # system, never a dead one. Last-known seats are kept.
                    if time.monotonic() - last_heartbeat > 1.0:
                        last_heartbeat = time.monotonic()
                        self.hub.publish(
                            self._snapshot(
                                running=True,
                                captured=captured,
                                processed=processed,
                                dropped=dropped,
                                latency=last_latency,
                                db_healthy=db_healthy,
                                model_healthy=model_healthy,
                                runtime=runtime,
                                health_status=source.health().status,
                                frame_seq=work.meta.seq,
                                suppressed=suppressed_total,
                                duplicates=duplicates_total,
                                pose_on=cfg.pose.enabled,
                                pose_healthy=pose_healthy,
                                dms_on=cfg.dms.enabled,
                                dms_healthy=dms_healthy,
                                driver_seat_id=driver_seat_id,
                                specialist_enabled=cfg.detection.child_specialist_enabled,
                                specialist_healthy=bool(
                                    specialist is not None and specialist.available
                                ),
                                specialist_frame_seq=(
                                    specialist_result.frame_seq if specialist_result else -1
                                ),
                                specialist_error=(
                                    specialist_result.error if specialist_result else None
                                ),
                            ),
                            self.hub.seats,
                            [],
                            self.hub.occupants,
                            rows,
                            counts,
                        )
                    continue
                if db_healthy:
                    try:
                        if out.seats:
                            self._repo.write_state_snapshot(out.seats)
                        write_occupants = getattr(self._repo, "write_occupant_snapshot", None)
                        if write_occupants is not None and out.occupants:
                            write_occupants(out.occupants)
                        for event in out.events:
                            self._repo.append_event(event)
                    except Exception as e:
                        log.warning("persist failed, transient-only: %s", type(e).__name__)
                        db_healthy = False
                if time.monotonic() - last_stale_check > 1.0:
                    last_stale_check = time.monotonic()
                    # Staleness is measured on the EVIDENCE clock (latest frame
                    # ts), never wall time — replay frames carry virtual times.
                    stale_events = engine.check_stale(work.meta.captured_at)
                    for event in stale_events:
                        if db_healthy:
                            try:
                                self._repo.append_event(event)
                            except Exception:
                                db_healthy = False
                    out.events.extend(stale_events)
                all_events = (out.events + all_events)[:200]
                health = source.health()
                health_writer.update(health)
                self.hub.publish(
                    self._snapshot(
                        running=True,
                        captured=captured,
                        processed=processed,
                        dropped=dropped,
                        latency=last_latency,
                        db_healthy=db_healthy,
                        model_healthy=model_healthy,
                        runtime=runtime,
                        health_status=health.status,
                        frame_seq=work.meta.seq,
                        suppressed=suppressed_total,
                        duplicates=duplicates_total,
                        pose_on=cfg.pose.enabled,
                        pose_healthy=pose_healthy,
                        dms_on=cfg.dms.enabled,
                        dms_healthy=dms_healthy,
                        driver_seat_id=driver_seat_id,
                        specialist_enabled=cfg.detection.child_specialist_enabled,
                        specialist_healthy=bool(specialist is not None and specialist.available),
                        specialist_frame_seq=(
                            specialist_result.frame_seq if specialist_result else -1
                        ),
                        specialist_error=(specialist_result.error if specialist_result else None),
                    ),
                    out.seats,
                    all_events,
                    out.occupants,
                    rows if out.seats else list(self.hub.seat_rows),
                    counts,
                )
        finally:
            health_writer.flush()
            try:
                if db_healthy:
                    self._repo.end_session(self._session_id)
            except Exception:
                pass
            try:
                source.close()
            except Exception:
                pass
            if runtime is not None:
                runtime.close()
            if specialist is not None:
                specialist.close()
            if age_estimator is not None:
                age_estimator.close()
            if face_mesh is not None:
                face_mesh.close()
            self._age_estimator = None
            self._age_cache.clear()
            self._engine = None
            # Persist the session's analytics snapshot in PLOMID so reopening
            # the same video with unchanged layout + settings reuses it
            # instead of recomputing. Best-effort: never fails shutdown.
            try:
                if processed > 0 and self._layout.seats:
                    from .seat_store import config_hash_for, save_analytics_snapshot

                    with self.hub.lock:
                        final_counts = dict(self.hub.counts)
                        final_rows = list(self.hub.seat_rows)
                    if final_counts.get("total"):
                        save_analytics_snapshot(
                            self._repo,
                            self._cfg.camera.path,
                            self._layout,
                            config_hash_for(self._cfg),
                            {
                                "session_id": self._session_id,
                                "layout_version": self._layout.version,
                                "counts": final_counts,
                                "seat_rows": [
                                    {
                                        "seat_id": r.get("seat_id"),
                                        "presence": r.get("presence"),
                                        "belt": r.get("belt"),
                                        "belt_observation": r.get("belt_observation"),
                                        "belt_observation_reason": r.get("belt_observation_reason"),
                                        "belt_observation_score": r.get("belt_observation_score"),
                                        "belt_confidence": r.get("belt_confidence"),
                                        "belt_confidence_semantics": r.get(
                                            "belt_confidence_semantics"
                                        ),
                                        "belt_status_semantics": r.get("belt_status_semantics"),
                                        "child": r.get("child"),
                                        "child_presence": r.get("child_presence"),
                                        "child_seat_presence": r.get("child_seat_presence"),
                                        "child_seat_type": r.get("child_seat_type"),
                                        "child_seat_type_candidate": r.get(
                                            "child_seat_type_candidate"
                                        ),
                                        "child_seat_occupancy": r.get("child_seat_occupancy"),
                                        "child_seat_occupancy_reason": r.get(
                                            "child_seat_occupancy_reason"
                                        ),
                                        "restraint_candidate": r.get("restraint_candidate", False),
                                        "child_seat_candidate": r.get(
                                            "child_seat_candidate", False
                                        ),
                                        "occupant_category": r.get("occupant_category"),
                                        "child_reason": r.get("child_reason"),
                                        "restraint_reason": r.get("restraint_reason"),
                                        "restraint_confidence": r.get("restraint_confidence"),
                                        "restraint_confidence_semantics": r.get(
                                            "restraint_confidence_semantics"
                                        ),
                                        "frame_seq": r.get("frame_seq"),
                                    }
                                    for r in final_rows
                                ],
                            },
                        )
            except Exception as e:
                log.warning("analytics snapshot degraded: %s", type(e).__name__)
            snap = self.hub.snapshot
            snap.running = False
            self.hub.publish(snap, self.hub.seats, self.hub.events)
            log.info("service stopped processed=%d dropped=%d", processed, dropped)


def load_demo_layout() -> SeatLayout:
    return load_layout("configs/seats/standard-5-seat-rhd.yaml")


def _encode_preview(pixels: Any) -> bytes | None:
    """One bounded JPEG for the live view. Transient; failures → None."""
    try:
        import cv2
        import numpy as np

        assert pixels is not None
        frame: np.ndarray = np.asanyarray(pixels)
        h, w = int(frame.shape[0]), int(frame.shape[1])
        scale = min(1.0, 960.0 / max(w, 1))
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ok or len(buf) > 300_000:
            return None
        return bytes(buf)
    except Exception:
        return None


def build_service(cfg: AppConfig, repository: contracts.Repository) -> Service:
    try:
        layout = load_layout(f"configs/seats/demo-{cfg.camera.camera_id}-v0.yaml")
        calibrated = True
    except Exception:
        # Uncalibrated: seats stay UNKNOWN, never guessed from wrong geometry.
        log.warning("no layout for camera %s: running UNCALIBRATED", cfg.camera.camera_id)
        layout = SeatLayout(camera_profile="uncalibrated", version="none", seats=[])
        calibrated = False
    service = Service(cfg, repository, layout)
    service.hub.snapshot.calibrated = calibrated
    return service
