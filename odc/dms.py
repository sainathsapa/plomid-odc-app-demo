"""Driver monitoring previews (candidate-grade, never safety claims).

Face landmarks (MediaPipe, local weights) provide three supporting
signals: eye-blink ratio over time (drowsiness proxy), jaw opening
(yawn proxy) and head yaw/pitch (distraction proxy), plus phone-near-body
from the generic detector. Every output is a CANDIDATE with its evidence
attached — drowsy/distracted/phone flags feed alerts and the seat table,
never protection logic. No face identity is computed or stored; only
transient per-frame points leave this module.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .logging import get_logger

log = get_logger("dms")
_FACE_LANDMARKERS: dict[tuple[str, int], Any] = {}
_FACE_LANDMARKERS_LOCK = threading.Lock()

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH_CORNERS = (61, 291)
EYE_CORNERS = (33, 133, 362, 263)

BLINK_CLOSED = 0.55
JAW_OPEN = 0.60
DROWSY_RATIO = 0.35
DROWSY_WINDOW = 24
DROWSY_MIN_OBS = 8
YAWN_FRAMES = 4
YAW_DEG = 35.0
PITCH_DOWN_DEG = -20.0
DISTRACT_FRAMES = 5


def ear(eye6: list[tuple[float, float]]) -> float:
    """Eye aspect ratio from 6 ordered eye points (p1 outer … p4 inner)."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6) = eye6
    vert = math.hypot(x2 - x6, y2 - y6) + math.hypot(x3 - x5, y3 - y5)
    horiz = max(math.hypot(x1 - x4, y1 - y4), 1e-9)
    return vert / (2.0 * horiz)


def yaw_pitch_from_matrix(data: list[float]) -> tuple[float | None, float | None]:
    """Row-major 4x4 face matrix -> (yaw, pitch) degrees. None on garbage."""
    if len(data) < 16:
        return None, None
    try:
        yaw = math.degrees(math.atan2(data[2], data[10]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -data[6]))))
        if not (math.isfinite(yaw) and math.isfinite(pitch)):
            return None, None
        return yaw, pitch
    except Exception:
        return None, None


def phone_near_body(phones: list[Any], boxes: list[Any]) -> bool:
    """A cell-phone center inside the lower two-thirds of any occupant box
    (hand zone, not a phone lying on the seat beside them). Pure."""
    for phone in phones:
        bbox = phone.bbox
        cx = (bbox.x_min + bbox.x_max) / 2.0
        cy = (bbox.y_min + bbox.y_max) / 2.0
        for box in boxes:
            if (
                box.x_min <= cx <= box.x_max
                and (box.y_min + (box.y_max - box.y_min) / 3.0) <= cy <= box.y_max
            ):
                return True
    return False


@dataclass
class FaceResult:
    bbox: tuple[float, float, float, float]  # frame-normalized xyxy
    blink: float  # avg eye-blink blendshape 0..1
    jaw: float  # jawOpen blendshape 0..1
    ear_value: float | None
    yaw: float | None
    pitch: float | None
    dots: dict[str, tuple[float, float]]  # overlay points (eyes/mouth)
    closed: bool = False


@dataclass
class DmsState:
    drowsy: bool = False
    distracted: bool = False
    phone: bool = False
    yawning: bool = False
    ear: float | None = None
    jaw: float = 0.0
    yaw: float | None = None
    pitch: float | None = None
    face: bool = False
    reason: str = "no-face"

    def to_dict(self) -> dict[str, Any]:
        return {
            "drowsy": self.drowsy,
            "distracted": self.distracted,
            "phone": self.phone,
            "yawning": self.yawning,
            "ear": self.ear,
            "jaw": round(self.jaw, 3),
            "yaw": round(self.yaw, 1) if self.yaw is not None else None,
            "pitch": round(self.pitch, 1) if self.pitch is not None else None,
            "face": self.face,
            "reason": self.reason,
        }


class FaceMesh:
    """Lazy face-landmarker wrapper. Heavy deps import on use, never at import."""

    def __init__(self, model_path: str, num_faces: int = 2) -> None:
        self._path = model_path
        self._num_faces = num_faces
        self._landmarker: Any = None

    def load(self) -> None:
        cache_key = (str(self._path), self._num_faces)
        with _FACE_LANDMARKERS_LOCK:
            cached = _FACE_LANDMARKERS.get(cache_key)
            if cached is not None:
                self._landmarker = cached
                log.info("reusing face landmarker (%s)", self._path)
                return
            import mediapipe as mp  # type: ignore[import-untyped]

            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self._path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=self._num_faces,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            _FACE_LANDMARKERS[cache_key] = self._landmarker
        log.info("face mesh loaded (%s)", self._path)

    def _blend(self, categories: Any, name: str) -> float:
        for cat in categories:
            if cat.category_name == name:
                return float(cat.score)
        return 0.0

    def estimate(self, frame: Any) -> list[FaceResult]:
        """Full-frame faces with blendshape state. Never raises."""
        import cv2
        import mediapipe as mp
        import numpy as np

        if self._landmarker is None:
            return []
        try:
            pixels: np.ndarray = np.asanyarray(frame.pixels)
            rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = self._landmarker.detect(image)
            out: list[FaceResult] = []
            for fi, landmarks in enumerate(result.face_landmarks):
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                bbox = (min(xs), min(ys), max(xs), max(ys))
                cats = result.face_blendshapes[fi] if fi < len(result.face_blendshapes) else []
                blink = (
                    self._blend(cats, "eyeBlinkLeft") + self._blend(cats, "eyeBlinkRight")
                ) / 2.0
                jaw = self._blend(cats, "jawOpen")
                try:
                    left = [(landmarks[i].x, landmarks[i].y) for i in LEFT_EYE]
                    right = [(landmarks[i].x, landmarks[i].y) for i in RIGHT_EYE]
                    ear_value = (ear(left) + ear(right)) / 2.0
                except Exception:
                    ear_value = None
                yaw = pitch = None
                try:
                    if result.facial_transformation_matrixes and fi < len(
                        result.facial_transformation_matrixes
                    ):
                        import numpy as _np

                        matrix = result.facial_transformation_matrixes[fi]
                        data = [float(v) for v in _np.asarray(matrix).reshape(-1)]
                        yaw, pitch = yaw_pitch_from_matrix(data)
                except Exception:
                    yaw = pitch = None
                dots = {}
                for label, idx in (
                    ("eye_l", 33),
                    ("eye_l_in", 133),
                    ("eye_r", 362),
                    ("eye_r_in", 263),
                    ("mouth_l", 61),
                    ("mouth_r", 291),
                ):
                    try:
                        dots[label] = (float(landmarks[idx].x), float(landmarks[idx].y))
                    except Exception:
                        pass
                out.append(
                    FaceResult(
                        bbox=bbox,
                        blink=blink,
                        jaw=jaw,
                        ear_value=ear_value,
                        yaw=yaw,
                        pitch=pitch,
                        dots=dots,
                        closed=blink >= BLINK_CLOSED,
                    )
                )
            return out
        except Exception as e:
            log.warning("face mesh degraded: %s", type(e).__name__)
            return []

    def close(self) -> None:
        """Keep the shared native model resident for the next session."""


@dataclass
class DmsTracker:
    """Per-seat temporal DMS votes. Fresh instance per video."""

    closed_window: int = DROWSY_WINDOW
    _closed: dict[str, deque[bool]] = field(default_factory=dict)
    _jaw_runs: dict[str, int] = field(default_factory=dict)
    _pose_runs: dict[str, int] = field(default_factory=dict)

    def update(
        self,
        seat_id: str,
        occupied: bool,
        face: FaceResult | None,
        phone_near: bool = False,
    ) -> DmsState:
        if not occupied or face is None:
            self._jaw_runs.pop(seat_id, None)
            self._pose_runs.pop(seat_id, None)
            return DmsState(face=False, reason="no-face")
        window = self._closed.setdefault(seat_id, deque(maxlen=self.closed_window))
        window.append(face.closed)
        counted = list(window)
        ratio = sum(1 for c in counted if c) / len(counted) if counted else 0.0
        drowsy = len(counted) >= DROWSY_MIN_OBS and ratio >= DROWSY_RATIO
        jaw_run = self._jaw_runs.get(seat_id, 0)
        jaw_run = jaw_run + 1 if face.jaw >= JAW_OPEN else 0
        self._jaw_runs[seat_id] = jaw_run
        yawning = jaw_run >= YAWN_FRAMES
        turned = (face.yaw is not None and abs(face.yaw) >= YAW_DEG) or (
            face.pitch is not None and face.pitch <= PITCH_DOWN_DEG
        )
        pose_run = self._pose_runs.get(seat_id, 0)
        pose_run = pose_run + 1 if turned else 0
        self._pose_runs[seat_id] = pose_run
        distracted = pose_run >= DISTRACT_FRAMES
        return DmsState(
            drowsy=drowsy,
            distracted=distracted,
            phone=bool(phone_near),
            yawning=yawning,
            ear=face.ear_value,
            jaw=face.jaw,
            yaw=face.yaw,
            pitch=face.pitch,
            face=True,
            reason=(
                f"blink-ratio {ratio:.2f} jaw {face.jaw:.2f} yaw {face.yaw} pitch {face.pitch}"
            ),
        )
