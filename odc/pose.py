"""Pose estimation adapters (Phase 5) — optional capability.

MediaPipe Pose Landmarker Lite is the working fallback (REVIEW_REQUIRED for
commercial use — bundle terms unverified). RTMPose stays preferred but has
no trustworthy official ONNX artifact (third-party ports only, lineage
unverified). The app runs identically with NullPose.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from . import contracts
from .domain import BBox, Detection, Keypoint, OccupantState
from .logging import get_logger
from .tracking import iou

log = get_logger("pose")
_POSE_LANDMARKERS: dict[tuple[str, float], tuple[Any, Any]] = {}
_POSE_LANDMARKERS_LOCK = threading.Lock()


def associate_pose_detections(
    detections: Sequence[Detection],
    occupants: Sequence[OccupantState],
    minimum_iou: float = 0.1,
) -> list[tuple[Detection, OccupantState]]:
    """Pair person detections with their active seat-associated tracks.

    Non-person COCO detections and unmatched person boxes are excluded. When
    duplicate boxes map to one track, retain the detection with greatest IoU.
    """
    if not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("minimum_iou must be in [0, 1]")
    tracked = [
        occupant
        for occupant in occupants
        if occupant.seat_id is not None and occupant.last_bbox is not None
    ]
    best_by_track: dict[str, tuple[float, Detection, OccupantState]] = {}
    for detection in detections:
        if detection.class_label != "person":
            continue
        matches = [
            (iou(detection.bbox, occupant.last_bbox), occupant)
            for occupant in tracked
            if occupant.last_bbox is not None
        ]
        if not matches:
            continue
        overlap, occupant = max(matches, key=lambda item: item[0])
        if overlap < minimum_iou:
            continue
        prior = best_by_track.get(occupant.track_id)
        if prior is None or overlap > prior[0]:
            best_by_track[occupant.track_id] = (overlap, detection, occupant)
    return [
        (detection, occupant)
        for _overlap, detection, occupant in sorted(
            best_by_track.values(), key=lambda item: item[2].track_id
        )
    ]


def classify_seated_posture(
    points: Sequence[Sequence[float]], frame_width: int, frame_height: int
) -> tuple[str, str, bool | None]:
    """Return a posture evidence state without an uncalibrated seat reference.

    MediaPipe pose landmarks do not identify the seat-back axis. A torso that
    appears vertical in the image may still be reclined in an infant carrier,
    so visible keypoints alone cannot support an upright/reclined label.
    """
    if frame_width <= 0 or frame_height <= 0 or len(points) < 25:
        return "UNKNOWN", "insufficient-pose-landmarks", None
    required = (11, 12, 23, 24)
    if any(len(points[index]) < 3 or float(points[index][2]) < 0.5 for index in required):
        return "BODY_PART_OCCLUDED", "shoulder-or-hip-landmark-not-visible", False

    return "UNKNOWN", "seat-relative-posture-reference-unavailable", True


def map_landmarks(
    landmarks_rel: list[tuple[float, float, float]],
    crop: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    visibility_thresh: float = 0.5,
) -> list[Keypoint]:
    """Crop-relative (x, y, visibility) → frame-normalized Keypoints."""
    x0, y0, x1, y1 = crop
    cw, ch = max(x1 - x0, 1), max(y1 - y0, 1)
    out: list[Keypoint] = []
    for lx, ly, vis in landmarks_rel:
        out.append(
            Keypoint(
                x=min(max((x0 + lx * cw) / frame_w, 0.0), 1.0),
                y=min(max((y0 + ly * ch) / frame_h, 0.0), 1.0),
                confidence=min(max(float(vis), 0.0), 1.0),
                visible=bool(vis >= visibility_thresh),
            )
        )
    return out


class NullPose(contracts.PoseEstimator):
    """No pose model — pipeline runs, keypoint evidence stays absent."""

    def estimate(self, frame: contracts.Frame, boxes: list[BBox]) -> dict[int, list[Keypoint]]:
        return {}

    def close(self) -> None:
        pass


class MediaPipePose(contracts.PoseEstimator):
    def __init__(self, model_path: str, visibility_thresh: float = 0.5) -> None:
        cache_key = (str(model_path), visibility_thresh)
        with _POSE_LANDMARKERS_LOCK:
            cached = _POSE_LANDMARKERS.get(cache_key)
            if cached is not None:
                self._mp, self._landmarker = cached
                self._thresh = visibility_thresh
                self.calls = 0
                self.empty = 0
                log.info("reusing pose landmarker (%s)", model_path)
                return
            import mediapipe as mp  # type: ignore[import-untyped]

            self._mp = mp
            self._thresh = visibility_thresh
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
            )
            self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
            _POSE_LANDMARKERS[cache_key] = (self._mp, self._landmarker)
        self.calls = 0
        self.empty = 0

    def estimate(self, frame: contracts.Frame, boxes: list[BBox]) -> dict[int, list[Keypoint]]:
        import cv2
        import numpy as np

        mp = self._mp
        out: dict[int, list[Keypoint]] = {}
        pixels = frame.pixels
        if pixels is None:
            return out
        try:
            h, w = int(pixels.shape[0]), int(pixels.shape[1])
            for idx, box in enumerate(boxes):
                pad = 0.1
                bw, bh = box.x_max - box.x_min, box.y_max - box.y_min
                x0 = max(int((box.x_min - pad * bw) * w), 0)
                y0 = max(int((box.y_min - pad * bh) * h), 0)
                x1 = min(int((box.x_max + pad * bw) * w), w)
                y1 = min(int((box.y_max + pad * bh) * h), h)
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = pixels[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                self.calls += 1
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2RGB),
                )
                result = self._landmarker.detect(image)
                if not result.pose_landmarks:
                    self.empty += 1
                    continue
                rel = [(lm.x, lm.y, lm.visibility) for lm in result.pose_landmarks[0]]
                out[idx] = map_landmarks(rel, (x0, y0, x1, y1), w, h, self._thresh)
        except Exception as e:
            log.warning("pose estimate degraded: %s: %s", type(e).__name__, e)
        return out

    def close(self) -> None:
        """Keep the shared native model resident for the next session."""
