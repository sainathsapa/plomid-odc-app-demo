"""MiVOLO V2 age and gender inference for tracked vehicle occupants."""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from .domain import BBox, OccupantState
from .logging import get_logger

log = get_logger("age")
MIVOLO_MODEL_ID = "mivolo-v2"
SEAT_CROP_OVERLAP_MARGIN = 0.10
_AGE_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, Any, Any]] = {}
_AGE_MODEL_CACHE_LOCK = threading.Lock()


def _seat_overlap_ratios(
    bbox: BBox, seat_polygons: dict[str, list[tuple[float, float]]]
) -> dict[str, float]:
    from .seats import _bbox_overlap_area

    area = max((bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min), 1e-9)
    return {
        seat_id: _bbox_overlap_area(bbox, polygon) / area
        for seat_id, polygon in seat_polygons.items()
    }


def _seat_has_dominant_box_overlap(
    bbox: BBox,
    seat_id: str | None,
    seat_polygons: dict[str, list[tuple[float, float]]],
    margin: float = SEAT_CROP_OVERLAP_MARGIN,
) -> bool:
    ratios = _seat_overlap_ratios(bbox, seat_polygons)
    owner_ratio = ratios.get(seat_id or "", 0.0)
    other_ratio = max(
        (ratio for candidate, ratio in ratios.items() if candidate != seat_id),
        default=0.0,
    )
    return owner_ratio > 0.0 and owner_ratio - other_ratio >= margin


@dataclass(frozen=True)
class AgeGenderPrediction:
    """Age and gender estimates for one current person track."""

    track_id: str
    seat_id: str | None
    age_years: float | None
    age_category: str
    age_class_confidence: float | None
    age_confidence_semantics: str
    gender_label: str
    gender_confidence: float | None
    gender_confidence_semantics: str
    face_visible: bool
    person_bbox: tuple[float, float, float, float] | None
    face_bbox: tuple[float, float, float, float] | None
    model_id: str
    frame_seq: int
    inference_ms: float | None
    reason: str
    age_observation_count: int = 1
    inference_bbox: tuple[float, float, float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the complete auditable per-track inference record."""
        return {
            "track_id": self.track_id,
            "seat_id": self.seat_id,
            "age_years": self.age_years,
            "age_category": self.age_category,
            "age_class_confidence": self.age_class_confidence,
            "age_confidence_semantics": self.age_confidence_semantics,
            "gender": self.gender_label,
            "gender_confidence": self.gender_confidence,
            "gender_confidence_semantics": self.gender_confidence_semantics,
            "face_visible": self.face_visible,
            "person_bbox": list(self.person_bbox) if self.person_bbox is not None else None,
            "face_bbox": list(self.face_bbox) if self.face_bbox is not None else None,
            "model_id": self.model_id,
            "frame_seq": self.frame_seq,
            "inference_ms": self.inference_ms,
            "reason": self.reason,
            "age_observation_count": self.age_observation_count,
            "inference_bbox": (
                list(self.inference_bbox) if self.inference_bbox is not None else None
            ),
        }


@dataclass
class _TrackHistory:
    seat_id: str | None
    ages: deque[float] = field(default_factory=lambda: deque(maxlen=3))
    genders: deque[tuple[str, float]] = field(default_factory=lambda: deque(maxlen=3))
    last_frame_seq: int = -1


def classify_estimated_age(
    age_years: float,
    infant_max_years: float = 2.0,
    child_max_years: float = 17.0,
    uncertainty_margin_years: float = 6.0,
) -> str:
    """Map apparent age to categories with an abstention band at adulthood.

    MiVOLO emits a regression estimate rather than calibrated class scores.
    The configurable margin leaves age estimates near the adult boundary
    unknown instead of treating a noisy estimate as a confident class.
    """
    if infant_max_years >= child_max_years:
        raise ValueError("infant_max_years must be lower than child_max_years")
    if uncertainty_margin_years < 0:
        raise ValueError("uncertainty_margin_years must be non-negative")
    if not math.isfinite(age_years) or age_years < 0:
        return "unknown"
    if age_years <= infant_max_years:
        return "infant"
    if age_years <= child_max_years - uncertainty_margin_years:
        return "child"
    if age_years > child_max_years + uncertainty_margin_years:
        return "adult"
    return "unknown"


class MiVOLOV2AgeEstimator:
    """Load one local MiVOLO V2 model and infer age/gender for person tracks.

    The checkpoint expects paired face and person tensors. A missing face is
    represented by the checkpoint's documented empty-face input while the
    actual person crop is retained. No detector score or box size is used to
    infer age.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        infant_max_years: float = 2.0,
        child_max_years: float = 17.0,
        uncertainty_margin_years: float = 6.0,
        gender_min_score: float = 0.55,
        history_max: int = 3,
        track_expiry_frames: int = 36,
    ) -> None:
        if infant_max_years >= child_max_years:
            raise ValueError("infant_max_years must be lower than child_max_years")
        if uncertainty_margin_years < 0:
            raise ValueError("uncertainty_margin_years must be non-negative")
        if not 0.0 <= gender_min_score <= 1.0:
            raise ValueError("gender_min_score must be in [0, 1]")
        self._path = Path(model_path)
        self._device_setting = device
        self._infant_max_years = infant_max_years
        self._child_max_years = child_max_years
        self._uncertainty_margin_years = uncertainty_margin_years
        self._gender_min_score = gender_min_score
        self._history_max = history_max
        self._track_expiry_frames = track_expiry_frames
        self._model: Any = None
        self._config: Any = None
        self._torch: Any = None
        self._device: Any = None
        self._history: dict[str, _TrackHistory] = {}

    @property
    def device(self) -> str:
        return str(self._device) if self._device is not None else "unloaded"

    def load(self) -> None:
        """Load local pinned source and weights once; network access is disabled."""
        if self._model is not None:
            return
        if not (self._path / "model.safetensors").is_file():
            raise FileNotFoundError(f"MiVOLO V2 checkpoint missing under {self._path}")

        vendor_root = Path(__file__).resolve().parent / "_vendor"
        if str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
        import torch

        from ._vendor.mivolo_hf.configuration_mivolo import MiVOLOConfig
        from ._vendor.mivolo_hf.modeling_mivolo import MiVOLOForImageClassification

        device = self._select_device(torch)
        cache_key = (str(self._path.resolve()), str(device))
        with _AGE_MODEL_CACHE_LOCK:
            cached = _AGE_MODEL_CACHE.get(cache_key)
            if cached is None:
                dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
                config = MiVOLOConfig.from_pretrained(str(self._path), local_files_only=True)
                model = MiVOLOForImageClassification.from_pretrained(
                    str(self._path),
                    config=config,
                    local_files_only=True,
                    torch_dtype=dtype,
                ).to(device)
                model.eval()
                cached = (torch, config, model, device)
                _AGE_MODEL_CACHE[cache_key] = cached
                log.info("loaded %s device=%s dtype=%s", MIVOLO_MODEL_ID, device, dtype)
            else:
                log.info("reusing loaded %s device=%s", MIVOLO_MODEL_ID, device)
        self._torch, self._config, self._model, self._device = cached

    def _select_device(self, torch: Any) -> Any:
        if self._device_setting != "auto":
            if self._device_setting == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("MiVOLO configured for CUDA but CUDA is unavailable")
            if self._device_setting == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MiVOLO configured for MPS but MPS is unavailable")
            return torch.device(self._device_setting)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def estimate(
        self,
        pixels: Any,
        occupants: list[OccupantState],
        faces: list[Any],
        frame_seq: int,
        layout: Any | None = None,
    ) -> dict[str, AgeGenderPrediction]:
        """Infer for tracked occupants and keep short track-local histories."""
        import numpy as np

        if self._model is None:
            raise RuntimeError("MiVOLO V2 is not loaded")
        if pixels is None:
            return {}
        image = np.asarray(pixels)
        if image.ndim != 3 or image.shape[2] != 3:
            return {}
        height, width = image.shape[:2]
        active = {
            item.track_id: item
            for item in occupants
            if item.track_id and item.seat_id is not None and item.last_bbox is not None
        }
        self._expire_history(set(active), frame_seq)
        if not active:
            return {}

        face_boxes = [box for face in faces if (box := self._face_bbox(face)) is not None]
        face_assignments = self._assign_faces(active, face_boxes)
        seat_geometries = (
            {seat.seat_id: seat for seat in layout.seats} if layout is not None else {}
        )
        seat_polygons = {seat_id: seat.polygon for seat_id, seat in seat_geometries.items()}
        face_images: list[Any] = []
        body_images: list[Any] = []
        track_ids: list[str] = []
        face_visible: dict[str, bool] = {}
        crop_boxes: dict[str, BBox] = {}
        for track_id, occupant in active.items():
            body_box = occupant.last_bbox
            if body_box is None:
                continue
            crop_box = body_box
            seat = seat_geometries.get(occupant.seat_id)
            if seat is not None and _seat_has_dominant_box_overlap(
                body_box, occupant.seat_id, seat_polygons
            ):
                from .seats import clip_bbox_to_seat_region

                crop_box = clip_bbox_to_seat_region(body_box, seat) or body_box
            body_crop = self._crop(image, crop_box, width, height, pad=0.0)
            if body_crop is None:
                continue
            face_idx = face_assignments.get(track_id)
            face_crop = None
            if face_idx is not None:
                face_box = face_boxes[face_idx]
                if seat is None or self._face_center_in_seat(face_box, seat):
                    face_crop = self._crop(image, face_box, width, height, pad=0.20)
            body_images.append(body_crop)
            face_images.append(face_crop)
            track_ids.append(track_id)
            face_visible[track_id] = face_crop is not None
            crop_boxes[track_id] = crop_box

        if not track_ids:
            return {}
        face_tensor = self._batch_tensor(face_images)
        body_tensor = self._batch_tensor(body_images)
        started = time.perf_counter()
        with self._torch.inference_mode():
            output = self._model(
                faces_input=face_tensor.to(device=self._device, dtype=self._model.dtype),
                body_input=body_tensor.to(device=self._device, dtype=self._model.dtype),
            )
        inference_ms = (time.perf_counter() - started) * 1000.0
        ages = output.age_output.detach().float().reshape(-1).cpu().tolist()
        gender_ids = output.gender_class_idx.detach().reshape(-1).cpu().tolist()
        gender_scores = output.gender_probs.detach().float().reshape(-1).cpu().tolist()
        predictions: dict[str, AgeGenderPrediction] = {}
        for index, track_id in enumerate(track_ids):
            if index >= len(ages) or index >= len(gender_ids) or index >= len(gender_scores):
                predictions[track_id] = self._unknown(
                    active[track_id], frame_seq, "missing-model-output"
                )
                continue
            age = float(ages[index])
            if not np.isfinite(age):
                predictions[track_id] = self._unknown(
                    active[track_id], frame_seq, "non-finite-age-output"
                )
                continue
            category, smoothed_age = self._record_age(
                track_id, active[track_id].seat_id, age, frame_seq
            )
            gender_id = int(gender_ids[index])
            raw_gender = str(self._config.gender_id2label.get(gender_id, "unknown"))
            gender_score = float(gender_scores[index])
            person_box = active[track_id].last_bbox
            if person_box is None:
                predictions[track_id] = self._unknown(
                    active[track_id], frame_seq, "person-box-unavailable"
                )
                continue
            gender_label = (
                "APPARENT_MAN"
                if raw_gender == "male"
                else "APPARENT_WOMAN"
                if raw_gender == "female"
                else "UNKNOWN"
            )
            if gender_score < self._gender_min_score:
                gender_label = "UNKNOWN"
            self._record_gender(track_id, gender_label, gender_score)
            gender_label, gender_score = self._smoothed_gender(track_id)
            predictions[track_id] = AgeGenderPrediction(
                track_id=track_id,
                seat_id=active[track_id].seat_id,
                age_years=round(smoothed_age, 2),
                age_category=category,
                age_class_confidence=None,
                age_confidence_semantics=(
                    "unavailable: MiVOLO V2 emits an age point estimate, not class probabilities"
                ),
                gender_label=gender_label,
                gender_confidence=round(gender_score, 4),
                gender_confidence_semantics="MiVOLO V2 gender softmax score; uncalibrated",
                face_visible=face_visible[track_id],
                person_bbox=(
                    person_box.x_min,
                    person_box.y_min,
                    person_box.x_max,
                    person_box.y_max,
                ),
                face_bbox=(
                    (
                        face_boxes[face_assignments[track_id]].x_min,
                        face_boxes[face_assignments[track_id]].y_min,
                        face_boxes[face_assignments[track_id]].x_max,
                        face_boxes[face_assignments[track_id]].y_max,
                    )
                    if track_id in face_assignments
                    else None
                ),
                model_id=MIVOLO_MODEL_ID,
                frame_seq=frame_seq,
                inference_ms=round(inference_ms, 3),
                reason="face-and-person-crops"
                if face_visible[track_id]
                else "person-only-model-input",
                age_observation_count=len(self._history[track_id].ages),
                inference_bbox=(
                    crop_boxes[track_id].x_min,
                    crop_boxes[track_id].y_min,
                    crop_boxes[track_id].x_max,
                    crop_boxes[track_id].y_max,
                ),
            )
        return predictions

    def _record_age(
        self, track_id: str, seat_id: str | None, age: float, frame_seq: int
    ) -> tuple[str, float]:
        history = self._history.get(track_id)
        if (
            history is None
            or history.seat_id != seat_id
            or frame_seq - history.last_frame_seq > self._track_expiry_frames
            or frame_seq < history.last_frame_seq
        ):
            history = _TrackHistory(
                seat_id=seat_id,
                ages=deque(maxlen=self._history_max),
                genders=deque(maxlen=self._history_max),
            )
            self._history[track_id] = history
        history.ages.append(age)
        history.last_frame_seq = frame_seq
        smoothed_age = float(median(history.ages))
        category = classify_estimated_age(
            smoothed_age,
            infant_max_years=self._infant_max_years,
            child_max_years=self._child_max_years,
            uncertainty_margin_years=self._uncertainty_margin_years,
        )
        return category, smoothed_age

    def _record_gender(self, track_id: str, gender: str, score: float) -> None:
        history = self._history[track_id]
        history.genders.append((gender, score))

    def _smoothed_gender(self, track_id: str) -> tuple[str, float]:
        history = self._history[track_id]
        known = [value for value in history.genders if value[0] != "UNKNOWN"]
        if not known:
            return "UNKNOWN", 0.0
        labels: dict[str, list[float]] = {}
        for label, score in known:
            labels.setdefault(label, []).append(score)
        label = max(labels, key=lambda key: (len(labels[key]), sum(labels[key])))
        return label, sum(labels[label]) / len(labels[label])

    def _expire_history(self, active: set[str], frame_seq: int) -> None:
        for track_id, state in list(self._history.items()):
            expired = frame_seq - state.last_frame_seq > self._track_expiry_frames
            if track_id not in active and expired:
                del self._history[track_id]

    def _batch_tensor(self, images: list[Any]) -> Any:
        import cv2
        import numpy as np

        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        output = []
        for image in images:
            if image is None:
                rgb = np.zeros((384, 384, 3), dtype=np.float32)
            else:
                source = np.asarray(image)
                height, width = source.shape[:2]
                ratio = min(384 / height, 384 / width)
                new_size = (int(round(width * ratio)), int(round(height * ratio)))
                resized = cv2.resize(source, new_size, interpolation=cv2.INTER_LINEAR)
                pad_x = (384 - new_size[0]) / 2
                pad_y = (384 - new_size[1]) / 2
                left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
                top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
                boxed = cv2.copyMakeBorder(
                    resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
                rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            normalized = (rgb - mean) / std
            output.append(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
        return self._torch.from_numpy(np.stack(output).astype(np.float32))

    @staticmethod
    def _crop(
        image: Any,
        box: BBox,
        width: int,
        height: int,
        pad: float,
    ) -> Any | None:
        import numpy as np

        box_width = max(0.0, box.x_max - box.x_min)
        box_height = max(0.0, box.y_max - box.y_min)
        x0 = max(0, int((box.x_min - pad * box_width) * width))
        y0 = max(0, int((box.y_min - pad * box_height) * height))
        x1 = min(width, int((box.x_max + pad * box_width) * width))
        y1 = min(height, int((box.y_max + pad * box_height) * height))
        crop = image[y0:y1, x0:x1]
        return np.ascontiguousarray(crop) if crop.size else None

    @staticmethod
    def _face_bbox(face: Any) -> BBox | None:
        try:
            values = face.bbox if hasattr(face, "bbox") else face
            x_min, y_min, x_max, y_max = (min(1.0, max(0.0, float(value))) for value in values)
        except (AttributeError, TypeError, ValueError):
            return None
        if x_max <= x_min or y_max <= y_min:
            return None
        return BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    @classmethod
    def _assign_faces(
        cls,
        occupants: dict[str, OccupantState],
        face_boxes: list[BBox],
    ) -> dict[str, int]:
        pairs: list[tuple[float, str, int]] = []
        for track_id, occupant in occupants.items():
            body = occupant.last_bbox
            if body is None:
                continue
            for index, face in enumerate(face_boxes):
                overlap = cls._intersection_over_face(face, body)
                if overlap >= 0.5:
                    pairs.append((overlap, track_id, index))
        pairs.sort(reverse=True)
        assigned_tracks: set[str] = set()
        assigned_faces: set[int] = set()
        result: dict[str, int] = {}
        for _score, track_id, face_index in pairs:
            if track_id in assigned_tracks or face_index in assigned_faces:
                continue
            assigned_tracks.add(track_id)
            assigned_faces.add(face_index)
            result[track_id] = face_index
        return result

    @staticmethod
    def _intersection_over_face(face: BBox, body: BBox) -> float:
        x0 = max(face.x_min, body.x_min)
        y0 = max(face.y_min, body.y_min)
        x1 = min(face.x_max, body.x_max)
        y1 = min(face.y_max, body.y_max)
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        area = max(1e-9, (face.x_max - face.x_min) * (face.y_max - face.y_min))
        return intersection / area

    @staticmethod
    def _face_center_in_seat(face: BBox, seat: Any) -> bool:
        from .seats import point_in_polygon

        center = ((face.x_min + face.x_max) / 2.0, (face.y_min + face.y_max) / 2.0)
        return point_in_polygon(*center, seat.polygon)

    @staticmethod
    def _unknown(occupant: OccupantState, frame_seq: int, reason: str) -> AgeGenderPrediction:
        return AgeGenderPrediction(
            track_id=occupant.track_id,
            seat_id=occupant.seat_id,
            age_years=None,
            age_category="unknown",
            age_class_confidence=None,
            age_confidence_semantics="unavailable",
            gender_label="UNKNOWN",
            gender_confidence=None,
            gender_confidence_semantics="unavailable",
            face_visible=False,
            person_bbox=(
                occupant.last_bbox.x_min,
                occupant.last_bbox.y_min,
                occupant.last_bbox.x_max,
                occupant.last_bbox.y_max,
            )
            if occupant.last_bbox is not None
            else None,
            face_bbox=None,
            model_id=MIVOLO_MODEL_ID,
            frame_seq=frame_seq,
            inference_ms=None,
            reason=reason,
            inference_bbox=None,
        )

    def close(self) -> None:
        """Clear per-session track history; keep shared weights resident."""
        self._history.clear()


def apply_age_predictions(
    rows: list[dict[str, Any]],
    predictions: dict[str, AgeGenderPrediction],
    layout: Any | None = None,
    child_candidate_max_years: float = 17.0,
) -> None:
    """Apply only predictions associated by current occupant track and seat."""
    from .seats import point_in_polygon

    for row in rows:
        seat_id = row.get("seat_id")
        tracks = [
            predictions[track_id]
            for track_id in row.get("track_ids", [])
            if track_id in predictions and predictions[track_id].seat_id == seat_id
        ]
        suppressed_tracks: dict[str, str] = {}
        if layout is not None:
            seat_polygons = {seat.seat_id: seat.polygon for seat in layout.seats}
            for prediction in tracks:
                if prediction.face_bbox is not None:
                    face_x = (prediction.face_bbox[0] + prediction.face_bbox[2]) / 2.0
                    face_y = (prediction.face_bbox[1] + prediction.face_bbox[3]) / 2.0
                    face_seats = {
                        candidate_seat
                        for candidate_seat, polygon in seat_polygons.items()
                        if point_in_polygon(face_x, face_y, polygon)
                    }
                    if len(face_seats) == 1 and seat_id not in face_seats:
                        suppressed_tracks[prediction.track_id] = (
                            "face-associated-with-different-seat"
                        )
                        continue
                if (
                    prediction.age_category in {"child", "infant"}
                    and prediction.person_bbox is not None
                ):
                    x0, y0, x1, y1 = prediction.person_bbox
                    person_box = BBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1)
                    ratios = _seat_overlap_ratios(person_box, seat_polygons)
                    other_ratio = max(
                        (ratio for candidate, ratio in ratios.items() if candidate != seat_id),
                        default=0.0,
                    )
                    if other_ratio >= 0.35 and not _seat_has_dominant_box_overlap(
                        person_box, seat_id, seat_polygons
                    ):
                        suppressed_tracks[prediction.track_id] = (
                            "person-box-seat-association-ambiguous"
                        )
        has_adult_track = any(prediction.age_category == "adult" for prediction in tracks)
        serialized = []
        decision_categories: dict[str, str] = {}
        for prediction in tracks:
            item = prediction.as_dict()
            if prediction.track_id in suppressed_tracks:
                item["classification_category"] = "unknown"
                item["classification_reason"] = suppressed_tracks[prediction.track_id]
                decision_categories[prediction.track_id] = "unknown"
            elif (
                has_adult_track
                and prediction.age_category in {"child", "infant"}
                and prediction.age_observation_count < 2
            ):
                item["classification_category"] = "unknown"
                item["classification_reason"] = (
                    "single-frame-child-estimate-conflicts-with-adult-track"
                )
                decision_categories[prediction.track_id] = "unknown"
            else:
                item["classification_category"] = prediction.age_category
                item["classification_reason"] = prediction.reason
                decision_categories[prediction.track_id] = prediction.age_category
            serialized.append(item)
        row["age_predictions"] = serialized
        row["age_model_id"] = MIVOLO_MODEL_ID
        row["age_class_confidence"] = None
        row["age_confidence_semantics"] = (
            "MiVOLO V2 age point estimate; no calibrated age-class probability"
        )
        row["gender"] = "UNKNOWN"
        row["gender_conf"] = None
        row["gender_confidence_semantics"] = "MiVOLO V2 softmax score; uncalibrated"
        row["age_reason"] = "no-current-track-prediction"
        if not tracks:
            row["child_candidate"] = False
            row["child_candidate_track_ids"] = []
            row["child_candidate_reason"] = "no-current-track-prediction"
            if row.get("presence") == "EMPTY_CANDIDATE":
                row["child"] = "No"
                row["occupant_category"] = "none"
                row["age_reason"] = "confirmed-empty-seat"
            else:
                row["child"] = "Unknown"
                row["occupant_category"] = "unknown"
            row["child_confidence"] = None
            row["child_confidence_semantics"] = "unavailable"
            continue

        categories = {decision_categories[prediction.track_id] for prediction in tracks}
        child_categories = categories & {"child", "infant"}
        if child_categories:
            category = (
                "mixed"
                if "adult" in categories
                else "infant"
                if "infant" in child_categories
                else "child"
            )
            child = "Yes"
            reason = (
                "mivolo-child-track-present-with-adult-track"
                if "adult" in categories
                else "mivolo-age-point-estimate"
            )
        elif categories == {"adult"}:
            category = "adult"
            child = "No"
            reason = "mivolo-age-point-estimate"
        else:
            category = "unknown"
            child = "Unknown"
            reason = "mivolo-uncertain-or-conflicting-track-estimates"
        row["occupant_category"] = category
        row["child"] = child
        row["child_presence"] = "YES" if child == "Yes" else "NO" if child == "No" else "UNKNOWN"
        row["child_confidence"] = None
        row["child_confidence_semantics"] = (
            "MiVOLO age point estimate; category is not a calibrated probability"
        )
        row["child_reason"] = reason
        candidate_track_ids = [
            prediction.track_id
            for prediction in tracks
            if prediction.track_id not in suppressed_tracks
            and prediction.age_years is not None
            and prediction.age_years <= child_candidate_max_years
        ]
        row["child_candidate"] = child == "Yes" or bool(candidate_track_ids)
        row["child_candidate_track_ids"] = candidate_track_ids
        row["child_candidate_reason"] = (
            "confirmed-child-age-category"
            if child == "Yes"
            else "apparent-age-point-estimate-within-child-boundary-unconfirmed"
            if candidate_track_ids
            else "no-current-child-range-age-evidence"
        )
        row["age_reason"] = next(iter(suppressed_tracks.values())) if suppressed_tracks else reason
        row["gender"] = (
            tracks[0].gender_label if len({p.gender_label for p in tracks}) == 1 else "UNKNOWN"
        )
        gender_scores = [p.gender_confidence for p in tracks if p.gender_confidence is not None]
        row["gender_conf"] = (
            min(gender_scores) if gender_scores and row["gender"] != "UNKNOWN" else None
        )
