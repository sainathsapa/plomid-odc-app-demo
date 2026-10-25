"""System Domain: PLOMID ODC -> bounded cabin specialist inference.

This module owns the optional local Grounding DINO stage for child-seat and
occupant candidates.  It deliberately returns candidate evidence rather than
pretending that open-vocabulary scores are calibrated probabilities.

Core Dependencies:
    Grounding DINO source tree and checkpoint already present under ``models``.
    NumPy/OpenCV-compatible frame arrays and the domain Detection contract.

Thread-Safety Profile:
    The worker owns the model and tokenizer.  The public request/poll methods
    communicate through bounded queues and are safe for one producer and one
    consumer.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .domain import BBox, Detection
from .logging import get_logger
from .vlm import cxcywh_to_frame_xyxy, ground, load_grounding_dino, preprocess_pil

log = get_logger("specialist")

SPECIALIST_MODEL_ID = "grounding-dino-swint-ogc"
SPECIALIST_PROMPT_VERSION = "gd-cabin-v1"
CHILD_SEAT_PROMPT = (
    "child car seat . infant car seat . rear-facing infant seat . "
    "forward-facing child seat . convertible car seat . booster seat . "
    "child . baby . infant . adult ."
)
ADULT_PHRASES = ("adult",)
CHILD_SEAT_PHRASES = (
    "child car seat",
    "infant car seat",
    "rear-facing infant seat",
    "forward-facing child seat",
    "convertible car seat",
    "booster seat",
)
CHILD_PHRASES = ("child", "baby", "infant")


@dataclass(frozen=True)
class SpecialistCandidate:
    """One open-vocabulary candidate with source-frame provenance."""

    label: str
    score: float
    bbox: BBox
    phrase: str
    frame_seq: int
    model_id: str = SPECIALIST_MODEL_ID
    prompt_version: str = SPECIALIST_PROMPT_VERSION


@dataclass(frozen=True)
class SpecialistResult:
    """Bounded specialist output or an explicit unavailable/error state."""

    session_id: str
    frame_seq: int
    candidates: tuple[SpecialistCandidate, ...] = ()
    status: str = "OK"
    error: str | None = None


@dataclass(frozen=True)
class _Job:
    session_id: str
    frame_seq: int
    pixels: Any


def _normalise_phrase(phrase: str) -> str:
    """Normalize model phrase text for deterministic routing."""
    return " ".join(str(phrase).lower().replace("-", " ").split())


def _candidate_label(phrase: str) -> str | None:
    """Map an open-vocabulary phrase to a bounded application label."""
    normalized = _normalise_phrase(phrase)
    normalized_seat_phrases = tuple(_normalise_phrase(item) for item in CHILD_SEAT_PHRASES)
    if any(token in normalized for token in normalized_seat_phrases):
        if "infant" in normalized or "rear facing" in normalized:
            return "infant seat"
        if "booster" in normalized:
            return "booster seat"
        return "child seat"
    if any(token in normalized for token in CHILD_PHRASES):
        return "child"
    if any(token in normalized for token in ADULT_PHRASES):
        return "adult"
    return None


def _iou(left: BBox, right: BBox) -> float:
    """Compute normalized box IoU for specialist deduplication."""
    x0 = max(left.x_min, right.x_min)
    y0 = max(left.y_min, right.y_min)
    x1 = min(left.x_max, right.x_max)
    y1 = min(left.y_max, right.y_max)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left.x_max - left.x_min) * max(0.0, left.y_max - left.y_min)
    right_area = max(0.0, right.x_max - right.x_min) * max(0.0, right.y_max - right.y_min)
    return intersection / max(left_area + right_area - intersection, 1e-9)


def deduplicate_candidates(candidates: list[SpecialistCandidate]) -> list[SpecialistCandidate]:
    """Keep the highest-scoring candidate for each physical label/box."""
    kept: list[SpecialistCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(
            candidate.label == prior.label and _iou(candidate.bbox, prior.bbox) >= 0.55
            for prior in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _infer_job(model: Any, tokenizer: Any, job: _Job, threshold: float) -> SpecialistResult:
    """Run one specialist job and convert raw model output to domain candidates."""
    # The specialist is a candidate stage; 512/800 input is materially
    # cheaper than the full 800/1333 transform while retaining cabin-scale
    # person and child-seat evidence.
    tensor, _, _ = preprocess_pil(job.pixels, shortest=512, longest=800)
    # One combined prompt performs one CPU model pass. Separate prompt passes
    # doubled latency and produced overlapping duplicate boxes.
    outputs = [
        ground(
            model,
            tokenizer,
            tensor,
            CHILD_SEAT_PROMPT,
            box_threshold=threshold,
            text_threshold=0.20,
            device="cpu",
        )
    ]
    candidates: list[SpecialistCandidate] = []
    for boxes, scores, phrases in outputs:
        for box, score, phrase in zip(boxes, scores, phrases, strict=True):
            label = _candidate_label(str(phrase))
            if label is None:
                continue
            x0, y0, x1, y1 = cxcywh_to_frame_xyxy(tuple(float(value) for value in box))
            candidates.append(
                SpecialistCandidate(
                    label=label,
                    score=float(score),
                    bbox=BBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1),
                    phrase=str(phrase),
                    frame_seq=job.frame_seq,
                )
            )
    return SpecialistResult(
        session_id=job.session_id,
        frame_seq=job.frame_seq,
        candidates=tuple(deduplicate_candidates(candidates)),
    )


class ChildSpecialist:
    """Single-slot asynchronous Grounding DINO worker.

    Raises:
        RuntimeError: If a request is made after the worker has been closed.

    Side Effects:
        Loads local model weights once and starts one daemon worker thread.

    Operational Boundaries:
        Requests are O(1) queue operations and retain at most one pending
        frame.  Slow model inference never blocks capture or the fast detector.
    """

    def __init__(
        self,
        checkpoint: str = "models/grounding-dino-swint-ogc.pth",
        config: str = "models/grounding-dino-code/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        threshold: float = 0.30,
    ) -> None:
        self._checkpoint = Path(checkpoint)
        self._config = Path(config)
        self._threshold = threshold
        self._jobs: queue.Queue[_Job | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[SpecialistResult] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="odc-child-specialist", daemon=True)
        self._started = False
        self._available = self._checkpoint.exists() and self._config.exists()

    @property
    def available(self) -> bool:
        """Return whether the local specialist resources are present."""
        return self._available

    def start(self) -> None:
        """Start the single worker; absent resources remain explicitly unavailable."""
        if self._started or not self._available:
            return
        self._started = True
        self._thread.start()

    def submit(self, session_id: str, frame_seq: int, pixels: Any) -> bool:
        """Submit only the newest frame, coalescing obsolete pending work."""
        if not self._available:
            return False
        if not self._started:
            self.start()
        job = _Job(session_id=session_id, frame_seq=frame_seq, pixels=pixels)
        try:
            self._jobs.put_nowait(job)
            return True
        except queue.Full:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                return False
            self._jobs.put_nowait(job)
            return True

    def poll(self) -> SpecialistResult | None:
        """Return the newest completed result without blocking the service."""
        newest: SpecialistResult | None = None
        while True:
            try:
                newest = self._results.get_nowait()
            except queue.Empty:
                return newest

    def _publish(self, result: SpecialistResult) -> None:
        try:
            self._results.put_nowait(result)
        except queue.Full:
            try:
                self._results.get_nowait()
            except queue.Empty:
                pass
            self._results.put_nowait(result)

    def _run(self) -> None:
        model: Any = None
        tokenizer: Any = None
        try:
            model, tokenizer = load_grounding_dino(
                str(self._config), str(self._checkpoint), device="cpu"
            )
        except Exception as error:
            self._available = False
            log.warning("child specialist unavailable: %s: %s", type(error).__name__, error)
            return
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                return
            try:
                self._publish(_infer_job(model, tokenizer, job, self._threshold))
            except Exception as error:
                log.warning("child specialist inference failed: %s", type(error).__name__)
                self._publish(
                    SpecialistResult(
                        session_id=job.session_id,
                        frame_seq=job.frame_seq,
                        status="ERROR",
                        error=f"{type(error).__name__}: {error}",
                    )
                )

    def close(self) -> None:
        """Stop the worker and release its bounded queues."""
        self._stop.set()
        if self._started:
            try:
                self._jobs.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=2.0)


def specialist_as_detections(result: SpecialistResult) -> list[Detection]:
    """Expose specialist candidates as namespaced domain detections."""
    return [
        Detection(
            class_label=candidate.label,
            score=candidate.score,
            bbox=candidate.bbox,
            evidence_kind="grounding-dino-candidate",
            model_id=candidate.model_id,
            frame_seq=candidate.frame_seq,
        )
        for candidate in result.candidates
    ]
