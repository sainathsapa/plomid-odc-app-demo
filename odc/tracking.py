"""ByteTrack-style multi-object tracker (Phase 5) — numpy only, no weights.

BYTE association idea: match high-confidence detections first, then use
low-confidence boxes to keep tracks alive through partial occlusion.
Deviation from upstream ByteTrack (documented, benchmarked): greedy IoU
matching instead of Hungarian — exact for sparse cabin scenes, O(n*m),
revisited only if benchmarks show fragmentation. Upstream:
FoundationVision/ByteTrack (ex-`ifzhang`), ECCV'22, MIT — association only,
no weights, so no checkpoint license applies. OC-SORT stays the challenger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC

import numpy as np

from . import contracts
from .domain import BBox, Detection, OccupantState
from .logging import get_logger

log = get_logger("tracking")


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5
    low_thresh: float = 0.2
    match_iou: float = 0.3
    max_age: int = 30
    min_hits: int = 3


def iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a.x_min, b.x_min), max(a.y_min, b.y_min)
    ix1, iy1 = min(a.x_max, b.x_max), min(a.y_max, b.y_max)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _to_xywh(bbox: BBox) -> np.ndarray:
    cx = (bbox.x_min + bbox.x_max) / 2
    cy = (bbox.y_min + bbox.y_max) / 2
    return np.array([cx, cy, bbox.x_max - bbox.x_min, bbox.y_max - bbox.y_min])


def _to_bbox(state: np.ndarray) -> BBox:
    cx, cy, w, h = (float(v) for v in state[:4])
    return BBox(
        x_min=min(max(cx - w / 2, 0.0), 1.0),
        y_min=min(max(cy - h / 2, 0.0), 1.0),
        x_max=min(max(cx + w / 2, 0.0), 1.0),
        y_max=min(max(cy + h / 2, 0.0), 1.0),
    )


class _Kalman:
    """Constant-velocity filter over [cx, cy, w, h, vx, vy, vw, vh]."""

    def __init__(self, initial: np.ndarray) -> None:
        self.x = np.zeros(8)
        self.x[:4] = initial
        self.p = np.eye(8) * 1.0
        self.p[4:, 4:] *= 10.0

    def predict(self) -> np.ndarray:
        f = np.eye(8)
        f[:4, 4:] = np.eye(4)
        self.x = f @ self.x
        q = np.eye(8) * 0.01
        q[4:, 4:] *= 1.0
        self.p = f @ self.p @ f.T + q
        return self.x[:4].copy()

    def update(self, z: np.ndarray) -> None:
        h = np.zeros((4, 8))
        h[:, :4] = np.eye(4)
        r = np.eye(4) * 0.1
        y = z - h @ self.x
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.p = (np.eye(8) - k @ h) @ self.p


@dataclass
class _Track:
    track_id: int
    kf: _Kalman
    bbox: BBox
    score: float
    label: str
    hits: int = 1
    age: int = 0
    confirmed: bool = False
    matched: bool = False


def _greedy_match(
    tracks: list[_Track], dets: list[Detection], iou_thresh: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    pairs: list[tuple[float, int, int]] = []
    for ti, track in enumerate(tracks):
        for di, det in enumerate(dets):
            if track.label != det.class_label:
                continue
            score = iou(track.bbox, det.bbox)
            if score >= iou_thresh:
                pairs.append((score, ti, di))
    pairs.sort(reverse=True)
    used_t, used_d = set(), set()
    matched: list[tuple[int, int]] = []
    for _, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matched.append((ti, di))
    unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
    unmatched_d = [i for i in range(len(dets)) if i not in used_d]
    return matched, unmatched_t, unmatched_d


class ByteTrackStyleTracker(contracts.Tracker):
    """Session-local tracks. IDs reset on session reset — never identity."""

    def __init__(self, cfg: TrackerConfig | None = None) -> None:
        self._cfg = cfg or TrackerConfig()
        self._tracks: list[_Track] = []
        self._next_id = 1
        self.switches = 0
        self.fragments = 0

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def _spawn(self, det: Detection) -> _Track:
        track = _Track(
            track_id=self._next_id,
            kf=_Kalman(_to_xywh(det.bbox)),
            bbox=det.bbox,
            score=det.score,
            label=det.class_label,
        )
        self._next_id += 1
        return track

    def update(self, frame: contracts.Frame, detections: list[Detection]) -> list[OccupantState]:
        from datetime import datetime

        for track in self._tracks:
            track.bbox = _to_bbox(track.kf.predict())
            track.age += 1
            track.matched = False

        high = [d for d in detections if d.score >= self._cfg.high_thresh]
        low = [d for d in detections if self._cfg.low_thresh <= d.score < self._cfg.high_thresh]

        matched, unmatched_t, unmatched_d = _greedy_match(self._tracks, high, self._cfg.match_iou)
        for ti, di in matched:
            track, det = self._tracks[ti], high[di]
            track.kf.update(_to_xywh(det.bbox))
            track.bbox, track.score, track.label = det.bbox, det.score, det.class_label
            track.age, track.hits = 0, track.hits + 1
            track.matched = True
            if track.hits >= self._cfg.min_hits:
                track.confirmed = True

        alive = [self._tracks[ti] for ti, _ in matched]
        lost = [self._tracks[ti] for ti in unmatched_t]
        if lost and low:
            matched2, _, _ = _greedy_match(lost, low, self._cfg.match_iou * 0.7)
            re_ids = {ti for ti, _ in matched2}
            for li, track in enumerate(lost):
                if li in re_ids:
                    di = next(di for ti, di in matched2 if ti == li)
                    det = low[di]
                    track.kf.update(_to_xywh(det.bbox))
                    track.bbox, track.score = det.bbox, det.score
                    track.age = 0
                    track.matched = True
                    alive.append(track)
                else:
                    alive.append(track)
        else:
            alive.extend(lost)

        for di in unmatched_d:
            det = high[di]
            if det.class_label == "person":
                spawned = self._spawn(det)
                spawned.matched = True
                spawned.confirmed = spawned.hits >= self._cfg.min_hits
                alive.append(spawned)

        kept: list[_Track] = []
        for track in alive:
            if track.age > self._cfg.max_age:
                if track.confirmed:
                    self.fragments += 1
                continue
            kept.append(track)
        self._tracks = kept

        now = datetime.now(UTC)
        states: list[OccupantState] = []
        for track in self._tracks:
            # Emit only tracks with evidence THIS frame; coasted tracks stay
            # internal for re-acquire, and the state engine bridges gaps.
            if not track.confirmed or not track.matched:
                continue
            states.append(
                OccupantState(
                    session_id="",
                    track_id=f"t{track.track_id}",
                    last_bbox=track.bbox,
                    last_score=track.score,
                    presence=None,
                    updated_at=now,
                )
            )
        return states

    @property
    def live_tracks(self) -> int:
        return sum(1 for t in self._tracks if t.confirmed)
