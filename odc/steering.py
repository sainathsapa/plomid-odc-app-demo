"""Candidate steering-side proposal from visible steering-wheel geometry.

This is an engineering MVP cue, not a vehicle-qualified classifier. It uses
OpenCV circle evidence in the lower cabin and temporal agreement across a
replay. A dedicated labelled steering-wheel model remains the production
upgrade. Ambiguous, mirrored, rear-facing, or occluded views return UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class OrientationProposal:
    orientation: str = "UNKNOWN"
    confidence: float = 0.0
    frames_checked: int = 0
    positive_frames: int = 0
    reason: str = "no consistent steering-wheel cue"
    mirror_status: str = "UNKNOWN"


def detect_steering_side(frame: Any) -> tuple[str, float]:
    """Return ``(LHD|RHD|UNKNOWN, score)`` for one unmirrored frame.

    The image-side mapping assumes a forward-facing, non-mirrored cabin view:
    wheel on image-left suggests LHD and image-right suggests RHD. The caller
    must aggregate this cue and expose it for operator confirmation.
    """
    if frame is None:
        return "UNKNOWN", 0.0
    pixels = np.asanyarray(frame)
    if pixels.ndim < 2 or pixels.shape[0] < 32 or pixels.shape[1] < 32:
        return "UNKNOWN", 0.0
    height, width = pixels.shape[:2]
    scale = min(1.0, 720.0 / max(height, width))
    small = cv2.resize(pixels, None, fx=scale, fy=scale) if scale < 1.0 else pixels
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape[:2]
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(18, int(w * 0.12)),
        param1=100,
        param2=28,
        minRadius=max(10, int(w * 0.045)),
        maxRadius=max(12, int(w * 0.25)),
    )
    candidates: list[tuple[float, str]] = []
    if circles is None:
        return "UNKNOWN", 0.0
    for cx, cy, radius in np.round(circles[0]).astype(int):
        # Steering wheels normally occupy the lower-middle cabin; exclude
        # mirrors, faces, and upper-window circles.
        if cy < int(h * 0.38) or cy > int(h * 0.96):
            continue
        if cx < int(w * 0.08) or cx > int(w * 0.92):
            continue
        side = "LHD" if cx < w / 2.0 else "RHD"
        radius_score = min(1.0, radius / max(w * 0.12, 1.0))
        lower_score = min(1.0, max(0.0, (cy / h - 0.38) / 0.58))
        candidates.append((0.55 * radius_score + 0.45 * lower_score, side))
    if not candidates:
        return "UNKNOWN", 0.0
    candidates.sort(reverse=True)
    return candidates[0][1], min(1.0, candidates[0][0])


def aggregate_orientation_votes(votes: list[tuple[str, float]]) -> OrientationProposal:
    """Require repeated, dominant side evidence before proposing a side."""
    valid = [(side, score) for side, score in votes if side in ("LHD", "RHD")]
    if not valid:
        return OrientationProposal(frames_checked=len(votes))
    totals = {side: sum(score for candidate, score in valid if candidate == side) for side in ("LHD", "RHD")}
    winner = max(totals, key=totals.get)
    loser = totals["RHD" if winner == "LHD" else "LHD"]
    winner_votes = [score for side, score in valid if side == winner]
    share = len(winner_votes) / len(valid)
    confidence = min(1.0, (sum(winner_votes) / len(winner_votes)) * share)
    if len(winner_votes) < 2 or share < 0.70 or totals[winner] <= loser * 1.35:
        return OrientationProposal(
            frames_checked=len(votes), positive_frames=len(valid),
            reason="steering cue ambiguous or inconsistent; operator confirmation required",
        )
    return OrientationProposal(
        orientation=winner,
        confidence=float(round(float(confidence), 3)),
        frames_checked=len(votes),
        positive_frames=len(valid),
        reason=f"consistent lower-cabin wheel cue on image-{winner.lower()} side; confirm camera is not mirrored",
    )


def propose_orientation(path: str | Path, max_frames: int = 16) -> OrientationProposal:
    """Probe a replay for repeated wheel-side evidence without raising."""
    source = Path(path)
    if not source.exists():
        return OrientationProposal(reason="missing video")
    cap = cv2.VideoCapture(str(source))
    votes: list[tuple[str, float]] = []
    checked = 0
    try:
        if not cap.isOpened():
            return OrientationProposal(reason="unreadable video")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        stride = max(1, total // max(1, max_frames)) if total else 1
        index = 0
        while checked < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if index % stride == 0:
                checked += 1
                votes.append(detect_steering_side(frame))
            index += 1
    finally:
        cap.release()
    result = aggregate_orientation_votes(votes)
    return OrientationProposal(
        orientation=result.orientation,
        confidence=result.confidence,
        frames_checked=checked,
        positive_frames=result.positive_frames,
        reason=result.reason,
        mirror_status="UNKNOWN",
    )
