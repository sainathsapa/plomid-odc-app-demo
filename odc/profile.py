"""Stage profiler (Phase 10) — measure before optimizing, prove parity after.

Profiles: decode → preprocess+infer → track → associate → engine → persist.
Reports per-stage P50/P95/mean + share. No optimization without a profile.
"""

from __future__ import annotations

import time
from typing import Any


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(pct / 100 * len(ordered)), len(ordered) - 1)]


class StageProfiler:
    def __init__(self) -> None:
        self.stages: dict[str, list[float]] = {}

    def time(self, stage: str, fn: Any, *args: Any) -> Any:
        start = time.monotonic()
        try:
            return fn(*args)
        finally:
            self.stages.setdefault(stage, []).append((time.monotonic() - start) * 1000)

    def report(self) -> dict[str, Any]:
        total_p50 = 0.0
        rows = {}
        for stage, values in self.stages.items():
            p50, p95 = _pct(values, 50), _pct(values, 95)
            mean = sum(values) / len(values)
            rows[stage] = {
                "n": len(values),
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "mean_ms": round(mean, 1),
            }
            total_p50 += p50
        for row in rows.values():
            row["share_pct"] = round(100 * row["p50_ms"] / total_p50, 1) if total_p50 else 0
        ordered = sorted(rows.items(), key=lambda kv: kv[1]["p50_ms"], reverse=True)
        return {
            "stages": dict(ordered),
            "bottleneck": ordered[0][0] if ordered else None,
            "total_p50_ms": round(total_p50, 1),
        }


def profile_clip(
    clip: str,
    max_frames: int,
    build_runtime: Any,
    threshold: float = 0.3,
    persist: Any = None,
) -> dict[str, Any]:
    """Full-chain profile over a clip. `persist(seats, events)` optional hook."""
    from .capture import FileReplaySource
    from .config import CameraConfig
    from .seats import associate_track_first, load_layout
    from .state import SeatStateEngine, StateRules
    from .tracking import ByteTrackStyleTracker

    profiler = StageProfiler()
    runtime = build_runtime()
    tracker = ByteTrackStyleTracker()
    layout = load_layout("configs/seats/demo-f1-v0.yaml")
    engine = SeatStateEngine("ses-prof", layout, StateRules())
    source = FileReplaySource(CameraConfig(path=clip))
    source.open()
    n = 0
    try:
        while n < max_frames:
            frame = profiler.time("decode", source.next)
            if frame is None or frame.pixels is None:
                break
            result = profiler.time("infer", runtime.infer, frame)
            persons = [d for d in result.detections if d.class_label == "person"]
            states = profiler.time("track", tracker.update, frame, persons)
            pairs = [(s.track_id, s.last_bbox) for s in states if s.last_bbox]
            assoc = profiler.time("associate", associate_track_first, pairs, layout)
            _ = assoc
            out = profiler.time("engine", engine.update, frame, states, layout.seats)
            if persist is not None and out.seats:
                profiler.time("persist", persist, out.seats, out.events)
            n += 1
    finally:
        source.close()
        runtime.close()
    report = profiler.report()
    report["frames"] = n
    return report
