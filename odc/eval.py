"""Detector evaluation harness (Phase 4) — observational reports.

No ground truth here: per-class counts, score distributions, latency
P50/P95, throughput, cold start. Labeled-metric evaluation lands in
Phase 9. Every report is stamped OBSERVATIONAL and carries the full
model manifest + hardware profile, so no number can be mistaken for a
product accuracy claim.
"""

from __future__ import annotations

import platform
import time
from collections import Counter
from typing import Any

from . import contracts
from .config import AppConfig
from .domain import ModelManifest


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(pct / 100 * len(ordered)), len(ordered) - 1)
    return ordered[idx]


def run_detector_eval(
    source: contracts.Source,
    runtime: contracts.DetectorRuntime,
    manifest: ModelManifest,
    cfg: AppConfig,
    max_frames: int | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    if threshold is not None and hasattr(runtime, "_threshold"):
        runtime._threshold = threshold
    source.open()
    latencies: list[float] = []
    counts: Counter[str] = Counter()
    scores: list[float] = []
    n = 0
    cold_start_ms: float | None = None
    try:
        while max_frames is None or n < max_frames:
            frame = source.next()
            if frame is None:
                break
            if frame.pixels is None:
                continue
            if n % cfg.processing.sample_every_n != 0:
                n += 1
                continue
            result = runtime.infer(frame)
            if cold_start_ms is None:
                cold_start_ms = result.duration_ms
            latencies.append(result.duration_ms)
            for det in result.detections:
                counts[det.class_label] += 1
                scores.append(det.score)
            n += 1
    finally:
        source.close()
    provider = getattr(runtime, "providers_active", ["unknown"])[0]
    return {
        "kind": "OBSERVATIONAL — no ground truth, not an accuracy claim",
        "model_id": manifest.model_id,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "threshold": threshold if threshold is not None else cfg.detection.threshold,
        "provider": provider,
        "hardware": {"platform": platform.platform(), "machine": platform.machine()},
        "frames_seen": n,
        "frames_inferred": len(latencies),
        "per_class_counts": dict(counts),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "cold_start_ms": cold_start_ms,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "timestamp": time.time(),
    }
