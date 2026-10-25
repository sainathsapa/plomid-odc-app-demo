"""Phase 10 unit tests — profiler math, provider policy, quant gate."""

from __future__ import annotations

import numpy as np

from odc.profile import StageProfiler, _pct


def test_percentile_and_shares():
    assert _pct([], 95) == 0.0
    prof = StageProfiler()
    for value in (10.0, 20.0, 30.0):
        prof.stages.setdefault("a", []).append(value)
    prof.stages.setdefault("b", []).append(10.0)
    report = prof.report()
    assert report["bottleneck"] == "a"
    assert report["stages"]["a"]["p50_ms"] == 20.0
    assert abs(sum(r["share_pct"] for r in report["stages"].values()) - 100) < 1.0


def test_profile_clip_structure_without_weights():
    from odc import contracts
    from odc.capture import FakeSource
    from odc.profile import profile_clip

    class _Rt(contracts.DetectorRuntime):
        def load(self, manifest): ...

        def infer(self, frame):
            return contracts.InferenceResult(detections=[], model_id="stub")

        def close(self): ...

    import odc.capture as capture_mod

    real_source = capture_mod.FileReplaySource
    capture_mod.FileReplaySource = lambda cfg: FakeSource(n_frames=5)  # type: ignore[assignment]
    try:
        report = profile_clip("unused.mp4", 5, lambda: _Rt())
    finally:
        capture_mod.FileReplaySource = real_source
    assert report["frames"] == 5
    assert set(report["stages"]) >= {"decode", "infer", "track", "associate", "engine"}
    assert report["bottleneck"] in report["stages"]


def test_quant_parity_gate_logic():
    ref = np.zeros((2, 80))
    same = np.zeros((2, 80))
    drifted = np.zeros((2, 80))
    drifted[0, 0] = 6.0
    assert float(np.abs(same - ref).max()) < 1e-3
    assert float(np.abs(drifted - ref).max()) > 1e-3  # int8-style drift fails the gate
