# ODC build plan — status tracker (plan v1.0 LOCKED 2026-10-13)

## Baseline environment (recorded, not edited)
- Demo target: Apple M2 Pro, 16 GB, macOS 15.4, Python 3.12.4, project
  `.venv` (psycopg 3.3.6). System has psql, cargo, fastapi/pydantic/pytest.
  Missing (Phase 4+): opencv-python, onnxruntime. No GPU.
- PLOMID: release binary `0.1.0`, scratch instance `127.0.0.1:55433`,
  loopback-only, temp data dir. No second database introduced.
- Footage: 8 Pexels clips verified in `footage/` + manifest; SVIRO reachable
  (bulk deferred); TICaM/Drive&Act requests pending owner.
- Training pipelines: none added. Untracked media ignored via
  `footage/.gitignore`.

## Phase status
| Phase | Status | Evidence |
|---|---|---|
| 0 audit/feasibility/baseline | DONE 2026-10-13 | smoke 10/10 (`scripts/smoke_plomid.py`); docs/odc ×6 this dir; footage manifest |
| 1 contracts/foundation | DONE 2026-10-13 | `odc` pkg (domain/contracts/config/health/cli/logging), 22 unit tests, ruff+format+mypy-strict green, `odc health --probe-db` HEALTHY live, requirements.lock |
| 2 PLOMID SQL repository | DONE 2026-10-13 | `PlomidRepository` + V001/V002 + runner; 7 integration tests live, 33 total green; findings F1–F5 in SQL capabilities |
| 3 camera/replay/health | DONE 2026-10-13 | `odc/capture.py` (file/live/fake sources, bounded queue, advisory quality, rate-limited health writer) + `odc replay`; 45 total green; F1/E1 full-pass counts verified |
| 4 detector eval | DONE 2026-10-13 | RT-DETRv2-S live: native MPS ref + legacy ONNX export + CPU parity ≤3e-5, CoreML rejected w/ evidence; `odc detect` + observational F1 report (P50 213ms CPU); 57 total green |
| 5 tracking/pose/seats | DONE 2026-10-13 | ByteTrack-style (numpy, greedy deviation recorded) + MediaPipe-lite fallback + demo layouts; twin-track duplicate finding → R14; 75 total green | |
| 6 temporal state machine | DONE 2026-10-13 | `odc/state.py` (hysteresis/dwell/stale/cooldown/reconcile) + full F1→SQL chain proven live; 93 total green | |
| 7 cabin zero-shot eval | DONE 2026-10-13 | Grounding miner live (694MB official ckpt, sha-pinned); 8×6×10 batch → verdicts: child-seat/belt/size UNSUPPORTED, misuse/object/blanket UNEVALUATED; 117 total green |
| 8 console | DONE 2026-10-13 | FastAPI+HTMX/Alpine/ECharts console (7 screens, SSE, dossier, benchmark card, tour) + service loop; live-fire verified; Playwright E2E; 110 total green |
| 9 validation lab | DONE 2026-10-13 | 22 hand-labeled frames, dev/holdout splits, `odc validate` + HTML gallery + compare + DB record; holdout caught night gap; 119 total green | |
| 10 optimization/HW | DONE 2026-10-13 | Stage profiler proves infer=99%; threads=8 in laptop profile; int8 rejected on parity; dev re-run identical; 129 total green |
| 11 security/privacy | DONE 2026-10-13 | Redaction, SQLi-guard test, payload caps, purge, retry/restart/fault tests, license scan clean, vuln scan clean; 139 total green | |
| 12 packaging/CDK | DONE 2026-10-13 | Dockerfile+compose verified (build+selftest in image), bootstrap/selftest/purge, backup drill, real screenshots, ODC_OPERATIONS.md; 144 total green | |
| 13 OEM/fusion/fleet | DONE 2026-10-13 | VSS fusion (file-replay + MQTT stub) + gated guardian + fleet API/page + OEM acceptance + production-gap plan; vector deferred w/ rationale; 152 total green |

## Verification log
- `scripts/smoke_plomid.py` → 10/10 PASS, exit 0, 2026-10-13 (one finding:
  no bare `float` alias → D8).
- `file *.mp4` → 8/8 valid ISO MP4.
- Phase 2: 33/33 green (26 unit + 7 live-PLOMID integration, fresh scratch
  instance per test); ruff + format + mypy-strict green. Findings F1–F5 in
  ODC_DECISIONS.md D11.
- Phase 3: 45/45 green (unit: fakes/queue/quality/disconnect/health-writer;
  integration: determinism, full-pass counts, corrupt/missing files, bounded
  live health writes, `odc replay` CLI exit 0).
- Phase 4: 57/57 green (unit: labels/decode/threshold/manifest; integration:
  CPU-pin, real F1 detections w/ provenance, cross-session determinism,
  MPS-native parity, measured report). F1 observational: 11 inferred frames,
  person×20/suitcase×12/phone×1/chair×1, CPU P50 213ms/P95 265ms/cold 264ms
  on M2 Pro. Optimum split (no optimum.onnxruntime in v2.3) → legacy
  torch.onnx exporter used (upstream path).
- Phase 5: 75/75 green. Tracker/association/pose evidence + R14 below.
- Phase 6: 93/93 green (14 temporal unit scenarios + 3 live chain tests:
  F1→SQL states/events reconciled, E1 brief-absence stays UNKNOWN, STALE
  persisted). Full chain runs detector→tracker→assoc→engine→PLOMID.
- Phase 8: 110/110 green (console unit + TestClient + Playwright E2E).
  Live-fire: `odc serve` replayed full F1 (326 frames, 0 dropped) into
  PLOMID; all 8 screens + SSE + dossier + benchmark-card served 200.
- Phase 7: 117/117 green. Evidence batch (8 clips × 6 frames × 10 gd-v1
  prompts) + verdicts file + miner integration test. No-training guard
  test enforced. CLIP signal recorded as second-opinion only.
- Phase 9: 119/119 green (metrics math + determinism + live dev/holdout +
  DB record). Dev gates pass at frozen thr 0.3; holdout caught the night
  gap (2 wrong, documented, release-gating).
- Phase 10: 129/129 green. Profile → threads → quant-gate → parity re-run.
  `odc profile` + `odc providers` shipped; laptop-m2pro-cpu profile locked.
- Phase 11: 139/139 green (redaction, SQLi guard, payload caps, purge,
  retry budget, restart, corrupt-weights, license/vuln scans). Recovery
  matrix tested; disk-full stays documented residual.
- Phase 12: 144/144 green (selftest/bootstrap/purge paths, live provision).
  Image builds; selftest honest inside container; backup/restore drilled.
- Phase 13: 152/152 green (fusion freshness, guardian gating matrix, fleet
  reads, new event types). VSS/file-replay/MQTT-stub tested; guardian
  advisory-only; OEM acceptance + production-gap docs written.
- Phase 1: `pytest` 22 passed; `ruff check` + `ruff format --check` green;
  `mypy --strict` 7 files clean; `odc version/config-check/health` exit 0;
  `odc health --probe-db` against live scratch PLOMID → overall HEALTHY
  (`SELECT 1 ok`). ruff idiom decisions: keep `(str, Enum)` + `Generic[T]`,
  per-file E501 ignore for SQL evidence scripts.
