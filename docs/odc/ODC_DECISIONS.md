# ODC decisions (date-stamped, append-only)

## 2026-10-13 — D1: Python-only application stack
ODC application code is Python only (FastAPI, psycopg3, OpenCV, ONNX Runtime,
NumPy, Pydantic v2). No C++/Rust/Node service in this repo. Rationale: PLOMID
is reached over PG wire; model runtimes ship Python packages.
Locked in `PLOMID_ODC_MASTER_BUILD_PLAN.md` Phase 1.

## 2026-10-13 — D2: Detector baseline flipped to RT-DETRv2-S
Was D-FINE-S default. Research 2026-10-13: RT-DETRv2 checkpoints are COCO-only
with Apache-2.0 checkpoint release (no Objects365 lineage); D-FINE `_obj365`
family not commercially cleared. Baseline = RT-DETRv2-S, challenger =
D-FINE-S/M `_coco` only. Both benchmarked on cabin clips before freeze.

## 2026-10-13 — D3: `ultralytics` package banned from runtime
Loading even Apache-licensed weights through `ultralytics` pulls AGPL into the
app. Use upstream/HF checkpoints + export paths only. YOLO-World excluded
(GPL-3.0).

## 2026-10-13 — D4: Interactive console stack
FastAPI + Jinja2 + HTMX partials + Alpine.js (SVG interactivity) + vendored
ECharts + Tailwind local build. JS libs vendored presentation only, no CDN
(air-gapped OEM labs), no build step, no Node service.

## 2026-10-13 — D5: Laptop reference target
Demo target = developer laptop (Apple M2 Pro, 16 GB, macOS 15.4, Python 3.12).
All latency/throughput figures are development targets, never ECU promises.
ECU profiles (QCOM/TDA4/Orin) stay matrix entries until hardware is in hand.

## 2026-10-13 — D6: Footage-first, Pexels core set
8 landscape 1080p+ Pexels-License clips downloaded + verified (`footage/`,
69 MB) + 9 backup links + dataset access map in `footage/README.md`. SVIRO
host reachable with per-vehicle packages (full set deferred: tens of GB).
Real-RGB datasets never in public video without written OK.

## 2026-10-13 — D7: Stale PLOMID docs corrected by execution
`future/vector-graph.mdx` ("vector not available") and `compatibility.mdx`
("FK not enforced") are outdated. Proved by `scripts/smoke_plomid.py` 10/10
against release binary `0.1.0` on 2026-10-13: FK enforced, `VECTOR(FLOAT32,n)`
+ HNSW + distance ops live. Plan trusts execution, not overview docs.

## 2026-10-13 — D8: No bare `float` alias in PLOMID SQL
`(x)::float` fails (`type "float" does not exist`); use `::double precision`
/ `::float8` / `::real`. Recorded in ODC_SQL_CAPABILITIES.md.

## 2026-10-13 — D9: TLS declined, loopback-only for dev
`sslmode=require` fails against this build. Dev runs on 127.0.0.1 with
explicit credentials; no remote-plaintext without opt-in. Phase 11 must
re-verify before promising transport security.

## 2026-10-13 — D10: Phase 1 pins
pydantic 2.14 / pydantic-settings / pyyaml / pytest 8 / ruff 0.17 / mypy 2.4
(strict) in `requirements.lock`. Deps for later phases deliberately absent
(FastAPI Ph8, OpenCV Ph3, onnxruntime Ph4). `odc health` reports plomid
UNKNOWN→DEGRADED overall unless `--probe-db` proves otherwise.

## 2026-10-13 — D11: Phase 2 dialect findings (all execution-proved)
- F1: inline column-level `REFERENCES` rejected — table-level
  `FOREIGN KEY … REFERENCES` only (`V001/V002` use named constraints).
- F2: multi-statement `execute()` does not apply whole files — the runner
  splits to single statements with comment stripping (`split_statements`).
- F3: bare `version` column binds to the `version()` function — tracker
  column is `schema_version`.
- F4: `LIMIT/OFFSET` params rejected — app-clamped int literal inlined;
  all values stay parameterized.
- F5: pool setup-callback path dropped — connection-per-operation (fits
  single-writer PLOMID at ~1 Hz snapshot cadence; pooling revisited in
  Phase 10 on profiling evidence only).
- Verified additionally: BIGSERIAL, `TEXT[] DEFAULT '{}'`, ON CONFLICT on
  every write path, information_schema/pg_indexes introspection.

## 2026-10-13 — D14: Apple acceleration (no CPU-only replay)
- Native reference runs on torch MPS (Metal) where available; ONNX Runtime
  tries CoreML EP first, CPU fallback second. Provider recorded per
  inference (`InferenceResult.provider`).
- CoreML EP has per-op coverage limits (DETR attention ops may stay on CPU
  via automatic node fallback) — measure, never assume. Phase 4 reports
  CPU-vs-CoreML P50/P95 + cross-provider parity on identical frames.
- CPU remains the guaranteed baseline; any provider failing clearly falls
  back with a logged reason (plan §2.4 hardware-portability rule).

## 2026-10-13 — D12: Phase 3 capture rules
- Replay determinism anchors virtual timestamps to file mtime by default
  (`replay_anchor: file-mtime` or explicit ISO) — same file+config,
  identical (seq, ts, shape); verified on F1 (10 frames) + E1 (389 frames).
- Quality signals (dark/bright/blurry/frozen) are ADVISORY ONLY, computed
  per frame, never persisted as truth.
- Camera health writes rate-limited (change-flush + min interval) and
  failure-resilient (logged, never raised); no metric-write storm.
- Live sources bound reconnect attempts with doubling backoff; stale/dead
  thresholds surface DEGRADED/UNHEALTHY distinctly; fakes always labelled.

## 2026-10-13 — D15: Phase 4 inference verdicts (all measured)
- CoreML EP REJECTED for rtdetrv2-s: all-300-query zeroed logits on real
  frames (529/590 nodes assigned yet numerically wrong) — manifest blocks it
  (never tried); loader prefers GPU-class providers (CUDA→CoreML→CPU) with a
  per-inference degeneracy tripwire that fails closed on silent corruption.
  (ORT has no MPS provider; torch-MPS stays the native reference path.)
- Preprocessing MUST be PIL-BILINEAR 640×640 /255 (matches HF reference to
  3e-5); cv2.INTER_LINEAR diverges (max|dbox| ~1.0) — runtime hardcodes PIL.
- Export: optimum 2.3 dropped optimum.onnxruntime → legacy torch.onnx
  (upstream path, opset 17); dynamo rejected (aten._is_all_true); tracer
  warnings accepted only because 3-frame parity gate passes.
- MPS-native is the reference (34–45ms warm), ORT-CPU the deployment path
  (~185ms infer); cold starts ~2.5s native / 264ms ORT.
- MPS cross-process caveat (found during Phase 7): torch MPS is exactly
  deterministic within a process but drifts ~5.0 in low-end logits ACROSS
  processes (top detections agree). Parity test therefore compares ORT-CPU
  against a FROZEN native fixture (tests/fixtures/rtdetrv2_native_ref.npz
  + provenance JSON), never live MPS. Deployment path unaffected.
- Reports are stamped OBSERVATIONAL (no ground truth); P/R/F1 waits for
  Phase 9 labels. No cabin accuracy claimed from COCO AP.

## 2026-10-13 — D16: Phase 5 tracking/pose/seats verdicts (all measured)
- Tracker: ByteTrack-style association (high/low two-stage, Kalman) with
  DOCUMENTED deviation — greedy IoU instead of Hungarian (exact for sparse
  cabins; OC-SORT stays challenger). Session-local IDs, reset verified.
- Twin-track finding: DETR duplicate boxes spawn twin tracks (F1 t1/t5 same
  adult) → ambiguity, not forced seats. Follow-up R14: duplicate-suppression
  study (NMS vs calibration vs query-dedup) feeds Phase 6; seat-first
  association already absorbs twins into one occupied seat.
- Association scoring verified sound on clean detections (F1: foreground→
  near-left 0.9, woman→mid 0.99, baby→child-seat 1.0); ambiguity preserved
  as seat_id None, never forced.
- Pose: MediaPipe Landmarker Lite working fallback (1 pose/33 kpts on F1,
  REVIEW_REQUIRED commercial); RTMPose has no trustworthy official ONNX
  (third-party ports only) — stays preferred-but-blocked; NullPose keeps
  the pipeline running without pose.
- Demo layouts demo-f1-v0/demo-r1-v0 are hand polygons from viewed frames,
  APPROXIMATE, replaced by calibration wizard (Ph8).

## 2026-10-13 — D17: Phase 6 temporal rules (all tested)
- EMPTY_CANDIDATE requires GOOD observations (healthy camera, clear frame,
  nobody ambiguous nearby) + sustained zero evidence; misses/occlusion/
  stale/poor-quality yield UNKNOWN/STALE — never EMPTY, never OCCUPIED.
- Two-tier occupancy (LIKELY→OCCUPIED) with separate dwell, release
  hysteresis (likely@3–5, unknown@15 misses), cooldown-suppressed repeat
  events, out-of-order drops, restart-safe counters.
- Cooldown compares wall-clock to last RECORDED event only (epoch default);
  frame time never gates it — replay frames may postdate wall clock.
- Evidence counters reset on every transition (no pre-armed EMPTY).
- reconcile_from_events: last-event-per-seat wins; current-state table is
  a cache of the append-only history. Rule confidences are deterministic
  functions of evidence counts, documented, never measured scores.

## 2026-10-13 — D18: Phase 8 console verdicts
- Tailwind dropped for hand-written token CSS (no Node build step keeps
  Python-only + offline honest); visual contract lives in style.css.
- Route handlers resolve the repo from app state (overridable, outage-safe);
  `odc serve` refuses to start without ODC_DSN — no silent memory store.
- Tests are DSN-only over pgwire (ODC_TEST_DSN, default :55460) with
  FK-safe row cleanup — no binary paths, no process management in tests.
- Live-fire found two real bugs: staleness used wall-clock (replay frames
  carry virtual times → evidence-clock fix + regression test) and looped
  replay read as out-of-order (restart resets the clock reference) plus
  empty publishes (dropped frames skip publish). Also added replay_loop.
- Playwright E2E kept: seat-click→inspector and timeline-chart over seeded
  SQL rows, real uvicorn + headless Chromium.

## 2026-10-13 — D19: Phase 7 zero-shot verdicts (all measured)
- Miner: Grounding DINO SwinT official ckpt (694MB, sha 3b3ca256) at pinned
  code rev 856dde20; transformers-5 compat shims documented in vlm.py;
  4s/question CPU. Eval-only (O365 lineage), never onboard/real-time.
- child-seat prompts: recall 1.0 on F1/C1/C2 BUT false-fire 1.0 on
  E1(empty!)/S1/S2/R1 → UNSUPPORTED. Fatal: E1 scores HIGHER (0.502) than
  C1 true hits (0.428) — no separable threshold exists at 0.35/0.25.
- belt prompts: S1 1.0 BUT E1/C2/F1 false-fire (seat seams read as belts)
  → UNSUPPORTED; belt stays CANDIDATE via pose-geometry + buckle fusion.
- small-occupant: fires on empty cabin → UNSUPPORTED; size rule must be
  seat-relative (absolute areas meaningless: F1 adults 0.5+ vs R1 0.2-0.35).
- misuse/object/blanket: UNEVALUATED (no positive footage); R15 tracks
  seat-ROI variant + synthetic misuse renders.
- Person-via-grounding: PROVISIONAL (night miss, empty-fire); person stays
  RT-DETR's job; grounding person prompt retired from product path.

## 2026-10-13 — D20: Phase 9 validation verdicts (all measured)
- 22 hand-labeled frames (viewed, ~0.05 precision, IoU 0.5); dev tunes,
  holdout (N1/S2) verifies once. Frozen operating point thr 0.3 from dev
  sweep (0.3/0.4/0.5); runtime default stays 0.5 (prior evidence stands).
- Dev: P .195/R .407 seat .816 (0 wrong). Box-level harshness is honest
  (hand-label precision, small R1 boxes, R14 duplicate flood quantified).
- Holdout caught a REAL gap: N1 dark-driver misses voted EMPTY (2 wrong).
  Global brightness can't split N1/E1 (~50); ROI means overlap too.
  Fix = methodology (dark→abstain documented; brightness insufficient) —
  NOT threshold tuning. Night gap release-gates; R16 tracks NIR research.
- N1 also hallucinates car/cat/parking-meter at 0.3-0.6 — out-of-domain
  behavior flagged, never shown as detections above threshold in product.

## 2026-10-13 — D21: Phase 10 optimization verdicts (all measured)
- Profile first: infer 99.0% (190ms P50), decode 1.5ms, track 0.3ms,
  engine 0.1ms, assoc ~0ms. Only inference matters; SQL writes ~7ms.
- Threads: 8 wins (169ms) vs default 190 / 4→237 / 12→246 (efficiency-core
  spillover). Locked in laptop-m2pro-cpu profile, NOT in code (machine
  specific). Targets: P95 ≤250ms, ≥4fps sustained — met.
- INT8 dynamic quant: 1.5× faster (119ms) BUT max|dbox| 0.97 parity failure
  → REJECTED, disabled, documented. Static/calibrated quant needs approved
  procedure + owner sign-off (R17). FP16/CPU skipped (no benefit, only risk).
- Parity: dev validation re-run after all Phase 10 activity is
  bit-identical to Phase 9 frozen metrics (P .1947/R .4074/seat .8158).
- Providers: `odc providers` reports ORT set + per-model allow/reject with
  reasons; disallowed providers raise, never silently engage.

## 2026-10-13 — D22: Phase 11 hardening verdicts
- Secrets: DSN env-only, scrub helpers, CLI/config outputs scanned clean.
- SQL: parameterized-only enforced by allowlist test (2 reviewed LIMIT
  literals); identifiers never user-built.
- Payloads bounded (evidence ≤2000 chars); oversize rejected, not truncated.
- Deps: pip-licenses 2026-10-13 — zero GPL-family (MIT/BSD/Apache/PSF/ISC/
  MPL-file-level); ban-test in suite. pip-audit: zero known vulns.
- Media: metadata-only DB, one transient hub JPEG, footage/ untracked,
  `odc purge` age-based deletion. No identity data anywhere.
- Recovery tested: camera/infer/SQL/config faults, restart (2 sessions),
  retry budget (3 then surface), corrupt weights → degraded. Disk-full is
  documented residual. No certification claims, anywhere.

## 2026-10-13 — D23: Phase 12 packaging verdicts
- Image `plomid-odc:phase12` builds (1.38GB, torch-dominated) and runs
  `version` + honest `selftest` (footage/model absent → fail/skip, never
  green lies). Weights/volume-mounted, never baked.
- `docker compose up` = PLOMID + replay console (compose references the
  published plomid image; local plomid build is the fallback).
- Backup/restore drilled live: bootstrap → rows → stop → tar (780K) →
  wipe → restore → rows + schema v2. Rollback = snapshot + previous image
  (migrations additive-only by design, tested V1→V2).
- Screenshots in docs/odc/assets are REAL outputs (seeded SQL session),
  scripted via scripts/capture_screenshots.py for reproducibility.
- README.md added (quickstart + honesty contract + docs map).

## 2026-10-13 — D24: Phase 13 integration verdicts (roadmap complete)
- Fusion: VSS-named signals (KUKSA databroker map + CAN-DBC fallback doc);
  file-replay deterministic; MQTT adapter fail-closed STALE without broker
  (paho stays optional, never imported hard). Camera never overrides bus.
- Guardian: additive enum members (UNATTENDED_OCCUPANT_CANDIDATE,
  FUSION_SIGNAL_STALE) with changelog here — no domain version bump needed
  for additive, backward-compatible additions. Advisory only, actuates
  nothing; dwell-gated, disarms on any broken condition.
- Fleet-lite: aggregates + criticals only, documented privacy contract.
- Vectors: DEFERRED with written rationale — no onboard embedder exists,
  relational queries suffice, storing crop embeddings would violate the
  no-identity rule. Revisit only with a justified, approved use.
- Production gap: every GAP row names owner + evidence plan; product label
  stays demonstrator until the table clears. No certification claims.

## Post-roadmap hardening (host-DB era, 2026-10-14)
- D25: GPU preference + stdout hygiene + uncalibrated honesty.
  Loader tries CUDA→CoreML→CPU minus blocked (CoreML measured-corrupt, never
  tried); per-inference degeneracy tripwire fails closed on silent provider
  corruption (the exact CoreML signature). ORT has no MPS provider;
  torch-MPS stays the native reference. CLI logs → stderr so stdout JSON
  stays machine-parseable.
- D26: random-video contract. Unknown geometry → empty seat list +
  UNCALIBRATED banner + person evidence still flows (a live-found crash
  fixed with regression test). `odc analyze FILE` probes any video without
  DB writes. 11-clip matrix measured: calibrated cabins reach LIKELY,
  night/partial → UNKNOWN, crowds/animals/offices → evidence without
  claims, corrupt files → clean exit 1. Hand-labeled empty-office
  assumption was wrong on viewing (busy office) — verify footage, and the
  matrix records it as process evidence.
- Seat 3-way scoring (correct/wrong/abstain) is the product metric;
  UNKNOWN is reported, never hidden. Compare + DB-record paths tested.
- D27: idle-by-default serve. The console boots DISCONNECTED; the operator
  picks source/clip/layout/threshold and presses Start (POST
  /api/service/start; 400 on bad input, 409 when already running). Stop
  keeps last states on screen. `--auto-start` preserves immediate-run for
  compose/demos. Tested: idle boot, rejections, full start/stop lifecycle.
- D28: operator control refinements. Clip/layout pickers are dropdowns fed
  by /api/service/sources (no hand-typed paths); live-camera device field
  appears only for live source; mode follows source (LIVE/REPLAY) in session
  metadata. Live branch proven via file-backed LiveCameraSource test.
- D29: apparent-size rule + bandwidth-free evaluation (2026-10-14).
  Absolute bbox areas are meaningless across viewpoints (measured: F1 adults
  0.5 vs R1 adults 0.05-0.26 vs C2 child 0.6) — rule is seat-relative ratio
  with temporal 3-vote gate, PROVISIONAL, close-up geometries only; never age.
  Male/female classification refused (unreliable + biometric-law toxic + zero
  NCAP/safety value). SVIRO deferred on download budget (2.6GB/vehicle);
  meanwhile: procedural exact-GT motion scenarios (6, all green) test
  tracker/engine logic with zero MB, and a 6-shot owner re-enactment list
  covers empty-CRS/blanket/misuse/night/entry-exit gaps. Verify footage,
  don't trust titles (X2 "empty office" held ~25 people).
- D30: OWNER OVERRIDE — gender-appearance classification implemented 2026-10-14
  against the D29 refusal, at explicit owner direction. Containment: (1)
  EXPERIMENTAL, disabled by default (`enable_gender_appearance: false`);
  (2) CLIP zero-shot only, no training; (3) never run on small occupants;
  (4) low-confidence → UNKNOWN (0.7 floor); (5) transient diagnostic ONLY —
  excluded from persisted state, dossier, and exports; (6) biometric-law
  exposure (GDPR/EU-AI-Act/BIPA) documented here and accepted by the owner
  as their risk decision. Revisit/removal on first legal review.
- D30 update (measured 2026-10-14): CLIP zero-shot on 13 person crops — 6
  confident calls, ALL correct (F1 woman .72/.82, R1 driver man .98, S2
  woman .93); 7 abstentions incl. all child boxes (0.51/0.49 coin-flip
  correctly refused). Operating characteristic: high precision, low
  coverage. Containment from D30 holds (default-off, transient-only).
- D31: D-FINE-S LEADING CHALLENGER (onnx-community conversion, provenance
  logged). Dev head-to-head @0.3: R .500 (vs .407), seat .895 (vs .816),
  unk .105 (vs .184), ~120ms CPU (vs ~190ms). Promotion track: official
  .pth (GitHub-releases pipe stalls — retry) → own export → native parity
  → re-validation → baseline-swap review. RTMDet DEFERRED: no trustworthy
  ready artifact (0-like ports only); two verified candidates suffice.
- D32: empty-stream heartbeat. Skipping publish on empty frames hid all
  signs of life (found live: service consumed 389 frames showing idle).
  Heartbeat ≤1/s keeps running/frames visible; last-known seats retained.
- D33: relative cabin-size child candidate (2026-10-23). Keep per-seat person
  box/seat-zone ratios for a bounded 60-frame window, then compare a currently
  observed seat against recent peer seats. Three of five votes at least 25%
  below the peer median creates a persisted `child-candidate:relative-cabin-size`
  reason and a candidate event. This is a geometry cue, not a validated age
  classifier; it carries no confidence score. It requires three usable peer
  seats and does not label a missing detection as a child. On the supplied clip,
  a fresh probe found three hotspots and inferred RHD from an image-side wheel
  cue; a full 1,267-frame CPU replay processed 212 inference frames and showed
  three mapped person detections, three occupied seats, and Seat 2 marked as a
  child candidate. The supplied saved four-zone geometry replay
  ended with three person boxes but only one seat assignment and no child
  candidate. Treat the saved map as a demonstrated association defect;
  start a new session with Auto to regenerate the proposal. Candidate state
  previously toggled repeatedly, so the clear rule now requires six consecutive
  contradictory observations; the complete replay ran with this rule enabled.
  Live event history now deduplicates stable IDs. Dedicated child-seat
  recognition remains unsupported by the active COCO checkpoint.
- D34: live event-history deduplication (2026-10-23). The service passes a
  cumulative event list to the hub on each frame; blindly prepending it made
  candidate transitions appear many times in the live timeline. Deduplicate
  by stable `event_id` when merging the bounded transient history. SQL event
  writes remain per new engine event.
