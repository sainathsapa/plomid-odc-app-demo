# PLOMID ODC — Single Source of Truth

**Document version:** 1.0
**Verification snapshot:** 2026-10-27
**Product status:** engineering demonstrator and pilot platform. Not production-qualified, not safety-certified, not an airbag or restraint controller.
**Scope:** everything an engineer, evaluator, event host, or pitch audience needs to understand what ODC does, how it works, what is proven, what is not, and where each claim comes from.

> **Reading rule.** This document consolidates the repository; it does not replace the locked source documents. When this guide and the code disagree, the code, tests, and a live probe win. Known disagreements are listed openly in [Section 20](#20-known-documentationcode-discrepancies). Nothing here is a certification claim.

**Contents**

| | |
|---|---|
| [1. One-page brief](#1-one-page-brief) | [13. Temporal state and events](#13-temporal-state-and-events) |
| [2. Capability matrix](#2-capability-matrix-in-detail) | [14. Persistence and schema](#14-persistence-and-schema) |
| [3. Architecture](#3-architecture) | [15. APIs and console](#15-apis-and-console) |
| [4. Runtime and deployment](#4-runtime-and-deployment) | [16. Validation and benchmarks](#16-validation-benchmarks-and-test-status) |
| [5. Models and governance](#5-models-and-model-governance) | [17. Privacy, security, legal](#17-privacy-security-and-legal) |
| [6. Datasets and footage](#6-datasets-footage-and-evaluation-media) | [18. Operations](#18-operations) |
| [7. End-to-end pipeline](#7-end-to-end-pipeline) | [19. Limitations and roadmap](#19-limitations-and-roadmap) |
| [8. Detection](#8-detection-in-detail) | [20. Known discrepancies](#20-known-documentationcode-discrepancies) |
| [9. Tracking](#9-tracking-in-detail) | [21. Event and pitch Q&A](#21-event-and-pitch-qa) |
| [10. Seat allocation](#10-seat-allocation-and-calibration) | [22. Glossary](#22-glossary) |
| [11. Age and gender](#11-age-and-gender) | [23. Source map](#23-source-map-and-verification) |
| [12. Belts, children, pose, DMS](#12-belts-children-pose-dms-masks-and-specialist) | [24. Document control](#24-document-control) |

---

## 1. One-page brief

### 1.1 What this is

PLOMID ODC turns a cabin camera stream or prerecorded video into structured, per-seat occupant evidence:

```text
camera / video
  -> decode + quality checks
  -> sampled detection
  -> tracking
  -> seat-geometry association
  -> temporal state reasoning
  -> optional attributes (age/gender, child, belt, pose, DMS)
  -> PLOMID SQL
  -> live console / timeline / APIs / dossier
```

It is built for the case where an OEM or fleet operator already has a cabin camera and wants explainable occupant state without sending video to a cloud, training a new model, or inventing certainty the camera cannot support.

### 1.2 The honest one-sentence pitch

> ODC is an edge, camera-only cabin-occupant demonstrator that produces per-seat states with explicit uncertainty, persists the evidence in PLOMID SQL, and shows the operator exactly why each state was chosen — without identifying people or claiming safety certification.

### 1.3 What works today

| Capability | Status | Short answer |
|---|---|---|
| Per-seat occupancy with temporal reasoning | PARTIAL | Works on configured demo layouts; installed-camera calibration is still required. |
| Multi-person tracking and seat association | VERIFIED | Session-local tracks, stable seat IDs, ambiguity becomes `UNKNOWN`. |
| Automatic seat-map proposal | PARTIAL | Candidate `auto-v1` geometry from sampled person/furniture evidence; operator review required. |
| Seat-map calibration editor | PARTIAL | Versioned layouts can be traced, validated, saved, and loaded; target-camera validation remains. |
| Age and apparent-gender estimates | PARTIAL | Local MiVOLO V2 point estimates with an abstention band; not exact age, not calibrated. |
| Adult/child candidate | PARTIAL | MiVOLO categories drive `Yes/No/Unknown`; known misses and 14.2% Unknown in the child audit. |
| Child-seat candidate | PARTIAL | Optional open-vocabulary specialist, default off, eval-only lineage; never promoted to certainty. |
| Belt-strap visibility | PARTIAL | Geometric strap detection; cannot prove buckle engagement or restraint correctness. |
| Pose keypoints / posture | PARTIAL | MediaPipe pose; 33 landmarks; preview-grade posture candidates. |
| DMS drowsy/distracted/phone | PARTIAL | MediaPipe face + COCO phone; preview candidates; disabled by default. |
| Temporal events and timeline | VERIFIED | State transitions and alerts with frame, reason, config, model provenance. |
| PLOMID persistence and schema | VERIFIED | Live schema V8; 17 domain tables; 12 enforced FKs plus documented soft links. |
| Operator console | IMPLEMENTED | Live, Seat map, Timeline, Diagnostics, Validation, Config, Data model, Fleet, Dossier. |
| Validation harness | IMPLEMENTED | Locked 22-frame dataset, dev/holdout discipline, reports. |
| Operator-triggered assistant | IMPLEMENTED | Local Ollama by default; hosted provider optional; never in the frame loop. |
| Identity / face recognition | UNSUPPORTED | Prohibited by design; no identity embeddings. |
| Exact age, breathing, heartbeat | UNSUPPORTED | Must remain unclaimed. |
| Belt-fastened certainty | UNSUPPORTED | Camera-only evidence cannot establish it. |
| Emotion recognition | UNSUPPORTED | Not implemented; not claimed. |
| Airbag / pretensioner / actuation | UNSUPPORTED | Advisory only; no actuation path. |
| Functional-safety / cybersecurity certification | UNSUPPORTED | No ISO 26262 / ASPICE / ISO/SAE 21434 claim. |
| Training / fine-tuning on customer data | UNSUPPORTED | Pretrained inference only; eval labels never update weights. |

**Status legend**

- **VERIFIED** — the code path exists, is exercised, and has a focused passing test or deterministic contract check in this repository.
- **IMPLEMENTED** — the code path exists and is exercised, but the evidence is broader or less formal than a focused verification.
- **PARTIAL** — works within stated limits; the limit is part of the answer.
- **UNSUPPORTED** — no code path, no evidence, or explicitly prohibited. Do not claim it.

These are capability labels. Model-governance labels such as `APPROVED`, `REVIEW_REQUIRED`, and `EVAL-ONLY` are separate and appear in [Section 5](#5-models-and-model-governance).

### 1.4 What it deliberately does not do

- It does not identify people, compute face embeddings, or match occupants across sessions.
- It does not train, fine-tune, distill, or update weights from footage or evaluation labels.
- It does not claim exact age, breathing, heartbeat, emotion, or belt-buckle state.
- It does not claim that “no detection” means an empty seat. Weak evidence becomes `UNKNOWN`/`STALE`.
- It does not control airbags, pretensioners, doors, or any actuator.
- It does not claim production qualification, functional safety, or cybersecurity certification.
- It does not require a cloud. The default assistant is local; hosted inference is opt-in.

### 1.5 The value proposition

| Buyer problem | ODC answer |
|---|---|
| “We already have cameras; can we use them?” | Replay and live-camera sources, normalized seat zones, CPU baseline. |
| “We cannot send cabin video to a cloud.” | Local inference; raw video stays on disk; PLOMID runs on the host; assistant is local by default. |
| “We need evidence, not a black-box label.” | Every state has confidence, visibility, reason codes, frame sequence, config version, model-set version. |
| “We cannot train a model for every vehicle.” | Pretrained inference only; geometry and thresholds are versioned configuration, not weights. |
| “How do we know it is not guessing?” | Ambiguity becomes `UNKNOWN`; `EMPTY_CANDIDATE` needs positive empty evidence; stale data expires. |
| “How do we integrate?” | REST/SSE APIs, PLOMID SQL, schema model, dossier export, OEM boundary docs. |
| “Can we pilot quickly?” | Upload a clip, auto-propose seats, run the console, inspect evidence, export results. |

The strongest honest commercial statement is:

> ODC is a working engineering demonstrator and a credible pilot platform for camera-based cabin occupancy and candidate occupant attributes. Production qualification, target-hardware validation, NIR/fisheye coverage, legal review, and OEM safety processes remain external work.

---

## 2. Capability matrix in detail

| Area | What is implemented | What is proven | What remains limited |
|---|---|---|---|
| Occupancy | Per-seat temporal state engine, dwell/hysteresis, stale expiry | Unit and integration tests; dev seat-level 81.6% correct, 0 wrong, 18.4% unknown | Installed-camera calibration; holdout 66.7% with two night errors |
| Detection | RT-DETRv2-S COCO, ONNX Runtime CPU, per-class floors, NMS, filters | SHA-pinned manifest, ONNX parity, live runs | Generic COCO; no child-seat or belt class |
| Tracking | ByteTrack-style Kalman tracker | Unit tests; session-local IDs | Occlusion, twins, no re-identification |
| Seat association | Normalized polygons, track-first/seat-first, ambiguity margin | Geometry tests; stable IDs in state/events/API | Depends on calibration quality |
| Auto seat map | 32-frame probe, heatmap, peak picking, row grouping | Replay probes; saved as `auto-v1` | Must be reviewed; one saved map assigned only 1 of 3 boxes |
| Calibration | Polygon editor, strict validation, versioned PLOMID store | Layout tests; save/load paths | Target-vehicle validation still needed |
| Age | MiVOLO V2, smoothing, abstention band | Focused age tests; child audit | Point estimate, not calibrated; adolescents abstain |
| Gender | MiVOLO apparent-gender softmax | Focused tests | Not identity, not sex, not calibrated; can be disabled |
| CLIP gender branch | Experimental zero-shot probe | Unit tests; D30 small probe | Default off; privacy gap if enabled; legal review required |
| Child | MiVOLO categories; optional specialist | Child audit; integration tests | Incomplete recall; child-seat classifier unsupported |
| Belt | Geometric strap detector, K-of-M voting | Synthetic and clip probes | No buckle certainty; dark/soft webbing fails |
| Pose | MediaPipe Pose Landmarker Lite | Keypoint tests | Bundle terms `REVIEW_REQUIRED`; posture is candidate-grade |
| DMS | MediaPipe Face Landmarker, blink/jaw/head pose, phone | Focused DMS tests | Preview-grade; disabled by default; not safety-certified |
| Masks | GrabCut fast path; optional SAM precise path | Unit tests; probe measurements | SAM is slow; masks do not classify |
| Specialist | Grounding DINO child/child-seat candidates | Integration regression | Eval-only O365 lineage; default off; false positives observed |
| Alerts | Belt, drowsy, distracted, phone, unattended | Alert tests | Candidate-grade; fusion-gated path needs vehicle bus |
| Guardian | Parked + locked + closed + occupied dwell | Unit tests | Advisory only; no actuation |
| Persistence | 17 domain tables, JSONB, vector geometry | Live schema and API probes | PLOMID dialect quirks; no TLS in current build |
| Console | 8 nav screens plus dossier/calibration routes | Integration tests; live use | Some routes are demo-depth, not OEM-depth |
| Assistant | Local Ollama / optional hosted | Live request/history paths | External-processing risk when hosted; explicit operator action only |
| Validation | 22-frame locked set, dev/holdout, reports | Reports and scripts present | Tiny sample; not population evidence |
| Night / NIR | Low-light quality checks | Holdout surfaced the gap | No NIR model or qualified night claim |
| Fisheye | Partial OpenCV handling | Documented path | Fisheye-aware undistort deferred |
| Fleet | Fleet page/API | Demo-lite | Not a fleet management product |
| OTA / ECU | Contracts and docs | None | No qualified target hardware or signed OTA |
| Identity / emotion / breathing | None | None | Prohibited or unsupported |

---

## 3. Architecture

### 3.1 System shape

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Operator console (FastAPI + Jinja2 + HTMX + Alpine + ECharts)      │
│ Live / Seat map / Timeline / Diagnostics / Validation / Config     │
│ Data model / Fleet / Dossier / Calibration                         │
└───────────────▲───────────────────────────────┬─────────────────────┘
                │ REST + SSE                    │ explicit assistant ask
┌───────────────┴───────────────────────────────▼─────────────────────┐
│ ODC service (Python 3.12, background thread)                        │
│ capture -> quality -> preprocess -> detect -> track -> seats        │
│ -> temporal state -> attributes -> alerts -> annotate -> publish    │
└───────────────┬───────────────────────────────┬─────────────────────┘
                │ SQL (PG wire, loopback)       │ transient frames
┌───────────────▼──────────────┐   ┌────────────▼────────────────────┐
│ PLOMID 0.1.0                 │   │ footage/ files                   │
│ schema: plomid_odc           │   │ in-memory preview JPEG           │
│ 17 domain tables, JSONB,     │   │ no raw video in SQL              │
│ VECTOR(FLOAT32,4) geometry   │   └─────────────────────────────────┘
└──────────────────────────────┘
```

### 3.2 Layers and responsibilities

| Layer | Responsibility | Main code |
|---|---|---|
| Configuration | YAML + environment + defaults, validated by Pydantic | `odc/config.py` |
| Domain contracts | Frames, detections, states, events, health, manifests | `odc/domain.py`, `odc/contracts.py` |
| Capture | Live camera or file replay, timestamps, decode health, reconnect | `odc/capture.py` |
| Runtime | ONNX Runtime provider selection, preprocessing, fail-closed loading | `odc/runtime.py` |
| Perception | Detection post-processing, masks, attributes, scene helpers | `odc/service.py`, `odc/segmentation.py`, `odc/attributes.py` |
| Tracking | ByteTrack-style association and session-local track IDs | `odc/tracking.py` |
| Seats | Normalized polygons, association, layout validation | `odc/seats.py`, `odc/layouts.py` |
| Auto-layout | Probe video, build heatmap, propose candidate seat zones | `odc/autolayout.py` |
| State | Dwell, hysteresis, stale expiry, reason codes, events | `odc/state.py` |
| Attributes | Age/gender, belt, pose, DMS, masks, specialist | `odc/age.py`, `odc/belt.py`, `odc/pose.py`, `odc/dms.py`, `odc/segmentation.py`, `odc/specialist.py` |
| Alerts / fusion | Candidate alerts, parked guardian, vehicle-signal interface | `odc/alerts.py`, `odc/guardian.py`, `odc/fusion.py` |
| Persistence | SQL-only repository, migrations, schema model | `odc/persistence.py`, `odc/migrate.py`, `sql/migrations/` |
| API / console | REST, SSE, pages, dossier, assistant | `odc/console/app.py`, `odc/console/templates/` |
| Validation | Replay evaluation, metrics, reports | `odc/validate.py`, `odc/eval.py`, `odc/cli.py` |

### 3.3 Data flow, stage by stage

1. **Capture.** `FileReplaySource` uses OpenCV `VideoCapture`; `LiveCameraSource` accepts a device index or stream target and reconnects with bounded exponential backoff.
2. **Frame metadata.** Each frame carries source ID, monotonic sequence, UTC/evidence timestamp, dimensions, mode (`LIVE`/`REPLAY`), and pixels.
3. **Preview.** Every captured frame can produce a bounded in-memory JPEG for smooth video, even when inference is sampled.
4. **Quality.** Dark, bright, blur, and frozen-frame checks run before trust is placed in the frame.
5. **Sampling.** Inference runs every `sample_every_n` captured frames (default 6). Captured, processed, and dropped counts are separate.
6. **Preprocess.** BGR to RGB, 640×640 PIL bilinear, float32 NCHW, manifest-approved provider order.
7. **Detect.** RT-DETRv2-S produces sigmoid class scores and normalized `cx,cy,w,h` boxes.
8. **Post-process.** Per-class floors, operator threshold, class-aware NMS, duplicate suppression, vehicle/cabin hallucination filters.
9. **Track.** Kalman/ByteTrack-style matching produces short-lived session-local track IDs.
10. **Associate.** Bounding boxes are scored against seat polygons; near-ties remain unassigned.
11. **State.** The temporal engine combines positive/negative evidence, dwell, visibility, staleness, and transitions.
12. **Attributes.** MiVOLO age/gender, belt geometry, pose, DMS, masks, and optional specialist evidence enrich seat rows.
13. **Alerts.** Rule-based candidates derive from seat rows and DMS; WARNING+ transitions can persist as events.
14. **Publish.** The hub exposes the current snapshot and bounded frame recap; SSE notifies the console.
15. **Persist.** Current seat/occupant state, events, frame observations, detections, metrics, layouts, and assistant interactions go to PLOMID.
16. **Serve.** The console renders live state, evidence, timeline, diagnostics, validation, and the data model.

### 3.4 Failure boundaries

| Failure | Behaviour |
|---|---|
| Camera decode failure | Visible unhealthy/degraded state; never converted into an empty seat. |
| Model missing/corrupt | Fail closed; model health unavailable; seats become `UNKNOWN`; no invented detections. |
| Unsupported/degenerate provider | Rejected; fall back only to an allowed provider; CoreML is blocked with evidence. |
| PLOMID unavailable | Service continues transient-only, logs and shows degraded; SQL truth is not fabricated. |
| Stale frames | `check_stale` expires states toward `UNKNOWN`/`STALE`. |
| Queue pressure | Bounded queue with explicit drop policy; captured/processed/dropped counters. |
| Restart | Durable state/events remain in PLOMID; the in-memory frame recap is lost by design. |

### 3.5 Why PLOMID only

- PLOMID is the sole durable database. ODC does not open SQLite, JSON sidecars, or a second store.
- The ODC schema lives inside the single `plomid` database as `plomid_odc`.
- SQL is the integration boundary: current state, events, frame evidence, layouts, metrics, validation, and assistant history are queryable.
- PLOMID’s PG-wire compatibility has been probed, not assumed. See [Section 14.6](#146-plomid-sql-capabilities-and-quirks).
- Raw video remains on disk under `footage/`; only structured, non-image evidence is persisted.

---

## 4. Runtime and deployment

| Item | Current reality |
|---|---|
| Language | Python 3.12 |
| Web | FastAPI, Jinja2, HTMX, Alpine.js, vendored ECharts, Tailwind |
| CV | OpenCV, NumPy, Pillow |
| Inference | ONNX Runtime for the detector; PyTorch/Transformers for MiVOLO and SAM; MediaPipe tasks for pose/face |
| Database | PLOMID 0.1.0 over PG wire, loopback, SCRAM; schema `plomid_odc` |
| Transport | HTTP on loopback for the console; no TLS in the current PLOMID build |
| Reference target | Apple M2 Pro laptop, CPU execution; macOS arm64 verified; Linux x86_64 via Docker expected |
| Windows | `setup-assets.bat` and `start.bat`; one-command local startup |
| Container | `docker compose up --build` runs the console against the host PLOMID; weights are not baked into the image |
| Model assets | Local, gitignored, SHA-verified; optional models may be absent and degrade visibly |

**Important deployment rule:** PLOMID runs on the host, never inside the ODC container. The console container reaches it through `host.docker.internal:5432`.

**Startup rule:** `odc serve` provisions the `plomid_odc` schema and applies migrations automatically. `odc bootstrap` remains a diagnostic path. The console idles until the operator presses Start unless `--auto-start` is used.

---

## 5. Models and model governance

### 5.1 Portfolio

| Model | Role | Status | License / lineage | Runtime and notes |
|---|---|---|---|---|
| **RT-DETRv2-S COCO** | Active person/object detector | **APPROVED** | Apache-2.0; COCO train2017; no Objects365 lineage | ONNX Runtime CPU; SHA-pinned; parity-gated |
| **D-FINE-S COCO** | Leading challenger | **CHALLENGER** | Apache-2.0 upstream `_coco`; third-party ONNX conversion provenance logged, not independently audited | ~120 ms CPU in dev head-to-head; not the baseline |
| **Grounding DINO SwinT OGC** | Child / child-seat candidate specialist | **EVAL-ONLY** | O365 lineage; not commercially cleared | 662 MiB local checkpoint; bounded single pending job; default off |
| **MiVOLO V2** | Age and apparent-gender estimator | **ACTIVE (local)** | Code Apache-2.0; weights Apache-2.0 per model card; training mix proprietary + open, not fully enumerated | Paired 384×384 face/person crops; CPU/CUDA/MPS; cached |
| **MediaPipe Pose Landmarker Lite** | 33 body landmarks | **REVIEW_REQUIRED** | Bundle commercial terms unverified | Local task bundle; enabled by default when present |
| **MediaPipe Face Landmarker** | 478 face points, blendshapes, head pose | **REVIEW_REQUIRED** | Bundle commercial terms unverified | Local task bundle; DMS disabled by default |
| **SAM ViT-B** | Optional precise masks | **APPROVED optional** | Apache-2.0 | Box-prompted; ~5–7 s/frame CPU; background single slot |
| **OpenCLIP ViT-B-32 (LAION)** | Experimental gender/second-opinion probe | **EVAL-ONLY** | LAION weights; default off | Only when MiVOLO absent; D30 legal risk |
| **ByteTrack-style tracker** | Tracking pattern | **APPROVED pattern** | MIT-style implementations; no weights | Kalman + IoU matching; session-local IDs |

### 5.2 Active detector: RT-DETRv2-S

- **Checkpoint:** `PekingU/rtdetr_v2_r18vd@5650961749fa93567c0d46fc7f43ea4f9e914107`
- **Weights SHA-256:** `d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115`
- **License:** Apache-2.0; COCO train2017; approved for commercial use in the manifest.
- **Input:** RGB 640×640, PIL bilinear resampling, `/255`, float32 NCHW.
- **Parity:** max `|dlogit| <= 2.4e-04`, max `|dbox| <= 3.0e-05` over three F1 frames; gate is `max|dbox| < 1e-3`.
- **Providers:** CUDA preferred, CPU fallback; CoreML is explicitly blocked after all 300 query logits were zeroed (sigmoid 0.5) on measured frames.
- **Output:** model-independent detections with class label, raw confidence, normalized box, and provenance.
- **Measured laptop profile:** 169 ms P50 / 189 ms P95 inference on M2 Pro CPU with 8 threads; target 200/250 ms; processed-FPS target at least 4. The native torch-MPS reference path measured 34–45 ms warm in development, but ONNX Runtime has no MPS provider and MPS is not the deployment path.
- **Failure:** missing weights, invalid manifest, unavailable provider, or degenerate output fails closed.

> These are named-laptop measurements, not ECU or product performance claims.

### 5.3 Why the detector is not enough

COCO gives `person`, `chair`, `couch`, `bag`, `phone`, `bottle`, and other everyday classes. It does **not** contain `child-seat`, `seatbelt`, `belted`, `infant`, or `driver`. Therefore:

- A person box is occupancy evidence, not age proof.
- A chair/couch box is furniture evidence, not a child-seat proof.
- A phone box near a person is a phone candidate, not proof of distraction.
- Generic detections are retained as observations; they are not silently promoted into safety claims.

### 5.4 Challengers, optional models and rejections

| Item | Decision | Reason |
|---|---|---|
| D-FINE-S | Leading challenger, not baseline | Better dev recall/seat numbers in a small head-to-head; promotion needs official weights, own export, native parity, and re-validation. |
| RTMDet-S | Deferred | No trustworthy ready artifact in the evaluated set. |
| YOLO-World | Rejected | GPL-3.0 and Ultralytics AGPL pull-in risk. |
| Objects365-derived checkpoints | Rejected | Uncleared lineage. |
| RT-DETRv4 | Track only | Immature at evaluation time. |
| Face/person identity embeddings | Banned | Requirements prohibit identity; no storage path. |
| CoreML execution provider | Blocked | Measured numerical corruption. |
| Dynamic INT8 | Rejected | Faster but failed box-parity tolerance. |
| FP16 on CPU | Rejected | No CPU speed benefit; precision risk without gain. |

### 5.5 Model governance rules

1. **Inference only.** No training, fine-tuning, LoRA, distillation, or weight updates.
2. **Manifests are the checkpoint contract.** Each artifact records source, revision, license, lineage, SHA, input profile, runtime profile, and score semantics.
3. **SHA verification fails closed.** A mismatch stops the fetch/load path.
4. **Parity gates exports.** An export that fails the box-parity gate is disabled, not silently used.
5. **Providers are allow/reject lists.** A disallowed provider raises rather than silently engaging.
6. **Thresholds live in config, not weights.** Operator, evaluation, and state thresholds are separate concepts.
7. **Model-set version is recorded** on state transitions and persisted evidence.
8. **Optional models may be absent.** The service degrades visibly; missing weights never become fake detections.

---

## 6. Datasets, footage and evaluation media

### 6.1 Core local clips

All eight core clips are local-only under `footage/`, Pexels License, landscape 1080p+ (S1 is 1280×720), and are not committed to Git.

| File | Scenario | Source | Size | Use |
|---|---|---|---:|---|
| `F1_family_ride.mp4` | Family ride, two adults front, baby rear, day | Pexels 6181413 | 7.6 MB | Golden-path candidate |
| `C1_baby_carseat.mp4` | Dad strapping baby into a child seat | Pexels 6181408 | 5.1 MB | Child seat + belt |
| `C2_child_seat.mp4` | Child in car seat, day | Pexels 6300815 | 4.9 MB | Posture / face |
| `E1_empty_cabin.mp4` | Empty parked BMW cabin pan | Pexels 3757013 | 4.0 MB | Empty-cabin negative |
| `N1_night_drive.mp4` | Driver POV at night, 40 s | Pexels 3554563 | 27 MB | Low-light test |
| `S1_seatbelt_closeup.mp4` | Hands + belt close-up | Pexels 8155681 | 1.3 MB | Belt evidence |
| `S2_belt_up.mp4` | Woman driver belts up, parked, 22 s | Pexels 5391318 | 11 MB | Belt + driver |
| `R1_rear_wide.mp4` | Four to five rear passengers in a taxi van | Pexels 38684176 | 19.3 MB | Multi-occupant |

The manifest was refetched and decode-verified on 2026-10-25. R1 is 3840×2160.

### 6.2 Robustness clips

Documented Mixkit License clips:

| File | Content | Status |
|---|---|---|
| `X1_street_crowd.mp4` | Busy street, many persons/cars/bikes | Documented; uncalibrated, no seats claimed |
| `X2_empty_office.mp4` | Busy office, about 25 people | Documented stress test |
| `X3_dog.mp4` | Dog being petted | Documented; current workspace may lack the file, which is a known test-only failure |

### 6.3 Evaluation annotations

`eval/annotations/v1.json` **is present** (4,362 bytes) and is referenced by `eval/datasets/v1.yaml`. It contains 22 hand-labeled frames from F1, R1, C2, E1 (dev) and N1, S2 (holdout): person boxes plus per-seat occupied/empty. Labels are eval-only and never used for training.

Additional local review annotations exist:

- `eval/annotations/local-c2-posture-review-2026-10-24.json`
- `eval/annotations/local-s2-belt-review-2026-10-24.json`

These are review artifacts, not a new locked benchmark.

### 6.4 External datasets

| Dataset | Access | Status | Use |
|---|---|---|---|
| SVIRO synthetic rear-seat | DFKI per-vehicle packages | Deferred: download budget | Safe with attribution “SVIRO, DFKI” |
| TICaM | DFKI request form | Request to file | Synthetic subset preferred for public demos |
| Drive&Act | Direct, TB-scale | Pull one NIR view on demand | No public raw clips without permission |
| DriveAHead | KIT email request | To file if head-pose needed | No public use |
| DMD / UTA-RLDD / YawDD / SFDDD | Form / Drive / IEEE / Kaggle | DMS-lite eval candidates | Check per-clip publish flags |

### 6.5 Media and licensing rules

- Every clip is free-license or consented-actor footage; license and source are recorded.
- Research datasets are eval-only, not redistributed, and never shown publicly beyond their terms.
- No clip is used for weight updates.
- Children in stock clips are platform model-released; non-identifying framing is preferred and should be noted in demos.
- Raw video is not stored in PLOMID. `footage/` stays on the demo machine and is gitignored.

### 6.6 Child-detection audit

`docs/child-detection-validation.md` is the authoritative child audit. Its status is **INCOMPLETE**. Key measured facts, with their limits:

- Across three short child-present clips: 97 `Yes`, 16 `Unknown`, 0 `No` in 113 child-seat observations; recall counting `Unknown` as a miss is 85.8%, Unknown rate 14.2%.
- One adult-only clip: 0 child `Yes` in 20 adult-seat observations.
- F1 development clip: 23/28 child `Yes`, 5 `Unknown`, 0/56 adult `Yes`.
- These are correlated frames from a few short clips, labeled by one reviewer; they are not population-calibrated and do not establish a production false-positive rate.
- The remaining misses are often frames where the child is blanket-covered and no person box exists. The system does not invent a child from an absent detection.

---

## 7. End-to-end pipeline

### 7.1 Capture and replay

- Replay preserves source timestamps and uses a replay anchor (default file mtime) so events align with a selected frame.
- Frame sequence numbers are monotonic.
- Decode health is explicit. Camera failure is surfaced as unhealthy/degraded; it never becomes an empty seat.
- Live camera reconnect uses bounded exponential backoff.
- A new video resets the transient hub, age cache, belt votes, and DMS state; a new session starts in PLOMID.

### 7.2 Preview versus inference

- Preview JPEGs can be produced for every captured frame, so the operator sees smooth video.
- Inference is sampled (`sample_every_n`, default 6).
- Captured, processed, and dropped counts are tracked separately.
- Empty output does not freeze the UI: the service publishes a heartbeat at most once per second and retains last-known seats (D32).

### 7.3 Quality and trust

Quality checks cover dark, bright, blur, and frozen-frame signals. Poor visibility can produce `UNKNOWN`/`OCCLUDED` rather than a confident state. The evidence clock is the frame timestamp, including replay timestamps.

### 7.4 Detection post-processing

1. Decode at a low floor (`min_decode_threshold` default 0.20).
2. Apply per-class floors, with the operator threshold as the minimum for every class.
3. Reject vehicle/cabin hallucinations that are geometrically implausible.
4. Apply class-aware NMS and duplicate suppression.
5. Retain persons and generic objects as observations.
6. Pass person evidence to tracking/state; do not promote furniture or generic objects into occupant identity.

### 7.5 Tracking

- ByteTrack-style matching with Kalman prediction.
- High threshold 0.5, low threshold 0.2, IoU match 0.3, max age 30 frames, minimum 3 hits.
- Track IDs are session-local and ephemeral; they are not identities.
- Tracks are confirmed after `min_hits`; missed tracks coast for `max_age`.
- No re-identification, face embedding, or cross-session linkage.

### 7.6 Seat association

- Seat polygons are normalized 0..1 and versioned by camera profile.
- Score = `0.7 * center_in_polygon + 0.3 * bbox_overlap_fraction`.
- If the best two seats are within the ambiguity margin (default 0.15), the track remains unassigned and downstream state becomes `UNKNOWN`.
- Track-first and seat-first association are available; the system does not force a detection into the nearest seat.
- Stable canonical IDs (`seat_1`, …) flow into state, events, PLOMID, APIs, and reports; visual labels are a separate display boundary.

### 7.7 Temporal state

See [Section 13](#13-temporal-state-and-events) for the exact rules.

### 7.8 Attributes

See [Section 11](#11-age-and-gender) and [Section 12](#12-belts-children-pose-dms-masks-and-specialist).

### 7.9 Alerts

- `belt`: occupied seat with a `Not detected` belt candidate, WARNING.
- `drowsy`, `distracted`, `phone`: DMS preview flags, WARNING.
- `unattended`: occupant continuously present past the timer without vehicle-bus context, WARNING; always worded “verify with door/ignition signals (not connected)”.
- Cooldown prevents repeated firing; WARNING+ raised/cleared transitions persist as `odc_event` rows.
- The critical fusion-gated forgotten-occupant path lives in `guardian.py` and needs parked + locked + closed bus signals; it is advisory only.

### 7.10 Persistence and publish

The service writes current seat/occupant state, events, per-frame seat rows, detections, metrics, layouts, and assistant interactions through the repository boundary. UI snapshots are bounded and transient; SQL is the durable truth.

---

## 8. Detection in detail

### 8.1 Input contract

| Stage | Value |
|---|---|
| Source pixel format | BGR from OpenCV |
| Model input | RGB |
| Resize | 640×640 |
| Resampling | PIL bilinear (OpenCV linear diverges) |
| Normalization | `/255`, float32 NCHW |
| Provider order | Manifest-approved; CUDA preferred, CPU fallback; CoreML blocked |
| Batch | Per sampled frame |

### 8.2 Output contract

- Class label (COCO name).
- Raw confidence score, retained as evidence; not a calibrated probability.
- Normalized box `[x_min, y_min, x_max, y_max]`.
- Provenance: model ID, manifest/version context, frame sequence.

### 8.3 Threshold semantics

There are three different numbers that are often confused:

| Number | Where it applies | Default |
|---|---|---|
| `detection.threshold` | Operator/console detection threshold; minimum score shown and used for every class | 0.7 |
| `eval_threshold` | Locked validation profile | 0.3 |
| `state.min_score` | Minimum person score for positive state evidence | 0.3 |

Per-class floors can tighten a class above the operator threshold. The validation and laptop profiles intentionally use 0.3, which is why documents quote both 0.3 and 0.7. Neither is a model weight.

### 8.4 What COCO does and does not give

**Useful classes:** person, chair, couch, bag, suitcase, bottle, cell phone, laptop, and other everyday objects.

**Not available:** child seat, infant, seatbelt, buckle, drowsiness, distraction, emotion, identity.

Consequences:

- A `chair` detection is a restraint/object candidate only when seat and occupancy evidence support it.
- A `person` detection is not adult/child proof.
- A `cell phone` near a person is a phone-near-body candidate, not proof of use.
- A missing `person` detection is not proof of an empty seat.

### 8.5 Measured detection results

Locked validation set, threshold 0.3, M2 Pro, RT-DETRv2-S:

| Split | Precision | Recall | F1 | Note |
|---|---:|---:|---:|---|
| Dev | .195 | .407 | .264 | Harsh box-level labels; small R1 boxes |
| Holdout | .429 | .500 | — | Night and belt clips |

These are box-level numbers on 22 hand-labeled frames. They are not product accuracy and should not be quoted as such. The seat-level 3-way metric is the product-relevant metric.

### 8.6 Failure behavior

- Missing ONNX weights → model unavailable; states `UNKNOWN`.
- Invalid manifest → fail closed.
- Unavailable provider → fall back only to an allowed provider.
- Numerically degenerate output → model unavailable; no detections invented.
- Overexposed/underexposed/blurred frames → quality signal; state may abstain.

---

## 9. Tracking in detail

### 9.1 Algorithm

- ByteTrack-style two-stage association: high-confidence detections first, then lower-confidence detections for unmatched tracks.
- Kalman filter predicts the next box.
- Greedy IoU matching with a 0.3 threshold.
- Tracks are confirmed after 3 hits and survive up to 30 missed frames.
- Each track has a session-local integer ID.

### 9.2 Why session-local IDs

- Identity is prohibited; track IDs are not person identities.
- A track can disappear and a new track can appear after occlusion.
- Events retain ephemeral track IDs for audit, but they are not stable across sessions.
- No gallery, no re-ID, no face embedding, no cross-video linkage.

### 9.3 Known limits

- Heavy occlusion, blanket coverage, and seat overlap can break tracks.
- Two persons in one seat zone can produce ambiguous association.
- DETR duplicate boxes can create twin tracks; duplicate suppression and seat-first association reduce the effect.
- Tracking does not fix a missing detection.

---

## 10. Seat allocation and calibration

### 10.1 Layout model

- A layout is a versioned set of normalized seat polygons with a camera profile.
- Seats have stable IDs (`seat_1`…), operator labels (“Seat 1”…), row/position metadata, and optional child-region hints.
- Coordinates are normalized 0..1, never hardcoded pixels.
- Validation rejects degenerate polygons, out-of-range points, duplicate IDs, bad names, path traversal, more than 12 seats, more than 32 points, and polygons below the minimum area.
- Seed layouts: standard 2-seat RHD, standard 5-seat RHD, standard 7-seat RHD. The first-run default is the 5-seat RHD.
- Seed YAML files are imported into PLOMID on first load; after that, the database row is the source of truth.

### 10.2 Automatic seat-map proposal (`auto-v1`)

When an operator starts an unknown replay without a layout:

1. Probe up to 32 evenly sampled frames with the same manifest-selected detector.
2. Accumulate person detections above the probe floor into a 12×12 normalized heatmap.
3. Add nearby median person width/height; add visible chair/couch and large bag-like furniture at reduced weight as an empty-cabin fallback.
4. Greedily pick peaks above 30% of the strongest hotspot, enforcing Chebyshev separation.
5. Person evidence owns the proposal when people are visible; furniture is only a fallback.
6. Convert peaks into clamped, globally non-overlapping x-partitions sized from nearby boxes.
7. Group regions into front-to-rear rows and assign generic one-based seat IDs.
8. Label the result `auto-v1`, never a curated calibration.
9. If no sustained person evidence exists, return `None`; the session runs uncalibrated rather than inventing seats.

`auto-v1` deliberately has no child-region hints, so child/restraint outputs stay conservative.

**Known defect:** in a supplied saved four-zone map, only one of three person boxes was assigned. The documented remedy is to start a new session with Auto and regenerate the proposal, then review it. Do not treat a saved map as validated just because it loads.

### 10.3 Calibration editor

- The Seat map page supports tracing polygons on a frozen frame.
- `/calibrate` and `/api/layouts/validate|save|apply` back the editor.
- Layouts are validated before storage; duplicate names require explicit overwrite.
- Saved layouts are versioned and stored in PLOMID, with a layout fingerprint for cache invalidation.
- Applying a layout to a running session hot-swaps the geometry for later frames and rekeys beliefs.
- Standard geometry remains a starting profile. For an installed camera, trace and validate zones before trusting seat association.

### 10.4 Association and unknown

- Association is evidence, not assignment by convenience.
- If a box straddles two seats, the system may leave `seat_id=None`.
- `UNKNOWN` is a valid product outcome; forced assignment would create flicker and false certainty.
- Seat-first association builds a candidate pool per zone and claims a track once, leaving conflicts unassigned.
- Track-first association selects the best seat only when evidence clears the ambiguity margin.

### 10.5 Operational guidance

1. Prefer a fresh Auto proposal for a new clip, then review.
2. For an installed camera, trace the actual seat zones and save a named layout.
3. Keep camera profile and layout version aligned.
4. Re-calibrate when the camera moves, the seat configuration changes, or perspective changes.
5. Do not compare seat IDs across vehicles unless the layout versions match.

---

## 11. Age and gender

There are **two separate paths**. Confusing them is the most common source of inaccurate documentation.

### 11.1 Default path: MiVOLO V2

- **Model:** `iitolstykh/mivolo_v2@53393526c220e34cdd7b722b36d22b6f9e5f4241`
- **Code/weights license:** Apache-2.0 per manifest/card; training data lineage is proprietary plus open-source and not fully enumerated.
- **Input:** paired 384×384 face/person RGB tensors, ImageNet normalization; missing face uses a normalized zero input.
- **Runtime:** local Transformers model, loaded once, CPU/CUDA/MPS selection.
- **Output:** regressed apparent-age point estimate and binary gender softmax. Neither is calibrated.
- **Smoothing:** recent observations per track; track expiry after 36 frames by default.
- **Enabled by default:** yes (`age.enabled=True`). If weights are absent, the service logs the failure and leaves classifications `UNKNOWN`; it does not fabricate.
- **Persistence:** age/gender fields appear in seat rows and frame-observation JSONB.

**Age bands** (defaults, configurable):

| Apparent age | Category |
|---|---|
| ≤ 2 | infant |
| > 2 through 11 | child |
| > 11 through 23 | unknown (abstention band) |
| > 23 | adult |

The six-year margin around the configured 17-year boundary deliberately leaves adolescents and near-boundary adults unresolved. It is a precision-oriented development heuristic, not a population-calibrated classifier.

**Per-seat child aggregation:**

- If any current track is `child`/`infant`, the seat is `child=Yes`.
- If child and adult tracks coexist, the seat category is `mixed`.
- If all tracks are adult, `child=No`.
- Otherwise `child=Unknown`.
- A one-frame child estimate is suppressed when another current track in the same seat estimates adult; repeated child evidence can resolve it.
- `child_confidence` is `None`; a point estimate is not a calibrated class probability.

**What “age” means here:** apparent age from a camera, not chronological age, not identity, not a legal or medical attribute. Do not quote exact years as fact.

### 11.2 Gender within MiVOLO

- Binary softmax labels are mapped to `APPARENT_MAN` / `APPARENT_WOMAN`.
- Minimum score default 0.55; below that, `UNKNOWN`.
- Smoothed over recent observations.
- Per-seat output is `UNKNOWN` unless all contributing tracks agree.
- Confidence is explicitly labeled “MiVOLO V2 softmax score; uncalibrated”.
- This is appearance categorization, not identity, not sex assigned at birth, and not a safety input.

### 11.3 Experimental CLIP appearance branch

- **Status:** experimental, default off (`detection.enable_gender_appearance=False`).
- **Model:** OpenCLIP ViT-B-32, LAION weights, eval-only.
- **Prompts:** “a man” / “a woman”.
- **Gates:** only when MiVOLO is absent; every 10th inference frame; box area at least 0.04; skip child seats and `child=Yes`; confidence floor 0.7; per-seat cache.
- **Intended containment:** display-only, transient, never aggregated, never a safety input.
- **Legal exposure:** biometric categorization under GDPR special-category data, EU AI Act, and BIPA-type regimes. D30 records the owner’s accepted risk and a first-legal-review removal option.

**Critical discrepancy:** the module docstring and D30 say “never persisted”, but the service writes the CLIP-derived `gender`/`gender_conf` fields into the same `rows` list that `write_frame_observation(..., rows, ...)` stores in `odc_frame_observation.payload`. If the branch is enabled, those fields can appear in PLOMID JSONB. The branch is therefore **not privacy-cleared** in its current form. Keep it disabled. If it is ever enabled, strip those fields before persistence or record a new explicit decision and privacy review. This is documented, not silently resolved.

### 11.4 What to say at an event

> “Age and gender are apparent estimates from a local model, with an explicit unknown band. We do not claim exact age, and the experimental gender branch is off. The default age/gender path is used for candidate occupant context, not identity or safety.”

---

## 12. Belts, children, pose, DMS, masks and specialist

### 12.1 Seatbelt strap detection

**Method:** classical geometry, no learned belt classifier.

1. Build a torso ROI from the person/seat geometry.
2. Convert to grayscale; run edge detection and probabilistic Hough/LSD segments.
3. Keep long, high-contrast diagonal segments.
4. Merge collinear segments into parallel pairs (a strap has two edges).
5. Fall back to an elongated bright-band check when edges are weak.
6. Confirm temporally per seat with K-of-M voting.

**Outputs:** `Detected`, `Not detected`, `Unknown` strap visibility.

**Hard limits:**

- `Detected` means a strap-like line pair is visible; it does **not** prove buckle engagement.
- `Not detected` is never emitted from one frame; it requires temporal evidence.
- A geometric clear-torso result is diagnostic; only a validated belt detector or buckle-bus signal can support a reliable negative.
- Dark clothing, night, soft webbing, occlusion, and unusual routing cause false negatives.
- Camera never overrides a vehicle-bus buckle signal.
- No airbag, pretensioner, or restraint actuation path exists.

**Alert wording:** belt alerts are candidate warnings, not “unbelted” assertions.

### 12.2 Child and child-seat logic

Current code truth:

- **MiVOLO age categories drive child classification.** The default path is described in Section 11.
- **Relative cabin-size voting is diagnostic-only.** `state.py` hardcodes `is_small = False`; the 25%-below-peer rule does not create a child candidate or event. It retains per-seat ratios in an in-memory diagnostic history. This is a change from the D33 description and the older `algo_model/attributes-pose-belt.md` text.
- **The Grounding DINO specialist is optional and default off** (`detection.child_specialist_enabled=False`). It can produce child/child-seat candidates, but its weights are O365-lineage and eval-only, and it produced a rear-facing-seat candidate on a reviewed empty cabin frame.
- **Empty child seats and occupied child seats are different questions.** A child-seat object candidate does not prove a child is present.
- **Child-seat recognition as a validated classifier is unsupported.** The active COCO checkpoint has no child-seat class.
- **No child classification from a single small box.** Apparent size is not age evidence.
- **No child classification from a missing detection.** Blanket-covered children remain `Unknown` rather than being invented.

**Child-seat candidate policy:** expose as candidate evidence only when corroborated by occupant, configured-region, or independent restraint evidence; never promote an open-vocabulary box to confirmed `YES` on its own.

### 12.3 Pose

- **Model:** MediaPipe Pose Landmarker Lite, local task bundle.
- **Output:** 33 landmarks with visibility, mapped from crop back to frame coordinates.
- **Use:** posture candidates, seat-row enrichment, overlays; not identity, not age, not safety.
- **Cadence:** enabled by default when the bundle is present; `every_n=2`; up to 3 targets.
- **Failure:** absence of a landmark is not evidence of an absent limb; pose degrades to unavailable.
- **Licensing:** bundle commercial terms are `REVIEW_REQUIRED`; per-card review before distribution.
- **Not supported:** breathing, heartbeat, medical posture diagnosis.

### 12.4 Driver monitoring (DMS)

- **Model:** MediaPipe Face Landmarker, 478 points, blendshapes, head-pose matrices.
- **Signals:** eye-blink ratio over time (drowsiness proxy), jaw opening (yawn proxy), head yaw/pitch (distraction proxy), plus phone-near-body from the generic detector.
- **Scope:** driver-seat-only preview, per the requirements.
- **Default:** disabled (`dms.enabled=False`); when enabled, `every_n=3`, up to 2 faces.
- **Output:** candidate flags with evidence; they feed alerts and the seat table, never protection logic.
- **Privacy:** no face identity, no embeddings, no stored face crops; only transient per-frame points leave the module.
- **Limits:** preview-grade; not a certified drowsiness detector; not medical; not a substitute for steering/eye-gaze sensors.
- **Licensing:** bundle terms `REVIEW_REQUIRED`.

### 12.5 Masks

- **Fast path:** classical bbox/ROI refinement (GrabCut-style), default for live.
- **Precise path:** SAM ViT-B, optional, box-prompted, background single-slot worker.
- **Cost:** roughly 5–7 seconds per frame on CPU in the measured probe; therefore not in the critical path.
- **Provenance:** precise masks are tagged with their source frame sequence; stale masks are not applied to a newer frame.
- **Limits:** a mask does not classify a person, prove occupancy, or prove a safety state.

### 12.6 Specialist (Grounding DINO)

- **Model:** Grounding DINO SwinT OGC, 662 MiB local checkpoint, eval-only O365 lineage.
- **Role:** bounded child/child-seat candidate evidence with separate occupant and child-seat prompts.
- **Execution:** one bounded pending job; obsolete frames coalesced; results carry session/source-frame provenance.
- **Default:** off.
- **Output handling:** duplicate prompt boxes removed; candidate boxes are not painted over the live image as raw open-vocabulary labels; decisions are retained in seat rows, diagnostics, and SQL.
- **Evidence:** local C1 clip returned child/infant-seat candidates; the same model returned a rear-facing-seat candidate on a reviewed empty cabin frame, so candidate promotion needs corroboration.

### 12.7 Fusion, guardian and alerts

- **Fusion interface:** COVESA VSS-style signal names with source and timestamp; stale signals are explicit and block safety-relevant logic. The interface and tests exist; no vehicle bus is connected in the demo.
- **Camera vs bus:** the camera never overrides the bus.
- **Guardian:** implemented as a standalone, gated advisory module. Its rule fires only when parked + locked + closed bus signals and a temporally confirmed occupied seat hold continuously for the dwell; anything missing disarms silently. It is not wired to a live vehicle bus in the demo and it actuates nothing.
- **Alerts:** candidate-grade, cooldown-gated, persisted at WARNING+ transitions. The live list is transient and scoped to one video.
- **Unattended alert:** camera-only timer path is worded as “verify with door/ignition signals (not connected)”. It is not a safety decision.

---

## 13. Temporal state and events

### 13.1 State model

| State | Meaning |
|---|---|
| `OCCUPIED` | Strong, sustained positive occupant evidence. |
| `LIKELY_OCCUPIED` | Positive evidence that has not yet met the full occupancy dwell/confirmation. |
| `EMPTY_CANDIDATE` | Healthy camera, visible seat, sustained zero person evidence. A negative candidate, not a certified vacancy. |
| `UNKNOWN` | Insufficient, ambiguous, occluded, or failed evidence. |
| `STALE` | Evidence is older than the stale horizon. |

Visibility states: `VISIBLE`, `PARTIALLY_OCCLUDED`, `OCCLUDED`, `OUT_OF_VIEW`, `CAMERA_UNHEALTHY`, `NOT_EVALUATED`.

### 13.2 Rules (defaults, versioned)

| Rule | Default | Effect |
|---|---:|---|
| `min_score` | 0.3 | Minimum person score for positive evidence |
| `min_positive_frames` | 3 | Positive frames before occupancy |
| `occupy_dwell_s` | 1.0 | Dwell before `LIKELY_OCCUPIED` |
| `confirm_extra_s` | 2.0 | Extra time before `OCCUPIED` |
| `release_misses_likely` | 5 | Misses before leaving `LIKELY_OCCUPIED` |
| `release_misses_unknown` | 15 | Misses before falling to `UNKNOWN` |
| `empty_dwell_frames` | 25 | Sustained empty evidence before `EMPTY_CANDIDATE` |
| `stale_after_s` | 5.0 | Evidence age before `STALE` |
| `cooldown_s` | 10.0 | Transition debounce |
| `ambiguity_margin` | 0.15 | Seat association near-tie threshold |
| `size_window` / `size_min_votes` | 5 / 3 | Diagnostic size history; not child evidence |

### 13.3 The safety rule

`EMPTY_CANDIDATE` requires **positive empty evidence**: healthy camera + visible seat + sustained zero person evidence. Missed detections, occlusion, camera failure, and stale data yield `UNKNOWN`/`STALE` — never `EMPTY`, never `OCCUPIED`.

This is the single most important design rule in the product. It is why the system sometimes appears to “abstain a lot”: false certainty about an empty seat is worse than an honest unknown.

### 13.4 Evidence clock

The evidence clock is the frame timestamp, including replay timestamps. Events remain aligned with a selected frame during recap. A replay anchor (default file mtime) keeps event timing reproducible.

### 13.5 Events

Event types include:

- `SESSION_STARTED`, `SESSION_ENDED`
- `SEAT_PRESENCE_CHANGED`
- `CHILD_CANDIDATE_DETECTED`, `CHILD_CANDIDATE_CLEARED`
- `OCCUPANT_TRACK_STARTED`, `OCCUPANT_TRACK_ENDED`
- `ATTRIBUTE_BECAME_UNKNOWN`
- `ALERT_RAISED`, `ALERT_CLEARED`
- `CAMERA_DEGRADED`, `MODEL_UNAVAILABLE`, `DATABASE_DEGRADED`
- `CONFIG_CHANGED`
- `UNATTENDED_OCCUPANT_CANDIDATE`
- `FUSION_SIGNAL_STALE`

Each event carries session, seat, prior/new state, reason codes, evidence summary, frame sequence, occurred-at timestamp, config version, and model-set version.

**D34 deduplication:** the service passes a cumulative event list to the hub each frame. The hub deduplicates by stable `event_id` when merging the bounded transient history; SQL event writes remain per new engine event. Without this, candidate transitions appeared many times in the live timeline.

### 13.6 Reconciliation

Current state can be reconciled from event history. This is deliberate: the SQL event stream is the audit trail, not just a UI log. Stale checks expire beliefs so a dead stream does not look alive.

### 13.7 What the relative-size path does now

The code computes per-seat person-box/seat-zone ratios and keeps a 60-frame window. It compares a currently observed seat to peer seats in memory for diagnostics. It **does not** set `is_small=True`, does not add a child candidate, and does not emit a child event from size. The older D33 text describes the former behavior. Current child classification is MiVOLO-driven.

---

## 14. Persistence and schema

### 14.1 PLOMID facts

- **Database:** PLOMID 0.1.0 release binary, PG wire v3, SCRAM auth, loopback.
- **Schema:** `plomid_odc`.
- **Schema version:** 8.
- **Domain tables:** 17.
- **Relations:** 12 enforced foreign keys plus 5 documented soft links.
- **Live snapshot at 2026-10-27:** 45 rows across the ODC tables; session `ses-1792941062`; camera `cabin0`; `HEALTHY`; 25.0 fps.
- **Rule:** new schema changes require a new migration (`V009+`). Never edit applied `V001`–`V008`.

### 14.2 Table catalog

| Table | Migration | Purpose |
|---|---|---|
| `odc_session` | V001 | Session metadata, mode, source, build/config versions, status |
| `odc_model_artifact` | V001 | Checkpoint identity, SHA, license, dataset terms, approval flag |
| `odc_camera_status` | V001 | Camera health, session binding, frame rate, last error |
| `odc_seat_state_current` | V001 | Current per-seat presence, confidence, visibility, reasons |
| `odc_event` | V001 | State/alert/health/config events with provenance |
| `odc_occupant_state_current` | V002 | Current per-track occupant attributes |
| `odc_inference_metric` | V002 | Latency, queue, frames, memory telemetry |
| `odc_validation_run` | V002 | Validation run metadata |
| `odc_validation_result` | V002 | Per-metric validation results |
| `odc_seat_layout` | V003 | Versioned seat geometry and camera profile |
| `odc_video_asset` | V003 | Video fingerprint/size/mtime metadata; no video bytes |
| `odc_video_seatmap` | V003 | Layout bound to a video fingerprint |
| `odc_probe_cache` | V003 | Cached auto-layout/probe results |
| `odc_analytics_snapshot` | V003 | Cached analytics keyed by layout/config hash |
| `odc_frame_detection` | V005 | Per-frame class, confidence, box vector, seat/track |
| `odc_frame_observation` | V006 | Complete per-frame seat rows, pose, specialist decisions |
| `odc_assistant_interaction` | V007 | Prompt, answer, provider, model, context, status |

`odc_schema_version` is migration bookkeeping owned by `odc/migrate.py`; `odc_relation_catalog` is a view added by V008 and rendered by `/relations`.

### 14.3 What is stored, and what is not

**Stored:**

- Session/source/config/model metadata.
- Current per-seat and per-occupant state.
- State/alert events with evidence timestamps and provenance.
- Per-frame seat rows, pose, specialist decisions.
- Per-frame detection class, confidence, normalized box, optional seat/track.
- Health and processing metrics.
- Versioned layouts and video fingerprints.
- Validation/probe/analytics records.
- Assistant prompt, answer, and exact structured context.

**Not stored:**

- Raw video or image pixels.
- Face images or face embeddings.
- Identity embeddings or identity galleries.
- A semantic occupant embedding invented from a detection box.
- Airbag/actuator commands.

### 14.4 JSONB and vector semantics

- `odc_frame_observation.payload` has `api_schema_version: odc.seat-observation.v1`, `seat_rows`, `pose`, and `specialist_decisions`.
- `odc_frame_detection.bbox` is `VECTOR(FLOAT32,4)` containing `[x_min, y_min, x_max, y_max]`. It is **geometry evidence**, not a semantic embedding.
- Face identity embeddings are prohibited by requirements.
- The active detector does not expose a validated occupant-semantic feature tensor; ODC does not pretend a box is one.
- Vector search/HNSW is a PLOMID capability, not a claim that ODC performs identity matching.

### 14.5 Relations

- **12 enforced FKs** cover session ownership of events, state, metrics, validations, layouts, video assets, detections, observations, assistant interactions, and related rows.
- **5 documented soft links** cannot be hard constraints:
  1. `odc_inference_metric.model_id` → `odc_model_artifact.model_id` (telemetry may precede artifact registration).
  2. `odc_occupant_state_current.seat_id` → `odc_seat_layout.seats[].seat_id` (seat IDs live inside layout JSON; geometry can be replaced while a session runs).
  3. `odc_event.seat_id` → `odc_seat_layout.seats[].seat_id` (historical events must outlive layout edits; the seat ID is a snapshot value).
  4. `odc_event.track_id` → `odc_occupant_state_current.track_id` (tracks are ephemeral; events retain the ID after the current-state row is gone).
  5. `odc_frame_detection.track_id` → `odc_occupant_state_current.track_id` (evidence keeps the track ID even when tracking expires or the occupant is reassigned).
- `/api/schema` exposes `odc.schema-model.v1`; `/relations` renders the same model.
- `information_schema.key_column_usage` omits FK local columns in PLOMID; read the catalog view instead.

### 14.6 PLOMID SQL capabilities and quirks

**Verified by `scripts/smoke_plomid.py` (10/10 PASS on PLOMID 0.1.0, port 55433):**

- PG wire v3 with psycopg3; `SELECT version()` returns PLOMID 0.1.0.
- PK, UNIQUE, NOT NULL, DEFAULT, CHECK.
- NUMERIC, BOOLEAN, TIMESTAMPTZ, JSONB, TEXT[]; `DEFAULT now()`.
- FK enforcement: valid and NULL pass, invalid rejected, atomic.
- `ON CONFLICT DO NOTHING/DO UPDATE` with `RETURNING`.
- JSONB `->>` and casts.
- `VECTOR(FLOAT32,n)` literal, l2/cosine/dot, `<->` Top-K, HNSW; dimension mismatch rejected.
- Parameterized SQL; no compatibility percentage is claimed.

**Quirks that shape the code:**

| Quirk | Consequence |
|---|---|
| FKs only in `CREATE TABLE` | Add FKs by rebuilding a table (`_fk` staging), not `ALTER TABLE ADD CONSTRAINT`. |
| View expressions persist lossily | `odc_relation_catalog` returns `pg_get_constraintdef()` text plus rules; the console parses it. |
| DDL + DML in one transaction can lose DML | Apply schema changes separately from data writes. |
| `information_schema.key_column_usage` omits FK local columns | Use the catalog view or `pg_get_constraintdef()`. |
| `LIMIT` needs a literal | The SQL allowlist has two reviewed literal sites. |
| Bare `version` is reserved | Quote or rename columns. |
| No bare `float` alias | Use `DOUBLE PRECISION`/`REAL` as appropriate. |
| No TLS in this build | Loopback-only development; re-verify before any pilot network exposure. |

**Migration design rule for V008-style FK work:** FKs only in `CREATE TABLE`; create `_fk` staging tables; one committed statement per transaction; copy data; swap tables; never edit applied migrations.

### 14.7 Retention and restart behavior

- Durable state/events/layouts/metrics remain after restart.
- The in-memory frame recap is bounded and lost on restart by design.
- Raw video remains on disk and is not deleted by the database.
- `odc purge --dir reports --older-than-days 30` handles age-based report deletion.
- Exact image replay after restart requires the original video source or an external media store; ODC does not store pixels.
- Retention schedules and DPIA are legal-owner gaps, not code claims.

### 14.8 Test data safety

- Integration fixtures use `test-cabin0`.
- Tests must delete only test-session/test-camera rows.
- The live `cabin0` camera row and live sessions are preserved.
- Do not run destructive cleanup against the demo database.

---

## 15. APIs and console

### 15.1 API surface

| Endpoint | Purpose |
|---|---|
| `GET /api/live/state` | Current live snapshot; `odc.live.v1` |
| `GET /api/live/counts` | Occupancy/child/alert counts |
| `GET /api/seats` | Seat rows, including persisted observation context |
| `GET /api/events` | Session event history |
| `GET /api/frames/history` | Bounded frame recap |
| `GET /api/frames/detections` | Per-frame detection evidence |
| `GET /api/frames/live-evidence` | Live frame evidence for the inspector |
| `GET /api/alerts` | Active/raised alerts |
| `GET /api/health` | Service/model/database/camera health |
| `GET /api/fleet` | Fleet/camera summary if configured |
| `GET /api/dossier` | Evidence dossier export |
| `GET /api/layouts/current` | Active layout |
| `POST /api/layouts/validate` | Validate a proposed layout |
| `POST /api/layouts/save` | Save a named layout |
| `POST /api/layouts/apply` | Apply a layout to the current session |
| `POST /api/service/start` | Start a session with source/layout/threshold |
| `POST /api/service/stop` | Stop the session |
| `POST /api/service/upload` | Upload a video for replay |
| `GET /api/service/sources` | Available video sources |
| `GET /api/service/status` | Service status |
| `GET /api/service/preview.jpeg` | Current preview JPEG |
| `GET /api/schema` | Live schema/relations model; `odc.schema-model.v1` |
| `POST /api/assistant/ask` | Explicit operator assistant request |
| `GET /api/assistant/history` | Persisted assistant interactions; `odc.assistant-history.v1` |
| `GET /sse/stream` | SSE notification stream; triggers snapshot polling |
| `GET /frame.jpeg`, `/frame-thumb.jpeg`, `/crop/{id}.jpeg` | Transient visual evidence |

### 15.2 Console screens

| Screen | Route | What it answers |
|---|---|---|
| **Live / Replay** | `/live` | Stream, seat map, counts, alerts, selected-seat evidence inspector, frame recap |
| **Seat map** | `/seats` | Cabin schematic, true zones, attributes, calibration editor |
| **Timeline** | `/timeline` | Every seat and alert event on one time base, with frame jump |
| **Diagnostics** | `/diagnostics` | Throughput, latency, model/camera/database health |
| **Validation** | `/validation` | Recorded evaluation runs and holdout discipline |
| **Config** | `/config` | Active geometry, thresholds, model manifest, versions |
| **Data model** | `/relations` | Enforced FKs and documented soft relations |
| **Fleet** | `/fleet` | Vehicles/cameras when configured |
| **Dossier** | `/dossier` | Evidence export for review |
| **Calibration** | `/calibrate` | Layout tracing/validation flow behind the Seat map editor |

The live workspace receives `/api/live/state` snapshots through SSE-triggered polling. Seat buttons, chips, map zones, tables, alerts, and the frame recap all use the same canonical `seat_id`. Selecting a seat highlights the button, schematic zone, calibrated overlay zone, and table row while opening the evidence inspector.

### 15.3 SSE

- The server exposes `/sse/stream`.
- SSE has been verified over a real uvicorn process (`data:` event received).
- A hang with FastAPI `TestClient`/`httpx` on Windows is a test-environment issue, not a production SSE failure; it is listed in Section 16.8.

### 15.4 Operator workflow

1. Confirm PLOMID is running on the host.
2. Start the console (`odc serve` or `start.bat`).
3. Upload/select a clip.
4. Accept or edit the auto seat map; calibrate if needed.
5. Start the session.
6. Watch live states, counts, alerts, and DMS/age/belt/pose evidence.
7. Select a seat to inspect reasons and frame evidence.
8. Use the timeline to jump to transitions.
9. Query the data model / SQL or export the dossier.
10. Stop the session; durable evidence remains in PLOMID.

---

## 16. Validation, benchmarks and test status

### 16.1 Validation plan

- Dataset v1: 22 hand-labeled frames from F1×5, R1×5, C2×3, E1×3 (dev) and N1×3 (night), S2×3 (belt) (holdout).
- Labels: person boxes plus per-seat occupied/empty; box precision about 0.05; IoU 0.5.
- Eval threshold frozen at 0.3.
- Dev tunes thresholds; holdout runs once per model/config change. Failures gate release and are never tuned against.
- Methodology fixes may re-run with documentation; the holdout caught the night-EMPTY gap this way.
- Eval labels are never training labels.

### 16.2 Metrics

- Detection: greedy-IoU precision/recall/F1 per class (person today).
- Seats: three-way correct / wrong / abstain. `UNKNOWN`/`STALE` are abstains, never counted as correct or wrong.
- Unknown rate reported alongside accuracy, never hidden.
- Latency P50/P95, frames, per-split gates.
- Dense-run stability: transitions, time-to-stable, chain tests.

### 16.3 Measured baselines

M2 Pro, RT-DETRv2-S, threshold 0.3:

| Split | Detection | Seat-level |
|---|---|---|
| Dev | P .195 / R .407 / F1 .264 | .816 correct, 0 wrong, .184 unknown |
| Holdout | P .429 / R .500 | .667 correct, 2 night wrong (documented gap) |

Box-level numbers are harsh because of 0.05-precision hand labels and small R1 boxes. Seat-level 3-way is the product metric. Duplicate-FP flood is tracked as R14.

**Do not say** “81.6% accurate” without the seat-level context and the holdout caveat. The honest sentence is: “On the locked dev set, 81.6% of seat observations were correct, 18.4% abstained, and none were wrong; on holdout, 66.7% were correct with two night errors.”

### 16.4 Child audit results

See Section 6.6. The headline is 85.8% recall with 14.2% Unknown across 113 child-seat observations in three short clips, 0/20 child `Yes` on one adult-only clip, and status **INCOMPLETE**. One reviewer, correlated frames, no population calibration, no production-readiness claim.

### 16.5 Acceptance gates

`eval/datasets/v1.yaml` freezes:

| Gate | Threshold |
|---|---:|
| Person precision, dev | ≥ .15 |
| Person recall, dev | ≥ .35 |
| Seat correct, dev | ≥ .75 |
| Seat unknown rate, dev | ≤ .30 |
| Holdout recall | ≥ .25 |

Any gate flipping PASS→FAIL blocks release.

### 16.6 Reports and regression

- `odc validate --split dev|holdout --out reports/<name>` writes `report.json`, `index.html`, and a failure gallery; optional `--record-to-db` fills the console.
- `--compare` diffs reports.
- Reports are developer artifacts, not the application’s durable store.
- Historical reports remain useful for diagnosis; do not treat stale reports as current acceptance evidence.

### 16.7 What v1 does not cover

- SVIRO box-level AP.
- Belt/blanket/misuse positives (R15).
- NIR/fisheye buckets (R16).
- Multi-vehicle splits.
- Event false-alarms/hour at scale.
- Population-level age calibration.
- Production functional-safety or cybersecurity validation.

### 16.8 Test status and known environment-only failures

Focused tests and live probes exist for the paths described here; specific pass/fail status must be re-checked with the commands in Section 23.5. The following are known environmental/test-only issues in this workspace and must not be presented as product failures:

| Symptom | Cause / scope |
|---|---|
| Missing `sh` | Windows environment lacks the shell expected by a test helper |
| `read_text()` cp1252 failure on `live.html` | Windows default encoding in a test, not the app |
| Missing `footage/X3_dog.mp4` | Robustness clip not present locally |
| `.env` leaks `ODC_MODE=LIVE` | Test isolation issue from local dotenv |
| Windows backslash SQL scan | Test path normalization |
| SSE `TestClient` hang | Windows httpx/TestClient interaction; real uvicorn SSE verified |
| Port-1 outage classified `TIMEOUT` not `CONNECTION` | Environment-specific error classification |

**Verification principle:** use focused tests and live probes for claims; do not cite an unrelated full-suite failure as a product defect, and do not hide a real failure behind “environmental”.

### 16.9 Reference performance profile

`configs/profiles/laptop-m2pro-cpu.yaml`:

- Hardware: Apple M2 Pro, 16 GB, macOS 15.4, 12 CPU (8 performance + 4 efficiency).
- Model: `rtdetrv2-s-coco`.
- Provider: CPUExecutionProvider; FP32.
- Input: 1920×1080 BGR → PIL-BILINEAR 640×640 /255.
- Threads: 8 intra-op, 1 inter-op.
- Sampling: every 6 frames.
- Threshold: 0.3.
- Targets: inference P50 ≤ 200 ms, P95 ≤ 250 ms, processed FPS ≥ 4.
- Measured: P50 169 ms, P95 189 ms.
- Rejected: CoreML (zeroed logits), dynamic INT8 (box-parity failure), FP16 CPU (no benefit, precision risk).

These are named-laptop measurements. They are not ECU, thermal-soak, or production targets.

---

## 17. Privacy, security and legal

### 17.1 Trust boundaries

```text
cabin camera ──*──> ODC process ──*──> PLOMID (loopback SQL, SCRAM)
     │                  │                    │
   frames             logs                 console (loopback HTTP)
 (transient)      (no frames/secrets)      structured history APIs
     │                                       │
   footage/                           states + pose + chat context
 (video files)                          in SQL / JSONB / VECTOR
```

### 17.2 Data minimization

- No raw video or image pixels in PLOMID.
- No face identity, face embeddings, or identity gallery.
- No semantic embedding invented from a detection box.
- Face landmarks and DMS points are transient per frame.
- Frame previews are bounded in-memory JPEGs.
- Only structured, non-image evidence is durable.
- Assistant context is structured current-state evidence, not raw video.

### 17.3 Identity prohibition

Identity recognition is banned by requirements. The system does not:

- match faces,
- store face embeddings,
- link tracks across sessions,
- infer names,
- expose a “driver identity” beyond the layout’s seat role.

“Driver” is a seat role (`seat_1` in the standard RHD layout), not a person identity. Visual left/right is never treated as driver identity.

### 17.4 Assistant privacy

- Default provider: local Ollama at `http://127.0.0.1:11434`, model `mistral`.
- Hosted option: Pollinations OpenAI-compatible endpoint, model `openai-fast`.
- The assistant is called only by an explicit operator request to `/api/assistant/ask`; it never runs in the frame loop.
- It receives structured current-state evidence, not raw video.
- It cannot create detections, change seat state, or write AI-generated claims into PLOMID state.
- Interactions (prompt, answer, exact context) are persisted in `odc_assistant_interaction` for audit.
- Hosted processing sends the question and context outside the instance. Set `ODC_ASSISTANT__ENABLED=false` where external processing is prohibited.
- Out-of-scope questions are refused with an explicit message.

### 17.5 Gender, age and special-category risk

- MiVOLO gender is an apparent categorization of a person; it can be special-category data under GDPR and can trigger EU AI Act/BIPA-type obligations.
- Age is personal data; apparent age is not exact age.
- The CLIP branch is experimental, default off, and has a documented persistence gap (Section 11.3).
- Legal review, DPIA, consent/HMI notice, and retention schedules remain owner/legal gaps.
- The safest demo posture: keep the CLIP branch off; describe age/gender as candidate context; do not aggregate or export it as a protected-characteristic profile.

### 17.6 Security controls

- DSN via environment only; no secrets on disk in committed config.
- Redaction helpers for logs/errors.
- Parameterized SQL only; f-string SQL is allowlisted and tested.
- Bounded payloads (evidence ≤ 2000 chars); oversize rejected, not truncated.
- Dependency license scan: zero GPL-family; ban-test in suite.
- `pip-audit`: zero known vulnerabilities at the recorded scan date.
- SQL injection, credential leak, and unsafe payload paths are covered by tests.
- No TLS in the current PLOMID build; loopback-only development. Re-verify before any network exposure.

### 17.7 Legal / compliance position

- Prototype: engineering demonstrator.
- Not ISO 26262 / ASPICE / ISO/SAE 21434 certified.
- No certification claim in code, UI, or dossier.
- DPIA, retention schedules, consent/HMI notice, cross-border plan: **GAP**, owner legal.
- Supplier rights: per-checkpoint BOM and redistribution review still required.
- Production entry criterion: the production-gap table has no open GAP rows without an accepted, dated mitigation.

---

## 18. Operations

### 18.1 Install and start

```sh
python3 -m venv .venv
./.venv/bin/pip install -r requirements.lock
./.venv/bin/python scripts/fetch_models.py
./.venv/bin/python scripts/export_onnx.py
./.venv/bin/python scripts/prepare_assets.py
export ODC_DSN="postgresql://plomid:plomid@127.0.0.1:5432/plomid"
./.venv/bin/python -m odc.cli bootstrap
./.venv/bin/python -m odc.cli selftest
./.venv/bin/python -m odc.cli serve
```

PLOMID must be running on the host. `odc serve` provisions the schema automatically; `bootstrap` is a diagnostic/alternative path. The console idles until Start unless `--auto-start` is used.

### 18.2 Windows

```bat
setup-assets.bat
start.bat
```

`start.bat` uses PLOMID on `:5432` if it is already running; otherwise it starts `..\plomid\target\release\plomid-server.exe`, provisions the ODC schema, and serves the console. Environment knobs include `ODC_DSN`, `ODC_PYTHON`, `PLOMID_BIN`, `PLOMID_DATA`, and `ODC_NO_SETUP`.

### 18.3 Docker

```sh
docker build -t plomid-odc .
docker compose up --build
```

The console runs on `:8080` and reaches the host PLOMID through `host.docker.internal:5432`. Footage/models/configs are mounted read-only. Weights are not baked into the image.

### 18.4 Health and readiness

- `odc health [--probe-db]` reports service/model/database/camera health.
- `odc selftest` is the readiness bundle; exit 0 means go.
- `odc providers` reports the ORT provider set and per-model allow/reject reasons.
- `odc profile` runs the named hardware profile.

### 18.5 Migrations

- Additive-only; no down-migrations by design.
- `odc serve` applies migrations automatically; `odc bootstrap` remains available.
- New changes: create `V009__...`, never edit V001–V008.
- FK additions: rebuild with `_fk` staging tables, FKs only in `CREATE TABLE`, one committed statement per transaction.
- V008 is the current schema head: five missing FKs added by rebuilds plus the relation catalog view. The same change set added `/api/schema` and detector-manifest recording; those are application changes, not migration contents.

### 18.6 Recovery and rollback

- Camera/model/SQL failures degrade visibly; retry budget is 3, then surface.
- Restart is clean; the two-session path is tested.
- Rollback: stop → restore the PLOMID data-directory snapshot → previous image → `odc selftest`.
- Backup/restore roundtrip was drilled on 2026-10-13.
- Rollback never deletes snapshots; retention policy governs them.
- Disk-full is a documented residual risk.

### 18.7 Purge and retention

- `odc purge --dir reports --older-than-days 30` deletes old report artifacts.
- PLOMID rows are durable until explicitly purged by the operator/retention process.
- Raw `footage/` files are outside PLOMID and must be managed separately.
- No automatic deletion of live evidence is claimed.

### 18.8 Troubleshooting

| Symptom | Cause → fix |
|---|---|
| Startup provisioning fails | PLOMID not listening on `127.0.0.1:5432`, wrong user/password, or wrong `ODC_DSN` |
| UI shows DISCONNECTED | Service not started or crashed; check serve logs |
| All seats STALE instantly | Clock mismatch; staleness uses the evidence clock; check replay anchor |
| FK errors on first write | Session not started; manual flows must `start_session` first |
| Slow inference | Check provider/threads; use the laptop profile; measure before changing |
| Corrupt/partial downloads | Fetch/export verify SHA; re-run the fetch |
| CoreML selected but outputs nonsense | Blocked by manifest; ensure the provider is rejected |
| Night results poor | Known gap; NIR/fisheye work is scheduled, not solved |

### 18.9 Data safety rules

- Use `test-cabin0` for test fixtures.
- Delete only test-session/test-camera rows.
- Preserve the live `cabin0` camera row and live sessions.
- Do not run destructive migrations or cleanup on the demo database.
- Do not edit applied migrations.

---

## 19. Limitations and roadmap

### 19.1 Current limitations (authoritative summary)

- Standard layout geometry is a starting profile; installed-camera calibration is required for reliable seat assignment.
- Generic COCO detection does not establish adult versus child.
- Child-seat, belt, and posture outputs are candidates with explicit reasons.
- Missing/occluded/failed frames do not prove an empty seat.
- Raw video is not stored in PLOMID; the in-memory recap is lost on restart.
- No airbag, pretensioner, OEM buckle-bus, or safety-critical actuation path exists.
- Production qualification, functional safety, cybersecurity, and OEM target-hardware validation remain external work.
- Night/NIR and fisheye coverage are gaps.
- Age is apparent, with an abstention band; child recall is incomplete.
- Gender appearance is uncalibrated; the CLIP branch is experimental and not privacy-cleared.
- DMS is preview-grade and disabled by default.
- The validation set is 22 frames plus eight local clips; it is not population evidence.
- The assistant is operator-only and can send structured context externally if configured.

### 19.2 OEM acceptance snapshot

| Area | Status |
|---|---|
| Occupancy all seats | MET on the demo/dev set; holdout night errors |
| Child seats | PARTIAL; classifier unsupported |
| Belt routing/misuse | GAP; candidate + fusion only; needs buckle bus and misuse footage |
| Occlusion/blankets/clutter | PARTIAL; UNKNOWN-gating tested |
| Sunlight/glare/temp soak | GAP; needs vehicle/chamber time |
| Night | GAP; holdout caught it; NIR path R16 |
| Camera position variants | GAP; needs target-vehicle geometry/clips |
| Fisheye undistort | PARTIAL; fisheye-aware deferred |
| Calibration wizard | PARTIAL; editor exists, target-camera validation remains |
| Real-time bounds | MET on laptop profile; ECU targets untested |
| Fault recovery | MET in the tested matrix |
| Upgrade/rollback | MET; V1→V2 and snapshot drill |
| OTA model packs | PARTIAL; signing specified, HSM flow GAP |
| eCall/restraint interfaces | GAP; contracts only |
| Validation dossier | MET; dossier export |
| Privacy/GDPR | MET for metadata-only design; DPIA GAP |
| ASPICE / ISO 26262 / 21434 | GAP; no claims |

### 19.3 Roadmap priorities

1. **Calibration quality:** target-camera seat-zone validation, auto-layout review UX, association-defect fixes.
2. **Night/NIR and fisheye:** qualified data, undistortion, low-light evaluation.
3. **Child recall:** independent subjects, second reviewer, more child-seat scenes, calibrated operating point.
4. **Belt evidence:** misuse footage, buckle-bus fusion, negative-state validation.
5. **DMS:** driver-seat validation, false-alarm rate, legal review.
6. **Schema/relations:** keep V009+ additive; expand catalog tests.
7. **Production gaps:** DPIA, SBOM, TARA, target hardware, safety case, OTA.
8. **Commercial pilot:** named OEM clips, acceptance gates, support/retention plan.

### 19.4 What would change the status

A capability moves from `PARTIAL` to `VERIFIED` only with a focused test or deterministic contract check and a documented operating envelope. It moves toward production only when the production-gap table has no open GAP rows without an accepted, dated mitigation. Marketing language never changes status.

---

## 20. Known documentation/code discrepancies

These are recorded deliberately. Do not silently “fix” one side without updating the other and, where required, a decision record.

| # | Document says | Code/evidence says | Current truth and action |
|---|---|---|---|
| 1 | CLIP gender is “never persisted” (module docstring, D30) | Service puts `gender`/`gender_conf` into `rows`; `write_frame_observation(..., rows, ...)` persists them in JSONB | **Privacy gap.** Keep `enable_gender_appearance=False`. If enabling, strip fields before persistence or record a new decision/privacy review. |
| 2 | D33 and `algo_model/attributes-pose-belt.md` describe an active relative-size child candidate | `state.py` hardcodes `is_small=False`; no size-based child event | **Code wins.** Relative size is diagnostic-only; MiVOLO drives child classification. Update stale docs before quoting D33 as current behavior. |
| 3 | OEM acceptance matrix says calibration wizard is GAP | `odc/layouts.py`, `/calibrate`, layout validation/save/apply exist | **Doc drift.** Current status is PARTIAL: editor exists, target-camera validation remains. |
| 4 | `algo_model/persistence-recap.md` cites V001–V004 | Schema is V008 with 17 tables and frame/assistant tables | **Doc drift.** Use Section 14 and the migration catalog. |
| 5 | Some docs quote detection threshold 0.3 as the product threshold | Operator default is 0.7; validation/profile use 0.3; state min score is 0.3 | **Context matters.** State which threshold is meant. |
| 6 | Architecture says “7 screens” | Console nav has 8 screens plus dossier/calibration routes | **Doc drift.** Use Section 15.2. |
| 7 | `odc.example.yaml` looks like the full config | It is minimal; `odc/config.py` contains feature flags and defaults | **Code wins.** Use config.py or the Config page for the full default set. |
| 8 | D30 says CLIP has high precision/low coverage | Probe was 13 crops, 6 confident calls, 7 abstentions | **Signal, not evaluation.** Do not present it as accuracy evidence. |

**Rule for future changes:** if a behavior changes, update the code, the focused test, this guide’s discrepancy register, and the relevant decision record in the same change. Never edit an applied migration.

---

## 21. Event and pitch Q&A

### 21.1 The 30-second pitch

> “ODC turns an existing cabin camera into per-seat occupant evidence. It runs pretrained models at the edge, tracks people, associates them to seat zones, and reasons over time — so a missed frame becomes ‘unknown’, not a false empty seat. Every state is explained and stored in PLOMID SQL. It does not identify people, does not train on your data, and does not claim safety certification. It is a demonstrator and pilot platform, ready for a bounded OEM pilot.”

### 21.2 The 2-minute demo narrative

1. “Here is a normal cabin clip. We upload it; no cloud.”
2. “ODC proposes seat zones from the video, then the operator reviews them. Geometry is versioned, not guessed per frame.”
3. “Live: occupancy, counts, age/child candidates, belt candidate, pose. Every state has a reason.”
4. “Select a seat: see confidence, visibility, frame evidence, and why it abstained.”
5. “Timeline: every transition, with frame jump and model/config version.”
6. “Data model: PLOMID schema, 12 enforced relations plus documented soft links.”
7. “Assistant: optional, operator-triggered, local by default.”
8. “What we do not claim: exact age, buckle certainty, identity, breathing, certification.”

### 21.3 Q&A

**Q1. What is ODC in one sentence?**
A camera-only cabin-occupant demonstrator that produces per-seat states with explicit uncertainty and persists the evidence in PLOMID SQL.

**Q2. Is it production-ready?**
No. It is an engineering demonstrator and pilot platform. Production requires target-camera calibration, NIR/fisheye coverage, legal review, OEM hardware validation, and functional-safety/cybersecurity processes.

**Q3. Is it certified?**
No. There is no ISO 26262, ASPICE, or ISO/SAE 21434 certification claim anywhere.

**Q4. Does it train on our data?**
No. Pretrained inference only. No training, fine-tuning, LoRA, distillation, or weight updates. Eval labels never update weights.

**Q5. Which detector do you use?**
RT-DETRv2-S COCO, Apache-2.0, SHA-pinned, ONNX Runtime CPU. D-FINE-S is a measured challenger, not the baseline.

**Q6. Why COCO? It has no child seat or belt.**
Correct. COCO is the generic person/object evidence layer. Child classification comes from MiVOLO age candidates; belt comes from geometric strap detection; child-seat candidacy comes from an optional eval-only specialist. None of these is promoted to certainty.

**Q7. How accurate is occupancy?**
On the locked dev set, 81.6% of seat observations were correct, 18.4% abstained, and none were wrong. On holdout, 66.7% were correct with two night errors. These are small-set engineering numbers, not product accuracy.

**Q8. Why so many unknowns?**
Because a missed detection must not become an empty seat. `UNKNOWN` is a safety feature. Calibration, night/NIR, and occlusion handling reduce it over time.

**Q9. Can it tell exact age?**
No. MiVOLO gives an apparent-age point estimate. We use infant ≤2, child ≤11, unknown above 11 through 23, adult >23 by default. Exact age is unsupported.

**Q10. Can it reliably detect a child?**
Partially. The child audit found 85.8% recall with 14.2% Unknown across 113 child-seat observations in three short clips, and 0/20 child `Yes` on one adult-only clip. It is not population-calibrated and status is INCOMPLETE.

**Q11. Can it prove a seatbelt is fastened?**
No. It detects visible strap-like geometry. It cannot prove buckle engagement, correct routing, or restraint performance. A buckle bus is needed for a reliable negative/positive.

**Q12. Can it detect breathing or heartbeat?**
No. Camera-only breathing/heartbeat is out of scope and must remain unclaimed.

**Q13. Does it identify people or recognize faces?**
No. Identity, face embeddings, and cross-session re-ID are prohibited. Track IDs are session-local and ephemeral.

**Q14. Is gender detection on?**
The default MiVOLO path produces apparent-gender estimates as candidate context. The experimental CLIP branch is off and is not privacy-cleared. We do not treat gender as identity or a safety input.

**Q15. Is this GDPR-compliant?**
The design is metadata-only, no raw video, no identity, and local by default. But DPIA, retention schedules, consent/HMI notice, and legal review are still owner gaps. Do not claim compliance certification.

**Q16. Does it need special cameras?**
It works with file replay and live camera sources. Fisheye/NIR coverage is partial/gap; target-camera calibration is required. It does not yet claim qualified NIR or fisheye performance.

**Q17. Does it work at night?**
It runs, but night is a documented gap. The holdout caught two night errors. NIR/low-light work is on the roadmap, not solved.

**Q18. What hardware does it need?**
A CPU baseline runs on a laptop: about 169 ms P50 / 189 ms P95 inference on the named M2 Pro profile, with 8 threads and 640×640 input. ECU/SoC targets are not qualified.

**Q19. How does seat mapping work?**
The video is probed, person/furniture hotspots become candidate zones, and the operator reviews them. An installed camera should be calibrated by tracing zones. Ambiguous association stays `UNKNOWN`.

**Q20. What happens if the camera is blocked or the model fails?**
The system shows degraded/unavailable and moves seats to `UNKNOWN`/`STALE`; it never turns failure into an empty seat.

**Q21. Does it control airbags or pretensioners?**
No. It has no actuation path. It is advisory-only.

**Q22. Why PLOMID instead of PostgreSQL?**
PLOMID is the single database and SQL boundary for the product; it supports the types, JSONB, vectors, FKs, and upserts ODC needs. The dialect was probed, not assumed. It has quirks, which are documented and tested.

**Q23. Can it run offline?**
Yes. Local inference, local PLOMID, local Ollama assistant. A hosted assistant is opt-in and can be disabled where external processing is prohibited.

**Q24. Does the assistant leak data?**
Local Ollama does not leave the machine. The hosted Pollinations provider receives the question and structured context outside the instance; disable it where that is not allowed. The assistant never writes state claims into PLOMID.

**Q25. What is the strongest differentiator?**
Temporal, explainable, uncertainty-first seat state with a single SQL evidence boundary — not just a detector demo. It shows why it abstained, not only what it guessed.

**Q26. What can you demo today?**
Replay upload, auto seat proposal, live occupancy, age/child candidates, belt/pose/DMS previews, evidence inspector, timeline, data model, SQL persistence, dossier, and optional assistant.

**Q27. What should we not say?**
Do not say certified, production-ready, exact age, breathing/heartbeat, proven belt fastening, identity recognition, emotion recognition, airbag control, 100% accuracy, or “trained on your data”.

**Q28. What is the commercial path?**
Demonstrator → bounded OEM pilot with target clips and calibration → production-gap closure → OEM qualification. The pilot is the honest near-term product.

### 21.4 Objection handling

| Objection | Response |
|---|---|
| “Your detection numbers are low.” | Box-level labels are harsh and small; the product metric is seat-level 3-way with abstention. Dev is 81.6% correct/0 wrong/18.4% unknown; holdout is 66.7% with two night errors. We do not hide the unknown rate. |
| “COCO is generic; this cannot work.” | COCO is one evidence layer. Age comes from MiVOLO, belt from geometry, child-seat from an optional specialist, and all are gated. We do not claim COCO solves child-seat or belt. |
| “Why not fine-tune?” | The program is pretrained-only by requirement. Fine-tuning would add legal, data-rights, and validation burden without solving the camera-domain gaps first. |
| “Why so many UNKNOWNs?” | False empty is worse than unknown. Occlusion, night, and calibration quality drive abstention; those are the roadmap. |
| “Can it prove belt status?” | No. It detects visible strap geometry. Buckle certainty needs a buckle bus or a validated belt sensor. |
| “Can it identify the driver?” | No. Driver is a seat role, not an identity. Identity is prohibited. |
| “Is gender detection ethical?” | The default path is apparent-gender candidate context; the experimental CLIP branch is off and not privacy-cleared. Legal review is required, and the feature can be disabled. |
| “Why PLOMID?” | Single SQL truth, JSONB/vector support, FKs and upserts, no sidecars. The dialect is probed and quirks are documented. |
| “Can we deploy next month?” | A bounded pilot, yes, with target clips, calibration, and legal review. Production qualification, no. |
| “What about night?” | It is a known gap; the holdout caught two errors. NIR and low-light validation are scheduled, not solved. |
| “What about occlusion/blankets?” | The system abstains when evidence is missing; it does not invent a child or an empty seat. More occlusion data is needed. |
| “How do you avoid false empties?” | `EMPTY_CANDIDATE` requires healthy camera + visible seat + sustained zero evidence; missing/stale/occluded evidence stays `UNKNOWN`/`STALE`. |
| “Is this just a dashboard?” | No. The dashboard is a view over a temporal state engine, SQL evidence model, validation harness, and model-governance layer. |
| “Can it run on an ECU?” | Not qualified. The CPU laptop baseline is measured; ECU/SoC targets are untested. |
| “What about privacy?” | No raw video, no identity, local by default. DPIA and retention schedules remain legal-owner gaps. |

### 21.5 Numbers cheat sheet

| Claim | Number | Context |
|---|---:|---|
| Detector CPU P50 / P95 | 169 / 189 ms | M2 Pro, 8 threads, 640×640, named profile |
| Detector native MPS reference | 34–45 ms warm | Development reference, not deployment |
| D-FINE-S CPU dev | ~120 ms | Challenger head-to-head, not baseline |
| Dev seat correct / wrong / unknown | .816 / 0 / .184 | Locked 22-frame set, 3-way |
| Holdout seat correct / wrong | .667 / 2 night | Same set, holdout discipline |
| Dev detection P/R/F1 | .195 / .407 / .264 | Box-level, harsh labels |
| Holdout detection P/R | .429 / .500 | Box-level |
| Child audit recall / unknown | 85.8% / 14.2% | 113 child-seat observations, 3 short clips, INCOMPLETE |
| Child audit adult false positives | 0/20 | One adult-only clip |
| Schema | V8 | 17 domain tables |
| Relations | 12 enforced + 5 soft | Live catalog |
| Sample cadence | every 6th frame | About 4 inference FPS on 25 fps source |
| Default operator threshold | 0.7 | Validation/profile use 0.3 |
| State min score | 0.3 | Positive person evidence |

### 21.6 Phrases to use and avoid

**Use:**

- “candidate,” “apparent,” “estimated,” “abstains,” “unknown,” “engineering demonstrator,” “pilot platform,” “measured on named hardware,” “documented gap.”
- “The system prefers an honest unknown over a false certainty.”

**Avoid:**

- “certified,” “production-ready,” “safety-grade,” “exact age,” “detects breathing,” “proves seatbelt,” “identifies,” “recognizes faces,” “emotion detection,” “airbag control,” “100% accurate,” “trained on your data.”

---

## 22. Glossary

| Term | Meaning |
|---|---|
| **ODC** | Occupant Detection / cabin state platform in this repository. |
| **PLOMID** | The sole database and SQL boundary; schema `plomid_odc`. |
| **Seat ID** | Stable canonical key such as `seat_1`; not a person identity. |
| **Track ID** | Session-local ephemeral person track; not an identity. |
| **`OCCUPIED`** | Sustained strong positive occupant evidence. |
| **`LIKELY_OCCUPIED`** | Positive evidence before full confirmation. |
| **`EMPTY_CANDIDATE`** | Positive empty evidence; not a certified vacancy. |
| **`UNKNOWN`** | Insufficient/ambiguous/occluded/failed evidence. |
| **`STALE`** | Evidence older than the stale horizon. |
| **`auto-v1`** | Automatic candidate seat-map proposal, never a curated calibration. |
| **MiVOLO** | Local age/apparent-gender estimator. |
| **RT-DETRv2-S** | Active COCO detector. |
| **D-FINE-S** | Measured detector challenger. |
| **Grounding DINO** | Optional open-vocabulary child/child-seat candidate specialist; eval-only. |
| **SAM** | Optional precise-mask model; slow, background only. |
| **DMS** | Driver-monitoring preview: blink/jaw/head-pose/phone candidates. |
| **K-of-M** | Temporal voting requiring multiple consistent observations. |
| **Temporal state** | State derived from evidence over time, not one frame. |
| **Evidence clock** | Frame timestamp used for staleness and event alignment. |
| **Dossier** | Structured evidence export for review. |
| **Soft relation** | A logical link that cannot be an enforced FK; documented in the schema model. |
| **`_fk` staging table** | Rebuild pattern for adding FKs under PLOMID’s `CREATE TABLE`-only constraint. |
| **FP/AQ** | False positive / abstention quality context; not a certification metric. |

---

## 23. Source map and verification

### 23.1 Primary documents

| Document | What it is for |
|---|---|
| `README.md` | Quickstart, honesty contract, commands, configuration entry point |
| `docs/odc/ODC_ARCHITECTURE.md` | Locked architecture and layer responsibilities |
| `docs/odc/ODC_REQUIREMENTS.md` | Buyer asks, non-negotiables, out-of-scope, AISIN traceability |
| `docs/odc/ODC_ONTOLOGY.md` | State/attribute vocabulary |
| `docs/odc/ODC_MODEL_SELECTION.md` | Model portfolio, licenses, rejections, benchmark gates |
| `docs/odc/ODC_VALIDATION_PLAN.md` | Dataset, splits, metrics, measured baselines |
| `docs/odc/ODC_DECISIONS.md` | D1–D34 engineering decisions and evidence |
| `docs/odc/ODC_SECURITY_PRIVACY.md` | Threat model, data flow, privacy posture |
| `docs/odc/ODC_SQL_CAPABILITIES.md` | Verified PLOMID SQL behavior and quirks |
| `docs/odc/ODC_OPERATIONS.md` | Install, run, health, recovery, compatibility |
| `docs/odc/ODC_OEM_ACCEPTANCE.md` | OEM acceptance matrix and gaps |
| `docs/odc/ODC_PRODUCTION_GAP.md` | Production-gap plan and entry criterion |
| `docs/odc/ODC_RISKS.md` | R1–R17 risk register |
| `docs/odc/ODC_ROBUSTNESS.md` | Robustness and degradation behavior |
| `docs/odc/ODC_BUILD_PLAN.md` | Phase/build history; can lag decisions |
| `docs/child-detection-validation.md` | Child audit, age-boundary analysis, INCOMPLETE status |
| `footage/README.md` | Clip sources, licenses, dataset access |
| `algo_model/*.md` | Per-area implementation notes; check against code for drift |

### 23.2 Code map

| Path | Responsibility |
|---|---|
| `odc/service.py` | Service loop, pipeline order, feature gates, hub snapshots |
| `odc/capture.py` | Replay/live sources, timestamps, reconnect |
| `odc/runtime.py` | ONNX loading, provider selection, preprocessing |
| `odc/state.py` | Temporal state engine, dwell/stale, relative-size diagnostics |
| `odc/tracking.py` | ByteTrack-style tracker |
| `odc/seats.py` | Layout model, association, geometry |
| `odc/autolayout.py` | Auto seat-map probe |
| `odc/layouts.py` | Layout validation/editor backend |
| `odc/seat_store.py` | PLOMID-backed layout/video/probe/analytics store |
| `odc/age.py` | MiVOLO age/gender, smoothing, row application |
| `odc/demographics.py` | Experimental CLIP gender probe |
| `odc/belt.py` | Geometric belt detection and temporal voting |
| `odc/pose.py` | MediaPipe pose |
| `odc/dms.py` | MediaPipe face and DMS candidates |
| `odc/segmentation.py` | Fast masks and SAM path |
| `odc/specialist.py` | Grounding DINO candidate worker |
| `odc/alerts.py` | Rule-based candidate alerts |
| `odc/guardian.py` | Parked-guardian advisory path |
| `odc/fusion.py` | VSS-style vehicle-signal interface |
| `odc/persistence.py` | SQL repository and schema model |
| `odc/migrate.py` | Migration runner/schema version |
| `odc/console/app.py` | FastAPI routes, SSE, console context |
| `odc/console/templates/` | Live, seats, timeline, diagnostics, validation, config, relations, fleet |
| `odc/validate.py`, `odc/eval.py` | Metrics and evaluation |
| `odc/ollama.py` | Explicit assistant boundary |
| `scripts/fetch_models.py` | SHA-verified model fetch |
| `scripts/export_onnx.py` | Detector ONNX export/parity |
| `scripts/prepare_assets.py` | MediaPipe bundles and footage |
| `scripts/smoke_plomid.py` | PLOMID SQL capability probe |

### 23.3 Manifests and migrations

| Path | Contents |
|---|---|
| `models/manifests/rtdetrv2-s-coco.json` | Active detector identity, SHA, license, parity |
| `models/manifests/dfine-s-coco.json` | Challenger identity, license, provenance |
| `models/manifests/grounding-dino-swint-ogc.json` | Specialist identity/lineage |
| `models/manifests/mivolo-v2.json` | Age/gender model identity, SHA, licenses |
| `sql/migrations/V001__odc_foundation.sql` | Session, model, camera, seat state, event |
| `sql/migrations/V002__odc_occupant_metrics_validation.sql` | Occupant state, metrics, validation |
| `sql/migrations/V003__odc_seat_store.sql` | Layouts, video assets, seatmaps, probes, analytics |
| `sql/migrations/V004__odc_layout_version_col.sql` | Layout version column |
| `sql/migrations/V005__odc_frame_detection_evidence.sql` | Per-frame detection vector |
| `sql/migrations/V006__odc_frame_seat_observations.sql` | Per-frame seat observation JSONB |
| `sql/migrations/V007__odc_assistant_interaction.sql` | Assistant interaction history |
| `sql/migrations/V008__odc_foreign_keys.sql` | FK rebuilds + relation catalog |

### 23.4 Tests and evidence

| Path | Evidence |
|---|---|
| `tests/unit/test_service.py` | Pipeline/service behavior |
| `tests/unit/test_seat_geometry.py` | Association/layout geometry |
| `tests/unit/test_age.py` | MiVOLO age categories and row logic |
| `tests/unit/test_belt.py` | Belt geometry/voting |
| `tests/unit/test_dms.py` | DMS candidates |
| `tests/unit/test_demographics.py` | CLIP gender gates |
| `tests/unit/test_layouts.py` | Layout validation |
| `tests/unit/test_detector.py` | Runtime/detector contract |
| `tests/unit/test_capture.py` | Capture behavior |
| `tests/integration/test_console_app.py` | Console/API integration |
| `tests/integration/test_child_detection_*.py` | Child path regressions |
| `eval/datasets/v1.yaml` | Locked dataset and gates |
| `eval/annotations/v1.json` | Locked hand labels |
| `reports/` | Generated validation and analysis artifacts (developer evidence) |

### 23.5 Re-verification checklist

Run these before quoting the guide at a new event or after a model/config upgrade:

```sh
# 1. Readiness
python -m odc.cli selftest
python -m odc.cli health --probe-db

# 2. Live schema/relations
curl http://127.0.0.1:8080/api/schema
curl http://127.0.0.1:8080/relations

# 3. Model/provider state
python -m odc.cli providers

# 4. Focused tests
pytest -q tests/unit/test_service.py tests/unit/test_seat_geometry.py
pytest -q tests/unit/test_age.py tests/unit/test_belt.py tests/unit/test_dms.py

# 5. Validation (one split at a time; do not tune on holdout)
python -m odc.cli validate --split dev --out reports/dev-$(date +%Y%m%d)
python -m odc.cli validate --split holdout --out reports/holdout-$(date +%Y%m%d)

# 6. SQL capability probe after a PLOMID upgrade
python scripts/smoke_plomid.py
```

**Checklist for a claim:**

- Is it `IMPLEMENTED`, `VERIFIED`, `PARTIAL`, or `UNSUPPORTED`?
- Is the evidence from code/tests or only a doc?
- Is it measured on named hardware and a named dataset?
- Does it expose an abstention/unknown rate?
- Does it accidentally imply identity, certification, exact age, or belt certainty?
- Is the privacy/assistant/gender path consistent with the event’s policy?

### 23.6 Live snapshot recorded for this version

| Item | Value |
|---|---|
| PLOMID | 0.1.0, schema V8 |
| ODC domain tables | 17 |
| Relations | 12 enforced + 5 soft |
| Live rows | 45 at the 2026-10-27 snapshot |
| Session | `ses-1792941062` |
| Camera | `cabin0`, `session_id NULL`, `HEALTHY`, 25.0 fps |
| Active detector | RT-DETRv2-S COCO, Apache-2.0, SHA-pinned |
| Pre-guide baseline commit | `f073a35` |

Treat the row count and live session as a snapshot, not a constant. Re-run the checklist before quoting them.

---

## 24. Document control

| Item | Value |
|---|---|
| Owner | ODC engineering |
| Version | 1.0 |
| Created | 2026-10-27 |
| Verification basis | Repository docs, manifests, code, tests, and a live PLOMID 0.1.0 probe |
| Update rule | Update when behavior, schema, model, validation, privacy, or console changes |
| Migration rule | New schema changes require V009+; never edit applied V001–V008 |
| Claim rule | Label capabilities; no production or certification claim; no training claim |
| Review trigger | Model swap, threshold change, schema change, privacy change, OEM pilot, legal review |

**Bottom line:** ODC is a credible, explainable cabin-occupant demonstrator with a real temporal state engine and a real SQL evidence boundary. Its strongest asset is honesty about uncertainty. Its remaining work is calibration, domain coverage, child/belt validation, and the OEM production process — not a bigger marketing claim.
