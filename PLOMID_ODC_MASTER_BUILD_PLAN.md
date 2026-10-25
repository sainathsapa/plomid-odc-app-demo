# PLOMID ODC — Master Build Plan

**Project:** PLOMID Occupant Detection Camera (ODC) / Cabin Occupant Intelligence  
**Plan status:** 🔒 LOCKED v1.0 — authoritative implementation roadmap, implement one phase at a time. Frozen 2026-10-13 after regulatory/competitor/model/edge/footage research. Changes only via dated decision records in `docs/odc/ODC_DECISIONS.md`; no silent scope edits.
**Amendment log:** Python-only stack · interactive enterprise console (HTMX+Alpine+ECharts) · RT-DETRv2 baseline flip · AGPL-import ban · TRT pin gate · VSS/KUKSA fusion · Uptane packs · named SoC matrix · CDK · SD-102 dossier · DMS-lite + parked-guardian · evaluator tour + benchmark card · footage-first (Pexels core set ✅) · laptop reference target.  
**Decision date:** 2026-10-13  
**Database rule:** PLOMID only, accessed only over its documented SQL protocol  
**Model rule:** Pretrained public/open-weight models only; zero-shot/inference only; no fine-tuning or training in this programme unless the product owner explicitly changes this constraint in a future decision record.

---

## 0. How Codex must use this plan

This file is the master specification. The person driving Codex will ask it to implement **one named phase at a time**. Do not implement the whole roadmap when asked for one phase.

### Mandatory instruction for every phase

1. Read this entire file and the existing repository's contribution instructions, architecture, build plan, SQL/protocol documentation, tests, and package/workspace conventions.
2. Audit the current implementation before editing. Identify existing features and reuse them; do not duplicate existing PLOMID code or create parallel architecture without evidence.
3. Identify the exact phase requested, its prerequisites, acceptance criteria, and out-of-scope work. If prerequisites are missing, implement only the safe prerequisite work that belongs to the requested phase and report the blocker; do not silently skip or pretend prerequisites are satisfied.
4. Write a brief implementation plan tied to existing files and components. Then implement that phase end-to-end, including tests, docs, telemetry, and failure handling required by the phase.
5. Maintain repository style, dependency policy, language boundaries, public APIs, and CI conventions. Avoid broad refactors unrelated to the requested phase.
6. Update the durable project plan/status docs with `DONE`, `IN PROGRESS`, `BLOCKED`, or `NOT STARTED`. A task is `DONE` only when its acceptance criteria have evidence. Never mark future work done merely because its interface exists.
7. Run targeted tests first, then the relevant full verification/build commands if practical. State exact commands and outcomes. Do not claim tests ran if they did not run.
8. Report: files changed; design choices; tests and results; limitations; open blockers; current phase status; next recommended phase. Do not begin the next phase automatically unless asked.
9. Do not invent PLOMID SQL syntax, SQL types, SQL functions, wire-protocol behavior, vector operators, transaction semantics, indexing support, or performance. Verify each capability in the actual repository/docs or through integration tests before using it.
10. Do not make up detections, performance figures, model benchmark outcomes, OEM requirements, or certification. Any simulated/demo output must be explicitly labelled **SIMULATED**.

### Suggested reusable Codex request

> Read `PLOMID_ODC_MASTER_BUILD_PLAN.md` in full and inspect the current repository and existing docs before making changes. Implement **Phase N — [phase title] only**. Follow the phase scope and acceptance criteria exactly. Use PLOMID exclusively through its documented SQL protocol for all durable application data. Do not add another database or use internal PLOMID APIs/storage. Use only pretrained/open-weight inference; do not train or fine-tune models. Do not assume model support for cabin-specific labels until benchmarked on the validation set. Update the build-plan status and relevant docs, run the phase tests and applicable repository verification, and report concrete evidence, failures, limitations, and next steps. Do not implement later phases.

---

## 1. Product goal and positioning

Build a credible, modular, edge-first **cabin occupant intelligence** platform around the existing vehicle cabin camera. It should evolve beyond a simple object detector into a measurable stream of occupant and cabin state, events, health signals, and validation evidence.

The system should be able to:

- Ingest one or more video sources supported by the target vehicle/ECU platform.
- Detect and track visible people and generic objects using approved pretrained models.
- Associate visible occupants with configurable seat regions when the camera geometry makes this possible.
- Estimate human body landmarks/posture where keypoints are sufficiently visible.
- Maintain temporal occupant-state estimates instead of changing state on every individual frame.
- Represent uncertainty explicitly (`UNKNOWN`, `OCCLUDED`, `NOT_VISIBLE`, `UNSUPPORTED`, `STALE`) and avoid converting missing evidence into a negative result.
- Generate structured state transitions/events with timestamps and provenance.
- Persist current state, event history, configuration, model metadata, and operational diagnostics in PLOMID **through SQL only**.
- Use PLOMID vector values and vector-search facilities through SQL only when the real PLOMID SQL implementation supports the needed operations and when an embedding has a justified use.
- Provide an engineering console with live/replay views, seat map, state timeline, SQL-backed history, diagnostics, and benchmark results.
- Run offline and recover cleanly from camera, model, SQL-connection, or process failures.
- Be portable across hardware by separating model/runtime adapters from the business/state logic.

### Product positioning

**Proposed name:** PLOMID ODC — Occupant & Cabin State Platform.

**Core proposition:** convert camera observations into persistent, queryable, auditable *state and event data* at the edge, with PLOMID as the sole database and SQL as the only application/database integration boundary.

Do not market the prototype as a production-certified occupant-classification or occupant-protection system. The early product is an **engineering demonstrator and pilot platform** until an OEM-specific validation, functional-safety, cybersecurity, privacy, and production process has been completed.

---

## 2. Non-negotiable architecture decisions

### 2.1 PLOMID is the only database

All durable application data lives in PLOMID. The application accesses it solely through PLOMID's documented SQL protocol.

- Use a standard SQL client/driver only if actual PLOMID wire-protocol and dialect compatibility are confirmed by repo docs and a smoke test. Do not assume every PostgreSQL feature is supported merely because a PostgreSQL-compatible wire protocol is exposed.
- Use parameterized SQL for every value. Do not concatenate untrusted values into SQL statements.
- Schema migrations, queries, health checks, transactions, prepared statements, bulk inserts, JSON fields, vector data types/operators/indexes, and upserts must be based on verified PLOMID capabilities. Create capability tests when needed.
- No SQLite, PostgreSQL, MySQL, DuckDB, Redis, MongoDB, ORM-managed local database, embedded database, or alternate database for runtime, development, or integration testing.
- Unit tests may use in-memory fake repository interfaces. Integration and end-to-end tests must use an actual PLOMID test instance/environment.
- Memory queues, ring buffers, and temporary cache files are allowed only for transient processing, bounded replay, model artifacts, and explicitly retention-limited media. They must not become a second durable system of record.
- Do not call internal PLOMID functions, internal Rust APIs, storage engines, file formats, or private IPC as a shortcut. The boundary is SQL.
- If a database/network outage occurs, use a bounded queue/backpressure policy and visibly degrade. Do not silently persist the truth elsewhere. If temporary buffering is enabled, it must be bounded, documented, encrypted where appropriate, and cleared/reconciled according to a recovery policy.

### 2.2 Pretrained inference only; no training

The first complete demonstrator must not require model training, fine-tuning, custom-head training, LoRA, distillation, or training infrastructure.

- Prefer public pretrained model checkpoints and deterministic post-processing/state logic.
- Do not implement or invoke training scripts, optimizers, train/validation loops for weight updates, or fine-tuning pipelines.
- Evaluation-only annotations and validation datasets are allowed. They may be used to measure models, tune deterministic thresholds, and compare architectures, but never to update learned weights.
- Keep thresholds/configuration changes separate from model weights and record them in versioned configuration.
- Do not claim a generic pretrained model reliably detects child seats, children, seat belts, or other in-cabin-specific classes until a measured, domain-relevant evaluation proves it.
- If available pretrained models cannot support a requested label to an acceptable level, expose that capability as `UNSUPPORTED` or `UNKNOWN` and document the evidence. Do not quietly train a replacement.

### 2.3 Modular inference architecture

Separate these layers:

1. **Capture:** camera/video input, timestamps, frame identity, reconnect logic.
2. **Frame pipeline:** resize/color conversion/normalization, bounded buffers, sampling and frame-drop policy.
3. **Model runtime:** model loading, inference provider selection, preprocessing/postprocessing adapters.
4. **Perception:** detections, keypoints, masks/embeddings when useful, raw confidence scores.
5. **Tracking and seat association:** stable short-lived track identity, seat-region geometry, association uncertainty.
6. **Temporal state engine:** state machines, smoothing, dwell times, event generation, stale/unknown handling.
7. **Persistence adapter:** SQL-only queries through PLOMID; no model-specific logic inside the persistence layer.
8. **Application API/console:** status, visualization, queryable state and event history.
9. **Validation and observability:** replay/evaluation harness, metrics, logs, model/runtime diagnostics.

The model framework must not leak into the product-facing state schema. Replacing D-FINE with another detector must not force database schema changes for every model output detail.

### 2.4 Hardware portability

Keep portable interfaces for CPU, NVIDIA CUDA/TensorRT, Intel OpenVINO, AMD/ROCm or MIGraphX where supported, Apple Core ML, and other available execution providers. Do not promise every accelerator works merely because ONNX Runtime has a provider; each provider/package/model/hardware combination must be tested. ONNX Runtime documents a range of execution providers and explicitly distinguishes production from preview availability: https://onnxruntime.ai/docs/execution-providers/.

Start with the hardware actually available to the developer and record it. Provide CPU fallback only where it meets the configured target. Do not ship an untested provider path as supported.

### 2.5 Privacy by default

- Process locally by default. No cloud vision/API dependency in the normal inference path.
- Do not store continuous raw video in PLOMID.
- Default media retention is off; if temporary frame capture is needed for debugging, use explicit opt-in, access control, encryption, a retention limit, and a clear delete operation.
- Do not create face recognition, persistent person identity, biometric identity, or identity embeddings as part of ODC.
- Never use embeddings as a covert occupant identity database.
- Avoid storing face crops, full-resolution frames, or unnecessary precise body keypoint histories when aggregate state suffices.
- Provide configurable redaction for diagnostic exports.

### 2.6 Safety and uncertainty

Every perception output is evidence with a confidence score and provenance, not ground truth. The temporal engine must use explicit states and transitions. The system must never infer “empty” from “not detected” when the image is occluded, stale, corrupted, or outside model coverage.

The prototype must not directly control airbags, seat-belt pretensioners, occupant-protection actuators, or safety-critical warning outputs. Any production integration must go through the OEM's independent functional-safety and validation process.

---

## 3. Model and algorithm strategy — no-training baseline

The right selection is a **benchmark-backed model portfolio**, not one model forced to perform every task. Published public benchmark scores are useful to shortlist candidates, but they measure generic datasets and hardware conditions—not cabin accuracy. The final selection must be recorded after testing on fixed, labelled evaluation clips representative of the intended camera, vehicle geometry, lighting, seat types, and occupants.

### 3.1 First-choice candidate matrix

| Function | Default candidate to evaluate first | Second candidate / fallback | Purpose and constraints |
|---|---|---|---|
| Generic person/object detection | **RT-DETRv2-S, COCO pretrained** | D-FINE-S/M/L as challengers (`_coco` checkpoints only); RT-DETRv4 only if a stable checkpoint/export path is verified | D-FINE's official model zoo reports COCO AP of 48.5 for S, 52.3 for M, 54.0 for L, and 55.8 for X; the repository states its latency benchmarks use a T4 GPU, batch size 1, FP16, TensorRT 10.4. The RT-DETR repository reports RT-DETRv2-S at about 48 AP and larger models up to about 54 AP in its published table under its own benchmark setup; the repository also notes RT-DETRv4 was released later, so verify checkpoint maturity and deployment parity before adding it. These are public reference figures only and must not be used as local performance claims or compared blindly across different model/export setups. Sources: https://github.com/Peterande/D-FINE and https://github.com/lyuwenyu/RT-DETR. Start with RT-DETRv2-S (COCO-only lineage, cleaner checkpoint license) for the compact baseline, then compare D-FINE-S/M (`_coco` checkpoints only, never Objects365-derived without separate clearance) and RT-DETRv2-M on the same in-cabin clips and target device. Pin TensorRT/provider versions and re-validate mAP after every export (deformable-attention ops are version-sensitive). Select by measured in-domain performance, license, export reliability, runtime, and memory—not headline AP alone. |
| Open-vocabulary / unknown-class localization | **Grounding DINO**, only as a secondary low-rate candidate | Another permissively licensed open-vocabulary detector if evaluated and license-cleared | Try fixed prompts for “child car seat”, “infant car seat”, “seat belt across torso”, “person”, “bag”, and relevant cabin objects. It may help localize unfamiliar concepts but is not a reliable attribute classifier. The code repository is Apache-2.0; checkpoint/data license must be reviewed separately. Source: https://github.com/IDEA-Research/GroundingDINO. |
| Pose / body keypoints | **RTMPose-S or RTMPose-M** from MMPose | MediaPipe Pose Landmarker (Lite/Full) | RTMPose provides published COCO pose metrics and throughput figures, but those depend on the detector, hardware, resolution, and deployment path. MMPose repository is Apache-2.0; audit the selected config/checkpoint and associated dataset terms. Sources: https://github.com/open-mmlab/mmpose and https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmpose. MediaPipe is a lighter alternative with 33 pose landmarks and platform paths; validate terms for exact model bundle. Source: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker. |
| Short-term object tracking | **ByteTrack-style tracking** | OC-SORT or seat-region-first association only if benchmark proves better | ByteTrack associates detections across frames and has MIT-licensed implementations. Keep tracking IDs session-local and ephemeral. Source: https://github.com/ifzhang/ByteTrack. |
| Portable inference API | **ONNX Runtime** where export is valid and verified | Model-native runtime for cases where ONNX export loses support or accuracy | Same abstract inference interface for all providers. Test accuracy parity after export and provider fallback. Source: https://onnxruntime.ai/docs/execution-providers/. |
| Optional semantic crop comparison | OpenCLIP/SigLIP family only after exact checkpoint review | Grounding DINO text prompts or a small vision-language model for offline engineering review | Use only as an exploratory auxiliary signal. Do not turn image-text similarity into a safety label without calibration and independent evidence. Each checkpoint's license/data and suitability must be verified. Sources: https://github.com/mlfoundations/open_clip and https://github.com/QwenLM/Qwen3-VL. |
| Optional segmentation/refinement | A model with publicly documented inference weights and a cleared license; evaluate only if masks are useful | Classical geometric/ROI methods | Segmentation does not by itself determine what a mask represents. A segmentation model must not be treated as a seatbelt classifier. |

**Licensing rule:** Apache-2.0 or MIT is preferred for code where possible, but a code license does not automatically clear a pretrained checkpoint, training data, dependency, tokenizer, model export, or redistribution rights. D-FINE explicitly cautions that its Objects365-trained checkpoint family may have dataset terms and should not automatically be assumed commercially cleared. For the first legal-review baseline, prefer a COCO-only checkpoint unless the Objects365 checkpoint is separately cleared. Maintain an explicit model and dependency bill of materials.

**Do not use YOLO-World as the default in this project without legal approval.** Its upstream repository states GPL-3.0 and offers commercial licensing; this may be incompatible with the desired proprietary distribution model without an appropriate legal path. Source: https://github.com/AILab-CVC/YOLO-World. Do not use Ultralytics packages/checkpoints unless their current AGPL/Enterprise implications have been explicitly accepted. Do not import any model through the `ultralytics` package: it pulls AGPL into the application even for Apache-licensed weights; use upstream/HF checkpoints only.

### 3.2 What pretrained models can and cannot promise

| Desired output | Initial no-training strategy | Initial product contract |
|---|---|---|
| Person present / bounding box | Generic pretrained detector + temporal confirmation | Available when tested in the supported camera geometry; expose score and visibility. |
| Object present | Generic detector for known classes; optional open-vocabulary candidate for uncommon classes | Only supported labels in the active model pack; everything else is unknown/unclassified. |
| Seat occupancy | Person/visible-body evidence mapped to calibrated seat ROI over time | `OCCUPIED`, `LIKELY_OCCUPIED`, `EMPTY_CANDIDATE`, or `UNKNOWN`; “empty” requires positive evidence and no visibility/health issue. |
| Adult versus child | Evaluate zero-shot visual candidates only as an explicitly provisional attribute; body size/pose alone is not age truth | `APPARENT_SMALL_OCCUPANT` / `APPARENT_LARGE_OCCUPANT` only if evidence is adequate; otherwise `UNKNOWN`. Do not state exact age, legal age, or certainty. Do not use it for restraint/airbag decisions. |
| Child restraint / child seat | Open-vocabulary text-conditioned localization experiment plus temporal/seat-region checks | `DETECTED_CANDIDATE` only after evaluation; if unreliable, mark capability unsupported instead of claiming absence. A candidate is not evidence that a child is safely restrained. |
| Seat-belt presence/fit | Open-vocabulary localization experiment and geometric consistency research using body keypoints; compare with any OEM buckle-switch signal if made available | `FASTENED`, `UNFASTENED`, `MISROUTED`, `OCCLUDED`, `UNKNOWN` must be separated. A camera-only status stays unsupported/unknown unless validation demonstrates reliable performance across behind-body routing, dark clothing, belt colors, night/NIR, posture, and occlusion. Never infer “fastened” solely because a belt-like line is visible. |
| Posture | Keypoints relative to seat/camera geometry; temporal filters | Describe measurable posture categories only after rule-level validation; raw keypoints can be absent or low confidence. Avoid medical or intent claims. |
| Driver distraction/drowsiness | Future optional feature; pose/gaze/action models require separate evaluation and legal/privacy review | Not part of minimum acceptance; no production alert until independently validated. |
| Forgotten child / unattended occupant | Occupancy state plus verified vehicle ignition/door/lock context from an authorized signal source | Do not implement from camera alone. Require verified vehicle-state interface and validated transition logic. |

### 3.3 Zero-shot class testing procedure

For each candidate model and target label:

1. Record code URL, exact revision, checkpoint URL, checksum, checkpoint license, upstream data, input dimensions, preprocessing, model size, and runtime support.
2. Run inference without modifying weights on a fixed evaluation set that includes representative positives, hard negatives, occlusions, lighting/night/NIR, camera angles, clothing, small occupants, empty seats, and child seats with/without visible children.
3. Report per-class precision, recall, false-positive/false-negative examples, confidence distribution, and failure categories. Do not report only aggregate mAP.
4. Record threshold selection separately from model selection and avoid threshold selection on the hidden final test split.
5. Compare model outputs against ground truth labels produced for evaluation only. Never use these labels to train/fine-tune.
6. If a class fails the predetermined gate, return `UNKNOWN`/`UNSUPPORTED` and explain it in the UI and API. Do not switch to a generative/VLM model merely to create a positive-looking demo.
7. Freeze the chosen model manifest/config and validation report for the phase release.

### 3.4 Datasets and evidence

- Consider **TICaM — Time-of-flight In-car Cabin Monitoring Dataset** as one evaluation-source candidate because it contains in-cabin modalities and detailed annotations. Check the live dataset download terms, consent/redistribution conditions, covered modalities, and class definitions before use. Paper: https://arxiv.org/abs/2103.11719.
- The published in-vehicle study “Integrated In-vehicle Monitoring System Using 3D Human Pose Estimation and Seat Belt Segmentation” describes combining pose, belt segmentation, and belt-status classification, but it evaluated on a private dataset. It is evidence that belt status is a specialized task, not proof that public generic weights solve it without training. Paper: https://arxiv.org/abs/2204.07946.
- Use COCO/Objects365 benchmark claims only as context for generic detectors. Do not equate street-scene mAP with in-cabin accuracy.
- If open datasets do not represent the target cabin, clearly report that coverage gap. Ask the OEM for permitted representative evaluation clips when available. The build can proceed on architecture and observable states, but any unvalidated feature remains explicitly provisional.

---

## 4. Algorithm design

### 4.1 Frame processing

- Use OpenCV and/or GStreamer based on the existing repo and source requirements. Prefer GStreamer for supported hardware-accelerated capture/decode paths; use OpenCV for common transforms and simple sources. Confirm source availability/license and actual camera interface during Phase 0.
- Support camera, video file, and replay source behind one interface.
- Each frame has a monotonic timestamp, UTC timestamp where available, source ID, sequence number, image dimensions, and capture health metadata.
- Maintain bounded frame queues. When inference is slower than capture, drop stale frames according to a documented policy instead of allowing unbounded latency.
- Separate capture rate from inference rate. Do not claim every frame was analysed if frames were sampled or dropped.
- Use ROI cropping only when seat geometry and object size permit it; record transform metadata so detections/keypoints can be mapped back to original frame coordinates.
- Avoid lossy transformations that materially harm small-seatbelt/child-seat evidence. Benchmark resolution and crop tradeoffs.

### 4.2 Detection postprocessing

- Use the official model preprocessing and decoding for the specific checkpoint. No generic assumption about tensor layouts or coordinate scaling.
- Normalize detections into a model-independent object:
  - `class_id`, `class_label`, `score`, normalized `bbox_xyxy`, source frame, model/version, provider, inference duration, and `evidence_kind`.
- Keep raw detector confidence separate from state confidence. A score from one model is not necessarily comparable with a score from another model.
- Use class-aware/non-class-aware NMS only as required by the model. Do not add duplicate NMS if the detector already performs it unless tested.
- Keep unknown/open-vocabulary outputs in a namespace separate from the stable product ontology. Never let arbitrary text prompts inject new durable enum values without schema/config validation.

### 4.3 Tracking and association

- Start with ByteTrack or a small audited implementation, run it on detection outputs, and measure ID switches/fragmentation in the cabin view.
- IDs are short-lived track IDs scoped to a camera session, not identities of real people. Randomize/reinitialize on session reset; no cross-trip identity linking.
- Vehicle seats are stable spatial regions, so a seat-first association method may outperform generic MOT for this use case. Benchmark both:
  1. detection + ByteTrack then associate tracks to seat polygons; and
  2. seat-ROI evidence aggregation with short-lived occupant tracks.
- Association features may include bbox center/overlap, keypoints relative to seat polygon, motion continuity, and temporal evidence. Keep the score components explainable.
- If two tracks overlap or a body crosses regions, preserve ambiguity as `UNKNOWN` rather than switching seats rapidly.

### 4.4 Pose and posture

- Run pose on detected person crops or selected seat ROIs, depending on model requirements.
- Store keypoint coordinates in normalized frame space plus visibility/confidence; never interpret missing keypoints as absent limbs.
- Basic posture rules must be deterministic, documented, and configurable; use temporal windows to suppress single-frame noise.
- Candidate descriptive outputs: `UPRIGHT`, `LEANING_FORWARD`, `LEANING_SIDEWAYS`, `RECLINED`, `BODY_PART_OCCLUDED`, `UNKNOWN`. Only enable categories whose definitions can be annotated and evaluated in the target geometry.
- Do not interpret posture as injury, emotional state, or intent. No claim of 3D posture accuracy from a single RGB camera unless depth/calibration evidence supports it.

### 4.5 Temporal evidence and state machines

- Every meaningful state is computed over a configurable window rather than from one frame.
- Use hysteresis: separate thresholds for entering and leaving a state, minimum dwell time, minimum positive frames, and cool-down/debounce for events.
- Use time-based windows, not just a fixed frame count, so sampled video and lower frame rates remain coherent.
- Every state change records prior state, new state, reason codes, evidence references, model versions, and processing timestamps.
- Stale data expires. If the camera or model goes unhealthy, occupant state transitions toward `UNKNOWN`/`STALE`; it must not remain a falsely certain `EMPTY` or `OCCUPIED` indefinitely.
- Make time synchronization and out-of-order frames explicit. Never let a delayed frame roll current state backward.

### 4.6 Cabin/seat configuration

- Store vehicle/camera geometry as a versioned configuration, not as hardcoded pixel constants in model code.
- Prefer normalized polygon coordinates relative to image size with calibration metadata.
- Support seat slots such as driver/front passenger/rear-left/rear-center/rear-right without assuming every vehicle has all seats or a left/right-drive layout.
- Provide an engineering calibration workflow and a clear “unconfigured geometry” state.
- Configuration edits must be versioned in PLOMID SQL and tied to the events/state that used them.

---

## 5. Data model and SQL design

This is a **logical schema proposal**, not executable DDL. The Codex agent must inspect PLOMID's real SQL dialect, supported types, constraint/index/upsert/transaction semantics, and vector syntax before implementing SQL. Do not copy this into production as literal DDL without verifying each construct.

### 5.1 Design principles

- Store operational truth and structured state as relational data accessible through SQL.
- Prefer typed columns for time, IDs, enums/status, scores, model references, and version fields. Use JSON only for flexible, versioned details that should not drive core joins/filters.
- Use bounded, explicit retention policies and indexes only where PLOMID supports them and measured query patterns justify them.
- Make all rows traceable to a camera/session/config/model version.
- Use append-only event/history records where possible; maintain a current-state row for fast querying.
- Do not store every raw frame or full video in the database. Store optional artifact references/hashes for permitted diagnostic clips only; ensure file retention/deletion cannot leave orphaned private media.
- All vector reads/writes/searches must use SQL through PLOMID. Exact vector type/operator/index syntax must be confirmed from existing PLOMID docs/source and integration tests.

### 5.2 Logical entities

| Table / entity | Purpose | Example core fields | Notes |
|---|---|---|---|
| `odc_vehicle_config` | Versioned vehicle and seat layout | `config_id`, `vehicle_profile`, `camera_profile`, `layout_json`, `version`, `created_at`, `created_by` | JSON is versioned configuration, not an excuse to put all tables into JSON. |
| `odc_camera` | Camera/source configuration and health identity | `camera_id`, `source_kind`, `profile_id`, `status`, `last_frame_at`, `last_error_code` | Never store credentials in plaintext; use secret-management conventions from the repo. |
| `odc_session` | One boot/run/replay session | `session_id`, `camera_id`, `source_kind`, `started_at`, `ended_at`, `build_version`, `config_id`, `status` | Track IDs are session-scoped. |
| `odc_model_artifact` | Model metadata and legal/provenance record | `model_id`, `family`, `checkpoint_sha256`, `source_url`, `code_revision`, `weights_license`, `dataset_terms`, `approved_for_commercial_use`, `input_profile`, `runtime_profile` | Store metadata, not necessarily weight bytes. `approved...` must be a real reviewed state, not auto-set true. |
| `odc_inference_metric` | Runtime performance samples/aggregates | `session_id`, `model_id`, `provider`, `captured_at`, `inference_ms`, `queue_ms`, `dropped_frames`, `memory_bytes` | Define sampling/retention to avoid unlimited writes. |
| `odc_observation` | Model-independent detection/keypoint facts | `observation_id`, `session_id`, `camera_id`, `frame_seq`, `captured_at`, `kind`, `label`, `score`, `bbox`, `keypoints_ref_or_json`, `model_id`, `visibility_state` | Only persist observations needed for product features and evidence; high-frequency data may be downsampled/retained briefly. |
| `odc_seat_state_current` | Latest state per seat | `session_id`, `seat_id`, `state`, `state_confidence`, `visibility`, `reason_codes`, `updated_at`, `last_positive_at`, `model_set_version`, `config_id` | Composite unique key should be session + seat. Exact constraint syntax depends on PLOMID. |
| `odc_occupant_state_current` | Latest state for a short-lived occupant track when needed | `session_id`, `track_id`, `seat_id`, `presence_state`, `apparent_size_class`, `posture_state`, `seatbelt_state`, `child_restraint_state`, `confidence_by_attribute`, `updated_at` | Each attribute has its own confidence and visibility; no single score should hide uncertainty. |
| `odc_event` | Append-only semantic state transitions | `event_id`, `session_id`, `event_type`, `severity`, `seat_id`, `track_id`, `occurred_at`, `recorded_at`, `prior_state`, `new_state`, `reason_codes`, `evidence_summary`, `config_id`, `model_set_version` | Do not generate a notification/alert for a single raw low-confidence detection. |
| `odc_camera_health` | Camera/stream diagnostics | `camera_id`, `session_id`, `checked_at`, `status`, `fps_observed`, `blur_score`, `exposure_status`, `occlusion_status`, `failure_code` | Quality metrics need validation; generic sharpness heuristics are advisory only. |
| `odc_validation_run` | Model/application benchmark run | `run_id`, `build_version`, `dataset_id`, `hardware_profile`, `model_manifest`, `started_at`, `completed_at`, `status` | Validation metadata must be reproducible. |
| `odc_validation_result` | Per-class and per-scenario metrics | `run_id`, `task`, `label`, `scenario`, `precision`, `recall`, `f1`, `false_positive_count`, `false_negative_count`, `latency_p50`, `latency_p95`, `notes` | Only persist evaluation results when useful; detailed raw report may also be a versioned local artifact. |
| `odc_embedding` (optional) | Semantic retrieval for engineering/validation diagnostics | `embedding_id`, `session_id`, `event_id`, `embedding_kind`, `model_id`, `vector`, `created_at`, `retention_class`, `source_hash` | No face/person identity embeddings. Enable only after actual PLOMID vector SQL support, compatible dimensionality, query cost, and retention are verified. |
| `odc_audit_event` (where required) | Changes to sensitive configuration and manual review actions | `audit_id`, `actor`, `action`, `target_type`, `target_id`, `occurred_at`, `before_hash`, `after_hash`, `reason` | Auditability depends on access control and tamper-resistant operations, not only having a table. |

### 5.3 State and enum contracts

Use a centrally defined, versioned ontology. Suggested logical enums:

- `PresenceState`: `OCCUPIED`, `LIKELY_OCCUPIED`, `EMPTY_CANDIDATE`, `UNKNOWN`, `STALE`.
- `VisibilityState`: `VISIBLE`, `PARTIALLY_OCCLUDED`, `OCCLUDED`, `OUT_OF_VIEW`, `CAMERA_UNHEALTHY`, `NOT_EVALUATED`.
- `AttributeState<T>`: value + confidence + evidence timestamp + reason + model/config version. Do not force a generic enum if the language's type system can encode a safer discriminated union.
- `SizeAppearance`: `APPARENT_SMALL_OCCUPANT`, `APPARENT_LARGE_OCCUPANT`, `UNKNOWN`, `UNSUPPORTED`. Do not name the field exact `age`.
- `SeatbeltState`: `FASTENED_CANDIDATE`, `UNFASTENED_CANDIDATE`, `MISROUTED_CANDIDATE`, `OCCLUDED`, `UNKNOWN`, `UNSUPPORTED`. Promote to definitive words only when evidence is sufficiently validated and permitted by product requirements.
- `RestraintState`: `CHILD_SEAT_CANDIDATE`, `BOOSTER_CANDIDATE`, `OTHER_OBJECT_CANDIDATE`, `NOT_DETECTED`, `OCCLUDED`, `UNKNOWN`, `UNSUPPORTED`. `NOT_DETECTED` is not the same as confirmed absent.
- `PostureState`: enable only evaluated labels; always include `UNKNOWN` and occlusion/quality outcomes.
- `EventType`: versioned constants such as `SEAT_PRESENCE_CHANGED`, `OCCUPANT_TRACK_STARTED`, `OCCUPANT_TRACK_ENDED`, `ATTRIBUTE_BECAME_UNKNOWN`, `CAMERA_DEGRADED`, `MODEL_UNAVAILABLE`, `DATABASE_DEGRADED`, `CONFIG_CHANGED`, `SESSION_STARTED`, `SESSION_ENDED`.

Never place exact model-specific class IDs in API/UI as permanent product semantics. Map model classes through a reviewed label adapter into this ontology.

### 5.4 Vector features — use selectively via SQL only

PLOMID is described by the product owner as supporting vectors, but Codex must validate the actual type/operators/indexes/limits from PLOMID's real SQL interface before relying on them.

Reasonable candidate uses:

- similarity search over **non-identifying** engineering event summaries or approved, short-lived diagnostic crop embeddings;
- grouping visually similar failure cases during offline evaluation;
- retrieval of related validation cases/model regressions for an engineering console;
- comparing non-person scene embeddings to find similar camera-quality failures.

Not permitted by default:

- face recognition or persistent identity embeddings;
- using nearest-neighbor distance to decide a safety-critical occupant state;
- replacing SQL's exact relational state/event queries with vector search;
- embedding every frame and storing it indefinitely;
- enabling vectors simply to advertise vector support without an actual, evaluated use case.

All vector usage goes through SQL statements. If PLOMID SQL vector support is not mature or not documented, ship relational ODC features first and leave the vector capability behind a feature flag until a verified PLOMID SQL integration test passes.

### 5.5 SQL access patterns

Implement a `PLOMID ODC repository` boundary with methods analogous to:

- `health_check()`
- `ensure_schema_version()`
- `start_session(...)` / `end_session(...)`
- `upsert_camera_status(...)`
- `write_state_snapshot(...)`
- `append_event(...)`
- `read_current_seat_states(...)`
- `read_event_timeline(...)`
- `write_metric_batch(...)`
- `read_model_manifest(...)`
- `write_validation_result(...)`
- optional `write_embedding(...)` / `search_similar_cases(...)`

These are logical operations, not mandated function names. Use the actual language/framework and driver conventions. Do not expose SQL connections to perception code. All repository operations must have bounded timeout/retry rules, structured errors, and cancellation behavior.

### 5.6 SQL, migration and retention gates

Before writing production DDL, test and document:

- SQL dialect and placeholder convention;
- type support for timestamps, booleans, integer/float, text, JSON, arrays, enum representation, UUID/ID strategy, and vector type;
- upsert/merge support or safe portable alternative;
- transaction and isolation behavior;
- indexes and unique constraints actually supported;
- prepared statement support and connection limits;
- result decoding and server error semantics;
- migration/versioning approach;
- vector serialization, dimension enforcement, similarity operators/indexes, and query-planner behavior if vectors are used.

No PLOMID capability may be inferred merely from PostgreSQL protocol compatibility. Database startup should fail clearly when a required schema/capability is missing; it must not quietly switch to another database.

---

## 6. Product feature inventory

### 6.1 Core perception

- Live camera/file replay input.
- People and supported generic object detection.
- Session-local tracking and stable seat-region association.
- Pose/keypoints when visible and valid.
- Configurable seat layout and camera profile.
- Confidence, visibility, age of evidence, and model provenance on every attribute.
- Camera quality and stream health.

### 6.2 Temporal cabin intelligence

- Occupied/empty-candidate/unknown state per seat.
- Entry/exit and state-transition timeline.
- Debouncing and hysteresis.
- Detection of state becoming stale or unknown.
- Candidate apparent size-class, child-seat and seatbelt signals only when validated; otherwise visible unsupported states.
- Occupant remains/camera view blocked/system degraded event.
- Optional forgotten-occupant workflow only after authorized ignition/door/lock signals are integrated.

### 6.3 Engineering console

- Live/replay stream, with capture timestamp and frames-processed/dropped badges.
- Overlay of boxes/keypoints/seat regions and per-signal confidence.
- Seat map with distinct states for occupied, empty candidate, unknown, stale, and unsupported.
- State/event timeline with filter by seat/session/event.
- Model/runtime diagnostics: model/checkpoint version, active execution provider, inference latency, queue depth, dropped frames, memory, database health.
- SQL-backed current state and event-history view.
- Model benchmark and validation report screens.
- Manual review labels for evaluation only, with provenance; no implicit training pipeline.
- Clearly separate live sensor output, replay, mocked sample data, and engineering annotations.

### 6.4 Operations and reliability

- Health checks and readiness/liveness status.
- Model loading/initialization errors.
- Camera disconnect/reconnect and recovery.
- PLOMID SQL connection backoff and visible degraded state.
- Bounded queues and graceful backpressure.
- Config/schema/version compatibility.
- Structured logs with secrets and personal image data excluded.
- Exported reproducible diagnostics without unapproved image/video content.

### 6.5 Later optional features, not MVP gates

- Multi-camera/vehicle fleet fleet-health summary.
- DMS-lite stretch (zero new models): head-pose candidacy from existing pose keypoints + phone-in-hand candidacy from the generic detector's phone class, temporal-gated like all attributes and labelled preview only — answers the driver-monitoring future without a DMS claim; no gaze/drowsiness certification implied (GSR ADDW tailwind, not compliance).
- Sensor fusion with OEM-authorized buckle switch, pressure mat, ignition, door, lock, seat adjustment, and vehicle motion signals.
- Parked-guardian mode: door/lock-triggered low-power watch (sub-sampled inference, motion-presence candidacy) as the honest camera-side half of the CPD story; any unattended-occupant alert requires fused vehicle signals and stays `FUSION_REQUIRED` without them — never a breathing-detection claim.
- Model comparison and non-training regression-cluster search with PLOMID vector SQL.
- Plug-in model adapters and downloadable approved model pack.
- Simulation/replay lab across vehicle profiles.
- Fleet analytics that sends only agreed aggregate/event data and respects OEM privacy policy.

---

## 7. Proposed phase roadmap

Phases are independently reviewable. Later phases may evolve based on repository audit or pilot requirements, but must not silently override the hard constraints in Sections 2 and 3.

## Phase 0 — Repository audit, feasibility, and evidence baseline

**Goal:** understand what exists and resolve unknowns before creating architecture.

**Tasks**

- Inspect repo layout, existing ODC/app code, PLOMID SQL protocol docs, SQL type/vector docs, drivers, migrations, logging/config, test harness, CI, web console patterns, and current build plan.
- Identify existing reusable components, package boundaries, naming conventions, dependency/security policy, and all active verification commands.
- Confirm how to connect to PLOMID through documented SQL (wire protocol, driver, dialect, authentication, TLS, timeout and server version behavior).
- Build a capability matrix based on source/documentation and executable smoke tests—not assumptions.
- Footage first (ordered): (1) Pexels core set DONE 2026-10-13 — 8 landscape 1080p+ clips in `footage/` (family ride, baby/child seat ×2, empty cabin, night drive, belt ×2, rear wide) + 9 backup links, all Pexels License, inventoried in `footage/README.md`; (2) download SVIRO (synthetic rear-seat, free research) immediately as the day-one moving picture; (2) file TICaM + Drive&Act access requests (research-only, eval-only, no redistribution); (3) record one actor-staged cabin clip with consent on the developer laptop camera as the golden-path demo footage; (4) evaluate synthetic vendors (Anyverse/SKY ENGINE) for belt-misuse/blanket/OOP eval sets. No clip is ever used for weight updates; every clip's license/consent is recorded in ODC_MODEL_LICENSES.md.
- Lock the developer laptop as the reference demo target: record CPU/GPU/RAM/OS, camera, target resolution/processing rate, memory envelope, and provisional P50/P95 targets as development targets only (never ECU promises); ECU profiles (QCOM/TDA4/Orin) stay matrix entries until hardware is in hand.
- Audit exact model/checkpoint license and distribution terms for candidate baseline models.
- Establish known-good test clips or legally usable evaluation datasets; define an evaluation split and a no-training rule.
- Record a baseline command report for existing checks without editing unrelated code.

**Deliverables**

- Updated/created `docs/odc/ODC_REQUIREMENTS.md`.
- Updated/created `docs/odc/ODC_ARCHITECTURE.md`.
- Updated/created `docs/odc/ODC_MODEL_SELECTION.md`.
- Updated/created `docs/odc/ODC_DECISIONS.md`.
- SQL/protocol capability matrix and list of verified integration examples.
- Risk register with unknowns, owners, and closure gates.

**Acceptance criteria**

- Existing ODC/PLOMID components are inventoried and reused or explicitly ruled out with reasons.
- SQL capabilities, driver, and limitations are documented from evidence.
- Candidate weights have a per-artifact license/provenance status: `APPROVED`, `REVIEW_REQUIRED`, or `REJECTED`.
- Hardware target and first evaluation path are recorded, or clearly marked blockers.
- No application database or model training pipeline is added.

**Out of scope:** full app implementation, weight training, claims of cabin detection accuracy.

## Phase 1 — Requirements, contracts, and application foundation

**Goal:** build a clean product skeleton and stable domain contracts without overbuilding.

**Tasks**

- Define versioned domain types for frame metadata, detections, keypoints, seat/occupant state, visibility, event, health, model metadata, and SQL error states.
- Define a model-independent perception contract and runtime interface.
- Define camera source interface supporting live source and prerecorded-file replay; fake source allowed only in tests/demo and always labelled.
- Define temporal state-engine interface, persistence repository interface, config structure, and application health model.
- Set up a Python-only project structure (`odc` package, `pyproject.toml`, locked deps). ODC application code is Python only (FastAPI, psycopg3, OpenCV, ONNX Runtime, NumPy, Pydantic v2). No C++/Rust/Node service in this repo; model runtimes are Python packages only. Do not port the application into a systems language or introduce a second frontend stack.
- Add configuration validation, logging, metrics, graceful shutdown, dependency lockfiles, and deterministic unit test fixtures.
- Add model manifest contract and licensing/provenance fields.
- Ensure a mode such as `replay`, `live`, `test`, and `demo` cannot be confused in the UI or stored session metadata.

**Acceptance criteria**

- App starts, validates configuration, reports health, exits cleanly.
- Domain contracts have unit tests and serialization round-trip tests where relevant.
- No hard dependency on a model vendor leaks into domain/state types.
- No second database dependency is introduced.
- All UI/mock output is visibly labelled as such.

## Phase 2 — PLOMID SQL repository and schema migration foundation

**Goal:** make PLOMID the real source of durable truth early, exclusively through SQL.

**Tasks**

- Implement PLOMID connection/config/auth using verified SQL protocol and an approved driver.
- Implement readiness query and documented capability smoke tests.
- Implement migration/version tracking using the repo's existing migration convention if present; otherwise a simple versioned SQL migration mechanism compatible with confirmed PLOMID semantics.
- Create the first verified subset of logical tables: session, model metadata, camera status, current seat/occupant state, append-only events, and operational metrics as needed.
- Implement parameterized repository methods, transaction boundaries where confirmed, connection timeout/cancellation, safe retry/idempotency, and observability.
- Add integration tests against a real PLOMID test instance. Unit tests can use fakes, but no database emulation that masks dialect differences.
- Include schema compatibility failure paths and clear startup errors.
- Document how tests provision PLOMID; never silently use SQLite/in-memory persistence in production.

**Acceptance criteria**

- All persisted state/event rows in integration tests are written/read using SQL over the supported protocol.
- A database outage is surfaced as degraded health; no alternate DB is created or used.
- Repeated event submission has a defined idempotency strategy where needed.
- Migration tests prove upgrade from the starting supported schema and fail clearly on incompatible schema.
- SQL syntax and features used are recorded as verified PLOMID capabilities.

## Phase 3 — Camera ingestion, frame handling, and camera health

**Goal:** reliable live and replay frame plumbing with visible operational state.

**Tasks**

- Implement capture adapters selected in Phase 0 (e.g. OpenCV/GStreamer) behind the source interface.
- Implement timestamping, frame sequence, bounded queues, sampling/drop counters, disconnect detection, reconnect/backoff, decode error handling, and graceful shutdown.
- Add frame quality signals such as frozen image, extreme darkness/overexposure, occlusion, low contrast, and blur only as advisory indicators; calibrate and document their limitations.
- Track actual captured and processed FPS separately.
- Support offline video replay at real-time or fast playback for deterministic evaluation.
- Persist camera/session health via PLOMID SQL with bounded update frequency.

**Acceptance criteria**

- File replay is deterministic for the same file/config.
- Live-source interruption does not deadlock or grow an unbounded queue.
- UI/logs expose stale/frozen/failed/healthy states distinctly.
- Database writes are rate-limited and resilient; no frame-by-frame metric-write storm.
- Tests cover disconnect, corrupt frame, slow consumer, stale frame and shutdown.

## Phase 4 — Pretrained model evaluation harness and generic detector

**Goal:** deploy an inference-only generic detector and choose the best model based on the target profile.

**Tasks**

- Download/use only legally approved pretrained checkpoints; verify checksum and record the manifest.
- Implement D-FINE-S COCO as the first candidate if its exact code and checkpoint are approved.
- Add D-FINE-M and D-FINE-L comparison paths only if useful for the available hardware; avoid downloading every model without a defined benchmark.
- Add model-native inference first; export to ONNX and use ONNX Runtime only if the export preserves model behavior and simplifies deployment. Validate parity on the same images.
- Test CPU and available GPU/runtime provider; report fallback behavior.
- Define consistent result output with boxes, labels, scores, original-frame coordinate mapping, model ID, provider, inference duration, and dropped/sample-frame metadata.
- Build evaluation metrics: per-class precision/recall/F1 and/or AP where appropriate; false-positive/false-negative sample list; P50/P95 latency; sustained throughput; memory; cold-start time.
- Separate generic dataset benchmarks from cabin-specific evaluation. Benchmark class support explicitly.

**Acceptance criteria**

- Real input frames produce real detections with model provenance; there are no hardcoded/fabricated detection boxes.
- Exact model checkpoint/hash/license is shown in report and saved in manifest.
- The winning detector is justified by measurements on available target hardware and a fixed evaluation set; otherwise retain candidate status.
- ONNX export/provider path has a parity report or remains disabled.
- Child-seat/seatbelt/age classes are not claimed based on generic person/object benchmark numbers.

## Phase 5 — Tracking, pose, and seat-region geometry

**Goal:** provide temporally stable, explainable occupant evidence.

**Tasks**

- Integrate ByteTrack or another approved tracker behind a stable interface; document tracker version/license.
- Add RTMPose-S/M candidate and MediaPipe fallback/alternative if architecture and license permit; benchmark both on target clips and hardware.
- Add pose visibility/quality semantics and robust mapping from crop coordinates to source-frame coordinates.
- Build seat geometry config/editor for normalized polygons and camera profile versioning.
- Benchmark track continuity and seat association variants; avoid long-term or across-session identity linking.
- Keep keypoints as observation-level data where needed and derive posture state only in the next reasoning phase.

**Acceptance criteria**

- Tracks are stable enough for the measured target clips and reset correctly on a session reset.
- Pose absence/occlusion does not become a false “limb absent” or “seat empty” claim.
- Seat association uncertainty is surfaced; ambiguous assignments are not forced.
- The app works with and without a pose model; optional capability failure does not crash the whole system.

## Phase 6 — Temporal reasoning and occupant state machine

**Goal:** convert frame observations into meaningful, time-stable seat states/events.

**Tasks**

- Implement per-seat temporal aggregation, hysteresis, dwell times, minimum positive evidence, cooldown/debounce, stale timeout, and state expiry.
- Keep `UNKNOWN`, `OCCLUDED`, `STALE`, `CAMERA_UNHEALTHY`, `MODEL_UNAVAILABLE`, and `UNSUPPORTED` distinct.
- Implement state transition event generation with reason codes and provenance.
- Add deterministic rule configuration with defaults, versioning and tests; do not encode thresholds in arbitrary scattered constants.
- Map evidence to occupancy candidate state by seat region; do not say `EMPTY` simply because a detector missed the person.
- Make out-of-order timestamps and skipped frames safe.
- Persist current state and append-only transitions through the Phase 2 PLOMID SQL repository.

**Acceptance criteria**

- Temporal tests cover flicker, one-frame miss, prolonged occlusion, stale camera, track swap, seat ambiguity, source restart and out-of-order frames.
- All events have session, seat/track when known, occurrence time, recorded time, reason codes, config version, model-set version, and prior/new state.
- Current state can be reconstructed/reconciled from a session's event history under the documented model.
- No hardcoded example confidence (e.g. `0.98`) is presented as a measured result unless a genuine inference produced it and provenance is displayed.

## Phase 7 — Child-seat, belt, and apparent-size zero-shot evaluation

**Goal:** determine which cabin-specific attributes can honestly be exposed without training.

**Tasks**

- Implement Grounding DINO only as an optional low-rate/ROI-based experiment when hardware permits. Try a reviewed prompt set and record prompt version. Use exact approved checkpoint and license.
- Optionally benchmark CLIP/SigLIP image-text similarity or a small VLM for engineering comparison only; this cannot decide final safety state. Keep it disabled by default until review of latency, license, model data, output stability, and test evidence.
- Explore seatbelt visual evidence only as a candidate: belt/torso region localization, pose geometry and optional mask evidence. Explicitly distinguish “belt-like feature detected” from “belt fastened correctly.”
- Evaluate child-seat variants and booster/carry-cot lookalikes against clutter and empty-seat negatives.
- Evaluate coarse apparent body-size category, but never equate it with true age or physical size.
- Introduce per-attribute state contracts and gating based on class-specific validation reports.
- For each attribute, decide `SUPPORTED_FOR_DEMO`, `PROVISIONAL`, or `UNSUPPORTED` with evidence and known limitations.

**Acceptance criteria**

- Every attribute has class-specific confusion analysis and false-negative/false-positive examples.
- A failing model results in `UNKNOWN`/`UNSUPPORTED`, not an invented positive or negative.
- No seatbelt/child-seat status drives vehicle action or safety-critical decision-making.
- No training/fine-tuning occurs and no custom weights are generated.
- Every optional model has exact license, checkpoint and data-provenance record.

## Phase 8 — Cabin engineering console, timeline, and data queries

**Goal:** make the system compelling and inspectable for engineering/OEM review.

**Tasks**

- Build the console as a Python-only FastAPI app: Jinja2 templates + HTMX partials + Alpine.js for SVG interactivity + ECharts (vendored) for timeline/charts + Tailwind (local build), all static vendored for offline pilots — no CDN, no build step. No React/Node service in this repo; all application logic is Python, JS libs are vendored presentation only. App shell: top status bar (LIVE/REPLAY/SIMULATED/DISCONNECTED/DEGRADED + DB/camera/model/session), sidebar nav, main view, right inspector (confidence/provenance/evidence), command palette (Ctrl+K), toasts. Every screen states loading/empty/error/degraded distinctly; every displayed state traces to a PLOMID SQL row or is labelled transient.
- Build live/replay camera view with SVG overlay layer (not baked pixels): toggleable boxes/keypoints/seat polygons, clickable tracks → inspector, layer legend, confidence/provenance panel, captured-vs-processed FPS + dropped badge.
- Build interactive SVG seat map with all supported/uncertain states clearly differentiated: click/hover/keyboard select per seat, multi-seat filter, tooltips with state + confidence + evidence age; calibration edit mode with draggable polygon vertices writing versioned `odc_vehicle_config` through review.
- Build event timeline (ECharts, SQL-backed): zoom/brush/pan over time, click event → evidence inspector, filters for session/seat/type/time, SSE-append with polling fallback.
- Build model/runtime diagnostics view with actual measured FPS, P50/P95 latency charts, queue depth, dropped frames, memory/provider and model/checkpoint.
- Add configuration/profile view with revision history and version diff review before apply.
- Add a guided 5-minute evaluator tour (scripted console walkthrough: live → seat map → time-travel → chaos recovery → dossier) and a one-page benchmark card export (hardware, FPS, P50/P95, per-class P/R from real runs only; never pledged or copied figures).
- Add status labels: `LIVE`, `REPLAY`, `SIMULATED`, `DISCONNECTED`, `DEGRADED`. Simulated sample data must never appear to be a live model output.
- Add design tokens (light/dark ops themes, density), accessible contrast + focus states, responsive sizing, clear empty/error/loading states and controlled SSE/polling with reconnect.

**Acceptance criteria**

- Every displayed state can be traced to a PLOMID SQL record or explicitly identified as transient live output.
- Seat map SVG is fully interactive (click/hover/keyboard, tooltips, calibration edit mode) and seat clicks filter timeline + inspector.
- Timeline supports zoom/brush/pan and click-to-evidence; video overlays are clickable to track inspector.
- UI is exercised with real inference, replay, no camera, model failure, and PLOMID outage.
- Playwright-Python E2E covers seat-click → inspector, timeline-click → evidence, and SSE reconnect.
- No UI-only fake detections or fabricated benchmark numbers.
- Event and current-state views agree with repository test queries.

## Phase 9 — Validation, replay lab, and regression suite

**Goal:** make the product measurable and stop regressions from masquerading as progress.

**Tasks**

- Create deterministic replay runner with fixed clips, frame sampling policy, config snapshot and model manifest.
- Define annotation format for evaluation-only ground truth; clearly separate annotation data from production code and prohibit use for weight updates.
- Split evaluation by scenario/vehicle/subject/time rather than randomly splitting nearly identical frames; reduce leakage.
- Report per-class metrics, seat-level state metrics, time-to-detect, event false alarms/hour, false negatives, detection latency, and uncertainty rates.
- Add difficult buckets: partial occlusion, blankets, child seats, empty child seats, rear-facing/forward-facing seat, multiple occupants, dark clothes, glare, low light/night/NIR, sunglasses, bags, shadows, camera vibration, shifted mount, different cabin geometries, unoccupied cabin and camera blocked.
- Add model/version/config regression comparisons and failure galleries under approved privacy rules.
- Use TICaM or other permitted open data as extra evaluation only, not as proof of target-vehicle generalization.

**Acceptance criteria**

- Benchmark can be run from a documented command and emits machine-readable and human-readable reports.
- Same model/config/data version reproduces metrics within expected nondeterministic tolerance.
- The final evaluation holdout is not used to tune thresholds.
- Every feature marked supported has a recorded acceptance threshold and result; unsupported items are visibly identified.

## Phase 10 — Performance optimization and heterogeneous hardware adapters

**Goal:** meet target latency/throughput on named device profiles without changing semantics.

**Tasks**

- Profile end-to-end stages before optimizing: capture/decode, preprocessing, detector, pose, tracker, state engine, SQL writes, UI.
- Benchmark input sizes, sampling rates, batch size (normally low-latency batch 1 for cabin streams), FP32/FP16/INT8 where supported, and thread settings.
- Consider ONNX Runtime execution providers after model export parity tests; NVIDIA TensorRT/DeepStream can be an optional NVIDIA adapter only if it provides clear value and conforms to existing architecture.
- Consider OpenVINO for validated Intel targets and provider-specific options for Apple/AMD/ARM only as confirmed by available systems and official support.
- Implement provider discovery, compatibility reporting, explicit fallback, and reproducible deployment profiles.
- Quantize only with a supported pretrained/model conversion pipeline and calibration procedure that does not train/finetune weights; if quantization violates the no-training/no-new-weight-artifact policy as interpreted by the owner, leave it disabled pending explicit approval.
- Ensure SQL batch/flush cadence is efficient without compromising event durability or freshness.

**Acceptance criteria**

- Target hardware profile is named and testable; metrics specify resolution, sample rate, runtime, provider, precision, ambient/power conditions if available.
- P50/P95 latency and sustained throughput meet the agreed requirement or the limitation is documented.
- Model output parity and state/event semantic parity are verified after optimization.
- Unsupported runtime providers fail clearly and do not silently switch to an untested path.

## Phase 11 — Security, privacy, and resilient operations

**Goal:** build trustworthy system behavior and keep sensitive information controlled.

**Tasks**

- Define threat model and data-flow map for camera, process, SQL, console, logs, diagnostics, updates and model artifacts.
- Apply least-privilege SQL account, secret handling, TLS where supported, safe configuration parsing, access control for console and exports, and redaction.
- Validate SQL parameterization and escaping; prohibit user input in dynamically generated SQL identifiers except through validated allowlists.
- Limit event payloads and log fields; never log full frames/base64/crops or credentials accidentally.
- Add retention configuration, deletion workflow and artifact expiration for any permitted diagnostic media.
- Add process-supervision, restart behavior, bounded queues, recovery test, database reconnect, disk-full handling, and safe degraded mode.
- Run dependency/license/vulnerability scans using repo-standard tools.
- Document that prototype is not ISO 26262 / ASPICE / ISO/SAE 21434 / regulatory certified unless formal independent evidence exists; map requirements to a later OEM safety/cybersecurity programme.

**Acceptance criteria**

- Security review has evidence for auth/permissions, SQL injection resistance, secret redaction, media retention and outage behavior.
- No unapproved raw media/identity data stored by default.
- Fault injection covers camera, inference, SQL connection, corrupted config, disk pressure, shutdown and restart.
- No certification/compliance claim without proof.

## Phase 12 — Packaging, installation, deployment and field diagnostics

**Goal:** reproducible local installation and pilot deployment.

**Tasks**

- Provide reproducible dev setup, runtime packaging, config examples without secrets, model-download verification, database bootstrap instructions, and sample video replay.
- Pin model checkpoints and dependencies with checksums; provide offline installation path where feasible. Sign model packs with TUF/Uptane-style metadata (hash + version, offline keys, rollback protection) and verify without live server contact for air-gapped OEM labs. Publish a named SoC compatibility matrix (Qualcomm SA8155P/SA8295P via QNN, TI TDA4VM/TDA4VH via TIDL, NVIDIA Orin via TensorRT+DLA, Ambarella CV, Renesas R-Car, ARM CPU fallback) with per-target TOPS/latency/power budgets. Ship an ODC Concept Development Kit (container + legal sample clips + vehicle-signal simulator + dossier template) so evaluators run in minutes.
- Add a self-test/diagnostic bundle that excludes sensitive frames by default.
- Define compatibility matrix for OS/CPU/GPU/runtime/provider/PLOMID version/model manifest.
- Add startup readiness checks for SQL, camera, model loading and config validation.
- Add pilot rollback and upgrade procedure; preserve schema/migration safety and model manifest history.
- Include sample dashboard/screenshots only if they are real outputs from the app or clearly marked as mock.

**Acceptance criteria**

- A new environment can be installed from docs with the documented PLOMID instance and approved model assets.
- Offline replay works without cloud service.
- Missing model weights, incorrect license approval, SQL failure or unsupported provider produce clear startup/readiness failures.
- Upgrade/rollback has been exercised on a test environment.

## Phase 13 — OEM integration, optional sensor fusion, fleet features and production gap plan

**Goal:** prepare a realistic pilot package and isolate later product expansions.

**Tasks**

- Define authorized interface contracts for ignition/door/lock/seatbelt buckle/pressure/seat position/vehicle motion inputs when supplied by the OEM; never guess these signals from the camera if the interface is available.
- Add forgotten-occupant event logic only with validated context inputs and an approved requirements specification.
- Evaluate fleet aggregate analytics and multi-camera scale; transmit only allowed/necessary event data and follow privacy contract.
- Optionally add vector SQL for similar engineering failures and model regression case retrieval after Phase 7's capability gate.
- Expose the vehicle-signal fusion interface on COVESA VSS names via KUKSA.val databroker (door/lock/buckle/seat-pressure as named signals) with a CAN-DBC gateway fallback for legacy programs; the camera never overrides bus signals.
- Structure validation and exports around the Euro NCAP 2026 scoreboard (belt routing + rear occupancy 10, classification incl. airbag/OoP/stature 10, presence incl. CPD + crash-occupancy/eCall 10): per-point evidence tables, worst-case blanket/night cases, and an SD-102-shaped dossier export (sensor principle, coverage maps, classification logic, timing budgets, ML demographics newborn–6YO, thresholds, interference). Camera-only CPD claims stay `FUSION_REQUIRED`; direct-sensing (radar/UWB vitals) arrives through the fusion interface, never as a camera-only claim.
- Create OEM acceptance matrix: camera position, calibration, cabin variants, sunlight, temperature, night/NIR, seatbelt routing, child seats, occlusion, system faults, timeouts, response behavior and validation data requirements.
- Create production-gap plan for functional safety, cybersecurity, privacy/legal, software lifecycle, traceability, quality management, supplier rights, hardware qualification and long-term model maintenance.
- Conduct pilot-readiness review; separate evidence-based demo capabilities from requested future/production capabilities.

**Acceptance criteria**

- A pilot dossier has the system architecture, model/license manifest, validation results, known gaps, privacy/security design, recovery behavior, supported hardware profile and OEM integration dependencies.
- Every production-critical feature has an explicitly named validation owner and evidence plan.
- No unvalidated cabin attribute is represented as production-ready.

---

## 8. Initial acceptance metrics and how to set numbers

Do not blindly copy a latency/AP number from a paper or another GPU into this product. In Phase 0, set a reference target device and agreed operating mode. Then freeze provisional numeric acceptance targets before comparing models.

Recommended metric categories:

| Area | Metric | Requirement format |
|---|---|---|
| Detection | per-class AP/precision/recall and false-negative rate | By class and scenario; set a threshold before final holdout evaluation. |
| Seat state | occupancy-state precision/recall, false occupied/empty rate, transition delay | Seat-level and time-window-based; `UNKNOWN` must be measured, not hidden. |
| Seatbelt / child restraint | class-specific sensitivity, specificity, false-negative and unknown rates | May remain unsupported without sufficient evidence. No universal target until OEM use case and risk are specified. |
| Temporal stability | false state changes per hour, event false alarms/hour, recovery time | Fixed replay and real target camera. |
| Latency | P50/P95 capture-to-state and inference latency | Named device, resolution, sampling rate, provider and model manifest. |
| Throughput | processed frames/s, capture FPS, dropped/stale frames | Keep captured and processed rates separate. |
| Resource | peak RAM/VRAM, model load time, CPU/GPU utilization, power where measurable | Named hardware and sustained-load duration. |
| Availability/recovery | camera reconnect time, SQL recovery time, crash recovery, data loss under defined fault test | State explicit durability guarantees and whether events were buffered or dropped. |
| Privacy | retained media count/age, sensitive fields in logs, deletion behavior | Default no raw-frame persistence; test retention policy. |
| Reproducibility | repeated metrics for same config/data/model | Versioned and machine-readable artifact outputs. |

A proposed early desktop-demo target can be selected after hardware is known (for example, a target processed rate and P95 latency on the developer's machine), but it must be labelled a development target and not an automotive ECU promise. If there is no real cabin evaluation set, the product can pass architecture/integration gates but cannot pass model-accuracy gates.

---

## 9. Documentation and repository artifacts

Inspect the repo first. Reuse equivalent documents or place these under the project's existing docs structure rather than creating duplicates.

| Document | Required contents |
|---|---|
| `ODC_BUILD_PLAN.md` | This phase map, per-task status, links to evidence, dependencies, blockers, next step. |
| `ODC_REQUIREMENTS.md` | Product scope, use cases, supported/unsupported claims, environment, acceptance metrics, assumptions. |
| `ODC_ARCHITECTURE.md` | Component diagram, data flow, failure boundaries, SQL-only PLOMID boundary, threading/backpressure. |
| `ODC_MODEL_SELECTION.md` | Candidate table, model/checkpoint hashes, evaluation evidence, class coverage, runtime metrics, selection decision. |
| `ODC_MODEL_LICENSES.md` | Code/model/checkpoint/dataset/dependency license inventory and approval status. |
| `ODC_DATA_DICTIONARY.md` | Logical schema, field descriptions, enums, version rules, retention and data sensitivity. |
| `ODC_SQL_CAPABILITIES.md` | Verified PLOMID SQL dialect/driver/types/transactions/upserts/index/vector capability and integration tests. |
| `ODC_VALIDATION_PLAN.md` | Test set sources/rights, splits, scenarios, metrics, thresholds, holdout policy, benchmark commands. |
| `ODC_RISKS.md` | Risk, likelihood/impact, owner, mitigation, evidence needed, status. |
| `ODC_DECISIONS.md` | Date-stamped decisions, why, considered alternatives, consequences, review date. |
| `ODC_SECURITY_PRIVACY.md` | Data flow, threat model, SQL auth, retention, media/embedding policy, logging and export controls. |
| `ODC_OPERATIONS.md` | Installation, model downloads, runtime profiles, health checks, recovery, rollback, diagnostics. |

Keep docs concise enough to be maintained. Do not copy large source files into docs or create multiple competing roadmap documents. The master roadmap remains authoritative; supporting docs explain details and evidence.

---

## 10. Quality strategy and test matrix

### Unit tests

- Domain schemas/enums and serialization.
- Frame timestamp and coordinate transform functions.
- Model output adapters, shape validation and invalid output handling.
- State machine hysteresis/dwell/expiry.
- Seat geometry mapping and ambiguous-association behavior.
- Event de-duplication and idempotency.
- Retention decisions and media redaction.
- SQL parameter-building helpers only through safe, constrained interfaces.

### Integration tests

- PLOMID SQL connection, migrations and real SQL reads/writes.
- Insert/query/recovery for current state/events/metrics.
- Unsupported SQL capability fails early.
- Model loading and inference on a small approved fixture.
- Tracker + detector integration and frame timestamps.
- Replay source, disconnect/reconnect, database outage and model-unavailable mode.
- Console queries use backend SQL results, not in-UI static fixtures.

### End-to-end and acceptance tests

- Known empty cabin, one visible occupant, multiple occupants, partial body, head-only/torso-only, overlap, child seat with occupant, empty child seat, blanket, bag, low light, glare, dark clothes, camera blocked, camera moved, malformed stream, SQL disconnect, restart/replay, delayed/out-of-order frame.
- Run each scenario in replay and verify state transitions, `UNKNOWN`/`OCCLUDED` logic, events, timeline and performance output.
- Re-run approved baseline after any model, preprocess, runtime provider, seat geometry, threshold, temporal-rule, schema, or model-manifest change.

### Definition of done for any phase

- Code is formatted/linted and type-checked as required by the repository.
- Tests required by the phase pass; all failures are reported honestly.
- Docs and plan status are updated.
- No untracked test data, model binaries, credentials, private clips, or generated dumps are committed accidentally.
- No unreviewed model or dependency license is silently accepted.
- Runtime has no hidden alternate database or unlabelled fake output.
- Acceptance evidence is linked to the phase status.

---

## 11. Risk register — known hard problems

| Risk | Why it matters | Mitigation / release gate |
|---|---|---|
| Generic pretrained model misses child seat or seatbelt | Generic benchmark classes may not cover these objects; belt is thin and often occluded | No-training open-vocabulary candidate evaluation; mark unsupported/unknown when evidence fails; request authorized sensor signals or revise no-training constraint only through a formal decision. |
| Confusing no detection with empty seat | May lead to false negative occupant state when occluded or model degraded | Explicit visibility/camera-health state and temporal logic; measure false-empty rate. |
| Age/child inference is uncertain | Appearance varies with perspective, pose, clothing, seat and occlusion | Use provisional apparent-size categories only; never exact age or safety/airbag decisions. |
| Seatbelt routing behind body is difficult | Belt may be hidden, similar to clothing, or incorrectly positioned despite being visible | Do not use belt-like line alone; test adversarial examples; remain unknown unless validated and compare with OEM buckle signals. |
| Public model checkpoint license differs from code license | Commercial redistribution may be restricted by weights/data terms | Exact checkpoint license/provenance review; bill of materials; block release for unresolved artifacts. |
| SQL protocol resembles PostgreSQL but dialect differs | Unsupported SQL features can break runtime or corrupt migrations | Capability matrix and live PLOMID integration tests; no assumption based only on wire protocol. |
| Database unavailable at event time | Event/state history could be missing or UI appear falsely current | Explicit degraded status, bounded buffers and documented durability/recovery semantics. PLOMID remains the only persistent store. |
| Hardware/provider drift | A model may behave or perform differently on CPU/NPU/GPU/quantized export | Versioned hardware profiles, parity tests, benchmark each supported combination. |
| Privacy exposure through media/logging | Cabin images and pose data can be sensitive | Process locally, no raw-frame persistence by default, no identity tracking, strict retention/export controls. |
| Demo looks more capable than evidence | OEM/investors may overread unvalidated values | Provenance on every state, confidence/unknown display, live/replay/simulated label, published limitations and validation report. |
| Scope creep | Too many advanced features can prevent a reliable core demonstrator | Phase boundaries, explicit out-of-scope, don't implement later phases early. |
| False sense of automotive readiness | A working prototype is not safety-certified | Dedicated production gap assessment, OEM validation, formal lifecycle and independent safety/security review. |

---

## 12. Release gates and feature labels

Use four capability labels in code, docs, API and UI:

- **`IMPLEMENTED`** — code exists and unit/integration behavior is verified, but this label alone says nothing about real-world accuracy.
- **`EVALUATED`** — tested on a named dataset/hardware/config; results and limitations are published.
- **`PILOT_CANDIDATE`** — passed agreed cabin-specific validation criteria for a named OEM profile, pending OEM pilot governance.
- **`PRODUCTION_QUALIFIED`** — only after formal OEM, functional-safety, cybersecurity, privacy and production qualification evidence. Codex must never assign this status by itself.

A capability may be `IMPLEMENTED` but `UNSUPPORTED` as a product attribute if the underlying pretrained model does not meet acceptance thresholds. This is expected for some child-seat/seatbelt/age requirements under the strict no-training constraint.

---

## 13. Recommended initial delivery order

For a strong early demo without pretending to have solved the hard automotive parts, the recommended sequence is:

1. Phase 0 audit — discover existing repo and verify PLOMID SQL, hardware and legal constraints.
2. Phase 1 contracts/foundation.
3. Phase 2 SQL-only PLOMID persistence.
4. Phase 3 camera and replay.
5. Phase 4 benchmarked pretrained detector.
6. Phase 5 tracking, pose and seat geometry.
7. Phase 6 temporal states/events.
8. Phase 8 console — it can now display honest live data, history and diagnostics.
9. Phase 9 validation harness — make actual supported classes measurable.
10. Phase 7 cabin-specialist zero-shot evaluation — gate each difficult attribute on evidence.
11. Phase 10 optimization for named hardware.
12. Phase 11 hardening/privacy.
13. Phase 12 reproducible packaging.
14. Phase 13 OEM and future feature integration.

Phase 7 appears later in the recommended order than a normal linear roadmap because a polished cabin-specific label is less valuable than reliable ingestion, states, persistence, and a transparent validation harness. It remains a required research gate before any child-seat or belt claim is made. Phase dependencies must be recorded in the status tracker.

---

## 14. What “winning” should mean for the demo/pilot

The demo should demonstrate a product, not a collection of impressive model names:

1. **Real inference:** live/replay input, actual model-generated boxes/keypoints, model ID and provider visible.
2. **Useful temporal intelligence:** state changes only after evidence persists; occlusion becomes `UNKNOWN`, not “empty”; health failure is visible.
3. **PLOMID differentiation:** current seat state and event history are written/read through SQL; engineers can query an event timeline and correlate it to model/config versions.
4. **Operational reality:** camera/database/model failure and recovery work, with no hidden second database.
5. **Measured performance:** local benchmark has hardware, input size, FPS, P50/P95 latency, memory, and known failure cases.
6. **Honest scope:** child-seat and seatbelt features are either backed by class-specific evidence or explicitly provisional/unsupported.
7. **Repeatable proof:** demo scripts and evaluation reports can be replayed by a reviewer without relying on fabricated values or a cloud service.

This is a stronger story than claiming every occupant attribute works with zero training. It shows a sound engineering foundation, a database-native event/state plane, and a clear evidence path to OEM pilot requirements.

---

## 15. External research references checked for this plan

These are starting references; on implementation, re-check current project revisions, checkpoint terms, and exact artifact licenses.

1. **D-FINE official repository and benchmark table:** https://github.com/Peterande/D-FINE — reports public COCO model-zoo scores and documents the measured hardware/runtime setup; also calls out licensing caution for Objects365-derived checkpoints.
2. **RT-DETR / RT-DETRv2 official repository:** https://github.com/lyuwenyu/RT-DETR — Apache-2.0 code repository with published RT-DETR/RT-DETRv2 model-zoo tables; validate exact checkpoint terms and export/runtime path. The repository notes a newer RT-DETRv4 release, which should remain experimental until verified.
3. **Grounding DINO official repository:** https://github.com/IDEA-Research/GroundingDINO — Apache-2.0 code repository for open-set text-conditioned object detection; verify checkpoint licenses separately.
4. **MMPose / RTMPose:** https://github.com/open-mmlab/mmpose and https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmpose — Apache-2.0 code repository and documented RTMPose benchmark/model zoo; pose results are not cabin-specific.
5. **MediaPipe Pose Landmarker:** https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker — documents Lite/Full/Heavy model bundles and 33 body landmarks; verify exact bundle rights and target runtime.
6. **ByteTrack:** https://github.com/ifzhang/ByteTrack — reference implementation for detection association; review selected implementation and license.
7. **ONNX Runtime execution providers:** https://onnxruntime.ai/docs/execution-providers/ — current provider matrix and distinction between deployment support levels.
8. **YOLO-World licensing:** https://github.com/AILab-CVC/YOLO-World — upstream states GPL-3.0; not defaulted into this proprietary-minded plan without explicit legal decision.
9. **Qwen3-VL (optional engineering candidate only):** https://github.com/QwenLM/Qwen3-VL — open model/repository family with multiple sizes; not a primary real-time safety classifier and may not be suitable for constrained ECU deployment.
10. **TICaM in-cabin dataset paper:** https://arxiv.org/abs/2103.11719 — useful evaluation-dataset candidate; validate current data license and task coverage before use.
11. **In-vehicle pose + seatbelt segmentation study:** https://arxiv.org/abs/2204.07946 — describes specialized pose/belt methods evaluated on a private dataset, illustrating why generic benchmark scores do not prove seatbelt performance.

---

## Final project rule

**Do not optimize for the longest feature list. Optimize for the strongest verified system.** Every implementation phase must produce usable, tested software, current docs, trustworthy SQL-backed state, and evidence. A feature that correctly returns `UNKNOWN` when evidence is weak is better than a polished but fabricated positive/negative result. PLOMID stays the only database, SQL stays the only database access path, and pretrained inference stays training-free throughout this plan.
