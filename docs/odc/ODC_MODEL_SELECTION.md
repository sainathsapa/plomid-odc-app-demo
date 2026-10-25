# ODC model selection (v1.1, evidence baseline 2026-10-23)

Rule: inference only. No training/fine-tuning/LoRA/distillation pipelines.
Thresholds live in versioned config, never in weights. Per-checkpoint card
re-verified before freeze; `_coco` vs `_obj365` lineage decides legality.

## Onboard portfolio
| Function | Baseline → challenger | Status 2026-10-13 | Cabin note |
|---|---|---|---|
| Person/object detection | RT-DETRv2-S (COCO, Apache-2.0 ckpt) → D-FINE-S/M (`_coco` only) → RTMDet-S (edge-latency axis) | RT-DETRv2-S APPROVED: `PekingU/rtdetrv2-s-r18vd@56509617`, sha `d18309d0…`, ONNX parity max\|dbox\|≤3e-5, CPU provider pinned, CoreML rejected w/ evidence | COCO person/chair/bag/phone/bottle; no child-seat/belt classes |
| Pose | RTMPose-S/M preferred-but-blocked (no official ONNX; third-party ports unverified) → MediaPipe Landmarker Lite working fallback | REVIEW_REQUIRED (bundle commercial terms unverified; per-card check before distribution) | 33 landmarks, crop→frame mapping verified; absence never becomes limb-absent |
| Tracking | ByteTrack-style → OC-SORT | APPROVED (pattern; MIT impls) | association only, no weights |
| Runtime | ONNX Runtime → native | APPROVED (packages) | TRT pinned + mAP re-validated per export |
| Child/infant and child-seat candidate stage | Grounding DINO SwinT OGC, local bounded worker, split CRS and occupant prompts | INTEGRATED / EVAL-ONLY WEIGHTS | 662 MiB local checkpoint; CPU fallback; raw candidate scores are not probabilities; checkpoint lineage is not commercially cleared |

## Restricted local specialist and offline tools
| Tool | Purpose | License flag |
|---|---|---|
| Grounding DINO SwinT (official ckpt sha 3b3ca256, code @856dde20) | bounded local specialist worker for child/child-seat candidate evidence | EVAL-ONLY (O365 lineage); local-use review required; not a certified age classifier |
| OpenCLIP ViT-B-32 (LAION) | second-opinion scorer, disabled by default | eval-only; S1→belt 1.0 / C2→seat 0.99 / E1→empty 0.94 on n=3 frames (signal, not evaluation) |
| Anyverse / SKY ENGINE | synthetic belt-misuse / blanket / OOP eval sets | commercial paid; eval-only |

## Rejected
YOLO-World (GPL-3.0) · Ultralytics package path (AGPL pull-in) ·
Objects365-derived checkpoints (uncleared) · RT-DETRv4 (immature, track only) ·
face/person-identity embeddings (banned by requirements).

## Benchmark gates (Phase 4/9)
Same clips + target device for all candidates; per-class P/R/F1 + FN
examples + confidence histograms; P50/P95 + sustained FPS + RAM/VRAM +
cold start; ONNX parity report or path disabled. Cabin accuracy decides;
COCO AP never quoted as product performance (plan figures already stale:
D-FINE zoo has moved).

## Amendment 2026-10-14 (face previews, SAM masks, belt geometry)
| Function | Choice | Status 2026-10-14 | Cabin note |
|---|---|---|---|
| Face landmarks | MediaPipe FaceLandmarker (float16 task, Google Storage pinned URL) | REVIEW_REQUIRED (bundle commercial terms unverified; same posture as pose fallback) | 478 pts + blendshapes (eyeBlink/jawOpen) + pose matrices; verified yaw/pitch ≈ 0 on frontal C2; drowsy/distracted/phone stay preview candidates |
| Instance masks (precise mode) | Meta SAM ViT-B (`facebook/sam-vit-base`, Apache-2.0) via transformers, box-prompted | APPROVED for optional use (checkpoint license Apache-2.0; weights local-only, gitignored) | iou 0.97/0.87 on C2 probe; ~5-7 s/frame CPU → background single-slot worker only, labelled with frame seq; fast GrabCut stays default |
| Belt strap | Classical LSD + collinear merge + parallel-pair (no weights) | APPROVED pattern (deterministic, unit-tested on synthetics) | High-contrast straps detect (S2 probe 0.9); soft/profile webbing often indeterminate; absence needs frontality (pose/DMS face), else Unknown |
Fetch: face task via `curl -L -o models/mediapipe-face/face_landmarker.task
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`;
SAM via `huggingface_hub.snapshot_download("facebook/sam-vit-base")` into
`models/sam-vit-base/`. Both degrade to documented-off when absent.

## Amendment 2026-10-23 (specialist integration and state contract)

The specialist path is connected to the live/replay service loop through one
bounded pending job. Obsolete frames are coalesced, every result carries
session/source-frame provenance, and duplicate prompt boxes are removed.
Child/infant occupant evidence and child-seat object evidence are separate
attributes; generic COCO `person` remains generic.

The local `C1_baby_carseat.mp4` test returned child/infant-seat candidates.
The same model also returned a rear-facing-seat candidate on a reviewed empty
cabin frame. Therefore an open-vocabulary child-seat result is exposed as a
candidate and is not promoted to confirmed `YES` without corroborating
occupant, configured-region, or independent restraint evidence. This is an
integration/regression result, not a production accuracy claim.
