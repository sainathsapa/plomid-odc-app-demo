# Model inventory

Vector status: RT-DETRv2 emits detections, not semantic embeddings. ODC stores
the normalized detection box as `VECTOR(FLOAT32,4)` for frame evidence and R&D
queries. Identity/face embeddings are not stored by design.

## Semantic embedding status

The active RT-DETRv2 and D-FINE ONNX artifacts expose detection outputs (class
scores and box coordinates), not a documented, validated occupant-semantic
feature tensor. Therefore the application does **not** pretend that a box is a
semantic embedding, does not invent an embedding from confidence values, and
does not store face or identity vectors. The `odc_frame_detection.bbox` column
is a real PLOMID vector containing geometry only: `[x_min, y_min, x_max,
y_max]`.

The assistant (local Ollama by default, or a configured OpenAI-compatible
hosted provider) is a separate, explicit operator explanation tool. It receives
structured current-state evidence and cannot create detections, change seat
state, or write model-generated claims into PLOMID. A hosted provider receives
that evidence outside the instance; disable the assistant when external
processing is not permitted. A future
semantic-search feature requires a separately approved feature model, fixed
dimension, provenance/license review, benchmark evidence, retention policy,
and a versioned migration; it is not part of this MVP.

| Model/component | Repository use | Evidence/status |
|---|---|---|
| RT-DETRv2-S COCO | Primary generic person/object detector via ONNX Runtime | Implemented; generic classes only; in-cabin classification not proven |
| D-FINE-S COCO | Manifest/challenger/runtime assets | Available for evaluation, not the default service model |
| Grounding DINO | Optional open-vocabulary localization | Secondary/exploratory; not a validated child/belt classifier |
| MediaPipe pose task | Optional body keypoints/posture evidence | Candidate posture evidence; not safety-qualified |
| SAM ViT-B | Optional precise segmentation branch | Background mask refinement; not semantic occupant classification |
| MediaPipe face task | Optional DMS-lite preview | Preview candidates only; no identity persistence |
| ByteTrack-style tracking | Short-term track continuity | Session-local association; no persistent person identity |

Exact checkpoint hashes, providers, licenses, and runtime requirements are in
`models/manifests/` and the model-selection documentation. Do not treat public
COCO metrics as cabin accuracy.

