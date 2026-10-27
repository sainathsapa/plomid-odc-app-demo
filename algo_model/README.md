# PLOMID Cabin algorithms and model guide

This folder documents the implementation that is actually present in this
repository. It is an engineering reference, not a claim of automotive safety
qualification. “Candidate” means visual evidence only; it does not mean a
calibrated age, buckle, or occupant-protection verdict.

## Runtime chain

`source → decode/quality → preprocessing → RT-DETR inference → post-processing
→ short-term tracking → calibrated seat association → temporal state engine →
PLOMID current state/events/metrics → live console`

Optional branches are pose, belt-stripe candidate detection, classical masks,
SAM masks, face/DMS preview, and appearance preview. They never replace the
core uncertainty contract.

The optional assistant is an explicit operator tool only.
`POST /api/assistant/ask` sends the user's question plus current structured
seat/detection context to the configured provider: local Ollama by default, or
an OpenAI-compatible hosted service such as Pollinations. It is never called
for each detection, frame, alert, or persistence write, and its response is not
treated as perception truth.

The live workspace exposes this as **Ask AI assistant**. If the configured
provider or model is unavailable, the UI reports that failure; it does not fall
back to fabricated text. The active detector stores real bounding-box geometry
in PLOMID's `vector(4)` type. Semantic AI embeddings from the detector are not
implemented because the loaded checkpoints do not provide a validated semantic
feature output.

## What is persisted

PLOMID stores session metadata, model manifests, current seat states, occupant
attributes, state events, health/metrics, per-frame boxes/confidence vectors,
layouts, video fingerprints, probe results and bounded analytics snapshots
through the documented SQL boundary.

The service keeps a bounded in-memory frame ring for operator recap. Raw
continuous video and every JPEG frame are deliberately not written to PLOMID;
the repository policy is structured SQL truth plus source media outside the
database. This avoids turning PLOMID into a video archive. A future evidence
retention feature must be explicitly approved, bounded, and versioned.

## Documents

- [customer/R&D technical specification](customer-rd-technical-spec.md)
- [automatic seat-map proposal](auto-seatmap.md)
- [capture and frame pipeline](capture-frame-pipeline.md)
- [detector runtime and preprocessing](detector-runtime.md)
- [post-processing and perception](perception-postprocessing.md)
- [tracking and seat association](tracking-seat-association.md)
- [temporal state reasoning](temporal-state.md)
- [optional attributes](attributes-pose-belt.md)
- [persistence and recap](persistence-recap.md)
- [console data flow](console-ui.md)
- [model inventory and maturity](models.md)
- [capability limits](limitations.md)

