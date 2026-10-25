# Detector runtime and preprocessing

Implementation: `odc/runtime.py`, `models/manifests/*.json`.

The active baseline is RT-DETRv2-S COCO through ONNX Runtime. BGR frames are
converted to RGB, resized to 640×640 with PIL bilinear resampling, normalized
to float32 NCHW, and passed to the manifest-approved provider order. Provider
selection prefers configured accelerator providers and falls back to CPU only
when available. Known blocked/degenerate providers are rejected.

The runtime decodes sigmoid logits and normalized `cx,cy,w,h` boxes into
model-independent `Detection` contracts. Raw detector confidence is retained
as evidence; it is not presented as a calibrated probability. Missing ONNX
weights, invalid manifests, unavailable providers, or numerically degenerate
outputs fail closed and set model health to unavailable.

