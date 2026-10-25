# ODC architecture (v1.0, locked 2026-10-13)

Python-only. PLOMID is the sole database, reached over PG wire via a
`persistence` repository boundary. No other durable store.

```
capture → frame-pipeline → model-runtime → perception → tracking/seats
  → state-engine → persistence (PLOMID SQL) → api/console → validation
```

## Layers (own package each, Phase 1)
1. `config` — pydantic-settings + YAML, versioned vehicle/camera/seat/model DSL.
2. `domain` — Pydantic v2 contracts: frame, detection, keypoints, seat/
   occupant state, visibility, event, health, model manifest, SQL errors.
3. `capture` — one `Source` interface: live (OpenCV/GStreamer) + file replay +
   test fake (labelled). Monotonic seq + UTC ts + health metadata; bounded
   queues, drop policy, reconnect/backoff.
4. `pipeline` — resize/color/normalize, ROI crops with inverse transforms.
5. `runtime` — abstract inference interface; ONNX Runtime providers + native
   fallback; parity-gated export; provider discovery + explicit fallback.
6. `perception` — model-independent detections/keypoints + raw scores +
   provenance; open-vocab outputs namespaced, never durable enums.
7. `tracking` — ByteTrack-style default, OC-SORT challenger; session-local
   ephemeral IDs; seat-polygon association with ambiguity → UNKNOWN.
8. `seats` — normalized polygons + camera profile versioning; calibration
   wizard + drag-vertex edit mode (reviewed, versioned).
9. `state` — per-seat hysteresis/dwell/debounce/stale-expiry; reason codes +
   evidence refs + model/config versions on every transition; out-of-order
   safe; current-state reconcilable from event history.
10. `persistence` — SQL-only repository (`health_check`, sessions, camera
    status, state snapshots, `append_event` idempotent via ON CONFLICT,
    metrics batches, validation results, optional embeddings via HNSW).
    Owns timeouts/retries; never leaks connections upward.
11. `api` — FastAPI: REST state/events/config + SSE channels + health/ready.
12. `console` — Jinja2 + HTMX + Alpine.js + vendored ECharts + Tailwind;
    shell + 7 screens + evaluator tour + benchmark card export.
13. `validation` — replay runner, annotation format (eval-only), holdout
    discipline, machine-readable reports.
14. `ops` — logging (structlog, no frames/secrets), supervision, bounded
    recovery, diagnostics bundle (redacted), install/OTA/CDK.

## Data flow
Frames are transient. Only states, events, config versions, model manifests,
metrics, and validation results persist — all as typed relational rows with
JSONB details and (gated) HNSW vectors for failure retrieval.

## Failure boundaries
Camera / model / SQL / process fail independently and visibly:
DEGRADED banners, STALE expiry toward UNKNOWN, bounded buffers, documented
buffered-vs-dropped semantics. Fault-injection (chaos) suite in Phase 11.
