# PLOMID Cabin UI Phase Freeze

Status: **READY FOR UI IMPLEMENTATION**  
Freeze date: 2026-10-23  
Scope: operator UI redesign only

## Frozen backend contracts

The UI may consume and present these contracts, but must not silently change
their meaning:

- Canonical seat keys are stable one-based IDs: `seat_1`, `seat_2`, etc.
- `row` and `display_number` are one-based user-facing values.
- Standard five-seat layout is two seats in Row 1 and three in Row 2.
- Standard seven-seat layout is two, three and two seats across Rows 1–3.
- Custom layouts may contain one-seat rows for buses, vans and special cabins.
- Front-row automatic proposals are capped at two seats.
- RHD/LHD is explicit layout metadata. Automatic steering detection remains a
  candidate proposal requiring operator review.
- Seat association is geometry/calibration based. Detection-array order is not
  a seat identity.
- `UNKNOWN`, `UNSUPPORTED`, processing failure and positive/negative evidence
  must remain visually distinct.
- Current state and historical events are separate concepts.
- PLOMID is the only durable database boundary.
- Per-frame detection geometry is stored in
  `plomid_odc.odc_frame_detection.bbox` as `VECTOR(FLOAT32,4)`.
- The bbox vector is geometry, not a semantic embedding.
- The local Mistral assistant is explicit user action only. It cannot create
  detections, change seat state or write AI-generated claims to PLOMID.
- Out-of-scope assistant questions return a scope response and are not sent to
  Ollama.

## Stable UI endpoints

- `GET /api/live/state` — current transient live state and latest frame.
- `GET /api/frames/history` — bounded in-memory replay index.
- `GET /api/frames/detections` — durable PLOMID frame detections.
- `GET /api/seats` — current or selected-session seat states.
- `POST /api/service/start` — starts/restarts a perception session.
- `POST /api/service/stop` — stops the active session.
- `POST /api/assistant/ask` — explicit local Mistral request.
- `GET /api/health` — application/component health.

## UI-only work allowed after freeze

- Layout, typography, spacing, responsive behavior and accessibility.
- Seat-map visual treatment and selection states.
- Timeline, evidence and event navigation.
- Assistant panel presentation and error/loading states.
- Diagnostics presentation and operator guidance.
- Screenshots, browser tests and visual regression tests.

## Backend changes requiring a new review

Do not change these during ordinary UI work:

- Seat IDs, row numbering, physical-role mapping or layout migration rules.
- Auto-layout grouping/capacity logic.
- Inference thresholds, tracking, seat association or state transitions.
- PLOMID schema, vector dimensions or persistence semantics.
- Assistant scope policy or automatic invocation behavior.

## Launch modes

Idle console, database healthy:

```bash
./start.sh
```

The UI should show `DISCONNECTED` for perception and `PLOMID healthy` while
waiting for the operator to choose a source and press Start.

Immediate default replay:

```bash
./start.sh --auto-start
```

The startup script provisions `plomid_odc` and applies migrations before the
console starts. If PLOMID restarts while a session is running, the service
re-probes and recovers persistence automatically.

## Verification at freeze

The complete repository test suite passed. Focused console, layout, assistant,
foundation and persistence tests also passed after the final idle-health and
auto-layout corrections.

Known product boundaries remain unchanged: semantic detector embeddings,
airbag control, safety qualification and production automotive certification
are outside this MVP.
