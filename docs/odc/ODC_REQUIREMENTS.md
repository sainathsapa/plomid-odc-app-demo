# ODC requirements (v1.0, locked 2026-10-13)

## Buyer asks (T-Hub / global auto corp) → product answer
1. Seat occupancy, all seats → temporal per-seat states + SQL history.
2. Adult / child / child-seat / object → gated candidates with per-class
   evidence; `PROVISIONAL`/`UNSUPPORTED` where unproven; fusion hooks.
3. Seatbelt + posture → belt-candidate + pose geometry + OEM buckle fusion;
   camera never overrides the bus.
4. Existing cameras → fisheye + NIR/day-switching + auto-calibration wizard.
5. Real-time → CPU baseline + adaptive sampling + named P50/P95 (dev laptop).
6. Safety-system integration → REST/WS/MQTT + OpenAPI + SDK + VSS fusion.
7. ECU / central compute → named SoC matrix + offline install + signed OTA.
8. Flexible cabin + future → versioned geometry, DMS-lite, parked-guardian,
   personalization-without-identity, fleet-lite.

## Non-negotiables
- PLOMID-only persistence, SQL-only boundary. No second database.
- Pretrained inference only; no training/fine-tuning; eval labels never
  update weights.
- Uncertainty explicit (`UNKNOWN/OCCLUDED/STALE/UNSUPPORTED`); never
  "no detection" → "empty". No `SIMULATED`/mock output presented as live.
- No actuation (airbags/pretensioners); advisory states only.
- No face recognition / identity embeddings; metadata-only, no raw video
  retention (GSR closed-loop compatible).
- Capability labels: IMPLEMENTED → EVALUATED → PILOT_CANDIDATE;
  PRODUCTION_QUALIFIED only by OEM process, never self-assigned.

## Out of scope (must stay unclaimed)
Breathing/heartbeat CPD camera-only; exact age; belt-fastened certainty;
certification (26262/ASPICE/21434) without evidence; ECU latency promises
from laptop numbers.

## Acceptance metric categories (targets frozen pre-holdout, Phase 9)
Per-class P/R/F1 + FN rate · seat-state precision/recall + transition delay ·
temporal false-alarms/hour · P50/P95 capture-to-state + inference (named
device/res/provider) · processed vs captured FPS + drops · RAM/VRAM + load
time · reconnect/recovery times + buffered-vs-dropped semantics ·
retention/media counts + deletion proof · repeat-run reproducibility.

## AISIN ODC traceability — MVP audit 2026-10-23

| AISIN requirement | Implementation | UI / contract | Test / evidence | Status |
|---|---|---|---|---|
| Occupancy across configured seats | `Service` → tracking → `SeatStateEngine`; versioned `SeatLayout` | Live cabin and Seat map | `tests/unit/test_service.py`, `test_seat_geometry.py`; standard layout contract | PARTIALLY_IMPLEMENTED |
| Adult, child, child-seat, object, unknown | Generic detector plus explicit candidate/unknown reason states; no calibrated age classifier | Seat attributes and seat detail | `tests/unit/test_demographics.py`, `test_segmentation_attributes.py`; model manifest/eval docs | PARTIALLY_IMPLEMENTED |
| Seatbelt and posture observations | Belt candidate tracker and optional pose path; camera cannot prove buckle fastening | Seat attributes, reasons and diagnostics | `tests/unit/test_belt.py`, `test_tracking_pose.py` | PARTIALLY_IMPLEMENTED |
| Stable seat association over time | Normalized polygons, track-first/seat-first association, stable canonical seat IDs | Live state/API/events use `seat_id` | `tests/unit/test_seat_geometry.py`; `layout_to_json` round-trip | VERIFIED |
| Camera and prerecorded video input | OpenCV live/replay sources with decode health and reconnect paths | Live/Replay source selector | `tests/unit/test_capture.py`; integration replay tests when dependencies are installed | IMPLEMENTED |
| Actual CV inference, tracking, state logic | ONNX Runtime manifest/checkpoint loading → detections → tracking → temporal state | Model/provider health and degraded status | `tests/unit/test_detector.py`, `test_service.py`; runtime loader | IMPLEMENTED |
| Current state, evidence, health, events | Service hub plus PLOMID current/event reads; unknown/stale are explicit | Live, Seat map, Timeline, Diagnostics | Console integration tests and SQL boundary docs | IMPLEMENTED |
| Configurable cabin layouts | Versioned standard 2/5/7 seeds and custom calibration; one-based operator labels | Layout selector, Seat map editor, Config page | `tests/unit/test_layouts.py::test_standard_layouts_have_one_based_multi_row_identity` | VERIFIED |
| Structured query/exportable outputs | PLOMID SQL current states/events/layouts; JSON APIs and dossier | `/api/seats`, `/api/events`, dossier | `tests/integration/test_console_app.py`, SQL capability docs | PARTIALLY_IMPLEMENTED |
| OEM integration boundary | Domain contracts, health/events, documented SQL boundary; no actuation | Dossier and prototype disclaimer | `docs/odc/ODC_OEM_ACCEPTANCE.md`, domain contracts | PARTIALLY_IMPLEMENTED |
| Driver attention | Driver-seat-only DMS-lite preview: face landmarks, blink/sleep proxy, head-turn distraction, yawning and phone-near-driver candidate; health/error state exposed | Live DMS column, driver seat identity, diagnostics health | `tests/unit/test_dms.py`, service/console tests, local face-model smoke when weights exist | PARTIALLY_IMPLEMENTED |
| Safety recommendations / airbag integration | No airbag control or certified recommendation path | Must remain future/unsupported | ODC requirements and OEM acceptance docs | NOT_IMPLEMENTED |

Verification rule: `IMPLEMENTED` means the code path exists and is exercised;
`VERIFIED` additionally has a focused passing test or deterministic contract
check in this repository. Generic detector scores are not age or belt
probabilities, and missing evidence never becomes a proven empty seat.
