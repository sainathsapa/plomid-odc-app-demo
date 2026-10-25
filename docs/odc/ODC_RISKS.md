# ODC risks (v1.0, locked 2026-10-13)

| # | Risk | Impact | Owner | Mitigation / gate | Status |
|---|---|---|---|---|---|
| R1 | Submission deadline unknown | Schedule cannot be planned | Owner | Ask T-Hub now; default 4-week thin-slice plan | OPEN |
| R2 | Consent actor clip missing | No golden-path ground truth | Owner | Record 30 s laptop clip; fallback F1 as demo | OPEN |
| R3 | Evaluator expects classifier certainty | PROVISIONAL reads as weakness | ODC | App frames gated-evidence as safety feature; per-class reports | OPEN |
| R4 | No labelled in-cabin eval set | Ph 7/9 accuracy gates blocked | ODC | SVIRO + TICaM-synth + Pexels negatives; request OEM clips | OPEN |
| R5 | NIR/fisheye domain gap | Day-RGB numbers don't transfer | ODC | Night/NIR + fisheye buckets mandatory in Ph 9; undistort first | OPEN |
| R6 | Checkpoint licenses unresolved | Legal blocks release | Owner | Per-card review before any download; BOM in ODC_MODEL_SELECTION | OPEN |
| R7 | No ECU hardware | Ph 10 unmeasurable | Owner | Laptop reference locked; SoC matrix entries only | ACCEPTED |
| R8 | TLS absent | No transport-security claim | PLOMID | Loopback-only dev; re-verify pre-pilot (D9) | ACCEPTED |
| R9 | False-empty seat state | Safety-critical miss | ODC | Visibility-gated states + stale expiry + FN-rate metric | MITIGATING (design) |
| R10 | Demo overclaim | Lost dossier trust | ODC | Honesty rule + labels + benchmark card from real runs only | MITIGATING (design) |
| R11 | Scope creep (DMS/fleet/CDK) | Core demo slips | ODC | Phase gates; DMS-lite/parked-guardian are stretch, clearly marked | WATCH |
| R12 | PLOMID behavior drift | Assumption rot | ODC | Re-run `scripts/smoke_plomid.py` on every PLOMID upgrade | MITIGATING (script) |
| R13 | Synthetic eval licensing | Paid vendors unbudgeted | Owner | Free tiers first (SVIRO/TICaM); vendor eval gated on budget | OPEN |
| R14 | DETR duplicate boxes → twin tracks | ODC | Duplicate-suppression study pre-Ph6; seat-first absorbs twins meanwhile | OPEN |
| R15 | Seat-ROI grounding variant + synthetic misuse renders | ODC | Phase 9 follow-up; full-frame baseline recorded | OPEN |
| R16 | Night/NIR visibility research (ROI brightness insufficient) | ODC | Holdout finding; Phase 10/13 | OPEN |
| R17 | Calibrated quantization procedure + owner sign-off | Owner | Dynamic-INT8 rejected on parity; static path needs approval | OPEN |
