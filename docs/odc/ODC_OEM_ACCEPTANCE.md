# ODC OEM acceptance matrix (v1, locked 2026-10-13)

Status per item: MET (evidence linked) · PARTIAL (works, limits stated) ·
GAP (named owner + evidence plan) · N/A (out of scope, stated).

## Camera & calibration
| Item | Status | Evidence / plan |
|---|---|---|
| Camera position variants (mirror/dome/pillar) | GAP (OEM) | needs target-vehicle geometry + clips |
| Fisheye undistort | PARTIAL | OpenCV path; fisheye-aware deferred (R) |
| NIR day/night switching | GAP (OEM) | needs NIR clips (R16) |
| Calibration wizard | GAP (ODC) | config schema done; UI editor scheduled |
| Seat variants (2-row/3-row, LHD/RHD) | PARTIAL | versioned layouts; 2 profiles demoed |

## Cabin scenarios
| Item | Status | Evidence / plan |
|---|---|---|
| Occupancy all seats | MET | dev seat .816, 0 wrong; holdout .667 |
| Child seats (empty + occupied) | PARTIAL | CRS UNSUPPORTED as classifier; SVIRO eval scheduled |
| Seatbelt routing/misuse | GAP (OEM+few) | candidate+fusion only; needs buckle bus + misuse footage |
| Occlusion / blankets / clutter | PARTIAL | UNKNOWN-gating tested; blanket renders scheduled |
| Sunlight / glare / temp soak | GAP (OEM) | needs vehicle + chamber time |
| Night | GAP | holdout caught it (2 wrong); NIR path R16 |

## System behavior
| Item | Status | Evidence / plan |
|---|---|---|
| Real-time bounds (P50/P95) | MET | laptop profile; ECU targets matrix entries |
| Camera/model/SQL faults + recovery | MET | fault matrix tested; chaos documented |
| Upgrade / rollback | MET | V1→V2 tested; snapshot drill recorded |
| OTA model packs | PARTIAL | Uptane-shape signing specified; HSM flow GAP (owner) |
| eCall / restraint interfaces | GAP (OEM) | contracts defined; vehicle side pending |

## Data & process
| Item | Status | Evidence / plan |
|---|---|---|
| Validation dossier (SD-102-shaped) | MET | dossier export + validation plan |
| Privacy/GDPR (metadata-only) | MET | closed-loop design; DPIA template GAP (owner legal) |
| ASPICE / ISO 26262 / 21434 | GAP | see ODC_PRODUCTION_GAP.md — no claims made |
