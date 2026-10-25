# ODC ontology v1 — evidence tags (adapter spec, locked 2026-10-13)

Model labels are NOT product tags. Every detector/pose/VLM output passes
through this reviewed adapter before it becomes product evidence. Tags map
into the `odc/domain.py` enums; no model class ID may appear in API/UI/SQL
as permanent semantics.

## Principles
1. A tag is `SUPPORTED` only with a named validation set + pass threshold.
   Everything else is `PROVISIONAL` (shown, caveated) or `UNSUPPORTED`.
2. Absence of evidence is never evidence of absence — `NOT_DETECTED`
   ≠ confirmed-absent. Empty/oosive states are DERIVED over time, not detected.
3. Zero-shot prompt strings are versioned config (`prompt_set_version`
   recorded on every event that uses them). Changing a word = new version.
4. Derived tags cite their rule + inputs (e.g. `size ← bbox/seat-area +
   temporal`). Thresholds live in config, never in prose.

## A. COCO-native evidence (RT-DETRv2-S / challengers)
| COCO class | ODC tag | Meaning | Caveat |
|---|---|---|---|
| person | PERSON | visible human body | size/age NEVER inferred from this alone |
| chair, couch | SEAT_LIKE | seat-shaped region | car seats trigger this; weak evidence, never an occupant |
| backpack, handbag, suitcase | LUGGAGE_OBJECT | object on/near seat | occupant-vs-object disambiguation needs seat ROI + temporal |
| bottle, cup | SMALL_OBJECT | clutter candidate | |
| book, laptop, cell phone | HAND_OBJECT | incl. phone-in-hand (DMS-lite) | phone needs hand/proximity rule, not class alone |
| teddy bear | SOFT_TOY | child-presence hint only | NEVER a child detector |
| umbrella, sports ball, etc. | LOOSE_OBJECT | generic clutter | |
| all other 60+ classes | IGNORED | logged, never productized | e.g. animals, food, vehicles |

## B. Cabin tags with NO COCO class (zero-shot / derived / fused)
| Tag | Source | Meaning | Gate |
|---|---|---|---|
| INFANT_SEAT_RF | prompt `rear facing infant car seat` | rear-facing baby seat | UNSUPPORTED (D19: fires on every vehicle seat, overlapping score bands) |
| CHILD_SEAT_FF | prompt `forward facing child car seat` | toddler seat | UNSUPPORTED (D19) |
| BOOSTER_SEAT | prompt `booster seat in a car` | booster | UNSUPPORTED (D19) |
| CHILD_RESTRAINT_ANY | union of the three | any CRS present | UNSUPPORTED (D19) |
| OCCUPIED_CRS | CRS tag + PERSON overlap + temporal | child possibly in seat | UNSUPPORTED (D19); never exact age |
| BELT_TORSO | prompt `seat belt across chest and torso` + pose geometry | belt-like evidence | UNSUPPORTED (D19: seat seams fire equally); CANDIDATE via geometry+fusion only |
| BELT_BEHIND | prompt `seat belt behind back` | misuse evidence | UNEVALUATED (no misuse footage) |
| BELT_BUCKLE | prompt `seat belt buckle` + OEM buckle signal | buckle evidence | UNSUPPORTED (D19); camera never overrides bus |
| BLANKET | prompt `blanket covering seat` | occlusion cause | UNEVALUATED (no blanket footage) |
| BAG_ON_SEAT | prompt `bag on car seat` + LUGGAGE_OBJECT | object-occupied seat | UNEVALUATED (O2 pending); COCO luggage stays the source |
| FEET_ON_DASH | foot/leg keypoints inside dash ROI | NCAP OoP evidence | needs pose + dash polygon |
| AIRBAG_PROXIMITY | head keypoints inside airbag ROI | NCAP OoP evidence | distance in normalized units, temporal |
| HEAD_POSE_DIR | head keypoint geometry (yaw bins) | DMS-lite attention hint | preview; no gaze claim |
| PHONE_IN_HAND | HAND_OBJECT(cell phone) + hand proximity + temporal | DMS-lite | preview |
| EMPTY_CANDIDATE_SEAT | seat ROI: no PERSON/CRS/object evidence over dwell window | derived state | requires healthy camera + visibility |
| APPARENT_SMALL / LARGE | bbox-area/seat-area ratio + temporal | provisional size | UNSUPPORTED as classifier (D19: absolute areas meaningless across viewpoints); seat-relative rule stays provisional |

## C. Explicitly UNSUPPORTED (red lines)
Exact age · breathing/heartbeat from camera · belt-FASTENED certainty ·
sleeping-vs-awake · identity/recognition · emotion/intent · airbag-grade
stature · live-vs-doll respiration.

## D. Grounding DINO prompt set v1 (`prompt_set_version: gd-v1`)
Exact strings, do not paraphrase without versioning:
1. `a person sitting in a car`
2. `rear facing infant car seat`
3. `forward facing child car seat`
4. `booster seat in a car`
5. `seat belt across chest and torso`
6. `seat belt behind a person's back`
7. `seat belt buckle`
8. `blanket covering a car seat`
9. `bag on a car seat`
10. `child in a car`
Run at low rate on seat ROIs only (never full-frame scan, never onboard
real-time); outputs namespaced `vlm:` until promoted by Ph7 evidence.

## E. Tag ↔ footage coverage (validation map for Ph7/Ph9)
| Tag | Positives | Hard negatives |
|---|---|---|
| PERSON | F1, R1, D1, D2, S2, N1 | E1, E2 |
| INFANT_SEAT_RF / CHILD_SEAT_FF | C1, C2, S3(backup) | E1, O2, adult-only R1 |
| BELT_TORSO / BELT_BUCKLE | S1, S2, C1, S3 | N1 (dark), O1 |
| BLANKET | synthetic (Anyverse/SVIRO eval) | F1, R1 |
| BAG_ON_SEAT / LOOSE_OBJECT | O1, O2 | E1, E2 |
| EMPTY_CANDIDATE_SEAT | E1, E2 | F1, R1, C1 |
| FEET_ON_DASH / AIRBAG_PROXIMITY | staged actor clip (owner) + synthetic | normal-seated F1/D2 |
| HEAD_POSE_DIR / PHONE_IN_HAND | D1, D2, O2 | N1 |
| OCCUPIED_CRS | C1, C2 | empty-seat + SVIRO empty-CRS renders |

`UNKNOWN` coverage is mandatory per tag: every tag must show evaluated
unknown-rate (occlusion/night/IR), not just precision/recall.
