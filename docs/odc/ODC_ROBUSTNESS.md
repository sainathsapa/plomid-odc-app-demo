# ODC random-video robustness (locked 2026-10-13, measured)

Evaluators will feed arbitrary footage. The contract, proven on 11 clips
(`odc analyze`, 30–60 frames each):

| Input class | Behavior (measured) | Example |
|---|---|---|
| Cabin, calibrated layout | Seat states + transitions; LIKELY within ~1–2 s virtual time | F1: 3/3 LIKELY; R1: 3/4 LIKELY (distant driver abstains); S2/E1 correct |
| Cabin, no layout (C1/C2/S1) | Person evidence reported; zero seat claims, 0 transitions | C1: 88 persons, no seats |
| Night/dark (N1) | Partial detection (41/60 frames) → UNKNOWN, never false-empty | driver LIKELY at 60f, UNKNOWN at 30f |
| Street crowd (X1) | 45 persons/frame + vehicles; zero seat claims (uncalibrated) | 1131 detections, 0 transitions |
| Busy office (X2) | ~25 real people tracked as persons; zero seat claims | title said "empty" — verify footage, don't trust titles |
| Animal + partial person (X3) | dog correctly `dog`; petting arm fires person (expected) | no identity, no claims |
| Corrupt/missing file | `CaptureError`, exit 1, no hang | tested |
| fisheye/NIR/vertical | resize path handles any size; accuracy unmeasured → UNKNOWN bias holds | limits stated, not tested |

## Rules that make this safe
1. **Uncalibrated geometry claims nothing.** No layout → empty seat list,
   UNCALIBRATED banner, person evidence still flows. (A crash here was
   found live and fixed with a regression test.)
2. **Dwell is time-based.** 30-frame windows (~1 s) may legitimately show
   UNKNOWN/LIKELY; OCCUPIED needs sustained evidence. Short runs are
   inconclusive by design, not failures.
3. **Duplicates → ambiguity → UNKNOWN.** DETR twin boxes (R14) degrade to
   abstention, never to wrong seats.
4. **Night misses → UNKNOWN; hallucinations (car/cat on dashboards) stay
   below product thresholds** and are flagged in validation.
5. **Nothing is stored** by analyze (transient probe); serve persists via SQL.

## Evaluator workflow
```sh
.venv/bin/python -m odc.cli analyze THEIRFILE.mp4 [--layout NAME] [--max-frames 60]
```
Calibrated layouts live in `configs/seats/`; without one the output says
`calibrated: false` and seat claims stay empty. That is the product working.
