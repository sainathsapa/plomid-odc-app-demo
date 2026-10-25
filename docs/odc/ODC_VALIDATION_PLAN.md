# ODC validation plan (v1, locked 2026-10-13)

## Dataset v1 (`eval/datasets/v1.yaml`, labels `eval/annotations/v1.json`)
22 hand-labeled frames (viewed, box precision ~0.05, IoU 0.5): F1×5, R1×5,
C2×3, E1×3 (dev) + N1×3 night, S2×3 belt (holdout). Person boxes + per-seat
occupied/empty. Eval-only; never training labels.

## Splits and holdout policy
Dev tunes thresholds (frozen: eval_threshold 0.3). Holdout runs ONCE per
model/config change; failures gate release and are NEVER tuned against.
Methodology fixes (not threshold changes) may re-run with documentation —
holdout v1 caught the night-EMPTY gap this way.

## Metrics
- Detection: greedy-IoU P/R/F1 per class (person today).
- Seats (3-way): correct / wrong / abstain; UNKNOWN/STALE never wrong.
  unknown_rate reported alongside accuracy, never hidden.
- Latency P50/P95, frames, per-split gates with frozen acceptance numbers.
- Dense-run stability (transitions, time-to-stable) via chain tests.

## Measured baselines (M2 Pro, rtdetrv2-s, thr 0.3)
- Dev: P .195 / R .407 / F1 .264; seat .816 correct, 0 wrong, .184 unknown.
- Holdout: P .429 / R .50; seat .667 correct + 2 night wrong (documented gap).
- Box-level numbers are harsh (0.05-precision hand labels, small R1 boxes);
  seat-level 3-way is the product metric. Duplicate-FP flood quantifies R14.

## Reports and regression
`odc validate --split dev|holdout --out reports/<name>` → report.json +
index.html + failure gallery + optional --record-to-db (fills console).
`--compare` diffs reports; any gate flipping PASS→FAIL blocks release.

## What v1 does NOT cover (scheduled)
SVIRO box-level AP · belt/blanket/misuse positives (R15) · NIR/fisheye
buckets (R16) · multi-vehicle splits · event false-alarms/hour at scale.
