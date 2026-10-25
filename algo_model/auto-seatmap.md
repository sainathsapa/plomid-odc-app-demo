# Automatic seat-map proposal

Implementation: `odc/autolayout.py` and the `/api/service/start` flow in
`odc/console/app.py`.

When an operator starts an unknown replay without selecting a layout, the
console does not guess from array order or reuse a previous video's geometry.
It runs a bounded probe (up to 32 evenly sampled frames) with the same
manifest-selected detector used by the service. Person detections above the
probe floor are accumulated into a 12×12 normalized heatmap. Nearby boxes also
contribute their median width/height. Visible chair/couch and sufficiently
large bag-like furniture evidence contributes at reduced weight so an empty
seat can still produce a candidate region.

Greedy peak picking keeps peaks above 30% of the strongest hotspot and enforces
Chebyshev separation. Person evidence owns the proposal when people are visible;
chair/couch evidence is only a fallback for an empty cabin. This prevents a
chair peak from becoming a fourth seat in a three-person cabin. Each peak
becomes a clamped, globally non-overlapping x-partition sized from nearby
median person boxes, with generic stable IDs `seat_1`, `seat_2`, … and no
child-seat hints.

Rows are inferred from vertical hotspot gaps. A row with one hotspot remains a
one-seat row; this supports buses, vans, limousines, and single-column cabin
views. If perspective makes multiple hotspots appear in one image band, the
proposal applies a documented capacity constraint of no more than two front
seats, then fills rear rows up to 3/2 seats. This candidate heuristic prevents
an invalid three-seat front row but must be reviewed or replaced by camera
calibration before engineering use.

The proposal is explicitly candidate geometry (`auto-v1`), not camera
calibration. It persists one-based row/column metadata and is saved to PLOMID as a video-fingerprint-bound layout and
reused only when the source fingerprint is unchanged. A missing/unreadable
video, unavailable model, or no sustained evidence produces an uncalibrated
session; detections cannot be force-assigned to a seat. Operators can replace
the proposal with a traced layout in the Seat map editor.

When the operator selects **Auto-propose from video**, the selection order is:
explicit auto request → fresh automatic probe → uncalibrated. Camera-curated,
bundled-clip, and cached video layouts are intentionally skipped so a bad auto
proposal can be rechecked. Legacy API calls without an explicit layout retain
the compatibility order: camera curated → bundled clip → saved video map →
automatic probe → uncalibrated.

Steering orientation now has a guarded replay proposal. The MVP samples lower-
cabin frames for repeated steering-wheel circle evidence, maps image-left to
LHD and image-right to RHD for an unmirrored forward-facing camera, and only
proposes a side after repeated dominant votes. The result includes confidence,
frame count, and a mirror warning. Ambiguous, mirrored, rear-facing, or
occluded views return `UNKNOWN`; the operator can override and confirm the
result. This is candidate-grade OpenCV evidence, not a production steering
classifier.

