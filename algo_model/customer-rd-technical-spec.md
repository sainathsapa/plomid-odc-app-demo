# PLOMID Cabin ODC — Customer and R&D Technical Specification

**Document type:** engineering research and MVP implementation note  
**Audience:** AISIN/customer engineering, R&D, validation, systems, and UX teams  
**Scope:** cabin-camera occupant perception, automatic candidate seat mapping, and operator console  
**Safety position:** prototype engineering demonstrator; not safety-qualified and not an airbag controller

## 1. Executive summary

PLOMID Cabin converts a supported cabin-camera stream or prerecorded video into
structured per-seat evidence:

```text
source → decode/quality → preprocessing → detector → post-processing
      → tracking → seat geometry association → temporal state reasoning
      → PLOMID SQL → console/replay evidence
```

Automatic seat mapping is a bounded candidate-calibration aid, not a learned
vehicle-layout classifier. It samples the video, detects sustained
person/furniture hotspots, separates peaks, estimates seat-sized regions,
groups them into front-to-rear rows, and stores the resulting layout against
the source video fingerprint. An operator must review/trace final zones for an
installed camera.

## 2. Automatic seat-map algorithm

### 2.1 Input and preconditions

- Input must be a readable prerecorded video for the current implementation.
- The file is fingerprinted by path-independent content/size/time metadata.
- The same RT-DETRv2-S ONNX manifest used by the service is loaded.
- Up to 32 evenly spaced frames are probed; probe failure never creates seats.
- A live camera with no replay file remains uncalibrated until live calibration
  is implemented.

### 2.2 Detection evidence

For each sampled frame, OpenCV decodes the frame and the ONNX detector runs at
the probe threshold. Person detections above the probe floor are filtered by
class-aware NMS and their centers are accumulated into a normalized 12×12
heatmap weighted by score. `chair`/`couch` evidence has reduced weight; large
bag-like objects contribute only when their area is seat-like. Furniture helps
visible empty-seat proposals but cannot prove occupancy. Nearby bbox
width/height values are retained for region sizing.

### 2.3 Peak selection and region construction

Candidate peaks must reach 30% of the strongest heatmap value and be separated
by at least two heatmap cells using Chebyshev distance. Greedy selection is
deterministic and capped at six seats. No peaks produces `UNCALIBRATED`; the
system does not invent a five-seat map.

Each peak becomes a normalized, clamped rectangle centered on its heatmap cell.
Width and height use the median nearby contributor bbox with minimum region
dimensions. A region is a geometric candidate, not a semantic segmentation
mask and not proof that a physical seat exists there.

### 2.4 Rows and columns

Peaks are grouped by vertical heatmap position. A vertical gap of at least two
cells starts a new row, capped at three rows. Peaks inside each row are sorted
by camera x coordinate. The generated records persist one-based `row`,
one-based `col`, stable `seat_1`… IDs, `display_number`, `layout_type:
auto-proposal`, and `version: auto-v1`.

This removes the former failure mode where rows were inferred only at render
time and all hotspots could appear in one row. A single hotspot is not padded
with a second seat: one-seat rows are valid for buses, vans, limousines, and
single-column camera views. The 2/3/2 capacity heuristic is used only when
multiple hotspots are collapsed into one image-depth band.

### 2.5 Auto-mode precedence

The operator can select **Auto-propose from video** explicitly. In this mode
the backend skips camera-specific demo layouts and cached video maps, probes
again, and writes the candidate map to PLOMID. This makes it possible to
recheck a bad previous proposal. Normal API calls without an explicit layout
remain backward-compatible and may reuse curated/saved geometry.

## 3. Steering-side and vehicle orientation

### 3.1 Current detector limitation

RT-DETRv2-S COCO has no validated `steering_wheel`, `driver_position`, or
`RHD/LHD` class. A generic person location is not enough to determine steering
side in rear-facing, mirrored, fisheye, or cropped views.

### 3.2 Current implementation

The console exposes `Unknown`, `RHD`, and `LHD` steering-side selection. Auto
replay probing now adds a candidate OpenCV steering-wheel cue: repeated lower-
cabin circle evidence on image-left proposes LHD and image-right proposes RHD,
subject to a temporal vote threshold. The response exposes proposal confidence,
frames checked, and a mirror warning. The operator can confirm or override;
the confirmed value is stored as layout metadata. Standard profiles provide
explicit RHD metadata. Physical roles are separate from drawing order.

The live workspace displays the active value as `Steering: UNKNOWN`, `Steering:
RHD`, or `Steering: LHD`, and the selected value is written with the
video-bound layout. It is a configuration fact, not an automatically inferred
model result.

### 3.3 Production research path

For production automatic orientation, replace or corroborate the cue with a
labelled steering-wheel/driver-side model or OEM calibration signal. Candidate
cues include steering-wheel localization, instrument-cluster geometry,
driver-side seat geometry, and camera-mount metadata. Require confidence,
mirror-state handling, an unknown/reject threshold, and human override. Do not
derive a safety or airbag signal from the current heuristic.

## 4. Why front/rear seats can be missed or merged

- occupants are occluded or absent during most probe frames;
- the camera is a close-up rather than a full-cabin view;
- front and rear seats project to similar vertical positions;
- detector boxes merge occupants or confidence is low in darkness;
- a cached map was reused instead of explicitly reprobed;
- the source is a live camera, for which automatic replay probing is not yet
  available;
- the actual camera is mirrored, cropped, or fisheye;
- generic COCO does not recognize the seat furniture.

The start response and diagnostics expose probe frames, persons, peaks, layout
source, orientation, and reason text. Candidate maps must be reviewed in the
Seat map editor; overlap warnings and unassigned evidence remain visible.

## 5. State and evidence semantics

The detector emits observations, not truth. Tracking creates session-local
track IDs. Seat association uses normalized polygons and leaves ambiguous or
out-of-zone observations unassigned. Temporal state logic applies dwell,
hysteresis, stale expiry, visibility, and camera/model health.

Generic person detection is not an adult/child classifier. A visible strap is
not proof of belt fastening. Missing detection is not proof of an empty seat.

## 6. PLOMID boundary and recap

PLOMID is the only durable database through the documented SQL repository. The
service persists sessions, layouts, source fingerprints, model metadata,
current seat states, occupant attributes, events, health, analytics, probe
cache, and processed-frame inference metrics.

Raw continuous video and every JPEG are not stored in PLOMID. The live console
keeps a bounded transient frame ring, while replay uses the original source
video. This is intentional privacy/storage behavior. A future durable image
evidence feature requires explicit retention, access-control, encryption, and
migration decisions.

## 7. Customer/R&D acceptance procedure

1. Start with a clean PLOMID schema and the supported environment.
2. Select **Auto-propose from video**, not a demo layout.
3. Confirm the response reports `auto-N`, probe frames, persons, peaks, and
   `UNKNOWN` orientation unless RHD/LHD was selected.
4. Verify the cabin schematic has separate rows and one-based seat labels.
5. Compare every proposed zone with the source; record missed rear seats,
   merged seats, overlaps, and out-of-zone people.
6. Trace/correct zones on Seat map and save a versioned layout.
7. Restart the same source and verify the intended saved layout is selected.
8. Compare SQL current states/events/metrics with the source frame clock.
9. Repeat on RHD/LHD, mirrored/fisheye, low-light, empty, child-seat, and
   occlusion scenarios.
10. Record seat precision/recall, missed-seat rate, row accuracy, orientation
    accuracy/unknown rate, latency, and recovery behavior.

## 8. Enterprise UX recommendations

- Persistent banner: `AUTO CANDIDATE — REVIEW CALIBRATION`.
- Show layout source, probe frames/persons/peaks, orientation, and PLOMID save
  status beside the map.
- Separate actions: `Auto-propose`, `Recompute auto`, `Use saved layout`, and
  `Trace/calibrate`.
- Require confirmation before applying an auto map to a running session.
- Show per-row seat count and highlight missing expected rows.
- Add an orientation confirmation dialog with mirror/driver-side guidance.
- Label `transient frame recap` versus `durable SQL evidence`.
- Add a validation checklist before marking a layout calibrated.

## 9. Regression evidence: 10385816 clip

The supplied `10385816-uhd_4096_2160_25fps-1791966142.mp4` was inspected with
the active RT-DETRv2 runtime. The sampled frame contains three person
detections: two large front occupants and one smaller rear occupant. The
corrected auto proposal creates non-overlapping candidate regions and maps the
three detections to `seat_1`, `seat_2`, and `seat_3`, with a two-seat first row
and a one-seat second row for the evidence visible in that clip. Full replay
validation must still confirm dwell-stabilized `OCCUPIED` states and a human
must review candidate geometry against the installed camera mount.

