# Child detection audit and validation

**Status: INCOMPLETE.** The current pipeline can produce child detections, but it misses or abstains on a clearly visible child in a separately sourced public clip. Available video volume is too small to establish acceptable child recall and adult false-positive rates. Do not treat this model as production ready.

## 2026-10-24 final recall and independent video pass

### F1 miss investigation and correction

All six prior F1 Unknown child-seat observations were inspected at source frames 49, 109, 121, 145, 313, and 325. Frames 49, 109, 121, 145, and 325 had **no person detection assigned to the child seat**. The baby was heavily covered by a blanket; the detector either returned only adult tracks or generic suitcase detections. No age model or seat-association rule can infer a child from those absent person detections without inventing evidence, so these remain Unknown. Frame 313 had a person detection spanning adult and child zones. Its box overlap was 57.5% with the child seat and 41.6% with the adjacent seat; the old guard rejected any child box overlapping two zones. The frame-325 adult false-positive candidate had 40.3% overlap with its assigned adult seat and 40.1% with the adjacent child zone.

The production change keeps the original full person box for seat association and audit. When the assigned seat has at least a 0.10 overlap-fraction advantage over every other configured seat, MiVOLO receives a crop clipped to that seat's zone. When the overlap is ambiguous, the full crop is retained and child confirmation remains suppressed. A face crop is used only if its center falls inside the assigned seat. Per-track output now records both the original `person_bbox` and actual `inference_bbox`. This does not use box dimensions as age evidence; crop localization reduces neighboring-seat pixels. At F1 frame 313, the seat-local crop changed the raw one-frame MiVOLO age estimate from 3.44 to 6.61 years, while the temporal track estimate was 4.29; both map to the existing child band. An earlier unrestricted crop experiment changed the adult frame-301 estimate into the child band and failed the adult regression; it was rejected. Current focused regression: F1 has 23/28 child Yes, 5 Unknown, and 0/56 adult Yes. The remaining five misses are exactly the no-person-detection frames listed above.

MiVOLO emits an age regression point estimate, not calibrated age probabilities. The current operating boundary is infant <=2, child <=11, Unknown >11 through 23, adult >23. The development-only selection below now gives this six-year band a reproducible, empirical rationale; it does not establish calibrated probabilities or population error rates. The older Pexels 4524519 clip generated many Unknown values (sampled child-seat estimates through 18.97), so older children may still be missed or left Unknown.

### Age-boundary development analysis (2026-10-24)

I added a reproducible, development-only analysis at `scripts/analyze_age_boundary_calibration.py`; it reads the independently seat-labeled F1, C1, C2, 6181423 child-present, and 9519139 adult-only development reports, groups raw age point estimates by video/seat/person-track, and scores track medians (minimum two observations). It excludes ambiguous seat/face associations, seat mismatches, unlabeled seats, and one-observation tracks. It does not interpret age as a probability and does not read the locked final-evaluation reports.

Command: `PYTHONPATH=. ./.venv/bin/python scripts/analyze_age_boundary_calibration.py`.

Observed evidence: 5 development clips, 7 child-occupant tracks, and 11 adult tracks. Child-track median estimates ranged 2.070–6.840 years; adult-track medians ranged 24.355–35.715 years. Child examples span four clips, adults span three. Applying the current presence rule (child evidence through 11, adult evidence above 23) to these track medians yields 7/7 child tracks and 11/11 adult tracks, with zero Unknown track medians and zero cross-class errors. In a 25-setting sensitivity grid, reducing the child cutoff to 6 leaves one child track Unknown, while raising the adult cutoff to 26 leaves one or two adult tracks Unknown; no tested setting creates a cross-class error. The development data support the current operating point over these observed track medians, but do not identify a unique optimum or estimate unseen error rates.

The analysis selects the largest whole-year symmetric uncertainty margin around the configured age-17 boundary that keeps every included development track on the correct side and leaves at least one year between the decision cutoffs and the nearest observed class extrema. The available margin is 6.355 years; rounded down, it selects 6 years, yielding child <=11 and adult >23, the current production defaults. Thus the boundary is now provisionally selected and reproducible on the available development tracks; it is **not population-calibrated**. The 1-year clearance is an explicit engineering guard, not a statistical confidence interval. The data remain small and correlated, and the region between observed child/adult estimates has no labeled tracks. C1 and 6181423 stored reports predate the current abstention-band configuration; analysis reapplies the current cutoffs to their raw age estimates and does not treat their serialized categories as current-policy results. Full machine-readable track values, exclusions, and candidate scores are in `reports/age-boundary-development-analysis.json`.

Verification: `PYTHONPATH=. ./.venv/bin/pytest -q -o addopts='' tests/unit/test_age_boundary_calibration.py tests/unit/test_age.py` → **23 passed**; `PYTHONPATH=. ./.venv/bin/pytest -q -o addopts='' tests/integration/test_child_detection_c1.py` → **1 passed** using the real Service/MiVOLO path; Ruff lint and format checks passed for the added analysis and tests. The C1 integration replay confirmed a child/infant result survives the production path with the selected default band. The full repository suite was not rerun for this calibration-only change.

### Independent public-video evaluation

The following clips were downloaded from Pexels Free License pages on 2026-10-24 and processed using the actual production `Service` via `scripts/evaluate_unseen_child_video.py`. The two new child clips were visually reviewed at timestamps across each clip before assigning the listed seat-level child ground truth. One reviewer supplied the labels; no second reviewer was available. Videos were not used for threshold selection. Frames are correlated within videos; frame counts are not independent subjects. The adult clip is an exterior side view into a car, not an interior camera view, and is a domain-mismatch hard negative.

| Video and source | Hash (SHA-256) | Format / samples | Child TP / Unknown / missed | Adult FP / adult samples |
|---|---|---|---:|---:|
| Kampus Production, [A Child Smiling at the Camera while Sitting in the Car](https://www.pexels.com/video/a-child-smiling-at-the-camera-while-sitting-in-the-car-6300815/) | `beed1bd799a9c35f65f41a45c3918c8aa5ac57671831b75d5556ac6a0588ec9f` | 3840x2160, 25 fps, 207 frames / 8.28 s; 18 child-seat rows | 18 / 0 / 0 | n/a |
| Martina Tomšič, [Children Riding a Car](https://www.pexels.com/video/children-riding-a-car-5016642/) | `cf87cdba7033e2e2d83b6ebe217cdbc8c63835bf4e7d55f771feb0ad56dafeed` | 2560x1440, 29.97 fps, 298 frames / 9.94 s; two child seats, 50 rows | 50 / 0 / 0 | n/a |
| Ketut Subiyanto, [Kids Wearing Their Seatbelts](https://www.pexels.com/video/kids-wearing-their-seatbelts-4524519/) | `79b909b1267e2097069072d29f4e3312b84682bc50072c8d42b634acf0c9a611` | 1920x1080, 25 fps, 266 frames / 10.64 s; 45 child-seat rows | 29 / 16 / 0 | n/a |
| WeStarMoney Rec, [Man Sitting Inside a Car in Slow Motion](https://www.pexels.com/video/man-sitting-inside-a-car-in-slow-motion-4281307/) | `d512aa12bf8da4597aae5edce2559ff97f856fd90320191d4823ce12467df991` | 1920x1080, 24 fps, 232 frames / 9.67 s; 20 adult rows | n/a | 0 / 20 |

Across three child-present videos: 97 Yes, 16 Unknown, 0 No among 113 child-seat observations; recall counting Unknown as a miss is 85.8%, and Unknown rate is 14.2%. On the one adult-only video, there were 0 child Yes in 20 adult-seat observations. F1 development footage adds 23/28 child Yes (82.1%) and 0/56 adult Yes. These aggregates describe four short clips and only three child subjects/scenes; no confidence interval over independent subjects is supportable. They do not establish a sufficiently low population false-positive rate or robust age classification. The 4524519 clip has 16 Unknown; the most difficult F1 blanket-occluded samples have no person boxes. These are known misses, not hidden as correct predictions.

Frame-level raw outputs, GT labels, candidate decisions, per-video metrics, provider, and speed are retained in [`reports/independent-pexels-6300815.json`](../reports/independent-pexels-6300815.json), [`reports/independent-pexels-5016642.json`](../reports/independent-pexels-5016642.json), [`reports/independent-pexels-4524519-current.json`](../reports/independent-pexels-4524519-current.json), [`reports/independent-pexels-4281307.json`](../reports/independent-pexels-4281307.json), and [`reports/f1-family-ride-current.json`](../reports/f1-family-ride-current.json). Child Unknown review sheets: [`reports/f1-recall-unknown-2026-10-24.jpg`](../reports/f1-recall-unknown-2026-10-24.jpg) and [`reports/pexels-4524519-child-unknown-2026-10-24.jpg`](../reports/pexels-4524519-child-unknown-2026-10-24.jpg). The adult-only video produced no false-positive sheet because there were no adult false positives.

Reproduce after downloading the Pexels files into `footage/uploads/` (the downloaded media is ignored by Git):

```sh
curl -fL -o footage/uploads/pexels-6300815-child-car.mp4 \
  https://videos.pexels.com/video-files/6300815/6300815-uhd_3840_2160_25fps.mp4
curl -fL -o footage/uploads/pexels-5016642-children-car.mp4 \
  https://videos.pexels.com/video-files/5016642/5016642-uhd_2560_1440_30fps.mp4
curl -fL -o footage/uploads/pexels-4281307-adult-car.mp4 \
  https://videos.pexels.com/video-files/4281307/4281307-hd_1920_1080_24fps.mp4
sha256sum footage/uploads/pexels-6300815-child-car.mp4 \
  footage/uploads/pexels-5016642-children-car.mp4 \
  footage/uploads/pexels-4281307-adult-car.mp4
# Run each clip through the production Service path. Seat layouts are versioned in configs/seats/.
PYTHONPATH=. MPLCONFIGDIR=/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/uploads/pexels-6300815-child-car.mp4 \
  --layout configs/seats/eval-pexels-6300815-v0.yaml --child-seats child-seat \
  --occupied-seats child-seat --sample-every-n 12 --age-every-n 1 \
  --scenario independent-child-visible-Pexels-6300815 \
  --out reports/independent-pexels-6300815.json
PYTHONPATH=. MPLCONFIGDIR=/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/uploads/pexels-5016642-children-car.mp4 \
  --layout configs/seats/eval-pexels-5016642-v0.yaml \
  --child-seats child-left child-right --occupied-seats child-left child-right \
  --sample-every-n 12 --age-every-n 1 \
  --scenario independent-children-visible-Pexels-5016642 \
  --out reports/independent-pexels-5016642.json
PYTHONPATH=. MPLCONFIGDIR=/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/uploads/pexels-4281307-adult-car.mp4 \
  --layout configs/seats/eval-pexels-4281307-v0.yaml --adult-seats driver-seat \
  --occupied-seats driver-seat --sample-every-n 12 --age-every-n 1 \
  --scenario independent-adult-only-Pexels-4281307-exterior-view \
  --out reports/independent-pexels-4281307.json
PYTHONPATH=. MPLCONFIGDIR=/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/uploads/unseen_pexels_4524519.mp4 \
  --layout configs/seats/eval-pexels-4524519-v0.yaml --child-seats child-center \
  --sample-every-n 6 --age-every-n 1 \
  --scenario independent-child-present-Pexels-4524519-current-pipeline \
  --out reports/independent-pexels-4524519-current.json
```

`ruff format` was applied to the touched Python files after it reported existing differences; formatter check now passes for all changed Python files. Full suite and focused results are listed in the final verification section below.

> **Recall change (2026-10-24):** MiVOLO age inference now receives person tracks from their first detector hit when enabled. A one-observation child-age estimate is held Unknown when another current track associated with that same seat estimates adult; repeated child-track evidence resolves the conflict. This was introduced after the one-hit experiment recovered child frames but caused an adult false positive. Production Service replay on the development F1 clip improved from 18/28 to 22/28 child Yes, with adult child-positive rows remaining 0/56; a separate adult-only upload replay yielded 0/58 child Yes. Six F1 child rows remain Unknown (five without a current track and one spanning multiple seats). This is a clip-level regression improvement only—not independent validation, calibrated age confidence, or production-readiness evidence. Raw frame reports: `reports/f1-family-ride-current.json`, `reports/adult-only-9519139-recall-check.json`; comparison: `reports/child-recall-change-2026-10-24.json`.

Reproduce the focused checks:

```sh
PYTHONPATH=. ./.venv/bin/pytest -q tests/unit/test_age.py
PYTHONPATH=. ./.venv/bin/pytest -q tests/integration/test_child_detection_f1_mixed.py
PYTHONPATH=. MPLCONFIGDIR=/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py \
  footage/uploads/9519139-uhd_4096_2160_25fps-1791966051.mp4 \
  --layout configs/seats/eval-upload-9519139-v0.yaml \
  --adult-seats passenger-left driver-right \
  --occupied-seats passenger-left driver-right \
  --sample-every-n 24 --age-every-n 1 \
  --scenario recall-change-adult-only-regression-check \
  --out /tmp/recall-adult-9519139.json
```

> **Ground-truth correction (2026-10-23):** Later visual review found the R1 rear-wide occupants are not reliably age-scorable from the mostly rear-facing view. All prior R1 adult/child confusion counts—including the reported three adult-to-child false positives and subsequent age-band comparison—are retained only as historical model-output diagnostics and are excluded from valid age-classification metrics. The rear-left child label previously copied from a spot-check note is not supported. R1 is now scored for seat occupancy only. This weakens the evidence used to select the six-year MiVOLO uncertainty band; treat it as a development heuristic, not a validated calibration.

> **Current worktree update (2026-10-23):** MiVOLO V2 is integrated into the production `Service` path. A previous R1 replay was called adult-only and showed three child outputs from person-only age estimates; later frame review found R1 ages unscorable, so those are no longer counted as confirmed false positives. The uncertainty-band mitigation remains a heuristic and is not independently calibrated. The system remains **INCOMPLETE** and is not ready to claim precise age classification.

### Age-boundary uncertainty policy (development correction)

The previous implementation mapped any MiVOLO regression estimate through age 17 directly to child. Development R1 produced false child labels at 11.16 and 16.14 years; that is insufficient evidence for a positive child classification. The configured boundary remains 17, but `age.uncertainty_margin_years` defaults to 6: ages through 11 are classified child, ages above 11 through 23 are Unknown, and estimates above 23 are classified adult. Infant estimates through 2 remain infant. The margin was chosen on development clips to cover the observed adult false-positive estimates; it has not been evaluated on an independent population. This deliberately leaves adolescents and some near-boundary adults unresolved, and age estimates above 23 are still not proof of adulthood. The policy is a precision-oriented mitigation, not a validated age classifier or a completion claim.

## Phase update: all-seat occupancy and classification evidence

The actual production `Service` path was replayed on local R1, E1, and 6181423 clips. Historical R1 output showed three age estimates categorized as child, but subsequent review found that the mostly rear-facing occupant ages are not independently scorable. The old 3-FP count and age estimate comparisons are not valid adult/child error metrics and must not be used to assert either false-positive reduction or calibration. Raw outputs remain useful for diagnosing the model's person-only estimates, which are not calibrated probabilities. MiVOLO alone therefore cannot safely resolve the ambiguity in the available rear view.

The E1 empty-cabin replay produced 41 confirmed empty seat rows and 24 Unknown rows, with zero occupied false positives. The Unknown frames show a cabin substantially obscured by vehicle structure/window and poor visibility; the contact sheet is `reports/phase1-E1-occupancy-unknown.jpg`. The replay's one-seat layout does not establish all-seat recall. The N1 adult night-drive replay produced no child Yes, but 133/200 driver-seat samples were Unknown under low visibility. The face landmark model is now skipped when the production service has no tracked occupants, avoiding unnecessary face inference in empty-cabin frames.

The 6181423 clip was rerun through the current Service with MiVOLO (39 snapshots; 34 child Yes, 5 child Unknown, 33 adult No, 6 adult Unknown; no false positives on the annotated rows). It is prior project debugging footage and is not held-out evidence. R1's person-only estimates near the child/adult boundary demonstrate unstable apparent-age output, but its occupant ages are not used as ground truth after review. The uncertainty margin was not selected using the 6181423 clip.

The age uncertainty policy was historically compared on R1 and C1. R1 age metrics are now excluded due to the ground-truth review above. C1 stayed at 26 child Yes, 2 No, and 5 Unknown across 33 child-labelled samples. The six-year margin was selected using limited development outputs and is not an independently calibrated threshold. Older-child behavior (particularly estimates 12–23) often abstains under this policy.

`scripts/make_evaluation_contact_sheet.py` now supports occupancy Unknown and occupancy false-positive sheets, and displays raw candidate evidence where age outputs are absent. The R1 false-positive sheet was generated and visually reviewed. Adult false positives and child-positive samples must be counted by video/subject, not by adjacent frames, in any final generalization claim.

Reproduce the R1 production replay and adult false-positive contact sheet with:

```sh
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/R1_rear_wide.mp4 \
  --layout configs/seats/demo-r1-v0.yaml \
  --adult-seats driver-front rear-left rear-center rear-right \
  --scenario development-multi-passenger-seat-occupancy \
  --out reports/phase1-R1-occupancy.json
PYTHONPATH=. ./.venv/bin/python scripts/make_evaluation_contact_sheet.py \
  reports/phase1-R1-occupancy.json reports/phase1-R1-adult-FP.jpg \
  --kind adult-false-positive --max-frames 8
```

Reproduce the revised age-policy runs and the affected regressions with:

```sh
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/R1_rear_wide.mp4 \
  --layout configs/seats/demo-r1-v0.yaml \
  --adult-seats driver-front rear-left rear-center rear-right \
  --scenario development-multi-passenger-seat-occupancy-age-uncertainty-band-v1 \
  --out reports/phase3-R1-age-band.json
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/C1_baby_carseat.mp4 \
  --layout configs/seats/demo-c1-v0.yaml \
  --child-seats infant-carrier --empty-seats seat-right \
  --scenario development-infant-with-adult-handler-age-uncertainty-band-v1 \
  --out reports/phase3-C1-age-band.json
env -u NO_COLOR PYTHONPATH=. ./.venv/bin/pytest -q tests/unit tests/integration/test_child_detection_c2.py
```

## MiVOLO V2 production integration

The local checkpoint at `models/mivolo-v2/model.safetensors` was loaded by the real `Service` path and executed on C1 video frames. The loader reported 290/290 checkpoint tensors loaded. MiVOLO performs one shared forward pass for its age estimate and apparent-gender softmax; the separate experimental CLIP gender model is bypassed whenever MiVOLO is active. Existing `gender` and `gender_conf` row fields are retained. Gender confidence is explicitly labelled an uncalibrated softmax score. MiVOLO emits a regression age point estimate, not age-class probabilities, so `age_class_confidence` is null.

Person crops come from the existing RT-DETR and ByteTrack-style occupant track. Face crops are taken from the existing local MediaPipe FaceLandmarker when one matches the person box; MiVOLO receives its supported person-only input if no face is found. Output is associated to the current track ID and seat ID. A child-labelled track yields seat-level child presence Yes even if an adult track shares the coarse seat polygon; that seat category is reported as `mixed`. An occupied seat with absent/failed age output remains Unknown. A confirmed empty seat remains No. The restraint field is independent.

For the first real C1 replay, before the mixed-track aggregation correction, MiVOLO produced actual child-age outputs (for example track `t2`, person-only input, estimated age 5.03 years) alongside the adult handler (`t3`, face+person input, estimated age 33.72 years). The broad carrier polygon associated both tracks to one seat, so the initial seat output was Unknown on 25 mixed predictions. The correction now preserves the positive child-track evidence as child presence Yes and reports the seat category as mixed. This aggregation correction is covered by deterministic logic tests; the updated real-video regression is being rerun. Other C1 frames showed MiVOLO estimates around 25 years for the child track, or lacked an active track. Those remain measured model/visibility misses, not resolved by the aggregation change.

The exact checkpoint reports `model_name=mivolo_d1_384`, `max_age=122`, and a 384-pixel input. The configured adulthood boundary is age 17, with the current six-year uncertainty margin described above: ≤2 infant, >2 through 11 child, >11 through 23 unknown, and >23 adult. These are application rules over a regression point estimate, not model-native classes or calibrated probabilities. MiVOLO was developed for apparent age and gender estimation on general person/face imagery, not validated vehicle-cabin child-restraint inputs. A point estimate that repeatedly jumps from ~5 to ~25 years on the same no-face body track demonstrates a material limitation.

Checkpoint/model references and licensing: [MiVOLO V2 model card](https://huggingface.co/iitolstykh/mivolo_v2) and [MiVOLO paper](https://arxiv.org/abs/2307.04616). The model card identifies Apache-2.0. The upstream implementation is [WildChlamydia/MiVOLO](https://github.com/WildChlamydia/MiVOLO), also Apache-2.0; a pinned subset is vendored under `odc/_vendor/mivolo/` and `odc/_vendor/mivolo_hf/` with its license. Checkpoint SHA-256 is recorded in each refreshed evaluation report. Model load compatibility was adapted for the installed Transformers 5.19 and timm 1.0 APIs; no weight files are vendored into Git.

Run an actual production-path clip replay with:

```sh
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache \
  ./.venv/bin/python scripts/evaluate_unseen_child_video.py \
  footage/C1_baby_carseat.mp4 \
  --layout configs/seats/demo-c1-v0.yaml \
  --child-seats infant-carrier --adult-seats seat-right \
  --scenario development-infant-with-adult-handler \
  --out reports/mivolo-C1-dev.json
```

The frame-level `age_predictions` preserve track, seat, age point estimate, category, face visibility, model ID, inference time, and reason. Review misses with `scripts/make_evaluation_contact_sheet.py`. Do not interpret the historical Grounding-DINO-only integration assertions as MiVOLO tests; the C1/C2 integration tests now require the local MiVOLO checkpoint and real Service execution.

### Current measured MiVOLO replays (development clips)

These were run through the actual `Service` and local RT-DETR, face landmark, and MiVOLO weights. Labels are independent visual scene/seat annotations, though repeated adjacent frames from one video are correlated and are not independent subjects. Both C1 and C2 are development footage already used in product work; they are not locked unseen evaluation videos.

| Video / condition | Child-labelled seat frames | TP | No/FN | Unknown | Precision | Recall (Unknown=miss) | Adult child-FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 infant restrained, adult handling | 33 | 26 | 2 | 5 | 100% | 78.8% | not separately annotated in final tight-ROI run |
| C2 child in vehicle seat | 35 | 33 | 0 | 2 | 100% | 94.3% | not separately annotated |
| Uploaded 9519139, two adults | 230 | 0 | 0 | 4 | n/a | n/a | 0/230 positive; 226 adult No, 4 Unknown |

The C1 scene has a broad seat-layout limitation: the child and handling parent overlap spatially, and the child track is absent in the first four sampled frames. On two later sampled frames, the child track's person-only estimate is 25.49 years and is mapped to adult. The original replay had a larger area where the parent was associated to the carrier; seat geometry was tightened, so the current replay counts 26 child Yes, two adult-estimate No and five Unknown. The adult review sheet is `reports/mivolo-C1-child-misses.jpg`; C2 has two initial frames with no person track (`reports/mivolo-C2-child-misses.jpg`).

On C2, 33 of 35 child-seat samples are Yes (30 child, 3 mixed child+adult tracks); two earliest samples remain Unknown because no tracked person was available. No is not converted to Yes based on restraint or seat name. C1 and C2 results show useful behavior on these particular scenes, but do not validate infants/children across subjects, camera views, lighting, or occlusion.

On the adult-only 9519139 local upload, the final MiVOLO code produced zero child positives among 230 adult-labelled seat rows and 4 Unknown. This directly improves on the historical Grounding-DINO adult hard-negative run (112/230 false positives), but it is a development comparison, not independent validation. The 9519139 uploaded file's source license metadata is not verified, so it is not for showcase publication. Its report is `reports/mivolo-adult-9519139.json`; the representative adult review sheet is `reports/mivolo-9519139-adult-review.jpg`.

CPU runtime on this Apple arm64 host, with CPUExecutionProvider for RT-DETR and CPU MiVOLO: C1 33 evidence snapshots in 37.0 s (~0.89 snapshots/s), median MiVOLO batch 198 ms; C2 35 snapshots in 48.1 s (~0.73 snapshots/s), median MiVOLO batch 406 ms; adult upload 115 snapshots in 334.3 s (~0.34 snapshots/s), median Service detector latency 251 ms. Full 4K video decode and CPU inference are not real-time. Face landmarks and enabled default pose add work. There is an `auto` device selector for CPU/CUDA/MPS; CUDA and MPS execution have not been validated on this host.

Run the final development reproductions:

```sh
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python scripts/evaluate_unseen_child_video.py \
  footage/C1_baby_carseat.mp4 --layout configs/seats/demo-c1-v0.yaml \
  --child-seats infant-carrier --empty-seats seat-right \
  --scenario development-infant-with-adult-handler-mivolo \
  --out reports/mivolo-C1-dev-fixed.json
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python scripts/evaluate_unseen_child_video.py \
  footage/C2_child_seat.mp4 --layout configs/seats/demo-c2-v0.yaml \
  --child-seats child-seat --scenario development-child-in-seat-mivolo \
  --out reports/mivolo-C2-dev.json
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python scripts/evaluate_unseen_child_video.py \
  footage/uploads/9519139-uhd_4096_2160_25fps-1791966051.mp4 \
  --layout configs/seats/eval-upload-9519139-v0.yaml \
  --adult-seats passenger-left driver-right --scenario adult-only-development-hard-negative \
  --out reports/mivolo-adult-9519139.json
```

### Showcase capability status

- **Seat occupancy across configured seats:** implemented using the existing RT-DETR person/object detection, track-to-seat association, and temporal seat state. Results remain view/layout dependent; unknown visibility stays unresolved. Seat geometry must be calibrated for each camera.
- **Child/adult/infant and apparent gender:** MiVOLO V2 is loaded once in production and supplies age and gender in one inference. Development replays show child recall of 78.8% on C1 and 94.3% on C2, plus zero child positives on one local adult-only upload. This is not enough subject/video diversity for precise-classification claims or production readiness. Apparent gender remains an uncalibrated visual estimate and must not be treated as verified gender.
- **Posture:** MediaPipe Pose Landmarker Lite is enabled by default using the downloaded local asset. Keypoints are visibility-gated and included per seat; no uncalibrated body measurements are asserted. Real-video end-to-end posture accuracy has not been separately scored.
- **Seatbelt:** current pipeline uses geometric strap candidates with temporal and visibility gates. No dedicated pretrained seatbelt checkpoint is present in the local model assets; this is candidate-grade and not validated for safety decisions. Missing/occluded evidence remains Unknown.
- **Child restraint / infant carrier:** restraint evidence remains an independent generic object candidate; local RT-DETR COCO has no child-seat/infant-carrier class. The downloaded files contain no validated dedicated restraint model. Empty restraint does not set child presence. Reliable restraint type/presence classification remains incomplete.
- **Hardware:** CPU inference ran on this host. Device selection supports `auto`, CPU, CUDA, and MPS where available, but only CPU was exercised. 4K playback is slower than real time on CPU.
- **Versioned vehicle API / persistence:** current live seat rows expose age predictions, age model and reason, confidence semantics, gender and gender score, occupancy, restraint, belt, pose and track IDs while preserving the current row shape. MiVOLO class confidence is `null` because the model does not emit one. Runtime model failures produce Unknown and an explicit reason. Dedicated external vehicle integration and functional safety certification are out of scope and not claimed.

**Acceptance remains INCOMPLETE.** The showcase can demonstrate real pretrained age/gender and occupancy inference, but the current evidence does not establish robust child precision/recall across independent subjects, child restraint classification is not supported by a local dedicated model, seatbelt inference is geometric rather than a validated pretrained detector, and hardware acceleration is untested. Avoid presenting the UI as an automotive safety product.

## Delivery phases and verification from 2026-10-23

### Phase 1 — Occupancy and all-seat association: implemented, locally exercised

Every configured polygon is represented in the seat map. Person detections flow through the existing tracker and `SeatStateEngine`; each occupant retains a track ID and seat association. C1/C2/adult clip reports exercise this path over 183 processed evidence snapshots (366 seat rows; not independent samples). Seat geometry is camera-specific; the C1 configuration was tightened after the handler was associated to the child carrier. Empty-seat confidence remains unresolved on some views (C1's right seat was Unknown on all 33 sampled rows), so calibrate each showcase layout and do not describe an unobserved seat as empty.

### Phase 2 — Age and apparent gender: implemented; recall evidence promising on C2, insufficient overall

MiVOLO V2 and shared gender inference are live in Service and the versioned `/api/v1/live/state` endpoint. Child results are: C1 26/33 Yes, 2 No, 5 Unknown; C2 33/35 Yes, 2 Unknown. The adult-only hard negative yielded 0/230 child Yes and 4 Unknown. Those clips do not establish robust performance across people and vehicles. The under-18 category mapping uses a point estimate and the C1 two missed children show why model class confidence cannot be asserted.

### Phase 3 — Pose, restraint and belt: partially implemented

MediaPipe pose is enabled for the showcase by default and keypoints are published with visibility flags; the C2 replay produced pose entries in 15 of 35 rows. This is a smoke check, not a posture accuracy evaluation. Child restraint class/presence has no dedicated checkpoint in the local assets. Belt output uses geometry, visibility and temporal aggregation, not a pretrained belt detector. Treat both as candidates, not vehicle-system decisions.

### Phase 4 — Device independence and startup: CPU exercised; accelerated devices not verified

MiVOLO chooses CUDA, then MPS, then CPU in `auto` mode; explicit device can be configured. This host ran CPU. Docker now installs the pinned PyTorch/torchvision/timm/Transformers runtime required by the adapter and mounts local model weights read-only. `models/manifests/mivolo-v2.json` pins source revision, license declaration, input semantics and checkpoint hash. A clean Docker build and GPU/MPS parity run were not performed in this work turn.

Commands executed in this work turn:

```sh
PYTHONPATH=. ./.venv/bin/python -m odc.cli config-check
PYTHONPATH=. ./.venv/bin/python -m odc.cli --help
./.venv/bin/ruff check odc/age.py odc/config.py odc/service.py scripts/evaluate_unseen_child_video.py scripts/make_evaluation_contact_sheet.py tests/unit/test_age.py tests/unit/test_console.py tests/integration/test_child_detection_c1.py tests/integration/test_child_detection_c2.py
env -u NO_COLOR PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/pytest -q tests/unit/test_age.py tests/unit/test_console.py::test_live_api_reports_mf_model_readiness tests/unit/test_console.py::test_versioned_live_api_exposes_additive_schema_version tests/unit/test_state.py tests/unit/test_segmentation_attributes.py tests/unit/test_service.py::test_service_runs_degraded_without_model tests/unit/test_service.py::test_service_shutdown_saves_analytics_snapshot
```

The focused command above passed. The complete `tests/unit` run is recorded in the final handoff after it completes. One earlier focused run failed a console fixture because it assumed gender-model status was controlled only by the old CLIP flag; the fixture was updated to explicitly disable MiVOLO for that legacy test, and the focused run passed afterward. A direct `python -m odc` invocation failed because this repository exposes `odc.cli`, not `odc.__main__`; the supported `python -m odc.cli --help` command passed. A strict mypy invocation reports existing errors in `steering.py`, `vlm.py`, `specialist.py`, and `state.py`, plus typing gaps in the upstream vendored Transformers model code; no false claim of a clean type check is made.

The aggregate `reports/unseen-child-evaluation.json` now embeds the frame-level Service reports for the three measured videos. Individual reports and contact sheets remain under the repository's ignored `reports/` directory; regenerate them with the commands above and contact-sheet script. No video or checkpoint weights are staged for commit.

## What was wrong

The fast person detector has no child/adult class. Age output comes from Grounding-DINO open-vocabulary phrase matches (`baby`, `child`, `adult`, and restraint phrases) associated to person tracks and seat ROIs. Its scores are text-image matching scores, not calibrated probabilities of occupant age. An empty restraint does not establish child occupancy. The user-reported 0.38 score gate is not the active default in this worktree: `DetectionConfig` currently has child candidate minimum 0.30, direct-evidence threshold 0.35, and temporal confirmation threshold 0.351. These remain heuristic cutoffs on uncalibrated phrase similarities, not probability thresholds.

The reproduced adult-only night baseline contains 22 `child=Yes` rows. Their recorded reason codes carry weak Grounding-DINO `child` matches at 0.311–0.342 attached to generic person detections. Those similarity values do not establish age. A separate earlier branch also called large person boxes adult evidence; that could suppress genuine children, though it did not produce the recorded 22 positive rows. The correction removes size and seat location as child/adult evidence, requires a specialist candidate to map to one occupied person track, associates adult evidence to that same track, and preserves missing or same-track conflicting age evidence as `Unknown`. The API fallback reason is now `no-age-evidence`. Detection-box area is not read by the child/adult decision. The remaining box-area check in `odc/state.py` is only for filtering oversized restraint-object proposals.

This did not make Grounding-DINO an age classifier. The specialist threshold and accumulated evidence remain heuristic gates on an uncalibrated score.

## Reproduced child miss

On development clip C1, the first corrected replay found both `adult` and `baby` candidates in one broad seat ROI. The old logic associated each candidate with any overlapping person in that seat, then turned distinct occupants into a seat-level age conflict. Production person detections contain separate adult and infant tracks. Selecting the best-overlap person track associates the adult candidate to `t3` and the baby candidate to `t2`. In a paced replay at the configured 5 fps, the service reports child Yes on 26/33 child-labelled snapshots, No on 1, and Unknown on 6. This improves the earlier replay but does not establish reliable recall; it is covered by a real-clip integration test.

On Pexels 4524519, the production `Service` path detected a `baby` phrase at similarity `0.30596`. It associated the candidate to the occupied `child-center` seat and person track. The candidate appeared on only one specialist source frame (`frame_seq=241`), below the current weak-match temporal confirmation requirement of two source frames. The final child output was therefore `Unknown`. This is a temporal confirmation miss after a valid but weak raw candidate, not a missing person detection or a lost API field. The frame-level record is in [`reports/unseen-child-evaluation.json`](../reports/unseen-child-evaluation.json).

The additional adult-only uploaded clip 9519139 exposes a still-unfixed classifier limitation. The woman in the passenger seat receives `child` candidates scored about 0.39–0.46 on person track `t2` across repeated specialist frames, while the driver receives `adult` on track `t1`. The paced production-path replay reports child Yes in 112 of 230 adult-labelled seat snapshots (48.7% false-positive rate with Unknown in the denominator), No in 112, and Unknown in 6. The passenger seat has no child-restraint evidence. The generated contact sheet shows the adult woman repeatedly labeled as child. Track association is correct; Grounding-DINO's phrase match still confuses an adult with a child. This local upload has no verified source URL/license metadata and is a development hard negative, not independent public evidence. The report is [`reports/dev-upload-9519139-adults.json`](../reports/dev-upload-9519139-adults.json); the contact sheet is [`reports/upload-9519139-adult-false-positives.jpg`](../reports/upload-9519139-adult-false-positives.jpg).

An experiment increased specialist cadence from every 30 source frames to every 6 on development clips. On the real child clip C2, operational recall (Unknown counted as a miss) fell from 65.7% to 51.4%. On adult-only S2 the same cadence produced 17 child false positives among 108 adult-labelled snapshots (15.7%). This measured alternative was rejected; the production default remains 30. This does not establish that the default is adequate.

## Production-path results

All rows below were generated by `scripts/evaluate_unseen_child_video.py`, which starts `odc.service.Service` and captures its seat rows, generic detections, raw specialist detections, association decisions, source frame sequence, provider, and final outputs. The runner uses a transient repository adapter; model and inference code follow the production service path. Sampled rows from the same video are correlated and must not be treated as independent subjects.

| Clip / split | Labels | Snapshots | TP | FP | TN | No/FN | Unknown | Child recall (Unknown=miss) | Adult FP rate (Unknown in denominator) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C2_child_seat.mp4` development | child seat | 35 | 23 | 0 | 0 | 0 | 12 | 65.7% | n/a |
| `C1_baby_carseat.mp4` development, paced 5 fps | infant with adult handling carrier | 33 | 26 | 0 | 0 | 1 | 6 | 78.8% | n/a |
| `S2_belt_up.mp4` development | adult driver | 108 | 0 | 0 | 7 | 0 | 101 | n/a | 0.0% |
| N1 night driver baseline | adult driver | 230 | 0 | 22 | 59 | 0 | 149 | n/a | 9.6% |
| N1 night driver corrected replay | adult driver | 255 | 0 | 0 | 10 | 0 | 245 | n/a | 0.0% |
| Uploaded `9519139` adult-only hard negative, paced 5 fps | two adults | 230 seat rows | 0 | 112 | 112 | 0 | 6 | n/a | 48.7% |
| Pexels 4524519 evaluation | clear child center; adjacent age unscored | 45 seat snapshots | 0 | 0 | 0 | 7 | 38 child rows; adjacent rows unscored | 0.0% | n/a |
| Pexels 6181423 | debug only; repeatedly inspected during development | prior report is not locked evaluation | — | — | — | — | — | — | — |

For Pexels 4524519, the 45 sampled center-seat child rows comprise 0 Yes, 7 No, and 38 Unknown. The adjacent occupant is visually age-ambiguous and is not used for adult metrics. The publisher describes two children, but only the clearly visible center occupant is scored as child in this evaluation. The clip contains no clearly annotatable child restraint; this is a child-without-restraint scenario. There are too few independent clips to report meaningful confidence intervals or claim generalization.

For the public child clip, sampled images show the center child clearly. In the raw records, person detections persist, but only three accepted age-specialist decisions appear: `adult` at source frame 61 (score 0.349, track `t1`), `child` at frame 91 (0.301, `t1`, one temporal hit, unconfirmed), and `child` at frame 241 (0.306, `t6`, one temporal hit, unconfirmed). The other sampled frames have no age candidate in the captured service output. The 38 Unknown rows carry `no-age-evidence`; the seven No rows carry `specialist-adult-occupant`. This means weak/sparse specialist output and temporal evidence explain most misses, while the adult prompt specifically explains the seven incorrect No outputs. The prediction was not lost in JSON serialization. This run took 97.5 s on arm64 CPU; recent-frame inference p50 was 203 ms and p95 257 ms. The paced adult-only upload took 276.6 s for 115 snapshots (0.416 snapshots/s), p50 262 ms and p95 376 ms, too slow for a smooth live showcase on this CPU setup.

N1 baseline and corrected sampling counts differ, so this is not a paired frame-by-frame comparison. The 6-frame-cadence experiment is in [`reports/dev-C2-frequency-6.json`](../reports/dev-C2-frequency-6.json) and [`reports/dev-S2-frequency-6.json`](../reports/dev-S2-frequency-6.json). Default-cadence adult and original baseline reports are in [`reports/fp-baseline-N1.json`](../reports/fp-baseline-N1.json), [`reports/fp-fixed-N1.json`](../reports/fp-fixed-N1.json), [`reports/fp-fixed-S2.json`](../reports/fp-fixed-S2.json), and [`reports/dev-upload-9519139-adults.json`](../reports/dev-upload-9519139-adults.json). The baseline N1 contact sheet is [`reports/fp-baseline-N1-contact-sheet.jpg`](../reports/fp-baseline-N1-contact-sheet.jpg). Adult false-positive and child-miss sample contact sheets are [`reports/upload-9519139-adult-false-positives.jpg`](../reports/upload-9519139-adult-false-positives.jpg) and [`reports/pexels-4524519-child-misses.jpg`](../reports/pexels-4524519-child-misses.jpg).

## Provenance and labeling

- Video: [Pexels, “Kids Wearing Their Seatbelts”](https://www.pexels.com/video/kids-wearing-their-seatbelts-4524519/), by Ketut Subiyanto. The source page lists 3840×2160, 25 fps, 11 seconds, and free use under the Pexels License. The local file is 1920×1080, 25 fps, 266 frames (10.64 seconds), SHA-256 `79b909b1267e2097069072d29f4e3312b84682bc50072c8d42b634acf0c9a611`. Local retrieval date was not recorded. No resizing or other video transformation was applied by the evaluator.
- Independent review: sampled video frames were inspected without using detector output to assign labels. The clearly visible center child is scored; the adjacent person is marked unscored. The source page description was checked as supplementary provenance, not substituted for frame-level visual review.
- Existing local uploads include Pexels 6181423 (now development/debug-only), 4524519, 10385816, 6954278, and 9519139, plus a byte-identical duplicate of 10385816. Sample frames from the other uploads were inspected; they did not provide additional verified child-present holdouts. Do not count a duplicate or adjacent frames as an independent video.
- Models: the COCO detector recognizes `person`, not age. Grounding-DINO phrase scores are uncalibrated. Runtime used CPU (`CPUExecutionProvider`); its custom extension was unavailable and fell back to CPU. The local Grounding-DINO checkpoint is not established here as cleared for commercial production; check the model manifest and upstream weight terms before deployment.

## Code and tests

Changes in this worktree:

- `odc/attributes.py`: no age-from-size fallback; missing age evidence is `Unknown` with reason `no-age-evidence`.
- `odc/state.py`: same-seat occupied-person association, adult/child evidence scoped to the person, specialist decision audit, weak-evidence temporal confirmation, and restraint-context gating.
- `odc/service.py`: frame history preserves raw specialist detections and each decision through the Service path.
- `odc/config.py`: separates restraint proposal area filtering from child score gates.
- `scripts/evaluate_unseen_child_video.py`: production-path runner with frame-level raw outputs, decisions, metrics, hashes, video metadata, timing, and provenance fields.
- `configs/seats/demo-c2-v0.yaml` and `configs/seats/eval-pexels-4524519-v0.yaml`: camera-specific association geometry; no age decision is based on box size.
- Regression tests cover uncertain age, box-size independence, child/adult person-track association, evidence expiry on track reassignment, weak evidence, Service history, and real C1/C2 integration clips.

Commands executed:

```sh
env -u NO_COLOR PYTHONPATH=. ./.venv/bin/pytest -q
env -u NO_COLOR PYTHONPATH=. ./.venv/bin/pytest -q tests/unit/test_state.py tests/unit/test_segmentation_attributes.py tests/unit/test_service.py

env -u NO_COLOR MPLCONFIGDIR=/private/tmp/plomid-mpl-cache PYTHONPATH=. ./.venv/bin/python scripts/evaluate_unseen_child_video.py footage/C2_child_seat.mp4 --layout configs/seats/demo-c2-v0.yaml --child-seats child-seat --scenario development-child-adult-overlap-no-size-classification --out reports/dev-C2-final.json
env -u NO_COLOR MPLCONFIGDIR=/private/tmp/plomid-mpl-cache PYTHONPATH=. ./.venv/bin/python scripts/evaluate_unseen_child_video.py footage/S2_belt_up.mp4 --layout configs/seats/demo-s2-v0.yaml --adult-seats driver --scenario adult-only-day-restraint-context-gate --out reports/fp-fixed-S2.json
env -u NO_COLOR MPLCONFIGDIR=/private/tmp/plomid-mpl-cache PYTHONPATH=. \\
  ./.venv/bin/python scripts/evaluate_unseen_child_video.py \\
  footage/uploads/unseen_pexels_4524519.mp4 \\
  --layout configs/seats/eval-pexels-4524519-v0.yaml \\
  --child-seats child-center --specialist-every-n 30 \\
  --video-title 'Kids Wearing Their Seatbelts' \\
  --source-url 'https://www.pexels.com/video/kids-wearing-their-seatbelts-4524519/' \\
  --publisher 'Ketut Subiyanto' \\
  --license 'Pexels License; free-use conditions on source page' \\
  --retrieved-date not-recorded --out reports/unseen-child-evaluation.json
```

The final full-suite command `env -u NO_COLOR PYTHONPATH=. ./.venv/bin/pytest -q` exited 0; 328 tests were collected, including both real child-video Service integration tests. Pytest showed environment-dependent skips and upstream CPU/custom-op and PyTorch deprecation warnings. After a formatting-only Service edit, a focused run once failed the one-second degraded-mode persistence assertion, then that test passed alone and the full affected unit group passed on retry (70 tests). This test is timing-sensitive under load; it was not weakened. `ruff check` passed for all changed Python source and test files. `git diff --check` passed.

## Current acceptance status

| Requirement | Status | Evidence |
|---|---|---|
| Size is not child/adult evidence | PASS in source | Decision path no longer uses occupant box area; regression test added. |
| Adult false-positive guard | FAIL | Paced adult-only 9519139: 112/230 false child positives (48.7%). S2 default cadence had 0/108 FP but 101/108 Unknown. |
| Genuine child detection | FAIL | Public Pexels 4524519: 0/45 child Yes, 38 Unknown, 7 No. C1: 26/33 Yes, 6 Unknown, 1 No. |
| Adult-only rejection across clips | FAIL | Paced uploaded 9519139 adult-only clip: 112 child false positives across 230 adult-labelled seat rows (48.7%). |
| Empty restraint never implies child | PASS in logic tests | Child output requires associated child-occupant evidence. |
| Service/evaluator path agreement | PASS for recorded path | Evaluator calls actual `Service`; decisions and raw detector output are serialized per snapshot. |
| Original frame 139 | BLOCKED | Original source video is not in the local uploads. |
| Multi-video independent validation | INCOMPLETE | One separately sourced child clip plus adult development clips; too few independent child videos. 6181423 is contaminated by repeated debugging. |
| Production readiness | FAIL / INCOMPLETE | Recall failure, high Unknown rate, uncalibrated model scores, insufficient independent validation, and checkpoint licensing not cleared. |

## Reproduction

Run the commands above from the repository root. The frame-level JSON records source-frame sequence, timestamp, GT, raw specialist phrase/score/box, association disposition, temporal confirmation, seat rows, and final child status. Inspect child failures with:

```sh
python3 -c 'import json; d=json.load(open("reports/unseen-child-evaluation.json")); [print(f["frame_seq"], f["specialist_detections"], f["specialist_decisions"], [(r["seat_id"],r["child"],r["child_reason"]) for r in f["rows"]]) for f in d["frames"] if any(r.get("ground_truth")=="child" and r["child"]!="Yes" for r in f["rows"])]'
```

Reproduce the paced adult-only hard-negative run and contact sheet with:

```sh
env -u NO_COLOR MPLCONFIGDIR=/private/tmp/plomid-mpl-cache PYTHONPATH=. \
  ./.venv/bin/python scripts/evaluate_unseen_child_video.py \
  footage/uploads/9519139-uhd_4096_2160_25fps-1791966051.mp4 \
  --layout configs/seats/eval-upload-9519139-v0.yaml \
  --adult-seats passenger-left driver-right \
  --scenario local-upload-two-adult-hard-negative-paced \
  --out reports/dev-upload-9519139-adults.json
env -u NO_COLOR PYTHONPATH=. ./.venv/bin/python \
  scripts/make_evaluation_contact_sheet.py \
  reports/dev-upload-9519139-adults.json \
  reports/upload-9519139-adult-false-positives.jpg \
  --kind adult-false-positive
```

The existing local assets contain MiVOLO V2, general COCO object detectors, face landmarks, and pose landmarks; none is a dedicated, validated cabin child/adult classifier. The current open-vocabulary detector alone does not meet the combined precision-and-recall objective. Work here is constrained to the already available pretrained assets and video clips; no training or fine-tuning is part of the implementation.

## 2026-10-23 cabin monitoring evidence refresh

This section supersedes prior “current” summaries above where values differ. All replays below use the actual production `odc.service.Service`; the evaluator records raw per-frame output in local JSON. Local videos are development/debug material except the Pexels 4524519 clip, which is independent of the current MiVOLO age-margin selection but had previously been evaluated with the older Grounding-DINO path. No locked video was used to choose thresholds. Confidence values in MiVOLO outputs are not calibrated probabilities.

### Fixes and regression evidence

- `odc/age.py`: MediaPipe face boxes can extend outside normalized image bounds. Passing a negative coordinate to validated `BBox` previously raised during face preparation and could discard an age batch. Coordinates are now clipped and degenerate/malformed boxes are omitted; unit tests cover both cases. The fresh Pexels run reports `age_loaded=true`, `age_error=null`.
- F1 cross-seat false child labels came from broad/overlapping seat polygons and person-only MiVOLO child estimates on boxes overlapping adjacent seat regions. `demo-f1-v0.yaml` now contains disjoint zones (layout version `demo-f1-v1`); face centers assigned uniquely to another zone and child/infant person boxes spanning multiple zones are surfaced as `unknown`, with an explicit reason. This is association uncertainty handling, not new age evidence.
- With the revised geometry and guards, F1 (55 snapshots × 3 seat rows) went from 4 child false positives in the initial replay to 0 in the final replay. Final: TP 36, FP 0, TN 80, FN 0, Unknown 49; child Unknown 19/55 (34.5%), adult Unknown 30/110 (27.3%). Child recall counting Unknown as a miss is 65.5%; the result is not sufficient for a precise-classification claim. The false-positive and child-miss sheets are `reports/phase4-F1-adult-FP.jpg` and `reports/phase4-F1-child-misses.jpg` (local ignored artifacts).
- `odc/belt.py`: a geometric no-strap result after six clear frames is no longer emitted as “Not detected”; it is `Unknown` unless a validated belt detector/electrical signal can support a negative. S2 footage showed a visible diagonal belt across the adult driver in sampled frames even though the prior algorithm returned “Not detected” on 44/108 rows. After the fail-safe change the same video returns Unknown 108/108, Detected 0, Not detected 0. This removes unsupported negative claims but confirms that belt detection remains incomplete.
- Posture output now explicitly reports `UNKNOWN` and a reason when keypoints exist without a validated posture classifier; pose keypoints remain separately exposed. The F1 run had keypoints on 34 rows (789 visible keypoints). This is not validated posture classification. Available restraint evidence remains generic chair/object evidence, not an occupant-in-restraint classifier; on F1 it was Unknown on 117/165 rows.

### Child/adult and occupancy evidence

The refreshed production replay on Pexels [“Kids Wearing Their Seatbelts”](https://www.pexels.com/video/kids-wearing-their-seatbelts-4524519/) by Ketut Subiyanto (Pexels License) processed 45 sampled child-seat rows: TP 27, Unknown 18, no false child positives. Recall counting Unknown as a miss is 60.0%. This is worse than the earlier run (28 Yes, 17 Unknown), and confirms that the model/association result is not stable enough to claim reliable child classification. The clip is 1920×1080, 25 fps, 266 frames / 10.64 s; SHA-256 `79b909b1267e2097069072d29f4e3312b84682bc50072c8d42b634acf0c9a611`. It ran at 0.614 sampled snapshots/s on arm64 CPU (MiVOLO batch median 437 ms, p95 1007 ms). See `reports/phase4-Pexels4524519-final.json` for frame-level output and provenance. The contact sheet can be recreated using the script below.

The historical R1 age-labelled cadence reports had 71/104 and 68/104 Unknown, but their age labels are invalidated by the visual review and are not used as adult-only performance evidence. F1 cadence 3 ran 0.84 snapshots/s versus 0.65 at cadence 1, but child Yes fell from 36 to 33 and child Unknown rose from 19 to 22. Keep cadence 1 as the default; cadence 3 is not a free performance improvement. E1’s earlier empty-cabin replay showed 41 empty and 24 Unknown, zero occupied false positives, but was only a one-seat view. These scene-specific layouts do not establish occupancy performance for all seats or arbitrary cabin cameras.

### Hardware and scope

The measured path used CPU and ONNX `CPUExecutionProvider` on arm64. CUDA is unavailable, MPS reports unavailable, and although CoreML is listed by ONNX Runtime, the RT-DETR manifest blocks it after observed corrupted/zeroed logits in parity checks. Therefore accelerated paths remain unverified/disabled. CPU replay speed is below real-time for multi-occupant use (F1 about 0.65 sampled snapshots/s); cadence 3 improves throughput but lost child positives, so it is not enabled by default. No throughput or accuracy claim is made for GPU/ANE acceleration.

Run the refreshed independent clip and contact sheet:

```sh
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/uploads/unseen_pexels_4524519.mp4 \
  --layout configs/seats/eval-pexels-4524519-v0.yaml \
  --child-seats child-center \
  --scenario heldout-MiVOLO-age-policy-external-child-video-not-used-for-margin-selection \
  --video-title 'Kids Wearing Their Seatbelts' \
  --source-url 'https://www.pexels.com/video/kids-wearing-their-seatbelts-4524519/' \
  --publisher 'Ketut Subiyanto / Pexels' --license 'Pexels License' \
  --retrieved-date 2026-10-13 --out reports/phase4-Pexels4524519-final.json
PYTHONPATH=. ./.venv/bin/python scripts/make_evaluation_contact_sheet.py \
  reports/phase4-Pexels4524519-final.json reports/phase4-Pexels4524519-review.jpg \
  --kind child-miss --max-frames 12
```

### Current acceptance

| Capability | Status |
|---|---|
| All-seat occupancy across these configured camera layouts | Implemented; limited scene-specific replay only, broader cabin/lighting/occlusion evidence still insufficient |
| Child/adult classification | Implemented with MiVOLO and association abstentions; insufficiently validated, misses/Unknown remain high |
| Restraint presence/occupied restraint distinction | Partial; no dedicated validated model, empty-vs-occupied distinction incomplete |
| Belt monitoring | Partial; positive geometric candidates exist, negative status safely abstains, representative worn-belt clip had no positive detections |
| Posture | Keypoints exposed; posture category explicitly Unknown pending validated classifier |
| Serialization and runtime fixes | Unit suite and actual Service/evaluator runs exercised; final full regression results recorded with this update |
| Hardware independence/performance | CPU works; CUDA/MPS unavailable, CoreML disabled due parity defect; CPU not real-time at full cadence |

Overall status remains **INCOMPLETE / not production-ready**. The next improvement must come from a better-supported inference method using pretrained assets, or a compatible pretrained checkpoint already available to the project. No dataset acquisition, training, or fine-tuning is proposed. This turn does not claim that a larger threshold or Unknown output solved child classification.

## Additional multi-capability implementation pass

The evaluator now separates seat occupancy labels from age labels. `--occupied-seats` / `--unoccupied-seats` lets an engineer score occupancy even when a rear-view occupant's age cannot be independently established. `--restraint-seats`, `--empty-restraint-seats`, belt visibility options, and optional frame-indexed JSON annotations keep those ground truths separate. Unknown is counted explicitly for each capability. `--sample-every-n` permits a faster diagnostic replay; sparse replays are not mixed with full-cadence metrics.

R1 rear-wide footage was visually reviewed after a previous note described its rear-left occupant as a child. The sampled view shows mostly the back of that occupant's head; age is not supportable from the available evidence. The old all-adult R1 label set and the later rear-left-child label set are both excluded from child/adult performance claims. The current R1 production replay scores all four seats for occupancy only, with age unscored; see `reports/showcase-R1-occupancy-v1-final.json`. This corrects an annotation risk rather than hiding a model error.

Pose keypoints are now passed through an explicit posture evidence function and serialized with `posture_landmarks_complete`; posture confidence remains unavailable. A first torso-axis heuristic was run on real F1 frames, and its contact sheet showed it labeled a visibly reclined infant as upright because the torso was vertical in image coordinates. That heuristic was removed. Current production output is `UNKNOWN` when landmarks exist but no calibrated seat/back reference is available, and `BODY_PART_OCCLUDED` when required shoulder/hip landmarks are missing. In the 28-snapshot F1 diagnostic replay, current posture statuses were 83 Unknown and 1 body-part-occluded across 84 rows; there are no validated posture classifications. The failed heuristic is not retained as a product result.

The current F1 Service diagnostic run used 28 snapshots (every 12 source frames) across three configured seats. It returned child TP 18, child Unknown 10, adult false positives 0, and adult Unknown 16. Occupancy was positive on 57/84 seat rows and Unknown on 27/84; sparse sampling makes this diagnostic only and is not directly comparable to the full cadence run. The known child-restraint seat returned a restraint candidate Yes on 12/28 samples and Unknown on 16/28; this is proxy evidence, not a child-seat-specific detector. No empty restraint clip is present in the local verified set, so empty-versus-occupied restraint performance remains unverified.

On S2, direct per-frame execution of the production geometric belt component using its recorded person crops produced only 4 strap candidates in 106 valid observations (frames 43, 55, 103, 109); the remaining observations were 58 clear and 44 unclear. The production temporal belt status stayed Unknown throughout the refreshed run because candidates were too sparse to meet the temporal confirmation rule. In F1, the Service emitted 12 belt Detected candidates in the 28-snapshot diagnostic replay; a review sheet is `reports/showcase-F1-belts.jpg`. These are strap visibility candidates, never fastening claims. The evidence shows viewpoint-specific inconsistency; belt detection is not complete.

Reproduce the efficient diagnostic replays:

```sh
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/F1_family_ride.mp4 \
  --layout configs/seats/demo-f1-v0.yaml \
  --child-seats child-seat-right \
  --adult-seats near-adult-left mid-adult-center \
  --restraint-seats child-seat-right --sample-every-n 12 \
  --scenario local-family-ride-posture-unknown-occupancy-restraint \
  --out reports/showcase-F1-parallel-final.json
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/R1_rear_wide.mp4 \
  --layout configs/seats/demo-r1-v0.yaml \
  --occupied-seats driver-front rear-left rear-center rear-right \
  --scenario local-four-seat-occupancy-replay-age-unscored-rear-views \
  --out reports/showcase-R1-occupancy-v1-final.json
```

These changes improve attribution, uncertainty reporting, and observable posture/occupancy diagnostics. They do not overcome the local asset limits: only MiVOLO supplies age estimates; COCO checkpoints have no child-seat or seatbelt class; the available Grounding DINO weights remain evaluation-only; and no local model provides validated seat-relative posture. Therefore all five requested areas are **not complete** and the product must not present these candidate outputs as precise safety decisions.

### R1 four-seat geometry correction

The four-seat R1 layout had a front driver zone overlapping the rear-center zone; the production seat matcher ranks a person's full-body box center and overlap, so the front driver's box was repeatedly assigned to the rear zone or left unassociated. Layout version `demo-r1-v1` uses four disjoint x-zones aligned to the viewed cabin. The actual Service replay used 26 snapshots × 4 seats (104 rows): old geometry gave 56 occupied/likely rows and 48 Unknown; v1 gave 74 occupied/likely and 30 Unknown, with zero occupied calls on unoccupied seats—but this clip has no independently annotated empty seat, so false-positive occupancy is unmeasured. Driver-front track presence rose from 3/26 to 24/26. Age is intentionally unscored in this replay. The updated layout has a regression asserting that seat polygons do not overlap. Current evidence: `reports/showcase-R1-occupancy-v1-final.json`.

### Possible-child evidence path

The output now carries `child_candidate`, `child_candidate_track_ids`, and a reason separately from the confirmed `child` field. MiVOLO point estimates at or below the configured child boundary (17 by default) can surface as **Possible child** while the final classification remains Unknown if it lies in the configured uncertainty band. Seat-association-suppressed tracks cannot create this candidate. The UI shows confirmed child and possible child separately, and the counter/API field is additive. This is a recall-preserving review cue; it is not a child classification or calibrated probability.

On the public Pexels child video, the refreshed actual-Service run returned 27 confirmed child rows, 18 Unknown (confirmed-child recall 60.0% when Unknown is a miss), and possible-child evidence on 42/45 independently labeled child rows (93.3%). Of the 18 Unknown, 15 carried an unconfirmed age-range cue and 3 lacked a current track. On the local 9519139 adult hard-negative, a sparse cadence-24 replay returned 0 possible-child candidates and 0 confirmed child Yes among 58 adult rows (54 No, 4 Unknown). This is one internal adult video with unverified original reuse metadata and adjacent frames; it is evidence against a broad candidate flood, not an independent false-positive guarantee. Frame reports are `reports/showcase-Pexels-child-candidate.json` and `reports/showcase-adult-9519139-candidates.json`; the public-video Unknown contact sheet is `reports/showcase-Pexels-child-candidate-review.jpg`.

## Occupancy, restraint, belt and posture correction pass (2026-10-24)

### Occupancy association

Source review found that `SeatStateEngine.update` previously computed one `ambiguous_nearby` Boolean for the whole cabin and passed it to every seat. A person whose box tied between two zones therefore blocked empty-seat confirmation in every unrelated seat. The engine now computes the competing seat set per ambiguous track and only blocks those seats. A deterministic three-seat regression test confirms that the two tied seats remain unresolved while a separate visible empty seat reaches `EMPTY_CANDIDATE`.

The local N1 night-driver layout also excluded many tracked detections: person boxes centered near x=0.49 while the old single-seat ROI ended at x=0.38. Some wide false/person boxes still overlapped the ROI, so eight rows had a track but none reached an occupied state in the 40-snapshot sparse baseline. The camera-specific layout is versioned to `demo-n1-v1` and now covers x=0.30–0.68. Through the production Service, this sparse replay changed to 17 occupied/likely, 23 Unknown, and 14 seat-associated track rows. This is a useful association correction, but the dark scene still leaves many frames unresolved and the sparse run is not a general recall estimate. The 40 rows contain 17 occupied TPs and 23 Unknown against the single occupied-seat label. The earlier full-cadence N1 report and this camera-specific update should not be generalized to other cameras.

The E1 production replay at source-frame cadence 6 scored 65 independently labeled empty-seat snapshots: 41 true empty classifications and 24 Unknown, with zero false occupied calls. CPU throughput was 2.04 sampled snapshots/s. This confirms the temporal empty path when enough observations accumulate; it does not validate other empty cabin views. R1's previously documented four-seat 104-row replay and F1's occupied-seat replay remain additional camera profiles. Across the local material, empty-seat and multi-seat coverage is still narrow.

### Restraint semantics

The local COCO model has no child-restraint class. Source inspection found the application promoted sustained generic `chair`/`couch` or large `backpack`/`handbag`/`suitcase` proxies to restraint `Yes`; layout names and apparent occupant size also influenced the result. Those signals can be an upholstery/object candidate but cannot verify a child restraint, its type, or whether a child is inside it. The live output now keeps these as `restraint_candidate`; `restraint`, type, and restraint occupancy remain `Unknown` without validated direct evidence. Seat IDs, occupant size, and seat occupancy no longer manufacture a restraint result. This corrects a false-certainty bug; it does not provide a restraint detector. In the C1 actual-Service sparse replay, generic objects produced candidates on 6 seat rows, while all 14 restraint-presence and restraint-occupancy rows remained Unknown. There is no verified empty-child-restraint local clip, so empty-versus-occupied-restraint performance is blocked by unavailable model evidence and ground truth.

### Belt status

The real-time output now retains per-frame geometric belt observations (`belt_observation`, reason, score and torso ROI) beside the temporal status, making it possible to distinguish “no usable torso crop” from “clear pixels” and “strap-like candidate.” A temporal `Detected` result means only a repeated strap-shaped image cue, not a verified belt or fastened buckle. The UI now calls this a “strap candidate” and no longer labels it “belt ok.” No pretrained seatbelt detector or vehicle buckle signal exists in the local assets. The actual Service replay on S2 sampled 54 frames of a visibly belted driver: 1 raw strap candidate (`bright-band`, score 0.75), 26 raw clear candidates, 22 unclear observations, and 5 frames without a usable torso crop. All 54 temporal belt statuses stayed Unknown; the single strap hit could not meet the three-vote confirmation rule. Manual review of the sole candidate contact sheet (`reports/current-S2-belt-raw-candidate.jpg`) shows an arm/bright clothing crossing the torso, making this a likely false cue rather than a reliable belt hit. Runtime was 2.01 snapshots/s on arm64 CPU. This is a measured miss, not a completed detector. No negative belt claim is made from absent geometric evidence. The frame-level output is `reports/current-S2-diagnostics.json`.

### Posture

The invalid torso-image-axis heuristic remains removed; it labeled the visibly reclined infant in `reports/showcase-F1-posture.jpg` as upright. Pose keypoints and missing-landmark status are preserved, but no local checkpoint provides a validated seat-relative posture classifier or calibrated per-seat back axis. The current Service returns `UNKNOWN` when required landmarks exist but reference calibration is unavailable, and `BODY_PART_OCCLUDED` when required landmarks are missing. This is a deliberate abstention, not a completed posture classifier. There are no independently posture-annotated local clips, so posture accuracy is unmeasured.

Reproduce the updated production-path diagnostic runs:

```sh
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/E1_empty_cabin.mp4 \
  --layout configs/seats/demo-e1-v0.yaml --unoccupied-seats driver \
  --sample-every-n 6 --age-every-n 30 --scenario local-empty-cabin-full-empty-dwell \
  --out reports/current-E1-full-capability.json
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/N1_night_drive.mp4 \
  --layout configs/seats/demo-n1-v0.yaml --adult-seats driver --occupied-seats driver \
  --sample-every-n 30 --age-every-n 30 --scenario local-night-driver-v1-occupancy \
  --out reports/current-N1-v1-capability.json
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/C1_baby_carseat.mp4 \
  --layout configs/seats/demo-c1-v0.yaml --child-seats infant-carrier \
  --restraint-seats infant-carrier --sample-every-n 30 \
  --out reports/current-C1-capability.json
```

Overall capability status remains **INCOMPLETE**: occupancy improved on three local camera configurations but still has Unknowns and limited coverage; restraints cannot be distinguished reliably with current assets; belts remain candidate-only and miss visible examples; posture remains unclassified without seat-relative references and annotated validation.

### Additional cross-view and posture audit (2026-10-24)

I ran the production-Service evaluator on additional local cabin clips and inspected the sampled evidence. These are correlated frames from a small number of videos, not independent subjects or a representative validation set.

| Video / condition | Processed seat rows | Occupancy result | Other observed result |
|---|---:|---|---|
| E1 empty cabin, daylight | 65, one configured empty seat | 41 empty candidates (TN), 24 Unknown, 0 occupied false positives | 2.04 snapshots/s on arm64 CPU |
| N1 night driver, revised ROI | 40, one configured occupied seat | 17 occupied, 23 Unknown, 0 empty calls | 14 track-associated rows; 1.42 snapshots/s |
| F1 family ride, mixed adults and child carrier | 84 rows across 3 seats | 57 occupied/likely, 27 Unknown, 0 occupied false positives in this all-occupied layout | 34 pose skeletons; posture 83 Unknown, 1 BODY_PART_OCCLUDED; 12 temporal belt detections remain unverified strap cues |
| C2 child seat, adult handler, adjacent adult | 36 rows across 2 seats | 26 occupied/likely, 10 Unknown in the earlier replay | Full reviewed child-seat replay: 13 occupied, 5 Unknown across 18 frames; all 18 restraint outputs Unknown |
| S2 adult driver, visible belt | 54 rows | 49 occupied, 5 Unknown | Belt output Unknown 54/54; one raw hit was visually judged arm/clothing |
| C1 infant carrier with adult handling | 14 rows across 2 seats | 2 occupied/likely, 12 Unknown in sparse replay | Restraint Unknown 14/14; generic candidates on 6 rows |

The C2 source frames at indices 1, 13, …, 205 were manually labeled independently of model outputs in [`eval/annotations/local-c2-posture-review-2026-10-24.json`](../eval/annotations/local-c2-posture-review-2026-10-24.json). A single reviewer judged the clearly visible child to be grossly upright/seated in the restraint in all 18 sampled frames. The actual Service returned 11 `UNKNOWN` and 7 `BODY_PART_OCCLUDED` posture outputs (0/18 non-Unknown classifications); the failure/contact sheet is [`reports/current-C2-posture-reviewed-failures.jpg`](../reports/current-C2-posture-reviewed-failures.jpg), and the annotated production run is [`reports/current-C2-posture-review.json`](../reports/current-C2-posture-review.json). This is direct evidence that pose landmarks do not currently deliver posture classification. The small, single-reviewer sequence is a failure audit, not a posture benchmark.

The C2 belt geometry reported `clear_candidate` in 18 of 36 seat rows, but those rows include both occupied areas and unavailable torso views; no independently labeled belt status was attached to this clip. This does not establish belt detection. The single S2 raw “strap” hit was manually inspected and appears to be the driver's arm/light clothing. It is a false cue in the available evidence, while the remaining S2 output is abstention. Do not use F1's 12 temporal `Detected` rows as belt-fastened decisions; they are unvalidated geometric candidates.

Restraint detection remains **not implemented as a validated class**. All available local restraints are occupied in the reviewed C1/C2/F1 scenes; the COCO object detector has no child-restraint class. Generic object proposals are explicitly candidates and cannot distinguish an empty restraint from one with a child. No verified empty-restraint clip was found among the examined local files, so empty-versus-occupied performance cannot be measured. No threshold change can remedy the missing class evidence.

Occupancy now has exercised examples for empty daylight, night, mixed/partially occluded family scenes, and a wide multi-seat view. Evidence remains narrow: E1 is one empty view, N1 one night view, and the F1/C2 seat rows represent repeated frames of just two clips. R1 is also one four-seat cabin replay and does not add lighting or subject diversity. The Unknown rates above are material and must remain in reported coverage; no claim of robustness across cabin viewpoints, lighting, or occlusion is made.

Reproduce the frame-annotated C2 posture audit and image review:

```sh
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/C2_child_seat.mp4 \
  --layout configs/seats/demo-c2-v0.yaml --child-seats child-seat \
  --occupied-seats child-seat --restraint-seats child-seat \
  --annotations eval/annotations/local-c2-posture-review-2026-10-24.json \
  --disable-age-model --sample-every-n 12 \
  --scenario local-C2-manually-reviewed-posture-occupancy-restraint \
  --out reports/current-C2-posture-review.json
PYTHONPATH=. ./.venv/bin/python scripts/make_evaluation_contact_sheet.py \
  reports/current-C2-posture-review.json \
  reports/current-C2-posture-reviewed-failures.jpg \
  --kind posture-reviewed-failure --max-frames 18
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/F1_family_ride.mp4 \
  --layout configs/seats/demo-f1-v0.yaml \
  --occupied-seats child-seat-right mid-adult-center near-adult-left \
  --child-seats child-seat-right --restraint-seats child-seat-right \
  --disable-age-model --sample-every-n 12 \
  --scenario local-F1-three-seat-occupancy-restraint-candidate-pose \
  --out reports/current-F1-capability.json
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/S2_belt_up.mp4 \
  --layout configs/seats/demo-s2-v0.yaml --occupied-seats driver \
  --belt-visible-seats driver --sample-every-n 12 --age-every-n 30 \
  --scenario local-s2-visible-belt-direct-evidence-diagnostics \
  --out reports/current-S2-diagnostics.json
```

The annotation covers only the C2 sampled sequence and is not a training set. It does not validate posture on other seat geometries or camera views.

Verification in this pass: `env -u NO_COLOR PYTHONPATH=. ./.venv/bin/pytest -q tests/unit` completed successfully; `./.venv/bin/ruff check scripts/evaluate_unseen_child_video.py scripts/make_evaluation_contact_sheet.py`, JSON parsing of the aggregate and C2 reports/annotations, and `git diff --check` also passed. The production evaluator and contact-sheet commands above were executed, and their generated outputs are retained in `reports/`.

### Pose-to-track association correction (2026-10-24)

Source inspection found pose inference was run on all sufficiently sized detections, including non-person classes, then each skeleton was assigned to a seat by polygon overlap. Belt and posture enrichment subsequently selected a seat's largest occupant box but could consume a different geometry-assigned skeleton. This allowed an object or a neighboring person to supply pose evidence to the wrong occupant.

The Service now filters pose input to `person` detections with IoU >= 0.10 against an active occupant track that already has a seat assignment. It keeps at most one detection per track (the greatest overlap), records the matched track ID and box with each skeleton, and uses the same track's skeleton for that occupant's belt ROI and posture fields. Unmatched detections provide no pose evidence. This fixes an association defect; it does not make MediaPipe a posture classifier.

Actual Service replays after the change show no functional improvement in the currently failing attributes. On C2, occupancy remained 13 occupied and 5 Unknown of 18 manually reviewed child-seat snapshots; restraint remained Unknown in all 18. Posture remained Unknown/occluded in all 18 (12 Unknown, 6 BODY_PART_OCCLUDED), versus 11/7 before the association change. On S2, belt status remained Unknown in all 54 snapshots, with the same 26 clear candidates, 22 unclear observations, 5 unavailable observations, and one bright-band cue at source frame 109. The cue still visually resembles arm/clothing. These runs confirm the change did not solve posture or belt detection.

The complete reports are [`reports/current-C2-pose-track-fix.json`](../reports/current-C2-pose-track-fix.json) and [`reports/current-S2-pose-track-fix.json`](../reports/current-S2-pose-track-fix.json). Focused tests cover person-class filtering, correct seat/track association, unmatched detections, duplicate boxes, and invalid IoU thresholds. The original broad capability blockers therefore remain: no validated restraint detector, no empty-versus-occupied restraint evidence, no reliable belt result on S2, no non-Unknown posture result on the reviewed C2 frames, and limited occupancy diversity.

### Seat-scoped pose crop follow-up (2026-10-24)

The C2 child track's detector box spans x=0.21–0.92, across both the child seat and the adjacent adult. Pose inference now clips the crop to the matched track's seat-zone bounds while retaining the original track box for association. The emitted skeleton includes both boxes so the crop can be audited. A synthetic geometry test and the real C2 production-Service integration pass; the C2 test verifies child-seat pose crops stay within x=0.27–0.64. This prevents the adjacent adult region from being fed as part of the child-seat pose crop.

The corrected crop does not make posture classifiable in this clip. The annotated C2 child-seat rows remain 12 `UNKNOWN` and 6 `BODY_PART_OCCLUDED` out of 18; the shoulder landmarks are visible in some frames, but the pose model places hips at the lower frame boundary with zero visibility. Posture remains incomplete, and the result is recorded in [`reports/current-C2-seat-clipped-posture.json`](../reports/current-C2-seat-clipped-posture.json). A seat-relative axis cannot recover landmarks absent from the image.

### Showcase readiness gate (2026-10-24)

**Decision: NO-GO for a showcase that represents all six requested functions as working monitoring capabilities.** Existing pretrained models remain in place. The application exposes uncertain evidence safely, but measured coverage and attribute detection are not adequate to present these outputs as reliable. Unknown is counted as unavailable, never as a correct detection.

Current MiVOLO age-band reports with manual seat-level labels:

| Video/report | Ground truth | Confirmed child Yes | Child No | Child Unknown | Other result |
|---|---:|---:|---:|---:|---|
| F1 mixed family, `reports/phase4-F1-final.json` | 55 child-seat rows; 110 adult-seat rows | 39/55 | 0/55 | 16/55 | 1 adult row falsely classified child; 80 adult No, 29 adult Unknown |
| Pexels 4524519, `reports/phase4-Pexels4524519-final.json` | 45 child-seat rows | 27/45 | 0/45 | 18/45 | No adult-negative rows in this clip |
| C1 baby/carrier, `reports/current-C1-capability.json` | 7 child-seat samples | 4/7 | 0/7 | 3/7 | MiVOLO loaded; small sample |

These are repeated frame/seat observations from few clips, not independent subject counts. Child-positive detections are real, but there is substantial abstention and an adult false positive in F1. The older 9519139 adult-only report predates the current six-year uncertainty band and is not current-policy validation.

Other reviewed production-Service evidence:

- **Occupancy:** R1 four-seat replay has 26 `OCCUPIED`, 30 `LIKELY_OCCUPIED`, and 48 `UNKNOWN` rows out of 104. E1 empty-cabin replay has 41 `EMPTY_CANDIDATE` and 24 `UNKNOWN` out of 65. F1 has 50 `OCCUPIED`, 7 `LIKELY_OCCUPIED`, and 27 `UNKNOWN` across three mapped seats. This covers several layouts and lighting conditions but not enough calibrated views to establish all-seat coverage. N1 night footage remains heavily Unknown.
- **Child restraints:** C1 shows an occupied carrier, while the current capability replay reports restraint Unknown for all 14 seat rows. Available models do not provide validated restraint-class evidence. There is no independently annotated empty-restraint sample that supports empty-versus-occupied performance claims.
- **Seatbelts:** S2 reports Unknown on all 54 sampled frames in `reports/current-S2-pose-track-fix.json`. Visual review of `reports/phase4-S2-belt-contact.jpg` confirms diagonal belt webbing across the driver in several frames. The current system misses assessable positive evidence. One raw bright-band candidate looks like arm/clothing and is not valid strap evidence.
- **Posture:** C2 has 18 manually reviewed upright-child samples; after track association outputs are 12 `UNKNOWN` and 6 `BODY_PART_OCCLUDED`, with zero classified posture results. Pose landmarks are available, but there is no validated seat-relative posture classifier.
- **Service/API/UI path:** the C1 and C2 production-Service integration tests passed. Versioned live-state API tests preserve Unknown, candidate, belt, restraint, and occlusion states. This verifies tested transport/display mappings, not perception accuracy.

Verification command executed successfully: **54 passed**.

```sh
PYTHONPATH=. ./.venv/bin/pytest -q \
  tests/unit/test_age.py tests/unit/test_pose_association.py \
  tests/unit/test_posture.py tests/unit/test_belt.py \
  tests/unit/test_service.py tests/unit/test_console.py \
  tests/integration/test_child_detection_c1.py \
  tests/integration/test_child_detection_c2.py
```

The C1 and C2 integration tests launch the production Service evaluator on local clips with the installed RT-DETR and MiVOLO checkpoints and assert a child prediction survives association and serialization. Unit coverage includes API output, Service, age policy, belt, posture, and track association. The full repository suite and accelerated-hardware paths were not run in this gate.

Machine-readable evidence is in [`reports/showcase-readiness-2026-10-24.json`](../reports/showcase-readiness-2026-10-24.json). A limited demo may show the seat overlays and selected child-positive clips if clearly labelled experimental. Do not present child/adult classification, restraint occupancy, belt status, or posture as consistently reliable monitoring. These gaps cannot be closed by relabeling the current Unknown outputs.

### Current-policy age-classification recheck (2026-10-24)

The previous F1 frame-325 adult false positive was replayed through the current production Service with the installed MiVOLO V2 checkpoint and current six-year uncertainty band. On this 28-snapshot replay, the child seat produced 18 `Yes`, 10 `Unknown`, and no `No`; the two adult seats produced 40 `No`, 16 `Unknown`, and zero `Yes`. Child recall counting Unknown as a miss is 18/28 (64.3%); adult-only false-positive rate in these mapped adult seat observations is 0/56. These are repeated observations from one development clip, not independent video-level estimates.

At source frame 325, the center adult's person box overlaps adjacent seat regions and MiVOLO's person-only estimate is 4.05 years. The raw track estimate is retained for audit, but final classification is `Unknown` with reason `person-box-overlaps-multiple-seats`; the adjacent child-seat row is also `Unknown` because it has no age evidence at that frame. The original stored F1 report's `Yes` for that adult is therefore a stale pre-safeguard result, not the current Service output. The real-Service regression is `tests/integration/test_child_detection_f1_mixed.py`.

The current-policy adult-only replay of the existing Pexels 9519139 clip sampled 23 frames and produced 0 child `Yes`, 42 `No`, and 4 `Unknown` across its two annotated adult seats. This is a policy recheck of an already available clip, not a newly independent evaluation. Together, these replays support the cross-seat abstention safeguard and no adult-to-child false positive in these samples, but do not establish child-classification reliability: child abstention remains substantial, and the model has no calibrated age-class probability. Full raw outputs and provenance are in `reports/current-F1-age-recheck.json` and `reports/current-adult-only-9519139.json`.

### S2 belt frame-label correction and production baseline (2026-10-24)

The earlier S2 evaluator used the seat-wide `--belt-visible-seats driver` flag, which assigned visible-belt ground truth to every sample in a clip that shows both fastening and later driving. That was not a valid frame-level annotation. A reviewer inspected 54 source frames without prediction overlays and labeled webbing visibility (not buckle fastening): frames 37–373 show belt webbing across/in front of the torso (29 samples); frames 1, 13, 25 and 385–637 show an assessable torso with no webbing across it (25 samples).

The corrected annotations are in [`eval/annotations/local-s2-belt-review-2026-10-24.json`](../eval/annotations/local-s2-belt-review-2026-10-24.json). The real production Service was replayed with the annotation file and the ordinary belt detector. All 54 statuses were still `Unknown`: 29 Unknown on visible-webbing samples and 25 Unknown on assessable no-webbing samples. Raw cue counts were 26 clear candidates, 22 unclear, 5 unavailable, and 1 strap candidate. Thus this review confirms positive-detection misses while also showing the prior “54 positive frames” count was incorrect. It does not validate the detector on other drivers or camera angles.

Auditable frame-level replay: [`reports/current-S2-belt-reviewed.json`](../reports/current-S2-belt-reviewed.json); full contact sheet: [`reports/current-S2-belt-reviewed-contact.jpg`](../reports/current-S2-belt-reviewed-contact.jpg). Reproduce with:

```sh
env -u NO_COLOR PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/S2_belt_up.mp4 \
  --layout configs/seats/demo-s2-v0.yaml --occupied-seats driver \
  --annotations eval/annotations/local-s2-belt-review-2026-10-24.json \
  --disable-age-model --sample-every-n 12 \
  --scenario local-S2-manually-reviewed-belt-visibility-v1 \
  --out reports/current-S2-belt-reviewed.json
PYTHONPATH=. ./.venv/bin/python scripts/make_evaluation_contact_sheet.py \
  reports/current-S2-belt-reviewed.json \
  reports/current-S2-belt-reviewed-contact.jpg \
  --kind belt-unknown --max-frames 54
```

### S2 belt candidate correction (partial)

Source review of visible samples found long single belt edges that the previous geometry detector rejected: the minimum line length was 0.60 of the torso-ROI diagonal and the minimum angle was 15 degrees. The installed detector's measured edges on reviewed visible frames were shorter and shallower. The candidate detector now accepts a single edge from 0.28 of the ROI diagonal and a 10-degree minimum. The temporal gate still requires at least 3 strap candidates in its 10-observation window; its vote fraction is 0.30 so repeated positives can survive intermittent clear frames. These changes retain `Unknown` when the evidence is insufficient and do not generate a `Not detected` state from a camera-only clear ROI.

The before/after real-Service run used the same 54 reviewed frames, source video, seat layout, and annotations. Before: `Detected=0`, `Unknown=54`, one raw strap candidate. After: `Detected=3`, `Unknown=51`, three raw strap candidates. All three confirmed strap outputs are on visible-webbing frames; none of the 25 reviewed no-webbing frames was labeled `Detected`. The other 26 visible frames remain Unknown, and the 25 no-webbing frames also remain Unknown; therefore the fix demonstrates positive recall improvement but does not complete the feature or establish generalization. Review [`reports/current-S2-belt-fixed.json`](../reports/current-S2-belt-fixed.json), the three-positive sheet [`reports/current-S2-belt-fixed-detected.jpg`](../reports/current-S2-belt-fixed-detected.jpg), and the Unknown examples [`reports/current-S2-belt-fixed-unknown.jpg`](../reports/current-S2-belt-fixed-unknown.jpg).

Regression coverage adds a single-edge synthetic detector case, repeated-vote and two-vote temporal cases, and a real production-Service S2 test. Commands and observed result:

```sh
PYTHONPATH=. ./.venv/bin/pytest -q \
  tests/unit/test_belt.py tests/unit/test_belt_fallback.py \
  tests/integration/test_seatbelt_detection_s2.py
./.venv/bin/ruff check odc/belt.py tests/unit/test_belt.py \
  tests/integration/test_seatbelt_detection_s2.py
```

Result: 24 belt tests passed (23 unit, 1 real-video Service integration); Ruff passed. The real-video assertion also verifies no confirmed belt detection on the 25 manually reviewed assessable no-webbing samples. The application still returns Unknown rather than a verified negative when the strap detector finds no candidate.

### S2 source-label correction and seat-scoped belt crop (2026-10-24)

An enlarged source-only review found the prior S2 annotation incorrectly labeled the post-fastening portion (frames 385–637) as belt-absent. The webbing remains visible across the driver's torso through the end of the clip. Updated ground truth is 51 visible-webbing frames (37–637) and only 3 assessable absent frames (1, 13, 25). The earlier 29/25 metrics and its claim of zero false detections over 25 negatives are superseded; only 3 negative samples remain, too few for a useful false-positive estimate.

The corrected-label production baseline produced `Detected=3`, `Unknown=51` (3/51 visible detections; 0/3 absent detections). Inspection found the application's person detector often merged driver and cabin into a box larger than 70% of the image, causing 22 belt observations to be rejected as `box-unreliable`. Belt analysis now clips the already seat-associated occupant box to that seat's configured zone, the same seat-scoped crop contract used for pose inference. The production replay then produced `Detected=8`, `Unknown=46`: 8/51 visible samples detected, 43 visible samples Unknown, and all 3 absent samples Unknown. This is a real positive change with no false positive in a very small negative sample; belt detection and reliable negative status remain incomplete.

Reproduce the corrected-label replay with:

```sh
env -u NO_COLOR PYTHONPATH=. MPLCONFIGDIR=/private/tmp/odc-mpl-cache ./.venv/bin/python \
  scripts/evaluate_unseen_child_video.py footage/S2_belt_up.mp4 \
  --layout configs/seats/demo-s2-v0.yaml --occupied-seats driver \
  --annotations eval/annotations/local-s2-belt-review-2026-10-24.json \
  --disable-age-model --sample-every-n 12 \
  --scenario current-S2-seat-clipped-belt-test \
  --out reports/current-S2-seat-clipped-belt.json
```

The corrected annotation is [`eval/annotations/local-s2-belt-review-2026-10-24.json`](../eval/annotations/local-s2-belt-review-2026-10-24.json). The full frame output is [`reports/current-S2-seat-clipped-belt.json`](../reports/current-S2-seat-clipped-belt.json); `tests/integration/test_seatbelt_detection_s2.py` checks the corrected 51/3 labels and rejects confirmed detections on assessable absence frames.

### Console evidence-label correction (2026-10-24)

The underlying occupancy enum is `EMPTY_CANDIDATE`, but the live and seat-summary pages previously titled its count simply “Empty”. That overstated what the state means. The pages now say “Empty candidates” and explain that the state is sustained clear-scene evidence, not verified absence. The additive API count `empty_candidates` is available while the legacy `empty` field remains unchanged for compatibility. Belt counts are labeled “Strap candidates” or “Belt status”, matching the candidate-only detector semantics. Console and attribute tests pass.

### Final verification and unresolved regression

- Focused age, tracking/pose association, pose-crop, and real F1 Service regression command: `PYTHONPATH=. ./.venv/bin/pytest -q -o addopts='' tests/unit/test_age.py tests/unit/test_tracking_pose.py tests/unit/test_pose_association.py tests/integration/test_child_detection_f1_mixed.py` — **41 passed** (74.90 s). The production F1 test asserts at least 23/28 child Yes, at most 5 child Unknown, no adult Yes in 56 rows, and frame 325 abstains on ambiguous seat overlap.
- Full suite command rerun after storage/API changes: `env -u NO_COLOR PYTHONPATH=. ./.venv/bin/pytest -q -o addopts=''` — **350 passed, 32 skipped, 1 failed** (274.95 s). The failure remains `tests/integration/test_validate_dataset.py::test_dev_split_report_structure`: expected `seat_wrong == 0`, got `1`, at validation dataset F1 source frame 90. The seat-vote code treats the no-child-person match as `EMPTY_CANDIDATE` under the legacy dataset layout, despite the annotation saying occupied. This is a real remaining occupancy/evaluation defect outside the child-age/storage code changed here; it was retained and not masked by changing the assertion. Reproduction: `env -u NO_COLOR PYTHONPATH=.:tests/integration ./.venv/bin/pytest -q -o addopts='' tests/integration/test_validate_dataset.py::test_dev_split_report_structure`.
- Ruff lint, Ruff format check on all changed Python files, and `git diff --check` pass. Strict mypy was also inspected and reports five existing typing errors in `odc/service.py:1455` and `scripts/evaluate_unseen_child_video.py:425,431-433`; these locations are outside the age/crop edits and were not broadened into this pass.

**Closure status:** the five occluded F1 misses are now explained and correctly remain Unknown; the sixth, association-dominated frame is recovered. Independent child-present and adult-only smoke evaluations were completed on three child videos and one adult-only video with the per-frame outputs preserved. The age-band review supports retaining the existing uncertainty policy but cannot calibrate it. The child-recall work is improved and reproducible, but the repository is **not fully closed / not production-ready**: five occluded child observations remain Unknown, the adult sample is only 20 rows from one exterior-view video, and the complete suite has one unrelated real-model seat-validation failure.

### PLOMID persistence and UI contract audit (2026-10-24)

Application state and chatbot history now use PLOMID as the durable store. Migrations V006–V007 add one versioned JSONB seat observation per processed frame and append-only assistant interactions; existing SQL state/events/layouts/metrics, JSONB analytics and probe payloads, and VECTOR detection boxes remain in their current tables. The console's history API and seat-detail view read the stored frame observation, so age classification, confidence semantics, restraint/belt/posture states, pose keypoints, and reasons are not dropped when reopening a session. Mistral/Ollama calls are user-triggered; prompt, response/status, model/provider, and the exact context supplied are stored in PLOMID and exposed through the session-scoped assistant-history API.

Runtime does not write CSV or JSON state sidecars. Uploaded source videos are retained under `footage/uploads`; JPEG frame previews are bounded in memory. JSON manifests and YAML configuration are repository-owned model/config assets, and existing evaluation JSON/CSV outputs are explicit developer artifacts, not the application's runtime store. Assistant history currently has no retention expiry, so define a retention policy before using real occupant data or user prompts in a deployment.

Verified against the local PLOMID server with `ODC_TEST_DSN=postgresql://plomid:secret@127.0.0.1:55437/plomid PYTHONPATH=. ./.venv/bin/pytest -q -o addopts='' tests/integration/test_persistence_pg.py tests/integration/test_console_app.py -k 'not sse_streams'`: **13 passed, 1 skipped, 1 deselected**. The integration coverage includes fresh migration, JSONB observation round-trip, VECTOR-compatible existing schema, assistant history, API/UI agreement, and stored unavailable/scope-guard assistant responses. This validates persistence/API behavior only; it does not improve the perception limitations documented above.
