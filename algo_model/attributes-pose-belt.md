# Optional attributes: pose, belt, child and DMS

Pose uses the configured MediaPipe/pose adapter when enabled and records
keypoint visibility plus posture candidates. Belt logic detects visible strap
segments and clear-torso candidates; it cannot prove buckle fastening or
restraint correctness. Child and child-seat values are explicitly candidate or
unknown states derived from apparent size, configured region hints, object
evidence, and temporal persistence. A relative child candidate is emitted only
when three seats have recent person-box evidence and one seat is at least 25%
below the peer median for three of five votes. Six consecutive contradictory
votes clear that candidate; short gaps do not. This state is stored as a
reason/event so current and historical views agree. It has no confidence score
because box-size ratios are not calibrated probabilities and do not establish
age. It is not emitted when fewer than three seats provide comparable evidence.
Automatic seat zones still require camera review: the supplied saved four-zone
map assigned only one of three person boxes, while a fresh probe produced a
three-seat map and a Seat 2 candidate on the supplied replay.

Face/DMS is an opt-in preview branch for drowsiness/distraction/phone-use
candidates. Appearance M/F output is experimental display-only metadata and is
not identity, age, or safety evidence. Airbag control, pretensioner control,
breathing, heartbeat, exact age, and production driver-attention claims are
unsupported.

