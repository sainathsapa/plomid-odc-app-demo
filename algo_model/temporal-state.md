# Temporal state reasoning

Implementation: `odc/state.py`, `odc/service.py`.

`SeatStateEngine` combines per-frame evidence with dwell, smoothing,
staleness, visibility, and transition rules. It emits current seat snapshots
and state-change events. Positive evidence can produce `OCCUPIED` or
`LIKELY_OCCUPIED`; weak, missing, stale, occluded, or unhealthy evidence stays
`UNKNOWN`/`STALE` rather than silently becoming empty. `EMPTY_CANDIDATE` is a
supported negative candidate, not a safety-certified vacancy assertion.

The evidence clock is the frame timestamp, including replay timestamps, so
events remain aligned with a selected frame during recap.

