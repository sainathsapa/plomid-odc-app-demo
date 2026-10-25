# Tracking and seat association

Implementation: `odc/tracking.py`, `odc/seats.py`, `odc/layouts.py`.

Short-lived track IDs are session-local. Association scores bbox center-in-zone
and bbox/polygon overlap. Track-first association selects the best seat only
when evidence clears the ambiguity margin. Seat-first association builds a
candidate pool per zone and claims a track once, leaving conflicts unassigned.

No detection outside configured geometry is forced into the nearest seat.
Ambiguous and uncalibrated results use `seat_id=None` and remain unknown.
Stable canonical seat IDs (`seat_1`, etc.) are carried into state, events,
PLOMID, APIs, and reports; visual display labels are a separate boundary.

