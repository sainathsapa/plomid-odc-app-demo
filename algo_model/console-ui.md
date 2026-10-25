# Console data flow and recommended UX

The live workspace receives `/api/live/state` snapshots through SSE-triggered
polling. Seat buttons, chips, map zones, tables, alerts, and the frame recap
all use the same canonical `seat_id`. Selecting a seat now highlights the
button, schematic zone, calibrated overlay zone, and table row while opening
the seat evidence inspector.

Recommended next UI additions:

1. Add a visible “Transient recap / durable SQL evidence” badge beside the
   frame history so operators understand restart behaviour.
2. Add a session replay route that loads PLOMID events/states by session and
   uses the source video fingerprint to validate the requested media.
3. Add a compact persistence panel showing last SQL write, queue depth, and
   whether each displayed value is SQL-backed or transient.
4. Add “calibration required” on standard geometry until camera-specific zones
   have been validated.

