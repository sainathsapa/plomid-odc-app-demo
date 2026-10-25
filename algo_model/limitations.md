# Current limitations and honest status

- Standard layout geometry is a starting profile; installed-camera calibration
  is required for reliable seat assignment.
- Generic COCO detection does not establish adult versus child.
- Child-seat, belt, and posture outputs are candidates with explicit reasons.
- Missing/occluded/failed frames do not prove an empty seat.
- Raw video is not stored in PLOMID and the in-memory frame recap is lost on
  restart.
- No airbag, pretensioner, OEM buckle-bus, or safety-critical actuation path
  exists.
- Production qualification, functional safety, cybersecurity, and OEM target
  hardware validation remain external work.

