# PLOMID persistence and frame recap

Implementation: `odc/persistence.py`, `odc/seat_store.py`, migrations
`sql/migrations/V001__...` through `V004__...`, and `ServiceHub.history`.

The SQL boundary persists:

- session/source/config/model metadata;
- current per-seat state and occupant attribute rows;
- state events with evidence timestamp, reason codes, frame sequence,
  configuration version, and model-set version;
- health and processing metrics;
- versioned seat layouts and video fingerprints;
- bounded validation/probe/analytics records.
- every processed inference frame's detection class, confidence, normalized
  bounding box, optional seat association, and metadata in
  `plomid_odc.odc_frame_detection`; the `bbox` column is a PLOMID
  `VECTOR(FLOAT32,4)` containing `[x_min,y_min,x_max,y_max]`.

The active RT-DETRv2/pose/face pipeline does **not** produce a semantic vector
embedding. It does persist the real detection box as a four-value PLOMID
vector, plus class and confidence. Face identity embeddings are intentionally
prohibited; the DMS face branch stores no identity vector.

The live service keeps the last configured number of JPEG previews in a bounded
in-memory ring. `/api/frames/history` lists those previews and `/frame.jpeg`
can show the selected frame while processing continues. This is a session
recap cache, not durable PLOMID storage. A restart clears it by design.

Each processed inference frame now also writes a structured row to
`odc_inference_metric` through `Repository.write_metric_batch`, including
session, model, provider, capture timestamp, latency, and dropped-frame
count. State-transition events additionally carry the exact evidence frame
sequence when available.

Therefore “everything” is not currently stored in PLOMID: structured truth is,
continuous raw frames are not. Historical state/event recap is durable through
PLOMID; exact image replay after restart requires the original video source or
an explicitly approved evidence-retention feature.

