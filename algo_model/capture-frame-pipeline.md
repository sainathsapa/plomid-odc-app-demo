# Capture and frame pipeline

Implementation: `odc/capture.py`, `odc/service.py`.

`FileReplaySource` uses OpenCV `VideoCapture`, preserves source timestamps for
replay, emits monotonically increasing frame sequence numbers, and reports
decode health. `LiveCameraSource` accepts a device index or stream target and
reconnects with bounded exponential backoff.

Each `Frame` contains source ID, sequence, UTC/evidence timestamp, dimensions,
mode (`LIVE` or `REPLAY`) and pixels. The service previews every captured frame
but samples inference according to `sample_every_n`; captured, processed and
dropped counts are kept separately.

Quality checks cover dark, bright, blur and frozen-frame signals. Decode or
camera failure is surfaced as unhealthy/degraded; it never becomes an empty
seat result.

