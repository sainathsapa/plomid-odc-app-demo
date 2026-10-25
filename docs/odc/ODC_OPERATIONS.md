# ODC operations (v1.1 — host-DB era, 2026-10-14)

## Install (new environment, ~10 min)
1. Python 3.12, `python3 -m venv .venv`, `pip install -r <pins>` — full pin
   list in `requirements.lock` (dev-mac wheels; Docker resolves linux).
2. Start PLOMID **on the host** (never containerized):
   `./../plomid/target/release/plomid-server --data /tmp/plomid-data --host 127.0.0.1 --port 5432 --username plomid --password plomid`.
3. `ODC_DSN` is optional for the local default; set
   `export ODC_DSN="postgresql://plomid:plomid@127.0.0.1:5432/plomid"` when
   using another endpoint.
4. Model/assets (or run `setup-assets.bat` on Windows, which does all three):
   - `.venv/bin/python scripts/fetch_models.py` — downloads + sha256-verifies
     every HF snapshot manifest in `models/manifests/` (fails closed on mismatch).
   - `.venv/bin/python scripts/export_onnx.py` — exports RT-DETRv2-S to
     `models/rtdetrv2-s-coco/onnx/model.onnx` (required by the runtime).
   - `.venv/bin/python scripts/prepare_assets.py` — fetches the MediaPipe pose +
     face task bundles and the eight clips pinned in `footage/README.md`, then
     verifies each clip decodes. `--check` reports without downloading.
5. `odc serve` automatically creates the `plomid_odc` schema, applies all
   migrations, and copies legacy `public.odc_*` rows idempotently. `bootstrap`
   remains available as a diagnostic command.
6. `.venv/bin/python -m odc.cli selftest` — readiness bundle, exit 0 = go.

## Docker / compose
- `docker build -t plomid-odc .` — runtime image (no weights baked).
- `docker compose up --build` — console on :8080 **against your host
  PLOMID** (`host.docker.internal:5432`; start it first). Footage/models/
  configs mount read-only; weights fetched before compose.

## Run
- Replay/demo: `odc serve` (automatically provisions local PLOMID on startup).
- Eval: `odc validate --split dev|holdout --out reports/<name>`.
- Profile: `odc profile`; providers: `odc providers`; dossier: console/API.
- Purge: `odc purge --dir reports --older-than-days 30`.

## Health, recovery, rollback
- Health: `odc health [--probe-db]`; readiness: `odc selftest`.
- Recovery: camera/model/SQL failures degrade visibly (D22); restart is
  clean (2-session test); retry budget 3 then surface.
- Migrations are additive-only (no down-migrations by design).
  Upgrade: new image + `odc bootstrap` (V1→V2 exercised in tests).
  Rollback: stop → restore data-dir snapshot → previous image → `odc
  selftest`. Snapshot procedure: stop server, copy data dir, start.
  Backup/restore roundtrip drilled 2026-10-13 (see verification log).
- Rollback never deletes: snapshots retained per retention policy.

## Compatibility matrix (verified unless noted)
| Layer | Status |
|---|---|
| OS/arch | macOS arm64 verified; linux x86_64 via Docker (expected) |
| Python | 3.12 |
| Runtime/provider | onnxruntime CPU verified; CoreML rejected w/ evidence; TRT/OpenVINO/QNN untested (matrix entries) |
| PLOMID | 0.1.0 release binary (beta.2 banner); loopback; SCRAM; no TLS in this build |
| Models | rtdetrv2-s-coco (sha-pinned, parity-gated); grounding eval-only; mediapipe-lite REVIEW_REQUIRED |
| Camera | file replay verified (8 clips); live camera implemented, hardware-untested |

## Troubleshooting
| Symptom | Cause → fix |
|---|---|
| Startup provisioning fails | Verify PLOMID is listening on `127.0.0.1:5432` with user `plomid` and password `plomid`, or set `ODC_DSN` |
| UI shows DISCONNECTED | service not started / crashed — check serve logs |
| All seats STALE instantly | clock mismatch — staleness uses evidence clock; check replay anchor |
| ` FOREIGN KEY` errors on first write | session not started — Service starts it; manual flows must start_session first |
| Slow inference | see laptop-m2pro-cpu profile (threads=8); measure before changing |
| Corrupt/partial downloads | fetch/exports verify sha; re-run fetch |
