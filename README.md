# PLOMID ODC — Occupant & Cabin State Platform

Edge cabin-occupant intelligence: camera observations → temporal seat
states → PLOMID SQL truth → engineering console. Pretrained inference only,
uncertainty explicit, no training in this program.

## 60-second start (demo)

Prerequisite: PLOMID running on the HOST (never containerized):
```sh
./../plomid/target/release/plomid-server --data /tmp/plomid-data \
  --host 127.0.0.1 --port 5432 --username plomid --password secret
```

Local run:
```sh
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.lock
./.venv/bin/python scripts/fetch_models.py   # downloads + sha-verifies
./.venv/bin/python scripts/export_onnx.py    # RT-DETRv2-S ONNX runtime file
./.venv/bin/python scripts/prepare_assets.py # MediaPipe tasks + footage clips
export ODC_DSN="postgresql://plomid:plomid@127.0.0.1:5432/plomid"  # optional local override
.venv/bin/python -m odc.cli bootstrap      # optional: serve also provisions automatically
.venv/bin/python -m odc.cli selftest       # readiness bundle, exit 0 = go
.venv/bin/python -m odc.cli serve          # console on :8080 (idles until YOU press Start)
.venv/bin/python -m odc.cli serve --auto-start  # ...or start the default session immediately
```

Windows one-command setup:
```bat
setup-assets.bat   :: once: checkpoints, ONNX export, MediaPipe bundles, footage
start.bat          :: uses PLOMID on :5432 if already running, otherwise starts
                   :: ..\plomid\target\release\plomid-server.exe, then serves
```

Or containerized console (same host DB):
```sh
docker compose up --build   # http://localhost:8080
```

On `serve` startup, PLOMID automatically creates/updates the dedicated
`plomid_odc` schema and migrates legacy public ODC rows without a manual
bootstrap step. The local default is `plomid`/`plomid` on `127.0.0.1:5432`;
set `ODC_DSN` for another supported endpoint.

Runtime structured state is stored in PLOMID, not in application JSON/CSV
sidecars. This includes current seat/occupant state, frame-level seat rows,
age/gender and confidence semantics, belt/restraint/posture evidence, pose
keypoints, specialist decisions, raw detection boxes, metrics, layouts, and
assistant interactions with the exact context supplied to local Ollama/Mistral.
`/api/live/state` is `odc.live.v1`; persisted seat rows expose
`odc.seat-observation.v1`; `/api/assistant/history` exposes
`odc.assistant-history.v1`. The Data model page (`/relations`) and
`/api/schema` (`odc.schema-model.v1`) show the live foreign-key catalog:
twelve enforced relations plus the documented soft links that cannot be hard
constraints. User videos remain files under `footage/`; frame
previews are bounded in-memory JPEGs and are not inserted into PLOMID. Model
weights and code-owned configuration remain local runtime assets. Evaluation
scripts may explicitly write developer reports when invoked with an output
path; those files are not the application's durable store.

The first-run console uses `configs/seats/standard-5-seat-rhd.yaml`: Row 1
contains Seat 1 (driver) and Seat 2 (front passenger); Row 2 contains Seats
3–5. Standard two- and seven-seat layouts are available from the layout
selector. `seat_1`…`seat_7` are stable SQL/event keys; “Seat 1”…”Seat 7” are
operator labels. RHD is explicit metadata and can be replaced by a versioned
orientation profile; visual left/right is never treated as driver identity.

Standard geometry is a starting profile, not camera calibration. For a real
installed camera, trace/validate seat zones on the Seat map page before relying
on seat association. A replay session is labelled replay; it is not presented
as a live camera.

Probe any video without a database:
```sh
.venv/bin/python -m odc.cli analyze THEIRFILE.mp4 [--layout NAME] [--max-frames 60]
```

## Commands

`replay` · `detect` · `analyze` · `validate --split dev|holdout` · `profile` ·
`providers` · `health` · `selftest` · `bootstrap` · `serve` · `purge`

## Configure (YAML file > env > `.env` > defaults)

```sh
cp .env.example .env   # then edit; .env is gitignored, never committed
```

`.env` keys: `ODC_DSN`, `ODC_MODE`, `ODC_CAMERA__PATH`, `ODC_CAMERA__TARGET_FPS`,
`ODC_DETECTION__THRESHOLD`, … (nested `__` form). A YAML file passed via
`--config` overrides everything except secrets handling.

Optional local assistant: install Ollama, run `ollama pull mistral`, and keep
the default `ODC_ASSISTANT__URL=http://127.0.0.1:11434`. Mistral is called only
when an operator explicitly invokes `/api/assistant/ask`; it never runs in the
frame loop and never writes AI-generated claims into PLOMID.

## Honesty contract

Every state carries confidence + visibility + provenance. No detection never
becomes empty — weak evidence yields `UNKNOWN`/`STALE`/`UNSUPPORTED`.
Reports are measured on named hardware, never pledged. Prototype: not
safety-certified, never claims to be.

## Docs

`PLOMID_ODC_MASTER_BUILD_PLAN.md` (locked v1.0) · `docs/odc/` (requirements,
architecture, model selection + licenses, ontology, SQL capabilities,
validation plan, risks, decisions, security/privacy, operations, build
status) · `footage/README.md` (eval media manifest).



cd plomid && $env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH; cargo build --release



sudo apt-get update && sudo apt-get install -y build-essential
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
cargo build --release --locked -p plomid-server


