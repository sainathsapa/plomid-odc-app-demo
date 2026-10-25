# ODC security & privacy (v1, locked 2026-10-13)

Prototype status: engineering demonstrator. NOT ISO 26262 / ASPICE /
ISO/SAE 21434 / regulatory certified. No certification claim is made
anywhere in code, UI, or dossier (verified by grep in CI-gated review).

## Data-flow map (trust boundaries marked *)
```
cabin camera ──*──> ODC process ──*──> PLOMID (loopback SQL, SCRAM)
     │                  │                    │
   frames             logs                 console (loopback HTTP)
 (transient)      (no frames/secrets)      structured history APIs
     │                                       │
   footage/                           states + pose + chat context
 (video files)                          in SQL / JSONB / VECTOR
```

## Threats → mitigations (evidence in parens)
| Threat | Mitigation |
|---|---|
| Credential leak via logs/errors | `odc/redact.py` scrubbers; DSN env-only; config-check excludes secrets (tests) |
| SQL injection | parameterized statements only; f-string-SQL allowlist test (2 reviewed LIMIT sites); no user input in identifiers |
| Malicious/oversize payloads | Pydantic bounds (confidence 0..1, normalized boxes, evidence ≤2000 chars); oversize rejected, not truncated |
| Stale/compromised deps | pip-licenses scan 2026-10-13 (zero GPL-family) + ban test; pip-audit zero vulns; lockfile frozen |
| Untrusted model weights | sha-pinned manifests; commercial flags reviewed; grounding eval-only |
| Raw media retention | video uploads remain local files; frame JPEG ring is bounded in memory; no video/JPEG bytes in PLOMID; `odc purge` applies only to explicit generated-artifact directories |
| DB outage → silent loss | degraded banners; bounded queues; retry budget + surfacing (tests) |
| Camera/model failure masked | distinct UNHEALTHY/MODEL_UNAVAILABLE states; service runs degraded, never fake (tests) |
| Remote exposure | serve binds loopback by default; non-loopback warns (no TLS in this PLOMID build — D9) |
| Disk pressure | bounded queues/buffers everywhere; purge workflow; residual: no quota enforcement yet |

## Privacy (GSR closed-loop compatible)
Edge-local inference; no cloud calls; Mistral runs through the configured local
Ollama endpoint. No face recognition or identity embeddings. PLOMID stores
structured seat/frame outputs, pose keypoints, specialist evidence, and
assistant prompts/replies with the exact context used for the answer. Raw
frames/videos remain outside PLOMID. Assistant history currently has no
automatic retention expiry; protect the database and define a retention policy
before deploying with real occupant footage or user prompts. There are no
runtime CSV/JSON sidecar files; offline R&D tools can explicitly generate
reports when requested.

## Recovery matrix (tested)
Camera disconnect / corrupt frames / slow consumer / model missing /
corrupt weights / DB outage+return / bad config / restart / shutdown —
see tests/unit/test_security.py, test_capture.py, test_service.py and the
verification log. Disk-full handling is documented residual, not tested.
