# ODC SQL capabilities (verified 2026-10-13, PLOMID 0.1.0 release binary)

Evidence: `scripts/smoke_plomid.py` — 10/10 PASS on
`127.0.0.1:55433`, plus source refs. Re-run before trusting after any
PLOMID upgrade. Never infer from PG compatibility.

## Verified (smoke probe)
| Capability | Probe | Evidence |
|---|---|---|
| Connect PG wire v3, psycopg3 | server_version | `SELECT version()` → PLOMID 0.1.0 |
| PK / UNIQUE / NOT NULL / DEFAULT / CHECK | ddl_core_types | tables created, defaults applied |
| NUMERIC / BOOLEAN / TIMESTAMPTZ / JSONB / TEXT[] | ddl_core_types | incl. `DEFAULT now()` |
| FK enforced (valid+NULL pass, invalid rejected, atomic) | fk_enforced | `row.rs:520`, `dml.rs:440,743`, test `fk_and_grouping.rs:68` |
| ON CONFLICT DO NOTHING / DO UPDATE + RETURNING | upsert_returning | idempotent writes = ODC event path |
| JSONB `->>` + casts | jsonb_ops | detail columns |
| VECTOR(FLOAT32,n) literal + l2/cosine/dot + `<->` Top-K + HNSW | vector_store_index | `VECTOR_FULL_QUALIFICATION.sql`, `ddl.rs:1480`, auto-HNSW on CREATE TABLE |
| Vector dim-mismatch rejected | vector_store_index | fail-closed inserts |
| BEGIN/COMMIT/ROLLBACK/SAVEPOINT | transactions | |
| information_schema FK introspection | catalog_introspection | dossier + console schema views |
| EXPLAIN plans | explain | 8-line plan for vector Top-K |
| TLS declined | tls_status | `sslmode=require` → OperationalError |

## Verified from source/tests (not yet smoke-covered)
- Window functions, CTEs incl. RECURSIVE, joins incl. lateral, GROUPING SETS/ROLLUP/CUBE, DISTINCT ON, FILTER (`PLOMID_FULL_PRODUCTION_QUALIFICATION.sql`, `compatibility.mdx:12`).
- `COPY STDIN/STDOUT` wire only; file COPY rejected.
- B-tree + ART indexes; unique/PK durable authority (`indexes/overview.mdx`).
- MVCC snapshot + conflict lanes (`transactions/behavior-contracts.mdx`).
- Vector methods: FLAT/HNSW/IVF_FLAT/IVF_SQ/IVF_PQ/DISKANN/SPARSE_INVERTED/BINARY; rejected: UNIQUE vector, composite vector, B-tree on vector (`ddl.rs:1492-1572`, `VECTOR_FULL_QUALIFICATION.sql:164-170`).
- Vector params text-only; binary results supported (`compatibility.mdx:26`).
- `plomid_vector_stats()`, `plomid_vector_indexes()`, `vector.ef_search`, REINDEX.

## Partial / dialect gotchas (ODC must code around)
- Bare `float` alias missing → use `double precision`/`float8`/`real` (D8).
- Inline column-level `REFERENCES` rejected → table-level named
  `FOREIGN KEY` constraints only (F1).
- Multi-statement `execute()` does not apply whole files → single-statement
  runner with comment stripping (F2).
- Bare `version` column binds to `version()` — never name a column `version` (F3).
- `LIMIT/OFFSET` integer literal or cast only, never params (F4).
- `UPDATE…FROM` / `COPY` unsupported inside explicit txns; `LIMIT/OFFSET` integer literal or cast only.
- `AND/OR` use is-true checks (`NULL AND TRUE → false`, not strict 3VL).
- `COUNT(DISTINCT)` dedups by `to_sql_text` (`1` ≡ `1.0`).
- `ILIKE` acts as `LIKE`; `FOR UPDATE` accepted, ignored.
- No partial indexes; no planner cost choice; single-writer engine → pool at app, keep txns short, retry whole txn on Conflict (`build-application.mdx:56-61`).

## Unsupported (do not design on)
- FK CASCADE (parsed, not enforced — RESTRICT semantics only).
- Foreign-key enforcement claimed absent in stale docs — actually enforced (D7).
- TLS transport (declined), replication/backups, multi-region/distributed.
- Stored-procedure execution, cursors, file COPY, continuous aggregates.
- No compatibility percentage claimed; wire+OID similarity ≠ semantics.

## Connection contract for ODC (Phase 2 detail)
- Driver: psycopg3 (+pool), parameterized `$s/%s` only, never interpolation.
- Defaults observed: max_connections 100, statement_timeout 30 s, idle-in-txn 60 s.
- ODC repository owns timeouts/retries/idempotency (`ON CONFLICT`), never exposes connections to perception code.
- ODC tables are namespaced under `plomid_odc`. Startup creates the schema,
  applies versioned migrations, and idempotently copies legacy `public.odc_*`
  rows into the namespace. Continuous video remains outside SQL; structured
  sessions, states, events, metrics, layouts, probes, and analytics are stored.

## Frame and assistant persistence added in ODC migrations V006–V007

Validated against the local PLOMID server using real SQL/PG-wire integration
tests. `odc_frame_observation` stores one JSONB payload per `(session_id,
frame_seq)` with `TIMESTAMPTZ`, a composite primary key, and session foreign
key. It contains the seat-row contract, pose landmarks, and specialist
decisions. `odc_assistant_interaction` stores explicit Mistral requests,
responses/status, frame/seat reference, and exact request context as JSONB;
history is queryable by session. Both tables use regular SQL indexes and
`ON CONFLICT` idempotency. Existing raw boxes continue to use
`VECTOR(FLOAT32,4)` in `odc_frame_detection`; inference metrics retain
timestamped rows for time-series queries.

PloMID dialect constraints are applied: SQL is sent one statement per call,
`LIMIT` is a bounded integer literal (not a bind parameter), and migrations
avoid semicolons in line comments because the migration runner splits on
semicolons. No frame pixels or video bytes are stored in SQL.
