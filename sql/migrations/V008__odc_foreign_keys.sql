-- ODC migration V008: complete foreign-key coverage + relation catalog.
--
-- PLOMID accepts FOREIGN KEY only inside CREATE TABLE. ALTER TABLE ADD
-- CONSTRAINT rejects FOREIGN, so each child table below is rebuilt once
-- with its new constraint and its rows copied forward.
--
-- PLOMID also discards DML when the same transaction contains DDL. The
-- migration runner therefore commits one statement per transaction; a
-- rebuild is three DDL statements around one INSERT ... SELECT.
--
-- Data safety: the old table is renamed to <table>_old only after the new
-- table is fully populated, so a failure never drops the only copy. The
-- final order is: populate staging, rename old, rename staging, drop old.
-- Re-running V008 after a partial failure may need the <table>_old table
-- removed by hand once its rows are confirmed in the new table.
--
-- odc_inference_metric.id is a surrogate with no dependents. It is copied
-- without the id so the new BIGSERIAL starts cleanly, because PLOMID has
-- no ALTER SEQUENCE.
--
-- model_id stays a documented soft relation: telemetry may be written
-- before a model artifact row exists, and adding a hard FK would turn a
-- missing provenance row into a lost metric.

DROP TABLE IF EXISTS plomid_odc.odc_camera_status_fk;

CREATE TABLE plomid_odc.odc_camera_status_fk (
    camera_id TEXT PRIMARY KEY,
    session_id TEXT,
    status TEXT NOT NULL,
    fps_observed DOUBLE PRECISION,
    failure_code TEXT,
    checked_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT camera_status_session_fk FOREIGN KEY (session_id)
        REFERENCES plomid_odc.odc_session (session_id)
);

INSERT INTO plomid_odc.odc_camera_status_fk
    (camera_id, session_id, status, fps_observed, failure_code, checked_at)
SELECT camera_id, session_id, status, fps_observed, failure_code, checked_at
FROM plomid_odc.odc_camera_status;

ALTER TABLE plomid_odc.odc_camera_status RENAME TO odc_camera_status_old;

ALTER TABLE plomid_odc.odc_camera_status_fk RENAME TO odc_camera_status;

DROP TABLE plomid_odc.odc_camera_status_old;

DROP TABLE IF EXISTS plomid_odc.odc_inference_metric_fk;

CREATE TABLE plomid_odc.odc_inference_metric_fk (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    captured_at TIMESTAMPTZ NOT NULL,
    inference_ms DOUBLE PRECISION NOT NULL,
    queue_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    dropped_frames BIGINT NOT NULL DEFAULT 0,
    memory_bytes BIGINT,
    CONSTRAINT metric_session_fk FOREIGN KEY (session_id)
        REFERENCES plomid_odc.odc_session (session_id)
);

INSERT INTO plomid_odc.odc_inference_metric_fk
    (session_id, model_id, provider, captured_at, inference_ms,
     queue_ms, dropped_frames, memory_bytes)
SELECT session_id, model_id, provider, captured_at, inference_ms,
       queue_ms, dropped_frames, memory_bytes
FROM plomid_odc.odc_inference_metric;

ALTER TABLE plomid_odc.odc_inference_metric RENAME TO odc_inference_metric_old;

ALTER TABLE plomid_odc.odc_inference_metric_fk RENAME TO odc_inference_metric;

DROP TABLE plomid_odc.odc_inference_metric_old;

CREATE INDEX odc_metric_session_lookup_idx ON plomid_odc.odc_inference_metric (session_id);

DROP TABLE IF EXISTS plomid_odc.odc_video_seatmap_fk;

CREATE TABLE plomid_odc.odc_video_seatmap_fk (
    video_key TEXT PRIMARY KEY,
    video_sha TEXT NOT NULL DEFAULT '',
    camera_id TEXT NOT NULL DEFAULT '',
    calibration_profile TEXT NOT NULL DEFAULT '',
    layout_version TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT 'auto',
    layout JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT video_seatmap_asset_fk FOREIGN KEY (video_key)
        REFERENCES plomid_odc.odc_video_asset (video_key)
);

INSERT INTO plomid_odc.odc_video_seatmap_fk
    (video_key, video_sha, camera_id, calibration_profile, layout_version,
     origin, layout, updated_at)
SELECT video_key, video_sha, camera_id, calibration_profile, layout_version,
       origin, layout, updated_at
FROM plomid_odc.odc_video_seatmap;

ALTER TABLE plomid_odc.odc_video_seatmap RENAME TO odc_video_seatmap_old;

ALTER TABLE plomid_odc.odc_video_seatmap_fk RENAME TO odc_video_seatmap;

DROP TABLE plomid_odc.odc_video_seatmap_old;

DROP TABLE IF EXISTS plomid_odc.odc_probe_cache_fk;

CREATE TABLE plomid_odc.odc_probe_cache_fk (
    video_key TEXT PRIMARY KEY,
    video_sha TEXT NOT NULL DEFAULT '',
    frames_probed INTEGER NOT NULL DEFAULT 0,
    persons_seen INTEGER NOT NULL DEFAULT 0,
    peaks INTEGER NOT NULL DEFAULT 0,
    reason TEXT NOT NULL DEFAULT '',
    layout JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT probe_cache_asset_fk FOREIGN KEY (video_key)
        REFERENCES plomid_odc.odc_video_asset (video_key)
);

INSERT INTO plomid_odc.odc_probe_cache_fk
    (video_key, video_sha, frames_probed, persons_seen, peaks, reason,
     layout, updated_at)
SELECT video_key, video_sha, frames_probed, persons_seen, peaks, reason,
       layout, updated_at
FROM plomid_odc.odc_probe_cache;

ALTER TABLE plomid_odc.odc_probe_cache RENAME TO odc_probe_cache_old;

ALTER TABLE plomid_odc.odc_probe_cache_fk RENAME TO odc_probe_cache;

DROP TABLE plomid_odc.odc_probe_cache_old;

DROP TABLE IF EXISTS plomid_odc.odc_analytics_snapshot_fk;

CREATE TABLE plomid_odc.odc_analytics_snapshot_fk (
    cache_key TEXT PRIMARY KEY,
    video_key TEXT NOT NULL,
    layout_hash TEXT NOT NULL DEFAULT '',
    config_hash TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT analytics_video_asset_fk FOREIGN KEY (video_key)
        REFERENCES plomid_odc.odc_video_asset (video_key)
);

INSERT INTO plomid_odc.odc_analytics_snapshot_fk
    (cache_key, video_key, layout_hash, config_hash, payload, updated_at)
SELECT cache_key, video_key, layout_hash, config_hash, payload, updated_at
FROM plomid_odc.odc_analytics_snapshot;

ALTER TABLE plomid_odc.odc_analytics_snapshot RENAME TO odc_analytics_snapshot_old;

ALTER TABLE plomid_odc.odc_analytics_snapshot_fk RENAME TO odc_analytics_snapshot;

DROP TABLE plomid_odc.odc_analytics_snapshot_old;

-- One row per enforced foreign key. PLOMID stores view expressions lossily:
-- array subscripts and CAST(... AS TEXT) come back as NULL on the next
-- connection, which silently emptied a richer version of this view. This
-- definition uses only plain column references and function calls that
-- survive persistence. Column names come from pg_get_constraintdef() and
-- are parsed by the console. relkind='r' filters stale index rows that
-- PLOMID leaves in pg_class after DROP TABLE.
DROP VIEW IF EXISTS plomid_odc.odc_relation_catalog;

CREATE VIEW plomid_odc.odc_relation_catalog AS
SELECT
    lc.relname AS source_table,
    rc.relname AS target_table,
    con.conname AS constraint_name,
    pg_get_constraintdef(con.oid) AS definition,
    rc2.update_rule AS on_update,
    rc2.delete_rule AS on_delete
FROM pg_constraint con
JOIN pg_class lc ON lc.oid = con.conrelid AND lc.relkind = 'r'
JOIN pg_namespace ln ON ln.oid = lc.relnamespace AND ln.nspname = 'plomid_odc'
JOIN pg_class rc ON rc.oid = con.confrelid AND rc.relkind = 'r'
JOIN pg_namespace rn ON rn.oid = rc.relnamespace AND rn.nspname = 'plomid_odc'
JOIN information_schema.referential_constraints rc2
    ON rc2.constraint_name = con.conname
WHERE con.contype = 'f'
