-- ODC migration V003: PLOMID-backed seat store.
-- Videos stay on disk (footage is the only filesystem dependency) while
-- every seat map and calibration and probe result and analytics snapshot
-- lives in these tables. No sidecar directories are used anywhere.
-- Plain DDL only (verified PLOMID 0.1.0 syntax subset with no bare float
-- and no inline REFERENCES, JSONB defaults match V002 style).

CREATE TABLE odc_seat_layout (
    layout_name TEXT PRIMARY KEY,
    camera_profile TEXT NOT NULL DEFAULT '',
    version TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT 'hand',
    seats JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE odc_video_asset (
    video_key TEXT PRIMARY KEY,
    video_path TEXT NOT NULL DEFAULT '',
    video_sha TEXT NOT NULL DEFAULT '',
    size_bytes BIGINT NOT NULL DEFAULT 0,
    mtime BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX odc_video_asset_path_idx ON odc_video_asset (video_path);

CREATE TABLE odc_video_seatmap (
    video_key TEXT PRIMARY KEY,
    video_sha TEXT NOT NULL DEFAULT '',
    camera_id TEXT NOT NULL DEFAULT '',
    calibration_profile TEXT NOT NULL DEFAULT '',
    layout_version TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT 'auto',
    layout JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE odc_probe_cache (
    video_key TEXT PRIMARY KEY,
    video_sha TEXT NOT NULL DEFAULT '',
    frames_probed INTEGER NOT NULL DEFAULT 0,
    persons_seen INTEGER NOT NULL DEFAULT 0,
    peaks INTEGER NOT NULL DEFAULT 0,
    reason TEXT NOT NULL DEFAULT '',
    layout JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE odc_analytics_snapshot (
    cache_key TEXT PRIMARY KEY,
    video_key TEXT NOT NULL DEFAULT '',
    layout_hash TEXT NOT NULL DEFAULT '',
    config_hash TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
