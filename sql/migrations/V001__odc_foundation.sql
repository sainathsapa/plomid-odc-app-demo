-- ODC migration V001: session, model, camera, seat state, events.
-- Verified PLOMID 0.1.0 syntax only: no bare float, no inline REFERENCES.
-- NOTE: odc_schema_version is owned by the migrate.py bootstrap, not by V001.

CREATE TABLE odc_session (
    session_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    mode TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    build_version TEXT NOT NULL,
    config_version TEXT NOT NULL DEFAULT 'unset',
    status TEXT NOT NULL DEFAULT 'OPEN'
);

CREATE TABLE odc_model_artifact (
    model_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    checkpoint_sha256 TEXT NOT NULL,
    source_url TEXT NOT NULL,
    code_revision TEXT NOT NULL,
    weights_license TEXT NOT NULL,
    dataset_terms TEXT NOT NULL,
    approved_for_commercial_use BOOLEAN NOT NULL DEFAULT FALSE,
    input_profile TEXT NOT NULL DEFAULT '',
    runtime_profile TEXT NOT NULL DEFAULT ''
);

CREATE TABLE odc_camera_status (
    camera_id TEXT PRIMARY KEY,
    session_id TEXT,
    status TEXT NOT NULL,
    fps_observed DOUBLE PRECISION,
    failure_code TEXT,
    checked_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE odc_seat_state_current (
    session_id TEXT NOT NULL,
    seat_id TEXT NOT NULL,
    presence TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    visibility TEXT NOT NULL,
    reason_codes TEXT[] NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL,
    last_positive_at TIMESTAMPTZ,
    model_set_version TEXT NOT NULL DEFAULT 'unset',
    config_version TEXT NOT NULL DEFAULT 'unset',
    PRIMARY KEY (session_id, seat_id),
    CONSTRAINT seat_state_session_fk FOREIGN KEY (session_id)
        REFERENCES odc_session (session_id)
);

CREATE TABLE odc_event (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'INFO',
    seat_id TEXT,
    track_id TEXT,
    occurred_at TIMESTAMPTZ NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    prior_state TEXT,
    new_state TEXT,
    reason_codes TEXT[] NOT NULL DEFAULT '{}',
    evidence_summary TEXT NOT NULL DEFAULT '',
    config_version TEXT NOT NULL DEFAULT 'unset',
    model_set_version TEXT NOT NULL DEFAULT 'unset',
    CONSTRAINT event_session_fk FOREIGN KEY (session_id)
        REFERENCES odc_session (session_id)
);

CREATE INDEX odc_event_session_idx ON odc_event (session_id);
