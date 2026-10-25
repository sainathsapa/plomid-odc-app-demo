-- ODC migration V002: occupant state, metrics, validation.
-- Upgrade path V001 -> V002 is covered by integration tests.

CREATE TABLE odc_occupant_state_current (
    session_id TEXT NOT NULL,
    track_id TEXT NOT NULL,
    seat_id TEXT,
    presence TEXT,
    presence_conf DOUBLE PRECISION CHECK (presence_conf IS NULL OR (presence_conf >= 0 AND presence_conf <= 1)),
    apparent_size TEXT,
    apparent_size_conf DOUBLE PRECISION CHECK (apparent_size_conf IS NULL OR (apparent_size_conf >= 0 AND apparent_size_conf <= 1)),
    posture TEXT,
    posture_conf DOUBLE PRECISION CHECK (posture_conf IS NULL OR (posture_conf >= 0 AND posture_conf <= 1)),
    seatbelt TEXT,
    seatbelt_conf DOUBLE PRECISION CHECK (seatbelt_conf IS NULL OR (seatbelt_conf >= 0 AND seatbelt_conf <= 1)),
    restraint TEXT,
    restraint_conf DOUBLE PRECISION CHECK (restraint_conf IS NULL OR (restraint_conf >= 0 AND restraint_conf <= 1)),
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (session_id, track_id),
    CONSTRAINT occupant_session_fk FOREIGN KEY (session_id)
        REFERENCES odc_session (session_id)
);

CREATE TABLE odc_inference_metric (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    captured_at TIMESTAMPTZ NOT NULL,
    inference_ms DOUBLE PRECISION NOT NULL,
    queue_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    dropped_frames BIGINT NOT NULL DEFAULT 0,
    memory_bytes BIGINT
);

CREATE TABLE odc_validation_run (
    run_id TEXT PRIMARY KEY,
    build_version TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    hardware_profile TEXT NOT NULL DEFAULT '',
    model_manifest JSONB NOT NULL DEFAULT '{}',
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'RUNNING'
);

CREATE TABLE odc_validation_result (
    run_id TEXT NOT NULL,
    task TEXT NOT NULL,
    label TEXT NOT NULL,
    scenario TEXT NOT NULL DEFAULT '',
    precision DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    f1 DOUBLE PRECISION,
    fp_count BIGINT,
    fn_count BIGINT,
    latency_p50_ms DOUBLE PRECISION,
    latency_p95_ms DOUBLE PRECISION,
    notes TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, task, label, scenario),
    CONSTRAINT validation_run_fk FOREIGN KEY (run_id)
        REFERENCES odc_validation_run (run_id)
);

CREATE INDEX odc_metric_session_idx ON odc_inference_metric (session_id);
CREATE INDEX odc_event_type_idx ON odc_event (event_type);
