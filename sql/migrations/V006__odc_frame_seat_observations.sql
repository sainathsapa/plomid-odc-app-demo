-- ODC migration V006: durable frame-level seat outputs for API and UI history.
-- Image pixels stay outside PLOMID. This stores structured, non-image evidence.

CREATE TABLE odc_frame_observation (
    session_id TEXT NOT NULL,
    frame_seq BIGINT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (session_id, frame_seq),
    CONSTRAINT frame_observation_session_fk FOREIGN KEY (session_id)
        REFERENCES odc_session (session_id)
);

CREATE INDEX odc_frame_observation_session_idx ON odc_frame_observation (session_id, frame_seq);
