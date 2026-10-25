-- ODC migration V005: per-frame detection evidence.
-- bbox is a normalized four-dimensional vector [x_min,y_min,x_max,y_max].
-- It is geometry evidence, not an identity embedding.

CREATE TABLE odc_frame_detection (
    detection_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    frame_seq BIGINT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    class_label TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    bbox VECTOR(FLOAT32, 4) NOT NULL,
    seat_id TEXT,
    track_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    CONSTRAINT frame_detection_session_fk FOREIGN KEY (session_id)
        REFERENCES odc_session (session_id)
);
