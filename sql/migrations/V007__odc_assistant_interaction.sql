-- ODC migration V007: durable, user-requested local assistant history.
-- Prompts, replies, and the exact structured context stay in PLOMID JSONB.

CREATE TABLE odc_assistant_interaction (
    interaction_id TEXT PRIMARY KEY,
    session_id TEXT,
    occurred_at TIMESTAMPTZ NOT NULL,
    frame_seq BIGINT,
    seat_id TEXT,
    provider TEXT NOT NULL,
    model TEXT,
    status TEXT NOT NULL,
    prompt TEXT NOT NULL,
    answer TEXT,
    context JSONB NOT NULL DEFAULT '{}',
    CONSTRAINT assistant_session_fk FOREIGN KEY (session_id)
        REFERENCES odc_session (session_id)
);

CREATE INDEX odc_assistant_session_idx ON odc_assistant_interaction (session_id, occurred_at);
