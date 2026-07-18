-- Defence responses — the "defence-readiness" reflective-question answers a student
-- submits against a flagged passage, plus the LLM judgment of that answer.
--
-- Free-capped v1 (per product decision): this table has NO credit_ledger /
-- credit_reservations integration. Task 6 layers a simple env-configured attempt
-- cap (DRAFTPROOF_DEFENCE_MAX_ATTEMPTS) at the API layer, not real credit metering.
--
-- One row per (scan, question, attempt): a student may retry a question up to the
-- attempt cap, each retry landing as its own row (attempt = 1, 2, ...) so the
-- judging history is preserved rather than overwritten.
--
-- status lifecycle: pending (row created, judging not yet run) -> judged (judgment
-- populated) or failed (judge_defence_answer returned None / judging errored).

CREATE TABLE IF NOT EXISTS defence_responses (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id        UUID NOT NULL REFERENCES scan_jobs(id) ON DELETE CASCADE,
    user_id        UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    question_index INT NOT NULL,
    dimension      TEXT,
    question       TEXT NOT NULL,
    anchor_quote   TEXT,
    answer_text    TEXT NOT NULL,
    attempt        INT NOT NULL DEFAULT 1,
    status         TEXT NOT NULL DEFAULT 'pending',
    judgment       JSONB,
    judge_model    TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    judged_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_defence_responses_scan_id ON defence_responses(scan_id);
CREATE INDEX IF NOT EXISTS idx_defence_responses_user_id ON defence_responses(user_id);
