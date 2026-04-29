-- Scan jobs table — bridges API and Celery worker
CREATE TABLE IF NOT EXISTS scan_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    input_text_hash TEXT NOT NULL,
    word_count      INTEGER NOT NULL DEFAULT 0,
    scan_type       TEXT NOT NULL DEFAULT 'scan',
    status          TEXT NOT NULL DEFAULT 'pending',
    tier            TEXT,
    finding_count   INTEGER,
    report_urls     JSONB DEFAULT '{}',
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_scan_jobs_user_id ON scan_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_scan_jobs_status ON scan_jobs(status);
