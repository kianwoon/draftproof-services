-- 005: Add rewrite_jobs table
CREATE TABLE rewrite_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID NOT NULL REFERENCES scan_jobs(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX ix_rewrite_jobs_scan_id ON rewrite_jobs(scan_id);
CREATE INDEX ix_rewrite_jobs_user_id ON rewrite_jobs(user_id);
CREATE INDEX ix_rewrite_jobs_status ON rewrite_jobs(status);
