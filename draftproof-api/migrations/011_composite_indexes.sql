-- 011: Composite indexes for common query patterns
-- (scan_id, status) on rewrite_jobs: covers API polling for a scan's active/completed rewrite.
-- The partial unique index from 008 only enforces one-active-per-scan; it doesn't serve general lookups.
CREATE INDEX IF NOT EXISTS idx_rewrite_jobs_scan_status
    ON rewrite_jobs(scan_id, status);

-- (user_id, created_at DESC) on scan_jobs: covers paginated user history / dashboard queries.
CREATE INDEX IF NOT EXISTS idx_scan_jobs_user_created_at
    ON scan_jobs(user_id, created_at DESC);
