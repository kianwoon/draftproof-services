-- Progress fields for rewrite jobs.
ALTER TABLE rewrite_jobs
  ADD COLUMN IF NOT EXISTS progress_percent INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS progress_message TEXT;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'rewrite_jobs_progress_percent_range'
  ) THEN
    ALTER TABLE rewrite_jobs
      ADD CONSTRAINT rewrite_jobs_progress_percent_range
      CHECK (progress_percent >= 0 AND progress_percent <= 100);
  END IF;
END $$;
