-- 008: Prevent duplicate active rewrite jobs for the same scan.
WITH duplicate_jobs AS (
  SELECT id
  FROM (
    SELECT
      id,
      row_number() OVER (PARTITION BY scan_id ORDER BY created_at DESC, id DESC) AS rn
    FROM rewrite_jobs
    WHERE status IN ('pending', 'processing', 'retrying')
  ) ranked
  WHERE rn > 1
),
released AS (
  UPDATE credit_reservations cr
  SET status = 'released',
      updated_at = now()
  FROM duplicate_jobs d
  WHERE cr.job_type = 'rewrite'
    AND cr.job_id = d.id
    AND cr.status = 'active'
  RETURNING cr.credit_account_id, cr.tokens_reserved
),
release_totals AS (
  SELECT credit_account_id, sum(tokens_reserved) AS tokens_reserved
  FROM released
  GROUP BY credit_account_id
)
UPDATE credit_accounts ca
SET reserved_tokens = GREATEST(0, ca.reserved_tokens - release_totals.tokens_reserved)
FROM release_totals
WHERE ca.id = release_totals.credit_account_id;

WITH duplicate_jobs AS (
  SELECT id
  FROM (
    SELECT
      id,
      row_number() OVER (PARTITION BY scan_id ORDER BY created_at DESC, id DESC) AS rn
    FROM rewrite_jobs
    WHERE status IN ('pending', 'processing', 'retrying')
  ) ranked
  WHERE rn > 1
)
UPDATE rewrite_jobs
SET status = 'failed',
    error = COALESCE(error, 'Superseded duplicate active rewrite'),
    progress_message = COALESCE(progress_message, 'Rewrite superseded'),
    completed_at = COALESCE(completed_at, now())
WHERE id IN (SELECT id FROM duplicate_jobs);

CREATE UNIQUE INDEX IF NOT EXISTS ux_rewrite_jobs_active_scan
  ON rewrite_jobs(scan_id)
  WHERE status IN ('pending', 'processing', 'retrying');
