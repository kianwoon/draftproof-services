-- Durable lifetime free-scan counter, decoupled from the report-retention purge.
--
-- The free-scan quota used to be enforced by COUNTing scan_jobs rows. The
-- housekeeping job (worker/app/db_cleanup.py::cleanup_old_report_rows) hard
-- deletes scan_jobs older than the retention window (default 3 days), so the
-- count silently dropped and the "5 free scans, ever" limit degraded into
-- "5 free scans per rolling retention window".
--
-- Fix: store the lifetime count on a durable column that cleanup never touches.
-- A per-job boolean acts as a compare-and-swap token so a failed/canceled free
-- scan refunds the counter exactly once (mirrors the credit_reservations
-- active->released idiom used for paid scans).

ALTER TABLE users
  ADD COLUMN IF NOT EXISTS free_scans_used INTEGER NOT NULL DEFAULT 0;

ALTER TABLE scan_jobs
  ADD COLUMN IF NOT EXISTS free_scan_counted BOOLEAN NOT NULL DEFAULT FALSE;

-- Backfill MUST run before the next cleanup, while historical free scans still
-- exist. Mark the rows that currently count toward the quota: short
-- (<= 800 words, the free threshold) and not failed/canceled.
UPDATE scan_jobs
   SET free_scan_counted = TRUE
 WHERE word_count <= 800
   AND status NOT IN ('failed', 'canceled')
   AND user_id IS NOT NULL;

-- Seed each user's durable counter from those same rows.
UPDATE users u
   SET free_scans_used = sub.c
  FROM (
        SELECT user_id, count(*) AS c
          FROM scan_jobs
         WHERE word_count <= 800
           AND status NOT IN ('failed', 'canceled')
           AND user_id IS NOT NULL
         GROUP BY user_id
       ) sub
 WHERE u.id = sub.user_id;
