-- 009: Release stranded 'active' credit reservations for scan jobs that have
-- already reached a terminal state (completed, failed) without capturing or
-- releasing the reservation.  This was possible before the release_scan_credits
-- fix because the scan task's timeout and max-retries-exhausted failure paths
-- had no corresponding release call.
--
-- Safe to re-run: the WHERE status = 'active' predicate is idempotent.
-- GREATEST(0, ...) guards against reserved_tokens going negative if the
-- accounting was already corrected manually.

WITH stranded AS (
    UPDATE credit_reservations cr
    SET    status     = 'released',
           updated_at = now()
    FROM   scan_jobs sj
    WHERE  cr.job_id    = sj.id
      AND  cr.job_type  = 'scan'
      AND  cr.status    = 'active'
      AND  sj.status   IN ('completed', 'failed')
    RETURNING cr.credit_account_id, cr.tokens_reserved
),
totals AS (
    SELECT credit_account_id,
           sum(tokens_reserved) AS tokens_to_free
    FROM   stranded
    GROUP  BY credit_account_id
)
UPDATE credit_accounts ca
SET    reserved_tokens = GREATEST(0, ca.reserved_tokens - totals.tokens_to_free)
FROM   totals
WHERE  ca.id = totals.credit_account_id;
