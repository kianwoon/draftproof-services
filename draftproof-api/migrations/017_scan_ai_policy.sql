-- Phase 1 of docs/plans/policy_risk_external_review_response.md: optional per-scan AI-policy
-- context. Captures THIS ASSIGNMENT's AI policy (not institution-wide -- one school can have
-- different rules across modules/assignments), so the policy_risk composer's two hypothetical
-- lenses (AI-allowed / AI-restricted) can eventually be presented as a single headlined reading
-- instead of two side-by-side guesses (Phase 2, not built yet). Nullable, default 'unknown' --
-- absent/unset is the common case and must stay byte-identical to today's reports.
ALTER TABLE scan_jobs
  ADD COLUMN IF NOT EXISTS ai_policy TEXT DEFAULT 'unknown';
