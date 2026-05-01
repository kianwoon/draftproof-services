-- Add ai_score column to scan_jobs for displaying calibrated AI risk score
ALTER TABLE scan_jobs ADD COLUMN IF NOT EXISTS ai_score NUMERIC(6,2);
