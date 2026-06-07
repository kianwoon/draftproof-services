-- Bounded report-list metadata for identifying scans without storing full text.
ALTER TABLE scan_jobs
  ADD COLUMN IF NOT EXISTS document_title TEXT,
  ADD COLUMN IF NOT EXISTS content_preview TEXT;
