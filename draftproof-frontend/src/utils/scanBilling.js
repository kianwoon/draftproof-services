export function countWords(text) {
  const trimmed = String(text || '').trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

// Credit cost for a scan: 1 token per started 1,000 words (min 1). Every scan is
// billed at this rate; new accounts instead start with a one-time welcome grant
// of free credits. Mirrors the backend scan_service._paid_scan_cost.
export function paidScanTokens(wordCount) {
  const count = Number(wordCount) || 0;
  return Math.max(1, Math.ceil(count / 1000));
}
