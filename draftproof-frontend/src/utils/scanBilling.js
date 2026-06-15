export const FREE_SCAN_WORD_LIMIT = 800;

export function countWords(text) {
  const trimmed = String(text || '').trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

export function scanTokensRequired(wordCount) {
  const count = Number(wordCount) || 0;
  if (count <= FREE_SCAN_WORD_LIMIT) return 0;
  return Math.max(1, Math.ceil(count / 1000));
}

// Credit cost when a scan is billed: 1 token per started 1,000 words.
// A short (<=800 word) doc costs 1 — the rate once the free quota is spent.
export function paidScanTokens(wordCount) {
  const count = Number(wordCount) || 0;
  return Math.max(1, Math.ceil(count / 1000));
}

// Effective cost shown to the user: short docs are free while free scans
// remain, then fall back to the paid rate so credits can pay for them.
// `freeRemaining` null/undefined means "not loaded yet" — assume free.
export function effectiveScanTokens(wordCount, freeRemaining) {
  const count = Number(wordCount) || 0;
  if (count > FREE_SCAN_WORD_LIMIT) return paidScanTokens(count);
  const remaining = freeRemaining == null ? Infinity : freeRemaining;
  return remaining > 0 ? 0 : paidScanTokens(count);
}
