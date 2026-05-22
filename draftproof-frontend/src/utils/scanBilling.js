export const FREE_SCAN_WORD_LIMIT = 500;

export function countWords(text) {
  const trimmed = String(text || '').trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

export function scanTokensRequired(wordCount) {
  const count = Number(wordCount) || 0;
  if (count <= FREE_SCAN_WORD_LIMIT) return 0;
  return Math.max(1, Math.ceil(count / 1000));
}
