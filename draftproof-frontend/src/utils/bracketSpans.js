// Pure helpers for bracket-grounding spans produced by the V6 rewrite pipeline.
// Each span is { start, end, kind } — a char range against the rewritten baseline
// text — where kind 'improved' renders GREEN (the rewrite grounded it) and 'kept'
// renders AMBER (the rewrite could not safely improve it, so the original was kept
// and the USER should edit it). Offsets are valid only against the unedited
// baseline, so these helpers are for read-only reference (never for live highlights
// over an edited textarea, where the offsets drift).

// Extract the exact AMBER ('kept') sentence strings from the spans, in document
// order, deduped and whitespace-normalized. These are the sentences the rewrite
// left as-is for the user to ground themselves.
export function keptSentences(text, spans) {
  const source = String(text || '');
  const n = source.length;
  const seen = new Set();
  const out = [];
  (Array.isArray(spans) ? spans : []).forEach((b) => {
    if (!b || b.kind !== 'kept') return;
    if (!Number.isInteger(b.start) || !Number.isInteger(b.end)) return;
    if (b.start < 0 || b.end > n || b.start >= b.end) return;
    const sentence = source.slice(b.start, b.end).replace(/\s+/g, ' ').trim();
    if (!sentence || seen.has(sentence)) return;
    seen.add(sentence);
    out.push(sentence);
  });
  return out;
}
