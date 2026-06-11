// Pure helpers for bracket-grounding spans produced by the V6 rewrite pipeline.
// Each span is { start, end, kind } — a char range against the rewritten baseline
// text — where kind 'improved' renders GREEN (the rewrite grounded it) and 'kept'
// renders AMBER (the rewrite could not safely improve it, so the original was kept
// and the USER should edit it).
//
// The original offsets are valid only against the unedited baseline, so we never
// reuse them once the user starts typing. Instead we keep the amber sentence STRINGS
// and re-find them in the *current* draft text (whitespace-tolerant) on every render:
// a highlight survives edits elsewhere and clears the moment its own sentence changes
// — exactly the signal we want ("amber clears as you rewrite it").

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

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Locate `sentence` inside `text`, tolerating whitespace differences (the stored
// sentence is single-spaced; the live draft may wrap across lines). Returns the
// first { start, end } match or null when the sentence is absent (e.g. edited).
export function findSentenceRange(text, sentence) {
  const needle = String(sentence || '').trim();
  if (!needle) return null;
  const pattern = needle.split(/\s+/).map(escapeRegExp).join('\\s+');
  try {
    const match = new RegExp(pattern).exec(String(text || ''));
    return match ? { start: match.index, end: match.index + match[0].length } : null;
  } catch {
    return null;
  }
}

// Split `text` into ordered { type: 'plain' | 'selected', text, index? } parts for
// the highlight-layer overlay. `ranges` is a list of { start, end, index }; the
// `index` rides through to the rendered span (used to wire click-to-jump). Overlaps
// are dropped (first-wins) so the parts always tile the text exactly once.
export function highlightParts(text, ranges) {
  const source = String(text || '');
  const sorted = (Array.isArray(ranges) ? ranges : [])
    .filter((r) => r && Number.isInteger(r.start) && Number.isInteger(r.end) && r.end > r.start)
    .sort((a, b) => a.start - b.start);

  const parts = [];
  let cursor = 0;
  sorted.forEach((range) => {
    const start = Math.max(cursor, range.start);
    const end = Math.min(source.length, range.end);
    if (end <= start) return;
    if (start > cursor) parts.push({ type: 'plain', text: source.slice(cursor, start) });
    parts.push({ type: 'selected', text: source.slice(start, end), index: range.index });
    cursor = end;
  });
  if (cursor < source.length) parts.push({ type: 'plain', text: source.slice(cursor) });
  return parts.filter((part) => part.text);
}
