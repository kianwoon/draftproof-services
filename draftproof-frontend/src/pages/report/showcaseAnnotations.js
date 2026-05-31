// Showcase annotation layer (frontend) — additive teaching notes laid on the original -> rewritten
// worked example. Pure heuristic; mirrors poc/report/showcase_annotations.py. It EXPLAINS the
// change (it does not modify any text). Returns technique KEYS so the view can i18n the copy.

const FIRST_PERSON = /\b(in my|when i\b|i have|i've|i saw|i watched|i keep|i notice|my students|my classroom|in my experience|i taught|i lead|i grade)\b/i;
const NUMBER = /\b\d[\d,.%/–-]*\b/g;
const PROPER = /\b[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|of|for|and|the))*\s+[A-Z][a-zA-Z]+\b/g;
const HEDGE = /\b(many|some|often|generally|various|numerous|things|stuff|people)\b/i;

// One change -> { techniques: [{ key, detail? }] }. Keys: concrete_anchor | lived_experience | hedge_cut | specific.
export function annotateChange(original, rewritten) {
  const o = (original || '').trim();
  const r = (rewritten || '').trim();
  const ol = o.toLowerCase();
  const techniques = [];

  const newNumbers = (r.match(NUMBER) || []).filter((n) => !o.includes(n));
  const newProper = (r.match(PROPER) || []).filter((e) => !ol.includes(e.toLowerCase()));
  const anchors = [...newNumbers, ...newProper].slice(0, 3);
  if (anchors.length) techniques.push({ key: 'concrete_anchor', detail: anchors.map((a) => `“${a}”`).join(', ') });

  if (FIRST_PERSON.test(r) && !FIRST_PERSON.test(ol)) techniques.push({ key: 'lived_experience' });
  if (HEDGE.test(ol) && !HEDGE.test(r.toLowerCase())) techniques.push({ key: 'hedge_cut' });
  if (!techniques.length) techniques.push({ key: 'specific' });

  return { techniques };
}

// sentence_comparison rows -> annotated changes (skips unchanged/empty). Accepts both the
// orig_sentence/new_sentence and original/rewritten key conventions.
export function annotateComparison(rows) {
  const out = [];
  for (const row of rows || []) {
    const original = row.orig_sentence ?? row.original ?? '';
    const rewritten = row.new_sentence ?? row.rewritten ?? '';
    const changed = row.changed;
    if (changed === false || String(changed).toLowerCase() === 'false') continue;
    if (!original || !rewritten || String(original).trim() === String(rewritten).trim()) continue;
    out.push({ original: String(original), rewritten: String(rewritten), ...annotateChange(original, rewritten) });
  }
  return out;
}

// Split a document into trimmed, non-empty paragraphs (blank-line separated).
function splitParagraphs(text) {
  return String(text || '').split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
}

// Align original vs rewritten BY PARAGRAPH INDEX — the rewrite edits in place and preserves
// document structure (headings + paragraphs stay in order), so index alignment is sound. If the
// paragraph counts ever differ (a rare merge/split), returns [] rather than mis-pair, so the
// showcase section simply doesn't render instead of showing wrong before/after.
export function paragraphPairs(original, rewritten) {
  const op = splitParagraphs(original);
  const np = splitParagraphs(rewritten);
  if (!op.length || op.length !== np.length) return [];
  return op.map((o, i) => ({ original: o, rewritten: np[i], changed: o.trim() !== np[i].trim() }));
}

// Convenience: full-document worked-example annotations from raw original/final text.
export function annotateDocument(original, rewritten) {
  return annotateComparison(paragraphPairs(original, rewritten));
}
