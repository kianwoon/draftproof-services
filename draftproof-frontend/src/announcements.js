// ─────────────────────────────────────────────────────────────────────────
// Site news ticker — EDIT THIS LIST to change the running announcements.
// After editing, commit + push to main (Koyeb auto-deploys).
//
// Each item:
//   id   – unique string (stable key; any short slug)
//   date – OPTIONAL. Short label shown as a chip before the text (e.g. "May 2026").
//   text – the announcement copy shown in the ticker
//   url  – OPTIONAL. If present, the item links out (opens in a new tab) and
//          shows a "Read more" cue. Omit for a plain, non-clickable notice.
//
// Items scroll in order and loop continuously. Keep entries concise.
// To hide the ticker entirely, leave this array empty.
// ─────────────────────────────────────────────────────────────────────────
export const announcements = [
  {
    id: 'turnitin-product-updates-2026',
    date: 'May 2026',
    text: 'Turnitin’s latest updates focus on configurable AI use, stronger AI detection, authorship review, writing-process evidence, and greater transparency in student submissions.',
    url: 'https://guides.turnitin.com/hc/en-us/articles/29645383597965-Turnitin-product-updates',
  },
];
