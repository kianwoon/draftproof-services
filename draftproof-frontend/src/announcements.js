// ─────────────────────────────────────────────────────────────────────────
// Site news — EDIT THIS LIST to change the featured announcement band that
// sits directly under the header. Only the FIRST item is shown (the band is a
// single static feature, not a scroller). After editing, commit + push to main
// (Koyeb auto-deploys).
//
// allow-hardcode: editorial marketing copy for the announcement band — these
// are human-authored display labels (chip/headline/pills), NOT a scoring,
// matching, or detection oracle. Nothing here is compared against user input.
//
// Each item:
//   id       – unique string (stable key; any short slug)
//   badge    – OPTIONAL. Source label shown in the green chip (e.g. "Turnitin").
//   date     – OPTIONAL. Short date shown after the badge in the chip (e.g. "May 2026").
//   headline – OPTIONAL. Bold lead-in shown before the supporting line.
//   text     – the supporting line shown after the headline.
//   emphasis – OPTIONAL. A single word inside `text` to italicise for emphasis.
//   pills    – OPTIONAL. Array of short topic labels shown as a row of chips.
//   url      – OPTIONAL. If present, the band shows a "Read update" link
//              (opens in a new tab). Omit for a plain, non-clickable notice.
//
// To hide the band entirely, leave this array empty.
// ─────────────────────────────────────────────────────────────────────────
export const announcements = [
  {
    id: "turnitin-integrity-insights-2026",
    badge: "Turnitin",
    date: "Jul 2026",
    text: "Turnitin today released its latest Learning Integrity Insights Report revealing a gap in how students use AI for writing and a clear shift toward educators taking ownership of how AI is used in their classrooms and lecture halls.",
    url: "https://www.turnitin.com/media-center/",
  },
];
