// ─────────────────────────────────────────────────────────────────────────
// update-ticker-message.mjs — local admin script to set the site news band.
//
// Rewrites src/announcements.js (the SOURCE, never dist/) with one active
// announcement, or clears the band. Run via the npm helper:
//
//   npm run update:ticker -- --headline "Lead-in" --text "Supporting line" \
//     --badge "Turnitin" --date "Jul 2026" --emphasis "before" \
//     --pill "Topic A" --pill "Topic B" --url "https://example.com"
//   npm run update:ticker -- --text "Message copy" --dry-run
//   npm run update:ticker -- --clear
//
// Flags:
//   --text "..."     Supporting line shown after the headline (required unless
//                    --clear).
//   --headline "..." Optional bold lead-in shown before the supporting line.
//   --badge "..."    Optional source label shown in the chip (e.g. "Turnitin").
//   --date "..."     Optional short date shown after the badge in the chip.
//   --emphasis "..." Optional single word inside --text to italicise. Ignored by
//                    the band if the word is not present in --text.
//   --pill "..."     Optional topic pill; repeat the flag to add several
//                    (--pill "A" --pill "B"). Order is preserved.
//   --url "..."      Optional http/https link; makes the band show the localized
//                    "Read update" cue (from i18n, not this script).
//   --id "..."       Optional stable slug; auto-generated from badge/date/text
//                    if omitted.
//   --dry-run        Print the generated announcements.js to stdout; write nothing.
//   --clear          Write an empty announcements array to hide the band.
//
// The data contract (announcement object shape) and consumer live in:
//   src/announcements.js, src/components/AnnouncementBanner.jsx
// ─────────────────────────────────────────────────────────────────────────

import { writeFileSync } from 'node:fs';

// Write target resolved relative to THIS module, so `npm run` (cwd = frontend
// root) and a bare `node scripts/...` from anywhere both hit src/, never dist/.
const TARGET_URL = new URL('../src/announcements.js', import.meta.url);

const BANNER = `// ─────────────────────────────────────────────────────────────────────────
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
//   emphasis – OPTIONAL. A single word inside \`text\` to italicise for emphasis.
//   pills    – OPTIONAL. Array of short topic labels shown as a row of chips.
//   url      – OPTIONAL. If present, the band shows a "Read update" link
//              (opens in a new tab). Omit for a plain, non-clickable notice.
//
// To hide the band entirely, leave this array empty.
// ─────────────────────────────────────────────────────────────────────────`;

function fail(message) {
  process.stderr.write(`update-ticker-message: ${message}\n`);
  process.exit(1);
}

// String flags accept a value; --pill is repeatable and accumulates into an
// array; the rest are booleans.
const STRING_FLAGS = ['text', 'headline', 'badge', 'date', 'emphasis', 'url', 'id'];
const ARRAY_FLAGS = ['pill'];
const BOOLEAN_FLAGS = ['dry-run', 'clear'];

// Minimal parser: supports `--flag value`, `--flag=value`, boolean flags, and
// repeatable array flags (--pill "A" --pill "B").
function parseArgs(argv) {
  const booleans = new Set(BOOLEAN_FLAGS);
  const arrays = new Set(ARRAY_FLAGS);
  const strings = new Set(STRING_FLAGS);
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) fail(`unexpected argument: ${token}`);
    const body = token.slice(2);
    const eq = body.indexOf('=');
    const name = eq === -1 ? body : body.slice(0, eq);
    if (booleans.has(name)) {
      if (eq !== -1) fail(`--${name} does not take a value`);
      out[name] = true;
      continue;
    }
    if (!strings.has(name) && !arrays.has(name)) fail(`unknown flag: --${name}`);
    let value;
    if (eq !== -1) {
      value = body.slice(eq + 1);
    } else {
      value = argv[i + 1];
      if (value === undefined || value.startsWith('--')) fail(`--${name} requires a value`);
      i += 1;
    }
    if (arrays.has(name)) {
      (out[name] ||= []).push(value);
    } else {
      out[name] = value;
    }
  }
  return out;
}

// Deterministic kebab slug from badge/date/text. Falls back to 'announcement'
// when the input is non-ASCII (e.g. Chinese copy) and reduces to empty.
function slugify(parts) {
  const slug = parts
    .filter(Boolean)
    .join(' ')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)
    .replace(/-+$/g, '');
  return slug || 'announcement';
}

function validateUrl(url) {
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    fail(`invalid --url: ${url}`);
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    fail(`--url must be http or https, got: ${parsed.protocol}`);
  }
}

// Render the announcements.js file body. `items` is [] for --clear. Field order
// mirrors the documented shape so diffs stay stable and reviewable.
function renderFile(items) {
  if (items.length === 0) {
    return `${BANNER}\nexport const announcements = [];\n`;
  }
  const blocks = items.map((item) => {
    const lines = [`    id: ${JSON.stringify(item.id)},`];
    if (item.badge) lines.push(`    badge: ${JSON.stringify(item.badge)},`);
    if (item.date) lines.push(`    date: ${JSON.stringify(item.date)},`);
    if (item.headline) lines.push(`    headline: ${JSON.stringify(item.headline)},`);
    lines.push(`    text: ${JSON.stringify(item.text)},`);
    if (item.emphasis) lines.push(`    emphasis: ${JSON.stringify(item.emphasis)},`);
    if (item.pills && item.pills.length) {
      const pillLines = item.pills.map((p) => `      ${JSON.stringify(p)},`).join('\n');
      lines.push(`    pills: [\n${pillLines}\n    ],`);
    }
    if (item.url) lines.push(`    url: ${JSON.stringify(item.url)},`);
    return `  {\n${lines.join('\n')}\n  },`;
  });
  return `${BANNER}\nexport const announcements = [\n${blocks.join('\n')}\n];\n`;
}

function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.clear) {
    const conflicts = [...STRING_FLAGS, ...ARRAY_FLAGS].filter((k) => args[k] !== undefined);
    if (conflicts.length) {
      fail(`--clear cannot be combined with ${conflicts.map((k) => `--${k}`).join(', ')}`);
    }
    const content = renderFile([]);
    if (args['dry-run']) {
      process.stdout.write(content);
      return;
    }
    writeFileSync(TARGET_URL, content);
    process.stdout.write('Ticker cleared (announcements is now empty).\n');
    return;
  }

  if (args.text === undefined) fail('nothing to do: pass --text "..." or --clear');
  if (args.text.trim() === '') fail('--text must not be empty (use --clear to hide the ticker)');
  if (args.id !== undefined && args.id.trim() === '') fail('--id must not be empty when provided');
  if (args.emphasis !== undefined && !args.text.includes(args.emphasis)) {
    fail(`--emphasis "${args.emphasis}" does not appear in --text (the band would ignore it)`);
  }
  const pills = (args.pill || []).map((p) => p.trim()).filter(Boolean);
  if (args.url !== undefined) validateUrl(args.url);

  const item = {
    id: (args.id && args.id.trim()) || slugify([args.badge, args.date, args.text]),
    text: args.text,
  };
  if (args.badge) item.badge = args.badge;
  if (args.date) item.date = args.date;
  if (args.headline) item.headline = args.headline;
  if (args.emphasis) item.emphasis = args.emphasis;
  if (pills.length) item.pills = pills;
  if (args.url) item.url = args.url;

  const content = renderFile([item]);
  if (args['dry-run']) {
    process.stdout.write(content);
    return;
  }
  writeFileSync(TARGET_URL, content);
  process.stdout.write(`Ticker updated (id: ${item.id}).\n`);
}

main();
