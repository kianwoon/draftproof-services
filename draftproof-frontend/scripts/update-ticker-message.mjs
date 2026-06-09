// ─────────────────────────────────────────────────────────────────────────
// update-ticker-message.mjs — local admin script to set the site news ticker.
//
// Rewrites src/announcements.js (the SOURCE, never dist/) with one active
// announcement, or clears the ticker. Run via the npm helper:
//
//   npm run update:ticker -- --text "Message copy" --date "Jun 2026" --url "https://example.com"
//   npm run update:ticker -- --text "Message copy" --dry-run
//   npm run update:ticker -- --clear
//
// Flags:
//   --text "..."   Replace the ticker with one active announcement (required
//                  unless --clear).
//   --date "..."   Optional date chip shown before the text (e.g. "Jun 2026").
//   --url "..."    Optional http/https link; makes the item clickable and shows
//                  the localized "Read more" cue (from i18n, not this script).
//   --id "..."     Optional stable slug; auto-generated from text/date if omitted.
//   --dry-run      Print the generated announcements.js to stdout; write nothing.
//   --clear        Write an empty announcements array to hide the ticker.
//
// The data contract (announcement object shape) and consumer live in:
//   src/announcements.js, src/components/NewsTicker.jsx
// ─────────────────────────────────────────────────────────────────────────

import { writeFileSync } from 'node:fs';

// Write target resolved relative to THIS module, so `npm run` (cwd = frontend
// root) and a bare `node scripts/...` from anywhere both hit src/, never dist/.
const TARGET_URL = new URL('../src/announcements.js', import.meta.url);

const BANNER = `// ─────────────────────────────────────────────────────────────────────────
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
// ─────────────────────────────────────────────────────────────────────────`;

function fail(message) {
  process.stderr.write(`update-ticker-message: ${message}\n`);
  process.exit(1);
}

// Minimal parser: supports `--flag value`, `--flag=value`, and boolean flags.
function parseArgs(argv) {
  const booleans = new Set(['dry-run', 'clear']);
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
    if (!['text', 'date', 'url', 'id'].includes(name)) fail(`unknown flag: --${name}`);
    let value;
    if (eq !== -1) {
      value = body.slice(eq + 1);
    } else {
      value = argv[i + 1];
      if (value === undefined || value.startsWith('--')) fail(`--${name} requires a value`);
      i += 1;
    }
    out[name] = value;
  }
  return out;
}

// Deterministic kebab slug from text/date. Falls back to 'announcement' when the
// input is non-ASCII (e.g. Chinese copy) and reduces to empty.
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

// Render the announcements.js file body. `items` is [] for --clear.
function renderFile(items) {
  if (items.length === 0) {
    return `${BANNER}\nexport const announcements = [];\n`;
  }
  const blocks = items.map((item) => {
    const lines = [`    id: ${JSON.stringify(item.id)},`];
    if (item.date) lines.push(`    date: ${JSON.stringify(item.date)},`);
    lines.push(`    text: ${JSON.stringify(item.text)},`);
    if (item.url) lines.push(`    url: ${JSON.stringify(item.url)},`);
    return `  {\n${lines.join('\n')}\n  },`;
  });
  return `${BANNER}\nexport const announcements = [\n${blocks.join('\n')}\n];\n`;
}

function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.clear) {
    const conflicts = ['text', 'date', 'url', 'id'].filter((k) => args[k] !== undefined);
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
  if (args.url !== undefined) validateUrl(args.url);

  const item = {
    id: (args.id && args.id.trim()) || slugify([args.date, args.text]),
    text: args.text,
  };
  if (args.date) item.date = args.date;
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
