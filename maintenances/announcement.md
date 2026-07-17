---
name: draftproof-announcements
description: Use this when maintaining DraftProof site ticker announcements, updating the running message, editing draftproof-frontend/src/announcements.js, or building the local ticker update script.
---

# DraftProof Announcement Skill

This file is written as an AI-agent skill. Load it before changing site ticker announcements, announcement update scripts, or ticker display behavior in this repo.

## Operating Rules

- Treat the ticker as a production-facing message surface.
- Keep one active announcement as the default operating mode unless the user explicitly asks for multiple messages.
- Do not hardcode production URLs, message copy, dates, or IDs inside script logic.
- Preserve the existing announcement data contract unless there is a clear product reason to change it.
- Validate input before writing files. Empty text is only allowed when intentionally clearing the ticker.
- Keep `AnnouncementBanner.jsx` unchanged unless the announcement object shape or display contract must change.
- Preserve unrelated worktree changes. This repo may already be dirty.

## Current Ticker Architecture

Source of truth:

```text
draftproof-frontend/src/announcements.js
```

Consumer:

```text
draftproof-frontend/src/components/AnnouncementBanner.jsx
```

The band is a single static feature under the header — only the FIRST array
item renders (not a scroller).

Localized ticker labels:

```text
draftproof-frontend/src/i18n/en/ticker.js
draftproof-frontend/src/i18n/zh/ticker.js
```

Announcement object shape:

```js
{
  id: 'short-stable-slug',
  badge: 'Turnitin',          // optional — source label in the chip
  date: 'May 2026',           // optional — date shown after the badge
  headline: 'Bold lead-in',   // optional — shown before the supporting line
  text: 'Supporting line',    // required
  emphasis: 'word',           // optional — one word inside text to italicise
  pills: ['Topic A', 'Topic B'], // optional — row of topic chips
  url: 'https://...'          // optional — shows the "Read update" link
}
```

An empty `announcements` array hides the band.

## Intended Local Script Contract

The local update script should live at:

```text
draftproof-frontend/scripts/update-ticker-message.mjs
```

Expected npm helper:

```bash
cd draftproof-frontend
npm run update:ticker -- \
  --headline "Lead-in" --text "Supporting line" \
  --badge "Turnitin" --date "Jun 2026" --emphasis "before" \
  --pill "Topic A" --pill "Topic B" \
  --url "https://example.com"
```

Expected script behavior:

- `--text "..."` sets the supporting line (required unless `--clear`).
- `--headline "..."` optionally adds the bold lead-in before the text.
- `--badge "..."` optionally sets the source label in the chip.
- `--date "..."` optionally adds the date chip (shown after the badge).
- `--emphasis "..."` optionally italicises one word; must appear in `--text`.
- `--pill "..."` optionally adds a topic pill; repeat the flag for several.
- `--url "..."` optionally shows the localized "Read update" link.
- `--id "..."` optionally sets a stable custom ID (else auto-slugged from badge/date/text).
- `--dry-run` prints the generated `announcements.js` content without writing.
- `--clear` writes an empty announcement array to hide the band.

## Validation Rules

- Reject empty `--text` unless `--clear` is used.
- Reject `--clear` combined with any content flag (`--text`, `--headline`, `--badge`, `--date`, `--emphasis`, `--pill`, `--url`, `--id`).
- Reject `--emphasis` when the word does not appear in `--text` (the band would ignore it).
- Accept only `http:` and `https:` URLs.
- Generate IDs from provided badge/date/text only when `--id` is absent.
- Escape generated JS values with structured serialization such as `JSON.stringify`.
- Write deterministic output so diffs are reviewable.

## Files To Inspect First

- `draftproof-frontend/src/announcements.js`
- `draftproof-frontend/src/components/NewsTicker.jsx`
- `draftproof-frontend/src/i18n/en/ticker.js`
- `draftproof-frontend/src/i18n/zh/ticker.js`
- `draftproof-frontend/package.json`
- `draftproof-frontend/scripts/`

## Verification

After implementing or changing the local update script, run a dry-run:

```bash
cd draftproof-frontend
npm run update:ticker -- --text "Test ticker message" --date "Jun 2026" --dry-run
```

Build the frontend:

```bash
cd draftproof-frontend
npm run build
```

Run whitespace validation:

```bash
git diff --check -- maintenances/announcement.md draftproof-frontend/scripts/update-ticker-message.mjs draftproof-frontend/src/announcements.js draftproof-frontend/package.json
```

## Administrator Runbook

1. Write the final public message copy.
2. Run the local update script in `--dry-run` mode first.
3. Confirm the output has exactly one active announcement unless intentionally clearing the ticker.
4. Run the script without `--dry-run` to update `src/announcements.js`.
5. Build the frontend before handoff or deployment.
6. Commit and push through the normal deploy flow.

## Common Failure Modes

- Script updates built `dist/` output instead of source: update `src/announcements.js` only.
- Invalid URL breaks the clickable announcement: reject invalid URL before writing.
- Multiple stale announcements accumulate: replace by default, append only if explicitly requested.
- Ticker is hidden unexpectedly: `announcements` is empty.
- Localized badge/read-more text is wrong: inspect ticker i18n files, not announcement data.
