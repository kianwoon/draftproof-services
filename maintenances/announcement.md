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
- Keep `NewsTicker.jsx` unchanged unless the announcement object shape or display contract must change.
- Preserve unrelated worktree changes. This repo may already be dirty.

## Current Ticker Architecture

Source of truth:

```text
draftproof-frontend/src/announcements.js
```

Consumer:

```text
draftproof-frontend/src/components/NewsTicker.jsx
```

Localized ticker labels:

```text
draftproof-frontend/src/i18n/en/ticker.js
draftproof-frontend/src/i18n/zh/ticker.js
```

Announcement object shape:

```js
{
  id: 'short-stable-slug',
  date: 'May 2026',       // optional
  text: 'Ticker message',
  url: 'https://...'      // optional
}
```

An empty `announcements` array hides the ticker.

## Intended Local Script Contract

The local update script should live at:

```text
draftproof-frontend/scripts/update-ticker-message.mjs
```

Expected npm helper:

```bash
cd draftproof-frontend
npm run update:ticker -- --text "Message copy" --date "Jun 2026" --url "https://example.com"
```

Expected script behavior:

- `--text "..."` replaces the ticker with one active announcement.
- `--date "..."` optionally adds the date chip.
- `--url "..."` optionally makes the announcement clickable and shows the existing localized read-more cue.
- `--id "..."` optionally sets a stable custom ID.
- `--dry-run` prints the generated `announcements.js` content without writing.
- `--clear` writes an empty announcement array to hide the ticker.

## Validation Rules

- Reject empty `--text` unless `--clear` is used.
- Reject `--clear` combined with `--text`, `--date`, `--url`, or `--id`.
- Accept only `http:` and `https:` URLs.
- Generate IDs from provided text/date only when `--id` is absent.
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
