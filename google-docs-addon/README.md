# DraftProof — Google Docs add-on (Phase 3)

A Google Docs **editor add-on** (Apps Script): highlight text in a Doc, scan it for
AI‑writing risk via DraftProof, billed to your credits. It reuses the **same**
key‑authenticated endpoints as the Word add‑in.

## How it fits together

```
Google Docs sidebar (Sidebar.html)
   └─ google.script.run ─▶ Code.gs (Apps Script, server-side)
        └─ UrlFetchApp ─▶ https://draftproof.app/api/ext/scan  + /scan/{id}  + /scan/{id}/report
             └─ Authorization: Bearer dp_live_…   (key stored in PropertiesService)
```

Unlike the Word pane (browser `fetch`, key in `localStorage`), the Google add‑on calls
the API **server‑side** via `UrlFetchApp` — so there's **no CORS** and the key lives in
**`PropertiesService` (per‑user, server‑side)**, never in client JS.

## Files
| File | Purpose |
|------|---------|
| `Code.gs` | Server: menu/sidebar, key storage, read selection, submit/poll/report via UrlFetchApp. |
| `Sidebar.html` | The sidebar UI (themed to match the site) — talks to `Code.gs` via `google.script.run`. |
| `appsscript.json` | Manifest: V8 runtime + OAuth scopes (`documents.currentonly`, `script.external_request`, `script.container.ui`). |

## Get an API key
Generate one at **https://draftproof.app/api-keys** (copy it once), then paste it into the
sidebar's key field. It's stored per‑user via `PropertiesService`.

## Deploy / test (Apps Script — this is NOT on the Koyeb pipeline)

**Option A — clasp (recommended).** clasp 3.x is installed.
```bash
clasp login                  # interactive: authorises YOUR Google account in a browser
# from this folder (google-docs-addon/):
clasp create-script --type standalone --title "DraftProof" --rootDir .
clasp push                   # uploads Code.gs, Sidebar.html, appsscript.json
clasp open-script            # opens the Apps Script editor
```
Then in the editor: **Deploy → Test deployments → Install** → open a Google Doc →
**Extensions → DraftProof → Open DraftProof**.

> v3 note: it's `create-script` / `clone-script` (not `create`/`clone`). A **standalone**
> script is the right base for a publishable add-on. `.claspignore` here whitelists only
> `Code.gs`, `Sidebar.html`, `appsscript.json`; `.clasp.json` (the scriptId) is gitignored.
> Keep `Sidebar.html` named exactly so `createHtmlOutputFromFile('Sidebar')` resolves.

**Option B — manual paste:**
1. In a Google Doc: **Extensions → Apps Script**.
2. Create files matching this folder: `Code.gs`, `Sidebar.html`, and set the manifest
   (`appsscript.json`) via **Project Settings → Show "appsscript.json"**.
3. Save → reload the Doc → **Extensions → DraftProof → Open DraftProof**.

## Publish (later)
Google Workspace Marketplace, via the Apps Script project → **Deploy → New deployment →
Add‑on**, then the Marketplace SDK listing (icons, screenshots, privacy + terms, OAuth
verification). Same testability caveat as AppSource: reviewers need a key + credits, so a
free‑scan trial / test account is required.

## Status
Scaffold — not yet tested in a live Doc. Backend it depends on (`/api/ext/*`) is live.
