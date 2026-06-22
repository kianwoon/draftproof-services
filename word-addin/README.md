# DraftProof — MS Word add-in (Phase 2)

An Office.js task-pane add-in: highlight text in Word, scan it for AI-writing risk
via DraftProof, billed to your DraftProof credits. **Plain static files** (vanilla
Office.js + `fetch`) — no build step.

## How it fits together

```
Word task pane (this folder)
  └─ served from the API domain at  https://draftproof.app/word-addin/taskpane.html
       └─ calls  /api/ext/scan  +  /api/ext/scan/{id}   (Phase 1 backend)
            └─ Authorization: Bearer dp_live_…   (the user's API key)
```

Because the pane is served from the **same origin** as the API, the `fetch` calls
are same-origin — **no CORS needed** in production. (The `Authorization`/`X-API-Key`
headers were added to the API CORS config anyway, for separate-host / local dev.)

## Files

| File | Purpose |
|------|---------|
| `manifest.xml` | Office add-in manifest (XML "add-in only" format). Points at `https://draftproof.app/word-addin/taskpane.html` + a Home-tab ribbon button. |
| `taskpane.html` | Pane markup: API-key setup, scan button, results. Loads office.js from Microsoft's CDN. |
| `taskpane.js` | Controller: read selection → POST `/api/ext/scan` → poll `/api/ext/scan/{id}` → render tier/scores. Key in `localStorage`. |
| `taskpane.css` | Styles. |

## Get an API key

The add-in needs a personal key. Generate one in the web app at
**https://draftproof.app/api-keys** (Dashboard → API keys), copy it once, and paste
it into the pane. Revoke/rotate from the same page.

> Security note: the task pane is a web context, so the key lives in the browser's
> `localStorage` — treat it like a password and revoke it if a device is lost.

## Production deploy (served from the API)

The root `Dockerfile` copies this folder into the API's static root:

```dockerfile
COPY word-addin/ ./static/word-addin/
```

so `https://draftproof.app/word-addin/taskpane.html` resolves via the existing SPA
static handler. Pushing to `main` redeploys the API (Koyeb). No separate host.

If your production domain is **not** `draftproof.app`, replace that host in
`manifest.xml` (`SourceLocation`, `AppDomains`, the `bt:Url`/`bt:Image` resources)
and in the `EXT` base in `taskpane.js` (currently relative, so usually no change).

## Sideload for development / testing

Word on the web or desktop, pointing at the **production** pane URL:

1. Generate a key at https://draftproof.app/api-keys.
2. **Word on the web:** Home ▸ Add-ins ▸ *More Add-ins* ▸ *My Add-ins* ▸
   *Upload My Add-in* ▸ choose `manifest.xml`.
   **Word desktop (Windows):** put `manifest.xml` in a
   [shared folder catalog](https://learn.microsoft.com/office/dev/add-ins/testing/create-a-network-shared-folder-catalog-for-task-pane-and-content-add-ins),
   trust it, then Insert ▸ My Add-ins ▸ Shared Folder.
   **Word desktop (Mac):** copy `manifest.xml` to
   `~/Library/Containers/com.microsoft.Word/Data/Documents/wef`.
3. Open the add-in from the Home tab (**DraftProof ▸ Scan selection**), paste your
   key, highlight text, and scan.

To test against a **local** pane (e.g. while editing this folder), serve it over
HTTPS and point `manifest.xml`'s `SourceLocation` at your local URL; set the `EXT`
base in `taskpane.js` to your API origin (the API CORS allow-list must include it).

## Publishing to AppSource (later)

Partner Center submission requires: a stable HTTPS-hosted pane (✓ via the API),
production icons at the declared sizes, a privacy URL + terms, and the XML manifest
passing the validator (`npx office-addin-manifest validate manifest.xml`).

## Status

Scaffold — not yet sideload-tested in a live Word client. Backend it depends on
(`/api/ext/*`, `/api/keys`) is Phase 1, complete and unit-tested.
