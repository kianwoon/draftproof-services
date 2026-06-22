# Publishing the DraftProof Google Docs add-on

Two stages: **deploy + self-test** (quick, after `clasp login`), then **public Marketplace
listing** (manual console work + Google review). Sizes below are the usual requirements —
confirm exact values in the console as you go.

## Stage 0 — Deploy & self-test (needs `clasp login`)
```bash
clasp login                                                   # interactive, your Google account
cd google-docs-addon
clasp create-script --type standalone --title "DraftProof" --rootDir .
clasp push                                                    # Code.gs, Sidebar.html, appsscript.json
clasp open-script
```
In the Apps Script editor → **Deploy → Test deployments → Install** → open any Google Doc →
**Extensions → DraftProof → Open DraftProof**. Paste an API key, highlight text, scan.

## Stage 1 — GCP project + OAuth consent screen
1. Apps Script editor → **Project Settings → Google Cloud Platform (GCP) Project →
   Change project** → set a **standard** GCP project number (create one at
   console.cloud.google.com if needed).
2. GCP → **APIs & Services → OAuth consent screen**:
   - User type **External** (or **Internal** to limit to your Workspace org — no verification).
   - App name **DraftProof**, support email, app logo (**120×120** PNG), app domain
     **draftproof.app**, authorized domain **draftproof.app**, developer contact.
   - **Scopes:** `documents.currentonly`, `script.external_request`, `script.container.ui`
     → these are **sensitive**, so Google **OAuth verification** is required.
   - **Privacy policy URL** + **Terms of service URL** (mandatory).
3. **Submit for verification.** Sensitive-scope review can take days–weeks and may ask for a
   short demo video showing how each scope is used.

## Stage 2 — Workspace Marketplace SDK + listing
1. GCP project → **Enable APIs → Google Workspace Marketplace SDK**.
2. Marketplace SDK → **App Configuration**:
   - Visibility: Public (or unlisted / your-domain for a soft launch).
   - **Editor add-on** extension → link the Apps Script **deployment** (script ID + a numbered
     deployment; create with `clasp create-deployment` or in the editor).
   - Scopes must match the consent screen.
3. Marketplace SDK → **Store Listing**: name, short + long description, category,
   **screenshots (1280×800)**, **app icon (128×128)** (+ 32×32), optional banner (220×140),
   support/privacy/terms URLs, languages (en + zh).
4. **Publish → submit for review.**

## The blocker (same as AppSource)
Reviewers must be able to **use** the add-on, which needs a key + credits. Provide a **test
account/key with credits** in the submission notes, or ship a **free-scan trial** so any
reviewer (and new user) can scan without paying. Decide this before submitting.

## Assets checklist
- [ ] App logo 120×120 (consent) + 128×128 (listing) + 32×32
- [ ] 1–5 screenshots 1280×800
- [ ] Short + long descriptions (reuse the AppSource copy)
- [ ] Privacy policy URL + Terms of service URL (must exist on draftproof.app)
- [ ] Support email / URL
- [ ] Test access for reviewers (key + credits) or free-scan trial
