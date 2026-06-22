# DraftProof add-on — store listing + OAuth copy

Paste-ready copy for the Google Workspace Marketplace **Store Listing** and the
**OAuth consent screen**. (Most of this also works for the Word add-in's AppSource
listing.) Keep it honest — DraftProof surfaces AI-writing *risk* + guidance; it is not
a guarantee against any detector.

## Store listing
- **App name:** DraftProof
- **Short description (≤ ~80 chars):**
  Scan highlighted text for AI-writing risk — with grounding & critical-thinking guidance.
- **Detailed description:**
  > DraftProof helps writers check their own work for AI-writing risk and improve it.
  > Highlight any passage in your document and scan it: DraftProof returns an
  > AI-likelihood and writing-quality read, critical-thinking questions to sharpen your
  > argument, and paragraph-level "issues" with the main problem and how to improve it.
  >
  > It's a writing-integrity coach, not a detector-beater: it shows where text reads as
  > generic or ungrounded and how to anchor it with your own specifics.
  >
  > Requires a free DraftProof account and an API key (generate one at
  > draftproof.app/api-keys). Scans draw on your DraftProof credits.
- **Category:** Productivity (or Education)
- **Graphics:** app icon **128×128** + **32×32** (PNG); 1–5 screenshots **1280×800**;
  optional banner 220×140.
- **Links:** website https://draftproof.app · privacy https://draftproof.app/privacy ·
  terms https://draftproof.app/terms · support support@draftproof.app
- **Languages:** English, Chinese (Simplified)

## OAuth consent screen — scope justifications
Reviewers ask *why* each scope is needed. Keep these tight and specific:

| Scope | Justification |
|-------|---------------|
| `…/auth/documents.currentonly` | Read the user's **current** document to get the highlighted selection to scan. Only the document the user opens the add-on in is accessed — never their other files. |
| `…/auth/script.external_request` | Send the selected text to the DraftProof API (`https://draftproof.app/api/ext`) to run the scan and return results. Core to the add-on's function. |
| `…/auth/script.container.ui` | Display the DraftProof sidebar UI inside the editor. |

If asked for a **demo video**: show opening a Doc → Extensions → DraftProof, pasting a
key, highlighting text, scanning, and the result — i.e., each scope used in context.

## Reviewer / tester access (required to pass review)
Reviewers must be able to run a scan, which needs a key + credits. Provide in the
submission notes either:
- a **test account + API key pre-loaded with credits**, or
- a **free-scan trial** (N free extension scans per user) so any reviewer can scan.

## Pricing
Free to install; usage billed via DraftProof credits (state this in the listing so
reviewers/users aren't surprised).
