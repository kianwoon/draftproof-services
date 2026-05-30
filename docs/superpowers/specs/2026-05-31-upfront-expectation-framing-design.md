# Up-front Expectation Framing — Design

- **Date:** 2026-05-31
- **Status:** Approved (locked)
- **Component:** `draftproof-frontend` (Rewrite page) + `draftproof-api` (response model) + worker/poc (estimate hoist)
- **Relation:** Opportunity #4 of the mitigation/trust thread. Built on the project objective: *mitigate the AI flag honestly; set honest expectations so the user trusts the rewrite and edits it.*

## Problem

After a rewrite, the user judges the result on the Rewrite page. If they see a residual AI-detection signal with no framing, it reads as **failure** — eroding trust and the willingness to review/edit. There is no up-front statement of what DraftProof does (mitigate + show a reviewable draft to learn from) versus what it does not (erase the perplexity floor / guarantee a pass). The honest external-detector estimate exists but only on the Report comparison page, not where the user first lands.

## Objective

A persistent framing banner at the top of the Rewrite page that sets *what this is / isn't* and shows the rewritten content's honest external-detector estimate, contextualized so a residual estimate reads as the **expected gate**, not a failure.

## Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Placement | Persistent banner at the top of the **Rewrite page** (above the diff) |
| 2 | Content | Framing copy **+ live external estimate**, contextualized |
| 3 | Approach | **Wire the estimate through** to the rewrite report (not co-locate elsewhere) |

## Architecture & feasibility context

`Rewrite.jsx` is a lean diff/text view: it only receives the fields **declared in `RewriteReportOut`** (`draftproof-api/app/models/__init__.py`) — `final_text`, `status`, `authorship_evidence`, etc. Scores and `detect_scan_rewritten` are **stripped** by that strict Pydantic model. The rewritten content's external estimate is reliably produced where the rewrite flow re-scans the rewritten text through the full detect report builder (`report.py:estimate_external_detector_likelihood`, attached to the badge). To reach the rewrite-page banner it must be **explicitly preserved + declared**.

## Components

### Component 1 — Framing copy (`draftproof-frontend/src/i18n/resources.js`, en + zh)
New `rewriteFraming` namespace:
- `title` — e.g. "Before you review this rewrite"
- `isCopy` — "DraftProof mitigates AI-detection risk and shows a reviewable draft to learn from — then edit with your own specifics."
- `isntCopy` — "This is not a 'make it pass' button. A residual estimate is expected: AI detectors score token predictability, which even strong, human-grounded writing can trigger."
- `estimateLabel` — "Honest external-detector estimate for this rewrite"
- `estimateContext` — "Treat external detectors as probabilistic signals, not verdicts."
- `action` — "Review the before/after, then replace the highlighted additions with your own real content."
Chinese (`zh`) translations for each (project uses `fallbackLng: 'en'`).

### Component 2 — Wire the rewritten external estimate to the rewrite report
- **Preserve through compaction:** add `"external_detector_estimate"` to `SCAN_BADGE_KEYS` in `worker/app/rewrite_scan_compaction.py` so the precomputed estimate survives detect-scan compaction. (`ai_components` are already preserved, so the estimate is recomputable as a fallback.)
- **Hoist to a top-level summary field:** where the rewrite report summary is assembled with `detect_scan_rewritten`, set `summary["external_detector_estimate"] = (detect_scan_rewritten.get("ai_risk_badge") or {}).get("external_detector_estimate")`. If absent, recompute via `estimate_external_detector_likelihood((badge or {}).get("ai_components") or {})`. The exact assembly site (poc `production.py` ~line 188 and/or `worker/app/tasks.py` ~line 1453) is pinned during planning; set it wherever `detect_scan_rewritten` is in scope, before serialization.
- **Declare in the response model:** add `external_detector_estimate: Optional[Any] = None` to `RewriteReportOut` (`draftproof-api/app/models/__init__.py`) — undeclared fields are silently stripped (the lesson paid for by the earlier authorship hotfix `9695c280`).

### Component 3 — Banner render (`draftproof-frontend/src/pages/Rewrite.jsx`)
- A framing `<section>` at the **top** of the page (above the diff section), reusing existing notice/section styling.
- Renders the framing copy (`is`/`isn't`/`action`) **always**.
- Renders the estimate line **only when** `report?.external_detector_estimate` is present — showing the score (and band, if present) with `estimateLabel` + `estimateContext`.

## Honesty guardrails

- The estimate is framed "expected / probabilistic," never "you failed / you passed."
- **Graceful degradation:** when `external_detector_estimate` is absent (old reports, the poc path, or a re-scan that didn't run), the banner shows framing copy with **no number** and does not crash.
- Reuses the existing honest `estimate_external_detector_likelihood`; introduces no new scoring or claims.

## Verification

1. **Backend unit test** — `estimate_external_detector_likelihood` over a representative `ai_components` dict yields a `{score, band, ...}` shape; the hoist helper returns `None` gracefully on a missing badge/components.
2. **Serialization check** — a rewrite report dict carrying `external_detector_estimate` survives `RewriteReportOut(**data)` (field declared) — assert the value is present on the model output.
3. **Frontend build** — `npm run build` clean; banner renders with the estimate (a report fixture that has it) and copy-only (a report without it).

## Out of scope (YAGNI)

- Co-locating the framing on the scan/Report page (the estimate already renders there; not duplicating now).
- Any change to how the estimate itself is computed.
- A pre-rewrite dialog change (the existing `RewriteNoticeDialog` is untouched).
