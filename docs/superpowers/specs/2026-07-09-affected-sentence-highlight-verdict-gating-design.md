# Design: Verdict-Gated Affected-Sentence Highlighting

**Date:** 2026-07-09
**Status:** Draft — Fable 5 review APPROVE-WITH-CHANGES incorporated (see §9); pending owner approval
**Scope:** Sentence-level "affected sentence" highlighting only. Does NOT touch the document AI score (already maxed: v7_fused, AUC 0.9955, verified firing).

---

## 1. Problem (evidence-grounded)

The document verdict and the per-sentence highlights currently **disagree**, and the highlights over-flag human writing.

**Live evidence (production, 2026-07-09):**
- Scan `6bdbf611`: document verdict = **GREEN ("clean")**, yet **4 sentences painted red** `#dc2626` "High-confidence AI signal (≥99% confident this is AI-like)" — e.g. *"…during his voluntary haircutting work with socially vulnerable groups…"*, *"…endorsement of Johnny's skill."* (specific, personal, human).
- Pattern holds across **all 15 most-recent scans**: each is green but shows **4–9** red `deberta_high` sentences.
- Scan `8a8d517e` (amber, `deep_scan` empty → fakespot fallback): **8 of 15** sentences flagged, including "As a VET hairdressing educator at Box Hill Institute…" — the fakespot *moderate* band (saturated) over-flagging.

**A user sees "you're clean" and then a wall of red sentences.** That is the accuracy defect for "highlight the affected sentences."

## 2. Root cause

Highlighting is **already** driven by the calibrated desklib deep-scan at the honest `sent_threshold = 0.999` bar (`poc/detect_v7/deep_scan_heatmap.py::_band_for_deep_scan_score`, two-band clean/high; a prior session deliberately rejected a lower band as uncalibrated). The detector is **not** the problem. Two things are:

1. **Per-sentence highlights are not gated by the document verdict.** The doc-level fused score correctly suppresses weak signal (`DOC_FLOOR = 0.3`, fused cutoffs 32/48/65), but each ≥0.999 sentence is painted red regardless. desklib saturates enough that ~24% of *human* sentences cross 0.999 (proportion 0.14–0.33 on the green docs above) — harmless aggregated into a suppressed doc score, alarming as individual red verdicts.
2. **The fakespot fallback over-flags.** When the deep-scan Modal call fails, `_compute_deberta_heatmap` falls back to `detect.deberta_signal.compose_from_sentences` — the 3-band clean/**moderate**/high fakespot map, whose `moderate` (~0.50–0.99) band saturates and paints human sentences "Strong AI signal."

## 3. Design

Make the **highlight tier a function of BOTH the per-sentence score AND the document verdict**, reusing existing calibration (fused tier + `DOC_FLOOR`) — no new uncalibrated threshold.

### 3.1 Verdict-gated tiering (primary)

Applied where the per-sentence band → highlight label/color is decided (`poc/report/report.py::_deberta_primary_signal`, using `poc/report/deberta.py` band metadata):

| Sentence desklib score | Doc verdict **non-green** (amber/orange/red) | Doc verdict **green** |
|---|---|---|
| **≥ 0.999** (`high` band) | **"High-confidence AI"** — red `#dc2626`, tier `high` (unchanged) | **"Possible AI — review"** — muted amber/slate, tier `low`, framed as a review candidate, NOT a verdict |
| **< 0.999** (`clean`) | no highlight | no highlight |

- **"Doc verdict" = the fused badge tier `report.ai_risk_badge["tier"]` — NOT the deep-scan proportion.** (Fable 5 review, resolves §8.) `DOC_FLOOR` is **display-only**: `builder.py:1363` passes the RAW proportion into `compute_fused_authority`, and `below_floor` never enters the verdict math (`pipeline_bridge.py:257-285`). Gating highlights on `proportion vs DOC_FLOOR` would elevate a display floor into a *second* verdict boundary and re-create this exact bug (a doc green-by-fused-math with proportion 0.35 would paint a red wall under a green headline). The tier is always present and is the one verdict the user sees; the proportion is not. So gate on the tier.
- **Rationale (Bayesian):** at 0.999, ~24% of human sentences cross (desklib saturation), so the per-sentence "≥99% confident" label is uncalibrated at sentence granularity — sentence PPV depends on the document prior. On a green doc the posterior for a crossing sentence is low, so a muted "Possible — review" is the *honest* reading, not hiding: the sentences stay listed for the user. Residual risk (a few genuine AI sentences in a mostly-human, proportion-~0.1 doc stay green+muted) is acceptable and matches `DOC_FLOOR`'s "insufficient evidence, not a verdict" philosophy.

### 3.2 Single presentation source (CRITICAL — Fable 5)

**The gate MUST be applied once, at the heatmap-row level, and every consumer must read the resulting band/color — never the raw score.** Today several surfaces color by RAW per-sentence score independently, so fixing only the band leaves the red wall intact where users actually see it (PDF/page). Therefore:

- Apply the green→muted remap **on the heatmap rows** inside `poc/report/report.py::_compute_deberta_heatmap` (after the source is chosen), so `_document_segments`, the badge tile, the renderers, and the paragraph explainer all inherit ONE gated result.
- Audit every score-keyed consumer and route it through the band/color, not the raw score (see §4).

### 3.3 Fallback fix (secondary)

When the deep-scan heatmap is unavailable and the code falls back to the fakespot map:
- **Re-map (do NOT delete) the saturated `moderate` band** so it no longer renders as an AI highlight, while its rows still exist for the headline denominator (deleting rows shrinks `_sync_deberta_headline_from_heatmap`'s denominator and distorts the tile — Fable 5). Only the fakespot `high` (≥0.99) band highlights, subject to the SAME §3.1 verdict gate.
- Record `signal_highlight_source = "fakespot_fallback"` (already tracked via `heatmap_source`) and surface a small "deep-scan unavailable — showing reduced highlights" note so a degraded run is never silently presented as authoritative.

### 3.4 New band metadata

Add one muted presentation band in `poc/report/deberta.py` (e.g. `review` / `possible`): slate or muted-amber color, label "Possible AI — review", tier `low`, description that frames it as a review candidate (grounding-first copy, consistent with the existing V7-aligned recommendations). The green-doc `high` sentences remap to this band.

## 4. Seams (exact edit sites)

**The gate is applied once (§3.2) in `_compute_deberta_heatmap`; every row below reads the band, never the raw score.**

| File / function | Change |
|---|---|
| `poc/report/report.py::_compute_deberta_heatmap` (~1352) | **Primary gate site.** After the heatmap source is chosen, remap each row's band by the doc's fused badge tier: green + `high` → muted `review`; non-green + `high` → `high` (unchanged). On fakespot fallback, remap `moderate` → non-highlighting (keep the row, §3.3). |
| `poc/report/deberta.py` | Add the muted `review`/`possible` band to `DEBERTA_HEAT_{COLORS,TIERS,LABELS,DESCRIPTIONS,RECOMMENDATIONS,READER_SUMMARY}`. |
| `poc/report/report.py::_deberta_primary_signal` (~1412) | Reads the (already-gated) row band — no separate verdict logic. |
| `poc/report/report.py::_document_segments` (~1437) | Unchanged (consumes gated rows). |
| **`poc/report/report.py::_sync_deberta_headline_from_heatmap` (~1387)** | **MISSED SURFACE (Fable 5):** rebuilds the badge `ai_signal_deberta` tile via `_build_headline` (0.99 flags) → tile still says "High AI-writing signal" on a green doc. Must consume the gated rows so the tile agrees with the map. |
| **`poc/report/render_panels.py`** | **MISSED SURFACE (Fable 5):** colors by RAW score, not band — `_DEBERTA_SEVERITY_COLORS` (~803), `_severity_color` (~821), `render_signal_highlights_intro` (~901), `render_highlighted_document` (~933), flagged-para chip (~860). Route ALL of these through the gated band/color, or the **PDF + page keep the red wall**. |
| **`poc/report/paragraph_explainer.py`** | **MISSED SURFACE (Fable 5):** score-keyed too — route through the gated band. |
| **`poc/report/render.py` (the ACTIVE PDF renderer, `render_report` @ worker/app/tasks.py:225)** | **MISSED SURFACE (Fable 5, post-impl review):** `render_panels.py` is only *part* of the PDF — `render.py` has its OWN score-keyed card path. `_deberta_paragraph_groups` (~1615) built the finding-card `title`/`tier`/`adjusted_risk` from raw `score >= 99`, and `_render_finding_card` (~1300) colored the FLAGGED-SENTENCES chip `#dc2626` from raw score → green docs kept a red wall on the finding cards. **Fix:** both now read the gated band from the segment signal's `title="deberta_<band>"`. Plus a display-only findings-tree stamp in `render_report` (~1646) so the degraded fallback (`_paragraph_finding_groups`, fed by `builder.py`'s ungated Tier.HIGH synthesis @ ~879) also mutes on green — no tier re-bucketing, so counts/scores are untouched. |
| `poc/detect_v7/deep_scan_heatmap.py` | No change (already single-band ≥0.999). |
| Verification | **RENDER + visually inspect PDF + page** for green & non-green docs (grep proves emit, not render). |

## 5. Non-goals

- Do **not** change the document AI score / fused authority / `poc/detect/` scoring (out of scope; already correct).
- Do **not** add a new detector or a new uncalibrated sentence threshold. The only thresholds are the existing calibrated `0.999` and the existing fused/`DOC_FLOOR` verdict boundaries.
- Do **not** touch `get_deep_scan_proportion` calibration/floor math.

## 6. Validation (before merge)

1. **ESL false-positive on highlights (primary success metric):** over the SCoCESLE human corpus, count sentences highlighted per essay under (a) current, (b) new gated design. Expectation: **red "High-confidence AI" highlights on green human docs → ~0**; muted "Possible — review" replaces them. Report the distribution; do not silently cap.
2. **Genuine-AI retention:** on the AI corpus (and any non-green docs), confirm ≥0.999 sentences still render as bold red "High-confidence AI" (the gate must NOT suppress highlights on non-green docs).
3. **Fallback path:** force a deep-scan failure (unset Modal URL locally) and confirm the fakespot fallback highlights `high` only, with the degraded-source note, and no `moderate` red-wall.
4. **Render parity:** regenerate a report PDF + page for one green and one amber/red doc; visually confirm the highlight tiers match and the green doc shows no red "High-confidence AI" sentences.

## 7. Rollout

- Behind the existing highlight path (no new global flag needed); the change is presentation-layer and additive. If a kill switch is wanted, gate the §3.1 remap on an env flag defaulting ON.
- No worker redeploy churn beyond the normal `poc/` change; no Modal cost change (same calls).

## 8. Resolved open question

**Q: Downgrade genuine ≥0.999 sentences to "Possible — review" on green docs — right, or hides real AI?**
**A (Fable 5): Correct, keep it.** Sentence "≥99% confident" is uncalibrated at sentence granularity (24% human cross-rate); PPV depends on the doc prior; a green doc gives a low posterior, so muted is the honest reading and nothing is hidden (sentences stay listed). Residual risk accepted.

**Q: Gate on fused tier, or on proportion vs DOC_FLOOR?**
**A (Fable 5): Fused tier.** `DOC_FLOOR` is display-only and never enters verdict math; gating on proportion would create a second verdict boundary and re-create the bug. Folded into §3.1.

## 9. Fable 5 review record (APPROVE-WITH-CHANGES)

Changes incorporated:
1. Gate on **fused badge tier**, not proportion (§3.1) — with the `DOC_FLOOR`-is-display-only evidence.
2. **Single presentation source:** apply the gate on heatmap rows in `_compute_deberta_heatmap` so all consumers inherit it (§3.2).
3. **Remap, don't delete** the fakespot `moderate` rows — deletion distorts the tile denominator (§3.3).
4. **Missed surfaces added to §4:** `_sync_deberta_headline_from_heatmap` (badge tile), `render_panels.py` (raw-score coloring — PDF/page), `paragraph_explainer.py`. **This was the critical catch — fixing only the segment band would have left the red wall on the PDF/page where users reported it.**

**Single most important implementation rule:** every score-keyed highlight consumer must read the gated band/color — one presentation source — or the green doc keeps its red wall.

## 10. Fable 5 post-implementation review record (verdict: REVISE → fixed)

Reviewed commit `6501c504`. Verdict **REVISE** — one confirmed red-wall survivor:

- **Critical catch:** the ACTIVE PDF renderer is `poc/report/render.py::render_report` (worker/app/tasks.py:225), which calls the (already-gated) `render_panels.py` functions **but also has its own** score-keyed finding-card path (`_deberta_paragraph_groups` + `_render_finding_card`) that §4 never enumerated. Green docs kept a red "High-confidence AI" wall on the PDF finding cards. **Fixed** (band-driven; validated green→muted / red→red via `_deberta_paragraph_groups` + `_render_finding_card` unit test).
- Confirmed PASS by review: primary gate reads the right fused badge tier; `{green, clean}` set correct (`acceptable`/word-tiers never occur on the badge); headline/tile denominator is score-based (unaffected); frontend band-parse correct.

**Recorded decisions (deliberate residuals, not defects):**
1. **Green-doc PDF shows muted "Possible AI — review" cards**, not an empty findings section. Matches the user's "Tiered — both, honestly labeled" choice; the spec requires no *red*, not no *cards*.
2. **Masthead finding COUNT (`n_high`) is not re-bucketed** on green docs (keeps `findings_by_tier`/scores untouched, per §5). A green doc can still show a non-zero high-finding count in the masthead while the map/cards are muted — a display-count coherence item deferred to avoid entangling the fix with scoring. Documented, not silently dropped.
3. **Degraded fallback** (segments absent → `_paragraph_finding_groups`): sentence chips mute via the findings-tree band stamp, but the card *header icon* still reads the ungated tier. Rare path (heatmap failure); acceptable residual.
4. Minor: stale legend copy in `report.js` (i18n en/zh, "strong" band) — cosmetic, out of scope.
