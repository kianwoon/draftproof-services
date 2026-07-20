"""Markdown → PDF renderer using WeasyPrint."""

import re

import markdown as md_lib
from weasyprint import HTML


_CSS = """
@page {
    size: A4;
    margin: 2cm 2.5cm;
    @bottom-center {
        content: "DraftProof Report | page " counter(page) " of " counter(pages);
        font-size: 8pt;
        color: #999;
    }
}

/* ── Design tokens — brand-mapped (KEEP-IN-SYNC with the web foundation
   00-foundation.css + the --sev-* severity scale; WeasyPrint has no color-mix,
   so the tint values below are pre-resolved). ─── */
:root {
    --dp-ink: #0D1B2A;        /* navy-950 */
    --dp-muted: #7A8694;      /* brand muted */
    --dp-line: #E4E2DD;       /* brand hairline (rgba(13,27,42,.10) on white) */
    --dp-bg-soft: #F4F2EE;    /* surface-soft */
    --dp-critical: #B3261E;   /* --sev-crit */
    --dp-high: #C2410C;       /* --sev-high */
    --dp-medium: #B45309;     /* --sev-med */
    --dp-low: #0F6E56;        /* --sev-low */
    --dp-human: #2F8C61;      /* brand green-600 */
    --dp-ai: #C2410C;
    /* Severity tint backgrounds (pre-resolved sev-*-bg mixed toward white). */
    --dp-crit-bg: #FCF0EE;
    --dp-high-bg: #FBEADF;
    --dp-med-bg: #FCF5E9;
    --dp-low-bg: #EEF7F3;
    --dp-neutral: #5F5E5A;
    --dp-neutral-bg: #F1F0EC;
    --dp-serif: Georgia, "Times New Roman", "DejaVu Serif", serif;
}

body {
    font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #222;
}

h1 { font-family: var(--dp-serif); font-size: 18pt; font-weight: 700; letter-spacing: -.01em; color: var(--dp-ink); border-bottom: 2px solid var(--dp-line); padding-bottom: 4pt; }
/* break-after: avoid-page keeps a section heading with its first block — prevents an
   orphaned "## Rewritten Content" alone at the bottom of an otherwise-empty page when
   the following block starts on the next page. */
h2 { font-family: var(--dp-serif); font-size: 14pt; font-weight: 700; letter-spacing: -.01em; color: var(--dp-ink); border-bottom: 1px solid var(--dp-line); padding-bottom: 3pt;
     break-after: avoid-page; page-break-after: avoid; }
h3 { font-family: var(--dp-serif); font-size: 12pt; font-weight: 700; color: var(--dp-ink); break-after: avoid-page; page-break-after: avoid; }

table {
    width: 100%;
    table-layout: auto;
    border-collapse: collapse;
    margin: 8pt 0;
    font-size: 9pt;
}
th {
    background: var(--dp-bg-soft);
    color: var(--dp-ink);
    font-weight: 700;
    text-align: left;
    padding: 5pt 7pt;
    border: 1px solid var(--dp-line);
    overflow-wrap: break-word;
    word-break: break-word;
}
td {
    padding: 4pt 7pt;
    border: 1px solid var(--dp-line);
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: break-word;
    word-break: break-word;
    -webkit-hyphens: auto;
    hyphens: auto;
}

code {
    background: #f4f4f4;
    padding: 1pt 3pt;
    border-radius: 2pt;
    font-size: 9pt;
    font-family: "SF Mono", "Menlo", "Monaco", monospace;
}

mark.placeholder {
    background: #fef08a;
    color: #1f2937;
    padding: 0 2pt;
    border-radius: 2pt;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}

pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 3pt;
    padding: 8pt 10pt;
    margin: 6pt 0;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    line-height: 1.4;
}

pre code {
    background: none;
    padding: 0;
    border-radius: 0;
    font-size: 8.5pt;
}

blockquote {
    border-left: 3px solid #ccc;
    margin: 8pt 0;
    padding: 4pt 12pt;
    color: #555;
    background: #fafafa;
}

hr { border: none; border-top: 1px solid #ddd; margin: 12pt 0; }

img[src^="https://img.shields.io"] { height: 18px; }

/* ── Masthead (shared by scan + rewrite reports) ───────────────── */
.dp-masthead {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    margin: 0 0 16pt;
    padding: 2pt 0 10pt;
    border-bottom: 2.5pt solid var(--dp-ink);
}

.dp-masthead-title {
    color: var(--dp-ink);
    font-size: 20pt;
    font-weight: 800;
    line-height: 1;
    letter-spacing: .01em;
}

.dp-masthead-sub {
    display: block;
    margin-top: 4pt;
    color: var(--dp-muted);
    font-size: 7.5pt;
    font-weight: 800;
    letter-spacing: .14em;
    text-transform: uppercase;
}

.dp-masthead-meta {
    color: var(--dp-muted);
    font-size: 7.5pt;
    font-weight: 800;
    letter-spacing: .1em;
    text-transform: uppercase;
    white-space: nowrap;
}

.dp-rewrite-outcome-panel {
    position: relative;
    margin: 10pt 0 16pt;
    padding: 12pt;
    border: 1px solid #cbd5e1;
    border-radius: 8pt;
    background: #f8fafc;
    page-break-inside: avoid;
}

.dp-rewrite-stamp {
    float: right;
    width: 128pt;
    margin: 0 0 9pt 12pt;
    padding: 7pt;
    border: 1.5px solid #475569;
    border-radius: 8pt;
    text-align: right;
    background: #f1f5f9;
    box-shadow: inset 0 0 0 3pt #e2e8f0;
}

.dp-rewrite-stamp span {
    display: block;
    color: #475569;
    font-size: 6.5pt;
    font-weight: 900;
    letter-spacing: .15em;
    text-transform: uppercase;
}

.dp-rewrite-stamp strong {
    display: block;
    color: inherit;
    /* Sized so the longest verdict word ("DETECTORS") fits the 128pt seal; wrap whole words. */
    font-size: 15pt;
    line-height: 1.06;
    overflow-wrap: break-word;
    word-break: normal;
}

.dp-rewrite-stamp em {
    display: block;
    color: #334155;
    font-size: 8pt;
    font-style: normal;
    font-weight: 800;
}

.dp-rewrite-scan-summary span {
    color: #475569;
    font-size: 7pt;
    font-weight: 900;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.dp-rewrite-scan-summary h3 {
    margin: 2pt 0 3pt;
    color: #111827;
    font-size: 13pt;
}

.dp-rewrite-scan-summary b {
    color: #92400e;
    font-size: 14pt;
}

.dp-rewrite-scan-summary p {
    margin: 5pt 0 7pt;
    color: #1f2937;
    font-weight: 700;
}

.dp-rewrite-scan-summary .dp-ai-reference-note {
    margin: 4pt 0 7pt;
    color: #64748b;
    font-size: 7pt;
    font-weight: 700;
    line-height: 1.4;
}

.dp-outcome-chips span {
    display: inline-block;
    margin: 0 4pt 4pt 0;
    padding: 3pt 5pt;
    border-radius: 9pt;
    background: #e2e8f0;
    color: #1f2937;
    font-size: 7pt;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: none;
}

.dp-outcome-bars {
    clear: both;
    margin-top: 8pt;
}

.dp-outcome-bars div {
    display: flex;
    justify-content: space-between;
    color: #1f2937;
    font-size: 8pt;
    font-weight: 800;
}

.dp-outcome-bars em {
    font-style: normal;
}

.dp-outcome-bar {
    display: block;
    height: 5pt;
    margin: 2pt 0 5pt;
    border-radius: 999pt;
    background: #e2e8f0;
    overflow: hidden;
}

.dp-outcome-bar b {
    display: block;
    height: 100%;
    border-radius: 999pt;
}

.dp-executive-chart {
    margin: 4pt 0 12pt;
}

/* ── At-a-glance severity strip (scan executive summary) ───────── */
.dp-glance {
    display: flex;
    flex-wrap: wrap;
    gap: 6pt;
    margin: 2pt 0 12pt;
}

.dp-glance span {
    display: inline-block;
    padding: 4pt 8pt;
    border-radius: 6pt;
    border: 1px solid var(--dp-line);
    background: var(--dp-bg-soft);
    color: var(--dp-ink);
    font-size: 8pt;
    font-weight: 800;
}

.dp-glance span b {
    font-size: 10pt;
    font-weight: 900;
}

.dp-glance .dp-glance--critical { border-color: var(--dp-critical); color: var(--dp-critical); }
.dp-glance .dp-glance--high { border-color: var(--dp-high); color: var(--dp-high); }
.dp-glance .dp-glance--medium { border-color: var(--dp-medium); color: var(--dp-medium); }
.dp-glance .dp-glance--low { border-color: var(--dp-low); color: var(--dp-low); }

/* ════════════ Enhanced report design system ════════════ */

.dp-section-intro {
    margin: -2pt 0 9pt;
    color: var(--dp-muted);
    font-size: 9pt;
    line-height: 1.45;
}

/* ── Hero "overall read" panel ── */
.dp-hero {
    margin: 2pt 0 13pt;
    padding: 13pt 15pt;
    border: 1px solid var(--dp-line);
    border-left: 4pt solid var(--dp-low);
    border-radius: 8pt;
    background: #fff;
    page-break-inside: avoid;
}
.dp-hero--good { border-left-color: var(--dp-human); }
.dp-hero--warn { border-left-color: var(--dp-high); }
.dp-hero--info { border-left-color: var(--dp-low); }
.dp-hero-read {
    margin: 0;
    color: var(--dp-ink);
    font-size: 13pt;
    font-weight: 700;
    line-height: 1.3;
}
/* The fused score reads as a commanding serif number (mirrors the web hero). */
.dp-hero-read b {
    font-family: var(--dp-serif);
    font-size: 26pt;
    font-weight: 700;
    letter-spacing: -.02em;
    color: var(--dp-ink);
}
.dp-hero-sub {
    margin: 6pt 0 0;
    color: var(--dp-muted);
    font-size: 9.5pt;
    line-height: 1.5;
}
/* Authorship-evidence-level lens ANCHOR (PDF) — the specific finding + a real
   flagged sentence + a fix direction under each band. Mirrors the web
   .merged-ael-anchor-* rules (KEEP-IN-SYNC with 06-report-overview.css); muted
   9pt to sit quietly under the band chip. */
.dp-ael-anchor {
    margin: 3pt 0 4pt;
    line-height: 1.5;
}
.dp-ael-anchor-head {
    color: var(--dp-ink);
    font-size: 9pt;
    font-weight: 700;
}
.dp-ael-anchor-eg {
    margin-top: 2pt;
    padding-left: 7pt;
    border-left: 1.5pt solid var(--dp-line);
    color: var(--dp-muted);
    font-size: 9pt;
    font-style: italic;
}
.dp-ael-anchor-fix {
    margin-top: 2pt;
    color: var(--dp-low);
    font-size: 9pt;
    font-weight: 700;
}
.dp-ael-anchor-fix::before { content: "→ "; }
/* AEL lens table — de-gridded into clean rows: dimension left (bold navy), band
   chip + finding right, hairline separators. Mirrors the web card list without
   WeasyPrint card overhead. */
.dp-ael-table { margin: 8pt 0 0; }
.dp-ael-table td { border: 0; border-bottom: 1px solid var(--dp-line); padding: 7pt 2pt; vertical-align: top; }
.dp-ael-table tr:last-child td { border-bottom: 0; }
.dp-ael-table td:first-child { width: 34%; font-weight: 700; color: var(--dp-ink); }
.dp-chip-strip {
    margin-top: 10pt;
}
/* Owner-mandated framing: detection tier = AI-risk authority (vs the composition
   breakdown below it = a display interpretation). KEEP-IN-SYNC:
   draftproof-frontend/src/styles/site-master/06-report-overview.css .merged-verdict-framing. */
.dp-verdict-framing {
    color: var(--dp-low);
    font-weight: 700;
}
.dp-statchip {
    display: inline-block;
    margin: 0 4pt 4pt 0;
    padding: 3pt 8pt;
    border: 1px solid var(--dp-line);
    border-radius: 999pt;
    background: var(--dp-bg-soft);
    color: var(--dp-muted);
    font-size: 6.8pt;
    font-weight: 800;
    letter-spacing: .07em;
    text-transform: uppercase;
}
.dp-statchip--good { color: var(--dp-low); background: var(--dp-low-bg); }
.dp-statchip--info { color: var(--dp-neutral); background: var(--dp-neutral-bg); }
.dp-statchip--warn { color: var(--dp-medium); background: var(--dp-med-bg); }
.dp-statchip--high { color: var(--dp-high); background: var(--dp-high-bg); }
.dp-statchip--crit { color: var(--dp-critical); background: var(--dp-crit-bg); }

/* ── Authorship clarity breakdown (4 category bars) — mirrors the web page's
   authorship-breakdown rows so the PDF is not just jammed label+share text. ── */
.dp-abd-bars {
    display: flex;
    flex-direction: column;
    gap: 5pt;
    margin: 9pt 0 4pt;
}
.dp-abd-row {
    display: block;
}
.dp-abd-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 3pt;
}
.dp-abd-label {
    font-size: 8.5pt;
    font-weight: 700;
    color: var(--dp-ink);
}
.dp-abd-bar-track {
    display: block;
    width: 100%;
    height: 6pt;
    border-radius: 3pt;
    background: var(--dp-bg-soft);
    overflow: hidden;
}
.dp-abd-bar-fill {
    display: block;
    height: 100%;
    border-radius: 3pt;
    background: var(--dp-human);
}
.dp-abd-band {
    white-space: nowrap;
    font-size: 8pt;
    font-weight: 700;
    color: var(--dp-muted);
}
.dp-abd-fill--student_owned { background: #1D9E75; }
.dp-abd-fill--ai_assisted_polished { background: #888780; }
.dp-abd-fill--ai_paraphrased { background: #D85A30; }
.dp-abd-fill--ai_generated_like { background: #D85A30; }
/* V8 three-way display fallback merged category — same fill as ai_paraphrased/ai_generated_like
   above, both of which it merges. */
.dp-abd-fill--ai_transformed { background: #D85A30; }

/* Tier-consistency guard caveat (poc/detect_v7/pipeline_bridge.py::_apply_tier_consistency_guard) —
   shown when breakdown.presentation === "mixed_signals". Same amber palette as .dp-statchip--warn.
   KEEP-IN-SYNC: draftproof-frontend/src/styles/site-master/06-report-overview.css .merged-comp-caveat. */
.dp-abd-caveat {
    margin: 0 0 6pt;
    padding: 4pt 7pt;
    border-radius: 4pt;
    background: var(--dp-med-bg);
    color: var(--dp-medium);
    font-size: 8pt;
    line-height: 1.4;
}

/* V8 three-way display fallback explainer — neutral/informational, not a warning.
   KEEP-IN-SYNC: draftproof-frontend/src/styles/site-master/06-report-overview.css
   .merged-comp-caveat--info. */
.dp-abd-caveat--info {
    background: var(--dp-bg-soft);
    color: var(--dp-muted);
}

/* ── "How this score is built" mini-bars (owner redesign 2026-07-16) — composite /
   deep-scan / fused, with a flag-line marker on the fused bar. Mirrors the web's
   .merged-scorebar-* ; reuses the verdict tier hexes, no new colour. Two-row layout
   (head line, then full-width track) — a flex:1 track collapses in WeasyPrint. ── */
.dp-sb { margin: 8pt 0 2pt; }
.dp-sb-head { margin: 0 0 5pt; font-size: 8.5pt; font-weight: 700; color: var(--dp-muted); }
.dp-sb-row { display: block; margin-bottom: 5pt; }
.dp-sb-head-line { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 3pt; }
.dp-sb-label { font-size: 8.5pt; font-weight: 700; color: var(--dp-ink); }
.dp-sb-val { font-family: var(--dp-serif); font-size: 9pt; font-weight: 700; color: var(--dp-ink); }
.dp-sb-track { display: block; position: relative; width: 100%; height: 6pt; border-radius: 3pt; background: var(--dp-bg-soft); overflow: hidden; }
.dp-sb-fill { display: block; height: 100%; border-radius: 3pt; background: #B7BEC7; }
.dp-sb-fill--muted { background: #B7BEC7; }
.dp-sb-fill--green { background: var(--dp-low); }
.dp-sb-fill--amber { background: var(--dp-medium); }
.dp-sb-fill--orange { background: var(--dp-high); }
.dp-sb-fill--red { background: var(--dp-critical); }
.dp-sb-marker { position: absolute; top: 0; bottom: 0; width: 1.5pt; margin-left: -0.75pt; background: var(--dp-ink); }
.dp-sb-cap { margin: 2pt 0 0; font-size: 7.5pt; color: var(--dp-muted); }

/* ── KPI stat row ── */
.dp-kpi-row {
    display: flex;
    gap: 8pt;
    margin: 4pt 0 14pt;
}
.dp-kpi {
    flex: 1;
    padding: 11pt 6pt;
    border: 1px solid var(--dp-line);
    border-radius: 8pt;
    background: #fff;
    text-align: center;
}
.dp-kpi b {
    display: block;
    color: var(--dp-ink);
    font-size: 19pt;
    font-weight: 800;
    line-height: 1;
}
.dp-kpi span {
    display: block;
    margin-top: 5pt;
    color: var(--dp-muted);
    font-size: 7pt;
    font-weight: 700;
}

/* Full-document highlighted blocks (Submitted/Rewritten Content) are usually taller
   than a page — let them break naturally so they start right after their heading
   instead of forcing a mostly-empty page. */
.dp-doc-flow .dp-hero { page-break-inside: auto; break-inside: auto; }

/* ── Rewrite exec-summary two-column comparison (Original vs Rewritten) ──
   display:table for reliable equal-width, top-aligned columns in WeasyPrint. */
.dp-cmp-grid {
    display: table;
    width: 100%;
    border-spacing: 8pt 0;
    margin: 4pt 0 12pt;
    table-layout: fixed;
}
.dp-cmp-cell {
    display: table-cell;
    width: 50%;
    vertical-align: top;
}
.dp-cmp-col {
    padding: 10pt 11pt;
    border: 1px solid var(--dp-line);
    border-radius: 8pt;
    background: #fff;
    /* When the column is taller than one page it must fragment (the two-column
       scan card is genuinely > 1 page). box-decoration-break: clone makes each
       page fragment carry its own complete border+padding+radius, so the risk
       cards that land on page 2 stay inside a bordered box instead of orphaning
       (WeasyPrint honors this on block children of table-cells). */
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}
/* Keep each atomic card whole across the page break — never split a fused hero
   card, a risk row, or the writing-reads bar group down the middle. */
.dp-cmp-col .dp-hero,
.dp-cmp-col .dp-policy-row,
.dp-cmp-col .dp-abd-bars,
.dp-cmp-col .dp-abd-row { break-inside: avoid; page-break-inside: avoid; }
/* A subhead ("Where the risk sits") must not be the last thing on a page. */
.dp-cmp-subhead { break-after: avoid; page-break-after: avoid; }
/* Keep the whole "Where the risk sits" section (heading + all rows) together so
   pagination breaks BEFORE it instead of splitting the risk list mid-way. */
.dp-cmp-col .dp-cmp-risk-group { break-inside: avoid; page-break-inside: avoid; }
/* Compact colored risk bands — mirror the web's .merged-axis (06-report-overview.css):
   label left, level word right, full-row tint by level. Far more compact than the old
   bordered chip cards, so the whole comparison card now fits without an ugly split. */
.dp-cmp-axis {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 8pt;
    padding: 3.5pt 6pt;
    margin: 3pt 0;
    border-radius: 4pt;
    background: var(--dp-bg-soft);
    color: var(--dp-muted);
    break-inside: avoid;
}
.dp-cmp-axis-name { font-weight: 600; }
.dp-cmp-axis-level { font-weight: 700; }
.dp-cmp-axis--low { background: var(--dp-low-bg); color: var(--dp-low); }
.dp-cmp-axis--medium { background: var(--dp-med-bg); color: var(--dp-medium); }
.dp-cmp-axis--high,
.dp-cmp-axis--critical { background: var(--dp-crit-bg); color: var(--dp-critical); }
.dp-cmp-kicker {
    margin: 0 0 3pt;
    color: var(--dp-muted);
    font-size: 7.5pt;
    font-weight: 800;
    letter-spacing: 0.06em;
}
.dp-cmp-pattern {
    margin: 0 0 6pt;
    color: var(--dp-ink);
    font-size: 10.5pt;
    font-weight: 800;
}
.dp-cmp-subhead {
    margin: 10pt 0 4pt;
    color: var(--dp-ink);
    font-size: 8.5pt;
    font-weight: 800;
}
.dp-cmp-col .dp-hero { margin: 8pt 0; }
.dp-cmp-col .dp-policy-row { margin: 3pt 0; }

/* ── Callout panels (colored left rail + tint) ── */
.dp-callout {
    margin: 10pt 0 12pt;
    padding: 10pt 13pt;
    border-left: 3pt solid var(--dp-line);
    border-radius: 0 7pt 7pt 0;
    background: var(--dp-bg-soft);
    page-break-inside: avoid;
}
.dp-callout--good { border-left-color: #16a34a; background: #ecfdf5; }
.dp-callout--warn { border-left-color: #ea580c; background: #fff7ed; }
.dp-callout--info { border-left-color: #2563eb; background: #eff6ff; }
.dp-callout--think { border-left-color: #7c3aed; background: #f5f3ff; }
.dp-callout-title {
    display: block;
    margin-bottom: 4pt;
    color: var(--dp-ink);
    font-family: var(--dp-serif);
    font-size: 11.5pt;
    font-weight: 700;
    letter-spacing: -.01em;
}
.dp-callout p { margin: 4pt 0 0; color: var(--dp-ink); font-size: 9pt; line-height: 1.5; }
.dp-callout ol, .dp-callout ul { margin: 4pt 0 0 15pt; padding: 0; }
.dp-callout li { margin: 4pt 0; color: var(--dp-ink); font-size: 9pt; line-height: 1.45; }

/* ── Policy row-cards ── */
.dp-policy-head {
    display: flex;
    gap: 10pt;
    margin: 8pt 0 2pt;
    color: var(--dp-muted);
    font-size: 6.8pt;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
}
.dp-policy-head .dp-ph-name { flex: 5; }
.dp-policy-head .dp-ph-issue { flex: 9; text-align: right; }
.dp-policy-row {
    display: flex;
    align-items: center;
    gap: 10pt;
    margin: 6pt 0;
    padding: 10pt 13pt;
    border: 1px solid var(--dp-line);
    border-radius: 8pt;
    background: #fff;
    page-break-inside: avoid;
}
.dp-policy-row--warn { background: #fff7ed; border-color: #fed7aa; }
.dp-policy-row--good { background: #ecfdf5; border-color: #bbf7d0; }
.dp-policy-name { flex: 5; color: var(--dp-ink); font-size: 10pt; font-weight: 800; }
.dp-policy-level { flex: 3; font-size: 11pt; font-weight: 900; color: var(--dp-muted); }
.dp-policy-row--warn .dp-policy-level { color: #c2410c; }
.dp-policy-row--good .dp-policy-level { color: #15803d; }
.dp-policy-issue { flex: 6; color: #6b7280; font-size: 8.5pt; }
/* Ordering-floor disclosure under the policy rows (KEEP-IN-SYNC with the web
   card's .policy-risk-floor-note) — informational, muted, not a risk accent. */
.dp-policy-floor-note { margin: 2pt 0 4pt; color: #64748b; font-size: 8pt; font-style: italic; }

/* ── Question cards (purple) ── */
.dp-q-card {
    margin: 6pt 0;
    padding: 9pt 12pt;
    border: 1px solid var(--dp-line);
    border-left: 3pt solid #7c3aed;
    border-radius: 0 7pt 7pt 0;
    background: #fff;
    page-break-inside: avoid;
}
.dp-q-claim { margin: 0 0 4pt; color: var(--dp-ink); font-size: 9.5pt; font-weight: 800; line-height: 1.35; }
.dp-q-claim em { font-style: italic; }
.dp-q-body { margin: 0; color: #374151; font-size: 9pt; line-height: 1.5; }
.dp-q-target { margin: 5pt 0 0; color: #6b7280; font-size: 7.5pt; font-style: italic; }

.dp-signal-card {
    border: 1px solid #d8e1ea;
    border-radius: 8pt;
    background: #fff;
    overflow: visible;
    break-inside: auto;
}

.dp-signal-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12pt;
    padding: 13pt;
    border-bottom: 1px solid #e2e8f0;
}

.dp-signal-title-row {
    display: flex;
    gap: 10pt;
    align-items: flex-start;
}

.dp-signal-icon {
    display: inline-block;
    width: 32pt;
    height: 32pt;
    color: #0f766e;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 7pt;
    font-size: 18pt;
    line-height: 31pt;
    text-align: center;
}

.dp-kicker,
.dp-ai-score span,
.dp-scan-head span,
.dp-ratio-copy span {
    display: block;
    color: #64748b;
    font-size: 6.8pt;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.dp-method-note {
    margin: 4pt 0 0;
    color: #64748b;
    font-size: 7pt;
    line-height: 1.4;
    max-width: 60ch;
}

.dp-signal-header h3 {
    margin: 3pt 0 7pt;
    color: #0f172a;
    font-size: 16pt;
    font-weight: 800;
    line-height: 1.15;
}

.dp-pill-row,
.dp-chip-row,
.dp-evidence-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4pt;
}

.dp-pill-row span,
.dp-chip-row strong,
.dp-evidence-row span {
    display: inline-block;
    min-height: 13pt;
    padding: 3pt 5pt;
    border-radius: 5pt;
    font-size: 6.8pt;
    font-weight: 800;
}

.dp-pill-row span {
    color: #0f766e;
    background: #f0fdfa;
    border: 1px solid #ccfbf1;
    text-transform: capitalize;
}

.dp-ai-score {
    min-width: 72pt;
    text-align: right;
}

.dp-ai-score strong {
    display: block;
    margin-top: 4pt;
    color: #b45309;
    font-size: 16pt;
    font-weight: 900;
    line-height: 1;
}

.dp-rating-seal {
    position: relative;
    min-width: 104pt;
    max-width: 142pt;
    min-height: 54pt;
    padding: 9pt 10pt;
    border: 2pt solid #b45309;
    border-radius: 7pt;
    text-align: right;
}

.dp-rating-seal span {
    display: block;
    color: #64748b;
    font-size: 6.5pt;
    font-weight: 800;
    letter-spacing: .08em;
    line-height: 1.1;
    text-transform: uppercase;
}

.dp-rating-seal strong {
    display: block;
    margin-top: 3pt;
    color: inherit;
    font-size: 18pt;
    font-weight: 900;
    line-height: .98;
    text-transform: uppercase;
    overflow-wrap: anywhere;
}

.dp-rating-seal em {
    display: block;
    margin-top: 4pt;
    color: #475569;
    font-size: 7.3pt;
    font-style: normal;
    font-weight: 800;
}

.dp-ai-reference-note {
    margin: 8pt 13pt 0;
    color: #64748b;
    font-size: 7.5pt;
    font-weight: 700;
    line-height: 1.45;
}

.dp-scan-head {
    display: flex;
    justify-content: space-between;
    gap: 12pt;
    padding: 12pt 13pt 6pt;
}

.dp-scan-head strong {
    display: block;
    margin-top: 3pt;
    color: #0f172a;
    font-size: 10.5pt;
    font-weight: 800;
}

.dp-scan-head em {
    color: #b45309;
    font-size: 12pt;
    font-style: normal;
    font-weight: 900;
}

.dp-ratio-card {
    display: block;
    margin: 4pt 13pt 12pt;
    padding: 10pt;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 7pt;
}

.dp-ratio-copy p {
    margin: 4pt 0 7pt;
    color: #1f2937;
    font-size: 9pt;
    font-weight: 650;
    line-height: 1.4;
}

.dp-chip-row strong {
    color: #334155;
    background: #fff;
    border: 1px solid #dbe4ea;
}

.dp-ratio-bars {
    display: block;
    margin-top: 8pt;
}

.dp-ratio-row {
    display: block;
    margin-bottom: 7pt;
    break-inside: avoid;
}

.dp-ratio-row > span,
.dp-ratio-row > strong {
    font-size: 7.8pt;
    font-weight: 800;
}

.dp-ratio-row > span {
    color: #475569;
}

.dp-ratio-row > strong {
    float: right;
    color: #0f172a;
}

.dp-bar-track {
    clear: both;
    height: 6pt;
    margin-top: 3pt;
    overflow: hidden;
    border-radius: 999pt;
    background: #e8eef5;
}

.dp-ratio-fill {
    height: 100%;
    min-width: 2pt;
    border-radius: inherit;
}

.dp-human { background: #16a34a; }
.dp-ai { background: #c2410c; }

.dp-evidence-row {
    margin: 8pt 13pt 0;
    padding-top: 8pt;
    border-top: 1px solid #e2e8f0;
}

.dp-evidence-row span {
    color: #334155;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

.dp-confidence-note {
    margin: 8pt 13pt 12pt;
    color: #475569;
    font-size: 8pt;
}

/* ── Finding cards (Findings section) ─────────── */

.dp-finding-card {
    --dp-accent: var(--dp-medium);
    margin: 0 0 14pt;
    border-left: 3.5pt solid var(--dp-accent);
    border-radius: 0 8pt 8pt 0;
    page-break-inside: avoid;
}

/* Severity-driven accent (left rail + section id + signal bar) */
.dp-finding-card--critical { --dp-accent: var(--dp-critical); }
.dp-finding-card--high { --dp-accent: var(--dp-high); }
.dp-finding-card--medium { --dp-accent: var(--dp-medium); }
.dp-finding-card--low { --dp-accent: var(--dp-low); }
.dp-finding-card--clean { --dp-accent: var(--dp-human); }

.dp-finding-card .dp-finding-section-id { color: var(--dp-accent); }
.dp-finding-card .dp-signal-strength-fill { background: var(--dp-accent); }

.dp-finding-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 11pt 13pt 10pt;
    border-bottom: 1px solid #e2e8f0;
}

.dp-finding-section-id {
    display: block;
    color: #64748b;
    font-size: 6.8pt;
    font-weight: 800;
    letter-spacing: .10em;
    text-transform: uppercase;
    margin-bottom: 3pt;
}

.dp-finding-type {
    color: #0f172a;
    font-size: 13pt;
    font-weight: 800;
    line-height: 1.2;
}

.dp-finding-count {
    color: #94a3b8;
    font-size: 8pt;
    font-weight: 700;
    white-space: nowrap;
}

.dp-finding-body {
    padding: 10pt 13pt 12pt;
}

.dp-finding-description {
    margin: 0 0 8pt;
    color: #374151;
    font-size: 9pt;
    line-height: 1.45;
}

.dp-finding-strength-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 2pt;
}

.dp-finding-strength-label {
    color: #64748b;
    font-size: 6.5pt;
    font-weight: 800;
    letter-spacing: .10em;
    text-transform: uppercase;
}

.dp-finding-strength-pct {
    color: #0f172a;
    font-size: 9pt;
    font-weight: 900;
}

.dp-signal-strength-bar {
    height: 5pt;
    border-radius: 999pt;
    background: #e2e8f0;
    overflow: hidden;
    margin-bottom: 8pt;
}

.dp-signal-strength-fill {
    height: 100%;
    min-width: 2pt;
    border-radius: inherit;
    background: #b45309;
}

.dp-tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4pt;
    margin-bottom: 6pt;
}

.dp-tag-chip {
    display: inline-block;
    padding: 2pt 6pt;
    border-radius: 4pt;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    color: #334155;
    font-size: 7pt;
    font-weight: 800;
}

.dp-also-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4pt;
    margin-bottom: 8pt;
}

.dp-also-label {
    color: #64748b;
    font-size: 6.5pt;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.dp-also-chip {
    display: inline-block;
    padding: 2pt 6pt;
    border-radius: 4pt;
    background: #fef3c7;
    border: 1px solid #fcd34d;
    color: #92400e;
    font-size: 7pt;
    font-weight: 800;
}

.dp-finding-subsection {
    margin: 6pt 0 0;
    padding: 7pt 9pt;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 5pt;
}

.dp-finding-subsection-label {
    color: #64748b;
    font-size: 6.5pt;
    font-weight: 900;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 3pt;
}

.dp-finding-subsection p {
    margin: 0;
    color: #1f2937;
    font-size: 8.5pt;
    line-height: 1.4;
}

.dp-finding-bullets {
    margin: 2pt 0 0 12pt;
    padding: 0;
    color: #1f2937;
    font-size: 8.5pt;
    line-height: 1.5;
}
.dp-evidence-list {
    margin: 2pt 0 0 12pt;
    padding: 0;
    list-style: none;
    font-size: 8.5pt;
    line-height: 1.5;
}
.dp-evidence-list li {
    margin-bottom: 3pt;
    display: flex;
    align-items: flex-start;
    gap: 4pt;
}
.dp-evidence-score {
    flex-shrink: 0;
    min-width: 26pt;
    text-align: center;
    padding: 1pt 3pt;
    border-radius: 3pt;
    color: #fff;
    font-size: 7.5pt;
    font-weight: 700;
}
.dp-evidence-text {
    color: #475569;
}
.dp-evidence-suggestion {
    margin-top: 2pt;
    color: #0f766e;
    font-weight: 600;
}
"""


def render_pdf(markdown_text: str, output_path: str) -> str:
    """Convert markdown text to a styled PDF file.

    Args:
        markdown_text: Full markdown content.
        output_path: Where to write the .pdf file.

    Returns:
        Absolute path to the generated PDF.
    """
    extensions = ["tables", "fenced_code"]
    html_body = md_lib.markdown(markdown_text, extensions=extensions)

    # Inject colgroup into wide tables (6-column: findings & false-positives)
    colgroup_6 = '<colgroup><col style="width:3%"/><col style="width:4%"/><col style="width:4%"/><col style="width:36%"/><col style="width:26%"/><col style="width:27%"/></colgroup>'
    # Only tables whose header row starts with # (findings/false-positives pattern)
    html_body = re.sub(
        r'(<table>\s*<thead>\s*<tr>\s*<th[^>]*>\s*#\s*</th>)',
        lambda m: m.group(1).replace('<table>', '<table>' + colgroup_6),
        html_body,
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><style>{_CSS}</style></head>
<body>{html_body}</body>
</html>"""

    HTML(string=html_doc).write_pdf(output_path)
    return output_path
