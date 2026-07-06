"""Enhanced scan-report panels (hero, KPI row, priority fixes, policy view).

Presentation-only builders for the redesigned PDF lead. They reuse render.py's
data helpers + label maps via lazy imports (avoids a circular import at load
time). Each returns a markdown/HTML string; an empty string means "no data —
caller falls back to the legacy lead".

allow-hardcode: the maps below are PRESENTATION copy keyed by machine CODES
(bucket / axis ids), mirroring the frontend i18n. They are display strings for
the PDF, never matched against document text — not a detect/scoring word-list.
"""
from html import escape


# Per-bucket "priority fix" copy, keyed by grounding-diagnosis bucket code.
# KEEP IN SYNC with render._DIAG_BUCKET_LABELS.
_DIAG_BUCKET_FIX = {
    "concrete_grounding": "Replace broad claims with named evidence — cite a real source, "
                          "dataset, case, or example, and the education context being discussed.",
    "authorship_trace": "Add your own evaluation. Don't only say a thing is important — say "
                        "which point matters most and why, in your own words.",
    "llm_patterning": "Break the uniform rhythm. Vary sentence structure and cut templated, "
                      "predictable transitions.",
    "language_texture": "Replace report-style filler with concrete, testable wording. Vague "
                        "phrases need sharper, checkable meaning.",
}

# allow-hardcode: presentation copy keyed by primary-driver bucket CODE — a clean
# noun phrase for the hero sentence (the bucket label "Grounding gap" reads badly
# inline as "grounding gap work"). Display strings, never matched against text.
_DRIVER_HERO = {
    "concrete_grounding": "stronger grounding in named evidence",
    "authorship_trace": "clearer signs of your own judgement",
    "llm_patterning": "less templated, more varied writing",
    "language_texture": "sharper, more concrete wording",
}

# Plain-English "what it means" copy for the submission-risk axis table.
# KEEP IN SYNC with render._SR_AXIS_ORDER and the frontend i18n.
_SR_AXIS_MEANING = {
    "text_pattern": "NOT a Turnitin score — don't compare it to the 20% line. Detectors over-flag fluent writing, so treat it as a heads-up, not a verdict.",
    "ownership": "Higher means the draft needs more visible judgement so you can explain your choices.",
    "citation": "Whether claims are tied to named, checkable sources.",
    "defence_readiness": "Whether you could defend these claims, tools, and examples if asked.",
    "policy_declaration": "The report can't know course policy or whether AI was used — you declare this.",
}


def _statchip(text: str, kind: str) -> str:
    return f'<span class="dp-statchip dp-statchip--{kind}">{escape(text)}</span>'


def _level_kind(level: str) -> str:
    """Map a risk level code to a chip/panel colour family."""
    lv = str(level or "").lower()
    if lv in ("low",):
        return "good"
    if lv in ("high", "severe", "critical"):
        return "warn"
    return "info"  # medium / moderate / unknown


def render_authenticity_dashboard(report_data: dict) -> str:
    """HTML panel for the authenticity dashboard; '' when absent (flag off / old report)."""
    badge = (report_data or {}).get("ai_risk_badge") or {}
    dash = badge.get("authenticity_dashboard")
    if not dash:
        return ""
    rows = []
    for key, label in (("learning_ownership", "Learning Ownership"), ("grounding", "Grounding"),
                       ("citation_quality", "Citation Quality")):
        tile = dash.get(key) or {}
        val = round(tile["score"]) if isinstance(tile.get("score"), (int, float)) else "—"
        rows.append(f'<div class="dp-kpi"><b>{escape(str(val))}</b><span>{escape(label)}</span></div>')
    ai = dash.get("ai_assistance") or {}
    rows.append(f'<div class="dp-kpi"><b>{escape(str(ai.get("band") or "—"))}</b><span>AI Assistance</span></div>')
    overall = dash.get("overall") or {}
    head = f'Authenticity — Overall {escape(str(overall.get("band") or "—"))}'
    return (f'<div class="dp-hero"><p class="dp-hero-read">{escape(head)}</p>'
            f'<div class="dp-kpi-row">{"".join(rows)}</div>'
            '<p class="dp-hero-sub">Guidance for your review — overlapping dimensions, not independent '
            'measurements or a Turnitin prediction.</p></div>')


def render_deberta_signal(report_data: dict) -> str:
    """HTML panel for the second-opinion DeBERTa AI signal; '' when absent (flag off / old report).

    STRICTLY ADVISORY — never feeds the tier or any gate.
    Renders the v2 schema (threshold-proportion): above the 20% floor it shows a band
    (amber/orange/red) + the proportion + flagged passages; below the floor it shows NO
    verdict, only the flagged passages (mirrors how mature detectors treat the low-reliability
    band). Legacy v1 reports (calibrated score) fall back to a simple headline read."""
    badge = (report_data or {}).get("ai_risk_badge") or {}
    sig = badge.get("ai_signal_deberta")
    if not sig:
        return ""
    if sig.get("available") is False:
        return (f'<div class="dp-hero"><p class="dp-hero-read">Second-opinion AI signal — '
                f'unavailable ({escape(str(sig.get("caveat") or ""))})</p></div>')

    is_v2 = sig.get("model_version") == "deberta_signal_v2"

    # --- Legacy v1 (calibrated score) fallback for old reports in R2 -----------------
    if not is_v2:
        score = sig.get("score")
        score_txt = f"{round(score)}%" if isinstance(score, (int, float)) else "—"
        band = str(sig.get("band") or "—")
        calibrated = "calibrated" if sig.get("calibrated") else "raw, uncalibrated — advisory only"
        return (f'<div class="dp-hero"><p class="dp-hero-read">'
                f'Second-opinion AI signal — {score_txt} ({band}, {calibrated})</p></div>')

    # --- v2 threshold-proportion ----------------------------------------------------
    pct = sig.get("signal_pct")
    band = str(sig.get("band") or "insufficient")
    above_floor = band != "insufficient"
    flagged = sig.get("flagged_passages") or []
    n_flagged = sig.get("sentences_flagged") if isinstance(sig.get("sentences_flagged"), int) else len(flagged)
    n_scored = sig.get("sentences_scored") if isinstance(sig.get("sentences_scored"), int) else 0

    # Flagged passages list (capped; highest-score first as built by compose).
    flagged_html = ""
    if flagged:
        items = "".join(
            f'<li><b>{round(float(p.get("score") or 0) * 100)}%</b> — {escape(_truncate(p.get("text") or ""))}</li>'
            for p in flagged
        )
        flagged_html = f'<ul class="dp-flagged">{items}</ul>'

    if above_floor:
        head = (f"Second-opinion AI signal — {pct}% of passages read as AI-like under a "
                f"separate detector ({band}). Advisory only — review the flagged passages.")
        return (f'<div class="dp-hero"><p class="dp-hero-read">{escape(head)}</p>'
                f'{flagged_html}</div>')

    # Below floor — no verdict, only the flagged passages (if any) + the floor note.
    if n_flagged:
        head = (f"Second-opinion AI signal — {n_flagged} of {n_scored} passages flagged for "
                f"review. Below the 20% reliability floor, so no overall verdict.")
    else:
        head = "Second-opinion AI signal — no high-confidence AI passages detected."
    return (f'<div class="dp-hero"><p class="dp-hero-read">{escape(head)}</p>'
            f'{flagged_html}</div>')


def _truncate(text: str, limit: int = 220) -> str:
    """Truncate a flagged passage for the PDF list cell."""
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# Category display order + labels, mirroring draftproof-frontend/src/i18n/en/report.js
# authorshipBreakdown.categories (and AuthorshipClarityBreakdown.jsx CATEGORY_ORDER).
# allow-hardcode: presentation copy for machine category CODES, not a scoring/matching list.
_ACB_CATEGORY_ORDER = (
    "student_owned",
    "ai_assisted_polished",
    "ai_paraphrased",
    "ai_generated_like",
)
_ACB_CATEGORY_LABELS = {
    "student_owned": "Student-owned",
    "ai_assisted_polished": "AI-assisted / polished",
    "ai_paraphrased": "AI-paraphrased",
    "ai_generated_like": "AI-generated-like",
}
_ACB_BAND_LABELS = {"Strong": "Strong", "Some": "Some", "Little": "Little", "None": "None"}
_ACB_FUSED_BAND_LABELS = {"low": "Low", "moderate": "Moderate", "high": "High", "critical": "Critical"}
_ACB_TIER_TO_BAND = {"green": "low", "amber": "moderate", "orange": "high", "red": "critical"}
_ACB_DEEP_SCAN_BAND_LABELS = {"amber": "Amber", "orange": "Orange", "red": "Red"}


# allow-hardcode: HTML markup/CSS class names + human-reviewed presentation copy for the
# PDF authorship panel below (headline + bars) — not a scoring/matching oracle over document
# text; band labels/cutoffs are read from the badge's own fields, never invented here.
def _authorship_headline_html(badge: dict, breakdown: dict) -> str:
    """The single AI-likelihood headline (fused, else deep-scan). Returns '' if neither."""
    tier_authority = badge.get("tier_authority")
    authoritative_tier = badge.get("tier")
    if isinstance(tier_authority, dict) and isinstance(tier_authority.get("fused_score"), (int, float)):
        fused_score = tier_authority["fused_score"]
        composite = tier_authority.get("composite_score")
        deep_scan_pct = round((tier_authority.get("proportion") or 0) * 100)
        band_chip = ""
        if authoritative_tier in _ACB_TIER_TO_BAND:
            band = _ACB_TIER_TO_BAND[authoritative_tier]
            band_chip = _statchip(_ACB_FUSED_BAND_LABELS.get(band, band), _level_kind(band))
        evidence = ""
        if isinstance(composite, (int, float)):
            evidence = (f"Behind this score: composite detector {round(composite)}%, "
                        f"deep-scan detector {deep_scan_pct}% (sentence-level). See the DraftProof scale above.")
        return ('<div class="dp-hero"><p class="dp-hero-read">DraftProof AI-likelihood '
                f'<b>{round(fused_score)}%</b> {band_chip}</p>'
                + (f'<p class="dp-hero-sub">{escape(evidence)}</p>' if evidence else "") + "</div>")
    deep_scan = breakdown.get("deep_scan") or {}
    band = deep_scan.get("band")
    if band and band in ("insufficient", "amber", "orange", "red"):
        proportion = deep_scan.get("proportion")
        if band == "insufficient" or not isinstance(proportion, (int, float)):
            return ('<div class="dp-hero"><p class="dp-hero-read">Deep-scan AI estimate '
                    f'{_statchip("insufficient evidence", "info")}</p>'
                    '<p class="dp-hero-sub">sentence-level signal — not a Turnitin score</p></div>')
        band_label = _ACB_DEEP_SCAN_BAND_LABELS.get(band, band)
        return ('<div class="dp-hero"><p class="dp-hero-read">Deep-scan AI estimate '
                f'<b>{round(proportion * 100)}%</b> {_statchip(band_label, _level_kind(band))}</p>'
                '<p class="dp-hero-sub">sentence-level signal — not a Turnitin score</p></div>')
    return ""


# allow-hardcode: presentation copy/markup for the per-paragraph deep-scan rows below —
# bands/proportions are read from the breakdown's own deep_scan.paragraphs fields
# (computed in poc/detect_v7/pipeline_bridge.py), never invented here.
def _deep_scan_paragraphs_html(breakdown: dict) -> str:
    """Compact per-paragraph deep-scan rows. '' when absent (quick scan /
    single paragraph / mapping fail-open) — mirrors the web panel's
    deepScanParagraphs section in MergedAuthorshipRisk.jsx."""
    deep_scan = breakdown.get("deep_scan") or {}
    paragraphs = deep_scan.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        return ""
    floor = deep_scan.get("reliability_floor")
    floor_pct = round(floor * 100) if isinstance(floor, (int, float)) else None
    rows = []
    for p in paragraphs:
        if not isinstance(p, dict):
            continue
        idx = p.get("index")
        proportion = p.get("proportion")
        band = p.get("band")
        count = p.get("sentence_count")
        flagged = p.get("flagged_count")
        if not isinstance(idx, int):
            continue
        insufficient = band == "insufficient" or not isinstance(proportion, (int, float))
        if insufficient:
            chip = _statchip("insufficient evidence", "info")
            value = ""
        else:
            chip = _statchip(_ACB_DEEP_SCAN_BAND_LABELS.get(band, band or ""), _level_kind(band))
            value = f"<b>{round(proportion * 100)}%</b> "
        # Show the arithmetic behind the verdict (mirrors the web panel):
        # "1 of 4 sentences flagged · 25% — below the 30% reliability floor".
        if isinstance(flagged, int) and isinstance(count, int) and floor_pct is not None:
            pct = round((proportion or 0) * 100)
            plural = "" if count == 1 else "s"
            if insufficient:
                detail = (f" · {flagged} of {count} sentence{plural} flagged · {pct}% — "
                          f"below the {floor_pct}% reliability floor, too few to judge this paragraph")
            else:
                detail = (f" · {flagged} of {count} sentence{plural} flagged · "
                          f"at or above the {floor_pct}% reliability floor")
        else:
            detail = (f" · {count} sentence{'' if count == 1 else 's'}"
                      if isinstance(count, int) else "")
        rows.append(
            '<p class="dp-hero-sub dp-dsp-row">'
            f"Paragraph {idx + 1}: {value}{chip}{escape(detail)}</p>"
        )
    if not rows:
        return ""
    return ('<div class="dp-dsp"><p class="dp-hero-sub"><b>Per-paragraph deep-scan signal</b> '
            "— same detector pass grouped by paragraph; short paragraphs are noisier</p>"
            + "".join(rows) + "</div>")


def _authorship_bars_html(breakdown: dict) -> str:
    """The 4 color-coded category bars + disclaimer. '' when no breakdown."""
    raw_shares = breakdown.get("document_breakdown_raw") or {}
    band_shares = breakdown.get("document_breakdown_bands") or {}
    rows = []
    for category in _ACB_CATEGORY_ORDER:
        raw = raw_shares.get(category)
        band = band_shares.get(category)
        has_raw = isinstance(raw, (int, float))
        width_pct = max(0.0, min(100.0, raw * 100)) if has_raw else 0.0
        band_label = _ACB_BAND_LABELS.get(band, band) if band else _ACB_BAND_LABELS["None"]
        text = f"{band_label} · {round(width_pct)}%" if has_raw else band_label
        rows.append(
            '<div class="dp-abd-row">'
            f'<span class="dp-abd-label">{escape(_ACB_CATEGORY_LABELS.get(category, category))}</span>'
            '<span class="dp-abd-bar-track">'
            f'<span class="dp-abd-bar-fill dp-abd-fill--{category}" style="width:{width_pct}%"></span></span>'
            f'<span class="dp-abd-band">{escape(text)}</span></div>'
        )
    out = '<div class="dp-abd-bars">' + "".join(rows) + "</div>"
    disclaimer = breakdown.get("disclaimer")
    if disclaimer:
        out += f'<p class="dp-hero-sub">{escape(str(disclaimer))}</p>'
    return out


def render_authorship_breakdown(report_data: dict) -> str:
    """HTML panel mirroring draftproof-frontend's AuthorshipClarityBreakdown.jsx:
    fused AI-likelihood headline (or deep-scan-only fallback), the 4 category bars,
    and the verbatim disclaimer. '' when badge.authorship_breakdown is absent
    (flag off / older report) — additive, fail-open, no hardcoded thresholds
    (band labels/cutoffs are read from the badge's own fields, never invented)."""
    badge = (report_data or {}).get("ai_risk_badge") or {}
    breakdown = badge.get("authorship_breakdown")
    if not breakdown:
        return ""

    # allow-hardcode: presentation copy/markup for the PDF panel, not a scoring/matching list.
    out = [
        '<div class="authorship-breakdown">',
        '<p class="dp-callout-title">Authorship clarity breakdown <span class="dp-statchip dp-statchip--info">Beta</span></p>',
        '<p class="dp-hero-sub">How this document\'s writing signals distribute across four '
        "authorship styles. The shares always add up to 100% — a composition of the mix, not an "
        "AI-probability. The deep-scan estimate below comes from a separate beta detector and may "
        "differ from Text-pattern risk in the summary above — different models, and both are "
        "signals rather than verdicts.</p>",
        _authorship_headline_html(badge, breakdown),
        _authorship_bars_html(breakdown),
        "</div>",
    ]
    return "\n".join(p for p in out if p)


def render_scan_lead(report, data, *, suppress_ai_likelihood: bool = False) -> str:
    """Scan-PDF lead. PAGE-PARITY contract (owner rule 2026-07-06: the PDF must
    show what the scan page shows — nothing more):

    In merged/V7 mode (``suppress_ai_likelihood=True``, i.e. an authorship
    breakdown is present) this renders ONLY the scan page's merged-card
    content: the hero verdict, the "Where the risk sits" axis levels, and the
    score-scale table. The old PDF-only blocks — KPI row, "Priority fixes",
    numeric policy cards ("AI allowed with declaration 34" etc.), the axis
    "What it means" column, and the "Plain-English verdict" — exist on the
    web ONLY in rewrite-comparison mode (Report.jsx gates PolicyRiskView on
    hasRewriteSignalComparison), so scan PDFs must not show them.

    Legacy mode (no breakdown — pre-V7 stored reports) keeps the original
    richer lead unchanged.

    Returns "" when the submission-risk diagnosis abstained, so render_report
    keeps the legacy lead as a fallback.
    """
    from .render import (
        _ai_likelihood_bands, _POLICY_ISSUE, _POLICY_LEVEL, _SR_LEVEL_LABELS,
        _SR_AXIS_ORDER, _DIAG_BUCKET_LABELS, _transformation_contribution_summary,
    )

    badge = report.ai_risk_badge or {}
    sr = badge.get("submission_risk") or {}
    overall = sr.get("overall") or {}
    sr_level = overall.get("level")
    if not sr_level or sr_level == "unknown":
        return ""

    diag = badge.get("grounding_diagnosis") or {}
    bands = _ai_likelihood_bands(badge)
    dp_band = bands.get("draftproof") or {}
    doc_ctx = data.get("document_context", {}) if isinstance(data, dict) else {}

    sr_label = overall.get("label") or f"{_SR_LEVEL_LABELS.get(sr_level, sr_level)} submission risk"
    driver_code = diag.get("primary_driver") or ""
    driver_label = diag.get("primary_driver_label") or ""
    driver_action = diag.get("primary_driver_action") or ""
    hero_phrase = _DRIVER_HERO.get(driver_code) or (driver_label.lower() if driver_label else "")

    out: list[str] = []

    # ── Hero "overall read" panel ──────────────────────────────────
    # allow-hardcode: student-facing display copy (the report banner), not a scoring or
    # matching word-list -- never compared against document text.
    # Lead with what the tier ACTUALLY measures -- ownership -- so a low tier is never
    # misread as "I'll pass Turnitin"; then the action. (Detector caveat lives in `sub`.)
    if sr_level == "low":
        read = "You can defend this as your own work"
    else:
        read = "Strengthen this before you can fully defend it as your own"
    if hero_phrase:
        read += f" — but before you submit, the draft needs {hero_phrase}"
    read += "."
    # Never let a low tier falsely reassure about external detectors: state the warning plainly.
    sub = ("Heads-up: AI detectors (Turnitin, GPTZero) over-flag fluent writing, so they may "
           "flag this even though it is your own work — a warning, not a verdict. Your "
           "protection is grounding your claims and keeping your drafts.")

    # Chip 1: scope the tier to what ACTUALLY drove it. Normally that's ownership, but when
    # the level was floored UP by the detector text-pattern axis (main_reason_code ==
    # 'text_pattern') ownership was NOT the driver, so don't mislabel it (L9; matches the
    # React band's reason-code attribution).
    chip_axis = "detector-pattern risk" if overall.get("main_reason_code") == "text_pattern" else "ownership risk"
    chips = [(f"{_SR_LEVEL_LABELS.get(sr_level, sr_level)} {chip_axis}", _level_kind(sr_level))]
    # No AI-style tier chip in the hero. A "Low/Moderate" valence here clashed with the
    # ownership chip and either over-alarmed (Moderate ~32% read as a Turnitin fail) OR
    # falsely reassured (a GREEN tier read as "detectors will pass me", though they
    # over-flag fluent writing -- raw top-k can be ~78%). The honest detector message lives
    # in `sub`; the calibrated number stays in the axis table with its heads-up framing.
    if driver_label:
        chips.append((driver_label, "warn"))
    wc = doc_ctx.get("word_count") or 0
    if wc and wc < 250:
        chips.append(("Low sample confidence", "info"))

    out.append(
        f'<div class="dp-hero dp-hero--{_level_kind(sr_level)}">'
        f'<p class="dp-hero-read">{escape(read)}</p>'
        f'<p class="dp-hero-sub">{escape(sub)}</p>'
        + (
            # Tier reason (matches the web page's overall_tier_reason): explains WHY the tier
            # is what it is — e.g. "the learned classifier is >=99% confident that 2 of 14
            # sentences are AI-generated (14% high-confidence)". Only shown when present.
            f'<p class="dp-hero-reason">{escape(tier_reason)}</p>'
            if (tier_reason := str(data.get("overall_tier_reason") or "").strip())
            else ""
        )
        + '<div class="dp-chip-strip">'
        + "".join(_statchip(t, k) for t, k in chips)
        + "</div></div>"
    )

    # Page-parity mode: the merged/V7 path (see docstring). Everything below
    # that the scan page does not render is skipped in this mode.
    page_parity = bool(suppress_ai_likelihood)

    # ── KPI stat row (legacy mode only — the scan page has no KPI row) ──
    contribution = {}
    try:
        transformation = badge.get("transformation_classification") or {}
        features = transformation.get("features") or {}
        from .render import _signal_chart_rows
        rows = _signal_chart_rows(features, badge)
        contribution = _transformation_contribution_summary(features, rows, badge) or {}
    except Exception:
        contribution = {}

    kpis: list[tuple[str, str]] = []
    # Submission-risk SCORE number intentionally omitted (web-page parity): a bare "19%"
    # next to a "Low" level reads as "<20% = Turnitin pass". The web report shows the level
    # word only; the PDF follows. The calibrated AI number stays below as the framed
    # "Text-pattern trigger" (with its "NOT a Turnitin score" axis note).
    if not suppress_ai_likelihood and isinstance(dp_band.get("score"), (int, float)):
        kpis.append((f"{round(dp_band['score'])}%", "Text-pattern trigger"))
    if isinstance(contribution.get("hcr"), (int, float)):
        kpis.append((f"{round(contribution['hcr'])}%", "Human contribution"))
    if wc:
        kpis.append((f"{wc}", "Words scanned"))
    # >= 1 (not 2): the merged block suppresses the AI-likelihood KPI (it shows once
    # in the co-located headline), so this row can legitimately carry a single stat
    # (e.g. "Words scanned") — never drop it just because it is alone.
    if not page_parity and len(kpis) >= 1:
        out.append(
            '<div class="dp-kpi-row">'
            + "".join(f'<div class="dp-kpi"><b>{escape(v)}</b><span>{escape(c)}</span></div>' for v, c in kpis)
            + "</div>"
        )

    # ── Priority fixes (top grounding buckets, by score) ───────────
    buckets = diag.get("buckets") or {}
    ranked = sorted(
        (k for k in _DIAG_BUCKET_LABELS if k in buckets and k in _DIAG_BUCKET_FIX),
        key=lambda k: -(buckets[k].get("score") or 0),
    )[:3]
    if ranked and not page_parity:
        items = "".join(f"<li>{escape(_DIAG_BUCKET_FIX[k])}</li>" for k in ranked)
        out.append(
            '<div class="dp-callout dp-callout--good">'
            '<span class="dp-callout-title">Priority fixes</span>'
            f"<ol>{items}</ol></div>"
        )

    # ── 1. Where the risk sits (page parity) / Submission and policy view (legacy) ──
    if page_parity:
        out.append("## 1. Where the risk sits")
        out.append(
            '<p class="dp-section-intro">Independent risk axes — the same read as the '
            "report page's merged card.</p>"
        )
    else:
        out.append("## 1. Submission and policy view")
        out.append(
            '<p class="dp-section-intro">This separates three things students often mix up: '
            "school policy, external-detector attention, and whether you can defend the work as your own.</p>"
        )

    pr = badge.get("policy_risk") or {}
    pa, prr = pr.get("ai_allowed") or {}, pr.get("ai_restricted") or {}
    policy_cards = []
    # Numeric policy cards are rewrite-comparison-page content — never on scan PDFs.
    if not page_parity and pa.get("level") and pa["level"] != "unknown":
        policy_cards.append(
            '<div class="dp-policy-head"><span class="dp-ph-name">Policy view</span>'
            '<span class="dp-ph-issue">Main issue</span></div>'
        )
        for label, p in (("AI allowed with declaration", pa), ("AI not allowed", prr)):
            kind = _level_kind(p.get("level"))
            lvl = f"{_POLICY_LEVEL.get(p.get('level'), 'Unknown')}"
            if isinstance(p.get("score"), (int, float)):
                lvl += f" {round(p['score'])}"
            issue = _POLICY_ISSUE.get(p.get("main_issue"), "")
            issue = issue[0].upper() + issue[1:] if issue else "—"
            policy_cards.append(
                f'<div class="dp-policy-row dp-policy-row--{kind}">'
                f'<span class="dp-policy-name">{escape(label)}</span>'
                f'<span class="dp-policy-level">{escape(lvl)}</span>'
                f'<span class="dp-policy-issue">{escape(issue)}</span></div>'
            )
        # Submission readiness card (the third row in the mockup).
        rk = _level_kind(sr_level)
        ready_note = "Can defend ownership" if sr_level == "low" else "Strengthen before you submit"
        policy_cards.append(
            f'<div class="dp-policy-row dp-policy-row--{rk}">'
            '<span class="dp-policy-name">Submission readiness</span>'
            f'<span class="dp-policy-level">{escape(_SR_LEVEL_LABELS.get(sr_level, sr_level))}</span>'
            f'<span class="dp-policy-issue">{escape(ready_note)}</span></div>'
        )
        out.append("".join(policy_cards))

    # Axis table (Axis / Current read / What it means).
    axes = sr.get("axes") or {}
    axis_rows = []
    for key in _SR_AXIS_ORDER:
        ax = axes.get(key) or {}
        label = ax.get("label") or key.replace("_", " ").title()
        if key == "policy_declaration":
            current = "Unknown — self-declare"
        else:
            current = _SR_LEVEL_LABELS.get(ax.get("level"), "Unknown")
            if key == "text_pattern":
                # Use the SAME canonical AI-likelihood number as the KPI "Text-pattern
                # trigger" (dp_band score = badge ai_likelihood_score) so the axis table
                # can't disagree with it. Previously this used display_score, which
                # diverged (e.g. 34% here vs the headline 32%).
                ai_pct = dp_band.get("score")
                if not isinstance(ai_pct, (int, float)):
                    ai_pct = ax.get("display_score")
                if not suppress_ai_likelihood and isinstance(ai_pct, (int, float)):
                    current += f" — DraftProof AI-likelihood ~{round(ai_pct)}%"
        meaning = _SR_AXIS_MEANING.get(key, "")
        axis_rows.append((label, current, meaning))
    if axis_rows:
        out.append("")
        if page_parity:
            # Page parity: the merged card shows axis name + level only.
            out.append("| Axis | Current read |")
            out.append("|------|--------------|")
            for a, c, _m in axis_rows:
                out.append(f"| {a} | {c} |")
        else:
            out.append("| Axis | Current read | What it means |")
            out.append("|------|--------------|----------------|")
            for a, c, m in axis_rows:
                out.append(f"| {a} | {c} | {m} |")
        out.append("")

    if page_parity:
        # Score-scale table — the page's "What do these numbers mean?" details
        # block. Also grounds the fused headline's "See the DraftProof scale"
        # reference, which previously pointed at nothing in the PDF.
        # allow-hardcode: display copy — KEEP IN SYNC with
        # draftproof-frontend/src/i18n/en/report.js report.submissionRisk.scale.
        out.append("**What do these numbers mean?**")
        out.append("")
        out.append("| DraftProof score | Reads as | What we measured |")
        out.append("|------------------|----------|-------------------|")
        out.append("| 0–32 | Low | ~6% or fewer real ESL students score this high — most human writing lands well under this |")
        out.append("| 32–48 | Medium | uncommon for genuine human writing (0.4% false-positive rate measured) |")
        out.append("| 48–65 | High | rare for genuine human writing (<1% false-positive rate measured) |")
        out.append("| 65+ | Critical | essentially never seen in genuine human writing in our testing (0% false-positive rate measured) |")
        out.append("")
        out.append(
            '<p class="dp-hero-sub">DraftProof\'s score is not comparable to Turnitin\'s '
            "percentage — the two tools measure different things on different scales.</p>"
        )
    else:
        # Plain-English verdict callout (amber) — legacy lead only; the scan
        # page has no such box.
        verdict = "The draft is not primarily risky because it looks AI-written. "
        if driver_label:
            verdict += f"It is risky because it reads too broad and under-evidenced — the main gap is {driver_label.lower()}."
        else:
            verdict += "It is risky when claims are broad and under-evidenced for a student submission."
        out.append(
            '<div class="dp-callout dp-callout--warn">'
            '<span class="dp-callout-title">Plain-English verdict</span>'
            f"<p>{escape(verdict)}</p></div>"
        )

    return "\n".join(out)


# allow-hardcode: presentation copy/markup for the merged PDF panel header below, not a
# scoring/matching list.
def render_merged_authorship_risk(report, data) -> str:
    """Unified PDF block: authorship composition bars + ONE AI-likelihood headline
    co-located at the top, then the richer submission-risk section (policy cards,
    priority fixes, axis table, verdict) with the AI number de-duped. Falls back
    gracefully: if there's no authorship breakdown, this is just render_scan_lead;
    if there's no submission-risk section, it's just the authorship panel."""
    badge = getattr(report, "ai_risk_badge", None) or {}
    breakdown = badge.get("authorship_breakdown")
    lead = render_scan_lead(report, data, suppress_ai_likelihood=bool(breakdown))
    if not breakdown:
        return lead
    header = ('<div class="authorship-breakdown">'
              '<p class="dp-callout-title">Authorship &amp; submission risk '
              '<span class="dp-statchip dp-statchip--info">Beta</span></p>'
              + _authorship_headline_html(badge, breakdown)
              + _authorship_bars_html(breakdown)
              + _deep_scan_paragraphs_html(breakdown)
              + "</div>")
    if not lead:
        return header
    return header + "\n" + lead


def _num(v):
    return v if isinstance(v, (int, float)) else None


def rewrite_hero(*, result_label, explanation, outcome, ai_improved, score_worse,
                 original_preserved, orig_ai, new_ai, orig_human, new_human,
                 orig_wq, new_wq, o_total, n_total,
                 orig_deep_scan=None, new_deep_scan=None) -> str:
    """Hero result panel + before→after KPI row for the rewrite report.

    Mirrors the scan report's hero/KPI design. All values are pre-computed by
    render_rewrite_report (already in the right units). orig/new_deep_scan are the V7
    deberta deep-scan proportions (percent); None when deep-scan is off — then the
    deep-scan chip/KPI are simply omitted (fail-open, never fabricated)."""
    good = (str(outcome) == "ai_mitigated") or (ai_improved and not score_worse)
    kind = "good" if good else ("info" if original_preserved else "warn")

    chips = [(result_label, kind)]
    if _num(orig_ai) is not None and _num(new_ai) is not None:
        chips.append((f"AI likelihood {orig_ai:.0f}→{new_ai:.0f}%",
                      "good" if new_ai < orig_ai else ("warn" if new_ai > orig_ai else "info")))
    if _num(orig_deep_scan) is not None and _num(new_deep_scan) is not None:
        chips.append((f"Deep-scan {orig_deep_scan:.0f}→{new_deep_scan:.0f}%",
                      "good" if new_deep_scan < orig_deep_scan else ("warn" if new_deep_scan > orig_deep_scan else "info")))
    if _num(orig_human) is not None and _num(new_human) is not None:
        chips.append((f"Human {orig_human:.0f}→{new_human:.0f}%",
                      "good" if new_human >= orig_human else "warn"))
    chips.append((f"Findings {o_total}→{n_total}", "good" if n_total <= o_total else "warn"))

    hero = (
        f'<div class="dp-hero dp-hero--{kind}">'
        f'<p class="dp-hero-read">Result: {escape(str(result_label))}</p>'
        f'<p class="dp-hero-sub">{escape(str(explanation))}</p>'
        '<div class="dp-chip-strip">'
        + "".join(_statchip(t, k) for t, k in chips)
        + "</div></div>"
    )

    kpis = []
    if _num(new_ai) is not None:
        kpis.append((f"{new_ai:.0f}%", f"AI likelihood (was {orig_ai:.0f}%)"))
    if _num(new_deep_scan) is not None and _num(orig_deep_scan) is not None:
        kpis.append((f"{new_deep_scan:.0f}%", f"Deep-scan AI (was {orig_deep_scan:.0f}%)"))
    if _num(new_human) is not None:
        kpis.append((f"{new_human:.0f}%", f"Human contribution (was {orig_human:.0f}%)"))
    if _num(new_wq) is not None:
        kpis.append((f"{new_wq:.0f}%", f"Grounding risk (was {orig_wq:.0f}%)"))
    kpis.append((f"{n_total}", f"Findings (was {o_total})"))
    kpi = (
        '<div class="dp-kpi-row">'
        + "".join(f'<div class="dp-kpi"><b>{escape(v)}</b><span>{escape(c)}</span></div>' for v, c in kpis)
        + "</div>"
    )
    return hero + "\n" + kpi


def render_question_cards(badge, section_no: int = 3) -> str:
    """Restyled Critical-Thinking questions as purple cards. "" when no questions.

    ``section_no`` is supplied by render.py's dynamic numbering (sections
    render conditionally per page parity, so the number can't be a literal)."""
    ctc = (badge or {}).get("critical_thinking_control") or {}
    rows = [q for q in (ctc.get("questions") or [])
            if isinstance(q, dict) and str(q.get("question") or "").strip()]
    if not rows:
        return ""
    out = [
        f"## {section_no}. Questions to sharpen your thinking",
        '<p class="dp-section-intro">These are not accusations. They push you to add evidence, '
        "scope, and judgement — the answers are yours to write.</p>",
    ]
    for i, q in enumerate(rows, 1):
        quote = str(q.get("anchor_quote") or "").strip()
        question = str(q.get("question") or "").strip()
        claim = f'{i}. Claim: <em>“{escape(quote)}”</em>' if quote else f"{i}."
        out.append(
            '<div class="dp-q-card">'
            f'<p class="dp-q-claim">{claim}</p>'
            f'<p class="dp-q-body">{escape(question)}</p>'
            '<p class="dp-q-target">Revision target: answer this in one specific sentence before '
            "rewriting the paragraph.</p></div>"
        )
    return "\n".join(out)


# ── PAGE-PARITY sections (owner rule 2026-07-06: scan PDF == scan page) ──────
# Everything below mirrors the scan page's components 1:1, reading the SAME
# report-JSON fields the frontend reads (scan_intelligence.document.segments /
# .paragraphs). KEEP IN SYNC with draftproof-frontend/src/pages/report/
# reportHelpers.js (buildFixFirstItems, buildParagraphSeverityBar,
# debertaSeverityColor) and SignalHighlights.jsx.

# DeBERTa score -> color, matching the page's per-sentence heatmap scale
# exactly (reportHelpers.js DEBERTA_SEVERITY_COLORS): red >=99, orange 90-98,
# amber 80-89, below 80 = clean.
_DEBERTA_SEVERITY_COLORS = ((99, "#dc2626"), (90, "#f97316"), (80, "#f59e0b"))
_DENSITY_CLEAN_COLOR = "#16a34a"


def _doc_intel(data: dict) -> dict:
    return ((data or {}).get("scan_intelligence") or {}).get("document") or {}


def _deberta_score_of(segment: dict):
    for s in segment.get("signals") or []:
        if s.get("key") == "ai_signal_deberta":
            try:
                return float(s.get("score") or 0)
            except (TypeError, ValueError):
                return None
    return None


def _severity_color(score) -> str:
    if isinstance(score, (int, float)):
        for floor, color in _DEBERTA_SEVERITY_COLORS:
            if score >= floor:
                return color
    return ""


def _segments_by_paragraph(data: dict) -> list[tuple[dict, list[dict]]]:
    """[(paragraph_row, [segment, ...]), ...] in document order."""
    intel = _doc_intel(data)
    segs = intel.get("segments") or []
    paras = intel.get("paragraphs") or []
    by_pid: dict[str, list[dict]] = {}
    for seg in segs:
        by_pid.setdefault(str(seg.get("paragraph_id") or "p001"), []).append(seg)
    ordered = sorted(paras, key=lambda p: int(p.get("start_char") or 0))
    out = []
    seen = set()
    for p in ordered:
        pid = str(p.get("paragraph_id") or "")
        seen.add(pid)
        out.append((p, by_pid.get(pid, [])))
    for pid, group in by_pid.items():  # paragraphs missing a row (defensive)
        if pid not in seen:
            out.append(({"paragraph_id": pid}, group))
    return out


# allow-hardcode: student-facing display copy — verbatim from the page's i18n
# (report.whatToFixFirst.*); KEEP IN SYNC with i18n/en/report.js.
def render_fix_first(data: dict, section_no: int, low_tone: bool) -> str:
    """The page's FixFirstChecklist ('Optional polish' / 'What to fix first'):
    up to 3 flagged paragraphs, each titled by its strongest flagged sentence
    (mirrors reportHelpers.buildFixFirstItems + the paragraph readerSummary
    formula). '' when the report has no flagged paragraphs."""
    rows = []
    for p, segs in _segments_by_paragraph(data):
        flagged = sorted(
            (s for s in segs if (_deberta_score_of(s) or 0) >= _DEBERTA_SEVERITY_COLORS[-1][0]),
            key=lambda s: -(_deberta_score_of(s) or 0),
        )
        if not flagged:
            continue
        top = flagged[0]
        words = str(top.get("text") or "").split()
        snippet = " ".join(words[:14]) + ("…" if len(words) > 14 else "")
        count = len(flagged)
        lead = (f"{count} sentences in this paragraph read this way; the strongest is"
                if count > 1 else "The strongest is")
        sids = [str(s.get("sentence_id") or "") for s in segs if s.get("sentence_id")]
        chip = (f"{sids[0].upper()}–{sids[-1].upper()}" if len(sids) > 1
                else (sids[0].upper() if sids else ""))
        rows.append((chip, f'{lead} "{snippet}".'))
        if len(rows) == 3:
            break
    if not rows:
        return ""
    if low_tone:
        kicker, title = "Polish", "Optional polish"
        intro = ("This already reads as your own work. To make it even harder to question, "
                 "ground these claims in a specific only you could know — a name, a number, "
                 "an observation. Sentence-level notes are in the highlights below.")
    else:
        kicker, title = "Repair Plan", "What to fix first"
        intro = ("These paragraphs are driving the report. Ground each claim in a specific "
                 "only you could know — a name, a number, an observation — then put it in "
                 "your own voice. Sentence-level fixes are in the highlights below.")
    items = "".join(
        f'<div class="dp-policy-row dp-policy-row--info">'
        f'<span class="dp-policy-name">{i}. {escape(text)}</span>'
        + (f'<span class="dp-policy-issue">{escape(chip)}</span>' if chip else "")
        + "</div>"
        for i, (chip, text) in enumerate(rows, 1)
    )
    return (f"## {section_no}. {title}\n"
            f'<p class="dp-section-intro">{escape(kicker)} — {escape(intro)}</p>\n'
            + items)


def render_signal_highlights_intro(data: dict) -> str:
    """The page's Signal-highlights header block: flagged-paragraph count, the
    per-sentence finding-density bar, and the 'Passages to review' chip.
    '' when the report carries no segments (legacy JSON)."""
    groups = _segments_by_paragraph(data)
    if not any(segs for _, segs in groups):
        return ""
    all_segs = [s for _, segs in groups for s in segs]
    total_len = sum(max(1, len(str(s.get("text") or ""))) for s in all_segs) or 1
    blocks = []
    for s in all_segs:
        width = max(0.4, len(str(s.get("text") or "")) / total_len * 100)
        color = _severity_color(_deberta_score_of(s)) or _DENSITY_CLEAN_COLOR
        blocks.append(
            f'<span style="display:inline-block;height:10px;width:{width:.2f}%;'
            f'background:{color};border-right:1px solid #fff"></span>'
        )
    flagged_paras = sum(
        1 for _, segs in groups
        if any((_deberta_score_of(s) or 0) >= _DEBERTA_SEVERITY_COLORS[-1][0] for s in segs)
    )
    out = [
        '<p class="dp-hero-sub"><b>Finding density by document</b> — '
        f"{flagged_paras} flagged paragraph{'s' if flagged_paras != 1 else ''}</p>",
        f'<div style="line-height:0;white-space:nowrap">{"".join(blocks)}</div>',
        '<p class="dp-hero-sub">Clean / fewer findings → denser / more severe</p>',
    ]
    if flagged_paras:
        out.append(_statchip(f"Passages to review (beta deep-scan): {flagged_paras}", "warn"))
    return "\n".join(out)


def render_highlighted_document(data: dict, original_text: str) -> str:
    """The page's 'Read full document' view: the submitted text with each
    classifier-flagged sentence underlined in its severity color. Falls back
    to '' when segments are absent (caller keeps the plain-text appendix)."""
    groups = _segments_by_paragraph(data)
    if not any(segs for _, segs in groups):
        return ""
    paras_html = []
    for _p, segs in groups:
        if not segs:
            continue
        parts = []
        for s in sorted(segs, key=lambda x: int(x.get("start_char") or 0)):
            text = escape(str(s.get("text") or "").strip())
            if not text:
                continue
            color = _severity_color(_deberta_score_of(s))
            if color:
                parts.append(f'<span style="color:{color};text-decoration:underline;'
                             f'font-weight:600">{text}</span>')
            else:
                parts.append(text)
        if parts:
            paras_html.append(f'<p>{" ".join(parts)}</p>')
    if not paras_html:
        return ""
    return ('<div class="dp-hero" style="border-left-color:#94a3b8">'
            "<p class=\"dp-hero-sub\"><b>Read full document</b> — flagged sentences "
            "are underlined in their severity color, matching the report page.</p>"
            + "".join(paras_html) + "</div>")
