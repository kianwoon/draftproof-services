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
    "text_pattern": "May draw external-detector attention, but it's a heads-up, not a verdict.",
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


def render_scan_lead(report, data) -> str:
    """Hero + KPI row + priority fixes + '1. Submission and policy view'.

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
    read = f"Overall read: {sr_label}"
    if hero_phrase:
        read += f" — but the draft needs {hero_phrase} before submission"
    read += "."
    sub = ("The text-pattern signal is not the main problem. ")
    if driver_action:
        sub += f"The bigger issue: {driver_action[0].lower() + driver_action[1:]}."
    else:
        sub += "The bigger issue is whether claims are tied to named evidence and your own judgement."

    chips = [(f"{_SR_LEVEL_LABELS.get(sr_level, sr_level)} submission risk", _level_kind(sr_level))]
    if dp_band.get("tier"):
        sig = {"GREEN": "Low", "AMBER": "Moderate", "ORANGE": "Elevated", "RED": "Elevated"}.get(
            str(dp_band["tier"]).upper(), "")
        if sig:
            chips.append((f"{sig} AI-writing signal", "good" if str(dp_band["tier"]).upper() == "GREEN" else "warn"))
    if driver_label:
        chips.append((driver_label, "warn"))
    wc = doc_ctx.get("word_count") or 0
    if wc and wc < 250:
        chips.append(("Low sample confidence", "info"))

    out.append(
        f'<div class="dp-hero dp-hero--{_level_kind(sr_level)}">'
        f'<p class="dp-hero-read">{escape(read)}</p>'
        f'<p class="dp-hero-sub">{escape(sub)}</p>'
        '<div class="dp-chip-strip">'
        + "".join(_statchip(t, k) for t, k in chips)
        + "</div></div>"
    )

    # ── KPI stat row ───────────────────────────────────────────────
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
    if isinstance(overall.get("risk"), (int, float)):
        kpis.append((f"{round(overall['risk'])}%", "Submission risk score"))
    if isinstance(dp_band.get("score"), (int, float)):
        kpis.append((f"{round(dp_band['score'])}%", "Text-pattern trigger"))
    if isinstance(contribution.get("hcr"), (int, float)):
        kpis.append((f"{round(contribution['hcr'])}%", "Human contribution"))
    if wc:
        kpis.append((f"{wc}", "Words scanned"))
    if len(kpis) >= 2:
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
    if ranked:
        items = "".join(f"<li>{escape(_DIAG_BUCKET_FIX[k])}</li>" for k in ranked)
        out.append(
            '<div class="dp-callout dp-callout--good">'
            '<span class="dp-callout-title">Priority fixes</span>'
            f"<ol>{items}</ol></div>"
        )

    # ── 1. Submission and policy view ──────────────────────────────
    out.append("## 1. Submission and policy view")
    out.append(
        '<p class="dp-section-intro">This separates three things students often mix up: '
        "school policy, external-detector attention, and whether you can defend the work as your own.</p>"
    )

    pr = badge.get("policy_risk") or {}
    pa, prr = pr.get("ai_allowed") or {}, pr.get("ai_restricted") or {}
    policy_cards = []
    if pa.get("level") and pa["level"] != "unknown":
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
            if key == "text_pattern" and isinstance(ax.get("display_score"), (int, float)):
                current += f" — AI-likelihood ~{round(ax['display_score'])}%"
        meaning = _SR_AXIS_MEANING.get(key, "")
        axis_rows.append((label, current, meaning))
    if axis_rows:
        out.append("")
        out.append("| Axis | Current read | What it means |")
        out.append("|------|--------------|----------------|")
        for a, c, m in axis_rows:
            out.append(f"| {a} | {c} | {m} |")
        out.append("")

    # Plain-English verdict callout (amber).
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


def _num(v):
    return v if isinstance(v, (int, float)) else None


def rewrite_hero(*, result_label, explanation, outcome, ai_improved, score_worse,
                 original_preserved, orig_ai, new_ai, orig_human, new_human,
                 orig_wq, new_wq, o_total, n_total) -> str:
    """Hero result panel + before→after KPI row for the rewrite report.

    Mirrors the scan report's hero/KPI design. All values are pre-computed by
    render_rewrite_report (already in the right units)."""
    good = (str(outcome) == "ai_mitigated") or (ai_improved and not score_worse)
    kind = "good" if good else ("info" if original_preserved else "warn")

    chips = [(result_label, kind)]
    if _num(orig_ai) is not None and _num(new_ai) is not None:
        chips.append((f"AI likelihood {orig_ai:.0f}→{new_ai:.0f}%",
                      "good" if new_ai < orig_ai else ("warn" if new_ai > orig_ai else "info")))
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


def render_question_cards(badge) -> str:
    """Restyled Critical-Thinking questions as purple cards. "" when no questions."""
    ctc = (badge or {}).get("critical_thinking_control") or {}
    rows = [q for q in (ctc.get("questions") or [])
            if isinstance(q, dict) and str(q.get("question") or "").strip()]
    if not rows:
        return ""
    out = [
        "## 3. Questions to sharpen your thinking",
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
