"""DraftProof Report Renderer — Markdown output from DraftReport.

Produces a structured report with:
  1. EXECUTIVE SUMMARY — overall tier badge, risk gauge, scanner scores
  2. FINDINGS BY SEVERITY — grouped by tier with collapsible details
  3. SCANNER BREAKDOWN — per-scanner tables with sentence-level detail
  4. REWRITE SUMMARY — before/after comparison (when present)
  5. APPENDIX — full sentence tables in collapsible sections
"""

from html import escape
from typing import Optional

from detect.transformation import TRANSFORMATION_SIGNAL_METADATA, transformation_signal_metadata

from .report import DraftReport, Tier, TIER_ICON, report_to_dict

# ── Scanner & Signal legend codes ──────────────────────────────────────
_SCANNER_CODES = {
    "predictability": "P",
    "ai_generation": "AI",
    "similarity": "S",
    "citation": "C",
}

_SIGNAL_CODES = {
    # Predictability
    "high_predictability": "HP",
    "medium_predictability": "MP",
    "low_predictability": "LP",
    "review_predictability": "RP",
    "high_topk_predictability": "HT",
    "style_shift": "SS",
    "low_burstiness": "LB",
    "repetitive_sentence_structure": "RS",
    "uniform_paragraph_structure": "UP",
    # Genericity
    "generic_phrase": "GP",
    "generic_policy_claim": "GC",
    "broad_education_claim": "BE",
    "formulaic_conclusion": "FC",
    "template_personal_reflection": "TP",
    "weak_source_grounding": "WG",
    "low_specificity": "LS",
    # Authorship / AI
    "elevated_ai_generation_likelihood": "EA",
    "high_ai_generation_likelihood": "HA",
    "moderate_ai_generation_likelihood": "MA",
    "medium_ai_generation_likelihood": "ME",
    "low_ai_generation_likelihood": "LA",
    "minimal_ai_generation_likelihood": "MI",
    "similarity_overlap": "SO",
    # Similarity
    "high_similarity": "HS",
    "medium_similarity": "MS",
    "paragraph_level_overlap": "PO",
    # Citation / Writing quality
    "missing_citation": "MC",
    "broken_citation": "BC",
    "uncited_claim": "UC",
    "grammar_issue": "GI",
    "fragment_sentence": "FS",
    "spelling_issue": "SI",
    "punctuation_issue": "PI",
}

_RISK_CODES = {
    "critical": "C",
    "high": "H",
    "medium": "M",
    "low": "L",
    "review": "R",
    "info": "I",
}

_FILTER_CODES = {
    "GlossaryFilter": "GL",
    "AcademicFilter": "AC",
    "CitationFilter": "CF",
    "PatternFilter": "PF",
    "LengthFilter": "LF",
    "QuotedSpeechFilter": "QS",
    "DomainFilter": "DF",
    "StructuralFilter": "SF",
}

_SIGNAL_CHART_ORDER = [
    "topk_pattern",
    "adjusted_ai_risk",
    "ai_likelihood",
    "calibrated_ai_risk",
    "calibration_confidence",
    "discourse_regularity_risk",
    "outline_to_text_expansion",
    "citation_grounding_risk",
    "human_anchor_score",
    "human_anchor_discount",
    "paraphrase_transformation_risk",
    "section_style_variance",
    "reporting_suppression",
    "rewrite_smoothness",
    "semantic_uniformity_risk",
    "signal_agreement_score",
    "source_similarity",
    "surface_similarity",
]

_SIGNAL_CHART_COLORS = {
    "topk_pattern": "#be123c",
    "ai_likelihood": "#c2410c",
    "adjusted_ai_risk": "#c2410c",
    "calibrated_ai_risk": "#c2410c",
    "citation_grounding_risk": "#c2410c",
    "human_anchor_score": "#16a34a",
    "human_anchor_discount": "#16a34a",
    "rewrite_smoothness": "#4f46e5",
    "outline_to_text_expansion": "#4f46e5",
    "discourse_regularity_risk": "#4f46e5",
    "source_similarity": "#0891b2",
    "surface_similarity": "#0891b2",
    "paraphrase_transformation_risk": "#0891b2",
    "semantic_uniformity_risk": "#7c3aed",
    "section_style_variance": "#2563eb",
    "signal_agreement_score": "#0f766e",
    "calibration_confidence": "#0f766e",
    "reporting_suppression": "#64748b",
}

# ── Layman labels for AI Likelihood components ──────────────────────────
_AI_COMPONENT_LABELS = {
    "predictability": ("Predictability", "How predictable the word choices are — higher means the text reads like statistically common patterns"),
    "topk_pattern": ("Common Word Patterns", "How many words are among the most statistically likely choices — AI models heavily favour these"),
    "generic_phrase_density": ("Generic Phrases", "Density of overused filler phrases commonly found in templated or AI-generated writing"),
    "burstiness_risk": ("Uniform Sentence Length", "How similar sentence lengths are across the text — human writing naturally varies more"),
    "repeated_sentence_structure_risk": ("Repeated Sentence Openings", "How often sentences start with the same or similar phrases"),
    "generic_assertion_risk": ("Generic Claims", "How many claims are broad and unspecific, lacking concrete evidence or examples"),
    "balanced_hedging_risk": ("Balanced Hedging", "Use of formulaic balancing phrases like 'on one hand... on the other hand' or 'while... it also'"),
    "style_shift_risk": ("Style Consistency", "Sudden shifts in writing style across sections — may indicate stitching from multiple sources"),
}

# ── Layman labels for Writing Quality components ────────────────────────
_WQ_COMPONENT_LABELS = {
    "broad_claim_risk": ("Broad Claims", "Claims that are too general and lack specific evidence or examples"),
    "lived_detail_risk": ("Weak Personal Detail", "Lack of specific, concrete experiences that show genuine first-hand engagement"),
    "citation_weakness_risk": ("Citation Weakness", "Missing, broken, or insufficient citation support for claims"),
    "unsupported_claim_risk": ("Unsupported Claims", "Statements presented as fact without any source or evidence"),
    "source_grounding_risk": ("Source Grounding", "How well the writing is grounded in verifiable sources rather than speculation"),
    "paragraph_progression_risk": ("Formulaic Structure", "Paragraphs following a predictable, template-like progression"),
    "signpost_paragraph_risk": ("Signpost Paragraphs", "Paragraphs that mainly signal what comes next rather than adding substance"),
    "paragraph_uniformity_risk": ("Paragraph Uniformity", "All paragraphs being similar in structure and length"),
    "repeated_starter_risk": ("Repeated Openings", "Multiple paragraphs starting with the same word or phrase"),
    "formulaic_conclusion_risk": ("Formulaic Conclusion", "Conclusion follows a predictable template pattern"),
    "source_grounding_strength": ("Source Strength", "Presence and quality of cited sources that back up the writing"),
    "domain_grounding_strength": ("Domain Knowledge", "Use of domain-specific terminology and concepts that show subject familiarity"),
    "grounding_credit": ("Grounding Credit", "Bonus credit applied when strong sources or domain knowledge are present"),
}

def _tf_pct(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) <= 1:
        number *= 100
    return max(0.0, min(100.0, number))


def _transformation_signals(features: dict) -> list[dict]:
    rows = []
    for key in TRANSFORMATION_SIGNAL_METADATA:
        value = _tf_pct((features or {}).get(key))
        if value is not None:
            meta = transformation_signal_metadata(key)
            rows.append({
                "key": key,
                "label": meta["label"],
                "description": meta["description"],
                "score": value,
            })
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def _transformation_contribution_summary(features: dict, signals: list[dict]) -> dict:
    human_anchor = _tf_pct((features or {}).get("human_anchor_score")) or 0.0
    grounding_quality = 100.0 - (_tf_pct((features or {}).get("citation_grounding_risk")) or 0.0)
    semantic_originality = 100.0 - max(
        _tf_pct((features or {}).get("source_similarity")) or 0.0,
        _tf_pct((features or {}).get("surface_similarity")) or 0.0,
    )

    ai_likelihood = (
        _tf_pct((features or {}).get("calibrated_ai_risk"))
        if (features or {}).get("calibrated_ai_risk") is not None
        else (
            _tf_pct((features or {}).get("adjusted_ai_risk"))
            if (features or {}).get("adjusted_ai_risk") is not None
            else _tf_pct((features or {}).get("ai_likelihood"))
        )
    ) or 0.0
    rewrite_smoothness = _tf_pct((features or {}).get("rewrite_smoothness")) or 0.0
    expansion = _tf_pct((features or {}).get("outline_to_text_expansion")) or 0.0
    patchwork = _tf_pct((features or {}).get("section_style_variance")) or 0.0
    grounding_risk = _tf_pct((features or {}).get("citation_grounding_risk")) or 0.0
    source_similarity = _tf_pct((features or {}).get("source_similarity")) or 0.0

    human_raw = human_anchor * 0.55 + grounding_quality * 0.25 + semantic_originality * 0.20
    ai_raw = (
        ai_likelihood * 0.35
        + rewrite_smoothness * 0.20
        + expansion * 0.15
        + grounding_risk * 0.15
        + patchwork * 0.10
        + source_similarity * 0.05
    )
    total = max(human_raw + ai_raw, 1.0)
    hcr = round((human_raw / total) * 100)
    atr = 100 - hcr

    top_drivers = [
        row["label"].lower()
        for row in signals
        if row.get("key") != "human_anchor_score"
    ][:4]

    if atr >= 70:
        summary = "AI transformation dominates this scan pattern."
    elif atr >= 55:
        summary = "AI transformation signals are stronger than the human anchor."
    elif hcr >= 65:
        summary = "Human contribution remains the stronger signal."
    else:
        summary = "Mixed authorship pattern: human anchoring and AI transformation signals are both visible."
    if top_drivers:
        summary += f" Main drivers: {' and '.join(top_drivers)}."

    return {
        "hcr": hcr,
        "atr": atr,
        "adjusted_ai_risk": round(_tf_pct((features or {}).get("adjusted_ai_risk")) or 0.0),
        "calibrated_ai_risk": round(ai_likelihood),
        "human_anchor_discount": round(_tf_pct((features or {}).get("human_anchor_discount")) or 0.0),
        "calibration_confidence": round(_tf_pct((features or {}).get("calibration_confidence")) or 0.0),
        "reporting_suppression": round(_tf_pct((features or {}).get("reporting_suppression")) or 0.0),
        "summary": summary,
    }


def _signal_chart_rows(features: dict, badge: dict | None = None) -> list[dict]:
    rows_by_key = {row["key"]: row for row in _transformation_signals(features)}
    topk_score = _tf_pct(((badge or {}).get("ai_components") or {}).get("topk_pattern"))
    if topk_score is not None and "topk_pattern" not in rows_by_key:
        label, description = _AI_COMPONENT_LABELS["topk_pattern"]
        rows_by_key["topk_pattern"] = {
            "key": "topk_pattern",
            "label": label,
            "description": description,
            "score": topk_score,
        }
    ordered = []
    for key in _SIGNAL_CHART_ORDER:
        row = rows_by_key.get(key)
        if row:
            ordered.append(row)
    ordered.extend(
        row
        for row in rows_by_key.values()
        if row.get("key") not in _SIGNAL_CHART_ORDER
    )
    return ordered


def _summary_stat_html(label: str, value: str, *, color: str = "#111827", extra_class: str = "") -> str:
    return (
        f'<div class="dp-summary-stat {extra_class}">'
        f'<span class="dp-summary-value" style="color:{color}">{escape(str(value))}</span>'
        f'<span class="dp-summary-label">{escape(label)}</span>'
        "</div>"
    )


_CALIBRATED_AUTHORSHIP_LEVELS = [
    {
        "min": 60,
        "level": 4,
        "label": "Strong AI-Style Signal",
        "short_label": "Strong AI Signal",
        "code": "ai_generated_signals",
    },
    {"min": 45, "level": 3, "label": "Likely AI-Assisted", "short_label": "Likely AI-Assisted", "code": "likely_ai"},
    {
        "min": 32,
        "level": 2,
        "label": "Possible AI-Assisted",
        "short_label": "Possible AI-Assisted",
        "code": "possible_ai_assisted",
    },
    {"min": 20, "level": 1, "label": "Unlikely AI-Assisted", "short_label": "Unlikely AI-Assisted", "code": "unlikely_ai"},
    {"min": 0, "level": 0, "label": "Good", "short_label": "Good", "code": "low_ai_signal"},
]


def _rating_for_calibrated_score(score: float) -> dict:
    for rating in _CALIBRATED_AUTHORSHIP_LEVELS:
        if score >= rating["min"]:
            return dict(rating)
    return dict(_CALIBRATED_AUTHORSHIP_LEVELS[-1])


def _authorship_rating_from_calibrated_risk(score, topk_score=None) -> dict:
    calibrated_score = _tf_pct(score)
    topk_score = _tf_pct(topk_score)
    rating = _rating_for_calibrated_score(calibrated_score) if calibrated_score is not None else {}

    def _apply_topk_floor(floor_code: str) -> None:
        nonlocal rating
        floor = next((dict(item) for item in _CALIBRATED_AUTHORSHIP_LEVELS if item["code"] == floor_code), {})
        if floor and (not rating or rating.get("level", -1) < floor["level"]):
            rating = {**floor, "topk_escalated": True, "topk_score": topk_score}

    if topk_score is not None:
        if topk_score >= 80:
            _apply_topk_floor("likely_ai")
        elif topk_score >= 70:
            _apply_topk_floor("possible_ai_assisted")

    if not rating:
        return {}
    rating["score"] = calibrated_score
    rating["topk_score"] = rating.get("topk_score", topk_score)
    return rating


def _authorship_rating_tone(rating: dict) -> dict:
    code = str((rating or {}).get("code") or (rating or {}).get("short_label") or (rating or {}).get("label") or "").lower()
    if "low_ai_signal" in code or "low signal" in code:
        return {"color": "#15803d", "bg": "#f0fdf4"}
    if "unlikely" in code:
        return {"color": "#0f766e", "bg": "#f0fdfa"}
    if "possible" in code:
        return {"color": "#b45309", "bg": "#fff7ed"}
    if "likely" in code:
        return {"color": "#c2410c", "bg": "#fff7ed"}
    if "generated" in code or "signals" in code:
        return {"color": "#b91c1c", "bg": "#fef2f2"}
    return {"color": "#334155", "bg": "#f8fafc"}


def _display_authorship_rating_from_badge(badge: dict) -> dict:
    features = (((badge or {}).get("transformation_classification") or {}).get("features") or {})
    topk_score = ((badge or {}).get("ai_components") or {}).get("topk_pattern")
    calibrated = _authorship_rating_from_calibrated_risk(features.get("calibrated_ai_risk"), topk_score)
    return calibrated or _authorship_rating_from_badge(badge)


def _executive_signal_chart_html(
    report: DraftReport,
    data: dict,
    *,
    n_critical: int,
    n_high: int,
    n_medium: int,
    n_low: int,
    total: int,
) -> str:
    badge = report.ai_risk_badge or {}
    transformation = badge.get("transformation_classification") or {}
    features = transformation.get("features") or {}
    rows = _signal_chart_rows(features, badge)
    if not badge or not transformation or not rows:
        return ""

    badge_tier = badge.get("tier", "")
    tier_label = _BADGE_TIER_LABELS.get(badge_tier, badge_tier or report.overall_tier.value.title())
    tier_color = {
        "GREEN": "#15803d",
        "AMBER": "#ca8a04",
        "ORANGE": "#b45309",
        "RED": "#b91c1c",
    }.get(badge_tier, "#334155")
    tier_bg = {
        "GREEN": "#ecfdf5",
        "AMBER": "#fefce8",
        "ORANGE": "#fff7ed",
        "RED": "#fef2f2",
    }.get(badge_tier, "#f8fafc")

    rating = _display_authorship_rating_from_badge(badge)
    rating_label = (
        rating.get("short_label")
        or rating.get("label")
        or badge.get("authorship_rating_label")
        or "Not Rated"
    )
    rating_tone = _authorship_rating_tone(rating)
    calibrated_score = _tf_pct(features.get("calibrated_ai_risk"))
    topk_score = _tf_pct((badge.get("ai_components") or {}).get("topk_pattern"))
    if rating.get("topk_escalated") and topk_score is not None:
        rating_detail = f"{topk_score:.0f}% top-k signal"
    elif calibrated_score is not None:
        rating_detail = f"{calibrated_score:.0f}% calibrated risk"
    else:
        rating_detail = f"{(_tf_pct(badge.get('ai_likelihood_score')) or 0.0):.0f}% raw signal"
    ai_score = _tf_pct(badge.get("ai_likelihood_score")) or 0.0
    writing_score = _tf_pct(badge.get("writing_quality_score")) or 0.0
    contribution = _transformation_contribution_summary(features, rows)

    stats = [
        (
            '<div class="dp-summary-stat dp-risk-stat" style="background:'
            f'{tier_bg};border-color:{tier_color}33">'
            f'<span class="dp-risk-icon" style="color:{tier_color}">◌</span>'
            '<span>'
            f'<span class="dp-summary-value" style="color:{tier_color}">{escape(tier_label)}</span>'
            '<span class="dp-summary-label">Risk Tier</span>'
            '</span></div>'
        ),
        _summary_stat_html("Total Findings", str(total)),
        _summary_stat_html("Authorship Rating", rating_label, color=rating_tone["color"]),
        _summary_stat_html("Raw AI-Style Signal", f"{ai_score:.2f}%", color=tier_color),
        _summary_stat_html("Writing Score", f"{writing_score:.2f}%", color="#4f46e5"),
    ]
    severity_stats = [
        ("Critical", n_critical, "#b91c1c"),
        ("High", n_high, "#b91c1c"),
        ("Medium", n_medium, "#b45309"),
        ("Low", n_low, "#15803d"),
    ]
    for label, count, color in severity_stats:
        if count:
            stats.append(_summary_stat_html(label, str(count), color=color))

    confidence = transformation.get("confidence")
    pills = []
    if confidence:
        pills.append(f"{str(confidence).title()} Confidence")
    pills.append("Not A Verdict")

    adjustment_chips = [
        f"Calibrated AI risk {contribution['calibrated_ai_risk']}%",
        f"Human anchor discount {contribution['human_anchor_discount']}%",
        f"Calibration confidence {contribution['calibration_confidence']}%",
        f"Reporting suppression {contribution['reporting_suppression']}%",
    ]

    evidence = transformation.get("evidence") or []
    doc_ctx = data.get("document_context", {}) if isinstance(data, dict) else {}
    confidence_note = ""
    if doc_ctx:
        word_count = doc_ctx.get("word_count", 0)
        sent_count = doc_ctx.get("sentence_count", 0)
        if word_count < 250 or sent_count < 10:
            confidence_note = (
                f"Low sample confidence: {word_count} words / {sent_count} sentences. "
                "Document-level scores are unstable."
            )
        elif word_count >= 800 or sent_count >= 25:
            confidence_note = (
                f"Medium-high sample confidence: {word_count} words / {sent_count} sentences."
            )
        else:
            confidence_note = (
                f"Medium sample confidence: {word_count} words / {sent_count} sentences."
            )

    html = [
        '<div class="dp-executive-chart">',
        '<div class="dp-summary-bar">',
        *stats,
        '</div>',
        '<section class="dp-signal-card">',
        '<header class="dp-signal-header">',
        '<div class="dp-signal-title-row">',
        '<div class="dp-signal-icon">⇄</div>',
        '<div>',
        '<span class="dp-kicker">Transformation Pattern</span>',
        f'<h3>{escape(transformation.get("label") or "Pattern analysis")}</h3>',
        '<div class="dp-pill-row">',
        ''.join(f'<span>{escape(pill)}</span>' for pill in pills),
        '</div>',
        '</div>',
        '</div>',
        (
            '<div class="dp-rating-seal" style="'
            f'--rating-color:{rating_tone["color"]};--rating-bg:{rating_tone["bg"]};'
            f'color:{rating_tone["color"]};border-color:{rating_tone["color"]};background:{rating_tone["bg"]}">'
        ),
        '<span>Authorship Rating</span>',
        f'<strong>{escape(rating_label)}</strong>',
        f'<em>{escape(rating_detail)}</em>',
        '</div>',
        '</header>',
        '<div class="dp-scan-head">',
        '<div>',
        '<span>Original Scan</span>',
        f'<strong>{escape(transformation.get("label") or "Pattern analysis")}</strong>',
        '</div>',
        f'<em>{ai_score:.1f}%</em>',
        '</div>',
        '<div class="dp-ratio-card">',
        '<div class="dp-ratio-copy">',
        '<span>Estimated Contribution</span>',
        f'<p>{escape(contribution["summary"])}</p>',
        '<div class="dp-chip-row">',
        ''.join(f'<strong>{escape(chip)}</strong>' for chip in adjustment_chips),
        '</div>',
        '</div>',
        '<div class="dp-ratio-bars">',
        '<div class="dp-ratio-row">',
        '<span>Human Contribution</span>',
        f'<strong>{contribution["hcr"]}%</strong>',
        '<div class="dp-bar-track">',
        f'<div class="dp-ratio-fill dp-human" style="width:{contribution["hcr"]}%"></div>',
        '</div></div>',
        '<div class="dp-ratio-row">',
        '<span>AI Transformation</span>',
        f'<strong>{contribution["atr"]}%</strong>',
        '<div class="dp-bar-track">',
        f'<div class="dp-ratio-fill dp-ai" style="width:{contribution["atr"]}%"></div>',
        '</div></div>',
        '</div>',
        '</div>',
        '<h3 class="dp-core-heading">Core Signals</h3>',
        '<div class="dp-core-bars">',
    ]

    for row in rows:
        key = row["key"]
        score = max(0, min(100, row.get("score") or 0))
        color = _SIGNAL_CHART_COLORS.get(key, "#0f766e")
        html.extend([
            '<div class="dp-core-row">',
            '<div class="dp-core-label">',
            f'<span>{escape(row["label"])}</span>',
            f'<strong>{score:.0f}%</strong>',
            '</div>',
            '<div class="dp-bar-track">',
            f'<div class="dp-core-fill" style="width:{score:.0f}%;background:{color}"></div>',
            '</div>',
            '</div>',
        ])

    html.extend([
        '</div>',
    ])
    if evidence:
        html.extend([
            '<div class="dp-evidence-row">',
            ''.join(f'<span>{escape(str(item))}</span>' for item in evidence[:3]),
            '</div>',
        ])
    if confidence_note:
        html.append(f'<p class="dp-confidence-note">{escape(confidence_note)}</p>')
    html.extend([
        '</section>',
        '</div>',
        '',
    ])
    return "\n".join(html)

# ── Tier display config ──────────────────────────────────────────────

_BADGE_TIER_LABELS = {
    "GREEN": "Low Risk",
    "AMBER": "Moderate Risk",
    "ORANGE": "High Risk",
    "RED": "Very High Risk",
}


def _authorship_rating_from_badge(badge: dict) -> dict:
    rating = (badge or {}).get("authorship_rating") or {}
    if rating:
        return rating

    label = (badge or {}).get("authorship_rating_label")
    code = (badge or {}).get("authorship_rating_code")
    if label:
        return {"label": label, "code": code or ""}

    score = (badge or {}).get("ai_likelihood_score")
    if not isinstance(score, (int, float)):
        return {}
    ai_score = score * 100 if abs(score) <= 1 else score
    writing_score = (badge or {}).get("writing_quality_score")
    writing_score = (
        writing_score * 100
        if isinstance(writing_score, (int, float)) and abs(writing_score) <= 1
        else writing_score
    )
    tier = str((badge or {}).get("tier") or "").upper()
    ai_components = (badge or {}).get("ai_components") or {}
    writing_components = (badge or {}).get("writing_components") or {}

    def _component_score(values: dict, key: str) -> float:
        value = values.get(key)
        if not isinstance(value, (int, float)):
            return 0.0
        return value * 100 if abs(value) <= 1 else value

    high_quality = (
        isinstance(writing_score, (int, float))
        and writing_score >= 65
        and str((badge or {}).get("writing_quality_tier") or "").upper() in {"HIGH_REVIEW", ""}
    )
    high_component_alignment = (
        ai_score >= 58
        and (
            _component_score(ai_components, "topk_pattern") >= 80
            or _component_score(ai_components, "qualifying_text_ai_density") >= 70
        )
        and _component_score(ai_components, "generic_assertion_risk") >= 80
        and (
            _component_score(writing_components, "unsupported_claim_risk") >= 80
            or _component_score(writing_components, "source_grounding_risk") >= 70
            or _component_score(writing_components, "broad_claim_risk") >= 70
        )
    )
    high_density_alignment = (
        ai_score >= 45
        and _component_score(ai_components, "qualifying_text_ai_density") >= 70
        and _component_score(ai_components, "topk_pattern") >= 60
        and _component_score(ai_components, "generic_assertion_risk") >= 80
        and (
            _component_score(writing_components, "unsupported_claim_risk") >= 75
            or _component_score(writing_components, "source_grounding_risk") >= 65
            or _component_score(writing_components, "broad_claim_risk") >= 65
        )
    )

    likely_component_alignment = (
        ai_score >= 48
        and (
            _component_score(ai_components, "topk_pattern") >= 70
            or _component_score(ai_components, "generic_assertion_risk") >= 70
            or (
                isinstance(writing_score, (int, float))
                and writing_score >= 55
                and (
                    _component_score(writing_components, "unsupported_claim_risk") >= 70
                    or _component_score(writing_components, "source_grounding_risk") >= 70
                    or _component_score(writing_components, "broad_claim_risk") >= 65
                )
            )
        )
    )

    if ai_score >= 65 or tier == "RED" or (ai_score >= 60 and high_quality) or high_component_alignment or high_density_alignment:
        return {
            "label": "AI-Generated / AI-Paraphrased Signals",
            "short_label": "AI Signals",
            "summary": "High AI-style signal strength across the detect pipeline, including patterns consistent with generated or AI-paraphrased text.",
            "confidence": (badge or {}).get("confidence") or "",
            "disclaimer": "This rating summarizes DraftProof detector signals. It is not proof of authorship.",
        }
    if likely_component_alignment:
        return {"label": "Likely AI", "short_label": "Likely AI"}
    if ai_score >= 32 or tier == "AMBER":
        return {"label": "Possible AI-Assisted", "short_label": "Possible AI"}
    if ai_score >= 20:
        return {"label": "Unlikely AI", "short_label": "Unlikely AI"}
    return {
        "label": "Low AI Signal",
        "short_label": "Low Signal",
        "summary": "AI-style signal strength is below the level DraftProof surfaces as an authorship concern.",
        "disclaimer": "Scores under 20% have a higher false-positive risk and should not be treated as an AI finding.",
    }


_TIER_BADGE = {
    Tier.CRITICAL: "![CRITICAL](https://img.shields.io/badge/Turnitin_Tier-CRITICAL-red)",
    Tier.HIGH:     "![HIGH](https://img.shields.io/badge/Turnitin_Tier-HIGH-orange)",
    Tier.MEDIUM:   "![MEDIUM](https://img.shields.io/badge/Turnitin_Tier-MEDIUM-yellow)",
    Tier.LOW:      "![LOW](https://img.shields.io/badge/Turnitin_Tier-LOW-blue)",
    Tier.CLEAN:    "![CLEAN](https://img.shields.io/badge/Turnitin_Tier-CLEAN-green)",
}

_TIER_EMOJI = {
    Tier.CRITICAL: "[!!!]",
    Tier.HIGH:     "[!!]",
    Tier.MEDIUM:   "[!]",
    Tier.LOW:      "[ok]",
    Tier.CLEAN:    "[clean]",
}

_SEVERITY_LABEL = {
    Tier.CRITICAL: "[!!!] Critical",
    Tier.HIGH:     "[!!] High",
    Tier.MEDIUM:   "[!] Medium",
    Tier.LOW:      "[ok] Low",
    Tier.CLEAN:    "[clean] Clean",
}


# ── Helpers ──────────────────────────────────────────────────────────

def _risk_gauge(value: float, width: int = 20) -> str:
    """Render a text risk gauge bar: `[========............] 0.42`"""
    filled = round(value * width)
    empty = width - filled
    bar = "=" * filled + "." * empty
    return f"`[{bar}]` `{value:.1%}`"


def _truncate(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."

def _pct(value) -> str:
    """Format a numeric value as percentage string."""
    try:
        return f"{float(value):.1%}"
    except (ValueError, TypeError):
        return str(value)


# ── Layman translation for finding detail/recommendation ────────────────
import re as _re

def _translate_detail(detail: str) -> str:
    """Convert technical finding detail into plain language for reports."""
    if not detail:
        return detail

    # Predictability: "Sentence scored 50.4% predictability (common ratio: 66.7%, category: statistical predictability)"
    m = _re.match(
        r"Sentence scored ([\d.]+)% predictability \(common ratio: ([\d.]+)%, category: (.+)\)",
        detail,
    )
    if m:
        score, ratio, cat = m.group(1), m.group(2), m.group(3).replace("_", " ")
        return f"Common word patterns detected ({score}% predictability, {ratio}% common words)"

    # Generic phrase: "Generic phrase detected: 'in summary'"
    m = _re.match(r"Generic phrase detected: '(.+)'", detail)
    if m:
        return f"Generic phrase found: \"{m.group(1)}\""

    # Style shift: "Predictability less_predictable (Δ24.7%)"
    m = _re.match(r"Predictability (\w+) \(Δ([\d.]+)%\)", detail)
    if m:
        direction = "more" if "less" in m.group(1) else "less"
        return f"Writing style shifted {direction} predictable here (Δ{m.group(2)}%)"

    # Specificity concern: "Specificity concern: adjusted to 89% review-level..."
    if detail.startswith("Specificity concern:"):
        m2 = _re.match(r"Specificity concern: adjusted to (\d+)%", detail)
        if m2:
            return f"Specificity scored {m2.group(1)}% after adjustment — may affect reliability"
        return "Specificity level adjusted after review"

    # AI-generation likelihood: "AI-generation likelihood: low_ai_generation_likelihood (score 29%)..."
    m = _re.match(r"AI-generation likelihood: (\w+) \(score (\d+)%\)\.?(.*)", detail)
    if m:
        label, score, rest = m.group(1), m.group(2), m.group(3)
        friendly_label = label.replace("_", " ")
        return f"AI likelihood assessed as {friendly_label} ({score}%)"

    # AI criterion: "criterion_name: Details: key: value, ..."
    if ": Details:" in detail:
        crit, rest = detail.split(": Details:", 1)
        crit_name = crit.strip().replace("_", " ").title()
        return f"{crit_name} flagged"

    # Similarity: "Exact Match: exact=92%, fuzzy=85%, semantic=78%"
    m = _re.match(r"(.+?): exact=(\d+)%, fuzzy=(\d+)%, semantic=(\d+)%", detail)
    if m:
        kind = m.group(1)
        return f"{kind} — high text overlap detected"

    # Similarity paragraph: "Paragraph 3 has 85% semantic overlap with source paragraph 2"
    m = _re.match(r"Paragraph (\d+) has (\d+)% semantic overlap", detail)
    if m:
        return f"Paragraph {m.group(1)} closely matches an external source ({m.group(2)}% overlap)"

    # AI criterion with score: "criterion_name score: 94% (medium). Details: ..."
    m = _re.match(r"(\w+) score: (\d+)% \(\w+\)\. Details:", detail)
    if m:
        name = m.group(1).replace("_", " ").title()
        return f"{name} scored {m.group(2)}%"

    # Fallback: strip raw "Details: key: value" noise
    if ". Details:" in detail:
        return detail.split(". Details:")[0]

    # Citation detail: pass through as-is (already human-readable)
    return detail


def _translate_recommendation(rec: str) -> str:
    """Convert technical recommendation into plain suggestion for reports."""
    if not rec:
        return rec

    # Predictability patterns
    if rec == "Predictable phrasing detected. Consider whether a more specific or original wording would strengthen this point.":
        return "Try rephrasing with more specific or original wording"
    if rec == "Normal prose rhythm for this sentence. Flagged for cluster context only.":
        return "Flagged as part of a cluster of similar patterns nearby"
    if "Consider restructuring with specific evidence, cited claims, or original phrasing." in rec:
        return "Restructure with specific evidence, cited claims, or original phrasing"

    # Generic phrase
    m = _re.match(r"Replace generic phrase '(.+)' with domain-specific or original wording\.", rec)
    if m:
        return f"Replace \"{m.group(1)}\" with more specific wording"

    # Style shift
    if rec.startswith("Writing predictability shifted"):
        return "Check if this tone change was intentional or needs smoothing"

    # AI generation
    if "Review text for AI-generated content." in rec:
        return "Review for AI-generated content — check for concrete details, personal voice, and sources"
    if "Some AI-like patterns detected." in rec:
        return "Some AI-like patterns found — add specific examples or personal observations"
    if rec == "Few AI-like patterns. Minor review recommended.":
        return "Few AI-like patterns — minor review suggested"
    if rec == "Minimal AI-like patterns. Standard review recommended.":
        return "Minimal AI-like patterns — standard review"

    # AI criterion recommendations
    crit_recs = {
        "Rephrase with less common word choices or more specific terminology.": "Rephrase with less common or more specific word choices",
        "Replace predictable phrasing with unexpected or domain-specific terms.": "Replace predictable phrasing with domain-specific terms",
        "Vary sentence structure — mix short and long sentences.": "Vary sentence length — mix short and long sentences",
        "Replace generic transitions with content-specific connectors.": "Replace generic transitions with content-specific ones",
        "Vary sentence openings and syntactic patterns.": "Vary sentence openings and structure",
        "Review section for consistency with surrounding text voice.": "Check this section's tone is consistent with the rest",
    }
    if rec in crit_recs:
        return crit_recs[rec]

    # Specificity
    if "Review domain-term detection results." in rec:
        return "Domain-term detection may be understated — review auto-detected terms"

    # Similarity
    if "Verify the source is properly cited and the wording is original." in rec:
        return "Verify this passage is properly cited and the wording is original"
    if "Review this paragraph for proper attribution and originality." in rec:
        return "Review this paragraph for proper attribution and originality"
    if rec.startswith("Closely paraphrased"):
        return "Closely paraphrased from an external source — rephrase or cite properly"

    return rec


def _count_by_scanner(findings_by_tier: dict, scanner: str) -> int:
    return sum(
        1 for fl in findings_by_tier.values() for f in fl if f.scanner == scanner
    )


def _high_count_by_scanner(findings_by_tier: dict, scanner: str) -> int:
    return sum(
        1 for f in findings_by_tier.get(Tier.HIGH.value, [])
        + findings_by_tier.get(Tier.CRITICAL.value, [])
        if f.scanner == scanner
    )


# ── Main render function (Markdown) ──────────────────────────────────

def render_report(report: DraftReport, verbose: bool = False) -> str:
    """Render a DraftReport as proper Markdown."""
    lines: list[str] = []
    data = report_to_dict(report)
    fb = report.findings_by_tier

    # ── HEADER ────────────────────────────────────────────────────
    tier = report.overall_tier
    emoji = _TIER_EMOJI.get(tier, "[?]")
    badge = _TIER_BADGE.get(tier, tier.value.upper())

    n_critical = len(fb.get(Tier.CRITICAL.value, []))
    n_high = len(fb.get(Tier.HIGH.value, []))
    n_medium = len(fb.get(Tier.MEDIUM.value, []))
    n_low = len(fb.get(Tier.LOW.value, []))
    total = report.finding_count

    lines.append(f"# DraftProof — Integrity Report")
    lines.append("")

    # Header badge: prefer AI Risk badge, else fallback to finding tier
    _shield_colors = {"GREEN": "green", "AMBER": "yellow", "ORANGE": "orange", "RED": "red"}
    if report.ai_risk_badge:
        _ab = report.ai_risk_badge
        _abt = _ab.get("tier", "")
        _abs = _ab.get("ai_likelihood_score", 0)
        _rating = _display_authorship_rating_from_badge(_ab)
        _rating_label = _rating.get("label") or _ab.get("authorship_rating_label")
        _sc = _shield_colors.get(_abt, "lightgrey")
        _abt_label = _BADGE_TIER_LABELS.get(_abt, _abt)
        lines.append(f"![{_abt_label}](https://img.shields.io/badge/Turnitin_AI_Tier-{_abt_label.replace(' ', '_')}-{_sc}) &nbsp; Score `{_abs:.2f}%`")
        if _rating_label:
            lines.append(f"**Authorship Rating:** {_rating_label}")

        # Writing Quality badge beside AI badge
        wq_tier_header = _ab.get("writing_quality_tier", "")
        wq_score_header = _ab.get("writing_quality_score", 0)
        _wq_labels = {"LOW": "Clean", "LIGHT_REVIEW": "Light+Review", "REVIEW": "Review", "HIGH_REVIEW": "Heavy+Review"}
        _wq_colors = {"LOW": "green", "LIGHT_REVIEW": "yellow", "REVIEW": "orange", "HIGH_REVIEW": "red"}
        if wq_tier_header:
            wq_lbl = _wq_labels.get(wq_tier_header, wq_tier_header)
            wq_clr = _wq_colors.get(wq_tier_header, "lightgrey")
            lines.append(f"![{wq_lbl}](https://img.shields.io/badge/Quality-{wq_lbl}-{wq_clr}) &nbsp; Score `{wq_score_header:.2f}%`")
    else:
        lines.append(f"**{badge}** &nbsp; `{tier.value.upper()}`")
    lines.append("")

    # ── SUBMITTED TEXT ────────────────────────────────────────────
    if report.original_text:
        lines.append("## Submitted Text")
        lines.append("")
        paragraphs = report.original_text.strip().split("\n")
        for p in paragraphs:
            p = p.strip()
            if p:
                lines.append(f"> {p}")
                lines.append(">")
        lines.append("")

    # ── 1. EXECUTIVE SUMMARY ──────────────────────────────────────
    lines.append('<div style="page-break-before: always;"></div>')
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")

    signal_chart = _executive_signal_chart_html(
        report,
        data,
        n_critical=n_critical,
        n_high=n_high,
        n_medium=n_medium,
        n_low=n_low,
        total=total,
    )
    if signal_chart:
        lines.append(signal_chart)
    else:
        display_tier = tier.value.upper()
        if report.ai_risk_badge:
            badge_tier_val = report.ai_risk_badge.get("tier", "")
            if badge_tier_val:
                display_tier = _BADGE_TIER_LABELS.get(badge_tier_val, badge_tier_val)

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| **Integrity Tier** | **{display_tier}** |")
        lines.append(f"| **Total Findings** | **{total}** |")
        lines.append(f"| Scan Time | `{report.scan_time_seconds:.1f}s` |")
        if report.generated_at:
            lines.append(f"| Generated | {report.generated_at} |")
        sev_parts = []
        if n_critical: sev_parts.append(f"{n_critical} Critical")
        if n_high: sev_parts.append(f"{n_high} High")
        if n_medium: sev_parts.append(f"{n_medium} Medium")
        if n_low: sev_parts.append(f"{n_low} Low")
        if sev_parts:
            lines.append(f"| **Breakdown** | {' / '.join(sev_parts)} |")
        lines.append("")

    # ── 2. FINDINGS BY SEVERITY ───────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Findings")
    lines.append("")

    finding_num = 0
    has_any = False
    used_scanners = set()
    used_signals = set()
    for tier_level in [Tier.CRITICAL, Tier.HIGH, Tier.MEDIUM, Tier.LOW]:
        findings = fb.get(tier_level.value, [])
        if not findings:
            continue
        has_any = True

        label = tier_level.value.capitalize()
        lines.append(f"### {label} ({len(findings)})")
        lines.append("")

        lines.append("| # | Src | Sig | Finding | Sentence | Suggestion |")
        lines.append("|--:|:---:|:---:|---------|----------|------------|")

        def _cell(text):
            if not text:
                return "—"
            return text.replace("|", "·").replace("\n", " ")

        for f in findings:
            finding_num += 1
            used_scanners.add(f.scanner)
            used_signals.add(f.title)
            scanner = _SCANNER_CODES.get(f.scanner, f.scanner)
            signal = _SIGNAL_CODES.get(f.title, f.title)
            detail = _cell(_translate_detail(f.detail))
            evidence = _cell(f.evidence)
            action = _cell(_translate_recommendation(f.recommendation))
            lines.append(f"| {finding_num} | {scanner} | {signal} | {detail} | {evidence} | {action} |")
        lines.append("")

        # Metadata excluded from reports — available in JSON output

    if not has_any:
        lines.append("*No findings detected. Text appears clean.*")
        lines.append("")

    # ── 2b. FALSE POSITIVES ────────────────────────────────────────
    fp = report.false_positives
    if fp:
        lines.append("---")
        lines.append("")
        lines.append(f"## False Positives Filtered ({len(fp)})")
        lines.append("")

        lines.append("| # | Orig | Adj | Reason | Filter | Sentence |")
        lines.append("|--:|:----:|:---:|--------|:------:|----------|")

        def _fp_cell(text):
            if not text:
                return "-"
            return text.replace("|", "·").replace("\n", " ")

        for i, entry in enumerate(fp, 1):
            orig = _RISK_CODES.get(entry.get("original_risk", "?").lower(),
                                   entry.get("original_risk", "?").upper())
            adj = _RISK_CODES.get(entry.get("adjusted_risk", "?").lower(),
                                  entry.get("adjusted_risk", "?").upper())
            reason = _fp_cell(entry.get("reason", ""))
            filt_full = entry.get("filter", "")
            filt = _FILTER_CODES.get(filt_full, filt_full)
            sent = _fp_cell(entry.get("sentence", ""))
            lines.append(f"| {i} | {orig} | {adj} | {reason} | {filt} | {sent} |")
        lines.append("")

    # ── 3. SCANNER BREAKDOWN (hidden from reports — data in JSON) ───

    # ── 4. REWRITE SUMMARY ────────────────────────────────────────
    if report.rewrite:
        rw = report.rewrite
        lines.append("---")
        lines.append("")
        lines.append("## Rewrite Summary")
        lines.append("")
        lines.append("| Metric | Before | After | Change |")
        lines.append("|--------|--------|-------|--------|")
        lines.append(f"| **Predictability Risk** | `{rw.original_risk:.1%}` | `{rw.final_risk:.1%}` | `{rw.improvement_risk:+.1%}` |")
        if rw.original_findings or rw.rewritten_findings:
            lines.append(f"| **Findings** | {rw.original_findings} | {rw.rewritten_findings} | `{rw.original_findings - rw.rewritten_findings:+d}` |")
        lines.append("")
        lines.append(f"**Passes:** {rw.passes_completed} | **Converged:** {'Yes' if rw.converged else 'No'} ({rw.convergence_reason})")
        lines.append("")

        if rw.pass_progression:
            lines.append("<details>")
            lines.append("<summary>Pass Progression</summary>")
            lines.append("")
            lines.append("| Pass | Risk | Common Ratio |")
            lines.append("|------|------|--------|")
            for i, p in enumerate(rw.pass_progression, 1):
                risk = p.get("risk", p.get("predictability_score", 0))
                top10 = p.get("top10_ratio", 0)
                lines.append(f"| {i} | `{risk:.1%}` | `{top10:.1%}` |")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        if rw.sentence_comparison:
            # Filter to only show sentences that actually changed
            changed = [
                (i, sc) for i, sc in enumerate(rw.sentence_comparison, 1)
                if sc.get("orig_sentence", "") != sc.get("new_sentence", "")
            ]
            if changed:
                lines.append(f"### Detailed Changes ({len(changed)} sentence(s))")
                lines.append("")
                lines.append("| # | Tier Change | Original | Rewritten |")
                lines.append("|---|-------------|----------|-----------|")
                for i, sc in changed:
                    orig_tier = sc.get("orig_tier", "?")
                    new_tier = sc.get("new_tier", "?")
                    orig_text = sc.get("orig_sentence", "").replace("\n", " ").replace("|", "·")
                    new_text = sc.get("new_sentence", "").replace("\n", " ").replace("|", "·")
                    tier_change = f"{orig_tier} → {new_tier}" if orig_tier != new_tier else orig_tier
                    lines.append(f"| {i} | {tier_change} | {orig_text} | {new_text} |")
                lines.append("")

    # ── LEGEND ─────────────────────────────────────────────────────
    if used_scanners or used_signals or fp:
        lines.append("---")
        lines.append("")
        lines.append("## Legend")
        lines.append("")

        scanner_legend = {v: k for k, v in _SCANNER_CODES.items() if k in used_scanners}
        signal_legend = {v: k for k, v in _SIGNAL_CODES.items() if k in used_signals}

        if scanner_legend:
            lines.append("**Source (Src):**")
            for code, full in sorted(scanner_legend.items()):
                lines.append(f"- `{code}` = {full}")
            lines.append("")

        if signal_legend:
            lines.append("**Signal (Sig):**")
            for code, full in sorted(signal_legend.items()):
                lines.append(f"- `{code}` = {full}")
            lines.append("")

        if fp:
            lines.append("**Risk level (Orig / Adj):**")
            for code, full in sorted(_RISK_CODES.items()):
                lines.append(f"- `{full}` = {code}")
            lines.append("")
            used_filters = {entry.get("filter", "") for entry in fp} - {""}
            filter_legend = {v: k for k, v in _FILTER_CODES.items() if k in used_filters}
            if filter_legend:
                lines.append("**Filter:**")
                for code, full in sorted(filter_legend.items()):
                    lines.append(f"- `{code}` = {full}")
                lines.append("")

    # ── FOOTER ────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    ts = f" | {report.generated_at}" if report.generated_at else ""
    lines.append(f"*Report generated by DraftProof{ts} | {report.scan_time_seconds:.1f}s scan time*")
    lines.append("")
    lines.append("> **Note:** This is a pre-submission integrity check, not a plagiarism or AI-authorship verdict. Signals should be reviewed in context.")
    lines.append("")

    return "\n".join(lines)


def _scanner_score(report: DraftReport, scanner: str) -> str:
    """Extract a score string for a scanner from the report summaries."""
    if scanner == "predictability" and report.predictability:
        return f"{report.predictability.overall_risk:.1%}"
    if scanner == "similarity" and report.similarity:
        return f"`{report.similarity.overall_risk}`"
    if scanner == "citation" and report.citation:
        return f"{report.citation.in_text_count} cites"
    if scanner == "ai_generation":
        return "see signals"
    return "—"


def _verdict(tier: Tier, total: int) -> str:
    if tier == Tier.CLEAN:
        return "No significant concerns detected. Text appears ready for submission."
    if tier == Tier.LOW:
        return "Minor signals detected. Low risk — review optional."
    if tier == Tier.MEDIUM:
        return "Moderate concerns found. Consider reviewing medium-severity findings before submission."
    if tier == Tier.HIGH:
        return "Significant concerns found. Review findings before submission."
    return "Critical issues detected. Revision strongly recommended before submission."


def render_markdown(report: DraftReport, verbose: bool = False) -> str:
    """Alias for render_report — outputs Markdown."""
    return render_report(report, verbose=verbose)


def print_report(report: DraftReport, verbose: bool = False, file=None):
    """Print a DraftReport to stdout or a file-like object."""
    text = render_report(report, verbose=verbose)
    if file is not None:
        file.write(text)
    else:
        print(text)


def print_markdown(report: DraftReport, verbose: bool = False, file=None):
    """Print a DraftReport as Markdown to stdout or a file-like object."""
    text = render_markdown(report, verbose=verbose)
    if file is not None:
        file.write(text)
    else:
        print(text)
