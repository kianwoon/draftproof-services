"""DraftProof Report Renderer — Markdown output from DraftReport.

Produces a structured report with:
  1. EXECUTIVE SUMMARY — overall tier badge, risk gauge, scanner scores
  2. FINDINGS BY SEVERITY — grouped by tier with collapsible details
  3. SCANNER BREAKDOWN — per-scanner tables with paragraph-level detail
  4. REWRITE SUMMARY — before/after comparison (when present)
  5. APPENDIX — full sentence tables in collapsible sections
"""

from html import escape
from typing import Optional

from detect.transformation import TRANSFORMATION_SIGNAL_METADATA, transformation_signal_metadata
from detect.turnitin_like import turnitin_like_ai_profile

from .report import DraftReport, Tier, TIER_ICON, report_to_dict, determine_actionability
from .paragraph_explainer import explanations_by_paragraph

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

# ── Layman labels for AI Likelihood components ──────────────────────────
_AI_COMPONENT_LABELS = {
    "predictability": ("Predictability", "How predictable the word choices are — higher means the text reads like statistically common patterns"),
    "topk_calibrated_risk": ("Calibrated Top-k Risk", "Calibrated risk from raw token-route concentration. Lower is safer and less likely to look machine-routed."),
    "topk_pattern": ("Raw Top-k Predictability", "Raw token-route concentration. Diagnostic only; calibrated Top-k risk controls the safe-band gate."),
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


_TURNITIN_AI_REFERENCE_THRESHOLD = 20
_TURNITIN_AI_REFERENCE_NOTE = (
    "Turnitin reference: AI scores below 20% may appear as *% instead of an exact percentage "
    "because low-range results are less reliable."
)


def _display_ai_score(value) -> float | None:
    # Canonical DraftProof AI-likelihood percent (badge ai_likelihood_score), shown directly.
    # Previously halved by a 0.5 "display multiplier"; removed so page, PDF, and email agree.
    return _tf_pct(value)


def _ai_reference_suffix(score) -> str | None:
    value = _tf_pct(score)
    if value is None:
        return None
    if value < _TURNITIN_AI_REFERENCE_THRESHOLD:
        return "below 20% reference"
    return "review threshold exceeded"


def _with_ai_reference(text: str, score) -> str:
    suffix = _ai_reference_suffix(score)
    return f"{text} · {suffix}" if suffix else text


def _report_transformation_feature_fallbacks(data: dict | None) -> dict:
    if not isinstance(data, dict):
        return {}
    intelligence = data.get("scan_intelligence") or {}
    transformation = intelligence.get("transformation") if isinstance(intelligence, dict) else {}
    features: dict = {}
    if isinstance(transformation, dict):
        for row in transformation.get("core_signals") or []:
            if not isinstance(row, dict) or not row.get("key"):
                continue
            score = _tf_pct(row.get("score"))
            if score is not None:
                features[str(row["key"])] = score
        contribution = transformation.get("contribution") or {}
        if isinstance(contribution, dict):
            for source_key, target_key in [
                ("calibrated_ai_risk", "calibrated_ai_risk"),
                ("adjusted_ai_risk", "adjusted_ai_risk"),
                ("human_anchor_discount", "human_anchor_discount"),
                ("calibration_confidence", "calibration_confidence"),
                ("reporting_suppression", "reporting_suppression"),
            ]:
                score = _tf_pct(contribution.get(source_key))
                if score is not None:
                    features.setdefault(target_key, score)
    return features


def _badge_with_report_calibration(badge: dict, data: dict | None = None) -> dict:
    if not isinstance(badge, dict):
        return badge
    fallback_features = _report_transformation_feature_fallbacks(data)
    if not fallback_features:
        return badge
    transformed = dict(badge.get("transformation_classification") or {})
    features = {
        **fallback_features,
        **(transformed.get("features") or {}),
    }
    transformed["features"] = features
    enriched = dict(badge)
    enriched["transformation_classification"] = transformed
    return enriched


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


def _transformation_contribution_summary(features: dict, signals: list[dict], badge: dict | None = None) -> dict:
    ai_likelihood = (
        _tf_pct((features or {}).get("calibrated_ai_risk"))
        if (features or {}).get("calibrated_ai_risk") is not None
        else (
            _tf_pct((features or {}).get("adjusted_ai_risk"))
            if (features or {}).get("adjusted_ai_risk") is not None
            else _tf_pct((features or {}).get("ai_likelihood"))
        )
    ) or 0.0
    turnitin_profile = turnitin_like_ai_profile(
        features=features or {},
        ai_components=(badge or {}).get("ai_components") or {},
    )
    atr = round(float(turnitin_profile.get("score") or 0.0))
    atr = max(0, min(100, atr))
    hcr = 100 - atr

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
        "turnitin_like_ai_score": round(float(turnitin_profile.get("score") or 0.0), 3),
        "turnitin_like_target_score": turnitin_profile.get("target_score"),
        "turnitin_like_target_gap": turnitin_profile.get("target_gap"),
        "turnitin_like_target_met": turnitin_profile.get("target_met"),
        "turnitin_like_components": turnitin_profile.get("components") or {},
        "turnitin_like_weighted_components": turnitin_profile.get("weighted_components") or {},
        "turnitin_like_component_contributions": turnitin_profile.get("component_contributions") or {},
        "turnitin_like_top_positive_drivers": turnitin_profile.get("top_positive_drivers") or [],
        "turnitin_like_human_anchor_suppression": turnitin_profile.get("human_anchor_suppression"),
        "summary": summary,
    }


_DRAFTPROOF_TIER_COLORS = {
    "GREEN": "#16a34a", "AMBER": "#d97706", "ORANGE": "#ea580c", "RED": "#dc2626",
}
_EXTERNAL_BAND_LABELS = {
    "low": ("unlikely to be flagged", "#16a34a"),
    "elevated": ("possibly flagged", "#d97706"),
    "high": ("likely to be flagged", "#dc2626"),
}


def _ai_likelihood_bands(badge: dict | None) -> dict:
    """Shared band->label/color mapping for the dual headline. Mirrors the JS
    ``aiLikelihoodBands`` in reportHelpers.js (kept in sync via the spec table)."""
    badge = badge or {}
    score = badge.get("ai_likelihood_score")
    tier = str(badge.get("tier") or "").upper()
    draftproof = None
    if isinstance(score, (int, float)):
        draftproof = {
            "score": round(score),
            "tier": tier or "AMBER",
            "color": _DRAFTPROOF_TIER_COLORS.get(tier, "#d97706"),
        }
    ext = badge.get("external_detector_estimate") or {}
    ext_score = ext.get("score")
    external = None
    if isinstance(ext_score, (int, float)):
        band = str(ext.get("band") or "").lower()
        label, color = _EXTERNAL_BAND_LABELS.get(band, ("estimated", "#475569"))
        external = {"score": round(ext_score), "band": band, "label": label, "color": color}
    return {"draftproof": draftproof, "external": external}


_AI_LIKELIHOOD_WHY = (
    "This is a calibrated estimate of how strict third-party detectors (Turnitin, GPTZero) may rate "
    "your text -- based on limited data and imperfect, since these detectors over-flag fluent "
    "writing (even genuine human writing). Treat it as a heads-up, not a verdict; your safeguard is "
    "grounding the content and finishing it in your own words."
)

# The badge's external_detector_estimate is the grouped external-detector proxy
# (detect.external_grouped_scoring.external_grouped_v1). Legacy segment/perplexity estimates may
# ride along under `alternates` for auditability, but they no longer decide the visible number.
# Surfaced as an ESTIMATE, not a vendor result. Set False to suppress the number again.
EXTERNAL_ESTIMATE_DISPLAY_ENABLED = True

_EXTERNAL_DEMOTED_NOTE = (
    "Third-party AI detectors (Turnitin, GPTZero) are imperfect and over-flag fluent writing -- "
    "including genuine human writing -- so a predicted score is not a reliable signal of your "
    "result, and we don't surface one here. Your safeguard is grounding the content, finishing it "
    "in your own words, and keeping your drafts as authorship evidence."
)


# Grounding-diagnosis lead: when ON, lead the AI Likelihood block with the primary
# DRIVER (what to fix) and demote the raw % below. The diagnosis is always computed
# in report.py; this flag only controls presentation. KEEP label strings in sync with
# detect.grounding_diagnosis.DRIVER_LABELS and the frontend i18n keys
# report.groundingDiagnosis.* (draftproof-frontend/src/i18n/en/report.js + zh/report.js).
GROUNDING_DIAGNOSIS_LEAD_ENABLED = True

# allow-hardcode: presentation labels for the 4 grounding-diagnosis buckets keyed
# by bucket CODE — display strings for the PDF, never matched against document text.
# KEEP IN SYNC with frontend i18n report.groundingDiagnosis.buckets.*.
_DIAG_BUCKET_LABELS = {
    "concrete_grounding": "Grounding gap",
    "authorship_trace": "Authorship uncertainty",
    "llm_patterning": "AI-like patterning",
    "language_texture": "Generic language texture",
}


def _grounding_diagnosis_lead(badge: dict) -> list[str]:
    """Lead lines for AI Likelihood: primary driver + 4-bucket breakdown.

    Returns [] when the diagnosis is absent or has no actionable driver (so the
    caller falls back to the plain DraftProof % headline).
    """
    diag = (badge or {}).get("grounding_diagnosis") or {}
    driver = diag.get("primary_driver")
    if not driver:
        return []
    label = diag.get("primary_driver_label") or ""
    action = diag.get("primary_driver_action") or ""
    lead = f"- **Main thing to fix: {label}"
    if action:
        lead += f" — {action}"
    lead += "**"
    if diag.get("caveat"):
        lead += f" _({diag['caveat']})_"
    buckets = diag.get("buckets") or {}
    parts = [
        f"{_DIAG_BUCKET_LABELS.get(k, k)} {round(buckets[k]['score'])}"
        for k in _DIAG_BUCKET_LABELS if k in buckets
    ]
    lines = [lead]
    if parts:
        # "Risk contributors (lower is better)" — make bar direction explicit.
        lines.append("- _Risk contributors (lower is better):_ " + " · ".join(parts))
    return lines


# allow-hardcode: presentation phrases for the scoped AI-writing-signal verdict,
# keyed by badge tier / external band CODE. Display strings, never matched against
# document text. KEEP IN SYNC with frontend i18n report.aiLikelihood.verdict*.
_SIGNAL_PHRASE = {"GREEN": "Low AI-writing signal", "AMBER": "Moderate AI-writing signal",
                  "ORANGE": "Elevated AI-writing signal", "RED": "Elevated AI-writing signal"}
_FLAG_PHRASE = {"low": "unlikely to be flagged by external detectors",
                "elevated": "may draw external-detector attention",
                "high": "likely to be flagged by external detectors"}


def _ai_signal_verdict(badge: dict) -> str:
    """One-line scoped read of the external-detector / AI-writing dimension. NOT an
    overall verdict (the Submission-risk band leads that)."""
    bands = _ai_likelihood_bands(badge)
    dp = bands.get("draftproof") or {}
    signal = _SIGNAL_PHRASE.get(str(dp.get("tier") or "").upper(), "AI-writing signal")
    ext = bands.get("external") or {}
    flag = _FLAG_PHRASE.get(str(ext.get("band") or "").lower(), "") if EXTERNAL_ESTIMATE_DISPLAY_ENABLED else ""
    diag = (badge or {}).get("grounding_diagnosis") or {}
    driver = diag.get("primary_driver_label") or ""
    verdict = signal
    if flag:
        verdict += f" — {flag}"
    verdict += "."
    if driver:
        verdict += f" The main writing issue to fix is the {driver}."
    return verdict


def _render_ai_likelihood_headline(badge: dict | None) -> str:
    bands = _ai_likelihood_bands(badge)
    dp = bands["draftproof"]
    if not dp:
        return ""
    badge = badge or {}
    out = ["## AI-writing signal", ""]
    # Scoped verdict leads (not a competing overall verdict).
    out.append(f"**{_ai_signal_verdict(badge)}**")
    out.append("")
    lead = _grounding_diagnosis_lead(badge) if GROUNDING_DIAGNOSIS_LEAD_ENABLED else []
    if lead:
        out.extend(lead)
        out.append("")
        out.append(f"- _Detail — DraftProof grounding signal: {dp['score']}% ({dp['tier']})_")
    else:
        out.append(f"- **DraftProof grounding signal: {dp['score']}% — {dp['tier']}** — improves as you ground the content (the signal to act on)")
    ext = bands["external"]
    if EXTERNAL_ESTIMATE_DISPLAY_ENABLED and ext:
        out.append(f"- **Turnitin / external (estimated): ~{ext['score']}% — {ext['label']}**")
    out.append("")
    tc = badge.get("transformation_classification") or {}
    meta = []
    if tc.get("label"):
        conf = tc.get("confidence")
        meta.append(f"Pattern: {tc['label']}" + (f" ({conf} confidence)" if conf else ""))
    if badge.get("authorship_rating_label"):
        meta.append(str(badge["authorship_rating_label"]))
    if meta:
        out.append(" · ".join(meta))
        out.append("")
    out.append(_AI_LIKELIHOOD_WHY if EXTERNAL_ESTIMATE_DISPLAY_ENABLED else _EXTERNAL_DEMOTED_NOTE)
    out.append("")
    return "\n".join(out)


_SUBMISSION_RISK_WHY = (
    "Submission risk is about whether you can stand behind this as your own work -- not "
    "whether it looks AI-written. Universities now allow AI use when it's declared, accurate, "
    "and cited; the offence is work you can't account for. The text-pattern axis is the external "
    "detector trigger (a heads-up, not a verdict). Declaration, course policy, group contribution, "
    "and oral-defence readiness aren't in the text -- only you can declare those."
)

# allow-hardcode: presentation level labels (machine level -> display), not a
# detect/scoring word-list. Mirrors frontend i18n report.submissionRisk.levels.*.
_SR_LEVEL_LABELS = {"low": "Low", "medium": "Medium", "high": "High", "unknown": "Unknown"}
_SR_AXIS_ORDER = ["text_pattern", "ownership", "citation", "defence_readiness", "policy_declaration"]


def _render_submission_risk_headline(badge: dict | None) -> str:
    """Lead the report with the 3-layer Submission-risk view. Returns "" when the
    diagnosis abstained (unknown), so the AI Likelihood section still leads as before."""
    sr = (badge or {}).get("submission_risk") or {}
    overall = sr.get("overall") or {}
    level = overall.get("level")
    if not level or level == "unknown":
        return ""
    axes = sr.get("axes") or {}
    out = ["## Submission Risk", ""]
    headline = overall.get("label") or f"{_SR_LEVEL_LABELS.get(level, level)} submission risk"
    out.append(f"- **{headline}**")
    if overall.get("main_reason"):
        out.append(f"  - Main reason: {overall['main_reason']}")
    out.append("")
    for key in _SR_AXIS_ORDER:
        ax = axes.get(key) or {}
        label = ax.get("label") or key
        if key == "policy_declaration":
            out.append(f"- {label}: Unknown — self-declare")
            continue
        row = f"- {label}: {_SR_LEVEL_LABELS.get(ax.get('level'), 'Unknown')}"
        if key == "text_pattern" and isinstance(ax.get("display_score"), (int, float)):
            row += f" (AI-likelihood ~{round(ax['display_score'])}% — external trigger, not a verdict)"
        out.append(row)
    out.append("")
    out.append(_SUBMISSION_RISK_WHY)
    out.append("")
    return "\n".join(out)


def _render_critical_thinking_questions(badge: dict | None) -> str:
    """Render the Critical Thinking reflective questions section (PDF parity with the
    web report). Empty string when there are no questions, so the section is omitted."""
    ctc = (badge or {}).get("critical_thinking_control") or {}
    questions = ctc.get("questions") or []
    rows = [q for q in questions if isinstance(q, dict) and str(q.get("question") or "").strip()]
    if not rows:
        return ""
    # allow-hardcode: section heading + intro are presentation copy (mirrors the
    # frontend i18n report.criticalThinking.*), not a detect/scoring word-list.
    out = ["## Questions to Sharpen Your Thinking", ""]
    out.append(
        "These are about your own draft. Use them to push your thinking further as you "
        "revise — the answers are yours to write."
    )
    out.append("")
    for i, q in enumerate(rows, 1):
        quote = str(q.get("anchor_quote") or "").strip()
        question = str(q.get("question") or "").strip()
        if quote:
            out.append(f"**{i}.** _“{quote}”_")
            out.append("")
            out.append(question)
        else:
            out.append(f"**{i}.** {question}")
        out.append("")
    return "\n".join(out)


def _signal_chart_rows(features: dict, badge: dict | None = None) -> list[dict]:
    rows_by_key = {row["key"]: row for row in _transformation_signals(features)}
    ai_components = (badge or {}).get("ai_components") or {}
    topk_calibrated_score = _tf_pct(ai_components.get("topk_calibrated_risk"))
    if topk_calibrated_score is not None and "topk_calibrated_risk" not in rows_by_key:
        label, description = _AI_COMPONENT_LABELS["topk_calibrated_risk"]
        rows_by_key["topk_calibrated_risk"] = {
            "key": "topk_calibrated_risk",
            "label": label,
            "description": description,
            "score": topk_calibrated_score,
        }
    topk_score = _tf_pct(ai_components.get("topk_pattern_raw", ai_components.get("topk_pattern")))
    if topk_score is not None and "topk_pattern" not in rows_by_key:
        label, description = _AI_COMPONENT_LABELS["topk_pattern"]
        rows_by_key["topk_pattern"] = {
            "key": "topk_pattern",
            "label": label,
            "description": description,
            "score": topk_score,
        }
    return sorted(
        rows_by_key.values(),
        key=lambda row: (-(row.get("score") or 0), _SIGNAL_CHART_ORDER.index(row["key"]) if row.get("key") in _SIGNAL_CHART_ORDER else len(_SIGNAL_CHART_ORDER), row.get("label") or ""),
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


def _count_value(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, number)


def _authorship_sample_limit(sample_context: dict | None = None, topk_calibration_eligible=None) -> dict | None:
    word_count = _count_value((sample_context or {}).get("word_count"))
    sentence_count = _count_value((sample_context or {}).get("sentence_count"))
    if word_count is None and sentence_count is None and topk_calibration_eligible is not False:
        return None

    very_short = (
        topk_calibration_eligible is False
        or (word_count is not None and word_count < 30)
        or (sentence_count is not None and sentence_count < 3)
    )
    limited = (
        very_short
        or (word_count is not None and word_count < 150)
        or (sentence_count is not None and sentence_count < 6)
    )
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "very_short": very_short,
        "limited": limited,
    }


def _strongest_supporting_ai_shape_signal(features: dict | None) -> dict | None:
    for key, label in [
        ("ai_likelihood", "AI likelihood"),
        ("semantic_uniformity_risk", "Semantic uniformity"),
        ("section_style_variance", "Patchwork variance"),
        ("rewrite_smoothness", "Rewrite smoothness"),
        ("outline_to_text_expansion", "Expansion pattern"),
        ("discourse_regularity_risk", "Discourse regularity"),
    ]:
        signal_score = _tf_pct((features or {}).get(key))
        if signal_score is not None and signal_score >= 50:
            return {"key": key, "label": label, "score": signal_score}
    return None


def _authorship_rating_from_calibrated_risk(
    score,
    topk_score=None,
    topk_calibrated_risk=None,
    features: dict | None = None,
    sample_context: dict | None = None,
    topk_calibration_eligible=None,
) -> dict:
    calibrated_score = _tf_pct(score)
    topk_score = _tf_pct(topk_score)
    topk_calibrated_score = _tf_pct(topk_calibrated_risk)
    supporting_signal = _strongest_supporting_ai_shape_signal(features)
    ai_likelihood_score = _tf_pct((features or {}).get("ai_likelihood"))
    human_anchor_score = _tf_pct((features or {}).get("human_anchor_score"))
    semantic_uniformity_score = _tf_pct((features or {}).get("semantic_uniformity_risk"))
    sample_limit = _authorship_sample_limit(sample_context, topk_calibration_eligible)
    if sample_limit and sample_limit["very_short"]:
        return {
            "label": "Too Short to Assess",
            "short_label": "Too Short",
            "code": "insufficient_sample",
            "level": -1,
            "sample_limited": True,
            "sample_context": sample_limit,
            "score": calibrated_score,
            "topk_score": topk_score,
            "topk_calibrated_risk": topk_calibrated_score,
        }
    rating = _rating_for_calibrated_score(calibrated_score) if calibrated_score is not None else {}

    turnitin_zero_like_human_profile = (
        calibrated_score is not None
        and calibrated_score <= 14
        and human_anchor_score is not None
        and human_anchor_score >= 75
        and (ai_likelihood_score is None or ai_likelihood_score <= 35)
        and (topk_calibrated_score is None or topk_calibrated_score <= 55)
        and (semantic_uniformity_score is None or semantic_uniformity_score <= 35)
    )
    if turnitin_zero_like_human_profile:
        rating = {
            **next(item for item in _CALIBRATED_AUTHORSHIP_LEVELS if item["code"] == "low_ai_signal"),
            "turnitin_zero_like_human_profile": True,
        }

    def _apply_topk_floor(floor_code: str, **extra) -> None:
        nonlocal rating
        if turnitin_zero_like_human_profile:
            return
        floor = next((dict(item) for item in _CALIBRATED_AUTHORSHIP_LEVELS if item["code"] == floor_code), {})
        if floor and (not rating or rating.get("level", -1) < floor["level"]):
            rating = {
                **floor,
                "topk_escalated": True,
                "topk_score": topk_score,
                "topk_calibrated_risk": topk_calibrated_score,
                "supporting_signal": supporting_signal,
                **extra,
            }

    strong_topk_whole_profile = (
        topk_score is not None
        and topk_calibrated_score is not None
        and topk_score >= 90
        and topk_calibrated_score >= 90
        and supporting_signal
        and calibrated_score is not None
        and calibrated_score >= 35
        and ai_likelihood_score is not None
        and ai_likelihood_score >= 55
        and (human_anchor_score is None or human_anchor_score <= 50)
    )
    if strong_topk_whole_profile:
        _apply_topk_floor("ai_generated_signals", topk_strong_signal=True)

    if topk_calibrated_score is not None:
        if topk_calibrated_score >= 80:
            _apply_topk_floor("likely_ai")
        elif topk_calibrated_score >= 70:
            _apply_topk_floor("possible_ai_assisted")

    moderate_ai_texture = (
        calibrated_score is not None
        and ai_likelihood_score is not None
        and topk_score is not None
        and topk_calibrated_score is not None
        and calibrated_score >= 15
        and 32 <= ai_likelihood_score < 45
        and topk_score >= 70
        and topk_calibrated_score >= 40
    )
    if moderate_ai_texture:
        _apply_topk_floor("possible_ai_assisted", moderate_ai_texture=True)

    if not rating:
        return {}
    if sample_limit and sample_limit["limited"] and rating.get("level", 0) > 2:
        rating = {
            **next(item for item in _CALIBRATED_AUTHORSHIP_LEVELS if item["code"] == "possible_ai_assisted"),
            "sample_limited": True,
            "sample_context": sample_limit,
            "original_rating": rating,
        }
    rating["score"] = calibrated_score
    rating["topk_score"] = rating.get("topk_score", topk_score)
    rating["topk_calibrated_risk"] = rating.get("topk_calibrated_risk", topk_calibrated_score)
    rating["supporting_signal"] = rating.get("supporting_signal", supporting_signal)
    return rating


def _authorship_rating_tone(rating: dict) -> dict:
    code = str((rating or {}).get("code") or (rating or {}).get("short_label") or (rating or {}).get("label") or "").lower()
    if "insufficient" in code or "too short" in code:
        return {"color": "#475569", "bg": "#f8fafc"}
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


def _ai_signal_stamp(score) -> dict:
    value = _tf_pct(score)
    if value is None:
        return {"label": "AI Review", "color": "#334155", "bg": "#f8fafc"}
    if value >= 60:
        return {"label": "High AI Signal", "color": "#b91c1c", "bg": "#fef2f2"}
    if value >= 40:
        return {"label": "Likely AI", "color": "#c2410c", "bg": "#fff7ed"}
    if value >= _TURNITIN_AI_REFERENCE_THRESHOLD:
        return {"label": "AI Review", "color": "#b45309", "bg": "#fff7ed"}
    return {"label": "Low AI Signal", "color": "#15803d", "bg": "#f0fdf4"}


def _display_authorship_rating_from_badge(
    badge: dict,
    sample_context: dict | None = None,
    data: dict | None = None,
) -> dict:
    badge = _badge_with_report_calibration(badge, data)
    features = (((badge or {}).get("transformation_classification") or {}).get("features") or {})
    ai_components = (badge or {}).get("ai_components") or {}
    topk_score = ai_components.get("topk_pattern_raw", ai_components.get("topk_pattern"))
    calibrated = _authorship_rating_from_calibrated_risk(
        features.get("calibrated_ai_risk"),
        topk_score,
        ai_components.get("topk_calibrated_risk"),
        features,
        sample_context,
        ai_components.get("topk_calibration_eligible"),
    )
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

    # Authorship rating (label/detail/tone) is no longer rendered here — the seal was removed to
    # mirror the web page mesh. The rating is shown once in the report header ("Authorship Rating:").
    doc_ctx = data.get("document_context", {}) if isinstance(data, dict) else {}
    contribution = _transformation_contribution_summary(features, rows, badge)

    confidence = transformation.get("confidence")
    pills = []
    if confidence:
        pills.append(f"{str(confidence).title()} Confidence")
    # "Not A Verdict" disclaimer is stated once in the repair plan / AI-likelihood note now,
    # mirroring the web page mesh — no longer repeated here.

    adjustment_chips = [
        f"Calibrated AI risk {contribution['calibrated_ai_risk']}%",
        f"Human anchor discount {contribution['human_anchor_discount']}%",
        f"Calibration confidence {contribution['calibration_confidence']}%",
        f"Reporting suppression {contribution['reporting_suppression']}%",
    ]

    evidence = transformation.get("evidence") or []
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
        # Rating seal removed to match the web page mesh: the authorship rating is shown once in
        # the header ("Authorship Rating:"), and its calibrated-risk detail no longer competes
        # with the AI Likelihood headline. Keeps page⇄PDF parity.
        '</header>',
        # "Original Scan" + pattern label removed: it just repeats the card header above (the AI
        # score is already in the AI Likelihood headline). Mirrors the web page mesh.
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
    ]
    if evidence:
        html.extend([
            '<div class="dp-evidence-row">',
            ''.join(f'<span>{escape(str(item))}</span>' for item in evidence[:3]),
            '</div>',
        ])
    if confidence_note:
        html.append(f'<p class="dp-confidence-note">{escape(confidence_note)}</p>')
    html.append(f'<p class="dp-ai-reference-note">{escape(_TURNITIN_AI_REFERENCE_NOTE)}</p>')
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

_SIGNAL_CATEGORY_LABELS = {
    "authorship_risk": "Authorship Risk",
    "ai_likelihood": "AI Likelihood",
    "predictability": "AI Likelihood",
    "genericity": "Generic Phrasing",
    "writing_quality": "Writing Quality",
    "similarity": "Similarity",
    "citation": "Citation",
    "structure": "Structure",
    "rewrite": "Rewrite",
}

_SIGNAL_CATEGORY_PRIORITY = [
    "authorship_risk", "ai_likelihood", "predictability",
    "genericity", "similarity", "citation", "writing_quality", "structure",
]

_ACTIONABILITY_LABELS = {
    "review_only": "Review Only",
    "auto_fixable": "Auto-Fixable",
    "manual_required": "Manual Required",
    "citation_repair": "Citation Repair",
    "optional_structure_review": "Optional Review",
    "no_action": "No Action",
}

_ACTIONABILITY_SORT = [
    "manual_required", "auto_fixable", "citation_repair",
    "optional_structure_review", "review_only", "no_action",
]

_TIER_SCORE_DEFAULTS = {"critical": 90, "high": 72, "medium": 52, "low": 28}


def _sig_cat_label(cat: str) -> str:
    return _SIGNAL_CATEGORY_LABELS.get(cat, cat.replace("_", " ").title())


def _group_signal_strength(group_findings: list) -> int:
    """Best available score as integer percent for a paragraph group."""
    scores = []
    for f in group_findings:
        meta = f.metadata or {}
        for key in ("score", "predictability_risk", "ai_likelihood"):
            v = meta.get(key)
            if isinstance(v, (int, float)) and 0 < v <= 1:
                scores.append(float(v))
                break
    if scores:
        return int(max(scores) * 100)
    return _TIER_SCORE_DEFAULTS.get(group_findings[0].tier.value if group_findings else "medium", 50)


def _render_finding_card(finding_num: int, tier_level: Tier, group: dict) -> str:
    """Return an HTML string for one paragraph finding card."""
    group_findings = group["findings"]
    sentence_ids = group.get("sentence_ids") or []

    # Section identifier
    if len(sentence_ids) > 1:
        sid_label = f"{sentence_ids[0]}&ndash;{sentence_ids[-1]}"
    elif sentence_ids:
        sid_label = sentence_ids[0]
    else:
        sid_label = group.get("paragraph_id") or f"#{finding_num}"

    # Primary signal category (highest priority among findings)
    all_cats = list(dict.fromkeys(
        f.signal_category or f.category or "predictability"
        for f in group_findings
    ))
    sorted_cats = sorted(
        all_cats,
        key=lambda c: _SIGNAL_CATEGORY_PRIORITY.index(c) if c in _SIGNAL_CATEGORY_PRIORITY else 99,
    )
    primary_cat = sorted_cats[0] if sorted_cats else "predictability"
    other_cats = sorted_cats[1:]

    # Signal strength
    signal_pct = _group_signal_strength(group_findings)

    # Tier / actionability chips
    tier_label = tier_level.value.upper()
    all_actions = [determine_actionability(f, group_findings) for f in group_findings]
    primary_action = min(
        all_actions,
        key=lambda a: _ACTIONABILITY_SORT.index(a) if a in _ACTIONABILITY_SORT else 99,
    )
    action_label = _ACTIONABILITY_LABELS.get(primary_action, primary_action.replace("_", " ").title())

    # Explanation fields
    expl = group.get("explanation") or {}
    if isinstance(expl, dict):
        summary = expl.get("reader_summary") or expl.get("summary") or ""
        main_issue = expl.get("main_issue") or ""
        why_flagged = expl.get("why_flagged") or []
        recommendation = expl.get("recommendation") or ""
        rewrite_hint = expl.get("rewrite_hint") or ""
    else:
        summary = main_issue = recommendation = rewrite_hint = ""
        why_flagged = []

    # Fallbacks when explainer hasn't run
    if not summary:
        summary = "; ".join(
            _translate_detail(f.detail) for f in group_findings if f.detail
        )[:300]
    if not recommendation:
        recommendation = "; ".join(
            _translate_recommendation(f.recommendation)
            for f in group_findings if f.recommendation
        )[:300]

    def _e(text: str) -> str:
        from html import escape
        return escape(str(text or ""))

    # Build chip HTML
    chips_html = (
        f'<span class="dp-tag-chip">{len(group_findings)} Finding{"s" if len(group_findings) != 1 else ""} In Paragraph</span>'
        f'<span class="dp-tag-chip">{_e(tier_label)} Priority</span>'
        f'<span class="dp-tag-chip">{_e(action_label)}</span>'
    )

    also_html = ""
    if other_cats:
        also_labels = " ".join(
            f'<span class="dp-also-chip">{_e(_sig_cat_label(c))}</span>'
            for c in other_cats
        )
        also_html = f"""
    <div class="dp-also-row">
      <span class="dp-also-label">ALSO DETECTED</span>
      {also_labels}
    </div>"""

    def _subsection(label: str, content: str) -> str:
        if not content:
            return ""
        return f"""
    <div class="dp-finding-subsection">
      <div class="dp-finding-subsection-label">{label}</div>
      <p>{_e(content)}</p>
    </div>"""

    # "What the reader may notice" as bullet list
    bullets_html = ""
    if isinstance(why_flagged, list) and why_flagged:
        items = "".join(f"<li>{_e(item)}</li>" for item in why_flagged if item)
        bullets_html = f"""
    <div class="dp-finding-subsection">
      <div class="dp-finding-subsection-label">WHAT THE READER MAY NOTICE</div>
      <ul class="dp-finding-bullets">{items}</ul>
    </div>"""

    return f"""<div class="dp-signal-card dp-finding-card">
  <div class="dp-finding-card-header">
    <div>
      <span class="dp-finding-section-id">{_e(sid_label)}</span>
      <div class="dp-finding-type">{_e(_sig_cat_label(primary_cat))}</div>
    </div>
    <div class="dp-finding-count">#{finding_num}</div>
  </div>
  <div class="dp-finding-body">
    {(f'<blockquote class="dp-finding-paragraph">{_e(_truncate(group.get("text") or "", 400))}</blockquote>') if group.get("text") else ""}
    {(f'<p class="dp-finding-description">{_e(summary)}</p>') if summary else ""}
    <div class="dp-finding-strength-row">
      <span class="dp-finding-strength-label">SIGNAL STRENGTH</span>
      <span class="dp-finding-strength-pct">{signal_pct}%</span>
    </div>
    <div class="dp-signal-strength-bar">
      <div class="dp-signal-strength-fill" style="width:{signal_pct}%"></div>
    </div>
    <div class="dp-tag-row">{chips_html}</div>
    {also_html}
    {_subsection("MAIN ISSUE TO FIX", main_issue)}
    {bullets_html}
    {_subsection("HOW TO IMPROVE THIS PARAGRAPH", recommendation)}
    {_subsection("REWRITE HINT", rewrite_hint)}
  </div>
</div>"""


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

def _table_cell(text) -> str:
    if not text:
        return "—"
    return str(text).replace("|", "·").replace("\n", " ")

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


def _paragraph_finding_groups(findings: list, data: dict) -> list[dict]:
    """Group report findings by paragraph while preserving finding metadata."""
    explanation_map = explanations_by_paragraph(data.get("paragraph_explanations"))
    segments = ((data.get("scan_intelligence") or {}).get("document") or {}).get("segments") or []
    paragraphs = ((data.get("scan_intelligence") or {}).get("document") or {}).get("paragraphs") or []
    segment_by_sentence = {
        str(segment.get("sentence_id") or ""): segment
        for segment in segments
        if segment.get("sentence_id")
    }
    paragraph_by_id = {
        str(paragraph.get("paragraph_id") or ""): paragraph
        for paragraph in paragraphs
        if paragraph.get("paragraph_id")
    }

    groups = {}
    ordered_keys = []
    for finding in findings:
        segment = segment_by_sentence.get(str(finding.sentence_id or ""))
        paragraph_id = str((segment or {}).get("paragraph_id") or "") if segment else ""
        key = paragraph_id or f"document::{finding.finding_id or len(ordered_keys) + 1}"
        if key not in groups:
            paragraph = paragraph_by_id.get(paragraph_id, {})
            group = {
                "paragraph_id": paragraph_id,
                "sentence_ids": list(paragraph.get("sentence_ids") or []),
                "text": paragraph.get("text") or (segment or {}).get("text") or finding.evidence,
                "findings": [],
                "explanation": explanation_map.get(paragraph_id),
            }
            groups[key] = group
            ordered_keys.append(key)
        groups[key]["findings"].append(finding)

    def group_start(group: dict) -> int:
        if group.get("paragraph_id") and group.get("paragraph_id") in paragraph_by_id:
            return int(paragraph_by_id[group["paragraph_id"]].get("start_char") or 0)
        return 10**9

    return sorted((groups[key] for key in ordered_keys), key=group_start)


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
        _abs = _display_ai_score(_ab.get("ai_likelihood_score")) or 0.0
        _rating = _display_authorship_rating_from_badge(_ab, data.get("document_context", {}), data)
        _rating_label = _rating.get("label") or _ab.get("authorship_rating_label")
        _sc = _shield_colors.get(_abt, "lightgrey")
        _abt_label = _BADGE_TIER_LABELS.get(_abt, _abt)
        lines.append(f"![{_abt_label}](https://img.shields.io/badge/DraftProof_AI-{_abt_label.replace(' ', '_')}-{_sc}) &nbsp; Score `{_abs:.0f}%`")
        if _rating_label:
            lines.append(f"**Authorship Rating:** {_rating_label}")
        # Dual headline: surface the Turnitin / external estimate on page 1 alongside
        # the DraftProof score (mirrors the web report's lead), not only deep in the
        # AI Likelihood section.
        _hdr_ext = _ai_likelihood_bands(_ab).get("external")
        if EXTERNAL_ESTIMATE_DISPLAY_ENABLED and _hdr_ext:
            lines.append(f"**Turnitin / external (estimated):** ~{_hdr_ext['score']}% — {_hdr_ext['label']}")
        lines.append("")
        lines.append(f"> {_TURNITIN_AI_REFERENCE_NOTE}")

        # Writing Quality badge beside AI badge
        wq_tier_header = _ab.get("writing_quality_tier", "")
        wq_score_header = _ab.get("writing_quality_score", 0)
        _wq_labels = {"LOW": "Clean", "LIGHT_REVIEW": "Light+Review", "REVIEW": "Review", "HIGH_REVIEW": "Heavy+Review"}
        _wq_colors = {"LOW": "green", "LIGHT_REVIEW": "yellow", "REVIEW": "orange", "HIGH_REVIEW": "red"}
        if wq_tier_header:
            wq_lbl = _wq_labels.get(wq_tier_header, wq_tier_header)
            wq_clr = _wq_colors.get(wq_tier_header, "lightgrey")
            lines.append(f"![{wq_lbl}](https://img.shields.io/badge/Quality-{wq_lbl}-{wq_clr}) &nbsp; Score `{wq_score_header:.0f}%`")
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

    # ── SUBMISSION RISK HEADLINE (leads; demotes AI-likelihood below) ──
    _submission = _render_submission_risk_headline(report.ai_risk_badge)
    if _submission:
        lines.append(_submission)

    # ── AI LIKELIHOOD (text-pattern axis + external trigger detail) ────
    _headline = _render_ai_likelihood_headline(report.ai_risk_badge)
    if _headline:
        lines.append(_headline)

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
    if _headline:
        lines.append("### How DraftProof calibrates this")
        lines.append("")
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

    # ── CRITICAL THINKING QUESTIONS ───────────────────────────────
    _ct_questions = _render_critical_thinking_questions(report.ai_risk_badge)
    if _ct_questions:
        lines.append('<div style="page-break-before: always;"></div>')
        lines.append("")
        lines.append(_ct_questions)

    # ── 2. FINDINGS BY SEVERITY ───────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Findings")
    lines.append("")

    finding_num = 0

    # Collect all findings across tiers into one list, then group by paragraph
    # in document order — mirrors the /report page layout.
    all_findings_flat = [
        f for tier_level in [Tier.CRITICAL, Tier.HIGH, Tier.MEDIUM, Tier.LOW]
        for f in fb.get(tier_level.value, [])
    ]
    _tier_order = [Tier.CRITICAL, Tier.HIGH, Tier.MEDIUM, Tier.LOW, Tier.CLEAN]

    if all_findings_flat:
        paragraph_groups = _paragraph_finding_groups(all_findings_flat, data)
        paragraph_count = len(paragraph_groups)
        total_findings = len(all_findings_flat)
        group_label = "paragraph" if paragraph_count == 1 else "paragraphs"
        finding_label = "finding" if total_findings == 1 else "findings"
        lines.append(f"*{paragraph_count} {group_label}, {total_findings} {finding_label}*")
        lines.append("")

        for group in paragraph_groups:
            finding_num += 1
            # Worst tier among this paragraph's findings
            worst_tier = next(
                (t for t in _tier_order if any(f.tier == t for f in group["findings"])),
                Tier.LOW,
            )
            lines.append(_render_finding_card(finding_num, worst_tier, group))
            lines.append("")

    if not all_findings_flat:
        lines.append("*No findings detected. Text appears clean.*")
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
