"""DraftProof Report Renderer — Markdown output from DraftReport.

Produces a structured report with:
  1. EXECUTIVE SUMMARY — overall tier badge, risk gauge, scanner scores
  2. FINDINGS BY SEVERITY — grouped by tier with collapsible details
  3. SCANNER BREAKDOWN — per-scanner tables with sentence-level detail
  4. REWRITE SUMMARY — before/after comparison (when present)
  5. APPENDIX — full sentence tables in collapsible sections
"""

from typing import Optional

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
        and _component_score(ai_components, "topk_pattern") >= 80
        and _component_score(ai_components, "generic_assertion_risk") >= 80
        and (
            _component_score(writing_components, "unsupported_claim_risk") >= 80
            or _component_score(writing_components, "source_grounding_risk") >= 70
            or _component_score(writing_components, "broad_claim_risk") >= 70
        )
    )

    if ai_score >= 65 or tier == "RED" or (ai_score >= 60 and high_quality) or high_component_alignment:
        return {
            "label": "AI-Generated Signals",
            "short_label": "AI-Generated",
            "summary": "High AI-style signal strength across the detect pipeline.",
            "confidence": (badge or {}).get("confidence") or "",
            "disclaimer": "This rating summarizes DraftProof detector signals. It is not proof of authorship.",
        }
    if ai_score >= 48 or tier == "ORANGE":
        return {"label": "Likely AI", "short_label": "Likely AI"}
    if ai_score >= 32 or tier == "AMBER":
        return {"label": "Possible AI-Assisted", "short_label": "Possible AI"}
    if ai_score >= 18:
        return {"label": "Unlikely AI", "short_label": "Unlikely AI"}
    return {"label": "Human-Likely", "short_label": "Human"}


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
        _rating = _authorship_rating_from_badge(_ab)
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

    # AI Risk Badge
    if report.ai_risk_badge:
        badge = report.ai_risk_badge
        badge_tier = badge.get("tier", "")

        # ── AI Generation Likelihood ──
        ai_score = badge.get("ai_likelihood_score", 0)
        shield_colors = {
            "GREEN": "green",
            "AMBER": "yellow",
            "ORANGE": "orange",
            "RED": "red",
        }
        shield_color = shield_colors.get(badge_tier, "lightgrey")
        badge_tier_label = _BADGE_TIER_LABELS.get(badge_tier, badge_tier)

        lines.append("### AI Generation Likelihood")
        lines.append("")
        lines.append(f"![{badge_tier_label}](https://img.shields.io/badge/Turnitin_AI_Tier-{badge_tier_label.replace(' ', '_')}-{shield_color})")
        lines.append("")
        lines.append(f"- **Score**: `{ai_score:.2f}%`")
        rating = _authorship_rating_from_badge(badge)
        if rating:
            lines.append(f"- **Authorship Rating**: **{rating.get('label', '')}**")
            if rating.get("summary"):
                lines.append(f"- **Rating Meaning**: {rating.get('summary')}")
            if rating.get("confidence"):
                lines.append(f"- **Rating Confidence**: `{str(rating.get('confidence')).title()}`")
            if rating.get("disclaimer"):
                lines.append(f"- **Important**: {rating.get('disclaimer')}")

        cluster_boost = badge.get("ai_cluster_boost", 0)
        cluster_name = badge.get("ai_cluster_name")
        if cluster_name:
            lines.append(f"- **Cluster**: {cluster_name} (+`{cluster_boost:.1f}%`)")

        ai_components = badge.get("ai_components", {})
        if ai_components:
            non_zero = {k: v for k, v in ai_components.items() if v > 0}
            if non_zero:
                lines.append("")
                lines.append("| Signal | Score | What It Means |")
                lines.append("|--------|------:|---------------|")
                for comp_name, comp_val in non_zero.items():
                    label, desc = _AI_COMPONENT_LABELS.get(comp_name, (comp_name, ""))
                    lines.append(f"| {label} | `{comp_val:.1f}%` | {desc} |")

        # ── Writing Quality Risk ──
        wq_tier = badge.get("writing_quality_tier", "LOW")
        wq_score = badge.get("writing_quality_score", 0)
        wq_labels = {
            "LOW": "Clean",
            "LIGHT_REVIEW": "Light Review",
            "REVIEW": "Review",
            "HIGH_REVIEW": "Heavy Review",
        }
        wq_colors = {
            "LOW": "green",
            "LIGHT_REVIEW": "yellow",
            "REVIEW": "orange",
            "HIGH_REVIEW": "red",
        }
        wq_label = wq_labels.get(wq_tier, wq_tier)
        wq_color = wq_colors.get(wq_tier, "lightgrey")

        lines.append("")
        lines.append("### Writing Quality Risk")
        lines.append("")
        lines.append(f"![{wq_label}](https://img.shields.io/badge/Quality-{wq_label.replace(' ', '_')}-{wq_color})")
        lines.append("")
        lines.append(f"- **Score**: `{wq_score:.2f}%`")

        wq_components = badge.get("writing_components", {})
        if wq_components:
            non_zero = {k: v for k, v in wq_components.items() if v > 0}
            if non_zero:
                lines.append("")
                lines.append("| Signal | Score | What It Means |")
                lines.append("|--------|------:|---------------|")
                for comp_name, comp_val in non_zero.items():
                    label, desc = _WQ_COMPONENT_LABELS.get(comp_name, (comp_name, ""))
                    lines.append(f"| {label} | `{comp_val:.1f}%` | {desc} |")

        # ── Combined Recommendation ──
        review_priority = badge.get("review_priority", "")
        if review_priority and review_priority != "clean":
            priority_labels = {
                "high_ai_concern": "High AI-generation concern",
                "serious_review": "Serious review needed",
                "ai_review": "AI-style patterns detected",
                "possible_humanised_ai": "Possible humanised AI or weak grounding",
                "strong_review": "Review strongly recommended",
                "mild_review": "Some patterns worth reviewing",
                "writing_review": "Not AI-like, but writing quality needs improvement",
                "note": "Minor quality concerns",
            }
            lines.append("")
            lines.append("### Recommendation")
            lines.append("")
            lines.append(priority_labels.get(review_priority, review_priority))

        lines.append("")

    # Build bar strings
    max_n = max(n_critical, n_high, n_medium, n_low, 1)
    def _bar(count):
        bar_w = round(count / max_n * 15) if max_n else 0
        return "#" * bar_w + "-" * (15 - bar_w)

    # Build scanner rows
    scanners_present = set()
    for findings_list in fb.values():
        for f in findings_list:
            scanners_present.add(f.scanner)
    scanner_rows = []
    for scanner in sorted(scanners_present):
        count = _count_by_scanner(fb, scanner)
        hc = _high_count_by_scanner(fb, scanner)
        code = _SCANNER_CODES.get(scanner, scanner)
        score_str = _scanner_score(report, scanner)
        scanner_rows.append((code, count, hc, score_str))

    # Metrics table — use badge tier if available, otherwise finding-based tier
    display_tier = tier.value.upper()
    display_emoji = emoji
    if report.ai_risk_badge:
        badge_tier_val = report.ai_risk_badge.get("tier", "")
        if badge_tier_val:
            display_tier = _BADGE_TIER_LABELS.get(badge_tier_val, badge_tier_val)
            badge_emoji_map = {"GREEN": "[ok]", "AMBER": "[!]", "ORANGE": "[!!]", "RED": "[!!!]"}
            display_emoji = badge_emoji_map.get(badge_tier_val, emoji)

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Integrity Tier** | **{display_tier}** |")
    lines.append(f"| **Total Findings** | **{total}** |")
    lines.append(f"| Scan Time | `{report.scan_time_seconds:.1f}s` |")
    if report.generated_at:
        lines.append(f"| Generated | {report.generated_at} |")

    # Severity counts inline
    sev_parts = []
    if n_critical: sev_parts.append(f"{n_critical} Critical")
    if n_high: sev_parts.append(f"{n_high} High")
    if n_medium: sev_parts.append(f"{n_medium} Medium")
    if n_low: sev_parts.append(f"{n_low} Low")
    if sev_parts:
        lines.append(f"| **Breakdown** | {' / '.join(sev_parts)} |")
    lines.append("")

    # Scanner results table
    if scanner_rows:
        lines.append("| Src | Found | H/C | Score |")
        lines.append("|-----|-------|-----|-------|")
        for code, count, hc, score_str in scanner_rows:
            lines.append(f"| {code} | {count} | {hc} | {score_str} |")
        lines.append("")

    # Risk gauges — suppressed when Layer 3 badge is present (cluster analysis replaces it)
    if not report.ai_risk_badge:
        signal_lines = []
        if report.predictability:
            p = report.predictability
            signal_lines.append(f"- **Predictability**: {_risk_gauge(p.overall_risk)}")
            if p.generic_phrases_found:
                phrases = ", ".join(f"`{ph}`" for ph in p.generic_phrases_found[:5])
                signal_lines.append(f"- **Generic Phrases**: {phrases}")
            dist_parts = [f"{k}: {v}" for k, v in p.risk_distribution.items()]
            signal_lines.append(f"- **Distribution**: {' | '.join(dist_parts)}")

        # Check for specificity and AI signals in findings
        for fl in fb.values():
            for f in fl:
                if f.title == "low_specificity" and f.metadata:
                    adjusted_risk = f.metadata.get("adjusted_specificity_concern",
                                                   f.metadata.get("adjusted_specificity_risk"))
                    raw_risk = f.metadata.get("raw_specificity_concern",
                                              f.metadata.get("raw_specificity_risk",
                                              f.metadata.get("specificity_risk", 0)))
                    spec_score = f.metadata.get("raw_specificity_score",
                                                f.metadata.get("specificity_score", 0))
                    dg_level = f.metadata.get("domain_grounding_level", "")
                    dg_idx = f.metadata.get("domain_grounding_index", "")

                    if adjusted_risk is not None and abs(adjusted_risk - raw_risk) > 0.05:
                        signal_lines.append(
                            f"- **Specificity**: `{adjusted_risk:.0%}` adjusted "
                            f"(raw concern: `{raw_risk:.0%}`)"
                        )
                        adj = f.metadata.get("adjustment", {})
                        if adj.get("reason"):
                            signal_lines.append(f"  - _Reason: {adj['reason']}_")
                    else:
                        signal_lines.append(f"- **Specificity Risk**: `{spec_score:.0%}`")

                    if dg_idx:
                        signal_lines.append(
                            f"- **Domain Grounding**: index `{dg_idx:.2f}`, level `{dg_level}`"
                        )
                if "ai_generation_likelihood" in f.title and f.metadata:
                    ai_lik = f.metadata.get("ai_likelihood", 0)
                    if ai_lik:
                        signal_lines.append(f"- **AI Likelihood**: `{ai_lik:.1%}`")

        if signal_lines:
            lines.append("### Risk Gauges")
            lines.append("")
            for sl in signal_lines:
                lines.append(sl)
            lines.append("")

    if report.similarity:
        s = report.similarity
        lines.append(f"- **Similarity Risk**: `{_pct(s.overall_risk)}`")
        dist_parts = [f"{k}: {v}" for k, v in s.risk_distribution.items()]
        lines.append(f"- **Distribution**: {' | '.join(dist_parts)}")
        lines.append("")

    if report.citation:
        c = report.citation
        lines.append(f"- **Citation Style**: `{c.citation_style}`")
        lines.append(f"- **In-text**: {c.in_text_count} | **Bibliography**: {c.bib_entry_count}")
        lines.append("")

    # Verdict — use badge tier when available
    if report.ai_risk_badge:
        badge_tier_val = report.ai_risk_badge.get("tier", "")
        badge_score = report.ai_risk_badge.get("ai_likelihood_score", 0)
        verdict_map = {
            "GREEN": f"Low risk across all clusters (score: {badge_score:.1f}%). Text appears ready for submission.",
            "AMBER": f"Moderate concerns detected (score: {badge_score:.1f}%). Review flagged areas before submission.",
            "ORANGE": f"Elevated integrity risk (score: {badge_score:.1f}%). Multiple clusters show aligned patterns — review recommended.",
            "RED": f"High integrity risk (score: {badge_score:.1f}%). Strong evidence across independent clusters — thorough review recommended before submission.",
        }
        verdict = verdict_map.get(badge_tier_val, _verdict(tier, total))
    else:
        verdict = _verdict(tier, total)
    lines.append(f"> **Verdict:** {verdict}")
    lines.append("")

    # Primary action — promoted above all findings
    action_summary = data.get("actionable_summary", {})
    pa = action_summary.get("primary_action", "")
    if pa and pa != "review_only":
        # Build specific label from actual findings, not generic text
        pa_labels = {
            "add_citations": "Add citations for uncited claims",
            "improve_specificity": "Strengthen specificity with concrete details",
            "reduce_formulaic_language": "Reduce formulaic phrasing",
            "address_findings": "Address flagged findings",
        }
        # Enhance "address_findings" with the actual finding title
        if pa == "address_findings":
            auto_fixable = action_summary.get("auto_fixable", [])
            if auto_fixable:
                first = auto_fixable[0]
                finding_title = first.get("title", "").replace("_", " ")
                finding_id = first.get("finding_id", "")
                pa_labels["address_findings"] = (
                    f"Address {finding_title} ({finding_id})"
                )
        lines.append(f"> **Primary Action:** {pa_labels.get(pa, pa.replace('_', ' ').title())}")
        # Secondary action: show review-level items or manual-required citations
        manual_req = action_summary.get("manual_required", [])
        review_only = action_summary.get("review_only", [])
        if pa != "add_citations" and manual_req:
            cite_manual = [e for e in manual_req if "citation" in e.get("title", "").lower()]
            if cite_manual:
                titles = [e.get("title", "").replace("_", " ") for e in cite_manual[:2]]
                lines.append(f"> **Secondary Action:** Add citation for {', '.join(titles)}")
        if review_only:
            titles = [r.get("title", "").replace("_", " ") for r in review_only[:2]]
            lines.append(f"> **Review:** {', '.join(titles)} — no immediate fix needed")
        lines.append("")

    # Tier derivation audit — use badge tier reason
    if report.ai_risk_badge:
        badge_tier = report.ai_risk_badge.get("tier", "")
        badge_score = report.ai_risk_badge.get("ai_likelihood_score", 0)
        badge_reasons = report.ai_risk_badge.get("reasons", [])
        badge_guardrails = report.ai_risk_badge.get("guardrails", [])

        tier_explanation = {
            "GREEN": "Low risk across all clusters. No significant concerns detected.",
            "AMBER": "Moderate concern in grounding quality. Review recommended.",
            "ORANGE": "Elevated risk — two clusters aligned or blended score above threshold.",
            "RED": "High risk — aligned evidence across independent clusters indicates writing process concerns.",
        }
        explanation = tier_explanation.get(badge_tier, "")
        badge_tier_display = _BADGE_TIER_LABELS.get(badge_tier, badge_tier)
        rating = _authorship_rating_from_badge(report.ai_risk_badge)
        if rating.get("label"):
            lines.append(
                f"> **Tier Reason:** {rating.get('label')} at {badge_score:.1f}% "
                f"(Tier {badge_tier_display}) — {rating.get('summary') or explanation}"
            )
        else:
            lines.append(f"> **Tier Reason:** Tier {badge_tier_display} at {badge_score:.1f}% — {explanation}")
        if rating.get("caution_notes"):
            lines.append(f"> **Rating Notes:** {' '.join(rating.get('caution_notes') or [])}")
        if badge_reasons:
            readable = [r.replace("_", " ") for r in badge_reasons]
            lines.append(f"> **Triggers:** {', '.join(readable)}")
        if badge_guardrails:
            readable = [g.replace("_", " ") for g in badge_guardrails]
            lines.append(f"> **Guardrails applied:** {', '.join(readable)}")
        lines.append("")
    elif report.overall_tier_reason:
        lines.append(f"> **Tier Reason:** {report.overall_tier_reason}")
        lines.append("")
    if report.raw_overall_tier != report.adjusted_overall_tier:
        if not report.ai_risk_badge:
            lines.append(f"> **Tier Adjustment:** Raw `{report.raw_overall_tier.upper()}` -> Adjusted `{report.adjusted_overall_tier.upper()}`")
            lines.append("")

    # Axis scores — suppressed when badge is present (badge cluster view replaces this)
    if not report.ai_risk_badge:
        axis_scores = data.get("axis_scores")
        if axis_scores:
            axis_labels = {
                "predictability": "Predictability",
                "similarity": "Similarity",
                "citation": "Citation",
                "specificity": "Specificity",
                "domain_grounding": "Domain Grounding",
            }
            axis_icons = {"clear": "[OK]", "review": "[~]", "attention": "[!]", "strong": "[*]", "moderate": "[+]", "weak": "[-]"}
            parts = []
            for key, label in axis_labels.items():
                val = axis_scores.get(key, "clear")
                icon = axis_icons.get(val, "[?]")
                parts.append(f"{label}: {icon} {val}")
            lines.append(f"> **Signal Axes:** {'  |  '.join(parts)}")
            lines.append("")

    # Reason codes — suppressed when badge is present
    if not report.ai_risk_badge:
        reason_codes = data.get("reason_codes")
        if reason_codes:
            code_labels = {
                "no_high_or_critical_findings": "No high/critical findings",
                "low_ai_pattern_score": "Low AI pattern score",
                "strong_domain_grounding": "Strong domain grounding",
                "mostly_review_only_findings": "Mostly review-only findings",
                "predictability_unconfirmed": "Predictability unconfirmed",
                "rewrite_not_recommended": "Rewrite not recommended",
            }
            readable = [code_labels.get(c, c) for c in reason_codes]
            lines.append(f"> **Tier Rationale:** {'; '.join(readable)}")
            lines.append("")

    # Short-text confidence warning
    doc_ctx = data.get("document_context", {})
    word_count = doc_ctx.get("word_count", 0)
    sent_count = doc_ctx.get("sentence_count", 0)
    if word_count >= 800 or sent_count >= 25:
        lines.append(
            f"> **Sample Confidence:** Medium-High — {word_count} words / "
            f"{sent_count} sentences. Enough body text for reliable document-level signals."
        )
        lines.append("")
    elif word_count < 250 or sent_count < 10:
        lines.append(
            f"> **Sample Confidence:** Low — only {word_count} words / "
            f"{sent_count} sentences. Document-level scores are unstable."
        )
        lines.append("")
    else:
        lines.append(
            f"> **Sample Confidence:** Medium — {word_count} words / "
            f"{sent_count} sentences. Enough for advisory signals, but results remain advisory."
        )
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
