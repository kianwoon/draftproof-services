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

# ── Tier display config ──────────────────────────────────────────────

_BADGE_TIER_LABELS = {
    "GREEN": "Low Risk",
    "AMBER": "Moderate Risk",
    "ORANGE": "High Risk",
    "RED": "Very High Risk",
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

    # Header badge: prefer calibrated AI Risk badge, else fallback to finding tier
    _shield_colors = {"GREEN": "green", "AMBER": "yellow", "ORANGE": "orange", "RED": "red"}
    if report.ai_risk_badge:
        _ab = report.ai_risk_badge
        _abt = _ab.get("tier", "")
        _abs = _ab.get("calibrated_ai_score", 0)
        _sc = _shield_colors.get(_abt, "lightgrey")
        _abt_label = _BADGE_TIER_LABELS.get(_abt, _abt)
        lines.append(f"![{_abt_label}](https://img.shields.io/badge/Turnitin_Tier-{_abt_label.replace(' ', '_')}-{_sc}) &nbsp; Score `{_abs:.2f}%`")
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
        badge_score = badge.get("calibrated_ai_score", 0)
        badge_band = badge.get("turnitin_like_band", "")
        if not badge_band:
            badge_band = badge_tier
        badge_band = badge_band.replace("_", " ").title()
        badge_gc = badge.get("grounding_credit", 0) or badge.get("grounding_components", {}).get("grounding_credit", 0)
        badge_red_flags = badge.get("red_flags", 0)
        badge_reasons = badge.get("pattern_reasons", []) or badge.get("reasons", [])

        shield_colors = {
            "GREEN": "green",
            "AMBER": "yellow",
            "ORANGE": "orange",
            "RED": "red",
        }
        shield_color = shield_colors.get(badge_tier, "lightgrey")

        lines.append("### AI Risk Badge")
        lines.append("")
        badge_tier_label = _BADGE_TIER_LABELS.get(badge_tier, badge_tier)
        lines.append(f"![{badge_tier_label}](https://img.shields.io/badge/Turnitin_Tier-{badge_tier_label.replace(' ', '_')}-{shield_color})")
        lines.append("")
        lines.append(f"- **Score**: `{badge_score:.2f}%`")
        if badge_gc > 0:
            lines.append(f"- **Grounding credit**: `{badge_gc:.1f}%`")
        # Cluster breakdown
        ai_style = badge.get("ai_style_score", 0)
        gq_score = badge.get("grounding_quality_risk", 0)
        proc_score = badge.get("structure_process_score", 0)
        if ai_style > 0 or gq_score > 0 or proc_score > 0:
            lines.append(f"- **Text pattern cluster**: `{ai_style:.1f}%`")
            lines.append(f"- **Grounding quality risk**: `{gq_score:.1f}%`")
            lines.append(f"- **Structure/process cluster**: `{proc_score:.1f}%`")
        wr_score = badge.get("writing_review_score", 0)
        wr_band = badge.get("writing_review_band", "")
        if wr_score > 0:
            wr_label = wr_band.replace("_", " ").title() if wr_band else ""
            lines.append(f"- **Writing review**: `{wr_score:.1f}%` ({wr_label})")
        if badge_red_flags > 0:
            lines.append(f"- **Red flags**: {badge_red_flags}/5")
        if badge_reasons:
            lines.append(f"- **Patterns**: {', '.join(badge_reasons)}")
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
        badge_score = report.ai_risk_badge.get("calibrated_ai_score", 0)
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
        badge_score = report.ai_risk_badge.get("calibrated_ai_score", 0)
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
        lines.append(f"> **Tier Reason:** Tier {badge_tier_display} at {badge_score:.1f}% — {explanation}")
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
                "no_rewrite_triggered": "No rewrite triggered",
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

        lines.append("| # | Src | Sig | Detail | Evidence | Action |")
        lines.append("|--:|:---:|:---:|--------|----------|--------|")

        def _cell(text):
            if not text:
                return "-"
            return text.replace("|", "·").replace("\n", " ")

        for f in findings:
            finding_num += 1
            used_scanners.add(f.scanner)
            used_signals.add(f.title)
            scanner = _SCANNER_CODES.get(f.scanner, f.scanner)
            signal = _SIGNAL_CODES.get(f.title, f.title)
            detail = _cell(f.detail)
            evidence = _cell(f.evidence)
            action = _cell(f.recommendation)
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
        lines.append(f"| **Risk Score** | `{rw.original_risk:.1%}` | `{rw.final_risk:.1%}` | `{rw.improvement_risk:+.1%}` |")
        lines.append(f"| **Common Ratio** | `{rw.original_top10:.1%}` | `{rw.final_top10:.1%}` | `{rw.improvement_top10:+.1%}` |")
        if rw.original_tier:
            lines.append(f"| **Tier** | {rw.original_tier.upper()} | {rw.rewritten_tier.upper()} | |")
        if rw.original_findings or rw.rewritten_findings:
            lines.append(f"| **Findings** | {rw.original_findings} | {rw.rewritten_findings} | `{rw.original_findings - rw.rewritten_findings:+d}` |")
        lines.append("")
        lines.append(f"- **Passes**: {rw.passes_completed}")
        lines.append(f"- **Converged**: {'Yes' if rw.converged else 'No'} ({rw.convergence_reason})")
        if rw.detect_loops_used:
            lines.append(f"- **Detect-rewrite loops**: {rw.detect_loops_used}")
        if rw.reverted:
            lines.append(f"- **Reverted**: Yes — {rw.revert_reason}")
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
            lines.append("<details>")
            lines.append("<summary>Sentence Comparison</summary>")
            lines.append("")
            for i, sc in enumerate(rw.sentence_comparison, 1):
                orig = sc.get("original", "").replace("\n", " ")
                rew = sc.get("rewritten", "").replace("\n", " ")
                lines.append(f"**#{i}**")
                lines.append(f"- Original: {orig}")
                lines.append(f"- Rewritten: {rew}")
                lines.append("")
            lines.append("")
            lines.append("</details>")
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
