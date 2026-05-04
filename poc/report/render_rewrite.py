"""Dedicated rewrite report renderer — produces markdown for rewrite PDFs."""

import time
import re
import html
from typing import List, Dict, Any, Optional


_TIER_ORDER = ["critical", "high", "medium", "low"]


def _count_findings(report_or_findings: dict) -> int:
    findings = report_or_findings.get("findings", report_or_findings) if report_or_findings else {}
    return sum(len(findings.get(t, [])) for t in _TIER_ORDER)


def _severity_score(findings: dict) -> int:
    """Weighted finding severity for rewrite outcome comparison."""
    weights = {"critical": 8, "high": 5, "medium": 2, "low": 1}
    return sum(len((findings or {}).get(t, [])) * weights[t] for t in weights)


def _review_burden(findings: dict) -> int:
    """Count findings that normally require user attention."""
    findings = findings or {}
    return (
        len(findings.get("critical", []))
        + len(findings.get("high", []))
        + len(findings.get("medium", []))
    )


def _badge(report: dict) -> dict:
    return (report or {}).get("ai_risk_badge") or {}


def _ai_score(report: dict) -> float:
    score = _badge(report).get("ai_likelihood_score", 0)
    return float(score) if isinstance(score, (int, float)) else 0.0


def _wq_score(report: dict) -> float:
    score = _badge(report).get("writing_quality_score", 0)
    return float(score) if isinstance(score, (int, float)) else 0.0


def _tier(report: dict) -> str:
    return (_badge(report).get("tier") or (report or {}).get("overall_tier", "?")).upper()


def _signal_label(name: str) -> str:
    return str(name or "").replace("_", " ").title()


def _highlight_placeholders(text: str) -> str:
    safe = html.escape(str(text or ""))
    return re.sub(
        r"(\[[^\[\]]+\])",
        r'<mark class="placeholder">\1</mark>',
        safe,
    )


def _manual_reason(item: dict) -> str:
    ftype = item.get("finding_type", "")
    if ftype == "low_specificity":
        return "Add a real example, source detail, or concrete observation. Rewriting alone cannot safely create specificity."
    if ftype in {"source_grounding", "polished_but_ungrounded", "uncited_claim", "uncited_in_body"}:
        return "Connect this claim to a source or soften/remove it if the evidence is not available."
    reason = item.get("reason", "")
    if reason.startswith("Detect classified"):
        return "Manual review is safer than automatic rewriting for this signal."
    return reason


def _top_user_actions(mitigation: dict, limit: int = 5) -> List[str]:
    drivers = mitigation.get("component_drivers") or []
    actions = []
    for driver in drivers:
        bucket = driver.get("bucket")
        component = driver.get("component", "")
        if bucket == "needs_source_or_example":
            if "unsupported" in component or "source" in component or "citation" in component:
                actions.append("Add source-backed support for broad or unsupported claims, or soften those claims.")
            elif "lived_detail" in component:
                actions.append("Add a real classroom/process detail from the author's experience.")
            else:
                actions.append("Narrow generic assertions to the exact classroom, unit, or hairdressing context.")
        elif bucket == "structure_guidance":
            actions.append("Revise paragraph openings and structure before retrying sentence-level rewrite.")
        elif bucket == "auto_rewrite":
            actions.append("Retry detector-gated sentence patches only after evidence and specificity gaps are addressed.")
        if len(actions) >= limit:
            break
    deduped = []
    for action in actions:
        if action not in deduped:
            deduped.append(action)
    return deduped[:limit]


def _bucket_plain_label(bucket: str) -> str:
    labels = {
        "needs_source_or_example": "Evidence or specificity",
        "structure_guidance": "Paragraph structure",
        "auto_rewrite": "Sentence predictability",
        "review_only": "Review only",
        "protected": "Protected text",
    }
    return labels.get(bucket, _signal_label(bucket))


def _driver_guidance(driver: dict) -> Dict[str, str]:
    component = str(driver.get("component", ""))
    bucket = driver.get("bucket", "")
    score = driver.get("score", 0)
    label = _signal_label(component)

    if component in {"generic_assertion_risk", "broad_claim_risk"}:
        issue = "The scan is reacting to broad claims that could fit many essays."
        action = "Limit the claim to the actual topic, class, unit, method, or observation."
    elif component == "unsupported_claim_risk":
        issue = "The scan sees claims that sound confident but are not visibly supported."
        action = "Add a citation, example, or qualifying phrase; remove the claim if support is unavailable."
    elif component in {"source_grounding_risk", "citation_weakness_risk"}:
        issue = "The text does not clearly connect key claims to source material."
        action = "Name the source, connect it to the claim, or add the missing in-text citation."
    elif component == "lived_detail_risk":
        issue = "The writing lacks concrete author/process detail."
        action = "Add a real classroom, project, client, lesson, or workflow detail supplied by the author."
    elif component == "topk_pattern":
        issue = "Some sentences follow very predictable word-choice paths."
        action = "Review the sentence structure; only keep an edit if the final scan improves."
    elif component in {"signpost_paragraph_risk", "paragraph_uniformity_risk"}:
        issue = "Paragraphs may open or progress in a formulaic way."
        action = "Vary paragraph openings and merge, split, or reorder paragraphs where the argument feels templated."
    elif bucket == "auto_rewrite":
        issue = "This is a local wording signal that automatic rewrite can attempt."
        action = driver.get("mitigation") or "Rewrite the affected sentence and rescan."
    elif bucket == "needs_source_or_example":
        issue = "This signal needs information the system should not invent."
        action = driver.get("mitigation") or "Add author-supplied evidence, specificity, or source support."
    elif bucket == "structure_guidance":
        issue = "This signal needs paragraph-level revision."
        action = driver.get("mitigation") or "Revise structure before retrying sentence-level rewrite."
    else:
        issue = driver.get("mitigation") or "Review this signal manually."
        action = "Review the highlighted text and decide whether an edit is needed."

    try:
        score_text = f"{float(score):.1f}%"
    except (TypeError, ValueError):
        score_text = str(score)

    return {
        "focus": _bucket_plain_label(bucket),
        "signal": label,
        "score": score_text,
        "issue": issue,
        "action": action,
    }


def _mode_label(mode: str) -> str:
    return str(mode or "guided_revision").replace("_", " ").title()


def _result_label(
    *,
    no_text_change: bool,
    original_preserved: bool,
    improved_with_review: bool,
    mixed_result: bool,
    regressed: bool,
    improved: bool,
    converged: bool,
) -> str:
    if no_text_change:
        return "Author Input Needed"
    if original_preserved:
        return "Original Preserved"
    if improved_with_review:
        return "Improved With Review"
    if mixed_result:
        return "Mixed Result"
    if regressed:
        return "Review Needed"
    if improved:
        return "Revision Improved"
    if converged:
        return "Revision Complete"
    return "Review Needed"


def render_rewrite_report(
    summary: dict,
    sentence_comparison: List[Dict[str, Any]],
    ai_findings: List[Dict[str, Any]],
    verbose: bool = False,
) -> str:
    """Render a standalone rewrite report as markdown.

    Args:
        summary: The rewrite summary dict from get_rewrite_summary_v2().
        sentence_comparison: Per-sentence before/after from ReportBuilder.
        ai_findings: The AI-flagged findings that triggered the rewrite.
        verbose: Include detailed per-sentence breakdown.

    Returns:
        Markdown string ready for PDF rendering.
    """
    lines: List[str] = []
    ts = time.strftime("%Y-%m-%d %H:%M")

    # ── Header ──────────────────────────────────────────────────────
    lines.append("# DraftProof — Rewrite Report")
    lines.append("")

    # ── Executive Summary ───────────────────────────────────────────
    lines.append("## Executive Summary")
    lines.append("")

    passes = summary.get("passes_completed", 0)
    converged = summary.get("converged", False)
    conv_reason = summary.get("convergence_reason", "")
    no_text_change = summary.get("no_text_change", False)
    no_text_change_reason = summary.get("no_text_change_reason", "")

    # Detect scan comparison.
    orig_scan = summary.get("detect_scan_original_saved") or summary.get("detect_scan_original", {})
    new_scan = summary.get("detect_scan_rewritten", {})
    attempted_scan = summary.get("detect_scan_attempted", {})
    rollback = summary.get("rollback_applied", False)
    mitigation = summary.get("mitigation_plan") or {}
    final_output_preserved = no_text_change

    if orig_scan and new_scan:
        orig_badge = _badge(orig_scan)
        new_badge = _badge(new_scan)

        orig_ai = _ai_score(orig_scan)
        new_ai = _ai_score(new_scan)
        ai_delta = new_ai - orig_ai

        orig_wq = _wq_score(orig_scan)
        new_wq = _wq_score(new_scan)
        wq_delta = new_wq - orig_wq

        orig_findings = orig_scan.get("findings", {})
        new_findings = new_scan.get("findings", {})
        o_total = _count_findings(orig_findings)
        n_total = _count_findings(new_findings)
        o_severity = _severity_score(orig_findings)
        n_severity = _severity_score(new_findings)
        o_review_burden = _review_burden(orig_findings)
        n_review_burden = _review_burden(new_findings)
        orig_axis = orig_scan.get("axis_scores", {})
        new_axis = new_scan.get("axis_scores", {})
        ai_improved = new_ai < orig_ai - 0.05
        ai_worse = new_ai > orig_ai + 0.05
        quality_improved = new_wq < orig_wq - 0.05
        findings_improved = n_total < o_total
        findings_worse = n_total > o_total
        severity_worse = n_severity > o_severity or n_review_burden > o_review_burden
        final_looks_original = (
            abs(new_ai - orig_ai) <= 0.05
            and abs(new_wq - orig_wq) <= 0.05
            and n_total == o_total
        )
        original_preserved = no_text_change or (rollback and final_looks_original)
        final_output_preserved = original_preserved
        improved_with_review = (
            not original_preserved
            and (ai_improved or quality_improved or findings_improved)
            and not findings_worse
            and not severity_worse
            and (ai_improved or quality_improved)
        )
        mixed_result = (
            not original_preserved
            and not improved_with_review
            and (ai_improved or quality_improved or findings_improved)
            and (findings_worse or severity_worse)
        )
        scan_regressed = (
            not original_preserved
            and not improved_with_review
            and not mixed_result
            and (
                ai_worse
                or ((findings_worse or severity_worse) and not ai_improved and not quality_improved)
            )
        )
        improved = (
            not original_preserved
            and not mixed_result
            and not scan_regressed
            and (
                (ai_improved or quality_improved or findings_improved)
                and not findings_worse
                and not severity_worse
            )
        )
        result_label = _result_label(
            no_text_change=no_text_change,
            original_preserved=original_preserved,
            improved_with_review=improved_with_review,
            mixed_result=mixed_result,
            regressed=scan_regressed,
            improved=improved,
            converged=converged,
        )

        lines.append("### Result")
        lines.append("")
        lines.append(f"**{result_label}**")
        lines.append("")
        if original_preserved and rollback and attempted_scan:
            lines.append("The attempted rewrite was not kept because the final scan did not improve.")
        elif no_text_change:
            lines.append("DraftProof found revision opportunities, but the main issues need evidence, examples, or source context from the author.")
        elif improved_with_review:
            lines.append("At least one measured risk signal improved. Review the new findings before keeping the final output.")
        elif mixed_result:
            lines.append("Some risk scores improved, but the final scan added findings or increased review burden. Review the revision plan before keeping the final output.")
        elif improved:
            lines.append("The final output reduced at least one measured risk signal.")
        else:
            lines.append("Review the revision plan below before making another pass.")
        lines.append("")

        lines.append("### Detect Scan Comparison")
        lines.append("")
        lines.append("| Metric | Original | Final Output | Change |")
        lines.append("|--------|----------|--------------|--------|")
        lines.append(f"| **AI Likelihood** | `{orig_ai:.1f}%` | `{new_ai:.1f}%` | `{ai_delta:+.1f}%` |")
        lines.append(f"| **Writing Quality Risk** | `{orig_wq:.1f}%` | `{new_wq:.1f}%` | `{wq_delta:+.1f}%` |")
        lines.append(f"| **Total Findings** | {o_total} | {n_total} | `{n_total - o_total:+d}` |")

        # Axis-level scores
        for axis in sorted(set(list(orig_axis.keys()) + list(new_axis.keys()))):
            o_val = orig_axis.get(axis, 0)
            n_val = new_axis.get(axis, 0)
            if isinstance(o_val, (int, float)) and isinstance(n_val, (int, float)):
                label = axis.replace("_", " ").title()
                delta = n_val - o_val
                lines.append(f"| {label} | `{o_val:.1%}` | `{n_val:.1%}` | `{delta:+.1%}` |")
        lines.append("")

        # Findings breakdown by tier
        lines.append("**Findings by Severity:**")
        lines.append("")
        lines.append("| Severity | Original | Final Output | Change |")
        lines.append("|----------|----------|--------------|--------|")
        for t in _TIER_ORDER:
            o_c = len(orig_findings.get(t, []))
            n_c = len(new_findings.get(t, []))
            lines.append(f"| {t.title()} | {o_c} | {n_c} | `{n_c - o_c:+d}` |")
        lines.append("")
    else:
        # Fallback to internal metrics if detect scan data not available
        imp_risk = summary.get("improvement_risk", 0)
        rollback = summary.get("rollback_applied", False)
        final_output_preserved = no_text_change or rollback
        result_label = _result_label(
            no_text_change=no_text_change,
            original_preserved=no_text_change or rollback,
            improved_with_review=False,
            mixed_result=False,
            regressed=False,
            improved=imp_risk > 0,
            converged=converged,
        )
        lines.append(f"**{result_label}**")
    lines.append("")

    # ── User-facing action list ─────────────────────────────────────
    next_actions = _top_user_actions(mitigation)
    if next_actions:
        lines.append("## What To Do Next")
        lines.append("")
        for i, action in enumerate(next_actions, 1):
            lines.append(f"{i}. {action}")
        lines.append("")

    if mitigation:
        lines.append("## Revision Brief")
        lines.append("")
        counts = mitigation.get("counts") or {}
        auto_count = counts.get("auto_rewrite", 0)
        evidence_count = counts.get("needs_source_or_example", 0)
        structure_count = counts.get("structure_guidance", 0)

        if evidence_count or structure_count:
            lines.append(
                "The scan found sentence-level rewrite targets, but the strongest remaining risk drivers require author-supplied evidence, concrete context, or paragraph-level revision. Automatic rewriting should not invent those details."
            )
        elif auto_count:
            lines.append(
                "The main remaining risk is local sentence predictability. Automatic rewrite can attempt these targets, but each edit still needs a rescan gate."
            )
        else:
            lines.append(
                "No safe automatic rewrite target was found. Use the guidance below as a manual revision checklist."
            )
        lines.append("")

        lines.append("| Focus Area | Signal | Score | What It Means | Concrete Revision |")
        lines.append("|------------|--------|------:|---------------|-------------------|")
        for driver in (mitigation.get("component_drivers") or [])[:6]:
            item = _driver_guidance(driver)
            lines.append(
                f"| {item['focus']} | {item['signal']} | `{item['score']}` | "
                f"{item['issue'].replace('|', '·')} | {item['action'].replace('|', '·')} |"
            )
        lines.append("")

        sentence_scope_text = (
            f"{auto_count} sentence-level target(s) identified; no sentence edits kept in the final output"
            if final_output_preserved
            else f"{auto_count} sentence-level target(s) identified for detector-gated editing"
        )
        scope_parts = [
            sentence_scope_text,
            f"{evidence_count} evidence/specificity item(s)",
            f"{structure_count} structure item(s)",
        ]
        lines.append("**Rewrite scope:** " + ", ".join(scope_parts) + ".")
        lines.append("")

        patterns = mitigation.get("reference_patterns") or []
        if patterns:
            lines.append("## Reference Revision Examples")
            lines.append("")
            lines.append(
                "For learning and revision guidance only. Do not submit these patterns as-is; replace the placeholders with the author's own evidence, source, and context."
            )
            lines.append("")
            for i, pattern in enumerate(patterns, 1):
                focus = str(pattern.get("focus", "Revision pattern")).replace("|", "·")
                excerpt = str(pattern.get("flagged_excerpt", "")).replace("|", "·")
                instead = _highlight_placeholders(
                    str(pattern.get("instead_of", "")).replace("|", "·")
                )
                try_pattern = _highlight_placeholders(
                    str(pattern.get("try_pattern", "")).replace("|", "·")
                )
                why = str(pattern.get("why", "")).replace("|", "·")
                note = str(pattern.get("application_note", "")).replace("|", "·")
                lines.append(f"### Pattern {i}: {focus}")
                lines.append("")
                if excerpt:
                    lines.append(f"**Flagged excerpt:** “{html.escape(excerpt)}”")
                    lines.append("")
                if instead:
                    lines.append(f"**Instead of:** {instead}")
                    lines.append("")
                lines.append(f"**Try this pattern:** {try_pattern}")
                lines.append("")
                if why:
                    lines.append(f"**Why this helps:** {why}")
                    lines.append("")
                if note:
                    lines.append(f"**How to apply:** {note}")
                    lines.append("")

    # ── Revision Plan ───────────────────────────────────────────────
    lines.append("## Revision Plan")
    lines.append("")

    if mitigation:
        counts = mitigation.get("counts") or {}
        mode = _mode_label(mitigation.get("primary_mode", "guided_revision"))
        lines.append(f"**Recommended path:** {mode}")
        lines.append("")
        lines.append("| Bucket | Count | Meaning |")
        lines.append("|--------|------:|---------|")
        bucket_labels = {
            "auto_rewrite": (
                "Sentence-level targets identified; no sentence edits kept in the final output"
                if final_output_preserved
                else "Sentence-level targets identified for detector-gated editing"
            ),
            "needs_source_or_example": "Needs author source, citation, example, or concrete detail",
            "structure_guidance": "Needs paragraph/section structure revision",
            "review_only": "Review signal; no automatic edit",
            "protected": "Protected material; preserve verbatim",
        }
        bucket_names = {
            "auto_rewrite": "Sentence Targets",
            "needs_source_or_example": "Needs Evidence",
            "structure_guidance": "Structure Work",
            "review_only": "Review Only",
            "protected": "Protected",
        }
        for key, label in bucket_labels.items():
            lines.append(f"| {bucket_names[key]} | {counts.get(key, 0)} | {label} |")
        lines.append("")

        drivers = mitigation.get("component_drivers") or []
        if drivers:
            lines.append("**Revision focus:**")
            lines.append("")
            lines.append("| Revision Focus | Signal Strength | Suggested Action |")
            lines.append("|--------|------:|------------|")
            for item in drivers[:8]:
                signal = str(item.get("component", "")).replace("_risk", "").replace("_", " ").title()
                score = item.get("score", 0)
                fix = str(item.get("mitigation", "")).replace("|", "·")
                lines.append(f"| {signal} | `{score:.1f}%` | {fix} |")
            lines.append("")

    # Manual actions remaining
    manual = summary.get("manual_actions", [])
    if manual:
        lines.append("### Manual Review Required")
        lines.append("")
        lines.append("These items need author judgment or source-backed detail before another rewrite attempt:")
        lines.append("")
        lines.append("| # | Risk | Finding | Reason | Evidence |")
        lines.append("|--:|:----:|---------|--------|----------|")
        for i, m in enumerate(manual, 1):
            ftype = m.get("finding_type", "?")
            risk = m.get("risk_level", "?").upper()
            reason = _manual_reason(m).replace("|", "·")
            evidence = m.get("evidence", "").replace("|", "·")
            lines.append(f"| {i} | {risk} | {ftype} | {reason} | {evidence} |")
        lines.append("")

    # Protected / Review-only
    protected = summary.get("protected_actions", [])
    if protected:
        lines.append(f"**Protected findings (skipped):** {len(protected)}")
        lines.append("")

    # ── Sentence Rewrites ───────────────────────────────────────────
    changed = [
        sc for sc in sentence_comparison
        if sc.get("orig_sentence", "") != sc.get("new_sentence", "")
    ]

    if not changed:
        pass
    else:
        lines.append("## Sentence Changes")
        lines.append("")
        lines.append(f"**{len(changed)} sentence(s) rewritten.**")
        lines.append("")
        lines.append("| # | Change | Signal Ratio | Original | Final Output |")
        lines.append("|--:|:-----------:|:------------:|----------|-----------|")
        for i, sc in enumerate(changed, 1):
            orig_tier = sc.get("orig_tier", "?")
            new_tier = sc.get("new_tier", "?")
            tier_change = f"{orig_tier} → {new_tier}" if orig_tier != new_tier else orig_tier
            orig_text = sc.get("orig_sentence", "").replace("\n", " ").replace("|", "·")
            new_text = sc.get("new_sentence", "").replace("\n", " ").replace("|", "·")
            orig_top10_s = sc.get("orig_top10", 0)
            new_top10_s = sc.get("new_top10", 0)
            ratio_change = f"{orig_top10_s:.0%} → {new_top10_s:.0%}"
            lines.append(f"| {i} | {tier_change} | {ratio_change} | {orig_text} | {new_text} |")
        lines.append("")

    # ── Legend ───────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by DraftProof Rewrite Pipeline*")
    lines.append(f"*{ts}*")
    lines.append("")

    return "\n".join(lines)
