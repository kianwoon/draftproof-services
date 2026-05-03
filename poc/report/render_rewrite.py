"""Dedicated rewrite report renderer — produces markdown for rewrite PDFs."""

import time
from typing import List, Dict, Any, Optional


_TIER_ORDER = ["critical", "high", "medium", "low"]


def _count_findings(report_or_findings: dict) -> int:
    findings = report_or_findings.get("findings", report_or_findings) if report_or_findings else {}
    return sum(len(findings.get(t, [])) for t in _TIER_ORDER)


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
        action = "Apply detector-gated sentence rewrites, then rescan; do not expect this alone to fix evidence gaps."
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


def _plain_signal_action(finding: dict) -> str:
    title = finding.get("title") or finding.get("finding_type") or ""
    title = str(title)
    if title in {"medium_predictability", "high_predictability", "high_topk_predictability"}:
        return "Rewrite the sentence structure and word path; synonym swaps are usually too weak."
    if title == "low_specificity":
        return "Add a concrete example, source detail, number, or author observation."
    if title in {"source_grounding", "polished_but_ungrounded"}:
        return "Attach the claim to evidence or soften it."
    if title in {"uniform_paragraph_structure", "low_burstiness"}:
        return "Revise paragraph rhythm or structure manually."
    if title in {"low_ai_generation_likelihood", "moderate_ai_generation_likelihood", "elevated_ai_generation_likelihood"}:
        return "This is a summary signal; revise the underlying sentence, evidence, and structure drivers."
    recommendation = finding.get("recommendation", "")
    return recommendation or "Review this signal manually."


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
    orig_scan = summary.get("detect_scan_original", {})
    new_scan = summary.get("detect_scan_rewritten", {})
    attempted_scan = summary.get("detect_scan_attempted", {})
    rollback = summary.get("rollback_applied", False)
    mitigation = summary.get("mitigation_plan") or {}

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
        attempted_ai = _ai_score(attempted_scan) if attempted_scan else None
        attempted_wq = _wq_score(attempted_scan) if attempted_scan else None
        attempted_total = _count_findings(attempted_scan) if attempted_scan else None

        orig_axis = orig_scan.get("axis_scores", {})
        new_axis = new_scan.get("axis_scores", {})

        lines.append("### Result")
        lines.append("")
        if rollback and attempted_scan:
            lines.append("The rewrite was tested, but DraftProof kept the original text because the attempted version increased the final scan risk.")
        elif no_text_change:
            lines.append("No automatic rewrite was applied. The remaining signals need author evidence, source support, or manual structure revision.")
        elif n_total < o_total or new_ai < orig_ai:
            lines.append("The rewrite reduced at least one final scan signal and was kept.")
        else:
            lines.append("The rewrite did not reduce the final scan risk. Review the action list below before retrying.")
        lines.append("")

        lines.append("### Detect Scan Comparison")
        lines.append("")
        if attempted_scan:
            lines.append("| Metric | Original | Attempted Rewrite | Final Output |")
            lines.append("|--------|----------|-------------------|--------------|")
            lines.append(f"| **AI Likelihood** | `{orig_ai:.1f}%` | `{attempted_ai:.1f}%` | `{new_ai:.1f}%` |")
            lines.append(f"| **Writing Quality Risk** | `{orig_wq:.1f}%` | `{attempted_wq:.1f}%` | `{new_wq:.1f}%` |")
            lines.append(f"| **Total Findings** | {o_total} | {attempted_total} | {n_total} |")
        else:
            lines.append("| Metric | Original | Final Output | Change |")
            lines.append("|--------|----------|--------------|--------|")
            lines.append(f"| **AI Likelihood** | `{orig_ai:.1f}%` | `{new_ai:.1f}%` | `{ai_delta:+.1f}%` |")
            lines.append(f"| **Writing Quality Risk** | `{orig_wq:.1f}%` | `{new_wq:.1f}%` | `{wq_delta:+.1f}%` |")
            lines.append(f"| **Total Findings** | {o_total} | {n_total} | `{n_total - o_total:+d}` |")

        # Axis-level scores
        if not attempted_scan:
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
        if attempted_scan:
            attempted_findings = attempted_scan.get("findings", {})
            lines.append("| Severity | Original | Attempted Rewrite | Final Output |")
            lines.append("|----------|----------|-------------------|--------------|")
        else:
            lines.append("| Severity | Original | Final Output | Change |")
            lines.append("|----------|----------|--------------|--------|")
        for t in _TIER_ORDER:
            o_c = len(orig_findings.get(t, []))
            n_c = len(new_findings.get(t, []))
            if attempted_scan:
                a_c = len(attempted_findings.get(t, []))
                lines.append(f"| {t.title()} | {o_c} | {a_c} | {n_c} |")
            else:
                lines.append(f"| {t.title()} | {o_c} | {n_c} | `{n_c - o_c:+d}` |")
        lines.append("")

        scan_regressed = new_ai > orig_ai + 0.05 or n_total > o_total
        if no_text_change:
            lines.append("**Outcome: No Automatic Rewrite Applied** — DraftProof kept the original text because the remaining signals require manual review or source-backed context.")
        elif rollback:
            lines.append("**Outcome: No Improvement** — DraftProof kept the original text because the final detect scan regressed.")
        elif scan_regressed:
            lines.append("**Outcome: No Improvement** — Final detect scan regressed despite local rewrite-target progress.")
        elif n_total < o_total or new_ai < orig_ai:
            lines.append("**Outcome: Improved** — Detect scan confirms reduced AI signals after rewrite.")
        elif converged:
            lines.append("**Outcome: Converged** — Rewrite targets met within acceptable bounds.")
        else:
            lines.append("**Outcome: No Improvement** — Final detect scan did not confirm a reduction in AI signals.")
    else:
        # Fallback to internal metrics if detect scan data not available
        orig_risk = summary.get("original_risk", 0)
        final_risk = summary.get("final_risk", 0)
        imp_risk = summary.get("improvement_risk", 0)
        rollback = summary.get("rollback_applied", False)
        if no_text_change:
            lines.append("**Outcome: No Automatic Rewrite Applied** — DraftProof kept the original text because the remaining signals require manual review or source-backed context.")
        elif rollback:
            lines.append("**Outcome: No Improvement** — DraftProof kept the original text because the final detect scan regressed.")
        elif converged:
            lines.append("**Outcome: Converged** — Rewrite targets met within acceptable bounds.")
        elif imp_risk > 0:
            lines.append("**Outcome: Partially Improved** — Some signals reduced, further review recommended.")
        else:
            lines.append("**Outcome: Floor Reached** — Remaining signals are structural, manual review needed.")
    lines.append("")

    lines.append(f"**Passes:** {passes} | **Converged:** {'Yes' if converged else 'No'}")
    if no_text_change_reason:
        lines.append(f"> {no_text_change_reason}")
    if conv_reason:
        lines.append(f"> {conv_reason}")
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

        scope_parts = [
            f"{auto_count} detector-gated sentence target(s)",
            f"{evidence_count} evidence/specificity item(s)",
            f"{structure_count} structure item(s)",
        ]
        lines.append("**Rewrite scope:** " + ", ".join(scope_parts) + ".")
        lines.append("")

    # ── AI Findings That Triggered Rewrite ──────────────────────────
    if ai_findings:
        if no_text_change:
            lines.append("## Signals Reviewed")
            lines.append("")
            auto_count = ((mitigation.get("counts") or {}) if mitigation else {}).get("auto_rewrite", 0)
            if auto_count:
                lines.append("These AI-related findings were reviewed. Automatic targets may exist, but no final rewrite was applied.")
            else:
                lines.append("These AI-related findings were reviewed, but they were not safe automatic rewrite targets.")
        else:
            lines.append("## Signals Considered")
        lines.append("")
        lines.append("| # | Risk | Signal | Action |")
        lines.append("|--:|:----:|--------|--------|")
        for i, f in enumerate(ai_findings, 1):
            title = f.get("title", f.get("finding_type", "unknown")).replace("|", "·")
            risk = (f.get("adjusted_risk") or f.get("risk_level") or "?").upper()
            recommendation = _plain_signal_action(f).replace("|", "·")
            lines.append(f"| {i} | {risk} | {title} | {recommendation} |")
        lines.append("")

    # ── Rewrite Summary ─────────────────────────────────────────────
    lines.append("## Rewrite Summary")
    lines.append("")

    if mitigation:
        counts = mitigation.get("counts") or {}
        mode = mitigation.get("primary_mode", "manual_review").replace("_", " ").title()
        lines.append("### Mitigation Plan")
        lines.append("")
        lines.append(f"**Primary mode:** {mode}")
        lines.append("")
        lines.append("| Bucket | Count | Meaning |")
        lines.append("|--------|------:|---------|")
        bucket_labels = {
            "auto_rewrite": "Detector-gated sentence patches",
            "needs_source_or_example": "Needs author source, citation, example, or concrete detail",
            "structure_guidance": "Needs paragraph/section structure revision",
            "review_only": "Review signal; no automatic edit",
            "protected": "Protected material; preserve verbatim",
        }
        for key, label in bucket_labels.items():
            lines.append(f"| {key.replace('_', ' ').title()} | {counts.get(key, 0)} | {label} |")
        lines.append("")

        drivers = mitigation.get("component_drivers") or []
        if drivers:
            lines.append("**Main badge drivers:**")
            lines.append("")
            lines.append("| Signal | Score | Mitigation |")
            lines.append("|--------|------:|------------|")
            for item in drivers[:8]:
                signal = str(item.get("component", "")).replace("_", " ")
                score = item.get("score", 0)
                fix = str(item.get("mitigation", "")).replace("|", "·")
                lines.append(f"| {signal} | `{score:.1f}%` | {fix} |")
            lines.append("")

    # Pass progression
    progression = summary.get("pass_progression", [])
    if len(progression) > 1:
        lines.append("### Pass Progression")
        lines.append("")
        lines.append("| Pass | Risk Score | Common Ratio |")
        lines.append("|------|-----------|-------------|")
        for p in progression:
            pnum = p.get("pass", 0)
            prisk = p.get("risk", 0)
            ptop10 = p.get("top10_ratio", 0)
            label = "Baseline" if pnum == 0 else f"Pass {pnum}"
            lines.append(f"| {label} | `{prisk:.1%}` | `{ptop10:.1%}` |")
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

    lines.append("## Sentence Rewrites")
    lines.append("")

    if not changed:
        lines.append("No sentences were modified during the rewrite pass.")
        lines.append("")
    else:
        lines.append(f"**{len(changed)} sentence(s) rewritten.**")
        lines.append("")
        lines.append("| # | Tier Change | Common Ratio | Original | Rewritten |")
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
