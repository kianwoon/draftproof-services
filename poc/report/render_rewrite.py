"""Dedicated rewrite report renderer — produces markdown for rewrite PDFs."""

import time
from typing import List, Dict, Any, Optional


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

    # Detect scan before/after — run full detect pipeline on both texts
    orig_scan = summary.get("detect_scan_original", {})
    new_scan = summary.get("detect_scan_rewritten", {})

    if orig_scan and new_scan:
        # Both are full report_to_dict() outputs with:
        #   overall_tier, ai_risk_badge, finding_count, findings{}, axis_scores
        orig_badge = orig_scan.get("ai_risk_badge") or {}
        new_badge = new_scan.get("ai_risk_badge") or {}

        orig_ai = orig_badge.get("ai_likelihood_score", 0)
        new_ai = new_badge.get("ai_likelihood_score", 0)
        ai_delta = new_ai - orig_ai

        orig_wq = orig_badge.get("writing_quality_score", 0)
        new_wq = new_badge.get("writing_quality_score", 0)
        wq_delta = new_wq - orig_wq

        orig_tier = (orig_badge.get("tier") or orig_scan.get("overall_tier", "?")).upper()
        new_tier = (new_badge.get("tier") or new_scan.get("overall_tier", "?")).upper()

        # Count findings by tier
        orig_findings = orig_scan.get("findings", {})
        new_findings = new_scan.get("findings", {})
        _tier_order = ["critical", "high", "medium", "low"]

        def _count_findings(fdict):
            return sum(len(fdict.get(t, [])) for t in _tier_order)

        o_total = _count_findings(orig_findings)
        n_total = _count_findings(new_findings)

        # Axis scores
        orig_axis = orig_scan.get("axis_scores", {})
        new_axis = new_scan.get("axis_scores", {})

        lines.append("### Detect Scan Comparison")
        lines.append("")
        lines.append("| Metric | Original | Rewritten | Change |")
        lines.append("|--------|----------|-----------|--------|")
        lines.append(f"| **Overall Tier** | {orig_tier} | {new_tier} | |")
        lines.append(f"| **AI Likelihood** | `{orig_ai:.1f}%` | `{new_ai:.1f}%` | `{ai_delta:+.1f}%` |")
        lines.append(f"| **Writing Quality Risk** | `{orig_wq:.1f}%` | `{new_wq:.1f}%` | `{wq_delta:+.1f}%` |")

        # Axis-level scores
        for axis in sorted(set(list(orig_axis.keys()) + list(new_axis.keys()))):
            o_val = orig_axis.get(axis, 0)
            n_val = new_axis.get(axis, 0)
            if isinstance(o_val, (int, float)) and isinstance(n_val, (int, float)):
                label = axis.replace("_", " ").title()
                delta = n_val - o_val
                lines.append(f"| {label} | `{o_val:.1%}` | `{n_val:.1%}` | `{delta:+.1%}` |")

        lines.append(f"| **Total Findings** | {o_total} | {n_total} | `{n_total - o_total:+d}` |")
        lines.append("")

        # Findings breakdown by tier
        lines.append("**Findings by Severity:**")
        lines.append("")
        lines.append("| Severity | Original | Rewritten | Change |")
        lines.append("|----------|----------|-----------|--------|")
        for t in _tier_order:
            o_c = len(orig_findings.get(t, []))
            n_c = len(new_findings.get(t, []))
            lines.append(f"| {t.title()} | {o_c} | {n_c} | `{n_c - o_c:+d}` |")
        lines.append("")

        # Overall outcome
        rollback = summary.get("rollback_applied", False)
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

    # Rewrite decision context
    decision = summary.get("rewrite_decision") or {}
    if decision:
        mode = decision.get("mode", "targeted")
        reason = decision.get("reason", "")
        targets = decision.get("targets", [])
        lines.append(f"**Mode:** {mode}")
        if reason:
            lines.append(f"- Reason: {reason}")
        if targets:
            lines.append(f"- Targets: {len(targets)} finding(s)")
        lines.append("")

    # ── AI Findings That Triggered Rewrite ──────────────────────────
    if ai_findings:
        if no_text_change:
            lines.append("## AI Findings Reviewed")
            lines.append("")
            lines.append("These AI-related findings were reviewed, but they were not safe automatic rewrite targets.")
        else:
            lines.append("## AI Findings Targeted for Rewrite")
        lines.append("")
        lines.append("| # | Risk | Signal | Detail | Location | Suggestion |")
        lines.append("|--:|:----:|:------:|--------|----------|------------|")
        for i, f in enumerate(ai_findings, 1):
            title = f.get("title", f.get("finding_type", "unknown")).replace("|", "·")
            risk = (f.get("adjusted_risk") or f.get("risk_level") or "?").upper()
            detail = f.get("detail", "").replace("|", "·")
            sentence_id = f.get("sentence_id") or "—"
            recommendation = f.get("recommendation", "").replace("|", "·")
            lines.append(f"| {i} | {risk} | {title} | {detail} | {sentence_id} | {recommendation} |")
        lines.append("")

    # ── Rewrite Summary ─────────────────────────────────────────────
    lines.append("## Rewrite Summary")
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
        lines.append("These findings could not be auto-rewritten and need manual attention:")
        lines.append("")
        lines.append("| # | Risk | Finding | Reason | Evidence |")
        lines.append("|--:|:----:|---------|--------|----------|")
        for i, m in enumerate(manual, 1):
            ftype = m.get("finding_type", "?")
            risk = m.get("risk_level", "?").upper()
            reason = m.get("reason", "").replace("|", "·")
            evidence = m.get("evidence", "").replace("|", "·")
            lines.append(f"| {i} | {risk} | {ftype} | {reason} | {evidence} |")
        lines.append("")

    # Protected / Review-only
    protected = summary.get("protected_actions", [])
    if protected:
        lines.append(f"**Protected findings (skipped):** {len(protected)}")
        lines.append("")

    # ── Rewrite Attempts ────────────────────────────────────────────
    loop_history = summary.get("loop_history", [])
    if loop_history:
        lines.append("### Rewrite Attempts")
        lines.append("")
        lines.append("| Loop | Paragraph | Status | Note |")
        lines.append("|:----:|:---------:|:------:|------|")
        for entry in loop_history:
            note = entry.get("note", "")
            loop = entry.get("loop", "?")
            if note == "original":
                continue
            reverted = entry.get("reverted", False)
            status = "Reverted" if reverted else "Applied"
            para = entry.get("paragraph", "")
            note_clean = note.replace("|", "·")
            lines.append(f"| {loop} | {para} | {status} | {note_clean} |")
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
