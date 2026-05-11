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


def _authorship_label(report: dict) -> str:
    badge = _badge(report)
    rating = badge.get("authorship_rating") or {}
    return (
        rating.get("short_label")
        or rating.get("label")
        or badge.get("authorship_rating_label")
        or ""
    )


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
                actions.append("Add a real process detail from the author's experience.")
            else:
                actions.append("Narrow generic assertions to the exact submitted context.")
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


def _guided_revision_checklist(mitigation: dict, limit: int = 5) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for driver in mitigation.get("component_drivers") or []:
        bucket = driver.get("bucket")
        component = str(driver.get("component", ""))
        if bucket not in {"needs_source_or_example", "structure_guidance"}:
            continue
        guidance = _driver_guidance(driver)
        if component == "unsupported_claim_risk":
            where = "Confident claims without visible support"
            add = "A source, concrete example, or softer claim wording"
            retry = "After claims are supported or narrowed"
        elif component == "source_grounding_risk":
            where = "Claims that cite theory but do not explain the source link"
            add = "Author/source name plus a sentence explaining how it supports the claim"
            retry = "After source-to-claim links are explicit"
        elif component == "citation_weakness_risk":
            where = "Paragraphs with weak or missing in-text citation linkage"
            add = "The missing citation or a clearer attribution phrase"
            retry = "After citation linkage is repaired"
        elif component in {"broad_claim_risk", "generic_assertion_risk"}:
            where = "Broad statements that could fit many drafts"
            add = "The exact context, method, participant group, or observation"
            retry = "After broad claims are narrowed to the draft context"
        elif component == "lived_detail_risk":
            where = "General claims about learning or practice"
            add = "A real process detail from the author's class, salon, lesson, or workflow"
            retry = "After lived/process detail is added"
        else:
            where = guidance["issue"]
            add = guidance["action"]
            retry = "After this manual revision is done"
        rows.append({
            "priority": guidance["signal"],
            "where": where,
            "add": add,
            "retry": retry,
        })
        if len(rows) >= limit:
            break
    return rows


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
        action = "Add a real project, participant, task, or workflow detail supplied by the author."
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
        quality_worse = new_wq > orig_wq + 0.05
        ai_first = summary.get("ai_first_mitigation") or {}
        ai_first_kept = bool(ai_first.get("kept")) and ai_improved
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
        score_improved = ai_improved or quality_improved or findings_improved
        score_worse = ai_worse or quality_worse
        improved_with_review = (
            not original_preserved
            and score_improved
            and not score_worse
            and not findings_worse
            and not severity_worse
            and (ai_improved or quality_improved)
        )
        mixed_result = (
            not original_preserved
            and not improved_with_review
            and score_improved
            and (score_worse or findings_worse or severity_worse)
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
        outcome = str(summary.get("outcome") or "")
        if ai_first_kept:
            result_label = "AI Mitigated"
        elif outcome == "ai_mitigated":
            result_label = "AI Mitigated"
        elif outcome == "partially_ai_mitigated":
            result_label = "Partially AI Mitigated"
        elif outcome == "cleanup_improved":
            result_label = "Cleanup Improved"
        elif outcome == "topk_blocked":
            result_label = "Top-k Blocked"

        lines.append("### Result")
        lines.append("")
        lines.append(f"**{result_label}**")
        lines.append("")
        if outcome == "cleanup_improved":
            lines.append("Review burden reduced, but AI-footprint signals are still high. Do not treat this as detector-safe mitigation.")
        elif outcome == "partially_ai_mitigated":
            lines.append("AI-footprint signals reduced, but the result is still not guaranteed to pass external detectors.")
        elif outcome == "topk_blocked":
            lines.append("Top-k predictability remains above the safe mark, so this rewrite is not detector-safe mitigation yet.")
        elif outcome == "ai_mitigated" or ai_first_kept:
            lines.append(
                "AI likelihood improved enough to keep the rewrite. Writing-quality or lower-severity changes are follow-up work."
            )
        elif original_preserved and rollback and attempted_scan:
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
        lines.append(f"| **AI Likelihood** | `{orig_ai:.2f}%` | `{new_ai:.2f}%` | `{ai_delta:+.2f}%` |")
        detect_scores = summary.get("detect_scores") or {}
        turnitin_before = summary.get("turnitin_like_ai_score_before", detect_scores.get("turnitin_like_ai_score_before"))
        turnitin_after = summary.get("turnitin_like_ai_score_after", detect_scores.get("turnitin_like_ai_score_after"))
        turnitin_drop = summary.get("turnitin_like_ai_score_drop", detect_scores.get("turnitin_like_ai_score_drop"))
        turnitin_target = summary.get("turnitin_like_target_score", detect_scores.get("turnitin_like_target_score"))
        if isinstance(turnitin_before, (int, float)) and isinstance(turnitin_after, (int, float)):
            target_note = f" target < `{float(turnitin_target):.0f}%`" if isinstance(turnitin_target, (int, float)) else ""
            lines.append(
                f"| **Turnitin-like AI Score** | `{float(turnitin_before):.2f}%` | "
                f"`{float(turnitin_after):.2f}%`{target_note} | `{float(turnitin_drop or 0.0):+.2f}% drop` |"
            )
        footprint_gate = summary.get("ai_footprint_gate") or {}
        footprint_drops = footprint_gate.get("drops") or {}
        footprint_before = (footprint_gate.get("before") or {}).get("external_ai_flag_risk")
        footprint_after = (footprint_gate.get("after") or {}).get("external_ai_flag_risk")
        if isinstance(footprint_before, (int, float)) and isinstance(footprint_after, (int, float)):
            lines.append(
                f"| **External AI Flag Proxy** | `{float(footprint_before):.2f}%` | "
                f"`{float(footprint_after):.2f}%` | `{float(footprint_drops.get('external_ai_flag_risk') or 0.0):+.2f}% drop` |"
            )
        footprint_before_authorship = (footprint_gate.get("before") or {}).get("authorship_footprint") or {}
        footprint_after_authorship = (footprint_gate.get("after") or {}).get("authorship_footprint") or {}
        topk_raw_before = footprint_before_authorship.get("topk_pattern_raw", footprint_before_authorship.get("topk_pattern"))
        topk_raw_after = footprint_after_authorship.get("topk_pattern_raw", footprint_after_authorship.get("topk_pattern"))
        if isinstance(topk_raw_before, (int, float)) and isinstance(topk_raw_after, (int, float)):
            lines.append(
                f"| **Raw Top-k Predictability** | `{float(topk_raw_before):.2f}%` | "
                f"`{float(topk_raw_after):.2f}%` | `{float(footprint_drops.get('topk_pattern_raw') or 0.0):+.2f}% drop` |"
            )
        topk_before = footprint_before_authorship.get("topk_calibrated_risk")
        topk_after = footprint_after_authorship.get("topk_calibrated_risk")
        if isinstance(topk_before, (int, float)) and isinstance(topk_after, (int, float)):
            lines.append(
                f"| **Calibrated Top-k Risk** | `{float(topk_before):.2f}%` | "
                f"`{float(topk_after):.2f}%` | `{float(footprint_drops.get('topk_calibrated_risk') or 0.0):+.2f}% drop` |"
            )
        orig_authorship = _authorship_label(orig_scan)
        new_authorship = _authorship_label(new_scan)
        if orig_authorship or new_authorship:
            lines.append(f"| **Authorship Rating** | {orig_authorship or '-'} | {new_authorship or '-'} | - |")
        lines.append(f"| **Writing Quality Risk** | `{orig_wq:.2f}%` | `{new_wq:.2f}%` | `{wq_delta:+.2f}%` |")
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

        formula_gap = summary.get("formula_gap_contract") or {}
        if isinstance(formula_gap, dict) and formula_gap.get("version"):
            portfolio = summary.get("formula_portfolio_plan") or formula_gap.get("formula_portfolio_plan") or {}
            positive_burden = portfolio.get("positive_ai_burden") if isinstance(portfolio, dict) else {}
            anchor_suppression = portfolio.get("human_anchor_suppression") if isinstance(portfolio, dict) else {}
            lines.append("**Formula Gap To <20:**")
            lines.append("")
            if isinstance(positive_burden, dict) and isinstance(anchor_suppression, dict):
                lines.append(
                    f"Positive AI burden `{float(positive_burden.get('before') or 0.0):.3f}` → "
                    f"`{float(positive_burden.get('after') or 0.0):.3f}`; "
                    f"Human-anchor suppression `{float(anchor_suppression.get('before') or 0.0):.3f}` → "
                    f"`{float(anchor_suppression.get('after') or 0.0):.3f}`."
                )
                lines.append("")
            lines.append("| Driver | Weighted Before | Weighted After | Drop |")
            lines.append("|--------|----------------:|---------------:|-----:|")
            drops = formula_gap.get("weighted_driver_drops") or {}
            for driver in (
                "ai_likelihood",
                "topk_calibrated_risk",
                "semantic_uniformity",
                "rewrite_smoothness",
                "patchwork_expansion",
                "signal_agreement",
                "human_anchor_suppression",
            ):
                row = drops.get(driver) if isinstance(drops, dict) else None
                if not isinstance(row, dict):
                    continue
                before = row.get("before")
                after = row.get("after")
                drop = row.get("drop", row.get("gain"))
                if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                    label = driver.replace("_", " ").title()
                    lines.append(
                        f"| {label} | `{float(before):.3f}` | `{float(after):.3f}` | `{float(drop or 0.0):+.3f}` |"
                    )
            if not formula_gap.get("target_met"):
                lines.append("")
                lines.append(
                    f"Remaining formula gap: `{float(formula_gap.get('remaining_formula_gap') or 0.0):.3f}`. "
                    f"{summary.get('why_not_below_20') or formula_gap.get('why_not_below_20') or ''}"
                )
            priority_plan = (
                (portfolio.get("driver_priorities") if isinstance(portfolio, dict) else None)
                or formula_gap.get("driver_priority_plan")
                or summary.get("driver_priority_plan")
                or []
            )
            if priority_plan:
                lines.append("")
                lines.append("Top control priorities:")
                for row in priority_plan[:3]:
                    if not isinstance(row, dict):
                        continue
                    lines.append(
                        f"- `{row.get('driver')}` via `{row.get('strategy_family')}` "
                        f"(expected net gain `{row.get('expected_net_gain')}`, "
                        f"headroom `{row.get('feasible_weighted_headroom')}`, "
                        f"safe observed drop `{row.get('observed_safe_drop', 0)}`)"
                    )
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

    ceiling = summary.get("mitigation_ceiling") or {}
    if ceiling:
        lines.append("## Mitigation Ceiling")
        lines.append("")
        safe = ceiling.get("safe_auto_result") or {}
        frontier = ceiling.get("candidate_frontier") or {}
        evidence_gap = ceiling.get("author_evidence_gap") or {}
        primary = str(ceiling.get("primary_blocker") or "unknown").replace("_", " ")
        lines.append(f"**Primary blocker:** {primary}.")
        lines.append("")
        lines.append("| Signal | Current Safe Result | Frontier |")
        lines.append("|--------|--------------------:|---------:|")
        lines.append(
            f"| Human Contribution | `{safe.get('human_contribution', '-')}` | "
            f"best safe `{frontier.get('best_safe_human', '-')}` / best seen `{frontier.get('best_seen_human', '-')}` |"
        )
        lines.append(
            f"| AI Score | `{safe.get('ai_score', '-')}` | best seen `{frontier.get('best_seen_ai', '-')}` |"
        )
        lines.append(
            f"| AI Authorship | `{safe.get('ai_authorship', '-')}` | drop `{safe.get('ai_authorship_drop', '-')}` |"
        )
        lines.append(
            f"| AI Transformation | `{safe.get('ai_transformation', '-')}` | drop `{safe.get('ai_transformation_drop', '-')}` |"
        )
        lines.append(
            f"| Candidate Pool | {frontier.get('safe_candidates', 0)} safe / "
            f"{frontier.get('scanned_candidates', 0)} scanned | "
            f"{frontier.get('blocked_candidates', 0)} blocked |"
        )
        lines.append("")
        if evidence_gap.get("enabled"):
            estimate = evidence_gap.get("estimated_human_after_completion") or {}
            lines.append(
                f"Author evidence is still the Human Contribution ceiling: "
                f"{evidence_gap.get('slot_count', 0)} real-author slot(s) are needed. "
                f"Heuristic Human range after completion: `{estimate.get('low', '-')}`-`{estimate.get('high', '-')}`."
            )
            lines.append("")
        actions = ceiling.get("recommended_next_actions") or []
        if actions:
            lines.append("**Next actions:**")
            lines.append("")
            for action in actions[:4]:
                lines.append(f"- {str(action).replace('|', '·')}")
            lines.append("")

    intake = summary.get("author_evidence_intake") or {}
    if intake.get("questions"):
        lines.append("## Author Evidence Intake")
        lines.append("")
        lines.append(
            "Use these prompts to collect the real author-owned details needed before another mitigation pass. "
            "DraftProof can place confirmed answers, but it should not invent these anchors."
        )
        lines.append("")
        lines.append("| Anchor | Paragraph | What To Ask | Answer Type |")
        lines.append("|--------|-----------|-------------|-------------|")
        for question in (intake.get("questions") or [])[:8]:
            qid = str(question.get("id") or "").replace("|", "·")
            paragraph = question.get("paragraph_index")
            text = str(question.get("question") or "").replace("|", "·")
            answer_type = str(question.get("answer_type") or "").replace("_", " ")
            lines.append(f"| {qid} | {paragraph} | {text} | {answer_type} |")
        lines.append("")
        policy = intake.get("close_gap_policy") or []
        if policy:
            lines.append("**LLM close-gap policy:**")
            lines.append("")
            for item in policy[:4]:
                lines.append(f"- {str(item).replace('|', '·')}")
            lines.append("")

    discovery = summary.get("author_context_discovery") or {}
    if discovery.get("context_cards"):
        lines.append("## Author Context Discovery")
        lines.append("")
        lines.append(
            "Use this LLM-assisted intake to close the author-owned context gap. "
            "The LLM may ask and shape answers, but confirmed user answers are required before generation."
        )
        lines.append("")
        lines.append("| Anchor | Gap | Safe Answer Shape |")
        lines.append("|--------|-----|-------------------|")
        for card in (discovery.get("context_cards") or [])[:8]:
            anchor = str(card.get("anchor_id") or "").replace("|", "·")
            gap = str(card.get("llm_follow_up_question") or card.get("gap") or "").replace("|", "·")
            shape = str(card.get("safe_answer_shape") or "").replace("|", "·")
            lines.append(f"| {anchor} | {gap} | {shape} |")
        lines.append("")
        handoff = discovery.get("handoff_env") or {}
        if handoff:
            lines.append(
                f"Confirmed answers feed back through `{handoff.get('json')}` "
                f"or `{handoff.get('file')}`."
            )
            lines.append("")
        gates = discovery.get("success_gate") or []
        if gates:
            lines.append("**Discovery success gate:**")
            lines.append("")
            for gate in gates[:4]:
                lines.append(f"- {str(gate).replace('|', '·')}")
            lines.append("")

    source_search = summary.get("source_grounding_search") or {}
    if source_search.get("enabled"):
        lines.append("## Source Grounding Search")
        lines.append("")
        status = str(source_search.get("status") or "ready").replace("_", " ")
        provider = str(source_search.get("provider") or "tavily")
        lines.append(
            f"**Status:** {status}. Provider: `{provider}`. "
            "These are public-source candidates for grounding quality; they are not author-owned context."
        )
        lines.append("")
        targets = source_search.get("claim_targets") or []
        results_by_claim = {
            item.get("claim_id"): item
            for item in (source_search.get("results") or [])
            if isinstance(item, dict)
        }
        if targets:
            lines.append("| Claim | Confidence | Query | Candidate Sources |")
            lines.append("|-------|------------|-------|-------------------|")
            for target in targets[:8]:
                claim = str(target.get("claim") or "").replace("|", "·")
                query = str(target.get("query") or "").replace("|", "·")
                result = results_by_claim.get(target.get("id")) or {}
                confidence = str(result.get("source_confidence") or "-").replace("_", " ")
                source_links = []
                for source in (result.get("sources") or [])[:3]:
                    title = str(source.get("title") or source.get("url") or "source").replace("|", "·")
                    url = str(source.get("url") or "").strip()
                    if url:
                        source_links.append(f"[{title}]({url})")
                    elif title:
                        source_links.append(title)
                sources = "<br>".join(source_links) if source_links else "-"
                lines.append(f"| {claim[:220]} | `{confidence}` | `{query[:180]}` | {sources} |")
            lines.append("")
        policy = source_search.get("policy") or []
        if policy:
            lines.append("**Search policy:**")
            lines.append("")
            for item in policy[:4]:
                lines.append(f"- {str(item).replace('|', '·')}")
            lines.append("")

    integration = summary.get("author_evidence_integration") or {}
    if integration.get("enabled"):
        lines.append("## Author Evidence Integration")
        lines.append("")
        status = str(integration.get("status") or "awaiting_user_answers").replace("_", " ")
        lines.append(
            f"**Status:** {status}. "
            f"Answers received: `{integration.get('answer_count', 0)}`; "
            f"valid answers: `{integration.get('accepted_answers', 0)}`; "
            f"applied: `{integration.get('applied_answers', 0)}`."
        )
        lines.append("")
        candidates = integration.get("candidates") or []
        if candidates:
            lines.append("| Anchor | Status | Human | AI Authorship | Findings | Reason |")
            lines.append("|--------|--------|------:|--------------:|---------:|--------|")
            for item in candidates[:8]:
                anchor = str(item.get("anchor_id") or "").replace("|", "·")
                item_status = str(item.get("status") or "").replace("_", " ")
                human = item.get("human_contribution", "-")
                authorship = item.get("ai_authorship", "-")
                findings = item.get("findings", "-")
                reason = str(item.get("reason") or "").replace("|", "·")
                lines.append(
                    f"| {anchor} | {item_status} | `{human}` | `{authorship}` | `{findings}` | {reason} |"
                )
            lines.append("")

    if mitigation:
        score_targets = mitigation.get("score_mitigation_targets") or []
        if score_targets:
            lines.append("## Risk Score Mitigation Targets")
            lines.append("")
            lines.append(
                "These are the highest-value score levers to work on before retrying automatic rewrite. The target is to move each driver out of the strongest-risk band without inventing evidence."
            )
            lines.append("")
            lines.append("| Priority | Score Driver | Current | Target | Reduction Needed | Best Mitigation |")
            lines.append("|----------|--------------|--------:|-------:|-----------------:|-----------------|")
            for item in score_targets[:6]:
                component = str(item.get("component", "")).replace("_risk", "").replace("_", " ").title()
                priority = str(item.get("priority", "")).title()
                current = item.get("current_score", 0)
                target = item.get("target_score", 0)
                reduction = item.get("reduction_needed", 0)
                action = str(item.get("action", "")).replace("|", "·")
                lines.append(
                    f"| {priority} | {component} | `{float(current):.1f}%` | "
                    f"`{float(target):.1f}%` | `{float(reduction):.1f}%` | {action} |"
                )
            lines.append("")

        risk_actions = mitigation.get("risk_mitigation_actions") or []
        if risk_actions:
            lines.append("## Risk Mitigation Actions")
            lines.append("")
            lines.append(
                "Use these actions when automatic rewrite cannot safely create the missing evidence or context. They are written to reduce the next scan's strongest score drivers without inventing facts."
            )
            lines.append("")
            lines.append("| Priority | Action | Score Driver | User Input Needed | Safe Edit Pattern |")
            lines.append("|----------|--------|--------------|-------------------|-------------------|")
            for item in risk_actions[:6]:
                priority = str(item.get("priority", "")).title()
                title = str(item.get("title", "")).replace("|", "·")
                component = str(item.get("component", "")).replace("_risk", "").replace("_", " ").title()
                current = item.get("current_score", 0)
                target = item.get("target_score", 0)
                needed = str(item.get("user_input_needed", "")).replace("|", "·")
                pattern = _highlight_placeholders(
                    str(item.get("safe_edit_pattern", "")).replace("|", "·")
                )
                lines.append(
                    f"| {priority} | {title} | {component} `{float(current):.1f}% -> {float(target):.1f}%` | "
                    f"{needed} | {pattern} |"
                )
            lines.append("")

        marked_suggestions = mitigation.get("marked_content_suggestions") or []
        if marked_suggestions:
            lines.append("## Suggested Additions For Review")
            lines.append("")
            lines.append(
                "These are not kept automatically in the final output. Bracketed text marks new content that DraftProof is proposing as a structure only; replace it with verified source, example, or author detail before using."
            )
            lines.append("")
            lines.append("| Priority | Where | Target Text | Suggested Addition | Why It Helps | User Review |")
            lines.append("|----------|-------|-------------|--------------------|--------------|-------------|")
            for item in marked_suggestions[:6]:
                priority = str(item.get("priority", "")).title()
                where = str(item.get("where", "")).replace("|", "·")
                target_text = str(item.get("target_text") or item.get("evidence") or "").replace("|", "·")
                target_text = html.escape(target_text)
                suggestion = _highlight_placeholders(
                    str(item.get("suggested_addition", "")).replace("|", "·")
                )
                why = str(item.get("why_it_helps", "")).replace("|", "·")
                note = str(item.get("user_note", "")).replace("|", "·")
                lines.append(
                    f"| {priority} | {where} | {target_text} | {suggestion} | {why} | {note} |"
                )
            lines.append("")

        checklist = _guided_revision_checklist(mitigation)
        if final_output_preserved and checklist:
            lines.append("## Guided Revision Checklist")
            lines.append("")
            lines.append(
                "DraftProof preserved the original because automatic sentence edits cannot safely supply missing evidence or source context. Use this checklist before retrying rewrite."
            )
            lines.append("")
            lines.append("| Priority Signal | Where To Look | Add Or Change | Retry When |")
            lines.append("|-----------------|---------------|---------------|------------|")
            for row in checklist:
                lines.append(
                    f"| {row['priority'].replace('|', '·')} | "
                    f"{row['where'].replace('|', '·')} | "
                    f"{row['add'].replace('|', '·')} | "
                    f"{row['retry'].replace('|', '·')} |"
                )
            lines.append("")

    marked_mitigation = summary.get("marked_mitigation_rewrite") or {}
    completion = summary.get("author_evidence_completion") or {}
    if completion.get("draft_text"):
        lines.append("## Author Evidence Completion Draft")
        lines.append("")
        estimate = completion.get("estimated_human_after_completion") or {}
        current_human = completion.get("current_human_contribution")
        target_human = completion.get("target_human_contribution")
        if estimate:
            lines.append(
                f"Current Human Contribution is `{current_human}%`; target is `{target_human}%`. "
                f"If the bracketed slots are completed with real evidence, the estimated Human range is "
                f"`{estimate.get('low')}%-{estimate.get('high')}%`. This is a heuristic estimate; rescan after completion."
            )
        else:
            lines.append(
                "This draft marks where the author should add real source evidence, concrete context, or defensible reasoning. It is not an accepted final rewrite."
            )
        lines.append("")
        lines.append("```text")
        lines.append(str(completion.get("draft_text", "")).strip())
        lines.append("```")
        lines.append("")
        slots = completion.get("slots") or []
        if slots:
            lines.append("| Slot | Paragraph | Role | What To Supply |")
            lines.append("|------|-----------|------|----------------|")
            for slot in slots[:12]:
                instruction = str(slot.get("instruction") or "").replace("|", "·")
                role = str(slot.get("paragraph_role") or "").replace("_", " ")
                paragraph_index = slot.get("paragraph_index")
                lines.append(
                    f"| {slot.get('slot', '')} | {paragraph_index} | {role} | {instruction} |"
                )
            lines.append("")

    if marked_mitigation.get("draft_text"):
        lines.append("## Marked Mitigation Draft")
        lines.append("")
        lines.append(
            "This is a marked scaffold, not the accepted final output. Replace every bracketed marker with verified author evidence, source detail, or concrete context before using it."
        )
        lines.append("")
        lines.append("```text")
        lines.append(str(marked_mitigation.get("draft_text", "")).strip())
        lines.append("```")
        lines.append("")
        changes = marked_mitigation.get("changes") or []
        if changes:
            lines.append("| # | Signal | What To Supply |")
            lines.append("|---|--------|----------------|")
            for item in changes[:8]:
                component = str(item.get("component", "")).replace("_", " ")
                needed = str(item.get("user_input_needed") or "Verified author/source detail").replace("|", "·")
                lines.append(f"| {item.get('index', '')} | {component} | {needed} |")
            lines.append("")

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

    attempted_changes = [
        sc for sc in (summary.get("attempted_sentence_comparison") or [])
        if sc.get("orig_sentence", "") != sc.get("new_sentence", "")
    ]
    if rollback and attempted_changes:
        lines.append("## Attempted Rewrite")
        lines.append("")
        lines.append("These edits passed local rewrite checks, but the final full scan regressed, so DraftProof preserved the original final output.")
        lines.append("")
        lines.append("| # | Original | Attempted Output |")
        lines.append("|--:|----------|------------------|")
        for i, sc in enumerate(attempted_changes[:12], 1):
            orig_text = sc.get("orig_sentence", "").replace("\n", " ").replace("|", "·")
            new_text = sc.get("new_sentence", "").replace("\n", " ").replace("|", "·")
            lines.append(f"| {i} | {orig_text} | {new_text} |")
        lines.append("")

    manual_suggestions = summary.get("manual_suggestions") or []
    if manual_suggestions:
        lines.append("## Manual Suggestions")
        lines.append("")
        lines.append("These candidates were not automatically kept. Review them manually before using them.")
        lines.append("")
        lines.append("| # | Finding | Reason | Original | Suggested |")
        lines.append("|--:|---------|--------|----------|-----------|")
        for i, item in enumerate(manual_suggestions[:16], 1):
            finding = str(item.get("finding_type", "?")).replace("|", "·")
            reason = str(item.get("rejection_reason", "")).replace("|", "·")
            orig_text = str(item.get("original_sentence", "")).replace("\n", " ").replace("|", "·")
            suggestion = str(item.get("suggested_sentence", "")).replace("\n", " ").replace("|", "·")
            lines.append(f"| {i} | {finding} | {reason} | {orig_text} | {suggestion} |")
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
