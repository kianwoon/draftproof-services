"""Rewrite verdict reframe — deterministic label engine (single source of truth).

Under the promoted fine-tune-v1 deep-scan detector, the rewrite writer's own output
(Cerebras gpt-oss-120b) scores fused ~65/orange, so any verdict keyed on
``final_risk < original_risk`` now stamps almost every rewrite "regressed / score_worse".
Per CLAUDE.md "Objective & Rewrite Philosophy" the rewrite is a *shown solution /
reviewable draft* — success = **content gaps surfaced & filled** (grounding, specifics),
NOT a lower detector score. Guards ANNOTATE, never suppress; the after-score stays visible.

This module holds the pure functions shared by every surface (production summary, PDF
render, web helpers passthrough, email). It reuses ONLY existing detector signals
(findings counts, grounding-quality risk, category component risks, bracket-grounding
anchors) — no new detector work.

Kill switch: ``DRAFTPROOF_REWRITE_VERDICT_REFRAME`` — default ON (owner approved
2026-07-15). Set to ``"0"``/``"false"`` (case-insensitive, whitespace-stripped) to revert
every surface to the legacy score-delta verdict byte-identically.
"""

from __future__ import annotations

import os
from typing import Any

ENV_VAR = "DRAFTPROOF_REWRITE_VERDICT_REFRAME"
# Default ON. Strict off-set mirrors detect_v7/pipeline_bridge.py's "1"/"true" contract,
# inverted because this switch defaults enabled.
_OFF = {"0", "false"}

# Deterministic label taxonomy (docs/plans/rewrite_verdict_reframe_scope.md §4).
LABEL_NO_USABLE_REWRITE = "no_usable_rewrite"
LABEL_GAPS_RESOLVED = "gaps_resolved"
LABEL_GAPS_PARTIALLY_RESOLVED = "gaps_partially_resolved"
LABEL_DRAFT_FOR_REVIEW = "draft_for_review"

# The AI-likelihood after-score stays VISIBLE on every surface; this note reframes it
# (annotate-never-suppress).
AI_LIKELIHOOD_NOTE = (
    "LLM-drafted text reads as AI — this stays high on the demo draft. "
    "Review the diff and make it yours; that is what lowers it."
)

# English PDF headlines per label (web headlines live in i18n report.js).
LABEL_TITLES = {
    LABEL_GAPS_RESOLVED: "Content gaps resolved",
    LABEL_GAPS_PARTIALLY_RESOLVED: "Gaps partially resolved",
    LABEL_DRAFT_FOR_REVIEW: "Draft ready to make yours",
    LABEL_NO_USABLE_REWRITE: "No usable rewrite — edit manually",
}

# Grounding-risk improvement threshold (percentage points; neg = better). Kept small so a
# genuine grounding drop registers while noise does not flip the headline.
_GROUNDING_IMPROVED_DELTA = -0.05
# Category component deltas are noisy on short paragraphs (scope §7 risk) — advisory only:
# a category is "worse" only past this positive margin, and never gates gaps_partially.
_CATEGORY_WORSE_DELTA = 0.05


def reframe_enabled() -> bool:
    """Kill switch, default ON. Only an explicit ``"0"``/``"false"`` disables."""
    raw = os.getenv(ENV_VAR, "1")
    return raw.strip().lower() not in _OFF


def _component_percent(report: Any, key: str) -> float | None:
    """Read a category component risk (generic_assertion_risk / citation_grounding_risk)
    from a scan report's ai_risk_badge.ai_components, scaled to percent. None when absent."""
    if not isinstance(report, dict):
        return None
    badge = report.get("ai_risk_badge")
    components = badge.get("ai_components") if isinstance(badge, dict) else None
    value = components.get(key) if isinstance(components, dict) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return round(number * 100 if abs(number) <= 1 else number, 3)


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return round(after - before, 3)


def build_gap_resolution(
    detect_scores: dict[str, Any],
    bracket_grounding_spans: Any,
    original_report: Any,
    rewritten_report: Any,
) -> dict[str, Any]:
    """Container of gap-resolution signals (scope §3). All from EXISTING detector data."""
    detect_scores = detect_scores if isinstance(detect_scores, dict) else {}
    findings_before = int(detect_scores.get("original_findings") or 0)
    findings_after = int(detect_scores.get("rewritten_findings") or 0)
    grounding_before = detect_scores.get("original_grounding_quality_risk")
    grounding_after = detect_scores.get("rewritten_grounding_quality_risk")
    generic_before = _component_percent(original_report, "generic_assertion_risk")
    generic_after = _component_percent(rewritten_report, "generic_assertion_risk")
    citation_before = _component_percent(original_report, "citation_grounding_risk")
    citation_after = _component_percent(rewritten_report, "citation_grounding_risk")
    anchors_added = sum(
        1
        for sp in (bracket_grounding_spans or [])
        if isinstance(sp, dict) and sp.get("kind") == "improved"
    )
    return {
        "findings_before": findings_before,
        "findings_after": findings_after,
        "findings_resolved": max(findings_before - findings_after, 0),
        "grounding_risk_before": grounding_before,
        "grounding_risk_after": grounding_after,
        "grounding_risk_delta": _delta(grounding_before, grounding_after),
        "generic_assertion_delta": _delta(generic_before, generic_after),
        "citation_grounding_delta": _delta(citation_before, citation_after),
        "anchors_added": anchors_added,
    }


def compute_verdict_label(
    *,
    changed: bool,
    no_text_change: bool,
    status: str,
    gap: dict[str, Any],
) -> str:
    """Deterministic label from GAP signals, not the AI score (scope §4)."""
    status = str(status or "")
    if no_text_change or not changed or status.startswith("original_preserved"):
        return LABEL_NO_USABLE_REWRITE

    gap = gap if isinstance(gap, dict) else {}
    findings_resolved = int(gap.get("findings_resolved") or 0) >= 1
    grounding_delta = gap.get("grounding_risk_delta")
    grounding_improved = grounding_delta is not None and grounding_delta <= _GROUNDING_IMPROVED_DELTA
    anchors_added = int(gap.get("anchors_added") or 0) >= 1

    generic_delta = gap.get("generic_assertion_delta") or 0.0
    citation_delta = gap.get("citation_grounding_delta") or 0.0
    category_worse = generic_delta > _CATEGORY_WORSE_DELTA or citation_delta > _CATEGORY_WORSE_DELTA

    if findings_resolved and grounding_improved and not category_worse:
        return LABEL_GAPS_RESOLVED
    if findings_resolved or grounding_improved or anchors_added:
        return LABEL_GAPS_PARTIALLY_RESOLVED
    return LABEL_DRAFT_FOR_REVIEW


def verdict_is_good(verdict_label: str) -> bool:
    """Green-hero predicate: a shown solution that measurably closed gaps."""
    return verdict_label in (LABEL_GAPS_RESOLVED, LABEL_GAPS_PARTIALLY_RESOLVED)


def gap_sub_line(gap: dict[str, Any]) -> str:
    """One-line gap summary: 'N findings resolved · grounding risk down X% · M specifics added'."""
    gap = gap if isinstance(gap, dict) else {}
    parts = [f"{int(gap.get('findings_resolved') or 0)} findings resolved"]
    delta = gap.get("grounding_risk_delta")
    if isinstance(delta, (int, float)) and delta < 0:
        parts.append(f"grounding risk down {abs(delta):.1f}%")
    anchors = int(gap.get("anchors_added") or 0)
    if anchors:
        parts.append(f"{anchors} specifics added")
    return " · ".join(parts)
