"""Read-time enrichment of a rewrite report's detect-scan badges.

The worker's rewrite re-scan can produce a "lite" badge (no grounding_diagnosis /
critical_thinking / submission_risk / policy_risk) — e.g. for reports written before
the worker enrichment shipped, or whenever the worker deploy lags. Without this, the
rewrite page falls back to the raw AI-likelihood %/Turnitin framing in the rewritten
column instead of the policy scores the submitted content shows.

The composers are PURE dict functions (app/_composers/, copied verbatim from
poc/detect/*). Running them when the API serves the report fixes the page for ALL
reports on load, independent of the worker. Additive only — never changes ai_likelihood
or tier. KEEP IN SYNC with the worker's enrichment (poc/rewrite_v6/production.py).
"""
from __future__ import annotations

from app._composers.grounding_diagnosis import diagnose_grounding_gap
from app._composers.critical_thinking import score_critical_thinking
from app._composers.submission_risk import score_submission_risk
from app._composers.policy_risk import score_policy_risk


def _enrich_badge(badge, axis_scores=None) -> None:
    if not isinstance(badge, dict) or badge.get("policy_risk"):
        return
    ai_components = badge.get("ai_components")
    if not isinstance(ai_components, dict):
        return  # no inputs available -> cannot enrich
    try:
        tc = badge.get("transformation_classification") or {}
        gd = diagnose_grounding_gap(
            ai_components=ai_components,
            writing_components=badge.get("writing_components") or {},
            transformation_features=tc.get("features") if isinstance(tc, dict) else None,
            scored_sentence_count=None,
        )
        ctc = score_critical_thinking(grounding_diagnosis=gd, scored_sentence_count=None)
        badge["grounding_diagnosis"] = gd
        badge["critical_thinking_control"] = ctc
        badge["submission_risk"] = score_submission_risk(
            ai_likelihood_score=badge.get("ai_likelihood_score"),
            ai_tier=badge.get("tier"),
            critical_thinking_control=ctc,
            axis_scores=axis_scores,
            scored_sentence_count=None,
        )
        badge["policy_risk"] = score_policy_risk(grounding_diagnosis=gd, critical_thinking_control=ctc)
    except Exception:
        pass


def enrich_rewrite_report(report_json):
    """Enrich the rewrite report's detect-scan badges in place; returns the report."""
    if not isinstance(report_json, dict):
        return report_json
    summary = report_json.get("summary")
    if not isinstance(summary, dict):
        return report_json
    for which in ("detect_scan_original_saved", "detect_scan_original", "detect_scan_rewritten"):
        sc = summary.get(which)
        if isinstance(sc, dict):
            _enrich_badge(sc.get("ai_risk_badge"), axis_scores=sc.get("axis_scores"))
    return report_json
