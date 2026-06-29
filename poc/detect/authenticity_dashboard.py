"""Additive Authenticity Dashboard composer.

Reads the existing ai_risk_badge (+ optional predictability for the AI confidence
interval) and emits a multi-dimension authenticity view. STRICTLY ADDITIVE: it never
feeds back into the tier, ai_likelihood_score, the external estimate, or any gate —
same contract as submission_risk.py. v1 ships 4 live dimensions + a weakest-link
Overall; Reasoning Consistency and Revision Evidence are phase-2 placeholders, and the
AI confidence interval is a clearly-tentative proxy (not a statistical interval).
"""
from __future__ import annotations

import statistics

MODEL_VERSION = "authenticity_dashboard_v1"

# Overall weights are overlap-aware: Learning Ownership is derivative of the Critical
# Thinking score (which Grounding feeds), so it is down-weighted to avoid double-counting.
WEIGHTS = {"grounding": 0.30, "citation_quality": 0.25, "ai_assistance": 0.30, "learning_ownership": 0.15}


def _clamp(v: float) -> float:
    return max(0.0, min(100.0, v))


def _tile(score, available: bool, caveat=None) -> dict:
    ok = available and isinstance(score, (int, float))
    return {"score": round(float(score), 1) if ok else None, "available": bool(ok), "caveat": caveat}


def compose_authenticity_dashboard(*, ai_risk_badge: dict, predictability: dict | None = None) -> dict:
    badge = ai_risk_badge or {}

    # Learning Ownership = Critical Thinking Control score (already 0-100, 100 = most ownership).
    ct_score = (badge.get("critical_thinking_control") or {}).get("score")
    learning_ownership = _tile(
        ct_score, ct_score is not None,
        caveat="How much you steer the thinking — derived from the critical-thinking signal.",
    )

    # Grounding = 100 - concrete_grounding gap; NULL (never a false 100) when no signal.
    gd = badge.get("grounding_diagnosis") or {}
    cg = (gd.get("buckets") or {}).get("concrete_grounding") or {}
    cg_avail = (cg.get("available") or 0) > 0 and isinstance(cg.get("score"), (int, float))
    grounding = _tile(
        _clamp(100.0 - cg["score"]) if cg_avail else None, cg_avail,
        caveat="Tentative — short submission." if gd.get("low_coverage") else None,
    )

    return {
        "version": MODEL_VERSION,
        "learning_ownership": learning_ownership,
        "grounding": grounding,
    }
