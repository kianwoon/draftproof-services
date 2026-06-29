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

_BAND_FROM_TIER = {"GREEN": "Low", "AMBER": "Moderate", "ORANGE": "High", "RED": "High"}
_CI_WIDEN = {"high": 1.0, "medium": 1.5, "low": 2.0}


def _clamp(v: float) -> float:
    return max(0.0, min(100.0, v))


def _tile(score, available: bool, caveat=None) -> dict:
    ok = available and isinstance(score, (int, float))
    return {"score": round(float(score), 1) if ok else None, "available": bool(ok), "caveat": caveat}


def _ai_ci(ai_score, confidence, predictability) -> dict | None:
    """Tentative proxy interval on the authenticity scale (100 - ai_score). Half-width from
    per-sentence predictability spread, widened by categorical confidence. NOT a statistical
    interval over the composite ai_likelihood — labeled tentative; the true CI is phase 2."""
    if not isinstance(ai_score, (int, float)):
        return None
    auth = _clamp(100.0 - ai_score)
    risks = [s.get("predictability_risk") for s in ((predictability or {}).get("all_sentences") or [])
             if isinstance(s.get("predictability_risk"), (int, float))]
    spread = statistics.pstdev(risks) * 100.0 if len(risks) >= 2 else 12.0
    half = min(40.0, spread * _CI_WIDEN.get(str(confidence or "").lower(), 1.5))
    return {"low": round(_clamp(auth - half), 1), "high": round(_clamp(auth + half), 1), "tentative": True}


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

    # Citation Quality = 100 - mean(citation_weakness_risk, source_grounding_risk).
    wc = badge.get("writing_components") or {}
    cite_parts = [wc.get("citation_weakness_risk"), wc.get("source_grounding_risk")]
    cite_parts = [p for p in cite_parts if isinstance(p, (int, float))]
    citation_quality = _tile(
        _clamp(100.0 - sum(cite_parts) / len(cite_parts)) if cite_parts else None,
        bool(cite_parts),
        caveat=None,
    )

    # AI Assistance: band reuses the tier (can never contradict the headline); score = 100 - likelihood.
    ai_score = badge.get("ai_likelihood_score")
    ai_auth = _clamp(100.0 - ai_score) if isinstance(ai_score, (int, float)) else None
    ai_assistance = {
        "band": _BAND_FROM_TIER.get(str(badge.get("tier") or "").upper()),
        "score": round(ai_auth, 1) if ai_auth is not None else None,
        "ci": _ai_ci(ai_score, badge.get("confidence"), predictability),
    }

    return {
        "version": MODEL_VERSION,
        "learning_ownership": learning_ownership,
        "grounding": grounding,
        "citation_quality": citation_quality,
        "ai_assistance": ai_assistance,
    }
