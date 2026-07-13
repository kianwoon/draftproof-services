"""Headline-confidence annotation for the AI-risk badge.

Problem (live scan d449aca9, 2026-07-13): the badge headline can read "0% /
green" while the SAME report page shows contrary evidence — a second-opinion
detector tile at 33% amber, a raw perplexity tier of HIGH, a low-confidence
verdict flag, or a deep scan below its reliability floor. Shipping the bare
percent next to that evidence is a contradictory display that erodes trust.

Fix philosophy ("guards ANNOTATE, never SUPPRESS"): this module never changes
the tier or the score and never hides the contrary evidence. It only computes
an ADDITIVE badge field::

    headline_confidence = {"level": "low", "reasons": [...], "detail": {...}}

that display surfaces (web, PDF, email) use to qualify the headline percent.
The field is emitted ONLY when the badge tier is low-risk (a clean-looking
headline) AND at least one concrete divergence reason fires — clean documents
with no contradiction get no field at all (additive, byte-identical otherwise).

Callsite: poc/report/report.py, AFTER _sync_deberta_headline_from_heatmap has
rebuilt the ai_signal_deberta tile from the deep-scan heatmap — computing this
earlier (in builder.py) would compare against the stale pre-sync tile.
"""
from __future__ import annotations

from typing import Any, Optional

# Badge tiers that read as "clean" to the user. The badge tier vocabulary is
# green/amber/orange/red on the DeBERTa-authoritative / V7-fused paths and
# low/medium/high on the legacy perplexity fallback (builder.py L1515); "clean"
# is included for the DB-tier vocabulary defensively. Only these tiers can
# carry a headline_confidence annotation — the whole point is a clean-looking
# headline contradicted by other evidence on the page.
_LOW_RISK_TIERS = {"green", "clean", "low"}

# Second-opinion tile bands that constitute contrary evidence. "insufficient"
# deliberately excluded: below the tile's own reliability floor it offers no
# verdict, so it cannot contradict the headline.
_TILE_DIVERGENT_BANDS = {"amber", "orange", "red"}

# Raw (pre-override) document tiers that contradict a clean headline. The raw
# tier vocabulary is low/medium/high (layer3_scoring.Tier); only the top band
# fires, "medium" is not treated as a contradiction.
_RAW_DIVERGENT_TIERS = {"high"}

_BREAKDOWN_BELOW_FLOOR_FLAG = "deep_scan_below_reliability_floor"
_FUSED_SIGNAL_SOURCE = "v7_fused"


def compute_headline_confidence(
    badge: Optional[dict], raw_tier: Optional[str]
) -> Optional[dict[str, Any]]:
    """Return the headline_confidence dict, or None when no annotation applies.

    Parameters
    ----------
    badge: the FINAL ai_risk_badge dict (after the ai_signal_deberta tile has
        been synced from the deep-scan heatmap). Fail-open on any malformed
        shape — this must never break a report build.
    raw_tier: the report's raw_overall_tier (pre-override perplexity tier,
        low/medium/high), or None when unavailable.
    """
    if not isinstance(badge, dict):
        return None
    tier = str(badge.get("tier") or "").strip().lower()
    if tier not in _LOW_RISK_TIERS:
        return None

    reasons: list[str] = []
    detail: dict[str, Any] = {}

    tile = badge.get("ai_signal_deberta")
    if (
        isinstance(tile, dict)
        and tile.get("available") is True
        and str(tile.get("band") or "").lower() in _TILE_DIVERGENT_BANDS
    ):
        reasons.append("second_opinion_divergence")
        pct = tile.get("signal_pct")
        if isinstance(pct, (int, float)) and not isinstance(pct, bool):
            detail["second_opinion_pct"] = pct

    if isinstance(raw_tier, str) and raw_tier.strip().lower() in _RAW_DIVERGENT_TIERS:
        reasons.append("raw_tier_divergence")
        detail["raw_tier"] = raw_tier.strip().lower()

    if badge.get("verdict_low_confidence") is True:
        reasons.append("verdict_low_confidence")

    breakdown = badge.get("authorship_breakdown")
    if (
        badge.get("signal_source") == _FUSED_SIGNAL_SOURCE
        and isinstance(breakdown, dict)
        and _BREAKDOWN_BELOW_FLOOR_FLAG in (breakdown.get("uncertainty_flags") or [])
    ):
        reasons.append("deep_scan_below_floor")

    if not reasons:
        return None
    return {"level": "low", "reasons": reasons, "detail": detail}
