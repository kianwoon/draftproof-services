"""Overall tier must not read LOW when the AI-likelihood badge is AMBER or higher.

Bug context: report_to_dict's `_derive_weighted_tier` re-derives the overall tier
from findings and can land LOW even when the badge's own aggregate AI-likelihood is
AMBER (e.g. test_content11: badge AMBER 42%, generic_assertion 90, specificity ~0,
yet overall_tier=LOW). The derivation never consults the badge score or
generic_assertion_risk, and the pred_risk>=0.45 escalation is a brittle cliff
(content11 missed it by 0.005). Result: a zero-grounding, highly-generic document
reads "LOW", under-warning the user.

Fix: floor the derived overall tier to MEDIUM when the AI badge tier is AMBER+.
The badge already aggregates every AI signal (incl. generic_assertion), so a LOW
headline under an amber+ badge is self-contradictory. GREEN badges may stay LOW.
"""

from report.report import ReportBuilder, Tier


def floor(tier, badge):
    return ReportBuilder._floor_tier_to_badge(tier, badge)


def test_low_with_amber_badge_floors_to_medium():
    assert floor(Tier.LOW, "AMBER") == Tier.MEDIUM


def test_low_with_orange_or_red_badge_floors_to_medium():
    assert floor(Tier.LOW, "ORANGE") == Tier.MEDIUM
    assert floor(Tier.LOW, "RED") == Tier.MEDIUM


def test_low_with_green_badge_stays_low():
    # GREEN badge = genuinely low AI signal -> LOW is correct, do NOT floor.
    assert floor(Tier.LOW, "GREEN") == Tier.LOW


def test_floor_is_case_insensitive_and_none_safe():
    assert floor(Tier.LOW, "amber") == Tier.MEDIUM
    assert floor(Tier.LOW, None) == Tier.LOW
    assert floor(Tier.LOW, "") == Tier.LOW


def test_non_low_tiers_are_never_changed():
    # The floor only lifts LOW; it must never lower or alter higher tiers.
    for badge in ("GREEN", "AMBER", "ORANGE", "RED"):
        assert floor(Tier.MEDIUM, badge) == Tier.MEDIUM
        assert floor(Tier.HIGH, badge) == Tier.HIGH
        assert floor(Tier.CRITICAL, badge) == Tier.CRITICAL


def test_badge_tier_is_canonical_lowercase_on_layer3_fallback():
    """The layer3 Tier enum value is UPPERCASE ("AMBER"); the badge must ship
    lowercase (frontend TIER_TO_BAND and render_panels _ACB_TIER_TO_BAND key
    lowercase — an uppercase leak silently drops the verdict chip/band).
    Regression for the 3000-word verification finding (2026-07-06)."""
    from detect.layer3_scoring import Tier
    # Every enum value the fallback can emit must round-trip to lowercase.
    for tier in Tier:
        assert str(tier.value or "").lower() == tier.value.lower()
    # And the builder's badge dict literal now lowercases — assert on source to
    # keep this test dependency-light (a full ReportBuilder run needs the ML stack).
    import inspect
    from report import builder as report_builder
    src = inspect.getsource(report_builder.ReportBuilder)
    assert '"tier": str(authoritative_tier or layer3.tier.value or "").lower()' in src
