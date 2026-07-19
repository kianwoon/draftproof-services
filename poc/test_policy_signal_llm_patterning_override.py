"""Tests for the ``ai_components`` llm-patterning override in
``report/builder.py`` — the fix for the "policy risk reads a stale AI signal"
bug (a document the badge calls 67%/Critical showed policy risk 26-34/Moderate
because ``policy_risk`` read the pre-fusion ~5-9 value, not the fused 67).

Root cause: the override that sets ``ai_components``'s
``predictability``/``topk_pattern``/``topk_pattern_raw``/``topk_calibrated_risk``
keys (which feed ``grounding_diagnosis``'s ``llm_patterning`` bucket and through
it ``policy_risk.surface_ai_text_signal``) gated on the EARLY DeBERTa CPU pass
(``authoritative_score.available`` + ``_deb_score``), never on the already-computed,
more-authoritative V7 fused/deep-scan value.

The fix routes those 4 keys through
``report.policy_signal.policy_signal_pct_for_llm_patterning``: when V7
tier-authority fusion actually applied, it uses the FUSED
``authoritative_ai_likelihood`` (0-100, the same value the badge shows);
otherwise it falls back byte-identically to the existing early-DeBERTa logic
(``_deb_score < 0.20`` suppression on the 0-1 fraction scale).

Units note (scoring-critical): ``_deb_score`` is a 0-1 fraction; the fused
``authoritative_ai_likelihood`` is 0-100. ``LOW_AI_SIGNAL_SUPPRESSION_PCT``
(= 20.0) is the 0-100 equivalent of the legacy 0.20 fraction threshold.

Run from poc/:  python -m pytest test_policy_signal_llm_patterning_override.py -q
"""
from __future__ import annotations

import pytest

from report.policy_signal import (
    policy_signal_pct_for_llm_patterning as H,
    LOW_AI_SIGNAL_SUPPRESSION_PCT as _THRESH,
)
from detect.grounding_diagnosis import diagnose_grounding_gap
from detect.critical_thinking import score_critical_thinking
from detect.policy_risk import score_policy_risk


# ── (a) tier-authority applied + genuinely HIGH fused → HIGH (not 0) ──────────

def test_tier_authority_applied_high_fused_returns_high():
    # Fused 67 (0-100). Must surface 67, NOT the early-DeBERTa 5.
    assert H(
        deberta_available=True,
        deberta_score_fraction=0.05,
        deberta_pct=5.0,
        tier_authority_applied=True,
        fused_ai_likelihood_pct=67.0,
    ) == pytest.approx(67.0)


def test_tier_authority_applied_without_early_deberta_still_uses_fused():
    # Fusion can apply even when the early DeBERTa CPU pass was unavailable
    # (composite falls back to layer3). The fused value must still drive the keys.
    assert H(
        deberta_available=False,
        deberta_score_fraction=None,
        deberta_pct=None,
        tier_authority_applied=True,
        fused_ai_likelihood_pct=67.0,
    ) == pytest.approx(67.0)


def test_tier_authority_overrides_stale_low_deberta():
    # THE BUG: early DeBERTa ~5 (which alone would suppress to 0) but the fused
    # authority is 67 → the override must reflect 67, not the stale suppression.
    assert H(
        deberta_available=True,
        deberta_score_fraction=0.05,
        deberta_pct=5.0,
        tier_authority_applied=True,
        fused_ai_likelihood_pct=67.0,
    ) == pytest.approx(67.0)


# ── (b) tier-authority applied + genuinely LOW fused → suppress to 0.0 ────────

def test_tier_authority_applied_low_fused_suppresses_to_zero():
    # Low true signal (12 < 20) must still suppress to 0.0 — even though the
    # early DeBERTa read was high (0.9) — because the FUSED value is now the
    # authority.
    assert H(
        deberta_available=True,
        deberta_score_fraction=0.90,
        deberta_pct=90.0,
        tier_authority_applied=True,
        fused_ai_likelihood_pct=12.0,
    ) == 0.0


def test_fused_at_threshold_is_not_suppressed():
    # Boundary: exactly at the threshold (20.0) is NOT below it → not suppressed.
    assert H(
        deberta_available=False,
        deberta_score_fraction=None,
        deberta_pct=None,
        tier_authority_applied=True,
        fused_ai_likelihood_pct=_THRESH,
    ) == pytest.approx(_THRESH)


def test_fused_just_below_threshold_is_suppressed():
    assert H(
        deberta_available=False,
        deberta_score_fraction=None,
        deberta_pct=None,
        tier_authority_applied=True,
        fused_ai_likelihood_pct=_THRESH - 0.01,
    ) == 0.0


# ── (c) tier-authority NOT applied → EXISTING early-DeBERTa logic, unchanged ──

def test_no_tier_authority_high_deberta_uses_deb_pct():
    # Legacy path: _deb_score 0.67 >= 0.20 → _deb_pct 67.0.
    assert H(
        deberta_available=True,
        deberta_score_fraction=0.67,
        deberta_pct=67.0,
        tier_authority_applied=False,
        fused_ai_likelihood_pct=None,
    ) == pytest.approx(67.0)


def test_no_tier_authority_low_deberta_suppresses():
    # Legacy path: _deb_score 0.05 < 0.20 → 0.0.
    assert H(
        deberta_available=True,
        deberta_score_fraction=0.05,
        deberta_pct=5.0,
        tier_authority_applied=False,
        fused_ai_likelihood_pct=None,
    ) == 0.0


def test_no_deberta_no_tier_authority_returns_none():
    # Neither authoritative path active → perplexity keys must be left untouched.
    assert H(
        deberta_available=False,
        deberta_score_fraction=None,
        deberta_pct=None,
        tier_authority_applied=False,
        fused_ai_likelihood_pct=None,
    ) is None


def test_legacy_deberta_threshold_is_byte_identical_to_020():
    # The legacy fraction threshold must remain exactly 0.20 (byte-identical
    # flag-off behavior).
    assert H(deberta_available=True, deberta_score_fraction=0.20, deberta_pct=20.0,
             tier_authority_applied=False, fused_ai_likelihood_pct=None) == pytest.approx(20.0)
    assert H(deberta_available=True, deberta_score_fraction=0.1999, deberta_pct=19.99,
             tier_authority_applied=False, fused_ai_likelihood_pct=None) == 0.0


def test_tier_authority_flag_on_but_not_applied_falls_back_to_deberta():
    # Flag ON but fusion did not apply (below-floor / unavailable / error →
    # tier_authority_applied=False). Must behave exactly like the legacy path.
    assert H(
        deberta_available=True,
        deberta_score_fraction=0.05,
        deberta_pct=5.0,
        tier_authority_applied=False,
        fused_ai_likelihood_pct=None,
    ) == 0.0


# ── (d) blast-radius regression: fused signal lifts llm_patterning + policy ───

def _chain(ai_pct: float) -> tuple[float, float]:
    """Run the real pure diagnosis chain the bug flows through and return
    (llm_patterning bucket score, policy_risk.surface_ai_text_signal)."""
    ai_components = {"predictability": ai_pct, "topk_pattern": ai_pct}
    diag = diagnose_grounding_gap(ai_components=ai_components, scored_sentence_count=30)
    ct = score_critical_thinking(grounding_diagnosis=diag, scored_sentence_count=30)
    pol = score_policy_risk(grounding_diagnosis=diag, critical_thinking_control=ct)
    return (
        diag["buckets"]["llm_patterning"]["score"],
        pol["base_signals"]["surface_ai_text_signal"],
    )


def test_blast_radius_fused_lifts_llm_patterning_and_policy():
    # BEFORE the fix (real bug): tier-authority applied with fused 67, but the
    # 4 keys kept the stale pre-fusion ~9 perplexity value → llm_patterning /
    # policy stayed near ~9.
    stale_llm, stale_pol = _chain(9.0)

    # AFTER the fix: the override routes the FUSED 67 into those keys.
    fixed_pct = H(
        deberta_available=True,
        deberta_score_fraction=0.05,   # early DeBERTa still reads ~5 (stale)
        deberta_pct=5.0,
        tier_authority_applied=True,
        fused_ai_likelihood_pct=67.0,  # the true fused authority
    )
    assert fixed_pct == pytest.approx(67.0)
    fixed_llm, fixed_pol = _chain(fixed_pct)

    # The llm_patterning bucket (and therefore policy_risk's surface_ai_text_signal)
    # must be MEANINGFULLY higher than the stale ~9 — not stuck near it.
    assert fixed_llm == pytest.approx(67.0)
    assert fixed_llm > stale_llm + 40.0
    assert fixed_pol > stale_pol + 40.0
