"""Contract test: the three AI-risk BADGE tier producers emit the SAME
vocabulary (green/amber/orange/red), and that vocabulary is distinct from the
two other tier vocabularies in the codebase.

Regression guard for the CLAUDE.md documentation bug (H1): ``scan_jobs.tier``
stores the AI-risk BADGE tier (``green``/``amber``/``orange``/``red``, produced
by ``poc/report/builder.py`` ``ai_risk_badge["tier"]`` and ``.lower()``ed). It is
NOT the V7 detection-result tier (``clean``/``acceptable``/``concerning``/
``strong``, Turnitin-style) and NOT the findings-based ``overall_tier``
(``critical``/``high``/``medium``/``low``/``clean``). A consumer built against
the wrong vocabulary would silently mis-color / mis-classify every scan.

The three badge-tier producers, each of which must emit only
{green, amber, orange, red}:
  - Layer3Scorer._derive_ai_tier            (perplexity fallback, layer3_scoring.py)
  - deberta_signal._derive_ai_tier_deberta  (DeBERTa authoritative, deberta_signal.py)
  - detect_v7...compute_fused_authority     (V7 fused, pipeline_bridge.py)

allow-hardcode: the score sweeps and vocabulary sets below are TEST CONSTANTS
pinning the tier-vocabulary contract, not scoring/matching logic.
"""
from __future__ import annotations

import pytest

from poc.detect.layer3_scoring import Layer3Scorer
from poc.detect.deberta_signal import _derive_ai_tier_deberta
from detect_v7.pipeline_bridge import compute_fused_authority
from poc.detect.postprocess import PostProcessor
from poc.detect.base import Finding, DetectResult

# The exact set stored in scan_jobs.tier (ai_risk_badge["tier"].lower()).
BADGE_TIER_VOCAB = {"green", "amber", "orange", "red"}
# V7 detection-result tier (Turnitin-style) — a DIFFERENT surface.
V7_DETECTION_TIER_VOCAB = {"clean", "acceptable", "concerning", "strong"}
# Findings-based overall_tier (report/models.py Tier) — a THIRD surface.
FINDINGS_TIER_VOCAB = {"critical", "high", "medium", "low", "clean"}


def test_badge_vocab_is_distinct_from_v7_detection_result_vocab():
    # The badge tier (what scan_jobs.tier stores) must not collide with the V7
    # detection-result tier — the value CLAUDE.md previously (incorrectly)
    # documented as scan_jobs.tier. If these ever overlap, a consumer cannot
    # tell which surface produced the value.
    assert BADGE_TIER_VOCAB.isdisjoint(V7_DETECTION_TIER_VOCAB)


class TestLayer3BadgeTier:
    """The default (perplexity) path — the one with no dedicated test before."""

    def _tier(self, score: float) -> str:
        # Mirror builder.py's badge construction: str(layer3.tier.value).lower().
        return Layer3Scorer()._derive_ai_tier(score).value.lower()

    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.0, "green"),
            (0.31, "green"),
            (0.32, "amber"),   # boundary: >= 0.32
            (0.47, "amber"),
            (0.48, "orange"),  # boundary: >= 0.48
            (0.64, "orange"),
            (0.65, "red"),     # boundary: >= 0.65
            (1.0, "red"),
        ],
    )
    def test_cutoffs(self, score, expected):
        got = self._tier(score)
        assert got == expected
        assert got in BADGE_TIER_VOCAB

    def test_all_outputs_in_badge_vocab(self):
        for score in [i / 100 for i in range(0, 101)]:
            assert self._tier(score) in BADGE_TIER_VOCAB


class TestDebertaBadgeTier:
    def test_none_is_green(self):
        assert _derive_ai_tier_deberta(None) == "green"

    def test_all_outputs_in_badge_vocab(self):
        for score in [i / 100 for i in range(0, 101)]:
            assert _derive_ai_tier_deberta(score) in BADGE_TIER_VOCAB

    def test_monotonic(self):
        order = {v: i for i, v in enumerate(["green", "amber", "orange", "red"])}
        prev = -1
        for score in [i / 100 for i in range(0, 101)]:
            idx = order[_derive_ai_tier_deberta(score)]
            assert idx >= prev, f"tier not monotonic at score={score}"
            prev = idx


class TestV7FusedBadgeTier:
    def test_all_outputs_in_badge_vocab(self):
        for composite in range(0, 101, 10):
            for prop in [0.0, 0.25, 0.5, 0.75, 1.0]:
                result = compute_fused_authority(composite, prop)
                assert result["tier"] in BADGE_TIER_VOCAB

    def test_output_shape(self):
        result = compute_fused_authority(50.0, 0.5)
        assert set(result) == {"fused_score", "tier"}
        assert result["tier"] in BADGE_TIER_VOCAB


class TestRecomputeRisk:
    """PostProcessor._recompute_risk — the empty-findings branch is a CAP at 0.2
    (never raises risk); the blend branch CAN raise risk above the scanner's own
    overall_risk when a high-severity finding survives post-processing."""

    @staticmethod
    def _f(risk_level: str) -> Finding:
        return Finding(
            finding_type="t",
            risk_level=risk_level,
            evidence_strength="weak",
            detail="",
            evidence="",
            recommendation="",
            suggested_action_type="",
        )

    def test_empty_findings_caps_high_risk(self):
        # A scanner reporting 0.9 risk with ALL findings filtered drops to the 0.2 cap.
        assert PostProcessor._recompute_risk([], 0.9) == 0.2

    def test_empty_findings_never_raises_low_risk(self):
        # A scanner reporting 0.1 risk with all findings filtered stays at 0.1
        # (the cap only bounds from above — it is NOT a floor).
        assert PostProcessor._recompute_risk([], 0.1) == 0.1

    def test_blend_uses_top_severity(self):
        findings = [self._f("high"), self._f("low")]
        # 0.6*original + 0.4*max_level (high=0.7)
        assert PostProcessor._recompute_risk(findings, 0.5) == round(0.5 * 0.6 + 0.7 * 0.4, 4)

    def test_blend_can_raise_above_scanner_risk(self):
        # A low scanner risk with a surviving 'high' finding blends UP.
        assert PostProcessor._recompute_risk([self._f("high")], 0.1) > 0.1
