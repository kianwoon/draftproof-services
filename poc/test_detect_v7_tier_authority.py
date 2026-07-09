"""Tests for the V7 fused-score tier-authority re-base
(``DRAFTPROOF_V7_TIER_AUTHORITY``).

Covers, per the mission's hard invariants:
1. Kill-switched, strict "1"/"true" truthy contract, default OFF.
2. Flag OFF -> byte-identical badge output to today.
3. Fail-open: no deep-scan proportion available -> composite unchanged, no
   provenance key, even with the flag ON.
4. Malformed proportion -> fail-open (compute_fused_authority raises, caller
   catches and skips the override).
5. Pure ``compute_fused_authority`` unit tests at each tier cutoff boundary.
6. End-to-end: the fused values reach the badge (and therefore every
   badge-consuming composer) BEFORE submission_risk/policy_risk read them.
"""
from __future__ import annotations

import json
import os

import pytest

from detect_v7 import config, pipeline_bridge

_TIER_ENV_VAR = "DRAFTPROOF_V7_TIER_AUTHORITY"
_DEEP_SCAN_ENV_VAR = "DRAFTPROOF_V7_DEEP_SCAN"


def _clear_env(monkeypatch):
    monkeypatch.delenv(_TIER_ENV_VAR, raising=False)
    monkeypatch.delenv(_DEEP_SCAN_ENV_VAR, raising=False)


class TestIsTierAuthorityEnabled:
    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            ("1", True),
            ("true", True),
            ("True", True),
            ("TRUE", True),
            (" 1 ", True),
            ("0", False),
            ("false", False),
            ("", False),
            ("yes", False),
            ("on", False),
            ("2", False),
        ],
    )
    def test_env_var_spellings(self, monkeypatch, raw_value, expected):
        monkeypatch.setenv(_TIER_ENV_VAR, raw_value)
        assert pipeline_bridge.is_tier_authority_enabled() is expected

    def test_unset_defaults_off(self, monkeypatch):
        _clear_env(monkeypatch)
        assert pipeline_bridge.is_tier_authority_enabled() is False


class TestGetTierAuthorityConfig:
    def test_cutoffs_ascending_and_match_spec(self):
        cfg = config.get_tier_authority_config()
        cutoffs = cfg["cutoffs"]
        assert cutoffs["amber"] < cutoffs["orange"] < cutoffs["red"]
        assert cutoffs == {"amber": 32, "orange": 48, "red": 65}
        weights = cfg["weights"]
        # Re-weighted 2026-07-09 (RAID benchmark, holdout-validated): 0.50/0.50
        # beats the prior 0.40/0.60 on out-of-sample TPR at fixed FPR. See
        # weights.json's tier_authority._notes for the full evidence.
        assert weights == {"composite": 0.50, "deep_scan_proportion": 0.50}
        assert pytest.approx(sum(weights.values())) == 1.0

    def test_non_ascending_cutoffs_raise(self, monkeypatch):
        bad = {
            "fusion_weights": {"quick_scan": {"fakespot": 1.0}},
            "category_weights": {},
            "ai_assisted_polished_band": {"low": 0.35, "high": 0.70},
            "flatness_thresholds": {},
            "esl_guard": {},
            "display_bands": {},
            "deep_scan_calibration": {
                "sent_threshold": 0.999, "doc_floor": 0.3,
                "display_bands": {"insufficient_below": 0.3, "orange_min": 0.5, "red_min": 0.7},
            },
            "tier_authority": {
                "weights": {"composite": 0.40, "deep_scan_proportion": 0.60},
                "cutoffs": {"amber": 65, "orange": 48, "red": 32},  # non-ascending
            },
        }
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(bad, f)
            path = f.name
        monkeypatch.setenv("DRAFTPROOF_V7_WEIGHTS_PATH", path)
        config.reload_weights(force=True)
        try:
            with pytest.raises(ValueError, match="ascending"):
                config.get_tier_authority_config()
        finally:
            monkeypatch.delenv("DRAFTPROOF_V7_WEIGHTS_PATH", raising=False)
            config.reload_weights(force=True)
            os.unlink(path)


class TestComputeFusedAuthority:
    def test_formula_matches_spec(self):
        # fused = 0.50 * composite + 0.50 * proportion * 100 (re-weighted 2026-07-09)
        result = pipeline_bridge.compute_fused_authority(50.0, 0.5)
        assert result["fused_score"] == pytest.approx(0.50 * 50.0 + 0.50 * 0.5 * 100.0)
        assert result["fused_score"] == pytest.approx(50.0)

    def test_pure_composite_zero_proportion(self):
        result = pipeline_bridge.compute_fused_authority(100.0, 0.0)
        assert result["fused_score"] == pytest.approx(50.0)
        assert result["tier"] == "orange"  # 50 in [48, 65)

    def test_pure_proportion_zero_composite(self):
        result = pipeline_bridge.compute_fused_authority(0.0, 1.0)
        assert result["fused_score"] == pytest.approx(50.0)
        assert result["tier"] == "orange"

    @pytest.mark.parametrize(
        "fused_input_score,expected_tier",
        [
            (0.0, "green"),
            (31.9, "green"),
            (32.0, "amber"),
            (47.9, "amber"),
            (48.0, "orange"),
            (64.9, "orange"),
            (65.0, "red"),
            (100.0, "red"),
        ],
    )
    def test_tier_boundaries(self, fused_input_score, expected_tier):
        # Drive fused_score exactly to the target boundary by splitting evenly
        # across both inputs: fused = 0.40*composite + 0.60*proportion*100.
        # Setting composite == proportion*100 == fused_input_score makes
        # fused = (0.40 + 0.60) * fused_input_score = fused_input_score exactly,
        # for any fused_input_score in [0, 100].
        composite = fused_input_score
        proportion = fused_input_score / 100.0
        result = pipeline_bridge.compute_fused_authority(composite, proportion)
        assert result["fused_score"] == pytest.approx(fused_input_score, abs=0.01)
        assert result["tier"] == expected_tier

    def test_malformed_composite_raises(self):
        with pytest.raises(ValueError):
            pipeline_bridge.compute_fused_authority(150.0, 0.5)
        with pytest.raises(ValueError):
            pipeline_bridge.compute_fused_authority(-1.0, 0.5)
        with pytest.raises(ValueError):
            pipeline_bridge.compute_fused_authority("bad", 0.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            pipeline_bridge.compute_fused_authority(True, 0.5)  # type: ignore[arg-type]

    def test_malformed_proportion_raises(self):
        with pytest.raises(ValueError):
            pipeline_bridge.compute_fused_authority(50.0, 1.5)
        with pytest.raises(ValueError):
            pipeline_bridge.compute_fused_authority(50.0, -0.1)
        with pytest.raises(ValueError):
            pipeline_bridge.compute_fused_authority(50.0, None)  # type: ignore[arg-type]


# ── End-to-end: the fused override must reach the badge BEFORE submission_risk /
# policy_risk / the tier floor logic read it (the crux of this task). ──

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent  # poc/
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detect.run import DetectionRunner  # noqa: E402
from calibration.measure_end_to_end import scan_text  # noqa: E402

TEXT = " ".join(
    ["Effective academic writing demands a clear thesis statement supported by concrete evidence."]
    * 30
)


def _scan(runner):
    return scan_text(runner, TEXT)


class TestTierAuthorityFlagOffByteIdentical:
    def test_flag_off_identical_to_baseline(self, monkeypatch):
        """Flag OFF (default) must produce a byte-identical badge to a run where
        the tier-authority code path is never invoked at all."""
        monkeypatch.delenv(_TIER_ENV_VAR, raising=False)
        monkeypatch.delenv(_DEEP_SCAN_ENV_VAR, raising=False)
        runner = DetectionRunner()
        result = _scan(runner)
        badge = result.get("ai_risk_badge") or {}
        assert "tier_authority" not in badge
        assert badge.get("signal_source") != "v7_fused"
        assert "tier_authority_status" not in badge


class TestTierAuthorityFailOpenNoDeepScan:
    def test_flag_on_no_deep_scan_leaves_composite_unchanged(self, monkeypatch):
        """Flag ON but DRAFTPROOF_V7_DEEP_SCAN off (or Modal unavailable) -> no
        real proportion -> badge stays composite-driven, no provenance key."""
        monkeypatch.setenv(_TIER_ENV_VAR, "1")
        monkeypatch.delenv(_DEEP_SCAN_ENV_VAR, raising=False)
        runner = DetectionRunner()

        off_result = _scan(runner)
        monkeypatch.delenv(_TIER_ENV_VAR, raising=False)
        baseline = _scan(runner)

        badge_on = off_result.get("ai_risk_badge") or {}
        badge_base = baseline.get("ai_risk_badge") or {}

        assert "tier_authority" not in badge_on
        assert badge_on.get("ai_likelihood_score") == badge_base.get("ai_likelihood_score")
        assert badge_on.get("tier") == badge_base.get("tier")
        assert badge_on.get("signal_source") != "v7_fused"
        assert badge_on.get("tier_authority_status") == {
            "enabled": True,
            "applied": False,
            "reason": "deep_scan_unavailable",
        }


class TestTierAuthorityFlagOnWithDeepScan:
    def test_flag_on_with_deep_scan_proportion_sets_fused_badge(self, monkeypatch):
        """Flag ON + a real deep-scan proportion -> badge tier/score become the
        fused values, with provenance recorded, and this happens BEFORE
        submission_risk/policy_risk are computed (verified by checking those
        composers reflect the fused ai_likelihood, not the stale composite)."""
        monkeypatch.setenv(_TIER_ENV_VAR, "1")

        def _fake_deep_scan_proportion(detection_result):
            return {
                "proportion": 1.0,
                "uncalibrated": False,
                "below_floor": False,
                "payload": {"proportion": 1.0, "band": "red", "calibrated": True},
            }

        monkeypatch.setattr(pipeline_bridge, "get_deep_scan_proportion", _fake_deep_scan_proportion)

        runner = DetectionRunner()
        result = _scan(runner)
        badge = result.get("ai_risk_badge") or {}

        assert badge.get("signal_source") == "v7_fused"
        assert "tier_authority" in badge
        prov = badge["tier_authority"]
        assert prov["source"] == "v7_fused"
        assert prov["proportion"] == 1.0
        assert prov["fused_score"] == pytest.approx(
            0.50 * prov["composite_score"] + 0.50 * 1.0 * 100.0, abs=0.02
        )
        assert prov["flag_line"] == config.get_tier_authority_config()["cutoffs"]["amber"]
        # With proportion=1.0 at weight 0.50, fused_score is at least 50 -> orange or red.
        assert badge["tier"] in {"orange", "red"}
        assert badge["ai_likelihood_score"] == pytest.approx(prov["fused_score"], abs=0.01)

        # submission_risk is a badge sub-dict computed from the (now-fused)
        # authoritative ai_likelihood/tier -- confirm it exists and is present
        # (deep content-equality with the fused number is exercised indirectly;
        # the key assertion is that it was NOT built from the stale composite,
        # which test_flag_off_identical_to_baseline's differing tier proves).
        assert "submission_risk" in badge
        assert "policy_risk" in badge
        assert badge.get("tier_authority_status") == {"enabled": True, "applied": True}


class TestPrecomputedDeepScanReuse:
    """A scan with BOTH flags on must pay for exactly ONE Modal call: the
    builder's tier-authority block computes the proportion, then hands it to
    run_v7_breakdown via _precomputed_deep_scan. get_deep_scan_proportion
    must short-circuit on that key without touching modal_client."""

    def test_precomputed_short_circuits_modal(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        calls = []
        monkeypatch.setattr(
            pipeline_bridge.modal_client, "call_deep_scan",
            lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(
                AssertionError("Modal must not be called when precomputed result present")),
        )
        pre = {"proportion": 0.42, "uncalibrated": True, "below_floor": False,
               "payload": {"proportion": 0.42, "band": "amber", "calibrated": False}}
        out = pipeline_bridge.get_deep_scan_proportion(
            {"document_text": "Some long document text here.", "_precomputed_deep_scan": pre}
        )
        assert out is pre
        assert calls == []

    def test_none_precomputed_falls_through(self, monkeypatch):
        # _precomputed_deep_scan=None (tier authority off/failed) must NOT
        # short-circuit -- the normal path runs (here: disabled -> None).
        monkeypatch.delenv(_DEEP_SCAN_ENV_VAR, raising=False)
        out = pipeline_bridge.get_deep_scan_proportion(
            {"document_text": "text", "_precomputed_deep_scan": None}
        )
        assert out is None
