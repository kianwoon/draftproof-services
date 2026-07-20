"""Tests for detect_v7.pipeline_bridge — the kill-switched, fail-open V7
Authorship Clarity Breakdown call-site wiring.

The fabricated ``detection_result`` shapes in these tests mirror the REAL
shape traced from ``poc/report/builder.py`` (~L1273-1412) and
``poc/detect/run.py::_collect_criterion_scores``:
- ``ai_components`` / ``writing_components``: flat dicts, 0-100 scale.
- ``criterion_scores``: dict of objects/dicts with a ``.value`` (0-1) field,
  keyed by the exact criteria/*.py ``name=`` literals (e.g.
  ``low_burstiness``, ``low_specificity``, ``style_shift``,
  ``repetitive_structure``, ``citation_grounding_gap``).
- ``ai_likelihood_score``: 0-100 scale composite (builder.py's authoritative
  score).
"""
from __future__ import annotations

import importlib

import pytest

from detect_v7 import config, modal_client, pipeline_bridge

_ENV_VAR = "DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN"
_DEEP_SCAN_ENV_VAR = "DRAFTPROOF_V7_DEEP_SCAN"


def _clear_env(monkeypatch):
    monkeypatch.delenv(_ENV_VAR, raising=False)
    monkeypatch.delenv(_DEEP_SCAN_ENV_VAR, raising=False)


def _realistic_detection_result(ai_likelihood_score: float = 62.5, text: str | None = None) -> dict:
    """A dict shaped like builder.py's ``ai_risk_badge`` at the V7 call site."""
    result = {
        "ai_likelihood_score": ai_likelihood_score,
        "ai_components": {
            "generic_assertion_risk": 55.0,
            "repeated_sentence_structure_risk": 40.0,
            "topk_pattern": 61.0,
        },
        "writing_components": {
            "source_grounding_risk": 48.0,
            "lived_detail_risk": 52.0,
            "citation_weakness_risk": 30.0,
            "formulaic_conclusion_risk": 20.0,
            "signpost_paragraph_risk": 25.0,
        },
        "criterion_scores": {
            "low_burstiness": {"name": "low_burstiness", "value": 0.55},
            "low_surprisal": {"name": "low_surprisal", "value": 0.60},
            "low_specificity": {"name": "low_specificity", "value": 0.50},
            "style_shift": {"name": "style_shift", "value": 0.30},
            "repetitive_structure": {"name": "repetitive_structure", "value": 0.35},
            "citation_grounding_gap": {"name": "citation_grounding_gap", "value": 0.40},
        },
        "transformation_classification": {
            "features": {"human_anchor_score": 0.45},
        },
        "qualifying_word_count": 320,
    }
    if text is not None:
        result["text"] = text
    return result


def _expected_calibrated_mean_proportion(sentence_scores: list[float]) -> float:
    """Doc-level deep-scan ``proportion`` under the shipped
    ``representation="calibrated_mean"`` contract (added 2026-07-14, config
    ``deep_scan_calibration``): ``clip((mean(sentence_scores)-lo)/(hi-lo),0,1)``
    — NOT the legacy flagged-fraction. Anchors are read from config so the
    expectation tracks any re-anchoring (no hardcoded checkpoint constants)."""
    cal = config.get_deep_scan_calibration()
    lo, hi = cal["mean_anchor_lo"], cal["mean_anchor_hi"]
    mean = sum(sentence_scores) / len(sentence_scores)
    return max(0.0, min(1.0, (mean - lo) / (hi - lo)))


def _inject_sentence_scores(monkeypatch, sentence_scores: list[float], *, calibrated: bool = True):
    """Drive the deep-scan DISPLAY logic (bands / per-paragraph rows / floor)
    with EXACT per-sentence scores.

    Since commit 18b0a7c5 the Modal call receives overlapping 3-sentence
    WINDOWS (deberta_windowing.build_windows) and pipeline_bridge aggregates
    them back to per-sentence scores; a single window score therefore cannot
    express an arbitrary per-sentence pattern (overlapping windows average).
    These tests are about the proportion/band/per-paragraph math GIVEN
    per-sentence scores, so we inject the aggregated scores directly — the
    windowing + aggregation transport is covered by deberta_windowing's own
    tests. ``call_deep_scan`` still returns a window-length score list so the
    length-match guard in get_deep_scan_proportion passes.

    ``sentence_scores`` MUST match ``split_sentences`` on the document text
    (production zips sentences with the aggregated scores)."""
    import detect.deberta_windowing as _dw  # same module pipeline_bridge imports at call time

    monkeypatch.setattr(
        modal_client,
        "call_deep_scan",
        lambda chunks, timeout_s=60.0: {
            "available": True,
            "calibrated": calibrated,
            "chunk_scores": [1.0] * len(chunks),
            "document_score": 1.0,
        },
    )
    monkeypatch.setattr(
        _dw,
        "aggregate",
        lambda sentences, windows, window_probs, size=3, step=1: {
            "sentence_scores": list(sentence_scores),
            "document_score": (sum(sentence_scores) / len(sentence_scores)) if sentence_scores else 0.0,
        },
    )


class TestIsV7Enabled:
    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            ("1", True),
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("0", False),
            ("false", False),
            ("False", False),
            ("", False),
            ("yes", False),
            ("on", False),
            ("2", False),
        ],
    )
    def test_env_var_spellings(self, monkeypatch, raw_value, expected):
        monkeypatch.setenv(_ENV_VAR, raw_value)
        assert pipeline_bridge.is_v7_enabled() is expected

    def test_unset_defaults_off(self, monkeypatch):
        _clear_env(monkeypatch)
        assert pipeline_bridge.is_v7_enabled() is False


class TestRunV7BreakdownDisabled:
    def test_disabled_by_default_returns_none_no_work_done(self, monkeypatch):
        _clear_env(monkeypatch)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result())
        assert result is None

    def test_explicit_zero_returns_none(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "0")
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result())
        assert result is None


class TestRunV7BreakdownEnabled:
    def test_enabled_with_realistic_shape_produces_valid_breakdown(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result())

        assert result is not None
        assert result["schema_version"] == "v7_phase1a"
        assert result["display_mode"] == "bands"
        assert set(result["document_breakdown_raw"].keys()) == {
            "student_owned",
            "ai_assisted_polished",
            "ai_paraphrased",
            "ai_generated_like",
        }
        total = sum(result["document_breakdown_raw"].values())
        assert total == pytest.approx(1.0, abs=1e-6)
        assert result["paragraph_count"] == 1
        assert result["primary_category"] in {
            "student_owned",
            "ai_assisted_polished",
            "ai_paraphrased",
            "ai_generated_like",
        }
        assert "disclaimer" in result
        assert "esl_guard_unavailable" in result["uncertainty_flags"]
        assert result["granularity"] == "document"

    def test_enabled_true_spelling_also_works(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "true")
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result())
        assert result is not None

    def test_raw_signals_carry_fused_detector_score(self, monkeypatch):
        """run_v7_breakdown must thread the fused calibrated detector score
        into the raw_signals dict handed to the adapter, under
        'calibrated_detector_score', equal to the fusion output — this is what
        enables the detector-gated specificity split (2026-07-08)."""
        monkeypatch.setenv(_ENV_VAR, "1")
        from detect_v7 import detector_fusion, signal_adapter

        seen: dict = {}
        real_adapt = signal_adapter.adapt_paragraph_signals

        def _spy(raw_signals):
            seen["raw"] = raw_signals
            return real_adapt(raw_signals)

        monkeypatch.setattr(pipeline_bridge.signal_adapter, "adapt_paragraph_signals", _spy)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result())
        assert result is not None
        assert "calibrated_detector_score" in seen["raw"]
        # Quick-scan path: composite alone -> fusion == the composite input.
        expected, _ = detector_fusion.compute_calibrated_detector_score(
            {"composite": pipeline_bridge._extract_calibrated_score(_realistic_detection_result())}
        )
        assert seen["raw"]["calibrated_detector_score"] == pytest.approx(expected)
        # And the split actually materialized "ok" on this fixture.
        adapted = real_adapt(seen["raw"])
        assert adapted["signal_status"]["specificity_student_evidence"] == "ok"
        assert adapted["signal_status"]["specificity_ai_evidence"] == "ok"

    def test_object_with_attributes_instead_of_dict(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")

        class FakeDetectionResult:
            ai_likelihood_score = 40.0
            ai_components = {"generic_assertion_risk": 30.0}
            writing_components = {"source_grounding_risk": 20.0}
            criterion_scores = {
                "low_burstiness": {"value": 0.4},
                "low_specificity": {"value": 0.3},
            }
            transformation_classification = {"features": {"human_anchor_score": 0.6}}
            qualifying_word_count = 150

        result = pipeline_bridge.run_v7_breakdown(FakeDetectionResult())
        assert result is not None
        assert result["paragraph_count"] == 1


class TestRunV7BreakdownFailsSafe:
    def test_missing_ai_likelihood_score_returns_none(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        detection_result = {"ai_components": {}, "writing_components": {}}
        result = pipeline_bridge.run_v7_breakdown(detection_result)
        assert result is None

    def test_missing_components_returns_none(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        detection_result = {"ai_likelihood_score": 50.0}
        result = pipeline_bridge.run_v7_breakdown(detection_result)
        assert result is None

    def test_malformed_input_caught_returns_none(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")

        class Exploding:
            @property
            def ai_likelihood_score(self):
                raise RuntimeError("boom")

        result = pipeline_bridge.run_v7_breakdown(Exploding())
        assert result is None

    def test_none_input_returns_none(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        result = pipeline_bridge.run_v7_breakdown(None)
        assert result is None

    def test_empty_dict_returns_none(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        result = pipeline_bridge.run_v7_breakdown({})
        assert result is None


class TestTierConsistencyGuard:
    """Tier-consistency display guard: a red/orange-tier document must never
    display an unchallenged ``student_owned`` primary category. Fires
    post-composition in the bridge (has both the badge ``tier`` and the
    composed breakdown), annotates via the existing mixed_signals mechanism,
    never mutates shares/primary_category itself (guards annotate, never
    suppress).
    """

    @staticmethod
    def _patch_fixed_breakdown(monkeypatch, primary_category: str, raw_shares=None):
        fixed = {
            "schema_version": "v7_phase1a",
            "document_breakdown_raw": raw_shares
            if raw_shares is not None
            else {
                "student_owned": 0.7,
                "ai_assisted_polished": 0.1,
                "ai_paraphrased": 0.1,
                "ai_generated_like": 0.1,
            },
            "document_breakdown_bands": {},
            "primary_category": primary_category,
            "primary_category_reliable": True,
            "confidence": "high",
            "paragraph_count": 1,
            "degraded_paragraph_count": 0,
            "display_mode": "bands",
            "degraded_display": False,
            "uncertainty_flags": [],
            "disclaimer": "x",
        }
        monkeypatch.setattr(
            pipeline_bridge.breakdown_composer,
            "compose_authorship_breakdown",
            lambda *a, **k: dict(fixed),
        )
        return fixed

    def test_guard_fires_strong_tier_student_owned(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        self._patch_fixed_breakdown(monkeypatch, "student_owned")
        detection_result = _realistic_detection_result()
        detection_result["tier"] = "strong"

        result = pipeline_bridge.run_v7_breakdown(detection_result)

        assert result is not None
        assert result["presentation"] == "mixed_signals"
        assert result["primary_category_reliable"] is False
        assert "tier_category_contradiction" in result["uncertainty_flags"]
        # shares/primary_category themselves are UNCHANGED (annotate, never suppress)
        assert result["primary_category"] == "student_owned"
        assert result["document_breakdown_raw"]["student_owned"] == 0.7

    def test_guard_fires_concerning_tier_student_owned(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        self._patch_fixed_breakdown(monkeypatch, "student_owned")
        detection_result = _realistic_detection_result()
        detection_result["tier"] = "concerning"

        result = pipeline_bridge.run_v7_breakdown(detection_result)

        assert result["presentation"] == "mixed_signals"
        assert result["primary_category_reliable"] is False
        assert "tier_category_contradiction" in result["uncertainty_flags"]

    def test_guard_does_not_fire_clean_tier(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        self._patch_fixed_breakdown(monkeypatch, "student_owned")
        detection_result = _realistic_detection_result()
        detection_result["tier"] = "clean"

        result = pipeline_bridge.run_v7_breakdown(detection_result)

        assert result.get("presentation") != "mixed_signals"
        assert result["primary_category_reliable"] is True
        assert "tier_category_contradiction" not in result["uncertainty_flags"]

    def test_guard_does_not_fire_acceptable_tier(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        self._patch_fixed_breakdown(monkeypatch, "student_owned")
        detection_result = _realistic_detection_result()
        detection_result["tier"] = "acceptable"

        result = pipeline_bridge.run_v7_breakdown(detection_result)

        assert result["primary_category_reliable"] is True
        assert "tier_category_contradiction" not in result["uncertainty_flags"]

    def test_guard_does_not_fire_non_student_owned_primary(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        # Neither the four-way primary NOR the merged display_primary is
        # student_owned (AI-dominant shares) -> guard must stay quiet.
        self._patch_fixed_breakdown(
            monkeypatch,
            "ai_generated_like",
            raw_shares={
                "student_owned": 0.1,
                "ai_assisted_polished": 0.1,
                "ai_paraphrased": 0.2,
                "ai_generated_like": 0.6,
            },
        )
        detection_result = _realistic_detection_result()
        detection_result["tier"] = "strong"

        result = pipeline_bridge.run_v7_breakdown(detection_result)

        assert result["display_primary"] == "ai_transformed"
        assert result["primary_category_reliable"] is True
        assert "tier_category_contradiction" not in result["uncertainty_flags"]

    def test_guard_silently_skips_when_tier_missing(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        self._patch_fixed_breakdown(monkeypatch, "student_owned")
        detection_result = _realistic_detection_result()
        # no "tier" key at all — fail-open: guard must not raise or fire.

        result = pipeline_bridge.run_v7_breakdown(detection_result)

        assert result is not None
        assert result["primary_category_reliable"] is True
        assert "tier_category_contradiction" not in result["uncertainty_flags"]


class TestThreeWayDisplayFallback:
    """V8 three-way display fallback: additive display_* fields that merge the
    two indistinguishable AI indicators (ai_paraphrased + ai_generated_like)
    into ai_transformed. The four-way fields stay byte-identical.
    """

    @staticmethod
    def _patch_breakdown(monkeypatch, raw_shares, primary_category):
        fixed = {
            "schema_version": "v7_phase1a",
            "document_breakdown_raw": dict(raw_shares),
            "document_breakdown_bands": {},
            "primary_category": primary_category,
            "primary_category_reliable": True,
            "confidence": "high",
            "paragraph_count": 1,
            "degraded_paragraph_count": 0,
            "display_mode": "bands",
            "degraded_display": False,
            "uncertainty_flags": [],
            "disclaimer": "x",
        }
        monkeypatch.setattr(
            pipeline_bridge.breakdown_composer,
            "compose_authorship_breakdown",
            lambda *a, **k: dict(fixed),
        )
        return fixed

    def test_display_shares_merge_and_passthrough(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        raw = {
            "student_owned": 0.5,
            "ai_assisted_polished": 0.2,
            "ai_paraphrased": 0.18,
            "ai_generated_like": 0.12,
        }
        self._patch_breakdown(monkeypatch, raw, "student_owned")
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result())

        assert result["display_taxonomy"] == "three_way"
        ds = result["display_shares"]
        assert set(ds) == {"student_owned", "ai_assisted_polished", "ai_transformed"}
        assert ds["student_owned"] == pytest.approx(0.5)
        assert ds["ai_assisted_polished"] == pytest.approx(0.2)
        assert ds["ai_transformed"] == pytest.approx(0.18 + 0.12)
        # sum of display shares equals sum of four-way shares (mass preserved)
        assert sum(ds.values()) == pytest.approx(sum(raw.values()))

    def test_display_primary_is_argmax_of_merged_shares(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        # four-way primary is ai_paraphrased (0.35), but merged ai_transformed
        # (0.35 + 0.30 = 0.65) is the display argmax.
        raw = {
            "student_owned": 0.20,
            "ai_assisted_polished": 0.15,
            "ai_paraphrased": 0.35,
            "ai_generated_like": 0.30,
        }
        self._patch_breakdown(monkeypatch, raw, "ai_paraphrased")
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result())

        assert result["primary_category"] == "ai_paraphrased"  # four-way unchanged
        assert result["display_primary"] == "ai_transformed"

    def test_four_way_fields_unchanged_by_display_composition(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        raw = {
            "student_owned": 0.5,
            "ai_assisted_polished": 0.2,
            "ai_paraphrased": 0.18,
            "ai_generated_like": 0.12,
        }
        self._patch_breakdown(monkeypatch, raw, "student_owned")
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result())

        assert result["document_breakdown_raw"] == raw
        assert result["primary_category"] == "student_owned"

    def test_display_primary_student_owned_triggers_guard_under_red_tier(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        # four-way primary is ai_generated_like (guard would NOT fire on it),
        # but display_primary is student_owned -> extended guard MUST fire.
        raw = {
            "student_owned": 0.45,
            "ai_assisted_polished": 0.15,
            "ai_paraphrased": 0.20,
            "ai_generated_like": 0.20,
        }
        self._patch_breakdown(monkeypatch, raw, "ai_generated_like")
        detection_result = _realistic_detection_result()
        detection_result["tier"] = "strong"
        result = pipeline_bridge.run_v7_breakdown(detection_result)

        assert result["display_primary"] == "student_owned"
        assert result["primary_category"] == "ai_generated_like"
        assert result["presentation"] == "mixed_signals"
        assert result["primary_category_reliable"] is False
        assert "tier_category_contradiction" in result["uncertainty_flags"]

    def test_display_primary_not_student_no_guard(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        raw = {
            "student_owned": 0.20,
            "ai_assisted_polished": 0.15,
            "ai_paraphrased": 0.35,
            "ai_generated_like": 0.30,
        }
        self._patch_breakdown(monkeypatch, raw, "ai_paraphrased")
        detection_result = _realistic_detection_result()
        detection_result["tier"] = "strong"
        result = pipeline_bridge.run_v7_breakdown(detection_result)

        assert result["display_primary"] == "ai_transformed"
        assert result["primary_category_reliable"] is True
        assert "tier_category_contradiction" not in result["uncertainty_flags"]

    def test_four_way_mode_emits_no_display_fields(self, monkeypatch, tmp_path):
        import json as _json
        from pathlib import Path

        monkeypatch.setenv(_ENV_VAR, "1")
        real = _json.loads(
            (Path(pipeline_bridge.config.__file__).resolve().parent / "weights.json").read_text(
                encoding="utf-8"
            )
        )
        real["display_fallback"]["mode"] = "four_way"
        fp = tmp_path / "four_way_weights.json"
        fp.write_text(_json.dumps(real), encoding="utf-8")
        monkeypatch.setenv("DRAFTPROOF_V7_WEIGHTS_PATH", str(fp))
        pipeline_bridge.config.reload_weights(force=True)

        raw = {
            "student_owned": 0.5,
            "ai_assisted_polished": 0.2,
            "ai_paraphrased": 0.18,
            "ai_generated_like": 0.12,
        }
        self._patch_breakdown(monkeypatch, raw, "student_owned")
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result())

        assert "display_taxonomy" not in result
        assert "display_shares" not in result
        assert "display_primary" not in result
        monkeypatch.delenv("DRAFTPROOF_V7_WEIGHTS_PATH", raising=False)
        pipeline_bridge.config.reload_weights(force=True)


class TestIsDeepScanEnabled:
    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            ("1", True),
            ("true", True),
            ("TRUE", True),
            ("0", False),
            ("false", False),
            ("", False),
            ("yes", False),
        ],
    )
    def test_env_var_spellings(self, monkeypatch, raw_value, expected):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, raw_value)
        assert pipeline_bridge.is_deep_scan_enabled() is expected

    def test_unset_defaults_off(self, monkeypatch):
        monkeypatch.delenv(_DEEP_SCAN_ENV_VAR, raising=False)
        assert pipeline_bridge.is_deep_scan_enabled() is False


class TestDeepScanFusion:
    def test_deep_scan_disabled_uses_1_detector_path(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.delenv(_DEEP_SCAN_ENV_VAR, raising=False)

        def _boom(*args, **kwargs):
            raise AssertionError("modal_client.call_deep_scan must not be called when deep scan is disabled")

        monkeypatch.setattr(modal_client, "call_deep_scan", _boom)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text="some document text"))
        assert result is not None
        assert "deep_scan_uncalibrated" not in result.get("uncertainty_flags", [])

    def test_deep_scan_success_uses_2_detector_fusion_and_flags_uncalibrated(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        def _fake_call(chunks, timeout_s=60.0):
            assert chunks == ["some document text"]
            return {
                "available": True,
                "calibrated": False,
                "chunk_scores": [0.9],
                "document_score": 0.9,
            }

        monkeypatch.setattr(modal_client, "call_deep_scan", _fake_call)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text="some document text"))

        assert result is not None
        assert "deep_scan_uncalibrated" in result.get("uncertainty_flags", [])
        total = sum(result["document_breakdown_raw"].values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_deep_scan_calibrated_true_does_not_set_flag(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        monkeypatch.setattr(
            modal_client,
            "call_deep_scan",
            lambda chunks, timeout_s=60.0: {
                "available": True,
                "calibrated": True,
                "chunk_scores": [0.5],
                "document_score": 0.5,
            },
        )
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text="some document text"))
        assert result is not None
        assert "deep_scan_uncalibrated" not in result.get("uncertainty_flags", [])

    def test_modal_unavailable_falls_back_to_1_detector(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        monkeypatch.setattr(modal_client, "call_deep_scan", lambda chunks, timeout_s=60.0: None)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text="some document text"))
        assert result is not None
        assert "deep_scan_uncalibrated" not in result.get("uncertainty_flags", [])

    def test_modal_error_falls_back_to_1_detector_no_crash(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        def _fake_call(chunks, timeout_s=60.0):
            raise RuntimeError("network exploded")

        monkeypatch.setattr(modal_client, "call_deep_scan", _fake_call)
        # run_v7_breakdown wraps its whole body in try/except, so even a
        # modal_client bug that somehow raises must not crash the scan.
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text="some document text"))
        assert result is None

    def test_deep_scan_enabled_no_document_text_falls_back(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        def _boom(*args, **kwargs):
            raise AssertionError("modal_client.call_deep_scan must not be called with no document text")

        monkeypatch.setattr(modal_client, "call_deep_scan", _boom)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result())  # no text=
        assert result is not None
        assert "deep_scan_uncalibrated" not in result.get("uncertainty_flags", [])

    def test_deep_scan_malformed_response_falls_back(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        monkeypatch.setattr(
            modal_client,
            "call_deep_scan",
            lambda chunks, timeout_s=60.0: {"available": False},
        )
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text="some document text"))
        assert result is not None
        assert "deep_scan_uncalibrated" not in result.get("uncertainty_flags", [])

    def test_multi_sentence_document_windows_before_modal_call(self, monkeypatch):
        """The Modal call must receive overlapping 3-sentence WINDOWS
        (deberta_windowing.build_windows), NOT one document-length chunk — the
        calibration (poc/calibration/v7_deberta_academic_baseline.json) is
        validated at window granularity, so a whole-document call is
        meaningless. Windowing replaced the earlier per-sentence chunking in
        commit 18b0a7c5 (isolated short sentences scored ~0 on the
        window-trained checkpoint and cratered the calibrated mean). A
        5-sentence document must therefore yield 3 overlapping windows, never a
        single whole-document chunk."""
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        doc_text = (
            "First sentence here. Second sentence here. Third sentence here. "
            "Fourth sentence here. Fifth sentence here."
        )  # 5 sentences -> build_windows(size=3, step=1) -> 3 overlapping windows
        captured = {}

        def _fake_call(chunks, timeout_s=60.0):
            captured["chunks"] = chunks
            return {
                "available": True,
                "calibrated": True,
                "chunk_scores": [0.9] * len(chunks),
                "document_score": 0.7,
            }

        monkeypatch.setattr(modal_client, "call_deep_scan", _fake_call)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text=doc_text))

        assert result is not None
        chunks = captured["chunks"]
        assert len(chunks) == 3
        assert chunks != [doc_text]
        assert all(len(c) < len(doc_text) for c in chunks)

    def test_proportion_computed_correctly_from_chunk_scores(self, monkeypatch):
        """The doc-level proportion is computed from the per-sentence scores:
        under representation=calibrated_mean it is
        ``clip((mean(sentence_scores)-lo)/(hi-lo),0,1)``. These distinct scores
        mean 0.945, giving proportion 0.5 — which is >= doc_floor (0.3), so the
        deep_scan payload is present with that proportion and NO
        reliability-floor flag is set."""
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        doc_text = "Sentence one is here. Sentence two follows. Sentence three next. Sentence four last."
        # Distinct per-sentence scores (not uniform) so the assertion genuinely
        # exercises the mean-over-sentences computation; mean == 0.945.
        sentence_scores = [0.96, 0.93, 0.96, 0.93]
        _inject_sentence_scores(monkeypatch, sentence_scores, calibrated=True)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text=doc_text))

        assert result is not None
        deep = result.get("deep_scan")
        assert deep is not None  # payload present — the proportion was computed, not skipped
        expected = _expected_calibrated_mean_proportion(sentence_scores)
        assert deep["proportion"] == pytest.approx(expected, abs=1e-6)
        assert expected >= config.get_deep_scan_calibration()["doc_floor"]
        assert "deep_scan_below_reliability_floor" not in result.get("uncertainty_flags", [])

    def test_proportion_below_floor_sets_reliability_flag(self, monkeypatch):
        """A doc-level deep-scan proportion below doc_floor (0.3): the fusion
        score is still computed and passed (not zeroed), but the low-reliability
        flag must be set. Post-2026-07-14 the doc-level signal is the CALIBRATED
        MEAN of the per-sentence scores (config representation=calibrated_mean),
        so ``_run_with_proportion(flagged=2)`` drives proportion == 0.2 < 0.3."""
        result = _run_with_proportion(monkeypatch, flagged=2)  # proportion 0.2 < doc_floor 0.3

        assert result is not None
        assert "deep_scan_below_reliability_floor" in result.get("uncertainty_flags", [])
        # fusion score must still be computed (not zeroed) — the breakdown weights sum to 1.
        total = sum(result["document_breakdown_raw"].values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_proportion_at_floor_does_not_set_flag(self, monkeypatch):
        """Exactly at doc_floor (0.3) must NOT be flagged as below floor."""
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        doc_text = "One two three. Four five six. Seven eight nine. Ten eleven twelve. Thirteen fourteen."

        monkeypatch.setattr(
            modal_client,
            "call_deep_scan",
            lambda chunks, timeout_s=60.0: {
                "available": True,
                "calibrated": True,
                # 5 sentences: exactly 30% (0.3) >= sent_threshold -> 1.5, use int math via 10 sentences instead
                "chunk_scores": [0.999, 0.999, 0.999, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                "document_score": 0.4,
            },
        )
        # Build a 10-sentence document to match chunk_scores length.
        doc_text_10 = " ".join(f"Sentence number {i}." for i in range(10))
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text=doc_text_10))

        assert result is not None
        assert "deep_scan_below_reliability_floor" not in result.get("uncertainty_flags", [])

    def test_single_sentence_no_punctuation_does_not_crash(self, monkeypatch):
        """A one-sentence document with no terminal punctuation still splits
        to exactly one usable sentence (split_sentences' fallback) — Modal
        is called with that single chunk, no crash."""
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        monkeypatch.setattr(
            modal_client,
            "call_deep_scan",
            lambda chunks, timeout_s=60.0: {
                "available": True,
                "calibrated": True,
                "chunk_scores": [0.5],
                "document_score": 0.5,
            },
        )
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text="nopunctuationhere"))
        assert result is not None

    def test_zero_sentences_falls_back_without_calling_modal(self, monkeypatch):
        """split_sentences on effectively-empty text yields no sentences —
        the deep-scan path must skip the Modal call entirely rather than
        calling it with an empty chunk list."""
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        def _boom(*args, **kwargs):
            raise AssertionError("modal_client.call_deep_scan must not be called with zero sentences")

        monkeypatch.setattr(modal_client, "call_deep_scan", _boom)
        monkeypatch.setattr(pipeline_bridge, "split_sentences", lambda text: [])
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text="some document text"))
        assert result is not None
        assert "deep_scan_uncalibrated" not in result.get("uncertainty_flags", [])

    def test_mismatched_chunk_scores_length_falls_back(self, monkeypatch):
        """If Modal returns a chunk_scores list whose length doesn't match
        the number of sentences sent, this is malformed — fall back rather
        than silently misaligning scores to sentences."""
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        doc_text = "Sentence one. Sentence two. Sentence three."

        monkeypatch.setattr(
            modal_client,
            "call_deep_scan",
            lambda chunks, timeout_s=60.0: {
                "available": True,
                "calibrated": True,
                "chunk_scores": [0.9, 0.9],  # only 2 scores for 3 sentences
                "document_score": 0.5,
            },
        )
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text=doc_text))

        assert result is not None
        assert "deep_scan_uncalibrated" not in result.get("uncertainty_flags", [])
        assert "deep_scan_below_reliability_floor" not in result.get("uncertainty_flags", [])


def _ten_sentence_doc() -> str:
    return " ".join(f"Sentence number {i}." for i in range(10))


def _run_with_proportion(monkeypatch, flagged: int, calibrated: bool = True):
    """Drive the doc-level deep-scan ``proportion`` to EXACTLY ``flagged/10``
    over a 10-sentence document, then exercise the display-band mapping.

    The proportion->band mapping (``_deep_scan_band`` + the weights.json
    display-band cutoffs) is representation-INDEPENDENT, but the default
    ``representation="calibrated_mean"`` computes the doc signal as
    ``clip((mean-lo)/(hi-lo),0,1)`` whose float round-trip cannot land on the
    exact band boundaries (e.g. an intended 0.70 comes back 0.6999…, flipping
    orange<->red). So this helper pins the config's documented
    ``representation="proportion"`` rollback (a config-only switch, see
    ``deep_scan_calibration._representation_notes``): proportion becomes the
    exact rational ``flagged/10`` (count of sentences >= sent_threshold). The
    calibrated_mean doc signal is covered by the per-paragraph tests. Anchors /
    threshold are read from config (no hardcoded checkpoint constants)."""
    monkeypatch.setenv(_ENV_VAR, "1")
    monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
    cal = dict(config.get_deep_scan_calibration())
    cal["representation"] = "proportion"
    monkeypatch.setattr(config, "get_deep_scan_calibration", lambda: cal)
    sent_threshold = cal["sent_threshold"]
    # exactly `flagged` sentences at/above the flag threshold, rest well below.
    scores = [sent_threshold] * flagged + [0.0] * (10 - flagged)
    _inject_sentence_scores(monkeypatch, scores, calibrated=calibrated)
    return pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text=_ten_sentence_doc()))


class TestDeepScanDisplayBand:
    """User-facing 'Deep-scan AI estimate' object (schema contract):
    {"proportion": float, "band": "insufficient"|"amber"|"orange"|"red",
    "calibrated": bool}. Present ONLY on deep-scan success; NEVER "green"."""

    def test_success_attaches_deep_scan_with_proportion_and_calibrated(self, monkeypatch):
        result = _run_with_proportion(monkeypatch, flagged=4, calibrated=True)
        assert result is not None
        ds = result["deep_scan"]
        assert ds == {"proportion": pytest.approx(0.4), "band": "amber", "calibrated": True}

    def test_calibrated_false_passthrough(self, monkeypatch):
        result = _run_with_proportion(monkeypatch, flagged=6, calibrated=False)
        assert result is not None
        assert result["deep_scan"]["calibrated"] is False

    def test_deep_scan_disabled_no_key(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.delenv(_DEEP_SCAN_ENV_VAR, raising=False)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text=_ten_sentence_doc()))
        assert result is not None
        assert "deep_scan" not in result

    def test_deep_scan_failed_no_key(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        monkeypatch.setattr(modal_client, "call_deep_scan", lambda chunks, timeout_s=60.0: None)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text=_ten_sentence_doc()))
        assert result is not None
        assert "deep_scan" not in result

    def test_deep_scan_no_text_no_key(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result())  # no text=
        assert result is not None
        assert "deep_scan" not in result

    @pytest.mark.parametrize(
        "flagged,expected_band",
        [
            (0, "insufficient"),   # 0.0 < 0.3 floor
            (2, "insufficient"),   # 0.2 < 0.3 floor
            (3, "amber"),          # exactly 0.3 (insufficient_below edge, inclusive band)
            (4, "amber"),          # 0.4 < orange_min 0.5
            (5, "orange"),         # exactly 0.5 (orange_min edge)
            (6, "orange"),         # 0.6 < red_min 0.7
            (7, "red"),            # exactly 0.7 (red_min edge)
            (10, "red"),           # 1.0
        ],
    )
    def test_band_boundaries_exact(self, monkeypatch, flagged, expected_band):
        result = _run_with_proportion(monkeypatch, flagged=flagged)
        assert result is not None
        ds = result["deep_scan"]
        assert ds["proportion"] == pytest.approx(flagged / 10.0)
        assert ds["band"] == expected_band

    @pytest.mark.parametrize("flagged", list(range(11)))
    def test_never_emits_green(self, monkeypatch, flagged):
        result = _run_with_proportion(monkeypatch, flagged=flagged)
        assert result is not None
        assert result["deep_scan"]["band"] in {"insufficient", "amber", "orange", "red"}


class TestDeepScanPerParagraphProportions:
    """Additive per-paragraph grouping of the SAME per-sentence Modal scores
    (zero extra Modal cost; document-level math untouched)."""

    _TEXT = (
        "First para sentence one. First para sentence two.\n\n"
        "Second para sentence."
    )

    def test_multi_paragraph_payload_carries_per_paragraph_proportions(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        # 3 sentences: para 1 = s0,s1 (both AI-like); para 2 = s2 (human).
        sentence_scores = [1.0, 1.0, 0.0]
        _inject_sentence_scores(monkeypatch, sentence_scores, calibrated=True)
        result = pipeline_bridge.run_v7_breakdown(
            _realistic_detection_result(text=self._TEXT)
        )
        assert result is not None
        deep = result.get("deep_scan")
        assert deep is not None
        # Doc-level signal is the CALIBRATED MEAN of the per-sentence scores
        # (representation=calibrated_mean), NOT the flagged fraction — and it is
        # unchanged by the additive per-paragraph grouping below.
        assert deep["proportion"] == pytest.approx(
            _expected_calibrated_mean_proportion(sentence_scores), abs=1e-6
        )
        rows = deep.get("paragraphs")
        assert rows is not None and len(rows) == 2
        assert rows[0] == {
            "index": 0,
            "sentence_count": 2,
            "flagged_count": 2,
            "proportion": 1.0,
            "band": "red",
        }
        assert deep["reliability_floor"] == pytest.approx(0.3)
        assert rows[1]["index"] == 1
        assert rows[1]["sentence_count"] == 1
        assert rows[1]["proportion"] == 0.0
        assert rows[1]["band"] == "insufficient"

    def test_single_paragraph_omits_paragraphs_key(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        monkeypatch.setattr(
            modal_client,
            "call_deep_scan",
            lambda chunks, timeout_s=60.0: {
                "available": True,
                "calibrated": True,
                "chunk_scores": [1.0] * len(chunks),
                "document_score": 1.0,
            },
        )
        result = pipeline_bridge.run_v7_breakdown(
            _realistic_detection_result(text="One paragraph only. Two sentences here.")
        )
        assert result is not None
        deep = result.get("deep_scan")
        assert deep is not None
        assert "paragraphs" not in deep

    def test_title_heading_is_not_a_paragraph_row(self, monkeypatch):
        """Live-report regression (2026-07-06): a standalone title block was
        emitted as 'Paragraph 1 — 100% (1 sentence)' and shifted body
        paragraph numbering (panel showed 4 paragraphs for a 3-paragraph
        essay). Headings are excluded from rows; body ordinals start at 0;
        the document proportion still counts the title's sentence."""
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        text = (
            "Critical Analysis of Over-Accommodation Standards\n\n"
            "First body paragraph sentence one. First body sentence two.\n\n"
            "Second body paragraph sentence one. Second body sentence two. Third one here.\n\n"
            "Third body paragraph sentence."
        )

        # split_sentences merges the unpunctuated title into the following
        # sentence in the normalized stream; don't assert an exact sentence
        # count in the fake — accept whatever segmentation produces.
        def fake_call(chunks, timeout_s=60.0):
            return {
                "available": True,
                "calibrated": True,
                "chunk_scores": [1.0] * len(chunks),
                "document_score": 1.0,
            }

        monkeypatch.setattr(modal_client, "call_deep_scan", fake_call)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text=text))
        assert result is not None
        rows = (result.get("deep_scan") or {}).get("paragraphs")
        assert rows is not None
        assert len(rows) == 3  # the title is NOT a paragraph row
        assert [r["index"] for r in rows] == [0, 1, 2]  # body ordinals, no gap

    def test_title_merged_flagged_sentence_counts_toward_first_body_paragraph(self, monkeypatch):
        """Live regression (2026-07-06, report fac2ff21): an unpunctuated title
        merges with the first body sentence in the normalized sentence stream;
        attributing that merged sentence to the (excluded) heading block made
        paragraph 1 show '0 of 4 flagged' while its first sentence was visibly
        underlined. The merged sentence extends past the heading range, so it
        belongs to the first body paragraph."""
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        text = (
            "Critical Analysis of Over-Accommodation Standards\n\n"
            "First body sentence which reads AI-like. Second body sentence here.\n\n"
            "Other paragraph sentence one. Other paragraph sentence two."
        )

        def fake_call(chunks, timeout_s=60.0):
            # Flag ONLY the first chunk — the merged title+first-body sentence.
            return {
                "available": True,
                "calibrated": True,
                "chunk_scores": [1.0] + [0.0] * (len(chunks) - 1),
                "document_score": 0.3,
            }

        monkeypatch.setattr(modal_client, "call_deep_scan", fake_call)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text=text))
        assert result is not None
        rows = (result.get("deep_scan") or {}).get("paragraphs")
        assert rows is not None and len(rows) == 2
        # Paragraph 1 owns the merged flagged sentence: 1 of 2 flagged, not 0.
        assert rows[0]["flagged_count"] == 1
        assert rows[0]["sentence_count"] == 2
        assert rows[1]["flagged_count"] == 0

    def test_display_sentence_count_uses_canonical_card_segmentation(self, monkeypatch):
        """Live regression (2026-07-09, report 07cb40b2): the Modal deep-scan
        splitter over-segments some paragraphs versus the canonical splitter the
        on-page card / sentence_map use (e.g. a lowercase name after a period —
        'johnny. johnny'). That made the deep-scan table show a 2-sentence
        paragraph as '1 of 3' while the card showed 2 sentences. The DISPLAY
        sentence_count must be re-based to the canonical per-paragraph count
        (structured_sentence_segments); flagged is clamped to that denominator.
        The document-level proportion is NOT re-based."""
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        # Body para 2 over-splits: deep splitter -> 3 sentences, canonical -> 2.
        text = (
            "First body sentence one here. First body sentence two here.\n\n"
            "The system helped johnny. johnny improved over the years. "
            "Everyone noticed the change."
        )

        # Deep splitter yields 5 sentences (2 + 3); flag only the last.
        sentence_scores = [0.0, 0.0, 0.0, 0.0, 1.0]
        _inject_sentence_scores(monkeypatch, sentence_scores, calibrated=True)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text=text))
        assert result is not None
        deep = result.get("deep_scan")
        rows = (deep or {}).get("paragraphs")
        assert rows is not None and len(rows) == 2
        # Denominators are the CANONICAL card counts (2 / 2), NOT the deep
        # splitter's over-segmented counts (2 / 3).
        assert [r["sentence_count"] for r in rows] == [2, 2]
        # Re-based flagged proportion for the over-split paragraph: 1 of 2 = 0.5,
        # not the pre-fix 1 of 3 = 0.333.
        assert rows[1]["flagged_count"] == 1
        assert rows[1]["proportion"] == pytest.approx(0.5)
        # Document-level proportion is the calibrated MEAN of the per-sentence
        # scores (representation=calibrated_mean) — decoupled from and NOT
        # re-based by the per-paragraph display counts (the ESL-gated fused-score
        # input must not move).
        assert deep["proportion"] == pytest.approx(
            _expected_calibrated_mean_proportion(sentence_scores), abs=1e-6
        )


class TestBuilderThreadsCriterionScores:
    """Regression test for the wiring bug documented in
    poc/calibration/v12_validation/false_ai_diagnosis.json: builder.py used
    to spread ``{**ai_risk_badge, "document_text": ..., "_precomputed_deep_scan": ...}``
    into ``run_v7_breakdown`` WITHOUT ``criterion_scores`` — that key lives in
    ``self._summaries`` (builder.py L370-371), never in the badge — so every
    criterion-derived signal (specificity_score, sentence_variance,
    sentence_smoothness, local_style_shift, detector_disagreement) was
    unconditionally "unavailable" in production. This drives a REAL
    ReportBuilder end-to-end (quick-scan, no Modal/network) and asserts the
    dict actually handed to run_v7_breakdown carries "criterion_scores" and
    it matches self._summaries's value byte-for-byte — the exact call site
    that was broken.
    """

    def test_criterion_scores_present_in_run_v7_breakdown_input(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.delenv(_DEEP_SCAN_ENV_VAR, raising=False)  # quick-scan: no Modal spend

        from detect.document_structure import normalize_submitted_text
        from detect.run import DetectionRunner
        from report import builder as builder_module
        from report.builder import ReportBuilder

        captured: dict = {}
        real_run_v7_breakdown = pipeline_bridge.run_v7_breakdown

        def _capturing_run_v7_breakdown(detection_result):
            captured["input"] = detection_result
            return real_run_v7_breakdown(detection_result)

        monkeypatch.setattr(
            builder_module, "run_v7_breakdown", _capturing_run_v7_breakdown, raising=False
        )
        # builder.py imports run_v7_breakdown via a LOCAL import inside the
        # method body (`from detect_v7.pipeline_bridge import run_v7_breakdown
        # as _run_v7_breakdown`) — patching the module-level name above is a
        # no-op for that call site, so patch the source pipeline_bridge
        # function itself, which the local import re-resolves at call time.
        monkeypatch.setattr(pipeline_bridge, "run_v7_breakdown", _capturing_run_v7_breakdown)

        text = (
            "The industrial revolution transformed European society in profound ways. "
            "Factories replaced workshops, and cities grew rapidly as workers migrated."
        ) * 3
        norm_text = normalize_submitted_text(text)
        runner = DetectionRunner()
        detection_result = runner.run_all(norm_text)

        b = ReportBuilder()
        b.add_detection_report(detection_result)
        if getattr(detection_result, "postprocess_results", None):
            b.add_postprocess_results(detection_result.postprocess_results)
        b.set_meta(scan_time=0.0, original_text=norm_text)

        assert "input" not in captured  # sanity: not yet called
        b.build()

        assert "input" in captured, "run_v7_breakdown was never invoked — check the kill switch"
        passed_in = captured["input"]
        assert "criterion_scores" in passed_in
        assert passed_in["criterion_scores"] == b._summaries.get("criterion_scores")
        assert passed_in["criterion_scores"], "criterion_scores must be non-empty on this fixture"


class TestPrecomputedDeepScanSentinel:
    """FIX 2: the three-state ``_precomputed_deep_scan`` contract that stops the
    builder's tier-authority call and run_v7_breakdown from BOTH hitting Modal."""

    def test_attempted_and_failed_sentinel_skips_second_modal_call(self, monkeypatch):
        # Tier-authority already tried Modal and it failed → sentinel handed in →
        # get_deep_scan_proportion must NOT re-call (the inconsistency FIX 2 kills).
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        def _boom(*a, **k):
            raise AssertionError("must not re-call Modal after an attempted-and-failed call")

        monkeypatch.setattr(modal_client, "call_deep_scan", _boom)
        out = pipeline_bridge.get_deep_scan_proportion(
            {
                "document_text": "A sentence here. Another one there.",
                "_precomputed_deep_scan": pipeline_bridge.DEEP_SCAN_ATTEMPTED_FAILED,
            }
        )
        assert out is None

    def test_not_attempted_none_still_calls_modal(self, monkeypatch):
        # Tier-authority OFF (None handed in) → breakdown may still call Modal.
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        called = {"n": 0}

        def _fake(chunks, timeout_s=60.0):
            called["n"] += 1
            return None  # only asserting it WAS called

        monkeypatch.setattr(modal_client, "call_deep_scan", _fake)
        out = pipeline_bridge.get_deep_scan_proportion(
            {
                "document_text": "A sentence here. Another one there. And a third one now.",
                "_precomputed_deep_scan": None,
            }
        )
        assert called["n"] == 1
        assert out is None

    def test_real_precomputed_result_is_reused_without_calling_modal(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("must not call Modal when a real precomputed result is present")

        monkeypatch.setattr(modal_client, "call_deep_scan", _boom)
        real = {
            "proportion": 0.42,
            "uncalibrated": False,
            "below_floor": False,
            "payload": {"proportion": 0.42},
        }
        out = pipeline_bridge.get_deep_scan_proportion({"_precomputed_deep_scan": real})
        assert out is real


class TestDeepScanCapPropagation:
    """FIX 3: a capped Modal response surfaces ``capped`` up through the
    bridge's returned provenance and display payload (no silent subset)."""

    def test_capped_response_propagates_flag(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        # capped=True → effective_windows truncates to the returned-score prefix,
        # so the length-match contract holds regardless of the true window count.
        monkeypatch.setattr(
            modal_client,
            "call_deep_scan",
            lambda chunks, timeout_s=60.0: {
                "available": True,
                "calibrated": True,
                "chunk_scores": [0.99],
                "capped": True,
            },
        )
        out = pipeline_bridge.get_deep_scan_proportion(
            {"document_text": "One here. Two there. Three yonder. Four is more."}
        )
        assert out is not None
        assert out["capped"] is True
        assert out["payload"].get("capped") is True
