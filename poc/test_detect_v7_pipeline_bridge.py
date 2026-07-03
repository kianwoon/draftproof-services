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

from detect_v7 import pipeline_bridge

_ENV_VAR = "DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN"


def _clear_env(monkeypatch):
    monkeypatch.delenv(_ENV_VAR, raising=False)


def _realistic_detection_result(ai_likelihood_score: float = 62.5) -> dict:
    """A dict shaped like builder.py's ``ai_risk_badge`` at the V7 call site."""
    return {
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

    def test_enabled_true_spelling_also_works(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "true")
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result())
        assert result is not None

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
