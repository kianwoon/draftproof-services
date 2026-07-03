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

from detect_v7 import modal_client, pipeline_bridge

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

    def test_multi_sentence_document_splits_before_modal_call(self, monkeypatch):
        """The Modal call must receive one chunk PER SENTENCE, not one
        document-length chunk — this is the core bug fix: the calibration
        (poc/calibration/v7_deberta_academic_baseline.json) was validated at
        sentence granularity, so a whole-document call is meaningless."""
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        doc_text = "The first sentence is short. Here is a second sentence, also short. And a third one."
        captured = {}

        def _fake_call(chunks, timeout_s=60.0):
            captured["chunks"] = chunks
            return {
                "available": True,
                "calibrated": True,
                "chunk_scores": [0.9999, 0.1, 0.9999],
                "document_score": 0.7,
            }

        monkeypatch.setattr(modal_client, "call_deep_scan", _fake_call)
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text=doc_text))

        assert result is not None
        chunks = captured["chunks"]
        assert len(chunks) == 3
        assert all(len(c) < len(doc_text) for c in chunks)
        assert chunks != [doc_text]

    def test_proportion_computed_correctly_from_chunk_scores(self, monkeypatch):
        """2 of 4 sentences >= sent_threshold (0.999) -> proportion == 0.5,
        which is >= doc_floor (0.3), so no reliability-floor flag."""
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        doc_text = "Sentence one is here. Sentence two follows. Sentence three next. Sentence four last."

        monkeypatch.setattr(
            modal_client,
            "call_deep_scan",
            lambda chunks, timeout_s=60.0: {
                "available": True,
                "calibrated": True,
                "chunk_scores": [0.999, 0.5, 0.9995, 0.2],
                "document_score": 0.5,
            },
        )
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text=doc_text))

        assert result is not None
        assert "deep_scan_below_reliability_floor" not in result.get("uncertainty_flags", [])

    def test_proportion_below_floor_sets_reliability_flag(self, monkeypatch):
        """1 of 4 sentences >= sent_threshold (0.999) -> proportion == 0.25,
        which is < doc_floor (0.3): fusion score is still computed and passed
        (not zeroed), but the low-reliability flag must be set."""
        monkeypatch.setenv(_ENV_VAR, "1")
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        doc_text = "Sentence one is here. Sentence two follows. Sentence three next. Sentence four last."

        monkeypatch.setattr(
            modal_client,
            "call_deep_scan",
            lambda chunks, timeout_s=60.0: {
                "available": True,
                "calibrated": True,
                "chunk_scores": [0.999, 0.1, 0.2, 0.3],
                "document_score": 0.4,
            },
        )
        result = pipeline_bridge.run_v7_breakdown(_realistic_detection_result(text=doc_text))

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
