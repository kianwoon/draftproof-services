"""Tests for detect_v7.deep_scan_heatmap.compose_deep_scan_heatmap.

Covers: output shape parity with compose_from_sentences, band assignment from
config.get_deep_scan_calibration()['sent_threshold'], fail-open (deep scan disabled ->
None; Modal unavailable/length-mismatch -> None), and too-short (<8-word) sentences
scoring None/clean.
"""
from __future__ import annotations

import pytest

from detect_v7 import config, modal_client
from detect_v7 import deep_scan_heatmap
from detect_v7 import pipeline_bridge

_DEEP_SCAN_ENV_VAR = "DRAFTPROOF_V7_DEEP_SCAN"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(_DEEP_SCAN_ENV_VAR, raising=False)
    # Reset the V7 weights cache to the real bundled weights.json — otherwise a
    # prior test file (alphabetically earlier) that loaded a fixture weights file
    # can leave config cached without deep_scan_calibration keys, which surfaces
    # here as a KeyError. Mirrors test_detect_v7_config.py's reset fixture.
    monkeypatch.delenv("DRAFTPROOF_V7_WEIGHTS_PATH", raising=False)
    config.reload_weights(force=True)
    yield
    monkeypatch.delenv("DRAFTPROOF_V7_WEIGHTS_PATH", raising=False)
    config.reload_weights(force=True)


def _sentences(*texts):
    return [
        {"sentence_id": f"s{i + 1:03d}", "paragraph_id": "p001", "text": t}
        for i, t in enumerate(texts)
    ]


LONG_A = "This sentence definitely has more than eight words in it for sure."
LONG_B = "Another qualifying sentence with plenty of words to pass the floor."
SHORT = "Too short."


class TestDisabledOrEmpty:
    def test_deep_scan_disabled_returns_none(self, monkeypatch):
        monkeypatch.delenv(_DEEP_SCAN_ENV_VAR, raising=False)

        def _boom(*a, **kw):
            raise AssertionError("must not call Modal when deep scan disabled")

        monkeypatch.setattr(modal_client, "call_deep_scan", _boom)
        assert deep_scan_heatmap.compose_deep_scan_heatmap(_sentences(LONG_A)) is None

    def test_empty_sentences_returns_none(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        assert deep_scan_heatmap.compose_deep_scan_heatmap([]) is None


class TestShape:
    def test_output_shape_matches_compose_from_sentences(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        def _fake_call(chunks, timeout_s=60.0):
            assert chunks == [LONG_A, LONG_B]
            return {"available": True, "calibrated": True, "chunk_scores": [0.9995, 0.2]}

        monkeypatch.setattr(modal_client, "call_deep_scan", _fake_call)
        result = deep_scan_heatmap.compose_deep_scan_heatmap(_sentences(LONG_A, LONG_B))
        assert result is not None
        assert result["available"] is True
        assert set(result.keys()) >= {"sentence_scores", "available"}
        rows = result["sentence_scores"]
        assert len(rows) == 2
        for row in rows:
            assert set(row.keys()) == {"sentence_id", "paragraph_id", "score", "band"}


class TestBandAssignment:
    # Highlights use the SAME calibrated sent_threshold (0.999) as
    # get_deep_scan_proportion — no separate, uncalibrated "display" threshold.
    # (An uncalibrated 0.99 split was proposed and rejected: weights.json's own
    # calibration notes show 0.99 gave a 68% FPR on human ESL essays at
    # doc-level, and there is no sentence-level calibration run for 0.99 either.)
    def test_score_at_or_above_threshold_is_high(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        threshold = config.get_deep_scan_calibration()["sent_threshold"]

        def _fake_call(chunks, timeout_s=60.0):
            return {"available": True, "calibrated": True, "chunk_scores": [threshold]}

        monkeypatch.setattr(modal_client, "call_deep_scan", _fake_call)
        result = deep_scan_heatmap.compose_deep_scan_heatmap(_sentences(LONG_A))
        assert result["sentence_scores"][0]["band"] == "high"

    def test_score_below_threshold_is_clean(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        threshold = config.get_deep_scan_calibration()["sent_threshold"]

        def _fake_call(chunks, timeout_s=60.0):
            return {"available": True, "calibrated": True, "chunk_scores": [threshold - 0.01]}

        monkeypatch.setattr(modal_client, "call_deep_scan", _fake_call)
        result = deep_scan_heatmap.compose_deep_scan_heatmap(_sentences(LONG_A))
        assert result["sentence_scores"][0]["band"] == "clean"


class TestTooShortSentences:
    def test_short_sentence_scores_none_clean(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        def _fake_call(chunks, timeout_s=60.0):
            assert chunks == [LONG_A]  # SHORT excluded from the Modal call
            return {"available": True, "calibrated": True, "chunk_scores": [0.999]}

        monkeypatch.setattr(modal_client, "call_deep_scan", _fake_call)
        result = deep_scan_heatmap.compose_deep_scan_heatmap(_sentences(SHORT, LONG_A))
        rows = {r["sentence_id"]: r for r in result["sentence_scores"]}
        assert rows["s001"]["score"] is None
        assert rows["s001"]["band"] == "clean"
        assert rows["s002"]["score"] is not None

    def test_all_short_sentences_returns_available_with_no_modal_call(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        def _boom(*a, **kw):
            raise AssertionError("must not call Modal when nothing qualifies")

        monkeypatch.setattr(modal_client, "call_deep_scan", _boom)
        result = deep_scan_heatmap.compose_deep_scan_heatmap(_sentences(SHORT))
        assert result["available"] is True
        assert result["sentence_scores"][0]["score"] is None
        assert result["sentence_scores"][0]["band"] == "clean"


class TestModelVersionProvenance:
    """2026-07-19: report.py used to hardcode the badge's ai_signal_deberta.model_version
    as "deberta_signal_v2" (poc/detect/deberta_signal.py's OWN constant) even when the
    heatmap it just rebuilt the tile from came from THIS module — a separate Modal
    checkpoint. This module is the actual source of truth for which checkpoint served
    the call, so it must surface a real identifier instead of letting the caller guess."""

    def test_model_version_sourced_from_modal_response_checkpoint(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        monkeypatch.delenv("DRAFTPROOF_MODAL_CHECKPOINT", raising=False)

        def _fake_call(chunks, timeout_s=60.0):
            return {
                "available": True,
                "calibrated": True,
                "chunk_scores": [0.9],
                "checkpoint": "desklib/ai-text-detector-academic-v1.01",
            }

        monkeypatch.setattr(modal_client, "call_deep_scan", _fake_call)
        result = deep_scan_heatmap.compose_deep_scan_heatmap(_sentences(LONG_A))
        assert result["model_version"] == "desklib/ai-text-detector-academic-v1.01"

    def test_model_version_falls_back_to_env_checkpoint_when_response_omits_it(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        monkeypatch.setenv("DRAFTPROOF_MODAL_CHECKPOINT", "draftproof/finetune-v1-gpt55")

        def _fake_call(chunks, timeout_s=60.0):
            return {"available": True, "calibrated": True, "chunk_scores": [0.9]}  # no "checkpoint" key

        monkeypatch.setattr(modal_client, "call_deep_scan", _fake_call)
        result = deep_scan_heatmap.compose_deep_scan_heatmap(_sentences(LONG_A))
        assert result["model_version"] == "draftproof/finetune-v1-gpt55"

    def test_model_version_none_when_no_source_available(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        monkeypatch.delenv("DRAFTPROOF_MODAL_CHECKPOINT", raising=False)

        def _fake_call(chunks, timeout_s=60.0):
            return {"available": True, "calibrated": True, "chunk_scores": [0.9]}

        monkeypatch.setattr(modal_client, "call_deep_scan", _fake_call)
        result = deep_scan_heatmap.compose_deep_scan_heatmap(_sentences(LONG_A))
        assert result.get("model_version") is None


class TestFailOpen:
    def test_modal_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
        monkeypatch.setattr(modal_client, "call_deep_scan", lambda *a, **kw: None)
        assert deep_scan_heatmap.compose_deep_scan_heatmap(_sentences(LONG_A)) is None

    def test_length_mismatch_returns_none(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        def _fake_call(chunks, timeout_s=60.0):
            return {"available": True, "calibrated": True, "chunk_scores": [0.9, 0.9]}  # too many

        monkeypatch.setattr(modal_client, "call_deep_scan", _fake_call)
        assert deep_scan_heatmap.compose_deep_scan_heatmap(_sentences(LONG_A)) is None

    def test_exception_returns_none(self, monkeypatch):
        monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")

        def _boom(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(modal_client, "call_deep_scan", _boom)
        assert deep_scan_heatmap.compose_deep_scan_heatmap(_sentences(LONG_A)) is None
