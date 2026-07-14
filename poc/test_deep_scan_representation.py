"""Tests for the calibrated-mean deep-scan doc-level representation
(owner-approved 2026-07-14).

Covers:
1. config.get_deep_scan_calibration validation: representation defaults to
   "proportion" when absent (backward compat), and calibrated_mean requires
   0 < lo < hi <= 1 (raises ValueError otherwise).
2. pipeline_bridge.get_deep_scan_proportion: calibrated_mean branch computes
   clip((mean(chunk_scores)-lo)/(hi-lo), 0, 1) as the doc-level "proportion"/
   payload/band/below_floor signal.
3. proportion representation (legacy) reproduces the existing threshold-
   proportion math exactly.
"""
from __future__ import annotations

import pytest

from detect_v7 import config, pipeline_bridge

_DEEP_SCAN_ENV_VAR = "DRAFTPROOF_V7_DEEP_SCAN"


class TestConfigValidation:
    def _patched_weights(self, monkeypatch, calibration_overrides, drop_keys=()):
        full_weights = dict(config._weights())
        calibration = dict(full_weights["deep_scan_calibration"])
        for k in drop_keys:
            calibration.pop(k, None)
        calibration.update(calibration_overrides)
        full_weights = {**full_weights, "deep_scan_calibration": calibration}
        monkeypatch.setattr(config, "_weights", lambda: full_weights)

    def test_defaults_to_proportion_when_representation_absent(self, monkeypatch):
        self._patched_weights(monkeypatch, {}, drop_keys=("representation",))
        calibration = config.get_deep_scan_calibration()
        assert calibration.get("representation", "proportion") == "proportion"

    def test_calibrated_mean_requires_valid_anchors(self, monkeypatch):
        # lo >= hi -> invalid
        self._patched_weights(
            monkeypatch,
            {"representation": "calibrated_mean", "mean_anchor_lo": 0.96, "mean_anchor_hi": 0.93},
        )
        with pytest.raises(ValueError):
            config.get_deep_scan_calibration()

        # lo <= 0 -> invalid
        self._patched_weights(
            monkeypatch,
            {"representation": "calibrated_mean", "mean_anchor_lo": 0.0, "mean_anchor_hi": 0.96},
        )
        with pytest.raises(ValueError):
            config.get_deep_scan_calibration()

        # hi > 1 -> invalid
        self._patched_weights(
            monkeypatch,
            {"representation": "calibrated_mean", "mean_anchor_lo": 0.93, "mean_anchor_hi": 1.5},
        )
        with pytest.raises(ValueError):
            config.get_deep_scan_calibration()

    def test_calibrated_mean_valid_anchors_pass(self, monkeypatch):
        self._patched_weights(
            monkeypatch,
            {"representation": "calibrated_mean", "mean_anchor_lo": 0.93, "mean_anchor_hi": 0.96},
        )
        calibration = config.get_deep_scan_calibration()
        assert calibration["representation"] == "calibrated_mean"
        assert calibration["mean_anchor_lo"] == 0.93
        assert calibration["mean_anchor_hi"] == 0.96


def _setup_modal(monkeypatch, chunk_scores):
    """chunk_scores here are the intended PER-SENTENCE aggregated scores. Since
    the doc-level path now scores 3-sentence WINDOWS and aggregates back
    (2026-07-14 windowing hotfix), the fake endpoint answers per-WINDOW; with a
    uniform per-window score the aggregate per-sentence mean equals it, so the
    doc-mean expectations below hold when chunk_scores are uniform — and for the
    legacy-proportion test we return one score per window such that the flagged
    fraction is preserved by echoing the requested list sized to the payload."""
    monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
    sentences = [f"Sentence number {i} has enough words." for i in range(len(chunk_scores))]
    monkeypatch.setattr(
        pipeline_bridge, "split_sentences", lambda text: sentences
    )

    def fake_call(chunks, *a, **k):
        # answer per submitted chunk; cycle the requested scores so uniform
        # lists stay uniform and mixed lists keep their composition roughly
        out = [chunk_scores[i % len(chunk_scores)] for i in range(len(chunks))]
        return {"available": True, "calibrated": True, "chunk_scores": out}

    monkeypatch.setattr(pipeline_bridge.modal_client, "call_deep_scan", fake_call)


class TestGetDeepScanProportionRepresentation:
    def test_calibrated_mean_below_lo_clips_to_zero(self, monkeypatch):
        calibration = dict(config.get_deep_scan_calibration())
        calibration.update(representation="calibrated_mean", mean_anchor_lo=0.93, mean_anchor_hi=0.96)
        monkeypatch.setattr(config, "get_deep_scan_calibration", lambda: calibration)
        _setup_modal(monkeypatch, [0.5, 0.6, 0.7])  # mean well below lo

        out = pipeline_bridge.get_deep_scan_proportion({"document_text": "doc text here."})
        assert out is not None
        assert out["proportion"] == 0.0
        assert out["payload"]["proportion"] == 0.0
        assert out["below_floor"] is True

    def test_calibrated_mean_above_hi_clips_to_one(self, monkeypatch):
        calibration = dict(config.get_deep_scan_calibration())
        calibration.update(representation="calibrated_mean", mean_anchor_lo=0.93, mean_anchor_hi=0.96)
        monkeypatch.setattr(config, "get_deep_scan_calibration", lambda: calibration)
        _setup_modal(monkeypatch, [0.99, 0.995, 0.999])  # mean well above hi

        out = pipeline_bridge.get_deep_scan_proportion({"document_text": "doc text here."})
        assert out is not None
        assert out["proportion"] == 1.0
        assert out["payload"]["proportion"] == 1.0
        assert out["below_floor"] is False

    def test_calibrated_mean_midpoint_gives_half(self, monkeypatch):
        calibration = dict(config.get_deep_scan_calibration())
        calibration.update(representation="calibrated_mean", mean_anchor_lo=0.90, mean_anchor_hi=1.00)
        monkeypatch.setattr(config, "get_deep_scan_calibration", lambda: calibration)
        # mean must be exactly (0.90+1.00)/2 = 0.95
        _setup_modal(monkeypatch, [0.95, 0.95, 0.95])

        out = pipeline_bridge.get_deep_scan_proportion({"document_text": "doc text here."})
        assert out is not None
        assert out["proportion"] == pytest.approx(0.5)
        assert out["payload"]["proportion"] == pytest.approx(0.5)

    def test_calibrated_mean_below_floor_uses_doc_floor(self, monkeypatch):
        calibration = dict(config.get_deep_scan_calibration())
        calibration.update(
            representation="calibrated_mean", mean_anchor_lo=0.90, mean_anchor_hi=1.00,
            doc_floor=0.6,
        )
        monkeypatch.setattr(config, "get_deep_scan_calibration", lambda: calibration)
        # mean 0.95 -> signal 0.5, below doc_floor=0.6
        _setup_modal(monkeypatch, [0.95, 0.95, 0.95])

        out = pipeline_bridge.get_deep_scan_proportion({"document_text": "doc text here."})
        assert out["below_floor"] is True

    def test_proportion_representation_reproduces_legacy_math(self, monkeypatch):
        calibration = dict(config.get_deep_scan_calibration())
        calibration["representation"] = "proportion"
        monkeypatch.setattr(config, "get_deep_scan_calibration", lambda: calibration)
        sent_threshold = calibration["sent_threshold"]
        # Post-windowing-hotfix (2026-07-14): the doc-level signal is computed
        # over WINDOW-AGGREGATED per-sentence scores. Legacy parity now means:
        # the proportion branch counts threshold-crossers over those aggregated
        # scores (not the mean branch). Reproduce the pipeline's aggregation
        # with the same cycling fake to derive the expectation.
        from detect.deberta_windowing import aggregate, build_windows

        chunk_scores = [sent_threshold + 0.001, sent_threshold - 0.5, sent_threshold + 0.01]
        _setup_modal(monkeypatch, chunk_scores)
        sentences = [f"Sentence number {i} has enough words." for i in range(len(chunk_scores))]
        windows = build_windows(sentences, size=3, step=1)
        window_scores = [chunk_scores[i % len(chunk_scores)] for i in range(len(windows))]
        agg = [x for x in aggregate(sentences, windows, window_scores, size=3, step=1)["sentence_scores"] if x is not None]
        expected = sum(1 for s in agg if s >= sent_threshold) / len(agg)

        out = pipeline_bridge.get_deep_scan_proportion({"document_text": "doc text here."})
        assert out["proportion"] == pytest.approx(expected)
        assert out["payload"]["proportion"] == pytest.approx(expected)


def test_calibrated_mean_scores_windowed_sentences_not_fragments(monkeypatch):
    """PROD INCIDENT 2026-07-14 (scan cf893c09, fused 0.0): the doc-level call
    sent RAW sentences to Modal, but the calibrated-mean anchors were fit on
    WINDOWED scores (3-sentence windows aggregated back to sentences — the unit
    every eval used, and the unit the heatmap path already scores). Isolated
    short fragments score near 0 on the window-trained model and crater the
    mean below the lo anchor (casual doc: windowed mean 0.996 vs raw 0.9198,
    under the ESL max). The doc-level path MUST window + aggregate like the
    harness: with 5 sentences and a fake Modal endpoint, the payload must be
    the 3-sentence windows (n-size+1 = 3 chunks), and the signal must equal
    the calibrated mean of the aggregated per-sentence scores."""
    from detect_v7 import config as v7config, pipeline_bridge
    from detect.deberta_windowing import build_windows, aggregate

    monkeypatch.setenv("DRAFTPROOF_V7_DEEP_SCAN", "1")
    sents = ["Alpha beta gamma delta epsilon zeta.",
             "Second sentence with several words here.",
             "Short one.",
             "Fourth sentence contains a normal number of words.",
             "Fifth sentence also has plenty of words in it."]
    doc = " ".join(sents)
    captured = {}

    def fake_call(chunks, *a, **k):
        captured["chunks"] = list(chunks)
        return {"available": True, "calibrated": True,
                "chunk_scores": [0.99] * len(chunks)}

    monkeypatch.setattr(pipeline_bridge.modal_client, "call_deep_scan", fake_call)
    real = v7config.get_deep_scan_calibration

    def patched():
        cfg = real()
        cfg.update({"representation": "calibrated_mean",
                    "mean_anchor_lo": 0.93, "mean_anchor_hi": 0.96})
        return cfg

    monkeypatch.setattr(v7config, "get_deep_scan_calibration", patched)
    out = pipeline_bridge.get_deep_scan_proportion({"document_text": doc})
    assert out is not None
    expected_windows = build_windows(sents, size=3, step=1)
    assert captured["chunks"] == expected_windows  # windowed payload, not raw sentences
    # all windows 0.99 -> every sentence aggregates to 0.99 -> mean 0.99 ->
    # signal = (0.99-0.93)/0.03 clipped to 1.0
    assert out["proportion"] == pytest.approx(1.0)
