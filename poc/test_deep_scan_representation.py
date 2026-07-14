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
    monkeypatch.setenv(_DEEP_SCAN_ENV_VAR, "1")
    sentences = [f"Sentence number {i}." for i in range(len(chunk_scores))]
    monkeypatch.setattr(
        pipeline_bridge, "split_sentences", lambda text: sentences
    )
    monkeypatch.setattr(
        pipeline_bridge.modal_client, "call_deep_scan",
        lambda sents: {"available": True, "calibrated": True, "chunk_scores": chunk_scores},
    )


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
        chunk_scores = [sent_threshold + 0.001, sent_threshold - 0.5, sent_threshold + 0.01]
        _setup_modal(monkeypatch, chunk_scores)

        expected = sum(1 for s in chunk_scores if s >= sent_threshold) / len(chunk_scores)

        out = pipeline_bridge.get_deep_scan_proportion({"document_text": "doc text here."})
        assert out["proportion"] == pytest.approx(expected)
        assert out["payload"]["proportion"] == pytest.approx(expected)
