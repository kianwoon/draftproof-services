"""Regression test for the V7 double-fusion wiring bug.

When ``tier_authority`` is ON, ``ai_risk_badge["ai_likelihood_score"]`` is the
FUSED authority score (0.4*composite + 0.6*deberta). Feeding that as the V7
category breakdown's ``composite`` detector input double-fuses it with
``deberta_large`` inside ``run_v7_breakdown``. The fix threads the RAW
``pre_fusion_composite`` from ``report/builder.py`` and makes
``_extract_calibrated_score`` PREFER it, falling back byte-identically to
``ai_likelihood_score`` when the key is absent / None / non-numeric (the
non-tier-authority / quick-scan / offline-tuning paths).

Run from poc/:  python -m pytest test_double_fusion_fix.py -q
"""
import os

import pytest

from detect_v7 import pipeline_bridge
from detect_v7.pipeline_bridge import _extract_calibrated_score, run_v7_breakdown


# --- _extract_calibrated_score: preference + byte-identical fallback ----------

def test_prefers_pre_fusion_composite_over_fused_score():
    # ai_likelihood_score = 70 (the FUSED authority), pre_fusion_composite = 40
    # (the RAW composite). The bridge must use the RAW composite.
    result = {"ai_likelihood_score": 70.0, "pre_fusion_composite": 40.0}
    assert _extract_calibrated_score(result) == pytest.approx(0.40)


def test_falls_back_to_ai_likelihood_when_key_absent():
    # Non-tier-authority / quick-scan path: key not present -> unchanged behavior.
    result = {"ai_likelihood_score": 70.0}
    assert _extract_calibrated_score(result) == pytest.approx(0.70)


def test_falls_back_when_pre_fusion_is_none():
    # builder.py sends pre_fusion_composite=None when tier-authority is OFF.
    result = {"ai_likelihood_score": 70.0, "pre_fusion_composite": None}
    assert _extract_calibrated_score(result) == pytest.approx(0.70)


def test_zero_pre_fusion_composite_is_used_not_skipped():
    # 0.0 is falsy but a VALID composite (a genuinely human-looking document can
    # score 0). It must be USED, not treated as absent — otherwise the fused
    # ai_likelihood_score (70) would leak back in and re-introduce double fusion.
    result = {"ai_likelihood_score": 70.0, "pre_fusion_composite": 0.0}
    assert _extract_calibrated_score(result) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", ["40", True, [], {}])
def test_falls_back_when_pre_fusion_invalid(bad):
    result = {"ai_likelihood_score": 70.0, "pre_fusion_composite": bad}
    assert _extract_calibrated_score(result) == pytest.approx(0.70)


def test_sub_one_percent_composite_is_not_misread_as_already_normalized():
    # Regression: a genuine very-human document can score e.g. 0.5 on the 0-100
    # scale (0.5% AI-likelihood). The old `if value > 1.0: /100` heuristic left
    # values <=1.0 UNDIVIDED, so 0.5 (meaning 0.5%) was silently read as 0.5 =
    # 50% by detector_fusion — a ~100x inflation for exactly the clean/ESL-safe
    # documents this score most needs to get right. Must always divide by 100.
    result = {"ai_likelihood_score": 70.0, "pre_fusion_composite": 0.5}
    assert _extract_calibrated_score(result) == pytest.approx(0.005)


def test_returns_none_when_no_score_at_all():
    assert _extract_calibrated_score({}) is None


# --- run_v7_breakdown: the composite fed to detector_fusion is pre-fusion -----

@pytest.fixture
def _v7_enabled_no_modal(monkeypatch):
    """Enable the breakdown, disable deep scan (no Modal call), and capture the
    ``detector_scores`` handed to ``compute_calibrated_detector_score``.
    """
    monkeypatch.setenv("DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN", "1")
    monkeypatch.delenv("DRAFTPROOF_V7_DEEP_SCAN", raising=False)
    captured = {}

    def _fake_fuse(detector_scores, *a, **k):
        captured["detector_scores"] = dict(detector_scores)
        return 0.5, {}

    monkeypatch.setattr(
        pipeline_bridge.detector_fusion,
        "compute_calibrated_detector_score",
        _fake_fuse,
    )
    return captured


def _base_result(**extra):
    # Minimal badge-shaped dict; components empty so run_v7_breakdown returns
    # None AFTER the fusion call we care about (fusion happens first).
    result = {"ai_likelihood_score": 70.0, "document_text": "x", "_precomputed_deep_scan": None}
    result.update(extra)
    return result


def test_breakdown_composite_input_is_pre_fusion(_v7_enabled_no_modal):
    run_v7_breakdown(_base_result(pre_fusion_composite=40.0))
    assert _v7_enabled_no_modal["detector_scores"]["composite"] == pytest.approx(0.40)


def test_breakdown_composite_input_falls_back_when_key_absent(_v7_enabled_no_modal):
    # Byte-identical to pre-fix behavior: composite == normalized ai_likelihood.
    run_v7_breakdown(_base_result())
    assert _v7_enabled_no_modal["detector_scores"]["composite"] == pytest.approx(0.70)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
