"""Regression test for the 2026-08-01 Fable follow-up #1: when the internal no-regression guard
reverts to the original text (DRAFTPROOF_V6_REGRESSION_GUARD_ADVISORY=0), `strict_safe_band_achieved`
and `kpi_finalization_status` were computed BEFORE the revert and could still claim a strict-safe
rewrite shipped even though the ORIGINAL document was what actually shipped. Fixed in
poc/rewrite_v6/production.py by forcing `changed`/`cleared` false inside the revert branch."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from poc.rewrite_v6 import production as v6_production


@pytest.fixture(autouse=True)
def _use_legacy_adapter_path(monkeypatch):
    # Legacy adapter path (run_v6_rewrite_all) is the simplest surface to drive detect_scores
    # through deterministically for this test; the fix under test lives entirely downstream of
    # either rewrite path in run_rewrite_pipeline_v6.
    monkeypatch.setattr(v6_production, "direct_rewrite_enabled", lambda: False)


def _report(text: str, score: float) -> dict:
    return {
        "input_text": text,
        "ai_score": score,
        "findings": [] if score < 50 else {"high": [{"finding_id": "f1"}]},
        "ai_risk_badge": {
            "ai_likelihood_score": score,
            "external_detector_estimate": {"score": score, "band": "high" if score >= 75 else "low"},
            "transformation_classification": {"label": "test"},
        },
        "scan_intelligence": {
            "transformation": {"contribution": {"human": 100 - score}, "core_signals": [{"key": "test"}]},
        },
    }


def test_regression_guard_revert_clears_strict_safe_claim(tmp_path, monkeypatch):
    """DRAFTPROOF_V6_REGRESSION_GUARD_ADVISORY=0 -> our detector regressed badly -> guard reverts to
    the original text. The shipped document IS the original, so strict_safe_band_achieved must be
    False and kpi_finalization_status must NOT claim strict_safe_auto_finalized."""
    monkeypatch.setenv("DRAFTPROOF_V6_REGRESSION_GUARD_ADVISORY", "0")
    original = "Original classroom note with stable detector risk and no findings at all here."
    rewritten = "Candidate classroom note with severe worse detector risk introduced by the rewrite."
    from poc.rewrite_v6.scan import scan_text

    scan = scan_text(original)

    def fake_run_v6_rewrite_all(_text, **_kwargs):
        return SimpleNamespace(
            initial_scan=scan,
            final_scan=scan_text(rewritten),
            passes=[],
            pass_trace=[],
            rewritten_text=rewritten,
            final_text_before_quality_repair=None,
            quality_repair=None,
        )

    def fake_scan_report(text, **_kwargs):
        # original scores low (13.38), rewritten scores much higher (66.05) -> incident 5bacaeb3 shape
        score = 13.38 if text == original else 66.05
        return _report(text, score)

    monkeypatch.setattr(v6_production, "run_v6_rewrite_all", fake_run_v6_rewrite_all)
    monkeypatch.setattr(v6_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))
    monkeypatch.setattr(v6_production, "render_rewrite_report", lambda **_kwargs: "# Rewrite")
    monkeypatch.setattr(
        v6_production, "_sentence_comparison",
        lambda original_text, final_text: [{"original": original_text, "final": final_text}],
    )
    monkeypatch.setattr(v6_production, "_scan_report_for_summary", fake_scan_report)

    result = v6_production.run_rewrite_pipeline_v6(
        detect_json={"input_text": original},
        output_dir=str(tmp_path),
        model="writer-model",
    )
    summary = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert summary["status"] == "original_preserved_regression_guard"
    assert summary["final_text"] == original
    assert summary["internal_regression_guard"]["reverted"] is True
    # The bug: these two used to still say "strict safe" / "auto finalized" post-revert.
    assert summary["strict_safe_band_achieved"] is False
    assert summary["kpi_finalization_status"] != "strict_safe_auto_finalized"
    assert summary["kpi_finalization_status"] == "original_preserved_regression_guard"
