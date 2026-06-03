from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from poc.rewrite_v6 import production as v6_production
from poc.rewrite_v6.pipeline import _dynamic_pass_limit
from poc.rewrite_v6.scan import scan_text, scan_text_with_report


@pytest.fixture(autouse=True)
def _use_legacy_adapter_path(monkeypatch):
    monkeypatch.setattr(v6_production, "direct_rewrite_enabled", lambda: False)


def test_v6_production_adapter_uses_configured_three_pass_budget(tmp_path, monkeypatch):
    captured: dict[str, int] = {}

    def fake_run_v6_rewrite_all(text, **kwargs):
        captured["max_passes"] = kwargs["max_passes"]
        scan = scan_text(text)
        return SimpleNamespace(
            initial_scan=scan,
            final_scan=scan,
            passes=[],
            pass_trace=[],
            rewritten_text=text,
            final_text_before_quality_repair=None,
            quality_repair=None,
        )

    monkeypatch.setenv("DRAFTPROOF_V6_MAX_PASSES", "3")
    monkeypatch.setattr(v6_production, "run_v6_rewrite_all", fake_run_v6_rewrite_all)
    monkeypatch.setattr(v6_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))
    monkeypatch.setattr(v6_production, "render_rewrite_report", lambda **_kwargs: "# Rewrite")
    monkeypatch.setattr(v6_production, "_sentence_comparison", lambda _original, _final: [])
    monkeypatch.setattr(v6_production, "_scan_report_for_summary", lambda text, **_kwargs: v6_production._scan_report_shape(scan_text(text).to_dict()))

    result = v6_production.run_rewrite_pipeline_v6(
        detect_json={"input_text": "The review moved from intake to approval."},
        output_dir=str(tmp_path),
        model="writer-model",
    )
    summary = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert captured["max_passes"] == 3
    assert summary["rewrite_effective_config"]["max_passes"] == 3


def test_v6_production_adapter_defaults_to_dynamic_pass_budget(tmp_path, monkeypatch):
    captured: dict[str, int | None] = {}

    def fake_run_v6_rewrite_all(text, **kwargs):
        captured["max_passes"] = kwargs["max_passes"]
        scan = scan_text(text)
        return SimpleNamespace(
            initial_scan=scan,
            final_scan=scan,
            passes=[],
            pass_trace=[],
            rewritten_text=text,
            final_text_before_quality_repair=None,
            quality_repair=None,
        )

    monkeypatch.delenv("DRAFTPROOF_V6_MAX_PASSES", raising=False)
    monkeypatch.setattr(v6_production, "run_v6_rewrite_all", fake_run_v6_rewrite_all)
    monkeypatch.setattr(v6_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))
    monkeypatch.setattr(v6_production, "render_rewrite_report", lambda **_kwargs: "# Rewrite")
    monkeypatch.setattr(v6_production, "_sentence_comparison", lambda _original, _final: [])
    monkeypatch.setattr(v6_production, "_scan_report_for_summary", lambda text, **_kwargs: v6_production._scan_report_shape(scan_text(text).to_dict()))

    result = v6_production.run_rewrite_pipeline_v6(
        detect_json={"input_text": "The review moved from intake to approval."},
        output_dir=str(tmp_path),
        model="writer-model",
    )
    summary = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert captured["max_passes"] is None
    assert summary["rewrite_effective_config"]["max_passes"] is None


def test_v6_dynamic_pass_budget_covers_each_finding_paragraph(monkeypatch):
    scan = SimpleNamespace(
        findings=[
            SimpleNamespace(paragraph_id="p001"),
            SimpleNamespace(paragraph_id="p002"),
            SimpleNamespace(paragraph_id="p003"),
            SimpleNamespace(paragraph_id="p004"),
        ]
    )

    monkeypatch.setenv("DRAFTPROOF_V6_MAX_DYNAMIC_PASSES", "2")

    assert _dynamic_pass_limit(scan) == 4


def test_v6_scan_uses_report_identity_without_losing_full_paragraph_text():
    text = (
        "Opening sentence. Report sentence one. Missing from report.\n\n"
        "Second paragraph starts. Report sentence two. Another missing sentence."
    )
    report = {
        "input_text": text,
        "scan_intelligence": {
            "document": {
                "paragraphs": [
                    {"paragraph_id": "p001", "text": "Report sentence one."},
                    {"paragraph_id": "p002", "text": "Report sentence two."},
                ]
            }
        },
        "sentence_map": {
            "s010": {"paragraph_id": "p001", "text": "Report sentence one."},
            "s020": {"paragraph_id": "p002", "text": "Report sentence two."},
        },
        "highlight_segments": [
            {
                "sentence_id": "s020",
                "paragraph_id": "p002",
                "text": "Report sentence two.",
                "signals": [
                    {
                        "finding_id": "f020",
                        "title": "medium_predictability",
                        "scanner": "predictability",
                        "score": 52,
                        "tier": "medium",
                    }
                ],
            }
        ],
    }

    scan = scan_text_with_report(text, report)

    assert [paragraph.id for paragraph in scan.paragraphs] == ["p001", "p002"]
    assert "Opening sentence" in scan.paragraphs[0].text
    assert "Missing from report" in scan.paragraphs[0].text
    assert "Another missing sentence" in scan.paragraphs[1].text
    assert any(sentence.id == "s010" for sentence in scan.paragraphs[0].sentences)
    assert any(sentence.id == "s020" for sentence in scan.paragraphs[1].sentences)
    assert any(finding.sentence_id == "s020" for finding in scan.findings)


def test_v6_production_passes_scan_json_aligned_source_scan(tmp_path, monkeypatch):
    captured = {}
    text = "Opening sentence. Report sentence one.\n\nSecond paragraph starts. Report sentence two."

    def fake_run_v6_rewrite_all(text_arg, **kwargs):
        captured["source_scan"] = kwargs["source_scan"]
        scan = kwargs["source_scan"]
        return SimpleNamespace(
            initial_scan=scan,
            final_scan=scan,
            passes=[],
            pass_trace=[],
            rewritten_text=text_arg,
            final_text_before_quality_repair=None,
            quality_repair=None,
        )

    report = {
        "input_text": text,
        "scan_intelligence": {
            "document": {
                "paragraphs": [
                    {"paragraph_id": "p001", "text": "Report sentence one."},
                    {"paragraph_id": "p002", "text": "Report sentence two."},
                ]
            }
        },
        "sentence_map": {
            "s010": {"paragraph_id": "p001", "text": "Report sentence one."},
            "s020": {"paragraph_id": "p002", "text": "Report sentence two."},
        },
        "highlight_segments": [
            {
                "sentence_id": "s020",
                "paragraph_id": "p002",
                "signals": [{"finding_id": "f020", "title": "medium_predictability", "score": 52}],
            }
        ],
    }

    monkeypatch.setattr(v6_production, "run_v6_rewrite_all", fake_run_v6_rewrite_all)
    monkeypatch.setattr(v6_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))
    monkeypatch.setattr(v6_production, "render_rewrite_report", lambda **_kwargs: "# Rewrite")
    monkeypatch.setattr(v6_production, "_sentence_comparison", lambda _original, _final: [])
    monkeypatch.setattr(v6_production, "_scan_report_for_summary", lambda text, **_kwargs: v6_production._scan_report_shape(scan_text(text).to_dict()))

    result = v6_production.run_rewrite_pipeline_v6(
        detect_json=report,
        output_dir=str(tmp_path),
        model="writer-model",
    )
    summary = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert captured["source_scan"].paragraphs[1].id == "p002"
    assert any(finding.sentence_id == "s020" for finding in captured["source_scan"].findings)
    assert summary["rewrite_effective_config"]["source_scan"] == "scan_json_aligned"


def test_v6_external_guard_preserves_original_on_severe_regression(tmp_path, monkeypatch):
    original = "Original classroom note with stable detector risk."
    rewritten = "Candidate classroom note with worse detector risk."
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
        score = 52 if text == original else 86
        return _report_with_external_score(text, score)

    monkeypatch.setattr(v6_production, "run_v6_rewrite_all", fake_run_v6_rewrite_all)
    monkeypatch.setattr(v6_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))
    monkeypatch.setattr(v6_production, "render_rewrite_report", lambda **_kwargs: "# Rewrite")
    monkeypatch.setattr(v6_production, "_sentence_comparison", lambda original_text, final_text: [{"original": original_text, "final": final_text}])
    monkeypatch.setattr(v6_production, "_scan_report_for_summary", fake_scan_report)

    result = v6_production.run_rewrite_pipeline_v6(
        detect_json={"input_text": original},
        output_dir=str(tmp_path),
        model="writer-model",
    )
    summary = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert summary["status"] == "original_preserved_external_guard"
    assert summary["final_text"] == original
    assert summary["external_detector_guard"]["blocked"] is True
    assert summary["external_detector_guard"]["candidate_score"] == 86
    assert summary["external_detector_guard"]["original_score"] == 52
    assert any(row.get("selected_source") == "external_detector_guard" for row in summary["v6_pass_trace"])


def test_v6_external_guard_allows_non_regression(tmp_path, monkeypatch):
    original = "Original classroom note with stable detector risk."
    rewritten = "Candidate classroom note with lower detector risk."
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
        score = 70 if text == original else 62
        return _report_with_external_score(text, score)

    monkeypatch.setattr(v6_production, "run_v6_rewrite_all", fake_run_v6_rewrite_all)
    monkeypatch.setattr(v6_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))
    monkeypatch.setattr(v6_production, "render_rewrite_report", lambda **_kwargs: "# Rewrite")
    monkeypatch.setattr(v6_production, "_sentence_comparison", lambda _original, _final: [])
    monkeypatch.setattr(v6_production, "_scan_report_for_summary", fake_scan_report)

    result = v6_production.run_rewrite_pipeline_v6(
        detect_json={"input_text": original},
        output_dir=str(tmp_path),
        model="writer-model",
    )
    summary = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert summary["final_text"] == rewritten
    assert summary["external_detector_guard"]["blocked"] is False
    assert summary["external_detector_guard"]["candidate_score"] == 62


def _report_with_external_score(text: str, score: float) -> dict:
    return {
        "input_text": text,
        "ai_score": score,
        "findings": [],
        "ai_risk_badge": {
            "ai_likelihood_score": score,
            "external_detector_estimate": {"score": score, "band": "high" if score >= 75 else "low"},
            "transformation_classification": {"label": "test"},
        },
        "scan_intelligence": {
            "transformation": {
                "contribution": {"human": 50},
                "core_signals": [{"key": "test"}],
            }
        },
    }
