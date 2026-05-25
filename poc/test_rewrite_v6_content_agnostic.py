from __future__ import annotations

import json
from pathlib import Path

from poc.rewrite_v6.pipeline import run_v6_rewrite, run_v6_rewrite_all
from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6 import production as v6_production
from poc.rewrite_v6.scan import scan_text
from poc.rewrite_v6.write import build_prompt
from poc.report.render_rewrite import render_rewrite_report


class FakeResponse:
    content = '{"variants":[]}'
    raw_content = content


class FakeClient:
    calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return FakeResponse()


class BadJsonResponse:
    content = '{"variants":[{"id":"v1","text":"broken"}'
    raw_content = content


class BadJsonClient:
    def __init__(self):
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return BadJsonResponse()


class StaticJsonResponse:
    def __init__(self, content: str):
        self.content = content
        self.raw_content = content


class StaticJsonClient:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return StaticJsonResponse(self.content)


def sample_text() -> str:
    return (
        "The intake process is built around a form, a queue, and a review. "
        "Clients submit details, wait for confirmation, and receive a response."
    )


def test_v6_prompt_uses_source_terms_and_not_fixed_domain_starts():
    scan = scan_text(sample_text())
    paragraph, plan = build_plan(scan)
    prompt = build_prompt(paragraph, plan)
    assert "source_terms" in prompt
    assert "form" in prompt
    assert "Start with" not in prompt


def test_v6_rewrite_preserves_source_when_writer_returns_no_candidate():
    client = FakeClient()
    result = run_v6_rewrite(sample_text(), writer_client=client)
    assert result.selected is not None
    assert result.selected.source == "source_preserved"
    assert "form" in result.selected.text
    assert "form" in result.rewritten_text
    assert client.calls == 1


def test_v6_bad_writer_json_preserves_source_instead_of_compiling():
    client = BadJsonClient()
    source = "This result shows an important concern because the process should improve."
    result = run_v6_rewrite(source, writer_client=client)
    assert result.selected is not None
    assert result.selected.source == "source_preserved"
    assert result.rewritten_text == source
    assert client.calls == 1


def test_v6_files_stay_below_1000_lines_and_do_not_import_v5():
    root = Path(__file__).resolve().parent / "rewrite_v6"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert len(text.splitlines()) < 1000, path
        assert "rewrite_v5" not in text, path


def test_v6_json_result_is_serializable():
    result = run_v6_rewrite("A process uses a step, a check, and a result.", writer_client=FakeClient())
    json.dumps(result.to_dict())


def test_v6_production_adapter_returns_worker_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(v6_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))
    monkeypatch.setattr(v6_production, "_scan_report", fake_full_scan_report)
    result = v6_production.run_rewrite_pipeline_v6(
        detect_json={"input_text": "The review moved from intake to approval."},
        output_dir=str(tmp_path),
        model="qwen/qwen3-30b-a3b-instruct-2507",
    )
    summary = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert result["status"] in {"ai_mitigated", "partial_candidate_not_strict_safe", "original_preserved"}
    assert Path(result["md_path"]).exists()
    assert Path(result["pdf_path"]).exists()
    assert summary["rewrite_pipeline_version"] == "rewrite_v6_scanner_planner_writer"
    assert summary["rewrite_effective_config"]["model"] == "qwen/qwen3-30b-a3b-instruct-2507"
    assert summary["final_text"]
    assert summary["detect_scores"]["original_ai_authorship"] == 2
    assert summary["detect_scores"]["rewritten_ai_authorship"] == 2
    assert summary["detect_scores"]["original_human_contribution"] == 96
    assert summary["detect_scores"]["rewritten_human_contribution"] == 96
    assert summary["detect_scores"]["original_ai_transformation"] == 4
    assert summary["detect_scores"]["rewritten_ai_transformation"] == 4
    assert summary["detect_scores"]["human_shift_score"] == 0
    assert summary["original_risk"] == 2
    assert summary["final_risk"] == 2
    rewritten_scan = summary["detect_scan_rewritten"]
    rewritten_badge = rewritten_scan["ai_risk_badge"]
    rewritten_intelligence = rewritten_scan["scan_intelligence"]["transformation"]
    assert rewritten_badge["transformation_classification"]["code"] == "low_ai_signal"
    assert rewritten_badge["ai_components"]["topk_calibrated_risk"] == 4
    assert rewritten_intelligence["contribution"]["human_contribution_ratio"] == 96
    assert rewritten_intelligence["core_signals"][0]["key"] == "topk_calibrated_risk"
    assert rewritten_scan["integrity_layers"]["layers"]["human_contribution_signal"]["score"] == 96


def test_v6_rewrite_pdf_metrics_use_detect_scores_when_scan_badges_are_compact():
    summary = {
        "outcome": "ai_mitigated",
        "status": "ai_mitigated",
        "detect_scores": {
            "original_ai_authorship": 70,
            "rewritten_ai_authorship": 30,
            "original_human_contribution": 56,
            "rewritten_human_contribution": 83,
            "original_ai_transformation": 44,
            "rewritten_ai_transformation": 17,
            "original_grounding_quality_risk": 90,
            "rewritten_grounding_quality_risk": 25,
        },
        "detect_scan_original": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {"ai_likelihood_score": 0, "writing_quality_score": 0},
        },
        "detect_scan_rewritten": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {"ai_likelihood_score": 0, "writing_quality_score": 0},
        },
    }
    report = render_rewrite_report(summary, [], [], original_text="Original.", final_text="Rewritten.")
    assert "| **AI Likelihood** | `35%` | `15%` | `-20%` |" in report
    assert "| **Human Contribution** | `56%` | `83%` | `+27%` |" in report
    assert "| **AI Transformation** | `44%` | `17%` | `-27%` |" in report
    assert "| **Grounding Quality Risk** | `90.00%` | `25.00%` | `-65.00%` |" in report
    assert "15% calibrated risk · below 20% reference" in report


def fake_full_scan_report(text: str) -> dict:
    return {
        "input_text": text,
        "findings": {"critical": [], "high": [], "medium": [], "low": []},
        "ai_score": 2,
        "writing_score": 3,
        "ai_risk_badge": {
            "ai_likelihood_score": 2,
            "writing_quality_score": 3,
            "tier": "green",
            "authorship_rating": {"label": "Good", "short_label": "Good", "code": "low_ai_signal"},
            "authorship_rating_label": "Good",
            "ai_components": {
                "topk_pattern_raw": 7,
                "topk_calibrated_risk": 4,
                "topk_calibration_eligible": True,
            },
            "transformation_classification": {
                "code": "low_ai_signal",
                "label": "Human contribution pattern",
                "confidence": "high",
                "features": {
                    "ai_likelihood": 2,
                    "human_anchor_score": 96,
                    "calibrated_ai_risk": 2,
                    "topk_calibrated_risk": 4,
                },
            },
        },
        "scan_intelligence": {
            "transformation": {
                "contribution": {
                    "human_contribution_ratio": 96,
                    "ai_transformation_ratio": 4,
                    "calibrated_ai_risk": 2,
                    "human_anchor_discount": 70,
                    "calibration_confidence": 92,
                    "reporting_suppression": 15,
                    "summary": "Human anchoring dominates the rewritten scan.",
                },
                "core_signals": [
                    {
                        "key": "topk_calibrated_risk",
                        "label": "Calibrated Top-k Risk",
                        "score": 4,
                    }
                ],
            }
        },
        "integrity_layers": {
            "layers": {
                "human_contribution_signal": {"score": 96},
                "ai_transformation_risk": {"score": 4},
            }
        },
    }


def test_v6_finding_methods_are_exposed_to_writer_contract():
    text = (
        "This result shows an important concern because the process should improve. "
        "However, therefore, the operation uses a review, a handoff, and a final check. "
        "According to Smith (2020), the method supports the outcome."
    )
    scan = scan_text(text)
    paragraph, plan = build_plan(scan)
    prompt = build_prompt(paragraph, plan)
    methods = {action.method for action in plan.actions}
    assert methods
    for method in methods:
        assert f'"{method}"' in prompt
    assert "author_proxy_policy" in prompt
    assert "author_proxy_provenance" in prompt
    assert "author_review_items" in prompt


def test_v6_author_proxy_prompt_does_not_block_reviewable_bridges():
    scan = scan_text("This result shows an important concern because the process should improve.")
    paragraph, plan = build_plan(scan)
    prompt = build_prompt(paragraph, plan)
    assert "source-derived terms only" not in prompt
    assert "record its provenance" in prompt
    assert "author-review provenance" in prompt


def test_v6_scanner_does_not_count_parenthetical_citation_commas_as_lists():
    scan = scan_text("Participants reviewed the service before choosing a provider (Riley et al., 2024).")
    assert not any("packed_list" in finding.tags for finding in scan.findings)


def test_v6_scanner_treats_parenthetical_citation_as_smoothing_support():
    scan = scan_text("The monthly review programme can evaluate staff attentiveness during the service experience (Riley, 2024).")
    assert not any("paraphrase_smoothing" in finding.tags for finding in scan.findings)


def test_v6_scanner_does_not_treat_local_must_as_unsupported_gap():
    scan = scan_text("The method gives participants a concrete way to grasp the pattern before they must name it.")
    assert not any("unsupported_claim_gap" in finding.tags for finding in scan.findings)


def test_v6_scanner_does_not_count_single_context_comma_as_list_density():
    scan = scan_text("During the first stage, the process moved from intake to review.")
    assert not any("packed_list" in finding.tags for finding in scan.findings)


def test_v6_scanner_does_not_count_single_contrast_comma_as_list_density():
    scan = scan_text("The result changed the process, not the whole service.")
    assert not any("packed_list" in finding.tags for finding in scan.findings)


def test_v6_scanner_does_not_overflag_short_result_sentence():
    scan = scan_text("They may experience frustration.")
    assert not scan.findings


def test_v6_scanner_does_not_mark_named_method_as_smooth_paraphrase():
    scan = scan_text("The Alpha Method gives participants a concrete way to grasp projection angles before naming them.")
    assert not any("paraphrase_smoothing" in finding.tags for finding in scan.findings)


def test_v6_scanner_flags_repeated_sentence_frames():
    scan = scan_text(
        "The review process changed after intake. "
        "Clients also use the portal. "
        "Clients also use phone support. "
        "Clients also use the front desk. "
        "The final step checks the result."
    )
    repeated = [finding for finding in scan.findings if "repeated_sentence_frame" in finding.tags]
    assert len(repeated) == 3
    assert {finding.evidence["repeated_frame"] for finding in repeated} == {"clients also use"}


def test_v6_selection_prefers_writer_over_source_when_writer_improves_route():
    source = (
        "The intake system is changing faster than many teams can manage. "
        "In the past, the process was built around the form, the queue, and the reviewer. "
        "Clients submitted details, waited for confirmation, and received a response. "
        "That model still exists, but it no longer fits every intake path today."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "Teams now handle intake in conditions that the older review model only partly explains. "
                    "The earlier process still matters because forms, queues, reviewers, confirmation, and responses gave clients a clear route through the work. "
                    "What has changed is the pressure around that route. "
                    "The model can still guide some cases, but it does not fit every intake path teams now manage."
                ),
            }
        ]
    })
    client = StaticJsonClient(writer_payload)
    result = run_v6_rewrite(source, writer_client=client)
    assert client.calls == 1
    assert result.selected is not None
    assert result.selected.source == "llm"
    assert "That structure also included" not in result.rewritten_text


def test_v6_selection_rejects_polished_lexical_inflation_without_provenance():
    source = (
        "The intake system is changing faster than many teams can manage. "
        "The old process used a form, a queue, and a reviewer."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "The contemporary intake ecosystem is undergoing accelerated transformation beyond organizational adaptation. "
                    "Its traditional procedural framework relied on documentation infrastructure, sequential prioritization, and evaluative personnel."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"
    assert result.rewritten_text == source


def test_v6_selection_rejects_weak_movement_with_source_drift():
    source = (
        "Today the workflow is changing faster than many teams can comfortably manage. "
        "In the past, the process was mostly built around the form, the queue, and the reviewer. "
        "Clients received updates from trusted staff, practiced the steps, and proved completion through checks. "
        "That model still exists, but it no longer fully reflects how clients move through intake today."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "Today the workflow is evolving at a pace that many teams struggle to keep up with. "
                    "In the past, intake centered on the form, the queue, and the reviewer. "
                    "The process was structured around these pillars, with clients receiving updates from trusted staff. "
                    "They practiced the steps and demonstrated completion through checks. "
                    "While this model still exists, it no longer captures how clients engage with intake today. "
                    "The shift reflects deeper changes in how support is accessed and processed."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"
    assert result.rewritten_text == source


def test_v6_selection_allows_reviewable_author_proxy_bridges():
    source = (
        "The intake system is changing faster than many teams can manage. "
        "The old process used a form, a queue, and a reviewer."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "Teams now manage intake under pressure that the older route only partly explains. "
                    "The form and queue still matter. "
                    "The reviewer remains part of the route, but the bridge between those steps needs author confirmation."
                ),
                "author_review_items": [
                    {
                        "item_id": "r001",
                        "provenance": "needs_author_confirmation",
                        "target_text": "bridge between those steps",
                        "generated_text": "needs author confirmation",
                        "user_input_needed": "confirm this bridge",
                        "author_task": "verify or replace",
                    }
                ],
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "llm"
    assert "author confirmation" in result.rewritten_text


def test_v6_writer_prompt_marks_unverified_bridges_for_review_not_blocking():
    scan = scan_text("This result shows a gap because the process should improve.")
    paragraph, plan = build_plan(scan)
    prompt = build_prompt(paragraph, plan)
    assert "author-review provenance" in prompt
    assert "user must review and owns facts, citations, anchors" in prompt
    assert "presented as source-confirmed without author-review provenance" in prompt
    assert "truth" not in prompt.casefold()


def test_v6_document_rewrite_preserves_source_when_writer_has_no_candidates():
    result = run_v6_rewrite_all(sample_text() + "\n\n" + sample_text(), writer_client=FakeClient(), max_passes=2)
    assert result.passes == []
    assert "form" in result.rewritten_text
    json.dumps(result.to_dict())
