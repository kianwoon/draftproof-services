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
    assert "coverage_beats_must_all_appear" in prompt
    assert "form" in prompt
    assert "Start with" not in prompt


def test_v6_packed_list_plan_uses_prose_rebuild_not_atomic_decomposition():
    source = "The process used a form, a queue, a reviewer, and a final check."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    methods = {action.method for action in plan.actions}
    prompt = build_prompt(paragraph, plan)
    payload = json.loads(prompt.split("\n", 1)[1])
    assert "list_rhythm_rebuild" in methods
    assert "atomic_decomposition" not in prompt
    assert "uncapped during golden-route discovery" in payload["required_shape"]["sentence_count"]
    assert payload["coverage_beats_must_all_appear"]
    assert "affected_sentence_routes" not in payload
    assert "generation_brief" not in payload
    assert "source_text" not in payload
    assert "source_beat_map" not in prompt
    assert "coverage_phrase" not in prompt
    assert source not in prompt
    assert "coverage_map" in prompt
    assert "reviewer" in prompt
    assert "Write one ordinary sentence for each coverage beat first" in prompt
    assert "Do not put three or more examples" in prompt


def test_v6_dense_list_coverage_uses_grouped_beats_not_one_item_chain():
    scan = scan_text(
        "The service uses calls, forms, queues, reviewers, messages, dashboards, and follow-up checks."
    )
    paragraph, plan = build_plan(scan)
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    beats = payload["coverage_beats_must_all_appear"]
    assert len(beats) < 7
    assert any("calls" in beat["coverage_capsule"].casefold() and "forms" in beat["coverage_capsule"].casefold() for beat in beats)
    assert any("queues" in beat["coverage_capsule"].casefold() and "reviewers" in beat["coverage_capsule"].casefold() for beat in beats)
    assert any("messages" in beat["coverage_capsule"].casefold() and "dashboards" in beat["coverage_capsule"].casefold() for beat in beats)
    assert "Do not turn grouped coverage terms into repeated one-item sentences" in build_prompt(paragraph, plan)


def test_v6_writer_prompt_blocks_unsubmitted_intensity_expansion():
    scan = scan_text("The service uses calls, forms, queues, reviewers, messages, dashboards, and follow-up checks.")
    paragraph, plan = build_plan(scan)
    prompt = build_prompt(paragraph, plan)
    payload = json.loads(prompt.split("\n", 1)[1])
    assert "Do not add unsubmitted intensity" in prompt
    assert "source relation only" in prompt
    assert "Do not use semicolons" in prompt
    assert "plain_route_contract_for_v1" in payload
    assert "common/essential/dynamic" in prompt
    assert "has become more important/challenging/serious" in prompt
    assert "become/becomes <descriptor A>, <descriptor B>, and <descriptor C>" in prompt
    assert "coverage capsule words" in prompt


def test_v6_coverage_keeps_late_contrast_terms():
    scan = scan_text(
        "A client with strong support may progress quickly, while another client may fall behind through no fault of their own."
    )
    paragraph, plan = build_plan(scan)
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    capsules = " ".join(beat["coverage_capsule"] for beat in payload["coverage_beats_must_all_appear"]).casefold()
    assert "fall" in capsules
    assert "behind" in capsules
    assert "fault" in capsules
    assert "Every source-side contrast must survive" in build_prompt(paragraph, plan)


def test_v6_writer_prompt_filters_selected_source_terms_from_context_terms():
    scan = scan_text(
        "Today, the workflow is changing faster than staff can manage. "
        "The process used a form, a queue, and a reviewer.\n\n"
        "Clients also use chat, email, and online forms before the review."
    )
    paragraph, plan = build_plan(scan, {p.id for p in scan.paragraphs if p.id != "p001"})
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    prompt_context_terms = payload["context_anchors"]["context_terms"]
    assert "Today" not in prompt_context_terms
    assert "form" not in [term.casefold() for term in prompt_context_terms]
    assert any(term.casefold() == "clients" for term in prompt_context_terms)


def test_v6_named_reference_extraction_is_not_domain_keyword_filtered():
    scan = scan_text(
        "Students in Group Alpha used Review Method (2024).\n\n"
        "This result shows an important concern because the process should improve."
    )
    _paragraph, plan = build_plan(scan)
    assert "Group Alpha" in plan.author_proxy_context["named_references"]
    assert "Review Method" in plan.author_proxy_context["named_references"]
    assert "Students" not in plan.author_proxy_context["named_references"]


def test_v6_planner_extracts_author_proxy_context_from_submitted_text():
    text = (
        "Before the review, Alpha Method was introduced by Smith (2020). "
        "The team called this step \"guided intake\".\n\n"
        "This result shows a serious concern because the process should improve across teams and client support.\n\n"
        "According to Rivera et al. (2021), the next review checked client response time."
    )
    scan = scan_text(text)
    paragraph, plan = build_plan(scan)
    context = plan.author_proxy_context
    assert paragraph.id == "p002"
    assert "Alpha Method" in context["named_references"]
    assert "Smith (2020)" in context["citation_spans"]
    assert "Rivera et al. (2021)" in context["citation_spans"]
    assert "2020" in context["years"]
    assert "2021" in context["years"]
    assert "guided intake" in context["quoted_terms"]
    prompt_payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    assert prompt_payload["context_anchors"]["named_references"] == context["named_references"]
    assert prompt_payload["context_anchors"]["named_references"]
    assert "context_anchors" in build_prompt(paragraph, plan)


def test_v6_planner_turns_findings_into_concrete_safe_route_targets():
    scan = scan_text(
        "This result shows an important concern because the process should improve across teams and client support. "
        "The review used a form, a queue, and a final check."
    )
    paragraph, plan = build_plan(scan)
    prompt_payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    route_steps = plan.ai_safe_route["sentence_route_steps"]
    targets = [
        target
        for step in route_steps
        for target in step.get("finding_resolution_targets", [])
    ]
    target_tags = {target["finding_tag"] for target in targets}
    assert {"author_anchor_gap", "unsupported_claim_gap", "packed_list"} <= target_tags
    assert any("inside the same beat" in target["safe_route"] for target in targets)
    assert any("two connected prose beats" in target["safe_route"] for target in targets)
    assert plan.ai_safe_route["golden_route"]["compact_formula"] == "Scope it. Anchor it. Show it. Explain it. Close it."
    assert plan.ai_safe_route["golden_route"]["question_rule"]
    assert plan.ai_safe_route["golden_route_questions"]
    assert plan.ai_safe_route["coverage_beats"]
    assert all(beat.get("coverage_terms") for beat in plan.ai_safe_route["coverage_beats"])
    assert any("merge only" in beat["merge_rule"] for beat in plan.ai_safe_route["coverage_beats"])
    assert not _contains_key(plan.ai_safe_route, "source_text")
    assert not _contains_key(plan.ai_safe_route, "coverage_phrase")
    assert prompt_payload["golden_question"]
    assert prompt_payload["golden_route"]["compact_formula"] == "Scope it. Anchor it. Show it. Explain it. Close it."
    assert prompt_payload["coverage_beats_must_all_appear"]
    assert any(beat.get("finding_tags") for beat in prompt_payload["coverage_beats_must_all_appear"])
    assert any(beat.get("finding_instruction") for beat in prompt_payload["coverage_beats_must_all_appear"])
    assert "generation_brief" not in prompt_payload
    assert "affected_sentence_routes" not in prompt_payload
    assert "what did the writer see, read, compare, struggle with, or decide" in build_prompt(paragraph, plan).casefold()
    assert "do not preserve exact source wording" in build_prompt(paragraph, plan).casefold()


def test_v6_author_anchor_instruction_reaches_writer_beat():
    scan = scan_text("This is an important concern because the process should improve across teams.")
    paragraph, plan = build_plan(scan)
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    instructions = " ".join(beat.get("finding_instruction", "") for beat in payload["coverage_beats_must_all_appear"])
    assert "concrete source relation before any evaluative word" in instructions
    assert "important, challenge, or concern" in instructions
    assert "observable role, pressure, decision, support, or contrast relation" in instructions


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def test_v6_planner_passes_polarity_markers_to_writer_prompt():
    scan = scan_text(
        "The process does not only reward people who remember facts. "
        "It rewards people who can analyse, adapt, communicate, and create."
    )
    paragraph, plan = build_plan(scan)
    markers = [
        marker
        for beat in plan.ai_safe_route["coverage_beats"]
        for marker in beat.get("polarity_markers", [])
    ]
    prompt_payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    prompt_markers = [
        marker
        for beat in prompt_payload["coverage_beats_must_all_appear"]
        for marker in beat.get("polarity_markers", [])
    ]
    assert any(marker.casefold() == "does not only" for marker in markers)
    assert any(marker.casefold() == "does not only" for marker in prompt_markers)
    assert any("not enough" in beat.get("polarity_instruction", "") for beat in prompt_payload["coverage_beats_must_all_appear"])
    assert "Preserve every beat's polarity_markers" in build_prompt(paragraph, plan)


def test_v6_no_longer_without_reflects_keeps_own_fact_capsule():
    scan = scan_text("Knowledge is no longer scarce. Access is no longer the biggest problem.")
    paragraph, plan = build_plan(scan)
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    capsules = [beat["coverage_capsule"] for beat in payload["coverage_beats_must_all_appear"]]
    assert any("Knowledge" in capsule and "scarce" in capsule for capsule in capsules)
    assert any("Access" in capsule and "problem" in capsule for capsule in capsules)
    assert all("young people learn" not in capsule for capsule in capsules)


def test_v6_list_rebuild_does_not_allow_sentence_count_compression():
    scan = scan_text(
        "The process used a form, a queue, and a reviewer. "
        "Clients submitted details, waited for confirmation, and received a response."
    )
    paragraph, plan = build_plan(scan)
    prompt = build_prompt(paragraph, plan)
    payload = json.loads(prompt.split("\n", 1)[1])
    assert "uncapped during golden-route discovery" in payload["required_shape"]["sentence_count"]
    assert "uncapped during golden-route discovery" in payload["required_shape"]["word_count"]
    assert "Write one ordinary sentence for each coverage beat first" in prompt


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


def test_v6_parse_variants_preserves_string_author_review_rows():
    from poc.rewrite_v6.write import parse_variants

    variants = parse_variants({
        "variants": [
            {
                "id": "v1",
                "text": "The route needs a bridge.",
                "author_proxy_provenance": ["bridge inferred from nearby context"],
                "author_review_items": ["confirm this bridge"],
                "coverage_map": [{"coverage_beat_id": "b1", "sentence": "The route needs a bridge."}],
            }
        ]
    })
    assert variants[0].author_proxy_provenance
    assert variants[0].author_proxy_provenance[0]["generated_text"] == "bridge inferred from nearby context"
    assert variants[0].author_review_items
    assert variants[0].author_review_items[0]["provenance"] == "needs_author_confirmation"
    assert variants[0].coverage_map
    assert variants[0].coverage_map[0]["coverage_beat_id"] == "b1"


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
    assert plan.ai_safe_route["sentence_route_steps"]
    assert "resolution_targets" not in prompt
    assert "author_proxy_policy" in prompt
    assert "author_proxy_provenance" in prompt
    assert "author_review_items" in prompt


def test_v6_planner_scopes_evaluative_unsupported_author_gap_instead_of_adding_conclusion():
    scan = scan_text("This is a serious concern because the modern process should improve across teams and client support.")
    _paragraph, plan = build_plan(scan)
    assert plan.actions[0].method == "claim_scope_repair"
    assert "not as an added conclusion" in plan.actions[0].operation
    assert "failed_route" in plan.paragraph_strategy
    assert "source-beat plan" in plan.paragraph_strategy["writer_instruction"]


def test_v6_scanner_does_not_mistake_this_is_for_first_person_i():
    scan = scan_text("This is a serious concern because the modern process should improve across teams and client support.")
    assert any("author_anchor_gap" in finding.tags for finding in scan.findings)


def test_v6_author_proxy_prompt_does_not_block_reviewable_bridges():
    scan = scan_text("This result shows an important concern because the process should improve.")
    paragraph, plan = build_plan(scan)
    prompt = build_prompt(paragraph, plan)
    assert "source-derived terms only" not in prompt
    assert "Use author_proxy_provenance or author_review_items" in prompt
    assert "author_proxy_provenance" in prompt


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
                    "Teams now handle intake under pressure the older route partly explains. "
                    "Forms and queues still matter. "
                    "Reviewers give clients a route through the work. "
                    "Confirmation and responses stay inside that route. "
                    "Some cases still fit the model. "
                    "Every intake path does not fit it now."
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


def test_v6_selection_rejects_mechanical_itemized_decomposition():
    source = (
        "The intake system is changing faster than many teams can manage. "
        "The old process used a form, a queue, and a reviewer. "
        "Clients submitted details, waited for confirmation, and received a response."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "The intake system carries this pressure. "
                    "The old process used a form. "
                    "The old process used a queue. "
                    "The old process used a reviewer. "
                    "Clients submitted details. "
                    "Clients waited for confirmation. "
                    "Clients received a response."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"
    assert result.rewritten_text == source


def test_v6_selection_rejects_finding_drop_when_risk_and_drift_get_worse():
    source = (
        "However, the current education system still carries many old habits. "
        "Many schools continue to place heavy pressure on grades, exams, and standard answers. "
        "This can encourage memorisation rather than understanding. "
        "Students may learn how to pass, but not always how to think deeply, solve problems, or connect ideas across subjects. "
        "This is a serious concern because the modern world does not only reward people who can remember facts. "
        "It rewards people who can analyse, adapt, communicate, and create."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "However, the current education system still carries many old habits. "
                    "Many schools continue to place heavy pressure on grades, exams, and standard answers, which often leads to memorisation over understanding. "
                    "This focus can result in students learning how to pass tests without always developing the ability to think deeply, solve problems, or connect ideas across subjects. "
                    "This is a serious concern because the modern world does not only reward people who can remember facts—it rewards those who can analyse, adapt, communicate, and create. "
                    "The gap between what schools emphasize and what the world demands highlights a growing mismatch that needs attention."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"
    assert result.rewritten_text == source


def test_v6_selection_rejects_extra_unmapped_conclusion_beat():
    source = (
        "The current process still carries old habits. "
        "Teams place heavy pressure on forms, queues, and standard answers. "
        "Clients may learn how to pass checks, but not how to explain problems."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "The current process still carries old habits. "
                    "Teams place heavy pressure on forms and queues, with standard answers treated as the goal. "
                    "Clients may learn how to pass checks, but not how to explain problems. "
                    "The shift shows why the whole system needs deeper transformation."
                ),
                "author_review_items": ["The final sentence is inferred from the draft."],
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"
    assert result.rewritten_text == source


def test_v6_selection_rejects_final_source_beat_replaced_by_conclusion():
    source = (
        "The process still carries old habits. "
        "Teams place pressure on forms, queues, and standard answers. "
        "It rewards people who can analyse, adapt, communicate, and create."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "The process still carries old habits. "
                    "Teams place pressure on forms and queues, with standard answers treated as the goal. "
                    "The shift in expectations demands a rethinking of how the process is structured."
                ),
                "author_review_items": ["The final sentence is inferred from the draft."],
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
    assert "author_proxy_provenance" in prompt
    assert "user must review and owns facts, citations, anchors" in prompt
    assert "truth" not in prompt.casefold()


def test_v6_document_rewrite_preserves_source_when_writer_has_no_candidates():
    result = run_v6_rewrite_all(sample_text() + "\n\n" + sample_text(), writer_client=FakeClient(), max_passes=2)
    assert result.passes == []
    assert "form" in result.rewritten_text
    json.dumps(result.to_dict())
