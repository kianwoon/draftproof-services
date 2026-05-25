from __future__ import annotations

import json
from pathlib import Path

from poc.rewrite_v6.pipeline import run_v6_rewrite, run_v6_rewrite_all
from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6 import production as v6_production
from poc.rewrite_v6.scan import scan_text
from poc.rewrite_v6.write import build_prompt


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


def test_v6_rewrite_runs_with_compiled_variant_without_llm_dependency():
    client = FakeClient()
    result = run_v6_rewrite(sample_text(), writer_client=client)
    assert result.selected is not None
    assert result.selected.source == "compiler"
    assert "form" in result.selected.text
    assert "form" in result.rewritten_text
    assert client.calls == 0


def test_v6_bad_writer_json_falls_back_to_compiler_candidate():
    client = BadJsonClient()
    result = run_v6_rewrite("This result shows an important concern because the process should improve.", writer_client=client)
    assert result.selected is not None
    assert result.selected.source == "compiler"
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


def test_v6_compiler_rebuilds_generic_context_and_goal_routes():
    text = (
        "A process creates pressure. "
        "This is a serious concern because the review depends on one narrow check. "
        "It can redesign the workflow for the reality the team now works inside. "
        "The goal should be to keep the result useful for the people using it."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "This is a serious concern because" not in result.selected.text
    if "It can redesign" in result.scan.findings[0].evidence["text"]:
        assert "It can redesign" not in result.selected.text
    if "goal should be" in result.scan.findings[0].evidence["text"]:
        assert "goal should be" not in result.selected.text


def test_v6_compiler_preserves_coauthor_citations_and_because_clauses():
    text = (
        "Lee and Morgan (2021) describe a process with planning, checking, and review. "
        "My analysis centered on that pattern, because the outcome changed after the review."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "Lee and Morgan (2021)" in result.selected.text
    assert "centered on because" not in result.selected.text


def test_v6_compiler_does_not_split_commas_inside_citations():
    text = "During the first stage, participants used reviews before selecting a provider (Riley et al., 2024)."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "et al. During" not in result.selected.text
    assert "(Riley et al., 2024)" in result.selected.text


def test_v6_compiler_reframes_author_year_claim_wrappers():
    text = "Morgan et al. (2024) states that service failure creates customer frustration and negative reviews."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "states that" not in result.selected.text
    assert "(Morgan et al., 2024)" in result.selected.text


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


def test_v6_compiler_removes_unsupported_important_and_rebuilds_it_has_route():
    text = (
        "A program affects the wider team. "
        "It has one of the strongest support patterns in the group. "
        "The review was an important period that aimed to improve the workflow and support the wider group during the change."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "It has one of" not in result.selected.text
    assert "important period" not in result.selected.text


def test_v6_compiler_rebuilds_it_also_faces_context_route():
    text = "The project has several limits. It also faces challenges related to access, inequality, and participation."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "It also faces" not in result.selected.text


def test_v6_compiler_revoices_example_carries_pressure_route():
    text = "The example demonstrates that the method carries this pressure: not only about completing the task."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "carries this pressure" not in result.selected.text


def test_v6_compiler_rebuilds_it_comes_from_context_route():
    text = "The method is easier to follow now. It comes from a process learners can follow, repeat, and check."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "It comes from" not in result.selected.text


def test_v6_compiler_repairs_pronoun_result_verb_route():
    text = "It directly affects the perceptions of future participants, creating a negative expectation."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "The directly affects" not in result.selected.text
    assert "It directly affects" not in result.selected.text
    assert "creating a negative" not in result.selected.text


def test_v6_compiler_moves_preposed_view_context_off_this_route():
    text = "In my view, this example demonstrates that the process is not only about access but also about confidence."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert not result.selected.text.startswith("In my view")
    assert not result.selected.text.startswith("This example")
    assert "not only about but also" not in result.selected.text


def test_v6_compiler_splits_not_only_subject_routes():
    text = (
        "Not only one learner but the whole group found common ground and learned together, "
        "thereby creating a shared result that changed the whole group discussion over time."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "Not only" not in result.selected.text
    assert "thereby creating" not in result.selected.text


def test_v6_compiler_splits_thereby_gerund_result_tail():
    text = "The teacher can determine whether students can connect the steps together, thereby understanding and creating connections."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "thereby understanding" not in result.selected.text


def test_v6_compiler_splits_then_action_clause():
    text = "The chart outlines the triangular form, then describe how projection influences the line."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert ", then describe" not in result.selected.text
    assert "then describes" in result.selected.text


def test_v6_compiler_splits_and_then_followup_action():
    text = "I encourage my students to find an approach that works best for them and then compare it with traditional techniques."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "and then compare" not in result.selected.text
    assert "My students then compare" in result.selected.text


def test_v6_compiler_repairs_transition_pronoun_and_when_routes():
    text = (
        "Hence, this falls under the process where the service must be provided through customer presence. "
        "The experience is affected by staff attention during the service process and the support routine when customers are actively assessing the quality of service."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "Hence" not in result.selected.text
    assert "This falls" not in result.selected.text
    assert " When " in result.selected.text


def test_v6_compiler_splits_including_whether_clauses():
    text = (
        "These reviewers will help rate behaviors identified from notes, "
        "including how quickly teams respond and whether they return to check the process."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "including how quickly" not in result.selected.text
    assert "The first check is" in result.selected.text
    assert "The second check is" in result.selected.text


def test_v6_compiler_revoices_passive_and_result_wrappers():
    text = (
        "The response is typically a result of unclear service. "
        "The attention given to participants is an aspect of determining how they feel. "
        "The results will be used by the team to identify areas where they need to improve."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "is typically a result of" not in result.selected.text
    assert "is an aspect of determining" not in result.selected.text
    assert "will be used by" not in result.selected.text


def test_v6_compiler_revoices_report_and_instruction_wrappers():
    text = (
        "The main reason is attributed to changing user habits. "
        "The review will proceed from field notes to interpret participant behavior. "
        "The mentor must ensure that participants understand the purpose of the first step. "
        "The move is best accompanied by practice."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "main reason is attributed to" not in result.selected.text
    assert "will proceed from" not in result.selected.text
    assert "must ensure" not in result.selected.text
    assert "is best accompanied by" not in result.selected.text


def test_v6_compiler_splits_report_claims_and_relative_used_to():
    text = (
        "With the low cost reducing friction, the annual report (2024) points out a shift where low cost becomes the main reason for participation. "
        "The model stands for a framework that can be used to measure understanding."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "main reason" not in result.selected.text
    assert "that can be used to" not in result.selected.text


def test_v6_compiler_splits_result_and_condition_routes():
    text = (
        "The videos omit the process, leading viewers to feel confident and underestimating the work. "
        "If the team keeps the routine then quality would not be accidental."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "leading viewers" not in result.selected.text
    assert " then " not in result.selected.text


def test_v6_compiler_removes_trailing_structural_labels():
    text = "This point demonstrates how service can affect customers in summary."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "in summary" not in result.selected.text.casefold()


def test_v6_compiler_splits_if_comma_result_routes():
    text = "If unclear service causes customers to be dissatisfied, they may experience frustration that encourages them to leave."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert not result.selected.text.startswith("If unclear")
    assert "that encourages" not in result.selected.text
    assert "They encourages" not in result.selected.text


def test_v6_compiler_splits_semicolon_and_while_routes():
    text = (
        "The content interferes with attention; true learning should involve transforming what people see. "
        "The mentor checks that participants understand the purpose while ensuring quality, consistency, and completion."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert ";" not in result.selected.text
    assert "should involve" not in result.selected.text
    assert " while ensuring" not in result.selected.text


def test_v6_compiler_splits_non_list_while_gerund_tail():
    text = "The mentor checks that students understand the purpose while ensuring the quality of completion."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert " while ensuring" not in result.selected.text
    assert "ensures the quality" in result.selected.text


def test_v6_compiler_repairs_report_style_residual_routes():
    text = (
        "Facilitating participant progression through targeted intervention is the primary responsibility of a mentor. "
        "Compared to older training, the greater challenge today is how to integrate new resources with practical work. "
        "The transition starts from early uncertainty to the realization that the task has technical boundaries. "
        "It made clear the activity could help someone."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "primary responsibility of" not in result.selected.text
    assert " to the realization" not in result.selected.text
    assert "It made clear" not in result.selected.text


def test_v6_compiler_revoices_heading_task_fragment():
    text = "Chart to outline the triangular form, then describe the result."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert not result.selected.text.startswith("Chart to")


def test_v6_compiler_repairs_for_me_priority_route():
    text = "For me is more important to teach students how to think, learn, and grow independently than simply passing an assessment."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "For me is more important" not in result.selected.text


def test_v6_compiler_splits_involves_represents_and_provides_through():
    text = (
        "The stage involves direct interaction and represents a key decision point. "
        "The added route provides a way of accessing support through the service experience."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "involves direct interaction and represents" not in result.selected.text
    assert "The route works through" in result.selected.text


def test_v6_compiler_splits_malformed_is_modal_predicate():
    text = "The framework is a model can measure participant understanding across several tasks."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "model can measure" not in result.selected.text
    assert "framework can measure" in result.selected.text.casefold()


def test_v6_compiler_does_not_break_valid_that_can_clause():
    text = "A repeated online pattern makes participants believe that professional skills can be mastered instantly."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "believe that professional skills" in result.selected.text
    assert "A can be mastered" not in result.selected.text


def test_v6_writer_prompt_marks_unverified_bridges_for_review_not_blocking():
    scan = scan_text("This result shows a gap because the process should improve.")
    paragraph, plan = build_plan(scan)
    prompt = build_prompt(paragraph, plan)
    assert "author-review provenance" in prompt
    assert "user must review and owns facts, citations, anchors" in prompt
    assert "presented as source-confirmed without author-review provenance" in prompt
    assert "truth" not in prompt.casefold()


def test_v6_compiler_splits_citation_led_requirement_lists():
    text = "Miller (2024) points out that practice-based knowledge often requires demonstration, instruction, and scaffolding."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert scan_text(result.selected.text).scores["finding_count"] == 0.0


def test_v6_compiler_preserves_semicolon_citation_groups_inside_parentheses():
    text = "The method supports classroom participation, communication practice, and peer review (Able, 2024; Baker et al., 2022)."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "(Able, 2024; Baker et al., 2022)" in result.selected.text


def test_v6_compiler_does_not_create_context_or_verb_phrase_fragments():
    text = (
        "Today’s workflow is changing faster than some teams can manage. "
        "In the first stage, the process was built around intake, review, and approval."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "Changing faster than" not in result.selected.text
    assert "In the first stage." not in result.selected.text


def test_v6_compiler_varies_repeated_context_list_frames():
    text = "In the first stage, the process was built around intake, review, and approval."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert result.selected.text.count("In the first stage") <= 1
    assert "That structure also included review" in result.selected.text


def test_v6_compiler_preserves_non_target_context_sentences():
    text = (
        "This result shows an important concern because the process should improve. "
        "The support step is clear enough for the current workflow."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "The support step is clear enough for the current workflow." in result.selected.text


def test_v6_compiler_does_not_rewrite_single_comma_result_as_list():
    text = "The hands had not settled on the tool, so the whole sequence stalled."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "on so the whole" not in result.selected.text


def test_v6_compiler_reuses_active_verb_for_list_items():
    text = "The process bundles planning, checking, review, and documentation into one workflow."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "The checking." not in result.selected.text
    assert "The review." not in result.selected.text


def test_v6_compiler_splits_long_from_who_and_showing_clauses():
    text = (
        "In my report, I tracked what the process actually demands from participants who arrive with different starting points. "
        "The card became evidence I could reflect on, showing a tangible impact beyond the room."
    )
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "In my report, I tracked" not in result.selected.text
    assert "participants arrive with different starting points" in result.selected.text.casefold()
    assert "showing a tangible" not in result.selected.text


def test_v6_compiler_splits_during_gerund_context():
    text = "The learner kept focus on each individual during a community event cutting hair for students with complex support needs and expectations."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert "event cutting" not in result.selected.text


def test_v6_compiler_moves_following_completion_context():
    text = "Following completion of the first check, customers can request additional support through the system mentioned previously."
    result = run_v6_rewrite(text, writer_client=FakeClient())
    assert result.selected is not None
    assert not result.selected.text.startswith("Following completion")


def test_v6_document_rewrite_runs_multiple_passes_without_v5():
    result = run_v6_rewrite_all(sample_text() + "\n\n" + sample_text(), writer_client=FakeClient(), max_passes=2)
    assert result.passes
    assert "form" in result.rewritten_text
    json.dumps(result.to_dict())
