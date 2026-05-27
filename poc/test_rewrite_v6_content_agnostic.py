from __future__ import annotations

import json
from pathlib import Path

from poc.rewrite_v6 import pipeline as v6_pipeline
from poc.rewrite_v6.pipeline import _planner_provider, _writer_model, _writer_provider, run_v6_rewrite, run_v6_rewrite_all
from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6.planner_llm import run_planner_llm
from poc.rewrite_v6.quality_repair import QualityRepairOperation, apply_quality_repair_operations
from poc.rewrite_v6 import production as v6_production
from poc.rewrite_v6.scan import findings_for_paragraph, scan_text
from poc.rewrite_v6.write import Variant, build_prompt, choose_variant
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
    assert "Do not write a repair trace" in prompt
    assert "Group related coverage beats into ordinary paragraph sentences" in prompt
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
    assert "Do not write a repair trace" in prompt
    assert "preferred_sentence_count" in prompt
    assert "Not always" in prompt
    assert "pair the first two items and the last two items" in prompt
    assert "not only reward beat" in prompt
    assert "success, readiness, or essential-skills ending" in prompt
    assert "Do not use semicolons" in prompt
    assert "plain_route_contract_for_v1" in payload
    assert "unsubmitted intensity" in payload["plain_route_contract_for_v1"]
    assert "descriptor lists" in payload["plain_route_contract_for_v1"]


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
    assert "chat" not in [term.casefold() for term in prompt_context_terms]
    assert "email" not in [term.casefold() for term in prompt_context_terms]


def test_v6_writer_prompt_does_not_import_neighbor_named_examples():
    scan = scan_text(
        "Clients use PortalX, AlphaTool, online courses, and remote forms before the review.\n\n"
        "This shift has made the reviewer role more important, not less important."
    )
    paragraph, plan = build_plan(scan, {"p001"})
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    assert paragraph.id == "p002"
    assert "PortalX" not in payload["context_anchors"]["named_references"]
    assert "AlphaTool" not in payload["content_word_boundary"]["allowed_content_terms"]
    assert "online" not in payload["content_word_boundary"]["allowed_content_terms"]
    assert "courses" not in payload["context_anchors"]["context_terms"]
    question_terms = [
        term
        for row in payload["author_route_questions"]
        for term in row.get("answer_basis_terms", [])
    ]
    assert "PortalX" not in question_terms


def test_v6_writer_context_terms_do_not_import_generic_neighbor_vocabulary():
    scan = scan_text(
        "Alpha Method was introduced by Smith (2020). Constant information filled the review space.\n\n"
        "This result shows a serious concern because the process should improve across teams.\n\n"
        "The next review mentioned a noisy environment and peer channels."
    )
    paragraph, plan = build_plan(scan)
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    context_terms = {term.casefold() for term in payload["context_anchors"]["context_terms"]}
    assert paragraph.id == "p002"
    assert "constant" not in context_terms
    assert "filled" not in context_terms
    assert "environment" not in context_terms
    assert "Alpha Method" in payload["context_anchors"]["named_references"]
    assert "Smith (2020)" in payload["context_anchors"]["citation_spans"]


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
    assert "Alpha Method" in prompt_payload["context_anchors"]["named_references"]
    assert "Smith" in prompt_payload["context_anchors"]["named_references"]
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
    assert plan.ai_safe_route["author_route_questions"]
    assert any("reviewable bridge narrows the claim" in row["question"] for row in plan.ai_safe_route["author_route_questions"])
    assert any("Which source items belong together" in row["question"] for row in plan.ai_safe_route["author_route_questions"])
    assert plan.ai_safe_route["coverage_beats"]
    assert all(beat.get("coverage_terms") for beat in plan.ai_safe_route["coverage_beats"])
    assert any("merge only" in beat["merge_rule"] for beat in plan.ai_safe_route["coverage_beats"])
    assert not _contains_key(plan.ai_safe_route, "source_text")
    assert not _contains_key(plan.ai_safe_route, "coverage_phrase")
    assert prompt_payload["golden_question"]
    assert prompt_payload["author_route_questions"]
    assert any(row["writer_duty"] for row in prompt_payload["author_route_questions"])
    assert "route_answer_cards" in prompt_payload["output_schema"]["variants"][0]
    assert prompt_payload["content_word_boundary"]["allowed_content_terms"]
    assert prompt_payload["golden_route"]["compact_formula"] == "Scope it. Anchor it. Show it. Explain it. Close it."
    assert prompt_payload["coverage_beats_must_all_appear"]
    assert any(beat.get("finding_tags") for beat in prompt_payload["coverage_beats_must_all_appear"])
    assert any(beat.get("finding_instruction") for beat in prompt_payload["coverage_beats_must_all_appear"])
    assert "generation_brief" not in prompt_payload
    assert "affected_sentence_routes" not in prompt_payload
    assert "what did the writer see, read, compare, struggle with, or decide" in build_prompt(paragraph, plan).casefold()
    assert "do not preserve exact source wording" in build_prompt(paragraph, plan).casefold()
    assert "Build the paragraph by answering author_route_questions" in build_prompt(paragraph, plan)
    assert "fill route_answer_cards for every author_route_question" in build_prompt(paragraph, plan)
    assert "route_answer_cards.draft_sentence is a route sketch, not a coverage limit" in build_prompt(paragraph, plan)
    assert "Do not write meta-prose such as the writer sees" in build_prompt(paragraph, plan)
    assert "do not preserve submitted facts as written" in build_prompt(paragraph, plan)
    assert "Stay inside content_word_boundary.allowed_content_terms" in build_prompt(paragraph, plan)


def test_v6_downgrades_unsafe_llm_planner_contracts_to_route_questions():
    scan = scan_text(
        "This result shows an important concern because the process should improve across teams. "
        "The review used a form, a queue, and a final check."
    )
    paragraph, plan = build_plan(scan)
    planner_payload = json.dumps({
        "planner_decision": {
            "paragraph_route": "Use the rewritten planner route.",
            "finding_contracts": [
                {
                    "source_sentence_id": "p001_s001",
                    "safe_rebuild_shape": "This result shows an important concern because the process should improve across teams.",
                }
            ],
            "paragraph_blueprint": [
                {
                    "step_id": "b001",
                    "safe_sentence_shape": "This result shows an important concern because the process should improve.",
                }
            ],
        }
    })
    updated = run_planner_llm(
        paragraph,
        plan,
        findings_for_paragraph(scan, paragraph.id),
        client=StaticJsonClient(planner_payload),
    )
    decision = updated.ai_safe_route["llm_planner_decision"]
    assert decision["status"] == "degraded_contract_gaps"
    assert decision["finding_contracts"]
    assert decision["paragraph_blueprint"]
    assert decision["do_not_copy_phrases"]
    assert "exact source anchor first" not in json.dumps(decision)
    serialized_decision = json.dumps(decision)
    assert "concrete source anchor" in serialized_decision or "Group related source terms" in serialized_decision
    assert "author_route_questions" in build_prompt(paragraph, updated)
    assert "deterministic fallback finding_contracts" in decision["fallback_instruction"]


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
    assert any("does not only reward" in beat.get("polarity_instruction", "") for beat in prompt_payload["coverage_beats_must_all_appear"])
    assert "Preserve every beat's polarity_markers" in build_prompt(paragraph, plan)
    assert "do not repeat not always across a sentence chain" in build_prompt(paragraph, plan)


def test_v6_no_longer_without_reflects_keeps_own_fact_capsule():
    scan = scan_text("Knowledge is no longer scarce. Access is no longer the biggest problem.")
    paragraph, plan = build_plan(scan)
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    capsules = [beat["coverage_capsule"] for beat in payload["coverage_beats_must_all_appear"]]
    assert any("Knowledge" in capsule and "scarce" in capsule for capsule in capsules)
    assert any("Access" in capsule and "problem" in capsule for capsule in capsules)
    assert all("young people learn" not in capsule for capsule in capsules)


def test_v6_negative_contrast_comma_stays_one_coverage_beat():
    scan = scan_text("This shift has made the role of teachers even more important, not less important.")
    paragraph, plan = build_plan(scan)
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    beats = [
        beat for beat in plan.ai_safe_route["coverage_beats"]
        if beat["source_sentence_id"] == "p001_s001"
    ]
    assert len(beats) == 1
    assert "less" in {term.casefold() for term in beats[0]["coverage_terms"]}
    assert payload["polarity_constraints"]
    assert "limits" in payload["polarity_constraints"][0]["forbidden_shapes"]
    assert "Use neighboring context only to name the bridge" in build_prompt(paragraph, plan)


def test_v6_not_only_polarity_requires_two_sided_contrast():
    source = (
        "The process does not only reward people who remember facts. "
        "It rewards people who can analyse, adapt, communicate, and create."
    )
    bad_payload = json.dumps({
        "variants": [
            {
                "id": "bad",
                "text": (
                    "The process rewards people who remember facts. "
                    "Analysis and adaptation matter in the next step."
                ),
            }
        ]
    })
    bad = run_v6_rewrite(source, writer_client=StaticJsonClient(bad_payload))
    assert bad.selected is not None
    assert bad.selected.source == "source_preserved"

    good_payload = json.dumps({
        "variants": [
            {
                "id": "good",
                "text": (
                    "Remember facts is not enough by itself. "
                    "People who analyse and adapt also matter. "
                    "People who communicate and create carry the next part."
                ),
            }
        ]
    })
    good = run_v6_rewrite(source, writer_client=StaticJsonClient(good_payload))
    assert good.selected is not None
    assert good.selected.source == "llm"
    prompt = build_prompt(*build_plan(scan_text(source)))
    assert "not enough by itself" in prompt
    assert "also matters" in prompt
    assert "not only a concern" in prompt


def test_v6_four_item_list_gets_grouped_sentence_plan_without_term_hardcode():
    scan = scan_text("The process asks people to compare, adjust, explain, and build.")
    paragraph, plan = build_plan(scan)
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    assert "paragraph_sentence_plan" in payload
    slot = payload["paragraph_sentence_plan"][0]
    assert len(slot["coverage_beat_ids"]) == 2
    assert "never rejoin the groups into a comma list" in slot["slot_goal"]
    assert any(row["grouped_relation"] for row in payload["coverage_beats_must_all_appear"])
    assert not any(row.get("pairing_hint") for row in payload["coverage_beats_must_all_appear"])


def test_v6_because_not_only_claim_splits_evaluation_and_cause_beats():
    scan = scan_text("This is a serious concern because the modern world does not only reward people who remember facts.")
    paragraph, plan = build_plan(scan)
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    beats = payload["coverage_beats_must_all_appear"]
    concern = next(row for row in beats if "concern" in {term.casefold() for term in row["coverage_terms"]})
    not_only = next(row for row in beats if "does not only" in [marker.casefold() for marker in row["polarity_markers"]])
    assert concern["beat_id"] != not_only["beat_id"]
    assert "concern" not in {term.casefold() for term in not_only["coverage_terms"]}


def test_v6_selection_rejects_malformed_not_only_concern():
    source = (
        "This is a serious concern because the modern world does not only reward people who remember facts. "
        "It rewards people who can analyse, adapt, communicate, and create."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "bad",
                "text": (
                    "The modern world rewards people who remember facts, not only a serious concern. "
                    "People who analyse and adapt also matter."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"


def test_v6_selection_rejects_not_only_without_also_side():
    source = (
        "The process does not only focus on what clients know, but also on how clients decide. "
        "Reviewers guide clients through the check."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "bad",
                "text": (
                    "The process does not only focus on what clients know. "
                    "Reviewers guide clients through the check."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"


def test_v6_selection_rejects_reversed_rather_than_contrast():
    source = (
        "The process can encourage memorisation rather than understanding. "
        "Clients may learn how to pass, but not always how to explain the result."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "bad",
                "text": (
                    "The process carries habits from old checks. "
                    "Understanding is encouraged rather than memorisation. "
                    "Clients learn how to pass but do not always explain the result."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"


def test_v6_rather_than_beat_starts_from_left_side_pressure():
    scan = scan_text("This can encourage memorisation rather than understanding.")
    paragraph, plan = build_plan(scan)
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    beat = next(
        row for row in payload["coverage_beats_must_all_appear"]
        if "rather than" in row.get("polarity_markers", [])
    )
    starters = {term.casefold() for term in beat["starter_terms"]}
    assert "memorisation" in starters
    assert "understanding" not in starters


def test_v6_not_always_contrast_is_not_grouped_with_positive_side():
    scan = scan_text("Students may learn how to pass, but not always how to think deeply, solve problems, or connect ideas.")
    paragraph, plan = build_plan(scan)
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    beats = payload["coverage_beats_must_all_appear"]
    positive = next(row for row in beats if "pass" in {term.casefold() for term in row["coverage_terms"]})
    limited = [row for row in beats if "not always" in [marker.casefold() for marker in row["polarity_markers"]]]
    assert all(positive["beat_id"] != row["beat_id"] for row in limited)
    assert "not always" not in [marker.casefold() for marker in positive["polarity_markers"]]
    assert len(limited) == 1
    limited_terms = {term.casefold() for term in limited[0]["coverage_terms"]}
    assert {"think", "solve", "connect"} <= limited_terms
    assert not limited[0]["coverage_capsule"].casefold().startswith("not always")
    assert "Use one contrast sentence" in build_prompt(paragraph, plan)


def test_v6_selection_rejects_not_always_scope_shift():
    source = (
        "Clients may learn how to pass, but not always how to explain the result. "
        "The review also asks clients to connect ideas."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "bad",
                "text": (
                    "Clients do not always learn to pass or explain the result. "
                    "The review asks clients to connect ideas."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"


def test_v6_selection_rejects_repeated_sentence_intent():
    source = (
        "Students may learn how to pass, but not always how to think deeply. "
        "They also need to solve problems across subjects."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "bad",
                "text": (
                    "Students learn to pass but do not always think deeply. "
                    "Students do not always think deeply about problems. "
                    "Students solve problems across subjects."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"


def test_v6_selection_rejects_unresolved_comma_list_contract():
    source = "The process rewards people who analyse, adapt, communicate, and create."
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "bad",
                "text": "The process rewards people who analyse, adapt, communicate, and create.",
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"


def test_v6_selection_rejects_unsubmitted_success_close():
    source = "The process rewards people who can analyse, adapt, communicate, and create."
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "bad",
                "text": (
                    "The process rewards people who can analyse and adapt. "
                    "Communicate and create carry the next part. "
                    "These abilities are essential for success."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"


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
    assert "Group related coverage beats into ordinary paragraph sentences" in prompt


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


def test_v6_writer_prompt_defaults_to_one_variant_to_avoid_duplicate_token_waste(monkeypatch):
    scan = scan_text("The process used a form, a queue, and a review.")
    paragraph, plan = build_plan(scan)
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    requirements = payload["variant_requirements"]
    assert [row["id"] for row in requirements] == ["v1"]
    assert payload["hard_generation_requirements"]["required_variant_ids"] == ["v1"]
    assert payload["hard_generation_requirements"]["exact_variant_count"] == 1
    monkeypatch.setenv("DRAFTPROOF_V6_WRITER_VARIANTS", "3")
    multi_payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    assert [row["id"] for row in multi_payload["variant_requirements"]] == ["v1", "v2", "v3"]
    assert "return exactly the ids" in build_prompt(paragraph, plan)
    focused_payload = json.loads(build_prompt(paragraph, plan, variant_focus=multi_payload["variant_requirements"][1]).split("\n", 1)[1])
    assert focused_payload["active_variant"]["id"] == "v2"
    assert len(focused_payload["output_schema"]["variants"]) == 1


def test_v6_parse_variants_drops_duplicate_texts():
    from poc.rewrite_v6.write import parse_variants

    variants = parse_variants({
        "variants": [
            {"id": "v1", "mode": "coverage_beat_generation", "text": "The route changes."},
            {"id": "v2", "mode": "golden_question_generation", "text": "The route changes."},
            {"id": "v3", "mode": "context_anchor_generation", "text": "The route changes differently."},
        ]
    })
    assert [variant.id for variant in variants] == ["v1", "v3"]
    assert variants[0].mode == "coverage_beat_generation"


def test_v6_files_stay_below_1000_lines_and_do_not_import_v5():
    root = Path(__file__).resolve().parent / "rewrite_v6"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert len(text.splitlines()) < 1000, path
        assert "rewrite_v5" not in text, path


def test_v6_json_result_is_serializable():
    result = run_v6_rewrite("A process uses a step, a check, and a result.", writer_client=FakeClient())
    payload = result.to_dict()
    json.dumps(payload)
    assert payload["variants"] == []
    assert payload["source_preserved"]["source"] == "source_preserved"


def test_v6_document_rewrite_reports_paragraph_progress():
    events = []

    run_v6_rewrite_all(
        "A process uses a form, a queue, and a review.",
        writer_client=FakeClient(),
        progress_callback=lambda percent, message: events.append((percent, message)),
    )

    assert any(percent > 62 for percent, _message in events)
    assert any("Planning V6 paragraph" in message for _percent, message in events)
    assert any("Writing V6 paragraph" in message for _percent, message in events)


def test_v6_document_rewrite_stops_before_llm_when_runtime_budget_is_tight():
    writer = FakeClient()
    events = []

    result = run_v6_rewrite_all(
        "A process uses a form, a queue, and a review.",
        writer_client=writer,
        progress_callback=lambda percent, message: events.append((percent, message)),
        runtime_budget_seconds=1,
        min_llm_request_seconds=30,
    )

    assert writer.calls == 0
    assert result.passes == []
    assert result.rewritten_text == "A process uses a form, a queue, and a review."
    assert any("runtime budget reached" in message for _percent, message in events)


def test_v6_document_rewrite_covers_unseen_finding_paragraphs_before_revisiting(monkeypatch):
    source = (
        "This process uses a form, a queue, and a review. "
        "This result shows a problem because the system should improve.\n\n"
        "This framework uses a policy, a standard, and a benchmark."
    )
    seen = []

    def fake_run(current, **kwargs):
        scan = scan_text(current)
        paragraph, plan = build_plan(scan, kwargs.get("excluded_paragraph_ids"))
        seen.append(paragraph.id)
        replacement = "This result shows a problem because the system should improve." if paragraph.id == "p001" else "The framework uses a policy standard."
        blocks = [replacement if block.id == paragraph.id else block.text for block in scan.paragraphs]
        return v6_pipeline.Result(scan=scan, plan=plan, variants=[], selected=None, rewritten_text="\n\n".join(blocks))

    monkeypatch.setattr(v6_pipeline, "run_v6_rewrite", fake_run)
    result = v6_pipeline.run_v6_rewrite_all(source, max_passes=2, residual_followup_passes=0)

    assert seen == ["p001", "p002"]
    assert [row["target_paragraph_id"] for row in result.pass_trace] == ["p001", "p002"]


def test_v6_quality_repair_applies_minimal_exact_operations():
    text = "Approach increases student motivation. I am an educator who need to listen."
    repaired, applied, skipped = apply_quality_repair_operations(text, [
        QualityRepairOperation("Approach increases", "This approach increases", "fragment"),
        QualityRepairOperation("who need", "who needs", "agreement"),
    ])

    assert repaired == "This approach increases student motivation. I am an educator who needs to listen."
    assert [operation.reason for operation in applied] == ["fragment", "agreement"]
    assert skipped == []


def test_v6_quality_repair_skips_expansive_or_protected_changes():
    text = "The unit SHBHCUT006 requires six assessments."
    repaired, applied, skipped = apply_quality_repair_operations(text, [
        QualityRepairOperation("SHBHCUT006", "SHBHCUT007", "typo"),
        QualityRepairOperation("The unit", "The practical unit that I use with learners in the classroom", "style"),
    ])

    assert repaired == text
    assert applied == []
    assert {row["skip_reason"] for row in skipped} == {"protected_token_changed", "replacement_too_expansive"}


def test_v6_quality_repair_runs_once_after_selected_rewrite(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_NATURALISATION_ENABLED", "0")
    source = "This process uses a form, a queue, and a review."
    rewritten = "I am an educator who need to listen."
    seen = {}

    class QualityClient:
        model = "grammer-test"

        def __init__(self):
            self.calls = 0

        def chat(self, _prompt, **kwargs):
            self.calls += 1
            seen["app_label"] = kwargs.get("app_label")
            return StaticJsonResponse(json.dumps({
                    "operations": [{
                    "find": "who need",
                    "replace": "who needs",
                    "reason": "agreement",
                }]
            }))

    def fake_run(current, **_kwargs):
        scan = scan_text(current)
        paragraph, plan = build_plan(scan)
        return v6_pipeline.Result(scan=scan, plan=plan, variants=[], selected=None, rewritten_text=rewritten)

    quality = QualityClient()
    monkeypatch.setattr(v6_pipeline, "run_v6_rewrite", fake_run)
    monkeypatch.setattr(v6_pipeline, "_acceptable_progress", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(v6_pipeline, "_cross_paragraph_regression", lambda *_args, **_kwargs: False)

    result = run_v6_rewrite_all(source, quality_client=quality, max_passes=1, residual_followup_passes=0)

    assert quality.calls == 1
    assert seen["app_label"] == "Grammer"
    assert result.final_text_before_quality_repair == rewritten
    assert result.rewritten_text == "I am an educator who needs to listen."
    assert result.quality_repair is not None
    assert result.quality_repair.status == "applied"


def test_v6_quality_repair_reverts_one_pass_scan_regression(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_NATURALISATION_ENABLED", "0")
    source = "This process uses a form, a queue, and a review."
    rewritten = "Approach increases student motivation."

    class QualityClient:
        model = "grammer-test"

        def chat(self, _prompt, **kwargs):
            return StaticJsonResponse(json.dumps({
                "operations": [{
                    "find": "Approach increases",
                    "replace": "This approach increases",
                    "reason": "fragment",
                }]
            }))

    def fake_run(current, **_kwargs):
        scan = scan_text(current)
        paragraph, plan = build_plan(scan)
        return v6_pipeline.Result(scan=scan, plan=plan, variants=[], selected=None, rewritten_text=rewritten)

    monkeypatch.setattr(v6_pipeline, "run_v6_rewrite", fake_run)
    monkeypatch.setattr(v6_pipeline, "_acceptable_progress", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(v6_pipeline, "_cross_paragraph_regression", lambda *_args, **_kwargs: False)

    monkeypatch.setenv("DRAFTPROOF_V6_GRAMMER_RISK_TOLERANCE", "0")

    result = run_v6_rewrite_all(source, quality_client=QualityClient(), max_passes=1, residual_followup_passes=0)

    assert result.rewritten_text == rewritten
    assert result.quality_repair is not None
    assert result.quality_repair.status == "reverted_scan_regression"
    assert result.quality_repair.skipped_operations[-1]["skip_reason"] == "scan_regression"


def test_v6_quality_repair_allows_minor_shape_risk_increase(monkeypatch):
    from poc.rewrite_v6.quality_repair import QualityRepairResult

    original = "Original text."
    repaired = "Repaired text."

    def fake_scan(text):
        scan = scan_text("The process uses a form.")
        scores = dict(scan.scores)
        scores["finding_count"] = 3.0
        scores["mean_sentence_shape_risk"] = 5.0 if text == original else 6.0
        return scan.__class__(
            source_text=scan.source_text,
            paragraphs=scan.paragraphs,
            findings=scan.findings,
            scores=scores,
        )

    monkeypatch.setattr(v6_pipeline, "scan_text", fake_scan)
    monkeypatch.setenv("DRAFTPROOF_V6_GRAMMER_RISK_TOLERANCE", "1")

    result = v6_pipeline._risk_safe_quality_repair(
        original,
        QualityRepairResult(original_text=original, repaired_text=repaired, status="applied"),
    )

    assert result is not None
    assert result.repaired_text == repaired
    assert result.status == "applied"


def test_v6_document_rewrite_honors_cancellation_before_llm():
    class Canceled(BaseException):
        pass

    writer = FakeClient()

    try:
        run_v6_rewrite_all(
            "A process uses a form, a queue, and a review.",
            writer_client=writer,
            cancellation_check=lambda: (_ for _ in ()).throw(Canceled()),
        )
    except Canceled:
        pass
    else:
        raise AssertionError("Expected cancellation to stop V6 before LLM calls")

    assert writer.calls == 0


def test_v6_production_adapter_returns_worker_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(v6_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))
    monkeypatch.setattr(v6_production, "_scan_report", fake_full_scan_report)
    events = []
    result = v6_production.run_rewrite_pipeline_v6(
        detect_json={"input_text": "The review moved from intake to approval."},
        output_dir=str(tmp_path),
        model="qwen/qwen3-30b-a3b-instruct-2507",
        progress_callback=lambda percent, message: events.append((percent, message)),
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
    assert any(percent > 62 for percent, _message in events)
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


def test_v6_rewrite_pdf_stamp_uses_calibrated_authorship_band():
    summary = {
        "outcome": "partial_candidate_not_strict_safe",
        "status": "partial_candidate_not_strict_safe",
        "detect_scores": {
            "original_ai_authorship": 70,
            "rewritten_ai_authorship": 43,
            "rewritten_human_contribution": 72,
            "rewritten_ai_transformation": 28,
        },
        "detect_scan_original": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {"ai_likelihood_score": 70, "writing_quality_score": 60},
        },
        "detect_scan_rewritten": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {
                "ai_likelihood_score": 43,
                "writing_quality_score": 72,
                "authorship_rating": {"label": "AI-Generated / AI-Paraphrased Signals", "short_label": "AI Signals"},
            },
            "scan_intelligence": {
                "transformation": {
                    "contribution": {
                        "human_contribution_ratio": 72,
                        "ai_transformation_ratio": 28,
                    }
                }
            },
        },
    }

    report = render_rewrite_report(summary, [], [], original_text="Original.", final_text="Rewritten.")

    assert "UNLIKELY AI-ASSISTED" in report
    assert "<h3>Unlikely AI-Assisted</h3>" in report
    assert "22% calibrated risk · review threshold exceeded" in report


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


def test_v6_selection_rejects_polarity_inversion_even_when_scanner_improves():
    source = (
        "This shift has made the role of reviewers more important, not less important. "
        "Reviewers guide clients to question sources, compare notes, develop judgment, and apply decisions."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "The shift is reducing the role of reviewers in this process. "
                    "Reviewers guide clients to question sources and compare notes. "
                    "Judgment and decisions carry the next part."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"


def test_v6_selection_rejects_polarity_omission_even_when_scanner_improves():
    source = (
        "This shift has made the role of reviewers more important, not less important. "
        "Reviewers guide clients to question sources, compare notes, develop judgment, and apply decisions."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "The shift changes the reviewer role in this process. "
                    "Reviewers guide clients to question sources and compare notes. "
                    "Judgment and decisions carry the next part."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"


def test_v6_selection_accepts_negated_reduction_for_not_less_polarity():
    source = (
        "This shift has made the role of reviewers more important, not less important. "
        "Reviewers guide clients to question sources and compare notes."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "The shift does not reduce the reviewer role. "
                    "Reviewers guide clients to question sources and compare notes."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "llm"


def test_v6_selection_rejects_unreviewed_bridge_terms_when_alternative_is_grounded():
    source = (
        "This shift has made the role of reviewers more important, not less important. "
        "Reviewers guide clients to question sources and compare notes."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "bad",
                "text": (
                    "Digital platforms changed the reviewer role, which remains important, not less. "
                    "Reviewers guide clients to question sources and compare notes."
                ),
            },
            {
                "id": "good",
                "text": (
                    "The shift keeps the reviewer role more important, not less. "
                    "Reviewers guide clients to question sources and compare notes."
                ),
            },
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.id == "good"


def test_v6_selection_accepts_material_risk_drop_without_finding_count_drop():
    source = (
        "This shift has made the role of reviewers even more important, not less important. "
        "Reviewers guide clients to question sources, compare viewpoints, develop judgment, and apply knowledge in real situations."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "The shift makes the reviewer role even more important, not less. "
                    "Reviewers guide clients to question sources and compare viewpoints. "
                    "They help clients develop judgment and apply knowledge in real situations."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "llm"


def test_v6_selection_accepts_finding_drop_when_risk_not_worse():
    source = (
        "Inclusive learning design must move beyond passive accommodation and instead focus on embedding metacognitive strategies. "
        "Complex spatial tasks such as forms, checks, queues, workflows, reviewers, and decisions strain working memory. "
        "The process demonstrates the difference between giving a person an answer and teaching them how to work."
    )
    candidate = (
        "Inclusive learning design moves beyond passive accommodation by embedding metacognitive strategies. "
        "Complex spatial tasks can strain working memory. "
        "Forms and checks carry one part of the process. "
        "Queues and reviewer decisions carry the next part. "
        "The process compares giving a person an answer with teaching them how to work."
    )
    paragraph = scan_text(source).paragraphs[0]
    selected = choose_variant([
        Variant(id="source_preserved", text=source, source="source_preserved"),
        Variant(id="v1", text=candidate, source="llm"),
    ], paragraph)

    assert selected is not None
    assert selected.id == "v1"


def test_v6_selected_bridge_terms_are_marked_for_author_review():
    source = (
        "This result shows an important concern because the process should improve. "
        "Teams use a form, a queue, and a review."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "The process carries an improvement concern. "
                    "Teams use forms and queues before review coordination."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "llm"
    assert result.selected.author_review_items
    assert result.selected.author_review_items[0]["provenance"] == "needs_author_confirmation"


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


def test_v6_selection_rejects_partial_four_item_source_list_coverage():
    source = (
        "The process still carries old habits. "
        "It rewards people who can compare, adjust, explain, and build."
    )
    writer_payload = json.dumps({
        "variants": [
            {
                "id": "v1",
                "text": (
                    "The process still carries old habits. "
                    "The process rewards people who can compare and adjust."
                ),
            }
        ]
    })
    result = run_v6_rewrite(source, writer_client=StaticJsonClient(writer_payload))
    assert result.selected is not None
    assert result.selected.source == "source_preserved"


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


def test_v6_glm47_planner_prefers_cerebras_provider(monkeypatch):
    for name in ("ROUTING_JSON", "ORDER", "ONLY", "IGNORE", "SORT"):
        monkeypatch.delenv(f"DRAFTPROOF_V6_PLANNER_PROVIDER_{name}", raising=False)
    monkeypatch.delenv("DRAFTPROOF_V6_PLANNER_ALLOW_FALLBACKS", raising=False)

    assert _planner_provider("z-ai/glm-4.7") == {"order": ["Cerebras"], "allow_fallbacks": True}


def test_v6_glm47_provider_compatibility_prefers_cerebras(monkeypatch):
    for name in ("PROVIDER_ROUTING_JSON", "PROVIDER_ORDER", "PROVIDER_ONLY", "PROVIDER_IGNORE", "PROVIDER_SORT", "PROVIDER_ALLOW_FALLBACKS"):
        monkeypatch.delenv(f"DRAFTPROOF_V6_WRITER_{name}", raising=False)

    assert _writer_provider("z-ai/glm-4.7") == {"order": ["Cerebras"], "allow_fallbacks": True}


def test_v6_planner_provider_env_overrides_default(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_PLANNER_PROVIDER_ORDER", "Cerebras,Fireworks")
    monkeypatch.setenv("DRAFTPROOF_V6_PLANNER_PROVIDER_SORT", "throughput")
    monkeypatch.setenv("DRAFTPROOF_V6_PLANNER_ALLOW_FALLBACKS", "0")

    assert _planner_provider("z-ai/glm-4.7") == {
        "order": ["Cerebras", "Fireworks"],
        "allow_fallbacks": False,
        "sort": "throughput",
    }
