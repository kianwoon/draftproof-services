from __future__ import annotations

import json
from dataclasses import replace

from poc.rewrite_v6.author_proxy import attach_author_proxy_pack, author_proxy_required
from poc.rewrite_v6.finding_pattern import classify_finding_pattern
from poc.rewrite_v6.pipeline import (
    _needs_writer_feedback_round,
    _select_variant,
    _selector_prompt,
    _writer_feedback_rounds,
    run_v6_rewrite,
)
from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6.prose_repair_rules import consolidated_prose_repair_rules
from poc.rewrite_v6.planner_llm import run_planner_llm
from poc.rewrite_v6.planner_llm import build_planner_prompt
from poc.rewrite_v6.scan import scan_text
from poc.rewrite_v6.selector_diagnostics import selection_diagnostics
from poc.rewrite_v6.write import build_prompt, write_variants
from poc.rewrite_v6.write import Variant
from poc.rewrite_v6.writer_feedback import needs_writer_feedback_retry, plan_with_writer_feedback


class EmptyVariantClient:
    def chat(self, prompt: str, **_kwargs):
        return type("Response", (), {"content": '{"variants":[]}', "raw_content": '{"variants":[]}'})()


class StaticJsonClient:
    def __init__(self, payload: str):
        self.payload = payload
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def chat(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(dict(kwargs))
        return type("Response", (), {"content": self.payload, "raw_content": self.payload})()


def test_v6_writer_generation_failure_is_visible_in_diagnostics():
    source = "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve."

    result = run_v6_rewrite(source, writer_client=EmptyVariantClient())

    assert result.selected.source == "source_preserved"
    assert result.candidate_diagnostics
    row = result.candidate_diagnostics[0]
    assert row["variant_id"] == "writer_generation"
    assert "writer_generation_failed" in row["blockers"]
    assert row["selector_source"] == "no_generated_variants_source_preserved"


def test_v6_finding_pattern_classifies_generic_paragraph_routes():
    cases = [
        (
            "benefit_risk_contrast",
            "The tool creates opportunities and risks. It can help teams plan, review, and improve work. Used well, it supports people who need help. The danger is that users may depend on generated answers without checking the work. This makes review difficult and raises questions about fairness and trust.",
        ),
        (
            "old_model_current_mismatch",
            "The system is changing faster than many teams can manage. In the past, the model was built around paper forms and manager approval. That model still exists, but it no longer reflects how people work today.",
        ),
        (
            "recommendation_process",
            "The process needs to evolve. Teams should keep core checks, but they must also include review notes, feedback, and improvement. The assessment should include final results and the working process.",
        ),
        (
            "turning_point_conclusion",
            "The organisation stands at a turning point. It can either keep old methods or redesign work for the reality teams now face. The goal should not be to protect the past. The goal should be to help people work responsibly.",
        ),
        (
            "old_model_current_mismatch",
            "However, the current process still carries many old habits. Teams still place pressure on forms, queues, and standard answers. This is a serious concern because the modern workplace rewards people who can compare, adapt, communicate, and build.",
        ),
    ]
    for expected, text in cases:
        scan = scan_text(text)
        _paragraph, plan = build_plan(scan, None, {"p001"})

        pattern = classify_finding_pattern(plan)

        assert pattern["pattern_id"] == expected


def test_v6_consolidated_prose_rules_cover_user_repair_contract():
    rules = {row["problem"]: row for row in consolidated_prose_repair_rules()}

    assert len(rules) == 25
    assert rules["balanced_both_sides_opener"]["simple_rule"] == "Do not announce benefits and risks. Show the tension."
    assert rules["used_well_template"]["preferred_shape"] == "when actor uses tool to do a specific action"
    assert rules["passive_institutional_voice"]["preferred_shape"] == "actor has a harder time judging/doing the task"
    assert rules["dependency_claim"]["preferred_shape"] == "visible behaviour + learning gap"
    assert rules["polished_work_claim"]["simple_rule"] == "Show the mismatch between output and ability."


def test_v6_writer_prompt_carries_consolidated_prose_rules_without_fixture_specific_filters():
    source = (
        "The review process helps teams compare drafts, check decisions, and improve work. "
        "However, teams may rely on the checklist without understanding why changes were made."
    )
    paragraph, plan = build_plan(scan_text(source))

    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    rules = {row["problem"]: row for row in payload["global_prose_repair_rules"]}

    assert rules["balanced_both_sides_opener"]["preferred_shape"] == "benefit + specific risk"
    assert rules["dependency_claim"]["preferred_shape"] == "visible behaviour + learning gap"
    assert payload["planner_brief"]["flow_plan"]
    assert "source_paragraph" in payload
    assert "writer_execution_plan" in payload


def test_v6_planner_prompt_uses_schema_contract_without_fixture_specific_risky_shapes():
    source = "The process helps teams review decisions, but repeated checks can weaken judgement."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)

    payload = json.loads(build_planner_prompt(paragraph, plan, scan.findings).split("\n", 1)[1])
    rule_text = json.dumps(payload["global_prose_repair_rules"], ensure_ascii=False)
    policy = payload["risky_source_shape_policy"]
    profile = payload["planner_agent_profile"]
    output_contract = payload["output_contract"]
    rule_text_full = "\n".join(payload["rules"])

    assert profile["role"] == "Planner agent"
    assert "Scanner findings are symptoms, not the final rewrite plan." in profile["input_boundary"]
    assert "Build 2-4 flow_plan steps that describe paragraph movement, not sentence repairs." in profile["operating_procedure"]
    assert "flow_plan must tell Writer what each paragraph beat does." in profile["output_contract"]
    assert "separate meaning-to-preserve from wording-to-copy" in profile["uses_basic_rules_to"]
    assert "Keep risky source terms out of flow_plan.must_include." in profile["must_do"]
    assert "No risky source term appears in flow_plan.must_include." in profile["self_check_before_return"]
    assert "balanced_both_sides_opener" in rule_text
    assert "used_well_template" in rule_text
    assert isinstance(policy["risky_terms_not_wording_requirements"], list)
    assert isinstance(policy["risky_shapes_in_this_paragraph"], list)
    shape = output_contract["required_output_shape"]["planner_decision"]
    assert "validated_construction_route" in shape
    assert "sentence_jobs" in shape["validated_construction_route"]
    assert "coverage_terms" in shape
    assert "safe_sentence_shape" not in json.dumps(output_contract)
    assert "finding_contracts" not in output_contract["required_keys"]
    assert "paragraph_blueprint" not in output_contract["required_keys"]
    assert "human_route" not in output_contract["required_keys"]
    assert "semantic_role_map" not in output_contract["required_keys"]
    assert "Do not use the field name sentence_jobs." not in rule_text_full
    assert "Use validated_construction_route.sentence_jobs only for ordered route logic" in rule_text_full


def test_v6_planner_merge_keeps_llm_flow_plan_source_scoped():
    source = "The process helps teams review decisions. Repeated checks can weaken judgement."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision()
    decision["flow_plan"] = [
        {
            "step_id": "opening",
            "function": "Show the practical process.",
            "source_basis": ["p001_s001"],
            "must_include": ["process", "teams", "review"],
        },
        {
            "step_id": "support",
            "function": "Show the concern.",
            "source_basis": ["p001_s002"],
            "must_include": ["checks", "judgement"],
        },
    ]
    planner = StaticJsonClient(json.dumps({"planner_decision": decision}))

    planned = run_planner_llm(paragraph, plan, scan.findings, client=planner)
    merged = planned.ai_safe_route["llm_planner_decision"]
    must_terms = [
        str(term).casefold()
        for row in merged["flow_plan"]
        for term in row.get("must_include", [])
    ]

    assert "process" in must_terms
    assert "judgement" in must_terms
    assert merged["validated_construction_route"]["sentence_jobs"]


def test_v6_planner_merge_combines_llm_and_source_coverage_terms():
    source = "The process helps teams review decisions. Repeated checks can weaken judgement."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision()
    decision["coverage_terms"] = ["review decisions", "weaken judgement"]
    planner = StaticJsonClient(json.dumps({"planner_decision": decision}))

    planned = run_planner_llm(paragraph, plan, scan.findings, client=planner)
    terms = planned.ai_safe_route["llm_planner_decision"]["coverage_terms"]
    lowered = [term.casefold() for term in terms]

    assert "review decisions" in terms
    assert "weaken judgement" in terms
    assert any(term in lowered for term in ("process", "teams"))


def test_v6_planner_merge_enforces_source_derived_copy_risk_guardrail():
    source = (
        "Technology has also created both opportunities and risks. "
        "AI tools can help students brainstorm ideas, explain difficult topics, improve writing, and practise skills. "
        "Used well, technology can support personalised learning and reduce barriers for students who need extra help. "
        "But there is also a danger. "
        "Students may become too dependent on AI-generated answers without understanding the work behind them. "
        "They may submit polished work that does not reflect their real ability. "
        "This makes assessment more difficult and raises questions about fairness, originality, and learning integrity."
    )
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision()
    decision["flow_plan"] = [
        {
            "step_id": "fp001",
            "function": "Open with the broad balanced claim.",
            "source_basis": ["p001_s001"],
            "must_include": ["technology creates opportunities and risks", "both opportunities and risks"],
        },
        {
            "step_id": "fp002",
            "function": "Carry the good-use condition.",
            "source_basis": ["p001_s002", "p001_s003"],
            "must_include": [
                "AI tools help students brainstorm ideas, explain difficult topics, improve writing, and practise skills",
                "Used well",
            ],
        },
    ]
    decision["validated_construction_route"] = {
        "route_id": "planner_route",
        "movement": "support -> risk -> consequence",
        "sentence_jobs": [
            {
                "job_id": "support",
                "job": "Carry source support.",
                "source_basis": ["p001_s001", "p001_s002", "p001_s003"],
                "must_use_meaning": [
                    "Technology creates both opportunities and risks",
                    "Used well",
                ],
                "must_not_use": [],
            },
            {
                "job_id": "consequence",
                "job": "Carry the closing consequence.",
                "source_basis": ["p001_s006", "p001_s007"],
                "must_use_meaning": [
                    "This makes assessment more difficult and raises questions about fairness, originality, and learning integrity",
                ],
                "must_not_use": [],
            },
        ],
        "do_not_copy": [],
        "validation_rules": [],
    }
    decision["coverage_terms"] = [
        "both opportunities and risks",
        "Used well",
        "raises questions about",
    ]
    planner = StaticJsonClient(json.dumps({"planner_decision": decision}))

    planned = run_planner_llm(paragraph, plan, scan.findings, client=planner)
    merged = planned.ai_safe_route["llm_planner_decision"]
    merged_text = json.dumps(merged, ensure_ascii=False)
    blocked = {item.casefold() for item in merged["do_not_copy_route"]}

    assert "both opportunities and risks" in blocked
    assert "used well" in blocked
    assert "raises questions about" in blocked
    assert "both opportunities and risks" not in json.dumps(merged["flow_plan"], ensure_ascii=False).casefold()
    assert "Used well" not in json.dumps(merged["validated_construction_route"], ensure_ascii=False)
    assert "raises questions about" not in json.dumps(merged["coverage_terms"], ensure_ascii=False).casefold()
    assert "[copy-blocked source wording]" in merged_text
    assert merged["coverage_beat_gaps"] == []


def test_v6_planner_merge_builds_validated_construction_route_from_planner_shape():
    source = "The process helps teams review decisions. Repeated checks can weaken judgement."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    planner = StaticJsonClient(json.dumps({"planner_decision": _planner_decision()}))

    planned = run_planner_llm(paragraph, plan, scan.findings, client=planner)
    route = planned.ai_safe_route["llm_planner_decision"]["validated_construction_route"]
    merged = planned.ai_safe_route["llm_planner_decision"]

    assert route["route_id"]
    assert len(route["sentence_jobs"]) >= 2
    assert all(not item.startswith("Do not ") for item in merged["do_not_copy_route"])
    assert all(";" not in item for item in merged["do_not_copy_route"])
    meanings = [row["meaning"] for row in merged["coverage_beats"]]
    assert all(all(len(item.split()) <= 9 for item in row["must_cover"]) for row in merged["coverage_beats"])
    assert all(
        phrase.casefold() not in row["meaning"].casefold()
        for row in merged["coverage_beats"]
        for phrase in row["must_not_copy"]
    )
    assert merged["coverage_beat_gaps"] == []
    assert all(len(meaning.split()) > 2 for meaning in meanings)
    assert "finding_contracts" not in merged
    assert "paragraph_blueprint" not in merged
    assert "safe_sentence_shape" not in json.dumps(merged)


def test_v6_planner_route_alignment_tracks_raw_to_merged_intent():
    source = "The process helps teams review decisions. Repeated checks can weaken judgement."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision()
    decision["validated_construction_route"] = {
        "route_id": "planner_route",
        "movement": "support -> dependence risk -> polished work mismatch -> assessment consequence",
        "sentence_jobs": [
            {"job_id": "sj001", "job": "show process support", "source_basis": ["p001_s001"], "must_use_meaning": ["review decisions"]},
            {"job_id": "sj002", "job": "show concern", "source_basis": ["p001_s002"], "must_use_meaning": ["weaken judgement"]},
        ],
        "do_not_copy": [],
        "validation_rules": [],
    }
    planner = StaticJsonClient(json.dumps({"planner_decision": decision}))

    planned = run_planner_llm(paragraph, plan, scan.findings, client=planner)
    merged = planned.ai_safe_route["llm_planner_decision"]

    assert merged["validated_construction_route"]["route_id"] == "planner_route"
    assert merged["raw_merged_route_alignment"] is True


def test_v6_planner_replaces_invalid_construction_route_before_writer_handoff():
    source = "The process helps teams review decisions. Repeated checks can weaken judgement."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision()
    decision["validated_construction_route"] = {
        "route_id": "bad_route",
        "movement": "topic -> list -> conclusion",
        "sentence_jobs": [
            {"job_id": "topic", "job": "Announce the topic."},
            {"job_id": "list", "job": "List the points."},
        ],
        "do_not_copy": [],
        "validation_rules": [],
    }
    planner = StaticJsonClient(json.dumps({"planner_decision": decision}))

    planned = run_planner_llm(paragraph, plan, scan.findings, client=planner)
    route = planned.ai_safe_route["llm_planner_decision"]["validated_construction_route"]

    assert route["route_id"] != "bad_route"
    assert len(route["sentence_jobs"]) >= 2
    assert all(row["job_id"] != "topic" for row in route["sentence_jobs"])


def test_v6_writer_prompt_profiles_writer_as_rule_using_agent():
    source = (
        "Technology has also created both opportunities and risks. "
        "AI tools can help students brainstorm ideas, explain difficult topics, improve writing, and practise skills."
    )
    paragraph, plan = build_plan(scan_text(source))

    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    profile = payload["writer_agent_profile"]

    assert profile["role"] == "Writer agent"
    assert "Planner route is the operating plan." in profile["input_boundary"]
    assert "Follow the Planner paragraph_route before choosing sentence wording." in profile["operating_procedure"]
    assert "Each variant must attempt scanner movement through route change, not synonym swaps." in profile["output_contract"]
    assert "show visible behaviour before abstract consequence" in profile["uses_basic_rules_to"]
    assert "Do not announce a broad topic opener when a source-specific tension is available." in profile["must_not_do"]
    assert "No variant repeats the same consequence." in profile["self_check_before_return"]
    assert any(rule["problem"] == "balanced_both_sides_opener" for rule in profile["basic_rules"])


def test_v6_writer_prompt_uses_planner_coverage_beats_as_meaning_obligations():
    source = (
        "The review process has a vague concern tail. "
        "Teams use checklists to compare drafts and check decisions. "
        "Repeated checks can weaken judgement."
    )
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision()
    decision["do_not_copy_route"] = ["vague concern tail"]
    decision["route_rewrite_guidance"] = ["Preserve the concern through a concrete source condition."]
    decision["flow_plan"] = [
        {"step_id": "fp001", "function": "Name the local review condition.", "source_basis": ["p001_s001"]},
        {"step_id": "fp002", "function": "Show what teams do in the process.", "source_basis": ["p001_s002"]},
        {"step_id": "fp003", "function": "Close with the source consequence.", "source_basis": ["p001_s003"]},
    ]
    decision["validated_construction_route"] = {
        "route_id": "review_process_consequence_route",
        "movement": "local condition -> team action -> consequence",
        "sentence_jobs": [
            {
                "job_id": "local_condition",
                "job": "Name the review condition without copying the blocked source phrase.",
                "source_basis": ["p001_s001"],
                "must_use_meaning": ["review process concern"],
                "must_not_use": ["vague concern tail"],
            },
            {
                "job_id": "team_action",
                "job": "Show the team action inside the process.",
                "source_basis": ["p001_s002"],
                "must_use_meaning": ["teams use checklists"],
                "must_not_use": [],
            },
            {
                "job_id": "consequence",
                "job": "Close with the source consequence.",
                "source_basis": ["p001_s003"],
                "must_use_meaning": ["weaken judgement"],
                "must_not_use": [],
            },
        ],
        "do_not_copy": ["vague concern tail"],
        "validation_rules": ["Closing consequence must stay in the final beat."],
    }
    planned = run_planner_llm(paragraph, plan, scan.findings, client=StaticJsonClient(json.dumps({"planner_decision": decision})))

    payload = json.loads(build_prompt(paragraph, planned).split("\n", 1)[1])
    prompt_text = json.dumps(payload)

    assert payload["coverage_beats"]
    assert "[copy-blocked source wording]" in payload["source_paragraph"]
    assert "vague concern tail" not in payload["source_paragraph"]
    assert "vague concern tail" not in prompt_text
    assert "Use coverage_beats as meaning obligations; coverage_terms are backup anchors only." in payload["style_contract"]
    assert set(payload["planner_brief"]) == {
        "status",
        "paragraph_route",
        "validated_construction_route",
        "coverage_beats",
        "do_not_copy_route",
        "route_rewrite_guidance",
        "coverage_terms",
        "author_proxy_request",
    }
    assert "raw_merged_route_alignment" not in payload["planner_brief"]
    assert "coverage_beat_gaps" not in payload["planner_brief"]
    assert payload["writer_handoff_priority"] == [
        "writer_execution_plan",
        "validated_construction_route.sentence_jobs",
        "coverage_beats",
        "coverage_beats.must_not_copy",
        "planner_brief.do_not_copy_route",
        "planner_brief.route_rewrite_guidance",
        "planner_brief.coverage_terms",
    ]
    assert "Follow writer_handoff_priority; never use coverage_terms as the main plan." in payload["style_contract"]
    assert "Use route_rewrite_guidance as route guidance, not wording to copy." in payload["style_contract"]
    assert "Obey route_sequence_guards before writing any variant." in payload["style_contract"]
    assert payload["writer_execution_plan"]
    assert payload["writer_execution_plan"][0]["route_job_id"] == "local_condition"
    assert "review process concern" in payload["writer_execution_plan"][0]["meaning_targets"]
    assert payload["route_sequence_guards"][0]["allowed_only_in_route_job"] == "consequence"
    assert "team_action" in payload["route_sequence_guards"][0]["must_appear_after_route_jobs"]
    assert any("checklists" in row["meaning"] for row in payload["coverage_beats"])
    assert all(len(row["meaning"].split()) > 2 for row in payload["coverage_beats"])


def test_v6_selector_prompt_carries_consolidated_prose_rules():
    paragraph = scan_text("AI can help students, but teachers must judge whether learning happened.").paragraphs[0]
    variants = [Variant(id="v1", source="llm", text="AI can help students understand work, but teachers must still judge learning.")]
    diagnostics = [
        {
            "variant_id": "v1",
            "blockers": [],
            "accepted_by_selector": True,
            "candidate_findings": 1,
            "candidate_mean_risk": 10.0,
            "risk_drop": 5.0,
        }
    ]

    payload = json.loads(_selector_prompt(paragraph, variants, diagnostics).split("\n", 1)[1])
    rule_text = " ".join(payload["global_prose_repair_rules"])
    profile = payload["selector_agent_profile"]

    assert profile["role"] == "Selector agent"
    assert "Selector must not write or edit prose." in profile["input_boundary"]
    assert "Among eligible candidates, choose lowest candidate_findings first." in profile["operating_procedure"]
    assert "Choosing a smoother candidate with worse findings." in profile["failure_modes_to_prevent"]
    assert "balanced_both_sides_opener" in rule_text
    assert "passive_institutional_voice" in rule_text
    assert "polished_work_claim" in rule_text


def test_v6_writer_feedback_stops_after_material_progress_candidate():
    diagnostics = [
        {
            "variant_id": "v1",
            "source": "llm",
            "accepted_by_selector": True,
            "blockers": [],
            "quality_warnings": [],
            "candidate_findings": 1,
            "source_findings": 3,
            "candidate_mean_risk": 21.0,
            "source_mean_risk": 28.0,
            "risk_drop": 7.0,
        },
        {
            "variant_id": "v2",
            "source": "llm",
            "accepted_by_selector": False,
            "blockers": ["compressed_final_consequence_list"],
            "candidate_findings": 2,
            "source_findings": 3,
            "risk_drop": 2.0,
        },
    ]

    assert needs_writer_feedback_retry(diagnostics) is False
    assert _needs_writer_feedback_round(diagnostics, feedback_round=0) is True
    assert _needs_writer_feedback_round(diagnostics, feedback_round=1) is False


def test_v6_writer_feedback_stops_after_material_progress_even_with_residual_findings():
    diagnostics = [
        {
            "variant_id": "v1",
            "source": "llm",
            "accepted_by_selector": True,
            "blockers": [],
            "quality_warnings": [],
            "candidate_findings": 3,
            "source_findings": 5,
            "candidate_mean_risk": 24.0,
            "source_mean_risk": 38.0,
            "risk_drop": 14.0,
        }
    ]

    assert needs_writer_feedback_retry(diagnostics) is False


def test_v6_writer_feedback_default_budget_is_small(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_WRITER_FEEDBACK_ROUNDS", raising=False)

    assert _writer_feedback_rounds() == 1


def test_v6_planner_source_contract_preserves_negation_scope_markers():
    source = (
        "Students may learn how to pass, but not always how to think deeply, "
        "solve problems, or connect ideas across subjects."
    )
    scan = scan_text(source)
    _paragraph, plan = build_plan(scan, None, {"p001"})
    terms = plan.actions[0].preserve_terms

    assert "not always" in terms
    assert "always" in terms
    assert plan.to_dict()["handoff_validation"]["planner_to_writer_contract"] == "passed"


def test_v6_full_orchestration_selects_valid_rewrite_across_generic_patterns():
    cases = [
        (
            "old_model_current_mismatch",
            "The system is changing faster than many teams can manage. In the past, the model was built around paper forms, manager approval, and weekly reports. That model still exists, but it no longer reflects how people work today.",
            "The system is changing faster than many teams can manage. In the past, work was built around paper forms and manager approval. Weekly reports were part of that model too. That model still exists, but it no longer reflects how people work today.",
        ),
            (
                "benefit_risk_contrast",
                "The tool creates opportunities and risks. It can help teams plan, review, and improve work. Used well, it supports people who need help. The danger is that users may depend on generated answers without checking the work. This makes review difficult and raises questions about fairness, originality, and trust.",
                "Technology creates opportunities and risks for teams. Teams can use the tool to plan and review work, then improve it when used well. The risk comes when users depend on generated answers without checking the work behind them. That dependence makes review difficult. Reviewers must judge whether the work is fair and original, and whether it can be trusted.",
            ),
        (
            "turning_point_conclusion",
            "The organisation stands at a turning point. It can either keep old methods in a new workplace, or it can redesign work for the reality teams now face. The goal should not be to protect the past. The goal should be to help people work responsibly.",
            "The organisation stands at a turning point. Leaders can keep old methods in a new workplace, or redesign work for the reality teams now face. The better aim is not protecting the past. The aim is helping people work responsibly.",
        ),
    ]
    for expected_pattern, source, rewrite in cases:
        writer = StaticJsonClient(json.dumps({"variants": [{"id": "v1", "text": rewrite}]}))
        selector = StaticJsonClient(json.dumps({"selected_id": "v1", "rationale": "valid rewrite"}))

        result = run_v6_rewrite(source, writer_client=writer, selector_client=selector)

        assert classify_finding_pattern(result.plan)["pattern_id"] == expected_pattern
        assert result.selected.id == "v1"
        assert result.selected.source == "llm"
        assert result.rewritten_text != source
        assert result.candidate_diagnostics[0]["selected_by_selector"] is True


def test_v6_author_proxy_feeds_planner_before_writer_handoff():
    source = "This result shows a serious concern because the process should improve across teams."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    findings = scan.findings
    request_decision = _planner_decision(
        author_proxy_request={
            "required": True,
            "reason": "author_anchor_gap requires submitted context before the broad claim can be scoped",
            "target_sentence_ids": ["p001_s001"],
            "finding_basis": ["author_anchor_gap"],
        },
        author_proxy_plan="request a small local grounding bridge",
    )
    planner = StaticJsonClient(json.dumps({"planner_decision": request_decision}))

    planned = run_planner_llm(paragraph, plan, findings, client=planner)

    assert author_proxy_required(planned) is True

    proxy = StaticJsonClient(json.dumps({
        "proxy_pack": {
            "usable_bridges": [
                {
                    "bridge_id": "b001",
                    "anchor_source": "previous_paragraph",
                    "anchor_text": "The team had queue delays and unclear ownership before the review.",
                    "usable_bridge": "queue delays and unclear ownership",
                    "target_sentence_ids": ["p001_s001"],
                    "integration_role": "scope broad claim",
                    "integration_instruction": "Use this as the local reason for the concern.",
                    "review_reason": "Author should confirm the bridge.",
                }
            ],
            "planner_guidance": {"use": "Ground the concern with the local operational reason.", "avoid": []},
            "author_review_items": [
                {
                    "provenance": "inferred_from_draft",
                    "target_text": "queue delays and unclear ownership",
                    "user_input_needed": "Confirm, replace, or remove before final submission.",
                }
            ],
        }
    }))

    packed = attach_author_proxy_pack(paragraph, planned, findings, client=proxy)

    assert author_proxy_required(packed) is False
    assert packed.paragraph_strategy["author_proxy_pack"]["usable_bridges"]

    replan_decision = _planner_decision(
        author_proxy_request={"required": False, "reason": "proxy pack is now integrated", "target_sentence_ids": []},
        author_proxy_plan="Use the operational reason to scope the concern inside the same source beat.",
    )
    replanner = StaticJsonClient(json.dumps({"planner_decision": replan_decision}))

    replanned = run_planner_llm(paragraph, packed, findings, client=replanner)

    assert "author_proxy_pack" in replanner.prompts[0]
    assert "queue delays and unclear ownership" in replanner.prompts[0]
    planner_decision = replanned.ai_safe_route["llm_planner_decision"]
    assert "queue delays and unclear ownership" in planner_decision["coverage_terms"]
    assert "author_proxy_plan" not in planner_decision


def test_v6_author_proxy_malformed_json_degrades_without_task_retry():
    source = "This result shows a serious concern because the process should improve across teams."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    findings = scan.findings
    request_decision = _planner_decision(
        author_proxy_request={
            "required": True,
            "reason": "author_anchor_gap requires submitted context before the broad claim can be scoped",
            "target_sentence_ids": ["p001_s001"],
            "finding_basis": ["author_anchor_gap"],
        },
    )
    planned = run_planner_llm(
        paragraph,
        plan,
        findings,
        client=StaticJsonClient(json.dumps({"planner_decision": request_decision})),
    )

    packed = attach_author_proxy_pack(
        paragraph,
        planned,
        findings,
        client=StaticJsonClient('{"proxy_pack":{"usable_bridges":[{"bridge_id":"b001"}]'),
    )

    pack = packed.paragraph_strategy["author_proxy_pack"]
    assert author_proxy_required(packed) is False
    assert pack["status"] == "failed"
    assert pack["usable_bridges"] == []
    assert pack["error_type"] == "JSONDecodeError"


def test_v6_author_proxy_promotes_grounded_review_bridge_to_usable_bridge():
    source = "This result creates a serious concern because the review process should improve."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision(
        author_proxy_request={
            "required": True,
            "reason": "anchor gap requires submitted context",
            "target_sentence_ids": ["p001_s001"],
            "finding_basis": ["author_anchor_gap"],
        }
    )
    planned = run_planner_llm(
        paragraph,
        plan,
        scan.findings,
        client=StaticJsonClient(json.dumps({"planner_decision": decision})),
    )
    strategy = dict(planned.paragraph_strategy)
    strategy["author_proxy_grounding"] = {
        "required": True,
        "bridge_anchors": [
            "The review queue had delayed handoffs before the process changed.",
            "Teams needed clearer ownership when checking submitted work.",
        ],
    }
    planned = replace(planned, paragraph_strategy=strategy)
    proxy = StaticJsonClient(json.dumps({
        "proxy_pack": {
            "usable_bridges": [],
            "author_review_items": [
                {
                    "provenance": "inferred_from_draft",
                    "target_text": "delayed handoffs and unclear ownership made review harder",
                    "user_input_needed": "Confirm, replace, or remove before final submission.",
                }
            ],
        }
    }))

    packed = attach_author_proxy_pack(paragraph, planned, scan.findings, client=proxy)
    bridges = packed.paragraph_strategy["author_proxy_pack"]["usable_bridges"]

    assert bridges
    assert bridges[0]["usable_bridge"] == "delayed handoffs and unclear ownership made review harder"
    assert bridges[0]["anchor_text"]


def test_v6_author_proxy_compacts_example_lists_before_bridge_validation():
    source = "This change made the review role more important."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision(
        author_proxy_request={
            "required": True,
            "reason": "anchor gap requires submitted context",
            "target_sentence_ids": ["p001_s001"],
            "finding_basis": ["author_anchor_gap"],
        }
    )
    planned = run_planner_llm(
        paragraph,
        plan,
        scan.findings,
        client=StaticJsonClient(json.dumps({"planner_decision": decision})),
    )
    strategy = dict(planned.paragraph_strategy)
    strategy["author_proxy_grounding"] = {
        "required": True,
        "bridge_anchors": [
            "Teams now receive requests from forms, calls, chat, and shared queues.",
            "Reviewers must judge which requests are complete and reliable.",
        ],
    }
    planned = replace(planned, paragraph_strategy=strategy)
    proxy = StaticJsonClient(json.dumps({
        "proxy_pack": {
            "usable_bridges": [],
            "author_review_items": [
                {
                    "target_text": "Because teams now receive requests from channels such as forms, calls, chat, and shared queues, reviewers must judge whether requests are reliable",
                    "user_input_needed": "Confirm, replace, or remove before final submission.",
                }
            ],
        }
    }))

    packed = attach_author_proxy_pack(paragraph, planned, scan.findings, client=proxy)
    bridge = packed.paragraph_strategy["author_proxy_pack"]["usable_bridges"][0]["usable_bridge"]

    assert "forms, calls, chat" not in bridge
    assert "submitted examples" in bridge


def test_v6_author_proxy_compacts_dash_example_lists_before_validation():
    source = "This change made the review role more important."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision(
        author_proxy_request={
            "required": True,
            "reason": "anchor gap requires submitted context",
            "target_sentence_ids": ["p001_s001"],
            "finding_basis": ["author_anchor_gap"],
        }
    )
    planned = run_planner_llm(
        paragraph,
        plan,
        scan.findings,
        client=StaticJsonClient(json.dumps({"planner_decision": decision})),
    )
    strategy = dict(planned.paragraph_strategy)
    strategy["author_proxy_grounding"] = {
        "required": True,
        "bridge_anchors": [
            "Teams now receive requests from forms, calls, chat, and shared queues.",
            "Reviewers must judge which requests are complete and reliable.",
        ],
    }
    planned = replace(planned, paragraph_strategy=strategy)
    proxy = StaticJsonClient(json.dumps({
        "proxy_pack": {
            "author_review_items": [
                {
                    "target_text": "request channels - forms, calls, chat, shared queues - mean reviewers must judge which requests are reliable",
                    "user_input_needed": "Confirm, replace, or remove before final submission.",
                }
            ],
        }
    }))

    packed = attach_author_proxy_pack(paragraph, planned, scan.findings, client=proxy)
    bridge = packed.paragraph_strategy["author_proxy_pack"]["usable_bridges"][0]["usable_bridge"]

    assert "forms, calls" not in bridge
    assert "submitted examples" in bridge


def test_v6_author_proxy_prompt_profiles_from_paragraph_and_context():
    source = "This shift has made the role of teachers even more important, not less important."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision(
        author_proxy_request={
            "required": True,
            "reason": "author_anchor_gap requires author intent from the local education context",
            "target_sentence_ids": ["p001_s001"],
            "finding_basis": ["author_anchor_gap"],
        }
    )
    planned = run_planner_llm(
        paragraph,
        plan,
        scan.findings,
        client=StaticJsonClient(json.dumps({"planner_decision": decision})),
    )
    strategy = dict(planned.paragraph_strategy)
    strategy["author_proxy_grounding"] = {
        "required": True,
        "bridge_anchors": [
            "Students learn from teachers, YouTube, TikTok, online courses, AI tools, search engines, social media, and peer communities.",
            "The real challenge is knowing which information is accurate, useful, ethical, and worth trusting.",
        ],
    }
    planned = replace(planned, paragraph_strategy=strategy)
    proxy = StaticJsonClient(json.dumps({"proxy_pack": {"usable_bridges": []}}))

    packed = attach_author_proxy_pack(paragraph, planned, scan.findings, client=proxy)

    payload = _prompt_payload(proxy.prompts[0])
    profile = payload["author_profile"]
    pack_profile = packed.paragraph_strategy["author_proxy_pack"]["author_profile"]
    context_text = " ".join(profile["local_context_terms"]).casefold()
    priorities_text = " ".join(profile["grounding_priorities"]).casefold()

    assert payload["task"] == "author_proxy_grounding_pack"
    assert payload["available_bridge_anchors"] == strategy["author_proxy_grounding"]["bridge_anchors"]
    assert "teachers" in context_text
    assert "students" in context_text
    assert "information" in context_text
    assert "actor" in " ".join(profile["expertise_scope"]).casefold()
    assert "teachers" in priorities_text
    assert pack_profile == profile
    assert proxy.kwargs[0]["app_label"] == "Author-proxy"


def test_v6_author_proxy_prompt_profiles_operational_context_without_education_role():
    source = "This result shows a serious concern because the process should improve across teams."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision(
        author_proxy_request={
            "required": True,
            "reason": "author_anchor_gap requires submitted context",
            "target_sentence_ids": ["p001_s001"],
            "finding_basis": ["author_anchor_gap"],
        },
    )
    planned = run_planner_llm(
        paragraph,
        plan,
        scan.findings,
        client=StaticJsonClient(json.dumps({"planner_decision": decision})),
    )
    strategy = dict(planned.paragraph_strategy)
    strategy["author_proxy_grounding"] = {
        "required": True,
        "bridge_anchors": ["The team had queue delays and unclear ownership before the review."],
    }
    planned = replace(planned, paragraph_strategy=strategy)
    proxy = StaticJsonClient(json.dumps({"proxy_pack": {"usable_bridges": []}}))

    attach_author_proxy_pack(paragraph, planned, scan.findings, client=proxy)

    profile = _prompt_payload(proxy.prompts[0])["author_profile"]
    context_text = " ".join(profile["local_context_terms"]).casefold()

    assert "queue" in context_text
    assert "ownership" in context_text
    assert "teams" in context_text
    assert "teachers" not in context_text


def test_v6_author_proxy_rejects_ungrounded_bridge_before_planner_handoff():
    source = "This is a serious concern because the modern world rewards people who can remember facts."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision(
        author_proxy_request={
            "required": True,
            "reason": "anchor gap",
            "target_sentence_ids": ["p001_s001"],
            "finding_basis": ["author_anchor_gap"],
        }
    )
    planned = run_planner_llm(
        paragraph,
        plan,
        scan.findings,
        client=StaticJsonClient(json.dumps({"planner_decision": decision})),
    )
    proxy = StaticJsonClient(json.dumps({
        "proxy_pack": {
            "usable_bridges": [
                {
                    "bridge_id": "b001",
                    "anchor_source": "previous_paragraph",
                    "anchor_text": "Education today should focus on how students think.",
                    "usable_bridge": "today's workplaces prize problem solving and creativity more than memorized facts",
                    "target_sentence_ids": ["p001_s001"],
                    "integration_role": "ground contrast",
                    "integration_instruction": "Use this bridge.",
                    "review_reason": "review",
                }
            ]
        }
    }))

    packed = attach_author_proxy_pack(paragraph, planned, scan.findings, client=proxy)

    assert packed.paragraph_strategy["author_proxy_pack"]["usable_bridges"] == []


def test_v6_author_proxy_rejects_generic_role_inflation_before_planner_handoff():
    source = "This shift has made the role of teachers even more important, not less important."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision(
        author_proxy_request={
            "required": True,
            "reason": "author_anchor_gap requires author intent",
            "target_sentence_ids": ["p001_s001"],
            "finding_basis": ["author_anchor_gap"],
        }
    )
    planned = run_planner_llm(
        paragraph,
        plan,
        scan.findings,
        client=StaticJsonClient(json.dumps({"planner_decision": decision})),
    )
    proxy = StaticJsonClient(json.dumps({
        "proxy_pack": {
            "usable_bridges": [
                {
                    "bridge_id": "b001",
                    "anchor_source": "previous_paragraph",
                    "anchor_text": "Students learn from teachers, YouTube, TikTok, online courses, AI tools, search engines, social media, and peer communities.",
                    "usable_bridge": "amid the flood of digital media, teachers become the critical filters and mentors",
                    "target_sentence_ids": ["p001_s001"],
                    "integration_role": "ground contrast",
                    "integration_instruction": "Use this bridge.",
                    "review_reason": "review",
                }
            ]
        }
    }))

    packed = attach_author_proxy_pack(paragraph, planned, scan.findings, client=proxy)

    assert packed.paragraph_strategy["author_proxy_pack"]["usable_bridges"] == []


def test_v6_planner_cannot_silently_drop_existing_author_proxy_pack():
    source = "This shift has made the role of teachers even more important, not less important."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    strategy = dict(plan.paragraph_strategy)
    strategy["author_proxy_pack"] = {
        "usable_bridges": [
            {
                "bridge_id": "b001",
                "anchor_source": "previous_paragraph",
                "anchor_text": "Students use many information sources and need to know what is worth trusting.",
                "usable_bridge": "students need help deciding which information is worth trusting",
                "target_sentence_ids": ["p001_s001"],
                "integration_role": "ground contrast",
                "integration_instruction": "Use as the reason teachers remain important.",
                "review_reason": "Author should confirm.",
            }
        ]
    }
    packed = replace(plan, paragraph_strategy=strategy)
    decision = _planner_decision(
        author_proxy_request={"required": False, "reason": "", "target_sentence_ids": [], "finding_basis": []},
        author_proxy_plan="no neighbor bridge needed",
    )

    replanned = run_planner_llm(
        paragraph,
        packed,
        scan.findings,
        client=StaticJsonClient(json.dumps({"planner_decision": decision})),
    )

    planner_decision = replanned.ai_safe_route["llm_planner_decision"]
    assert "students need help deciding which information is worth trusting" in planner_decision["coverage_terms"]
    assert "author_proxy_plan" not in planner_decision


def test_v6_writer_retry_feedback_carries_exact_missing_required_terms():
    source = "The modern world rewards people who can analyse, adapt, communicate, and create."
    paragraph, plan = build_plan(scan_text(source))
    diagnostics = [
        {
            "variant_id": "v1",
            "source": "llm",
            "blockers": ["required_source_terms_missing"],
            "quality_warnings": ["required_source_terms_missing_review_required"],
            "missing_required_terms": ["communicate", "create"],
            "candidate_findings": 3,
            "source_findings": 5,
            "candidate_mean_risk": 21.0,
            "source_mean_risk": 34.0,
            "candidate_finding_details": [],
        }
    ]

    retried = plan_with_writer_feedback(plan, diagnostics)
    feedback = retried.ai_safe_route["writer_retry_feedback"][0]

    assert feedback["missing_required_terms"] == ["communicate", "create"]
    assert "communicate, create" in feedback["required_fix"]


def test_v6_writer_retry_feedback_prioritizes_scope_marker_reuse():
    paragraph, plan = build_plan(scan_text("A teacher is no longer just someone who delivers information."))
    diagnostics = [
        {
            "variant_id": "v1",
            "source": "llm",
            "blockers": ["source_scope_marker_reused_as_content"],
            "quality_warnings": ["unsupported_semantic_padding_review_required"],
            "missing_required_terms": [],
            "candidate_findings": 3,
            "source_findings": 3,
            "candidate_mean_risk": 20.0,
            "source_mean_risk": 34.0,
            "candidate_finding_details": [],
            "candidate_text": "The role is longer than delivery.",
        }
    ]

    feedback = plan_with_writer_feedback(plan, diagnostics).ai_safe_route["writer_retry_feedback"][0]

    assert "guarded scope-marker" in feedback["required_fix"]
    assert "source relation" in feedback["required_fix"]


def test_v6_writer_call_keeps_writer_role_when_proxy_request_exists_without_pack():
    source = "This result shows a serious concern because the process should improve across teams."
    scan = scan_text(source)
    paragraph, plan = build_plan(scan)
    decision = _planner_decision(
        author_proxy_request={
            "required": True,
            "reason": "author_anchor_gap requires submitted context",
            "target_sentence_ids": ["p001_s001"],
            "finding_basis": ["author_anchor_gap"],
        },
    )
    planned = run_planner_llm(
        paragraph,
        plan,
        scan.findings,
        client=StaticJsonClient(json.dumps({"planner_decision": decision})),
    )
    writer = StaticJsonClient(json.dumps({"variants": [{"id": "v1", "text": source}]}))

    write_variants(paragraph, planned, client=writer)

    assert writer.kwargs[0]["app_label"] == "Writer"


def test_v6_multi_finding_paragraph_does_not_accept_risk_only_without_finding_drop():
    source = (
        "The process creates both opportunities and risks. "
        "Tools help teams plan work and review tickets. "
        "They also update notes and allow staff to practise responses. "
        "When used well, the process supports teams who need extra help. "
        "Staff may become dependent on generated answers without understanding the work behind them, and they may submit polished updates that do not reflect their real ability. "
        "This makes review difficult and raises questions about fairness, originality, and trust."
    )
    candidate = (
        "The process creates both opportunities and risks. "
        "Tools help teams plan work and review tickets. "
        "They further update notes and enable staff to practise responses. "
        "When used well, the process can support teams who need extra help. "
        "Yet a danger exists: staff may become dependent on generated answers without understanding the work behind them, and they may submit polished updates that do not reflect their real ability. "
        "Review becomes harder to gauge, which makes evaluation difficult and raises questions of fairness, originality, and trust."
    )
    paragraph = scan_text(source).paragraphs[0]
    rows = selection_diagnostics(
        [
            Variant(id="source_preserved", text=source, source="source_preserved"),
            Variant(id="v1", text=candidate, source="llm"),
        ],
        paragraph,
    )

    row = rows[0]
    assert row["source_findings"] > 1
    assert row["candidate_findings"] == row["source_findings"]
    assert row["risk_drop"] > 0
    assert row["accepted_by_selector"] is False
    assert "insufficient_scanner_movement" in row["blockers"]


def test_v6_selector_keeps_missing_terms_as_review_warning_after_material_progress():
    source = (
        "The process creates opportunities and risks. "
        "Tools help teams plan, review, improve work, practise skills, and support people who need help. "
        "This makes review difficult and raises questions about fairness, originality, and trust."
    )
    candidate = (
        "The process gives teams practical support. "
        "Tools help teams plan and review work while practising skills. "
        "Review still becomes difficult when polished work hides ability."
    )
    paragraph = scan_text(source).paragraphs[0]
    row = selection_diagnostics(
        [
            Variant(id="source_preserved", text=source, source="source_preserved"),
            Variant(id="v1", text=candidate, source="llm"),
        ],
        paragraph,
    )[0]

    assert row["candidate_findings"] < row["source_findings"]
    assert "required_source_terms_missing_review_required" in row["quality_warnings"]
    assert "required_source_terms_missing" not in row["blockers"]


def test_v6_selector_metric_contract_overrides_worse_llm_choice():
    paragraph = scan_text("A teacher is no longer just someone who delivers information.").paragraphs[0]
    variants = [
        Variant(id="v1", mode="coverage_beat_generation", text="A good teacher helps students judge information.", source="llm"),
        Variant(id="v3", mode="context_anchor_generation", text="Teachers also help students think.", source="llm"),
    ]
    diagnostics = [
        {
            "variant_id": "v1",
            "blockers": [],
            "accepted_by_selector": True,
            "candidate_findings": 2,
            "candidate_mean_risk": 11.308,
            "risk_drop": 16.119,
        },
        {
            "variant_id": "v3",
            "blockers": [],
            "accepted_by_selector": True,
            "candidate_findings": 3,
            "candidate_mean_risk": 13.546,
            "risk_drop": 13.881,
        },
    ]
    selector = StaticJsonClient(json.dumps({"selected_id": "v3", "rationale": "subjectively smoother"}))

    selected, updated = _select_variant(
        paragraph=paragraph,
        variants=variants,
        diagnostics=diagnostics,
        selector_client=selector,
    )

    assert selected.id == "v1"
    selected_row = next(row for row in updated if row["variant_id"] == "v1")
    assert selected_row["selected_by_selector"] is True
    assert selected_row["selector_source"] == "selector_llm"
    assert "selector_metric_contract" in selected_row["selector_rationale"]


def test_v6_blocks_not_less_important_reframed_as_less_overlooked():
    source = "This shift has made the role of teachers even more important, not less important."
    candidate = "The shift made the role of teachers even more important and less overlooked."
    paragraph = scan_text(source).paragraphs[0]
    rows = selection_diagnostics(
        [
            Variant(id="source_preserved", text=source, source="source_preserved"),
            Variant(id="v1", text=candidate, source="llm"),
        ],
        paragraph,
    )

    assert "source_contrast_reframed" in rows[0]["blockers"]


def test_v6_writer_retry_feedback_targets_repeated_proxy_sentence_intent():
    paragraph, plan = build_plan(scan_text("This shift has made the role of teachers even more important, not less important."))
    diagnostics = [
        {
            "variant_id": "v1",
            "source": "llm",
            "blockers": ["repeated_sentence_intent"],
            "quality_warnings": ["candidate_contract_warning"],
            "candidate_findings": 5,
            "source_findings": 3,
            "candidate_mean_risk": 17.2,
            "source_mean_risk": 27.4,
            "candidate_text": (
                "This shift is driven by digital information sources. "
                "It has made the role of teachers even more important. "
                "Education should focus on the words \"how students think.\""
            ),
            "candidate_finding_details": [
                {
                    "text": "This shift is driven by digital information sources.",
                    "tags": ["context_anchor_gap", "paragraph_rhythm"],
                }
            ],
        }
    ]

    feedback = plan_with_writer_feedback(plan, diagnostics).ai_safe_route["writer_retry_feedback"][0]

    assert "standalone proxy/context sentence" in feedback["required_fix"]
    assert "same source beat" in feedback["failed_sentences"][0]["repair_instruction"]
    assert any("the words" in item for item in feedback["do_not_repeat"])


def test_v6_writer_retry_feedback_names_malformed_verb_chain():
    source = (
        "A team may complete the task, but not always how to explain the reasoning, "
        "check the result, or connect the steps."
    )
    _paragraph, plan = build_plan(scan_text(source))
    diagnostics = [
        {
            "variant_id": "v1",
            "source": "llm",
            "blockers": ["malformed_serial_verb_chain", "malformed_negation_order"],
            "integrity_blockers": ["malformed_serial_verb_chain", "malformed_negation_order"],
            "quality_warnings": ["malformed_serial_verb_chain_review_required"],
            "missing_required_terms": [],
            "candidate_findings": 2,
            "source_findings": 3,
            "candidate_mean_risk": 20.0,
            "source_mean_risk": 34.0,
            "candidate_text": (
                "A team may complete the task, but they can can not always explain the reasoning, "
                "check the result or connect the steps."
            ),
            "candidate_finding_details": [
                {
                    "text": (
                        "A team may complete the task, but they can can not always explain the reasoning, "
                        "check the result or connect the steps."
                    ),
                    "tags": ["packed_list", "sentence_overload"],
                }
            ],
        }
    ]

    feedback = plan_with_writer_feedback(plan, diagnostics).ai_safe_route["writer_retry_feedback"][0]

    assert "same auxiliary" in feedback["required_fix"]
    assert "negation scope" in feedback["required_fix"]
    assert "verb chain first" in feedback["failed_sentences"][0]["repair_instruction"]


def test_v6_writer_retry_feedback_names_local_negation_order_defect():
    source = "A team may complete the task, but not always how to explain the result."
    _paragraph, plan = build_plan(scan_text(source))
    diagnostics = [
        {
            "variant_id": "v1",
            "source": "llm",
            "blockers": ["malformed_negation_order"],
            "integrity_blockers": ["malformed_negation_order"],
            "quality_warnings": ["malformed_negation_order_review_required"],
            "candidate_findings": 1,
            "source_findings": 2,
            "candidate_mean_risk": 20.0,
            "source_mean_risk": 34.0,
            "candidate_text": "A team may complete the task, but they are do not always able to explain the result.",
            "candidate_finding_details": [
                {
                    "text": "A team may complete the task, but they are do not always able to explain the result.",
                    "tags": ["sentence_overload"],
                }
            ],
        }
    ]

    feedback = plan_with_writer_feedback(plan, diagnostics).ai_safe_route["writer_retry_feedback"][0]

    assert "negation scope" in feedback["required_fix"]
    assert "negation scope first" in feedback["failed_sentences"][0]["repair_instruction"]


def test_v6_writer_retry_feedback_adds_residual_obligations_for_anchor_and_overload():
    source = "A process has pressure. It creates a problem. Teams may pass, but not always know how to explain, check, or improve the result."
    _paragraph, plan = build_plan(scan_text(source))
    diagnostics = [
        {
            "variant_id": "v1",
            "source": "llm",
            "accepted_by_selector": True,
            "blockers": [],
            "quality_warnings": ["compressed_list_repair_review_required"],
            "candidate_findings": 3,
            "source_findings": 5,
            "candidate_mean_risk": 22.0,
            "source_mean_risk": 35.0,
            "candidate_finding_details": [
                {
                    "text": "This pressure creates a problem.",
                    "tags": ["context_anchor_gap"],
                },
                {
                    "text": "Teams may pass, but they do not always know how to explain, check, or improve the result.",
                    "tags": ["packed_list", "sentence_overload", "broad_claim", "paragraph_rhythm"],
                },
                {
                    "text": "This is a concern because the process now asks people to explain, check, and improve.",
                    "tags": ["predictable_start", "context_anchor_gap", "author_anchor_gap", "unsupported_claim_gap"],
                },
            ],
        }
    ]

    feedback = plan_with_writer_feedback(plan, diagnostics).ai_safe_route["writer_retry_feedback"][0]

    obligations = " ".join(feedback["residual_rewrite_obligations"])
    assert "named actor/action" in obligations
    assert "overloaded list" in obligations
    assert "unsupported broad explanation" in obligations
    assert "demonstrative" in feedback["failed_sentences"][0]["repair_instruction"]


def test_v6_blocks_not_always_scope_inversion():
    source = (
        "Students may learn how to pass, but not always how to think deeply, "
        "solve problems, or connect ideas across subjects."
    )
    candidate = (
        "Students may always learn how to pass, yet they miss opportunities "
        "to think deeply, solve problems, or connect ideas across subjects."
    )
    paragraph = scan_text(source).paragraphs[0]
    rows = selection_diagnostics(
        [
            Variant(id="source_preserved", text=source, source="source_preserved"),
            Variant(id="v1", text=candidate, source="llm"),
        ],
        paragraph,
    )

    assert rows[0]["accepted_by_selector"] is False
    assert "not_always_scope_inversion" in rows[0]["blockers"]


def _planner_decision(
    *,
    author_proxy_request: dict | None = None,
    author_proxy_plan: str = "no neighbor bridge needed",
) -> dict:
    return {
        "repair_unit": "paragraph",
        "paragraph_problem": "The paragraph has a broad claim that needs a paragraph-level route.",
        "flow_plan": [
            {
                "step_id": "fp001",
                "function": "Anchor the broad claim to the local source condition.",
                "source_basis": ["p001_s001"],
            },
            {
                "step_id": "fp002",
                "function": "Close on the source consequence without adding a new topic.",
                "source_basis": ["p001_s001"],
            },
        ],
        "semantic_role_map": {"local_condition": ["result"], "consequence": ["improve", "teams"]},
        "human_route": {
            "route_type": "claim_grounding_gap",
            "movement": "local condition -> concern -> consequence",
            "paragraph_jobs": ["Ground the concern.", "Preserve the consequence."],
        },
        "paragraph_route": "local condition -> concern -> consequence",
        "author_proxy_request": author_proxy_request
        or {"required": False, "reason": "", "target_sentence_ids": [], "finding_basis": []},
        "finding_pattern": {"pattern_id": "claim_grounding_gap"},
        "author_proxy_plan": author_proxy_plan,
        "do_not_copy_route": ["generic concern tail"],
    }


def _prompt_payload(prompt: str) -> dict:
    return json.loads(prompt.split("\n", 1)[1])
