from __future__ import annotations

import json

from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6 import pipeline as v6_pipeline
from poc.rewrite_v6.paragraph_architecture import apply_architecture_split_text, architecture_split_contract
from poc.rewrite_v6.pipeline import _acceptable_progress, _cross_paragraph_regression, _dynamic_pass_limit, _report_target_paragraph_ids, _same_text, run_v6_rewrite_all
from poc.rewrite_v6.planner_llm import build_planner_prompt
from poc.rewrite_v6.prose_quality import drop_redundant_adjacent_sentence_intent, has_fragment_or_trace_sentences, repair_generated_prose
from poc.rewrite_v6.integrity_guard import candidate_integrity_blockers
from poc.rewrite_v6.repair_windows import RepairWindow, compose_window_rewrite, select_repair_window
from poc.rewrite_v6.report_contracts import apply_report_signal_contracts, extract_report_signal_contracts
from poc.rewrite_v6.scan import findings_for_paragraph, scan_text
from poc.rewrite_v6.selector_diagnostics import selection_diagnostics
from poc.rewrite_v6.write import Variant, build_prompt, choose_variant


class EmptyVariantClient:
    def __init__(self):
        self.prompts: list[str] = []

    def chat(self, prompt: str, **_kwargs):
        self.prompts.append(prompt)
        return type("Response", (), {"content": '{"variants":[]}', "raw_content": '{"variants":[]}'})()


class SequencedVariantClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def chat(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(dict(kwargs))
        content = self.responses.pop(0)
        return type("Response", (), {"content": content, "raw_content": content})()


def test_v6_extracts_full_report_signal_contracts_without_blocking_author_proxy():
    report = {
        "scan_intelligence": {
            "transformation": {
                "core_signals": [
                    {"key": "citation_grounding_risk", "score": 80},
                    {"key": "topk_pattern_raw", "score": 71},
                    {"key": "human_anchor_score", "score": 48},
                    {"key": "rewrite_smoothness", "score": 34},
                    {"key": "section_style_variance", "score": 45},
                ]
            }
        },
        "ai_mitigation": {
            "component_actions": [
                {"component": "unsupported_claim_risk", "current_score": 90},
                {"component": "generic_assertion_risk", "current_score": 65},
            ]
        },
    }

    contracts = extract_report_signal_contracts(report)
    groups = {row["signal_group"] for row in contracts}

    assert {
        "grounding_route",
        "predictability_route",
        "human_anchor_route",
        "thinking_path_route",
        "source_coverage_route",
        "claim_scope_route",
        "context_specificity_route",
    } <= groups
    assert all("do not present unsupported external facts as verified" in row["author_proxy_policy"] for row in contracts)


def test_v6_extracts_generic_contracts_from_report_findings_with_fractional_scores():
    report = {
        "findings": {
            "medium": [
                {"title": "medium_predictability", "category": "predictability", "score": 0.51},
                {"title": "low_specificity", "signal_category": "genericity", "score": 0.44},
                {"title": "uniform_paragraph_structure", "category": "ai_generation", "score": 0.48},
                {"title": "semantic_drift", "category": "semantic_shape", "score": 0.62},
            ]
        }
    }

    groups = {row["signal_group"] for row in extract_report_signal_contracts(report)}

    assert {
        "predictability_route",
        "context_specificity_route",
        "thinking_path_route",
        "source_coverage_route",
    } <= groups


def test_v6_report_signal_contracts_reach_planner_and_writer_prompts():
    scan = scan_text("This process uses a form, a queue, and a review because the system should improve.")
    paragraph, plan = build_plan(scan)
    plan = apply_report_signal_contracts(
        plan,
        [{
            "signal_group": "grounding_route",
            "score": 80,
            "writer_obligation": "keep claims near submitted support",
            "target_excerpts": ["form queue review"],
        }],
    )

    planner_payload = json.loads(build_planner_prompt(paragraph, plan, findings_for_paragraph(scan, paragraph.id)).split("\n", 1)[1])
    writer_payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])

    planner_contracts = planner_payload["deterministic_route_skeleton"]["document_signal_contracts"]
    direct_writer_contracts = writer_payload["document_signal_contracts"]
    assert planner_contracts[0]["signal_group"] == "grounding_route"
    assert planner_contracts[0]["target_excerpt_count"] == 1
    assert planner_contracts[0]["target_excerpt_samples"] == ["form queue review"]
    assert direct_writer_contracts[0]["signal_group"] == "grounding_route"
    assert direct_writer_contracts[0]["writer_obligation"] == "keep claims near submitted support"
    assert direct_writer_contracts[0]["target_excerpt_count"] == 1
    assert direct_writer_contracts[0]["target_excerpt_samples"] == ["form queue review"]
    assert "document_signal_contracts" not in writer_payload.get("planner_decision", {})
    assert "document_signal_contracts" not in writer_payload["writer_execution_contract"]
    assert "document_signal_contracts as the primary build contract" in " ".join(writer_payload["generation_rules"])


def test_v6_dense_findings_become_paragraph_repair_unit_without_beat_explosion():
    text = (
        "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve. "
        "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve. "
        "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve."
    )

    paragraph, plan = build_plan(scan_text(text))

    assert plan.paragraph_strategy["repair_unit"] == "paragraph"
    assert plan.paragraph_strategy["dense_paragraph_plan"]["finding_interpretation_rule"].startswith("Scanner findings")
    assert len(plan.ai_safe_route["coverage_beats"]) == len(paragraph.sentences)
    assert all("one final sentence per finding" not in beat["merge_rule"] for beat in plan.ai_safe_route["coverage_beats"])


def test_v6_dense_paragraph_plan_reaches_planner_and_writer_payloads():
    text = (
        "This process is important because teams should improve. "
        "This method is important because students should improve. "
        "This issue is important because schools should improve. "
        "This problem is important because teachers should improve."
    )
    scan = scan_text(text)
    paragraph, plan = build_plan(scan)

    planner_payload = json.loads(build_planner_prompt(paragraph, plan, findings_for_paragraph(scan, paragraph.id)).split("\n", 1)[1])
    writer_payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])

    assert planner_payload["deterministic_route_skeleton"]["paragraph_strategy"]["repair_unit"] == "paragraph"
    assert "one paragraph flow" in " ".join(planner_payload["rules"])
    assert writer_payload["paragraph_repair_plan"]["repair_unit"] == "paragraph"
    assert writer_payload["writer_execution_contract"]["paragraph_repair_unit"] == "paragraph"
    assert writer_payload["paragraph_repair_plan"]["dense_paragraph_plan"]["semantic_role_map"]
    assert writer_payload["paragraph_repair_plan"]["dense_paragraph_plan"]["human_route"]
    assert "natural compact list is allowed" in writer_payload["paragraph_repair_plan"]["semantic_role_rule"]
    assert "lived reasoning" in writer_payload["paragraph_repair_plan"]["human_route_rule"]
    assert "do not write one final sentence per finding" in writer_payload["paragraph_repair_plan"]["rule"]


def test_v6_report_target_excerpts_map_to_local_paragraph_ids_without_raw_id_coupling():
    text = (
        "First paragraph has ordinary setup.\n\n"
        "The salon paragraph explains haircutting structure, projection angle, working memory, and learner confidence."
    )
    report = {
        "scan_intelligence": {"transformation": {"core_signals": [{"key": "topk_pattern_raw", "score": 70}]}},
        "rewrite_decision": {"targets": ["f022"]},
        "findings": {
            "medium": [{
                "finding_id": "f022",
                "rewrite_context": {
                    "paragraph_id": "p999",
                    "paragraph_excerpt": "haircutting structure place strain on working memory and learner confidence",
                },
            }]
        },
    }

    contracts = extract_report_signal_contracts(report)
    scan = scan_text(text)
    paragraph, _ = build_plan(scan, priority_paragraph_ids=_report_target_paragraph_ids(scan, contracts))

    assert _report_target_paragraph_ids(scan, contracts) == {"p002"}
    assert paragraph.id == "p002"


def test_v6_report_target_acceptance_does_not_block_non_worse_anchor_repairs():
    scan = scan_text("The method uses a form and a review.")

    assert _acceptable_progress(scan, scan, report_targeted=True)
    assert not _acceptable_progress(scan, scan, report_targeted=False)


def test_v6_acceptance_rejects_non_target_paragraph_regression():
    before = scan_text("This is an important process because the team should improve.\n\nPlain stable paragraph.")
    after = scan_text("This process improved through a team review.\n\nThis is an important issue because the stable paragraph should improve.")

    assert _cross_paragraph_regression(before, after, "p001")


def test_v6_cross_paragraph_regression_allows_target_paragraph_split():
    before = scan_text("This process has an important issue because teams should improve.\n\nStable paragraph has a plain sentence.")
    after = scan_text("The process has an issue.\n\nTeams should improve through review.\n\nStable paragraph has a plain sentence.")

    assert not _cross_paragraph_regression(before, after, "p001")


def test_v6_cross_paragraph_regression_allows_small_recalibration_when_target_split_improves():
    before = scan_text(
        "Plain paragraph has one stable finding because teams should improve.\n\n"
        "The target paragraph has a list, a second item, and a third item because it should improve. "
        "This target paragraph has another issue because it should improve.\n\n"
        "Another plain paragraph has a stable issue because teams should improve."
    )
    after = scan_text(
        "Plain paragraph has one stable finding because teams should improve. This nearby sentence should improve.\n\n"
        "The target paragraph has a list.\n\n"
        "A second target item remains visible.\n\n"
        "Another plain paragraph has a stable issue because teams should improve."
    )

    assert int(before.scores["finding_count"]) > int(after.scores["finding_count"])
    assert not _cross_paragraph_regression(before, after, "p002")


def test_v6_dynamic_pass_limit_scales_by_finding_paragraphs_not_total_paragraphs():
    text = "\n\n".join([
        "This is an important process because teams should improve.",
        "This is an important method because students should improve.",
        "Plain paragraph without a scanner pattern.",
        "This is an important issue because schools should improve.",
        "Plain context for the document.",
        "This is an important problem because teachers should improve.",
        "Another ordinary paragraph for context.",
        "This is an important result because learners should improve.",
    ])
    scan = scan_text(text)

    assert len(scan.paragraphs) == 8
    assert len({finding.paragraph_id for finding in scan.findings}) == 5
    assert _dynamic_pass_limit(scan) == 10


def test_v6_pipeline_schedules_report_target_even_without_local_findings():
    text = (
        "First paragraph has ordinary setup.\n\n"
        "The salon paragraph explains haircutting structure, projection angle, working memory, and learner confidence."
    )
    contracts = [{
        "signal_group": "predictability_route",
        "score": 70,
        "writer_obligation": "change report-target route",
        "target_excerpts": ["haircutting structure place strain on working memory and learner confidence"],
    }]
    writer = EmptyVariantClient()

    result = run_v6_rewrite_all(text, writer_client=writer, max_passes=2, report_signal_contracts=contracts)

    assert len(writer.prompts) == 1
    assert "haircutting structure" in writer.prompts[0]
    assert "ordinary setup" not in writer.prompts[0]
    assert result.pass_trace[0]["status"] == "no_change"
    assert result.pass_trace[0]["target_paragraph_id"] == "p002"
    assert not result.passes


def test_v6_selects_sentence_window_for_overloaded_target_paragraph():
    text = (
        "Opening setup stays outside the repair. "
        "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve. "
        "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve. "
        "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve. "
        "A later sentence stays outside the first repair. "
        "Another later sentence stays outside the first repair. "
        "A final sentence stays outside the first repair."
    )
    scan = scan_text(text)
    paragraph = scan.paragraphs[0]

    window = select_repair_window(paragraph, findings_for_paragraph(scan, paragraph.id), max_sentences=3, max_words=75)

    assert window is not None
    assert 1 <= window.start_sentence_index <= window.end_sentence_index < len(paragraph.sentences) - 1
    assert len(window.source_sentence_ids) <= 3
    assert "Opening setup stays outside" not in window.source_text
    assert "A final sentence stays outside" not in window.source_text


def test_v6_composes_window_rewrite_without_replacing_whole_paragraph():
    scan = scan_text(
        "Opening setup stays outside. "
        "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve. "
        "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve. "
        "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve. "
        "A final sentence stays outside."
    )
    paragraph = scan.paragraphs[0]
    window = RepairWindow(
        paragraph_id=paragraph.id,
        start_sentence_index=1,
        end_sentence_index=3,
        source_text=" ".join(sentence.text for sentence in paragraph.sentences[1:4]),
        source_sentence_ids=[sentence.id for sentence in paragraph.sentences[1:4]],
        finding_count=3,
        max_severity=50,
    )

    rewritten = compose_window_rewrite(
        scan.paragraphs,
        window,
        Variant(id="v1", text="The process moves through forms and reviews before approval checks.", source="llm"),
    )

    assert rewritten.startswith("Opening setup stays outside.")
    assert "The process moves through forms and reviews before approval checks." in rewritten
    assert rewritten.endswith("A final sentence stays outside.")
    assert "queues, labels, reviews, approvals, and checks" in scan.source_text


def test_v6_pipeline_sends_semantic_paragraph_to_writer_before_window_fallback():
    text = (
        "Opening setup stays outside the repair. "
        "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve. "
        "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve. "
        "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve. "
        "A later sentence stays outside the first repair. "
        "Another later sentence stays outside the first repair. "
        "A final sentence stays outside the first repair."
    )
    client = EmptyVariantClient()

    result = v6_pipeline.run_v6_rewrite(text, writer_client=client, priority_paragraph_ids={"p001"})

    assert result.plan.paragraph_id == "p001"
    assert "repair_window" not in result.plan.ai_safe_route
    assert client.prompts
    prompt_payload = json.loads(client.prompts[0].split("\n", 1)[1])
    prompt_text = json.dumps(prompt_payload)
    assert "forms" in prompt_text
    assert "queues" in prompt_text
    assert "labels" in prompt_text
    assert "Opening setup stays outside the repair" in prompt_text
    assert "A final sentence stays outside the first repair" in prompt_text


def test_v6_trace_records_review_warnings_for_non_winning_candidate():
    text = "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve."
    response = json.dumps({
        "variants": [{
            "id": "v1",
            "text": text + " Today.",
            "coverage_map": [],
            "route_answer_cards": [],
            "author_proxy_provenance": [],
            "author_review_items": [],
        }]
    })
    client = SequencedVariantClient([response, '{"variants":[]}'])
    selector = SequencedVariantClient(['{"selected_id":"v1","rationale":"test selection"}'])

    result = run_v6_rewrite_all(text, writer_client=client, selector_client=selector, max_passes=1)

    row = result.pass_trace[0]
    assert row["status"] == "not_improved"
    assert row["selected_source"] == "llm"
    assert row["candidate_diagnostics"][0]["variant_id"] == "v1"
    assert row["candidate_diagnostics"][0]["candidate_findings"] >= row["candidate_diagnostics"][0]["source_findings"]


def test_v6_document_rewrite_runs_residual_followup_after_accepted_paragraph(monkeypatch):
    source = "This process uses a form, a queue, and a review. This result shows a problem because the system should improve."
    seen = []

    def fake_run(current, **kwargs):
        scan = scan_text(current)
        paragraph, plan = build_plan(scan, kwargs.get("excluded_paragraph_ids"), kwargs.get("priority_paragraph_ids"))
        seen.append(paragraph.id)
        replacement = "The process uses a form queue. This result shows a problem because the system should improve."
        if len(seen) > 1:
            replacement = "The process uses a form queue."
        return v6_pipeline.Result(scan=scan, plan=plan, variants=[], selected=None, rewritten_text=replacement)

    monkeypatch.setattr(v6_pipeline, "run_v6_rewrite", fake_run)
    result = v6_pipeline.run_v6_rewrite_all(source, max_passes=1, residual_followup_passes=1)

    assert seen == ["p001", "p001"]
    assert [row["status"] for row in result.pass_trace] == ["accepted", "accepted_residual"]
    assert result.final_scan.scores["finding_count"] == 0


def test_v6_residual_followup_continues_across_split_child_targets(monkeypatch):
    source = "This method uses forms, queues, labels, reviews, approvals, and checks because students should improve."
    split_candidate = (
        "This method uses forms, queues, labels, reviews, approvals, and checks.\n\n"
        "This result shows a problem because students should improve."
    )
    final_replacement = "Students improve after checking the result."
    final_candidate = (
        "This method uses forms, queues, labels, reviews, approvals, and checks.\n\n"
        f"{final_replacement}"
    )
    seen: list[tuple[str, set[str] | None]] = []

    def fake_run(current, **kwargs):
        scan = scan_text(current)
        paragraph, plan = build_plan(scan, kwargs.get("excluded_paragraph_ids"), kwargs.get("priority_paragraph_ids"))
        seen.append((paragraph.id, kwargs.get("priority_paragraph_ids")))
        if len(seen) == 1:
            return v6_pipeline.Result(
                scan=scan,
                plan=plan,
                variants=[],
                selected=Variant(id="v1", text=split_candidate, source="llm"),
                rewritten_text=split_candidate,
            )
        if paragraph.id == "p001":
            return v6_pipeline.Result(scan=scan, plan=plan, variants=[], selected=None, rewritten_text=current)
        return v6_pipeline.Result(
            scan=scan,
            plan=plan,
            variants=[],
            selected=Variant(id="v2", text=final_replacement, source="llm"),
            rewritten_text=final_candidate,
        )

    monkeypatch.setattr(v6_pipeline, "run_v6_rewrite", fake_run)
    result = v6_pipeline.run_v6_rewrite_all(source, max_passes=1, residual_followup_passes=1)

    assert [row["status"] for row in result.pass_trace] == ["accepted", "no_change_residual", "accepted_residual"]
    assert result.pass_trace[0]["target_paragraph_id"] == "p001,p002"
    assert result.pass_trace[1]["target_paragraph_id"] == "p001,p002"
    assert result.pass_trace[2]["target_paragraph_id"] == "p002"
    assert [paragraph_id for paragraph_id, _ in seen] == ["p001", "p001", "p002"]


def test_v6_no_change_detection_ignores_paragraph_spacing_normalization():
    assert _same_text("First paragraph.\n\nSecond paragraph.", "First paragraph. Second paragraph.")


def test_v6_writer_prompt_surfaces_required_sentence_groups_as_coverage_loss_contract():
    text = (
        "Learner performance is assessed against industry benchmarks, in my delivery of SHBHCUT006 I adapted the scaffolding approach with the Octagon method, and salon safety and commercial time constraints remained consistent for learners."
    )
    scan = scan_text(text)
    paragraph, plan = build_plan(scan)
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    groups = payload["coverage_loss_contract"]

    assert groups
    assert any("salon" in " ".join(row["source_terms_to_carry"]).casefold() for row in groups)
    assert "cover every group in the final text" in " ".join(payload["generation_rules"])
    assert "one final sentence per group" not in build_prompt(paragraph, plan)
    assert "same row sequence joined" not in build_prompt(paragraph, plan)
    assert "coverage_map to prove group coverage" not in build_prompt(paragraph, plan)
    assert "do not output route_fragments, route_answer_cards, coverage_map, sentence_rows" in build_prompt(paragraph, plan)
    assert "Do not write repair-trace labels" in " ".join(payload["generation_rules"])


def test_v6_writer_prompt_allows_architecture_split_for_overloaded_paragraphs():
    text = (
        "The standard requires support and adjustment while the assessment keeps industry requirements. "
        "The learner performance is assessed against benchmarks, standards, safety, timing, and commercial expectations. "
        "I adapted scaffolding through method changes, role practice, client work, and reflection. "
        "This creates a concern because support can become over-accommodation when expectations fall. "
        "The workplace would not accept incomplete service, unsafe timing, or weak results. "
        "The conclusion is that pathways can change while standards remain."
    )
    paragraph, plan = build_plan(scan_text(text))
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])

    assert payload["architecture_split_contract"]["active"]
    assert payload["required_shape"]["paragraph_count"] == "as many functional paragraphs as needed"
    assert payload["architecture_split_contract"]["functional_groups"]


def test_v6_architecture_splitter_breaks_overloaded_single_block_by_function():
    text = (
        "The standard requires support and adjustment while the assessment keeps industry requirements. "
        "The learner performance is assessed against benchmarks and commercial expectations. "
        "I adapted scaffolding through method changes and role practice. "
        "The task still needed client work and reflection. "
        "This creates a concern because support can become over-accommodation when expectations fall. "
        "The workplace would not accept incomplete service or unsafe timing. "
        "The conclusion is that pathways can change while standards remain."
    )
    split = apply_architecture_split_text(text, {"active": True})

    assert "\n\n" in split
    assert len(split.split("\n\n")) >= 2


def test_v6_architecture_splitter_preserves_inactive_text():
    text = "The task changed. The standard remained."
    assert apply_architecture_split_text(text, {"active": False}) == text


def test_v6_architecture_splitter_backfills_missing_source_anchors():
    text = (
        "Learner performance is assessed against industry benchmarks. "
        "I adapted the scaffolding approach. "
        "The practical assessment criteria remained consistent."
    )
    contract = {
        "active": True,
        "functional_groups": [{
            "source_texts": [
                "Learner performance is assessed against industry benchmarks, although I adapted the scaffolding approach, the practical assessment criteria, salon safety and commercial time constraints have still remained entirely consistent for all learners."
            ],
            "must_survive_terms": ["salon", "safety", "commercial", "time", "constraints", "learners"],
        }],
    }
    split = apply_architecture_split_text(text, contract)

    assert "Salon safety and commercial time constraints remained entirely consistent for all learners." in split


def test_v6_architecture_splitter_decompresses_comma_heavy_candidate_sentences():
    text = (
        "The first sentence stays ordinary. "
        "The model is flawed, educators lower expectations, the system compromises standards, the result creates an unreal environment. "
        "The final sentence stays ordinary."
    )
    split = apply_architecture_split_text(text, {"active": True})

    assert "Educators lower expectations." in split
    assert "the system compromises standards," not in split


def test_v6_writer_cleanup_drops_redundant_adjacent_sentence_intent():
    text = (
        "Developing awareness of how they learn matters more than endless help and sympathy for diverse learners prone to frustration. "
        "Endless help and sympathy for diverse learners prone to frustration is not enough by itself. "
        "Teaching learners to learn and adapt is more effective than simply helping them."
    )

    cleaned = drop_redundant_adjacent_sentence_intent(text)

    assert cleaned.count("endless help and sympathy") == 1
    assert "Teaching learners to learn and adapt" in cleaned


def test_v6_generated_prose_repair_preserves_not_only_and_repairs_gerund_evidence_fragment():
    source = (
        "Johnny demonstrated skill in the meeting. "
        "The thank cards do not only serve as endorsement of Johnny skill but also attest to professional integrity."
    )
    candidate = (
        "Johnny demonstrated skill in the meeting. "
        "Demonstrating the ability to support clients. "
        "The thank cards serve as endorsement of Johnny skill and attest to professional integrity."
    )

    repaired = repair_generated_prose(candidate, source)

    assert "Johnny demonstrated the ability to support clients." in repaired
    assert "do not only serve as endorsement of Johnny skill." in repaired
    assert "The same cards also attest to professional integrity." in repaired


def test_v6_generated_prose_repair_splits_client_needs_and_not_only_density():
    candidate = (
        "He demonstrated the ability to accurately identify and gently address the needs of clients with Down syndrome and autism. "
        "The thank cards do not only serve as an endorsement of Johnny skill but also attest to professional integrity."
    )

    repaired = repair_generated_prose(candidate, candidate)

    assert "He identified the needs of clients with Down syndrome and autism." in repaired
    assert "He gently addressed those needs." in repaired
    assert "The thank cards do not only serve as an endorsement of Johnny skill." in repaired
    assert "The same cards also attest to professional integrity." in repaired


def test_v6_generated_prose_repair_revoices_displayed_quality_evidence():
    candidate = "During voluntary haircutting work with socially vulnerable groups he displayed a high degree of empathy and professional patience."

    repaired = repair_generated_prose(candidate, candidate)

    assert repaired == "Voluntary haircutting work with socially vulnerable groups showed his empathy and professional patience."


def test_v6_writer_schema_keeps_trace_objects_out_of_model_output():
    paragraph, plan = build_plan(scan_text("The process uses a form, a queue, and a review."))
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    variant_schema = payload["output_schema"]["variants"][0]

    assert "text" in variant_schema
    assert "coverage_map" not in variant_schema
    assert "sentence_rows" not in variant_schema


def test_v6_selected_author_proxy_candidate_labels_short_new_bridge_terms_for_review():
    paragraph = scan_text("Department guidance says adjustment must not undermine performance standards.").paragraphs[0]
    variant = Variant(id="v1", source="llm", text="Department guidance oversees adjustment and performance standards.")

    selected = choose_variant([variant], paragraph)

    assert selected and selected.author_review_items
    assert "oversees" in selected.author_review_items[0]["target_text"]


def test_v6_selector_rejects_fragment_and_repair_trace_candidate_even_when_different():
    paragraph = scan_text(
        "The method keeps assessment requirements visible. The teacher adapted the task while safety standards remained in place."
    ).paragraphs[0]
    candidate = Variant(
        id="v1",
        source="llm",
        text=(
            "The method keeps assessment requirements visible. "
            "The includes the requirements. "
            "A real setting is the context. "
            "Safety standards remained in place."
        ),
    )

    selected = choose_variant([Variant(id="source_preserved", source="source_preserved", text=paragraph.text), candidate], paragraph)

    assert selected and selected.source == "source_preserved"


def test_v6_fragment_detector_rejects_subordinate_and_object_split_fragments():
    text = (
        "I actively adapted. "
        "The scaffolding approach replaced dense technical jargon. "
        "When the education system compromises core standards to spare frustration."
    )

    assert has_fragment_or_trace_sentences(text)


def test_v6_fragment_detector_rejects_leading_connector_and_relative_fragments():
    text = (
        "Class videos help students relate what they are watching to cutting. "
        "And need repeated practice to master them. "
        "Which are new for many learners because it relate to geometric concepts."
    )

    assert has_fragment_or_trace_sentences(text)


def test_v6_fragment_detector_rejects_single_hard_subordinate_fragment():
    text = (
        "The videos help students relate what they are watching to cutting. "
        "As the seven procedures require multitasking during a haircut."
    )

    assert has_fragment_or_trace_sentences(text)


def test_v6_fragment_detector_allows_subordinate_opener_with_main_clause():
    text = (
        "At the beginning he was quite reserved. "
        "As we got to know each other through casual conversation I learned about some of his past learning experiences. "
        "With guidance and role-playing activities he gradually became more confident."
    )

    assert not has_fragment_or_trace_sentences(text)


def test_v6_generated_prose_repair_merges_example_when_fragment():
    candidate = (
        "The approach uses visual references. "
        "For example, when I guide them to the 12-o’clock projection. "
        "They can understand where to project the hair."
    )

    repaired = repair_generated_prose(candidate, candidate)

    assert repaired == (
        "The approach uses visual references. "
        "When I guide them to the 12-o’clock projection, they can understand where to project the hair."
    )
    assert not has_fragment_or_trace_sentences(repaired)


def test_v6_generated_prose_repair_revoices_broad_belief_to_teaching_method_anchor():
    candidate = "I believe educators should use clearer and more accessible approaches to support student understanding, in practice."
    source = "Practical learning is not only demonstration, and this aligns with my teaching method."

    repaired = repair_generated_prose(candidate, source)

    assert repaired == "My teaching method uses clearer and more accessible approaches to support student understanding."
    assert not findings_for_paragraph(scan_text(repaired), "p001")


def test_v6_generated_prose_repair_revoices_guidance_understanding_sentence():
    candidate = "When I guide them to a 12-o’clock projection, they readily understand where to project the hair directly."

    repaired = repair_generated_prose(candidate, candidate)

    assert repaired == "The 12-o’clock projection shows where to project the hair."


def test_v6_generated_prose_repair_revoices_no_comma_guidance_example_sentence():
    candidate = "For example, when I guide them to the 12-o’clock projection they can easily understand where to project the hair straightaway as an example."

    repaired = repair_generated_prose(candidate, candidate)

    assert repaired == "The 12-o’clock projection shows where to project the hair."


def test_v6_generated_prose_repair_revoices_lowercase_guidance_seeing_sentence():
    candidate = "when I guide them to a 12-o’clock projection, they can easily see where to project the hair straightaway."

    repaired = repair_generated_prose(candidate, candidate)

    assert repaired == "The 12-o’clock projection shows where to project the hair."
    assert not findings_for_paragraph(scan_text(repaired), "p001")


def test_v6_generated_prose_repair_tolerates_example_marker_punctuation():
    expected = "The 12-o’clock projection shows where to project the hair."
    candidates = [
        "For example when I guide them to the 12-o’clock projection they can easily understand where to project the hair.",
        "For example: when I guide them to the 12-o’clock projection they can easily understand where to project the hair.",
        "For example - when I guide them to the 12-o’clock projection they can easily understand where to project the hair.",
    ]

    assert [repair_generated_prose(candidate, candidate) for candidate in candidates] == [expected, expected, expected]


def test_v6_generated_prose_repair_revoices_example_shows_guidance_sentence():
    candidate = "An example shows that when I guide them to the 12-o’clock projection they can easily understand the task."

    repaired = repair_generated_prose(candidate, candidate)

    assert repaired == "The 12-o’clock projection shows the task."


def test_v6_generated_prose_repair_revoices_pronoun_simplification():
    candidate = (
        "The approach aligns with CAST’s principle. "
        "It applies a simplification of abstract technical concepts, turning them into familiar visual references."
    )

    repaired = repair_generated_prose(candidate, candidate)

    assert repaired == (
        "The approach aligns with CAST’s principle. "
        "The approach simplifies abstract technical concepts into familiar visual references."
    )


def test_v6_writer_prompt_blocks_standalone_connector_fragments():
    paragraph, plan = build_plan(scan_text("The process uses forms, queues, and reviews because teams should improve."))
    prompt = build_prompt(paragraph, plan)

    assert "No final sentence may start with And, But, Or, Which, Where, That, or As" in prompt
    assert "Every final sentence must stand alone" in prompt


def test_v6_writer_accepts_risk_mitigating_candidate_with_review_warning():
    source = (
        "However, the current education system still carries many old habits. "
        "Many schools continue to place heavy pressure on grades, exams, and standard answers. "
        "This can encourage memorisation rather than understanding. "
        "Students may learn how to pass, but not always how to think deeply, solve problems, or connect ideas across subjects. "
        "This is a serious concern because the modern world does not only reward people who can remember facts. "
        "It rewards people who can analyse, adapt, communicate, and create."
    )
    client = SequencedVariantClient([
        json.dumps({"variants": [{
                "id": "v1",
                "text": (
                    "Many schools still place heavy pressure on grades, exams, and standard answers. "
                    "This pressure can encourage memorisation rather than understanding. "
                    "Students may learn how to pass, but they do not always learn to think deeply, solve problems, or connect ideas across subjects. "
                    "This remains a serious concern because the modern world does not only reward people who remember facts; it rewards people who analyse, adapt, communicate, and create."
                ),
            }]}),
    ])
    selector = SequencedVariantClient(['{"selected_id":"v1","rationale":"test selection"}'])

    result = v6_pipeline.run_v6_rewrite(source, writer_client=client, selector_client=selector, priority_paragraph_ids={"p001"})

    assert len(client.prompts) == 1
    assert result.selected and result.selected.id == "v1"
    assert result.candidate_diagnostics[0]["quality_warnings"]
    assert any(
        "source_polarity_changed" in warning
        for warning in result.candidate_diagnostics[0]["quality_warnings"]
    )
    assert result.rewritten_text.startswith(
        "Many schools still place heavy pressure on grades, exams, and standard answers. "
        "This pressure can encourage memorisation rather than understanding. "
        "Students may learn how to pass."
    )


def test_v6_writer_accepts_risk_mitigating_first_pass_with_review_items():
    source = (
        "This also aligns with CAST’s (2024) principle of multiple means of representation by simplifying abstract technical concepts into more familiar visual references. "
        "For example, when I guide them to 12 o’clock projection, they can easily understand where to project the hair straightaway. "
        "Compared to the existing resource, it is usually called perpendicular distribution and projected to 90 degrees from baseline (head shape). "
        "This makes technical terminology easier for diverse learners to understand. "
        "I believe educators should use clearer and more accessible approaches to support student understanding. "
        "Practical learning is not only demonstration, but it also needs listening, understanding, proper guidance and scaffolding (Billett, 2013), and this aligns with my teaching method."
    )
    client = SequencedVariantClient([
        json.dumps({"variants": [{
            "id": "v1",
            "text": "This also aligns with CAST’s (2024) principle. And needs improvement.",
        }]}),
        json.dumps({"variants": [{
            "id": "retry_v1",
            "text": (
                "CAST’s (2024) principle supports multiple means of representation. "
                "The approach simplifies abstract technical concepts into familiar visual references. "
                "For example, when I guide them to the 12-o’clock projection they can easily understand where to project the hair straightaway as an example. "
                "Compared to the existing resource, the method is usually called perpendicular distribution and projected to 90 degrees from baseline head shape. "
                "The approach makes technical terminology easier for diverse learners. "
                "My teaching method uses clearer and more accessible approaches to support student understanding. "
                "Practical learning is not only demonstration. Listening and understanding are also needed. Proper guidance and scaffolding are required (Billett, 2013)."
            ),
        }]}),
    ])
    selector = SequencedVariantClient(['{"selected_id":"v1","rationale":"test selection"}'])

    result = v6_pipeline.run_v6_rewrite(source, writer_client=client, selector_client=selector, priority_paragraph_ids={"p001"})

    assert len(client.prompts) == 1
    assert result.selected and result.selected.id == "v1"
    assert result.selected.author_review_items
    assert "and needs improvement." in result.selected.text
    assert any(
        "source_terms_missing" in item.get("target_text", "")
        for item in result.selected.author_review_items
    )


def test_v6_integrity_guard_rejects_broken_citation_and_grammar_shapes():
    text = (
        "The method uses source support (Smith. They also carries 2024). "
        "The involves explaining the process. "
        "Learners learned accepting feedback. "
        "The created an inclusive learning environment (CAST 2024; Jwad 2022). "
        "Proper guidance and scaffolding from Billett (2013) aligns with this teaching method."
    )

    blockers = candidate_integrity_blockers(text)

    assert "broken_citation_shape" in blockers
    assert "citation_name_wrapper" in blockers
    assert "malformed_subject_verb_agreement" in blockers
    assert "dangling_article_predicate" in blockers
    assert "malformed_verb_complement" in blockers


def test_v6_integrity_guard_rejects_remaining_production_bad_shapes():
    text = (
        "Without Inclusive Learning Design and UDL support many learners become difficult. "
        "Learners are unwilling to learn because the task requires technical understanding and repeated practice. "
        "Help them build self-confidence and develop critical thinking. "
        "Promote inclusive learning by incorporating role-playing activities. "
        "Such actions created an inclusive learning environment thereby (CAST, 2024). "
        "The same outcome was observed in later studies (Jwad et al., 2022). "
        "Further evidence supports this finding (Lawrie et al., 2017)."
    )

    blockers = candidate_integrity_blockers(text)

    assert "unsupported_learner_blame_shape" in blockers
    assert "bare_instruction_fragment" in blockers
    assert "stranded_thereby_citation" in blockers
    assert "citation_report_sentence" in blockers


def test_v6_integrity_guard_rejects_planner_language_leakage():
    text = (
        "Coverage beat uses forms and queues. "
        "Guide relationship sees labels and reviews. "
        "The writer_execution_contract keeps the source slot visible. "
        "Guide prompts students to compare sources. "
        "Guide function directs students toward sources. "
        "Apply knowledge occurs in real situations. "
        "In the second source sentence, schools continue their work. "
        "The source groups schools with their continuation. "
        "Teacher is no longer just someone who delivers information."
    )

    blockers = candidate_integrity_blockers(text)

    assert "planner_language_leakage" in blockers


def test_v6_integrity_guard_rejects_malformed_serial_verb_chain():
    text = "Students learn to pass but not always to think deeply solve problems. People analyse adapt communicate and create."

    blockers = candidate_integrity_blockers(text)

    assert "malformed_serial_verb_chain" in blockers


def test_v6_integrity_guard_rejects_malformed_nominal_stack():
    text = "Beyond accuracy students must evaluate usefulness ethics trustworthiness."

    blockers = candidate_integrity_blockers(text)

    assert "malformed_nominal_stack" in blockers


def test_v6_selector_accepts_risk_mitigating_integrity_warning_for_review():
    source = "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve."
    paragraph = scan_text(source).paragraphs[0]
    candidate = Variant(
        id="v1",
        source="writer",
        text=(
            "The process uses forms and queues. "
            "Labels and reviews support approval checks (Smith. They also carries 2024). "
            "The checks help teams improve."
        ),
    )

    selected = choose_variant([Variant(id="source_preserved", source="source_preserved", text=source), candidate], paragraph)

    assert selected and selected.source == "writer"
    assert selected.author_review_items
    assert any(
        "broken_citation_shape" in item.get("target_text", "")
        for item in selected.author_review_items
    )


def test_v6_selector_flags_over_decomposed_risk_mitigation_for_review():
    source = "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve."
    paragraph = scan_text(source).paragraphs[0]
    candidate = Variant(
        id="v1",
        source="writer",
        text=(
            "The process uses forms. "
            "Queues support the work. "
            "Labels support the work. "
            "Reviews support the work. "
            "Approvals support the work. "
            "Checks help teams improve."
        ),
    )

    selected = choose_variant([Variant(id="source_preserved", source="source_preserved", text=source), candidate], paragraph)
    diagnostics = selection_diagnostics([Variant(id="source_preserved", source="source_preserved", text=source), candidate], paragraph)

    assert selected and selected.source == "writer"
    assert any(
        "sentence_count_expansion" in item.get("target_text", "")
        for item in selected.author_review_items or []
    )
    assert "sentence_count_expansion_review_required" in diagnostics[0]["quality_warnings"]
    assert "short_sentence_chain_review_required" in diagnostics[0]["quality_warnings"]


def test_v6_selector_prefers_less_mechanical_retry_over_zero_finding_short_chain():
    source = (
        "The process uses forms, queues, labels, and reviews because teams should improve. "
        "The team records updates, notes, and approvals because communication should improve."
    )
    paragraph = scan_text(source).paragraphs[0]
    over_cut = Variant(
        id="v1",
        source="writer",
        text=(
            "The process uses forms. "
            "Queues support the work. "
            "Labels support the work. "
            "Reviews support improvement. "
            "The team records updates. "
            "Notes support communication. "
            "Approvals support communication."
        ),
    )
    cleaner = Variant(
        id="retry_v1",
        source="writer",
        text=(
            "The process uses forms and queues while labels and reviews support team improvement. "
            "The team records updates and notes while approvals support communication."
        ),
    )

    selected = choose_variant([Variant(id="source_preserved", source="source_preserved", text=source), over_cut, cleaner], paragraph)

    assert selected and selected.id == "retry_v1"


def test_v6_candidate_diagnostics_include_missing_terms_for_retry_feedback():
    source = "The process uses forms, queues, labels, reviews, approvals, and checks because teams should improve."
    response = json.dumps({
        "variants": [{
            "id": "v1",
            "text": "The process uses forms and queues because teams should improve.",
            "coverage_map": [],
            "route_answer_cards": [],
            "author_proxy_provenance": [],
            "author_review_items": [],
        }]
    })
    client = SequencedVariantClient([response, '{"variants":[]}'])

    result = run_v6_rewrite_all(source, writer_client=client, max_passes=1)
    diagnostic = result.pass_trace[0]["candidate_diagnostics"][0]

    assert "required_source_terms_missing_review_required" in diagnostic["quality_warnings"]
    assert set(diagnostic["missing_required_terms"]) & {"labels", "reviews", "approvals", "checks"}


def test_v6_citation_anchor_recipe_keeps_attribution_as_source_to_claim_relation():
    scan = scan_text("According to cognitive load theory, complex spatial task such as constructing haircutting structure place a significant strain on working memory.")
    paragraph, plan = build_plan(scan)
    recipe = plan.ai_safe_route["construction_recipes"][0]

    assert "source attribution phrase" in recipe["build_route"]
    assert any("attribution" in step for step in recipe["build_steps"])
