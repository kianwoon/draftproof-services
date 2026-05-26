from __future__ import annotations

import json

from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6.paragraph_architecture import apply_architecture_split_text, architecture_split_contract
from poc.rewrite_v6.pipeline import _acceptable_progress, _cross_paragraph_regression, _report_target_paragraph_ids, _same_text, run_v6_rewrite_all
from poc.rewrite_v6.planner_llm import build_planner_prompt
from poc.rewrite_v6.prose_quality import drop_redundant_adjacent_sentence_intent, has_fragment_or_trace_sentences, repair_generated_prose
from poc.rewrite_v6.report_contracts import apply_report_signal_contracts, extract_report_signal_contracts
from poc.rewrite_v6.scan import findings_for_paragraph, scan_text
from poc.rewrite_v6.write import Variant, build_prompt, choose_variant


class EmptyVariantClient:
    def __init__(self):
        self.prompts: list[str] = []

    def chat(self, prompt: str, **_kwargs):
        self.prompts.append(prompt)
        return type("Response", (), {"content": '{"variants":[]}', "raw_content": '{"variants":[]}'})()


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
    writer_contracts = writer_payload["planner_decision"]["document_signal_contracts"]
    direct_writer_contracts = writer_payload["document_signal_contracts"]
    assert planner_contracts[0]["signal_group"] == "grounding_route"
    assert planner_contracts[0]["target_excerpts"] == ["form queue review"]
    assert writer_contracts[0]["writer_obligation"] == "keep claims near submitted support"
    assert direct_writer_contracts[0]["signal_group"] == "grounding_route"
    assert "document_signal_contracts as the primary build contract" in " ".join(writer_payload["generation_rules"])


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

    assert writer.prompts
    assert "haircutting structure" in writer.prompts[0]
    assert "ordinary setup" not in writer.prompts[0]
    assert result.pass_trace[0]["status"] == "no_change"
    assert result.pass_trace[0]["target_paragraph_id"] == "p002"
    assert not result.passes


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
    assert "sentence_row_id for each covered group" in " ".join(payload["generation_rules"])
    assert "cover every group in coverage_map" in " ".join(payload["generation_rules"])
    assert "one final sentence per group" not in build_prompt(paragraph, plan)
    assert "same row sequence joined" not in build_prompt(paragraph, plan)
    assert "coverage_map to prove group coverage" in build_prompt(paragraph, plan)
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


def test_v6_writer_schema_keeps_coverage_map_separate_from_sentence_text():
    paragraph, plan = build_plan(scan_text("The process uses a form, a queue, and a review."))
    payload = json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])
    variant_schema = payload["output_schema"]["variants"][0]
    coverage_row = variant_schema["coverage_map"][0]
    sentence_row = variant_schema["sentence_rows"][0]

    assert "sentence" not in coverage_row
    assert coverage_row["sentence_row_id"]
    assert "sentence" in sentence_row


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


def test_v6_citation_anchor_recipe_keeps_attribution_as_source_to_claim_relation():
    scan = scan_text("According to cognitive load theory, complex spatial task such as constructing haircutting structure place a significant strain on working memory.")
    paragraph, plan = build_plan(scan)
    recipe = plan.ai_safe_route["construction_recipes"][0]

    assert "source attribution phrase" in recipe["build_route"]
    assert any("attribution" in step for step in recipe["build_steps"])
