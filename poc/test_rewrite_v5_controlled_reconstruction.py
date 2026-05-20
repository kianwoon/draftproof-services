import json
from pathlib import Path
from types import SimpleNamespace

import rewrite_v5.production as v5_production
import rewrite_v5.residual_comb as v5_residual_comb
from rewrite_v3.text_integrity import minimal_replacement_text_integrity
from rewrite_v5.cluster_mass import build_cluster_mass_prompt
from rewrite_v5.experiment import (
    _is_safe_candidate,
    _section_from_cluster,
    _stack_summary,
    build_fact_map,
    build_recomposer_prompt,
    build_section_units,
    fact_map_integrity,
)
from rewrite_v5.residual_comb import (
    build_risky_window_cleanup_prompt,
    build_direct_scanner_leapfrog_prompt,
    build_residual_cluster_prompt,
    build_residual_cluster_route_plan_prompt,
    build_residual_cluster_retune_prompt,
    build_route_blueprint,
    build_unsafe_cluster_cleanup_prompt,
    generate_residual_cluster_route_plan,
    run_v5_residual_cluster_comb_experiment,
    _best_full_document_candidate,
    _best_balanced_ai_topk_candidate,
    _balanced_ai_topk_sort_value,
    _direct_scanner_candidate_strong_enough,
    _direct_scanner_batch_policy,
    _should_continue_direct_scanner_batches,
    generate_residual_cluster_seed_variants,
    _expand_to_local_text_boundaries,
    _full_document_candidate_beats_scores,
    _generate_loose_variants_from_builder,
    _has_risky_window_cleanup_movement,
    _has_balanced_ai_topk_movement,
    _has_unsafe_cluster_cleanup_movement,
    _has_full_document_fallback_movement,
    _ordered_density_cluster_rows,
    _parallel_single_variant_max_tokens,
    _parse_route_plan,
    _runtime_budget_exhausted,
    _runtime_budget_stop_record,
    _section_apply_boundary_integrity,
    _run_unsafe_cluster_cleanup_pass,
    _serial_variant_max_tokens,
    _should_start_with_unsafe_cluster_cleanup,
    _unsafe_cluster_cleanup_stop_after_misses,
    _unsafe_cluster_probe_round_limit,
    _text_integrity_regression,
    _would_discard_structural_progress,
    _has_incremental_movement,
    _residual_candidate_sort_key,
    _adaptive_initial_variant_count,
    _adaptive_retune_variant_count,
    _adaptive_writer_feedback,
    _should_generate_adaptive_remainder,
    _should_retune_residual_candidate,
)
from rewrite_v5.models import RecompositionVariant, SectionUnit


def _sample_route_plan() -> dict:
    return {
        "content_profile": "narrative_or_case_reflection",
        "primary_metric": "topk_density",
        "cluster_role": "evidence_or_example",
        "dominant_failure_pattern": "event_summary",
        "route_strategy": "event_first_rebuild",
        "profile_reason": "The cluster follows an event and its visible outcome.",
        "failed_route": "The current route summarizes the result too quickly.",
        "replacement_route": "Start from the event, then show the visible outcome.",
        "topk_route_diagnosis": {
            "infected_unit_id": "u001",
            "current_route": "broad result opener",
            "predictable_path": "service changed -> confidence",
            "primary_operator": "CLAUSE_ROUTE_CHANGE",
            "replacement_route": "event evidence -> visible result",
            "insufficient_edit": "Only swapping changed with improved.",
        },
        "source_block_plan": [
            {
                "block_id": "b01",
                "current_job": "Summarize result.",
                "rewrite_job": "Start from the event and end with the visible outcome.",
                "must_preserve": ["The thank-you card made the result visible."],
            }
        ],
        "target_sentence_jobs": [
            {
                "sentence_id": "s001",
                "source_preview": "The service changed the student's confidence.",
                "current_weakness": "Broad result opener.",
                "rewrite_job": "Show service before result.",
                "avoid_copying": ["The service changed"],
            }
        ],
        "affected_unit_actions": [
            {
                "unit_id": "u001",
                "affected_text": "The service changed the student's confidence.",
                "problem_role": "The unit states the result before the evidence.",
                "required_action": "Move the broad result after the visible event.",
                "operator_stack": ["CLAUSE_ROUTE_CHANGE"],
                "must_preserve": ["The service changed the student's confidence."],
                "insufficient_edit": "Only replacing service or confidence with synonyms.",
            }
        ],
        "must_change": ["Move the broad result after the visible event."],
        "must_preserve": [
            {
                "source_quote": "The thank-you card made the result visible.",
                "preserve_as": "visible outcome",
            }
        ],
        "sentence_plan": ["Open with the service.", "End with the thank-you card outcome."],
        "avoid_phrases": ["The service changed"],
        "length_target": "same_length",
        "reason_this_should_move_score": "Event-first route should reduce the broad summary pattern.",
    }


class _FakeV5LLMResponse:
    def __init__(self, content: str, *, model: str = "fake/model", finish_reason: str = "stop") -> None:
        self.content = content
        self.model = model
        self._finish_reason = finish_reason
        self.usage = {"total_tokens": 12}
        self.raw = {
            "provider": "fake-provider",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "native_finish_reason": finish_reason,
                    "message": {"content": content},
                }
            ],
        }

    @property
    def raw_content(self) -> str:
        return self.content

    @property
    def finish_reason(self) -> str:
        return self._finish_reason

    @property
    def native_finish_reason(self) -> str:
        return self._finish_reason


def test_v5_sections_group_heading_with_following_paragraphs():
    text = (
        "Introduction\n"
        "The first section frames the learning issue.\n\n"
        "Practice Context\n"
        "The second section explains a practical difficulty.\n"
        "It includes a supported claim (Lee, 2024).\n\n"
        "Conclusion\n"
        "The final section returns to the main point."
    )

    sections = build_section_units(text, {})

    assert [section.heading for section in sections] == ["Introduction", "Practice Context", "Conclusion"]
    assert sections[1].text.startswith("Practice Context\n")
    assert "supported claim" in sections[1].text
    assert sections[1].metadata["ordinal"] == 2
    assert sections[1].metadata["section_count"] == 3


def test_v5_fact_map_preserves_citations_and_generic_routes():
    text = (
        "Practice Context\n"
        "The lesson became difficult when students had to connect feedback with action. "
        "In my reflection, the issue was not motivation but the way support was introduced. "
        "The wider claim is supported by inclusive design research (Lee, 2024)."
    )
    section = build_section_units(text, {})[0]

    fact_map = build_fact_map(section, {})

    assert fact_map.section_role == "introduction/background framing"
    assert "(Lee, 2024)" in fact_map.citations
    assert any("my reflection" in item for item in fact_map.personal_observations)
    assert not any("Johnny" in item for item in fact_map.better_route)
    assert not any("Inclusive Learning Environment" in item for item in fact_map.better_route)


def test_v5_prompt_uses_small_schema_and_no_detector_language():
    text = (
        "Body Section\n"
        "Students need time to connect feedback with action. "
        "The source claim stays close to practice (Lee, 2024)."
    )
    section = build_section_units(text, {})[0]
    fact_map = build_fact_map(section, {})
    prompt = build_recomposer_prompt(section=section, fact_map=fact_map, variant_count=2)
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "controlled_section_reconstruction_from_fact_map"
    assert len(payload["output_schema"]["variants"]) == 2
    assert set(payload["output_schema"]["variants"][0].keys()) == {"variant_id", "text"}
    lowered = prompt.casefold()
    assert "ai detector" not in lowered
    assert "bypass" not in lowered


def test_v5_cluster_mass_prompt_matches_length_preserved_experiment_shape():
    cluster = type("Cluster", (), {
        "cluster_id": "cluster_001",
        "text": (
            "At the beginning of the course, Johnny needed support in class. "
            "I used role-playing activities to help him take part."
        ),
        "start_char": 10,
        "end_char": 130,
        "risk_score": 12.0,
        "sentence_count": 2,
        "metadata": {},
    })()
    section = _section_from_cluster(cluster)
    fact_map = build_fact_map(section, {})

    prompt = build_cluster_mass_prompt(
        section=section,
        fact_map=fact_map,
        variant_count=3,
        min_word_ratio=0.90,
        max_word_ratio=1.50,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "length_preserved_route_window_replacement"
    assert payload["route_window"]["target_word_count"]["preferred"] == section.word_count
    assert payload["route_window"]["target_word_count"]["min"] == round(section.word_count * 0.90)
    assert payload["route_window"]["target_word_count"]["max"] == round(section.word_count * 1.50)
    assert any("Replace the whole route window" in item for item in payload["replacement_goal"])
    assert len(payload["output_schema"]["variants"]) == 3
    lowered = prompt.casefold()
    assert "ai detector" not in lowered
    assert "bypass" not in lowered


def test_v5_residual_cluster_prompt_uses_compact_repair_task():
    cluster = type("Cluster", (), {
        "cluster_id": "cluster_001",
        "text": (
            "Johnny needed support in class. "
            "The role-play helped him lead a salon group."
        ),
        "start_char": 5,
        "end_char": 90,
        "risk_score": 10.0,
        "sentence_count": 2,
        "metadata": {},
    })()
    section = _section_from_cluster(cluster)

    goal = {
        "eligible_span_density_gate": {
            "top_unsafe_clusters": [
                {"preview": "Johnny needed support in class."},
            ],
        },
    }
    prompt = build_residual_cluster_prompt(section=section, local_goal=goal, variant_count=4)
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "residual_cluster_route_bump"
    assert payload["cluster"]["section_id"] == "route_001"
    assert payload["cluster"]["source_event_beats"]
    assert payload["cluster"]["source_blocks"]
    assert payload["cluster"]["source_phrase_anchors"]
    assert payload["cluster"]["referential_continuity"]["preserve_opening_subject"] is False
    assert payload["custom_route_plan"] is None
    assert payload["fallback_route_blueprint"]["steps"]
    assert payload["coverage_guidance"]["requirements"]
    assert payload["length_guidance"]["preferred_min_words"] > section.word_count
    assert payload["fallback_route_blueprint"]["sentence_jobs"]
    assert payload["remaining_problem_sentences"] == ["Johnny needed support in class."]
    assert len(payload["output_schema"]["variants"]) == 4
    assert all(set(row.keys()) == {"variant_id", "text"} for row in payload["output_schema"]["variants"])
    lowered = prompt.casefold()
    assert "ai detector" not in lowered
    assert "bypass" not in lowered
    assert "current_route" not in lowered
    assert "relevant to the source topic" in lowered
    assert "fake citations" in lowered
    assert "source viewpoint" in lowered
    assert "source-near" not in lowered


def test_v5_residual_retune_prompt_focuses_on_remaining_sentence_without_scores():
    cluster = type("Cluster", (), {
        "cluster_id": "cluster_002",
        "text": "The student finished the task. The feedback changed his confidence.",
        "start_char": 5,
        "end_char": 80,
        "risk_score": 9.0,
        "sentence_count": 2,
        "metadata": {},
    })()
    section = _section_from_cluster(cluster)
    goal = {
        "eligible_span_density_gate": {
            "top_unsafe_clusters": [
                {"preview": "The feedback changed his confidence."},
            ],
        },
    }

    prompt = build_residual_cluster_retune_prompt(
        section=section,
        current_best_text="The student finished the task and got feedback.",
        local_goal=goal,
        variant_count=2,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "residual_cluster_retune"
    assert payload["cluster"]["source_event_beats"]
    assert payload["cluster"]["source_blocks"]
    assert payload["cluster"]["source_phrase_anchors"]
    assert payload["cluster"]["referential_continuity"]["opening_subject"] == "The"
    assert payload["custom_route_plan"] is None
    assert payload["fallback_route_blueprint"]["steps"]
    assert payload["length_guidance"]["preferred_min_words"] > section.word_count
    assert "retune_focus" in payload
    assert "candidate_non_source_terms_to_reduce" in payload
    assert payload["fallback_route_blueprint"]["sentence_jobs"]
    assert payload["remaining_problem_sentences"] == ["The feedback changed his confidence."]
    assert len(payload["output_schema"]["variants"]) == 2
    lowered = prompt.casefold()
    assert "ai detector" not in lowered
    assert "bypass" not in lowered
    assert "relevant to the source topic" in lowered
    assert "fake citations" in lowered


def test_v5_residual_route_plan_prompt_builds_custom_planner_task():
    cluster = type("Cluster", (), {
        "cluster_id": "cluster_002",
        "text": "The service changed the student's confidence. The thank-you card made the result visible.",
        "start_char": 5,
        "end_char": 90,
        "risk_score": 10.0,
        "sentence_count": 2,
        "metadata": {},
    })()
    section = _section_from_cluster(cluster)
    goal = {
        "eligible_span_density_gate": {
            "top_unsafe_clusters": [{"preview": "The service changed the student's confidence."}],
            "top_sentence_targets": [
                {"sentence_id": "s001", "preview": "The service changed the student's confidence.", "word_count": 6},
            ],
            "recommended_actions": ["target_longest_unsafe_cluster"],
        },
    }

    prompt = build_residual_cluster_route_plan_prompt(section=section, local_goal=goal)
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "score_causal_cluster_route_plan"
    assert payload["cluster"]["source_event_beats"]
    assert payload["cluster"]["source_blocks"]
    assert payload["cluster"]["referential_continuity"]["opening_subject"] == "The"
    assert payload["scanner_local_findings"]["top_sentence_targets"][0]["sentence_id"] == "s001"
    assert payload["affected_content_map"][0]["unit_id"] == "u001"
    assert payload["affected_content_map"][0]["is_scanner_target"] is True
    assert "primary_metric_options" in payload
    assert "topk_operator_options" in payload
    assert "CLAUSE_ROUTE_CHANGE" in payload["topk_operator_options"]
    assert "content_profile_rubrics" in payload
    assert "broad_explanatory_report" in payload["content_profile_rubrics"]
    broad_rubric = payload["content_profile_rubrics"]["broad_explanatory_report"]
    assert "source-supported specificity" in broad_rubric["planning_focus"]
    assert "concrete framing or explanatory bridge" in broad_rubric["planning_focus"]
    assert "personal experience" not in broad_rubric["avoid"]
    assert "cluster_role_options" in payload
    assert "failure_pattern_options" in payload
    assert "route_strategy_options" in payload
    assert "route_plan" in payload["output_schema"]
    lowered = prompt.casefold()
    assert "controlled_expansion_move_options" in payload
    assert "controlled expansion" in lowered
    assert "do not use a reflective-practice route for broad report content" not in lowered
    assert set(payload["output_schema"]["route_plan"].keys()) == {
        "content_profile",
        "primary_metric",
        "cluster_role",
        "dominant_failure_pattern",
        "route_strategy",
        "profile_reason",
        "failed_route",
        "replacement_route",
        "topk_route_diagnosis",
        "source_block_plan",
        "target_sentence_jobs",
        "affected_unit_actions",
        "must_change",
        "must_preserve",
        "controlled_expansion",
        "sentence_plan",
        "avoid_phrases",
        "length_target",
        "reason_this_should_move_score",
    }
    lowered = prompt.casefold()
    assert "ai detector" not in lowered
    assert "bypass" not in lowered


def test_v5_residual_route_plan_parser_requires_source_supported_steps():
    raw = json.dumps({
        "route_plan": {
            "content_profile": "narrative_or_case_reflection",
            "primary_metric": "topk_density",
            "cluster_role": "evidence_or_example",
            "dominant_failure_pattern": "event_summary",
            "route_strategy": "event_first_rebuild",
            "profile_reason": "The cluster follows a service result and a thank-you card outcome.",
            "failed_route": "The current route starts with broad interpretation before showing the event.",
            "replacement_route": "Start from the service moment, then show how the thank-you card made the result visible.",
            "topk_route_diagnosis": {
                "infected_unit_id": "u001",
                "current_route": "broad interpretation before event evidence",
                "predictable_path": "service changed -> confidence",
                "primary_operator": "CLAUSE_ROUTE_CHANGE",
                "replacement_route": "service moment -> thank-you card -> visible result",
                "insufficient_edit": "Only changing changed to improved.",
            },
            "source_block_plan": [
                {
                    "block_id": "b01",
                    "current_job": "Summarize the service and thank-you card.",
                    "rewrite_job": "Show the service first, then the thank-you card as visible outcome.",
                    "must_preserve": [
                        "The service changed the student's confidence.",
                        "The thank-you card made the result visible.",
                    ],
                }
            ],
            "target_sentence_jobs": [
                {
                    "sentence_id": "s001",
                    "source_preview": "The service changed the student's confidence.",
                    "current_weakness": "Broad result before visible event.",
                    "rewrite_job": "Move the service result after the visible thank-you card.",
                    "avoid_copying": ["The service changed"],
                }
            ],
            "affected_unit_actions": [
                {
                    "unit_id": "u001",
                    "affected_text": "The service changed the student's confidence.",
                    "problem_role": "The unit carries broad result wording before the evidence.",
                    "required_action": "Move the result after the thank-you card evidence.",
                    "operator_stack": ["CLAUSE_ROUTE_CHANGE"],
                    "must_preserve": ["The service changed the student's confidence."],
                    "insufficient_edit": "Changing only one noun while keeping the same route.",
                }
            ],
            "must_change": [
                "Move the interpretation after the service and thank-you card.",
                "Replace the broad opening with event movement.",
            ],
            "must_preserve": [
                {
                    "source_quote": "The service changed the student's confidence.",
                    "preserve_as": "service impact on confidence",
                },
                {
                    "source_quote": "The thank-you card made the result visible.",
                    "preserve_as": "visible outcome",
                },
            ],
            "sentence_plan": [
                "Open with the service moment.",
                "Use the thank-you card to make the result visible.",
            ],
            "avoid_phrases": ["The service changed"],
            "length_target": "same_length",
            "reason_this_should_move_score": "The route should move because it replaces a broad claim with event-first evidence.",
        }
    })

    parsed, diagnostics = _parse_route_plan(
        raw,
        source_text="The service changed the student's confidence. The thank-you card made the result visible.",
    )
    unsupported, unsupported_diagnostics = _parse_route_plan(
        raw,
        source_text="Nothing from the quoted source appears here.",
    )

    assert diagnostics["status"] == "ok"
    assert diagnostics["content_profile"] == "narrative_or_case_reflection"
    assert diagnostics["dominant_failure_pattern"] == "event_summary"
    assert diagnostics["route_strategy"] == "event_first_rebuild"
    assert diagnostics["source_block_plan_count"] == 1
    assert diagnostics["target_sentence_job_count"] == 1
    assert parsed["cluster_role"] == "evidence_or_example"
    assert parsed["primary_metric"] == "topk_density"
    assert parsed["topk_route_diagnosis"]["primary_operator"] == "CLAUSE_ROUTE_CHANGE"
    assert parsed["affected_unit_actions"][0]["unit_id"] == "u001"
    assert parsed["affected_unit_actions"][0]["operator_stack"] == ["CLAUSE_ROUTE_CHANGE"]
    assert parsed["replacement_route"].startswith("Start from the service moment")
    assert parsed["must_preserve"][0]["source_quote"] == "The service changed the student's confidence."
    assert parsed["avoid_phrases"][0] == "The service changed"
    assert parsed["length_target"] == "same_length"
    assert unsupported is None
    assert unsupported_diagnostics["status"] == "schema_failed"
    assert unsupported_diagnostics["dropped_must_preserve_count"] == 2


def test_v5_residual_route_plan_rejects_summarized_preserve_anchors():
    raw = json.dumps({
        "route_plan": {
            "content_profile": "narrative_or_case_reflection",
            "cluster_role": "evidence_or_example",
            "dominant_failure_pattern": "event_summary",
            "route_strategy": "event_first_rebuild",
            "profile_reason": "The cluster follows a service result and visible outcome.",
            "failed_route": "The current route starts with broad interpretation before showing the event.",
            "replacement_route": "Start from the service moment, then show the thank-you card as evidence.",
            "source_block_plan": [
                {
                    "block_id": "b01",
                    "current_job": "Summarize event.",
                    "rewrite_job": "Show visible event before interpretation.",
                    "must_preserve": ["The service changed the student's confidence."],
                }
            ],
            "target_sentence_jobs": [
                {
                    "sentence_id": "s001",
                    "source_preview": "The service changed the student's confidence.",
                    "current_weakness": "Broad result first.",
                    "rewrite_job": "Move result later.",
                    "avoid_copying": ["The service changed"],
                }
            ],
            "must_change": ["Move the interpretation after the visible event."],
            "must_preserve": [
                {
                    "source_quote": "The fact that this was a learning experience for Johnny.",
                    "preserve_as": "learning experience",
                }
            ],
            "sentence_plan": ["Open with the service moment."],
            "avoid_phrases": ["The service changed"],
            "length_target": "same_length",
            "reason_this_should_move_score": "The route should move because it replaces summary with evidence.",
        }
    })

    parsed, diagnostics = _parse_route_plan(
        raw,
        source_text="The service changed the student's confidence. The thank-you card made the result visible.",
    )

    assert parsed is None
    assert diagnostics["status"] == "schema_failed"
    assert diagnostics["dropped_must_preserve_count"] == 1


def test_v5_route_plan_uses_fallback_gateway_after_invalid_planner_output():
    source = (
        "The service changed the student's confidence. "
        "The thank-you card made the result visible."
    )
    section = SectionUnit(
        section_id="density_cluster_001",
        heading="Density cluster",
        text=source,
        start_char=0,
        end_char=len(source),
        paragraph_count=1,
        word_count=10,
        metadata={},
    )
    calls: list[str] = []

    class FakeGateway:
        provider = None

        def __init__(self, model: str, response: str) -> None:
            self.model = model
            self.response = response

        def chat(self, *_args, **_kwargs):
            calls.append(self.model)
            return _FakeV5LLMResponse(self.response, model=self.model)

    primary = FakeGateway("z-ai/glm-5.1", '{"route_plan": {"failed_route": "missing fields"}}')
    fallback = FakeGateway("deepseek/deepseek-v3.2", json.dumps({"route_plan": _sample_route_plan()}))

    plan, diagnostics, _prompt, _raw = generate_residual_cluster_route_plan(
        section=section,
        local_goal={},
        planner_gateway=primary,
        fallback_gateway=fallback,
    )

    assert calls == ["z-ai/glm-5.1", "deepseek/deepseek-v3.2"]
    assert plan is not None
    assert diagnostics["status"] == "ok"
    assert diagnostics["planner_fallback_used"] is True
    assert diagnostics["planner_model_requested"] == "deepseek/deepseek-v3.2"
    assert diagnostics["primary_planner_attempt"]["planner_model_requested"] == "z-ai/glm-5.1"


def test_v5_route_plan_uses_scanner_derived_fallback_when_planners_fail():
    source = (
        "The United States includes people from many ethnic, cultural, and religious backgrounds. "
        "This diversity has contributed to creativity and innovation in many fields. "
        "It has also created challenges related to racism, discrimination, and social inequality."
    )
    section = SectionUnit(
        section_id="route_001",
        heading="Density cluster",
        text=source,
        start_char=0,
        end_char=len(source),
        paragraph_count=1,
        word_count=len(source.split()),
        metadata={},
    )

    class FailingGateway:
        provider = None

        def __init__(self, model: str) -> None:
            self.model = model

        def chat(self, *_args, **_kwargs):
            return _FakeV5LLMResponse('{"not_route_plan": true}', model=self.model)

    plan, diagnostics, _prompt, _raw = generate_residual_cluster_route_plan(
        section=section,
        local_goal={},
        planner_gateway=FailingGateway("z-ai/glm-5.1"),
        fallback_gateway=FailingGateway("deepseek/deepseek-v3.2"),
    )

    assert plan is not None
    assert diagnostics["status"] == "ok"
    assert diagnostics["deterministic_fallback_used"] is True
    assert diagnostics["route_plan_source"] == "scanner_derived_fallback"
    assert plan["affected_unit_actions"]
    assert plan["topk_route_diagnosis"]["primary_operator"] == "CLAUSE_ROUTE_CHANGE"


def test_v5_scanner_derived_fallback_uses_structured_generic_pressure_not_keywords():
    source = (
        "Cities change when transport, housing, jobs, and public services move at different speeds. "
        "The result is often a broad pressure on daily life."
    )
    section = SectionUnit(
        section_id="route_generic",
        heading="Density cluster",
        text=source,
        start_char=0,
        end_char=len(source),
        paragraph_count=1,
        word_count=len(source.split()),
        metadata={},
    )
    local_goal = {
        "eligible_span_density_gate": {
            "top_sentence_targets": [
                {
                    "sentence_id": "s001",
                    "preview": "Cities change when transport, housing, jobs, and public services move at different speeds.",
                    "word_count": 12,
                    "generic_hits": 3,
                }
            ],
            "recommended_actions": ["CLAUSE_ROUTE_CHANGE"],
        }
    }

    class FailingGateway:
        provider = None
        model = "z-ai/glm-5.1"

        def chat(self, *_args, **_kwargs):
            return _FakeV5LLMResponse('{"not_route_plan": true}', model=self.model)

    plan, diagnostics, _prompt, _raw = generate_residual_cluster_route_plan(
        section=section,
        local_goal=local_goal,
        planner_gateway=FailingGateway(),
        fallback_gateway=None,
    )

    assert diagnostics["route_plan_source"] == "scanner_derived_fallback"
    assert plan["content_profile"] == "broad_explanatory_report"
    assert plan["controlled_expansion"]["required"] is True
    assert plan["controlled_expansion"]["move"] == "explanatory_bridge"


def test_v5_route_plan_truncation_skips_second_planner_call():
    source = (
        "The paragraph gives a broad explanation of a topic. "
        "It lists related points in a predictable order. "
        "The next sentence repeats the same report-style movement."
    )
    section = SectionUnit(
        section_id="route_001",
        heading="Density cluster",
        text=source,
        start_char=0,
        end_char=len(source),
        paragraph_count=1,
        word_count=len(source.split()),
        metadata={},
    )
    calls: list[str] = []

    class TruncatedGateway:
        provider = None

        def __init__(self, model: str, finish_reason: str = "length") -> None:
            self.model = model
            self.finish_reason = finish_reason

        def chat(self, *_args, **_kwargs):
            calls.append(self.model)
            return _FakeV5LLMResponse('{"route_plan": {"failed_route": "cut off"', model=self.model, finish_reason=self.finish_reason)

    plan, diagnostics, _prompt, _raw = generate_residual_cluster_route_plan(
        section=section,
        local_goal={},
        planner_gateway=TruncatedGateway("z-ai/glm-5.1"),
        fallback_gateway=TruncatedGateway("deepseek/deepseek-v3.2", finish_reason="stop"),
    )

    assert calls == ["z-ai/glm-5.1"]
    assert plan is not None
    assert diagnostics["deterministic_fallback_used"] is True
    assert diagnostics["planner_fallback_used"] is False
    assert diagnostics["failed_planner_finish_reason"] == "length"


def test_v5_residual_prompt_uses_executable_brief_without_fallback_noise():
    cluster = type("Cluster", (), {
        "cluster_id": "cluster_004",
        "text": (
            "The service changed the student's confidence. "
            "The thank-you card made the result visible."
        ),
        "start_char": 5,
        "end_char": 90,
        "risk_score": 10.0,
        "sentence_count": 2,
        "metadata": {},
    })()
    section = _section_from_cluster(cluster)
    route_plan = _sample_route_plan()

    prompt = build_residual_cluster_prompt(
        section=section,
        local_goal={},
        variant_count=1,
        route_plan=route_plan,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["execution_brief"] == route_plan
    assert payload["writer_execution_card"]["main_operator"] == "CLAUSE_ROUTE_CHANGE"
    assert payload["writer_execution_card"]["route_to_write"] == "event evidence -> visible result"
    assert payload["writer_execution_card"]["unit_actions"][0]["required_action"]
    assert payload["writer_variant_plan"][0]["variant_id"] == "v1"
    assert payload["writer_variant_plan"][0]["main_operator"] == "CLAUSE_ROUTE_CHANGE"
    assert payload["coverage_guidance"]["requirements"]
    assert "fallback_route_blueprint" not in payload
    assert "custom_route_plan" not in payload
    assert payload["length_guidance"]["preferred_max_words"] == round(section.word_count * 1.10)
    lowered = prompt.casefold()
    assert "follow execution_brief.replacement_route" in lowered
    assert "fallback_route_blueprint" not in lowered
    assert "source-near" not in lowered
    assert "outside examples" not in lowered


def test_v5_residual_prompt_can_carry_score_feedback_for_adaptive_retry():
    section = SectionUnit(
        section_id="density_cluster_001",
        heading="Density cluster",
        text="The service changed the student's confidence. The thank-you card made the result visible.",
        start_char=0,
        end_char=83,
        paragraph_count=1,
        word_count=12,
        metadata={},
    )
    feedback = {
        "reason": "topk_route_not_moved",
        "primary_metric": "topk_density",
        "required_correction": "Break the predictable sentence path.",
    }

    prompt = build_residual_cluster_prompt(
        section=section,
        local_goal={},
        variant_count=2,
        route_plan=_sample_route_plan(),
        adaptive_feedback=feedback,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["score_feedback"] == feedback
    assert any("did not move top-k" in rule for rule in payload["adaptive_retry_rules"])
    assert not any("outside examples" in rule for rule in payload["adaptive_retry_rules"])


def test_v5_broad_report_writer_requires_controlled_expansion_without_old_guards():
    section = SectionUnit(
        section_id="density_cluster_usa",
        heading="Density cluster",
        text=(
            "The United States has a large economy and an important role in the world. "
            "It has many industries, strong universities, and influence in politics and culture. "
            "These factors make the country powerful, but they also create pressure and disagreement."
        ),
        start_char=0,
        end_char=231,
        paragraph_count=1,
        word_count=38,
        metadata={},
    )
    route_plan = {
        **_sample_route_plan(),
        "content_profile": "broad_explanatory_report",
        "cluster_role": "background_context",
        "dominant_failure_pattern": "category_dump",
        "route_strategy": "group_and_bridge",
        "controlled_expansion": {
            "required": True,
            "move": "concrete_framing",
            "instruction": "Frame the broad country claim through one concrete relation between economy, institutions, and public pressure.",
            "why_needed": "The current route stacks categories without a bridge.",
        },
        "must_preserve": [
            {"source_quote": "large economy", "preserve_as": "economic claim"}
        ],
    }

    prompt = build_residual_cluster_prompt(
        section=section,
        local_goal={},
        variant_count=3,
        route_plan=route_plan,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))
    lowered = prompt.casefold()

    assert payload["writer_execution_card"]["controlled_expansion"]["required"] is True
    assert payload["writer_execution_card"]["controlled_expansion"]["move"] == "concrete_framing"
    assert all(row["controlled_expansion_move"] == "concrete_framing" for row in payload["writer_variant_plan"])
    assert "source-near" not in lowered
    assert "outside examples" not in lowered
    assert "unverifiable named facts" not in lowered


def test_v5_adaptive_writer_feedback_triggers_topk_route_retry(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_ADAPTIVE_WRITER", raising=False)
    route_plan = {**_sample_route_plan(), "primary_metric": "topk_density"}
    weak_row = {
        "apply_status": {"applied": True},
        "scores": {"rank_delta": 1.0, "topk_delta": 0.0, "unsafe_cluster_count_delta": 0.0},
        "local_scores": {
            "unsafe_cluster_count": 1,
            "unsafe_word_ratio": 20.0,
            "unsafe_cluster_count_delta": 0.0,
            "unsafe_word_ratio_delta": 0.0,
            "topk_delta": 0.0,
            "rank_delta": 0.0,
        },
        "incremental": {
            "ai_delta": 0.0,
            "topk_delta": 0.0,
            "external_delta": 0.0,
            "rank_delta": 0.0,
            "unsafe_cluster_count_delta": 0.0,
        },
    }

    feedback = _adaptive_writer_feedback([weak_row], route_plan=route_plan, selected=weak_row)

    assert _adaptive_initial_variant_count(5, route_plan) == 2
    assert _adaptive_retune_variant_count(5, route_plan) == 2
    assert feedback["reason"] == "topk_route_not_moved"
    assert _should_generate_adaptive_remainder(feedback, remaining_count=3, best_candidate=weak_row)
    assert not _should_retune_residual_candidate(weak_row, route_plan=route_plan, adaptive_feedback=feedback)


def test_v5_direct_scanner_leapfrog_prompt_uses_scanner_cluster_and_small_variants():
    section = SectionUnit(
        section_id="density_cluster_001",
        heading="Density cluster cleanup",
        text=(
            "Students received knowledge from trusted sources, practiced it through homework, "
            "and proved their learning through tests. That model still exists, but it no longer "
            "fully reflects how young people learn today."
        ),
        start_char=0,
        end_char=178,
        paragraph_count=1,
        word_count=27,
        metadata={"before_context": "", "after_context": "Now, students are surrounded by information."},
    )
    route_plan = {
        **_sample_route_plan(),
        "content_profile": "broad_explanatory_report",
        "cluster_role": "background_context",
        "dominant_failure_pattern": "claim_chain",
        "route_strategy": "group_and_bridge",
        "must_preserve": [
            {"source_quote": "Students received knowledge from trusted sources", "preserve_as": "old learning source"}
        ],
        "length_target": "same_length",
    }
    density_cluster = {
        "start_sentence": 2,
        "end_sentence": 3,
        "sentence_count": 2,
        "word_count": 27,
        "preview": "Students received knowledge from trusted sources...",
        "generic_hits": ["trusted sources"],
        "transition_count": 1,
    }

    prompt = build_direct_scanner_leapfrog_prompt(
        section=section,
        density_cluster=density_cluster,
        route_plan=route_plan,
        variant_count=5,
        batch_index=2,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "direct_scanner_cluster_leapfrog"
    assert payload["scanner_focus"]["source"] == "eligible_span_density.top_unsafe_clusters"
    assert payload["execution_brief"] == route_plan
    assert payload["writer_execution_card"]["primary_metric"] == "topk_density"
    assert payload["writer_execution_card"]["operator_execution_notes"][0]["operator"] == "CLAUSE_ROUTE_CHANGE"
    assert len(payload["writer_variant_plan"]) == 5
    assert payload["writer_variant_plan"][1]["route_shape"] != payload["writer_variant_plan"][0]["route_shape"]
    assert payload["length_guidance"]["small_variant_allowed"] is True
    assert payload["length_guidance"]["preferred_min_words"] < section.word_count
    assert payload["retry_batch"]["batch_index"] == 2
    assert len(payload["output_schema"]["variants"]) == 5
    lowered = prompt.casefold()
    assert "fake-human" in lowered
    assert "return the whole document" in lowered
    assert "controlled specificity" in lowered
    assert "source-near" not in lowered
    assert "do not upgrade factual wording into encyclopedia-style substitutes" not in lowered


def test_v5_balanced_ai_topk_selector_rejects_external_only_movement():
    external_only = {
        "apply_status": {"applied": True},
        "incremental": {
            "ai_delta": 0.1,
            "topk_delta": 0.0,
            "topk_calibrated_risk_delta": 0.0,
            "external_delta": 20.0,
            "risky_window_count_delta": 0.0,
            "unsafe_word_ratio_delta": 0.0,
        },
    }
    balanced = {
        "apply_status": {"applied": True},
        "incremental": {
            "ai_delta": 0.8,
            "topk_delta": 1.1,
            "topk_calibrated_risk_delta": 2.0,
            "external_delta": 2.0,
            "risky_window_count_delta": 0.0,
            "unsafe_word_ratio_delta": 1.0,
        },
    }

    assert not _has_balanced_ai_topk_movement(external_only)
    assert _has_balanced_ai_topk_movement(balanced)
    assert _best_balanced_ai_topk_candidate([external_only, balanced]) is balanced
    assert _balanced_ai_topk_sort_value(balanced) > _balanced_ai_topk_sort_value(external_only)


def test_v5_direct_scanner_adaptive_batch_policy_continues_only_for_weak_candidates(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_BATCH_POLICY", raising=False)
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_RUN_ALL_BATCHES", raising=False)
    weak = {
        "apply_status": {"applied": True},
        "incremental": {
            "ai_delta": 0.5,
            "topk_delta": 0.0,
            "topk_calibrated_risk_delta": 0.0,
            "external_delta": 2.0,
        },
    }
    strong = {
        "apply_status": {"applied": True},
        "incremental": {
            "ai_delta": 1.2,
            "topk_delta": 1.1,
            "topk_calibrated_risk_delta": 2.8,
            "external_delta": 4.0,
        },
    }

    assert _direct_scanner_batch_policy() == "adaptive"
    assert not _direct_scanner_candidate_strong_enough(weak)
    assert _direct_scanner_candidate_strong_enough(strong)
    assert _should_continue_direct_scanner_batches(
        weak,
        batch_policy="adaptive",
        batch_index=1,
        max_batches=2,
    )
    assert not _should_continue_direct_scanner_batches(
        strong,
        batch_policy="adaptive",
        batch_index=1,
        max_batches=2,
    )


def test_v5_residual_candidate_sort_prefers_cleared_local_cluster():
    uncleared = {
        "local_scores": {
            "unsafe_cluster_count": 1,
            "unsafe_word_ratio": 14.0,
            "unsafe_cluster_count_delta": 0.0,
            "topk_calibrated_risk_delta": 60.0,
            "unsafe_word_ratio_delta": 80.0,
        },
        "incremental": {"rank_delta": 8.0, "ai_delta": 4.0, "unsafe_cluster_count_delta": 0.0},
        "scores": {"rank_delta": 8.0},
    }
    cleared = {
        "local_scores": {
            "unsafe_cluster_count": 0,
            "unsafe_word_ratio": 0.0,
            "topk_calibrated_risk": 20.0,
            "unsafe_cluster_count_delta": 1.0,
            "topk_calibrated_risk_delta": 40.0,
            "unsafe_word_ratio_delta": 100.0,
        },
        "incremental": {"rank_delta": 4.0, "ai_delta": 2.0, "unsafe_cluster_count_delta": 1.0},
        "scores": {"rank_delta": 4.0},
    }

    assert _residual_candidate_sort_key(cleared) > _residual_candidate_sort_key(uncleared)


def test_v5_residual_acceptance_requires_local_cluster_clearance():
    uncleared = {
        "local_scores": {"unsafe_cluster_count": 1, "unsafe_word_ratio": 10.0},
        "incremental": {"rank_delta": 12.0, "ai_delta": 4.0},
    }
    cleared = {
        "local_scores": {"unsafe_cluster_count": 0, "unsafe_word_ratio": 0.0, "topk_calibrated_risk": 30.0},
        "incremental": {"rank_delta": 0.2, "ai_delta": 0.0},
    }

    assert not _has_incremental_movement(uncleared)
    assert _has_incremental_movement(cleared)


def test_v5_residual_acceptance_preserves_directional_local_improvement():
    partial = {
        "local_scores": {
            "unsafe_cluster_count": 1,
            "unsafe_word_ratio": 40.0,
            "unsafe_cluster_count_delta": 0.0,
            "unsafe_word_ratio_delta": 25.0,
            "topk_delta": 12.0,
            "rank_delta": 30.0,
        },
        "incremental": {"rank_delta": 0.6, "ai_delta": 0.2},
    }
    local_worse = {
        "local_scores": {
            "unsafe_cluster_count": 1,
            "unsafe_word_ratio": 80.0,
            "unsafe_cluster_count_delta": -1.0,
            "unsafe_word_ratio_delta": 20.0,
            "topk_delta": 12.0,
            "rank_delta": 30.0,
        },
        "incremental": {"rank_delta": 4.0, "ai_delta": 2.0},
    }

    assert _has_incremental_movement(partial)
    assert not _has_incremental_movement(local_worse)


def test_v5_residual_acceptance_keeps_local_cluster_count_drop_without_topk_delta():
    cluster_count_drop = {
        "local_scores": {
            "unsafe_cluster_count": 1,
            "unsafe_word_ratio": 47.0,
            "unsafe_cluster_count_delta": 1.0,
            "unsafe_word_ratio_delta": 25.0,
            "topk_delta": 0.0,
            "rank_delta": 10.0,
        },
        "incremental": {
            "unsafe_cluster_count_delta": 1.0,
            "rank_delta": 0.8,
            "ai_delta": 0.0,
        },
    }

    assert _has_incremental_movement(cluster_count_drop)


def test_v5_residual_seed_generator_stays_disabled_for_content_agnostic_output():
    cluster = type("Cluster", (), {
        "cluster_id": "cluster_001",
        "text": (
            "At the beginning of the course, he required a support worker to accompany him to class "
            "and barely interacted with others. "
            "At the beginning, he was quite reserved, but as we got to know each other through casual "
            "conversation, I learned about some of his past learning experiences. "
            "With my patient guidance and role-playing activities, he gradually became more confident. "
            "During one role-playing activity, I had Johnny take on the role of hair salon manager in "
            "a group project. "
            "He successfully led the group and received positive feedback from his teammates."
        ),
        "start_char": 5,
        "end_char": 90,
        "risk_score": 10.0,
        "sentence_count": 5,
        "metadata": {},
    })()
    section = _section_from_cluster(cluster)

    variants = generate_residual_cluster_seed_variants(section=section)

    assert variants
    assert variants[0].variant_id == "route_seed_1"
    assert "That starting point mattered before the next step." in variants[0].text


def test_v5_residual_seed_generator_skips_unmatched_clusters():
    cluster = type("Cluster", (), {
        "cluster_id": "cluster_002",
        "text": "Students compare techniques and reflect on their progress after practice.",
        "start_char": 5,
        "end_char": 80,
        "risk_score": 9.0,
        "sentence_count": 1,
        "metadata": {},
    })()
    section = _section_from_cluster(cluster)

    assert generate_residual_cluster_seed_variants(section=section) == []


def test_v5_residual_prompt_preserves_pronoun_subject_continuity():
    cluster = type("Cluster", (), {
        "cluster_id": "cluster_003",
        "text": (
            "He is currently working part-time in the hospitality industry. "
            "This example shows that inclusive education can build confidence."
        ),
        "start_char": 5,
        "end_char": 120,
        "risk_score": 9.0,
        "sentence_count": 2,
        "metadata": {},
        "before_context": "After that, Johnny has become more confident.",
        "after_context": "Through role-playing exercises, the class became more accepting.",
    })()
    section = _section_from_cluster(cluster)

    prompt = build_residual_cluster_prompt(section=section, local_goal={}, variant_count=1)
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    continuity = payload["cluster"]["referential_continuity"]
    assert payload["cluster"]["before_context"].startswith("After that, Johnny")
    assert "Johnny" in continuity["named_references_in_cluster"]
    assert continuity["opening_subject"] == "He"
    assert continuity["preferred_reference"] == "Johnny"
    assert continuity["preserve_opening_subject"] is True
    assert "do not generalize" in continuity["instruction"]
    assert "do not explain the reference parenthetically" in continuity["instruction"]
    assert any("referring to" in item for item in payload["method"])


def test_v5_risky_window_acceptance_prioritizes_window_removal_over_rank():
    candidate = {
        "incremental": {
            "risky_window_count_delta": 1.0,
            "rank_delta": -0.4,
            "unsafe_cluster_count_delta": 0.0,
            "topk_calibrated_risk_delta": 0.0,
            "ai_delta": 0.0,
        }
    }
    topk_regression = {
        "incremental": {
            "risky_window_count_delta": 1.0,
            "rank_delta": 4.0,
            "unsafe_cluster_count_delta": 0.0,
            "topk_calibrated_risk_delta": -0.1,
            "ai_delta": 0.0,
        }
    }

    assert _has_risky_window_cleanup_movement(candidate)
    assert not _has_risky_window_cleanup_movement(topk_regression)


def test_v5_unsafe_cluster_acceptance_prioritizes_cluster_movement_over_rank_external():
    cluster_drop = {
        "incremental": {
            "unsafe_cluster_count_delta": 1.0,
            "rank_delta": -2.0,
            "external_delta": -1.0,
            "external_ai_flag_risk_delta": -1.0,
            "risky_window_count_delta": 0.0,
            "topk_calibrated_risk_delta": 0.0,
            "ai_delta": 0.0,
        },
        "local_scores": {},
    }
    ai_regression = {
        "incremental": {
            "unsafe_cluster_count_delta": 1.0,
            "rank_delta": 4.0,
            "external_delta": 4.0,
            "risky_window_count_delta": 0.0,
            "topk_calibrated_risk_delta": 0.0,
            "ai_delta": -0.1,
        },
        "local_scores": {},
    }

    assert _has_unsafe_cluster_cleanup_movement(cluster_drop)
    assert not _has_unsafe_cluster_cleanup_movement(ai_regression)


def test_v5_unsafe_cluster_later_round_rejects_tiny_local_only_gain(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_MIN_GAIN_START_ROUND", "2")
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_MIN_AI_DELTA", "0.5")
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_MIN_TOPK_DELTA", "0.5")
    tiny_local_only = {
        "incremental": {
            "unsafe_cluster_count_delta": 0.0,
            "risky_window_count_delta": 0.0,
            "topk_calibrated_risk_delta": 0.0,
            "ai_delta": 0.2,
            "topk_delta": 0.4,
        },
        "local_scores": {
            "unsafe_cluster_count_delta": 0.0,
            "unsafe_word_ratio_delta": 6.0,
            "topk_delta": 2.0,
            "rank_delta": 1.0,
        },
    }
    enough_ai = {
        **tiny_local_only,
        "incremental": {**tiny_local_only["incremental"], "ai_delta": 0.5},
    }
    enough_topk = {
        **tiny_local_only,
        "incremental": {**tiny_local_only["incremental"], "topk_delta": 0.5},
    }
    cluster_drop = {
        **tiny_local_only,
        "incremental": {**tiny_local_only["incremental"], "unsafe_cluster_count_delta": 1.0},
    }

    assert _has_unsafe_cluster_cleanup_movement(tiny_local_only, cleanup_index=1)
    assert not _has_unsafe_cluster_cleanup_movement(tiny_local_only, cleanup_index=2)
    assert _has_unsafe_cluster_cleanup_movement(enough_ai, cleanup_index=2)
    assert _has_unsafe_cluster_cleanup_movement(enough_topk, cleanup_index=2)
    assert _has_unsafe_cluster_cleanup_movement(cluster_drop, cleanup_index=2)


def test_v5_unsafe_cluster_cleanup_stops_after_consecutive_misses(tmp_path, monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_STOP_AFTER_MISSES", "3")
    assert _unsafe_cluster_cleanup_stop_after_misses() == 3
    text = "This paragraph still has a generic unsafe cluster."
    section = SectionUnit(
        section_id="density-1",
        heading="",
        text=text,
        start_char=0,
        end_char=len(text),
        paragraph_count=1,
        word_count=8,
        metadata={},
    )
    generated_rounds: list[int] = []

    monkeypatch.setattr(
        v5_residual_comb,
        "_density_gate_for_report",
        lambda _text, _report: {"safe": False, "unsafe_cluster_count": 1},
    )
    monkeypatch.setattr(
        v5_residual_comb,
        "_select_density_cluster_section",
        lambda *_args, **_kwargs: (section, {"cluster_id": "density-1"}, ("density-1",)),
    )
    monkeypatch.setattr(
        v5_residual_comb,
        "generate_residual_cluster_route_plan",
        lambda **_kwargs: ({"status": "test_plan"}, {"status": "ok"}, "plan prompt", "plan completion"),
    )

    def fake_generate_variants(**_kwargs):
        generated_rounds.append(len(generated_rounds) + 1)
        return (
            [RecompositionVariant(variant_id="v1", text=text, word_count=8)],
            {"status": "ok"},
            "prompt",
            "completion",
        )

    def fake_score_variant(**_kwargs):
        return {
            "variant_id": "v1",
            "text": text,
            "candidate_text": text,
            "apply_status": {"applied": True},
            "scores": {"rank_delta": 0.0},
            "incremental": {
                "unsafe_cluster_count_delta": 0.0,
                "risky_window_count_delta": 0.0,
                "topk_calibrated_risk_delta": 0.0,
                "ai_delta": 0.0,
                "topk_delta": 0.0,
            },
            "local_scores": {
                "unsafe_cluster_count_delta": 0.0,
                "unsafe_word_ratio_delta": 0.0,
                "topk_delta": 0.0,
                "rank_delta": 0.0,
            },
        }

    monkeypatch.setattr(v5_residual_comb, "generate_unsafe_cluster_cleanup_variants", fake_generate_variants)
    monkeypatch.setattr(v5_residual_comb, "_score_residual_variant", fake_score_variant)

    *_state, rounds, _best = _run_unsafe_cluster_cleanup_pass(
        original_text=text,
        baseline_report={},
        baseline_scores={},
        current_text=text,
        current_report={},
        current_goal={},
        current_scores={"ai": 50.0},
        gateway=SimpleNamespace(),
        planner_gateway=SimpleNamespace(),
        output_dir=tmp_path,
        global_best_candidate=None,
        max_rounds=8,
        variant_count=1,
    )

    assert generated_rounds == [1, 2, 3]
    assert [row["reason"] for row in rounds] == [
        "no_unsafe_cluster_movement",
        "no_unsafe_cluster_movement",
        "no_unsafe_cluster_movement",
        "unsafe_cluster_miss_limit_reached",
    ]
    assert rounds[-1]["consecutive_no_unsafe_cluster_movement"] == 3


def test_v5_risky_window_cleanup_prompt_targets_whole_window_without_detector_language():
    section = build_section_units(
        "Johnny built confidence through role playing. "
        "The proverb should connect to that practical outcome.",
        {},
    )[0]

    prompt = build_risky_window_cleanup_prompt(
        section=section,
        current_scores={"risky_window_count": 2, "unsafe_cluster_count": 7},
        variant_count=3,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "residual_route_window_cleanup"
    assert payload["window"]["source_window"] == section.text
    assert payload["length_guidance"]["preferred_min_words"] <= section.word_count
    assert len(payload["output_schema"]["variants"]) == 3
    assert any("Replace the full source_window only" in item for item in payload["required_moves"])
    lowered = prompt.casefold()
    assert "ai detector" not in lowered
    assert "bypass" not in lowered
    assert "authorship" not in lowered


def test_v5_risky_window_cleanup_prompt_can_use_route_plan_brief():
    section = build_section_units(
        "The service changed the student's confidence. "
        "The thank-you card made the result visible.",
        {},
    )[0]
    route_plan = _sample_route_plan()

    prompt = build_risky_window_cleanup_prompt(
        section=section,
        current_scores={"risky_window_count": 2, "unsafe_cluster_count": 7},
        variant_count=1,
        route_plan=route_plan,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "residual_route_window_cleanup"
    assert payload["execution_brief"] == route_plan
    assert payload["writer_execution_card"]["main_operator"] == "CLAUSE_ROUTE_CHANGE"
    assert payload["writer_variant_plan"][0]["execution_rule"]
    assert payload["source_blocks"]
    assert payload["coverage_guidance"]["requirements"]
    assert any("Follow execution_brief.replacement_route" in item for item in payload["method"])
    assert payload["length_guidance"]["preferred_max_words"] == round(section.word_count * 1.10)


def test_v5_unsafe_cluster_cleanup_prompt_is_local_without_old_source_near_guard():
    section = build_section_units(
        "The client gave him a card after the event. "
        "That made the learning visible.",
        {},
    )[0]

    prompt = build_unsafe_cluster_cleanup_prompt(
        section=section,
        density_cluster={
            "sentence_count": 2,
            "word_count": section.word_count,
            "preview": "The client gave him a card after the event.",
            "generic_hits": 0,
            "transition_count": 0,
        },
        variant_count=2,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "single_density_cluster_cleanup"
    assert payload["cluster"]["source_cluster"] == section.text
    assert payload["editorial_findings"]["preview"].startswith("The client")
    assert payload["length_guidance"]["preferred_max_words"] >= section.word_count
    assert len(payload["output_schema"]["variants"]) == 2
    lowered = prompt.casefold()
    assert "ai detector" not in lowered
    assert "bypass" not in lowered
    assert "source-near" not in lowered


def test_v5_unsafe_cluster_cleanup_prompt_can_use_route_plan_brief():
    section = build_section_units(
        "The service changed the student's confidence. "
        "The thank-you card made the result visible.",
        {},
    )[0]
    route_plan = _sample_route_plan()

    prompt = build_unsafe_cluster_cleanup_prompt(
        section=section,
        density_cluster={
            "sentence_count": 2,
            "word_count": section.word_count,
            "preview": "The service changed the student's confidence.",
            "generic_hits": 0,
            "transition_count": 0,
        },
        variant_count=1,
        route_plan=route_plan,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "single_density_cluster_cleanup"
    assert payload["execution_brief"] == route_plan
    assert payload["writer_execution_card"]["route_to_break"] == "service changed -> confidence"
    assert payload["writer_variant_plan"][0]["must_differ_from_other_variants"]
    assert payload["source_blocks"]
    assert payload["coverage_guidance"]["requirements"]
    assert any("Follow execution_brief.replacement_route" in item for item in payload["method"])
    assert payload["length_guidance"]["preferred_max_words"] == round(section.word_count * 1.10)


def test_v5_parallel_variant_fanout_uses_one_variant_prompts(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_PARALLEL_VARIANTS", "true")
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_LLM_FANOUT", "2")
    calls: list[dict] = []

    def prompt_builder(count: int) -> str:
        payload = {
            "task": "test_parallel_generation",
            "source_word_count": 70,
            "writer_variant_plan": [
                {
                    "variant_id": f"v{index}",
                    "main_operator": "CLAUSE_ROUTE_CHANGE",
                    "route_shape": f"shape{index}",
                    "execution_rule": f"execute shape {index}",
                    "must_differ_from_other_variants": "use a distinct route",
                }
                for index in range(1, count + 1)
            ],
            "output_schema": {
                "variants": [
                    {"variant_id": f"v{index}", "text": "..."}
                    for index in range(1, count + 1)
                ]
            },
        }
        return "Return valid JSON only.\n" + json.dumps(payload)

    class FakeGateway:
        model = "deepseek/deepseek-v3.2"
        provider = None

        def chat(self, prompt: str, **kwargs):
            payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))
            calls.append({"prompt": payload, "kwargs": kwargs})
            lane = payload["parallel_generation_lane"]["lane"]
            assert len(payload["output_schema"]["variants"]) == 1
            assert len(payload["writer_variant_plan"]) == 1
            assert payload["assigned_writer_variant"]["route_shape"] == f"shape{lane}"
            return _FakeV5LLMResponse(json.dumps({
                "variants": [
                    {
                        "variant_id": "v1",
                        "text": (
                            f"This replacement keeps the source claim and changes "
                            f"the sentence route for lane {lane}."
                        ),
                    }
                ]
            }))

    variants, diagnostics, prompt_log, completion_log = _generate_loose_variants_from_builder(
        prompt_builder=prompt_builder,
        gateway=FakeGateway(),
        variant_count=3,
    )

    assert [variant.variant_id for variant in variants] == ["v1", "v2", "v3"]
    assert diagnostics["parallel_variant_generation"] is True
    assert diagnostics["parallel_call_count"] == 3
    assert diagnostics["parallel_worker_limit"] == 2
    assert len(calls) == 3
    assert [
        call["prompt"]["assigned_writer_variant"]["route_shape"]
        for call in calls
    ] == ["shape1", "shape2", "shape3"]
    assert all(call["kwargs"]["max_tokens"] == 1460 for call in calls)
    assert json.loads(prompt_log)["parallel_variant_generation"] is True
    assert json.loads(completion_log)["parallel_variant_generation"] is True


def test_v5_parallel_variant_fanout_defaults_to_parallel(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_PARALLEL_VARIANTS", raising=False)
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_LLM_FANOUT", "2")
    calls: list[dict] = []

    def prompt_builder(count: int) -> str:
        payload = {
            "task": "test_serial_generation",
            "source_word_count": 70,
            "output_schema": {
                "variants": [
                    {"variant_id": f"v{index}", "text": "..."}
                    for index in range(1, count + 1)
                ]
            },
        }
        return "Return valid JSON only.\n" + json.dumps(payload)

    class FakeGateway:
        model = "deepseek/deepseek-v3.2"
        provider = None

        def chat(self, prompt: str, **kwargs):
            payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))
            calls.append(payload)
            assert len(payload["output_schema"]["variants"]) == 1
            return _FakeV5LLMResponse(json.dumps({
                "variants": [
                    {
                        "variant_id": "v1",
                        "text": "This serial replacement keeps the same source claim and route.",
                    }
                ]
            }))

    variants, diagnostics, _, _ = _generate_loose_variants_from_builder(
        prompt_builder=prompt_builder,
        gateway=FakeGateway(),
        variant_count=3,
    )

    assert len(calls) == 3
    assert len(variants) == 3
    assert diagnostics["parallel_variant_generation"] is True
    assert diagnostics["parallel_worker_limit"] == 2


def test_v5_parallel_variant_fanout_can_be_disabled(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_PARALLEL_VARIANTS", "false")
    calls: list[dict] = []

    def prompt_builder(count: int) -> str:
        payload = {
            "task": "test_serial_generation",
            "source_word_count": 70,
            "output_schema": {
                "variants": [
                    {"variant_id": f"v{index}", "text": "..."}
                    for index in range(1, count + 1)
                ]
            },
        }
        return "Return valid JSON only.\n" + json.dumps(payload)

    class FakeGateway:
        model = "deepseek/deepseek-v3.2"
        provider = None

        def chat(self, prompt: str, **kwargs):
            payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))
            calls.append(payload)
            return _FakeV5LLMResponse(json.dumps({
                "variants": [
                    {
                        "variant_id": "v1",
                        "text": "This serial replacement keeps the same source claim and route.",
                    }
                ]
            }))

    variants, diagnostics, _, _ = _generate_loose_variants_from_builder(
        prompt_builder=prompt_builder,
        gateway=FakeGateway(),
        variant_count=3,
    )

    assert len(calls) == 1
    assert len(calls[0]["output_schema"]["variants"]) == 3
    assert len(variants) == 1
    assert "parallel_variant_generation" not in diagnostics


def test_v5_parallel_single_variant_token_cap_scales_with_source_words():
    prompt = "Return valid JSON only.\n" + json.dumps({
        "cluster": {"source_word_count": 315},
        "length_guidance": {"preferred_max_words": 360},
    })

    assert _parallel_single_variant_max_tokens(prompt, 8000) == 3780
    assert _parallel_single_variant_max_tokens(prompt, 2500) == 2500


def test_v5_serial_variant_token_cap_scales_with_source_words(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_SERIAL_MAX_TOKENS", raising=False)
    prompt = "Return valid JSON only.\n" + json.dumps({
        "cluster": {"source_word_count": 315},
        "length_guidance": {"preferred_max_words": 360},
    })

    assert _serial_variant_max_tokens(prompt, 5) == 4950
    assert _serial_variant_max_tokens(prompt, 2) == 2520


def test_v5_runtime_budget_stop_record_marks_exhaustion():
    started_at = 100.0

    assert _runtime_budget_exhausted(started_at, 1.0)
    record = _runtime_budget_stop_record(
        phase="unsafe_cluster_cleanup",
        round_index=3,
        started_at=started_at,
        max_seconds=1.0,
        current_scores={"ai": 35.0},
    )

    assert record["status"] == "stopped"
    assert record["reason"] == "runtime_budget_exhausted"
    assert record["runtime_budget"]["enabled"] is True
    assert record["current_scores"]["ai"] == 35.0


def test_v5_production_runtime_budget_scales_with_input_length(monkeypatch):
    monkeypatch.delenv("REWRITE_SOFT_TIME_LIMIT_SECONDS", raising=False)
    config = {
        "runtime_base_seconds": 120,
        "runtime_seconds_per_100_words": 25.0,
        "runtime_min_seconds": 180,
        "runtime_max_seconds": 720,
        "runtime_soft_limit_buffer_seconds": 120,
    }
    short_text = "word " * 100
    long_text = "word " * 1500

    assert v5_production._v5_runtime_budget_seconds(short_text, config) == 180
    assert v5_production._v5_runtime_budget_seconds(long_text, config) == 495


def test_v5_provider_routing_defaults_to_throughput(monkeypatch):
    for name in (
        "DRAFTPROOF_REWRITE_V5_PROVIDER_ROUTING_JSON",
        "DRAFTPROOF_OPENROUTER_PROVIDER_ROUTING_JSON",
        "OPENROUTER_PROVIDER_ROUTING_JSON",
        "LLM_PROVIDER_ROUTING_JSON",
        "DRAFTPROOF_REWRITE_V5_PROVIDER_SORT",
        "DRAFTPROOF_OPENROUTER_PROVIDER_SORT",
        "OPENROUTER_PROVIDER_SORT",
        "DRAFTPROOF_REWRITE_V5_ALLOW_FALLBACKS",
        "DRAFTPROOF_OPENROUTER_ALLOW_FALLBACKS",
        "OPENROUTER_ALLOW_FALLBACKS",
        "DRAFTPROOF_REWRITE_V5_PROVIDER_ORDER",
        "DRAFTPROOF_OPENROUTER_PROVIDER_ORDER",
        "OPENROUTER_PROVIDER_ORDER",
        "DRAFTPROOF_REWRITE_V5_PROVIDER_ONLY",
        "DRAFTPROOF_OPENROUTER_PROVIDER_ONLY",
        "OPENROUTER_PROVIDER_ONLY",
        "DRAFTPROOF_REWRITE_V5_PROVIDER_IGNORE",
        "DRAFTPROOF_OPENROUTER_PROVIDER_IGNORE",
        "OPENROUTER_PROVIDER_IGNORE",
    ):
        monkeypatch.delenv(name, raising=False)

    assert v5_production._v5_provider_routing() == {
        "allow_fallbacks": True,
        "sort": "throughput",
    }


def test_v5_provider_routing_honors_v5_env_overrides(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_PROVIDER_SORT", "latency")
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_PROVIDER_ORDER", "friendli,together")
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_PROVIDER_IGNORE", "slow-provider")
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_ALLOW_FALLBACKS", "false")

    assert v5_production._v5_provider_routing() == {
        "allow_fallbacks": False,
        "sort": "latency",
        "order": ["friendli", "together"],
        "ignore": ["slow-provider"],
    }


def test_v5_provider_routing_accepts_json_override(monkeypatch):
    monkeypatch.setenv(
        "DRAFTPROOF_REWRITE_V5_PROVIDER_ROUTING_JSON",
        json.dumps({"order": ["friendli"], "allow_fallbacks": False}),
    )
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_PROVIDER_SORT", "throughput")

    assert v5_production._v5_provider_routing() == {
        "order": ["friendli"],
        "allow_fallbacks": False,
    }


def test_v5_compact_rounds_keep_cleanup_observability_without_raw_payloads():
    rounds = [
        {
            "round": 1,
            "phase": "unsafe_cluster_cleanup",
            "status": "skipped",
            "reason": "no_unsafe_cluster_movement",
            "density_gate": {
                "safe": False,
                "unsafe_cluster_count": 10,
                "top_unsafe_clusters": [{"preview": "large raw payload should be dropped"}],
            },
            "generator_diagnostics": {
                "route_plan": {
                    "status": "ok",
                    "content_profile": "broad_explanatory_report",
                    "cluster_role": "mixed_section",
                    "dominant_failure_pattern": "category_dump",
                    "route_strategy": "group_and_bridge",
                    "source_block_plan_count": 3,
                    "target_sentence_job_count": 5,
                    "length_target": "same_length",
                },
                "llm_generation": {
                    "status": "ok",
                    "valid_variant_count": 5,
                    "variant_count": 5,
                    "structured_output_mode": "json_schema",
                },
            },
            "selected": {
                "section_id": "density_cluster_001",
                "variant_id": "v1",
                "label": "density_v1",
                "word_count": 80,
                "scores": {"ai_delta": 1.0, "rank_delta": 2.0},
                "incremental": {"unsafe_cluster_count_delta": 0.0, "rank_delta": 2.0},
                "local_scores": {"unsafe_cluster_count": 1, "topk_delta": 4.0},
                "text": "candidate",
            },
        }
    ]

    compact = v5_production._compact_v5_rounds(rounds)

    assert compact[0]["phase"] == "unsafe_cluster_cleanup"
    assert compact[0]["density_gate"]["unsafe_cluster_count"] == 10
    assert "top_unsafe_clusters" not in compact[0]["density_gate"]
    assert compact[0]["generator_diagnostics"]["route_plan"]["content_profile"] == "broad_explanatory_report"
    assert compact[0]["selected"]["incremental"]["rank_delta"] == 2.0


def test_v5_tail_cleanup_acceptance_is_scanner_owned():
    risky_window_drop = {
        "incremental": {
            "risky_window_count_delta": 1.0,
            "rank_delta": 0.5,
            "unsafe_cluster_count_delta": 0.0,
            "topk_calibrated_risk_delta": 0.0,
            "ai_delta": 0.0,
        },
    }
    risky_window_no_drop = {
        "incremental": {
            "risky_window_count_delta": 0.0,
            "rank_delta": 5.0,
            "unsafe_cluster_count_delta": 0.0,
            "topk_calibrated_risk_delta": 0.0,
            "ai_delta": 0.0,
        },
    }
    risky_window_cluster_regression = {
        "incremental": {
            "risky_window_count_delta": 1.0,
            "rank_delta": 1.0,
            "unsafe_cluster_count_delta": -1.0,
            "topk_calibrated_risk_delta": 0.0,
            "ai_delta": 0.0,
        },
    }
    cluster_drop = {
        "incremental": {
            "unsafe_cluster_count_delta": 1.0,
            "rank_delta": 0.4,
            "risky_window_count_delta": 0.0,
            "topk_calibrated_risk_delta": 0.0,
            "ai_delta": 0.0,
        },
    }
    cluster_rank_regression = {
        "incremental": {
            "unsafe_cluster_count_delta": 1.0,
            "rank_delta": -0.5,
            "risky_window_count_delta": 0.0,
            "topk_calibrated_risk_delta": 0.0,
            "ai_delta": 0.0,
        },
    }
    cluster_risky_window_regression = {
        "incremental": {
            "unsafe_cluster_count_delta": 1.0,
            "rank_delta": 0.5,
            "risky_window_count_delta": -1.0,
            "topk_calibrated_risk_delta": 0.0,
            "ai_delta": 0.0,
        },
    }
    cluster_external_regression = {
        "incremental": {
            "unsafe_cluster_count_delta": 1.0,
            "rank_delta": 0.5,
            "external_delta": -0.1,
            "risky_window_count_delta": 0.0,
            "topk_calibrated_risk_delta": 0.0,
            "ai_delta": 0.0,
        },
    }
    cluster_local_directional = {
        "incremental": {
            "unsafe_cluster_count_delta": 0.0,
            "rank_delta": 0.5,
            "external_delta": 0.2,
            "risky_window_count_delta": 0.0,
            "topk_calibrated_risk_delta": 0.0,
            "ai_delta": 0.1,
        },
        "local_scores": {
            "unsafe_cluster_count_delta": 0.0,
            "unsafe_word_ratio_delta": 12.0,
            "topk_delta": 3.0,
            "rank_delta": 2.0,
        },
    }

    assert _has_risky_window_cleanup_movement(risky_window_drop)
    assert not _has_risky_window_cleanup_movement(risky_window_no_drop)
    assert not _has_risky_window_cleanup_movement(risky_window_cluster_regression)
    assert _has_unsafe_cluster_cleanup_movement(cluster_drop)
    assert _has_unsafe_cluster_cleanup_movement(cluster_local_directional)
    assert _has_unsafe_cluster_cleanup_movement(cluster_rank_regression)
    assert not _has_unsafe_cluster_cleanup_movement(cluster_risky_window_regression)
    assert _has_unsafe_cluster_cleanup_movement(cluster_external_regression)


def test_v5_global_best_fallback_keeps_better_full_document_candidate():
    current_scores = {
        "ai_delta": 0.83,
        "rank_delta": 2.131,
        "topk_calibrated_risk_delta": 1.037,
        "topk_delta": 0.38,
        "external_ai_flag_risk_delta": 2.904,
    }
    phase_accepted = {
        "apply_status": {"applied": True},
        "scores": current_scores,
    }
    stage_skipped_but_better = {
        "apply_status": {"applied": True},
        "scores": {
            "ai_delta": 3.15,
            "rank_delta": 2.982,
            "topk_calibrated_risk_delta": 2.21,
            "topk_delta": 0.81,
            "external_ai_flag_risk_delta": 3.967,
            "unsafe_cluster_count_delta": 1.0,
        },
    }
    narrow_counter_only = {
        "apply_status": {"applied": True},
        "scores": {
            "ai_delta": -0.09,
            "rank_delta": 2.3,
            "topk_calibrated_risk_delta": 1.9,
            "external_ai_flag_risk_delta": 2.5,
            "unsafe_cluster_count_delta": 1.0,
        },
    }

    best = _best_full_document_candidate([phase_accepted, stage_skipped_but_better, narrow_counter_only])

    assert not _has_full_document_fallback_movement(narrow_counter_only)
    assert best is stage_skipped_but_better
    assert _full_document_candidate_beats_scores(stage_skipped_but_better, current_scores)
    assert not _full_document_candidate_beats_scores(phase_accepted, current_scores)


def test_v5_global_best_fallback_rejects_density_regression():
    candidate = {
        "apply_status": {"applied": True},
        "scores": {
            "ai_delta": 2.0,
            "rank_delta": 1.0,
            "topk_calibrated_risk_delta": 0.5,
            "external_ai_flag_risk_delta": 1.0,
            "unsafe_cluster_count_delta": -1.0,
        },
    }

    assert not _has_full_document_fallback_movement(candidate)


def test_v5_phase_order_starts_with_unsafe_cluster_cleanup_when_density_is_unsafe():
    assert _should_start_with_unsafe_cluster_cleanup(
        density_gate={"safe": False},
        unsafe_cluster_cleanup_rounds=12,
    )
    assert not _should_start_with_unsafe_cluster_cleanup(
        density_gate={"safe": True},
        unsafe_cluster_cleanup_rounds=12,
    )
    assert not _should_start_with_unsafe_cluster_cleanup(
        density_gate={"safe": False},
        unsafe_cluster_cleanup_rounds=0,
    )


def test_v5_unsafe_cluster_first_uses_bounded_probe(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_PROBE_ROUNDS", raising=False)
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_PROBE_SHARE", "0.25")

    assert _unsafe_cluster_probe_round_limit(12) == 3
    assert _unsafe_cluster_probe_round_limit(4) == 1

    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_PROBE_ROUNDS", "2")
    assert _unsafe_cluster_probe_round_limit(12) == 2


def test_v5_core_route_runs_before_unsafe_cluster_probe_by_default(tmp_path, monkeypatch):
    module = v5_production.run_v5_residual_cluster_comb_experiment.__globals__
    cleanup_calls: list[dict] = []
    core_sections: list[str] = []

    def fake_scan(text):
        return {"input_text": text, "ai_score": 50.0, "findings": {}}

    def fake_goal(**_kwargs):
        return SimpleNamespace(to_dict=lambda: {"unsafe_cluster_count": 3})

    def fake_score(_text, _report, _goal):
        return {
            "ai": 50.0,
            "topk": 98.0,
            "external": 60.0,
            "rank": 120.0,
            "risky_window_count": 4,
            "unsafe_word_ratio": 70.0,
            "unsafe_cluster_count": 3,
            "topk_calibrated_risk": 96.0,
            "qualifying_text_ai_density": 60.0,
            "ai_authorship": 50.0,
            "external_ai_flag_risk": 55.0,
        }

    def fake_cleanup_pass(**kwargs):
        cleanup_calls.append({
            "max_rounds": kwargs["max_rounds"],
            "selection_mode": kwargs["selection_mode"],
            "route_plan_enabled": kwargs["route_plan_enabled"],
        })
        return (
            kwargs["current_text"],
            kwargs["current_report"],
            kwargs["current_goal"],
            kwargs["current_scores"],
            [{
                "round": 1,
                "phase": "unsafe_cluster_cleanup",
                "status": "skipped",
                "reason": "test_cleanup_probe",
            }],
            kwargs["global_best_candidate"],
        )

    def fake_cluster_units(**_kwargs):
        return [{"id": "cluster-1", "text": "The cluster needs a route rewrite."}]

    def fake_section_from_cluster(_cluster):
        section = SectionUnit(
            section_id="cluster-1",
            heading="",
            text="The cluster needs a route rewrite.",
            start_char=0,
            end_char=len("The cluster needs a route rewrite."),
            paragraph_count=1,
            word_count=6,
            metadata={},
        )
        core_sections.append(section.section_id)
        return section

    def fake_seed_variants(**_kwargs):
        return [SimpleNamespace(variant_id="seed1", text="The route rewrite worked.")]

    def fake_score_variant(**_kwargs):
        return {
            "variant_id": "seed1",
            "label": "seed_seed1",
            "text": "The route rewrite worked.",
            "candidate_text": "The route rewrite worked.",
            "candidate_report": fake_scan("The route rewrite worked."),
            "candidate_goal": {"unsafe_cluster_count": 2},
            "apply_status": {"applied": True},
            "scores": {
                "ai": 49.0,
                "topk": 97.0,
                "external": 59.0,
                "rank": 118.0,
                "risky_window_count": 4,
                "unsafe_word_ratio": 65.0,
                "unsafe_cluster_count": 2,
                "topk_calibrated_risk": 95.0,
                "qualifying_text_ai_density": 58.0,
                "ai_authorship": 49.0,
                "external_ai_flag_risk": 54.0,
                "ai_delta": 1.0,
                "topk_delta": 1.0,
                "external_delta": 1.0,
                "rank_delta": 2.0,
                "unsafe_cluster_count_delta": 1.0,
                "topk_calibrated_risk_delta": 1.0,
            },
            "local_scores": {
                "unsafe_cluster_count": 0,
                "unsafe_word_ratio": 0,
                "unsafe_cluster_count_delta": 1.0,
                "unsafe_word_ratio_delta": 5.0,
                "topk_delta": 1.0,
                "rank_delta": 2.0,
            },
            "incremental": {
                "unsafe_cluster_count_delta": 1.0,
                "rank_delta": 2.0,
                "ai_delta": 1.0,
                "topk_delta": 1.0,
                "external_delta": 1.0,
                "risky_window_count_delta": 0.0,
                "topk_calibrated_risk_delta": 1.0,
                "external_ai_flag_risk_delta": 1.0,
            },
        }

    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_PROBE_SHARE", "0.25")
    monkeypatch.setitem(module, "_scan_report", fake_scan)
    monkeypatch.setitem(module, "evaluate_rewrite_goal", fake_goal)
    monkeypatch.setitem(module, "_score_summary", fake_score)
    monkeypatch.setitem(module, "build_eligible_span_density_contract", lambda *_args, **_kwargs: {"safe": False})
    monkeypatch.setitem(module, "_run_unsafe_cluster_cleanup_pass", fake_cleanup_pass)
    monkeypatch.setitem(module, "build_cluster_repair_units", fake_cluster_units)
    monkeypatch.setitem(module, "_section_from_cluster", fake_section_from_cluster)
    monkeypatch.setitem(module, "generate_residual_cluster_seed_variants", fake_seed_variants)
    monkeypatch.setitem(module, "_score_residual_variant", fake_score_variant)

    result = run_v5_residual_cluster_comb_experiment(
        input_text="The cluster needs a route rewrite.",
        output_dir=tmp_path,
        max_rounds=1,
        variant_count=1,
        retune_variant_count=1,
        risky_window_cleanup_rounds=0,
        unsafe_cluster_cleanup_rounds=4,
        final_risky_window_cleanup_rounds=0,
        max_seconds=60,
        api_key="test-key",
    )

    assert result["phase_order"]["unsafe_cluster_first"] is False
    assert result["phase_order"]["unsafe_cluster_probe_rounds"] == 0
    assert result["phase_order"]["core_route_rounds"] == 1
    assert result["rounds"][0]["status"] == "accepted"
    assert core_sections == ["cluster-1"]
    assert cleanup_calls == [
        {"max_rounds": 4, "selection_mode": "scanner", "route_plan_enabled": True},
    ]


def test_v5_density_gate_defaults_to_legacy_contract_and_opt_in_preferred(monkeypatch):
    module = v5_production.run_v5_residual_cluster_comb_experiment.__globals__
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_USE_REPAIR_UNITS_DENSITY", raising=False)
    monkeypatch.setitem(
        module,
        "build_eligible_span_density_contract",
        lambda _text, _report: {"source": "legacy_density", "unsafe_cluster_count": 14},
    )
    monkeypatch.setitem(
        module,
        "build_preferred_eligible_span_density_contract",
        lambda _text, _report: {"source": "repair_units_density", "unsafe_cluster_count": 16},
    )

    default_gate = module["_density_gate_for_report"]("Some text.", {})
    assert default_gate["source"] == "legacy_density"
    assert default_gate["unsafe_cluster_count"] == 14

    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_USE_REPAIR_UNITS_DENSITY", "true")
    opt_in_gate = module["_density_gate_for_report"]("Some text.", {})
    assert opt_in_gate["source"] == "repair_units_density"
    assert opt_in_gate["unsafe_cluster_count"] == 16


def test_v5_goal_density_enrichment_overrides_existing_goal_density(monkeypatch):
    module = v5_production.run_v5_residual_cluster_comb_experiment.__globals__
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_USE_REPAIR_UNITS_DENSITY", raising=False)
    monkeypatch.setitem(
        module,
        "build_eligible_span_density_contract",
        lambda _text, _report: {"source": "legacy_density", "unsafe_cluster_count": 14},
    )
    monkeypatch.setitem(
        module,
        "build_preferred_eligible_span_density_contract",
        lambda _text, _report: {"source": "repair_units_density", "unsafe_cluster_count": 16},
    )

    enriched = module["_with_v5_density_gate"](
        "Some text.",
        {},
        {"eligible_span_density_gate": {"source": "stale_goal_density", "unsafe_cluster_count": 99}},
    )

    assert enriched["eligible_span_density_gate"]["source"] == "legacy_density"
    assert enriched["eligible_span_density_gate"]["unsafe_cluster_count"] == 14


def test_v5_core_round_skips_failed_cluster_and_continues_to_next_target(tmp_path, monkeypatch):
    module = v5_production.run_v5_residual_cluster_comb_experiment.__globals__
    attempted_sections: list[str] = []

    original_text = "Bad cluster needs repair. Good cluster can improve."

    def fake_scan(text):
        return {"input_text": text, "ai_score": 50.0, "findings": {}}

    def fake_goal(**_kwargs):
        return SimpleNamespace(to_dict=lambda: {"eligible_span_density_gate": {"safe": False}})

    def fake_score(_text, _report, _goal):
        return {
            "ai": 50.0,
            "topk": 90.0,
            "external": 45.0,
            "rank": 100.0,
            "risky_window_count": 2,
            "unsafe_word_ratio": 40.0,
            "unsafe_cluster_count": 2,
            "topk_calibrated_risk": 90.0,
            "qualifying_text_ai_density": 60.0,
            "ai_authorship": 50.0,
            "external_ai_flag_risk": 45.0,
        }

    def fake_cluster_units(**_kwargs):
        return [
            {"id": "bad", "text": "Bad cluster needs repair."},
            {"id": "good", "text": "Good cluster can improve."},
        ]

    def fake_section_from_cluster(cluster):
        text = cluster["text"]
        start = original_text.index(text)
        return SectionUnit(
            section_id=cluster["id"],
            heading="",
            text=text,
            start_char=start,
            end_char=start + len(text),
            paragraph_count=1,
            word_count=len(text.split()),
            metadata={},
        )

    def fake_seed_variants(section, **_kwargs):
        attempted_sections.append(section.section_id)
        return [RecompositionVariant(
            variant_id=f"{section.section_id}_seed",
            text=f"{section.text} revised",
            word_count=len(section.text.split()) + 1,
        )]

    def fake_score_variant(*, section, variant, **_kwargs):
        if section.section_id == "bad":
            return {
                "section_id": section.section_id,
                "variant_id": variant.variant_id,
                "label": "seed_bad",
                "text": variant.text,
                "candidate_text": original_text,
                "candidate_report": fake_scan(original_text),
                "candidate_goal": {"eligible_span_density_gate": {"safe": False}},
                "apply_status": {"applied": True},
                "scores": {"ai": 50.0, "topk": 90.0, "external": 45.0, "rank": 100.0},
                "local_scores": {
                    "unsafe_cluster_count": 1,
                    "unsafe_word_ratio": 40.0,
                    "unsafe_cluster_count_delta": 0.0,
                    "unsafe_word_ratio_delta": 0.0,
                    "topk_delta": 0.0,
                    "rank_delta": 0.0,
                },
                "incremental": {
                    "unsafe_cluster_count_delta": 0.0,
                    "rank_delta": 0.0,
                    "ai_delta": 0.0,
                    "topk_delta": 0.0,
                    "external_delta": 0.0,
                    "risky_window_count_delta": 0.0,
                    "topk_calibrated_risk_delta": 0.0,
                    "external_ai_flag_risk_delta": 0.0,
                },
            }
        return {
            "section_id": section.section_id,
            "variant_id": variant.variant_id,
            "label": "seed_good",
            "text": variant.text,
            "candidate_text": original_text.replace(section.text, variant.text),
            "candidate_report": fake_scan(original_text),
            "candidate_goal": {"eligible_span_density_gate": {"safe": False}},
            "apply_status": {"applied": True},
            "scores": {
                "ai": 48.0,
                "topk": 88.0,
                "external": 43.0,
                "rank": 95.0,
                "unsafe_cluster_count": 1,
            },
            "local_scores": {
                "unsafe_cluster_count": 0,
                "unsafe_word_ratio": 0.0,
                "unsafe_cluster_count_delta": 1.0,
                "unsafe_word_ratio_delta": 40.0,
                "topk_delta": 2.0,
                "rank_delta": 5.0,
            },
            "incremental": {
                "unsafe_cluster_count_delta": 1.0,
                "rank_delta": 5.0,
                "ai_delta": 2.0,
                "topk_delta": 2.0,
                "external_delta": 2.0,
                "risky_window_count_delta": 0.0,
                "topk_calibrated_risk_delta": 2.0,
                "external_ai_flag_risk_delta": 2.0,
            },
        }

    monkeypatch.setitem(module, "_scan_report", fake_scan)
    monkeypatch.setitem(module, "evaluate_rewrite_goal", fake_goal)
    monkeypatch.setitem(module, "_score_summary", fake_score)
    monkeypatch.setitem(module, "build_eligible_span_density_contract", lambda *_args, **_kwargs: {"safe": False})
    monkeypatch.setitem(module, "build_cluster_repair_units", fake_cluster_units)
    monkeypatch.setitem(module, "_section_from_cluster", fake_section_from_cluster)
    monkeypatch.setitem(module, "generate_residual_cluster_seed_variants", fake_seed_variants)
    monkeypatch.setitem(module, "generate_residual_cluster_route_plan", lambda **_kwargs: ({}, {}, "{}", "{}"))
    monkeypatch.setitem(module, "generate_residual_cluster_variants", lambda **_kwargs: ([], {}, "{}", "{}"))
    monkeypatch.setitem(module, "generate_residual_cluster_retunes", lambda **_kwargs: ([], {}, "{}", "{}"))
    monkeypatch.setitem(module, "_score_residual_variant", fake_score_variant)

    result = run_v5_residual_cluster_comb_experiment(
        input_text=original_text,
        output_dir=tmp_path,
        max_rounds=2,
        variant_count=1,
        retune_variant_count=1,
        risky_window_cleanup_rounds=0,
        unsafe_cluster_cleanup_rounds=0,
        final_risky_window_cleanup_rounds=0,
        direct_scanner_leapfrog_rounds=0,
        max_seconds=60,
        api_key="test-key",
    )

    assert attempted_sections == ["bad", "good"]
    assert [row["section"]["section_id"] for row in result["rounds"]] == ["bad", "good"]
    assert [row["reason"] for row in result["rounds"]] == [
        "no_incremental_movement",
        "accepted_incremental_movement",
    ]


def test_v5_production_defaults_protect_winning_phase_budget(monkeypatch):
    for key in (
        "DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_LEAPFROG_ROUNDS",
        "DRAFTPROOF_REWRITE_V5_RUNTIME_BASE_SECONDS",
        "DRAFTPROOF_REWRITE_V5_RUNTIME_SECONDS_PER_100_WORDS",
        "DRAFTPROOF_REWRITE_V5_RUNTIME_MIN_SECONDS",
        "DRAFTPROOF_REWRITE_V5_RUNTIME_MAX_SECONDS",
        "DRAFTPROOF_REWRITE_V5_SOFT_LIMIT_BUFFER_SECONDS",
        "REWRITE_SOFT_TIME_LIMIT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)

    config = v5_production._production_config()
    benchmark_text = "word " * 1526

    assert config["direct_scanner_leapfrog_rounds"] == 0
    assert config["runtime_min_seconds"] >= 900
    assert config["runtime_max_seconds"] >= 1800
    assert v5_production._v5_runtime_budget_seconds(benchmark_text, config) >= 1400


def test_v5_budget_order_prefers_clearable_unsafe_clusters_without_content_rules():
    clusters = [
        {"word_count": 268, "sentence_count": 18, "risk_score": 96.48},
        {"word_count": 129, "sentence_count": 8, "risk_score": 43.072},
        {"word_count": 24, "sentence_count": 2, "risk_score": 8.972},
    ]

    scanner_order = _ordered_density_cluster_rows(clusters, selection_mode="scanner")
    clearable_order = _ordered_density_cluster_rows(clusters, selection_mode="clearable")

    assert [row[0] for row in scanner_order] == [1, 2, 3]
    assert [row[0] for row in clearable_order] == [3, 2, 1]


def test_v5_global_best_fallback_does_not_discard_accepted_structural_progress():
    current_scores = {
        "ai_delta": 0.37,
        "rank_delta": 0.932,
        "unsafe_cluster_count_delta": 1.0,
        "risky_window_count_delta": 0.0,
        "topk_calibrated_risk_delta": 0.0,
        "topk_delta": 0.0,
    }
    ai_only_candidate_scores = {
        "ai_delta": 1.56,
        "rank_delta": -0.805,
        "unsafe_cluster_count_delta": 0.0,
        "risky_window_count_delta": 1.0,
        "topk_calibrated_risk_delta": 0.0,
        "topk_delta": 0.0,
    }

    assert _would_discard_structural_progress(ai_only_candidate_scores, current_scores)
    assert not _full_document_candidate_beats_scores(
        {
            "apply_status": {"applied": True},
            "scores": ai_only_candidate_scores,
        },
        current_scores,
    )


def test_v5_global_best_fallback_keeps_partial_ai_and_window_movement_despite_rank_regression():
    current_scores = {
        "ai_delta": 0.0,
        "rank_delta": 0.0,
        "topk_calibrated_risk_delta": 0.0,
        "topk_delta": 0.0,
        "external_ai_flag_risk_delta": 0.0,
        "external_delta": 0.0,
        "risky_window_count_delta": 0.0,
        "unsafe_cluster_count_delta": 0.0,
    }
    partial_candidate = {
        "apply_status": {"applied": True},
        "scores": {
            "ai_delta": 1.57,
            "rank_delta": -2.836,
            "topk_calibrated_risk_delta": 0.0,
            "topk_delta": 0.0,
            "external_ai_flag_risk_delta": 2.693,
            "external_delta": 12.394,
            "risky_window_count_delta": 1.0,
            "unsafe_cluster_count_delta": 0.0,
        },
    }
    lower_ai_candidate = {
        "apply_status": {"applied": True},
        "scores": {
            "ai_delta": 0.41,
            "rank_delta": -0.039,
            "topk_calibrated_risk_delta": 0.027,
            "topk_delta": 0.01,
            "external_ai_flag_risk_delta": 1.0,
            "external_delta": 0.464,
            "risky_window_count_delta": 0.0,
            "unsafe_cluster_count_delta": 1.0,
        },
    }

    best = _best_full_document_candidate([lower_ai_candidate, partial_candidate])

    assert _has_full_document_fallback_movement(partial_candidate)
    assert _full_document_candidate_beats_scores(partial_candidate, current_scores)
    assert best is partial_candidate


def test_v5_tail_cleanup_rejects_spliced_word_period_artifacts():
    integrity = minimal_replacement_text_integrity(
        "Teachers need to support students to become their own teachers.tie, 2009)."
    )
    duplicate = minimal_replacement_text_integrity("The result was tied to assessment.assessment.")
    ordinary = minimal_replacement_text_integrity("The result was tied to assessment.")
    abbreviation = minimal_replacement_text_integrity("The U.S. healthcare system is expensive.")
    missing_space = minimal_replacement_text_integrity("The result was tied to assessment (Hattie, 2009).assessment.")
    nested_parenthetical = minimal_replacement_text_integrity(
        "Teachers support students to become their own teachers (Through role playing, "
        "Johnny improved (Hattie, 2009)."
    )

    assert not integrity["passed"]
    assert "embedded_sentence_punctuation_word_artifact" in integrity["failures"]
    assert not duplicate["passed"]
    assert "embedded_sentence_punctuation_word_artifact" in duplicate["failures"]
    assert not missing_space["passed"]
    assert "sentence_punctuation_spacing_artifact" in missing_space["failures"]
    assert not nested_parenthetical["passed"]
    assert "nested_parenthetical_artifact" in nested_parenthetical["failures"]
    assert ordinary["passed"]
    assert abbreviation["passed"]


def test_v5_tail_cleanup_expands_mid_word_scanner_boundaries():
    text = "Students can share skills to help more people. The proverb follows."
    start = text.index("help")
    end = text.index("The")
    cut_start = start + len("help mo")
    cut_end = end + len("Th")

    expanded_start, expanded_end = _expand_to_local_text_boundaries(text, cut_start, cut_end)

    assert text[expanded_start:expanded_end] == text


def test_v5_tail_cleanup_expands_mid_sentence_risky_window_boundaries():
    text = (
        "The Civil Rights Movement was an important period that aimed to end racial segregation "
        "and improve equality for African Americans.\n\n"
        "Despite its success, the United States also faces many serious issues. "
        "Critics argue that the benefits of economic growth are not equally shared among all people."
    )
    cut_start = text.index("equality")
    cut_end = text.index("equally shared")

    expanded_start, expanded_end = _expand_to_local_text_boundaries(text, cut_start, cut_end)
    expanded = text[expanded_start:expanded_end]

    assert expanded.startswith("The Civil Rights Movement")
    assert expanded.endswith("equally shared among all people.")


def test_v5_tail_cleanup_rejects_mid_sentence_apply_boundaries():
    text = "The first sentence is stable. The second sentence should stay whole."
    section = SectionUnit(
        section_id="w001",
        heading="Risky window cleanup",
        text="sentence should stay",
        start_char=text.index("sentence should"),
        end_char=text.index(" whole"),
        paragraph_count=1,
        word_count=3,
        metadata={},
    )

    integrity = _section_apply_boundary_integrity(text, section)

    assert not integrity["passed"]
    assert "left_boundary_inside_sentence" in integrity["failures"]
    assert "right_boundary_inside_sentence" in integrity["failures"]


def test_v5_apply_integrity_gate_allows_preexisting_document_artifacts_only():
    before = minimal_replacement_text_integrity("The source already has this issue.This sentence continues.")
    same_after = minimal_replacement_text_integrity("The source already has this issue.This sentence continues.")
    worse_after = minimal_replacement_text_integrity(
        "The source already has this issue.This sentence continues. A new issue appears.Another sentence."
    )

    assert not before["passed"]
    assert _text_integrity_regression(before, same_after)["passed"]
    regression = _text_integrity_regression(before, worse_after)
    assert not regression["passed"]
    assert "sentence_punctuation_spacing_count" in regression["metric_regressions"]


def test_v5_production_adapter_returns_v5_report_contract(tmp_path, monkeypatch):
    emitted_checkpoints: list[dict] = []
    provider_routing = {"sort": "throughput", "allow_fallbacks": True}
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_PROVIDER_ROUTING_JSON", json.dumps(provider_routing))
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_PLANNER_MODEL", raising=False)
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_NORMALIZER_MODEL", raising=False)

    def fake_residual_comb(**kwargs):
        assert kwargs["max_rounds"] == 6
        assert kwargs["variant_count"] == 5
        assert kwargs["retune_variant_count"] == 5
        assert kwargs["provider"] == provider_routing
        assert kwargs["planner_model"] == "z-ai/glm-5.1"
        callback = kwargs.get("accepted_checkpoint_callback")
        assert callback is not None
        callback({
            "schema_version": "rewrite_v5_accepted_checkpoint.v1",
            "stage": "v5_residual_cluster_comb",
            "sequence": 1,
            "phase": "unsafe_cluster_cleanup",
            "round": 1,
            "reason": "accepted_unsafe_cluster_movement",
            "baseline_scores": {"ai": 40.0},
            "scores": {"ai": 34.0},
            "goal": {"goal_met": False},
            "accepted": {"section_id": "density_cluster_001", "variant_id": "v1"},
            "rewritten_document": "This is the rewritten document.",
        })
        return {
            "stage": "v5_residual_cluster_comb",
            "baseline_scores": {
                "ai": 40.0,
                "topk": 84.0,
                "external": 37.0,
                "rank": 104.0,
                "unsafe_cluster_count": 8,
            },
            "final_scores": {
                "ai": 34.0,
                "topk": 76.0,
                "external": 29.0,
                "rank": 82.0,
                "unsafe_cluster_count": 4,
            },
            "goal": {"status": "mitigation_failed_no_safe_candidate", "goal_met": False},
            "rounds": [
                {
                    "round": 1,
                    "status": "accepted",
                    "accepted": {
                        "section_id": "route_001",
                        "variant_id": "v1",
                        "label": "initial_v1",
                        "word_count": 20,
                        "text": "Rewritten cluster text.",
                        "scores": {"ai_delta": 6.0, "topk_delta": 8.0},
                        "local_scores": {"unsafe_cluster_count": 0, "topk_delta": 20.0},
                    },
                    "candidates": [{"variant_id": "v1"}],
                }
            ],
            "phase_order": {
                "unsafe_cluster_first": True,
                "reason": "eligible_span_density_unsafe",
            },
            "rewritten_document": "This is the rewritten document.",
        }

    monkeypatch.setattr(v5_production, "run_v5_residual_cluster_comb_experiment", fake_residual_comb)
    monkeypatch.setattr(v5_production, "_scan_report", lambda text: {
        "input_text": text,
        "ai_score": 34.0,
        "ai_risk_badge": {"ai_likelihood_score": 34.0},
        "findings": {},
    })
    monkeypatch.setattr(v5_production, "evaluate_rewrite_goal", lambda **_: SimpleNamespace(
        to_dict=lambda: {
            "status": "mitigation_failed_no_safe_candidate",
            "goal_met": False,
            "reason": "candidate_failed_strict_detector_safe_goal",
        }
    ))
    monkeypatch.setattr(v5_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))

    result = v5_production.run_rewrite_pipeline_v5(
        detect_json={
            "input_text": "This is the original document.",
            "ai_score": 40.0,
            "ai_risk_badge": {"ai_likelihood_score": 40.0},
            "findings": {},
        },
        output_dir=str(tmp_path),
        model="deepseek/deepseek-v3.2",
        checkpoint_callback=emitted_checkpoints.append,
    )

    summary = json.loads(Path(result["json_path"]).read_text())
    assert result["status"] == "rewrite_candidate_generated_needs_external_review"
    assert summary["rewrite_pipeline_version"] == "rewrite_v5_residual_cluster_comb"
    assert summary["rewrite_engine_mode"] == "v5_residual_cluster_comb_production"
    assert summary["candidate_generation_status"]["accepted_count"] == 1
    assert summary["partial_rewrite_preserved"] is True
    assert summary["partial_rewrite_preservation_reason"] == "safe_progress_kept_despite_strict_goal_miss"
    assert summary["v5_scores"]["deltas"]["ai_delta"] == 6.0
    assert summary["rewrite_effective_config"]["provider_routing"] == provider_routing
    assert summary["rewrite_effective_config"]["planner_model"] == "z-ai/glm-5.1"
    assert summary["final_text"] == "This is the rewritten document."
    layer = summary["rewrite_layers"]["v5_residual_cluster_comb"]
    assert layer["phase_order"]["unsafe_cluster_first"] is True
    assert emitted_checkpoints
    assert emitted_checkpoints[0]["status"] == "rewrite_candidate_generated_needs_external_review"
    assert emitted_checkpoints[0]["final_text"] == "This is the rewritten document."
    assert emitted_checkpoints[0]["summary"]["checkpoint_recovery_available"] is True


def test_v5_production_treats_global_best_fallback_as_selected_candidate(tmp_path, monkeypatch):
    def fake_residual_comb(**kwargs):
        return {
            "stage": "v5_residual_cluster_comb",
            "baseline_scores": {"ai": 55.0, "rank": 140.0},
            "final_scores": {"ai": 51.0, "rank": 135.0},
            "goal": {
                "status": "mitigation_failed_no_safe_candidate",
                "goal_met": False,
                "reason": "candidate_failed_strict_detector_safe_goal",
            },
            "rounds": [],
            "risky_window_cleanup_rounds": [],
            "unsafe_cluster_cleanup_rounds": [],
            "final_risky_window_cleanup_rounds": [],
            "eligible_span_density_gate": {},
            "phase_order": {"unsafe_cluster_first": False, "reason": "default_route_then_cleanup"},
            "global_best_fallback": {
                "applied": True,
                "reason": "best_full_document_candidate_superseded_phase_accepted_result",
                "selected": {
                    "section_id": "density_cluster_004",
                    "variant_id": "v3",
                    "label": "density_v3",
                    "word_count": 72,
                    "scores": {"ai_delta": 4.0, "rank_delta": 5.0},
                    "text": "fallback candidate",
                },
            },
            "rewritten_document": "This is the fallback-selected document.",
        }

    monkeypatch.setattr(v5_production, "run_v5_residual_cluster_comb_experiment", fake_residual_comb)
    monkeypatch.setattr(v5_production, "_scan_report", lambda text: {
        "input_text": text,
        "ai_score": 51.0,
        "ai_risk_badge": {"ai_likelihood_score": 51.0},
        "findings": {},
    })
    monkeypatch.setattr(v5_production, "evaluate_rewrite_goal", lambda **_: SimpleNamespace(
        to_dict=lambda: {
            "status": "mitigation_failed_no_safe_candidate",
            "goal_met": False,
            "reason": "candidate_failed_strict_detector_safe_goal",
        }
    ))
    monkeypatch.setattr(v5_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))

    result = v5_production.run_rewrite_pipeline_v5(
        detect_json={
            "input_text": "This is the original document.",
            "ai_score": 55.0,
            "ai_risk_badge": {"ai_likelihood_score": 55.0},
            "findings": {},
        },
        output_dir=str(tmp_path),
        model="deepseek/deepseek-v3.2",
    )

    summary = json.loads(Path(result["json_path"]).read_text())
    layer = summary["rewrite_layers"]["v5_residual_cluster_comb"]
    assert result["status"] == "rewrite_candidate_generated_needs_external_review"
    assert summary["candidate_generation_status"]["accepted_count"] == 1
    assert summary["selected_candidate"]["section_id"] == "density_cluster_004"
    assert layer["global_best_fallback"]["applied"] is True
    assert layer["phase_order"]["reason"] == "default_route_then_cleanup"


def test_v5_route_blueprint_moves_generic_opener_after_concrete_event():
    section = build_section_units(
        "For Johnny, this became an important learning experience. "
        "During the service, he showed patience and adjusted his communication. "
        "Two days later, he received a thank-you card.",
        {},
    )[0]
    section = type(section)(
        section_id=section.section_id,
        heading=section.heading,
        text=section.text,
        start_char=section.start_char,
        end_char=section.end_char,
        paragraph_count=section.paragraph_count,
        word_count=section.word_count,
        metadata={"source_metadata": {"gate_cluster": {"generic_hits": 1}}},
    )
    goal = {
        "eligible_span_density_gate": {
            "top_unsafe_clusters": [
                {"preview": "For Johnny, this became an important learning experience."},
            ],
        },
    }

    blueprint = build_route_blueprint(section=section, local_goal=goal)

    assert blueprint["strategy"] == "event_first_rebuild"
    assert blueprint["start_source_beat_index"] == 1
    assert blueprint["steps"][0]["source_beat"].startswith("During the service")
    assert blueprint["sentence_jobs"][0].startswith("Sentence 1")
    assert any("For Johnny" in item for item in blueprint["avoid_openers"])


def test_v5_route_window_prompt_does_not_require_heading():
    cluster = type("Cluster", (), {
        "cluster_id": "cluster_003",
        "text": "The learner struggled with the skill. The teacher adjusted the practice route.",
        "start_char": 12,
        "end_char": 91,
        "risk_score": 8.0,
        "sentence_count": 2,
        "metadata": {},
    })()
    section = _section_from_cluster(cluster)
    fact_map = build_fact_map(section, {})
    prompt = build_recomposer_prompt(section=section, fact_map=fact_map, variant_count=1)
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert section.heading == ""
    assert payload["section"]["heading"] == ""
    assert any("Do not add a heading" in rule for rule in payload["reconstruction_rules"])


def test_v5_fact_integrity_requires_citations_and_protected_terms():
    text = "Body\nThe supported claim cites prior work (Lee, 2024) and keeps UNIT301 visible."
    section = build_section_units(text, {})[0]
    fact_map = build_fact_map(section, {})

    failed = fact_map_integrity(fact_map, "Body\nThe supported claim cites prior work.")
    passed = fact_map_integrity(fact_map, "Body\nThe supported claim cites prior work (Lee, 2024) and keeps UNIT301 visible.")

    assert not failed["passed"]
    assert any(row["reason"] == "citation_missing" for row in failed["failures"])
    assert passed["passed"]


def test_v5_acceptance_does_not_select_rejected_or_no_movement_candidate():
    rejected = {
        "apply_status": {"applied": True},
        "source_grounding": {"passed": False},
        "fact_integrity": {"passed": True},
        "scores": {"external_delta": 5.0, "rank_delta": 5.0, "unsafe_word_ratio_delta": 5.0},
    }
    flat = {
        "apply_status": {"applied": True},
        "source_grounding": {"passed": True},
        "fact_integrity": {"passed": True},
        "scores": {"external_delta": 0.0, "rank_delta": 0.0, "unsafe_word_ratio_delta": 0.0},
    }
    useful = {
        "apply_status": {"applied": True},
        "source_grounding": {"passed": True},
        "fact_integrity": {"passed": True},
        "scores": {
            "external_delta": 0.1,
            "rank_delta": 0.2,
            "ai_delta": 0.0,
            "topk_delta": 0.0,
            "topk_calibrated_risk_delta": 1.2,
            "qualifying_text_ai_density_delta": 0.0,
            "external_ai_flag_risk_delta": 0.0,
            "unsafe_word_ratio_delta": 0.0,
        },
    }
    cluster_only = {
        "apply_status": {"applied": True},
        "source_grounding": {"passed": True},
        "fact_integrity": {"passed": True},
        "scores": {
            "external_delta": 0.1,
            "rank_delta": 0.2,
            "ai_delta": -0.02,
            "topk_delta": -0.04,
            "topk_calibrated_risk_delta": -0.109,
            "qualifying_text_ai_density_delta": -0.03,
            "external_ai_flag_risk_delta": -0.002,
            "unsafe_word_ratio_delta": -0.046,
            "unsafe_cluster_count_delta": 1.0,
        },
    }
    negligible = {
        "apply_status": {"applied": True},
        "source_grounding": {"passed": True},
        "fact_integrity": {"passed": True},
        "scores": {
            "external_delta": 0.001,
            "rank_delta": 0.001,
            "ai_delta": 0.0,
            "topk_delta": 0.0,
            "topk_calibrated_risk_delta": 0.0,
            "qualifying_text_ai_density_delta": 0.0,
            "external_ai_flag_risk_delta": 0.001,
            "unsafe_word_ratio_delta": 0.0,
            "unsafe_cluster_count_delta": 0.0,
        },
    }

    assert not _is_safe_candidate(rejected)
    assert not _is_safe_candidate(flat)
    assert not _is_safe_candidate(cluster_only)
    assert not _is_safe_candidate(negligible)
    assert _is_safe_candidate(useful)


def test_v5_stack_summary_reports_core_improvement():
    baseline = {
        "ai": 40.0,
        "topk": 80.0,
        "external": 35.0,
        "rank": 100.0,
        "risky_window_count": 2,
        "unsafe_word_ratio": 12.0,
        "unsafe_cluster_count": 8,
        "topk_calibrated_risk": 45.0,
        "qualifying_text_ai_density": 44.0,
        "ai_authorship": 38.0,
        "external_ai_flag_risk": 36.0,
    }
    final = {
        "ai": 34.0,
        "topk": 77.0,
        "external": 31.0,
        "rank": 81.0,
        "risky_window_count": 1,
        "unsafe_word_ratio": 8.0,
        "unsafe_cluster_count": 5,
        "topk_calibrated_risk": 37.0,
        "qualifying_text_ai_density": 40.0,
        "ai_authorship": 35.0,
        "external_ai_flag_risk": 33.0,
    }

    summary = _stack_summary(
        baseline=baseline,
        final=final,
        route_result={"accepted_candidate": {"variant_id": "v1"}},
        cleanup_result={"accepted": [{"round": 1}, {"round": 2}]},
    )

    assert summary["route_accepted"] is True
    assert summary["cleanup_accepted_count"] == 2
    assert summary["deltas"]["ai_delta"] == 6.0
    assert summary["all_core_improved"] is True
