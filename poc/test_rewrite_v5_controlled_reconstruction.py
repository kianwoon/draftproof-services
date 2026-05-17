import json
from pathlib import Path
from types import SimpleNamespace

import rewrite_v5.production as v5_production
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
    build_residual_cluster_prompt,
    build_residual_cluster_route_plan_prompt,
    build_residual_cluster_retune_prompt,
    build_route_blueprint,
    build_unsafe_cluster_cleanup_prompt,
    _best_full_document_candidate,
    generate_residual_cluster_seed_variants,
    _expand_to_local_text_boundaries,
    _full_document_candidate_beats_scores,
    _has_risky_window_cleanup_movement,
    _has_unsafe_cluster_cleanup_movement,
    _has_full_document_fallback_movement,
    _parse_route_plan,
    _has_incremental_movement,
    _residual_candidate_sort_key,
)


def _sample_route_plan() -> dict:
    return {
        "content_profile": "narrative_or_case_reflection",
        "cluster_role": "evidence_or_example",
        "dominant_failure_pattern": "event_summary",
        "route_strategy": "event_first_rebuild",
        "profile_reason": "The cluster follows an event and its visible outcome.",
        "failed_route": "The current route summarizes the result too quickly.",
        "replacement_route": "Start from the event, then show the visible outcome.",
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
    assert "do not invent new names" in lowered
    assert "source viewpoint" in lowered
    assert "source-near" in lowered


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
    assert "do not invent new names" in lowered


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

    assert payload["task"] == "profile_aware_cluster_route_plan"
    assert payload["cluster"]["source_event_beats"]
    assert payload["cluster"]["source_blocks"]
    assert payload["cluster"]["referential_continuity"]["opening_subject"] == "The"
    assert payload["scanner_local_findings"]["top_sentence_targets"][0]["sentence_id"] == "s001"
    assert "content_profile_rubrics" in payload
    assert "broad_explanatory_report" in payload["content_profile_rubrics"]
    assert "cluster_role_options" in payload
    assert "failure_pattern_options" in payload
    assert "route_strategy_options" in payload
    assert "route_plan" in payload["output_schema"]
    assert set(payload["output_schema"]["route_plan"].keys()) == {
        "content_profile",
        "cluster_role",
        "dominant_failure_pattern",
        "route_strategy",
        "profile_reason",
        "failed_route",
        "replacement_route",
        "source_block_plan",
        "target_sentence_jobs",
        "must_change",
        "must_preserve",
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
            "cluster_role": "evidence_or_example",
            "dominant_failure_pattern": "event_summary",
            "route_strategy": "event_first_rebuild",
            "profile_reason": "The cluster follows a service result and a thank-you card outcome.",
            "failed_route": "The current route starts with broad interpretation before showing the event.",
            "replacement_route": "Start from the service moment, then show how the thank-you card made the result visible.",
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
    assert payload["coverage_guidance"]["requirements"]
    assert "fallback_route_blueprint" not in payload
    assert "custom_route_plan" not in payload
    assert payload["length_guidance"]["preferred_max_words"] == round(section.word_count * 1.10)
    lowered = prompt.casefold()
    assert "follow execution_brief.replacement_route" in lowered
    assert "fallback_route_blueprint" not in lowered


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

    assert variants == []


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
    assert payload["source_blocks"]
    assert payload["coverage_guidance"]["requirements"]
    assert any("Follow execution_brief.replacement_route" in item for item in payload["method"])
    assert payload["length_guidance"]["preferred_max_words"] == round(section.word_count * 1.10)


def test_v5_unsafe_cluster_cleanup_prompt_is_local_and_source_near():
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
    assert payload["source_blocks"]
    assert payload["coverage_guidance"]["requirements"]
    assert any("Follow execution_brief.replacement_route" in item for item in payload["method"])
    assert payload["length_guidance"]["preferred_max_words"] == round(section.word_count * 1.10)


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
        "incremental": {"risky_window_count_delta": 1.0, "rank_delta": 0.5},
    }
    risky_window_no_drop = {
        "incremental": {"risky_window_count_delta": 0.0, "rank_delta": 5.0},
    }
    cluster_drop = {
        "incremental": {"unsafe_cluster_count_delta": 1.0, "rank_delta": 0.4},
    }
    cluster_rank_regression = {
        "incremental": {"unsafe_cluster_count_delta": 1.0, "rank_delta": -0.5},
    }

    assert _has_risky_window_cleanup_movement(risky_window_drop)
    assert not _has_risky_window_cleanup_movement(risky_window_no_drop)
    assert _has_unsafe_cluster_cleanup_movement(cluster_drop)
    assert not _has_unsafe_cluster_cleanup_movement(cluster_rank_regression)


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

    assert text[expanded_start:expanded_end].startswith("more people.")
    assert text[expanded_start:expanded_end].endswith("The")


def test_v5_production_adapter_returns_v5_report_contract(tmp_path, monkeypatch):
    def fake_residual_comb(**kwargs):
        assert kwargs["max_rounds"] == 6
        assert kwargs["variant_count"] == 5
        assert kwargs["retune_variant_count"] == 5
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
    )

    summary = json.loads(Path(result["json_path"]).read_text())
    assert result["status"] == "rewrite_candidate_generated_needs_external_review"
    assert summary["rewrite_pipeline_version"] == "rewrite_v5_residual_cluster_comb"
    assert summary["rewrite_engine_mode"] == "v5_residual_cluster_comb_production"
    assert summary["candidate_generation_status"]["accepted_count"] == 1
    assert summary["v5_scores"]["deltas"]["ai_delta"] == 6.0
    assert summary["final_text"] == "This is the rewritten document."


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
