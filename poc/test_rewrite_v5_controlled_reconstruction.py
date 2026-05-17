import json

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
    build_residual_cluster_prompt,
    build_residual_cluster_retune_prompt,
    build_route_blueprint,
    generate_residual_cluster_seed_variants,
    _has_incremental_movement,
    _residual_candidate_sort_key,
)


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
    assert payload["cluster"]["source_phrase_anchors"]
    assert payload["route_blueprint"]["steps"]
    assert payload["length_guidance"]["preferred_min_words"] > section.word_count
    assert payload["route_blueprint"]["sentence_jobs"]
    assert payload["remaining_problem_sentences"] == ["Johnny needed support in class."]
    assert len(payload["output_schema"]["variants"]) == 4
    assert all(set(row.keys()) == {"variant_id", "text"} for row in payload["output_schema"]["variants"])
    lowered = prompt.casefold()
    assert "ai detector" not in lowered
    assert "bypass" not in lowered
    assert "current_route" not in lowered
    assert "better_route" not in lowered
    assert "do not invent new names" in lowered
    assert "teacher viewpoint" in lowered
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
    assert payload["cluster"]["source_phrase_anchors"]
    assert payload["route_blueprint"]["steps"]
    assert payload["length_guidance"]["preferred_min_words"] > section.word_count
    assert "retune_focus" in payload
    assert "candidate_non_source_terms_to_reduce" in payload
    assert payload["route_blueprint"]["sentence_jobs"]
    assert payload["remaining_problem_sentences"] == ["The feedback changed his confidence."]
    assert len(payload["output_schema"]["variants"]) == 2
    lowered = prompt.casefold()
    assert "ai detector" not in lowered
    assert "bypass" not in lowered
    assert "do not invent new names" in lowered


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


def test_v5_residual_seed_generator_builds_source_gated_role_route():
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

    assert len(variants) == 1
    assert variants[0].variant_id == "route_seed_1"
    assert "At the beginning" not in variants[0].text
    assert "required a support worker to accompany" not in variants[0].text
    assert "hair salon manager" in variants[0].text
    assert "I had to understand that starting point" in variants[0].text


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
