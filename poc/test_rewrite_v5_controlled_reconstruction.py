import json
from pathlib import Path
from types import SimpleNamespace

import rewrite_v5.production as v5_production
import rewrite_v5.residual_comb as v5_residual_comb
from report import render_rewrite as rewrite_report
from rewrite_v3.text_integrity import minimal_replacement_text_integrity
from rewrite_v5.cluster_mass import build_cluster_mass_prompt
from rewrite_v5.experiment import (
    _is_safe_candidate,
    _section_from_cluster,
    _stack_summary,
    _variants_response_format,
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
    build_borderline_verdict_cleanup_prompt,
    build_final_topk_sentence_route_prompt,
    build_safe_band_author_proxy_revision_plan_prompt,
    build_safe_band_evidence_pack_prompt,
    build_safe_band_evidence_repair_prompt,
    build_route_blueprint,
    build_unsafe_cluster_cleanup_prompt,
    generate_residual_cluster_route_plan,
    run_v5_residual_cluster_comb_experiment,
    _best_full_document_candidate,
    _best_safe_band_evidence_repair_candidate,
    _best_balanced_ai_topk_candidate,
    _best_borderline_verdict_candidate,
    _balanced_ai_topk_sort_value,
    _direct_scanner_candidate_strong_enough,
    _direct_scanner_batch_policy,
    _author_proxy_candidate_audit,
    _should_skip_core_after_direct_accept,
    _should_continue_direct_scanner_batches,
    generate_residual_cluster_seed_variants,
    _expand_to_local_text_boundaries,
    _full_document_candidate_beats_scores,
    _generate_loose_variants_from_builder,
    _has_risky_window_cleanup_movement,
    _has_balanced_ai_topk_movement,
    _has_unsafe_cluster_cleanup_movement,
    _has_full_document_fallback_movement,
    _has_safe_band_evidence_repair_movement,
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
    _has_core_round_acceptance_movement,
    _residual_candidate_sort_key,
    _adaptive_initial_variant_count,
    _adaptive_retune_variant_count,
    _adaptive_writer_feedback,
    _adaptive_cutoff_blocker_state,
    _adaptive_cutoff_runtime_budget_seconds,
    _adaptive_cutoff_stop_event,
    _borderline_verdict_should_run,
    _has_borderline_verdict_movement,
    _borderline_rejected_candidate_feedback,
    _borderline_verdict_candidate_crosses_boundary,
    _incremental_deltas,
    _score_full_document_variant,
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


def test_v5_residual_cluster_prompt_attaches_author_proxy_context():
    section = build_section_units(
        "The placement helped me understand the client consultation process. "
        "I also noticed that timing affected how comfortable the client felt.",
        {},
    )[0]
    author_proxy_context = {
        "schema_version": "author_proxy_context.v1",
        "active": True,
        "mode": "non_interrupting_author_proxy_draft",
        "review_required": True,
        "primary_mode": "author_grounded_evidence_rebuild",
        "required_inputs": ["specific class observation"],
        "allowed_provenance": ["source_preserved", "inferred_from_draft", "needs_author_confirmation"],
        "quality_bar": {"target": "highest_quality_grounded_candidate", "basis": "submitted_content_only"},
        "review_cards": [{
            "card_id": "target-01",
            "provenance": "needs_author_confirmation",
            "target_text": "understand the client consultation process",
            "user_input_needed": "Confirm the real observation or replace it.",
        }],
    }

    prompt = build_residual_cluster_prompt(
        section=section,
        local_goal={},
        variant_count=2,
        author_proxy_context=author_proxy_context,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["author_proxy_context"]["mode"] == "non_interrupting_author_proxy_draft"
    assert payload["author_proxy_context"]["review_required"] is True
    assert payload["author_proxy_context"]["quality_bar"]["target"] == "highest_quality_grounded_candidate"
    assert payload["author_proxy_context"]["review_cards"][0]["card_id"] == "target-01"
    assert payload["author_proxy_context"]["authorship_evidence_contract"]["schema_version"] == "authorship_evidence_contract.v1"
    assert payload["author_proxy_quality_contract"]["target"] == "highest_quality_grounded_candidate"
    assert "submitted source_text" in payload["author_proxy_quality_contract"]["basis"]
    assert any("No variant contains placeholders" in row for row in payload["author_proxy_quality_contract"]["self_check_before_return"])
    assert payload["output_schema"]["variants"][0]["author_proxy_provenance"]
    assert payload["output_schema"]["variants"][0]["author_review_items"]
    assert "Do not invent personal experiences" in " ".join(payload["author_proxy_rules"])
    assert "highest-quality polished candidate" in " ".join(payload["author_proxy_rules"])
    assert payload["author_proxy_candidate_audit_contract"]["applied_by"].startswith("DraftProof controller")
    assert "author_proxy_provenance" in payload["author_proxy_candidate_audit_contract"]["model_output_schema"]
    assert payload["provenance_contract"]["needs_author_confirmation"]
    assert "acceptance_note" in payload["provenance_contract"]


def test_v5_author_proxy_variant_parser_keeps_candidate_review_items():
    raw = json.dumps({
        "variants": [{
            "variant_id": "v1",
            "text": "The placement showed how timing affected the client consultation.",
            "author_proxy_provenance": [{
                "item_id": "p001",
                "provenance": "inferred_from_draft",
                "target_text": "timing affected consultation",
                "generated_text": "timing affected the client consultation",
                "user_input_needed": "",
                "author_task": "",
            }],
            "author_review_items": [{
                "item_id": "r001",
                "provenance": "needs_author_confirmation",
                "target_text": "client consultation",
                "generated_text": "client consultation",
                "user_input_needed": "Confirm the actual consultation detail.",
                "author_task": "Verify the consultation detail.",
            }],
        }]
    })

    variants, diagnostics = v5_residual_comb._parse_loose_variants(raw)

    assert diagnostics["status"] == "ok"
    assert variants[0].author_proxy_provenance[0]["provenance"] == "inferred_from_draft"
    assert variants[0].author_review_items[0]["item_id"] == "r001"


def test_v5_author_proxy_structured_output_requires_candidate_review_fields():
    schema = _variants_response_format(2, include_author_proxy_fields=True)
    variant_schema = schema["json_schema"]["schema"]["properties"]["variants"]["items"]

    assert "author_proxy_provenance" in variant_schema["required"]
    assert "author_review_items" in variant_schema["required"]
    assert variant_schema["properties"]["author_proxy_provenance"]["type"] == "array"
    assert variant_schema["properties"]["author_review_items"]["type"] == "array"


def test_v5_author_proxy_quality_does_not_override_scanner_movement():
    context = {"active": True, "review_required": True, "review_cards": [{"card_id": "target-01"}]}
    grounded = {
        "apply_status": {"applied": True},
        "local_scores": {"unsafe_cluster_count_delta": 0},
        "incremental": {"ai_delta": 1.0, "topk_delta": 1.0},
        "scores": {"ai_delta": 1.0, "topk_delta": 1.0},
        "author_proxy_quality": {
            "active": True,
            "score": 0.92,
        },
    }
    thin_scanner_win = {
        "apply_status": {"applied": True},
        "local_scores": {"unsafe_cluster_count_delta": 0},
        "incremental": {"ai_delta": 4.0, "topk_delta": 4.0},
        "scores": {"ai_delta": 4.0, "topk_delta": 4.0},
        "author_proxy_quality": {
            "active": True,
            "score": 0.41,
        },
    }

    quality = v5_residual_comb._author_proxy_quality_score(
        source_text="The placement helped me understand the client consultation process.",
        candidate_text="The placement helped me understand how timing affected the client consultation process.",
        context=context,
        provenance=[{"item_id": "p001"}],
        review_items=[{"item_id": "r001"}],
        audit={"active": True, "safety_gate": {"passed": True}},
    )

    assert quality["score"] > 0.5
    assert v5_residual_comb._residual_candidate_sort_key(thin_scanner_win) > v5_residual_comb._residual_candidate_sort_key(grounded)


def test_v5_author_proxy_quality_is_tiebreaker_after_scanner_scores():
    high_quality = {
        "apply_status": {"applied": True},
        "incremental": {"risky_window_count_delta": 1.0, "ai_delta": 4.0, "topk_delta": 4.0},
        "scores": {"ai_delta": 4.0, "topk_delta": 4.0, "rank_delta": 10.0},
        "author_proxy_quality": {"active": True, "score": 0.91},
    }
    low_quality = {
        "apply_status": {"applied": True},
        "incremental": {"risky_window_count_delta": 1.0, "ai_delta": 4.0, "topk_delta": 4.0},
        "scores": {"ai_delta": 4.0, "topk_delta": 4.0, "rank_delta": 10.0},
        "author_proxy_quality": {"active": True, "score": 0.42},
    }
    stronger_scanner = {
        "apply_status": {"applied": True},
        "incremental": {"risky_window_count_delta": 1.0, "ai_delta": 8.0, "topk_delta": 7.0},
        "scores": {"ai_delta": 8.0, "topk_delta": 7.0, "rank_delta": 20.0},
        "author_proxy_quality": {"active": True, "score": 0.5},
    }

    assert v5_residual_comb._risky_window_cleanup_sort_key(high_quality) > v5_residual_comb._risky_window_cleanup_sort_key(low_quality)
    assert v5_residual_comb._risky_window_cleanup_sort_key(stronger_scanner) > v5_residual_comb._risky_window_cleanup_sort_key(high_quality)


def test_v5_final_topk_sentence_route_rejects_ai_regression_by_default():
    topk_only = {
        "apply_status": {"applied": True},
        "incremental": {
            "topk_delta": 1.0,
            "topk_calibrated_risk_delta": 1.0,
            "ai_delta": -0.1,
            "risky_window_count_delta": 0.0,
            "unsafe_cluster_count_delta": 0.0,
        },
    }
    balanced = {
        "apply_status": {"applied": True},
        "incremental": {
            "topk_delta": 0.5,
            "topk_calibrated_risk_delta": 0.5,
            "ai_delta": 0.0,
            "risky_window_count_delta": 0.0,
            "unsafe_cluster_count_delta": 0.0,
        },
    }

    assert not v5_residual_comb._has_final_topk_sentence_route_movement(topk_only)
    assert v5_residual_comb._has_final_topk_sentence_route_movement(balanced)


def test_v5_final_topk_sentence_route_can_score_single_repair_salvage(tmp_path, monkeypatch):
    original_text = "First risky sentence. Second risky sentence. Stable ending."
    current_scores = {"ai": 40.0, "topk": 80.0, "unsafe_cluster_count": 2}
    scan_scores = iter([
        {"ai": 41.0, "topk": 79.0, "unsafe_cluster_count": 1},
        {"ai": 39.0, "topk": 78.0, "unsafe_cluster_count": 2},
    ])

    monkeypatch.setattr(v5_residual_comb, "_scan_report", lambda _text: {"input_text": _text})
    monkeypatch.setattr(
        v5_residual_comb,
        "evaluate_rewrite_goal",
        lambda **_: SimpleNamespace(to_dict=lambda: {}),
    )
    monkeypatch.setattr(v5_residual_comb, "_with_v5_density_gate", lambda _text, _report, goal: goal)
    monkeypatch.setattr(v5_residual_comb, "_score_summary", lambda *_args: next(scan_scores))
    monkeypatch.setattr(v5_residual_comb, "_add_deltas", lambda scores, baseline: scores.update({
        "ai_delta": 50.0 - scores["ai"],
        "topk_delta": 90.0 - scores["topk"],
        "unsafe_cluster_count_delta": 3.0 - scores["unsafe_cluster_count"],
    }))

    full_row = v5_residual_comb._score_final_topk_sentence_route_variant(
        original_text=original_text,
        baseline_report={},
        baseline_scores={"ai": 50.0, "topk": 90.0, "unsafe_cluster_count": 3},
        current_text=original_text,
        current_scores=current_scores,
        targets=[
            {"target_id": "t001", "sentence": "First risky sentence."},
            {"target_id": "t002", "sentence": "Second risky sentence."},
        ],
        variant={
            "variant_id": "v1",
            "repairs": [
                {"target_id": "t001", "after": "First repair."},
                {"target_id": "t002", "after": "Second repair."},
            ],
        },
        output_dir=tmp_path,
        label="full",
    )
    partial_row = v5_residual_comb._score_final_topk_sentence_route_variant(
        original_text=original_text,
        baseline_report={},
        baseline_scores={"ai": 50.0, "topk": 90.0, "unsafe_cluster_count": 3},
        current_text=original_text,
        current_scores=current_scores,
        targets=[
            {"target_id": "t001", "sentence": "First risky sentence."},
            {"target_id": "t002", "sentence": "Second risky sentence."},
        ],
        variant={
            "variant_id": "v1_t002",
            "repairs": [{"target_id": "t002", "after": "Second repair."}],
        },
        output_dir=tmp_path,
        label="partial",
        require_all_targets=False,
    )

    assert full_row["apply_status"]["partial_candidate"] is False
    assert partial_row["apply_status"]["partial_candidate"] is True
    assert partial_row["apply_status"]["applied_repair_count"] == 1
    assert not v5_residual_comb._has_final_topk_sentence_route_movement(full_row)
    assert v5_residual_comb._has_final_topk_sentence_route_movement(partial_row)
    assert v5_residual_comb._best_final_topk_sentence_route_candidate([full_row, partial_row]) is partial_row


def test_v5_final_topk_prompt_carries_safe_band_kpi_contract():
    prompt = build_final_topk_sentence_route_prompt(
        current_scores={
            "ai": 33.32,
            "topk": 73.68,
            "topk_calibrated_risk": 30.586,
            "qualifying_text_ai_density": 40.73,
        },
        current_goal={
            "ai_footprint_gate": {
                "safe_band_thresholds": {
                    "topk_calibrated_risk": 25.0,
                    "qualifying_text_ai_density": 35.0,
                },
                "remaining_ai_footprint_drivers": [
                    {"driver": "topk_calibrated_risk", "value": 30.586, "safe_band": 25.0},
                    {"driver": "qualifying_text_ai_density", "value": 40.73, "safe_band": 35.0},
                ],
                "texture_blockers": [{"driver": "topk_calibrated_risk"}],
                "after": {
                    "semantic_footprint": {
                        "generic_assertion_risk": 80.0,
                        "unsupported_claim_risk": 70.0,
                        "qualifying_text_ai_density": 40.73,
                    },
                    "authorship_footprint": {
                        "rewrite_smoothness": 36.2,
                    },
                },
            }
        },
        targets=[{
            "target_id": "t001",
            "sentence": "This shows that support matters for learning.",
        }],
        variant_count=1,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["kpi_contract"]["objective"] == "clear_strict_ai_safe_band_with_grounded_author_proxy_revision"
    assert payload["kpi_contract"]["targets"]["topk_calibrated_risk"] == 25.0
    assert payload["kpi_contract"]["targets"]["qualifying_text_ai_density"] == 35.0
    assert payload["kpi_contract"]["gaps"]["topk_calibrated_risk"] == 5.586
    assert payload["kpi_contract"]["secondary_density_drivers"]["generic_assertion_risk"] == 80.0
    assert payload["kpi_contract"]["secondary_density_drivers"]["unsupported_claim_risk"] == 70.0
    assert payload["current_scores"]["qualifying_text_ai_density"] == 40.73
    assert any("qualifying_text_ai_density" in rule for rule in payload["rules"])
    assert payload["safe_band_replacement_method"][0].startswith("Treat kpi_contract.gaps")
    assert any("secondary_density_drivers" in row for row in payload["safe_band_replacement_method"])
    assert any("repeat" in row for row in payload["safe_band_replacement_method"])
    assert any("self-repetition" in row for row in payload["rules"])
    assert "safe-band gap" in payload["variant_plan"][0]["goal"]
    assert any("kpi_contract" in rule for rule in payload["rules"])


def test_v5_final_topk_targets_include_paragraph_evidence_context():
    text = (
        "The class began with a short demonstration. "
        "Students talk during this time, and their comments expose what they learned. "
        "I used those comments to decide what to show again.\n\n"
        "The second paragraph is unrelated."
    )
    targets = v5_residual_comb._final_topk_sentence_route_targets(
        text,
        {},
        {
            "eligible_span_density_gate": {
                "top_sentence_targets": [{
                    "sentence_id": "s002",
                    "preview": "Students talk during this time, and their comments expose what they learned.",
                    "top10_ratio": 0.7,
                    "top50_ratio": 0.9,
                    "predictability_risk": 0.5,
                }]
            }
        },
    )

    assert targets[0]["context"]["paragraph_index"] == 1
    assert "short demonstration" in targets[0]["context"]["paragraph"]
    assert targets[0]["context"]["before_sentences"] == ["The class began with a short demonstration."]
    assert targets[0]["context"]["after_sentences"] == ["I used those comments to decide what to show again."]


def test_v5_final_topk_selector_prefers_strict_safe_band_clearance():
    partial_movement = {
        "apply_status": {"applied": True},
        "incremental": {
            "topk_delta": 3.0,
            "topk_calibrated_risk_delta": 6.0,
            "ai_delta": 1.0,
        },
        "scores": {
            "topk_calibrated_risk": 30.0,
            "qualifying_text_ai_density": 41.0,
        },
        "candidate_goal": {
            "strict_ai_safe_band_achieved": False,
            "ai_footprint_gate": {
                "remaining_ai_footprint_drivers": [
                    {"driver": "topk_calibrated_risk", "value": 30.0, "safe_band": 25.0},
                    {"driver": "qualifying_text_ai_density", "value": 41.0, "safe_band": 35.0},
                ]
            },
        },
    }
    strict_safe = {
        "apply_status": {"applied": True},
        "incremental": {
            "topk_delta": 0.5,
            "topk_calibrated_risk_delta": 1.0,
            "ai_delta": 0.1,
        },
        "scores": {
            "topk_calibrated_risk": 24.0,
            "qualifying_text_ai_density": 34.0,
        },
        "candidate_goal": {
            "strict_ai_safe_band_achieved": True,
            "ai_footprint_gate": {
                "safe_band": True,
                "remaining_ai_footprint_drivers": [],
            },
        },
    }

    assert v5_residual_comb._best_final_topk_sentence_route_candidate([partial_movement, strict_safe]) is strict_safe


def test_v5_safe_band_evidence_repair_prompt_carries_kpi_and_author_contract():
    section = SectionUnit(
        section_id="safe_band_evidence_repair_t001",
        heading="Safe-band evidence repair",
        text="Students talk after practice, and I use those comments to decide what to show again.",
        start_char=0,
        end_char=81,
        paragraph_count=1,
        word_count=15,
        metadata={
            "selection_reason": "top_safe_band_sentence_target",
            "target_sentence": "Students talk after practice.",
            "scanner_focus": {"top10_ratio": 0.7},
            "before_context": "Before this paragraph.",
            "after_context": "After this paragraph.",
        },
    )
    prompt = build_safe_band_evidence_repair_prompt(
        section=section,
        current_scores={
            "ai": 33.32,
            "topk_calibrated_risk": 30.586,
            "qualifying_text_ai_density": 40.73,
            "ai_authorship": 33.0,
        },
        current_goal={
            "ai_footprint_gate": {
                "safe_band_thresholds": {
                    "topk_calibrated_risk": 25.0,
                    "qualifying_text_ai_density": 35.0,
                },
                "remaining_ai_footprint_drivers": [
                    {"driver": "topk_calibrated_risk", "value": 30.586, "safe_band": 25.0},
                    {"driver": "qualifying_text_ai_density", "value": 40.73, "safe_band": 35.0},
                ],
            }
        },
        variant_count=2,
        author_proxy_context={
            "active": True,
            "mode": "non_interrupting_author_proxy_draft",
            "review_required": True,
            "required_inputs": ["confirm class observation"],
            "allowed_provenance": ["source_preserved", "inferred_from_draft", "needs_author_confirmation"],
            "review_cards": [{"card_id": "target-01", "target_text": "class observation"}],
            "quality_bar": {"target": "highest_quality_grounded_candidate"},
        },
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "safe_band_evidence_repair"
    assert payload["section"]["selection_reason"] == "top_safe_band_sentence_target"
    assert payload["kpi_contract"]["gaps"]["qualifying_text_ai_density"] == 5.73
    assert payload["materiality_gate"]["minimum_changed_source_sentences"] == 1
    assert payload["author_proxy_context"]["authorship_evidence_contract"]["schema_version"] == "authorship_evidence_contract.v1"
    assert "author_proxy_provenance" in payload["output_schema"]["variants"][0]


def test_v5_safe_band_evidence_repair_section_uses_whole_target_paragraph():
    text = (
        "The first paragraph is not the problem.\n\n"
        "My teaching now includes actions that build emotional safety. "
        "Students talk during this time, and their comments expose what they learned. "
        "From this sharing, I learn from them too.\n\n"
        "The last paragraph is separate."
    )

    section = v5_residual_comb._safe_band_evidence_repair_section(
        text,
        {},
        {
            "eligible_span_density_gate": {
                "top_sentence_targets": [{
                    "sentence_id": "s002",
                    "preview": "Students talk during this time, and their comments expose what they learned.",
                    "top10_ratio": 0.7,
                    "top50_ratio": 0.9,
                    "predictability_risk": 0.5,
                }]
            }
        },
    )

    assert section is not None
    assert section.metadata["paragraph_index"] == 2
    assert section.metadata["selection_reason"] == "top_safe_band_sentence_target"
    assert section.text.startswith("My teaching now includes actions")
    assert section.text.endswith("I learn from them too.")
    assert text[section.start_char:section.end_char] == section.text


def test_v5_safe_band_evidence_repair_section_handles_single_newline_paragraphs():
    text = (
        "Title line\n"
        "The setup paragraph is separate.\n"
        "Students talk during this time, and their comments expose what they learned. "
        "I use those comments to decide what to show again.\n"
        "The last paragraph is separate."
    )

    section = v5_residual_comb._safe_band_evidence_repair_section(
        text,
        {},
        {
            "eligible_span_density_gate": {
                "top_sentence_targets": [{
                    "sentence_id": "s003",
                    "preview": "Students talk during this time, and their comments expose what they learned.",
                }]
            }
        },
    )

    assert section is not None
    assert section.metadata["paragraph_delimiter"] == "single_newline"
    assert section.metadata["paragraph_index"] == 3
    assert section.text.startswith("Students talk during this time")
    assert "The setup paragraph" not in section.text
    assert "The last paragraph" not in section.text


def test_v5_safe_band_evidence_repair_sections_start_with_composite_window():
    text = (
        "Opening paragraph.\n"
        "Students talk during this time, and their comments expose what they learned. "
        "I use those comments to decide what to show again.\n"
        "A middle paragraph links the two target areas.\n"
        "This creates an illusion of simplicity, prompting the thought, \"I can do that.\" "
        "The class then needs sectioning practice.\n"
        "Closing paragraph."
    )

    sections = v5_residual_comb._safe_band_evidence_repair_sections(
        text,
        {},
        {
            "eligible_span_density_gate": {
                "top_sentence_targets": [
                    {
                        "sentence_id": "s002",
                        "preview": "Students talk during this time, and their comments expose what they learned.",
                    },
                    {
                        "sentence_id": "s004",
                        "preview": "This creates an illusion of simplicity, prompting the thought, \"I can do that.\"",
                    },
                ]
            }
        },
        limit=3,
    )

    assert sections[0].metadata["selection_reason"] == "composite_top_safe_band_sections"
    assert "Students talk during this time" in sections[0].text
    assert "This creates an illusion" in sections[0].text
    assert sections[1].metadata["selection_reason"] == "top_safe_band_sentence_target"


def test_v5_safe_band_evidence_pack_prompt_requires_all_section_replacements():
    sections = [
        SectionUnit(
            section_id="safe_band_evidence_repair_t001",
            heading="Safe-band evidence repair",
            text=(
                "Each lesson ends with group reflection. "
                "Students talk during this time, and their comments expose what they learned. "
                "I use those comments to decide what to show again."
            ),
            start_char=10,
            end_char=170,
            paragraph_count=1,
            word_count=25,
            metadata={"target_sentence": "Students talk during this time, and their comments expose what they learned."},
        ),
        SectionUnit(
            section_id="safe_band_evidence_repair_t002",
            heading="Safe-band evidence repair",
            text=(
                "Videos can make the haircut look easy. "
                "This creates an illusion of simplicity. "
                "The classroom task exposes the sectioning problem."
            ),
            start_char=200,
            end_char=330,
            paragraph_count=1,
            word_count=20,
            metadata={"target_sentence": "This creates an illusion of simplicity."},
        ),
    ]
    revision_plan = {
        "evidence_ledger": {
            "sections": [
                {
                    "section_id": "safe_band_evidence_repair_t001",
                    "author_owned_evidence": ["group reflection", "comments decide what to show again"],
                    "weak_or_generic_claims": ["comments expose what they learned"],
                    "protected_anchors": ["group reflection"],
                    "author_review_gaps": [],
                },
                {
                    "section_id": "safe_band_evidence_repair_t002",
                    "author_owned_evidence": ["videos make haircut look easy", "sectioning problem"],
                    "weak_or_generic_claims": ["illusion of simplicity"],
                    "protected_anchors": ["sectioning"],
                    "author_review_gaps": [],
                },
            ],
        },
        "revision_plan": [
            {
                "section_id": "safe_band_evidence_repair_t001",
                "required_moves": ["rebuild reflection route around what students said"],
                "sentences_to_rebuild": ["Students talk during this time"],
            },
            {
                "section_id": "safe_band_evidence_repair_t002",
                "required_moves": ["connect video polish to sectioning obstacle"],
                "sentences_to_rebuild": ["This creates an illusion of simplicity"],
            },
        ],
    }

    prompt = build_safe_band_evidence_pack_prompt(
        sections=sections,
        current_scores={"topk_calibrated_risk": 30.0, "qualifying_text_ai_density": 40.0},
        current_goal={"ai_footprint_gate": {"remaining_ai_footprint_drivers": []}},
        variant_count=2,
        revision_plan=revision_plan,
        author_proxy_context={
            "active": True,
            "mode": "non_interrupting_author_proxy_draft",
            "review_required": True,
            "allowed_provenance": ["source_preserved", "inferred_from_draft", "needs_author_confirmation"],
        },
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "safe_band_evidence_multi_replacement_pack"
    assert payload["architecture_stage"] == "author_proxy_writer_from_evidence_ledger_and_revision_plan"
    assert payload["single_product_judge"]["judge"] == "DraftProof internal safe-band and authorship-integrity scoring"
    assert payload["single_product_judge"]["ignored_as_acceptance_judge"] == "external AI flag score"
    assert payload["revision_compiler_contract"]["schema_version"] == "author_proxy_revision_compiler_contract.v1"
    assert "citation_rhythm" in payload["revision_compiler_contract"]["control_axes"]
    assert payload["evidence_ledger"]["sections"][0]["author_owned_evidence"][0] == "group reflection"
    assert payload["revision_plan"][1]["required_moves"][0] == "connect video polish to sectioning obstacle"
    assert [row["section_id"] for row in payload["sections"]] == [
        "safe_band_evidence_repair_t001",
        "safe_band_evidence_repair_t002",
    ]
    replacements = payload["output_schema"]["variants"][0]["replacements"]
    assert [row["section_id"] for row in replacements] == [
        "safe_band_evidence_repair_t001",
        "safe_band_evidence_repair_t002",
    ]
    assert payload["author_proxy_context"]["authorship_evidence_contract"]["schema_version"] == "authorship_evidence_contract.v1"


def test_v5_safe_band_evidence_pack_uses_density_sections_for_density_blocker(monkeypatch):
    current_text = (
        "Each lesson ends with group reflection. Students talk during this time, and their comments expose what they learned. "
        "I use those comments to decide what to show again.\n\n"
        "Videos can make the haircut look easy. This creates an illusion of simplicity. "
        "The classroom task exposes the sectioning problem."
    )
    density_section = SectionUnit(
        section_id="safe_band_density_section_t001",
        heading="Safe-band density section repair",
        text="Each lesson ends with group reflection. Students talk during this time, and their comments expose what they learned.",
        start_char=0,
        end_char=107,
        paragraph_count=1,
        word_count=15,
        metadata={"selection_reason": "ai_mitigation_density_target_segment"},
    )
    monkeypatch.setattr(
        v5_residual_comb,
        "_safe_band_density_section_repair_sections",
        lambda *_args, **_kwargs: [density_section],
    )

    sections = v5_residual_comb._safe_band_evidence_pack_sections(
        current_text,
        {},
        {
            "ai_footprint_gate": {
                "remaining_ai_footprint_drivers": [
                    {"driver": "qualifying_text_ai_density", "value": 39.2, "safe_band": 35.0},
                ],
            }
        },
        limit=4,
    )

    assert sections == [density_section]
    payload = v5_residual_comb._safe_band_pack_section_payload(density_section, index=1)
    assert payload["materiality_gate"]["contract"] == "density_section_repair"
    assert payload["materiality_gate"]["minimum_changed_source_sentence_ratio"] == 0.4


def test_v5_safe_band_evidence_pack_prompt_lists_density_hard_rejection_contract():
    section = SectionUnit(
        section_id="safe_band_density_section_large",
        heading="Density",
        text=" ".join(f"Source sentence {index} keeps author evidence." for index in range(1, 11)),
        start_char=0,
        end_char=400,
        paragraph_count=1,
        word_count=60,
        metadata={"selection_reason": "ai_mitigation_density_target_segment"},
    )

    prompt = build_safe_band_evidence_pack_prompt(
        sections=[section],
        current_scores={
            "topk_calibrated_risk": 24.3,
            "qualifying_text_ai_density": 38.4,
            "unsafe_cluster_count": 0,
            "risky_window_count": 0,
        },
        current_goal={
            "ai_footprint_gate": {
                "safe_band_thresholds": {
                    "topk_calibrated_risk": 25.0,
                    "qualifying_text_ai_density": 35.0,
                },
                "remaining_ai_footprint_drivers": [
                    {"driver": "qualifying_text_ai_density", "value": 38.4, "safe_band": 35.0},
                ],
            }
        },
        variant_count=1,
    )
    payload = json.loads(prompt.split("\n", 1)[1])
    contract = payload["density_pack_hard_rejection_contract"]

    assert contract["applies"] is True
    assert contract["requirements"][0]["section_id"] == "safe_band_density_section_large"
    assert contract["requirements"][0]["minimum_changed_source_sentences"] >= 4
    assert "whole pack is rejected" in contract["rule"]
    section_contract = payload["sections"][0]["materiality_gate"]["contextual_anchor_contract"]
    assert section_contract["target_contextual_anchor_density"] == 0.45
    assert "low_context_sentence_examples" in section_contract
    assert any("contextual_anchor_contract" in rule for rule in payload["pack_rules"])


def test_v5_safe_band_density_section_prompt_exposes_contextual_anchor_contract():
    section = SectionUnit(
        section_id="safe_band_density_section_s001",
        heading="Density",
        text=(
            "The first broad sentence explains learning in general. "
            "I ask students to hold the comb while they check the sectioning. "
            "The final broad sentence explains why this matters for confidence."
        ),
        start_char=0,
        end_char=170,
        paragraph_count=1,
        word_count=27,
        metadata={"selection_reason": "ai_mitigation_density_target_segment"},
    )

    prompt = v5_residual_comb.build_safe_band_density_section_repair_prompt(
        section=section,
        current_scores={"topk_calibrated_risk": 24.3, "qualifying_text_ai_density": 38.4},
        current_goal={
            "ai_footprint_gate": {
                "remaining_ai_footprint_drivers": [
                    {"driver": "qualifying_text_ai_density", "value": 38.4, "safe_band": 35.0},
                ],
            }
        },
        variant_count=1,
    )
    payload = json.loads(prompt.split("\n", 1)[1])
    contract = payload["section"]["contextual_anchor_contract"]

    assert contract["schema_version"] == "safe_band_density_contextual_anchor_contract.v1"
    assert contract["target_contextual_anchor_density"] == 0.45
    assert contract["additional_contextual_sentences_needed"] >= 1
    assert "contextual_anchor_contract" in " ".join(payload["rules"])


def test_v5_safe_band_evidence_pack_skips_oversized_density_sections(monkeypatch):
    sections = [
        SectionUnit("safe_band_density_section_small1", "Density", "Small first section.", 0, 20, 1, 76, {"selection_reason": "ai_mitigation_density_target_segment"}),
        SectionUnit("safe_band_density_section_large", "Density", "Large section.", 21, 50, 1, 278, {"selection_reason": "ai_mitigation_density_target_segment"}),
        SectionUnit("safe_band_density_section_small2", "Density", "Small second section.", 51, 80, 1, 196, {"selection_reason": "ai_mitigation_density_target_segment"}),
        SectionUnit("safe_band_density_section_small3", "Density", "Small third section.", 81, 110, 1, 100, {"selection_reason": "ai_mitigation_density_target_segment"}),
    ]
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_sections", lambda *_args, **_kwargs: sections)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_evidence_pack_max_section_words", lambda: 220)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_evidence_pack_max_source_words", lambda: 420)

    selected = v5_residual_comb._safe_band_evidence_pack_sections(
        "source text",
        {},
        {
            "ai_footprint_gate": {
                "remaining_ai_footprint_drivers": [
                    {"driver": "qualifying_text_ai_density", "value": 39.2, "safe_band": 35.0},
                ],
            }
        },
        limit=4,
    )

    assert [section.section_id for section in selected] == [
        "safe_band_density_section_small1",
        "safe_band_density_section_small2",
        "safe_band_density_section_small3",
    ]
    assert sum(section.word_count for section in selected) == 372


def test_v5_safe_band_evidence_pack_defaults_cover_large_density_windows(monkeypatch):
    sections = [
        SectionUnit("safe_band_density_section_intro", "Density", "Intro density section.", 0, 20, 1, 74, {"selection_reason": "ai_mitigation_density_target_segment"}),
        SectionUnit("safe_band_density_section_large", "Density", "Large density section.", 21, 50, 1, 278, {"selection_reason": "ai_mitigation_density_target_segment"}),
        SectionUnit("safe_band_density_section_method", "Density", "Method density section.", 51, 80, 1, 196, {"selection_reason": "ai_mitigation_density_target_segment"}),
        SectionUnit("safe_band_density_section_video", "Density", "Video density section.", 81, 110, 1, 100, {"selection_reason": "ai_mitigation_density_target_segment"}),
    ]
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_PACK_MAX_SECTION_WORDS", raising=False)
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_PACK_MAX_SOURCE_WORDS", raising=False)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_sections", lambda *_args, **_kwargs: sections)

    selected = v5_residual_comb._safe_band_evidence_pack_sections(
        "source text",
        {},
        {
            "ai_footprint_gate": {
                "remaining_ai_footprint_drivers": [
                    {"driver": "qualifying_text_ai_density", "value": 38.44, "safe_band": 35.0},
                ],
            }
        },
        limit=4,
    )

    assert [section.section_id for section in selected] == [
        "safe_band_density_section_intro",
        "safe_band_density_section_large",
        "safe_band_density_section_method",
        "safe_band_density_section_video",
    ]
    assert sum(section.word_count for section in selected) == 648


def test_v5_safe_band_author_proxy_revision_plan_prompt_precedes_writer():
    sections = [
        SectionUnit(
            section_id="safe_band_evidence_repair_t001",
            heading="Safe-band evidence repair",
            text=(
                "Each lesson ends with group reflection. "
                "Students talk during this time, and their comments expose what they learned. "
                "I use those comments to decide what to show again."
            ),
            start_char=10,
            end_char=170,
            paragraph_count=1,
            word_count=25,
            metadata={
                "target_sentence": "Students talk during this time, and their comments expose what they learned.",
                "before_context": "The class struggled with projection.",
                "after_context": "The next lesson returns to sectioning.",
            },
        ),
        SectionUnit(
            section_id="safe_band_evidence_repair_t002",
            heading="Safe-band evidence repair",
            text=(
                "Videos can make the haircut look easy. "
                "This creates an illusion of simplicity. "
                "The classroom task exposes the sectioning problem."
            ),
            start_char=200,
            end_char=330,
            paragraph_count=1,
            word_count=20,
            metadata={"target_sentence": "This creates an illusion of simplicity."},
        ),
    ]

    prompt = build_safe_band_author_proxy_revision_plan_prompt(
        sections=sections,
        current_scores={"topk_calibrated_risk": 30.0, "qualifying_text_ai_density": 40.0},
        current_goal={
            "ai_footprint_gate": {
                "safe_band_thresholds": {"topk_calibrated_risk": 25.0, "qualifying_text_ai_density": 35.0},
                "remaining_ai_footprint_drivers": [
                    {"driver": "topk_calibrated_risk", "value": 30.0, "safe_band": 25.0},
                ],
            }
        },
        author_proxy_context={
            "active": True,
            "mode": "non_interrupting_author_proxy_draft",
            "review_required": True,
            "allowed_provenance": ["source_preserved", "inferred_from_draft", "needs_author_confirmation"],
        },
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "safe_band_author_proxy_evidence_ledger_and_revision_plan"
    assert payload["architecture_stage"] == "evidence_ledger_then_revision_plan_before_writer"
    assert payload["single_product_judge"]["ignored_as_acceptance_judge"] == "external AI flag score"
    assert payload["revision_compiler_contract"]["schema_version"] == "author_proxy_revision_compiler_contract.v1"
    assert "sentence_shape" in payload["revision_compiler_contract"]["control_axes"]
    assert payload["kpi_contract"]["gaps"]["topk_calibrated_risk"] == 5.0
    assert payload["sections"][0]["materiality_gate"]["minimum_changed_source_sentences"] == 2
    assert payload["sections"][0]["revision_compiler_contract"]["source_profile"]["sentence_count"] == 3
    assert payload["output_schema"]["evidence_ledger"]["sections"][0]["author_owned_evidence"]
    assert payload["output_schema"]["revision_plan"][0]["required_moves"]
    assert payload["output_schema"]["revision_plan"][0]["prose_shape_plan"]
    assert payload["output_schema"]["revision_plan"][0]["abstraction_density_plan"]
    assert payload["output_schema"]["revision_plan"][0]["citation_rhythm_plan"]
    assert payload["output_schema"]["revision_plan"][0]["closure_plan"]
    assert payload["author_proxy_context"]["mode"] == "non_interrupting_author_proxy_draft"


def test_v5_author_proxy_revision_compiler_audit_catches_polished_wrapper_risk():
    audit = v5_residual_comb._author_proxy_revision_compiler_audit(
        source_text=(
            "I watched the student stop at the mirror. "
            "Her sectioning slipped when she turned the comb. "
            "I changed the next demonstration to slow that step down."
        ),
        candidate_text=(
            "This experience demonstrates the importance of effective teaching practice. "
            "According to Smith (2024), reflective pedagogy supports student learning. "
            "As a result, the overall process highlights the significance of inclusive educational development."
        ),
    )

    assert audit["schema_version"] == "author_proxy_revision_compiler_audit.v1"
    assert audit["passed"] is False
    assert "citation_rhythm_not_expanded" in audit["failed_checks"]
    assert "closure_not_polished_wrapper" in audit["failed_checks"]


def test_v5_safe_band_pack_rejects_candidate_that_only_wraps_anchors():
    source = (
        "I watched the student stop at the mirror. "
        "Her sectioning slipped when she turned the comb. "
        "I changed the next demonstration to slow that step down."
    )
    section = SectionUnit(
        section_id="safe_band_evidence_repair_t001",
        heading="Safe-band evidence repair",
        text=source,
        start_char=0,
        end_char=len(source),
        paragraph_count=1,
        word_count=24,
        metadata={"target_sentence": "Her sectioning slipped when she turned the comb."},
    )
    candidate, status, materiality = v5_residual_comb._apply_safe_band_evidence_pack_variant(
        current_text=source,
        sections=[section],
        variant={
            "variant_id": "v1",
            "replacements": [
                {
                    "section_id": "safe_band_evidence_repair_t001",
                    "text": (
                        "This classroom moment demonstrates the importance of effective practice. "
                        "According to Smith (2024), reflective pedagogy supports student learning. "
                        "As a result, the process highlights the significance of inclusive educational development."
                    ),
                }
            ],
        },
    )

    assert candidate
    assert status["applied"] is False
    assert materiality["passed"] is False
    assert materiality["sections"][0]["reason"] == "candidate_revision_compiler_failed"
    assert materiality["sections"][0]["revision_compiler_audit"]["passed"] is False


def test_v5_safe_band_author_proxy_revision_plan_sanitizer_requires_all_sections():
    sections = [
        SectionUnit(
            section_id="s1",
            heading="A",
            text="One source sentence. Another sentence.",
            start_char=0,
            end_char=37,
            paragraph_count=1,
            word_count=5,
            metadata={},
        ),
        SectionUnit(
            section_id="s2",
            heading="B",
            text="Third source sentence. Fourth sentence.",
            start_char=38,
            end_char=75,
            paragraph_count=1,
            word_count=5,
            metadata={},
        ),
    ]
    incomplete = {
        "evidence_ledger": {
            "sections": [
                {
                    "section_id": "s1",
                    "author_owned_evidence": ["One source sentence"],
                    "weak_or_generic_claims": [],
                    "protected_anchors": [],
                    "author_review_gaps": [],
                }
            ]
        },
        "revision_plan": [
            {
                "section_id": "s1",
                "section_job": "repair first section",
                "required_moves": ["change route"],
                "sentences_to_rebuild": ["One source sentence"],
                "claim_narrowing": [],
                "materiality_requirement": "change two sentences",
                "author_review_items": [],
            }
        ],
    }
    plan, diagnostics = v5_residual_comb._sanitize_safe_band_author_proxy_revision_plan(
        incomplete,
        sections=sections,
    )

    assert plan is None
    assert diagnostics["reason"] == "plan_missing_required_sections"
    assert diagnostics["missing_ledger_section_ids"] == ["s2"]
    assert diagnostics["missing_plan_section_ids"] == ["s2"]


def test_v5_safe_band_evidence_pack_response_keeps_author_proxy_fields_compact():
    schema = v5_residual_comb._safe_band_evidence_pack_response_format(
        2,
        3,
        include_author_proxy_fields=True,
    )
    variant_schema = schema["json_schema"]["schema"]["properties"]["variants"]["items"]

    assert variant_schema["properties"]["author_proxy_provenance"]["maxItems"] == 4
    assert variant_schema["properties"]["author_review_items"]["maxItems"] == 4


def test_v5_safe_band_evidence_pack_composite_combines_material_sections():
    first = SectionUnit(
        section_id="s1",
        heading="Safe-band evidence repair",
        text="First source sentence stays here. Target sentence must change. Third source sentence stays.",
        start_char=0,
        end_char=82,
        paragraph_count=1,
        word_count=11,
        metadata={"target_sentence": "Target sentence must change."},
    )
    second = SectionUnit(
        section_id="s2",
        heading="Safe-band evidence repair",
        text="Fourth source sentence stays here. Another target must change. Sixth source sentence stays.",
        start_char=83,
        end_char=169,
        paragraph_count=1,
        word_count=11,
        metadata={"target_sentence": "Another target must change."},
    )
    variants = [
        {
            "variant_id": "v1",
            "replacements": [
                {
                    "section_id": "s1",
                    "text": "First idea is rebuilt around the author action. The target is recast with a new route. The closing sentence changes too.",
                },
                {"section_id": "s2", "text": second.text},
            ],
            "author_proxy_provenance": [{"item_id": "p1", "target_text": "s1", "generated_text": "rebuilt", "provenance": "inferred_from_draft"}],
            "author_review_items": [],
        },
        {
            "variant_id": "v2",
            "replacements": [
                {"section_id": "s1", "text": first.text},
                {
                    "section_id": "s2",
                    "text": "Fourth idea is rebuilt around the teaching choice. The second target is recast. The ending changes as well.",
                },
            ],
            "author_proxy_provenance": [{"item_id": "p2", "target_text": "s2", "generated_text": "rebuilt", "provenance": "inferred_from_draft"}],
            "author_review_items": [],
        },
    ]

    composite = v5_residual_comb._safe_band_evidence_pack_composite_variant(
        sections=[first, second],
        variants=variants,
    )

    assert composite["variant_id"] == "composite_material_pack"
    assert composite["source_variant_ids"] == ["v1", "v2"]
    assert composite["replacements"][0]["text"] != first.text
    assert composite["replacements"][1]["text"] != second.text
    assert [item["target_text"] for item in composite["author_proxy_provenance"]] == ["s1", "s2"]


def test_v5_safe_band_evidence_pack_partial_material_variant_keeps_only_passing_sections(monkeypatch):
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_min_word_ratio", lambda: 0.5)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_max_word_ratio", lambda: 1.8)
    first = SectionUnit(
        section_id="safe_band_density_section_s1",
        heading="Density",
        text="First source sentence stays. Target sentence must change. Third source sentence stays.",
        start_char=0,
        end_char=75,
        paragraph_count=1,
        word_count=10,
        metadata={"selection_reason": "ai_mitigation_density_target_segment", "target_sentence": "Target sentence must change."},
    )
    second = SectionUnit(
        section_id="safe_band_density_section_s2",
        heading="Density",
        text="Fourth source sentence stays. Another target must change. Sixth source sentence stays.",
        start_char=76,
        end_char=150,
        paragraph_count=1,
        word_count=10,
        metadata={"selection_reason": "ai_mitigation_density_target_segment", "target_sentence": "Another target must change."},
    )
    variants = [
        {
            "variant_id": "v1",
            "replacements": [
                {
                    "section_id": "safe_band_density_section_s1",
                    "text": "The first idea is rebuilt around the author action. The target changes route. The ending changes too.",
                },
                {"section_id": "safe_band_density_section_s2", "text": second.text},
            ],
        },
        {
            "variant_id": "v2",
            "replacements": [
                {"section_id": "safe_band_density_section_s1", "text": first.text},
                {
                    "section_id": "safe_band_density_section_s2",
                    "text": "The fourth idea is rebuilt around the teaching choice. The second target is recast. The ending changes as well.",
                },
            ],
        },
    ]

    partial = v5_residual_comb._safe_band_evidence_pack_partial_material_variant(
        sections=[first, second],
        variants=variants,
    )

    assert partial["variant_id"] == "partial_material_pack"
    assert partial["partial_pack"] is True
    assert [row["section_id"] for row in partial["replacements"]] == [
        "safe_band_density_section_s1",
        "safe_band_density_section_s2",
    ]


def test_v5_safe_band_evidence_pack_section_probe_variants_deduplicate_by_section_text():
    section = SectionUnit(
        section_id="safe_band_density_section_s1",
        heading="Density",
        text="The source sentence needs a new route. The second sentence should also move.",
        start_char=0,
        end_char=72,
        paragraph_count=1,
        word_count=12,
        metadata={"selection_reason": "ai_mitigation_density_target_segment"},
    )
    variants = [
        {
            "variant_id": "v1",
            "replacements": [
                {
                    "section_id": "safe_band_density_section_s1",
                    "text": "I rebuild the first point around the class moment. The second point now moves with it.",
                }
            ],
        },
        {
            "variant_id": "v2",
            "replacements": [
                {
                    "section_id": "safe_band_density_section_s1",
                    "text": "I rebuild the first point around the class moment. The second point now moves with it.",
                }
            ],
        },
    ]

    probes = v5_residual_comb._safe_band_evidence_pack_section_probe_variants(
        sections=[section],
        variants=variants,
    )

    assert len(probes) == 1
    assert probes[0]["partial_pack"] is True
    assert probes[0]["replacements"] == [
        {
            "section_id": "safe_band_density_section_s1",
            "text": "I rebuild the first point around the class moment. The second point now moves with it.",
        }
    ]


def test_v5_safe_band_evidence_pack_section_probes_add_length_normalized_density_variant(monkeypatch):
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_min_word_ratio", lambda: 0.5)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_max_word_ratio", lambda: 1.5)
    section = SectionUnit(
        section_id="safe_band_density_section_s1",
        heading="Density",
        text=(
            "First source idea stays here for class. "
            "Target sentence must change route today. "
            "Third source idea stays here too."
        ),
        start_char=0,
        end_char=103,
        paragraph_count=1,
        word_count=17,
        metadata={
            "selection_reason": "ai_mitigation_density_target_segment",
            "target_sentence": "Target sentence must change route today.",
        },
    )
    variants = [
        {
            "variant_id": "v1",
            "replacements": [
                {
                    "section_id": "safe_band_density_section_s1",
                    "text": (
                        "I start with the class moment before naming the problem. "
                        "The target idea now follows the student's hesitation. "
                        "The closing point links it back to practice. "
                        "Extra background sentence adds unnecessary detail beyond the source."
                    ),
                }
            ],
        }
    ]

    probes = v5_residual_comb._safe_band_evidence_pack_section_probe_variants(
        sections=[section],
        variants=variants,
    )
    normalized = [probe for probe in probes if probe["variant_id"].endswith("_length_normalized")]

    assert len(normalized) == 1
    replacement = normalized[0]["replacements"][0]["text"]
    assert v5_residual_comb.word_count(replacement) <= round(v5_residual_comb.word_count(section.text) * 1.5)
    materiality = v5_residual_comb._safe_band_density_section_repair_materiality(
        source_text=section.text,
        candidate_text=replacement,
        target_sentence=section.metadata["target_sentence"],
    )
    assert materiality["passed"] is True


def test_v5_safe_band_evidence_pack_scored_section_composite_uses_only_internal_judge_movers():
    current_scores = {
        "ai": 30.63,
        "topk_calibrated_risk": 24.357,
        "qualifying_text_ai_density": 38.44,
        "ai_authorship": 31.0,
        "unsafe_cluster_count": 0,
        "unsafe_word_ratio": 0.0,
        "risky_window_count": 0,
    }
    safe_row = {
        "variant_id": "section_probe_safe_band_density_section_s1_v1",
        "text": "The author-owned replacement moves the density section.",
        "apply_status": {
            "applied": True,
            "applied_sections": [{"section_id": "safe_band_density_section_s1"}],
        },
        "safe_band_evidence_materiality": {
            "passed": True,
            "sections": [{"contract": "density_section_repair", "passed": True}],
        },
        "safe_band_evidence_pack_materiality": {
            "passed": True,
            "sections": [{"contract": "density_section_repair", "passed": True}],
        },
        "safe_band_quality_materiality": {"passed": True},
        "scores": {
            "ai": 30.5,
            "topk_calibrated_risk": 24.2,
            "qualifying_text_ai_density": 37.4,
            "ai_authorship": 31.0,
            "unsafe_cluster_count": 0,
            "unsafe_word_ratio": 0.0,
            "risky_window_count": 0,
        },
        "incremental": {
            "ai_delta": 0.13,
            "topk_calibrated_risk_delta": 0.157,
            "qualifying_text_ai_density_delta": 1.04,
            "ai_authorship_delta": 0.0,
            "unsafe_cluster_count_delta": 0.0,
            "unsafe_word_ratio_delta": 0.0,
            "risky_window_count_delta": 0.0,
        },
    }
    unsafe_regression = {
        **safe_row,
        "variant_id": "section_probe_safe_band_density_section_s2_v1",
        "text": "A replacement that looks material but introduces an unsafe cluster.",
        "apply_status": {
            "applied": True,
            "applied_sections": [{"section_id": "safe_band_density_section_s2"}],
        },
        "scores": {
            "ai": 30.2,
            "topk_calibrated_risk": 23.9,
            "qualifying_text_ai_density": 37.1,
            "ai_authorship": 31.0,
            "unsafe_cluster_count": 1,
            "unsafe_word_ratio": 4.0,
            "risky_window_count": 0,
        },
        "incremental": {
            "ai_delta": 0.43,
            "topk_calibrated_risk_delta": 0.457,
            "qualifying_text_ai_density_delta": 1.34,
            "ai_authorship_delta": 0.0,
            "unsafe_cluster_count_delta": -1.0,
            "unsafe_word_ratio_delta": -4.0,
            "risky_window_count_delta": 0.0,
        },
    }
    small_safe_mover = {
        **safe_row,
        "variant_id": "section_probe_safe_band_density_section_s3_v1",
        "text": "A small replacement that is safe enough to compose.",
        "apply_status": {
            "applied": True,
            "applied_sections": [{"section_id": "safe_band_density_section_s3"}],
        },
        "scores": {
            "ai": 30.61,
            "topk_calibrated_risk": 24.32,
            "qualifying_text_ai_density": 38.4,
            "ai_authorship": 31.0,
            "unsafe_cluster_count": 0,
            "unsafe_word_ratio": 0.0,
            "risky_window_count": 0,
        },
        "incremental": {
            "ai_delta": 0.02,
            "topk_calibrated_risk_delta": 0.037,
            "qualifying_text_ai_density_delta": 0.04,
            "ai_authorship_delta": 0.0,
            "unsafe_cluster_count_delta": 0.0,
            "unsafe_word_ratio_delta": 0.0,
            "risky_window_count_delta": 0.0,
        },
    }

    composite = v5_residual_comb._safe_band_evidence_pack_scored_section_composite_variant(
        [safe_row, unsafe_regression, small_safe_mover],
        current_scores=current_scores,
    )

    assert composite["variant_id"] == "scored_section_composite"
    assert composite["partial_pack"] is True
    assert composite["replacements"] == [
        {
            "section_id": "safe_band_density_section_s1",
            "text": "The author-owned replacement moves the density section.",
        },
        {
            "section_id": "safe_band_density_section_s3",
            "text": "A small replacement that is safe enough to compose.",
        },
    ]


def test_v5_safe_band_controlled_operation_deletes_exact_target_sentence(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_SAFE_BAND_CONTROLLED_OPERATION_MIN_WORD_RATIO", "0.5")
    current_text = (
        "The reflection starts after class. "
        "Students talk during this time, and their comments expose what they learned. "
        "I use the comments to choose the next demonstration."
    )
    variant = {
        "variant_id": "delete_t001",
        "operation": "delete_exact_target_sentence",
        "target_id": "t001",
        "sentence": "Students talk during this time, and their comments expose what they learned.",
    }

    candidate, apply_status, materiality = v5_residual_comb._apply_safe_band_controlled_operation_variant(
        current_text=current_text,
        variant=variant,
    )

    assert apply_status["applied"] is True
    assert apply_status["scope"] == "safe_band_controlled_operation"
    assert materiality["passed"] is True
    assert "Students talk during this time" not in candidate
    assert "The reflection starts after class." in candidate
    assert "I use the comments" in candidate


def test_v5_safe_band_controlled_operation_keeps_suffix_after_strong_punctuation(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_SAFE_BAND_CONTROLLED_OPERATION_MIN_WORD_RATIO", "0.5")
    current_text = (
        "The reflection starts after class. "
        "In standard class activities, it's genuinely hard to see what he's capable of — he tends to fade into the background. "
        "I use the comments to choose the next demonstration."
    )
    sentence = "In standard class activities, it's genuinely hard to see what he's capable of — he tends to fade into the background."
    variant = {
        "variant_id": "suffix_t002",
        "operation": "keep_suffix_after_strong_punctuation",
        "target_id": "t002",
        "sentence": sentence,
        "replacement": "He tends to fade into the background.",
    }

    candidate, apply_status, materiality = v5_residual_comb._apply_safe_band_controlled_operation_variant(
        current_text=current_text,
        variant=variant,
    )

    assert apply_status["applied"] is True
    assert apply_status["operation"] == "keep_suffix_after_strong_punctuation"
    assert materiality["reason"] == "controlled_strong_punctuation_suffix_kept"
    assert "standard class activities" not in candidate
    assert "He tends to fade into the background." in candidate


def test_v5_safe_band_controlled_operation_suffix_candidate_rejects_short_fragment(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_SAFE_BAND_CONTROLLED_OPERATION_MIN_SUFFIX_WORDS", "4")

    assert v5_residual_comb._strong_punctuation_suffix_candidate("Broad setup — done.") is None


def test_v5_safe_band_controlled_operation_variants_include_suffix_candidate():
    sentence = "The setup is broad — the specific result remains visible."
    variants = v5_residual_comb._safe_band_controlled_operation_variants(
        current_text=f"Opening. {sentence} Closing.",
        targets=[{"target_id": "t001", "sentence": sentence}],
    )

    operations = {variant["operation"] for variant in variants}
    suffix_variant = next(variant for variant in variants if variant["operation"] == "keep_suffix_after_strong_punctuation")
    assert operations == {"delete_exact_target_sentence", "keep_suffix_after_strong_punctuation"}
    assert suffix_variant["replacement"] == "The specific result remains visible."


def test_v5_safe_band_controlled_operation_targets_include_density_segments():
    broad_sentence = (
        "Demonstrations and scaffolding support this shift, but their value lies in enabling the necessary cycle "
        "where students try, get it wrong, receive feedback, and try again."
    )
    text = f"Concrete source detail stays. {broad_sentence}"
    targets = v5_residual_comb._safe_band_controlled_operation_targets(
        text,
        {
            "ai_mitigation": {
                "target_segments": [
                    {
                        "segment_id": "s026",
                        "sentence_id": "s026",
                        "text": broad_sentence,
                        "lever": "predictability_reduction",
                        "bucket": "auto_candidate",
                        "user_input_needed": "none if no evidence/source gap is attached",
                        "primary_signal": {
                            "key": "ai_likelihood",
                            "score": 40,
                            "rewrite_permission": "suggestion_only",
                        },
                    },
                    {
                        "segment_id": "s001",
                        "sentence_id": "s001",
                        "text": "Concrete source detail stays.",
                        "lever": "reasoning_continuity",
                        "bucket": "structure_revision",
                        "user_input_needed": "the missing connection between adjacent claims",
                        "primary_signal": {
                            "key": "authorship_risk",
                            "score": 73,
                            "rewrite_permission": "suggestion_only",
                        },
                    },
                ]
            }
        },
        {
            "ai_footprint_gate": {
                "safe_band_thresholds": {"topk_calibrated_risk": 25.0, "qualifying_text_ai_density": 35.0},
                "remaining_ai_footprint_drivers": [
                    {"driver": "qualifying_text_ai_density", "value": 38.41, "safe_band": 35.0},
                ],
            }
        },
        current_scores={
            "topk_calibrated_risk": 22.296,
            "qualifying_text_ai_density": 38.41,
            "unsafe_cluster_count": 0,
            "risky_window_count": 0,
        },
    )

    assert targets[0]["source"] == "ai_mitigation_density_target_segments"
    assert targets[0]["sentence"] == broad_sentence
    assert all(target["sentence"] != "Concrete source detail stays." for target in targets)


def test_v5_safe_band_controlled_operation_rejects_non_unique_target_sentence():
    current_text = "Repeat this sentence. Keep middle. Repeat this sentence."
    variant = {
        "variant_id": "delete_t001",
        "operation": "delete_exact_target_sentence",
        "target_id": "t001",
        "sentence": "Repeat this sentence.",
    }

    _candidate, apply_status, materiality = v5_residual_comb._apply_safe_band_controlled_operation_variant(
        current_text=current_text,
        variant=variant,
    )

    assert apply_status["applied"] is False
    assert apply_status["reason"] == "target_sentence_not_unique"
    assert materiality["match_count"] == 2


def test_v5_safe_band_controlled_operation_best_candidate_uses_existing_movement_gate():
    current_scores = {"topk_calibrated_risk": 30.0, "qualifying_text_ai_density": 40.0}
    row = {
        "apply_status": {"applied": True},
        "incremental": {
            "ai_delta": 0.1,
            "ai_authorship_delta": 0.0,
            "unsafe_cluster_count_delta": 1.0,
            "unsafe_word_ratio_delta": 2.0,
            "risky_window_count_delta": 0.0,
        },
        "scores": {"topk_calibrated_risk": 29.6, "qualifying_text_ai_density": 39.9},
        "safe_band_evidence_materiality": {"passed": True},
    }

    assert _has_safe_band_evidence_repair_movement(row, current_scores=current_scores)
    assert _best_safe_band_evidence_repair_candidate([row], current_scores=current_scores) is row


def test_v5_safe_band_evidence_repair_allows_tiny_unsafe_word_ratio_regression(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_REPAIR_UNSAFE_WORD_RATIO_REGRESSION_TOLERANCE", "0.1")
    current_scores = {"topk_calibrated_risk": 30.0, "qualifying_text_ai_density": 40.0}
    row = {
        "apply_status": {"applied": True},
        "incremental": {
            "ai_delta": 0.1,
            "ai_authorship_delta": 0.0,
            "unsafe_cluster_count_delta": 0.0,
            "unsafe_word_ratio_delta": -0.05,
            "risky_window_count_delta": 0.0,
        },
        "scores": {"topk_calibrated_risk": 29.7, "qualifying_text_ai_density": 39.9},
        "safe_band_evidence_materiality": {"passed": True},
    }
    too_much_regression = {
        **row,
        "incremental": {
            **row["incremental"],
            "unsafe_word_ratio_delta": -0.2,
        },
    }

    assert _has_safe_band_evidence_repair_movement(row, current_scores=current_scores)
    assert not _has_safe_band_evidence_repair_movement(too_much_regression, current_scores=current_scores)


def test_v5_density_checkpoint_accepts_topk_regression_when_topk_remains_safe(monkeypatch):
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_checkpoint_min_density_delta", lambda: 0.5)
    current_scores = {
        "ai": 30.49,
        "topk_calibrated_risk": 23.916,
        "qualifying_text_ai_density": 38.3,
        "unsafe_cluster_count": 0,
        "risky_window_count": 0,
    }
    row = {
        "apply_status": {"applied": True},
        "safe_band_density_section_materiality": {"passed": True},
        "scores": {
            "ai": 30.09,
            "topk_calibrated_risk": 24.449,
            "qualifying_text_ai_density": 36.29,
            "ai_authorship": 30.0,
            "unsafe_cluster_count": 0,
            "unsafe_word_ratio": 0.0,
            "risky_window_count": 0,
        },
        "candidate_goal": {
            "ai_footprint_gate": {
                "safe_band_thresholds": {
                    "topk_calibrated_risk": 25.0,
                    "qualifying_text_ai_density": 35.0,
                    "ai_authorship": 35.0,
                },
                "remaining_ai_footprint_drivers": [
                    {"driver": "qualifying_text_ai_density", "value": 36.29, "safe_band": 35.0},
                ],
            }
        },
        "incremental": {
            "ai_delta": 0.4,
            "ai_authorship_delta": 0.0,
            "topk_calibrated_risk_delta": -0.533,
            "qualifying_text_ai_density_delta": 2.01,
            "unsafe_cluster_count_delta": 0.0,
            "unsafe_word_ratio_delta": 0.0,
            "risky_window_count_delta": 0.0,
        },
    }

    assert v5_residual_comb._has_density_safe_band_checkpoint_movement(row, current_scores=current_scores)


def test_v5_safe_band_controlled_operation_loop_rescores_after_acceptance(monkeypatch, tmp_path):
    accepted_checkpoints = []
    current_scores = {"ai": 33.0, "topk_calibrated_risk": 30.0, "qualifying_text_ai_density": 40.0}
    current_goal = {
        "ai_footprint_gate": {
            "remaining_ai_footprint_drivers": [
                {"driver": "topk_calibrated_risk", "value": 30.0, "safe_band": 25.0},
                {"driver": "qualifying_text_ai_density", "value": 40.0, "safe_band": 35.0},
            ]
        }
    }

    monkeypatch.setattr(v5_residual_comb, "_safe_band_controlled_operation_round_limit", lambda: 3)
    monkeypatch.setattr(
        v5_residual_comb,
        "_safe_band_controlled_operation_targets",
        lambda text, _report, _goal, **_kwargs: [{"target_id": "t001", "sentence": "Before target."}] if text == "before text" else [],
    )
    monkeypatch.setattr(
        v5_residual_comb,
        "_safe_band_controlled_operation_variants",
        lambda current_text, targets: [{"variant_id": "delete_t001", "operation": "delete_exact_target_sentence", "sentence": "Before target."}]
        if targets else [],
    )

    def fake_score(**_kwargs):
        return {
            "section_id": "safe_band_controlled_operation",
            "variant_id": "delete_t001",
            "apply_status": {"applied": True},
            "scores": {"ai": 32.9, "topk_calibrated_risk": 29.6, "qualifying_text_ai_density": 39.8},
            "incremental": {
                "ai_delta": 0.1,
                "ai_authorship_delta": 0.0,
                "unsafe_cluster_count_delta": 0.0,
                "unsafe_word_ratio_delta": 0.0,
                "risky_window_count_delta": 0.0,
            },
            "candidate_text": "after one",
            "candidate_report": {},
            "candidate_goal": {
                "ai_footprint_gate": {
                    "remaining_ai_footprint_drivers": [
                        {"driver": "topk_calibrated_risk", "value": 29.6, "safe_band": 25.0},
                        {"driver": "qualifying_text_ai_density", "value": 39.8, "safe_band": 35.0},
                    ]
                }
            },
            "safe_band_evidence_materiality": {"passed": True},
            "safe_band_controlled_operation_materiality": {"passed": True},
        }

    monkeypatch.setattr(v5_residual_comb, "_score_safe_band_controlled_operation_variant", fake_score)

    text, _report, _goal, scores, rounds, _best = v5_residual_comb._run_safe_band_controlled_operation_loop(
        original_text="original text",
        baseline_report={},
        baseline_scores=current_scores,
        current_text="before text",
        current_report={},
        current_goal=current_goal,
        current_scores=current_scores,
        output_dir=tmp_path,
        global_best_candidate=None,
        accepted_checkpoint_callback=accepted_checkpoints.append,
    )

    assert text == "after one"
    assert scores["topk_calibrated_risk"] == 29.6
    assert [row["status"] for row in rounds] == ["accepted", "skipped"]
    assert rounds[1]["reason"] == "no_safe_band_controlled_operation_targets"
    assert accepted_checkpoints[0]["round"] == 1


def test_v5_safe_band_sentence_replacement_loop_accepts_scanner_gap_movement(monkeypatch, tmp_path):
    accepted_checkpoints = []
    current_scores = {"ai": 33.0, "topk_calibrated_risk": 30.0, "qualifying_text_ai_density": 40.0}
    current_goal = {
        "ai_footprint_gate": {
            "remaining_ai_footprint_drivers": [
                {"driver": "topk_calibrated_risk", "value": 30.0, "safe_band": 25.0},
                {"driver": "qualifying_text_ai_density", "value": 40.0, "safe_band": 35.0},
            ]
        }
    }

    monkeypatch.setattr(v5_residual_comb, "_safe_band_sentence_replacement_round_limit", lambda: 1)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_sentence_replacement_target_limit", lambda: 2)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_sentence_replacement_variant_count", lambda: 2)
    monkeypatch.setattr(
        v5_residual_comb,
        "_safe_band_controlled_operation_targets",
        lambda *_args, **_kwargs: [
            {"target_id": "t001", "sentence": "The generic sentence remains."},
            {"target_id": "t002", "sentence": "The second generic sentence remains."},
        ],
    )
    monkeypatch.setattr(
        v5_residual_comb,
        "generate_safe_band_sentence_replacement_variants",
        lambda **_kwargs: ([{"variant_id": "v1", "repairs": []}], {"status": "ok"}, "prompt", "completion"),
    )

    def fake_score(**_kwargs):
        return {
            "section_id": "safe_band_sentence_replacement",
            "variant_id": "v1",
            "apply_status": {"applied": True},
            "scores": {"ai": 32.8, "topk_calibrated_risk": 29.4, "qualifying_text_ai_density": 39.7},
            "incremental": {
                "ai_delta": 0.2,
                "ai_authorship_delta": 0.0,
                "unsafe_cluster_count_delta": 0.0,
                "unsafe_word_ratio_delta": 0.0,
                "risky_window_count_delta": 0.0,
            },
            "candidate_text": "replacement text",
            "candidate_report": {},
            "candidate_goal": {
                "ai_footprint_gate": {
                    "remaining_ai_footprint_drivers": [
                        {"driver": "topk_calibrated_risk", "value": 29.4, "safe_band": 25.0},
                        {"driver": "qualifying_text_ai_density", "value": 39.7, "safe_band": 35.0},
                    ]
                }
            },
        }

    monkeypatch.setattr(v5_residual_comb, "_score_final_topk_sentence_route_variant", fake_score)

    text, _report, _goal, scores, rounds, _best = v5_residual_comb._run_safe_band_sentence_replacement_loop(
        original_text="original text",
        baseline_report={},
        baseline_scores=current_scores,
        current_text="current text",
        current_report={},
        current_goal=current_goal,
        current_scores=current_scores,
        gateway=object(),
        output_dir=tmp_path,
        global_best_candidate=None,
        accepted_checkpoint_callback=accepted_checkpoints.append,
    )

    assert text == "replacement text"
    assert scores["topk_calibrated_risk"] == 29.4
    assert rounds[0]["status"] == "accepted"
    assert rounds[0]["phase"] == "safe_band_sentence_replacement"
    assert accepted_checkpoints[0]["phase"] == "safe_band_sentence_replacement"


def test_v5_safe_band_sentence_replacement_defaults_are_bounded_density_pass():
    assert v5_residual_comb._safe_band_sentence_replacement_round_limit() == 3
    assert v5_residual_comb._safe_band_sentence_replacement_variant_count() == 5


def test_v5_safe_band_density_section_repair_runs_only_when_density_is_remaining_blocker():
    current_goal = {
        "ai_footprint_gate": {
            "safe_band_thresholds": {"topk_calibrated_risk": 25.0, "qualifying_text_ai_density": 35.0},
            "remaining_ai_footprint_drivers": [
                {"driver": "qualifying_text_ai_density", "value": 38.41, "safe_band": 35.0},
            ],
        }
    }

    assert v5_residual_comb._safe_band_density_section_repair_should_run(
        current_scores={
            "topk_calibrated_risk": 22.296,
            "qualifying_text_ai_density": 38.41,
            "unsafe_cluster_count": 0,
            "risky_window_count": 0,
        },
        current_goal=current_goal,
    )
    assert not v5_residual_comb._safe_band_density_section_repair_should_run(
        current_scores={
            "topk_calibrated_risk": 26.2,
            "qualifying_text_ai_density": 38.41,
            "unsafe_cluster_count": 0,
            "risky_window_count": 0,
        },
        current_goal=current_goal,
    )
    assert v5_residual_comb._safe_band_density_section_repair_should_run(
        current_scores={
            "topk_calibrated_risk": 25.636,
            "qualifying_text_ai_density": 39.37,
            "unsafe_cluster_count": 0,
            "risky_window_count": 0,
        },
        current_goal={
            "ai_footprint_gate": {
                "safe_band_thresholds": {"topk_calibrated_risk": 25.0, "qualifying_text_ai_density": 35.0},
                "remaining_ai_footprint_drivers": [
                    {"driver": "topk_calibrated_risk", "value": 25.636, "safe_band": 25.0},
                    {"driver": "qualifying_text_ai_density", "value": 39.37, "safe_band": 35.0},
                ],
            }
        },
    )
    assert not v5_residual_comb._safe_band_density_section_repair_should_run(
        current_scores={
            "topk_calibrated_risk": 22.296,
            "qualifying_text_ai_density": 38.41,
            "unsafe_cluster_count": 1,
            "risky_window_count": 0,
        },
        current_goal=current_goal,
    )


def test_v5_safe_band_density_section_prompt_is_author_proxy_density_first():
    section = SectionUnit(
        section_id="safe_band_density_section_s001",
        heading="Safe-band density section repair",
        text="The student needs support in practical class. This matters because inclusive teaching is important.",
        start_char=0,
        end_char=93,
        paragraph_count=1,
        word_count=14,
        metadata={
            "selection_reason": "ai_mitigation_density_target_segment",
            "scanner_focus": {"source": "ai_mitigation.target_segments"},
        },
    )
    prompt = v5_residual_comb.build_safe_band_density_section_repair_prompt(
        section=section,
        current_scores={"topk_calibrated_risk": 22.296, "qualifying_text_ai_density": 38.41},
        current_goal={
            "ai_footprint_gate": {
                "safe_band_thresholds": {"topk_calibrated_risk": 25.0, "qualifying_text_ai_density": 35.0},
                "remaining_ai_footprint_drivers": [
                    {"driver": "qualifying_text_ai_density", "value": 38.41, "safe_band": 35.0},
                ],
                "after": {
                    "semantic_footprint": {
                        "generic_assertion_risk": 80.0,
                        "unsupported_claim_risk": 55.0,
                    }
                },
            }
        },
        variant_count=2,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "safe_band_density_section_repair"
    assert payload["single_product_judge"]["judge"] == "DraftProof internal safe-band and authorship-integrity scoring"
    assert payload["single_product_judge"]["ignored_as_acceptance_judge"] == "external AI flag score"
    assert payload["density_only_trigger"]["remaining_target"] == "qualifying_text_ai_density"
    assert payload["density_only_trigger"]["ai_authorship_must_not_increase"] is True
    assert payload["kpi_contract"]["secondary_density_drivers"]["generic_assertion_risk"] == 80.0
    assert payload["section"]["source_voice_profile"]["first_person_count"] == 0
    assert payload["materiality_gate"]["minimum_changed_source_sentences"] == 2
    assert payload["materiality_gate"]["minimum_changed_source_sentence_ratio"] == 0.4
    assert payload["materiality_gate"]["minimum_word_count"] >= 1
    assert payload["materiality_gate"]["maximum_word_count"] >= payload["materiality_gate"]["minimum_word_count"]
    assert any("only changes one target sentence" in rule for rule in payload["materiality_gate"]["reject_if"])
    assert any("between" in rule and "words" in rule for rule in payload["rules"])
    assert any("Change at least" in rule for rule in payload["rules"])
    assert any("repeat" in rule for rule in payload["rules"])
    assert payload["writer_variant_plan"][0]["goal"].startswith("Density-only section rebuild")


def test_v5_safe_band_density_section_selection_prefers_authorship_gap_segment():
    paragraph_one = "Predictable sentence stays here. It has no missing author-owned evidence."
    paragraph_two = "The report says students struggle with sectioning before cutting. The current bridge is too broad."
    current_text = f"{paragraph_one}\n\n{paragraph_two}"
    sections = v5_residual_comb._safe_band_density_section_repair_sections(
        current_text,
        {
            "ai_mitigation": {
                "target_segments": [
                    {
                        "segment_id": "s001",
                        "text": "Predictable sentence stays here.",
                        "lever": "predictability_reduction",
                        "bucket": "auto_candidate",
                        "user_input_needed": "none if no evidence/source gap is attached",
                        "primary_signal": {"key": "ai_likelihood", "score": 40, "tier": "low"},
                    },
                    {
                        "segment_id": "s002",
                        "text": "The current bridge is too broad.",
                        "lever": "reasoning_continuity",
                        "bucket": "structure_revision",
                        "user_input_needed": "the missing connection between the classroom evidence and the claim",
                        "primary_signal": {"key": "authorship_risk", "score": 73, "tier": "high"},
                    },
                ]
            }
        },
        {},
        limit=1,
    )

    assert len(sections) == 1
    assert sections[0].text == paragraph_two
    assert sections[0].metadata["selection_reason"] == "ai_mitigation_density_target_segment"


def test_v5_safe_band_density_section_selection_skips_spent_ranges():
    paragraph_one = "The first broad bridge remains generic. It needs a grounded route."
    paragraph_two = "The second broad bridge remains generic. It needs a grounded route."
    current_text = f"{paragraph_one}\n\n{paragraph_two}"
    first_start = current_text.index(paragraph_one)
    first_end = first_start + len(paragraph_one)

    sections = v5_residual_comb._safe_band_density_section_repair_sections(
        current_text,
        {
            "ai_mitigation": {
                "target_segments": [
                    {
                        "segment_id": "s001",
                        "text": "The first broad bridge remains generic.",
                        "lever": "reasoning_continuity",
                        "bucket": "structure_revision",
                        "user_input_needed": "none",
                        "primary_signal": {"key": "authorship_risk", "score": 80, "tier": "high"},
                    },
                    {
                        "segment_id": "s002",
                        "text": "The second broad bridge remains generic.",
                        "lever": "reasoning_continuity",
                        "bucket": "structure_revision",
                        "user_input_needed": "none",
                        "primary_signal": {"key": "authorship_risk", "score": 78, "tier": "high"},
                    },
                ]
            }
        },
        {},
        limit=2,
        exclude_ranges={(first_start, first_end)},
    )

    assert [section.section_id for section in sections] == ["safe_band_density_section_s002"]


def test_v5_safe_band_density_materiality_rejects_repetition_and_compression(monkeypatch):
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_min_word_ratio", lambda: 0.75)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_max_word_ratio", lambda: 2.0)
    source = (
        "Students first need to see how sectioning controls the haircut. "
        "The teacher demonstration gives them a visible starting point. "
        "Practice then shows where their hand position still needs feedback."
    )
    repeated = (
        "Sectioning has to come before the cut because it gives the student a route. "
        "Sectioning has to come before the cut because it gives the student a route. "
        "The teacher can then use the attempt to point out hand position and next feedback."
    )
    compressed = "Sectioning should come first. Feedback follows."

    repeated_result = v5_residual_comb._safe_band_density_section_repair_materiality(
        source_text=source,
        candidate_text=repeated,
    )
    compressed_result = v5_residual_comb._safe_band_density_section_repair_materiality(
        source_text=source,
        candidate_text=compressed,
    )

    assert not repeated_result["passed"]
    assert repeated_result["reason"] == "density_section_repair_repetition_regression"
    assert not compressed_result["passed"]
    assert compressed_result["reason"] == "density_section_repair_length_out_of_bounds"


def test_v5_safe_band_density_materiality_requires_section_level_change(monkeypatch):
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_min_word_ratio", lambda: 0.5)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_max_word_ratio", lambda: 1.8)
    source = (
        "The opening claim stays mostly the same. "
        "The second sentence keeps the same bridge. "
        "The third sentence is the target sentence. "
        "The fourth sentence keeps the same implication. "
        "The final sentence keeps the same closure."
    )
    candidate = (
        "The opening claim stays mostly the same. "
        "The second sentence keeps the same bridge. "
        "The target sentence now describes the classroom action in different words. "
        "The fourth sentence keeps the same implication. "
        "The final sentence keeps the same closure."
    )

    result = v5_residual_comb._safe_band_density_section_repair_materiality(
        source_text=source,
        candidate_text=candidate,
        target_sentence="The third sentence is the target sentence.",
    )

    assert not result["passed"]
    assert result["reason"] == "density_section_repair_too_few_changed_sentences"
    assert result["density_required_changed_sentence_count"] == 2


def test_v5_safe_band_density_materiality_allows_long_section_tightening(monkeypatch):
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_min_word_ratio", lambda: 0.88)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_long_section_word_threshold", lambda: 12)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_long_section_min_word_ratio", lambda: 0.82)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_max_word_ratio", lambda: 1.12)
    source = (
        "First sentence keeps the author voice and classroom context. "
        "Second sentence explains the same source material in a broad way. "
        "Third sentence is the target sentence that needs a new route. "
        "Fourth sentence keeps a practical implication. "
        "Fifth sentence adds the closing observation."
    )
    candidate = (
        "First sentence keeps the author voice and classroom context. "
        "The second sentence narrows the source material to practical class. "
        "The target now follows the classroom action instead of a broad route. "
        "Fourth sentence keeps a practical implication."
    )

    result = v5_residual_comb._safe_band_density_section_repair_materiality(
        source_text=source,
        candidate_text=candidate,
        target_sentence="Third sentence is the target sentence that needs a new route.",
    )

    assert result["passed"]
    assert result["minimum_word_ratio"] == 0.82
    assert result["word_ratio"] < 0.88


def test_v5_safe_band_density_materiality_allows_adjacent_duplicate_cleanup(monkeypatch):
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_min_word_ratio", lambda: 0.88)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_max_word_ratio", lambda: 1.12)
    source = (
        "In my AT2 report, I focused on inclusive learning design. "
        "In my AT2 report, I focused on inclusive learning design. "
        "I then linked the issue to practical haircut structures."
    )
    candidate = (
        "In my AT2 report, I focused on inclusive learning design. "
        "I then linked the issue to practical haircut structures."
    )

    result = v5_residual_comb._safe_band_density_section_repair_materiality(
        source_text=source,
        candidate_text=candidate,
    )

    assert result["passed"] is True
    assert result["reason"] == "density_section_adjacent_duplicate_cleanup"
    assert result["word_ratio"] < result["minimum_word_ratio"]
    assert result["duplicate_cleanup_audit"]["removed_duplicate_sentence_count"] == 1


def test_v5_safe_band_density_section_repair_adds_deterministic_duplicate_cleanup_variant():
    section = SectionUnit(
        "safe_band_density_section_s001",
        "Density",
        (
            "In my AT2 report, I focused on inclusive learning design. "
            "In my AT2 report, I focused on inclusive learning design. "
            "I then linked the issue to practical haircut structures."
        ),
        0,
        160,
        1,
        23,
        {},
    )

    variants = v5_residual_comb._safe_band_density_section_repair_deterministic_variants(section)

    assert len(variants) == 1
    assert variants[0].variant_id == "deterministic_adjacent_duplicate_cleanup"
    assert variants[0].text.count("inclusive learning design") == 1
    assert variants[0].author_proxy_provenance[0]["provenance"] == "duplicate_source_cleanup"


def test_v5_safe_band_density_materiality_rejects_source_voice_shift(monkeypatch):
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_min_word_ratio", lambda: 0.5)
    source = (
        "In my AT2 report, I wrote about what it's actually like for students in the practical class. "
        "They're trying to manage sectioning and cutting at the same time. "
        "I use that example to show why support has to be concrete."
    )
    candidate = (
        "The AT2 report examined the practical realities for students in the practical class. "
        "The students are trying to manage sectioning and cutting at the same time. "
        "This example demonstrates why support has to be concrete."
    )

    result = v5_residual_comb._safe_band_density_section_repair_materiality(
        source_text=source,
        candidate_text=candidate,
    )

    assert not result["passed"]
    assert result["reason"] == "density_section_repair_voice_shift"
    assert result["voice_audit"]["first_person_preserved"] is False
    assert result["voice_audit"]["contractions_preserved"] is False


def test_v5_safe_band_density_section_loop_accepts_density_gap_movement(monkeypatch, tmp_path):
    accepted_checkpoints = []
    section = SectionUnit(
        section_id="safe_band_density_section_s001",
        heading="Safe-band density section repair",
        text="The broad sentence remains. Another broad sentence remains. A third sentence remains.",
        start_char=0,
        end_char=75,
        paragraph_count=1,
        word_count=11,
        metadata={"selection_reason": "ai_mitigation_density_target_segment"},
    )
    current_scores = {
        "ai": 31.24,
        "topk_calibrated_risk": 22.296,
        "qualifying_text_ai_density": 38.41,
        "unsafe_cluster_count": 0,
        "risky_window_count": 0,
    }
    current_goal = {
        "ai_footprint_gate": {
            "safe_band_thresholds": {"topk_calibrated_risk": 25.0, "qualifying_text_ai_density": 35.0},
            "remaining_ai_footprint_drivers": [
                {"driver": "qualifying_text_ai_density", "value": 38.41, "safe_band": 35.0},
            ],
        }
    }
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_round_limit", lambda: 1)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_sections", lambda *_args, **_kwargs: [section])
    monkeypatch.setattr(
        v5_residual_comb,
        "generate_safe_band_density_section_repair_variants",
        lambda **_kwargs: (
            [RecompositionVariant("v1", "The classroom example now carries the claim. The teacher links support to the visible task. The limit stays clear.", 18)],
            {"status": "ok"},
            "prompt",
            "completion",
        ),
    )
    monkeypatch.setattr(
        v5_residual_comb,
        "_safe_band_density_section_repair_materiality",
        lambda **_kwargs: {"passed": True, "reason": "material_density_section_route_change"},
    )

    def fake_score(**_kwargs):
        return {
            "section_id": "safe_band_density_section_s001",
            "variant_id": "v1",
            "apply_status": {"applied": True},
            "scores": {
                "ai": 31.0,
                "topk_calibrated_risk": 22.0,
                "qualifying_text_ai_density": 34.9,
                "unsafe_cluster_count": 0,
                "risky_window_count": 0,
            },
            "incremental": {
                "ai_delta": 0.24,
                "ai_authorship_delta": 0.0,
                "topk_calibrated_risk_delta": 0.296,
                "qualifying_text_ai_density_delta": 3.51,
                "unsafe_cluster_count_delta": 0.0,
                "unsafe_word_ratio_delta": 0.0,
                "risky_window_count_delta": 0.0,
            },
            "candidate_text": "accepted density text",
            "candidate_report": {},
            "candidate_goal": {
                "strict_ai_safe_band_achieved": True,
                "ai_footprint_gate": {"safe_band": True, "remaining_ai_footprint_drivers": []},
            },
        }

    monkeypatch.setattr(v5_residual_comb, "_score_residual_variant", fake_score)

    text, _report, _goal, scores, rounds, _best = v5_residual_comb._run_safe_band_density_section_repair_loop(
        original_text="original",
        baseline_report={},
        baseline_scores=current_scores,
        current_text=section.text,
        current_report={},
        current_goal=current_goal,
        current_scores=current_scores,
        gateway=object(),
        output_dir=tmp_path,
        global_best_candidate=None,
        accepted_checkpoint_callback=accepted_checkpoints.append,
    )

    assert text == "accepted density text"
    assert scores["qualifying_text_ai_density"] == 34.9
    assert rounds[0]["phase"] == "safe_band_density_section_repair"
    assert rounds[0]["status"] == "accepted"
    assert accepted_checkpoints[0]["phase"] == "safe_band_density_section_repair"


def test_v5_safe_band_density_section_loop_excludes_accepted_section_next_round(monkeypatch, tmp_path):
    first = SectionUnit(
        "density_s001",
        "Density",
        "First broad sentence. More broad text. Third broad text.",
        0,
        55,
        1,
        9,
        {},
    )
    second = SectionUnit(
        "density_s002",
        "Density",
        "Second broad sentence. More broad text. Third broad text.",
        57,
        113,
        1,
        9,
        {},
    )
    current_scores = {
        "ai": 31.24,
        "topk_calibrated_risk": 22.296,
        "qualifying_text_ai_density": 38.41,
        "unsafe_cluster_count": 0,
        "unsafe_word_ratio": 0.0,
        "risky_window_count": 0,
    }
    current_goal = {
        "ai_footprint_gate": {
            "safe_band_thresholds": {"topk_calibrated_risk": 25.0, "qualifying_text_ai_density": 35.0},
            "remaining_ai_footprint_drivers": [
                {"driver": "qualifying_text_ai_density", "value": 38.41, "safe_band": 35.0},
            ],
        }
    }
    selector_calls: list[set[tuple[int, int]]] = []

    def fake_sections(*_args, **kwargs):
        excluded = set(kwargs.get("exclude_ranges") or set())
        selector_calls.append(excluded)
        if not excluded:
            return [first]
        return [second]

    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_round_limit", lambda: 2)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_sections", fake_sections)
    monkeypatch.setattr(
        v5_residual_comb,
        "generate_safe_band_density_section_repair_variants",
        lambda **_kwargs: (
            [RecompositionVariant("v1", "replacement text with enough words for materiality", 7)],
            {"status": "ok"},
            "prompt",
            "completion",
        ),
    )
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_materiality", lambda **_kwargs: {"passed": True})

    def fake_score(**kwargs):
        section = kwargs["section"]
        first_round = section.section_id == "density_s001"
        density = 37.8 if first_round else 34.8
        return {
            "section_id": section.section_id,
            "variant_id": "v1",
            "apply_status": {"applied": True},
            "scores": {
                "ai": 31.0,
                "topk_calibrated_risk": 22.0,
                "qualifying_text_ai_density": density,
                "unsafe_cluster_count": 0,
                "unsafe_word_ratio": 0.0,
                "risky_window_count": 0,
            },
            "incremental": {
                "ai_delta": 0.24,
                "ai_authorship_delta": 0.0,
                "topk_calibrated_risk_delta": 0.296,
                "qualifying_text_ai_density_delta": 0.61 if first_round else 3.0,
                "unsafe_cluster_count_delta": 0.0,
                "unsafe_word_ratio_delta": 0.0,
                "risky_window_count_delta": 0.0,
            },
            "candidate_text": "after first density repair" if first_round else "after second density repair",
            "candidate_report": {},
            "candidate_goal": (
                {
                    "ai_footprint_gate": {
                        "safe_band_thresholds": {"topk_calibrated_risk": 25.0, "qualifying_text_ai_density": 35.0},
                        "remaining_ai_footprint_drivers": [
                            {"driver": "qualifying_text_ai_density", "value": 37.8, "safe_band": 35.0},
                        ],
                    }
                }
                if first_round
                else {"strict_ai_safe_band_achieved": True, "ai_footprint_gate": {"safe_band": True, "remaining_ai_footprint_drivers": []}}
            ),
        }

    monkeypatch.setattr(v5_residual_comb, "_score_residual_variant", fake_score)

    text, _report, _goal, scores, rounds, _best = v5_residual_comb._run_safe_band_density_section_repair_loop(
        original_text="original",
        baseline_report={},
        baseline_scores=current_scores,
        current_text=f"{first.text}\n\n{second.text}",
        current_report={},
        current_goal=current_goal,
        current_scores=current_scores,
        gateway=object(),
        output_dir=tmp_path,
        global_best_candidate=None,
    )

    assert text == "after second density repair"
    assert scores["qualifying_text_ai_density"] == 34.8
    assert rounds[0]["accepted"]["section_id"] == "density_s001"
    assert rounds[1]["sections"][0]["section_id"] == "density_s002"
    assert selector_calls[0] == set()
    assert selector_calls[1] == {(0, 55)}


def test_v5_safe_band_density_section_loop_tries_next_section_after_no_movement(monkeypatch, tmp_path):
    first = SectionUnit("density_s001", "Density", "First broad sentence. More broad text. Third broad text.", 0, 55, 1, 9, {})
    second = SectionUnit("density_s002", "Density", "Second broad sentence. More broad text. Third broad text.", 57, 113, 1, 9, {})
    current_scores = {
        "ai": 31.24,
        "topk_calibrated_risk": 22.296,
        "qualifying_text_ai_density": 38.41,
        "unsafe_cluster_count": 0,
        "risky_window_count": 0,
    }
    current_goal = {
        "ai_footprint_gate": {
            "safe_band_thresholds": {"topk_calibrated_risk": 25.0, "qualifying_text_ai_density": 35.0},
            "remaining_ai_footprint_drivers": [
                {"driver": "qualifying_text_ai_density", "value": 38.41, "safe_band": 35.0},
            ],
        }
    }
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_round_limit", lambda: 1)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_sections", lambda *_args, **_kwargs: [first, second])
    monkeypatch.setattr(
        v5_residual_comb,
        "generate_safe_band_density_section_repair_variants",
        lambda **_kwargs: ([RecompositionVariant("v1", "replacement text with enough words for materiality", 7)], {"status": "ok"}, "prompt", "completion"),
    )
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_materiality", lambda **_kwargs: {"passed": True})

    def fake_score(**kwargs):
        section = kwargs["section"]
        accepted = section.section_id == "density_s002"
        return {
            "section_id": section.section_id,
            "variant_id": "v1",
            "apply_status": {"applied": True},
            "scores": {
                "ai": 31.0 if accepted else 31.24,
                "topk_calibrated_risk": 22.0,
                "qualifying_text_ai_density": 34.9 if accepted else 38.4,
                "unsafe_cluster_count": 0,
                "risky_window_count": 0,
            },
            "incremental": {
                "ai_delta": 0.24 if accepted else 0.0,
                "ai_authorship_delta": 0.0,
                "topk_calibrated_risk_delta": 0.296,
                "qualifying_text_ai_density_delta": 3.51 if accepted else 0.01,
                "unsafe_cluster_count_delta": 0.0,
                "unsafe_word_ratio_delta": 0.0,
                "risky_window_count_delta": 0.0,
            },
            "candidate_text": "accepted second section" if accepted else "first section no movement",
            "candidate_report": {},
            "candidate_goal": (
                {"strict_ai_safe_band_achieved": True, "ai_footprint_gate": {"safe_band": True, "remaining_ai_footprint_drivers": []}}
                if accepted
                else current_goal
            ),
        }

    monkeypatch.setattr(v5_residual_comb, "_score_residual_variant", fake_score)

    text, _report, _goal, _scores, rounds, _best = v5_residual_comb._run_safe_band_density_section_repair_loop(
        original_text="original",
        baseline_report={},
        baseline_scores=current_scores,
        current_text=f"{first.text}\n\n{second.text}",
        current_report={},
        current_goal=current_goal,
        current_scores=current_scores,
        gateway=object(),
        output_dir=tmp_path,
        global_best_candidate=None,
    )

    assert text == "accepted second section"
    assert rounds[0]["status"] == "accepted"
    assert len(rounds[0]["section_attempts"]) == 2
    assert rounds[0]["accepted"]["section_id"] == "density_s002"


def test_v5_safe_band_density_section_loop_selects_strongest_section_candidate(monkeypatch, tmp_path):
    first = SectionUnit("density_s001", "Density", "First broad sentence. More broad text. Third broad text.", 0, 55, 1, 9, {})
    second = SectionUnit("density_s002", "Density", "Second broad sentence. More broad text. Third broad text.", 57, 113, 1, 9, {})
    current_scores = {
        "ai": 31.24,
        "topk_calibrated_risk": 22.296,
        "qualifying_text_ai_density": 38.41,
        "unsafe_cluster_count": 0,
        "risky_window_count": 0,
    }
    current_goal = {
        "ai_footprint_gate": {
            "safe_band_thresholds": {"topk_calibrated_risk": 25.0, "qualifying_text_ai_density": 35.0},
            "remaining_ai_footprint_drivers": [
                {"driver": "qualifying_text_ai_density", "value": 38.41, "safe_band": 35.0},
            ],
        }
    }
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_round_limit", lambda: 1)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_sections", lambda *_args, **_kwargs: [first, second])
    monkeypatch.setattr(
        v5_residual_comb,
        "generate_safe_band_density_section_repair_variants",
        lambda **_kwargs: ([RecompositionVariant("v1", "replacement text with enough words for materiality", 7)], {"status": "ok"}, "prompt", "completion"),
    )
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_materiality", lambda **_kwargs: {"passed": True})

    def fake_score(**kwargs):
        section = kwargs["section"]
        strong = section.section_id == "density_s002"
        density = 34.8 if strong else 38.2
        return {
            "section_id": section.section_id,
            "variant_id": "v1",
            "apply_status": {"applied": True},
            "scores": {
                "ai": 31.0,
                "topk_calibrated_risk": 22.0,
                "qualifying_text_ai_density": density,
                "unsafe_cluster_count": 0,
                "risky_window_count": 0,
            },
            "incremental": {
                "ai_delta": 0.24,
                "ai_authorship_delta": 0.0,
                "topk_calibrated_risk_delta": 0.296,
                "qualifying_text_ai_density_delta": 3.61 if strong else 0.21,
                "unsafe_cluster_count_delta": 0.0,
                "unsafe_word_ratio_delta": 0.0,
                "risky_window_count_delta": 0.0,
            },
            "candidate_text": "strong second section" if strong else "weak first section",
            "candidate_report": {},
            "candidate_goal": (
                {"strict_ai_safe_band_achieved": True, "ai_footprint_gate": {"safe_band": True, "remaining_ai_footprint_drivers": []}}
                if strong
                else {
                    "ai_footprint_gate": {
                        "remaining_ai_footprint_drivers": [
                            {"driver": "qualifying_text_ai_density", "value": density, "safe_band": 35.0}
                        ]
                    }
                }
            ),
        }

    monkeypatch.setattr(v5_residual_comb, "_score_residual_variant", fake_score)

    text, _report, _goal, scores, rounds, _best = v5_residual_comb._run_safe_band_density_section_repair_loop(
        original_text="original",
        baseline_report={},
        baseline_scores=current_scores,
        current_text=f"{first.text}\n\n{second.text}",
        current_report={},
        current_goal=current_goal,
        current_scores=current_scores,
        gateway=object(),
        output_dir=tmp_path,
        global_best_candidate=None,
    )

    assert text == "strong second section"
    assert scores["qualifying_text_ai_density"] == 34.8
    assert len(rounds[0]["section_attempts"]) == 2
    assert rounds[0]["accepted"]["section_id"] == "density_s002"


def test_v5_safe_band_evidence_repair_runs_density_section_loop_first(monkeypatch, tmp_path):
    call_order: list[str] = []
    section = SectionUnit("density_s001", "Density", "First broad sentence. More broad text.", 0, 45, 1, 7, {})
    current_scores = {
        "ai": 31.24,
        "topk_calibrated_risk": 25.1,
        "qualifying_text_ai_density": 38.41,
        "unsafe_cluster_count": 0,
        "risky_window_count": 0,
    }
    current_goal = {
        "ai_footprint_gate": {
            "safe_band": False,
            "safe_band_thresholds": {"topk_calibrated_risk": 25.0, "qualifying_text_ai_density": 35.0},
            "remaining_ai_footprint_drivers": [
                {"driver": "topk_calibrated_risk", "value": 25.1, "safe_band": 25.0},
                {"driver": "qualifying_text_ai_density", "value": 38.41, "safe_band": 35.0},
            ],
        }
    }

    def unchanged(name):
        def inner(**kwargs):
            call_order.append(name)
            return (
                kwargs["current_text"],
                kwargs["current_report"],
                kwargs["current_goal"],
                kwargs["current_scores"],
                [{"phase": name, "status": "skipped"}],
                kwargs["global_best_candidate"],
            )
        return inner

    monkeypatch.setattr(v5_residual_comb, "_safe_band_evidence_repair_sections", lambda *_args, **_kwargs: [section])
    monkeypatch.setattr(v5_residual_comb, "_run_safe_band_density_section_repair_loop", unchanged("density"))
    monkeypatch.setattr(v5_residual_comb, "_run_safe_band_controlled_operation_loop", unchanged("controlled"))
    monkeypatch.setattr(v5_residual_comb, "_run_safe_band_sentence_replacement_loop", unchanged("sentence"))
    monkeypatch.setattr(v5_residual_comb, "_safe_band_evidence_pack_enabled", lambda: False)

    _text, _report, _goal, _scores, rounds, _best = v5_residual_comb._run_safe_band_evidence_repair_pass(
        original_text="original",
        baseline_report={},
        baseline_scores=current_scores,
        current_text=section.text,
        current_report={},
        current_goal=current_goal,
        current_scores=current_scores,
        gateway=object(),
        output_dir=tmp_path,
        global_best_candidate=None,
    )

    assert call_order[:3] == ["density", "controlled", "sentence"]
    assert [row["phase"] for row in rounds[:3]] == ["density", "controlled", "sentence"]


def test_v5_safe_band_evidence_pack_apply_uses_all_replacements_in_reverse_offset_order():
    current_text = (
        "Intro paragraph.\n"
        "Each lesson ends with group reflection. Students talk during this time, and their comments expose what they learned. I use those comments again.\n"
        "Middle paragraph.\n"
        "Videos can make the haircut look easy. This creates an illusion of simplicity. The classroom task exposes sectioning."
    )
    first = v5_residual_comb._section_from_paragraph_containing_text(
        current_text,
        "Students talk during this time, and their comments expose what they learned.",
        section_id="s1",
        selection_reason="test",
        scanner_focus={},
    )
    second = v5_residual_comb._section_from_paragraph_containing_text(
        current_text,
        "This creates an illusion of simplicity.",
        section_id="s2",
        selection_reason="test",
        scanner_focus={},
    )
    variant = {
        "variant_id": "v1",
        "replacements": [
            {
                "section_id": "s1",
                "text": (
                    "I close the lesson with group reflection. Students name what made sense and where the work caught them. "
                    "Those comments tell me what to demonstrate again."
                ),
            },
            {
                "section_id": "s2",
                "text": (
                    "Online videos make the haircut look finished before the hard part is visible. "
                    "In class, the first barrier is sectioning, not confidence. "
                    "That is where I slow the task down."
                ),
            },
        ],
    }

    candidate, apply_status, materiality = v5_residual_comb._apply_safe_band_evidence_pack_variant(
        current_text=current_text,
        sections=[first, second],
        variant=variant,
    )

    assert apply_status["applied"] is True
    assert materiality["passed"] is True
    assert "Students name what made sense" in candidate
    assert "Online videos make the haircut look finished" in candidate
    assert "Students talk during this time" not in candidate
    assert "This creates an illusion of simplicity" not in candidate


def test_v5_safe_band_evidence_pack_applies_density_materiality_contract(monkeypatch):
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_min_word_ratio", lambda: 0.5)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_max_word_ratio", lambda: 1.8)
    current_text = (
        "I start with a quick demonstration. The explanation still sounds broad and polished. "
        "I then ask students to mark what they missed."
    )
    section = SectionUnit(
        section_id="safe_band_density_section_t001",
        heading="Safe-band density section repair",
        text=current_text,
        start_char=0,
        end_char=len(current_text),
        paragraph_count=1,
        word_count=21,
        metadata={"selection_reason": "ai_mitigation_density_target_segment"},
    )
    weak_variant = {
        "variant_id": "v1",
        "replacements": [
            {
                "section_id": "safe_band_density_section_t001",
                "text": (
                    "I start with a quick demonstration. The broad explanation is recast around what students missed. "
                    "I then ask students to mark what they missed."
                ),
            }
        ],
    }

    _candidate, apply_status, materiality = v5_residual_comb._apply_safe_band_evidence_pack_variant(
        current_text=current_text,
        sections=[section],
        variant=weak_variant,
    )

    assert apply_status["applied"] is False
    assert materiality["sections"][0]["contract"] == "density_section_repair"
    assert materiality["sections"][0]["reason"] == "density_section_repair_too_few_changed_sentences"


def test_v5_safe_band_evidence_pack_rejects_author_placeholders_in_candidate_text(monkeypatch):
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_min_word_ratio", lambda: 0.5)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_max_word_ratio", lambda: 1.8)
    current_text = (
        "I start with a quick demonstration. The explanation still sounds broad and polished. "
        "I then ask students to mark what they missed."
    )
    section = SectionUnit(
        section_id="safe_band_density_section_t001",
        heading="Safe-band density section repair",
        text=current_text,
        start_char=0,
        end_char=len(current_text),
        paragraph_count=1,
        word_count=21,
        metadata={"selection_reason": "ai_mitigation_density_target_segment"},
    )
    variant = {
        "variant_id": "v1",
        "replacements": [
            {
                "section_id": "safe_band_density_section_t001",
                "text": (
                    "I begin with the demonstration before naming the gap. "
                    "The explanation changes route around what students missed. "
                    "I then ask them to mark the unclear step [needs_author: add exact class detail]."
                ),
            }
        ],
    }

    _candidate, apply_status, materiality = v5_residual_comb._apply_safe_band_evidence_pack_variant(
        current_text=current_text,
        sections=[section],
        variant=variant,
    )

    assert apply_status["applied"] is False
    assert materiality["sections"][0]["reason"] == "candidate_contains_author_placeholder"


def test_v5_safe_band_evidence_pack_rejects_unbalanced_double_quote(monkeypatch):
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_min_word_ratio", lambda: 0.5)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_max_word_ratio", lambda: 1.8)
    current_text = (
        "I start with a quick demonstration. The explanation still sounds broad and polished. "
        "I then ask students to mark what they missed."
    )
    section = SectionUnit(
        section_id="safe_band_density_section_t001",
        heading="Safe-band density section repair",
        text=current_text,
        start_char=0,
        end_char=len(current_text),
        paragraph_count=1,
        word_count=21,
        metadata={"selection_reason": "ai_mitigation_density_target_segment"},
    )
    variant = {
        "variant_id": "v1",
        "replacements": [
            {
                "section_id": "safe_band_density_section_t001",
                "text": (
                    "I begin with the demonstration before naming the gap. "
                    "The explanation changes route around what students missed. "
                    "I then ask students to say \"what step confused them."
                ),
            }
        ],
    }

    _candidate, apply_status, materiality = v5_residual_comb._apply_safe_band_evidence_pack_variant(
        current_text=current_text,
        sections=[section],
        variant=variant,
    )

    assert apply_status["applied"] is False
    assert materiality["sections"][0]["reason"] == "candidate_quote_integrity_issue"


def test_v5_safe_band_evidence_repair_selector_prefers_strict_safe_and_rejects_authorship_regression():
    current_scores = {
        "ai": 33.32,
        "ai_authorship": 33.0,
        "topk_calibrated_risk": 30.586,
        "qualifying_text_ai_density": 40.73,
        "unsafe_cluster_count": 2,
        "risky_window_count": 0,
    }
    partial = {
        "apply_status": {"applied": True},
        "incremental": {
            "ai_delta": 0.1,
            "ai_authorship_delta": 0.1,
            "topk_calibrated_risk_delta": 0.4,
            "qualifying_text_ai_density_delta": 1.0,
            "unsafe_cluster_count_delta": 0,
            "risky_window_count_delta": 0,
        },
        "scores": {
            "topk_calibrated_risk": 30.186,
            "qualifying_text_ai_density": 39.73,
        },
        "candidate_goal": {
            "ai_footprint_gate": {
                "remaining_ai_footprint_drivers": [
                    {"driver": "topk_calibrated_risk", "value": 30.186, "safe_band": 25.0},
                    {"driver": "qualifying_text_ai_density", "value": 39.73, "safe_band": 35.0},
                ]
            }
        },
        "safe_band_evidence_materiality": {"passed": True},
    }
    authorship_regression = {
        **partial,
        "incremental": {
            **partial["incremental"],
            "ai_authorship_delta": -0.2,
            "qualifying_text_ai_density_delta": 2.0,
        },
    }
    strict_safe = {
        **partial,
        "incremental": {
            **partial["incremental"],
            "topk_calibrated_risk_delta": 0.2,
            "qualifying_text_ai_density_delta": 0.2,
        },
        "candidate_goal": {
            "strict_ai_safe_band_achieved": True,
            "ai_footprint_gate": {"safe_band": True, "remaining_ai_footprint_drivers": []},
        },
    }

    assert _has_safe_band_evidence_repair_movement(partial, current_scores=current_scores)
    assert not _has_safe_band_evidence_repair_movement(authorship_regression, current_scores=current_scores)
    assert _best_safe_band_evidence_repair_candidate([partial, authorship_regression, strict_safe], current_scores=current_scores) is strict_safe


def test_v5_density_checkpoint_movement_allows_bounded_ai_tradeoff(monkeypatch):
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_checkpoint_min_density_delta", lambda: 0.5)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_checkpoint_ai_regression_tolerance", lambda: 1.0)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_checkpoint_authorship_regression_tolerance", lambda: 1.0)
    current_scores = {
        "ai": 31.97,
        "ai_authorship": 32.0,
        "topk_calibrated_risk": 25.112,
        "qualifying_text_ai_density": 39.23,
        "unsafe_cluster_count": 0,
        "unsafe_word_ratio": 0.0,
        "risky_window_count": 0,
    }
    row = {
        "apply_status": {"applied": True},
        "scores": {
            "ai": 32.41,
            "ai_authorship": 32.0,
            "topk_calibrated_risk": 23.548,
            "qualifying_text_ai_density": 38.2,
            "unsafe_cluster_count": 0,
            "unsafe_word_ratio": 0.0,
            "risky_window_count": 0,
        },
        "incremental": {
            "ai_delta": -0.44,
            "ai_authorship_delta": 0.0,
            "topk_calibrated_risk_delta": 1.564,
            "qualifying_text_ai_density_delta": 1.03,
            "unsafe_cluster_count_delta": 0.0,
            "unsafe_word_ratio_delta": 0.0,
            "risky_window_count_delta": 0.0,
        },
        "safe_band_density_section_materiality": {"passed": True},
        "safe_band_evidence_materiality": {"passed": True},
    }

    assert v5_residual_comb._has_density_safe_band_checkpoint_movement(row, current_scores=current_scores)
    assert _has_safe_band_evidence_repair_movement(row, current_scores=current_scores)

    row["incremental"] = {**row["incremental"], "ai_delta": -1.1}
    assert not v5_residual_comb._has_density_safe_band_checkpoint_movement(row, current_scores=current_scores)

    bounded_authorship_tradeoff = {
        **row,
        "scores": {
            **row["scores"],
            "ai": 32.64,
            "ai_authorship": 33.0,
            "topk_calibrated_risk": 24.541,
            "qualifying_text_ai_density": 38.47,
        },
        "incremental": {
            **row["incremental"],
            "ai_delta": -0.67,
            "ai_authorship_delta": -1.0,
            "topk_calibrated_risk_delta": 0.571,
            "qualifying_text_ai_density_delta": 0.76,
        },
    }
    assert v5_residual_comb._has_density_safe_band_checkpoint_movement(
        bounded_authorship_tradeoff,
        current_scores=current_scores,
    )

    too_much_authorship_tradeoff = {
        **bounded_authorship_tradeoff,
        "incremental": {
            **bounded_authorship_tradeoff["incremental"],
            "ai_authorship_delta": -1.1,
        },
    }
    assert not v5_residual_comb._has_density_safe_band_checkpoint_movement(
        too_much_authorship_tradeoff,
        current_scores=current_scores,
    )


def test_v5_safe_band_evidence_repair_rejects_document_repetition_regression():
    current_scores = {"ai": 33.0, "topk_calibrated_risk": 30.0, "qualifying_text_ai_density": 40.0}
    row = {
        "apply_status": {"applied": True},
        "incremental": {
            "ai_delta": 0.2,
            "ai_authorship_delta": 0.0,
            "topk_calibrated_risk_delta": 0.2,
            "qualifying_text_ai_density_delta": 1.0,
            "unsafe_cluster_count_delta": 0.0,
            "unsafe_word_ratio_delta": 0.0,
            "risky_window_count_delta": 0.0,
        },
        "scores": {"topk_calibrated_risk": 29.8, "qualifying_text_ai_density": 39.0},
        "candidate_goal": {
            "ai_footprint_gate": {
                "remaining_ai_footprint_drivers": [
                    {"driver": "topk_calibrated_risk", "value": 29.8, "safe_band": 25.0},
                    {"driver": "qualifying_text_ai_density", "value": 39.0, "safe_band": 35.0},
                ]
            }
        },
        "safe_band_evidence_materiality": {"passed": True},
        "safe_band_quality_materiality": {
            "passed": False,
            "reason": "document_repetition_regression",
        },
    }

    assert not _has_safe_band_evidence_repair_movement(row, current_scores=current_scores)


def test_v5_safe_band_evidence_repair_materiality_rejects_near_copy():
    source = (
        "Each lesson ends with my group reflection. "
        "Students talk during this time, and their comments expose what they learned. "
        "From this sharing, I learn from them too."
    )
    near_copy = source.replace("time, and", "time;")
    rebuilt = (
        "I close the lesson with group reflection rather than treating practice as finished. "
        "During that discussion, students name what made sense and where the task still caught them. "
        "Those comments give me a second source of evidence for what I need to demonstrate again."
    )

    rejected = v5_residual_comb._safe_band_evidence_repair_materiality(
        source_text=source,
        candidate_text=near_copy,
        target_sentence="Students talk during this time, and their comments expose what they learned.",
    )
    accepted = v5_residual_comb._safe_band_evidence_repair_materiality(
        source_text=source,
        candidate_text=rebuilt,
        target_sentence="Students talk during this time, and their comments expose what they learned.",
    )

    assert not rejected["passed"]
    assert rejected["reason"] == "near_copy_or_target_route_unchanged"
    assert accepted["passed"]
    assert accepted["changed_sentence_count"] >= 2


def test_v5_author_proxy_context_is_attached_to_all_accept_capable_prompts():
    section = SectionUnit(
        section_id="density_cluster_001",
        heading="Density cluster cleanup",
        text=(
            "The placement helped me understand the client consultation process. "
            "Timing also changed how comfortable the client felt."
        ),
        start_char=0,
        end_char=118,
        paragraph_count=1,
        word_count=17,
        metadata={"before_context": "Before the placement, the process felt abstract.", "after_context": ""},
    )
    density_cluster = {
        "sentence_count": 2,
        "word_count": 17,
        "preview": "The placement helped me understand...",
        "generic_hits": ["process"],
        "transition_count": 1,
    }
    author_proxy_context = {
        "schema_version": "author_proxy_context.v1",
        "active": True,
        "mode": "non_interrupting_author_proxy_draft",
        "review_required": True,
        "primary_mode": "author_grounded_evidence_rebuild",
        "required_inputs": ["specific class observation"],
        "allowed_provenance": ["source_preserved", "inferred_from_draft", "needs_author_confirmation"],
        "quality_bar": {"target": "highest_quality_grounded_candidate"},
        "review_cards": [{"card_id": "target-01", "target_text": "client consultation process"}],
    }
    route_plan = _sample_route_plan()

    prompts = [
        build_direct_scanner_leapfrog_prompt(
            section=section,
            density_cluster=density_cluster,
            route_plan=route_plan,
            author_proxy_context=author_proxy_context,
        ),
        build_risky_window_cleanup_prompt(
            section=section,
            current_scores={"risky_window_count": 1, "unsafe_cluster_count": 1},
            route_plan=route_plan,
            author_proxy_context=author_proxy_context,
        ),
        build_unsafe_cluster_cleanup_prompt(
            section=section,
            density_cluster=density_cluster,
            route_plan=route_plan,
            author_proxy_context=author_proxy_context,
        ),
        build_borderline_verdict_cleanup_prompt(
            current_text=section.text,
            current_scores={"ai": 22.0, "topk": 18.0, "risky_window_count": 0, "unsafe_cluster_count": 0},
            density_gate={"safe": True},
            author_proxy_context=author_proxy_context,
        ),
        build_final_topk_sentence_route_prompt(
            current_scores={"ai": 22.0, "topk": 18.0, "risky_window_count": 0, "unsafe_cluster_count": 0},
            targets=[{"target_id": "t001", "sentence": "Timing also changed how comfortable the client felt."}],
            variant_count=2,
            author_proxy_context=author_proxy_context,
        ),
        build_safe_band_evidence_repair_prompt(
            section=section,
            current_scores={
                "ai": 33.0,
                "topk_calibrated_risk": 30.0,
                "qualifying_text_ai_density": 40.0,
                "risky_window_count": 0,
                "unsafe_cluster_count": 1,
            },
            current_goal={
                "ai_footprint_gate": {
                    "remaining_ai_footprint_drivers": [
                        {"driver": "topk_calibrated_risk", "value": 30.0, "safe_band": 25.0},
                    ],
                },
            },
            variant_count=2,
            author_proxy_context=author_proxy_context,
        ),
    ]

    for prompt in prompts:
        payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))
        assert payload["author_proxy_context"]["mode"] == "non_interrupting_author_proxy_draft"
        assert payload["author_proxy_context"]["review_required"] is True
        assert payload["author_proxy_candidate_audit_contract"]["model_output_schema"]
        assert "author review" in payload["author_proxy_candidate_audit_contract"]["acceptance_note"]


def test_v5_author_proxy_audit_flags_new_concrete_references():
    context = {
        "active": True,
        "review_required": True,
        "required_inputs": ["confirm the real observation"],
        "review_cards": [{
            "card_id": "target-01",
            "provenance": "needs_author_confirmation",
            "target_text": "class support",
            "user_input_needed": "Confirm what happened.",
        }],
    }

    audit = _author_proxy_candidate_audit(
        "The class support improved.",
        "The class support improved in Week 7 at Riverdale.",
        context,
        phase="unit",
    )

    assert audit["active"] is True
    assert audit["review_required"] is True
    assert audit["review_items"][0]["item_id"] == "target-01"
    assert "7" in audit["novel_candidate_references"]["numbers"]
    assert "Riverdale" in audit["novel_candidate_references"]["named_references"]
    assert audit["safety_gate"]["passed"] is False
    assert audit["safety_gate"]["requires_author_review"] is True


def test_v5_safe_band_repair_rejects_failed_author_proxy_safety_gate():
    row = {
        "apply_status": {"applied": True},
        "candidate_goal": {"strict_ai_safe_band_achieved": True},
        "scores": {
            "ai_delta": 6.0,
            "topk_calibrated_risk_delta": 8.0,
            "unsafe_cluster_count_delta": 2.0,
            "risky_window_count_delta": 1.0,
        },
        "incremental": {
            "ai_delta": 1.0,
            "ai_authorship_delta": 1.0,
            "unsafe_cluster_count_delta": 0.0,
            "unsafe_word_ratio_delta": 0.0,
            "risky_window_count_delta": 0.0,
        },
        "safe_band_evidence_materiality": {"passed": True},
        "safe_band_quality_materiality": {"passed": True},
        "author_proxy_audit": {
            "active": True,
            "safety_gate": {
                "passed": False,
                "reason": "candidate_introduced_concrete_references_not_present_in_source",
            },
        },
    }

    assert not _has_safe_band_evidence_repair_movement(row, current_scores={"ai": 32.0})
    assert _best_safe_band_evidence_repair_candidate([row], current_scores={"ai": 32.0}) is None
    assert not _has_full_document_fallback_movement(row)
    assert _best_full_document_candidate([row]) is None


def test_v5_safe_band_repair_skips_pack_after_strict_safe_checkpoint(monkeypatch, tmp_path):
    section = SectionUnit(
        "safe_band_s001",
        "Safe-band",
        "The draft still has a density gap that needs a grounded author-owned repair.",
        0,
        74,
        1,
        13,
        {},
    )
    current_scores = {
        "ai": 31.24,
        "topk_calibrated_risk": 22.296,
        "qualifying_text_ai_density": 38.41,
        "unsafe_cluster_count": 0,
        "risky_window_count": 0,
    }
    current_goal = {
        "ai_footprint_gate": {
            "safe_band_thresholds": {
                "topk_calibrated_risk": 25.0,
                "qualifying_text_ai_density": 35.0,
            },
            "remaining_ai_footprint_drivers": [
                {"driver": "qualifying_text_ai_density", "value": 38.41, "safe_band": 35.0},
            ],
        },
    }

    monkeypatch.setattr(v5_residual_comb, "_safe_band_evidence_repair_sections", lambda *_args, **_kwargs: [section])
    monkeypatch.setattr(v5_residual_comb, "_safe_band_controlled_operation_enabled", lambda: False)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_sentence_replacement_enabled", lambda: True)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_section_repair_should_run", lambda **_kwargs: False)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_evidence_pack_enabled", lambda: True)
    monkeypatch.setattr(
        v5_residual_comb,
        "_safe_band_evidence_pack_sections",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("pack should not run")),
    )

    def fake_sentence_loop(**kwargs):
        return (
            "strict safe text",
            {"input_text": "strict safe text"},
            {"strict_ai_safe_band_achieved": True, "ai_footprint_gate": {"safe_band": True}},
            {
                "ai": 27.79,
                "topk_calibrated_risk": 21.938,
                "qualifying_text_ai_density": 33.18,
                "unsafe_cluster_count": 0,
                "risky_window_count": 0,
            },
            [{"round": 1, "phase": "safe_band_sentence_replacement", "status": "accepted"}],
            kwargs["global_best_candidate"],
        )

    monkeypatch.setattr(v5_residual_comb, "_run_safe_band_sentence_replacement_loop", fake_sentence_loop)

    text, _report, _goal, scores, rounds, _best = v5_residual_comb._run_safe_band_evidence_repair_pass(
        original_text="original text",
        baseline_report={},
        baseline_scores={"ai": 33.0},
        current_text=section.text,
        current_report={},
        current_goal=current_goal,
        current_scores=current_scores,
        gateway=object(),
        output_dir=tmp_path,
        global_best_candidate=None,
    )

    assert text == "strict safe text"
    assert scores["qualifying_text_ai_density"] == 33.18
    assert rounds[-1]["phase"] == "safe_band_evidence_pack"
    assert rounds[-1]["status"] == "skipped"
    assert rounds[-1]["reason"] == "strict_safe_band_already_achieved"


def test_v5_safe_band_density_first_runs_pack_before_single_section_loop(monkeypatch, tmp_path):
    section = SectionUnit(
        "safe_band_density_section_s001",
        "Density",
        "This section still needs coordinated author-owned density repair.",
        0,
        65,
        1,
        9,
        {"selection_reason": "ai_mitigation_density_target_segment"},
    )
    current_scores = {
        "ai": 31.97,
        "topk_calibrated_risk": 25.112,
        "qualifying_text_ai_density": 39.23,
        "unsafe_cluster_count": 0,
        "risky_window_count": 0,
    }
    current_goal = {
        "ai_footprint_gate": {
            "safe_band_thresholds": {
                "topk_calibrated_risk": 25.0,
                "qualifying_text_ai_density": 35.0,
            },
            "remaining_ai_footprint_drivers": [
                {"driver": "qualifying_text_ai_density", "value": 39.23, "safe_band": 35.0},
            ],
        },
    }
    calls: list[str] = []

    monkeypatch.setattr(v5_residual_comb, "_safe_band_evidence_repair_sections", lambda *_args, **_kwargs: [section])
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_first_repair_should_run", lambda **_kwargs: True)

    def fake_pack_attempt(**kwargs):
        calls.append("pack")
        return (
            "strict density pack text",
            {"input_text": "strict density pack text"},
            {"strict_ai_safe_band_achieved": True, "ai_footprint_gate": {"safe_band": True}},
            {
                "ai": 28.0,
                "topk_calibrated_risk": 24.8,
                "qualifying_text_ai_density": 34.7,
                "unsafe_cluster_count": 0,
                "risky_window_count": 0,
            },
            [{"round": 0, "phase": "safe_band_evidence_pack", "status": "accepted"}],
            kwargs["global_best_candidate"],
            True,
        )

    monkeypatch.setattr(v5_residual_comb, "_run_safe_band_evidence_pack_attempt", fake_pack_attempt)
    monkeypatch.setattr(
        v5_residual_comb,
        "_run_safe_band_density_section_repair_loop",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("single-section density loop should be fallback only")),
    )

    text, _report, _goal, scores, rounds, _best = v5_residual_comb._run_safe_band_evidence_repair_pass(
        original_text="original text",
        baseline_report={},
        baseline_scores={"ai": 48.0},
        current_text=section.text,
        current_report={},
        current_goal=current_goal,
        current_scores=current_scores,
        gateway=object(),
        output_dir=tmp_path,
        global_best_candidate=None,
    )

    assert calls == ["pack"]
    assert text == "strict density pack text"
    assert scores["qualifying_text_ai_density"] == 34.7
    assert rounds[0]["phase"] == "safe_band_evidence_pack"


def test_v5_author_proxy_active_prevents_direct_scanner_core_skip(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_SKIP_CORE_AFTER_DIRECT_SCANNER_ACCEPT", raising=False)

    assert _should_skip_core_after_direct_accept(
        direct_scanner_accepted_count=1,
        author_proxy_context={},
    )
    assert not _should_skip_core_after_direct_accept(
        direct_scanner_accepted_count=1,
        author_proxy_context={"active": True},
    )


def test_v5_production_author_proxy_review_status_uses_candidate_audit():
    context = {"active": True, "review_required": True, "review_cards": [{"card_id": "target-01"}]}

    assert v5_production._accepted_author_proxy_review_required(
        [{"author_proxy_audit": {
            "active": True,
            "review_required": True,
            "safety_gate": {"requires_author_review": True},
        }}],
        context=context,
        final_text="Changed text.",
        original_text="Original text.",
    )
    assert v5_production._accepted_author_proxy_review_required(
        [{"author_proxy_audit": {"active": False}}],
        context=context,
        final_text="Changed text.",
        original_text="Original text.",
    )


def test_v5_rewrite_report_author_proxy_status_overrides_external_stamp():
    summary = {
        "outcome": "rewrite_candidate_generated_needs_author_review",
        "strict_goal_status": "mitigation_failed_no_safe_candidate",
        "public_candidate_warning": "author_proxy_candidate_requires_review",
        "best_candidate_author_review_required": True,
        "best_candidate_external_review_required": False,
        "author_proxy_context": {"active": True, "review_required": True},
    }
    rewritten_scan = {
        "ai_risk_badge": {
            "ai_likelihood_score": 18,
            "authorship_rating": {"short_label": "Human / uncertain pattern"},
        },
        "scan_intelligence": {
            "transformation": {
                "contribution": {
                    "human_contribution_ratio": 0.98,
                    "ai_transformation_ratio": 0.02,
                    "calibrated_ai_risk": 0.16,
                }
            }
        },
    }

    html = rewrite_report._outcome_stamp_html(summary, "Author review required", rewritten_scan)

    assert rewrite_report._requires_author_review(summary)
    assert not rewrite_report._requires_external_review(summary)
    assert "AUTHOR REVIEW" in html
    assert "Author review required" in html
    assert "External review required" not in html


def test_v5_rewrite_report_strict_safe_kpi_suppresses_legacy_external_stamp():
    summary = {
        "outcome": "rewrite_candidate_generated_needs_author_review",
        "strict_goal_status": "mitigation_failed_no_safe_candidate",
        "public_candidate_warning": "author_proxy_candidate_requires_review",
        "best_candidate_author_review_required": True,
        "best_candidate_external_review_required": False,
        "strict_safe_band_achieved": True,
        "kpi_finalization_status": "strict_safe_author_review_required",
    }

    assert rewrite_report._requires_author_review(summary)
    assert not rewrite_report._requires_external_review(summary)


def test_v5_rewrite_report_does_not_mark_preserved_original_as_author_review():
    summary = {
        "outcome": "original_preserved",
        "status": "original_preserved",
        "no_text_change": True,
        "author_proxy_context": {"active": True, "review_required": True},
    }

    stamp = rewrite_report._rewrite_stamp(summary, 31.0)

    assert not rewrite_report._requires_author_review(summary)
    assert stamp["label"] != "Author Review"


def test_v5_rewrite_report_includes_author_review_cards():
    section = rewrite_report._author_review_card_section({
        "author_review_cards": [{
            "kind": "candidate_author_review",
            "provenance": "needs_author_confirmation",
            "target_text": "client consultation",
            "instruction": "Candidate added a consultation detail.",
            "user_input_needed": "Confirm the actual consultation detail.",
            "author_task": "Verify the detail before submission.",
        }]
    })

    text = "\n".join(section)
    assert "Author Review Cards" in text
    assert "client consultation" in text
    assert "Confirm the actual consultation detail" in text


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
    assert plan["controlled_expansion"]["move"] in {"concrete_framing", "contrast_or_specific_angle", "scope_limit", "practical_consequence"}


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
    assert payload["writer_style_card"]["reader_level"] == "bachelor_degree"
    assert payload["writer_execution_card"]["style_card"]["target_texture"]
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
    assert "plain bachelor-level" in lowered
    assert "source-level vocabulary" in lowered
    assert "polished institutional" in lowered
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


def test_v5_late_core_acceptance_rejects_tiny_non_structural_gain():
    current_scores = {
        "ai_delta": 8.0,
        "topk_delta": 6.0,
        "unsafe_cluster_count_delta": 2.0,
        "risky_window_count_delta": 2.0,
    }
    tiny_gain = {
        "local_scores": {
            "unsafe_cluster_count": 0,
            "unsafe_word_ratio": 0.0,
            "topk_calibrated_risk": 20.0,
        },
        "incremental": {
            "rank_delta": 0.4,
            "ai_delta": 0.2,
            "topk_delta": 0.2,
            "unsafe_cluster_count_delta": 0.0,
            "risky_window_count_delta": 0.0,
        },
    }
    structural_gain = {
        **tiny_gain,
        "incremental": {
            **tiny_gain["incremental"],
            "unsafe_cluster_count_delta": 1.0,
        },
    }

    assert _has_incremental_movement(tiny_gain)
    assert not _has_core_round_acceptance_movement(tiny_gain, current_scores=current_scores, round_index=5)
    assert _has_core_round_acceptance_movement(structural_gain, current_scores=current_scores, round_index=5)
    assert _has_core_round_acceptance_movement(tiny_gain, current_scores=current_scores, round_index=2)


def test_v5_adaptive_cutoff_uses_scanner_blocker_state(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_SAFE_RISKY_WINDOWS", raising=False)
    scores = {
        "risky_window_count": 0,
        "unsafe_cluster_count": 4,
        "unsafe_word_ratio": 11.0,
    }
    density_gate = {
        "safe": True,
        "unsafe_cluster_count": 4,
        "unsafe_eligible_word_ratio": 11.0,
        "thresholds": {
            "max_unsafe_cluster_count": 4,
            "max_unsafe_eligible_word_ratio": 35.0,
        },
    }

    state = _adaptive_cutoff_blocker_state(scores, density_gate)
    event = _adaptive_cutoff_stop_event(
        phase="before_unsafe_cluster_cleanup",
        current_scores=scores,
        density_gate=density_gate,
    )

    assert state["safe"]
    assert event is not None
    assert event["reason"] == "scanner_blockers_safe"


def test_v5_adaptive_cutoff_keeps_running_when_risky_windows_remain(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_SAFE_RISKY_WINDOWS", raising=False)
    scores = {
        "risky_window_count": 1,
        "unsafe_cluster_count": 4,
        "unsafe_word_ratio": 11.0,
    }
    density_gate = {
        "safe": True,
        "unsafe_cluster_count": 4,
        "unsafe_eligible_word_ratio": 11.0,
        "thresholds": {
            "max_unsafe_cluster_count": 4,
            "max_unsafe_eligible_word_ratio": 35.0,
        },
    }

    state = _adaptive_cutoff_blocker_state(scores, density_gate)

    assert not state["safe"]
    assert _adaptive_cutoff_stop_event(
        phase="before_risky_window_cleanup",
        current_scores=scores,
        density_gate=density_gate,
    ) is None


def test_v5_adaptive_runtime_budget_scales_with_length_and_pressure(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_ADAPTIVE_RUNTIME_BUDGET", raising=False)
    density_gate = {
        "safe": False,
        "unsafe_cluster_count": 4,
        "unsafe_eligible_word_ratio": 20.0,
        "thresholds": {
            "max_unsafe_cluster_count": 4,
            "max_unsafe_eligible_word_ratio": 35.0,
        },
    }
    pressured_density_gate = {
        **density_gate,
        "unsafe_cluster_count": 10,
        "unsafe_eligible_word_ratio": 65.0,
    }
    low_pressure = _adaptive_cutoff_runtime_budget_seconds(
        original_text="one two three four five",
        baseline_density_gate=density_gate,
        baseline_scores={"risky_window_count": 0},
    )
    long_low_pressure = _adaptive_cutoff_runtime_budget_seconds(
        original_text=" ".join(["word"] * 800),
        baseline_density_gate=density_gate,
        baseline_scores={"risky_window_count": 0},
    )
    high_pressure = _adaptive_cutoff_runtime_budget_seconds(
        original_text=" ".join(["word"] * 800),
        baseline_density_gate=pressured_density_gate,
        baseline_scores={"risky_window_count": 6},
    )

    assert low_pressure is not None
    assert long_low_pressure is not None
    assert high_pressure is not None
    assert long_low_pressure > low_pressure
    assert high_pressure > long_low_pressure
    assert high_pressure <= 720.0


def test_v5_author_proxy_safe_band_runs_before_legacy_cleanup(tmp_path, monkeypatch):
    calls: list[str] = []
    current_scores = {
        "ai": 48.0,
        "risky_window_count": 1,
        "unsafe_word_ratio": 52.0,
        "unsafe_cluster_count": 10,
        "topk_calibrated_risk": 80.0,
        "qualifying_text_ai_density": 65.0,
        "ai_authorship": 48.0,
    }
    current_goal = {
        "status": "mitigation_failed_no_safe_candidate",
        "goal_met": False,
        "reason": "candidate_failed_strict_detector_safe_goal",
        "ai_footprint_gate": {
            "safe_band": False,
            "safe_band_thresholds": {
                "topk_calibrated_risk": 25.0,
                "qualifying_text_ai_density": 35.0,
            },
            "remaining_ai_footprint_drivers": [
                {"driver": "topk_calibrated_risk", "value": 80.0, "safe_band": 25.0},
                {"driver": "qualifying_text_ai_density", "value": 65.0, "safe_band": 35.0},
            ],
        },
    }

    monkeypatch.setattr(v5_residual_comb, "_scan_report", lambda _text: {})
    monkeypatch.setattr(
        v5_residual_comb,
        "evaluate_rewrite_goal",
        lambda **_kwargs: SimpleNamespace(to_dict=lambda: dict(current_goal)),
    )
    monkeypatch.setattr(v5_residual_comb, "_with_v5_density_gate", lambda _text, _report, goal: dict(goal))
    monkeypatch.setattr(v5_residual_comb, "_score_summary", lambda *_args, **_kwargs: dict(current_scores))
    monkeypatch.setattr(
        v5_residual_comb,
        "_density_gate_for_report",
        lambda *_args, **_kwargs: {
            "safe": False,
            "unsafe_cluster_count": 10,
            "unsafe_eligible_word_ratio": 52.0,
            "thresholds": {
                "max_unsafe_cluster_count": 4,
                "max_unsafe_eligible_word_ratio": 35.0,
            },
        },
    )
    monkeypatch.setattr(v5_residual_comb, "build_cluster_repair_units", lambda **_kwargs: [])
    monkeypatch.setattr(v5_residual_comb, "_safe_band_evidence_repair_should_run", lambda **_kwargs: True)
    monkeypatch.setattr(v5_residual_comb, "_borderline_verdict_should_run", lambda **_kwargs: False)
    monkeypatch.setattr(v5_residual_comb, "_final_topk_sentence_route_should_run", lambda **_kwargs: False)
    monkeypatch.setattr(v5_residual_comb, "_safe_band_density_first_repair_should_run", lambda **_kwargs: False)

    def fake_safe_band_pack(**kwargs):
        calls.append("safe_band")
        return (
            kwargs["current_text"],
            kwargs["current_report"],
            kwargs["current_goal"],
            kwargs["current_scores"],
            [{"phase": "safe_band_evidence_pack", "status": "skipped"}],
            kwargs["global_best_candidate"],
            False,
        )

    def fake_risky_cleanup(**kwargs):
        calls.append("risky_window_cleanup")
        return (
            kwargs["current_text"],
            kwargs["current_report"],
            kwargs["current_goal"],
            kwargs["current_scores"],
            [{"phase": "risky_window_cleanup", "status": "skipped"}],
            kwargs["global_best_candidate"],
        )

    monkeypatch.setattr(v5_residual_comb, "_run_safe_band_evidence_pack_attempt", fake_safe_band_pack)
    monkeypatch.setattr(v5_residual_comb, "_run_risky_window_cleanup_pass", fake_risky_cleanup)

    payload = run_v5_residual_cluster_comb_experiment(
        input_text="The submitted draft needs author-owned repair.",
        output_dir=tmp_path,
        max_rounds=1,
        risky_window_cleanup_rounds=1,
        unsafe_cluster_cleanup_rounds=0,
        final_risky_window_cleanup_rounds=0,
        direct_scanner_leapfrog_rounds=0,
        max_seconds=999,
        api_key="test-key",
    )

    assert calls == ["safe_band", "risky_window_cleanup"]
    assert payload["phase_order"]["safe_band_evidence_repair"]["pre_core_author_proxy_pack"] is True
    assert payload["safe_band_evidence_repair_rounds"][0]["phase"] == "safe_band_evidence_pack"


def test_v5_borderline_verdict_runs_only_after_local_blockers_are_safe(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V5_BORDERLINE_VERDICT_CLEANUP", raising=False)
    safe_density = {
        "safe": True,
        "unsafe_cluster_count": 2,
        "unsafe_eligible_word_ratio": 4.0,
        "thresholds": {
            "max_unsafe_cluster_count": 4,
            "max_unsafe_eligible_word_ratio": 35.0,
        },
    }
    unsafe_density = {
        **safe_density,
        "safe": False,
        "unsafe_cluster_count": 8,
        "unsafe_eligible_word_ratio": 50.0,
    }

    assert _borderline_verdict_should_run(
        current_scores={"ai": 49.0, "external": 42.0, "risky_window_count": 0},
        density_gate=safe_density,
    )
    assert not _borderline_verdict_should_run(
        current_scores={"ai": 49.0, "external": 42.0, "risky_window_count": 0},
        density_gate=unsafe_density,
    )
    assert not _borderline_verdict_should_run(
        current_scores={"ai": 35.0, "external": 30.0, "risky_window_count": 0},
        density_gate=safe_density,
    )


def test_v5_borderline_prompt_targets_global_texture_not_local_cleanup():
    prompt = build_borderline_verdict_cleanup_prompt(
        current_text="First paragraph stays here.\n\nSecond paragraph stays here.",
        current_scores={
            "ai": 49.0,
            "external": 42.0,
            "topk": 79.0,
            "unsafe_cluster_count": 2,
            "risky_window_count": 0,
        },
        density_gate={
            "safe": True,
            "top_sentence_targets": [
                {
                    "sentence_id": "s001",
                    "preview": "This shift has made the role of teachers even more important.",
                    "top10_ratio": 0.7,
                    "predictability_risk": 0.5,
                }
            ],
        },
        variant_count=2,
    )
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "borderline_whole_document_texture_pass"
    assert payload["length_policy"]["preserve_paragraph_count"] == 2
    assert "whole-document texture" in payload["scanner_state"]["remaining_problem"]
    assert payload["target_outcome"]["preferred_ai_below"] == 45.0
    assert "preferred_external_below" not in payload["target_outcome"]
    assert payload["target_outcome"]["max_risky_windows_after"] == 3.0
    assert payload["writer_variant_plan"][0]["lane_goal"] == "plain_source_near_route"
    assert any("polished abstract bridge" in item for item in payload["editorial_action"])
    assert payload["output_schema"]["variants"][0]["text"] == "full replacement document"


def test_v5_borderline_acceptance_requires_verdict_gain_without_structural_regression():
    good = {
        "apply_status": {"applied": True},
        "incremental": {
            "ai_delta": 1.2,
            "external_delta": 0.1,
            "topk_delta": -0.1,
            "topk_calibrated_risk_delta": 0.0,
            "unsafe_cluster_count_delta": 0.0,
            "risky_window_count_delta": 0.0,
        },
    }
    regressed = {
        **good,
        "incremental": {
            **good["incremental"],
            "unsafe_cluster_count_delta": -1.0,
        },
    }
    verdict_clear_with_bounded_regression = {
        "apply_status": {"applied": True},
        "scores": {
            "ai": 43.5,
            "ai_authorship": 44.0,
            "external": 40.5,
            "risky_window_count": 2,
            "unsafe_cluster_count": 5,
            "unsafe_word_ratio": 20.0,
        },
        "incremental": {
            "ai_delta": 4.0,
            "external_delta": 2.0,
            "topk_delta": 1.0,
            "topk_calibrated_risk_delta": 2.0,
            "unsafe_cluster_count_delta": -3.0,
            "risky_window_count_delta": -2.0,
        },
    }
    weak = {
        **good,
        "incremental": {
            **good["incremental"],
            "ai_delta": 0.2,
            "external_delta": 0.2,
        },
    }

    assert _has_borderline_verdict_movement(good)
    assert not _has_borderline_verdict_movement(regressed)
    assert _has_borderline_verdict_movement(verdict_clear_with_bounded_regression)
    assert not _has_borderline_verdict_movement(weak)


def test_v5_borderline_selector_prefers_boundary_candidate_over_mild_gain():
    mild_gain = {
        "variant_id": "v1",
        "apply_status": {"applied": True},
        "scores": {
            "ai": 46.4,
            "ai_authorship": 46.0,
            "external": 45.0,
            "risky_window_count": 0,
            "unsafe_cluster_count": 2,
            "unsafe_word_ratio": 4.0,
            "ai_delta": 20.0,
        },
        "incremental": {
            "ai_delta": 3.0,
            "external_delta": 2.0,
            "external_ai_flag_risk_delta": 2.0,
            "ai_authorship_delta": 3.0,
            "topk_delta": 1.0,
            "topk_calibrated_risk_delta": 1.0,
            "risky_window_count_delta": 0.0,
            "unsafe_cluster_count_delta": 0.0,
        },
    }
    crosses_boundary = {
        "variant_id": "v2",
        "apply_status": {"applied": True},
        "scores": {
            "ai": 44.2,
            "ai_authorship": 44.0,
            "external": 40.7,
            "risky_window_count": 3,
            "unsafe_cluster_count": 5,
            "unsafe_word_ratio": 20.0,
            "ai_delta": 22.0,
        },
        "incremental": {
            "ai_delta": 2.0,
            "external_delta": 1.0,
            "external_ai_flag_risk_delta": 1.0,
            "ai_authorship_delta": 2.0,
            "topk_delta": 0.5,
            "topk_calibrated_risk_delta": 1.0,
            "risky_window_count_delta": -1.0,
            "unsafe_cluster_count_delta": -3.0,
        },
    }

    assert _best_borderline_verdict_candidate([mild_gain, crosses_boundary])["variant_id"] == "v2"


def test_v5_borderline_accepts_scanner_badge_boundary_when_external_score_is_lagging():
    scanner_badge_clear = {
        "variant_id": "v5",
        "apply_status": {"applied": True},
        "candidate_report": {
            "ai_risk_badge": {
                "authorship_rating_code": "possible_ai_assisted",
                "authorship_rating_label": "Possible AI-Assisted",
                "tier": "AMBER",
                "authorship_rating": {"code": "possible_ai_assisted", "risk_level": "medium"},
            }
        },
        "scores": {
            "ai": 44.8,
            "ai_authorship": 45.0,
            "external": 99.0,
            "risky_window_count": 0,
            "unsafe_cluster_count": 4,
            "unsafe_word_ratio": 20.0,
        },
        "incremental": {
            "ai_delta": 3.0,
            "external_delta": 1.0,
            "external_ai_flag_risk_delta": 2.0,
            "ai_authorship_delta": 3.0,
            "topk_delta": 1.0,
            "topk_calibrated_risk_delta": 2.0,
            "risky_window_count_delta": 0.0,
            "unsafe_cluster_count_delta": -2.0,
        },
    }
    likely_ai_badge = {
        **scanner_badge_clear,
        "candidate_report": {
            "ai_risk_badge": {
                "authorship_rating_code": "likely_ai",
                "authorship_rating": {"code": "likely_ai", "risk_level": "high"},
            }
        },
    }

    assert _borderline_verdict_candidate_crosses_boundary(scanner_badge_clear)
    assert _has_borderline_verdict_movement(scanner_badge_clear)
    assert not _borderline_verdict_candidate_crosses_boundary(likely_ai_badge)


def test_v5_borderline_feedback_for_rejected_boundary_candidate():
    row = {
        "variant_id": "v2",
        "apply_status": {"applied": True},
        "candidate_report": {
            "ai_risk_badge": {
                "authorship_rating_code": "possible_ai_assisted",
                "authorship_rating": {"code": "possible_ai_assisted", "risk_level": "medium"},
            }
        },
        "scores": {
            "ai": 44.5,
            "ai_authorship": 44.0,
            "external": 41.5,
            "external_ai_flag_risk": 44.0,
            "topk": 81.0,
            "topk_calibrated_risk": 49.0,
            "risky_window_count": 2,
            "unsafe_cluster_count": 7,
            "unsafe_word_ratio": 24.0,
        },
        "incremental": {
            "ai_delta": 4.0,
            "external_delta": 6.0,
            "topk_delta": -2.0,
            "topk_calibrated_risk_delta": -7.0,
            "risky_window_count_delta": -2.0,
            "unsafe_cluster_count_delta": -4.0,
        },
    }

    feedback = _borderline_rejected_candidate_feedback(
        row,
        current_scores={
            "topk": 78.0,
            "topk_calibrated_risk": 42.0,
            "risky_window_count": 0,
            "unsafe_cluster_count": 3,
        },
    )

    assert feedback["status"] == "previous_candidate_reduced_verdict_score_but_failed_local_safety"
    assert feedback["previous_variant_id"] == "v2"
    assert any("unsafe cluster" in item for item in feedback["must_fix"])
    assert "external_ai_flag_risk" not in feedback["previous_scores"]


def test_v5_incremental_deltas_ignore_external_ai_flag_as_second_judge():
    scores = {
        "ai": 31.0,
        "topk": 72.0,
        "external": 30.0,
        "ai_authorship": 32.0,
        "external_ai_flag_risk": 80.0,
    }
    current_scores = {
        "ai": 34.0,
        "topk": 75.0,
        "external": 33.0,
        "ai_authorship": 34.0,
        "external_ai_flag_risk": 20.0,
    }

    deltas = _incremental_deltas(scores, current_scores)

    assert deltas["ai_delta"] == 3.0
    assert deltas["topk_delta"] == 3.0
    assert deltas["external_delta"] == 3.0
    assert deltas["ai_authorship_delta"] == 2.0
    assert "external_ai_flag_risk_delta" not in deltas


def test_v5_full_document_variant_rejects_compression_before_scan(tmp_path):
    current_text = " ".join(["source"] * 120)
    candidate = RecompositionVariant(variant_id="v1", text="short text", word_count=2)

    row = _score_full_document_variant(
        original_text=current_text,
        baseline_report={},
        baseline_scores={},
        current_text=current_text,
        current_scores={"ai": 50.0},
        variant=candidate,
        output_dir=tmp_path,
        label="compressed",
    )

    assert row["apply_status"]["applied"] is False
    assert row["apply_status"]["reason"] == "candidate_compressed_too_much"


def test_v5_author_proxy_document_window_rejects_compression_before_scan(tmp_path):
    current_text = " ".join(f"source{i}" for i in range(120))
    section = SectionUnit(
        section_id="w001",
        heading="Full document window",
        text=current_text,
        start_char=0,
        end_char=len(current_text),
        paragraph_count=1,
        word_count=120,
        metadata={},
    )
    candidate = RecompositionVariant(
        variant_id="v1",
        text=" ".join(f"candidate{i}" for i in range(96)),
        word_count=96,
    )

    row = v5_residual_comb._score_residual_variant(
        original_text=current_text,
        baseline_report={},
        baseline_scores={},
        current_text=current_text,
        current_scores={"ai": 50.0},
        section=section,
        variant=candidate,
        output_dir=tmp_path,
        label="window_v1",
        author_proxy_context={"active": True},
    )

    assert row["apply_status"]["applied"] is False
    assert row["apply_status"]["reason"] == "author_proxy_document_window_compressed_too_much"
    assert row["apply_status"]["minimum_candidate_words"] == 108


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


def test_v5_production_runtime_budget_uses_adaptive_scanner_budget(monkeypatch):
    monkeypatch.delenv("REWRITE_SOFT_TIME_LIMIT_SECONDS", raising=False)
    monkeypatch.setattr(v5_production, "_v5_adaptive_runtime_budget_seconds", lambda **_: 300.0)
    config = {
        "runtime_base_seconds": 900,
        "runtime_seconds_per_100_words": 40.0,
        "runtime_min_seconds": 900,
        "runtime_max_seconds": 1800,
        "runtime_soft_limit_buffer_seconds": 60,
    }

    budget = v5_production._v5_runtime_budget_seconds(
        "word " * 500,
        config,
        original_report={"status": "ok"},
    )

    assert budget == 300


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
        api_key="test-key",
        risky_window_cleanup_rounds=0,
        unsafe_cluster_cleanup_rounds=4,
        final_risky_window_cleanup_rounds=0,
        max_seconds=60,
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


def test_v5_compact_payload_includes_borderline_verdict_rounds():
    payload = {
        "stage": "v5_residual_cluster_comb",
        "baseline_scores": {"ai": 70.0},
        "final_scores": {"ai": 49.0},
        "rounds": [],
        "risky_window_cleanup_rounds": [],
        "unsafe_cluster_cleanup_rounds": [],
        "final_risky_window_cleanup_rounds": [],
        "borderline_verdict_cleanup_rounds": [
            {
                "round": 1,
                "phase": "borderline_verdict_cleanup",
                "status": "accepted",
                "reason": "accepted_borderline_verdict_movement",
                "accepted": {
                    "section_id": "full_document",
                    "variant_id": "v2",
                    "scores": {"ai_delta": 2.0},
                },
                "candidates": [{"variant_id": "v1"}, {"variant_id": "v2"}],
            }
        ],
    }

    compact = v5_production._compact_v5_payload(payload)

    assert compact["accepted_rounds"] == 1
    assert compact["borderline_verdict_cleanup_rounds"][0]["status"] == "accepted"
    assert compact["borderline_verdict_cleanup_rounds"][0]["candidate_count"] == 2


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


def test_v5_global_best_fallback_rejects_density_safe_ai_breakthrough_with_unsafe_regression():
    current_scores = {
        "ai": 35.18,
        "topk": 73.68,
        "unsafe_cluster_count": 2,
        "risky_window_count": 0,
        "unsafe_cluster_count_delta": 10.0,
        "topk_delta": 19.09,
        "topk_calibrated_risk_delta": 49.691,
    }
    candidate = {
        "apply_status": {"applied": True},
        "candidate_goal": {
            "eligible_span_density_gate": {
                "safe": True,
                "unsafe_cluster_count": 4,
            }
        },
        "scores": {
            "ai": 33.58,
            "topk": 74.81,
            "risky_window_count": 0,
            "unsafe_cluster_count": 4,
            "ai_delta": 14.74,
            "topk_delta": 17.96,
            "topk_calibrated_risk_delta": 48.652,
            "external_ai_flag_risk_delta": 20.971,
            "unsafe_cluster_count_delta": 8.0,
            "risky_window_count_delta": 1.0,
        },
    }

    assert _would_discard_structural_progress(candidate["scores"], current_scores)
    assert not _full_document_candidate_beats_scores(candidate, current_scores)


def test_v5_global_best_fallback_keeps_density_safe_ai_breakthrough_without_unsafe_regression():
    current_scores = {
        "ai": 35.18,
        "topk": 73.68,
        "topk_calibrated_risk": 31.2,
        "unsafe_word_ratio": 12.0,
        "unsafe_cluster_count": 4,
        "risky_window_count": 0,
        "unsafe_cluster_count_delta": 8.0,
        "topk_delta": 19.09,
        "topk_calibrated_risk_delta": 49.691,
    }
    candidate = {
        "apply_status": {"applied": True},
        "candidate_goal": {
            "eligible_span_density_gate": {
                "safe": True,
                "unsafe_cluster_count": 2,
            }
        },
        "scores": {
            "ai": 33.58,
            "topk": 74.81,
            "topk_calibrated_risk": 31.0,
            "unsafe_word_ratio": 10.0,
            "risky_window_count": 0,
            "unsafe_cluster_count": 2,
            "ai_delta": 14.74,
            "topk_delta": 17.96,
            "topk_calibrated_risk_delta": 48.652,
            "external_ai_flag_risk_delta": 20.971,
            "unsafe_cluster_count_delta": 10.0,
            "risky_window_count_delta": 1.0,
        },
    }

    assert _full_document_candidate_beats_scores(candidate, current_scores)


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
            "eligible_span_density_gate": {
                "source": "scanner.repair_units_v2",
                "safe": False,
                "unsafe_cluster_count": 10,
            },
        }
    ))
    monkeypatch.setattr(
        v5_production,
        "_with_v5_density_gate",
        lambda _text, _report, goal: {
            **dict(goal),
            "eligible_span_density_gate": {
                "source": "eligible_span_density_v1",
                "safe": True,
                "unsafe_cluster_count": 4,
            },
        },
    )
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
    assert result["status"] == "partial_candidate_not_strict_safe"
    assert summary["rewrite_pipeline_version"] == "rewrite_v5_residual_cluster_comb"
    assert summary["rewrite_engine_mode"] == "v5_residual_cluster_comb_production"
    assert summary["candidate_generation_status"]["accepted_count"] == 1
    assert summary["partial_rewrite_preserved"] is True
    assert summary["partial_rewrite_preservation_reason"] == "safe_progress_kept_despite_strict_goal_miss"
    assert summary["public_candidate_warning"] == "candidate_not_strict_safe"
    assert summary["best_candidate_external_review_required"] is False
    assert summary["kpi_finalization_status"] == "partial_candidate_not_strict_safe"
    assert summary["v5_scores"]["deltas"]["ai_delta"] == 6.0
    assert summary["rewrite_effective_config"]["provider_routing"] == provider_routing
    assert summary["rewrite_effective_config"]["planner_model"] == "z-ai/glm-5.1"
    assert summary["final_text"] == "This is the rewritten document."
    assert summary["rewrite_goal_status"]["eligible_span_density_gate"]["source"] == "eligible_span_density_v1"
    assert summary["rewrite_goal_status"]["eligible_span_density_gate"]["safe"] is True
    assert summary["rewrite_goal_status"]["eligible_span_density_gate"]["unsafe_cluster_count"] == 4
    layer = summary["rewrite_layers"]["v5_residual_cluster_comb"]
    assert layer["phase_order"]["unsafe_cluster_first"] is True
    assert emitted_checkpoints
    assert emitted_checkpoints[0]["status"] == "partial_candidate_not_strict_safe"
    assert emitted_checkpoints[0]["final_text"] == "This is the rewritten document."
    assert emitted_checkpoints[0]["summary"]["checkpoint_recovery_available"] is True
    assert emitted_checkpoints[0]["summary"]["best_candidate_external_review_required"] is False


def test_v5_author_proxy_context_is_built_from_mitigation_plan():
    context = v5_production._build_author_proxy_context(
        {
            "input_text": "The draft mentions classroom support but not the author's exact observation.",
            "ai_mitigation": {
                "primary_mode": "author_grounded_evidence_rebuild",
                "readiness": {
                    "requires_user_input": True,
                    "required_inputs": ["author observation", "source detail"],
                },
                "target_segments": [{
                    "paragraph_id": "p001",
                    "bucket": "source_grounding",
                    "lever": "add_concrete_observation",
                    "text": "classroom support",
                    "action": "Add author-owned evidence for this claim.",
                    "user_input_needed": "The real classroom observation.",
                }],
                "component_actions": [{
                    "component": "evidence bridge",
                    "action": "Confirm the link to a source.",
                    "user_input_needed": "Source page or lecture note.",
                }],
            },
        },
        "The draft mentions classroom support but not the author's exact observation.",
    )

    assert context["active"] is True
    assert context["review_required"] is True
    assert context["mode"] == "non_interrupting_author_proxy_draft"
    assert context["required_inputs"] == ["author observation", "source detail"]
    assert context["quality_bar"]["target"] == "highest_quality_grounded_candidate"
    assert context["quality_bar"]["basis"] == "submitted_content_only"
    assert context["review_cards"][0]["card_id"] == "target-01"
    assert context["review_cards"][0]["provenance"] == "needs_author_confirmation"
    evidence_contract = context["authorship_evidence_contract"]
    assert evidence_contract["schema_version"] == "authorship_evidence_contract.v1"
    assert evidence_contract["basis"] == "submitted_content_only"
    assert evidence_contract["required_inputs"] == ["author observation", "source detail"]
    assert evidence_contract["evidence_slots"][0]["slot_id"] == "target-01"
    assert any("qualifying AI density" in item for item in evidence_contract["kpi_alignment"])


def test_v5_author_proxy_context_defaults_to_non_interrupting_author_proxy():
    context = v5_production._build_author_proxy_context(
        {"input_text": "The submitted draft already contains the author's examples."},
        "The submitted draft already contains the author's examples.",
    )

    assert context["active"] is True
    assert context["review_required"] is False
    assert context["mode"] == "non_interrupting_author_proxy_draft"
    assert context["review_cards"] == []
    assert context["quality_bar"]["target"] == "highest_quality_grounded_candidate"
    assert context["authorship_evidence_contract"]["basis"] == "submitted_content_only"


def test_v5_author_proxy_runtime_uses_legacy_budget_floor(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_RUNTIME_BASE_SECONDS", "900")
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_RUNTIME_SECONDS_PER_100_WORDS", "0")
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_RUNTIME_MIN_SECONDS", "900")
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_RUNTIME_MAX_SECONDS", "1800")
    monkeypatch.setenv("REWRITE_SOFT_TIME_LIMIT_SECONDS", "1800")
    monkeypatch.setattr(v5_production, "_v5_adaptive_runtime_budget_seconds", lambda **_: 120.0)

    config = v5_production._production_config()
    budget = v5_production._v5_runtime_budget_seconds(
        "The draft needs author-owned evidence.",
        config,
        original_report={
            "input_text": "The draft needs author-owned evidence.",
            "ai_mitigation": {
                "readiness": {
                    "requires_user_input": True,
                    "required_inputs": ["author observation"],
                },
                "target_segments": [{
                    "paragraph_id": "p001",
                    "text": "author-owned evidence",
                    "action": "Confirm evidence.",
                    "user_input_needed": "author observation",
                }],
            },
        },
    )

    assert budget == 900


def test_v5_production_author_proxy_candidate_requires_author_review(tmp_path, monkeypatch):
    emitted_checkpoints: list[dict] = []
    captured_author_contexts: list[dict] = []

    def fake_residual_comb(**kwargs):
        captured_author_contexts.append(kwargs["author_proxy_context"])
        callback = kwargs.get("accepted_checkpoint_callback")
        assert callback is not None
        callback({
            "schema_version": "rewrite_v5_accepted_checkpoint.v1",
            "stage": "v5_residual_cluster_comb",
            "sequence": 1,
            "phase": "unsafe_cluster_cleanup",
            "round": 1,
            "reason": "accepted_unsafe_cluster_movement",
            "baseline_scores": {"ai": 46.0},
            "scores": {"ai": 36.0},
            "goal": {"goal_met": False},
            "accepted": {
                "section_id": "density_cluster_001",
                "variant_id": "v1",
                "author_review_items": [{
                    "item_id": "candidate-01",
                    "provenance": "needs_author_confirmation",
                    "target_text": "narrower classroom observation",
                    "generated_text": "Candidate narrowed the claim into a classroom observation.",
                    "user_input_needed": "Confirm the observed classroom detail.",
                    "author_task": "Replace this with your own exact observation if needed.",
                }],
                "author_proxy_audit": {
                    "active": True,
                    "review_required": True,
                    "safety_gate": {"requires_author_review": True},
                },
            },
            "rewritten_document": "This rewritten candidate adds a narrower classroom observation.",
        })
        return {
            "stage": "v5_residual_cluster_comb",
            "baseline_scores": {"ai": 46.0, "topk": 80.0, "rank": 120.0},
            "final_scores": {"ai": 36.0, "topk": 72.0, "rank": 100.0},
            "goal": {"status": "mitigation_failed_no_safe_candidate", "goal_met": False},
            "rounds": [{
                "round": 1,
                "status": "accepted",
                "accepted": {
                    "section_id": "density_cluster_001",
                    "variant_id": "v1",
                    "text": "This rewritten candidate adds a narrower classroom observation.",
                    "word_count": 8,
                    "scores": {"ai_delta": 10.0},
                    "author_review_items": [{
                        "item_id": "candidate-01",
                        "provenance": "needs_author_confirmation",
                        "target_text": "narrower classroom observation",
                        "generated_text": "Candidate narrowed the claim into a classroom observation.",
                        "user_input_needed": "Confirm the observed classroom detail.",
                        "author_task": "Replace this with your own exact observation if needed.",
                    }],
                    "author_proxy_audit": {
                        "active": True,
                        "review_required": True,
                        "safety_gate": {"requires_author_review": True},
                    },
                },
                "candidates": [{"variant_id": "v1"}],
            }],
            "phase_order": {"unsafe_cluster_first": True, "reason": "eligible_span_density_unsafe"},
            "rewritten_document": "This rewritten candidate adds a narrower classroom observation.",
        }

    monkeypatch.setattr(v5_production, "run_v5_residual_cluster_comb_experiment", fake_residual_comb)
    monkeypatch.setattr(v5_production, "_scan_report", lambda text: {
        "input_text": text,
        "ai_score": 36.0,
        "ai_risk_badge": {"ai_likelihood_score": 36.0},
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
            "input_text": "The draft says classroom support helped the learner improve.",
            "ai_score": 46.0,
            "ai_risk_badge": {"ai_likelihood_score": 46.0},
            "findings": {},
            "ai_mitigation": {
                "primary_mode": "author_grounded_evidence_rebuild",
                "readiness": {"requires_user_input": True, "required_inputs": ["real observation"]},
                "target_segments": [{
                    "paragraph_id": "p001",
                    "bucket": "source_grounding",
                    "lever": "specificity",
                    "text": "classroom support helped the learner improve",
                    "action": "Confirm this with an author-owned observation.",
                    "user_input_needed": "Real observed classroom detail.",
                }],
            },
        },
        output_dir=str(tmp_path),
        model="deepseek/deepseek-v3.2",
        checkpoint_callback=emitted_checkpoints.append,
    )

    summary = json.loads(Path(result["json_path"]).read_text())
    assert captured_author_contexts[0]["active"] is True
    assert result["status"] == "partial_candidate_not_strict_safe"
    assert summary["outcome"] == "partial_candidate_not_strict_safe"
    assert summary["public_candidate_warning"] == "candidate_not_strict_safe"
    assert summary["kpi_finalization_status"] == "partial_candidate_not_strict_safe"
    assert summary["best_candidate_author_review_required"] is False
    assert summary["best_candidate_external_review_required"] is False
    assert summary["author_review_cards"][0]["card_id"] == "target-01"
    assert any(
        card.get("kind") == "candidate_author_review"
        and card.get("target_text") == "narrower classroom observation"
        for card in summary["author_review_cards"]
    )
    assert emitted_checkpoints[0]["status"] == "partial_candidate_not_strict_safe"
    assert emitted_checkpoints[0]["summary"]["best_candidate_author_review_required"] is False
    assert any(
        card.get("kind") == "candidate_author_review"
        for card in emitted_checkpoints[0]["summary"]["author_review_cards"]
    )


def test_v5_production_reports_strict_safe_author_proxy_kpi_separately(tmp_path, monkeypatch):
    def fake_residual_comb(**kwargs):
        callback = kwargs.get("accepted_checkpoint_callback")
        assert callback is not None
        accepted = {
            "section_id": "safe_band_sentence_replacement",
            "variant_id": "v2",
            "text": "This is the strict-safe author-proxy candidate.",
            "word_count": 7,
            "scores": {
                "ai": 27.79,
                "topk_calibrated_risk": 21.938,
                "qualifying_text_ai_density": 33.18,
                "unsafe_cluster_count": 0,
            },
            "author_proxy_audit": {
                "active": True,
                "review_required": True,
                "safety_gate": {"passed": True, "requires_author_review": True},
            },
        }
        goal = {
            "status": "ai_mitigated",
            "goal_met": True,
            "strict_ai_safe_band_achieved": True,
            "ai_footprint_gate": {"safe_band": True, "remaining_ai_footprint_drivers": []},
        }
        callback({
            "schema_version": "rewrite_v5_accepted_checkpoint.v1",
            "stage": "v5_residual_cluster_comb",
            "sequence": 1,
            "phase": "safe_band_sentence_replacement",
            "round": 1,
            "reason": "accepted_safe_band_sentence_replacement",
            "baseline_scores": {"ai": 31.24},
            "scores": accepted["scores"],
            "goal": goal,
            "accepted": accepted,
            "rewritten_document": accepted["text"],
        })
        return {
            "stage": "v5_residual_cluster_comb",
            "baseline_scores": {"ai": 31.24, "topk": 64.67, "rank": 60.945},
            "final_scores": {
                "ai": 27.79,
                "topk": 64.28,
                "rank": 59.635,
                "topk_calibrated_risk": 21.938,
                "qualifying_text_ai_density": 33.18,
                "unsafe_cluster_count": 0,
            },
            "goal": goal,
            "safe_band_evidence_repair_rounds": [{
                "round": 1,
                "phase": "safe_band_sentence_replacement",
                "status": "accepted",
                "accepted": accepted,
            }],
            "rounds": [],
            "phase_order": {"unsafe_cluster_first": False, "reason": "safe_band_evidence_repair"},
            "rewritten_document": accepted["text"],
        }

    monkeypatch.setattr(v5_production, "run_v5_residual_cluster_comb_experiment", fake_residual_comb)
    monkeypatch.setattr(v5_production, "_scan_report", lambda text: {
        "input_text": text,
        "ai_score": 27.79,
        "ai_risk_badge": {"ai_likelihood_score": 27.79},
        "findings": {},
    })
    monkeypatch.setattr(v5_production, "evaluate_rewrite_goal", lambda **_: SimpleNamespace(
        to_dict=lambda: {
            "status": "ai_mitigated",
            "goal_met": True,
            "strict_ai_safe_band_achieved": True,
            "ai_footprint_gate": {"safe_band": True, "remaining_ai_footprint_drivers": []},
        }
    ))
    monkeypatch.setattr(v5_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))

    result = v5_production.run_rewrite_pipeline_v5(
        detect_json={
            "input_text": "This is the original author-proxy draft.",
            "ai_score": 31.24,
            "ai_risk_badge": {"ai_likelihood_score": 31.24},
            "findings": {},
            "ai_mitigation": {
                "primary_mode": "author_grounded_evidence_rebuild",
                "readiness": {"requires_user_input": True, "required_inputs": ["real observation"]},
                "target_segments": [{
                    "paragraph_id": "p001",
                    "bucket": "source_grounding",
                    "lever": "specificity",
                    "text": "original author-proxy draft",
                    "action": "Confirm this with author-owned evidence.",
                    "user_input_needed": "Real observed detail.",
                }],
            },
        },
        output_dir=str(tmp_path),
        model="deepseek/deepseek-v3.2",
    )

    summary = json.loads(Path(result["json_path"]).read_text())
    assert result["status"] == "rewrite_candidate_generated_needs_author_review"
    assert summary["strict_goal_status"] == "ai_mitigated"
    assert summary["rewrite_goal_status"]["goal_met"] is True
    assert summary["strict_safe_band_achieved"] is True
    assert summary["kpi_finalization_status"] == "strict_safe_author_review_required"
    assert summary["best_candidate_author_review_required"] is True
    assert summary["best_candidate_external_review_required"] is False


def test_v5_production_recomputes_full_goal_when_experiment_goal_is_compact(tmp_path, monkeypatch):
    def fake_residual_comb(**_kwargs):
        return {
            "stage": "v5_residual_cluster_comb",
            "baseline_scores": {"ai": 48.0, "topk": 80.0, "rank": 100.0},
            "final_scores": {
                "ai": 32.0,
                "topk": 55.0,
                "rank": 70.0,
                "topk_calibrated_risk": 21.0,
                "qualifying_text_ai_density": 20.0,
                "unsafe_cluster_count": 0,
            },
            "goal": {
                "status": "mitigation_failed_no_safe_candidate",
                "goal_met": False,
                "reason": "compact_experiment_goal",
            },
            "rounds": [{
                "round": 1,
                "status": "accepted",
                "accepted": {
                    "section_id": "safe_band_repair",
                    "variant_id": "v1",
                    "text": "safe candidate",
                },
                "candidates": [{"variant_id": "v1"}],
            }],
            "phase_order": {"unsafe_cluster_first": False, "reason": "default_route_then_cleanup"},
            "rewritten_document": "This is the safe candidate.",
        }

    monkeypatch.setattr(v5_production, "run_v5_residual_cluster_comb_experiment", fake_residual_comb)
    monkeypatch.setattr(v5_production, "_scan_report", lambda text: {
        "input_text": text,
        "ai_score": 32.0,
        "ai_risk_badge": {"ai_likelihood_score": 32.0},
        "findings": {},
    })
    monkeypatch.setattr(v5_production, "evaluate_rewrite_goal", lambda **_: SimpleNamespace(
        to_dict=lambda: {
            "status": "ai_mitigated",
            "goal_met": True,
            "strict_ai_safe_band_achieved": True,
            "ai_footprint_gate": {
                "safe_band": True,
                "remaining_ai_footprint_drivers": [],
            },
            "eligible_span_density_gate": {"safe": True},
        }
    ))
    monkeypatch.setattr(v5_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))

    result = v5_production.run_rewrite_pipeline_v5(
        detect_json={
            "input_text": "This is the original risky draft.",
            "ai_score": 48.0,
            "ai_risk_badge": {"ai_likelihood_score": 48.0},
            "findings": {},
        },
        output_dir=str(tmp_path),
        model="deepseek/deepseek-v3.2",
    )

    summary = json.loads(Path(result["json_path"]).read_text())
    assert result["status"] == "ai_mitigated"
    assert summary["strict_goal_status"] == "ai_mitigated"
    assert summary["rewrite_goal_status"]["ai_footprint_gate"]["safe_band"] is True
    assert summary["strict_safe_band_achieved"] is True
    assert summary["kpi_finalization_status"] == "strict_safe_auto_finalized"


def test_v5_production_treats_global_best_fallback_as_selected_candidate(tmp_path, monkeypatch):
    def fake_residual_comb(**kwargs):
        return {
            "stage": "v5_residual_cluster_comb",
            "baseline_scores": {"ai": 55.0, "rank": 140.0},
            "final_scores": {"ai": 51.0, "rank": 135.0, "topk_calibrated_risk": 24.5},
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
            "candidate_ledger": [
                {
                    "rank": 1,
                    "source": "global_best_candidate",
                    "section_id": "full_document",
                    "variant_id": "fallback",
                    "label": "fallback_full_document",
                    "word_count": 5,
                    "scores": {"ai": 51.0, "topk_calibrated_risk": 24.5},
                    "goal": {"status": "mitigation_failed_no_safe_candidate", "goal_met": False},
                    "text": "This is the fallback-selected document.",
                }
            ],
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
    assert result["status"] == "partial_candidate_not_strict_safe"
    assert summary["candidate_generation_status"]["accepted_count"] == 1
    assert summary["public_candidate_warning"] == "candidate_not_strict_safe"
    assert summary["best_candidate_external_review_required"] is False
    assert summary["kpi_finalization_status"] == "partial_candidate_not_strict_safe"
    assert summary["selected_candidate"]["section_id"] == "density_cluster_004"
    assert summary["candidate_ledger"][0]["text"] == "This is the fallback-selected document."
    assert summary["candidate_ledger"][0]["scores"]["topk_calibrated_risk"] == 24.5
    assert layer["global_best_fallback"]["applied"] is True
    assert layer["candidate_ledger"][0]["text"] == "This is the fallback-selected document."
    assert layer["phase_order"]["reason"] == "default_route_then_cleanup"


def test_v5_residual_comb_starts_from_historical_seed_when_it_beats_original(tmp_path, monkeypatch):
    original = (
        "The original draft explains the teaching problem in a broad way and repeats the same conclusion. "
        "It says the lesson was useful but does not tie the observation to a concrete classroom decision."
    )
    seed = (
        "The revised draft keeps the teaching problem focused on the observed classroom decision. "
        "It links the lesson to the student's response and explains the limit without repeating the conclusion."
    )

    def fake_scan(text):
        return {"input_text": text}

    def fake_goal(**kwargs):
        text = kwargs["candidate_text"]
        topk = 31.69 if text == seed else 80.277
        density = 42.61 if text == seed else 65.12
        return SimpleNamespace(to_dict=lambda: {
            "status": "mitigation_failed_no_safe_candidate",
            "goal_met": False,
            "ai_footprint_gate": {
                "safe_band": False,
                "remaining_ai_footprint_drivers": [
                    {"driver": "topk_calibrated_risk", "value": topk, "safe_band": 25.0},
                    {"driver": "qualifying_text_ai_density", "value": density, "safe_band": 35.0},
                ],
            },
        })

    def fake_score(_input_text, report, _goal):
        text = report["input_text"]
        if text == seed:
            return {
                "ai": 34.02,
                "topk": 74.88,
                "external": 32.577,
                "rank": 65.076,
                "risky_window_count": 0,
                "unsafe_word_ratio": 3.403,
                "unsafe_cluster_count": 2,
                "topk_calibrated_risk": 31.69,
                "qualifying_text_ai_density": 42.61,
                "ai_authorship": 34.0,
                "external_ai_flag_risk": 32.224,
            }
        return {
            "ai": 48.32,
            "topk": 92.77,
            "external": 45.634,
            "rank": 136.22,
            "risky_window_count": 1,
            "unsafe_word_ratio": 54.535,
            "unsafe_cluster_count": 12,
            "topk_calibrated_risk": 80.277,
            "qualifying_text_ai_density": 65.12,
            "ai_authorship": 48.0,
            "external_ai_flag_risk": 51.918,
        }

    monkeypatch.setattr(v5_residual_comb, "_scan_report", fake_scan)
    monkeypatch.setattr(v5_residual_comb, "evaluate_rewrite_goal", fake_goal)
    monkeypatch.setattr(v5_residual_comb, "_score_summary", fake_score)
    monkeypatch.setattr(v5_residual_comb, "_with_v5_density_gate", lambda _text, _report, goal: goal)
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_BORDERLINE_VERDICT_ENABLED", "0")
    monkeypatch.setenv("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_ENABLED", "0")
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_REPAIR_ENABLED", "0")

    payload = v5_residual_comb.run_v5_residual_cluster_comb_experiment(
        input_text=original,
        output_dir=tmp_path,
        max_rounds=0,
        variant_count=1,
        retune_variant_count=1,
        api_key="test-key",
        risky_window_cleanup_rounds=0,
        unsafe_cluster_cleanup_rounds=0,
        final_risky_window_cleanup_rounds=0,
        direct_scanner_leapfrog_rounds=0,
        seed_candidate_texts=[seed],
    )

    assert payload["rewritten_document"] == seed
    assert payload["seed_recovery"]["applied"] is True
    assert payload["seed_recovery"]["selected"]["variant_id"] == "seed_01"
    assert payload["accepted_checkpoints"][0]["phase"] == "historical_seed_recovery"
    assert payload["phase_order"]["reason"] == "historical_seed_targeted_safe_band_repair"
    assert payload["phase_order"]["seed_recovery_targeted_repair"] is True
    assert payload["phase_order"]["core_route_rounds"] == 0
    assert payload["rounds"][0]["status"] == "skipped"
    assert payload["rounds"][0]["reason"] == "historical_seed_targeted_safe_band_repair"
    assert payload["final_scores"]["topk_calibrated_risk"] == 31.69


def test_v5_residual_comb_routes_density_blocker_to_safe_band_before_final_topk(tmp_path, monkeypatch):
    original = "The original draft repeats a broad teaching point without grounded classroom movement."
    seed = "The revised draft keeps the classroom movement but still has a density gap."
    call_order: list[str] = []

    def fake_scan(text):
        return {"input_text": text}

    def fake_goal(**kwargs):
        text = kwargs["candidate_text"]
        density = 39.23 if text == seed else 65.12
        topk = 25.112 if text == seed else 80.277
        return SimpleNamespace(to_dict=lambda: {
            "status": "mitigation_failed_no_safe_candidate",
            "goal_met": False,
            "ai_footprint_gate": {
                "safe_band": False,
                "safe_band_thresholds": {"topk_calibrated_risk": 25.0, "qualifying_text_ai_density": 35.0},
                "remaining_ai_footprint_drivers": [
                    {"driver": "topk_calibrated_risk", "value": topk, "safe_band": 25.0},
                    {"driver": "qualifying_text_ai_density", "value": density, "safe_band": 35.0},
                ],
            },
        })

    def fake_score(_input_text, report, _goal):
        text = report["input_text"]
        if text == seed:
            return {
                "ai": 31.97,
                "topk": 67.73,
                "external": 29.525,
                "rank": 61.278,
                "risky_window_count": 0,
                "unsafe_word_ratio": 0.0,
                "unsafe_cluster_count": 0,
                "topk_calibrated_risk": 25.112,
                "qualifying_text_ai_density": 39.23,
                "ai_authorship": 32.0,
                "external_ai_flag_risk": 30.901,
            }
        return {
            "ai": 48.32,
            "topk": 92.77,
            "external": 45.634,
            "rank": 136.22,
            "risky_window_count": 1,
            "unsafe_word_ratio": 54.535,
            "unsafe_cluster_count": 12,
            "topk_calibrated_risk": 80.277,
            "qualifying_text_ai_density": 65.12,
            "ai_authorship": 48.0,
            "external_ai_flag_risk": 51.918,
        }

    def fake_safe_band_pack(**kwargs):
        call_order.append("safe_band")
        return (
            kwargs["current_text"],
            kwargs["current_report"],
            kwargs["current_goal"],
            kwargs["current_scores"],
            [{"phase": "safe_band_density_section_repair", "status": "skipped"}],
            kwargs["global_best_candidate"],
            False,
        )

    def fake_final_topk(**kwargs):
        call_order.append("final_topk")
        return (
            kwargs["current_text"],
            kwargs["current_report"],
            kwargs["current_goal"],
            kwargs["current_scores"],
            [{"phase": "final_topk_sentence_route", "status": "skipped"}],
            kwargs["global_best_candidate"],
        )

    monkeypatch.setattr(v5_residual_comb, "_scan_report", fake_scan)
    monkeypatch.setattr(v5_residual_comb, "evaluate_rewrite_goal", fake_goal)
    monkeypatch.setattr(v5_residual_comb, "_score_summary", fake_score)
    monkeypatch.setattr(v5_residual_comb, "_with_v5_density_gate", lambda _text, _report, goal: goal)
    monkeypatch.setattr(v5_residual_comb, "_run_safe_band_evidence_pack_attempt", fake_safe_band_pack)
    monkeypatch.setattr(v5_residual_comb, "_run_final_topk_sentence_route_pass", fake_final_topk)
    monkeypatch.setattr(v5_residual_comb, "_final_topk_sentence_route_should_run", lambda **_kwargs: True)
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_BORDERLINE_VERDICT_ENABLED", "0")
    monkeypatch.setenv("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_REPAIR_ENABLED", "1")
    monkeypatch.setenv("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_ENABLED", "1")

    payload = v5_residual_comb.run_v5_residual_cluster_comb_experiment(
        input_text=original,
        output_dir=tmp_path,
        max_rounds=0,
        variant_count=1,
        retune_variant_count=1,
        api_key="test-key",
        risky_window_cleanup_rounds=0,
        unsafe_cluster_cleanup_rounds=0,
        final_risky_window_cleanup_rounds=0,
        direct_scanner_leapfrog_rounds=0,
        seed_candidate_texts=[seed],
    )

    assert call_order[:2] == ["safe_band", "final_topk"]
    assert payload["phase_order"]["safe_band_evidence_repair"]["pre_core_author_proxy_pack"] is True
    assert payload["safe_band_evidence_repair_rounds"][0]["phase"] == "safe_band_density_section_repair"


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
