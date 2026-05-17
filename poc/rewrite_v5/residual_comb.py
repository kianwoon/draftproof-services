"""Residual cluster comb-through experiment for V5.

This path turns the single-cluster proof into an iterative experiment:
scan the current document, rebuild the strongest remaining cluster, score the
cluster and full document, accept useful movement, then repeat from the new
scan. It is intentionally isolated from production.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from llm.gateway import LLMConfig, LLMGateway
from rewrite_v2.goal_contract import evaluate_rewrite_goal
from rewrite_v2.structured_output import structured_json_request_options
from rewrite_v3.document_units import word_count
from rewrite_v3.pipeline import _scan_report
from rewrite_v3.text_integrity import minimal_replacement_text_integrity
from rewrite_controller.eligible_span_density import build_eligible_span_density_contract
from rewrite_v4.cluster_patch import build_cluster_repair_units
from rewrite_v4.validation import parse_json_object

from .experiment import (
    _add_deltas,
    _merge_provider_options,
    _score_summary,
    _section_from_cluster,
    _sentences,
    _variants_response_format,
    apply_section_variant,
)
from .models import RecompositionVariant, SectionUnit


_ROUTE_PLAN_CONTENT_PROFILES = {
    "reflective_practice_academic": {
        "use_when": "The cluster is built around a course, classroom, workplace, student, client, practice event, or author reflection.",
        "planning_focus": "Turn broad reflection into event or process movement: context -> difficulty -> action/choice -> observed result -> limited judgment.",
        "avoid": "Do not convert the writer into a detached encyclopedia voice.",
    },
    "broad_explanatory_report": {
        "use_when": "The cluster explains a country, institution, topic, history, culture, economy, technology, or other broad subject for a report-style essay.",
        "planning_focus": "Replace category dumping with grouped topic progression: topic frame -> grouped facts -> contrast or limit -> bridge to next topic.",
        "avoid": "Do not add personal experience, force reflective teacher/student language, or upgrade simple report wording into encyclopedia-style phrasing.",
    },
    "argumentative_explanatory_essay": {
        "use_when": "The cluster makes a claim, gives reasons, weighs concerns, or argues for a position.",
        "planning_focus": "Make the reasoning route explicit: claim -> reason -> evidence already present -> qualification -> narrower conclusion.",
        "avoid": "Do not add new evidence or stronger certainty than the source supports.",
    },
    "technical_or_process_explanation": {
        "use_when": "The cluster explains a method, mechanism, procedure, tool, system, or step-by-step process.",
        "planning_focus": "Make the process route concrete: condition -> step or mechanism -> constraint -> consequence -> next step.",
        "avoid": "Do not turn process wording into abstract commentary.",
    },
    "narrative_or_case_reflection": {
        "use_when": "The cluster follows a person, event, case, incident, sequence, or change over time.",
        "planning_focus": "Make the route follow the case: starting state -> turning point -> response -> outcome -> reflection.",
        "avoid": "Do not replace the case with a generic moral summary.",
    },
    "mixed_or_unknown": {
        "use_when": "The cluster mixes several modes or none of the specific profiles clearly fits.",
        "planning_focus": "Choose the dominant paragraph job and rebuild only that route while preserving all source claims.",
        "avoid": "Do not apply a specialist rubric that conflicts with the source.",
    },
}


_ROUTE_PLAN_CLUSTER_ROLES = {
    "background_context": "Introduces or frames a topic before later detail.",
    "evidence_or_example": "Provides facts, examples, cases, citations, or observed material.",
    "reasoning_or_analysis": "Connects claims, causes, consequences, or judgments.",
    "process_or_method": "Explains how something works or how an action is performed.",
    "contrast_or_problem": "Names a limitation, tension, challenge, or counterpoint.",
    "conclusion_or_synthesis": "Pulls prior points together or states the final implication.",
    "mixed_section": "Contains multiple paragraph jobs inside one cluster.",
}


_ROUTE_PLAN_FAILURE_PATTERNS = {
    "category_dump": "The cluster stacks topic categories or facts without enough route hierarchy.",
    "event_summary": "The cluster summarizes a case or event before showing the event movement.",
    "claim_chain": "The cluster links claims but leaves reasoning, evidence, or qualification underdeveloped.",
    "process_blur": "The cluster explains a method or process without clear condition, step, constraint, and consequence.",
    "transition_stack": "The cluster relies on repeated transition openers instead of real relation changes.",
    "conclusion_smoothing": "The cluster over-smooths the ending into a generic broad conclusion.",
    "mixed": "The cluster contains more than one failure pattern.",
}


_ROUTE_PLAN_STRATEGIES = {
    "group_and_bridge": "Group related source beats and add source-supported bridges between topic groups.",
    "event_first_rebuild": "Move the event, case, or action before broad interpretation.",
    "claim_reason_evidence": "Separate claim, reason, source evidence, and qualification.",
    "mechanism_consequence": "Make the process route visible through mechanism, constraint, and consequence.",
    "contrast_then_limit": "Use contrast to narrow a broad claim and limit the conclusion.",
    "mixed_route_rebuild": "Use more than one route move while preserving every source block.",
}


def run_v5_residual_cluster_comb_experiment(
    *,
    input_text: str,
    output_dir: str | Path,
    max_rounds: int = 5,
    variant_count: int = 3,
    retune_variant_count: int = 4,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_body: dict[str, Any] | None = None,
    risky_window_cleanup_rounds: int | None = None,
    unsafe_cluster_cleanup_rounds: int | None = None,
    cleanup_variant_count: int | None = None,
    final_risky_window_cleanup_rounds: int | None = None,
) -> dict[str, Any]:
    """Iteratively treat the strongest residual cluster and rescan."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_text = str(input_text or "")
    baseline_report = _scan_report(original_text)
    baseline_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=original_text,
        original_report=baseline_report,
        candidate_report=baseline_report,
    ).to_dict()
    baseline_scores = _score_summary(original_text, baseline_report, baseline_goal)
    current_text = original_text
    current_report = baseline_report
    current_goal = baseline_goal
    current_scores = baseline_scores
    global_best_candidate: dict[str, Any] | None = None
    gateway = LLMGateway(LLMConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_MAX_TOKENS", 8000, minimum=1000, maximum=12000),
        temperature=_float_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_TEMPERATURE", 0.35, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_TOP_P", 0.9, minimum=0.1, maximum=1.0),
        timeout=180,
        extra_body=extra_body,
    ))

    rounds: list[dict[str, Any]] = []
    for round_index in range(1, max(1, int(max_rounds or 1)) + 1):
        round_dir = out_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        cluster_units = build_cluster_repair_units(
            text=current_text,
            report=current_report,
            goal=current_goal,
            limit=1,
            context_chars=220,
        )
        if not cluster_units:
            rounds.append({"round": round_index, "status": "stopped", "reason": "no_residual_cluster"})
            break
        section = _section_from_cluster(cluster_units[0])
        local_source_goal = _local_goal(section.text, section.text)
        seed_variants = generate_residual_cluster_seed_variants(section=section, local_goal=local_source_goal)
        seed_rows = [
            _score_residual_variant(
                original_text=original_text,
                baseline_report=baseline_report,
                baseline_scores=baseline_scores,
                current_text=current_text,
                current_scores=current_scores,
                section=section,
                variant=variant,
                output_dir=round_dir,
                label=f"seed_{variant.variant_id}",
            )
            for variant in seed_variants
        ]
        best_seed = _best_residual_candidate(seed_rows)
        seed_accepted = best_seed if best_seed and _has_incremental_movement(best_seed) else None
        diagnostics: dict[str, Any] = {
            "seed_variant_count": len(seed_variants),
            "seed_short_circuited": bool(seed_accepted),
        }
        retune_diagnostics: dict[str, Any] | None = None
        rows: list[dict[str, Any]] = list(seed_rows)
        retuned_rows: list[dict[str, Any]] = []
        route_plan: dict[str, Any] | None = None
        route_plan_diagnostics: dict[str, Any] | None = None
        if not seed_accepted:
            route_plan, route_plan_diagnostics, plan_prompt, plan_completion = generate_residual_cluster_route_plan(
                section=section,
                local_goal=local_source_goal,
                gateway=gateway,
            )
            (round_dir / "route_plan_prompt.json.txt").write_text(plan_prompt)
            (round_dir / "route_plan_completion.json.txt").write_text(plan_completion)
            variants, llm_diagnostics, prompt, completion = generate_residual_cluster_variants(
                section=section,
                local_goal=local_source_goal,
                gateway=gateway,
                variant_count=variant_count,
                route_plan=route_plan,
            )
            diagnostics = {
                **diagnostics,
                "route_plan": route_plan_diagnostics,
                "llm_generation": llm_diagnostics,
            }
            (round_dir / "cluster_prompt.json.txt").write_text(prompt)
            (round_dir / "cluster_completion.json.txt").write_text(completion)
            rows.extend([
                _score_residual_variant(
                    original_text=original_text,
                    baseline_report=baseline_report,
                    baseline_scores=baseline_scores,
                    current_text=current_text,
                    current_scores=current_scores,
                    section=section,
                    variant=variant,
                    output_dir=round_dir,
                    label=f"initial_{variant.variant_id}",
                )
                for variant in variants
            ])
        best_initial = _best_residual_candidate(rows)
        if not seed_accepted and best_initial and _needs_retune(best_initial):
            retuned, retune_diagnostics, retune_prompt, retune_completion = generate_residual_cluster_retunes(
                section=section,
                current_best_text=str(best_initial.get("text") or ""),
                local_goal=best_initial.get("local_goal") or {},
                gateway=gateway,
                variant_count=retune_variant_count,
                route_plan=route_plan,
            )
            (round_dir / "retune_prompt.json.txt").write_text(retune_prompt)
            (round_dir / "retune_completion.json.txt").write_text(retune_completion)
            retuned_rows = [
                _score_residual_variant(
                    original_text=original_text,
                    baseline_report=baseline_report,
                    baseline_scores=baseline_scores,
                    current_text=current_text,
                    current_scores=current_scores,
                    section=section,
                    variant=variant,
                    output_dir=round_dir,
                    label=f"retune_{variant.variant_id}",
                )
                for variant in retuned
            ]
        all_rows = rows + retuned_rows
        global_best_candidate = _best_full_document_candidate([global_best_candidate, *all_rows])
        best = _best_residual_candidate(all_rows)
        accepted = best if best and _has_incremental_movement(best) else None
        round_payload = {
            "round": round_index,
            "status": "accepted" if accepted else "stopped",
            "reason": "accepted_incremental_movement" if accepted else "no_incremental_movement",
            "section": section.to_dict(),
            "generator_diagnostics": diagnostics,
            "retune_diagnostics": retune_diagnostics,
            "current_scores": current_scores,
            "candidates": [_compact_residual_row(row) for row in all_rows],
            "selected": _compact_residual_row(best),
            "accepted": _compact_residual_row(accepted),
        }
        rounds.append(round_payload)
        (round_dir / "round_result.json").write_text(json.dumps(round_payload, ensure_ascii=False, indent=2))
        if not accepted:
            break
        current_text = str(accepted.get("candidate_text") or current_text)
        current_report = accepted.get("candidate_report") if isinstance(accepted.get("candidate_report"), dict) else _scan_report(current_text)
        current_goal = accepted.get("candidate_goal") if isinstance(accepted.get("candidate_goal"), dict) else evaluate_rewrite_goal(
            original_text=original_text,
            candidate_text=current_text,
            original_report=baseline_report,
            candidate_report=current_report,
        ).to_dict()
        current_scores = accepted.get("scores") if isinstance(accepted.get("scores"), dict) else _score_summary(original_text, current_report, current_goal)
        (out_dir / f"after_round_{round_index:02d}.txt").write_text(current_text)

    cleanup_variants = cleanup_variant_count if cleanup_variant_count is not None else variant_count
    risky_window_rounds: list[dict[str, Any]] = []
    if _cleanup_round_limit(
        risky_window_cleanup_rounds,
        env_name="DRAFTPROOF_REWRITE_V5_RISKY_WINDOW_CLEANUP_ROUNDS",
        default=2,
    ) > 0:
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            risky_window_rounds,
            global_best_candidate,
        ) = _run_risky_window_cleanup_pass(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            gateway=gateway,
            output_dir=out_dir / "risky_window_cleanup",
            global_best_candidate=global_best_candidate,
            max_rounds=_cleanup_round_limit(
                risky_window_cleanup_rounds,
                env_name="DRAFTPROOF_REWRITE_V5_RISKY_WINDOW_CLEANUP_ROUNDS",
                default=2,
            ),
            variant_count=cleanup_variants,
        )

    unsafe_cluster_rounds: list[dict[str, Any]] = []
    if _cleanup_round_limit(
        unsafe_cluster_cleanup_rounds,
        env_name="DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_CLEANUP_ROUNDS",
        default=12,
    ) > 0:
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            unsafe_cluster_rounds,
            global_best_candidate,
        ) = _run_unsafe_cluster_cleanup_pass(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            gateway=gateway,
            output_dir=out_dir / "unsafe_cluster_cleanup",
            global_best_candidate=global_best_candidate,
            max_rounds=_cleanup_round_limit(
                unsafe_cluster_cleanup_rounds,
                env_name="DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_CLEANUP_ROUNDS",
                default=12,
            ),
            variant_count=cleanup_variants,
        )

    final_risky_window_rounds: list[dict[str, Any]] = []
    if _cleanup_round_limit(
        final_risky_window_cleanup_rounds,
        env_name="DRAFTPROOF_REWRITE_V5_FINAL_RISKY_WINDOW_CLEANUP_ROUNDS",
        default=2,
    ) > 0:
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            final_risky_window_rounds,
            global_best_candidate,
        ) = _run_risky_window_cleanup_pass(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            gateway=gateway,
            output_dir=out_dir / "final_risky_window_cleanup",
            global_best_candidate=global_best_candidate,
            max_rounds=_cleanup_round_limit(
                final_risky_window_cleanup_rounds,
                env_name="DRAFTPROOF_REWRITE_V5_FINAL_RISKY_WINDOW_CLEANUP_ROUNDS",
                default=2,
            ),
            variant_count=cleanup_variants,
        )

    global_best_fallback = {
        "applied": False,
        "reason": "phase_accepted_result_remained_best",
        "selected": _compact_residual_row(global_best_candidate),
        "previous_final_scores": current_scores,
    }
    if global_best_candidate and _full_document_candidate_beats_scores(global_best_candidate, current_scores):
        previous_scores = current_scores
        current_text, current_report, current_goal, current_scores = _accepted_state(
            accepted=global_best_candidate,
            original_text=original_text,
            baseline_report=baseline_report,
        )
        (out_dir / "after_global_best_fallback.txt").write_text(current_text)
        global_best_fallback = {
            "applied": True,
            "reason": "best_full_document_candidate_superseded_phase_accepted_result",
            "selected": _compact_residual_row(global_best_candidate),
            "previous_final_scores": previous_scores,
            "final_scores": current_scores,
        }

    density_gate = build_eligible_span_density_contract(current_text, current_report)
    payload = {
        "stage": "v5_residual_cluster_comb",
        "baseline_scores": baseline_scores,
        "rounds": rounds,
        "risky_window_cleanup_rounds": risky_window_rounds,
        "unsafe_cluster_cleanup_rounds": unsafe_cluster_rounds,
        "final_risky_window_cleanup_rounds": final_risky_window_rounds,
        "global_best_fallback": global_best_fallback,
        "final_scores": current_scores,
        "eligible_span_density_gate": density_gate,
        "goal": {
            "status": current_goal.get("status"),
            "goal_met": current_goal.get("goal_met"),
            "reason": current_goal.get("reason"),
        },
        "rewritten_document": current_text,
    }
    (out_dir / "v5_residual_comb_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (out_dir / "v5_residual_comb_rewritten_document.txt").write_text(current_text)
    return payload


def _cleanup_round_limit(value: int | None, *, env_name: str, default: int) -> int:
    if value is not None:
        return max(0, min(12, int(value or 0)))
    return _int_env(env_name, default, minimum=0, maximum=12)


def build_residual_cluster_prompt(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any] | None = None,
    variant_count: int = 3,
    route_plan: dict[str, Any] | None = None,
) -> str:
    variants = max(1, min(5, int(variant_count or 1)))
    plan = route_plan if _route_plan_valid(route_plan) else None
    payload = {
        "task": "residual_cluster_route_bump",
        "cluster": {
            "section_id": section.section_id,
            "source_text": section.text,
            "before_context": _section_before_context(section),
            "after_context": _section_after_context(section),
            "source_word_count": section.word_count,
            "source_block_count": section.paragraph_count,
            "source_blocks": _source_blocks(section.text),
            "source_event_beats": _source_event_beats(section.text),
            "source_phrase_anchors": _source_phrase_anchors(section.text),
            "referential_continuity": _referential_continuity(
                section.text,
                before_context=_section_before_context(section),
            ),
        },
        "length_guidance": _length_guidance_for_route_plan(section=section, route_plan=plan),
        "coverage_guidance": _coverage_guidance_for_route_plan(section=section, route_plan=plan),
        "remaining_problem_sentences": _local_unsafe_previews(local_goal or {}),
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, variants + 1)
            ]
        },
    }
    if plan:
        payload["execution_brief"] = plan
        payload["method"] = _custom_route_writer_method()
    else:
        payload["custom_route_plan"] = None
        payload["fallback_route_blueprint"] = build_route_blueprint(section=section, local_goal=local_goal or {})
        payload["method"] = _fallback_route_writer_method()
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_residual_cluster_retune_prompt(
    *,
    section: SectionUnit,
    current_best_text: str,
    local_goal: dict[str, Any],
    variant_count: int = 4,
    route_plan: dict[str, Any] | None = None,
) -> str:
    variants = max(1, min(5, int(variant_count or 1)))
    focus = _local_unsafe_previews(local_goal)
    plan = route_plan if _route_plan_valid(route_plan) else None
    payload = {
        "task": "residual_cluster_retune",
        "cluster": {
            "section_id": section.section_id,
            "original_source_text": section.text,
            "before_context": _section_before_context(section),
            "after_context": _section_after_context(section),
            "source_block_count": section.paragraph_count,
            "source_blocks": _source_blocks(section.text),
            "source_event_beats": _source_event_beats(section.text),
            "source_phrase_anchors": _source_phrase_anchors(section.text),
            "referential_continuity": _referential_continuity(
                section.text,
                before_context=_section_before_context(section),
            ),
            "current_best_text": current_best_text,
        },
        "length_guidance": _length_guidance_for_route_plan(section=section, route_plan=plan),
        "coverage_guidance": _coverage_guidance_for_route_plan(section=section, route_plan=plan),
        "remaining_problem_sentences": focus,
        "retune_focus": _retune_focus_from_goal(local_goal or {}),
        "candidate_non_source_terms_to_reduce": _non_source_terms(section.text, current_best_text),
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, variants + 1)
            ]
        },
    }
    if plan:
        payload["execution_brief"] = plan
        payload["method"] = _custom_route_retune_method()
    else:
        payload["custom_route_plan"] = None
        payload["fallback_route_blueprint"] = build_route_blueprint(section=section, local_goal=local_goal or {})
        payload["method"] = _fallback_route_retune_method()
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_residual_cluster_variants(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any] | None = None,
    gateway: LLMGateway,
    variant_count: int = 3,
    route_plan: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    prompt = build_residual_cluster_prompt(
        section=section,
        local_goal=local_goal or {},
        variant_count=variant_count,
        route_plan=route_plan,
    )
    return _generate_loose_variants(prompt=prompt, gateway=gateway, variant_count=variant_count)


def build_residual_cluster_route_plan_prompt(*, section: SectionUnit, local_goal: dict[str, Any] | None = None) -> str:
    payload = {
        "task": "profile_aware_cluster_route_plan",
        "cluster": {
            "section_id": section.section_id,
            "source_text": section.text,
            "before_context": _section_before_context(section),
            "after_context": _section_after_context(section),
            "source_word_count": section.word_count,
            "source_block_count": section.paragraph_count,
            "source_blocks": _source_blocks(section.text),
            "source_event_beats": _source_event_beats(section.text),
            "source_phrase_anchors": _source_phrase_anchors(section.text),
            "referential_continuity": _referential_continuity(
                section.text,
                before_context=_section_before_context(section),
            ),
        },
        "scanner_local_findings": {
            "unsafe_previews": _local_unsafe_previews(local_goal or {}),
            "top_sentence_targets": _local_top_sentence_targets(local_goal or {}),
            "recommended_actions": _local_recommended_actions(local_goal or {}),
        },
        "content_profile_rubrics": _ROUTE_PLAN_CONTENT_PROFILES,
        "cluster_role_options": _ROUTE_PLAN_CLUSTER_ROLES,
        "failure_pattern_options": _ROUTE_PLAN_FAILURE_PATTERNS,
        "route_strategy_options": _ROUTE_PLAN_STRATEGIES,
        "planning_rules": [
            "Act as a prompt planner, not the final writer.",
            "Derive a cluster-specific executable brief from the source text and scanner findings.",
            "First choose content_profile and cluster_role from the supplied options.",
            "Then choose dominant_failure_pattern and route_strategy from the supplied options.",
            "Use the chosen content_profile rubric to design the route; do not use a reflective-practice route for broad report content.",
            "Use the chosen cluster_role to decide what the cluster is supposed to do in the document.",
            "When cluster.source_block_count is greater than 1, the replacement route must cover every source block instead of compressing the cluster into only the opening topic.",
            "Make source_block_plan cover every cluster.source_blocks item.",
            "Make target_sentence_jobs focus on scanner_local_findings.top_sentence_targets and give one executable rewrite job per target.",
            "Do not mention scores, scanner names, authorship labels, or risk labels in the plan fields.",
            "Describe failed_route as the current sentence movement problem in plain editorial language.",
            "Describe replacement_route as the new route the writer should follow, using source-supported events and claims.",
            "Make must_change concrete enough that the writer can execute it without seeing fallback rules.",
            "Make must_preserve exact source anchors: each source_quote must be copied verbatim from cluster.source_text.",
            "Do not put summaries in must_preserve.source_quote; never write phrases like 'the fact that' unless those words are in cluster.source_text.",
            "Use must_preserve.preserve_as only as a short meaning label for the exact source quote.",
            "If a preservation item cannot be copied exactly from cluster.source_text, omit it.",
            "Make sentence_plan an ordered set of sentence jobs, not labels to copy into the answer.",
            "Use avoid_phrases only for source phrases or polished substitutes that would keep the same weak pattern.",
            "Choose length_target from same_length, slight_expand, or expand.",
            "Use same_length when route can change without added bridging, slight_expand when one bridge is needed, and expand only when compression is the main weakness.",
            "Explain reason_this_should_move_score as a plain cause-effect expectation about route movement, not a score promise.",
            "Do not add facts, examples, people, places, dates, dialogue, or outside knowledge.",
            "Preserve cluster.referential_continuity in the replacement route.",
            "If a pronoun is linked to a name in before_context, plan for natural name/pronoun continuity and do not tell the writer to explain the reference parenthetically.",
            "Use plain editorial task language that a writer can execute.",
        ],
        "output_schema": {
            "route_plan": {
                "content_profile": "reflective_practice_academic | broad_explanatory_report | argumentative_explanatory_essay | technical_or_process_explanation | narrative_or_case_reflection | mixed_or_unknown",
                "cluster_role": "background_context | evidence_or_example | reasoning_or_analysis | process_or_method | contrast_or_problem | conclusion_or_synthesis | mixed_section",
                "dominant_failure_pattern": "category_dump | event_summary | claim_chain | process_blur | transition_stack | conclusion_smoothing | mixed",
                "route_strategy": "group_and_bridge | event_first_rebuild | claim_reason_evidence | mechanism_consequence | contrast_then_limit | mixed_route_rebuild",
                "profile_reason": "one sentence explaining why this profile fits the cluster",
                "failed_route": "...",
                "replacement_route": "...",
                "source_block_plan": [
                    {
                        "block_id": "b01",
                        "current_job": "what this source block does now",
                        "rewrite_job": "what the writer must make this block do",
                        "must_preserve": ["exact or source-near material from this block"]
                    }
                ],
                "target_sentence_jobs": [
                    {
                        "sentence_id": "s001",
                        "source_preview": "exact preview from scanner_local_findings.top_sentence_targets or source text",
                        "current_weakness": "why this sentence route is weak",
                        "rewrite_job": "specific instruction for this sentence's role",
                        "avoid_copying": ["phrase to avoid keeping"]
                    }
                ],
                "must_change": ["..."],
                "must_preserve": [
                    {
                        "source_quote": "exact substring copied from cluster.source_text",
                        "preserve_as": "short meaning label",
                    }
                ],
                "sentence_plan": ["..."],
                "avoid_phrases": ["..."],
                "length_target": "same_length | slight_expand | expand",
                "reason_this_should_move_score": "...",
            }
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_residual_cluster_route_plan(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any],
    gateway: LLMGateway,
) -> tuple[dict[str, Any] | None, dict[str, Any], str, str]:
    prompt = build_residual_cluster_route_plan_prompt(section=section, local_goal=local_goal)
    structured = structured_json_request_options(getattr(gateway, "model", None), _route_plan_response_format())
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured.get("provider"))
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a route_plan object.",
        response_format=structured.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_REWRITE_V5_ROUTE_PLAN_TEMPERATURE", 0.12, minimum=0.0, maximum=0.8),
        top_p=_float_env("DRAFTPROOF_REWRITE_V5_ROUTE_PLAN_TOP_P", 0.72, minimum=0.1, maximum=1.0),
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V5_ROUTE_PLAN_MAX_TOKENS", 2600, minimum=800, maximum=6000),
    )
    raw = response.raw_content or response.content
    parsed, diagnostics = _parse_route_plan(raw, source_text=section.text)
    return parsed, {
        **diagnostics,
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "structured_output_mode": structured.get("structured_output_mode"),
    }, prompt, raw


def _length_guidance_for_route_plan(*, section: SectionUnit, route_plan: dict[str, Any] | None) -> dict[str, Any]:
    source_words = max(1, int(section.word_count or word_count(section.text)))
    if not route_plan:
        return {
            "source_words": source_words,
            "preferred_min_words": max(source_words + 10, round(source_words * 1.12)),
            "preferred_max_words": round(source_words * 1.35),
            "purpose": "use enough words to rebuild the route; do not compress the cluster",
        }
    target = _length_target(route_plan.get("length_target"))
    if target == "same_length":
        return {
            "source_words": source_words,
            "preferred_min_words": round(source_words * 0.90),
            "preferred_max_words": round(source_words * 1.10),
            "purpose": "execute the route change at roughly source length",
        }
    if target == "slight_expand":
        return {
            "source_words": source_words,
            "preferred_min_words": max(source_words, round(source_words * 1.00)),
            "preferred_max_words": round(source_words * 1.20),
            "purpose": "add only the bridge needed by the executable route",
        }
    return {
        "source_words": source_words,
        "preferred_min_words": max(source_words + 10, round(source_words * 1.12)),
        "preferred_max_words": round(source_words * 1.35),
        "purpose": "use enough words to rebuild the route; do not compress the cluster",
    }


def _coverage_guidance_for_route_plan(*, section: SectionUnit, route_plan: dict[str, Any] | None) -> dict[str, Any]:
    profile = _content_profile((route_plan or {}).get("content_profile"))
    role = _cluster_role((route_plan or {}).get("cluster_role"))
    beats = _source_event_beats(section.text)
    source_blocks = _source_blocks(section.text)
    requirements = [
        "Represent every source block and every central source beat in the replacement.",
        "Do not drop later source material just because the opening topic has been repaired.",
    ]
    if profile == "broad_explanatory_report" or role == "mixed_section":
        requirements.extend([
            "Keep the report-style breadth: topic groups may be reorganized, but economics, public issues, institutions, or international-role material already present must not disappear.",
            "Use grouping and bridges instead of compressing several source blocks into one summary paragraph.",
        ])
    elif profile == "reflective_practice_academic":
        requirements.append("Keep the practice context, writer action or decision, observed response, and reflection if they appear in the source.")
    elif profile == "technical_or_process_explanation":
        requirements.append("Keep the condition, process step, constraint, and consequence if they appear in the source.")
    return {
        "source_block_count": max(1, int(section.paragraph_count or len(source_blocks) or 1)),
        "source_sentence_count_visible_to_planner": len(beats),
        "preserve_source_block_count_when_possible": int(section.paragraph_count or 1) > 1,
        "requirements": requirements,
    }


def _custom_route_writer_method() -> list[str]:
    return [
        "Use execution_brief.content_profile and execution_brief.cluster_role to choose the right kind of route movement.",
        "Use execution_brief.dominant_failure_pattern and execution_brief.route_strategy to decide what must actually change.",
        "Execute execution_brief.source_block_plan block by block; each block must keep its central source material.",
        "Execute execution_brief.target_sentence_jobs for the risky sentences; do not leave those sentence routes unchanged.",
        "Satisfy coverage_guidance before style changes; do not omit source blocks or central source beats.",
        "When execution_brief.content_profile is broad_explanatory_report, keep source-level vocabulary and change grouping/bridges; do not upgrade factual wording into encyclopedia-style substitutes.",
        "Follow execution_brief.replacement_route while rewriting the whole cluster.",
        "Satisfy every execution_brief.must_change item.",
        "Preserve every execution_brief.must_preserve.source_quote; use preserve_as only as the meaning hint.",
        "Follow execution_brief.sentence_plan in order, but do not copy plan labels into the replacement.",
        "Avoid execution_brief.avoid_phrases unless the phrase is a required source term.",
        "Use length_guidance; do not compress the cluster into a summary.",
        "Change remaining_problem_sentences most strongly.",
        "Keep the same people, event, activity, outcome, point of view, and referential continuity.",
        "Use simple source-near wording where it works.",
        "Do not invent new names, dates, places, dialogue, objects, clients, classmates, or side events.",
        "Do not replace the source scenario with a different scene.",
        "Do not return a fragment; the replacement must cover the whole source cluster.",
        "Do not write a plan, label, explanation of the method, or bullet list.",
    ]


def _custom_route_retune_method() -> list[str]:
    return [
        "Use execution_brief.content_profile and execution_brief.cluster_role to choose the right kind of route movement.",
        "Use execution_brief.dominant_failure_pattern and execution_brief.route_strategy to decide what must actually change.",
        "Execute execution_brief.source_block_plan block by block; each block must keep its central source material.",
        "Execute execution_brief.target_sentence_jobs for the risky sentences; do not leave those sentence routes unchanged.",
        "Satisfy coverage_guidance before style changes; do not omit source blocks or central source beats.",
        "When execution_brief.content_profile is broad_explanatory_report, keep source-level vocabulary and change grouping/bridges; do not upgrade factual wording into encyclopedia-style substitutes.",
        "Follow execution_brief.replacement_route while rewriting the whole cluster again.",
        "Satisfy every execution_brief.must_change item while focusing on remaining_problem_sentences.",
        "Preserve every execution_brief.must_preserve.source_quote; use preserve_as only as the meaning hint.",
        "Follow execution_brief.sentence_plan in order, but do not copy plan labels into the replacement.",
        "Avoid execution_brief.avoid_phrases unless the phrase is a required source term.",
        "Use retune_focus and candidate_non_source_terms_to_reduce only to clean the current best wording.",
        "Use length_guidance; do not compress the cluster into a summary.",
        "Keep the same people, event, activity, outcome, point of view, and referential continuity.",
        "Do not invent new names, dates, places, dialogue, objects, clients, classmates, or side events.",
        "Do not replace the source scenario with a different scene.",
        "Do not return a fragment; the replacement must cover the whole source cluster.",
        "Do not write a plan, label, explanation of the method, or bullet list.",
    ]


def _fallback_route_writer_method() -> list[str]:
    return [
        "Rewrite this cluster as a concrete route window, not as phrase repair.",
        "Follow fallback_route_blueprint.steps in order.",
        "Use fallback_route_blueprint.sentence_jobs.",
        "Each sentence job should carry a full source beat or bridge, not a compressed summary.",
        "Follow length_guidance as writing guidance; fuller source-grounded wording is preferred over compression.",
        "Do not copy fallback_route_blueprint labels or step names into the replacement.",
        "Change the route of remaining_problem_sentences most strongly.",
        "Do not keep an avoid_openers item as the opening sentence.",
        "Preserve each source-supported beat unless two adjacent beats are naturally merged.",
        "Stay source-near: keep simple source words when they already work.",
        "Use cluster.source_phrase_anchors where they fit naturally.",
        "Do not upgrade simple source wording into formal education-theory labels.",
        "Keep the same people, event, activity, outcome, and point of view.",
        "Preserve cluster.referential_continuity; do not replace a specific source subject with a generic category label.",
        "If cluster.referential_continuity gives an established name, use the name or the source pronoun naturally; do not write explanatory referent phrases like 'referring to'.",
        "If the source uses I or my, keep that teacher viewpoint instead of replacing it with a detached narrator.",
        "Do not invent new names, dates, places, dialogue, weather, objects, clients, classmates, or side events.",
        "Do not replace the source scenario with a different scene.",
        "Do not return a fragment; the replacement must cover the whole source cluster.",
        "Do not write a plan, label, explanation of the method, or bullet list.",
        "Avoid abstract summary language. Make the movement happen through the event.",
    ]


def _fallback_route_retune_method() -> list[str]:
    return [
        "Rewrite the whole cluster again, but focus on the remaining problem sentence route.",
        "Follow fallback_route_blueprint.steps in order.",
        "Use fallback_route_blueprint.sentence_jobs.",
        "Each sentence job should carry a full source beat or bridge, not a compressed summary.",
        "Follow length_guidance as writing guidance; fuller source-grounded wording is preferred over compression.",
        "Do not copy fallback_route_blueprint labels or step names into the replacement.",
        "Break any packed sentence into clearer event movement if needed.",
        "Preserve each source_event_beats item unless two adjacent beats are naturally merged.",
        "Stay source-near: keep simple source words when they already work.",
        "Use cluster.source_phrase_anchors where they fit naturally.",
        "Reduce candidate_non_source_terms_to_reduce by replacing them with source wording where possible.",
        "Do not upgrade simple source wording into formal education-theory labels.",
        "If the source uses I or my, keep that teacher viewpoint instead of replacing it with a detached narrator.",
        "Preserve cluster.referential_continuity; do not replace a specific source subject with a generic category label.",
        "If cluster.referential_continuity gives an established name, use the name or the source pronoun naturally; do not write explanatory referent phrases like 'referring to'.",
        "Use concrete classroom, action, or event wording from the same source scenario instead of summary wording.",
        "Do not invent new names, dates, places, dialogue, weather, objects, clients, classmates, or side events.",
        "Do not replace the source scenario with a different scene.",
        "Do not return a fragment; the replacement must cover the whole source cluster.",
        "Do not write a plan, label, explanation of the method, or bullet list.",
    ]


def generate_residual_cluster_seed_variants(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any] | None = None,
) -> list[RecompositionVariant]:
    """Generate scanner-scored route seeds before asking the model.

    These are not accepted directly. They enter the same local and full-document
    scoring path as LLM candidates, so a bad seed is only a measured miss.
    """

    del local_goal
    texts = _student_support_role_progression_seed_texts(section.text)
    variants: list[RecompositionVariant] = []
    seen: set[str] = set()
    for index, text in enumerate(texts, start=1):
        normalized = " ".join(str(text or "").split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        variants.append(RecompositionVariant(
            variant_id=f"route_seed_{index}",
            text=normalized,
            word_count=word_count(normalized),
        ))
    return variants


def generate_residual_cluster_retunes(
    *,
    section: SectionUnit,
    current_best_text: str,
    local_goal: dict[str, Any],
    gateway: LLMGateway,
    variant_count: int = 4,
    route_plan: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    prompt = build_residual_cluster_retune_prompt(
        section=section,
        current_best_text=current_best_text,
        local_goal=local_goal,
        variant_count=variant_count,
        route_plan=route_plan,
    )
    return _generate_loose_variants(prompt=prompt, gateway=gateway, variant_count=variant_count)


def build_risky_window_cleanup_prompt(
    *,
    section: SectionUnit,
    current_scores: dict[str, Any] | None = None,
    variant_count: int = 5,
    route_plan: dict[str, Any] | None = None,
) -> str:
    variants = max(1, min(5, int(variant_count or 1)))
    source_words = max(1, int(section.word_count or word_count(section.text)))
    payload = {
        "task": "residual_route_window_cleanup",
        "goal": "Rewrite only this window so the route feels naturally edited instead of mechanically conclusive.",
        "window": {
            "section_id": section.section_id,
            "source_window": section.text,
            "before_context": _section_before_context(section),
            "after_context": _section_after_context(section),
            "source_word_count": source_words,
            "source_event_beats": _source_event_beats(section.text),
            "source_phrase_anchors": _source_phrase_anchors(section.text),
            "referential_continuity": _referential_continuity(
                section.text,
                before_context=_section_before_context(section),
            ),
        },
        "editorial_findings": [
            "The selected window still reads too much like a neat route summary.",
            "Change the order or bridge between source beats instead of replacing isolated words.",
            "Make the interpretation follow concrete source actions, outcomes, citations, or quoted material already in the window.",
        ],
        "required_moves": [
            "Replace the full source_window only.",
            "Start from a concrete source beat when the window currently opens with a broad conclusion.",
            "Keep citations, quoted material, names, and assessment references already present in source_window.",
            "Use plain bachelor-level wording; do not polish the window into a generic academic conclusion.",
        ],
        "length_guidance": {
            "source_words": source_words,
            "preferred_min_words": max(8, round(source_words * 0.80)),
            "preferred_max_words": max(12, round(source_words * 1.20)),
        },
        "constraints": [
            "Do not add new facts, examples, names, dates, citations, dialogue, headings, bullets, markdown, HTML, or commentary.",
            "Do not change the source viewpoint.",
            "Do not return the whole document.",
            "Do not write a plan or explanation.",
        ],
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, variants + 1)
            ]
        },
    }
    if current_scores:
        payload["current_state"] = {
            "risky_window_count": current_scores.get("risky_window_count"),
            "unsafe_cluster_count": current_scores.get("unsafe_cluster_count"),
        }
    if route_plan:
        payload.update({
            "goal": "Rewrite only this window by executing the planned failed-route to replacement-route brief.",
            "execution_brief": route_plan,
            "source_blocks": _source_blocks(section.text),
            "coverage_guidance": _coverage_guidance_for_route_plan(section=section, route_plan=route_plan),
            "length_guidance": _length_guidance_for_route_plan(section=section, route_plan=route_plan),
            "method": _custom_route_writer_method(),
        })
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_unsafe_cluster_cleanup_prompt(
    *,
    section: SectionUnit,
    density_cluster: dict[str, Any],
    variant_count: int = 5,
    route_plan: dict[str, Any] | None = None,
) -> str:
    variants = max(1, min(5, int(variant_count or 1)))
    source_words = max(1, int(section.word_count or word_count(section.text)))
    payload = {
        "task": "single_density_cluster_cleanup",
        "goal": "Rewrite only this local cluster so it keeps the same meaning but stops reading as a predictable sentence unit.",
        "cluster": {
            "section_id": section.section_id,
            "source_cluster": section.text,
            "before_context": _section_before_context(section),
            "after_context": _section_after_context(section),
            "source_word_count": source_words,
            "source_event_beats": _source_event_beats(section.text),
            "source_phrase_anchors": _source_phrase_anchors(section.text),
            "referential_continuity": _referential_continuity(
                section.text,
                before_context=_section_before_context(section),
            ),
        },
        "editorial_findings": {
            "sentence_count": density_cluster.get("sentence_count"),
            "word_count": density_cluster.get("word_count"),
            "preview": density_cluster.get("preview"),
            "broad_generic_pressure": bool(density_cluster.get("generic_hits")),
            "transition_pressure": bool(density_cluster.get("transition_count")),
        },
        "repair_moves": [
            "Change the local route, not just one synonym.",
            "Tie any broad claim back to concrete words already in source_cluster or nearby context.",
            "Keep the wording plain and source-near.",
            "If the cluster contains an obvious splice or duplicate word, repair it cleanly while preserving meaning.",
        ],
        "must_preserve": [
            "same factual meaning",
            "same source viewpoint",
            "citations already present in source_cluster",
            "direct quoted text already present in source_cluster",
        ],
        "length_guidance": {
            "source_words": source_words,
            "preferred_min_words": max(8, round(source_words * 0.80)),
            "preferred_max_words": max(12, round(source_words * 1.25)),
        },
        "constraints": [
            "Do not add new facts, examples, names, dates, citations, dialogue, headings, bullets, markdown, HTML, or commentary.",
            "Do not make the writing casual or slangy.",
            "Do not return the whole document.",
            "Do not write a plan or explanation.",
        ],
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, variants + 1)
            ]
        },
    }
    if route_plan:
        payload.update({
            "goal": "Rewrite only this local cluster by executing the planned failed-route to replacement-route brief.",
            "execution_brief": route_plan,
            "source_blocks": _source_blocks(section.text),
            "coverage_guidance": _coverage_guidance_for_route_plan(section=section, route_plan=route_plan),
            "length_guidance": _length_guidance_for_route_plan(section=section, route_plan=route_plan),
            "method": _custom_route_writer_method(),
        })
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_risky_window_cleanup_variants(
    *,
    section: SectionUnit,
    current_scores: dict[str, Any],
    gateway: LLMGateway,
    variant_count: int = 5,
    route_plan: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    prompt = build_risky_window_cleanup_prompt(
        section=section,
        current_scores=current_scores,
        variant_count=variant_count,
        route_plan=route_plan,
    )
    return _generate_loose_variants(prompt=prompt, gateway=gateway, variant_count=variant_count)


def generate_unsafe_cluster_cleanup_variants(
    *,
    section: SectionUnit,
    density_cluster: dict[str, Any],
    gateway: LLMGateway,
    variant_count: int = 5,
    route_plan: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    prompt = build_unsafe_cluster_cleanup_prompt(
        section=section,
        density_cluster=density_cluster,
        variant_count=variant_count,
        route_plan=route_plan,
    )
    return _generate_loose_variants(prompt=prompt, gateway=gateway, variant_count=variant_count)


def _generate_loose_variants(
    *,
    prompt: str,
    gateway: LLMGateway,
    variant_count: int,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    variants = max(1, min(5, int(variant_count or 1)))
    structured = structured_json_request_options(getattr(gateway, "model", None), _variants_response_format(variants))
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured.get("provider"))
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a variants array.",
        response_format=structured.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_TEMPERATURE", 0.35, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_TOP_P", 0.9, minimum=0.1, maximum=1.0),
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_MAX_TOKENS", 8000, minimum=1000, maximum=12000),
    )
    raw = response.raw_content or response.content
    parsed, diagnostics = _parse_loose_variants(raw)
    return parsed, {
        **diagnostics,
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "structured_output_mode": structured.get("structured_output_mode"),
    }, prompt, raw


def _parse_route_plan(raw: str, *, source_text: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    payload, diagnostics = parse_json_object(raw, required_keys={"route_plan"})
    if payload is None:
        return None, diagnostics
    plan = payload.get("route_plan")
    if not isinstance(plan, dict):
        return None, {**diagnostics, "status": "schema_failed", "reason": "route_plan_not_object"}
    sanitized = _sanitize_route_plan(plan, source_text=source_text)
    if not _route_plan_valid(sanitized):
        return None, {
            **diagnostics,
            "status": "schema_failed",
            "reason": "route_plan_has_no_executable_brief",
            "route_plan_keys": sorted(plan.keys()),
            "dropped_must_preserve_count": _must_preserve_input_count(plan.get("must_preserve")) - len(sanitized.get("must_preserve") or []),
        }
    return sanitized, {
        **diagnostics,
        "status": "ok",
        "content_profile": sanitized.get("content_profile"),
        "cluster_role": sanitized.get("cluster_role"),
        "dominant_failure_pattern": sanitized.get("dominant_failure_pattern"),
        "route_strategy": sanitized.get("route_strategy"),
        "source_block_plan_count": len(sanitized.get("source_block_plan") or []),
        "target_sentence_job_count": len(sanitized.get("target_sentence_jobs") or []),
        "must_change_count": len(sanitized.get("must_change") or []),
        "must_preserve_count": len(sanitized.get("must_preserve") or []),
        "sentence_plan_count": len(sanitized.get("sentence_plan") or []),
        "length_target": sanitized.get("length_target"),
    }


def _sanitize_route_plan(plan: dict[str, Any], *, source_text: str) -> dict[str, Any]:
    source = str(source_text or "")
    return {
        "content_profile": _content_profile(plan.get("content_profile")),
        "cluster_role": _cluster_role(plan.get("cluster_role")),
        "dominant_failure_pattern": _failure_pattern(plan.get("dominant_failure_pattern")),
        "route_strategy": _route_strategy(plan.get("route_strategy")),
        "profile_reason": _short_string(plan.get("profile_reason"), limit=220),
        "failed_route": _short_string(plan.get("failed_route"), limit=320),
        "replacement_route": _short_string(plan.get("replacement_route"), limit=360),
        "source_block_plan": _sanitize_source_block_plan(plan.get("source_block_plan"), source_text=source, limit=8),
        "target_sentence_jobs": _sanitize_target_sentence_jobs(plan.get("target_sentence_jobs"), source_text=source, limit=8),
        "must_change": _string_list(plan.get("must_change"), limit=8),
        "must_preserve": _sanitize_must_preserve(plan.get("must_preserve"), source_text=source, limit=16),
        "sentence_plan": _string_list(plan.get("sentence_plan"), limit=8),
        "avoid_phrases": _supported_or_short_list(plan.get("avoid_phrases"), source_text=source, limit=12),
        "length_target": _length_target(plan.get("length_target")),
        "reason_this_should_move_score": _short_string(plan.get("reason_this_should_move_score"), limit=320),
    }


def _sanitize_current_route(value: Any, *, source_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in value if isinstance(value, list) else []:
        if not isinstance(row, dict):
            continue
        quote = _supported_quote(row.get("source_quote"), source_text)
        if not quote:
            continue
        rows.append({
            "source_quote": quote,
            "function": _short_string(row.get("function"), limit=160),
            "weakness": _short_string(row.get("weakness"), limit=180),
        })
        if len(rows) >= 8:
            break
    return rows


def _sanitize_better_route(value: Any, *, source_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value if isinstance(value, list) else [], start=1):
        if not isinstance(row, dict):
            continue
        quotes = [
            quote
            for quote in (_supported_quote(item, source_text) for item in _raw_list(row.get("source_quotes")))
            if quote
        ]
        if not quotes:
            continue
        job = _short_string(row.get("job"), limit=260)
        if not job:
            continue
        rows.append({
            "job_id": _short_string(row.get("job_id"), limit=24) or f"j{index}",
            "job": job,
            "source_quotes": quotes[:3],
            "avoid_copying": _supported_or_short_list(row.get("avoid_copying"), source_text=source_text, limit=4),
        })
        if len(rows) >= 8:
            break
    return rows


def _sanitize_phrase_repaths(value: Any, *, source_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in value if isinstance(value, list) else []:
        if not isinstance(row, dict):
            continue
        source = _supported_quote(row.get("source"), source_text)
        direction = _short_string(row.get("plain_direction"), limit=220)
        if not source or not direction:
            continue
        rows.append({"source": source, "plain_direction": direction})
        if len(rows) >= 10:
            break
    return rows


def _sanitize_source_block_plan(value: Any, *, source_text: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value if isinstance(value, list) else [], start=1):
        if not isinstance(row, dict):
            continue
        rewrite_job = _short_string(row.get("rewrite_job"), limit=260)
        if not rewrite_job:
            continue
        rows.append({
            "block_id": _short_string(row.get("block_id"), limit=24) or f"b{index:02d}",
            "current_job": _short_string(row.get("current_job"), limit=180),
            "rewrite_job": rewrite_job,
            "must_preserve": _supported_or_short_list(row.get("must_preserve"), source_text=source_text, limit=4),
        })
        if len(rows) >= limit:
            break
    return rows


def _sanitize_target_sentence_jobs(value: Any, *, source_text: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value if isinstance(value, list) else [], start=1):
        if not isinstance(row, dict):
            continue
        source_preview = _supported_quote(row.get("source_preview"), source_text) or _short_string(row.get("source_preview"), limit=240)
        rewrite_job = _short_string(row.get("rewrite_job"), limit=260)
        if not source_preview or not rewrite_job:
            continue
        rows.append({
            "sentence_id": _short_string(row.get("sentence_id"), limit=32) or f"s{index:03d}",
            "source_preview": source_preview,
            "current_weakness": _short_string(row.get("current_weakness"), limit=180),
            "rewrite_job": rewrite_job,
            "avoid_copying": _supported_or_short_list(row.get("avoid_copying"), source_text=source_text, limit=4),
        })
        if len(rows) >= limit:
            break
    return rows


def _route_plan_valid(plan: Any) -> bool:
    return (
        isinstance(plan, dict)
        and _content_profile(plan.get("content_profile")) in set(_ROUTE_PLAN_CONTENT_PROFILES)
        and _cluster_role(plan.get("cluster_role")) in set(_ROUTE_PLAN_CLUSTER_ROLES)
        and _failure_pattern(plan.get("dominant_failure_pattern")) in set(_ROUTE_PLAN_FAILURE_PATTERNS)
        and _route_strategy(plan.get("route_strategy")) in set(_ROUTE_PLAN_STRATEGIES)
        and bool(plan.get("source_block_plan"))
        and bool(plan.get("target_sentence_jobs"))
        and bool(plan.get("replacement_route"))
        and bool(plan.get("must_change"))
        and bool(plan.get("must_preserve"))
        and bool(plan.get("sentence_plan"))
        and _length_target(plan.get("length_target")) in {"same_length", "slight_expand", "expand"}
    )


def _content_profile(value: Any) -> str:
    profile = _short_string(value, limit=80)
    if profile in _ROUTE_PLAN_CONTENT_PROFILES:
        return profile
    return "mixed_or_unknown"


def _cluster_role(value: Any) -> str:
    role = _short_string(value, limit=80)
    if role in _ROUTE_PLAN_CLUSTER_ROLES:
        return role
    return "mixed_section"


def _failure_pattern(value: Any) -> str:
    pattern = _short_string(value, limit=80)
    if pattern in _ROUTE_PLAN_FAILURE_PATTERNS:
        return pattern
    return "mixed"


def _route_strategy(value: Any) -> str:
    strategy = _short_string(value, limit=80)
    if strategy in _ROUTE_PLAN_STRATEGIES:
        return strategy
    return "mixed_route_rebuild"


def _length_target(value: Any) -> str:
    target = _short_string(value, limit=40)
    if target in {"same_length", "slight_expand", "expand"}:
        return target
    return "slight_expand"


def _route_plan_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rewrite_v5_cluster_route_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "route_plan": {
                        "type": "object",
                        "properties": {
                            "content_profile": {
                                "type": "string",
                                "enum": list(_ROUTE_PLAN_CONTENT_PROFILES.keys()),
                            },
                            "cluster_role": {
                                "type": "string",
                                "enum": list(_ROUTE_PLAN_CLUSTER_ROLES.keys()),
                            },
                            "dominant_failure_pattern": {
                                "type": "string",
                                "enum": list(_ROUTE_PLAN_FAILURE_PATTERNS.keys()),
                            },
                            "route_strategy": {
                                "type": "string",
                                "enum": list(_ROUTE_PLAN_STRATEGIES.keys()),
                            },
                            "profile_reason": {"type": "string"},
                            "failed_route": {"type": "string"},
                            "replacement_route": {"type": "string"},
                            "source_block_plan": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 8,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "block_id": {"type": "string"},
                                        "current_job": {"type": "string"},
                                        "rewrite_job": {"type": "string"},
                                        "must_preserve": {
                                            "type": "array",
                                            "minItems": 0,
                                            "maxItems": 4,
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["block_id", "current_job", "rewrite_job", "must_preserve"],
                                    "additionalProperties": False,
                                },
                            },
                            "target_sentence_jobs": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 8,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "sentence_id": {"type": "string"},
                                        "source_preview": {"type": "string"},
                                        "current_weakness": {"type": "string"},
                                        "rewrite_job": {"type": "string"},
                                        "avoid_copying": {
                                            "type": "array",
                                            "minItems": 0,
                                            "maxItems": 4,
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["sentence_id", "source_preview", "current_weakness", "rewrite_job", "avoid_copying"],
                                    "additionalProperties": False,
                                },
                            },
                            "must_change": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 8,
                                "items": {"type": "string"},
                            },
                            "must_preserve": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 16,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_quote": {"type": "string"},
                                        "preserve_as": {"type": "string"},
                                    },
                                    "required": ["source_quote", "preserve_as"],
                                    "additionalProperties": False,
                                },
                            },
                            "sentence_plan": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 8,
                                "items": {"type": "string"},
                            },
                            "avoid_phrases": {
                                "type": "array",
                                "minItems": 0,
                                "maxItems": 12,
                                "items": {"type": "string"},
                            },
                            "length_target": {
                                "type": "string",
                                "enum": ["same_length", "slight_expand", "expand"],
                            },
                            "reason_this_should_move_score": {"type": "string"},
                        },
                        "required": [
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
                        ],
                        "additionalProperties": False,
                    }
                },
                "required": ["route_plan"],
                "additionalProperties": False,
            },
        },
    }


def _parse_loose_variants(raw: str) -> tuple[list[RecompositionVariant], dict[str, Any]]:
    payload, diagnostics = parse_json_object(raw, required_keys={"variants"})
    if payload is None:
        return [], diagnostics
    rows = payload.get("variants")
    if not isinstance(rows, list):
        return [], {**diagnostics, "status": "schema_failed", "reason": "variants_not_array"}
    variants: list[RecompositionVariant] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            rejected.append({"index": index, "reason": "variant_not_object"})
            continue
        if set(row.keys()) != {"variant_id", "text"}:
            rejected.append({"index": index, "reason": "variant_keys_mismatch", "keys": sorted(row.keys())})
            continue
        variant_id = str(row.get("variant_id") or "").strip()
        text = str(row.get("text") or "").strip()
        if not variant_id or not text:
            rejected.append({"index": index, "reason": "empty_variant"})
            continue
        integrity = minimal_replacement_text_integrity(text)
        if not integrity.get("passed"):
            rejected.append({"index": index, "variant_id": variant_id, "reason": "text_integrity_failed", "text_integrity": integrity})
            continue
        variants.append(RecompositionVariant(variant_id=variant_id, text=text, word_count=word_count(text)))
    return variants, {**diagnostics, "status": "ok" if variants else "schema_failed", "variant_count": len(variants), "rejected": rejected}


def _score_residual_variant(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_scores: dict[str, Any],
    current_text: str,
    current_scores: dict[str, Any],
    section: SectionUnit,
    variant: RecompositionVariant,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    candidate_text, apply_status = apply_section_variant(current_text, section, variant.text)
    if not apply_status.get("applied"):
        return {
            "section_id": section.section_id,
            "variant_id": variant.variant_id,
            "label": label,
            "text": variant.text,
            "word_count": variant.word_count,
            "apply_status": apply_status,
            "scores": current_scores,
            "incremental": {},
            "local_scores": {},
            "local_goal": {},
        }
    candidate_report = _scan_report(candidate_text)
    candidate_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=baseline_report,
        candidate_report=candidate_report,
    ).to_dict()
    scores = _score_summary(original_text, candidate_report, candidate_goal)
    _add_deltas(scores, baseline_scores)
    local_before_report = _scan_report(section.text)
    local_before_goal = evaluate_rewrite_goal(
        original_text=section.text,
        candidate_text=section.text,
        original_report=local_before_report,
        candidate_report=local_before_report,
    ).to_dict()
    local_before_scores = _score_summary(section.text, local_before_report, local_before_goal)
    local_after_report = _scan_report(variant.text)
    local_after_goal = evaluate_rewrite_goal(
        original_text=section.text,
        candidate_text=variant.text,
        original_report=local_before_report,
        candidate_report=local_after_report,
    ).to_dict()
    local_after_scores = _score_summary(section.text, local_after_report, local_after_goal)
    _add_deltas(local_after_scores, local_before_scores)
    safe_name = label.replace("/", "_")
    (output_dir / f"{safe_name}.txt").write_text(candidate_text)
    (output_dir / f"{safe_name}_cluster.txt").write_text(variant.text)
    (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
    return {
        "section_id": section.section_id,
        "variant_id": variant.variant_id,
        "label": label,
        "text": variant.text,
        "word_count": variant.word_count,
        "apply_status": apply_status,
        "scores": scores,
        "incremental": _incremental_deltas(scores, current_scores),
        "local_scores": local_after_scores,
        "local_goal": local_after_goal,
        "candidate_text": candidate_text,
        "candidate_report": candidate_report,
        "candidate_goal": candidate_goal,
    }


def _run_risky_window_cleanup_pass(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_scores: dict[str, Any],
    current_text: str,
    current_report: dict[str, Any],
    current_goal: dict[str, Any],
    current_scores: dict[str, Any],
    gateway: LLMGateway,
    output_dir: Path,
    global_best_candidate: dict[str, Any] | None,
    max_rounds: int,
    variant_count: int,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, Any]] = []
    skipped: set[tuple[Any, ...]] = set()
    for cleanup_index in range(1, max(0, int(max_rounds or 0)) + 1):
        target = _select_risky_window_section(current_text, current_report, skipped=skipped)
        if target is None:
            rounds.append({"round": cleanup_index, "phase": "risky_window_cleanup", "status": "stopped", "reason": "no_risky_window_target"})
            break
        section, signature = target
        round_dir = output_dir / f"round_{cleanup_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        route_plan, route_plan_diagnostics, route_plan_prompt, route_plan_completion = generate_residual_cluster_route_plan(
            section=section,
            local_goal=_local_goal(section.text, section.text),
            gateway=gateway,
        )
        (round_dir / "route_plan_prompt.json.txt").write_text(route_plan_prompt)
        (round_dir / "route_plan_completion.json.txt").write_text(route_plan_completion)
        variants, llm_diagnostics, prompt, completion = generate_risky_window_cleanup_variants(
            section=section,
            current_scores=current_scores,
            gateway=gateway,
            variant_count=variant_count,
            route_plan=route_plan,
        )
        diagnostics = {
            "route_plan": route_plan_diagnostics,
            "llm_generation": llm_diagnostics,
        }
        (round_dir / "prompt.json.txt").write_text(prompt)
        (round_dir / "completion.json.txt").write_text(completion)
        rows = [
            _score_residual_variant(
                original_text=original_text,
                baseline_report=baseline_report,
                baseline_scores=baseline_scores,
                current_text=current_text,
                current_scores=current_scores,
                section=section,
                variant=variant,
                output_dir=round_dir,
                label=f"window_{variant.variant_id}",
            )
            for variant in variants
        ]
        global_best_candidate = _best_full_document_candidate([global_best_candidate, *rows])
        selected = _best_risky_window_cleanup_candidate(rows)
        accepted = selected if selected and _has_risky_window_cleanup_movement(selected) else None
        round_payload = {
            "round": cleanup_index,
            "phase": "risky_window_cleanup",
            "status": "accepted" if accepted else "skipped",
            "reason": "accepted_risky_window_movement" if accepted else "no_risky_window_movement",
            "section": section.to_dict(),
            "generator_diagnostics": diagnostics,
            "current_scores": current_scores,
            "candidates": [_compact_residual_row(row) for row in rows],
            "selected": _compact_residual_row(selected),
            "accepted": _compact_residual_row(accepted),
        }
        rounds.append(round_payload)
        (round_dir / "round_result.json").write_text(json.dumps(round_payload, ensure_ascii=False, indent=2))
        if not accepted:
            skipped.add(signature)
            continue
        current_text, current_report, current_goal, current_scores = _accepted_state(
            accepted=accepted,
            original_text=original_text,
            baseline_report=baseline_report,
        )
        (output_dir / f"after_round_{cleanup_index:02d}.txt").write_text(current_text)
    return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate


def _run_unsafe_cluster_cleanup_pass(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_scores: dict[str, Any],
    current_text: str,
    current_report: dict[str, Any],
    current_goal: dict[str, Any],
    current_scores: dict[str, Any],
    gateway: LLMGateway,
    output_dir: Path,
    global_best_candidate: dict[str, Any] | None,
    max_rounds: int,
    variant_count: int,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, Any]] = []
    skipped: set[tuple[Any, ...]] = set()
    for cleanup_index in range(1, max(0, int(max_rounds or 0)) + 1):
        density = build_eligible_span_density_contract(current_text, current_report)
        if density.get("safe"):
            rounds.append({
                "round": cleanup_index,
                "phase": "unsafe_cluster_cleanup",
                "status": "stopped",
                "reason": "eligible_span_density_safe",
                "density_gate": _compact_density_gate(density),
            })
            break
        target = _select_density_cluster_section(current_text, current_report, density, skipped=skipped)
        if target is None:
            rounds.append({
                "round": cleanup_index,
                "phase": "unsafe_cluster_cleanup",
                "status": "stopped",
                "reason": "no_density_cluster_target",
                "density_gate": _compact_density_gate(density),
            })
            break
        section, density_cluster, signature = target
        round_dir = output_dir / f"round_{cleanup_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        route_plan, route_plan_diagnostics, route_plan_prompt, route_plan_completion = generate_residual_cluster_route_plan(
            section=section,
            local_goal=_local_goal(section.text, section.text),
            gateway=gateway,
        )
        (round_dir / "route_plan_prompt.json.txt").write_text(route_plan_prompt)
        (round_dir / "route_plan_completion.json.txt").write_text(route_plan_completion)
        variants, llm_diagnostics, prompt, completion = generate_unsafe_cluster_cleanup_variants(
            section=section,
            density_cluster=density_cluster,
            gateway=gateway,
            variant_count=variant_count,
            route_plan=route_plan,
        )
        diagnostics = {
            "route_plan": route_plan_diagnostics,
            "llm_generation": llm_diagnostics,
        }
        (round_dir / "prompt.json.txt").write_text(prompt)
        (round_dir / "completion.json.txt").write_text(completion)
        rows = [
            _score_residual_variant(
                original_text=original_text,
                baseline_report=baseline_report,
                baseline_scores=baseline_scores,
                current_text=current_text,
                current_scores=current_scores,
                section=section,
                variant=variant,
                output_dir=round_dir,
                label=f"density_{variant.variant_id}",
            )
            for variant in variants
        ]
        global_best_candidate = _best_full_document_candidate([global_best_candidate, *rows])
        selected = _best_unsafe_cluster_cleanup_candidate(rows)
        accepted = selected if selected and _has_unsafe_cluster_cleanup_movement(selected) else None
        round_payload = {
            "round": cleanup_index,
            "phase": "unsafe_cluster_cleanup",
            "status": "accepted" if accepted else "skipped",
            "reason": "accepted_unsafe_cluster_movement" if accepted else "no_unsafe_cluster_movement",
            "section": section.to_dict(),
            "density_cluster": density_cluster,
            "density_gate": _compact_density_gate(density),
            "generator_diagnostics": diagnostics,
            "current_scores": current_scores,
            "candidates": [_compact_residual_row(row) for row in rows],
            "selected": _compact_residual_row(selected),
            "accepted": _compact_residual_row(accepted),
        }
        rounds.append(round_payload)
        (round_dir / "round_result.json").write_text(json.dumps(round_payload, ensure_ascii=False, indent=2))
        if not accepted:
            skipped.add(signature)
            continue
        current_text, current_report, current_goal, current_scores = _accepted_state(
            accepted=accepted,
            original_text=original_text,
            baseline_report=baseline_report,
        )
        (output_dir / f"after_round_{cleanup_index:02d}.txt").write_text(current_text)
    return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate


def _accepted_state(
    *,
    accepted: dict[str, Any],
    original_text: str,
    baseline_report: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    text = str(accepted.get("candidate_text") or "")
    report = accepted.get("candidate_report") if isinstance(accepted.get("candidate_report"), dict) else _scan_report(text)
    goal = accepted.get("candidate_goal") if isinstance(accepted.get("candidate_goal"), dict) else evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=text,
        original_report=baseline_report,
        candidate_report=report,
    ).to_dict()
    scores = accepted.get("scores") if isinstance(accepted.get("scores"), dict) else _score_summary(original_text, report, goal)
    return text, report, goal, scores


def _best_risky_window_cleanup_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if (row.get("apply_status") or {}).get("applied")]
    if not eligible:
        return None
    return max(eligible, key=_risky_window_cleanup_sort_key)


def _best_unsafe_cluster_cleanup_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if (row.get("apply_status") or {}).get("applied")]
    if not eligible:
        return None
    return max(eligible, key=_unsafe_cluster_cleanup_sort_key)


def _best_full_document_candidate(rows: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    eligible = [row for row in rows if isinstance(row, dict) and _has_full_document_fallback_movement(row)]
    if not eligible:
        return None
    return max(eligible, key=_full_document_candidate_sort_key)


def _full_document_candidate_beats_scores(row: dict[str, Any], current_scores: dict[str, Any]) -> bool:
    if not _has_full_document_fallback_movement(row):
        return False
    current_row = {
        "apply_status": {"applied": True},
        "scores": current_scores,
    }
    return _full_document_candidate_sort_key(row) > _full_document_candidate_sort_key(current_row)


def _full_document_candidate_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return (
        _number(scores.get("ai_delta")),
        _number(scores.get("rank_delta")),
        _number(scores.get("topk_calibrated_risk_delta")),
        _number(scores.get("topk_delta")),
        _number(scores.get("external_ai_flag_risk_delta")),
        _number(scores.get("external_delta")),
        _number(scores.get("qualifying_text_ai_density_delta")),
        _number(scores.get("unsafe_cluster_count_delta")),
        _number(scores.get("risky_window_count_delta")),
        _number(scores.get("unsafe_word_ratio_delta")),
    )


def _has_full_document_fallback_movement(row: dict[str, Any]) -> bool:
    if not (row.get("apply_status") or {}).get("applied"):
        return False
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    if _number(scores.get("ai_delta")) < 0:
        return False
    if _number(scores.get("rank_delta")) < 0:
        return False
    if _number(scores.get("topk_calibrated_risk_delta")) < 0:
        return False
    if _number(scores.get("external_ai_flag_risk_delta")) < 0:
        return False
    if _number(scores.get("unsafe_cluster_count_delta")) < 0:
        return False
    if _number(scores.get("risky_window_count_delta")) < 0:
        return False
    return any(
        _number(scores.get(key)) > 0
        for key in (
            "ai_delta",
            "rank_delta",
            "topk_calibrated_risk_delta",
            "topk_delta",
            "external_ai_flag_risk_delta",
            "external_delta",
            "qualifying_text_ai_density_delta",
            "unsafe_cluster_count_delta",
            "risky_window_count_delta",
            "unsafe_word_ratio_delta",
        )
    )


def _risky_window_cleanup_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return (
        _number(incremental.get("risky_window_count_delta")),
        _number(incremental.get("rank_delta")),
        _number(incremental.get("unsafe_cluster_count_delta")),
        _number(incremental.get("external_delta")),
        _number(incremental.get("topk_delta")),
        _number(incremental.get("ai_delta")),
        _number(scores.get("rank_delta")),
    )


def _unsafe_cluster_cleanup_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return (
        _number(incremental.get("unsafe_cluster_count_delta")),
        _number(incremental.get("rank_delta")),
        _number(incremental.get("external_delta")),
        _number(incremental.get("topk_delta")),
        _number(incremental.get("ai_delta")),
        _number(scores.get("rank_delta")),
    )


def _has_risky_window_cleanup_movement(row: dict[str, Any]) -> bool:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    return (
        _number(incremental.get("risky_window_count_delta")) > 0
        and _number(incremental.get("rank_delta")) > 0
    )


def _has_unsafe_cluster_cleanup_movement(row: dict[str, Any]) -> bool:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    return (
        _number(incremental.get("unsafe_cluster_count_delta")) > 0
        and _number(incremental.get("rank_delta")) > 0
    )


def _best_residual_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if (row.get("apply_status") or {}).get("applied")]
    if not eligible:
        return None
    return max(eligible, key=_residual_candidate_sort_key)


def _residual_candidate_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    local = row.get("local_scores") if isinstance(row.get("local_scores"), dict) else {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    return (
        1.0 if _local_cluster_cleared(local) else 0.0,
        _number(local.get("unsafe_cluster_count_delta")),
        _number(local.get("topk_calibrated_risk_delta")),
        _number(local.get("unsafe_word_ratio_delta")),
        _number(incremental.get("unsafe_cluster_count_delta")),
        _number(incremental.get("rank_delta")),
        _number(incremental.get("ai_delta")),
        _number(scores.get("rank_delta")),
    )


def _needs_retune(row: dict[str, Any]) -> bool:
    local = row.get("local_scores") if isinstance(row.get("local_scores"), dict) else {}
    if not _local_cluster_cleared(local):
        return True
    return _number(local.get("topk_calibrated_risk")) > 25.0


def _has_incremental_movement(row: dict[str, Any]) -> bool:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    local = row.get("local_scores") if isinstance(row.get("local_scores"), dict) else {}
    if not _local_cluster_cleared(local) and not _local_cluster_directionally_improved(local):
        return False
    return any(
        _number(incremental.get(key)) > 0
        for key in ("unsafe_cluster_count_delta", "rank_delta", "ai_delta", "topk_delta", "external_delta")
    )


def _local_cluster_cleared(local_scores: dict[str, Any]) -> bool:
    return (
        _number(local_scores.get("unsafe_cluster_count")) <= 0
        and _number(local_scores.get("unsafe_word_ratio")) <= 0
    )


def _local_cluster_directionally_improved(local_scores: dict[str, Any]) -> bool:
    return (
        _number(local_scores.get("unsafe_cluster_count_delta")) >= 0
        and _number(local_scores.get("unsafe_word_ratio_delta")) > 0
        and _number(local_scores.get("topk_delta")) > 0
        and _number(local_scores.get("rank_delta")) > 0
    )


def _incremental_deltas(scores: dict[str, Any], current_scores: dict[str, Any]) -> dict[str, Any]:
    lower_is_better = (
        "ai",
        "topk",
        "external",
        "rank",
        "risky_window_count",
        "unsafe_word_ratio",
        "unsafe_cluster_count",
        "topk_calibrated_risk",
        "qualifying_text_ai_density",
        "ai_authorship",
        "external_ai_flag_risk",
    )
    result: dict[str, Any] = {}
    for key in lower_is_better:
        if key in scores and key in current_scores:
            result[f"{key}_delta"] = round(_number(current_scores.get(key)) - _number(scores.get(key)), 3)
    return result


def _select_risky_window_section(
    current_text: str,
    current_report: dict[str, Any],
    *,
    skipped: set[tuple[Any, ...]],
) -> tuple[SectionUnit, tuple[Any, ...]] | None:
    footprint = current_report.get("ai_footprint_profile") if isinstance(current_report, dict) else {}
    windows = footprint.get("top_risky_windows") if isinstance(footprint, dict) else []
    for ordinal, row in enumerate(windows if isinstance(windows, list) else [], start=1):
        if not isinstance(row, dict):
            continue
        section = _section_from_window_row(current_text, row, ordinal=ordinal)
        if section is None:
            continue
        signature = (
            row.get("window_id"),
            section.start_char,
            section.end_char,
            _short_string(section.text, limit=120),
        )
        if signature in skipped:
            continue
        return section, signature
    return None


def _select_density_cluster_section(
    current_text: str,
    current_report: dict[str, Any],
    density: dict[str, Any],
    *,
    skipped: set[tuple[Any, ...]],
) -> tuple[SectionUnit, dict[str, Any], tuple[Any, ...]] | None:
    clusters = density.get("top_unsafe_clusters") if isinstance(density, dict) else []
    for ordinal, cluster in enumerate(clusters if isinstance(clusters, list) else [], start=1):
        if not isinstance(cluster, dict):
            continue
        section = _section_from_density_cluster(current_text, current_report, cluster, ordinal=ordinal)
        if section is None:
            continue
        signature = (
            cluster.get("start_sentence"),
            cluster.get("end_sentence"),
            _short_string(str(cluster.get("preview") or section.text), limit=120),
        )
        if signature in skipped:
            continue
        return section, cluster, signature
    return None


def _section_from_window_row(current_text: str, row: dict[str, Any], *, ordinal: int) -> SectionUnit | None:
    start = _optional_int(row.get("start_index"))
    end = _optional_int(row.get("end_index"))
    source_text = str(row.get("source_text") or "").strip()
    if start is None or end is None or start < 0 or end <= start or end > len(current_text):
        if not source_text:
            return None
        located = current_text.find(source_text)
        if located < 0:
            return None
        start = located
        end = located + len(source_text)
    start, end = _expand_to_local_text_boundaries(current_text, start, end)
    text = current_text[start:end]
    if not text.strip():
        return None
    return SectionUnit(
        section_id=str(row.get("window_id") or f"risk_window_{ordinal:03d}"),
        heading="Risky window cleanup",
        text=text,
        start_char=start,
        end_char=end,
        paragraph_count=max(1, text.count("\n\n") + 1),
        word_count=word_count(text),
        metadata={
            "before_context": current_text[max(0, start - 420):start],
            "after_context": current_text[end:min(len(current_text), end + 420)],
            "source_metadata": {"risky_window": row},
        },
    )


def _section_from_density_cluster(
    current_text: str,
    current_report: dict[str, Any],
    cluster: dict[str, Any],
    *,
    ordinal: int,
) -> SectionUnit | None:
    rows = _sentence_rows_by_index(current_report)
    if not rows:
        return None
    start_index = _optional_int(cluster.get("start_sentence"))
    end_index = _optional_int(cluster.get("end_sentence"))
    if start_index is None:
        return None
    if end_index is None:
        end_index = start_index
    selected = [row for row in rows if start_index <= int(row.get("sentence_index") or 0) <= end_index]
    if not selected:
        return None
    start = min(int(row.get("start_char") or 0) for row in selected)
    end = max(int(row.get("end_char") or 0) for row in selected)
    if start < 0 or end <= start or end > len(current_text):
        return None
    start, end = _expand_to_local_text_boundaries(current_text, start, end)
    text = current_text[start:end]
    if not text.strip():
        return None
    return SectionUnit(
        section_id=f"density_cluster_{ordinal:03d}",
        heading="Density cluster cleanup",
        text=text,
        start_char=start,
        end_char=end,
        paragraph_count=max(1, text.count("\n\n") + 1),
        word_count=word_count(text),
        metadata={
            "before_context": current_text[max(0, start - 420):start],
            "after_context": current_text[end:min(len(current_text), end + 420)],
            "source_metadata": {"density_cluster": cluster},
        },
    )


def _sentence_rows_by_index(report: dict[str, Any]) -> list[dict[str, Any]]:
    sentence_map = report.get("sentence_map") if isinstance(report, dict) else {}
    rows: list[dict[str, Any]] = []
    if not isinstance(sentence_map, dict):
        return rows
    ordered = sorted(
        [row for row in sentence_map.values() if isinstance(row, dict)],
        key=lambda item: (_optional_int(item.get("start_char")) or 0, _optional_int(item.get("end_char")) or 0),
    )
    for index, row in enumerate(ordered):
        start = _optional_int(row.get("start_char"))
        end = _optional_int(row.get("end_char"))
        if start is None or end is None:
            continue
        rows.append({
            **row,
            "sentence_index": index,
            "start_char": start,
            "end_char": end,
        })
    return rows


def _compact_density_gate(density: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "safe",
        "unsafe_sentence_count",
        "unsafe_word_count",
        "unsafe_eligible_word_ratio",
        "longest_unsafe_span_words",
        "unsafe_cluster_count",
        "thresholds",
        "recommended_actions",
    )
    return {key: density.get(key) for key in keys}


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expand_to_local_text_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    source = str(text or "")
    left = max(0, min(len(source), int(start)))
    right = max(left, min(len(source), int(end)))
    while left > 0 and left < len(source) and source[left - 1].isalnum() and source[left].isalnum():
        left -= 1
    while right > 0 and right < len(source) and source[right - 1].isalnum() and source[right].isalnum():
        right += 1
    while right < len(source) and source[right] in ".,;:!?":
        right += 1
    return left, right


def _local_unsafe_previews(local_goal: dict[str, Any]) -> list[str]:
    gate = local_goal.get("eligible_span_density_gate") if isinstance(local_goal, dict) else {}
    clusters = gate.get("top_unsafe_clusters") if isinstance(gate, dict) else []
    previews: list[str] = []
    for row in clusters if isinstance(clusters, list) else []:
        if not isinstance(row, dict):
            continue
        preview = str(row.get("preview") or "").strip()
        if preview:
            previews.append(preview[:320])
        if len(previews) >= 3:
            break
    return previews


def _local_top_sentence_targets(local_goal: dict[str, Any]) -> list[dict[str, Any]]:
    gate = local_goal.get("eligible_span_density_gate") if isinstance(local_goal, dict) else {}
    targets = gate.get("top_sentence_targets") if isinstance(gate, dict) else []
    rows: list[dict[str, Any]] = []
    for row in targets if isinstance(targets, list) else []:
        if not isinstance(row, dict):
            continue
        preview = str(row.get("preview") or "").strip()
        if not preview:
            continue
        rows.append({
            "sentence_id": row.get("sentence_id"),
            "preview": preview[:320],
            "word_count": row.get("word_count"),
            "generic_hits": row.get("generic_hits"),
        })
        if len(rows) >= 5:
            break
    return rows


def _local_recommended_actions(local_goal: dict[str, Any]) -> list[str]:
    gate = local_goal.get("eligible_span_density_gate") if isinstance(local_goal, dict) else {}
    return _string_list(gate.get("recommended_actions") if isinstance(gate, dict) else [], limit=6)


def _retune_focus_from_goal(local_goal: dict[str, Any]) -> list[str]:
    gate = local_goal.get("eligible_span_density_gate") if isinstance(local_goal, dict) else {}
    footprint = local_goal.get("ai_footprint_gate") if isinstance(local_goal, dict) else {}
    after = footprint.get("after") if isinstance(footprint, dict) else {}
    authorship = after.get("authorship_footprint") if isinstance(after, dict) else {}
    focus: list[str] = []
    if isinstance(gate, dict) and not gate.get("top_unsafe_clusters"):
        focus.append("The unsafe sentence density is cleared; keep that route but make the wording less formally paraphrased.")
    if isinstance(authorship, dict) and _number(authorship.get("topk_calibrated_risk")) > 25.0:
        focus.append("The remaining problem is calibrated top-k: stay closer to simple source wording and avoid formal substitute terms.")
    return focus[:3]


def _source_phrase_anchors(text: str) -> list[str]:
    anchors: list[str] = []
    for sentence in _source_event_beats(text):
        chunks = [sentence]
        for splitter in (", ", "; ", " and ", " but "):
            next_chunks: list[str] = []
            for chunk in chunks:
                next_chunks.extend(part.strip() for part in chunk.split(splitter) if part.strip())
            chunks = next_chunks
        for chunk in chunks:
            cleaned = " ".join(chunk.strip(" .,:;!?()[]{}\"'“”").split())
            if 2 <= len(cleaned.split()) <= 8 and cleaned not in anchors:
                anchors.append(cleaned)
            if len(anchors) >= 12:
                return anchors
    return anchors


def _non_source_terms(source_text: str, candidate_text: str) -> list[str]:
    source_terms = {_normalize_term(token) for token in str(source_text or "").split()}
    source_terms.discard("")
    terms: list[str] = []
    for token in str(candidate_text or "").split():
        normalized = _normalize_term(token)
        if len(normalized) < 7 or normalized in source_terms or normalized in terms:
            continue
        terms.append(normalized)
        if len(terms) >= 16:
            break
    return terms


def _normalize_term(token: str) -> str:
    return str(token or "").strip(" \t\r\n.,:;!?()[]{}\"'“”‘’").casefold()


def _raw_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _string_list(value: Any, *, limit: int) -> list[str]:
    items: list[str] = []
    for item in _raw_list(value):
        text = _short_string(item, limit=260)
        if text and text not in items:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def _supported_or_short_list(value: Any, *, source_text: str, limit: int) -> list[str]:
    items: list[str] = []
    for item in _raw_list(value):
        text = _supported_quote(item, source_text) or _short_string(item, limit=160)
        if text and text not in items:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def _sanitize_must_preserve(value: Any, *, source_text: str, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in _raw_list(value):
        if not isinstance(row, dict):
            continue
        quote = _supported_quote(row.get("source_quote"), source_text)
        preserve_as = _short_string(row.get("preserve_as"), limit=180)
        if not quote or not preserve_as:
            continue
        item = {"source_quote": quote, "preserve_as": preserve_as}
        if item not in rows:
            rows.append(item)
        if len(rows) >= limit:
            break
    return rows


def _must_preserve_input_count(value: Any) -> int:
    return len(_raw_list(value))


def _short_string(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    return text[:limit]


def _supported_quote(value: Any, source_text: str) -> str:
    quote = _short_string(value, limit=260)
    if not quote:
        return ""
    return quote if quote in source_text else ""


def build_route_blueprint(*, section: SectionUnit, local_goal: dict[str, Any] | None = None) -> dict[str, Any]:
    beats = _source_event_beats(section.text)
    problem_previews = _local_unsafe_previews(local_goal or {})
    source_metadata = section.metadata.get("source_metadata") if isinstance(section.metadata, dict) else {}
    gate_cluster = source_metadata.get("gate_cluster") if isinstance(source_metadata, dict) else {}
    generic_hits = _number(gate_cluster.get("generic_hits")) if isinstance(gate_cluster, dict) else 0.0
    start_index = _route_start_index(beats=beats, problem_previews=problem_previews, generic_hits=generic_hits)
    ordered_indexes = list(range(start_index, len(beats))) + list(range(0, start_index))
    steps: list[dict[str, Any]] = []
    for ordinal, beat_index in enumerate(ordered_indexes):
        if beat_index >= len(beats):
            continue
        instruction = _route_step_instruction(
            ordinal=ordinal,
            beat_index=beat_index,
            start_index=start_index,
            total=len(beats),
        )
        steps.append({
            "step_id": f"step_{len(steps) + 1:02d}",
            "source_beat_index": beat_index,
            "source_beat": beats[beat_index],
            "instruction": instruction,
        })
        if ordinal == 0 and start_index == 0 and len(beats) >= 4:
            steps.append({
                "step_id": f"step_{len(steps) + 1:02d}",
                "source_beat_index": beat_index,
                "source_beat": beats[beat_index],
                "instruction": "Add one bridge sentence explaining why this starting point shaped the next teaching action.",
            })
    avoid_openers = [beats[index] for index in range(0, start_index)]
    avoid_openers.extend(preview for preview in problem_previews if preview)
    return {
        "strategy": "event_first_rebuild" if start_index > 0 else "starting_point_bridge_rebuild",
        "start_source_beat_index": start_index,
        "avoid_openers": avoid_openers[:4],
        "steps": steps,
        "sentence_jobs": _sentence_jobs_for_blueprint(beats=beats, start_index=start_index),
    }


def _route_start_index(*, beats: list[str], problem_previews: list[str], generic_hits: float) -> int:
    if len(beats) <= 1:
        return 0
    first = beats[0]
    first_is_problem = any(first and first in preview for preview in problem_previews)
    if generic_hits > 0 and first_is_problem:
        return 1
    return 0


def _route_step_instruction(*, ordinal: int, beat_index: int, start_index: int, total: int) -> str:
    if ordinal == 0 and start_index > 0:
        return "Open from this concrete event or action instead of the source's broad summary opener."
    if ordinal == 0:
        return "Open from this starting point, but change the sentence route and avoid copying the source opener."
    if start_index > 0 and beat_index < start_index:
        return "Fold this source meaning into the ending as interpretation; do not use it as the opening."
    if ordinal == total - 1:
        return "Finish with this outcome, making the result visible rather than abstract."
    return "Unpack this beat through action, cause, or response before moving to the next beat."


def _sentence_jobs_for_blueprint(*, beats: list[str], start_index: int) -> list[str]:
    if not beats:
        return []
    jobs: list[str] = []
    if start_index == 0:
        jobs.append("Sentence 1: restate the starting situation using simple source-near wording.")
        if len(beats) >= 4:
            jobs.append("Sentence 2: explain why this starting point shaped the teacher's next action.")
        for index, beat in enumerate(beats[1:], start=2):
            if index == len(beats):
                jobs.append("Final sentence: make the outcome visible in plain wording.")
            else:
                jobs.append("Next sentence: move through the next source beat without formal theory language.")
        return jobs
    jobs.append("Sentence 1: start with the first concrete event/action beat, not the broad source opener.")
    for index, beat in enumerate(beats[start_index + 1:], start=start_index + 2):
        if index == len(beats):
            jobs.append("Next sentence: make the event consequence visible before interpreting it.")
        else:
            jobs.append("Next sentence: move to the next source beat in plain event wording.")
    for _ in beats[:start_index]:
        jobs.append("Final sentence: fold the skipped broad source opener into the meaning after the event.")
    return jobs


def _student_support_role_progression_seed_texts(text: str) -> list[str]:
    """Source-gated route seed for learner support -> role -> outcome clusters."""

    source = str(text or "")
    folded = source.casefold()
    required_terms = (
        "support worker",
        "interacted",
        "past learning",
        "role-playing",
        "group project",
        "positive feedback",
    )
    if any(term not in folded for term in required_terms):
        return []

    pronouns = _dominant_learner_pronouns(source)
    course_word = "course" if "course" in folded else "class"
    role_label = _role_label_from_source(source)
    task_label = _task_label_from_role(role_label)
    outcome_tail = "received positive feedback from them afterward"
    if "teammates" in folded:
        outcome_tail = "received positive feedback from them afterward"
    elif "client" in folded:
        outcome_tail = "received positive feedback afterward"

    student_ref = "the student"
    subject = pronouns["subject"]
    obj = pronouns["object"]
    poss = pronouns["possessive"]
    return [
        (
            f"When {student_ref} first joined the {course_word}, {subject} came to class with a support worker "
            f"and barely interacted with the group. I had to understand that starting point before expecting {obj} "
            f"to participate. Casual conversation gave me a better picture of {poss} past learning experiences. "
            f"After that, patient guidance and role-play moved the confidence work onto {task_label} {subject} "
            f"could actually handle. The group project gave {obj} a specific role rather than a general instruction "
            f"to join in. As the {role_label}, {subject} had to guide the task and communicate with teammates. "
            f"{subject.capitalize()} managed that responsibility successfully, led the group, and {outcome_tail}."
        )
    ]


def _dominant_learner_pronouns(text: str) -> dict[str, str]:
    tokens = [token.strip(" \t\r\n.,:;!?()[]{}\"'“”‘’").casefold() for token in str(text or "").split()]
    counts = {key: tokens.count(key) for key in ("he", "she", "they")}
    if counts.get("she", 0) > counts.get("he", 0) and counts.get("she", 0) >= counts.get("they", 0):
        return {"subject": "she", "object": "her", "possessive": "her"}
    if counts.get("they", 0) > counts.get("he", 0):
        return {"subject": "they", "object": "them", "possessive": "their"}
    return {"subject": "he", "object": "him", "possessive": "his"}


def _role_label_from_source(text: str) -> str:
    tokens = [
        token.strip(" \t\r\n.,:;!?()[]{}\"'“”‘’")
        for token in str(text or "").split()
        if token.strip(" \t\r\n.,:;!?()[]{}\"'“”‘’")
    ]
    for index, token in enumerate(tokens):
        if token.casefold() != "manager":
            continue
        start = max(0, index - 2)
        phrase_tokens = tokens[start:index + 1]
        if phrase_tokens:
            return " ".join(phrase_tokens)
    anchors = _source_phrase_anchors(text)
    for anchor in anchors:
        lowered = anchor.casefold()
        if "manager" in lowered and len(anchor.split()) <= 5:
            return anchor
    folded = str(text or "").casefold()
    if "manager" in folded:
        return "manager"
    return "specific role"


def _task_label_from_role(role_label: str) -> str:
    lowered = str(role_label or "").casefold()
    if "salon" in lowered:
        return "a salon task"
    if "manager" in lowered:
        return "a group task"
    return "a concrete group task"


def _local_goal(original_text: str, candidate_text: str) -> dict[str, Any]:
    original_report = _scan_report(original_text)
    candidate_report = original_report if original_text == candidate_text else _scan_report(candidate_text)
    return evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=original_report,
        candidate_report=candidate_report,
    ).to_dict()


def _source_event_beats(text: str, *, limit: int = 18) -> list[str]:
    return [sentence for sentence in _sentences(text)[:max(1, int(limit or 1))] if sentence.strip()]


def _source_blocks(text: str, *, limit: int = 8) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in str(text or "").split("\n\n"):
        cleaned = " ".join(block.split())
        if not cleaned:
            continue
        blocks.append({
            "block_id": f"b{len(blocks) + 1:02d}",
            "word_count": word_count(cleaned),
            "preview": _short_string(cleaned, limit=260),
        })
        if len(blocks) >= max(1, int(limit or 1)):
            break
    return blocks


def _section_before_context(section: SectionUnit) -> str:
    value = section.metadata.get("before_context") if isinstance(section.metadata, dict) else ""
    return _short_string(value, limit=420)


def _section_after_context(section: SectionUnit) -> str:
    value = section.metadata.get("after_context") if isinstance(section.metadata, dict) else ""
    return _short_string(value, limit=420)


def _referential_continuity(text: str, *, before_context: str = "") -> dict[str, Any]:
    sentences = _source_event_beats(text)
    first_sentence = sentences[0] if sentences else ""
    first_token = ""
    for token in first_sentence.split():
        cleaned = token.strip(" \t\r\n.,:;!?()[]{}\"'“”‘’")
        if cleaned:
            first_token = cleaned
            break
    personal_pronouns = {"he", "she", "they", "him", "her", "them"}
    preserve_opening_subject = first_token.casefold() in personal_pronouns
    named_refs = _named_references_from_text(before_context)
    for sentence in sentences:
        for token in sentence.split():
            cleaned = token.strip(" \t\r\n.,:;!?()[]{}\"'“”‘’")
            if not cleaned or not cleaned[:1].isupper() or cleaned == first_token:
                continue
            if cleaned not in named_refs:
                named_refs.append(cleaned)
            if len(named_refs) >= 6:
                break
        if len(named_refs) >= 6:
            break
    return {
        "opening_subject": first_token,
        "preserve_opening_subject": preserve_opening_subject,
        "named_references_in_cluster": named_refs,
        "preferred_reference": named_refs[0] if named_refs else "",
        "instruction": (
            "Use the same opening subject or an established named reference from context naturally; do not generalize the subject and do not explain the reference parenthetically."
            if preserve_opening_subject else
            "Keep source-supported names and pronouns consistent."
        ),
    }


def _named_references_from_text(text: str) -> list[str]:
    refs: list[str] = []
    sentence_initials = {
        sentence.split()[0].strip(" \t\r\n.,:;!?()[]{}\"'“”‘’")
        for sentence in _source_event_beats(text)
        if sentence.split()
    }
    for token in str(text or "").split():
        cleaned = token.strip(" \t\r\n.,:;!?()[]{}\"'“”‘’")
        if not cleaned or not cleaned[:1].isupper() or cleaned in sentence_initials:
            continue
        if cleaned not in refs:
            refs.append(cleaned)
        if len(refs) >= 6:
            break
    return refs


def _compact_residual_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "section_id": row.get("section_id"),
        "variant_id": row.get("variant_id"),
        "label": row.get("label"),
        "word_count": row.get("word_count"),
        "scores": row.get("scores"),
        "incremental": row.get("incremental"),
        "local_scores": row.get("local_scores"),
        "apply_status": row.get("apply_status"),
        "text": row.get("text"),
    }


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))
