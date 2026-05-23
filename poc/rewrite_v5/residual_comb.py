"""Residual cluster comb-through experiment for V5.

This path turns the single-cluster proof into an iterative experiment:
scan the current document, rebuild the strongest remaining cluster, score the
cluster and full document, accept useful movement, then repeat from the new
scan. It is intentionally isolated from production.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

from llm.gateway import LLMConfig, LLMGateway
from detect.layer3_scoring import _sentence_has_concrete_or_context
from rewrite_v2.goal_contract import evaluate_rewrite_goal
from rewrite_v2.structured_output import structured_json_request_options
from rewrite_v3.document_units import word_count
from rewrite_v3.pipeline import _scan_report
from rewrite_v3.text_integrity import minimal_replacement_text_integrity
from rewrite_controller.eligible_span_density import (
    build_eligible_span_density_contract,
    build_preferred_eligible_span_density_contract,
)
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
        "use_when": "The cluster is built around learning, work, practice, an observed case, a concrete event, or author reflection.",
        "planning_focus": "Turn broad reflection into event or process movement: context -> difficulty -> action/choice -> observed result -> limited judgment.",
        "avoid": "Do not convert the writer into a detached encyclopedia voice.",
    },
    "broad_explanatory_report": {
        "use_when": "The cluster explains a country, institution, topic, history, culture, economy, technology, or other broad subject for a report-style essay.",
        "planning_focus": "Replace category dumping with source-supported specificity: topic frame -> concrete framing or explanatory bridge -> grouped facts -> contrast or limit -> bridge to next topic.",
        "avoid": "Do not leave broad report wording as abstract category labels when the source gives usable specifics or context.",
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


_CONTROLLED_EXPANSION_MOVES = {
    "none": "No added bridge is required; route movement can happen through sentence order and clause route.",
    "explanatory_bridge": "Add a short bridge that explains why two source beats belong together.",
    "concrete_framing": "Frame a broad claim through concrete source terms or practical context.",
    "scope_limit": "Narrow an over-broad claim by stating its limit or condition.",
    "practical_consequence": "Attach the claim to a practical consequence already implied by the source route.",
    "contrast_or_specific_angle": "Use a contrast or sharper angle to break category-list movement.",
}


_HARD_WRITER_FAILURES = [
    "corrupted JSON or malformed output",
    "unsafe markup, markdown, labels, or commentary",
    "broken source meaning or missing hard anchors",
    "fake personal story when the source has no personal story",
    "fake citation, statistic, date, named event, or named factual claim",
    "junk text or repeated filler",
]


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
    planner_model: str | None = None,
    provider: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    risky_window_cleanup_rounds: int | None = None,
    unsafe_cluster_cleanup_rounds: int | None = None,
    cleanup_variant_count: int | None = None,
    final_risky_window_cleanup_rounds: int | None = None,
    direct_scanner_leapfrog_rounds: int | None = None,
    direct_scanner_leapfrog_variant_count: int | None = None,
    direct_scanner_leapfrog_batches: int | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    accepted_checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    author_proxy_context: dict[str, Any] | None = None,
    seed_candidate_texts: list[str] | None = None,
    max_seconds: float | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Iteratively treat the strongest residual cluster and rescan."""

    def raise_if_canceled() -> None:
        if cancellation_check is not None:
            cancellation_check()

    raise_if_canceled()
    started_at = time.monotonic()
    budget_seconds = _runtime_budget_seconds(max_seconds)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_text = str(input_text or "")
    baseline_report = _scan_report(original_text)
    raise_if_canceled()
    baseline_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=original_text,
        original_report=baseline_report,
        candidate_report=baseline_report,
    ).to_dict()
    baseline_goal = _with_v5_density_gate(original_text, baseline_report, baseline_goal)
    baseline_scores = _score_summary(original_text, baseline_report, baseline_goal)
    current_text = original_text
    current_report = baseline_report
    current_goal = baseline_goal
    current_scores = baseline_scores
    global_best_candidate: dict[str, Any] | None = None
    seed_candidate_rows: list[dict[str, Any]] = []
    seed_recovery: dict[str, Any] = {"applied": False, "reason": "no_seed_candidate"}
    accepted_checkpoints: list[dict[str, Any]] = []
    author_proxy_context = author_proxy_context if isinstance(author_proxy_context, dict) else {}

    def record_accepted_checkpoint(event: dict[str, Any]) -> None:
        raise_if_canceled()
        checkpoint = _accepted_checkpoint_payload(
            event=event,
            sequence=len(accepted_checkpoints) + 1,
            stage="v5_residual_cluster_comb",
            baseline_scores=baseline_scores,
            output_dir=out_dir,
        )
        accepted_checkpoints.append(_compact_accepted_checkpoint(checkpoint))
        if accepted_checkpoint_callback is None:
            return
        try:
            accepted_checkpoint_callback(checkpoint)
        except Exception as exc:
            accepted_checkpoints[-1]["callback_error"] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    if seed_candidate_texts:
        seed_candidate_rows = _score_seed_candidate_texts(
            seed_candidate_texts,
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_scores=current_scores,
            output_dir=out_dir / "seed_candidates",
            author_proxy_context=author_proxy_context,
        )
        global_best_candidate = _best_full_document_candidate([global_best_candidate, *seed_candidate_rows])
        if global_best_candidate and _full_document_candidate_beats_scores(global_best_candidate, current_scores):
            previous_scores = current_scores
            current_text, current_report, current_goal, current_scores = _accepted_state(
                accepted=global_best_candidate,
                original_text=original_text,
                baseline_report=baseline_report,
            )
            (out_dir / "after_seed_recovery.txt").write_text(current_text)
            record_accepted_checkpoint({
                "phase": "historical_seed_recovery",
                "round": None,
                "reason": "historical_candidate_superseded_original_start",
                "accepted": global_best_candidate,
                "rewritten_document": current_text,
                "scores": current_scores,
                "goal": current_goal,
            })
            seed_recovery = {
                "applied": True,
                "reason": "historical_candidate_superseded_original_start",
                "selected": _compact_residual_row(global_best_candidate),
                "previous_scores": previous_scores,
                "final_scores": current_scores,
            }
        else:
            seed_recovery = {
                "applied": False,
                "reason": "no_seed_candidate_beat_current_state",
                "selected": _compact_residual_row(global_best_candidate),
            }
    gateway = LLMGateway(LLMConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_MAX_TOKENS", 8000, minimum=1000, maximum=12000),
        temperature=_float_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_TEMPERATURE", 0.35, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_TOP_P", 0.9, minimum=0.1, maximum=1.0),
        provider=provider,
        timeout=180,
        extra_body=extra_body,
        cancellation_check=raise_if_canceled,
    ))
    planner_gateway = _planner_gateway(
        fallback_gateway=gateway,
        api_key=api_key,
        model=planner_model,
        base_url=base_url,
        provider=provider,
        extra_body=extra_body,
        cancellation_check=raise_if_canceled,
    )

    cleanup_variants = cleanup_variant_count if cleanup_variant_count is not None else variant_count
    risky_window_limit = _cleanup_round_limit(
        risky_window_cleanup_rounds,
        env_name="DRAFTPROOF_REWRITE_V5_RISKY_WINDOW_CLEANUP_ROUNDS",
        default=2,
    )
    unsafe_cluster_limit = _cleanup_round_limit(
        unsafe_cluster_cleanup_rounds,
        env_name="DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_CLEANUP_ROUNDS",
        default=12,
    )
    final_risky_window_limit = _cleanup_round_limit(
        final_risky_window_cleanup_rounds,
        env_name="DRAFTPROOF_REWRITE_V5_FINAL_RISKY_WINDOW_CLEANUP_ROUNDS",
        default=2,
    )
    borderline_verdict_variant_count = _borderline_verdict_variant_count(cleanup_variants)
    final_topk_sentence_route_enabled = _final_topk_sentence_route_enabled()
    safe_band_evidence_repair_enabled = _safe_band_evidence_repair_enabled()
    direct_scanner_limit = _cleanup_round_limit(
        direct_scanner_leapfrog_rounds,
        env_name="DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_LEAPFROG_ROUNDS",
        default=0,
    )
    direct_scanner_variants = (
        max(1, min(5, int(direct_scanner_leapfrog_variant_count or variant_count or 1)))
    )
    direct_scanner_batches = max(
        1,
        min(
            3,
            int(
                direct_scanner_leapfrog_batches
                if direct_scanner_leapfrog_batches is not None
                else _int_env("DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_LEAPFROG_BATCHES", 2, minimum=1, maximum=3)
            ),
        ),
    )
    seed_recovery_targeted_repair = bool(seed_recovery.get("applied")) and _bool_env(
        "DRAFTPROOF_REWRITE_V5_SEED_RECOVERY_TARGETED_REPAIR_ONLY",
        True,
    )
    if seed_recovery_targeted_repair:
        direct_scanner_limit = 0
        risky_window_limit = 0
        unsafe_cluster_limit = 0
        final_risky_window_limit = 0
    baseline_density_gate = _density_gate_for_report(current_text, current_report)
    if budget_seconds is None:
        budget_seconds = _adaptive_cutoff_runtime_budget_seconds(
            original_text=original_text,
            baseline_density_gate=baseline_density_gate,
            baseline_scores=baseline_scores,
        )
    adaptive_cutoff_events: list[dict[str, Any]] = []
    unsafe_cluster_first = (
        _bool_env("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_PROBE_BEFORE_CORE", False)
        and _should_start_with_unsafe_cluster_cleanup(
            density_gate=baseline_density_gate,
            unsafe_cluster_cleanup_rounds=unsafe_cluster_limit,
        )
    )
    unsafe_cluster_probe_limit = _unsafe_cluster_probe_round_limit(unsafe_cluster_limit) if unsafe_cluster_first else 0
    remaining_unsafe_cluster_limit = (
        max(0, unsafe_cluster_limit - unsafe_cluster_probe_limit)
        if unsafe_cluster_first
        else unsafe_cluster_limit
    )
    core_round_limit = 0 if seed_recovery_targeted_repair else max(1, int(max_rounds or 1))
    phase_order = {
        "unsafe_cluster_first": unsafe_cluster_first,
        "reason": (
            "historical_seed_targeted_safe_band_repair"
            if seed_recovery_targeted_repair
            else "eligible_span_density_unsafe"
            if unsafe_cluster_first
            else "default_route_then_cleanup"
        ),
        "seed_recovery_targeted_repair": seed_recovery_targeted_repair,
        "unsafe_cluster_selection_mode": "clearable" if unsafe_cluster_first and budget_seconds is not None else "scanner",
        "unsafe_cluster_probe_rounds": unsafe_cluster_probe_limit,
        "remaining_unsafe_cluster_rounds": remaining_unsafe_cluster_limit,
        "core_route_rounds": core_round_limit,
        "unsafe_cluster_probe_route_planning": not (unsafe_cluster_first and budget_seconds is not None),
        "unsafe_cluster_cleanup_route_planning": True,
        "direct_scanner_leapfrog_rounds": direct_scanner_limit,
        "direct_scanner_leapfrog_variants": direct_scanner_variants,
        "direct_scanner_leapfrog_batches": direct_scanner_batches,
        "direct_scanner_batch_policy": _direct_scanner_batch_policy(),
        "direct_scanner_route_planning": _bool_env("DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_ROUTE_PLANNING", True),
        "direct_scanner_selection_policy": "balanced_ai_topk",
        "author_proxy_context_active": _author_proxy_active(author_proxy_context),
        "planner_model": getattr(planner_gateway, "model", None),
        "writer_model": getattr(gateway, "model", None),
        "initial_density_gate": _compact_density_gate(baseline_density_gate),
        "borderline_verdict_cleanup": {
            "enabled": _borderline_verdict_cleanup_enabled(),
            "variant_count": borderline_verdict_variant_count,
            "pass_budget_seconds": _borderline_verdict_pass_budget_seconds(),
        },
        "final_topk_sentence_route": {
            "enabled": final_topk_sentence_route_enabled,
            "target_limit": _final_topk_sentence_route_target_limit(),
            "batch_size": _final_topk_sentence_route_batch_size(),
            "variant_count": _final_topk_sentence_route_variant_count(),
            "min_topk_delta": _final_topk_sentence_route_min_topk_delta(),
            "min_calibrated_delta": _final_topk_sentence_route_min_calibrated_delta(),
        },
        "safe_band_evidence_repair": {
            "enabled": safe_band_evidence_repair_enabled,
            "variant_count": _safe_band_evidence_repair_variant_count(),
            "section_limit": _safe_band_evidence_repair_section_limit(),
            "composite_window_enabled": _safe_band_evidence_repair_composite_window_enabled(),
            "composite_max_words": _safe_band_evidence_repair_composite_max_words(),
            "controlled_operation_enabled": _safe_band_controlled_operation_enabled(),
            "controlled_operation_target_limit": _safe_band_controlled_operation_target_limit(),
            "controlled_operation_round_limit": _safe_band_controlled_operation_round_limit(),
            "controlled_operation_min_suffix_words": _safe_band_controlled_operation_min_suffix_words(),
            "sentence_replacement_enabled": _safe_band_sentence_replacement_enabled(),
            "sentence_replacement_round_limit": _safe_band_sentence_replacement_round_limit(),
            "sentence_replacement_target_limit": _safe_band_sentence_replacement_target_limit(),
            "sentence_replacement_variant_count": _safe_band_sentence_replacement_variant_count(),
            "density_section_repair_enabled": _safe_band_density_section_repair_enabled(),
            "density_section_repair_round_limit": _safe_band_density_section_repair_round_limit(),
            "density_section_repair_section_limit": _safe_band_density_section_repair_section_limit(),
            "density_section_repair_variant_count": _safe_band_density_section_repair_variant_count(),
            "density_section_repair_min_section_words": _safe_band_density_section_repair_min_section_words(),
            "density_section_repair_max_section_words": _safe_band_density_section_repair_max_section_words(),
            "pack_enabled": _safe_band_evidence_pack_enabled(),
            "author_proxy_plan_enabled": _safe_band_author_proxy_plan_enabled(),
            "pack_composite_enabled": _safe_band_evidence_pack_composite_enabled(),
            "pack_variant_count": _safe_band_evidence_pack_variant_count(),
            "pack_section_limit": _safe_band_evidence_pack_section_limit(),
            "pack_max_section_words": _safe_band_evidence_pack_max_section_words(),
            "pack_max_source_words": _safe_band_evidence_pack_max_source_words(),
            "pack_partial_min_sections": _safe_band_evidence_pack_partial_min_sections(),
            "min_gap_delta": _safe_band_evidence_repair_min_gap_delta(),
            "density_checkpoint_min_density_delta": _safe_band_density_checkpoint_min_density_delta(),
            "density_checkpoint_ai_regression_tolerance": _safe_band_density_checkpoint_ai_regression_tolerance(),
            "density_checkpoint_authorship_regression_tolerance": _safe_band_density_checkpoint_authorship_regression_tolerance(),
            "density_checkpoint_max_ai_score": _safe_band_density_checkpoint_max_ai_score(),
            "density_checkpoint_topk_regression_tolerance": _safe_band_density_checkpoint_topk_regression_tolerance(),
            "unsafe_word_ratio_regression_tolerance": _safe_band_evidence_repair_unsafe_word_ratio_regression_tolerance(),
        },
        "adaptive_cutoff": {
            "enabled": _adaptive_cutoff_enabled(),
            "runtime_budget_enabled": budget_seconds is not None,
            "runtime_budget_seconds": round(float(budget_seconds), 3) if budget_seconds is not None else None,
            "initial_blocker_state": _adaptive_cutoff_blocker_state(
                baseline_scores,
                baseline_density_gate,
            ),
        },
    }

    rounds: list[dict[str, Any]] = []
    direct_scanner_rounds: list[dict[str, Any]] = []
    risky_window_rounds: list[dict[str, Any]] = []
    unsafe_cluster_rounds: list[dict[str, Any]] = []
    final_risky_window_rounds: list[dict[str, Any]] = []
    borderline_verdict_rounds: list[dict[str, Any]] = []
    final_topk_sentence_route_rounds: list[dict[str, Any]] = []
    safe_band_evidence_repair_rounds: list[dict[str, Any]] = []
    skipped_core_signatures: set[tuple[Any, ...]] = set()
    if seed_recovery_targeted_repair:
        rounds.append({
            "round": 0,
            "phase": "residual_cluster_comb",
            "status": "skipped",
            "reason": "historical_seed_targeted_safe_band_repair",
            "current_scores": current_scores,
        })
    if direct_scanner_limit > 0 and not _runtime_budget_exhausted(started_at, budget_seconds):
        _emit_progress(progress_callback, 67, "Running V5 direct scanner-cluster leapfrog")
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            direct_scanner_rounds,
            global_best_candidate,
        ) = _run_direct_scanner_leapfrog_pass(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            gateway=gateway,
            planner_gateway=planner_gateway,
            output_dir=out_dir / "direct_scanner_leapfrog",
            global_best_candidate=global_best_candidate,
            max_rounds=direct_scanner_limit,
            variant_count=direct_scanner_variants,
            max_batches=direct_scanner_batches,
            progress_callback=progress_callback,
            progress_percent=68,
            accepted_checkpoint_callback=record_accepted_checkpoint,
            started_at=started_at,
            max_seconds=budget_seconds,
            author_proxy_context=author_proxy_context,
        )
        raise_if_canceled()

    direct_scanner_accepted_count = sum(
        1
        for row in direct_scanner_rounds
        if isinstance(row, dict) and isinstance(row.get("accepted"), dict)
    )
    skip_core_after_direct = _should_skip_core_after_direct_accept(
        direct_scanner_accepted_count=direct_scanner_accepted_count,
        author_proxy_context=author_proxy_context,
    )
    phase_order["skip_core_after_direct_scanner_accept"] = skip_core_after_direct
    phase_order["direct_scanner_accepted_rounds"] = direct_scanner_accepted_count

    if unsafe_cluster_probe_limit > 0 and not _runtime_budget_exhausted(started_at, budget_seconds):
        _emit_progress(progress_callback, 67, "Probing V5 unsafe clusters")
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            unsafe_cluster_probe_rounds,
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
            planner_gateway=planner_gateway,
            output_dir=out_dir / "unsafe_cluster_cleanup_probe",
            global_best_candidate=global_best_candidate,
            max_rounds=unsafe_cluster_probe_limit,
            variant_count=cleanup_variants,
            selection_mode=str(phase_order["unsafe_cluster_selection_mode"]),
            route_plan_enabled=bool(phase_order["unsafe_cluster_probe_route_planning"]),
            progress_callback=progress_callback,
            progress_percent=68,
            accepted_checkpoint_callback=record_accepted_checkpoint,
            started_at=started_at,
            max_seconds=budget_seconds,
            author_proxy_context=author_proxy_context,
        )
        unsafe_cluster_rounds.extend(unsafe_cluster_probe_rounds)
        raise_if_canceled()
        event = _adaptive_cutoff_stop_event(
            phase="after_unsafe_cluster_probe",
            current_scores=current_scores,
            density_gate=_density_gate_for_report(current_text, current_report),
        )
        if event:
            adaptive_cutoff_events.append(event)

    if skip_core_after_direct:
        rounds.append({
            "round": 0,
            "phase": "residual_cluster_comb",
            "status": "skipped",
            "reason": "direct_scanner_leapfrog_accepted",
            "current_scores": current_scores,
        })

    for round_index in range(1, 0 if skip_core_after_direct else core_round_limit + 1):
        raise_if_canceled()
        if _runtime_budget_exhausted(started_at, budget_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="residual_cluster_comb",
                round_index=round_index,
                started_at=started_at,
                max_seconds=budget_seconds,
                current_scores=current_scores,
            ))
            break
        _emit_progress(
            progress_callback,
            _residual_progress_percent(round_index, max_rounds=max_rounds),
            f"V5 cluster route round {round_index}",
        )
        event = _adaptive_cutoff_stop_event(
            phase="before_core_round",
            current_scores=current_scores,
            density_gate=_density_gate_for_report(current_text, current_report),
        )
        if event:
            adaptive_cutoff_events.append({**event, "round": round_index})
            rounds.append({
                "round": round_index,
                "phase": "residual_cluster_comb",
                "status": "stopped",
                "reason": event["reason"],
                "adaptive_cutoff": event,
                "current_scores": current_scores,
            })
            break
        round_dir = out_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        cluster_units = build_cluster_repair_units(
            text=current_text,
            report=current_report,
            goal=current_goal,
            limit=max(1, core_round_limit),
            context_chars=220,
        )
        section: SectionUnit | None = None
        section_signature: tuple[Any, ...] | None = None
        for cluster_unit in cluster_units:
            candidate_section = _section_from_cluster(cluster_unit)
            candidate_signature = _core_section_signature(candidate_section)
            if candidate_signature in skipped_core_signatures:
                continue
            section = candidate_section
            section_signature = candidate_signature
            break
        if section is None:
            rounds.append({"round": round_index, "status": "stopped", "reason": "no_residual_cluster"})
            break
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
                author_proxy_context=author_proxy_context,
                author_proxy_phase="residual_cluster_comb_seed",
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
            if _runtime_budget_exhausted(started_at, budget_seconds):
                rounds.append(_runtime_budget_stop_record(
                    phase="residual_cluster_comb",
                    round_index=round_index,
                    started_at=started_at,
                    max_seconds=budget_seconds,
                    current_scores=current_scores,
                ))
                break
            _emit_progress(
                progress_callback,
                _residual_progress_percent(round_index, max_rounds=max_rounds),
                f"Planning V5 cluster route {round_index}",
            )
            route_plan, route_plan_diagnostics, plan_prompt, plan_completion = generate_residual_cluster_route_plan(
                section=section,
                local_goal=local_source_goal,
                planner_gateway=planner_gateway,
                fallback_gateway=gateway,
                author_proxy_context=author_proxy_context,
            )
            raise_if_canceled()
            (round_dir / "route_plan_prompt.json.txt").write_text(plan_prompt)
            (round_dir / "route_plan_completion.json.txt").write_text(plan_completion)
            if _runtime_budget_exhausted(started_at, budget_seconds):
                diagnostics = {
                    **diagnostics,
                    "route_plan": route_plan_diagnostics,
                    "llm_generation": {"status": "skipped", "reason": "runtime_budget_exhausted_after_route_plan"},
                }
            else:
                _emit_progress(
                    progress_callback,
                    min(75, _residual_progress_percent(round_index, max_rounds=max_rounds) + 1),
                    f"Generating V5 cluster candidates {round_index}",
                )
                requested_variant_count = max(1, min(5, int(variant_count or 1)))
                initial_variant_count = _adaptive_initial_variant_count(requested_variant_count, route_plan)
                diagnostics["adaptive_writer"] = {
                    "enabled": _adaptive_writer_enabled(route_plan),
                    "requested_variant_count": requested_variant_count,
                    "initial_variant_count": initial_variant_count,
                    "remaining_variant_count": max(0, requested_variant_count - initial_variant_count),
                }
                variants, llm_diagnostics, prompt, completion = generate_residual_cluster_variants(
                    section=section,
                    local_goal=local_source_goal,
                    gateway=gateway,
                    variant_count=initial_variant_count,
                    route_plan=route_plan,
                    author_proxy_context=author_proxy_context,
                )
                raise_if_canceled()
                diagnostics = {
                    **diagnostics,
                    "route_plan": route_plan_diagnostics,
                    "llm_generation": llm_diagnostics,
                }
                (round_dir / "cluster_prompt.json.txt").write_text(prompt)
                (round_dir / "cluster_completion.json.txt").write_text(completion)
                initial_rows = [
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
                        author_proxy_context=author_proxy_context,
                        author_proxy_phase="residual_cluster_comb_initial",
                    )
                    for variant in variants
                ]
                rows.extend(initial_rows)
                best_probe = _best_residual_candidate(initial_rows)
                feedback = _adaptive_writer_feedback(initial_rows, route_plan=route_plan, selected=best_probe)
                diagnostics["adaptive_writer"] = {
                    **(diagnostics.get("adaptive_writer") if isinstance(diagnostics.get("adaptive_writer"), dict) else {}),
                    "initial_feedback": feedback,
                }
                remaining_variant_count = max(0, requested_variant_count - initial_variant_count)
                if (
                    _adaptive_writer_enabled(route_plan)
                    and _should_generate_adaptive_remainder(
                        feedback,
                        remaining_count=remaining_variant_count,
                        best_candidate=best_probe,
                    )
                    and not _runtime_budget_exhausted(started_at, budget_seconds)
                ):
                    adaptive_variants, adaptive_diagnostics, adaptive_prompt, adaptive_completion = generate_residual_cluster_variants(
                        section=section,
                        local_goal=local_source_goal,
                        gateway=gateway,
                        variant_count=remaining_variant_count,
                        route_plan=route_plan,
                        adaptive_feedback=feedback,
                        author_proxy_context=author_proxy_context,
                    )
                    raise_if_canceled()
                    (round_dir / "cluster_adaptive_prompt.json.txt").write_text(adaptive_prompt)
                    (round_dir / "cluster_adaptive_completion.json.txt").write_text(adaptive_completion)
                    adaptive_rows = [
                        _score_residual_variant(
                            original_text=original_text,
                            baseline_report=baseline_report,
                            baseline_scores=baseline_scores,
                            current_text=current_text,
                            current_scores=current_scores,
                            section=section,
                            variant=variant,
                            output_dir=round_dir,
                            label=f"adaptive_{variant.variant_id}",
                            author_proxy_context=author_proxy_context,
                            author_proxy_phase="residual_cluster_comb_adaptive",
                        )
                        for variant in adaptive_variants
                    ]
                    rows.extend(adaptive_rows)
                    diagnostics["adaptive_writer"] = {
                        **(diagnostics.get("adaptive_writer") if isinstance(diagnostics.get("adaptive_writer"), dict) else {}),
                        "adaptive_retry": {
                            "triggered": True,
                            "variant_count": remaining_variant_count,
                            "llm_generation": adaptive_diagnostics,
                            "feedback_after_retry": _adaptive_writer_feedback(adaptive_rows, route_plan=route_plan),
                        },
                    }
                elif _adaptive_writer_enabled(route_plan) and remaining_variant_count > 0:
                    diagnostics["adaptive_writer"] = {
                        **(diagnostics.get("adaptive_writer") if isinstance(diagnostics.get("adaptive_writer"), dict) else {}),
                        "adaptive_retry": {
                            "triggered": False,
                            "reason": "early_incremental_movement_or_no_retry_needed",
                        },
                    }
        best_initial = _best_residual_candidate(rows)
        adaptive_feedback = _adaptive_writer_feedback(rows, route_plan=route_plan, selected=best_initial) if rows else {}
        if isinstance(diagnostics.get("adaptive_writer"), dict):
            diagnostics["adaptive_writer"] = {
                **diagnostics["adaptive_writer"],
                "final_feedback": adaptive_feedback,
            }
        if (
            not seed_accepted
            and best_initial
            and _should_retune_residual_candidate(
                best_initial,
                route_plan=route_plan,
                adaptive_feedback=adaptive_feedback,
            )
            and not _runtime_budget_exhausted(started_at, budget_seconds)
        ):
            _emit_progress(
                progress_callback,
                min(75, _residual_progress_percent(round_index, max_rounds=max_rounds) + 2),
                f"Retuning V5 cluster candidate {round_index}",
            )
            effective_retune_variant_count = _adaptive_retune_variant_count(retune_variant_count, route_plan)
            retuned, retune_diagnostics, retune_prompt, retune_completion = generate_residual_cluster_retunes(
                section=section,
                current_best_text=str(best_initial.get("text") or ""),
                local_goal=best_initial.get("local_goal") or {},
                gateway=gateway,
                variant_count=effective_retune_variant_count,
                route_plan=route_plan,
                author_proxy_context=author_proxy_context,
            )
            raise_if_canceled()
            retune_diagnostics = {
                **(retune_diagnostics or {}),
                "requested_variant_count": retune_variant_count,
                "effective_variant_count": effective_retune_variant_count,
                "adaptive_retune": _adaptive_writer_enabled(route_plan),
            }
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
                    author_proxy_context=author_proxy_context,
                    author_proxy_phase="residual_cluster_comb_retune",
                )
                for variant in retuned
            ]
        elif not seed_accepted and best_initial and _needs_retune(best_initial):
            retune_diagnostics = {
                "status": "skipped",
                "reason": (
                    "runtime_budget_exhausted_before_retune"
                    if _runtime_budget_exhausted(started_at, budget_seconds)
                    else "adaptive_retune_not_useful"
                ),
                "adaptive_feedback": adaptive_feedback,
            }
        all_rows = rows + retuned_rows
        global_best_candidate = _best_full_document_candidate([global_best_candidate, *all_rows])
        best = _best_residual_candidate(all_rows)
        accepted = best if best and _has_core_round_acceptance_movement(
            best,
            current_scores=current_scores,
            round_index=round_index,
        ) else None
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
            if section_signature is not None:
                skipped_core_signatures.add(section_signature)
            continue
        current_text = str(accepted.get("candidate_text") or current_text)
        current_report = accepted.get("candidate_report") if isinstance(accepted.get("candidate_report"), dict) else _scan_report(current_text)
        current_goal = accepted.get("candidate_goal") if isinstance(accepted.get("candidate_goal"), dict) else evaluate_rewrite_goal(
            original_text=original_text,
            candidate_text=current_text,
            original_report=baseline_report,
            candidate_report=current_report,
        ).to_dict()
        current_goal = _with_v5_density_gate(current_text, current_report, current_goal)
        current_scores = accepted.get("scores") if isinstance(accepted.get("scores"), dict) else _score_summary(original_text, current_report, current_goal)
        (out_dir / f"after_round_{round_index:02d}.txt").write_text(current_text)
        record_accepted_checkpoint({
            "phase": "residual_cluster_comb",
            "round": round_index,
            "reason": "accepted_incremental_movement",
            "accepted": accepted,
            "rewritten_document": current_text,
            "scores": current_scores,
            "goal": current_goal,
        })
        _emit_progress(
            progress_callback,
            min(75, _residual_progress_percent(round_index, max_rounds=max_rounds) + 2),
            f"Accepted V5 cluster round {round_index}",
        )
        event = _adaptive_cutoff_stop_event(
            phase="after_core_accept",
            current_scores=current_scores,
            density_gate=_density_gate_for_report(current_text, current_report),
        )
        if event:
            adaptive_cutoff_events.append({**event, "round": round_index})
            break
        if _runtime_budget_exhausted(started_at, budget_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="residual_cluster_comb",
                round_index=round_index + 1,
                started_at=started_at,
                max_seconds=budget_seconds,
                current_scores=current_scores,
            ))
            break

    event = (
        _adaptive_cutoff_stop_event(
            phase="before_risky_window_cleanup",
            current_scores=current_scores,
            density_gate=_density_gate_for_report(current_text, current_report),
        )
        if risky_window_limit > 0 and not _runtime_budget_exhausted(started_at, budget_seconds)
        else None
    )
    if event:
        adaptive_cutoff_events.append(event)
        risky_window_rounds.append({
            "round": 0,
            "phase": "risky_window_cleanup",
            "status": "skipped",
            "reason": event["reason"],
            "adaptive_cutoff": event,
            "current_scores": current_scores,
        })
    elif not _runtime_budget_exhausted(started_at, budget_seconds) and risky_window_limit > 0:
        _emit_progress(progress_callback, 76, "Cleaning V5 risky windows")
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
            planner_gateway=planner_gateway,
            output_dir=out_dir / "risky_window_cleanup",
            global_best_candidate=global_best_candidate,
            max_rounds=risky_window_limit,
            variant_count=cleanup_variants,
            progress_callback=progress_callback,
            progress_percent=76,
            accepted_checkpoint_callback=record_accepted_checkpoint,
            started_at=started_at,
            max_seconds=budget_seconds,
            author_proxy_context=author_proxy_context,
        )
        raise_if_canceled()

    event = (
        _adaptive_cutoff_stop_event(
            phase="before_unsafe_cluster_cleanup",
            current_scores=current_scores,
            density_gate=_density_gate_for_report(current_text, current_report),
        )
        if remaining_unsafe_cluster_limit > 0 and not _runtime_budget_exhausted(started_at, budget_seconds)
        else None
    )
    if event:
        adaptive_cutoff_events.append(event)
        unsafe_cluster_rounds.append({
            "round": 0,
            "phase": "unsafe_cluster_cleanup",
            "status": "skipped",
            "reason": event["reason"],
            "adaptive_cutoff": event,
            "current_scores": current_scores,
        })
    elif (
        not _runtime_budget_exhausted(started_at, budget_seconds)
        and remaining_unsafe_cluster_limit > 0
    ):
        _emit_progress(progress_callback, 77, "Cleaning V5 unsafe clusters")
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
            planner_gateway=planner_gateway,
            output_dir=out_dir / "unsafe_cluster_cleanup",
            global_best_candidate=global_best_candidate,
            max_rounds=remaining_unsafe_cluster_limit,
            variant_count=cleanup_variants,
            selection_mode="scanner",
            route_plan_enabled=True,
            progress_callback=progress_callback,
            progress_percent=77,
            accepted_checkpoint_callback=record_accepted_checkpoint,
            started_at=started_at,
            max_seconds=budget_seconds,
            author_proxy_context=author_proxy_context,
        )
        raise_if_canceled()

    event = (
        _adaptive_cutoff_stop_event(
            phase="before_final_risky_window_cleanup",
            current_scores=current_scores,
            density_gate=_density_gate_for_report(current_text, current_report),
        )
        if final_risky_window_limit > 0 and not _runtime_budget_exhausted(started_at, budget_seconds)
        else None
    )
    if event:
        adaptive_cutoff_events.append(event)
        final_risky_window_rounds.append({
            "round": 0,
            "phase": "final_risky_window_cleanup",
            "status": "skipped",
            "reason": event["reason"],
            "adaptive_cutoff": event,
            "current_scores": current_scores,
        })
    elif not _runtime_budget_exhausted(started_at, budget_seconds) and final_risky_window_limit > 0:
        _emit_progress(progress_callback, 78, "Final V5 risky window cleanup")
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
            planner_gateway=planner_gateway,
            output_dir=out_dir / "final_risky_window_cleanup",
            global_best_candidate=global_best_candidate,
            max_rounds=final_risky_window_limit,
            variant_count=cleanup_variants,
            progress_callback=progress_callback,
            progress_percent=78,
            accepted_checkpoint_callback=record_accepted_checkpoint,
            started_at=started_at,
            max_seconds=budget_seconds,
            author_proxy_context=author_proxy_context,
        )
        raise_if_canceled()

    if (
        not _runtime_budget_exhausted(started_at, budget_seconds)
        and _borderline_verdict_should_run(current_scores=current_scores, density_gate=_density_gate_for_report(current_text, current_report))
    ):
        _emit_progress(progress_callback, 79, "Running V5 borderline texture pass")
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            borderline_verdict_rounds,
            global_best_candidate,
        ) = _run_borderline_verdict_cleanup_pass(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            gateway=gateway,
            output_dir=out_dir / "borderline_verdict_cleanup",
            global_best_candidate=global_best_candidate,
            variant_count=borderline_verdict_variant_count,
            progress_callback=progress_callback,
            progress_percent=79,
            accepted_checkpoint_callback=record_accepted_checkpoint,
            started_at=started_at,
            max_seconds=budget_seconds,
            author_proxy_context=author_proxy_context,
        )
        raise_if_canceled()

    if (
        safe_band_evidence_repair_enabled
        and not _runtime_budget_exhausted(started_at, budget_seconds)
        and _safe_band_density_first_repair_should_run(current_scores=current_scores, current_goal=current_goal)
    ):
        phase_order["safe_band_evidence_repair"]["density_first_route"] = True
        _emit_progress(progress_callback, 80, "Running V5 density-first safe-band repair")
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            safe_band_evidence_repair_rounds,
            global_best_candidate,
        ) = _run_safe_band_evidence_repair_pass(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            gateway=gateway,
            output_dir=out_dir / "safe_band_evidence_repair",
            global_best_candidate=global_best_candidate,
            progress_callback=progress_callback,
            progress_percent=80,
            accepted_checkpoint_callback=record_accepted_checkpoint,
            started_at=started_at,
            max_seconds=budget_seconds,
            author_proxy_context=author_proxy_context,
        )
        raise_if_canceled()

    if (
        final_topk_sentence_route_enabled
        and not _runtime_budget_exhausted(started_at, budget_seconds)
        and _final_topk_sentence_route_should_run(current_scores=current_scores, density_gate=_density_gate_for_report(current_text, current_report))
    ):
        _emit_progress(progress_callback, 80, "Running V5 final top-k sentence route resolver")
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            final_topk_sentence_route_rounds,
            global_best_candidate,
        ) = _run_final_topk_sentence_route_pass(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            gateway=gateway,
            output_dir=out_dir / "final_topk_sentence_route",
            global_best_candidate=global_best_candidate,
            progress_callback=progress_callback,
            progress_percent=80,
            accepted_checkpoint_callback=record_accepted_checkpoint,
            started_at=started_at,
            max_seconds=budget_seconds,
            author_proxy_context=author_proxy_context,
        )
        raise_if_canceled()

    if (
        safe_band_evidence_repair_enabled
        and not safe_band_evidence_repair_rounds
        and not _runtime_budget_exhausted(started_at, budget_seconds)
        and _safe_band_evidence_repair_should_run(current_scores=current_scores, current_goal=current_goal)
    ):
        _emit_progress(progress_callback, 81, "Running V5 safe-band evidence repair")
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            safe_band_evidence_repair_rounds,
            global_best_candidate,
        ) = _run_safe_band_evidence_repair_pass(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            gateway=gateway,
            output_dir=out_dir / "safe_band_evidence_repair",
            global_best_candidate=global_best_candidate,
            progress_callback=progress_callback,
            progress_percent=81,
            accepted_checkpoint_callback=record_accepted_checkpoint,
            started_at=started_at,
            max_seconds=budget_seconds,
            author_proxy_context=author_proxy_context,
        )
        raise_if_canceled()

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
        record_accepted_checkpoint({
            "phase": "global_best_fallback",
            "round": None,
            "reason": "best_full_document_candidate_superseded_phase_accepted_result",
            "accepted": global_best_candidate,
            "rewritten_document": current_text,
            "scores": current_scores,
            "goal": current_goal,
        })
        global_best_fallback = {
            "applied": True,
            "reason": "best_full_document_candidate_superseded_phase_accepted_result",
            "selected": _compact_residual_row(global_best_candidate),
            "previous_final_scores": previous_scores,
            "final_scores": current_scores,
        }

    density_gate = _density_gate_for_report(current_text, current_report)
    candidate_ledger = _residual_candidate_ledger(
        seed_candidate_rows=seed_candidate_rows,
        global_best_candidate=global_best_candidate,
        final_text=current_text,
        final_scores=current_scores,
        final_goal=current_goal,
    )
    payload = {
        "stage": "v5_residual_cluster_comb",
        "baseline_scores": baseline_scores,
        "direct_scanner_leapfrog_rounds": direct_scanner_rounds,
        "rounds": rounds,
        "risky_window_cleanup_rounds": risky_window_rounds,
        "unsafe_cluster_cleanup_rounds": unsafe_cluster_rounds,
        "final_risky_window_cleanup_rounds": final_risky_window_rounds,
        "borderline_verdict_cleanup_rounds": borderline_verdict_rounds,
        "final_topk_sentence_route_rounds": final_topk_sentence_route_rounds,
        "safe_band_evidence_repair_rounds": safe_band_evidence_repair_rounds,
        "seed_candidate_rows": [_compact_residual_row(row) for row in seed_candidate_rows],
        "seed_recovery": seed_recovery,
        "phase_order": phase_order,
        "accepted_checkpoints": accepted_checkpoints,
        "global_best_fallback": global_best_fallback,
        "final_scores": current_scores,
        "eligible_span_density_gate": density_gate,
        "candidate_ledger": candidate_ledger,
        "adaptive_cutoff": {
            "enabled": _adaptive_cutoff_enabled(),
            "events": adaptive_cutoff_events,
            "final_blocker_state": _adaptive_cutoff_blocker_state(current_scores, density_gate),
        },
        "runtime_budget": _runtime_budget_payload(started_at, budget_seconds),
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


def _accepted_checkpoint_payload(
    *,
    event: dict[str, Any],
    sequence: int,
    stage: str,
    baseline_scores: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    accepted = event.get("accepted") if isinstance(event.get("accepted"), dict) else {}
    rewritten_document = str(event.get("rewritten_document") or accepted.get("candidate_text") or "")
    checkpoint = {
        "schema_version": "rewrite_v5_accepted_checkpoint.v1",
        "stage": stage,
        "sequence": max(1, int(sequence or 1)),
        "phase": event.get("phase"),
        "round": event.get("round"),
        "reason": event.get("reason"),
        "created_at_epoch": round(time.time(), 3),
        "baseline_scores": baseline_scores,
        "scores": event.get("scores") if isinstance(event.get("scores"), dict) else accepted.get("scores"),
        "goal": event.get("goal") if isinstance(event.get("goal"), dict) else accepted.get("candidate_goal"),
        "accepted": _compact_residual_row(accepted),
        "rewritten_document": rewritten_document,
    }
    checkpoint_dir = output_dir / "accepted_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    safe_phase = str(checkpoint.get("phase") or "accepted").replace("/", "_")
    filename = f"checkpoint_{checkpoint['sequence']:03d}_{safe_phase}.json"
    (checkpoint_dir / filename).write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2))
    (checkpoint_dir / "latest_checkpoint.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2))
    (checkpoint_dir / "latest_rewritten.txt").write_text(rewritten_document)
    return checkpoint


def _compact_accepted_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": checkpoint.get("schema_version"),
        "stage": checkpoint.get("stage"),
        "sequence": checkpoint.get("sequence"),
        "phase": checkpoint.get("phase"),
        "round": checkpoint.get("round"),
        "reason": checkpoint.get("reason"),
        "created_at_epoch": checkpoint.get("created_at_epoch"),
        "scores": checkpoint.get("scores"),
        "goal": checkpoint.get("goal"),
        "accepted": checkpoint.get("accepted"),
        "rewritten_word_count": word_count(str(checkpoint.get("rewritten_document") or "")),
    }


def _residual_candidate_ledger_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    goal = row.get("goal") if isinstance(row.get("goal"), dict) else {}
    strict = 0.0 if goal.get("goal_met") is True or goal.get("strict_ai_safe_band_achieved") is True else 1.0
    return (
        strict,
        _number(scores.get("topk_calibrated_risk") or 999.0),
        _number(scores.get("qualifying_text_ai_density") or 999.0),
        _number(scores.get("ai") or 999.0),
        _number(scores.get("unsafe_cluster_count") or 999.0),
    )


def _residual_candidate_ledger_text(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("candidate_text") or row.get("text") or row.get("rewritten_document") or "").strip()


def _residual_candidate_ledger_entry(source: str, row: dict[str, Any], text: str) -> dict[str, Any]:
    goal = row.get("candidate_goal") if isinstance(row.get("candidate_goal"), dict) else row.get("goal")
    goal = goal if isinstance(goal, dict) else {}
    return {
        "schema_version": "rewrite_candidate_ledger.v1",
        "source": source,
        "section_id": row.get("section_id"),
        "variant_id": row.get("variant_id"),
        "label": row.get("label"),
        "word_count": row.get("word_count") or word_count(text),
        "scores": row.get("scores") if isinstance(row.get("scores"), dict) else {},
        "goal": {
            "status": goal.get("status"),
            "goal_met": goal.get("goal_met"),
            "reason": goal.get("reason"),
            "strict_ai_safe_band_achieved": goal.get("strict_ai_safe_band_achieved"),
        },
        "text": text,
    }


def _residual_candidate_ledger(
    *,
    seed_candidate_rows: list[dict[str, Any]],
    global_best_candidate: dict[str, Any] | None,
    final_text: str,
    final_scores: dict[str, Any],
    final_goal: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(source: str, row: dict[str, Any] | None, *, fallback_text: str = "") -> None:
        if not isinstance(row, dict):
            return
        text = _residual_candidate_ledger_text(row) or str(fallback_text or "").strip()
        normalized = " ".join(text.split())
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        rows.append(_residual_candidate_ledger_entry(source, row, text))

    for row in seed_candidate_rows or []:
        add("historical_seed_candidate", row)
    add("global_best_candidate", global_best_candidate)
    add(
        "final_text",
        {
            "section_id": "full_document",
            "variant_id": "final",
            "label": "final_text",
            "word_count": word_count(final_text),
            "scores": final_scores,
            "goal": final_goal,
            "text": final_text,
        },
    )
    rows.sort(key=_residual_candidate_ledger_sort_key)
    for index, row in enumerate(rows[:5], start=1):
        row["rank"] = index
    return rows[:5]


def _should_start_with_unsafe_cluster_cleanup(
    *,
    density_gate: dict[str, Any],
    unsafe_cluster_cleanup_rounds: int,
) -> bool:
    if int(unsafe_cluster_cleanup_rounds or 0) <= 0:
        return False
    if not isinstance(density_gate, dict):
        return False
    return density_gate.get("safe") is False


def _unsafe_cluster_probe_round_limit(total_rounds: int) -> int:
    total = max(0, min(12, int(total_rounds or 0)))
    if total <= 0:
        return 0
    configured = os.environ.get("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_PROBE_ROUNDS")
    if configured is not None:
        try:
            return max(0, min(total, int(configured or 0)))
        except (TypeError, ValueError):
            pass
    share = _float_env(
        "DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_PROBE_SHARE",
        0.25,
        minimum=0.0,
        maximum=1.0,
    )
    if share <= 0:
        return 0
    return max(1, min(total, round(total * share)))


def _unsafe_cluster_cleanup_stop_after_misses() -> int:
    return _int_env(
        "DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_STOP_AFTER_MISSES",
        3,
        minimum=0,
        maximum=12,
    )


def _runtime_budget_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def _adaptive_cutoff_enabled() -> bool:
    return _bool_env("DRAFTPROOF_REWRITE_V5_ADAPTIVE_CUTOFF", True)


def _adaptive_cutoff_runtime_budget_seconds(
    *,
    original_text: str,
    baseline_density_gate: dict[str, Any],
    baseline_scores: dict[str, Any],
) -> float | None:
    if not _adaptive_cutoff_enabled():
        return None
    if not _bool_env("DRAFTPROOF_REWRITE_V5_ADAPTIVE_RUNTIME_BUDGET", True):
        return None
    words = max(1, int(word_count(str(original_text or ""))))
    state = _adaptive_cutoff_blocker_state(baseline_scores, baseline_density_gate)
    pressure = (
        _number(state.get("risky_window_over_limit"))
        + _number(state.get("unsafe_cluster_over_limit"))
        + (_number(state.get("unsafe_word_ratio_over_limit")) / 10.0)
    )
    base_seconds = _float_env(
        "DRAFTPROOF_REWRITE_V5_CUTOFF_BASE_SECONDS",
        75.0,
        minimum=0.0,
        maximum=900.0,
    )
    seconds_per_100_words = _float_env(
        "DRAFTPROOF_REWRITE_V5_CUTOFF_SECONDS_PER_100_WORDS",
        30.0,
        minimum=0.0,
        maximum=300.0,
    )
    seconds_per_blocker = _float_env(
        "DRAFTPROOF_REWRITE_V5_CUTOFF_SECONDS_PER_BLOCKER",
        20.0,
        minimum=0.0,
        maximum=180.0,
    )
    min_seconds = _float_env(
        "DRAFTPROOF_REWRITE_V5_CUTOFF_MIN_SECONDS",
        180.0,
        minimum=30.0,
        maximum=1800.0,
    )
    max_seconds = _float_env(
        "DRAFTPROOF_REWRITE_V5_CUTOFF_MAX_SECONDS",
        720.0,
        minimum=60.0,
        maximum=3600.0,
    )
    computed = base_seconds + (words / 100.0 * seconds_per_100_words) + (pressure * seconds_per_blocker)
    if _borderline_verdict_cleanup_enabled():
        computed += _borderline_verdict_pass_budget_seconds()
    return max(min_seconds, min(max_seconds, computed))


def _adaptive_cutoff_blocker_state(
    current_scores: dict[str, Any],
    density_gate: dict[str, Any],
) -> dict[str, Any]:
    scores = current_scores if isinstance(current_scores, dict) else {}
    density = density_gate if isinstance(density_gate, dict) else {}
    risky_limit = _float_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_RISKY_WINDOWS",
        0.0,
        minimum=0.0,
        maximum=50.0,
    )
    unsafe_cluster_limit = _density_gate_threshold(
        density,
        "max_unsafe_cluster_count",
        env_name="DRAFTPROOF_REWRITE_V5_DEFAULT_SAFE_UNSAFE_CLUSTERS",
        default=4.0,
        maximum=200.0,
    )
    unsafe_word_ratio_limit = _density_gate_threshold(
        density,
        "max_unsafe_eligible_word_ratio",
        env_name="DRAFTPROOF_REWRITE_V5_DEFAULT_SAFE_UNSAFE_WORD_RATIO",
        default=35.0,
        maximum=100.0,
    )
    risky_windows = _number(scores.get("risky_window_count"))
    unsafe_clusters = _number(density.get("unsafe_cluster_count"))
    if unsafe_clusters <= 0 and "unsafe_cluster_count" not in density:
        unsafe_clusters = _number(scores.get("unsafe_cluster_count"))
    unsafe_word_ratio = _number(density.get("unsafe_eligible_word_ratio"))
    if unsafe_word_ratio <= 0 and "unsafe_eligible_word_ratio" not in density:
        unsafe_word_ratio = _number(scores.get("unsafe_word_ratio"))
    risky_over = max(0.0, risky_windows - risky_limit)
    unsafe_cluster_over = max(0.0, unsafe_clusters - unsafe_cluster_limit)
    unsafe_word_ratio_over = max(0.0, unsafe_word_ratio - unsafe_word_ratio_limit)
    unsafe_density_safe = density.get("safe") is True or (
        unsafe_cluster_over <= 0.0
        and unsafe_word_ratio_over <= 0.0
    )
    return {
        "safe": risky_over <= 0.0 and unsafe_density_safe,
        "risky_window_count": risky_windows,
        "risky_window_limit": risky_limit,
        "risky_window_over_limit": risky_over,
        "unsafe_cluster_count": unsafe_clusters,
        "unsafe_cluster_limit": unsafe_cluster_limit,
        "unsafe_cluster_over_limit": unsafe_cluster_over,
        "unsafe_word_ratio": unsafe_word_ratio,
        "unsafe_word_ratio_limit": unsafe_word_ratio_limit,
        "unsafe_word_ratio_over_limit": unsafe_word_ratio_over,
        "unsafe_density_safe": unsafe_density_safe,
        "density_gate_safe": density.get("safe"),
    }


def _adaptive_cutoff_stop_event(
    *,
    phase: str,
    current_scores: dict[str, Any],
    density_gate: dict[str, Any],
) -> dict[str, Any] | None:
    if not _adaptive_cutoff_enabled():
        return None
    state = _adaptive_cutoff_blocker_state(current_scores, density_gate)
    if not state.get("safe"):
        return None
    return {
        "phase": phase,
        "reason": "scanner_blockers_safe",
        "blocker_state": state,
        "density_gate": _compact_density_gate(density_gate),
    }


def _borderline_verdict_cleanup_enabled() -> bool:
    return _bool_env("DRAFTPROOF_REWRITE_V5_BORDERLINE_VERDICT_CLEANUP", True)


def _borderline_verdict_variant_count(default_count: int | None = None) -> int:
    configured = os.environ.get("DRAFTPROOF_REWRITE_V5_BORDERLINE_VERDICT_VARIANTS")
    if configured is not None:
        return _int_env("DRAFTPROOF_REWRITE_V5_BORDERLINE_VERDICT_VARIANTS", 2, minimum=1, maximum=5)
    return max(1, min(5, int(default_count or 2)))


def _borderline_verdict_pass_budget_seconds() -> float:
    return _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_VERDICT_BUDGET_SECONDS",
        180.0,
        minimum=0.0,
        maximum=600.0,
    )


def _borderline_verdict_max_rounds() -> int:
    return _int_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_VERDICT_MAX_ROUNDS",
        2,
        minimum=1,
        maximum=3,
    )


def _borderline_min_word_ratio() -> float:
    return _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_MIN_WORD_RATIO",
        0.85,
        minimum=0.5,
        maximum=1.0,
    )


def _borderline_max_word_ratio() -> float:
    return _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_MAX_WORD_RATIO",
        1.35,
        minimum=1.0,
        maximum=2.0,
    )


def _author_proxy_document_window_ratio() -> float:
    return _float_env(
        "DRAFTPROOF_AUTHOR_PROXY_DOCUMENT_WINDOW_RATIO",
        0.75,
        minimum=0.25,
        maximum=1.0,
    )


def _author_proxy_document_min_word_ratio() -> float:
    return _float_env(
        "DRAFTPROOF_AUTHOR_PROXY_DOCUMENT_MIN_WORD_RATIO",
        0.90,
        minimum=0.5,
        maximum=1.0,
    )


def _borderline_verdict_should_run(
    *,
    current_scores: dict[str, Any],
    density_gate: dict[str, Any],
) -> bool:
    if not _borderline_verdict_cleanup_enabled():
        return False
    if not isinstance(density_gate, dict) or density_gate.get("safe") is not True:
        return False
    state = _adaptive_cutoff_blocker_state(current_scores, density_gate)
    if not state.get("safe"):
        return False
    ai_threshold = _float_env("DRAFTPROOF_REWRITE_V5_BORDERLINE_AI_THRESHOLD", 45.0, minimum=0.0, maximum=100.0)
    authorship_threshold = _float_env("DRAFTPROOF_REWRITE_V5_BORDERLINE_AUTHORSHIP_THRESHOLD", 45.0, minimum=0.0, maximum=100.0)
    external_threshold = _float_env("DRAFTPROOF_REWRITE_V5_BORDERLINE_EXTERNAL_THRESHOLD", 40.0, minimum=0.0, maximum=100.0)
    density_threshold = _float_env("DRAFTPROOF_REWRITE_V5_BORDERLINE_QUALIFYING_DENSITY_THRESHOLD", 70.0, minimum=0.0, maximum=100.0)
    return any(
        _number(current_scores.get(key)) >= threshold
        for key, threshold in (
            ("ai", ai_threshold),
            ("ai_authorship", authorship_threshold),
            ("external", external_threshold),
            ("qualifying_text_ai_density", density_threshold),
        )
    )


def _borderline_sentence_pressure(density_gate: dict[str, Any]) -> list[dict[str, Any]]:
    density = density_gate if isinstance(density_gate, dict) else {}
    targets = density.get("top_sentence_targets") if isinstance(density.get("top_sentence_targets"), list) else []
    rows: list[dict[str, Any]] = []
    for row in targets:
        if not isinstance(row, dict):
            continue
        preview = str(row.get("preview") or "").strip()
        if not preview:
            continue
        rows.append({
            "sentence_id": row.get("sentence_id"),
            "preview": preview[:280],
            "word_count": row.get("word_count"),
            "generic_hits": row.get("generic_hits"),
            "top10_ratio": row.get("top10_ratio"),
            "top50_ratio": row.get("top50_ratio"),
            "predictability_risk": row.get("predictability_risk"),
        })
        if len(rows) >= 8:
            break
    return rows


def _borderline_verdict_target_outcome() -> dict[str, Any]:
    return {
        "preferred_ai_below": _float_env(
            "DRAFTPROOF_REWRITE_V5_BORDERLINE_ACCEPT_AI_BELOW",
            45.0,
            minimum=0.0,
            maximum=100.0,
        ),
        "preferred_authorship_below": _float_env(
            "DRAFTPROOF_REWRITE_V5_BORDERLINE_ACCEPT_AUTHORSHIP_BELOW",
            45.0,
            minimum=0.0,
            maximum=100.0,
        ),
        "max_risky_windows_after": _float_env(
            "DRAFTPROOF_REWRITE_V5_BORDERLINE_MAX_RISKY_WINDOWS_AFTER",
            3.0,
            minimum=0.0,
            maximum=20.0,
        ),
        "max_unsafe_clusters_after": _float_env(
            "DRAFTPROOF_REWRITE_V5_BORDERLINE_MAX_UNSAFE_CLUSTERS_AFTER",
            5.0,
            minimum=0.0,
            maximum=50.0,
        ),
        "max_unsafe_word_ratio_after": _float_env(
            "DRAFTPROOF_REWRITE_V5_BORDERLINE_MAX_UNSAFE_WORD_RATIO_AFTER",
            35.0,
            minimum=0.0,
            maximum=100.0,
        ),
    }


def _borderline_writer_variant_plan(
    *,
    variant_count: int,
    target_outcome: dict[str, Any],
) -> list[dict[str, Any]]:
    templates = [
        {
            "variant_id": "v1",
            "lane_goal": "plain_source_near_route",
            "must_change": [
                "replace abstract bridge phrases with simpler source-level wording",
                "make the old-model to current-learning transition less polished",
                "keep concrete school terms visible",
            ],
        },
        {
            "variant_id": "v2",
            "lane_goal": "sentence_pressure_route_break",
            "must_change": [
                "rewrite the remaining high-pressure sentences by changing their starting logic",
                "break broad claim to explanation patterns",
                "use shorter connective sentences where the current bridge is too smooth",
            ],
        },
        {
            "variant_id": "v3",
            "lane_goal": "concrete_relation_reframe",
            "must_change": [
                "turn broad institutional labels into concrete relations between students, teachers, tools, and assessment",
                "avoid replacing simple words with formal academic wording",
                "keep the argument intact while making the relation more direct",
            ],
        },
        {
            "variant_id": "v4",
            "lane_goal": "list_and_rhythm_break",
            "must_change": [
                "break list-heavy sentence rhythm without dropping listed ideas",
                "vary sentence weight across the document",
                "avoid stacked categories that read like a summary template",
            ],
        },
        {
            "variant_id": "v5",
            "lane_goal": "decisive_boundary_push",
            "must_change": [
                "make a stronger whole-document texture change than mild copyediting",
                "touch several paragraph bridge sentences while preserving paragraph order",
                "prefer plain explanatory movement over polished conclusion wording",
            ],
        },
    ]
    variants = max(1, min(5, int(variant_count or 1)))
    selected = [dict(row) for row in templates[:variants]]
    for index, row in enumerate(selected, start=1):
        row["variant_id"] = f"v{index}"
        row["target_outcome"] = target_outcome
        row["must_preserve"] = [
            "same paragraph count",
            "same central argument",
            "no new personal story or named factual claim",
            "no compression below the minimum word policy",
        ]
    return selected


def _paragraph_count(text: str) -> int:
    return max(1, sum(1 for block in str(text or "").split("\n\n") if block.strip()))


def _density_gate_threshold(
    density_gate: dict[str, Any],
    key: str,
    *,
    env_name: str,
    default: float,
    maximum: float,
) -> float:
    density = density_gate if isinstance(density_gate, dict) else {}
    thresholds = density.get("thresholds") if isinstance(density.get("thresholds"), dict) else {}
    if key in thresholds:
        value = _number(thresholds.get(key))
        if value > 0:
            return value
    return _float_env(env_name, default, minimum=0.0, maximum=maximum)


def _planner_gateway(
    *,
    fallback_gateway: LLMGateway,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
    provider: dict[str, Any] | None,
    extra_body: dict[str, Any] | None,
    cancellation_check: Callable[[], None] | None = None,
) -> LLMGateway:
    requested_model = (
        str(model or "").strip()
        or str(os.environ.get("DRAFTPROOF_REWRITE_V5_PLANNER_MODEL") or "").strip()
        or str(os.environ.get("DRAFTPROOF_REWRITE_V5_NORMALIZER_MODEL") or "").strip()
    )
    if not requested_model or requested_model == str(getattr(fallback_gateway, "model", "") or ""):
        return fallback_gateway
    return LLMGateway(LLMConfig(
        api_key=api_key,
        model=requested_model,
        base_url=base_url,
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V5_ROUTE_PLAN_MAX_TOKENS", 2600, minimum=800, maximum=6000),
        temperature=_float_env("DRAFTPROOF_REWRITE_V5_ROUTE_PLAN_TEMPERATURE", 0.12, minimum=0.0, maximum=0.8),
        top_p=_float_env("DRAFTPROOF_REWRITE_V5_ROUTE_PLAN_TOP_P", 0.72, minimum=0.1, maximum=1.0),
        provider=provider,
        timeout=180,
        extra_body=extra_body,
        cancellation_check=cancellation_check,
    ))


def _runtime_elapsed_seconds(started_at: float) -> float:
    return max(0.0, time.monotonic() - float(started_at))


def _runtime_budget_exhausted(started_at: float, max_seconds: float | None) -> bool:
    return max_seconds is not None and _runtime_elapsed_seconds(started_at) >= float(max_seconds)


def _runtime_budget_payload(started_at: float, max_seconds: float | None) -> dict[str, Any]:
    elapsed = _runtime_elapsed_seconds(started_at)
    remaining = None if max_seconds is None else max(0.0, float(max_seconds) - elapsed)
    return {
        "enabled": max_seconds is not None,
        "max_seconds": round(float(max_seconds), 3) if max_seconds is not None else None,
        "elapsed_seconds": round(elapsed, 3),
        "remaining_seconds": round(remaining, 3) if remaining is not None else None,
        "exhausted": max_seconds is not None and remaining == 0.0,
    }


def _runtime_budget_stop_record(
    *,
    phase: str,
    round_index: int,
    started_at: float,
    max_seconds: float | None,
    current_scores: dict[str, Any],
) -> dict[str, Any]:
    return {
        "round": round_index,
        "phase": phase,
        "status": "stopped",
        "reason": "runtime_budget_exhausted",
        "runtime_budget": _runtime_budget_payload(started_at, max_seconds),
        "current_scores": current_scores,
    }


def _emit_progress(callback: Callable[[int, str], None] | None, percent: int, message: str) -> None:
    if callback is None:
        return
    callback(max(0, min(99, int(percent))), str(message or ""))


def _residual_progress_percent(round_index: int, *, max_rounds: int) -> int:
    total = max(1, int(max_rounds or 1))
    index = max(1, int(round_index or 1))
    return max(67, min(75, 66 + round((min(index, total) / total) * 8)))


def _author_proxy_active(context: dict[str, Any] | None) -> bool:
    return isinstance(context, dict) and bool(context.get("active"))


def _author_proxy_review_items(context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not _author_proxy_active(context):
        return []
    cards = context.get("review_cards") if isinstance(context, dict) else []
    if not isinstance(cards, list):
        return []
    items: list[dict[str, Any]] = []
    for index, card in enumerate(cards[:12], start=1):
        if not isinstance(card, dict):
            continue
        items.append({
            "item_id": str(card.get("card_id") or f"author-review-{index:02d}"),
            "provenance": card.get("provenance") or "needs_author_confirmation",
            "target_text": card.get("target_text"),
            "user_input_needed": card.get("user_input_needed"),
        })
    return items


def _author_proxy_item_list(value: Any, *, limit: int = 8) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, item in enumerate(value if isinstance(value, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        compact = {
            "item_id": str(item.get("item_id") or item.get("card_id") or f"author-item-{index:02d}")[:80],
            "provenance": str(item.get("provenance") or "needs_author_confirmation")[:80],
            "target_text": str(item.get("target_text") or "")[:500],
            "generated_text": str(item.get("generated_text") or item.get("text") or "")[:500],
            "user_input_needed": str(item.get("user_input_needed") or "")[:300],
            "author_task": str(item.get("author_task") or "")[:300],
        }
        if compact["target_text"] or compact["generated_text"] or compact["user_input_needed"]:
            items.append(compact)
        if len(items) >= limit:
            break
    return items


def _author_proxy_output_variant_template() -> dict[str, Any]:
    return {
        "variant_id": "v1",
        "text": "...",
        "author_proxy_provenance": [
            {
                "item_id": "p001",
                "provenance": "source_preserved | inferred_from_draft | needs_author_confirmation | must_replace",
                "target_text": "source phrase or gap being resolved",
                "generated_text": "candidate wording tied to that source/gap",
                "user_input_needed": "empty when no author input is needed",
                "author_task": "verify, replace, or remove when author input is needed",
            }
        ],
        "author_review_items": [
            {
                "item_id": "r001",
                "provenance": "needs_author_confirmation",
                "target_text": "exact claim or detail in the candidate",
                "generated_text": "candidate wording needing author confirmation",
                "user_input_needed": "specific author-owned detail needed",
                "author_task": "what the author should check before submission",
            }
        ],
    }


def _prompt_author_proxy_active(prompt: str) -> bool:
    prefix = "Return valid JSON only.\n"
    if not str(prompt or "").startswith(prefix):
        return False
    try:
        payload = json.loads(str(prompt)[len(prefix):])
    except json.JSONDecodeError:
        return False
    context = payload.get("author_proxy_context") if isinstance(payload, dict) else {}
    return isinstance(context, dict) and bool(context.get("review_required") or context.get("mode"))


_REFERENCE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*|\d+(?:[.,]\d+)?%?")


def _reference_tokens(text: str) -> list[str]:
    return [match.group(0).strip("'_-") for match in _REFERENCE_TOKEN_RE.finditer(str(text or ""))]


def _sentence_initial_tokens(text: str) -> set[str]:
    initials: set[str] = set()
    for sentence in re.split(r"(?:^|[.!?]\s+|\n+)", str(text or "")):
        tokens = _reference_tokens(sentence)
        if tokens:
            initials.add(tokens[0])
    return initials


def _concrete_reference_inventory(text: str) -> dict[str, list[str]]:
    tokens = _reference_tokens(text)
    sentence_initials = _sentence_initial_tokens(text)
    numbers: list[str] = []
    named_references: list[str] = []
    for token in tokens:
        if any(char.isdigit() for char in token):
            if token not in numbers:
                numbers.append(token)
            continue
        if len(token) <= 2 or token in sentence_initials:
            continue
        if token[:1].isupper() and any(char.islower() for char in token[1:]):
            if token not in named_references:
                named_references.append(token)
    return {
        "numbers": numbers[:20],
        "named_references": named_references[:20],
    }


def _novel_reference_tokens(source_text: str, candidate_text: str) -> dict[str, list[str]]:
    source = _concrete_reference_inventory(source_text)
    candidate = _concrete_reference_inventory(candidate_text)
    source_numbers = set(source.get("numbers") or [])
    source_names = set(source.get("named_references") or [])
    return {
        "numbers": [token for token in candidate.get("numbers", []) if token not in source_numbers],
        "named_references": [
            token
            for token in candidate.get("named_references", [])
            if token not in source_names
        ],
    }


def _author_proxy_candidate_audit(
    source_text: str,
    candidate_text: str,
    context: dict[str, Any] | None,
    *,
    phase: str,
) -> dict[str, Any]:
    if not _author_proxy_active(context):
        return {}
    review_items = _author_proxy_review_items(context)
    novel_references = _novel_reference_tokens(source_text, candidate_text)
    has_novel_reference = bool(novel_references["numbers"] or novel_references["named_references"])
    review_required = bool(
        context.get("review_required")
        or review_items
        or context.get("required_inputs")
        or has_novel_reference
    )
    if has_novel_reference:
        reason = "candidate_introduced_concrete_references_not_present_in_source"
    elif review_required:
        reason = "author_proxy_review_required"
    else:
        reason = "source_grounded_candidate"
    return {
        "schema_version": "author_proxy_candidate_audit.v1",
        "active": True,
        "phase": phase,
        "review_required": review_required,
        "required_inputs": context.get("required_inputs") or [],
        "review_items": review_items,
        "novel_candidate_references": novel_references,
        "safety_gate": {
            "passed": not has_novel_reference,
            "requires_author_review": review_required,
            "reason": reason,
        },
    }


def _content_token_set(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _reference_tokens(text)
        if len(token) > 2 and not token.isdigit()
    }


def _bounded_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def _author_proxy_quality_score(
    *,
    source_text: str,
    candidate_text: str,
    context: dict[str, Any] | None,
    provenance: list[dict[str, Any]] | None = None,
    review_items: list[dict[str, Any]] | None = None,
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not _author_proxy_active(context):
        return {}
    source_tokens = _content_token_set(source_text)
    candidate_tokens = _content_token_set(candidate_text)
    shared = source_tokens & candidate_tokens
    source_words = max(1, word_count(source_text))
    candidate_words = max(1, word_count(candidate_text))
    length_ratio = min(source_words, candidate_words) / max(source_words, candidate_words)
    support_ratio = _bounded_ratio(len(shared), len(candidate_tokens) or 1)
    coverage_ratio = _bounded_ratio(len(shared), len(source_tokens) or 1)
    sentence_count = max(1, len(_sentences(candidate_text)))
    avg_sentence_words = candidate_words / sentence_count
    ideal_sentence_words = _float_env(
        "DRAFTPROOF_AUTHOR_PROXY_IDEAL_SENTENCE_WORDS",
        22.0,
        minimum=8.0,
        maximum=40.0,
    )
    sentence_shape_window = max(8.0, ideal_sentence_words * 1.25)
    sentence_shape_score = 1.0 - min(abs(avg_sentence_words - ideal_sentence_words) / sentence_shape_window, 1.0)
    placeholder_penalty = 0.25 if re.search(r"\[[^\[\]]+\]|\bTBD\b|<[^>]+>", candidate_text, re.IGNORECASE) else 0.0
    audit_data = audit if isinstance(audit, dict) else {}
    safety_gate = audit_data.get("safety_gate") if isinstance(audit_data.get("safety_gate"), dict) else {}
    novel = audit_data.get("novel_candidate_references") if isinstance(audit_data.get("novel_candidate_references"), dict) else {}
    novel_count = len(novel.get("numbers") or []) + len(novel.get("named_references") or [])
    novel_reference_penalty = min(0.18, novel_count * 0.04)
    provenance_count = len(provenance or [])
    review_count = len(review_items or [])
    provenance_score = 1.0 if provenance_count else 0.45
    review_specificity = min(1.0, review_count / max(1, len(_author_proxy_review_items(context))))
    safety_score = 1.0 if safety_gate.get("passed", True) else 0.75
    score = (
        support_ratio * 0.28
        + coverage_ratio * 0.18
        + length_ratio * 0.16
        + sentence_shape_score * 0.12
        + provenance_score * 0.14
        + review_specificity * 0.06
        + safety_score * 0.06
        - placeholder_penalty
        - novel_reference_penalty
    )
    return {
        "schema_version": "author_proxy_quality.v1",
        "active": True,
        "score": round(max(0.0, min(1.0, score)), 4),
        "basis": "submitted_content_only",
        "source_support_ratio": round(support_ratio, 4),
        "source_coverage_ratio": round(coverage_ratio, 4),
        "length_ratio": round(length_ratio, 4),
        "sentence_shape_score": round(max(0.0, sentence_shape_score), 4),
        "provenance_item_count": provenance_count,
        "review_item_count": review_count,
        "novel_reference_count": novel_count,
        "placeholder_penalty": round(placeholder_penalty, 4),
    }


def _author_proxy_quality_sort_value(row: dict[str, Any]) -> float:
    quality = row.get("author_proxy_quality") if isinstance(row.get("author_proxy_quality"), dict) else {}
    return _number(quality.get("score")) if quality.get("active") else 0.0


def _should_skip_core_after_direct_accept(
    *,
    direct_scanner_accepted_count: int,
    author_proxy_context: dict[str, Any] | None,
) -> bool:
    return (
        int(direct_scanner_accepted_count or 0) > 0
        and _bool_env("DRAFTPROOF_REWRITE_V5_SKIP_CORE_AFTER_DIRECT_SCANNER_ACCEPT", True)
        and not _author_proxy_active(author_proxy_context)
    )


def _author_proxy_evidence_contract(context: dict[str, Any]) -> dict[str, Any]:
    existing = context.get("authorship_evidence_contract")
    if isinstance(existing, dict) and existing.get("schema_version"):
        return existing
    evidence_slots = []
    for card in context.get("review_cards") if isinstance(context.get("review_cards"), list) else []:
        if not isinstance(card, dict):
            continue
        evidence_slots.append({
            "slot_id": card.get("card_id"),
            "kind": card.get("kind"),
            "bucket": card.get("bucket"),
            "target_text": card.get("target_text"),
            "required_input": card.get("user_input_needed"),
            "provenance": card.get("provenance"),
        })
    return {
        "schema_version": "authorship_evidence_contract.v1",
        "basis": "submitted_content_only",
        "required_inputs": context.get("required_inputs") or [],
        "evidence_slots": evidence_slots,
        "rules": [
            "Use submitted draft material as the evidence source of record.",
            "Narrow unsupported claims instead of inventing support.",
            "Keep provisional bridge material visible through review provenance.",
        ],
    }


def _attach_author_proxy_context(payload: dict[str, Any], context: dict[str, Any] | None) -> None:
    if not isinstance(context, dict) or not context.get("active"):
        return
    compact_context = {
        "schema_version": context.get("schema_version"),
        "mode": context.get("mode"),
        "review_required": bool(context.get("review_required")),
        "primary_mode": context.get("primary_mode"),
        "required_inputs": context.get("required_inputs") or [],
        "allowed_provenance": context.get("allowed_provenance") or [],
        "review_cards": (context.get("review_cards") or [])[:8],
        "quality_bar": context.get("quality_bar") or {},
        "authorship_evidence_contract": _author_proxy_evidence_contract(context),
    }
    payload["author_proxy_context"] = compact_context
    payload["author_proxy_rules"] = [
        "Continue the rewrite; do not stop to ask the author questions.",
        "Produce the highest-quality polished candidate possible from the submitted content, not a cautious stub.",
        "You may draft provisional bridging/context only from the submitted draft, nearby context, and existing source/citation material.",
        "Do not invent personal experiences, citations, numbers, dates, named events, institutions, source facts, or classroom details.",
        "If a needed detail is not in the source text, keep the language conditional or narrow the claim instead of fabricating support.",
        "Treat author_proxy_context.review_cards as author-review obligations for the final product.",
        "Do not write bracketed placeholders in the rewritten text; produce a readable candidate that the author can later verify.",
        "Do not compress document-sized windows into summaries; preserve full coverage, paragraph intent, examples, and approximate source length.",
    ]
    payload["author_proxy_quality_contract"] = {
        "target": "highest_quality_grounded_candidate",
        "basis": "Use only submitted source_text, before_context, after_context, source blocks, phrase anchors, event beats, and existing citation/source material.",
        "quality_order": [
            "Preserve the author's meaning, thesis, and scope.",
            "Make the writing more specific by mining concrete wording, events, relationships, and constraints already present in the submitted content.",
            "Improve paragraph logic, transitions, and sentence rhythm so the candidate reads like careful human revision.",
            "Use precise academic wording without generic filler, synonym-swapping, or template-like phrasing.",
            "When a high-value detail is missing, narrow the claim into a polished author-review statement instead of inventing evidence.",
        ],
        "self_check_before_return": [
            "Every variant is complete, readable, and submission-shaped.",
            "Every added detail is either present in or reasonably inferred from the submitted content.",
            "No variant contains placeholders, fake citations, fabricated facts, or vague filler.",
            "The strongest variant should be useful for the author to revise further, even before author confirmation.",
        ],
    }
    output_schema = payload.get("output_schema") if isinstance(payload.get("output_schema"), dict) else {}
    schema_variants = output_schema.get("variants") if isinstance(output_schema.get("variants"), list) else []
    first_variant_schema = schema_variants[0] if schema_variants and isinstance(schema_variants[0], dict) else {}
    uses_text_variant_schema = "text" in first_variant_schema
    payload["author_proxy_candidate_audit_contract"] = {
        "applied_by": "DraftProof controller after each candidate is scored.",
        "model_output_schema": (
            "For each variant, return variant_id, text, author_proxy_provenance, and author_review_items."
            if uses_text_variant_schema
            else "Return the task's variants schema exactly; DraftProof will derive candidate review evidence after applying repairs."
        ),
        "review_required": bool(context.get("review_required")),
        "audit_fields": [
            "required_inputs",
            "review_items",
            "novel_candidate_references",
            "safety_gate",
        ],
        "acceptance_note": "A scanner-accepted candidate remains marked for manual author review when this context is active.",
    }
    payload["provenance_contract"] = {
        "source_preserved": "Exact or directly preserved material already present in the submitted text.",
        "inferred_from_draft": "Low-risk inference from submitted wording or nearby context.",
        "needs_author_confirmation": "Plausible author-proxy drafting that must be checked by the author.",
        "must_replace": "Material that should be replaced with a real author/source detail before submission.",
        "acceptance_note": "Unverified proxy material prevents the candidate from being labelled as final AI mitigation.",
    }
    if uses_text_variant_schema:
        output_schema["variants"] = [_author_proxy_output_variant_template()]
        payload["output_schema"] = output_schema


def _safe_band_kpi_contract(current_scores: dict[str, Any], current_goal: dict[str, Any] | None = None) -> dict[str, Any]:
    goal = current_goal if isinstance(current_goal, dict) else {}
    gate = goal.get("ai_footprint_gate") if isinstance(goal.get("ai_footprint_gate"), dict) else {}
    after = gate.get("after") if isinstance(gate.get("after"), dict) else {}
    remaining = gate.get("remaining_ai_footprint_drivers") if isinstance(gate.get("remaining_ai_footprint_drivers"), list) else []
    blockers = gate.get("texture_blockers") if isinstance(gate.get("texture_blockers"), list) else []
    safe_thresholds = gate.get("safe_band_thresholds") if isinstance(gate.get("safe_band_thresholds"), dict) else {}

    def threshold_for(driver: str, default: float) -> float:
        value = safe_thresholds.get(driver)
        return _number(value) if value is not None else default

    def footprint_value(driver: str) -> float | None:
        for bucket in ("authorship_footprint", "semantic_footprint", "grounding_footprint", "structural_footprint"):
            values = after.get(bucket) if isinstance(after.get(bucket), dict) else {}
            if driver in values:
                return _number(values.get(driver))
        if driver in after:
            return _number(after.get(driver))
        return None

    targets = {
        "topk_calibrated_risk": threshold_for("topk_calibrated_risk", 25.0),
        "qualifying_text_ai_density": threshold_for("qualifying_text_ai_density", 35.0),
    }
    current = {
        "ai": current_scores.get("ai"),
        "topk": current_scores.get("topk"),
        "topk_calibrated_risk": current_scores.get("topk_calibrated_risk"),
        "qualifying_text_ai_density": current_scores.get("qualifying_text_ai_density"),
        "ai_authorship": current_scores.get("ai_authorship"),
        "unsafe_cluster_count": current_scores.get("unsafe_cluster_count"),
        "risky_window_count": current_scores.get("risky_window_count"),
    }
    density_driver_names = (
        "generic_assertion_risk",
        "unsupported_claim_risk",
        "broad_claim_risk",
        "source_grounding_risk",
        "rewrite_smoothness",
        "semantic_uniformity",
        "discourse_regularity",
    )
    secondary_density_drivers = {
        name: value
        for name in density_driver_names
        if (value := footprint_value(name)) is not None
    }
    gaps = {
        key: round(max(0.0, _number(current.get(key)) - _number(limit)), 3)
        for key, limit in targets.items()
    }
    return {
        "schema_version": "rewrite_kpi_contract.v1",
        "objective": "clear_strict_ai_safe_band_with_grounded_author_proxy_revision",
        "current": current,
        "targets": targets,
        "gaps": gaps,
        "secondary_density_drivers": secondary_density_drivers,
        "remaining_ai_footprint_drivers": remaining,
        "texture_blockers": blockers,
        "acceptance_priority": [
            "Prefer candidates that clear every strict safe-band target.",
            "If no candidate clears the band, prefer the largest combined reduction in topk_calibrated_risk and qualifying_text_ai_density without AI/authorship regression.",
            "Do not trade lower top-k for unsupported claims, fabricated evidence, or compressed coverage.",
        ],
    }


def _target_sentence_context(current_text: str, sentence: str) -> dict[str, Any]:
    sentence = str(sentence or "").strip()
    if not sentence:
        return {}
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", current_text or "") if paragraph.strip()]
    for paragraph_index, paragraph in enumerate(paragraphs, start=1):
        if sentence not in paragraph:
            continue
        sentences = _sentences(paragraph)
        sentence_index = next((index for index, item in enumerate(sentences) if item.strip() == sentence), None)
        before = sentences[max(0, sentence_index - 2):sentence_index] if sentence_index is not None else []
        after = sentences[sentence_index + 1:sentence_index + 3] if sentence_index is not None else []
        return {
            "paragraph_index": paragraph_index,
            "paragraph": paragraph[:1800],
            "before_sentences": before,
            "after_sentences": after,
            "context_rule": (
                "Use this paragraph context to rebuild the target sentence route with grounded specificity; "
                "do not introduce facts outside the paragraph or authorship evidence contract."
            ),
        }
    return {}


def build_residual_cluster_prompt(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any] | None = None,
    variant_count: int = 3,
    route_plan: dict[str, Any] | None = None,
    adaptive_feedback: dict[str, Any] | None = None,
    author_proxy_context: dict[str, Any] | None = None,
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
        "writer_style_card": _writer_style_card(section=section, route_plan=plan),
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
        payload["writer_execution_card"] = _writer_execution_card(section=section, route_plan=plan)
        payload["writer_variant_plan"] = _writer_variant_plan(variant_count=variants, route_plan=plan)
        feedback = adaptive_feedback if isinstance(adaptive_feedback, dict) else {}
        if feedback:
            payload["score_feedback"] = feedback
            payload["adaptive_retry_rules"] = _adaptive_retry_rules(feedback)
        payload["method"] = _custom_route_writer_method()
    else:
        payload["custom_route_plan"] = None
        payload["fallback_route_blueprint"] = build_route_blueprint(section=section, local_goal=local_goal or {})
        payload["method"] = _fallback_route_writer_method()
    _attach_author_proxy_context(payload, author_proxy_context)
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_residual_cluster_retune_prompt(
    *,
    section: SectionUnit,
    current_best_text: str,
    local_goal: dict[str, Any],
    variant_count: int = 4,
    route_plan: dict[str, Any] | None = None,
    author_proxy_context: dict[str, Any] | None = None,
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
        "writer_style_card": _writer_style_card(
            section=section,
            route_plan=plan,
            current_best_text=current_best_text,
        ),
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
        payload["writer_execution_card"] = _writer_execution_card(section=section, route_plan=plan)
        payload["writer_variant_plan"] = _writer_variant_plan(variant_count=variants, route_plan=plan)
        payload["method"] = _custom_route_retune_method()
    else:
        payload["custom_route_plan"] = None
        payload["fallback_route_blueprint"] = build_route_blueprint(section=section, local_goal=local_goal or {})
        payload["method"] = _fallback_route_retune_method()
    _attach_author_proxy_context(payload, author_proxy_context)
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_residual_cluster_variants(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any] | None = None,
    gateway: LLMGateway,
    variant_count: int = 3,
    route_plan: dict[str, Any] | None = None,
    adaptive_feedback: dict[str, Any] | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    return _generate_loose_variants_from_builder(
        prompt_builder=lambda count: build_residual_cluster_prompt(
            section=section,
            local_goal=local_goal or {},
            variant_count=count,
            route_plan=route_plan,
            adaptive_feedback=adaptive_feedback,
            author_proxy_context=author_proxy_context,
        ),
        gateway=gateway,
        variant_count=variant_count,
    )


def build_residual_cluster_route_plan_prompt(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any] | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> str:
    affected_content_map = _affected_content_map(section=section, local_goal=local_goal or {})
    payload = {
        "task": "score_causal_cluster_route_plan",
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
        "affected_content_map": affected_content_map,
        "primary_metric_options": [
            "topk_density",
            "unsafe_cluster_count",
            "risky_window_count",
            "ai_likelihood",
            "external_proxy",
            "rank",
            "mixed",
        ],
        "topk_operator_options": sorted(_TOPK_ROUTE_OPERATORS),
        "controlled_expansion_move_options": _CONTROLLED_EXPANSION_MOVES,
        "content_profile_rubrics": _ROUTE_PLAN_CONTENT_PROFILES,
        "cluster_role_options": _ROUTE_PLAN_CLUSTER_ROLES,
        "failure_pattern_options": _ROUTE_PLAN_FAILURE_PATTERNS,
        "route_strategy_options": _ROUTE_PLAN_STRATEGIES,
        "planning_rules": [
            "Act as a prompt planner, not the final writer.",
            "Derive a cluster-specific executable brief from the source text and scanner findings.",
            "Use affected_content_map as the source of truth for which content units carry the problem.",
            "Choose primary_metric from primary_metric_options and make the plan target that metric first.",
            "Do not only summarize scanner findings; bind every planned action to an affected content unit.",
            "When primary_metric is topk_density, diagnose the predictable sentence route in topk_route_diagnosis.",
            "For topk_density, prefer route disruption over paraphrase: CLAUSE_ROUTE_CHANGE first, then LIST_RHYTHM_BREAK, ABSTRACT_TO_PRACTICAL_FRAME, GENERIC_TRANSITION_REMOVAL, SENTENCE_WEIGHT_VARIATION.",
            "For each affected unit, provide operator_stack from topk_operator_options; the first operator must be the main action.",
            "Do not use synonym replacement as a top-k action unless it is incidental to a route change.",
            "First choose content_profile and cluster_role from the supplied options.",
            "Then choose dominant_failure_pattern and route_strategy from the supplied options.",
            "Use the chosen content_profile rubric to design the route; for broad_explanatory_report, do not let old wording block controlled expansion when genericness is the problem.",
            "When the affected units are broad, generic, category-stacked, or compressed, set controlled_expansion.required to true and choose one executable expansion move.",
            "Use the chosen cluster_role to decide what the cluster is supposed to do in the document.",
            "When cluster.source_block_count is greater than 1, the replacement route must cover every source block instead of compressing the cluster into only the opening topic.",
            "Make source_block_plan cover every cluster.source_blocks item.",
            "Make target_sentence_jobs focus on scanner_local_findings.top_sentence_targets and give one executable rewrite job per target.",
            "Make affected_unit_actions cover the affected_content_map rows where is_scanner_target is true.",
            "Each affected_unit_actions row must explain what in that exact unit must change and what would be an insufficient surface edit.",
            "Do not mention scores, scanner names, authorship labels, or risk labels in the plan fields.",
            "Describe failed_route as the current sentence movement problem in plain editorial language.",
            "Describe replacement_route as the new route the writer should follow, using source-supported events and claims.",
            "Make must_change concrete enough that the writer can execute it without seeing fallback rules.",
            "Make must_preserve contain only hard source anchors that must survive; do not turn every source beat into a preservation blocker.",
            "Each must_preserve.source_quote must be copied verbatim from cluster.source_text.",
            "Do not put summaries in must_preserve.source_quote; never write phrases like 'the fact that' unless those words are in cluster.source_text.",
            "Use must_preserve.preserve_as only as a short meaning label for the exact source quote.",
            "If a preservation item cannot be copied exactly from cluster.source_text, omit it.",
            "Make sentence_plan an ordered set of sentence jobs, not labels to copy into the answer.",
            "Use avoid_phrases only for source phrases or polished substitutes that would keep the same weak pattern.",
            "Choose length_target from same_length, slight_expand, or expand.",
            "Use same_length when route can change without added bridging, slight_expand when one bridge is needed, and expand only when compression is the main weakness.",
            "Explain reason_this_should_move_score as a plain cause-effect expectation about route movement, not a score promise.",
            "Any added bridge, specificity, or framing must be relevant to the source topic and consistent with nearby context.",
            "Hard failures are fake personal stories, fake citations, fake statistics, fake dates, fake named events, corrupted output, junk text, or broken meaning.",
            "Preserve cluster.referential_continuity in the replacement route.",
            "If a pronoun is linked to a name in before_context, plan for natural name/pronoun continuity and do not tell the writer to explain the reference parenthetically.",
            "Use plain editorial task language that a writer can execute.",
        ],
        "output_schema": {
            "route_plan": {
                "content_profile": "reflective_practice_academic | broad_explanatory_report | argumentative_explanatory_essay | technical_or_process_explanation | narrative_or_case_reflection | mixed_or_unknown",
                "primary_metric": "topk_density | unsafe_cluster_count | risky_window_count | ai_likelihood | external_proxy | rank | mixed",
                "cluster_role": "background_context | evidence_or_example | reasoning_or_analysis | process_or_method | contrast_or_problem | conclusion_or_synthesis | mixed_section",
                "dominant_failure_pattern": "category_dump | event_summary | claim_chain | process_blur | transition_stack | conclusion_smoothing | mixed",
                "route_strategy": "group_and_bridge | event_first_rebuild | claim_reason_evidence | mechanism_consequence | contrast_then_limit | mixed_route_rebuild",
                "profile_reason": "one sentence explaining why this profile fits the cluster",
                "failed_route": "...",
                "replacement_route": "...",
                "topk_route_diagnosis": {
                    "infected_unit_id": "u001",
                    "current_route": "how the affected sentence currently travels",
                    "predictable_path": "the expected word or phrase path that must be broken",
                    "primary_operator": "CLAUSE_ROUTE_CHANGE | LIST_RHYTHM_BREAK | ABSTRACT_TO_PRACTICAL_FRAME | GENERIC_TRANSITION_REMOVAL | SENTENCE_WEIGHT_VARIATION",
                    "replacement_route": "new sentence route, not a synonym swap",
                    "insufficient_edit": "what would preserve the same top-k path"
                },
                "source_block_plan": [
                    {
                        "block_id": "b01",
                        "current_job": "what this source block does now",
                        "rewrite_job": "what the writer must make this block do",
                        "must_preserve": ["exact hard source material from this block"]
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
                "affected_unit_actions": [
                    {
                        "unit_id": "u001",
                        "affected_text": "exact affected content unit text or exact supported preview",
                        "problem_role": "what this content unit is doing that keeps the pattern",
                        "required_action": "specific route/content action for this unit",
                        "operator_stack": ["CLAUSE_ROUTE_CHANGE"],
                        "must_preserve": ["source words or claims to keep from this unit"],
                        "insufficient_edit": "what would be too superficial to count"
                    }
                ],
                "must_change": ["..."],
                "must_preserve": [
                    {
                        "source_quote": "exact substring copied from cluster.source_text",
                        "preserve_as": "short meaning label",
                    }
                ],
                "controlled_expansion": {
                    "required": True,
                    "move": "none | explanatory_bridge | concrete_framing | scope_limit | practical_consequence | contrast_or_specific_angle",
                    "instruction": "one executable expansion instruction for the writer",
                    "why_needed": "why this expansion should help the affected route"
                },
                "sentence_plan": ["..."],
                "avoid_phrases": ["..."],
                "length_target": "same_length | slight_expand | expand",
                "reason_this_should_move_score": "...",
            }
        },
    }
    _attach_author_proxy_context(payload, author_proxy_context)
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_residual_cluster_route_plan(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any],
    gateway: LLMGateway | None = None,
    planner_gateway: LLMGateway | None = None,
    fallback_gateway: LLMGateway | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any], str, str]:
    primary_gateway = planner_gateway or gateway
    if primary_gateway is None:
        raise ValueError("generate_residual_cluster_route_plan requires a planner gateway")
    plan, diagnostics, prompt, raw = _generate_residual_cluster_route_plan_once(
        section=section,
        local_goal=local_goal,
        gateway=primary_gateway,
        author_proxy_context=author_proxy_context,
    )
    diagnostics = {
        **diagnostics,
        "planner_model_requested": getattr(primary_gateway, "model", None),
        "planner_fallback_used": False,
    }
    if _route_plan_valid(plan):
        return plan, diagnostics, prompt, raw
    if _route_plan_failure_is_truncation(diagnostics):
        scanner_plan = _scanner_derived_route_plan(section=section, local_goal=local_goal)
        if _route_plan_valid(scanner_plan):
            return scanner_plan, _scanner_derived_route_plan_diagnostics(
                diagnostics,
                planner_fallback_used=False,
            ), prompt, raw
    if not _should_retry_route_plan_with_fallback(primary_gateway, fallback_gateway):
        scanner_plan = _scanner_derived_route_plan(section=section, local_goal=local_goal)
        if _route_plan_valid(scanner_plan):
            return scanner_plan, _scanner_derived_route_plan_diagnostics(
                diagnostics,
                planner_fallback_used=False,
            ), prompt, raw
        return plan, diagnostics, prompt, raw

    fallback_plan, fallback_diagnostics, fallback_prompt, fallback_raw = _generate_residual_cluster_route_plan_once(
        section=section,
        local_goal=local_goal,
        gateway=fallback_gateway,
        author_proxy_context=author_proxy_context,
    )
    fallback_diagnostics = {
        **fallback_diagnostics,
        "planner_model_requested": getattr(fallback_gateway, "model", None),
        "planner_fallback_used": True,
        "primary_planner_attempt": _compact_route_plan_attempt(diagnostics),
    }
    if not _route_plan_valid(fallback_plan):
        scanner_plan = _scanner_derived_route_plan(section=section, local_goal=local_goal)
        if _route_plan_valid(scanner_plan):
            return scanner_plan, _scanner_derived_route_plan_diagnostics(
                fallback_diagnostics,
                planner_fallback_used=True,
            ), fallback_prompt, fallback_raw
    return fallback_plan, fallback_diagnostics, fallback_prompt, fallback_raw


def _scanner_derived_route_plan(*, section: SectionUnit, local_goal: dict[str, Any]) -> dict[str, Any] | None:
    source_text = str(section.text or "")
    sentences = _sentences(source_text)
    if not source_text.strip() or not sentences:
        return None
    affected_units = _affected_content_map(section=section, local_goal=local_goal)
    target_units = [row for row in affected_units if row.get("is_scanner_target")] or affected_units[:3]
    if not target_units:
        target_units = [{
            "unit_id": "u001",
            "source_text": sentences[0],
            "preserve_candidates": _source_phrase_anchors(sentences[0])[:5],
        }]
    source_blocks = _source_blocks(source_text)
    anchors = _source_phrase_anchors(source_text)
    preserve_quotes = anchors[:6] or sentences[:2]
    must_preserve = [
        {"source_quote": quote, "preserve_as": "source material"}
        for quote in preserve_quotes
        if quote in source_text
    ][:6]
    if not must_preserve:
        must_preserve = [{"source_quote": sentences[0], "preserve_as": "opening source claim"}]
    source_block_plan = []
    for index, block in enumerate(source_blocks or [{"block_id": "b01", "text": source_text}], start=1):
        block_text = str((block.get("text") or block.get("preview")) if isinstance(block, dict) else "")
        block_anchor = next((quote for quote in preserve_quotes if quote in block_text), "")
        source_block_plan.append({
            "block_id": str(block.get("block_id") or f"b{index:02d}") if isinstance(block, dict) else f"b{index:02d}",
            "current_job": "Carries source material in a predictable report route.",
            "rewrite_job": "Keep this block's source material but change how the sentence route groups and bridges it.",
            "must_preserve": [block_anchor] if block_anchor else [],
        })
        if len(source_block_plan) >= 8:
            break
    target_sentence_jobs = []
    affected_unit_actions = []
    for index, unit in enumerate(target_units[:5], start=1):
        source_preview = str(unit.get("source_text") or unit.get("affected_text") or "").strip()
        if not source_preview:
            continue
        preserve_candidates = [
            item for item in unit.get("preserve_candidates", [])
            if isinstance(item, str) and item in source_text
        ]
        if not preserve_candidates:
            preserve_candidates = [source_preview] if source_preview in source_text else []
        target_sentence_jobs.append({
            "sentence_id": str(unit.get("unit_id") or f"u{index:03d}"),
            "source_preview": source_preview,
            "current_weakness": "The unit follows a predictable explanatory route.",
            "rewrite_job": "Change the sentence route before preserving the same source claim.",
            "avoid_copying": [],
        })
        affected_unit_actions.append({
            "unit_id": str(unit.get("unit_id") or f"u{index:03d}"),
            "affected_text": source_preview,
            "problem_role": "This unit carries the route that needs the strongest movement.",
            "required_action": "Re-route the unit from broad report phrasing into a source-specific sentence path.",
            "operator_stack": ["CLAUSE_ROUTE_CHANGE", "LIST_RHYTHM_BREAK", "SENTENCE_WEIGHT_VARIATION"],
            "must_preserve": preserve_candidates[:4],
            "insufficient_edit": "Changing synonyms while keeping the same opener, list order, or broad claim path.",
        })
    if not target_sentence_jobs or not affected_unit_actions:
        return None
    first_unit = affected_unit_actions[0]
    primary_metric = "topk_density" if _local_top_sentence_targets(local_goal) else "unsafe_cluster_count"
    content_profile = _scanner_derived_content_profile(
        section=section,
        local_goal=local_goal,
        affected_units=affected_units,
    )
    raw_plan = {
        "content_profile": content_profile,
        "primary_metric": primary_metric,
        "cluster_role": "mixed_section" if len(source_blocks) > 1 else "reasoning_or_analysis",
        "dominant_failure_pattern": "category_dump" if len(sentences) >= 4 else "claim_chain",
        "route_strategy": "group_and_bridge",
        "profile_reason": "Derived from source shape and scanner affected units after planner output failed validation.",
        "failed_route": "The current cluster keeps too much predictable report movement across affected units.",
        "replacement_route": "Start from the source subject, group related source beats, then bridge to the next claim without broad smoothing.",
        "topk_route_diagnosis": {
            "infected_unit_id": first_unit["unit_id"],
            "current_route": "predictable report-style claim or list movement",
            "predictable_path": first_unit["affected_text"],
            "primary_operator": "CLAUSE_ROUTE_CHANGE",
            "replacement_route": "source subject/action -> grouped source beat -> narrow bridge",
            "insufficient_edit": "Synonym replacement that keeps the same sentence opener or list rhythm.",
        },
        "source_block_plan": source_block_plan,
        "target_sentence_jobs": target_sentence_jobs,
        "affected_unit_actions": affected_unit_actions,
        "must_change": [
            "Change the route of the affected units rather than smoothing the same wording.",
            "Group related source beats before adding a narrow bridge.",
        ],
        "must_preserve": must_preserve,
        "sentence_plan": [
            "Open from the source subject or action.",
            "Group the strongest source beats.",
            "Use a narrow bridge to the next source claim.",
        ],
        "avoid_phrases": [],
        "length_target": "same_length",
        "reason_this_should_move_score": "The plan attacks the affected sentence route directly instead of asking for a broader paraphrase.",
        "controlled_expansion": _scanner_derived_controlled_expansion(
            content_profile=content_profile,
            primary_metric=primary_metric,
            affected_units=affected_units,
            sentence_count=len(sentences),
        ),
    }
    sanitized = _sanitize_route_plan(raw_plan, source_text=source_text)
    return sanitized if _route_plan_valid(sanitized) else None


def _scanner_derived_route_plan_diagnostics(
    failed_diagnostics: dict[str, Any],
    *,
    planner_fallback_used: bool,
) -> dict[str, Any]:
    return {
        **failed_diagnostics,
        "status": "ok",
        "route_plan_source": "scanner_derived_fallback",
        "deterministic_fallback_used": True,
        "planner_fallback_used": planner_fallback_used,
        "failed_planner_status": failed_diagnostics.get("status"),
        "failed_planner_finish_reason": failed_diagnostics.get("finish_reason"),
        "failed_planner_native_finish_reason": failed_diagnostics.get("native_finish_reason"),
    }


def _scanner_derived_content_profile(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any],
    affected_units: list[dict[str, Any]],
) -> str:
    metadata = section.metadata if isinstance(section.metadata, dict) else {}
    for key in ("content_profile", "document_profile", "scanner_content_profile"):
        profile = _content_profile(metadata.get(key) or local_goal.get(key))
        if profile != "mixed_or_unknown":
            return profile
    targets = _local_top_sentence_targets(local_goal)
    generic_pressure = any(_number(row.get("generic_hits")) > 0 for row in targets)
    block_count = max(1, int(section.paragraph_count or 1))
    sentence_count = len(_sentences(section.text))
    target_count = sum(1 for row in affected_units if row.get("is_scanner_target"))
    if generic_pressure or target_count >= 3 or sentence_count >= 4:
        return "broad_explanatory_report"
    if block_count > 1:
        return "mixed_or_unknown"
    return "mixed_or_unknown"


def _scanner_derived_controlled_expansion(
    *,
    content_profile: str,
    primary_metric: str,
    affected_units: list[dict[str, Any]],
    sentence_count: int,
) -> dict[str, Any]:
    target_count = sum(1 for row in affected_units if row.get("is_scanner_target"))
    required = (
        _content_profile(content_profile) == "broad_explanatory_report"
        and (_primary_metric(primary_metric) in {"topk_density", "unsafe_cluster_count"} or target_count >= 3 or sentence_count >= 4)
    )
    if not required:
        return {"required": False, "move": "none", "instruction": "", "why_needed": ""}
    move = _controlled_expansion_move_for_context(
        content_profile=content_profile,
        primary_metric=primary_metric,
        target_count=target_count,
        sentence_count=sentence_count,
    )
    return {
        "required": True,
        "move": move,
        "instruction": _controlled_expansion_instruction(move),
        "why_needed": "The scanner fallback sees broad affected-unit movement that will not be fixed by preserving the same route.",
    }


def _route_plan_failure_is_truncation(diagnostics: dict[str, Any]) -> bool:
    finish_reason = str(diagnostics.get("finish_reason") or diagnostics.get("native_finish_reason") or "").lower()
    if finish_reason == "length":
        return True
    error = str(diagnostics.get("error") or "").casefold()
    return "unterminated string" in error or "expected" in error and str(diagnostics.get("raw_length") or "")


def _generate_residual_cluster_route_plan_once(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any],
    gateway: LLMGateway,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any], str, str]:
    prompt = build_residual_cluster_route_plan_prompt(
        section=section,
        local_goal=local_goal,
        author_proxy_context=author_proxy_context,
    )
    structured = structured_json_request_options(getattr(gateway, "model", None), _route_plan_response_format())
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured.get("provider"))
    started = time.monotonic()
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a route_plan object.",
        response_format=structured.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_REWRITE_V5_ROUTE_PLAN_TEMPERATURE", 0.12, minimum=0.0, maximum=0.8),
        top_p=_float_env("DRAFTPROOF_REWRITE_V5_ROUTE_PLAN_TOP_P", 0.72, minimum=0.1, maximum=1.0),
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V5_ROUTE_PLAN_MAX_TOKENS", 2600, minimum=800, maximum=6000),
    )
    elapsed = time.monotonic() - started
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
        "elapsed_seconds": round(elapsed, 3),
    }, prompt, raw


def _should_retry_route_plan_with_fallback(
    primary_gateway: LLMGateway,
    fallback_gateway: LLMGateway | None,
) -> bool:
    if fallback_gateway is None:
        return False
    if fallback_gateway is primary_gateway:
        return False
    return str(getattr(fallback_gateway, "model", "") or "") != str(getattr(primary_gateway, "model", "") or "")


def _compact_route_plan_attempt(diagnostics: dict[str, Any]) -> dict[str, Any]:
    usage = diagnostics.get("usage") if isinstance(diagnostics.get("usage"), dict) else {}
    return {
        "status": diagnostics.get("status"),
        "reason": diagnostics.get("reason"),
        "model": diagnostics.get("model"),
        "planner_model_requested": diagnostics.get("planner_model_requested"),
        "provider": diagnostics.get("provider"),
        "structured_output_mode": diagnostics.get("structured_output_mode"),
        "finish_reason": diagnostics.get("finish_reason"),
        "native_finish_reason": diagnostics.get("native_finish_reason"),
        "elapsed_seconds": diagnostics.get("elapsed_seconds"),
        "total_tokens": usage.get("total_tokens"),
        "cost": usage.get("cost"),
    }


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


def _writer_style_card(
    *,
    section: SectionUnit,
    route_plan: dict[str, Any] | None,
    current_best_text: str | None = None,
) -> dict[str, Any]:
    plan = route_plan if _route_plan_valid(route_plan) else {}
    profile = _content_profile(plan.get("content_profile"))
    role = _cluster_role(plan.get("cluster_role"))
    candidate_terms = _non_source_terms(section.text, current_best_text or "")[:8] if current_best_text else []
    texture_rules = [
        "Use plain bachelor-level report or essay wording.",
        "Keep the source's natural vocabulary level when it is already clear.",
        "Make route movement through sentence relation, clause order, grouping, or a narrow bridge.",
        "Do not upgrade simple source wording into formal labels, journal-style phrasing, or professional copywriting voice.",
        "Do not make every sentence equally balanced, equally polished, or driven by the same newly added route word.",
    ]
    if profile == "broad_explanatory_report":
        texture_rules.append("For broad report content, add specificity through concrete framing or a limited bridge, not abstract summary language.")
    elif profile in {"reflective_practice_academic", "narrative_or_case_reflection"}:
        texture_rules.append("For reflective or case content, keep the practical event, action, or observed result ahead of broad interpretation.")
    return {
        "reader_level": "bachelor_degree",
        "content_profile": profile,
        "cluster_role": role,
        "target_texture": [
            "plain report or essay prose",
            "clear but not over-polished",
            "source-level wording with visible route movement",
            "slightly varied sentence weight",
        ],
        "texture_rules": texture_rules,
        "source_tone_anchors": _source_phrase_anchors(section.text)[:8],
        "candidate_terms_to_reduce": candidate_terms,
    }


def _writer_execution_card(*, section: SectionUnit, route_plan: dict[str, Any] | None) -> dict[str, Any]:
    plan = route_plan if _route_plan_valid(route_plan) else {}
    topk = plan.get("topk_route_diagnosis") if isinstance(plan.get("topk_route_diagnosis"), dict) else {}
    actions = plan.get("affected_unit_actions") if isinstance(plan.get("affected_unit_actions"), list) else []
    controlled_expansion = _controlled_expansion_for_writer(plan)
    operator_stack: list[str] = []
    for row in actions:
        if not isinstance(row, dict):
            continue
        for operator in row.get("operator_stack") if isinstance(row.get("operator_stack"), list) else []:
            normalized = _topk_route_operator(operator)
            if normalized not in operator_stack:
                operator_stack.append(normalized)
    primary_operator = _topk_route_operator(topk.get("primary_operator"))
    if primary_operator not in operator_stack:
        operator_stack.insert(0, primary_operator)
    do_not_do = _string_list(plan.get("avoid_phrases"), limit=6)
    if topk.get("insufficient_edit"):
        do_not_do.insert(0, str(topk.get("insufficient_edit")))
    unit_actions = []
    for row in actions[:5]:
        if not isinstance(row, dict):
            continue
        unit_actions.append({
            "unit_id": row.get("unit_id"),
            "affected_text": row.get("affected_text"),
            "required_action": row.get("required_action"),
            "operator_stack": row.get("operator_stack") or [primary_operator],
            "insufficient_edit": row.get("insufficient_edit"),
        })
    return {
        "source_section_id": section.section_id,
        "primary_metric": plan.get("primary_metric") or "mixed",
        "main_operator": primary_operator,
        "operator_stack": operator_stack[:5],
        "operator_execution_notes": _operator_execution_notes(operator_stack[:5]),
        "controlled_expansion": controlled_expansion,
        "style_card": _writer_style_card(section=section, route_plan=plan),
        "hard_failures": _HARD_WRITER_FAILURES,
        "route_to_break": topk.get("predictable_path") or plan.get("failed_route"),
        "route_to_write": topk.get("replacement_route") or plan.get("replacement_route"),
        "do_not_do": do_not_do[:8],
        "unit_actions": unit_actions,
        "sentence_plan": _string_list(plan.get("sentence_plan"), limit=6),
        "must_preserve": [
            row.get("source_quote")
            for row in plan.get("must_preserve", [])
            if isinstance(row, dict) and row.get("source_quote")
        ][:8],
    }


def _operator_execution_notes(operators: list[str]) -> list[dict[str, str]]:
    notes = []
    for operator in operators:
        notes.append({
            "operator": _topk_route_operator(operator),
            "writer_action": _TOPK_ROUTE_OPERATOR_WRITER_ACTIONS.get(
                _topk_route_operator(operator),
                _TOPK_ROUTE_OPERATOR_WRITER_ACTIONS["CLAUSE_ROUTE_CHANGE"],
            ),
        })
    return notes


def _writer_variant_plan(*, variant_count: int, route_plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    count = max(1, min(5, int(variant_count or 1)))
    plan = route_plan if _route_plan_valid(route_plan) else {}
    primary_operator = _topk_route_operator((plan.get("topk_route_diagnosis") or {}).get("primary_operator") if isinstance(plan.get("topk_route_diagnosis"), dict) else None)
    controlled_expansion = _controlled_expansion_for_writer(plan)
    operators = [primary_operator]
    for row in plan.get("affected_unit_actions", []) if isinstance(plan.get("affected_unit_actions"), list) else []:
        if not isinstance(row, dict):
            continue
        for item in row.get("operator_stack") if isinstance(row.get("operator_stack"), list) else []:
            operator = _topk_route_operator(item)
            if operator not in operators:
                operators.append(operator)
    fallback_shapes = [
        {
            "route_shape": "main_operator_direct",
            "execution_rule": "Use the main operator directly and make the route change visible before polishing wording.",
        },
        {
            "route_shape": "subject_or_clause_reanchor",
            "execution_rule": "Change the sentence subject or opening clause before preserving the source claim.",
        },
        {
            "route_shape": "bridge_then_example",
            "execution_rule": "Build a clearer bridge from the affected unit to the source example or consequence.",
        },
        {
            "route_shape": "sentence_boundary_shift",
            "execution_rule": "Change one sentence boundary or sentence weight when it helps break the repeated route.",
        },
        {
            "route_shape": "grouped_source_beats",
            "execution_rule": "Group repeated source beats into a cleaner route while preserving distinct claims.",
        },
    ]
    rows = []
    for index in range(count):
        operator = operators[index % len(operators)]
        shape = fallback_shapes[index % len(fallback_shapes)]
        rows.append({
            "variant_id": f"v{index + 1}",
            "main_operator": operator,
            "route_shape": shape["route_shape"],
            "execution_rule": shape["execution_rule"],
            "controlled_expansion_move": (
                controlled_expansion["move"]
                if controlled_expansion.get("required")
                else "none"
            ),
            "controlled_expansion_instruction": (
                controlled_expansion["instruction"]
                if controlled_expansion.get("required")
                else ""
            ),
            "must_differ_from_other_variants": "Use a different opener, clause order, or sentence boundary from the other variants.",
        })
    return rows


def _adaptive_retry_rules(feedback: dict[str, Any]) -> list[str]:
    reason = str(feedback.get("reason") or "")
    rules = [
        "Treat score_feedback as the reason the previous candidate batch failed.",
        "Do not repeat the rejected sentence route, opener pattern, or list rhythm.",
        "Make the next variants visibly execute writer_execution_card.main_operator.",
        "If previous variants sounded over-polished, reduce formal labels and return to writer_style_card.source_tone_anchors.",
        "Preserve the paragraph role and avoid only hard failures: fake personal stories, fake citations, fake statistics, fake dates, fake named events, broken meaning, markup, junk, or corrupted output.",
    ]
    if reason == "topk_route_not_moved":
        rules.insert(1, "The previous batch did not move top-k; every next variant must change the affected sentence route before changing wording.")
    elif reason == "unsafe_cluster_regressed":
        rules.insert(1, "The previous best candidate increased unsafe clusters; reduce broad substitute phrasing and keep the route closer to source content.")
    elif reason == "no_incremental_movement":
        rules.insert(1, "The previous batch changed wording without useful score movement; use a different route shape, not a smoother paraphrase.")
    return rules


def _custom_route_writer_method() -> list[str]:
    return [
        "Treat writer_execution_card as the highest-priority execution summary; it is the compact version of execution_brief.",
        "Treat writer_style_card as the tone boundary for the replacement.",
        "If assigned_writer_variant is present, execute that one lane brief above the full writer_variant_plan.",
        "Follow the assigned route shape so variants are genuinely different route executions, not near-duplicate paraphrases.",
        "Use execution_brief.content_profile and execution_brief.cluster_role to choose the right kind of route movement.",
        "Use execution_brief.primary_metric to understand which scanner movement the rewrite is supposed to cause.",
        "When execution_brief.primary_metric is topk_density, use execution_brief.topk_route_diagnosis to break the predictable next-word path.",
        "For top-k work, the main edit must change sentence route; synonym swaps are insufficient.",
        "For top-k work, directly execute writer_execution_card.route_to_write and avoid writer_execution_card.do_not_do.",
        "Use execution_brief.dominant_failure_pattern and execution_brief.route_strategy to decide what must actually change.",
        "Execute every execution_brief.affected_unit_actions row using its operator_stack; if affected_text is only paraphrased, the rewrite is not enough.",
        "Execute execution_brief.source_block_plan block by block; each block must keep its central source material.",
        "Execute execution_brief.target_sentence_jobs for the risky sentences; do not leave those sentence routes unchanged.",
        "Use coverage_guidance to keep the replacement complete, but do not let coverage copy the old route.",
        "When writer_execution_card.controlled_expansion.required is true, every variant must execute its controlled expansion move.",
        "For broad_explanatory_report, controlled expansion may add explanatory bridges, concrete framing, scope limits, practical consequences, contrasts, or a sharper specific angle.",
        "Write controlled expansion in the same plain source-level vocabulary; do not upgrade it into polished institutional or journal-style phrasing.",
        "Follow execution_brief.replacement_route while rewriting the whole cluster.",
        "Satisfy every execution_brief.must_change item.",
        "Preserve hard anchors from execution_brief.must_preserve, but do not treat ordinary source wording as untouchable.",
        "Follow execution_brief.sentence_plan in order, but do not copy plan labels into the replacement.",
        "Avoid execution_brief.avoid_phrases unless the phrase is a required source term.",
        "Use length_guidance; do not compress the cluster into a summary.",
        "Use writer_style_card.source_tone_anchors where they fit naturally.",
        "Reduce writer_style_card.candidate_terms_to_reduce when retuning a previous candidate.",
        "Do not make every sentence equally polished, equally balanced, or driven by the same newly added route word.",
        "Change remaining_problem_sentences most strongly.",
        "Keep the same paragraph role, point of view, and referential continuity.",
        "Any new connective wording, specificity, or framing must be relevant to the source topic and consistent with nearby context.",
        "Do not trigger writer_execution_card.hard_failures.",
        "Do not return a fragment; the replacement must cover the whole source cluster.",
        "Do not write a plan, label, explanation of the method, or bullet list.",
    ]


def _custom_route_retune_method() -> list[str]:
    return [
        "Treat writer_execution_card as the highest-priority execution summary; it is the compact version of execution_brief.",
        "Treat writer_style_card as the tone boundary for the replacement.",
        "If assigned_writer_variant is present, execute that one lane brief above the full writer_variant_plan.",
        "Follow the assigned route shape so variants are genuinely different route executions, not near-duplicate paraphrases.",
        "Use execution_brief.content_profile and execution_brief.cluster_role to choose the right kind of route movement.",
        "Use execution_brief.primary_metric to understand which scanner movement the rewrite is supposed to cause.",
        "When execution_brief.primary_metric is topk_density, use execution_brief.topk_route_diagnosis to break the predictable next-word path.",
        "For top-k work, the main edit must change sentence route; synonym swaps are insufficient.",
        "For top-k work, directly execute writer_execution_card.route_to_write and avoid writer_execution_card.do_not_do.",
        "Use execution_brief.dominant_failure_pattern and execution_brief.route_strategy to decide what must actually change.",
        "Execute every execution_brief.affected_unit_actions row using its operator_stack; if affected_text is only paraphrased, the rewrite is not enough.",
        "Execute execution_brief.source_block_plan block by block; each block must keep its central source material.",
        "Execute execution_brief.target_sentence_jobs for the risky sentences; do not leave those sentence routes unchanged.",
        "Use coverage_guidance to keep the replacement complete, but do not let coverage copy the old route.",
        "When writer_execution_card.controlled_expansion.required is true, every variant must execute its controlled expansion move.",
        "For broad_explanatory_report, controlled expansion may add explanatory bridges, concrete framing, scope limits, practical consequences, contrasts, or a sharper specific angle.",
        "Write controlled expansion in the same plain source-level vocabulary; do not upgrade it into polished institutional or journal-style phrasing.",
        "Follow execution_brief.replacement_route while rewriting the whole cluster again.",
        "Satisfy every execution_brief.must_change item while focusing on remaining_problem_sentences.",
        "Preserve hard anchors from execution_brief.must_preserve, but do not treat ordinary source wording as untouchable.",
        "Follow execution_brief.sentence_plan in order, but do not copy plan labels into the replacement.",
        "Avoid execution_brief.avoid_phrases unless the phrase is a required source term.",
        "Use retune_focus and candidate_non_source_terms_to_reduce only to clean the current best wording.",
        "Use length_guidance; do not compress the cluster into a summary.",
        "Use writer_style_card.source_tone_anchors where they fit naturally.",
        "Reduce writer_style_card.candidate_terms_to_reduce before adding new phrasing.",
        "Do not make every sentence equally polished, equally balanced, or driven by the same newly added route word.",
        "Keep the same paragraph role, point of view, and referential continuity.",
        "Any new connective wording, specificity, or framing must be relevant to the source topic and consistent with nearby context.",
        "Do not trigger writer_execution_card.hard_failures.",
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
        "Use simple wording where it works, but do not let old wording block required route movement.",
        "Use cluster.source_phrase_anchors where they fit naturally.",
        "Do not upgrade simple source wording into formal domain-theory labels.",
        "Keep the same source subjects, actions, evidence, outcome, and point of view.",
        "Preserve cluster.referential_continuity; do not replace a specific source subject with a generic category label.",
        "If cluster.referential_continuity gives an established name, use the name or the source pronoun naturally; do not write explanatory referent phrases like 'referring to'.",
        "If the source uses I or my, keep that source viewpoint instead of replacing it with a detached narrator.",
        "Any new connective wording, specificity, or framing must be relevant to the source topic and consistent with nearby context.",
        "Reject hard failures: fake personal stories, fake citations, fake statistics, fake dates, fake named events, broken meaning, markup, junk, or corrupted output.",
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
        "Use simple wording where it works, but do not let old wording block required route movement.",
        "Use cluster.source_phrase_anchors where they fit naturally.",
        "Reduce candidate_non_source_terms_to_reduce by replacing them with source wording where possible.",
        "Do not upgrade simple source wording into formal domain-theory labels.",
        "If the source uses I or my, keep that source viewpoint instead of replacing it with a detached narrator.",
        "Preserve cluster.referential_continuity; do not replace a specific source subject with a generic category label.",
        "If cluster.referential_continuity gives an established name, use the name or the source pronoun naturally; do not write explanatory referent phrases like 'referring to'.",
        "Use concrete source action, process, or event wording from the same source context instead of summary wording.",
        "Any new connective wording, specificity, or framing must be relevant to the source topic and consistent with nearby context.",
        "Reject hard failures: fake personal stories, fake citations, fake statistics, fake dates, fake named events, broken meaning, markup, junk, or corrupted output.",
        "Do not return a fragment; the replacement must cover the whole source cluster.",
        "Do not write a plan, label, explanation of the method, or bullet list.",
    ]


def generate_residual_cluster_seed_variants(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any] | None = None,
) -> list[RecompositionVariant]:
    """Generate source-derived route seeds before asking the model.

    Seeds are intentionally generic: they are built from the current section's
    sentence beats and go through the same scanner scoring gates as LLM output.
    """

    del local_goal
    texts = _source_derived_route_seed_texts(section)
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
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    return _generate_loose_variants_from_builder(
        prompt_builder=lambda count: build_residual_cluster_retune_prompt(
            section=section,
            current_best_text=current_best_text,
            local_goal=local_goal,
            variant_count=count,
            route_plan=route_plan,
            author_proxy_context=author_proxy_context,
        ),
        gateway=gateway,
        variant_count=variant_count,
    )


def build_risky_window_cleanup_prompt(
    *,
    section: SectionUnit,
    current_scores: dict[str, Any] | None = None,
    variant_count: int = 5,
    route_plan: dict[str, Any] | None = None,
    author_proxy_context: dict[str, Any] | None = None,
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
            "Any added bridge, specificity, or framing must be relevant to the source topic and consistent with nearby context.",
            "Reject hard failures: fake personal stories, fake citations, fake statistics, fake dates, fake named events, broken meaning, markup, junk, or corrupted output.",
            "Keep the source-supported viewpoint and stance.",
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
            "writer_execution_card": _writer_execution_card(section=section, route_plan=route_plan),
            "writer_style_card": _writer_style_card(section=section, route_plan=route_plan),
            "writer_variant_plan": _writer_variant_plan(variant_count=variants, route_plan=route_plan),
            "source_blocks": _source_blocks(section.text),
            "coverage_guidance": _coverage_guidance_for_route_plan(section=section, route_plan=route_plan),
            "length_guidance": _length_guidance_for_route_plan(section=section, route_plan=route_plan),
            "method": _custom_route_writer_method(),
        })
    _attach_author_proxy_context(payload, author_proxy_context)
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_unsafe_cluster_cleanup_prompt(
    *,
    section: SectionUnit,
    density_cluster: dict[str, Any],
    variant_count: int = 5,
    route_plan: dict[str, Any] | None = None,
    author_proxy_context: dict[str, Any] | None = None,
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
            "When broad_generic_pressure is true, add a concrete frame, explanatory bridge, scope limit, or practical consequence instead of a synonym swap.",
            "Use plain wording, but do not let old wording block the route change.",
            "If the cluster contains an obvious splice or duplicate word, repair it cleanly while preserving meaning.",
        ],
        "must_preserve": [
            "same factual meaning",
            "same source-supported viewpoint or stance",
            "citations already present in source_cluster",
            "direct quoted text already present in source_cluster",
        ],
        "length_guidance": {
            "source_words": source_words,
            "preferred_min_words": max(8, round(source_words * 0.80)),
            "preferred_max_words": max(12, round(source_words * 1.25)),
        },
        "constraints": [
            "Any added bridge, specificity, or framing must be relevant to the source topic and consistent with nearby context.",
            "Reject hard failures: fake personal stories, fake citations, fake statistics, fake dates, fake named events, broken meaning, markup, junk, or corrupted output.",
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
            "writer_execution_card": _writer_execution_card(section=section, route_plan=route_plan),
            "writer_style_card": _writer_style_card(section=section, route_plan=route_plan),
            "writer_variant_plan": _writer_variant_plan(variant_count=variants, route_plan=route_plan),
            "source_blocks": _source_blocks(section.text),
            "coverage_guidance": _coverage_guidance_for_route_plan(section=section, route_plan=route_plan),
            "length_guidance": _length_guidance_for_route_plan(section=section, route_plan=route_plan),
            "method": _custom_route_writer_method(),
        })
    _attach_author_proxy_context(payload, author_proxy_context)
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_borderline_verdict_cleanup_prompt(
    *,
    current_text: str,
    current_scores: dict[str, Any],
    density_gate: dict[str, Any],
    variant_count: int = 2,
    round_index: int = 1,
    retry_feedback: dict[str, Any] | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> str:
    variants = max(1, min(5, int(variant_count or 1)))
    source_words = max(1, word_count(current_text))
    target_outcome = _borderline_verdict_target_outcome()
    payload = {
        "task": "borderline_whole_document_texture_pass",
        "goal": (
            "Revise the full current_text once so the whole document reads less uniformly polished "
            "and more like a careful bachelor-level edited draft, while preserving meaning."
        ),
        "round": {
            "round_index": max(1, int(round_index or 1)),
            "instruction": (
                "This is a verdict-focused pass. Mild copyediting is not enough; the validator will keep only "
                "a candidate with measurable scanner movement and safe structure."
            ),
        },
        "current_text": current_text,
        "scanner_state": {
            "local_blockers_cleared": True,
            "remaining_problem": "whole-document texture is still too uniform, abstract, or template-like",
            "scores": {
                "ai": current_scores.get("ai"),
                "ai_authorship": current_scores.get("ai_authorship"),
                "external": current_scores.get("external"),
                "topk": current_scores.get("topk"),
                "topk_calibrated_risk": current_scores.get("topk_calibrated_risk"),
                "qualifying_text_ai_density": current_scores.get("qualifying_text_ai_density"),
                "risky_window_count": current_scores.get("risky_window_count"),
                "unsafe_cluster_count": current_scores.get("unsafe_cluster_count"),
            },
            "remaining_sentence_pressure": _borderline_sentence_pressure(density_gate),
        },
        "target_outcome": target_outcome,
        "editorial_action": [
            "Keep the existing paragraph order and argument.",
            "Replace polished abstract bridge wording with plain source-level wording.",
            "Where a sentence uses a broad institutional label, turn it into a simpler concrete relation already implied by the paragraph.",
            "Vary sentence weight slightly; do not make every bridge sentence equally smooth.",
            "Do not chase local unsafe clusters; the local blockers are already safe.",
            "Touch more than one paragraph when the whole document has a uniform texture problem.",
        ],
        "style_boundary": [
            "plain bachelor-level report or essay prose",
            "clear, natural, and source-near",
            "not casual, slangy, decorative, or fake-personal",
            "not polished institutional, marketing, journal-style, or textbook-summary phrasing",
        ],
        "length_policy": {
            "source_words": source_words,
            "do_not_compress_below_words": round(source_words * _borderline_min_word_ratio()),
            "more_words_allowed_if_needed": True,
            "preserve_paragraph_count": _paragraph_count(current_text),
        },
        "hard_failures": [
            "new personal story",
            "fake citation",
            "fake statistic",
            "fake date",
            "fake named event",
            "broken meaning",
            "lost paragraph",
            "markdown",
            "HTML",
            "commentary",
            "junk or corrupted output",
        ],
        "writer_variant_plan": _borderline_writer_variant_plan(
            variant_count=variants,
            target_outcome=target_outcome,
        ),
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "full replacement document"}
                for index in range(1, variants + 1)
            ]
        },
    }
    if isinstance(retry_feedback, dict) and retry_feedback:
        payload["previous_rejection_feedback"] = retry_feedback
    _attach_author_proxy_context(payload, author_proxy_context)
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_direct_scanner_leapfrog_prompt(
    *,
    section: SectionUnit,
    density_cluster: dict[str, Any],
    route_plan: dict[str, Any] | None = None,
    variant_count: int = 5,
    batch_index: int = 1,
    author_proxy_context: dict[str, Any] | None = None,
) -> str:
    variants = max(1, min(5, int(variant_count or 1)))
    source_words = max(1, int(section.word_count or word_count(section.text)))
    min_words = max(8, round(source_words * _float_env(
        "DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_MIN_LENGTH_RATIO",
        0.65,
        minimum=0.35,
        maximum=1.0,
    )))
    max_words = max(min_words + 1, round(source_words * _float_env(
        "DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_MAX_LENGTH_RATIO",
        1.15,
        minimum=0.75,
        maximum=1.5,
    )))
    plan = route_plan if _route_plan_valid(route_plan) else None
    payload: dict[str, Any] = {
        "task": "direct_scanner_cluster_leapfrog",
        "goal": "Rewrite only this scanner-selected cluster with a stronger route, then the validator will rescan the full document.",
        "cluster": {
            "section_id": section.section_id,
            "source_cluster": section.text,
            "before_context": _section_before_context(section),
            "after_context": _section_after_context(section),
            "source_word_count": source_words,
            "source_block_count": section.paragraph_count,
            "source_blocks": _source_blocks(section.text),
            "source_event_beats": _source_event_beats(section.text),
            "source_phrase_anchors": _source_phrase_anchors(section.text),
            "referential_continuity": _referential_continuity(
                section.text,
                before_context=_section_before_context(section),
            ),
        },
        "scanner_focus": {
            "source": "eligible_span_density.top_unsafe_clusters",
            "sentence_count": density_cluster.get("sentence_count"),
            "word_count": density_cluster.get("word_count"),
            "preview": density_cluster.get("preview"),
            "generic_hit_count": len(density_cluster.get("generic_hits") or [])
            if isinstance(density_cluster.get("generic_hits"), list)
            else None,
            "transition_count": density_cluster.get("transition_count"),
        },
        "length_guidance": {
            "source_words": source_words,
            "small_variant_allowed": True,
            "preferred_min_words": min_words,
            "preferred_max_words": max_words,
            "purpose": (
                "Use the shortest complete route that preserves the central source meaning. "
                "Do not summarize, but group repeated material when the source repeats the same job."
            ),
        },
        "target_texture": [
            "plain bachelor-level report or essay wording",
            "concrete wording that moves the route",
            "uneven but clear sentence route",
            "no polished abstract summary",
        ],
        "selection_hint": [
            "The validator will prefer candidates that move both document-level route pattern and predictable-span pressure.",
            "Do not optimize by adding fake-human noise, slang, errors, or decorative detail.",
            "Do not only improve smoothness; produce a genuinely different route through the same source material.",
        ],
        "method": [
            "Treat writer_execution_card as the highest-priority execution summary when it is present.",
            "Follow writer_variant_plan so variants are genuinely different route executions, not near-duplicate paraphrases.",
            "Follow execution_brief.replacement_route when execution_brief is present.",
            "For top-k work, directly execute writer_execution_card.route_to_write and avoid writer_execution_card.do_not_do.",
            "Satisfy execution_brief.must_change and target_sentence_jobs without copying plan labels.",
            "Write a route-changing replacement for the whole selected cluster, not a synonym patch.",
            "Keep the central source facts, subjects, actions, outcomes, citations, quotations, and viewpoint.",
            "For broad_explanatory_report clusters, use controlled specificity, explanatory bridges, concrete framing, scope limits, practical consequences, contrasts, or sharper specific angles when genericness is the blocker.",
            "Remove repeated category-dump wording when it does not carry a distinct source fact.",
            "Do not make every sentence the same length or the same polished shape.",
        ],
        "constraints": [
            "Any added bridge, specificity, or framing must be relevant to the source topic and consistent with nearby context.",
            "Reject hard failures: fake personal stories, fake citations, fake statistics, fake dates, fake named events, broken meaning, markup, junk, or corrupted output.",
            "Do not make the writing casual or slangy.",
            "Do not return the whole document.",
            "Do not write a plan or explanation.",
            "Return a complete replacement for source_cluster only.",
        ],
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, variants + 1)
            ]
        },
    }
    if plan:
        payload["execution_brief"] = plan
        payload["writer_execution_card"] = _writer_execution_card(section=section, route_plan=plan)
        payload["writer_style_card"] = _writer_style_card(section=section, route_plan=plan)
        payload["writer_variant_plan"] = _writer_variant_plan(variant_count=variants, route_plan=plan)
        payload["coverage_guidance"] = _coverage_guidance_for_route_plan(section=section, route_plan=plan)
    else:
        payload["execution_brief"] = None
        payload["fallback_route_blueprint"] = build_route_blueprint(section=section, local_goal=_local_goal(section.text, section.text))
    if int(batch_index or 1) > 1:
        payload["retry_batch"] = {
            "batch_index": int(batch_index),
            "instruction": "Use a different sentence route from prior attempts. Do not repeat the same opener or ending shape.",
        }
    _attach_author_proxy_context(payload, author_proxy_context)
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_risky_window_cleanup_variants(
    *,
    section: SectionUnit,
    current_scores: dict[str, Any],
    gateway: LLMGateway,
    variant_count: int = 5,
    route_plan: dict[str, Any] | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    return _generate_loose_variants_from_builder(
        prompt_builder=lambda count: build_risky_window_cleanup_prompt(
            section=section,
            current_scores=current_scores,
            variant_count=count,
            route_plan=route_plan,
            author_proxy_context=author_proxy_context,
        ),
        gateway=gateway,
        variant_count=variant_count,
    )


def generate_unsafe_cluster_cleanup_variants(
    *,
    section: SectionUnit,
    density_cluster: dict[str, Any],
    gateway: LLMGateway,
    variant_count: int = 5,
    route_plan: dict[str, Any] | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    return _generate_loose_variants_from_builder(
        prompt_builder=lambda count: build_unsafe_cluster_cleanup_prompt(
            section=section,
            density_cluster=density_cluster,
            variant_count=count,
            route_plan=route_plan,
            author_proxy_context=author_proxy_context,
        ),
        gateway=gateway,
        variant_count=variant_count,
    )


def generate_borderline_verdict_cleanup_variants(
    *,
    current_text: str,
    current_scores: dict[str, Any],
    density_gate: dict[str, Any],
    gateway: LLMGateway,
    variant_count: int = 2,
    round_index: int = 1,
    retry_feedback: dict[str, Any] | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    return _generate_loose_variants_from_builder(
        prompt_builder=lambda count: build_borderline_verdict_cleanup_prompt(
            current_text=current_text,
            current_scores=current_scores,
            density_gate=density_gate,
            variant_count=count,
            round_index=round_index,
            retry_feedback=retry_feedback,
            author_proxy_context=author_proxy_context,
        ),
        gateway=gateway,
        variant_count=variant_count,
    )


def generate_direct_scanner_leapfrog_variants(
    *,
    section: SectionUnit,
    density_cluster: dict[str, Any],
    gateway: LLMGateway,
    variant_count: int = 5,
    route_plan: dict[str, Any] | None = None,
    batch_index: int = 1,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    return _generate_loose_variants_from_builder(
        prompt_builder=lambda count: build_direct_scanner_leapfrog_prompt(
            section=section,
            density_cluster=density_cluster,
            route_plan=route_plan,
            variant_count=count,
            batch_index=batch_index,
            author_proxy_context=author_proxy_context,
        ),
        gateway=gateway,
        variant_count=variant_count,
    )


def _generate_loose_variants_from_builder(
    *,
    prompt_builder: Any,
    gateway: LLMGateway,
    variant_count: int,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    variants = max(1, min(5, int(variant_count or 1)))
    if variants <= 1 or not _bool_env("DRAFTPROOF_REWRITE_V5_PARALLEL_VARIANTS", True):
        prompt = prompt_builder(variants)
        return _generate_loose_variants(
            prompt=prompt,
            gateway=gateway,
            variant_count=variants,
            max_tokens=_serial_variant_max_tokens(prompt, variants),
        )
    fanout = _int_env("DRAFTPROOF_REWRITE_V5_LLM_FANOUT", 3, minimum=1, maximum=5)
    if fanout <= 1:
        prompt = prompt_builder(variants)
        return _generate_loose_variants(
            prompt=prompt,
            gateway=gateway,
            variant_count=variants,
            max_tokens=_serial_variant_max_tokens(prompt, variants),
        )
    return _generate_parallel_loose_variants(
        prompt_builder=prompt_builder,
        gateway=gateway,
        variant_count=variants,
        worker_limit=min(variants, fanout),
    )


def _generate_parallel_loose_variants(
    *,
    prompt_builder: Any,
    gateway: LLMGateway,
    variant_count: int,
    worker_limit: int,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    base_prompt = prompt_builder(variant_count)
    base_max_tokens = _int_env(
        "DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_MAX_TOKENS",
        8000,
        minimum=1000,
        maximum=12000,
    )
    prompts = [
        _parallel_lane_prompt(base_prompt, lane_index=index, lane_count=variant_count)
        for index in range(1, variant_count + 1)
    ]

    def run_lane(index: int, prompt: str) -> dict[str, Any]:
        try:
            parsed, diagnostics, sent_prompt, raw = _generate_loose_variants(
                prompt=prompt,
                gateway=gateway,
                variant_count=1,
                max_tokens=_parallel_single_variant_max_tokens(prompt, base_max_tokens),
            )
            return {
                "index": index,
                "status": "ok",
                "variants": parsed,
                "diagnostics": diagnostics,
                "prompt": sent_prompt,
                "completion": raw,
            }
        except Exception as exc:  # Keep one bad lane from wasting other valid candidates.
            return {
                "index": index,
                "status": "exception",
                "variants": [],
                "diagnostics": {
                    "status": "exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                "prompt": prompt,
                "completion": "",
            }

    lane_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(worker_limit))) as executor:
        futures = {
            executor.submit(run_lane, index, prompt): index
            for index, prompt in enumerate(prompts, start=1)
        }
        for future in as_completed(futures):
            lane_results.append(future.result())
    lane_results.sort(key=lambda row: int(row.get("index") or 0))

    merged_variants: list[RecompositionVariant] = []
    rejected: list[dict[str, Any]] = []
    for lane in lane_results:
        index = int(lane.get("index") or 0)
        lane_variants = lane.get("variants") if isinstance(lane.get("variants"), list) else []
        if not lane_variants:
            rejected.append({
                "index": index,
                "reason": "lane_no_valid_variant",
                "diagnostics": lane.get("diagnostics"),
            })
            continue
        variant = lane_variants[0]
        merged_variants.append(RecompositionVariant(
            variant_id=f"v{index}",
            text=variant.text,
            word_count=variant.word_count,
            author_proxy_provenance=variant.author_proxy_provenance,
            author_review_items=variant.author_review_items,
        ))

    prompt_log = json.dumps({
        "parallel_variant_generation": True,
        "requested_variant_count": variant_count,
        "worker_limit": worker_limit,
        "prompts": [
            {"index": row.get("index"), "prompt": row.get("prompt")}
            for row in lane_results
        ],
    }, ensure_ascii=False, indent=2)
    completion_log = json.dumps({
        "parallel_variant_generation": True,
        "requested_variant_count": variant_count,
        "worker_limit": worker_limit,
        "completions": [
            {
                "index": row.get("index"),
                "status": row.get("status"),
                "diagnostics": row.get("diagnostics"),
                "completion": row.get("completion"),
            }
            for row in lane_results
        ],
    }, ensure_ascii=False, indent=2)
    lane_diagnostics = [
        {
            "index": row.get("index"),
            "status": row.get("status"),
            "diagnostics": _compact_parallel_lane_diagnostics(row.get("diagnostics")),
        }
        for row in lane_results
    ]
    return merged_variants, {
        "status": "ok" if merged_variants else "schema_failed",
        "parallel_variant_generation": True,
        "requested_variant_count": variant_count,
        "variant_count": len(merged_variants),
        "parallel_call_count": len(lane_results),
        "parallel_worker_limit": worker_limit,
        "rejected": rejected,
        "lanes": lane_diagnostics,
    }, prompt_log, completion_log


def _parallel_lane_prompt(prompt: str, *, lane_index: int, lane_count: int) -> str:
    prefix = "Return valid JSON only.\n"
    if not prompt.startswith(prefix):
        return prompt
    try:
        payload = json.loads(prompt[len(prefix):])
    except json.JSONDecodeError:
        return prompt
    if not isinstance(payload, dict):
        return prompt
    assigned_variant = _assigned_writer_variant(payload, lane_index=lane_index)
    if assigned_variant:
        payload["assigned_writer_variant"] = assigned_variant
        payload["writer_variant_plan"] = [assigned_variant]
    output_schema = payload.get("output_schema") if isinstance(payload.get("output_schema"), dict) else {}
    if isinstance(output_schema, dict):
        output_schema["variants"] = [
            _author_proxy_output_variant_template()
            if _author_proxy_active(payload.get("author_proxy_context"))
            else {"variant_id": "v1", "text": "..."}
        ]
        payload["output_schema"] = output_schema
    payload["parallel_generation_lane"] = {
        "lane": lane_index,
        "total_lanes": lane_count,
        "assigned_variant_id": assigned_variant.get("variant_id") if assigned_variant else f"v{lane_index}",
        "instruction": (
            "Produce one independent replacement candidate for this lane by executing assigned_writer_variant when present. "
            "Do not mention the lane or add commentary."
        ),
    }
    return prefix + json.dumps(payload, ensure_ascii=False, indent=2)


def _assigned_writer_variant(payload: dict[str, Any], *, lane_index: int) -> dict[str, Any]:
    plans = payload.get("writer_variant_plan")
    if not isinstance(plans, list) or not plans:
        return {}
    index = max(0, min(len(plans) - 1, int(lane_index or 1) - 1))
    row = plans[index] if isinstance(plans[index], dict) else {}
    if not row:
        return {}
    return dict(row)


def _parallel_single_variant_max_tokens(prompt: str, base_max_tokens: int) -> int:
    configured = os.environ.get("DRAFTPROOF_REWRITE_V5_PARALLEL_MAX_TOKENS")
    if configured:
        return _int_env("DRAFTPROOF_REWRITE_V5_PARALLEL_MAX_TOKENS", 2600, minimum=800, maximum=12000)
    source_words = _max_prompt_word_count_hint(prompt)
    if source_words:
        return max(1000, min(base_max_tokens, int(source_words * 8) + 900))
    return max(1000, min(base_max_tokens, 3000))


def _serial_variant_max_tokens(prompt: str, variant_count: int) -> int:
    configured = os.environ.get("DRAFTPROOF_REWRITE_V5_SERIAL_MAX_TOKENS")
    if configured:
        return _int_env("DRAFTPROOF_REWRITE_V5_SERIAL_MAX_TOKENS", 3600, minimum=1000, maximum=12000)
    base_max_tokens = _int_env(
        "DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_MAX_TOKENS",
        8000,
        minimum=1000,
        maximum=12000,
    )
    source_words = _max_prompt_word_count_hint(prompt)
    if source_words:
        variants = max(1, min(5, int(variant_count or 1)))
        estimated = int(source_words * variants * 2.25) + 900
        return max(1200, min(base_max_tokens, estimated))
    return max(1200, min(base_max_tokens, 3600))


def _max_prompt_word_count_hint(prompt: str) -> int:
    prefix = "Return valid JSON only.\n"
    if not prompt.startswith(prefix):
        return 0
    try:
        payload = json.loads(prompt[len(prefix):])
    except json.JSONDecodeError:
        return 0
    values: list[int] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"source_word_count", "word_count", "preferred_min_words", "preferred_max_words"}:
                    try:
                        number = int(item)
                    except (TypeError, ValueError):
                        number = 0
                    if 0 < number < 10000:
                        values.append(number)
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(payload)
    return max(values) if values else 0


def _compact_parallel_lane_diagnostics(value: Any) -> dict[str, Any]:
    diagnostics = value if isinstance(value, dict) else {}
    return {
        "status": diagnostics.get("status"),
        "reason": diagnostics.get("reason"),
        "variant_count": diagnostics.get("variant_count"),
        "elapsed_seconds": diagnostics.get("elapsed_seconds"),
        "finish_reason": diagnostics.get("finish_reason"),
        "native_finish_reason": diagnostics.get("native_finish_reason"),
        "structured_output_mode": diagnostics.get("structured_output_mode"),
        "error_type": diagnostics.get("error_type"),
    }


def _generate_loose_variants(
    *,
    prompt: str,
    gateway: LLMGateway,
    variant_count: int,
    max_tokens: int | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    variants = max(1, min(5, int(variant_count or 1)))
    include_author_proxy_fields = _prompt_author_proxy_active(prompt)
    structured = structured_json_request_options(
        getattr(gateway, "model", None),
        _variants_response_format(variants, include_author_proxy_fields=include_author_proxy_fields),
    )
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured.get("provider"))
    started = time.monotonic()
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a variants array.",
        response_format=structured.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_TEMPERATURE", 0.35, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_TOP_P", 0.9, minimum=0.1, maximum=1.0),
        max_tokens=max_tokens if max_tokens is not None else _int_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_MAX_TOKENS", 8000, minimum=1000, maximum=12000),
    )
    elapsed = time.monotonic() - started
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
        "author_proxy_variant_schema": include_author_proxy_fields,
        "elapsed_seconds": round(elapsed, 3),
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
        "controlled_expansion": sanitized.get("controlled_expansion"),
        "sentence_plan_count": len(sanitized.get("sentence_plan") or []),
        "length_target": sanitized.get("length_target"),
    }


def _sanitize_route_plan(plan: dict[str, Any], *, source_text: str) -> dict[str, Any]:
    source = str(source_text or "")
    return {
        "content_profile": _content_profile(plan.get("content_profile")),
        "primary_metric": _primary_metric(plan.get("primary_metric")),
        "cluster_role": _cluster_role(plan.get("cluster_role")),
        "dominant_failure_pattern": _failure_pattern(plan.get("dominant_failure_pattern")),
        "route_strategy": _route_strategy(plan.get("route_strategy")),
        "profile_reason": _short_string(plan.get("profile_reason"), limit=220),
        "failed_route": _short_string(plan.get("failed_route"), limit=320),
        "replacement_route": _short_string(plan.get("replacement_route"), limit=360),
        "topk_route_diagnosis": _sanitize_topk_route_diagnosis(plan.get("topk_route_diagnosis")),
        "source_block_plan": _sanitize_source_block_plan(plan.get("source_block_plan"), source_text=source, limit=8),
        "target_sentence_jobs": _sanitize_target_sentence_jobs(plan.get("target_sentence_jobs"), source_text=source, limit=8),
        "affected_unit_actions": _sanitize_affected_unit_actions(plan.get("affected_unit_actions"), source_text=source, limit=8),
        "must_change": _string_list(plan.get("must_change"), limit=8),
        "must_preserve": _sanitize_must_preserve(plan.get("must_preserve"), source_text=source, limit=16),
        "sentence_plan": _string_list(plan.get("sentence_plan"), limit=8),
        "avoid_phrases": _supported_or_short_list(plan.get("avoid_phrases"), source_text=source, limit=12),
        "length_target": _length_target(plan.get("length_target")),
        "reason_this_should_move_score": _short_string(plan.get("reason_this_should_move_score"), limit=320),
        "controlled_expansion": _sanitize_controlled_expansion(plan.get("controlled_expansion")),
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


def _sanitize_affected_unit_actions(value: Any, *, source_text: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value if isinstance(value, list) else [], start=1):
        if not isinstance(row, dict):
            continue
        affected_text = _supported_quote(row.get("affected_text"), source_text) or _short_string(row.get("affected_text"), limit=260)
        required_action = _short_string(row.get("required_action"), limit=300)
        insufficient_edit = _short_string(row.get("insufficient_edit"), limit=220)
        if not affected_text or not required_action or not insufficient_edit:
            continue
        rows.append({
            "unit_id": _short_string(row.get("unit_id"), limit=24) or f"u{index:03d}",
            "affected_text": affected_text,
            "problem_role": _short_string(row.get("problem_role"), limit=220),
            "required_action": required_action,
            "operator_stack": _operator_stack(row.get("operator_stack")),
            "must_preserve": _supported_or_short_list(row.get("must_preserve"), source_text=source_text, limit=5),
            "insufficient_edit": insufficient_edit,
        })
        if len(rows) >= limit:
            break
    return rows


def _sanitize_topk_route_diagnosis(value: Any) -> dict[str, str]:
    row = value if isinstance(value, dict) else {}
    return {
        "infected_unit_id": _short_string(row.get("infected_unit_id"), limit=24),
        "current_route": _short_string(row.get("current_route"), limit=260),
        "predictable_path": _short_string(row.get("predictable_path"), limit=260),
        "primary_operator": _topk_route_operator(row.get("primary_operator")),
        "replacement_route": _short_string(row.get("replacement_route"), limit=300),
        "insufficient_edit": _short_string(row.get("insufficient_edit"), limit=220),
    }


def _sanitize_controlled_expansion(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    move = _controlled_expansion_move(row.get("move"))
    required = bool(row.get("required")) and move != "none"
    return {
        "required": required,
        "move": move if required else "none",
        "instruction": _short_string(row.get("instruction"), limit=260),
        "why_needed": _short_string(row.get("why_needed"), limit=220),
    }


def _controlled_expansion_move(value: Any) -> str:
    move = _short_string(value, limit=80)
    if move in _CONTROLLED_EXPANSION_MOVES:
        return move
    return "none"


def _controlled_expansion_move_for_context(
    *,
    content_profile: str,
    primary_metric: str,
    target_count: int,
    sentence_count: int,
) -> str:
    profile = _content_profile(content_profile)
    metric = _primary_metric(primary_metric)
    if profile != "broad_explanatory_report":
        return "explanatory_bridge"
    if metric == "topk_density":
        return "contrast_or_specific_angle" if target_count >= 3 else "concrete_framing"
    if metric == "unsafe_cluster_count":
        return "scope_limit" if target_count >= 3 else "practical_consequence"
    if sentence_count >= 4:
        return "concrete_framing"
    return "explanatory_bridge"


def _controlled_expansion_instruction(move: str) -> str:
    normalized = _controlled_expansion_move(move)
    instructions = {
        "explanatory_bridge": "Add one short bridge that explains why two source beats belong together without turning it into a broad conclusion.",
        "concrete_framing": "Frame the broad claim through concrete terms already visible in the cluster before returning to the wider point.",
        "scope_limit": "Narrow the broad claim by stating its condition, limit, or practical boundary in plain wording.",
        "practical_consequence": "Attach the claim to one practical consequence already implied by the source route.",
        "contrast_or_specific_angle": "Use one contrast or sharper angle to break category-list movement while keeping the same source meaning.",
    }
    return instructions.get(normalized, "")


def _route_plan_needs_controlled_expansion(plan: dict[str, Any]) -> bool:
    expansion = plan.get("controlled_expansion") if isinstance(plan.get("controlled_expansion"), dict) else {}
    if bool(expansion.get("required")) and _controlled_expansion_move(expansion.get("move")) != "none":
        return True
    profile = _content_profile(plan.get("content_profile"))
    failure = _failure_pattern(plan.get("dominant_failure_pattern"))
    metric = _primary_metric(plan.get("primary_metric"))
    return (
        profile == "broad_explanatory_report"
        and failure in {"category_dump", "claim_chain", "transition_stack", "mixed"}
        and metric in {"topk_density", "unsafe_cluster_count", "risky_window_count", "mixed"}
    )


def _controlled_expansion_for_writer(plan: dict[str, Any]) -> dict[str, Any]:
    expansion = plan.get("controlled_expansion") if isinstance(plan.get("controlled_expansion"), dict) else {}
    if bool(expansion.get("required")) and _controlled_expansion_move(expansion.get("move")) != "none":
        return {
            "required": True,
            "move": _controlled_expansion_move(expansion.get("move")),
            "instruction": _short_string(expansion.get("instruction"), limit=260),
            "why_needed": _short_string(expansion.get("why_needed"), limit=220),
        }
    if _route_plan_needs_controlled_expansion(plan):
        actions = plan.get("affected_unit_actions") if isinstance(plan.get("affected_unit_actions"), list) else []
        sentence_jobs = plan.get("target_sentence_jobs") if isinstance(plan.get("target_sentence_jobs"), list) else []
        move = _controlled_expansion_move_for_context(
            content_profile=_content_profile(plan.get("content_profile")),
            primary_metric=_primary_metric(plan.get("primary_metric")),
            target_count=sum(1 for row in actions if isinstance(row, dict)),
            sentence_count=max(len(sentence_jobs), len(actions)),
        )
        return {
            "required": True,
            "move": move,
            "instruction": _controlled_expansion_instruction(move),
            "why_needed": "The route plan identifies broad generic movement that cannot be fixed by preserving the old route.",
        }
    return {
        "required": False,
        "move": "none",
        "instruction": "",
        "why_needed": "",
    }


def _topk_route_operator(value: Any) -> str:
    operator = _short_string(value, limit=80)
    if operator in _TOPK_ROUTE_OPERATORS:
        return operator
    return "CLAUSE_ROUTE_CHANGE"


def _operator_stack(value: Any) -> list[str]:
    operators: list[str] = []
    for item in _raw_list(value):
        operator = _topk_route_operator(item)
        if operator and operator not in operators:
            operators.append(operator)
        if len(operators) >= 5:
            break
    return operators or ["CLAUSE_ROUTE_CHANGE"]


def _route_plan_valid(plan: Any) -> bool:
    return (
        isinstance(plan, dict)
        and _content_profile(plan.get("content_profile")) in set(_ROUTE_PLAN_CONTENT_PROFILES)
        and _primary_metric(plan.get("primary_metric")) in _PRIMARY_METRIC_OPTIONS
        and _cluster_role(plan.get("cluster_role")) in set(_ROUTE_PLAN_CLUSTER_ROLES)
        and _failure_pattern(plan.get("dominant_failure_pattern")) in set(_ROUTE_PLAN_FAILURE_PATTERNS)
        and _route_strategy(plan.get("route_strategy")) in set(_ROUTE_PLAN_STRATEGIES)
        and bool(plan.get("source_block_plan"))
        and bool(plan.get("target_sentence_jobs"))
        and bool(plan.get("affected_unit_actions"))
        and bool(plan.get("topk_route_diagnosis"))
        and bool(plan.get("replacement_route"))
        and bool(plan.get("must_change"))
        and bool(plan.get("must_preserve"))
        and bool(plan.get("sentence_plan"))
        and _length_target(plan.get("length_target")) in {"same_length", "slight_expand", "expand"}
    )


_PRIMARY_METRIC_OPTIONS = {
    "topk_density",
    "unsafe_cluster_count",
    "risky_window_count",
    "ai_likelihood",
    "external_proxy",
    "rank",
    "mixed",
}


_TOPK_ROUTE_OPERATORS = {
    "CLAUSE_ROUTE_CHANGE",
    "LIST_RHYTHM_BREAK",
    "ABSTRACT_TO_PRACTICAL_FRAME",
    "GENERIC_TRANSITION_REMOVAL",
    "SENTENCE_WEIGHT_VARIATION",
}


_TOPK_ROUTE_OPERATOR_WRITER_ACTIONS = {
    "CLAUSE_ROUTE_CHANGE": "Change the sentence starting logic and cause-effect path; do not keep the same opener with different words.",
    "LIST_RHYTHM_BREAK": "Break a stacked list into grouped meaning or a practical contrast while preserving the listed source material.",
    "ABSTRACT_TO_PRACTICAL_FRAME": "Replace abstract summary movement with a concrete source action, condition, decision, or consequence.",
    "GENERIC_TRANSITION_REMOVAL": "Remove formulaic transition movement and make the source subject or action carry the bridge.",
    "SENTENCE_WEIGHT_VARIATION": "Vary sentence weight by letting one short bridge or longer concrete sentence carry the route change.",
}


def _primary_metric(value: Any) -> str:
    metric = _short_string(value, limit=80)
    if metric in _PRIMARY_METRIC_OPTIONS:
        return metric
    return "mixed"


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
                            "primary_metric": {
                                "type": "string",
                                "enum": sorted(_PRIMARY_METRIC_OPTIONS),
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
                            "topk_route_diagnosis": {
                                "type": "object",
                                "properties": {
                                    "infected_unit_id": {"type": "string"},
                                    "current_route": {"type": "string"},
                                    "predictable_path": {"type": "string"},
                                    "primary_operator": {
                                        "type": "string",
                                        "enum": sorted(_TOPK_ROUTE_OPERATORS),
                                    },
                                    "replacement_route": {"type": "string"},
                                    "insufficient_edit": {"type": "string"},
                                },
                                "required": [
                                    "infected_unit_id",
                                    "current_route",
                                    "predictable_path",
                                    "primary_operator",
                                    "replacement_route",
                                    "insufficient_edit",
                                ],
                                "additionalProperties": False,
                            },
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
                            "affected_unit_actions": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 8,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "unit_id": {"type": "string"},
                                        "affected_text": {"type": "string"},
                                        "problem_role": {"type": "string"},
                                        "required_action": {"type": "string"},
                                        "operator_stack": {
                                            "type": "array",
                                            "minItems": 1,
                                            "maxItems": 5,
                                            "items": {
                                                "type": "string",
                                                "enum": sorted(_TOPK_ROUTE_OPERATORS),
                                            },
                                        },
                                        "must_preserve": {
                                            "type": "array",
                                            "minItems": 0,
                                            "maxItems": 5,
                                            "items": {"type": "string"},
                                        },
                                        "insufficient_edit": {"type": "string"},
                                    },
                                    "required": [
                                        "unit_id",
                                        "affected_text",
                                        "problem_role",
                                        "required_action",
                                        "operator_stack",
                                        "must_preserve",
                                        "insufficient_edit",
                                    ],
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
                            "controlled_expansion": {
                                "type": "object",
                                "properties": {
                                    "required": {"type": "boolean"},
                                    "move": {
                                        "type": "string",
                                        "enum": list(_CONTROLLED_EXPANSION_MOVES.keys()),
                                    },
                                    "instruction": {"type": "string"},
                                    "why_needed": {"type": "string"},
                                },
                                "required": ["required", "move", "instruction", "why_needed"],
                                "additionalProperties": False,
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
        allowed_keys = {"variant_id", "text", "author_proxy_provenance", "author_review_items"}
        if not set(row.keys()).issubset(allowed_keys):
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
        variants.append(RecompositionVariant(
            variant_id=variant_id,
            text=text,
            word_count=word_count(text),
            author_proxy_provenance=_author_proxy_item_list(row.get("author_proxy_provenance")),
            author_review_items=_author_proxy_item_list(row.get("author_review_items")),
        ))
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
    author_proxy_context: dict[str, Any] | None = None,
    author_proxy_phase: str | None = None,
) -> dict[str, Any]:
    author_proxy_audit = _author_proxy_candidate_audit(
        section.text,
        variant.text,
        author_proxy_context,
        phase=author_proxy_phase or label,
    )
    author_proxy_quality = _author_proxy_quality_score(
        source_text=section.text,
        candidate_text=variant.text,
        context=author_proxy_context,
        provenance=variant.author_proxy_provenance,
        review_items=variant.author_review_items,
        audit=author_proxy_audit,
    )
    source_words = max(1, int(section.word_count or word_count(section.text)))
    replacement_words = max(1, int(variant.word_count or word_count(variant.text)))
    current_words = max(1, word_count(current_text))
    document_window_ratio = source_words / current_words
    if (
        _author_proxy_active(author_proxy_context)
        and document_window_ratio >= _author_proxy_document_window_ratio()
        and replacement_words < round(source_words * _author_proxy_document_min_word_ratio())
    ):
        return {
            "section_id": section.section_id,
            "variant_id": variant.variant_id,
            "label": label,
            "text": variant.text,
            "word_count": variant.word_count,
            "apply_status": {
                "applied": False,
                "reason": "author_proxy_document_window_compressed_too_much",
                "source_words": source_words,
                "candidate_words": replacement_words,
                "minimum_candidate_words": round(source_words * _author_proxy_document_min_word_ratio()),
                "document_window_ratio": round(document_window_ratio, 4),
            },
            "scores": current_scores,
            "incremental": {},
            "local_scores": {},
            "local_goal": {},
            "author_proxy_audit": author_proxy_audit,
            "author_proxy_quality": author_proxy_quality,
            "author_proxy_provenance": variant.author_proxy_provenance,
            "author_review_items": variant.author_review_items,
        }
    boundary_integrity = _section_apply_boundary_integrity(current_text, section)
    if not boundary_integrity.get("passed"):
        return {
            "section_id": section.section_id,
            "variant_id": variant.variant_id,
            "label": label,
            "text": variant.text,
            "word_count": variant.word_count,
            "apply_status": {
                "applied": False,
                "reason": "unsafe_section_apply_boundary",
                "boundary_integrity": boundary_integrity,
            },
            "scores": current_scores,
            "incremental": {},
            "local_scores": {},
            "local_goal": {},
            "author_proxy_audit": author_proxy_audit,
            "author_proxy_quality": author_proxy_quality,
            "author_proxy_provenance": variant.author_proxy_provenance,
            "author_review_items": variant.author_review_items,
        }
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
            "author_proxy_audit": author_proxy_audit,
            "author_proxy_quality": author_proxy_quality,
            "author_proxy_provenance": variant.author_proxy_provenance,
            "author_review_items": variant.author_review_items,
        }
    source_integrity = minimal_replacement_text_integrity(current_text)
    candidate_integrity = minimal_replacement_text_integrity(candidate_text)
    integrity_regression = _text_integrity_regression(source_integrity, candidate_integrity)
    if not integrity_regression.get("passed"):
        return {
            "section_id": section.section_id,
            "variant_id": variant.variant_id,
            "label": label,
            "text": variant.text,
            "word_count": variant.word_count,
            "apply_status": {
                **apply_status,
                "applied": False,
                "reason": "candidate_text_integrity_regressed_after_apply",
                "source_integrity": source_integrity,
                "candidate_integrity": candidate_integrity,
                "integrity_regression": integrity_regression,
            },
            "scores": current_scores,
            "incremental": {},
            "local_scores": {},
            "local_goal": {},
            "author_proxy_audit": author_proxy_audit,
            "author_proxy_quality": author_proxy_quality,
            "author_proxy_provenance": variant.author_proxy_provenance,
            "author_review_items": variant.author_review_items,
        }
    candidate_report = _scan_report(candidate_text)
    candidate_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=baseline_report,
        candidate_report=candidate_report,
    ).to_dict()
    candidate_goal = _with_v5_density_gate(candidate_text, candidate_report, candidate_goal)
    scores = _score_summary(original_text, candidate_report, candidate_goal)
    _add_deltas(scores, baseline_scores)
    local_before_report = _scan_report(section.text)
    local_before_goal = evaluate_rewrite_goal(
        original_text=section.text,
        candidate_text=section.text,
        original_report=local_before_report,
        candidate_report=local_before_report,
    ).to_dict()
    local_before_goal = _with_v5_density_gate(section.text, local_before_report, local_before_goal)
    local_before_scores = _score_summary(section.text, local_before_report, local_before_goal)
    local_after_report = _scan_report(variant.text)
    local_after_goal = evaluate_rewrite_goal(
        original_text=section.text,
        candidate_text=variant.text,
        original_report=local_before_report,
        candidate_report=local_after_report,
    ).to_dict()
    local_after_goal = _with_v5_density_gate(variant.text, local_after_report, local_after_goal)
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
        "author_proxy_audit": author_proxy_audit,
        "author_proxy_quality": author_proxy_quality,
        "author_proxy_provenance": variant.author_proxy_provenance,
        "author_review_items": variant.author_review_items,
    }


def _score_full_document_variant(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_scores: dict[str, Any],
    current_text: str,
    current_scores: dict[str, Any],
    variant: RecompositionVariant,
    output_dir: Path,
    label: str,
    author_proxy_context: dict[str, Any] | None = None,
    author_proxy_phase: str | None = None,
) -> dict[str, Any]:
    candidate_text = str(variant.text or "").strip()
    author_proxy_audit = _author_proxy_candidate_audit(
        current_text,
        candidate_text,
        author_proxy_context,
        phase=author_proxy_phase or label,
    )
    author_proxy_quality = _author_proxy_quality_score(
        source_text=current_text,
        candidate_text=candidate_text,
        context=author_proxy_context,
        provenance=variant.author_proxy_provenance,
        review_items=variant.author_review_items,
        audit=author_proxy_audit,
    )
    source_words = max(1, word_count(current_text))
    candidate_words = max(1, word_count(candidate_text))
    apply_status: dict[str, Any] = {
        "applied": True,
        "scope": "full_document",
        "source_words": source_words,
        "candidate_words": candidate_words,
    }
    if not candidate_text:
        apply_status.update({"applied": False, "reason": "empty_candidate_text"})
    elif candidate_words < round(source_words * _borderline_min_word_ratio()):
        apply_status.update({"applied": False, "reason": "candidate_compressed_too_much"})
    elif candidate_words > round(source_words * _borderline_max_word_ratio()):
        apply_status.update({"applied": False, "reason": "candidate_expanded_too_much"})
    elif _paragraph_count(candidate_text) != _paragraph_count(current_text):
        apply_status.update({
            "applied": False,
            "reason": "paragraph_count_changed",
            "source_paragraph_count": _paragraph_count(current_text),
            "candidate_paragraph_count": _paragraph_count(candidate_text),
        })
    if apply_status.get("applied"):
        source_integrity = minimal_replacement_text_integrity(current_text)
        candidate_integrity = minimal_replacement_text_integrity(candidate_text)
        integrity_regression = _text_integrity_regression(source_integrity, candidate_integrity)
        if not integrity_regression.get("passed"):
            apply_status.update({
                "applied": False,
                "reason": "candidate_text_integrity_regressed",
                "source_integrity": source_integrity,
                "candidate_integrity": candidate_integrity,
                "integrity_regression": integrity_regression,
            })
    if not apply_status.get("applied"):
        return {
            "section_id": "full_document",
            "variant_id": variant.variant_id,
            "label": label,
            "text": candidate_text,
            "word_count": candidate_words,
            "apply_status": apply_status,
            "scores": current_scores,
            "incremental": {},
            "local_scores": {},
            "local_goal": {},
            "author_proxy_audit": author_proxy_audit,
            "author_proxy_quality": author_proxy_quality,
            "author_proxy_provenance": variant.author_proxy_provenance,
            "author_review_items": variant.author_review_items,
        }

    candidate_report = _scan_report(candidate_text)
    candidate_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=baseline_report,
        candidate_report=candidate_report,
    ).to_dict()
    candidate_goal = _with_v5_density_gate(candidate_text, candidate_report, candidate_goal)
    scores = _score_summary(original_text, candidate_report, candidate_goal)
    _add_deltas(scores, baseline_scores)
    safe_name = label.replace("/", "_")
    (output_dir / f"{safe_name}.txt").write_text(candidate_text)
    (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
    return {
        "section_id": "full_document",
        "variant_id": variant.variant_id,
        "label": label,
        "text": candidate_text,
        "word_count": candidate_words,
        "apply_status": apply_status,
        "scores": scores,
        "incremental": _incremental_deltas(scores, current_scores),
        "local_scores": {},
        "local_goal": {},
        "candidate_text": candidate_text,
        "candidate_report": candidate_report,
        "candidate_goal": candidate_goal,
        "author_proxy_audit": author_proxy_audit,
        "author_proxy_quality": author_proxy_quality,
        "author_proxy_provenance": variant.author_proxy_provenance,
        "author_review_items": variant.author_review_items,
    }


def _score_seed_candidate_texts(
    seed_candidate_texts: list[str],
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_scores: dict[str, Any],
    current_text: str,
    current_scores: dict[str, Any],
    output_dir: Path,
    author_proxy_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    limit = _int_env(
        "DRAFTPROOF_REWRITE_V5_SEED_CANDIDATE_LIMIT",
        3,
        minimum=1,
        maximum=8,
    )
    original_normalized = " ".join(str(original_text or "").split())
    for index, text in enumerate(seed_candidate_texts or [], start=1):
        candidate_text = str(text or "").strip()
        normalized = " ".join(candidate_text.split())
        if not normalized or normalized == original_normalized or normalized in seen:
            continue
        seen.add(normalized)
        if len(rows) >= limit:
            break
        variant = RecompositionVariant(
            variant_id=f"seed_{len(rows) + 1:02d}",
            text=candidate_text,
            word_count=word_count(candidate_text),
        )
        row = _score_full_document_variant(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_scores=current_scores,
            variant=variant,
            output_dir=output_dir,
            label=f"historical_seed_{len(rows) + 1:02d}",
            author_proxy_context=author_proxy_context,
            author_proxy_phase="historical_rewrite_seed",
        )
        row["seed_candidate"] = {
            "source": "historical_rewrite_candidate",
            "input_index": index,
        }
        rows.append(row)
    return rows


def _run_direct_scanner_leapfrog_pass(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_scores: dict[str, Any],
    current_text: str,
    current_report: dict[str, Any],
    current_goal: dict[str, Any],
    current_scores: dict[str, Any],
    gateway: LLMGateway,
    planner_gateway: LLMGateway,
    output_dir: Path,
    global_best_candidate: dict[str, Any] | None,
    max_rounds: int,
    variant_count: int,
    max_batches: int,
    progress_callback: Callable[[int, str], None] | None = None,
    progress_percent: int = 68,
    accepted_checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    started_at: float | None = None,
    max_seconds: float | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, Any]] = []
    skipped: set[tuple[Any, ...]] = set()
    for round_index in range(1, max(0, int(max_rounds or 0)) + 1):
        if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="direct_scanner_leapfrog",
                round_index=round_index,
                started_at=started_at,
                max_seconds=max_seconds,
                current_scores=current_scores,
            ))
            break
        _emit_progress(progress_callback, progress_percent, f"V5 direct scanner leapfrog {round_index}")
        density = _density_gate_for_report(current_text, current_report)
        if density.get("safe"):
            rounds.append({
                "round": round_index,
                "phase": "direct_scanner_leapfrog",
                "status": "stopped",
                "reason": "eligible_span_density_safe",
                "density_gate": _compact_density_gate(density),
            })
            break
        target = _select_density_cluster_section(
            current_text,
            current_report,
            density,
            skipped=skipped,
            selection_mode="scanner",
        )
        if target is None:
            rounds.append({
                "round": round_index,
                "phase": "direct_scanner_leapfrog",
                "status": "stopped",
                "reason": "no_density_cluster_target",
                "density_gate": _compact_density_gate(density),
            })
            break
        section, density_cluster, signature = target
        round_dir = output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        route_plan_enabled = _bool_env("DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_ROUTE_PLANNING", True)
        if route_plan_enabled:
            route_plan, route_plan_diagnostics, route_plan_prompt, route_plan_completion = generate_residual_cluster_route_plan(
                section=section,
                local_goal=_local_goal(section.text, section.text),
                planner_gateway=planner_gateway,
                fallback_gateway=gateway,
                author_proxy_context=author_proxy_context,
            )
        else:
            route_plan = None
            route_plan_diagnostics = {
                "status": "skipped",
                "reason": "direct_scanner_route_planning_disabled",
            }
            route_plan_prompt = ""
            route_plan_completion = ""
        (round_dir / "route_plan_prompt.json.txt").write_text(route_plan_prompt)
        (round_dir / "route_plan_completion.json.txt").write_text(route_plan_completion)
        if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="direct_scanner_leapfrog",
                round_index=round_index,
                started_at=started_at,
                max_seconds=max_seconds,
                current_scores=current_scores,
            ))
            break

        rows: list[dict[str, Any]] = []
        batch_diagnostics: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        accepted: dict[str, Any] | None = None
        batch_policy = _direct_scanner_batch_policy()
        for batch_index in range(1, max(1, int(max_batches or 1)) + 1):
            if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
                batch_diagnostics.append({"batch_index": batch_index, "status": "skipped", "reason": "runtime_budget_exhausted"})
                break
            variants, llm_diagnostics, prompt, completion = generate_direct_scanner_leapfrog_variants(
                section=section,
                density_cluster=density_cluster,
                gateway=gateway,
                variant_count=variant_count,
                route_plan=route_plan,
                batch_index=batch_index,
                author_proxy_context=author_proxy_context,
            )
            batch_dir = round_dir / f"batch_{batch_index:02d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / "prompt.json.txt").write_text(prompt)
            (batch_dir / "completion.json.txt").write_text(completion)
            batch_rows = [
                _score_residual_variant(
                    original_text=original_text,
                    baseline_report=baseline_report,
                    baseline_scores=baseline_scores,
                    current_text=current_text,
                    current_scores=current_scores,
                    section=section,
                    variant=variant,
                    output_dir=batch_dir,
                    label=f"direct_b{batch_index}_{variant.variant_id}",
                    author_proxy_context=author_proxy_context,
                    author_proxy_phase="direct_scanner_leapfrog",
                )
                for variant in variants
            ]
            for row in batch_rows:
                row["selection_policy"] = "balanced_ai_topk"
                row["direct_scanner_batch"] = batch_index
            rows.extend(batch_rows)
            global_best_candidate = _best_full_document_candidate([global_best_candidate, *batch_rows])
            selected = _best_balanced_ai_topk_candidate(rows)
            accepted = selected if selected and _has_balanced_ai_topk_movement(selected) else None
            batch_diagnostics.append({
                "batch_index": batch_index,
                "llm_generation": llm_diagnostics,
                "candidate_count": len(batch_rows),
                "selected": _compact_residual_row(selected),
                "accepted": bool(accepted),
                "batch_policy": batch_policy,
                "continue_reason": _direct_scanner_next_batch_reason(
                    selected,
                    batch_policy=batch_policy,
                    batch_index=batch_index,
                    max_batches=max_batches,
                ),
            })
            if not _should_continue_direct_scanner_batches(
                selected,
                batch_policy=batch_policy,
                batch_index=batch_index,
                max_batches=max_batches,
            ):
                break

        selected = selected or _best_balanced_ai_topk_candidate(rows)
        accepted = accepted or (selected if selected and _has_balanced_ai_topk_movement(selected) else None)
        round_payload = {
            "round": round_index,
            "phase": "direct_scanner_leapfrog",
            "status": "accepted" if accepted else "skipped",
            "reason": "accepted_balanced_ai_topk_movement" if accepted else "no_balanced_ai_topk_movement",
            "section": section.to_dict(),
            "density_cluster": density_cluster,
            "density_gate": _compact_density_gate(density),
            "generator_diagnostics": {
                "route_plan": route_plan_diagnostics,
                "batches": batch_diagnostics,
                "selection_policy": "balanced_ai_topk",
                "batch_policy": batch_policy,
            },
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
        (output_dir / f"after_round_{round_index:02d}.txt").write_text(current_text)
        if accepted_checkpoint_callback is not None:
            accepted_checkpoint_callback({
                "phase": "direct_scanner_leapfrog",
                "round": round_index,
                "reason": "accepted_balanced_ai_topk_movement",
                "accepted": accepted,
                "rewritten_document": current_text,
                "scores": current_scores,
                "goal": current_goal,
            })
        _emit_progress(progress_callback, progress_percent, f"Accepted V5 direct scanner leapfrog {round_index}")
    return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate


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
    planner_gateway: LLMGateway,
    output_dir: Path,
    global_best_candidate: dict[str, Any] | None,
    max_rounds: int,
    variant_count: int,
    progress_callback: Callable[[int, str], None] | None = None,
    progress_percent: int = 76,
    accepted_checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    started_at: float | None = None,
    max_seconds: float | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, Any]] = []
    skipped: set[tuple[Any, ...]] = set()
    for cleanup_index in range(1, max(0, int(max_rounds or 0)) + 1):
        if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="risky_window_cleanup",
                round_index=cleanup_index,
                started_at=started_at,
                max_seconds=max_seconds,
                current_scores=current_scores,
            ))
            break
        _emit_progress(progress_callback, progress_percent, f"V5 risky window cleanup {cleanup_index}")
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
            planner_gateway=planner_gateway,
            fallback_gateway=gateway,
            author_proxy_context=author_proxy_context,
        )
        (round_dir / "route_plan_prompt.json.txt").write_text(route_plan_prompt)
        (round_dir / "route_plan_completion.json.txt").write_text(route_plan_completion)
        if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="risky_window_cleanup",
                round_index=cleanup_index,
                started_at=started_at,
                max_seconds=max_seconds,
                current_scores=current_scores,
            ))
            break
        variants, llm_diagnostics, prompt, completion = generate_risky_window_cleanup_variants(
            section=section,
            current_scores=current_scores,
            gateway=gateway,
            variant_count=variant_count,
            route_plan=route_plan,
            author_proxy_context=author_proxy_context,
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
                author_proxy_context=author_proxy_context,
                author_proxy_phase="risky_window_cleanup",
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
        if accepted_checkpoint_callback is not None:
            accepted_checkpoint_callback({
                "phase": "risky_window_cleanup",
                "round": cleanup_index,
                "reason": "accepted_risky_window_movement",
                "accepted": accepted,
                "rewritten_document": current_text,
                "scores": current_scores,
                "goal": current_goal,
            })
        _emit_progress(progress_callback, progress_percent, f"Accepted V5 risky window cleanup {cleanup_index}")
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
    planner_gateway: LLMGateway,
    output_dir: Path,
    global_best_candidate: dict[str, Any] | None,
    max_rounds: int,
    variant_count: int,
    selection_mode: str = "scanner",
    route_plan_enabled: bool = True,
    progress_callback: Callable[[int, str], None] | None = None,
    progress_percent: int = 77,
    accepted_checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    started_at: float | None = None,
    max_seconds: float | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, Any]] = []
    skipped: set[tuple[Any, ...]] = set()
    consecutive_unsafe_misses = 0
    unsafe_miss_limit = _unsafe_cluster_cleanup_stop_after_misses()
    for cleanup_index in range(1, max(0, int(max_rounds or 0)) + 1):
        if unsafe_miss_limit > 0 and consecutive_unsafe_misses >= unsafe_miss_limit:
            rounds.append({
                "round": cleanup_index,
                "phase": "unsafe_cluster_cleanup",
                "status": "stopped",
                "reason": "unsafe_cluster_miss_limit_reached",
                "consecutive_no_unsafe_cluster_movement": consecutive_unsafe_misses,
                "miss_limit": unsafe_miss_limit,
                "current_scores": current_scores,
            })
            break
        if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="unsafe_cluster_cleanup",
                round_index=cleanup_index,
                started_at=started_at,
                max_seconds=max_seconds,
                current_scores=current_scores,
            ))
            break
        _emit_progress(progress_callback, progress_percent, f"V5 unsafe cluster cleanup {cleanup_index}")
        density = _density_gate_for_report(current_text, current_report)
        if density.get("safe"):
            rounds.append({
                "round": cleanup_index,
                "phase": "unsafe_cluster_cleanup",
                "status": "stopped",
                "reason": "eligible_span_density_safe",
                "density_gate": _compact_density_gate(density),
            })
            break
        target = _select_density_cluster_section(
            current_text,
            current_report,
            density,
            skipped=skipped,
            selection_mode=selection_mode,
        )
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
        if route_plan_enabled:
            route_plan, route_plan_diagnostics, route_plan_prompt, route_plan_completion = generate_residual_cluster_route_plan(
                section=section,
                local_goal=_local_goal(section.text, section.text),
                planner_gateway=planner_gateway,
                fallback_gateway=gateway,
                author_proxy_context=author_proxy_context,
            )
        else:
            route_plan = None
            route_plan_diagnostics = {"status": "skipped", "reason": "budget_clearable_unsafe_cluster_cleanup"}
            route_plan_prompt = ""
            route_plan_completion = ""
        (round_dir / "route_plan_prompt.json.txt").write_text(route_plan_prompt)
        (round_dir / "route_plan_completion.json.txt").write_text(route_plan_completion)
        if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="unsafe_cluster_cleanup",
                round_index=cleanup_index,
                started_at=started_at,
                max_seconds=max_seconds,
                current_scores=current_scores,
            ))
            break
        variants, llm_diagnostics, prompt, completion = generate_unsafe_cluster_cleanup_variants(
            section=section,
            density_cluster=density_cluster,
            gateway=gateway,
            variant_count=variant_count,
            route_plan=route_plan,
            author_proxy_context=author_proxy_context,
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
                author_proxy_context=author_proxy_context,
                author_proxy_phase="unsafe_cluster_cleanup",
            )
            for variant in variants
        ]
        global_best_candidate = _best_full_document_candidate([global_best_candidate, *rows])
        selected = _best_unsafe_cluster_cleanup_candidate(rows)
        accepted = (
            selected
            if selected and _has_unsafe_cluster_cleanup_movement(selected, cleanup_index=cleanup_index)
            else None
        )
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
            consecutive_unsafe_misses += 1
            skipped.add(signature)
            continue
        consecutive_unsafe_misses = 0
        current_text, current_report, current_goal, current_scores = _accepted_state(
            accepted=accepted,
            original_text=original_text,
            baseline_report=baseline_report,
        )
        (output_dir / f"after_round_{cleanup_index:02d}.txt").write_text(current_text)
        if accepted_checkpoint_callback is not None:
            accepted_checkpoint_callback({
                "phase": "unsafe_cluster_cleanup",
                "round": cleanup_index,
                "reason": "accepted_unsafe_cluster_movement",
                "accepted": accepted,
                "rewritten_document": current_text,
                "scores": current_scores,
                "goal": current_goal,
            })
        _emit_progress(progress_callback, progress_percent, f"Accepted V5 unsafe cluster cleanup {cleanup_index}")
    return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate


def _run_borderline_verdict_cleanup_pass(
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
    variant_count: int,
    progress_callback: Callable[[int, str], None] | None = None,
    progress_percent: int = 79,
    accepted_checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    started_at: float | None = None,
    max_seconds: float | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, Any]] = []
    retry_feedback: dict[str, Any] | None = None
    for round_index in range(1, _borderline_verdict_max_rounds() + 1):
        if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="borderline_verdict_cleanup",
                round_index=round_index,
                started_at=started_at,
                max_seconds=max_seconds,
                current_scores=current_scores,
            ))
            break
        density = _density_gate_for_report(current_text, current_report)
        if not _borderline_verdict_should_run(current_scores=current_scores, density_gate=density):
            rounds.append({
                "round": round_index,
                "phase": "borderline_verdict_cleanup",
                "status": "skipped",
                "reason": "not_borderline_or_blockers_not_safe",
                "current_scores": current_scores,
                "density_gate": _compact_density_gate(density),
            })
            break

        round_dir = output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        variants, llm_diagnostics, prompt, completion = generate_borderline_verdict_cleanup_variants(
            current_text=current_text,
            current_scores=current_scores,
            density_gate=density,
            gateway=gateway,
            variant_count=variant_count,
            round_index=round_index,
            retry_feedback=retry_feedback,
            author_proxy_context=author_proxy_context,
        )
        (round_dir / "borderline_prompt.json.txt").write_text(prompt)
        (round_dir / "borderline_completion.json.txt").write_text(completion)
        rows = [
            _score_full_document_variant(
                original_text=original_text,
                baseline_report=baseline_report,
                baseline_scores=baseline_scores,
                current_text=current_text,
                current_scores=current_scores,
                variant=variant,
                output_dir=round_dir,
                label=f"borderline_r{round_index}_{variant.variant_id}",
                author_proxy_context=author_proxy_context,
                author_proxy_phase="borderline_verdict_cleanup",
            )
            for variant in variants
        ]
        global_best_candidate = _best_full_document_candidate([global_best_candidate, *rows])
        selected = _best_borderline_verdict_candidate(rows)
        accepted = selected if selected and _has_borderline_verdict_movement(selected) else None
        round_payload = {
            "round": round_index,
            "phase": "borderline_verdict_cleanup",
            "status": "accepted" if accepted else "skipped",
            "reason": "accepted_borderline_verdict_movement" if accepted else "no_borderline_verdict_movement",
            "generator_diagnostics": llm_diagnostics,
            "retry_feedback": retry_feedback,
            "current_scores": current_scores,
            "density_gate": _compact_density_gate(density),
            "candidates": [_compact_residual_row(row) for row in rows],
            "selected": _compact_residual_row(selected),
            "accepted": _compact_residual_row(accepted),
        }
        rounds.append(round_payload)
        (round_dir / "round_result.json").write_text(json.dumps(round_payload, ensure_ascii=False, indent=2))
        if not accepted:
            retry_feedback = _borderline_rejected_candidate_feedback(selected, current_scores=current_scores)
            if retry_feedback and round_index < _borderline_verdict_max_rounds():
                continue
            break
        current_text, current_report, current_goal, current_scores = _accepted_state(
            accepted=accepted,
            original_text=original_text,
            baseline_report=baseline_report,
        )
        (output_dir / f"after_round_{round_index:02d}.txt").write_text(current_text)
        if accepted_checkpoint_callback is not None:
            accepted_checkpoint_callback({
                "phase": "borderline_verdict_cleanup",
                "round": round_index,
                "reason": "accepted_borderline_verdict_movement",
                "accepted": accepted,
                "rewritten_document": current_text,
                "scores": current_scores,
                "goal": current_goal,
            })
        _emit_progress(progress_callback, progress_percent, f"Accepted V5 borderline texture pass {round_index}")
        if _borderline_verdict_candidate_crosses_boundary(accepted):
            break
    return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate


def _run_final_topk_sentence_route_pass(
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
    progress_callback: Callable[[int, str], None] | None = None,
    progress_percent: int = 80,
    accepted_checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    started_at: float | None = None,
    max_seconds: float | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, Any]] = []
    targets = _final_topk_sentence_route_targets(current_text, current_report, current_goal)
    if not targets:
        rounds.append({
            "round": 0,
            "phase": "final_topk_sentence_route",
            "status": "skipped",
            "reason": "no_topk_sentence_targets",
            "current_scores": current_scores,
            "density_gate": _compact_density_gate(_density_gate_for_report(current_text, current_report)),
        })
        return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate

    batch_size = _final_topk_sentence_route_batch_size()
    variant_count = _final_topk_sentence_route_variant_count()
    for batch_index, offset in enumerate(range(0, len(targets), batch_size), start=1):
        if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="final_topk_sentence_route",
                round_index=batch_index,
                started_at=started_at,
                max_seconds=max_seconds,
                current_scores=current_scores,
            ))
            break
        batch_targets = targets[offset:offset + batch_size]
        batch_targets = [target for target in batch_targets if str(target.get("sentence") or "") in current_text]
        if not batch_targets:
            continue
        _emit_progress(progress_callback, progress_percent, f"V5 final top-k sentence route batch {batch_index}")
        round_dir = output_dir / f"batch_{batch_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        variants, llm_diagnostics, prompt, completion = generate_final_topk_sentence_route_variants(
                current_text=current_text,
                current_scores=current_scores,
                current_goal=current_goal,
                targets=batch_targets,
                gateway=gateway,
                variant_count=variant_count,
                author_proxy_context=author_proxy_context,
        )
        (round_dir / "topk_sentence_route_prompt.json.txt").write_text(prompt)
        (round_dir / "topk_sentence_route_completion.json.txt").write_text(completion)
        rows = [
            _score_final_topk_sentence_route_variant(
                original_text=original_text,
                baseline_report=baseline_report,
                baseline_scores=baseline_scores,
                current_text=current_text,
                current_scores=current_scores,
                targets=batch_targets,
                variant=variant,
                output_dir=round_dir,
                label=f"topk_sentence_route_b{batch_index}_{variant.get('variant_id')}",
                author_proxy_context=author_proxy_context,
                author_proxy_phase="final_topk_sentence_route",
            )
            for variant in variants
        ]
        partial_variants: list[dict[str, Any]] = []
        for variant in variants:
            variant_id = str(variant.get("variant_id") or "").strip()
            for repair in variant.get("repairs") if isinstance(variant.get("repairs"), list) else []:
                if not isinstance(repair, dict):
                    continue
                target_id = str(repair.get("target_id") or "").strip()
                if not target_id:
                    continue
                partial_variants.append({
                    "variant_id": f"{variant_id}_{target_id}",
                    "repairs": [repair],
                })
        rows.extend(
            _score_final_topk_sentence_route_variant(
                original_text=original_text,
                baseline_report=baseline_report,
                baseline_scores=baseline_scores,
                current_text=current_text,
                current_scores=current_scores,
                targets=batch_targets,
                variant=variant,
                output_dir=round_dir,
                label=f"topk_sentence_route_b{batch_index}_{variant.get('variant_id')}_partial",
                author_proxy_context=author_proxy_context,
                author_proxy_phase="final_topk_sentence_route_partial",
                require_all_targets=False,
            )
            for variant in partial_variants
        )
        selected = _best_final_topk_sentence_route_candidate(rows)
        accepted = selected if selected and _has_final_topk_sentence_route_movement(selected) else None
        round_payload = {
            "round": batch_index,
            "phase": "final_topk_sentence_route",
            "status": "accepted" if accepted else "skipped",
            "reason": "accepted_topk_sentence_route_movement" if accepted else "no_topk_sentence_route_movement",
            "generator_diagnostics": llm_diagnostics,
            "current_scores": current_scores,
            "targets": batch_targets,
            "candidates": [_compact_residual_row(row) | {
                "applied_repairs": len(row.get("applied_repairs") or []),
            } for row in rows],
            "selected": _compact_residual_row(selected),
            "accepted": _compact_residual_row(accepted),
        }
        rounds.append(round_payload)
        (round_dir / "round_result.json").write_text(json.dumps(round_payload, ensure_ascii=False, indent=2))
        global_best_candidate = _best_full_document_candidate([global_best_candidate, *rows])
        if not accepted:
            continue
        current_text, current_report, current_goal, current_scores = _accepted_state(
            accepted=accepted,
            original_text=original_text,
            baseline_report=baseline_report,
        )
        (output_dir / f"after_batch_{batch_index:02d}.txt").write_text(current_text)
        if accepted_checkpoint_callback is not None:
            accepted_checkpoint_callback({
                "phase": "final_topk_sentence_route",
                "round": batch_index,
                "reason": "accepted_topk_sentence_route_movement",
                "accepted": accepted,
                "rewritten_document": current_text,
                "scores": current_scores,
                "goal": current_goal,
            })
    return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate


def _run_safe_band_evidence_pack_attempt(
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
    progress_callback: Callable[[int, str], None] | None = None,
    progress_percent: int = 81,
    accepted_checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    started_at: float | None = None,
    max_seconds: float | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None, bool]:
    if not _safe_band_evidence_pack_enabled() or _runtime_budget_exhausted(started_at, max_seconds):
        return current_text, current_report, current_goal, current_scores, [], global_best_candidate, False
    pack_sections = _safe_band_evidence_pack_sections(
        current_text,
        current_report,
        current_goal,
        limit=_safe_band_evidence_pack_section_limit(),
    )
    if len(pack_sections) < 2:
        return current_text, current_report, current_goal, current_scores, [], global_best_candidate, False

    output_dir.mkdir(parents=True, exist_ok=True)
    pack_revision_plan: dict[str, Any] | None = None
    plan_diagnostics: dict[str, Any] = {"status": "skipped", "reason": "safe_band_author_proxy_plan_disabled"}
    if _safe_band_author_proxy_plan_enabled() and not _runtime_budget_exhausted(started_at, max_seconds):
        _emit_progress(progress_callback, progress_percent, "V5 safe-band author-proxy evidence plan")
        pack_revision_plan, plan_diagnostics, plan_prompt, plan_completion = generate_safe_band_author_proxy_revision_plan(
            sections=pack_sections,
            current_scores=current_scores,
            current_goal=current_goal,
            gateway=gateway,
            author_proxy_context=author_proxy_context,
        )
        (output_dir / "safe_band_author_proxy_revision_plan_prompt.json.txt").write_text(plan_prompt)
        (output_dir / "safe_band_author_proxy_revision_plan_completion.json.txt").write_text(plan_completion)
        (output_dir / "safe_band_author_proxy_revision_plan.json").write_text(
            json.dumps(pack_revision_plan or {}, ensure_ascii=False, indent=2)
        )
    _emit_progress(progress_callback, progress_percent, "V5 safe-band evidence repair pack")
    pack_variants, pack_diagnostics, pack_prompt, pack_completion = generate_safe_band_evidence_pack_variants(
        sections=pack_sections,
        current_scores=current_scores,
        current_goal=current_goal,
        gateway=gateway,
        variant_count=_safe_band_evidence_pack_variant_count(),
        revision_plan=pack_revision_plan,
        author_proxy_context=author_proxy_context,
    )
    (output_dir / "safe_band_evidence_pack_prompt.json.txt").write_text(pack_prompt)
    (output_dir / "safe_band_evidence_pack_completion.json.txt").write_text(pack_completion)
    if _safe_band_evidence_pack_composite_enabled():
        composite_variant = _safe_band_evidence_pack_composite_variant(
            sections=pack_sections,
            variants=pack_variants,
        )
        if composite_variant is not None:
            pack_variants = [*pack_variants, composite_variant]
    partial_variant = _safe_band_evidence_pack_partial_material_variant(
        sections=pack_sections,
        variants=pack_variants,
    )
    if partial_variant is not None:
        pack_variants = [*pack_variants, partial_variant]
    pack_rows = [
        _score_safe_band_evidence_pack_variant(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_scores=current_scores,
            sections=pack_sections,
            variant=variant,
            output_dir=output_dir,
            label=f"safe_band_evidence_pack_{variant.get('variant_id')}",
            author_proxy_context=author_proxy_context,
        )
        for variant in pack_variants
    ]
    for row in pack_rows:
        _attach_safe_band_quality_materiality(row, current_text=current_text)
    section_probe_rows = _score_safe_band_evidence_pack_section_probes(
        original_text=original_text,
        baseline_report=baseline_report,
        baseline_scores=baseline_scores,
        current_text=current_text,
        current_scores=current_scores,
        sections=pack_sections,
        variants=pack_variants,
        output_dir=output_dir,
        author_proxy_context=author_proxy_context,
    )
    for row in section_probe_rows:
        _attach_safe_band_quality_materiality(row, current_text=current_text)
    section_composite_variant = _safe_band_evidence_pack_scored_section_composite_variant(
        section_probe_rows,
        current_scores=current_scores,
    )
    section_composite_row: dict[str, Any] | None = None
    if section_composite_variant is not None:
        section_composite_row = _score_safe_band_evidence_pack_variant(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_scores=current_scores,
            sections=pack_sections,
            variant=section_composite_variant,
            output_dir=output_dir,
            label="safe_band_evidence_pack_scored_section_composite",
            author_proxy_context=author_proxy_context,
        )
        _attach_safe_band_quality_materiality(section_composite_row, current_text=current_text)
        pack_rows.append(section_composite_row)
    pack_selected = _best_safe_band_evidence_repair_candidate(pack_rows, current_scores=current_scores)
    pack_accepted = (
        pack_selected
        if pack_selected and _has_safe_band_evidence_repair_movement(pack_selected, current_scores=current_scores)
        else None
    )
    pack_payload = {
        "round": 0,
        "phase": "safe_band_evidence_pack",
        "status": "accepted" if pack_accepted else "skipped",
        "reason": "accepted_safe_band_evidence_pack_movement" if pack_accepted else "no_safe_band_evidence_pack_movement",
        "sections": [section.to_dict() for section in pack_sections],
        "generator_diagnostics": pack_diagnostics,
        "revision_plan_diagnostics": plan_diagnostics,
        "revision_plan": pack_revision_plan,
        "current_scores": current_scores,
        "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
        "candidates": [_compact_residual_row(row) for row in pack_rows],
        "section_probe_candidates": [_compact_residual_row(row) for row in section_probe_rows],
        "section_composite": _compact_residual_row(section_composite_row),
        "selected": _compact_residual_row(pack_selected),
        "accepted": _compact_residual_row(pack_accepted),
    }
    (output_dir / "round_result.json").write_text(json.dumps(pack_payload, ensure_ascii=False, indent=2))
    global_best_candidate = _best_full_document_candidate([global_best_candidate, *pack_rows])
    if not pack_accepted:
        return current_text, current_report, current_goal, current_scores, [pack_payload], global_best_candidate, False

    current_text, current_report, current_goal, current_scores = _accepted_state(
        accepted=pack_accepted,
        original_text=original_text,
        baseline_report=baseline_report,
    )
    (output_dir.parent / "after_safe_band_evidence_pack.txt").write_text(current_text)
    if accepted_checkpoint_callback is not None:
        accepted_checkpoint_callback({
            "phase": "safe_band_evidence_pack",
            "round": 0,
            "reason": "accepted_safe_band_evidence_pack_movement",
            "accepted": pack_accepted,
            "rewritten_document": current_text,
            "scores": current_scores,
            "goal": current_goal,
        })
    return current_text, current_report, current_goal, current_scores, [pack_payload], global_best_candidate, True


def _run_safe_band_evidence_repair_pass(
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
    progress_callback: Callable[[int, str], None] | None = None,
    progress_percent: int = 81,
    accepted_checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    started_at: float | None = None,
    max_seconds: float | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, Any]] = []
    if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
        rounds.append(_runtime_budget_stop_record(
            phase="safe_band_evidence_repair",
            round_index=1,
            started_at=started_at,
            max_seconds=max_seconds,
            current_scores=current_scores,
        ))
        return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate

    sections = _safe_band_evidence_repair_sections(
        current_text,
        current_report,
        current_goal,
        limit=_safe_band_evidence_repair_section_limit(),
    )
    if not sections:
        rounds.append({
            "round": 1,
            "phase": "safe_band_evidence_repair",
            "status": "skipped",
            "reason": "no_safe_band_evidence_section",
            "current_scores": current_scores,
            "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
        })
        return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate

    density_first = _safe_band_density_first_repair_should_run(current_scores=current_scores, current_goal=current_goal)
    pack_attempted = False
    if density_first and not _runtime_budget_exhausted(started_at, max_seconds):
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            pack_rounds,
            global_best_candidate,
            pack_accepted,
        ) = _run_safe_band_evidence_pack_attempt(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            gateway=gateway,
            output_dir=output_dir / "density_first_pack",
            global_best_candidate=global_best_candidate,
            progress_callback=progress_callback,
            progress_percent=progress_percent,
            accepted_checkpoint_callback=accepted_checkpoint_callback,
            started_at=started_at,
            max_seconds=max_seconds,
            author_proxy_context=author_proxy_context,
        )
        rounds.extend(pack_rounds)
        pack_attempted = bool(pack_rounds)
        if pack_accepted:
            return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate
        sections = _safe_band_evidence_repair_sections(
            current_text,
            current_report,
            current_goal,
            limit=_safe_band_evidence_repair_section_limit(),
        )
    if density_first and not _runtime_budget_exhausted(started_at, max_seconds):
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            density_rounds,
            global_best_candidate,
        ) = _run_safe_band_density_section_repair_loop(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            gateway=gateway,
            output_dir=output_dir / "density_section_repair",
            global_best_candidate=global_best_candidate,
            progress_callback=progress_callback,
            progress_percent=progress_percent,
            accepted_checkpoint_callback=accepted_checkpoint_callback,
            started_at=started_at,
            max_seconds=max_seconds,
            author_proxy_context=author_proxy_context,
        )
        rounds.extend(density_rounds)
        sections = _safe_band_evidence_repair_sections(
            current_text,
            current_report,
            current_goal,
            limit=_safe_band_evidence_repair_section_limit(),
        )

    if _safe_band_controlled_operation_enabled() and not _runtime_budget_exhausted(started_at, max_seconds):
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            operation_rounds,
            global_best_candidate,
        ) = _run_safe_band_controlled_operation_loop(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            output_dir=output_dir / "controlled_operation",
            global_best_candidate=global_best_candidate,
            progress_callback=progress_callback,
            progress_percent=progress_percent,
            accepted_checkpoint_callback=accepted_checkpoint_callback,
            started_at=started_at,
            max_seconds=max_seconds,
            author_proxy_context=author_proxy_context,
        )
        rounds.extend(operation_rounds)
        sections = _safe_band_evidence_repair_sections(
            current_text,
            current_report,
            current_goal,
            limit=_safe_band_evidence_repair_section_limit(),
        )

    if _safe_band_sentence_replacement_enabled() and not _runtime_budget_exhausted(started_at, max_seconds):
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            replacement_rounds,
            global_best_candidate,
        ) = _run_safe_band_sentence_replacement_loop(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            gateway=gateway,
            output_dir=output_dir / "sentence_replacement",
            global_best_candidate=global_best_candidate,
            progress_callback=progress_callback,
            progress_percent=progress_percent,
            accepted_checkpoint_callback=accepted_checkpoint_callback,
            started_at=started_at,
            max_seconds=max_seconds,
            author_proxy_context=author_proxy_context,
        )
        rounds.extend(replacement_rounds)
        sections = _safe_band_evidence_repair_sections(
            current_text,
            current_report,
            current_goal,
            limit=_safe_band_evidence_repair_section_limit(),
        )

    if (
        not density_first
        and _safe_band_density_section_repair_should_run(current_scores=current_scores, current_goal=current_goal)
        and not _runtime_budget_exhausted(started_at, max_seconds)
    ):
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            density_rounds,
            global_best_candidate,
        ) = _run_safe_band_density_section_repair_loop(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            gateway=gateway,
            output_dir=output_dir / "density_section_repair",
            global_best_candidate=global_best_candidate,
            progress_callback=progress_callback,
            progress_percent=progress_percent,
            accepted_checkpoint_callback=accepted_checkpoint_callback,
            started_at=started_at,
            max_seconds=max_seconds,
            author_proxy_context=author_proxy_context,
        )
        rounds.extend(density_rounds)
        sections = _safe_band_evidence_repair_sections(
            current_text,
            current_report,
            current_goal,
            limit=_safe_band_evidence_repair_section_limit(),
        )

    if _candidate_strict_safe_band_achieved({"candidate_goal": current_goal, "scores": current_scores}):
        rounds.append({
            "round": 0,
            "phase": "safe_band_evidence_pack",
            "status": "skipped",
            "reason": "strict_safe_band_already_achieved",
            "current_scores": current_scores,
            "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
        })
        return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate

    variant_count = _safe_band_evidence_repair_variant_count()
    if not pack_attempted:
        (
            current_text,
            current_report,
            current_goal,
            current_scores,
            pack_rounds,
            global_best_candidate,
            pack_accepted,
        ) = _run_safe_band_evidence_pack_attempt(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_report=current_report,
            current_goal=current_goal,
            current_scores=current_scores,
            gateway=gateway,
            output_dir=output_dir / "pack",
            global_best_candidate=global_best_candidate,
            progress_callback=progress_callback,
            progress_percent=progress_percent,
            accepted_checkpoint_callback=accepted_checkpoint_callback,
            started_at=started_at,
            max_seconds=max_seconds,
            author_proxy_context=author_proxy_context,
        )
        rounds.extend(pack_rounds)
        if pack_accepted:
            return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate

    for round_index, section in enumerate(sections, start=1):
        if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="safe_band_evidence_repair",
                round_index=round_index,
                started_at=started_at,
                max_seconds=max_seconds,
                current_scores=current_scores,
            ))
            break
        round_dir = output_dir / f"section_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        _emit_progress(progress_callback, progress_percent, f"V5 safe-band evidence repair paragraph {round_index}")
        variants, llm_diagnostics, prompt, completion = generate_safe_band_evidence_repair_variants(
            section=section,
            current_scores=current_scores,
            current_goal=current_goal,
            gateway=gateway,
            variant_count=variant_count,
            author_proxy_context=author_proxy_context,
        )
        (round_dir / "safe_band_evidence_repair_prompt.json.txt").write_text(prompt)
        (round_dir / "safe_band_evidence_repair_completion.json.txt").write_text(completion)
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
                label=f"safe_band_evidence_repair_s{round_index}_{variant.variant_id}",
                author_proxy_context=author_proxy_context,
                author_proxy_phase="safe_band_evidence_repair",
            )
            for variant in variants
        ]
        for row in rows:
            row["safe_band_evidence_materiality"] = _safe_band_evidence_repair_materiality(
                source_text=section.text,
                candidate_text=str(row.get("text") or ""),
                target_sentence=str((section.metadata or {}).get("target_sentence") or ""),
            )
            _attach_safe_band_quality_materiality(row, current_text=current_text)
        selected = _best_safe_band_evidence_repair_candidate(rows, current_scores=current_scores)
        accepted = selected if selected and _has_safe_band_evidence_repair_movement(selected, current_scores=current_scores) else None
        round_payload = {
            "round": round_index,
            "phase": "safe_band_evidence_repair",
            "status": "accepted" if accepted else "skipped",
            "reason": "accepted_safe_band_evidence_repair_movement" if accepted else "no_safe_band_evidence_repair_movement",
            "section": section.to_dict(),
            "generator_diagnostics": llm_diagnostics,
            "current_scores": current_scores,
            "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
            "candidates": [_compact_residual_row(row) for row in rows],
            "selected": _compact_residual_row(selected),
            "accepted": _compact_residual_row(accepted),
        }
        rounds.append(round_payload)
        (round_dir / "round_result.json").write_text(json.dumps(round_payload, ensure_ascii=False, indent=2))
        global_best_candidate = _best_full_document_candidate([global_best_candidate, *rows])
        if not accepted:
            continue

        current_text, current_report, current_goal, current_scores = _accepted_state(
            accepted=accepted,
            original_text=original_text,
            baseline_report=baseline_report,
        )
        (output_dir / "after_safe_band_evidence_repair.txt").write_text(current_text)
        if accepted_checkpoint_callback is not None:
            accepted_checkpoint_callback({
                "phase": "safe_band_evidence_repair",
                "round": round_index,
                "reason": "accepted_safe_band_evidence_repair_movement",
                "accepted": accepted,
                "rewritten_document": current_text,
                "scores": current_scores,
                "goal": current_goal,
            })
        break
    return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate


def _run_safe_band_controlled_operation_loop(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_scores: dict[str, Any],
    current_text: str,
    current_report: dict[str, Any],
    current_goal: dict[str, Any],
    current_scores: dict[str, Any],
    output_dir: Path,
    global_best_candidate: dict[str, Any] | None,
    progress_callback: Callable[[int, str], None] | None = None,
    progress_percent: int = 81,
    accepted_checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    started_at: float | None = None,
    max_seconds: float | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, Any]] = []
    round_limit = _safe_band_controlled_operation_round_limit()
    for round_index in range(1, round_limit + 1):
        if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="safe_band_controlled_operation",
                round_index=round_index,
                started_at=started_at,
                max_seconds=max_seconds,
                current_scores=current_scores,
            ))
            break
        if _candidate_strict_safe_band_achieved({"candidate_goal": current_goal, "scores": current_scores}):
            rounds.append({
                "round": round_index,
                "phase": "safe_band_controlled_operation",
                "status": "skipped",
                "reason": "strict_safe_band_already_achieved",
                "current_scores": current_scores,
                "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
            })
            break

        round_dir = output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        _emit_progress(progress_callback, progress_percent, f"V5 safe-band controlled operation {round_index}")
        operation_targets = _safe_band_controlled_operation_targets(
            current_text,
            current_report,
            current_goal,
            current_scores=current_scores,
        )
        operation_variants = _safe_band_controlled_operation_variants(
            current_text=current_text,
            targets=operation_targets,
        )
        if not operation_variants:
            payload = {
                "round": round_index,
                "phase": "safe_band_controlled_operation",
                "status": "skipped",
                "reason": "no_safe_band_controlled_operation_targets",
                "targets": operation_targets,
                "current_scores": current_scores,
                "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
                "candidates": [],
                "selected": None,
                "accepted": None,
            }
            rounds.append(payload)
            (round_dir / "round_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            break

        operation_rows = [
            _score_safe_band_controlled_operation_variant(
                original_text=original_text,
                baseline_report=baseline_report,
                baseline_scores=baseline_scores,
                current_text=current_text,
                current_scores=current_scores,
                variant=variant,
                output_dir=round_dir,
                label=f"safe_band_controlled_operation_r{round_index}_{variant.get('variant_id')}",
                author_proxy_context=author_proxy_context,
            )
            for variant in operation_variants
        ]
        for row in operation_rows:
            _attach_safe_band_quality_materiality(row, current_text=current_text)
        selected = _best_safe_band_evidence_repair_candidate(operation_rows, current_scores=current_scores)
        accepted = selected if selected and _has_safe_band_evidence_repair_movement(selected, current_scores=current_scores) else None
        payload = {
            "round": round_index,
            "phase": "safe_band_controlled_operation",
            "status": "accepted" if accepted else "skipped",
            "reason": "accepted_safe_band_controlled_operation" if accepted else "no_safe_band_controlled_operation_movement",
            "targets": operation_targets,
            "current_scores": current_scores,
            "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
            "candidates": [_compact_residual_row(row) for row in operation_rows],
            "selected": _compact_residual_row(selected),
            "accepted": _compact_residual_row(accepted),
        }
        rounds.append(payload)
        (round_dir / "round_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        global_best_candidate = _best_full_document_candidate([global_best_candidate, *operation_rows])
        if not accepted:
            break

        current_text, current_report, current_goal, current_scores = _accepted_state(
            accepted=accepted,
            original_text=original_text,
            baseline_report=baseline_report,
        )
        (output_dir / f"after_round_{round_index:02d}.txt").write_text(current_text)
        if accepted_checkpoint_callback is not None:
            accepted_checkpoint_callback({
                "phase": "safe_band_controlled_operation",
                "round": round_index,
                "reason": "accepted_safe_band_controlled_operation",
                "accepted": accepted,
                "rewritten_document": current_text,
                "scores": current_scores,
                "goal": current_goal,
            })
    return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate


def _run_safe_band_sentence_replacement_loop(
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
    progress_callback: Callable[[int, str], None] | None = None,
    progress_percent: int = 81,
    accepted_checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    started_at: float | None = None,
    max_seconds: float | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, Any]] = []
    round_limit = _safe_band_sentence_replacement_round_limit()
    for round_index in range(1, round_limit + 1):
        if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="safe_band_sentence_replacement",
                round_index=round_index,
                started_at=started_at,
                max_seconds=max_seconds,
                current_scores=current_scores,
            ))
            break
        if _candidate_strict_safe_band_achieved({"candidate_goal": current_goal, "scores": current_scores}):
            rounds.append({
                "round": round_index,
                "phase": "safe_band_sentence_replacement",
                "status": "skipped",
                "reason": "strict_safe_band_already_achieved",
                "current_scores": current_scores,
                "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
            })
            break

        targets = _safe_band_controlled_operation_targets(
            current_text,
            current_report,
            current_goal,
            current_scores=current_scores,
        )[:_safe_band_sentence_replacement_target_limit()]
        if not targets:
            payload = {
                "round": round_index,
                "phase": "safe_band_sentence_replacement",
                "status": "skipped",
                "reason": "no_safe_band_sentence_replacement_targets",
                "targets": [],
                "current_scores": current_scores,
                "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
                "candidates": [],
                "selected": None,
                "accepted": None,
            }
            rounds.append(payload)
            (output_dir / f"round_{round_index:02d}_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            break

        round_dir = output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        _emit_progress(progress_callback, progress_percent, f"V5 safe-band sentence replacement {round_index}")
        variants, llm_diagnostics, prompt, completion = generate_safe_band_sentence_replacement_variants(
            current_text=current_text,
            current_scores=current_scores,
            current_goal=current_goal,
            targets=targets,
            gateway=gateway,
            variant_count=_safe_band_sentence_replacement_variant_count(),
            author_proxy_context=author_proxy_context,
        )
        (round_dir / "safe_band_sentence_replacement_prompt.json.txt").write_text(prompt)
        (round_dir / "safe_band_sentence_replacement_completion.json.txt").write_text(completion)
        rows = [
            _score_final_topk_sentence_route_variant(
                original_text=original_text,
                baseline_report=baseline_report,
                baseline_scores=baseline_scores,
                current_text=current_text,
                current_scores=current_scores,
                targets=targets,
                variant=variant,
                output_dir=round_dir,
                label=f"safe_band_sentence_replacement_r{round_index}_{variant.get('variant_id')}",
                author_proxy_context=author_proxy_context,
                author_proxy_phase="safe_band_sentence_replacement",
                require_all_targets=False,
            )
            for variant in variants
        ]
        for row in rows:
            _attach_safe_band_quality_materiality(row, current_text=current_text)
        selected = _best_safe_band_evidence_repair_candidate(rows, current_scores=current_scores)
        accepted = selected if selected and _has_safe_band_evidence_repair_movement(selected, current_scores=current_scores) else None
        payload = {
            "round": round_index,
            "phase": "safe_band_sentence_replacement",
            "status": "accepted" if accepted else "skipped",
            "reason": "accepted_safe_band_sentence_replacement" if accepted else "no_safe_band_sentence_replacement_movement",
            "targets": targets,
            "generator_diagnostics": llm_diagnostics,
            "current_scores": current_scores,
            "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
            "candidates": [_compact_residual_row(row) for row in rows],
            "selected": _compact_residual_row(selected),
            "accepted": _compact_residual_row(accepted),
        }
        rounds.append(payload)
        (round_dir / "round_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        global_best_candidate = _best_full_document_candidate([global_best_candidate, *rows])
        if not accepted:
            break

        current_text, current_report, current_goal, current_scores = _accepted_state(
            accepted=accepted,
            original_text=original_text,
            baseline_report=baseline_report,
        )
        (output_dir / f"after_round_{round_index:02d}.txt").write_text(current_text)
        if accepted_checkpoint_callback is not None:
            accepted_checkpoint_callback({
                "phase": "safe_band_sentence_replacement",
                "round": round_index,
                "reason": "accepted_safe_band_sentence_replacement",
                "accepted": accepted,
                "rewritten_document": current_text,
                "scores": current_scores,
                "goal": current_goal,
            })
    return current_text, current_report, current_goal, current_scores, rounds, global_best_candidate


def _run_safe_band_density_section_repair_loop(
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
    progress_callback: Callable[[int, str], None] | None = None,
    progress_percent: int = 81,
    accepted_checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    started_at: float | None = None,
    max_seconds: float | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, Any]] = []
    round_limit = _safe_band_density_section_repair_round_limit()
    spent_density_section_ranges: set[tuple[int, int]] = set()
    for round_index in range(1, round_limit + 1):
        if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
            rounds.append(_runtime_budget_stop_record(
                phase="safe_band_density_section_repair",
                round_index=round_index,
                started_at=started_at,
                max_seconds=max_seconds,
                current_scores=current_scores,
            ))
            break
        if not _safe_band_density_section_repair_should_run(current_scores=current_scores, current_goal=current_goal):
            rounds.append({
                "round": round_index,
                "phase": "safe_band_density_section_repair",
                "status": "skipped",
                "reason": "density_section_repair_not_needed",
                "current_scores": current_scores,
                "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
            })
            break

        sections = _safe_band_density_section_repair_sections(
            current_text,
            current_report,
            current_goal,
            limit=_safe_band_density_section_repair_section_limit(),
            exclude_ranges=spent_density_section_ranges,
        )
        if not sections:
            payload = {
                "round": round_index,
                "phase": "safe_band_density_section_repair",
                "status": "skipped",
                "reason": "no_safe_band_density_section",
                "sections": [],
                "current_scores": current_scores,
                "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
                "candidates": [],
                "selected": None,
                "accepted": None,
            }
            rounds.append(payload)
            (output_dir / f"round_{round_index:02d}_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            break

        round_dir = output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        all_rows: list[dict[str, Any]] = []
        section_attempts: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        accepted: dict[str, Any] | None = None
        sections_by_id: dict[str, SectionUnit] = {section.section_id: section for section in sections}
        for section_index, section in enumerate(sections, start=1):
            if started_at is not None and _runtime_budget_exhausted(started_at, max_seconds):
                break
            section_dir = round_dir / f"section_{section_index:02d}"
            section_dir.mkdir(parents=True, exist_ok=True)
            _emit_progress(progress_callback, progress_percent, f"V5 safe-band density section repair {round_index}.{section_index}")
            rows: list[dict[str, Any]] = []
            prompt = ""
            completion = ""
            deterministic_variants = _safe_band_density_section_repair_deterministic_variants(section)
            llm_diagnostics: dict[str, Any] = {
                "status": "not_requested",
                "reason": "no_deterministic_density_repair_candidate",
                "deterministic_variant_count": len(deterministic_variants),
            }
            if deterministic_variants:
                rows.extend([
                    _score_residual_variant(
                        original_text=original_text,
                        baseline_report=baseline_report,
                        baseline_scores=baseline_scores,
                        current_text=current_text,
                        current_scores=current_scores,
                        section=section,
                        variant=variant,
                        output_dir=section_dir,
                        label=f"safe_band_density_section_repair_r{round_index}_s{section_index}_{variant.variant_id}",
                        author_proxy_context=author_proxy_context,
                        author_proxy_phase="safe_band_density_section_repair",
                    )
                    for variant in deterministic_variants
                ])
                for row in rows:
                    materiality = _safe_band_density_section_repair_materiality(
                        source_text=section.text,
                        candidate_text=str(row.get("text") or ""),
                        target_sentence=str((section.metadata or {}).get("target_sentence") or ""),
                    )
                    row["safe_band_evidence_materiality"] = materiality
                    row["safe_band_density_section_materiality"] = materiality
                    _attach_safe_band_quality_materiality(row, current_text=current_text)
                deterministic_selected = _best_safe_band_density_section_candidate(rows, current_scores=current_scores)
                if deterministic_selected and _has_density_safe_band_checkpoint_movement(deterministic_selected, current_scores=current_scores):
                    llm_diagnostics = {
                        "status": "skipped",
                        "reason": "deterministic_density_repair_candidate_accepted",
                        "deterministic_variant_count": len(deterministic_variants),
                    }
                else:
                    llm_diagnostics = {
                        "status": "requested",
                        "reason": "deterministic_density_repair_candidate_not_accepted",
                        "deterministic_variant_count": len(deterministic_variants),
                    }
            if llm_diagnostics.get("status") != "skipped":
                variants, generated_diagnostics, prompt, completion = generate_safe_band_density_section_repair_variants(
                    section=section,
                    current_scores=current_scores,
                    current_goal=current_goal,
                    gateway=gateway,
                    variant_count=_safe_band_density_section_repair_variant_count(),
                    author_proxy_context=author_proxy_context,
                )
                llm_diagnostics = {
                    **(generated_diagnostics if isinstance(generated_diagnostics, dict) else {}),
                    **llm_diagnostics,
                    "llm_variant_count": len(variants),
                }
                rows.extend([
                    _score_residual_variant(
                        original_text=original_text,
                        baseline_report=baseline_report,
                        baseline_scores=baseline_scores,
                        current_text=current_text,
                        current_scores=current_scores,
                        section=section,
                        variant=variant,
                        output_dir=section_dir,
                        label=f"safe_band_density_section_repair_r{round_index}_s{section_index}_{variant.variant_id}",
                        author_proxy_context=author_proxy_context,
                        author_proxy_phase="safe_band_density_section_repair",
                    )
                    for variant in variants
                ])
            (section_dir / "safe_band_density_section_repair_prompt.json.txt").write_text(prompt)
            (section_dir / "safe_band_density_section_repair_completion.json.txt").write_text(completion)
            for row in rows:
                if isinstance(row.get("safe_band_density_section_materiality"), dict):
                    continue
                materiality = _safe_band_density_section_repair_materiality(
                    source_text=section.text,
                    candidate_text=str(row.get("text") or ""),
                    target_sentence=str((section.metadata or {}).get("target_sentence") or ""),
                )
                row["safe_band_evidence_materiality"] = materiality
                row["safe_band_density_section_materiality"] = materiality
                _attach_safe_band_quality_materiality(row, current_text=current_text)
            section_selected = _best_safe_band_density_section_candidate(rows, current_scores=current_scores)
            section_accepted = (
                section_selected
                if section_selected and _has_density_safe_band_checkpoint_movement(section_selected, current_scores=current_scores)
                else None
            )
            section_attempts.append({
                "section_index": section_index,
                "section": section.to_dict(),
                "generator_diagnostics": llm_diagnostics,
                "candidate_count": len(rows),
                "selected": _compact_residual_row(section_selected),
                "accepted": _compact_residual_row(section_accepted),
            })
            all_rows.extend(rows)
        selected = _best_safe_band_density_section_candidate(all_rows, current_scores=current_scores)
        accepted = selected if selected and _has_density_safe_band_checkpoint_movement(selected, current_scores=current_scores) else None
        accepted_section = (
            sections_by_id.get(str(accepted.get("section_id") or ""))
            if isinstance(accepted, dict)
            else None
        )
        payload = {
            "round": round_index,
            "phase": "safe_band_density_section_repair",
            "status": "accepted" if accepted else "skipped",
            "reason": "accepted_safe_band_density_section_repair" if accepted else "no_safe_band_density_section_movement",
            "section": accepted_section.to_dict() if accepted_section else (sections[0].to_dict() if sections else None),
            "sections": [section.to_dict() for section in sections],
            "excluded_section_ranges": [
                {"start_char": start, "end_char": end}
                for start, end in sorted(spent_density_section_ranges)
            ],
            "section_attempts": section_attempts,
            "candidate_section_count": len(sections),
            "current_scores": current_scores,
            "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
            "candidates": [_compact_residual_row(row) for row in all_rows],
            "selected": _compact_residual_row(selected),
            "accepted": _compact_residual_row(accepted),
        }
        rounds.append(payload)
        (round_dir / "round_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        global_best_candidate = _best_full_document_candidate([global_best_candidate, *all_rows])
        if not accepted:
            break

        if accepted_section is not None:
            spent_density_section_ranges.add(_safe_band_section_range_signature(accepted_section))
        current_text, current_report, current_goal, current_scores = _accepted_state(
            accepted=accepted,
            original_text=original_text,
            baseline_report=baseline_report,
        )
        (output_dir / f"after_round_{round_index:02d}.txt").write_text(current_text)
        if accepted_checkpoint_callback is not None:
            accepted_checkpoint_callback({
                "phase": "safe_band_density_section_repair",
                "round": round_index,
                "reason": "accepted_safe_band_density_section_repair",
                "accepted": accepted,
                "rewritten_document": current_text,
                "scores": current_scores,
                "goal": current_goal,
            })
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


def generate_final_topk_sentence_route_variants(
    *,
    current_text: str,
    current_scores: dict[str, Any],
    current_goal: dict[str, Any] | None = None,
    targets: list[dict[str, Any]],
    gateway: LLMGateway,
    variant_count: int,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    variants = max(1, min(5, int(variant_count or 1)))
    prompt = build_final_topk_sentence_route_prompt(
        current_scores=current_scores,
        current_goal=current_goal,
        targets=targets,
        variant_count=variants,
        author_proxy_context=author_proxy_context,
    )
    structured = structured_json_request_options(
        getattr(gateway, "model", None),
        _final_topk_sentence_route_response_format(variants, len(targets)),
    )
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured.get("provider"))
    started = time.monotonic()
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a variants array.",
        response_format=structured.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_TEMPERATURE", 0.58, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_TOP_P", 0.92, minimum=0.1, maximum=1.0),
        max_tokens=_int_env("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_MAX_TOKENS", 3500, minimum=1200, maximum=9000),
    )
    elapsed = time.monotonic() - started
    raw = response.raw_content or response.content
    parsed, diagnostics = parse_json_object(raw, required_keys={"variants"})
    if parsed is None:
        return [], {
            **diagnostics,
            "model": response.model,
            "provider": response.raw.get("provider"),
            "usage": response.usage,
            "finish_reason": response.finish_reason,
            "native_finish_reason": response.native_finish_reason,
            "structured_output_mode": structured.get("structured_output_mode"),
            "elapsed_seconds": round(elapsed, 3),
        }, prompt, raw
    rows = _sanitize_final_topk_sentence_route_variants(parsed.get("variants"), targets=targets)
    return rows, {
        **diagnostics,
        "status": "ok",
        "valid_variant_count": len(rows),
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "structured_output_mode": structured.get("structured_output_mode"),
        "elapsed_seconds": round(elapsed, 3),
    }, prompt, raw


def generate_safe_band_sentence_replacement_variants(
    *,
    current_text: str,
    current_scores: dict[str, Any],
    current_goal: dict[str, Any] | None = None,
    targets: list[dict[str, Any]],
    gateway: LLMGateway,
    variant_count: int,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    variants = max(1, min(5, int(variant_count or 1)))
    prompt = build_final_topk_sentence_route_prompt(
        current_scores=current_scores,
        current_goal=current_goal,
        targets=targets,
        variant_count=variants,
        author_proxy_context=author_proxy_context,
    )
    structured = structured_json_request_options(
        getattr(gateway, "model", None),
        _final_topk_sentence_route_response_format(variants, len(targets)),
    )
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured.get("provider"))
    started = time.monotonic()
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with safe-band sentence replacement variants.",
        response_format=structured.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_SAFE_BAND_SENTENCE_REPLACEMENT_TEMPERATURE", 0.25, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_SAFE_BAND_SENTENCE_REPLACEMENT_TOP_P", 0.82, minimum=0.1, maximum=1.0),
        max_tokens=_int_env("DRAFTPROOF_SAFE_BAND_SENTENCE_REPLACEMENT_MAX_TOKENS", 6500, minimum=1600, maximum=10000),
    )
    elapsed = time.monotonic() - started
    raw = response.raw_content or response.content
    parsed, diagnostics = parse_json_object(raw, required_keys={"variants"})
    if parsed is None:
        return [], {
            **diagnostics,
            "model": response.model,
            "provider": response.raw.get("provider"),
            "usage": response.usage,
            "finish_reason": response.finish_reason,
            "native_finish_reason": response.native_finish_reason,
            "structured_output_mode": structured.get("structured_output_mode"),
            "elapsed_seconds": round(elapsed, 3),
        }, prompt, raw
    rows = _sanitize_final_topk_sentence_route_variants(parsed.get("variants"), targets=targets)
    return rows, {
        **diagnostics,
        "status": "ok",
        "valid_variant_count": len(rows),
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "structured_output_mode": structured.get("structured_output_mode"),
        "elapsed_seconds": round(elapsed, 3),
    }, prompt, raw


def build_final_topk_sentence_route_prompt(
    *,
    current_scores: dict[str, Any],
    current_goal: dict[str, Any] | None = None,
    targets: list[dict[str, Any]],
    variant_count: int,
    author_proxy_context: dict[str, Any] | None = None,
) -> str:
    payload = {
        "task": "final_topk_sentence_route_replacement",
        "objective": (
            "Treat exact high top-k sentences by changing sentence route, not just words. "
            "When a safe-band KPI contract is present, reduce the combined safe-band gap rather than only lowering top-k. "
            "Do not rewrite unrelated sentences."
        ),
        "current_scores": {
            key: current_scores.get(key)
            for key in (
                "ai",
                "topk",
                "topk_calibrated_risk",
                "qualifying_text_ai_density",
                "ai_authorship",
                "external",
                "unsafe_cluster_count",
                "risky_window_count",
            )
        },
        "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
        "target_sentences": targets,
        "method": (
            "For each sentence, identify the sentence job, describe the current route, "
            "then write a same-meaning replacement with a different route."
        ),
        "safe_band_replacement_method": [
            "Treat kpi_contract.gaps as the acceptance target for this late-stage pass.",
            "If qualifying_text_ai_density remains above target, avoid adding smooth explanatory filler or broad academic closure.",
            "Use kpi_contract.secondary_density_drivers to identify why density remains high before writing replacements.",
            "When generic_assertion_risk, unsupported_claim_risk, broad_claim_risk, or source_grounding_risk is high, replace broad claims with narrower source-owned observations, named teaching actions, or explicit limits.",
            "Prefer shorter evidence-linked clauses, concrete source anchors, and practical action over abstract summary.",
            "Do not repeat an idea already stated in the target sentence's before_context or after_context; merge or narrow instead.",
            "Do not trade a small top-k gain for higher AI, higher authorship risk, or higher qualifying density.",
        ],
        "operators": [
            "CLAUSE_ROUTE_CHANGE",
            "ABSTRACT_TO_ACTION",
            "LIST_TO_SPECIFIC_CONCERN",
            "BRIDGE_DELETE_OR_MERGE",
            "SENTENCE_SPLIT",
            "CONCRETE_STUDENT_OR_TEACHER_ACTION",
            "ENDING_DEPREDICT",
        ],
        "rules": [
            "Use every target_id exactly once in each variant.",
            "Preserve the meaning of each sentence.",
            "Do not add fake citations, statistics, dates, named events, or personal anecdotes.",
            "Do not make the writing more academic.",
            "Avoid synonym replacement as the main method.",
            "Prefer plain wording over polished wording.",
            "Avoid neat abstract list endings.",
            "Avoid not only/but also structure.",
            "The after text can be one or two sentences if that breaks the route better.",
            "Use the kpi_contract targets as the primary late-stage success contract.",
            "Prefer grounded specificity from the submitted content over surface-level humanizing.",
            "Use target_sentences[].context to preserve paragraph logic and avoid isolated sentence polishing.",
            "If kpi_contract.gaps.qualifying_text_ai_density is above zero, the variant should aim to lower qualifying density without increasing AI/authorship risk.",
            "Reject self-repetition: do not restate the same source fact, classroom action, named framework, or student observation twice in nearby sentences.",
        ],
        "variant_plan": [
            {"variant_id": f"v{index}", "goal": _safe_band_sentence_replacement_variant_goal(index)}
            for index in range(1, max(1, min(5, int(variant_count or 1))) + 1)
        ],
        "output_schema": {
            "variants": [
                {
                    "variant_id": "v1",
                    "repairs": [
                        {
                            "target_id": "t001",
                            "sentence_job": "bridge",
                            "current_route": "old route",
                            "repair_route": "new route",
                            "after": "replacement",
                        }
                    ],
                }
            ]
        },
    }
    _attach_author_proxy_context(payload, author_proxy_context)
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_safe_band_density_section_repair_variants(
    *,
    section: SectionUnit,
    current_scores: dict[str, Any],
    current_goal: dict[str, Any] | None,
    gateway: LLMGateway,
    variant_count: int,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    variants = max(1, min(5, int(variant_count or 1)))
    prompt = build_safe_band_density_section_repair_prompt(
        section=section,
        current_scores=current_scores,
        current_goal=current_goal,
        variant_count=variants,
        author_proxy_context=author_proxy_context,
    )
    return _generate_loose_variants(
        prompt=prompt,
        gateway=gateway,
        variant_count=variants,
        max_tokens=_int_env(
            "DRAFTPROOF_SAFE_BAND_DENSITY_SECTION_REPAIR_MAX_TOKENS",
            5200,
            minimum=1600,
            maximum=10000,
        ),
    )


def build_safe_band_density_section_repair_prompt(
    *,
    section: SectionUnit,
    current_scores: dict[str, Any],
    current_goal: dict[str, Any] | None,
    variant_count: int,
    author_proxy_context: dict[str, Any] | None = None,
) -> str:
    variants = max(1, min(5, int(variant_count or 1)))
    metadata = section.metadata if isinstance(section.metadata, dict) else {}
    min_word_ratio = _safe_band_density_section_repair_min_word_ratio_for_text(section.text)
    min_word_count = max(1, round(section.word_count * min_word_ratio))
    max_word_count = max(min_word_count, round(section.word_count * _safe_band_density_section_repair_max_word_ratio()))
    contextual_anchor_contract = _safe_band_density_contextual_anchor_contract(section.text)
    payload = {
        "task": "safe_band_density_section_repair",
        "architecture_stage": "late_density_only_author_proxy_section_repair",
        "objective": (
            "Replace this selected paragraph or window as the author-proxy using submitted content only. "
            "The purpose is to reduce the remaining qualifying_text_ai_density safe-band gap by improving grounded authorship, "
            "not by surface paraphrase, word spinning, detector bypass, or runtime repetition."
        ),
        "single_product_judge": {
            "judge": "DraftProof internal safe-band and authorship-integrity scoring",
            "ignored_as_acceptance_judge": "external AI flag score",
            "reason": "The section repair must improve the internal grounded-authorship blockers that remain after top-k and unsafe clusters are clear.",
        },
        "density_only_trigger": {
            "topk_already_safe": _safe_band_kpi_contract(current_scores, current_goal).get("gaps", {}).get("topk_calibrated_risk") == 0,
            "remaining_target": "qualifying_text_ai_density",
            "ai_authorship_must_not_increase": True,
            "do_not_optimize_for": "external_ai_flag_risk",
        },
        "section": {
            "section_id": section.section_id,
            "heading": section.heading,
            "source_text": section.text,
            "source_word_count": section.word_count,
            "paragraph_count": section.paragraph_count,
            "before_context": metadata.get("before_context"),
            "after_context": metadata.get("after_context"),
            "selection_reason": metadata.get("selection_reason"),
            "target_sentence": metadata.get("target_sentence"),
            "scanner_focus": metadata.get("scanner_focus"),
            "source_voice_profile": _safe_band_density_source_voice_profile(section.text),
            "contextual_anchor_contract": contextual_anchor_contract,
        },
        "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
        "method": [
            "Diagnose why this section still reads as qualifying-density heavy: broad claim, unsupported certainty, smooth generic closure, source-grounding gap, or repeated idea.",
            "Rebuild the section around author-owned evidence already present in source_text, before_context, after_context, citations, named anchors, or technical terms.",
            "Where support is missing, narrow the claim or mark the gap for author review; do not fill it with invented evidence.",
            "Keep the paragraph's argument job and author viewpoint, but change the route at section level rather than polishing individual sentences.",
            "Prefer specific teaching action, observed constraint, source limit, or practical consequence over abstract summary.",
        ],
        "materiality_gate": {
            "minimum_changed_source_sentences": _safe_band_density_section_repair_min_changed_sentences(section.text),
            "minimum_changed_source_sentence_ratio": _safe_band_density_section_repair_min_changed_sentence_ratio(),
            "minimum_word_ratio": min_word_ratio,
            "maximum_word_ratio": _safe_band_density_section_repair_max_word_ratio(),
            "minimum_word_count": min_word_count,
            "maximum_word_count": max_word_count,
            "contextual_anchor_contract": contextual_anchor_contract,
            "reject_if": [
                "the replacement only changes one target sentence while leaving the rest of the section route intact",
                "the replacement repeats the same source fact, framework, or conclusion in nearby sentences",
                "the replacement compresses coverage below minimum_word_ratio",
                "the replacement is shorter than minimum_word_count or longer than maximum_word_count",
                "the replacement drops source voice markers such as first-person framing or contractions that are present in source_text",
                "the replacement raises ai_authorship or turns the author's voice into detached polished academic narration",
                "the replacement adds new concrete references that are not already in submitted content",
                "the replacement merely swaps words while retaining the same broad academic route",
                "the replacement leaves low-context qualifying sentences abstract when submitted author/domain/source context is available",
            ],
        },
        "rules": [
            "Return replacement text for section.source_text only, not the whole document.",
            "Preserve citations, named people, technical codes, quoted wording, and concrete source anchors.",
            "Do not add fake citations, dates, statistics, institutions, named events, personal experiences, or classroom observations.",
            "Do not make the prose more polished, generic, abstract, or template-like.",
            f"Change at least {_safe_band_density_section_repair_min_changed_sentences(section.text)} source sentences; do not leave the opening, bridge, and closing route all intact.",
            "Keep the author's first-person framing when it exists in the source; do not convert it into detached report language.",
            "Follow section.source_voice_profile; preserve the submitted voice markers while improving clarity and grounding.",
            "Do not use a paraphrase-only rewrite, word spinner style, deliberate errors, slang, or decorative humanizing noise.",
            "Do not repeat the same source fact, classroom action, framework explanation, or teaching implication twice.",
            f"Each replacement must stay between {min_word_count} and {max_word_count} words.",
            (
                "Use section.contextual_anchor_contract: add submitted author/domain/source context to at least "
                f"{contextual_anchor_contract.get('additional_contextual_sentences_needed')} low-context qualifying sentence(s) where possible, "
                "without inventing facts or adding bracketed placeholders."
            ),
            "A contextual anchor means a concrete action, technical/domain mechanic, source relation, practical constraint, or observed process already present in source_text or nearby context.",
            "If a claim lacks support, narrow it and include an author_review_items entry instead of inventing support.",
        ],
        "writer_variant_plan": [
            {
                "variant_id": f"v{index}",
                "goal": _safe_band_density_section_repair_variant_goal(index),
            }
            for index in range(1, variants + 1)
        ],
        "output_schema": {
            "variants": [
                _author_proxy_output_variant_template()
                if _author_proxy_active(author_proxy_context)
                else {"variant_id": f"v{index}", "text": "replacement paragraph/window only"}
                for index in range(1, variants + 1)
            ]
        },
    }
    _attach_author_proxy_context(payload, author_proxy_context)
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_safe_band_evidence_repair_variants(
    *,
    section: SectionUnit,
    current_scores: dict[str, Any],
    current_goal: dict[str, Any] | None,
    gateway: LLMGateway,
    variant_count: int,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    return _generate_loose_variants_from_builder(
        prompt_builder=lambda count: build_safe_band_evidence_repair_prompt(
            section=section,
            current_scores=current_scores,
            current_goal=current_goal,
            variant_count=count,
            author_proxy_context=author_proxy_context,
        ),
        gateway=gateway,
        variant_count=variant_count,
    )


def generate_safe_band_author_proxy_revision_plan(
    *,
    sections: list[SectionUnit],
    current_scores: dict[str, Any],
    current_goal: dict[str, Any] | None,
    gateway: LLMGateway,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any], str, str]:
    prompt = build_safe_band_author_proxy_revision_plan_prompt(
        sections=sections,
        current_scores=current_scores,
        current_goal=current_goal,
        author_proxy_context=author_proxy_context,
    )
    structured = structured_json_request_options(
        getattr(gateway, "model", None),
        _safe_band_author_proxy_revision_plan_response_format(len(sections)),
    )
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured.get("provider"))
    started = time.monotonic()
    response = gateway.chat(
        prompt,
        system="Return only valid JSON matching the requested evidence ledger and revision plan schema.",
        response_format=structured.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_SAFE_BAND_AUTHOR_PROXY_PLAN_TEMPERATURE", 0.18, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_SAFE_BAND_AUTHOR_PROXY_PLAN_TOP_P", 0.82, minimum=0.1, maximum=1.0),
        max_tokens=_int_env("DRAFTPROOF_SAFE_BAND_AUTHOR_PROXY_PLAN_MAX_TOKENS", 5200, minimum=1600, maximum=10000),
    )
    elapsed = time.monotonic() - started
    raw = response.raw_content or response.content
    parsed, diagnostics = parse_json_object(raw, required_keys={"evidence_ledger", "revision_plan"})
    plan, parse_diagnostics = _sanitize_safe_band_author_proxy_revision_plan(parsed, sections=sections)
    return plan, {
        **diagnostics,
        **parse_diagnostics,
        "status": "ok" if plan is not None else "schema_failed",
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "structured_output_mode": structured.get("structured_output_mode"),
        "elapsed_seconds": round(elapsed, 3),
    }, prompt, raw


def build_safe_band_author_proxy_revision_plan_prompt(
    *,
    sections: list[SectionUnit],
    current_scores: dict[str, Any],
    current_goal: dict[str, Any] | None,
    author_proxy_context: dict[str, Any] | None = None,
) -> str:
    section_rows = [_safe_band_pack_section_payload(section, index=index) for index, section in enumerate(sections, start=1)]
    density_section_count = sum(1 for section in sections if _section_uses_density_safe_band_contract(section))
    payload = {
        "task": "safe_band_author_proxy_evidence_ledger_and_revision_plan",
        "architecture_stage": "evidence_ledger_then_revision_plan_before_writer",
        "objective": (
            "Plan an author-owned revision pack before any rewriting. The plan must identify usable submitted evidence, "
            "content gaps, and concrete section-level moves that will let the writer produce high-quality replacements "
            "without paraphrase-only or bypass-style edits."
        ),
        "single_product_judge": {
            "judge": "DraftProof internal safe-band and authorship-integrity scoring",
            "ignored_as_acceptance_judge": "external AI flag score",
            "reason": "The planner must improve grounded authorship and safe-band blockers, not optimize for another detector.",
        },
        "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
        "required_section_ids": [row["section_id"] for row in section_rows],
        "reconstruction_mode": {
            "mode": "density_blocker_author_proxy_pack" if density_section_count else "safe_band_author_proxy_pack",
            "density_section_count": density_section_count,
            "reason": (
                "qualifying_text_ai_density remains the main blocker; coordinate enough section-level route changes to move the whole-document density score."
                if density_section_count
                else "Coordinate selected safe-band repairs into one author-owned revision pack."
            ),
        },
        "sections": section_rows,
        "planner_rules": [
            "Use only submitted source_text, before_context, after_context, citations, and named anchors from the selected sections.",
            "Do not invent classroom observations, student behavior, citations, dates, institutions, statistics, or personal experiences.",
            "Prefer narrowing unsupported claims over filling them with invented evidence.",
            "For each section, require enough concrete route change to pass materiality_gate.minimum_changed_source_sentences.",
            "When materiality_gate.contract is density_section_repair, plan a section-level route rebuild that preserves coverage and voice while lowering generic density.",
            "Use the exact section_id values from required_section_ids; do not invent generic IDs such as safe_band_evidence_repair_t001.",
            "Make the writer's job explicit: which sentences to rebuild, which source evidence to retain, which claims to narrow, and what must be marked for author review.",
        ],
        "output_schema": {
            "evidence_ledger": {
                "sections": [
                    *[
                    {
                        "section_id": row["section_id"],
                        "author_owned_evidence": ["submitted fact, example, citation, classroom action, or named anchor"],
                        "weak_or_generic_claims": ["claim that currently reads broad, generic, or under-evidenced"],
                        "protected_anchors": ["citation, technical code, named person, quoted wording, or fact that must remain"],
                        "author_review_gaps": ["missing author-owned detail to verify later"],
                    }
                    for row in section_rows
                    ]
                ]
            },
            "revision_plan": [
                *[
                {
                    "section_id": row["section_id"],
                    "section_job": "what this section must do in the argument",
                    "required_moves": ["specific route and evidence moves the writer must perform"],
                    "sentences_to_rebuild": ["target or surrounding source sentence to materially rebuild"],
                    "claim_narrowing": ["unsupported claim to narrow, or empty"],
                    "materiality_requirement": "change target route and at least minimum_changed_source_sentences",
                    "author_review_items": ["review obligation, or empty"],
                }
                for row in section_rows
                ]
            ],
        },
    }
    _attach_author_proxy_context(payload, author_proxy_context)
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_safe_band_evidence_repair_prompt(
    *,
    section: SectionUnit,
    current_scores: dict[str, Any],
    current_goal: dict[str, Any] | None,
    variant_count: int,
    author_proxy_context: dict[str, Any] | None = None,
) -> str:
    variants = max(1, min(5, int(variant_count or 1)))
    metadata = section.metadata if isinstance(section.metadata, dict) else {}
    payload = {
        "task": "safe_band_evidence_repair",
        "objective": (
            "Replace the selected paragraph/window with a higher-quality author-grounded revision "
            "that reduces the remaining strict safe-band blockers without using paraphrase-only or bypass-style edits."
        ),
        "section": {
            "section_id": section.section_id,
            "heading": section.heading,
            "source_text": section.text,
            "source_word_count": section.word_count,
            "paragraph_count": section.paragraph_count,
            "before_context": metadata.get("before_context"),
            "after_context": metadata.get("after_context"),
            "selection_reason": metadata.get("selection_reason"),
            "target_sentence": metadata.get("target_sentence"),
            "scanner_focus": metadata.get("scanner_focus"),
        },
        "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
        "method": [
            "Treat this as paragraph evidence repair, not sentence polishing.",
            "Find the paragraph's actual job in the submitted draft, then rebuild the route around author-owned detail already present in source_text or nearby context.",
            "Replace generic claims with narrower, concrete, evidence-linked wording when the draft supports it.",
            "If the needed detail is missing, narrow the claim and mark the gap for author review instead of inventing support.",
            "Keep citations, quotations, named people, technical codes, classroom events, and source anchors intact.",
        ],
        "materiality_gate": {
            "minimum_changed_source_sentences": _safe_band_evidence_repair_min_changed_sentences(section.text),
            "target_sentence_must_change_route": bool(metadata.get("target_sentence")),
            "reject_if": [
                "the only change is punctuation, connector replacement, or a synonym swap",
                "the target sentence still follows the same subject -> generic verb -> list route",
                "fewer than the minimum_changed_source_sentences are materially rebuilt",
            ],
            "required_move": (
                "Rebuild the paragraph route by changing the target sentence and at least one surrounding sentence. "
                "Keep source meaning, but change how the evidence is introduced, connected, and limited."
            ),
        },
        "rules": [
            "Return replacement text for section.source_text only, not the whole document.",
            "Preserve factual meaning, paragraph role, citations, protected terms, and author viewpoint.",
            "Do not add fake citations, dates, statistics, institutions, named events, personal experiences, or observations.",
            "Do not compress the paragraph into a summary.",
            "Do not make the paragraph more polished, generic, or template-like.",
            "Do not introduce slang, deliberate errors, or decorative humanizing noise.",
            "Keep the replacement near the source word count unless a slight expansion is needed to preserve evidence.",
            "Do not return a near-copy of section.source_text; the replacement must pass materiality_gate.",
        ],
        "variant_plan": [
            {
                "variant_id": f"v{index}",
                "goal": _safe_band_evidence_repair_variant_goal(index),
            }
            for index in range(1, variants + 1)
        ],
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "replacement paragraph/window only"}
                for index in range(1, variants + 1)
            ]
        },
    }
    _attach_author_proxy_context(payload, author_proxy_context)
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_safe_band_evidence_pack_variants(
    *,
    sections: list[SectionUnit],
    current_scores: dict[str, Any],
    current_goal: dict[str, Any] | None,
    gateway: LLMGateway,
    variant_count: int,
    revision_plan: dict[str, Any] | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    variants = max(1, min(5, int(variant_count or 1)))
    prompt = build_safe_band_evidence_pack_prompt(
        sections=sections,
        current_scores=current_scores,
        current_goal=current_goal,
        variant_count=variants,
        revision_plan=revision_plan,
        author_proxy_context=author_proxy_context,
    )
    structured = structured_json_request_options(
        getattr(gateway, "model", None),
        _safe_band_evidence_pack_response_format(
            variants,
            len(sections),
            include_author_proxy_fields=_prompt_author_proxy_active(prompt),
        ),
    )
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured.get("provider"))
    started = time.monotonic()
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a variants array.",
        response_format=structured.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_SAFE_BAND_EVIDENCE_PACK_TEMPERATURE", 0.42, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_SAFE_BAND_EVIDENCE_PACK_TOP_P", 0.9, minimum=0.1, maximum=1.0),
        max_tokens=_int_env("DRAFTPROOF_SAFE_BAND_EVIDENCE_PACK_MAX_TOKENS", 8000, minimum=1600, maximum=12000),
    )
    elapsed = time.monotonic() - started
    raw = response.raw_content or response.content
    parsed, diagnostics = parse_json_object(raw, required_keys={"variants"})
    if parsed is None:
        return [], {
            **diagnostics,
            "model": response.model,
            "provider": response.raw.get("provider"),
            "usage": response.usage,
            "finish_reason": response.finish_reason,
            "native_finish_reason": response.native_finish_reason,
            "structured_output_mode": structured.get("structured_output_mode"),
            "elapsed_seconds": round(elapsed, 3),
        }, prompt, raw
    rows, parse_diagnostics = _sanitize_safe_band_evidence_pack_variants(parsed.get("variants"), sections=sections)
    return rows, {
        **diagnostics,
        **parse_diagnostics,
        "status": "ok" if rows else "schema_failed",
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "structured_output_mode": structured.get("structured_output_mode"),
        "elapsed_seconds": round(elapsed, 3),
    }, prompt, raw


def build_safe_band_evidence_pack_prompt(
    *,
    sections: list[SectionUnit],
    current_scores: dict[str, Any],
    current_goal: dict[str, Any] | None,
    variant_count: int,
    revision_plan: dict[str, Any] | None = None,
    author_proxy_context: dict[str, Any] | None = None,
) -> str:
    variants = max(1, min(5, int(variant_count or 1)))
    section_rows = [_safe_band_pack_section_payload(section, index=index) for index, section in enumerate(sections, start=1)]
    plan = revision_plan if isinstance(revision_plan, dict) else {}
    density_section_count = sum(1 for section in sections if _section_uses_density_safe_band_contract(section))
    density_hard_requirements = [
        {
            "section_id": row["section_id"],
            "minimum_changed_source_sentences": row.get("materiality_gate", {}).get("minimum_changed_source_sentences"),
            "minimum_word_count": row.get("materiality_gate", {}).get("minimum_word_count"),
            "maximum_word_count": row.get("materiality_gate", {}).get("maximum_word_count"),
        }
        for row in section_rows
        if row.get("materiality_gate", {}).get("contract") == "density_section_repair"
    ]
    payload = {
        "task": "safe_band_evidence_multi_replacement_pack",
        "architecture_stage": "author_proxy_writer_from_evidence_ledger_and_revision_plan",
        "objective": (
            "Produce coordinated replacements for all selected sections so DraftProof can score one whole-document candidate. "
            "Do not optimize a single sentence; move the document-level safe-band signals through grounded author evidence."
        ),
        "single_product_judge": {
            "judge": "DraftProof internal safe-band and authorship-integrity scoring",
            "ignored_as_acceptance_judge": "external AI flag score",
            "decision": "Candidates are accepted only by scanner movement, safe-band gap movement, materiality, and authorship integrity.",
        },
        "kpi_contract": _safe_band_kpi_contract(current_scores, current_goal),
        "required_section_ids": [row["section_id"] for row in section_rows],
        "reconstruction_mode": {
            "mode": "density_blocker_author_proxy_pack" if density_section_count else "safe_band_author_proxy_pack",
            "density_section_count": density_section_count,
            "success_shape": (
                "Move qualifying_text_ai_density at document level by rebuilding multiple grounded sections in one candidate."
                if density_section_count
                else "Move the remaining safe-band gap through coordinated grounded replacements."
            ),
        },
        "sections": section_rows,
        "density_pack_hard_rejection_contract": {
            "applies": bool(density_hard_requirements),
            "requirements": density_hard_requirements,
            "rule": (
                "For every listed density section, count the source sentences whose route you materially rebuild. "
                "If the count is below minimum_changed_source_sentences, the whole pack is rejected before scanner scoring."
            ),
            "near_copy_definition": (
                "A sentence does not count as changed when it keeps the same claim order with only synonym swaps, "
                "connector changes, or a short inserted phrase."
            ),
            "word_band_rule": "Each replacement must also stay inside its section-specific minimum_word_count and maximum_word_count.",
        },
        "evidence_ledger": plan.get("evidence_ledger") if isinstance(plan.get("evidence_ledger"), dict) else {},
        "revision_plan": plan.get("revision_plan") if isinstance(plan.get("revision_plan"), list) else [],
        "pack_rules": [
            "Return every section_id exactly once in each variant.",
            "Use the exact section_id values from required_section_ids; do not invent generic IDs.",
            "Each replacement must be only the replacement text for that section.",
            "Follow revision_plan before writing; do not write directly from scanner pressure.",
            "Coordinate the replacements so the document reads as one author-owned revision, not separate paraphrases.",
            "Use submitted material, nearby context, citations, and source anchors only.",
            "Narrow unsupported claims instead of adding new evidence.",
            "Do not add fake citations, dates, statistics, institutions, named events, personal experiences, or classroom details.",
            "Change the target sentence route and at least one surrounding sentence in every section.",
            "For every section, materially rebuild at least the listed minimum_changed_source_sentences; one near-copy section rejects the whole pack.",
            "For density_section_repair sections, use density_pack_hard_rejection_contract as the controlling contract even if revision_plan text is vague or inconsistent.",
            "For density_section_repair sections, deliberately rebuild enough sentence routes to meet the numeric minimum; do not preserve most source sentences unchanged.",
            "For density_section_repair sections, use materiality_gate.contextual_anchor_contract to turn low-context qualifying sentences into grounded author/domain/source-context sentences when the submitted text supports it.",
            "For density_section_repair sections, preserve the section word-count band, voice markers, citations, and source anchors while changing the section route.",
            "For density_section_repair sections, replacement word count must stay within materiality_gate.minimum_word_count and materiality_gate.maximum_word_count.",
            "Do not solve density by compressing coverage, adding decorative noise, or making the prose more detached and polished.",
            "Preserve section facts, citations, named people, technical codes, author stance, and paragraph role.",
            "Keep author_proxy_provenance and author_review_items compact: include only the highest-risk items, maximum four of each per variant.",
        ],
        "author_proxy_writer_method": [
            "Read each section's evidence ledger first.",
            "Choose author-owned details, citations, classroom actions, and constraints already present in source_text or nearby context.",
            "Turn broad claims into narrower claims when evidence is thin.",
            "Rebuild the section route around concrete author evidence rather than synonym substitution.",
            "Keep any provisional bridge visible in author_proxy_provenance or author_review_items.",
        ],
        "variant_plan": [
            {
                "variant_id": f"v{index}",
                "goal": _safe_band_evidence_pack_variant_goal(index),
            }
            for index in range(1, variants + 1)
        ],
        "output_schema": {
            "variants": [
                {
                    "variant_id": "v1",
                    "replacements": [
                        {"section_id": row["section_id"], "text": "replacement for this section only"}
                        for row in section_rows
                    ],
                    "author_proxy_provenance": [],
                    "author_review_items": [],
                }
            ]
        },
    }
    _attach_author_proxy_context(payload, author_proxy_context)
    payload["output_schema"] = {
        "variants": [
            {
                "variant_id": "v1",
                "replacements": [
                    {"section_id": row["section_id"], "text": "replacement for this section only"}
                    for row in section_rows
                ],
                "author_proxy_provenance": [_author_proxy_output_variant_template()["author_proxy_provenance"][0]],
                "author_review_items": [_author_proxy_output_variant_template()["author_review_items"][0]],
            }
        ]
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _safe_band_pack_section_payload(section: SectionUnit, *, index: int) -> dict[str, Any]:
    metadata = section.metadata if isinstance(section.metadata, dict) else {}
    density_contract = _section_uses_density_safe_band_contract(section)
    materiality_gate = {
        "minimum_changed_source_sentences": (
            _safe_band_density_section_repair_min_changed_sentences(section.text)
            if density_contract
            else _safe_band_evidence_repair_min_changed_sentences(section.text)
        ),
        "target_sentence_must_change_route": bool(metadata.get("target_sentence")),
    }
    if density_contract:
        min_word_ratio = _safe_band_density_section_repair_min_word_ratio_for_text(section.text)
        materiality_gate.update({
            "contract": "density_section_repair",
            "minimum_changed_source_sentence_ratio": _safe_band_density_section_repair_min_changed_sentence_ratio(),
            "minimum_word_ratio": min_word_ratio,
            "maximum_word_ratio": _safe_band_density_section_repair_max_word_ratio(),
            "minimum_word_count": max(1, round(section.word_count * min_word_ratio)),
            "maximum_word_count": max(1, round(section.word_count * _safe_band_density_section_repair_max_word_ratio())),
            "source_voice_profile": _safe_band_density_source_voice_profile(section.text),
            "contextual_anchor_contract": _safe_band_density_contextual_anchor_contract(section.text),
        })
    return {
        "section_id": section.section_id,
        "index": index,
        "source_text": section.text,
        "source_word_count": section.word_count,
        "target_sentence": metadata.get("target_sentence"),
        "selection_reason": metadata.get("selection_reason"),
        "before_context": metadata.get("before_context"),
        "after_context": metadata.get("after_context"),
        "scanner_focus": metadata.get("scanner_focus"),
        "materiality_gate": materiality_gate,
    }


def _section_uses_density_safe_band_contract(section: SectionUnit) -> bool:
    metadata = section.metadata if isinstance(section.metadata, dict) else {}
    section_id = str(getattr(section, "section_id", "") or "")
    reason = str(metadata.get("selection_reason") or "")
    unit_type = str(metadata.get("unit_type") or "")
    return (
        section_id.startswith("safe_band_density_section_")
        or "density" in reason
        or "density" in unit_type
        or metadata.get("density_section_expanded") is True
    )


def _safe_band_density_contextual_anchor_contract(text: str) -> dict[str, Any]:
    sentences = [sentence.strip() for sentence in _sentences(str(text or "")) if sentence.strip()]
    qualifying = [
        sentence
        for sentence in sentences
        if _safe_band_density_qualifying_sentence(sentence)
    ]
    contextual = [
        sentence
        for sentence in qualifying
        if _sentence_has_concrete_or_context(sentence)
    ]
    density = len(contextual) / max(1, len(qualifying))
    target = _safe_band_density_contextual_anchor_target()
    needed = max(0, int((target * max(1, len(qualifying))) + 0.999) - len(contextual))
    low_context = [
        sentence
        for sentence in qualifying
        if not _sentence_has_concrete_or_context(sentence)
    ]
    return {
        "schema_version": "safe_band_density_contextual_anchor_contract.v1",
        "qualifying_sentence_count": len(qualifying),
        "contextual_anchor_sentence_count": len(contextual),
        "contextual_anchor_density": round(density, 3),
        "target_contextual_anchor_density": target,
        "additional_contextual_sentences_needed": needed,
        "low_context_sentence_examples": low_context[:4],
        "rule": (
            "When possible, rebuild low-context qualifying sentences around submitted author actions, domain mechanics, "
            "source/citation relation, practical constraints, or observed process details already present in the draft/context."
        ),
    }


def _safe_band_density_qualifying_sentence(sentence: str) -> bool:
    stripped = str(sentence or "").strip()
    if len(stripped.split()) < 8:
        return False
    lower = stripped.lower()
    if lower.startswith(("http", "www.")):
        return False
    if re.match(r"^\s*(references|bibliography|works cited)\b", lower):
        return False
    if stripped.startswith(("-", "*", "•")):
        return False
    if stripped.count('"') >= 2 or stripped.count("'") >= 2:
        return False
    return True


def _safe_band_density_contextual_anchor_target() -> float:
    return _float_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_CONTEXTUAL_ANCHOR_TARGET",
        0.45,
        minimum=0.0,
        maximum=1.0,
    )


def _score_safe_band_evidence_pack_variant(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_scores: dict[str, Any],
    current_text: str,
    current_scores: dict[str, Any],
    sections: list[SectionUnit],
    variant: dict[str, Any],
    output_dir: Path,
    label: str,
    author_proxy_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_text, apply_status, materiality = _apply_safe_band_evidence_pack_variant(
        current_text=current_text,
        sections=sections,
        variant=variant,
    )
    source_pack_text = "\n\n".join(section.text for section in sections)
    replacement_pack_text = "\n\n".join(
        str(item.get("text") or "")
        for item in variant.get("replacements", [])
        if isinstance(item, dict)
    )
    author_proxy_audit = _author_proxy_candidate_audit(
        source_pack_text,
        replacement_pack_text,
        author_proxy_context,
        phase="safe_band_evidence_pack",
    )
    author_proxy_quality = _author_proxy_quality_score(
        source_text=source_pack_text,
        candidate_text=replacement_pack_text,
        context=author_proxy_context,
        provenance=_author_proxy_item_list(variant.get("author_proxy_provenance")),
        review_items=_author_proxy_item_list(variant.get("author_review_items")),
        audit=author_proxy_audit,
    )
    if not apply_status.get("applied"):
        return {
            "section_id": "safe_band_evidence_pack",
            "variant_id": variant.get("variant_id"),
            "label": label,
            "text": replacement_pack_text,
            "word_count": word_count(replacement_pack_text),
            "apply_status": apply_status,
            "scores": current_scores,
            "incremental": {},
            "local_scores": {},
            "local_goal": {},
            "candidate_text": candidate_text,
            "safe_band_evidence_materiality": materiality,
            "safe_band_evidence_pack_materiality": materiality,
            "author_proxy_audit": author_proxy_audit,
            "author_proxy_quality": author_proxy_quality,
            "author_proxy_provenance": _author_proxy_item_list(variant.get("author_proxy_provenance")),
            "author_review_items": _author_proxy_item_list(variant.get("author_review_items")),
        }
    source_integrity = minimal_replacement_text_integrity(current_text)
    candidate_integrity = minimal_replacement_text_integrity(candidate_text)
    integrity_regression = _text_integrity_regression(source_integrity, candidate_integrity)
    if not integrity_regression.get("passed"):
        apply_status = {
            **apply_status,
            "applied": False,
            "reason": "candidate_text_integrity_regressed_after_pack_apply",
            "source_integrity": source_integrity,
            "candidate_integrity": candidate_integrity,
            "integrity_regression": integrity_regression,
        }
        return {
            "section_id": "safe_band_evidence_pack",
            "variant_id": variant.get("variant_id"),
            "label": label,
            "text": replacement_pack_text,
            "word_count": word_count(replacement_pack_text),
            "apply_status": apply_status,
            "scores": current_scores,
            "incremental": {},
            "local_scores": {},
            "local_goal": {},
            "candidate_text": candidate_text,
            "safe_band_evidence_materiality": materiality,
            "safe_band_evidence_pack_materiality": materiality,
            "author_proxy_audit": author_proxy_audit,
            "author_proxy_quality": author_proxy_quality,
            "author_proxy_provenance": _author_proxy_item_list(variant.get("author_proxy_provenance")),
            "author_review_items": _author_proxy_item_list(variant.get("author_review_items")),
        }
    candidate_report = _scan_report(candidate_text)
    candidate_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=baseline_report,
        candidate_report=candidate_report,
    ).to_dict()
    candidate_goal = _with_v5_density_gate(candidate_text, candidate_report, candidate_goal)
    scores = _score_summary(original_text, candidate_report, candidate_goal)
    _add_deltas(scores, baseline_scores)
    safe_name = label.replace("/", "_")
    (output_dir / f"{safe_name}.txt").write_text(candidate_text)
    (output_dir / f"{safe_name}_pack.txt").write_text(replacement_pack_text)
    (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
    return {
        "section_id": "safe_band_evidence_pack",
        "variant_id": variant.get("variant_id"),
        "label": label,
        "text": replacement_pack_text,
        "word_count": word_count(replacement_pack_text),
        "apply_status": apply_status,
        "scores": scores,
        "incremental": _incremental_deltas(scores, current_scores),
        "local_scores": {},
        "local_goal": {},
        "candidate_text": candidate_text,
        "candidate_report": candidate_report,
        "candidate_goal": candidate_goal,
        "safe_band_evidence_materiality": materiality,
        "safe_band_evidence_pack_materiality": materiality,
        "author_proxy_audit": author_proxy_audit,
        "author_proxy_quality": author_proxy_quality,
        "author_proxy_provenance": _author_proxy_item_list(variant.get("author_proxy_provenance")),
        "author_review_items": _author_proxy_item_list(variant.get("author_review_items")),
    }


def _safe_band_controlled_operation_targets(
    current_text: str,
    current_report: dict[str, Any],
    current_goal: dict[str, Any],
    *,
    current_scores: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    limit = _safe_band_controlled_operation_target_limit()
    targets: list[dict[str, Any]] = []
    seed_targets = (
        _safe_band_density_controlled_operation_targets(
            current_text,
            current_report,
            current_goal,
            current_scores=current_scores or {},
        )
        if current_scores is not None and _safe_band_density_section_repair_should_run(current_scores=current_scores, current_goal=current_goal)
        else []
    )
    seed_targets.extend(_final_topk_sentence_route_targets(current_text, current_report, current_goal))
    seen: set[str] = set()
    for target in seed_targets:
        sentence = str(target.get("sentence") or "").strip()
        key = sentence.casefold()
        if not sentence or sentence not in current_text or key in seen:
            continue
        seen.add(key)
        targets.append({
            "target_id": str(target.get("target_id") or f"t{len(targets) + 1:03d}"),
            "sentence": sentence,
            "operation": "delete_exact_target_sentence",
            "source": target.get("source"),
            "predictability_risk": target.get("predictability_risk"),
            "top10_ratio": target.get("top10_ratio"),
            "top50_ratio": target.get("top50_ratio"),
            "context": target.get("context"),
        })
        if len(targets) >= limit:
            break
    return targets


def _safe_band_density_controlled_operation_targets(
    current_text: str,
    current_report: dict[str, Any],
    current_goal: dict[str, Any],
    *,
    current_scores: dict[str, Any],
) -> list[dict[str, Any]]:
    contract = _safe_band_kpi_contract(current_scores, current_goal)
    gaps = contract.get("gaps") if isinstance(contract.get("gaps"), dict) else {}
    if _number(gaps.get("qualifying_text_ai_density")) <= 0 or _number(gaps.get("topk_calibrated_risk")) > 0:
        return []
    mitigation = current_report.get("ai_mitigation") if isinstance(current_report.get("ai_mitigation"), dict) else {}
    segments = mitigation.get("target_segments") if isinstance(mitigation.get("target_segments"), list) else []
    rows: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            continue
        sentence = str(segment.get("text") or "").strip()
        if not sentence or sentence not in current_text:
            continue
        if not _safe_band_density_controlled_segment_is_auto_safe(segment):
            continue
        signal = segment.get("primary_signal") if isinstance(segment.get("primary_signal"), dict) else {}
        word_total = word_count(sentence)
        rows.append((
            (
                _number(signal.get("score")),
                1.0 if str(segment.get("lever") or "") == "predictability_reduction" else 0.0,
                min(float(word_total), 35.0),
                -abs(22.0 - float(word_total)),
            ),
            {
                "target_id": f"d{len(rows) + 1:03d}",
                "sentence_id": segment.get("sentence_id") or segment.get("segment_id"),
                "sentence": sentence,
                "source": "ai_mitigation_density_target_segments",
                "predictability_risk": signal.get("score"),
                "context": _target_sentence_context(current_text, sentence),
                "segment": segment,
            },
        ))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [row for _score, row in rows[:_safe_band_controlled_operation_target_limit()]]


def _safe_band_density_controlled_segment_is_auto_safe(segment: dict[str, Any]) -> bool:
    user_input = str(segment.get("user_input_needed") or "").strip().casefold()
    if user_input and not user_input.startswith("none"):
        return False
    bucket = str(segment.get("bucket") or "")
    lever = str(segment.get("lever") or "")
    signal = segment.get("primary_signal") if isinstance(segment.get("primary_signal"), dict) else {}
    permission = str(signal.get("rewrite_permission") or "")
    if permission and permission not in {"auto_apply", "auto_candidate", "suggestion_only"}:
        return False
    return bucket == "auto_candidate" or lever == "predictability_reduction"


def _safe_band_controlled_operation_variants(
    *,
    current_text: str,
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        sentence = str(target.get("sentence") or "").strip()
        if not sentence or sentence not in current_text:
            continue
        variants.append({
            "variant_id": f"delete_{target.get('target_id') or index}",
            "operation": "delete_exact_target_sentence",
            "target_id": target.get("target_id"),
            "sentence": sentence,
            "operation_reason": "controller_owned_safe_band_atomic_edit",
        })
        suffix = _strong_punctuation_suffix_candidate(sentence)
        if suffix:
            variants.append({
                "variant_id": f"suffix_{target.get('target_id') or index}",
                "operation": "keep_suffix_after_strong_punctuation",
                "target_id": target.get("target_id"),
                "sentence": sentence,
                "replacement": suffix,
                "operation_reason": "controller_owned_safe_band_atomic_edit",
            })
    return variants


def _score_safe_band_controlled_operation_variant(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_scores: dict[str, Any],
    current_text: str,
    current_scores: dict[str, Any],
    variant: dict[str, Any],
    output_dir: Path,
    label: str,
    author_proxy_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_text, apply_status, materiality = _apply_safe_band_controlled_operation_variant(
        current_text=current_text,
        variant=variant,
    )
    author_proxy_audit = _author_proxy_candidate_audit(
        current_text,
        candidate_text,
        author_proxy_context,
        phase="safe_band_controlled_operation",
    )
    author_proxy_quality = _author_proxy_quality_score(
        source_text=current_text,
        candidate_text=candidate_text,
        context=author_proxy_context,
        audit=author_proxy_audit,
    )
    if not apply_status.get("applied"):
        return {
            "section_id": "safe_band_controlled_operation",
            "variant_id": variant.get("variant_id"),
            "label": label,
            "text": candidate_text,
            "word_count": word_count(candidate_text),
            "apply_status": apply_status,
            "scores": current_scores,
            "incremental": {},
            "candidate_text": candidate_text,
            "safe_band_evidence_materiality": materiality,
            "safe_band_controlled_operation_materiality": materiality,
            "author_proxy_audit": author_proxy_audit,
            "author_proxy_quality": author_proxy_quality,
        }
    candidate_report = _scan_report(candidate_text)
    candidate_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=baseline_report,
        candidate_report=candidate_report,
    ).to_dict()
    candidate_goal = _with_v5_density_gate(candidate_text, candidate_report, candidate_goal)
    scores = _score_summary(original_text, candidate_report, candidate_goal)
    _add_deltas(scores, baseline_scores)
    safe_name = label.replace("/", "_")
    (output_dir / f"{safe_name}.txt").write_text(candidate_text)
    (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
    return {
        "section_id": "safe_band_controlled_operation",
        "variant_id": variant.get("variant_id"),
        "label": label,
        "text": str(variant.get("sentence") or ""),
        "word_count": word_count(candidate_text),
        "apply_status": apply_status,
        "scores": scores,
        "incremental": _incremental_deltas(scores, current_scores),
        "candidate_text": candidate_text,
        "candidate_report": candidate_report,
        "candidate_goal": candidate_goal,
        "safe_band_evidence_materiality": materiality,
        "safe_band_controlled_operation_materiality": materiality,
        "author_proxy_audit": author_proxy_audit,
        "author_proxy_quality": author_proxy_quality,
    }


def _apply_safe_band_controlled_operation_variant(
    *,
    current_text: str,
    variant: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    operation = str(variant.get("operation") or "").strip()
    sentence = str(variant.get("sentence") or "").strip()
    source = str(current_text or "")
    supported_operations = {"delete_exact_target_sentence", "keep_suffix_after_strong_punctuation"}
    if operation not in supported_operations or not sentence:
        materiality = {"passed": False, "reason": "unsupported_or_empty_controlled_operation"}
        return source, {"applied": False, "reason": "unsupported_or_empty_controlled_operation"}, materiality
    if source.count(sentence) != 1:
        materiality = {
            "passed": False,
            "reason": "target_sentence_not_unique",
            "match_count": source.count(sentence),
        }
        return source, {"applied": False, "reason": "target_sentence_not_unique", "match_count": source.count(sentence)}, materiality
    replacement = ""
    if operation == "keep_suffix_after_strong_punctuation":
        replacement = str(variant.get("replacement") or "").strip()
        expected = _strong_punctuation_suffix_candidate(sentence)
        min_suffix_words = _safe_band_controlled_operation_min_suffix_words()
        if not replacement or replacement != expected or word_count(replacement) < min_suffix_words:
            materiality = {
                "passed": False,
                "reason": "invalid_controlled_suffix_replacement",
                "replacement_words": word_count(replacement),
                "minimum_replacement_words": min_suffix_words,
            }
            return source, {"applied": False, "reason": "invalid_controlled_suffix_replacement"}, materiality
    candidate = _replace_controlled_sentence(source, sentence, replacement)
    source_words = word_count(source)
    candidate_words = word_count(candidate)
    word_ratio = candidate_words / max(1, source_words)
    paragraph_count_preserved = _paragraph_count(candidate) == _paragraph_count(source)
    min_ratio = _safe_band_controlled_operation_min_word_ratio()
    applied = bool(candidate.strip()) and candidate != source and word_ratio >= min_ratio and paragraph_count_preserved
    if not applied:
        reason = "controlled_operation_materiality_failed"
    elif operation == "keep_suffix_after_strong_punctuation":
        reason = "controlled_strong_punctuation_suffix_kept"
    else:
        reason = "controlled_target_sentence_deleted"
    materiality = {
        "passed": applied,
        "reason": reason,
        "operation": operation,
        "target_id": variant.get("target_id"),
        "replacement": replacement,
        "source_words": source_words,
        "candidate_words": candidate_words,
        "word_ratio": round(word_ratio, 4),
        "minimum_word_ratio": min_ratio,
        "paragraph_count_preserved": paragraph_count_preserved,
    }
    return candidate, {
        "applied": applied,
        "scope": "safe_band_controlled_operation",
        "operation": operation,
        "target_id": variant.get("target_id"),
        "replacement": replacement,
        "source_words": source_words,
        "candidate_words": candidate_words,
        "word_ratio": round(word_ratio, 4),
        **({} if applied else {"reason": reason}),
    }, materiality


def _strong_punctuation_suffix_candidate(sentence: str) -> str | None:
    source = str(sentence or "").strip()
    if not source:
        return None
    delimiters = (" — ",)
    for delimiter in delimiters:
        if source.count(delimiter) != 1:
            continue
        _before, after = source.split(delimiter, 1)
        suffix = after.strip()
        if word_count(suffix) < _safe_band_controlled_operation_min_suffix_words():
            continue
        if suffix and suffix[-1] not in ".!?":
            suffix += "."
        return suffix[:1].upper() + suffix[1:] if suffix else None
    return None


def _replace_controlled_sentence(source: str, sentence: str, replacement: str) -> str:
    offset = source.find(sentence)
    if offset < 0:
        return source
    before = source[:offset]
    after = source[offset + len(sentence):]
    replacement = str(replacement or "").strip()
    if before.endswith(" ") and after.startswith(" "):
        after = after[1:]
    if before.endswith("\n") and after.startswith(" "):
        after = after.lstrip(" ")
    middle = replacement
    if middle and before and not before.endswith(("\n", " ")):
        middle = " " + middle
    if middle and after and not after.startswith(("\n", " ")):
        middle = middle + " "
    candidate = before + middle + after
    return re.sub(r"[ \t]{2,}", " ", candidate).strip()


def _score_final_topk_sentence_route_variant(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_scores: dict[str, Any],
    current_text: str,
    current_scores: dict[str, Any],
    targets: list[dict[str, Any]],
    variant: dict[str, Any],
    output_dir: Path,
    label: str,
    author_proxy_context: dict[str, Any] | None = None,
    author_proxy_phase: str | None = None,
    require_all_targets: bool = True,
) -> dict[str, Any]:
    lookup = {str(target.get("target_id") or ""): str(target.get("sentence") or "") for target in targets}
    candidate_text = current_text
    applied: list[dict[str, Any]] = []
    used: set[str] = set()
    for repair in variant.get("repairs") if isinstance(variant.get("repairs"), list) else []:
        if not isinstance(repair, dict):
            continue
        target_id = str(repair.get("target_id") or "")
        before = lookup.get(target_id) or ""
        after = str(repair.get("after") or "").strip()
        if not target_id or target_id in used or not before or not after or before == after:
            continue
        if before not in candidate_text:
            continue
        candidate_text = candidate_text.replace(before, after, 1)
        used.add(target_id)
        applied.append({
            "target_id": target_id,
            "before": before,
            "after": after,
            "sentence_job": repair.get("sentence_job"),
            "current_route": repair.get("current_route"),
            "repair_route": repair.get("repair_route"),
        })
    candidate_words = word_count(candidate_text)
    applied_target_count = len(applied)
    required_target_count = len(targets) if require_all_targets else 1
    apply_status: dict[str, Any] = {
        "applied": applied_target_count >= required_target_count,
        "scope": "final_topk_sentence_route",
        "target_count": len(targets),
        "required_target_count": required_target_count,
        "applied_repair_count": applied_target_count,
        "partial_candidate": not require_all_targets,
        "source_words": word_count(current_text),
        "candidate_words": candidate_words,
    }
    if not apply_status["applied"]:
        apply_status["reason"] = (
            "no_target_sentence_repaired"
            if not require_all_targets
            else "not_all_target_sentences_repaired"
        )
    elif _paragraph_count(candidate_text) != _paragraph_count(current_text):
        apply_status.update({
            "applied": False,
            "reason": "paragraph_count_changed",
            "source_paragraph_count": _paragraph_count(current_text),
            "candidate_paragraph_count": _paragraph_count(candidate_text),
        })
    else:
        source_integrity = minimal_replacement_text_integrity(current_text)
        candidate_integrity = minimal_replacement_text_integrity(candidate_text)
        integrity_regression = _text_integrity_regression(source_integrity, candidate_integrity)
        if not integrity_regression.get("passed"):
            apply_status.update({
                "applied": False,
                "reason": "candidate_text_integrity_regressed",
                "source_integrity": source_integrity,
                "candidate_integrity": candidate_integrity,
                "integrity_regression": integrity_regression,
            })
    if not apply_status.get("applied"):
        author_proxy_audit = _author_proxy_candidate_audit(
            current_text,
            candidate_text,
            author_proxy_context,
            phase=author_proxy_phase or label,
        )
        author_proxy_quality = _author_proxy_quality_score(
            source_text=current_text,
            candidate_text=candidate_text,
            context=author_proxy_context,
            audit=author_proxy_audit,
        )
        return {
            "section_id": "final_topk_sentence_route",
            "variant_id": variant.get("variant_id"),
            "label": label,
            "text": candidate_text,
            "word_count": candidate_words,
            "apply_status": apply_status,
            "scores": current_scores,
            "incremental": {},
            "candidate_text": candidate_text,
            "applied_repairs": applied,
            "author_proxy_audit": author_proxy_audit,
            "author_proxy_quality": author_proxy_quality,
        }
    author_proxy_audit = _author_proxy_candidate_audit(
        current_text,
        candidate_text,
        author_proxy_context,
        phase=author_proxy_phase or label,
    )
    author_proxy_quality = _author_proxy_quality_score(
        source_text=current_text,
        candidate_text=candidate_text,
        context=author_proxy_context,
        audit=author_proxy_audit,
    )
    candidate_report = _scan_report(candidate_text)
    candidate_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=baseline_report,
        candidate_report=candidate_report,
    ).to_dict()
    candidate_goal = _with_v5_density_gate(candidate_text, candidate_report, candidate_goal)
    scores = _score_summary(original_text, candidate_report, candidate_goal)
    _add_deltas(scores, baseline_scores)
    safe_name = label.replace("/", "_")
    (output_dir / f"{safe_name}.txt").write_text(candidate_text)
    (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
    return {
        "section_id": "final_topk_sentence_route",
        "variant_id": variant.get("variant_id"),
        "label": label,
        "text": candidate_text,
        "word_count": candidate_words,
        "apply_status": apply_status,
        "scores": scores,
        "incremental": _incremental_deltas(scores, current_scores),
        "candidate_text": candidate_text,
        "candidate_report": candidate_report,
        "candidate_goal": candidate_goal,
        "applied_repairs": applied,
        "author_proxy_audit": author_proxy_audit,
        "author_proxy_quality": author_proxy_quality,
    }


def _final_topk_sentence_route_targets(
    current_text: str,
    current_report: dict[str, Any],
    current_goal: dict[str, Any],
) -> list[dict[str, Any]]:
    limit = _final_topk_sentence_route_target_limit()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    density = (
        current_goal.get("eligible_span_density_gate")
        if isinstance(current_goal.get("eligible_span_density_gate"), dict)
        else _density_gate_for_report(current_text, current_report)
    )
    for row in density.get("top_sentence_targets") if isinstance(density.get("top_sentence_targets"), list) else []:
        if not isinstance(row, dict):
            continue
        sentence = str(row.get("preview") or "").strip()
        if not sentence or sentence not in current_text or sentence.casefold() in seen:
            continue
        seen.add(sentence.casefold())
        rows.append({
            "target_id": f"t{len(rows) + 1:03d}",
            "sentence_id": row.get("sentence_id"),
            "sentence": sentence,
            "context": _target_sentence_context(current_text, sentence),
            "source": "eligible_span_density_gate",
            "top10_ratio": row.get("top10_ratio"),
            "top50_ratio": row.get("top50_ratio"),
            "predictability_risk": row.get("predictability_risk"),
        })
        if len(rows) >= limit:
            return rows

    predictability = current_report.get("predictability") if isinstance(current_report.get("predictability"), dict) else {}
    sentence_rows = predictability.get("all_sentences") or predictability.get("sentences") or []
    ranked: list[dict[str, Any]] = []
    for index, item in enumerate(sentence_rows if isinstance(sentence_rows, list) else []):
        if not isinstance(item, dict):
            continue
        sentence = str(item.get("sentence") or item.get("text") or "").strip()
        if not sentence or sentence not in current_text or sentence.casefold() in seen:
            continue
        top10 = _number(item.get("top10_ratio") or item.get("top_10_ratio") or item.get("top10"))
        top50 = _number(item.get("top50_ratio") or item.get("top_50_ratio") or item.get("top50"))
        risk = _number(item.get("predictability_risk") or item.get("risk") or item.get("score"))
        ranked.append({
            "sentence_id": item.get("sentence_id") or f"s{index + 1:03d}",
            "sentence": sentence,
            "source": "predictability_sentence_rows",
            "top10_ratio": round(top10, 4),
            "top50_ratio": round(top50, 4),
            "predictability_risk": round(risk, 4),
            "route_score": round(top10 * 0.55 + top50 * 0.20 + risk * 0.20, 4),
        })
    ranked.sort(key=lambda row: _number(row.get("route_score")), reverse=True)
    for row in ranked:
        sentence = str(row.get("sentence") or "")
        if sentence.casefold() in seen:
            continue
        seen.add(sentence.casefold())
        rows.append({
            "target_id": f"t{len(rows) + 1:03d}",
            "context": _target_sentence_context(current_text, sentence),
            **row,
        })
        if len(rows) >= limit:
            break
    return rows


def _sanitize_final_topk_sentence_route_variants(value: Any, *, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_ids = {str(target.get("target_id") or "") for target in targets}
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value if isinstance(value, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        variant_id = str(item.get("variant_id") or f"v{index}")
        repairs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for repair in item.get("repairs") if isinstance(item.get("repairs"), list) else []:
            if not isinstance(repair, dict):
                continue
            target_id = str(repair.get("target_id") or "")
            if target_id not in target_ids or target_id in seen:
                continue
            after = " ".join(str(repair.get("after") or "").split())
            if not after:
                continue
            repairs.append({
                "target_id": target_id,
                "sentence_job": str(repair.get("sentence_job") or "")[:80],
                "current_route": str(repair.get("current_route") or "")[:180],
                "repair_route": str(repair.get("repair_route") or "")[:180],
                "after": after,
            })
            seen.add(target_id)
        if repairs:
            rows.append({
                "variant_id": variant_id,
                "repairs": repairs,
            })
    return rows


def _best_final_topk_sentence_route_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if (row.get("apply_status") or {}).get("applied")]
    if not eligible:
        return None
    accepted = [row for row in eligible if _has_final_topk_sentence_route_movement(row)]
    if accepted:
        return max(accepted, key=_final_topk_sentence_route_sort_key)
    return max(eligible, key=_final_topk_sentence_route_sort_key)


def _has_final_topk_sentence_route_movement(row: dict[str, Any]) -> bool:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    min_topk = _final_topk_sentence_route_min_topk_delta()
    min_calibrated = _final_topk_sentence_route_min_calibrated_delta()
    min_ai = _final_topk_sentence_route_min_ai_delta()
    if (
        _number(incremental.get("topk_delta")) < min_topk
        and _number(incremental.get("topk_calibrated_risk_delta")) < min_calibrated
    ):
        return False
    if _number(incremental.get("ai_delta")) < min_ai:
        return False
    if _number(incremental.get("risky_window_count_delta")) < 0:
        return False
    if _number(incremental.get("unsafe_cluster_count_delta")) < 0:
        return False
    return True


def _final_topk_sentence_route_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    return (
        1.0 if _candidate_strict_safe_band_achieved(row) else 0.0,
        -_candidate_safe_band_gap(row),
        _number(incremental.get("topk_delta")) * 3.0 + _number(incremental.get("topk_calibrated_risk_delta")) * 1.5,
        _number(incremental.get("topk_delta")),
        _number(incremental.get("topk_calibrated_risk_delta")),
        _number(incremental.get("ai_delta")),
        _number(incremental.get("external_delta")),
        _number(incremental.get("unsafe_cluster_count_delta")),
        _author_proxy_quality_sort_value(row),
    )


def _candidate_strict_safe_band_achieved(row: dict[str, Any]) -> bool:
    goal = row.get("candidate_goal") if isinstance(row.get("candidate_goal"), dict) else {}
    if goal.get("strict_ai_safe_band_achieved") is True:
        return True
    gate = goal.get("ai_footprint_gate") if isinstance(goal.get("ai_footprint_gate"), dict) else {}
    return bool(gate.get("safe_band"))


def _candidate_safe_band_gap(row: dict[str, Any]) -> float:
    goal = row.get("candidate_goal") if isinstance(row.get("candidate_goal"), dict) else {}
    gate = goal.get("ai_footprint_gate") if isinstance(goal.get("ai_footprint_gate"), dict) else {}
    remaining = gate.get("remaining_ai_footprint_drivers") if isinstance(gate.get("remaining_ai_footprint_drivers"), list) else []
    gap = 0.0
    for item in remaining:
        if not isinstance(item, dict):
            continue
        gap += max(0.0, _number(item.get("value")) - _number(item.get("safe_band")))
    if gap > 0:
        return round(gap, 3)
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return round(
        max(0.0, _number(scores.get("topk_calibrated_risk")) - 25.0)
        + max(0.0, _number(scores.get("qualifying_text_ai_density")) - 35.0),
        3,
    )


def _candidate_topk_calibrated_safe(row: dict[str, Any]) -> bool:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    goal = row.get("candidate_goal") if isinstance(row.get("candidate_goal"), dict) else {}
    gate = goal.get("ai_footprint_gate") if isinstance(goal.get("ai_footprint_gate"), dict) else {}
    thresholds = gate.get("safe_band_thresholds") if isinstance(gate.get("safe_band_thresholds"), dict) else {}
    threshold = _number(thresholds.get("topk_calibrated_risk") if thresholds.get("topk_calibrated_risk") is not None else 25.0)
    return _number(scores.get("topk_calibrated_risk")) <= threshold


def _safe_band_evidence_repair_section(
    current_text: str,
    current_report: dict[str, Any],
    current_goal: dict[str, Any],
) -> SectionUnit | None:
    sections = _safe_band_evidence_repair_sections(current_text, current_report, current_goal, limit=1)
    return sections[0] if sections else None


def _safe_band_evidence_repair_sections(
    current_text: str,
    current_report: dict[str, Any],
    current_goal: dict[str, Any],
    *,
    limit: int,
) -> list[SectionUnit]:
    max_sections = max(1, int(limit or 1))
    sections: list[SectionUnit] = []
    seen: set[tuple[int, int]] = set()

    def add(section: SectionUnit | None) -> None:
        if section is None or len(sections) >= max_sections:
            return
        signature = (section.start_char, section.end_char)
        if signature in seen:
            return
        seen.add(signature)
        sections.append(section)

    target_sections: list[SectionUnit] = []
    targets = _final_topk_sentence_route_targets(current_text, current_report, current_goal)
    for target in targets:
        sentence = str(target.get("sentence") or "").strip()
        section = _section_from_paragraph_containing_text(
            current_text,
            sentence,
            section_id=f"safe_band_evidence_repair_{str(target.get('target_id') or 'target')}",
            selection_reason="top_safe_band_sentence_target",
            scanner_focus=target,
        )
        if section is not None:
            target_sections.append(section)
    if _safe_band_evidence_repair_composite_window_enabled():
        add(_safe_band_evidence_composite_section(
            current_text,
            _first_unique_sections(target_sections, limit=max_sections),
        ))
        if len(sections) >= max_sections:
            return sections
    for section in target_sections:
        add(section)
        if len(sections) >= max_sections:
            return sections

    density = (
        current_goal.get("eligible_span_density_gate")
        if isinstance(current_goal.get("eligible_span_density_gate"), dict)
        else _density_gate_for_report(current_text, current_report)
    )
    clusters = density.get("top_unsafe_clusters") if isinstance(density.get("top_unsafe_clusters"), list) else []
    for ordinal, cluster in enumerate(clusters, start=1):
        if not isinstance(cluster, dict):
            continue
        preview = str(cluster.get("preview") or "").strip()
        for candidate in [preview, *_sentences(preview)]:
            add(_section_from_paragraph_containing_text(
                current_text,
                candidate,
                section_id=f"safe_band_evidence_repair_cluster_{ordinal:03d}",
                selection_reason="top_unsafe_cluster_paragraph",
                scanner_focus=cluster,
            ))
            if len(sections) >= max_sections:
                return sections
        fallback = _section_from_density_cluster(current_text, current_report, cluster, ordinal=ordinal)
        if fallback is not None:
            add(SectionUnit(
                section_id=f"safe_band_evidence_repair_cluster_{ordinal:03d}",
                heading="Safe-band evidence repair",
                text=fallback.text,
                start_char=fallback.start_char,
                end_char=fallback.end_char,
                paragraph_count=fallback.paragraph_count,
                word_count=fallback.word_count,
                metadata={
                    **(fallback.metadata or {}),
                    "selection_reason": "top_unsafe_cluster_window",
                    "scanner_focus": cluster,
                },
            ))
            if len(sections) >= max_sections:
                return sections
    return sections


def _safe_band_evidence_pack_sections(
    current_text: str,
    current_report: dict[str, Any],
    current_goal: dict[str, Any],
    *,
    limit: int,
) -> list[SectionUnit]:
    target_sections: list[SectionUnit] = []
    seen: set[tuple[int, int]] = set()
    max_sections = max(1, int(limit or 1))
    max_section_words = _safe_band_evidence_pack_max_section_words()
    max_source_words = _safe_band_evidence_pack_max_source_words()
    source_words = 0

    def add(section: SectionUnit | None) -> None:
        nonlocal source_words
        if section is None or len(target_sections) >= max_sections:
            return
        if section.word_count > max_section_words:
            return
        if target_sections and source_words + section.word_count > max_source_words:
            return
        signature = (section.start_char, section.end_char)
        if signature in seen:
            return
        seen.add(signature)
        target_sections.append(section)
        source_words += section.word_count

    gate = current_goal.get("ai_footprint_gate") if isinstance(current_goal.get("ai_footprint_gate"), dict) else {}
    remaining = gate.get("remaining_ai_footprint_drivers") if isinstance(gate.get("remaining_ai_footprint_drivers"), list) else []
    density_remaining = any(
        isinstance(item, dict)
        and str(item.get("driver") or "") == "qualifying_text_ai_density"
        and _number(item.get("value")) > _number(item.get("safe_band"))
        for item in remaining
    )
    if density_remaining:
        for section in _safe_band_density_section_repair_sections(
            current_text,
            current_report,
            current_goal,
            limit=max_sections,
        ):
            add(section)
            if len(target_sections) >= max_sections:
                return target_sections

    for target in _final_topk_sentence_route_targets(current_text, current_report, current_goal):
        section = _section_from_paragraph_containing_text(
            current_text,
            str(target.get("sentence") or ""),
            section_id=f"safe_band_evidence_repair_{str(target.get('target_id') or 'target')}",
            selection_reason="top_safe_band_sentence_target",
            scanner_focus=target,
        )
        if section is None:
            continue
        add(section)
        if len(target_sections) >= max_sections:
            break
    return target_sections


def _safe_band_density_section_repair_sections(
    current_text: str,
    current_report: dict[str, Any],
    current_goal: dict[str, Any],
    *,
    limit: int,
    exclude_ranges: set[tuple[int, int]] | None = None,
) -> list[SectionUnit]:
    max_sections = max(1, int(limit or 1))
    sections: list[SectionUnit] = []
    seen: set[tuple[int, int]] = set()
    excluded = set(exclude_ranges or set())

    def add(section: SectionUnit | None) -> None:
        if section is None or len(sections) >= max_sections:
            return
        signature = _safe_band_section_range_signature(section)
        if signature in seen:
            return
        if _safe_band_section_range_excluded(signature, excluded):
            return
        seen.add(signature)
        sections.append(section)

    mitigation = current_report.get("ai_mitigation") if isinstance(current_report.get("ai_mitigation"), dict) else {}
    segments = mitigation.get("target_segments") if isinstance(mitigation.get("target_segments"), list) else []
    ranked_segments: list[tuple[tuple[float, ...], int, dict[str, Any]]] = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text or text not in current_text:
            continue
        ranked_segments.append((_safe_band_density_segment_sort_key(segment), index, segment))
    ranked_segments.sort(key=lambda row: row[0], reverse=True)
    for _score, index, segment in ranked_segments:
        segment_id = str(segment.get("segment_id") or segment.get("sentence_id") or f"s{index:03d}")
        add(_safe_band_density_expanded_section(
            current_text,
            _section_from_paragraph_containing_text(
                current_text,
                str(segment.get("text") or ""),
                section_id=f"safe_band_density_section_{segment_id}",
                selection_reason="ai_mitigation_density_target_segment",
                scanner_focus={
                    "source": "ai_mitigation.target_segments",
                    "segment": segment,
                    "density_drivers": _safe_band_kpi_contract({}, current_goal).get("secondary_density_drivers"),
                },
            ),
        ))
        if len(sections) >= max_sections:
            return sections

    for section in _safe_band_evidence_repair_sections(
        current_text,
        current_report,
        current_goal,
        limit=max_sections,
    ):
        add(_safe_band_density_expanded_section(
            current_text,
            SectionUnit(
                section_id=f"safe_band_density_section_{section.section_id}",
                heading="Safe-band density section repair",
                text=section.text,
                start_char=section.start_char,
                end_char=section.end_char,
                paragraph_count=section.paragraph_count,
                word_count=section.word_count,
                metadata={
                    **(section.metadata or {}),
                    "selection_reason": "fallback_safe_band_section_for_density",
                },
            ),
        ))
        if len(sections) >= max_sections:
            break
    return sections


def _safe_band_density_section_repair_deterministic_variants(section: SectionUnit) -> list[RecompositionVariant]:
    deduped = _safe_band_adjacent_duplicate_sentence_cleanup(section.text)
    if not deduped or " ".join(deduped.split()) == " ".join(str(section.text or "").split()):
        return []
    return [
        RecompositionVariant(
            "deterministic_adjacent_duplicate_cleanup",
            deduped,
            word_count(deduped),
            author_proxy_provenance=[
                {
                    "item_id": "deterministic-adjacent-duplicate-cleanup",
                    "provenance": "duplicate_source_cleanup",
                    "target_text": section.text,
                    "generated_text": deduped,
                    "user_input_needed": "",
                    "author_task": "",
                }
            ],
            author_review_items=[],
        )
    ]


def _safe_band_adjacent_duplicate_sentence_cleanup(text: str) -> str:
    sentences = [sentence.strip() for sentence in _sentences(str(text or "")) if sentence.strip()]
    if len(sentences) < 2:
        return str(text or "").strip()
    kept: list[str] = []
    previous_signature = ""
    removed = 0
    for sentence in sentences:
        signature = _safe_band_duplicate_sentence_signature(sentence)
        if signature and signature == previous_signature:
            removed += 1
            continue
        kept.append(sentence)
        previous_signature = signature
    if removed <= 0:
        return str(text or "").strip()
    return " ".join(kept).strip()


def _safe_band_duplicate_sentence_signature(sentence: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(sentence or "").lower()).strip()


def _safe_band_density_duplicate_cleanup_materiality(source_text: str, candidate_text: str) -> dict[str, Any]:
    expected = _safe_band_adjacent_duplicate_sentence_cleanup(source_text)
    if not expected or " ".join(expected.split()) != " ".join(str(candidate_text or "").split()):
        return {"passed": False, "reason": "not_adjacent_duplicate_cleanup"}
    source_sentences = [sentence for sentence in _sentences(str(source_text or "")) if sentence.strip()]
    expected_sentences = [sentence for sentence in _sentences(expected) if sentence.strip()]
    removed_count = len(source_sentences) - len(expected_sentences)
    return {
        "passed": removed_count > 0,
        "reason": "density_section_adjacent_duplicate_cleanup" if removed_count > 0 else "no_adjacent_duplicate_removed",
        "removed_duplicate_sentence_count": max(0, removed_count),
    }


def _safe_band_section_range_signature(section: SectionUnit) -> tuple[int, int]:
    return (int(section.start_char or 0), int(section.end_char or 0))


def _safe_band_section_range_excluded(
    signature: tuple[int, int],
    excluded_ranges: set[tuple[int, int]],
    *,
    min_overlap_ratio: float = 0.5,
) -> bool:
    if not excluded_ranges:
        return False
    start, end = signature
    if end <= start:
        return False
    width = end - start
    for excluded_start, excluded_end in excluded_ranges:
        overlap = min(end, excluded_end) - max(start, excluded_start)
        if overlap <= 0:
            continue
        excluded_width = max(1, excluded_end - excluded_start)
        if overlap / max(1, min(width, excluded_width)) >= min_overlap_ratio:
            return True
    return False


def _safe_band_density_expanded_section(current_text: str, section: SectionUnit | None) -> SectionUnit | None:
    if section is None:
        return None
    min_words = _safe_band_density_section_repair_min_section_words()
    if section.word_count >= min_words:
        return section
    source = str(current_text or "")
    if section.start_char < 0 or section.end_char <= section.start_char or section.end_char > len(source):
        return section
    delimiter = "\n\n" if "\n\n" in source else "\n"
    max_words = _safe_band_density_section_repair_max_section_words()
    end = section.end_char
    best_end = end
    while end < len(source):
        while end < len(source) and source[end] in "\r\n":
            end += 1
        if end >= len(source):
            break
        next_boundary = source.find(delimiter, end)
        candidate_end = len(source) if next_boundary < 0 else next_boundary
        candidate_text = source[section.start_char:candidate_end]
        candidate_words = word_count(candidate_text)
        if candidate_words > max_words:
            break
        best_end = candidate_end
        if candidate_words >= min_words:
            break
        if next_boundary < 0:
            break
        end = candidate_end + len(delimiter)
    if best_end <= section.end_char:
        return section
    expanded_text = source[section.start_char:best_end]
    return SectionUnit(
        section_id=section.section_id,
        heading=section.heading,
        text=expanded_text,
        start_char=section.start_char,
        end_char=best_end,
        paragraph_count=max(section.paragraph_count, expanded_text.count(delimiter) + 1),
        word_count=word_count(expanded_text),
        metadata={
            **(section.metadata or {}),
            "density_section_expanded": True,
            "original_section_word_count": section.word_count,
            "minimum_section_words": min_words,
        },
    )


def _safe_band_density_segment_sort_key(segment: dict[str, Any]) -> tuple[float, ...]:
    signal = segment.get("primary_signal") if isinstance(segment.get("primary_signal"), dict) else {}
    score = _number(signal.get("score"))
    user_input = str(segment.get("user_input_needed") or "").strip().casefold()
    needs_author_gap = 0.0 if not user_input or user_input.startswith("none") else 1.0
    lever = str(segment.get("lever") or "")
    bucket = str(segment.get("bucket") or "")
    signal_key = str(signal.get("key") or "")
    structured_density_match = 1.0 if any(
        value in {"reasoning_continuity", "structure_revision", "authorship_risk", "specificity", "domain_grounding"}
        for value in (lever, bucket, signal_key)
    ) else 0.0
    return (
        needs_author_gap,
        structured_density_match,
        score,
        1.0 if str(signal.get("tier") or "").casefold() == "high" else 0.0,
    )


def _apply_safe_band_evidence_pack_variant(
    *,
    current_text: str,
    sections: list[SectionUnit],
    variant: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    replacement_rows = variant.get("replacements") if isinstance(variant.get("replacements"), list) else []
    replacement_by_id: dict[str, str] = {}
    for row in replacement_rows:
        if not isinstance(row, dict):
            continue
        section_id = str(row.get("section_id") or "").strip()
        text = str(row.get("text") or "").strip()
        if section_id and text and section_id not in replacement_by_id:
            replacement_by_id[section_id] = text
    partial_pack = variant.get("partial_pack") is True
    required_ids = [
        section.section_id
        for section in sections
        if not partial_pack or section.section_id in replacement_by_id
    ]
    missing = [section_id for section_id in required_ids if section_id not in replacement_by_id]
    extra = sorted(set(replacement_by_id) - set(required_ids))
    source = str(current_text or "")
    materiality_rows: list[dict[str, Any]] = []
    if missing or extra:
        materiality = {
            "passed": False,
            "reason": "pack_replacement_section_mismatch",
            "missing_section_ids": missing,
            "extra_section_ids": extra,
            "sections": materiality_rows,
        }
        return source, {
            "applied": False,
            "reason": "pack_replacement_section_mismatch",
            "missing_section_ids": missing,
            "extra_section_ids": extra,
        }, materiality

    candidate_text = source
    applied: list[dict[str, Any]] = []
    sections_to_apply = [section for section in sections if section.section_id in required_ids]
    for section in sorted(sections_to_apply, key=lambda item: item.start_char, reverse=True):
        replacement = replacement_by_id.get(section.section_id, "")
        target_sentence = str((section.metadata or {}).get("target_sentence") or "")
        if _section_uses_density_safe_band_contract(section):
            materiality_row = _safe_band_density_section_repair_materiality(
                source_text=section.text,
                candidate_text=replacement,
                target_sentence=target_sentence,
            )
            materiality_row["contract"] = "density_section_repair"
        else:
            materiality_row = _safe_band_evidence_repair_materiality(
                source_text=section.text,
                candidate_text=replacement,
                target_sentence=target_sentence,
            )
            materiality_row["contract"] = "safe_band_evidence_repair"
        if _candidate_contains_author_placeholder(replacement):
            materiality_row = {
                **materiality_row,
                "passed": False,
                "reason": "candidate_contains_author_placeholder",
                "placeholder_audit": {
                    "passed": False,
                    "reason": "author_review_gap_written_into_candidate_text",
                },
            }
        if _candidate_has_quote_integrity_issue(section.text, replacement):
            materiality_row = {
                **materiality_row,
                "passed": False,
                "reason": "candidate_quote_integrity_issue",
                "quote_audit": {
                    "passed": False,
                    "reason": "candidate_introduced_unbalanced_double_quote",
                },
            }
        materiality_row["section_id"] = section.section_id
        materiality_rows.append(materiality_row)
        if section.start_char < 0 or section.end_char <= section.start_char or section.end_char > len(candidate_text):
            materiality = {
                "passed": False,
                "reason": "invalid_pack_section_offsets",
                "sections": list(reversed(materiality_rows)),
            }
            return candidate_text, {"applied": False, "reason": "invalid_pack_section_offsets", "section_id": section.section_id}, materiality
        if candidate_text[section.start_char:section.end_char] != section.text:
            materiality = {
                "passed": False,
                "reason": "pack_section_slice_mismatch",
                "sections": list(reversed(materiality_rows)),
            }
            return candidate_text, {"applied": False, "reason": "pack_section_slice_mismatch", "section_id": section.section_id}, materiality
        candidate_text = candidate_text[:section.start_char] + replacement + candidate_text[section.end_char:]
        applied.append({
            "section_id": section.section_id,
            "start_char": section.start_char,
            "end_char": section.end_char,
            "source_words": section.word_count,
            "replacement_words": word_count(replacement),
        })
    materiality_rows = list(reversed(materiality_rows))
    materiality = {
        "passed": bool(materiality_rows) and all(row.get("passed") for row in materiality_rows),
        "reason": (
            "material_multi_section_pack"
            if materiality_rows and all(row.get("passed") for row in materiality_rows)
            else "one_or_more_pack_sections_near_copy"
        ),
        "section_count": len(sections_to_apply),
        "requested_section_count": len(sections),
        "partial_pack": partial_pack,
        "sections": materiality_rows,
    }
    return candidate_text, {
        "applied": materiality["passed"],
        "scope": "safe_band_evidence_pack",
        "section_count": len(sections),
        "applied_sections": list(reversed(applied)),
        "source_words": word_count(source),
        "candidate_words": word_count(candidate_text),
        **({} if materiality["passed"] else {"reason": materiality["reason"]}),
    }, materiality


def _candidate_contains_author_placeholder(text: str) -> bool:
    candidate = str(text or "")
    if not candidate.strip():
        return False
    return bool(re.search(
        r"\[(?:needs_author|author|verify|insert|todo|tbd)[^\[\]]*\]|\b(?:TODO|TBD)\b|<[^<>]+>",
        candidate,
        re.IGNORECASE,
    ))


def _candidate_has_quote_integrity_issue(source_text: str, candidate_text: str) -> bool:
    source_count = _normalized_double_quote_count(source_text)
    candidate_count = _normalized_double_quote_count(candidate_text)
    if candidate_count % 2 == 0:
        return False
    if source_count % 2 != 0:
        return False
    return True


def _normalized_double_quote_count(text: str) -> int:
    value = str(text or "").replace("“", '"').replace("”", '"')
    return value.count('"')


def _safe_band_evidence_pack_composite_variant(
    *,
    sections: list[SectionUnit],
    variants: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not sections or not variants:
        return None
    replacements_by_variant: list[dict[str, str]] = []
    variant_by_id: dict[str, dict[str, Any]] = {}
    for index, variant in enumerate(variants, start=1):
        if not isinstance(variant, dict):
            continue
        variant_id = str(variant.get("variant_id") or f"v{index}")
        variant_by_id[variant_id] = variant
        rows = variant.get("replacements") if isinstance(variant.get("replacements"), list) else []
        replacement_by_id: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            section_id = str(row.get("section_id") or "").strip()
            text = str(row.get("text") or "").strip()
            if section_id and text and section_id not in replacement_by_id:
                replacement_by_id[section_id] = text
        replacements_by_variant.append({"variant_id": variant_id, **replacement_by_id})

    selected: list[dict[str, str]] = []
    selected_variant_ids: list[str] = []
    for section in sections:
        target_sentence = str((section.metadata or {}).get("target_sentence") or "")
        candidates: list[tuple[tuple[float, ...], str, str]] = []
        for row in replacements_by_variant:
            variant_id = str(row.get("variant_id") or "")
            replacement = str(row.get(section.section_id) or "").strip()
            if not replacement:
                continue
            materiality = _safe_band_evidence_repair_materiality(
                source_text=section.text,
                candidate_text=replacement,
                target_sentence=target_sentence,
            )
            if not materiality.get("passed"):
                continue
            length_ratio = word_count(replacement) / max(1, section.word_count)
            candidates.append((
                (
                    _number(materiality.get("changed_sentence_count")),
                    1.0 if materiality.get("target_sentence_changed") else 0.0,
                    -abs(1.0 - length_ratio),
                ),
                variant_id,
                replacement,
            ))
        if not candidates:
            return None
        _key, variant_id, replacement = max(candidates, key=lambda item: item[0])
        selected.append({"section_id": section.section_id, "text": replacement})
        selected_variant_ids.append(variant_id)

    provenance: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    for variant_id in selected_variant_ids:
        variant = variant_by_id.get(variant_id) or {}
        provenance.extend(_author_proxy_item_list(variant.get("author_proxy_provenance"), limit=4))
        review_items.extend(_author_proxy_item_list(variant.get("author_review_items"), limit=4))
    return {
        "variant_id": "composite_material_pack",
        "replacements": selected,
        "source_variant_ids": selected_variant_ids,
        "author_proxy_provenance": _dedupe_author_proxy_items(provenance, limit=4),
        "author_review_items": _dedupe_author_proxy_items(review_items, limit=4),
    }


def _safe_band_evidence_pack_partial_material_variant(
    *,
    sections: list[SectionUnit],
    variants: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not sections or not variants:
        return None
    variant_by_id: dict[str, dict[str, Any]] = {}
    replacements_by_variant: list[dict[str, str]] = []
    for index, variant in enumerate(variants, start=1):
        if not isinstance(variant, dict):
            continue
        variant_id = str(variant.get("variant_id") or f"v{index}")
        variant_by_id[variant_id] = variant
        replacements: dict[str, str] = {"variant_id": variant_id}
        for row in variant.get("replacements") if isinstance(variant.get("replacements"), list) else []:
            if not isinstance(row, dict):
                continue
            section_id = str(row.get("section_id") or "").strip()
            text = str(row.get("text") or "").strip()
            if section_id and text and section_id not in replacements:
                replacements[section_id] = text
        replacements_by_variant.append(replacements)

    selected: list[dict[str, str]] = []
    selected_variant_ids: list[str] = []
    density_selected = 0
    for section in sections:
        target_sentence = str((section.metadata or {}).get("target_sentence") or "")
        candidates: list[tuple[tuple[float, ...], str, str]] = []
        density_contract = _section_uses_density_safe_band_contract(section)
        for row in replacements_by_variant:
            variant_id = str(row.get("variant_id") or "")
            replacement = str(row.get(section.section_id) or "").strip()
            if not replacement:
                continue
            materiality = (
                _safe_band_density_section_repair_materiality(
                    source_text=section.text,
                    candidate_text=replacement,
                    target_sentence=target_sentence,
                )
                if density_contract
                else _safe_band_evidence_repair_materiality(
                    source_text=section.text,
                    candidate_text=replacement,
                    target_sentence=target_sentence,
                )
            )
            if not materiality.get("passed"):
                continue
            length_ratio = word_count(replacement) / max(1, section.word_count)
            candidates.append((
                (
                    1.0 if density_contract else 0.0,
                    _number(materiality.get("changed_sentence_count")),
                    1.0 if materiality.get("target_sentence_changed") else 0.0,
                    -abs(1.0 - length_ratio),
                ),
                variant_id,
                replacement,
            ))
        if not candidates:
            continue
        _key, variant_id, replacement = max(candidates, key=lambda item: item[0])
        selected.append({"section_id": section.section_id, "text": replacement})
        selected_variant_ids.append(variant_id)
        if density_contract:
            density_selected += 1

    if len(selected) < _safe_band_evidence_pack_partial_min_sections() or density_selected <= 0:
        return None

    provenance: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    for variant_id in selected_variant_ids:
        variant = variant_by_id.get(variant_id) or {}
        provenance.extend(_author_proxy_item_list(variant.get("author_proxy_provenance"), limit=4))
        review_items.extend(_author_proxy_item_list(variant.get("author_review_items"), limit=4))
    return {
        "variant_id": "partial_material_pack",
        "partial_pack": True,
        "replacements": selected,
        "source_variant_ids": selected_variant_ids,
        "author_proxy_provenance": _dedupe_author_proxy_items(provenance, limit=4),
        "author_review_items": _dedupe_author_proxy_items(review_items, limit=4),
    }


def _score_safe_band_evidence_pack_section_probes(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_scores: dict[str, Any],
    current_text: str,
    current_scores: dict[str, Any],
    sections: list[SectionUnit],
    variants: list[dict[str, Any]],
    output_dir: Path,
    author_proxy_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    probe_variants = _safe_band_evidence_pack_section_probe_variants(
        sections=sections,
        variants=variants,
    )
    return [
        _score_safe_band_evidence_pack_variant(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_scores=baseline_scores,
            current_text=current_text,
            current_scores=current_scores,
            sections=sections,
            variant=variant,
            output_dir=output_dir,
            label=f"safe_band_evidence_pack_{variant.get('variant_id')}",
            author_proxy_context=author_proxy_context,
        )
        for variant in probe_variants
    ]


def _safe_band_evidence_pack_section_probe_variants(
    *,
    sections: list[SectionUnit],
    variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not sections or not variants:
        return []
    section_by_id = {section.section_id: section for section in sections}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_probe(section_id: str, replacement: str, variant: dict[str, Any], variant_id: str, *, suffix: str = "") -> None:
        signature = (section_id, " ".join(replacement.split()))
        if signature in seen:
            return
        seen.add(signature)
        probe_id = f"section_probe_{_safe_token(section_id)}_{_safe_token(variant_id)}{suffix}"
        rows.append({
            "variant_id": probe_id,
            "partial_pack": True,
            "section_probe": {
                "section_id": section_id,
                "source_variant_id": variant_id,
                **({"normalized": True} if suffix else {}),
            },
            "replacements": [{"section_id": section_id, "text": replacement}],
            "source_variant_ids": [variant_id],
            "author_proxy_provenance": _author_proxy_item_list(variant.get("author_proxy_provenance"), limit=4),
            "author_review_items": _author_proxy_item_list(variant.get("author_review_items"), limit=4),
        })

    for variant_index, variant in enumerate(variants, start=1):
        if not isinstance(variant, dict):
            continue
        variant_id = str(variant.get("variant_id") or f"v{variant_index}")
        for row in variant.get("replacements") if isinstance(variant.get("replacements"), list) else []:
            if not isinstance(row, dict):
                continue
            section_id = str(row.get("section_id") or "").strip()
            replacement = str(row.get("text") or "").strip()
            section = section_by_id.get(section_id)
            if section is None or not replacement:
                continue
            add_probe(section_id, replacement, variant, variant_id)
            normalized = _safe_band_density_length_normalized_replacement(section, replacement)
            if normalized and " ".join(normalized.split()) != " ".join(replacement.split()):
                add_probe(section_id, normalized, variant, variant_id, suffix="_length_normalized")
    return rows


def _safe_band_density_length_normalized_replacement(section: SectionUnit, replacement: str) -> str:
    if not _section_uses_density_safe_band_contract(section):
        return ""
    target_sentence = str((section.metadata or {}).get("target_sentence") or "")
    source_text = str(section.text or "")
    candidate = str(replacement or "").strip()
    if not source_text.strip() or not candidate:
        return ""
    materiality = _safe_band_density_section_repair_materiality(
        source_text=source_text,
        candidate_text=candidate,
        target_sentence=target_sentence,
    )
    if materiality.get("passed"):
        return ""
    source_words = max(1, word_count(source_text))
    min_words = max(1, round(source_words * _safe_band_density_section_repair_min_word_ratio_for_text(source_text)))
    max_words = max(1, round(source_words * _safe_band_density_section_repair_max_word_ratio()))
    if word_count(candidate) <= max_words:
        return ""
    sentences = [sentence.strip() for sentence in _sentences(candidate) if sentence.strip()]
    if len(sentences) < 2:
        return ""

    current = sentences
    visited = {" ".join(current)}
    best_text = ""
    best_key: tuple[float, ...] | None = None
    while len(current) > 1:
        options: list[tuple[tuple[float, ...], list[str], str, dict[str, Any]]] = []
        for index in range(len(current)):
            removed_sentence = current[index]
            trial_sentences = [sentence for pos, sentence in enumerate(current) if pos != index]
            trial_signature = " ".join(trial_sentences)
            if trial_signature in visited:
                continue
            trial_text = " ".join(trial_sentences).strip()
            trial_words = word_count(trial_text)
            if trial_words < min_words:
                continue
            trial_materiality = _safe_band_density_section_repair_materiality(
                source_text=source_text,
                candidate_text=trial_text,
                target_sentence=target_sentence,
            )
            over_budget = max(0, trial_words - max_words)
            removed_source_overlap = _safe_band_token_overlap_ratio(removed_sentence, source_text)
            key = (
                1.0 if trial_materiality.get("passed") else 0.0,
                1.0 if trial_words <= max_words else 0.0,
                _number(trial_materiality.get("changed_sentence_count")),
                1.0 if trial_materiality.get("target_sentence_changed") else 0.0,
                1.0 if (trial_materiality.get("repetition_audit") or {}).get("passed") else 0.0,
                1.0 if (trial_materiality.get("voice_audit") or {}).get("passed") else 0.0,
                -removed_source_overlap,
                -float(over_budget),
                -abs(float(trial_words - source_words)),
            )
            options.append((key, trial_sentences, trial_text, trial_materiality))
        if not options:
            break
        key, current, text, trial_materiality = max(options, key=lambda item: item[0])
        visited.add(" ".join(current))
        if best_key is None or key > best_key:
            best_key = key
            best_text = text
        if trial_materiality.get("passed"):
            return text
    return best_text if best_key and best_key[0] > 0 else ""


def _safe_band_evidence_pack_scored_section_composite_variant(
    rows: list[dict[str, Any]],
    *,
    current_scores: dict[str, Any],
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if _has_safe_band_evidence_repair_movement(row, current_scores=current_scores)
        or _safe_band_evidence_pack_section_safe_for_composition(row)
    ]
    if not eligible:
        return None

    best_by_section: dict[str, dict[str, Any]] = {}
    for row in eligible:
        section_id = _safe_band_evidence_pack_row_single_section_id(row)
        if not section_id:
            continue
        existing = best_by_section.get(section_id)
        if existing is None or _safe_band_evidence_repair_sort_key(row, current_scores=current_scores) > _safe_band_evidence_repair_sort_key(existing, current_scores=current_scores):
            best_by_section[section_id] = row
    if not best_by_section:
        return None

    replacements: list[dict[str, str]] = []
    source_variant_ids: list[str] = []
    provenance: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    for section_id, row in best_by_section.items():
        replacement = str(row.get("text") or "").strip()
        if not replacement:
            continue
        replacements.append({"section_id": section_id, "text": replacement})
        source_variant_ids.append(str(row.get("variant_id") or section_id))
        provenance.extend(_author_proxy_item_list(row.get("author_proxy_provenance"), limit=4))
        review_items.extend(_author_proxy_item_list(row.get("author_review_items"), limit=4))
    if not replacements:
        return None
    return {
        "variant_id": "scored_section_composite",
        "partial_pack": True,
        "replacements": replacements,
        "source_variant_ids": source_variant_ids,
        "author_proxy_provenance": _dedupe_author_proxy_items(provenance, limit=4),
        "author_review_items": _dedupe_author_proxy_items(review_items, limit=4),
    }


def _safe_band_evidence_pack_section_safe_for_composition(row: dict[str, Any]) -> bool:
    if not (row.get("apply_status") or {}).get("applied"):
        return False
    if not _author_proxy_candidate_auto_finalizable(row):
        return False
    materiality = row.get("safe_band_evidence_pack_materiality")
    if not isinstance(materiality, dict) or not materiality.get("passed"):
        return False
    quality = row.get("safe_band_quality_materiality") if isinstance(row.get("safe_band_quality_materiality"), dict) else {}
    if quality and not quality.get("passed"):
        return False
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    if _number(incremental.get("unsafe_cluster_count_delta")) < 0:
        return False
    if _number(incremental.get("unsafe_word_ratio_delta")) < -_safe_band_evidence_repair_unsafe_word_ratio_regression_tolerance():
        return False
    if _number(incremental.get("risky_window_count_delta")) < 0:
        return False
    if (
        _number(incremental.get("topk_calibrated_risk_delta")) < -_safe_band_density_checkpoint_topk_regression_tolerance()
        and not _candidate_topk_calibrated_safe(row)
    ):
        return False
    if _number(incremental.get("ai_delta")) < -_safe_band_density_checkpoint_ai_regression_tolerance():
        return False
    if not _density_checkpoint_authorship_bounds_ok(row, incremental=incremental):
        return False
    return (
        _number(incremental.get("qualifying_text_ai_density_delta")) > 0
        or _number(incremental.get("topk_calibrated_risk_delta")) > 0
        or _number(incremental.get("ai_delta")) > 0
    )


def _safe_band_evidence_pack_row_single_section_id(row: dict[str, Any]) -> str:
    probe = row.get("section_probe") if isinstance(row.get("section_probe"), dict) else {}
    section_id = str(probe.get("section_id") or "").strip()
    if section_id:
        return section_id
    apply_status = row.get("apply_status") if isinstance(row.get("apply_status"), dict) else {}
    applied = apply_status.get("applied_sections") if isinstance(apply_status.get("applied_sections"), list) else []
    if len(applied) == 1 and isinstance(applied[0], dict):
        return str(applied[0].get("section_id") or "").strip()
    return ""


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "")).strip("_")
    return token or "item"


def _safe_band_token_overlap_ratio(text: str, reference: str) -> float:
    tokens = set(_normalized_word_tokens(text))
    if not tokens:
        return 0.0
    reference_tokens = set(_normalized_word_tokens(reference))
    if not reference_tokens:
        return 0.0
    return len(tokens & reference_tokens) / max(1, len(tokens))


def _dedupe_author_proxy_items(items: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("provenance") or ""),
            str(item.get("target_text") or ""),
            str(item.get("generated_text") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max(1, int(limit or 1)):
            break
    return deduped


def _sanitize_safe_band_evidence_pack_variants(
    value: Any,
    *,
    sections: list[SectionUnit],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    section_ids = {section.section_id for section in sections}
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(value if isinstance(value, list) else [], start=1):
        if not isinstance(item, dict):
            rejected.append({"index": index, "reason": "variant_not_object"})
            continue
        variant_id = str(item.get("variant_id") or f"v{index}")[:80]
        replacements: list[dict[str, str]] = []
        seen: set[str] = set()
        for replacement in item.get("replacements") if isinstance(item.get("replacements"), list) else []:
            if not isinstance(replacement, dict):
                continue
            section_id = str(replacement.get("section_id") or "").strip()
            text = str(replacement.get("text") or "").strip()
            if section_id not in section_ids or section_id in seen or not text:
                continue
            integrity = minimal_replacement_text_integrity(text)
            if not integrity.get("passed"):
                rejected.append({
                    "index": index,
                    "variant_id": variant_id,
                    "section_id": section_id,
                    "reason": "replacement_text_integrity_failed",
                    "text_integrity": integrity,
                })
                continue
            replacements.append({"section_id": section_id, "text": text})
            seen.add(section_id)
        if set(seen) != section_ids:
            rejected.append({
                "index": index,
                "variant_id": variant_id,
                "reason": "missing_required_replacements",
                "missing_section_ids": sorted(section_ids - seen),
            })
            continue
        rows.append({
            "variant_id": variant_id,
            "replacements": replacements,
            "author_proxy_provenance": _author_proxy_item_list(item.get("author_proxy_provenance")),
            "author_review_items": _author_proxy_item_list(item.get("author_review_items")),
        })
    return rows, {"valid_variant_count": len(rows), "rejected": rejected}


def _sanitize_safe_band_author_proxy_revision_plan(
    value: Any,
    *,
    sections: list[SectionUnit],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(value, dict):
        return None, {"reason": "plan_not_object"}
    section_ids = [section.section_id for section in sections]
    section_id_set = set(section_ids)
    ledger = value.get("evidence_ledger") if isinstance(value.get("evidence_ledger"), dict) else {}
    ledger_sections_raw = ledger.get("sections") if isinstance(ledger.get("sections"), list) else []
    ledger_sections: list[dict[str, Any]] = []
    seen_ledger: set[str] = set()
    rejected: list[dict[str, Any]] = []

    def text_list(row: dict[str, Any], key: str, *, limit: int = 8, item_limit: int = 500) -> list[str]:
        items: list[str] = []
        for item in row.get(key) if isinstance(row.get(key), list) else []:
            text = str(item or "").strip()
            if text and text not in items:
                items.append(text[:item_limit])
            if len(items) >= limit:
                break
        return items

    for index, row in enumerate(ledger_sections_raw, start=1):
        if not isinstance(row, dict):
            rejected.append({"index": index, "reason": "ledger_section_not_object"})
            continue
        section_id = str(row.get("section_id") or "").strip()
        if section_id not in section_id_set or section_id in seen_ledger:
            rejected.append({"index": index, "section_id": section_id, "reason": "ledger_section_id_invalid_or_duplicate"})
            continue
        seen_ledger.add(section_id)
        ledger_sections.append({
            "section_id": section_id,
            "author_owned_evidence": text_list(row, "author_owned_evidence"),
            "weak_or_generic_claims": text_list(row, "weak_or_generic_claims"),
            "protected_anchors": text_list(row, "protected_anchors"),
            "author_review_gaps": text_list(row, "author_review_gaps"),
        })

    plan_raw = value.get("revision_plan") if isinstance(value.get("revision_plan"), list) else []
    plan_rows: list[dict[str, Any]] = []
    seen_plan: set[str] = set()
    for index, row in enumerate(plan_raw, start=1):
        if not isinstance(row, dict):
            rejected.append({"index": index, "reason": "revision_plan_row_not_object"})
            continue
        section_id = str(row.get("section_id") or "").strip()
        if section_id not in section_id_set or section_id in seen_plan:
            rejected.append({"index": index, "section_id": section_id, "reason": "plan_section_id_invalid_or_duplicate"})
            continue
        seen_plan.add(section_id)
        plan_rows.append({
            "section_id": section_id,
            "section_job": str(row.get("section_job") or "")[:500],
            "required_moves": text_list(row, "required_moves", limit=10),
            "sentences_to_rebuild": text_list(row, "sentences_to_rebuild", limit=8),
            "claim_narrowing": text_list(row, "claim_narrowing", limit=6),
            "materiality_requirement": str(row.get("materiality_requirement") or "")[:500],
            "author_review_items": text_list(row, "author_review_items", limit=8),
        })

    missing_ledger = [section_id for section_id in section_ids if section_id not in seen_ledger]
    missing_plan = [section_id for section_id in section_ids if section_id not in seen_plan]
    if missing_ledger or missing_plan:
        return None, {
            "reason": "plan_missing_required_sections",
            "missing_ledger_section_ids": missing_ledger,
            "missing_plan_section_ids": missing_plan,
            "rejected": rejected,
        }
    return {
        "schema_version": "safe_band_author_proxy_revision_plan.v1",
        "evidence_ledger": {"sections": ledger_sections},
        "revision_plan": plan_rows,
    }, {
        "valid_ledger_section_count": len(ledger_sections),
        "valid_revision_plan_count": len(plan_rows),
        "rejected": rejected,
    }


def _safe_band_author_proxy_revision_plan_response_format(section_count: int) -> dict[str, Any]:
    text_array_schema = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 0,
        "maxItems": 10,
    }
    ledger_section_schema = {
        "type": "object",
        "properties": {
            "section_id": {"type": "string"},
            "author_owned_evidence": text_array_schema,
            "weak_or_generic_claims": text_array_schema,
            "protected_anchors": text_array_schema,
            "author_review_gaps": text_array_schema,
        },
        "required": [
            "section_id",
            "author_owned_evidence",
            "weak_or_generic_claims",
            "protected_anchors",
            "author_review_gaps",
        ],
        "additionalProperties": False,
    }
    plan_row_schema = {
        "type": "object",
        "properties": {
            "section_id": {"type": "string"},
            "section_job": {"type": "string"},
            "required_moves": text_array_schema,
            "sentences_to_rebuild": text_array_schema,
            "claim_narrowing": text_array_schema,
            "materiality_requirement": {"type": "string"},
            "author_review_items": text_array_schema,
        },
        "required": [
            "section_id",
            "section_job",
            "required_moves",
            "sentences_to_rebuild",
            "claim_narrowing",
            "materiality_requirement",
            "author_review_items",
        ],
        "additionalProperties": False,
    }
    count = max(1, int(section_count or 1))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "safe_band_author_proxy_revision_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "evidence_ledger": {
                        "type": "object",
                        "properties": {
                            "sections": {
                                "type": "array",
                                "items": ledger_section_schema,
                                "minItems": count,
                                "maxItems": count,
                            }
                        },
                        "required": ["sections"],
                        "additionalProperties": False,
                    },
                    "revision_plan": {
                        "type": "array",
                        "items": plan_row_schema,
                        "minItems": count,
                        "maxItems": count,
                    },
                },
                "required": ["evidence_ledger", "revision_plan"],
                "additionalProperties": False,
            },
        },
    }


def _safe_band_evidence_pack_response_format(
    variant_count: int,
    section_count: int,
    *,
    include_author_proxy_fields: bool,
) -> dict[str, Any]:
    replacement_schema = {
        "type": "object",
        "properties": {
            "section_id": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["section_id", "text"],
        "additionalProperties": False,
    }
    variant_properties: dict[str, Any] = {
        "variant_id": {"type": "string"},
        "replacements": {
            "type": "array",
            "items": replacement_schema,
            "minItems": max(1, int(section_count or 1)),
            "maxItems": max(1, int(section_count or 1)),
        },
    }
    required = ["variant_id", "replacements"]
    if include_author_proxy_fields:
        item_schema = {
            "type": "object",
            "properties": {
                "item_id": {"type": "string"},
                "provenance": {"type": "string"},
                "target_text": {"type": "string"},
                "generated_text": {"type": "string"},
                "user_input_needed": {"type": "string"},
                "author_task": {"type": "string"},
            },
            "required": ["item_id", "provenance", "target_text", "generated_text", "user_input_needed", "author_task"],
            "additionalProperties": False,
        }
        variant_properties["author_proxy_provenance"] = {"type": "array", "items": item_schema, "minItems": 0, "maxItems": 4}
        variant_properties["author_review_items"] = {"type": "array", "items": item_schema, "minItems": 0, "maxItems": 4}
        required.extend(["author_proxy_provenance", "author_review_items"])
    variant_schema = {
        "type": "object",
        "properties": variant_properties,
        "required": required,
        "additionalProperties": False,
    }
    count = max(1, min(5, int(variant_count or 1)))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "safe_band_evidence_pack_variants",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "variants": {
                        "type": "array",
                        "items": variant_schema,
                        "minItems": count,
                        "maxItems": count,
                    }
                },
                "required": ["variants"],
                "additionalProperties": False,
            },
        },
    }


def _safe_band_evidence_composite_section(current_text: str, sections: list[SectionUnit]) -> SectionUnit | None:
    unique = sorted(
        {
            (section.start_char, section.end_char): section
            for section in sections
            if isinstance(section, SectionUnit)
        }.values(),
        key=lambda section: section.start_char,
    )
    if len(unique) < 2:
        return None
    start = min(section.start_char for section in unique)
    end = max(section.end_char for section in unique)
    if start < 0 or end <= start or end > len(current_text):
        return None
    text = current_text[start:end]
    if not text.strip():
        return None
    max_words = _safe_band_evidence_repair_composite_max_words()
    if word_count(text) > max_words:
        return None
    delimiter = "\n\n" if "\n\n" in text else "\n"
    scanner_focus = [
        (section.metadata or {}).get("scanner_focus")
        for section in unique
        if isinstance((section.metadata or {}).get("scanner_focus"), dict)
    ]
    first = unique[0]
    last = unique[-1]
    return SectionUnit(
        section_id=f"safe_band_evidence_repair_window_{first.section_id}_{last.section_id}",
        heading="Safe-band evidence repair window",
        text=text,
        start_char=start,
        end_char=end,
        paragraph_count=max(1, text.count(delimiter) + 1),
        word_count=word_count(text),
        metadata={
            "unit_type": "safe_band_evidence_repair_composite_window",
            "selection_reason": "composite_top_safe_band_sections",
            "paragraph_delimiter": "blank_line" if delimiter == "\n\n" else "single_newline",
            "target_sentence": str((first.metadata or {}).get("target_sentence") or ""),
            "target_sections": [section.to_dict() for section in unique],
            "scanner_focus": {"sections": scanner_focus},
            "before_context": current_text[max(0, start - 420):start],
            "after_context": current_text[end:min(len(current_text), end + 420)],
        },
    )


def _first_unique_sections(sections: list[SectionUnit], *, limit: int) -> list[SectionUnit]:
    selected: list[SectionUnit] = []
    seen: set[tuple[int, int]] = set()
    for section in sections:
        signature = (section.start_char, section.end_char)
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(section)
        if len(selected) >= max(1, int(limit or 1)):
            break
    return selected


def _section_from_paragraph_containing_text(
    current_text: str,
    needle: str,
    *,
    section_id: str,
    selection_reason: str,
    scanner_focus: dict[str, Any],
) -> SectionUnit | None:
    source = str(current_text or "")
    target = str(needle or "").strip()
    if not source or not target:
        return None
    offset = source.find(target)
    if offset < 0:
        return None
    delimiter = "\n\n" if "\n\n" in source else "\n"
    start = source.rfind(delimiter, 0, offset)
    start = 0 if start < 0 else start + len(delimiter)
    end = source.find(delimiter, offset + len(target))
    end = len(source) if end < 0 else end
    while start < end and source[start] in "\r\n":
        start += 1
    while end > start and source[end - 1] in "\r\n":
        end -= 1
    paragraph = source[start:end]
    if not paragraph.strip():
        return None
    paragraph_index = source[:start].count("\n\n") + 1
    return SectionUnit(
        section_id=section_id,
        heading="Safe-band evidence repair",
        text=paragraph,
        start_char=start,
        end_char=end,
        paragraph_count=max(1, paragraph.count("\n\n") + 1),
        word_count=word_count(paragraph),
        metadata={
            "unit_type": "safe_band_evidence_repair_paragraph",
            "paragraph_index": paragraph_index if delimiter == "\n\n" else source[:start].count("\n") + 1,
            "paragraph_delimiter": "blank_line" if delimiter == "\n\n" else "single_newline",
            "target_sentence": target,
            "selection_reason": selection_reason,
            "scanner_focus": scanner_focus,
            "before_context": source[max(0, start - 420):start],
            "after_context": source[end:min(len(source), end + 420)],
        },
    )


def _safe_band_evidence_repair_should_run(*, current_scores: dict[str, Any], current_goal: dict[str, Any]) -> bool:
    if not _safe_band_evidence_repair_enabled():
        return False
    if _candidate_strict_safe_band_achieved({"candidate_goal": current_goal, "scores": current_scores}):
        return False
    return _safe_band_gap_for_scores(current_scores, current_goal) > 0


def _safe_band_density_section_repair_should_run(*, current_scores: dict[str, Any], current_goal: dict[str, Any]) -> bool:
    if not _safe_band_density_section_repair_enabled():
        return False
    if _candidate_strict_safe_band_achieved({"candidate_goal": current_goal, "scores": current_scores}):
        return False
    contract = _safe_band_kpi_contract(current_scores, current_goal)
    gaps = contract.get("gaps") if isinstance(contract.get("gaps"), dict) else {}
    if _number(gaps.get("qualifying_text_ai_density")) <= 0:
        return False
    if _number(gaps.get("topk_calibrated_risk")) > _safe_band_density_section_repair_max_topk_gap():
        return False
    if _number(current_scores.get("unsafe_cluster_count")) > 0:
        return False
    if _number(current_scores.get("risky_window_count")) > 0:
        return False
    remaining = contract.get("remaining_ai_footprint_drivers") if isinstance(contract.get("remaining_ai_footprint_drivers"), list) else []
    max_topk_gap = _safe_band_density_section_repair_max_topk_gap()
    non_density_blockers = [
        item
        for item in remaining
        if isinstance(item, dict)
        and str(item.get("driver") or "") != "qualifying_text_ai_density"
        and _number(item.get("value")) > _number(item.get("safe_band"))
        and not (
            str(item.get("driver") or "") == "topk_calibrated_risk"
            and max(0.0, _number(item.get("value")) - _number(item.get("safe_band"))) <= max_topk_gap
        )
    ]
    return not non_density_blockers


def _safe_band_density_first_repair_should_run(*, current_scores: dict[str, Any], current_goal: dict[str, Any]) -> bool:
    return _bool_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_FIRST_REPAIR",
        True,
    ) and _safe_band_density_section_repair_should_run(
        current_scores=current_scores,
        current_goal=current_goal,
    )


def _safe_band_density_section_repair_max_topk_gap() -> float:
    return _float_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_MAX_TOPK_GAP",
        1.0,
        minimum=0.0,
        maximum=10.0,
    )


def _safe_band_gap_for_scores(current_scores: dict[str, Any], current_goal: dict[str, Any] | None = None) -> float:
    contract = _safe_band_kpi_contract(current_scores, current_goal)
    gaps = contract.get("gaps") if isinstance(contract.get("gaps"), dict) else {}
    return round(sum(max(0.0, _number(value)) for value in gaps.values()), 3)


def _best_safe_band_evidence_repair_candidate(
    rows: list[dict[str, Any]],
    *,
    current_scores: dict[str, Any],
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if (row.get("apply_status") or {}).get("applied")
        and _author_proxy_candidate_auto_finalizable(row)
    ]
    if not eligible:
        return None
    accepted = [row for row in eligible if _has_safe_band_evidence_repair_movement(row, current_scores=current_scores)]
    if accepted:
        return max(accepted, key=lambda row: _safe_band_evidence_repair_sort_key(row, current_scores=current_scores))
    return max(eligible, key=lambda row: _safe_band_evidence_repair_sort_key(row, current_scores=current_scores))


def _best_safe_band_density_section_candidate(
    rows: list[dict[str, Any]],
    *,
    current_scores: dict[str, Any],
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if (row.get("apply_status") or {}).get("applied")
        and _author_proxy_candidate_auto_finalizable(row)
    ]
    if not eligible:
        return None
    accepted = [row for row in eligible if _has_density_safe_band_checkpoint_movement(row, current_scores=current_scores)]
    if accepted:
        return max(accepted, key=lambda row: _safe_band_evidence_repair_sort_key(row, current_scores=current_scores))
    return max(eligible, key=lambda row: _safe_band_evidence_repair_sort_key(row, current_scores=current_scores))


def _has_safe_band_evidence_repair_movement(
    row: dict[str, Any],
    *,
    current_scores: dict[str, Any],
) -> bool:
    if not (row.get("apply_status") or {}).get("applied"):
        return False
    if not _author_proxy_candidate_auto_finalizable(row):
        return False
    materiality = row.get("safe_band_evidence_materiality") if isinstance(row.get("safe_band_evidence_materiality"), dict) else {}
    if materiality and not materiality.get("passed"):
        return False
    quality = row.get("safe_band_quality_materiality") if isinstance(row.get("safe_band_quality_materiality"), dict) else {}
    if quality and not quality.get("passed"):
        return False
    if _has_density_safe_band_checkpoint_movement(row, current_scores=current_scores):
        return True
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    if _number(incremental.get("ai_delta")) < _safe_band_evidence_repair_min_ai_delta():
        return False
    if _number(incremental.get("ai_authorship_delta")) < _safe_band_evidence_repair_min_authorship_delta():
        return False
    if _number(incremental.get("unsafe_cluster_count_delta")) < 0:
        return False
    if _number(incremental.get("unsafe_word_ratio_delta")) < -_safe_band_evidence_repair_unsafe_word_ratio_regression_tolerance():
        return False
    if _number(incremental.get("risky_window_count_delta")) < 0:
        return False
    if _candidate_strict_safe_band_achieved(row):
        return True
    current_gap = _safe_band_gap_for_scores(current_scores)
    candidate_gap = _candidate_safe_band_gap(row)
    if current_gap - candidate_gap >= _safe_band_evidence_repair_min_gap_delta():
        return True
    return False


def _has_density_safe_band_checkpoint_movement(
    row: dict[str, Any],
    *,
    current_scores: dict[str, Any],
) -> bool:
    materiality = row.get("safe_band_density_section_materiality")
    if not isinstance(materiality, dict):
        pack_materiality = row.get("safe_band_evidence_pack_materiality")
        if isinstance(pack_materiality, dict) and any(
            isinstance(section, dict) and section.get("contract") == "density_section_repair"
            for section in pack_materiality.get("sections") or []
        ):
            materiality = pack_materiality
    if not isinstance(materiality, dict) or not materiality.get("passed"):
        return False
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    if not _density_checkpoint_authorship_bounds_ok(row, incremental=incremental):
        return False
    if _number(incremental.get("unsafe_cluster_count_delta")) < 0:
        return False
    if _number(incremental.get("unsafe_word_ratio_delta")) < -_safe_band_evidence_repair_unsafe_word_ratio_regression_tolerance():
        return False
    if _number(incremental.get("risky_window_count_delta")) < 0:
        return False
    if _number(incremental.get("ai_delta")) < -_safe_band_density_checkpoint_ai_regression_tolerance():
        return False
    density_delta = _number(incremental.get("qualifying_text_ai_density_delta"))
    topk_delta = _number(incremental.get("topk_calibrated_risk_delta"))
    if density_delta < _safe_band_density_checkpoint_min_density_delta():
        return False
    if (
        topk_delta < -_safe_band_density_checkpoint_topk_regression_tolerance()
        and not _candidate_topk_calibrated_safe(row)
    ):
        return False
    current_gap = _safe_band_gap_for_scores(current_scores)
    candidate_gap = _candidate_safe_band_gap(row)
    return current_gap - candidate_gap >= _safe_band_evidence_repair_min_gap_delta()


def _density_checkpoint_authorship_bounds_ok(row: dict[str, Any], *, incremental: dict[str, Any]) -> bool:
    if _number(incremental.get("ai_authorship_delta")) >= _safe_band_evidence_repair_min_authorship_delta():
        return True
    if _number(incremental.get("ai_authorship_delta")) < -_safe_band_density_checkpoint_authorship_regression_tolerance():
        return False
    if _number(incremental.get("ai_delta")) < -_safe_band_density_checkpoint_ai_regression_tolerance():
        return False
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    goal = row.get("candidate_goal") if isinstance(row.get("candidate_goal"), dict) else {}
    gate = goal.get("ai_footprint_gate") if isinstance(goal.get("ai_footprint_gate"), dict) else {}
    thresholds = gate.get("safe_band_thresholds") if isinstance(gate.get("safe_band_thresholds"), dict) else {}
    ai_authorship_limit = _number(thresholds.get("ai_authorship") if thresholds.get("ai_authorship") is not None else 35.0)
    ai_limit = _safe_band_density_checkpoint_max_ai_score()
    return _number(scores.get("ai_authorship")) <= ai_authorship_limit and _number(scores.get("ai")) <= ai_limit


def _safe_band_evidence_repair_sort_key(
    row: dict[str, Any],
    *,
    current_scores: dict[str, Any],
) -> tuple[float, ...]:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    current_gap = _safe_band_gap_for_scores(current_scores)
    candidate_gap = _candidate_safe_band_gap(row)
    return (
        1.0 if _author_proxy_candidate_auto_finalizable(row) else 0.0,
        1.0 if _candidate_strict_safe_band_achieved(row) else 0.0,
        1.0 if (row.get("safe_band_evidence_materiality") or {}).get("passed") else 0.0,
        1.0 if (row.get("safe_band_quality_materiality") or {}).get("passed") else 0.0,
        current_gap - candidate_gap,
        -candidate_gap,
        _number(incremental.get("qualifying_text_ai_density_delta")),
        _number(incremental.get("topk_calibrated_risk_delta")),
        _number(incremental.get("ai_delta")),
        _number(incremental.get("ai_authorship_delta")),
        _number(incremental.get("unsafe_cluster_count_delta")),
        _number(incremental.get("risky_window_count_delta")),
        _author_proxy_quality_sort_value(row),
    )


def _safe_band_density_section_repair_materiality(
    *,
    source_text: str,
    candidate_text: str,
    target_sentence: str = "",
) -> dict[str, Any]:
    base = _safe_band_evidence_repair_materiality(
        source_text=source_text,
        candidate_text=candidate_text,
        target_sentence=target_sentence,
    )
    density_required = _safe_band_density_section_repair_min_changed_sentences(source_text)
    source_words = max(1, word_count(str(source_text or "")))
    candidate_words = max(1, word_count(str(candidate_text or "")))
    ratio = candidate_words / source_words
    repetition = _safe_band_density_repetition_audit(source_text, candidate_text)
    voice = _safe_band_density_voice_audit(source_text, candidate_text)
    min_ratio = _safe_band_density_section_repair_min_word_ratio_for_text(source_text)
    max_ratio = _safe_band_density_section_repair_max_word_ratio()
    length_ok = min_ratio <= ratio <= max_ratio
    density_change_ok = _number(base.get("changed_sentence_count")) >= density_required
    duplicate_cleanup = _safe_band_density_duplicate_cleanup_materiality(source_text, candidate_text)
    duplicate_cleanup_ok = bool(duplicate_cleanup.get("passed")) and repetition.get("passed") is True and voice.get("passed") is True
    passed = (
        bool(base.get("passed")) and density_change_ok and length_ok and repetition.get("passed") is True and voice.get("passed") is True
    ) or duplicate_cleanup_ok
    if not density_change_ok:
        reason = "density_section_repair_too_few_changed_sentences"
    elif not base.get("passed"):
        reason = str(base.get("reason") or "near_copy_or_target_route_unchanged")
    elif not length_ok:
        reason = "density_section_repair_length_out_of_bounds"
    elif repetition.get("passed") is not True:
        reason = "density_section_repair_repetition_regression"
    elif voice.get("passed") is not True:
        reason = "density_section_repair_voice_shift"
    elif duplicate_cleanup_ok:
        reason = "density_section_adjacent_duplicate_cleanup"
    else:
        reason = "material_density_section_route_change"
    if duplicate_cleanup_ok:
        reason = "density_section_adjacent_duplicate_cleanup"
    return {
        **base,
        "passed": passed,
        "reason": reason,
        "source_word_count": source_words,
        "candidate_word_count": candidate_words,
        "word_ratio": round(ratio, 4),
        "density_required_changed_sentence_count": density_required,
        "density_min_changed_sentence_ratio": _safe_band_density_section_repair_min_changed_sentence_ratio(),
        "minimum_word_ratio": min_ratio,
        "maximum_word_ratio": max_ratio,
        "repetition_audit": repetition,
        "voice_audit": voice,
        "duplicate_cleanup_audit": duplicate_cleanup,
    }


def _attach_safe_band_quality_materiality(row: dict[str, Any], *, current_text: str) -> None:
    if not isinstance(row, dict) or not (row.get("apply_status") or {}).get("applied"):
        return
    candidate_text = str(row.get("candidate_text") or "")
    if not candidate_text:
        candidate_text = str(row.get("text") or "")
    if not candidate_text or candidate_text == str(current_text or ""):
        return
    row["safe_band_quality_materiality"] = _safe_band_document_quality_materiality(
        source_text=current_text,
        candidate_text=candidate_text,
    )


def _safe_band_document_quality_materiality(*, source_text: str, candidate_text: str) -> dict[str, Any]:
    repetition = _safe_band_density_repetition_audit(source_text, candidate_text)
    return {
        "passed": repetition.get("passed") is True,
        "reason": (
            "document_quality_preserved"
            if repetition.get("passed") is True
            else "document_repetition_regression"
        ),
        "repetition_audit": repetition,
    }


def _safe_band_density_source_voice_profile(text: str) -> dict[str, Any]:
    first_person_sentences = []
    for sentence in _sentences(str(text or "")):
        if _first_person_count(sentence) > 0:
            first_person_sentences.append(sentence)
        if len(first_person_sentences) >= 3:
            break
    contraction_count = _contraction_count(text)
    first_person_count = _first_person_count(text)
    return {
        "first_person_count": first_person_count,
        "contraction_count": contraction_count,
        "first_person_source_sentences": first_person_sentences,
        "voice_instruction": (
            "Keep direct first-person reflective voice and contractions where they are part of the submitted source voice."
            if first_person_count or contraction_count else
            "Keep the existing source voice; do not make the section more detached or formally polished."
        ),
    }


def _safe_band_density_voice_audit(source_text: str, candidate_text: str) -> dict[str, Any]:
    source_first_person = _first_person_count(source_text)
    candidate_first_person = _first_person_count(candidate_text)
    source_contractions = _contraction_count(source_text)
    candidate_contractions = _contraction_count(candidate_text)
    first_person_preserved = source_first_person == 0 or candidate_first_person > 0
    contractions_preserved = source_contractions == 0 or candidate_contractions > 0
    passed = first_person_preserved and contractions_preserved
    return {
        "passed": passed,
        "source_first_person_count": source_first_person,
        "candidate_first_person_count": candidate_first_person,
        "source_contraction_count": source_contractions,
        "candidate_contraction_count": candidate_contractions,
        "first_person_preserved": first_person_preserved,
        "contractions_preserved": contractions_preserved,
    }


def _first_person_count(text: str) -> int:
    markers = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
    return sum(1 for token in _normalized_word_tokens(text) if token in markers)


def _contraction_count(text: str) -> int:
    count = 0
    for raw in str(text or "").split():
        stripped = raw.strip(" \t\r\n.,:;!?()[]{}\"“”‘’")
        if "'" not in stripped and "’" not in stripped:
            continue
        letters = [char for char in stripped if char.isalnum()]
        if len(letters) >= 2:
            count += 1
    return count


def _safe_band_density_repetition_audit(source_text: str, candidate_text: str) -> dict[str, Any]:
    source_sentence_counts = _normalized_sentence_counts(source_text)
    candidate_sentence_counts = _normalized_sentence_counts(candidate_text)
    repeated_sentences = [
        key
        for key, count in candidate_sentence_counts.items()
        if count > max(1, source_sentence_counts.get(key, 0)) and len(key.split()) >= 8
    ]
    source_ngrams = _normalized_ngram_counts(source_text, size=6)
    candidate_ngrams = _normalized_ngram_counts(candidate_text, size=6)
    repeated_ngrams = [
        key
        for key, count in candidate_ngrams.items()
        if count > max(1, source_ngrams.get(key, 0))
    ]
    max_repeated_ngrams = _safe_band_density_section_repair_max_new_repeated_ngrams()
    passed = not repeated_sentences and len(repeated_ngrams) <= max_repeated_ngrams
    return {
        "passed": passed,
        "new_repeated_sentence_count": len(repeated_sentences),
        "new_repeated_ngram_count": len(repeated_ngrams),
        "max_new_repeated_ngrams": max_repeated_ngrams,
        "repeated_sentence_examples": repeated_sentences[:3],
        "repeated_ngram_examples": repeated_ngrams[:5],
    }


def _normalized_sentence_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sentence in _sentences(str(text or "")):
        key = " ".join(_normalized_word_tokens(sentence))
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _normalized_ngram_counts(text: str, *, size: int) -> dict[str, int]:
    tokens = _normalized_word_tokens(text)
    width = max(2, int(size or 2))
    counts: dict[str, int] = {}
    if len(tokens) < width:
        return counts
    for index in range(0, len(tokens) - width + 1):
        window = tokens[index:index + width]
        if len(set(window)) < max(3, width - 2):
            continue
        key = " ".join(window)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _normalized_word_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in str(text or "").casefold().split():
        cleaned = "".join(char for char in raw if char.isalnum())
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _safe_band_evidence_repair_materiality(
    *,
    source_text: str,
    candidate_text: str,
    target_sentence: str,
) -> dict[str, Any]:
    source_sentences = [sentence.strip() for sentence in _sentences(source_text) if sentence.strip()]
    candidate = str(candidate_text or "")
    if not source_sentences or not candidate.strip():
        return {
            "passed": False,
            "reason": "empty_source_or_candidate",
            "changed_sentence_count": 0,
            "required_changed_sentence_count": 1,
        }
    unchanged = [sentence for sentence in source_sentences if sentence in candidate]
    changed_count = max(0, len(source_sentences) - len(unchanged))
    required = _safe_band_evidence_repair_min_changed_sentences(source_text)
    target = str(target_sentence or "").strip()
    target_changed = not target or target not in candidate
    passed = changed_count >= required and target_changed
    reason = "material_paragraph_route_change" if passed else "near_copy_or_target_route_unchanged"
    return {
        "passed": passed,
        "reason": reason,
        "changed_sentence_count": changed_count,
        "required_changed_sentence_count": required,
        "source_sentence_count": len(source_sentences),
        "unchanged_sentence_count": len(unchanged),
        "target_sentence_changed": target_changed,
    }


def _safe_band_evidence_repair_min_changed_sentences(source_text: str) -> int:
    source_count = len([sentence for sentence in _sentences(source_text) if sentence.strip()])
    configured = _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_REPAIR_MIN_CHANGED_SENTENCES", 2, minimum=1, maximum=8)
    return max(1, min(configured, source_count))


def _safe_band_evidence_repair_variant_goal(index: int) -> str:
    goals = [
        "Rebuild the paragraph from the reflection practice: concrete classroom action -> student evidence -> narrow teaching implication.",
        "Rebuild the paragraph around the Johnny contrast: group reflection limit -> support need -> what the teacher can and cannot infer.",
        "Rebuild the paragraph as a process account: trust-building action -> reflection routine -> evidence gathered -> limitation.",
        "Rebuild the paragraph by changing sentence order while preserving every concrete fact and keeping the claim narrower.",
        "Rebuild the paragraph with shorter uneven sentence routes and less generic explanatory closure.",
    ]
    index = max(1, min(len(goals), int(index or 1)))
    return goals[index - 1]


def _safe_band_evidence_pack_variant_goal(index: int) -> str:
    goals = [
        "Coordinate all replacements around author evidence: concrete classroom action, observed student response, and narrow teaching implication.",
        "Coordinate all replacements around content gaps: replace generic claims with source-owned specifics and mark unsupported bridges for author review.",
        "Coordinate all replacements around sentence-route diversity: change starts, evidence placement, and paragraph endings without changing facts.",
        "Coordinate all replacements around practical teaching process: action -> evidence -> limit -> next decision.",
        "Coordinate all replacements around lower qualifying density: remove broad claims and spend words on already-submitted evidence.",
    ]
    index = max(1, min(len(goals), int(index or 1)))
    return goals[index - 1]


def _safe_band_evidence_repair_enabled() -> bool:
    return _bool_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_REPAIR_ENABLED", True)


def _safe_band_evidence_pack_enabled() -> bool:
    return _bool_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_PACK_ENABLED", True)


def _safe_band_author_proxy_plan_enabled() -> bool:
    return _bool_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_AUTHOR_PROXY_PLAN_ENABLED", True)


def _safe_band_evidence_pack_composite_enabled() -> bool:
    return _bool_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_PACK_COMPOSITE_ENABLED", False)


def _safe_band_controlled_operation_enabled() -> bool:
    return _bool_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_CONTROLLED_OPERATION_ENABLED", True)


def _safe_band_controlled_operation_target_limit() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_CONTROLLED_OPERATION_TARGET_LIMIT", 5, minimum=1, maximum=12)


def _safe_band_controlled_operation_round_limit() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_CONTROLLED_OPERATION_ROUNDS", 3, minimum=1, maximum=8)


def _safe_band_controlled_operation_min_word_ratio() -> float:
    return _float_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_CONTROLLED_OPERATION_MIN_WORD_RATIO", 0.95, minimum=0.5, maximum=1.0)


def _safe_band_controlled_operation_min_suffix_words() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_CONTROLLED_OPERATION_MIN_SUFFIX_WORDS", 4, minimum=1, maximum=20)


def _safe_band_sentence_replacement_enabled() -> bool:
    return _bool_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_SENTENCE_REPLACEMENT_ENABLED", True)


def _safe_band_sentence_replacement_round_limit() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_SENTENCE_REPLACEMENT_ROUNDS", 3, minimum=1, maximum=5)


def _safe_band_sentence_replacement_target_limit() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_SENTENCE_REPLACEMENT_TARGET_LIMIT", 3, minimum=1, maximum=6)


def _safe_band_sentence_replacement_variant_count() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_SENTENCE_REPLACEMENT_VARIANTS", 5, minimum=1, maximum=5)


def _safe_band_sentence_replacement_variant_goal(index: int) -> str:
    goals = [
        "Same meaning, lower safe-band gap: reduce top-k route and qualifying density together using concrete source wording.",
        "Density-first replacement: remove broad explanatory closure, keep facts, and write a shorter practical-action sentence.",
        "Authorship-preserving replacement: keep author viewpoint and source anchors while making the route less template-like.",
        "Context-linked replacement: connect the sentence to its before/after context without adding new facts.",
        "Plain-language replacement: use direct classroom/process wording and avoid polished academic phrasing.",
    ]
    index = max(1, min(len(goals), int(index or 1)))
    return goals[index - 1]


def _safe_band_density_section_repair_enabled() -> bool:
    return _bool_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_REPAIR_ENABLED", True)


def _safe_band_density_section_repair_round_limit() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_REPAIR_ROUNDS", 2, minimum=1, maximum=4)


def _safe_band_density_section_repair_section_limit() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_REPAIR_SECTION_LIMIT", 2, minimum=1, maximum=5)


def _safe_band_density_section_repair_variant_count() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_REPAIR_VARIANTS", 3, minimum=1, maximum=5)


def _safe_band_density_section_repair_min_section_words() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_REPAIR_MIN_SECTION_WORDS", 80, minimum=20, maximum=240)


def _safe_band_density_section_repair_max_section_words() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_REPAIR_MAX_SECTION_WORDS", 260, minimum=80, maximum=900)


def _safe_band_density_section_repair_min_word_ratio() -> float:
    return _float_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_REPAIR_MIN_WORD_RATIO", 0.88, minimum=0.5, maximum=1.0)


def _safe_band_density_section_repair_min_word_ratio_for_text(source_text: str) -> float:
    base = _safe_band_density_section_repair_min_word_ratio()
    if word_count(str(source_text or "")) < _safe_band_density_section_repair_long_section_word_threshold():
        return base
    return min(base, _safe_band_density_section_repair_long_section_min_word_ratio())


def _safe_band_density_section_repair_long_section_word_threshold() -> int:
    return _int_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_REPAIR_LONG_SECTION_WORD_THRESHOLD",
        220,
        minimum=80,
        maximum=900,
    )


def _safe_band_density_section_repair_long_section_min_word_ratio() -> float:
    return _float_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_REPAIR_LONG_SECTION_MIN_WORD_RATIO",
        0.82,
        minimum=0.5,
        maximum=1.0,
    )


def _safe_band_density_section_repair_max_word_ratio() -> float:
    return _float_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_REPAIR_MAX_WORD_RATIO", 1.12, minimum=1.0, maximum=1.8)


def _safe_band_density_section_repair_max_new_repeated_ngrams() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_REPAIR_MAX_NEW_REPEATED_NGRAMS", 1, minimum=0, maximum=12)


def _safe_band_density_section_repair_min_changed_sentence_ratio() -> float:
    return _float_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_REPAIR_MIN_CHANGED_SENTENCE_RATIO",
        0.4,
        minimum=0.0,
        maximum=1.0,
    )


def _safe_band_density_section_repair_min_changed_sentences(source_text: str) -> int:
    source_count = len([sentence for sentence in _sentences(source_text) if sentence.strip()])
    if source_count <= 0:
        return 1
    base = _safe_band_evidence_repair_min_changed_sentences(source_text)
    ratio_required = int((source_count * _safe_band_density_section_repair_min_changed_sentence_ratio()) + 0.999)
    configured_cap = _int_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_SECTION_REPAIR_MAX_CHANGED_SENTENCE_REQUIREMENT",
        6,
        minimum=1,
        maximum=12,
    )
    return max(1, min(source_count, configured_cap, max(base, ratio_required)))


def _safe_band_density_section_repair_variant_goal(index: int) -> str:
    goals = [
        "Density-only section rebuild: narrow broad claims and spend words on submitted evidence already in the section.",
        "Grounding-first rebuild: keep citations and anchors, replace unsupported certainty with source-owned limits.",
        "Author-proxy rebuild: preserve the author's viewpoint while removing smooth generic closure and repeated ideas.",
        "Practical-action rebuild: move from abstract claim to teaching action, observed constraint, and limited implication.",
        "Coverage-preserving rebuild: change the section route without compressing the author's evidence.",
    ]
    index = max(1, min(len(goals), int(index or 1)))
    return goals[index - 1]


def _safe_band_evidence_pack_variant_count() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_PACK_VARIANTS", 3, minimum=1, maximum=5)


def _safe_band_evidence_pack_section_limit() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_PACK_SECTION_LIMIT", 4, minimum=2, maximum=6)


def _safe_band_evidence_pack_max_section_words() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_PACK_MAX_SECTION_WORDS", 320, minimum=80, maximum=700)


def _safe_band_evidence_pack_max_source_words() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_PACK_MAX_SOURCE_WORDS", 700, minimum=160, maximum=1200)


def _safe_band_evidence_pack_partial_min_sections() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_PACK_PARTIAL_MIN_SECTIONS", 2, minimum=1, maximum=6)


def _safe_band_evidence_repair_variant_count() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_REPAIR_VARIANTS", 3, minimum=1, maximum=5)


def _safe_band_evidence_repair_section_limit() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_REPAIR_SECTION_LIMIT", 3, minimum=1, maximum=8)


def _safe_band_evidence_repair_composite_window_enabled() -> bool:
    return _bool_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_REPAIR_COMPOSITE_WINDOW", True)


def _safe_band_evidence_repair_composite_max_words() -> int:
    return _int_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_REPAIR_COMPOSITE_MAX_WORDS", 800, minimum=120, maximum=1600)


def _safe_band_evidence_repair_min_gap_delta() -> float:
    return _float_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_REPAIR_MIN_GAP_DELTA", 0.1, minimum=0.0, maximum=20.0)


def _safe_band_evidence_repair_min_ai_delta() -> float:
    return _float_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_REPAIR_MIN_AI_DELTA", 0.0, minimum=-10.0, maximum=20.0)


def _safe_band_evidence_repair_min_authorship_delta() -> float:
    return _float_env("DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_REPAIR_MIN_AUTHORSHIP_DELTA", 0.0, minimum=-10.0, maximum=20.0)


def _safe_band_evidence_repair_unsafe_word_ratio_regression_tolerance() -> float:
    return _float_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_BAND_EVIDENCE_REPAIR_UNSAFE_WORD_RATIO_REGRESSION_TOLERANCE",
        0.1,
        minimum=0.0,
        maximum=5.0,
    )


def _safe_band_density_checkpoint_min_density_delta() -> float:
    return _float_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_CHECKPOINT_MIN_DENSITY_DELTA",
        0.5,
        minimum=0.0,
        maximum=10.0,
    )


def _safe_band_density_checkpoint_ai_regression_tolerance() -> float:
    return _float_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_CHECKPOINT_AI_REGRESSION_TOLERANCE",
        1.0,
        minimum=0.0,
        maximum=5.0,
    )


def _safe_band_density_checkpoint_authorship_regression_tolerance() -> float:
    return _float_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_CHECKPOINT_AUTHORSHIP_REGRESSION_TOLERANCE",
        1.0,
        minimum=0.0,
        maximum=5.0,
    )


def _safe_band_density_checkpoint_max_ai_score() -> float:
    return _float_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_CHECKPOINT_MAX_AI_SCORE",
        35.0,
        minimum=0.0,
        maximum=100.0,
    )


def _safe_band_density_checkpoint_topk_regression_tolerance() -> float:
    return _float_env(
        "DRAFTPROOF_REWRITE_V5_SAFE_BAND_DENSITY_CHECKPOINT_TOPK_REGRESSION_TOLERANCE",
        0.25,
        minimum=0.0,
        maximum=10.0,
    )


def _final_topk_sentence_route_should_run(*, current_scores: dict[str, Any], density_gate: dict[str, Any]) -> bool:
    if not _final_topk_sentence_route_enabled():
        return False
    topk = _number(current_scores.get("topk"))
    calibrated = _number(current_scores.get("topk_calibrated_risk"))
    if topk >= _float_env("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_MIN_TOPK", 72.0, minimum=0.0, maximum=100.0):
        return True
    if calibrated >= _float_env("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_MIN_CALIBRATED_RISK", 30.0, minimum=0.0, maximum=100.0):
        return True
    return bool(density_gate.get("top_sentence_targets"))


def _final_topk_sentence_route_response_format(variant_count: int, target_count: int) -> dict[str, Any]:
    repairs_schema = {
        "type": "object",
        "properties": {
            "target_id": {"type": "string"},
            "sentence_job": {"type": "string"},
            "current_route": {"type": "string"},
            "repair_route": {"type": "string"},
            "after": {"type": "string"},
        },
        "required": ["target_id", "sentence_job", "current_route", "repair_route", "after"],
        "additionalProperties": False,
    }
    variant_schema = {
        "type": "object",
        "properties": {
            "variant_id": {"type": "string"},
            "repairs": {
                "type": "array",
                "items": repairs_schema,
                "minItems": max(1, int(target_count or 1)),
                "maxItems": max(1, int(target_count or 1)),
            },
        },
        "required": ["variant_id", "repairs"],
        "additionalProperties": False,
    }
    count = max(1, min(5, int(variant_count or 1)))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "final_topk_sentence_route_variants",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "variants": {
                        "type": "array",
                        "items": variant_schema,
                        "minItems": count,
                        "maxItems": count,
                    },
                },
                "required": ["variants"],
                "additionalProperties": False,
            },
        },
    }


def _final_topk_sentence_route_enabled() -> bool:
    return _bool_env("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_ENABLED", True)


def _final_topk_sentence_route_target_limit() -> int:
    return _int_env("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_TARGET_LIMIT", 8, minimum=1, maximum=16)


def _final_topk_sentence_route_batch_size() -> int:
    return _int_env("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_BATCH_SIZE", 2, minimum=1, maximum=4)


def _final_topk_sentence_route_variant_count() -> int:
    return _int_env("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_VARIANTS", 4, minimum=1, maximum=5)


def _final_topk_sentence_route_min_topk_delta() -> float:
    return _float_env("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_MIN_TOPK_DELTA", 0.01, minimum=0.0, maximum=20.0)


def _final_topk_sentence_route_min_calibrated_delta() -> float:
    return _float_env("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_MIN_CALIBRATED_DELTA", 0.0, minimum=0.0, maximum=50.0)


def _final_topk_sentence_route_min_ai_delta() -> float:
    return _float_env("DRAFTPROOF_FINAL_TOPK_SENTENCE_ROUTE_MIN_AI_DELTA", 0.0, minimum=-10.0, maximum=20.0)


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


def _best_borderline_verdict_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if (row.get("apply_status") or {}).get("applied")]
    if not eligible:
        return None
    accepted = [row for row in eligible if _has_borderline_verdict_movement(row)]
    if accepted:
        return max(accepted, key=_borderline_verdict_sort_key)
    return max(eligible, key=_borderline_verdict_sort_key)


def _best_balanced_ai_topk_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if (row.get("apply_status") or {}).get("applied") and _has_balanced_ai_topk_movement(row)
    ]
    if not eligible:
        return None
    return max(eligible, key=_balanced_ai_topk_sort_value)


def _best_full_document_candidate(rows: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    eligible = [row for row in rows if isinstance(row, dict) and _has_full_document_fallback_movement(row)]
    if not eligible:
        return None
    return max(eligible, key=_full_document_candidate_sort_key)


def _full_document_candidate_beats_scores(row: dict[str, Any] | None, current_scores: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    if not _has_full_document_fallback_movement(row):
        return False
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    if _ai_first_density_safe_candidate_beats_scores(row, current_scores):
        return True
    if _would_discard_structural_progress(scores, current_scores):
        return False
    current_row = {
        "apply_status": {"applied": True},
        "scores": current_scores,
    }
    return _full_document_candidate_sort_key(row) > _full_document_candidate_sort_key(current_row)


def _would_discard_structural_progress(candidate_scores: dict[str, Any], current_scores: dict[str, Any]) -> bool:
    for key in (
        "unsafe_cluster_count_delta",
        "risky_window_count_delta",
        "topk_calibrated_risk_delta",
        "topk_delta",
    ):
        current_value = _number(current_scores.get(key))
        if current_value > 0 and _number(candidate_scores.get(key)) < current_value:
            return True
    return False


def _ai_first_density_safe_candidate_beats_scores(row: dict[str, Any], current_scores: dict[str, Any]) -> bool:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    if not _candidate_density_safe(row):
        return False
    current_unsafe_clusters = _number(current_scores.get("unsafe_cluster_count"))
    candidate_unsafe_clusters = _number(scores.get("unsafe_cluster_count"))
    if candidate_unsafe_clusters > current_unsafe_clusters:
        return False
    current_risky_windows = _number(current_scores.get("risky_window_count"))
    candidate_risky_windows = _number(scores.get("risky_window_count"))
    if candidate_risky_windows > current_risky_windows:
        return False
    current_unsafe_ratio = _number(current_scores.get("unsafe_word_ratio"))
    candidate_unsafe_ratio = _number(scores.get("unsafe_word_ratio"))
    if candidate_unsafe_ratio - current_unsafe_ratio > _safe_band_evidence_repair_unsafe_word_ratio_regression_tolerance():
        return False
    current_calibrated = _number(current_scores.get("topk_calibrated_risk"))
    candidate_calibrated = _number(scores.get("topk_calibrated_risk"))
    if candidate_calibrated > current_calibrated + _safe_band_density_checkpoint_topk_regression_tolerance():
        return False
    current_ai = _number(current_scores.get("ai"))
    candidate_ai = _number(scores.get("ai"))
    if current_ai <= 0 or candidate_ai <= 0:
        return False
    min_ai_gain = _float_env(
        "DRAFTPROOF_REWRITE_V5_AI_FIRST_FALLBACK_MIN_AI_GAIN",
        0.25,
        minimum=0.0,
        maximum=20.0,
    )
    if current_ai - candidate_ai < min_ai_gain:
        return False
    max_topk_regression = _float_env(
        "DRAFTPROOF_REWRITE_V5_AI_FIRST_FALLBACK_MAX_TOPK_REGRESSION",
        1.5,
        minimum=0.0,
        maximum=20.0,
    )
    current_topk = _number(current_scores.get("topk"))
    candidate_topk = _number(scores.get("topk"))
    if current_topk > 0 and candidate_topk > 0 and candidate_topk - current_topk > max_topk_regression:
        return False
    return True


def _candidate_density_safe(row: dict[str, Any]) -> bool:
    goal = row.get("candidate_goal") if isinstance(row.get("candidate_goal"), dict) else {}
    density = goal.get("eligible_span_density_gate") if isinstance(goal.get("eligible_span_density_gate"), dict) else {}
    if "safe" in density:
        return bool(density.get("safe"))
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    unsafe_clusters = _number(scores.get("unsafe_cluster_count"))
    return unsafe_clusters <= _float_env(
        "DRAFTPROOF_REWRITE_V5_AI_FIRST_FALLBACK_MAX_UNSAFE_CLUSTERS",
        4.0,
        minimum=0.0,
        maximum=20.0,
    )


def _full_document_candidate_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return (
        1.0 if _author_proxy_candidate_auto_finalizable(row) else 0.0,
        1.0 if _candidate_strict_safe_band_achieved(row) else 0.0,
        -_candidate_safe_band_gap(row),
        _number(scores.get("ai_delta")),
        _number(scores.get("external_delta")),
        _number(scores.get("risky_window_count_delta")),
        _number(scores.get("unsafe_cluster_count_delta")),
        _number(scores.get("topk_calibrated_risk_delta")),
        _number(scores.get("topk_delta")),
        _number(scores.get("qualifying_text_ai_density_delta")),
        _number(scores.get("unsafe_word_ratio_delta")),
        _number(scores.get("rank_delta")),
        _author_proxy_quality_sort_value(row),
    )


def _has_full_document_fallback_movement(row: dict[str, Any]) -> bool:
    if not (row.get("apply_status") or {}).get("applied"):
        return False
    if not _author_proxy_candidate_auto_finalizable(row):
        return False
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    if _number(scores.get("ai_delta")) < 0:
        return False
    if _number(scores.get("topk_calibrated_risk_delta")) < 0:
        return False
    if _number(scores.get("unsafe_cluster_count_delta")) < 0:
        return False
    if _number(scores.get("risky_window_count_delta")) < 0:
        return False
    return any(
        _number(scores.get(key)) > 0
        for key in (
            "ai_delta",
            "topk_calibrated_risk_delta",
            "topk_delta",
            "external_delta",
            "qualifying_text_ai_density_delta",
            "unsafe_cluster_count_delta",
            "risky_window_count_delta",
            "unsafe_word_ratio_delta",
        )
    )


def _author_proxy_candidate_auto_finalizable(row: dict[str, Any]) -> bool:
    audit = row.get("author_proxy_audit") if isinstance(row.get("author_proxy_audit"), dict) else {}
    if not audit or not audit.get("active"):
        return True
    safety_gate = audit.get("safety_gate") if isinstance(audit.get("safety_gate"), dict) else {}
    return safety_gate.get("passed") is not False


def _risky_window_cleanup_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return (
        _number(incremental.get("risky_window_count_delta")),
        _number(incremental.get("ai_delta")),
        _number(incremental.get("topk_delta")),
        _number(incremental.get("unsafe_cluster_count_delta")),
        _number(incremental.get("rank_delta")),
        _number(incremental.get("external_delta")),
        _number(scores.get("ai_delta")),
        _number(scores.get("topk_delta")),
        _number(scores.get("unsafe_cluster_count_delta")),
        _number(scores.get("rank_delta")),
        _author_proxy_quality_sort_value(row),
    )


def _unsafe_cluster_cleanup_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return (
        _number(incremental.get("unsafe_cluster_count_delta")),
        _number(incremental.get("ai_delta")),
        _number(incremental.get("topk_delta")),
        _number(incremental.get("external_delta")),
        _number(incremental.get("rank_delta")),
        _number(scores.get("unsafe_cluster_count_delta")),
        _number(scores.get("ai_delta")),
        _number(scores.get("topk_delta")),
        _number(scores.get("rank_delta")),
        _author_proxy_quality_sort_value(row),
    )


def _borderline_verdict_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    boundary_crossed = 1.0 if _borderline_verdict_candidate_crosses_boundary(row) else 0.0
    return (
        boundary_crossed,
        _borderline_verdict_boundary_margin(scores),
        _number(incremental.get("ai_delta")),
        _number(incremental.get("external_delta")),
        _number(incremental.get("ai_authorship_delta")),
        _number(incremental.get("qualifying_text_ai_density_delta")),
        _number(incremental.get("topk_calibrated_risk_delta")),
        _number(incremental.get("topk_delta")),
        _number(scores.get("ai_delta")),
        _number(scores.get("external_delta")),
        _number(scores.get("rank_delta")),
        _author_proxy_quality_sort_value(row),
    )


def _borderline_verdict_boundary_margin(scores: dict[str, Any]) -> float:
    if not isinstance(scores, dict):
        return 0.0
    target = _borderline_verdict_target_outcome()
    ai_margin = _number(target.get("preferred_ai_below")) - _number(scores.get("ai"))
    authorship_margin = _number(target.get("preferred_authorship_below")) - _number(scores.get("ai_authorship"))
    return max(ai_margin, authorship_margin)


def _balanced_ai_topk_sort_value(row: dict[str, Any]) -> tuple[float, ...]:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    ai_delta = _number(incremental.get("ai_delta"))
    topk_delta = _number(incremental.get("topk_delta"))
    topk_risk_delta = _number(incremental.get("topk_calibrated_risk_delta"))
    external_delta = _number(incremental.get("external_delta"))
    risky_window_delta = _number(incremental.get("risky_window_count_delta"))
    unsafe_word_delta = _number(incremental.get("unsafe_word_ratio_delta"))
    unsafe_cluster_delta = _number(incremental.get("unsafe_cluster_count_delta"))
    rank_delta = _number(incremental.get("rank_delta"))
    balanced_bonus = min(max(ai_delta, 0.0), max(topk_delta, 0.0))
    weighted = (
        (ai_delta * 4.0)
        + (topk_delta * 5.0)
        + (topk_risk_delta * 1.7)
        + (balanced_bonus * 2.0)
        + (external_delta * 0.25)
        + (risky_window_delta * 0.8)
        + (unsafe_word_delta * 0.05)
        + (unsafe_cluster_delta * 0.2)
        + (rank_delta * 0.05)
    )
    return (
        weighted,
        ai_delta,
        topk_delta,
        topk_risk_delta,
        external_delta,
        risky_window_delta,
        unsafe_word_delta,
        unsafe_cluster_delta,
        rank_delta,
        _author_proxy_quality_sort_value(row),
    )


def _direct_scanner_batch_policy() -> str:
    value = os.environ.get("DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_BATCH_POLICY")
    if value is None:
        legacy_run_all = os.environ.get("DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_RUN_ALL_BATCHES")
        if legacy_run_all is not None:
            return "all" if _bool_env("DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_RUN_ALL_BATCHES") else "stop_on_accept"
        return "adaptive"
    normalized = value.strip().casefold()
    if normalized in {"all", "adaptive", "stop_on_accept"}:
        return normalized
    return "adaptive"


def _should_continue_direct_scanner_batches(
    row: dict[str, Any] | None,
    *,
    batch_policy: str,
    batch_index: int,
    max_batches: int,
) -> bool:
    if int(batch_index or 0) >= max(1, int(max_batches or 1)):
        return False
    if batch_policy == "all":
        return True
    if batch_policy == "stop_on_accept":
        return not _has_balanced_ai_topk_movement(row or {})
    return not _direct_scanner_candidate_strong_enough(row)


def _direct_scanner_next_batch_reason(
    row: dict[str, Any] | None,
    *,
    batch_policy: str,
    batch_index: int,
    max_batches: int,
) -> str:
    if int(batch_index or 0) >= max(1, int(max_batches or 1)):
        return "max_batches_reached"
    if batch_policy == "all":
        return "policy_all_batches"
    if batch_policy == "stop_on_accept":
        return "accepted_candidate_found" if _has_balanced_ai_topk_movement(row or {}) else "no_accepted_candidate"
    return "strong_candidate_found" if _direct_scanner_candidate_strong_enough(row) else "candidate_not_strong_enough"


def _direct_scanner_candidate_strong_enough(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict) or not _has_balanced_ai_topk_movement(row):
        return False
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    ai_delta = _number(incremental.get("ai_delta"))
    topk_delta = _number(incremental.get("topk_delta"))
    topk_risk_delta = _number(incremental.get("topk_calibrated_risk_delta"))
    external_delta = _number(incremental.get("external_delta"))
    strong_ai = _float_env(
        "DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_STRONG_AI_DELTA",
        1.0,
        minimum=0.0,
        maximum=20.0,
    )
    strong_topk = _float_env(
        "DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_STRONG_TOPK_DELTA",
        1.0,
        minimum=0.0,
        maximum=20.0,
    )
    strong_topk_risk = _float_env(
        "DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_STRONG_TOPK_RISK_DELTA",
        2.5,
        minimum=0.0,
        maximum=40.0,
    )
    strong_external = _float_env(
        "DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_STRONG_EXTERNAL_DELTA",
        5.0,
        minimum=0.0,
        maximum=50.0,
    )
    if ai_delta >= strong_ai and (topk_delta >= strong_topk or topk_risk_delta >= strong_topk_risk):
        return True
    return ai_delta >= (strong_ai * 1.8) and external_delta >= strong_external


def _has_risky_window_cleanup_movement(row: dict[str, Any]) -> bool:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    return (
        _number(incremental.get("risky_window_count_delta")) > 0
        and _number(incremental.get("unsafe_cluster_count_delta")) >= 0
        and _number(incremental.get("topk_calibrated_risk_delta")) >= 0
        and _number(incremental.get("ai_delta")) >= 0
    )


def _has_borderline_verdict_movement(row: dict[str, Any]) -> bool:
    if not (row.get("apply_status") or {}).get("applied"):
        return False
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    min_topk_delta = _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_MIN_TOPK_DELTA",
        -0.35,
        minimum=-10.0,
        maximum=10.0,
    )
    min_topk_risk_delta = _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_MIN_TOPK_RISK_DELTA",
        -0.35,
        minimum=-10.0,
        maximum=10.0,
    )
    if _number(incremental.get("topk_delta")) < min_topk_delta:
        return False
    if _number(incremental.get("topk_calibrated_risk_delta")) < min_topk_risk_delta:
        return False
    min_ai = _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_MIN_AI_DELTA",
        1.0,
        minimum=0.0,
        maximum=20.0,
    )
    min_external = _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_MIN_EXTERNAL_DELTA",
        1.0,
        minimum=0.0,
        maximum=20.0,
    )
    directional = (
        _number(incremental.get("ai_delta")) >= min_ai
        or _number(incremental.get("external_delta")) >= min_external
        or _number(incremental.get("ai_authorship_delta")) >= min_ai
    )
    if not directional:
        return False
    if (
        _number(incremental.get("risky_window_count_delta")) >= 0
        and _number(incremental.get("unsafe_cluster_count_delta")) >= 0
    ):
        return True
    return _borderline_verdict_candidate_crosses_boundary(row)


def _borderline_verdict_candidate_crosses_boundary(row: dict[str, Any]) -> bool:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    required_score_keys = {
        "ai",
        "ai_authorship",
        "risky_window_count",
        "unsafe_cluster_count",
        "unsafe_word_ratio",
    }
    if not required_score_keys.issubset(set(scores)):
        return False
    if not _borderline_verdict_score_target_met(scores, row=row):
        return False
    if _number(scores.get("risky_window_count")) > _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_MAX_RISKY_WINDOWS_AFTER",
        3.0,
        minimum=0.0,
        maximum=20.0,
    ):
        return False
    if _number(scores.get("unsafe_cluster_count")) > _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_MAX_UNSAFE_CLUSTERS_AFTER",
        5.0,
        minimum=0.0,
        maximum=50.0,
    ):
        return False
    if _number(scores.get("unsafe_word_ratio")) > _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_MAX_UNSAFE_WORD_RATIO_AFTER",
        35.0,
        minimum=0.0,
        maximum=100.0,
    ):
        return False
    return True


def _borderline_verdict_score_target_met(scores: dict[str, Any], row: dict[str, Any] | None = None) -> bool:
    badge_boundary = _borderline_verdict_badge_boundary(row)
    if badge_boundary is not None:
        return badge_boundary
    ai_clear = _number(scores.get("ai")) <= _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_ACCEPT_AI_BELOW",
        45.0,
        minimum=0.0,
        maximum=100.0,
    )
    authorship_clear = _number(scores.get("ai_authorship")) <= _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_ACCEPT_AUTHORSHIP_BELOW",
        45.0,
        minimum=0.0,
        maximum=100.0,
    )
    return ai_clear or authorship_clear


def _borderline_verdict_badge_clears_likely_ai(row: dict[str, Any] | None) -> bool:
    return _borderline_verdict_badge_boundary(row) is True


def _borderline_verdict_badge_boundary(row: dict[str, Any] | None) -> bool | None:
    if not isinstance(row, dict):
        return None
    report = row.get("candidate_report") if isinstance(row.get("candidate_report"), dict) else {}
    badge = report.get("ai_risk_badge") if isinstance(report.get("ai_risk_badge"), dict) else {}
    code = str(
        badge.get("authorship_rating_code")
        or ((badge.get("authorship_rating") or {}).get("code") if isinstance(badge.get("authorship_rating"), dict) else "")
    ).strip().casefold()
    risk_level = str(
        (badge.get("authorship_rating") or {}).get("risk_level")
        if isinstance(badge.get("authorship_rating"), dict)
        else ""
    ).strip().casefold()
    if not code:
        return None
    if risk_level == "high":
        return False
    return code not in {"likely_ai", "ai_generated", "ai_generated_signals"}


def _borderline_rejected_candidate_feedback(
    row: dict[str, Any] | None,
    *,
    current_scores: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(row, dict) or not (row.get("apply_status") or {}).get("applied"):
        return {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    if not _borderline_verdict_score_target_met(scores, row=row):
        return {}
    issues: list[str] = []
    if _number(incremental.get("topk_delta")) < 0:
        issues.append("top-k density worsened; change the route without creating a more predictable sentence path")
    if _number(incremental.get("topk_calibrated_risk_delta")) < 0:
        issues.append("calibrated top-k risk worsened; avoid smoother template-like bridges")
    if _number(incremental.get("risky_window_count_delta")) < 0:
        issues.append("risky windows increased; keep the score movement but make the affected windows less uniformly polished")
    if _number(incremental.get("unsafe_cluster_count_delta")) < 0:
        issues.append("unsafe clusters increased; do not spread the edit across too many new high-pressure clusters")
    if _number(scores.get("unsafe_cluster_count")) > _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_MAX_UNSAFE_CLUSTERS_AFTER",
        5.0,
        minimum=0.0,
        maximum=50.0,
    ):
        issues.append("unsafe cluster count after rewrite is above the allowed borderline tolerance")
    if _number(scores.get("risky_window_count")) > _float_env(
        "DRAFTPROOF_REWRITE_V5_BORDERLINE_MAX_RISKY_WINDOWS_AFTER",
        3.0,
        minimum=0.0,
        maximum=20.0,
    ):
        issues.append("risky window count after rewrite is above the allowed borderline tolerance")
    if not issues:
        return {}
    return {
        "status": "previous_candidate_reduced_verdict_score_but_failed_local_safety",
        "previous_variant_id": row.get("variant_id"),
        "previous_scores": {
            "ai": scores.get("ai"),
            "ai_authorship": scores.get("ai_authorship"),
            "external": scores.get("external"),
            "topk": scores.get("topk"),
            "topk_calibrated_risk": scores.get("topk_calibrated_risk"),
            "risky_window_count": scores.get("risky_window_count"),
            "unsafe_cluster_count": scores.get("unsafe_cluster_count"),
        },
        "current_scores_to_protect": {
            "topk": current_scores.get("topk"),
            "topk_calibrated_risk": current_scores.get("topk_calibrated_risk"),
            "risky_window_count": current_scores.get("risky_window_count"),
            "unsafe_cluster_count": current_scores.get("unsafe_cluster_count"),
        },
        "must_fix": issues,
        "next_attempt": [
            "Preserve the verdict-score improvement pattern, but do not let top-k, risky windows, or unsafe clusters regress beyond tolerance.",
            "Use smaller route changes in the windows that became unsafe; do not rewrite the whole document into a new polished template.",
            "Prefer plain concrete sentence routes over broad abstract bridges.",
        ],
    }


def _has_balanced_ai_topk_movement(row: dict[str, Any]) -> bool:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    ai_delta = _number(incremental.get("ai_delta"))
    topk_delta = _number(incremental.get("topk_delta"))
    topk_risk_delta = _number(incremental.get("topk_calibrated_risk_delta"))
    external_delta = _number(incremental.get("external_delta"))
    risky_window_delta = _number(incremental.get("risky_window_count_delta"))
    unsafe_word_delta = _number(incremental.get("unsafe_word_ratio_delta"))
    minimum_ai_delta = _float_env(
        "DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_MIN_AI_DELTA",
        -0.15,
        minimum=-5.0,
        maximum=5.0,
    )
    minimum_topk_delta = _float_env(
        "DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_MIN_TOPK_DELTA",
        -0.50,
        minimum=-10.0,
        maximum=10.0,
    )
    balanced_floor = _float_env(
        "DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_BALANCED_FLOOR",
        0.40,
        minimum=0.0,
        maximum=10.0,
    )
    if ai_delta < minimum_ai_delta:
        return False
    if topk_delta < minimum_topk_delta and ai_delta < 1.5:
        return False
    if ai_delta < 0 and topk_delta < 1.0:
        return False
    if ai_delta < balanced_floor and topk_delta < balanced_floor:
        return False
    return any(
        value > 0
        for value in (
            ai_delta,
            topk_delta,
            topk_risk_delta,
            external_delta,
            risky_window_delta,
            unsafe_word_delta,
        )
    )


def _has_unsafe_cluster_cleanup_movement(
    row: dict[str, Any],
    *,
    cleanup_index: int | None = None,
) -> bool:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    local = row.get("local_scores") if isinstance(row.get("local_scores"), dict) else {}
    has_cluster_count_drop = _number(incremental.get("unsafe_cluster_count_delta")) > 0
    has_local_cluster_movement = _local_cluster_directionally_improved(local)
    return (
        (has_cluster_count_drop or has_local_cluster_movement)
        and _has_unsafe_cluster_cleanup_marginal_gain(row, cleanup_index=cleanup_index)
        and _number(incremental.get("risky_window_count_delta")) >= 0
        and _number(incremental.get("unsafe_cluster_count_delta")) >= 0
        and _number(incremental.get("topk_calibrated_risk_delta")) >= 0
        and _number(incremental.get("ai_delta")) >= 0
    )


def _has_unsafe_cluster_cleanup_marginal_gain(
    row: dict[str, Any],
    *,
    cleanup_index: int | None = None,
) -> bool:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    if _number(incremental.get("unsafe_cluster_count_delta")) > 0:
        return True
    start_round = _int_env(
        "DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_MIN_GAIN_START_ROUND",
        2,
        minimum=1,
        maximum=12,
    )
    if cleanup_index is None or int(cleanup_index or 0) < start_round:
        return True
    min_ai = _float_env(
        "DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_MIN_AI_DELTA",
        0.5,
        minimum=0.0,
        maximum=20.0,
    )
    min_topk = _float_env(
        "DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_MIN_TOPK_DELTA",
        0.5,
        minimum=0.0,
        maximum=20.0,
    )
    return (
        _number(incremental.get("ai_delta")) >= min_ai
        or _number(incremental.get("topk_delta")) >= min_topk
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
        _number(incremental.get("ai_delta")),
        _number(incremental.get("topk_delta")),
        _number(incremental.get("rank_delta")),
        _number(scores.get("ai_delta")),
        _number(scores.get("topk_delta")),
        _number(scores.get("rank_delta")),
        _author_proxy_quality_sort_value(row),
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


def _has_core_round_acceptance_movement(
    row: dict[str, Any],
    *,
    current_scores: dict[str, Any],
    round_index: int,
) -> bool:
    if not _has_incremental_movement(row):
        return False
    if not _late_core_acceptance_gate_active(current_scores=current_scores, round_index=round_index):
        return True
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    if _number(incremental.get("unsafe_cluster_count_delta")) > 0:
        return True
    if _number(incremental.get("risky_window_count_delta")) > 0:
        return True
    min_ai = _float_env("DRAFTPROOF_REWRITE_V5_LATE_CORE_MIN_AI_DELTA", 0.5, minimum=0.0, maximum=20.0)
    min_topk = _float_env("DRAFTPROOF_REWRITE_V5_LATE_CORE_MIN_TOPK_DELTA", 0.5, minimum=0.0, maximum=20.0)
    return _number(incremental.get("ai_delta")) >= min_ai and _number(incremental.get("topk_delta")) >= min_topk


def _late_core_acceptance_gate_active(
    *,
    current_scores: dict[str, Any],
    round_index: int,
) -> bool:
    start_round = _int_env("DRAFTPROOF_REWRITE_V5_LATE_CORE_GATE_START_ROUND", 5, minimum=1, maximum=20)
    if int(round_index or 0) < start_round:
        return False
    trigger_ai = _float_env("DRAFTPROOF_REWRITE_V5_LATE_CORE_GATE_AFTER_AI_DELTA", 5.0, minimum=0.0, maximum=100.0)
    trigger_topk = _float_env("DRAFTPROOF_REWRITE_V5_LATE_CORE_GATE_AFTER_TOPK_DELTA", 5.0, minimum=0.0, maximum=100.0)
    trigger_structural = _float_env("DRAFTPROOF_REWRITE_V5_LATE_CORE_GATE_AFTER_STRUCTURAL_DELTA", 2.0, minimum=0.0, maximum=100.0)
    return any(
        _number(value) >= threshold
        for value, threshold in (
            (current_scores.get("ai_delta"), trigger_ai),
            (current_scores.get("topk_delta"), trigger_topk),
            (current_scores.get("unsafe_cluster_count_delta"), trigger_structural),
            (current_scores.get("risky_window_count_delta"), trigger_structural),
        )
    )


def _adaptive_writer_enabled(route_plan: dict[str, Any] | None = None) -> bool:
    return _route_plan_valid(route_plan) and _bool_env("DRAFTPROOF_REWRITE_V5_ADAPTIVE_WRITER", True)


def _adaptive_initial_variant_count(requested_count: int, route_plan: dict[str, Any] | None = None) -> int:
    requested = max(1, min(5, int(requested_count or 1)))
    if not _adaptive_writer_enabled(route_plan) or requested <= 2:
        return requested
    configured = _int_env("DRAFTPROOF_REWRITE_V5_ADAPTIVE_INITIAL_VARIANTS", 2, minimum=1, maximum=5)
    return max(1, min(requested, configured))


def _adaptive_retune_variant_count(requested_count: int, route_plan: dict[str, Any] | None = None) -> int:
    requested = max(1, min(5, int(requested_count or 1)))
    if not _adaptive_writer_enabled(route_plan):
        return requested
    configured = _int_env("DRAFTPROOF_REWRITE_V5_ADAPTIVE_RETUNE_VARIANTS", 2, minimum=1, maximum=5)
    return max(1, min(requested, configured))


def _adaptive_writer_feedback(
    rows: list[dict[str, Any]],
    *,
    route_plan: dict[str, Any] | None = None,
    selected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    applied = [row for row in rows if (row.get("apply_status") or {}).get("applied")]
    selected_row = selected or _best_residual_candidate(rows)
    topk_movers = [row for row in applied if _row_has_topk_movement(row)]
    cluster_movers = [row for row in applied if _row_has_cluster_movement(row)]
    unsafe_regressions = [row for row in applied if _row_has_unsafe_cluster_regression(row)]
    reason = "candidate_promising"
    if not applied:
        reason = "no_applied_candidates"
    elif selected_row and _row_has_unsafe_cluster_regression(selected_row):
        reason = "unsafe_cluster_regressed"
    elif _primary_metric_from_plan(route_plan) == "topk_density" and not topk_movers:
        reason = "topk_route_not_moved"
    elif selected_row and not _has_incremental_movement(selected_row):
        reason = "no_incremental_movement"
    return {
        "reason": reason,
        "primary_metric": _primary_metric_from_plan(route_plan),
        "candidate_count": len(rows),
        "applied_count": len(applied),
        "topk_movement_count": len(topk_movers),
        "cluster_movement_count": len(cluster_movers),
        "unsafe_cluster_regression_count": len(unsafe_regressions),
        "selected": _compact_residual_row(selected_row),
        "required_correction": _adaptive_required_correction(reason),
    }


def _adaptive_required_correction(reason: str) -> str:
    if reason == "topk_route_not_moved":
        return "Break the predictable sentence path with clause-route change before polishing wording."
    if reason == "unsafe_cluster_regressed":
        return "Stop broad replacement wording that increases unsafe clusters; keep source content and change only route."
    if reason == "no_incremental_movement":
        return "Use a different route shape instead of another surface paraphrase."
    if reason == "no_applied_candidates":
        return "Return complete replacement text that can be applied to the source cluster."
    return "Continue the promising direction without adding broad filler."


def _should_generate_adaptive_remainder(
    feedback: dict[str, Any],
    *,
    remaining_count: int,
    best_candidate: dict[str, Any] | None,
) -> bool:
    if remaining_count <= 0:
        return False
    if best_candidate and _has_incremental_movement(best_candidate):
        return False
    return str(feedback.get("reason") or "") in {
        "topk_route_not_moved",
        "unsafe_cluster_regressed",
        "no_incremental_movement",
        "no_applied_candidates",
    }


def _should_retune_residual_candidate(
    row: dict[str, Any] | None,
    *,
    route_plan: dict[str, Any] | None = None,
    adaptive_feedback: dict[str, Any] | None = None,
) -> bool:
    if not row or not _needs_retune(row):
        return False
    if not _adaptive_writer_enabled(route_plan):
        return True
    reason = str((adaptive_feedback or {}).get("reason") or "")
    if reason in {"topk_route_not_moved", "unsafe_cluster_regressed", "no_incremental_movement"} and not _has_incremental_movement(row):
        return False
    return _row_has_topk_movement(row) or _row_has_cluster_movement(row) or _has_incremental_movement(row)


def _primary_metric_from_plan(route_plan: dict[str, Any] | None) -> str:
    if not isinstance(route_plan, dict):
        return "mixed"
    return _primary_metric(route_plan.get("primary_metric"))


def _row_has_topk_movement(row: dict[str, Any]) -> bool:
    local = row.get("local_scores") if isinstance(row.get("local_scores"), dict) else {}
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return any(
        _number(value) > 0
        for value in (
            local.get("topk_delta"),
            local.get("topk_calibrated_risk_delta"),
            incremental.get("topk_delta"),
            incremental.get("topk_calibrated_risk_delta"),
            scores.get("topk_delta"),
            scores.get("topk_calibrated_risk_delta"),
        )
    )


def _row_has_cluster_movement(row: dict[str, Any]) -> bool:
    local = row.get("local_scores") if isinstance(row.get("local_scores"), dict) else {}
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return any(
        _number(value) > 0
        for value in (
            local.get("unsafe_cluster_count_delta"),
            local.get("unsafe_word_ratio_delta"),
            incremental.get("unsafe_cluster_count_delta"),
            incremental.get("unsafe_word_ratio_delta"),
            scores.get("unsafe_cluster_count_delta"),
            scores.get("unsafe_word_ratio_delta"),
        )
    )


def _row_has_unsafe_cluster_regression(row: dict[str, Any]) -> bool:
    local = row.get("local_scores") if isinstance(row.get("local_scores"), dict) else {}
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return any(
        _number(value) < 0
        for value in (
            local.get("unsafe_cluster_count_delta"),
            incremental.get("unsafe_cluster_count_delta"),
            scores.get("unsafe_cluster_count_delta"),
        )
    )


def _local_cluster_cleared(local_scores: dict[str, Any]) -> bool:
    return (
        _number(local_scores.get("unsafe_cluster_count")) <= 0
        and _number(local_scores.get("unsafe_word_ratio")) <= 0
    )


def _local_cluster_directionally_improved(local_scores: dict[str, Any]) -> bool:
    if (
        _number(local_scores.get("unsafe_cluster_count_delta")) > 0
        and _number(local_scores.get("unsafe_word_ratio_delta")) >= 0
        and _number(local_scores.get("rank_delta")) >= 0
    ):
        return True
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
        section = _section_from_window_row(current_text, current_report, row, ordinal=ordinal)
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
    selection_mode: str = "scanner",
) -> tuple[SectionUnit, dict[str, Any], tuple[Any, ...]] | None:
    clusters = density.get("top_unsafe_clusters") if isinstance(density, dict) else []
    for ordinal, cluster in _ordered_density_cluster_rows(
        clusters if isinstance(clusters, list) else [],
        selection_mode=selection_mode,
    ):
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


def _ordered_density_cluster_rows(
    clusters: list[Any],
    *,
    selection_mode: str = "scanner",
) -> list[tuple[int, dict[str, Any]]]:
    rows = [
        (ordinal, cluster)
        for ordinal, cluster in enumerate(clusters, start=1)
        if isinstance(cluster, dict)
    ]
    if selection_mode != "clearable":
        return rows
    return sorted(rows, key=lambda item: _clearable_density_cluster_sort_key(item[1], original_ordinal=item[0]))


def _clearable_density_cluster_sort_key(cluster: dict[str, Any], *, original_ordinal: int) -> tuple[float, ...]:
    word_count_value = max(1.0, _number(cluster.get("word_count")))
    sentence_count = max(1.0, _number(cluster.get("sentence_count")))
    return (
        word_count_value,
        sentence_count,
        -_number(cluster.get("risk_score")),
        float(original_ordinal),
    )


def _section_from_window_row(
    current_text: str,
    current_report: dict[str, Any],
    row: dict[str, Any],
    *,
    ordinal: int,
) -> SectionUnit | None:
    sentence_bounds = _sentence_id_bounds(current_report, row.get("sentence_ids"))
    if sentence_bounds is not None:
        start, end = sentence_bounds
    else:
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
    start_index = _optional_int(cluster.get("start_sentence"))
    end_index = _optional_int(cluster.get("end_sentence"))
    direct_start = _optional_int(cluster.get("start_char"))
    direct_end = _optional_int(cluster.get("end_char"))
    if direct_start is not None and direct_end is not None and 0 <= direct_start < direct_end <= len(current_text):
        start = direct_start
        end = direct_end
    else:
        rows = _sentence_rows_by_index(current_report)
        if not rows:
            return None
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


def _sentence_id_bounds(report: dict[str, Any], sentence_ids: Any) -> tuple[int, int] | None:
    ids = [str(item) for item in sentence_ids if str(item or "")] if isinstance(sentence_ids, list) else []
    if not ids:
        return None
    sentence_map = report.get("sentence_map") if isinstance(report, dict) else {}
    if not isinstance(sentence_map, dict):
        return None
    rows: list[dict[str, Any]] = []
    for sentence_id in ids:
        row = sentence_map.get(sentence_id)
        if not isinstance(row, dict):
            return None
        start = _optional_int(row.get("start_char"))
        end = _optional_int(row.get("end_char"))
        if start is None or end is None or end <= start:
            return None
        rows.append({"start_char": start, "end_char": end})
    return (
        min(int(row["start_char"]) for row in rows),
        max(int(row["end_char"]) for row in rows),
    )


def _compact_density_gate(density: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "safe",
        "source",
        "unsafe_sentence_count",
        "unsafe_word_count",
        "unsafe_eligible_word_ratio",
        "longest_unsafe_span_words",
        "unsafe_cluster_count",
        "thresholds",
        "recommended_actions",
    )
    return {key: density.get(key) for key in keys}


def _density_gate_for_report(current_text: str, current_report: dict[str, Any]) -> dict[str, Any]:
    if _bool_env("DRAFTPROOF_REWRITE_V5_USE_REPAIR_UNITS_DENSITY", False):
        return build_preferred_eligible_span_density_contract(current_text, current_report)
    return build_eligible_span_density_contract(current_text, current_report)


def _with_v5_density_gate(text: str, report: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(goal or {})
    enriched["eligible_span_density_gate"] = _density_gate_for_report(text, report)
    return enriched


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
    left = _expand_left_to_sentence_boundary(source, left)
    right = _expand_right_to_sentence_boundary(source, right)
    return left, right


def _expand_left_to_sentence_boundary(source: str, left: int) -> int:
    if left <= 0:
        return 0
    index = max(0, min(len(source), left))
    while index > 0:
        prev = source[index - 1]
        if prev in ".!?":
            break
        if prev == "\n" and index >= 2 and source[index - 2] == "\n":
            break
        index -= 1
    while index < len(source) and source[index].isspace():
        index += 1
    return index


def _expand_right_to_sentence_boundary(source: str, right: int) -> int:
    if right >= len(source):
        return len(source)
    index = max(0, min(len(source), right))
    if _is_clean_right_boundary(source, index):
        return index
    while index < len(source):
        ch = source[index]
        index += 1
        if ch in ".!?":
            while index < len(source) and source[index] in "\"'”’)]}":
                index += 1
            break
        if ch == "\n" and index < len(source) and source[index] == "\n":
            break
    return index


def _section_apply_boundary_integrity(source: str, section: SectionUnit) -> dict[str, Any]:
    text = str(source or "")
    start = int(section.start_char)
    end = int(section.end_char)
    failures: list[str] = []
    if start < 0 or end <= start or end > len(text):
        failures.append("invalid_section_offsets")
    else:
        if not _is_clean_left_boundary(text, start):
            failures.append("left_boundary_inside_sentence")
        if not _is_clean_right_boundary(text, end):
            failures.append("right_boundary_inside_sentence")
    return {
        "passed": not failures,
        "failures": failures,
        "start_char": start,
        "end_char": end,
    }


def _text_integrity_regression(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_metrics = before.get("metrics") if isinstance(before, dict) else {}
    after_metrics = after.get("metrics") if isinstance(after, dict) else {}
    metric_keys = (
        "format_count",
        "control_count",
        "emoji_like_count",
        "embedded_word_period_count",
        "sentence_punctuation_spacing_count",
        "nested_parenthetical_count",
        "repeated_markup_run_count",
    )
    regressions: list[str] = []
    for key in metric_keys:
        if _number(after_metrics.get(key)) > _number(before_metrics.get(key)):
            regressions.append(key)
    before_failures = set(before.get("failures") or []) if isinstance(before, dict) else set()
    after_failures = set(after.get("failures") or []) if isinstance(after, dict) else set()
    new_failures = sorted(str(item) for item in after_failures - before_failures)
    return {
        "passed": not regressions and not new_failures,
        "metric_regressions": regressions,
        "new_failures": new_failures,
    }


def _is_clean_left_boundary(source: str, start: int) -> bool:
    if start <= 0:
        return True
    if start >= len(source):
        return True
    previous_index = _previous_non_space_index(source, start)
    if previous_index is None:
        return True
    return source[previous_index] in ".!?\n"


def _is_clean_right_boundary(source: str, end: int) -> bool:
    if end >= len(source):
        return True
    previous_index = _previous_non_space_index(source, end)
    if previous_index is None:
        return True
    if source[previous_index] in ".!?":
        return True
    next_index = _next_non_space_index(source, end)
    return next_index is None or source[next_index] == "\n"


def _previous_non_space_index(source: str, end: int) -> int | None:
    index = min(len(source), max(0, end)) - 1
    while index >= 0:
        if not source[index].isspace():
            return index
        index -= 1
    return None


def _next_non_space_index(source: str, start: int) -> int | None:
    index = max(0, min(len(source), start))
    while index < len(source):
        if not source[index].isspace():
            return index
        index += 1
    return None


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


def _affected_content_map(*, section: SectionUnit, local_goal: dict[str, Any]) -> list[dict[str, Any]]:
    targets = _local_top_sentence_targets(local_goal)
    target_previews = [str(row.get("preview") or "") for row in targets if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for index, sentence in enumerate(_sentences(section.text), start=1):
        sentence_text = " ".join(str(sentence or "").split())
        if not sentence_text:
            continue
        matched_targets = [
            row for row in targets
            if _text_units_overlap(sentence_text, str(row.get("preview") or ""))
        ]
        is_target = bool(matched_targets) or not target_previews
        rows.append({
            "unit_id": f"u{index:03d}",
            "source_text": sentence_text,
            "is_scanner_target": is_target,
            "scanner_target_ids": [
                str(row.get("sentence_id") or "")
                for row in matched_targets
                if str(row.get("sentence_id") or "").strip()
            ],
            "content_role_hint": _sentence_content_role_hint(index=index, total=len(_sentences(section.text))),
            "planner_job": (
                "diagnose the exact content movement in this unit and assign a required action"
                if is_target
                else "preserve unless needed for continuity with affected units"
            ),
            "preserve_candidates": _source_phrase_anchors(sentence_text)[:5],
        })
        if len(rows) >= 8:
            break
    return rows


def _text_units_overlap(left: str, right: str) -> bool:
    left_text = " ".join(str(left or "").split())
    right_text = " ".join(str(right or "").split())
    if not left_text or not right_text:
        return False
    return left_text in right_text or right_text in left_text


def _sentence_content_role_hint(*, index: int, total: int) -> str:
    if index <= 1:
        return "opening_frame"
    if index >= max(1, total):
        return "closing_or_bridge"
    return "middle_development"


def _retune_focus_from_goal(local_goal: dict[str, Any]) -> list[str]:
    gate = local_goal.get("eligible_span_density_gate") if isinstance(local_goal, dict) else {}
    footprint = local_goal.get("ai_footprint_gate") if isinstance(local_goal, dict) else {}
    after = footprint.get("after") if isinstance(footprint, dict) else {}
    authorship = after.get("authorship_footprint") if isinstance(after, dict) else {}
    focus: list[str] = []
    if isinstance(gate, dict) and not gate.get("top_unsafe_clusters"):
        focus.append("The unsafe sentence density is cleared; keep that route but make the wording less formally paraphrased.")
    if isinstance(authorship, dict) and _number(authorship.get("topk_calibrated_risk")) > 25.0:
        focus.append("The remaining problem is calibrated top-k: change the sentence route and avoid formal substitute terms.")
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
                "instruction": "Add one bridge sentence explaining why this starting point shaped the next source action or claim.",
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
        jobs.append("Sentence 1: restate the starting situation in simple wording while changing the route.")
        if len(beats) >= 4:
            jobs.append("Sentence 2: explain why this starting point shaped the next source action or claim.")
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


def _source_derived_route_seed_texts(section: SectionUnit) -> list[str]:
    beats = _source_event_beats(section.text, limit=8)
    if len(beats) < 3:
        return []
    seeds: list[str] = []
    bridge_seed = _starting_point_bridge_seed(beats)
    if bridge_seed:
        seeds.append(bridge_seed)
    reordered_seed = _reordered_route_seed(section=section, beats=beats)
    if reordered_seed:
        seeds.append(reordered_seed)
    return seeds


def _starting_point_bridge_seed(beats: list[str]) -> str:
    first = _clean_sentence(beats[0])
    rest = [_clean_sentence(beat) for beat in beats[1:] if _clean_sentence(beat)]
    if not first or len(rest) < 2:
        return ""
    bridge = "That starting point mattered before the next step."
    return " ".join([first, bridge, *rest])


def _reordered_route_seed(*, section: SectionUnit, beats: list[str]) -> str:
    blueprint = build_route_blueprint(section=section, local_goal={})
    start_index = int(blueprint.get("start_source_beat_index") or 0)
    if start_index <= 0 or start_index >= len(beats):
        return ""
    ordered = beats[start_index:] + beats[:start_index]
    cleaned = [_clean_sentence(beat) for beat in ordered if _clean_sentence(beat)]
    if len(cleaned) < 3:
        return ""
    return " ".join(cleaned)


def _clean_sentence(value: Any) -> str:
    return " ".join(str(value or "").split())


def _core_section_signature(section: SectionUnit) -> tuple[Any, ...]:
    return (
        section.start_char,
        section.end_char,
        _short_string(section.text, limit=120),
    )


def _local_goal(original_text: str, candidate_text: str) -> dict[str, Any]:
    original_report = _scan_report(original_text)
    candidate_report = original_report if original_text == candidate_text else _scan_report(candidate_text)
    goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=original_report,
        candidate_report=candidate_report,
    ).to_dict()
    return _with_v5_density_gate(candidate_text, candidate_report, goal)


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
        "authorship_rating": _compact_authorship_rating(row.get("candidate_report")),
        "incremental": row.get("incremental"),
        "local_scores": row.get("local_scores"),
        "apply_status": row.get("apply_status"),
        "author_proxy_audit": row.get("author_proxy_audit"),
        "author_proxy_quality": row.get("author_proxy_quality"),
        "author_proxy_provenance": row.get("author_proxy_provenance"),
        "author_review_items": row.get("author_review_items"),
        "safe_band_evidence_materiality": row.get("safe_band_evidence_materiality"),
        "safe_band_density_section_materiality": row.get("safe_band_density_section_materiality"),
        "safe_band_quality_materiality": row.get("safe_band_quality_materiality"),
        "safe_band_evidence_pack_materiality": row.get("safe_band_evidence_pack_materiality"),
        "safe_band_controlled_operation_materiality": row.get("safe_band_controlled_operation_materiality"),
        "selection_policy": row.get("selection_policy"),
        "direct_scanner_batch": row.get("direct_scanner_batch"),
        "text": row.get("text"),
    }


def _compact_authorship_rating(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    badge = report.get("ai_risk_badge") if isinstance(report.get("ai_risk_badge"), dict) else {}
    rating = badge.get("authorship_rating") if isinstance(badge.get("authorship_rating"), dict) else {}
    return {
        "code": badge.get("authorship_rating_code") or rating.get("code"),
        "label": badge.get("authorship_rating_label") or rating.get("label"),
        "risk_level": rating.get("risk_level"),
        "tier": badge.get("tier"),
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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))
