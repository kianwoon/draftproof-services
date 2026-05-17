"""Production adapter for rewrite V5 residual cluster comb-through."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from report.pdf import render_pdf
from report.render_rewrite import render_rewrite_report
from rewrite_v2.goal_contract import RewriteGoalStatus, evaluate_rewrite_goal
from rewrite_v2.pipeline import _extract_original_text, _sentence_comparison
from rewrite_v3.document_units import word_count
from rewrite_v3.pipeline import _scan_report
from rewrite_v4.production import (
    _compact_scan_for_rewrite_report,
    _detect_scores,
    _truncate_text,
)

from .residual_comb import _compact_density_gate, run_v5_residual_cluster_comb_experiment


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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


def _production_config() -> dict[str, Any]:
    return {
        "max_rounds": _int_env("DRAFTPROOF_REWRITE_V5_MAX_ROUNDS", 6, minimum=1, maximum=10),
        "variant_count": _int_env("DRAFTPROOF_REWRITE_V5_VARIANTS", 5, minimum=1, maximum=5),
        "retune_variant_count": _int_env("DRAFTPROOF_REWRITE_V5_RETUNE_VARIANTS", 5, minimum=1, maximum=5),
        "risky_window_cleanup_rounds": _int_env("DRAFTPROOF_REWRITE_V5_RISKY_WINDOW_CLEANUP_ROUNDS", 2, minimum=0, maximum=12),
        "unsafe_cluster_cleanup_rounds": _int_env("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_CLEANUP_ROUNDS", 12, minimum=0, maximum=12),
        "unsafe_cluster_probe_share": _float_env("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_PROBE_SHARE", 0.25, minimum=0.0, maximum=1.0),
        "final_risky_window_cleanup_rounds": _int_env("DRAFTPROOF_REWRITE_V5_FINAL_RISKY_WINDOW_CLEANUP_ROUNDS", 2, minimum=0, maximum=12),
        "cleanup_variant_count": _int_env("DRAFTPROOF_REWRITE_V5_CLEANUP_VARIANTS", 5, minimum=1, maximum=5),
        "required_ai_drop": _float_env("DRAFTPROOF_REWRITE_V5_REQUIRED_AI_DROP", 5.0, minimum=0.0, maximum=100.0),
        "runtime_base_seconds": _int_env("DRAFTPROOF_REWRITE_V5_RUNTIME_BASE_SECONDS", 120, minimum=30, maximum=1200),
        "runtime_seconds_per_100_words": _float_env("DRAFTPROOF_REWRITE_V5_RUNTIME_SECONDS_PER_100_WORDS", 25.0, minimum=0.0, maximum=300.0),
        "runtime_min_seconds": _int_env("DRAFTPROOF_REWRITE_V5_RUNTIME_MIN_SECONDS", 180, minimum=60, maximum=1800),
        "runtime_max_seconds": _int_env("DRAFTPROOF_REWRITE_V5_RUNTIME_MAX_SECONDS", 720, minimum=90, maximum=7200),
        "runtime_soft_limit_buffer_seconds": _int_env("DRAFTPROOF_REWRITE_V5_RUNTIME_SOFT_LIMIT_BUFFER_SECONDS", 120, minimum=30, maximum=1800),
    }


def _v5_extra_body() -> dict[str, Any] | None:
    extra: dict[str, Any] = {}
    effort = os.environ.get("DRAFTPROOF_REWRITE_V5_REASONING_EFFORT", "none").strip() or "none"
    if _bool_env("DRAFTPROOF_REWRITE_V5_DISABLE_REASONING", True):
        extra["reasoning"] = {"effort": effort, "exclude": True}
        extra["include_reasoning"] = False
    elif _bool_env("DRAFTPROOF_REWRITE_V5_EXCLUDE_REASONING", True):
        extra["reasoning"] = {"exclude": True}
        extra["include_reasoning"] = False
    return extra or None


def run_rewrite_pipeline_v5(
    *,
    detect_json: dict[str, Any],
    output_dir: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started = time.time()

    def progress(percent: int, message: str) -> None:
        if progress_callback:
            progress_callback(percent, message)

    progress(62, "Starting V5 cluster rewrite")
    original_text = _extract_original_text(detect_json)
    config = _production_config()
    runtime_budget_seconds = _v5_runtime_budget_seconds(original_text, config)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_version = "rewrite_v5_residual_cluster_comb"

    def emit_checkpoint(checkpoint: dict[str, Any]) -> None:
        if checkpoint_callback is None:
            return
        checkpoint_text = str(checkpoint.get("rewritten_document") or "")
        if not checkpoint_text.strip() or checkpoint_text.strip() == original_text.strip():
            return
        checkpoint_scores = checkpoint.get("scores") if isinstance(checkpoint.get("scores"), dict) else {}
        baseline_scores = checkpoint.get("baseline_scores") if isinstance(checkpoint.get("baseline_scores"), dict) else {}
        checkpoint_summary = {
            "rewrite_pipeline_version": pipeline_version,
            "rewrite_engine_mode": "v5_residual_cluster_comb_production",
            "outcome": "rewrite_candidate_generated_needs_external_review",
            "public_status": "rewrite_candidate_generated_needs_external_review",
            "partial_rewrite_preserved": True,
            "partial_rewrite_preservation_reason": "accepted_checkpoint_saved_before_pipeline_completion",
            "checkpoint_recovery_available": True,
            "checkpoint": _compact_v5_checkpoint(checkpoint),
            "v5_scores": {
                "original": baseline_scores,
                "final": checkpoint_scores,
                "deltas": _score_deltas(baseline_scores, checkpoint_scores),
            },
            "candidate_generation_status": {
                "accepted_count": int(checkpoint.get("sequence") or 1),
                "reason": "accepted_checkpoint",
            },
            "selected_candidate": checkpoint.get("accepted"),
            "final_text": checkpoint_text,
            "no_text_change": False,
        }
        checkpoint_callback({
            "status": "rewrite_candidate_generated_needs_external_review",
            "checkpoint_schema_version": "rewrite_v5_uploaded_checkpoint.v1",
            "original_text": original_text,
            "final_text": checkpoint_text,
            "summary": checkpoint_summary,
            "checkpoint": checkpoint,
        })

    progress(66, "Running V5 residual cluster comb")
    payload = run_v5_residual_cluster_comb_experiment(
        input_text=original_text,
        output_dir=out_dir,
        max_rounds=int(config["max_rounds"]),
        variant_count=int(config["variant_count"]),
        retune_variant_count=int(config["retune_variant_count"]),
        api_key=api_key,
        model=model,
        base_url=base_url,
        extra_body=_v5_extra_body(),
        risky_window_cleanup_rounds=int(config["risky_window_cleanup_rounds"]),
        unsafe_cluster_cleanup_rounds=int(config["unsafe_cluster_cleanup_rounds"]),
        cleanup_variant_count=int(config["cleanup_variant_count"]),
        final_risky_window_cleanup_rounds=int(config["final_risky_window_cleanup_rounds"]),
        progress_callback=progress,
        accepted_checkpoint_callback=emit_checkpoint,
        max_seconds=runtime_budget_seconds,
    )

    final_text = str(payload.get("rewritten_document") or original_text)
    original_report = detect_json
    final_report = _scan_report(final_text) if final_text.strip() != original_text.strip() else original_report
    final_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=final_text,
        original_report=original_report,
        candidate_report=final_report,
    ).to_dict()
    rounds = payload.get("rounds") if isinstance(payload.get("rounds"), list) else []
    risky_window_cleanup_rounds = (
        payload.get("risky_window_cleanup_rounds")
        if isinstance(payload.get("risky_window_cleanup_rounds"), list)
        else []
    )
    unsafe_cluster_cleanup_rounds = (
        payload.get("unsafe_cluster_cleanup_rounds")
        if isinstance(payload.get("unsafe_cluster_cleanup_rounds"), list)
        else []
    )
    final_risky_window_cleanup_rounds = (
        payload.get("final_risky_window_cleanup_rounds")
        if isinstance(payload.get("final_risky_window_cleanup_rounds"), list)
        else []
    )
    all_rounds = rounds + risky_window_cleanup_rounds + unsafe_cluster_cleanup_rounds + final_risky_window_cleanup_rounds
    accepted = [
        row.get("accepted")
        for row in all_rounds
        if isinstance(row, dict) and isinstance(row.get("accepted"), dict)
    ]
    global_best_fallback = payload.get("global_best_fallback") if isinstance(payload.get("global_best_fallback"), dict) else {}
    if global_best_fallback.get("applied") and isinstance(global_best_fallback.get("selected"), dict):
        accepted.append(global_best_fallback["selected"])
    no_text_change = final_text.strip() == original_text.strip()
    if no_text_change:
        public_status = RewriteGoalStatus.ORIGINAL_PRESERVED.value
    elif final_goal.get("goal_met"):
        public_status = RewriteGoalStatus.AI_MITIGATED.value
    elif accepted:
        public_status = "rewrite_candidate_generated_needs_external_review"
    else:
        public_status = "no_safe_rewrite_applied"
    partial_rewrite_preserved = (
        public_status == "rewrite_candidate_generated_needs_external_review"
        and not no_text_change
        and bool(accepted)
        and not bool(final_goal.get("goal_met"))
    )

    elapsed = time.time() - started
    original_scores = payload.get("baseline_scores") if isinstance(payload.get("baseline_scores"), dict) else {}
    final_scores = payload.get("final_scores") if isinstance(payload.get("final_scores"), dict) else {}
    deltas = _score_deltas(original_scores, final_scores)
    detect_scores = _detect_scores(original_report, final_report, original_scores, final_scores)
    original_scan_compact = _compact_scan_for_rewrite_report(original_report)
    final_scan_compact = _compact_scan_for_rewrite_report(final_report)
    summary = {
        "rewrite_pipeline_version": pipeline_version,
        "rewrite_engine_mode": "v5_residual_cluster_comb_production",
        "outcome": public_status,
        "public_status": public_status,
        "partial_rewrite_preserved": partial_rewrite_preserved,
        "partial_rewrite_preservation_reason": (
            "safe_progress_kept_despite_strict_goal_miss"
            if partial_rewrite_preserved
            else ""
        ),
        "public_candidate_warning": (
            "best_candidate_requires_external_review"
            if public_status == "rewrite_candidate_generated_needs_external_review"
            else ""
        ),
        "best_candidate_external_review_required": public_status == "rewrite_candidate_generated_needs_external_review",
        "rewrite_goal_status": {
            **final_goal,
            "status": public_status if public_status != "no_safe_rewrite_applied" else final_goal.get("status"),
            "goal_met": bool(final_goal.get("goal_met")),
            "reason": final_goal.get("reason") or public_status,
        },
        "strict_goal_status": final_goal.get("status"),
        "reference_ai": original_scores.get("ai"),
        "required_ai_drop": config["required_ai_drop"],
        "target_ai_score": (
            float(original_scores.get("ai")) - float(config["required_ai_drop"])
            if isinstance(original_scores.get("ai"), (int, float))
            else None
        ),
        "rewrite_effective_config": {
            **config,
            "model": model,
            "base_url_configured": bool(base_url),
            "runtime_budget_seconds": runtime_budget_seconds,
        },
        "v5_scores": {
            "original": original_scores,
            "final": final_scores,
            "deltas": deltas,
        },
        "detect_scores": detect_scores,
        "original_risk": detect_scores.get("original_ai"),
        "final_risk": detect_scores.get("rewritten_ai"),
        "converged": bool(final_goal.get("goal_met")),
        "convergence_reason": public_status,
        "candidate_generation_status": {
            "generated_count": _generated_candidate_count(all_rounds),
            "accepted_count": len(accepted),
            "reason": pipeline_version,
        },
        "candidate_trace": _compact_v5_candidate_trace(accepted),
        "candidate_loop_trace": _compact_v5_rounds(all_rounds),
        "selected_candidate": _compact_v5_candidate_trace([accepted[-1]])[0] if accepted else None,
        "rewrite_layers": {"v5_residual_cluster_comb": _compact_v5_payload(payload)},
        "stage_timings": [{
            "stage": pipeline_version,
            "seconds": round(elapsed, 3),
            "selected": bool(accepted),
            "stop_reason": public_status,
        }],
        "detect_scan_original_saved": original_scan_compact,
        "detect_scan_original": original_scan_compact,
        "detect_scan_rewritten": final_scan_compact,
        "final_text": final_text,
        "no_text_change": no_text_change,
        "no_text_change_reason": "v5_no_safe_candidate" if no_text_change else "",
    }
    sentence_comparison = _sentence_comparison(original_text, final_text)
    result_obj = SimpleNamespace(
        summary=summary,
        sentence_comparison=sentence_comparison,
        rewrite_plan=None,
        mp_result=SimpleNamespace(
            original_text=original_text,
            final_text=final_text,
            converged=bool(final_goal.get("goal_met")),
            convergence_reason=public_status,
            passes=[],
        ),
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"draftproof_rewrite_v5_{ts}.md"
    pdf_path = out_dir / f"draftproof_rewrite_v5_{ts}.pdf"
    json_path = out_dir / f"draftproof_rewrite_v5_{ts}.json"
    md_text = render_rewrite_report(summary=summary, sentence_comparison=sentence_comparison, ai_findings=[], verbose=False)
    md_path.write_text(md_text, encoding="utf-8")
    render_pdf(md_text, str(pdf_path))
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    progress(88, "V5 cluster rewrite complete")
    return {
        "status": public_status,
        "md_path": str(md_path),
        "pdf_path": str(pdf_path),
        "json_path": str(json_path),
        "result": result_obj,
        "elapsed": elapsed,
    }


def _score_deltas(original_scores: dict[str, Any], final_scores: dict[str, Any]) -> dict[str, float]:
    keys = (
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
    deltas: dict[str, float] = {}
    for key in keys:
        if key in original_scores and key in final_scores:
            try:
                deltas[f"{key}_delta"] = round(float(original_scores.get(key)) - float(final_scores.get(key)), 3)
            except (TypeError, ValueError):
                continue
    return deltas


def _v5_runtime_budget_seconds(original_text: str, config: dict[str, Any]) -> int:
    words = max(1, word_count(str(original_text or "")))
    estimated = float(config.get("runtime_base_seconds") or 0) + (
        (words / 100.0) * float(config.get("runtime_seconds_per_100_words") or 0.0)
    )
    soft_limit = _int_env("REWRITE_SOFT_TIME_LIMIT_SECONDS", 900, minimum=180, maximum=7200)
    soft_cap = max(
        int(config.get("runtime_min_seconds") or 60),
        soft_limit - int(config.get("runtime_soft_limit_buffer_seconds") or 120),
    )
    return max(
        int(config.get("runtime_min_seconds") or 60),
        min(
            int(config.get("runtime_max_seconds") or 720),
            soft_cap,
            int(round(estimated)),
        ),
    )


def _generated_candidate_count(rounds: list[Any]) -> int:
    count = 0
    for row in rounds:
        if isinstance(row, dict) and isinstance(row.get("candidates"), list):
            count += len(row.get("candidates") or [])
    return count


def _compact_v5_candidate_trace(rows: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
        local = row.get("local_scores") if isinstance(row.get("local_scores"), dict) else {}
        incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
        compact.append({
            "section_id": row.get("section_id"),
            "variant_id": row.get("variant_id"),
            "label": row.get("label"),
            "word_count": row.get("word_count"),
            "ai_delta": scores.get("ai_delta") or incremental.get("ai_delta"),
            "topk_delta": scores.get("topk_delta") or incremental.get("topk_delta"),
            "external_delta": scores.get("external_delta") or incremental.get("external_delta"),
            "rank_delta": scores.get("rank_delta") or incremental.get("rank_delta"),
            "unsafe_cluster_count_delta": scores.get("unsafe_cluster_count_delta") or incremental.get("unsafe_cluster_count_delta"),
            "local_unsafe_cluster_count": local.get("unsafe_cluster_count"),
            "local_topk_delta": local.get("topk_delta"),
            "incremental": incremental or None,
            "text": _truncate_text(row.get("text"), 900),
            "scores_after": scores,
        })
    return compact


def _compact_v5_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    return {
        "schema_version": checkpoint.get("schema_version"),
        "stage": checkpoint.get("stage"),
        "sequence": checkpoint.get("sequence"),
        "phase": checkpoint.get("phase"),
        "round": checkpoint.get("round"),
        "reason": checkpoint.get("reason"),
        "created_at_epoch": checkpoint.get("created_at_epoch"),
        "scores": checkpoint.get("scores") if isinstance(checkpoint.get("scores"), dict) else {},
        "goal": checkpoint.get("goal") if isinstance(checkpoint.get("goal"), dict) else {},
        "accepted": checkpoint.get("accepted") if isinstance(checkpoint.get("accepted"), dict) else {},
        "rewritten_word_count": word_count(str(checkpoint.get("rewritten_document") or "")),
    }


def _compact_v5_generator_diagnostics(diagnostics: Any) -> dict[str, Any] | None:
    if not isinstance(diagnostics, dict):
        return None
    route_plan = diagnostics.get("route_plan") if isinstance(diagnostics.get("route_plan"), dict) else {}
    llm_generation = diagnostics.get("llm_generation") if isinstance(diagnostics.get("llm_generation"), dict) else diagnostics
    compact: dict[str, Any] = {
        "route_plan": {
            "status": route_plan.get("status"),
            "reason": route_plan.get("reason"),
            "content_profile": route_plan.get("content_profile"),
            "cluster_role": route_plan.get("cluster_role"),
            "dominant_failure_pattern": route_plan.get("dominant_failure_pattern"),
            "route_strategy": route_plan.get("route_strategy"),
            "source_block_plan_count": route_plan.get("source_block_plan_count"),
            "target_sentence_job_count": route_plan.get("target_sentence_job_count"),
            "length_target": route_plan.get("length_target"),
            "finish_reason": route_plan.get("finish_reason"),
            "native_finish_reason": route_plan.get("native_finish_reason"),
        } if route_plan else None,
        "llm_generation": {
            "status": llm_generation.get("status"),
            "reason": llm_generation.get("reason"),
            "valid_variant_count": llm_generation.get("valid_variant_count"),
            "variant_count": llm_generation.get("variant_count"),
            "finish_reason": llm_generation.get("finish_reason"),
            "native_finish_reason": llm_generation.get("native_finish_reason"),
            "structured_output_mode": llm_generation.get("structured_output_mode"),
        } if isinstance(llm_generation, dict) else None,
    }
    return compact


def _compact_v5_rounds(rounds: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rounds:
        if not isinstance(row, dict):
            continue
        compact.append({
            "round": row.get("round"),
            "phase": row.get("phase"),
            "status": row.get("status"),
            "reason": row.get("reason"),
            "current_scores": row.get("current_scores") if isinstance(row.get("current_scores"), dict) else None,
            "runtime_budget": row.get("runtime_budget") if isinstance(row.get("runtime_budget"), dict) else None,
            "density_gate": _compact_density_gate(row.get("density_gate"))
            if isinstance(row.get("density_gate"), dict)
            else None,
            "generator_diagnostics": _compact_v5_generator_diagnostics(row.get("generator_diagnostics")),
            "accepted": _compact_v5_candidate_trace([row.get("accepted")])[0]
            if isinstance(row.get("accepted"), dict)
            else None,
            "selected": _compact_v5_candidate_trace([row.get("selected")])[0]
            if isinstance(row.get("selected"), dict)
            else None,
            "candidate_count": len(row.get("candidates") or []) if isinstance(row.get("candidates"), list) else 0,
        })
    return compact


def _compact_v5_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rounds = payload.get("rounds") if isinstance(payload.get("rounds"), list) else []
    risky_window_cleanup_rounds = (
        payload.get("risky_window_cleanup_rounds")
        if isinstance(payload.get("risky_window_cleanup_rounds"), list)
        else []
    )
    unsafe_cluster_cleanup_rounds = (
        payload.get("unsafe_cluster_cleanup_rounds")
        if isinstance(payload.get("unsafe_cluster_cleanup_rounds"), list)
        else []
    )
    final_risky_window_cleanup_rounds = (
        payload.get("final_risky_window_cleanup_rounds")
        if isinstance(payload.get("final_risky_window_cleanup_rounds"), list)
        else []
    )
    all_rounds = rounds + risky_window_cleanup_rounds + unsafe_cluster_cleanup_rounds + final_risky_window_cleanup_rounds
    return {
        "stage": payload.get("stage"),
        "baseline_scores": payload.get("baseline_scores"),
        "final_scores": payload.get("final_scores"),
        "eligible_span_density_gate": payload.get("eligible_span_density_gate"),
        "runtime_budget": payload.get("runtime_budget") if isinstance(payload.get("runtime_budget"), dict) else None,
        "phase_order": payload.get("phase_order") if isinstance(payload.get("phase_order"), dict) else None,
        "accepted_checkpoints": payload.get("accepted_checkpoints") if isinstance(payload.get("accepted_checkpoints"), list) else [],
        "goal": payload.get("goal"),
        "accepted_rounds": sum(1 for row in all_rounds if isinstance(row, dict) and row.get("accepted")),
        "rounds": _compact_v5_rounds(rounds),
        "risky_window_cleanup_rounds": _compact_v5_rounds(risky_window_cleanup_rounds),
        "unsafe_cluster_cleanup_rounds": _compact_v5_rounds(unsafe_cluster_cleanup_rounds),
        "final_risky_window_cleanup_rounds": _compact_v5_rounds(final_risky_window_cleanup_rounds),
        "global_best_fallback": payload.get("global_best_fallback"),
    }
