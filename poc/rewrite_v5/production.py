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
from rewrite_v3.pipeline import _scan_report
from rewrite_v4.production import (
    _compact_scan_for_rewrite_report,
    _detect_scores,
    _truncate_text,
)

from .residual_comb import run_v5_residual_cluster_comb_experiment


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
        "final_risky_window_cleanup_rounds": _int_env("DRAFTPROOF_REWRITE_V5_FINAL_RISKY_WINDOW_CLEANUP_ROUNDS", 2, minimum=0, maximum=12),
        "cleanup_variant_count": _int_env("DRAFTPROOF_REWRITE_V5_CLEANUP_VARIANTS", 5, minimum=1, maximum=5),
        "required_ai_drop": _float_env("DRAFTPROOF_REWRITE_V5_REQUIRED_AI_DROP", 5.0, minimum=0.0, maximum=100.0),
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
) -> dict[str, Any]:
    started = time.time()

    def progress(percent: int, message: str) -> None:
        if progress_callback:
            progress_callback(percent, message)

    progress(62, "Starting V5 cluster rewrite")
    original_text = _extract_original_text(detect_json)
    config = _production_config()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    elapsed = time.time() - started
    original_scores = payload.get("baseline_scores") if isinstance(payload.get("baseline_scores"), dict) else {}
    final_scores = payload.get("final_scores") if isinstance(payload.get("final_scores"), dict) else {}
    deltas = _score_deltas(original_scores, final_scores)
    detect_scores = _detect_scores(original_report, final_report, original_scores, final_scores)
    original_scan_compact = _compact_scan_for_rewrite_report(original_report)
    final_scan_compact = _compact_scan_for_rewrite_report(final_report)
    pipeline_version = "rewrite_v5_residual_cluster_comb"
    summary = {
        "rewrite_pipeline_version": pipeline_version,
        "rewrite_engine_mode": "v5_residual_cluster_comb_production",
        "outcome": public_status,
        "public_status": public_status,
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
            "text": _truncate_text(row.get("text"), 900),
            "scores_after": scores,
        })
    return compact


def _compact_v5_rounds(rounds: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rounds:
        if not isinstance(row, dict):
            continue
        compact.append({
            "round": row.get("round"),
            "status": row.get("status"),
            "reason": row.get("reason"),
            "current_scores": row.get("current_scores") if isinstance(row.get("current_scores"), dict) else None,
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
        "goal": payload.get("goal"),
        "accepted_rounds": sum(1 for row in all_rounds if isinstance(row, dict) and row.get("accepted")),
        "rounds": _compact_v5_rounds(rounds),
        "risky_window_cleanup_rounds": _compact_v5_rounds(risky_window_cleanup_rounds),
        "unsafe_cluster_cleanup_rounds": _compact_v5_rounds(unsafe_cluster_cleanup_rounds),
        "final_risky_window_cleanup_rounds": _compact_v5_rounds(final_risky_window_cleanup_rounds),
        "global_best_fallback": payload.get("global_best_fallback"),
    }
