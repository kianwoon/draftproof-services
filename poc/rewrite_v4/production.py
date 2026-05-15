"""Production adapter for rewrite V4.

The experiment module owns the V4 control flow. This adapter keeps the worker
contract stable: it returns the same report paths and result namespace shape as
V2/V3 while routing configuration through environment variables.
"""

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

from .experiment import run_v4_iterative_rewrite


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
        "include_llm_normalizer": _bool_env("DRAFTPROOF_REWRITE_V4_INCLUDE_LLM_NORMALIZER", False),
        "variant_count": _int_env("DRAFTPROOF_REWRITE_V4_VARIANTS", 2, minimum=1, maximum=4),
        "max_rounds": _int_env("DRAFTPROOF_REWRITE_V4_MAX_ROUNDS", 2, minimum=1, maximum=4),
        "groups_per_round": _int_env("DRAFTPROOF_REWRITE_V4_GROUPS_PER_ROUND", 3, minimum=1, maximum=8),
        "stop_after_accepted": _int_env("DRAFTPROOF_REWRITE_V4_STOP_AFTER_ACCEPTED", 2, minimum=1, maximum=8),
        "strong_ai_delta": _float_env("DRAFTPROOF_REWRITE_V4_STRONG_AI_DELTA", 10.0, minimum=0.0, maximum=100.0),
        "required_ai_drop": _float_env("DRAFTPROOF_REWRITE_V4_REQUIRED_AI_DROP", 5.0, minimum=0.0, maximum=100.0),
    }


def _v4_extra_body() -> dict[str, Any] | None:
    extra: dict[str, Any] = {}
    effort = os.environ.get("DRAFTPROOF_REWRITE_V4_REASONING_EFFORT", "none").strip() or "none"
    if _bool_env("DRAFTPROOF_REWRITE_V4_DISABLE_REASONING", True):
        extra["reasoning"] = {"effort": effort, "exclude": True}
        extra["include_reasoning"] = False
    elif _bool_env("DRAFTPROOF_REWRITE_V4_EXCLUDE_REASONING", True):
        extra["reasoning"] = {"exclude": True}
        extra["include_reasoning"] = False
    return extra or None


def run_rewrite_pipeline_v4(
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

    progress(62, "Starting normalized rewrite V4")
    original_text = _extract_original_text(detect_json)
    config = _production_config()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress(66, "Running V4 repair candidates")
    payload = run_v4_iterative_rewrite(
        input_text=original_text,
        output_dir=out_dir,
        include_llm_normalizer=bool(config["include_llm_normalizer"]),
        variant_count=int(config["variant_count"]),
        max_rounds=int(config["max_rounds"]),
        groups_per_round=int(config["groups_per_round"]),
        stop_after_accepted=int(config["stop_after_accepted"]),
        strong_ai_delta=float(config["strong_ai_delta"]),
        api_key=api_key,
        model=model,
        base_url=base_url,
        extra_body=_v4_extra_body(),
    )
    final_text = str(payload.get("rewritten_document") or original_text)
    original_report = detect_json
    final_report = _load_final_scan(out_dir) or original_report
    final_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=final_text,
        original_report=original_report,
        candidate_report=final_report,
    ).to_dict()
    accepted = payload.get("accepted") if isinstance(payload.get("accepted"), list) else []
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
    v4_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    original_scores = v4_summary.get("original_scores") if isinstance(v4_summary.get("original_scores"), dict) else {}
    final_scores = v4_summary.get("final_scores") if isinstance(v4_summary.get("final_scores"), dict) else {}
    deltas = v4_summary.get("deltas") if isinstance(v4_summary.get("deltas"), dict) else {}
    detect_scores = _detect_scores(original_report, final_report, original_scores, final_scores)
    summary = {
        "rewrite_pipeline_version": "rewrite_v4_normalized_repair",
        "rewrite_engine_mode": "v4_fast_production",
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
        "v4_scores": {
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
            "generated_count": _generated_candidate_count(payload),
            "accepted_count": len(accepted),
            "reason": "v4_normalized_repair",
        },
        "candidate_trace": accepted,
        "candidate_loop_trace": payload.get("rounds") if isinstance(payload.get("rounds"), list) else [],
        "selected_candidate": accepted[-1] if accepted else None,
        "stage_timings": [{
            "stage": "rewrite_v4_normalized_repair",
            "seconds": round(elapsed, 3),
            "selected": bool(accepted),
            "stop_reason": public_status,
        }],
        "detect_scan_original": original_report,
        "detect_scan_rewritten": final_report,
        "final_text": final_text,
        "no_text_change": no_text_change,
        "no_text_change_reason": "v4_no_safe_candidate" if no_text_change else "",
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
    md_path = out_dir / f"draftproof_rewrite_v4_{ts}.md"
    pdf_path = out_dir / f"draftproof_rewrite_v4_{ts}.pdf"
    json_path = out_dir / f"draftproof_rewrite_v4_{ts}.json"
    md_text = render_rewrite_report(summary=summary, sentence_comparison=sentence_comparison, ai_findings=[], verbose=False)
    md_path.write_text(md_text, encoding="utf-8")
    render_pdf(md_text, str(pdf_path))
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    progress(88, "Normalized rewrite V4 complete")
    return {
        "status": public_status,
        "md_path": str(md_path),
        "pdf_path": str(pdf_path),
        "json_path": str(json_path),
        "result": result_obj,
        "elapsed": elapsed,
    }


def _load_final_scan(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "v4_final_scan.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _generated_candidate_count(payload: dict[str, Any]) -> int:
    count = 0
    for round_row in payload.get("rounds") or []:
        if not isinstance(round_row, dict):
            continue
        for block in round_row.get("candidates") or []:
            if not isinstance(block, dict):
                continue
            for result in block.get("results") or []:
                if isinstance(result, dict):
                    count += 1
    return count


def _detect_scores(
    original_report: dict[str, Any],
    final_report: dict[str, Any],
    original_scores: dict[str, Any],
    final_scores: dict[str, Any],
) -> dict[str, Any]:
    original_ai = _ai_score(original_report, original_scores.get("ai"))
    rewritten_ai = _ai_score(final_report, final_scores.get("ai"))
    original_contribution = _contribution_scores(original_report)
    rewritten_contribution = _contribution_scores(final_report)
    return {
        "original_ai": original_ai,
        "rewritten_ai": rewritten_ai,
        "original_ai_authorship": original_ai,
        "rewritten_ai_authorship": rewritten_ai,
        "original_human_contribution": original_contribution.get("human_contribution"),
        "rewritten_human_contribution": rewritten_contribution.get("human_contribution"),
        "original_ai_transformation": original_contribution.get("ai_transformation"),
        "rewritten_ai_transformation": rewritten_contribution.get("ai_transformation"),
        "original_grounding_quality_risk": original_contribution.get("grounding_quality_risk"),
        "rewritten_grounding_quality_risk": rewritten_contribution.get("grounding_quality_risk"),
        "human_shift_score": _score_delta(original_ai, rewritten_ai),
        "original_findings": _count_findings(original_report.get("findings")),
        "rewritten_findings": _count_findings(final_report.get("findings")),
    }


def _ai_score(report: dict[str, Any], fallback: Any = None) -> float | None:
    badge = report.get("ai_risk_badge") if isinstance(report, dict) else {}
    value = report.get("ai_score") if isinstance(report, dict) else None
    if not isinstance(value, (int, float)) and isinstance(badge, dict):
        value = badge.get("ai_likelihood_score")
    if not isinstance(value, (int, float)):
        value = fallback
    return round(float(value), 3) if isinstance(value, (int, float)) else None


def _contribution_scores(report: dict[str, Any]) -> dict[str, float | None]:
    intelligence = report.get("scan_intelligence") if isinstance(report, dict) else {}
    transformation = intelligence.get("transformation") if isinstance(intelligence, dict) else {}
    contribution = transformation.get("contribution") if isinstance(transformation, dict) else {}
    layers = report.get("integrity_layers") if isinstance(report, dict) else {}
    layer_rows = layers.get("layers") if isinstance(layers, dict) else {}
    human_layer = layer_rows.get("human_contribution_signal") if isinstance(layer_rows, dict) else {}
    ai_layer = layer_rows.get("ai_transformation_risk") if isinstance(layer_rows, dict) else {}
    badge = report.get("ai_risk_badge") if isinstance(report, dict) else {}
    writing_components = badge.get("writing_components") if isinstance(badge, dict) else {}

    human = _first_number(
        contribution.get("human_contribution_ratio") if isinstance(contribution, dict) else None,
        contribution.get("human_contribution") if isinstance(contribution, dict) else None,
        contribution.get("human_ratio") if isinstance(contribution, dict) else None,
        human_layer.get("score") if isinstance(human_layer, dict) else None,
    )
    ai = _first_number(
        contribution.get("ai_transformation_ratio") if isinstance(contribution, dict) else None,
        contribution.get("ai_transformation") if isinstance(contribution, dict) else None,
        contribution.get("transformation_ratio") if isinstance(contribution, dict) else None,
        ai_layer.get("score") if isinstance(ai_layer, dict) else None,
    )
    grounding = _first_number(
        writing_components.get("grounding_risk") if isinstance(writing_components, dict) else None,
        writing_components.get("source_grounding_risk") if isinstance(writing_components, dict) else None,
    )
    return {
        "human_contribution": human,
        "ai_transformation": ai,
        "grounding_quality_risk": grounding,
    }


def _first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, (int, float)):
            return round(float(value), 3)
    return None


def _score_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return round(float(before) - float(after), 3)


def _count_findings(findings: Any) -> int | None:
    if not isinstance(findings, dict):
        return None
    return sum(len(rows) for rows in findings.values() if isinstance(rows, list))
