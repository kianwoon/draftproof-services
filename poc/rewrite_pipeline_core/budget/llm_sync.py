"""LLM call accounting helpers for rewrite summaries."""

from __future__ import annotations

from rewrite_controller import RewriteRunBudget


def _record_rewrite_llm_calls(summary: dict, phase_key: str, calls: int | float | str | None) -> int:
    """Record phase and total LLM calls in one place.

    Several rewrite phases maintain their own budgets. The report must reflect
    actual phase usage so production logs and JSON summaries do not disagree.
    """
    if not isinstance(summary, dict):
        return 0
    try:
        call_count = max(0, int(calls or 0))
    except (TypeError, ValueError):
        call_count = 0
    summary[str(phase_key)] = call_count
    if call_count:
        try:
            prior = int(summary.get("llm_calls_used") or 0)
        except (TypeError, ValueError):
            prior = 0
        summary["llm_calls_used"] = prior + call_count
    else:
        summary.setdefault("llm_calls_used", 0)
    return call_count


def _sync_rewrite_llm_call_totals(summary: dict, budget: RewriteRunBudget | None = None) -> dict:
    """Reconcile legacy phase counters with the final global controller ledger."""
    if not isinstance(summary, dict):
        return {}
    global_llm_calls = 0
    global_stage_names: set[str] = set()
    if isinstance(budget, RewriteRunBudget):
        global_llm_calls = max(0, int(budget.llm_calls_used or 0))
        global_stage_names = {
            str(stage.get("stage") or "")
            for stage in (budget.stages or [])
            if isinstance(stage, dict)
        }
    non_global_phase_calls: dict[str, int] = {}
    if "authenticity_mitigation" not in global_stage_names:
        try:
            authenticity_calls = int(summary.get("authenticity_llm_calls_used") or 0)
        except (TypeError, ValueError):
            authenticity_calls = 0
        if authenticity_calls > 0:
            non_global_phase_calls["authenticity_mitigation"] = authenticity_calls

    total_calls = global_llm_calls + sum(non_global_phase_calls.values())
    try:
        legacy_total = max(0, int(summary.get("llm_calls_used") or 0))
    except (TypeError, ValueError):
        legacy_total = 0
    if total_calls <= 0:
        total_calls = legacy_total

    summary["global_llm_calls_used"] = global_llm_calls
    segment_summary = summary.get("segment_window_density_controller")
    if isinstance(segment_summary, dict):
        try:
            summary["segment_window_llm_calls_used"] = max(0, int(segment_summary.get("llm_calls") or 0))
        except (TypeError, ValueError):
            summary["segment_window_llm_calls_used"] = 0
    segment_followup_summary = summary.get("segment_window_density_controller_followup")
    if isinstance(segment_followup_summary, dict):
        try:
            summary["segment_window_followup_llm_calls_used"] = max(0, int(segment_followup_summary.get("llm_calls") or 0))
        except (TypeError, ValueError):
            summary["segment_window_followup_llm_calls_used"] = 0
    remaining_cluster_summary = summary.get("remaining_cluster_density_controller")
    if isinstance(remaining_cluster_summary, dict):
        try:
            summary["remaining_cluster_llm_calls_used"] = max(0, int(remaining_cluster_summary.get("llm_calls") or 0))
        except (TypeError, ValueError):
            summary["remaining_cluster_llm_calls_used"] = 0
    window_coverage_summary = summary.get("window_coverage_density_optimizer")
    if isinstance(window_coverage_summary, dict):
        try:
            summary["window_coverage_llm_calls_used"] = max(0, int(window_coverage_summary.get("llm_calls") or 0))
        except (TypeError, ValueError):
            summary["window_coverage_llm_calls_used"] = 0
    compiler_summary = summary.get("rewrite_compiler")
    if isinstance(compiler_summary, dict):
        try:
            summary["rewrite_compiler_llm_calls_used"] = max(0, int(compiler_summary.get("llm_calls_used") or 0))
        except (TypeError, ValueError):
            summary["rewrite_compiler_llm_calls_used"] = 0
    summary["llm_calls_used"] = total_calls
    summary["llm_calls_breakdown"] = {
        "source": "global_controller_ledger_plus_non_global_phases",
        "global_controller_llm_calls": global_llm_calls,
        "non_global_phase_llm_calls": non_global_phase_calls,
        "legacy_incremental_total_before_sync": legacy_total,
        "total": total_calls,
    }
    return summary["llm_calls_breakdown"]
