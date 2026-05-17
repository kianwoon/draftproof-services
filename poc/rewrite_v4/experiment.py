"""End-to-end V4 experiment runner.

This module is intentionally isolated from production orchestration. It proves
the V4 control flow with fresh same-run baseline scans:

scanner -> normalizer -> repair brief -> generator -> validator/rescan -> select.
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

from llm.gateway import LLMConfig, LLMGateway
from rewrite_v2.goal_contract import evaluate_rewrite_goal
from rewrite_v3.pipeline import _ai_footprint_profile, _badge_ai, _footprint_risk, _scan_report, _topk
from rewrite_v3.scanner_contract import build_scan_contract
from rewrite_v3.scanner_controlled_executor import scanner_controlled_metrics, scanner_controlled_rank
from rewrite_v3.target_executor import apply_target_replacements, group_rewrite_targets

from .budget_lanes import BudgetLane, budget_lanes_for_unit, budget_profile_passes, edit_budget_profile, generate_budget_lane_variants
from .cluster_patch import apply_cluster_variant, build_cluster_repair_units, generate_cluster_variants
from .generator import generate_variants
from .models import CandidateVariant, ClusterRepairUnit, RepairBrief
from .normalizer import deterministic_repair_brief, enrichment_repair_brief, llm_repair_brief, scanner_evidence_for_group, tutor_repair_brief
from .validation import source_grounding_integrity, strategy_compliance_integrity


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


def run_v4_experiment(
    *,
    input_text: str,
    output_dir: str | Path,
    unit_ids: set[str] | None = None,
    include_llm_normalizer: bool = True,
    include_tutor_normalizer: bool = False,
    include_enrichment_normalizer: bool = False,
    variant_count: int = 3,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_report = _scan_report(input_text)
    (out_dir / "fresh_baseline_scan.json").write_text(json.dumps(baseline_report, ensure_ascii=False, indent=2))
    baseline_goal = evaluate_rewrite_goal(
        original_text=input_text,
        candidate_text=input_text,
        original_report=baseline_report,
        candidate_report=baseline_report,
    ).to_dict()
    baseline_metrics = _metrics(input_text=input_text, report=baseline_report, goal=baseline_goal)
    baseline = _score_summary(baseline_report, baseline_metrics)

    contract = build_scan_contract(baseline_report, input_text)
    groups = group_rewrite_targets(
        original_text=input_text,
        rewrite_target_profile=contract.rewrite_target_profile,
        max_groups=20,
    )
    if unit_ids:
        groups = [group for group in groups if str(group.unit_id) in unit_ids]

    gateway = LLMGateway(LLMConfig(api_key=api_key, model=model, base_url=base_url, max_tokens=8000, temperature=0.25, top_p=0.85, timeout=180, extra_body=extra_body))
    experiments: list[dict[str, Any]] = []
    for group in groups:
        briefs: list[RepairBrief] = [deterministic_repair_brief(group)]
        if include_llm_normalizer:
            briefs.append(llm_repair_brief(group, gateway))
        if include_tutor_normalizer:
            briefs.append(tutor_repair_brief(group, gateway))
        if include_enrichment_normalizer:
            briefs.append(enrichment_repair_brief(group, gateway))
        for brief in briefs:
            variants, generator_diagnostics, prompt, completion = generate_variants(
                group=group,
                repair_brief=brief,
                gateway=gateway,
                variant_count=variant_count,
            )
            stem = f"{group.unit_id}_{brief.normalizer}"
            (out_dir / f"{stem}_generator_prompt.json.txt").write_text(prompt)
            (out_dir / f"{stem}_generator_completion.json.txt").write_text(completion)
            result_rows = [
                _score_variant(
                    input_text=input_text,
                    baseline_report=baseline_report,
                    baseline=baseline,
                    group=group,
                    repair_brief=brief,
                    variant=variant,
                    output_dir=out_dir,
                    stem=stem,
                )
                for variant in variants
            ]
            experiments.append({
                "unit_id": group.unit_id,
                "group_id": group.group_id,
                "operation": group.operation,
                "source_text": group.source_text,
                "scanner_evidence": scanner_evidence_for_group(group),
                "repair_brief": brief.to_dict(),
                "generator_diagnostics": generator_diagnostics,
                "results": result_rows,
            })

    summary = _rank_results(experiments)
    payload = {
        "baseline": baseline,
        "selected_unit_ids": sorted(unit_ids or []),
        "experiments": experiments,
        "summary_ranked": summary,
        "best_safe_candidate": _best_safe(summary),
    }
    (out_dir / "v4_experiment_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def run_v4_iterative_rewrite(
    *,
    input_text: str,
    output_dir: str | Path,
    unit_ids: set[str] | None = None,
    include_llm_normalizer: bool = True,
    include_tutor_normalizer: bool = False,
    include_enrichment_normalizer: bool = False,
    variant_count: int = 3,
    max_rounds: int = 3,
    groups_per_round: int | None = None,
    stop_after_accepted: int | None = None,
    strong_ai_delta: float | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a full rewritten document by applying safe V4 candidates one at a time."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    current_text = str(input_text or "")
    original_report = _scan_report(current_text)
    original_goal = evaluate_rewrite_goal(
        original_text=input_text,
        candidate_text=input_text,
        original_report=original_report,
        candidate_report=original_report,
    ).to_dict()
    original_metrics = _metrics(input_text=input_text, report=original_report, goal=original_goal)
    original_baseline = _score_summary(original_report, original_metrics)

    gateway = LLMGateway(LLMConfig(api_key=api_key, model=model, base_url=base_url, max_tokens=8000, temperature=0.25, top_p=0.85, timeout=180, extra_body=extra_body))
    accepted: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    current_report = original_report
    current_goal = original_goal
    current_baseline = original_baseline

    for round_index in range(1, max(1, int(max_rounds or 1)) + 1):
        contract = build_scan_contract(current_report, current_text)
        groups = group_rewrite_targets(
            original_text=current_text,
            rewrite_target_profile=contract.rewrite_target_profile,
            max_groups=20,
        )
        if unit_ids:
            groups = [group for group in groups if str(group.unit_id) in unit_ids]
        groups = [
            group for group in groups
            if str(group.unit_id or group.group_id) not in {str(row.get("unit_id")) for row in accepted}
        ]
        ranked_groups = _rank_groups_for_v4(groups)
        if groups_per_round:
            primary_count = max(1, int(groups_per_round))
            primary_groups = ranked_groups[:primary_count]
            fallback_groups = ranked_groups[primary_count:]
        else:
            primary_groups = ranked_groups
            fallback_groups = []
        round_log: dict[str, Any] = {
            "round": round_index,
            "baseline": current_baseline,
            "target_count": len(ranked_groups),
            "primary_target_count": len(primary_groups),
            "fallback_target_count": len(fallback_groups),
            "groups_per_round": groups_per_round,
            "candidates": [],
        }
        if not ranked_groups:
            round_log["stop_reason"] = "no_target_groups"
            rounds.append(round_log)
            break

        best: dict[str, Any] | None = None
        for batch_label, batch_groups in (
            ("primary", primary_groups),
            ("fallback_after_no_safe_primary_candidate", fallback_groups),
        ):
            if best is not None or not batch_groups:
                continue
            if batch_label != "primary":
                round_log["fallback_reason"] = batch_label
            for group in batch_groups:
                briefs: list[RepairBrief] = [deterministic_repair_brief(group)]
                if include_llm_normalizer:
                    briefs.append(llm_repair_brief(group, gateway))
                if include_tutor_normalizer:
                    briefs.append(tutor_repair_brief(group, gateway))
                if include_enrichment_normalizer:
                    briefs.append(enrichment_repair_brief(group, gateway))
                for brief in briefs:
                    variants, generator_diagnostics, prompt, completion = generate_variants(
                        group=group,
                        repair_brief=brief,
                        gateway=gateway,
                        variant_count=variant_count,
                    )
                    stem = f"round{round_index}_{group.unit_id}_{brief.normalizer}"
                    (out_dir / f"{stem}_generator_prompt.json.txt").write_text(prompt)
                    (out_dir / f"{stem}_generator_completion.json.txt").write_text(completion)
                    result_rows = [
                        _score_variant_against_current(
                            original_text=input_text,
                            current_text=current_text,
                            current_report=current_report,
                            current_baseline=current_baseline,
                            group=group,
                            repair_brief=brief,
                            variant=variant,
                            output_dir=out_dir,
                            stem=stem,
                        )
                        for variant in variants
                    ]
                    candidate_block = {
                        "unit_id": group.unit_id,
                        "group_id": group.group_id,
                        "batch": batch_label,
                        "repair_brief": brief.to_dict(),
                        "generator_diagnostics": generator_diagnostics,
                        "results": result_rows,
                    }
                    round_log["candidates"].append(candidate_block)
                    for row in result_rows:
                        compact = _compact_result_row(row, brief)
                        if not _is_safe_positive(compact):
                            continue
                        if best is None or _candidate_sort_key(compact) > _candidate_sort_key(best):
                            best = {
                                **compact,
                                "candidate_text": row.get("candidate_text"),
                                "candidate_report": row.get("candidate_report"),
                                "candidate_goal": row.get("candidate_goal"),
                            }

        if best is None:
            round_log["stop_reason"] = "no_safe_positive_candidate_after_ranked_targets"
            rounds.append(round_log)
            break

        current_text = str(best.pop("candidate_text"))
        current_report = best.pop("candidate_report")
        current_goal = best.pop("candidate_goal")
        current_metrics = _metrics(input_text=input_text, report=current_report, goal=current_goal)
        current_baseline = _score_summary(current_report, current_metrics)
        accepted.append({
            **best,
            "round": round_index,
            "scores_after": current_baseline,
        })
        round_log["accepted"] = accepted[-1]
        rounds.append(round_log)
        if stop_after_accepted is not None and len(accepted) >= max(1, int(stop_after_accepted)):
            break
        if strong_ai_delta is not None and _num(accepted[-1].get("ai_delta")) >= float(strong_ai_delta):
            break

    final_goal = evaluate_rewrite_goal(
        original_text=input_text,
        candidate_text=current_text,
        original_report=original_report,
        candidate_report=current_report,
    ).to_dict()
    final_metrics = _metrics(input_text=input_text, report=current_report, goal=final_goal)
    final_scores = _score_summary(current_report, final_metrics)
    summary = {
        "original_scores": original_baseline,
        "final_scores": final_scores,
        "deltas": {
            "ai_delta": round(_num(original_baseline.get("ai")) - _num(final_scores.get("ai")), 3),
            "topk_delta": round(_num(original_baseline.get("topk")) - _num(final_scores.get("topk")), 3),
            "external_delta": round(_num(original_baseline.get("external")) - _num(final_scores.get("external")), 3),
            "rank_delta": round(_num(original_baseline.get("rank")) - _num(final_scores.get("rank")), 3),
        },
        "accepted_count": len(accepted),
        "goal": {
            "status": final_goal.get("status"),
            "goal_met": final_goal.get("goal_met"),
            "reason": final_goal.get("reason"),
        },
    }
    payload = {
        "summary": summary,
        "accepted": accepted,
        "rounds": rounds,
        "config": {
            "include_llm_normalizer": include_llm_normalizer,
            "include_tutor_normalizer": include_tutor_normalizer,
            "include_enrichment_normalizer": include_enrichment_normalizer,
            "variant_count": variant_count,
            "max_rounds": max_rounds,
            "groups_per_round": groups_per_round,
            "stop_after_accepted": stop_after_accepted,
            "strong_ai_delta": strong_ai_delta,
        },
        "rewritten_document": current_text,
    }
    (out_dir / "v4_iterative_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (out_dir / "v4_rewritten_document.txt").write_text(current_text)
    (out_dir / "v4_final_scan.json").write_text(json.dumps(current_report, ensure_ascii=False, indent=2))
    return payload


def run_v4_layered_rewrite(
    *,
    input_text: str,
    output_dir: str | Path,
    unit_ids: set[str] | None = None,
    include_llm_normalizer: bool = True,
    include_tutor_normalizer: bool = False,
    include_enrichment_normalizer: bool = False,
    variant_count: int = 3,
    max_rounds: int = 3,
    groups_per_round: int | None = None,
    stop_after_accepted: int | None = None,
    strong_ai_delta: float | None = None,
    layer3_enabled: bool = True,
    layer3_max_rounds: int = 2,
    layer3_max_clusters: int = 4,
    layer3_variant_count: int = 2,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the V4 sequence: local normalized repairs, then Layer 3 budget lanes."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer1_payload = run_v4_iterative_rewrite(
        input_text=input_text,
        output_dir=out_dir,
        unit_ids=unit_ids,
        include_llm_normalizer=include_llm_normalizer,
        include_tutor_normalizer=include_tutor_normalizer,
        include_enrichment_normalizer=include_enrichment_normalizer,
        variant_count=variant_count,
        max_rounds=max_rounds,
        groups_per_round=groups_per_round,
        stop_after_accepted=stop_after_accepted,
        strong_ai_delta=strong_ai_delta,
        api_key=api_key,
        model=model,
        base_url=base_url,
        extra_body=extra_body,
    )
    current_text = str(layer1_payload.get("rewritten_document") or input_text or "")
    layer1_goal = ((layer1_payload.get("summary") or {}).get("goal") or {})
    run_layer3 = bool(layer3_enabled) and not bool(layer1_goal.get("goal_met"))
    layer3_payload: dict[str, Any] | None = None
    if run_layer3:
        layer3_payload = run_v4_budget_lane_experiment(
            input_text=current_text,
            output_dir=out_dir / "layer3_budget_lanes",
            max_rounds=layer3_max_rounds,
            max_clusters=layer3_max_clusters,
            variant_count=layer3_variant_count,
            api_key=api_key,
            model=model,
            base_url=base_url,
            extra_body=extra_body,
        )
        if layer3_payload.get("accepted"):
            current_text = str(layer3_payload.get("rewritten_document") or current_text)

    original_report = _scan_report(str(input_text or ""))
    layer3_scan_path = out_dir / "layer3_budget_lanes" / "v4_budget_lane_final_scan.json"
    layer1_scan_path = out_dir / "v4_final_scan.json"
    final_report = (
        _read_json_file(layer3_scan_path)
        if layer3_payload and layer3_payload.get("accepted")
        else _read_json_file(layer1_scan_path)
    )
    if not isinstance(final_report, dict):
        final_report = _scan_report(current_text)
    final_goal = evaluate_rewrite_goal(
        original_text=str(input_text or ""),
        candidate_text=current_text,
        original_report=original_report,
        candidate_report=final_report,
    ).to_dict()
    original_goal = evaluate_rewrite_goal(
        original_text=str(input_text or ""),
        candidate_text=str(input_text or ""),
        original_report=original_report,
        candidate_report=original_report,
    ).to_dict()
    original_scores = _score_summary(original_report, _metrics(input_text=str(input_text or ""), report=original_report, goal=original_goal))
    original_scores["unsafe_cluster_count"] = _unsafe_cluster_count(original_goal)
    original_scores["unsafe_word_ratio"] = _unsafe_word_ratio(original_goal, original_scores)
    _add_goal_driver_snapshot(original_scores, original_goal)
    final_scores = _score_summary(final_report, _metrics(input_text=str(input_text or ""), report=final_report, goal=final_goal))
    final_scores["unsafe_cluster_count"] = _unsafe_cluster_count(final_goal)
    final_scores["unsafe_word_ratio"] = _unsafe_word_ratio(final_goal, final_scores)
    _add_goal_driver_snapshot(final_scores, final_goal)
    _add_score_deltas(final_scores, original_scores)
    layer1_accepted = [
        {"layer": "normalized_repair", **row}
        for row in (layer1_payload.get("accepted") or [])
        if isinstance(row, dict)
    ]
    layer3_accepted = [
        {"layer": "budget_lane_optimizer", **row}
        for row in ((layer3_payload or {}).get("accepted") or [])
        if isinstance(row, dict)
    ]
    payload = {
        "summary": {
            "original_scores": original_scores,
            "final_scores": final_scores,
            "deltas": {
                "ai_delta": final_scores.get("ai_delta"),
                "topk_delta": final_scores.get("topk_delta"),
                "external_delta": final_scores.get("external_delta"),
                "rank_delta": final_scores.get("rank_delta"),
                "topk_calibrated_risk_delta": final_scores.get("topk_calibrated_risk_delta"),
                "qualifying_text_ai_density_delta": final_scores.get("qualifying_text_ai_density_delta"),
                "ai_authorship_delta": final_scores.get("ai_authorship_delta"),
                "external_ai_flag_risk_delta": final_scores.get("external_ai_flag_risk_delta"),
                "unsafe_word_ratio_delta": final_scores.get("unsafe_word_ratio_delta"),
                "unsafe_cluster_delta": final_scores.get("unsafe_cluster_delta"),
            },
            "accepted_count": len(layer1_accepted) + len(layer3_accepted),
            "goal": {
                "status": final_goal.get("status"),
                "goal_met": final_goal.get("goal_met"),
                "reason": final_goal.get("reason"),
            },
        },
        "accepted": layer1_accepted + layer3_accepted,
        "rounds": layer1_payload.get("rounds") or [],
        "layers": {
            "normalized_repair": {
                "summary": layer1_payload.get("summary"),
                "accepted_count": len(layer1_accepted),
            },
            "budget_lane_optimizer": {
                "enabled": bool(layer3_enabled),
                "ran": bool(layer3_payload),
                "accepted_count": len(layer3_accepted),
                "summary": {
                    "original_scores": (layer3_payload or {}).get("original_scores"),
                    "final_scores": (layer3_payload or {}).get("final_scores"),
                    "goal": (layer3_payload or {}).get("goal"),
                } if layer3_payload else None,
            },
        },
        "config": {
            **(layer1_payload.get("config") or {}),
            "layer3_enabled": layer3_enabled,
            "layer3_max_rounds": layer3_max_rounds,
            "layer3_max_clusters": layer3_max_clusters,
            "layer3_variant_count": layer3_variant_count,
        },
        "rewritten_document": current_text,
    }
    (out_dir / "v4_layered_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (out_dir / "v4_rewritten_document.txt").write_text(current_text)
    (out_dir / "v4_final_scan.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2))
    return payload


def run_v4_fast_rewrite(
    *,
    input_text: str,
    output_dir: str | Path,
    unit_ids: set[str] | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Production-shaped V4 experiment with bounded search cost."""

    return run_v4_iterative_rewrite(
        input_text=input_text,
        output_dir=output_dir,
        unit_ids=unit_ids,
        include_llm_normalizer=False,
        include_tutor_normalizer=True,
        include_enrichment_normalizer=False,
        variant_count=2,
        max_rounds=2,
        groups_per_round=3,
        stop_after_accepted=2,
        strong_ai_delta=10.0,
        api_key=api_key,
        model=model,
        base_url=base_url,
        extra_body=extra_body,
    )


def run_v4_cluster_patch_experiment(
    *,
    input_text: str,
    output_dir: str | Path,
    max_clusters: int = 4,
    variant_count: int = 2,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Experiment with bounded patches over scanner unsafe clusters."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    current_text = str(input_text or "")
    baseline_report = _scan_report(current_text)
    baseline_goal = evaluate_rewrite_goal(
        original_text=current_text,
        candidate_text=current_text,
        original_report=baseline_report,
        candidate_report=baseline_report,
    ).to_dict()
    baseline_metrics = _metrics(input_text=current_text, report=baseline_report, goal=baseline_goal)
    baseline = _score_summary(baseline_report, baseline_metrics)
    baseline["unsafe_cluster_count"] = _unsafe_cluster_count(baseline_goal)
    baseline["unsafe_word_ratio"] = _unsafe_word_ratio(baseline_goal, baseline)
    _add_goal_driver_snapshot(baseline, baseline_goal)

    units = build_cluster_repair_units(
        text=current_text,
        report=baseline_report,
        goal=baseline_goal,
        limit=max_clusters,
    )
    gateway = LLMGateway(LLMConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=8000,
        temperature=0.25,
        top_p=0.85,
        timeout=180,
        extra_body=extra_body,
    ))
    rows: list[dict[str, Any]] = []
    for unit in units:
        variants, diagnostics, prompt, completion = generate_cluster_variants(
            unit=unit,
            gateway=gateway,
            variant_count=variant_count,
        )
        stem = f"{unit.cluster_id}"
        (out_dir / f"{stem}_cluster_prompt.json.txt").write_text(prompt)
        (out_dir / f"{stem}_cluster_completion.json.txt").write_text(completion)
        for variant in variants:
            row = _score_cluster_variant(
                original_text=current_text,
                baseline_report=baseline_report,
                baseline_goal=baseline_goal,
                baseline=baseline,
                unit=unit,
                variant=variant,
                output_dir=out_dir,
                stem=stem,
            )
            row["generator_diagnostics"] = diagnostics
            rows.append(row)

    ranked = sorted(rows, key=_cluster_candidate_sort_key, reverse=True)
    best = next((row for row in ranked if _is_safe_cluster_positive(row)), None)
    combo_ranked = _rank_cluster_combinations(
        original_text=current_text,
        baseline_report=baseline_report,
        baseline=baseline,
        units=units,
        rows=rows,
        output_dir=out_dir,
    )
    best_combo = combo_ranked[0] if combo_ranked else None
    payload = {
        "baseline": baseline,
        "cluster_units": [unit.to_dict() for unit in units],
        "results": rows,
        "summary_ranked": [_compact_cluster_row(row) for row in ranked],
        "best_safe_candidate": _compact_cluster_row(best) if best else None,
        "combo_ranked": [_compact_cluster_combo_row(row) for row in combo_ranked],
        "best_combo_candidate": _compact_cluster_combo_row(best_combo) if best_combo else None,
    }
    final_row = best_combo or best
    if final_row:
        payload["rewritten_document"] = final_row.get("candidate_text", "")
        payload["final_scores"] = (final_row.get("scores") or {})
    (out_dir / "v4_cluster_patch_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    if final_row and final_row.get("candidate_text"):
        (out_dir / "v4_cluster_rewritten_document.txt").write_text(str(final_row.get("candidate_text") or ""))
    return payload


def run_v4_residual_cluster_experiment(
    *,
    input_text: str,
    output_dir: str | Path,
    max_rounds: int = 2,
    max_clusters: int = 6,
    variant_count: int = 2,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Iteratively patch remaining residual clusters after the main cluster layer."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_text = str(input_text or "")
    original_report = _scan_report(original_text)
    original_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=original_text,
        original_report=original_report,
        candidate_report=original_report,
    ).to_dict()
    original_metrics = _metrics(input_text=original_text, report=original_report, goal=original_goal)
    original_scores = _score_summary(original_report, original_metrics)
    original_scores["unsafe_cluster_count"] = _unsafe_cluster_count(original_goal)
    original_scores["unsafe_word_ratio"] = _unsafe_word_ratio(original_goal, original_scores)
    _add_goal_driver_snapshot(original_scores, original_goal)

    gateway = LLMGateway(LLMConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=8000,
        temperature=0.25,
        top_p=0.85,
        timeout=180,
        extra_body=extra_body,
    ))
    current_text = original_text
    current_report = original_report
    current_goal = original_goal
    current_scores = dict(original_scores)
    rounds: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []

    for round_index in range(1, max(1, int(max_rounds or 1)) + 1):
        units = build_cluster_repair_units(
            text=current_text,
            report=current_report,
            goal=current_goal,
            limit=max_clusters,
        )
        round_dir = out_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for unit in units:
            variants, diagnostics, prompt, completion = generate_cluster_variants(
                unit=unit,
                gateway=gateway,
                variant_count=variant_count,
                mode="residual_cluster_splitter",
            )
            stem = f"round{round_index}_{unit.cluster_id}"
            (round_dir / f"{stem}_prompt.json.txt").write_text(prompt)
            (round_dir / f"{stem}_completion.json.txt").write_text(completion)
            for variant in variants:
                row = _score_cluster_variant(
                    original_text=current_text,
                    baseline_report=current_report,
                    baseline_goal=current_goal,
                    baseline=current_scores,
                    unit=unit,
                    variant=variant,
                    output_dir=round_dir,
                    stem=stem,
                )
                row["generator_diagnostics"] = diagnostics
                rows.append(row)

        ranked = sorted(rows, key=_residual_candidate_sort_key, reverse=True)
        combo_ranked = _rank_cluster_combinations(
            original_text=current_text,
            baseline_report=current_report,
            baseline=current_scores,
            units=units,
            rows=rows,
            output_dir=round_dir,
            sort_key=_residual_combo_sort_key,
        )
        best = next((row for row in combo_ranked if _is_safe_residual_positive(row)), None)
        if best is None:
            best = next((row for row in ranked if _is_safe_residual_positive(row)), None)
        round_log = {
            "round": round_index,
            "baseline": current_scores,
            "cluster_units": [unit.to_dict() for unit in units],
            "summary_ranked": [_compact_cluster_row(row) for row in ranked],
            "combo_ranked": [_compact_cluster_combo_row(row) for row in combo_ranked],
            "accepted": _compact_cluster_combo_row(best) if best and best.get("combo_id") else _compact_cluster_row(best),
        }
        rounds.append(round_log)
        if best is None:
            round_log["stop_reason"] = "no_safe_residual_candidate"
            break
        current_text = str(best.get("candidate_text") or current_text)
        current_report = best.get("candidate_report") if isinstance(best.get("candidate_report"), dict) else _scan_report(current_text)
        current_goal = best.get("candidate_goal") if isinstance(best.get("candidate_goal"), dict) else evaluate_rewrite_goal(
            original_text=original_text,
            candidate_text=current_text,
            original_report=original_report,
            candidate_report=current_report,
        ).to_dict()
        current_metrics = _metrics(input_text=original_text, report=current_report, goal=current_goal)
        current_scores = _score_summary(current_report, current_metrics)
        current_scores["unsafe_cluster_count"] = _unsafe_cluster_count(current_goal)
        current_scores["unsafe_word_ratio"] = _unsafe_word_ratio(current_goal, current_scores)
        _add_goal_driver_snapshot(current_scores, current_goal)
        _add_score_deltas(current_scores, original_scores)
        accepted.append({
            **(_compact_cluster_combo_row(best) if best.get("combo_id") else _compact_cluster_row(best) or {}),
            "round": round_index,
            "scores_after": current_scores,
        })

    final_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=current_text,
        original_report=original_report,
        candidate_report=current_report,
    ).to_dict()
    final_scores = _score_summary(current_report, _metrics(input_text=original_text, report=current_report, goal=final_goal))
    final_scores["unsafe_cluster_count"] = _unsafe_cluster_count(final_goal)
    final_scores["unsafe_word_ratio"] = _unsafe_word_ratio(final_goal, final_scores)
    _add_goal_driver_snapshot(final_scores, final_goal)
    _add_score_deltas(final_scores, original_scores)
    payload = {
        "original_scores": original_scores,
        "final_scores": final_scores,
        "accepted": accepted,
        "rounds": rounds,
        "rewritten_document": current_text,
        "goal": {
            "status": final_goal.get("status"),
            "goal_met": final_goal.get("goal_met"),
            "reason": final_goal.get("reason"),
        },
    }
    (out_dir / "v4_residual_cluster_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (out_dir / "v4_residual_rewritten_document.txt").write_text(current_text)
    (out_dir / "v4_residual_final_scan.json").write_text(json.dumps(current_report, ensure_ascii=False, indent=2))
    (out_dir / "v4_residual_final_goal.json").write_text(json.dumps(final_goal, ensure_ascii=False, indent=2))
    return payload


def run_v4_budget_lane_experiment(
    *,
    input_text: str,
    output_dir: str | Path,
    max_rounds: int = 2,
    max_clusters: int = 6,
    variant_count: int = 2,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Layer 3: widen edit budgets by lane, but accept only measured score gains."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_text = str(input_text or "")
    original_report = _scan_report(original_text)
    original_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=original_text,
        original_report=original_report,
        candidate_report=original_report,
    ).to_dict()
    original_metrics = _metrics(input_text=original_text, report=original_report, goal=original_goal)
    original_scores = _score_summary(original_report, original_metrics)
    original_scores["unsafe_cluster_count"] = _unsafe_cluster_count(original_goal)
    original_scores["unsafe_word_ratio"] = _unsafe_word_ratio(original_goal, original_scores)
    _add_goal_driver_snapshot(original_scores, original_goal)

    gateway = LLMGateway(LLMConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=8000,
        temperature=0.25,
        top_p=0.85,
        timeout=180,
        extra_body=extra_body,
    ))
    current_text = original_text
    current_report = original_report
    current_goal = original_goal
    current_scores = dict(original_scores)
    rounds: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []

    for round_index in range(1, max(1, int(max_rounds or 1)) + 1):
        units = build_cluster_repair_units(
            text=current_text,
            report=current_report,
            goal=current_goal,
            limit=max_clusters,
        )
        round_dir = out_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_log: dict[str, Any] = {
            "round": round_index,
            "baseline": current_scores,
            "cluster_units": [unit.to_dict() for unit in units],
            "lanes": [],
        }
        active_blocker = _active_layer3_blocker(current_goal, current_scores)
        round_log["active_blocker"] = active_blocker
        if not units:
            round_log["stop_reason"] = "no_cluster_units"
            rounds.append(round_log)
            break

        best: dict[str, Any] | None = None
        material_best: dict[str, Any] | None = None
        fallback_best: dict[str, Any] | None = None
        blocker_best: dict[str, Any] | None = None
        all_rows: list[dict[str, Any]] = []
        for lane in _ordered_budget_lanes(units):
            lane_rows: list[dict[str, Any]] = []
            lane_log = {
                "lane": lane.to_dict(),
                "results": lane_rows,
                "accepted": None,
                "blocker_accepted": None,
                "material_accepted": None,
                "safe_fallback": None,
            }
            round_log["lanes"].append(lane_log)
            for unit in units:
                if lane.lane_id not in {item.lane_id for item in budget_lanes_for_unit(unit.word_count, risk_score=unit.risk_score)}:
                    continue
                variants, diagnostics, prompt, completion = generate_budget_lane_variants(
                    unit=unit,
                    lane=lane,
                    gateway=gateway,
                    variant_count=variant_count,
                    blocker_contract=active_blocker,
                )
                stem = f"round{round_index}_{lane.lane_id}_{unit.cluster_id}"
                (round_dir / f"{stem}_prompt.json.txt").write_text(prompt)
                (round_dir / f"{stem}_completion.json.txt").write_text(completion)
                for variant in variants:
                    row = _score_budget_lane_variant(
                        original_text=current_text,
                        baseline_report=current_report,
                        baseline_goal=current_goal,
                        baseline=current_scores,
                        unit=unit,
                        lane=lane,
                        variant=variant,
                        output_dir=round_dir,
                        stem=stem,
                    )
                    row["generator_diagnostics"] = diagnostics
                    lane_rows.append(row)
                    all_rows.append(row)
            ranked = sorted(lane_rows, key=_budget_lane_candidate_sort_key, reverse=True)
            lane_log["results"] = [_compact_budget_lane_row(row) for row in ranked]
            lane_blocker_best = next((row for row in ranked if _is_blocker_budget_lane_positive(row, active_blocker)), None)
            lane_safe_best = next((row for row in ranked if _is_safe_budget_lane_positive(row)), None)
            lane_material_best = next((row for row in ranked if _is_material_budget_lane_positive(row)), None)
            lane_log["blocker_accepted"] = _compact_budget_lane_row(lane_blocker_best)
            lane_log["safe_fallback"] = _compact_budget_lane_row(lane_safe_best)
            lane_log["material_accepted"] = _compact_budget_lane_row(lane_material_best)
            if lane_blocker_best is not None:
                blocker_best = _better_budget_lane_row(blocker_best, lane_blocker_best)
            if lane_safe_best is not None:
                fallback_best = _better_budget_lane_row(fallback_best, lane_safe_best)
            if lane_material_best is not None:
                material_best = _better_budget_lane_row(material_best, lane_material_best)
            lane_log["accepted"] = _compact_budget_lane_row(lane_blocker_best or lane_material_best)

        rounds.append(round_log)
        pack_ranked = _rank_budget_lane_patch_packs(
            original_text=current_text,
            baseline_report=current_report,
            baseline=current_scores,
            units=units,
            rows=all_rows,
            output_dir=round_dir,
            active_blocker=active_blocker,
        )
        round_log["pack_ranked"] = [_compact_budget_lane_pack_row(row) for row in pack_ranked]
        pack_best = next((row for row in pack_ranked if _is_blocker_budget_lane_positive(row, active_blocker)), None)
        round_log["pack_best_accepted"] = _compact_budget_lane_pack_row(pack_best)

        best = pack_best or (blocker_best if active_blocker.get("active") else material_best)
        if best is None and not active_blocker.get("active"):
            best = material_best
        round_log["material_best_accepted"] = _compact_budget_lane_row(best)
        if (
            best is None
            and fallback_best is not None
            and not active_blocker.get("active")
            and _bool_env("DRAFTPROOF_REWRITE_V4_LAYER3_ACCEPT_WEAK_FALLBACK", True)
        ):
            best = fallback_best
            round_log["weak_fallback_accepted"] = _compact_budget_lane_row(best)
        if best is None:
            round_log["stop_reason"] = (
                "no_blocker_clearing_budget_lane_candidate"
                if active_blocker.get("active")
                else "no_safe_budget_lane_candidate"
            )
            break

        current_text = str(best.get("candidate_text") or current_text)
        current_report = best.get("candidate_report") if isinstance(best.get("candidate_report"), dict) else _scan_report(current_text)
        current_goal = best.get("candidate_goal") if isinstance(best.get("candidate_goal"), dict) else evaluate_rewrite_goal(
            original_text=original_text,
            candidate_text=current_text,
            original_report=original_report,
            candidate_report=current_report,
        ).to_dict()
        current_metrics = _metrics(input_text=original_text, report=current_report, goal=current_goal)
        current_scores = _score_summary(current_report, current_metrics)
        current_scores["unsafe_cluster_count"] = _unsafe_cluster_count(current_goal)
        current_scores["unsafe_word_ratio"] = _unsafe_word_ratio(current_goal, current_scores)
        _add_goal_driver_snapshot(current_scores, current_goal)
        _add_score_deltas(current_scores, original_scores)
        accepted.append({
            **(_compact_budget_lane_row(best) or {}),
            "round": round_index,
            "scores_after": current_scores,
        })

    final_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=current_text,
        original_report=original_report,
        candidate_report=current_report,
    ).to_dict()
    final_scores = _score_summary(current_report, _metrics(input_text=original_text, report=current_report, goal=final_goal))
    final_scores["unsafe_cluster_count"] = _unsafe_cluster_count(final_goal)
    final_scores["unsafe_word_ratio"] = _unsafe_word_ratio(final_goal, final_scores)
    _add_goal_driver_snapshot(final_scores, final_goal)
    _add_score_deltas(final_scores, original_scores)
    payload = {
        "layer": "v4_budget_lane_optimizer",
        "original_scores": original_scores,
        "final_scores": final_scores,
        "accepted": accepted,
        "rounds": rounds,
        "rewritten_document": current_text,
        "goal": {
            "status": final_goal.get("status"),
            "goal_met": final_goal.get("goal_met"),
            "reason": final_goal.get("reason"),
        },
    }
    (out_dir / "v4_budget_lane_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (out_dir / "v4_budget_lane_rewritten_document.txt").write_text(current_text)
    (out_dir / "v4_budget_lane_final_scan.json").write_text(json.dumps(current_report, ensure_ascii=False, indent=2))
    (out_dir / "v4_budget_lane_final_goal.json").write_text(json.dumps(final_goal, ensure_ascii=False, indent=2))
    return payload


def _score_cluster_variant(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_goal: dict[str, Any],
    baseline: dict[str, Any],
    unit: ClusterRepairUnit,
    variant: CandidateVariant,
    output_dir: Path,
    stem: str,
) -> dict[str, Any]:
    candidate_text, apply_status = apply_cluster_variant(original_text, unit, variant.text)
    source_grounding = source_grounding_integrity(unit.text, variant.text)
    if not apply_status.get("applied"):
        return _rejected_cluster_row(
            unit=unit,
            variant=variant,
            baseline=baseline,
            reason=str(apply_status.get("reason") or "cluster_apply_failed"),
            apply_status=apply_status,
            source_grounding=source_grounding,
        )
    if not source_grounding.get("passed"):
        return _rejected_cluster_row(
            unit=unit,
            variant=variant,
            baseline=baseline,
            reason="source_grounding_failed",
            apply_status=apply_status,
            source_grounding=source_grounding,
            candidate_text=candidate_text,
        )
    candidate_report = _scan_report(candidate_text)
    candidate_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=baseline_report,
        candidate_report=candidate_report,
    ).to_dict()
    candidate_metrics = _metrics(input_text=original_text, report=candidate_report, goal=candidate_goal)
    scores = _score_summary(candidate_report, candidate_metrics)
    scores["unsafe_cluster_count"] = _unsafe_cluster_count(candidate_goal)
    scores["unsafe_word_ratio"] = _unsafe_word_ratio(candidate_goal, scores)
    _add_goal_driver_snapshot(scores, candidate_goal)
    scores["ai_delta"] = round(_num(baseline.get("ai")) - _num(scores.get("ai")), 3)
    scores["topk_delta"] = round(_num(baseline.get("topk")) - _num(scores.get("topk")), 3)
    scores["external_delta"] = round(_num(baseline.get("external")) - _num(scores.get("external")), 3)
    scores["rank_delta"] = round(_num(baseline.get("rank")) - _num(scores.get("rank")), 3)
    scores["unsafe_cluster_delta"] = int(baseline.get("unsafe_cluster_count") or 0) - int(scores.get("unsafe_cluster_count") or 0)
    scores["unsafe_word_ratio_delta"] = round(_num(baseline.get("unsafe_word_ratio")) - _num(scores.get("unsafe_word_ratio")), 3)
    _add_blocker_deltas(scores, baseline)
    safe_name = f"{stem}_{variant.variant_id}"
    (output_dir / f"{safe_name}.txt").write_text(candidate_text)
    (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
    return {
        "cluster_id": unit.cluster_id,
        "variant_id": variant.variant_id,
        "word_count": variant.word_count,
        "text": variant.text,
        "apply_status": apply_status,
        "source_grounding": source_grounding,
        "scores": scores,
        "candidate_text": candidate_text,
        "candidate_report": candidate_report,
        "candidate_goal": candidate_goal,
        "goal": {
            "status": candidate_goal.get("status"),
            "goal_met": candidate_goal.get("goal_met"),
            "reason": candidate_goal.get("reason"),
        },
    }


def _score_budget_lane_variant(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_goal: dict[str, Any],
    baseline: dict[str, Any],
    unit: ClusterRepairUnit,
    lane: BudgetLane,
    variant: CandidateVariant,
    output_dir: Path,
    stem: str,
) -> dict[str, Any]:
    candidate_text, apply_status = apply_cluster_variant(original_text, unit, variant.text)
    edit_profile = edit_budget_profile(unit.text, variant.text)
    source_grounding = source_grounding_integrity(unit.text, variant.text, repair_mode=_budget_lane_repair_mode(lane))
    if not apply_status.get("applied"):
        row = _rejected_cluster_row(
            unit=unit,
            variant=variant,
            baseline=baseline,
            reason=str(apply_status.get("reason") or "cluster_apply_failed"),
            apply_status=apply_status,
            source_grounding=source_grounding,
        )
        row["budget_lane"] = lane.to_dict()
        row["edit_budget_profile"] = edit_profile
        return row
    if not budget_profile_passes(edit_profile, lane):
        row = _rejected_cluster_row(
            unit=unit,
            variant=variant,
            baseline=baseline,
            reason="budget_lane_failed",
            apply_status=apply_status,
            source_grounding=source_grounding,
            candidate_text=candidate_text,
        )
        row["budget_lane"] = lane.to_dict()
        row["edit_budget_profile"] = edit_profile
        return row
    if not source_grounding.get("passed"):
        row = _rejected_cluster_row(
            unit=unit,
            variant=variant,
            baseline=baseline,
            reason="source_grounding_failed",
            apply_status=apply_status,
            source_grounding=source_grounding,
            candidate_text=candidate_text,
        )
        row["budget_lane"] = lane.to_dict()
        row["edit_budget_profile"] = edit_profile
        return row
    candidate_report = _scan_report(candidate_text)
    candidate_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=baseline_report,
        candidate_report=candidate_report,
    ).to_dict()
    candidate_metrics = _metrics(input_text=original_text, report=candidate_report, goal=candidate_goal)
    scores = _score_summary(candidate_report, candidate_metrics)
    scores["unsafe_cluster_count"] = _unsafe_cluster_count(candidate_goal)
    scores["unsafe_word_ratio"] = _unsafe_word_ratio(candidate_goal, scores)
    _add_goal_driver_snapshot(scores, candidate_goal)
    _add_score_deltas(scores, baseline)
    safe_name = f"{stem}_{variant.variant_id}"
    (output_dir / f"{safe_name}.txt").write_text(candidate_text)
    (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
    return {
        "cluster_id": unit.cluster_id,
        "variant_id": variant.variant_id,
        "budget_lane": lane.to_dict(),
        "edit_budget_profile": edit_profile,
        "word_count": variant.word_count,
        "text": variant.text,
        "apply_status": apply_status,
        "source_grounding": source_grounding,
        "scores": scores,
        "candidate_text": candidate_text,
        "candidate_report": candidate_report,
        "candidate_goal": candidate_goal,
        "goal": {
            "status": candidate_goal.get("status"),
            "goal_met": candidate_goal.get("goal_met"),
            "reason": candidate_goal.get("reason"),
        },
    }


def _rank_budget_lane_patch_packs(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline: dict[str, Any],
    units: list[ClusterRepairUnit],
    rows: list[dict[str, Any]],
    output_dir: Path,
    active_blocker: dict[str, Any],
) -> list[dict[str, Any]]:
    unit_by_id = {unit.cluster_id: unit for unit in units}
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if not ((row.get("apply_status") or {}).get("applied")):
            continue
        if not ((row.get("source_grounding") or {}).get("passed")):
            continue
        if row.get("cluster_id") not in unit_by_id:
            continue
        if row.get("budget_lane") and row.get("edit_budget_profile"):
            lane_data = row.get("budget_lane") or {}
            lane = BudgetLane(
                lane_id=str(lane_data.get("lane_id") or "unknown"),
                changed_source_ratio_max=_num(lane_data.get("changed_source_ratio_max")),
                growth_ratio_max=_num(lane_data.get("growth_ratio_max")),
                shrink_ratio_max=_num(lane_data.get("shrink_ratio_max")),
                operation_families=tuple(lane_data.get("operation_families") or ()),
                instruction=str(lane_data.get("instruction") or ""),
            )
            if not budget_profile_passes(row.get("edit_budget_profile") or {}, lane):
                continue
        eligible.append(row)
    if not eligible:
        return []

    sort_key = lambda row: _blocker_budget_lane_candidate_sort_key(row, active_blocker)
    per_cluster: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(eligible, key=sort_key, reverse=True):
        cluster_id = str(row.get("cluster_id") or "")
        if not cluster_id:
            continue
        per_cluster.setdefault(cluster_id, [])
        if len(per_cluster[cluster_id]) < _int_env("DRAFTPROOF_REWRITE_V4_LAYER3_PACK_VARIANTS_PER_CLUSTER", 2, minimum=1, maximum=5):
            per_cluster[cluster_id].append(row)

    selected_rows = [row for rows_for_cluster in per_cluster.values() for row in rows_for_cluster]
    selected_rows = sorted(selected_rows, key=sort_key, reverse=True)[
        :_int_env("DRAFTPROOF_REWRITE_V4_LAYER3_PACK_CANDIDATE_LIMIT", 10, minimum=1, maximum=30)
    ]
    max_size = min(
        len({row.get("cluster_id") for row in selected_rows}),
        _int_env("DRAFTPROOF_REWRITE_V4_LAYER3_PACK_MAX_SIZE", 4, minimum=2, maximum=8),
    )
    if max_size < 2:
        return []
    max_evals = _int_env("DRAFTPROOF_REWRITE_V4_LAYER3_PACK_EVAL_LIMIT", 50, minimum=1, maximum=500)
    pack_rows: list[dict[str, Any]] = []
    for size in range(2, max_size + 1):
        for combo in combinations(selected_rows, size):
            if len(pack_rows) >= max_evals:
                pack_rows.sort(key=sort_key, reverse=True)
                return pack_rows
            cluster_ids = [str(row.get("cluster_id") or "") for row in combo]
            if len(set(cluster_ids)) != len(cluster_ids):
                continue
            candidate_text = original_text
            apply_statuses: list[dict[str, Any]] = []
            for row in sorted(combo, key=lambda item: unit_by_id[str(item.get("cluster_id"))].start_char, reverse=True):
                unit = unit_by_id[str(row.get("cluster_id"))]
                candidate_text, status = apply_cluster_variant(candidate_text, unit, str(row.get("text") or ""))
                apply_statuses.append(status)
                if not status.get("applied"):
                    break
            if not all(status.get("applied") for status in apply_statuses):
                continue
            candidate_report = _scan_report(candidate_text)
            goal_eval = evaluate_rewrite_goal(
                original_text=original_text,
                candidate_text=candidate_text,
                original_report=baseline_report,
                candidate_report=candidate_report,
            )
            candidate_goal = goal_eval.to_dict() if hasattr(goal_eval, "to_dict") else dict(goal_eval or {})
            candidate_metrics = _metrics(input_text=original_text, report=candidate_report, goal=candidate_goal)
            scores = _score_summary(candidate_report, candidate_metrics)
            scores["unsafe_cluster_count"] = _unsafe_cluster_count(candidate_goal)
            scores["unsafe_word_ratio"] = _unsafe_word_ratio(candidate_goal, scores)
            _add_goal_driver_snapshot(scores, candidate_goal)
            _add_score_deltas(scores, baseline)
            pack_id = "_".join(f"{row.get('cluster_id')}-{row.get('variant_id')}" for row in combo)
            safe_name = f"pack_{len(pack_rows) + 1:03d}"
            (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
            pack_rows.append({
                "combo_id": pack_id,
                "cluster_id": "cluster_pack",
                "variant_id": safe_name,
                "budget_lane": {"lane_id": "clearance_pack"},
                "edit_budget_profile": {"changed_source_ratio": 0.0, "growth_ratio": 0.0, "shrink_ratio": 0.0},
                "apply_status": {"applied": True, "pack_size": size},
                "source_grounding": {"passed": True, "repair_mode": "cluster_clearance_pack"},
                "patches": [
                    {
                        "cluster_id": row.get("cluster_id"),
                        "variant_id": row.get("variant_id"),
                        "budget_lane": (row.get("budget_lane") or {}).get("lane_id") if isinstance(row.get("budget_lane"), dict) else None,
                        "text": row.get("text"),
                    }
                    for row in combo
                ],
                "apply_statuses": apply_statuses,
                "scores": scores,
                "candidate_text": candidate_text,
                "candidate_report": candidate_report,
                "candidate_goal": candidate_goal,
                "goal": {
                    "status": candidate_goal.get("status"),
                    "goal_met": candidate_goal.get("goal_met"),
                    "reason": candidate_goal.get("reason"),
                },
                "active_blocker": active_blocker,
            })
    pack_rows.sort(key=sort_key, reverse=True)
    return pack_rows


def _budget_lane_repair_mode(lane: BudgetLane) -> str:
    if lane.lane_id in {"route", "aggressive", "clearance"}:
        return "source_near_route_rebuild"
    return "source_preserving_repair"


def _rejected_cluster_row(
    *,
    unit: ClusterRepairUnit,
    variant: CandidateVariant,
    baseline: dict[str, Any],
    reason: str,
    apply_status: dict[str, Any],
    source_grounding: dict[str, Any],
    candidate_text: str | None = None,
) -> dict[str, Any]:
    scores = {
        **baseline,
        "ai_delta": 0.0,
        "topk_delta": 0.0,
        "external_delta": 0.0,
        "rank_delta": 0.0,
        "risky_window_delta": 0,
        "unsafe_cluster_delta": 0,
        "unsafe_word_ratio_delta": 0.0,
    }
    row: dict[str, Any] = {
        "cluster_id": unit.cluster_id,
        "variant_id": variant.variant_id,
        "word_count": variant.word_count,
        "text": variant.text,
        "apply_status": apply_status,
        "source_grounding": source_grounding,
        "scores": scores,
        "goal": {
            "status": "rejected_candidate",
            "goal_met": False,
            "reason": reason,
        },
    }
    if candidate_text is not None:
        row["candidate_text"] = candidate_text
    return row


def _rank_cluster_combinations(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline: dict[str, Any],
    units: list[ClusterRepairUnit],
    rows: list[dict[str, Any]],
    output_dir: Path,
    sort_key: Any | None = None,
) -> list[dict[str, Any]]:
    unit_by_id = {unit.cluster_id: unit for unit in units}
    eligible = [
        row for row in rows
        if ((row.get("apply_status") or {}).get("applied"))
        and ((row.get("source_grounding") or {}).get("passed"))
        and row.get("cluster_id") in unit_by_id
    ]
    if not eligible:
        return []
    single_sort_key = _residual_candidate_sort_key if sort_key is not None else _cluster_candidate_sort_key
    eligible = sorted(eligible, key=single_sort_key, reverse=True)[
        :_int_env("DRAFTPROOF_REWRITE_V4_CLUSTER_COMBO_CANDIDATE_LIMIT", 8, minimum=1, maximum=20)
    ]
    max_size = min(
        len(eligible),
        _int_env("DRAFTPROOF_REWRITE_V4_CLUSTER_COMBO_MAX_SIZE", 4, minimum=1, maximum=6),
    )
    max_evals = _int_env("DRAFTPROOF_REWRITE_V4_CLUSTER_COMBO_EVAL_LIMIT", 40, minimum=1, maximum=500)
    combo_rows: list[dict[str, Any]] = []
    for size in range(1, max_size + 1):
        for combo in combinations(eligible, size):
            if len(combo_rows) >= max_evals:
                combo_rows.sort(key=sort_key or _cluster_combo_sort_key, reverse=True)
                return combo_rows
            cluster_ids = [str(row.get("cluster_id") or "") for row in combo]
            if len(set(cluster_ids)) != len(cluster_ids):
                continue
            candidate_text = original_text
            apply_statuses: list[dict[str, Any]] = []
            for row in sorted(combo, key=lambda item: unit_by_id[str(item.get("cluster_id"))].start_char, reverse=True):
                unit = unit_by_id[str(row.get("cluster_id"))]
                candidate_text, status = apply_cluster_variant(candidate_text, unit, str(row.get("text") or ""))
                apply_statuses.append(status)
                if not status.get("applied"):
                    break
            if not all(status.get("applied") for status in apply_statuses):
                continue
            candidate_report = _scan_report(candidate_text)
            candidate_goal = evaluate_rewrite_goal(
                original_text=original_text,
                candidate_text=candidate_text,
                original_report=baseline_report,
                candidate_report=candidate_report,
            ).to_dict()
            candidate_metrics = _metrics(input_text=original_text, report=candidate_report, goal=candidate_goal)
            scores = _score_summary(candidate_report, candidate_metrics)
            scores["unsafe_cluster_count"] = _unsafe_cluster_count(candidate_goal)
            scores["unsafe_word_ratio"] = _unsafe_word_ratio(candidate_goal, scores)
            _add_goal_driver_snapshot(scores, candidate_goal)
            scores["ai_delta"] = round(_num(baseline.get("ai")) - _num(scores.get("ai")), 3)
            scores["topk_delta"] = round(_num(baseline.get("topk")) - _num(scores.get("topk")), 3)
            scores["external_delta"] = round(_num(baseline.get("external")) - _num(scores.get("external")), 3)
            scores["rank_delta"] = round(_num(baseline.get("rank")) - _num(scores.get("rank")), 3)
            scores["unsafe_cluster_delta"] = int(baseline.get("unsafe_cluster_count") or 0) - int(scores.get("unsafe_cluster_count") or 0)
            scores["unsafe_word_ratio_delta"] = round(_num(baseline.get("unsafe_word_ratio")) - _num(scores.get("unsafe_word_ratio")), 3)
            _add_blocker_deltas(scores, baseline)
            combo_id = "_".join(f"{row.get('cluster_id')}-{row.get('variant_id')}" for row in combo)
            safe_name = f"combo_{len(combo_rows) + 1:03d}"
            (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
            combo_rows.append({
                "combo_id": combo_id,
                "patches": [
                    {
                        "cluster_id": row.get("cluster_id"),
                        "variant_id": row.get("variant_id"),
                        "text": row.get("text"),
                    }
                    for row in combo
                ],
                "apply_statuses": apply_statuses,
                "scores": scores,
                "candidate_text": candidate_text,
                "candidate_report": candidate_report,
                "candidate_goal": candidate_goal,
                "goal": {
                    "status": candidate_goal.get("status"),
                    "goal_met": candidate_goal.get("goal_met"),
                    "reason": candidate_goal.get("reason"),
                },
            })
    combo_rows.sort(key=sort_key or _cluster_combo_sort_key, reverse=True)
    return combo_rows


def _score_variant(
    *,
    input_text: str,
    baseline_report: dict[str, Any],
    baseline: dict[str, Any],
    group: Any,
    repair_brief: RepairBrief,
    variant: CandidateVariant,
    output_dir: Path,
    stem: str,
) -> dict[str, Any]:
    candidate_text, apply_status = apply_target_replacements(
        original_text=input_text,
        target_groups=[group],
        replacements=[{"group_id": group.group_id, "replacement_text": variant.text}],
    )
    source_grounding = source_grounding_integrity(group.source_text, variant.text, repair_mode=repair_brief.repair_mode)
    if not source_grounding.get("passed"):
        return _rejected_variant_row(
            group=group,
            variant=variant,
            apply_status=apply_status,
            baseline=baseline,
            reason="source_grounding_failed",
            source_grounding=source_grounding,
            repair_brief=repair_brief,
        )
    strategy_compliance = strategy_compliance_integrity(variant.text, repair_brief.mitigation_strategy)
    if not strategy_compliance.get("passed"):
        return _rejected_variant_row(
            group=group,
            variant=variant,
            apply_status=apply_status,
            baseline=baseline,
            reason="strategy_compliance_failed",
            source_grounding=source_grounding,
            strategy_compliance=strategy_compliance,
            repair_brief=repair_brief,
        )
    candidate_report = _scan_report(candidate_text)
    candidate_goal = evaluate_rewrite_goal(
        original_text=input_text,
        candidate_text=candidate_text,
        original_report=baseline_report,
        candidate_report=candidate_report,
    ).to_dict()
    candidate_metrics = _metrics(input_text=input_text, report=candidate_report, goal=candidate_goal)
    scores = _score_summary(candidate_report, candidate_metrics)
    safe_name = f"{stem}_{variant.variant_id}"
    (output_dir / f"{safe_name}.txt").write_text(candidate_text)
    (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
    return {
        "unit_id": group.unit_id,
        "group_id": group.group_id,
        "variant_id": variant.variant_id,
        "repair_mode": repair_brief.repair_mode,
        "external_review_required": bool(source_grounding.get("external_review_required")),
        "word_count": variant.word_count,
        "text": variant.text,
        "source_grounding": source_grounding,
        "strategy_compliance": strategy_compliance,
        "apply_status": apply_status,
        "scores": {
            **scores,
            "ai_delta": round(float(baseline.get("ai") or 0.0) - float(scores.get("ai") or 0.0), 3),
            "topk_delta": round(float(baseline.get("topk") or 0.0) - float(scores.get("topk") or 0.0), 3),
            "external_delta": round(float(baseline.get("external") or 0.0) - float(scores.get("external") or 0.0), 3),
            "rank_delta": round(float(baseline.get("rank") or 0.0) - float(scores.get("rank") or 0.0), 3),
        },
        "goal": {
            "status": candidate_goal.get("status"),
            "goal_met": candidate_goal.get("goal_met"),
            "reason": candidate_goal.get("reason"),
        },
    }


def _score_variant_against_current(
    *,
    original_text: str,
    current_text: str,
    current_report: dict[str, Any],
    current_baseline: dict[str, Any],
    group: Any,
    repair_brief: RepairBrief,
    variant: CandidateVariant,
    output_dir: Path,
    stem: str,
) -> dict[str, Any]:
    candidate_text, apply_status = apply_target_replacements(
        original_text=current_text,
        target_groups=[group],
        replacements=[{"group_id": group.group_id, "replacement_text": variant.text}],
    )
    source_grounding = source_grounding_integrity(group.source_text, variant.text, repair_mode=repair_brief.repair_mode)
    if not source_grounding.get("passed"):
        return _rejected_variant_row(
            group=group,
            variant=variant,
            apply_status=apply_status,
            baseline=current_baseline,
            reason="source_grounding_failed",
            source_grounding=source_grounding,
            candidate_text=candidate_text,
            repair_brief=repair_brief,
        )
    strategy_compliance = strategy_compliance_integrity(variant.text, repair_brief.mitigation_strategy)
    if not strategy_compliance.get("passed"):
        return _rejected_variant_row(
            group=group,
            variant=variant,
            apply_status=apply_status,
            baseline=current_baseline,
            reason="strategy_compliance_failed",
            source_grounding=source_grounding,
            strategy_compliance=strategy_compliance,
            candidate_text=candidate_text,
            repair_brief=repair_brief,
        )
    candidate_report = _scan_report(candidate_text)
    candidate_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=current_report,
        candidate_report=candidate_report,
    ).to_dict()
    candidate_metrics = _metrics(input_text=original_text, report=candidate_report, goal=candidate_goal)
    scores = _score_summary(candidate_report, candidate_metrics)
    safe_name = f"{stem}_{variant.variant_id}"
    (output_dir / f"{safe_name}.txt").write_text(candidate_text)
    (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
    return {
        "unit_id": group.unit_id,
        "group_id": group.group_id,
        "variant_id": variant.variant_id,
        "repair_mode": repair_brief.repair_mode,
        "external_review_required": bool(source_grounding.get("external_review_required")),
        "word_count": variant.word_count,
        "text": variant.text,
        "source_grounding": source_grounding,
        "strategy_compliance": strategy_compliance,
        "apply_status": apply_status,
        "scores": {
            **scores,
            "ai_delta": round(_num(current_baseline.get("ai")) - _num(scores.get("ai")), 3),
            "topk_delta": round(_num(current_baseline.get("topk")) - _num(scores.get("topk")), 3),
            "external_delta": round(_num(current_baseline.get("external")) - _num(scores.get("external")), 3),
            "rank_delta": round(_num(current_baseline.get("rank")) - _num(scores.get("rank")), 3),
        },
        "candidate_text": candidate_text,
        "candidate_report": candidate_report,
        "candidate_goal": candidate_goal,
        "goal": {
            "status": candidate_goal.get("status"),
            "goal_met": candidate_goal.get("goal_met"),
            "reason": candidate_goal.get("reason"),
        },
    }


def _rejected_variant_row(
    *,
    group: Any,
    variant: CandidateVariant,
    apply_status: dict[str, Any],
    baseline: dict[str, Any],
    reason: str,
    source_grounding: dict[str, Any],
    strategy_compliance: dict[str, Any] | None = None,
    repair_brief: RepairBrief,
    candidate_text: str | None = None,
) -> dict[str, Any]:
    scores = {
        **baseline,
        "ai_delta": 0.0,
        "topk_delta": 0.0,
        "external_delta": 0.0,
        "rank_delta": 0.0,
    }
    row = {
        "unit_id": group.unit_id,
        "group_id": group.group_id,
        "variant_id": variant.variant_id,
        "repair_mode": repair_brief.repair_mode,
        "external_review_required": bool(source_grounding.get("external_review_required")),
        "word_count": variant.word_count,
        "text": variant.text,
        "apply_status": apply_status,
        "source_grounding": source_grounding,
        "strategy_compliance": strategy_compliance or {"passed": True, "failures": []},
        "scores": scores,
        "goal": {
            "status": "rejected_candidate",
            "goal_met": False,
            "reason": reason,
        },
    }
    if candidate_text is not None:
        row["candidate_text"] = candidate_text
        row["candidate_report"] = {}
        row["candidate_goal"] = row["goal"]
    return row


def _rank_groups_for_v4(groups: list[Any]) -> list[Any]:
    def group_score(group: Any) -> tuple[float, float, str]:
        score = 0.0
        max_driver = 0.0
        for target in getattr(group, "targets", ()) or ():
            if not isinstance(target, dict):
                continue
            for driver in target.get("dominant_drivers") or []:
                if not isinstance(driver, dict):
                    continue
                value = _num(driver.get("score"))
                score += min(value, 1.0)
                max_driver = max(max_driver, value)
        text_words = len(str(getattr(group, "source_text", "") or "").split())
        score += min(text_words, 90) / 200.0
        return (score, max_driver, str(getattr(group, "unit_id", "") or ""))

    return sorted(groups, key=group_score, reverse=True)


def _metrics(*, input_text: str, report: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    return scanner_controlled_metrics(
        report=report,
        goal=goal,
        footprint_risk=_footprint_risk(_ai_footprint_profile(report)),
        ai_score=_badge_ai(report),
        topk_score=_topk(report),
    )


def _score_summary(report: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "ai": _badge_ai(report),
        "topk": _topk(report),
        "external": metrics.get("external_proxy_score"),
        "rank": scanner_controlled_rank(metrics),
        "risky_window_count": metrics.get("risky_window_count"),
        "unsafe_word_ratio": metrics.get("unsafe_word_ratio"),
    }


def _rank_results(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        normalizer = ((experiment.get("repair_brief") or {}).get("normalizer"))
        rejected_tasks = len((experiment.get("repair_brief") or {}).get("rejected_tasks") or [])
        for result in experiment.get("results") or []:
            scores = result.get("scores") or {}
            rows.append({
                "unit_id": result.get("unit_id"),
                "group_id": result.get("group_id"),
                "normalizer": normalizer,
                "variant_id": result.get("variant_id"),
                "repair_mode": result.get("repair_mode"),
                "external_review_required": bool(result.get("external_review_required")),
                "word_count": result.get("word_count"),
                "ai_delta": scores.get("ai_delta"),
                "topk_delta": scores.get("topk_delta"),
                "external_delta": scores.get("external_delta"),
                "rank_delta": scores.get("rank_delta"),
                "ai": scores.get("ai"),
                "topk": scores.get("topk"),
                "external": scores.get("external"),
                "rank": scores.get("rank"),
                "rejected_normalizer_task_count": rejected_tasks,
                "text": result.get("text"),
            })
    rows.sort(
        key=lambda row: (
            _num(row.get("rank_delta")),
            _num(row.get("ai_delta")),
            _num(row.get("external_delta")),
        ),
        reverse=True,
    )
    return rows


def _compact_result_row(row: dict[str, Any], brief: RepairBrief) -> dict[str, Any]:
    scores = row.get("scores") or {}
    return {
        "unit_id": row.get("unit_id"),
        "group_id": row.get("group_id"),
        "normalizer": brief.normalizer,
        "variant_id": row.get("variant_id"),
        "repair_mode": row.get("repair_mode") or brief.repair_mode,
        "external_review_required": bool(row.get("external_review_required")),
        "word_count": row.get("word_count"),
        "ai_delta": scores.get("ai_delta"),
        "topk_delta": scores.get("topk_delta"),
        "external_delta": scores.get("external_delta"),
        "rank_delta": scores.get("rank_delta"),
        "ai": scores.get("ai"),
        "topk": scores.get("topk"),
        "external": scores.get("external"),
        "rank": scores.get("rank"),
        "rejected_normalizer_task_count": len(brief.rejected_tasks),
        "text": row.get("text"),
    }


def _compact_cluster_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    scores = row.get("scores") or {}
    return {
        "cluster_id": row.get("cluster_id"),
        "variant_id": row.get("variant_id"),
        "word_count": row.get("word_count"),
        "ai_delta": scores.get("ai_delta"),
        "topk_delta": scores.get("topk_delta"),
        "external_delta": scores.get("external_delta"),
        "rank_delta": scores.get("rank_delta"),
        "risky_window_delta": scores.get("risky_window_delta"),
        "unsafe_cluster_delta": scores.get("unsafe_cluster_delta"),
        "unsafe_word_ratio_delta": scores.get("unsafe_word_ratio_delta"),
        "topk_calibrated_risk_delta": scores.get("topk_calibrated_risk_delta"),
        "qualifying_text_ai_density_delta": scores.get("qualifying_text_ai_density_delta"),
        "ai_authorship_delta": scores.get("ai_authorship_delta"),
        "external_ai_flag_risk_delta": scores.get("external_ai_flag_risk_delta"),
        "ai": scores.get("ai"),
        "topk": scores.get("topk"),
        "external": scores.get("external"),
        "rank": scores.get("rank"),
        "unsafe_cluster_count": scores.get("unsafe_cluster_count"),
        "unsafe_word_ratio": scores.get("unsafe_word_ratio"),
        "topk_calibrated_risk": scores.get("topk_calibrated_risk"),
        "qualifying_text_ai_density": scores.get("qualifying_text_ai_density"),
        "ai_authorship": scores.get("ai_authorship"),
        "external_ai_flag_risk": scores.get("external_ai_flag_risk"),
        "goal": row.get("goal"),
        "text": row.get("text"),
    }


def _compact_cluster_combo_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    scores = row.get("scores") or {}
    return {
        "combo_id": row.get("combo_id"),
        "patches": [
            {"cluster_id": patch.get("cluster_id"), "variant_id": patch.get("variant_id")}
            for patch in row.get("patches") or []
            if isinstance(patch, dict)
        ],
        "ai_delta": scores.get("ai_delta"),
        "topk_delta": scores.get("topk_delta"),
        "external_delta": scores.get("external_delta"),
        "rank_delta": scores.get("rank_delta"),
        "unsafe_cluster_delta": scores.get("unsafe_cluster_delta"),
        "unsafe_word_ratio_delta": scores.get("unsafe_word_ratio_delta"),
        "topk_calibrated_risk_delta": scores.get("topk_calibrated_risk_delta"),
        "qualifying_text_ai_density_delta": scores.get("qualifying_text_ai_density_delta"),
        "ai_authorship_delta": scores.get("ai_authorship_delta"),
        "external_ai_flag_risk_delta": scores.get("external_ai_flag_risk_delta"),
        "ai": scores.get("ai"),
        "topk": scores.get("topk"),
        "external": scores.get("external"),
        "rank": scores.get("rank"),
        "unsafe_cluster_count": scores.get("unsafe_cluster_count"),
        "unsafe_word_ratio": scores.get("unsafe_word_ratio"),
        "topk_calibrated_risk": scores.get("topk_calibrated_risk"),
        "qualifying_text_ai_density": scores.get("qualifying_text_ai_density"),
        "ai_authorship": scores.get("ai_authorship"),
        "external_ai_flag_risk": scores.get("external_ai_flag_risk"),
        "goal": row.get("goal"),
    }


def _compact_budget_lane_pack_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    scores = row.get("scores") or {}
    return {
        "combo_id": row.get("combo_id"),
        "patch_count": len(row.get("patches") or []),
        "patches": [
            {
                "cluster_id": patch.get("cluster_id"),
                "variant_id": patch.get("variant_id"),
                "budget_lane": patch.get("budget_lane"),
            }
            for patch in row.get("patches") or []
            if isinstance(patch, dict)
        ],
        "ai_delta": scores.get("ai_delta"),
        "topk_delta": scores.get("topk_delta"),
        "external_delta": scores.get("external_delta"),
        "rank_delta": scores.get("rank_delta"),
        "risky_window_delta": scores.get("risky_window_delta"),
        "unsafe_cluster_delta": scores.get("unsafe_cluster_delta"),
        "unsafe_word_ratio_delta": scores.get("unsafe_word_ratio_delta"),
        "topk_calibrated_risk_delta": scores.get("topk_calibrated_risk_delta"),
        "qualifying_text_ai_density_delta": scores.get("qualifying_text_ai_density_delta"),
        "ai_authorship_delta": scores.get("ai_authorship_delta"),
        "external_ai_flag_risk_delta": scores.get("external_ai_flag_risk_delta"),
        "ai": scores.get("ai"),
        "topk": scores.get("topk"),
        "external": scores.get("external"),
        "rank": scores.get("rank"),
        "unsafe_cluster_count": scores.get("unsafe_cluster_count"),
        "unsafe_word_ratio": scores.get("unsafe_word_ratio"),
        "topk_calibrated_risk": scores.get("topk_calibrated_risk"),
        "qualifying_text_ai_density": scores.get("qualifying_text_ai_density"),
        "ai_authorship": scores.get("ai_authorship"),
        "external_ai_flag_risk": scores.get("external_ai_flag_risk"),
        "goal": row.get("goal"),
    }


def _compact_budget_lane_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    if row.get("patches"):
        return _compact_budget_lane_pack_row(row)
    compact = _compact_cluster_row(row) or {}
    compact["budget_lane"] = (row.get("budget_lane") or {}).get("lane_id")
    compact["edit_budget_profile"] = row.get("edit_budget_profile")
    return compact


def _best_safe(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if _is_safe_positive(row):
            return row
    return None


def _is_safe_positive(row: dict[str, Any]) -> bool:
    rank_delta = _num(row.get("rank_delta"))
    rank_tolerance = _float_env("DRAFTPROOF_REWRITE_V4_RANK_REGRESSION_TOLERANCE", 0.25, minimum=0.0, maximum=2.0)
    topk_delta = _num(row.get("topk_delta"))
    external_delta = _num(row.get("external_delta"))
    ai_delta = _num(row.get("ai_delta"))
    return (
        ai_delta > 0.0
        and external_delta >= 0.0
        and (
            rank_delta > 0.0
            or (
                rank_delta >= -rank_tolerance
                and topk_delta >= 0.0
            )
        )
        and int(row.get("rejected_normalizer_task_count") or 0) == 0
    )


def _is_safe_cluster_positive(row: dict[str, Any]) -> bool:
    scores = row.get("scores") or {}
    if not ((row.get("apply_status") or {}).get("applied")):
        return False
    if not ((row.get("source_grounding") or {}).get("passed")):
        return False
    external_delta = _num(scores.get("external_delta"))
    rank_delta = _num(scores.get("rank_delta"))
    ai_delta = _num(scores.get("ai_delta"))
    cluster_delta = _num(scores.get("unsafe_cluster_delta"))
    unsafe_ratio_delta = _num(scores.get("unsafe_word_ratio_delta"))
    topk_delta = _num(scores.get("topk_delta"))
    return (
        external_delta >= 0.0
        and rank_delta >= 0.0
        and (ai_delta > 0.0 or topk_delta > 0.0)
        and (cluster_delta > 0.0 or unsafe_ratio_delta >= 1.0)
    )


def _is_safe_residual_positive(row: dict[str, Any]) -> bool:
    scores = row.get("scores") or {}
    if not all((status or {}).get("applied") for status in row.get("apply_statuses") or [row.get("apply_status")]):
        return False
    if row.get("source_grounding") is not None and not ((row.get("source_grounding") or {}).get("passed")):
        return False
    external_delta = _num(scores.get("external_delta"))
    rank_delta = _num(scores.get("rank_delta"))
    return (
        external_delta >= -0.1
        and rank_delta >= -0.5
        and (
            _num(scores.get("topk_calibrated_risk_delta")) >= 1.0
            or _num(scores.get("qualifying_text_ai_density_delta")) >= 1.0
            or _num(scores.get("unsafe_cluster_delta")) > 0.0
            or _num(scores.get("unsafe_word_ratio_delta")) >= 1.0
            or _num(scores.get("external_ai_flag_risk_delta")) >= 0.5
        )
    )


def _is_safe_budget_lane_positive(row: dict[str, Any]) -> bool:
    scores = row.get("scores") or {}
    if not ((row.get("apply_status") or {}).get("applied")):
        return False
    if row.get("budget_lane") and row.get("edit_budget_profile"):
        lane_data = row.get("budget_lane") or {}
        lane = BudgetLane(
            lane_id=str(lane_data.get("lane_id") or "unknown"),
            changed_source_ratio_max=_num(lane_data.get("changed_source_ratio_max")),
            growth_ratio_max=_num(lane_data.get("growth_ratio_max")),
            shrink_ratio_max=_num(lane_data.get("shrink_ratio_max")),
            operation_families=tuple(lane_data.get("operation_families") or ()),
            instruction=str(lane_data.get("instruction") or ""),
        )
        if not budget_profile_passes(row.get("edit_budget_profile") or {}, lane):
            return False
    if not ((row.get("source_grounding") or {}).get("passed")):
        return False
    external_delta = _num(scores.get("external_delta"))
    rank_delta = _num(scores.get("rank_delta"))
    ai_delta = _num(scores.get("ai_delta"))
    return (
        external_delta >= -0.1
        and rank_delta >= -0.5
        and (
            ai_delta > 0.0
            or _num(scores.get("topk_delta")) > 0.0
            or _num(scores.get("topk_calibrated_risk_delta")) >= 1.0
            or _num(scores.get("qualifying_text_ai_density_delta")) >= 1.0
            or _num(scores.get("unsafe_cluster_delta")) > 0.0
            or _num(scores.get("unsafe_word_ratio_delta")) >= 1.0
            or _num(scores.get("external_ai_flag_risk_delta")) >= 0.5
        )
    )


def _is_blocker_budget_lane_positive(row: dict[str, Any], blocker: dict[str, Any] | None) -> bool:
    if not isinstance(blocker, dict) or not blocker.get("active"):
        return _is_material_budget_lane_positive(row)
    if not _is_safe_budget_lane_positive(row):
        return False
    scores = row.get("scores") or {}
    primary = str(blocker.get("primary") or "")
    if primary == "unsafe_cluster_count":
        return _num(scores.get("unsafe_cluster_delta")) > 0.0
    if primary == "unsafe_eligible_word_ratio":
        minimum_delta = _num(blocker.get("minimum_delta")) if blocker.get("minimum_delta") is not None else 1.0
        return _num(scores.get("unsafe_word_ratio_delta")) >= minimum_delta
    if primary == "topk_calibrated_risk":
        minimum_delta = _num(blocker.get("minimum_delta")) if blocker.get("minimum_delta") is not None else 2.0
        return _num(scores.get("topk_calibrated_risk_delta")) >= minimum_delta
    if primary == "external_ai_flag_risk":
        minimum_delta = _num(blocker.get("minimum_delta")) if blocker.get("minimum_delta") is not None else 1.0
        return _num(scores.get("external_ai_flag_risk_delta")) >= minimum_delta
    return _is_material_budget_lane_positive(row)


def _is_material_budget_lane_positive(row: dict[str, Any]) -> bool:
    if not _is_safe_budget_lane_positive(row):
        return False
    scores = row.get("scores") or {}
    candidate_goal = row.get("candidate_goal") if isinstance(row.get("candidate_goal"), dict) else {}
    gate = candidate_goal.get("ai_footprint_gate") if isinstance(candidate_goal.get("ai_footprint_gate"), dict) else {}
    thresholds = gate.get("thresholds") if isinstance(gate.get("thresholds"), dict) else {}
    return (
        _num(scores.get("topk_calibrated_risk_delta")) >= _material_threshold(thresholds, "topk_calibrated_risk", 2.0)
        or _num(scores.get("unsafe_word_ratio_delta")) >= _float_env("DRAFTPROOF_REWRITE_V4_LAYER3_MATERIAL_UNSAFE_WORD_DELTA", 2.0, minimum=0.0, maximum=25.0)
        or _num(scores.get("external_ai_flag_risk_delta")) >= _material_threshold(thresholds, "external_ai_flag_risk", 1.0)
        or _num(scores.get("ai_delta")) >= _material_threshold(thresholds, "ai_likelihood", 0.5)
        or _num(scores.get("unsafe_cluster_delta")) > 0.0
        or _num(scores.get("risky_window_delta")) > 0.0
    )


def _material_threshold(thresholds: dict[str, Any], key: str, default: float) -> float:
    env_name = f"DRAFTPROOF_REWRITE_V4_LAYER3_MATERIAL_{key.upper()}_DELTA"
    configured = os.environ.get(env_name)
    if configured is not None:
        return _float_env(env_name, default, minimum=0.0, maximum=100.0)
    value = thresholds.get(key)
    if value is None:
        return default
    return max(0.0, _num(value))


def _better_budget_lane_row(current: dict[str, Any] | None, candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max([current, candidate], key=_budget_lane_candidate_sort_key)


def _active_layer3_blocker(goal: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
    density = goal.get("eligible_span_density_gate") if isinstance(goal.get("eligible_span_density_gate"), dict) else {}
    density_thresholds = density.get("thresholds") if isinstance(density.get("thresholds"), dict) else {}
    unsafe_cluster_count = _num(scores.get("unsafe_cluster_count"))
    max_cluster_count = (
        _num(density_thresholds.get("max_unsafe_cluster_count"))
        if density_thresholds.get("max_unsafe_cluster_count") is not None
        else 4.0
    )
    if unsafe_cluster_count > max_cluster_count:
        return {
            "active": True,
            "primary": "unsafe_cluster_count",
            "current": unsafe_cluster_count,
            "target_max": max_cluster_count,
            "required_delta_to_safe": max(0.0, unsafe_cluster_count - max_cluster_count),
            "acceptance_rule": "candidate must reduce unsafe_cluster_count on full-document rescan",
        }

    unsafe_ratio = _num(scores.get("unsafe_word_ratio"))
    max_ratio = (
        _num(density_thresholds.get("max_unsafe_eligible_word_ratio"))
        if density_thresholds.get("max_unsafe_eligible_word_ratio") is not None
        else 35.0
    )
    if unsafe_ratio > max_ratio:
        return {
            "active": True,
            "primary": "unsafe_eligible_word_ratio",
            "current": unsafe_ratio,
            "target_max": max_ratio,
            "minimum_delta": min(2.0, max(0.5, unsafe_ratio - max_ratio)),
            "acceptance_rule": "candidate must reduce unsafe eligible word ratio on full-document rescan",
        }

    gate = goal.get("ai_footprint_gate") if isinstance(goal.get("ai_footprint_gate"), dict) else {}
    after = gate.get("after") if isinstance(gate.get("after"), dict) else {}
    authorship = after.get("authorship_footprint") if isinstance(after.get("authorship_footprint"), dict) else {}
    thresholds = gate.get("thresholds") if isinstance(gate.get("thresholds"), dict) else {}
    topk = _num(scores.get("topk_calibrated_risk") or authorship.get("topk_calibrated_risk"))
    topk_delta = _num(thresholds.get("topk_calibrated_risk")) if thresholds.get("topk_calibrated_risk") is not None else 2.0
    if topk > 35.0:
        return {
            "active": True,
            "primary": "topk_calibrated_risk",
            "current": topk,
            "target_max": 35.0,
            "minimum_delta": topk_delta,
            "acceptance_rule": "candidate must reduce calibrated top-k risk on full-document rescan",
        }

    external = _num(scores.get("external_ai_flag_risk"))
    external_delta = _num(thresholds.get("external_ai_flag_risk")) if thresholds.get("external_ai_flag_risk") is not None else 1.5
    if external > 35.0:
        return {
            "active": True,
            "primary": "external_ai_flag_risk",
            "current": external,
            "target_max": 35.0,
            "minimum_delta": external_delta,
            "acceptance_rule": "candidate must reduce external AI flag risk on full-document rescan",
        }

    return {"active": False, "primary": "", "acceptance_rule": "material positive movement"}


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        _num(row.get("ai_delta")),
        _num(row.get("external_delta")),
        _num(row.get("rank_delta")),
        _num(row.get("topk_delta")),
    )


def _cluster_candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    scores = row.get("scores") or {}
    return (
        _num(scores.get("unsafe_cluster_delta")),
        _num(scores.get("unsafe_word_ratio_delta")),
        _num(scores.get("external_delta")),
        _num(scores.get("rank_delta")),
        _num(scores.get("ai_delta")),
    )


def _cluster_combo_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    scores = row.get("scores") or {}
    return (
        _num(scores.get("external_delta")),
        _num(scores.get("rank_delta")),
        _num(scores.get("unsafe_word_ratio_delta")),
        _num(scores.get("topk_delta")),
        _num(scores.get("ai_delta")),
        _num(scores.get("unsafe_cluster_delta")),
    )


def _residual_candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float]:
    scores = row.get("scores") or {}
    return (
        _num(scores.get("topk_calibrated_risk_delta")),
        _num(scores.get("qualifying_text_ai_density_delta")),
        _num(scores.get("risky_window_delta")),
        _num(scores.get("unsafe_cluster_delta")),
        _num(scores.get("unsafe_word_ratio_delta")),
        _num(scores.get("external_ai_flag_risk_delta")),
        _num(scores.get("rank_delta")),
        _num(scores.get("ai_delta")),
    )


def _residual_combo_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float]:
    return _residual_candidate_sort_key(row)


def _blocker_budget_lane_candidate_sort_key(row: dict[str, Any], blocker: dict[str, Any] | None) -> tuple[float, ...]:
    if not isinstance(blocker, dict) or not blocker.get("active"):
        return _budget_lane_candidate_sort_key(row)
    scores = row.get("scores") or {}
    primary = str(blocker.get("primary") or "")
    if primary == "unsafe_cluster_count":
        return (
            _num(scores.get("unsafe_cluster_delta")),
            _num(scores.get("unsafe_word_ratio_delta")),
            _num(scores.get("topk_calibrated_risk_delta")),
            _num(scores.get("external_ai_flag_risk_delta")),
            _num(scores.get("rank_delta")),
            _num(scores.get("external_delta")),
            _num(scores.get("ai_delta")),
        )
    if primary == "unsafe_eligible_word_ratio":
        return (
            _num(scores.get("unsafe_word_ratio_delta")),
            _num(scores.get("unsafe_cluster_delta")),
            _num(scores.get("topk_calibrated_risk_delta")),
            _num(scores.get("external_ai_flag_risk_delta")),
            _num(scores.get("rank_delta")),
            _num(scores.get("external_delta")),
            _num(scores.get("ai_delta")),
        )
    if primary == "topk_calibrated_risk":
        return (
            _num(scores.get("topk_calibrated_risk_delta")),
            _num(scores.get("unsafe_cluster_delta")),
            _num(scores.get("unsafe_word_ratio_delta")),
            _num(scores.get("external_ai_flag_risk_delta")),
            _num(scores.get("rank_delta")),
            _num(scores.get("external_delta")),
            _num(scores.get("ai_delta")),
        )
    if primary == "external_ai_flag_risk":
        return (
            _num(scores.get("external_ai_flag_risk_delta")),
            _num(scores.get("unsafe_cluster_delta")),
            _num(scores.get("unsafe_word_ratio_delta")),
            _num(scores.get("topk_calibrated_risk_delta")),
            _num(scores.get("rank_delta")),
            _num(scores.get("external_delta")),
            _num(scores.get("ai_delta")),
        )
    return _budget_lane_candidate_sort_key(row)


def _budget_lane_candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float]:
    return _residual_candidate_sort_key(row)


def _ordered_budget_lanes(units: list[ClusterRepairUnit]) -> list[BudgetLane]:
    lanes_by_id: dict[str, BudgetLane] = {}
    order = ("conservative", "route", "aggressive", "clearance")
    for unit in units:
        for lane in budget_lanes_for_unit(unit.word_count, risk_score=unit.risk_score):
            lanes_by_id.setdefault(lane.lane_id, lane)
    return sorted(lanes_by_id.values(), key=lambda lane: order.index(lane.lane_id) if lane.lane_id in order else len(order))


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _add_score_deltas(scores: dict[str, Any], baseline: dict[str, Any]) -> None:
    scores["ai_delta"] = round(_num(baseline.get("ai")) - _num(scores.get("ai")), 3)
    scores["topk_delta"] = round(_num(baseline.get("topk")) - _num(scores.get("topk")), 3)
    scores["external_delta"] = round(_num(baseline.get("external")) - _num(scores.get("external")), 3)
    scores["rank_delta"] = round(_num(baseline.get("rank")) - _num(scores.get("rank")), 3)
    scores["risky_window_delta"] = int(baseline.get("risky_window_count") or 0) - int(scores.get("risky_window_count") or 0)
    scores["unsafe_cluster_delta"] = int(baseline.get("unsafe_cluster_count") or 0) - int(scores.get("unsafe_cluster_count") or 0)
    scores["unsafe_word_ratio_delta"] = round(_num(baseline.get("unsafe_word_ratio")) - _num(scores.get("unsafe_word_ratio")), 3)
    _add_blocker_deltas(scores, baseline)


def _add_goal_driver_snapshot(scores: dict[str, Any], goal: dict[str, Any]) -> None:
    scores["topk_calibrated_risk"] = _goal_driver_value(goal, "topk_calibrated_risk")
    scores["qualifying_text_ai_density"] = _goal_driver_value(goal, "qualifying_text_ai_density")
    scores["ai_authorship"] = _goal_driver_value(goal, "ai_authorship")
    scores["external_ai_flag_risk"] = _goal_driver_value(goal, "external_ai_flag_risk")


def _add_blocker_deltas(scores: dict[str, Any], baseline: dict[str, Any]) -> None:
    for key in (
        "topk_calibrated_risk",
        "qualifying_text_ai_density",
        "ai_authorship",
        "external_ai_flag_risk",
    ):
        scores[f"{key}_delta"] = round(_num(baseline.get(key)) - _num(scores.get(key)), 3)


def _goal_driver_value(goal: dict[str, Any], driver: str) -> float:
    gate = goal.get("ai_footprint_gate") if isinstance(goal.get("ai_footprint_gate"), dict) else {}
    after = gate.get("after") if isinstance(gate.get("after"), dict) else {}
    for bucket in (
        "authorship_footprint",
        "structural_footprint",
        "semantic_footprint",
        "grounding_footprint",
    ):
        values = after.get(bucket) if isinstance(after.get(bucket), dict) else {}
        if driver in values:
            return _num(values.get(driver))
    if driver in after:
        return _num(after.get(driver))
    return 0.0


def _unsafe_cluster_count(goal: dict[str, Any]) -> int:
    density = goal.get("eligible_span_density_gate") if isinstance(goal.get("eligible_span_density_gate"), dict) else {}
    try:
        return int(density.get("unsafe_cluster_count") or 0)
    except Exception:
        return 0


def _unsafe_word_ratio(goal: dict[str, Any], fallback: dict[str, Any] | None = None) -> float:
    density = goal.get("eligible_span_density_gate") if isinstance(goal.get("eligible_span_density_gate"), dict) else {}
    value = density.get("unsafe_eligible_word_ratio")
    if value is None and fallback:
        value = fallback.get("unsafe_word_ratio")
    return _num(value)


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
