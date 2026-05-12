"""Post-Top-k strict safe-band optimizer orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import time


@dataclass(frozen=True)
class PostTopkSafeBandOptimizerDeps:
    env_flag: Callable[[str, bool], bool]
    best_ai_search_selectable: Callable[[], bool]
    strict_ai_safe_band_status: Callable[[dict | None], dict]
    safe_topk_calibrated_limit: Callable[[], float]
    float_env: Callable[[str, float], float]
    verified_candidate_scans_used: Callable[[], int]
    extend_candidate_scan_budget: Callable[[dict, int, int], None]
    authorship_transformation_texture_driver_map: Callable[[str, dict | None], dict]
    authorship_transformation_texture_candidates: Callable[..., list[tuple[str, str, dict]]]
    generic_assertion_compiler_candidates: Callable[..., list[tuple[str, str, dict]]]
    blocker_operation_candidates: Callable[..., list[tuple[str, str, dict]]]
    content_pruning_candidates: Callable[..., list[tuple[str, str, dict]]]
    phase_budget_block_record: Callable[[str, dict], bool]
    record_phase_llm_call: Callable[[str], None]
    authorship_transformation_texture_patch_prompt: Callable[[str, dict | None], str]
    phase_chat_sampling_kwargs: Callable[..., dict]
    extract_post_topk_patch_candidates: Callable[..., list[dict]]
    apply_post_topk_patches: Callable[[str, list[dict]], tuple[str, list[dict]]]
    texture_candidate_family: Callable[[str | None], str]
    detect_protected_spans: Callable[[str], Any]
    review_burden: Callable[[dict | None], float]
    weighted_severity: Callable[[dict | None], float]
    finding_total: Callable[[dict | None], float]
    critical_high_count: Callable[[dict | None], int]
    turnitin_like_ai_profile: Callable[[dict | None], dict]
    search_budget_exhausted: Callable[[str], bool]
    ai_candidate_quality_reject_reason: Callable[[str], str]
    ai_search_protected_loss_reason: Callable[..., str]
    check_semantic_drift: Callable[..., Any]
    full_scan_report_dict: Callable[[str], dict]
    record_verified_candidate_scan: Callable[[], None]
    ai_footprint_gate_status: Callable[..., dict]
    turnitin_like_ai_gate_status: Callable[..., dict]
    formula_gap_contract: Callable[..., dict]
    badge_ai: Callable[[dict | None], float | int | None]
    contribution_scores: Callable[[dict | None], dict]
    integrity_scores: Callable[[dict | None], dict]
    formula_gap_candidate_rank: Callable[..., tuple]
    human_shift_score: Callable[..., dict]
    strict_safe_candidate_rank: Callable[..., tuple]
    accept_selected_candidate: Callable[..., None]
    get_best_text: Callable[[], str]
    get_best_report: Callable[[], dict | None]
    get_best_strategy: Callable[[], str]


def run_post_topk_ai_safe_band_optimizer(
    trigger_phase: str,
    *,
    search_summary: dict,
    search_budget: dict,
    gateway: Any,
    hard_llm_cap: int,
    original_report_dict: dict | None,
    original_review_burden: float,
    original_severity: float,
    saved_critical_high: int,
    search_source_text: str,
    deps: PostTopkSafeBandOptimizerDeps,
) -> None:
    texture_phase = "authorship_transformation_texture_controller"
    if not deps.env_flag("DRAFTPROOF_AUTHORSHIP_TRANSFORMATION_TEXTURE_CONTROLLER", True):
        disabled_summary = {"enabled": False, "reason": "disabled", "kind": texture_phase}
        search_summary[texture_phase] = disabled_summary
        search_summary["post_topk_optimizer"] = disabled_summary
        return
    if not deps.best_ai_search_selectable() or not isinstance(deps.get_best_report(), dict):
        skipped_summary = {"enabled": True, "skipped": True, "reason": "no_selectable_base", "kind": texture_phase}
        search_summary[texture_phase] = skipped_summary
        search_summary["post_topk_optimizer"] = skipped_summary
        return
    base_strict = deps.strict_ai_safe_band_status(deps.get_best_report())
    base_profile = base_strict.get("profile") or {}
    if float(base_profile.get("topk_calibrated_risk", 100.0)) >= deps.safe_topk_calibrated_limit():
        skipped_summary = {
            "enabled": True,
            "skipped": True,
            "reason": "base_topk_not_safe",
            "base_strict_safe_band": base_strict,
            "kind": texture_phase,
        }
        search_summary[texture_phase] = skipped_summary
        search_summary["post_topk_optimizer"] = skipped_summary
        return

    try:
        scan_reserve = max(0, int(deps.float_env("DRAFTPROOF_POST_TOPK_SCAN_RESERVE", 12.0)))
    except (TypeError, ValueError):
        scan_reserve = 12
    if scan_reserve > 0:
        previous_max = int(search_budget.get("max_candidate_scans") or 0)
        current_scans = deps.verified_candidate_scans_used()
        deps.extend_candidate_scan_budget(search_budget, current_scans, scan_reserve)

    try:
        llm_reserve = max(0, int(deps.float_env("DRAFTPROOF_POST_TOPK_LLM_RESERVE", 2.0)))
    except (TypeError, ValueError):
        llm_reserve = 2
    if llm_reserve > 0:
        search_budget["max_llm_calls"] = min(
            hard_llm_cap,
            max(
                int(search_budget.get("max_llm_calls") or 0),
                int(search_summary.get("llm_calls") or 0) + llm_reserve,
            ),
        )

    max_scans = max(1, int(deps.float_env("DRAFTPROOF_POST_TOPK_MAX_CANDIDATE_SCANS", 6.0)))
    driver_map = deps.authorship_transformation_texture_driver_map(deps.get_best_text(), deps.get_best_report())
    summary = {
        "enabled": True,
        "kind": texture_phase,
        "trigger_phase": trigger_phase,
        "base_strategy": deps.get_best_strategy(),
        "base_strict_safe_band": base_strict,
        "texture_driver_map": {
            "generic_sentence_ratio": driver_map.get("generic_sentence_ratio"),
            "generic_sentence_count": driver_map.get("generic_sentence_count"),
            "sentence_count": driver_map.get("sentence_count"),
            "authorship_drivers": driver_map.get("authorship_drivers"),
            "transformation_drivers": driver_map.get("transformation_drivers"),
            "top_blocks": [
                {
                    "paragraph_index": row.get("paragraph_index"),
                    "role": row.get("role"),
                    "generic_sentence_ratio": row.get("generic_sentence_ratio"),
                    "authorship_driver_score": row.get("authorship_driver_score"),
                    "transformation_driver_score": row.get("transformation_driver_score"),
                    "texture_driver_score": row.get("texture_driver_score"),
                    "low_value_generic_block": row.get("low_value_generic_block"),
                }
                for row in (driver_map.get("ranked_blocks") or [])[:6]
            ],
        },
        "candidate_count": 0,
        "scanned": 0,
        "selected": False,
        "selected_strategy": None,
        "scan_reserve_added": scan_reserve,
        "llm_reserve_added": llm_reserve,
    }
    summary["driver_map"] = summary["texture_driver_map"]
    search_summary[texture_phase] = summary
    search_summary["post_topk_optimizer"] = summary

    candidates: list[tuple[str, str, dict]] = []
    seen: set[str] = {str(deps.get_best_text() or "").strip()}

    def add_candidate(strategy: str, candidate_text: str, meta: dict | None = None) -> None:
        normalized = str(candidate_text or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append((strategy, normalized, meta or {}))

    texture_candidates = deps.authorship_transformation_texture_candidates(deps.get_best_text(), deps.get_best_report(), limit=12)
    summary["texture_candidate_count"] = len(texture_candidates)
    for strategy, candidate_text, meta in texture_candidates:
        add_candidate(strategy, candidate_text, {**meta, "post_topk_optimizer": True})
    if deps.env_flag("DRAFTPROOF_LEGACY_POST_TOPK_CANDIDATES", False):
        for strategy, candidate_text, meta in deps.generic_assertion_compiler_candidates(deps.get_best_text(), deps.get_best_report(), limit=2):
            add_candidate(
                f"legacy_post_topk_{strategy}",
                candidate_text,
                {**meta, "post_topk_optimizer": True, "legacy_post_topk_candidate": True},
            )
        for strategy, candidate_text, meta in deps.blocker_operation_candidates(deps.get_best_text(), deps.get_best_report(), limit=2):
            add_candidate(
                f"legacy_post_topk_{strategy}",
                candidate_text,
                {**meta, "post_topk_optimizer": True, "legacy_post_topk_candidate": True},
            )
        for strategy, candidate_text, meta in deps.content_pruning_candidates(deps.get_best_text(), deps.get_best_report(), limit=1):
            add_candidate(
                f"legacy_post_topk_{strategy}",
                candidate_text,
                {**meta, "post_topk_optimizer": True, "legacy_post_topk_candidate": True},
            )

    if (
        deps.env_flag("DRAFTPROOF_POST_TOPK_LLM_PATCHES", True)
        and int(search_summary.get("llm_calls") or 0) < int(search_budget.get("max_llm_calls") or 0)
    ):
        patch_candidates_total = 0
        patch_batches = max(1, int(deps.float_env("DRAFTPROOF_POST_TOPK_LLM_BATCHES", 3.0)))
        summary["llm_patch_batches_requested"] = patch_batches
        summary["llm_patch_batches_used"] = 0
        summary["llm_patch_candidate_count"] = 0
        for batch_index in range(1, patch_batches + 1):
            if int(search_summary.get("llm_calls") or 0) >= int(search_budget.get("max_llm_calls") or 0):
                summary["llm_patch_stop_reason"] = "total_llm_budget_exhausted"
                break
            if deps.phase_budget_block_record(texture_phase, summary):
                summary["llm_patch_stop_reason"] = "phase_llm_budget_exhausted"
                break
            try:
                search_summary["llm_calls"] += 1
                deps.record_phase_llm_call(texture_phase)
                response = gateway.chat(
                    deps.authorship_transformation_texture_patch_prompt(deps.get_best_text(), deps.get_best_report()),
                    system=(
                        "You are DraftProof's authorship/transformation texture controller. "
                        "Return only valid JSON paragraph patches."
                    ),
                    **deps.phase_chat_sampling_kwargs(
                        "DRAFTPROOF_POST_TOPK",
                        temperature_env="DRAFTPROOF_POST_TOPK_TEMPERATURE",
                        temperature_default=0.35,
                        max_tokens_env="DRAFTPROOF_POST_TOPK_MAX_TOKENS",
                        max_tokens_default=2200,
                    ),
                )
                summary["llm_patch_batches_used"] += 1
                patch_candidates = deps.extract_post_topk_patch_candidates(response.content, max_candidates=2)
                patch_candidates_total += len(patch_candidates)
                summary["llm_patch_candidate_count"] = patch_candidates_total
                for index, patch_candidate in enumerate(patch_candidates, start=1):
                    patched_text, applied = deps.apply_post_topk_patches(deps.get_best_text(), patch_candidate.get("patches") or [])
                    if applied and patched_text.strip() != deps.get_best_text().strip():
                        add_candidate(
                            f"texture_llm_patch_b{batch_index}_c{index}",
                            patched_text,
                            {
                                "post_topk_optimizer": True,
                                "authorship_transformation_texture_controller": True,
                                "texture_candidate_family": deps.texture_candidate_family(
                                    (applied[0] or {}).get("operation_type") if applied else None
                                ),
                                "llm_patch": True,
                                "llm_patch_batch": batch_index,
                                "llm_reason": patch_candidate.get("reason"),
                                "applied_post_topk_patches": applied,
                            },
                        )
            except Exception as exc:
                summary["llm_error"] = str(exc)
                break

    summary["candidate_count"] = len(candidates)
    base_protected = deps.detect_protected_spans(deps.get_best_text())
    base_review_burden = deps.review_burden(deps.get_best_report())
    base_severity = deps.weighted_severity(deps.get_best_report())
    base_finding_total = deps.finding_total(deps.get_best_report())
    base_critical_high = deps.critical_high_count(deps.get_best_report())
    base_formula_profile = deps.turnitin_like_ai_profile(deps.get_best_report())
    selected = None
    selected_rank = None
    partial_selected = None
    partial_selected_rank = None
    best_diagnostic = None
    best_diagnostic_rank = None

    def num(value, default=0.0) -> float:
        return float(value) if isinstance(value, (int, float)) else float(default)

    for strategy, candidate_text, meta in candidates[:max_scans]:
        if deps.search_budget_exhausted("post_topk_optimizer"):
            break
        candidate_eval = {
            "strategy": strategy,
            "deterministic": not bool(meta.get("llm_patch")),
            "post_topk_optimizer": True,
            "authorship_transformation_texture_controller": True,
            "passed_local_checks": False,
            **meta,
        }
        if deps.ai_candidate_quality_reject_reason(candidate_text):
            candidate_eval["reason"] = deps.ai_candidate_quality_reject_reason(candidate_text)
            search_summary["candidates"].append(candidate_eval)
            continue
        protected_loss = deps.ai_search_protected_loss_reason(deps.get_best_text(), candidate_text, base_protected)
        if protected_loss:
            candidate_eval["reason"] = "protected_span_lost " + protected_loss
            search_summary["candidates"].append(candidate_eval)
            continue
        drift = deps.check_semantic_drift(deps.get_best_text(), candidate_text, threshold=0.15)
        candidate_eval["drift_similarity"] = round(drift.similarity, 3)
        if not drift.accepted:
            candidate_eval["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
            candidate_eval["drift_reasons"] = drift.reasons[:10]
            search_summary["candidates"].append(candidate_eval)
            continue
        candidate_eval["passed_local_checks"] = True
        try:
            scan_t0 = time.time()
            candidate_report = deps.full_scan_report_dict(candidate_text)
            candidate_eval["scan_seconds"] = round(time.time() - scan_t0, 3)
            deps.record_verified_candidate_scan()
        except Exception as exc:
            candidate_eval["passed_local_checks"] = False
            candidate_eval["reason"] = f"candidate_scan_error {exc}"
            search_summary["candidates"].append(candidate_eval)
            continue

        after_strict = deps.strict_ai_safe_band_status(candidate_report)
        after_profile = after_strict.get("profile") or {}
        gate = deps.ai_footprint_gate_status(
            original_report_dict,
            candidate_report,
            review_burden_delta=deps.review_burden(candidate_report) - original_review_burden,
            weighted_severity_delta=deps.weighted_severity(candidate_report) - original_severity,
            critical_high_delta=deps.critical_high_count(candidate_report) - saved_critical_high,
            ai_score_regressed=False,
        )
        turnitin_like_gate = deps.turnitin_like_ai_gate_status(
            original_report_dict,
            candidate_report,
            review_burden_delta=deps.review_burden(candidate_report) - original_review_burden,
            weighted_severity_delta=deps.weighted_severity(candidate_report) - original_severity,
            critical_high_delta=deps.critical_high_count(candidate_report) - saved_critical_high,
            ai_score_regressed=False,
        )
        formula_gap_contract = deps.formula_gap_contract(
            original_report_dict,
            candidate_report,
            source_text=search_source_text,
            candidate_text=candidate_text,
        )
        candidate_eval.update({
            "ai": deps.badge_ai(candidate_report),
            "human_contribution": deps.contribution_scores(candidate_report).get("human"),
            "ai_transformation": deps.contribution_scores(candidate_report).get("ai_transformation"),
            "ai_authorship": deps.integrity_scores(candidate_report).get("ai_authorship"),
            "findings": deps.finding_total(candidate_report),
            "review_burden": deps.review_burden(candidate_report),
            "weighted_severity": deps.weighted_severity(candidate_report),
            "critical_high_findings": deps.critical_high_count(candidate_report),
            "strict_ai_safe_band": after_strict,
            "ai_footprint_gate": gate,
            "turnitin_like_ai_gate": turnitin_like_gate,
            "formula_gap_contract": formula_gap_contract,
            "formula_gap_rank": list(
                deps.formula_gap_candidate_rank(formula_gap_contract, turnitin_like_gate)
            ),
        })
        reject_reasons = []
        if num(formula_gap_contract.get("score_after"), 100.0) > num(base_formula_profile.get("score"), 100.0) + 0.001:
            reject_reasons.append("formula_score_regressed")
        if num(after_profile.get("topk_calibrated_risk"), 100.0) >= deps.safe_topk_calibrated_limit():
            reject_reasons.append("topk_calibrated_regressed_or_unsafe")
        for key in ("ai_authorship", "ai_transformation", "external_ai_flag_risk"):
            if num(after_profile.get(key)) > num(base_profile.get(key)) + 0.001:
                reject_reasons.append(f"{key}_regressed")
        if deps.finding_total(candidate_report) > base_finding_total:
            reject_reasons.append("findings_regressed")
        if deps.review_burden(candidate_report) > base_review_burden:
            reject_reasons.append("review_burden_regressed")
        if deps.weighted_severity(candidate_report) > base_severity:
            reject_reasons.append("weighted_severity_regressed")
        if deps.critical_high_count(candidate_report) > base_critical_high:
            reject_reasons.append("critical_high_regressed")
        if not after_strict.get("achieved"):
            reject_reasons.append("strict_safe_band_not_reached")
        if not formula_gap_contract.get("target_met"):
            reject_reasons.append("formula_gap_target_not_reached")

        candidate_eval["post_topk_reject_reasons"] = reject_reasons
        candidate_eval["selection_status"] = {
            "selectable": not reject_reasons,
            "success": not reject_reasons,
            "reason": "accepted_post_topk_strict_safe_band" if not reject_reasons else reject_reasons[0],
            "post_topk_optimizer": True,
            "authorship_transformation_texture_controller": True,
            "strict_ai_safe_band_achieved": bool(after_strict.get("achieved")),
            "ai_footprint_gate": gate,
            "turnitin_like_ai_gate": turnitin_like_gate,
            "formula_gap_contract": formula_gap_contract,
            "formula_gap_rank": candidate_eval["formula_gap_rank"],
            "ai_footprint_outcome_class": gate.get("outcome_class"),
            "topk_safe_band_achieved": True,
            "human_shift_score": deps.human_shift_score(
                original_report_dict,
                candidate_report,
                drift_similarity=candidate_eval.get("drift_similarity"),
                review_burden_delta=deps.review_burden(candidate_report) - original_review_burden,
                weighted_severity_delta=deps.weighted_severity(candidate_report) - original_severity,
            ).get("score"),
        }
        summary["scanned"] += 1
        search_summary["candidates"].append(candidate_eval)
        rank = (
            deps.formula_gap_candidate_rank(formula_gap_contract, turnitin_like_gate),
            deps.strict_safe_candidate_rank(
                deps.get_best_report(),
                candidate_report,
                review_burden_delta=deps.review_burden(candidate_report) - base_review_burden,
                weighted_severity_delta=deps.weighted_severity(candidate_report) - base_severity,
                critical_high_delta=deps.critical_high_count(candidate_report) - base_critical_high,
            ),
        )
        diagnostic_rank = (
            1 if formula_gap_contract.get("target_met") else 0,
            num(formula_gap_contract.get("score_drop"), 0.0),
            num(formula_gap_contract.get("weighted_driver_drop_efficiency"), 0.0),
            1 if num(after_profile.get("topk_calibrated_risk"), 100.0) < deps.safe_topk_calibrated_limit() else 0,
            num(base_profile.get("external_ai_flag_risk")) - num(after_profile.get("external_ai_flag_risk")),
            num(base_profile.get("ai_authorship")) - num(after_profile.get("ai_authorship")),
            num(base_profile.get("ai_transformation")) - num(after_profile.get("ai_transformation")),
            num(base_profile.get("generic_assertion_risk")) - num(after_profile.get("generic_assertion_risk")),
            num(base_profile.get("topk_calibrated_risk")) - num(after_profile.get("topk_calibrated_risk")),
            -len(reject_reasons),
            -num(deps.badge_ai(candidate_report), 999.0),
        )
        if best_diagnostic_rank is None or diagnostic_rank > best_diagnostic_rank:
            best_diagnostic_rank = diagnostic_rank
            best_diagnostic = {
                "strategy": strategy,
                "texture_candidate_family": meta.get("texture_candidate_family"),
                "rank": diagnostic_rank,
                "reject_reasons": reject_reasons,
                "strict_ai_safe_band": after_strict,
                "ai": candidate_eval.get("ai"),
                "human_contribution": candidate_eval.get("human_contribution"),
                "ai_authorship": candidate_eval.get("ai_authorship"),
                "ai_transformation": candidate_eval.get("ai_transformation"),
                "external_ai_flag_risk": after_profile.get("external_ai_flag_risk"),
                "generic_assertion_risk": after_profile.get("generic_assertion_risk"),
                "topk_calibrated_risk": after_profile.get("topk_calibrated_risk"),
                "formula_gap_contract": formula_gap_contract,
            }
        hard_reject_reasons = [
            reason for reason in reject_reasons
            if reason not in {"strict_safe_band_not_reached", "formula_gap_target_not_reached"}
        ]
        partial_driver_moved = bool(
            num(base_formula_profile.get("score")) - num(formula_gap_contract.get("score_after"), 100.0) > 0.25
            or num(base_profile.get("external_ai_flag_risk")) - num(after_profile.get("external_ai_flag_risk")) > 0.25
            or num(base_profile.get("ai_authorship")) - num(after_profile.get("ai_authorship")) > 0.25
            or num(base_profile.get("ai_transformation")) - num(after_profile.get("ai_transformation")) > 0.25
            or num(base_profile.get("generic_assertion_risk")) - num(after_profile.get("generic_assertion_risk")) > 0.25
        )
        if (
            not hard_reject_reasons
            and not after_strict.get("achieved")
            and partial_driver_moved
            and (partial_selected_rank is None or diagnostic_rank > partial_selected_rank)
        ):
            partial_selected_rank = diagnostic_rank
            partial_selected = {
                "strategy": strategy,
                "text": candidate_text,
                "report": candidate_report,
                "eval": candidate_eval,
                "rank": diagnostic_rank,
            }
        if not reject_reasons and (selected_rank is None or rank > selected_rank):
            selected_rank = rank
            selected = {
                "strategy": strategy,
                "text": candidate_text,
                "report": candidate_report,
                "eval": candidate_eval,
                "rank": rank,
            }

    if selected:
        selected_report = selected["report"]
        selected_ai = deps.badge_ai(selected_report)
        selected_status = selected["eval"].get("selection_status") or {}
        selected_status.update({
            "ai_footprint_mitigation": True,
            "partial_ai_footprint_mitigation": False,
            "authorship_transformation_texture_controller": True,
            "topk_safe_band_achieved": True,
            "strict_ai_safe_band_achieved": True,
        })
        deps.accept_selected_candidate(
            text=selected["text"],
            report=selected_report,
            ai=selected_ai,
            strategy=selected["strategy"],
            selection_status=selected_status,
            candidate_eval=selected["eval"],
            partial=False,
        )
        summary.update({
            "selected": True,
            "selected_strategy": selected["strategy"],
            "selected_ai": selected_ai,
            "selected_strict_ai_safe_band": deps.strict_ai_safe_band_status(selected_report),
        })
    elif partial_selected:
        partial_report = partial_selected["report"]
        partial_ai = deps.badge_ai(partial_report)
        partial_eval = partial_selected["eval"]
        partial_gate = partial_eval.get("ai_footprint_gate") or {}
        partial_status = {
            **(partial_eval.get("selection_status") or {}),
            "selectable": True,
            "success": True,
            "reason": "accepted_post_topk_partial_safe_band",
            "post_topk_optimizer": True,
            "authorship_transformation_texture_controller": True,
            "strict_ai_safe_band_achieved": False,
            "ai_footprint_mitigation": False,
            "partial_ai_footprint_mitigation": True,
            "topk_safe_band_achieved": True,
            "ai_footprint_gate": partial_gate,
            "ai_footprint_outcome_class": partial_gate.get("outcome_class") or "partially_ai_mitigated",
        }
        deps.accept_selected_candidate(
            text=partial_selected["text"],
            report=partial_report,
            ai=partial_ai,
            strategy=partial_selected["strategy"],
            selection_status=partial_status,
            candidate_eval=partial_eval,
            partial=True,
        )
        summary.update({
            "selected": True,
            "selected_partial": True,
            "selected_strategy": partial_selected["strategy"],
            "selected_ai": partial_ai,
            "selected_strict_ai_safe_band": deps.strict_ai_safe_band_status(partial_report),
            "reason": "accepted_best_non_regressing_partial_candidate",
        })
    else:
        summary["reason"] = "no_candidate_reached_strict_safe_band"
        summary["best_preserved_strategy"] = deps.get_best_strategy()
        summary["best_rejected_candidate"] = best_diagnostic
        summary["remaining_strict_safe_band_drivers"] = base_strict.get("remaining") or []
