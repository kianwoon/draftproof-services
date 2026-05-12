"""Post-selection AI-density breaker phase."""

from __future__ import annotations

import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable

from rewrite_pipeline_core.gates.ai_density_breaker import (
    _AI_DENSITY_GENERIC_RE,
    _AI_DENSITY_TRANSITION_RE,
)
from rewrite_pipeline_core.scoring.profiles import (
    _ai_footprint_flatten,
    _ai_footprint_profile,
    _turnitin_like_ai_profile,
    _turnitin_like_component_drops,
)


@dataclass(frozen=True)
class PostSelectionDensityDeps:
    env_flag: Callable[..., bool]
    float_env: Callable[[str, float], float]
    split_sentences: Callable[[str], list[str]]
    text_word_count: Callable[[str], int]
    logical_paragraphs: Callable[[str], list[str]]
    join_logical_paragraphs: Callable[[list[str]], str]
    ai_density_breaker_map: Callable[[str, dict | None], dict]
    ai_density_edit_budget: Callable[[str], dict]
    ai_density_breaker_sentence_route: Callable[[str], tuple[str, list]]
    ai_density_breaker_canonical_fact_sentence: Callable[[str], bool]
    splice_sentences_by_text: Callable[[str, dict[str, str]], str]
    remove_sentences_by_text: Callable[[str, list[str]], str]
    compress_score_drag_paragraph: Callable[..., str]
    run_full_scan_report_dict: Callable[[str], dict]
    check_semantic_drift: Callable[..., Any]
    detect_protected_spans: Callable[[str], Any]
    ai_candidate_quality_reject_reason: Callable[[str], str]
    ai_search_protected_loss_reason: Callable[..., str]
    strict_ai_safe_band_status: Callable[[dict | None], dict]
    report_review_burden: Callable[[dict | None], int | float]
    report_weighted_severity: Callable[[dict | None], int | float]
    critical_high_count: Callable[[dict | None], int | float]


def ai_density_patchwork_budget_status(
    source_text: str,
    candidate_text: str,
    meta: dict | None = None,
    *,
    deps: PostSelectionDensityDeps,
) -> dict:
    budget = deps.ai_density_edit_budget(source_text)
    meta = meta if isinstance(meta, dict) else {}
    edited_count = meta.get("edited_sentence_count")
    if not isinstance(edited_count, (int, float)):
        source_sentences = deps.split_sentences(source_text)
        candidate_sentences = deps.split_sentences(candidate_text)
        matcher = SequenceMatcher(None, source_sentences, candidate_sentences, autojunk=False)
        edited_count = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            edited_count += max(i2 - i1, j2 - j1)
    edited_count = int(max(0, edited_count or 0))
    sentence_count = int(budget.get("sentence_count") or 0)
    ratio = edited_count / max(1, sentence_count)
    accepted = bool(
        edited_count <= int(budget.get("max_edited_sentences") or 0)
        and ratio <= float(budget.get("max_edited_sentence_ratio") or 1.0)
    )
    return {
        "version": "ai_density_patchwork_budget_v1",
        "accepted": accepted,
        "edited_sentence_count": edited_count,
        "edited_sentence_ratio": round(ratio, 3),
        "budget": budget,
        "reason": "within_patchwork_budget" if accepted else "patchwork_edit_budget_exceeded",
    }


def turnitin_like_positive_burden_drop(before_report: dict | None, after_report: dict | None) -> float:
    before = _turnitin_like_ai_profile(before_report)
    after = _turnitin_like_ai_profile(after_report)
    return round(float(before.get("raw_positive_score") or 0.0) - float(after.get("raw_positive_score") or 0.0), 3)


def post_selection_ai_density_breaker_candidates(
    current_text: str,
    current_report: dict | None,
    *,
    limit: int = 8,
    deps: PostSelectionDensityDeps,
) -> list[tuple[str, str, dict]]:
    """Build bounded post-selection candidates without touching rewrite core."""
    density_map = deps.ai_density_breaker_map(current_text, current_report)
    paragraphs = deps.logical_paragraphs(current_text)
    candidates: list[tuple[str, str, dict]] = []
    seen = {str(current_text or "").strip()}
    limit = max(1, int(limit or 1))
    edit_budget = deps.ai_density_edit_budget(current_text)
    max_edits = max(1, int(edit_budget.get("max_edited_sentences") or 1))

    def patchwork(candidate: str, meta: dict | None = None) -> dict:
        return ai_density_patchwork_budget_status(current_text, candidate, meta, deps=deps)

    def add(strategy: str, candidate: str, meta: dict) -> None:
        if len(candidates) >= limit:
            return
        normalized = str(candidate or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append((strategy, normalized, {**meta, "post_selection_ai_density_breaker_candidate": True}))

    window_targets = density_map.get("top_density_windows") or []
    for take_windows in (1, 2, 3):
        replacements: dict[str, str] = {}
        operations = []
        edited_indexes: set[int] = set()
        for window in window_targets[:take_windows]:
            editable_rows = sorted(
                [row for row in (window.get("editable_sentences") or []) if isinstance(row, dict)],
                key=lambda row: int(row.get("sentence_index") if isinstance(row.get("sentence_index"), int) else 10**9),
            )
            for row in editable_rows:
                if len(edited_indexes) >= max_edits:
                    break
                sentence_index = row.get("sentence_index")
                if not isinstance(sentence_index, int) or sentence_index in edited_indexes:
                    continue
                original = str(row.get("sentence") or "")
                replacement, ops = deps.ai_density_breaker_sentence_route(original)
                if ops and replacement != original:
                    replacements[original] = replacement
                    edited_indexes.add(sentence_index)
                    operations.append({
                        "sentence_index": sentence_index,
                        "operations": ops,
                        "score": row.get("score"),
                        "top10_ratio": row.get("top10_ratio"),
                        "window": {
                            "start_sentence": window.get("start_sentence"),
                            "end_sentence": window.get("end_sentence"),
                            "score": window.get("score"),
                        },
                    })
            if len(edited_indexes) >= max_edits:
                break
        if operations:
            candidate_text = deps.splice_sentences_by_text(current_text, replacements)
            add(
                f"density_window_patch_top{take_windows}",
                candidate_text,
                {
                    "operation": "coordinated_density_window_patch",
                    "edited_sentence_count": len(operations),
                    "target_window_count": take_windows,
                    "operations": operations,
                    "edit_budget": edit_budget,
                    "patchwork_budget": patchwork(candidate_text, {"edited_sentence_count": len(operations)}),
                    "density_breaker_map": {
                        "version": density_map.get("version"),
                        "top_density_windows": window_targets[:take_windows],
                    },
                },
            )

    sentence_targets = [
        row for row in density_map.get("top_sentence_targets") or []
        if not row.get("canonical_fact_preserved")
    ]
    if not candidates and deps.env_flag("DRAFTPROOF_DENSITY_BREAKER_SCATTER_FALLBACK", False):
        replacements: dict[str, str] = {}
        operations = []
        for row in sentence_targets[: min(2, max_edits)]:
            original = str(row.get("sentence") or "")
            replacement, ops = deps.ai_density_breaker_sentence_route(original)
            if ops and replacement != original:
                replacements[original] = replacement
                operations.append({
                    "sentence_index": row.get("sentence_index"),
                    "operations": ops,
                    "score": row.get("score"),
                    "top10_ratio": row.get("top10_ratio"),
                })
        if operations:
            candidate_text = deps.splice_sentences_by_text(current_text, replacements)
            add(
                "density_route_patch_top2_fallback",
                candidate_text,
                {
                    "operation": "scatter_density_route_fallback",
                    "edited_sentence_count": len(operations),
                    "operations": operations,
                    "edit_budget": edit_budget,
                    "patchwork_budget": patchwork(candidate_text, {"edited_sentence_count": len(operations)}),
                    "density_breaker_map": {
                        "version": density_map.get("version"),
                        "top_runs": density_map.get("contiguous_ai_density_runs"),
                    },
                },
            )

    for paragraph_row in (density_map.get("top_generic_paragraphs") or [])[:4]:
        index = paragraph_row.get("paragraph_index")
        if not isinstance(index, int) or index < 0 or index >= len(paragraphs):
            continue
        if paragraph_row.get("protected") or float(paragraph_row.get("score") or 0.0) <= 2.5:
            continue
        replacement = deps.compress_score_drag_paragraph(paragraphs[index], max_remove=1)
        if replacement.strip() and replacement.strip() != paragraphs[index].strip():
            next_paragraphs = list(paragraphs)
            next_paragraphs[index] = replacement
            candidate_text = deps.join_logical_paragraphs(next_paragraphs)
            add(
                f"generic_block_compress_p{index + 1}",
                candidate_text,
                {
                    "operation": "generic_block_compress",
                    "paragraph_index": index,
                    "paragraph_score": paragraph_row.get("score"),
                    "edited_sentence_count": 1,
                    "edit_budget": edit_budget,
                    "patchwork_budget": patchwork(candidate_text, {"edited_sentence_count": 1}),
                    "target_preview": paragraph_row.get("preview"),
                },
            )

    removable = []
    for row in sentence_targets:
        sentence = str(row.get("sentence") or "")
        if deps.ai_density_breaker_canonical_fact_sentence(sentence):
            continue
        if len(removable) >= 4:
            break
        if (
            _AI_DENSITY_TRANSITION_RE.search(sentence.strip())
            or len(_AI_DENSITY_GENERIC_RE.findall(sentence)) >= 2
        ) and deps.text_word_count(sentence) <= 28:
            removable.append(row)
    for take in (1, 2):
        selected = removable[:take]
        if not selected:
            continue
        candidate = deps.remove_sentences_by_text(
            current_text,
            [str(row.get("sentence") or "") for row in selected],
        )
        add(
            f"low_value_generic_sentence_remove_{take}",
            candidate,
            {
                "operation": "low_value_generic_remove",
                "removed_sentence_count": len(selected),
                "removed_sentences": [str(row.get("sentence") or "")[:220] for row in selected],
                "edited_sentence_count": len(selected),
                "edit_budget": edit_budget,
                "patchwork_budget": patchwork(candidate, {"edited_sentence_count": len(selected)}),
            },
        )

    return candidates[:limit]


def post_selection_ai_density_breaker_acceptance(
    current_report: dict | None,
    candidate_report: dict | None,
    *,
    review_burden_delta: int | float,
    weighted_severity_delta: int | float,
    critical_high_delta: int | float,
    float_env: Callable[[str, float], float],
) -> dict:
    """Strict local acceptance for the isolated density breaker layer."""
    base_profile = _turnitin_like_ai_profile(current_report)
    candidate_profile = _turnitin_like_ai_profile(candidate_report)
    formula_drop = round(float(base_profile.get("score") or 0.0) - float(candidate_profile.get("score") or 0.0), 3)
    base_flat = _ai_footprint_flatten(_ai_footprint_profile(current_report))
    candidate_flat = _ai_footprint_flatten(_ai_footprint_profile(candidate_report))

    def drop(key: str) -> float:
        return round(float(base_flat.get(key) or 0.0) - float(candidate_flat.get(key) or 0.0), 3)

    drops = {
        key: drop(key)
        for key in (
            "topk_calibrated_risk",
            "topk_pattern_raw",
            "qualifying_text_ai_density",
            "external_ai_flag_risk",
            "ai_likelihood",
            "rewrite_smoothness",
            "ai_authorship",
            "ai_transformation",
            "semantic_uniformity",
        )
    }
    component_drops = _turnitin_like_component_drops(base_profile, candidate_profile)
    positive_ai_burden_drop = turnitin_like_positive_burden_drop(current_report, candidate_report)
    smoothness_max_regression = float_env("DRAFTPROOF_DENSITY_BREAKER_MAX_SMOOTHNESS_REGRESSION", 0.30)
    improvement_epsilon = 0.001
    severe_smoothness_regression = float_env("DRAFTPROOF_DENSITY_BREAKER_SEVERE_SMOOTHNESS_REGRESSION", 4.0)
    severe_semantic_regression = float_env("DRAFTPROOF_DENSITY_BREAKER_SEVERE_SEMANTIC_REGRESSION", 3.0)
    severe_density_regression = float_env("DRAFTPROOF_DENSITY_BREAKER_SEVERE_DENSITY_REGRESSION", 3.0)
    reject_reasons = []
    slippage_notes = []
    protected_regressions = []
    core_progress_ok = bool(
        formula_drop > improvement_epsilon
        and positive_ai_burden_drop >= -improvement_epsilon
        and drops["external_ai_flag_risk"] >= -improvement_epsilon
        and drops["ai_likelihood"] >= -improvement_epsilon
        and float(review_burden_delta or 0.0) <= 0.0
        and float(weighted_severity_delta or 0.0) <= 0.0
        and float(critical_high_delta or 0.0) <= 0.0
    )
    if formula_drop <= improvement_epsilon:
        reject_reasons.append("formula_score_not_reduced")
    if positive_ai_burden_drop < -improvement_epsilon:
        reject_reasons.append("positive_ai_burden_regressed")
    for key in ("topk_calibrated_risk", "ai_authorship", "ai_transformation", "external_ai_flag_risk", "ai_likelihood"):
        if drops[key] < -0.001:
            protected_regressions.append(f"{key}_regressed")
    reject_reasons.extend(protected_regressions)
    if drops["rewrite_smoothness"] < -smoothness_max_regression:
        if drops["rewrite_smoothness"] < -severe_smoothness_regression or not core_progress_ok or protected_regressions:
            reject_reasons.append("rewrite_smoothness_regressed")
        else:
            slippage_notes.append("rewrite_smoothness_regressed_but_total_formula_improved")
    if drops["semantic_uniformity"] < -severe_semantic_regression:
        reject_reasons.append("semantic_uniformity_severely_regressed")
    elif drops["semantic_uniformity"] < -0.001:
        slippage_notes.append("semantic_uniformity_regressed_but_total_formula_improved")
    if drops["qualifying_text_ai_density"] < -severe_density_regression:
        reject_reasons.append("qualifying_text_ai_density_severely_regressed")
    elif drops["qualifying_text_ai_density"] < -0.001:
        slippage_notes.append("qualifying_text_ai_density_regressed_but_total_formula_improved")
    if float(review_burden_delta or 0.0) > 0.0:
        reject_reasons.append("review_burden_regressed")
    if float(weighted_severity_delta or 0.0) > 0.0:
        reject_reasons.append("weighted_severity_regressed")
    if float(critical_high_delta or 0.0) > 0.0:
        reject_reasons.append("critical_high_regressed")
    selectable = not reject_reasons
    return {
        "version": "post_selection_ai_density_breaker_acceptance_v1",
        "selectable": selectable,
        "reason": "accepted_density_breaker_improvement" if selectable else reject_reasons[0],
        "formula_score_before": base_profile.get("score"),
        "formula_score_after": candidate_profile.get("score"),
        "formula_score_drop": formula_drop,
        "driver_drops": drops,
        "turnitin_like_component_drops": component_drops,
        "component_slippage": {
            key: round(abs(value), 3)
            for key, value in drops.items()
            if isinstance(value, (int, float)) and value < -0.001
        },
        "component_slippage_accepted": bool(selectable and slippage_notes),
        "component_slippage_notes": slippage_notes,
        "positive_ai_burden_before": base_profile.get("raw_positive_score"),
        "positive_ai_burden_after": candidate_profile.get("raw_positive_score"),
        "positive_ai_burden_drop": positive_ai_burden_drop,
        "review_burden_delta": review_burden_delta,
        "weighted_severity_delta": weighted_severity_delta,
        "critical_high_delta": critical_high_delta,
        "thresholds": {
            "improvement_epsilon": improvement_epsilon,
            "max_smoothness_regression": smoothness_max_regression,
            "severe_smoothness_regression": severe_smoothness_regression,
            "severe_semantic_regression": severe_semantic_regression,
            "severe_density_regression": severe_density_regression,
        },
    }


def run_post_selection_ai_density_breaker(
    current_text: str,
    current_report: dict | None,
    original_report: dict | None,
    *,
    scan_func=None,
    drift_checker=None,
    max_scans: int | None = None,
    deps: PostSelectionDensityDeps,
) -> dict:
    """Run an isolated post-selection AI-density breaker."""
    if not deps.env_flag("DRAFTPROOF_POST_SELECTION_AI_DENSITY_BREAKER", True):
        return {"enabled": False, "reason": "disabled"}
    if not isinstance(current_text, str) or not current_text.strip() or not isinstance(current_report, dict):
        return {"enabled": False, "reason": "missing_current_selection"}
    scan_func = scan_func or deps.run_full_scan_report_dict
    drift_checker = drift_checker or deps.check_semantic_drift
    max_scans = max(
        0,
        int(max_scans if isinstance(max_scans, int) else deps.float_env("DRAFTPROOF_DENSITY_BREAKER_MAX_SCANS", 3.0)),
    )
    if max_scans <= 0:
        return {"enabled": True, "selected": False, "reason": "scan_budget_zero"}
    candidates = post_selection_ai_density_breaker_candidates(
        current_text,
        current_report,
        limit=max_scans,
        deps=deps,
    )
    density_map = deps.ai_density_breaker_map(current_text, current_report)
    summary = {
        "enabled": True,
        "version": "post_selection_ai_density_breaker_v1",
        "selected": False,
        "selected_text": current_text,
        "selected_report": current_report,
        "selected_strategy": None,
        "candidate_count": len(candidates),
        "scans_used": 0,
        "density_breaker_map": density_map,
        "edit_budget": deps.ai_density_edit_budget(current_text),
        "base_summary": {
            "turnitin_like_ai_score": _turnitin_like_ai_profile(current_report).get("score"),
            "ai_footprint": deps.strict_ai_safe_band_status(current_report).get("profile"),
        },
        "candidates": [],
        "reason": "no_candidate_selected",
    }
    if not candidates:
        summary["reason"] = "no_density_breaker_candidates"
        return summary

    protected = deps.detect_protected_spans(current_text)
    best_eval = None
    best_rank = None
    best_text = current_text
    best_report = current_report
    seen = {current_text.strip()}
    for strategy, candidate_text, meta in candidates:
        if summary["scans_used"] >= max_scans:
            break
        candidate_eval = {
            "strategy": strategy,
            "operation": (meta or {}).get("operation"),
            "meta": meta or {},
        }
        normalized = str(candidate_text or "").strip()
        if not normalized or normalized in seen:
            candidate_eval["reason"] = "unchanged_or_duplicate_candidate"
            summary["candidates"].append(candidate_eval)
            continue
        seen.add(normalized)
        local_reason = deps.ai_candidate_quality_reject_reason(normalized)
        if local_reason:
            candidate_eval["reason"] = local_reason
            summary["candidates"].append(candidate_eval)
            continue
        budget_status = ai_density_patchwork_budget_status(current_text, normalized, meta or {}, deps=deps)
        candidate_eval["patchwork_budget"] = budget_status
        if not budget_status.get("accepted"):
            candidate_eval["reason"] = budget_status.get("reason") or "patchwork_edit_budget_exceeded"
            summary["candidates"].append(candidate_eval)
            continue
        protected_loss = deps.ai_search_protected_loss_reason(current_text, normalized, protected)
        if protected_loss:
            candidate_eval["reason"] = "protected_span_lost " + protected_loss
            summary["candidates"].append(candidate_eval)
            continue
        try:
            drift = drift_checker(current_text, normalized, threshold=0.80)
        except TypeError:
            drift = drift_checker(current_text, normalized)
        candidate_eval["drift_similarity"] = round(float(getattr(drift, "similarity", 1.0)), 3)
        if not bool(getattr(drift, "accepted", True)):
            candidate_eval["reason"] = "semantic_drift " + "; ".join(list(getattr(drift, "reasons", []) or [])[:3])
            candidate_eval["drift_reasons"] = list(getattr(drift, "reasons", []) or [])[:10]
            summary["candidates"].append(candidate_eval)
            continue
        scan_t0 = time.time()
        try:
            candidate_report = scan_func(normalized)
        except Exception as exc:
            candidate_eval["reason"] = f"candidate_scan_error {exc}"
            summary["candidates"].append(candidate_eval)
            continue
        summary["scans_used"] += 1
        candidate_eval["scan_seconds"] = round(time.time() - scan_t0, 3)
        review_delta = deps.report_review_burden(candidate_report) - deps.report_review_burden(current_report)
        severity_delta = deps.report_weighted_severity(candidate_report) - deps.report_weighted_severity(current_report)
        critical_delta = deps.critical_high_count(candidate_report) - deps.critical_high_count(current_report)
        acceptance = post_selection_ai_density_breaker_acceptance(
            current_report,
            candidate_report,
            review_burden_delta=review_delta,
            weighted_severity_delta=severity_delta,
            critical_high_delta=critical_delta,
            float_env=deps.float_env,
        )
        candidate_eval.update({
            "acceptance": acceptance,
            "selectable": acceptance.get("selectable"),
            "reason": acceptance.get("reason"),
            "formula_score": acceptance.get("formula_score_after"),
            "formula_score_drop": acceptance.get("formula_score_drop"),
            "driver_drops": acceptance.get("driver_drops"),
            "strict_ai_safe_band": deps.strict_ai_safe_band_status(candidate_report),
        })
        summary["candidates"].append(candidate_eval)
        if not acceptance.get("selectable"):
            continue
        rank = (
            1 if _turnitin_like_ai_profile(candidate_report).get("target_met") else 0,
            float(acceptance.get("formula_score_drop") or 0.0),
            float(acceptance.get("positive_ai_burden_drop") or 0.0),
            float((acceptance.get("driver_drops") or {}).get("external_ai_flag_risk") or 0.0),
            float((acceptance.get("driver_drops") or {}).get("qualifying_text_ai_density") or 0.0),
            float((acceptance.get("driver_drops") or {}).get("topk_calibrated_risk") or 0.0),
            float((acceptance.get("driver_drops") or {}).get("ai_likelihood") or 0.0),
            -float(_turnitin_like_ai_profile(candidate_report).get("score") or 100.0),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_eval = candidate_eval
            best_text = normalized
            best_report = candidate_report

    if best_eval:
        best_eval["selected"] = True
        summary.update({
            "selected": True,
            "selected_text": best_text,
            "selected_report": best_report,
            "selected_strategy": best_eval.get("strategy"),
            "reason": best_eval.get("reason"),
            "selected_candidate": {
                key: best_eval.get(key)
                for key in (
                    "strategy",
                    "operation",
                    "formula_score",
                    "formula_score_drop",
                    "driver_drops",
                    "positive_ai_burden_drop",
                    "positive_ai_burden_before",
                    "positive_ai_burden_after",
                    "patchwork_budget",
                    "drift_similarity",
                    "acceptance",
                )
            },
            "final_summary": {
                "turnitin_like_ai_score": _turnitin_like_ai_profile(best_report).get("score"),
                "ai_footprint": deps.strict_ai_safe_band_status(best_report).get("profile"),
            },
        })
    return summary
