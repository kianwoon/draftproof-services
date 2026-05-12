"""Post-density Human Anchor probe phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import re
import time

from rewrite_pipeline_core.scoring.profiles import (
    _ai_footprint_flatten,
    _ai_footprint_profile,
    _turnitin_like_ai_profile,
    _turnitin_like_component_drops,
)


@dataclass(frozen=True)
class PostDensityHumanAnchorProbeDeps:
    env_flag: Callable[..., bool]
    float_env: Callable[[str, float], float]
    run_full_scan_report_dict: Callable[[str], dict]
    check_semantic_drift: Callable[..., Any]
    human_anchor_driver_contract: Callable[..., dict]
    human_anchor_suppression_frontier: Callable[..., dict]
    ai_density_breaker_map: Callable[..., dict]
    logical_paragraphs: Callable[[str], list[str]]
    split_sentences: Callable[[str], list[str]]
    ai_density_breaker_canonical_fact_sentence: Callable[[str], bool]
    post_density_human_anchor_probe_context: Callable[[str, str], tuple[str, str]]
    ai_density_edit_budget: Callable[[str], dict]
    splice_sentences_by_text: Callable[[str, dict[str, str]], str]
    ai_density_patchwork_budget_status: Callable[..., dict]
    detect_protected_spans: Callable[[str], Any]
    ai_candidate_quality_reject_reason: Callable[[str], str]
    ai_search_protected_loss_reason: Callable[..., str]
    report_review_burden: Callable[[dict | None], int | float]
    report_weighted_severity: Callable[[dict | None], int | float]
    critical_high_count: Callable[[dict | None], int | float]
    strict_ai_safe_band_status: Callable[[dict | None], dict]
    turnitin_like_positive_burden_drop: Callable[[dict | None, dict | None], float]


def build_post_density_human_anchor_probe_candidates(
    current_text: str,
    current_report: dict | None,
    *,
    limit: int = 3,
    deps: PostDensityHumanAnchorProbeDeps | None = None,
) -> list[tuple[str, str, dict]]:
    """Build a tiny post-density Human Anchor probe candidate set.

    This is intentionally separate from the existing amplifier. It only runs
    after density work has selected a safer baseline, and it edits very few
    non-canonical spans so patchwork risk stays bounded.
    """
    if deps is None:
        raise ValueError("PostDensityHumanAnchorProbeDeps is required")
    if not deps.env_flag("DRAFTPROOF_POST_DENSITY_HUMAN_ANCHOR_PROBE", True):
        return []
    if not isinstance(current_text, str) or not current_text.strip() or not isinstance(current_report, dict):
        return []
    profile = _turnitin_like_ai_profile(current_report)
    contract = deps.human_anchor_driver_contract(current_report, text=current_text)
    before = contract.get("before") if isinstance(contract.get("before"), dict) else {}
    try:
        suppression = float(profile.get("human_anchor_suppression") or 0.0)
    except (TypeError, ValueError):
        suppression = 0.0
    try:
        target_gap = float(profile.get("target_gap") or 0.0)
    except (TypeError, ValueError):
        target_gap = 0.0
    try:
        lived_detail_risk = float(before.get("lived_detail_risk") or 0.0)
    except (TypeError, ValueError):
        lived_detail_risk = 0.0
    try:
        human_anchor_score = float(before.get("human_anchor_score") or 0.0)
    except (TypeError, ValueError):
        human_anchor_score = 0.0
    headroom = max(0.0, 45.0 - suppression)
    if target_gap <= 0.0 or headroom <= 0.0:
        return []
    if lived_detail_risk < 55.0 and human_anchor_score >= 50.0:
        return []

    density_map = deps.ai_density_breaker_map(current_text, current_report)
    paragraphs = deps.logical_paragraphs(current_text)
    paragraph_lookup = {}
    flat_index = 0
    for paragraph_index, paragraph in enumerate(paragraphs):
        for sentence in deps.split_sentences(paragraph):
            paragraph_lookup[flat_index] = {
                "paragraph_index": paragraph_index,
                "paragraph": paragraph,
                "sentence": sentence,
            }
            flat_index += 1

    target_rows: list[dict] = []
    seen_indexes: set[int] = set()
    for window in density_map.get("top_density_windows") or []:
        for row in window.get("editable_sentences") or []:
            if not isinstance(row, dict):
                continue
            sentence_index = row.get("sentence_index")
            if not isinstance(sentence_index, int) or sentence_index in seen_indexes:
                continue
            sentence = str(row.get("sentence") or "")
            if (
                not sentence.strip()
                or deps.ai_density_breaker_canonical_fact_sentence(sentence)
                or re.search(r"https?://|www\.|^\s*references?\s*$", sentence, flags=re.I)
            ):
                continue
            context_text, operation = deps.post_density_human_anchor_probe_context(
                sentence,
                str((paragraph_lookup.get(sentence_index) or {}).get("paragraph") or ""),
            )
            if not context_text:
                continue
            target_rows.append({
                **row,
                "sentence_index": sentence_index,
                "sentence": sentence,
                "context_text": context_text,
                "context_operation": operation,
                "window": {
                    "start_sentence": window.get("start_sentence"),
                    "end_sentence": window.get("end_sentence"),
                    "score": window.get("score"),
                },
            })
            seen_indexes.add(sentence_index)

    if not target_rows:
        for row in density_map.get("top_sentence_targets") or []:
            if not isinstance(row, dict):
                continue
            sentence_index = row.get("sentence_index")
            sentence = str(row.get("sentence") or "")
            if (
                not isinstance(sentence_index, int)
                or sentence_index in seen_indexes
                or row.get("canonical_fact_preserved")
                or deps.ai_density_breaker_canonical_fact_sentence(sentence)
            ):
                continue
            context_text, operation = deps.post_density_human_anchor_probe_context(
                sentence,
                str((paragraph_lookup.get(sentence_index) or {}).get("paragraph") or ""),
            )
            target_rows.append({
                **row,
                "sentence_index": sentence_index,
                "sentence": sentence,
                "context_text": context_text,
                "context_operation": operation,
            })
            seen_indexes.add(sentence_index)
            if len(target_rows) >= 4:
                break

    if not target_rows:
        return []
    target_rows.sort(
        key=lambda row: (
            float(row.get("score") or 0.0),
            float(row.get("top10_ratio") or 0.0),
        ),
        reverse=True,
    )

    edit_budget = deps.ai_density_edit_budget(current_text)
    max_edits = max(1, min(4, int(edit_budget.get("max_edited_sentences") or 1)))
    max_probe_targets = max(1, min(3, max_edits // 2 if max_edits >= 2 else 1))
    profiles = [
        ("post_density_anchor_probe_one_spot", 1),
        ("post_density_anchor_probe_two_spots", min(2, max_probe_targets)),
        ("post_density_anchor_probe_three_spots", min(3, max_probe_targets)),
    ]
    candidates: list[tuple[str, str, dict]] = []
    seen_texts = {current_text.strip()}

    for strategy, take in profiles[:max(1, int(limit or 1))]:
        if take <= 0:
            continue
        selected = sorted(target_rows[:take], key=lambda row: int(row.get("sentence_index") or 0))
        replacements: dict[str, str] = {}
        changes = []
        for row in selected:
            sentence = str(row.get("sentence") or "")
            addition = str(row.get("context_text") or "").strip()
            if not sentence.strip() or not addition:
                continue
            replacement = f"{sentence.strip()} {addition}"
            if replacement.strip() == sentence.strip():
                continue
            replacements[sentence] = replacement
            changes.append({
                "sentence_index": row.get("sentence_index"),
                "operation": row.get("context_operation"),
                "original": sentence[:200],
                "addition": addition[:200],
                "score": row.get("score"),
                "top10_ratio": row.get("top10_ratio"),
                "window": row.get("window"),
            })
        if not changes:
            continue
        candidate = deps.splice_sentences_by_text(current_text, replacements).strip()
        if not candidate or candidate in seen_texts:
            continue
        seen_texts.add(candidate)
        meta = {
            "operation": "post_density_human_anchor_probe",
            "post_density_human_anchor_probe_candidate": True,
            "scope": "bounded_implied_context_after_density",
            "edited_sentence_count": len(changes),
            "changed_sentence_frames": len(changes),
            "edit_budget": edit_budget,
            "patchwork_budget": deps.ai_density_patchwork_budget_status(
                current_text,
                candidate,
                {"edited_sentence_count": len(changes)},
            ),
            "human_anchor_driver_contract_before": contract,
            "targeted_drivers": [
                "human_anchor_suppression",
                "lived_detail_risk",
                "turnitin_like_ai_score",
            ],
            "changes": changes,
        }
        candidates.append((strategy, candidate, meta))
        if len(candidates) >= max(1, int(limit or 1)):
            break
    return candidates

def post_density_human_anchor_probe_acceptance(
    current_report: dict | None,
    candidate_report: dict | None,
    *,
    review_burden_delta: int | float,
    weighted_severity_delta: int | float,
    critical_high_delta: int | float,
    deps: PostDensityHumanAnchorProbeDeps | None = None,
) -> dict:
    """Accept only measured Human Anchor suppression that does not backfire."""
    if deps is None:
        raise ValueError("PostDensityHumanAnchorProbeDeps is required")
    base_profile = _turnitin_like_ai_profile(current_report)
    candidate_profile = _turnitin_like_ai_profile(candidate_report)
    formula_drop = round(float(base_profile.get("score") or 0.0) - float(candidate_profile.get("score") or 0.0), 3)
    positive_ai_burden_drop = deps.turnitin_like_positive_burden_drop(current_report, candidate_report)
    component_drops = _turnitin_like_component_drops(base_profile, candidate_profile)
    base_flat = _ai_footprint_flatten(_ai_footprint_profile(current_report))
    candidate_flat = _ai_footprint_flatten(_ai_footprint_profile(candidate_report))

    def drop(key: str) -> float:
        return round(float(base_flat.get(key) or 0.0) - float(candidate_flat.get(key) or 0.0), 3)

    driver_drops = {
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
    human_anchor_gain = round(float(component_drops.get("human_anchor_suppression") or 0.0), 3)
    reject_reasons = []
    if formula_drop <= 0.001:
        reject_reasons.append("formula_score_not_reduced")
    if human_anchor_gain <= 0.001:
        reject_reasons.append("human_anchor_suppression_not_increased")
    if positive_ai_burden_drop < -0.001:
        reject_reasons.append("positive_ai_burden_regressed")
    for key in ("topk_calibrated_risk", "ai_authorship", "ai_transformation", "external_ai_flag_risk"):
        if driver_drops.get(key, 0.0) < -0.001:
            reject_reasons.append(f"{key}_regressed")
    if float(review_burden_delta or 0.0) > 0.0:
        reject_reasons.append("review_burden_regressed")
    if float(weighted_severity_delta or 0.0) > 0.0:
        reject_reasons.append("weighted_severity_regressed")
    if float(critical_high_delta or 0.0) > 0.0:
        reject_reasons.append("critical_high_regressed")
    selectable = not reject_reasons
    return {
        "version": "post_density_human_anchor_probe_acceptance_v1",
        "selectable": selectable,
        "reason": "accepted_post_density_human_anchor_gain" if selectable else reject_reasons[0],
        "formula_score_before": base_profile.get("score"),
        "formula_score_after": candidate_profile.get("score"),
        "formula_score_drop": formula_drop,
        "positive_ai_burden_before": base_profile.get("raw_positive_score"),
        "positive_ai_burden_after": candidate_profile.get("raw_positive_score"),
        "positive_ai_burden_drop": positive_ai_burden_drop,
        "human_anchor_suppression_before": base_profile.get("human_anchor_suppression"),
        "human_anchor_suppression_after": candidate_profile.get("human_anchor_suppression"),
        "human_anchor_suppression_gain": human_anchor_gain,
        "turnitin_like_component_drops": component_drops,
        "driver_drops": driver_drops,
        "review_burden_delta": review_burden_delta,
        "weighted_severity_delta": weighted_severity_delta,
        "critical_high_delta": critical_high_delta,
    }

def run_post_density_human_anchor_probe(
    current_text: str,
    current_report: dict | None,
    original_report: dict | None,
    *,
    scan_func=None,
    drift_checker=None,
    max_scans: int | None = None,
    deps: PostDensityHumanAnchorProbeDeps | None = None,
) -> dict:
    """Run a small Human Anchor probe after the density breaker baseline."""
    if deps is None:
        raise ValueError("PostDensityHumanAnchorProbeDeps is required")
    if not deps.env_flag("DRAFTPROOF_POST_DENSITY_HUMAN_ANCHOR_PROBE", True):
        return {"enabled": False, "reason": "disabled"}
    if not isinstance(current_text, str) or not current_text.strip() or not isinstance(current_report, dict):
        return {"enabled": False, "reason": "missing_current_selection"}
    scan_func = scan_func or deps.run_full_scan_report_dict
    drift_checker = drift_checker or deps.check_semantic_drift
    max_scans = max(
        0,
        int(max_scans if isinstance(max_scans, int) else deps.float_env("DRAFTPROOF_POST_DENSITY_HUMAN_ANCHOR_MAX_SCANS", 3.0)),
    )
    profile = _turnitin_like_ai_profile(current_report)
    contract = deps.human_anchor_driver_contract(current_report, text=current_text)
    frontier = deps.human_anchor_suppression_frontier(current_text, current_report, None)
    summary = {
        "enabled": True,
        "version": "post_density_human_anchor_probe_v1",
        "selected": False,
        "selected_text": current_text,
        "selected_report": current_report,
        "selected_strategy": None,
        "candidate_count": 0,
        "scans_used": 0,
        "base_summary": {
            "turnitin_like_ai_score": profile.get("score"),
            "target_gap": profile.get("target_gap"),
            "positive_ai_burden": profile.get("raw_positive_score"),
            "human_anchor_suppression": profile.get("human_anchor_suppression"),
        },
        "human_anchor_driver_contract": contract,
        "human_anchor_suppression_frontier": frontier,
        "candidates": [],
        "reason": "no_candidate_selected",
    }
    if max_scans <= 0:
        summary["reason"] = "scan_budget_zero"
        return summary
    if bool(profile.get("target_met")):
        summary["reason"] = "turnitin_like_target_already_met"
        return summary
    candidates = build_post_density_human_anchor_probe_candidates(
        current_text,
        current_report,
        limit=max_scans,
        deps=deps,
    )
    summary["candidate_count"] = len(candidates)
    if not candidates:
        summary["reason"] = "no_human_anchor_probe_candidates"
        return summary

    protected = deps.detect_protected_spans(current_text)
    seen = {current_text.strip()}
    best_eval = None
    best_rank = None
    best_text = current_text
    best_report = current_report
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
        budget_status = deps.ai_density_patchwork_budget_status(current_text, normalized, meta or {})
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
        acceptance = post_density_human_anchor_probe_acceptance(
            current_report,
            candidate_report,
            review_burden_delta=review_delta,
            weighted_severity_delta=severity_delta,
            critical_high_delta=critical_delta,
            deps=deps,
        )
        candidate_eval.update({
            "acceptance": acceptance,
            "selectable": acceptance.get("selectable"),
            "reason": acceptance.get("reason"),
            "formula_score": acceptance.get("formula_score_after"),
            "formula_score_drop": acceptance.get("formula_score_drop"),
            "human_anchor_suppression_gain": acceptance.get("human_anchor_suppression_gain"),
            "positive_ai_burden_drop": acceptance.get("positive_ai_burden_drop"),
            "driver_drops": acceptance.get("driver_drops"),
            "strict_ai_safe_band": deps.strict_ai_safe_band_status(candidate_report),
            "human_anchor_driver_contract": deps.human_anchor_driver_contract(
                current_report,
                candidate_report,
                text=normalized,
            ),
        })
        summary["candidates"].append(candidate_eval)
        if not acceptance.get("selectable"):
            continue
        rank = (
            1 if _turnitin_like_ai_profile(candidate_report).get("target_met") else 0,
            float(acceptance.get("formula_score_drop") or 0.0),
            float(acceptance.get("human_anchor_suppression_gain") or 0.0),
            float(acceptance.get("positive_ai_burden_drop") or 0.0),
            float((acceptance.get("driver_drops") or {}).get("external_ai_flag_risk") or 0.0),
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
                    "human_anchor_suppression_gain",
                    "positive_ai_burden_drop",
                    "driver_drops",
                    "patchwork_budget",
                    "drift_similarity",
                    "acceptance",
                    "human_anchor_driver_contract",
                )
            },
            "final_summary": {
                "turnitin_like_ai_score": _turnitin_like_ai_profile(best_report).get("score"),
                "positive_ai_burden": _turnitin_like_ai_profile(best_report).get("raw_positive_score"),
                "human_anchor_suppression": _turnitin_like_ai_profile(best_report).get("human_anchor_suppression"),
                "ai_footprint": deps.strict_ai_safe_band_status(best_report).get("profile"),
            },
        })
    return summary
