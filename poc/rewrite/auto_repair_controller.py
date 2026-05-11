"""Automated repair compiler for DraftProof rewrite.

This module is intentionally separate from ``rewrite_pipeline.py``.  The
pipeline owns scanning, scoring, and report assembly; this controller owns only
operator planning, tiny patch generation, gate orchestration, and candidate
selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Callable


GENERIC_CONNECTOR_RE = re.compile(
    r"^(?:Furthermore|Moreover|Additionally|In conclusion|Overall|Therefore|"
    r"However|At the same time|In addition|Despite|Another important|"
    r"One of the|This|These|It is important)\b",
    re.I,
)

GENERIC_TERM_RE = re.compile(
    r"\b(?:important|significant|major|strong|influential|different|many|"
    r"various|complex|global|modern|today|society|culture|system|country|"
    r"world|success|challenge|opportunity|influence|impact|role|feature|"
    r"strength|development|diversity|economy|people)\b",
    re.I,
)

CANONICAL_FACT_RE = re.compile(
    r"(?:\b\d{3,4}\b|https?://|www\.|\[[^\]]+\]|\([^)]*\d{4}[^)]*\)|"
    r"\b[A-Z]{2,}[A-Z0-9-]{2,}\b)",
)

TERM_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "into", "about",
    "also", "have", "has", "had", "been", "were", "was", "are", "but",
    "not", "can", "will", "would", "could", "should", "its", "their",
    "there", "these", "those", "than", "then", "when", "where", "what",
}

OPERATORS = (
    "REMOVE_GENERIC_CONNECTOR",
    "SHORTEN_EXPLANATORY_SENTENCE",
    "SPLIT_OVERLONG_SENTENCE",
    "REDUCE_SYMMETRIC_CADENCE",
    "RESTORE_LOCAL_ROUGHNESS",
    "COMPRESS_ABSTRACT_CLAIM",
    "INCREASE_SPECIFICITY_FROM_EXISTING_ANCHORS",
    "MOVE_EXAMPLE_EARLIER",
    "DELETE_EMPTY_META_SENTENCE",
)


@dataclass
class AutoRepairDependencies:
    split_sentences: Callable[[str], list[str]]
    text_word_count: Callable[[str], int]
    geometry_risk_map: Callable[..., dict]
    sentence_texture_risk_map: Callable[..., list[dict]]
    ordered_concept_terms: Callable[..., list[str]]
    is_canonical_fact_sentence: Callable[[str], bool]
    splice_sentence: Callable[[str, int, str], str]
    repair_aggression_score: Callable[[str, str], dict]
    locality_score: Callable[[str, str], dict]
    detect_protected_spans: Callable[[str], Any]
    protected_loss_reason: Callable[[str, str, Any], str]
    concept_origin_reject_reason: Callable[[str, str], str]
    drift_checker: Callable[..., Any]
    scan_func: Callable[[str], dict]
    turnitin_profile: Callable[[dict | None], dict]
    turnitin_gate_status: Callable[..., dict]
    strict_safe_status: Callable[[dict | None], dict]
    contribution_scores: Callable[[dict | None], dict]
    integrity_scores: Callable[[dict | None], dict]
    badge_ai: Callable[[dict | None], float | None]
    finding_total: Callable[[dict | None], int]
    review_burden: Callable[[dict | None], int]
    weighted_severity: Callable[[dict | None], int]
    critical_high_count: Callable[[dict | None], int]


def operator_contract(operator: str) -> dict:
    return {
        "operator": str(operator or "").strip().upper(),
        "input_scope": "one_sentence_or_two",
        "allowed_change_ratio": 0.15,
        "must_preserve_anchors": True,
        "must_not_add_claims": True,
        "must_not_increase_ai_authorship": True,
    }


def _normalize_content_term(token: str) -> str:
    value = re.sub(r"[^a-z0-9_-]", "", str(token or "").lower()).strip("_-")
    if len(value) <= 3 or value in TERM_STOPWORDS:
        return ""
    if value.endswith("ing") and len(value) > 6:
        value = value[:-3]
    elif value.endswith(("ed", "es")) and len(value) > 5:
        value = value[:-2]
    elif value.endswith("s") and len(value) > 5:
        value = value[:-1]
    return "" if len(value) <= 3 or value in TERM_STOPWORDS else value


def _content_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9'_-]{2,}", str(text or "")):
        normalized = _normalize_content_term(token)
        if normalized:
            terms.add(normalized)
    return terms


def _candidate_adds_content_terms(source_text: str, candidate_text: str) -> bool:
    """Return true only when a patch introduces new content terms.

    The global concept-origin guard is still used for additive patches, but
    deterministic micro-operators such as connector deletion, clause splitting,
    and sentence compression should not be rejected as "new claims" when their
    term inventory is unchanged.
    """
    return bool(_content_terms(candidate_text) - _content_terms(source_text))


def sentence_risk_types(sentence: str, row: dict | None, deps: AutoRepairDependencies) -> list[str]:
    value = str(sentence or "").strip()
    row = row if isinstance(row, dict) else {}
    drivers = row.get("drivers") if isinstance(row.get("drivers"), dict) else {}
    risks: list[str] = []
    if GENERIC_CONNECTOR_RE.search(value):
        risks.append("generic_connector")
    if deps.text_word_count(value) >= 22 or float(drivers.get("cadence_uniformity") or 0.0) >= 0.5:
        risks.append("smooth_explanatory_cadence")
    if len(GENERIC_TERM_RE.findall(value)) >= 2:
        risks.append("abstract_density_high")
    if float(drivers.get("repeated_opening") or 0.0) > 0.0:
        risks.append("repeated_opening_route")
    if float(drivers.get("clause_balance") or 0.0) >= 0.5:
        risks.append("balanced_clause_route")
    if re.search(r"\bfor example\b", value, re.I) and deps.text_word_count(value) >= 16:
        risks.append("example_buried_late")
    if re.search(r"^(?:It is important to note|Understanding .+ requires|This (?:shows|means|highlights|demonstrates))\b", value, re.I):
        risks.append("empty_meta_sentence")
    return risks


def choose_operators(risk_types: list[str], sentence: str, deps: AutoRepairDependencies) -> list[str]:
    selected: list[str] = []
    if "empty_meta_sentence" in risk_types:
        selected.append("DELETE_EMPTY_META_SENTENCE")
    if "generic_connector" in risk_types:
        selected.append("REMOVE_GENERIC_CONNECTOR")
    if "smooth_explanatory_cadence" in risk_types:
        selected.extend(["SHORTEN_EXPLANATORY_SENTENCE", "SPLIT_OVERLONG_SENTENCE"])
    if "abstract_density_high" in risk_types:
        selected.append("COMPRESS_ABSTRACT_CLAIM")
    if "repeated_opening_route" in risk_types or "balanced_clause_route" in risk_types:
        selected.extend(["REDUCE_SYMMETRIC_CADENCE", "RESTORE_LOCAL_ROUGHNESS"])
    if "example_buried_late" in risk_types:
        selected.append("MOVE_EXAMPLE_EARLIER")
    if deps.ordered_concept_terms(sentence, limit=2):
        selected.append("INCREASE_SPECIFICITY_FROM_EXISTING_ANCHORS")
    ordered = []
    for operator in selected:
        if operator in OPERATORS and operator not in ordered:
            ordered.append(operator)
    return ordered[:5]


def compile_plan(text: str, report_dict: dict | None, deps: AutoRepairDependencies, *, max_windows: int = 3) -> dict:
    geometry_map = deps.geometry_risk_map(text, report_dict, limit=max(8, max_windows * 3))
    texture_map = deps.sentence_texture_risk_map(text, report_dict, limit=max(8, max_windows * 3))
    texture_by_index = {
        int(row.get("sentence_index")): row
        for row in texture_map
        if isinstance(row, dict) and isinstance(row.get("sentence_index"), int)
    }
    rows = []
    for row in geometry_map.get("sentence_hotspots") or []:
        if not isinstance(row, dict):
            continue
        sentence = str(row.get("sentence") or "")
        sentence_id = int(row.get("sentence_index") if isinstance(row.get("sentence_index"), int) else -1)
        if sentence_id < 0 or not sentence.strip():
            continue
        if row.get("protected") or deps.is_canonical_fact_sentence(sentence):
            risks = ["canonical_fact"]
            operators: list[str] = []
            action = "canonical_fact_preserve"
        else:
            risks = sentence_risk_types(sentence, row, deps)
            texture_row = texture_by_index.get(sentence_id) or {}
            if float(texture_row.get("risk") or 0.0) > 0.0 and "scanner_texture_pointer" not in risks:
                risks.append("scanner_texture_pointer")
            operators = choose_operators(risks, sentence, deps)
            action = operators[0].lower() if operators else "preserve"
        rows.append({
            "sentence_id": sentence_id,
            "sentence": sentence,
            "risk_type": risks,
            "recommended_operator": operators[0] if operators else None,
            "operators": operators,
            "action": action,
            "weighted_geometry_drag": row.get("weighted_geometry_drag"),
            "drivers": row.get("drivers"),
            "canonical_fact_preserve": action == "canonical_fact_preserve",
            "texture_risk": (texture_by_index.get(sentence_id) or {}).get("risk"),
        })
    editable = [row for row in rows if row.get("operators")]
    editable.sort(
        key=lambda row: (
            float(row.get("weighted_geometry_drag") or 0.0),
            float(row.get("texture_risk") or 0.0),
        ),
        reverse=True,
    )
    preserved = [row for row in rows if row.get("canonical_fact_preserve")]
    return {
        "version": "auto_repair_compile_plan_v1",
        "target_shape": "scan_to_operator_to_candidate_to_rescan",
        "operator_mode": "deterministic_patch_generator",
        "geometry_risk_map": {
            "version": geometry_map.get("version"),
            "sentence_count": geometry_map.get("sentence_count"),
            "dominant_weighted_drivers": geometry_map.get("dominant_weighted_drivers"),
        },
        "risk_map": (editable + preserved)[:max(1, max_windows) + 8],
        "selected_windows": editable[:max(1, max_windows)],
        "canonical_preserved_count": len(preserved),
    }


def apply_operator(sentence: str, operator: str, deps: AutoRepairDependencies) -> tuple[str, list[str]]:
    original = str(sentence or "").strip()
    if not original or deps.is_canonical_fact_sentence(original):
        return original, []
    candidate = original
    op = str(operator or "").strip().upper()
    applied: list[str] = []
    if op == "REMOVE_GENERIC_CONNECTOR":
        updated = re.sub(
            r"^(?:Furthermore|Moreover|Additionally|In conclusion|Overall|Therefore|However|"
            r"At the same time|In addition|Despite this|Despite its success),?\s+",
            "",
            candidate,
            count=1,
            flags=re.I,
        ).strip()
        if updated and updated != candidate:
            candidate = updated[0].upper() + updated[1:]
            applied.append(op)
    elif op == "SHORTEN_EXPLANATORY_SENTENCE":
        for pattern, replacement in (
            (r"\bIt is important to note that\s+", ""),
            (r"\bThis means that\s+", "That means "),
            (r"\bThis shows that\s+", "That shows "),
            (r"\bin many different ways\b", ""),
            (r"\bfor many decades\b", ""),
            (r"\bmodern society\b", "society"),
        ):
            updated = re.sub(pattern, replacement, candidate, count=1, flags=re.I)
            if updated != candidate:
                candidate = updated
                applied.append(op)
                break
    elif op == "SPLIT_OVERLONG_SENTENCE":
        if deps.text_word_count(candidate) >= 18:
            for pattern, prefix in (
                (r"\s+because\s+", "Because "),
                (r"\s+but\s+", "But "),
                (r"\s+while\s+", "While "),
                (r"\s+which\s+", "That "),
                (r"\s+and\s+", ""),
            ):
                match = re.search(pattern, candidate, flags=re.I)
                if not match:
                    continue
                left = candidate[:match.start()].strip(" ,;")
                right = candidate[match.end():].strip(" ,;")
                if deps.text_word_count(left) >= 6 and deps.text_word_count(right) >= 5:
                    right = f"{prefix}{right[0].lower() + right[1:]}" if prefix and len(right) > 1 else right.capitalize()
                    candidate = f"{left}. {right}"
                    applied.append(op)
                    break
    elif op in {"REDUCE_SYMMETRIC_CADENCE", "RESTORE_LOCAL_ROUGHNESS"}:
        if "," in candidate:
            first, rest = candidate.split(",", 1)
            if 3 <= deps.text_word_count(first) <= 9 and deps.text_word_count(rest) >= 6:
                candidate = f"{rest.strip()} {first.strip().lower()}."
                candidate = re.sub(r"\.\.$", ".", candidate)
                applied.append(op)
    elif op == "COMPRESS_ABSTRACT_CLAIM":
        for pattern, replacement in (
            (r"\bis often described as one of the most influential\b", "has wide influence"),
            (r"\bhas shaped\b", "shapes"),
            (r"\bis known for\b", "is marked by"),
            (r"\bplays? (?:a|an) (?:important|significant|major|crucial) role in\b", "matters in"),
            (r"\bhas a (?:strong|significant|major) influence on\b", "influences"),
            (r"\bone of the biggest strengths of\b", "one strength of"),
            (r"\bAnother important feature of\b", "Another part of"),
        ):
            updated = re.sub(pattern, replacement, candidate, count=1, flags=re.I)
            if updated != candidate:
                candidate = updated
                applied.append(op)
                break
    elif op == "INCREASE_SPECIFICITY_FROM_EXISTING_ANCHORS":
        terms = deps.ordered_concept_terms(candidate, limit=2)
        if terms and re.search(r"\bthis\b", candidate, re.I):
            updated = re.sub(r"\bThis\b", terms[0].capitalize(), candidate, count=1)
            updated = re.sub(r"\bthis\b", terms[0], updated, count=1)
            if updated != candidate:
                candidate = updated
                applied.append(op)
    elif op == "MOVE_EXAMPLE_EARLIER":
        match = re.search(r"(.+?),\s*for example,\s*(.+)", candidate, flags=re.I)
        if match and deps.text_word_count(match.group(1)) >= 5:
            candidate = f"For example, {match.group(2).strip()} {match.group(1).strip().lower()}."
            candidate = re.sub(r"\.\.$", ".", candidate)
            applied.append(op)
    elif op == "DELETE_EMPTY_META_SENTENCE":
        if re.search(r"^(?:It is important to note|Understanding .+ requires|This (?:shows|means|highlights|demonstrates))\b", candidate, re.I):
            candidate = ""
            applied.append(op)
    candidate = re.sub(r"\s{2,}", " ", candidate).strip()
    return (candidate, applied) if applied and candidate != original else (original, [])


def candidate_pool(text: str, plan: dict, deps: AutoRepairDependencies, *, limit: int = 8) -> list[tuple[str, str, dict]]:
    candidates: list[tuple[str, str, dict]] = []
    seen = {str(text or "").strip()}
    for window in plan.get("selected_windows") or []:
        if not isinstance(window, dict):
            continue
        sentence = str(window.get("sentence") or "")
        sentence_id = int(window.get("sentence_id") if isinstance(window.get("sentence_id"), int) else -1)
        for operator in window.get("operators") or []:
            replacement, applied = apply_operator(sentence, operator, deps)
            if not applied or replacement == sentence:
                continue
            candidate_text = deps.splice_sentence(text, sentence_id, replacement)
            normalized = candidate_text.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            aggression = deps.repair_aggression_score(text, candidate_text)
            locality = deps.locality_score(text, candidate_text)
            candidates.append((
                f"auto_repair_{operator.lower()}_s{sentence_id + 1}",
                candidate_text,
                {
                    "auto_repair_controller": True,
                    "operator": operator,
                    "operator_contract": operator_contract(operator),
                    "sentence_id": sentence_id,
                    "risk_type": window.get("risk_type") or [],
                    "original_sentence": sentence,
                    "replacement_sentence": replacement,
                    "applied_operations": applied,
                    "repair_aggression": aggression,
                    "locality": locality,
                },
            ))
            if len(candidates) >= limit:
                return candidates
    return candidates


def mechanical_gate(
    current_text: str,
    candidate_text: str,
    meta: dict | None,
    deps: AutoRepairDependencies,
    *,
    cumulative_aggression: float,
    cumulative_locality: float,
) -> dict:
    meta = meta if isinstance(meta, dict) else {}
    aggression = meta.get("repair_aggression") if isinstance(meta.get("repair_aggression"), dict) else deps.repair_aggression_score(current_text, candidate_text)
    locality = meta.get("locality") if isinstance(meta.get("locality"), dict) else deps.locality_score(current_text, candidate_text)
    aggression_score = float(aggression.get("score") or 0.0)
    locality_ratio = float(locality.get("changed_sentence_ratio") or 0.0)
    reject_reasons: list[str] = []
    if aggression_score > 0.12:
        reject_reasons.append("repair_aggression_over_patch_budget")
    if locality_ratio > 0.15:
        reject_reasons.append("locality_over_pass_budget")
    if cumulative_aggression + aggression_score > 0.22:
        reject_reasons.append("cumulative_aggression_budget_exhausted")
    if cumulative_locality + locality_ratio > 0.45:
        reject_reasons.append("cumulative_locality_budget_exhausted")
    if GENERIC_CONNECTOR_RE.search(str(meta.get("replacement_sentence") or "").strip()):
        reject_reasons.append("banned_connector_found")
    protected_loss = deps.protected_loss_reason(
        current_text,
        candidate_text,
        deps.detect_protected_spans(current_text),
    )
    if protected_loss:
        reject_reasons.append("protected_span_lost " + protected_loss)
    if _candidate_adds_content_terms(current_text, candidate_text):
        concept_reason = deps.concept_origin_reject_reason(current_text, candidate_text)
        if concept_reason:
            reject_reasons.append("new_claim_added " + concept_reason)
    return {
        "passed": not reject_reasons,
        "reject_reasons": reject_reasons,
        "repair_aggression": aggression,
        "locality": locality,
        "cumulative_aggression_after": round(cumulative_aggression + aggression_score, 3),
        "cumulative_locality_after": round(cumulative_locality + locality_ratio, 3),
    }


def acceptance_status(
    current_report: dict | None,
    candidate_report: dict | None,
    original_report: dict | None,
    deps: AutoRepairDependencies,
    *,
    mechanical: dict,
) -> dict:
    current_profile = deps.turnitin_profile(current_report)
    candidate_profile = deps.turnitin_profile(candidate_report)
    formula_drop = round(float(current_profile.get("score") or 0.0) - float(candidate_profile.get("score") or 0.0), 3)
    current_contribution = deps.contribution_scores(current_report)
    candidate_contribution = deps.contribution_scores(candidate_report)
    current_integrity = deps.integrity_scores(current_report)
    candidate_integrity = deps.integrity_scores(candidate_report)
    human_delta = round(float(candidate_contribution.get("human") or 0.0) - float(current_contribution.get("human") or 0.0), 3)
    ai_transformation_drop = round(float(current_contribution.get("ai_transformation") or 0.0) - float(candidate_contribution.get("ai_transformation") or 0.0), 3)
    ai_authorship_drop = round(float(current_integrity.get("ai_authorship") or 0.0) - float(candidate_integrity.get("ai_authorship") or 0.0), 3)
    ai_score_drop = round(float(deps.badge_ai(current_report) or 0.0) - float(deps.badge_ai(candidate_report) or 0.0), 3)
    review_delta = deps.review_burden(candidate_report) - deps.review_burden(current_report)
    severity_delta = deps.weighted_severity(candidate_report) - deps.weighted_severity(current_report)
    critical_delta = deps.critical_high_count(candidate_report) - deps.critical_high_count(current_report)
    findings_delta = deps.finding_total(candidate_report) - deps.finding_total(current_report)
    turnitin_gate = deps.turnitin_gate_status(
        current_report,
        candidate_report,
        review_burden_delta=review_delta,
        weighted_severity_delta=severity_delta,
        critical_high_delta=critical_delta,
        ai_score_regressed=ai_score_drop < -0.001,
    )
    reject_reasons: list[str] = []
    if not mechanical.get("passed"):
        reject_reasons.extend(mechanical.get("reject_reasons") or ["mechanical_gate_failed"])
    if review_delta > 0:
        reject_reasons.append("review_burden_regressed")
    if severity_delta > 0:
        reject_reasons.append("weighted_severity_regressed")
    if critical_delta > 0:
        reject_reasons.append("critical_high_regressed")
    if findings_delta > 0:
        reject_reasons.append("findings_regressed")
    if ai_authorship_drop < -0.001:
        reject_reasons.append("ai_authorship_delta_gt_0")
    if not (human_delta > 0.0 or ai_transformation_drop > 0.0 or ai_score_drop > 0.0 or formula_drop > 0.0):
        reject_reasons.append("no_pareto_progress")
    aggression_score = float((mechanical.get("repair_aggression") or {}).get("score") or 0.0)
    locality_ratio = float((mechanical.get("locality") or {}).get("changed_sentence_ratio") or 0.0)
    rank_score = round(
        max(0.0, formula_drop) * 4.0
        + max(0.0, human_delta) * 3.0
        + max(0.0, ai_transformation_drop) * 2.0
        + max(0.0, ai_score_drop)
        + max(0.0, ai_authorship_drop)
        - aggression_score * 2.0
        - locality_ratio,
        3,
    )
    return {
        "accepted": not reject_reasons,
        "reason": "accepted_auto_repair_candidate" if not reject_reasons else reject_reasons[0],
        "reject_reasons": reject_reasons,
        "rank_score": rank_score,
        "turnitin_like_ai_gate": turnitin_gate,
        "formula_score_before": current_profile.get("score"),
        "formula_score_after": candidate_profile.get("score"),
        "formula_score_drop": formula_drop,
        "human_delta": human_delta,
        "ai_transformation_drop": ai_transformation_drop,
        "ai_authorship_drop": ai_authorship_drop,
        "ai_score_drop": ai_score_drop,
        "findings_delta": findings_delta,
        "review_burden_delta": review_delta,
        "weighted_severity_delta": severity_delta,
        "critical_high_delta": critical_delta,
    }


def run_auto_repair_controller(
    current_text: str,
    current_report: dict | None,
    original_report: dict | None,
    deps: AutoRepairDependencies,
    *,
    max_rounds: int = 2,
    max_scans: int = 4,
    target_human: float = 80.0,
) -> dict:
    if not isinstance(current_text, str) or not current_text.strip() or not isinstance(current_report, dict):
        return {"enabled": False, "reason": "missing_current_selection"}
    max_rounds = max(0, int(max_rounds or 0))
    max_scans = max(0, int(max_scans or 0))
    current_text_value = current_text
    current_report_value = current_report
    start_profile = deps.turnitin_profile(current_report_value)
    summary = {
        "enabled": True,
        "version": "auto_repair_controller_v1",
        "selected": False,
        "selected_text": current_text_value,
        "selected_report": current_report_value,
        "selected_strategy": None,
        "operator_catalog": [operator_contract(op) for op in OPERATORS],
        "target_human": float(target_human),
        "max_rounds": max_rounds,
        "max_scans": max_scans,
        "scans_used": 0,
        "rounds": [],
        "candidates": [],
        "selected_candidate": None,
        "score_before": start_profile.get("score"),
        "score_after": start_profile.get("score"),
        "human_before": deps.contribution_scores(current_report_value).get("human"),
        "human_after": deps.contribution_scores(current_report_value).get("human"),
        "reason": "no_candidate_selected",
    }
    if max_rounds <= 0 or max_scans <= 0:
        summary["reason"] = "budget_zero"
        return summary
    cumulative_aggression = 0.0
    cumulative_locality = 0.0
    accepted_rounds = 0
    for round_id in range(1, max_rounds + 1):
        if summary["scans_used"] >= max_scans:
            summary["reason"] = "scan_budget_exhausted"
            break
        current_human = deps.contribution_scores(current_report_value).get("human")
        if isinstance(current_human, (int, float)) and float(current_human) >= float(target_human):
            summary["reason"] = "target_human_reached"
            break
        if bool(deps.turnitin_profile(current_report_value).get("target_met")):
            summary["reason"] = "turnitin_like_target_met"
            break
        plan = compile_plan(current_text_value, current_report_value, deps, max_windows=2)
        remaining_scans = max_scans - int(summary["scans_used"] or 0)
        pool = candidate_pool(current_text_value, plan, deps, limit=min(8, remaining_scans))
        round_summary = {
            "round_id": round_id,
            "score_before": deps.turnitin_profile(current_report_value).get("score"),
            "human_before": current_human,
            "risk_map": plan.get("risk_map"),
            "selected_windows": plan.get("selected_windows"),
            "candidate_count": len(pool),
            "scanned": 0,
            "selected": False,
        }
        if not pool:
            round_summary["reason"] = "no_operator_candidates"
            summary["rounds"].append(round_summary)
            summary["reason"] = "no_operator_candidates"
            break
        best_eval = None
        best_rank = None
        best_text = current_text_value
        best_report = current_report_value
        for strategy, candidate_text, meta in pool:
            candidate_eval = {
                "round_id": round_id,
                "strategy": strategy,
                "operator": (meta or {}).get("operator"),
                "operator_contract": (meta or {}).get("operator_contract"),
                "sentence_id": (meta or {}).get("sentence_id"),
                "risk_type": (meta or {}).get("risk_type"),
                "applied_operations": (meta or {}).get("applied_operations"),
                "repair_aggression": (meta or {}).get("repair_aggression"),
                "locality": (meta or {}).get("locality"),
            }
            mechanical = mechanical_gate(
                current_text_value,
                candidate_text,
                meta,
                deps,
                cumulative_aggression=cumulative_aggression,
                cumulative_locality=cumulative_locality,
            )
            candidate_eval["mechanical_gate"] = mechanical
            if not mechanical.get("passed"):
                candidate_eval["reason"] = (mechanical.get("reject_reasons") or ["mechanical_gate_failed"])[0]
                summary["candidates"].append(candidate_eval)
                continue
            try:
                drift = deps.drift_checker(current_text_value, candidate_text, threshold=0.85)
            except TypeError:
                drift = deps.drift_checker(current_text_value, candidate_text)
            candidate_eval["drift_similarity"] = round(float(getattr(drift, "similarity", 1.0)), 3)
            if not bool(getattr(drift, "accepted", True)):
                candidate_eval["reason"] = "semantic_drift " + "; ".join(list(getattr(drift, "reasons", []) or [])[:3])
                candidate_eval["drift_reasons"] = list(getattr(drift, "reasons", []) or [])[:10]
                summary["candidates"].append(candidate_eval)
                continue
            if summary["scans_used"] >= max_scans:
                candidate_eval["reason"] = "scan_budget_exhausted"
                summary["candidates"].append(candidate_eval)
                break
            scan_t0 = time.time()
            try:
                candidate_report = deps.scan_func(candidate_text)
            except Exception as exc:
                candidate_eval["reason"] = f"candidate_scan_error {exc}"
                summary["candidates"].append(candidate_eval)
                continue
            summary["scans_used"] += 1
            round_summary["scanned"] += 1
            acceptance = acceptance_status(
                current_report_value,
                candidate_report,
                original_report,
                deps,
                mechanical=mechanical,
            )
            candidate_eval.update({
                "scan_seconds": round(time.time() - scan_t0, 3),
                "acceptance": acceptance,
                "accepted": acceptance.get("accepted"),
                "reason": acceptance.get("reason"),
                "formula_score": acceptance.get("formula_score_after"),
                "formula_score_drop": acceptance.get("formula_score_drop"),
                "human_delta": acceptance.get("human_delta"),
                "ai_transformation_drop": acceptance.get("ai_transformation_drop"),
                "ai_authorship_drop": acceptance.get("ai_authorship_drop"),
                "ai_score_drop": acceptance.get("ai_score_drop"),
                "rank_score": acceptance.get("rank_score"),
                "strict_ai_safe_band": deps.strict_safe_status(candidate_report),
            })
            summary["candidates"].append(candidate_eval)
            if not acceptance.get("accepted"):
                continue
            rank = (
                1 if deps.turnitin_profile(candidate_report).get("target_met") else 0,
                float(acceptance.get("rank_score") or 0.0),
                float(acceptance.get("formula_score_drop") or 0.0),
                float(acceptance.get("human_delta") or 0.0),
                -float(deps.turnitin_profile(candidate_report).get("score") or 100.0),
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_eval = candidate_eval
                best_text = candidate_text
                best_report = candidate_report
        if not best_eval:
            round_summary["reason"] = "no_safe_pareto_candidate"
            summary["rounds"].append(round_summary)
            summary["reason"] = "no_safe_pareto_candidate"
            break
        best_eval["selected"] = True
        current_text_value = best_text
        current_report_value = best_report
        accepted_rounds += 1
        cumulative_aggression = float((best_eval.get("mechanical_gate") or {}).get("cumulative_aggression_after") or cumulative_aggression)
        cumulative_locality = float((best_eval.get("mechanical_gate") or {}).get("cumulative_locality_after") or cumulative_locality)
        round_summary.update({
            "selected": True,
            "selected_strategy": best_eval.get("strategy"),
            "selected_operator": best_eval.get("operator"),
            "score_after": deps.turnitin_profile(current_report_value).get("score"),
            "human_after": deps.contribution_scores(current_report_value).get("human"),
            "selected_candidate": {
                key: best_eval.get(key)
                for key in (
                    "strategy",
                    "operator",
                    "sentence_id",
                    "risk_type",
                    "formula_score_drop",
                    "human_delta",
                    "ai_transformation_drop",
                    "ai_authorship_drop",
                    "ai_score_drop",
                    "rank_score",
                    "mechanical_gate",
                    "acceptance",
                )
            },
        })
        summary["rounds"].append(round_summary)
    final_profile = deps.turnitin_profile(current_report_value)
    summary.update({
        "selected": accepted_rounds > 0,
        "selected_text": current_text_value,
        "selected_report": current_report_value,
        "selected_strategy": (
            ((summary.get("rounds") or [])[-1] or {}).get("selected_strategy")
            if accepted_rounds > 0 else None
        ),
        "selected_candidate": (
            ((summary.get("rounds") or [])[-1] or {}).get("selected_candidate")
            if accepted_rounds > 0 else None
        ),
        "accepted_rounds": accepted_rounds,
        "score_after": final_profile.get("score"),
        "score_drop": round(float(start_profile.get("score") or 0.0) - float(final_profile.get("score") or 0.0), 3),
        "human_after": deps.contribution_scores(current_report_value).get("human"),
        "human_delta": round(
            float(deps.contribution_scores(current_report_value).get("human") or 0.0)
            - float(deps.contribution_scores(current_report).get("human") or 0.0),
            3,
        ),
        "cumulative_aggression": round(cumulative_aggression, 3),
        "cumulative_locality": round(cumulative_locality, 3),
        "target_met": bool(final_profile.get("target_met")),
    })
    if accepted_rounds > 0 and summary.get("reason") in {"no_candidate_selected", "no_safe_pareto_candidate"}:
        summary["reason"] = "accepted_auto_repair_progress"
    return summary
