"""Hard safety validation for compiler candidates."""

from __future__ import annotations

from typing import Any

from .signals import content_terms, protected_anchor_terms


def _adds_content_terms(source_text: str, candidate_text: str) -> bool:
    return bool(content_terms(candidate_text) - content_terms(source_text))


def validate_candidate(
    current_text: str,
    candidate_text: str,
    meta: dict | None,
    deps: Any,
    *,
    cumulative_aggression: float = 0.0,
    cumulative_locality: float = 0.0,
    max_patchwork_ratio: float = 0.18,
    max_patch_aggression: float = 0.18,
    max_cumulative_aggression: float = 0.32,
    max_cumulative_locality: float = 0.50,
) -> dict:
    meta = meta if isinstance(meta, dict) else {}
    reject_reasons: list[str] = []
    if not str(candidate_text or "").strip() or str(candidate_text or "").strip() == str(current_text or "").strip():
        reject_reasons.append("no_text_change")
    aggression = deps.repair_aggression_score(current_text, candidate_text)
    locality = deps.locality_score(current_text, candidate_text)
    aggression_score = float((aggression or {}).get("score") or 0.0)
    locality_ratio = float((locality or {}).get("changed_sentence_ratio") or 0.0)
    if aggression_score > max_patch_aggression:
        reject_reasons.append("repair_aggression_over_patch_budget")
    if locality_ratio > max_patchwork_ratio:
        reject_reasons.append("patchwork_budget_exceeded")
    if cumulative_aggression + aggression_score > max_cumulative_aggression:
        reject_reasons.append("cumulative_aggression_budget_exhausted")
    if cumulative_locality + locality_ratio > max_cumulative_locality:
        reject_reasons.append("cumulative_locality_budget_exhausted")
    protected_reason = deps.protected_loss_reason(
        current_text,
        candidate_text,
        deps.detect_protected_spans(current_text),
    )
    if protected_reason:
        reject_reasons.append("anchor_or_protected_fact_lost " + protected_reason)
    current_anchors = protected_anchor_terms(current_text)
    candidate_lower = str(candidate_text or "").lower()
    lost_anchors = sorted(anchor for anchor in current_anchors if anchor and anchor not in candidate_lower)
    if lost_anchors:
        reject_reasons.append("fact_inventory_lost " + ", ".join(lost_anchors[:5]))
    if _adds_content_terms(current_text, candidate_text):
        concept_reason = deps.concept_origin_reject_reason(current_text, candidate_text)
        if concept_reason:
            reject_reasons.append("new_claim_added " + concept_reason)
    try:
        drift = deps.drift_checker(current_text, candidate_text, threshold=0.85)
    except TypeError:
        drift = deps.drift_checker(current_text, candidate_text)
    drift_similarity = float(getattr(drift, "similarity", 1.0) or 1.0)
    if not bool(getattr(drift, "accepted", True)):
        reject_reasons.append("semantic_drift " + "; ".join(list(getattr(drift, "reasons", []) or [])[:3]))
    return {
        "passed": not reject_reasons,
        "reject_reasons": reject_reasons,
        "repair_aggression": aggression,
        "locality": locality,
        "patchwork_ratio": locality_ratio,
        "cumulative_aggression_after": round(cumulative_aggression + aggression_score, 3),
        "cumulative_locality_after": round(cumulative_locality + locality_ratio, 3),
        "drift_similarity": round(drift_similarity, 3),
        "operator_contract": meta.get("operator_contract"),
    }
