"""Rewrite billing decisions and historical seed-text extraction.

Extracted from tasks.py. Pure functions consumed by the ``run_rewrite`` task.
"""
from __future__ import annotations


# Outcomes that should never bill — they did not deliver rewritten content.
NON_BILLABLE_REWRITE_OUTCOMES = {
    "clean",
    "mitigation_failed_no_safe_candidate",
    "needs_author_context",
    "no_safe_rewrite_applied",
    "original_preserved",
    "partial_candidate_not_strict_safe",
    "skipped",
    "topk_blocked",
}

EXTERNAL_REVIEW_REWRITE_STATUSES = {
    "rewrite_candidate_generated_needs_external_review",
    "rewrite_candidate_generated_needs_author_review",
}

EXTERNAL_REVIEW_REWRITE_WARNINGS = {
    "best_candidate_requires_external_review",
    "author_proxy_candidate_requires_review",
}


def _normalized_billable_text(value) -> str:
    return " ".join(str(value or "").split())


def _rewrite_billing_decision(pipeline_result: dict, rewrite_json: dict) -> dict:
    """Return whether a rewrite reservation should be captured.

    Billing is tied to delivered rewritten content, not merely to a worker task
    reaching the artifact-upload phase.
    """
    pipeline_result = pipeline_result if isinstance(pipeline_result, dict) else {}
    rewrite_json = rewrite_json if isinstance(rewrite_json, dict) else {}
    summary = rewrite_json.get("summary") if isinstance(rewrite_json.get("summary"), dict) else {}

    status = str(pipeline_result.get("status") or rewrite_json.get("status") or "").strip()
    outcome = str(summary.get("outcome") or summary.get("strict_goal_status") or "").strip()
    strict_goal_status = str(summary.get("strict_goal_status") or "").strip()
    public_warning = str(summary.get("public_candidate_warning") or "").strip()
    normalized_status = status.lower()
    normalized_outcome = outcome.lower()
    normalized_strict_goal_status = strict_goal_status.lower()
    normalized_public_warning = public_warning.lower()

    if normalized_status in NON_BILLABLE_REWRITE_OUTCOMES:
        return {
            "billable": False,
            "reason": f"non_billable_status:{normalized_status}",
            "status": status,
            "outcome": outcome,
        }
    if normalized_status in EXTERNAL_REVIEW_REWRITE_STATUSES:
        return {
            "billable": False,
            "reason": f"external_review_required_status:{normalized_status}",
            "status": status,
            "outcome": outcome,
        }
    if normalized_outcome in NON_BILLABLE_REWRITE_OUTCOMES:
        return {
            "billable": False,
            "reason": f"non_billable_outcome:{normalized_outcome}",
            "status": status,
            "outcome": outcome,
        }
    if normalized_outcome in EXTERNAL_REVIEW_REWRITE_STATUSES:
        return {
            "billable": False,
            "reason": f"external_review_required_outcome:{normalized_outcome}",
            "status": status,
            "outcome": outcome,
        }
    if normalized_strict_goal_status in NON_BILLABLE_REWRITE_OUTCOMES:
        return {
            "billable": False,
            "reason": f"non_billable_strict_goal:{normalized_strict_goal_status}",
            "status": status,
            "outcome": outcome,
        }
    if summary.get("best_candidate_external_review_required") is True:
        return {
            "billable": False,
            "reason": "external_review_required",
            "status": status,
            "outcome": outcome,
        }
    if normalized_public_warning in EXTERNAL_REVIEW_REWRITE_WARNINGS:
        return {
            "billable": False,
            "reason": f"external_review_required_warning:{normalized_public_warning}",
            "status": status,
            "outcome": outcome,
        }
    if summary.get("rollback_applied") or summary.get("no_text_change"):
        return {
            "billable": False,
            "reason": "original_text_preserved",
            "status": status,
            "outcome": outcome,
        }

    original_text = _normalized_billable_text(rewrite_json.get("original_text"))
    final_text = _normalized_billable_text(rewrite_json.get("final_text") or summary.get("final_text"))
    text_changed = bool(original_text and final_text and original_text != final_text)
    if not final_text:
        reason = "empty_final_text"
    elif not original_text:
        reason = "missing_original_text"
    elif not text_changed:
        reason = "final_text_unchanged"
    else:
        reason = "rewritten_content_delivered"

    return {
        "billable": text_changed,
        "reason": reason,
        "status": status,
        "outcome": outcome,
        "text_changed": text_changed,
    }


def _seed_text_word_count(text: str) -> int:
    return len(str(text or "").split())


def _historical_seed_is_full_document_candidate(text: str, original_text: str) -> bool:
    text_words = _seed_text_word_count(text)
    original_words = _seed_text_word_count(original_text)
    if text_words <= 0:
        return False
    if original_words <= 0:
        return text_words >= 120
    if original_words < 120:
        return text_words >= max(4, int(original_words * 0.55))
    return text_words >= max(120, int(original_words * 0.55))


def _historical_seed_text_from_entry(entry) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if not isinstance(entry, dict):
        return ""
    for key in ("text", "candidate_text", "rewritten_document", "final_text", "rewritten_text"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    selected = entry.get("selected") if isinstance(entry.get("selected"), dict) else {}
    accepted = entry.get("accepted") if isinstance(entry.get("accepted"), dict) else {}
    for nested in (selected, accepted):
        if not nested:
            continue
        text = _historical_seed_text_from_entry(nested)
        if text:
            return text
    return ""


def _append_historical_seed_candidates(candidates: list[str], entries, original_text: str) -> None:
    if isinstance(entries, dict):
        text = _historical_seed_text_from_entry(entries)
        if _historical_seed_is_full_document_candidate(text, original_text):
            candidates.append(text)
        return
    if not isinstance(entries, list):
        return
    ordered = sorted(
        [entry for entry in entries if isinstance(entry, (dict, str))],
        key=lambda entry: (
            entry.get("rank") if isinstance(entry, dict) and isinstance(entry.get("rank"), (int, float)) else 999,
        ),
    )
    for entry in ordered:
        text = _historical_seed_text_from_entry(entry)
        if _historical_seed_is_full_document_candidate(text, original_text):
            candidates.append(text)


def _dedupe_historical_seed_texts(candidates: list[str], original_text: str, *, limit: int) -> list[str]:
    original_normalized = _normalized_billable_text(original_text)
    seen: set[str] = set()
    seeds: list[str] = []
    for value in candidates:
        text = str(value or "").strip()
        normalized = _normalized_billable_text(text)
        if (
            not normalized
            or normalized == original_normalized
            or normalized in seen
            or not _historical_seed_is_full_document_candidate(text, original_text)
        ):
            continue
        seen.add(normalized)
        seeds.append(text)
        if len(seeds) >= max(1, int(limit or 1)):
            break
    return seeds


def _rewrite_has_paragraph_obligation_hard_stop(summary: dict) -> bool:
    if not isinstance(summary, dict):
        return False
    hard_stop = summary.get("paragraph_obligation_hard_stop")
    if isinstance(hard_stop, dict) and hard_stop.get("active"):
        return True
    if str(summary.get("no_text_change_reason") or "") == "v5_unresolved_paragraph_findings":
        return True
    generation_status = summary.get("candidate_generation_status")
    return (
        isinstance(generation_status, dict)
        and str(generation_status.get("reason") or "") == "unresolved_paragraph_findings"
    )


def _historical_rewrite_seed_texts(rewrite_json: dict | None, original_text: str, *, limit: int = 3) -> list[str]:
    if not isinstance(rewrite_json, dict):
        return []
    summary = rewrite_json.get("summary") if isinstance(rewrite_json.get("summary"), dict) else {}
    if _rewrite_has_paragraph_obligation_hard_stop(summary):
        return []
    candidates: list[str] = []
    _append_historical_seed_candidates(candidates, rewrite_json.get("candidate_ledger"), original_text)
    _append_historical_seed_candidates(candidates, summary.get("candidate_ledger"), original_text)
    layers = summary.get("rewrite_layers") if isinstance(summary.get("rewrite_layers"), dict) else {}
    v5_layer = layers.get("v5_residual_cluster_comb") if isinstance(layers.get("v5_residual_cluster_comb"), dict) else {}
    _append_historical_seed_candidates(candidates, v5_layer.get("seed_candidate_rows"), original_text)
    _append_historical_seed_candidates(candidates, v5_layer.get("seed_recovery"), original_text)
    _append_historical_seed_candidates(candidates, v5_layer.get("global_best_fallback"), original_text)
    _append_historical_seed_candidates(candidates, v5_layer.get("accepted_checkpoints"), original_text)
    candidates.extend([
        rewrite_json.get("final_text"),
        summary.get("final_text"),
        rewrite_json.get("rewritten_text"),
        summary.get("rewritten_document"),
    ])
    selected = summary.get("selected_candidate") if isinstance(summary.get("selected_candidate"), dict) else {}
    candidates.extend([
        selected.get("candidate_text"),
        selected.get("text"),
    ])
    return _dedupe_historical_seed_texts(candidates, original_text, limit=limit)
