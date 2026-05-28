from __future__ import annotations

from dataclasses import replace
from typing import Any

from .plan import Plan


_RETRY_WARNINGS = {
    "candidate_contract_violation_review_required",
    "compressed_list_repair_review_required",
    "mechanical_transition_stack_review_required",
    "required_source_terms_missing_review_required",
    "source_polarity_changed_review_required",
    "unsupported_semantic_padding_review_required",
}


def needs_writer_feedback_retry(diagnostics: list[dict[str, Any]]) -> bool:
    generated = [row for row in diagnostics if row.get("source") != "source_preserved"]
    if not generated:
        return False
    if any(row.get("blockers") for row in generated):
        return True
    eligible = [row for row in generated if row.get("accepted_by_selector")]
    if not eligible:
        return True
    return all(_RETRY_WARNINGS & set(row.get("quality_warnings") or []) for row in eligible)


def plan_with_writer_feedback(plan: Plan, diagnostics: list[dict[str, Any]]) -> Plan:
    route = dict(plan.ai_safe_route)
    route["writer_retry_feedback"] = [
        {
            "variant_id": str(row.get("variant_id") or ""),
            "blockers": list(row.get("blockers") or [])[:6],
            "quality_warnings": list(row.get("quality_warnings") or [])[:8],
            "candidate_findings": row.get("candidate_findings"),
            "source_findings": row.get("source_findings"),
            "candidate_mean_risk": row.get("candidate_mean_risk"),
            "source_mean_risk": row.get("source_mean_risk"),
            "missing_required_terms": list(row.get("missing_required_terms") or [])[:12],
            "finding_tag_summary": _finding_tag_summary(row),
            "failed_sentences": _failed_sentences(row),
            "do_not_repeat": _do_not_repeat(row),
            "required_fix": _required_fix(row),
            "retry_goal": _retry_goal(row),
        }
        for row in _feedback_rows(diagnostics)
    ]
    return replace(plan, ai_safe_route=route)


def _feedback_rows(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in diagnostics if row.get("source") != "source_preserved"]
    if not rows:
        return []
    blocked = [row for row in rows if row.get("blockers")]
    pool = blocked or rows
    return [min(pool, key=lambda row: (len(row.get("blockers") or []), float(row.get("candidate_mean_risk") or 999.0)))]


def _required_fix(row: dict[str, Any]) -> str:
    blockers = set(row.get("blockers") or [])
    warnings = set(row.get("quality_warnings") or [])
    if "required_source_terms_missing" in blockers:
        missing = ", ".join(str(term) for term in row.get("missing_required_terms") or [])
        return f"Preserve every required source term, especially the paragraph closing beat. Missing terms: {missing}."
    if "writer_generation_failed" in blockers:
        return "Return the requested JSON variants array with complete paragraph text for every requested variant id."
    if "planner_language_leakage" in blockers:
        return "Remove visible planning, bridge, evidence, or research meta-language."
    if "source_scope_marker_reused_as_content" in blockers:
        return "Do not reuse guarded scope-marker words as new content. If the source says 'no longer', keep that relation only; do not create phrases like a longer period, longer process, longer role, or longer change."
    if "premature_assessment_consequence" in blockers:
        return (
            "Move final consequence terms to the final route beat only. First write the planned support, risk, and mismatch beats."
        )
    if "duplicated_assessment_consequence" in blockers:
        return (
            "Write the final consequence once only, at the end. Do not repeat the failed consequence route."
        )
    if "vague_danger_opener" in blockers:
        return "Remove vague abstract risk-label wording. Show the risk as student dependence on generated answers without understanding the work."
    if "repeated_sentence_intent" in blockers:
        return (
            "Remove repeated intent: do not add a standalone proxy/context sentence and then repeat the same source claim. "
            "Fuse the bridge into the relevant source beat, and do not split quoted concepts into 'the words ...' sentences."
        )
    if "candidate_contract_violation_review_required" in warnings:
        return "Keep the same source route but rebuild the risky shape instead of appending a new claim."
    if "compressed_list_repair_review_required" in warnings:
        return "Use natural compact lists only where items share one role; do not over-compress the paragraph."
    if "source_polarity_changed_review_required" in warnings:
        return "Preserve uncertainty and modality from the source; do not turn a possible risk into a certain outcome."
    if "unsupported_semantic_padding_review_required" in warnings:
        return "Remove unsupported padding words added to a source-aligned sentence. Rebuild the sentence from submitted source terms instead of adding explanatory filler."
    if "insufficient_scanner_movement" in blockers:
        return "Change the paragraph route enough to reduce scanner findings; lowering mean risk alone is not enough for this retry."
    return "Rewrite again from the Planner route with cleaner paragraph flow and no new claims."


def _failed_sentences(row: dict[str, Any]) -> list[dict[str, Any]]:
    details = row.get("candidate_finding_details")
    if not isinstance(details, list):
        return []
    return [
        {
            "text": str(item.get("text") or "")[:240],
            "tags": list(item.get("tags") or [])[:6],
            "repair_instruction": _repair_instruction(item, row),
        }
        for item in details
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ][:5]


def _repair_instruction(item: dict[str, Any], row: dict[str, Any]) -> str:
    tags = set(item.get("tags") or [])
    blockers = set(row.get("blockers") or [])
    if "unsupported_evidence_tail" in blockers:
        return "Remove invented research/evidence wording; keep only claims supported by the source paragraph or Planner route."
    if "packed_list" in tags and "sentence_overload" in tags:
        return "Split overloaded relations into two natural sentences without losing source terms."
    if "packed_list" in tags:
        return "Keep related list items together only when they share the same role; avoid a long scanner-like list."
    if "context_anchor_gap" in tags or "author_anchor_gap" in tags:
        return "Name the concrete source relation inside the same source beat; do not start with This/It/That or append a separate explanation sentence."
    if _looks_like_final_consequence(str(item.get("text") or "")):
        return "Keep this consequence as the final paragraph beat only; remove any earlier duplicate consequence sentence."
    if "predictable_start" in tags:
        return "Change the sentence opening and actor route, not just one synonym."
    if "paragraph_rhythm" in tags:
        return "Vary sentence jobs and length while preserving the same paragraph argument."
    return "Rebuild this sentence from the Planner route; do not copy the failed shape."


def _do_not_repeat(row: dict[str, Any]) -> list[str]:
    text = str(row.get("candidate_text") or "").strip()
    finding_texts = [
        str(item.get("text") or "").strip()
        for item in row.get("candidate_finding_details") or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    sentences = [part.strip() for part in text.replace("?", ".").replace("!", ".").split(".") if part.strip()]
    blocked = [sentence[:220] for sentence in finding_texts]
    for sentence in sentences:
        if any(marker in sentence.casefold() for marker in (
            "research",
            "evidence",
            "supports this",
            "as a result",
            "consequently",
            "the words",
            "this makes assessment",
        )):
            blocked.append(sentence[:220])
    return _dedupe(blocked)[:5]


def _retry_goal(row: dict[str, Any]) -> str:
    findings = row.get("candidate_findings")
    source_findings = row.get("source_findings")
    risk = row.get("candidate_mean_risk")
    return (
        f"Beat this failed candidate, not merely rewrite it: produce fewer than {findings} candidate findings "
        f"and below the source finding count {source_findings}, or produce a clearly lower mean risk than {risk} "
        f"without repeating any failed sentence route. Preserve required source terms and remove listed defects."
    )


def _finding_tag_summary(row: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in row.get("candidate_finding_details") or []:
        if not isinstance(item, dict):
            continue
        for tag in item.get("tags") or []:
            key = str(tag).strip()
            if key:
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _dedupe(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            rows.append(text)
            seen.add(key)
    return rows


def _looks_like_final_consequence(text: str) -> bool:
    lowered = text.casefold()
    consequence_markers = ("consequence", "result", "harder", "difficult", "judge", "review", "evaluate")
    return sum(1 for marker in consequence_markers if marker in lowered) >= 2
