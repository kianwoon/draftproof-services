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
    eligible = [row for row in generated if row.get("accepted_by_selector")]
    if not eligible:
        return True
    if all(_RETRY_WARNINGS & set(row.get("quality_warnings") or []) for row in eligible):
        return True
    best = _best_eligible_row(eligible)
    if int(best.get("candidate_findings") or 0) <= 0:
        return False
    if _material_progress_candidate(best):
        return False
    if int(best.get("candidate_findings") or 0) > 0:
        return True
    return all(_RETRY_WARNINGS & set(row.get("quality_warnings") or []) for row in eligible)


def plan_with_writer_feedback(plan: Plan, diagnostics: list[dict[str, Any]]) -> Plan:
    route = dict(plan.ai_safe_route)
    route["writer_retry_feedback"] = [
        {
            "variant_id": str(row.get("variant_id") or ""),
            "blockers": _row_blockers(row)[:8],
            "quality_warnings": list(row.get("quality_warnings") or [])[:8],
            "candidate_findings": row.get("candidate_findings"),
            "source_findings": row.get("source_findings"),
            "candidate_mean_risk": row.get("candidate_mean_risk"),
            "source_mean_risk": row.get("source_mean_risk"),
            "missing_required_terms": list(row.get("missing_required_terms") or [])[:12],
            "finding_tag_summary": _finding_tag_summary(row),
            "failed_sentences": _failed_sentences(row),
            "residual_rewrite_obligations": _residual_rewrite_obligations(row),
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
    eligible = [row for row in rows if row.get("accepted_by_selector")]
    if eligible:
        best = _best_eligible_row(eligible)
        if int(best.get("candidate_findings") or 0) > 0:
            return [best]
    blocked = [row for row in rows if _row_blockers(row)]
    pool = blocked or rows
    return [min(pool, key=lambda row: (len(_row_blockers(row)), float(row.get("candidate_mean_risk") or 999.0)))]


def _row_blockers(row: dict[str, Any]) -> list[str]:
    return _dedupe([
        *[str(item) for item in row.get("blockers") or [] if str(item).strip()],
        *[str(item) for item in row.get("integrity_blockers") or [] if str(item).strip()],
    ])


def _best_eligible_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            int(row.get("candidate_findings") or 999),
            float(row.get("candidate_mean_risk") or 999.0),
            -float(row.get("risk_drop") or 0.0),
        ),
    )


def _material_progress_candidate(row: dict[str, Any]) -> bool:
    source_findings = int(row.get("source_findings") or 0)
    candidate_findings = int(row.get("candidate_findings") or 0)
    finding_drop = source_findings - candidate_findings
    risk_drop = float(row.get("risk_drop") or 0.0)
    if source_findings <= 0:
        return risk_drop >= 5.0
    if candidate_findings <= 1 and finding_drop >= 1 and risk_drop >= 0.0:
        return True
    if finding_drop >= 2 and risk_drop >= -2.0:
        return True
    return finding_drop >= 1 and risk_drop >= 5.0


def _required_fix(row: dict[str, Any]) -> str:
    blockers = set(_row_blockers(row))
    warnings = set(row.get("quality_warnings") or [])
    fixes: list[str] = []
    if "source_frame_reified_as_entity" in blockers:
        fixes.append("Preserve the source frame as an effect, tension, opportunity, or risk. Do not rewrite it as the topic creating a concrete tool or entity.")
    if "required_source_beat_missing" in blockers:
        fixes.append(_source_beat_missing_fix(row))
    if "required_source_terms_missing" in blockers:
        missing = ", ".join(str(term) for term in row.get("missing_required_terms") or [])
        fixes.append(f"Preserve every required source term, especially the paragraph closing beat. Missing terms: {missing}.")
    if "writer_generation_failed" in blockers:
        fixes.append("Return the requested JSON variants array with complete paragraph text for every requested variant id.")
    if "planner_language_leakage" in blockers:
        fixes.append("Remove visible planning, bridge, evidence, or research meta-language.")
    if "malformed_subject_verb_agreement" in blockers:
        fixes.append(
            "Repair grammar before scoring: do not append leftover route terms with 'and' to an unrelated clause."
        )
    if "malformed_serial_verb_chain" in blockers:
        fixes.append(
            "Repair the malformed verb chain before retrying. Do not repeat the same auxiliary around a negation, "
            "and do not attach a negation marker to an incomplete verb phrase. Use a complete clause or preserve "
            "the source negation scope grammatically."
        )
    if "malformed_negation_order" in blockers:
        fixes.append(
            "Preserve negation scope with a grammatical clause. Use either the source negation structure or a complete "
            "new clause; do not mix both shapes."
        )
    if "missing_verb_after_negation_scope" in blockers:
        fixes.append(
            "Repair the negation clause before retrying: after do/does/did not always, include the missing verb before "
            "any how/what/when/where/why/whether phrase."
        )
    if "copy_blocked_source_phrase" in blockers:
        fixes.append("Preserve meaning without copying any source phrase that the Planner marked as copy-blocked.")
    if "dangling_additive_tail" in blockers:
        fixes.append("Remove dangling additive tails; every final phrase must attach to the sentence grammar.")
    if "source_scope_marker_reused_as_content" in blockers:
        fixes.append("Do not reuse guarded scope-marker words as new content. Keep the source relation only; do not create a new phrase from the marker word.")
    if "premature_assessment_consequence" in blockers:
        fixes.append("Move final consequence terms to the final route beat only. First write the planned support, risk, and mismatch beats.")
    if "transition_label_final_consequence" in blockers:
        fixes.append(
            "Rewrite the final consequence without a transition-label opener and without compressing abstract concerns into one list."
        )
    if "compressed_final_consequence_list" in blockers:
        fixes.append(
            "Preserve the final consequence terms, but distribute them across short final beats instead of one long abstract comma list."
        )
    if "duplicated_assessment_consequence" in blockers:
        fixes.append("Write the final consequence once only, at the end. Do not repeat the failed consequence route.")
    if "vague_danger_opener" in blockers:
        fixes.append("Remove vague abstract risk-label wording. Show the source risk as visible behaviour and the missing work or reasoning.")
    if "repeated_sentence_intent" in blockers:
        fixes.append(
            "Remove repeated intent: do not add a standalone proxy/context sentence and then repeat the same source claim. "
            "Fuse the bridge into the relevant source beat, and do not split quoted concepts into 'the words ...' sentences."
        )
    if "source_beat_reintroduced_after_closing" in blockers:
        fixes.append(
            "Keep source beat order intact. Do not add an earlier support or condition beat after the paragraph has already reached its closing consequence."
        )
    if fixes:
        return " ".join(_dedupe(fixes))
    if int(row.get("candidate_findings") or 0) > 0:
        obligations = _residual_rewrite_obligations(row)
        if obligations:
            return " ".join(obligations)
        return "Clear the remaining scanner finding in this candidate. Use failed_sentences to rewrite the flagged sentence with a different route, no transition label, and no compressed abstract list."
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


def _source_beat_missing_fix(row: dict[str, Any]) -> str:
    gaps = row.get("source_beat_coverage_gaps")
    if not isinstance(gaps, list):
        return "Restore the missing source beat. Do not delete a source list or closing judgement beat to improve scanner score."
    missing_terms: list[str] = []
    for gap in gaps:
        if isinstance(gap, dict) and float(gap.get("coverage_ratio") or 0.0) <= 0.0:
            missing_terms.extend(str(term) for term in gap.get("missing_terms") or [] if str(term).strip())
    if missing_terms:
        return "Restore the missing source beat with these required terms: " + ", ".join(_dedupe(missing_terms)[:12]) + "."
    return "Restore the missing source beat. Do not delete a source list or closing judgement beat to improve scanner score."


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
    text = str(item.get("text") or "")
    if "malformed_serial_verb_chain" in blockers and _has_local_verb_chain_defect(text):
        return "Fix the verb chain first: remove doubled auxiliaries and rebuild the contrast as one grammatical clause."
    if "malformed_negation_order" in blockers and _has_local_negation_order_defect(text):
        return "Fix negation scope first: keep the negation marker attached to a grammatical verb phrase or rebuild as a complete clause."
    if "missing_verb_after_negation_scope" in blockers and _has_missing_verb_after_negation_scope(text):
        return "Fix the missing verb after the negation marker before changing style or sentence rhythm."
    if "repeated_sentence_intent" in blockers:
        return "Fuse the bridge into the same source beat; do not repeat the same intent as a standalone proxy/context sentence."
    if "unsupported_evidence_tail" in blockers:
        return "Remove invented research/evidence wording; keep only claims supported by the source paragraph or Planner route."
    if "compressed_final_consequence_list" in blockers and _looks_like_final_consequence(str(item.get("text") or "")):
        return "Split this final consequence into short final beats. Preserve required judgement terms without a comma-list ending."
    if "transition_label_final_consequence" in blockers and _looks_like_final_consequence(str(item.get("text") or "")):
        return "Remove the transition-label opener from this final consequence and rebuild the sentence route."
    if {"predictable_start", "context_anchor_gap"} & tags and {"author_anchor_gap", "unsupported_claim_gap", "broad_claim"} & tags:
        return "Rebuild this as a grounded source consequence. Do not start with a demonstrative or concern label; name the actor/action and keep only source-supported scope."
    if "context_anchor_gap" in tags and _starts_with_demonstrative_or_label(text):
        return "Replace the demonstrative or label opener with the concrete actor/action from the same paragraph; do not leave This/It/That as the subject."
    if "packed_list" in tags and "sentence_overload" in tags and {"broad_claim", "paragraph_rhythm"} & tags:
        return "Split this overloaded claim into separate sentence jobs: one contrast or condition, then one scoped consequence. Do not keep every action in one list."
    if "packed_list" in tags and "sentence_overload" in tags:
        return "Split overloaded relations into two natural sentences without losing source terms; each sentence should do one job only."
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


def _residual_rewrite_obligations(row: dict[str, Any]) -> list[str]:
    details = [item for item in row.get("candidate_finding_details") or [] if isinstance(item, dict)]
    obligations: list[str] = []
    if any("context_anchor_gap" in set(item.get("tags") or []) and _starts_with_demonstrative_or_label(str(item.get("text") or "")) for item in details):
        obligations.append("Replace residual demonstrative or label starts with a named actor/action from the same paragraph.")
    if any({"packed_list", "sentence_overload"} <= set(item.get("tags") or []) for item in details):
        obligations.append("Break overloaded list sentences into separate sentence jobs; keep only one compact list where all items share the same role.")
    if any({"author_anchor_gap", "unsupported_claim_gap"} & set(item.get("tags") or []) for item in details):
        obligations.append("Remove unsupported broad explanation from residual findings; keep the sentence inside submitted source scope or approved proxy scope.")
    if any("predictable_start" in set(item.get("tags") or []) for item in details):
        obligations.append("Change predictable residual openers by changing the sentence job, not by swapping one opener word.")
    return _dedupe(obligations)


def _starts_with_demonstrative_or_label(text: str) -> bool:
    lowered = str(text or "").strip().casefold()
    return lowered.startswith(("this ", "that ", "it ", "these ", "those ", "there is ", "there are "))


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
            "created",
            "produced",
            "generated",
            "research",
            "evidence",
            "supports this",
            "as a result",
            "consequently",
            "therefore",
            "thus",
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
        f"and aim for zero remaining findings before selection. Stay below the source finding count {source_findings}, "
        f"or produce a clearly lower mean risk than {risk} without repeating any failed sentence route. "
        f"Preserve required source terms and remove listed defects."
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


def _has_local_verb_chain_defect(text: str) -> bool:
    tokens = _word_tokens(text)
    if any(left == right and left in _AUXILIARY_TOKENS for left, right in zip(tokens, tokens[1:])):
        return True
    return _has_local_negation_order_defect(text)


def _has_local_negation_order_defect(text: str) -> bool:
    tokens = _word_tokens(text)
    for left, middle, right in zip(tokens, tokens[1:], tokens[2:]):
        if left in _BE_AUXILIARY_TOKENS and middle in {"do", "does", "did"} and right == "not":
            return True
    for index, token in enumerate(tokens[:-1]):
        if token not in {"not", "n't"}:
            continue
        window = tokens[index + 1 : index + 5]
        if window and window[0] in {"how", "what", "why", "where", "when"}:
            return True
        if len(window) >= 2 and window[0] in {"always", "really", "fully"} and window[1] in {"how", "what", "why", "where", "when"}:
            return True
    return False


def _has_missing_verb_after_negation_scope(text: str) -> bool:
    tokens = _word_tokens(text)
    for first, second, third, fourth in zip(tokens, tokens[1:], tokens[2:], tokens[3:]):
        if first in {"do", "does", "did"} and second == "not" and third == "always" and fourth in {"how", "what", "why", "where", "when", "whether"}:
            return True
    return False


def _word_tokens(text: str) -> list[str]:
    normalized = "".join(ch.casefold() if ch.isalnum() or ch == "'" else " " for ch in str(text or ""))
    return [part for part in normalized.split() if part]


_AUXILIARY_TOKENS = {
    "am",
    "are",
    "be",
    "been",
    "being",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "was",
    "were",
    "will",
    "would",
}

_BE_AUXILIARY_TOKENS = {"am", "are", "be", "been", "being", "is", "was", "were"}
