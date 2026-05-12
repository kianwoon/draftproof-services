"""Author evidence and mitigation-ceiling reporting layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AuthorEvidenceReportingDeps:
    contribution_scores: Callable[[dict | None], dict]
    logical_paragraphs: Callable[[str], list[str]]
    join_logical_paragraphs: Callable[[list[str]], str]
    paragraph_component_targets: Callable[..., list[dict]]
    paragraph_role: Callable[..., str]


def build_author_evidence_completion_layer(
    text: str,
    report_dict: dict | None,
    *,
    target_human: int = 80,
    max_slots: int = 5,
    deps: AuthorEvidenceReportingDeps,
) -> dict:
    """Build a user-completion draft for missing real anchors."""
    if not isinstance(text, str) or not text.strip() or not isinstance(report_dict, dict):
        return {}
    contribution = deps.contribution_scores(report_dict)
    current_human = contribution.get("human")
    if isinstance(current_human, (int, float)) and current_human >= target_human:
        return {
            "enabled": False,
            "reason": "human_target_already_met",
            "current_human_contribution": round(float(current_human), 3),
            "target_human_contribution": target_human,
        }

    badge = report_dict.get("ai_risk_badge") or {}
    writing_components = badge.get("writing_components") or {}
    ai_components = badge.get("ai_components") or {}
    lived_detail_risk = float(writing_components.get("lived_detail_risk") or 0.0)
    source_grounding_risk = float(writing_components.get("source_grounding_risk") or 0.0)
    unsupported_claim_risk = float(writing_components.get("unsupported_claim_risk") or 0.0)
    generic_assertion_risk = float(ai_components.get("generic_assertion_risk") or 0.0)
    density_risk = float(ai_components.get("qualifying_text_ai_density") or 0.0)

    if max(lived_detail_risk, source_grounding_risk, unsupported_claim_risk, generic_assertion_risk, density_risk) < 55:
        return {
            "enabled": False,
            "reason": "no_strong_missing_anchor_signal",
            "current_human_contribution": current_human,
            "target_human_contribution": target_human,
        }

    paragraphs = deps.logical_paragraphs(text)
    if not paragraphs:
        return {}
    target_limit = max(1, int(max_slots or 1))
    targets = deps.paragraph_component_targets(text, report_dict, limit=max(target_limit * 2, target_limit))
    target_indexes = []
    for target in targets:
        index = target.get("index")
        if isinstance(index, int) and index not in target_indexes:
            target_indexes.append(index)
        if len(target_indexes) >= target_limit:
            break
    if not target_indexes:
        ranked = sorted(
            enumerate(paragraphs),
            key=lambda item: len(item[1].split()),
            reverse=True,
        )
        target_indexes = [index for index, paragraph in ranked[:target_limit] if len(paragraph.split()) >= 12]

    slots = []
    patched = list(paragraphs)
    for slot_number, index in enumerate(target_indexes, start=1):
        if index < 0 or index >= len(patched):
            continue
        paragraph = patched[index]
        role = deps.paragraph_role(
            paragraph,
            {"source_gap": source_grounding_risk >= 55, "generic_assertion_hits": 4, "word_count": len(paragraph.split())},
            is_last=index == len(paragraphs) - 1,
        )
        if source_grounding_risk >= 60 or unsupported_claim_risk >= 65:
            instruction = (
                "add one real source, task example, work artefact, feedback note, "
                "or observation that proves this claim"
            )
        elif lived_detail_risk >= 65:
            instruction = (
                "add one real context, workplace, task, assessment, or feedback detail "
                "that you can defend"
            )
        else:
            instruction = "add one real author reasoning detail that narrows this broad claim without inventing evidence"
        marker = f" [[ADD REAL AUTHOR ANCHOR {slot_number}: {instruction}]]"
        stripped = paragraph.rstrip()
        terminal = ""
        if stripped and stripped[-1] in ".!?":
            terminal = stripped[-1]
            stripped = stripped[:-1].rstrip()
        patched[index] = f"{stripped}{marker}{terminal}"
        slots.append({
            "slot": slot_number,
            "paragraph_index": index,
            "paragraph_role": role,
            "instruction": instruction,
            "target_paragraph_preview": paragraph[:220],
            "why_needed": (
                "Human Contribution is capped because the scan sees broad claims, weak lived detail, "
                "or missing source/context anchors. DraftProof will not invent those anchors."
            ),
        })

    if not slots:
        return {}

    current_value = float(current_human) if isinstance(current_human, (int, float)) else 0.0
    slot_lift = min(18.0, len(slots) * 3.5)
    grounding_lift = 0.0
    if source_grounding_risk >= 60:
        grounding_lift += min(8.0, len(slots) * 1.5)
    if lived_detail_risk >= 65:
        grounding_lift += min(8.0, len(slots) * 1.5)
    estimated_low = min(float(target_human), current_value + max(2.0, slot_lift * 0.45))
    estimated_high = min(float(target_human), current_value + slot_lift + grounding_lift)
    return {
        "enabled": True,
        "kind": "author_evidence_completion",
        "auto_apply": False,
        "status": "requires_author_completion",
        "current_human_contribution": round(current_value, 3) if current_value else current_human,
        "target_human_contribution": target_human,
        "estimated_human_after_completion": {
            "low": int(round(estimated_low)),
            "high": int(round(estimated_high)),
            "basis": "Heuristic estimate only; final score requires user-supplied real anchors and rescan.",
        },
        "current_blockers": {
            "lived_detail_risk": lived_detail_risk,
            "source_grounding_risk": source_grounding_risk,
            "unsupported_claim_risk": unsupported_claim_risk,
            "generic_assertion_risk": generic_assertion_risk,
            "qualifying_text_ai_density": density_risk,
        },
        "draft_text": deps.join_logical_paragraphs(patched),
        "slots": slots,
        "instructions": [
            "Replace every [[ADD REAL AUTHOR ANCHOR ...]] marker with a true source, example, observation, feedback moment, limitation, or author judgement.",
            "Do not keep bracket markers in the final submission.",
            "Delete any marker you cannot truthfully support, and narrow the surrounding claim instead.",
            "Rescan only after all markers are resolved with real author-owned evidence.",
        ],
    }


def build_mitigation_ceiling_diagnostics(
    summary: dict,
    author_evidence_completion: dict | None = None,
    *,
    target_human: int = 80,
) -> dict:
    """Explain why automatic mitigation stopped below the Human target."""
    if not isinstance(summary, dict):
        return {}
    scores = summary.get("detect_scores") or {}
    search = summary.get("ai_mitigation_search") or {}
    generation = summary.get("generation_layer") or {}
    candidates = [
        item for item in (search.get("candidates") or [])
        if isinstance(item, dict) and isinstance(item.get("ai"), (int, float))
    ]
    safe_candidates = [
        item for item in candidates
        if (item.get("selection_status") or {}).get("selectable")
    ]
    blocked_candidates = [
        item for item in candidates
        if not (item.get("selection_status") or {}).get("selectable")
    ]

    def num(value, default=None):
        return float(value) if isinstance(value, (int, float)) else default

    original_human = num(scores.get("original_human_contribution"), 0.0)
    final_human = num(scores.get("rewritten_human_contribution"), original_human)
    original_ai = num(scores.get("original_ai"))
    final_ai = num(scores.get("rewritten_ai"))
    original_authorship = num(scores.get("original_ai_authorship"))
    final_authorship = num(scores.get("rewritten_ai_authorship"))
    original_transform = num(scores.get("original_ai_transformation"))
    final_transform = num(scores.get("rewritten_ai_transformation"))

    human_values = [final_human] + [num(item.get("human_contribution"), final_human) for item in candidates]
    safe_human_values = [final_human] + [num(item.get("human_contribution"), final_human) for item in safe_candidates]
    ai_values = [value for value in [final_ai] + [num(item.get("ai")) for item in candidates] if value is not None]
    best_seen_human = max(human_values)
    best_safe_human = max(safe_human_values)
    best_seen_ai = min(ai_values) if ai_values else final_ai

    reason_counts: dict[str, int] = {}
    for item in blocked_candidates:
        status = item.get("selection_status") or {}
        auth = status.get("authenticity_gate") or {}
        reason = str(auth.get("reason") or status.get("reason") or item.get("reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    completion = author_evidence_completion or summary.get("author_evidence_completion") or {}
    estimate = completion.get("estimated_human_after_completion") or {}
    estimated_high = num(estimate.get("high"))
    missing_slots = len(completion.get("slots") or [])
    evidence_blocked = bool(completion.get("enabled")) and missing_slots > 0

    if evidence_blocked and estimated_high is not None and estimated_high < target_human:
        primary = "missing_author_owned_evidence_and_context"
    elif best_seen_human <= final_human + 2:
        primary = "human_score_ceiling_from_generic_claims"
    elif reason_counts:
        primary = max(reason_counts.items(), key=lambda item: item[1])[0]
    else:
        primary = "candidate_pool_exhausted"

    recommended_next = []
    if evidence_blocked:
        recommended_next.append("Complete the Author Evidence slots with real examples, source support, or author observations, then rescan.")
    if candidates and best_safe_human < target_human:
        recommended_next.append("Keep automatic generation in micro-local mode; broad rewrites are no longer the main bottleneck.")
    if reason_counts.get("review_burden_regressed") or reason_counts.get("weighted_severity_regressed"):
        recommended_next.append("Use finding-local patching for review-burden and severity blockers instead of rewriting whole paragraphs.")
    if not recommended_next:
        recommended_next.append("Increase candidate diversity and rescan only if semantic and authorship gates stay non-regressing.")

    return {
        "schema_version": "mitigation_ceiling.v1",
        "target_human_contribution": target_human,
        "primary_blocker": primary,
        "safe_auto_result": {
            "human_contribution": round(final_human, 3) if final_human is not None else None,
            "human_gain": round(final_human - original_human, 3) if final_human is not None else None,
            "ai_score": round(final_ai, 3) if final_ai is not None else None,
            "ai_drop": round(original_ai - final_ai, 3) if original_ai is not None and final_ai is not None else None,
            "ai_authorship": round(final_authorship, 3) if final_authorship is not None else None,
            "ai_authorship_drop": round(original_authorship - final_authorship, 3) if original_authorship is not None and final_authorship is not None else None,
            "ai_transformation": round(final_transform, 3) if final_transform is not None else None,
            "ai_transformation_drop": round(original_transform - final_transform, 3) if original_transform is not None and final_transform is not None else None,
            "review_burden": scores.get("rewritten_review_burden"),
            "weighted_severity": scores.get("rewritten_weighted_severity"),
        },
        "candidate_frontier": {
            "scanned_candidates": len(candidates),
            "safe_candidates": len(safe_candidates),
            "blocked_candidates": len(blocked_candidates),
            "best_seen_human": round(best_seen_human, 3),
            "best_safe_human": round(best_safe_human, 3),
            "best_seen_ai": round(best_seen_ai, 3) if best_seen_ai is not None else None,
            "blocked_reason_counts": reason_counts,
            "blocked_human_winner_repair": search.get("blocked_human_winner_repair"),
        },
        "author_evidence_gap": {
            "enabled": evidence_blocked,
            "slot_count": missing_slots,
            "estimated_human_after_completion": estimate or None,
            "current_blockers": completion.get("current_blockers") or {},
        },
        "generation_layer": {
            "selected": generation.get("selected"),
            "selected_strategy": generation.get("selected_strategy"),
            "selection_reason": generation.get("selection_reason"),
            "candidate_count": generation.get("candidate_count"),
        },
        "recommended_next_actions": recommended_next,
    }


def build_author_context_discovery_layer(
    author_evidence_intake: dict | None,
    report_dict: dict | None = None,
    *,
    max_items: int = 5,
) -> dict:
    """Create an LLM-supervised context discovery contract."""
    intake = author_evidence_intake or {}
    if not isinstance(intake, dict) or not intake.get("questions"):
        return {}
    questions = [q for q in (intake.get("questions") or []) if isinstance(q, dict)]
    if not questions:
        return {}
    limit = max(1, int(max_items or 1))
    badge = (report_dict or {}).get("ai_risk_badge") or {}
    writing_components = badge.get("writing_components") or {}
    ai_components = badge.get("ai_components") or {}
    context_cards = []
    for question in questions[:limit]:
        answer_type = str(question.get("answer_type") or "real_example_or_observation")
        paragraph_role = str(question.get("paragraph_role") or "generic_claim_heavy")
        if answer_type == "source_or_citation":
            safe_shape = "Name the source, class reading, module material, or citation, then state in one sentence which claim it supports."
            follow_up = "Can you verify the source title, author/year, or class material name?"
            evidence_kind = "verifiable_source"
        elif answer_type == "practice_observation":
            safe_shape = "Describe one observed task: who/what was involved, what went wrong or changed, and what feedback or action followed."
            follow_up = "What did you actually observe, and what changed after feedback or another attempt?"
            evidence_kind = "observed_process"
        elif answer_type == "author_judgement":
            safe_shape = "State one limitation or judgement you would defend, tied to the paragraph's existing claim."
            follow_up = "What would you personally limit, qualify, or emphasise here?"
            evidence_kind = "author_judgement"
        else:
            safe_shape = "Give one concrete example or author observation in 1-3 sentences. Include the situation, the visible action/problem, and the reason it supports this claim."
            follow_up = "What real example or observation made this claim true for you?"
            evidence_kind = "real_example"
        context_cards.append({
            "anchor_id": question.get("id"),
            "paragraph_index": question.get("paragraph_index"),
            "paragraph_role": paragraph_role,
            "answer_type": answer_type,
            "evidence_kind": evidence_kind,
            "gap": question.get("question"),
            "target_preview": question.get("target_preview"),
            "llm_follow_up_question": follow_up,
            "safe_answer_shape": safe_shape,
            "minimum_specificity": [
                "one concrete noun or source name",
                "one action, judgement, or observed change",
                "one sentence explaining why it supports the paragraph",
            ],
            "do_not_accept": [
                "maybe/possibly answers",
                "generic claims without a visible situation or source",
                "new statistics, institutions, citations, or experiences generated by the LLM",
            ],
        })

    return {
        "enabled": True,
        "kind": "author_context_discovery",
        "status": "ready_for_llm_supervised_intake",
        "auto_apply": False,
        "purpose": "Use the LLM to ask targeted questions and shape user-confirmed context, then pass only confirmed answers into the rewrite generator.",
        "scanner_context": {
            "lived_detail_risk": writing_components.get("lived_detail_risk"),
            "source_grounding_risk": writing_components.get("source_grounding_risk"),
            "unsupported_claim_risk": writing_components.get("unsupported_claim_risk"),
            "generic_assertion_risk": ai_components.get("generic_assertion_risk"),
            "qualifying_text_ai_density": ai_components.get("qualifying_text_ai_density"),
        },
        "context_cards": context_cards,
        "llm_task_prompt": (
            "You are DraftProof's AUTHOR_CONTEXT_DISCOVERY assistant. Ask the user for the listed missing context. "
            "You may propose answer shapes, but you must not answer on the user's behalf. Mark every answer as confirmed, "
            "uncertain, or not_available. Only confirmed answers with permission_to_use=true may be passed to generation."
        ),
        "answer_payload_schema": {
            "answers": [{
                "anchor_id": "anchor_1",
                "answer": "user-confirmed source, observation, example, limitation, or judgement",
                "confidence": "confirmed|uncertain|not_available",
                "permission_to_use": True,
            }]
        },
        "handoff_env": {
            "json": "DRAFTPROOF_AUTHOR_EVIDENCE_ANSWERS_JSON",
            "file": "DRAFTPROOF_AUTHOR_EVIDENCE_ANSWERS_FILE",
        },
        "success_gate": [
            "answer passes relevance check for its paragraph",
            "answer is confirmed and permissioned by the user",
            "generation uses each confirmed anchor at most once",
            "rescan does not regress AI Authorship, AI Transformation, review burden, severity, or drift gates",
        ],
    }
