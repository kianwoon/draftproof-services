"""Scan-outcome strategy router and candidate generation for rewrite V2."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .content_router import (
    ALL_STRATEGY_FAMILIES,
    ContentRoute,
    FULL_DOCUMENT_STRATEGY_FAMILIES,
)


class StrategyKind(str, Enum):
    TARGETED = "targeted"
    FULL_REWRITE = "full_rewrite"


@dataclass(frozen=True)
class RewriteStrategy:
    strategy_id: str
    kind: StrategyKind
    targeted_drivers: list[str]
    editable_scope: str
    protected_anchors: list[str] = field(default_factory=list)
    expected_metric_movement: list[str] = field(default_factory=list)
    max_candidates: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload


_ALL_STRATEGY_FAMILIES = ALL_STRATEGY_FAMILIES
_FULL_DOC_FAMILIES = FULL_DOCUMENT_STRATEGY_FAMILIES


def _component_values(report: dict | None) -> dict[str, float]:
    badge = (report or {}).get("ai_risk_badge") or {}
    transform = badge.get("transformation_classification") or {}
    features = transform.get("features") or {}
    ai_components = badge.get("ai_components") or {}
    layers = (((report or {}).get("integrity_layers") or {}).get("layers") or {})
    return {
        "ai_likelihood": float(badge.get("ai_likelihood_score") or features.get("ai_likelihood") or 0.0),
        "topk_calibrated_risk": float(ai_components.get("topk_calibrated_risk") or 0.0),
        "qualifying_text_ai_density": float(ai_components.get("qualifying_text_ai_density") or 0.0),
        "ai_authorship": float((layers.get("ai_authorship") or {}).get("score") or 0.0),
        "ai_transformation": float((layers.get("ai_transformation") or {}).get("score") or 0.0),
        "external_ai_flag_risk": float(ai_components.get("external_ai_flag_risk") or 0.0),
    }


def _anchor_allowed(token: str, source_text: str) -> bool:
    cleaned = token.strip()
    if not cleaned:
        return False
    if re.search(r"\d", cleaned):
        return True
    parts = cleaned.split()
    if len(parts) >= 2:
        return True
    stop_anchors = {
        "Although", "However", "This", "These", "Those", "Another", "Understanding",
        "One", "At", "In", "On", "The", "For", "When", "While", "Because",
    }
    if cleaned in stop_anchors:
        return False
    return len(cleaned) > 3 and source_text.count(cleaned) >= 2


def _extract_anchors(report: dict | None, limit: int = 24) -> list[str]:
    anchors: list[str] = []
    source_text = " ".join(
        str((row or {}).get("text") or "")
        for row in ((report or {}).get("sentence_map") or {}).values()
        if isinstance(row, dict)
    )
    for row in (((report or {}).get("scan_intelligence") or {}).get("document") or {}).get("preservation_inventory") or []:
        if isinstance(row, dict):
            value = str(row.get("text") or row.get("value") or "").strip()
            if value and value not in anchors:
                anchors.append(value)
    sentence_map = (report or {}).get("sentence_map") or {}
    for row in list(sentence_map.values())[:8] if isinstance(sentence_map, dict) else []:
        sentence = str((row or {}).get("text") or "").strip()
        for token in re.findall(r"\b[A-Z][A-Za-z0-9&'.-]{2,}(?:\s+[A-Z][A-Za-z0-9&'.-]{2,}){0,4}\b|\b\d+(?:\.\d+)?%?\b", sentence):
            if _anchor_allowed(token, source_text) and token not in anchors:
                anchors.append(token)
            if len(anchors) >= limit:
                return anchors
    return anchors[:limit]


def route_strategies(
    scan_report: dict | None,
    *,
    full_rewrite_allowed: bool = True,
    content_route: ContentRoute | dict[str, Any] | None = None,
) -> list[RewriteStrategy]:
    values = _component_values(scan_report)
    rewrite_briefs = (scan_report or {}).get("rewrite_edit_briefs") or []
    segments = (((scan_report or {}).get("scan_intelligence") or {}).get("document") or {}).get("segments") or []
    localized = bool(rewrite_briefs or segments)
    broad_drivers = [
        key for key, value in values.items()
        if key in {"topk_calibrated_risk", "qualifying_text_ai_density", "ai_authorship", "ai_transformation"}
        and value >= 35.0
    ]
    anchors = _extract_anchors(scan_report)
    broad_document_risk = any(
        values.get(key, 0.0) >= 35.0
        for key in ("topk_calibrated_risk", "qualifying_text_ai_density", "ai_authorship", "ai_transformation")
    )
    full_candidate_budget = max(1, min(4, int(os.environ.get("DRAFTPROOF_REWRITE_V2_FULL_CANDIDATES", "3"))))
    strategies: list[RewriteStrategy] = []
    if localized:
        strategies.append(RewriteStrategy(
            strategy_id="scan_targeted_driver_mitigation",
            kind=StrategyKind.TARGETED,
            targeted_drivers=broad_drivers or ["ai_likelihood"],
            editable_scope="rewrite_edit_briefs_and_scan_segments",
            protected_anchors=anchors,
            expected_metric_movement=["reduce localized AI-pattern drivers", "preserve protected anchors"],
            max_candidates=max(1, min(6, int(os.environ.get("DRAFTPROOF_REWRITE_V2_TARGETED_CANDIDATES", "4")))),
        ))
    allow_full_after_targeted = os.environ.get("DRAFTPROOF_REWRITE_V2_ALLOW_FULL_AFTER_TARGETED", "0").lower() in {"1", "true", "yes"}
    allowed_families = set()
    if isinstance(content_route, ContentRoute):
        allowed_families = set(content_route.allowed_strategy_families)
    elif isinstance(content_route, dict):
        allowed_families = set(content_route.get("allowed_strategy_families") or [])
    full_doc_allowed_by_content = not allowed_families or bool(allowed_families & _FULL_DOC_FAMILIES)
    if full_rewrite_allowed and full_doc_allowed_by_content and (broad_drivers or not localized) and (not localized or allow_full_after_targeted):
        strategies.append(RewriteStrategy(
            strategy_id="scan_full_document_mitigation",
            kind=StrategyKind.FULL_REWRITE,
            targeted_drivers=broad_drivers or ["ai_likelihood", "topk_calibrated_risk"],
            editable_scope="full_document_with_scan_constraints",
            protected_anchors=anchors,
            expected_metric_movement=["reduce document-level AI signature", "preserve claims and anchors"],
            max_candidates=full_candidate_budget if broad_document_risk else 1,
        ))
    if not strategies:
        strategies.append(RewriteStrategy(
            strategy_id="scan_targeted_minimal_mitigation",
            kind=StrategyKind.TARGETED,
            targeted_drivers=["ai_likelihood"],
            editable_scope="rewrite_decision_targets",
            protected_anchors=anchors,
            expected_metric_movement=["reduce AI likelihood"],
            max_candidates=1,
        ))
    return strategies


def clean_candidate_output(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    if text.startswith('"""') and text.endswith('"""'):
        text = text[3:-3].strip()
    return text


def _brief_problem_spans(brief: dict[str, Any]) -> list[str]:
    signals = brief.get("signals") if isinstance(brief.get("signals"), dict) else {}
    spans = signals.get("predictable_token_spans")
    return [str(span).strip() for span in spans if str(span).strip()] if isinstance(spans, list) else []


def _brief_problem_tokens(brief: dict[str, Any], limit: int = 8) -> list[str]:
    signals = brief.get("signals") if isinstance(brief.get("signals"), dict) else {}
    tokens = signals.get("problem_tokens")
    if not isinstance(tokens, list):
        return []
    result = []
    for token in tokens[:limit]:
        if isinstance(token, dict):
            value = str(token.get("token") or "").strip()
            if value:
                result.append(value)
    return result


def _domain_anchors(brief: dict[str, Any], limit: int = 8) -> list[str]:
    generic = {
        "often", "described", "influential", "countries", "modern", "history",
        "important", "major", "key", "role", "shaped", "shaping",
    }
    anchors = brief.get("domain_anchors")
    if not isinstance(anchors, list):
        return []
    result = []
    for anchor in anchors:
        value = str(anchor or "").strip()
        if value and value.lower() not in generic:
            result.append(value)
        if len(result) >= limit:
            break
    return result


def _content_keywords(text: str, limit: int = 28) -> list[str]:
    stop = {
        "the", "and", "that", "this", "with", "from", "into", "about", "also",
        "like", "many", "more", "most", "often", "every", "both", "its",
        "has", "have", "had", "was", "were", "are", "is", "been", "being",
        "described", "known", "requires", "looking", "understanding",
    }
    result: list[str] = []
    for token in re.findall(r"\b[A-Za-z][A-Za-z'-]{3,}\b|\b\d{4}\b", text):
        cleaned = token.strip()
        lower = cleaned.lower()
        if lower in stop or lower in {item.lower() for item in result}:
            continue
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _paragraph_entities(text: str, limit: int = 16) -> list[str]:
    result: list[str] = []
    for token in re.findall(r"\b[A-Z][A-Za-z0-9&'.-]{2,}(?:\s+[A-Z][A-Za-z0-9&'.-]{2,}){0,4}\b|\b\d{4}\b", text):
        cleaned = token.strip()
        if _anchor_allowed(cleaned, text) and cleaned not in result:
            result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _compact_paragraph_briefs(scan_report: dict | None) -> list[dict[str, Any]]:
    briefs = (scan_report or {}).get("rewrite_edit_briefs") or []
    if not isinstance(briefs, list):
        return []
    by_paragraph: dict[str, dict[str, Any]] = {}
    for brief in briefs:
        if not isinstance(brief, dict):
            continue
        target = str(brief.get("target_sentence") or "").strip()
        paragraph = str(brief.get("paragraph_excerpt") or "").strip()
        if not target:
            continue
        key = str(brief.get("paragraph_id") or paragraph or target)
        row = by_paragraph.setdefault(key, {
            "finding_ids": [],
            "paragraph_id": brief.get("paragraph_id"),
            "paragraph_role": brief.get("paragraph_role"),
            "target_word_range": {
                "min": max(45, int(len((paragraph or target).split()) * 0.75)),
                "max": max(75, int(len((paragraph or target).split()) * 1.15)),
            },
            "context_keywords": [],
            "required_entities": [],
            "problem_spans": [],
            "problem_tokens": [],
            "protected_spans": brief.get("protected_spans") or [],
            "usable_anchors": _domain_anchors(brief),
            "instructions": [],
        })
        finding_id = str(brief.get("finding_id") or "").strip()
        if finding_id and finding_id not in row["finding_ids"]:
            row["finding_ids"].append(finding_id)
        for keyword in _content_keywords(" ".join([target, paragraph])):
            if keyword not in row["context_keywords"]:
                row["context_keywords"].append(keyword)
        for entity in _paragraph_entities(" ".join([target, paragraph])):
            if entity not in row["required_entities"]:
                row["required_entities"].append(entity)
        for span in _brief_problem_spans(brief):
            if span not in row["problem_spans"]:
                row["problem_spans"].append(span)
        for token in _brief_problem_tokens(brief):
            if token not in row["problem_tokens"]:
                row["problem_tokens"].append(token)
        instruction = str(brief.get("instruction") or "").strip()
        if instruction and instruction not in row["instructions"]:
            row["instructions"].append(instruction)
    max_briefs = max(1, int(os.environ.get("DRAFTPROOF_REWRITE_V2_TARGETED_PARAGRAPHS", "4")))
    for row in by_paragraph.values():
        row["context_keywords"] = row["context_keywords"][:32]
        row["required_entities"] = row["required_entities"][:16]
    return list(by_paragraph.values())[:max_briefs]


def targeted_paragraph_briefs(scan_report: dict | None) -> list[dict[str, Any]]:
    return _compact_paragraph_briefs(scan_report)


def build_strategy_prompt(original_text: str, scan_report: dict, strategy: RewriteStrategy) -> str:
    text = str(original_text or "")
    text_samples = {
        "opening": text[:900],
        "middle": text[max(0, len(text) // 2 - 450): max(0, len(text) // 2 - 450) + 900] if len(text) > 1800 else "",
        "ending": text[-900:] if len(text) > 900 else "",
        "word_count": len(text.split()),
    }
    brief = {
        "strategy": strategy.to_dict(),
        "rewrite_decision": (scan_report or {}).get("rewrite_decision"),
        "rewrite_edit_briefs": _compact_paragraph_briefs(scan_report),
        "scan_intelligence": {
            "transformation": ((scan_report or {}).get("scan_intelligence") or {}).get("transformation"),
            "generation_handoff": ((scan_report or {}).get("scan_intelligence") or {}).get("generation_handoff"),
        },
        "integrity_layers": (scan_report or {}).get("integrity_layers"),
        "text_samples": text_samples,
    }
    return (
        "Rewrite the document to reduce AI-generated detection risk.\n"
        "Preserve the user's meaning, facts, citations, numeric claims, names, and required anchors.\n"
        "Do not add unsupported facts, sources, methods, examples, or author experiences.\n"
        "Return only the complete rewritten document, no commentary.\n\n"
        f"SCAN-DRIVEN STRATEGY JSON:\n{json.dumps(brief, indent=2, default=str)[:12000]}\n\n"
        "The full original document is intentionally not provided. Use the scan-driven strategy JSON only."
    )


def build_single_paragraph_reconstruction_prompt(
    brief: dict[str, Any],
    strategy: RewriteStrategy,
    *,
    tactic: str = "plain_student_draft",
) -> str:
    preferred_word_count = int((brief.get("target_word_range") or {}).get("source") or 0)
    if not preferred_word_count:
        word_range = brief.get("target_word_range") if isinstance(brief.get("target_word_range"), dict) else {}
        min_words = int(word_range.get("min") or 0)
        max_words = int(word_range.get("max") or 0)
        preferred_word_count = int((min_words + max_words) / 2) if min_words and max_words else 0
    tactic_instructions = {
        "plain_student_draft": "Use plain student analytical prose. Keep sentence choices direct and slightly uneven.",
        "constraint_first": "Start from a concrete constraint, tension, or limitation before naming broad influence.",
        "choppy_analytic": "Use shorter analytic sentences mixed with one longer sentence. Avoid smooth balanced flow.",
        "list_breaker": "Avoid list-like compression. Spread categories across different sentence shapes.",
        "contrast_opening": "Open with contrast or qualification instead of a broad claim.",
        "specific_noun_action": "Start with a concrete noun and action. Avoid abstract summary openings.",
        "minimal_carrier": "Use short factual carrier sentences. Keep verbs simple. Break broad claims into small plain statements.",
        "compressed_power": "Compress the paragraph into fewer concrete claims. Prefer nouns and observable examples over explanation language.",
        "broken_choppy": "Use uneven but complete sentences. Do not use sentence fragments.",
        "simple_subject_stack": "Start most sentences with concrete subjects from the keyword/entity list. Avoid abstract framing.",
    }
    payload = {
        "strategy": strategy.to_dict(),
        "tactic": tactic,
        "tactic_instruction": tactic_instructions.get(tactic, tactic_instructions["plain_student_draft"]),
        "task": "Regenerate this one paragraph from structured context only.",
        "output_schema": {
            "paragraph_id": str(brief.get("paragraph_id") or ""),
            "rewritten_paragraph": "replacement paragraph only",
            "rationale": "short reason tied to paragraph rhythm and problem spans",
        },
        "bad_patterns_to_avoid": [
            "is often described",
            "often described as",
            "one of the most",
            "in modern history",
            "highly influential",
            "key role",
            "played a key role",
            "shaping modern history",
            "significantly affected",
            "influence has extended",
            "worldwide",
            "for decades",
            "emerging from",
            "stands as",
            "significant global entity",
            "various sectors",
            "notable over time",
            "swiftly ascended",
            "melting pot",
            "grapples with",
            "pressing",
            "complex relationship",
            "complex interplay",
            "deeply intertwined",
            "multifaceted",
            "dynamic landscape",
            "continues to shape",
            "evaluating",
            "judging",
            "coexist",
            "persistent challenges",
        ],
        "paragraph_brief": brief,
        "preferred_word_count": preferred_word_count or None,
    }
    word_count_instruction = (
        f"The original paragraph is about {preferred_word_count} words. "
        f"Write the replacement paragraph at about {preferred_word_count} words. Do not summarize it into a shorter paragraph.\n"
        if preferred_word_count
        else ""
    )
    return (
        "DraftProof paragraph reconstruction.\n"
        "Regenerate exactly one paragraph from the structured brief only.\n"
        f"Tactic: {tactic}. {tactic_instructions.get(tactic, tactic_instructions['plain_student_draft'])}\n"
        f"{word_count_instruction}"
        "The original paragraph prose is intentionally not provided.\n"
        "Do not infer or recreate the original sentence order.\n"
        "Break predictable spans, repeated openings, transition rhythm, and paragraph-level uniformity.\n"
        "Prefer minimal factual carrier sentences over polished summary prose.\n"
        "Use complete sentences. Do not create sentence fragments or one-word list sentences.\n"
        "Keep most sentences between 8 and 24 words unless a required entity forces a longer sentence.\n"
        "Do not explain the paragraph's importance or evaluate the topic. State the content plainly.\n"
        "Keep plain student analytical prose. No ornate, promotional, encyclopedic, or marketing language.\n"
        "Every item in paragraph_brief.required_entities must appear verbatim in rewritten_paragraph.\n"
        "Preserve all required entities, numbers, protected spans, and meaning implied by the keywords.\n"
        "Do not add unsupported facts, examples, citations, author experience, or commentary.\n"
        "Return valid JSON only with keys: paragraph_id, rewritten_paragraph, rationale.\n\n"
        f"PARAGRAPH_RECONSTRUCTION_JSON:\n{json.dumps(payload, indent=2, default=str)}"
    )


def build_targeted_resolution_prompt(scan_report: dict, strategy: RewriteStrategy) -> str:
    briefs = _compact_paragraph_briefs(scan_report)
    payload = {
        "strategy": strategy.to_dict(),
        "task": "Regenerate each paragraph from structured context only. Do not rewrite the full document.",
        "output_schema": {
            "candidates": [{
                "candidate_id": "variant_1",
                "patches": [{
                    "finding_ids": ["string"],
                    "paragraph_id": "string",
                    "rewritten_paragraph": "replacement paragraph only",
                    "rationale": "short reason tied to problem spans and paragraph rhythm",
                }],
            }]
        },
        "bad_patterns_to_avoid": [
            "is often described",
            "often described as",
            "one of the most",
            "in modern history",
            "highly influential",
            "key role",
            "played a key role",
            "shaping modern history",
            "significantly affected",
            "influence has extended",
            "worldwide",
            "for decades",
            "emerging from",
            "stands as",
            "significant global entity",
            "various sectors",
            "notable over time",
            "swiftly ascended",
            "melting pot",
            "grapples with",
            "pressing",
            "complex relationship",
            "complex interplay",
            "deeply intertwined",
            "multifaceted",
            "dynamic landscape",
            "continues to shape",
            "evaluating",
            "judging",
            "coexist",
            "persistent challenges",
        ],
        "required_patch_count_per_candidate": len(briefs),
        "paragraph_rewrite_briefs": briefs,
    }
    return (
        "DraftProof targeted AI-mitigation pass.\n"
        "Create 4 distinct candidate patch sets.\n"
        f"Each candidate patch set must contain exactly {len(briefs)} patches: one replacement paragraph for every paragraph brief.\n"
        "Regenerate paragraphs from the structured brief only; the original paragraph prose is intentionally not provided.\n"
        "Break predictable spans, repeated openings, transition rhythm, and paragraph-level uniformity.\n"
        "Prefer minimal factual carrier sentences over polished summary prose.\n"
        "Do not explain the paragraph's importance or evaluate the topic. State the content plainly.\n"
        "Do not infer or recreate the original sentence order.\n"
        "Do not produce generic paraphrases such as 'highly influential', 'key role', or 'shaping modern history'.\n"
        "Do not make the prose ornate, promotional, or encyclopedic. Keep it plain, specific, and student-draft natural.\n"
        "Prefer a concrete subject/action/constraint over broad praise or summary language.\n"
        "Preserve facts, meaning, names, dates, numeric claims, and protected spans.\n"
        "Do not add unsupported facts, examples, citations, author experience, or commentary.\n"
        "Return valid JSON only, matching the requested schema.\n\n"
        f"TARGETED_PAYLOAD_JSON:\n{json.dumps(payload, indent=2, default=str)[:30000]}"
    )
