"""Scan-outcome strategy router and candidate generation for rewrite V2."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class StrategyKind(str, Enum):
    TARGETED = "targeted"
    FULL_REWRITE = "full_rewrite"


@dataclass(frozen=True)
class ContentRoute:
    content_mode: str
    confidence: float
    reasons: list[str]
    allowed_strategy_families: list[str]
    blocked_strategy_families: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


_FULL_DOC_FAMILIES = {
    "entity_locked_full_reconstruction",
    "keyword_locked_short_texture",
    "author_stance_thesis_reframe",
    "author_stance_texture_pass",
}
_ALL_STRATEGY_FAMILIES = {
    "academic_all_section_compact_reconstruction",
    "academic_cited_section_density_resolver",
    "targeted_paragraph_reconstruction",
    "unsafe_cluster_rescue",
    *_FULL_DOC_FAMILIES,
}


def _paragraph_count(text: str) -> int:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", str(text or "").strip()) if p.strip()]
    if len(paragraphs) > 1:
        return len(paragraphs)
    sentences = re.findall(r"[^.!?]+[.!?]", str(text or ""))
    return max(1, min(12, len(sentences) // 4)) if sentences else 0


def _citation_count(text: str) -> int:
    patterns = [
        r"\([A-Z][A-Za-z' -]+(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2}(?:,\s*p\.?\s*\d+)?\)",
        r"\b[A-Z][A-Za-z' -]+(?:\s+et\s+al\.|(?:,\s*[A-Z][A-Za-z' -]+)*(?:,\s*(?:and|&)\s*[A-Z][A-Za-z' -]+)?)?\s+\((?:19|20)\d{2}[a-z]?\)",
        r"\[(?:\d+|[A-Za-z]+(?:,\s*(?:19|20)\d{2})?)\]",
        r"\b(?:doi|DOI):\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
        r"\b(?:et al\.|References|Bibliography|Works Cited)\b",
    ]
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)


def _quote_count(text: str) -> int:
    return len(re.findall(r"[\"“][^\"”]{8,}[\"”]", text or ""))


def _quote_stats(text: str) -> dict[str, int]:
    spans = re.findall(r"[\"“]([^\"”]{3,})[\"”]", text or "")
    short_terms = 0
    long_or_sentence = 0
    for span in spans:
        words = re.findall(r"\b[\w'-]+\b", span)
        stripped = span.strip()
        if len(words) <= 5 and not re.search(r"[.!?;:]", stripped):
            short_terms += 1
        else:
            long_or_sentence += 1
    return {
        "total": len(spans),
        "short_terms": short_terms,
        "long_or_sentence": long_or_sentence,
    }


def _scan_document_profile(scan_report: dict | None) -> dict[str, Any]:
    report = scan_report or {}
    handoff = report.get("generation_handoff") if isinstance(report.get("generation_handoff"), dict) else {}
    profile = handoff.get("document_profile") if isinstance(handoff.get("document_profile"), dict) else {}
    if profile:
        return profile
    intelligence = report.get("scan_intelligence") if isinstance(report.get("scan_intelligence"), dict) else {}
    nested_handoff = intelligence.get("generation_handoff") if isinstance(intelligence.get("generation_handoff"), dict) else {}
    nested_profile = nested_handoff.get("document_profile") if isinstance(nested_handoff.get("document_profile"), dict) else {}
    return nested_profile if isinstance(nested_profile, dict) else {}


def _academic_profile_signal(text: str, scan_report: dict | None) -> list[str]:
    reasons: list[str] = []
    profile = _scan_document_profile(scan_report)
    document_type = str(profile.get("document_type") or "").lower()
    if re.search(r"\b(reflective_or_analytical_submission|academic|literature|analytical|submission|essay)\b", document_type):
        reasons.append(f"document_type={document_type}")
    lowered = text.lower()
    if re.search(r"\b(literature review|taxonomy|pedagogy|educator|vocational education|vet\b|according to research|argue|suggests|report\s*\(?(?:19|20)\d{2}\)?)\b", lowered):
        reasons.append("academic_discourse_markers")
    reference_register = ((scan_report or {}).get("generation_handoff") or {}).get("reference_register")
    if isinstance(reference_register, list) and reference_register:
        reasons.append(f"reference_register={len(reference_register)}")
    return reasons[:4]


def _bullet_line_ratio(lines: list[str]) -> float:
    if not lines:
        return 0.0
    bullet_lines = [
        line for line in lines
        if re.match(r"^\s*(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+", line)
    ]
    return len(bullet_lines) / max(1, len(lines))


def _has_first_person_stance(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\b(i think|i find|i believe|i argue|i feel|i noticed|my view|my experience|in my opinion)\b", lowered))


def _route_payload(mode: str, confidence: float, reasons: list[str]) -> ContentRoute:
    allowed_by_mode = {
        "broad_explanatory_essay": {
            "targeted_paragraph_reconstruction",
            "unsafe_cluster_rescue",
            "entity_locked_full_reconstruction",
            "keyword_locked_short_texture",
            "author_stance_thesis_reframe",
            "author_stance_texture_pass",
        },
        "academic_cited_text": {
            "academic_all_section_compact_reconstruction",
            "academic_cited_section_density_resolver",
            "targeted_paragraph_reconstruction",
            "unsafe_cluster_rescue",
        },
        "technical_content": {"targeted_paragraph_reconstruction", "unsafe_cluster_rescue"},
        "regulated_policy_content": {"targeted_paragraph_reconstruction", "unsafe_cluster_rescue"},
        "structured_list_table": {"targeted_paragraph_reconstruction"},
        "quote_heavy": {"targeted_paragraph_reconstruction", "unsafe_cluster_rescue"},
        "short_text": {"targeted_paragraph_reconstruction"},
        "personal_reflection": {
            "targeted_paragraph_reconstruction",
            "unsafe_cluster_rescue",
            "author_stance_thesis_reframe",
            "author_stance_texture_pass",
        },
        "creative_marketing": {"targeted_paragraph_reconstruction", "unsafe_cluster_rescue"},
        "generic_expository": {
            "targeted_paragraph_reconstruction",
            "unsafe_cluster_rescue",
            "entity_locked_full_reconstruction",
            "keyword_locked_short_texture",
            "author_stance_thesis_reframe",
            "author_stance_texture_pass",
        },
    }
    allowed = sorted(allowed_by_mode.get(mode, {"targeted_paragraph_reconstruction"}))
    blocked = sorted(_ALL_STRATEGY_FAMILIES - set(allowed))
    return ContentRoute(
        content_mode=mode,
        confidence=round(max(0.0, min(1.0, confidence)), 3),
        reasons=reasons[:8],
        allowed_strategy_families=allowed,
        blocked_strategy_families=blocked,
    )


def classify_content_route(original_text: str, scan_report: dict | None = None) -> ContentRoute:
    """Classify document shape so V2 only runs strategies appropriate to the content."""
    text = str(original_text or "")
    stripped = text.strip()
    words = re.findall(r"\b[\w'-]+\b", stripped)
    word_count = len(words)
    paragraphs = _paragraph_count(stripped)
    lines = [line for line in stripped.splitlines() if line.strip()]
    bullet_ratio = _bullet_line_ratio(lines)
    citation_count = _citation_count(stripped)
    quote_stats = _quote_stats(stripped)
    quote_count = quote_stats["total"]
    long_quote_count = quote_stats["long_or_sentence"]
    short_quote_count = quote_stats["short_terms"]
    academic_reasons = _academic_profile_signal(stripped, scan_report)
    lowered = stripped.lower()
    sentence_map_size = len((scan_report or {}).get("sentence_map") or {})

    if bullet_ratio >= 0.35 or re.search(r"\|.+\|", stripped):
        return _route_payload("structured_list_table", 0.86, [f"bullet_line_ratio={bullet_ratio:.2f}", "structured lines require shape preservation"])
    if citation_count >= 2 or (academic_reasons and citation_count >= 1) or (academic_reasons and word_count >= 450):
        reasons = [f"citation_count={citation_count}"] + academic_reasons
        if short_quote_count:
            reasons.append(f"short_conceptual_quotes={short_quote_count}")
        return _route_payload("academic_cited_text", 0.86, reasons + ["academic citations require citation-preserving rewrite"])
    if long_quote_count >= 3 or (long_quote_count >= 2 and word_count < 450):
        return _route_payload("quote_heavy", 0.82, [f"quote_count={quote_count}", f"long_quote_count={long_quote_count}", "quoted material should remain strict anchors"])
    if re.search(r"\b(api|sdk|json|yaml|http|endpoint|database|function|class|method|stack trace|schema|repository|deployment|kubernetes|docker)\b", lowered) or re.search(r"[{}<>]|```|/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", stripped):
        return _route_payload("technical_content", 0.82, ["technical markers detected", "technical content needs term and structure preservation"])
    if re.search(r"\b(shall|must not|compliance|contract|liability|statutory|regulation|clinical|diagnosis|dosage|patient|medical|legal|terms and conditions)\b", lowered) or re.search(r"\b(?:company|privacy|security|data|workplace|use|refund|billing)\s+policy\b|\bpolicy\s+(?:states|requires|prohibits|allows|applies)\b", lowered):
        return _route_payload("regulated_policy_content", 0.82, ["regulated wording detected", "obligations and claims need minimal targeted edits"])
    if _has_first_person_stance(stripped) and re.search(r"\b(my|i|me)\b", lowered):
        return _route_payload("personal_reflection", 0.78, ["first-person stance markers detected", "author voice already exists"])
    if paragraphs >= 6 and word_count >= 100:
        return _route_payload("broad_explanatory_essay", 0.8, [f"paragraphs={paragraphs}", f"word_count={word_count}", "multi-paragraph explanatory essay shape"])
    if word_count < 120:
        return _route_payload("short_text", 0.88, [f"word_count={word_count}", "short input limits safe reconstruction context"])
    if re.search(r"\b(buy|subscribe|limited offer|brand|customers|conversion|campaign|sales|pricing|launch|product)\b", lowered):
        return _route_payload("creative_marketing", 0.74, ["marketing or product language detected", "essay thesis rewrite would change genre"])
    if sentence_map_size >= 8 and word_count >= 240:
        return _route_payload("broad_explanatory_essay", 0.72, [f"sentence_map_size={sentence_map_size}", f"word_count={word_count}", "scan map indicates document-level essay"])
    return _route_payload("generic_expository", 0.62, [f"paragraphs={paragraphs}", f"word_count={word_count}", "fallback expository mode"])


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
    brief = {
        "strategy": strategy.to_dict(),
        "rewrite_decision": (scan_report or {}).get("rewrite_decision"),
        "rewrite_edit_briefs": (scan_report or {}).get("rewrite_edit_briefs"),
        "scan_intelligence": {
            "transformation": ((scan_report or {}).get("scan_intelligence") or {}).get("transformation"),
            "generation_handoff": ((scan_report or {}).get("scan_intelligence") or {}).get("generation_handoff"),
        },
        "integrity_layers": (scan_report or {}).get("integrity_layers"),
    }
    return (
        "Rewrite the document to reduce AI-generated detection risk.\n"
        "Preserve the user's meaning, facts, citations, numeric claims, names, and required anchors.\n"
        "Do not add unsupported facts, sources, methods, examples, or author experiences.\n"
        "Return only the complete rewritten document, no commentary.\n\n"
        f"SCAN-DRIVEN STRATEGY JSON:\n{json.dumps(brief, indent=2, default=str)[:12000]}\n\n"
        f"ORIGINAL DOCUMENT:\n{original_text}"
    )


def build_single_paragraph_reconstruction_prompt(
    brief: dict[str, Any],
    strategy: RewriteStrategy,
    *,
    tactic: str = "plain_student_draft",
) -> str:
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
    }
    return (
        "DraftProof paragraph reconstruction.\n"
        "Regenerate exactly one paragraph from the structured brief only.\n"
        f"Tactic: {tactic}. {tactic_instructions.get(tactic, tactic_instructions['plain_student_draft'])}\n"
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
