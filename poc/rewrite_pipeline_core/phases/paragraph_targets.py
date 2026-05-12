"""Paragraph role and target-selection helpers for rewrite phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import re


@dataclass(frozen=True)
class ParagraphTargetDeps:
    logical_paragraphs: Callable[[str], list[str]]
    text_word_count: Callable[[str], int]
    float_env: Callable[[str, float], float]
    env_flag: Callable[[str, bool], bool]


def is_heading_like_paragraph(paragraph: str) -> bool:
    text = str(paragraph or "").strip()
    if not text:
        return False
    if re.search(r"[.!?:;]\s*$", text):
        return False
    return bool(len(text.split()) <= 9)


def orphan_heading_reason(text: str, *, deps: ParagraphTargetDeps) -> str:
    paragraphs = deps.logical_paragraphs(text)
    for index, paragraph in enumerate(paragraphs):
        if not is_heading_like_paragraph(paragraph):
            continue
        if index == 0 and len(paragraphs) > 1:
            continue
        next_paragraph = paragraphs[index + 1] if index + 1 < len(paragraphs) else ""
        if not next_paragraph or is_heading_like_paragraph(next_paragraph):
            return f"orphan_heading:{paragraph[:60]}"
    return ""


def paragraph_sentence_starters(paragraph: str) -> list[str]:
    starters = []
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        words = re.findall(r"\b[A-Za-z][A-Za-z']*\b", sentence)
        if words:
            starters.append(words[0].lower())
    return starters


PARAGRAPH_CITATION_RE = re.compile(
    r"(?:\([A-Z][A-Za-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)|"
    r"\b[A-Z][A-Za-z]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z]+)?\s*\(\d{4}\))"
)


def paragraph_role(paragraph: str, drivers: dict | None = None, *, is_last: bool = False) -> str:
    drivers = drivers or {}
    text = str(paragraph or "")
    citation_count = len(PARAGRAPH_CITATION_RE.findall(text))
    first_person_count = len(re.findall(r"\b(?:I|my|me)\b", text))
    process_count = len(re.findall(
        r"\b(?:task|process|method|procedure|tool|material|step|check|review|"
        r"feedback|draft|participant|client|user|case|condition|constraint|"
        r"measurement|test|testing|observed|practice|workflow)\b",
        text,
        flags=re.I,
    ))
    source_gap = bool(drivers.get("source_gap"))
    generic_hits = int(drivers.get("generic_assertion_hits") or 0)
    word_count = max(1, int(drivers.get("word_count") or len(text.split()) or 1))
    if first_person_count >= 3 and process_count >= 6:
        return "human_anchor_rich"
    if is_last or re.search(r"^\s*(?:Conclusion|This review has discussed)\b", text, re.I):
        return "conclusion_template_risk"
    if citation_count >= 2 and first_person_count == 0:
        return "source_summary_heavy"
    if process_count >= 8 and first_person_count >= 1:
        return "technical_process_rich"
    if source_gap or generic_hits / max(word_count / 100.0, 1.0) >= 4.0:
        return "generic_claim_heavy"
    return "mixed"


def paragraph_component_targets(
    text: str,
    raw_json: dict,
    limit: int = 3,
    *,
    deps: ParagraphTargetDeps,
) -> list[dict]:
    """Rank logical paragraphs by their likely contribution to AI-style score."""
    paragraphs = deps.logical_paragraphs(text)
    if not paragraphs:
        return []
    total_words = max(1, deps.text_word_count(text))
    average_paragraph_words = total_words / max(1, len(paragraphs))
    configured_min_words = int(deps.float_env("DRAFTPROOF_PARAGRAPH_TARGET_MIN_WORDS", 45.0))
    if deps.env_flag("DRAFTPROOF_ADAPTIVE_SHORT_PARAGRAPH_TARGETS", False):
        effective_min_words = max(
            12,
            min(configured_min_words, int(max(average_paragraph_words * 0.75, 12.0))),
        )
    else:
        effective_min_words = configured_min_words
    briefs = (raw_json or {}).get("rewrite_edit_briefs") or []
    scored = []
    generic_re = re.compile(
        r"\b(?:should|must|need(?:s|ed)?|requires?|important|significant|"
        r"supports?|helps?|allows?|enables?|creates?|means|is|are|can|will)\b",
        re.I,
    )
    concrete_re = re.compile(
        r"\b(?:[A-Z]{2,}[A-Z0-9-]{2,}|\d+(?:\.\d+)?%?|"
        r"\([A-Z][A-Za-z]+,\s*\d{4}\)|I|my|"
        r"(?i:case|client|participant|user|tool|method|procedure|task|workflow|"
        r"condition|constraint|measurement|test|feedback|observed|practice))\b"
    )
    for index, paragraph in enumerate(paragraphs):
        words = paragraph.split()
        matching_briefs = []
        for brief in briefs:
            if not isinstance(brief, dict):
                continue
            target_sentence = (brief.get("target_sentence") or "").strip()
            if target_sentence and target_sentence in paragraph:
                matching_briefs.append(brief)
        generic_hits = len(generic_re.findall(paragraph))
        concrete_hits = len(concrete_re.findall(paragraph))
        starters = paragraph_sentence_starters(paragraph)
        repeated_starter_count = len(starters) - len(set(starters))
        has_citation = bool(PARAGRAPH_CITATION_RE.search(paragraph))
        brief_score = sum(
            float(((b.get("signals") or {}).get("score") or 0.0) or 0.0)
            for b in matching_briefs
        )
        source_gap = 0 if has_citation else 1
        score = (
            len(matching_briefs) * 5.0
            + brief_score * 8.0
            + min(generic_hits / max(len(words) / 90.0, 1.0), 8.0)
            + source_gap * 2.0
            + min(repeated_starter_count, 4) * 0.75
            - min(concrete_hits, 12) * 0.20
        )
        if score <= 0:
            continue
        drivers = {
            "rewrite_brief_count": len(matching_briefs),
            "predictability_score_sum": round(brief_score, 4),
            "generic_assertion_hits": generic_hits,
            "concrete_anchor_hits": concrete_hits,
            "source_gap": bool(source_gap),
            "repeated_sentence_starters": repeated_starter_count,
            "word_count": len(words),
        }
        role = paragraph_role(paragraph, drivers, is_last=index == len(paragraphs) - 1)
        if (
            role == "conclusion_template_risk"
            and len(words) < 8
            and index + 1 < len(paragraphs)
            and not is_heading_like_paragraph(paragraphs[index + 1])
        ):
            continue
        if len(words) < effective_min_words and role != "conclusion_template_risk":
            continue
        role_score_adjustment = {
            "generic_claim_heavy": 80.0,
            "conclusion_template_risk": 70.0,
            "source_summary_heavy": 50.0,
            "mixed": 20.0,
            "technical_process_rich": -35.0,
            "human_anchor_rich": -100.0,
        }.get(role, 0.0)
        adjusted_score = score + role_score_adjustment
        if role == "human_anchor_rich" and not source_gap:
            adjusted_score -= 50.0
        if adjusted_score <= 0:
            continue
        scored.append({
            "index": index,
            "paragraph": paragraph,
            "previous_paragraph": paragraphs[index - 1] if index > 0 else "",
            "next_paragraph": paragraphs[index + 1] if index + 1 < len(paragraphs) else "",
            "score": round(adjusted_score, 3),
            "raw_score": round(score, 3),
            "role": role,
            "drivers": drivers,
            "target_sentences": [
                (b.get("target_sentence") or "") for b in matching_briefs[:5]
            ],
            "problem_spans": [
                span
                for b in matching_briefs[:4]
                for span in (((b.get("signals") or {}).get("predictable_token_spans")) or [])[:3]
            ][:10],
            "domain_anchors": list(dict.fromkeys(
                anchor
                for b in matching_briefs[:4]
                for anchor in (b.get("domain_anchors") or [])[:6]
            ))[:16],
        })
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:max(0, limit)]
