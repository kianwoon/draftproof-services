"""Writing-quality evaluator for deterministic compiler outputs."""

from __future__ import annotations

import re
from typing import Any

from rewrite_controller.quality_gate import evaluate_text_quality_regression

from .signals import content_terms, logical_paragraphs, protected_anchor_terms, split_sentences


def _term_preservation_ratio(source: str, candidate: str) -> float:
    source_terms = content_terms(source)
    if not source_terms:
        return 1.0
    candidate_terms = content_terms(candidate)
    return round(len(source_terms & candidate_terms) / max(1, len(source_terms)), 3)


def evaluate_quality(
    current_text: str,
    candidate_text: str,
    meta: dict | None,
    deps: Any,
    *,
    min_term_preservation: float = 0.86,
) -> dict:
    meta = meta if isinstance(meta, dict) else {}
    reject_reasons: list[str] = []
    source_words = deps.text_word_count(current_text)
    candidate_words = deps.text_word_count(candidate_text)
    if source_words and candidate_words < max(20, int(source_words * 0.78)):
        reject_reasons.append("over_compressed_document")
    if source_words and candidate_words > int(source_words * 1.08) + 10:
        reject_reasons.append("unwanted_expansion")
    source_paragraphs = logical_paragraphs(current_text)
    candidate_paragraphs = logical_paragraphs(candidate_text)
    if len(source_paragraphs) >= 3 and len(candidate_paragraphs) < max(1, int(len(source_paragraphs) * 0.70)):
        reject_reasons.append("paragraph_role_collapse")
    term_ratio = _term_preservation_ratio(current_text, candidate_text)
    if term_ratio < min_term_preservation:
        reject_reasons.append(f"required_claim_terms_lost {term_ratio:.3f}<{min_term_preservation:.3f}")
    source_anchor_count = len(protected_anchor_terms(current_text))
    candidate_anchor_count = len(protected_anchor_terms(candidate_text))
    if candidate_anchor_count < source_anchor_count:
        reject_reasons.append("source_alignment_anchor_count_regressed")
    candidate_sentences = split_sentences(candidate_text)
    if candidate_sentences:
        very_short = sum(1 for sentence in candidate_sentences if deps.text_word_count(sentence) <= 3)
        if very_short / max(1, len(candidate_sentences)) > 0.18:
            reject_reasons.append("readability_fragmentation")
    if re.search(r"\b(?:I|my|we|our)\b", candidate_text) and not re.search(r"\b(?:I|my|we|our)\b", current_text):
        reject_reasons.append("synthetic_personal_voice_added")
    if re.search(r"\b(?:sort of|kind of|you know|honestly)\b", candidate_text, re.I):
        reject_reasons.append("academic_tone_degraded")
    malformed = evaluate_text_quality_regression(current_text, candidate_text)
    if not malformed.get("passed", True):
        reject_reasons.extend(malformed.get("reject_reasons") or ["malformed_text_regression"])
    return {
        "passed": not reject_reasons,
        "reject_reasons": sorted(set(reject_reasons)),
        "term_preservation_ratio": term_ratio,
        "source_words": source_words,
        "candidate_words": candidate_words,
        "paragraph_count_before": len(source_paragraphs),
        "paragraph_count_after": len(candidate_paragraphs),
        "malformed_text_guard": malformed,
        "quality_guard": {
            "argument_continuity": "pass" if not reject_reasons else "fail",
            "min_required_claims_preserved": term_ratio >= min_term_preservation,
            "paragraph_role_preserved": "paragraph_role_collapse" not in reject_reasons,
            "readability_not_degraded": "readability_fragmentation" not in reject_reasons,
            "academic_tone": "academic_tone_degraded" not in reject_reasons,
            "source_alignment": "source_alignment_anchor_count_regressed" not in reject_reasons,
        },
    }
