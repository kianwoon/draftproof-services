from __future__ import annotations

import re

from rewrite_pipeline_core.config import _float_env
from rewrite_pipeline_core.phases.paragraph_targets import (
    PARAGRAPH_CITATION_RE as _PARAGRAPH_CITATION_RE,
    ParagraphTargetDeps,
    orphan_heading_reason as _core_orphan_heading_reason,
)
from rewrite_pipeline_core.text_processing.quality_artifacts import (
    _DANGLING_FRAGMENT_JOIN_RE,
    _SYNTHETIC_ANCHOR_RE,
    _external_detector_style_artifact_reason,
    _normalize_known_heading_boundaries,
    _repeated_long_sequence_reason,
    _repeated_sentence_opening_reason,
    _synthetic_meta_anchor_artifact_reason,
)
from rewrite_pipeline_core.text_processing.text_utils import (
    _logical_paragraphs,
    _normalize_protected_text,
    _protected_number_set,
    _text_word_count,
)



def _quality_orphan_heading_reason(text: str) -> str:
    return _core_orphan_heading_reason(
        text,
        deps=ParagraphTargetDeps(
            logical_paragraphs=_logical_paragraphs,
            text_word_count=_text_word_count,
            float_env=_float_env,
            env_flag=lambda _name, default=False: bool(default),
        ),
    )
def _ai_candidate_quality_reject_reason(
    candidate: str,
    *,
    allow_repeated_long_sequence: bool = False,
) -> str:
    if not isinstance(candidate, str) or not candidate.strip():
        return "empty_candidate"
    if "[[REVIEW:" in candidate:
        return "review_markers_not_auto_kept"
    synthetic_meta_anchor = _synthetic_meta_anchor_artifact_reason(candidate)
    if synthetic_meta_anchor:
        return synthetic_meta_anchor
    synthetic_anchors = _SYNTHETIC_ANCHOR_RE.findall(candidate)
    sentence_count = max(1, len(re.findall(r"(?<=[.!?])\s+", candidate)) + 1)
    max_anchor_count = max(3, min(8, sentence_count // 8))
    for artifact in (
        "For this task:",
        "During the practical work:",
        "During feedback:",
        "When the task is underway:",
        "In assessment:",
    ):
        if re.search(r"\b" + re.escape(artifact), candidate, re.I):
            return f"synthetic_anchor_artifact:{artifact}"
    if len(synthetic_anchors) > max_anchor_count:
        return f"synthetic_anchor_overuse {len(synthetic_anchors)}>{max_anchor_count}"
    lowered = candidate.lower()
    if re.search(r"\bith only\b", lowered):
        return "broken_word_fragment"
    if re.search(r"\b(?:introduction|conclusion)[ \t]+(?:inclusive|this|the)\b", candidate, re.I):
        return "heading_merged_into_sentence"
    orphan_heading = _quality_orphan_heading_reason(candidate)
    if orphan_heading:
        return orphan_heading
    if (
        re.search(
            r"\b(?:(?:research|studies|evidence)\s+(?:shows?|suggests?|indicates?|finds?)|"
            r"a\s+study\s+(?:shows?|suggests?|indicates?|finds?))\b",
            candidate,
            re.I,
        )
        and not _PARAGRAPH_CITATION_RE.search(candidate)
        and not re.search(r"https?://|doi\.org", candidate, re.I)
    ):
        return "unsupported_source_attribution"
    if _DANGLING_FRAGMENT_JOIN_RE.search(candidate):
        return "dangling_sentence_fragment_join"
    external_style_artifact = _external_detector_style_artifact_reason(candidate)
    if external_style_artifact:
        return external_style_artifact
    if not allow_repeated_long_sequence:
        repeated = _repeated_long_sequence_reason(candidate)
        if repeated:
            return repeated
        repeated_opening = _repeated_sentence_opening_reason(candidate)
        if repeated_opening:
            return repeated_opening

    sentences = [
        re.sub(r"\s+", " ", s.strip()).lower()
        for s in re.split(r"(?<=[.!?])\s+", candidate)
        if len(s.split()) >= 8
    ]
    seen = set()
    duplicates = 0
    for sentence in sentences:
        if sentence in seen:
            duplicates += 1
        seen.add(sentence)
    if duplicates:
        return "duplicated_sentence_fragments"
    return ""


def _ai_search_signal_brief(raw_json: dict) -> str:
    badge = (raw_json or {}).get("ai_risk_badge") or {}
    ai_components = badge.get("ai_components") or {}
    writing_components = badge.get("writing_components") or {}
    parts = []
    if isinstance(ai_components, dict):
        ranked = sorted(
            ((k, v) for k, v in ai_components.items() if isinstance(v, (int, float))),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked:
            parts.append("AI component drivers: " + ", ".join(f"{k}={v:.2f}%" for k, v in ranked[:8]))
    if isinstance(writing_components, dict):
        ranked = sorted(
            ((k, v) for k, v in writing_components.items() if isinstance(v, (int, float))),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked:
            parts.append("Writing-risk context: " + ", ".join(f"{k}={v:.2f}%" for k, v in ranked[:8]))
    suggestions = (((raw_json or {}).get("rewrite_guidance") or {}).get("guided_revision") or {}).get("risk_mitigation_actions") or []
    action_lines = []
    for item in suggestions[:6]:
        if isinstance(item, dict):
            title = item.get("title") or item.get("action_type")
            pattern = item.get("safe_edit_pattern")
            if title and pattern:
                action_lines.append(f"{title}: {pattern}")
    if action_lines:
        parts.append("Scanner rewrite actions: " + " | ".join(action_lines))
    return "\n".join(parts)


def _source_repair_brief(source_text: str) -> str:
    """Describe visible source damage the full-document rewrite must repair."""
    if not isinstance(source_text, str) or not source_text.strip():
        return ""
    notes = []
    lowered = source_text.lower()
    if "ith only" in lowered:
        notes.append(
            "The source already contains a broken word fragment like 'ith only'. "
            "Repair it from context instead of preserving the typo."
        )
    if re.search(r"\b(?:introduction|conclusion)\s+(?:inclusive|this|the)\b", source_text, re.I):
        notes.append(
            "Some headings appear merged into following sentences. Keep the same headings/content, "
            "but separate merged heading text cleanly."
        )

    sentences = [
        re.sub(r"\s+", " ", s.strip()).lower()
        for s in re.split(r"(?<=[.!?])\s+", source_text)
        if len(s.split()) >= 8
    ]
    seen = set()
    duplicate_count = 0
    for sentence in sentences:
        if sentence in seen:
            duplicate_count += 1
        seen.add(sentence)
    if duplicate_count:
        notes.append(
            f"The source appears to contain {duplicate_count} accidental repeated sentence(s). "
            "Keep one clean version and remove duplicated fragments."
        )
    if not notes:
        return ""
    return "Source repair requirements:\n- " + "\n- ".join(notes)


def _repair_candidate_source_damage(candidate: str) -> tuple[str, list[str]]:
    """Repair obvious inherited source corruption before candidate gates."""
    if not isinstance(candidate, str) or not candidate:
        return candidate, []
    repaired = candidate
    repairs = []

    next_text = re.sub(r"\bith only\b", "With only", repaired, flags=re.I)
    if next_text != repaired:
        repaired = next_text
        repairs.append("fixed_broken_with_fragment")

    repaired, heading_repairs = _normalize_known_heading_boundaries(repaired)
    repairs.extend(heading_repairs)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", repaired) if p.strip()]
    if paragraphs:
        seen_sentences = set()
        removed_duplicates = 0
        removed_fragments = 0
        rebuilt_paragraphs = []
        for paragraph in paragraphs:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
            if not sentences:
                rebuilt_paragraphs.append(paragraph)
                continue
            kept = []
            normalized_sentences = [
                re.sub(r"\s+", " ", s).strip().lower()
                for s in sentences
            ]
            for sentence_index, sentence in enumerate(sentences):
                key = re.sub(r"\s+", " ", sentence).strip().lower()
                if len(sentence.split()) >= 8 and any(
                    prior
                    and prior != key
                    and len(prior.split()) >= 8
                    and prior in key
                    for prior in seen_sentences
                ):
                    removed_fragments += 1
                    repairs.append("removed_dangling_prefix:embedded_prior_sentence")
                    continue
                first_alpha = re.search(r"[A-Za-z]", sentence)
                if first_alpha and first_alpha.group(0).islower() and len(sentence.split()) >= 2:
                    contained_elsewhere = any(
                        other_index != sentence_index
                        and key
                        and key in other
                        for other_index, other in enumerate(normalized_sentences)
                    ) or any(key and key in prior for prior in seen_sentences)
                    if contained_elsewhere:
                        removed_fragments += 1
                        continue
                if len(sentence.split()) >= 4 and key in seen_sentences:
                    removed_duplicates += 1
                    continue
                if len(sentence.split()) >= 4:
                    seen_sentences.add(key)
                kept.append(sentence)
            if kept:
                rebuilt_paragraphs.append(" ".join(kept))
        if removed_duplicates:
            repaired = "\n\n".join(rebuilt_paragraphs)
            repairs.append(f"removed_duplicate_sentences:{removed_duplicates}")
        if removed_fragments:
            repaired = "\n\n".join(rebuilt_paragraphs)
            repairs.append(f"removed_duplicate_fragments:{removed_fragments}")

    repaired = re.sub(r"\n{3,}", "\n\n", repaired).strip()
    return repaired, repairs


def _source_repair_drift_false_positive(candidate: str, reasons: list[str]) -> bool:
    """Return True only for named-entity drift caused by source-damage repair."""
    if not isinstance(candidate, str) or not reasons:
        return False
    candidate_l = re.sub(r"\s+", " ", candidate).lower()
    for reason in reasons:
        match = re.match(r"lost_named_entity:\s+'([^']+)'", str(reason))
        if not match:
            return False
        entity = re.sub(r"\s+", " ", match.group(1)).strip()
        entity_l = entity.lower()
        if entity_l in candidate_l:
            continue
        words = [
            word.lower()
            for word in re.findall(r"\b[A-Za-z][A-Za-z]+\b", entity)
            if word.lower() not in {"introduction", "conclusion"}
        ]
        if words and all(word in candidate_l for word in words):
            continue
        return False
    return True


_AI_SEARCH_ENTITY_NOISE = {
    "the", "this", "these", "that", "with", "when", "while", "where",
    "people", "person", "participants", "participant", "users", "user",
    "process", "practice", "context",
    "introduction", "conclusion", "however", "therefore", "because",
    "centre", "center",
}


def _ai_search_drift_false_positive(candidate: str, reasons: list[str], similarity: float) -> bool:
    """Relax entity-only drift noise for high-similarity full-document candidates."""
    if not isinstance(candidate, str) or not reasons or similarity < 0.90:
        return False
    candidate_l = re.sub(r"\s+", " ", candidate).lower()
    for reason in reasons:
        match = re.match(r"lost_named_entity:\s+'([^']+)'", str(reason))
        if not match:
            return False
        entity = re.sub(r"\s+", " ", match.group(1)).strip()
        entity_l = entity.lower()
        if entity_l in candidate_l:
            continue
        words = [word.lower() for word in re.findall(r"\b[A-Za-z][A-Za-z]+\b", entity)]
        if not words:
            return False
        if any(word in {"introduction", "conclusion"} for word in words):
            # Full-document drift sees repaired headings such as
            # "Topic Introduction Detail" as lost entities. The
            # protected-span check has already guarded citations/numbers/quotes;
            # this is layout damage, not semantic loss.
            continue
        if all(word in _AI_SEARCH_ENTITY_NOISE for word in words):
            continue
        content_words = [
            word
            for word in words
            if word not in _AI_SEARCH_ENTITY_NOISE
        ]
        if content_words and all(word in candidate_l for word in content_words):
            continue
        return False
    return True


_AI_SEARCH_CRITICAL_ENTITY_RE = re.compile(
    r"\b(?:[A-Z]{2,}[A-Z0-9-]{2,}|\d{3,4}|"
    r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+"
    r"(?:Institute|Institution|University|College|Centre|Center|School|Department|Agency|Authority|Council|Group|Corporation|Company))\b",
)


def _ai_search_entity_drift_scan_allowed(candidate: str, reasons: list[str], similarity: float) -> bool:
    """Allow scoring high-similarity candidates with only non-critical entity drift.

    This does not relax protected spans. It only prevents the scoring loop from
    throwing away otherwise useful full-document candidates because the generic
    drift guard misread sentence starts or repaired headings as named entities.
    """
    if not isinstance(candidate, str) or not reasons or similarity < 0.92:
        return False
    candidate_l = re.sub(r"\s+", " ", candidate).lower()
    for reason in reasons:
        match = re.match(r"lost_named_entity:\s+'([^']+)'", str(reason))
        if not match:
            return False
        entity = re.sub(r"\s+", " ", match.group(1)).strip()
        if entity.lower() in candidate_l:
            continue
        words = [word.lower() for word in re.findall(r"\b[A-Za-z][A-Za-z]+\b", entity)]
        if words and all(word in _AI_SEARCH_ENTITY_NOISE for word in words):
            continue
        if _AI_SEARCH_CRITICAL_ENTITY_RE.search(entity):
            return False
    return True


def _ai_search_quote_drift_scan_allowed(candidate: str, reasons: list[str], similarity: float) -> bool:
    """Allow scoring quote-marker drift after protected-span preservation passed."""
    if not isinstance(candidate, str) or not reasons or similarity < 0.70:
        return False
    return all(str(reason).startswith("quote_lost:") for reason in reasons)


def _document_recreate_drift_scan_allowed(
    candidate: str,
    reasons: list[str],
    similarity: float,
    extra: dict | None,
) -> bool:
    """Allow scanner scoring for recreate/remove candidates with example-entity drift.

    Recreate candidates are allowed to remove low-value examples. The protected
    span gate already ran before semantic drift, so numbers, citations, and
    exact protected strings remain guarded. This only prevents named-example
    drift from blocking the scanner from measuring the actual AI footprint.
    """
    if not (
        isinstance(candidate, str)
        and isinstance(extra, dict)
        and reasons
        and similarity >= _float_env("DRAFTPROOF_RECREATE_DRIFT_SCAN_MIN_SIMILARITY", 0.45)
        and (
            extra.get("internet_reinforced_reauthoring")
            or extra.get("strict_safe_candidate")
            or extra.get("post_topk_optimizer")
        )
    ):
        return False
    candidate_l = re.sub(r"\s+", " ", candidate).lower()
    for reason in reasons:
        reason_s = str(reason or "")
        if not reason_s.startswith(("lost_named_entity:", "new_named_entity:")):
            return False
        match = re.match(r"(?:lost_named_entity|new_named_entity):\s+'([^']+)'", reason_s)
        if not match:
            return False
        entity = re.sub(r"\s+", " ", match.group(1)).strip()
        if entity.lower() in candidate_l:
            continue
        words = [word.lower() for word in re.findall(r"\b[A-Za-z][A-Za-z]+\b", entity)]
        if words and all(word in _AI_SEARCH_ENTITY_NOISE for word in words):
            continue
        if _AI_SEARCH_CRITICAL_ENTITY_RE.search(entity):
            return False
    return True


def _reconstruction_drift_scan_allowed(candidate: str, reasons: list[str], similarity: float) -> bool:
    """Allow reconstruction scoring for non-substantive drift noise.

    The protected-span check runs immediately before this function. If it has
    passed, quote text, numbers, and citation names are still present. The
    keyword drift guard may still report quote loss when curly/straight quote
    marks differ or when the phrase is preserved without quotation marks; that
    should not block scanner scoring for reconstruction candidates.
    """
    if not isinstance(candidate, str) or not reasons:
        return False
    if all(str(reason).startswith("quote_lost:") for reason in reasons):
        return similarity >= 0.70
    if similarity < 0.78:
        return False
    candidate_l = re.sub(r"\s+", " ", candidate).lower()
    for reason in reasons:
        if str(reason).startswith("quote_lost:"):
            continue
        match = re.match(r"lost_named_entity:\s+'([^']+)'", str(reason))
        if not match:
            return False
        entity = re.sub(r"\s+", " ", match.group(1)).strip()
        if entity.lower() in candidate_l:
            continue
        if _AI_SEARCH_CRITICAL_ENTITY_RE.search(entity):
            return False
        words = [word.lower() for word in re.findall(r"\b[A-Za-z][A-Za-z]+\b", entity)]
        if not words or any(word not in _AI_SEARCH_ENTITY_NOISE for word in words):
            return False
    return True


def _protected_code_anchor_set(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b", str(text or ""))
    }


def _normalize_direct_quote_content(text: str) -> str:
    normalized = _normalize_protected_text(text).strip()
    normalized = normalized.strip('"').strip("'").strip()
    return normalized.strip(" ,.;:!?")


def _ai_search_protected_loss_reason(original: str, candidate: str, protected) -> str:
    """Lenient protected-span check for full-document AI candidates.

    The generic protected-span guard is byte-exact. That is too strict for
    AI-search candidates because the detector currently marks punctuation
    fragments such as ", 2017" and ". 149" as protected citations. For this
    stage, preserve the substance: numbers, quote content, and citation names.
    """
    candidate_norm = _normalize_protected_text(candidate)
    candidate_numbers = _protected_number_set(candidate)

    for number in sorted(_protected_number_set(original)):
        if number not in candidate_numbers:
            return f"number_lost:{number}"

    for code_anchor in sorted(_protected_code_anchor_set(original)):
        if code_anchor not in candidate_norm:
            return f"code_anchor_lost:{code_anchor}"

    for span in protected or []:
        span_text = original[span.start_char:span.end_char]
        span_norm = _normalize_protected_text(span_text).strip('"').strip("'")
        if not span_norm:
            continue
        if span.reason == "direct_quote":
            quote_norm = _normalize_direct_quote_content(span_text)
            quote_words = [
                word.lower()
                for word in re.findall(r"\b[A-Za-z][A-Za-z]+\b", quote_norm)
                if word.lower() not in {"the", "a", "an", "is", "are", "was", "were", "did", "do", "does"}
            ]
            quote_first = (quote_words[0] if quote_words else "").lower()
            short_question_prompt = bool(
                len(quote_words) <= 10
                and not _protected_number_set(span_text)
                and (
                    "?" in span_text
                    or quote_first in {"what", "how", "why", "when", "where", "who", "which", "can", "could", "should", "would"}
                    or _normalize_direct_quote_content(span_text).lower().startswith("did ")
                )
            )
            if short_question_prompt:
                continue
            if (
                quote_norm
                and quote_norm not in candidate_norm
                and "?" in span_text
                and len(quote_words) <= 10
                and quote_words
                and sum(1 for word in quote_words if word in candidate_norm) / max(len(quote_words), 1) >= 0.60
            ):
                continue
            if quote_norm and quote_norm not in candidate_norm:
                return f"quote_lost:{quote_norm[:40]}"
            continue
        if span.reason == "citation":
            names = [
                name.lower()
                for name in re.findall(r"\b[A-Z][A-Za-z]{2,}\b", span_text)
                if name.lower() not in {"pp"}
            ]
            missing_names = [name for name in names if name not in candidate_norm]
            if missing_names:
                return f"citation_name_lost:{missing_names[0]}"

    return ""

