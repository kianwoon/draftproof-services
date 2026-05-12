from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ContentPruningCandidateDeps:
    env_flag: Callable[[str, bool], bool]
    float_env: Callable[[str, float], float]
    logical_paragraphs: Callable[[str], list[str]]
    join_logical_paragraphs: Callable[[list[str]], str]
    split_sentences: Callable[[str], list[str]]
    text_word_count: Callable[[str], int]
    paragraph_component_targets: Callable[..., list[dict]]
    paragraph_role: Callable[..., str]
    detect_protected_spans: Callable[[str], list]


def content_pruning_candidates(
    source_text: str,
    raw_json: dict | None,
    *,
    limit: int = 4,
    deps: ContentPruningCandidateDeps,
) -> list[tuple[str, str, dict]]:
    """Create deletion/compression candidates for paragraphs that drag scores down."""
    if not deps.env_flag("DRAFTPROOF_CONTENT_PRUNING_REPAIR", True):
        return []
    paragraphs = deps.logical_paragraphs(source_text)
    if len(paragraphs) < 3:
        return []
    source_words = deps.text_word_count(source_text)
    min_words = max(1, int(source_words * deps.float_env("DRAFTPROOF_CONTENT_PRUNING_MIN_WORD_RATIO", 0.75)))
    targets = deps.paragraph_component_targets(source_text, raw_json or {}, limit=max(limit * 2, 4))
    protected = deps.detect_protected_spans(source_text)

    def paragraph_has_protected_anchor(index: int) -> bool:
        before = deps.join_logical_paragraphs(paragraphs[:index])
        start = len(before) + (2 if before else 0)
        end = start + len(paragraphs[index])
        return any(span.start_char >= start and span.end_char <= end for span in protected)

    candidates: list[tuple[str, str, dict]] = []
    seen_texts: set[str] = set()
    for target in targets:
        index = int(target.get("index", 0) or 0)
        if index < 0 or index >= len(paragraphs):
            continue
        paragraph = paragraphs[index]
        words = deps.text_word_count(paragraph)
        role = target.get("role") or deps.paragraph_role(paragraph, target.get("drivers") or {})
        drivers = target.get("drivers") or {}
        if role in {"human_anchor_rich", "technical_process_rich", "source_summary_heavy"}:
            continue
        if paragraph_has_protected_anchor(index):
            continue
        if words < 35:
            continue
        generic_hits = int(drivers.get("generic_assertion_hits") or 0)
        concrete_hits = int(drivers.get("concrete_anchor_hits") or 0)
        generic_density = generic_hits / max(words / 100.0, 1.0)
        source_gap = bool(drivers.get("source_gap"))
        if generic_density < 3.0 and not source_gap and role != "conclusion_template_risk":
            continue

        variants: list[tuple[str, str, dict]] = []
        delete_paragraphs = list(paragraphs)
        delete_paragraphs.pop(index)
        delete_text = deps.join_logical_paragraphs(delete_paragraphs)
        if deps.text_word_count(delete_text) >= min_words:
            variants.append((
                f"content_pruning_delete_p{index + 1}",
                delete_text,
                {
                    "operation": "delete_paragraph",
                    "paragraph_index": index,
                    "paragraph_role": role,
                    "removed_words": words,
                    "drivers": drivers,
                },
            ))

        sentences = deps.split_sentences(paragraph)
        if len(sentences) >= 3:
            sentence_scores = []
            for sentence_index, sentence in enumerate(sentences):
                sentence_words = deps.text_word_count(sentence)
                generic_sentence_hits = len(re.findall(
                    r"\b(?:important|significant|should|must|need(?:s)?|can|will|"
                    r"helps?|allows?|enables?|creates?|means|shows?|suggests?|"
                    r"highlights?|underscores?)\b",
                    sentence,
                    flags=re.I,
                ))
                concrete_sentence_hits = len(re.findall(
                    r"\b(?:\d+(?:\.\d+)?%?|\bI\b|\bmy\b|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|"
                    r"source|citation|reference|example|evidence|case|condition)\b",
                    sentence,
                    flags=re.I,
                ))
                sentence_scores.append((
                    generic_sentence_hits * 2.0
                    + sentence_words / 35.0
                    - concrete_sentence_hits * 0.8,
                    sentence_index,
                    sentence,
                ))
            sentence_scores.sort(reverse=True)
            remove_count = 1 if len(sentences) < 5 else 2
            remove_indexes = {idx for _score, idx, _sentence in sentence_scores[:remove_count]}
            kept_sentences = [
                sentence for idx, sentence in enumerate(sentences)
                if idx not in remove_indexes
            ]
            compressed = " ".join(kept_sentences).strip()
            if (
                compressed
                and compressed != paragraph
                and deps.text_word_count(compressed) >= max(25, int(words * 0.45))
            ):
                compressed_paragraphs = list(paragraphs)
                compressed_paragraphs[index] = compressed
                compressed_text = deps.join_logical_paragraphs(compressed_paragraphs)
                if deps.text_word_count(compressed_text) >= min_words:
                    variants.append((
                        f"content_pruning_compress_p{index + 1}",
                        compressed_text,
                        {
                            "operation": "compress_paragraph",
                            "paragraph_index": index,
                            "paragraph_role": role,
                            "removed_sentence_indexes": sorted(remove_indexes),
                            "removed_words": words - deps.text_word_count(compressed),
                            "drivers": drivers,
                        },
                    ))

        for strategy, candidate_text, meta in variants:
            if candidate_text.strip() == source_text.strip() or candidate_text in seen_texts:
                continue
            seen_texts.add(candidate_text)
            candidates.append((strategy, candidate_text, meta))
            if len(candidates) >= max(1, limit):
                return candidates
    return candidates
