from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Pattern


@dataclass(frozen=True)
class GenericAssertionCompilerDeps:
    env_flag: Callable[[str, bool], bool]
    float_env: Callable[[str, float], float]
    blocker_scores: Callable[[dict | None], dict]
    logical_paragraphs: Callable[[str], list[str]]
    join_logical_paragraphs: Callable[[list[str]], str]
    split_sentences: Callable[[str], list[str]]
    text_word_count: Callable[[str], int]
    paragraph_component_targets: Callable[..., list[dict]]
    paragraph_role: Callable[..., str]
    safe_index: Callable[[Any, int], int]
    narrow_generic_claim_text: Callable[[str], str]
    paragraph_citation_re: Pattern[str]
    generic_terms_re: Pattern[str]
    generic_protected_sentence_re: Pattern[str]


def generic_assertion_sentence_score(sentence: str, deps: GenericAssertionCompilerDeps) -> float:
    words = max(1, deps.text_word_count(sentence))
    generic_hits = len(deps.generic_terms_re.findall(sentence))
    modal_hits = len(re.findall(r"\b(?:should|must|need(?:s|ed)?|can|will|may)\b", sentence, flags=re.I))
    broad_noun_hits = len(re.findall(
        r"\b(?:people|participants?|users?|readers?|writers?|work(?:ers)?|"
        r"teams?|systems?|process(?:es)?|practice|skills?|tasks?)\b",
        sentence,
        flags=re.I,
    ))
    protected_hits = len(deps.generic_protected_sentence_re.findall(sentence))
    return generic_hits * 2.4 + modal_hits * 1.2 + broad_noun_hits * 0.35 + words / 45.0 - protected_hits * 2.6


def generic_assertion_compiler_candidates(
    source_text: str,
    raw_json: dict | None,
    *,
    limit: int = 4,
    deps: GenericAssertionCompilerDeps,
) -> list[tuple[str, str, dict]]:
    """Compile broad generic-assertion blockers into bounded deterministic candidates."""
    if not deps.env_flag("DRAFTPROOF_GENERIC_ASSERTION_COMPILER", True):
        return []
    blockers = deps.blocker_scores(raw_json)
    generic_risk = float(blockers.get("generic_assertion_risk") or 0.0)
    unsupported_risk = float(blockers.get("unsupported_claim_risk") or 0.0)
    broad_risk = float(blockers.get("broad_claim_risk") or 0.0)
    if max(generic_risk, unsupported_risk, broad_risk) < 60.0:
        return []

    paragraphs = deps.logical_paragraphs(source_text)
    if len(paragraphs) < 3:
        return []
    source_words = deps.text_word_count(source_text)
    min_ratio = deps.float_env("DRAFTPROOF_GENERIC_ASSERTION_MIN_WORD_RATIO", 0.75)
    min_words = max(1, int(source_words * min_ratio))
    targets = deps.paragraph_component_targets(source_text, raw_json or {}, limit=max(limit * 3, 8))
    targeted_indexes = {deps.safe_index(target.get("index"), -1) for target in targets}
    for index, paragraph in enumerate(paragraphs):
        if index in targeted_indexes:
            continue
        words = deps.text_word_count(paragraph)
        if words < 12:
            continue
        generic_hits = len(deps.generic_terms_re.findall(paragraph))
        if generic_hits < 2:
            continue
        drivers = {
            "rewrite_brief_count": 0,
            "predictability_score_sum": 0,
            "generic_assertion_hits": generic_hits,
            "concrete_anchor_hits": len(deps.generic_protected_sentence_re.findall(paragraph)),
            "source_gap": not bool(deps.paragraph_citation_re.search(paragraph)),
            "repeated_sentence_starters": 0,
            "word_count": words,
        }
        role = deps.paragraph_role(paragraph, drivers, is_last=index == len(paragraphs) - 1)
        score = generic_assertion_sentence_score(paragraph, deps)
        targets.append({
            "index": index,
            "paragraph": paragraph,
            "score": round(score + 60.0, 3),
            "raw_score": round(score, 3),
            "role": role,
            "drivers": drivers,
            "target_sentences": [],
            "problem_spans": [],
            "domain_anchors": [],
        })
    targets.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    candidates: list[tuple[str, str, dict]] = []
    seen: set[str] = {str(source_text or "").strip()}

    def add(strategy: str, candidate_paragraphs: list[str], meta: dict) -> None:
        candidate_text = deps.join_logical_paragraphs(candidate_paragraphs)
        normalized = candidate_text.strip()
        if not normalized or normalized in seen:
            return
        if deps.text_word_count(candidate_text) < min_words:
            return
        seen.add(normalized)
        candidates.append((
            strategy,
            candidate_text,
            {
                **meta,
                "generic_assertion_compiler": True,
                "blockers": {
                    "generic_assertion_risk": generic_risk,
                    "unsupported_claim_risk": unsupported_risk,
                    "broad_claim_risk": broad_risk,
                },
            },
        ))

    for target in targets:
        if len(candidates) >= max(1, limit):
            break
        index = int(target.get("index", 0) or 0)
        if index < 0 or index >= len(paragraphs):
            continue
        paragraph = paragraphs[index]
        drivers = target.get("drivers") or {}
        role = target.get("role") or deps.paragraph_role(paragraph, drivers, is_last=index == len(paragraphs) - 1)
        if role in {"human_anchor_rich", "technical_process_rich"}:
            continue
        sentences = deps.split_sentences(paragraph)
        if len(sentences) < 2:
            narrowed = deps.narrow_generic_claim_text(paragraph)
            if narrowed.strip() and narrowed.strip() != paragraph.strip():
                next_paragraphs = list(paragraphs)
                next_paragraphs[index] = narrowed
                add(
                    f"generic_assertion_narrow_p{index + 1}",
                    next_paragraphs,
                    {"operation": "narrow_single_block", "paragraph_index": index, "paragraph_role": role},
                )
            continue

        removable = []
        for sentence_index, sentence in enumerate(sentences):
            sentence_text = sentence.strip()
            if not sentence_text:
                continue
            if deps.generic_protected_sentence_re.search(sentence_text):
                continue
            score = generic_assertion_sentence_score(sentence_text, deps)
            generic_hits = len(deps.generic_terms_re.findall(sentence_text))
            if score >= 3.0 and generic_hits >= 1:
                removable.append((score, sentence_index, sentence_text))
        removable.sort(reverse=True)

        if removable:
            remove_count = 1 if len(sentences) < 5 else min(2, len(removable))
            remove_indexes = {idx for _score, idx, _sentence in removable[:remove_count]}
            kept = [sentence for sentence_index, sentence in enumerate(sentences) if sentence_index not in remove_indexes]
            if len(kept) >= 1:
                replacement = deps.narrow_generic_claim_text(" ".join(kept).strip())
                if replacement and replacement.strip() != paragraph.strip():
                    next_paragraphs = list(paragraphs)
                    next_paragraphs[index] = replacement
                    add(
                        f"generic_assertion_prune_p{index + 1}",
                        next_paragraphs,
                        {
                            "operation": "remove_generic_assertion_sentence",
                            "paragraph_index": index,
                            "paragraph_role": role,
                            "removed_sentence_indexes": sorted(remove_indexes),
                            "removed_sentences": [sentence for _score, _idx, sentence in removable[:remove_count]],
                            "drivers": drivers,
                        },
                    )

        narrowed = deps.narrow_generic_claim_text(paragraph)
        if narrowed.strip() and narrowed.strip() != paragraph.strip():
            next_paragraphs = list(paragraphs)
            next_paragraphs[index] = narrowed
            add(
                f"generic_assertion_narrow_p{index + 1}",
                next_paragraphs,
                {
                    "operation": "narrow_generic_claim_language",
                    "paragraph_index": index,
                    "paragraph_role": role,
                    "drivers": drivers,
                },
            )

    if len(candidates) < max(1, limit):
        changed = []
        combined = list(paragraphs)
        for target in targets[:6]:
            index = int(target.get("index", 0) or 0)
            if index < 0 or index >= len(combined):
                continue
            role = target.get("role") or deps.paragraph_role(
                combined[index],
                target.get("drivers") or {},
                is_last=index == len(combined) - 1,
            )
            if role in {"human_anchor_rich", "technical_process_rich"}:
                continue
            narrowed = deps.narrow_generic_claim_text(combined[index])
            if narrowed.strip() and narrowed.strip() != combined[index].strip():
                combined[index] = narrowed
                changed.append(index)
            if len(changed) >= 4:
                break
        if changed:
            add(
                "generic_assertion_multi_narrow",
                combined,
                {"operation": "multi_narrow_generic_claim_language", "paragraph_indexes": changed},
            )

    return candidates[:max(1, limit)]
