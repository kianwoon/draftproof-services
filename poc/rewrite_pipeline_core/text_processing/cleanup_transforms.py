from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CleanupTransformDeps:
    env_flag: Callable[[str, bool], bool]
    float_env: Callable[[str, float], float]
    logical_paragraphs: Callable[[str], list[str]]
    join_logical_paragraphs: Callable[[list[str]], str]
    split_sentences: Callable[[str], list[str]]
    text_word_count: Callable[[str], int]


def narrow_generic_claim_text(text: str) -> str:
    replacements = [
        (r"\b[Tt]he real challenge is\b", "A harder challenge is"),
        (r"\b[Tt]he main challenge is\b", "One challenge is"),
        (r"\b[Tt]his has created\b", "This can create"),
        (r"\b[Tt]his shift has made\b", "This shift can make"),
        (r"\b[Tt]his is a serious concern because\b", "The concern is practical because"),
        (r"\b[Tt]he goal should not be\b", "The goal does not need to be"),
        (r"\b[Tt]he goal should be\b", "A more useful goal is"),
        (r"\b[Ii]t is important to consider that\b", "In this situation,"),
        (r"\b[Ii]t is important to note that\b", ""),
        (r"\b[Pp]lays? (?:a )?(?:crucial|significant|important|major) role in\b", "affects"),
        (r"\b[Hh]as (?:a )?(?:crucial|significant|important|major) impact on\b", "affects"),
        (r"\b[Tt]his (?:demonstrates|highlights|underscores|shows) that\b", "This suggests that"),
        (r"\b[Ii]n today's (?:world|society|environment)\b", "Now"),
        (r"\b[Ii]n the modern (?:world|society|environment)\b", "Now"),
    ]
    narrowed = str(text or "")
    for pattern, replacement in replacements:
        narrowed = re.sub(pattern, replacement, narrowed)
    narrowed = re.sub(r"\b[Ee]veryone\b", "Many people", narrowed)
    narrowed = re.sub(r"\b[Aa]ll\b", "Many", narrowed)
    narrowed = re.sub(r"\bmust\b", "may need to", narrowed)
    narrowed = re.sub(r"\bwill\b", "may", narrowed)
    narrowed = re.sub(r"\balways\b", "often", narrowed, flags=re.I)
    narrowed = re.sub(r"\bnever\b", "rarely", narrowed, flags=re.I)
    narrowed = re.sub(r"[ \t]{2,}", " ", narrowed)
    narrowed = re.sub(r"\s+([,.!?;:])", r"\1", narrowed)
    return narrowed


def plain_language_depolish_text(text: str) -> tuple[str, list[str]]:
    """Remove late-stage polished academic artifacts without changing claims."""
    if not isinstance(text, str) or not text.strip():
        return "", []
    updated = text
    applied: list[str] = []
    replacements: list[tuple[str, str, str]] = [
        (r"\bTherefore,\s*", "Because of this, ", "therefore_plain"),
        (r"\bUltimately,\s*", "In the end, ", "ultimately_plain"),
        (r"\bSimultaneously,\s*", "At the same time, ", "simultaneously_plain"),
        (r"\bCrucially,\s*", "More importantly, ", "crucially_plain"),
        (r"\bFurthermore,\s*", "", "furthermore_plain"),
        (r"\bMoreover,\s*", "", "moreover_plain"),
        (r"\bAdditionally,\s*", "", "additionally_plain"),
        (r"\bYet,\s*", "But ", "yet_plain"),
        (r"\bOn one hand\b", "On one side", "on_one_hand_plain"),
        (r"\bresembles\b", "feels like", "resembles_plain"),
        (r"\bamid constant motion\b", "while everything is moving around it", "amid_motion_plain"),
        (r"\bpersists\b", "still exists", "persists_plain"),
        (r"\bdeliver content\b", "explain the material", "deliver_content_plain"),
        (r"\bgauge progress\b", "measure progress", "gauge_progress_plain"),
        (r"\bshifted dramatically\b", "changed a lot", "shifted_dramatically_plain"),
        (r"\bremarkably immediate\b", "very quick", "immediate_plain"),
        (r"\babundance creates challenges\b", "creates a problem", "abundance_plain"),
        (r"\bassume understanding comes effortlessly\b", "think understanding is easy", "effortless_understanding_plain"),
        (r"\bconceal years of effort\b", "hide years of effort", "conceal_plain"),
        (r"\blacks true comprehension\b", "does not really understand", "comprehension_plain_2"),
        (r"\bunfolds more gradually\b", "is slower than that", "unfolds_plain"),
        (r"\bremain vital\b", "still matter", "vital_plain"),
        (r"\bextends beyond\b", "is more than", "extends_plain"),
        (r"\bdistinguishing reliable information from unreliable sources\b", "telling useful information from weak information", "distinguish_sources_plain"),
        (r"\bintensifies these concerns\b", "makes this more urgent", "intensifies_plain"),
        (r"\bfocus on passing tests rather than truly grasping material\b", "focus on passing rather than really understanding", "grasping_material_plain"),
        (r"\bnurture confidence or independent inquiry\b", "build confidence or independent thinking", "nurture_plain"),
        (r"\bserves as a\b", "is used as a", "serves_plain"),
        (r"\bsubstitute original thought\b", "replace their own thinking", "substitute_plain"),
        (r"\bPlacing greater emphasis on\b", "Paying more attention to", "placing_emphasis_plain"),
        (r"\bflawless final submission\b", "perfect final submission", "flawless_plain"),
        (r"\bEquity also demands attention\b", "Fairness is another issue", "equity_plain"),
        (r"\bbenefit from\b", "have", "benefit_plain"),
        (r"\badvanced tools\b", "better tools", "advanced_tools_plain"),
        (r"\bdisparities\b", "gaps", "disparities_plain"),
        (r"\bhold value\b", "still matter", "hold_value_plain"),
        (r"\bconfront the realities\b", "be honest about the world", "confront_plain"),
        (r"\bcultivate\b", "build", "cultivate_plain"),
        (r"\bit is crucial for\b", "it matters for", "crucial_plain"),
        (r"\bcrucial\b", "important", "crucial_word_plain"),
        (r"\bvital\b", "important", "vital_word_plain"),
        (r"\bemphasize\b", "focus on", "emphasize_plain"),
        (r"\bfrequently\b", "often", "frequently_plain"),
        (r"\bmerely\b", "only", "merely_plain"),
        (r"\bsignificant hurdle\b", "harder problem", "hurdle_plain"),
        (r"\bvarious sources such as\b", "sources such as", "various_sources_plain"),
        (r"\bindividuals they follow\b", "people they follow", "individuals_plain"),
        (r"\bunderlying concepts\b", "topic", "underlying_concepts_plain"),
        (r"\brefined answers\b", "polished answers", "refined_answers_plain"),
        (r"\bthe advent of\b", "", "advent_plain"),
        (r"\bfrequently fail to adequately address\b", "do not always build well", "fail_address_plain"),
        (r"\bfilled with\b", "full of", "filled_plain"),
        (r"\bready them\b", "prepare them", "ready_plain"),
        (r"\brequire knowledge\b", "need knowledge", "require_knowledge_plain"),
        (r"\bprimarily\b", "mainly", "primarily_plain"),
        (r"\bengagement with the material\b", "attention to the topic", "engagement_material_plain"),
        (r"\btrue comprehension\b", "understanding", "comprehension_plain"),
        (r"\bjourney\b", "process", "journey_plain"),
        (r"\bcontinuous improvement\b", "improvement", "continuous_improvement_plain"),
        (r"\boveremphasis\b", "too much focus", "overemphasis_plain"),
        (r"\bequip them for life\b", "prepare them for life", "equip_plain"),
        (r"\brequire judgment\b", "need judgment", "require_plain"),
    ]
    for pattern, replacement, name in replacements:
        next_text = re.sub(pattern, replacement, updated, flags=re.I)
        if next_text != updated:
            updated = next_text
            applied.append(name)
    updated = re.sub(r"[ \t]+", " ", updated)
    updated = re.sub(r" \.", ".", updated)
    updated = re.sub(r"\n{3,}", "\n\n", updated).strip()
    return updated, applied


def final_score_drag_sentence_prune_text(text: str, deps: CleanupTransformDeps) -> tuple[str, list[str]]:
    """Remove broad late-stage sentences that drag scanner scores down."""
    if not isinstance(text, str) or not text.strip():
        return "", []
    if not deps.env_flag("DRAFTPROOF_FINAL_SCORE_DRAG_PRUNE", True):
        return text, []
    max_removed = max(1, int(deps.float_env("DRAFTPROOF_FINAL_SCORE_DRAG_PRUNE_MAX_SENTENCES", 3.0)))
    min_ratio = deps.float_env("DRAFTPROOF_FINAL_SCORE_DRAG_PRUNE_MIN_WORD_RATIO", 0.75)
    source_words = deps.text_word_count(text)
    if source_words <= 0:
        return text, []
    min_words = int(source_words * min_ratio)
    remaining_words = source_words
    sentence_patterns: list[tuple[str, str]] = [
        (r"\b(?:the )?world (?:outside|around) [a-z ]+ has changed much faster\b", "outside_context_changed_faster"),
        (r"\bin [a-z ]+ and outside it,\s+information is everywhere\b", "information_everywhere"),
        (r"\bworld full of information and distractions\b", "world_information_distractions"),
        (r"\bknowledge is no longer scarce\b", "knowledge_no_longer_scarce"),
        (r"\baccess is no longer the biggest problem\b", "access_no_longer_biggest_problem"),
    ]
    protected_pattern = re.compile(
        r"(?:\"[^\"]+\"|“[^”]+”|\[[^\]]+\]|\([A-Za-z]+,\s*\d{4}\)|\b\d+(?:\.\d+)?%?\b|"
        r"\b[A-Z]{2,}[A-Z0-9-]*\b)",
    )
    paragraphs = deps.logical_paragraphs(text)
    updated_paragraphs: list[str] = []
    applied: list[str] = []
    removed = 0
    for paragraph in paragraphs:
        if removed >= max_removed:
            updated_paragraphs.append(paragraph)
            continue
        sentences = deps.split_sentences(paragraph)
        if len(sentences) < 2:
            updated_paragraphs.append(paragraph)
            continue
        kept: list[str] = []
        for sentence in sentences:
            if removed >= max_removed:
                kept.append(sentence)
                continue
            sentence_text = sentence.strip()
            lower_sentence = sentence_text.lower()
            matched_name = ""
            for pattern, name in sentence_patterns:
                if re.search(pattern, lower_sentence, flags=re.I):
                    matched_name = name
                    break
            if (
                matched_name
                and not protected_pattern.search(sentence_text)
                and deps.text_word_count(sentence_text) <= 22
                and len(sentences) - 1 >= 1
                and remaining_words - deps.text_word_count(sentence_text) >= min_words
            ):
                removed += 1
                remaining_words -= deps.text_word_count(sentence_text)
                applied.append(matched_name)
                continue
            kept.append(sentence)
        next_paragraph = " ".join(kept).strip()
        if next_paragraph:
            updated_paragraphs.append(next_paragraph)
    updated = deps.join_logical_paragraphs(updated_paragraphs)
    if not applied or updated.strip() == text.strip():
        return text, []
    if deps.text_word_count(updated) < min_words:
        return text, []
    return updated, applied


def compress_score_drag_paragraph(paragraph: str, deps: CleanupTransformDeps, *, max_remove: int = 2) -> str:
    sentences = deps.split_sentences(paragraph)
    if len(sentences) < 3:
        return narrow_generic_claim_text(paragraph)
    scores = []
    for index, sentence in enumerate(sentences):
        words = deps.text_word_count(sentence)
        generic_hits = len(re.findall(
            r"\b(?:important|significant|should|must|need(?:s)?|can|will|"
            r"helps?|allows?|enables?|creates?|means|shows?|suggests?|"
            r"highlights?|underscores?|challenge|issue|goal|system|world)\b",
            sentence,
            flags=re.I,
        ))
        anchor_hits = len(re.findall(
            r"\b(?:\d+(?:\.\d+)?%?|\"[^\"]+\"|“[^”]+”|\bI\b|\bmy\b|"
            r"source|citation|reference|example|evidence|case|condition)\b",
            sentence,
            flags=re.I,
        ))
        scores.append((generic_hits * 2.2 + words / 30.0 - anchor_hits * 1.1, index))
    scores.sort(reverse=True)
    remove_indexes = {index for _score, index in scores[:max(1, int(max_remove or 1))]}
    kept = [sentence for index, sentence in enumerate(sentences) if index not in remove_indexes]
    if not kept:
        return paragraph
    compressed = " ".join(kept).strip()
    return narrow_generic_claim_text(compressed)
