"""Deterministic marked-grounding candidate generation."""

from __future__ import annotations

import re


def ai_search_marked_grounding_candidates(source_text: str) -> list[tuple[str, str]]:
    """Create deterministic marked-addition candidates for missing grounding.

    These are intentionally visible to the user. They target the detector's
    generic assertion and source-grounding drivers without inventing evidence.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", source_text.strip()) if p.strip()]
    paragraph_sentences = [
        [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
        for paragraph in paragraphs
    ]
    sentences = [sentence for paragraph in paragraph_sentences for sentence in paragraph]
    if len(sentences) < 8:
        return []

    concrete_re = re.compile(
        r"\b\d+\b|\baccording to\b|\bfor example\b|\bfor instance\b|"
        r"\bin my\b|\bI (?:saw|noticed|found|observed|learned|worked)\b|"
        r"\bwe (?:found|observed|measured|tested)\b|\"",
        re.I,
    )
    assertion_re = re.compile(
        r"\b(is|are|was|has|have|can|should|must|needs?|creates?|makes?|requires?|means)\b",
        re.I,
    )
    review_notes = [
        "[[REVIEW: For example, add the exact source, observed moment, or local detail that proves this point.]]",
        "[[REVIEW: Add the specific evidence behind this claim, or soften it if the evidence is only limited.]]",
        "[[REVIEW: Name the source or concrete example the reader should connect to this sentence.]]",
        "[[REVIEW: Add one real detail from the task, workplace, context, or observed situation before keeping this claim.]]",
        "[[REVIEW: If this is based on experience, add what was seen, who was involved, and what changed.]]",
        "[[REVIEW: Add a citation or replace this with a narrower claim the draft can support.]]",
        "[[REVIEW: Give one concrete process step here, not just the general conclusion.]]",
        "[[REVIEW: Add the limitation or condition under which this statement is true.]]",
    ]

    scored = []
    for index, sentence in enumerate(sentences):
        words = sentence.split()
        if len(words) < 10:
            continue
        has_concrete = bool(concrete_re.search(sentence))
        has_assertion = bool(assertion_re.search(sentence))
        score = (2 if has_assertion else 0) + (2 if not has_concrete else 0) + min(len(words) / 35, 1)
        if score >= 2:
            scored.append((score, index))
    scored.sort(reverse=True)
    target_indexes = sorted(index for _, index in scored[:8])
    if not target_indexes:
        return []

    concrete_prefixes = [
        "In practice, ",
        "For this task, ",
        "During the practical work, ",
        "In assessment, ",
        "When the task is underway, ",
        "During feedback, ",
    ]

    def build_marked(limit: int, label: str) -> tuple[str, str]:
        note_indexes = set(target_indexes[:limit])
        rebuilt_paragraphs = []
        flat_index = 0
        used = 0
        for paragraph in paragraph_sentences:
            rebuilt = []
            for sentence in paragraph:
                rebuilt.append(sentence)
                if flat_index in note_indexes:
                    rebuilt.append(review_notes[used % len(review_notes)])
                    used += 1
                flat_index += 1
            rebuilt_paragraphs.append(" ".join(rebuilt))
        return label, "\n\n".join(rebuilt_paragraphs)

    def _contextualize_sentence(sentence: str, prefix: str) -> str:
        stripped = sentence.strip()
        if not stripped:
            return stripped
        if re.match(
            r"^(?:this|it|these|they|people|participants|users|the process|"
            r"the task|the method|the challenge|the standard)\b",
            stripped,
            re.I,
        ):
            return prefix + stripped[0].lower() + stripped[1:]
        return prefix.rstrip(", ") + ": " + stripped

    def build_process_anchors(label: str, *, limit: int) -> tuple[str, str]:
        anchor_indexes = set(target_indexes[:limit])
        rebuilt_paragraphs = []
        flat_index = 0
        used = 0
        for paragraph in paragraph_sentences:
            rebuilt = []
            anchored_this_paragraph = False
            for sentence in paragraph:
                words = sentence.split()
                has_concrete = bool(concrete_re.search(sentence))
                should_anchor = (
                    flat_index in anchor_indexes
                    and not anchored_this_paragraph
                    and len(words) >= 8
                    and not has_concrete
                )
                if should_anchor:
                    prefix = concrete_prefixes[used % len(concrete_prefixes)]
                    rebuilt.append(_contextualize_sentence(sentence, prefix))
                    anchored_this_paragraph = True
                    used += 1
                else:
                    rebuilt.append(sentence)
                flat_index += 1
            rebuilt_paragraphs.append(" ".join(rebuilt))
        return label, "\n\n".join(rebuilt_paragraphs)

    return [
        build_process_anchors("deterministic_process_anchor_generic", limit=min(4, len(target_indexes))),
        build_process_anchors("deterministic_process_anchor_all", limit=min(8, len(target_indexes))),
        build_marked(min(4, len(target_indexes)), "deterministic_marked_grounding_light"),
        build_marked(min(8, len(target_indexes)), "deterministic_marked_grounding_strong"),
    ]
