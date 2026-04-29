"""Criterion: Repetitive sentence structure.

AI often repeats sentence forms: "This highlights...", "This demonstrates...",
"This shows...", "This suggests...".

Measures: repeated_starter_ratio, sentence_length_variance.
"""

import re
from typing import Dict, List

from ._types import CriterionScore


def _split_sentences(text: str) -> List[str]:
    from poc.detect.utils import split_sentences
    return split_sentences(text)


def _first_words(sentence: str, n: int = 3) -> str:
    words = re.findall(r'\b\w+\b', sentence.lower())
    return ' '.join(words[:n])


def score(
    content: str,
    *,
    repeated_starter_threshold: float = 0.25,
    **kwargs,
) -> CriterionScore:
    sentences = _split_sentences(content)
    if len(sentences) < 3:
        return CriterionScore(
            name="repetitive_structure",
            value=0.0,
            label="low",
            details={"note": "fewer than 3 sentences"},
        )

    # First-3-words pattern counting
    starters = [_first_words(s) for s in sentences]
    starter_counts: Dict[str, int] = {}
    for s in starters:
        starter_counts[s] = starter_counts.get(s, 0) + 1

    repeated_starters = sum(1 for s in starters if starter_counts[s] >= 2)
    repeated_starter_ratio = repeated_starters / len(starters)

    # Sentence length variance
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) > 1:
        mean_l = sum(lengths) / len(lengths)
        var_l = sum((l - mean_l) ** 2 for l in lengths) / len(lengths)
        length_cv = (var_l ** 0.5) / mean_l if mean_l > 0 else 0.0
    else:
        length_cv = 0.0

    # First-word-only repetition (e.g. "This..." appearing many times)
    first_words = [s.split()[0].lower() for s in sentences if s.split()]
    fw_counts: Dict[str, int] = {}
    for w in first_words:
        fw_counts[w] = fw_counts.get(w, 0) + 1
    max_fw_ratio = max(fw_counts.values()) / len(first_words) if first_words else 0.0

    # Composite score
    value = min(1.0, repeated_starter_ratio + max_fw_ratio * 0.5)

    if repeated_starter_ratio >= repeated_starter_threshold:
        label = "high"
    elif repeated_starter_ratio >= repeated_starter_threshold * 0.6:
        label = "medium"
    else:
        label = "low"

    # Find the most repeated starters for flagging
    flagged = [
        f'"{s}" (×{starter_counts[s]})'
        for s in sorted(set(starters), key=lambda x: starter_counts[x], reverse=True)
        if starter_counts[s] >= 2
    ][:5]

    return CriterionScore(
        name="repetitive_structure",
        value=round(value, 4),
        label=label,
        details={
            "repeated_starter_ratio": round(repeated_starter_ratio, 4),
            "max_first_word_ratio": round(max_fw_ratio, 4),
            "sentence_length_cv": round(length_cv, 4),
        },
        flagged_excerpts=flagged,
    )
