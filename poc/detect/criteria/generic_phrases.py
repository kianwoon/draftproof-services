"""Criterion: High generic phrase density.

Detects formulaic academic writing patterns that AI models produce at high
rates: generic transitions, safe hedges, broad conclusions, empty intensifiers.
"""

import re
from typing import Dict, List, Optional

from ._types import CriterionScore

# Import from unified source — single source of truth
from poc.predictability.phrase_packs import get_generic_phrases, get_phrases_for_packs

ALL_PHRASES = get_generic_phrases()


def _split_sentences(text: str) -> List[str]:
    from poc.detect.utils import split_sentences
    return split_sentences(text)


def score(
    content: str,
    *,
    custom_phrases: Optional[List[str]] = None,
    density_threshold: float = 0.30,
    **kwargs,
) -> CriterionScore:
    sentences = _split_sentences(content)
    if not sentences:
        return CriterionScore(
            name="generic_phrase_density",
            value=0.0,
            label="low",
            details={"density": 0.0},
        )

    phrases = custom_phrases or ALL_PHRASES
    text_lower = content.lower()

    matched_phrases = [p for p in phrases if p in text_lower]
    phrase_count = len(matched_phrases)
    density = phrase_count / len(sentences)

    # Normalise: density 0.5+ = very generic (value=1), 0 = no phrases (value=0)
    value = min(1.0, density / 0.5)

    if density >= density_threshold:
        label = "high"
    elif density >= density_threshold * 0.5:
        label = "medium"
    else:
        label = "low"

    # Find which sentences contain generic phrases
    flagged = []
    for sent in sentences:
        sent_lower = sent.lower()
        hits = [p for p in phrases if p in sent_lower]
        if hits:
            flagged.append(f'"{sent[:100]}" ({", ".join(hits[:3])})')
    flagged = flagged[:10]

    return CriterionScore(
        name="generic_phrase_density",
        value=round(value, 4),
        label=label,
        details={
            "generic_phrase_count": phrase_count,
            "total_sentences": len(sentences),
            "density": round(density, 4),
            "threshold": density_threshold,
        },
        flagged_excerpts=flagged,
    )
