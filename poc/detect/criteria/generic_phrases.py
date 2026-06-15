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

# NO-HARDCODE: the curated formal-transition subset (therefore/however/moreover…) and the
# research-claim phrase list (research suggests/studies show…) were removed. Both are baked
# content/stylistic lists; "generic phrasing" is captured statistically by the predictability
# scanner's surprisal/top-k signals. Empty sets keep the criterion's logic intact (it matches
# nothing here and leans on the statistical criteria).
LOW_VALUE_TRANSITIONS: frozenset = frozenset()
RESEARCH_CLAIM_PHRASES: frozenset = frozenset()

# Citation proximity pattern
_CITE_PATTERN = re.compile(r'\([^)]*\d{4}[^)]*\)|\[\d+\]')


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

    # Context-aware weighting: not all phrases count equally
    weighted_count = 0.0
    low_value_counts = {}

    for p in matched_phrases:
        p_lower = p.lower().strip()

        # Low-value transitions: only count after 3+ occurrences
        if p_lower in LOW_VALUE_TRANSITIONS:
            low_value_counts[p_lower] = low_value_counts.get(p_lower, 0) + 1
            if low_value_counts[p_lower] >= 3:
                weighted_count += 0.3
            continue

        # Research-claim phrases: reduce weight if citation nearby
        if p_lower in RESEARCH_CLAIM_PHRASES:
            has_cite = bool(_CITE_PATTERN.search(content))
            weighted_count += 0.2 if has_cite else 1.0
            continue

        # Everything else: standard weight
        weighted_count += 1.0

    density = weighted_count / len(sentences)
    raw_phrase_count = len(matched_phrases)

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
            "generic_phrase_count": raw_phrase_count,
            "weighted_phrase_count": round(weighted_count, 1),
            "total_sentences": len(sentences),
            "density": round(density, 4),
            "raw_density": round(raw_phrase_count / len(sentences), 4),
            "threshold": density_threshold,
        },
        flagged_excerpts=flagged,
    )
