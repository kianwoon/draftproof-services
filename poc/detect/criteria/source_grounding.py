"""Criterion: Source grounding — detects unsupported factual claims.

Checks whether factual assertions in the text are backed by citations,
specific evidence, or domain grounding. A high unsupported-claim ratio
indicates the writing may lack source anchoring.

Algorithm:
1. Split text into sentences
2. Classify each as: factual_claim, supported_claim, or non_claim
3. Check citation proximity for each factual claim
4. Score = 1 - (unsupported_count / max(claim_count, 1))
   → 1.0 = all claims supported, 0.0 = none supported
"""

import re
from typing import Dict, List, Optional

from ._types import CriterionScore


# ── Citation markers ──────────────────────────────────────────────────

# Author-year: (Smith, 2023), (Smith & Jones, 2023), (Smith et al., 2023)
_CITE_AUTHOR_YEAR = re.compile(
    r'\([A-Z][a-z]+(?:\s+(?:&|and)\s+[A-Z][a-z]+)?'
    r'(?:\s+et\s+al\.)?,\s*\d{4}[a-z]?\)'
)

# Numeric: [1], [1,2], [1-3]
_CITE_NUMERIC = re.compile(r'\[[\d,\s\-–]+\]')

# Superscript-style: ¹ ² ³ or ^1 ^2
_CITE_SUPERSCRIPT = re.compile(r'[¹²³⁰-⁹]+|\^\d+')

# Narrative citation: Author (Year) — e.g., "Ingram and Jackson (2026)", "Smith (2023)"
_CITE_NARRATIVE = re.compile(
    r'[A-Z][a-z]+(?:\s+(?:and|&|et\s+al\.?)\s+[A-Z][a-z]+)*\s*\(\d{4}[a-z]?\)'
)

# Any citation pattern
_CITE_ANY = re.compile(
    f'(?:{_CITE_AUTHOR_YEAR.pattern}|{_CITE_NARRATIVE.pattern}|{_CITE_NUMERIC.pattern}|{_CITE_SUPERSCRIPT.pattern})'
)

# ── Factual claim indicators ──────────────────────────────────────────

# Sentences with these patterns are likely factual claims
_CAUSAL_PATTERNS = re.compile(
    r'\b(?:leads?\s+to|causes?|results?\s+in|contributes?\s+to|'
    r'increases?|decreases?|reduces?|improves?|affects?|'
    r'prevents?|enables?|requires?|necessitates?)\b',
    re.I,
)

# Statistical/quantitative content
_QUANTITATIVE = re.compile(
    r'\b\d+(?:\.\d+)?%\b'          # percentages
    r'|\b\d+(?:\.\d+)?\s*(?:times?|fold|percent)\b'  # multipliers
    r'|\b(?:one\s+in|two\s+in|three\s+in)\s+\d+\b'   # "one in 3"
    r'|\b(?:majority|minority|most|major)\s+of\b'     # "majority of"
)

# Hedged research claims without citation — high-risk unsupported
_HEDGED_CLAIM = re.compile(
    r'\b(?:research\s+(?:suggests?|shows?|indicates?|finds?)\s+(?:that\s+)?)'
    r'|\b(?:studies\s+(?:show|indicate|suggest|find|demonstrate)\s+(?:that\s+)?)'
    r'|\b(?:evidence\s+(?:suggests?|indicates?|shows?)\s+(?:that\s+)?)'
    r'|\b(?:it\s+(?:has\s+been|is)\s+(?:shown|established|proven|demonstrated)\s+(?:that\s+)?)'
    r'|\b(?:the\s+literature\s+(?:suggests?|indicates?|shows?))',
    re.I,
)

# ── Non-claim indicators (exclude these) ──────────────────────────────

_OPINION_MARKERS = re.compile(
    r'\b(?:I\s+(?:believe|think|feel|argue|contend|suggest|propose)|'
    r'in\s+my\s+(?:experience|opinion|view|practice)|'
    r'personally[,.]?\s+I|from\s+my\s+perspective)\b',
    re.I,
)

_META_COMMENTARY = re.compile(
    r'^(?:this\s+(?:review|paper|essay|article|section|chapter)|'
    r'the\s+(?:following|above)|'
    r'(?:first|second|third|finally),?\s+(?:this|we|I)\s+will|'
    r'(?:in\s+)?(?:this|the)\s+(?:section|part|chapter|review),?\s+|'
    r'by\s+(?:looking\s+at|examining|analyzing|considering)\s+|'
    r'(?:this|the)\s+(?:paper|review|article|essay)\s+(?:will|shall))',
    re.I,
)

_QUESTION = re.compile(r'\?\s*$')

_FIRST_PERSON = re.compile(r'\b(?:I|me|my|mine|myself)\b', re.I)


def _split_sentences(text: str) -> List[str]:
    from poc.detect.utils import split_sentences
    return split_sentences(text)


def _has_citation_nearby(text: str, sent_start: int, sent_end: int, window: int = 400) -> bool:
    """Check if a citation exists within `window` chars of the sentence."""
    lo = max(0, sent_start - window)
    hi = min(len(text), sent_end + window)
    region = text[lo:hi]
    return bool(_CITE_ANY.search(region))


def _get_sentence_positions(text: str, sentences: List[str]) -> List[tuple]:
    """Find (start, end) positions of each sentence in the original text."""
    positions = []
    search_start = 0
    for sent in sentences:
        idx = text.find(sent, search_start)
        if idx == -1:
            # Fallback: approximate
            idx = search_start
        positions.append((idx, idx + len(sent)))
        search_start = idx + len(sent)
    return positions


def score(content: str, **kwargs) -> CriterionScore:
    """Evaluate source grounding by checking claim-citation alignment.

    Returns CriterionScore where:
    - value = grounding score (1.0 = all claims supported, 0.0 = none)
    - label = "high" (well grounded) / "medium" / "low" (poorly grounded)
    """
    sentences = _split_sentences(content)
    if not sentences:
        return CriterionScore(
            name="source_grounding",
            value=1.0,
            label="high",
            details={"claim_count": 0, "unsupported_count": 0},
        )

    positions = _get_sentence_positions(content, sentences)

    claim_count = 0
    supported_count = 0
    unsupported_claims = []
    hedged_claims = []

    for i, (sent, (start, end)) in enumerate(zip(sentences, positions)):
        stripped = sent.strip()

        # Skip non-claims
        if _QUESTION.search(stripped):
            continue
        if _META_COMMENTARY.match(stripped):
            continue
        if _OPINION_MARKERS.search(stripped) and not _QUANTITATIVE.search(stripped):
            continue
        # Pure first-person without factual content → non-claim
        if _FIRST_PERSON.search(stripped) and not _QUANTITATIVE.search(stripped) and not _CAUSAL_PATTERNS.search(stripped):
            continue
        # Short sentences (< 8 words) unlikely to be substantive claims
        if len(stripped.split()) < 8:
            continue

        # Check if sentence contains a citation itself
        has_inline_cite = bool(_CITE_ANY.search(stripped))
        has_nearby_cite = _has_citation_nearby(content, start, end)

        # Detect factual claim indicators
        has_causal = bool(_CAUSAL_PATTERNS.search(stripped))
        has_quantitative = bool(_QUANTITATIVE.search(stripped))
        is_hedged_claim = bool(_HEDGED_CLAIM.search(stripped))

        # Must have at least one claim indicator
        is_claim = has_causal or has_quantitative or is_hedged_claim

        if not is_claim:
            continue

        claim_count += 1

        if has_inline_cite or has_nearby_cite:
            supported_count += 1
        else:
            excerpt = stripped
            unsupported_claims.append(excerpt)
            if is_hedged_claim:
                hedged_claims.append(excerpt)

    # Calculate grounding score
    if claim_count == 0:
        grounding_score = 1.0
    else:
        grounding_score = 1.0 - (len(unsupported_claims) / claim_count)

    grounding_score = max(0.0, min(1.0, grounding_score))

    # Label: invert — "high" label means well grounded (high score)
    if grounding_score >= 0.80:
        label = "high"
    elif grounding_score >= 0.50:
        label = "medium"
    else:
        label = "low"

    # Concern = 1 - grounding (for the authorship concern formula)
    concern = 1.0 - grounding_score

    return CriterionScore(
        name="source_grounding",
        value=round(concern, 4),  # concern score (0 = well grounded, 1 = no support)
        label=label,
        details={
            "claim_count": claim_count,
            "supported_count": supported_count,
            "unsupported_count": len(unsupported_claims),
            "hedged_claim_count": len(hedged_claims),
            "grounding_score": round(grounding_score, 4),
            "concern": round(concern, 4),
        },
        flagged_excerpts=unsupported_claims[:10],
    )
