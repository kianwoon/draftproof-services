"""Shared utilities for the detect pipeline.

Provides abbreviation-aware sentence splitting and other text utilities
used across multiple detectors and criteria.
"""

import re
from typing import List, Tuple

# Abbreviations that should NOT trigger a sentence break
_ABBREVIATIONS = frozenset({
    # Titles
    "dr", "mr", "mrs", "ms", "mx", "prof", "sr", "jr", "st", "gen",
    "rep", "sen", "gov", "lt", "sgt", "capt", "cmdr", "rev", "hon",
    # Academic / Latin
    "e.g", "i.e", "etc", "vs", "cf", "al", "et", "ibid", "op", "cit",
    "ca", "circa", "approx", "def", "ref", "vol", "no", "pp", "ch",
    "fig", "tab", "eq", "sec", "app", "dept", "univ", "inst",
    # Geo / political
    "u.s", "u.k", "u.n", "e.u", "a.m", "p.m", "gmt", "est", "pst",
    # Common single-letter abbreviations
    "a", "b", "c", "d", "e", "f",  # used in list items like "a) ..."
})

# Pattern: abbreviation followed by period, not a sentence break
_ABBREV_PERIOD = re.compile(
    r'\b(' + '|'.join(re.escape(a) for a in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r')\.',
    re.IGNORECASE,
)

# Pattern: single uppercase letter followed by period (initials, "U.S.")
_INITIAL_PERIOD = re.compile(r'\b([A-Z])\.')

# Pattern: ellipsis
_ELLIPSIS = re.compile(r'\.{2,}')

# Sentence boundary: period/exclamation/question followed by whitespace and uppercase
_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"“”])'
)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, handling abbreviations and edge cases.

    Protects against false breaks on:
    - Abbreviations: "e.g.", "Dr.", "U.S."
    - Ellipsis: "..."
    - Quoted text: "He said 'hello.' Then left."

    Returns list of sentence strings with whitespace stripped.
    """
    if not text or not text.strip():
        return []

    # Phase 1: protect abbreviation periods — replace ALL dots in/after abbreviations
    # For multi-dot abbrevs like "U.S.", protect ALL internal dots too
    protected = _ABBREV_PERIOD.sub(_protect_abbrev, text)

    # Phase 2: protect initial periods (J.R.R., single initials)
    protected = _INITIAL_PERIOD.sub(lambda m: m.group(1) + '\x00', protected)

    # Phase 3: protect ellipsis — replace with markers so they don't split
    protected = re.sub(r'\.{2,}', lambda m: '\x01' * len(m.group()), protected)

    # Phase 4: split on sentence boundaries
    raw_parts = _SENTENCE_BOUNDARY.split(protected)

    # Phase 5: restore protected characters and clean up
    sentences = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        # Restore abbreviation periods
        part = part.replace('\x00', '.')
        # Restore ellipsis
        part = part.replace('\x01', '.')
        sentences.append(part)

    return sentences


def _protect_abbrev(match: re.Match) -> str:
    """Replace abbreviation+period, protecting ALL dots in multi-dot abbrevs."""
    abbrev = match.group(1)
    # Replace all dots in the abbreviation itself, plus the trailing one
    return abbrev.replace('.', '\x00') + '\x00'


def extract_capitalized_phrases(text: str, min_length: int = 2, min_occurrences: int = 2) -> List[str]:
    """Extract capitalized multi-word phrases from text.

    Useful for auto-detecting domain terms from content.

    Args:
        text: Input text.
        min_length: Minimum number of words in a phrase.
        min_occurrences: Minimum times a phrase must appear.

    Returns:
        List of unique phrases sorted by frequency (descending).
    """
    # Match capitalized phrases (2+ words starting with uppercase)
    pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
    matches = pattern.findall(text)

    from collections import Counter
    counts = Counter(matches)
    return [
        phrase for phrase, count in counts.most_common()
        if count >= min_occurrences and len(phrase.split()) >= min_length
    ]


def extract_quoted_terms(text: str, min_occurrences: int = 2) -> List[str]:
    """Extract terms in quotation marks that appear multiple times.

    Returns unique quoted terms appearing >= min_occurrences times.
    """
    # Match single or double quoted terms
    pattern = re.compile(r'["“”]([^"“”]{2,50})["“”]')
    matches = pattern.findall(text)

    from collections import Counter
    counts = Counter(matches)
    return [term for term, count in counts.most_common() if count >= min_occurrences]
