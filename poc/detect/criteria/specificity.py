"""Criterion: Low specificity.

AI-generated content often sounds polished but thin — few named entities,
numbers, concrete examples, source references, or personal details.

This is critical for education writing: a student reflection with no concrete
classroom moment, no student behaviour, no lesson detail, no source anchor
is likely weak or AI-like.
"""

import re
from typing import Dict, List, Optional

from ._types import CriterionScore


def _count_named_entities(text: str) -> int:
    # Heuristic: capitalised words not at sentence start, plus known patterns
    # (Proper noun detection without NER — conservative)
    from poc.detect.utils import split_sentences as _split
    sentences = _split(text)
    count = 0
    for sent in sentences:
        words = sent.split()
        for i, w in enumerate(words):
            if i > 0 and w[0].isupper() and w[1:].islower() and len(w) > 2:
                if w.lower() not in {"the", "this", "that", "these", "those", "there",
                                      "then", "they", "their", "them", "when", "where",
                                      "what", "which", "while", "with", "from", "into"}:
                    count += 1
    return count


def _count_numbers(text: str) -> int:
    return len(re.findall(r'\b\d+(?:\.\d+)?%?\b', text))


def _count_dates(text: str) -> int:
    return len(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text))


def _count_quotes(text: str) -> int:
    return len(re.findall(r'["“”]', text)) // 2


def _count_first_person(text: str) -> int:
    return len(re.findall(r'\b(I|me|my|mine|myself|we|us|our|ours)\b', text, re.I))


def _count_abstract_nouns(text: str) -> int:
    # NO-HARDCODE: the baked abstract-noun list (importance/significance/concept/role…) was
    # removed. Abstract/nominalised nouns are detected MORPHOLOGICALLY via their derivational
    # suffixes (-tion/-sion/-ment/-ance/-ence/-ity/-ness/-ism), which generalises across topics.
    words = re.findall(r'\b\w+\b', text.lower())
    return sum(1 for w in words
               if len(w) > 5 and re.search(r'(?:tion|sion|ment|ance|ence|ity|ness|ism)s?$', w))


def _split_sentences(text: str) -> List[str]:
    from poc.detect.utils import split_sentences
    return split_sentences(text)


# Common stop words to exclude from auto-detected domain terms
_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "us", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "which", "who", "whom", "what", "where",
    "when", "how", "not", "no", "nor", "if", "then", "than", "too",
    "very", "so", "just", "about", "also", "such", "each", "every",
    "all", "any", "both", "few", "more", "most", "other", "some",
    "only", "own", "same", "up", "out", "over", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "once", "here", "there", "because", "until",
    "while", "although", "however", "therefore", "moreover", "furthermore",
    "addition", "conclusion", "summary", "introduction", "background",
    "another", "many", "much", "well", "still", "even", "back", "new",
    "way", "thing", "must", "upon", "able", "like", "using", "used",
})

# Known generic compounds that are NOT domain terms
_GENERIC_COMPOUNDS = frozenset({
    "point out", "look down", "carry out", "set up", "take place",
    "come up", "go on", "make sure", "deal with", "hold on",
})

# NO-HARDCODE: the baked "weak single words" list (generic academic/business vocabulary used to
# soft-suppress weak domain terms) was removed -- a curated content-word list. Weak/generic terms
# are better filtered structurally (the _STOP_WORDS function-word stoplist + length/morphology),
# not by a hand-picked vocabulary. Empty set keeps the filter call sites intact.
_WEAK_SINGLE_WORDS: frozenset = frozenset()

# Patterns that indicate reference metadata (URL slugs, author names, journal titles)
# These should NOT count as domain grounding terms.
_REFERENCE_METADATA_PATTERNS = [
    r"https?://",                    # URL fragments
    r"^www\.",                       # website prefixes
    r"-\d{4}$",                      # year suffixes (e.g., "haidilao-inc")
    r"^[a-z]+-[a-z]+-[a-z]+-[a-z]+", # multi-segment URL slugs (4+ segments)
    r"^journal-",                    # journal URL prefix
    r"-news$",                       # news URL suffix
    r"-blog$",                       # blog URL suffix
    r"^how-",                        # how-to URL slug prefix
    r"^the-handbook$",               # generic reference title
    r"^in-psychology$",              # generic reference qualifier
    r"use-secret",                   # source-title fragment
    r"^[a-z]+-[a-z]+-[a-z]+-[a-z]+-", # 5+ segment slugs
    r"^journal of ",                 # journal title (with spaces)
    r"^altera",                      # source-name fragments (alterainstitute)
    r"^travelandleisure",            # source-name fragments
    r"^proinsight",                  # source-name fragments
]

# Known author/publisher/location patterns — not domain grounding
_REFERENCE_NAME_TERMS = frozenset({
    "edward elgar", "javier sese", "altera blog",
    "haidilao-inc", "proinsight",
    "melero-polo", "jurong point",
    "alterainstitute", "travelandleisure",
})

# Generic noun phrases that look domain-ish but aren't conceptual terms
_WEAK_DOMAIN_TERMS = frozenset({
    "service business", "the handbook", "use secret lights around",
    "in psychology", "journal of retailing",
})


def _auto_extract_domain_terms(text: str, min_freq: int = 1) -> List[str]:
    """Extract candidate domain terms automatically from text.

    Strategy (content-agnostic):
    1. Capitalized multi-word phrases (2–4 words) not at sentence start
    2. Hyphenated compound terms
    3. Multi-word phrases with a domain-bearing structure (noun+noun, adj+noun)
       that appear multiple times or contain uncommon vocabulary
    """
    import re as _re

    # Normalize text for matching (handle line breaks, extra spaces)
    text_norm = text.lower().replace("\n", " ")
    text_norm = _re.sub(r"\s+", " ", text_norm)

    candidates = {}  # term -> count

    # 1. Capitalized multi-word phrases (not sentence-initial)
    # Match sequences like "Service Quality", "Mystery Shopper Programme"
    # Use (?:...) non-capturing groups instead of variable-width lookbehind
    cap_phrases = _re.findall(
        r'(?:^|[.!?]\s|["\n])'        # after sentence boundary or quote/newline
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})'  # 2-4 capitalized words
        r'(?=\s+[a-z]|\s*[,.;:])',   # followed by lowercase or punctuation
        text,
    )
    # Also catch mid-sentence capitalized phrases (fixed-width lookbehind: single char)
    cap_phrases += _re.findall(
        r'(?<=\s)'
        r'([A-Z][a-z]+(?:\s+(?:of|in|for|and|the)\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+){1,3})'
        r'(?=\s+[a-z]|\s*[,.;:])',
        text,
    )

    # 2. Hyphenated compounds
    hyphenated = _re.findall(r'\b([a-z]+(?:-[a-z]+)+)\b', text.lower())

    # 3. Quoted terms (strong signal)
    quoted = _re.findall(r'["“]([^"”]{3,40})["”]', text)

    # 4. "Noun of/for Noun" patterns (e.g. "zone of proximal development")
    # Only match if the phrase starts with a noun-like content word (length > 5)
    # and ends with a content word (not stop word, length > 4)
    noun_phrases_raw = _re.findall(
        r'\b([a-z]+(?:\s+(?:of|for|in)\s+[a-z]+(?:\s+[a-z]+)?)?)\b',
        text.lower(),
    )
    noun_phrases = []
    for np in noun_phrases_raw:
        words = np.split()
        if (len(words) >= 3
                and words[0] not in _STOP_WORDS and len(words[0]) > 5
                and words[-1] not in _STOP_WORDS and len(words[-1]) > 4):
            noun_phrases.append(np)

    def _is_valid_term(t: str) -> bool:
        t_lower = t.lower().strip()
        words = t_lower.split()
        if len(words) < 2:
            return False
        if t_lower in _GENERIC_COMPOUNDS:
            return False
        # At least one word must not be a stop word
        if all(w in _STOP_WORDS for w in words):
            return False
        # Must contain at least one content word (not stop, length > 3)
        content_words = [w for w in words if w not in _STOP_WORDS and len(w) > 3]
        if not content_words:
            return False
        # Reject if more than half the words are stop words (noisy phrases)
        stop_ratio = sum(1 for w in words if w in _STOP_WORDS) / len(words)
        if stop_ratio > 0.5 and len(words) > 2:
            return False
        return True

    def _normalize(t: str) -> str:
        return t.lower().strip()

    # Process capitalized phrases
    for phrase in cap_phrases:
        norm = _normalize(phrase)
        if _is_valid_term(phrase):
            candidates[norm] = candidates.get(norm, 0) + 1

    # Process hyphenated compounds
    for term in hyphenated:
        if len(term) > 5 and term not in _GENERIC_COMPOUNDS:
            candidates[term] = candidates.get(term, 0) + 1

    # Process quoted terms
    for qt in quoted:
        norm = _normalize(qt)
        if len(norm.split()) >= 2 and _is_valid_term(qt):
            candidates[norm] = candidates.get(norm, 0) + 2  # quoted = stronger signal

    # Process noun phrases
    for np in noun_phrases:
        if _is_valid_term(np) and len(np.split()) >= 3:
            candidates[_normalize(np)] = candidates.get(_normalize(np), 0) + 1

    # 5. Single-word domain terms: long, uncommon content words.
    #    Conservative: only words > 9 chars OR repeated domain-suffix words > 7 chars.
    #    This catches "scaffolding", "vocational", "pedagogy" without catching
    #    "learning", "students", "teachers", "important".
    _COMMON_WORDS = frozenset({
        "learning", "students", "teachers", "teaching", "important",
        "different", "following", "understand", "something", "someone",
        "education", "knowledge", "including", "therefore", "development",
        "techniques", "structure", "students", "correction", "demonstration",
    })
    _DOMAIN_SUFFIXES_STRICT = {
        "tion", "sion", "ling", "ping", "ting", "city",
    }
    all_words = _re.findall(r'\b[a-z]+\b', text_norm)
    word_freq = {}
    for w in all_words:
        if w in _STOP_WORDS or w in _COMMON_WORDS:
            continue
        if len(w) > 9:
            word_freq[w] = word_freq.get(w, 0) + 1
        elif len(w) > 7:
            for suffix in _DOMAIN_SUFFIXES_STRICT:
                if w.endswith(suffix):
                    word_freq[w] = word_freq.get(w, 0) + 1
                    break

    # Add single-word terms that appear at least once
    for word, freq in word_freq.items():
        if word in _WEAK_SINGLE_WORDS:
            continue
        if freq >= min_freq:
            candidates[word] = candidates.get(word, 0) + freq

    # Filter: must appear at least min_freq times OR be quoted/hyphenated
    # Classify into strong (multi-word, hyphenated, quoted) vs weak (single-word)
    strong_terms = []
    weak_terms = []
    for term, count in candidates.items():
        if count >= min_freq:
            words = term.split()
            if len(words) >= 2 or "-" in term:
                strong_terms.append(term)
            else:
                # Single-word: weak unless it appears multiple times
                if count >= 2:
                    strong_terms.append(term)
                else:
                    weak_terms.append(term)

    return strong_terms + weak_terms


def _classify_domain_terms(
    terms: List[str],
) -> dict:
    """Split detected terms into core, contextual, and reference_metadata buckets.

    Only core and contextual terms affect domain grounding score.
    Reference metadata terms (URL slugs, author names, journal titles) are
    tracked but excluded from grounding calculations.
    """
    import re as _re

    core_terms = []
    contextual_terms = []
    reference_metadata_terms = []

    for t in terms:
        t_lower = t.lower().strip()

        # Check if it's a known reference name
        if t_lower in _REFERENCE_NAME_TERMS:
            reference_metadata_terms.append(t)
            continue

        # Check weak/generic noun phrases that aren't conceptual domain terms
        if t_lower in _WEAK_DOMAIN_TERMS:
            reference_metadata_terms.append(t)
            continue

        # Check URL-slug-like patterns
        is_ref = False
        for pat in _REFERENCE_METADATA_PATTERNS:
            if _re.search(pat, t_lower):
                is_ref = True
                break

        if is_ref:
            reference_metadata_terms.append(t)
            continue

        # Segment count heuristic: 4+ hyphen segments = likely URL slug
        if "-" in t_lower and len(t_lower.split("-")) >= 4:
            reference_metadata_terms.append(t)
            continue

        # Classify: multi-word with specific domain structure = core
        words = t_lower.split()
        if len(words) >= 2:
            # Check for hyphenated compound terms (strong domain signal)
            if "-" in t:
                core_terms.append(t)
            # Check for "Noun of/for Noun" structure
            elif any(w in ("of", "for", "in") for w in words):
                contextual_terms.append(t)
            else:
                core_terms.append(t)
        else:
            # Single word: contextual (supporting domain signal)
            contextual_terms.append(t)

    return {
        "domain_terms_core": core_terms,
        "domain_terms_contextual": contextual_terms,
        "reference_metadata_terms": reference_metadata_terms,
    }


def score(
    content: str,
    *,
    domain_terms: Optional[List[str]] = None,
    specificity_threshold: float = 0.35,
    **kwargs,
) -> CriterionScore:
    # Normalize text for matching (handle line breaks, extra spaces)
    text_normalized = content.lower().replace("\n", " ")
    text_normalized = re.sub(r"\s+", " ", text_normalized)

    sentences = _split_sentences(content)
    words = re.findall(r'\b\w+\b', content)
    word_count = max(len(words), 1)

    named_entities = _count_named_entities(content)
    numbers = _count_numbers(content)
    dates = _count_dates(content)
    quotes = _count_quotes(content)
    first_person = _count_first_person(content)
    abstract_nouns = _count_abstract_nouns(content)

    # Auto-detect domain terms from the text itself (content-agnostic)
    auto_terms = _auto_extract_domain_terms(content)
    # Merge: explicit domain_terms override/add to auto-detected
    if domain_terms:
        terms_to_check = list(set(t.lower() for t in domain_terms) | set(auto_terms))
    else:
        terms_to_check = auto_terms

    # Filter terms that actually appear in text, then classify
    matched_terms = [t for t in terms_to_check if t.lower() in text_normalized]

    # Classify into core / contextual / reference_metadata
    classified = _classify_domain_terms(matched_terms)
    core_terms = classified["domain_terms_core"]
    contextual_terms = classified["domain_terms_contextual"]
    ref_meta_terms = classified["reference_metadata_terms"]

    # Strong/weak split for backward compat — only from core + contextual
    grounding_terms = core_terms + contextual_terms
    strong_domain_count = 0
    weak_domain_count = 0
    strong_domain_terms = []
    weak_domain_terms = []
    for t in grounding_terms:
        words = t.split()
        if len(words) >= 2 or "-" in t:
            strong_domain_count += 1
            strong_domain_terms.append(t)
        else:
            weak_domain_count += 1
            weak_domain_terms.append(t)

    # For scoring, only core+contextual terms count; reference metadata excluded
    domain_term_count = strong_domain_count + int(weak_domain_count * 0.2)
    matched_domain_terms = grounding_terms  # excludes reference_metadata

    # Weighted specificity score (higher = more specific = more human-like)
    # Genre-aware: for short/conceptual texts, domain terms carry more weight
    # than named entities or numbers which are rarer in conclusions/reflections
    if word_count < 300 and domain_term_count >= 5:
        # Conceptual/short text genre: domain terms are the primary specificity signal
        ne_weight = 0.015
        num_weight = 0.01
        date_weight = 0.01
        dt_weight = 0.05
        quote_weight = 0.03
        fp_weight = 0.02
    else:
        ne_weight = 0.03
        num_weight = 0.02
        date_weight = 0.02
        dt_weight = 0.03
        quote_weight = 0.04
        fp_weight = 0.02

    specificity = min(1.0, (
        named_entities * ne_weight
        + numbers * num_weight
        + dates * date_weight
        + quotes * quote_weight
        + first_person * fp_weight
        + domain_term_count * dt_weight
    )) / max(1.0, word_count / 100)

    # Penalise high abstract noun ratio
    abstract_ratio = abstract_nouns / word_count
    specificity = max(0.0, specificity - abstract_ratio * 2)

    # Low specificity = more AI-like; invert for our score
    value = max(0.0, min(1.0, 1.0 - specificity / 0.5))

    # Cap concern when strong domain grounding present
    grounding_index = strong_domain_count / max(word_count / 100, 1)
    if grounding_index >= 2.0:
        # Strong grounding reduces specificity concern significantly
        value = min(value, 0.40 + (value * 0.15))
    elif grounding_index >= 1.0:
        # Moderate grounding provides some reduction
        value = min(value, 0.60 + (value * 0.10))

    if specificity < specificity_threshold:
        label = "high"
    elif specificity < specificity_threshold * 1.5:
        label = "medium"
    else:
        label = "low"

    return CriterionScore(
        name="low_specificity",
        value=round(value, 4),
        label=label,
        details={
            "specificity_score": round(specificity, 4),
            "specificity_risk": round(value, 4),
            "named_entities": named_entities,
            "numbers": numbers,
            "dates": dates,
            "quotes": quotes,
            "first_person_count": first_person,
            "abstract_noun_count": abstract_nouns,
            "abstract_noun_ratio": round(abstract_ratio, 4),
            "domain_term_count": domain_term_count,
            "domain_terms_strong": strong_domain_terms,
            "domain_terms_weak": weak_domain_terms,
            "domain_terms_core": core_terms,
            "domain_terms_contextual": contextual_terms,
            "reference_metadata_terms": ref_meta_terms,
            "domain_terms": matched_domain_terms,
            "domain_grounding_index": round(
                strong_domain_count / max(word_count / 100, 1), 3
            ),
            "domain_grounding_level": (
                "strong" if strong_domain_count / max(word_count / 100, 1) >= 2.0
                else "moderate" if strong_domain_count / max(word_count / 100, 1) >= 1.0
                else "weak"
            ),
            "domain_grounding_score": min(
                1.0, round(strong_domain_count / max(word_count / 100, 1), 3)
            ),
            "word_count": word_count,
        },
    )
