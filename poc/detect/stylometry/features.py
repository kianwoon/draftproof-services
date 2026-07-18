"""Paragraph-level stylometric fingerprinting — pure, dependency-free measurement.

Extracts a fixed set of writing-style measurements per paragraph (sentence-length
distribution, lexical diversity, punctuation habits, discourse-marker usage, function-
word density, readability, passive-voice and subordination rates, lexical density, and
academic-vocabulary usage). This module is a standalone library: `extract_fingerprints`
is a pure function with zero I/O and zero calls into any other `poc/detect/` module
besides `document_structure.py` (reused only for paragraph/sentence segmentation and
stable IDs). It is NOT wired into the detection pipeline yet — a later task builds an
outlier detector and a `ConsistencyDetector` on top of these fingerprints.

No external NLP dependency (no spaCy/nltk/textstat) is used anywhere in this module.
Three heuristics are therefore hand-rolled approximations, each documented at its
definition below:
  - Syllable counting (for Flesch Reading Ease) via vowel-group counting, not a
    phonetic dictionary lookup.
  - Passive-voice detection via a be-form + past-participle regex (IRREGULAR_PARTICIPLES
    plus a "-ed" regex fallback), not a parser.
  - Academic-vocabulary matching via the AWL headword list plus a small, deliberately
    conservative inflectional-suffix strip, not a lemmatizer/stemmer.
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Any

from ..document_structure import StructuredSentence, structured_sentences
from .lexicons import (
    ACADEMIC_VOCAB,
    BE_FORMS,
    FUNCTION_WORDS,
    IRREGULAR_PARTICIPLES,
    SUBORDINATING_CONJUNCTIONS,
    TRANSITION_MARKERS,
)

# ---------------------------------------------------------------------------
# Named constants — every threshold/weight used below is named here, not inlined.
# ---------------------------------------------------------------------------

# "Per 100 words" normalization base shared by punctuation/transition/function-word/
# academic-vocab rates.
_RATE_PER_WORDS_BASE = 100.0

# Word tokenizer: ASCII letters plus an internal apostrophe (for contractions such as
# "don't", "it's"). This is an approximation — it does not recognize accented Unicode
# letters (e.g. "café" tokenizes as "caf") — acceptable for an English-focused,
# dependency-free heuristic library.
_WORD_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Vowel-group syllable-count heuristic: counts contiguous runs of a/e/i/o/u/y as one
# syllable each, then subtracts one for a silent trailing "e" (e.g. "make" -> 1, not
# 2). This is a documented approximation, not a CMU-dictionary-style phonetic count.
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")
_MIN_SYLLABLES_PER_WORD = 1

# Dash detection: an em/en dash character anywhere, or a single/double hyphen
# surrounded by whitespace (so compound words like "well-known" are not counted).
_DASH_RE = re.compile(r"[–—]|(?<=\s)--?(?=\s)")

# Flesch Reading Ease (Flesch, R. (1948). "A new readability yardstick." Journal of
# Applied Psychology, 32(3), 221-233) published formula constants.
_FLESCH_BASE = 206.835
_FLESCH_WORDS_PER_SENTENCE_WEIGHT = 1.015
_FLESCH_SYLLABLES_PER_WORD_WEIGHT = 84.6
# Returned when word_count or sentence_count is 0 (e.g. a punctuation-only paragraph
# with no countable word tokens). Deliberately `None`, not a float, so this degenerate
# "no data to compute a score from" case can never be confused with a legitimately
# computed score -- including a real paragraph whose Flesch score happens to be low
# or even exactly 0.0. Callers must handle `flesch_reading_ease is None` explicitly.
_FLESCH_DEGENERATE_SCORE = None

# Passive-voice detection: a be-form optionally followed by a single "-ly" adverb, then
# a candidate past participle. This is a shallow regex approximation of "be + V-en/V-ed"
# passive constructions, not a syntactic parse — it will miss passives with more than
# one intervening word and can mis-flag a be-form followed by a coincidental "-ed"/
# irregular-participle-shaped adjective (e.g. "was tired"). Documented, accepted trade-off.
_PASSIVE_ADVERB_GAP = r"(?:\s+\w+ly\b)?"
_ED_PARTICIPLE_RE = re.compile(r"^[a-z]+ed$")

# Academic-vocabulary matching: after an exact AWL-headword check, strip one of these
# common regular inflectional endings and retry. This intentionally does NOT attempt
# full stemming/lemmatization (no derivational endings like "-tion"/"-ity"/"-al", no
# irregular spelling changes like y->ies or e-deletion before "-ing"/"-ed") — it is a
# conservative approximation that under-counts rather than over-counts.
_ACADEMIC_VOCAB_SUFFIXES = ("ing", "ed", "es", "s")


@dataclass(frozen=True)
class ParagraphFingerprint:
    """A pure measurement snapshot of one paragraph's writing style.

    Every field is a deterministic function of the paragraph's text — no scoring,
    thresholds, or verdicts are attached at this layer. Downstream tasks (outlier
    detection, then the ConsistencyDetector) interpret these fingerprints.
    """

    paragraph_id: str
    sentence_length_mean: float
    sentence_length_std: float
    word_length_mean: float
    root_ttr: float
    punctuation_rates: dict[str, float]
    transition_rate: float
    function_word_rate: float
    flesch_reading_ease: float | None
    passive_voice_rate: float
    subordination_rate: float
    lexical_density: float
    academic_vocab_rate: float
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "paragraph_id": self.paragraph_id,
            "sentence_length_mean": self.sentence_length_mean,
            "sentence_length_std": self.sentence_length_std,
            "word_length_mean": self.word_length_mean,
            "root_ttr": self.root_ttr,
            "punctuation_rates": dict(self.punctuation_rates),
            "transition_rate": self.transition_rate,
            "function_word_rate": self.function_word_rate,
            "flesch_reading_ease": self.flesch_reading_ease,
            "passive_voice_rate": self.passive_voice_rate,
            "subordination_rate": self.subordination_rate,
            "lexical_density": self.lexical_density,
            "academic_vocab_rate": self.academic_vocab_rate,
            "word_count": self.word_count,
        }


def extract_fingerprints(text: str) -> list[ParagraphFingerprint]:
    """Extract one ParagraphFingerprint per paragraph in `text`.

    Pure function: zero I/O, zero network/model calls. Paragraph/sentence segmentation
    and paragraph IDs are reused verbatim from `document_structure.structured_sentences`
    (the same stable `p001`, `p002`, ... IDs the rest of the detect pipeline uses) —
    this module does not re-implement splitting.
    """
    sentence_rows = structured_sentences(text)
    if not sentence_rows:
        return []

    paragraphs: dict[str, list[StructuredSentence]] = {}
    paragraph_order: list[str] = []
    for row in sentence_rows:
        if row.paragraph_id not in paragraphs:
            paragraphs[row.paragraph_id] = []
            paragraph_order.append(row.paragraph_id)
        paragraphs[row.paragraph_id].append(row)

    return [
        _build_fingerprint(paragraph_id, paragraphs[paragraph_id])
        for paragraph_id in paragraph_order
    ]


def _build_fingerprint(paragraph_id: str, rows: list[StructuredSentence]) -> ParagraphFingerprint:
    sentence_texts = [row.sentence for row in rows]
    sentence_count = len(sentence_texts)
    paragraph_text = " ".join(text.strip() for text in sentence_texts if text.strip())

    tokens = _extract_word_tokens(paragraph_text)
    word_count = len(tokens)
    lower_tokens = [token.lower() for token in tokens]
    sentence_word_counts = [len(_extract_word_tokens(text)) for text in sentence_texts]

    return ParagraphFingerprint(
        paragraph_id=paragraph_id,
        sentence_length_mean=_safe_mean(sentence_word_counts),
        sentence_length_std=_safe_pstdev(sentence_word_counts),
        word_length_mean=_safe_mean([len(token) for token in tokens]),
        root_ttr=_root_ttr(lower_tokens),
        punctuation_rates=_punctuation_rates(paragraph_text, word_count),
        transition_rate=_rate_per_words(_count_phrase_matches(paragraph_text, _TRANSITION_PATTERN), word_count),
        function_word_rate=_rate_per_words(sum(1 for t in lower_tokens if t in FUNCTION_WORDS), word_count),
        flesch_reading_ease=_flesch_reading_ease(tokens, sentence_count),
        passive_voice_rate=_passive_voice_rate(sentence_texts),
        subordination_rate=_subordination_rate(paragraph_text, sentence_count),
        lexical_density=_lexical_density(lower_tokens),
        academic_vocab_rate=_rate_per_words(_count_academic_vocab(lower_tokens), word_count),
        word_count=word_count,
    )


# ---------------------------------------------------------------------------
# Tokenization / shared arithmetic helpers
# ---------------------------------------------------------------------------


def _extract_word_tokens(text: str) -> list[str]:
    return _WORD_TOKEN_RE.findall(text)


def _safe_mean(values: list[int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _safe_pstdev(values: list[int]) -> float:
    return float(statistics.pstdev(values)) if values else 0.0


def _rate_per_words(count: int, word_count: int) -> float:
    if word_count == 0:
        return 0.0
    return (count / word_count) * _RATE_PER_WORDS_BASE


def _root_ttr(lower_tokens: list[str]) -> float:
    """Root TTR / Guiraud's index: unique types / sqrt(total tokens) — a
    length-corrected lexical-diversity measure (Guiraud, P. (1954). Les caracteres
    statistiques du vocabulaire. Presses Universitaires de France)."""
    if not lower_tokens:
        return 0.0
    unique_types = len(set(lower_tokens))
    return unique_types / math.sqrt(len(lower_tokens))


# ---------------------------------------------------------------------------
# Punctuation
# ---------------------------------------------------------------------------


def _punctuation_rates(paragraph_text: str, word_count: int) -> dict[str, float]:
    counts = {
        "comma": paragraph_text.count(","),
        "semicolon": paragraph_text.count(";"),
        "colon": paragraph_text.count(":"),
        "dash": len(_DASH_RE.findall(paragraph_text)),
        "paren": paragraph_text.count("("),
    }
    return {key: _rate_per_words(value, word_count) for key, value in counts.items()}


# ---------------------------------------------------------------------------
# Phrase matching (transitions, subordination) — shared alternation builder
# ---------------------------------------------------------------------------


def _phrase_pattern(phrases: frozenset[str]) -> re.Pattern[str]:
    """Compile a case-insensitive alternation over `phrases`, longest phrase first, so
    a multi-word phrase always wins over a shorter phrase/word that is its prefix."""
    ordered = sorted(phrases, key=len, reverse=True)
    alternation = "|".join(re.escape(phrase) for phrase in ordered)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


_TRANSITION_PATTERN = _phrase_pattern(TRANSITION_MARKERS)
_SUBORDINATION_PATTERN = _phrase_pattern(SUBORDINATING_CONJUNCTIONS)
_BE_FORM_ALTERNATION = "|".join(re.escape(form) for form in sorted(BE_FORMS, key=len, reverse=True))
_PASSIVE_RE = re.compile(
    rf"\b(?:{_BE_FORM_ALTERNATION})\b{_PASSIVE_ADVERB_GAP}\s+(\w+)",
    re.IGNORECASE,
)


def _count_phrase_matches(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def _subordination_rate(paragraph_text: str, sentence_count: int) -> float:
    if sentence_count == 0:
        return 0.0
    return _count_phrase_matches(paragraph_text, _SUBORDINATION_PATTERN) / sentence_count


# ---------------------------------------------------------------------------
# Passive voice
# ---------------------------------------------------------------------------


def _is_past_participle(word: str) -> bool:
    lowered = word.lower()
    return lowered in IRREGULAR_PARTICIPLES or bool(_ED_PARTICIPLE_RE.match(lowered))


def _passive_voice_rate(sentence_texts: list[str]) -> float:
    sentence_count = len(sentence_texts)
    if sentence_count == 0:
        return 0.0
    passive_count = 0
    for sentence_text in sentence_texts:
        for match in _PASSIVE_RE.finditer(sentence_text):
            if _is_past_participle(match.group(1)):
                passive_count += 1
    return passive_count / sentence_count


# ---------------------------------------------------------------------------
# Readability (Flesch Reading Ease)
# ---------------------------------------------------------------------------


def _count_syllables(word: str) -> int:
    lowered = word.lower()
    count = len(_VOWEL_GROUP_RE.findall(lowered))
    if lowered.endswith("e") and count > 1:
        count -= 1
    return max(count, _MIN_SYLLABLES_PER_WORD)


def _flesch_reading_ease(tokens: list[str], sentence_count: int) -> float | None:
    word_count = len(tokens)
    if word_count == 0 or sentence_count == 0:
        return _FLESCH_DEGENERATE_SCORE
    syllable_count = sum(_count_syllables(token) for token in tokens)
    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllable_count / word_count
    return (
        _FLESCH_BASE
        - _FLESCH_WORDS_PER_SENTENCE_WEIGHT * words_per_sentence
        - _FLESCH_SYLLABLES_PER_WORD_WEIGHT * syllables_per_word
    )


# ---------------------------------------------------------------------------
# Lexical density / academic vocabulary
# ---------------------------------------------------------------------------


def _lexical_density(lower_tokens: list[str]) -> float:
    """Content-word ratio: (tokens not in FUNCTION_WORDS) / total tokens — the
    standard lexical-density definition (Halliday, M. A. K. (1985). Spoken and Written
    Language. Deakin University Press)."""
    if not lower_tokens:
        return 0.0
    content_tokens = sum(1 for token in lower_tokens if token not in FUNCTION_WORDS)
    return content_tokens / len(lower_tokens)


def _matches_academic_vocab(token: str) -> bool:
    if token in ACADEMIC_VOCAB:
        return True
    for suffix in _ACADEMIC_VOCAB_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix):
            if token[: -len(suffix)] in ACADEMIC_VOCAB:
                return True
    return False


def _count_academic_vocab(lower_tokens: list[str]) -> int:
    return sum(1 for token in lower_tokens if _matches_academic_vocab(token))
