"""Paragraph-level stylometric outlier detection — pure, dependency-free measurement.

Builds on Task 1's `ParagraphFingerprint` (`poc/detect/stylometry/features.py`) to flag
paragraphs whose writing-style fingerprint deviates sharply from the rest of the
document — one piece of the "consistency risk" signal (e.g. a single AI-generated or
outsourced paragraph dropped into an otherwise consistently-authored document). This
module is a standalone library: `detect_outliers` is a pure function with zero I/O and
zero calls into any other `poc/detect/` module besides `features.py`'s
`ParagraphFingerprint` type. It is NOT wired into the detection pipeline yet — a later
task builds a `ConsistencyDetector` that consumes this alongside the fingerprints.

Method — modified (robust) z-score per feature dimension (Iglewicz, B. and Hoaglin, D.
(1993), "How to Detect and Handle Outliers", ASQC Basic References in Quality Control,
Vol. 16): median/MAD instead of mean/stdev, because a single wildly-different paragraph
would otherwise drag the mean/stdev toward itself and hide its own deviation. Each
paragraph's z-score for each feature is computed **leave-one-out**: the median/MAD
baseline is taken over every OTHER paragraph, excluding the paragraph being scored, so
a shifted paragraph cannot dilute its own outlier signal by contributing to its own
baseline.

A paragraph's overall `outlier_score` is the MAX of its per-feature z-scores (not a
sum or a combined multi-dimensional distance such as a Euclidean norm across all
dimensions). This is a deliberate choice: `OUTLIER_THRESHOLD` is calibrated in
single-dimension "robust-z units" (the Iglewicz & Hoaglin convention), and this
fingerprint has ~16 feature dimensions — summing or norm-combining that many
dimensions would inflate an all-dimensions-mildly-noisy paragraph's combined score
past the threshold even when no single dimension is actually anomalous, silently
miscalibrating the threshold. Taking the max keeps every comparison against
`OUTLIER_THRESHOLD` in the same, consistent single-dimension unit, and a paragraph
that is genuinely off-voice on even one dimension (e.g. sentence length or
function-word rate) is exactly the "different voice" signal this detector exists to
catch.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .features import ParagraphFingerprint

# ---------------------------------------------------------------------------
# Named constants — every threshold/weight used below is named here, not inlined.
# ---------------------------------------------------------------------------

# A document needs at least this many paragraphs before per-paragraph outlier
# statistics (median/MAD over "the rest of the document") are meaningful at all —
# below this floor we fail open (no findings) rather than flag noise from a
# statistically unsupportable sample. Value given by the task brief.
MIN_PARAGRAPHS = 6

# Paragraphs shorter than this are excluded from the outlier pool entirely: their
# per-100-word rate features (transition_rate, passive_voice_rate, punctuation
# rates, ...) are noisy at low word counts — one extra comma in a 10-word paragraph
# swings its rate wildly, which would look like a stylistic outlier for reasons
# unrelated to an actual voice shift. 50 words matches the existing "too short to
# reliably score" floor already used elsewhere in poc/detect/ for the same reason
# (see detect/base.py._assess_confidence and detect/ai_generation.py, both of which
# treat word_count < 50 as unreliable-signal territory).
MIN_WORDS_PER_PARAGRAPH = 50

# Modified z-score outlier threshold, in robust-z units — Iglewicz & Hoaglin (1993)'s
# recommended cutoff for the modified z-score defined by _MODIFIED_Z_CONSTANT below.
# A paragraph whose max per-feature robust z-score exceeds this is flagged.
OUTLIER_THRESHOLD = 3.5

# The 0.75 quantile of the standard normal distribution — the standard scaling
# constant that makes MAD comparable to a standard deviation under normality
# (Iglewicz & Hoaglin (1993)'s modified z-score: 0.6745 * (x - median) / MAD).
_MODIFIED_Z_CONSTANT = 0.6745

# sqrt(pi/2) -- the scaling constant that makes MEAN absolute deviation comparable
# to a standard deviation under normality. Iglewicz & Hoaglin (1993)'s documented
# fallback scale estimator for when MAD itself is zero (a tie-heavy/low-diversity
# sample where the median absolute deviation collapses to 0 even though the sample
# is not perfectly uniform).
_MEAN_AD_SCALE_CONSTANT = 1.253314

# Sentinel z-score used only when BOTH MAD and mean-absolute-deviation are zero --
# i.e. every other paragraph has the numerically identical value on this dimension,
# so there is no spread whatsoever to normalize against. A perfectly uniform
# baseline has, by definition, zero tolerance for any deviation, so a genuinely
# differing value is reported as clearly exceeding OUTLIER_THRESHOLD. Deliberately a
# finite multiple of OUTLIER_THRESHOLD (not `math.inf`): report artifacts are
# serialized to JSON (uploaded to R2, parsed by the frontend) and JSON has no
# `Infinity` literal, so an infinite score would break every downstream consumer.
_ZERO_SPREAD_SENTINEL_Z_SCORE = OUTLIER_THRESHOLD * 2.0

# "Top-N deviating features" count named in the task brief (report copy shows up to
# the top 3 dimensions that drove a paragraph's outlier flag).
_TOP_FEATURES_COUNT = 3


class OutlierStrategy(Enum):
    """Which statistical method `detect_outliers` uses to score paragraphs."""

    ROBUST_ZSCORE = "robust_zscore"
    # Future work: local outlier factor needs a denser neighborhood (n>=12
    # paragraphs) for a meaningful local-density comparison than the leave-one-out
    # global median/MAD approach used here. Not implemented in this task — stub
    # only, raises NotImplementedError below.
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"


@dataclass(frozen=True)
class OutlierResult:
    """One flagged paragraph: which paragraph, how anomalous, and on what features.

    `top_deviating_features` holds human-readable feature names (e.g. "sentence
    length", "passive voice rate") — not raw ParagraphFingerprint field names —
    since this feeds report copy shown to end users in a later task.
    """

    paragraph_id: str
    outlier_score: float
    top_deviating_features: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feature vector — every numeric dimension of ParagraphFingerprint, paired with its
# human-readable report name. `word_count` is deliberately excluded: it measures
# paragraph length, not writing style, and every rate-based feature above is already
# normalized per-100-words, so word_count itself carries no style-consistency signal
# (it is used only as the MIN_WORDS_PER_PARAGRAPH eligibility filter, below).
# ---------------------------------------------------------------------------

_FeatureExtractor = Callable[[ParagraphFingerprint], "float | None"]

_FEATURE_EXTRACTORS: tuple[tuple[str, _FeatureExtractor], ...] = (
    ("sentence length", lambda fp: fp.sentence_length_mean),
    ("sentence length variability", lambda fp: fp.sentence_length_std),
    ("word length", lambda fp: fp.word_length_mean),
    ("vocabulary diversity", lambda fp: fp.root_ttr),
    ("transition word usage", lambda fp: fp.transition_rate),
    ("function word rate", lambda fp: fp.function_word_rate),
    ("readability", lambda fp: fp.flesch_reading_ease),
    ("passive voice rate", lambda fp: fp.passive_voice_rate),
    ("subordinate clause rate", lambda fp: fp.subordination_rate),
    ("lexical density", lambda fp: fp.lexical_density),
    ("academic vocabulary rate", lambda fp: fp.academic_vocab_rate),
    ("comma usage", lambda fp: fp.punctuation_rates.get("comma")),
    ("semicolon usage", lambda fp: fp.punctuation_rates.get("semicolon")),
    ("colon usage", lambda fp: fp.punctuation_rates.get("colon")),
    ("dash usage", lambda fp: fp.punctuation_rates.get("dash")),
    ("parenthetical usage", lambda fp: fp.punctuation_rates.get("paren")),
)


def detect_outliers(
    fingerprints: list[ParagraphFingerprint],
    strategy: OutlierStrategy = OutlierStrategy.ROBUST_ZSCORE,
) -> list[OutlierResult]:
    """Flag paragraphs whose stylometric fingerprint deviates sharply from the rest
    of the document.

    Pure function: zero I/O, deterministic, depends only on `ParagraphFingerprint`.
    Returns a findings list (only the paragraphs that exceed `OUTLIER_THRESHOLD`),
    not an annotation of every paragraph — empty when the document is too short to
    support outlier statistics (`MIN_PARAGRAPHS` / `MIN_WORDS_PER_PARAGRAPH`) or when
    no paragraph deviates enough to be flagged.
    """
    if strategy is OutlierStrategy.LOCAL_OUTLIER_FACTOR:
        raise NotImplementedError(
            "OutlierStrategy.LOCAL_OUTLIER_FACTOR is future work (needs n>=12 "
            "paragraphs for a meaningful local-density neighborhood); only "
            "ROBUST_ZSCORE is implemented in this module."
        )

    eligible = [fp for fp in fingerprints if fp.word_count >= MIN_WORDS_PER_PARAGRAPH]
    if len(eligible) < MIN_PARAGRAPHS:
        return []

    results: list[OutlierResult] = []
    for index, fp in enumerate(eligible):
        feature_scores = _score_paragraph_features(index, fp, eligible)
        if not feature_scores:
            continue

        outlier_score = max(score for _name, score in feature_scores)
        if outlier_score <= OUTLIER_THRESHOLD:
            continue

        # Only report dimensions that actually deviate -- a paragraph flagged on
        # just 1 or 2 dimensions must not have its remaining "top 3" slots padded
        # with non-deviating (z == 0.0) feature names, which would misrepresent
        # those dimensions as having driven the flag in report copy.
        deviating = [(name, score) for name, score in feature_scores if score > 0.0]
        top_features = [
            name
            for name, _score in sorted(deviating, key=lambda item: item[1], reverse=True)[
                :_TOP_FEATURES_COUNT
            ]
        ]
        results.append(
            OutlierResult(
                paragraph_id=fp.paragraph_id,
                outlier_score=outlier_score,
                top_deviating_features=top_features,
            )
        )

    return results


def _score_paragraph_features(
    index: int,
    fp: ParagraphFingerprint,
    eligible: list[ParagraphFingerprint],
) -> list[tuple[str, float]]:
    """Per-feature leave-one-out robust z-scores for one paragraph.

    Skips a feature dimension entirely (does not contribute a 0.0) whenever the
    paragraph's own value is `None`, or when no OTHER paragraph has a usable
    (non-`None`) value for that dimension — `flesch_reading_ease` is the field that
    can legitimately be `None` (see features.py), and treating `None` as `0.0` would
    fabricate a spurious extreme deviation instead of honestly having no data.
    """
    scores: list[tuple[str, float]] = []
    for human_name, extractor in _FEATURE_EXTRACTORS:
        value = extractor(fp)
        if value is None:
            continue

        others = [
            other_value
            for other_index, other_fp in enumerate(eligible)
            if other_index != index
            for other_value in (extractor(other_fp),)
            if other_value is not None
        ]
        if not others:
            continue

        scores.append((human_name, _robust_z_score(value, others)))
    return scores


def _robust_z_score(value: float, others: list[float]) -> float:
    """Modified (median/MAD) z-score of `value` against the `others` baseline, with
    the Iglewicz & Hoaglin (1993) mean-absolute-deviation fallback for when MAD
    itself is zero, and a finite sentinel for the fully-degenerate (zero-spread)
    case. Always returns a finite float -- see `_ZERO_SPREAD_SENTINEL_Z_SCORE`."""
    median_other = statistics.median(others)
    deviations = [abs(other - median_other) for other in others]
    mad = statistics.median(deviations)

    if mad != 0.0:
        return abs(_MODIFIED_Z_CONSTANT * (value - median_other) / mad)

    # MAD collapsed to zero (a tie-heavy "others" pool). Fall back to mean
    # absolute deviation, the documented alternate scale estimator for this case.
    mean_ad = statistics.mean(deviations)
    if mean_ad != 0.0:
        return abs((value - median_other) / (_MEAN_AD_SCALE_CONSTANT * mean_ad))

    # Every other paragraph has the numerically identical value on this dimension --
    # zero spread by either estimator. See _ZERO_SPREAD_SENTINEL_Z_SCORE for why a
    # differing value gets a finite sentinel rather than dividing by zero.
    return 0.0 if value == median_other else _ZERO_SPREAD_SENTINEL_Z_SCORE
