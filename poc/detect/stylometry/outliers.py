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

Per-feature z-scores stay in single-dimension "robust-z units" (never summed or
norm-combined across dimensions — that would inflate an all-dimensions-mildly-noisy
paragraph past any single-dimension threshold and silently miscalibrate it). The
paragraph-level FLAG DECISION is a **k-of-m rule** (`_is_outlier`): a paragraph is an
outlier only when at least `OUTLIER_MIN_DEVIATING_FEATURES` of its per-feature
z-scores exceed `OUTLIER_PER_FEATURE_Z_THRESHOLD`. The old rule flagged on the MAX
per-feature z alone (k=1 vs a single-comparison 3.5 cutoff); over ~16 leave-one-out
comparisons on small n that fired on ~31% of paragraphs in genuine single-author
human prose. Requiring several correlated dimensions to deviate at once — the actual
signature of a different authorial voice — holds the human per-paragraph false-
positive rate under 5% while still catching a genuinely off-voice paragraph. Both
constants are empirically calibrated: see `OUTLIER_PER_FEATURE_Z_THRESHOLD` below and
its provenance script `poc/calibration/derive_outlier_threshold.py`. `outlier_score`
(reported for ranking/JSON) remains the MAX per-feature z of a flagged paragraph.
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

# --- Paragraph flag decision (k-of-m) -------------------------------------------
# The OLD rule was "flag if the MAX per-feature robust z-score exceeds 3.5". 3.5 is
# Iglewicz & Hoaglin (1993)'s cutoff for a SINGLE modified-z comparison — but the
# score is a MAX over m (~16) leave-one-out per-feature comparisons on small n, so a
# max-over-many crossed 3.5 far too often: it flagged ~31% of paragraphs (per-para
# FP) in genuine single-author human prose (49 Gutenberg essays, 100% of docs — see
# calibration/consistency_human_precision.json BEFORE this change). Normal-theory
# Bonferroni/Šidák does NOT fix it: the small-n MAD z-scores have fat tails, so a
# threshold derived from normal quantiles undershoots the empirical FP.
#
# The replacement is an EMPIRICALLY CALIBRATED k-of-m rule: flag a paragraph only
# when at least OUTLIER_MIN_DEVIATING_FEATURES of its per-feature robust z-scores
# exceed OUTLIER_PER_FEATURE_Z_THRESHOLD. Both constants are the argmax-sensitivity
# point subject to per-paragraph FP <= 5% on a seeded author-stratified human
# derivation set, validated on a held-out set.
# Provenance (the constants' ONLY justification — do not hand-edit these):
#   poc/calibration/derive_outlier_threshold.py           (derivation script)
#   poc/calibration/outlier_threshold_derivation.json     (sweep + chosen rule + holdout)
# Chosen 2026-07-21 by calibration/derive_outlier_threshold.py (seed 20260721,
# author-stratified 65/35 split of the 49 Gutenberg + eligible ESL human corpus):
# rule "3-of-m @ 5.25". Derivation per-paragraph FP 0.618% (was ~31%), holdout FP
# 2.041% (both << 5% target); multi-author concat pseudo-docs flagged 1.55x more than
# clean human prose (informativeness guardrail met); the ESL proficiency-parity gate
# (consistency_esl_rates.py --mode concat) PASSES (gap +1.84pt, under the 5pt limit —
# the old max>3.5 rule only "passed" it via saturation, flagging ~100% of pseudo-docs
# in both bands); shifted-paragraph fixture still flagged. Full sweep + frontier +
# holdout + ESL parity per candidate: outlier_threshold_derivation.json.
OUTLIER_PER_FEATURE_Z_THRESHOLD = 5.25
OUTLIER_MIN_DEVIATING_FEATURES = 3

# Retained: the single-comparison Iglewicz & Hoaglin cutoff, still used to size the
# zero-spread sentinel below and reported by the derivation script as the (failed)
# uncorrected baseline. NOT the live flag threshold anymore (see k-of-m above).
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
    Returns a findings list (only the paragraphs the k-of-m `_is_outlier` decision
    flags), not an annotation of every paragraph — empty when the document is too short
    to support outlier statistics (`MIN_PARAGRAPHS` / `MIN_WORDS_PER_PARAGRAPH`) or when
    no paragraph deviates on enough dimensions to be flagged.
    """
    results: list[OutlierResult] = []
    for paragraph_id, feature_scores in score_paragraphs(fingerprints, strategy):
        if not feature_scores:
            continue

        outlier_score = max(score for _name, score in feature_scores)
        if not _is_outlier(feature_scores):
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
                paragraph_id=paragraph_id,
                outlier_score=outlier_score,
                top_deviating_features=top_features,
            )
        )

    return results


def score_paragraphs(
    fingerprints: list[ParagraphFingerprint],
    strategy: OutlierStrategy = OutlierStrategy.ROBUST_ZSCORE,
) -> list[tuple[str, list[tuple[str, float]]]]:
    """Per-paragraph raw feature z-scores for every ELIGIBLE paragraph, before any
    flag decision is applied.

    Returns `[(paragraph_id, [(feature_name, robust_z), ...]), ...]` for the
    paragraphs that clear the `MIN_WORDS_PER_PARAGRAPH` / `MIN_PARAGRAPHS` eligibility
    floors — empty when the document is too short. This is the single scoring code
    path shared by `detect_outliers` (which applies `_is_outlier` on top) and the
    threshold-derivation script (`poc/calibration/derive_outlier_threshold.py`),
    guaranteeing the derived decision rule is calibrated against exactly the z-scores
    production sees — no re-implementation drift.
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

    return [
        (fp.paragraph_id, _score_paragraph_features(index, fp, eligible))
        for index, fp in enumerate(eligible)
    ]


def _is_outlier(feature_scores: list[tuple[str, float]]) -> bool:
    """The paragraph-level flag decision: k-of-m — a paragraph is an outlier only
    when at least `OUTLIER_MIN_DEVIATING_FEATURES` of its per-feature robust z-scores
    exceed `OUTLIER_PER_FEATURE_Z_THRESHOLD`.

    Provenance: both constants are empirically calibrated to hold per-paragraph
    false-positive rate <= 5% on genuine single-author human prose — see
    `poc/calibration/derive_outlier_threshold.py` (the derivation script) and its
    committed evidence `poc/calibration/outlier_threshold_derivation.json`. A pure
    max-threshold rule (K=1) is the degenerate special case; requiring K>=2 is what
    suppresses the dominant human false-positive mode (a single feature spiking on
    the zero-spread sentinel or a small-n MAD tail) while still catching a genuinely
    off-voice paragraph, which deviates on several correlated dimensions at once.
    """
    deviating = sum(
        1 for _name, score in feature_scores if score > OUTLIER_PER_FEATURE_Z_THRESHOLD
    )
    return deviating >= OUTLIER_MIN_DEVIATING_FEATURES


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
