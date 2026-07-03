"""V7 document-level aggregation — word-count-weighted mean of per-paragraph
Authorship Clarity Breakdown category shares, per
``docs/draftproof_v7_authorship_clarity_spec.md`` §6.1 ("Document aggregation:
word-count-weighted mean of paragraph shares (v2 §17 unchanged)").

Pure aggregation only: no new weights/thresholds live here. The only numbers
this module produces are derived directly from the ``score_paragraph`` outputs
and the caller-supplied word counts (CLAUDE.md no-hardcode rule).
"""
from __future__ import annotations

from typing import Any

_CATEGORY_NAMES: tuple[str, ...] = (
    "student_owned",
    "ai_assisted_polished",
    "ai_paraphrased",
    "ai_generated_like",
)


def aggregate_document(
    paragraph_results: list[dict[str, Any]],
    paragraph_word_counts: list[int],
) -> dict[str, Any]:
    """Word-count-weighted mean of paragraph-level category scoring.

    Parameters
    ----------
    paragraph_results: one ``category_scoring.score_paragraph()`` output dict
        per paragraph, in the same order as ``paragraph_word_counts``.
    paragraph_word_counts: qualifying word count per paragraph, same order/
        length as ``paragraph_results``.

    Returns
    -------
    dict with the 4 document-level category shares (sum to 1.0),
    ``paragraph_count``, ``degraded_paragraph_count`` (paragraphs where
    ``degraded`` is True), and ``primary_category`` (argmax of the
    document-level shares).

    Raises
    ------
    ValueError: empty input, mismatched list lengths, or zero total word
        count. This function fails loud rather than returning a nonsense/NaN
        result (precision-first rule) — a caller passing degenerate input has
        a bug that must surface immediately, not silently produce a document
        breakdown that looks valid but isn't.
    """
    if not paragraph_results:
        raise ValueError(
            "aggregate_document: paragraph_results is empty — cannot "
            "aggregate a document with zero paragraphs."
        )
    if len(paragraph_results) != len(paragraph_word_counts):
        raise ValueError(
            "aggregate_document: paragraph_results and paragraph_word_counts "
            f"must be the same length (got {len(paragraph_results)} vs "
            f"{len(paragraph_word_counts)}) — inputs are parallel lists, one "
            "entry per paragraph, in the same order."
        )

    total_words = sum(paragraph_word_counts)
    if total_words <= 0:
        raise ValueError(
            "aggregate_document: total qualifying word count across all "
            f"paragraphs is {total_words} — cannot compute a word-count-"
            "weighted mean with zero (or negative) total words."
        )

    document_shares: dict[str, float] = {category: 0.0 for category in _CATEGORY_NAMES}
    degraded_paragraph_count = 0

    for paragraph_result, word_count in zip(paragraph_results, paragraph_word_counts):
        paragraph_weight = word_count / total_words
        for category in _CATEGORY_NAMES:
            document_shares[category] += paragraph_result[category] * paragraph_weight
        if paragraph_result.get("degraded"):
            degraded_paragraph_count += 1

    primary_category = max(document_shares.items(), key=lambda item: item[1])[0]

    return {
        "student_owned": document_shares["student_owned"],
        "ai_assisted_polished": document_shares["ai_assisted_polished"],
        "ai_paraphrased": document_shares["ai_paraphrased"],
        "ai_generated_like": document_shares["ai_generated_like"],
        "paragraph_count": len(paragraph_results),
        "degraded_paragraph_count": degraded_paragraph_count,
        "primary_category": primary_category,
    }
