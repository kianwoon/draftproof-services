"""Additive Authorship Clarity Breakdown composer (Phase 1A).

Reads an already-computed document-level aggregate (``aggregate.py``) plus the
per-paragraph ``category_scoring.score_paragraph()`` outputs and shapes them
into the report-facing dict per ``docs/draftproof_v7_authorship_clarity_spec.md``
§6.1 (display policy) and §8 (surfaces/rollout). STRICTLY ADDITIVE and PURE:
this module never touches ``poc/detect/``, ``worker/``, or
``draftproof-api/`` — wiring it into the scan task / report.py / a kill
switch is a follow-up integration step, out of scope here (mirrors the
house pattern used by ``poc/detect/authenticity_dashboard.py`` and
``poc/detect/submission_risk.py``, but does not import from either).

Phase 1 (D5): the user-facing surfaces show BANDS ONLY, never exact
percentages — the raw shares are still included in the returned dict for
internal/API/future-proofing use (spec: "The API JSON may carry the raw
shares from day one ... the user-facing surfaces show bands").
"""
from __future__ import annotations

from typing import Any

from . import config

MODEL_VERSION = "breakdown_composer_v1"

# Disclaimer text, verbatim per spec §26 (task instructions require the
# exact wording below — do not paraphrase).
DISCLAIMER = (
    "DraftProof provides authorship clarity signals and writing-risk "
    "analysis. It does not determine misconduct. Final judgement belongs "
    "to the school, teacher, or relevant academic policy."
)

_CATEGORY_NAMES: tuple[str, ...] = (
    "student_owned",
    "ai_assisted_polished",
    "ai_paraphrased",
    "ai_generated_like",
)

# Structural-completeness threshold for flagging the document as degraded
# enough that the reader should distrust the breakdown, NOT a scoring
# threshold — it decides whether more than half of the paragraphs had
# missing signals, which is a completeness/coverage check over the document
# structure rather than a tuned scoring weight. weights.json's scope is
# scoring formula numbers (fusion/category weights, thresholds that shape a
# score); this ratio never touches a score, only a display-mode decision.
# Kept as a local, auditable constant rather than added to weights.json for
# that reason (see task instructions).
_DEGRADED_RATIO_THRESHOLD = 0.5

_UNCERTAINTY_FLAG_NO_COMPARISON = "paraphrase_without_original_draft"


def _band_for_share(share: float, bands: dict[str, float]) -> str:
    if share >= bands["strong_min"]:
        return "Strong"
    if share >= bands["some_min"]:
        return "Some"
    if share >= bands["little_min"]:
        return "Little"
    return "None"


def compose_authorship_breakdown(
    document_aggregate: dict[str, Any],
    paragraph_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Shape a document aggregate + paragraph results into the report-facing
    Authorship Clarity Breakdown block.

    Parameters
    ----------
    document_aggregate: output of ``aggregate.aggregate_document()``.
    paragraph_results: the same per-paragraph ``score_paragraph()`` outputs
        that were passed into ``aggregate_document()`` (used here to detect
        the no-comparison-text paraphrase uncertainty flag and the
        mixed-signals presentation ratio).

    Returns
    -------
    dict shaped for report.py consumption — see module docstring / spec §6.1
    and §8 for the field contract. Always includes ``display_mode: "bands"``
    (Phase 1 never shows exact percentages to end users, per spec D5) plus
    the raw shares for internal/API use.
    """
    bands = config.get_display_bands()

    raw_shares = {
        category: document_aggregate[category] for category in _CATEGORY_NAMES
    }
    banded_shares = {
        category: _band_for_share(raw_shares[category], bands)
        for category in _CATEGORY_NAMES
    }

    paragraph_count = document_aggregate["paragraph_count"]
    degraded_paragraph_count = document_aggregate["degraded_paragraph_count"]

    degraded_ratio_high = (
        paragraph_count > 0
        and (degraded_paragraph_count / paragraph_count) > _DEGRADED_RATIO_THRESHOLD
    )
    mixed_signals_predominant = False
    if paragraph_results:
        mixed_count = sum(
            1 for p in paragraph_results if p.get("presentation") == "mixed_signals"
        )
        mixed_signals_predominant = (mixed_count / len(paragraph_results)) > _DEGRADED_RATIO_THRESHOLD

    uncertainty_flags: list[str] = []
    if any(_paragraph_has_no_comparison(p) for p in paragraph_results):
        uncertainty_flags.append(_UNCERTAINTY_FLAG_NO_COMPARISON)

    return {
        "schema_version": "v7_phase1a",
        "document_breakdown_raw": raw_shares,
        "document_breakdown_bands": banded_shares,
        "primary_category": document_aggregate["primary_category"],
        "confidence": _document_confidence(paragraph_results),
        "paragraph_count": paragraph_count,
        "degraded_paragraph_count": degraded_paragraph_count,
        "display_mode": "bands",
        "degraded_display": degraded_ratio_high or mixed_signals_predominant,
        "uncertainty_flags": uncertainty_flags,
        "disclaimer": DISCLAIMER,
    }


def _paragraph_has_no_comparison(paragraph_result: dict[str, Any]) -> bool:
    """A paragraph signals the no-comparison-text paraphrase fallback when its
    ``missing_signals`` includes ``semantic_drift`` — the adapter only marks
    that signal missing (vs. ``None``-but-``not_implemented``) when comparison
    text was unavailable and the paraphrase category fell back to the
    without-comparison weight variant (spec D9).
    """
    return "semantic_drift" in (paragraph_result.get("missing_signals") or [])


def _document_confidence(paragraph_results: list[dict[str, Any]]) -> Any:
    """Document-level confidence: "low" if any paragraph reports low
    confidence, else the confidence of the (first) primary paragraph if all
    agree, else None. This is a straightforward worst-case rollup — not a
    new scoring rule — matching the house convention of pessimistic
    aggregation for confidence-style fields.
    """
    if not paragraph_results:
        return None
    if any(p.get("confidence") == "low" for p in paragraph_results):
        return "low"
    return paragraph_results[0].get("confidence")
