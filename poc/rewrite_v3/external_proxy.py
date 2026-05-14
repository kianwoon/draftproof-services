"""External-detector proxy gates for rewrite V3 selection."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExternalProxyDecision:
    accepted: bool
    family: str
    reasons: list[str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def evaluate_external_proxy(
    *,
    family: str,
    reference_ai: float | None,
    candidate_ai: float | None,
    reference_wq: float | None,
    candidate_wq: float | None,
    reference_topk: float | None,
    candidate_topk: float | None,
    compression: dict[str, Any],
    validation_passed: bool,
    compression_accepted: bool,
    semantic_safe: bool,
) -> ExternalProxyDecision:
    """Decide whether a non-strict V3 candidate is safe enough to apply.

    This is intentionally stricter than the public goal contract. It exists
    only for externally calibrated candidates where internal DraftProof scores
    are known to be incomplete proxies for third-party detectors.
    """

    normalized_family = str(family or "document_rhythm")
    ref_ai = _number(reference_ai)
    cand_ai = _number(candidate_ai)
    ref_wq = _number(reference_wq)
    cand_wq = _number(candidate_wq)
    ref_topk = _number(reference_topk)
    cand_topk = _number(candidate_topk)
    ai_delta = None if ref_ai is None or cand_ai is None else ref_ai - cand_ai
    wq_delta = None if ref_wq is None or cand_wq is None else ref_wq - cand_wq
    topk_delta = None if ref_topk is None or cand_topk is None else ref_topk - cand_topk
    status = str(compression.get("status") or "")

    reasons: list[str] = []
    if not validation_passed:
        reasons.append("validation_failed")
    if not compression_accepted:
        reasons.append("compression_rejected")
    if not semantic_safe:
        reasons.append("semantic_drift")
    if ref_ai is None or cand_ai is None:
        reasons.append("missing_ai_scores")

    max_internal_backfire = _float_env("DRAFTPROOF_REWRITE_V3_MAX_INTERNAL_AI_BACKFIRE", 3.0)
    if ref_ai is not None and cand_ai is not None and cand_ai > ref_ai + max_internal_backfire:
        reasons.append("internal_ai_backfire")

    if normalized_family == "document_rhythm":
        min_topk_drop = _float_env("DRAFTPROOF_REWRITE_V3_RHYTHM_MIN_TOPK_DROP", 8.0)
        max_wq_drop = _float_env("DRAFTPROOF_REWRITE_V3_RHYTHM_MAX_WQ_DROP", 6.0)
        if topk_delta is None or topk_delta < min_topk_drop:
            reasons.append("insufficient_topk_drop")
        if wq_delta is not None and wq_delta > max_wq_drop:
            reasons.append("writing_quality_collapse")
        if status == "above_ceiling":
            reasons.append("broad_candidate_too_long")

    metrics = {
        "reference_ai": ref_ai,
        "candidate_ai": cand_ai,
        "ai_delta": ai_delta,
        "reference_wq": ref_wq,
        "candidate_wq": cand_wq,
        "wq_delta": wq_delta,
        "reference_topk": ref_topk,
        "candidate_topk": cand_topk,
        "topk_delta": topk_delta,
        "compression_status": status,
    }
    return ExternalProxyDecision(
        accepted=not reasons,
        family=normalized_family,
        reasons=reasons,
        metrics=metrics,
    )
