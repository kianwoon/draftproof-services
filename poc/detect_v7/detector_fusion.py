"""V7 detector fusion — combines whichever calibrated AI-detector scores are
available (quick scan: composite alone; deep scan: composite + deberta_large)
into a single ``calibrated_detector_score`` in [0, 1].

The "composite" input is the badge's composite calibrated
``ai_likelihood_score`` (which may itself be DeBERTa-authoritative when that
path fires) — the only calibrated detector score available at the quick-scan
call site; there is no isolated raw-fakespot signal at this call site (see
``pipeline_bridge.py``'s module docstring for the full trace).

This module performs FUSION ONLY, not calibration: every input score is
assumed to already be calibrated/ESL-safe (e.g. the composite's isotonic
calibrator, or an equivalently calibrated deberta_large score). No weight or
threshold literal lives in this file — every number comes from
``detect_v7/weights.json`` via ``config.get_fusion_weights`` (CLAUDE.md
no-hardcode rule).

Unsupported detector combinations are NOT silently coerced to a nearby known
set; ``config.get_fusion_weights`` raises ``ValueError`` and this module lets
that propagate (fail-loud, precision-first — see project rules).
"""
from __future__ import annotations

from typing import Optional

from . import config


def compute_calibrated_detector_score(
    detector_scores: dict[str, float],
) -> tuple[float, dict]:
    """Fuse the given calibrated detector scores into one score.

    Parameters
    ----------
    detector_scores:
        Mapping of detector name -> calibrated score in [0, 1]. Recognized
        combinations (see ``config.get_fusion_weights``): {"composite"}
        (quick scan) or {"composite", "deberta_large"} (deep scan, 2-detector).
        Any other combination raises ``ValueError`` (propagated from
        ``config.get_fusion_weights``, not swallowed here).

    Returns
    -------
    (calibrated_detector_score, detail) where ``detail`` documents which
    fusion weight set was used and the per-detector contribution, for
    debugging/audit — it carries no additional scoring logic of its own.
    """
    weight_map = config.get_fusion_weights(sorted(detector_scores.keys()))

    contributions: dict[str, float] = {}
    total = 0.0
    for detector_name, weight in weight_map.items():
        score = detector_scores[detector_name]
        contribution = weight * score
        contributions[detector_name] = contribution
        total += contribution

    detail = {
        "weights_used": dict(weight_map),
        "contributions": contributions,
        "detector_scores": dict(detector_scores),
    }
    return total, detail


def compute_detector_disagreement(
    detector_scores: dict[str, float],
) -> Optional[float]:
    """Fusion-layer detector disagreement: max - min spread across the given
    calibrated detector scores.

    IMPORTANT — this is DIFFERENT from the ``detector_disagreement`` V7
    signal produced by ``signal_adapter.adapt_paragraph_signals``. That
    adapter signal is a pass-through of ``poc/detect/``'s existing
    ``detector_agreement_risk`` GROUP score, which measures agreement among
    the OLD, larger detector ensemble used throughout ``poc/detect/``. This
    function instead measures disagreement specifically among the 1-2
    calibrated detectors available to the V7 deep-scan fusion path
    (composite / deberta_large) -- a narrower, deep-scan-specific notion the
    old ensemble signal does not capture. ``category_scoring.py`` should
    prefer the adapter's ``detector_disagreement`` signal as the primary
    source for the ESL co-trigger guard; this function is provided
    separately for callers that specifically want 2-detector fusion-level
    disagreement (e.g. deep-scan diagnostics/audit).

    Uses max - min (simple spread) rather than std: with at most 2 detectors
    in the currently supported combinations, max - min and a 2-point std are
    monotonic transforms of each other, so max - min is preferred for being
    directly interpretable as "how far apart are the two scores".

    Returns None when fewer than 2 detector scores are available (disagreement
    is undefined for a single detector).
    """
    if len(detector_scores) < 2:
        return None
    values = list(detector_scores.values())
    return max(values) - min(values)
