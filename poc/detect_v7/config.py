"""Config loader for V7 Authorship Clarity Breakdown.

Loads ``weights.json`` (fusion weights, category weights, thresholds) and
exposes typed accessors. This module contains NO numeric weight/threshold
literals of its own — every scoring number lives in ``weights.json`` per the
project's no-hardcode rule (CLAUDE.md). Adding a weight here as a Python
literal instead of a JSON value is a rule violation.

Path resolution:
- Default: ``weights.json`` bundled next to this file (``Path(__file__).parent``),
  never the process cwd.
- Override: set ``DRAFTPROOF_V7_WEIGHTS_PATH`` to point at an alternate file
  (used by tests and future calibration runs).

A missing weights file is a hard error (``FileNotFoundError``) — this module
never silently falls back to built-in defaults.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_WEIGHTS_ENV_VAR = "DRAFTPROOF_V7_WEIGHTS_PATH"
_BUNDLED_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.json"

# Module-level cache of the loaded weights dict, keyed by the resolved path
# so switching DRAFTPROOF_V7_WEIGHTS_PATH is visible after a reload.
_cache: dict[str, Any] | None = None
_cache_path: Path | None = None


def _resolve_path() -> Path:
    override = os.environ.get(_WEIGHTS_ENV_VAR)
    if override:
        return Path(override)
    return _BUNDLED_WEIGHTS_PATH


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"V7 weights file not found at resolved path: {path} "
            f"(set via {_WEIGHTS_ENV_VAR} or bundled default "
            f"{_BUNDLED_WEIGHTS_PATH}). Refusing to fall back to built-in "
            f"defaults — see poc/detect_v7/config.py precision-first rule."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reload_weights(force: bool = True) -> dict[str, Any]:
    """(Re)load weights.json from the currently resolved path.

    Always re-resolves the path (so an updated DRAFTPROOF_V7_WEIGHTS_PATH env
    var takes effect). If force is False and the cache already matches the
    resolved path, the cached dict is returned without re-reading the file.
    """
    global _cache, _cache_path
    path = _resolve_path()
    if not force and _cache is not None and _cache_path == path:
        return _cache
    data = _load(path)
    _cache = data
    _cache_path = path
    return data


def _weights() -> dict[str, Any]:
    if _cache is None:
        return reload_weights(force=True)
    # Re-resolve to detect an env var change since the last load.
    if _cache_path != _resolve_path():
        return reload_weights(force=True)
    return _cache


def get_fusion_weights(detectors_available: list[str]) -> dict[str, float]:
    """Pick the fusion weight set matching the given available detectors.

    Recognized combinations:
    - {"composite"}                    -> quick_scan
    - {"composite", "deberta_large"}   -> deep_scan_2detector

    "composite" is the badge's composite calibrated ai_likelihood_score
    (renamed from the historically mislabeled "fakespot" — see
    ``weights.json``'s ``_notes.fusion_weights`` and
    ``pipeline_bridge.py``'s module docstring for the full trace).

    Any other combination (including empty, or containing unrecognized
    detector names, or the 3-detector inert set) raises ValueError — this
    function never guesses a fallback weight set.
    """
    available = set(detectors_available)
    fusion = _weights()["fusion_weights"]

    if available == {"composite"}:
        return dict(fusion["quick_scan"])
    if available == {"composite", "deberta_large"}:
        return dict(fusion["deep_scan_2detector"])

    raise ValueError(
        f"Unsupported/unrecognized detector combination for fusion: "
        f"{sorted(available)!r}. Supported combinations: "
        f"['composite'] (quick_scan) or ['composite', 'deberta_large'] "
        f"(deep_scan_2detector). The 3-detector set is inert/unused "
        f"(spec §4, OU-Advacheck on hold) and cannot be selected here."
    )


def get_category_weights(
    category: str, has_comparison_text: bool = False
) -> list[tuple[str, float, str]]:
    """Return [(signal_name, weight, direction), ...] for the given category.

    ``direction`` is one of "direct" (use the signal value as-is),
    "inverse" (use 1 - signal_value), or "midrange" (use the
    ai_assisted_polished_band low/high thresholds via get_ai_assisted_polished_band()
    instead of the raw value). Getting this wrong silently flips the category
    score, so callers MUST branch on it rather than assuming "direct".

    For category == "ai_paraphrased", selects the with/without-comparison
    variant based on has_comparison_text (spec D9). Other categories ignore
    the flag. Unknown category raises ValueError.
    """
    category_weights = _weights()["category_weights"]

    if category == "ai_paraphrased":
        key = (
            "ai_paraphrased_with_comparison"
            if has_comparison_text
            else "ai_paraphrased_without_comparison"
        )
    elif category in (
        "student_owned",
        "ai_assisted_polished",
        "ai_generated_like",
    ):
        key = category
    else:
        raise ValueError(
            f"Unknown V7 category: {category!r}. Expected one of "
            f"'student_owned', 'ai_assisted_polished', 'ai_paraphrased', "
            f"'ai_generated_like'."
        )

    if key not in category_weights:
        raise ValueError(
            f"Category key {key!r} (resolved from category={category!r}, "
            f"has_comparison_text={has_comparison_text!r}) not present in "
            f"weights.json category_weights."
        )

    return [
        (entry["signal"], entry["weight"], entry.get("direction", "direct"))
        for entry in category_weights[key]
    ]


def get_flatness_thresholds() -> dict[str, float]:
    return dict(_weights()["flatness_thresholds"])


def get_esl_guard_config() -> dict[str, Any]:
    return dict(_weights()["esl_guard"])


def get_deep_scan_calibration() -> dict[str, Any]:
    return dict(_weights()["deep_scan_calibration"])


def get_deep_scan_display_bands() -> dict[str, float]:
    """Display-band cutoffs for the user-facing 'Deep-scan AI estimate'
    (bands: insufficient / amber / orange / red — NO 'green' exists, per the
    poc/detect/deberta_signal.py philosophy: below the reliability floor is
    'insufficient evidence', never 'clean').

    Enforces the contract that ``insufficient_below`` equals
    ``deep_scan_calibration.doc_floor`` (both read from weights.json — the
    check compares two config values, it hardcodes neither). A mismatch is a
    config error and raises ValueError rather than silently banding with a
    floor that disagrees with the reliability floor.
    """
    calibration = _weights()["deep_scan_calibration"]
    bands = dict(calibration["display_bands"])
    if bands["insufficient_below"] != calibration["doc_floor"]:
        # allow-hardcode: human-readable error message text, not a scoring/matching list
        raise ValueError(
            f"weights.json contract violation: display_bands.insufficient_below "
            f"({bands['insufficient_below']!r}) must equal deep_scan_calibration.doc_floor "
            f"({calibration['doc_floor']!r}) — below the reliability floor is "
            f"insufficient evidence, never a band."
        )
    return bands


def get_display_bands() -> dict[str, float]:
    return dict(_weights()["display_bands"])


def get_ai_assisted_polished_band() -> dict[str, float]:
    return dict(_weights()["ai_assisted_polished_band"])


def get_paraphrase_mismatch_normalization() -> dict[str, float]:
    """Return the {p10, p90} quantile-normalization bounds for the
    ``paraphrase_mismatch`` derived signal (V8 Task 5, the Phase A interaction
    winner ``specificity_student_evidence × calibrated_detector_score``).

    ``signal_adapter.py`` row 9c reads these to spread the compressed raw
    product across [0,1] via ``clamp01((p - p10) / (p90 - p10))``. The values
    are the study percentiles from
    ``poc/calibration/v12_validation/phase_a_interaction_study.json``; keeping
    them in ``weights.json`` (not as Python literals) upholds the no-hardcode
    rule. A missing block raises ``KeyError`` (fail loud), matching
    ``get_display_consistency_guard_config``'s direct-lookup pattern.
    """
    return dict(_weights()["paraphrase_mismatch_normalization"])


def get_tier_authority_config() -> dict[str, Any]:
    """Return the fused-score tier-authority config (weights + cutoffs).

    Validates that cutoffs are strictly ascending (amber < orange < red) —
    a config error here must fail loudly rather than silently produce a
    tier ordering that contradicts itself. Raises ``ValueError`` on a
    malformed/non-ascending cutoff set.
    """
    section = _weights()["tier_authority"]
    cutoffs = section["cutoffs"]
    amber, orange, red = cutoffs["amber"], cutoffs["orange"], cutoffs["red"]
    if not (amber < orange < red):
        # allow-hardcode: human-readable error message text, not a scoring/matching list
        raise ValueError(
            f"weights.json contract violation: tier_authority.cutoffs must be "
            f"strictly ascending (amber < orange < red), got amber={amber!r}, "
            f"orange={orange!r}, red={red!r}."
        )
    return {
        "weights": dict(section["weights"]),
        "cutoffs": dict(cutoffs),
    }


def get_display_consistency_guard_config() -> dict[str, Any]:
    """Return the tier-consistency display guard config (badge tiers that
    trigger the ``student_owned``-primary contradiction annotation — see
    ``weights.json``'s ``display_consistency_guard._notes`` for the
    measured motivation). A missing block raises ``KeyError`` (fail loud),
    matching ``get_esl_guard_config``/``get_deep_scan_calibration``'s
    unvalidated direct-lookup pattern — this section has no derived-value
    contract to additionally check, unlike ``get_tier_authority_config``'s
    ascending-cutoffs invariant.
    """
    return dict(_weights()["display_consistency_guard"])
