"""Isotonic recalibration + band mapping for the DeBERTa signal.

Pure math. ``fit_isotonic`` is run on the SCoCESLE corpus (Phase 0 Task 0.5); the fitted
calibrator is applied at inference time. ``map_to_band`` puts the score on the SAME band
legend the composite shows the user, so the two scores are directly comparable.

Band legend + cutoffs are anchored to the composite, not invented here:
  - Internal ai_score->Tier thresholds: poc/detect/layer3_scoring.py::_derive_ai_tier
    (<0.32 GREEN, <0.48 AMBER, <0.65 ORANGE, else RED).
  - Frontend render of those keys: draftproof-frontend/src/pages/report/reportHelpers.js
    (green=Low, amber=Moderate, orange=High, red=Critical Risk) and i18n/en/report.js
    "tiers". The score is on a 0-100 scale, so the 0.32/0.48/0.65 cutoffs become 32/48/65.
"""
from __future__ import annotations

import pickle
from pathlib import Path

# Composite's ai_likelihood band cutoffs on the 0-100 scale (mirrors _derive_ai_tier
# thresholds 0.32/0.48/0.65 x100). KEEP IN SYNC with layer3_scoring._derive_ai_tier and
# reportHelpers.js — if the composite's thresholds change, update these too.
_BAND_CUTOFFS = [(32.0, "green"), (48.0, "amber"), (65.0, "orange")]
# < 32 -> green, [32,48) -> amber, [48,65) -> orange, >= 65 -> red


def fit_isotonic(raw_scores, labels):
    """Fit an isotonic regressor mapping raw model prob -> calibrated prob (monotonic).

    sklearn is imported lazily so this pure-math module stays importable without the ML
    stack (matches the test/unit-isolation goal)."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_scores, labels)
    return iso


def save_calibrator(iso, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(iso, f)


def load_calibrator(path: str):
    # SECURITY: pickle is required to deserialize a fitted sklearn IsotonicRegression
    # (numpy internals, no clean JSON form). Safe here because the .pkl is produced by our
    # own fit step (poc/calibration/deberta_fit_calibrator.py) on the local SCoCESLE corpus
    # and committed to this repo — it is NEVER loaded from user uploads or third parties.
    # If that trust boundary ever changes, switch to JSON of X_thresholds_/y_thresholds_ + np.interp.
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_isotonic(raw_scores, iso):
    return [float(x) for x in iso.predict(raw_scores)]


def map_to_band(score_100: float) -> str:
    """Map a 0-100 AI-like score to the composite's traffic-light band for comparability."""
    for cutoff, band in _BAND_CUTOFFS:
        if score_100 < cutoff:
            return band
    return "red"
