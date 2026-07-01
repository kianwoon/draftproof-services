"""Additive DeBERTa AI-signal composer — a second, independent AI-writing detector score.

STRICTLY ADDITIVE: it never feeds back into the tier, ai_likelihood_score, the external
estimate, or any gate — same contract as ``authenticity_dashboard.py`` and ``submission_risk.py``.
The off-the-shelf checkpoint runs locally on the worker (no third-party text upload). The score
is optionally calibrated (isotonic on SCoCESLE) and band-mapped onto the composite's traffic-light
legend (green/amber/orange/red) so the two scores are directly comparable.

Output schema (always present when the kill-switch is on; ``available=False`` on any failure):
  score (0-100 float|int|None), band (green|amber|orange|red|None),
  confidence (low|medium), model_version (str), calibrated (bool), available (bool), caveat (str)
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

MODEL_VERSION = "deberta_signal_v1"
_MIN_WORDS = 150

from .deberta_model import score_windows  # module-level so tests can monkeypatch
from .deberta_windowing import split_sentences, build_windows, aggregate
from .deberta_calibrate import map_to_band, apply_isotonic, load_calibrator


def _enabled() -> bool:
    return os.getenv("DRAFTPROOF_DEBERTA_SIGNAL", "1").strip().lower() in {"1", "true", "yes", "on"}


def _word_count(text: str) -> int:
    return len((text or "").split())


def _calibrated_path() -> str | None:
    """Path to a fitted isotonic calibrator (set in Phase 0 after the SCoCESLE fit), or None = raw."""
    return os.getenv("DRAFTPROOF_DEBERTA_CALIBRATOR")


def compose(text: str) -> dict:
    """Score one document end-to-end. Always returns a schema dict; available=False on any failure."""
    if _word_count(text) < _MIN_WORDS:
        return {
            "score": None, "band": None, "confidence": "low",
            "model_version": MODEL_VERSION, "calibrated": False,
            "available": False,
            "caveat": f"too short (need >= {_MIN_WORDS} words for windowed signal)",
        }

    sents = split_sentences(text)
    windows = build_windows(sents, size=3, step=1)
    probs = score_windows(windows)
    if probs is None:
        return {
            "score": None, "band": None, "confidence": "low",
            "model_version": MODEL_VERSION, "calibrated": False,
            "available": False,
            "caveat": "detector unavailable (model load or inference failed)",
        }

    agg = aggregate(sents, windows, probs, size=3, step=1)
    doc_raw = agg["document_score"]

    cal_path = _calibrated_path()
    calibrated = False
    try:
        if cal_path and os.path.exists(cal_path):
            iso = load_calibrator(cal_path)
            doc_cal = apply_isotonic([doc_raw], iso)[0]
            calibrated = True
        else:
            doc_cal = doc_raw
    except Exception as e:  # noqa: BLE001 — fall back to raw rather than fail the field
        logger.warning("[deberta] calibration apply failed, using raw: %s", e)
        doc_cal = doc_raw

    score_100 = max(0.0, min(100.0, doc_cal * 100.0))
    band = map_to_band(score_100)
    caveat = (
        "calibrated on DraftProof ESL corpus"
        if calibrated
        else "raw checkpoint probability, uncalibrated — advisory only"
    )
    confidence = "medium" if calibrated else "low"

    return {
        "score": round(score_100, 1), "band": band, "confidence": confidence,
        "model_version": MODEL_VERSION, "calibrated": calibrated,
        "available": True, "caveat": caveat,
    }


def maybe_attach(text: str) -> dict | None:
    """Return the composed field, or None when the kill-switch is off. Fail-open on any error."""
    if not _enabled():
        return None
    try:
        return compose(text)
    except Exception as e:  # noqa: BLE001 — never break the scan
        logger.warning("[deberta] compose failed (fail-open): %s", e)
        return {
            "score": None, "band": None, "confidence": "low",
            "model_version": MODEL_VERSION, "calibrated": False,
            "available": False, "caveat": f"error: {type(e).__name__}",
        }
