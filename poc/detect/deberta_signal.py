"""Additive DeBERTa AI-signal composer — a second, independent AI-writing detector score.

STRICTLY ADDITIVE: it never feeds back into the tier, ai_likelihood_score, the external
estimate, or any gate — same contract as ``authenticity_dashboard.py`` and ``submission_risk.py``.
The off-the-shelf checkpoint runs locally on the worker (no third-party text upload).

DESIGN (v2, threshold-proportion — derived from the Turnitin AI-detection technical breakdown):
  The document is broken into overlapping sentence windows; each window is scored 0-1 by
  the checkpoint; each sentence inherits the mean of its covering windows. A sentence is
  "AI-like" if its score >= SENT_THRESHOLD (0.99 — the model's high-confidence band where
  ESL false-positives are rare: ~3.3% doc-level ESL FPR at this threshold on SCoCESLE).
  The document signal = the PROPORTION of sentences that are AI-like, expressed as a %.

  Below DOC_FLOOR_PCT (20) the signal is reported as "insufficient" — no verdict, only the
  flagged passages. This mirrors Turnitin's suppression of the unreliable 1-19% band: a
  low proportion of high-confidence sentences is not stable enough to call "green/safe",
  and the honest output is "here are the few passages to review", not a confident verdict.

Why threshold-proportion, not mean+calibration: averaging window scores (the v1 approach)
let a degenerate isotonic calibrator collapse any non-perfect score to ~0 (the production
0%/14% bugs). Proportion-over-threshold has no averaging and no calibration to degenerate;
the number is explainable ("N% of your sentences the detector is >=99% sure are AI") and
structurally cannot produce that failure mode.

Honest limit: this is a post-hoc statistical detector. It localizes obvious/clean AI text
well but, like all such detectors, misses paraphrased/humanized text and carries residual
ESL bias. It is advisory only.

Output schema (always present when the kill-switch is on; ``available=False`` on failure):
  signal_pct (int 0-100|None), sentences_scored (int), sentences_flagged (int),
  flagged_passages (list[{sentence_id,score,text}]), band (insufficient|amber|orange|red|None),
  confidence (low|medium), model_version (str), available (bool), caveat (str)
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

MODEL_VERSION = "deberta_signal_v2"
_MIN_WORDS = 150

# A sentence is "AI-like" if its window-mean probability >= this. 0.99 is the model's
# high-confidence band: on SCoCESLE it yields ~3.3% document-level ESL FPR (vs ~21% at
# 0.80), because clean AI text piles up at ~1.0 while humans rarely produce a sentence
# this confident. Tightening this lowers ESL FPR and AI recall together.
SENT_THRESHOLD = 0.99
# Below this proportion of AI-like sentences, the signal is "insufficient" for a verdict
# (band = "insufficient"); only the flagged passages are surfaced. Mirrors Turnitin's
# suppression of the 1-19% band (unreliable). >= floor -> a real band (amber/orange/red).
DOC_FLOOR_PCT = 20
# Cap on the number of flagged passages persisted, to bound payload size. Highest-scoring
# first; the count fields always reflect the true total even when the list is truncated.
_MAX_FLAGGED_PERSISTED = 10

from .deberta_model import score_windows  # module-level so tests can monkeypatch
from .deberta_windowing import split_sentences, build_windows, aggregate


def _enabled() -> bool:
    return os.getenv("DRAFTPROOF_DEBERTA_SIGNAL", "1").strip().lower() in {"1", "true", "yes", "on"}


def _word_count(text: str) -> int:
    return len((text or "").split())


def _band_for(signal_pct: float) -> str:
    """Map the proportion signal to a band. No "green": a signal that fires is never
    declared "safe" — below floor is explicitly "insufficient evidence", not "clean"."""
    if signal_pct is None:
        return "insufficient"
    if signal_pct < DOC_FLOOR_PCT:
        return "insufficient"
    if signal_pct < 40:
        return "amber"
    if signal_pct < 65:
        return "orange"
    return "red"


def compose(text: str) -> dict:
    """Score one document end-to-end. Always returns a schema dict; available=False on failure."""
    if _word_count(text) < _MIN_WORDS:
        return {
            "signal_pct": None, "sentences_scored": 0, "sentences_flagged": 0,
            "flagged_passages": [], "band": "insufficient",
            "confidence": "low", "model_version": MODEL_VERSION, "available": False,
            "caveat": f"too short (need >= {_MIN_WORDS} words for windowed signal)",
        }

    sents = split_sentences(text)
    windows = build_windows(sents, size=3, step=1)
    probs = score_windows(windows)
    if probs is None:
        return {
            "signal_pct": None, "sentences_scored": 0, "sentences_flagged": 0,
            "flagged_passages": [], "band": "insufficient",
            "confidence": "low", "model_version": MODEL_VERSION, "available": False,
            "caveat": "detector unavailable (model load or inference failed)",
        }

    agg = aggregate(sents, windows, probs, size=3, step=1)
    sentence_scores = agg["sentence_scores"]

    # Build the headline from per-sentence scores via the SHARED helper. The naive splitter
    # here produces integer-indexed sentences (no canonical sNNN); the helper accepts any
    # (sentence_id, score, text) tuples, so we pass the integer index as the id.
    scored_tuples = [
        (i, s, sents[i])
        for i, s in enumerate(sentence_scores) if s is not None
    ]
    return _build_headline(scored_tuples)


def _build_headline(scored_sentences: list[tuple]) -> dict:
    """Shared headline builder. Takes [(sentence_id, score, text), ...] and returns the full
    ai_signal_deberta dict (signal_pct, flagged_passages, band, caveat, ...). Used by both
    compose() (naive-splitter sentences) and the report's canonical-sentence path, so the
    headline math is IDENTICAL regardless of which sentence splitter produced the inputs —
    eliminating the tile-vs-map inconsistency that arose from two different splitters."""
    n_scored = len(scored_sentences)
    flagged = [(sid, s, txt) for sid, s, txt in scored_sentences if s >= SENT_THRESHOLD]
    n_flagged = len(flagged)
    signal_pct = round(100.0 * n_flagged / n_scored) if n_scored else 0

    # Flagged passages: highest score first, capped. Carry the sentence_id the caller passed
    # (canonical sNNN from the report, or an integer index from compose()'s naive split).
    flagged_sorted = sorted(flagged, key=lambda t: t[1], reverse=True)[:_MAX_FLAGGED_PERSISTED]
    flagged_passages = [
        {"sentence_id": sid, "score": round(float(s), 3), "text": txt}
        for sid, s, txt in flagged_sorted
    ]

    band = _band_for(signal_pct)
    above_floor = band != "insufficient"
    confidence = "medium" if above_floor else "low"
    caveat = (
        f"{signal_pct}% of sentences score >= {SENT_THRESHOLD:.2f} under a second detector. "
        "Advisory only — post-hoc detectors miss paraphrased text and carry residual ESL bias; "
        "review the flagged passages rather than treating the number as a verdict."
        if above_floor else
        f"{n_flagged} of {n_scored} sentences score >= {SENT_THRESHOLD:.2f} under a second "
        f"detector — below the {DOC_FLOOR_PCT}% reliability floor, so this signal offers no "
        "overall verdict. A high bar was set to avoid unfairly flagging fluent ESL writing; "
        "other detectors with a lower bar may read the flagged passages (and the document) "
        "differently. Review the flagged passages rather than treating 'no verdict' as clean."
    )

    return {
        "signal_pct": signal_pct, "sentences_scored": n_scored,
        "sentences_flagged": n_flagged, "flagged_passages": flagged_passages,
        "band": band, "confidence": confidence, "model_version": MODEL_VERSION,
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
            "signal_pct": None, "sentences_scored": 0, "sentences_flagged": 0,
            "flagged_passages": [], "band": "insufficient",
            "confidence": "low", "model_version": MODEL_VERSION, "available": False,
            "caveat": f"error: {type(e).__name__}",
        }


# ─── Per-sentence heatmap (for the Signal-highlights paragraph map) ───────────
#
# The headline compose() collapses to one document number + the few ≥0.99 flags. But the
# "Submitted Content / Signal highlights" map needs a RELATIVE intensity per sentence — a
# graduated heatmap — so the user can see which passages read more AI-like than others, not
# just the binary top-confidence flags (Turnitin's graduated per-sentence scoring, doc lines
# 82-87). compose_from_sentences() returns that per-sentence score map, keyed to the report's
# CANONICAL sentence ids (sNNN/pNNN), so the highlight layer can join without fragile text-match.

# Graduated heatmap bands (per-sentence DeBERTa score -> band). Distinct from the document
# _band_for() above: that maps the headline proportion; this maps a single sentence's score.
# clean = neutral (most human writing), low/moderate = graduated amber/orange (AI-like but
# below the high-confidence bar), high = the ≥0.99 band (same threshold as the headline).
_HEAT_BAND_CUTOFFS = [(0.50, "clean"), (0.80, "low"), (0.99, "moderate")]  # >=0.99 -> high


def band_for_sentence(score: float) -> str:
    """Map a per-sentence DeBERTa score to a heatmap band: clean|low|moderate|high.

    Distinct from _band_for() (which maps the document-level proportion signal). This is the
    graduated per-sentence scale used to color the Signal-highlights paragraph map."""
    if score is None:
        return "clean"
    for cutoff, band in _HEAT_BAND_CUTOFFS:
        if score < cutoff:
            return band
    return "high"


def compose_from_sentences(sentences: list[dict]) -> dict | None:
    """Score pre-segmented sentences and return a per-sentence heatmap, keyed to canonical ids.

    Each input dict carries {sentence_id (sNNN), paragraph_id (pNNN), text}. We window/aggregate
    THOSE exact sentences (no re-splitting) so the returned scores align 1:1 with the report's
    sentence structure — avoiding the naive-splitter id mismatch. Returns None on any failure
    (fail-open; the caller falls back to no heatmap).

    Output: {"sentence_scores": [{sentence_id, paragraph_id, score, band}, ...],
             "available": bool, "model_version": str}
    The score is the mean of covering windows (same aggregate() as compose); band is via
    band_for_sentence(). Sentences too short to window are returned with score None / band clean.
    """
    if not _enabled() or not sentences:
        return None
    try:
        texts = [str(s.get("text") or "").strip() for s in sentences]
        if not any(texts):
            return None
        windows = build_windows(texts, size=3, step=1)
        probs = score_windows(windows)
        if probs is None:
            return {"sentence_scores": [], "available": False,
                    "model_version": MODEL_VERSION, "caveat": "detector unavailable"}
        agg = aggregate(texts, windows, probs, size=3, step=1)
        out = []
        for i, sent in enumerate(sentences):
            raw = agg["sentence_scores"][i]  # may be None for uncovered
            # Skip the heatmap on stubs: a short sentence (e.g. a 3-word bridge) scores as
            # noise in isolation, and downstream contracts expect plain (signal-less) segments
            # for unscored spans. Floor matches the predictability scanner's >=8-word convention.
            too_short = len(texts[i].split()) < 8
            if raw is None or too_short:
                score = None
            else:
                score = round(float(raw), 3)
            out.append({
                "sentence_id": sent.get("sentence_id"),
                "paragraph_id": sent.get("paragraph_id"),
                "score": score,
                "band": band_for_sentence(score) if score is not None else "clean",
            })
        return {"sentence_scores": out, "available": True, "model_version": MODEL_VERSION}
    except Exception as e:  # noqa: BLE001 — fail-open, never break the report
        logger.warning("[deberta] compose_from_sentences failed (fail-open): %s", e)
        return None


def headline_from_heatmap(heatmap: dict | None, sentences: list[dict]) -> dict | None:
    """Build the ai_signal_deberta HEADLINE dict from the canonical-sentence heatmap, so the
    tile and the map share ONE sentence segmentation (no naive-splitter mismatch).

    `heatmap` is the output of compose_from_sentences(); `sentences` is the same canonical
    sentence list passed to it (needed to recover the text for flagged_passages). Returns the
    full headline dict via the shared _build_headline helper, or None if the heatmap is
    unavailable (caller falls back to compose(text))."""
    if not heatmap or not heatmap.get("available"):
        return None
    rows = heatmap.get("sentence_scores") or []
    by_sid = {r.get("sentence_id"): r for r in rows}
    scored_tuples = []
    for sent in sentences:
        sid = sent.get("sentence_id")
        row = by_sid.get(sid)
        if not row:
            continue
        score = row.get("score")
        if score is not None:  # skip clean/None (short stubs, uncovered)
            scored_tuples.append((sid, score, sent.get("text", "")))
    if not scored_tuples:
        return None
    return _build_headline(scored_tuples)

