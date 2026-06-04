"""Honest register/polish coaching for the shown rewrite.

This is COACHING, not a score lever. AiDetector and similar strict detectors flag POLISH / FLUENCY
(even cadence, balanced structure, polished phrasing), not grounding -- so register coaching CANNOT and
DOES NOT lower the external score. Its value: it points the user at the spots that read most machine-
polished so they can rewrite them in their own plainer voice (the durable, honest fix the user owns).

Design constraints (verified, not assumed):
- Reuse EXISTING content-agnostic signals (register_score, estimate_burstiness_risk) -- no new lexicons.
- The per-sentence axis is `register_score` ONLY (the viability gate showed `balanced`/`formulaic`
  fire ~0 per-sentence and burstiness is doc-level). Abstraction is already coached by
  `_ungrounded_claims`/bracket-amber, so we DEDUP against it: register coaching only flags GROUNDED
  sentences (those carrying a hard concrete) that still read polished -- a genuinely distinct surface.
- Fail-open: any import/scoring error returns None (no coaching, never a crash).
"""

from __future__ import annotations

import os
from typing import Any


def register_coaching_enabled() -> bool:
    """Kill switch. Default ON; set DRAFTPROOF_V6_REGISTER_COACHING=0 to disable."""
    raw = os.environ.get("DRAFTPROOF_V6_REGISTER_COACHING", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


# Honest framing shown with every register-coaching surface. NOT a promise of a lower score.
# allow-hardcode: this is a single human-facing COACHING COPY string shown in the report (PDF + page),
# not a detect/scoring/matching word-list. Sentence selection is done by the content-agnostic
# register_score signal below; this string only frames the coaching message. It names the recurring
# machine-smooth/complete pattern holistically (polish, seamless anecdotes, flawless example lists)
# rather than a single facet -- those are the spots to put back into the author's own messier voice.
COACHING_NOTE = (
    "Strict AI detectors flag writing that reads machine-smooth and machine-complete -- polished "
    "phrasing, seamless personal anecdotes, and flawless lists of examples -- not grounding. Rewriting "
    "these spots in your own messier, more selective voice will NOT lower the detector score, but it "
    "makes the draft genuinely yours: keep the one real example you would actually mention, say it "
    "plainly, and cut the seamless completeness."
)


def _load_signals():
    """Return (split_sentences, register_score, estimate_burstiness_risk, _sentence_has_hard_concrete)
    via the codebase's dual-import convention, or None on failure (fail-open)."""
    try:
        from detect.layer3_scoring import (
            split_sentences,
            register_score,
            estimate_burstiness_risk,
            _sentence_has_hard_concrete,
        )
    except Exception:
        try:
            from poc.detect.layer3_scoring import (  # type: ignore
                split_sentences,
                register_score,
                estimate_burstiness_risk,
                _sentence_has_hard_concrete,
            )
        except Exception:
            return None
    return split_sentences, register_score, estimate_burstiness_risk, _sentence_has_hard_concrete


def build_register_coaching(
    text: str,
    *,
    limit: int = 4,
    min_words: int = 8,
    rhythm_threshold: float = 0.6,
) -> dict[str, Any] | None:
    """Build the honest register/polish coaching payload for the FINAL rewritten text.

    Returns None when coaching cannot be produced precisely (too few sentences, no grounded sentence to
    coach, or signals unavailable) -- abstaining is preferred over a vague/blurred surface.

    Shape:
      {
        "offenders": [{"text": str, "register_score": float}, ...],   # grounded-but-polished, top-N
        "worked_contrast": {"polished": {...}, "plain": {...}},        # user's own high vs low example
        "rhythm_even": bool, "rhythm_score": float,                    # doc-level (NOT per-sentence)
        "note": COACHING_NOTE,
      }
    """
    if not isinstance(text, str) or not text.strip():
        return None
    signals = _load_signals()
    if signals is None:
        return None
    split_sentences, register_score, estimate_burstiness_risk, _sentence_has_hard_concrete = signals

    try:
        sentences = [s.strip() for s in split_sentences(text) if len(s.split()) >= min_words]
    except Exception:
        return None
    if len(sentences) < 4:
        return None  # too short to rank meaningfully

    try:
        all_scored = [(float(register_score(s)), s) for s in sentences]
        # DEDUP vs grounding coaching: only GROUNDED sentences (hard concrete) are register candidates,
        # so this surface is additive to _ungrounded_claims, not a restatement of "add specifics".
        grounded = [(r, s) for (r, s) in all_scored if _sentence_has_hard_concrete(s)]
    except Exception:
        return None
    if not grounded:
        # Every polished sentence is also ungrounded -> already coached by the grounding surface.
        # Abstain from a redundant register list; still offer the doc-level contrast/rhythm below.
        grounded = []

    offenders: list[dict[str, Any]] = []
    if grounded:
        mean_reg = sum(r for r, _ in grounded) / len(grounded)
        # Relative threshold (no magic absolute): the sentences more polished than the author's OWN mean.
        flagged = sorted([(r, s) for (r, s) in grounded if r > mean_reg], key=lambda x: -x[0])[:limit]
        offenders = [{"text": s, "register_score": round(r, 3)} for r, s in flagged]

    # Worked contrast: the user's own most-polished vs plainest sentence (teaches "plainer reads less
    # machine-smooth" from their own text). Uses all scorable sentences.
    all_scored_sorted = sorted(all_scored, key=lambda x: -x[0])
    high_r, high_s = all_scored_sorted[0]
    low_r, low_s = all_scored_sorted[-1]
    worked_contrast = None
    if high_s != low_s and (high_r - low_r) > 0:
        worked_contrast = {
            "polished": {"text": high_s, "register_score": round(high_r, 3)},
            "plain": {"text": low_s, "register_score": round(low_r, 3)},
        }

    try:
        rhythm_score = float(estimate_burstiness_risk(text))
    except Exception:
        rhythm_score = 0.0

    # Abstain entirely if there is nothing precise to show.
    if not offenders and worked_contrast is None and rhythm_score < rhythm_threshold:
        return None

    return {
        "offenders": offenders,
        "worked_contrast": worked_contrast,
        "rhythm_even": bool(rhythm_score >= rhythm_threshold),
        "rhythm_score": round(rhythm_score, 3),
        "note": COACHING_NOTE,
    }
