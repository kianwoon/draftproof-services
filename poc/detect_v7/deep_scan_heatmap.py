"""Deep-scan (Modal) per-sentence heatmap — mirrors
``poc/detect/deberta_signal.py::compose_from_sentences`` output shape exactly, so the
Signal-highlights map/fix-first consumers (``poc/report/report.py::_document_segments``)
can swap their source without changing shape.

STRICTLY SEPARATE from ``pipeline_bridge.get_deep_scan_proportion``: that call powers the
V7 tier-authority fused score and its own calibration/floor math must not be touched. This
module makes its OWN Modal call over the same canonical sentences, for the highlights map
only. Two independent, intentional Modal calls (see module docstring in pipeline_bridge.py
for the "one paid call" note — that note is about get_deep_scan_proportion vs
run_v7_breakdown sharing ONE call; this is a genuinely different consumer, the highlights
map, which needs full per-sentence scores that get_deep_scan_proportion discards).

Fail-open: returns None on ANY failure (deep scan disabled, no sentences, Modal
unavailable/error, or a length mismatch between input sentences and returned chunk_scores).
The caller (report.py) falls back to the existing ai_signal_deberta fakespot heatmap.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from . import config, modal_client
from .pipeline_bridge import is_deep_scan_enabled

logger = logging.getLogger(__name__)


def _band_for_deep_scan_score(score: Optional[float], sent_threshold: float) -> str:
    """Two-band deep-scan heatmap: score >= sent_threshold (config-driven, no
    hardcode) -> "high"; else "clean". Unlike the fakespot heatmap's 3-band
    clean/moderate/high scale, the deep-scan model's calibration only exposes
    ONE meaningful cut point today (``sent_threshold`` from
    ``config.get_deep_scan_calibration()`` — the SAME threshold
    ``get_deep_scan_proportion`` uses to decide "flagged", so the highlight map
    and the fused-tier proportion never disagree on which sentences count),
    so there is no separate "moderate" band here.

    NOTE (2026-07-04): a peer-session message during this task proposed a
    separate, lower "highlight_sent_threshold" (0.99) so more sentences would
    be marked for review than the 0.999 proportion counts. That was rejected:
    weights.json's own calibration notes document that 0.99 gave a 68% FPR
    on human ESL essays at doc-level, and there is no sentence-level
    calibration run for 0.99 either — using it here would silently relabel
    ~40% of genuinely human sentences as "AI signal" with zero calibration
    backing it. This function stays on the one calibrated, gate-tested
    threshold (0.999) until a real calibration run justifies a second one."""
    if score is None:
        return "clean"
    return "high" if score >= sent_threshold else "clean"


def compose_deep_scan_heatmap(sentences: list[dict]) -> Optional[dict]:
    """Score pre-segmented canonical sentences via the Modal deep-scan endpoint and return
    a per-sentence heatmap in the SAME shape as
    ``poc.detect.deberta_signal.compose_from_sentences``:

    ``{"sentence_scores": [{sentence_id, paragraph_id, score, band}, ...], "available": bool}``

    Each input dict carries ``{sentence_id (sNNN), paragraph_id (pNNN), text}`` — the exact
    canonical sentence list ``poc/report/report.py``'s ``_source_segments(complete=True)``
    produces. Sentences with fewer than 8 words (matching
    ``compose_from_sentences``'s and the predictability scanner's floor) are skipped:
    score None / band "clean". Band assignment uses ``sent_threshold`` from
    ``config.get_deep_scan_calibration()`` — no hardcoded cutoff.

    Fail-open: returns ``None`` when deep scan is disabled
    (``pipeline_bridge.is_deep_scan_enabled()``), there are no sentences, the Modal call
    fails/returns malformed data, or the returned ``chunk_scores`` length does not match the
    number of sentences sent. Never raises.
    """
    if not is_deep_scan_enabled() or not sentences:
        return None
    try:
        texts = [str(s.get("text") or "").strip() for s in sentences]
        if not any(texts):
            return None

        # Only send qualifying (>=8-word) sentences to Modal — short stubs are excluded
        # up front (mirrors compose_from_sentences's too-short skip) rather than scored
        # and discarded, since the deep-scan endpoint has no internal windowing floor.
        qualifying_indices = [i for i, t in enumerate(texts) if len(t.split()) >= 8]
        if not qualifying_indices:
            return {
                "sentence_scores": [
                    {
                        "sentence_id": sent.get("sentence_id"),
                        "paragraph_id": sent.get("paragraph_id"),
                        "score": None,
                        "band": "clean",
                    }
                    for sent in sentences
                ],
                "available": True,
            }

        chunks = [texts[i] for i in qualifying_indices]
        modal_response = modal_client.call_deep_scan(chunks)
        chunk_scores = modal_response.get("chunk_scores") if isinstance(modal_response, dict) else None
        if not (
            isinstance(modal_response, dict)
            and modal_response.get("available") is True
            and isinstance(chunk_scores, list)
            and len(chunk_scores) == len(chunks)
            and all(isinstance(s, (int, float)) and not isinstance(s, bool) for s in chunk_scores)
        ):
            logger.info(
                "detect_v7.deep_scan_heatmap: Modal deep scan unavailable/failed/malformed; "
                "skipping deep-scan heatmap (fail-open)."
            )
            return None

        calibration = config.get_deep_scan_calibration()
        sent_threshold = calibration["sent_threshold"]

        score_by_index: dict[int, float] = {
            qualifying_indices[j]: float(chunk_scores[j]) for j in range(len(qualifying_indices))
        }

        out = []
        for i, sent in enumerate(sentences):
            raw = score_by_index.get(i)
            score = round(raw, 3) if raw is not None else None
            out.append(
                {
                    "sentence_id": sent.get("sentence_id"),
                    "paragraph_id": sent.get("paragraph_id"),
                    "score": score,
                    "band": _band_for_deep_scan_score(score, sent_threshold),
                }
            )
        return {"sentence_scores": out, "available": True}
    except Exception as e:  # noqa: BLE001 — fail-open, never break the report
        logger.warning("detect_v7.deep_scan_heatmap: compose_deep_scan_heatmap failed (fail-open): %s", e)
        return None
