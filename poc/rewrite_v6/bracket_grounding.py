"""Bracket-grounding -- the last-stage treatment for generic (un-grounded) sentences.

For each generic sentence the showcase selector finds, ask the model (qwen) to make it more concrete
WITHOUT inventing facts:
  - model returns an improvement -> single bracket  [suggested replacement]   (writer verifies/edits)
  - model returns nothing        -> double bracket  [[original sentence]]      (writer grounds it)

The brackets are the point: the shipped draft is a SHOWN solution, and the brackets tell the writer
exactly which spans are machine suggestions to verify or replace with their own real detail. No code
quality gate -- the model decides; the bracket flags it for the human. Content-agnostic selection
(reuses the scanner's structural-concreteness check via _generic_candidates). MUTATES rewritten_text.
Never raises; on disable / no candidates / any failure the text is returned unchanged.

allow-hardcode: the `rules` below are the model coaching PROMPT (human-reviewed guidance), not a
detect/allow/scoring word-list in code logic.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)


def bracket_grounding_enabled() -> bool:
    """Feature flag. Code default OFF; production enables it via the worker entrypoint. Set
    DRAFTPROOF_V6_BRACKET_GROUNDING=1 to enable, =0 to disable."""
    return os.environ.get("DRAFTPROOF_V6_BRACKET_GROUNDING", "0").strip().lower() in {"1", "true", "yes", "on"}


def _max_sentences() -> int:
    raw = os.environ.get("DRAFTPROOF_V6_BRACKET_MAX_SENTENCES", "").strip()
    try:
        v = int(raw)
        if v > 0:
            return v
    except (TypeError, ValueError):
        pass
    return 8


def _max_tokens() -> int:
    raw = os.environ.get("DRAFTPROOF_V6_BRACKET_MAX_TOKENS", "").strip()
    try:
        v = int(raw)
        if v > 0:
            return v
    except (TypeError, ValueError):
        pass
    return 8000


_SYSTEM = "You are a writing coach. Return only valid JSON."


def _build_prompt(candidates: list[str]) -> str:
    payload = {
        "task": "improve_if_possible",
        "rules": [
            "Make each sentence more concrete and specific WITHOUT inventing any facts (no fake numbers, names, companies, statistics, or events).",
            "If you can honestly improve it, return the improved sentence in 'improved'. If you cannot, return 'improved' as an empty string.",
        ],
        "sentences": [{"i": i, "text": s} for i, s in enumerate(candidates)],
        "output_schema": {"results": [{"i": 0, "improved": "improved sentence or empty string"}]},
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False)


def apply_bracket_grounding(
    text: str,
    *,
    gateway: Any,
    cancellation_check: Callable[[], None] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Ground the generic sentences. Returns (clean_text, spans) -- NO literal brackets in the text.
    spans = [{"start","end","kind"}] over clean_text, kind 'improved' (qwen generated a better version,
    rendered GREEN) or 'kept' (qwen could not improve, original kept, rendered AMBER).
    (text, []) on disable / no candidates / failure."""
    original = str(text or "")
    if not bracket_grounding_enabled() or not original.strip():
        return original, []
    if cancellation_check:
        cancellation_check()
    try:
        from .predictability_showcase import _generic_candidates
        from .json_io import parse_json

        candidates = _generic_candidates(original, _max_sentences())
        if not candidates:
            return original, []
        # Ask qwen to improve each generic sentence. If the call fails (timeout / network / parse),
        # FALL BACK to empty results -> every candidate becomes a 'kept' (amber) span, so the writer
        # still sees the generic sentences flagged instead of the feature silently producing nothing.
        results = {}
        try:
            response = gateway.chat(
                _build_prompt(candidates),
                system=_SYSTEM,
                temperature=0.4,
                top_p=0.9,
                max_tokens=_max_tokens(),
                response_format={"type": "json_object"},
                app_label="BracketGrounding",
            )
            raw = getattr(response, "raw_content", "") or getattr(response, "content", "") or ""
            data = parse_json(raw)
            results = {r["i"]: (r.get("improved") or "").strip()
                       for r in (data.get("results") or []) if isinstance(r, dict) and "i" in r} if isinstance(data, dict) else {}
        except Exception:
            logger.warning("bracket_grounding qwen call failed; falling back to amber-kept spans", exc_info=True)
            results = {}

        # locate each candidate; choose its clean replacement + colour kind
        located = []
        for i, sentence in enumerate(candidates):
            idx = original.find(sentence)
            if idx < 0:
                continue  # not locatable verbatim (e.g. spans a line break) -> leave untouched
            improved = results.get(i, "")
            if improved and improved != sentence:
                located.append((idx, idx + len(sentence), improved, "improved"))
            else:
                located.append((idx, idx + len(sentence), sentence, "kept"))
        located.sort()

        # rebuild CLEAN text, tracking each replacement's offset span in the new text
        out: list[str] = []
        spans: list[dict[str, Any]] = []
        cursor = 0
        pos = 0
        for start_o, end_o, replacement, kind in located:
            if start_o < cursor:
                continue  # overlapping / duplicate match -> skip
            gap = original[cursor:start_o]
            out.append(gap); pos += len(gap)
            span_start = pos
            out.append(replacement); pos += len(replacement)
            spans.append({"start": span_start, "end": pos, "kind": kind})
            cursor = end_o
        out.append(original[cursor:])
        new_text = "".join(out)
        logger.info("bracket_grounding: candidates=%d spans=%d", len(candidates), len(spans))
        return new_text, spans
    except Exception:
        logger.warning("bracket_grounding failed", exc_info=True)
        return original, []
