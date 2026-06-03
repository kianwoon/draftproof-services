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
    """Bracket the generic sentences. Returns (new_text, applied). (text, []) on disable/no-op/failure."""
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

        current = original
        applied: list[dict[str, Any]] = []
        for i, sentence in enumerate(candidates):
            if sentence not in current:
                continue  # couldn't locate verbatim (e.g. spans a line break) -> leave untouched
            improved = results.get(i, "")
            if improved and improved != sentence:
                # qwen generated a better version -> DOUBLE brackets (heavily AI-generated, review it)
                replacement, kind = f"[[{improved}]]", "double"
            else:
                # qwen could not improve it -> keep the original in SINGLE brackets (lightly flagged)
                replacement, kind = f"[{sentence}]", "single"
            current = current.replace(sentence, replacement, 1)
            applied.append({"original": sentence, "replacement": replacement, "bracket": kind})
        logger.info("bracket_grounding: candidates=%d applied=%d", len(candidates), len(applied))
        return current, applied
    except Exception:
        logger.warning("bracket_grounding failed", exc_info=True)
        return original, []
