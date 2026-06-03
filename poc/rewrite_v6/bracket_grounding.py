"""Bracket-grounding -- the last-stage treatment for generic (un-grounded) sentences.

For each generic sentence the showcase selector finds, ask the model (qwen) to GROUND it -- anchor the
same claim in a concrete illustrative detail (a specific moment, number, name, or first-hand detail)
that the author will later replace with their own real specifics:
  - model grounds it      -> GREEN span ('improved')  the grounded version replaces the original
  - model cannot ground it -> AMBER span ('kept')      the original sentence is kept, flagged to ground

The shipped draft is a SHOWN solution: green = a machine-grounded example to verify/replace, amber =
"you still need to ground this in your own words". Mere rewording / synonym-swap / linking words are
NOT grounding. The model PROPOSES; a deterministic GATE decides: a replacement is accepted as green
ONLY if it passes the scanner's structural-concreteness oracle (_is_grounded) that the original
failed -- qwen's self-report is NOT trusted (it never returns empty; it appends filler and self-rates
"improved"). Replacements the gate can't confirm grounded fall to amber (kept). Content-agnostic selection (reuses the
scanner's structural-concreteness check via _generic_candidates). MUTATES rewritten_text -- the caller
MUST re-scan the shipped text after this stage so the report's scores describe the bytes the user
receives. Never raises; on disable / no candidates / any failure the text is returned unchanged.

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


# allow-hardcode: _SYSTEM and the `rules` below are the model coaching PROMPT (human-reviewed
# guidance), not a detect/allow/scoring word-list in code logic. The few illustrative connective
# words are examples shown to the model, never matched against the input.
_SYSTEM = (
    "You are a writing coach. You receive sentences that state a claim WITHOUT concrete grounding. "
    "Rewrite a sentence ONLY when you can anchor the SAME claim in a concrete, illustrative detail -- a "
    "specific moment, number, name, place, or first-hand observation -- that the author will later "
    "replace with their own real specifics. Keep the claim's meaning and stance. If the only change "
    "you could make is rewording, reordering, or adding linking words, do not rewrite it. Return only "
    "valid JSON."
)


def _build_prompt(candidates: list[str]) -> str:
    # allow-hardcode: the `rules` strings are model coaching guidance (a prompt), not code logic.
    payload = {
        "task": "ground_if_possible",
        "rules": [
            # Aligned with the objective: grounding (adding a concrete illustrative anchor) IS the fix;
            # synonym-swapping / reordering / linking words stay generic and are NOT a fix.
            "Rewrite a sentence ONLY if you can add a CONCRETE illustrative anchor (a specific moment, "
            "number, name, place, or first-hand detail) that grounds the SAME claim. The anchor is an "
            "example the author will swap for their own real detail -- keep the claim's meaning and stance.",
            "Do NOT merely reword, swap synonyms, reorder clauses, or add transition/linking words "
            "(e.g. however, moreover, subsequently, next, this combination): that does not ground the "
            "claim. If you cannot add a concrete anchor, return 'improved' as an empty string -- the "
            "original sentence is then kept unchanged.",
            "Keep grammar clean and the sentence self-contained.",
        ],
        "sentences": [{"i": i, "text": s} for i, s in enumerate(candidates)],
        "output_schema": {"results": [{"i": 0, "improved": "grounded sentence, or empty string if you cannot ground it"}]},
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False)


def apply_bracket_grounding(
    text: str,
    *,
    gateway: Any,
    fallback_gateway: Any = None,
    cancellation_check: Callable[[], None] | None = None,
    diag: dict[str, Any] | None = None,
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
        from .predictability_showcase import _generic_candidates, _is_grounded
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
            # GATE over trust (do NOT believe qwen's self-report -- it never returns empty, it always
            # appends filler and self-rates "improved"). Accept the replacement as GREEN only if it
            # PASSES the scanner's own structural-concreteness check that the original FAILED (the
            # candidate was selected because _is_grounded(original) is False). Same signal the detector
            # scores grounding with -> "improved" is consistent with how the product measures grounding.
            # Anything the gate doesn't confirm grounded -> AMBER ("kept"), original preserved.
            if improved and improved != sentence and _is_grounded(improved) is True:
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
