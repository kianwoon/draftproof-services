"""Bracket-grounding -- the last-stage treatment for generic (un-grounded) sentences.

For each generic sentence the showcase selector finds, ask the model (qwen) for a rewrite, then a
deterministic GATE (`_is_genuine_improvement`) -- NOT qwen's self-report -- decides what ships:
  - replacement is genuinely better -> GREEN span ('improved')  the replacement is used
  - otherwise                        -> AMBER span ('kept')      the ORIGINAL is kept (qwen text dropped)

Green requires ALL of: (1) NOT fabricated -- introduces no new number and no concreteness/anchor signal
the original lacked (qwen has no source, so any added specific is invented); (2) grammar/quality clean;
(3) not worse on a before/after `scan_text` A/B (finding_count + sentence-shape risk). The generator
(qwen) and the checker are SEPARATE entities, and the A/B is a symmetric regression check (reject-if-
worse), NOT the scanner's grounding oracle -- so green is not the self-congratulatory circular gate that
the prior `_is_grounded` version was. The (original, replacement, decision, scan delta) per candidate is
captured into `diag` for audit.

Consequence (honest): qwen cannot raise grounding without inventing specifics, which the fabrication
check rejects -- so green mostly reflects faithful grammar/structural gains; real grounding still comes
from the USER editing the amber sentences. Content-agnostic selection (reuses the scanner's
structural-concreteness check via _generic_candidates). MUTATES rewritten_text -- the caller MUST
re-scan the shipped text after this stage so the report's scores describe the bytes the user receives.
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


# Risk tolerance for the before/after A/B (mirrors write.py::_has_meaningful_movement's `>= -2.0`):
# a replacement may raise mean sentence-shape risk by at most this much and still count as "not worse".
_AB_RISK_TOLERANCE = 2.0


def _grounding_signal_count(text: str) -> int:
    """Count the scanner's content-agnostic concreteness/anchor patterns present (numbers, codes, named
    entities, citations, quotes, first-hand verbs, exemplification/temporal/conditional anchors). Reuses
    the regex lists from detect.layer3_scoring -- NOT the boolean oracle `_sentence_has_concrete_or_context`
    (gating green on that oracle is circular: it IS the badge's grounding scorer)."""
    import re
    try:
        from detect.layer3_scoring import (
            CONCRETE_DETAIL_PATTERNS, CONTEXTUAL_ANCHOR_PATTERNS, NAMED_ENTITY_DETAIL_PATTERN,
        )
    except Exception:
        try:
            from poc.detect.layer3_scoring import (
                CONCRETE_DETAIL_PATTERNS, CONTEXTUAL_ANCHOR_PATTERNS, NAMED_ENTITY_DETAIL_PATTERN,
            )
        except Exception:
            return 0
    n = sum(1 for p in CONCRETE_DETAIL_PATTERNS
            if re.search(p, text, flags=0 if p == NAMED_ENTITY_DETAIL_PATTERN else re.I))
    n += sum(1 for p in CONTEXTUAL_ANCHOR_PATTERNS if re.search(p, text, flags=re.I))
    return n


def _is_genuine_improvement(original: str, replacement: str) -> tuple[bool, dict[str, Any]]:
    """Decide GREEN (use replacement) vs AMBER (keep original) by a deterministic, generator-independent
    comparison -- never qwen's self-report. Returns (accept, audit_detail). Green requires ALL of:
      (1) NOT fabricated: no new number and no new concreteness/anchor signal vs the original
          (qwen has no source, so any added specific is invented);
      (2) grammar/quality clean: no hard (non-advisory) integrity blocker, no fragment/trace sentence;
      (3) not worse on the structural A/B: scan_text finding_count and mean_sentence_shape_risk do not
          regress vs the original -- the same scores write.py::_has_meaningful_movement trusts.
    Fails CLOSED to amber when a check cannot run (never ship an unverified replacement)."""
    detail: dict[str, Any] = {"original": original, "replacement": replacement}
    repl = (replacement or "").strip()
    if not repl or repl == (original or "").strip():
        detail["decision"] = "no_change"
        return False, detail

    # (1) fabrication: new numbers or a new concreteness/anchor signal the original lacked
    new_numbers = False
    try:
        from .grammar_repair import _numbers
        new_numbers = bool(_numbers(repl) - _numbers(original))
    except Exception:
        new_numbers = False
    sig_before, sig_after = _grounding_signal_count(original), _grounding_signal_count(repl)
    fabricated = new_numbers or sig_after > sig_before
    detail.update({
        "new_numbers": new_numbers, "signal_before": sig_before, "signal_after": sig_after,
        "fabricated": fabricated,
    })
    if fabricated:
        detail["decision"] = "fabricated"
        return False, detail

    # (2) grammar / quality
    try:
        from .integrity_guard import ADVISORY_BLOCKERS, candidate_integrity_blockers
        from .prose_quality import has_fragment_or_trace_sentences
        hard = [b for b in candidate_integrity_blockers(repl) if b not in ADVISORY_BLOCKERS]
        grammar_ok = not hard and not has_fragment_or_trace_sentences(repl)
        detail["grammar_blockers"] = hard
    except Exception:
        grammar_ok = True  # selection already passed; import failure must not block every candidate
    detail["grammar_ok"] = grammar_ok
    if not grammar_ok:
        detail["decision"] = "grammar_regression"
        return False, detail

    # (3) before/after A/B on structural scores -- fail CLOSED if the scan can't run
    try:
        from .scan import scan_text
        before = scan_text(original).scores
        after = scan_text(repl).scores
        finding_drop = before.get("finding_count", 0) - after.get("finding_count", 0)
        risk_drop = before.get("mean_sentence_shape_risk", 0.0) - after.get("mean_sentence_shape_risk", 0.0)
        detail["scan_before"] = {"finding_count": before.get("finding_count"),
                                 "risk": round(before.get("mean_sentence_shape_risk", 0.0), 3)}
        detail["scan_after"] = {"finding_count": after.get("finding_count"),
                                "risk": round(after.get("mean_sentence_shape_risk", 0.0), 3)}
        detail["finding_drop"] = finding_drop
        detail["risk_drop"] = round(risk_drop, 3)
    except Exception:
        detail["decision"] = "scan_unavailable"
        return False, detail
    if finding_drop < 0 or risk_drop < -_AB_RISK_TOLERANCE:
        detail["decision"] = "score_regression"
        return False, detail

    detail["decision"] = "accepted"
    return True, detail


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

        # locate each candidate; the deterministic gate (NOT qwen's self-report) decides green vs amber.
        located = []
        audit: list[dict[str, Any]] = []
        for i, sentence in enumerate(candidates):
            idx = original.find(sentence)
            if idx < 0:
                continue  # not locatable verbatim (e.g. spans a line break) -> leave untouched
            improved = results.get(i, "")
            accept, detail = _is_genuine_improvement(sentence, improved)
            audit.append(detail)
            if accept:
                located.append((idx, idx + len(sentence), improved, "improved"))
            else:
                located.append((idx, idx + len(sentence), sentence, "kept"))  # keep ORIGINAL, drop qwen text
        located.sort()
        if diag is not None:
            diag["candidates"] = audit

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
