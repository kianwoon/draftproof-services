"""Exact character spans for top-k predictability highlighting.

Produces precise (start_char, end_char) ranges over a document so the UI can shade
actionable sentences that score HIGH for top-k predictability and underline actionable
runs around predictable (GPT-2 top-10) tokens inside them. Raw spans are retained
separately for audit. Raw spans are exact tokenizer offsets; actionable spans are derived
from those offsets and expanded only to readable word boundaries for editing.

Why exact offsets: many grammatical tokens are naturally top-10, so predictable tokens are
pervasive. Fuzzy string-matching short tokens would be imprecise, so this module maps each
token to its real character position and then separates raw evidence from editable spans.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# A predictable "word run" is >= this many consecutive top-10 tokens. Single isolated
# function words are not marked; only sustained predictable stretches.
_MIN_RUN_TOKENS = 2
_MAX_SENTENCE_SPANS = 400
_MAX_WORD_SPANS = 800
_DEFAULT_ACTIONABLE_MIN_LEXICAL_CHARS = 6
_DEFAULT_ACTIONABLE_MIN_VISIBLE_CHARS = 12
_DEFAULT_ACTIONABLE_MIN_TERMS = 2
_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim surrounding whitespace so a span underlines words, not leading spaces."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def compute_predictability_highlights(
    text: str,
    *,
    scanner: Optional[Any] = None,
    max_tokens: int = 384,
) -> dict[str, list[list[int]]]:
    """Return exact char spans for top-k highlighting of ``text``.

    {
        "sentences":             [[start, end], ...],  # raw HIGH top-k sentences
        "actionable_sentences":  [[start, end], ...],  # HIGH sentences with editable runs
        "words":                 [[start, end], ...],  # actionable predictable runs
        "actionable_words":      [[start, end], ...],  # explicit alias for words
        "raw_words":             [[start, end], ...],  # all runs of >=2 top-10 tokens
    }

    Spans index ``text`` directly: ``text[start:end]`` is the highlighted slice. Never
    raises; returns empty lists on any failure so it can never break the rewrite pipeline.
    """
    empty = {"sentences": [], "actionable_sentences": [], "words": [], "actionable_words": [], "raw_words": []}
    body = str(text or "")
    if not body.strip():
        return empty
    try:
        from poc.detect.document_structure import structured_sentence_segments

        if scanner is None:
            from poc.predictability.scanner import PredictabilityScanner

            scanner = PredictabilityScanner(model_name="gpt2")

        sentences: list[list[int]] = []
        actionable_sentences: list[list[int]] = []
        raw_words: list[list[int]] = []
        actionable_words: list[list[int]] = []

        for row in structured_sentence_segments(body):
            sentence = str(row.get("sentence") or "")
            base = row.get("start_char")
            end = row.get("end_char")
            if not sentence.strip() or not isinstance(base, int) or not isinstance(end, int):
                continue
            # Trust the structural offset only if it actually matches the slice.
            if body[base:end] != sentence:
                located = body.find(sentence)
                if located < 0:
                    continue
                base = located

            result = scanner.scan_sentence(sentence)
            token_results = getattr(result, "token_results", None) or []
            if not token_results:
                continue

            high_sentence = str(getattr(result, "risk_label", "")).lower() == "high"
            sentence_span = [base, base + len(sentence)]
            if high_sentence:
                sentences.append(sentence_span)

            encoded = scanner.tokenizer(
                sentence,
                truncation=True,
                max_length=max_tokens,
                return_offsets_mapping=True,
            )
            offsets = encoded.get("offset_mapping") or []
            # scan_sentence scores input_ids[1:], so token_results[i] aligns to offsets[i+1].
            if len(offsets) < len(token_results) + 1:
                continue

            sentence_raw_words: list[list[int]] = []
            sentence_actionable_words: list[list[int]] = []
            run: list[int] = []
            for i, tok in enumerate(token_results):
                if getattr(tok, "top_10", False):
                    run.append(i)
                    continue
                _flush_run(body, base, run, offsets, sentence_raw_words, sentence_actionable_words)
                run = []
            _flush_run(body, base, run, offsets, sentence_raw_words, sentence_actionable_words)
            raw_words.extend(sentence_raw_words)
            if topk_grounding_gate_enabled():
                sentence_actionable_words = _apply_grounding_gate(body, sentence, sentence_actionable_words)
            actionable_words.extend(sentence_actionable_words)
            if high_sentence and sentence_actionable_words:
                actionable_sentences.append(sentence_span)

            if len(sentences) >= _MAX_SENTENCE_SPANS or len(raw_words) >= _MAX_WORD_SPANS:
                break

        return {
            "sentences": sentences[:_MAX_SENTENCE_SPANS],
            "actionable_sentences": actionable_sentences[:_MAX_SENTENCE_SPANS],
            "words": actionable_words[:_MAX_WORD_SPANS],
            "actionable_words": actionable_words[:_MAX_WORD_SPANS],
            "raw_words": raw_words[:_MAX_WORD_SPANS],
        }
    except Exception:
        logger.warning("Predictability highlight computation failed", exc_info=True)
        return empty


def _flush_run(
    body: str,
    base: int,
    run: list[int],
    offsets: list[Any],
    raw_words: list[list[int]],
    actionable_words: list[list[int]],
) -> None:
    if len(run) < _MIN_RUN_TOKENS:
        return
    first, last = run[0], run[-1]
    char_start = base + int(offsets[first + 1][0])
    char_end = base + int(offsets[last + 1][1])
    char_start, char_end = _trim(body, char_start, char_end)
    if char_end > char_start:
        raw_words.append([char_start, char_end])
        action_start, action_end = _expand_to_word_boundaries(body, char_start, char_end)
        if action_end > action_start and _is_actionable_predictable_run(body[action_start:action_end]):
            actionable_words.append([action_start, action_end])


def _expand_to_word_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand a tokenizer fragment span to whole readable words for repair targeting."""
    body = str(text or "")
    n = len(body)
    start = max(0, min(int(start), n))
    end = max(start, min(int(end), n))
    while start > 0 and _is_word_body_char(body[start - 1]):
        start -= 1
    while end < n and _is_word_body_char(body[end]):
        end += 1
    return _trim_edge_punctuation(body, *_trim(body, start, end))


def _is_word_body_char(ch: str) -> bool:
    return ch.isalnum() or ch in {"'", "’", "-", "‑", "–"}


def _trim_edge_punctuation(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and not text[start].isalnum():
        start += 1
    while end > start and not text[end - 1].isalnum():
        end -= 1
    return start, end


# --- grounding gate (EXPERIMENT, off by default) -------------------------------------------------
# Differentiates a REAL top-k problem from a false alarm. A high-top-k run is a false alarm when it
# is a concrete anchor or sits inside a grounded sentence (proven n=9: ~73-89% of flagged spans).
# This gate drops those from the ACTIONABLE/displayed set so the report stops red-flagging good
# content; the genuinely-formulaic (un-grounded, non-anchor) runs survive -- the real problem.
# Display-only: raw_words / raw sentences are never touched (audit trail preserved).
# Opt in with DRAFTPROOF_TOPK_GROUNDING_GATE=1. Content-agnostic: it reuses the scanner's structural
# concreteness check + pure text shape (digits, numerals, hyphen compounds) -- no phrase/word lists.
_DIGIT_RE = re.compile(r"\d")
_COMPOUND_RE = re.compile(r"[A-Za-z0-9]+[‐‑–—-][A-Za-z0-9]+")
# Numerals are a CLOSED LINGUISTIC CLASS (like digits), not a content/answer list. Patched here
# because the shared concreteness detector counts digits but misses spelled-out quantities.
_NUMWORD_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|dozen|half|third|quarter|quarters|thirds|first|second|fourth|fifth)\b",
    re.IGNORECASE,
)


def topk_grounding_gate_enabled() -> bool:
    return os.environ.get("DRAFTPROOF_TOPK_GROUNDING_GATE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _has_numeral(text: str) -> bool:
    return bool(_DIGIT_RE.search(str(text or "")) or _NUMWORD_RE.search(str(text or "")))


def _structurally_concrete(text: str) -> bool:
    """The scanner's content-agnostic concreteness check (numbers, named entities, quotes, first-person
    lived detail, temporal anchors). Fail-open to False so a missing import never hides a real run."""
    try:
        from poc.detect.layer3_scoring import _sentence_has_concrete_or_context
    except Exception:
        try:
            from detect.layer3_scoring import _sentence_has_concrete_or_context
        except Exception:
            return False
    try:
        return bool(_sentence_has_concrete_or_context(str(text or "")))
    except Exception:
        return False


def _run_is_anchor(run_text: str) -> bool:
    """A run that itself carries concrete content (number, hyphen compound, or named/lived detail)."""
    t = str(run_text or "")
    return _has_numeral(t) or bool(_COMPOUND_RE.search(t)) or _structurally_concrete(t)


def _sentence_is_grounded(sentence: str) -> bool:
    s = str(sentence or "")
    return _has_numeral(s) or _structurally_concrete(s)


def _apply_grounding_gate(body: str, sentence: str, spans: list[list[int]]) -> list[list[int]]:
    """Drop the false-alarm runs. If the whole sentence is grounded (anchor-context), none of its
    predictable runs is actionable; otherwise drop only the runs that are themselves anchors."""
    if _sentence_is_grounded(sentence):
        return []
    return [sp for sp in spans if not _run_is_anchor(body[sp[0]:sp[1]])]


def _is_actionable_predictable_run(value: str) -> bool:
    """Return True when a raw top-10 run has enough lexical mass to edit safely.

    GPT-2 top-10 runs often contain only grammatical scaffolding. Those runs are real
    measurements, but they are poor rewrite targets: changing them usually damages grammar
    more than it moves document-level top-k. This gate uses only text shape, not phrase
    lists, so it stays content-agnostic.
    """
    text = str(value or "").strip()
    if not text:
        return False
    terms = _TERM_RE.findall(text)
    if len(terms) < _actionable_min_terms():
        return False
    visible = sum(1 for ch in text if not ch.isspace())
    if visible < _actionable_min_visible_chars():
        return False
    lexical_terms = [term for term in terms if _has_lexical_mass(term)]
    if len(lexical_terms) >= 2:
        return True
    if len(lexical_terms) == 1 and len(re.sub(r"[^A-Za-z0-9]", "", lexical_terms[0])) >= 9 and len(terms) >= 3:
        return True
    return False


def _has_lexical_mass(term: str) -> bool:
    core = re.sub(r"[^A-Za-z0-9]", "", str(term or ""))
    if not core:
        return False
    if any(ch.isdigit() for ch in core):
        return True
    return sum(1 for ch in core if ch.isalpha()) >= _actionable_min_lexical_chars()


def _actionable_min_lexical_chars() -> int:
    return _positive_int_env(
        "DRAFTPROOF_TOPK_HIGHLIGHT_ACTIONABLE_MIN_LEXICAL_CHARS",
        _DEFAULT_ACTIONABLE_MIN_LEXICAL_CHARS,
    )


def _actionable_min_visible_chars() -> int:
    return _positive_int_env(
        "DRAFTPROOF_TOPK_HIGHLIGHT_ACTIONABLE_MIN_VISIBLE_CHARS",
        _DEFAULT_ACTIONABLE_MIN_VISIBLE_CHARS,
    )


def _actionable_min_terms() -> int:
    return _positive_int_env(
        "DRAFTPROOF_TOPK_HIGHLIGHT_ACTIONABLE_MIN_TERMS",
        _DEFAULT_ACTIONABLE_MIN_TERMS,
    )


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return default
