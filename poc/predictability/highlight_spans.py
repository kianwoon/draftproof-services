"""Exact character spans for top-k predictability highlighting.

Produces precise (start_char, end_char) ranges over a document so the UI can shade the
sentences that score HIGH for top-k predictability and underline the runs of predictable
(GPT-2 top-10) tokens inside them. Every span is an EXACT char range derived from the
tokenizer's offset mapping -- never a string match -- so what is highlighted is exactly
what the scanner measured.

Why exact offsets: ~70% of grammatical English is function words that are always top-10,
so predictable tokens are pervasive. Highlighting them honestly shows the intrinsic AI-text
signal (a rewrite does not remove it); fuzzy string-matching short tokens like "the"/"of"
would be imprecise, so this module maps each token to its real character position instead.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# A predictable "word run" is >= this many consecutive top-10 tokens. Single isolated
# function words are not marked; only sustained predictable stretches.
_MIN_RUN_TOKENS = 2
_MAX_SENTENCE_SPANS = 400
_MAX_WORD_SPANS = 800


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
        "sentences": [[start, end], ...],  # sentences whose risk_label == "high"
        "words":     [[start, end], ...],  # runs of >=2 consecutive top-10 tokens
    }

    Spans index ``text`` directly: ``text[start:end]`` is the highlighted slice. Never
    raises; returns empty lists on any failure so it can never break the rewrite pipeline.
    """
    empty = {"sentences": [], "words": []}
    body = str(text or "")
    if not body.strip():
        return empty
    try:
        from poc.detect.document_structure import structured_sentence_segments

        if scanner is None:
            from poc.predictability.scanner import PredictabilityScanner

            scanner = PredictabilityScanner(model_name="gpt2")

        sentences: list[list[int]] = []
        words: list[list[int]] = []

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

            if str(getattr(result, "risk_label", "")).lower() == "high":
                sentences.append([base, base + len(sentence)])

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

            run: list[int] = []
            for i, tok in enumerate(token_results):
                if getattr(tok, "top_10", False):
                    run.append(i)
                    continue
                _flush_run(body, base, run, offsets, words)
                run = []
            _flush_run(body, base, run, offsets, words)

            if len(sentences) >= _MAX_SENTENCE_SPANS or len(words) >= _MAX_WORD_SPANS:
                break

        return {
            "sentences": sentences[:_MAX_SENTENCE_SPANS],
            "words": words[:_MAX_WORD_SPANS],
        }
    except Exception:
        logger.warning("Predictability highlight computation failed", exc_info=True)
        return empty


def _flush_run(
    body: str,
    base: int,
    run: list[int],
    offsets: list[Any],
    words: list[list[int]],
) -> None:
    if len(run) < _MIN_RUN_TOKENS:
        return
    first, last = run[0], run[-1]
    char_start = base + int(offsets[first + 1][0])
    char_end = base + int(offsets[last + 1][1])
    char_start, char_end = _trim(body, char_start, char_end)
    if char_end > char_start:
        words.append([char_start, char_end])
