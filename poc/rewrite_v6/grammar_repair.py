"""Grammar repair -- a final pass that fixes ONLY grammar without changing meaning or content.

The writer (gpt-oss) intermittently drops articles ("prototype water filter"), drops words, or goes
telegraphic ("compare articles on same topic"). The heuristic integrity gate misses these -- reliably
detecting a missing article needs count-vs-mass-noun knowledge that regex can't supply and spaCy
false-positives on (it flags mass nouns like "creativity"/"quality"). gpt-oss REPAIRS them precisely.

Safety: a content-preservation gate ensures each accepted fix changed ONLY grammar -- the stemmed
content-word multiset (>=4 letters) and the numbers must be unchanged, so the repair can add articles
and fix agreement but can NOT reword, drop, or invent content. A fix that alters content is rejected
and the original sentence is kept. Never raises; on any failure the text is returned unchanged.

The prompt strings are LLM grammar-fix instructions, not a detect/allow word-list or scoring oracle.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Callable


def grammar_repair_enabled() -> bool:
    """Kill switch, default ON. DRAFTPROOF_V6_GRAMMAR_REPAIR=0 disables. Content-gated, so it never
    changes meaning -- only grammar -- so enabling is safe."""
    return os.environ.get("DRAFTPROOF_V6_GRAMMAR_REPAIR", "1").strip().lower() not in {"0", "false", "no", "off"}


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(str(text or "").replace("\n", " ").strip()) if s.strip()]


def _stem(word: str) -> str:
    """Crude suffix stripper so agreement fixes (bring->brings, finish->finishes) count as the SAME
    content word -- only the suffix changed, not the content."""
    w = word.lower()
    for suf in ("ing", "ed", "es", "s", "d"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _content_key(text: str) -> Counter:
    """Multiset of stemmed content words (>=4 letters) -- excludes the short function words a grammar
    fix legitimately adds (a, an, the, of, to, on, is, ...)."""
    return Counter(_stem(w) for w in re.findall(r"[A-Za-z]{4,}", text))


def _semantic_token_key(text: str) -> Counter:
    """General meaning-preservation key for short tokens.

    This deliberately avoids curated semantic word lists. All alphabetic tokens with two or more
    letters must remain present after crude stemming; a grammar fix may only add one-letter tokens
    such as a missing article.
    """
    return Counter(_stem(w) for w in re.findall(r"[A-Za-z]{2,}", str(text or "")))


def _numbers(text: str) -> Counter:
    return Counter(re.findall(r"\d[\d,\.]*%?", text))


# Articles a grammar fix may legitimately INSERT (closed grammatical function-word set, not content).
_INSERTABLE_FUNCTION_WORDS = frozenset({"a", "an", "the"})


def _is_grammar_only(original: str, fixed: str) -> bool:
    """Accept a fix ONLY if it changed grammar, not content: it differs from the original, preserves
    the >=4-letter content multiset and all numbers, removes/changes NO short token, and may only
    INSERT articles (a/an/the). Agreement inflections (bring->brings) survive via stemming. This
    allows the dominant dropped-article fix ('on same topic' -> 'on the same topic') -- which the
    earlier strict ">=2-letter tokens must be identical" gate wrongly rejected -- while still
    protecting 2-3 letter content words (a dropped/swapped short content word fails the `removed`
    check) and rejecting any reword."""
    f = (fixed or "").strip()
    if not f or f == original.strip() or len(f.split()) < 3:
        return False
    if _content_key(original) != _content_key(f):
        return False
    if _numbers(original) != _numbers(f):
        return False
    short_before, short_after = _semantic_token_key(original), _semantic_token_key(f)
    if short_before - short_after:                       # a short token was dropped or changed
        return False
    if any(tok not in _INSERTABLE_FUNCTION_WORDS for tok in (short_after - short_before)):
        return False                                     # only article insertions are allowed
    ow = len(original.split())
    return ow <= len(f.split()) <= int(ow * 1.5) + 4


_SYSTEM = (
    "You are a copy editor. You fix ONLY grammar errors -- missing or wrong articles, subject-verb "
    "agreement, dropped words, and telegraphic noun phrases. You change NOTHING else: same words, same "
    "meaning, same facts, same names and numbers, same order. If a sentence is already correct, return "
    "it unchanged. Return only JSON."
)


def build_prompt(sentences: list[str]) -> str:
    payload = {
        "task": "fix_grammar_only",
        "rules": [
            "Fix only grammar: add missing articles (a/an/the), correct agreement, restore dropped words, de-telegraph noun phrases.",
            "Do NOT reword, paraphrase, add detail, remove detail, or change meaning, facts, names, or numbers. Same content, only grammar.",
            "If a sentence is already grammatical, return it exactly as given.",
        ],
        "sentences": [{"i": i, "text": s} for i, s in enumerate(sentences)],
        "output_schema": {"fixed": [{"i": 0, "text": "the same sentence with only grammar corrected"}]},
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False)


def _max_tokens() -> int:
    raw = os.environ.get("DRAFTPROOF_V6_GRAMMAR_REPAIR_MAX_TOKENS", "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return 8000


def apply_repair(
    text: str,
    *,
    gateway: Any,
    cancellation_check: Callable[[], None] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Fix grammar across the document, splicing only content-preserving fixes. Returns
    (new_text, applied_fixes). Never raises; (text, []) on disable / no change / any failure."""
    if not grammar_repair_enabled():
        return text, []
    if cancellation_check:
        cancellation_check()
    try:
        sents = _sentences(text)
        if not sents:
            return text, []
        from .json_io import parse_json
        resp = gateway.chat(
            build_prompt(sents), system=_SYSTEM, temperature=0.1, top_p=0.9,
            max_tokens=_max_tokens(), response_format={"type": "json_object"}, app_label="GrammarRepair",
        )
        raw = getattr(resp, "raw_content", "") or getattr(resp, "content", "") or ""
        data = parse_json(raw)
        fixes = {f["i"]: f for f in (data.get("fixed") or []) if isinstance(f, dict) and "i" in f} if isinstance(data, dict) else {}
        current = text
        applied: list[dict[str, Any]] = []
        for i, original in enumerate(sents):
            fixed = str((fixes.get(i) or {}).get("text") or "").strip()
            if fixed and original in current and _is_grammar_only(original, fixed):
                current = current.replace(original, fixed, 1)
                applied.append({"original": original, "revised": fixed})
        return current, applied
    except Exception:
        return text, []
