"""Surgical top-k reduction -- a MODERATE pass after the QC reviewer (product-owner directed).

Targets ONLY the highest-top-k (most statistically predictable) sentences and rewrites them to be
less predictable while staying FLUENT and FAITHFUL. Each fix is GATED: applied only if it genuinely
lowers that sentence's top-k AND keeps grammar + meaning + length. Stops once the document reaches
the target (default ~0.50) -- moderate by design, never crushed to gibberish.

HONEST CAVEAT (do not forget): lowering OUR GPT-2 top-k via wording is anti-correlated with real
third-party detectors -- proven this session (a clean reword moved GPTZero 78->100). This pass lowers
the DISPLAYED top-k; it is NOT proven to lower real GPTZero/Turnitin flagging. Keep the external
estimate honest (see feedback_expose_ugly_side) and validate against GPTZero before trusting it.

allow-hardcode: the _SYSTEM / prompt strings below are the LLM rewrite instructions, not a
detect/allow word-list or scoring oracle.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable


def topk_surgical_enabled() -> bool:
    """Feature flag, default OFF. Measured on real docs: the gated pass finds 0 FLUENT fixes that
    lower top-k -- top-k is dominated by function words (grammar), so the only way down is gibberish,
    which the fluency gate correctly rejects. Shipping it ON would add GPT-2 latency for zero gain.
    DRAFTPROOF_V6_TOPK_SURGICAL=1 to enable anyway (it will no-op on fluent text)."""
    return os.environ.get("DRAFTPROOF_V6_TOPK_SURGICAL", "0").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return default


def _target() -> float:
    """Stop once the document's content-word top-k fraction reaches this (~0.50 = 'well balanced')."""
    return _env_float("DRAFTPROOF_V6_TOPK_TARGET", 0.50)


def _high_cutoff() -> float:
    """Only treat sentences whose own top-k fraction is at/above this (the HIGH offenders)."""
    return _env_float("DRAFTPROOF_V6_TOPK_HIGH_CUTOFF", 0.50)


def _max_fixes() -> int:
    """Moderate cap on sentences rewritten per document."""
    return _env_int("DRAFTPROOF_V6_TOPK_MAX_FIXES", 8)


def _topk_k() -> int:
    return _env_int("DRAFTPROOF_V6_TOPK_K", 10)


# ---------------------------------------------------------------------------
# GPT-2 per-sentence top-k (same model the predictability scanner uses)
# ---------------------------------------------------------------------------
_MODEL = None
_TOK = None
_UNAVAIL = False


def _gpt2():
    global _MODEL, _TOK, _UNAVAIL
    if _UNAVAIL:
        return None, None
    if _MODEL is None:
        try:
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
            _TOK = GPT2TokenizerFast.from_pretrained("gpt2")
            _MODEL = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        except Exception:
            _UNAVAIL = True
            return None, None
    return _MODEL, _TOK


def _topk_fraction(text: str, k: int) -> float:
    """Fraction of ALL tokens whose GPT-2 next-token rank is < k -- matches the predictability
    scanner's measure (poc/predictability/scanner.py: top_10 = rank <= 10, over every token), which
    is what the report's Raw Top-k Predictability shows. 0.0 if GPT-2 unavailable / too short."""
    model, tok = _gpt2()
    if model is None or not str(text).strip():
        return 0.0
    import torch
    ids = tok(text)["input_ids"][:1024]
    if len(ids) < 2:
        return 0.0
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]
    hits = 0
    for i in range(1, len(ids)):
        if int((logits[i - 1] > logits[i - 1][ids[i]]).sum().item()) < k:
            hits += 1
    tot = len(ids) - 1
    return round(hits / tot, 4) if tot else 0.0


def _sentence_topk(sentence: str, k: int) -> float:
    return _topk_fraction(sentence, k)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(str(text or "").replace("\n", " ").strip()) if s.strip()]


def _doc_topk(text: str, k: int) -> float:
    """Document ALL-token top-k fraction over the FULL document (continuous context, chunked at the
    1024-token GPT-2 limit) -- matches the scanner's Raw Top-k Predictability. Per-sentence context
    reset under-reads, which previously made the pass early-return; this uses full context."""
    model, tok = _gpt2()
    if model is None:
        return 0.0
    import torch
    ids = tok(str(text))["input_ids"]
    if len(ids) < 2:
        return 0.0
    hits = tot = 0
    for start in range(0, len(ids), 1024):
        chunk = ids[start:start + 1024]
        if len(chunk) < 2:
            continue
        with torch.no_grad():
            logits = model(torch.tensor([chunk])).logits[0]
        for i in range(1, len(chunk)):
            tot += 1
            if int((logits[i - 1] > logits[i - 1][chunk[i]]).sum().item()) < k:
                hits += 1
    return round(hits / tot, 4) if tot else 0.0


_SYSTEM = (
    "You rewrite individual sentences to use less statistically predictable wording while keeping the "
    "EXACT meaning, all facts, names and numbers, and clean fluent grammar. Never reverse the claim, "
    "never add or drop information, never produce awkward or telegraphic prose. Return only JSON."
)


def build_prompt(sentences: list[str]) -> str:
    payload = {
        "task": "reduce_predictable_wording_per_sentence",
        "rules": [
            "Rewrite each sentence to rely less on the most expected wording -- choose more particular, less common phrasing by re-routing the sentence, not by swapping a single synonym.",
            "Preserve EXACT meaning, stance, facts, names and numbers. Keep it fluent and grammatical. No awkward, telegraphic, or thesaurus-salad output.",
        ],
        "sentences": [{"i": i, "text": s} for i, s in enumerate(sentences)],
        "output_schema": {"fixes": [{"i": 0, "text": "less-predictable rewrite of the same sentence"}]},
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False)


def _is_safe_fix(original: str, candidate: str, k: int) -> bool:
    """Gate: keep a fix ONLY if it genuinely lowers the sentence's top-k AND stays fluent + faithful.
    Fluent = not broken grammar; faithful = no polarity inversion + length within +/-40% of source."""
    from .direct_rewrite import _has_broken_grammar
    from .selector_diagnostics import _severe_polarity_inversion
    from .text import Paragraph
    c = (candidate or "").strip()
    if not c or c == original.strip() or len(c.split()) < 3:
        return False
    ow = len(original.split())
    if not (0.6 * ow <= len(c.split()) <= 1.4 * ow + 4):
        return False
    if _has_broken_grammar(c):
        return False
    if _severe_polarity_inversion(c, Paragraph(id="tk", index=0, text=original, sentences=[])):
        return False
    return _sentence_topk(c, k) < _sentence_topk(original, k)


def apply_surgical(
    text: str,
    *,
    gateway: Any,
    cancellation_check: Callable[[], None] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Moderately reduce top-k by rewriting only the HIGH-top-k sentences, gated. Returns
    (new_text, applied_fixes). Never raises; returns (text, []) on disable / no high offenders /
    already at target / GPT-2 unavailable / any failure."""
    if not topk_surgical_enabled():
        return text, []
    if cancellation_check:
        cancellation_check()
    try:
        k = _topk_k()
        if _doc_topk(text, k) <= _target():
            return text, []  # already balanced -- do nothing
        cutoff = _high_cutoff()
        scored = sorted(
            ((s, _sentence_topk(s, k)) for s in _split(text) if len(s.split()) >= 8),
            key=lambda x: x[1], reverse=True,
        )
        candidates = [s for s, tk in scored if tk >= cutoff][: _max_fixes()]
        if not candidates:
            return text, []
        from .json_io import parse_json
        resp = gateway.chat(
            build_prompt(candidates), system=_SYSTEM, temperature=0.6, top_p=0.95,
            max_tokens=_env_int("DRAFTPROOF_V6_TOPK_MAX_TOKENS", 4000),
            response_format={"type": "json_object"}, app_label="TopkSurgical",
        )
        raw = getattr(resp, "raw_content", "") or getattr(resp, "content", "") or ""
        data = parse_json(raw)
        fixes = {f["i"]: f for f in (data.get("fixes") or []) if isinstance(f, dict) and "i" in f} if isinstance(data, dict) else {}
        current = text
        applied: list[dict[str, Any]] = []
        for i, original in enumerate(candidates):
            if _doc_topk(current, k) <= _target():
                break  # moderate: stop once balanced
            cand = str((fixes.get(i) or {}).get("text") or "").strip()
            if cand and original in current and _is_safe_fix(original, cand, k):
                current = current.replace(original, cand, 1)
                applied.append({"original": original, "revised": cand})
        return current, applied
    except Exception:
        return text, []
