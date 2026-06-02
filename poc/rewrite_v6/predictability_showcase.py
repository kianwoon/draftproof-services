"""Predictability SHOWCASE -- a TEACHING layer that runs AFTER the QC reviewer.

It is NOT a humanizer and NOT a mutation of the shipped rewrite. It ANNOTATES the final reviewed text
with worked examples so the user LEARNS to write more distinctively themselves. For each
high-predictability sentence it surfaces: the statistically-predictable words (GPT-2 top-k hits), a
less-predictable but meaning/grammar/fact-preserving alternative (LLM), and the MEASURED reduction.
Only validated reductions are shown, so every example teaches something true.

Why a showcase and not an auto-fix: top-k predictability is the intrinsic LLM-prose floor and cannot
be honestly lowered by rewriting the *submission* (proven -- gaming it backfires/produces gibberish,
external detectors are unmoved). The only genuine lever is a human writing more distinctively, so the
product's job is to TEACH that by example. This module is that engine. Never expose
'perplexity'/'top-k' jargon to end users -- the report copy frames it as "predictable phrasing".
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable


def showcase_enabled() -> bool:
    """Feature flag. Default OFF: this teaching layer adds GPT-2 + LLM latency to every rewrite and
    has no UI yet, so it ships dark. Enable with DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE=1 once the
    frontend renders it. Annotate-only, so toggling never changes the shipped rewrite."""
    return os.environ.get("DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return default


def _topk() -> int:
    """A content word counts as 'predictable' if its GPT-2 next-token rank is within this k."""
    return _env_int("DRAFTPROOF_V6_SHOWCASE_TOPK", 10)


def _max_sentences() -> int:
    """Cap on showcased examples -- the most-predictable sentences, to bound latency/cost."""
    return _env_int("DRAFTPROOF_V6_SHOWCASE_MAX_SENTENCES", 8)


# ---------------------------------------------------------------------------
# GPT-2 word-rank scoring (same model the predictability scanner uses)
# ---------------------------------------------------------------------------
_MODEL = None
_TOKENIZER = None
_GPT2_UNAVAILABLE = False


def _ensure_gpt2():
    """Lazy-load gpt2 once. Returns (model, tokenizer) or (None, None) if torch/transformers are
    unavailable -- the feature then no-ops gracefully rather than breaking the rewrite."""
    global _MODEL, _TOKENIZER, _GPT2_UNAVAILABLE
    if _GPT2_UNAVAILABLE:
        return None, None
    if _MODEL is None:
        try:
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
            _TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
            _MODEL = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        except Exception:
            _GPT2_UNAVAILABLE = True
            return None, None
    return _MODEL, _TOKENIZER


def _is_content_token(decoded: str) -> bool:
    """A word-initial (leading-space) alphabetic token of >=4 chars. The >=4 length is a
    content-agnostic filter (no banned-word list): it drops the short function words ('the', 'of',
    'to') whose predictability isn't useful to teach, keeping meaning-bearing words."""
    return decoded[:1] == " " and decoded.strip().isalpha() and len(decoded.strip()) >= 4


def _word_ranks(text: str) -> list[tuple[str, int]]:
    """Per content word, its GPT-2 next-token rank in the sentence context. Returns
    [(word, rank), ...]; empty if GPT-2 is unavailable or text is too short."""
    model, tok = _ensure_gpt2()
    if model is None or not text.strip():
        return []
    import torch
    ids = tok(text)["input_ids"][:1024]
    if len(ids) < 2:
        return []
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]
    out: list[tuple[str, int]] = []
    for i in range(1, len(ids)):
        decoded = tok.decode([ids[i]])
        if not _is_content_token(decoded):
            continue
        dist = logits[i - 1]
        rank = int((dist > dist[ids[i]]).sum().item())
        out.append((decoded.strip(), rank))
    return out


def _predictability(text: str, k: int) -> float:
    """Fraction of content words whose GPT-2 rank is within top-k (0..1). This is the showcase's
    self-consistent 'predictability' metric for before/after comparison."""
    ranks = _word_ranks(text)
    if not ranks:
        return 0.0
    hits = sum(1 for _, r in ranks if r < k)
    return round(hits / len(ranks), 4)


def _flagged_words(text: str, k: int) -> list[str]:
    """The content words GPT-2 finds predictable (rank < k), de-duplicated, original order."""
    seen: set[str] = set()
    flagged: list[str] = []
    for w, r in _word_ranks(text):
        key = w.lower()
        if r < k and key not in seen:
            seen.add(key)
            flagged.append(w)
    return flagged


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(str(text or "").replace("\n", " ").strip()) if s.strip()]


@dataclass
class ShowcaseItem:
    sentence: str
    flagged_words: list[str]
    suggestion: str
    why: str
    score_before: float
    score_after: float

    @property
    def reduction(self) -> float:
        return round(self.score_before - self.score_after, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence": self.sentence,
            "flagged_words": self.flagged_words,
            "suggestion": self.suggestion,
            "why": self.why,
            "predictability_before": self.score_before,
            "predictability_after": self.score_after,
            "reduction": self.reduction,
        }


def _candidate_sentences(text: str, k: int, limit: int) -> list[dict[str, Any]]:
    """The most-predictable sentences worth a lesson: those with flagged content words, ranked by
    predictability score, capped at `limit`."""
    cands: list[dict[str, Any]] = []
    for s in _sentences(text):
        flagged = _flagged_words(s, k)
        if not flagged:
            continue
        cands.append({"sentence": s, "flagged_words": flagged, "score": _predictability(s, k)})
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands[:limit]


def build_showcase_prompt(candidates: list[dict[str, Any]]) -> str:
    payload = {
        "task": "teach_less_predictable_phrasing",
        "note": "Each sentence relies on statistically predictable word choices (listed). Show a more distinctive way to say the SAME thing.",
        "rules": [
            "Preserve the exact meaning, all facts, numbers, names, dates, and correct grammar. Add nothing, drop nothing.",
            "Make the wording less generic/predictable -- more specific, concrete, or distinctive -- not just rarer synonyms.",
            "'why' is one short plain-language sentence on what made the original read as templated. Never use the words 'perplexity' or 'top-k'.",
        ],
        "sentences": [{"i": i, "text": c["sentence"], "predictable_words": c["flagged_words"]} for i, c in enumerate(candidates)],
        "output_schema": {"examples": [{"i": 0, "suggestion": "more distinctive version", "why": "one-line reason"}]},
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False)


_SYSTEM = (
    "You are a writing coach. You receive sentences whose wording is statistically predictable and the "
    "specific predictable words. For each, show a more distinctive way to say the SAME thing -- "
    "preserving meaning, facts, and grammar exactly -- plus a one-line reason. Return only JSON."
)


def _is_teachable(item: ShowcaseItem) -> bool:
    """Validation gate: a suggestion is only shown if it actually teaches something true.

    Default rule (tune via the constants below): the suggestion must genuinely REDUCE measured
    predictability, stay grammatical, differ from the original, and not be a stub. This is the
    quality bar that keeps the showcase honest -- no example with a fake or zero 'reduction'.
    """
    from .direct_rewrite import _has_broken_grammar
    s = (item.suggestion or "").strip()
    if not s or len(s.split()) < 3 or s == item.sentence:
        return False
    if _has_broken_grammar(s):
        return False
    return item.reduction >= _min_reduction()


def _min_reduction() -> float:
    """Minimum measured predictability drop for an example to be shown (default 0.05 = 5 points).
    Env-tunable (DRAFTPROOF_V6_SHOWCASE_MIN_REDUCTION) so example frequency can be dialed without a
    redeploy: a cleaner rewrite yields fewer validated lessons, so lower this to surface more -- at
    the cost of weaker lessons. Placement is (a): the showcase runs on the post-reviewer rewrite."""
    raw = os.environ.get("DRAFTPROOF_V6_SHOWCASE_MIN_REDUCTION", "").strip()
    if raw:
        try:
            value = float(raw)
            if value >= 0:
                return value
        except ValueError:
            pass
    return 0.05


def generate_showcase(
    text: str,
    *,
    gateway: Any,
    cancellation_check: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Produce validated teaching examples for `text`. Never raises; returns [] on disable/any
    failure/GPT-2 unavailable (the rewrite is unaffected -- this is annotate-only)."""
    if not showcase_enabled():
        return []
    if cancellation_check:
        cancellation_check()
    try:
        k = _topk()
        candidates = _candidate_sentences(text, k, _max_sentences())
        if not candidates:
            return []
        from .json_io import parse_json
        response = gateway.chat(
            build_showcase_prompt(candidates),
            system=_SYSTEM,
            temperature=0.7,
            top_p=0.95,
            max_tokens=_env_int("DRAFTPROOF_V6_SHOWCASE_MAX_TOKENS", 8000),
            response_format={"type": "json_object"},
            app_label="PredictabilityShowcase",
        )
        raw = getattr(response, "raw_content", "") or getattr(response, "content", "") or ""
        data = parse_json(raw)
        examples = {e["i"]: e for e in (data.get("examples") or []) if isinstance(e, dict) and "i" in e} if isinstance(data, dict) else {}
        items: list[dict[str, Any]] = []
        for i, c in enumerate(candidates):
            ex = examples.get(i)
            if not ex:
                continue
            suggestion = str(ex.get("suggestion") or "").strip()
            if not suggestion:
                continue
            item = ShowcaseItem(
                sentence=c["sentence"],
                flagged_words=c["flagged_words"],
                suggestion=suggestion,
                why=str(ex.get("why") or "").strip(),
                score_before=c["score"],
                score_after=_predictability(suggestion, k),
            )
            if _is_teachable(item):
                items.append(item.to_dict())
        return items
    except Exception:
        return []
