"""Grounding SHOWCASE -- a TEACHING layer that runs AFTER the QC reviewer.

It SHOWS the user worked before->after examples that turn a GENERIC claim into GROUNDED, specific
writing, then says "now use your OWN real detail." DraftProof does the teaching (demonstrates the
technique on a concrete example); the user supplies their truth -- the normal teacher/student split,
not a hand-off of responsibility.

Two hard rules from the product owner:
  1. Only GENUINELY GOOD examples are shown -- a generic sentence that actually becomes concretely
     grounded, faithful, and grammatical. If none clear the bar, nothing is shown. We never teach a
     weak example: "if the showcase is no good, then no teaching."
  2. Graded on GROUNDING (real, movable, honest), NOT on AI-detector scores. The detector is a noisy
     oracle that over-flags even genuine human writing and often moves the WRONG way when writing
     improves -- so showing it next to a lesson makes a good lesson look like a failure. We measure
     the thing we teach: did the writing get concretely grounded?

Annotate-only: never mutates the shipped rewrite. Content-agnostic (reuses the scanner's structural
concreteness check -- no banned/allow word lists). No GPT-2/LLM-detector loop -> light and safe.
Never expose 'perplexity'/'top-k'/'AI-detector' jargon to end users in the copy.

allow-hardcode: the _SYSTEM string and build_showcase_prompt rules below are the LLM coaching prompt
(instructions + output schema), human-reviewed guidance -- NOT a detect/allow word-list or scoring oracle.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable


def showcase_enabled() -> bool:
    """Feature flag, default ON. Set DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE=0 to disable. Annotate-only,
    so toggling never changes the shipped rewrite -- only whether the teaching layer runs."""
    return os.environ.get("DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _max_sentences() -> int:
    """Cap on showcased examples -- the most useful generic sentences, to bound LLM cost."""
    raw = os.environ.get("DRAFTPROOF_V6_SHOWCASE_MAX_SENTENCES", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return 8


def _is_grounded(sentence: str) -> bool | None:
    """Reuse the scanner's content-agnostic structural-concreteness check (numbers, named entities,
    quotes, first-person lived detail, exemplification, temporal anchors). True = the sentence has
    real grounding; False = generic claim. None if the detector can't be imported (fail-open)."""
    try:
        from detect.layer3_scoring import _sentence_has_concrete_or_context
    except Exception:
        try:
            from poc.detect.layer3_scoring import _sentence_has_concrete_or_context
        except Exception:
            return None
    try:
        return bool(_sentence_has_concrete_or_context(sentence))
    except Exception:
        return None


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(str(text or "").replace("\n", " ").strip()) if s.strip()]


def _generic_candidates(text: str, limit: int) -> list[str]:
    """The generic (un-grounded) sentences -- the ones grounding can genuinely improve. These are the
    teachable cases; a sentence that is already grounded has no lesson to show."""
    out: list[str] = []
    for s in _sentences(text):
        if len(s.split()) < 6:           # too short to ground meaningfully
            continue
        if _is_grounded(s) is False:      # only flag confidently-generic sentences
            out.append(s)
        if len(out) >= limit:
            break
    return out


@dataclass
class ShowcaseItem:
    sentence: str          # the generic original
    suggestion: str        # a grounded worked example (the user replaces specifics with their own)
    why: str               # one-line, plain language: what concrete detail was added
    grounded_before: bool = False
    grounded_after: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence": self.sentence,
            "suggestion": self.suggestion,
            "why": self.why,
            "grounded_before": self.grounded_before,
            "grounded_after": self.grounded_after,
        }


_SYSTEM = (
    "You are a writing coach. You receive sentences that state a claim WITHOUT concrete grounding. For "
    "each, show a more grounded version of the SAME claim -- anchored in a specific moment, number, "
    "name, or first-hand detail -- as an ILLUSTRATIVE example the writer will replace with their own "
    "real specifics. Keep the claim's meaning and stance; keep grammar clean. Return only JSON."
)


def build_showcase_prompt(candidates: list[str]) -> str:
    payload = {
        "task": "teach_grounding_by_worked_example",
        "note": "Each sentence states a claim without concrete grounding. Show how to ground it.",
        "rules": [
            "Keep the original claim's meaning and stance; never reverse it. Add concrete specifics (a number, a named example, a first-hand moment); keep grammar clean.",
            "Make the specifics realistic but clearly example-level -- the writer will swap in their own real detail.",
            "'why' is one short plain sentence naming the concrete detail you added. Do not mention detectors, perplexity, or AI scores.",
        ],
        "sentences": [{"i": i, "text": s} for i, s in enumerate(candidates)],
        "output_schema": {"examples": [{"i": 0, "grounded": "a more grounded version of the same claim", "why": "adds a specific ..."}]},
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False)


def _is_teachable(original: str, grounded: str) -> bool:
    """The quality gate: show ONLY genuinely good examples. The grounded version must actually become
    grounded (concrete anchors present), stay grammatical, be non-trivial, and differ from the
    original. If it doesn't truly improve, it is not taught."""
    from .direct_rewrite import _has_broken_grammar
    g = (grounded or "").strip()
    if not g or len(g.split()) < 5 or g == original.strip():
        return False
    if _is_grounded(g) is not True:        # must actually be grounded now
        return False
    if _has_broken_grammar(g):
        return False
    return True


def _showcase_max_tokens() -> int:
    raw = os.environ.get("DRAFTPROOF_V6_SHOWCASE_MAX_TOKENS", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return 8000


def generate_showcase(
    text: str,
    *,
    gateway: Any,
    cancellation_check: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Produce validated grounding examples for `text`. Never raises; returns [] on disable / no
    generic sentences / no good examples / any failure (annotate-only: the rewrite is unaffected).
    [] is a correct outcome -- we'd rather show nothing than a weak lesson."""
    if not showcase_enabled():
        return []
    if cancellation_check:
        cancellation_check()
    try:
        candidates = _generic_candidates(text, _max_sentences())
        if not candidates:
            return []
        from .json_io import parse_json
        response = gateway.chat(
            build_showcase_prompt(candidates),
            system=_SYSTEM,
            temperature=0.7,
            top_p=0.95,
            max_tokens=_showcase_max_tokens(),
            response_format={"type": "json_object"},
            app_label="GroundingShowcase",
        )
        raw = getattr(response, "raw_content", "") or getattr(response, "content", "") or ""
        data = parse_json(raw)
        examples = {e["i"]: e for e in (data.get("examples") or []) if isinstance(e, dict) and "i" in e} if isinstance(data, dict) else {}
        items: list[dict[str, Any]] = []
        for i, original in enumerate(candidates):
            ex = examples.get(i)
            if not ex:
                continue
            grounded = str(ex.get("grounded") or ex.get("suggestion") or "").strip()
            if not _is_teachable(original, grounded):
                continue
            items.append(ShowcaseItem(
                sentence=original,
                suggestion=grounded,
                why=str(ex.get("why") or "").strip(),
            ).to_dict())
        return items
    except Exception:
        return []
