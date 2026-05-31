"""LLM-authored worked-example teaching CASES for the rewrite showcase.

For each rewrite, this authors a small set of case studies from the ACTUAL before/after content:
for the strongest grounding moves (generic/abstract claim -> concrete, first-hand, specific), it
explains the move, why it lands, and the transferable rule the user applies to THEIR OWN writing.

This is a TEACHING layer — a worked example to learn from, like a teacher working a problem on the
board. It is NOT a humanizer / detection-evasion tool: the prompt teaches grounding & specificity
and never instructs on evading any detector. The user studies the cases, then rewrites their own
work with their real specifics.

Robust by design: any failure path (disabled, no real change, LLM error, bad JSON) returns [] so the
rewrite is never blocked or altered.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Paragraphs shorter than this are treated as headings (e.g. "Introduction") and skipped — there is
# no grounding lesson in a heading.
_HEADING_MAX_WORDS = 11


def showcase_cases_enabled() -> bool:
    """Kill switch. Default ON; set DRAFTPROOF_SHOWCASE_CASES_ENABLED=0 to disable."""
    return os.environ.get("DRAFTPROOF_SHOWCASE_CASES_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", str(text or "")) if p.strip()]


def _word_count(text: str) -> int:
    return len(str(text or "").split())


def changed_paragraph_pairs(original_text: str, final_text: str, *, limit: int = 6) -> list[dict]:
    """Align original<->rewritten BY PARAGRAPH INDEX (the rewrite edits in place and preserves
    structure) and return the changed, non-heading pairs. If paragraph counts differ (a rare
    merge/split), returns [] rather than mis-pair."""
    op = _split_paragraphs(original_text)
    np = _split_paragraphs(final_text)
    if not op or len(op) != len(np):
        return []
    pairs: list[dict] = []
    for o, n in zip(op, np):
        if o.strip() == n.strip():
            continue
        if _word_count(o) < _HEADING_MAX_WORDS and _word_count(n) < _HEADING_MAX_WORDS:
            continue
        pairs.append({"original": o, "rewritten": n})
        if len(pairs) >= limit:
            break
    return pairs


_SYSTEM = (
    "You are a writing teacher building short worked-example CASES from a before/after rewrite, so a "
    "student learns to ground and specify their OWN writing. You teach the technique; you never help "
    "text evade AI detection. Quote the real text verbatim and keep every explanation concrete."
)

_CASE_FIELDS = ("submitted_quote", "marker_sees", "move_label", "rewritten_quote", "why_it_lands", "your_rule")


def _build_prompt(pairs: list[dict], *, max_cases: int) -> str:
    blocks = []
    for i, p in enumerate(pairs, 1):
        blocks.append(f"[Paragraph {i}]\nSUBMITTED:\n{p['original']}\n\nREWRITTEN:\n{p['rewritten']}")
    body = "\n\n".join(blocks)
    return (
        "Below are paragraphs a student submitted, beside the rewritten versions.\n\n"
        f"{body}\n\n"
        f"Choose the {max_cases} STRONGEST changes where the rewrite replaced a generic, abstract, or "
        "unverifiable claim with concrete, first-hand, or specific content — a real name/place/number, "
        "a lived observation, or a precise mechanism. SKIP any change that is only cosmetic rewording "
        "with no grounding gain (return fewer cases rather than padding with weak ones).\n\n"
        "For each chosen change, write a teaching case as a JSON object with EXACTLY these fields:\n"
        '- "submitted_quote": the weak phrase, quoted verbatim from SUBMITTED (one sentence or clause).\n'
        '- "marker_sees": 2-3 sentences on why that reads generic — what a reader cannot picture or verify.\n'
        '- "move_label": a short name for the technique, e.g. "Abstraction -> observed instance".\n'
        '- "rewritten_quote": the grounded phrase, quoted verbatim from REWRITTEN.\n'
        '- "why_it_lands": 2-3 sentences on why the grounded version works (verifiable particulars, lived texture).\n'
        '- "your_rule": 1-2 sentences addressed to "you" — a rule the student reuses in their own writing.\n\n'
        "Write in the same language as the text. Be concrete and specific. Never mention AI detectors or "
        'evasion. Return ONLY JSON of the form {"cases": [ ... ]}.'
    )


def _coerce_cases(raw: Any, *, max_cases: int) -> list[dict]:
    items = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        case = {f: str(item.get(f) or "").strip() for f in _CASE_FIELDS}
        # The two quotes are the spine of a case; without a why/rule it is not yet a lesson.
        if not case["submitted_quote"] or not case["rewritten_quote"]:
            continue
        if not (case["why_it_lands"] or case["your_rule"]):
            continue
        out.append(case)
        if len(out) >= max_cases:
            break
    return out


def author_showcase_cases(
    original_text: str,
    final_text: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    cancellation_check=None,
    client=None,
    max_cases: int = 4,
) -> list[dict]:
    """Author worked teaching cases from the rewrite. Returns [] on disable / no real change / any
    failure — it must never block or alter the rewrite. Pass `client` to inject a fake in tests."""
    try:
        if not showcase_cases_enabled():
            return []
        pairs = changed_paragraph_pairs(original_text, final_text)
        if not pairs:
            return []
        if client is None:
            from .llm_config import showcase_gateway
            client = showcase_gateway(api_key=api_key, base_url=base_url, cancellation_check=cancellation_check)
        prompt = _build_prompt(pairs, max_cases=max_cases)
        resp = client.chat(prompt, system=_SYSTEM, response_format={"type": "json_object"})
        content = getattr(resp, "content", resp)
        content = content if isinstance(content, str) else str(content or "")
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            from .json_io import parse_json  # tolerant parser used across v6
            parsed = parse_json(content)
        return _coerce_cases(parsed, max_cases=max_cases)
    except Exception as exc:  # pragma: no cover - safety net; never break the rewrite
        logger.warning("showcase_cases authoring failed: %s", exc)
        return []
