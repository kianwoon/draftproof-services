"""Lean, finding-targeted writer prompt.

The default writer brief (`build_writer_brief_prompt`) ships ~31KB / ~7.9K tokens per paragraph
per request: planner_brief, writer_execution_plan, coverage_beats, construction_route,
style_contract, hard_reject_if, prose_repair_rules, etc. That floods the LLM context and buries
the one thing that matters -- the paragraph's actual scanner findings.

This builder sends only what's needed to resolve those findings:
  source paragraph + the flagged sentences and why + must-keep terms + meaning/polarity rules +
  concise burstiness/predictability guidance.

Goal alignment: lower token-predictability by concrete phrasing (NOT by splitting into short
uniform sentences, which tanks burstiness) and deliberately vary sentence length. Process one
paragraph at a time so context stays small. Enabled with DRAFTPROOF_V6_LEAN_WRITER=1.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .plan import Plan
from .rewrite_playbook import playbook_entries
from .text import Paragraph
from .writer_brief_prompt import _must_keep_terms, _variant_requirements


def _predictability_mode() -> bool:
    """Stage 2: feed the graded predictable token-spans to the writer and free those tokens from
    must_keep so it can break them. Shares the Stage 1 switch (DRAFTPROOF_V6_SCANNER_PREDICTABILITY)
    so the whole predictability path turns on together. Off by default."""
    return os.environ.get("DRAFTPROOF_V6_SCANNER_PREDICTABILITY", "").strip().lower() in {"1", "true", "yes", "on"}

# Scanner finding tag -> concise fix. Every fix breaks predictability via concreteness, never by
# splitting; list/rhythm fixes explicitly preserve sentence-length variation.
_FINDING_FIX: dict[str, str] = {
    "packed_list": "Packs many comma-listed items in a predictable 'A, B, C, and D' enumeration. Re-express the relationship concretely; a natural list is fine, but break the mechanical rhythm.",
    "predictable_start": "Opens with a predictable filler subject (This/They/It/These/Those). Start from a concrete actor, setting, or observation instead.",
    "context_anchor_gap": "Abstract claim with no concrete anchor. Ground it in a specific actor, action, or setting already present in the source.",
    "paragraph_rhythm": "Sentence length/shape is too uniform with its neighbours. Make it noticeably shorter or longer to vary the rhythm.",
    "abstract_noun_stack": "Abstract-noun pile-up. Convert to actor-action wording: who does what.",
    "broad_claim": "Claim is too broad. Narrow it by setting, task, actor, or condition.",
    "transition_stack": "Stacked transition/cause-effect glue. State the next concrete claim directly.",
    "semantic_bridge_gap": "Reads as smooth filler between ideas. Replace with a concrete source relation.",
}
_DEFAULT_FIX = "Break the predictable token path with specific, concrete phrasing -- do not split into short uniform sentences."

_POLARITY_MARKERS = ("not only", "not always", "no longer", "rather than", "instead of", "without", "not less")


def build_lean_finding_prompt(paragraph: Paragraph, plan: Plan) -> str:
    lowered = str(paragraph.text or "").casefold()
    polarity = [marker for marker in _POLARITY_MARKERS if marker in lowered]
    examples = _playbook_examples(plan, paragraph) if _predictability_mode() else []
    payload: dict[str, Any] = {
        "task": "rewrite_one_paragraph_to_clear_its_findings",
        "source_paragraph": paragraph.text,
        "findings_to_resolve": _paragraph_findings(plan),
        "must_keep_terms": _resolved_must_keep_terms(paragraph, plan),
        **({"rewrite_examples": examples} if examples else {}),
        "meaning_rules": [
            "Preserve the submitted meaning, factual scope, and first-person voice when present.",
            "Keep every must_keep_term; rephrasing around them is fine, dropping or replacing them is not.",
            *[f"Preserve the '{marker}' relationship exactly; do not invert, drop, or weaken it." for marker in polarity],
        ],
        "rewrite_guidance": [
            "Goal: lower AI-detection risk by resolving findings_to_resolve while keeping the meaning intact.",
            "Lower predictability through concrete, specific word choices -- NOT by splitting into many short sentences.",
            "Vary sentence length deliberately: include at least one short sentence (4-8 words) and at least one longer coordinated sentence (about 25-40 words). Never write a run of similar-length sentences.",
            "Rewrite the paragraph as a whole; do not produce one sentence per finding.",
            *(
                [
                    "STRUCTURE IS THE BIGGEST LEVER. Give EVERY sentence a different opening and a different "
                    "shape -- never start two sentences with the same word or the same subject-verb frame. "
                    "Deliberately mix a very short sentence (4-8 words) with a long one (25-40 words).",
                    "Cut hedging hard. Replace may, might, can, could, should, often, generally, tends to, "
                    "and similar qualifiers with direct statements wherever the meaning allows -- dense "
                    "hedging is a strong AI signal.",
                    "Ground broad, generic claims in a specific concrete detail: a named tool already in the "
                    "source, a concrete scene, or a first-person observation. As the Author-Proxy you MAY add "
                    "such a grounding detail to fix an author_anchor / context_anchor / generic finding -- but "
                    "list any detail not directly in the source in author_review_items for the student to "
                    "confirm. Never invent facts, statistics, citations, names, or events.",
                    "Each finding lists predictable_phrases_to_rephrase: rephrase every one of them with "
                    "concrete, specific wording. Words inside them are NOT in must_keep -- replace them freely "
                    "as long as the meaning survives. Where a finding lists keep_unchanged, leave it intact.",
                    "rewrite_examples shows the SHAPE of the move to make for each problem (before -> better). "
                    "Apply the same kind of transformation to THIS paragraph's content -- do not copy the "
                    "example wording.",
                ]
                if _predictability_mode()
                else []
            ),
        ],
        "variant_requirements": _variant_requirements(plan),
        "output_schema": {
            "variants": [
                {
                    "id": "v1|v2|v3",
                    "mode": "one of variant_requirements.mode",
                    "text": "rewritten paragraph only",
                    "author_proxy_provenance": [],
                    "author_review_items": [],
                }
            ]
        },
        "output_rules": [
            "Return exactly the requested variant ids; each variant is one complete rewritten paragraph.",
            "Each variant must resolve the findings_to_resolve and keep every must_keep_term.",
            "Return valid JSON only.",
        ],
    }
    return "Return valid JSON only with a variants array.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _playbook_examples(plan: Plan, paragraph: Paragraph) -> list[dict[str, str]]:
    """Concrete before->better transformations matched to this paragraph's findings + patterns."""
    tags: list[str] = []
    for action in plan.actions:
        tags.extend(str(tag) for tag in (action.tags or []) if str(tag).strip())
        if action.predictable_spans:
            tags.append("review_predictability")
    return playbook_entries(tags, paragraph.text)


def _paragraph_findings(plan: Plan) -> list[dict[str, Any]]:
    mode = _predictability_mode()
    rows: list[dict[str, Any]] = []
    for action in plan.actions:
        tags = [str(tag) for tag in (action.tags or []) if str(tag).strip()]
        spans = [str(span) for span in (action.predictable_spans or []) if str(span).strip()] if mode else []
        # In predictability mode a sentence is worth rewriting if the graded detector flagged
        # predictable spans even when no structural tag fired.
        if not tags and not spans:
            continue
        seen: set[str] = set()
        fixes: list[str] = []
        for tag in tags:
            fix = _FINDING_FIX.get(tag, _DEFAULT_FIX)
            if fix not in seen:
                seen.add(fix)
                fixes.append(fix)
        if not fixes:
            fixes.append(_DEFAULT_FIX)
        row: dict[str, Any] = {"sentence": action.source_text, "issues": tags, "fix": " ".join(fixes)}
        if spans:
            row["predictable_phrases_to_rephrase"] = spans[:8]
            protected = [str(span) for span in (action.protected_spans or []) if str(span).strip()]
            if protected:
                row["keep_unchanged"] = protected[:8]
        rows.append(row)
    return rows


def _resolved_must_keep_terms(paragraph: Paragraph, plan: Plan) -> list[str]:
    """Meaning anchors the writer must keep. In predictability mode, the flagged predictable
    single-word tokens are removed (so the writer can change them) while named/quoted/multi-word
    anchors, scope markers, and the detector's protected spans stay."""
    must_keep = _must_keep_terms(paragraph, plan)
    if not _predictability_mode():
        return must_keep
    span_words: set[str] = set()
    protected: list[str] = []
    for action in plan.actions:
        for span in action.predictable_spans or []:
            for word in re.findall(r"[a-z']+", str(span).casefold()):
                if len(word) > 2:
                    span_words.add(word)
        protected.extend(str(span) for span in (action.protected_spans or []) if str(span).strip())
    reduced: list[str] = []
    for term in must_keep:
        lowered = term.casefold()
        # keep multi-word anchors, scope/polarity markers, and proper-noun/quoted style terms
        if " " in term or lowered in _POLARITY_MARKERS or term[:1].isupper():
            reduced.append(term)
            continue
        if lowered in span_words:
            continue  # predictable filler -- let the writer replace it
        reduced.append(term)
    for span in protected:
        if span not in reduced:
            reduced.append(span)
    return reduced
