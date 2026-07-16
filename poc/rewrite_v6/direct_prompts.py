"""System prompts for the lean direct-rewrite writer (extracted from direct_rewrite.py).

Pure prompt text, no logic — split out to keep direct_rewrite.py under the 1500-LOC limit. Two
lanes: ``SYSTEM`` (control) and ``DIVERSIFIED_SYSTEM`` (author-proxy route diversity). Both instruct
the writer to lower AI-detection risk by grounding generic claims in the author's own voice while
never fabricating a real statistic, citation, or named entity.

allow-hardcode: every string in this module is LLM coaching guidance (a system prompt shown to the
writer model), NOT a detect/scoring word-list matched against user input. Moved verbatim from
direct_rewrite.py where it carried the same allow-hardcode rationale.
"""
from __future__ import annotations

from typing import Any


# Prompt guidance for the two enhanced-scan targets. Fabrication-safe: both instruct qualify /
# attribute / ground-from-author-material, and explicitly forbid inventing a source, statistic, or
# named entity — mirroring the critical_thinking handling in direct_rewrite.py.
_FLAGGED_SENTENCES_INSTRUCTION = (
    "flagged_sentences are the EXACT sentences the scan flagged, each with its issue: 'grounding' = "
    "the sentence is generic/unanchored -> rewrite it to carry a concrete anchor as the AUTHOR would "
    "(an author-proxy observation, condition, mechanism, or attributed source that fits the claim's "
    "basis; flag any illustrative specific in author_review_items; do NOT invent a real statistic, "
    "citation, or named entity). 'reasoning' = there is a logic jump -> make the connecting reasoning "
    "explicit from what the author already wrote; do NOT fabricate new evidence to bridge it. Fix EACH "
    "flagged sentence in place; leave the rest of the paragraph's meaning intact."
)
_UNSUPPORTED_CLAIMS_INSTRUCTION = (
    "unsupported_claims are claims the scan could NOT verify against their cited source (the 'verdict' "
    "says how). Do NOT delete the claim and do NOT invent a source, statistic, or study to prop it up. "
    "Instead: for 'contradicted', soften or correct the claim to match what the source actually "
    "supports, or attribute the dispute ('some sources dispute ...'); for 'paywalled' or 'unresolved', "
    "qualify it ('reportedly', 'the cited figure suggests') and make the author's own reasoning "
    "explicit. Note each such change in author_review_items so the author can supply the real support."
)


def apply_scan_target_instructions(payload: dict[str, Any], diagnosis: dict[str, Any] | None) -> None:
    """Attach per-sentence (P1) and per-claim (P2) enhanced-scan targets to a writer payload, in place.

    No-op when the diagnosis carries neither — so with both kill switches off the payload is unchanged.
    allow-hardcode: the instruction strings are LLM coaching guidance (a prompt), not a detect/scoring
    word-list matched against user input.
    """
    flagged_sentences = [r for r in ((diagnosis or {}).get("flagged_sentences") or []) if isinstance(r, dict)]
    if flagged_sentences:
        payload["flagged_sentences"] = flagged_sentences
        payload["instructions"].append(_FLAGGED_SENTENCES_INSTRUCTION)
    unsupported_claims = [r for r in ((diagnosis or {}).get("unsupported_claims") or []) if isinstance(r, dict)]
    if unsupported_claims:
        payload["unsupported_claims"] = unsupported_claims
        payload["instructions"].append(_UNSUPPORTED_CLAIMS_INSTRUCTION)


SYSTEM = (
    "You produce a SUGGESTED rewrite of a flagged paragraph for the author to review and edit. The "
    "author sees your changes in a before/after diff, so adding new content to fix the problem is "
    "expected and encouraged -- that is the mitigation. Your goal: lower AI-detection risk by making "
    "the writing specific, concrete, and human -- in the author's PLAIN, everyday voice, NOT an "
    "elevated, over-polished, ornamented register (avoid sweeping moral generalisations, clichéd metaphors, and a "
    "smooth balanced cadence) -- while staying in the same subject and tone as the source. Where the "
    "paragraph is generic or lacks a concrete anchor, ADD grounding AS THE "
    "AUTHOR WOULD -- you are the author's proxy. CHOOSE THE GROUNDING MODE THAT FITS HOW THE CLAIM "
    "IS GROUNDED; do NOT default to first-person. Read each claim's basis, then ground it the fitting way:\n"
    "- The author's OWN first-hand involvement (the draft signals they did / saw / built / taught it): "
    "first-person observation -- 'In my own <setting>, I have repeatedly seen <the concrete particular> "
    "...'. First-person is ONE strong mode; use it where it genuinely fits. But NEVER attach a "
    "first-person frame ('in my work', 'a client of mine', 'I tracked', 'my own survey') to a claim the "
    "author drew from headlines, reports, or common knowledge -- that invents experience they never had.\n"
    "- A figure or claim from HEADLINES / reports / COMMON KNOWLEDGE: attribute and qualify the source "
    "('widely reported that ...', 'the cited 3.3% figure suggests ...') and make the author's OWN "
    "reasoning or analysis explicit -- do NOT restate it as personal experience.\n"
    "- A general UNSOURCED assertion: ground it in a concrete mechanism, a condition (when / after / if), "
    "a decision someone faces, or the consequence that follows.\n"
    "- A prediction or opinion: surface the reasoning behind it, not a fabricated anecdote.\n"
    "Put a CONCRETE particular INSIDE whichever mode you choose, drawn from the AUTHOR'S OWN subject and "
    "context, NEVER a baked-in topic; a bare frame with no concrete particular is weak. The illustrative "
    "specifics (hedged figures like 'about a third', example scenarios, attributed sources) show the shape "
    "of a real anchor; flag EACH in author_review_items for the author to confirm or replace, and do NOT "
    "present a SPECIFIC named real institution, real person, or an exact statistic as a verified fact. "
    "Preserve the author's actual ARGUMENT and meaning -- do NOT flip a balanced 'not only X "
    "but also Y' into 'Y over X', and do not drop the author's existing ideas. List what you added "
    "in author_review_items so the author can confirm or replace it. "
    'Return JSON only: {"rewrite": "...", "author_review_items": [{"added": "...", "why": "..."}]}.'
)


DIVERSIFIED_SYSTEM = (
    "You produce a SUGGESTED rewrite of a flagged paragraph for the author to review and edit. The "
    "author sees your changes in a before/after diff. Your goal is to lower AI-detection risk by "
    "making the writing specific, concrete, author-owned, and plain-spoken while preserving the "
    "source argument. Ground each broad claim in the shape that fits ITS BASIS -- match the grounding to "
    "how the author actually knows the claim. A claim the draft signals as FIRST-HAND (the author saw, "
    "tried, taught, or managed it) may use first-person observation. A claim drawn from REPORTS, "
    "HEADLINES, or COMMON KNOWLEDGE must instead ATTRIBUTE and qualify the source, make the author's own "
    "reasoning explicit, or use a condition / consequence / actor frame -- NEVER invent first-hand "
    "experience the author never had. Route diversity means changing the shape of that grounding beat -- "
    "observed process, local constraint, decision moment, source use, actor interaction, condition "
    "trigger, or risk consequence -- not replacing it with detached examples. First-person observation "
    "is ONE mode, valid only when the draft genuinely signals first-hand involvement; for reported or "
    "formal prose, use local actor / source / condition / risk framing without forcing first person. "
    "Figures or scale details are optional support inside an author-proxy beat; they must not "
    "be the paragraph spine. List added scenarios, observations, bridge wording, or figures in "
    "author_review_items. Do not present a specific named real institution, real person, citation, "
    "external fact, or exact statistic as verified unless it already appears in the source. "
    'Return JSON only: {"rewrite": "...", "author_review_items": [{"added": "...", "why": "..."}]}.'
)


__all__ = ["SYSTEM", "DIVERSIFIED_SYSTEM"]
