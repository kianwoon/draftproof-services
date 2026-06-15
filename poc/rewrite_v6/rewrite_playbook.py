"""Domain-neutral before->after rewrite playbook (from production research).

The structural levers (vary openings, cut hedging, vary length, ground claims) are validated to
drop the graded score 71->43 -- but the gpt-oss writer does not execute them from abstract rules
(measured flat at ~51). This playbook gives the writer the SHAPE of each fix, selected per paragraph
by its findings and text patterns, so the model can apply the transformation rather than guess.

DraftProof must cope with ANY document type (education, science, business, law, ...), so every
`before`/`better` here is DOMAIN-NEUTRAL: it names the transformation shape only and never any
subject-matter content. The content-specific concreteness comes from the per-paragraph scanner
diagnosis, not from this file. No hardcoded topics.

Each entry: id, the plain rule, a `before` shape, and a `better` move. `tags` triggers the entry
when a matching finding tag is present; `pattern` triggers it when the regex matches the source.
"""

from __future__ import annotations

import re
from typing import Any

# (id, rule, before, better, trigger_tags, trigger_pattern) -- before/better are SHAPE only, never
# topic-specific. The scanner diagnosis supplies the content; the playbook supplies the move.
_PLAYBOOK: list[dict[str, Any]] = [
    {"id": "generic_opening", "rule": "Start with the pressure, not the topic.",
     "before": "A broad topic announcement (e.g. 'The field of X is changing').",
     "better": "Open on the specific tension, problem, or action the paragraph is really about.",
     "tags": ["generic_opening", "predictable_start"], "pattern": r"^\s*(today'?s|in today'?s|in the (modern|current) world|in recent years)\b"},
    {"id": "broad_claim", "rule": "Narrow the scope.",
     "before": "A sweeping claim about a whole field or 'the world'.",
     "better": "Pin it to a specific actor, case, setting, or condition.",
     "tags": ["broad_claim"], "pattern": None},
    {"id": "predictable_start", "rule": "Avoid obvious opener words; begin with the actual tension or action.",
     "before": "An over-used opener ('In today's world', 'Nowadays').",
     "better": "Begin with the concrete tension or action itself.",
     "tags": ["predictable_start"], "pattern": r"^\s*(in (today'?s|the modern) world|nowadays|these days)\b"},
    {"id": "template_contrast", "rule": "Break the past-vs-present rhythm with a concrete shift.",
     "before": "A stock time contrast ('In the past... Today...').",
     "better": "Name the concrete shift instead of the time frame.",
     "tags": [], "pattern": r"\bin the past\b|\bno longer\b.*\btoday\b|\btoday\b.*\bin the past\b"},
    {"id": "packed_list", "rule": "Split the list into jobs; give each item a distinct role.",
     "before": "Three or more items packed into one comma list.",
     "better": "Give the key items their own clauses, each doing a distinct job.",
     "tags": ["packed_list"], "pattern": None},
    {"id": "sentence_overload", "rule": "One sentence, one main job; break long sentences into 2-3 different shapes.",
     "before": "One sentence carrying 4-5 separate ideas at once.",
     "better": "Break it into two or three sentences, each a different shape.",
     "tags": ["sentence_overload"], "pattern": None},
    {"id": "flat_rhythm", "rule": "Vary sentence length: mix short, medium, and long.",
     "before": "Several sentences of near-identical length.",
     "better": "Follow a long, specific sentence with a short, blunt one.",
     "tags": ["paragraph_rhythm"], "pattern": None},
    {"id": "ai_smoothness", "rule": "Add controlled friction: a grounded clause, exception, or real constraint.",
     "before": "Everything flows too neatly, with no friction.",
     "better": "Insert a concrete exception, limit, or constraint that interrupts the smooth flow.",
     "tags": ["semantic_bridge_gap"], "pattern": None},
    {"id": "context_anchor_gap", "rule": "Add who / where / when.",
     "before": "An unanchored claim with no actor, setting, or moment.",
     "better": "Ground it in a specific actor, setting, or moment.",
     "tags": ["context_anchor_gap"], "pattern": None},
    {"id": "abstract_noun_stack", "rule": "Turn abstract nouns into actions (who does what).",
     "before": "A pile of abstract nouns ('X concerns emerge').",
     "better": "Convert to actor-and-action: who does what, and to whom.",
     "tags": ["abstract_noun_stack"], "pattern": None},
    # NO-HARDCODE: removed the `generic_benefit_risk` entry -- its trigger keyed on baked benefit/
    # risk domain phrases ("opportunities and risks", "benefits and challenges/drawbacks"), which
    # routed any benefit/risk-flavoured paragraph to a pros/cons rewrite shape. The STRUCTURAL
    # balanced-pair shape it also matched ("both X and Y") is already covered by `over_balanced`.
    {"id": "predictable_connector", "rule": "Avoid overused transitions; use natural continuation.",
     "before": "An over-used transition ('Furthermore', 'Moreover', 'In conclusion').",
     "better": "State the next concrete point directly, without the connector.",
     "tags": ["transition_stack"], "pattern": r"\b(furthermore|moreover|in conclusion|additionally|consequently)\b"},
    {"id": "neat_final_sentence", "rule": "Don't wrap perfectly; end on a specific implication or unresolved pressure.",
     "before": "A tidy moral wrap-up ('Therefore, X must adapt').",
     "better": "End on a specific consequence or an unresolved tension.",
     "tags": ["generic_conclusion"], "pattern": r"\b(therefore|thus|in conclusion|overall|ultimately)\b"},
    {"id": "repeated_paragraph_shape", "rule": "Change the paragraph's job order.",
     "before": "topic -> explain -> example -> conclusion.",
     "better": "Vary the job order, e.g. example -> tension -> explanation -> implication.",
     "tags": ["repeated_sentence_frame", "paragraph_rhythm"], "pattern": None},
    {"id": "over_polished", "rule": "Use plainer words.",
     "before": "Inflated phrasing ('facilitate', 'utilize', 'leverage').",
     "better": "Swap in plainer, more direct words.",
     "tags": ["over_formal_synonym"], "pattern": r"\b(facilitate|utilis[ez]e|leverage|optimis[ez]e|enhance|foster)\b"},
    {"id": "weak_human_voice", "rule": "Add judgment.",
     "before": "A flat statement of importance ('This is important').",
     "better": "Say why it matters, in concrete terms specific to this point.",
     "tags": [], "pattern": r"\b(this is (important|significant|crucial)|it is important to note)\b"},
    {"id": "same_openings", "rule": "Rotate subject, action, condition, and consequence across sentence openings.",
     "before": "Consecutive sentences starting the same way.",
     "better": "Open one sentence on the actor, the next on the action, the next on the consequence.",
     "tags": ["repeated_opening"], "pattern": None},
    {"id": "over_balanced", "rule": "Vary the phrasing so contrasts aren't all the same symmetrical template.",
     "before": "Every contrast uses the same symmetrical template.",
     "better": "Vary how contrasts are expressed -- but keep the original balance and meaning intact.",
     "tags": [], "pattern": r"\bnot only\b.*\bbut also\b|\bboth\b.{0,40}\band\b"},
    {"id": "topk_route", "rule": "Change the route, not the synonyms.",
     "before": "Predictable wording swapped only for synonyms.",
     "better": "Change the sentence route and structure, not just the individual words.",
     "tags": ["review_predictability", "predictability", "ai_likelihood", "topk_pattern"], "pattern": None},
    {"id": "generic_conclusion", "rule": "Close with the consequence, not a moral.",
     "before": "A generic closing ('X should prepare for the future').",
     "better": "Close with the specific consequence that follows from this paragraph.",
     "tags": ["generic_conclusion"], "pattern": None},
    {"id": "no_author_anchor", "rule": "Add a lived, reviewable detail.",
     "before": "An abstract statement with no observable, concrete detail.",
     "better": "Add a concrete, observable detail or scenario the author can confirm or replace.",
     "tags": ["author_anchor_gap"], "pattern": None},
]

# Core structural moves always worth showing -- the dominant graded levers.
_ALWAYS_ON = {"flat_rhythm", "same_openings", "ai_smoothness"}


def playbook_entries(finding_tags: list[str], source_text: str, *, limit: int = 7) -> list[dict[str, str]]:
    """Select the most relevant before->after examples for a paragraph (tags + text patterns)."""
    tagset = {str(tag).strip() for tag in (finding_tags or []) if str(tag).strip()}
    text = str(source_text or "")
    selected: list[dict[str, str]] = []
    seen: set[str] = set()

    def _take(entry: dict[str, Any]) -> None:
        if entry["id"] in seen:
            return
        seen.add(entry["id"])
        selected.append({"problem": entry["id"], "rule": entry["rule"], "before": entry["before"], "better": entry["better"]})

    for entry in _PLAYBOOK:
        if tagset & set(entry.get("tags") or []):
            _take(entry)
        elif entry.get("pattern") and re.search(entry["pattern"], text, flags=re.I | re.M):
            _take(entry)
    for entry in _PLAYBOOK:
        if entry["id"] in _ALWAYS_ON:
            _take(entry)
    return selected[:limit]
