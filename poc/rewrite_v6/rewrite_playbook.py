"""Concrete before->after rewrite playbook (from production research).

The structural levers (vary openings, cut hedging, vary length, ground claims) are validated to
drop the graded score 71->43 -- but the gpt-oss writer does not execute them from abstract rules
(measured flat at ~51). This playbook gives the writer CONCRETE before->after examples, selected
per paragraph by its findings and text patterns, so the model can copy the transformation rather
than guess at it.

Each entry: id, the plain rule, a `before` pattern, and a `better` move. `tags` triggers the entry
when a matching finding tag is present; `pattern` triggers it when the regex matches the source.
"""

from __future__ import annotations

import re
from typing import Any

# (id, rule, before, better, trigger_tags, trigger_pattern)
_PLAYBOOK: list[dict[str, Any]] = [
    {"id": "generic_opening", "rule": "Start with the pressure, not the topic.",
     "before": "Today's education system is changing...",
     "better": "Schools now face a harder problem: students learn from more places than teachers can fully control.",
     "tags": ["generic_opening", "predictable_start"], "pattern": r"^\s*(today'?s|in today'?s|in the (modern|current) world|in recent years)\b"},
    {"id": "broad_claim", "rule": "Narrow the scope.",
     "before": "Technology has changed education.",
     "better": "AI tools have changed how students draft, revise, and prove their own learning.",
     "tags": ["broad_claim"], "pattern": None},
    {"id": "predictable_start", "rule": "Avoid obvious opener words; begin with the actual tension or action.",
     "before": "In today's world...",
     "better": "Begin with the concrete tension or action itself.",
     "tags": ["predictable_start"], "pattern": r"^\s*(in (today'?s|the modern) world|nowadays|these days)\b"},
    {"id": "template_contrast", "rule": "Break the past-vs-present rhythm with a concrete shift.",
     "before": "In the past... Today...",
     "better": "The old model assumed knowledge came from fixed sources; that assumption no longer holds.",
     "tags": [], "pattern": r"\bin the past\b|\bno longer\b.*\btoday\b|\btoday\b.*\bin the past\b"},
    {"id": "packed_list", "rule": "Split the list into jobs; give each item a distinct role.",
     "before": "drafts, reflection, discussion, feedback, improvement",
     "better": "Drafts show the first attempt. Reflection shows what changed. Feedback shows whether learning happened.",
     "tags": ["packed_list"], "pattern": None},
    {"id": "sentence_overload", "rule": "One sentence, one main job; break long sentences into 2-3 different shapes.",
     "before": "A long sentence carrying 4-5 ideas at once.",
     "better": "Break it into two or three sentences, each a different shape.",
     "tags": ["sentence_overload"], "pattern": None},
    {"id": "flat_rhythm", "rule": "Vary sentence length: mix short, medium, and long.",
     "before": "20 words, 22 words, 21 words.",
     "better": "Follow a long, specific sentence with a short, blunt one.",
     "tags": ["paragraph_rhythm"], "pattern": None},
    {"id": "ai_smoothness", "rule": "Add controlled friction: a grounded clause, exception, or real constraint.",
     "before": "Everything flows too neatly.",
     "better": "Insert a concrete exception or limit that interrupts the smooth flow.",
     "tags": ["semantic_bridge_gap"], "pattern": None},
    {"id": "context_anchor_gap", "rule": "Add who / where / when.",
     "before": "Students use AI tools.",
     "better": "A student drafting an essay may reach for AI to shape the structure before they fully understand the argument.",
     "tags": ["context_anchor_gap"], "pattern": None},
    {"id": "abstract_noun_stack", "rule": "Turn abstract nouns into actions (who does what).",
     "before": "learning integrity concerns emerge",
     "better": "Teachers cannot tell whether the student actually understood the work.",
     "tags": ["abstract_noun_stack"], "pattern": None},
    {"id": "generic_benefit_risk", "rule": "Replace 'opportunities and risks' with the concrete trade-off.",
     "before": "AI creates opportunities and risks.",
     "better": "AI can explain a hard topic in seconds, but it can also hide whether the student did the thinking.",
     "tags": [], "pattern": r"\b(opportunities and risks|benefits and (risks|challenges|drawbacks)|both .{0,30}\band\b)\b"},
    {"id": "predictable_connector", "rule": "Avoid overused transitions; use natural continuation.",
     "before": "Furthermore / Moreover / In conclusion...",
     "better": "That creates a second problem.",
     "tags": ["transition_stack"], "pattern": r"\b(furthermore|moreover|in conclusion|additionally|consequently)\b"},
    {"id": "neat_final_sentence", "rule": "Don't wrap perfectly; end on a specific implication or unresolved pressure.",
     "before": "Therefore, education must adapt.",
     "better": "Which leaves one question schools still cannot answer.",
     "tags": ["generic_conclusion"], "pattern": r"\b(therefore|thus|in conclusion|overall|ultimately)\b"},
    {"id": "repeated_paragraph_shape", "rule": "Change the paragraph's job order.",
     "before": "topic -> explain -> example -> conclusion",
     "better": "example -> tension -> explanation -> implication",
     "tags": ["repeated_sentence_frame", "paragraph_rhythm"], "pattern": None},
    {"id": "over_polished", "rule": "Use plainer words.",
     "before": "facilitate personalised learning outcomes",
     "better": "help students get support at their own pace.",
     "tags": ["over_formal_synonym"], "pattern": r"\b(facilitate|utilis[ez]e|leverage|optimis[ez]e|enhance|foster)\b"},
    {"id": "weak_human_voice", "rule": "Add judgment.",
     "before": "This is important.",
     "better": "That matters because the final answer alone no longer proves learning.",
     "tags": [], "pattern": r"\b(this is (important|significant|crucial)|it is important to note)\b"},
    {"id": "same_openings", "rule": "Rotate subject, action, condition, and consequence across sentence openings.",
     "before": "Students... Students... Students...",
     "better": "Open one sentence on the actor, the next on the action, the next on the consequence.",
     "tags": ["repeated_opening"], "pattern": None},
    {"id": "over_balanced", "rule": "Avoid symmetrical phrasing; make the trade-off concrete and uneven.",
     "before": "both benefits and risks",
     "better": "The gain is real but small; the cost lands later and is harder to see.",
     "tags": [], "pattern": r"\bnot only\b.*\bbut also\b|\bboth\b.{0,40}\band\b"},
    {"id": "topk_route", "rule": "Change the route, not the synonyms.",
     "before": "Assessment should include not only final answers...",
     "better": "To judge real learning, look at the answer and at how the student reached it.",
     "tags": ["review_predictability", "predictability", "ai_likelihood", "topk_pattern"], "pattern": None},
    {"id": "generic_conclusion", "rule": "Close with the consequence, not a moral.",
     "before": "Education should prepare students for the future.",
     "better": "The harder task is teaching students to defend their thinking when answers are easy to generate.",
     "tags": ["generic_conclusion"], "pattern": None},
    {"id": "no_author_anchor", "rule": "Add a lived, reviewable detail.",
     "before": "Teachers guide students.",
     "better": "In a classroom, the teacher sees the hesitation, the scratched-out draft, and the correction.",
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
