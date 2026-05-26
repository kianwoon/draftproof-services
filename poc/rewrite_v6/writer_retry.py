from __future__ import annotations

import json
from typing import Any

from .json_io import parse_json
from .plan import Plan
from .prose_quality import has_fragment_or_trace_sentences
from .scan import scan_text
from .selector_diagnostics import selection_diagnostics
from .text import Paragraph
from .write import ChatClient, Variant, build_prompt, parse_variants


def write_retry_variants(
    paragraph: Paragraph,
    plan: Plan,
    *,
    client: ChatClient,
    rejected_variants: list[Variant],
) -> list[Variant]:
    feedback = _retry_feedback(paragraph, rejected_variants)
    if not feedback["rejected_candidates"]:
        return []
    prompt = (
        build_prompt(paragraph, plan, variant_focus={
            "id": "retry_v1",
            "mode": "defect_repair_generation",
            "route": "repair the rejected candidate defects while preserving source coverage",
            "distinctive_obligation": "Use the defect_feedback rows as mandatory corrections; do not repeat rejected malformed sentence shapes.",
        })
        + "\n\nDefect feedback from rejected candidates:\n"
        + json.dumps(feedback, ensure_ascii=False, indent=2)
    )
    response = client.chat(
        prompt,
        system=(
            "Return valid JSON only with a variants array containing retry_v1. "
            "Repair the listed defects directly. Every final sentence must be grammatical, source-grounded, and complete."
        ),
        temperature=0.08,
        top_p=0.7,
        max_tokens=None,
        response_format={"type": "json_object"},
    )
    try:
        return parse_variants(parse_json(getattr(response, "raw_content", "") or response.content))
    except (Exception, ValueError):
        return []


def _retry_feedback(paragraph: Paragraph, variants: list[Variant]) -> dict[str, Any]:
    source_scan = scan_text(paragraph.text)
    diagnostics = {row["variant_id"]: row for row in selection_diagnostics(variants, paragraph)}
    rows: list[dict[str, Any]] = []
    for variant in variants:
        if variant.source == "source_preserved":
            continue
        candidate_scan = scan_text(variant.text)
        defects: list[str] = []
        if has_fragment_or_trace_sentences(variant.text):
            defects.append("malformed_fragment_or_trace_sentence")
        if candidate_scan.scores["finding_count"] >= source_scan.scores["finding_count"]:
            defects.append("no_finding_count_drop")
        if candidate_scan.scores["mean_sentence_shape_risk"] >= source_scan.scores["mean_sentence_shape_risk"]:
            defects.append("no_sentence_shape_risk_drop")
        defects.extend(str(blocker) for blocker in diagnostics.get(variant.id, {}).get("blockers", []))
        defects = list(dict.fromkeys(defects))
        if not defects:
            continue
        rows.append({
            "variant_id": variant.id,
            "defects": defects,
            "candidate_findings": int(candidate_scan.scores["finding_count"]),
            "source_findings": int(source_scan.scores["finding_count"]),
            "candidate_mean_risk": candidate_scan.scores["mean_sentence_shape_risk"],
            "source_mean_risk": source_scan.scores["mean_sentence_shape_risk"],
            "rejected_text_excerpt": variant.text[:700],
            "repair_instruction": (
                "Generate a new candidate from source meaning. Do not lightly edit this rejected text. "
                "Keep source coverage, but rebuild malformed or non-moving sentences as complete sentence rows."
            ),
        })
    return {
        "source_paragraph_id": paragraph.id,
        "source_text_excerpt": paragraph.text[:900],
        "rejected_candidates": rows[:3],
        "mandatory_retry_rules": [
            "Do not return source-preserved wording.",
            "Do not include standalone connector or subordinate fragments.",
            "Do not split a condition from its main clause.",
            "Reduce finding count or sentence-shape risk while preserving source coverage.",
        ],
    }
