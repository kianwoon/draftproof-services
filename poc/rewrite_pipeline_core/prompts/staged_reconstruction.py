from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class StagedReconstructionPromptDeps:
    text_word_count: Callable[[str], int]
    float_env: Callable[[str, float], float]
    anchor_lock_mapping: Callable[[list[str]], list[dict]]
    freeze_anchor_payload: Callable[[dict, list[dict]], dict]


def staged_generation_section_plan(
    context_ledger: dict,
    *,
    max_sections: int | None = None,
    deps: StagedReconstructionPromptDeps,
) -> list[dict]:
    """Create bounded section plans so the LLM never receives the full document."""
    if max_sections is None:
        try:
            max_sections = int(os.environ.get("DRAFTPROOF_STAGED_REGENERATION_SECTIONS", "6"))
        except ValueError:
            max_sections = 6
    max_sections = max(1, max_sections)
    handoff = (context_ledger or {}).get("generation_handoff") or {}
    handoff_units = [
        unit for unit in handoff.get("section_generation_units") or []
        if isinstance(unit, dict) and unit.get("heading")
    ]

    def section_required_anchors(unit: dict) -> list[str]:
        anchors: list[str] = []
        for value in unit.get("must_preserve_anchors") or []:
            text = str(value or "").strip()
            if text and text not in anchors:
                anchors.append(text)
        for point in unit.get("meaning_inventory") or []:
            if not isinstance(point, dict):
                continue
            for value in point.get("anchors") or []:
                text = str(value or "").strip()
                if text and text not in anchors:
                    anchors.append(text)
        return anchors

    if handoff_units:
        title = ((handoff.get("document_profile") or {}).get("title") or "").strip()
        selected_units = handoff_units[:max_sections]
        if len(handoff_units) > max_sections:
            conclusion = next(
                (
                    unit for unit in reversed(handoff_units)
                    if re.search(r"\bconclusion\b|closing|final", str(unit.get("heading") or ""), re.I)
                ),
                None,
            )
            if conclusion and conclusion not in selected_units:
                selected_units = selected_units[:max(0, max_sections - 1)] + [conclusion]
        return [
            {
                "section_index": index,
                "section_id": unit.get("section_id"),
                "title": title,
                "heading": unit.get("heading"),
                "target_words": ((unit.get("target_words") or {}).get("ideal") or (unit.get("target_words") or {}).get("max") or 180),
                "target_word_band": unit.get("target_words") or {},
                "must_preserve_anchors": section_required_anchors(unit),
                "citation_keys_used": unit.get("citation_keys_used") or [],
                "claim_inventory_slice": unit.get("meaning_inventory") or [],
                "target_signal_slice": unit.get("detector_risks_to_reduce") or [],
                "paragraph_plan_slice": [{
                    "section_id": unit.get("section_id"),
                    "role": unit.get("role"),
                    "citation_keys_used": unit.get("citation_keys_used") or [],
                    "must_preserve_anchors": section_required_anchors(unit),
                    "generation_instruction": unit.get("generation_instruction") or {},
                }],
            }
            for index, unit in enumerate(selected_units, start=1)
        ]
    headings = [
        str(item).strip()
        for item in (context_ledger or {}).get("headings") or []
        if str(item).strip() and not re.match(r"^(?:references|reference list|bibliography|works cited)$", str(item).strip(), re.I)
    ]
    if not headings:
        roles = (context_ledger or {}).get("paragraph_roles") or []
        headings = [
            str(row.get("role") or f"Section {idx}").replace("_", " ").title()
            for idx, row in enumerate(roles[:max_sections], start=1)
            if isinstance(row, dict)
        ] or ["Context", "Main Reasoning", "Conclusion"]

    title = headings[0] if len(headings) > 1 else ""
    body_headings_all = headings[1:] if title else headings
    body_headings = body_headings_all[:max_sections]
    if len(body_headings_all) > max_sections:
        conclusion = next(
            (
                heading
                for heading in reversed(body_headings_all)
                if re.search(r"\bconclusion\b|closing|final", heading, re.I)
            ),
            "",
        )
        if conclusion and conclusion not in body_headings:
            body_headings = body_headings[:max(0, max_sections - 1)] + [conclusion]
    claims = [
        str(item).strip()
        for item in (context_ledger or {}).get("claim_inventory") or []
        if str(item).strip()
    ]
    target_segments = (context_ledger or {}).get("target_segment_signals") or []
    paragraph_plans = (context_ledger or {}).get("paragraph_plans") or []
    word_band = (context_ledger or {}).get("word_count_band") or {}
    reference_words = deps.text_word_count("\n".join((context_ledger or {}).get("reference_entries") or []))
    total_target = int((word_band.get("min_words") or 0) + (word_band.get("max_words") or 0)) // 2
    if total_target <= 0:
        total_target = max(900, len(claims) * 55)
    body_target = max(450, total_target - reference_words - deps.text_word_count(title))
    target_inflation = deps.float_env("DRAFTPROOF_STAGED_SECTION_TARGET_INFLATION", 1.18)
    per_section = max(120, int((body_target * target_inflation) / max(1, len(body_headings))))

    plans: list[dict] = []
    claim_window = max(2, math.ceil(len(claims) / max(1, len(body_headings))))
    segment_window = max(1, math.ceil(len(target_segments) / max(1, len(body_headings)))) if target_segments else 1
    for index, heading in enumerate(body_headings, start=1):
        claim_start = (index - 1) * claim_window
        segment_start = (index - 1) * segment_window
        plans.append({
            "section_index": index,
            "title": title,
            "heading": heading,
            "target_words": per_section,
            "claim_inventory_slice": claims[claim_start:claim_start + claim_window],
            "target_signal_slice": target_segments[segment_start:segment_start + segment_window],
            "paragraph_plan_slice": paragraph_plans[max(0, index - 1):index + 2],
        })
    return plans


def staged_reconstruction_section_prompt(
    context_ledger: dict,
    gate_controls: dict,
    section_plan: dict,
    *,
    strategy: str,
    attempt_index: int,
    deps: StagedReconstructionPromptDeps,
) -> str:
    target_band = section_plan.get("target_word_band") or {}
    target_min = target_band.get("min")
    target_max = target_band.get("max")
    target_ideal = target_band.get("ideal") or section_plan.get("target_words")
    target_text = (
        f"between {target_min} and {target_max} words; aim for about {target_ideal} words"
        if isinstance(target_min, int) and isinstance(target_max, int)
        else f"about {section_plan.get('target_words')} words"
    )
    required_anchors = section_plan.get("must_preserve_anchors") or []
    allowed_citations = section_plan.get("citation_keys_used") or []
    handoff = (context_ledger or {}).get("generation_handoff") or {}
    all_reference_labels = []
    for ref in handoff.get("reference_register") or []:
        if not isinstance(ref, dict):
            continue
        label = str(ref.get("citation_key") or "").strip()
        if label and label not in all_reference_labels:
            all_reference_labels.append(label)
    disallowed_citations = [
        label for label in all_reference_labels
        if label not in allowed_citations
    ][:20]
    preserve_context = {
        "protected_facts": (context_ledger or {}).get("protected_facts") or [],
        "preservation_inventory": (context_ledger or {}).get("preservation_inventory") or {},
        "preserve_terms": (context_ledger or {}).get("preserve_terms") or [],
        "do_not_add": (context_ledger or {}).get("do_not_add") or [],
        "industry_baseline_focus": (context_ledger or {}).get("industry_baseline_focus") or {},
        "human_contribution_contract": (context_ledger or {}).get("human_contribution_contract") or {},
    }
    section_context = {
        "schema_version": "section_generation_context.v1",
        "section": section_plan,
        "preserve_context": preserve_context,
        "scanner_gate_feedback": gate_controls,
    }
    strategy_controls = []
    strategy_name = str(strategy or "").strip().lower()
    anchor_lock_mapping = deps.anchor_lock_mapping(required_anchors)
    prompt_section_context = (
        deps.freeze_anchor_payload(section_context, anchor_lock_mapping)
        if anchor_lock_mapping
        else section_context
    )
    prompt_required_anchors = (
        [item["placeholder"] for item in anchor_lock_mapping]
        if anchor_lock_mapping
        else required_anchors
    )
    if strategy_name == "plain_direct_voice_rebuild":
        strategy_controls = [
            "PLAIN_DIRECT_VOICE_REBUILD is active.",
            "Write like a plain submitted draft, not like a polished model answer.",
            "Use simple vocabulary and direct sentences. Repeating ordinary words is acceptable.",
            "Do not upgrade the essay into academic style. Do not add sophisticated connectors or balanced paragraph architecture.",
            "Keep some uneven development: one idea may be short, another may be a little over-explained, and not every link needs a transition.",
            "Avoid phrases such as rapidly evolving, crucial role, significant impact, furthermore, moreover, additionally, this highlights, and in conclusion.",
            "Do not add new facts, citations, examples, dates, institutions, or personal events.",
            "Preserve the same meaning and required anchors.",
        ]
    elif strategy_name == "human_gain_repair":
        strategy_controls = [
            "HUMAN_GAIN_REPAIR is active.",
            "Patch the section through controlled human-anchor amplification, not broad rewriting.",
            "Use only existing anchors, section-local context, and safe lived-observation phrasing licensed by the source stance.",
            "Do not invent new concrete details. If a concrete detail is not in required anchors or context ledger, leave it out.",
            "Prefer one small reasoning trace and one workshop/process anchor over generic academic expansion.",
            "Keep mild unevenness and rough edges; do not make every sentence equally polished.",
        ]
    elif strategy_name == "authorship_distribution_repair":
        strategy_controls = [
            "AUTHORSHIP_DISTRIBUTION_REPAIR is active.",
            "Primary target is lower AI Authorship, not prettier prose and not more explanation.",
            "Avoid the polished essay route. Do not write claim -> explanation -> implication repeatedly.",
            "Use distributional texture: one compressed sentence, one slightly longer causal sentence, one plain restart, and one under-explained but clear practical point.",
            "Break clean transition chains. Use plain moves such as 'But', 'So', 'The issue is', or no transition when the link is obvious.",
            "Reduce semantic uniformity by giving adjacent sentences different jobs: observation, limitation, consequence, then a narrow judgement.",
            "Do not add new facts, examples, dates, institutions, personal events, citations, or evidence.",
            "Do not add random errors, slang, typos, or artificial noise.",
        ]
    elif strategy_name == "authorship_texture_repair":
        strategy_controls = [
            "AUTHORSHIP_TEXTURE_REPAIR is active.",
            "Do not add more semantic human anchors as the main move; fix authorship texture.",
            "Preserve the same meaning points and required anchors while changing cadence and pacing.",
            "Use natural asymmetry: one short sentence, one longer practical sentence, and one delayed connection.",
            "Reduce clean transition logic. Avoid neat claim -> explanation -> implication paragraph routes.",
            "Vary information density without adding facts: compress one idea, leave one practical point less over-explained.",
            "Keep acceptable local friction, but do not add typos, grammar damage, fake randomness, or invented details.",
        ]
    elif strategy_name in {"low_smoothness_rebuild", "asymmetric_paragraph_route"}:
        strategy_controls = [
            "LOW_SMOOTHNESS_AUTHORSHIP_REPAIR is active.",
            "Keep meaning and anchors, but lower clean explanatory cadence.",
            "Prefer uneven paragraph movement over balanced academic flow.",
            "Use fewer connector phrases. Let some sentences sit next to each other without over-explaining the link.",
            "Compress generic claims instead of expanding them.",
            "Do not add fabricated grounding or decorative personal detail.",
        ]
    if anchor_lock_mapping:
        strategy_controls.append(
            "Anchor lock is active. Copy every [[DP_ANCHOR_###]] placeholder exactly; the pipeline restores real anchors after generation."
        )
    return (
        "DraftProof staged AI-Mitigation generation.\n"
        "Generate only this section body from the section context ledger. "
        "Do not output the section heading, references, labels, comments, markdown fences, or explanation.\n"
        "The original submitted prose is unavailable by design. Use only the structured context below.\n\n"
        f"Attempt: {attempt_index}. Strategy family: {strategy}.\n"
        f"Section heading owned by assembler: {section_plan.get('heading')}\n"
        f"Target length for this section body: {target_text}. This is guidance, not the final acceptance gate.\n"
        f"Required section anchors to preserve exactly; missing any one invalidates the output: {json.dumps(prompt_required_anchors, ensure_ascii=False)}\n"
        f"Allowed citation/source keys for this section: {json.dumps(allowed_citations, ensure_ascii=False)}\n"
        f"Disallowed citation/source keys for this section: {json.dumps(disallowed_citations, ensure_ascii=False)}\n\n"
        "Section context ledger:\n"
        f"{json.dumps(prompt_section_context, ensure_ascii=False)[:7000]}\n\n"
        "Generation controls:\n"
        f"- Word-count target: stay near {target_text}; scanner/gate quality is more important than exact length.\n"
        "- Write enough section body to survive cleanup that removes headings, repeated sentences, labels, and filler.\n"
        "- Do not repeat any full sentence or near-duplicate sentence; repeated sentences are removed before scoring and can make the candidate too short.\n"
        "- Preserve meaning, citations, years, names, numbers, quotes, source relations, and domain terms that are relevant to this section.\n"
        "- Use every required section anchor unless it is clearly a fragment; unit codes and named institutions are mandatory.\n"
        "- Do not mention disallowed source names, author groups, citations, frameworks, or evidence from other sections.\n"
        "- If allowed citation/source keys is empty, do not mention any source author, cited study, framework, or reference name.\n"
        "- Do not add new evidence, personal events, workplace observations, institutions, dates, statistics, sources, or citations.\n"
        "- Do not make a polished template essay paragraph. Use local reasoning, uneven sentence lengths, and section-specific causal links.\n"
        "- If the section context is thin, expand only by connecting the provided meaning points and anchors; do not invent facts.\n"
        + ("".join(f"- {item}\n" for item in strategy_controls) if strategy_controls else "")
        + "- Return prose only."
    )
