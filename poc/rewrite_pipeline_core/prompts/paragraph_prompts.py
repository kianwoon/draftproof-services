"""Paragraph-level prompt builders for rewrite mitigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import json
import re


@dataclass(frozen=True)
class ParagraphPromptDeps:
    ai_search_signal_brief: Callable[[dict], str]
    protected_anchor_brief_for_prompt: Callable[..., list[dict]]
    anchor_values_from_brief: Callable[[list[dict]], list[str]]
    anchor_lock_mapping: Callable[[list[str]], list[dict]]
    freeze_anchor_text: Callable[[str, list[dict]], str]
    freeze_anchor_payload: Callable[[Any, list[dict]], Any]
    restore_anchor_placeholders: Callable[[str, list[dict] | None], str]
    logical_paragraphs: Callable[[str], list[str]]
    join_logical_paragraphs: Callable[[list[str]], str]
    float_env: Callable[[str, float], float]
    text_word_count: Callable[[str], int]


def paragraph_component_prompt(
    target: dict,
    raw_json: dict,
    attempt_index: int,
    *,
    reference_ai=None,
    required_ai_drop: float | None = None,
    target_ai_score: float | None = None,
    candidate_count: int = 1,
    confirmed_author_anchors: str = "",
    deps: ParagraphPromptDeps,
) -> str:
    signal_brief = deps.ai_search_signal_brief(raw_json)
    drivers = target.get("drivers") or {}
    candidate_count = max(1, int(candidate_count or 1))
    protected_source = "\n\n".join(
        part for part in [
            target.get("previous_paragraph") or "",
            target.get("paragraph") or "",
            target.get("next_paragraph") or "",
        ]
        if part
    )
    protected_anchors = deps.protected_anchor_brief_for_prompt(protected_source)
    anchor_lock = deps.anchor_lock_mapping(deps.anchor_values_from_brief(protected_anchors))
    frozen_target = deps.freeze_anchor_text(target.get("paragraph") or "", anchor_lock)
    frozen_previous = deps.freeze_anchor_text(target.get("previous_paragraph") or "[none]", anchor_lock)
    frozen_next = deps.freeze_anchor_text(target.get("next_paragraph") or "[none]", anchor_lock)
    frozen_domain_anchors = deps.freeze_anchor_payload((target.get("domain_anchors") or [])[:16], anchor_lock)
    required_placeholders = [
        item["placeholder"]
        for item in anchor_lock
        if item.get("placeholder") and item["placeholder"] in frozen_target
    ]
    target_char_count = len(frozen_target)
    min_char_count = max(80, int(target_char_count * deps.float_env("DRAFTPROOF_PARAGRAPH_COMPONENT_MIN_CHAR_RATIO", 0.35)))
    max_char_count = max(
        min_char_count,
        int(target_char_count * deps.float_env("DRAFTPROOF_PARAGRAPH_COMPONENT_MAX_CHAR_RATIO", 1.25)),
    )
    if deps.text_word_count(frozen_target) >= 25:
        min_char_count = max(min_char_count, 90)
    min_word_count = 21 if deps.text_word_count(target.get("paragraph") or "") >= 25 else 0
    min_word_rule = (
        f"- Target replacement shape: more than {min_word_count - 1} words if possible, but scanner improvement overrides length.\n"
        if min_word_count
        else ""
    )
    return (
        "DraftProof paragraph-component AI mitigation.\n"
        "Rewrite only the target paragraph.\n"
        "Goal: improve the Human Contribution formula after this paragraph is patched back into the document.\n"
        "This is formula-driver repair, not generic paraphrasing.\n\n"
        f"Measured success condition: reference AI={reference_ai}, required drop>={required_ai_drop}, target AI<={target_ai_score}.\n"
        "The candidate will be rescanned; do not make a mild paraphrase.\n\n"
        "Direct Human Contribution formula drivers to protect:\n"
        "- Lower AI likelihood by removing predictable, polished sentence routes.\n"
        "- Lower rewrite smoothness by keeping natural unevenness, not by adding errors.\n"
        "- Lower outline-to-text expansion by compressing over-explained claims; do not add explanatory bulk.\n"
        "- Lower discourse regularity by avoiding neat claim -> explanation -> implication/conclusion cadence.\n"
        "- Lower semantic uniformity by varying sentence purpose, but do not add new facts.\n"
        "Hard fail patterns:\n"
        "- adding extra explanation to sound clearer\n"
        "- creating a more complete academic paragraph\n"
        "- using broad setup sentences before the actual point\n"
        "- ending with a tidy lesson, implication, or summary\n\n"
        f"{signal_brief}\n\n"
        + (f"{confirmed_author_anchors}\n\n" if confirmed_author_anchors else "")
        + f"Paragraph driver score: {target.get('score')}\n"
        f"Drivers: {json.dumps(drivers, ensure_ascii=False)}\n"
        "Target sentences from scan:\n"
        + "\n".join(f"- {s}" for s in (target.get("target_sentences") or [])[:5])
        + "\nProblem spans:\n"
        + "\n".join(f"- {s}" for s in (target.get("problem_spans") or [])[:10])
        + "\nDomain anchors already present nearby:\n"
        + ", ".join(str(a) for a in frozen_domain_anchors)
        + "\n\nPrevious paragraph context:\n"
        f"{frozen_previous}\n\n"
        "TARGET PARAGRAPH:\n"
        f"<TARGET_PARAGRAPH>\n{frozen_target}\n</TARGET_PARAGRAPH>\n\n"
        "Next paragraph context:\n"
        f"{frozen_next}\n\n"
        "Rewrite rules:\n"
        "- Preserve all citations, years, numbers, names, unit codes, and source references.\n"
        "- Preserve every protected anchor exactly, including quoted phrases.\n"
        f"- Protected anchors: {json.dumps(protected_anchors, ensure_ascii=False)[:2200]}\n"
        f"- Anchor placeholders required in replacement: {json.dumps(required_placeholders, ensure_ascii=False)}\n"
        "- If a protected quote is awkward, keep the quote exactly and rewrite around it.\n"
        "- Do not invent new evidence, sources, people, institutions, or events.\n"
        "- Prefer deletion, compression, and reordering over adding new sentences.\n"
        f"- Target replacement length: {min_char_count}-{max_char_count} characters after placeholders are restored. This is guidance, not a hard gate.\n"
        + min_word_rule
        +
        "- Keep the replacement near the original length or shorter unless an anchor must be preserved.\n"
        "- Break generic assertion flow: avoid broad claims unless tied to the local process.\n"
        "- Start from concrete action, participant behavior, source relation, or practical consequence before broad explanation.\n"
        "- Change paragraph architecture: reorder claim/example/source relation where meaning allows.\n"
        "- Convert generic claims into specific process observations using only anchors already present nearby.\n"
        "- Vary sentence length and clause order enough that this is not a synonym swap.\n"
        "- Change sentence openings and sentence routes. Do not polish with academic filler.\n"
        "- Keep author voice and first-person observation where it already exists.\n"
        "- Remove duplicate fragments if present inside the target paragraph.\n"
        "- Copy every required [[DP_ANCHOR_###]] placeholder exactly; the pipeline restores real anchors after generation.\n"
        f"- Batch attempt {attempt_index}: make each option materially different from generic rephrasing.\n\n"
        f"Return exactly {candidate_count} alternative replacement paragraphs using this exact format:\n"
        "<CANDIDATE_1>\nreplacement paragraph only\n</CANDIDATE_1>\n"
        "<CANDIDATE_2>\nreplacement paragraph only\n</CANDIDATE_2>\n"
        "...continue until the requested candidate count.\n"
        "Do not include commentary outside the candidate tags."
    )


def paragraph_generation_anchor_context(target: dict | None, *, deps: ParagraphPromptDeps) -> dict:
    target = target or {}
    protected_source = "\n\n".join(
        part for part in [
            target.get("previous_paragraph") or "",
            target.get("paragraph") or "",
            target.get("next_paragraph") or "",
        ]
        if part
    )
    protected_anchors = deps.protected_anchor_brief_for_prompt(protected_source)
    anchor_lock = deps.anchor_lock_mapping(deps.anchor_values_from_brief(protected_anchors))
    frozen_target = deps.freeze_anchor_text(target.get("paragraph") or "", anchor_lock)
    target_placeholders = [
        item["placeholder"]
        for item in anchor_lock
        if item.get("placeholder") and item["placeholder"] in frozen_target
    ]
    target_char_count = len(frozen_target)
    min_char_count = max(
        80,
        int(target_char_count * deps.float_env("DRAFTPROOF_PARAGRAPH_COMPONENT_MIN_CHAR_RATIO", 0.35)),
    )
    max_char_count = max(
        min_char_count,
        int(target_char_count * deps.float_env("DRAFTPROOF_PARAGRAPH_COMPONENT_MAX_CHAR_RATIO", 1.25)),
    )
    if deps.text_word_count(frozen_target) >= 25:
        min_char_count = max(min_char_count, 90)
    min_word_count = 21 if deps.text_word_count(target.get("paragraph") or "") >= 25 else 0
    return {
        "protected_anchors": protected_anchors,
        "anchor_lock": anchor_lock,
        "frozen_target": frozen_target,
        "frozen_previous": deps.freeze_anchor_text(target.get("previous_paragraph") or "[none]", anchor_lock),
        "frozen_next": deps.freeze_anchor_text(target.get("next_paragraph") or "[none]", anchor_lock),
        "frozen_domain_anchors": deps.freeze_anchor_payload((target.get("domain_anchors") or [])[:16], anchor_lock),
        "required_placeholders": target_placeholders,
        "min_char_count": min_char_count,
        "max_char_count": max_char_count,
        "min_word_count": min_word_count,
    }


def human_signal_amplification_prompt(
    target: dict,
    raw_json: dict,
    attempt_index: int,
    *,
    candidate_count: int = 3,
    confirmed_author_anchors: str = "",
    deps: ParagraphPromptDeps,
) -> str:
    role = str(target.get("role") or "mixed")
    anchor_context = paragraph_generation_anchor_context(target, deps=deps)
    min_word_rule = (
        f"Target replacement shape: more than {anchor_context['min_word_count'] - 1} words if possible, but scanner improvement overrides length.\n"
        if anchor_context["min_word_count"]
        else ""
    )
    operation = {
        "source_summary_heavy": "add a source-to-practice bridge",
        "generic_claim_heavy": "narrow the claim with one author-reasoning trace",
        "conclusion_template_risk": "remove summary cadence and add a reflective limitation",
        "technical_process_rich": "preserve the technical process and make only a micro-repair",
    }.get(role, "add one controlled author-reasoning bridge")
    role_rule = {
        "generic_claim_heavy": (
            "For this role, replace one broad claim with a bounded author judgement, concern, "
            "or reasoning trace that is already implied by the paragraph. Include exactly one "
            "plain author-reasoning phrase such as 'I would...' or 'the issue I would check...' "
            "only when it does not introduce a new fact, example, source, event, institution, "
            "statistic, or personal evidence."
        ),
        "source_summary_heavy": (
            "For this role, keep the source claim intact and add one sentence that explains how "
            "the source changes a practical decision already present nearby."
        ),
        "conclusion_template_risk": (
            "For this role, reduce the neat closing-summary shape and add one limitation, tension, "
            "or unresolved judgement already implied by the document."
        ),
    }.get(role, "Use one small author-reasoning move without expanding the evidence base.")
    return (
        "DraftProof HUMAN_SIGNAL_AMPLIFICATION_REPAIR.\n"
        "You are repairing one paragraph only.\n"
        f"Paragraph role: {role}.\n"
        f"Controlled operation: {operation}.\n\n"
        f"Role-specific rule: {role_rule}\n\n"
        "Goal:\n"
        "Increase authentic author contribution using reasoning already implied by this paragraph, while improving the formula drivers.\n"
        "Do not increase AI Authorship, AI Transformation, review burden, or severity.\n\n"
        "Formula-driver constraints:\n"
        "- Do not add explanatory bulk; expansion is a failure.\n"
        "- Avoid neat claim -> explanation -> conclusion structure.\n"
        "- Prefer one compressed sentence, one uneven sentence, or one deleted generic sentence over a smoother rewrite.\n"
        "- Do not end with a polished implication sentence.\n"
        "- Keep the paragraph close to the original length or shorter.\n\n"
        "Allowed:\n"
        "- connect a source claim to a practical decision already present in the context\n"
        "- add one limitation or condition already implied by the paragraph\n"
        "- add one author judgement or reasoning trace if it changes no factual claim\n"
        "- make a claim more specific using existing local context\n"
        "- reduce template conclusion cadence\n"
        "- vary sentence purpose without making the paragraph more polished\n\n"
        "Forbidden:\n"
        "- invent new evidence, citations, dates, names, people, institutions, examples, or statistics\n"
        "- add a new source\n"
        "- rewrite the full paragraph into a smoother academic paragraph\n"
        "- use generic connectors such as Furthermore, Moreover, Additionally, This highlights, This underscores, In conclusion\n"
        "- change or remove citations, years, numbers, unit codes, source names, or quoted text\n\n"
        "Acceptance gate used by the scanner:\n"
        "- Human Contribution must increase by at least 2\n"
        "- AI Authorship must not increase\n"
        "- AI Transformation must not increase\n"
        "- review burden and weighted severity must not increase\n"
        "- semantic drift and anchor loss must be false\n\n"
        + (f"{confirmed_author_anchors}\n\n" if confirmed_author_anchors else "")
        + f"Drivers: {json.dumps(target.get('drivers') or {}, ensure_ascii=False)}\n"
        f"Protected anchors: {json.dumps(anchor_context['protected_anchors'], ensure_ascii=False)[:2200]}\n"
        f"Anchor placeholders required in replacement: {json.dumps(anchor_context['required_placeholders'], ensure_ascii=False)}\n"
        f"Target replacement length: {anchor_context['min_char_count']}-{anchor_context['max_char_count']} characters after placeholders are restored. This is guidance, not a hard gate.\n"
        + min_word_rule
        +
        "Copy every required [[DP_ANCHOR_###]] placeholder exactly; the pipeline restores real anchors after generation.\n"
        "Domain anchors already present nearby:\n"
        + ", ".join(str(a) for a in anchor_context["frozen_domain_anchors"])
        + "\n\nPrevious paragraph context:\n"
        f"{anchor_context['frozen_previous']}\n\n"
        "TARGET PARAGRAPH:\n"
        f"<TARGET_PARAGRAPH>\n{anchor_context['frozen_target']}\n</TARGET_PARAGRAPH>\n\n"
        "Next paragraph context:\n"
        f"{anchor_context['frozen_next']}\n\n"
        f"Attempt {attempt_index}: return exactly {candidate_count} alternatives using this format:\n"
        "<CANDIDATE_1>\nreplacement paragraph only\n</CANDIDATE_1>\n"
        "<CANDIDATE_2>\nreplacement paragraph only\n</CANDIDATE_2>\n"
        "...continue until the requested candidate count.\n"
        "No commentary outside tags."
    )


def author_reasoning_amplification_prompt(
    target: dict,
    raw_json: dict,
    attempt_index: int,
    *,
    candidate_count: int = 3,
    deps: ParagraphPromptDeps,
) -> str:
    role = str(target.get("role") or "mixed")
    anchor_context = paragraph_generation_anchor_context(target, deps=deps)
    min_word_rule = (
        f"Target replacement shape: more than {anchor_context['min_word_count'] - 1} words if possible, but scanner improvement overrides length.\n"
        if anchor_context["min_word_count"]
        else ""
    )
    return (
        "DraftProof AUTHOR_REASONING_AMPLIFICATION_REPAIR.\n"
        "You are repairing one paragraph only.\n"
        f"Paragraph role: {role}.\n\n"
        "Purpose:\n"
        "Increase Human Contribution using author reasoning that is already implied by the paragraph, "
        "without adding author evidence, lived experience, sources, dates, people, statistics, or new events.\n\n"
        "This is not evidence insertion. This is reasoning-shape repair.\n\n"
        "Formula-driver constraints:\n"
        "- Do not expand the paragraph to explain more.\n"
        "- Do not make the paragraph more complete, balanced, or academic.\n"
        "- Reduce neat discourse shape: avoid claim -> explanation -> conclusion.\n"
        "- Prefer compression, deletion of broad setup, and rougher sequencing.\n"
        "- Keep length close to the original or shorter.\n\n"
        "Allowed operations, choose one per candidate:\n"
        "- narrow one broad claim into a defensible condition\n"
        "- add one judgement about what the author would check, question, or prioritise\n"
        "- add one limitation or tension already implied by the paragraph\n"
        "- replace a neat summary sentence with a more specific reasoning consequence\n"
        "- remove an over-clean transition if the paragraph works without it\n\n"
        "Required texture:\n"
        "- keep the paragraph a little uneven\n"
        "- use one shorter sentence if natural\n"
        "- avoid balanced claim -> explanation -> conclusion rhythm\n"
        "- do not make the paragraph more polished\n\n"
        "Forbidden:\n"
        "- do not write 'in my class', 'I noticed', 'I saw', or any lived observation unless it already appears in the paragraph\n"
        "- do not invent examples, sources, citations, dates, people, places, institutions, statistics, or events\n"
        "- do not add generic connectors such as Furthermore, Moreover, Additionally, This highlights, This underscores, In conclusion\n"
        "- do not rewrite the whole paragraph into academic style\n"
        "- do not change citations, numbers, names, unit codes, quoted text, or chronology\n\n"
        "Acceptance gate used by the scanner:\n"
        "- Human Contribution should increase\n"
        "- AI Authorship must not increase\n"
        "- AI Transformation must not increase\n"
        "- findings, review burden, and weighted severity must not increase\n"
        "- semantic drift must be false\n\n"
        f"Drivers: {json.dumps(target.get('drivers') or {}, ensure_ascii=False)}\n"
        f"Protected anchors: {json.dumps(anchor_context['protected_anchors'], ensure_ascii=False)[:2200]}\n"
        f"Anchor placeholders required in replacement: {json.dumps(anchor_context['required_placeholders'], ensure_ascii=False)}\n"
        f"Target replacement length: {anchor_context['min_char_count']}-{anchor_context['max_char_count']} characters after placeholders are restored. This is guidance, not a hard gate.\n"
        + min_word_rule
        +
        "Copy every required [[DP_ANCHOR_###]] placeholder exactly; the pipeline restores real anchors after generation.\n"
        "Domain anchors already present nearby:\n"
        + ", ".join(str(a) for a in anchor_context["frozen_domain_anchors"])
        + "\n\nPrevious paragraph context:\n"
        f"{anchor_context['frozen_previous']}\n\n"
        "TARGET PARAGRAPH:\n"
        f"<TARGET_PARAGRAPH>\n{anchor_context['frozen_target']}\n</TARGET_PARAGRAPH>\n\n"
        "Next paragraph context:\n"
        f"{anchor_context['frozen_next']}\n\n"
        f"Attempt {attempt_index}: return exactly {candidate_count} alternatives using this format:\n"
        "<CANDIDATE_1>\nreplacement paragraph only\n</CANDIDATE_1>\n"
        "<CANDIDATE_2>\nreplacement paragraph only\n</CANDIDATE_2>\n"
        "...continue until the requested candidate count.\n"
        "No commentary outside tags."
    )


def extract_paragraph_component_candidates(output: str, limit: int) -> list[str]:
    text = str(output or "").strip()
    if not text:
        return []
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    tagged = re.findall(
        r"<CANDIDATE_(\d+)>\s*(.*?)\s*</CANDIDATE_\1>",
        text,
        flags=re.I | re.S,
    )
    if tagged:
        ordered = sorted(tagged, key=lambda item: int(item[0]))
        return [
            body.strip()
            for _, body in ordered[:max(1, limit)]
            if body.strip()
        ]
    marker_matches = re.findall(
        r"(?ims)^\s*(?:candidate|option)\s*\d+\s*[:.-]\s*(.*?)(?=^\s*(?:candidate|option)\s*\d+\s*[:.-]|\Z)",
        text,
    )
    if marker_matches:
        return [
            body.strip()
            for body in marker_matches[:max(1, limit)]
            if body.strip()
        ]
    return [text]


def clean_paragraph_component_candidate(
    candidate: str,
    original_paragraph: str,
    anchor_lock: list[dict] | None = None,
    *,
    deps: ParagraphPromptDeps,
) -> tuple[str, str]:
    text = str(candidate or "").strip()
    if not text:
        return "", "empty_candidate"
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"^(?:replacement|rewritten)\s+paragraph\s*:\s*", "", text, flags=re.I).strip()
    missing_placeholders = [
        item.get("placeholder")
        for item in (anchor_lock or [])
        if item.get("placeholder")
        and item["placeholder"] in deps.freeze_anchor_text(original_paragraph, anchor_lock)
        and item["placeholder"] not in text
    ]
    if missing_placeholders:
        return "", "anchor_placeholder_lost:" + ",".join(str(item) for item in missing_placeholders)
    text = deps.restore_anchor_placeholders(text, anchor_lock)
    paragraphs = deps.logical_paragraphs(text)
    if not paragraphs:
        return "", "empty_candidate"
    text = " ".join(" ".join(p.split()) for p in paragraphs)
    if text == " ".join(str(original_paragraph or "").split()):
        return "", "unchanged_paragraph"
    return text, ""


def paragraph_anchor_lock(target: dict | None, *, deps: ParagraphPromptDeps) -> list[dict]:
    target = target or {}
    protected_source = "\n\n".join(
        part for part in [
            target.get("previous_paragraph") or "",
            target.get("paragraph") or "",
            target.get("next_paragraph") or "",
        ]
        if part
    )
    return deps.anchor_lock_mapping(
        deps.anchor_values_from_brief(deps.protected_anchor_brief_for_prompt(protected_source))
    )


def clean_source_sentence_candidate(candidate: str, original_sentence: str, *, deps: ParagraphPromptDeps) -> tuple[str, str]:
    text = str(candidate or "").strip()
    if not text:
        return "", "empty_candidate"
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"^(?:replacement|rewritten)\s+(?:sentence|window)\s*:\s*", "", text, flags=re.I).strip()
    paragraphs = deps.logical_paragraphs(text)
    if not paragraphs:
        return "", "empty_candidate"
    text = " ".join(" ".join(p.split()) for p in paragraphs)
    if text == " ".join(str(original_sentence or "").split()):
        return "", "unchanged_sentence"
    orig_len = max(1, len(str(original_sentence or "")))
    min_len = max(40, int(orig_len * 0.45))
    max_len = max(120, int(orig_len * 2.20))
    if len(text) < min_len:
        return "", f"sentence_window_too_short {len(text)}<{min_len}"
    if len(text) > max_len:
        return "", f"sentence_window_too_long {len(text)}>{max_len}"
    return text, ""


def splice_paragraph(text: str, paragraph_index: int, replacement: str, *, deps: ParagraphPromptDeps) -> str:
    paragraphs = deps.logical_paragraphs(text)
    if paragraph_index < 0 or paragraph_index >= len(paragraphs):
        return text
    paragraphs[paragraph_index] = replacement.strip()
    return deps.join_logical_paragraphs(paragraphs)
