"""Prompt builders for targeted claim and texture repairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import json


@dataclass(frozen=True)
class TargetedRepairPromptDeps:
    blocker_scores: Callable[[dict | None], dict]
    logical_paragraphs: Callable[[str], list[str]]
    paragraph_component_targets: Callable[..., list[dict]]
    text_word_count: Callable[[str], int]
    word_count_band: Callable[..., dict]
    float_env: Callable[[str, float], float]
    protected_anchor_brief_for_prompt: Callable[..., list[dict]]
    sentence_texture_risk_map: Callable[..., list[dict]]
    safe_topk_calibrated_limit: Callable[[], float]
    ai_search_signal_brief: Callable[[dict], str]
    source_repair_brief: Callable[[str], str]


def claim_narrowing_repair_prompt(
    source_text: str,
    report_dict: dict | None,
    *,
    candidate_count: int = 2,
    deps: TargetedRepairPromptDeps,
) -> str:
    blockers = deps.blocker_scores(report_dict)
    paragraphs = deps.logical_paragraphs(source_text)
    target_blocks = []
    for target in deps.paragraph_component_targets(source_text, report_dict or {}, limit=8):
        paragraph = target.get("paragraph") or ""
        role = str(target.get("role") or "")
        drivers = target.get("drivers") or {}
        if role in {"generic_claim_heavy", "conclusion_template_risk", "mixed"}:
            target_blocks.append({
                "paragraph_index": target.get("index"),
                "role": role,
                "score": target.get("score"),
                "drivers": drivers,
                "preview": paragraph[:520],
            })
    if not target_blocks:
        target_blocks = [
            {"paragraph_index": i, "role": "candidate", "preview": p[:520]}
            for i, p in enumerate(paragraphs)
            if deps.text_word_count(p) >= 35
        ][:5]
    word_band = deps.word_count_band(
        source_text,
        variance=deps.float_env("DRAFTPROOF_CLAIM_NARROWING_WORD_VARIANCE", 0.25),
    )
    protected_anchors = deps.protected_anchor_brief_for_prompt(source_text)
    return (
        "DraftProof CLAIM_NARROWING_REPAIR.\n"
        "Do not humanize generically. Reduce unsupported_claim_risk and broad_claim_risk directly.\n\n"
        f"Current blocker scores: {json.dumps(blockers, ensure_ascii=False)}\n"
        f"Word-count band: source={word_band['source_word_count']}, min={word_band['min_words']}, max={word_band['max_words']}.\n\n"
        "Protected anchors. These exact spans must appear unchanged in every candidate, including quote text:\n"
        f"{json.dumps(protected_anchors, ensure_ascii=False)[:2600]}\n\n"
        "Target blocks:\n"
        f"{json.dumps(target_blocks, ensure_ascii=False)[:7000]}\n\n"
        "Required operations:\n"
        "- weaken absolute claims\n"
        "- add scope limits\n"
        "- convert universal claims into conditional or observed/context-bound claims\n"
        "- remove overreach\n"
        "- delete or compress unsupported broad claims when narrowing is not enough\n"
        "- preserve existing anchors, source relations, and meaning coverage\n\n"
        "Forbidden:\n"
        "- do not add new facts, citations, dates, names, statistics, examples, institutions, or author experiences\n"
        "- do not drop, paraphrase, normalize, or reword protected anchors. If a quote is awkward, keep the quote exactly and rewrite around it.\n"
        "- do not make the prose more polished or academic\n"
        "- do not use generic connectors such as Furthermore, Moreover, Additionally, This highlights, This underscores, In conclusion\n\n"
        "Acceptance target:\n"
        "- unsupported_claim_risk must drop\n"
        "- broad_claim_risk must drop\n"
        "- AI Authorship must not increase\n"
        "- AI Transformation must drop or stay safe\n"
        "- findings/review/severity must not regress\n\n"
        "SOURCE DOCUMENT:\n"
        f"<SOURCE_DOCUMENT>\n{source_text.strip()}\n</SOURCE_DOCUMENT>\n\n"
        f"Return exactly {max(1, int(candidate_count or 1))} complete document candidates using this exact format:\n"
        "<CANDIDATE_1>\ncomplete document only\n</CANDIDATE_1>\n"
        "<CANDIDATE_2>\ncomplete document only\n</CANDIDATE_2>\n"
        "...continue until the requested candidate count.\n"
        "No commentary outside tags."
    )


def topk_texture_repair_prompt(
    source_text: str,
    report_dict: dict | None,
    *,
    candidate_count: int = 2,
    deps: TargetedRepairPromptDeps,
) -> str:
    blockers = deps.blocker_scores(report_dict)
    risk_map = deps.sentence_texture_risk_map(source_text, report_dict, limit=10)
    target_sentences = [
        {
            "sentence_index": item.get("sentence_index"),
            "risk": item.get("risk"),
            "sentence": item.get("sentence"),
            "drivers": item.get("drivers"),
        }
        for item in risk_map[:8]
    ]
    word_band = deps.word_count_band(
        source_text,
        variance=deps.float_env("DRAFTPROOF_TOPK_TEXTURE_WORD_VARIANCE", 0.25),
    )
    protected_anchors = deps.protected_anchor_brief_for_prompt(source_text)
    safe_topk = deps.safe_topk_calibrated_limit()
    return (
        "DraftProof TOPK_TEXTURE_REPAIR.\n"
        "Reduce predictable phrasing after claim narrowing. Do not add facts.\n\n"
        f"Current blocker scores: {json.dumps(blockers, ensure_ascii=False)}\n"
        f"Word-count band: source={word_band['source_word_count']}, min={word_band['min_words']}, max={word_band['max_words']}.\n\n"
        "Protected anchors. These exact spans must appear unchanged in every candidate, including quote text:\n"
        f"{json.dumps(protected_anchors, ensure_ascii=False)[:2600]}\n\n"
        "Highest-risk sentence texture targets:\n"
        f"{json.dumps(target_sentences, ensure_ascii=False)[:5000]}\n\n"
        "Allowed operations:\n"
        "- replace generic sentence openings\n"
        "- break balanced sentence rhythm\n"
        "- remove polished connectors\n"
        "- vary sentence length lightly\n"
        "- use one or two short fragments where a full polished sentence is too predictable\n"
        "- move a qualifier or contrast to the front of a sentence when it preserves meaning\n"
        "- replace bland verbs such as is/has/plays/supports/shapes with more specific plain verbs already implied by the sentence\n"
        "- patch only high-risk sentences or adjacent clauses\n\n"
        f"If calibrated Top-k risk is {safe_topk:.0f} or higher, light polishing is not enough. Make visible route changes in the target sentences while keeping the same facts.\n\n"
        "Hard rule:\n"
        "- Do not add new facts, citations, statistics, examples, institutions, names, dates, or author experiences.\n"
        "- Preserve claim scope after narrowing.\n"
        "- Do not drop, paraphrase, normalize, or reword protected anchors. If a quote is awkward, keep the quote exactly and rewrite around it.\n"
        "- Do not rewrite into smoother academic prose.\n\n"
        "Acceptance target:\n"
        f"- calibrated Top-k risk must move below {safe_topk:.0f}; raw Top-k is diagnostic only\n"
        "- predictability should drop\n"
        "- AI Authorship must not increase\n"
        "- unsupported/broad claims must not regress\n\n"
        "SOURCE DOCUMENT:\n"
        f"<SOURCE_DOCUMENT>\n{source_text.strip()}\n</SOURCE_DOCUMENT>\n\n"
        f"Return exactly {max(1, int(candidate_count or 1))} complete document candidates using this exact format:\n"
        "<CANDIDATE_1>\ncomplete document only\n</CANDIDATE_1>\n"
        "<CANDIDATE_2>\ncomplete document only\n</CANDIDATE_2>\n"
        "...continue until the requested candidate count.\n"
        "No commentary outside tags."
    )


def ai_search_prompt(
    source_text: str,
    raw_json: dict,
    strategy: str,
    *,
    reference_ai=None,
    required_ai_drop: float | None = None,
    target_ai_score: float | None = None,
    confirmed_author_anchors: str = "",
    deps: TargetedRepairPromptDeps,
) -> str:
    signal_brief = deps.ai_search_signal_brief(raw_json)
    repair_brief = deps.source_repair_brief(source_text)
    protected_anchors = deps.protected_anchor_brief_for_prompt(source_text)
    strategy_lines = {
        "syntax_demolition": [
            "Strategy: syntax demolition.",
            "Break original sentence routes. Do not keep the same subject-verb-object path when meaning allows.",
            "Split some long balanced sentences and merge a few short neighboring sentences where natural.",
        ],
        "paragraph_resequence": [
            "Strategy: paragraph resequencing.",
            "Change how paragraphs arrive at their point. Start from a concrete action, mistake, observation, or consequence before the broad claim.",
            "Avoid the same explanatory order used in the source.",
        ],
        "plain_workshop_voice": [
            "Strategy: plain workshop voice.",
            "Make the draft sound like a knowledgeable person explaining work they have actually seen.",
            "Use concrete verbs and uneven sentence length. Keep useful roughness.",
        ],
        "review_marked_grounding": [
            "Strategy: review-marked grounding.",
            "Where the scan points to unsupported or generic claims, add clearly marked author-review material using [[REVIEW: ...]] brackets.",
            "The bracketed material must be framed as a place for the user to verify or replace, not as a fabricated fact.",
            "Use these marked additions to break generic claim flow with source/evidence prompts, concrete context prompts, or careful limitation prompts.",
        ],
        "source_bridge_rebuild": [
            "Strategy: source bridge rebuild.",
            "Do not leave historical or technical claims floating. Add source-bridge sentences only when they can be phrased as review prompts.",
            "Use [[REVIEW: add source here explaining ...]] where the source is missing, then reconnect the claim in plainer wording.",
        ],
        "claim_narrowing": [
            "Strategy: claim narrowing.",
            "Turn broad claims into context-limited claims. Add cautious wording where evidence is implied but not supplied.",
            "Do not add new sources or fabricated evidence.",
        ],
        "cadence_disruption": [
            "Strategy: cadence disruption.",
            "Break repeated clean essay cadence. Vary openings, sentence length, and paragraph rhythm.",
            "Avoid polished connector chains and abstract summary language.",
        ],
        "anchor_first_rebuild": [
            "Strategy: anchor-first rebuild.",
            "For each paragraph, preserve the factual anchors, then rebuild the surrounding explanation from scratch.",
            "Prefer domain details already present in the source over new vocabulary.",
        ],
        "confirmed_anchor_threading": [
            "Strategy: confirmed-anchor threading.",
            "Use only the confirmed author anchors provided below, and only where they directly fit the draft's existing topic.",
            "Thread one confirmed anchor into the relevant paragraph as a reasoning hinge, then keep the surrounding text restrained.",
            "Do not invent a wider story, new evidence, new dates, new institutions, or extra examples.",
        ],
        "confirmed_anchor_process_voice": [
            "Strategy: confirmed-anchor process voice.",
            "Where a confirmed author anchor describes an observation or process, let that observation carry the paragraph before the abstract claim.",
            "Use a practical order: what happened, what that showed, then the narrower claim.",
            "Keep uneven sentence length and avoid making the paragraph smoother or more essay-like.",
        ],
        "confirmed_anchor_asymmetry": [
            "Strategy: confirmed-anchor asymmetry.",
            "Use a confirmed anchor in one relevant place, leave unrelated areas mostly untouched in meaning, and avoid balanced claim-explanation-summary cadence.",
            "Allow a slight topic wobble or late connection if it is natural and still faithful to the source.",
            "Do not spread the anchor mechanically across multiple paragraphs.",
        ],
        "confirmed_anchor_claim_narrowing": [
            "Strategy: confirmed-anchor claim narrowing.",
            "Use confirmed author anchors to narrow broad claims into a specific condition, practical situation, or observed consequence.",
            "If an anchor does not directly support a claim, do not use it there.",
            "Prefer a smaller true claim over a broader polished claim.",
        ],
    }.get(strategy, ["Strategy: rewrite for lower measured AI score."])
    lines = [
        "DraftProof AI-score mitigation search.",
        "Objective: produce the lowest measured AI-likelihood score among candidate drafts.",
        (
            "Measured success condition: "
            f"reference AI={reference_ai}, required drop>={required_ai_drop}, "
            f"target AI<={target_ai_score}."
        ),
        "The next scan, not your explanation, decides whether this candidate succeeds.",
        "This is not copyediting. This is not polish. This is detector-targeted reconstruction.",
        *strategy_lines,
        "Use the detector signals as rewrite levers:",
        "- High generic assertion risk: replace broad claims with narrower claims tied to existing source, task, condition, or process details.",
        "- High qualifying AI density: change paragraph architecture, not just words; vary where claims, examples, and source relations appear.",
        "- High top-k predictability: rebuild clause order, split/merge sentence routes, and use less expected verbs while preserving meaning.",
        "- Source/citation gaps: narrow or qualify the claim unless the source already exists in the draft.",
        "- Repeated starters/rhythm: vary openings naturally without mechanical prefixes.",
        "Hard constraints:",
        "Keep the same topic, stance, factual claims, numbers, names, quotes, citations, unit codes, and chronology.",
        "Protected anchors. These exact spans must appear unchanged in the output, including quote text:",
        json.dumps(protected_anchors, ensure_ascii=False)[:2600],
        "Do not drop, paraphrase, normalize, or reword protected anchors. If a quote is awkward, keep the quote exactly and rewrite around it.",
        "Do not invent new evidence, citations, sources, dates, institutions, or examples.",
        "If evidence is missing and the strategy allows marked grounding, use [[REVIEW: ...]] bracketed text instead of inventing the evidence.",
        "Do not summarize or shorten the document. Keep length within about 85% to 115% of the source, except remove accidental duplicate fragments if the source already contains prior rewrite damage.",
        "Do not leave any non-protected source sentence verbatim. Rebuild every sentence route.",
        "Change most sentence openings and vary paragraph openings. Avoid preserving the same paragraph order inside every paragraph.",
        "Avoid generic polished phrases: crucial, significant, essential, framework, landscape, operational obstacles, technical rigor, facilitates, enables, embedded within, especially evident.",
        "Use concrete wording, varied sentence routes, and paragraph-level reconstruction.",
    ]
    if signal_brief:
        lines.append(signal_brief)
    if confirmed_author_anchors:
        lines.append(confirmed_author_anchors)
    if repair_brief:
        lines.append(repair_brief)
    lines.extend([
        "SOURCE DRAFT:\n<TARGET_DOCUMENT>\n" + source_text + "\n</TARGET_DOCUMENT>",
        "Output the complete rewritten draft only.",
        "No commentary, no bullets, no headings added by you, no score estimate.",
    ])
    return "\n".join(lines)


def ai_search_feedback_prompt(
    source_text: str,
    raw_json: dict,
    search_summary: dict,
    attempt_index: int,
    *,
    deps: TargetedRepairPromptDeps,
) -> str:
    """Build a score-aware retry prompt from actual candidate outcomes."""
    signal_brief = deps.ai_search_signal_brief(raw_json)
    repair_brief = deps.source_repair_brief(source_text)
    reference_ai = search_summary.get("reference_ai")
    required_drop = search_summary.get("required_ai_drop")
    target_score = search_summary.get("target_ai_score")
    best_attempt = search_summary.get("best_attempt") or {}
    candidate_lines = []
    for item in (search_summary.get("candidates") or [])[-10:]:
        if not isinstance(item, dict):
            continue
        bits = [str(item.get("strategy") or "candidate")]
        if item.get("ai") is not None:
            bits.append(f"AI={item.get('ai')}")
            bits.append(f"delta={item.get('ai_delta_vs_reference')}")
        if item.get("writing_quality") is not None:
            bits.append(f"WQ={item.get('writing_quality')}")
        if item.get("findings") is not None:
            bits.append(f"findings={item.get('findings')}")
        selection = item.get("selection_status") or {}
        if selection.get("reason"):
            bits.append(f"selection={selection.get('reason')}")
        if item.get("reason"):
            bits.append(f"blocked={item.get('reason')}")
        drift_reasons = item.get("drift_reasons") or item.get("drift_reasons_relaxed")
        if drift_reasons:
            bits.append("drift=" + " | ".join(str(x) for x in drift_reasons[:5]))
        candidate_lines.append("- " + "; ".join(bits))
    scoreboard = "\n".join(candidate_lines) or "- No candidate reached scoring yet."

    return (
        "DraftProof already tried candidate rewrites and rescanned what passed local checks.\n"
        f"Reference AI score: {reference_ai}. Required drop: {required_drop}. Target AI score: {target_score}.\n"
        "Your task is to beat the required target, not to polish and not to make a tiny reduction.\n"
        f"Current best attempt: {best_attempt or '[none]'}\n\n"
        f"{signal_brief}\n\n"
        f"{repair_brief}\n\n"
        "Candidate scoreboard from the actual detector:\n"
        f"{scoreboard}\n\n"
        "What the next attempt must do:\n"
        "- Return the complete rewritten document only.\n"
        "- Preserve all unit codes, source names, citations, years, numbers, and quotes.\n"
        "- Specifically reduce generic assertions, qualifying-text AI density, and top-k predictability.\n"
        "- If earlier candidates only changed wording, change paragraph structure and claim order this time.\n"
        "- Rewrite the highest-driver paragraphs more aggressively while preserving all protected facts.\n"
        "- Rebuild paragraph flow where needed: start from local action, participant behavior, or source relation before broad claims.\n"
        "- Do not add fake facts. If evidence is missing, narrow the claim instead of inventing support.\n"
        "- Avoid mechanical anchor prefixes and visible review markers in the final document.\n"
        "- Repair inherited source damage: broken words, merged headings, and duplicate sentence fragments.\n"
        f"- This is feedback attempt {attempt_index}; make a materially different full-document candidate.\n\n"
        "SOURCE DOCUMENT:\n"
        f"{source_text.strip()}\n\n"
        "Return only the complete rewritten document."
    )


def blocked_human_candidate_repair_prompt(
    source_text: str,
    blocked_candidate: str,
    raw_json: dict,
    blocked_summary: dict,
    attempt_index: int,
    *,
    deps: TargetedRepairPromptDeps,
) -> str:
    """Build a focused repair prompt for a high-Human candidate blocked by gates."""
    signal_brief = deps.ai_search_signal_brief(raw_json)
    selection = blocked_summary.get("selection_status") or {}
    authenticity = selection.get("authenticity_gate") or {}
    failure_controls = []
    if blocked_summary.get("critical_high_findings", 0) > blocked_summary.get("saved_critical_high", 0):
        failure_controls.append(
            "Remove whatever created the new critical/high finding. Usually this means narrowing unsupported claims, deleting over-strong judgement, or restoring a safer factual scope."
        )
    if selection.get("reason") in {"best_candidate_below_required_ai_drop", "candidate_not_below_reference"}:
        failure_controls.append(
            "Reduce AI score further by changing cadence and sentence routes, but preserve the Human Contribution gain."
        )
    if authenticity.get("ai_authorship_regression_blocked") or authenticity.get("ai_authorship_regressed"):
        failure_controls.append(
            "Lower AI Authorship texture: less clean explanation, less balanced structure, fewer polished transitions."
        )
    if authenticity.get("review_burden_regressed"):
        failure_controls.append(
            "Reduce review burden: remove unsupported broad claims introduced by the candidate."
        )
    if authenticity.get("weighted_severity_regressed"):
        failure_controls.append(
            "Reduce weighted severity: weaken or qualify any candidate sentence that sounds more assertive than the source."
        )
    if not failure_controls:
        failure_controls.append(
            "Keep the Human gain while fixing the detector gate failure shown in the scorecard."
        )

    return (
        "DraftProof BLOCKED_HUMAN_WINNER_REPAIR.\n"
        "A candidate moved Human Contribution in the right direction but failed the acceptance gate. "
        "Repair only the failure. Do not restart the rewrite.\n\n"
        f"Blocked candidate scorecard: {json.dumps(blocked_summary, ensure_ascii=False)[:2600]}\n\n"
        f"{signal_brief}\n\n"
        "Repair controls:\n"
        + "\n".join(f"- {item}" for item in failure_controls)
        + "\n\nHard constraints:\n"
        "- Return a complete document, not notes.\n"
        "- Preserve all factual claims, names, numbers, dates, quotes, citations, unit codes, and chronology from the source or blocked candidate.\n"
        "- Do not invent sources, examples, personal experiences, institutions, statistics, or evidence.\n"
        "- Preserve the Human Contribution gain: keep bounded author reasoning traces already present in the blocked candidate unless they caused the gate failure.\n"
        "- If the blocked candidate added an unsafe unsupported claim, narrow it rather than adding evidence.\n"
        "- Do not add bracket markers, commentary, headings, markdown fences, or explanations.\n"
        f"- Repair attempt {attempt_index}: make the smallest complete-document repair that can pass the gates.\n\n"
        "SOURCE DOCUMENT FOR FACT CHECKING:\n"
        f"<SOURCE_DOCUMENT>\n{source_text.strip()}\n</SOURCE_DOCUMENT>\n\n"
        "BLOCKED CANDIDATE TO REPAIR:\n"
        f"<BLOCKED_CANDIDATE>\n{blocked_candidate.strip()}\n</BLOCKED_CANDIDATE>\n\n"
        "Return only the repaired complete document."
    )
