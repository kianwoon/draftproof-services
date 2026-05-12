from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from rewrite_controller.ai_search_selection import ai_search_candidate_rank
from rewrite_pipeline_core.config import _float_env
from rewrite_pipeline_core.prompts.reconstruction_helpers import (
    _clean_section_candidate,
    _human_gain_stage_target,
)
from rewrite_pipeline_core.scoring.profiles import (
    _contribution_scores,
    _feature_percent,
    _formula_gap_candidate_rank,
    _integrity_scores,
)
from rewrite_pipeline_core.text_processing.text_utils import (
    _split_sentences,
    _text_word_count,
)


def _human_shift_score(
    original_report: dict,
    candidate_report: dict,
    *,
    drift_similarity: float | None = None,
    review_burden_delta: int = 0,
    weighted_severity_delta: int = 0,
) -> dict:
    """Score how strongly a candidate moves toward authentic human contribution.

    Positive components reward real mitigation movement. Penalties protect
    meaning, grounding, and review burden so candidate ranking cannot chase a
    lower AI score by making the document worse.
    """
    original = _contribution_scores(original_report)
    candidate = _contribution_scores(candidate_report)
    original_integrity = _integrity_scores(original_report)
    candidate_integrity = _integrity_scores(candidate_report)

    def _delta(original_value, candidate_value, *, direction: str = "increase"):
        if not isinstance(original_value, (int, float)) or not isinstance(candidate_value, (int, float)):
            return 0.0
        if direction == "decrease":
            return float(original_value) - float(candidate_value)
        return float(candidate_value) - float(original_value)

    ai_authorship_reduction = _delta(
        original_integrity.get("ai_authorship"),
        candidate_integrity.get("ai_authorship"),
        direction="decrease",
    )
    human_contribution_gain = _delta(original.get("human"), candidate.get("human"))
    ai_transformation_reduction = _delta(
        original.get("ai_transformation"),
        candidate.get("ai_transformation"),
        direction="decrease",
    )
    human_anchor_gain = _delta(
        _feature_percent(original_report, "human_anchor_score"),
        _feature_percent(candidate_report, "human_anchor_score"),
    )
    grounding_risk_reduction = _delta(
        original_integrity.get("grounding"),
        candidate_integrity.get("grounding"),
        direction="decrease",
    )
    rewrite_smoothness_reduction = _delta(
        _feature_percent(original_report, "rewrite_smoothness"),
        _feature_percent(candidate_report, "rewrite_smoothness"),
        direction="decrease",
    )
    semantic_uniformity_reduction = _delta(
        _feature_percent(original_report, "semantic_uniformity_risk"),
        _feature_percent(candidate_report, "semantic_uniformity_risk"),
        direction="decrease",
    )

    grounding_regression_penalty = max(0.0, -grounding_risk_reduction) * 1.5
    smoothness_regression_penalty = max(0.0, -rewrite_smoothness_reduction) * 0.8
    semantic_uniformity_regression_penalty = max(0.0, -semantic_uniformity_reduction) * 0.8
    review_burden_penalty = max(0, int(review_burden_delta or 0)) * 3.0
    weighted_severity_penalty = max(0, int(weighted_severity_delta or 0)) * 1.5
    meaning_drift_penalty = 0.0
    if isinstance(drift_similarity, (int, float)):
        meaning_drift_penalty = max(0.0, 0.94 - float(drift_similarity)) * 25.0

    components = {
        "ai_authorship_reduction": round(ai_authorship_reduction, 3),
        "human_contribution_gain": round(human_contribution_gain, 3),
        "ai_transformation_reduction": round(ai_transformation_reduction, 3),
        "human_anchor_gain": round(human_anchor_gain, 3),
        "grounding_risk_reduction": round(grounding_risk_reduction, 3),
        "rewrite_smoothness_reduction": round(rewrite_smoothness_reduction, 3),
        "semantic_uniformity_reduction": round(semantic_uniformity_reduction, 3),
        "grounding_regression_penalty": round(grounding_regression_penalty, 3),
        "meaning_drift_penalty": round(meaning_drift_penalty, 3),
        "rewrite_smoothness_regression_penalty": round(smoothness_regression_penalty, 3),
        "semantic_uniformity_regression_penalty": round(semantic_uniformity_regression_penalty, 3),
        "review_burden_penalty": round(review_burden_penalty, 3),
        "weighted_severity_penalty": round(weighted_severity_penalty, 3),
    }
    score = (
        ai_authorship_reduction * 1.0
        + human_contribution_gain * 1.4
        + ai_transformation_reduction * 1.2
        + human_anchor_gain * 0.7
        + max(0.0, grounding_risk_reduction) * 0.35
        + max(0.0, rewrite_smoothness_reduction) * 0.35
        + max(0.0, semantic_uniformity_reduction) * 0.35
        - grounding_regression_penalty
        - meaning_drift_penalty
        - smoothness_regression_penalty
        - semantic_uniformity_regression_penalty
        - review_burden_penalty
        - weighted_severity_penalty
    )
    return {
        "score": round(score, 3),
        "components": components,
        "weights": {
            "ai_authorship_reduction": 1.0,
            "human_contribution_gain": 1.4,
            "ai_transformation_reduction": 1.2,
            "human_anchor_gain": 0.7,
            "grounding_risk_reduction": 0.35,
            "rewrite_smoothness_reduction": 0.35,
            "semantic_uniformity_reduction": 0.35,
            "grounding_regression_penalty": -1.5,
            "meaning_drift_penalty": -1.0,
            "rewrite_smoothness_regression_penalty": -1.0,
            "semantic_uniformity_regression_penalty": -1.0,
            "review_burden_penalty": -1.0,
            "weighted_severity_penalty": -1.0,
        },
    }


def _human_shift_rank_key(gate: dict | None) -> tuple:
    gate = gate or {}
    score = gate.get("human_shift_score")
    candidate_human = gate.get("candidate_human")
    ai_authorship_delta = gate.get("ai_authorship_delta")
    stage_target = _human_gain_stage_target(candidate_human)
    return (
        1 if gate.get("success") else 0,
        1 if not isinstance(ai_authorship_delta, (int, float)) or ai_authorship_delta >= 0 else 0,
        1 if isinstance(candidate_human, (int, float)) and candidate_human >= stage_target else 0,
        float(gate.get("human_delta")) if isinstance(gate.get("human_delta"), (int, float)) else -9999.0,
        float(score) if isinstance(score, (int, float)) else -9999.0,
        float(ai_authorship_delta) if isinstance(ai_authorship_delta, (int, float)) else -9999.0,
        float(gate.get("ai_transformation_delta")) if isinstance(gate.get("ai_transformation_delta"), (int, float)) else -9999.0,
    )


def _is_better_human_shift_candidate(candidate_gate: dict | None, best_gate: dict | None) -> bool:
    if best_gate is None:
        return True
    return _human_shift_rank_key(candidate_gate) > _human_shift_rank_key(best_gate)


def _goal_climb_candidate_rank(
    selection_status: dict | None,
    candidate_eval: dict | None,
    *,
    candidate_ai=None,
    candidate_review_burden: int | float = 0,
    candidate_weighted_severity: int | float = 0,
    candidate_finding_total: int | float = 0,
    original_review_burden: int | float = 0,
    original_weighted_severity: int | float = 0,
    original_finding_total: int | float = 0,
) -> tuple:
    """Rank AI-mitigation candidates by progress toward the Human 80 goal.

    AI score is intentionally late in the key. A lower AI score is not a win
    when Human Shift, authorship texture, or review burden move the wrong way.
    """
    status = selection_status or {}
    eval_data = candidate_eval or {}
    gate = status.get("authenticity_gate") if isinstance(status.get("authenticity_gate"), dict) else status

    def num(value, default=-9999.0) -> float:
        return float(value) if isinstance(value, (int, float)) else float(default)

    candidate_human = num(gate.get("candidate_human", eval_data.get("human_contribution")))
    stage_target = _human_gain_stage_target(candidate_human)
    target_human = _float_env("DRAFTPROOF_AUTHENTICITY_TARGET_HUMAN", 80.0)
    turnitin_gate = status.get("turnitin_like_ai_gate") if isinstance(status.get("turnitin_like_ai_gate"), dict) else {}
    formula_gap_contract = (
        status.get("formula_gap_contract")
        if isinstance(status.get("formula_gap_contract"), dict)
        else eval_data.get("formula_gap_contract")
        if isinstance(eval_data.get("formula_gap_contract"), dict)
        else {}
    )
    return ai_search_candidate_rank(
        status,
        eval_data,
        candidate_ai=candidate_ai,
        candidate_review_burden=candidate_review_burden,
        candidate_weighted_severity=candidate_weighted_severity,
        candidate_finding_total=candidate_finding_total,
        original_review_burden=original_review_burden,
        original_weighted_severity=original_weighted_severity,
        original_finding_total=original_finding_total,
        target_human=target_human,
        stage_target=stage_target,
        formula_gap_rank=_formula_gap_candidate_rank(formula_gap_contract, turnitin_gate),
    )


def _anchor_lock_mapping(anchors: list[str] | tuple[str, ...] | None) -> list[dict]:
    """Create deterministic placeholders for anchors that should not be rewritten."""
    unique: list[str] = []
    for raw in anchors or []:
        value = str(raw or "").strip()
        if len(value) < 4 and not re.fullmatch(r"\d+(?:\.\d+)?%?", value):
            continue
        if value not in unique:
            unique.append(value)
    unique.sort(key=len, reverse=True)
    return [
        {"placeholder": f"[[DP_ANCHOR_{index:03d}]]", "value": value}
        for index, value in enumerate(unique, start=1)
    ]


def _anchor_values_from_brief(anchors: list[dict] | tuple[dict, ...] | None) -> list[str]:
    values: list[str] = []
    for item in anchors or []:
        if not isinstance(item, dict):
            continue
        value = str(item.get("text") or item.get("value") or "").strip()
        if value and value not in values:
            values.append(value)
    return values


def _freeze_anchor_text(text: str, mapping: list[dict] | None) -> str:
    frozen = str(text or "")
    for item in mapping or []:
        value = str(item.get("value") or "")
        placeholder = str(item.get("placeholder") or "")
        if value and placeholder:
            if re.fullmatch(r"\d+(?:\.\d+)?%?", value):
                pattern = rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])"
            else:
                pattern = re.escape(value)
            frozen = re.sub(pattern, placeholder, frozen)
    return frozen


def _restore_anchor_placeholders(text: str, mapping: list[dict] | None) -> str:
    restored = str(text or "")
    for item in mapping or []:
        value = str(item.get("value") or "")
        placeholder = str(item.get("placeholder") or "")
        if value and placeholder:
            restored = restored.replace(placeholder, value)
    return restored


def _freeze_anchor_payload(payload, mapping: list[dict] | None):
    if isinstance(payload, str):
        return _freeze_anchor_text(payload, mapping)
    if isinstance(payload, list):
        return [_freeze_anchor_payload(item, mapping) for item in payload]
    if isinstance(payload, tuple):
        return tuple(_freeze_anchor_payload(item, mapping) for item in payload)
    if isinstance(payload, dict):
        return {
            key: _freeze_anchor_payload(value, mapping)
            for key, value in payload.items()
        }
    return payload


def _repair_aggression_score(original_text: str, candidate_text: str) -> dict:
    """Estimate how invasive a repair is so texture repair stays micro-local."""
    original_tokens = re.findall(r"\w+|[^\w\s]", str(original_text or ""))
    candidate_tokens = re.findall(r"\w+|[^\w\s]", str(candidate_text or ""))
    if not original_tokens and not candidate_tokens:
        ratio = 1.0
    else:
        ratio = SequenceMatcher(None, original_tokens, candidate_tokens).ratio()
    changed_tokens_ratio = round(1.0 - ratio, 3)
    original_sentences = _split_sentences(str(original_text or ""))
    candidate_sentences = _split_sentences(str(candidate_text or ""))
    sentence_count_delta = abs(len(candidate_sentences) - len(original_sentences))
    sentence_delta_ratio = round(
        sentence_count_delta / max(1, len(original_sentences)),
        3,
    )
    original_words = _text_word_count(str(original_text or ""))
    candidate_words = _text_word_count(str(candidate_text or ""))
    word_delta_ratio = round(
        abs(candidate_words - original_words) / max(1, original_words),
        3,
    )
    score = round(
        changed_tokens_ratio + min(1.0, sentence_delta_ratio) * 0.5 + min(1.0, word_delta_ratio) * 0.5,
        3,
    )
    return {
        "score": score,
        "changed_tokens_ratio": changed_tokens_ratio,
        "sentence_delta_ratio": sentence_delta_ratio,
        "word_delta_ratio": word_delta_ratio,
        "original_words": original_words,
        "candidate_words": candidate_words,
    }


def _sentence_texture_risk_map(text: str, raw_json: dict | None = None, limit: int = 5) -> list[dict]:
    """Build a sentence-level repair map from scanner pointers and local texture signals."""
    sentences = _split_sentences(text)
    if not sentences:
        return []
    pointer_scores: dict[int, float] = {}
    raw_json = raw_json or {}
    for brief in raw_json.get("rewrite_edit_briefs") or []:
        if not isinstance(brief, dict):
            continue
        index = brief.get("sentence_index")
        if isinstance(index, int) and 0 <= index < len(sentences):
            signal_bonus = 0.25
            for value in (brief.get("signals") or {}).values():
                if isinstance(value, (int, float)):
                    signal_bonus += min(0.4, float(value) / 250.0)
                elif isinstance(value, str) and re.search(r"predict|uniform|smooth|generic|cadence", value, re.I):
                    signal_bonus += 0.15
            pointer_scores[index] = max(pointer_scores.get(index, 0.0), signal_bonus)
    for segment in ((raw_json.get("ai_mitigation") or {}).get("target_segments") or []):
        if not isinstance(segment, dict):
            continue
        seg_text = str(segment.get("text") or "").strip()
        if not seg_text:
            continue
        for index, sentence in enumerate(sentences):
            if seg_text in sentence or sentence in seg_text:
                signal = segment.get("primary_signal") or {}
                value = signal.get("score")
                pointer_scores[index] = max(
                    pointer_scores.get(index, 0.0),
                    0.35 + (min(0.4, float(value) / 250.0) if isinstance(value, (int, float)) else 0.0),
                )
    generic_re = re.compile(
        r"\b(?:important|significant|crucial|essential|plays? a role|"
        r"this (?:shows|highlights|demonstrates|emphasizes)|"
        r"furthermore|moreover|additionally|in conclusion|overall|"
        r"increasingly|integrat(?:e|es|ing)|supports?|enables?|enhances?)\b",
        re.I,
    )
    transition_re = re.compile(r"^(?:This|These|Such|In this|Overall|Therefore|However|Additionally|Furthermore)\b")
    rows: list[dict] = []
    for index, sentence in enumerate(sentences):
        words = re.findall(r"\b[\w'-]+\b", sentence)
        length_score = min(0.25, max(0, len(words) - 22) / 120.0)
        generic_score = min(0.35, len(generic_re.findall(sentence)) * 0.12)
        transition_score = 0.15 if transition_re.search(sentence.strip()) else 0.0
        score = round(pointer_scores.get(index, 0.0) + length_score + generic_score + transition_score, 3)
        rows.append({
            "sentence_index": index,
            "sentence": sentence,
            "risk": score,
            "drivers": {
                "scanner_pointer": round(pointer_scores.get(index, 0.0), 3),
                "length": round(length_score, 3),
                "generic_phrase": round(generic_score, 3),
                "transition_cleanliness": round(transition_score, 3),
            },
        })
    rows.sort(key=lambda row: row["risk"], reverse=True)
    return rows[:max(1, limit)]


def _micro_texture_window(
    text: str,
    raw_json: dict | None = None,
    *,
    max_sentences: int = 2,
    exclude_sentence_indexes: set[int] | list[int] | tuple[int, ...] | None = None,
) -> dict:
    sentences = _split_sentences(text)
    if not sentences:
        return {"sentences": [], "start": 0, "end": 0, "text": "", "risk_map": []}
    risk_map = _sentence_texture_risk_map(text, raw_json, limit=5)
    excluded = {int(item) for item in (exclude_sentence_indexes or []) if isinstance(item, int)}
    selected = next(
        (
            row for row in risk_map
            if "\n" not in str(row.get("sentence") or "")
            and int(row.get("sentence_index", -1)) not in excluded
        ),
        None,
    )
    if selected is None:
        selected = next(
            (
                row for row in risk_map
                if int(row.get("sentence_index", -1)) not in excluded
            ),
            None,
        )
    if selected is None:
        return {"sentences": sentences, "start": 0, "end": 0, "text": "", "risk_map": risk_map}
    start = int(selected.get("sentence_index", 0))
    end = min(len(sentences), start + max(1, max_sentences))
    return {
        "sentences": sentences,
        "start": start,
        "end": end,
        "text": " ".join(sentences[start:end]).strip(),
        "risk_map": risk_map,
    }


def _splice_sentence_window(text: str, start: int, end: int, replacement: str) -> str:
    sentences = _split_sentences(text)
    if start < 0 or end <= start or start >= len(sentences):
        return text
    end = min(end, len(sentences))
    replacement_sentences = _split_sentences(replacement)
    if not replacement_sentences:
        return text
    rebuilt = sentences[:start] + replacement_sentences + sentences[end:]
    return " ".join(rebuilt).strip()


def _splice_sentence_for_auto_repair(text: str, sentence_index: int, replacement: str) -> str:
    sentences = _split_sentences(text)
    if sentence_index < 0 or sentence_index >= len(sentences):
        return text
    replacement_sentences = _split_sentences(replacement)
    rebuilt = sentences[:sentence_index] + replacement_sentences + sentences[sentence_index + 1:]
    return " ".join(sentence.strip() for sentence in rebuilt if sentence and sentence.strip()).strip()


def _micro_texture_repair_prompt(
    source_text: str,
    raw_json: dict | None,
    anchors: list[str] | None = None,
    *,
    max_sentences: int = 1,
    exclude_sentence_indexes: set[int] | list[int] | tuple[int, ...] | None = None,
    mode: str = "authorship_texture_repair",
) -> tuple[str, dict]:
    """Build an operation-level prompt for one local authorship texture patch."""
    window = _micro_texture_window(
        source_text,
        raw_json,
        max_sentences=max_sentences,
        exclude_sentence_indexes=exclude_sentence_indexes,
    )
    mapping = _anchor_lock_mapping(anchors or [])
    frozen_window = _freeze_anchor_text(window.get("text") or "", mapping)
    frozen_context = {
        "window_start_sentence": window.get("start"),
        "window_end_sentence": window.get("end"),
        "risk_map": _freeze_anchor_payload(window.get("risk_map") or [], mapping),
        "target_window": frozen_window,
        "required_placeholders": [item["placeholder"] for item in mapping if item["placeholder"] in frozen_window],
    }
    mode_name = str(mode or "authorship_texture_repair").strip().lower()
    if mode_name == "authorship_suppression_repair":
        objective = (
            "DraftProof micro-local AUTHORSHIP_SUPPRESSION_REPAIR.\n"
            "Objective: reduce AI Authorship first. Human Contribution is only a bonus.\n"
            "Patch only the target sentence window. Do not improve the whole section.\n"
        )
        allowed = (
            "- shorten_over_complete_explanation\n"
            "- break_repeated_sentence_cadence\n"
            "- remove_neat_claim_explain_conclude_flow\n"
            "- reduce_polished_transition\n"
            "- leave one clear point slightly less over-explained\n"
        )
        forbidden_extra = (
            "- adding anchors\n"
            "- adding first-person\n"
            "- expanding explanations\n"
            "- improving clarity by smoothing every link\n"
        )
    else:
        objective = (
            "DraftProof micro-local AUTHORSHIP_TEXTURE_REPAIR.\n"
            "Patch only the target sentence window. Do not rewrite the surrounding section.\n"
        )
        allowed = (
            "- shorten_transition\n"
            "- reduce_explanation\n"
            "- alter_sentence_length\n"
            "- reduce_connector_strength\n"
            "- slight_pacing_asymmetry\n"
        )
        forbidden_extra = ""
    prompt = (
        objective +
        "Return only the replacement sentence window, not the full section.\n\n"
        f"Repair context:\n{json.dumps(frozen_context, ensure_ascii=False)}\n\n"
        "Allowed operations:\n"
        f"{allowed}\n"
        "Forbidden operations:\n"
        "- reorder_paragraph\n"
        "- add_new_claim\n"
        "- semantic_expansion\n"
        "- rewrite_whole_sentence_cluster\n"
        "- add examples, sources, dates, numbers, institutions, citations, or evidence\n"
        "- add typos, fake randomness, or grammar damage\n\n"
        f"{forbidden_extra}"
        "Hard constraints:\n"
        "- Preserve meaning and all [[DP_ANCHOR_###]] placeholders exactly.\n"
        "- Keep replacement within about 70% to 130% of the target window word count.\n"
        "- Prefer lower authorship regularity over polished explanation.\n"
        "- Output prose only."
    )
    return prompt, {
        "schema_version": "micro_texture_repair.v1",
        "window": window,
        "anchor_lock": mapping,
        "frozen_target_window": frozen_window,
    }


def _clean_micro_texture_candidate(output: str, repair_info: dict) -> tuple[str, str]:
    text = _clean_section_candidate(output or "", "")
    if not text:
        return "", "empty_micro_texture_candidate"
    mapping = (repair_info or {}).get("anchor_lock") or []
    text = _restore_anchor_placeholders(text, mapping)
    target_words = _text_word_count((repair_info or {}).get("window", {}).get("text") or "")
    candidate_words = _text_word_count(text)
    if target_words and candidate_words < max(3, int(target_words * 0.70)):
        return "", f"micro_texture_candidate_too_short {candidate_words}<{int(target_words * 0.70)}"
    if target_words and candidate_words > max(8, int(target_words * 1.30)):
        return "", f"micro_texture_candidate_too_long {candidate_words}>{int(target_words * 1.30)}"
    return text, ""


_MASKED_REPAIR_GENERIC_RE = re.compile(
    r"\b(?:Furthermore|Moreover|Additionally|In conclusion|Overall|Therefore|However|"
    r"In the past|This shift has made|The real challenge is|"
    r"It is important to note that|This highlights|This shows|This demonstrates|"
    r"plays? a crucial role|significant impact|important|crucial|essential)\b",
    re.I,
)


def _masked_span_repair_prompt(
    source_text: str,
    raw_json: dict | None,
    *,
    exclude_sentence_indexes: set[int] | list[int] | tuple[int, ...] | None = None,
) -> tuple[str, dict]:
    """Mask only a high-risk local span so generation cannot rewrite the section."""
    window = _micro_texture_window(
        source_text,
        raw_json,
        max_sentences=1,
        exclude_sentence_indexes=exclude_sentence_indexes,
    )
    sentence = str(window.get("text") or "")
    if not sentence:
        return "", {"reason": "no_mask_window", "window": window}

    mask_start = mask_end = -1
    mask_text = ""
    match = _MASKED_REPAIR_GENERIC_RE.search(sentence)
    if match:
        mask_start, mask_end = match.span()
        mask_text = match.group(0)
    if mask_start < 0:
        for brief in (raw_json or {}).get("rewrite_edit_briefs") or []:
            if not isinstance(brief, dict) or brief.get("sentence_index") != window.get("start"):
                continue
            for token in brief.get("problem_tokens") or []:
                token_text = str(token or "").strip()
                if len(token_text) < 4:
                    continue
                pos = sentence.lower().find(token_text.lower())
                if pos >= 0:
                    mask_start, mask_end = pos, pos + len(token_text)
                    mask_text = sentence[pos:mask_end]
                    break
            if mask_start >= 0:
                break
    if mask_start < 0:
        words = list(re.finditer(r"\b[\w'-]+\b", sentence))
        if not words:
            return "", {"reason": "no_maskable_span", "window": window}
        # Last resort: mask a short opening phrase, not the whole sentence.
        first = words[0]
        last = words[min(2, len(words) - 1)]
        mask_start, mask_end = first.start(), last.end()
        mask_text = sentence[mask_start:mask_end]

    masked_sentence = f"{sentence[:mask_start]}[[MASK]]{sentence[mask_end:]}"
    prompt = (
        "DraftProof partial masked regeneration.\n"
        "Replace only [[MASK]] in the sentence. Do not rewrite any other words.\n"
        "Objective: reduce AI Authorship texture without semantic expansion.\n\n"
        f"Masked sentence: {masked_sentence}\n"
        f"Original masked span: {mask_text}\n\n"
        "Rules:\n"
        "- Return only the replacement text for [[MASK]].\n"
        "- Replacement may be empty if deleting the span reads naturally.\n"
        "- Use at most 8 words.\n"
        "- Do not add claims, evidence, examples, citations, numbers, names, first-person, or explanation.\n"
        "- Prefer plain wording or no connector over polished academic phrasing."
    )
    return prompt, {
        "schema_version": "masked_span_repair.v1",
        "window": window,
        "sentence": sentence,
        "masked_sentence": masked_sentence,
        "mask_text": mask_text,
        "mask_start": mask_start,
        "mask_end": mask_end,
    }


def _clean_masked_span_replacement(output: str) -> str:
    text = _clean_section_candidate(output or "", "")
    text = re.sub(r"\[\[/?MASK\]\]", "", text, flags=re.I).strip()
    text = text.strip("\"'` ")
    words = re.findall(r"\b[\w'-]+\b", text)
    if len(words) > 8:
        return " ".join(words[:8])
    return text


def _deterministic_masked_span_replacements(mask_text: str) -> list[str]:
    """Return safe local replacements before spending an LLM call.

    These are intentionally tiny span-level alternatives, not sentence rewrites.
    The scanner still decides whether any candidate is kept.
    """
    key = re.sub(r"\s+", " ", str(mask_text or "").strip()).lower()
    replacements = {
        "important": ["vital", "needed", "useful"],
        "in the past": ["Earlier"],
        "this shift has made": ["That shift makes"],
        "the real challenge is": ["The harder part is"],
        "this is a": ["A"],
        "this has created": ["This creates", "It creates"],
        "this can encourage": ["This may lead to", "It can lead to"],
        "this makes assessment": ["Assessment becomes"],
        "in other words": ["Put simply", "Simply"],
    }.get(key, [])
    unique: list[str] = []
    for item in replacements:
        clean = _clean_masked_span_replacement(item)
        if clean not in unique:
            unique.append(clean)
    return unique


def _deterministic_sentence_route_bundle(source_text: str) -> tuple[str, list[dict]]:
    """Apply a small set of safe sentence-route edits as one candidate.

    These are not synonym swaps. They target common scanner-visible discourse
    routes while preserving the surrounding claim and all anchors.
    """
    text = str(source_text or "")
    edits = [
        ("In the past", "Earlier"),
        ("This shift has made", "That shift makes"),
        ("The real challenge is", "The harder part is"),
    ]
    applied: list[dict] = []
    candidate = text
    for old, new in edits:
        pattern = re.compile(r"\b" + re.escape(old) + r"\b", re.I)
        if not pattern.search(candidate):
            continue
        candidate = pattern.sub(new, candidate, count=1)
        applied.append({"mask_text": old, "replacement": new})
    return candidate, applied


def _apply_masked_span_replacement(source_text: str, repair_info: dict, replacement: str) -> str:
    sentence = str((repair_info or {}).get("sentence") or "")
    start = int((repair_info or {}).get("mask_start") or 0)
    end = int((repair_info or {}).get("mask_end") or 0)
    if not sentence or end < start:
        return source_text
    replacement = str(replacement or "").strip()
    repaired_sentence = f"{sentence[:start]}{replacement}{sentence[end:]}"
    repaired_sentence = re.sub(r"\s+([,.;:!?])", r"\1", repaired_sentence)
    repaired_sentence = re.sub(r"\s{2,}", " ", repaired_sentence).strip()
    if sentence and sentence in str(source_text or ""):
        return str(source_text or "").replace(sentence, repaired_sentence, 1)
    window = (repair_info or {}).get("window") or {}
    return _splice_sentence_window(
        source_text,
        int(window.get("start") or 0),
        int(window.get("end") or 0),
        repaired_sentence,
    )


def _locality_score(original_text: str, candidate_text: str) -> dict:
    original_sentences = _split_sentences(original_text)
    candidate_sentences = _split_sentences(candidate_text)
    max_len = max(len(original_sentences), len(candidate_sentences), 1)
    changed = 0
    for index in range(max_len):
        left = original_sentences[index] if index < len(original_sentences) else ""
        right = candidate_sentences[index] if index < len(candidate_sentences) else ""
        if left != right:
            changed += 1
    ratio = round(changed / max_len, 3)
    return {
        "changed_sentences": changed,
        "total_sentences": max_len,
        "changed_sentence_ratio": ratio,
    }


def _micro_repair_gain_efficiency(human_gain: float, aggression_delta: float) -> float:
    """Measure attribution gain per unit of repair aggression."""
    try:
        human_gain_value = float(human_gain)
    except (TypeError, ValueError):
        human_gain_value = 0.0
    try:
        aggression_value = float(aggression_delta)
    except (TypeError, ValueError):
        aggression_value = 0.0
    if aggression_value <= 0:
        return 9999.0 if human_gain_value > 0 else 0.0
    return round(human_gain_value / aggression_value, 3)


def _micro_texture_iteration_status(
    attempts: list[dict] | None = None,
    *,
    baseline_scan: dict | None = None,
    previous_scan: dict | None = None,
    current_scan: dict | None = None,
) -> dict:
    """Stop/go policy for iterative micro-local texture repair.

    The generator may create several tiny patches, but the loop must stop
    before many local edits add up to a disguised section rewrite.
    """
    attempts = attempts or []

    def num(source: dict | None, key: str, default: float = 0.0) -> float:
        if not isinstance(source, dict):
            return default
        value = source.get(key)
        return float(value) if isinstance(value, (int, float)) else default

    def scan_from_attempt(index: int) -> dict:
        if not attempts:
            return {}
        try:
            item = attempts[index]
        except IndexError:
            return {}
        return item.get("scan_scores") or item.get("scan") or {}

    if current_scan is None:
        current_scan = scan_from_attempt(-1)
    if previous_scan is None:
        previous_scan = scan_from_attempt(-2) if len(attempts) >= 2 else (baseline_scan or {})
    if baseline_scan is None:
        baseline_scan = previous_scan or {}

    cumulative_aggression = 0.0
    max_locality_ratio = 0.0
    latest_aggression = 0.0
    for index, item in enumerate(attempts):
        aggression = item.get("repair_aggression") or {}
        if not isinstance(aggression, dict):
            aggression = {}
        score = num(aggression, "score", num(item, "repair_aggression_score"))
        cumulative_aggression += max(0.0, score)
        if index == len(attempts) - 1:
            latest_aggression = max(0.0, score)
        locality = item.get("locality") or {}
        if isinstance(locality, dict):
            max_locality_ratio = max(max_locality_ratio, num(locality, "changed_sentence_ratio"))

    current_human = num(current_scan, "human")
    previous_human = num(previous_scan, "human")
    baseline_human = num(baseline_scan, "human", previous_human)
    marginal_human_gain = current_human - previous_human
    total_human_gain = current_human - baseline_human
    ai_authorship_delta = num(current_scan, "ai_authorship") - num(previous_scan, "ai_authorship")
    ai_transformation_delta = num(current_scan, "ai_transformation") - num(previous_scan, "ai_transformation")
    findings_delta = num(current_scan, "findings") - num(previous_scan, "findings")
    gain_efficiency = _micro_repair_gain_efficiency(marginal_human_gain, latest_aggression)

    max_total_aggression = _float_env("DRAFTPROOF_MICRO_TEXTURE_MAX_TOTAL_AGGRESSION", 0.18)
    max_locality = _float_env("DRAFTPROOF_TEXTURE_REPAIR_MAX_LOCALITY", 0.25)
    min_human_gain = _float_env("DRAFTPROOF_MICRO_TEXTURE_MIN_HUMAN_GAIN", 1.0)
    min_gain_efficiency = _float_env("DRAFTPROOF_MICRO_TEXTURE_MIN_GAIN_EFFICIENCY", 10.0)
    max_iterations = int(_float_env("DRAFTPROOF_MICRO_TEXTURE_MAX_ITERATIONS", 5.0))

    stop_reasons: list[str] = []
    if attempts and cumulative_aggression > max_total_aggression:
        stop_reasons.append("cumulative_aggression_budget_exhausted")
    if attempts and max_locality_ratio > max_locality:
        stop_reasons.append("repair_locality_high")
    if attempts and ai_authorship_delta > 0:
        stop_reasons.append("ai_authorship_regression")
    if attempts and findings_delta > 0:
        stop_reasons.append("findings_regression")
    if attempts and marginal_human_gain < min_human_gain:
        stop_reasons.append("diminishing_human_gain")
    if attempts and marginal_human_gain > 0 and gain_efficiency < min_gain_efficiency:
        stop_reasons.append("gain_efficiency_low")
    if attempts and len(attempts) > max_iterations:
        stop_reasons.append("max_iterations_reached")

    return {
        "continue": not stop_reasons,
        "stop_reasons": stop_reasons,
        "metrics": {
            "attempt_count": len(attempts),
            "cumulative_aggression": round(cumulative_aggression, 3),
            "max_total_aggression": round(max_total_aggression, 3),
            "latest_aggression": round(latest_aggression, 3),
            "max_locality_changed_sentence_ratio": round(max_locality_ratio, 3),
            "max_locality_limit": round(max_locality, 3),
            "marginal_human_gain": round(marginal_human_gain, 3),
            "total_human_gain": round(total_human_gain, 3),
            "ai_authorship_delta": round(ai_authorship_delta, 3),
            "ai_transformation_delta": round(ai_transformation_delta, 3),
            "findings_delta": round(findings_delta, 3),
            "gain_efficiency": gain_efficiency,
            "min_human_gain": round(min_human_gain, 3),
            "min_gain_efficiency": round(min_gain_efficiency, 3),
            "max_iterations": max_iterations,
        },
    }


def _iterative_micro_texture_repair(
    source_text: str,
    raw_json: dict | None,
    *,
    baseline_scan: dict,
    generate_replacement,
    scan_candidate,
    anchors: list[str] | None = None,
    max_attempts: int | None = None,
) -> dict:
    """Run iterative micro-local texture repair with deterministic stop controls.

    `generate_replacement(prompt, repair_info, attempt_index)` supplies one
    replacement window. `scan_candidate(text)` returns the scanner metrics for
    the full candidate text. The loop accepts only patches that pass the
    iteration policy.
    """
    try:
        limit = int(max_attempts if max_attempts is not None else _float_env("DRAFTPROOF_MICRO_TEXTURE_MAX_ITERATIONS", 5.0))
    except (TypeError, ValueError):
        limit = 5
    limit = max(0, limit)
    current_text = str(source_text or "")
    previous_scan = dict(baseline_scan or {})
    accepted_attempts: list[dict] = []
    rejected_attempts: list[dict] = []
    repaired_indexes: set[int] = set()
    stop_reason = "max_attempts_reached" if limit == 0 else ""

    for attempt_index in range(1, limit + 1):
        window = _micro_texture_window(
            current_text,
            raw_json,
            max_sentences=1,
            exclude_sentence_indexes=repaired_indexes,
        )
        if not window.get("text"):
            stop_reason = "no_unrepaired_texture_window"
            break
        prompt, repair_info = _micro_texture_repair_prompt(
            current_text,
            raw_json,
            anchors or [],
            max_sentences=1,
            exclude_sentence_indexes=repaired_indexes,
        )
        raw_replacement = generate_replacement(prompt, repair_info, attempt_index)
        replacement, clean_reason = _clean_micro_texture_candidate(str(raw_replacement or ""), repair_info)
        attempt = {
            "attempt": attempt_index,
            "window_start": window.get("start"),
            "window_end": window.get("end"),
            "target_window": window.get("text"),
            "accepted": False,
        }
        if clean_reason:
            attempt["reason"] = clean_reason
            rejected_attempts.append(attempt)
            stop_reason = clean_reason
            break
        candidate_text = _splice_sentence_window(
            current_text,
            int(window.get("start") or 0),
            int(window.get("end") or 0),
            replacement,
        )
        if candidate_text == current_text:
            attempt["reason"] = "micro_texture_no_change"
            rejected_attempts.append(attempt)
            stop_reason = "micro_texture_no_change"
            break
        attempt.update({
            "replacement_window": replacement,
            "repair_aggression": _repair_aggression_score(current_text, candidate_text),
            "locality": _locality_score(current_text, candidate_text),
        })
        try:
            scan_scores = scan_candidate(candidate_text)
        except Exception as exc:
            attempt["reason"] = f"candidate_scan_error {exc}"
            rejected_attempts.append(attempt)
            stop_reason = attempt["reason"]
            break
        attempt["scan_scores"] = scan_scores or {}
        iteration_status = _micro_texture_iteration_status(
            accepted_attempts + [attempt],
            baseline_scan=baseline_scan,
            previous_scan=previous_scan,
            current_scan=attempt["scan_scores"],
        )
        attempt["iteration_status"] = iteration_status
        if not iteration_status.get("continue"):
            attempt["reason"] = ",".join(iteration_status.get("stop_reasons") or []) or "iteration_policy_stop"
            rejected_attempts.append(attempt)
            stop_reason = attempt["reason"]
            break
        attempt["accepted"] = True
        accepted_attempts.append(attempt)
        repaired_indexes.add(int(window.get("start") or 0))
        current_text = candidate_text
        previous_scan = dict(attempt["scan_scores"])
    else:
        stop_reason = "max_attempts_reached"

    final_status = _micro_texture_iteration_status(
        accepted_attempts,
        baseline_scan=baseline_scan,
        previous_scan=baseline_scan if len(accepted_attempts) <= 1 else accepted_attempts[-2].get("scan_scores"),
        current_scan=previous_scan,
    )
    return {
        "text": current_text,
        "scan_scores": previous_scan,
        "accepted_attempts": accepted_attempts,
        "rejected_attempts": rejected_attempts,
        "attempt_count": len(accepted_attempts) + len(rejected_attempts),
        "accepted_count": len(accepted_attempts),
        "stop_reason": stop_reason,
        "iteration_status": final_status,
        "repaired_sentence_indexes": sorted(repaired_indexes),
    }
