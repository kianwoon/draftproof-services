"""Paragraph-portfolio prompt templates for broad assisted footprint problems."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from ..prompt_contract import phrase_level_spans, span_rows
from ..target_executor import TargetGroup, required_protected_anchors_for_source


TEMPLATE_ID = "paragraph_portfolio.v1"
STRATEGY_ID = "paragraph_preserving_broad_reconstruction"

ROLE_CHOICES = (
    "opening_frame",
    "background_context",
    "evidence_or_example_unit",
    "contrast_or_problem_unit",
    "implication_unit",
    "closing_frame",
    "technical_or_policy_unit",
    "citation_or_evidence_unit",
    "unknown_unit",
)

OPERATOR_CHOICES = (
    "BREAK_SURVEY_TEMPLATE",
    "MOVE_SOURCE_ANCHOR_FORWARD",
    "REDUCE_BALANCED_CLAUSE_PAIRING",
    "COMPRESS_GENERIC_OVERVIEW",
    "TOPK_SPAN_REPATH",
    "PRESERVE_FACTUAL_CONTEXT",
    "DELETE_GENERIC_CONCLUSION_FRAME",
    "REBALANCE_SENTENCE_ROUTE",
)


DRIVER_FOCUS = {
    "predictability_score": "predictable sentence route",
    "ai_signal_score": "generated overview texture",
    "ai_likelihood": "model-like paragraph movement",
    "unsafe_word_share": "weak process reasoning / weak cause-effect ownership",
}


OPERATION_CONTRACTS = {
    "BREAK_SURVEY_TEMPLATE": {
        "movement": "Keep the same facts, but change the paragraph route so it does not move as broad claim, tidy explanation, polished closing.",
        "method": "Start from the source paragraph's concrete subject when possible, then explain the implication in plain language.",
    },
    "MOVE_SOURCE_ANCHOR_FORWARD": {
        "movement": "Move a concrete source-supported anchor closer to the opening instead of starting with a broad overview.",
        "method": "Use the source paragraph's own examples or terms as the entry point, then connect back to the main claim.",
    },
    "REDUCE_BALANCED_CLAUSE_PAIRING": {
        "movement": "Reduce overly balanced clause pairs and make the reasoning less symmetrical.",
        "method": "Prefer one clear judgment followed by a short qualification when the source supports it.",
    },
    "COMPRESS_GENERIC_OVERVIEW": {
        "movement": "Remove empty overview phrasing without dropping factual coverage.",
        "method": "Keep the paragraph near the preferred word count by spending words on source-supported meaning, not formal framing.",
    },
    "TOPK_SPAN_REPATH": {
        "movement": "Change the most predictable wording route while preserving meaning.",
        "method": "Use clause movement, deletion of empty phrasing, or plainer source-supported wording.",
    },
    "PRESERVE_FACTUAL_CONTEXT": {
        "movement": "Keep factual coverage stable while making the paragraph less polished and less generic.",
        "method": "Keep simple source wording when it is already direct; avoid elevated substitutions.",
    },
    "DELETE_GENERIC_CONCLUSION_FRAME": {
        "movement": "Remove generic concluding frame language while preserving the paragraph's final point.",
        "method": "End on the source-supported judgment rather than a neat summary formula.",
    },
    "REBALANCE_SENTENCE_ROUTE": {
        "movement": "Vary sentence route without changing facts or paragraph role.",
        "method": "Mix direct short sentences with one explanatory sentence where the source needs it.",
    },
}


@dataclass(frozen=True)
class PromptTemplatePayload:
    template_id: str
    strategy_id: str
    prompt_stage: str
    prompt: str
    scanner_context_used: tuple[str, ...]

    def to_trace(self) -> dict[str, Any]:
        return {
            "prompt_template_id": self.template_id,
            "strategy_id": self.strategy_id,
            "prompt_stage": self.prompt_stage,
            "prompt_chars": len(self.prompt),
            "scanner_context_used": list(self.scanner_context_used),
        }


def _json_payload(header: str, payload: dict[str, Any]) -> str:
    return f"{header}\n\nPAYLOAD:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"


def _int_env(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _limit_text(text: Any, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    clipped = value[:max(0, limit - 1)].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return f"{clipped}..."


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _compact_driver_rows(rows: list[dict[str, Any]], *, limit: int = 2) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in sorted(rows or [], key=lambda item: _number(item.get("score")), reverse=True)[:limit]:
        compact.append({
            "key": row.get("key"),
            "score": row.get("score"),
            "label": _limit_text(row.get("label"), 48),
        })
    return compact


def _driver_focus(rows: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    focus: list[str] = []
    for row in sorted(rows or [], key=lambda item: _number(item.get("score")), reverse=True):
        key = str(row.get("key") or "")
        label = DRIVER_FOCUS.get(key)
        if label and label not in focus:
            focus.append(label)
        if len(focus) >= limit:
            break
    return focus


def _compact_anchors(rows: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        text = _limit_text(row.get("text"), 100)
        if not text:
            continue
        compact.append({
            "text": text,
            "kind": row.get("kind") or row.get("type") or row.get("category"),
            "severity": row.get("severity"),
        })
        if len(compact) >= limit:
            break
    return compact


def _compact_footprint(profile: dict[str, Any]) -> dict[str, Any]:
    profile = profile if isinstance(profile, dict) else {}
    top_windows = []
    for window in profile.get("top_risky_windows") or []:
        if not isinstance(window, dict):
            continue
        top_windows.append({
            "window_id": window.get("window_id"),
            "paragraph_id": window.get("paragraph_id"),
            "label": window.get("label"),
            "score": window.get("score"),
            "word_count": window.get("word_count"),
        })
        if len(top_windows) >= 3:
            break
    return {
        "schema_version": profile.get("schema_version"),
        "fraction_ai": profile.get("fraction_ai"),
        "fraction_ai_assisted": profile.get("fraction_ai_assisted"),
        "fraction_human": profile.get("fraction_human"),
        "risky_window_density": profile.get("risky_window_density"),
        "risky_window_count": profile.get("risky_window_count"),
        "high_confidence_risky_window_count": profile.get("high_confidence_risky_window_count"),
        "max_risky_window_words": profile.get("max_risky_window_words"),
        "top_risky_windows": top_windows,
    }


def _compact_problem_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    inventory = inventory if isinstance(inventory, dict) else {}
    groups = []
    for group in inventory.get("problem_groups") or []:
        if not isinstance(group, dict):
            continue
        groups.append({
            "group_id": group.get("group_id"),
            "scope_level": group.get("scope_level"),
            "problem_shape": group.get("problem_shape"),
            "target_ids": list(group.get("target_ids") or [])[:8],
            "anchor_pressure": group.get("anchor_pressure"),
            "semantic_edit_cost": group.get("semantic_edit_cost"),
            "allowed_operations": list(group.get("allowed_operations") or [])[:4],
            "blocked_operations": list(group.get("blocked_operations") or [])[:4],
        })
        if len(groups) >= 8:
            break
    return {
        "schema_version": inventory.get("schema_version"),
        "problem_groups": groups,
    }


def _global_context_summary(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "content_mode": context.get("content_mode"),
        "strategy_family": context.get("strategy_family"),
        "ai_footprint_profile": _compact_footprint(context.get("ai_footprint_profile") or {}),
        "problem_inventory": _compact_problem_inventory(context.get("problem_inventory") or {}),
        "target_profile_summary": context.get("target_profile_summary") or {},
    }


def _planner_context(context: dict[str, Any]) -> dict[str, Any]:
    source_limit = _int_env("DRAFTPROOF_REWRITE_V3_PORTFOLIO_PLANNER_SOURCE_CHARS", 160, low=80, high=420)
    compact_groups = []
    for group in context.get("target_groups") or []:
        drivers = _compact_driver_rows(list(group.get("dominant_drivers") or []), limit=2)
        compact_groups.append({
            "group_id": group.get("group_id"),
            "unit_id": group.get("unit_id"),
            "operation": group.get("operation"),
            "source_excerpt": _limit_text(group.get("source_text"), source_limit),
            "dominant_drivers": drivers,
            "protected_anchors": _compact_anchors(list(group.get("protected_anchors") or []), limit=3),
            "soft_guidance_anchors": _compact_anchors(list(group.get("soft_guidance_anchors") or []), limit=3),
            "word_count_guide": group.get("word_count_guide") or {},
            "target_ids": list(group.get("target_ids") or [])[:3],
        })
    return {
        **_global_context_summary(context),
        "target_groups": compact_groups,
    }


def _plan_rows_for_groups(planner_output: dict[str, Any], group_ids: set[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for row in (planner_output or {}).get("paragraph_plans") or []:
        if not isinstance(row, dict):
            continue
        group_id = str(row.get("group_id") or "")
        if group_ids is not None and group_id not in group_ids:
            continue
        rows.append({
            "group_id": group_id,
            "paragraph_role": row.get("paragraph_role"),
            "risk_drivers": list(row.get("risk_drivers") or [])[:4],
            "hard_anchors": list(row.get("hard_anchors") or [])[:4],
            "soft_anchors": list(row.get("soft_anchors") or [])[:5],
            "repeated_patterns": [
                _limit_text(item, 100)
                for item in list(row.get("repeated_patterns") or [])[:3]
            ],
            "recommended_operator": row.get("recommended_operator"),
            "rewrite_aggression_limit": row.get("rewrite_aggression_limit"),
        })
    return rows


def _plan_by_group(planner_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("group_id") or ""): row
        for row in _plan_rows_for_groups(planner_output)
        if str(row.get("group_id") or "")
    }


def _execution_contract(group: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    operator = str(plan.get("recommended_operator") or "PRESERVE_FACTUAL_CONTEXT")
    contract = OPERATION_CONTRACTS.get(operator) or OPERATION_CONTRACTS["PRESERVE_FACTUAL_CONTEXT"]
    drivers = _compact_driver_rows(list(group.get("dominant_drivers") or []), limit=3)
    return {
        "operator": operator,
        "risk_focus": _driver_focus(drivers),
        "movement": contract["movement"],
        "method": contract["method"],
        "style_boundary": [
            "Use direct plain prose.",
            "Keep simple source wording when it is already direct.",
            "Avoid formulaic transition openings; start from the paragraph's source subject where possible.",
            "Do not reuse the same kind of opener across adjacent paragraphs.",
            "Do not upgrade the source into elevated formal wording.",
            "Do not use tidy textbook phrasing when the source can stay simpler.",
            "Do not use metaphorical flourish unless it is already supported by the source.",
        ],
    }


def _target_sentence_ids(group: TargetGroup) -> list[str]:
    values: list[Any] = []
    for target in group.targets:
        values.extend(target.get("sentence_ids") or [])
    return [
        str(item)
        for item in _unique_ordered_text(values)
        if str(item or "")
    ]


def _target_paragraph_ids(group: TargetGroup) -> list[str]:
    values: list[Any] = [group.unit_id]
    for target in group.targets:
        values.extend([
            target.get("paragraph_id"),
            target.get("parent_unit_id"),
            target.get("unit_id"),
        ])
    return [
        str(item)
        for item in _unique_ordered_text(values)
        if str(item or "")
    ]


def _unique_ordered_text(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _reconstruction_context(context: dict[str, Any], planner_output: dict[str, Any]) -> dict[str, Any]:
    before_after_limit = _int_env("DRAFTPROOF_REWRITE_V3_PORTFOLIO_CONTEXT_CHARS", 100, low=40, high=240)
    compact_groups = []
    group_ids = set()
    plans = _plan_by_group(planner_output)
    for group in context.get("target_groups") or []:
        group_id = str(group.get("group_id") or "")
        group_ids.add(group_id)
        plan = plans.get(group_id, {})
        protected_anchors = list(group.get("protected_anchors") or [])
        required_protected_anchors = list(required_protected_anchors_for_source(
            str(group.get("source_text") or ""),
            protected_anchors,
        ))
        required_anchor_text = {str(anchor.get("text") or "") for anchor in required_protected_anchors}
        out_of_scope_protected_anchors = [
            anchor for anchor in protected_anchors
            if str(anchor.get("text") or "") and str(anchor.get("text") or "") not in required_anchor_text
        ]
        compact_groups.append({
            "group_id": group_id,
            "unit_id": group.get("unit_id"),
            "operation": group.get("operation"),
            "source_text": group.get("source_text"),
            "before_context": _limit_text(group.get("before_context"), before_after_limit),
            "after_context": _limit_text(group.get("after_context"), before_after_limit),
            "execution_contract": _execution_contract(group, plan),
            "required_movement": group.get("required_movement") or {},
            "protected_anchors": _compact_anchors(protected_anchors, limit=4),
            "required_protected_anchors": _compact_anchors(required_protected_anchors, limit=4),
            "out_of_scope_protected_anchors": _compact_anchors(out_of_scope_protected_anchors, limit=4),
            "soft_guidance_anchors": _compact_anchors(list(group.get("soft_guidance_anchors") or []), limit=5),
            "word_count_guide": group.get("word_count_guide") or {},
            "target_ids": list(group.get("target_ids") or [])[:3],
        })
    summary = _global_context_summary(context)
    return {
        "content_mode": summary.get("content_mode"),
        "strategy_family": summary.get("strategy_family"),
        "target_profile_summary": summary.get("target_profile_summary"),
        "footprint_summary": summary.get("ai_footprint_profile"),
        "target_groups": compact_groups,
        "paragraph_plans": _plan_rows_for_groups(planner_output, group_ids),
    }


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for index, char in enumerate(text or ""):
        if char not in ".!?":
            continue
        end = index + 1
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    tail = (text or "")[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _problem_tokens_from_spans(spans: list[str], *, extra_tokens: list[Any] | None = None, limit: int = 10) -> list[str]:
    tokens: list[Any] = []
    tokens.extend(extra_tokens or [])
    for span in spans:
        tokens.extend(str(span or "").replace(",", " ").replace(";", " ").replace(":", " ").split())
    return _unique_ordered_text(tokens)[:limit]


def _fallback_predictable_spans(text: str) -> list[str]:
    candidates: list[tuple[float, str]] = []
    for sentence in _split_sentences(text):
        words = sentence.split()
        if len(words) < 6:
            continue
        punctuation_weight = sentence.count(",") * 1.5 + sentence.count(":") * 2.0 + sentence.count(";") * 2.0
        length_weight = min(len(words), 24) / 24
        candidates.append((punctuation_weight + length_weight, sentence))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return _unique_ordered_text([sentence for _, sentence in candidates])[:3]


def _brief_matches_group(brief: dict[str, Any], group: dict[str, Any]) -> bool:
    sentence_ids = set(str(item) for item in group.get("sentence_ids") or [] if str(item or ""))
    paragraph_ids = set(str(item) for item in group.get("paragraph_ids") or [] if str(item or ""))
    brief_sentence_ids = set(str(item) for item in brief.get("sentence_ids") or [] if str(item or ""))
    if str(brief.get("sentence_id") or ""):
        brief_sentence_ids.add(str(brief.get("sentence_id")))
    brief_paragraph_ids = set(str(item) for item in brief.get("paragraph_ids") or [] if str(item or ""))
    if str(brief.get("paragraph_id") or ""):
        brief_paragraph_ids.add(str(brief.get("paragraph_id")))
    if sentence_ids and brief_sentence_ids and sentence_ids.intersection(brief_sentence_ids):
        return True
    if paragraph_ids and brief_paragraph_ids and paragraph_ids.intersection(brief_paragraph_ids):
        return True
    source_text = str(group.get("source_text") or "")
    target_sentence = str(brief.get("target_sentence") or "").strip()
    if target_sentence and target_sentence in source_text:
        return True
    return any(str(span or "").strip() and str(span).strip() in source_text for span in brief.get("predictable_token_spans") or [])


def _topk_targets_for_group(group: dict[str, Any], replacement_text: str, briefs: list[dict[str, Any]]) -> dict[str, Any]:
    spans: list[Any] = []
    problem_tokens: list[Any] = []
    source_sentences: list[Any] = []
    span_source = "scanner_exact"
    for brief in briefs:
        if not _brief_matches_group(brief, group):
            continue
        spans.extend(brief.get("predictable_token_spans") or [])
        problem_tokens.extend(brief.get("problem_tokens") or [])
        sentence = str(brief.get("target_sentence") or "").strip()
        if sentence:
            source_sentences.append(_limit_text(sentence, 180))
    raw_predictable_spans = [
        span for span in _unique_ordered_text(spans)
        if str(span or "") and str(span) in replacement_text
    ]
    phrase_payload = phrase_level_spans(raw_predictable_spans, replacement_text, limit=6)
    predictable_spans = phrase_payload["predictable_spans"]
    if not predictable_spans:
        span_source = "structural_fallback"
        predictable_spans = _fallback_predictable_spans(replacement_text)
    preferred_words = int((group.get("word_count_guide") or {}).get("preferred_words") or 0)
    max_changed_words = max(10, min(24, round((preferred_words or max(1, len(replacement_text.split()))) * 0.28)))
    max_changed_spans = max(1, min(3, len(predictable_spans) or 1))
    high_quality_spans = [span for span in predictable_spans if len(span) >= 12 and len(span.split()) >= 3]
    required_modified_spans = min(2, len(high_quality_spans)) if span_source == "scanner_exact" else 1
    if span_source == "scanner_exact" and preferred_words < 60:
        required_modified_spans = min(1, len(high_quality_spans))
    required_modified_spans = max(1, required_modified_spans) if predictable_spans else 1
    return {
        "span_source": span_source,
        "raw_predictable_spans": raw_predictable_spans[:6],
        "predictable_spans": predictable_spans,
        "predictable_spans_in_source": predictable_spans,
        "predictable_span_rows": span_rows(predictable_spans),
        "rejected_predictable_spans": phrase_payload["rejected_predictable_spans"],
        "expanded_predictable_spans": phrase_payload["expanded_predictable_spans"],
        "required_modified_spans": required_modified_spans,
        "source_sentences": _unique_ordered_text(source_sentences)[:3],
        "problem_tokens": _problem_tokens_from_spans(predictable_spans, extra_tokens=problem_tokens),
        "avoid_phrases": predictable_spans[:6],
        "locality_limits": {
            "max_changed_spans": max_changed_spans,
            "max_changed_words": max_changed_words,
            "max_sentence_changes": min(2, max_changed_spans),
        },
        "allowed_operations": [
            "CLAUSE_ROUTE_CHANGE",
            "DELETE_EMPTY_PHRASE",
            "LIST_BREAK",
            "CONCRETE_SOURCE_WORDING",
            "TOPK_SPAN_REPATH",
        ],
    }


def _topk_context(
    context: dict[str, Any],
    planner_output: dict[str, Any],
    replacements: list[dict[str, str]],
) -> dict[str, Any]:
    compact_groups = []
    group_ids = set()
    replacements_by_group = {
        str(row.get("group_id") or ""): str(row.get("replacement_text") or "")
        for row in replacements or []
        if isinstance(row, dict)
    }
    briefs = [
        row for row in context.get("predictability_briefs") or []
        if isinstance(row, dict)
    ]
    for group in context.get("target_groups") or []:
        group_id = str(group.get("group_id") or "")
        group_ids.add(group_id)
        target_contract = _topk_targets_for_group(group, replacements_by_group.get(group_id, ""), briefs)
        compact_groups.append({
            "group_id": group_id,
            "dominant_drivers": _compact_driver_rows(list(group.get("dominant_drivers") or []), limit=2),
            "required_movement": group.get("required_movement") or {},
            "protected_anchors": _compact_anchors(list(group.get("protected_anchors") or []), limit=4),
            "word_count_guide": group.get("word_count_guide") or {},
            "topk_repair_contract": target_contract,
        })
    return {
        "target_profile_summary": (context.get("target_profile_summary") or {}),
        "target_groups": compact_groups,
        "paragraph_plans": _plan_rows_for_groups(planner_output, group_ids),
    }


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _driver_rows(group: TargetGroup) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in group.targets:
        for driver in target.get("dominant_drivers") or []:
            if isinstance(driver, dict):
                rows.append({
                    "key": driver.get("key"),
                    "score": driver.get("score"),
                    "label": driver.get("label"),
                })
    return rows


def _required_movement(group: TargetGroup) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for target in group.targets:
        movement = target.get("required_movement") if isinstance(target.get("required_movement"), dict) else {}
        for key, value in movement.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                combined[key] = max(float(combined.get(key, 0.0)), float(value))
            elif key not in combined:
                combined[key] = value
    return combined


def _target_ids(group: TargetGroup) -> list[str]:
    return [
        str(target.get("target_id") or "")
        for target in group.targets
        if str(target.get("target_id") or "")
    ]


def paragraph_portfolio_context(
    *,
    target_groups: list[TargetGroup],
    scan_contract: Any,
    content_mode: str,
    strategy_family: str,
) -> dict[str, Any]:
    return {
        "content_mode": content_mode,
        "strategy_family": strategy_family,
        "ai_footprint_profile": getattr(scan_contract, "ai_footprint_profile", {}) or {},
        "problem_inventory": getattr(scan_contract, "problem_inventory", {}) or {},
        "target_profile_summary": {
            "document_shape": (getattr(scan_contract, "rewrite_target_profile", {}) or {}).get("document_shape"),
            "driver_summary": getattr(scan_contract, "target_driver_summary", {}) or {},
            "operation_mix": getattr(scan_contract, "target_operation_mix", {}) or {},
            "target_scope_policy": getattr(scan_contract, "target_scope_policy", ""),
        },
        "target_groups": [
            {
                "group_id": group.group_id,
                "unit_id": group.unit_id,
                "operation": group.operation,
                "source_text": group.source_text,
                "before_context": group.before_context[-360:],
                "after_context": group.after_context[:360],
                "dominant_drivers": _driver_rows(group),
                "required_movement": _required_movement(group),
                "protected_anchors": list(group.protected_anchors),
                "soft_guidance_anchors": list(group.soft_guidance_anchors),
                "word_count_guide": dict(group.word_count_guide),
                "target_ids": _target_ids(group),
                "sentence_ids": _target_sentence_ids(group),
                "paragraph_ids": _target_paragraph_ids(group),
            }
            for group in target_groups
        ],
        "predictability_briefs": list(getattr(scan_contract, "predictability_briefs", ()) or ()),
    }


def build_paragraph_portfolio_planner_prompt(context: dict[str, Any]) -> PromptTemplatePayload:
    payload = {
        "template_id": TEMPLATE_ID,
        "strategy_id": STRATEGY_ID,
        "prompt_stage": "planner",
        "task": "Analyze the scanner-targeted paragraph portfolio. Do not rewrite.",
        "role_choices": list(ROLE_CHOICES),
        "operator_choices": list(OPERATOR_CHOICES),
        "scanner_context": _planner_context(context),
        "rules": [
            "Return JSON only.",
            "Do not rewrite prose in this stage.",
            "Use only the provided compact scanner context and source excerpts.",
            "Separate protected hard anchors from soft guidance anchors.",
            "Identify repeated prose patterns from the provided source_text, not from a fixed phrase list.",
            "Choose operators from operator_choices only.",
        ],
        "response_schema": {
            "paragraph_plans": [
                {
                    "group_id": "tg001",
                    "paragraph_role": "opening_frame",
                    "risk_drivers": ["driver keys or labels"],
                    "hard_anchors": ["exact anchors that must be preserved"],
                    "soft_anchors": ["topics or terms to keep as coverage hints"],
                    "repeated_patterns": ["patterns observed in source_text"],
                    "recommended_operator": "BREAK_SURVEY_TEMPLATE",
                    "rewrite_aggression_limit": "low|medium|high",
                }
            ]
        },
    }
    return PromptTemplatePayload(
        template_id=TEMPLATE_ID,
        strategy_id=STRATEGY_ID,
        prompt_stage="planner",
        prompt=_json_payload("You are a V3 rewrite planning module, not a prose writer.", payload),
        scanner_context_used=(
            "ai_footprint_profile",
            "problem_inventory",
            "rewrite_target_profile.targets",
            "dominant_drivers",
            "required_movement",
            "hard_anchors",
            "soft_guidance_anchors",
            "word_count_guide",
        ),
    )


def parse_paragraph_portfolio_plan(raw: str) -> dict[str, Any]:
    payload = _parse_json_object(raw)
    rows = payload.get("paragraph_plans") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {"paragraph_plans": []}
    plans: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        group_id = str(row.get("group_id") or "").strip()
        if not group_id:
            continue
        operator = str(row.get("recommended_operator") or "").strip()
        role = str(row.get("paragraph_role") or "unknown_unit").strip()
        plans.append({
            "group_id": group_id,
            "paragraph_role": role if role in ROLE_CHOICES else "unknown_unit",
            "risk_drivers": list(row.get("risk_drivers") or []),
            "hard_anchors": list(row.get("hard_anchors") or []),
            "soft_anchors": list(row.get("soft_anchors") or []),
            "repeated_patterns": list(row.get("repeated_patterns") or []),
            "recommended_operator": operator if operator in OPERATOR_CHOICES else "PRESERVE_FACTUAL_CONTEXT",
            "rewrite_aggression_limit": str(row.get("rewrite_aggression_limit") or "medium"),
        })
    return {"paragraph_plans": plans}


def validate_paragraph_portfolio_plan(plan: dict[str, Any], target_groups: list[TargetGroup]) -> dict[str, Any]:
    expected = {group.group_id for group in target_groups}
    rows = plan.get("paragraph_plans") if isinstance(plan, dict) else []
    actual = {
        str(row.get("group_id") or "")
        for row in rows or []
        if isinstance(row, dict)
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return {
        "passed": not missing and not extra,
        "missing_group_ids": missing,
        "extra_group_ids": extra,
        "expected_group_count": len(expected),
        "actual_group_count": len(actual),
    }


def fallback_paragraph_portfolio_plan(target_groups: list[TargetGroup]) -> dict[str, Any]:
    plans: list[dict[str, Any]] = []
    for group in target_groups:
        plans.append({
            "group_id": group.group_id,
            "paragraph_role": "unknown_unit",
            "risk_drivers": [
                str(driver.get("key") or driver.get("label") or "")
                for driver in _driver_rows(group)
                if str(driver.get("key") or driver.get("label") or "")
            ],
            "hard_anchors": [
                str(anchor.get("text") or "")
                for anchor in group.protected_anchors
                if str(anchor.get("text") or "")
            ],
            "soft_anchors": [
                str(anchor.get("text") or "")
                for anchor in group.soft_guidance_anchors
                if str(anchor.get("text") or "")
            ],
            "repeated_patterns": [],
            "recommended_operator": "BREAK_SURVEY_TEMPLATE",
            "rewrite_aggression_limit": "medium",
        })
    return {"paragraph_plans": plans}


def build_paragraph_portfolio_reconstruction_prompt(
    context: dict[str, Any],
    planner_output: dict[str, Any],
) -> PromptTemplatePayload:
    payload = {
        "template_id": TEMPLATE_ID,
        "strategy_id": STRATEGY_ID,
        "prompt_stage": "paragraph_reconstruction",
        "task": "Rewrite each target paragraph according to its plan and scanner context.",
        "scanner_context": _reconstruction_context(context, planner_output),
        "rules": [
            "Return JSON only with a replacements array.",
            "Return one replacement for every target group.",
            "Rewrite only the source_text for each target group.",
            "Preserve hard anchors exactly.",
            "Copy each required_protected_anchors.text exactly as provided, including punctuation and quote style.",
            "Do not force out_of_scope_protected_anchors into replacement_text; they are shown only for diagnostics and are not part of this target source_text.",
            "Use soft anchors as coverage hints, not exact strings.",
            "Preserve factual meaning and paragraph role.",
            "Do not add unsupported facts, sources, names, dates, numbers, headings, bullets, markdown, labels, or commentary.",
            "Use word_count_guide as a preferred length guide, not a min/max band.",
            "Do not compress the paragraph into a summary.",
            "Follow execution_contract.movement and execution_contract.method.",
            "Do not solve the task by synonym swapping or elevated paraphrase.",
            "Keep local continuity with before_context and after_context.",
            "Escape any straight quotation marks inside JSON string values.",
            "Do not add optional quotation marks inside replacement_text. If a quoted phrase is not a protected hard anchor, write it without quote marks so the JSON remains valid.",
        ],
        "response_schema": {
            "replacements": [
                {
                    "group_id": "tg001",
                    "replacement_text": "replacement paragraph only",
                }
            ]
        },
    }
    return PromptTemplatePayload(
        template_id=TEMPLATE_ID,
        strategy_id=STRATEGY_ID,
        prompt_stage="paragraph_reconstruction",
        prompt=_json_payload("You are a bounded paragraph reconstruction engine.", payload),
        scanner_context_used=(
            "planner_output",
            "source_text",
            "before_context",
            "after_context",
            "execution_contract",
            "required_movement",
            "hard_anchors",
            "soft_guidance_anchors",
            "word_count_guide",
        ),
    )


def parse_paragraph_portfolio_replacements(raw: str) -> list[dict[str, str]]:
    payload = _parse_json_object(raw)
    rows = payload.get("replacements") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    replacements: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        group_id = str(row.get("group_id") or "").strip()
        replacement = str(row.get("replacement_text") or "").strip()
        if group_id and replacement:
            replacements.append({"group_id": group_id, "replacement_text": replacement})
    return replacements


def build_paragraph_portfolio_topk_prompt(
    *,
    context: dict[str, Any],
    planner_output: dict[str, Any],
    replacements: list[dict[str, str]],
) -> PromptTemplatePayload:
    payload = {
        "template_id": TEMPLATE_ID,
        "strategy_id": STRATEGY_ID,
        "prompt_stage": "topk_repair",
        "task": "Patch only the reconstructed replacement paragraphs where scanner drivers indicate predictable wording.",
        "scanner_context": _topk_context(context, planner_output, replacements),
        "planner_output": {
            "paragraph_plans": _plan_rows_for_groups(planner_output),
        },
        "current_replacements": replacements,
        "rules": [
            "Return JSON only with a replacements array.",
            "Return one row per replacement you modify; unchanged rows may be omitted.",
            "Do not rewrite the full document.",
            "Do not introduce unsupported facts, sources, names, dates, numbers, headings, bullets, markdown, labels, or commentary.",
            "Preserve hard anchors exactly.",
            "Patch only topk_repair_contract.predictable_spans_in_source and their local wording path.",
            "When predictable_span_rows are present, report modified_span_ids using those exact ids.",
            "Do not guess changed span counts; count a span only when changed_spans.source_span exactly equals or fully contains one predictable_span_rows.text item.",
            "Use raw_predictable_spans and rejected_predictable_spans only as diagnostics; do not count them as repaired spans.",
            "Modify at least topk_repair_contract.required_modified_spans phrase spans when span_source is scanner_exact.",
            "Use only clause movement, deletion of empty phrasing, list breaking, or concrete source-supported wording.",
            "Do not perform a full sentence rewrite unless the entire sentence is listed as a predictable_span.",
            "Stay inside topk_repair_contract.locality_limits.",
        ],
        "response_schema": {
            "replacements": [
                {
                    "group_id": "tg001",
                    "replacement_text": "patched replacement paragraph only",
                    "changed_spans": [
                        {"span_id": "ps001", "before": "old local phrase", "after": "new local phrase", "operation": "TOPK_SPAN_REPATH"}
                    ],
                    "modified_span_ids": ["ps001"],
                    "predictable_spans_modified_count": 0,
                    "new_claims_added": False,
                    "hard_anchors_preserved": True,
                    "changed_word_estimate": 0,
                }
            ]
        },
    }
    return PromptTemplatePayload(
        template_id=TEMPLATE_ID,
        strategy_id=STRATEGY_ID,
        prompt_stage="topk_repair",
        prompt=_json_payload("You are a local predictability repair module.", payload),
        scanner_context_used=(
            "planner_output",
            "current_replacements",
            "dominant_drivers",
            "predictability_briefs",
            "predictable_spans",
            "problem_tokens",
            "locality_limits",
            "required_movement",
            "hard_anchors",
        ),
    )
