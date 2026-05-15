"""Scanner-controlled local search executor for rewrite V3.

This module turns scanner findings into an executable bounded search:
rank target groups from scanner evidence, ask for local variants, and let the
pipeline rescan each applied variant before accepting it. The code uses typed
scanner fields only; it does not route from content keywords.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .document_units import word_count
from .prompt_contract import group_action_contract
from .scanner_contract import predictability_briefs_from_report
from .target_executor import TargetGroup


DRIVER_FOCUS = {
    "predictability_score": "predictable sentence route",
    "ai_signal_score": "generated overview texture",
    "ai_likelihood": "model-like paragraph movement",
    "unsafe_word_share": "weak process reasoning / weak cause-effect ownership",
}

VARIANT_OPERATORS = (
    "CLAUSE_ROUTE_CHANGE",
    "BROAD_CLAIM_NARROWING",
    "CAUSE_EFFECT_OWNERSHIP",
    "CLAIM_OWNERSHIP_REPAIR",
)


@dataclass(frozen=True)
class ScannerControlledConfig:
    max_rounds: int = 2
    groups_per_round: int = 4
    variants_per_group: int = 3
    min_accept_delta: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _scan_intelligence(report: dict[str, Any] | None) -> dict[str, Any]:
    payload = report if isinstance(report, dict) else {}
    scan_intel = payload.get("scan_intelligence")
    return scan_intel if isinstance(scan_intel, dict) else {}


def _target_profile(report: dict[str, Any] | None) -> dict[str, Any]:
    payload = report if isinstance(report, dict) else {}
    candidates = [
        payload.get("rewrite_target_profile"),
        ((_scan_intelligence(payload).get("document") or {}).get("rewrite_target_profile")),
        _scan_intelligence(payload).get("rewrite_target_profile"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def scanner_controlled_metrics(
    *,
    report: dict[str, Any],
    goal: dict[str, Any],
    footprint_risk: float,
    ai_score: float | None,
    topk_score: float | None,
) -> dict[str, Any]:
    footprint = report.get("ai_footprint_profile")
    if not isinstance(footprint, dict):
        footprint = ((_scan_intelligence(report).get("document") or {}).get("ai_footprint_profile"))
    if not isinstance(footprint, dict):
        footprint = {}
    eligible = goal.get("eligible_span_density_gate") if isinstance(goal.get("eligible_span_density_gate"), dict) else {}
    proxy = goal.get("external_detector_proxy") if isinstance(goal.get("external_detector_proxy"), dict) else {}
    targets = _target_profile(report).get("targets")
    target_count = len(targets) if isinstance(targets, list) else 0
    return {
        "ai": round(_number(ai_score), 3),
        "topk": round(_number(topk_score), 3),
        "footprint_risk": round(_number(footprint_risk), 3),
        "fraction_ai": footprint.get("fraction_ai"),
        "fraction_ai_assisted": footprint.get("fraction_ai_assisted"),
        "risky_window_density": footprint.get("risky_window_density"),
        "risky_window_count": footprint.get("risky_window_count"),
        "assisted_window_count": footprint.get("assisted_window_count"),
        "unsafe_cluster_count": eligible.get("unsafe_cluster_count"),
        "unsafe_word_ratio": eligible.get("unsafe_eligible_word_ratio"),
        "external_proxy_score": proxy.get("score"),
        "external_proxy_safe": proxy.get("safe"),
        "target_count": target_count,
    }


def scanner_controlled_rank(metrics: dict[str, Any]) -> float:
    """Lower is better. Weight hard scanner blockers above cosmetic badge drops."""

    return round(
        _number(metrics.get("footprint_risk"))
        + _number(metrics.get("external_proxy_score")) * 0.35
        + _number(metrics.get("topk")) * 0.05
        + _number(metrics.get("ai")) * 0.08
        + _number(metrics.get("unsafe_cluster_count")) * 0.9
        + _number(metrics.get("unsafe_word_ratio")) * 0.12,
        3,
    )


def blocker_rows_for_group(report: dict[str, Any], group: TargetGroup, *, limit: int = 6) -> list[dict[str, Any]]:
    radar = _scan_intelligence(report).get("blocker_radar")
    radar_payload = radar if isinstance(radar, dict) else {}
    rows = radar_payload.get("dominant_blockers") if isinstance(radar_payload.get("dominant_blockers"), list) else []
    paragraph_id = str(group.unit_id or "")
    sentence_ids: set[str] = set()
    for target in group.targets:
        if not isinstance(target, dict):
            continue
        sentence_ids.update(str(item) for item in (target.get("sentence_ids") or []))

    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        paragraph_ids = {str(item) for item in (row.get("paragraph_ids") or [])}
        blocker_sentence_ids = {str(item) for item in (row.get("sentence_ids") or [])}
        if (
            paragraph_id in paragraph_ids
            or sentence_ids.intersection(blocker_sentence_ids)
            or str(row.get("scope") or "") == "document_wide"
        ):
            selected.append({
                "key": row.get("key"),
                "severity": row.get("severity"),
                "score": row.get("score"),
                "scope": row.get("scope"),
                "diagnostic_flags": row.get("diagnostic_flags"),
                "paragraph_ids": list(row.get("paragraph_ids") or [])[:8],
                "sentence_ids": list(row.get("sentence_ids") or [])[:8],
                "diagnostic": row.get("diagnostic"),
            })
        if len(selected) >= limit:
            break
    return selected


def weak_human_levers(report: dict[str, Any], *, limit: int = 4) -> list[dict[str, Any]]:
    contract = _scan_intelligence(report).get("human_contribution_contract")
    payload = contract if isinstance(contract, dict) else {}
    weak = {str(item) for item in (payload.get("weak_subsignals") or [])}
    rows: list[dict[str, Any]] = []
    for row in payload.get("subsignals") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("key") or "") in weak or str(row.get("label") or "") == "weak":
            rows.append({
                "key": row.get("key"),
                "score": row.get("score"),
                "label": row.get("label"),
                "rewrite_lever": row.get("rewrite_lever"),
            })
        if len(rows) >= limit:
            break
    return rows


def _driver_focus_for_group(group: TargetGroup, *, limit: int = 4) -> list[str]:
    focus: list[str] = []
    for target in group.targets:
        if not isinstance(target, dict):
            continue
        for driver in sorted(target.get("dominant_drivers") or [], key=lambda item: _number(item.get("score")) if isinstance(item, dict) else 0.0, reverse=True):
            if not isinstance(driver, dict):
                continue
            label = DRIVER_FOCUS.get(str(driver.get("key") or ""))
            if label and label not in focus:
                focus.append(label)
            if len(focus) >= limit:
                return focus
    return focus


def protected_anchor_placeholders(group: TargetGroup) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for index, anchor in enumerate(group.protected_anchors, start=1):
        if not isinstance(anchor, dict):
            continue
        text = str(anchor.get("text") or "")
        if text:
            anchors.append({
                "placeholder": f"[[DP_ANCHOR_{index:03d}]]",
                "text": text,
                "kind": anchor.get("kind"),
                "blocking": bool(anchor.get("blocking")),
            })
    return anchors


def _placeholder_ranges(text: str) -> list[tuple[int, int]]:
    value = str(text or "")
    ranges: list[tuple[int, int]] = []
    marker = "[[DP_ANCHOR_"
    search_from = 0
    while True:
        start = value.find(marker, search_from)
        if start < 0:
            break
        end = value.find("]]", start)
        if end < 0:
            ranges.append((start, len(value)))
            break
        ranges.append((start, end + 2))
        search_from = end + 2
    return ranges


def _inside_ranges(index: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in ranges)


def _anchor_boundary_ok(text: str, start: int, end: int, source: str) -> bool:
    if not source:
        return False
    if source.isdigit():
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        return not before.isdigit() and not after.isdigit()
    return True


def freeze_protected_anchors(text: str, anchors: list[dict[str, Any]]) -> str:
    original = str(text or "")
    placeholder_ranges = _placeholder_ranges(original)
    candidates: list[tuple[int, int, str]] = []
    for anchor in sorted(anchors, key=lambda row: len(str(row.get("text") or "")), reverse=True):
        source = str(anchor.get("text") or "")
        placeholder = str(anchor.get("placeholder") or "")
        if not source or not placeholder:
            continue
        search_from = 0
        while True:
            start = original.find(source, search_from)
            if start < 0:
                break
            end = start + len(source)
            if not _inside_ranges(start, placeholder_ranges) and _anchor_boundary_ok(original, start, end, source):
                candidates.append((start, end, placeholder))
            search_from = max(end, start + 1)

    selected: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, placeholder in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(not (end <= used_start or start >= used_end) for used_start, used_end in occupied):
            continue
        selected.append((start, end, placeholder))
        occupied.append((start, end))
    if not selected:
        return original

    pieces: list[str] = []
    cursor = 0
    for start, end, placeholder in selected:
        pieces.append(original[cursor:start])
        pieces.append(placeholder)
        cursor = end
    pieces.append(original[cursor:])
    return "".join(pieces)


def protected_placeholder_integrity(text: str, expected_placeholders: list[str] | None = None) -> dict[str, Any]:
    value = str(text or "")
    failures: list[str] = []
    placeholders: list[str] = []
    marker = "[[DP_ANCHOR_"
    search_from = 0
    while True:
        start = value.find(marker, search_from)
        if start < 0:
            break
        end = value.find("]]", start)
        if end < 0:
            failures.append("dangling_anchor_placeholder")
            break
        token = value[start:end + 2]
        placeholders.append(token)
        inner = token[len(marker):-2]
        if len(inner) != 3 or not inner.isdigit():
            failures.append("malformed_anchor_placeholder")
        if token.find(marker, 2) >= 0:
            failures.append("nested_anchor_placeholder")
        search_from = end + 2

    expected = [str(item) for item in expected_placeholders or [] if str(item or "")]
    missing = [placeholder for placeholder in expected if placeholder not in value]
    if missing:
        failures.append("anchor_placeholder_missing")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "placeholders": placeholders,
        "missing_placeholders": missing,
    }


def restore_protected_anchor_placeholders(text: str, group: TargetGroup) -> str:
    restored = str(text or "")
    for anchor in protected_anchor_placeholders(group):
        placeholder = str(anchor.get("placeholder") or "")
        source = str(anchor.get("text") or "")
        if placeholder and source:
            restored = restored.replace(placeholder, source)
    return restored


def _movement_contract(report: dict[str, Any], group: TargetGroup) -> dict[str, Any]:
    blockers = blocker_rows_for_group(report, group, limit=5)
    flags = {
        "texture_pressure": False,
        "evidence_gap": False,
        "source_dependency": False,
        "author_context_gap": False,
    }
    blocker_focus: list[str] = []
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        key = str(blocker.get("key") or "")
        if key and key not in blocker_focus:
            blocker_focus.append(key)
        diagnostic_flags = blocker.get("diagnostic_flags") if isinstance(blocker.get("diagnostic_flags"), dict) else {}
        for flag in flags:
            flags[flag] = bool(flags[flag] or diagnostic_flags.get(flag))
    operations: list[str] = []
    if flags["texture_pressure"] or "predictable sentence route" in _driver_focus_for_group(group):
        operations.append("Change sentence route and relation, not synonyms.")
    if flags["evidence_gap"] or flags["author_context_gap"]:
        operations.append("Narrow broad claims or attach them to reasoning already present in source_text.")
    if flags["source_dependency"]:
        operations.append("Make the source-to-claim relation explicit without adding new evidence.")
    if not operations:
        operations.append("Keep meaning stable while making the paragraph less polished and less generic.")
    return {
        "risk_focus": _driver_focus_for_group(group),
        "blocker_focus": blocker_focus[:5],
        "operations": operations,
        "variant_plan": [
            {"variant_id": f"v{index}", "operator": operator}
            for index, operator in enumerate(VARIANT_OPERATORS, start=1)
        ],
        "style_boundary": [
            "Use direct plain prose.",
            "Keep simple source wording when it is already direct.",
            "Avoid formulaic transition openings.",
            "Do not upgrade the source into elevated formal wording.",
            "Do not use tidy textbook phrasing when the source can stay simpler.",
        ],
    }


def rank_scanner_target_groups(
    *,
    report: dict[str, Any],
    goal: dict[str, Any],
    groups: list[TargetGroup],
) -> list[TargetGroup]:
    eligible = goal.get("eligible_span_density_gate") if isinstance(goal.get("eligible_span_density_gate"), dict) else {}
    top_sentence_ids = {
        str(row.get("sentence_id"))
        for row in (eligible.get("top_sentence_targets") or [])
        if isinstance(row, dict)
    }
    radar = _scan_intelligence(report).get("blocker_radar")
    radar_payload = radar if isinstance(radar, dict) else {}
    blocker_scores: dict[str, float] = {}
    for row in radar_payload.get("dominant_blockers") or []:
        if not isinstance(row, dict):
            continue
        score = _number(row.get("score")) / 100.0
        for paragraph_id in row.get("paragraph_ids") or []:
            key = str(paragraph_id)
            blocker_scores[key] = blocker_scores.get(key, 0.0) + score

    def group_score(group: TargetGroup) -> float:
        score = blocker_scores.get(str(group.unit_id), 0.0)
        for target in group.targets:
            if not isinstance(target, dict):
                continue
            target_sentence_ids = {str(item) for item in (target.get("sentence_ids") or [])}
            if top_sentence_ids.intersection(target_sentence_ids):
                score += 2.0
            for driver in target.get("dominant_drivers") or []:
                if isinstance(driver, dict):
                    score += min(_number(driver.get("score")), 1.0)
        score += min(word_count(group.source_text), 80) / 200.0
        return score

    return sorted(groups, key=group_score, reverse=True)


def build_scanner_controlled_prompt(
    *,
    report: dict[str, Any],
    group: TargetGroup,
    variants_per_group: int,
    ownership_repair_mode: bool = False,
) -> str:
    anchors = protected_anchor_placeholders(group)
    operators = (
        ("CLAIM_OWNERSHIP_REPAIR",) * max(1, int(variants_per_group))
        if ownership_repair_mode
        else VARIANT_OPERATORS[: max(1, int(variants_per_group))]
    )
    variant_plan = [
        {"variant_id": f"v{index}", "operator": operator}
        for index, operator in enumerate(operators, start=1)
    ]
    movement_contract = _movement_contract(report, group)
    movement_contract["variant_plan"] = variant_plan
    scanner_action_contract = group_action_contract(
        group=group,
        predictability_briefs=predictability_briefs_from_report(report),
    )
    payload = {
        "group_id": group.group_id,
        "unit_id": group.unit_id,
        "operation": group.operation,
        "source_text": freeze_protected_anchors(group.source_text, anchors),
        "before_context": freeze_protected_anchors(group.before_context[-420:], anchors),
        "after_context": freeze_protected_anchors(group.after_context[:420], anchors),
        "protected_anchors": anchors,
        "word_count_guide": dict(group.word_count_guide),
        "movement_contract": movement_contract,
        "scanner_action_contract": scanner_action_contract,
        "weak_human_levers": [
            {
                "key": row.get("key"),
                "rewrite_lever": row.get("rewrite_lever"),
            }
            for row in weak_human_levers(report)
        ],
        "repair_mode": "claim_ownership_repair" if ownership_repair_mode else "scanner_controlled_span_repair",
        "instructions": [
            f"Create exactly {len(variant_plan)} replacement variants for source_text only.",
            "Each variant must use its assigned movement_contract.variant_plan operator as the main change.",
            "Use movement_contract as the rewrite contract.",
            "Use scanner_action_contract as the span and operation contract.",
            "Use scanner_action_contract.ownership_contract as the ownership contract.",
            "Do not just change point of view; every kept variant must improve author_trace, specific_context, or real_judgment using only source_text, before_context, after_context, anchors, or scanner context.",
            "Do not add fake first-person experience or unsupported anecdotes; preserve source viewpoint unless the source context already supports a viewpoint shift.",
            *(
                [
                    "This is an ownership repair pass: every kept variant must report at least one real ownership_changes row.",
                    "Prioritize clear author trace, specific context, and real judgment over broad paragraph smoothing.",
                    "Only modify local wording needed to make the claim feel owned by the source context.",
                ]
                if ownership_repair_mode
                else []
            ),
            "If scanner_action_contract.citation_pressure_zone is true, keep citation-adjacent wording source-like and do not upgrade it into smoother academic paraphrase.",
            "Patch scanner_action_contract.topk_repair_contract.predictable_spans_in_source when span_source is scanner_exact.",
            "When predictable_span_rows are present, report modified_span_ids using those exact ids.",
            "Do not guess predictable_spans_modified_count; only count a span if changed_spans.source_span exactly equals or fully contains one predictable_span_rows.text item.",
            "Stay inside scanner_action_contract.topk_repair_contract.locality_limits.",
            "Each variant must modify at least scanner_action_contract.topk_repair_contract.required_modified_spans predictable spans when span_source is scanner_exact.",
            "If an operator cannot modify enough predictable_spans_in_source without breaking meaning, omit that variant.",
            "If a variant cannot satisfy its assigned operator, omit it entirely; do not return no-op variants, notes, explanations, or unchanged replacement_text.",
            "Do not intensify source claims, learner ability, certainty, or outcome strength; keep replacement wording at the same strength or weaker unless source_text already supports stronger wording.",
            "Preserve protected anchor placeholders exactly; do not alter placeholder punctuation, brackets, or numbering.",
            "Preserve the same core meaning and claims.",
            "Do not add unsupported facts, sources, names, dates, numbers, headings, bullets, markdown, labels, or commentary.",
            "Keep within +/-15% of preferred_words.",
            "Do not summarize the paragraph, but you may remove broad filler when it improves precision.",
            "Do not use synonym swapping, polished academic smoothing, tidy textbook phrasing, or formulaic transition openings.",
            "Return only JSON matching response_schema.",
        ],
        "response_schema": {
            "variants": [
                {
                    "variant_id": "v1",
                    "operator_used": "CLAUSE_ROUTE_CHANGE",
                    "replacement_text": "plain replacement only",
                    "changed_spans": [
                        {
                            "span_id": "ps001",
                            "source_span": "scanner exact span",
                            "before": "old local phrase",
                            "after": "new local phrase",
                            "operation": "TOPK_SPAN_REPATH",
                        }
                    ],
                    "modified_span_ids": ["ps001"],
                    "predictable_spans_modified_count": 0,
                    "ownership_changes": [
                        {
                            "before": "generic local claim",
                            "after": "owned local judgment",
                            "operation": "CLAIM_OWNERSHIP_REPAIR",
                            "trace_source": "source_text",
                        }
                    ],
                    "ownership_elements_supported": ["author_trace", "specific_context", "real_judgment"],
                    "protected_anchors_preserved": True,
                    "new_claims_added": False,
                }
            ]
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def parse_scanner_controlled_variants(raw: str, *, limit: int = 3) -> list[dict[str, Any]]:
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
        return []
    rows = payload.get("variants") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    variants: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        replacement = str(row.get("replacement_text") or "").strip()
        if not replacement:
            continue
        variant_id = str(row.get("variant_id") or f"v{index}").strip()
        variants.append({
            "variant_id": variant_id,
            "operator_used": row.get("operator_used"),
            "replacement_text": replacement,
            "changed_spans": row.get("changed_spans") if isinstance(row.get("changed_spans"), list) else [],
            "modified_span_ids": row.get("modified_span_ids") if isinstance(row.get("modified_span_ids"), list) else [],
            "predictable_spans_modified_count": row.get("predictable_spans_modified_count"),
            "ownership_changes": row.get("ownership_changes") if isinstance(row.get("ownership_changes"), list) else [],
            "ownership_elements_supported": row.get("ownership_elements_supported") if isinstance(row.get("ownership_elements_supported"), list) else [],
        })
        if len(variants) >= max(1, int(limit or 1)):
            break
    return variants


def _normalized_text(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def _declared_changed_span_count(variant: dict[str, Any]) -> int:
    value = variant.get("predictable_spans_modified_count")
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    count = 0
    for row in variant.get("changed_spans") or []:
        if not isinstance(row, dict):
            continue
        before = _normalized_text(str(row.get("before") or row.get("source_span") or ""))
        after = _normalized_text(str(row.get("after") or ""))
        if before and before != after:
            count += 1
    return count


def _span_source_text(row: dict[str, Any]) -> str:
    return _normalized_text(str(row.get("source_span") or row.get("before") or ""))


def _actual_modified_span_ids(
    *,
    spans: list[dict[str, str]],
    variant: dict[str, Any],
    replacement_text: str,
) -> list[str]:
    changed_rows = [row for row in variant.get("changed_spans") or [] if isinstance(row, dict)]
    changed_sources = [_span_source_text(row) for row in changed_rows]
    changed_ids = {str(row.get("span_id") or "") for row in changed_rows if str(row.get("span_id") or "")}
    reported_ids = {str(item) for item in variant.get("modified_span_ids") or [] if str(item or "")}
    normalized_replacement = _normalized_text(replacement_text)
    actual: list[str] = []
    for row in spans:
        span_id = str(row.get("id") or "")
        span_text = str(row.get("text") or "")
        normalized_span = _normalized_text(span_text)
        if not span_id or not normalized_span:
            continue
        source_claimed = any(
            source == normalized_span or normalized_span in source
            for source in changed_sources
            if source
        )
        id_claimed = span_id in changed_ids or span_id in reported_ids
        if (source_claimed or id_claimed) and normalized_span not in normalized_replacement:
            actual.append(span_id)
    return actual


def scanner_controlled_variant_gate(
    *,
    report: dict[str, Any],
    group: TargetGroup,
    variant: dict[str, Any],
    replacement_text: str,
    require_ownership: bool = False,
) -> dict[str, Any]:
    raw_replacement = str(variant.get("replacement_text") or "")
    anchors = protected_anchor_placeholders(group)
    frozen_source = freeze_protected_anchors(group.source_text, anchors)
    expected_placeholders = [
        str(row.get("placeholder") or "")
        for row in anchors
        if str(row.get("placeholder") or "") and str(row.get("placeholder") or "") in frozen_source
    ]
    placeholder_integrity = protected_placeholder_integrity(raw_replacement, expected_placeholders)
    if not placeholder_integrity["passed"]:
        return {
            "passed": False,
            "reason": "anchor_placeholder_corruption",
            "placeholder_integrity": placeholder_integrity,
            "predictable_spans_modified_count": 0,
        }
    if _normalized_text(replacement_text) == _normalized_text(group.source_text):
        return {
            "passed": False,
            "reason": "no_material_change",
            "predictable_spans_modified_count": 0,
        }
    action_contract = group_action_contract(
        group=group,
        predictability_briefs=predictability_briefs_from_report(report),
    )
    ownership_rows = [
        row for row in variant.get("ownership_changes") or []
        if isinstance(row, dict)
        and _normalized_text(str(row.get("before") or ""))
        and _normalized_text(str(row.get("before") or "")) != _normalized_text(str(row.get("after") or ""))
    ]
    ownership_elements = {
        str(item)
        for item in variant.get("ownership_elements_supported") or []
        if str(item or "") in {"author_trace", "specific_context", "real_judgment"}
    }
    ownership_passed = bool(ownership_rows or ownership_elements)
    if require_ownership and not ownership_passed:
        return {
            "passed": False,
            "reason": "ownership_repair_required",
            "predictable_spans_modified_count": 0,
            "ownership_change_count": 0,
            "ownership_elements_supported": [],
        }
    topk_contract = action_contract.get("topk_repair_contract") if isinstance(action_contract.get("topk_repair_contract"), dict) else {}
    span_rows = [
        {"id": str(row.get("id") or ""), "text": str(row.get("text") or "")}
        for row in topk_contract.get("predictable_span_rows") or []
        if isinstance(row, dict)
    ]
    if not span_rows:
        span_rows = [
            {"id": f"ps{index:03d}", "text": str(span)}
            for index, span in enumerate(topk_contract.get("predictable_spans_in_source") or topk_contract.get("predictable_spans") or [], start=1)
        ]
    spans = [
        row
        for row in span_rows
        if row["text"] and row["text"] in str(group.source_text or "")
    ]
    span_texts = [
        row["text"]
        for row in spans
    ]
    if topk_contract.get("span_source") != "scanner_exact" or not spans:
        return {
            "passed": True,
            "reason": "no_exact_span_gate",
            "predictable_spans_modified_count": 0,
            "ownership_change_count": len(ownership_rows),
            "ownership_elements_supported": sorted(ownership_elements),
        }
    actual_modified_ids = _actual_modified_span_ids(
        spans=spans,
        variant=variant,
        replacement_text=replacement_text,
    )
    actual_count = len(actual_modified_ids)
    declared_count = _declared_changed_span_count(variant)
    modified_count = actual_count
    required_count = int(topk_contract.get("required_modified_spans") or min(2, len(spans)))
    required_count = max(1, min(required_count, len(spans)))
    if modified_count < required_count:
        if require_ownership and ownership_passed:
            return {
                "passed": True,
                "reason": "ownership_repair_passed_without_required_topk_movement",
                "predictable_spans_modified_count": modified_count,
                "declared_predictable_spans_modified_count": declared_count,
                "self_report_mismatch": declared_count != modified_count,
                "actual_modified_span_ids": actual_modified_ids,
                "required_predictable_spans_modified": required_count,
                "available_predictable_spans": len(spans),
                "predictable_spans": span_texts,
                "ownership_change_count": len(ownership_rows),
                "ownership_elements_supported": sorted(ownership_elements),
            }
        return {
            "passed": False,
            "reason": "insufficient_predictable_span_movement",
            "predictable_spans_modified_count": modified_count,
            "declared_predictable_spans_modified_count": declared_count,
            "self_report_mismatch": declared_count != modified_count,
            "actual_modified_span_ids": actual_modified_ids,
            "required_predictable_spans_modified": required_count,
            "available_predictable_spans": len(spans),
            "predictable_spans": span_texts,
            "ownership_change_count": len(ownership_rows),
            "ownership_elements_supported": sorted(ownership_elements),
        }
    return {
        "passed": True,
        "reason": "passed",
        "predictable_spans_modified_count": modified_count,
        "declared_predictable_spans_modified_count": declared_count,
        "self_report_mismatch": declared_count != modified_count,
        "actual_modified_span_ids": actual_modified_ids,
        "required_predictable_spans_modified": required_count,
        "available_predictable_spans": len(spans),
        "predictable_spans": span_texts,
        "ownership_change_count": len(ownership_rows),
        "ownership_elements_supported": sorted(ownership_elements),
    }


def _alpha_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in str(text or "").casefold():
        if char.isalpha():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _average_token_length(text: str) -> float:
    tokens = _alpha_tokens(text)
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return {
        (tokens[index], tokens[index + 1])
        for index in range(0, max(0, len(tokens) - 1))
    }


def _source_likeness_ratio(source_text: str, replacement_text: str) -> float:
    source_bigrams = _bigrams(_alpha_tokens(source_text))
    if not source_bigrams:
        return 0.0
    replacement_bigrams = _bigrams(_alpha_tokens(replacement_text))
    return len(source_bigrams.intersection(replacement_bigrams)) / len(source_bigrams)


def _novel_content_token_ratio(source_text: str, replacement_text: str) -> float:
    source_tokens = set(_alpha_tokens(source_text))
    replacement_tokens = [token for token in _alpha_tokens(replacement_text) if len(token) >= 4]
    if not replacement_tokens:
        return 0.0
    novel_count = sum(1 for token in replacement_tokens if token not in source_tokens)
    return novel_count / len(replacement_tokens)


def scanner_controlled_candidate_quality(
    *,
    action_contract: dict[str, Any],
    variant_gate: dict[str, Any],
    source_text: str,
    replacement_text: str,
    variant: dict[str, Any],
) -> dict[str, Any]:
    topk_contract = action_contract.get("topk_repair_contract") if isinstance(action_contract.get("topk_repair_contract"), dict) else {}
    limits = topk_contract.get("locality_limits") if isinstance(topk_contract.get("locality_limits"), dict) else {}
    required = max(1, int(variant_gate.get("required_predictable_spans_modified") or topk_contract.get("required_modified_spans") or 1))
    actual = max(0, int(variant_gate.get("predictable_spans_modified_count") or 0))
    movement_score = min(1.0, actual / required) * 45.0
    changed_rows = [row for row in variant.get("changed_spans") or [] if isinstance(row, dict)]
    max_changed_spans = max(1, int(limits.get("max_changed_spans") or 3))
    locality_overage = max(0, len(changed_rows) - max_changed_spans)
    locality_score = max(0.0, 20.0 - locality_overage * 6.0)
    source_words = max(1, len(str(source_text or "").split()))
    replacement_words = max(1, len(str(replacement_text or "").split()))
    ratio_delta = abs(replacement_words - source_words) / source_words
    preservation_score = max(0.0, 20.0 - ratio_delta * 80.0)
    avg_delta = _average_token_length(replacement_text) - _average_token_length(source_text)
    polish_risk = max(0.0, min(20.0, avg_delta * 18.0))
    source_likeness_ratio = _source_likeness_ratio(source_text, replacement_text)
    source_likeness_score = source_likeness_ratio * (14.0 if action_contract.get("citation_pressure_zone") else 6.0)
    novel_token_ratio = _novel_content_token_ratio(source_text, replacement_text)
    novel_token_penalty = min(10.0, novel_token_ratio * (8.0 if action_contract.get("citation_pressure_zone") else 4.0))
    citation_smoothing_risk = polish_risk if action_contract.get("citation_pressure_zone") else 0.0
    self_report_penalty = 5.0 if variant_gate.get("self_report_mismatch") else 0.0
    ownership_rows = [
        row for row in variant.get("ownership_changes") or []
        if isinstance(row, dict)
        and _normalized_text(str(row.get("before") or ""))
        and _normalized_text(str(row.get("before") or "")) != _normalized_text(str(row.get("after") or ""))
    ]
    ownership_elements = {
        str(item)
        for item in variant.get("ownership_elements_supported") or []
        if str(item or "") in {"author_trace", "specific_context", "real_judgment"}
    }
    ownership_score = min(10.0, len(ownership_rows) * 3.0 + len(ownership_elements) * 1.5)
    span_operations = [
        str(row.get("operation") or "")
        for row in changed_rows
        if str(row.get("operation") or "")
    ]
    score = round(
        movement_score
        + locality_score
        + preservation_score
        + source_likeness_score
        + ownership_score
        - polish_risk
        - novel_token_penalty
        - citation_smoothing_risk
        - self_report_penalty,
        3,
    )
    return {
        "score": score,
        "movement_score": round(movement_score, 3),
        "locality_score": round(locality_score, 3),
        "preservation_score": round(preservation_score, 3),
        "source_likeness_ratio": round(source_likeness_ratio, 4),
        "source_likeness_score": round(source_likeness_score, 3),
        "novel_token_ratio": round(novel_token_ratio, 4),
        "novel_token_penalty": round(novel_token_penalty, 3),
        "polish_risk": round(polish_risk, 3),
        "citation_smoothing_risk": round(citation_smoothing_risk, 3),
        "self_report_penalty": round(self_report_penalty, 3),
        "ownership_score": round(ownership_score, 3),
        "ownership_change_count": len(ownership_rows),
        "ownership_elements_supported": sorted(ownership_elements),
        "actual_modified_span_ids": list(variant_gate.get("actual_modified_span_ids") or []),
        "operator_used": variant.get("operator_used"),
        "span_operations": span_operations,
        "source_words": source_words,
        "replacement_words": replacement_words,
    }


__all__ = [
    "ScannerControlledConfig",
    "blocker_rows_for_group",
    "build_scanner_controlled_prompt",
    "freeze_protected_anchors",
    "parse_scanner_controlled_variants",
    "protected_placeholder_integrity",
    "rank_scanner_target_groups",
    "restore_protected_anchor_placeholders",
    "scanner_controlled_candidate_quality",
    "scanner_controlled_variant_gate",
    "scanner_controlled_metrics",
    "scanner_controlled_rank",
    "weak_human_levers",
]
