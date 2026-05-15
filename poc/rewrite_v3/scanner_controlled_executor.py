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


def freeze_protected_anchors(text: str, anchors: list[dict[str, Any]]) -> str:
    frozen = str(text or "")
    for anchor in anchors:
        source = str(anchor.get("text") or "")
        placeholder = str(anchor.get("placeholder") or "")
        if source and placeholder:
            frozen = frozen.replace(source, placeholder)
    return frozen


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
) -> str:
    anchors = protected_anchor_placeholders(group)
    variant_plan = [
        {"variant_id": f"v{index}", "operator": operator}
        for index, operator in enumerate(VARIANT_OPERATORS[: max(1, int(variants_per_group))], start=1)
    ]
    movement_contract = _movement_contract(report, group)
    movement_contract["variant_plan"] = variant_plan
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
        "weak_human_levers": [
            {
                "key": row.get("key"),
                "rewrite_lever": row.get("rewrite_lever"),
            }
            for row in weak_human_levers(report)
        ],
        "instructions": [
            f"Create exactly {len(variant_plan)} replacement variants for source_text only.",
            "Each variant must use its assigned movement_contract.variant_plan operator as the main change.",
            "Use movement_contract as the rewrite contract.",
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
                    "protected_anchors_preserved": True,
                    "new_claims_added": False,
                }
            ]
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def parse_scanner_controlled_variants(raw: str, *, limit: int = 3) -> list[dict[str, str]]:
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
    variants: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        replacement = str(row.get("replacement_text") or "").strip()
        if not replacement:
            continue
        variant_id = str(row.get("variant_id") or f"v{index}").strip()
        variants.append({"variant_id": variant_id, "replacement_text": replacement})
        if len(variants) >= max(1, int(limit or 1)):
            break
    return variants


__all__ = [
    "ScannerControlledConfig",
    "blocker_rows_for_group",
    "build_scanner_controlled_prompt",
    "parse_scanner_controlled_variants",
    "rank_scanner_target_groups",
    "scanner_controlled_metrics",
    "scanner_controlled_rank",
    "weak_human_levers",
]
