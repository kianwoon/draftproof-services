"""Typed scan contract for rewrite V3 routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .document_units import document_units, word_count


class RewriteRiskClass(str, Enum):
    BROAD_PROSE = "broad_prose"
    CITED_ACADEMIC = "cited_academic"
    TECHNICAL_STRUCTURED = "technical_structured"
    REGULATED_POLICY = "regulated_policy"
    QUOTE_OR_EVIDENCE_HEAVY = "quote_or_evidence_heavy"
    PERSONAL_REFLECTIVE = "personal_reflective"
    CREATIVE_MARKETING = "creative_marketing"
    SHORT_OR_SPARSE = "short_or_sparse"


@dataclass(frozen=True)
class ScanContract:
    word_count: int
    unit_count: int
    heading_count: int
    avg_unit_words: float
    ai_score: float | None
    topk_score: float | None
    writing_quality_score: float | None
    content_mode: str
    content_mode_confidence: float
    mode_scores: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    hard_anchor_count: int = 0
    citation_anchor_count: int = 0
    quote_anchor_count: int = 0
    dedup_anchor_count: int = 0
    quote_count: int = 0
    quote_density: float = 0.0
    citation_count: int = 0
    citation_density: float = 0.0
    citation_key_count: int = 0
    reference_count: int = 0
    evidence_anchor_score: float = 0.0
    anchor_preservation_pressure: float = 0.0
    quote_role_counts: dict[str, int] = field(default_factory=dict)
    ai_footprint_profile: dict[str, Any] = field(default_factory=dict)
    rewrite_target_profile: dict[str, Any] = field(default_factory=dict)
    rewrite_targets: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    target_driver_summary: dict[str, int] = field(default_factory=dict)
    target_operation_mix: dict[str, int] = field(default_factory=dict)
    target_scope_policy: str = ""
    footprint_fraction_ai: float = 0.0
    footprint_fraction_ai_assisted: float = 0.0
    footprint_fraction_human: float = 0.0
    risky_window_density: float = 0.0
    max_risky_window_words: int = 0
    high_confidence_risky_window_count: int = 0
    risky_window_count: int = 0
    footprint_confidence: str = "low"
    findings_count: int = 0
    rewrite_brief_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _first_route(scan_report: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        scan_report.get("rewrite_v2_content_route"),
        scan_report.get("content_route"),
        ((scan_report.get("scan_intelligence") or {}).get("mitigation_inputs") or {}).get("rewrite_v2_content_route"),
        ((scan_report.get("scan_intelligence") or {}).get("generation_handoff") or {}).get("rewrite_v2_content_route"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _dict_at(payload: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _list_at(payload: dict[str, Any], path: tuple[str, ...]) -> list[Any]:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return []
        current = current.get(key)
    return current if isinstance(current, list) else []


def _unique_values(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    unique: list[Any] = []
    for value in values or []:
        key = str(sorted(value.items())) if isinstance(value, dict) else str(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _anchor_key(anchor: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(anchor.get("text") or "").strip(),
        str(anchor.get("kind") or anchor.get("type") or "").strip(),
        str(anchor.get("category") or "").strip(),
        str(anchor.get("severity") or "").strip(),
    )


def _dedupe_anchors(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for anchor in anchors or []:
        if not isinstance(anchor, dict):
            continue
        key = _anchor_key(anchor)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        unique.append(anchor)
    return unique


def _generation_handoffs(scan_report: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        scan_report.get("generation_handoff"),
        _dict_at(scan_report, ("scan_intelligence", "generation_handoff")),
        _dict_at(scan_report, ("scan_intelligence", "mitigation_inputs", "generation_handoff")),
    ]
    return [candidate for candidate in candidates if isinstance(candidate, dict) and candidate]


def _citation_key_count(scan_report: dict[str, Any]) -> int:
    keys: list[Any] = []
    references: list[Any] = []
    for handoff in _generation_handoffs(scan_report):
        references.extend(handoff.get("reference_register") or [])
        for unit in handoff.get("section_generation_units") or []:
            if not isinstance(unit, dict):
                continue
            keys.extend(unit.get("citation_keys_used") or [])
            for meaning in unit.get("meaning_inventory") or []:
                if isinstance(meaning, dict):
                    keys.extend(meaning.get("citation_keys") or [])
    return max(len(_unique_values(keys)), len(_unique_values(references)))


def _routing_anchor_metrics(scan_report: dict[str, Any], word_total: int) -> dict[str, Any]:
    routing_candidates = [
        scan_report.get("rewrite_routing_signals"),
        _dict_at(scan_report, ("scan_intelligence", "rewrite_routing_signals")),
        _dict_at(scan_report, ("scan_intelligence", "mitigation_inputs", "rewrite_routing_signals")),
        _dict_at(scan_report, ("generation_handoff", "rewrite_routing_signals")),
    ]
    for candidate in routing_candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("anchor_metrics"), dict):
            metrics = dict(candidate["anchor_metrics"])
            role_counts = metrics.get("quote_role_counts") if isinstance(metrics.get("quote_role_counts"), dict) else {}
            metrics["quote_role_counts"] = {str(key): int(value or 0) for key, value in role_counts.items()}
            return metrics

    inventories = [
        ((scan_report.get("scan_intelligence") or {}).get("document") or {}).get("preservation_inventory"),
        ((scan_report.get("scan_intelligence") or {}).get("mitigation_inputs") or {}).get("preservation_inventory"),
        ((scan_report.get("scan_intelligence") or {}).get("mitigation_inputs") or {})
        .get("rewrite_constraints", {})
        .get("preservation_inventory"),
        scan_report.get("rewrite_constraints", {}).get("preservation_inventory") if isinstance(scan_report.get("rewrite_constraints"), dict) else None,
        (((scan_report.get("scan_intelligence") or {}).get("mitigation_inputs") or {}).get("rewrite_handoff") or {})
        .get("rewrite_constraints", {})
        .get("preservation_inventory"),
        ((scan_report.get("ai_mitigation") or {}).get("rewrite_handoff") or {})
        .get("rewrite_constraints", {})
        .get("preservation_inventory"),
    ]
    inventory: list[dict[str, Any]] = []
    quote_values: list[Any] = []
    citation_values: list[Any] = []
    for candidate in inventories:
        if isinstance(candidate, dict) and isinstance(candidate.get("anchors"), list):
            inventory.extend(item for item in candidate["anchors"] if isinstance(item, dict))
            quote_values.extend(candidate.get("quotes") or [])
            citation_values.extend(candidate.get("citations") or [])
        elif isinstance(candidate, list):
            inventory.extend(item for item in candidate if isinstance(item, dict))
    anchors = _dedupe_anchors(inventory)
    hard = 0
    citations = 0
    quotes = 0
    role_counts: dict[str, int] = {}
    for item in anchors:
        kind = str(item.get("kind") or item.get("type") or "")
        category = str(item.get("category") or "")
        severity = str(item.get("severity") or "")
        if severity.startswith("hard"):
            hard += 1
        if kind in {"citation", "source_citation"} or category == "citation":
            citations += 1
        if kind in {"direct_quote", "quote"} or category == "quote":
            quotes += 1
            role = str(item.get("quote_role") or item.get("anchor_role") or item.get("role") or "unknown_quote")
            role_counts[role] = role_counts.get(role, 0) + 1
    quote_count = max(quotes, len(_unique_values(quote_values)))
    citation_count = max(citations, len(_unique_values(citation_values)))
    citation_key_count = _citation_key_count(scan_report)
    citation_signal = max(citation_count, citation_key_count)
    direct_evidence_score = min(
        1.0,
        role_counts.get("direct_quote", 0) * 0.35
        + role_counts.get("evidence_quote", 0) * 0.45
        + role_counts.get("citation_quote", 0) * 0.45
        + citation_signal * 0.18
        + hard * 0.08,
    )
    untyped_quote_score = 0.0 if direct_evidence_score >= 0.5 else min(0.12, quote_count * 0.03)
    words = max(1, int(word_total or 0))
    return {
        "dedup_anchor_count": len(anchors),
        "hard_anchor_count": hard,
        "citation_anchor_count": citations,
        "quote_anchor_count": quotes,
        "quote_count": quote_count,
        "quote_density": round(quote_count / words, 4),
        "citation_count": citation_count,
        "citation_density": round(citation_count / words, 4),
        "citation_key_count": citation_key_count,
        "reference_count": 0,
        "evidence_anchor_score": round(min(1.0, direct_evidence_score + untyped_quote_score), 3),
        "anchor_preservation_pressure": round(min(1.0, direct_evidence_score + hard * 0.08 + citation_signal * 0.08), 3),
        "quote_role_counts": role_counts,
    }


def _findings_count(scan_report: dict[str, Any]) -> int:
    findings = scan_report.get("findings")
    if not isinstance(findings, dict):
        return 0
    return sum(len(value) for value in findings.values() if isinstance(value, list))


def _ai_footprint_profile(scan_report: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        scan_report.get("ai_footprint_profile"),
        _dict_at(scan_report, ("scan_intelligence", "ai_footprint_profile")),
        _dict_at(scan_report, ("scan_intelligence", "document", "ai_footprint_profile")),
        _dict_at(scan_report, ("authorship_window_profile", "ai_footprint_profile")),
        _dict_at(scan_report, ("scan_intelligence", "authorship_window_profile", "ai_footprint_profile")),
        _dict_at(scan_report, ("scan_intelligence", "document", "authorship_window_profile", "ai_footprint_profile")),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate

    legacy_candidates = [
        scan_report.get("authorship_window_profile"),
        _dict_at(scan_report, ("scan_intelligence", "authorship_window_profile")),
        _dict_at(scan_report, ("scan_intelligence", "document", "authorship_window_profile")),
    ]
    for legacy in legacy_candidates:
        if not isinstance(legacy, dict) or not legacy:
            continue
        windows = legacy.get("windows") if isinstance(legacy.get("windows"), list) else []
        risky = [
            window for window in windows
            if isinstance(window, dict) and str(window.get("label") or "") in {"ai_generated", "moderately_ai_assisted"}
        ]
        word_total = int(legacy.get("word_count") or 0) or sum(int(window.get("word_count") or 0) for window in windows if isinstance(window, dict))
        denominator = max(1, word_total)
        risky_words = sum(int(window.get("word_count") or 0) for window in risky)
        return {
            "schema_version": "ai_footprint_profile.v2_legacy_adapter",
            "basis": "legacy_authorship_windows",
            "word_count": word_total,
            "window_count": len(windows),
            "fraction_ai": float(legacy.get("fraction_ai") or 0.0),
            "fraction_ai_assisted": float(legacy.get("fraction_ai_assisted") or 0.0),
            "fraction_human": float(legacy.get("fraction_human") or 0.0),
            "risky_window_density": round(risky_words / denominator, 4),
            "assisted_window_density": float(legacy.get("fraction_ai") or 0.0) + float(legacy.get("fraction_ai_assisted") or 0.0),
            "max_risky_window_words": max([int(window.get("word_count") or 0) for window in risky] or [0]),
            "max_ai_window_words": int(legacy.get("max_ai_window_words") or 0),
            "max_ai_assisted_window_words": int(legacy.get("max_ai_assisted_window_words") or 0),
            "high_confidence_risky_window_count": sum(
                1 for window in risky if str(window.get("confidence") or "").lower() == "high"
            ),
            "risky_window_count": len(risky),
            "assisted_window_count": int(legacy.get("num_ai_segments") or 0) + int(legacy.get("num_ai_assisted_segments") or 0),
            "top_risky_windows": risky[:8],
            "false_positive_guard": {"enabled": True, "caution": False},
            "confidence": "low",
        }
    badge = scan_report.get("ai_risk_badge") if isinstance(scan_report.get("ai_risk_badge"), dict) else {}
    ai_score = _float_or_none(badge.get("ai_likelihood_score"))
    if ai_score is not None:
        assisted = max(0.0, min(1.0, ai_score / 100.0))
        return {
            "schema_version": "ai_footprint_profile.v2_badge_adapter",
            "basis": "badge_ai_likelihood",
            "word_count": 0,
            "window_count": 0,
            "fraction_ai": 0.0,
            "fraction_ai_assisted": round(assisted, 4),
            "fraction_human": round(max(0.0, 1.0 - assisted), 4),
            "risky_window_density": round(max(0.0, assisted - 0.35), 4),
            "assisted_window_density": round(assisted, 4),
            "max_risky_window_words": 0,
            "max_ai_window_words": 0,
            "max_ai_assisted_window_words": 0,
            "high_confidence_risky_window_count": 0,
            "risky_window_count": 0,
            "assisted_window_count": 0,
            "top_risky_windows": [],
            "false_positive_guard": {"enabled": True, "caution": False},
            "confidence": "low",
        }
    return {}


def _rewrite_target_profile(scan_report: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        scan_report.get("rewrite_target_profile"),
        _dict_at(scan_report, ("scan_intelligence", "rewrite_target_profile")),
        _dict_at(scan_report, ("scan_intelligence", "document", "rewrite_target_profile")),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def build_scan_contract(scan_report: dict[str, Any], original_text: str) -> ScanContract:
    units = document_units(original_text)
    total_words = word_count(original_text)
    route = _first_route(scan_report)
    badge = scan_report.get("ai_risk_badge") if isinstance(scan_report.get("ai_risk_badge"), dict) else {}
    components = badge.get("ai_components") if isinstance(badge.get("ai_components"), dict) else {}
    anchor_metrics = _routing_anchor_metrics(scan_report, total_words)
    footprint = _ai_footprint_profile(scan_report)
    target_profile = _rewrite_target_profile(scan_report)
    rewrite_targets = target_profile.get("targets") if isinstance(target_profile.get("targets"), list) else []
    mode_scores = route.get("mode_scores") if isinstance(route.get("mode_scores"), list) else []
    return ScanContract(
        word_count=total_words,
        unit_count=len(units),
        heading_count=sum(1 for unit in units if unit.is_heading),
        avg_unit_words=round(sum(unit.word_count for unit in units) / max(1, len(units)), 2),
        ai_score=_float_or_none(badge.get("ai_likelihood_score")),
        topk_score=_float_or_none(components.get("topk_pattern_raw")),
        writing_quality_score=_float_or_none(badge.get("writing_quality_score")),
        content_mode=str(route.get("primary_mode") or route.get("content_mode") or "unknown"),
        content_mode_confidence=float(route.get("confidence") or route.get("content_mode_confidence") or 0.0),
        mode_scores=tuple(row for row in mode_scores if isinstance(row, dict)),
        hard_anchor_count=int(anchor_metrics.get("hard_anchor_count") or 0),
        citation_anchor_count=int(anchor_metrics.get("citation_anchor_count") or 0),
        quote_anchor_count=int(anchor_metrics.get("quote_anchor_count") or 0),
        dedup_anchor_count=int(anchor_metrics.get("dedup_anchor_count") or 0),
        quote_count=int(anchor_metrics.get("quote_count") or anchor_metrics.get("quote_anchor_count") or 0),
        quote_density=float(anchor_metrics.get("quote_density") or 0.0),
        citation_count=int(anchor_metrics.get("citation_count") or anchor_metrics.get("citation_anchor_count") or 0),
        citation_density=float(anchor_metrics.get("citation_density") or 0.0),
        citation_key_count=int(anchor_metrics.get("citation_key_count") or 0),
        reference_count=int(anchor_metrics.get("reference_count") or 0),
        evidence_anchor_score=float(anchor_metrics.get("evidence_anchor_score") or 0.0),
        anchor_preservation_pressure=float(anchor_metrics.get("anchor_preservation_pressure") or 0.0),
        quote_role_counts=dict(anchor_metrics.get("quote_role_counts") or {}),
        ai_footprint_profile=dict(footprint),
        rewrite_target_profile=dict(target_profile),
        rewrite_targets=tuple(row for row in rewrite_targets if isinstance(row, dict)),
        target_driver_summary={
            str(key): int(value or 0)
            for key, value in (target_profile.get("driver_summary") or {}).items()
        } if isinstance(target_profile.get("driver_summary"), dict) else {},
        target_operation_mix={
            str(key): int(value or 0)
            for key, value in (target_profile.get("operation_mix") or {}).items()
        } if isinstance(target_profile.get("operation_mix"), dict) else {},
        target_scope_policy=str(target_profile.get("target_scope_policy") or ""),
        footprint_fraction_ai=float(footprint.get("fraction_ai") or 0.0),
        footprint_fraction_ai_assisted=float(footprint.get("fraction_ai_assisted") or 0.0),
        footprint_fraction_human=float(footprint.get("fraction_human") or 0.0),
        risky_window_density=float(footprint.get("risky_window_density") or 0.0),
        max_risky_window_words=int(footprint.get("max_risky_window_words") or 0),
        high_confidence_risky_window_count=int(footprint.get("high_confidence_risky_window_count") or 0),
        risky_window_count=int(footprint.get("risky_window_count") or 0),
        footprint_confidence=str(footprint.get("confidence") or "low"),
        findings_count=_findings_count(scan_report),
        rewrite_brief_count=len(scan_report.get("rewrite_edit_briefs") or []),
    )
