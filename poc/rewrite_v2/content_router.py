"""Scan-driven semantic content routing for rewrite V2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContentRoute:
    content_mode: str
    confidence: float
    reasons: list[str]
    allowed_strategy_families: list[str]
    blocked_strategy_families: list[str]
    mode_scores: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONTENT_MODES = {
    "broad_explanatory_essay",
    "academic_cited_text",
    "technical_content",
    "regulated_policy_content",
    "structured_list_table",
    "quote_heavy",
    "short_text",
    "personal_reflection",
    "creative_marketing",
    "generic_expository",
    "hybrid_guarded",
}

FULL_DOCUMENT_STRATEGY_FAMILIES = {
    "entity_locked_full_reconstruction",
    "keyword_locked_short_texture",
    "author_stance_thesis_reframe",
    "author_stance_texture_pass",
}

ALL_STRATEGY_FAMILIES = {
    "academic_anchor_repair_texture_pass",
    "academic_all_section_compact_reconstruction",
    "academic_cited_section_density_resolver",
    "targeted_paragraph_reconstruction",
    "unsafe_cluster_rescue",
    *FULL_DOCUMENT_STRATEGY_FAMILIES,
}

ALLOWED_BY_MODE = {
    "broad_explanatory_essay": {
        "targeted_paragraph_reconstruction",
        "unsafe_cluster_rescue",
        "entity_locked_full_reconstruction",
        "keyword_locked_short_texture",
        "author_stance_thesis_reframe",
        "author_stance_texture_pass",
    },
    "academic_cited_text": {
        "academic_anchor_repair_texture_pass",
        "academic_all_section_compact_reconstruction",
        "academic_cited_section_density_resolver",
        "targeted_paragraph_reconstruction",
        "unsafe_cluster_rescue",
    },
    "technical_content": {"targeted_paragraph_reconstruction", "unsafe_cluster_rescue"},
    "regulated_policy_content": {"targeted_paragraph_reconstruction", "unsafe_cluster_rescue"},
    "structured_list_table": {"targeted_paragraph_reconstruction"},
    "quote_heavy": {"targeted_paragraph_reconstruction", "unsafe_cluster_rescue"},
    "short_text": {"targeted_paragraph_reconstruction"},
    "personal_reflection": {
        "targeted_paragraph_reconstruction",
        "unsafe_cluster_rescue",
        "author_stance_thesis_reframe",
        "author_stance_texture_pass",
    },
    "creative_marketing": {"targeted_paragraph_reconstruction", "unsafe_cluster_rescue"},
    "hybrid_guarded": {"targeted_paragraph_reconstruction"},
    "generic_expository": {
        "targeted_paragraph_reconstruction",
        "unsafe_cluster_rescue",
        "entity_locked_full_reconstruction",
        "keyword_locked_short_texture",
        "author_stance_thesis_reframe",
        "author_stance_texture_pass",
    },
}

def route_payload(
    mode: str,
    confidence: float,
    reasons: list[str],
    mode_scores: list[dict[str, Any]] | None = None,
) -> ContentRoute:
    safe_mode = mode if mode in CONTENT_MODES else "hybrid_guarded"
    allowed = sorted(ALLOWED_BY_MODE.get(safe_mode, {"targeted_paragraph_reconstruction"}))
    blocked = sorted(ALL_STRATEGY_FAMILIES - set(allowed))
    return ContentRoute(
        content_mode=safe_mode,
        confidence=round(max(0.0, min(1.0, float(confidence or 0.0))), 3),
        reasons=[str(reason) for reason in list(reasons or [])[:8] if str(reason)],
        allowed_strategy_families=allowed,
        blocked_strategy_families=blocked,
        mode_scores=list(mode_scores or [])[:8],
    )


def guarded_content_route(reason: str = "scan_content_route_missing") -> ContentRoute:
    return route_payload(
        "hybrid_guarded",
        0.35,
        [reason, "targeted-only fallback until scan provides typed route"],
        [{"content_mode": "hybrid_guarded", "score": 0.35, "source": "guarded_fallback"}],
    )


def _dict_at(payload: dict[str, Any] | None, path: tuple[str, ...]) -> dict[str, Any]:
    current: Any = payload or {}
    for key in path:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _first_route_payload(scan_report: dict[str, Any] | None) -> dict[str, Any]:
    report = scan_report or {}
    paths = (
        ("rewrite_v2_content_route",),
        ("content_route",),
        ("generation_handoff", "rewrite_v2_content_route"),
        ("scan_intelligence", "generation_handoff", "rewrite_v2_content_route"),
        ("scan_intelligence", "mitigation_inputs", "rewrite_v2_content_route"),
        ("ai_mitigation", "rewrite_v2_content_route"),
    )
    for path in paths:
        value = _dict_at(report, path)
        if value:
            return value
    return {}


def _mode_scores_from_payload(payload: dict[str, Any], mode: str, confidence: float) -> list[dict[str, Any]]:
    scores = payload.get("mode_scores")
    if isinstance(scores, list):
        return [row for row in scores if isinstance(row, dict)]
    secondary = payload.get("secondary_modes")
    rows = [{"content_mode": mode, "score": confidence, "source": "scan_contract"}]
    if isinstance(secondary, list):
        for item in secondary:
            if isinstance(item, str) and item in CONTENT_MODES:
                rows.append({"content_mode": item, "score": max(0.0, confidence - 0.08), "source": "scan_contract_secondary"})
    return rows


def content_route_from_scan(scan_report: dict[str, Any] | None) -> ContentRoute | None:
    payload = _first_route_payload(scan_report)
    if payload:
        mode = str(payload.get("primary_mode") or payload.get("content_mode") or "").strip()
        confidence = float(payload.get("confidence") or payload.get("content_mode_confidence") or 0.0)
        if mode in CONTENT_MODES and confidence > 0:
            reasons = payload.get("reasons") if isinstance(payload.get("reasons"), list) else []
            mode_scores = _mode_scores_from_payload(payload, mode, confidence)
            return route_payload(mode, confidence, reasons or ["scan-provided V2 content route"], mode_scores)
    return None


def content_route_from_semantic_payload(payload: dict[str, Any] | None) -> ContentRoute | None:
    if not isinstance(payload, dict):
        return None
    mode = str(payload.get("primary_mode") or payload.get("content_mode") or "").strip()
    if mode not in CONTENT_MODES:
        return None
    confidence = float(payload.get("confidence") or 0.0)
    if confidence <= 0:
        return None
    reasons = payload.get("reasons") if isinstance(payload.get("reasons"), list) else []
    secondary = payload.get("secondary_modes") if isinstance(payload.get("secondary_modes"), list) else []
    mode_scores = [{"content_mode": mode, "score": confidence, "source": "semantic_classifier"}]
    for item in secondary:
        if isinstance(item, str) and item in CONTENT_MODES and item != mode:
            mode_scores.append({"content_mode": item, "score": max(0.0, confidence - 0.08), "source": "semantic_classifier_secondary"})
    if confidence < 0.62:
        return route_payload(
            "hybrid_guarded",
            confidence,
            list(reasons) + ["semantic classifier confidence below routing threshold"],
            mode_scores,
        )
    return route_payload(mode, confidence, reasons or ["semantic classifier route"], mode_scores)


def _bounded_string(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _first_items(value: Any, limit: int) -> list[Any]:
    return list(value[:limit]) if isinstance(value, list) else []


def _compact_section_units(units: Any, limit: int = 8) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for unit in _first_items(units, limit):
        if not isinstance(unit, dict):
            continue
        compact.append({
            "section_id": unit.get("section_id"),
            "heading": unit.get("heading"),
            "role": unit.get("role"),
            "word_count": unit.get("word_count"),
            "citation_count": unit.get("citation_count"),
            "text_preview": _bounded_string(unit.get("text"), 420),
        })
    return compact


def _compact_anchor_register(anchors: Any, limit: int = 12) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for anchor in _first_items(anchors, limit):
        if not isinstance(anchor, dict):
            continue
        compact.append({
            "text": _bounded_string(anchor.get("text"), 120),
            "kind": anchor.get("kind") or anchor.get("type"),
            "severity": anchor.get("severity"),
        })
    return compact


def _compact_target_segments(segments: Any, limit: int = 8) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for segment in _first_items(segments, limit):
        if not isinstance(segment, dict):
            continue
        compact.append({
            "id": segment.get("id") or segment.get("segment_id") or segment.get("sentence_id"),
            "type": segment.get("type") or segment.get("kind"),
            "risk": segment.get("risk") or segment.get("risk_score"),
            "preview": _bounded_string(segment.get("text") or segment.get("preview"), 300),
        })
    return compact


def _compact_generation_handoff(handoff: Any) -> dict[str, Any]:
    payload = handoff if isinstance(handoff, dict) else {}
    return {
        "document_profile": payload.get("document_profile") if isinstance(payload.get("document_profile"), dict) else {},
        "logical_outline": _bounded_string(payload.get("logical_outline"), 900),
        "section_generation_units": _compact_section_units(payload.get("section_generation_units")),
        "anchor_register": _compact_anchor_register(payload.get("anchor_register")),
        "reference_count": len(payload.get("reference_register") or []) if isinstance(payload.get("reference_register"), list) else 0,
        "generation_constraints": payload.get("generation_constraints") if isinstance(payload.get("generation_constraints"), dict) else {},
    }


def _compact_rewrite_constraints(constraints: Any) -> dict[str, Any]:
    payload = constraints if isinstance(constraints, dict) else {}
    return {
        "preserve_terms": _first_items(payload.get("preserve_terms"), 16),
        "allowed_additions": _first_items(payload.get("allowed_additions"), 8),
        "do_not_add": _first_items(payload.get("do_not_add"), 8),
        "rewrite_rule": payload.get("rewrite_rule"),
        "max_change_scope": payload.get("max_change_scope"),
    }


def _compact_ai_mitigation(plan: Any) -> dict[str, Any]:
    payload = plan if isinstance(plan, dict) else {}
    return {
        "philosophy": payload.get("philosophy"),
        "primary_mode": payload.get("primary_mode"),
        "readiness": payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {},
        "counts": payload.get("counts") if isinstance(payload.get("counts"), dict) else {},
        "target_segments": _compact_target_segments(payload.get("target_segments")),
        "guardrails": payload.get("guardrails") if isinstance(payload.get("guardrails"), dict) else {},
    }


def _compact_scan_document(document: Any) -> dict[str, Any]:
    payload = document if isinstance(document, dict) else {}
    return {
        "word_count": payload.get("word_count"),
        "paragraph_count": payload.get("paragraph_count"),
        "sentence_count": payload.get("sentence_count"),
        "document_shape": payload.get("document_shape"),
        "dominant_risk_drivers": _first_items(payload.get("dominant_risk_drivers"), 8),
    }


def _text_samples(text: str, sample_chars: int = 900) -> dict[str, str]:
    value = str(text or "")
    if len(value) <= sample_chars * 2:
        return {"opening": _bounded_string(value, sample_chars)}
    middle_start = max(0, (len(value) // 2) - (sample_chars // 2))
    return {
        "opening": _bounded_string(value[:sample_chars], sample_chars),
        "middle": _bounded_string(value[middle_start:middle_start + sample_chars], sample_chars),
        "ending": _bounded_string(value[-sample_chars:], sample_chars),
    }


def semantic_route_prompt(original_text: str, scan_report: dict[str, Any] | None) -> str:
    report = scan_report or {}
    intelligence = report.get("scan_intelligence") if isinstance(report.get("scan_intelligence"), dict) else {}
    handoff = report.get("generation_handoff") if isinstance(report.get("generation_handoff"), dict) else {}
    if not handoff:
        handoff = _dict_at(report, ("scan_intelligence", "generation_handoff"))
    compact_scan = {
        "document": _compact_scan_document(intelligence.get("document")),
        "generation_handoff": _compact_generation_handoff(handoff),
        "rewrite_constraints": _compact_rewrite_constraints(report.get("rewrite_constraints")),
        "ai_mitigation": _compact_ai_mitigation(report.get("ai_mitigation")),
    }
    payload = {
        "task": "Classify the submitted document for DraftProof rewrite V2 routing.",
        "content_modes": sorted(CONTENT_MODES - {"hybrid_guarded"}),
        "fallback_mode": "hybrid_guarded",
        "requirements": [
            "Choose a primary mode by document purpose and rewrite safety, not by isolated words.",
            "Return secondary modes when preservation contracts overlap.",
            "Use hybrid_guarded when confidence is low or modes conflict.",
            "Do not recommend full-document reconstruction for academic, technical, regulated, list/table, or quote-heavy content.",
        ],
        "scan_summary": compact_scan,
        "submitted_text_samples": _text_samples(str(original_text or "")),
    }
    return (
        "Return JSON only with keys: primary_mode, secondary_modes, confidence, reasons, "
        "allowed_strategy_families, blocked_strategy_families.\n"
        f"CONTENT_ROUTE_CLASSIFICATION:\n{json.dumps(payload, ensure_ascii=False, default=str)}"
    )
