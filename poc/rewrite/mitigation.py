"""Signal-aware mitigation planning for rewrite reports.

This layer separates what can be safely patched automatically from signals
that need author evidence, source grounding, or structural revision.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from rewrite.planner import RewritePlan, RewriteAction


LOCAL_STYLE_TYPES = {
    "high_predictability",
    "medium_predictability",
    "high_topk_predictability",
    "low_surprisal",
    "low_surprisal_pattern",
    "formulaic_sentence",
    "generic_formulaic_language",
    "generic_phrase",
    "mechanical_transition",
    "generic_enumeration",
    "style_shift",
    "repetitive_sentence_structure",
}

STRUCTURE_TYPES = {
    "uniform_paragraph_structure",
    "low_burstiness",
    "repeated_sentence_structure",
    "paragraph_progression",
    "signpost_paragraph",
    "formulaic_conclusion",
}

GROUNDING_TYPES = {
    "low_specificity",
    "source_grounding",
    "polished_but_ungrounded",
    "unsupported_claim",
    "uncited_claim",
    "uncited_in_body",
    "missing_from_bib",
    "citation_weakness",
    "weak_source_grounding",
    "moderate_ai_generation_likelihood",
    "elevated_ai_generation_likelihood",
}

PROTECTED_TYPES = {
    "exact_copy",
    "direct_quote_mismatch",
}


def _finding_id(action: RewriteAction) -> str:
    meta = action.finding.metadata or {}
    return str(meta.get("finding_id") or getattr(action.finding, "id", ""))


def _sentence_id(action: RewriteAction) -> str:
    loc = action.finding.location or {}
    return str(loc.get("sentence_id") or "")


def _safe_evidence(action: RewriteAction, limit: int = 140) -> str:
    evidence = action.finding.evidence
    if not isinstance(evidence, str):
        evidence = str(evidence)
    return evidence[:limit]


def _action_summary(action: RewriteAction) -> str:
    ftype = action.finding.finding_type
    if ftype in LOCAL_STYLE_TYPES:
        return "Generate detector-gated sentence patch; accept only if local signal and final scan do not regress."
    if ftype in STRUCTURE_TYPES:
        return "Provide paragraph-level restructuring guidance; do not sentence-paraphrase automatically."
    if ftype in GROUNDING_TYPES:
        return "Ask for source, citation, or concrete author example; otherwise narrow or soften unsupported claim."
    if ftype in PROTECTED_TYPES:
        return "Do not rewrite automatically; preserve quoted or protected material."
    return "Review manually before applying any rewrite."


def _bucket_for_action(action: RewriteAction) -> str:
    ftype = action.finding.finding_type
    if action.fixability == "protected" or ftype in PROTECTED_TYPES:
        return "protected"
    if action.fixability in {"auto", "partial"} and ftype in LOCAL_STYLE_TYPES:
        return "auto_rewrite"
    if ftype in STRUCTURE_TYPES:
        return "structure_guidance"
    if ftype in GROUNDING_TYPES or action.required_inputs:
        return "needs_source_or_example"
    if action.fixability in {"auto", "partial"}:
        return "auto_rewrite"
    return "review_only"


def _item(action: RewriteAction) -> Dict[str, Any]:
    return {
        "finding_id": _finding_id(action),
        "finding_type": action.finding.finding_type,
        "risk_level": action.finding.risk_level,
        "scanner": (action.finding.metadata or {}).get("scanner", ""),
        "sentence_id": _sentence_id(action),
        "fixability": action.fixability,
        "action": action.action_type,
        "reason": action.reason,
        "required_inputs": action.required_inputs,
        "evidence": _safe_evidence(action),
        "mitigation": _action_summary(action),
    }


def _component_items(raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    badge = (raw_json or {}).get("ai_risk_badge") or {}
    components = badge.get("ai_components") or {}
    writing = badge.get("writing_components") or {}
    items = []
    component_rules = {
        "generic_assertion_risk": ("needs_source_or_example", "Narrow broad assertions or add source-backed detail."),
        "unsupported_claim_risk": ("needs_source_or_example", "Add evidence/citation or soften/remove unsupported claims."),
        "source_grounding_risk": ("needs_source_or_example", "Attach claims to source material supplied by the author."),
        "broad_claim_risk": ("needs_source_or_example", "Replace broad claim with context-limited wording."),
        "lived_detail_risk": ("needs_source_or_example", "Add real process detail or classroom observation supplied by the author."),
        "citation_weakness_risk": ("needs_source_or_example", "Repair citation/source linkage manually."),
        "paragraph_uniformity_risk": ("structure_guidance", "Vary paragraph structure and length at section level."),
        "signpost_paragraph_risk": ("structure_guidance", "Reduce formulaic signposting and revise paragraph openings."),
        "topk_pattern": ("auto_rewrite", "Use GPT-2-guided structural sentence patches."),
        "predictability": ("auto_rewrite", "Use GPT-2-guided structural sentence patches."),
        "generic_phrase_density": ("auto_rewrite", "Replace generic phrases with document-specific wording."),
        "burstiness_risk": ("structure_guidance", "Adjust rhythm by splitting, shortening, or merging sentences."),
    }
    for name, value in {**components, **writing}.items():
        if not isinstance(value, (int, float)) or value < 50:
            continue
        bucket, mitigation = component_rules.get(
            name,
            ("review_only", "Review this component manually before changing text."),
        )
        items.append({
            "component": name,
            "score": round(float(value), 2),
            "bucket": bucket,
            "mitigation": mitigation,
        })
    return sorted(items, key=lambda item: item["score"], reverse=True)


def build_mitigation_plan(
    plan: RewritePlan | None,
    raw_json: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a JSON-serializable mitigation plan for product/reporting."""
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "auto_rewrite": [],
        "needs_source_or_example": [],
        "structure_guidance": [],
        "review_only": [],
        "protected": [],
    }
    if plan:
        for action in plan.actions:
            bucket = _bucket_for_action(action)
            buckets.setdefault(bucket, []).append(_item(action))

    component_drivers = _component_items(raw_json or {})
    counts = Counter({key: len(value) for key, value in buckets.items()})

    primary_mode = "auto_rewrite"
    if counts["needs_source_or_example"] or any(i["bucket"] == "needs_source_or_example" for i in component_drivers):
        primary_mode = "guided_revision"
    elif counts["structure_guidance"] or any(i["bucket"] == "structure_guidance" for i in component_drivers):
        primary_mode = "structure_revision"
    elif not counts["auto_rewrite"]:
        primary_mode = "manual_review"

    return {
        "primary_mode": primary_mode,
        "counts": dict(counts),
        "buckets": buckets,
        "component_drivers": component_drivers,
    }
