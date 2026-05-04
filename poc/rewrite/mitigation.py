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
    if isinstance(evidence, dict):
        evidence = evidence.get("summary") or evidence.get("affected_span") or ""
    elif not isinstance(evidence, str):
        evidence = str(evidence)
    return evidence[:limit]


def _quote_evidence(action: RewriteAction, limit: int = 220) -> str:
    evidence = action.finding.evidence
    if not isinstance(evidence, str):
        return ""
    cleaned = " ".join(evidence.split())
    if not cleaned or len(cleaned.split()) < 5:
        return ""
    metric_only = (
        cleaned.endswith("domain terms")
        or cleaned in {"low_specificity"}
        or " words, " in cleaned
    )
    if metric_only:
        return ""
    return cleaned[:limit]


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
        "topk_pattern": ("auto_rewrite", "Review sentence structure; only keep edits if the final scan improves."),
        "predictability": ("auto_rewrite", "Review sentence structure; only keep edits if the final scan improves."),
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


def _component_driver_bucket_item(driver: Dict[str, Any]) -> Dict[str, Any]:
    component = str(driver.get("component") or "")
    score = float(driver.get("score") or 0.0)
    if score >= 80:
        risk_level = "high"
    elif score >= 60:
        risk_level = "medium"
    else:
        risk_level = "review"
    return {
        "finding_id": "",
        "finding_type": component,
        "risk_level": risk_level,
        "scanner": "ai_risk_badge",
        "sentence_id": "",
        "fixability": "manual",
        "action": "guided_revision",
        "reason": f"Document-level component driver scored {score:.1f}.",
        "required_inputs": [],
        "evidence": "",
        "mitigation": driver.get("mitigation") or "Review this document-level signal manually.",
        "component_driver": True,
        "score": score,
    }


def _score_mitigation_targets(component_drivers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prioritized score levers for reducing the next detect scan.

    These are not promises of exact badge movement. They identify the component
    scores most worth reducing and the revision action that can plausibly move
    them below review thresholds without inventing content.
    """
    targets: List[Dict[str, Any]] = []
    for driver in component_drivers:
        component = str(driver.get("component") or "")
        score = float(driver.get("score") or 0.0)
        bucket = str(driver.get("bucket") or "review_only")
        if score < 50:
            continue
        if score >= 80:
            target = 55.0
            priority = "highest"
        elif score >= 65:
            target = 50.0
            priority = "high"
        else:
            target = 45.0
            priority = "medium"
        reduction = max(0.0, score - target)
        if bucket == "needs_source_or_example":
            if component == "unsupported_claim_risk":
                action = "Add citation/example support or soften confident unsupported claims."
            elif component == "source_grounding_risk":
                action = "Connect sources directly to the claims they support."
            elif component == "citation_weakness_risk":
                action = "Repair in-text citation linkage around key claims."
            elif component in {"broad_claim_risk", "generic_assertion_risk"}:
                action = "Narrow broad claims to the exact class, unit, method, or observation."
            else:
                action = driver.get("mitigation") or "Add author-supplied evidence or concrete detail."
        elif bucket == "auto_rewrite":
            action = "Retry detector-gated sentence rewrite after evidence/source gaps are reduced."
        elif bucket == "structure_guidance":
            action = "Revise paragraph structure before retrying sentence-level rewrite."
        else:
            action = driver.get("mitigation") or "Review this signal manually."
        targets.append({
            "component": component,
            "bucket": bucket,
            "current_score": round(score, 2),
            "target_score": round(target, 2),
            "reduction_needed": round(reduction, 2),
            "priority": priority,
            "action": action,
        })
    bucket_rank = {
        "needs_source_or_example": 0,
        "structure_guidance": 1,
        "auto_rewrite": 2,
        "review_only": 3,
    }
    return sorted(
        targets,
        key=lambda item: (
            bucket_rank.get(item["bucket"], 9),
            -float(item["reduction_needed"]),
            -float(item["current_score"]),
        ),
    )


def _reference_pattern(component: str) -> Dict[str, str] | None:
    patterns = {
        "generic_assertion_risk": {
            "focus": "Narrow a broad claim",
            "instead_of": "[Broad claim] is important because it helps learners improve.",
            "try_pattern": (
                "In [specific class/activity/context], [your source or observation] "
                "shows that [narrow claim]. For example, [author-supplied evidence] "
                "suggests [limited conclusion]."
            ),
            "why": "This turns a general statement into a context-bound claim with room for evidence.",
        },
        "broad_claim_risk": {
            "focus": "Limit the claim scope",
            "instead_of": "[Topic] improves student outcomes.",
            "try_pattern": (
                "For [specific learner group/context], [specific method] may support "
                "[specific outcome] when [condition or evidence] is present."
            ),
            "why": "This avoids overclaiming and makes the statement easier to support.",
        },
        "unsupported_claim_risk": {
            "focus": "Support or soften a claim",
            "instead_of": "[Confident claim] clearly proves [conclusion].",
            "try_pattern": (
                "[Your source/example] indicates that [supported claim]. If evidence is limited, "
                "phrase it as: This may suggest [careful conclusion], rather than [strong conclusion]."
            ),
            "why": "This gives the user a choice: add evidence or reduce certainty.",
        },
        "source_grounding_risk": {
            "focus": "Connect source to claim",
            "instead_of": "[Claim] without showing where it came from.",
            "try_pattern": (
                "According to [source/author], [source idea]. This supports [your claim] "
                "because [explain the link in your own words]."
            ),
            "why": "This makes the source relationship visible instead of leaving the claim floating.",
        },
        "citation_weakness_risk": {
            "focus": "Repair citation linkage",
            "instead_of": "[Claim] with no clear in-text source.",
            "try_pattern": (
                "[Author/source] reports [specific evidence], which is relevant here because "
                "[connect evidence to your paragraph point]."
            ),
            "why": "This ties the citation to the paragraph argument instead of dropping it in loosely.",
        },
        "lived_detail_risk": {
            "focus": "Add author/process detail",
            "instead_of": "The activity helped learners understand the concept.",
            "try_pattern": (
                "During [specific activity/session], I observed [specific learner action or result]. "
                "That detail matters because [explain what it shows]."
            ),
            "why": "Real process detail makes the writing less generic without inventing facts.",
        },
        "topk_pattern": {
            "focus": "Vary sentence path",
            "instead_of": "This demonstrates the importance of [topic] in [field].",
            "try_pattern": (
                "Start from the concrete context first: [specific situation]. Then state the point: "
                "[narrow claim]."
            ),
            "why": "Changing the sentence route is usually stronger than swapping individual words.",
        },
        "predictability": {
            "focus": "Vary sentence path",
            "instead_of": "This demonstrates the importance of [topic] in [field].",
            "try_pattern": (
                "Start from the concrete context first: [specific situation]. Then state the point: "
                "[narrow claim]."
            ),
            "why": "Changing the sentence route is usually stronger than swapping individual words.",
        },
        "signpost_paragraph_risk": {
            "focus": "Replace formulaic paragraph opening",
            "instead_of": "This paragraph will discuss [topic].",
            "try_pattern": (
                "Open with the evidence or situation: [specific observation/source detail]. "
                "Then explain how it connects to [paragraph point]."
            ),
            "why": "This avoids template-like signposting and moves the paragraph into the argument faster.",
        },
        "paragraph_uniformity_risk": {
            "focus": "Vary paragraph structure",
            "instead_of": "Every paragraph follows claim, explanation, summary.",
            "try_pattern": (
                "Mix paragraph roles: one paragraph can start with evidence, another with a problem, "
                "and another with a short author reflection tied to the source."
            ),
            "why": "Varied paragraph roles reduce a uniform, template-like structure.",
        },
        "burstiness_risk": {
            "focus": "Vary rhythm",
            "instead_of": "Several sentences of similar length and shape.",
            "try_pattern": (
                "Use one short sentence for the key point. Follow it with a longer sentence that "
                "adds [specific evidence or context]."
            ),
            "why": "Changing rhythm helps the prose feel edited by a person, not generated from one pattern.",
        },
    }
    pattern = patterns.get(component)
    if not pattern:
        return None
    return {"component": component, **pattern}


def _pattern_bucket(component: str) -> str:
    pattern = _reference_pattern(component)
    if not pattern:
        return ""
    if component in {
        "generic_assertion_risk",
        "broad_claim_risk",
        "unsupported_claim_risk",
        "source_grounding_risk",
        "citation_weakness_risk",
        "lived_detail_risk",
    }:
        return "needs_source_or_example"
    if component in {"paragraph_uniformity_risk", "signpost_paragraph_risk", "burstiness_risk"}:
        return "structure_guidance"
    return "auto_rewrite"


def _find_flagged_excerpt(bucket_items: List[Dict[str, Any]]) -> str:
    for item in bucket_items:
        evidence = item.get("flagged_excerpt") or item.get("evidence", "")
        if not isinstance(evidence, str):
            continue
        cleaned = " ".join(evidence.split())
        if len(cleaned.split()) >= 5 and " words, " not in cleaned and cleaned != "low_specificity":
            return cleaned
    return ""


def _reference_patterns(
    component_drivers: List[Dict[str, Any]],
    buckets: Dict[str, List[Dict[str, Any]]],
    limit: int = 4,
) -> List[Dict[str, str]]:
    selected = []
    seen = set()
    for driver in component_drivers:
        component = driver.get("component", "")
        pattern = _reference_pattern(component)
        if not pattern or pattern["focus"] in seen:
            continue
        bucket = _pattern_bucket(component)
        excerpt = _find_flagged_excerpt(buckets.get(bucket, []))
        if not excerpt and bucket != "needs_source_or_example":
            excerpt = _find_flagged_excerpt(buckets.get("auto_rewrite", []))
        selected.append({
            **pattern,
            "flagged_excerpt": excerpt,
            "application_note": (
                "Use this pattern on the quoted finding above."
                if excerpt
                else "This is a document-level signal. Apply the pattern to broad claims or weakly supported points in the draft."
            ),
        })
        seen.add(pattern["focus"])
        if len(selected) >= limit:
            break
    return sorted(
        selected,
        key=lambda item: 0 if item.get("flagged_excerpt") else 1,
    )


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
            item = _item(action)
            quote = _quote_evidence(action)
            if quote:
                item["flagged_excerpt"] = quote
            buckets.setdefault(bucket, []).append(item)

    component_drivers = _component_items(raw_json or {})
    for driver in component_drivers:
        bucket = driver.get("bucket")
        if bucket not in {"needs_source_or_example", "structure_guidance", "review_only"}:
            continue
        buckets.setdefault(bucket, []).append(_component_driver_bucket_item(driver))
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
        "score_mitigation_targets": _score_mitigation_targets(component_drivers),
        "reference_patterns": _reference_patterns(component_drivers, buckets),
    }
