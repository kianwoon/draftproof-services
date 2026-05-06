"""AI mitigation planning from scan intelligence.

The detect phase should be able to explain *how* a draft could move toward a
more human-authored signal profile before any rewrite engine runs. This module
does not generate replacement prose. It converts scan signals into a stable
plan of authenticity levers, target spans, required user inputs, and guardrails.
"""

from __future__ import annotations

from typing import Any, Dict, List


AUTHENTICITY_GUARDRAILS = [
    "Do not inject typos, grammar errors, random sentence breaks, or synonym-spun wording.",
    "Do not invent personal observations, citations, examples, named entities, dates, numbers, or sources.",
    "Preserve quoted text, citations, proper nouns, numbers, and assignment-specific terms.",
    "Prefer author-supplied grounding, source-to-claim links, and concrete process detail over surface paraphrase.",
    "Keep automatic edits limited to scanner-approved local wording signals.",
]


COMPONENT_RULES: Dict[str, Dict[str, str]] = {
    "unsupported_claim_risk": {
        "lever": "reasoning_continuity",
        "bucket": "needs_author_evidence",
        "action": "Support the claim with a real source/example or soften the certainty.",
        "user_input": "source, citation, observation, or permission to reduce certainty",
    },
    "source_grounding_risk": {
        "lever": "source_grounding",
        "bucket": "needs_author_evidence",
        "action": "Connect the claim to a source idea and explain the link in the author's words.",
        "user_input": "source name and the exact source idea supporting the claim",
    },
    "citation_weakness_risk": {
        "lever": "source_grounding",
        "bucket": "needs_author_evidence",
        "action": "Attach citation support directly to the claim it proves or qualifies.",
        "user_input": "correct citation and cited evidence",
    },
    "generic_assertion_risk": {
        "lever": "specificity",
        "bucket": "needs_author_context",
        "action": "Replace reusable assertions with a concrete setting, task, method, or constraint.",
        "user_input": "specific class, unit, workflow, observation, or source detail",
    },
    "broad_claim_risk": {
        "lever": "specificity",
        "bucket": "needs_author_context",
        "action": "Narrow broad claims to the exact context and limit the conclusion.",
        "user_input": "learner group, condition, method, or evidence boundary",
    },
    "lived_detail_risk": {
        "lever": "human_grounding",
        "bucket": "needs_author_context",
        "action": "Add real author/process detail that is already true for the work.",
        "user_input": "real observation, session, workflow, decision, or constraint",
    },
    "qualifying_text_ai_density": {
        "lever": "cognitive_authenticity",
        "bucket": "paragraph_rebuild",
        "action": "Rebuild dense paragraphs around context, evidence, author reasoning, and a limited conclusion.",
        "user_input": "author-owned evidence or source-backed paragraph route",
    },
    "semantic_uniformity_risk": {
        "lever": "cognitive_authenticity",
        "bucket": "paragraph_rebuild",
        "action": "Add sharper paragraph roles and section-specific reasoning so the meaning does not stay too evenly shaped.",
        "user_input": "which paragraphs should carry evidence, author reasoning, contrast, limitation, or conclusion",
    },
    "discourse_regularity_risk": {
        "lever": "natural_variance",
        "bucket": "structure_revision",
        "action": "Vary the discourse route so paragraphs do not advance with the same smooth explanatory pattern.",
        "user_input": "author choice on where to add contrast, a pause, a concrete example, or a limitation",
    },
    "semantic_drift_risk": {
        "lever": "reasoning_continuity",
        "bucket": "structure_revision",
        "action": "Add bridging reasoning where adjacent ideas jump too far apart.",
        "user_input": "the missing connection between adjacent claims, examples, or source ideas",
    },
    "paraphrase_transformation_risk": {
        "lever": "source_grounding",
        "bucket": "manual_source_revision",
        "action": "Compare meaning against source material and add author interpretation rather than only changed wording.",
        "user_input": "source text, citation, and the author's own interpretation",
    },
    "paragraph_uniformity_risk": {
        "lever": "natural_variance",
        "bucket": "structure_revision",
        "action": "Vary paragraph roles so not every paragraph follows the same claim-explain-summary shape.",
        "user_input": "which paragraph should open from evidence, problem, reflection, or conclusion",
    },
    "signpost_paragraph_risk": {
        "lever": "natural_variance",
        "bucket": "structure_revision",
        "action": "Replace formulaic signposting with the paragraph's concrete evidence or situation.",
        "user_input": "the concrete issue, evidence, or situation the paragraph is about",
    },
    "burstiness_risk": {
        "lever": "natural_variance",
        "bucket": "structure_revision",
        "action": "Vary sentence rhythm with intentional short emphasis and longer contextual follow-up.",
        "user_input": "author choice on which point deserves emphasis",
    },
    "topk_pattern": {
        "lever": "predictability_reduction",
        "bucket": "auto_candidate",
        "action": "Change sentence route using existing paragraph context, not synonym swaps.",
        "user_input": "none if no evidence/source gap is attached",
    },
    "predictability": {
        "lever": "predictability_reduction",
        "bucket": "auto_candidate",
        "action": "Change sentence route using existing paragraph context, not synonym swaps.",
        "user_input": "none if no evidence/source gap is attached",
    },
    "generic_phrase_density": {
        "lever": "predictability_reduction",
        "bucket": "auto_candidate",
        "action": "Replace generic connectors with context-specific wording already supported nearby.",
        "user_input": "none if nearby context is sufficient",
    },
}


SIGNAL_RULES: Dict[str, Dict[str, str]] = {
    "ai_likelihood": COMPONENT_RULES["predictability"],
    "rewrite_smoothness": {
        "lever": "cognitive_authenticity",
        "bucket": "needs_author_context",
        "action": "Replace overly smooth generic explanation with author reasoning or process detail.",
        "user_input": "author reasoning, limitation, or process observation",
    },
    "human_anchor_score": COMPONENT_RULES["lived_detail_risk"],
    "grounding_risk": COMPONENT_RULES["source_grounding_risk"],
    "source_similarity": {
        "lever": "source_grounding",
        "bucket": "manual_source_revision",
        "action": "Revise source-adjacent wording with source notes and citation context.",
        "user_input": "source card, citation, and author's interpretation",
    },
    "section_style_variance": {
        "lever": "reasoning_continuity",
        "bucket": "structure_revision",
        "action": "Review whether the section shift reflects real draft history or an unsupported rewrite jump.",
        "user_input": "draft history or author explanation of the section's reasoning path",
    },
    "semantic_uniformity": COMPONENT_RULES["semantic_uniformity_risk"],
    "semantic_uniformity_risk": COMPONENT_RULES["semantic_uniformity_risk"],
    "discourse_regularity": COMPONENT_RULES["discourse_regularity_risk"],
    "discourse_regularity_risk": COMPONENT_RULES["discourse_regularity_risk"],
    "semantic_drift": COMPONENT_RULES["semantic_drift_risk"],
    "semantic_drift_risk": COMPONENT_RULES["semantic_drift_risk"],
    "paraphrase_transformation_risk": COMPONENT_RULES["paraphrase_transformation_risk"],
}


def _score(value: Any) -> int:
    if not isinstance(value, (int, float)):
        return 0
    numeric = float(value)
    if numeric <= 1.0:
        numeric *= 100.0
    return max(0, min(100, int(round(numeric))))


def _priority(score: int, bucket: str) -> str:
    if bucket in {"needs_author_evidence", "needs_author_context", "paragraph_rebuild"} and score >= 50:
        return "highest" if score >= 75 else "high"
    if score >= 75:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def _target_score(score: int, bucket: str) -> int:
    if score >= 75:
        return 55 if bucket != "auto_candidate" else 50
    if score >= 50:
        return 45
    return max(0, score - 10)


def _rule_for_component(component: str) -> Dict[str, str]:
    return COMPONENT_RULES.get(component, {
        "lever": "reviewed_authenticity",
        "bucket": "review_only",
        "action": "Review this score driver manually before changing text.",
        "user_input": "author judgment",
    })


def _component_actions(badge: Dict[str, Any]) -> List[Dict[str, Any]]:
    components = {}
    components.update(badge.get("ai_components") or {})
    components.update(badge.get("writing_components") or {})

    actions: List[Dict[str, Any]] = []
    for component, raw in components.items():
        score = _score(raw)
        if score < 35:
            continue
        rule = _rule_for_component(str(component))
        bucket = rule["bucket"]
        target = _target_score(score, bucket)
        actions.append({
            "component": component,
            "lever": rule["lever"],
            "bucket": bucket,
            "current_score": score,
            "target_score": target,
            "reduction_needed": max(0, score - target),
            "priority": _priority(score, bucket),
            "action": rule["action"],
            "user_input_needed": rule["user_input"],
            "auto_apply": bucket == "auto_candidate",
        })

    bucket_rank = {
        "needs_author_evidence": 0,
        "needs_author_context": 1,
        "paragraph_rebuild": 2,
        "structure_revision": 3,
        "manual_source_revision": 4,
        "auto_candidate": 5,
        "review_only": 6,
    }
    return sorted(
        actions,
        key=lambda item: (
            bucket_rank.get(item["bucket"], 9),
            -item["reduction_needed"],
            -item["current_score"],
        ),
    )


def _semantic_layer_actions(scan_intelligence: Dict[str, Any]) -> List[Dict[str, Any]]:
    semantic = ((scan_intelligence or {}).get("semantic_layer") or {})
    actions: List[Dict[str, Any]] = []
    for component in (
        "semantic_uniformity_risk",
        "discourse_regularity_risk",
        "semantic_drift_risk",
        "paraphrase_transformation_risk",
    ):
        score = _score(semantic.get(component))
        if score < 35:
            continue
        rule = _rule_for_component(component)
        bucket = rule["bucket"]
        target = _target_score(score, bucket)
        actions.append({
            "component": component,
            "lever": rule["lever"],
            "bucket": bucket,
            "current_score": score,
            "target_score": target,
            "reduction_needed": max(0, score - target),
            "priority": _priority(score, bucket),
            "action": rule["action"],
            "user_input_needed": rule["user_input"],
            "auto_apply": False,
            "source": "semantic_layer",
        })
    return actions


def _signal_rule(signal: Dict[str, Any]) -> Dict[str, str]:
    key = str(signal.get("key") or "")
    title = str(signal.get("title") or "").lower()
    if "semantic_drift" in title or key == "semantic_drift":
        return COMPONENT_RULES["semantic_drift_risk"]
    if "semantic_uniformity" in title or key == "semantic_uniformity":
        return COMPONENT_RULES["semantic_uniformity_risk"]
    if "discourse_regularity" in title or key == "discourse_regularity":
        return COMPONENT_RULES["discourse_regularity_risk"]
    if "ground" in title or "citation" in title:
        return SIGNAL_RULES["grounding_risk"]
    if "specificity" in title:
        return COMPONENT_RULES["generic_assertion_risk"]
    if "predictability" in title or "topk" in title or "surprisal" in title:
        return COMPONENT_RULES["predictability"]
    if "generic" in title or "smooth" in title:
        return SIGNAL_RULES["rewrite_smoothness"]
    return SIGNAL_RULES.get(key, {
        "lever": "reviewed_authenticity",
        "bucket": "review_only",
        "action": "Review this span manually before changing it.",
        "user_input": "author judgment",
    })


def _target_segments(scan_intelligence: Dict[str, Any], limit: int = 12) -> List[Dict[str, Any]]:
    segments = (((scan_intelligence or {}).get("document") or {}).get("segments") or [])
    targets: List[Dict[str, Any]] = []
    for segment in segments:
        signals = segment.get("signals") or []
        if not signals:
            continue
        primary = signals[0]
        rule = _signal_rule(primary)
        permission = primary.get("rewrite_permission") or "suggestion_only"
        bucket = rule["bucket"]
        auto_apply = permission == "auto" and bucket == "auto_candidate"
        targets.append({
            "segment_id": segment.get("segment_id") or segment.get("sentence_id"),
            "sentence_id": segment.get("sentence_id"),
            "paragraph_id": segment.get("paragraph_id"),
            "text": segment.get("text", ""),
            "primary_signal": {
                "finding_id": primary.get("finding_id"),
                "key": primary.get("key"),
                "title": primary.get("title"),
                "score": primary.get("score"),
                "tier": primary.get("tier"),
                "rewrite_permission": permission,
            },
            "lever": rule["lever"],
            "bucket": bucket,
            "action": rule["action"],
            "user_input_needed": "none" if auto_apply else rule["user_input"],
            "auto_apply": auto_apply,
        })
        if len(targets) >= limit:
            break

    bucket_rank = {
        "needs_author_evidence": 0,
        "needs_author_context": 1,
        "paragraph_rebuild": 2,
        "structure_revision": 3,
        "manual_source_revision": 4,
        "auto_candidate": 5,
        "review_only": 6,
    }
    return sorted(
        targets,
        key=lambda item: (
            bucket_rank.get(item["bucket"], 9),
            0 if item["auto_apply"] else 1,
            -_score((item.get("primary_signal") or {}).get("score")),
        ),
    )


def _baseline(scan_intelligence: Dict[str, Any], badge: Dict[str, Any]) -> Dict[str, Any]:
    contribution = (
        ((scan_intelligence or {}).get("transformation") or {})
        .get("contribution") or {}
    )
    human = _score(contribution.get("human_contribution_ratio"))
    ai = _score(contribution.get("ai_transformation_ratio"))
    if not human and not ai:
        ai = _score(badge.get("ai_likelihood_score"))
        human = max(0, 100 - ai)
    return {
        "human_contribution_ratio": human,
        "ai_transformation_ratio": ai,
        "target_human_contribution_ratio": min(100, human + 10 if ai >= 40 else human + 5),
        "target_ai_transformation_ratio": max(0, ai - 10 if ai >= 40 else ai - 5),
        "summary": contribution.get("summary") or "",
    }


def _primary_mode(actions: List[Dict[str, Any]], targets: List[Dict[str, Any]]) -> str:
    buckets = [a["bucket"] for a in actions] + [t["bucket"] for t in targets]
    if any(b in buckets for b in ("needs_author_evidence", "needs_author_context")):
        return "guided_authenticity_revision"
    if "paragraph_rebuild" in buckets:
        return "paragraph_authenticity_rebuild"
    if "structure_revision" in buckets:
        return "structure_revision"
    if any(t.get("auto_apply") for t in targets):
        return "targeted_ai_signal_rewrite"
    return "review_only"


def _counts(actions: List[Dict[str, Any]], targets: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in actions + targets:
        bucket = str(item.get("bucket") or "review_only")
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _required_inputs(actions: List[Dict[str, Any]], targets: List[Dict[str, Any]]) -> List[str]:
    inputs = []
    for item in actions + targets:
        needed = str(item.get("user_input_needed") or "")
        if not needed or needed == "none":
            continue
        if needed not in inputs:
            inputs.append(needed)
    return inputs[:8]


def build_ai_mitigation_plan(
    *,
    scan_intelligence: Dict[str, Any],
    ai_risk_badge: Dict[str, Any],
    rewrite_plan: Dict[str, Any] | None = None,
    rewrite_constraints: Dict[str, Any] | None = None,
    rewrite_edit_briefs: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build the AI-Mitigation phase contract from scan output."""
    badge = ai_risk_badge or {}
    actions = _component_actions(badge) + _semantic_layer_actions(scan_intelligence)
    targets = _target_segments(scan_intelligence)
    mode = _primary_mode(actions, targets)
    counts = _counts(actions, targets)
    auto_target_ids = [
        target["segment_id"]
        for target in targets
        if target.get("auto_apply")
    ]
    return {
        "schema_version": "ai_mitigation.v1",
        "philosophy": "authenticity_mitigation",
        "objective": {
            "reduce": "detectable AI transformation signals",
            "increase": [
                "human grounding",
                "specificity",
                "cognitive authenticity",
                "reasoning continuity",
                "contextual anchoring",
                "natural variance",
            ],
            "avoid": [
                "typo injection",
                "synonym spinning",
                "random sentence breaking",
                "unsupported detail generation",
            ],
        },
        "baseline": _baseline(scan_intelligence, badge),
        "primary_mode": mode,
        "readiness": {
            "auto_rewrite_allowed": bool(auto_target_ids) and mode == "targeted_ai_signal_rewrite",
            "auto_target_segment_ids": auto_target_ids,
            "requires_user_input": mode in {
                "guided_authenticity_revision",
                "paragraph_authenticity_rebuild",
                "structure_revision",
            },
            "required_inputs": _required_inputs(actions, targets),
        },
        "counts": counts,
        "component_actions": actions,
        "target_segments": targets,
        "guardrails": AUTHENTICITY_GUARDRAILS,
        "rewrite_handoff": {
            "rewrite_plan": rewrite_plan or {},
            "rewrite_constraints": rewrite_constraints or {},
            "rewrite_edit_brief_count": len(rewrite_edit_briefs or []),
            "allowed_automatic_scope": "scanner-approved local targets only",
        },
    }
