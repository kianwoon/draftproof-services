"""Rewrite planner: maps detect findings to fixability, actions, scopes, and eligibility.

Fixability buckets:
  auto      — writing-style issue, safe to revise automatically
  partial   — can revise expression, but source grounding must be preserved
  manual    — rewriting cannot fix this (missing sources, formatting)
  protected — must not be altered automatically (quotes, citations)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detect.base import Finding, DetectResult
from rewrite.scorer import RISK_WEIGHTS, weighted_finding_score


# ── Fixability classification ────────────────────────────────────────

FIXABILITY_AUTO = "auto"
FIXABILITY_PARTIAL = "partial"
FIXABILITY_MANUAL = "manual"
FIXABILITY_PROTECTED = "protected"

# Signals that justify auto-rewriting medium_predictability
COMPANION_SIGNALS = {
    "generic_phrase",
    "style_shift",
    "formulaic_sentence",
    "close_paraphrase",
    "patchwriting",
    "semantic_overlap",
    "mechanical_transition",
    "generic_enumeration",
    "vague_claim",
    "similarity_overlap",
    "weak_source_grounding",
}


@dataclass
class FixabilityDecision:
    finding_id: str
    finding_type: str
    fixability: str  # "auto" | "partial" | "manual" | "protected"
    action: str
    scope: str
    reason: str
    required_inputs: List[str] = field(default_factory=list)


# What to do with each finding type → (fixability, action, scope, reason)
FINDING_ROUTING: Dict[str, dict] = {
    "high_predictability": {
        "fixability": FIXABILITY_AUTO,
        "action": "suggest_rewrite",
        "scope": "sentence",
        "reason": "Writing-style issue can be safely revised.",
    },
    "medium_predictability": {
        "fixability": FIXABILITY_PARTIAL,
        "action": "review_only",
        "scope": "sentence",
        "reason": "Medium predictability alone is review-level. Auto-rewrite only if paired with a companion signal.",
    },
    "review_predictability": {
        "fixability": FIXABILITY_MANUAL,
        "action": "review_only",
        "scope": "sentence",
        "reason": "Review-band predictability: normal prose. Not an automatic rewrite target.",
    },
    "low_predictability": {
        "fixability": FIXABILITY_AUTO,
        "action": "suggest_rewrite",
        "scope": "span",
        "reason": "Minor style issue, safe to revise.",
    },
    "formulaic_sentence": {
        "fixability": FIXABILITY_AUTO,
        "action": "suggest_rewrite",
        "scope": "sentence",
        "reason": "Formulaic structure can be safely restructured.",
    },
    "generic_phrase": {
        "fixability": FIXABILITY_AUTO,
        "action": "suggest_rewrite",
        "scope": "span",
        "reason": "Generic phrase can be replaced with specific language.",
    },
    "style_shift": {
        "fixability": FIXABILITY_AUTO,
        "action": "suggest_rewrite",
        "scope": "sentence",
        "reason": "Inconsistent style can be smoothed.",
    },
    "close_paraphrase": {
        "fixability": FIXABILITY_PARTIAL,
        "action": "rewrite_from_source_card",
        "scope": "paragraph",
        "reason": "Can revise expression, but source grounding must be preserved.",
        "required_inputs": ["source_card", "citation"],
    },
    "patchwriting": {
        "fixability": FIXABILITY_PARTIAL,
        "action": "rewrite_from_source_card",
        "scope": "paragraph",
        "reason": "Can revise expression, but source grounding must be preserved.",
        "required_inputs": ["source_card", "citation"],
    },
    "semantic_overlap": {
        "fixability": FIXABILITY_PARTIAL,
        "action": "rewrite_from_source_card",
        "scope": "paragraph",
        "reason": "Can revise expression, but source grounding must be preserved.",
        "required_inputs": ["source_card"],
    },
    "paragraph_level_overlap": {
        "fixability": FIXABILITY_PARTIAL,
        "action": "rewrite_from_source_card",
        "scope": "paragraph",
        "reason": "Can revise expression, but source grounding must be preserved.",
        "required_inputs": ["source_card"],
    },
    "exact_copy": {
        "fixability": FIXABILITY_PROTECTED,
        "action": "manual_quotation_required",
        "scope": "span",
        "reason": "Direct quotes must not be altered automatically.",
        "required_inputs": ["source_quote"],
    },
    "missing_from_bib": {
        "fixability": FIXABILITY_MANUAL,
        "action": "manual_reference_fix",
        "scope": "span",
        "reason": "Rewriting cannot create a valid bibliography entry.",
        "required_inputs": ["reference"],
    },
    "uncited_in_body": {
        "fixability": FIXABILITY_MANUAL,
        "action": "manual_citation_required",
        "scope": "span",
        "reason": "Rewriting cannot create a valid source.",
        "required_inputs": ["source", "citation"],
    },
    "uncited_claim": {
        "fixability": FIXABILITY_MANUAL,
        "action": "manual_citation_required",
        "scope": "span",
        "reason": "Rewriting cannot create a valid source.",
        "required_inputs": ["source", "citation"],
    },
    "direct_quote_mismatch": {
        "fixability": FIXABILITY_PROTECTED,
        "action": "manual_quote_review",
        "scope": "span",
        "reason": "Direct quotes must not be altered automatically.",
        "required_inputs": ["source_quote"],
    },
}

DEFAULT_ROUTING = {
    "fixability": FIXABILITY_MANUAL,
    "action": "review_manually",
    "scope": "sentence",
    "reason": "Unknown finding type — requires manual review.",
}


def route_finding(finding: Finding) -> FixabilityDecision:
    """Route a finding to its fixability bucket.

    Honours detect-pipeline actionability when available:
    - review_only / manual_required → force manual
    - auto_fixable / auto_rewrite_candidate → force auto
    - Otherwise → fall through to FINDING_ROUTING table
    """
    # If detect already classified fixability, honour it
    # But allow review_only findings with a recommendation to be rewritten
    # (they were classified before the expanded rephrasable_types fix)
    if finding.actionability in ("review_only",):
        has_route = finding.finding_type in FINDING_ROUTING
        has_rec = bool(getattr(finding, "recommendation", ""))
        if has_route and has_rec:
            route = FINDING_ROUTING[finding.finding_type]
            # Override fixability: if the table says manual, upgrade to partial
            # so the rewrite engine will actually attempt rephrasing
            fix = route["fixability"]
            if fix == FIXABILITY_MANUAL:
                fix = FIXABILITY_PARTIAL
            return FixabilityDecision(
                finding_id=getattr(finding, "id", str(id(finding))),
                finding_type=finding.finding_type,
                fixability=fix,
                action="suggest_rewrite" if fix != FIXABILITY_MANUAL else route["action"],
                scope=route["scope"],
                reason=f"Review-only finding with recommendation — attempting rewrite.",
                required_inputs=route.get("required_inputs", []),
            )
    if finding.actionability in ("review_only", "manual_required", "no_action"):
        return FixabilityDecision(
            finding_id=getattr(finding, "id", str(id(finding))),
            finding_type=finding.finding_type,
            fixability="manual",
            action="review_manually",
            scope="sentence",
            reason=f"Detect classified as {finding.actionability}.",
        )
    if finding.actionability in ("auto_fixable", "auto_rewrite_candidate"):
        route = FINDING_ROUTING.get(finding.finding_type, DEFAULT_ROUTING)
        return FixabilityDecision(
            finding_id=getattr(finding, "id", str(id(finding))),
            finding_type=finding.finding_type,
            fixability="auto",
            action=route["action"],
            scope=route["scope"],
            reason=f"Detect classified as auto_fixable.",
            required_inputs=route.get("required_inputs", []),
        )

    # Default: route by finding_type table
    route = FINDING_ROUTING.get(finding.finding_type, DEFAULT_ROUTING)
    return FixabilityDecision(
        finding_id=getattr(finding, "id", str(id(finding))),
        finding_type=finding.finding_type,
        fixability=route["fixability"],
        action=route["action"],
        scope=route["scope"],
        reason=route["reason"],
        required_inputs=route.get("required_inputs", []),
    )


# ── Edit radius per scope ────────────────────────────────────────────

EDIT_RADIUS = {
    "span": {"max_char_delta": 0.20, "max_semantic_drift": 0.06},
    "sentence": {"max_char_delta": 0.35, "max_semantic_drift": 0.10},
    "sentence_pair": {"max_char_delta": 0.40, "max_semantic_drift": 0.11},
    "paragraph": {"max_char_delta": 0.45, "max_semantic_drift": 0.12},
}


# ── Rewrite scope by finding type ────────────────────────────────────

REWRITE_SCOPE = {
    "generic_phrase": "span",
    "high_predictability": "sentence",
    "medium_predictability": "sentence",
    "style_shift": "sentence_pair",
    "semantic_match": "paragraph",
    "uncited_claim": "manual",
    "missing_citation": "manual",
    "direct_quote_mismatch": "protected",
    "close_paraphrase": "paragraph",
    "patchwriting": "paragraph",
    "exact_copy": "protected",
}


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class RewriteAction:
    finding: Finding
    action_type: str
    scope: str          # "span" | "sentence" | "sentence_pair" | "paragraph" | "full"
    eligible: bool      # True = auto-rewritable (auto or partial fixability)
    fixability: str     # "auto" | "partial" | "manual" | "protected"
    weight: float
    source_card: Optional[dict] = None
    reason: str = ""
    required_inputs: List[str] = field(default_factory=list)


@dataclass
class RewritePlan:
    actions: List[RewriteAction]
    auto_fixable: List[RewriteAction]      # fixability in {auto, partial}, action != review_only
    manual_required: List[RewriteAction]   # fixability == manual
    protected: List[RewriteAction]         # fixability == protected
    review_only: List[RewriteAction]       # action == review_only (medium alone, review band)
    total_weighted_risk: float
    auto_risk: float
    manual_risk: float
    protected_risk: float
    # Rewritable-only scoring (auto + partial, excludes manual + protected + review-only)
    rewritable_risk: float


# ── Planner ──────────────────────────────────────────────────────────

class RewritePlanner:
    """Classify findings into fixability buckets and build a rewrite plan."""

    def plan(self, detect_results: List[DetectResult]) -> RewritePlan:
        # Collect all finding types for compound-signal check
        all_finding_types = set()
        for dr in detect_results:
            for f in dr.findings:
                all_finding_types.add(f.finding_type)

        has_companion = bool(all_finding_types & COMPANION_SIGNALS)

        actions = []
        for dr in detect_results:
            for f in dr.findings:
                decision = route_finding(f)

                # Compound-signal override: medium_predictability with companion → auto
                if (f.finding_type == "medium_predictability" and has_companion):
                    decision = FixabilityDecision(
                        finding_id=decision.finding_id,
                        finding_type=decision.finding_type,
                        fixability=FIXABILITY_AUTO,
                        action="suggest_rewrite",
                        scope="sentence",
                        reason="Medium predictability paired with companion signal → auto-rewrite.",
                    )

                eligible = decision.fixability in {FIXABILITY_AUTO, FIXABILITY_PARTIAL}
                weight = RISK_WEIGHTS.get(f.risk_level, 1)
                actions.append(RewriteAction(
                    finding=f,
                    action_type=decision.action,
                    scope=decision.scope,
                    eligible=eligible,
                    fixability=decision.fixability,
                    weight=weight,
                    reason=decision.reason,
                    required_inputs=decision.required_inputs,
                ))

        # Sort: highest weight first, then by scope (narrower first)
        scope_order = {"span": 0, "sentence": 1, "sentence_pair": 2, "paragraph": 3, "full": 4}
        actions.sort(key=lambda a: (-a.weight, scope_order.get(a.scope, 5)))

        auto_fixable = [a for a in actions if a.fixability in {FIXABILITY_AUTO, FIXABILITY_PARTIAL} and a.action_type != "review_only"]
        manual_required = [a for a in actions if a.fixability == FIXABILITY_MANUAL]
        protected = [a for a in actions if a.fixability == FIXABILITY_PROTECTED]
        review_only = [a for a in actions if a.action_type == "review_only"]

        return RewritePlan(
            actions=actions,
            auto_fixable=auto_fixable,
            manual_required=manual_required,
            protected=protected,
            review_only=review_only,
            total_weighted_risk=sum(a.weight for a in actions),
            auto_risk=sum(a.weight for a in auto_fixable),
            manual_risk=sum(a.weight for a in manual_required),
            protected_risk=sum(a.weight for a in protected),
            rewritable_risk=sum(a.weight for a in auto_fixable),
        )

    def plan_summary(self, plan: RewritePlan) -> str:
        """Human-readable summary of the plan."""
        lines = [
            f"Rewrite plan: {len(plan.auto_fixable)} auto-fixable, "
            f"{len(plan.manual_required)} manual, "
            f"{len(plan.protected)} protected, "
            f"{len(plan.review_only)} review-only",
        ]
        if plan.auto_fixable:
            lines.append("Auto-fixable:")
            for a in plan.auto_fixable[:10]:
                lines.append(f"  [{a.finding.risk_level}] {a.finding.finding_type} → {a.action_type} ({a.scope}, {a.fixability})")
        if plan.manual_required:
            lines.append("Manual required:")
            for a in plan.manual_required[:10]:
                lines.append(f"  [{a.finding.risk_level}] {a.finding.finding_type} → {a.action_type}")
        if plan.protected:
            lines.append("Protected (not changed):")
            for a in plan.protected[:10]:
                lines.append(f"  [{a.finding.risk_level}] {a.finding.finding_type} → {a.action_type}")
        if plan.review_only:
            lines.append("Review-only (not auto-rewritten):")
            for a in plan.review_only[:10]:
                lines.append(f"  [{a.finding.risk_level}] {a.finding.finding_type} → {a.action_type}")
        return "\n".join(lines)
