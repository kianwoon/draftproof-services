"""AI-Generation Signal Detector — composite multi-criteria classifier.

Consumes criteria from ``detect/criteria/`` and produces a weighted composite
AI-generation likelihood score.  Requires multiple aligned signals before
assigning medium or high likelihood.

Phase 1–2: rule-based classifier.  Phase 3 (ML) is future work.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .base import BaseDetector, DetectResult, Finding
from .criteria import CriterionScore, ALL_CRITERIA


# --- Thresholds (5-band) ---

LIKELIHOOD_THRESHOLDS = {
    "high": 0.75,
    "elevated": 0.60,
    "moderate": 0.40,
    "low": 0.20,
    "minimal": 0.00,
}

# Multi-signal rules: require N aligned criteria at specified levels
MULTI_SIGNAL_RULES = [
    # (min_criteria, min_level, result_label)
    (3, "high", "elevated_ai_generation_likelihood"),
    (2, "high", "moderate_ai_generation_likelihood"),
    (4, "medium", "moderate_ai_generation_likelihood"),
    (2, "medium", "low_ai_generation_likelihood"),
]

POLICY_MESSAGE = (
    "AI-generation likelihood is a probabilistic assessment based on multiple "
    "statistical and linguistic signals. It does not constitute proof of AI authorship. "
    "Formal academic writing, second-language writing, and template-based assignments "
    "may produce elevated scores. Always review findings in context."
)


class AIGenerationSignalDetector(BaseDetector):
    """Composite detector that scores text on multiple AI-generation criteria."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self._custom_weights = weights

    @property
    def name(self) -> str:
        return "ai_generation"

    @property
    def detector_version(self) -> str:
        return "0.1.0"

    @property
    def policy_message(self) -> str:
        return POLICY_MESSAGE

    def detect(self, content: str, **kwargs) -> DetectResult:
        """Run all criteria and combine into composite AI-generation score.

        Kwargs forwarded to criteria:
            sentence_metrics: list[dict] from PredictabilityScanner (avoids double GPT-2)
            domain_terms: list[str] for specificity criterion
            custom_phrases: list[str] override for generic_phrases criterion
            predictability_value: float (0-1) from predictability detector
            specificity_value: float (0-1) from specificity criterion
            citation_findings_count: int from citation detector
        """
        # Resolve weights
        weights = self._custom_weights or {name: w for name, _, w in ALL_CRITERIA}

        # Score each criterion
        criteria_results: List[CriterionScore] = []
        feature_summary: Dict[str, float] = {}

        for criterion_name, score_fn, default_weight in ALL_CRITERIA:
            weight = weights.get(criterion_name, default_weight)
            if weight == 0:
                # Diagnostic-only criteria (e.g. paragraph_uniformity) still run
                pass

            result = score_fn(content, **kwargs)
            criteria_results.append(result)
            feature_summary[criterion_name] = result.value

        # Weighted composite score
        total_weight = sum(weights.get(cr.name, 0) for cr in criteria_results)
        if total_weight > 0:
            composite = sum(
                weights.get(cr.name, 0) * cr.value for cr in criteria_results
            ) / total_weight
        else:
            composite = 0.0
        composite = min(1.0, max(0.0, composite))

        # Classify using multi-signal rules (override pure threshold)
        label = self._classify_by_rules(criteria_results)
        if label is None:
            # Fallback to threshold-based
            if composite >= LIKELIHOOD_THRESHOLDS["high"]:
                label = "elevated_ai_generation_likelihood"
            elif composite >= LIKELIHOOD_THRESHOLDS["elevated"]:
                label = "moderate_ai_generation_likelihood"
            elif composite >= LIKELIHOOD_THRESHOLDS["low"]:
                label = "low_ai_generation_likelihood"
            else:
                label = "minimal_ai_generation_likelihood"

        # Build findings
        findings = self._build_findings(label, composite, criteria_results)

        # Confidence
        confidence, confidence_reason = self._assess_confidence_ai(content, criteria_results)

        # Risk distribution
        dist = {}
        for f in findings:
            dist[f.risk_level] = dist.get(f.risk_level, 0) + 1

        # Overall risk from composite
        overall_risk = min(1.0, composite)

        return DetectResult(
            scanner=self.name,
            overall_risk=round(overall_risk, 4),
            confidence=confidence,
            confidence_reason=confidence_reason,
            risk_distribution=dist,
            findings=findings,
            policy_message=self.policy_message,
            raw={
                "criteria": [
                    {"name": cr.name, "value": cr.value, "label": cr.label, "details": cr.details}
                    for cr in criteria_results
                ],
                "composite_score": round(composite, 4),
                "label": label,
            },
            detector_version=self.detector_version,
            classification_target="ai_generated_content",
            likelihood_score=round(composite, 4),
            feature_summary=feature_summary,
        )

    def _classify_by_rules(self, results: List[CriterionScore]) -> Optional[str]:
        """Multi-signal rule classification. Returns None if no rule matches."""
        high_count = sum(1 for r in results if r.label == "high")
        medium_count = sum(1 for r in results if r.label in ("high", "medium"))

        for min_criteria, min_level, result_label in MULTI_SIGNAL_RULES:
            if min_level == "high" and high_count >= min_criteria:
                return result_label
            if min_level == "medium" and medium_count >= min_criteria:
                return result_label

        return None

    def _build_findings(
        self,
        label: str,
        composite: float,
        results: List[CriterionScore],
    ) -> List[Finding]:
        findings: List[Finding] = []

        # Top-level likelihood finding
        if "high" in label or "elevated" in label:
            risk = "high"
            evidence = "strong"
        elif "medium" in label or "moderate" in label:
            risk = "medium"
            evidence = "moderate"
        else:
            risk = "low"
            evidence = "weak"

        # Describe contributing criteria
        high_criteria = [r.name for r in results if r.label == "high"]
        medium_criteria = [r.name for r in results if r.label == "medium"]

        detail_parts = []
        if high_criteria:
            detail_parts.append(f"High signals: {', '.join(high_criteria)}")
        if medium_criteria:
            detail_parts.append(f"Medium signals: {', '.join(medium_criteria)}")

        detail = f"AI-generation likelihood: {label} (score {composite:.0%}). "
        detail += "; ".join(detail_parts) if detail_parts else "No strong individual signals."

        findings.append(Finding(
            finding_type=label,
            risk_level=risk,
            evidence_strength=evidence,
            detail=detail,
            evidence=", ".join(high_criteria[:3]) if high_criteria else "composite",
            recommendation=self._recommendation_for_label(label),
            suggested_action_type="review_manually",
            metadata={"composite_score": round(composite, 4)},
        ))

        # Per-criterion findings for high/medium criteria
        for r in results:
            if r.label in ("high", "medium") and r.value >= 0.35:
                # Build user-facing detail: lead with adjusted score, not raw
                if r.name == "low_specificity" and r.details:
                    adj_risk = r.details.get("adjusted_specificity_risk")
                    adj_label = r.details.get("adjusted_risk_label", "review")
                    if adj_risk is not None and abs(adj_risk - r.value) > 0.05:
                        detail = (
                            f"Specificity concern: adjusted to {adj_risk:.0%} {adj_label}. "
                            f"Raw signal: {r.value:.0%}. "
                        )
                        adj_reason = r.details.get("adjustment", {}).get("reason", "")
                        if adj_reason:
                            detail += f"Reason: {adj_reason}. "
                    else:
                        detail = f"{r.name} score: {r.value:.0%} ({r.label}). "
                else:
                    detail = f"{r.name} score: {r.value:.0%} ({r.label}). "
                detail += (self._format_details(r.details) if r.details else "")
                findings.append(Finding(
                    finding_type=self._criterion_to_finding_type(r.name),
                    risk_level="medium" if r.label == "medium" else "high",
                    evidence_strength="moderate",
                    detail=detail,
                    evidence=r.flagged_excerpts[0][:200] if r.flagged_excerpts else "",
                    recommendation=self._recommendation_for_criterion(r.name),
                    suggested_action_type=self._action_for_criterion(r.name),
                    # Carry the anchored excerpts in metadata so the report builder can
                    # surface them (the report rebuilds `evidence`, so passing them via
                    # metadata is the only way they survive to the user-facing report).
                    metadata={**(r.details or {}),
                              "flagged_excerpts": list(r.flagged_excerpts or [])[:3]},
                ))

        return findings

    def _assess_confidence_ai(
        self, content: str, results: List[CriterionScore]
    ) -> tuple:
        """Confidence with AI-specific downgrades."""
        word_count = len(content.split())

        if word_count < 50:
            base = ("low", "Text too short (<50 words); AI-generation signals unreliable.")
        elif word_count < 150:
            base = ("low", "Text <150 words; multi-signal analysis underpowered.")
        elif word_count < 300:
            base = ("medium", "Moderate text length; some signals may be underpowered.")
        else:
            base = ("high", "Sufficient length for multi-signal analysis.")

        confidence, reason = base

        # Downgrade if all criteria returned low (nothing to be confident about)
        if all(r.label == "low" for r in results):
            confidence = "low"
            reason += " All individual criteria are low."
        elif confidence == "high" and sum(1 for r in results if r.label == "high") < 2:
            confidence = "medium"
            reason += " Fewer than 2 high-signal criteria."

        return confidence, reason

    @staticmethod
    def _format_details(details: dict) -> str:
        parts = []
        # Keys that are counts/averages (not ratios) — display as numbers, not percentages
        raw_number_keys = {"avg_paragraph_words", "paragraph_count", "word_count",
                           "sentence_count", "domain_term_count", "named_entities"}
        for k, v in details.items():
            if isinstance(v, float):
                if k in raw_number_keys:
                    parts.append(f"{k}: {v:.1f}")
                else:
                    parts.append(f"{k}: {v:.0%}")
            else:
                parts.append(f"{k}: {v}")
        return "Details: " + ", ".join(parts)

    @staticmethod
    def _criterion_to_finding_type(name: str) -> str:
        mapping = {
            "low_surprisal": "low_surprisal_pattern",
            "topk_predictability": "high_topk_predictability",
            "low_burstiness": "low_burstiness",
            "generic_phrase_density": "generic_formulaic_language",
            "low_specificity": "low_specificity",
            "repetitive_structure": "repetitive_sentence_structure",
            "style_shift": "ai_like_style_shift",
            "citation_grounding_gap": "polished_but_ungrounded",
            "paragraph_uniformity": "uniform_paragraph_structure",
        }
        return mapping.get(name, name)

    @staticmethod
    def _recommendation_for_label(label: str) -> str:
        if "high" in label or "elevated" in label:
            return "Review text for AI-generated content. Check for concrete details, personal voice, and source grounding."
        if "moderate" in label:
            return "Some AI-like patterns detected. Add specific examples, personal observations, or source-based analysis."
        if "low" in label:
            return "Few AI-like patterns. Minor review recommended."
        return "Minimal AI-like patterns. Standard review recommended."

    @staticmethod
    def _recommendation_for_criterion(name: str) -> str:
        recs = {
            "low_surprisal": "Rephrase with less common word choices or more specific terminology.",
            "topk_predictability": "Replace predictable phrasing with unexpected or domain-specific terms.",
            "low_burstiness": "Vary sentence structure — mix short and long sentences.",
            "generic_phrase_density": "Replace generic transitions with content-specific connectors.",
            "low_specificity": "Review domain-term detection results. If auto-detected terms were missed, specificity may be understated.",
            "repetitive_structure": "Vary sentence openings and syntactic patterns.",
            "style_shift": "Review section for consistency with surrounding text voice.",
            "citation_grounding_gap": "Add specific citations, examples, or source-based analysis.",
            "paragraph_uniformity": "Vary paragraph lengths and structure.",
        }
        return recs.get(name, "Review and revise flagged passages.")

    @staticmethod
    def _action_for_criterion(name: str) -> str:
        actions = {
            "low_surprisal": "add_specific_example",
            "topk_predictability": "add_user_interpretation",
            "low_burstiness": "optional_structure_review",
            "generic_phrase_density": "add_specific_example",
            "low_specificity": "add_specific_example",
            "repetitive_structure": "optional_structure_review",
            "style_shift": "review_manually",
            "citation_grounding_gap": "citation_repair",
            "paragraph_uniformity": "optional_structure_review",
        }
        return actions.get(name, "review_manually")
