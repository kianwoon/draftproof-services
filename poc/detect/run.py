"""Detection orchestrator — runs all configured detectors on content.

Returns a DetectionReport with rule-based (non-averaged) risk aggregation.

Usage:
    from detect import DetectionRunner
    runner = DetectionRunner()                      # auto-detect domain
    runner = DetectionRunner(domain="legal")        # explicit domain
    report = runner.run_all(content, source_sentences=[...], bib_text="...")
"""

import hashlib
from typing import List, Optional, Dict, Any

from .base import BaseDetector, DetectResult, Finding, DetectionReport, RewriteDecision
from .predictability import PredictabilityDetector
from .similarity import SimilarityDetector
from .citation import CitationDetector
from .ai_generation import AIGenerationSignalDetector
from .postprocess import PostProcessor, PostProcessConfig
from .profiles import DomainProfile, resolve_profile
from .thresholds import ThresholdConfig

_FAIRNESS_CAVEATS = [
    "Formal academic writing and second-language writing may trigger "
    "predictability or style findings. Review findings alongside source "
    "grounding and draft history.",
]


class DetectionRunner:
    """Orchestrates multiple detectors. Auto-wires based on provided inputs.

    Args:
        detectors: Custom detector list (overrides defaults).
        postprocess: Enable post-processing filters.
        domain: Domain hint — "auto" to detect from content, or a specific
                domain name, or "default" for generic analysis.
        profile: Explicit DomainProfile (overrides domain hint).
    """

    def __init__(
        self,
        detectors: Optional[List[BaseDetector]] = None,
        postprocess: bool = True,
        domain: str = "auto",
        profile: Optional[DomainProfile] = None,
    ):
        self._custom_detectors = detectors
        self._postprocess = postprocess
        self._domain = domain
        self._profile = profile

    def run_all(self, content: str, **kwargs) -> DetectionReport:
        """Run all applicable detectors on content.

        Args:
            content: The text to analyze.
            **kwargs:
                source_sentences: List[str] — enables similarity detection
                source_paragraphs: List[str] — enables paragraph-level similarity
                bib_text: str — enables citation detection
                auto_extract_bibliography: bool — auto-split refs from content
                predictability_model: str — GPT-2 model name (default: gpt2-medium)
                domain_profile: str — domain tuning profile name
                phrase_packs: List[str] — override phrase packs
        """
        # Resolve profile: use explicit profile, or detect from content
        if self._profile is not None:
            profile = self._profile
        else:
            domain = kwargs.pop("domain_profile", self._domain)
            profile = resolve_profile(content, domain)

        detectors = self._build_detectors(profile=profile, **kwargs)
        # Inject domain_terms from profile if not explicitly provided
        if "domain_terms" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["domain_terms"] = profile.resolve_domain_terms(content)
        # Inject phrase packs from profile
        if "custom_phrases" not in kwargs:
            from poc.predictability.phrase_packs import get_phrases_for_packs
            kwargs = dict(kwargs)
            kwargs["custom_phrases"] = get_phrases_for_packs(profile.phrase_packs)
        scanner_results: List[DetectResult] = []
        for d in detectors:
            result = d.detect(content, **kwargs)
            scanner_results.append(result)
            # Forward predictability sentence metrics to ai_generation scanner
            if d.__class__.__name__ == "PredictabilityDetector":
                pred_raw = result.raw if hasattr(result, "raw") and result.raw else {}
                sentences = pred_raw.get("sentences", [])
                if sentences and "sentence_metrics" not in kwargs:
                    # Convert SentenceResult objects to dicts for criteria
                    sentence_dicts = []
                    for s in sentences:
                        if isinstance(s, dict):
                            sentence_dicts.append(s)
                        else:
                            sentence_dicts.append({
                                "sentence": getattr(s, "sentence", ""),
                                "avg_surprisal": getattr(s, "avg_surprisal", 5.0),
                                "top_10_ratio": getattr(s, "top_10_ratio", 0.0),
                                "top_50_ratio": getattr(s, "top_50_ratio", 0.0),
                                "predictability_risk": getattr(s, "predictability_risk", 0.0),
                                "risk_label": getattr(s, "risk_label", ""),
                            })
                    kwargs = dict(kwargs)
                    kwargs["sentence_metrics"] = sentence_dicts
                if "predictability_value" not in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["predictability_value"] = result.overall_risk

        # Apply post-processing filters to each scanner result
        pp_results = []
        if self._postprocess:
            processor = PostProcessor(
                config=profile.postprocess,
                domain_terms=kwargs.get("domain_terms"),
            )
            processed_results = []
            for r in scanner_results:
                pp = processor.process(r, original_text=content)
                pp_results.append(pp)
                processed_results.append(pp.result)
            scanner_results = processed_results

        top_findings = self._select_top_findings(scanner_results)
        overall_risk = self._aggregate_risk(scanner_results)
        overall_priority = self._compute_priority(scanner_results)
        overall_confidence = self._aggregate_confidence(scanner_results)
        caveats = self._collect_caveats(scanner_results)

        summary = self._generate_summary(scanner_results, overall_risk, overall_priority)

        # Classify actionability for each finding
        self._classify_actionability(top_findings, scanner_results)

        # Compute actionability distribution
        actionability_dist = self._compute_actionability_distribution(top_findings)

        # Compute rewrite decision
        rewrite_decision = self._compute_rewrite_decision(
            top_findings, actionability_dist, overall_risk
        )

        # Collect criterion scores from AIGenerationSignalDetector
        criterion_scores = self._collect_criterion_scores(scanner_results)

        return DetectionReport(
            overall_review_priority=overall_priority,
            overall_risk=overall_risk,
            confidence=overall_confidence,
            scanner_results=scanner_results,
            top_findings=top_findings,
            caveats=caveats,
            summary=summary,
            postprocess_results=pp_results,
            rewrite_decision=rewrite_decision,
            actionability_distribution=actionability_dist,
            criterion_scores=criterion_scores,
        )

    def run_single(self, content: str, scanner: str, **kwargs) -> DetectResult:
        """Run a single detector by name."""
        detectors = {
            "predictability": lambda: PredictabilityDetector(
                model_name=kwargs.get("predictability_model", "gpt2-medium")
            ),
            "ai_generation": lambda: AIGenerationSignalDetector(),
            "similarity": lambda: SimilarityDetector(),
            "citation": lambda: CitationDetector(
                auto_extract_bibliography=kwargs.get("auto_extract_bibliography", True)
            ),
        }
        if scanner not in detectors:
            raise ValueError(f"Unknown scanner: {scanner}. Choose from: {list(detectors.keys())}")
        return detectors[scanner]().detect(content, **kwargs)

    def _build_detectors(self, profile: Optional[DomainProfile] = None, **kwargs) -> List[BaseDetector]:
        if self._custom_detectors:
            return self._custom_detectors

        detectors: List[BaseDetector] = [
            PredictabilityDetector(
                model_name=kwargs.get("predictability_model", "gpt2-medium")
            ),
            AIGenerationSignalDetector(),
        ]
        if kwargs.get("source_sentences"):
            detectors.append(SimilarityDetector())
        if kwargs.get("bib_text") or kwargs.get("auto_extract_bibliography", True):
            detectors.append(CitationDetector(
                auto_extract_bibliography=kwargs.get("auto_extract_bibliography", True)
            ))
        return detectors

    @staticmethod
    def _aggregate_risk(results: List[DetectResult]) -> float:
        """Rule-based risk aggregation. Never simply averages scores."""
        has_high_similarity = False
        has_no_citation = False
        has_high_citation_risk = False
        has_unsupported_claims = False
        has_high_predictability = False
        has_high_ai_likelihood = False
        has_polished_ungrounded = False
        has_structural_high_only = False
        ai_likelihood_score = 0.0

        for r in results:
            if r.scanner == "similarity" and r.overall_risk >= 0.7:
                has_high_similarity = True
                for f in r.findings:
                    if f.risk_level == "high" and not f.metadata.get("citation_nearby"):
                        has_no_citation = True
            if r.scanner == "citation":
                if r.overall_risk >= 0.6:
                    has_high_citation_risk = True
                for f in r.findings:
                    if f.finding_type == "uncited_claim":
                        has_unsupported_claims = True
            if r.scanner == "predictability" and r.overall_risk >= 0.6:
                has_high_predictability = True
            if r.scanner == "ai_generation":
                ai_likelihood_score = r.likelihood_score
                if r.likelihood_score >= 0.60:  # elevated or high
                    has_high_ai_likelihood = True
                for f in r.findings:
                    if f.finding_type == "polished_but_ungrounded":
                        has_polished_ungrounded = True

        # Structural signal guardrail: check if only structural signals are HIGH
        structural_types = {"uniform_paragraph_structure", "repetitive_sentence_structure", "low_burstiness"}
        all_high_findings = []
        for r in results:
            all_high_findings.extend(f for f in r.findings if f.risk_level == "high")
        if all_high_findings:
            non_structural_high = [f for f in all_high_findings if f.finding_type not in structural_types]
            if not non_structural_high:
                has_structural_high_only = True

        # Corroborating signals for structural guardrail
        has_corroborating_signal = (
            has_high_ai_likelihood
            or has_high_predictability
            or has_high_similarity
            or has_high_citation_risk
            or has_polished_ungrounded
        )

        # Rule-based priority (ordered by severity)
        if has_high_similarity and has_no_citation:
            return 0.9
        if has_high_citation_risk and has_unsupported_claims:
            return 0.8
        if has_high_ai_likelihood and has_polished_ungrounded:
            return 0.8
        if has_high_similarity:
            return 0.7
        if has_high_ai_likelihood:
            return 0.7
        if has_high_citation_risk:
            return 0.65
        if has_high_predictability and has_unsupported_claims:
            return 0.55
        if has_high_ai_likelihood:
            return 0.5

        # Structural guardrail: structural alone cannot produce HIGH tier
        if has_structural_high_only and not has_corroborating_signal:
            if ai_likelihood_score < 0.25:
                return 0.35  # cap at MEDIUM, not HIGH
            return 0.4

        if has_high_predictability:
            return 0.4

        # Fallback: take max scanner risk
        return max((r.overall_risk for r in results), default=0.0)

    @staticmethod
    def _compute_priority(results: List[DetectResult]) -> str:
        risk = DetectionRunner._aggregate_risk(results)
        if risk >= 0.7:
            return "high"
        if risk >= 0.4:
            return "medium"
        return "low"

    @staticmethod
    def _aggregate_confidence(results: List[DetectResult]) -> str:
        levels = {"high": 3, "medium": 2, "low": 1}
        scores = [levels.get(r.confidence, 1) for r in results]
        if not scores:
            return "low"
        avg = sum(scores) / len(scores)
        if avg >= 2.5:
            return "high"
        if avg >= 1.5:
            return "medium"
        return "low"

    @staticmethod
    def _select_top_findings(results: List[DetectResult], limit: int = 10) -> List[Finding]:
        all_findings = []
        for r in results:
            all_findings.extend(r.findings)
        all_findings.sort(key=lambda f: {"high": 3, "medium": 2, "low": 1}.get(f.risk_level, 0),
                          reverse=True)
        return all_findings[:limit]

    @staticmethod
    def _collect_caveats(results: List[DetectResult]) -> List[str]:
        caveats = list(_FAIRNESS_CAVEATS)
        for r in results:
            if r.policy_message and r.policy_message not in caveats:
                caveats.append(r.policy_message)
        return caveats

    @staticmethod
    def _collect_criterion_scores(results: List[DetectResult]) -> dict:
        """Extract criterion scores from AIGenerationSignalDetector's raw output."""
        for r in results:
            if r.scanner != "ai_generation":
                continue
            raw = r.raw if hasattr(r, "raw") and r.raw else {}
            criteria_list = raw.get("criteria", [])
            if not criteria_list:
                continue
            scores = {}
            for c in criteria_list:
                name = c.get("name") if isinstance(c, dict) else getattr(c, "name", None)
                value = c.get("value") if isinstance(c, dict) else getattr(c, "value", None)
                if name and value is not None:
                    scores[name] = c if isinstance(c, dict) else c
            return scores
        return {}

    @staticmethod
    def _generate_summary(results: List[DetectResult], risk: float,
                          priority: str) -> str:
        parts = []
        for r in results:
            finding_count = len(r.findings)
            high_count = sum(1 for f in r.findings if f.risk_level == "high")
            if finding_count == 0:
                parts.append(f"{r.scanner}: no findings")
            elif high_count > 0:
                parts.append(f"{r.scanner}: {high_count} high-priority finding(s) out of {finding_count}")
            else:
                parts.append(f"{r.scanner}: {finding_count} finding(s)")
        detail = "; ".join(parts)
        return f"Review priority: {priority} (risk {risk:.0%}). {detail}."

    @staticmethod
    def _classify_actionability(top_findings: List[Finding], results: List[DetectResult]) -> None:
        """Set actionability on each finding in-place."""
        citation_types = {"uncited_claim", "missing_reference", "unused_reference", "verify_reference"}
        structural_types = {"uniform_paragraph_structure", "repetitive_sentence_structure", "low_burstiness"}
        ai_gen_scanners = {"predictability", "ai_generation"}

        # Get AI likelihood for context
        ai_likelihood = 0.0
        for r in results:
            if r.scanner == "ai_generation":
                ai_likelihood = r.likelihood_score

        # Finding types that can be mitigated by rephrasing
        rephrasable_types = {
            "generic_phrase", "high_predictability", "low_surprisal",
            "close_paraphrase", "topk_predictability", "low_specificity",
        }

        for f in top_findings:
            if f.finding_type in citation_types:
                f.actionability = "citation_repair"
            elif f.finding_type in structural_types:
                f.actionability = "optional_structure_review"
            elif f.risk_level in ("high", "medium") and (
                f.suggested_action_type in ("add_specific_example", "add_user_interpretation")
                or f.finding_type in rephrasable_types
            ):
                f.actionability = "auto_rewrite_candidate"
            elif f.risk_level == "low" or "minimal" in f.finding_type:
                f.actionability = "no_action"
            elif f.risk_level in ("high", "medium"):
                f.actionability = "review_only"
            else:
                f.actionability = "review_only"

            # Downgrade auto_rewrite_candidate if AI likelihood is low
            if f.actionability == "auto_rewrite_candidate" and ai_likelihood < 0.25:
                f.actionability = "review_only"

    @staticmethod
    def _compute_actionability_distribution(findings: List[Finding]) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for f in findings:
            key = f.actionability or "review_only"
            dist[key] = dist.get(key, 0) + 1
        return dist

    @staticmethod
    def _compute_rewrite_decision(
        findings: List[Finding],
        actionability_dist: Dict[str, int],
        overall_risk: float,
    ) -> RewriteDecision:
        """Detection owns the rewrite gate — the rewrite engine never guesses."""
        auto_count = actionability_dist.get("auto_rewrite_candidate", 0)
        citation_count = actionability_dist.get("citation_repair", 0)
        structural_count = actionability_dist.get("optional_structure_review", 0)

        # No rewrite if only structural/review/citation findings
        if auto_count == 0:
            allowed = []
            if citation_count > 0:
                allowed.append("citation_repair")
            if structural_count > 0:
                allowed.append("optional_structure_review")

            if citation_count > 0 and structural_count == 0:
                reason = "Citation repair needed, not content rewrite."
            elif structural_count > 0 and citation_count == 0:
                reason = "Structural signals only. No content rewrite justified."
            else:
                reason = "No auto-fixable findings. Signals are review-only."

            return RewriteDecision(
                run_rewrite=False,
                reason=reason,
                mode="none",
                targets=[],
                full_rewrite_allowed=False,
                allowed_actions=allowed,
            )

        # Targeted rewrite for auto-fixable findings
        targets = [f"i" for i, f in enumerate(findings)
                    if f.actionability == "auto_rewrite_candidate"]
        # Use stable IDs from finding metadata if available
        target_ids = []
        for f in findings:
            if f.actionability == "auto_rewrite_candidate":
                fid = f.metadata.get("finding_id", f"idx_{id(f)}")
                target_ids.append(fid)

        return RewriteDecision(
            run_rewrite=True,
            reason=f"{auto_count} auto-fixable finding(s) detected.",
            mode="targeted",
            targets=target_ids,
            full_rewrite_allowed=overall_risk >= 0.8,
            allowed_actions=["auto_rewrite_candidate"],
        )
