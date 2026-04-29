"""DraftProof Analyzer — integrated writing-process risk analysis.

Combines all modules into a single analysis pipeline:
- Predictability Scanner
- Style Analyzer
- Structural Fingerprinter
- Cross-Draft Diff Engine
- Confidence Bands
- Domain Phrase Packs

Produces a RiskReport with multi-axis scores and explainable findings.
Not an AI detector. Produces review-priority signals only.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict

try:
    from .risk_report import (
        RiskReport, AxisScore, ReviewPriority, ConfidenceLevel,
        word_count_to_confidence, compute_review_priority,
    )
    from .confidence import should_suppress_verdict, cap_verdict
    from .phrase_packs import PhrasePackLoader
except ImportError:
    from risk_report import (
        RiskReport, AxisScore, ReviewPriority, ConfidenceLevel,
        word_count_to_confidence, compute_review_priority,
    )
    from confidence import should_suppress_verdict, cap_verdict
    from phrase_packs import PhrasePackLoader


@dataclass
class AnalyzerConfig:
    domains: List[str] = None          # phrase pack domains to load
    reference_draft: Optional[str] = None  # previous draft for comparison
    scanner_model: str = "gpt2-medium"

    def __post_init__(self):
        if self.domains is None:
            self.domains = ["generic_academic"]


class DraftProofAnalyzer:
    """Unified writing-process risk analyzer."""

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()

        # Load phrase packs
        self._packs = PhrasePackLoader()
        for domain in self.config.domains:
            self._packs.load_domain(domain)

        # Lazy-load heavy modules
        self._scanner = None
        self._style_analyzer = None
        self._fingerprinter = None
        self._diff_engine = None

    # ── Main analysis entry point ──────────────────────────────────

    def analyze(self, text: str,
                reference_draft: Optional[str] = None) -> RiskReport:

        word_count = len(text.split())
        confidence = word_count_to_confidence(word_count)

        # 1. Predictability risk
        pred_risk, pred_reasons = self._run_predictability(text)

        # 2. Style uniformity risk
        style_risk, style_reasons = self._run_style(text)

        # 3. Structural reuse risk (needs reference draft)
        struct_risk = None
        struct_reasons = []
        ref = reference_draft or self.config.reference_draft
        if ref:
            struct_risk, struct_reasons = self._run_structural(text, ref)

        # 4. Draft evolution risk (needs reference draft)
        evo_risk = None
        evo_reasons = []
        if ref:
            evo_risk, evo_reasons = self._run_evolution(text, ref, struct_risk)

        # 5. Source grounding (placeholder — not yet implemented)
        source_risk = None
        source_reasons = []

        # 6. Citation integrity (placeholder)
        cite_risk = None
        cite_reasons = []

        # 7. Authorship evidence (placeholder)
        auth_risk = None
        auth_reasons = []

        # Build axis scores
        pred_axis = AxisScore("predictability_risk", pred_risk, confidence, pred_reasons)
        style_axis = AxisScore("style_uniformity_risk", style_risk, confidence, style_reasons)
        struct_axis = AxisScore("structural_reuse_risk", struct_risk, confidence, struct_reasons) if struct_risk is not None else None
        source_axis = AxisScore("source_grounding_risk", source_risk, confidence, source_reasons) if source_risk is not None else None
        evo_axis = AxisScore("draft_evolution_risk", evo_risk, confidence, evo_reasons) if evo_risk is not None else None
        cite_axis = AxisScore("citation_integrity_risk", cite_risk, confidence, cite_reasons) if cite_risk is not None else None
        auth_axis = AxisScore("authorship_evidence_strength", auth_risk, confidence, auth_reasons) if auth_risk is not None else None

        # Compute overall review priority
        priority, reasons = compute_review_priority(
            predictability=pred_risk,
            style_uniformity=style_risk,
            structural_reuse=struct_risk,
            source_grounding=source_risk,
            draft_evolution=evo_risk,
            citation_integrity=cite_risk,
            confidence=confidence,
        )

        # Apply confidence caps
        capped_risk = 0.0
        note = ""
        if not should_suppress_verdict(confidence):
            capped_risk, note = cap_verdict(confidence, sum(
                s.score for s in [pred_axis, style_axis]
                + ([a for a in [struct_axis, source_axis, evo_axis, cite_axis, auth_axis]
                    if a is not None])
            ) / max(len([a for a in [struct_axis, source_axis, evo_axis, cite_axis, auth_axis]
                         if a is not None]) + 2, 1))

        if note:
            reasons.append(note)

        return RiskReport(
            predictability_risk=pred_axis,
            style_uniformity_risk=style_axis,
            structural_reuse_risk=struct_axis,
            source_grounding_risk=source_axis,
            draft_evolution_risk=evo_axis,
            citation_integrity_risk=cite_axis,
            authorship_evidence_strength=auth_axis,
            overall_review_priority=priority,
            review_reasons=reasons,
            confidence=confidence,
        )

    # ── Module runners ─────────────────────────────────────────────

    def _run_predictability(self, text: str) -> tuple:
        try:
            from .scanner import PredictabilityScanner
        except ImportError:
            from scanner import PredictabilityScanner
        if self._scanner is None:
            phrases = self._packs.get_all_phrases()
            self._scanner = PredictabilityScanner(
                model_name=self.config.scanner_model,
                custom_phrases=phrases,
            )
        result = self._scanner.scan_text(text)
        reasons = []
        high = sum(1 for s in result['sentences'] if s.predictability_risk >= 0.50)
        med = sum(1 for s in result['sentences'] if 0.35 <= s.predictability_risk < 0.50)
        if high > 0:
            reasons.append(f"{high} of {len(result['sentences'])} sentences scored high predictability")
        if med > len(result['sentences']) * 0.5:
            reasons.append(f"{med} of {len(result['sentences'])} sentences scored medium predictability")
        return result['overall_risk'], reasons

    def _run_style(self, text: str) -> tuple:
        from style_analyzer import StyleAnalyzer
        if self._style_analyzer is None:
            self._style_analyzer = StyleAnalyzer()
        profile = self._style_analyzer.analyze(text)
        reasons = [d.finding for d in profile.dimensions if d.finding and d.score >= 0.35]
        return profile.overall_style_risk, reasons

    def _run_structural(self, text: str, reference: str) -> tuple:
        try:
            from .structural_fingerprint import StructuralFingerprinter
        except ImportError:
            from structural_fingerprint import StructuralFingerprinter
        if self._fingerprinter is None:
            self._fingerprinter = StructuralFingerprinter()
        fp_a = self._fingerprinter.fingerprint(reference)
        fp_b = self._fingerprinter.fingerprint(text)
        comp = self._fingerprinter.compare(fp_a, fp_b)
        reasons = []
        if comp.structural_reuse_risk >= 0.70:
            reasons.append("Near-identical argument structure to reference draft")
        elif comp.structural_reuse_risk >= 0.50:
            reasons.append("Significant structural overlap with reference draft")
        if comp.topic_sequence_similarity >= 0.70:
            reasons.append("Topic sequence closely matches reference")
        if comp.function_sequence_similarity >= 0.70:
            reasons.append("Paragraph function sequence closely matches reference")
        return comp.structural_reuse_risk, reasons

    def _run_evolution(self, text: str, reference: str,
                       struct_risk: Optional[float]) -> tuple:
        try:
            from .cross_draft_diff import CrossDraftEngine
        except ImportError:
            from cross_draft_diff import CrossDraftEngine
        if self._diff_engine is None:
            self._diff_engine = CrossDraftEngine()
        diff = self._diff_engine.diff(reference, text,
                                      structural_similarity=struct_risk)
        reasons = []
        if diff.draft_evolution_risk >= 0.60:
            reasons.append("Wording changed but argument did not develop")
        if diff.revision_quality.surface_rewrite >= 0.60:
            reasons.append(f"{diff.revision_quality.surface_rewrite:.0%} of changes are surface-level")
        if diff.revision_metrics.new_claims_added == 0:
            reasons.append("No new claims added between drafts")
        return diff.draft_evolution_risk, reasons
