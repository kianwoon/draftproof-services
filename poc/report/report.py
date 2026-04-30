"""DraftProof Report -- structured report from detection and rewrite results.

Aggregates DetectResult + rewrite data into a single DraftReport with
tiered risk scoring.

Two ways to build:
  1. New unified API:  builder.add_detection(result)  — accepts DetectResult
  2. Legacy API:       builder.add_predictability(dict) / add_similarity(obj) / add_citation(obj)

Run:  cd poc/report && python demo.py
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Risk tiers ──────────────────────────────────────────────────────

class Tier(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CLEAN = "clean"


TIER_ORDER = [Tier.CRITICAL, Tier.HIGH, Tier.MEDIUM, Tier.LOW, Tier.CLEAN]
TIER_ICON = {
    Tier.CRITICAL: "[!]",
    Tier.HIGH: "[H]",
    Tier.MEDIUM: "[M]",
    Tier.LOW: "[L]",
    Tier.CLEAN: "[=]",
}

_RISK_LEVEL_TO_TIER = {
    "high": Tier.HIGH,
    "medium": Tier.MEDIUM,
    "low": Tier.LOW,
}


# ── Findings (unified across scanners) ──────────────────────────────

@dataclass
class Finding:
    """Single actionable finding from any scanner."""
    tier: Tier
    category: str
    scanner: str
    title: str
    detail: str
    evidence: str
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    finding_id: str = ""
    raw_risk: str = ""
    adjusted_risk: str = ""
    sentence_id: str = ""
    signal_category: str = ""   # "writing_quality" | "genericity" | "predictability" | "authorship_risk"


# ── Section summaries ───────────────────────────────────────────────

@dataclass
class PredictabilitySummary:
    overall_risk: float
    risk_distribution: Dict[str, int]
    sentences: List[Dict[str, Any]]
    style_shifts: List[Dict[str, Any]]
    generic_phrases_found: List[str]


@dataclass
class SimilaritySummary:
    overall_risk: str
    risk_distribution: Dict[str, int]
    matches: List[Dict[str, Any]]


@dataclass
class CitationSummary:
    citation_style: str
    in_text_count: int
    bib_entry_count: int
    findings: List[Dict[str, Any]]
    stats: Dict[str, int]


@dataclass
class RewriteSummary:
    original_risk: float
    original_top10: float
    final_risk: float
    final_top10: float
    passes_completed: int
    converged: bool
    convergence_reason: str
    improvement_risk: float
    improvement_top10: float
    pass_progression: List[Dict[str, float]]
    original_tier: str = ""
    original_findings: int = 0
    original_distribution: Dict[str, int] = None
    rewritten_tier: str = ""
    rewritten_findings: int = 0
    rewritten_distribution: Dict[str, int] = None
    sentence_comparison: List[Dict[str, Any]] = None
    # Post-rewrite detect verification
    post_detect_findings: int = 0
    post_detect_distribution: Dict[str, int] = None
    post_detect_improvement: int = 0
    detect_loops_used: int = 0
    detect_loop_history: List[Dict[str, Any]] = None
    reverted: bool = False
    revert_reason: str = ""
    # New multi-signal fields
    weighted_risk_original: float = 0.0
    weighted_risk_final: float = 0.0
    drift_score: float = 0.0
    protected_spans_count: int = 0
    regression_rejections: int = 0
    auto_fixable_count: int = 0
    manual_required_count: int = 0
    protected_count: int = 0
    manual_actions: List[Dict[str, str]] = None
    protected_actions: List[Dict[str, str]] = None
    # Fixability-aware scoring
    rewritable_risk_original: float = 0.0
    rewritable_risk_final: float = 0.0
    # Outcome classification
    outcome: str = ""  # "improved" | "partially_improved" | "floor_reached" | "manual_required" | "rejected_for_drift"
    floor_reasons: List[Dict[str, str]] = None
    # Voice & surface
    voice_preserved: bool = True
    voice_warnings: List[str] = None
    rewrite_surface_ratio: float = 1.0


# ── Main report ─────────────────────────────────────────────────────

@dataclass
class DraftReport:
    """Structured report from a full DraftProof scan."""
    overall_tier: Tier
    finding_count: int
    findings_by_tier: Dict[str, List[Finding]]

    predictability: Optional[PredictabilitySummary] = None
    similarity: Optional[SimilaritySummary] = None
    citation: Optional[CitationSummary] = None
    rewrite: Optional[RewriteSummary] = None
    manual_actions: List[Dict[str, str]] = None

    scan_time_seconds: float = 0.0
    generated_at: str = ""
    original_text: str = ""
    rewritten_text: str = ""
    false_positives: Optional[List[Dict[str, str]]] = None
    raw_overall_tier: str = ""
    adjusted_overall_tier: str = ""
    overall_tier_reason: str = ""
    rewrite_priority_tier: str = ""
    rewrite_priority_reason: str = ""
    rewrite_decision: Optional[Dict[str, Any]] = None
    actionability_distribution: Optional[Dict[str, int]] = None
    def to_dict(self) -> dict:
        """Serialize the report to a JSON-ready dict."""
        import json
        plan = report_to_dict(self)
        plan["original_text"] = self.original_text
        plan["rewritten_text"] = self.rewritten_text
        return plan

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to a JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ── Builder ─────────────────────────────────────────────────────────


def determine_actionability(f: "Finding", all_findings: list = None) -> str:
    """Classify finding into actionability bucket.

    Module-level function so both ReportBuilder.build() and report_to_dict()
    can call it without duplication.
    """
    adj = f.adjusted_risk.lower()
    title = f.title
    # No-action: low-risk signals
    if "low_ai_generation" in title or "minimal_ai_generation" in title or "low_specificity_likelihood" in title:
        return "no_action"
    if adj in ("low", "review", "clean"):
        return "review_only"
    # Protected: quotes, citations
    if f.category in ("citation", "integrity") or "quote" in title:
        # Citation findings are citation_repair (suggest but don't rewrite)
        if f.category == "citation":
            return "citation_repair"
        return "manual_required"
    # High predictability: auto-fixable only if score is clearly high AND top10 is high
    if "high_predictability" in title or "high_topk_predictability" in title:
        # Gate: require both high score AND high top10 ratio for auto-fix
        score = f.metadata.get("score", 0) if f.metadata else 0
        top10 = f.metadata.get("top10_ratio", 0) if f.metadata else 0
        if score >= 0.60 and top10 >= 0.70:
            return "auto_fixable"
        # High but not extreme: auto-fix only if a document-level signal confirms
        # (low_specificity, uncited_claim) OR a co-located sentence-level signal exists
        if all_findings:
            _DOC_SIGNALS = {"low_specificity", "uncited_claim"}
            doc_paired = any(
                other.title in _DOC_SIGNALS
                for other in all_findings
                if other.finding_id != f.finding_id
                and other.adjusted_risk.lower() not in ("low", "review", "clean")
            )
            if doc_paired:
                return "auto_fixable"
            _COLOCATED_SIGNALS = {
                "generic_phrase", "similarity_overlap",
                "style_shift", "weak_source_grounding",
            }
            if f.sentence_id:
                colocated = any(
                    other.title in _COLOCATED_SIGNALS
                    for other in all_findings
                    if other.finding_id != f.finding_id
                    and other.sentence_id == f.sentence_id
                    and other.adjusted_risk.lower() not in ("low", "review", "clean")
                )
                if colocated:
                    return "auto_fixable"
        return "review_only"
    # Document-level low_specificity: auto-fixable only if NOT already downgraded
    # If AI likelihood is low and domain grounding is strong, specificity is review-level
    if title == "low_specificity":
        has_adjustment = (
            f.metadata
            and isinstance(f.metadata, dict)
            and f.metadata.get("adjustment")
        )
        if has_adjustment:
            return "review_only"
        return "auto_fixable"
    # Medium predictability: review_only unless paired with a sentence-level signal
    # AT THE SAME LOCATION. Document-level signals (uncited_claim, low_specificity)
    # apply to the whole text — they do NOT justify rewriting every medium sentence.
    # Only co-located signals (same sentence_id) justify auto-fixing.
    if "medium_predictability" in title:
        if all_findings and f.sentence_id:
            _COLOCATED_SIGNALS = {
                "generic_phrase", "similarity_overlap",
                "style_shift", "weak_source_grounding",
            }
            paired = any(
                other.title in _COLOCATED_SIGNALS
                for other in all_findings
                if other.finding_id != f.finding_id
                and other.sentence_id == f.sentence_id
                and other.adjusted_risk.lower() not in ("low", "review", "clean")
            )
            if paired:
                return "auto_fixable"
        return "review_only"
    # uniform_paragraph_structure with downgrade adjustment: optional review, not rewrite
    if title == "uniform_paragraph_structure":
        has_downgrade = (
            f.metadata
            and isinstance(f.metadata, dict)
            and f.metadata.get("adjustment", {}).get("filter") == "UniformStructureDowngrade"
        )
        if has_downgrade:
            return "optional_structure_review"
    # Other high/medium AI-generation signals: auto-fixable
    if adj in ("high", "medium") and f.scanner == "ai_generation":
        return "auto_fixable"
    return "review_only"


class ReportBuilder:
    """Builds a DraftReport from detect + rewrite results."""

    def __init__(self):
        self._findings: List[Finding] = []
        self._pred_summary: Optional[PredictabilitySummary] = None
        self._sim_summary: Optional[SimilaritySummary] = None
        self._cite_summary: Optional[CitationSummary] = None
        self._rewrite_summary: Optional[RewriteSummary] = None
        self._original_text = ""
        self._rewritten_text = ""
        self._scan_time = 0.0
        self._summaries: Dict[str, Any] = {}
        self._false_positives: List[Dict[str, str]] = []
        self._sentence_id_map: Dict[str, str] = {}

    def set_meta(self, scan_time: float = 0.0,
                 original_text: str = "", rewritten_text: str = ""):
        self._scan_time = scan_time
        self._original_text = original_text
        self._rewritten_text = rewritten_text
        self._generated_at = self._sgt_now()
        return self

    @staticmethod
    def _sgt_now() -> str:
        from datetime import datetime, timezone, timedelta
        sgt = timezone(timedelta(hours=8))
        return datetime.now(sgt).strftime("%Y-%m-%d %H:%M SGT")

    def add_postprocess_results(self, pp_results: list) -> "ReportBuilder":
        """Extract false-positive reclassifications from PostProcessResult list."""
        for pp in pp_results:
            raw = pp.raw_findings
            filtered = pp.result.findings
            for i, orig in enumerate(raw):
                matched = filtered[i] if i < len(filtered) else None
                if matched and matched.metadata.get("reclassified"):
                    reason = matched.metadata.get("reclassify_reason", "unknown")
                    snippet = (orig.evidence or orig.detail or "")[:60]
                    # Infer filter from reason text
                    filter_name = ""
                    if "domain-term" in reason:
                        filter_name = "GlossaryFilter"
                    elif "quoted term" in reason or "citation" in reason.lower() or "academic" in reason.lower():
                        filter_name = "AcademicFilter"
                    elif "severity" in reason.lower() or "threshold" in reason.lower() or "within" in reason:
                        filter_name = "SeverityAdjustor"
                    self._false_positives.append({
                        "original_risk": orig.risk_level,
                        "adjusted_risk": matched.risk_level,
                        "reason": reason,
                        "filter": filter_name,
                        "sentence": snippet,
                    })
        return self

    # ── NEW: Unified detection input ─────────────────────────────────

    def add_detection(self, result) -> "ReportBuilder":
        """Accept a DetectResult from the detect module.

        Works for any detector — predictability, similarity, or citation.
        """
        try:
            from poc.detect.base import DetectResult
        except ImportError:
            from detect.base import DetectResult

        if not isinstance(result, DetectResult):
            raise TypeError(f"Expected DetectResult, got {type(result).__name__}")

        scanner = result.scanner

        # Route to the right summary builder
        if scanner == "predictability":
            self._build_predictability_summary(result)
        elif scanner == "similarity":
            self._build_similarity_summary(result)
        elif scanner == "citation":
            self._build_citation_summary(result)

        # Convert all findings to report Findings with tiers
        for f in result.findings:
            tier = _RISK_LEVEL_TO_TIER.get(f.risk_level, Tier.LOW)
            # Upgrade certain finding types
            if f.finding_type == "exact_copy":
                tier = Tier.CRITICAL
            # Match sentence_id from predictability sentences
            sent_id = ""
            if scanner == "predictability" and self._sentence_id_map:
                snippet = (f.evidence or "")[:60]
                sent_id = self._sentence_id_map.get(snippet, "")
            # Inject scanner-level metadata into finding metadata
            meta = dict(f.metadata) if f.metadata else {}
            if scanner == "ai_generation" and result.likelihood_score:
                meta["ai_likelihood"] = result.likelihood_score
            # NOTE: document-level predictability_risk is NOT injected into
            # per-finding metadata — it belongs in document_context only.
            # Each finding already carries its own sentence-level score.
            # Determine signal_category from the detect Finding
            signal_cat = getattr(f, 'signal_category', '')
            if not signal_cat:
                try:
                    from detect.base import classify_finding_category
                    signal_cat = classify_finding_category(f.finding_type)
                except ImportError:
                    signal_cat = "predictability"
            self._findings.append(Finding(
                tier=tier,
                category=scanner,
                scanner=scanner,
                title=f.finding_type,
                detail=f.detail,
                evidence=f.evidence,
                recommendation=f.recommendation,
                metadata=meta,
                sentence_id=sent_id,
                signal_category=signal_cat,
            ))

        return self

    def add_detection_report(self, detection_report) -> "ReportBuilder":
        """Accept a DetectionReport from the detect module's DetectionRunner.

        Feeds all scanner results through add_detection(), and stores
        confidence, caveats, and summary metadata.
        """
        try:
            from poc.detect.base import DetectionReport as DR
        except ImportError:
            from detect.base import DetectionReport as DR

        if not isinstance(detection_report, DR):
            raise TypeError(
                f"Expected DetectionReport, got {type(detection_report).__name__}"
            )

        for dr in detection_report.scanner_results:
            self.add_detection(dr)

        self._summaries["overall_confidence"] = detection_report.confidence
        self._summaries["caveats"] = detection_report.caveats
        self._summaries["detection_summary"] = detection_report.summary
        self._summaries["overall_review_priority"] = (
            detection_report.overall_review_priority
        )

        # Thread rewrite_decision and actionability_distribution
        if hasattr(detection_report, "rewrite_decision") and detection_report.rewrite_decision:
            from dataclasses import asdict
            rd = detection_report.rewrite_decision
            self._summaries["rewrite_decision"] = asdict(rd) if hasattr(rd, "__dataclass_fields__") else {}
        if hasattr(detection_report, "actionability_distribution") and detection_report.actionability_distribution:
            self._summaries["actionability_distribution"] = dict(detection_report.actionability_distribution)

        return self

    def _build_predictability_summary(self, result: "DetectResult"):
        raw = result.raw
        if raw is None:
            return

        sentences = []
        generic_phrases = []
        sent_idx = 0
        for s in raw.get("sentences", []):
            if getattr(s, "error", None):
                continue
            sent_idx += 1
            sent_id = f"s{sent_idx:03d}"
            sentences.append({
                "sentence_id": sent_id,
                "sentence": s.sentence,
                "risk_label": s.risk_label,
                "risk": s.predictability_risk,
                "avg_probability": s.avg_probability,
                "avg_surprisal": s.avg_surprisal,
                "top10_ratio": s.top_10_ratio,
                "top50_ratio": s.top_50_ratio,
                "start_char": getattr(s, "start_char", 0),
                "end_char": getattr(s, "end_char", 0),
                "paragraph_id": getattr(s, "paragraph_id", ""),
            })
            for p in getattr(s, "matched_generic_phrases", []):
                generic_phrases.append(p)

        # Build sentence_id lookup by truncated text for matching findings
        self._sentence_id_map = {}
        for sent in sentences:
            snippet = sent["sentence"][:60]
            self._sentence_id_map[snippet] = sent["sentence_id"]

        self._pred_summary = PredictabilitySummary(
            overall_risk=result.overall_risk,
            risk_distribution=result.risk_distribution,
            sentences=sentences,
            style_shifts=raw.get("style_shifts", []),
            generic_phrases_found=generic_phrases,
        )

    def _build_similarity_summary(self, result: "DetectResult"):
        raw = result.raw
        if raw is None:
            return

        risk_str = "high" if result.overall_risk >= 0.7 else (
            "medium" if result.overall_risk >= 0.4 else "low"
        )
        matches = []
        for f in result.findings:
            matches.append({
                "draft_sentence": f.metadata.get("draft_sentence", ""),
                "source_sentence": f.metadata.get("source_sentence", ""),
                "risk_type": f.finding_type,
                "risk_level": f.risk_level,
                "exact_score": f.metadata.get("exact_score", 0),
                "fuzzy_score": f.metadata.get("fuzzy_score", 0),
                "semantic_score": f.metadata.get("semantic_score", 0),
                "citation_nearby": f.metadata.get("citation_nearby", False),
                "recommendation": f.recommendation,
            })

        self._sim_summary = SimilaritySummary(
            overall_risk=risk_str,
            risk_distribution=result.risk_distribution,
            matches=matches,
        )

    def _build_citation_summary(self, result: "DetectResult"):
        raw = result.raw
        if raw is None:
            return

        findings_list = []
        for f in result.findings:
            findings_list.append({
                "finding_type": f.finding_type,
                "risk_level": f.risk_level,
                "detail": f.detail,
                "evidence": f.evidence,
                "recommendation": f.recommendation,
            })

        self._cite_summary = CitationSummary(
            citation_style=raw.citation_style if raw else "unknown",
            in_text_count=len(raw.in_text_citations) if raw else 0,
            bib_entry_count=len(raw.bib_entries) if raw else 0,
            findings=findings_list,
            stats=raw.stats if raw else {},
        )

    # ── Rewrite ──────────────────────────────────────────────────────

    def add_rewrite(self, mp, post_detect_results=None, **kwargs) -> "ReportBuilder":
        """Accept a MultiPassResult + optional post-rewrite detect results."""
        progression = []
        progression.append({
            "pass": 0,
            "risk": mp.original_metrics.risk,
            "top10": mp.original_metrics.top10_ratio,
            "surprisal": mp.original_metrics.surprisal,
        })
        for p in mp.passes:
            progression.append({
                "pass": p.pass_number,
                "risk": p.risk,
                "top10": p.top10_ratio,
                "surprisal": p.surprisal,
            })

        improvement_risk = mp.original_metrics.risk - mp.final_metrics.risk
        improvement_top10 = mp.original_metrics.top10_ratio - mp.final_metrics.top10_ratio

        # Build per-sentence comparison from original vs final metrics
        sentence_comparison = []
        orig_details = mp.original_metrics.sentence_details or []
        final_details = mp.final_metrics.sentence_details or []
        max_idx = max(len(orig_details), len(final_details))
        for i in range(max_idx):
            o = orig_details[i] if i < len(orig_details) else {}
            f = final_details[i] if i < len(final_details) else {}
            sentence_comparison.append({
                "index": i + 1,
                "orig_tier": o.get("label", "?"),
                "orig_risk": o.get("risk", 0),
                "orig_top10": o.get("top10_ratio", 0),
                "orig_sentence": o.get("sentence", "")[:80],
                "new_tier": f.get("label", "?"),
                "new_risk": f.get("risk", 0),
                "new_top10": f.get("top10_ratio", 0),
                "new_sentence": f.get("sentence", "")[:80],
            })

        # Compute tiers and distributions from sentence details
        def _tier_from_risk(risk: float) -> str:
            if risk >= 0.7:
                return "CRITICAL"
            elif risk >= 0.55:
                return "HIGH"
            elif risk >= 0.4:
                return "MEDIUM"
            elif risk >= 0.25:
                return "LOW"
            return "CLEAN"

        def _dist_from_details(details: list) -> Dict[str, int]:
            dist = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for d in details:
                label = (d.get("label") or "").lower()
                if label in dist:
                    dist[label] += 1
            return dist

        orig_dist = _dist_from_details(orig_details)
        final_dist = _dist_from_details(final_details)
        orig_tier = _tier_from_risk(mp.original_metrics.risk)
        final_tier = _tier_from_risk(mp.final_metrics.risk)
        orig_finding_count = sum(orig_dist.values())
        final_finding_count = sum(final_dist.values())

        # Post-rewrite detect verification
        post_detect_findings = 0
        post_detect_dist = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        if post_detect_results:
            for pdr in post_detect_results:
                for f in pdr.findings:
                    post_detect_findings += 1
                    lvl = (f.risk_level or "").lower()
                    if lvl in post_detect_dist:
                        post_detect_dist[lvl] += 1

        self._rewrite_summary = RewriteSummary(
            original_risk=mp.original_metrics.risk,
            original_top10=mp.original_metrics.top10_ratio,
            final_risk=mp.final_metrics.risk,
            final_top10=mp.final_metrics.top10_ratio,
            passes_completed=len(mp.passes),
            converged=mp.converged,
            convergence_reason=mp.convergence_reason,
            improvement_risk=improvement_risk,
            improvement_top10=improvement_top10,
            pass_progression=progression,
            sentence_comparison=sentence_comparison,
            original_tier=orig_tier,
            original_findings=orig_finding_count,
            original_distribution=orig_dist,
            rewritten_tier=final_tier,
            rewritten_findings=final_finding_count,
            rewritten_distribution=final_dist,
            post_detect_findings=post_detect_findings,
            post_detect_distribution=post_detect_dist,
            post_detect_improvement=orig_finding_count - post_detect_findings,
            detect_loops_used=kwargs.get("detect_loops_used", 0),
            detect_loop_history=kwargs.get("detect_loop_history", None),
            reverted=kwargs.get("reverted", False),
            revert_reason=kwargs.get("revert_reason", ""),
        )
        self._rewritten_text = mp.final_text

        if improvement_top10 < 0.05 and mp.original_metrics.top10_ratio > 0.55:
            self._findings.append(Finding(
                tier=Tier.MEDIUM,
                category="rewrite",
                scanner="rewriter",
                title="Text near predictability floor",
                detail=f"Rewrite only improved top-10 ratio by {improvement_top10:.0%} "
                       f"over {len(mp.passes)} pass(es)",
                evidence=f"Converged: {mp.convergence_reason}",
                recommendation="The remaining predictability is likely structural. "
                               "Consider restructuring the argument or adding original analysis.",
            ))

        return self

    # ── Build ────────────────────────────────────────────────────────

    @staticmethod
    def _derive_weighted_tier(findings: List[Finding], raw_tier: Tier) -> Tier:
        """Weighted multi-signal tier derivation.

        Instead of blunt 'highest finding wins', considers:
        - Critical findings always escalate
        - Similarity/citation high findings escalate
        - low_specificity only escalates to HIGH if paired with high AI-generation
        - Predictability only escalates if overall risk is high enough
        """
        critical_count = sum(1 for f in findings if f.tier == Tier.CRITICAL)
        high_count = sum(1 for f in findings if f.tier == Tier.HIGH)
        medium_count = sum(1 for f in findings if f.tier == Tier.MEDIUM)
        manual_count = sum(1 for f in findings
                          if determine_actionability(f, findings) == "manual_required")

        high_titles = {f.title for f in findings if f.tier == Tier.HIGH}
        high_categories = {f.category for f in findings if f.tier == Tier.HIGH}

        # Extract AI-generation likelihood from findings metadata
        ai_likelihood = 0.0
        for f in findings:
            if f.metadata and isinstance(f.metadata, dict):
                ai_likelihood = max(ai_likelihood, f.metadata.get("ai_likelihood", 0.0))

        # Extract predictability overall risk
        pred_risk = 0.0
        for f in findings:
            if f.metadata and isinstance(f.metadata, dict):
                pred_risk = max(pred_risk, f.metadata.get("predictability_risk", 0.0))

        # Extract specificity confidence
        low_spec_confidence = "moderate"
        for f in findings:
            if f.title == "low_specificity" and f.metadata:
                dt_count = f.metadata.get("domain_term_count", 0)
                ne_count = f.metadata.get("named_entities", 0)
                if dt_count == 0 and ne_count > 20:
                    low_spec_confidence = "low"
                elif dt_count > 0:
                    low_spec_confidence = "high"

        # Critical always escalates
        if critical_count > 0:
            return Tier.CRITICAL

        # Uniform paragraph structure alone should not force HIGH.
        # It is a layout/style signal, not an authorship-risk signal.
        high_risk_titles = high_titles - {"uniform_paragraph_structure"}
        effective_high_count = len(high_risk_titles)

        # Similarity/integrity high findings always escalate
        if "similarity" in high_categories and effective_high_count <= 2:
            return Tier.HIGH

        # Citation integrity issues
        citation_high = sum(1 for f in findings if f.tier == Tier.HIGH
                           and f.category == "citation")
        if citation_high >= 3:
            return Tier.HIGH

        # low_specificity HIGH: only escalate to HIGH if AI-likelihood is also high
        has_low_spec_high = "low_specificity" in high_risk_titles
        if has_low_spec_high:
            if ai_likelihood >= 0.50:
                return Tier.HIGH
            elif ai_likelihood >= 0.30:
                return Tier.MEDIUM
            else:
                # Low AI likelihood — probably a false-positive specificity score
                # Check predictability as secondary signal
                if pred_risk >= 0.55:
                    return Tier.HIGH
                elif pred_risk >= 0.40:
                    return Tier.MEDIUM
                return Tier.MEDIUM if manual_count > 0 else Tier.LOW

        # High predictability overall risk
        if pred_risk >= 0.60:
            return Tier.HIGH
        if pred_risk >= 0.45:
            return Tier.MEDIUM

        # Manual-required issues
        if manual_count >= 2:
            return Tier.MEDIUM
        if manual_count == 1:
            return Tier.MEDIUM if medium_count >= 3 else Tier.LOW

        # Medium findings count
        if medium_count >= 5 and high_count == 0:
            return Tier.MEDIUM

        # Fall through to raw tier, but cap at MEDIUM if no critical/high signals
        if raw_tier in (Tier.CRITICAL, Tier.HIGH) and effective_high_count == 0:
            return Tier.MEDIUM
        if raw_tier == Tier.HIGH:
            return Tier.HIGH

        return raw_tier

    def build(self) -> DraftReport:
        # Assign sequential finding IDs
        for i, f in enumerate(self._findings):
            f.finding_id = f"f{i+1:03d}"
            # Default: raw_risk = adjusted_risk = current tier
            if not f.raw_risk:
                f.raw_risk = f.tier.value
            if not f.adjusted_risk:
                f.adjusted_risk = f.tier.value

        # Adjust low_specificity risk when AI likelihood is low
        # If AI likelihood < 0.30 and specificity is "high", downgrade to "medium"
        # to avoid contradictory "AI risk low but AI-related specificity high"
        ai_likelihood_val = 0.0
        for f in self._findings:
            if "ai_generation" in f.title and f.metadata:
                ai_likelihood_val = f.metadata.get("ai_likelihood", 0.0)
                break
        for f in self._findings:
            if f.title == "low_specificity":
                # Always store raw values for transparency
                raw_risk = f.metadata.get("specificity_risk", 0.0) if f.metadata else 0.0
                raw_score = f.metadata.get("specificity_score", 0.0) if f.metadata else 0.0
                f.metadata["raw_specificity_concern"] = raw_risk
                f.metadata["raw_specificity_score"] = raw_score

                if ai_likelihood_val < 0.30 and f.tier == Tier.HIGH:
                    f.tier = Tier.MEDIUM
                    f.adjusted_risk = "medium"
                    adj_risk = round(raw_risk * (ai_likelihood_val / 0.30), 4)
                    f.metadata["adjusted_specificity_concern"] = adj_risk
                    f.metadata["adjustment"] = {
                        "reason": (
                            f"AI likelihood ({ai_likelihood_val:.1%}) is low and "
                            f"domain grounding is present; specificity concern "
                            f"downgraded from high to review-level"
                        ),
                        "filter": "AIContradictionAdjustment",
                        "raw_specificity_concern": raw_risk,
                        "adjusted_specificity_concern": adj_risk,
                    }
                    # Update finding detail to lead with adjusted score
                    f.detail = (
                        f"Specificity concern: adjusted to {adj_risk:.0%} review-level. "
                        f"Raw signal: {raw_risk:.0%}. "
                        f"Reason: AI likelihood is low and domain grounding is strong."
                    )
                else:
                    f.metadata["adjusted_specificity_concern"] = raw_risk

        # Downgrade uniform_paragraph_structure when AI likelihood is low
        # Layout/style signal alone should not force HIGH tier
        for f in self._findings:
            if f.title == "uniform_paragraph_structure" and f.tier == Tier.HIGH:
                spec_concern = 0.0
                for sf in self._findings:
                    if sf.title == "low_specificity" and sf.metadata:
                        spec_concern = max(spec_concern,
                            sf.metadata.get("adjusted_specificity_concern", 0.0))
                if ai_likelihood_val < 0.25 and spec_concern < 0.50:
                    f.tier = Tier.MEDIUM
                    f.adjusted_risk = "medium"
                    f.metadata["adjustment"] = {
                        "reason": (
                            f"Paragraph uniformity downgraded from high to medium: "
                            f"AI likelihood is {ai_likelihood_val:.1%} and "
                            f"specificity concern is low ({spec_concern:.1%}). "
                            f"Layout signal alone does not indicate authorship risk."
                        ),
                        "filter": "UniformStructureDowngrade",
                    }
                    f.detail = (
                        f"Uniform paragraph structure: adjusted to medium (advisory). "
                        f"Raw: {f.raw_risk}. "
                        f"Reason: low AI likelihood and good domain grounding."
                    )

        # Match false positives back to findings and set raw_risk/adjusted_risk
        for fp in self._false_positives:
            fp_snippet = fp.get("sentence", "")
            for f in self._findings:
                f_evidence = (f.evidence or "")[:60]
                if fp_snippet and f_evidence and fp_snippet in f_evidence:
                    f.raw_risk = fp["original_risk"]
                    f.adjusted_risk = fp["adjusted_risk"]
                    f.metadata["adjustment"] = {
                        "filter": fp.get("filter", ""),
                        "reason": fp.get("reason", ""),
                    }
                    break

        findings_by_tier: Dict[str, List[Finding]] = {t.value: [] for t in Tier}
        for f in self._findings:
            findings_by_tier[f.tier.value].append(f)

        # Raw overall tier (from scanner-assigned tiers)
        raw_tier = Tier.CLEAN
        for t in TIER_ORDER:
            if findings_by_tier.get(t.value):
                raw_tier = t
                break

        # Adjusted overall tier — weighted multi-signal derivation
        adjusted_tier = self._derive_weighted_tier(self._findings, raw_tier)

        # Generate tier reason — human-readable weighted derivation
        fp_filters = [fp.get("filter", "") for fp in self._false_positives]
        filter_counts = {}
        for fl in fp_filters:
            if fl:
                filter_counts[fl] = filter_counts.get(fl, 0) + 1
        downgraded_count = len(self._false_positives)
        reason_parts = []

        # Categorize findings by signal_category
        cat_counts = {}
        for f in self._findings:
            cat = f.signal_category or "predictability"
            if f.tier.value not in ("low", "clean"):
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        if raw_tier != adjusted_tier:
            reason_parts.append(
                f"Overall tier adjusted from {raw_tier.value.upper()} to "
                f"{adjusted_tier.value.upper()}."
            )

        # AI likelihood context
        ai_val = 0.0
        for f in self._findings:
            if "ai_generation" in f.title and f.metadata:
                ai_val = f.metadata.get("ai_likelihood", 0.0)
                break

        # Build human-readable tier reason
        tier_label = adjusted_tier.value.upper()
        # Count non-low findings
        non_low = sum(1 for f in self._findings if f.tier.value not in ("low", "clean"))
        # Check for citation/integrity findings
        citation_findings = [f for f in self._findings
                            if f.category in ("citation", "integrity")
                            and f.tier.value not in ("low", "clean")]
        # Check for high/critical findings
        has_high_critical = any(f.tier in (Tier.HIGH, Tier.CRITICAL) for f in self._findings)

        # Construct natural-language reason
        reasons = []
        # Predictability summary
        pred_count = cat_counts.get("predictability", 0)
        if pred_count > 0:
            reasons.append(f"{pred_count} normal academic predictability signal{'s' if pred_count != 1 else ''}")
        if citation_findings:
            reasons.append("a citation issue")
        generic_count = cat_counts.get("genericity", 0)
        if generic_count > 0:
            reasons.append(f"{generic_count} generic phrasing signal{'s' if generic_count != 1 else ''}")
        quality_count = cat_counts.get("writing_quality", 0)
        if quality_count > 0:
            reasons.append(f"{quality_count} writing quality note{'s' if quality_count != 1 else ''}")
        structure_count = cat_counts.get("structure", 0)
        # Also count uniform_paragraph_structure downgraded to medium
        if structure_count == 0:
            structure_count = sum(1 for f in self._findings
                                 if f.title == "uniform_paragraph_structure"
                                 and f.tier.value not in ("low", "clean"))
        if structure_count > 0:
            reasons.append(f"{structure_count} paragraph structure note{'s' if structure_count != 1 else ''}")

        if reasons:
            reason_parts.append(
                f"Overall tier is {tier_label} because the paper has "
                + ", ".join(reasons[:-1]) + (" and " if len(reasons) > 1 else "") + reasons[-1]
                + "."
            )
        else:
            reason_parts.append(f"Overall tier is {tier_label}.")

        ai_clause_parts = []
        if ai_val > 0:
            ai_clause_parts.append(f"AI likelihood is {ai_val:.1%}")
        if not has_high_critical:
            ai_clause_parts.append("no high or critical findings were detected")
        if ai_clause_parts:
            reason_parts.append(", and ".join(ai_clause_parts) + ".")

        # Predictability context (numeric detail)
        if self._pred_summary:
            pred_risk = getattr(self._pred_summary, 'overall_risk', None)
            if pred_risk is not None:
                reason_parts.append(f"Document predictability is {pred_risk:.1%}.")

        # Primary action
        high_findings = [f for f in self._findings if f.tier == Tier.HIGH]
        if high_findings:
            primary = high_findings[0]
            reason_parts.append(
                f"Primary concern: {primary.title.replace('_', ' ')} "
                f"({primary.signal_category or 'unknown'} category)."
            )

        if downgraded_count:
            filter_str = ", ".join(f"{c}x{n}" for c, n in filter_counts.items())
            reason_parts.append(f"{downgraded_count} finding(s) downgraded ({filter_str}).")

        overall_tier_reason = " ".join(reason_parts) if reason_parts else ""

        # Populate original detection stats into rewrite summary
        if self._rewrite_summary and self._pred_summary:
            self._rewrite_summary.original_tier = raw_tier.value.upper()
            self._rewrite_summary.original_findings = len(self._findings)
            self._rewrite_summary.original_distribution = dict(self._pred_summary.risk_distribution)

        # Assign FP IDs and link to findings
        for j, fp in enumerate(self._false_positives):
            fp["fp_id"] = f"fp{j+1:03d}"
            fp_snippet = fp.get("sentence", "")
            for f in self._findings:
                f_evidence = (f.evidence or "")[:60]
                if fp_snippet and f_evidence and fp_snippet in f_evidence:
                    fp["finding_id"] = f.finding_id
                    fp["sentence_id"] = f.sentence_id
                    break

        # Compute rewrite_priority_tier based on actionable issues via determine_actionability
        auto_fixable = [f for f in self._findings
                        if determine_actionability(f, self._findings) == "auto_fixable"]
        manual_required = [f for f in self._findings
                           if determine_actionability(f, self._findings) == "manual_required"]
        review_only = [f for f in self._findings
                       if determine_actionability(f, self._findings) == "review_only"]
        has_critical = any(f.category in ("citation", "similarity")
                          and f.adjusted_risk == "high"
                          for f in self._findings)
        if has_critical or len(auto_fixable) >= 4:
            rp_tier = "high"
        elif len(auto_fixable) >= 2:
            rp_tier = "medium"
        elif len(auto_fixable) >= 1:
            rp_tier = "low"
        else:
            rp_tier = "none"
        rp_parts = []
        if auto_fixable:
            rp_parts.append(f"{len(auto_fixable)} auto-fixable issue(s)")
        if manual_required:
            rp_parts.append(f"{len(manual_required)} manual-required issue(s)")
        rp_parts.append(f"{len(review_only)} review-only signal(s)")
        if auto_fixable:
            rp_parts.append(f"Use targeted {'specificity revision' if any(f.title == 'low_specificity' for f in auto_fixable) else 'rewrite'}")
        rp_reason = "; ".join(rp_parts) + "." if rp_parts else "No actionable rewrite issues."

        return DraftReport(
            overall_tier=adjusted_tier,
            finding_count=len(self._findings),
            findings_by_tier=findings_by_tier,
            predictability=self._pred_summary,
            similarity=self._sim_summary,
            citation=self._cite_summary,
            rewrite=self._rewrite_summary,
            scan_time_seconds=self._scan_time,
            generated_at=getattr(self, "_generated_at", ""),
            original_text=self._original_text,
            rewritten_text=self._rewritten_text,
            false_positives=self._false_positives or None,
            raw_overall_tier=raw_tier.value,
            adjusted_overall_tier=adjusted_tier.value,
            overall_tier_reason=overall_tier_reason,
            rewrite_priority_tier=rp_tier,
            rewrite_priority_reason=rp_reason,
            rewrite_decision=self._summaries.get("rewrite_decision"),
            actionability_distribution=self._summaries.get("actionability_distribution"),
        )


# ── Report to dict ──────────────────────────────────────────────────

def report_to_dict(report: DraftReport) -> Dict[str, Any]:
    """Convert report to JSON-serializable dict with full rewrite intelligence."""

    import re as _re

    def _structured_evidence(f: Finding) -> Any:
        """Build structured evidence for document-level findings."""
        if f.title == "low_specificity":
            # Use metadata directly if available (avoids regex-parsing detail string)
            if f.metadata and isinstance(f.metadata, dict):
                metrics = f.metadata
            else:
                detail = f.detail or ""
                metrics = {}
                for key in ["specificity_score", "named_entities", "numbers",
                             "dates", "domain_term_count", "word_count",
                             "abstract_noun_count", "abstract_noun_ratio"]:
                    m = _re.search(rf"'{key}':\s*([0-9.]+)", detail)
                    if m:
                        metrics[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
            raw_risk = metrics.get("raw_specificity_concern", metrics.get("raw_specificity_risk", metrics.get("specificity_risk", 0)))
            adj_risk = metrics.get("adjusted_specificity_concern", metrics.get("adjusted_specificity_risk", raw_risk))
            # Human-readable concern label
            if adj_risk < 0.30:
                display_concern = "low"
            elif adj_risk < 0.50:
                display_concern = "review-level"
            elif adj_risk < 0.70:
                display_concern = "moderate"
            else:
                display_concern = "high"
            dg_idx = metrics.get("domain_grounding_index", "")
            dg_level = metrics.get("domain_grounding_level", "")
            result_evidence = {
                "type": "document_level",
                "summary": (f"{int(metrics.get('word_count', 0))} words, "
                            f"{int(metrics.get('named_entities', 0))} named entities, "
                            f"{int(metrics.get('numbers', 0))} numbers, "
                            f"{int(metrics.get('dates', 0))} dates, "
                            f"{int(metrics.get('domain_term_count', 0))} domain terms"),
                "affected_span": "full_document",
                "metrics": metrics,
                "raw_specificity_concern": round(raw_risk, 4),
                "adjusted_specificity_concern": round(adj_risk, 4),
                "display_specificity_concern": display_concern,
            }
            if dg_idx:
                result_evidence["domain_grounding_index"] = dg_idx
                result_evidence["domain_grounding_level"] = dg_level
            adj = f.metadata.get("adjustment") if f.metadata else None
            if adj:
                result_evidence["adjustment_reason"] = adj.get("reason", "")
            return result_evidence
        return f.evidence

    _PAIRED_SIGNALS = {
        "generic_phrase", "low_specificity", "similarity_overlap",
        "uncited_claim", "style_shift", "weak_source_grounding",
        "high_predictability", "high_topk_predictability",
        "low_burstiness", "repetitive_sentence_structure",
    }

    def _determine_actionability(f: Finding, all_findings: list = None) -> str:
        """Delegate to module-level determine_actionability."""
        return determine_actionability(f, all_findings)

    def _tier_findings(tier: Tier) -> list:
        return [
            {
                "finding_id": f.finding_id,
                "category": f.category,
                "signal_category": f.signal_category or None,
                "title": f.title,
                "scanner": f.scanner,
                "score": f.metadata.get("score") if f.metadata else None,
                "top10_ratio": f.metadata.get("top10_ratio") if f.metadata else None,
                "subtype": f.metadata.get("subtype") if f.metadata else None,
                "raw_risk": f.raw_risk,
                "adjusted_risk": f.adjusted_risk,
                "actionability": _determine_actionability(f, all_findings),
                "sentence_id": f.sentence_id or None,
                "evidence": _structured_evidence(f),
                "recommendation": f.recommendation,
                "adjustment": f.metadata.get("adjustment") if f.metadata else None,
                "detail": f.detail,
            }
            for f in report.findings_by_tier.get(tier.value, [])
        ]

    all_findings = []
    for tier_val in ("critical", "high", "medium", "low"):
        all_findings.extend(report.findings_by_tier.get(tier_val, []))

    # Build rewrite plan
    auto_fixable = []
    review_only = []
    no_action = []
    manual_required = []
    citation_repairs = []
    priority = 0
    for f in all_findings:
        bucket = _determine_actionability(f, all_findings)
        entry = {
            "finding_id": f.finding_id,
            "title": f.title,
            "scanner": f.scanner,
        }
        if bucket == "citation_repair":
            entry["scope"] = "sentence"
            entry["action"] = "add_citation_or_link_existing_reference"
            entry["priority"] = len(citation_repairs) + 1
            entry["adjusted_risk"] = f.adjusted_risk
            entry["sentence_id"] = f.sentence_id or None
            entry["safe_auto_suggestion"] = True
            entry["requires_user_confirmation"] = True
            citation_repairs.append(entry)
        elif bucket == "auto_fixable":
            action = "suggest_rewrite"
            scope = "sentence"
            if f.title == "low_specificity":
                scope = "paragraph"
                action = "add_concrete_domain_context"
            elif "predictability" in f.title:
                action = "rewrite_with_personal_voice"
            entry["scope"] = scope
            entry["action"] = action
            entry["priority"] = priority
            entry["adjusted_risk"] = f.adjusted_risk
            auto_fixable.append(entry)
        elif bucket == "review_only":
            entry["reason"] = (f.metadata.get("adjustment", {}).get("reason", "")
                               if f.metadata else "")
            review_only.append(entry)
        elif bucket == "no_action":
            entry["reason"] = f.detail[:100] if f.detail else ""
            no_action.append(entry)
        elif bucket == "manual_required":
            entry["reason"] = "Requires manual intervention"
            manual_required.append(entry)
        elif bucket == "optional_structure_review":
            entry["reason"] = (
                f.metadata.get("adjustment", {}).get("reason", "Advisory structure signal")
                if f.metadata else "Advisory structure signal"
            )
            review_only.append(entry)

    # Determine rewrite mode and overall action
    if citation_repairs and not auto_fixable:
        rewrite_mode = "none"
        overall_action = "manual_citation_repair"
    elif citation_repairs and auto_fixable:
        rewrite_mode = "targeted"
        overall_action = "targeted_citation_and_rewrite"
    elif auto_fixable:
        rewrite_mode = "targeted" if len(auto_fixable) <= 3 else "comprehensive"
        top = auto_fixable[0]
        if top["title"] == "low_specificity":
            overall_action = "specificity_revision"
        else:
            overall_action = "predictability_revision"
    elif not auto_fixable and not citation_repairs and not manual_required:
        # Advisory-only: all findings are review-only or optional
        rewrite_mode = "none"
        overall_action = "optional_structure_review"
    else:
        overall_action = "review_only"
        rewrite_mode = "none"

    # Build primary goals from auto_fixable and citation_repair findings
    primary_goals = []
    primary_action = None
    # Citation repair goals go first
    for cr in citation_repairs:
        primary_goals.append(f"Add citation for {cr.get('title', 'claim').replace('_', ' ')} ({cr.get('finding_id', '')})")
    for af in auto_fixable:
        if af["action"] == "add_concrete_domain_context":
            primary_goals.append("Add domain-specific context and concrete examples")
        elif af["action"] == "rewrite_with_personal_voice":
            primary_goals.append(f"Rewrite high-predictability sentence ({af['finding_id']})")
        else:
            primary_goals.append(f"Address {af['title']} ({af['finding_id']})")
    # Add preservation goals from review_only with academic filter
    for fp in (report.false_positives or []):
        if fp.get("filter") == "AcademicFilter":
            sent = fp.get("sentence", "")
            # Extract quoted term
            m = _re.search(r"'([^']+)'", fp.get("reason", ""))
            if m:
                primary_goals.append(f"Preserve quoted term \"{m.group(1)}\"")

    # Promote citation/uncited actions above predictability noise
    citation_goals = [g for g in primary_goals if "uncited" in g.lower() or "citation" in g.lower()]
    # Also check manual_required for citation findings (e.g. missing_citation)
    citation_manual = [
        e for e in manual_required
        if "citation" in e.get("title", "").lower() or "uncited" in e.get("title", "").lower()
    ]
    non_citation_goals = [g for g in primary_goals if g not in citation_goals]
    primary_goals = citation_goals + non_citation_goals

    # Check if specificity has been downgraded to review-level
    specificity_is_review = False
    for f in all_findings:
        if f.title == "low_specificity" and f.adjusted_risk in ("review", "low", "clean"):
            specificity_is_review = True
            break

    # Set primary_action: citation needs trump predictability;
    # specificity only promoted if NOT already downgraded to review-level
    if citation_goals or citation_manual:
        primary_action = "add_citations"
    elif (any("specificity" in g.lower() for g in primary_goals)
          and not specificity_is_review):
        primary_action = "improve_specificity"
    elif any("predictability" in g.lower() or "rewrite" in g.lower() for g in primary_goals):
        primary_action = "reduce_formulaic_language"
    elif primary_goals:
        primary_action = "address_findings"
    else:
        primary_action = "review_only"

    # ── Tier derivation audit trail ─────────────────────────────────────
    tier_derivation = {
        "overall_tier": report.overall_tier.value,
        "raw_tier": report.raw_overall_tier,
        "adjusted_tier": report.adjusted_overall_tier,
        "reason": report.overall_tier_reason,
        "rewrite_priority_tier": report.rewrite_priority_tier,
    }
    # Add trigger info from low_specificity if present
    for f in all_findings:
        if f.title == "low_specificity" and f.metadata:
            tier_derivation["trigger"] = "low_specificity"
            tier_derivation["trigger_confidence"] = (
                "low" if f.metadata.get("domain_term_count", 0) == 0
                and f.metadata.get("named_entities", 0) > 10
                else "moderate"
            )
            tier_derivation["specificity_detail"] = {
                k: f.metadata.get(k) for k in
                ("raw_specificity_score", "raw_specificity_concern",
                 "adjusted_specificity_concern", "display_specificity_concern",
                 "named_entities", "numbers",
                 "domain_term_count", "domain_terms", "word_count")
                if k in f.metadata
            }
            break

    # ── Domain profile audit ─────────────────────────────────────────────
    domain_profile = {}
    for f in all_findings:
        if f.title == "low_specificity" and f.metadata:
            domain_profile = {
                "domain_term_count": f.metadata.get("domain_term_count", 0),
                "matched_domain_terms": f.metadata.get("domain_terms", []),
                "auto_detected": True,
            }
            break

    result: Dict[str, Any] = {
        "raw_overall_tier": report.raw_overall_tier,
        "adjusted_overall_tier": report.adjusted_overall_tier,
        "overall_tier": report.overall_tier.value,
        "overall_tier_reason": report.overall_tier_reason,
        "tier_derivation": tier_derivation,
        "domain_profile": domain_profile,
        "rewrite_priority_tier": report.rewrite_priority_tier,
        "rewrite_priority_reason": report.rewrite_priority_reason,
        "rewrite_decision": report.rewrite_decision,
        "actionability_distribution": report.actionability_distribution,
        "document_context": {
            "word_count": len(report.original_text.split()) if report.original_text else 0,
            "sentence_count": len(report.predictability.sentences) if report.predictability else 0,
        },
        "finding_count": report.finding_count,
        "findings": {
            "critical": _tier_findings(Tier.CRITICAL),
            "high": _tier_findings(Tier.HIGH),
            "medium": _tier_findings(Tier.MEDIUM),
            "low": _tier_findings(Tier.LOW),
        },
        "false_positives": report.false_positives,
        "rewrite_plan": {
            "mode": rewrite_mode,
            "overall_action": overall_action,
            "auto_fixable": auto_fixable,
            "review_only": review_only,
            "no_action": no_action,
            "manual_required": manual_required,
            "citation_repairs": citation_repairs,
        },
        "actionable_summary": {
            "rewrite_mode": rewrite_mode,
            "overall_action": overall_action,
            "primary_action": primary_action,
            "auto_rewrite_count": len(auto_fixable),
            "review_only_count": len(review_only),
            "manual_required_count": len(manual_required),
            "no_action_count": len(no_action),
            "citation_repair_count": len(citation_repairs),
            "auto_fixable": auto_fixable,
            "review_only": review_only,
            "manual_required": manual_required,
            "citation_repairs": citation_repairs,
            "primary_goals": primary_goals,
            "signal_categories": {
                cat: sum(1 for f in all_findings if f.signal_category == cat)
                for cat in ("writing_quality", "genericity",
                            "predictability", "authorship_risk")
                if any(f.signal_category == cat for f in all_findings)
            },
        },
    }

    # ── Rewrite constraints ─────────────────────────────────────────────
    preserve_terms = []
    for fp in (report.false_positives or []):
        if fp.get("filter") == "AcademicFilter":
            m = _re.search(r"'([^']+)'", fp.get("reason", ""))
            if m:
                preserve_terms.append(f'"{m.group(1)}"')
    # Also preserve terms from review_only findings
    for ro in review_only:
        if ro.get("title") == "review_predictability":
            ev = ro.get("evidence", "")
            # Extract quoted terms from evidence
            for qm in _re.finditer(r'"([^"]+)"', ev):
                preserve_terms.append(f'"{qm.group(1)}"')

    # Derive domain-specific safe additions from detected domain_terms
    domain_terms = []
    # Check findings evidence for domain_terms (populated from criterion metadata)
    for tier_name, flist in result.get("findings", {}).items():
        for f_info in flist:
            ev = f_info.get("evidence", {})
            if isinstance(ev, dict):
                dt = ev.get("metrics", {}).get("domain_terms", [])
                if isinstance(dt, list) and dt:
                    domain_terms = dt
                    break
        if domain_terms:
            break

    specificity_guidance = []
    if domain_terms:
        specificity_guidance.append(
            "concrete actions implied by existing terms: " + ", ".join(domain_terms[:6])
        )
    specificity_guidance.extend([
        "step-by-step process descriptions",
        "teacher/student interaction details",
        "technique-specific vocabulary already in text",
    ])

    result["rewrite_constraints"] = {
        "preserve_terms": preserve_terms,
        "do_not_add": [
            "new citation or reference",
            "unsupported study or statistic",
            "named entity not in original text (person, place, year)",
            "fabricated number or percentage",
            "date or year not in original text",
        ],
        "allowed_additions": specificity_guidance,
        "rewrite_rule": "If specificity is missing, add concrete domain action from implied context, never fabricated facts.",
        "max_change_scope": rewrite_mode,
        "full_rewrite_allowed": rewrite_mode == "comprehensive" and len(auto_fixable) >= 4,
    }

    if report.predictability:
        result["predictability"] = {
            "overall_risk": report.predictability.overall_risk,
            "risk_distribution": report.predictability.risk_distribution,
            "generic_phrases": report.predictability.generic_phrases_found,
            "sentences": [
                {"sentence_id": s.get("sentence_id", ""),
                 "text": s["sentence"][:100], "risk": s["risk_label"],
                 "score": s["risk"], "top10": s["top10_ratio"]}
                for s in report.predictability.sentences
            ],
            "score_derivation": {
                "step1_formula": "score = 0.45×top10_ratio + 0.25×top50_ratio + 0.20×(1/(1+surprisal)) + 0.10×generic_score",
                "step2_formula": "document_score = mean(sentence_score) for body sentences >= 8 words",
                "step3_postprocess": "document_score×0.6 + max_categorical×0.4 — weighted average of scanner probability and categorical severity",
                "raw_sentence_scores": [
                    round(s["risk"], 4) for s in report.predictability.sentences
                ],
                "raw_mean": round(
                    sum(s["risk"] for s in report.predictability.sentences)
                    / max(len(report.predictability.sentences), 1), 4
                ),
                "overall_risk": report.predictability.overall_risk,
                "included_sentence_count": len(report.predictability.sentences),
                "risk_thresholds": {
                    "high": "score >= 0.55 AND top10_ratio >= 0.70",
                    "medium": "score >= 0.45",
                    "review": "score >= 0.35",
                    "low": "score < 0.35",
                },
                "score_weights": {
                    "top_10_ratio": 0.45,
                    "top_50_ratio": 0.25,
                    "surprisal": 0.20,
                    "generic_phrases": 0.10,
                },
            },
        }
        # Full sentence map keyed by sentence_id for rewrite module
        result["sentence_map"] = {
            s.get("sentence_id", f"s{i+1:03d}"): {
                "paragraph_id": s.get("paragraph_id", ""),
                "start_char": s.get("start_char", 0),
                "end_char": s.get("end_char", 0),
                "text": s["sentence"],
            }
            for i, s in enumerate(report.predictability.sentences)
            if s.get("sentence_id")
        }

    if report.similarity:
        result["similarity"] = {
            "overall_risk": report.similarity.overall_risk,
            "risk_distribution": report.similarity.risk_distribution,
            "matches": report.similarity.matches,
        }

    if report.citation:
        result["citation"] = {
            "style": report.citation.citation_style,
            "in_text_count": report.citation.in_text_count,
            "bib_entry_count": report.citation.bib_entry_count,
            "findings": report.citation.findings,
            "stats": report.citation.stats,
        }

    if report.rewrite:
        result["rewrite"] = {
            "original_risk": report.rewrite.original_risk,
            "final_risk": report.rewrite.final_risk,
            "original_top10": report.rewrite.original_top10,
            "final_top10": report.rewrite.final_top10,
            "improvement_risk": report.rewrite.improvement_risk,
            "improvement_top10": report.rewrite.improvement_top10,
            "passes": report.rewrite.passes_completed,
            "converged": report.rewrite.converged,
            "convergence_reason": report.rewrite.convergence_reason,
            "progression": report.rewrite.pass_progression,
        }

    result["scan_time_seconds"] = report.scan_time_seconds
    if report.generated_at:
        result["generated_at"] = report.generated_at

    return result
