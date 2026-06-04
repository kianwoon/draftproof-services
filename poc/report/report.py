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
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from detect.scoring import extract_signals, calculate_authorship_concern, estimate_citation_risk
from report.authorship_evidence import build_authorship_evidence, strengthen_anchor_sentences
from detect.authorship_windows import build_ai_footprint_profile, build_authorship_window_profile
from detect.document_structure import structured_sentence_segments
from detect.repair_units import build_repair_units_v2
from detect.rewrite_targets import build_problem_inventory, build_rewrite_target_profile
from detect.layer3_scoring import Layer3Scorer, build_layer3_input_from_text, estimate_external_detector_likelihood, estimate_external_detector_segment_fraction
from detect.external_grouped_scoring import estimate_external_grouped_score
from detect.transformation import (
    TRANSFORMATION_SIGNAL_METADATA,
    classify_transformation_from_scan,
    transformation_signal_metadata,
)
from detect.topk_calibration import calibrate_topk_risk
from detect.turnitin_like import turnitin_like_ai_profile
from report.contribution import contribution_pair_int

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


def _topk_calibration_fields_for_summary(
    pred_summary: Any,
    raw_topk_pattern: Any,
    criterion_scores: Dict[str, Any] | None,
) -> Dict[str, Any]:
    details = {}
    if criterion_scores and "topk_predictability" in criterion_scores:
        cs = criterion_scores["topk_predictability"]
        details = cs.get("details", {}) if isinstance(cs, dict) else getattr(cs, "details", {}) or {}
    eligible_sentence_count = 0
    if pred_summary is not None:
        sentences = getattr(pred_summary, "sentences", None)
        if isinstance(sentences, list):
            eligible_sentence_count = sum(1 for item in sentences if item)
    return calibrate_topk_risk(
        raw_topk_pattern,
        avg_top10_ratio=(details or {}).get("avg_top10_ratio"),
        eligible_sentence_count=eligible_sentence_count,
    )


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
class SemanticShapeSummary:
    model_name: str
    embedding_model_attached: bool
    sentence_count: int
    paragraph_count: int
    adjacent_similarity_mean: float
    adjacent_similarity_std: float
    paragraph_similarity_mean: float
    paragraph_similarity_std: float
    semantic_uniformity_risk: float
    discourse_regularity_risk: float
    semantic_drift_risk: float


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
    # Detect scan scores (same as shown in scan report)
    detect_ai_likelihood: float = 0.0      # AI Generation Likelihood from detect scan
    detect_writing_quality: float = 0.0    # Writing Quality Risk from detect scan


# ── Main report ─────────────────────────────────────────────────────

@dataclass
class DraftReport:
    """Structured report from a full DraftProof scan."""
    overall_tier: Tier
    finding_count: int
    findings_by_tier: Dict[str, List[Finding]]

    predictability: Optional[PredictabilitySummary] = None
    similarity: Optional[SimilaritySummary] = None
    semantic_shape: Optional[SemanticShapeSummary] = None
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
    axis_scores: Optional[Dict[str, str]] = None
    reason_codes: List[str] = field(default_factory=list)
    authorship_concern_score: float = 0.0
    authorship_concern_confidence: str = "low"
    authorship_concern_signals: Optional[Dict[str, Any]] = None
    ai_risk_badge: Optional[Dict[str, Any]] = None
    paragraph_explanations: Optional[Dict[str, Any]] = None
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
    detected_actionability = ""
    if f.metadata and isinstance(f.metadata, dict):
        detected_actionability = f.metadata.get("actionability", "")
    if detected_actionability:
        aliases = {
            "auto_rewrite_candidate": "auto_fixable",
            "optional_structure_review": "optional_structure_review",
            "citation_repair": "citation_repair",
            "review_only": "review_only",
            "no_action": "no_action",
            "manual_required": "manual_required",
            "auto_fixable": "auto_fixable",
        }
        if detected_actionability in aliases:
            mapped = aliases[detected_actionability]
            if mapped == "auto_fixable" and f.adjusted_risk.lower() != "medium":
                return "review_only"
            return mapped

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
    # High predictability is review guidance, not an automatic rewrite target.
    # The rewrite pipeline is intentionally limited to medium findings.
    if "high_predictability" in title or "high_topk_predictability" in title:
        return "review_only"
    # Document-level low_specificity: auto-fixable only if NOT already downgraded
    # If AI likelihood is low and domain grounding is strong, specificity is review-level
    if title == "low_specificity":
        return "manual_required"
    # Medium predictability: only auto-fix if co-located with document-level signals
    # (low_specificity, uncited_claim) that confirm AI origin. Medium predictability
    # alone is common in clear human writing — auto-rewriting it makes things worse.
    if "medium_predictability" in title:
        if all_findings:
            _DOC_SIGNALS = {"low_specificity", "uncited_claim"}
            doc_paired = any(
                af.title in _DOC_SIGNALS or
                (af.metadata or {}).get("signal_category", "") in _DOC_SIGNALS
                for af in all_findings
                if af is not f
            )
            if doc_paired:
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
    # Document-level AI summary/structure signals are not safe automatic
    # sentence rewrites. They require source context, concrete examples, or
    # structural revision guidance.
    if title in {
        "moderate_ai_generation_likelihood",
        "elevated_ai_generation_likelihood",
        "uniform_paragraph_structure",
        "low_burstiness",
        "source_grounding",
        "polished_but_ungrounded",
    }:
        return "manual_required"
    # Other high/medium AI-generation signals: auto-fixable
    if adj == "medium" and f.scanner == "ai_generation":
        return "auto_fixable"
    return "review_only"


class ReportBuilder:
    """Builds a DraftReport from detect + rewrite results."""

    def __init__(self):
        self._findings: List[Finding] = []
        self._pred_summary: Optional[PredictabilitySummary] = None
        self._sim_summary: Optional[SimilaritySummary] = None
        self._semantic_summary: Optional[SemanticShapeSummary] = None
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
        Uses duck-typing to handle dual-import path conflicts.
        """
        # Duck-type: accept any object with the expected DetectResult attributes
        if not hasattr(result, "scanner") or not hasattr(result, "findings"):
            raise TypeError(f"Expected DetectResult, got {type(result).__name__}")

        scanner = result.scanner

        # Route to the right summary builder
        if scanner == "predictability":
            self._build_predictability_summary(result)
        elif scanner == "similarity":
            self._build_similarity_summary(result)
        elif scanner == "semantic_shape":
            self._build_semantic_shape_summary(result)
        elif scanner == "citation":
            self._build_citation_summary(result)

        # Convert all findings to report Findings with tiers
        for f in result.findings:
            tier = _RISK_LEVEL_TO_TIER.get(f.risk_level, Tier.LOW)
            # Upgrade certain finding types
            if f.finding_type == "exact_copy":
                tier = Tier.CRITICAL
            # Match sentence_id for ALL findings, not just predictability
            sent_id = ""
            loc = getattr(f, 'location', None) or {}

            # Strategy 1: direct snippet match (works for predictability)
            if self._sentence_id_map:
                snippet = (f.evidence or "")[:60]
                sent_id = self._sentence_id_map.get(snippet, "")

            # Strategy 2: use start_char from location to find containing sentence
            if not sent_id and loc.get("start_char") is not None and self._pred_summary:
                char_pos = loc["start_char"]
                for sent in self._pred_summary.sentences:
                    s_start = sent.get("start_char", 0)
                    s_end = sent.get("end_char", 0)
                    if s_start <= char_pos < s_end:
                        sent_id = sent.get("sentence_id", "")
                        break

            # Strategy 3: sentence_index from location (detect pipeline sets this)
            if not sent_id and loc.get("sentence_index") is not None and self._pred_summary:
                idx = loc["sentence_index"]
                if 0 <= idx < len(self._pred_summary.sentences):
                    sent_id = self._pred_summary.sentences[idx].get("sentence_id", "")
            # Inject scanner-level metadata into finding metadata
            meta = dict(f.metadata) if f.metadata else {}
            actionability = getattr(f, "actionability", "")
            if actionability == "auto_rewrite_candidate":
                actionability = "auto_fixable"
            if actionability:
                meta["actionability"] = actionability
            evidence_strength = getattr(f, "evidence_strength", "")
            if evidence_strength:
                meta["evidence_strength"] = evidence_strength
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
                finding_id=meta.get("finding_id", ""),
                sentence_id=sent_id,
                signal_category=signal_cat,
            ))

        return self

    def add_detection_report(self, detection_report) -> "ReportBuilder":
        """Accept a DetectionReport from the detect module's DetectionRunner.

        Feeds all scanner results through add_detection(), and stores
        confidence, caveats, and summary metadata.
        """
        if not (hasattr(detection_report, "scanner_results")
                and hasattr(detection_report, "confidence")):
            raise TypeError(
                f"Expected DetectionReport-like object, got {type(detection_report).__name__}"
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

        # Thread criterion scores for Phase 2 authorship concern signals
        if hasattr(detection_report, "criterion_scores") and detection_report.criterion_scores:
            self._summaries["criterion_scores"] = detection_report.criterion_scores

        return self

    @staticmethod
    def _dict_risk_to_float(s: dict) -> float:
        """Extract numeric risk from a predictability sentence dict.

        Report JSON uses 'risk' for the label ('review', 'medium') and 'score'
        for the numeric value. Scanner objects use 'predictability_risk' (float).
        """
        # Prefer numeric predictability_risk
        pr = s.get("predictability_risk")
        if isinstance(pr, (int, float)):
            return float(pr)
        # Fall back to score (numeric)
        sc = s.get("score")
        if isinstance(sc, (int, float)):
            return float(sc)
        # risk might be numeric in some formats
        r = s.get("risk")
        if isinstance(r, (int, float)):
            return float(r)
        return 0.0

    def _build_predictability_summary(self, result: "DetectResult"):
        raw = result.raw
        if raw is None:
            return

        def _token_signal_fields(token_results) -> Dict[str, Any]:
            tokens = []
            for tr in token_results or []:
                if isinstance(tr, dict):
                    token = str(tr.get("token", ""))
                    rank = tr.get("rank")
                    prob = tr.get("probability")
                    surprisal = tr.get("surprisal")
                    top10 = bool(tr.get("top_10", tr.get("top10", False)))
                    top50 = bool(tr.get("top_50", tr.get("top50", False)))
                else:
                    token = str(getattr(tr, "token", ""))
                    rank = getattr(tr, "rank", None)
                    prob = getattr(tr, "probability", None)
                    surprisal = getattr(tr, "surprisal", None)
                    top10 = bool(getattr(tr, "top_10", False))
                    top50 = bool(getattr(tr, "top_50", False))
                token_raw = token
                token_clean = token.strip()
                if not token_clean:
                    continue
                item = {
                    "token": token_clean,
                    "raw_token": token_raw,
                    "rank": int(rank) if isinstance(rank, int) else rank,
                    "probability": round(float(prob), 6) if isinstance(prob, (int, float)) else prob,
                    "surprisal": round(float(surprisal), 4) if isinstance(surprisal, (int, float)) else surprisal,
                    "top10": top10,
                    "top50": top50,
                }
                tokens.append(item)

            ranked = sorted(
                [
                    t for t in tokens
                    if t.get("top10") and re.search(r"[A-Za-z0-9]", str(t.get("token", "")))
                ],
                key=lambda t: (
                    t.get("rank") if isinstance(t.get("rank"), int) else 999999,
                    -(t.get("probability") or 0),
                ),
            )
            spans = []
            current = []
            for t in tokens:
                if t.get("top10"):
                    current.append(str(t.get("raw_token") or t.get("token") or ""))
                elif current:
                    if len(current) >= 2:
                        span = "".join(current)
                        span = re.sub(r"\s+", " ", span).strip()
                        if span:
                            spans.append(span)
                    current = []
            if current and len(current) >= 2:
                span = "".join(current)
                span = re.sub(r"\s+", " ", span).strip()
                if span:
                    spans.append(span)

            return {
                "top_predicted_tokens": ranked[:10],
                "predictable_token_spans": [s for s in spans if s][:6],
            }

        sentences = []
        generic_phrases = []
        sent_idx = 0
        for s in raw.get("sentences", []):
            if getattr(s, "error", None):
                continue
            if isinstance(s, dict):
                # Carried forward from report JSON — use dict access
                if s.get("error"):
                    continue
                sent_idx += 1
                sent_id = s.get("sentence_id", f"s{sent_idx:03d}")
                token_fields = _token_signal_fields(s.get("token_results", []))
                sentences.append({
                    "sentence_id": sent_id,
                    "sentence": s.get("sentence") or s.get("text", ""),
                    "risk_label": s.get("risk_label", "") or (s.get("risk", "") if isinstance(s.get("risk"), str) else ""),
                    "risk": self._dict_risk_to_float(s),
                    "avg_probability": s.get("avg_probability", 0),
                    "avg_surprisal": s.get("avg_surprisal", 0),
                    "top10_ratio": s.get("top_10_ratio") or s.get("top10", 0),
                    "top50_ratio": s.get("top_50_ratio") or s.get("top50", 0),
                    "start_char": s.get("start_char", 0),
                    "end_char": s.get("end_char", 0),
                    "paragraph_id": s.get("paragraph_id", ""),
                    **token_fields,
                })
            else:
                # Scanner object — use attribute access
                sent_idx += 1
                sent_id = f"s{sent_idx:03d}"
                token_fields = _token_signal_fields(getattr(s, "token_results", []))
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
                    **token_fields,
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

    def _build_semantic_shape_summary(self, result: "DetectResult"):
        raw = result.raw
        if raw is None:
            return

        def get(name: str, default=0.0):
            if isinstance(raw, dict):
                return raw.get(name, default)
            return getattr(raw, name, default)

        self._semantic_summary = SemanticShapeSummary(
            model_name=get("model_name", result.model_name or "unknown"),
            embedding_model_attached=bool(get("embedding_model_attached", False)),
            sentence_count=int(get("sentence_count", 0) or 0),
            paragraph_count=int(get("paragraph_count", 0) or 0),
            adjacent_similarity_mean=float(get("adjacent_similarity_mean", 0.0) or 0.0),
            adjacent_similarity_std=float(get("adjacent_similarity_std", 0.0) or 0.0),
            paragraph_similarity_mean=float(get("paragraph_similarity_mean", 0.0) or 0.0),
            paragraph_similarity_std=float(get("paragraph_similarity_std", 0.0) or 0.0),
            semantic_uniformity_risk=float(get("semantic_uniformity_risk", 0.0) or 0.0),
            discourse_regularity_risk=float(get("discourse_regularity_risk", 0.0) or 0.0),
            semantic_drift_risk=float(get("semantic_drift_risk", 0.0) or 0.0),
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
                "orig_tier": o.get("label") or o.get("risk_label", "?"),
                "orig_risk": o.get("risk") or o.get("predictability_risk", 0),
                "orig_top10": o.get("top10_ratio") or o.get("top_10_ratio", 0),
                "orig_sentence": o.get("sentence", ""),
                "new_tier": f.get("label") or f.get("risk_label", "?"),
                "new_risk": f.get("risk") or f.get("predictability_risk", 0),
                "new_top10": f.get("top10_ratio") or f.get("top_10_ratio", 0),
                "new_sentence": f.get("sentence", ""),
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
                label = (d.get("label") or d.get("risk_label") or "").lower()
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
    def _evidence_weight(strength: str) -> float:
        return {"strong": 1.0, "moderate": 0.6, "weak": 0.2}.get(strength, 0.2)

    @staticmethod
    def _derive_weighted_tier(findings: List[Finding], raw_tier: Tier) -> Tier:
        """Weighted multi-signal tier derivation.

        Instead of blunt 'highest finding wins', considers:
        - Critical findings always escalate
        - Similarity/citation high findings escalate
        - low_specificity only escalates to HIGH if paired with high AI-generation
        - Predictability only escalates with corroborating signals
        - Evidence strength weights findings (strong > moderate > weak)
        - Review-only/weak findings don't drive tier escalation
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

        # Extract domain grounding strength
        domain_grounding_strong = False
        for f in findings:
            if f.metadata and isinstance(f.metadata, dict):
                dg = f.metadata.get("domain_grounding_index", 0)
                if dg >= 0.8:
                    domain_grounding_strong = True

        # Evidence-weighted effective counts
        ew = ReportBuilder._evidence_weight
        effective_high = sum(ew(getattr(f, 'evidence_strength', 'weak'))
                            for f in findings if f.tier == Tier.HIGH)
        effective_medium = sum(ew(getattr(f, 'evidence_strength', 'weak'))
                              for f in findings if f.tier == Tier.MEDIUM)

        # Count actionable (non-review-only) medium findings
        actionable_medium = sum(
            1 for f in findings
            if f.tier == Tier.MEDIUM
            and determine_actionability(f, findings) not in ("review_only", "no_action")
        )
        strong_evidence_medium = sum(
            1 for f in findings
            if f.tier == Tier.MEDIUM
            and getattr(f, 'evidence_strength', 'weak') in ('strong', 'moderate')
        )

        # Corroborating signals check
        has_corroborating = (
            "similarity" in high_categories
            or any(f.tier == Tier.HIGH and f.category == "citation" for f in findings)
            or manual_count > 0
            or ai_likelihood >= 0.40
            or any(f.title == "low_specificity" and f.tier == Tier.HIGH for f in findings)
        )

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

        # Predictability: only escalates with corroborating signals
        if pred_risk >= 0.60:
            return Tier.HIGH if has_corroborating else Tier.MEDIUM
        if pred_risk >= 0.45:
            return Tier.MEDIUM if has_corroborating else Tier.LOW

        # Manual-required issues
        if manual_count >= 2:
            return Tier.MEDIUM
        if manual_count == 1:
            return Tier.MEDIUM if effective_medium >= 1.8 else Tier.LOW

        # Medium findings: only escalate if actionable or strong-evidence
        effective_for_threshold = max(actionable_medium, strong_evidence_medium)
        if effective_for_threshold >= 5 and high_count == 0:
            return Tier.MEDIUM
        # Mostly weak/review-only — don't let quantity alone escalate
        if actionable_medium < 2 and strong_evidence_medium < 3 and medium_count >= 5:
            return Tier.LOW

        # Fall through to raw tier, but cap at MEDIUM if no critical/high signals
        if raw_tier in (Tier.CRITICAL, Tier.HIGH) and effective_high_count == 0:
            return Tier.MEDIUM
        if raw_tier == Tier.HIGH:
            return Tier.HIGH

        return raw_tier

    @staticmethod
    def _floor_tier_to_badge(adjusted_tier: Tier, badge_tier_value: str) -> Tier:
        """Overall tier must not read LOW when the AI-likelihood badge is AMBER+.

        ``_derive_weighted_tier`` re-derives the overall tier from per-finding tiers
        and a brittle ``pred_risk >= 0.45`` cliff; it never consults the badge's own
        aggregate AI-likelihood (which already folds in generic_assertion, topk, etc.).
        So a highly-generic, zero-grounding document can land LOW while its badge is
        AMBER (observed: test_content11 -> badge AMBER 42%, generic_assertion 90,
        overall LOW). A LOW headline under an amber+ badge contradicts the badge and
        under-warns the user, so floor it to MEDIUM. GREEN badges (genuinely low AI
        signal) may stay LOW. Only LOW is lifted; higher tiers are never altered.
        """
        if adjusted_tier == Tier.LOW and str(badge_tier_value or "").upper() in ("AMBER", "ORANGE", "RED"):
            return Tier.MEDIUM
        return adjusted_tier

    # Domain grounding built purely from generic domain-term counts, with ZERO concrete
    # anchors anywhere (no named entities, numbers, dates, or quotes), is unanchored
    # vocabulary -- not subject-matter grounding. Cap it so it can't inflate the
    # human-anchor discount (transformation.py) or the writing-quality grounding credit
    # (layer3). Real grounded writing carries at least one concrete anchor.
    _DOMAIN_GROUNDING_UNANCHORED_CAP = 0.30

    @staticmethod
    def _gate_domain_grounding(strength: float, concrete_anchor_count: int) -> float:
        if (concrete_anchor_count or 0) <= 0:
            return min(float(strength), ReportBuilder._DOMAIN_GROUNDING_UNANCHORED_CAP)
        return float(strength)

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
        # NOTE: pre-floor value. The badge floor (below, after the badge is computed)
        # may lift adjusted_tier LOW->MEDIUM and, when it does, REPLACES overall_tier_reason
        # entirely — so this tier_label string must not be appended to the reason after the floor.
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

        # ── Axis scores: per-signal status ──
        axis_scores = {}
        # Predictability axis
        pred_meta_risk = 0.0
        for f in self._findings:
            if f.metadata and isinstance(f.metadata, dict):
                pred_meta_risk = max(pred_meta_risk, f.metadata.get("predictability_risk", 0.0))
        if pred_meta_risk >= 0.60:
            axis_scores["predictability"] = "attention"
        elif pred_meta_risk >= 0.40:
            axis_scores["predictability"] = "review"
        else:
            axis_scores["predictability"] = "clear"
        # Similarity axis
        has_sim_issue = any(f.category == "similarity" and f.tier in (Tier.HIGH, Tier.MEDIUM)
                           for f in self._findings)
        axis_scores["similarity"] = "attention" if has_sim_issue else "clear"
        # Citation axis
        cite_count = sum(1 for f in self._findings if f.category == "citation"
                        and f.tier.value not in ("low", "clean"))
        if cite_count >= 2:
            axis_scores["citation"] = "attention"
        elif cite_count == 1:
            axis_scores["citation"] = "review"
        else:
            axis_scores["citation"] = "clear"
        # Specificity axis
        spec_risk = 0.0
        for f in self._findings:
            if f.title == "low_specificity" and f.metadata:
                spec_risk = f.metadata.get("specificity_risk", 0.0)
        if spec_risk >= 0.50:
            axis_scores["specificity"] = "attention"
        elif spec_risk >= 0.30:
            axis_scores["specificity"] = "review"
        else:
            axis_scores["specificity"] = "clear"
        # Domain grounding axis
        dg_level = "weak"
        concrete_anchor_count = 0
        for f in self._findings:
            if f.title == "low_specificity" and f.metadata:
                dg_level = f.metadata.get("domain_grounding_level", "weak")
                # Concrete anchors: real subject-matter grounding co-occurs with at least
                # one of these; their total absence means "domain terms" are generic vocab.
                concrete_anchor_count = (
                    int(f.metadata.get("named_entities", 0) or 0)
                    + int(f.metadata.get("numbers", 0) or 0)
                    + int(f.metadata.get("dates", 0) or 0)
                    + int(f.metadata.get("quotes", 0) or 0)
                )
        axis_scores["domain_grounding"] = dg_level

        # ── Authorship concern score ──
        concern_signals = extract_signals(
            predictability_summary=self._pred_summary,
            similarity_summary=self._sim_summary,
            citation_summary=self._cite_summary,
            findings=self._findings,
            criterion_scores=self._summaries.get("criterion_scores"),
        )
        concern = calculate_authorship_concern(
            signals=concern_signals,
            word_count=len(self._original_text.split()) if self._original_text else 0,
            has_sources=bool(self._sim_summary and self._sim_summary.matches),
            has_bibliography=bool(self._cite_summary and self._cite_summary.bib_entry_count > 0),
            has_draft_history=False,
        )

        # ── AI Risk Badge (Layer 3 Scoring) ──
        sig = concern["signals"] or {}
        ai_lik = 0.0
        dg_idx = 0.0
        for f in self._findings:
            if f.metadata and isinstance(f.metadata, dict):
                ai_lik = max(ai_lik, f.metadata.get("ai_likelihood", 0.0))
                dg_idx = max(dg_idx, f.metadata.get("domain_grounding_index", 0.0))

        domain_grounding_strength = max(0.0, min(float(dg_idx or 0.0), 1.0))
        if dg_level == "strong":
            domain_grounding_strength = max(domain_grounding_strength, 0.75)
        elif dg_level == "moderate":
            domain_grounding_strength = max(domain_grounding_strength, 0.55)
        elif domain_grounding_strength > 0:
            domain_grounding_strength = max(domain_grounding_strength, 0.30)
        # Concrete-anchor gate: don't credit "domain grounding" assembled only from
        # generic vocabulary when the document has zero concrete anchors.
        domain_grounding_strength = self._gate_domain_grounding(
            domain_grounding_strength, concrete_anchor_count
        )

        n_high = sum(1 for f in self._findings if f.tier == Tier.HIGH)
        n_critical = sum(1 for f in self._findings if f.tier == Tier.CRITICAL)

        # Citation risk from cite summary
        cite_risk = 0.50
        if self._cite_summary:
            uncited = sum(1 for f in self._cite_summary.findings if "uncited" in f.get("type", "").lower())
            total_claims = max(self._cite_summary.in_text_count + uncited, 1)
            cite_risk = estimate_citation_risk(
                in_text_citations=self._cite_summary.in_text_count,
                bibliography_count=self._cite_summary.bib_entry_count,
                uncited_claims=uncited,
                total_claims=total_claims,
            ) if self._cite_summary.bib_entry_count > 0 else 0.50
        # sig["source_grounding"] is a CONCERN (0 = well grounded, 1 = fully unsupported -- see
        # detect/criteria/source_grounding.py:210), so the grounding STRENGTH is its complement.
        # Previously this concern was consumed directly as a strength, inverting the signal: a
        # well-grounded text (concern -> 0) reported source_grounding_strength 0 -> risk 100% (worst),
        # which is how a rewrite that REDUCED uncited claims showed Grounding risk jumping to 100%.
        # Only invert when the signal is present; a missing signal contributes no strength (it falls
        # through to the in-text citation estimator below), never a spurious "fully grounded".
        sg_concern = sig.get("source_grounding")
        sg_strength_signal = (1.0 - sg_concern) if isinstance(sg_concern, (int, float)) else 0.0
        source_grounding_strength = min(
            sg_strength_signal,
            0.30 if not (self._cite_summary and self._cite_summary.bib_entry_count > 0) else 1.0,
        )
        source_grounding_strength = max(
            source_grounding_strength,
            _estimate_in_text_source_grounding_strength(self._original_text or ""),
        )

        # Raw GPT-2 Top-k stays visible as a diagnostic signal, but the
        # authorship/transformation model must consume the calibrated Top-k
        # risk. Otherwise strict-safe can pass calibrated Top-k while
        # AI Authorship is still punished by the impossible raw scale.
        raw_topk_pattern = sig.get("topk_pattern_risk", 0.0) or 0.0
        topk_calibration = _topk_calibration_fields_for_summary(
            self._pred_summary,
            raw_topk_pattern,
            self._summaries.get("criterion_scores"),
        )
        calibrated_topk_for_authorship = (
            (topk_calibration.get("topk_calibrated_risk") or 0.0) / 100.0
        )

        # Build Layer3Input from scanner outputs + text-derived signals
        layer3_input = build_layer3_input_from_text(
            self._original_text or "",
            predictability=sig.get("predictability", 0.0) or 0.0,
            topk_pattern=calibrated_topk_for_authorship,
            generic_phrase_density=sig.get("genericity", 0.0) or 0.0,
            # broad_claim_risk & unsupported_claim_risk: auto-computed from text
            citation_weakness_risk=cite_risk,
            source_grounding_strength=source_grounding_strength,
            domain_grounding_strength=domain_grounding_strength,
            semantic_uniformity_risk=(
                self._semantic_summary.semantic_uniformity_risk
                if self._semantic_summary else None
            ),
            discourse_regularity_risk=(
                self._semantic_summary.discourse_regularity_risk
                if self._semantic_summary else None
            ),
            semantic_drift_risk=(
                self._semantic_summary.semantic_drift_risk
                if self._semantic_summary else None
            ),
            human_provenance_positive=False,
            verified_ai_provenance=False,
        )

        layer3 = Layer3Scorer().score(layer3_input)
        transformation = classify_transformation_from_scan(
            layer3_input,
            layer3,
            similarity_summary=self._sim_summary,
        )

        ai_components = {k: round(v * 100, 2) for k, v in layer3.ai_phase.components.items()}
        ai_components["topk_authorship_component"] = ai_components.get("topk_pattern")
        ai_components.update(topk_calibration)
        # Compatibility: topk_pattern remains the raw scanner score. The
        # calibrated risk is a separate safe-band gate.
        ai_components["topk_pattern"] = topk_calibration.get("topk_pattern_raw", raw_topk_pattern)

        writing_components = {k: round(v * 100, 2) for k, v in layer3.writing_phase.components.items()}

        # External-facing detector proxy. This is additive only: it does NOT affect the tier,
        # ai_likelihood_score, or any rewrite gate. Legacy estimates remain attached for auditability.
        _pred_sentences = self._pred_summary.sentences if self._pred_summary else None
        legacy_segment_fraction = estimate_external_detector_segment_fraction(_pred_sentences)
        legacy_likelihood = estimate_external_detector_likelihood(ai_components)
        external_detector_estimate = estimate_external_grouped_score(
            sentences=_pred_sentences,
            ai_components=ai_components,
            writing_components=writing_components,
            transformation_features=transformation.features,
            criterion_scores=self._summaries.get("criterion_scores"),
            legacy_segment_fraction=legacy_segment_fraction,
            legacy_likelihood=legacy_likelihood,
        )

        ai_risk_badge = {
            # AI Generation (Phase 1)
            "tier": layer3.tier.value,
            "ai_likelihood_score": round(layer3.ai_likelihood_score * 100, 2),
            "external_detector_estimate": external_detector_estimate,
            "authorship_rating": layer3.authorship_rating,
            "authorship_rating_label": layer3.authorship_rating.get("label"),
            "authorship_rating_code": layer3.authorship_rating.get("code"),
            "ai_cluster_boost": round(layer3.ai_cluster_boost * 100, 2) if layer3.ai_cluster_boost else 0,
            "ai_cluster_name": layer3.ai_cluster_name,
            "ai_components": ai_components,

            # Writing Quality (Phase 2)
            "writing_quality_tier": layer3.writing_quality_tier.value,
            "writing_quality_score": round(layer3.writing_quality_score * 100, 2),
            "writing_components": writing_components,

            # Combined
            "review_priority": layer3.review_priority,
            "confidence": layer3.confidence.value,
            "reasons": layer3.reasons,
            "guardrails": layer3.guardrails,
            "red_flags": n_high + n_critical,
            "transformation_classification": {
                "code": transformation.code,
                "label": transformation.label,
                "confidence": transformation.confidence,
                "evidence": transformation.evidence,
                "features": transformation.features,
                "is_verdict": transformation.is_verdict,
            },
        }

        badge_ai_score = ai_risk_badge.get("ai_likelihood_score", 0.0) / 100
        if ai_val > 0 and badge_ai_score > 0:
            overall_tier_reason = overall_tier_reason.replace(
                f"AI likelihood is {ai_val:.1%}",
                f"AI likelihood is {badge_ai_score:.1%}",
            )

        # Badge floor: the headline tier must not read LOW when the AI-likelihood badge
        # is AMBER+ (the badge aggregates generic_assertion/topk/etc.; the weighted
        # derivation does not). Lifts only LOW -> MEDIUM; never lowers a higher tier.
        pre_floor_tier = adjusted_tier
        adjusted_tier = self._floor_tier_to_badge(adjusted_tier, ai_risk_badge.get("tier"))
        if adjusted_tier != pre_floor_tier:
            # Self-contained reason (do NOT append the stale pre-floor text, which still
            # asserts the LOW tier). Signal detail remains available in ai_components/findings.
            overall_tier_reason = (
                f"Overall tier is {adjusted_tier.value.upper()} (raised from "
                f"{pre_floor_tier.value.upper()}): the AI-likelihood badge is "
                f"{str(ai_risk_badge.get('tier') or '').upper()} at {badge_ai_score:.1%}. A document "
                f"with amber-or-higher AI-likelihood is not rated below {adjusted_tier.value.upper()}; "
                f"see the AI signal breakdown for the contributing signals."
            )

        # ── Reason codes: structured tier explanation ──
        reason_codes = []
        rewrite_decision = self._summaries.get("rewrite_decision") or {}
        rewrite_is_recommended = bool(rewrite_decision.get("run_rewrite"))
        if not has_high_critical:
            reason_codes.append("no_high_or_critical_findings")
        if badge_ai_score < 0.25:
            reason_codes.append("low_ai_pattern_score")
        if axis_scores.get("domain_grounding") == "strong":
            reason_codes.append("strong_domain_grounding")
        if len(review_only) > len(auto_fixable) + len(manual_required):
            reason_codes.append("mostly_review_only_findings")
        if pred_meta_risk < 0.40:
            reason_codes.append("predictability_unconfirmed")
        if not rewrite_is_recommended:
            reason_codes.append("rewrite_not_recommended")

        # Inject detect scan scores into the rewrite summary so the rewrite report shows
        # the same risk scores the user sees in the scan report. Must run BEFORE the
        # return; this block was previously dead code placed after it.
        if self._rewrite_summary:
            self._rewrite_summary.detect_ai_likelihood = ai_risk_badge.get("ai_likelihood_score", 0.0)
            self._rewrite_summary.detect_writing_quality = ai_risk_badge.get("writing_quality_score", 0.0)

        return DraftReport(
            overall_tier=adjusted_tier,
            finding_count=len(self._findings),
            findings_by_tier=findings_by_tier,
            predictability=self._pred_summary,
            similarity=self._sim_summary,
            semantic_shape=self._semantic_summary,
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
            axis_scores=axis_scores,
            reason_codes=reason_codes,
            authorship_concern_score=concern["score"],
            authorship_concern_confidence=concern["confidence_label"],
            authorship_concern_signals=concern["signals"],
            ai_risk_badge=ai_risk_badge,
        )


# ── Report to dict ──────────────────────────────────────────────────


def _concern_tier_from_score(score: float) -> str:
    if score >= 0.65:
        return "urgent_review"
    if score >= 0.40:
        return "needs_attention"
    if score >= 0.25:
        return "review_recommended"
    if score >= 0.15:
        return "light_review"
    return "clear"


def _is_weak_only(signals: Optional[Dict[str, Any]]) -> bool:
    if not signals:
        return False
    strong = {"source_grounding", "citation_integrity", "draft_evolution", "structural_reuse"}
    has_weak = any(signals.get(k) is not None and signals.get(k, 0) > 0
                   for k in ("predictability", "genericity", "specificity"))
    has_strong = any(signals.get(k) is not None and signals.get(k, 0) >= 0.25 for k in strong)
    return has_weak and not has_strong


def _estimate_in_text_source_grounding_strength(text: str) -> float:
    """Bounded source strength from in-text source relationships without a bibliography object."""
    text = text or ""
    url_count = len(re.findall(r"\b(?:https?://|www\.|doi\.org/)\S+", text, flags=re.I))
    has_reference_section = bool(re.search(
        r"(?im)^\s*(?:references|reference list|bibliography|works cited|sources)\s*$",
        text,
    ))
    reference_tail = ""
    if has_reference_section:
        parts = re.split(
            r"(?im)^\s*(?:references|reference list|bibliography|works cited|sources)\s*$",
            text,
            maxsplit=1,
        )
        reference_tail = parts[-1] if len(parts) > 1 else ""
    reference_years = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", reference_tail))
    parenthetical = len(re.findall(
        r"\((?:[A-Z][A-Za-z'’.-]+(?:\s+(?:&|and)\s+[A-Z][A-Za-z'’.-]+)?|[A-Z][A-Za-z'’.-]+\s+et\s+al\.?|[A-Z]{2,})\s*,?\s*(?:19|20)\d{2}[a-z]?\)",
        text,
    ))
    narrative = len(re.findall(
        r"\b[A-Z][A-Za-z'’.-]+(?:\s+(?:and|&|et\s+al\.?)\s+[A-Z][A-Za-z'’.-]+)*\s*\((?:19|20)\d{2}[a-z]?\)",
        text,
    ))
    source_relations = len(re.findall(
        r"\b(?:states|argues|explains|shows|describes|defines|discusses|notes?|offers|focus(?:es)? on|highlight(?:s)?|according to)\b",
        text,
        flags=re.I,
    ))
    citation_count = parenthetical + narrative
    reference_signal = max(url_count, reference_years if has_reference_section else 0)
    if citation_count >= 6 and source_relations >= 4:
        return 0.70
    if citation_count >= 4 and source_relations >= 2:
        return 0.60
    if reference_signal >= 3 and source_relations >= 2:
        return 0.55
    if has_reference_section and reference_signal >= 2:
        return 0.50
    if citation_count >= 2 and source_relations >= 1:
        return 0.45
    if reference_signal >= 1 and source_relations >= 1:
        return 0.40
    if citation_count >= 1:
        return 0.35
    if has_reference_section and reference_signal >= 1:
        return 0.30
    return 0.0


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
            domain_terms = metrics.get("domain_terms", [])
            matched_domain_term_count = len(domain_terms) if isinstance(domain_terms, list) else 0
            weighted_domain_term_count = int(metrics.get("domain_term_count", 0))
            result_evidence = {
                "type": "document_level",
                "summary": (f"{int(metrics.get('word_count', 0))} words, "
                            f"{int(metrics.get('named_entities', 0))} named entities, "
                            f"{int(metrics.get('numbers', 0))} numbers, "
                            f"{int(metrics.get('dates', 0))} dates, "
                            f"{matched_domain_term_count} matched domain terms "
                            f"({weighted_domain_term_count} weighted)"),
                "affected_span": "full_document",
                "metrics": metrics,
                "matched_domain_term_count": matched_domain_term_count,
                "weighted_domain_term_count": weighted_domain_term_count,
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

    pred_sentences = report.predictability.sentences if report.predictability else []
    pred_by_id = {
        s.get("sentence_id"): s
        for s in pred_sentences
        if s.get("sentence_id")
    }
    pred_index_by_id = {
        s.get("sentence_id"): i
        for i, s in enumerate(pred_sentences)
        if s.get("sentence_id")
    }
    paragraph_by_id = {}
    for s in pred_sentences:
        pid = s.get("paragraph_id")
        sid = s.get("sentence_id")
        if pid and sid:
            paragraph_by_id.setdefault(pid, []).append(s)

    def _content_terms(text: str, limit: int = 12) -> list:
        stopwords = {
            "about", "after", "again", "also", "being", "because", "before",
            "between", "could", "does", "every", "from", "have", "into",
            "many", "more", "most", "need", "needs", "only", "same",
            "should", "simply", "some", "than", "that", "their", "them",
            "there", "these", "they", "this", "those", "time", "when",
            "where", "while", "with", "without", "would",
            "another", "complex", "especially", "explain", "gives", "issue",
            "layer", "limits", "memory", "multiple", "problem", "process",
            "working", "information", "elements", "handled", "must",
        }
        role_terms = {"learner", "learners", "student", "students", "teacher", "teachers", "educator", "educators"}
        text = _re.sub(r"\([^)]*\d{4}[^)]*\)", " ", text or "")
        terms = []
        for word in _re.findall(r"\b[A-Za-z][A-Za-z-]{3,}\b", text):
            lower = word.lower()
            if word.isupper() or (word[0].isupper() and lower not in role_terms):
                continue
            word = lower if lower in role_terms else word
            if lower not in stopwords and lower not in {t.lower() for t in terms}:
                terms.append(word)
            if len(terms) >= limit:
                break
        return terms

    def _rewrite_signal_instruction(f: Finding, anchors: list) -> str:
        anchor_text = ", ".join(anchors[:8]) if anchors else "nearby paragraph terms"
        if f.title in ("medium_predictability", "high_predictability"):
            return (
                "Break the common-word path. Rebuild the sentence around a concrete "
                f"condition, observation, or action supported nearby. Use anchors if natural: {anchor_text}."
            )
        if f.title in ("high_topk_predictability", "low_surprisal", "low_surprisal_pattern"):
            return (
                "Change the sentence opening and token path. Start from a concrete "
                f"domain object, action, or constraint using nearby anchors: {anchor_text}."
            )
        if f.title == "low_specificity":
            return (
                "Do not auto-rewrite this as a sentence patch. Add only supported concrete detail "
                "from existing domain terms, source material, or the author's stated context."
            )
        return "Use the finding signal and nearby context to avoid generic paraphrase."

    def _rewrite_context_for_finding(f: Finding) -> Optional[Dict[str, Any]]:
        sid = f.sentence_id
        sent = pred_by_id.get(sid)
        if not sent:
            if f.title == "low_specificity":
                domain_terms = []
                if f.metadata and isinstance(f.metadata, dict):
                    domain_terms = f.metadata.get("domain_terms", []) or []
                return {
                    "scope": "document",
                    "signal_instruction": _rewrite_signal_instruction(f, domain_terms[:8]),
                    "domain_anchors": domain_terms[:12],
                    "safe_addition_types": [
                        "source-backed detail",
                        "existing domain term",
                        "author observation already present in the draft",
                    ],
                }
            return None

        idx = pred_index_by_id.get(sid, -1)
        previous_sentence = pred_sentences[idx - 1]["sentence"] if idx > 0 else ""
        next_sentence = pred_sentences[idx + 1]["sentence"] if 0 <= idx < len(pred_sentences) - 1 else ""
        pid = sent.get("paragraph_id")
        paragraph_items = paragraph_by_id.get(pid, []) if pid else []
        paragraph_text = " ".join(item.get("sentence", "") for item in paragraph_items)
        paragraph_idx = next(
            (i for i, item in enumerate(paragraph_items) if item.get("sentence_id") == sid),
            -1,
        )
        if paragraph_idx >= 0:
            focus_items = paragraph_items[max(0, paragraph_idx - 4): paragraph_idx + 2]
            focus_text = " ".join(item.get("sentence", "") for item in focus_items)
        else:
            focus_text = " ".join(x for x in (previous_sentence, sent.get("sentence", ""), next_sentence) if x)
        anchors = _content_terms(focus_text) or _content_terms(paragraph_text)

        return {
            "scope": "sentence",
            "sentence_id": sid,
            "paragraph_id": pid,
            "previous_sentence": previous_sentence,
            "next_sentence": next_sentence,
            "paragraph_excerpt": paragraph_text[:700],
            "domain_anchors": anchors,
            "problem_tokens": sent.get("top_predicted_tokens", []),
            "predictable_token_spans": sent.get("predictable_token_spans", []),
            "signal_instruction": _rewrite_signal_instruction(f, anchors),
            "predictability_metrics": {
                "score": sent.get("risk"),
                "risk_label": sent.get("risk_label"),
                "top10_ratio": sent.get("top10_ratio"),
                "top50_ratio": sent.get("top50_ratio"),
                "avg_surprisal": sent.get("avg_surprisal"),
            },
        }

    def _sentence_index_from_id(sentence_id: str) -> Optional[int]:
        if not sentence_id:
            return None
        m = _re.match(r"s0*(\d+)$", sentence_id)
        if not m:
            return None
        return max(0, int(m.group(1)) - 1)

    def _paragraph_role(sentence_id: str, paragraph_items: list) -> str:
        if not sentence_id or not paragraph_items:
            return "unknown"
        paragraph_idx = next(
            (i for i, item in enumerate(paragraph_items) if item.get("sentence_id") == sentence_id),
            -1,
        )
        if paragraph_idx < 0:
            return "unknown"
        sentence_text = paragraph_items[paragraph_idx].get("sentence", "").strip().lower()
        if paragraph_idx == 0 and sentence_id in {"s001", "s002"}:
            return "intro"
        if any(marker in sentence_text for marker in ("according to", "(", "et al.", "explains", "argues", "suggests")):
            return "evidence"
        if any(marker in sentence_text for marker in ("i ", "my ", "in my context", "i see", "i usually", "from my")):
            return "reflection"
        if any(marker in sentence_text for marker in ("however", "because of this", "another issue", "at the same time")):
            return "transition"
        if sentence_text.startswith(("in conclusion", "overall", "this review has argued")):
            return "conclusion"
        if paragraph_idx == len(paragraph_items) - 1 and len(pred_sentences) >= 4:
            return "conclusion" if sentence_id == pred_sentences[-1].get("sentence_id") else "reflection"
        return "unknown"

    def _protected_spans_for_sentence(sentence: str) -> list:
        spans = []

        def add(kind: str, pattern: str):
            for m in _re.finditer(pattern, sentence or ""):
                text = m.group(0).strip()
                if text:
                    spans.append({
                        "text": text,
                        "type": kind,
                        "start": m.start(),
                        "end": m.end(),
                    })

        add("citation", r"\([A-Z][A-Za-z .,&-]+,\s*(?:n\.d\.|\d{4})[^)]*\)")
        add("quote", r'"[^"]+"|“[^”]+”')
        add("url", r"https?://\S+")
        add("number", r"\b\d+(?:\.\d+)?%?\b")
        add("unit_code", r"\b[A-Z]{3,}[A-Z0-9]{2,}\b")
        add("institution", r"\b(?:Box Hill Institute|Certificate III|Australian Government|Department of Employment and Workplace Relations)\b")

        unique = []
        seen = set()
        for span in spans:
            key = (span["text"], span["type"], span["start"])
            if key not in seen:
                seen.add(key)
                unique.append(span)
        return unique

    def _rewrite_permission(f: Finding, bucket: str) -> str:
        if bucket in {"citation_repair", "manual_required"}:
            return "manual"
        if bucket == "optional_structure_review":
            return "suggestion_only"
        if bucket in {"review_only", "no_action"}:
            return "suggestion_only"
        if f.title in {"low_specificity", "close_paraphrase", "patchwriting", "semantic_overlap", "paragraph_level_overlap", "similarity_overlap"}:
            return "manual"
        if f.category in {"citation", "similarity", "integrity"}:
            return "manual"
        return "auto" if bucket == "auto_fixable" else "suggestion_only"

    def _rewrite_edit_brief_for_finding(f: Finding) -> Optional[Dict[str, Any]]:
        sid = f.sentence_id
        sent = pred_by_id.get(sid)
        if not sent:
            return None

        idx = pred_index_by_id.get(sid, -1)
        pid = sent.get("paragraph_id", "")
        paragraph_items = paragraph_by_id.get(pid, []) if pid else []
        paragraph_text = " ".join(item.get("sentence", "") for item in paragraph_items)
        target_sentence = sent.get("sentence", "")
        anchors = _content_terms(
            " ".join(x for x in (
                pred_sentences[idx - 1]["sentence"] if idx > 0 else "",
                target_sentence,
                pred_sentences[idx + 1]["sentence"] if 0 <= idx < len(pred_sentences) - 1 else "",
                paragraph_text,
            ) if x)
        )
        bucket = _determine_actionability(f, all_findings)
        signals = {
            "finding_type": f.title,
            "risk": sent.get("risk_label"),
            "score": sent.get("risk"),
            "top10_ratio": sent.get("top10_ratio"),
            "top50_ratio": sent.get("top50_ratio"),
            "avg_surprisal": sent.get("avg_surprisal"),
            "problem_tokens": sent.get("top_predicted_tokens", []),
            "predictable_token_spans": sent.get("predictable_token_spans", []),
            "signal_category": f.signal_category or (f.metadata or {}).get("signal_category"),
        }
        return {
            "finding_id": f.finding_id,
            "sentence_id": sid,
            "paragraph_id": pid,
            "sentence_index": _sentence_index_from_id(sid),
            "target_sentence": target_sentence,
            "previous_sentence": pred_sentences[idx - 1]["sentence"] if idx > 0 else "",
            "next_sentence": pred_sentences[idx + 1]["sentence"] if 0 <= idx < len(pred_sentences) - 1 else "",
            "paragraph_excerpt": paragraph_text[:900],
            "paragraph_role": _paragraph_role(sid, paragraph_items),
            "signals": signals,
            "domain_anchors": anchors,
            "protected_spans": _protected_spans_for_sentence(target_sentence),
            "rewrite_permission": _rewrite_permission(f, bucket),
            "instruction": _rewrite_signal_instruction(f, anchors),
        }

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
                "sentence_index": _sentence_index_from_id(f.sentence_id),
                "evidence": _structured_evidence(f),
                "recommendation": f.recommendation,
                "rewrite_context": _rewrite_context_for_finding(f),
                "adjustment": f.metadata.get("adjustment") if f.metadata else None,
                "detail": f.detail,
            }
            for f in report.findings_by_tier.get(tier.value, [])
        ]

    def _rewrite_edit_briefs() -> list:
        briefs = []
        seen = set()
        for f in all_findings:
            brief = _rewrite_edit_brief_for_finding(f)
            if not brief:
                continue
            key = brief.get("finding_id") or (brief.get("sentence_id"), brief.get("signals", {}).get("finding_type"))
            if key in seen:
                continue
            seen.add(key)
            briefs.append(brief)
        return briefs

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

    detect_rewrite_decision = report.rewrite_decision or {}
    if detect_rewrite_decision:
        decision_mode = detect_rewrite_decision.get("mode")
        if decision_mode in ("targeted", "full", "none"):
            rewrite_mode = decision_mode
        if not detect_rewrite_decision.get("run_rewrite", False):
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
                "domain_term_count": len(f.metadata.get("domain_terms", []) or []),
                "weighted_domain_term_count": f.metadata.get("domain_term_count", 0),
                "matched_domain_terms": f.metadata.get("domain_terms", []),
                "auto_detected": True,
            }
            break

    local_actionability_distribution = {}
    for f in all_findings:
        bucket = _determine_actionability(f, all_findings)
        local_actionability_distribution[bucket] = local_actionability_distribution.get(bucket, 0) + 1

    serialized_rewrite_decision = dict(report.rewrite_decision or {})
    if serialized_rewrite_decision:
        serialized_rewrite_decision["allowed_actions"] = [
            "auto_fixable" if a == "auto_rewrite_candidate" else a
            for a in serialized_rewrite_decision.get("allowed_actions", [])
        ]
        if serialized_rewrite_decision.get("run_rewrite"):
            serialized_rewrite_decision["targets"] = [
                f["finding_id"] for f in auto_fixable
            ]
            serialized_rewrite_decision["run_rewrite"] = bool(auto_fixable)
            if not auto_fixable:
                serialized_rewrite_decision["mode"] = "none"
                serialized_rewrite_decision["allowed_actions"] = []
            serialized_rewrite_decision["reason"] = (
                f"{len(auto_fixable)} auto-fixable finding(s) detected."
                if auto_fixable
                else "No medium auto-fixable findings. Signals are review-only."
            )

    def _clamp01(value: Any) -> float:
        try:
            numeric = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if numeric > 1.0:
            numeric = numeric / 100.0
        return max(0.0, min(1.0, numeric))

    def _pct(value: Any) -> int:
        return int(round(_clamp01(value) * 100))

    def _transformation_signal_rows(features: Dict[str, Any]) -> list:
        rows = []
        for key in TRANSFORMATION_SIGNAL_METADATA:
            if key in (features or {}):
                meta = transformation_signal_metadata(key)
                rows.append({
                    "key": key,
                    "label": meta["label"],
                    "description": meta["description"],
                    "family": meta["family"],
                    "higher_score_means": meta["higher_score_means"],
                    "score": _pct(features.get(key)),
                    "raw_score": round(_clamp01(features.get(key)), 4),
                })
        rows.sort(key=lambda item: item["score"], reverse=True)
        return rows

    def _transformation_contribution(
        features: Dict[str, Any],
        signals: list,
        ai_components: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        calibrated_ai = (
            _clamp01(features.get("calibrated_ai_risk"))
            if features.get("calibrated_ai_risk") is not None
            else (
                _clamp01(features.get("adjusted_ai_risk"))
                if features.get("adjusted_ai_risk") is not None
                else _clamp01(features.get("ai_likelihood"))
            )
        )
        turnitin_profile = turnitin_like_ai_profile(
            features=features,
            ai_components=ai_components or {},
        )
        atr = int(round(float(turnitin_profile.get("score") or 0.0)))
        atr = max(0, min(100, atr))
        hcr = 100 - atr

        top_drivers = [row["label"].lower() for row in signals[:2] if row.get("score", 0) > 0]
        if atr >= 70:
            summary = "AI transformation signals dominate the scan profile."
        elif hcr >= 70:
            summary = "Human anchoring dominates, with limited AI transformation signal."
        else:
            summary = "Mixed authorship pattern: human anchoring and AI transformation signals are both visible."
        if top_drivers:
            summary += " Main drivers: " + " and ".join(top_drivers) + "."

        return {
            "human_contribution_ratio": hcr,
            "ai_transformation_ratio": atr,
            "adjusted_ai_risk": _pct(features.get("adjusted_ai_risk")),
            "calibrated_ai_risk": _pct(calibrated_ai),
            "human_anchor_discount": _pct(features.get("human_anchor_discount")),
            "calibration_confidence": _pct(features.get("calibration_confidence")),
            "reporting_suppression": _pct(features.get("reporting_suppression")),
            "turnitin_like_ai_score": round(float(turnitin_profile.get("score") or 0.0), 3),
            "turnitin_like_target_score": turnitin_profile.get("target_score"),
            "turnitin_like_target_gap": turnitin_profile.get("target_gap"),
            "turnitin_like_target_met": turnitin_profile.get("target_met"),
            "turnitin_like_components": turnitin_profile.get("components") or {},
            "turnitin_like_weighted_components": turnitin_profile.get("weighted_components") or {},
            "turnitin_like_component_contributions": turnitin_profile.get("component_contributions") or {},
            "turnitin_like_top_positive_drivers": turnitin_profile.get("top_positive_drivers") or [],
            "turnitin_like_human_anchor_suppression": turnitin_profile.get("human_anchor_suppression"),
            "turnitin_like_score_version": turnitin_profile.get("version"),
            "summary": summary,
        }

    def _risk_label(score: int, *, high: int = 65, medium: int = 40) -> str:
        if score >= high:
            return "high"
        if score >= medium:
            return "medium"
        return "low"

    def _top_components(components: Dict[str, Any], *, limit: int = 4) -> list:
        rows = []
        for key, value in (components or {}).items():
            if key in {"source_grounding_strength", "domain_grounding_strength", "grounding_credit"}:
                continue
            score = _pct(value)
            if score <= 0:
                continue
            rows.append({"key": key, "score": score})
        rows.sort(key=lambda item: item["score"], reverse=True)
        return rows[:limit]

    def _grounding_quality_score(writing_components: Dict[str, Any]) -> int:
        components = writing_components or {}
        weighted = (
            _clamp01(components.get("source_grounding_risk")) * 0.30
            + _clamp01(components.get("citation_weakness_risk")) * 0.25
            + _clamp01(components.get("unsupported_claim_risk")) * 0.20
            + _clamp01(components.get("broad_claim_risk")) * 0.15
            + _clamp01(components.get("lived_detail_risk")) * 0.10
        )
        return _pct(weighted)

    def _combined_integrity_label(ai_score: int, grounding_score: int) -> Dict[str, str]:
        ai_band = "High AI" if ai_score >= 50 else "Low AI"
        grounding_band = "Weakly grounded" if grounding_score >= 50 else "Well grounded"
        code = f"{ai_band.lower().replace(' ', '_')}_{grounding_band.lower().replace(' ', '_')}"
        summaries = {
            "high_ai_weakly_grounded": "Machine-like authorship signals are visible and grounding quality also needs review.",
            "high_ai_well_grounded": "Machine-like authorship signals are visible, but grounding quality is not the main issue.",
            "low_ai_weakly_grounded": "AI authorship signal is limited; the main concern is grounding or evidence quality.",
            "low_ai_well_grounded": "AI authorship and grounding risk are both limited in the current scan.",
        }
        return {
            "code": code,
            "label": f"{ai_band} / {grounding_band}",
            "summary": summaries.get(code, ""),
        }

    def _integrity_layers(
        badge: Dict[str, Any],
        transformation: Dict[str, Any],
        contribution: Dict[str, Any],
    ) -> Dict[str, Any]:
        features = (transformation or {}).get("features") or {}
        ai_components = (badge or {}).get("ai_components") or {}
        writing_components = (badge or {}).get("writing_components") or {}
        ai_authorship_score = _pct((badge or {}).get("ai_likelihood_score"))
        grounding_score = _grounding_quality_score(writing_components)
        ai_transformation_score = int(contribution.get("ai_transformation_ratio") or _pct(features.get("calibrated_ai_risk")))
        human_score = int(contribution.get("human_contribution_ratio") or _pct(features.get("human_anchor_score")))
        human_score, ai_transformation_score = contribution_pair_int(human_score, ai_transformation_score)
        interpretation = _combined_integrity_label(ai_authorship_score, grounding_score)
        return {
            "schema_version": "integrity_layers.v1",
            "policy": {
                "grounding_is_not_ai_authorship": True,
                "summary": "Grounding weakness is reported as writing-integrity risk, not direct evidence of AI authorship.",
            },
            "layers": {
                "ai_authorship_risk": {
                    "score": ai_authorship_score,
                    "tier": (badge or {}).get("tier"),
                    "label": _risk_label(ai_authorship_score),
                    "source": "mechanical/statistical authorship signals",
                    "signals": _top_components(ai_components),
                    "excludes": [
                        "source_grounding_risk",
                        "citation_weakness_risk",
                        "unsupported_claim_risk",
                    ],
                },
                "ai_transformation_risk": {
                    "score": ai_transformation_score,
                    "label": _risk_label(ai_transformation_score),
                    "classification": {
                        "code": (transformation or {}).get("code"),
                        "label": (transformation or {}).get("label"),
                        "confidence": (transformation or {}).get("confidence"),
                    },
                    "signals": [
                        row for row in _transformation_signal_rows(features)
                        if row.get("family") != "grounding"
                    ][:5],
                },
                "grounding_quality_risk": {
                    "score": grounding_score,
                    "label": _risk_label(grounding_score),
                    "source": "citation, evidence, specificity, and support signals",
                    "signals": _top_components({
                        key: value
                        for key, value in writing_components.items()
                        if key in {
                            "source_grounding_risk",
                            "citation_weakness_risk",
                            "unsupported_claim_risk",
                            "broad_claim_risk",
                            "lived_detail_risk",
                        }
                    }),
                },
                "human_contribution_signal": {
                    "score": human_score,
                    "label": "strong" if human_score >= 65 else "mixed" if human_score >= 40 else "limited",
                    "source": "human anchoring, local reasoning, unevenness, and transformation balance",
                    "signals": [
                        row for row in _transformation_signal_rows(features)
                        if row.get("family") == "human_anchor"
                    ][:4],
                },
            },
            "combined_interpretation": interpretation,
            "recommended_use": {
                "ai_authorship_risk": "Use for AI-pattern review and mitigation gating.",
                "ai_transformation_risk": "Use to decide whether the text looks AI-rewritten, expanded, or paraphrased.",
                "grounding_quality_risk": "Use for source, evidence, and academic-quality feedback.",
                "human_contribution_signal": "Use to judge whether author-owned thinking is still visible.",
            },
        }

    def _industry_component_score(*values: Any) -> int:
        for value in values:
            if value is not None:
                return _pct(value)
        return 0

    def _industry_baseline(
        badge: Dict[str, Any],
        transformation: Dict[str, Any],
        contribution: Dict[str, Any],
        integrity_layers: Dict[str, Any],
        human_contract: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Turnitin-style calibration contract for authorship-vs-grounding separation.

        This is an engineering baseline, not a claim about a vendor's private
        implementation. The purpose is to expose the classes of signals the
        rewrite gate should optimize against.
        """
        features = (transformation or {}).get("features") or {}
        ai_components = (badge or {}).get("ai_components") or {}
        writing_components = (badge or {}).get("writing_components") or {}
        layers = (integrity_layers or {}).get("layers") or {}
        ai_authorship_layer = layers.get("ai_authorship_risk") or {}
        human_layer = layers.get("human_contribution_signal") or {}
        grounding_layer = layers.get("grounding_quality_risk") or {}
        ai_transform_layer = layers.get("ai_transformation_risk") or {}
        subsignals = {
            item.get("key"): item
            for item in (human_contract or {}).get("subsignals", [])
            if isinstance(item, dict) and item.get("key")
        }

        def subscore(key: str) -> int:
            return _industry_component_score((subsignals.get(key) or {}).get("score"))

        turnitin_profile = turnitin_like_ai_profile(
            features=features,
            ai_components=ai_components,
        )

        positive_authorship = [
            {
                "key": "human_anchor",
                "score": _industry_component_score(features.get("human_anchor_score"), human_layer.get("score")),
                "weight": -0.18,
                "meaning": "Concrete author-owned context suppresses AI certainty.",
            },
            {
                "key": "authorship_friction",
                "score": max(subscore("local_constraint_awareness"), subscore("causal_reasoning")),
                "weight": -0.12,
                "meaning": "Bounded judgment, causal reasoning, and tradeoffs create human-side friction.",
            },
            {
                "key": "local_irregularity",
                "score": max(0, 100 - _industry_component_score(features.get("paragraph_uniformity_risk"))),
                "weight": -0.08,
                "meaning": "Natural paragraph asymmetry suppresses template certainty.",
            },
            {
                "key": "domain_cognition",
                "score": subscore("domain_cognition"),
                "weight": -0.07,
                "meaning": "Operational domain reasoning is positive authorship evidence.",
            },
        ]
        positive_authorship.sort(key=lambda row: row["score"], reverse=True)

        authorship_components = [
            {
                "key": "token_predictability",
                "score": _industry_component_score(
                    ai_components.get("topk_calibrated_risk"),
                    ai_components.get("token_predictability"),
                    features.get("ai_likelihood"),
                    badge.get("ai_likelihood_score"),
                ),
                "weight": 0.28,
                "meaning": "Next-token regularity and low-surprise token paths.",
            },
            {
                "key": "burstiness_regularization",
                "score": _industry_component_score(
                    ai_components.get("low_burstiness"),
                    features.get("paragraph_uniformity_risk"),
                    features.get("section_style_variance"),
                ),
                "weight": 0.16,
                "meaning": "Even sentence length, pacing, and paragraph rhythm.",
            },
            {
                "key": "discourse_shape_regularization",
                "score": _industry_component_score(features.get("discourse_regularity_risk")),
                "weight": 0.14,
                "meaning": "Managed intro-development-conclusion flow and repeated paragraph jobs.",
            },
            {
                "key": "semantic_uniformity",
                "score": _industry_component_score(features.get("semantic_uniformity_risk")),
                "weight": 0.14,
                "meaning": "Stable meaning-flow with limited local drift or pressure.",
            },
            {
                "key": "template_phrase_signal",
                "score": _industry_component_score(
                    ai_components.get("generic_assertion_risk"),
                    ai_components.get("qualifying_text_density"),
                    features.get("outline_to_text_expansion"),
                ),
                "weight": 0.13,
                "meaning": "Generic academic phrase/template behavior.",
            },
            {
                "key": "rewrite_smoothness",
                "score": _industry_component_score(features.get("rewrite_smoothness")),
                "weight": 0.10,
                "meaning": "Over-polished prose with low local reasoning texture.",
            },
            {
                "key": "surface_or_source_similarity",
                "score": _industry_component_score(
                    features.get("surface_similarity"),
                    features.get("source_similarity"),
                    features.get("paraphrase_transformation_risk"),
                ),
                "weight": 0.05,
                "meaning": "Close surface or paraphrase relation where source material is available.",
            },
        ]
        authorship_components.sort(key=lambda row: row["score"], reverse=True)

        human_components = [
            {
                "key": "lived_process_detail",
                "score": subscore("lived_process_detail"),
                "meaning": "Concrete process, action, and observation detail from the submitted context.",
            },
            {
                "key": "domain_cognition",
                "score": subscore("domain_cognition"),
                "meaning": "Domain-specific operational relationships rather than glossary terms.",
            },
            {
                "key": "causal_reasoning",
                "score": subscore("causal_reasoning"),
                "meaning": "Cause, consequence, condition, and limitation links.",
            },
            {
                "key": "source_claim_ownership",
                "score": subscore("source_claim_ownership"),
                "meaning": "Author explains what a source or anchor does for the claim.",
            },
            {
                "key": "local_constraint_awareness",
                "score": subscore("local_constraint_awareness"),
                "meaning": "Judgment, limitation, and tradeoff language.",
            },
            {
                "key": "natural_variance",
                "score": subscore("natural_variance"),
                "meaning": "Uneven paragraph purpose and non-template rhythm.",
            },
        ]

        grounding_components = [
            {
                "key": "source_grounding_risk",
                "score": _industry_component_score(writing_components.get("source_grounding_risk")),
                "meaning": "Claims may need clearer source relation or evidence support.",
            },
            {
                "key": "citation_weakness_risk",
                "score": _industry_component_score(writing_components.get("citation_weakness_risk")),
                "meaning": "Citation formatting, coverage, or source linkage may be weak.",
            },
            {
                "key": "unsupported_claim_risk",
                "score": _industry_component_score(writing_components.get("unsupported_claim_risk")),
                "meaning": "Claims may be broader than the submitted support.",
            },
            {
                "key": "broad_claim_risk",
                "score": _industry_component_score(writing_components.get("broad_claim_risk")),
                "meaning": "Claims may need narrowing to the actual context.",
            },
        ]
        grounding_components.sort(key=lambda row: row["score"], reverse=True)

        return {
            "schema_version": "industry_baseline.v1",
            "baseline": "Turnitin-style market-leader approximation",
            "disclaimer": "Engineering approximation from observable detector behavior; not a vendor claim.",
            "policy": {
                "grounding_is_not_ai_authorship": True,
                "weak_grounding_can_be_human": True,
                "human_noise_is_not_typo_injection": True,
                "positive_human_signal_is_not_inverse_ai_only": True,
            },
            "score_formula": {
                "turnitin_like_ai_score": "0.45*ai_likelihood + 0.20*topk_calibrated_risk + 0.12*semantic_uniformity + 0.10*rewrite_smoothness + 0.08*patchwork_expansion + 0.05*signal_agreement - human_anchor_suppression",
                "ai_authorship_risk": "token_predictability + burstiness_regularization + discourse_shape_regularization + semantic_uniformity + template_phrase_signal + rewrite_smoothness + similarity - human_anchor - authorship_friction - local_irregularity",
                "grounding_quality_risk": "source_grounding + citation_weakness + unsupported_claim + broad_claim",
                "human_contribution_signal": "lived_process_detail + domain_cognition + causal_reasoning + source_claim_ownership + local_constraint_awareness + natural_variance",
            },
            "turnitin_like_ai_score": turnitin_profile,
            "layers": {
                "ai_authorship_risk": {
                    "score": _industry_component_score(ai_authorship_layer.get("score")),
                    "label": ai_authorship_layer.get("label"),
                    "positive_components": authorship_components,
                    "suppressors": positive_authorship,
                    "excludes": [
                        "source_grounding_risk",
                        "citation_weakness_risk",
                        "unsupported_claim_risk",
                        "broad_claim_risk",
                    ],
                    "mitigation_target": "Reduce statistical/template regularity while increasing meaningful authorship friction.",
                },
                "ai_transformation_risk": {
                    "score": _industry_component_score(ai_transform_layer.get("score")),
                    "label": ai_transform_layer.get("label"),
                    "driver_source": "scanner transformation features",
                    "mitigation_target": "Reduce rewrite-smoothness, expansion, paraphrase, and semantic-uniformity signals.",
                },
                "human_contribution_signal": {
                    "score": _industry_component_score(human_layer.get("score")),
                    "label": human_layer.get("label"),
                    "components": human_components,
                    "mitigation_target": "Increase grounded reasoning continuity and local domain cognition without fabricating new facts.",
                },
                "grounding_quality_risk": {
                    "score": _industry_component_score(grounding_layer.get("score")),
                    "label": grounding_layer.get("label"),
                    "components": grounding_components,
                    "separate_from_ai_authorship": True,
                    "mitigation_target": "Narrow unsupported claims or ask user to prepare evidence; do not count weakness as AI authorship.",
                },
            },
            "rewrite_gate_objectives": {
                "primary": "Human Contribution >= 80",
                "secondary": "AI Authorship must not regress unless major human breakthrough is achieved.",
                "quality_guard": "No critical/high/review-burden/severity regression.",
                "word_count_guard": "Regenerated content must remain within the submitted word-count band.",
            },
        }

    def _fallback_sentence_segments(text: str) -> list:
        return structured_sentence_segments(text or "")

    def _scored_segment_row(i: int, s: dict, fallback: dict) -> dict:
        return {
            "sentence_id": s.get("sentence_id", fallback.get("sentence_id") or f"s{i + 1:03d}"),
            "paragraph_id": s.get("paragraph_id") or fallback.get("paragraph_id") or "p001",
            "source_paragraph_id": s.get("source_paragraph_id") or fallback.get("source_paragraph_id") or "",
            "virtual_paragraph_id": s.get("virtual_paragraph_id") or fallback.get("virtual_paragraph_id") or s.get("paragraph_id") or fallback.get("paragraph_id") or "p001",
            "sentence_index": i,
            "start_char": s.get("start_char") if s.get("start_char") is not None else fallback.get("start_char", 0),
            "end_char": s.get("end_char") if s.get("end_char") is not None else fallback.get("end_char", 0),
            "sentence": s.get("sentence") or fallback.get("sentence", ""),
            "predictability": {
                "score": s.get("risk"),
                "risk_label": s.get("risk_label"),
                "top10_ratio": s.get("top10_ratio"),
                "top50_ratio": s.get("top50_ratio"),
                "avg_surprisal": s.get("avg_surprisal"),
                "top_predicted_tokens": s.get("top_predicted_tokens", []),
                "predictable_token_spans": s.get("predictable_token_spans", []),
            },
        }

    def _source_segments(complete: bool = False) -> list:
        # ``complete=False`` (default): scored sentences only — the exact set the
        # rewrite-handoff profiles (repair units, authorship windows, generation
        # handoff) consume. Kept scored-only so that handoff stays byte-identical.
        #
        # ``complete=True``: EVERY sentence in the submitted document. The
        # predictability scanner only scores sentences with >= 8 words
        # (poc/predictability/scanner.py floor), so ``pred_sentences`` is a SUBSET;
        # building the rendered document from it alone dropped short sentences from
        # the "submitted content" view. For the display surface we base segments on
        # the full structural split and JOIN scored rows by normalized text so the
        # display reconstructs the whole document. Unscored sentences ride along as
        # plain (no-signal) segments — inert to signal-driven downstream consumers.
        structured_segments = _fallback_sentence_segments(report.original_text or "")
        if not pred_sentences:
            return structured_segments
        if not complete or not structured_segments:
            rows = []
            for i, s in enumerate(pred_sentences):
                fallback = structured_segments[i] if i < len(structured_segments) else {}
                rows.append(_scored_segment_row(i, s, fallback))
            return rows

        def _key(text) -> str:
            return " ".join(str(text or "").split())

        # Queue scored rows per normalized text so duplicate sentences match in order.
        scored_by_text: dict = {}
        for s in pred_sentences:
            scored_by_text.setdefault(_key(s.get("sentence")), []).append(s)

        rows = []
        for i, seg in enumerate(structured_segments):
            text = seg.get("sentence", "")
            queue = scored_by_text.get(_key(text))
            matched = queue.pop(0) if queue else None
            if matched is not None:
                rows.append({
                    "sentence_id": matched.get("sentence_id") or seg.get("sentence_id") or f"s{i + 1:03d}",
                    "paragraph_id": seg.get("paragraph_id") or matched.get("paragraph_id") or "p001",
                    "source_paragraph_id": seg.get("source_paragraph_id") or matched.get("source_paragraph_id") or "",
                    "virtual_paragraph_id": seg.get("virtual_paragraph_id") or seg.get("paragraph_id") or "p001",
                    "sentence_index": i,
                    "start_char": seg.get("start_char", 0),
                    "end_char": seg.get("end_char", 0),
                    "sentence": text,
                    "predictability": {
                        "score": matched.get("risk"),
                        "risk_label": matched.get("risk_label"),
                        "top10_ratio": matched.get("top10_ratio"),
                        "top50_ratio": matched.get("top50_ratio"),
                        "avg_surprisal": matched.get("avg_surprisal"),
                        "top_predicted_tokens": matched.get("top_predicted_tokens", []),
                        "predictable_token_spans": matched.get("predictable_token_spans", []),
                    },
                })
            else:
                # Unscored sentence (below the predictability word floor). Synthetic id
                # (``u_`` prefix) can't collide with scored ``sNNN`` ids, and the empty
                # signal set means ``_document_segments`` marks it un-highlighted and
                # signal-driven consumers (e.g. rewrite_v6 _report_findings) skip it.
                struct_id = seg.get("sentence_id") or f"s{i + 1:03d}"
                rows.append({
                    "sentence_id": f"u_{struct_id}",
                    "paragraph_id": seg.get("paragraph_id") or "p001",
                    "source_paragraph_id": seg.get("source_paragraph_id") or "",
                    "virtual_paragraph_id": seg.get("virtual_paragraph_id") or seg.get("paragraph_id") or "p001",
                    "sentence_index": i,
                    "start_char": seg.get("start_char", 0),
                    "end_char": seg.get("end_char", 0),
                    "sentence": text,
                    "predictability": {},
                })
        # Safety net: a scored sentence that didn't text-match any structural segment
        # (pathological splitting) must still appear so its highlight/findings are never
        # lost. In practice the join is exhaustive, so this is normally empty.
        leftover_index = len(structured_segments)
        for queue in scored_by_text.values():
            for s in queue:
                rows.append(_scored_segment_row(leftover_index, s, {}))
                leftover_index += 1
        rows.sort(key=lambda r: (r.get("start_char") or 0, r.get("sentence_index") or 0))
        return rows

    def _signal_descriptor(f: Finding) -> Dict[str, str]:
        title = (f.title or "").lower()
        category = (f.category or "").lower()
        if "ground" in title or "citation" in title or category == "citation":
            return {
                "key": "grounding_risk",
                "label": "Grounding risk",
                "description": "Claim or citation support needs review.",
                "color": "#9a3412",
            }
        if "specificity" in title:
            return {
                "key": "human_anchor_score",
                "label": "Human anchor",
                "description": "The section may need more concrete human context.",
                "color": "#15803d",
            }
        if "similarity" in title or "paraphrase" in title or category == "similarity":
            return {
                "key": "source_similarity",
                "label": "Source similarity",
                "description": "Meaning may be too close to source material.",
                "color": "#0369a1",
            }
        if "style_shift" in title or "variance" in title:
            return {
                "key": "section_style_variance",
                "label": "Patchwork variance",
                "description": "Style differs from nearby writing.",
                "color": "#2563eb",
            }
        if "predictability" in title or "topk" in title or "surprisal" in title:
            return {
                "key": "ai_likelihood",
                "label": "AI likelihood",
                "description": "Statistical predictability is elevated.",
                "color": "#9a3412",
            }
        if "generic" in title or "smooth" in title:
            return {
                "key": "rewrite_smoothness",
                "label": "Rewrite smoothness",
                "description": "Language appears polished but generic.",
                "color": "#4338ca",
            }
        return {
            "key": f.signal_category or category or "scan_signal",
            "label": (f.signal_category or f.category or "Scan signal").replace("_", " ").title(),
            "description": f.detail[:160] if f.detail else "Scanner finding attached to this span.",
            "color": "#475569",
        }

    def _finding_score(f: Finding) -> int:
        if f.metadata:
            for key in ("score", "risk", "ai_likelihood", "top10_ratio"):
                if key in f.metadata:
                    return _pct(f.metadata.get(key))
        tier_scores = {"critical": 95, "high": 80, "medium": 55, "low": 25, "clean": 0}
        return tier_scores.get((f.tier.value if f.tier else "").lower(), 0)

    findings_by_sentence = {}
    document_level_findings = []
    for finding in all_findings:
        if finding.sentence_id:
            findings_by_sentence.setdefault(finding.sentence_id, []).append(finding)
        else:
            document_level_findings.append(finding)

    def _segment_signal(f: Finding) -> Dict[str, Any]:
        descriptor = _signal_descriptor(f)
        bucket = _determine_actionability(f, all_findings)
        return {
            "finding_id": f.finding_id,
            "key": descriptor["key"],
            "label": descriptor["label"],
            "description": descriptor["description"],
            "color": descriptor["color"],
            "category": f.category,
            "scanner": f.scanner,
            "title": f.title,
            "tier": f.tier.value if f.tier else "",
            "score": _finding_score(f),
            "actionability": bucket,
            "rewrite_permission": _rewrite_permission(f, bucket),
            "recommendation": f.recommendation,
        }

    def _document_segments(complete: bool = False) -> list:
        segments = []
        for item in _source_segments(complete):
            sid = item.get("sentence_id")
            signals = [_segment_signal(f) for f in findings_by_sentence.get(sid, [])]
            signals.sort(key=lambda entry: entry.get("score", 0), reverse=True)
            primary = signals[0] if signals else None
            segment = {
                "segment_id": sid,
                "type": "sentence",
                "sentence_id": sid,
                "paragraph_id": item.get("paragraph_id") or "",
                "sentence_index": item.get("sentence_index"),
                "start_char": item.get("start_char", 0),
                "end_char": item.get("end_char", 0),
                "text": item.get("sentence", ""),
                "signals": signals,
                "primary_signal": primary,
                "highlight": {
                    "enabled": bool(primary),
                    "color": primary.get("color") if primary else None,
                    "label": primary.get("label") if primary else None,
                    "tooltip": primary.get("description") if primary else None,
                },
                "predictability": item.get("predictability", {}),
            }
            segments.append(segment)
        return segments

    def _paragraph_map(segments: list) -> list:
        paragraphs = {}
        for segment in segments:
            pid = segment.get("paragraph_id") or "p001"
            entry = paragraphs.setdefault(pid, {
                "paragraph_id": pid,
                "sentence_ids": [],
                "start_char": segment.get("start_char", 0),
                "end_char": segment.get("end_char", 0),
                "text_parts": [],
                "finding_count": 0,
                "signals": {},
            })
            entry["sentence_ids"].append(segment.get("sentence_id"))
            if segment.get("text"):
                entry["text_parts"].append(segment.get("text"))
            entry["start_char"] = min(entry["start_char"], segment.get("start_char", entry["start_char"]))
            entry["end_char"] = max(entry["end_char"], segment.get("end_char", entry["end_char"]))
            entry["finding_count"] += len(segment.get("signals") or [])
            for signal in segment.get("signals") or []:
                key = signal["key"]
                current = entry["signals"].get(key)
                if not current or signal.get("score", 0) > current.get("score", 0):
                    entry["signals"][key] = {
                        "key": key,
                        "label": signal["label"],
                        "score": signal.get("score", 0),
                        "color": signal.get("color"),
                    }
        rows = []
        for entry in paragraphs.values():
            signals = sorted(entry.pop("signals").values(), key=lambda item: item["score"], reverse=True)
            text_parts = entry.pop("text_parts", [])
            entry["text"] = " ".join(part.strip() for part in text_parts if part and part.strip())
            entry["top_signals"] = signals[:3]
            entry["primary_signal"] = signals[0] if signals else None
            rows.append(entry)
        rows.sort(key=lambda item: item["start_char"])
        return rows

    def _radar_severity(score: float) -> str:
        score = max(0.0, min(100.0, float(score or 0.0)))
        if score >= 85:
            return "critical"
        if score >= 70:
            return "high"
        if score >= 45:
            return "medium"
        if score > 0:
            return "low"
        return "clean"

    def _radar_component_profile(key: str) -> Dict[str, str]:
        profiles = {
            "topk_pattern": {
                "layer": "ai_authorship_risk",
                "label": "Raw Top-k predictability",
                "diagnostic": "Raw GPT-2 token path is statistically predictable.",
            },
            "topk_pattern_raw": {
                "layer": "ai_authorship_risk",
                "label": "Raw Top-k predictability",
                "diagnostic": "Raw GPT-2 token path is statistically predictable.",
            },
            "topk_calibrated_risk": {
                "layer": "ai_authorship_risk",
                "label": "Calibrated Top-k risk",
                "diagnostic": "Calibrated token-route risk is above the product safe band.",
            },
            "predictability": {
                "layer": "ai_authorship_risk",
                "label": "Predictability",
                "diagnostic": "Sentence wording follows a common probability path.",
            },
            "qualifying_text_ai_density": {
                "layer": "ai_authorship_risk",
                "label": "Qualifying text density",
                "diagnostic": "Qualifying language is dense enough to look machine-shaped.",
            },
            "generic_assertion_risk": {
                "layer": "ai_authorship_risk",
                "label": "Generic assertion risk",
                "diagnostic": "Claims are stated in reusable generic form.",
            },
            "burstiness_risk": {
                "layer": "ai_authorship_risk",
                "label": "Burstiness risk",
                "diagnostic": "Sentence rhythm may be too even.",
            },
            "repeated_sentence_structure_risk": {
                "layer": "ai_authorship_risk",
                "label": "Repeated sentence structure",
                "diagnostic": "Sentence structure repeats across the draft.",
            },
            "unsupported_claim_risk": {
                "layer": "grounding_quality_risk",
                "label": "Unsupported claim risk",
                "diagnostic": "Claims need visible support, narrowing, or controller review.",
            },
            "broad_claim_risk": {
                "layer": "grounding_quality_risk",
                "label": "Broad claim risk",
                "diagnostic": "Claims are wider than the visible support.",
            },
            "citation_weakness_risk": {
                "layer": "grounding_quality_risk",
                "label": "Citation weakness",
                "diagnostic": "Source linkage is weak or not visible enough.",
            },
            "source_grounding_risk": {
                "layer": "grounding_quality_risk",
                "label": "Source grounding risk",
                "diagnostic": "Source-to-claim connection is underdeveloped.",
            },
            "lived_detail_risk": {
                "layer": "human_contribution_gap",
                "label": "Lived/process detail gap",
                "diagnostic": "Author-owned process detail is thin.",
            },
            "paragraph_progression_risk": {
                "layer": "ai_transformation_risk",
                "label": "Paragraph progression risk",
                "diagnostic": "Paragraph movement may be too managed or generic.",
            },
            "ai_likelihood": {
                "layer": "ai_authorship_risk",
                "label": "AI likelihood",
                "diagnostic": "Combined AI-authorship texture signal is elevated.",
            },
            "rewrite_smoothness": {
                "layer": "ai_transformation_risk",
                "label": "Rewrite smoothness",
                "diagnostic": "Language is smooth in a way associated with transformation.",
            },
            "outline_to_text_expansion": {
                "layer": "ai_transformation_risk",
                "label": "Expansion pattern",
                "diagnostic": "The draft expands ideas in an outline-to-prose pattern.",
            },
            "semantic_uniformity_risk": {
                "layer": "ai_transformation_risk",
                "label": "Semantic uniformity",
                "diagnostic": "Meaning flow is too even across the draft.",
            },
            "discourse_regularity_risk": {
                "layer": "ai_transformation_risk",
                "label": "Discourse regularity",
                "diagnostic": "Argument structure is too regular.",
            },
            "section_style_variance": {
                "layer": "ai_transformation_risk",
                "label": "Section style variance",
                "diagnostic": "Style shifts across sections need review.",
            },
        }
        return profiles.get(key, {
            "layer": "scan_signal",
            "label": key.replace("_", " ").title(),
            "diagnostic": "Scanner metric requires review.",
        })

    def _radar_signal_matches(component_key: str, signal: Dict[str, Any]) -> bool:
        title = str(signal.get("title") or "").lower()
        key = str(signal.get("key") or "").lower()
        if component_key in {"topk_pattern", "topk_pattern_raw", "topk_calibrated_risk"}:
            return "topk" in title
        if component_key == "predictability":
            return "predictability" in title or key == "ai_likelihood"
        if component_key == "generic_assertion_risk":
            return "generic" in title or "assertion" in title
        if component_key == "qualifying_text_ai_density":
            return "qualifying" in title
        if component_key == "burstiness_risk":
            return "burst" in title
        if component_key == "repeated_sentence_structure_risk":
            return "repetitive" in title or "structure" in title
        if component_key in {"unsupported_claim_risk", "broad_claim_risk"}:
            return "unsupported" in title or "broad" in title or "claim" in title
        if component_key in {"citation_weakness_risk", "source_grounding_risk"}:
            return "citation" in title or "source" in title or "grounding" in title
        if component_key == "lived_detail_risk":
            return "specificity" in title or "lived" in title
        if component_key == "rewrite_smoothness":
            return key == "rewrite_smoothness" or "generic" in title or "smooth" in title
        if component_key in {"semantic_uniformity_risk", "discourse_regularity_risk"}:
            return "semantic" in title or "discourse" in title or key in {"semantic_drift", "authorship_risk"}
        return False

    def _blocker_radar(
        badge: Dict[str, Any],
        features: Dict[str, Any],
        writing_components: Dict[str, Any],
        segments: list,
        paragraph_rows: list,
    ) -> Dict[str, Any]:
        """Scanner-owned blocker map.

        This is deliberately diagnostic only. It reports what is dragging the
        score, where it appears, and how confident/localized the signal is. It
        does not choose repair, recreation, or removal; the rewrite controller
        owns that policy decision.
        """
        badge = badge or {}
        features = features or {}
        writing_components = writing_components or {}
        ai_components = badge.get("ai_components") or {}
        total_sentences = max(1, len(segments or []))
        calibration_confidence = _pct(features.get("calibration_confidence"))

        metric_sources = [
            ("ai_components", ai_components, {
                "topk_pattern",
                "topk_pattern_raw",
                "topk_calibrated_risk",
                "predictability",
                "qualifying_text_ai_density",
                "generic_assertion_risk",
                "burstiness_risk",
                "repeated_sentence_structure_risk",
            }),
            ("writing_components", writing_components, {
                "unsupported_claim_risk",
                "broad_claim_risk",
                "citation_weakness_risk",
                "source_grounding_risk",
                "lived_detail_risk",
                "paragraph_progression_risk",
            }),
            ("transformation_features", features, {
                "ai_likelihood",
                "rewrite_smoothness",
                "outline_to_text_expansion",
                "semantic_uniformity_risk",
                "discourse_regularity_risk",
                "section_style_variance",
            }),
        ]

        blockers = []
        for source, metrics, keys in metric_sources:
            for key in keys:
                if key not in metrics:
                    continue
                score = _pct(metrics.get(key))
                if score < 25:
                    continue
                profile = _radar_component_profile(key)
                matched_segments = []
                matched_paragraph_ids = set()
                for segment in segments or []:
                    signals = segment.get("signals") or []
                    if any(_radar_signal_matches(key, signal) for signal in signals):
                        sid = segment.get("sentence_id")
                        if sid:
                            matched_segments.append(sid)
                        pid = segment.get("paragraph_id")
                        if pid:
                            matched_paragraph_ids.add(pid)
                if matched_segments:
                    footprint = len(set(matched_segments)) / total_sentences
                    scope = (
                        "localized"
                        if footprint <= 0.25
                        else "mixed"
                        if footprint <= 0.60
                        else "document_wide"
                    )
                else:
                    footprint = 1.0 if score >= 45 else 0.0
                    scope = "document_wide" if score >= 45 else "unlocalized"
                    matched_paragraph_ids = {
                        row.get("paragraph_id")
                        for row in paragraph_rows or []
                        if row.get("finding_count", 0) > 0
                    }
                flags = {
                    "evidence_gap": key in {
                        "unsupported_claim_risk",
                        "broad_claim_risk",
                        "citation_weakness_risk",
                        "source_grounding_risk",
                    },
                    "source_dependency": key in {
                        "citation_weakness_risk",
                        "source_grounding_risk",
                    },
                    "texture_pressure": key in {
                        "topk_pattern",
                        "topk_pattern_raw",
                        "topk_calibrated_risk",
                        "predictability",
                        "qualifying_text_ai_density",
                        "burstiness_risk",
                        "repeated_sentence_structure_risk",
                        "ai_likelihood",
                        "rewrite_smoothness",
                        "semantic_uniformity_risk",
                        "discourse_regularity_risk",
                    },
                    "author_context_gap": key in {
                        "lived_detail_risk",
                        "unsupported_claim_risk",
                        "broad_claim_risk",
                    },
                }
                blockers.append({
                    "key": key,
                    "label": profile["label"],
                    "layer": profile["layer"],
                    "metric_source": source,
                    "score": score,
                    "severity": _radar_severity(score),
                    "confidence": (
                        "high"
                        if score >= 70 and calibration_confidence >= 45
                        else "medium"
                        if score >= 45
                        else "low"
                    ),
                    "scope": scope,
                    "sentence_ids": sorted(set(matched_segments)),
                    "paragraph_ids": sorted(pid for pid in matched_paragraph_ids if pid),
                    "footprint_ratio": round(min(1.0, max(0.0, footprint)), 4),
                    "diagnostic": profile["diagnostic"],
                    "diagnostic_flags": flags,
                })

        blockers.sort(
            key=lambda item: (
                item["score"],
                len(item.get("sentence_ids") or []),
            ),
            reverse=True,
        )
        layer_pressure = {}
        for blocker in blockers:
            layer = blocker["layer"]
            layer_pressure[layer] = max(layer_pressure.get(layer, 0), blocker["score"])
        return {
            "schema_version": "blocker_radar.v1",
            "policy": {
                "scanner_role": "diagnose_only",
                "controller_role": "choose repair, recreate_from_context, or remove/defer using this radar and rewrite gates",
                "no_strategy_selected_by_scanner": True,
            },
            "calibration_confidence": calibration_confidence,
            "dominant_blockers": blockers[:8],
            "blockers": blockers,
            "layer_pressure": layer_pressure,
            "location_summary": {
                "localized_count": sum(1 for item in blockers if item.get("scope") == "localized"),
                "mixed_count": sum(1 for item in blockers if item.get("scope") == "mixed"),
                "document_wide_count": sum(1 for item in blockers if item.get("scope") == "document_wide"),
                "unlocalized_count": sum(1 for item in blockers if item.get("scope") == "unlocalized"),
            },
            "controller_inputs": {
                "has_evidence_gaps": any(item["diagnostic_flags"]["evidence_gap"] for item in blockers),
                "has_texture_pressure": any(item["diagnostic_flags"]["texture_pressure"] for item in blockers),
                "has_author_context_gap": any(item["diagnostic_flags"]["author_context_gap"] for item in blockers),
                "document_wide_pressure": any(item.get("scope") == "document_wide" and item.get("score", 0) >= 45 for item in blockers),
            },
        }

    def _unique_preserve(rows: list, value: str, kind: str, reason: str, priority: int) -> None:
        value = " ".join(str(value or "").split()).strip()
        if not value:
            return
        lower = value.lower()
        if any(item.get("text", "").lower() == lower and item.get("kind") == kind for item in rows):
            return
        rows.append({
            "text": value,
            "kind": kind,
            "reason": reason,
            "priority": priority,
        })

    def _preservation_inventory(text: str) -> Dict[str, Any]:
        """Extract scanner-owned anchors required for meaning-preserving regeneration."""
        text = text or ""
        anchors: list[dict] = []
        for match in _re.finditer(r'"([^"\n]{2,160})"|“([^”\n]{2,160})”|‘([^’\n]{2,120})’', text):
            quoted = next((group for group in match.groups() if group), "")
            _unique_preserve(anchors, quoted, "quote", "quoted/source wording", 100)
        for match in _re.finditer(
            r"\((?:[A-Z][A-Za-z'’.-]+(?:\s+(?:&|and)\s+[A-Z][A-Za-z'’.-]+)?|[A-Z][A-Za-z'’.-]+\s+et\s+al\.)\s*,\s*(?:19|20)\d{2}[a-z]?\)",
            text,
        ):
            _unique_preserve(anchors, match.group(0), "citation", "author-year citation", 100)
        for match in _re.finditer(r"\b(?:19|20)\d{2}[a-z]?\b", text):
            _unique_preserve(anchors, match.group(0), "year", "year/date anchor", 95)
        for match in _re.finditer(r"\b\d+(?:\.\d+)?\s*(?:%|percent|degrees?|hours?|weeks?|months?|years?)?\b", text, _re.I):
            _unique_preserve(anchors, match.group(0), "number", "number/measurement anchor", 90)
        for match in _re.finditer(r"\b[A-Z]{2,}[A-Z0-9]*(?:[-/][A-Z0-9]{2,})*\b", text):
            _unique_preserve(anchors, match.group(0), "acronym", "acronym or unit code", 86)
        entity_pattern = (
            r"\b[A-Z][A-Za-z'’.-]*"
            r"(?:\s+(?:(?:of|for|and|&|the|in|at)\s+)?(?:[A-Z][A-Za-z'’.-]*|I{2,3}|IV|V))*"
        )
        stop_entities = {
            "Today", "In", "This", "That", "The", "A", "An", "Many", "Students",
            "Teachers", "Learners", "However", "Therefore", "Education",
            "Access", "Another", "Assessment", "Because", "Because of", "But",
            "But the", "In the", "Knowledge", "Not", "Now", "Schools",
            "Technology", "They", "Used", "When",
        }
        for match in _re.finditer(entity_pattern, text):
            entity = match.group(0).strip()
            entity = _re.sub(r"^(?:At|By|In|For|With|From|This|The)\s+", "", entity).strip()
            if entity in stop_entities or len(entity) < 3:
                continue
            words = entity.split()
            if len(words) == 1:
                token = words[0]
                is_mixed_case = any(ch.islower() for ch in token) and any(ch.isupper() for ch in token[1:])
                is_acronym_like = token.isupper() and len(token) > 1
                if not (is_mixed_case or is_acronym_like):
                    continue
            if words and words[-1].lower() in {"of", "for", "and", "the", "in", "at"}:
                continue
            if len(words) == 1 and entity.lower() in {"teacher", "student", "learner"}:
                continue
            _unique_preserve(anchors, entity, "name_or_entity", "proper noun or named entity", 78)

        domain_terms = []
        for tier_name, flist in result.get("findings", {}).items():
            for f_info in flist:
                ev = f_info.get("evidence", {})
                if isinstance(ev, dict):
                    terms = ev.get("metrics", {}).get("domain_terms", [])
                    if isinstance(terms, list):
                        for term in terms:
                            term = str(term or "").strip()
                            if term and term.lower() not in {t.lower() for t in domain_terms}:
                                domain_terms.append(term)
                                _unique_preserve(anchors, term, "domain_term", "domain keyword from specificity layer", 70)

        headings = []
        for heading in _logical_document_outline(text).get("headings", []):
            if heading not in headings:
                headings.append(heading)
                _unique_preserve(anchors, heading, "heading", "section heading", 88)

        anchors.sort(key=lambda item: (-item["priority"], item["text"].lower()))
        return {
            "schema_version": "preservation_inventory.v1",
            "anchors": anchors[:80],
            "quotes": [a["text"] for a in anchors if a["kind"] == "quote"][:30],
            "citations": [a["text"] for a in anchors if a["kind"] == "citation"][:30],
            "years": [a["text"] for a in anchors if a["kind"] == "year"][:30],
            "numbers": [a["text"] for a in anchors if a["kind"] == "number"][:30],
            "names_entities": [a["text"] for a in anchors if a["kind"] == "name_or_entity"][:40],
            "domain_terms": domain_terms[:40],
            "headings": headings[:20],
        }

    def _word_count(text: str) -> int:
        return len(_re.findall(r"[A-Za-z0-9']+", text or ""))

    def _logical_document_outline(text: str) -> Dict[str, Any]:
        """Parse title, line-level headings, body sections, and references.

        Scan must own this because generation is based on structure and context,
        not direct modification of submitted prose.
        """
        text = text or ""
        lines = text.splitlines()
        nonempty = [(idx, line.strip()) for idx, line in enumerate(lines) if line.strip()]
        if not nonempty:
            return {"title": "", "headings": [], "sections": [], "reference_entries": []}

        title = nonempty[0][1]
        ref_start_line = None
        for idx, line in nonempty:
            if _re.match(r"^(?:references|reference list|bibliography|works cited)$", line, _re.I):
                ref_start_line = idx
                break

        def char_pos_for_line(line_index: int) -> int:
            if line_index <= 0:
                return 0
            return sum(len(line) + 1 for line in lines[:line_index])

        def is_heading(line: str, *, first_line: bool = False) -> bool:
            if first_line:
                return False
            if _re.match(r"^(?:references|reference list|bibliography|works cited)$", line, _re.I):
                return True
            words = line.split()
            if not words or len(words) > 12:
                return False
            if _re.search(r"[.!?;:]$", line):
                return False
            if _re.search(r"\(\d{4}\)|https?://|doi\.", line, _re.I):
                return False
            starts_like_heading = line[0].isupper()
            has_lowercase_words = any(any(ch.islower() for ch in word) for word in words)
            return starts_like_heading and has_lowercase_words

        sections: list[dict] = []
        current: dict | None = None
        body_end_line = ref_start_line if ref_start_line is not None else len(lines)
        for idx, raw_line in enumerate(lines[:body_end_line]):
            line = raw_line.strip()
            if not line:
                continue
            if idx == nonempty[0][0]:
                continue
            if is_heading(line):
                if current:
                    current["end_char"] = max(current["start_char"], char_pos_for_line(idx) - 1)
                    current["text"] = "\n".join(current.pop("_lines")).strip()
                    current["word_count"] = _word_count(current["text"])
                    sections.append(current)
                current = {
                    "section_id": f"sec_{len(sections) + 1:03d}",
                    "heading": line,
                    "start_char": char_pos_for_line(idx),
                    "_lines": [],
                }
                continue
            if current is None:
                current = {
                    "section_id": f"sec_{len(sections) + 1:03d}",
                    "heading": "Main Body",
                    "start_char": char_pos_for_line(idx),
                    "_lines": [],
                }
            current["_lines"].append(line)
        if current:
            current["end_char"] = max(
                current["start_char"],
                char_pos_for_line(body_end_line) - 1 if body_end_line <= len(lines) else len(text),
            )
            current["text"] = "\n".join(current.pop("_lines")).strip()
            current["word_count"] = _word_count(current["text"])
            sections.append(current)

        reference_entries: list[dict] = []
        if ref_start_line is not None:
            current_ref = ""
            for raw_line in lines[ref_start_line + 1:]:
                line = raw_line.strip()
                if not line:
                    if current_ref:
                        reference_entries.append({"reference_id": f"ref_{len(reference_entries) + 1:03d}", "full_reference": current_ref.strip()})
                        current_ref = ""
                    continue
                starts_entry = bool(_re.search(r"\(\d{4}\)|\(\s*n\.d\.\s*\)|https?://|doi\.", line, _re.I))
                if current_ref and starts_entry:
                    reference_entries.append({"reference_id": f"ref_{len(reference_entries) + 1:03d}", "full_reference": current_ref.strip()})
                    current_ref = line
                else:
                    current_ref = f"{current_ref} {line}".strip() if current_ref else line
            if current_ref:
                reference_entries.append({"reference_id": f"ref_{len(reference_entries) + 1:03d}", "full_reference": current_ref.strip()})

        return {
            "title": title,
            "headings": [section.get("heading") for section in sections if section.get("heading")],
            "sections": sections,
            "reference_entries": reference_entries,
        }

    def _citation_keys(text: str) -> list[str]:
        keys = []
        for match in _re.finditer(r"\(([A-Z][^)]+?,\s*(?:19|20)\d{2}[a-z]?)\)", text or ""):
            key = " ".join(match.group(1).split())
            if key not in keys:
                keys.append(key)
        narrative_pattern = (
            r"\b([A-Z][A-Za-z'’.-]+"
            r"(?:\s+(?:and|&)\s+[A-Z][A-Za-z'’.-]+)?"
            r"(?:\s+et\s+al\.)?)\s*\(((?:19|20)\d{2}[a-z]?)\)"
        )
        for match in _re.finditer(narrative_pattern, text or ""):
            key = f"{' '.join(match.group(1).split())}, {match.group(2)}"
            if key not in keys:
                keys.append(key)
        return keys[:12]

    def _section_role(heading: str, index: int, total: int) -> str:
        lower = (heading or "").lower()
        if "intro" in lower:
            return "context_and_thesis"
        if "lost" in lower or "challenge" in lower:
            return "problem_and_causal_explanation"
        if "show" in lower or "demonstrat" in lower:
            return "instructional_design_and_support"
        if "adjustment" in lower or "classroom" in lower:
            return "reasonable_adjustment_and_constraints"
        if "standard" in lower or "access" in lower:
            return "standards_access_and_author_judgement"
        if "conclusion" in lower or index == total:
            return "synthesis_and_closure"
        return "development"

    def _anchor_register_from_inventory(preservation_inventory: Dict[str, Any]) -> Dict[str, Any]:
        anchors = preservation_inventory or {}
        unit_codes = []
        institutions = []
        cohort_terms = []
        for item in anchors.get("anchors", []) or []:
            text_value = item.get("text") if isinstance(item, dict) else ""
            if not text_value:
                continue
            if _re.match(r"^[A-Z]{2,}[A-Z0-9/-]*$", text_value):
                if text_value not in unit_codes:
                    unit_codes.append(text_value)
            if _re.search(r"\b(?:Institute|University|Department|Government|CAST|CESE|UNESCO|TAFE)\b", text_value):
                if text_value not in institutions:
                    institutions.append(text_value)
            if _re.match(r"^[A-Z]{2,}\d{2,}$", text_value):
                cohort_terms.append(text_value)
        return {
            "institutions": institutions[:20],
            "unit_codes": unit_codes[:30],
            "cohort_terms": cohort_terms[:20],
            "technical_terms": anchors.get("domain_terms") or [],
            "numbers": anchors.get("numbers") or [],
            "years": anchors.get("years") or [],
            "citations": anchors.get("citations") or [],
            "names_entities": anchors.get("names_entities") or [],
        }

    def _meaning_inventory_for_section(section_text: str, preservation_inventory: Dict[str, Any]) -> list[dict]:
        stop_words = {
            "about", "after", "again", "also", "because", "before", "being", "between",
            "class", "could", "does", "from", "have", "into", "more", "must", "need",
            "only", "should", "some", "than", "that", "their", "there", "these", "this",
            "through", "when", "where", "while", "with", "without", "learners", "learner",
            "training", "teaching", "practice", "practical",
        }
        preserve_terms = []
        for key in ("domain_terms", "names_entities", "citations", "years", "numbers"):
            preserve_terms.extend((preservation_inventory or {}).get(key) or [])
        sentences = [
            sentence.strip()
            for sentence in _re.split(r"(?<=[.!?])\s+", section_text or "")
            if sentence.strip()
        ]
        rows = []
        for index, sentence in enumerate(sentences[:10], start=1):
            anchors = [
                term for term in preserve_terms
                if term and term in sentence
            ][:10]
            keywords = []
            for token in _re.findall(r"[A-Za-z][A-Za-z'-]{3,}", sentence):
                lower = token.lower()
                if lower in stop_words:
                    continue
                if lower not in {k.lower() for k in keywords}:
                    keywords.append(token)
                if len(keywords) >= 10:
                    break
            lower_sentence = sentence.lower()
            if any(word in lower_sentence for word in ("i see", "i notice", "i usually", "i do not", "i have seen", "i may", "i want")):
                claim_type = "author_observation"
            elif any(word in lower_sentence for word in ("source", "states", "argue", "explain", "describe", "defines")):
                claim_type = "source_relation"
            elif any(word in lower_sentence for word in ("because", "therefore", "so", "which means", "this means", "if")):
                claim_type = "causal_reasoning"
            else:
                claim_type = "context_or_development"
            rows.append({
                "point_id": f"mp_{index:03d}",
                "claim_type": claim_type,
                "keywords": keywords,
                "anchors": anchors,
                "citation_keys": _citation_keys(sentence),
                "author_stance": "first_person_observation" if claim_type == "author_observation" else "",
            })
        return rows

    def _generation_handoff(
        text: str,
        segments: list,
        preservation_inventory: Dict[str, Any],
        human_contract: Dict[str, Any],
        industry_baseline: Dict[str, Any],
    ) -> Dict[str, Any]:
        outline = _logical_document_outline(text or "")
        word_count = _word_count(text or "")
        target_variance = 0.25
        target_min = int(word_count * (1.0 - target_variance))
        target_max = int(word_count * (1.0 + target_variance))
        references = []
        for ref in outline.get("reference_entries", []) or []:
            full = ref.get("full_reference") or ""
            year_match = _re.search(r"\((?:19|20)\d{2}[a-z]?\)", full)
            author = full.split(".")[0].strip() if full else ""
            citation_key = f"{author} {year_match.group(0)}" if author and year_match else author
            references.append({
                "reference_id": ref.get("reference_id"),
                "citation_key": citation_key,
                "full_reference": full,
                "preserve_exactly": True,
            })

        body_word_count = sum(section.get("word_count", 0) for section in outline.get("sections", []) or []) or max(1, word_count)
        section_units = []
        total_sections = len(outline.get("sections", []) or [])
        for index, section in enumerate(outline.get("sections", []) or [], start=1):
            section_text = section.get("text") or ""
            section_words = section.get("word_count") or 0
            proportional_min = max(80, int(target_min * (section_words / max(1, body_word_count))))
            proportional_max = max(proportional_min + 20, int(target_max * (section_words / max(1, body_word_count))))
            section_signals = []
            start = section.get("start_char", 0)
            end = section.get("end_char", start)
            for segment in segments or []:
                if segment.get("start_char", 0) <= end and segment.get("end_char", 0) >= start:
                    for signal in segment.get("signals", []) or []:
                        key = signal.get("key")
                        if key and key not in {s.get("key") for s in section_signals}:
                            section_signals.append({
                                "key": key,
                                "label": signal.get("label"),
                                "score": signal.get("score"),
                                "rewrite_permission": signal.get("rewrite_permission"),
                            })
            section_units.append({
                "section_id": section.get("section_id"),
                "heading": section.get("heading"),
                "role": _section_role(section.get("heading", ""), index, total_sections),
                "source_span": {
                    "start_char": start,
                    "end_char": end,
                    "source_text_exposed_to_generator": False,
                },
                "current_word_count": section_words,
                "target_words": {
                    "min": proportional_min,
                    "max": proportional_max,
                    "ideal": max(proportional_min, int((proportional_min + proportional_max) / 2)),
                },
                "meaning_inventory": _meaning_inventory_for_section(section_text, preservation_inventory),
                "citation_keys_used": _citation_keys(section_text),
                "must_preserve_anchors": [
                    anchor.get("text")
                    for anchor in (preservation_inventory.get("anchors") or [])
                    if isinstance(anchor, dict)
                    and anchor.get("text")
                    and anchor.get("text") in section_text
                    and not (anchor.get("kind") == "number" and _re.match(r"^\d$", str(anchor.get("text") or "")))
                    and not (
                        anchor.get("kind") == "name_or_entity"
                        and (
                            len(str(anchor.get("text") or "").split()) > 6
                            or _re.match(r"^(?:At|By|In|For|With|From|This|The)\b", str(anchor.get("text") or ""))
                        )
                    )
                ][:25],
                "detector_risks_to_reduce": section_signals[:8],
                "generation_instruction": {
                    "generate_new_section": True,
                    "do_not_copy_sentence_order": True,
                    "do_not_add_new_evidence": True,
                    "preserve_meaning_not_sentence_order": True,
                },
            })

        return {
            "schema_version": "generation_handoff.v1",
            "source_policy": {
                "expose_original_prose_to_generator": False,
                "generation_mode": "context_regeneration",
                "preserve_meaning_not_sentence_order": True,
            },
            "document_profile": {
                "title": outline.get("title") or "",
                "document_type": "reflective_or_analytical_submission",
                "word_count": word_count,
                "body_word_count": body_word_count,
                "reference_count": len(references),
                "target_word_band": {
                    "min": target_min,
                    "max": target_max,
                    "variance": target_variance,
                },
            },
            "logical_outline": [
                {
                    "section_id": unit.get("section_id"),
                    "heading": unit.get("heading"),
                    "role": unit.get("role"),
                    "current_word_count": unit.get("current_word_count"),
                    "target_words": unit.get("target_words"),
                }
                for unit in section_units
            ],
            "anchor_register": _anchor_register_from_inventory(preservation_inventory),
            "reference_register": references,
            "section_generation_units": section_units,
            "generation_constraints": {
                "do_not_expose_original_prose": True,
                "preserve_references_exactly": True,
                "do_not_invent_evidence": True,
                "word_count_variance": target_variance,
                "target_human_contribution": 80,
                "target_ai_transformation": 20,
                "user_evidence_footnote": (
                    ((human_contract or {}).get("generation_readiness") or {}).get("user_evidence_footnote")
                    or "Keep ready real notes, sources, observations, or process evidence that support the claims if review is needed."
                ),
            },
            "industry_baseline_focus": (industry_baseline or {}).get("rewrite_gate_objectives") or {},
        }

    def _unique_structured_values(values: list) -> list:
        seen = set()
        unique = []
        for value in values or []:
            if isinstance(value, dict):
                key = tuple(sorted((str(k), str(v)) for k, v in value.items()))
            else:
                key = str(value)
            if key in seen:
                continue
            seen.add(key)
            unique.append(value)
        return unique

    def _dedupe_preservation_anchors(anchors: list) -> list[dict]:
        seen = set()
        unique = []
        for anchor in anchors or []:
            if not isinstance(anchor, dict):
                continue
            text_value = str(anchor.get("text") or "").strip()
            if not text_value:
                continue
            key = (
                text_value,
                str(anchor.get("kind") or anchor.get("type") or ""),
                str(anchor.get("category") or ""),
                str(anchor.get("severity") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(anchor)
        return unique

    def _generation_handoff_citation_keys(generation_handoff: Dict[str, Any]) -> list:
        keys = []
        for unit in (generation_handoff or {}).get("section_generation_units") or []:
            if not isinstance(unit, dict):
                continue
            keys.extend(unit.get("citation_keys_used") or [])
            for meaning in unit.get("meaning_inventory") or []:
                if isinstance(meaning, dict):
                    keys.extend(meaning.get("citation_keys") or [])
        return _unique_structured_values(keys)

    def _rewrite_routing_signals(
        preservation_inventory: Dict[str, Any],
        generation_handoff: Dict[str, Any],
        *,
        word_count: int,
    ) -> Dict[str, Any]:
        anchors = _dedupe_preservation_anchors((preservation_inventory or {}).get("anchors") or [])
        quote_values = _unique_structured_values((preservation_inventory or {}).get("quotes") or [])
        citation_values = _unique_structured_values((preservation_inventory or {}).get("citations") or [])
        reference_register = _unique_structured_values((generation_handoff or {}).get("reference_register") or [])
        citation_keys = _generation_handoff_citation_keys(generation_handoff or {})

        quote_anchor_count = sum(
            1
            for anchor in anchors
            if str(anchor.get("kind") or anchor.get("type") or anchor.get("category") or "") in {"quote", "direct_quote"}
        )
        citation_anchor_count = sum(
            1
            for anchor in anchors
            if str(anchor.get("kind") or anchor.get("type") or anchor.get("category") or "") in {"citation", "source_citation"}
        )
        hard_anchor_count = sum(1 for anchor in anchors if str(anchor.get("severity") or "").startswith("hard"))
        role_counts = {
            "direct_quote": 0,
            "evidence_quote": 0,
            "citation_quote": 0,
            "concept_quote": 0,
            "title_quote": 0,
            "dialogue_quote": 0,
            "ordinary_quote": 0,
            "unknown_quote": 0,
        }
        for anchor in anchors:
            role = str(anchor.get("quote_role") or anchor.get("anchor_role") or anchor.get("role") or "")
            if role in role_counts:
                role_counts[role] += 1
            elif str(anchor.get("kind") or anchor.get("type") or anchor.get("category") or "") in {"quote", "direct_quote"}:
                role_counts["unknown_quote"] += 1

        quote_count = max(len(quote_values), quote_anchor_count)
        citation_count = max(len(citation_values), citation_anchor_count)
        citation_signal_count = max(citation_count, len(citation_keys), len(reference_register))
        direct_evidence_score = min(
            1.0,
            (
                role_counts["direct_quote"] * 0.35
                + role_counts["evidence_quote"] * 0.45
                + role_counts["citation_quote"] * 0.45
                + citation_signal_count * 0.18
                + hard_anchor_count * 0.08
            ),
        )
        untyped_quote_score = 0.0 if direct_evidence_score >= 0.5 else min(0.12, quote_count * 0.03)
        evidence_anchor_score = min(1.0, direct_evidence_score + untyped_quote_score)
        anchor_preservation_pressure = min(
            1.0,
            direct_evidence_score
            + hard_anchor_count * 0.08
            + citation_signal_count * 0.08,
        )
        words = max(1, int(word_count or 0))
        return {
            "schema_version": "rewrite_routing_signals.v1",
            "anchor_metrics": {
                "raw_anchor_count": len((preservation_inventory or {}).get("anchors") or []),
                "dedup_anchor_count": len(anchors),
                "quote_count": quote_count,
                "quote_density": round(quote_count / words, 4),
                "citation_count": citation_count,
                "citation_density": round(citation_count / words, 4),
                "citation_key_count": len(citation_keys),
                "reference_count": len(reference_register),
                "hard_anchor_count": hard_anchor_count,
                "quote_role_counts": role_counts,
                "evidence_anchor_score": round(evidence_anchor_score, 3),
                "anchor_preservation_pressure": round(anchor_preservation_pressure, 3),
            },
            "routing_policy": {
                "quote_count_is_not_quote_heavy": True,
                "untyped_quotes_are_low_confidence": True,
                "chunking_requires_preservation_pressure": True,
            },
        }

    def _count_pattern(text: str, pattern: str) -> int:
        return len(_re.findall(pattern, text or "", flags=_re.I))

    def _human_contribution_contract(
        text: str,
        segments: list,
        paragraph_rows: list,
        integrity_layers: Dict[str, Any],
        features: Dict[str, Any],
        writing_components: Dict[str, Any],
        preservation_inventory: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Explain what is missing for Human Contribution and how rewrite can target it."""
        text = text or ""
        layers = (integrity_layers.get("layers") or {}) if isinstance(integrity_layers, dict) else {}
        current_human = _pct((layers.get("human_contribution_signal") or {}).get("score"))
        ai_transformation = _pct((layers.get("ai_transformation_risk") or {}).get("score"))
        ai_authorship = _pct((layers.get("ai_authorship_risk") or {}).get("score"))
        grounding = _pct((layers.get("grounding_quality_risk") or {}).get("score"))
        domain_terms = (preservation_inventory or {}).get("domain_terms") or []
        hard_anchors = (preservation_inventory or {}).get("anchors") or []
        process_markers = _count_pattern(
            text,
            r"\b(?:when|while|before|after|during|step|process|practice|feedback|observe|adjust|compare|check|try|repeat)\b",
        )
        causal_markers = _count_pattern(
            text,
            r"\b(?:because|therefore|so|which means|this means|as a result|leads to|depends on|if|unless)\b",
        )
        judgment_markers = _count_pattern(
            text,
            r"\b(?:I think|I notice|I see|I do not|I usually|may|might|can|cannot|should|needs?|risk|limit|tension|challenge)\b",
        )
        source_markers = _count_pattern(text, r"\b(?:according to|states|argues|explains|shows|describes|source|citation)\b|\([A-Z][^)]+,\s*(?:19|20)\d{2}")
        paragraph_count = max(1, len(paragraph_rows or []))
        word_count = max(1, len(text.split()))

        def score(name: str, value: float, evidence: str, action: str) -> Dict[str, Any]:
            value = max(0, min(100, int(round(value))))
            return {
                "key": name,
                "score": value,
                "label": "strong" if value >= 70 else "mixed" if value >= 45 else "weak",
                "evidence": evidence,
                "rewrite_lever": action,
            }

        subsignals = [
            score(
                "lived_process_detail",
                100 - _pct(writing_components.get("lived_detail_risk")),
                f"{process_markers} process/practice markers",
                "Add concrete process reasoning already implied by the draft; do not invent personal events.",
            ),
            score(
                "domain_cognition",
                min(100, len(domain_terms) * 4 + len(hard_anchors) * 2),
                f"{len(domain_terms)} domain terms and {len(hard_anchors)} preserved anchors",
                "Keep domain terms and use them to explain relationships, not as a glossary list.",
            ),
            score(
                "causal_reasoning",
                min(100, causal_markers * 12),
                f"{causal_markers} causal or conditional markers",
                "Make cause-effect links explicit where the submitted claims already imply them.",
            ),
            score(
                "source_claim_ownership",
                100 - _pct(writing_components.get("source_grounding_risk")),
                f"{source_markers} source-relation markers",
                "Connect source ideas to claims in the author's own reasoning, or narrow unsupported claims.",
            ),
            score(
                "local_constraint_awareness",
                min(100, judgment_markers * 8),
                f"{judgment_markers} judgement, limitation, or constraint markers",
                "Add bounded judgement, limitation, or tradeoff language from the submitted context.",
            ),
            score(
                "natural_variance",
                100 - max(
                    _pct(features.get("paragraph_uniformity_risk")),
                    _pct(features.get("discourse_regularity_risk")),
                    _pct(features.get("semantic_uniformity_risk")),
                ),
                "paragraph/discourse/semantic uniformity risks inverted",
                "Vary paragraph jobs and sentence route; avoid the same claim-explain-summary pattern.",
            ),
        ]

        weak_keys = [item["key"] for item in subsignals if item["score"] < 45]
        medium_keys = [item["key"] for item in subsignals if 45 <= item["score"] < 70]
        # Auto reachability must be conservative. Domain terms, citations, and
        # source-looking structure are not new human evidence; they only give
        # bounded room to strengthen reasoning already present in the submission.
        auto_safe_keys = {
            "causal_reasoning",
            "source_claim_ownership",
            "local_constraint_awareness",
            "natural_variance",
        }
        weak_auto_keys = [key for key in weak_keys if key in auto_safe_keys]
        medium_auto_keys = [key for key in medium_keys if key in auto_safe_keys]
        auto_gain_potential = min(
            16,
            len(weak_auto_keys) * 5 + len(medium_auto_keys) * 2,
        )
        assume_author_evidence = os.environ.get(
            "DRAFTPROOF_ASSUME_AUTHOR_EVIDENCE",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        evidence_gap_penalty = 0 if assume_author_evidence else 12 if grounding >= 65 else 5 if grounding >= 45 else 0
        implicit_evidence_gain = (
            min(
                8,
                process_markers * 0.30
                + causal_markers * 0.70
                + source_markers * 0.50,
            )
            if assume_author_evidence
            else 0
        )
        texture_pressure = max(ai_authorship, ai_transformation)
        total_auto_gain = auto_gain_potential + implicit_evidence_gain - evidence_gap_penalty
        if texture_pressure >= 60:
            total_auto_gain = min(total_auto_gain, 8)
        elif texture_pressure >= 45:
            total_auto_gain = min(total_auto_gain, 12)
        auto_reachable = max(
            current_human,
            min(100, current_human + total_auto_gain),
        )
        author_input_gain = 20 if grounding >= 45 or weak_keys or medium_keys else 12
        with_author_input = min(
            100,
            max(auto_reachable, current_human + total_auto_gain + author_input_gain),
        )

        paragraph_levers = []
        for paragraph in (paragraph_rows or [])[:12]:
            top_signals = paragraph.get("top_signals") or []
            pid = paragraph.get("paragraph_id")
            if top_signals:
                primary = top_signals[0]
                signal_key = primary.get("key")
            else:
                primary = {}
                signal_key = "human_anchor"
            if signal_key in {"ai_likelihood", "rewrite_smoothness"}:
                lever = "Change sentence route and paragraph role; avoid generic transitions."
            elif signal_key in {"grounding_risk", "source_similarity"}:
                lever = "Narrow the claim or add source-to-claim reasoning from existing source relations."
            elif signal_key == "human_anchor_score":
                lever = "Add concrete process, constraint, or judgement already implied by the paragraph."
            else:
                lever = "Assign a clearer paragraph job and add local reasoning continuity."
            paragraph_levers.append({
                "paragraph_id": pid,
                "sentence_ids": paragraph.get("sentence_ids") or [],
                "current_top_signals": top_signals[:3],
                "recommended_role": (
                    "source_to_claim_reasoning"
                    if signal_key in {"grounding_risk", "source_similarity"}
                    else "process_or_constraint_reasoning"
                    if signal_key == "human_anchor_score"
                    else "asymmetric_reasoning_route"
                ),
                "rewrite_lever": lever,
            })

        readiness = {
            "auto_regeneration_possible": auto_reachable > current_human + 5,
            "target_human_contribution": 80,
            "estimated_auto_reachable_human_contribution": int(round(auto_reachable)),
            "estimated_with_author_input_human_contribution": int(round(with_author_input)),
            "assume_author_evidence_from_submission": assume_author_evidence,
            "requires_author_input_for_80": auto_reachable < 80,
            "user_evidence_footnote": (
                "DraftProof can reconstruct from the submitted write-up, but you should keep ready any real notes, sources, examples, observations, or process evidence that support the claims if review is needed."
            ),
            "reason": (
                "Scanner signals suggest automatic regeneration may reach the target without new facts."
                if auto_reachable >= 80
                else "Human Contribution above 80 likely needs real author evidence, source-specific grounding, or stronger author-owned process context."
            ),
        }
        return {
            "schema_version": "human_contribution_contract.v1",
            "current_human_contribution": current_human,
            "target_human_contribution": 80,
            "current_ai_transformation": ai_transformation,
            "current_ai_authorship": ai_authorship,
            "current_grounding_risk": grounding,
            "subsignals": subsignals,
            "weak_subsignals": weak_keys,
            "medium_subsignals": medium_keys,
            "paragraph_levers": paragraph_levers,
            "generation_readiness": readiness,
            "safe_generation_levers": [
                item["rewrite_lever"] for item in subsignals if item["score"] < 70
            ][:8],
            "blocked_or_author_needed_levers": [
                "new personal observation",
                "new citation or source evidence",
                "new named institution, date, statistic, or example",
            ] if auto_reachable < 80 else [],
            "assumption_policy": {
                "mode": (
                    "implicit_author_evidence"
                    if assume_author_evidence
                    else "explicit_evidence_required"
                ),
                "summary": (
                    "Treat submitted claims as author-owned context for reconstruction when evidence is not separately uploaded. "
                    "Generation may strengthen reasoning and narrow claims, but must not invent citations, dates, names, statistics, or new events."
                    if assume_author_evidence
                    else "Do not assume missing evidence exists outside the submission."
                ),
            },
        }

    def _scan_intelligence() -> Dict[str, Any]:
        badge = report.ai_risk_badge or {}
        transformation = badge.get("transformation_classification") or {}
        features = transformation.get("features") or {}
        writing_components = badge.get("writing_components") or {}
        ai_components = badge.get("ai_components") or {}
        transformation_signals = _transformation_signal_rows(features)
        for key, label, description in (
            (
                "topk_pattern_raw",
                "Raw Top-k Predictability",
                "Raw GPT-2 token-route concentration. Diagnostic only; not the safe-band gate.",
            ),
            (
                "topk_calibrated_risk",
                "Calibrated Top-k Risk",
                "Calibrated risk from raw GPT-2 Top-k. Safe-band target: below 25%.",
            ),
        ):
            value = ai_components.get(key)
            if isinstance(value, (int, float)) and not any(row.get("key") == key for row in transformation_signals):
                transformation_signals.append({
                    "key": key,
                    "label": label,
                    "description": description,
                    "family": "ai_authorship_risk",
                    "higher_score_means": "higher token-route risk",
                    "score": round(max(0.0, min(100.0, float(value))), 2),
                    "raw_score": round(max(0.0, min(100.0, float(value))) / 100.0, 4),
                    "metric_source": "ai_components",
                })
        transformation_signals.sort(key=lambda item: item["score"], reverse=True)
        contribution = _transformation_contribution(features, transformation_signals, ai_components)
        integrity_layers = _integrity_layers(badge, transformation, contribution)
        segments = _document_segments()
        paragraph_rows = _paragraph_map(segments)
        # Display surface: the complete document (scored + unscored short sentences),
        # used ONLY for the rendered "submitted content" (document.segments /
        # highlight_segments / document.paragraphs). The handoff profiles below keep
        # consuming the scored-only ``segments``/``paragraph_rows`` so their output is
        # byte-identical to before this display fix.
        display_segments = _document_segments(complete=True)
        display_paragraph_rows = _paragraph_map(display_segments)
        authorship_window_profile = build_authorship_window_profile(
            source_text=report.original_text or "",
            segments=segments,
            paragraphs=paragraph_rows,
        )
        ai_footprint_profile = (
            authorship_window_profile.get("ai_footprint_profile")
            if isinstance(authorship_window_profile.get("ai_footprint_profile"), dict)
            else build_ai_footprint_profile(authorship_window_profile)
        )
        doc_findings = [_segment_signal(f) for f in document_level_findings]
        doc_findings.sort(key=lambda entry: entry.get("score", 0), reverse=True)
        preservation_inventory = _preservation_inventory(report.original_text or "")
        rewrite_target_profile = build_rewrite_target_profile(
            source_text=report.original_text or "",
            authorship_window_profile=authorship_window_profile,
            ai_footprint_profile=ai_footprint_profile,
            preservation_inventory=preservation_inventory,
        )
        problem_inventory = build_problem_inventory(
            rewrite_target_profile=rewrite_target_profile,
            ai_footprint_profile=ai_footprint_profile,
        )
        blocker_radar = _blocker_radar(
            badge,
            features,
            writing_components,
            segments,
            paragraph_rows,
        )
        repair_units_v2 = build_repair_units_v2(
            source_text=report.original_text or "",
            segments=segments,
            paragraph_rows=paragraph_rows,
            blocker_radar=blocker_radar,
            authorship_window_profile=authorship_window_profile,
            rewrite_target_profile=rewrite_target_profile,
        )
        human_contract = _human_contribution_contract(
            report.original_text or "",
            segments,
            paragraph_rows,
            integrity_layers,
            features,
            writing_components,
            preservation_inventory,
        )
        industry_baseline = _industry_baseline(
            badge,
            transformation,
            contribution,
            integrity_layers,
            human_contract,
        )
        generation_handoff = _generation_handoff(
            report.original_text or "",
            segments,
            preservation_inventory,
            human_contract,
            industry_baseline,
        )
        rewrite_routing_signals = _rewrite_routing_signals(
            preservation_inventory,
            generation_handoff,
            word_count=len((report.original_text or "").split()),
        )
        generation_handoff["rewrite_routing_signals"] = rewrite_routing_signals
        return {
            "schema_version": "scan_intelligence.v1",
            "purpose": {
                "reader_report": "Explain the scan through transformation pattern, core signals, and highlighted source spans.",
                "mitigation_pipeline": "Provide stable span ids, risk signals, permissions, and preservation constraints for downstream rewrite planning.",
            },
            "document": {
                "word_count": len(report.original_text.split()) if report.original_text else 0,
                "sentence_count": len(display_segments),
                "paragraph_count": len({s.get("paragraph_id") for s in display_segments if s.get("paragraph_id")}),
                "segments": display_segments,
                "paragraphs": display_paragraph_rows,
                "authorship_window_profile": authorship_window_profile,
                "ai_footprint_profile": ai_footprint_profile,
                "rewrite_target_profile": rewrite_target_profile,
                "problem_inventory": problem_inventory,
                "repair_units_v2": repair_units_v2,
                "preservation_inventory": preservation_inventory,
                "anchor_metrics": rewrite_routing_signals.get("anchor_metrics") or {},
            },
            "transformation": {
                "classification": transformation,
                "contribution": contribution,
                "core_signals": transformation_signals,
                "strongest_signals": transformation_signals[:3],
            },
            "integrity_layers": integrity_layers,
            "blocker_radar": blocker_radar,
            "industry_baseline": industry_baseline,
            "human_contribution_contract": human_contract,
            "generation_handoff": generation_handoff,
            "rewrite_routing_signals": rewrite_routing_signals,
            "authorship_window_profile": authorship_window_profile,
            "ai_footprint_profile": ai_footprint_profile,
            "rewrite_target_profile": rewrite_target_profile,
            "problem_inventory": problem_inventory,
            "repair_units_v2": repair_units_v2,
            "calibration": {
                "raw_ai_likelihood": _pct(features.get("ai_likelihood")),
                "adjusted_ai_risk": _pct(features.get("adjusted_ai_risk")),
                "calibrated_ai_risk": _pct(features.get("calibrated_ai_risk")),
                "human_anchor_discount": _pct(features.get("human_anchor_discount")),
                "signal_agreement_score": _pct(features.get("signal_agreement_score")),
                "calibration_confidence": _pct(features.get("calibration_confidence")),
                "reporting_suppression": _pct(features.get("reporting_suppression")),
                "policy": "Conservative reporting: human anchors and low-confidence coverage suppress AI certainty before report interpretation.",
            },
            "semantic_layer": {
                "status": (
                    "embedding_analysis_ready"
                    if report.semantic_shape and report.semantic_shape.embedding_model_attached
                    else "hashed_vector_fallback_ready"
                    if report.semantic_shape
                    else "heuristic_proxy_ready"
                ),
                "semantic_uniformity_risk": _pct(features.get("semantic_uniformity_risk")),
                "discourse_regularity_risk": _pct(features.get("discourse_regularity_risk")),
                "semantic_drift_risk": (
                    _pct(report.semantic_shape.semantic_drift_risk)
                    if report.semantic_shape else 0
                ),
                "paraphrase_transformation_risk": _pct(features.get("paraphrase_transformation_risk")),
                "embedding_model_attached": bool(report.semantic_shape and report.semantic_shape.embedding_model_attached),
                "model_name": report.semantic_shape.model_name if report.semantic_shape else "not_attached",
                "adjacent_similarity_mean": (
                    round(report.semantic_shape.adjacent_similarity_mean, 4)
                    if report.semantic_shape else 0.0
                ),
                "adjacent_similarity_std": (
                    round(report.semantic_shape.adjacent_similarity_std, 4)
                    if report.semantic_shape else 0.0
                ),
                "paragraph_similarity_mean": (
                    round(report.semantic_shape.paragraph_similarity_mean, 4)
                    if report.semantic_shape else 0.0
                ),
                "paragraph_similarity_std": (
                    round(report.semantic_shape.paragraph_similarity_std, 4)
                    if report.semantic_shape else 0.0
                ),
                "next_upgrade": "Use sentence-transformer embeddings in production and add source-aware semantic comparison where source material is available.",
            },
            "signal_inventory": {
                "ai_components": badge.get("ai_components") or {},
                "writing_components": badge.get("writing_components") or {},
                "authorship_concern": report.authorship_concern_signals or {},
                "document_level_signals": doc_findings,
                "actionability_distribution": report.actionability_distribution or local_actionability_distribution,
            },
            "trajectory_analysis": {
                "status": "not_available_without_revision_history",
                "available_now": False,
                "future_signals": [
                    "idea_evolution",
                    "reasoning_continuity",
                    "semantic_drift",
                    "revision_path",
                    "cognitive_consistency",
                ],
                "required_inputs": [
                    "draft_history",
                    "timestamped_revisions",
                    "author_notes_or_outline",
                    "accepted_and_rejected_rewrite_operations",
                ],
            },
            "mitigation_inputs": {
                "rewrite_plan": None,
                "rewrite_constraints": None,
                "rewrite_edit_briefs": None,
                "preservation_inventory": preservation_inventory,
                "human_contribution_contract": human_contract,
                "industry_baseline": industry_baseline,
                "generation_handoff": generation_handoff,
                "rewrite_routing_signals": rewrite_routing_signals,
                "authorship_window_profile": authorship_window_profile,
                "blocker_radar": blocker_radar,
                "repair_units_v2": repair_units_v2,
                "target_segment_ids": [
                    segment["segment_id"]
                    for segment in segments
                    if any(sig.get("rewrite_permission") == "auto" for sig in segment.get("signals", []))
                ],
                "manual_review_segment_ids": [
                    segment["segment_id"]
                    for segment in segments
                    if any(sig.get("rewrite_permission") == "manual" for sig in segment.get("signals", []))
                ],
            },
            "guardrails": {
                "is_authorship_verdict": False,
                "preserve_original_text": True,
                "requires_user_confirmation_for_manual_signals": True,
                "badge_guardrails": badge.get("guardrails") or [],
            },
        }

    result: Dict[str, Any] = {
        "raw_overall_tier": report.raw_overall_tier,
        "adjusted_overall_tier": report.adjusted_overall_tier,
        "overall_tier": report.overall_tier.value,
        "overall_tier_reason": report.overall_tier_reason,
        "tier_derivation": tier_derivation,
        "domain_profile": domain_profile,
        "rewrite_priority_tier": report.rewrite_priority_tier,
        "rewrite_priority_reason": report.rewrite_priority_reason,
        "rewrite_decision": serialized_rewrite_decision or None,
        "actionability_distribution": report.actionability_distribution or local_actionability_distribution,
        "axis_scores": report.axis_scores,
        "reason_codes": report.reason_codes,
        "authorship_evidence": build_authorship_evidence(
            report.authorship_concern_signals,
            false_positives=report.false_positives,
            confidence=report.authorship_concern_confidence,
            strengthen_examples=strengthen_anchor_sentences({"rewrite_edit_briefs": _rewrite_edit_briefs()}),
        ),
        "authorship_concern": {
            "score": report.authorship_concern_score,
            "concern_tier": _concern_tier_from_score(report.authorship_concern_score),
            "confidence": report.authorship_concern_confidence,
            "weak_signal_only": _is_weak_only(report.authorship_concern_signals),
            "signals": report.authorship_concern_signals,
            "available_signal_count": sum(
                1 for v in (report.authorship_concern_signals or {}).values()
                if v is not None
            ),
            "total_signal_count": len(report.authorship_concern_signals or {}),
        },
        "ai_risk_badge": report.ai_risk_badge,
        "paragraph_explanations": report.paragraph_explanations,
        "integrity_layers": _integrity_layers(
            report.ai_risk_badge or {},
            ((report.ai_risk_badge or {}).get("transformation_classification") or {}),
            _transformation_contribution(
                (((report.ai_risk_badge or {}).get("transformation_classification") or {}).get("features") or {}),
                _transformation_signal_rows(
                    (((report.ai_risk_badge or {}).get("transformation_classification") or {}).get("features") or {})
                ),
                (report.ai_risk_badge or {}).get("ai_components") or {},
            ),
        ),
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
        "rewrite_edit_briefs": _rewrite_edit_briefs(),
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
        "full_rewrite_allowed": (
            bool(detect_rewrite_decision.get("full_rewrite_allowed"))
            if detect_rewrite_decision else rewrite_mode == "full"
        ),
    }
    preservation_inventory = _preservation_inventory(report.original_text or "")
    preserved_anchor_terms = [
        anchor["text"]
        for anchor in preservation_inventory.get("anchors", [])
        if anchor.get("kind") in {
            "quote",
            "citation",
            "year",
            "number",
            "acronym",
            "name_or_entity",
            "domain_term",
            "heading",
        }
    ]
    for term in preserved_anchor_terms:
        if term not in result["rewrite_constraints"]["preserve_terms"]:
            result["rewrite_constraints"]["preserve_terms"].append(term)
    result["rewrite_constraints"]["preservation_inventory"] = preservation_inventory

    if report.predictability:
        result["predictability"] = {
            "overall_risk": report.predictability.overall_risk,
            "risk_distribution": report.predictability.risk_distribution,
            "generic_phrases": report.predictability.generic_phrases_found,
            "sentences": [
                {"sentence_id": s.get("sentence_id", ""),
                 "text": s["sentence"][:100], "risk": s["risk_label"],
                 "score": s["risk"], "top10": s["top10_ratio"],
                 "top_predicted_tokens": s.get("top_predicted_tokens", []),
                 "predictable_token_spans": s.get("predictable_token_spans", [])}
                for s in report.predictability.sentences
            ],
            "all_sentences": [
                {"sentence": s["sentence"],
                 "sentence_id": s.get("sentence_id", ""),
                 "predictability_risk": s["risk"],
                 "risk_label": s["risk_label"],
                 "top10_ratio": s["top10_ratio"],
                 "top50_ratio": s["top50_ratio"],
                 "avg_probability": s["avg_probability"],
                 "avg_surprisal": s["avg_surprisal"],
                 "top_predicted_tokens": s.get("top_predicted_tokens", []),
                 "predictable_token_spans": s.get("predictable_token_spans", [])}
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
        structured_segments = structured_sentence_segments(report.original_text or "")
        sentence_map = {}
        for i, s in enumerate(report.predictability.sentences):
            fallback = structured_segments[i] if i < len(structured_segments) else {}
            sentence_id = s.get("sentence_id") or fallback.get("sentence_id") or f"s{i+1:03d}"
            sentence_map[sentence_id] = {
                "paragraph_id": s.get("paragraph_id") or fallback.get("paragraph_id") or "",
                "source_paragraph_id": s.get("source_paragraph_id") or fallback.get("source_paragraph_id") or "",
                "virtual_paragraph_id": s.get("virtual_paragraph_id") or fallback.get("virtual_paragraph_id") or s.get("paragraph_id") or fallback.get("paragraph_id") or "",
                "start_char": s.get("start_char") if s.get("start_char") is not None else fallback.get("start_char", 0),
                "end_char": s.get("end_char") if s.get("end_char") is not None else fallback.get("end_char", 0),
                "text": s.get("sentence") or fallback.get("sentence", ""),
            }
        result["sentence_map"] = sentence_map

    if report.similarity:
        result["similarity"] = {
            "overall_risk": report.similarity.overall_risk,
            "risk_distribution": report.similarity.risk_distribution,
            "matches": report.similarity.matches,
        }

    if report.semantic_shape:
        result["semantic_shape"] = {
            "model_name": report.semantic_shape.model_name,
            "embedding_model_attached": report.semantic_shape.embedding_model_attached,
            "sentence_count": report.semantic_shape.sentence_count,
            "paragraph_count": report.semantic_shape.paragraph_count,
            "adjacent_similarity_mean": report.semantic_shape.adjacent_similarity_mean,
            "adjacent_similarity_std": report.semantic_shape.adjacent_similarity_std,
            "paragraph_similarity_mean": report.semantic_shape.paragraph_similarity_mean,
            "paragraph_similarity_std": report.semantic_shape.paragraph_similarity_std,
            "semantic_uniformity_risk": report.semantic_shape.semantic_uniformity_risk,
            "discourse_regularity_risk": report.semantic_shape.discourse_regularity_risk,
            "semantic_drift_risk": report.semantic_shape.semantic_drift_risk,
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
            "detect_ai_likelihood": report.rewrite.detect_ai_likelihood,
            "detect_writing_quality": report.rewrite.detect_writing_quality,
        }

    scan_intelligence = _scan_intelligence()
    scan_intelligence["mitigation_inputs"]["rewrite_plan"] = result.get("rewrite_plan")
    scan_intelligence["mitigation_inputs"]["rewrite_constraints"] = result.get("rewrite_constraints")
    scan_intelligence["mitigation_inputs"]["rewrite_edit_briefs"] = result.get("rewrite_edit_briefs")
    from detect.mitigation import build_ai_mitigation_plan
    ai_mitigation = build_ai_mitigation_plan(
        scan_intelligence=scan_intelligence,
        ai_risk_badge=report.ai_risk_badge or {},
        rewrite_plan=result.get("rewrite_plan"),
        rewrite_constraints=result.get("rewrite_constraints"),
        rewrite_edit_briefs=result.get("rewrite_edit_briefs"),
    )
    scan_intelligence["mitigation_inputs"]["ai_mitigation_plan"] = ai_mitigation
    industry_baseline = scan_intelligence.get("industry_baseline") or {}
    if isinstance(ai_mitigation, dict):
        ai_mitigation["industry_baseline"] = industry_baseline
    result["ai_mitigation"] = ai_mitigation
    result["industry_baseline"] = industry_baseline
    result["generation_handoff"] = scan_intelligence.get("generation_handoff") or {}
    result["rewrite_routing_signals"] = scan_intelligence.get("rewrite_routing_signals") or {}
    result["authorship_window_profile"] = scan_intelligence.get("authorship_window_profile") or {}
    result["ai_footprint_profile"] = scan_intelligence.get("ai_footprint_profile") or {}
    result["rewrite_target_profile"] = scan_intelligence.get("rewrite_target_profile") or {}
    result["problem_inventory"] = scan_intelligence.get("problem_inventory") or {}
    result["repair_units_v2"] = scan_intelligence.get("repair_units_v2") or {}
    result["scan_intelligence"] = scan_intelligence
    result["highlight_segments"] = scan_intelligence["document"]["segments"]

    result["scan_time_seconds"] = report.scan_time_seconds
    if report.generated_at:
        result["generated_at"] = report.generated_at

    return result
