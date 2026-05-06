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
from detect.layer3_scoring import Layer3Scorer, build_layer3_input_from_text
from detect.transformation import classify_transformation_from_scan

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
        for f in self._findings:
            if f.title == "low_specificity" and f.metadata:
                dg_level = f.metadata.get("domain_grounding_level", "weak")
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

        # Build Layer3Input from scanner outputs + text-derived signals
        layer3_input = build_layer3_input_from_text(
            self._original_text or "",
            predictability=sig.get("predictability", 0.0) or 0.0,
            topk_pattern=sig.get("topk_pattern_risk", 0.0) or 0.0,
            generic_phrase_density=sig.get("genericity", 0.0) or 0.0,
            # broad_claim_risk & unsupported_claim_risk: auto-computed from text
            citation_weakness_risk=cite_risk,
            # source_grounding: cap at 0.30 when no actual sources/bibliography exist
            source_grounding_strength=min(
                sig.get("source_grounding", 0.0) or 0.0,
                0.30 if not (self._cite_summary and self._cite_summary.bib_entry_count > 0) else 1.0,
            ),
            domain_grounding_strength=min(dg_idx, 0.30),
            human_provenance_positive=False,
            verified_ai_provenance=False,
        )

        layer3 = Layer3Scorer().score(layer3_input)
        transformation = classify_transformation_from_scan(
            layer3_input,
            layer3,
            similarity_summary=self._sim_summary,
        )

        ai_risk_badge = {
            # AI Generation (Phase 1)
            "tier": layer3.tier.value,
            "ai_likelihood_score": round(layer3.ai_likelihood_score * 100, 2),
            "authorship_rating": layer3.authorship_rating,
            "authorship_rating_label": layer3.authorship_rating.get("label"),
            "authorship_rating_code": layer3.authorship_rating.get("code"),
            "ai_cluster_boost": round(layer3.ai_cluster_boost * 100, 2) if layer3.ai_cluster_boost else 0,
            "ai_cluster_name": layer3.ai_cluster_name,
            "ai_components": {k: round(v * 100, 2) for k, v in layer3.ai_phase.components.items()},

            # Writing Quality (Phase 2)
            "writing_quality_tier": layer3.writing_quality_tier.value,
            "writing_quality_score": round(layer3.writing_quality_score * 100, 2),
            "writing_components": {k: round(v * 100, 2) for k, v in layer3.writing_phase.components.items()},

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
            axis_scores=axis_scores,
            reason_codes=reason_codes,
            authorship_concern_score=concern["score"],
            authorship_concern_confidence=concern["confidence_label"],
            authorship_concern_signals=concern["signals"],
            ai_risk_badge=ai_risk_badge,
        )

        # Inject detect scan scores into rewrite summary so rewrite report
        # shows the same risk scores the user sees in the scan report.
        if self._rewrite_summary:
            self._rewrite_summary.detect_ai_likelihood = ai_risk_badge.get("ai_likelihood_score", 0.0)
            self._rewrite_summary.detect_writing_quality = ai_risk_badge.get("writing_quality_score", 0.0)


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
            "detect_ai_likelihood": report.rewrite.detect_ai_likelihood,
            "detect_writing_quality": report.rewrite.detect_writing_quality,
        }

    result["scan_time_seconds"] = report.scan_time_seconds
    if report.generated_at:
        result["generated_at"] = report.generated_at

    return result
