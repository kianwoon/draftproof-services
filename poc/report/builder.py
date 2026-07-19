"""ReportBuilder — assembles a DraftReport from detect + rewrite results.

Extracted from report.py. Imports the data models, actionability, DeBERTa
metadata, and grounding helpers from their sibling modules.
"""
import logging
import os
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

from detect.scoring import extract_signals, calculate_authorship_concern, estimate_citation_risk
from detect.document_structure import structured_sentence_segments
from detect.layer3_scoring import Layer3Scorer, build_layer3_input_from_text, estimate_external_detector_likelihood, estimate_external_detector_segment_fraction
from detect.external_grouped_scoring import estimate_external_grouped_score
from detect.grounding_diagnosis import diagnose_grounding_gap
from detect.critical_thinking import score_critical_thinking
from detect.submission_risk import score_submission_risk
from detect.policy_risk import score_policy_risk
from detect.transformation import classify_transformation_from_scan
from report.models import (
    Tier,
    TIER_ORDER,
    _RISK_LEVEL_TO_TIER,
    Finding,
    PredictabilitySummary,
    SimilaritySummary,
    SemanticShapeSummary,
    CitationSummary,
    RewriteSummary,
    DraftReport,
)
from report.actionability import determine_actionability
from report.deberta import deberta_authorship_rating as _deberta_authorship_rating
from report.grounding import (
    _topk_calibration_fields_for_summary,
    estimate_in_text_source_grounding_strength as _estimate_in_text_source_grounding_strength,
)
from report.policy_signal import policy_signal_pct_for_llm_patterning


def _lean_gate_scan_active() -> bool:
    """True when an internal rewrite gate scan asked to skip DISPLAY-ONLY DeBERTa passes.

    Set by ``poc.rewrite_v6.runtime.lean_gate_scan()``. Only the display-only DeBERTa sites honor
    this (findings synthesis here, heatmaps in report.py); the AUTHORITATIVE score override must
    always run so the gate number is byte-identical. Fail-open: unset/0 → never skip anything."""
    return str(os.environ.get("DRAFTPROOF_LEAN_GATE_SCAN", "") or "").strip().lower() in ("1", "true", "yes", "on")


def _evidence_weight(strength: str) -> float:
    return {"strong": 1.0, "moderate": 0.6, "weak": 0.2}.get(strength, 0.2)


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
    ew = _evidence_weight
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
        self._deberta_heatmap: dict | None = None  # cached canonical-sentence heatmap (map+tile share it)
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
            if scanner == "consistency" and loc.get("paragraph_id"):
                # Thread the paragraph_id through into metadata — it lives on
                # the detect Finding's `location` (poc/detect/consistency.py),
                # which this loop otherwise only reads for sentence_id matching
                # (Strategies 1-3 above) and never persists onto the converted
                # report Finding. poc/report/consistency_panel.py's per-paragraph
                # rows need it (see poc/report/report.py's caller).
                meta["paragraph_id"] = loc["paragraph_id"]
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
    def _floor_tier_to_badge(adjusted_tier: Tier, badge_tier_value: str) -> Tier:
        """Overall tier must not read LOW when the AI-likelihood badge is AMBER+.

        ``_derive_weighted_tier`` re-derives the overall tier from per-finding tiers
        and a brittle ``pred_risk >= 0.45`` cliff; it never consults the badge's own
        aggregate AI-likelihood (which already folds in generic_assertion, topk, etc.).
        So a highly-generic, zero-grounding document can land LOW while its badge is
        AMBER (observed: test_content11 -> badge AMBER 42%, generic_assertion 90,
        overall LOW). A LOW headline under an amber+ badge contradicts the badge and
        under-warns the user, so floor it to MEDIUM. GREEN badges (genuinely low AI
        signal) may stay LOW. LOW and CLEAN are both lifted — a CLEAN internal tier under
        an amber+ badge under-warns identically (L8); higher tiers are never altered.
        """
        if adjusted_tier in (Tier.LOW, Tier.CLEAN) and str(badge_tier_value or "").upper() in ("AMBER", "ORANGE", "RED"):
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

        # DeBERTa authoritative findings synthesis (gated). Must run BEFORE findings_by_tier so
        # the synthesized findings enter the actionability/repair pipeline. Synthesizes a Finding
        # per high-confidence (>=0.99) sentence so the actionable_summary + repair_units have
        # DeBERTa-native edit targets instead of perplexity 'medium_predictability' findings.
        try:
            from detect.deberta_signal import (  # noqa: E402
                _authoritative_enabled,
                compose_from_sentences as _compose_heatmap_syn,
            )
            # DISPLAY-ONLY: synthesized findings drive the actionability/repair UI, never a rewrite
            # gate decision. Skip the whole DeBERTa forward pass under lean_gate_scan() — the
            # authoritative SCORE override below still runs, so ai_score is byte-identical.
            if _authoritative_enabled() and not _lean_gate_scan_active():
                _canon_syn = [
                    {"sentence_id": it.get("sentence_id"),
                     "paragraph_id": it.get("paragraph_id") or "p001",
                     "text": it.get("sentence", "")}
                    for it in structured_sentence_segments(self._original_text or "")
                ]
                if _canon_syn:
                    _heat_syn = _compose_heatmap_syn(_canon_syn)
                    _heat_rows_syn = {r.get("sentence_id"): r
                                      for r in ((_heat_syn or {}).get("sentence_scores") or [])}
                    for _sent_syn in _canon_syn:
                        _row_syn = _heat_rows_syn.get(_sent_syn.get("sentence_id"))
                        _sc_syn = (_row_syn or {}).get("score")
                        if _sc_syn is None or _sc_syn < 0.99:
                            continue
                        self._findings.append(Finding(
                            tier=Tier.HIGH,
                            category="ai_generation",
                            scanner="deberta",
                            title="high_confidence_ai_sentence",
                            detail=f"Learned classifier >=99% confident this sentence is AI-generated (score {_sc_syn:.2f}).",
                            evidence=_sent_syn.get("text", ""),
                            recommendation="Revoice this sentence in your own words with concrete, verifiable detail from your work.",
                            metadata={"deberta_score": round(_sc_syn, 3), "source": "deberta_authoritative"},
                            finding_id=f"deberta_{_sent_syn.get('sentence_id')}",
                            adjusted_risk="high",
                            sentence_id=_sent_syn.get("sentence_id", ""),
                            signal_category="authorship_risk",
                        ))
        except Exception:
            pass

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
        adjusted_tier = _derive_weighted_tier(self._findings, raw_tier)

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
        # Specificity signal (per-doc), computed once here so it can drive BOTH the
        # specificity axis below and the citation axis's no-citations grounding gate.
        spec_risk = 0.0
        for f in self._findings:
            if f.title == "low_specificity" and f.metadata:
                spec_risk = f.metadata.get("specificity_risk", 0.0)

        # Citation axis. cite_count counts citation PROBLEM findings only. 0 problems
        # is "clear" when the doc actually cites sources (in_text/bib > 0). But an
        # UNSOURCED doc also raises 0 citation findings -- reading that as "clear" (Low)
        # is false reassurance for exactly the ungrounded-claims case DraftProof exists
        # to catch (empirically confirmed: an unsourced generic-claims essay scored
        # Citation risk Low). With no citations detected, gate on grounding: generic /
        # ungrounded prose (specificity concern) -> "review" (actionable: tie claims to
        # sources); concretely grounded prose (specificity clear) -> "clear" (its claims
        # are anchored in real detail even without formal cites). This flags the risk
        # helpfully without over-flagging well-grounded or ESL writing. has_citations
        # uses in_text_count so inline cites (e.g. "(Smith, 2020)") count too.
        cite_count = sum(1 for f in self._findings if f.category == "citation"
                        and f.tier.value not in ("low", "clean"))
        has_citations = bool(self._cite_summary and (
            getattr(self._cite_summary, "in_text_count", 0) > 0
            or getattr(self._cite_summary, "bib_entry_count", 0) > 0))
        if cite_count >= 2:
            axis_scores["citation"] = "attention"
        elif has_citations:
            # The doc demonstrably cites sources: 0-1 flagged citation problems is not
            # enough to call the whole doc's citation risk elevated. A single citation
            # finding is often suppressed from the displayed report, which produced an
            # *unexplained* Medium on a properly-cited doc (empirically confirmed). Only
            # 2+ problems escalate a cited doc.
            axis_scores["citation"] = "clear"
        elif cite_count == 1:
            # No citations detected AND a citation-problem finding fired.
            axis_scores["citation"] = "review"
        elif spec_risk >= 0.30:
            # No citations + generic/ungrounded claims -> nudge to add grounding.
            axis_scores["citation"] = "review"
        else:
            axis_scores["citation"] = "clear"

        # Specificity axis (reuses spec_risk computed above)
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

        # DeBERTa authoritative override (gated, fail-open). When the flag is ON and DeBERTa
        # returns a score, the authoritative ai_likelihood_score + tier come from the learned
        # classifier's >=0.99 high-confidence proportion (the FAIR operating point, ~1-3% ESL FPR),
        # NOT the perplexity Layer3 math. When the flag is off or DeBERTa is unavailable (short
        # text / model error), authoritative_score stays None and everything falls through to the
        # existing perplexity path — zero regression.
        authoritative_score = None
        try:
            from detect.deberta_signal import (  # noqa: E402
                _authoritative_enabled,
                compose_from_sentences as _compose_heatmap,
                headline_from_heatmap as _headline_from_heatmap,
                _derive_ai_tier_deberta as _derive_deb_tier,
            )
            if not _authoritative_enabled():
                raise ImportError  # skip to the except → authoritative_score stays None
            # Compute from the SAME canonical sentences the tile/map use (structured_sentence_segments),
            # not the naive splitter — so the badge and the tile share one sentence source and can
            # never disagree on the proportion (the tile-vs-badge 18.75-vs-14 mismatch).
            _canon = [
                {"sentence_id": it.get("sentence_id"),
                 "paragraph_id": it.get("paragraph_id") or "p001",
                 "text": it.get("sentence", "")}
                for it in structured_sentence_segments(self._original_text or "")
            ]
            if _canon:
                _heat = _compose_heatmap(_canon)
                _headline = _headline_from_heatmap(_heat, _canon)
                if _headline and _headline.get("available"):
                    authoritative_score = {
                        "ai_likelihood_score": round(_headline["signal_pct"] / 100, 4),
                        "tier": _derive_deb_tier(_headline["signal_pct"] / 100),
                        "n_high_confidence": _headline.get("sentences_flagged", 0),
                        "n_scored": _headline.get("sentences_scored", 0),
                        "confidence": _headline.get("confidence", "low"),
                        "model_version": _headline.get("model_version"),
                        "available": True,
                    }
                    # NOTE: DeBERTa findings synthesis moved to BEFORE findings_by_tier (above)
                    # so they enter the actionability/repair pipeline. This block only computes
                    # the authoritative SCORE; the findings are synthesized earlier.
        except Exception:
            authoritative_score = None
        if authoritative_score and authoritative_score.get("available"):
            _deb_score = authoritative_score["ai_likelihood_score"]
            _deb_tier = authoritative_score["tier"]
            authoritative_ai_likelihood = round(_deb_score * 100, 2)
            authoritative_tier = _deb_tier
            # Override the Layer3 result's score so EVERY downstream consumer — the transformation
            # classification, submission risk, badge, reason codes — all see the SAME DeBERTa
            # authoritative score. Without this, the Writing-signal pattern would still read the
            # perplexity score (0.318) while the badge reads DeBERTa (0.1875) — two verdicts on
            # one page. The tier is left to _derive_ai_tier_deberta at badge-build time.
            from dataclasses import replace as _dc_replace
            layer3 = _dc_replace(layer3, ai_likelihood_score=_deb_score)
        else:
            authoritative_ai_likelihood = None
            authoritative_tier = None

        # V7 fused-score tier-authority re-base (kill-switched, fail-open). When
        # DRAFTPROOF_V7_TIER_AUTHORITY is truthy AND a real deep-scan sentence
        # proportion is available, the authoritative ai_likelihood_score + tier
        # become the FUSED score (0.40 * composite + 0.60 * deep_scan_proportion * 100)
        # instead of the composite/DeBERTa-authoritative value alone (see
        # poc/calibration/v7_fused_gate_result.json — GATE PASS). This MUST run
        # here, before submission_risk/policy_risk/the badge dict are built below,
        # so every badge-consuming composer sees the fused values, not stale
        # composite ones (the exact bug this task exists to fix). Mirrors the
        # DeBERTa-authoritative override pattern immediately above: reassigns the
        # same authoritative_ai_likelihood/authoritative_tier variables that
        # already flow into submission_risk, the badge dict, and the tier floor
        # logic below, so no downstream consumer needs to change.
        tier_authority_provenance = None
        # Additive, present ONLY when DRAFTPROOF_V7_TIER_AUTHORITY is enabled
        # (never set when the flag is off, preserving byte-identical flag-off
        # output). Surfaces WHY the fused override did or did not apply, since
        # today the only signal was the silent absence of `tier_authority` —
        # a fail-open branch (e.g. deep scan unavailable) looked identical to
        # "flag was never on" from the badge's shape alone.
        tier_authority_status = None
        # Hoisted: also handed to run_v7_breakdown below as _precomputed_deep_scan
        # so the SAME Modal result serves both the tier override and the breakdown
        # panel — one paid call per scan, never two.
        _deep_scan_for_authority = None
        # Hoisted so it is in scope at the run_v7_breakdown call below even when
        # tier-authority is OFF (stays None -> the bridge falls back to
        # ai_likelihood_score, byte-identical). When ON it holds the RAW composite
        # ai_likelihood BEFORE the DeBERTa fusion, so the category breakdown's
        # "composite" detector input is NOT the already-fused authority score
        # (which would double-fuse with deberta_large). Aligns prod with the
        # offline-tuned weights (capture_signals.py runs with tier-authority UNSET).
        _pre_fusion_composite = None
        # Resolved outside the try/except below so the exception branch still
        # knows the true flag state (an exception inside the try must not be
        # allowed to fabricate "enabled": True when the flag was actually off).
        _tier_authority_flag_on = False
        try:
            from detect_v7.pipeline_bridge import (  # noqa: E402
                is_tier_authority_enabled as _is_tier_authority_enabled,
            )
            _tier_authority_flag_on = _is_tier_authority_enabled()
        except Exception:
            _tier_authority_flag_on = False
        try:
            from detect_v7.config import get_tier_authority_config as _get_tier_authority_config  # noqa: E402
            from detect_v7.pipeline_bridge import (  # noqa: E402
                compute_fused_authority as _compute_fused_authority,
                get_deep_scan_proportion as _get_deep_scan_proportion,
            )
            if _tier_authority_flag_on:
                _pre_fusion_composite = (
                    authoritative_ai_likelihood
                    if authoritative_ai_likelihood is not None
                    else round(layer3.ai_likelihood_score * 100, 2)
                )
                _deep_scan_for_authority = _get_deep_scan_proportion(
                    {"document_text": self._original_text}
                )
                # Below-floor abstain (weights.json tier_authority.below_floor_policy,
                # default "abstain"): a proportion below the reliability floor is
                # INSUFFICIENT EVIDENCE — deberta_signal.py's own "below floor = no
                # verdict, never clean" philosophy — so it must not fuse as
                # exculpatory 0 and re-base the tier (live scan d449aca9: 0.0 below
                # the 0.3 floor dragged a HIGH raw tier to green/0%). Badge stays
                # composite-driven; the deep-scan result still flows to
                # run_v7_breakdown below (panel + uncertainty flags unchanged).
                _below_floor_abstain = (
                    _deep_scan_for_authority is not None
                    and _deep_scan_for_authority.get("below_floor") is True
                    and _get_tier_authority_config()["below_floor_policy"] == "abstain"
                )
                if _below_floor_abstain:
                    logger.warning(
                        "report.builder: V7 tier-authority abstaining — deep-scan "
                        "proportion %s is below the reliability floor; badge stays "
                        "composite-driven (below_floor_policy=abstain).",
                        _deep_scan_for_authority.get("proportion"),
                    )
                    tier_authority_status = {
                        "enabled": True,
                        "applied": False,
                        "reason": "deep_scan_below_floor",
                    }
                elif _deep_scan_for_authority is not None:
                    _fused = _compute_fused_authority(
                        _pre_fusion_composite, _deep_scan_for_authority["proportion"]
                    )
                    tier_authority_provenance = {
                        "source": "v7_fused",
                        "fused_score": _fused["fused_score"],
                        "composite_score": _pre_fusion_composite,
                        "proportion": _deep_scan_for_authority["proportion"],
                        "flag_line": _get_tier_authority_config()["cutoffs"]["amber"],
                    }
                    authoritative_ai_likelihood = _fused["fused_score"]
                    authoritative_tier = _fused["tier"]
                    tier_authority_status = {"enabled": True, "applied": True}
                else:
                    # Fail-open: no deep scan proportion available, badge stays
                    # composite-driven (authoritative_ai_likelihood/tier untouched).
                    logger.warning(
                        "report.builder: V7 tier-authority enabled but no deep-scan "
                        "proportion available; badge stays composite-driven "
                        "(fail-open, non-fatal)."
                    )
                    tier_authority_status = {
                        "enabled": True,
                        "applied": False,
                        "reason": "deep_scan_unavailable",
                    }
        except Exception:
            logger.exception(
                "report.builder: V7 tier-authority override failed; falling back to "
                "composite-driven badge (additive, non-fatal)."
            )
            tier_authority_provenance = None
            if _tier_authority_flag_on:
                logger.warning(
                    "report.builder: V7 tier-authority enabled but override raised "
                    "an exception; badge stays composite-driven (fail-open, non-fatal)."
                )
                tier_authority_status = {"enabled": True, "applied": False, "reason": "error"}
            else:
                tier_authority_status = None

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
        # When the DeBERTa authoritative path is active, record its high-confidence counts in
        # ai_components so downstream consumers (audit, UI) can see the basis of the score. The
        # perplexity component keys are retained for fallback continuity.
        _deb_available = bool(authoritative_score and authoritative_score.get("available"))
        _deb_pct = None
        if _deb_available:
            _deb_pct = round(authoritative_score["ai_likelihood_score"] * 100, 2)
            ai_components["deberta_high_confidence_proportion"] = _deb_pct
            ai_components["deberta_n_high_confidence"] = authoritative_score["n_high_confidence"]
            ai_components["deberta_n_scored"] = authoritative_score["n_scored"]
        # Override the perplexity keys that feed the grounding diagnosis's llm_patterning bucket
        # (and through it, policy_risk). When the V7 tier-authority fusion actually APPLIED, these
        # must track the FUSED authoritative_ai_likelihood (0-100 — the badge's own score), NOT the
        # stale early-DeBERTa read: that stale read is the exact bug where a fused 67 surfaced ~5-9
        # to policy_risk. When fusion did not apply, fall back byte-identically to the legacy
        # early-DeBERTa <0.20 suppression. See report/policy_signal.py for the units-correct rules.
        _tier_authority_applied = bool(
            _tier_authority_flag_on
            and tier_authority_status
            and tier_authority_status.get("applied")
        )
        _policy_pct = policy_signal_pct_for_llm_patterning(
            deberta_available=_deb_available,
            deberta_score_fraction=(_deb_score if _deb_available else None),
            deberta_pct=_deb_pct,
            tier_authority_applied=_tier_authority_applied,
            fused_ai_likelihood_pct=(authoritative_ai_likelihood if _tier_authority_applied else None),
        )
        if _policy_pct is not None:
            ai_components["predictability"] = _policy_pct
            ai_components["topk_pattern"] = _policy_pct
            ai_components["topk_pattern_raw"] = _policy_pct
            ai_components["topk_calibrated_risk"] = _policy_pct

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

        # Additive grounding diagnosis: re-groups the SAME components into a
        # 4-bucket profile and names the primary thing to fix. Additive only --
        # does NOT affect tier, ai_likelihood_score, or any gate (see grounding_diagnosis).
        grounding_diagnosis = diagnose_grounding_gap(
            ai_components=ai_components,
            writing_components=writing_components,
            transformation_features=transformation.features,
            scored_sentence_count=len(_pred_sentences) if _pred_sentences else None,
        )

        # Additive Critical Thinking Control diagnosis: re-groups the SAME
        # grounding signals into student-control dimensions and names the weakest
        # one to work on. Additive only -- does NOT affect tier, ai_likelihood,
        # or any gate (see detect.critical_thinking). The 2 LLM-judged dimensions
        # (alternative_comparison/reflection) attach separately in Phase 2.
        critical_thinking_control = score_critical_thinking(
            grounding_diagnosis=grounding_diagnosis,
            scored_sentence_count=len(_pred_sentences) if _pred_sentences else None,
        )

        # Additive Submission-risk view: re-groups the SAME signals into a 3-layer
        # breakdown (text-pattern / ownership / academic) and leads with whether the
        # work is defensible as the student's own -- demoting the AI-likelihood % to
        # one axis. Additive only -- does NOT affect tier, ai_likelihood, the external
        # estimate, or any gate (see detect.submission_risk). Declaration/policy stay
        # 'unknown -- self-declare' (text cannot reveal them).
        submission_risk = score_submission_risk(
            ai_likelihood_score=(authoritative_ai_likelihood
                                 if authoritative_ai_likelihood is not None
                                 else round(layer3.ai_likelihood_score * 100, 2)),
            ai_tier=(authoritative_tier or layer3.tier.value),
            critical_thinking_control=critical_thinking_control,
            axis_scores=axis_scores,
            scored_sentence_count=len(_pred_sentences) if _pred_sentences else None,
        )

        # Additive policy-risk view: re-weights the SAME grounding + critical-thinking
        # signals into two policy-interpreted scores (AI-allowed vs AI-restricted).
        # Additive only -- does NOT affect tier, ai_likelihood, or any gate. Declaration
        # and process aren't in the text -> confirm-yourself factors (see detect.policy_risk).
        policy_risk = score_policy_risk(
            grounding_diagnosis=grounding_diagnosis,
            critical_thinking_control=critical_thinking_control,
        )

        ai_risk_badge = {
            # AI Generation (Phase 1). When the DeBERTa authoritative flag is on, tier + score
            # come from the learned classifier's >=0.99 proportion (fair operating point); else
            # the perplexity Layer3 math. The signal_source field records which path produced
            # the authoritative numbers so the UI / audit can tell.
            # Canonical LOWERCASE. The authoritative/fused paths already produce
            # lowercase, but the layer3 fallback's Tier enum value is UPPERCASE
            # ("AMBER") — leaking that broke every lowercase-keyed consumer
            # (frontend TIER_TO_BAND chip, render_panels _ACB_TIER_TO_BAND)
            # whenever both authoritative paths failed (observed 2026-07-06 on a
            # 3000-word verification run with Modal + local DeBERTa unavailable).
            "tier": str(authoritative_tier or layer3.tier.value or "").lower(),
            "ai_likelihood_score": (authoritative_ai_likelihood
                                    if authoritative_ai_likelihood is not None
                                    else round(layer3.ai_likelihood_score * 100, 2)),
            "signal_source": ("v7_fused" if tier_authority_provenance is not None
                              else "deberta_authoritative" if authoritative_ai_likelihood is not None
                              else "perplexity_layer3"),
            "external_detector_estimate": external_detector_estimate,
            "grounding_diagnosis": grounding_diagnosis,
            "critical_thinking_control": critical_thinking_control,
            "submission_risk": submission_risk,
            "policy_risk": policy_risk,
            # Authorship rating: DeBERTa-derived when the authoritative flag is on (so the rating's
            # score/label match the DeBERTa badge, not the perplexity Layer3 components); else the
            # perplexity Layer3 rating. Without this override the gauges show ~33% 'Moderate'
            # (perplexity) next to an 18.75% amber DeBERTa badge — two verdicts on one page.
            "authorship_rating": (_deberta_authorship_rating(authoritative_score, layer3)
                                  if authoritative_score and authoritative_score.get("available")
                                  else layer3.authorship_rating),
            "authorship_rating_label": (_deberta_authorship_rating(authoritative_score, layer3).get("label")
                                        if authoritative_score and authoritative_score.get("available")
                                        else layer3.authorship_rating.get("label")),
            "authorship_rating_code": (_deberta_authorship_rating(authoritative_score, layer3).get("code")
                                       if authoritative_score and authoritative_score.get("available")
                                       else layer3.authorship_rating.get("code")),
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
            "verdict_low_confidence": getattr(layer3, "verdict_low_confidence", False),
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

        # V7 tier-authority provenance: present ONLY when the fused override actually
        # applied (flag on + real deep-scan proportion available). Absent otherwise
        # (flag off, or flag on but no deep scan) — see the override block above.
        if tier_authority_provenance is not None:
            ai_risk_badge["tier_authority"] = tier_authority_provenance
        # V7 tier-authority STATUS: present ONLY when DRAFTPROOF_V7_TIER_AUTHORITY
        # is enabled (never when the flag is off — preserves the exact flag-off
        # byte-identical invariant). Unlike `tier_authority` above, this key is
        # set even on a fail-open branch (no deep scan available, or an
        # exception), so a caller can distinguish "flag off" from "flag on but
        # the fused override didn't fire" without guessing from key absence alone.
        if tier_authority_status is not None:
            ai_risk_badge["tier_authority_status"] = tier_authority_status

        from detect.authenticity_dashboard import maybe_attach as _attach_dashboard
        _dash = _attach_dashboard(ai_risk_badge, predictability=None)  # predictability added read-time (Task 7)
        if _dash is not None:
            ai_risk_badge["authenticity_dashboard"] = _dash

        # Additive V7 Authorship Clarity Breakdown (spec 2026-07-03/04). STRICTLY ADDITIVE —
        # never feeds tier/ai_likelihood/badge/any gate; kill-switched off by default; fail-open
        # (run_v7_breakdown catches all exceptions internally and returns None).
        from detect_v7.pipeline_bridge import run_v7_breakdown as _run_v7_breakdown
        # document_text is required for the deep-scan path (sentence-level Modal scoring);
        # ai_risk_badge alone is scores-only and would silently fall back to quick-scan
        # (observed live 2026-07-04: "no document text available on detection_result").
        # Spread into a copy — never mutate the badge that ships in the report.
        _v7_breakdown = _run_v7_breakdown({
            **ai_risk_badge,
            "document_text": self._original_text,
            # Reuse the tier-authority block's Modal result (None when that block
            # didn't run) — get_deep_scan_proportion short-circuits on this key so
            # a scan never pays for two identical deep-scan calls.
            "_precomputed_deep_scan": _deep_scan_for_authority,
            # signal_adapter's criterion-derived signals (specificity_score,
            # sentence_variance, sentence_smoothness, local_style_shift,
            # detector_disagreement) read this key — it is NOT part of the badge,
            # it lives in self._summaries (set at L370-371 from
            # detection_report.criterion_scores). Without it those 5 signals are
            # unconditionally "unavailable". Fail-open preserved: .get() -> None
            # when absent, identical to today's behavior.
            "criterion_scores": self._summaries.get("criterion_scores"),
            # RAW composite ai_likelihood (0-100), BEFORE V7 tier-authority
            # DeBERTa fusion. None when tier-authority is OFF -> the bridge falls
            # back to ai_likelihood_score (byte-identical). Present + numeric only
            # on the fused path, where ai_likelihood_score is ALREADY
            # 0.4*composite + 0.6*deberta; feeding that as the bridge's "composite"
            # would double-fuse with deberta_large. See pipeline_bridge
            # ._extract_calibrated_score.
            "pre_fusion_composite": _pre_fusion_composite,
        })
        if _v7_breakdown is not None:
            ai_risk_badge["authorship_breakdown"] = _v7_breakdown

        # Additive second-opinion DeBERTa AI signal (spec 2026-07-01). STRICTLY ADDITIVE —
        # never feeds tier/ai_likelihood/external/gate; advisory comparison score only.
        # Fail-open: maybe_attach already catches/logs internally and returns available=False;
        # this outer guard is defense-in-depth against an import error so the report never breaks.
        #
        # IMPORTANT: the headline is built from the CANONICAL sentence heatmap (the same
        # per-sentence scores the Signal-highlights map uses), NOT from compose()'s own naive
        # splitter. This guarantees the tile and the map agree on which passages are flagged —
        # two different sentence splitters previously caused the tile to flag passages the map
        # colored differently. The headline here is a PLACEHOLDER — report_to_dict rebuilds it
        # from the SAME _source_segments heatmap the map renders, so the tile and map always agree.
        # compose(text) is used now purely so the badge field exists; report_to_dict overrides it.
        try:
            from detect.deberta_signal import maybe_attach as _attach_deberta  # noqa: E402
            _deberta = _attach_deberta(self._original_text)
            if _deberta is not None:
                ai_risk_badge["ai_signal_deberta"] = _deberta
        except Exception:
            pass

        badge_ai_score = ai_risk_badge.get("ai_likelihood_score", 0.0) / 100
        if ai_val > 0 and badge_ai_score > 0:
            overall_tier_reason = overall_tier_reason.replace(
                f"AI likelihood is {ai_val:.1%}",
                f"AI likelihood is {badge_ai_score:.1%}",
            )

        # Badge floor: the headline tier must not read LOW when the AI-likelihood badge
        # is AMBER+ (the badge aggregates generic_assertion/topk/etc.; the weighted
        # derivation does not). Lifts only LOW -> MEDIUM; never lowers a higher tier.
        # UNDER DEBERTA AUTHORITATIVE: skip the floor entirely. The tier IS the DeBERTa badge
        # tier — no perplexity-finding inflation. A 14% document reads green/low, not medium.
        is_deberta_authoritative = ai_risk_badge.get("signal_source") == "deberta_authoritative"
        _v7_ta = ai_risk_badge.get("tier_authority") if isinstance(ai_risk_badge.get("tier_authority"), dict) else None
        is_v7_fused = ai_risk_badge.get("signal_source") == "v7_fused" and _v7_ta is not None
        if is_deberta_authoritative and authoritative_score and authoritative_score.get("available"):
            # Tier follows DeBERTa directly: green->LOW, amber->MEDIUM, orange->HIGH, red->CRITICAL
            _deb_tier_map = {"green": Tier.LOW, "amber": Tier.MEDIUM,
                             "orange": Tier.HIGH, "red": Tier.CRITICAL}
            adjusted_tier = _deb_tier_map.get(authoritative_tier or "", adjusted_tier)
        elif is_v7_fused:
            # V7 fused authority set the BADGE tier (see ai_risk_badge["tier"] above); the OVERALL tier
            # + reason must follow it too, else the PDF hero reads "Overall tier is HIGH" while the fused
            # badge is LOW. The fused score IS the authoritative tier -> skip the badge floor.
            _fused_tier_map = {"green": Tier.LOW, "amber": Tier.MEDIUM,
                               "orange": Tier.HIGH, "red": Tier.CRITICAL}
            adjusted_tier = _fused_tier_map.get(authoritative_tier or "", adjusted_tier)
        else:
            pre_floor_tier = adjusted_tier
            adjusted_tier = self._floor_tier_to_badge(adjusted_tier, ai_risk_badge.get("tier"))
            if adjusted_tier != pre_floor_tier:
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
        # When the DeBERTa authoritative path produced the score, the reason text/codes must
        # reflect THAT signal (high-confidence sentence counts), not the perplexity findings
        # (pred_meta_risk / predictability_unconfirmed), so the explanation matches the score.
        is_deberta_authoritative = ai_risk_badge.get("signal_source") == "deberta_authoritative"
        if is_deberta_authoritative and authoritative_score and authoritative_score.get("available"):
            n_hc = authoritative_score["n_high_confidence"]
            n_sc = authoritative_score["n_scored"]
            overall_tier_reason = (
                f"Overall tier is {adjusted_tier.value.upper()}: the learned classifier is "
                f">=99% confident that {n_hc} of {n_sc} sentences are AI-generated "
                f"({badge_ai_score:.0%} high-confidence). The perplexity-family signal is "
                f"retained as a secondary read but does not set this score."
            )
            if not has_high_critical:
                reason_codes.append("no_high_or_critical_findings")
            if axis_scores.get("domain_grounding") == "strong":
                reason_codes.append("strong_domain_grounding")
            if len(review_only) > len(auto_fixable) + len(manual_required):
                reason_codes.append("mostly_review_only_findings")
            if n_hc == 0:
                reason_codes.append("deberta_no_high_confidence_sentences")
            if not rewrite_is_recommended:
                reason_codes.append("rewrite_not_recommended")
        else:
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

        # V7 fused authority: the reason must name the FUSED re-base (composite + deep-scan), not the
        # stale layer3 "Overall tier is HIGH" text — else the PDF hero (the only surface that renders
        # overall_tier_reason) contradicts the fused LOW badge. Reason_codes keep the generic set above.
        if is_v7_fused:
            _comp = _v7_ta.get("composite_score")
            _deep = round((_v7_ta.get("proportion") or 0) * 100)
            _comp_str = f"composite {round(_comp)}% + " if isinstance(_comp, (int, float)) else ""
            overall_tier_reason = (
                f"Overall tier is {adjusted_tier.value.upper()}: the fused AI-likelihood is "
                f"{badge_ai_score:.0%} ({_comp_str}deep-scan {_deep}% sentence-level). This re-bases the "
                f"tier to the calibrated fused score; the perplexity-family signals are a secondary read."
            )

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
            deberta_heatmap=getattr(self, "_deberta_heatmap", None),
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
