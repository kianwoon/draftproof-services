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
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

from detect.scoring import extract_signals, calculate_authorship_concern, estimate_citation_risk
from report.authorship_evidence import build_authorship_evidence, strengthen_anchor_sentences
from detect.authorship_windows import build_ai_footprint_profile, build_authorship_window_profile
from detect.document_structure import structured_sentence_segments
from detect.repair_units import build_repair_units_v2
from detect.rewrite_targets import build_problem_inventory, build_rewrite_target_profile
from detect.layer3_scoring import Layer3Scorer, build_layer3_input_from_text, estimate_external_detector_likelihood, estimate_external_detector_segment_fraction
from detect.external_grouped_scoring import estimate_external_grouped_score
from detect.grounding_diagnosis import diagnose_grounding_gap
from detect.critical_thinking import score_critical_thinking, score_critical_thinking_per_paragraph
from detect.submission_risk import score_submission_risk
from detect.policy_risk import score_policy_risk
from detect.transformation import (
    TRANSFORMATION_SIGNAL_METADATA,
    classify_transformation_from_scan,
    transformation_signal_metadata,
)
from detect.topk_calibration import calibrate_topk_risk
from detect.turnitin_like import turnitin_like_ai_profile
from report.contribution import contribution_pair_int

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Data models, tiers, DeBERTa metadata, actionability, and grounding helpers live in sibling
# modules; re-imported here for backward compatibility (external code imports these names from
# report.report). report_to_dict's closures reference the underscore-prefixed aliases below.
from report.models import (
    Tier,
    TIER_ORDER,
    TIER_ICON,
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
from report.deberta import (
    DEBERTA_HEAT_COLORS as _DEBERTA_HEAT_COLORS,
    DEBERTA_HEAT_TIERS as _DEBERTA_HEAT_TIERS,
    DEBERTA_HEAT_LABELS as _DEBERTA_HEAT_LABELS,
    DEBERTA_HEAT_DESCRIPTIONS as _DEBERTA_HEAT_DESCRIPTIONS,
    DEBERTA_HEAT_RECOMMENDATIONS as _DEBERTA_HEAT_RECOMMENDATIONS,
    DEBERTA_HEAT_READER_SUMMARY as _DEBERTA_HEAT_READER_SUMMARY,
    deberta_authorship_rating as _deberta_authorship_rating,
)
from report.grounding import (
    _topk_calibration_fields_for_summary,
    concern_tier_from_score as _concern_tier_from_score,
    is_weak_only as _is_weak_only,
    estimate_in_text_source_grounding_strength as _estimate_in_text_source_grounding_strength,
)


from report.builder import ReportBuilder


# ── Report to dict ──────────────────────────────────────────────────
# Authorship-concern / grounding helpers (_concern_tier_from_score,
# _is_weak_only, _estimate_in_text_source_grounding_strength) are imported
# above from report.grounding.


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
            # Anchored examples: the vaguest sentences to ground. Threaded through the
            # finding metadata (the report rebuilds `evidence`, dropping the criterion's).
            examples = [s for s in (metrics.get("flagged_excerpts") or [])
                        if isinstance(s, str) and s.strip()][:3]
            if not examples and isinstance(f.evidence, str) and f.evidence.strip():
                examples = [f.evidence.strip()]
            # allow-hardcode: display-only summary template (the metric labels rendered to
            # the user), not detection logic — never matched against document text.
            summary = (f"{int(metrics.get('word_count', 0))} words, "
                       f"{int(metrics.get('named_entities', 0))} named entities, "
                       f"{int(metrics.get('numbers', 0))} numbers, "
                       f"{int(metrics.get('dates', 0))} dates, "
                       f"{matched_domain_term_count} matched domain terms "
                       f"({weighted_domain_term_count} weighted)")
            if examples:
                summary += f". Vaguest sentence to ground: “{examples[0][:140]}”"
            result_evidence = {
                "type": "document_level",
                "summary": summary,
                "affected_span": "full_document",
                "example_sentences": examples,
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
        # Closed-class function-word stoplist only (NO-HARDCODE): the overfit content-word tail
        # (another/complex/issue/layer/memory/process/working/elements/handled…) and the
        # education-role allowlist (learner/student/teacher/educator) were removed -- both were
        # domain/essay-specific content. Key-term anchors are now extracted structurally:
        # lowercase, non-stopword, sufficiently long tokens.
        stopwords = {
            "about", "after", "again", "also", "being", "because", "before",
            "between", "could", "does", "every", "from", "have", "into",
            "many", "more", "most", "need", "needs", "only", "same",
            "should", "simply", "some", "than", "that", "their", "them",
            "there", "these", "they", "this", "those", "time", "when",
            "where", "while", "with", "without", "would",
        }
        text = _re.sub(r"\([^)]*\d{4}[^)]*\)", " ", text or "")
        terms = []
        for word in _re.findall(r"\b[A-Za-z][A-Za-z-]{3,}\b", text):
            lower = word.lower()
            if word.isupper() or word[0].isupper():   # skip acronyms + proper nouns (structural)
                continue
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
        if top["title"] == "high_confidence_ai_sentence":
            overall_action = "ai_voice_revision"
        elif top["title"] == "low_specificity":
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
        if af["title"] == "high_confidence_ai_sentence":
            primary_goals.append(f"Revoice high-confidence AI sentence ({af['finding_id']})")
        elif af["action"] == "add_concrete_domain_context":
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
    elif any("revoice" in g.lower() for g in primary_goals):
        primary_action = "revoice_ai_sentences"
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

    # DeBERTa heatmap → the SOLE signal source for Signal highlights (per Turnitin: learned
    # classifier replaces the perplexity family). Colors the map AND rebuilds the tile headline
    # from the same per-sentence scores, so the two sections can never disagree.
    # Provenance of the LAST _compute_deberta_heatmap() call, so the document dict can tell the
    # frontend which detector actually produced the highlight map (fakespot vs deep-scan), for
    # the legend copy — set inside the closure below, read after the call in _scan_intelligence.
    heatmap_source = {"value": "fakespot"}

    def _compute_deberta_heatmap() -> list:
        """Return the per-sentence heatmap that drives Signal-highlights + fix-first, computed
        from _source_segments(complete=True) — the EXACT sentence list the map renders. This is
        the single source of truth: the map colors from it AND the tile headline is rebuilt from
        it, so the two sections can never disagree on which passages are flagged. (build()'s
        structured_sentence_segments cache is deliberately NOT used: it splits sentences
        differently than _source_segments, which caused the tile-vs-map flag mismatch in
        production. See _sync_deberta_headline_from_heatmap.)

        SOURCE SWITCH (2026-07-04): when the V7 deep-scan detector is enabled
        (detect_v7.pipeline_bridge.is_deep_scan_enabled()) and actually returns a usable
        heatmap, that becomes the source — so the whole panel (headline fused score +
        highlights + fix-first) reads from ONE consistent model. Fail-open: disabled/
        unavailable/error falls back to the existing ai_signal_deberta fakespot heatmap,
        byte-unchanged. The segment-signal key stays "ai_signal_deberta" either way (the
        frontend filters on that key and does not care about provenance) — this keeps
        SignalHighlights/FixFirstChecklist and the DebertaSignal second-opinion tile's
        consumer (badge.ai_signal_deberta, a SEPARATE field never touched here) both
        working unchanged."""
        sens = []
        for item in _source_segments(complete=True):
            sens.append({
                "sentence_id": item.get("sentence_id"),
                "paragraph_id": item.get("paragraph_id") or "p001",
                "text": item.get("sentence", ""),
            })
        if not sens:
            return []

        try:
            from detect_v7.pipeline_bridge import is_deep_scan_enabled  # noqa: E402
            from detect_v7.deep_scan_heatmap import compose_deep_scan_heatmap  # noqa: E402
            if is_deep_scan_enabled():
                deep_result = compose_deep_scan_heatmap(sens)
                if deep_result and deep_result.get("available"):
                    heatmap_source["value"] = "deep_scan"
                    return deep_result.get("sentence_scores") or []
        except Exception:
            logger.exception(
                "report.report: deep-scan heatmap failed; falling back to ai_signal_deberta "
                "fakespot heatmap (additive, non-fatal)."
            )

        try:
            from detect.deberta_signal import compose_from_sentences  # noqa: E402
        except Exception:
            return []
        result = compose_from_sentences(sens)
        if not result or not result.get("available"):
            return []
        return result.get("sentence_scores") or []

    def _sync_deberta_headline_from_heatmap(heatmap_rows: list) -> None:
        """Rebuild the badge's ai_signal_deberta headline from the SAME heatmap the map uses, so
        the tile's flagged_passages / signal_pct are derived from _source_segments — identical
        sentence boundaries to the map. Without this, the tile would carry build()'s headline
        (from structured_sentence_segments, a different split) and disagree with the map."""
        if not heatmap_rows:
            return
        try:
            from detect.deberta_signal import headline_from_heatmap  # noqa: E402
        except Exception:
            return
        # Reconstruct the canonical sentence list the heatmap was built from, so
        # headline_from_heatmap can recover each flagged passage's text.
        sens = [
            {"sentence_id": r.get("sentence_id"),
             "paragraph_id": r.get("paragraph_id"),
             "text": next((it.get("sentence", "") for it in _source_segments(complete=True)
                           if it.get("sentence_id") == r.get("sentence_id")), "")}
            for r in heatmap_rows
        ]
        heat = {"available": True, "sentence_scores": heatmap_rows, "model_version": "deberta_signal_v2"}
        new_headline = headline_from_heatmap(heat, sens)
        badge = getattr(report, "ai_risk_badge", None) or {}
        if new_headline and badge.get("ai_signal_deberta") is not None:
            report.ai_risk_badge["ai_signal_deberta"] = new_headline

    def _deberta_primary_signal(heat_row: dict) -> dict | None:
        """Build a primary_signal dict (the shape _segment_signal returns) from a DeBERTa
        heatmap row. This is the SOLE signal on a segment — no perplexity secondaries.

        Carries band-native guidance (recommendation + reader_summary) so the issue-card body
        is coherent with the DeBERTa highlight color — both come from the learned classifier."""
        band = heat_row.get("band") or "clean"
        score = heat_row.get("score")
        return {
            "finding_id": f"deberta_{heat_row.get('sentence_id')}",
            "key": "ai_signal_deberta",
            "label": _DEBERTA_HEAT_LABELS.get(band, "AI signal"),
            "description": _DEBERTA_HEAT_DESCRIPTIONS.get(band, ""),
            "color": _DEBERTA_HEAT_COLORS.get(band, "#94a3b8"),
            "category": "ai_detection",
            "scanner": "deberta",
            "title": f"deberta_{band}",
            "tier": _DEBERTA_HEAT_TIERS.get(band, ""),
            "score": round(float(score) * 100) if score is not None else 0,
            "actionability": "review",
            "rewrite_permission": "advisory",
            "recommendation": _DEBERTA_HEAT_RECOMMENDATIONS.get(band),
            "reader_summary": _DEBERTA_HEAT_READER_SUMMARY.get(band),
        }

    def _document_segments(complete: bool = False) -> list:
        segments = []
        deberta_heat = _compute_deberta_heatmap()
        _sync_deberta_headline_from_heatmap(deberta_heat)
        deberta_by_sid = {row["sentence_id"]: row for row in deberta_heat} if deberta_heat else {}
        for item in _source_segments(complete):
            sid = item.get("sentence_id")
            # DeBERTa is the sole signal here (no perplexity secondaries — see _deberta_primary_signal).
            # A clean band (score None or < 0.50) → human-like → no signal, plain segment.
            heat_row = deberta_by_sid.get(sid)
            heat_band = (heat_row or {}).get("band")
            if heat_row is not None and heat_band and heat_band != "clean":
                primary = _deberta_primary_signal(heat_row)
                segment_signals = [primary] if primary else []
            else:
                primary = None
                segment_signals = []

            segment = {
                "segment_id": sid,
                "type": "sentence",
                "sentence_id": sid,
                "paragraph_id": item.get("paragraph_id") or "",
                "sentence_index": item.get("sentence_index"),
                "start_char": item.get("start_char", 0),
                "end_char": item.get("end_char", 0),
                "text": item.get("sentence", ""),
                "signals": segment_signals,
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
                "flagged_sentences": [],  # DeBERTa-native per-sentence evidence (no perplexity)
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
                # Collect DeBERTa-native per-sentence evidence for the issue card. This is the
                # ONLY guidance the card shows — band + score + the actual sentence text — so the
                # advice is coherent with the highlight color (both from the learned classifier).
                if key == "ai_signal_deberta":
                    entry["flagged_sentences"].append({
                        "sentence_id": segment.get("sentence_id"),
                        "text": segment.get("text") or "",
                        "score": signal.get("score", 0),
                        "band": signal.get("title", "").replace("deberta_", ""),
                        "tier": signal.get("tier", ""),
                        "color": signal.get("color"),
                        "recommendation": signal.get("recommendation"),
                        "reader_summary": signal.get("reader_summary"),
                    })
        rows = []
        for entry in paragraphs.values():
            signals = sorted(entry.pop("signals").values(), key=lambda item: item["score"], reverse=True)
            flagged = entry.pop("flagged_sentences", [])
            # Sort evidence by score desc; the card surfaces the strongest sentences first.
            flagged.sort(key=lambda s: s.get("score", 0), reverse=True)
            text_parts = entry.pop("text_parts", [])
            entry["text"] = " ".join(part.strip() for part in text_parts if part and part.strip())
            entry["top_signals"] = signals[:3]
            entry["primary_signal"] = signals[0] if signals else None
            entry["flagged_sentences"] = flagged
            # Paragraph-dominant guidance: the strongest flagged sentence's band drives the
            # reader_summary + recommendation shown at the top of the card.
            if flagged:
                top = flagged[0]
                entry["reader_summary"] = top.get("reader_summary")
                entry["recommendation"] = top.get("recommendation")
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
        # NO-HARDCODE: entity filtering is purely STRUCTURAL -- no baked vocabulary. A capitalised
        # token is kept as a named entity only if it is genuinely proper-noun-shaped: a single token
        # must be mixed-case (e.g. "iPhone") or acronym-like (e.g. "NASA"); a plain capitalised common
        # word (sentence-opener, generic noun) fails that gate. Leading determiners/prepositions are
        # stripped and trailing function words rejected. The previous `stop_entities` set and a
        # `{teacher,student,learner}` guard baked domain/common words; both were redundant with these
        # structural gates (verified: anchor output is byte-identical without them).
        for match in _re.finditer(entity_pattern, text):
            entity = match.group(0).strip()
            entity = _re.sub(r"^(?:At|By|In|For|With|From|This|The)\s+", "", entity).strip()
            if len(entity) < 3:
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
        # NO-HARDCODE: purely STRUCTURAL (positional) role -- no heading vocabulary, no
        # domain/subject words. The previous version branched on a baked heading->role keyword
        # ladder ("lost"/"challenge"/"show"/"demonstrat"/"adjustment"/"classroom"/"standard"/
        # "access") overfit to one education essay; a business/science/legal doc fell through to
        # "development" for everything. `role` is descriptive handoff metadata -- no downstream
        # logic branches on its value -- so a position-based role is both agnostic and sufficient.
        if total <= 1:
            return "document_body"
        if index <= 1:
            return "opening_context"
        if index >= total:
            return "closing_synthesis"
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

        # Per-paragraph Critical Thinking tags (deterministic, additive). Map each
        # paragraph's flagged findings to its weakest lead-eligible control dimension
        # so the report UI (and, later, the rewrite) can act on the specific thinking
        # gap. Mutates the additive critical_thinking_control object; never gates
        # anything. Fail-open: a failure here must not break report serialization.
        try:
            _sid_to_pid = {
                str(seg.get("sentence_id")): str(seg.get("paragraph_id") or "")
                for seg in segments if seg.get("paragraph_id")
            }
            _ct_findings_by_paragraph: Dict[str, list] = {}
            for _sid, _flist in findings_by_sentence.items():
                _pid = _sid_to_pid.get(str(_sid))
                if not _pid:
                    continue
                for _f in _flist:
                    _ct_findings_by_paragraph.setdefault(_pid, []).append({
                        # Finding.title carries the precise detector finding_type
                        # (report.py sets title=f.finding_type); signal_category is
                        # the coarse fallback.
                        "finding_type": getattr(_f, "title", "") or "",
                        "signal_category": getattr(_f, "signal_category", "") or "",
                        "score": _finding_score(_f),
                    })
            _ctc = (report.ai_risk_badge or {}).get("critical_thinking_control")
            if isinstance(_ctc, dict):
                _ctc["paragraphs"] = score_critical_thinking_per_paragraph(_ct_findings_by_paragraph)
        except Exception:
            pass
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
                # Which detector produced the Signal-highlights/fix-first per-sentence scores on
                # THIS report: "deep_scan" (V7 Modal detector, same one the panel headlines) or
                # "fakespot" (fail-open default — deep scan off/unavailable). Lets the frontend
                # legend (SignalHighlights.jsx) label the map's actual source instead of always
                # saying "second opinion" when it is really the primary deep-scan detector.
                "signal_highlight_source": heatmap_source["value"],
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
    # NO-HARDCODE: these describe the SHAPE of allowed concrete additions only -- never a
    # subject/domain. The domain content comes from the document's own `domain_terms` above, not
    # from baked examples. (Previously injected "teacher/student interaction details", which biased
    # every rewrite -- business, legal, science -- toward an education frame.)
    specificity_guidance.extend([
        "step-by-step description of a process already implied by the text",
        "concrete interactions, actors, or scenarios drawn from the document's own context",
        "specific terms and vocabulary already present in the text",
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
