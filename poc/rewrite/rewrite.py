"""DraftProof Rewrite — quality-first multi-signal rewrite engine.

Accepts content + detect findings, produces rewritten text with reduced risk
and a rendered report.

Architecture:
  Detect Findings → FixabilityRouter → RewritePlanner → Candidate Rewrite
  → Guards (drift + voice + protected) → Scorer → Transactional Apply
  → Changed-region Re-detect → Final Full Detect → Report

The system behaves like:
  Can I safely fix this?
    ├─ yes → make smallest safe edit
    ├─ partly → revise only with source/context support
    ├─ no → create manual action
    └─ unsafe → preserve original and explain why
"""

import sys
import os
import json
import time
import subprocess
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional, Dict, Any, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rewriter"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "predictability"))

from detect.base import DetectResult, Finding
from detect.run import DetectionRunner
from rewriter import (
    PredictabilityScanner,
    compute_metrics,
    multi_pass_rewrite,
    MultiPassResult,
    RewriteResult,
    rewrite_text,
)
from style_analyzer import StyleAnalyzer, StyleProfile
from report import ReportBuilder, render_report, render_markdown, report_to_dict
from rewrite.config import (
    RewriteConfig, RewriteBudget, should_continue, LoopDecision,
    RewriteOutcome, RewriteSurface, FloorReason, classify_floor,
    compute_rewrite_surface,
)
from rewrite.planner import (
    RewritePlanner, RewritePlan, RewriteAction,
    route_finding, FINDING_ROUTING, EDIT_RADIUS,
    FIXABILITY_AUTO, FIXABILITY_PARTIAL, FIXABILITY_MANUAL, FIXABILITY_PROTECTED,
)
from rewrite.deterministic import run_deterministic
from rewrite.guards import (
    detect_protected_spans, check_semantic_drift, DriftCheck,
    RegressionMemory, mask_protected_spans,
    protected_spans_preserved, affected_region, transactional_apply, TransactionResult,
    PredictabilityGuard, _extract_named_entities,
)
from rewrite.scorer import (
    weighted_finding_score, weighted_rewritable_risk,
    score_candidate, best_candidate, CandidateScore,
    FIXABILITY_WEIGHT,
)
from rewrite.voice import VoiceGuard, VoiceProfile, analyze_voice
from rewrite.mitigation import build_mitigation_plan
from llm.gateway import LLMGateway, LLMConfig

logger = logging.getLogger(__name__)

REWRITE_RUNTIME_VERSION = "context-aware-rewrite-v3"


def _metrics_from_detect(detect_report, text: str):
    """Extract predictability metrics from an already-run DetectionReport.

    Avoids running compute_metrics() which would trigger another
    full predictability scan (~28s). Falls back to compute_metrics
    only if the predictability scanner didn't produce results.
    """
    # Resolve PassMetrics type via MultiPassResult's field annotations.
    # This avoids fragile direct imports from rewriter.rewriter which
    # may not resolve correctly in all import contexts.
    _PM = MultiPassResult.__dataclass_fields__["original_metrics"].type

    for sr in detect_report.scanner_results:
        if sr.scanner == "predictability" and sr.raw:
            raw = sr.raw
            # Handle dict raw (from targeted rescan / report JSON / live scanner scan_text())
            if isinstance(raw, dict):
                # Prefer all_sentences (full data with sentence text) over short-form sentences
                sents = raw.get("all_sentences") or raw.get("sentences", [])
                if sents:
                    overall_risk = raw.get("overall_risk", 0.5)
                    # Normalize all items to canonical dict format used by compute_metrics:
                    # {label, risk, top10_ratio, surprisal, sentence}
                    normalized = []
                    t10s = []
                    for i, s in enumerate(sents):
                        if isinstance(s, dict):
                            # Try all possible key names (report JSON vs asdict format)
                            d = {
                                "index": i,
                                "label": s.get("label") or s.get("risk_label", "low"),
                                "risk": s.get("risk") or s.get("predictability_risk", 0),
                                "top10_ratio": s.get("top10_ratio") or s.get("top_10_ratio") or s.get("top10", 0),
                                "surprisal": s.get("surprisal") or s.get("avg_surprisal", 0),
                                "sentence": s.get("sentence") or s.get("text", ""),
                            }
                        else:
                            # SentenceResult object
                            d = {
                                "index": i,
                                "label": getattr(s, "risk_label", "low"),
                                "risk": getattr(s, "predictability_risk", 0),
                                "top10_ratio": getattr(s, "top_10_ratio", 0),
                                "surprisal": getattr(s, "avg_surprisal", 0),
                                "sentence": getattr(s, "sentence", ""),
                            }
                        normalized.append(d)
                        t10_val = d["top10_ratio"]
                        if isinstance(t10_val, (int, float)):
                            t10s.append(t10_val)
                    top10 = sum(t10s) / max(len(t10s), 1) if t10s else 0.0
                    return _PM(
                        pass_number=0, text=text, risk=overall_risk,
                        top10_ratio=top10, surprisal=0.0, sentence_details=normalized,
                    )
            # Handle object raw (from live scanner)
            elif hasattr(raw, "sentence_details"):
                return _PM(
                    pass_number=0, text=text,
                    risk=raw.overall_risk if hasattr(raw, "overall_risk") else 0.5,
                    top10_ratio=raw.top10_ratio if hasattr(raw, "top10_ratio") else 0.0,
                    surprisal=0.0, sentence_details=raw.sentence_details,
                )
    # Fallback: run compute_metrics only if detect didn't include predictability
    return compute_metrics(text, PredictabilityScanner())


# ── AI-only finding filter ───────────────────────────────────────────

def _is_ai_finding(f: Finding) -> bool:
    """Check if a Finding came from the ai_generation scanner."""
    meta = f.metadata or {}
    return (
        meta.get("scanner") == "ai_generation"
        or meta.get("category") == "ai_generation"
    )


def _filter_ai_findings(detect_results: List[DetectResult], target_severity: str = "medium") -> List[DetectResult]:
    """Keep only the target severity findings from the ai_generation scanner.

    Returns new DetectResult list with only medium AI findings.
    Non-AI scanners (predictability, similarity, citation) are dropped entirely.
    LOW and info findings are skipped — not worth the LLM cost to rewrite.
    HIGH/CRITICAL findings are also skipped because they need user review,
    evidence, or structure work instead of automatic sentence rewriting.
    """
    filtered = []
    for dr in detect_results:
        if dr.scanner == "ai_generation":
            ai_findings = dr.findings
        else:
            ai_findings = [f for f in dr.findings if _is_ai_finding(f)]
        ai_findings = [f for f in ai_findings if f.risk_level == target_severity]
        if not ai_findings:
            continue
        filtered.append(DetectResult(
            scanner="ai_generation",
            overall_risk=dr.overall_risk,
            confidence=dr.confidence,
            confidence_reason=dr.confidence_reason,
            risk_distribution=dr.risk_distribution,
            findings=ai_findings,
            policy_message=dr.policy_message,
            raw=dr.raw,
            feature_summary=dr.feature_summary or {},
        ))
    return filtered


def _filter_ai_guided_findings(
    detect_results: List[DetectResult],
    rewrite_context: Optional[Any] = None,
    target_severity: str = "medium",
) -> List[DetectResult]:
    """AI rewrite target set, expanded with contributing sentence signals.

    The AI-generation scanner often reports document-level summary findings
    while the concrete editable evidence lives in the predictability scanner.
    When the badge says predictability/top-k is a meaningful AI driver, include
    medium predictability sentences as rewrite targets. High findings remain
    manual/review guidance because automatic rewriting has been too regressive.
    """
    raw_json = getattr(rewrite_context, "raw_json", None) if rewrite_context else None
    auto_ids = _rewrite_plan_auto_ids(raw_json)

    filtered = _filter_ai_findings(detect_results, target_severity=target_severity)
    if auto_ids:
        filtered = _filter_detect_results_by_finding_ids(filtered, auto_ids)

    ai_components = ((raw_json or {}).get("ai_risk_badge") or {}).get("ai_components") or {}
    use_predictability = (
        ai_components.get("predictability", 0) >= 40
        or ai_components.get("topk_pattern", 0) >= 50
    )
    if not use_predictability:
        return filtered

    seen = {
        ((f.metadata or {}).get("finding_id"), f.finding_type, f.evidence)
        for dr in filtered for f in dr.findings
    }
    supporting_types = {
        "medium_predictability",
        "low_surprisal_pattern",
        "generic_formulaic_language",
        "formulaic_sentence",
        "generic_phrase",
    }
    for dr in detect_results:
        if dr.scanner not in {"predictability", "ai_generation"}:
            continue
        support = []
        for f in dr.findings:
            key = ((f.metadata or {}).get("finding_id"), f.finding_type, f.evidence)
            if key in seen:
                continue
            finding_id = (f.metadata or {}).get("finding_id")
            if auto_ids and finding_id not in auto_ids:
                continue
            if (
                f.finding_type in supporting_types
                and f.risk_level == target_severity
                and f.evidence
            ):
                support.append(f)
                seen.add(key)
        if not support:
            continue
        filtered.append(DetectResult(
            scanner=dr.scanner,
            overall_risk=dr.overall_risk,
            confidence=dr.confidence,
            confidence_reason=dr.confidence_reason,
            risk_distribution=dr.risk_distribution,
            findings=support,
            policy_message=dr.policy_message,
            raw=dr.raw,
            feature_summary=dr.feature_summary or {},
        ))
    return filtered


def _rewrite_plan_auto_ids(raw_json: Optional[dict]) -> set:
    """IDs that the scan phase explicitly allowed for automatic rewrite."""
    if not isinstance(raw_json, dict):
        return set()
    plan = raw_json.get("rewrite_plan") or {}
    ids = set()
    for item in plan.get("auto_fixable") or []:
        if isinstance(item, dict) and item.get("finding_id"):
            ids.add(item["finding_id"])
    return ids


def _filter_detect_results_by_finding_ids(
    detect_results: List[DetectResult],
    allowed_ids: set,
) -> List[DetectResult]:
    if not allowed_ids:
        return detect_results
    filtered = []
    for dr in detect_results:
        keep = [
            f for f in dr.findings
            if (f.metadata or {}).get("finding_id") in allowed_ids
        ]
        if not keep:
            continue
        filtered.append(DetectResult(
            scanner=dr.scanner,
            overall_risk=dr.overall_risk,
            confidence=dr.confidence,
            confidence_reason=dr.confidence_reason,
            risk_distribution=dr.risk_distribution,
            findings=keep,
            policy_message=dr.policy_message,
            raw=dr.raw,
            feature_summary=dr.feature_summary or {},
        ))
    return filtered


def _filter_by_severity(detect_results: List[DetectResult], target_severity: str = "medium") -> List[DetectResult]:
    """Keep only target-severity findings from ALL scanners.

    Used when ai_only=False — rewrites predictability, AI, citation, etc.
    but only for medium findings significant enough to warrant an LLM call
    while still safe enough for automatic rewrite.
    """
    filtered = []
    for dr in detect_results:
        keep = [f for f in dr.findings if f.risk_level == target_severity]
        if not keep:
            continue
        filtered.append(DetectResult(
            scanner=dr.scanner,
            overall_risk=dr.overall_risk,
            confidence=dr.confidence,
            confidence_reason=dr.confidence_reason,
            risk_distribution=dr.risk_distribution,
            findings=keep,
            policy_message=dr.policy_message,
            raw=dr.raw,
            feature_summary=dr.feature_summary or {},
        ))
    return filtered


def _target_findings_for_mode(
    detect_results: List[DetectResult],
    ai_only: bool,
    rewrite_context: Optional[Any] = None,
) -> List[Finding]:
    """Return findings in the same scope the rewrite is allowed to target."""
    target_results = (
        _filter_ai_guided_findings(detect_results, rewrite_context)
        if ai_only else _filter_by_severity(detect_results)
    )
    return [f for dr in target_results for f in dr.findings]


def _ai_likelihood_from_results(detect_results: List[DetectResult]) -> float:
    """Extract the AI-generation likelihood score from scanner results."""
    for dr in detect_results:
        if dr.scanner == "ai_generation":
            return dr.likelihood_score or dr.overall_risk or 0.0
    return 0.0


def _original_ai_likelihood(detect_results: List[DetectResult], rewrite_context: Optional[Any]) -> float:
    """Prefer the original report badge when available, else scanner score."""
    if rewrite_context and hasattr(rewrite_context, "raw_json"):
        badge = (rewrite_context.raw_json or {}).get("ai_risk_badge") or {}
        score = badge.get("ai_likelihood_score")
        if isinstance(score, (int, float)):
            return float(score) / 100.0
    return _ai_likelihood_from_results(detect_results)


# ── Chip-in: use local `claude` CLI when no API key ──────────────────

REWRITE_CHIPIN_PROMPT = """You are a conservative paragraph-aware risk-mitigation editor. Your ONLY job is to propose small replacement candidates for the marked sentence.

This is NOT a writing-improvement task. Do not make the text sound better, smarter, smoother, more persuasive, more academic, or more complete. A candidate is useful only if it lowers the flagged detector signal while preserving the author's existing facts, voice, scope, and level of detail.

HOW PREDICTABILITY DETECTION WORKS:
A GPT-2 language model scores each word by how likely it was to appear next, given the preceding words. When many words rank among GPT-2's top-10 predictions, the sentence reads like AI output — predictable, generic, formulaic.

To reduce predictability, BREAK expected word paths without polishing the writing:
- Do not just swap one word for a synonym.
- For predictability findings, use Sentence Total Reconstruction: discard the original syntax, retain only the core technical keywords and claim, then rebuild a new one-sentence route from paragraph-supported context.
- Expand context only with words or details already present in the target, neighboring sentences, paragraph excerpt, scan anchors, or allowed additions.
- Keep the sentence grounded in the paragraph's existing concrete detail.
- Preserve the same author voice, including plain wording and first-person classroom reflection when present.
- Keep the author's word level. If the source says students, use students; do not upgrade to learners. If it says make/use/help, do not upgrade to constructing/requires/facilitates.
- Prefer nearby concrete nouns/actions over new abstract academic vocabulary.
- Do not imitate automated humanizer style; avoid random synonym swaps, inflated phrasing, or context-destroying paraphrase.

VALID CANDIDATE CONTRACT:
- The replacement must be supported by the target sentence or neighboring paragraph text.
- It may change clause order, sentence opening, rhythm, and ordinary wording.
- It may reuse nearby concrete words already present in the paragraph.
- It must not add a new method, cause, outcome, motivation, source relationship, example, citation, judgment, or explanation.
- If a detail is merely plausible but not present nearby, do not include it.
- Prefer a modest near-original candidate over an impressive unsupported rewrite.

TRANSFORMATION RULES (apply the one matching the finding type):

For predictability / common_words / topk_predictability / low_surprisal findings:
- If GPT-2 analysis is provided, use it: the flagged tokens are the most predictable words. Replace them OR restructure the sentence so they become less predictable in context.
- Prefer concrete, specific wording from the surrounding context over generic academic terms.
- NEVER use these predictable words: crucial, vital, essential, significant, notable, furthermore, moreover, additionally, demonstrates, highlights, underscores, plays, key role.
- NEVER introduce abstract polish such as technical accuracy, operational obstacles, visible learning framework, digital landscape, complex implementation, or similar noun stacks.
- NEVER make student-written wording sound like a textbook sentence. Avoid frames like "Constructing X requires...", "learners frequently fail...", "serves as a case", or "toward professional precision".
- NEVER invent support or causal links such as "through guided practice", "this shows", "this proves", "which encourages", "sparks interest", or "helps them continue improving" unless those exact ideas already appear nearby.

For formulaic_sentence / generic_phrase findings:
- Restructure the sentence to break its formulaic pattern. Move the subject to a different position. Change clause order. Start the sentence differently than it currently starts.

For style_shift / repetitive_structure findings:
- Change the sentence opening to differ from the previous sentence's opening. If the previous sentence starts with "The X...", start this one with a participle, adverb, or dependent clause instead.

For burstiness findings:
- If this sentence is similar in length to neighbors, make it noticeably shorter (cut filler words) or merge it with context.

For ai_generation findings:
- Fix only the specific sentence-level signal. Do not try to solve document-level specificity, grounding, or overall AI-likelihood by inventing details.

CRITICAL CONSTRAINTS:
- Keep the SAME factual meaning. Zero new information.
- Do not add support, evidence, examples, causes, benefits, or conclusions.
- All proper nouns, numbers, dates, citations, quoted text — copy verbatim.
- Replace only the sentence marked by <TARGET> in the paragraph context.
- Do NOT exceed the character limit.
- Output only the requested numbered candidate lines. No explanations."""


def _make_chipin_rewrite_fn(detect_context: str) -> callable:
    """Create a rewrite function that calls the local `claude` CLI."""
    def rewrite_fn(text: str, span_info: str) -> Optional[str]:
        max_chars = int(len(text) * 1.40)
        wants_candidates = "Return exactly 3 candidates" in span_info or "Return exactly 6 candidates" in span_info
        output_instruction = (
            "Return numbered replacement candidates and no commentary."
            if wants_candidates else
            "Output ONLY the rewritten text. No quotes, no commentary."
        )
        user_msg = (
            f"{detect_context}\n\n{span_info}\n\n"
            f"Current text ({len(text)} chars):\n{text}\n\n"
            f"Generate detector-safe replacement text for the marked target only. "
            f"This is risk mitigation, not prose improvement. "
            f"CRITICAL: Respect any stricter character limit in the task; otherwise do not exceed {max_chars} characters. "
            f"Preserve facts, scope, author voice, and detail level. "
            f"Do not add unsupported methods, examples, causes, benefits, motivations, conclusions, citations, or source links. "
            f"{output_instruction}"
        )
        try:
            result = subprocess.run(
                ["claude", "--print",
                 "--system-prompt", REWRITE_CHIPIN_PROMPT,
                 "--model", "sonnet",
                 user_msg],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                output = result.stdout.strip()
                # Strip triple quotes the LLM sometimes wraps output in
                if output.startswith('"""') and output.endswith('"""'):
                    output = output[3:-3].strip()
                elif output.startswith("```") and output.endswith("```"):
                    output = output[3:-3].strip()
                return output
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    return rewrite_fn


# ── Return type ──────────────────────────────────────────────────────

@dataclass
class RewriteModuleResult:
    mp_result: MultiPassResult
    report: Any
    text_report: str
    markdown_report: str
    summary: dict
    report_md_path: Optional[str]
    report_json_path: Optional[str]
    post_rewrite_detect: List[DetectResult]
    detect_loops_used: int
    detect_loop_history: List[dict]
    rewrite_plan: Optional[RewritePlan] = None
    manual_actions: List[RewriteAction] = field(default_factory=list)
    regression_memory_summary: List[dict] = field(default_factory=list)
    # New: outcome, floor, voice, surface
    outcome: RewriteOutcome = RewriteOutcome.MANUAL_REQUIRED
    floor_reasons: List[FloorReason] = field(default_factory=list)
    voice_profile: Optional[VoiceProfile] = None
    rewrite_surface: Optional[RewriteSurface] = None
    voice_guard_warnings: List[str] = field(default_factory=list)
    final_detect_report: Any = None  # reuse in pipeline to avoid redundant scan
    rewrite_checkpoints: List[Dict[str, Any]] = field(default_factory=list)


# ── Extract rewrite guidance from detect results ──────────────────────

def _extract_rewrite_guidance(detect_results: List[DetectResult]) -> dict:
    """Convert detect results into structured rewrite guidance."""
    guidance = {
        "predictability_findings": [],
        "similarity_findings": [],
        "citation_findings": [],
        "ai_generation_findings": [],
    }
    for dr in detect_results:
        for f in dr.findings:
            entry = {
                "finding_type": f.finding_type,
                "risk_level": f.risk_level,
                "evidence": f.evidence,
                "detail": f.detail,
                "recommendation": f.recommendation,
                "action_type": f.suggested_action_type or "",
                "metadata": f.metadata or {},
                "signal_category": getattr(f, "signal_category", "") or (f.metadata or {}).get("signal_category", ""),
            }
            if dr.scanner == "ai_generation":
                guidance["ai_generation_findings"].append(entry)
            elif dr.scanner == "predictability":
                guidance["predictability_findings"].append(entry)
            elif dr.scanner == "similarity":
                guidance["similarity_findings"].append(entry)
            elif dr.scanner == "citation":
                guidance["citation_findings"].append(entry)
    return guidance


def _build_detect_context(
    guidance: dict,
    domain_terms: Optional[List[str]] = None,
    rewrite_constraints: Optional[dict] = None,
) -> str:
    """Build a context string from detect findings to inject into rewrite prompts."""
    sections = []

    # AI generation findings — primary rewrite targets
    ai = guidance.get("ai_generation_findings", [])
    if ai:
        high_risk = [f for f in ai if f["risk_level"] in ("high", "medium", "critical")]
        if high_risk:
            lines = ["AI PATTERN ISSUES (reduce AI-detectable patterns):"]
            for f in high_risk:
                sig = f.get("signal_category", "")
                label = f"  - [{sig}] " if sig else "  - "
                lines.append(f"{label}\"{f['evidence']}\"")
                meta = f.get("metadata", {})
                badge_comp = meta.get("badge_components", {})
                if badge_comp:
                    top_signals = sorted(badge_comp.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:3]
                    signal_str = ", ".join(f"{k}={v:.0%}" for k, v in top_signals if isinstance(v, (int, float)))
                    if signal_str:
                        lines.append(f"    Top signals: {signal_str}")
                if f["recommendation"]:
                    lines.append(f"    Fix: {f['recommendation']}")
            sections.append("\n".join(lines))

    pred = guidance["predictability_findings"]
    if pred:
        high_risk = [f for f in pred if f["risk_level"] in ("high", "medium")]
        if high_risk:
            lines = ["PREDICTABILITY ISSUES (rewrite these spans):"]
            for f in high_risk:
                lines.append(f"  - {f['evidence']}")
                meta = f.get("metadata", {})
                if "score" in meta and isinstance(meta["score"], (int, float)):
                    lines.append(f"    Predictability score: {meta['score']:.3f}")
                if "top10_ratio" in meta and isinstance(meta["top10_ratio"], (int, float)):
                    lines.append(f"    Top-10 ratio: {meta['top10_ratio']:.1%}")
                if "avg_surprisal" in meta and isinstance(meta["avg_surprisal"], (int, float)):
                    lines.append(f"    Avg surprisal: {meta['avg_surprisal']:.2f}")
                if f["recommendation"]:
                    lines.append(f"    Fix: {f['recommendation']}")
            sections.append("\n".join(lines))

    sim = guidance["similarity_findings"]
    if sim:
        high_risk = [f for f in sim if f["risk_level"] in ("high", "medium")]
        if high_risk:
            lines = ["SIMILARITY ISSUES (rephrase to reduce source overlap):"]
            for f in high_risk:
                lines.append(f"  - {f['evidence']}")
                if f["recommendation"]:
                    lines.append(f"    Fix: {f['recommendation']}")
            sections.append("\n".join(lines))

    if domain_terms:
        lines = ["DOMAIN TERMS (preserve these in any rewrite):"]
        lines.append("  " + ", ".join(domain_terms[:20]))
        sections.append("\n".join(lines))

    # Rewrite constraints from detect pipeline
    if rewrite_constraints:
        constraint_lines = []
        preserve = rewrite_constraints.get("preserve_terms", [])
        if preserve:
            constraint_lines.append(f"MUST PRESERVE: {', '.join(preserve)}")
        do_not_add = rewrite_constraints.get("do_not_add", [])
        if do_not_add:
            constraint_lines.append("DO NOT ADD:")
            for item in do_not_add:
                constraint_lines.append(f"  - {item}")
        allowed = rewrite_constraints.get("allowed_additions", [])
        if allowed:
            constraint_lines.append("ALLOWED additions:")
            for item in allowed:
                constraint_lines.append(f"  + {item}")
        rule = rewrite_constraints.get("rewrite_rule", "")
        if rule:
            constraint_lines.append(f"RULE: {rule}")
        if constraint_lines:
            sections.append("REWRITE CONSTRAINTS:\n" + "\n".join(constraint_lines))

    return "\n\n".join(sections) if sections else ""


def _pct(value: Any) -> str:
    """Format scanner percentages that may arrive as 0-1 or 0-100."""
    if not isinstance(value, (int, float)):
        return ""
    return f"{value:.1f}%" if value > 1 else f"{value * 100:.1f}%"


def _build_scan_signal_brief(raw_json: Optional[dict]) -> str:
    """Build an actionable rewrite brief from the full scan report.

    This is intentionally separate from target selection. Document-level
    signals should guide how a target sentence is corrected, but they should
    not cause the rewriter to invent unsupported facts.
    """
    if not raw_json:
        return ""

    lines = ["SCAN SIGNAL BRIEF (use these signals to choose corrections):"]

    badge = raw_json.get("ai_risk_badge") or {}
    ai_components = badge.get("ai_components") or {}
    writing_components = badge.get("writing_components") or {}
    axis_scores = raw_json.get("axis_scores") or {}

    if badge:
        score = badge.get("ai_likelihood_score")
        quality = badge.get("writing_quality_score")
        tier = badge.get("tier")
        parts = []
        if isinstance(score, (int, float)):
            parts.append(f"AI likelihood {_pct(score)}")
        if tier:
            parts.append(f"tier {tier}")
        if isinstance(quality, (int, float)):
            parts.append(f"writing-quality risk {_pct(quality)}")
        if parts:
            lines.append("- Overall: " + ", ".join(parts))

    high_components = []
    for name, value in {**ai_components, **writing_components}.items():
        if isinstance(value, (int, float)) and value >= 50:
            high_components.append((name, value))
    if high_components:
        high_components.sort(key=lambda item: item[1], reverse=True)
        rendered = ", ".join(f"{name}={_pct(value)}" for name, value in high_components[:8])
        lines.append(f"- Strongest scanner drivers: {rendered}")

    if axis_scores:
        rendered = ", ".join(f"{k}={v}" for k, v in axis_scores.items())
        lines.append(f"- Axis assessment: {rendered}")

    constraints = raw_json.get("rewrite_constraints") or {}
    allowed = constraints.get("allowed_additions") or []
    do_not_add = constraints.get("do_not_add") or []
    if allowed:
        lines.append("- Safe correction material from scan:")
        for item in allowed[:5]:
            lines.append(f"  + {item}")
    if do_not_add:
        lines.append("- Hard limits:")
        for item in do_not_add[:5]:
            lines.append(f"  - Do not add {item}.")

    # Pull document-level specificity metrics without making it an edit target.
    low_spec_metrics = None
    findings = raw_json.get("findings") or {}
    for tier in ("critical", "high", "medium", "low"):
        for finding in findings.get(tier, []):
            if finding.get("title") != "low_specificity":
                continue
            evidence = finding.get("evidence")
            if isinstance(evidence, dict):
                low_spec_metrics = evidence.get("metrics") or {}
                break
        if low_spec_metrics:
            break
    if low_spec_metrics:
        lines.append(
            "- Specificity signal: "
            f"{int(low_spec_metrics.get('named_entities', 0))} named entities, "
            f"{int(low_spec_metrics.get('numbers', 0))} numbers, "
            f"{int(low_spec_metrics.get('dates', 0))} dates, "
            f"domain grounding={low_spec_metrics.get('domain_grounding_level', 'unknown')}."
        )
        domain_terms = low_spec_metrics.get("domain_terms") or []
        if domain_terms:
            lines.append("- Preserve/use existing domain terms where natural: " + ", ".join(domain_terms[:10]))

    correction_rules = []
    if ai_components.get("topk_pattern", 0) >= 50 or ai_components.get("predictability", 0) >= 45:
        correction_rules.append(
            "For predictable sentences, replace generic scaffolding with concrete nouns/verbs already implied by context."
        )
    if ai_components.get("generic_assertion_risk", 0) >= 50 or writing_components.get("broad_claim_risk", 0) >= 50:
        correction_rules.append(
            "For broad claims, narrow the claim or attach it to the sentence's existing classroom/process context."
        )
    if writing_components.get("lived_detail_risk", 0) >= 50:
        correction_rules.append(
            "Add lived/process detail only when it can be inferred from existing wording; otherwise keep the claim narrower."
        )
    if writing_components.get("source_grounding_risk", 0) >= 50:
        correction_rules.append(
            "Do not fabricate sources; prefer source-neutral phrasing or mark the need for source support."
        )
    if ai_components.get("burstiness_risk", 0) >= 45:
        correction_rules.append(
            "Vary rhythm by making the target sentence clearly shorter or more direct when meaning is preserved."
        )
    if correction_rules:
        lines.append("- Correction rules:")
        for rule in correction_rules:
            lines.append(f"  * {rule}")

    return "\n".join(lines)


def _sentence_signal_context(rewrite_context: Optional[Any], finding: Finding, sent_idx: int) -> str:
    """Return sentence-specific scan metrics for the rewrite prompt."""
    if not rewrite_context or not hasattr(rewrite_context, "raw_json"):
        return ""
    raw = rewrite_context.raw_json or {}
    sentence_id = (finding.location or {}).get("sentence_id")
    if not sentence_id and sent_idx >= 0:
        sentence_id = f"s{sent_idx + 1:03d}"

    parts = []
    sentence_map = raw.get("sentence_map") or {}
    if sentence_id and isinstance(sentence_map, dict) and sentence_id in sentence_map:
        info = sentence_map[sentence_id]
        if info.get("paragraph_id"):
            parts.append(f"paragraph={info['paragraph_id']}")

    for sent in (raw.get("predictability") or {}).get("sentences", []):
        if sent.get("sentence_id") == sentence_id:
            if sent.get("risk"):
                parts.append(f"predictability={sent['risk']}")
            if isinstance(sent.get("score"), (int, float)):
                parts.append(f"score={sent['score']:.3f}")
            if isinstance(sent.get("top10") or sent.get("top10_ratio") or sent.get("top_10_ratio"), (int, float)):
                t10 = sent.get("top10") or sent.get("top10_ratio") or sent.get("top_10_ratio")
                parts.append(f"top10={t10:.1%}")
            break

    if not parts:
        return ""
    return "Sentence scan metrics: " + ", ".join(parts)


# ── Signal-enriched prompt construction ──────────────────────────────

def _matches_sentence(sent_data: dict, finding: Finding, sent_idx: int) -> bool:
    """Check if a sentence data dict corresponds to this finding's location."""
    sid = sent_data.get("sentence_id", "")
    loc = finding.location or {}
    if loc.get("sentence_id") and sid == loc["sentence_id"]:
        return True
    if sent_idx >= 0 and sid == f"s{sent_idx + 1:03d}":
        return True
    return False


def _extract_flagged_tokens(sent_data: dict) -> Optional[str]:
    """Get the most predictable tokens from sentence-level data."""
    tokens = sent_data.get("top_predicted_tokens", [])
    if tokens:
        formatted = []
        for t in tokens[:5]:
            if isinstance(t, dict):
                token = t.get("token", "")
                rank = t.get("rank")
                prob = t.get("probability")
                detail = f'"{token}"'
                if rank is not None:
                    detail += f" rank={rank}"
                if isinstance(prob, (int, float)):
                    detail += f" p={prob:.1%}"
                formatted.append(detail)
            else:
                formatted.append(f'"{t}"')
        return ", ".join(formatted)
    return None


_FINDING_STRATEGIES = {
    "high_predictability": "Replace predictable words with natural alternatives implied by context. Prefer concrete nouns/verbs over generic ones.",
    "medium_predictability": "Use the GPT-2 token signal to change clause order or context around predictable words. Avoid synonym-only edits.",
    "high_topk_predictability": "Break the commonly predicted token sequence by changing sentence structure, clause order, or the surrounding context of flagged words.",
    "low_surprisal": "Introduce less expected word choices while keeping meaning identical.",
    "formulaic_sentence": "Break the formulaic structure. Move subject position, change clause order, and change sentence opening.",
    "generic_phrase": "Replace generic phrase with specific wording from surrounding context.",
    "style_shift": "Adjust this sentence's opening and structure to differ from the previous sentence.",
    "repetitive_structure": "Vary the sentence pattern — change opening, clause order, or rhythm.",
    "burstiness": "Shorten this sentence by cutting filler, or merge with neighboring context for rhythm variety.",
    "ai_generation": "Address the specific signal flagged above. Do NOT rewrite broadly.",
    "generic_formulaic_language": "Replace generic academic phrasing with natural wording from the document's context.",
    "mechanical_transition": "Replace formulaic transition with natural connective logic from context.",
    "generic_enumeration": "Break the numbered-list pattern — restructure as flowing prose.",
    "vague_claim": "Make the claim more specific using details already present in surrounding text.",
}


def _derive_strategy(finding: Finding, enriched_text: str) -> str:
    """Derive a concrete, finding-specific rewrite strategy."""
    if "Strategy:" in enriched_text:
        for line in enriched_text.split("\n"):
            line = line.strip()
            if line.startswith("Strategy:"):
                return line.replace("Strategy: ", "", 1)
    return _FINDING_STRATEGIES.get(
        finding.finding_type,
        finding.suggested_action_type or "Rephrase to reduce flagged pattern",
    )


STRUCTURAL_REWRITE_FINDINGS = {
    "high_predictability",
    "medium_predictability",
    "high_topk_predictability",
    "low_surprisal",
    "low_surprisal_pattern",
    "formulaic_sentence",
    "generic_formulaic_language",
    "repetitive_sentence_structure",
}


def _needs_structural_rewrite(finding: Finding) -> bool:
    return finding.finding_type in STRUCTURAL_REWRITE_FINDINGS


def _max_candidate_chars(original_sentence: str, finding: Finding) -> int:
    """Allow more room when GPT-2 says structure, not a word, is the problem."""
    ratio = 1.70 if _needs_structural_rewrite(finding) else 1.20
    return int(len(original_sentence) * ratio)


def _rewrite_task_instruction(finding: Finding, max_chars: int) -> str:
    """Finding-specific instruction for the LLM rewrite call."""
    if _needs_structural_rewrite(finding):
        return (
            "Use the GPT-2 signal to produce ONE structurally different sentence. "
            "Use Sentence Total Reconstruction: do not preserve the original syntax; "
            "retain the core technical keywords and claim, then rebuild the sentence route from supported paragraph context. "
            "Do not solve this with synonym swaps alone. Change clause order, sentence opening, "
            "or the context around the flagged predictable tokens while preserving the same facts. "
            f"MUST NOT exceed {max_chars} characters. "
            "Follow the candidate-output format below."
        )
    return (
        "Apply a narrow in-place edit to fix the finding above. "
        "Change only what is needed. "
        f"MUST NOT exceed {max_chars} characters. "
        "Follow the candidate-output format below."
    )


def _candidate_task_instruction(
    finding: Finding,
    max_chars: int,
    rewrite_operation: Optional[Dict[str, Any]] = None,
) -> str:
    """Ask the LLM for several paragraph-aware sentence candidates."""
    base = _rewrite_task_instruction(finding, max_chars)
    shapes = []
    if isinstance(rewrite_operation, dict):
        shapes = [
            str(item).strip()
            for item in (rewrite_operation.get("candidate_shapes") or [])
            if str(item).strip()
        ][:3]
    while len(shapes) < 3:
        defaults = [
            "minimal plain edit",
            "clause-reordered edit using paragraph detail",
            "short conservative edit",
        ]
        shapes.append(defaults[len(shapes)])
    common_rules = (
        "Return exactly 3 candidates as numbered lines:\n"
        f"1. <{shapes[0]}>\n"
        f"2. <{shapes[1]}>\n"
        f"3. <{shapes[2]}>\n"
        "Each candidate must be ONE sentence and must replace only the <TARGET> sentence. "
        "This is detector risk mitigation, not writing improvement. "
        "Do not include explanations. "
        "For predictability findings, apply Sentence Total Reconstruction: wipe the original sentence syntax, "
        "retain only core technical keywords and the same claim, then rebuild a different sentence route using supported context. "
        "Do not preserve the original phrase links just with synonyms. "
        "Do not make the sentence more formal, broader, smoother, or more academic. "
        "Do not make the idea sound more complete than the original. "
        "Keep the original word level: do not replace 'students' with 'learners', "
        "'make' with 'constructing', or simple classroom wording with textbook phrasing. "
        "Do not add facts, citations, names, dates, statistics, examples, causes, benefits, "
        "methods, source relationships, motivations, or conclusions. "
        "If a word or idea is not in the target sentence, previous sentence, next sentence, "
        "paragraph excerpt, domain anchors, or allowed additions, do not introduce it. "
        "A near-original candidate is better than an unsupported rewrite. "
        "Avoid abstract noun stacks and generic polish such as 'crucial', 'significant', "
        "'essential', 'technical accuracy', 'operational obstacles', "
        "'visible learning framework', 'digital landscape', 'learners frequently fail', "
        "'Constructing ... requires', 'serves as a case', 'online info', "
        "'stands as', 'guided practice', 'this shows', 'this proves', "
        "'which encourages', 'continuing to improve', and 'sparks interest'."
    )
    if _requires_medium_exit(finding):
        return (
            f"{base}\n"
            "The current target is a MEDIUM predictability finding. Aim to fall below "
            f"{MEDIUM_PREDICTABILITY_CEILING:.2f}; if that is not possible, still produce the strongest safe measurable reduction. "
            "Avoid generic/predictable frames such as 'This is...', 'Because of this...', "
            "'The goal should be...', 'modern world', 'not only...', and 'people who can'. "
            "Use the Signal instruction, Domain anchors, and Allowed concrete additions above. "
            "Each candidate should convert the sentence into a supported concrete situation, observation, or domain action instead of a generic paraphrase. "
            "Do not use metaphorical polish when a concrete domain noun/action is available. "
            "Use varied clause order, concrete everyday phrasing, and a less formulaic sentence path. "
            f"{common_rules}"
        )
    return (
        f"{base}\n"
        f"{common_rules}"
    )


def _parse_rewrite_candidates(output: Optional[str], original_sentence: str = "") -> List[str]:
    """Parse numbered/bulleted LLM output into distinct sentence candidates."""
    if not output:
        return []
    text = output.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()
    candidates: List[str] = []
    current: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(?:candidate\s*)?[\(\[]?([1-9])[\)\].:-]\s*(.+)$", line, re.I)
        bullet = re.match(r"^[-*]\s+(.+)$", line)
        if match:
            if current:
                candidates.append(" ".join(current).strip())
            current = [match.group(2).strip()]
        elif bullet:
            if current:
                candidates.append(" ".join(current).strip())
            current = [bullet.group(1).strip()]
        elif current:
            current.append(line)
        else:
            current = [line]
    if current:
        candidates.append(" ".join(current).strip())

    cleaned: List[str] = []
    seen = set()
    for cand in candidates:
        cand = cand.strip().strip('"').strip("'").strip()
        cand = re.sub(r"^(?:minimal|structural|plain|natural)\s*(?:edit)?\s*:\s*", "", cand, flags=re.I)
        cand = " ".join(cand.split())
        if not cand or cand == original_sentence:
            continue
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(cand)
        if len(cleaned) >= 6:
            break
    return cleaned


def _paragraph_context(
    sentences: List[str],
    para_map: Dict[int, int],
    sent_idx: int,
) -> str:
    """Return the full paragraph containing the target sentence."""
    target_para = para_map.get(sent_idx)
    if target_para is None:
        start = max(0, sent_idx - 1)
        end = min(len(sentences), sent_idx + 2)
        return " ".join(sentences[start:end])
    return " ".join(
        s for i, s in enumerate(sentences)
        if para_map.get(i) == target_para
    )


def _marked_paragraph(paragraph: str, sentence: str) -> str:
    if sentence in paragraph:
        return paragraph.replace(sentence, f"<TARGET>{sentence}</TARGET>", 1)
    return f"<TARGET>{sentence}</TARGET>"


def _domain_anchor_terms(
    rewrite_context: Optional[Any],
    paragraph: str,
    sentence: str,
    limit: int = 12,
) -> List[str]:
    """Return domain terms from the scan that are relevant to this target."""
    if not rewrite_context or not hasattr(rewrite_context, "raw_json"):
        return []
    raw = rewrite_context.raw_json or {}
    domain_profile = raw.get("domain_profile") or {}
    terms = domain_profile.get("matched_domain_terms") or []
    if not isinstance(terms, list):
        return []

    context_text = f"{paragraph} {sentence}"
    context = context_text.lower()
    priority_terms = []
    if sentence and sentence in paragraph:
        before = paragraph.split(sentence, 1)[0]
        before_sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", before)
            if s.strip()
        ]
        focus = " ".join(before_sentences[-2:])
        focus_stopwords = {
            "about", "after", "again", "also", "being", "because", "before",
            "between", "could", "does", "every", "from", "have", "into",
            "many", "more", "most", "need", "needs", "only", "same",
            "should", "simply", "some", "than", "that", "their", "them",
            "there", "these", "they", "this", "those", "time", "when",
            "where", "while", "with", "without", "would",
            "another", "complex", "especially", "explain", "gives", "issue",
            "layer", "limits", "memory", "multiple", "problem", "process",
            "working", "information",
            "elements", "handled", "must",
        }
        for word in re.findall(r"\b[A-Za-z][A-Za-z-]{3,}\b", focus):
            clean = word.strip()
            lower = clean.lower()
            if lower not in focus_stopwords and clean not in priority_terms:
                priority_terms.append(clean)

    matched = []
    fallback = []
    for term in terms:
        if not isinstance(term, str):
            continue
        clean = term.strip()
        if not clean:
            continue
        lower = clean.lower()
        if lower in context:
            matched.append(clean)
        elif len(fallback) < limit:
            fallback.append(clean)

    constraints = raw.get("rewrite_constraints") or {}
    for item in constraints.get("allowed_additions") or []:
        for chunk in re.split(r"[:;,]", str(item)):
            clean = chunk.strip()
            if clean and clean.lower() in context:
                matched.append(clean)

    stopwords = {
        "about", "after", "again", "being", "because", "before", "between",
        "could", "every", "from", "have", "into", "many", "more", "most",
        "only", "same", "should", "simply", "some", "than", "that", "their",
        "them", "there", "these", "they", "this", "those", "time", "when",
        "where", "while", "with", "without", "would",
    }
    for word in re.findall(r"\b[A-Za-z][A-Za-z-]{3,}\b", context_text):
        lower = word.lower()
        if lower not in stopwords:
            matched.append(word)

    ranked = sorted(set(matched), key=lambda t: (-len(t.split()), -len(t), t.lower()))
    ranked = [t for t in ranked if t.lower() not in {"same time"}]
    ordered = []
    for term in priority_terms + ranked:
        lower = term.lower()
        if lower not in {t.lower() for t in ordered}:
            ordered.append(term)
        if len(ordered) >= limit:
            return ordered
    if ordered:
        return ordered
    return fallback[:limit]


def _rewrite_edit_brief_for_target(
    rewrite_context: Optional[Any],
    finding: Finding,
) -> Dict[str, Any]:
    """Return the detect-produced edit brief for this finding when available."""
    if not rewrite_context or not hasattr(rewrite_context, "raw_json"):
        return {}
    raw = rewrite_context.raw_json or {}
    briefs = raw.get("rewrite_edit_briefs") or []
    if not isinstance(briefs, list):
        return {}
    finding_id = _finding_id(finding)
    loc = finding.location or {}
    sentence_id = loc.get("sentence_id")
    for brief in briefs:
        if not isinstance(brief, dict):
            continue
        if finding_id and brief.get("finding_id") == finding_id:
            return brief
        if (
            sentence_id
            and brief.get("sentence_id") == sentence_id
            and (brief.get("signals") or {}).get("finding_type") == finding.finding_type
        ):
            return brief
    return {}


def _brief_signal_lines(brief: Dict[str, Any]) -> List[str]:
    if not brief:
        return []
    lines = []
    role = brief.get("paragraph_role")
    if role:
        lines.append(f"Paragraph role: {role}")
    instruction = brief.get("instruction")
    if instruction:
        lines.append("Detect edit instruction: " + str(instruction))
    signals = brief.get("signals") or {}
    if signals:
        metrics = []
        for key in ("score", "top10_ratio", "top50_ratio", "avg_surprisal"):
            value = signals.get(key)
            if isinstance(value, (int, float)):
                metrics.append(f"{key}={value:.3f}")
        if metrics:
            lines.append("Edit-brief metrics: " + ", ".join(metrics))
        tokens = signals.get("problem_tokens") or []
        if tokens:
            formatted = []
            for item in tokens[:6]:
                if isinstance(item, dict):
                    token = item.get("token", "")
                    rank = item.get("rank")
                    text = f'"{token}"'
                    if rank is not None:
                        text += f" rank={rank}"
                    formatted.append(text)
                elif isinstance(item, str):
                    formatted.append(f'"{item}"')
            if formatted:
                lines.append("Edit-brief problem tokens: " + ", ".join(formatted))
        spans = signals.get("predictable_token_spans") or []
        if spans:
            lines.append(
                "Edit-brief predictable spans: "
                + ", ".join(f'"{span}"' for span in spans[:5])
            )
    return lines


def _protected_texts_from_brief(brief: Dict[str, Any]) -> List[str]:
    protected = []
    for item in (brief or {}).get("protected_spans") or []:
        if isinstance(item, dict) and item.get("text"):
            protected.append(str(item["text"]))
    return protected


def _paragraph_coherence_reject_reason(
    original_sentence: str,
    candidate_sentence: str,
    previous_sentence: str,
    next_sentence: str,
    domain_anchor_terms: Optional[List[str]] = None,
    paragraph_context: str = "",
) -> str:
    """Reject candidates that work locally but harm paragraph coherence."""
    for label, neighbor in (("previous", previous_sentence), ("next", next_sentence)):
        if not neighbor:
            continue
        similarity = SequenceMatcher(
            None,
            candidate_sentence.lower(),
            neighbor.lower(),
            autojunk=False,
        ).ratio()
        if similarity >= 0.88:
            return f"duplicates_{label}_sentence {similarity:.2f}"

    anchors = domain_anchor_terms or []
    orig_coverage = _term_coverage(original_sentence, anchors)
    cand_coverage = _term_coverage(candidate_sentence, anchors)
    min_anchor_coverage = max(1, int((orig_coverage * 0.35) + 0.999))
    if orig_coverage >= 2 and cand_coverage < min_anchor_coverage:
        return f"domain_anchor_loss {orig_coverage}->{cand_coverage}"

    abstract_patterns = [
        r"\b(?:framework|landscape|rigor|implementation|engagement|outcomes?|obstacles?|oversight)\b",
        r"\b(?:complex|technical|operational|visible|digital|geometric)\s+\w+",
    ]
    orig_abstract = sum(len(re.findall(p, original_sentence, re.I)) for p in abstract_patterns)
    cand_abstract = sum(len(re.findall(p, candidate_sentence, re.I)) for p in abstract_patterns)
    if cand_abstract > orig_abstract:
        return f"unsupported_abstraction {orig_abstract}->{cand_abstract}"

    surrounding = " ".join(
        part for part in (original_sentence, previous_sentence, next_sentence, paragraph_context) if part
    ).lower()
    unsupported_additions = [
        r"\bonline info\b",
        r"\bstands as\b",
        r"\bsole place\b",
        r"\bmany now\b",
        r"\bcurrent learning environment\b",
        r"\blearning environment\b",
        r"\bacquiring knowledge happens\b",
        r"\bknowledge happens\b",
        r"\bmore places\b",
        r"\bother places\b",
        r"\bother sources\b",
        r"\bmany other sources\b",
        r"\bmany other channels\b",
        r"\bjust school\b",
        r"\bthis method\b",
        r"\bthe method\b",
        r"\bby focusing on\b",
        r"\bfocusing on each procedure\b",
        r"\bfrom simple to precise work\b",
        r"\bscaffolding\b",
        r"\bbreaking down each step\b",
        r"\bgrasp(?:s|ed|ing)?\b",
        r"\baccurate cuts?\b",
        r"\bguided practice\b",
        r"\bsparks? (?:their |student )?interest\b",
        r"\bpractic(?:e|ing) further\b",
        r"\bcontinuing to improve\b",
        r"\bdaily life\b",
        r"\bworks? in practice\b",
        r"\breveal(?:s|ed|ing)? how this relationship works\b",
        r"\b(?:observe|notice|see) how this relationship works\b",
        r"\bhow this relationship works\b",
        r"\bembedded within\b",
        r"\bespecially evident\b",
        r"\bespecially clear\b",
        r"\bexecut(?:e|es|ed|ing) a controlled\b",
        r"\bshift(?:s|ed|ing)? from observing\b",
        r"\bfrom my experience teaching\b",
        r"\bpart of how technical skills are taught\b",
    ]
    for pattern in unsupported_additions:
        match = re.search(pattern, candidate_sentence, re.I)
        if match and not re.search(pattern, surrounding, re.I):
            return f"unsupported_new_phrase '{match.group(0)}'"

    new_terms = _unsupported_new_content_words(
        candidate_sentence,
        surrounding,
        domain_anchor_terms or [],
    )
    if len(new_terms) >= 3:
        return "unsupported_new_terms " + ",".join(new_terms[:5])

    return ""


def _manual_suggestion_item(
    *,
    finding: Finding,
    original_sentence: str,
    candidate_sentence: str,
    rejection_reason: str,
    paragraph_role: str = "",
) -> Dict[str, Any]:
    return {
        "finding_id": _finding_id(finding),
        "finding_type": finding.finding_type,
        "risk_level": finding.risk_level,
        "scanner_target": (finding.metadata or {}).get("scanner") or (finding.metadata or {}).get("category") or "",
        "sentence_id": (finding.location or {}).get("sentence_id"),
        "paragraph_role": paragraph_role or "unknown",
        "original_sentence": original_sentence,
        "suggested_sentence": candidate_sentence,
        "rejection_reason": rejection_reason,
        "why_review_manually": (
            "This candidate preserved enough meaning to be useful, but an automatic guard rejected it. "
            "Review it manually before using it."
        ),
    }


def _rewrite_constraint_lines(rewrite_context: Optional[Any], limit: int = 3) -> List[str]:
    if not rewrite_context or not hasattr(rewrite_context, "raw_json"):
        return []
    constraints = (rewrite_context.raw_json or {}).get("rewrite_constraints") or {}
    lines: List[str] = []
    allowed = constraints.get("allowed_additions") or []
    if allowed:
        lines.append("Allowed concrete additions from scan: " + "; ".join(str(x) for x in allowed[:limit]))
    do_not_add = constraints.get("do_not_add") or []
    if do_not_add:
        lines.append("Do not add: " + "; ".join(str(x) for x in do_not_add[:limit]))
    rule = constraints.get("rewrite_rule")
    if rule:
        lines.append("Rewrite rule from scan: " + str(rule))
    return lines


def _signal_driven_instruction(
    finding: Finding,
    enriched_text: str,
    domain_anchors: List[str],
) -> str:
    """Convert scanner signals into concrete LLM instructions."""
    anchors = ", ".join(domain_anchors[:8]) if domain_anchors else "terms already present in the paragraph"
    if finding.finding_type in ("medium_predictability", "high_predictability"):
        return (
            "Signal instruction: This is a common-word / predictable-path sentence. "
            "Do not make it smoother. Use Sentence Total Reconstruction: discard the original syntax, keep the core technical anchors and same claim, "
            "then rebuild around a concrete observation, condition, or action already supported by the paragraph. "
            f"Use 1-3 relevant domain anchors if natural: {anchors}. "
            "Replace broad wording such as 'theory is there', 'many of them', 'not enough', 'can be effective', or 'at the same time' with a specific situation from the context."
        )
    if finding.finding_type in ("high_topk_predictability", "low_surprisal", "low_surprisal_pattern"):
        return (
            "Signal instruction: This is a top-k predictability pattern. "
            "Use Sentence Total Reconstruction: change the sentence opening and word path, not just synonyms. "
            f"Start from a domain object, action, constraint, or observed situation using anchors such as: {anchors}."
        )
    if finding.finding_type in ("formulaic_sentence", "generic_phrase", "generic_formulaic_language"):
        return (
            "Signal instruction: This is formulaic language. "
            "Break the template by changing clause order and replacing generic academic phrasing with context-specific detail already present nearby."
        )
    if "Specificity" in enriched_text or finding.finding_type in ("vague_claim", "low_specificity"):
        return (
            "Signal instruction: Specificity is weak. "
            "Add only supported concrete detail from the paragraph or scan constraints; do not invent new facts, citations, names, dates, or statistics."
        )
    return (
        "Signal instruction: Use the scanner evidence above as the edit target. "
        "Prefer supported, domain-specific wording over generic paraphrase."
    )


def _signal_list(edit_brief: Dict[str, Any], metadata_context: Dict[str, Any], key: str) -> List[Any]:
    values: List[Any] = []
    signals = edit_brief.get("signals") if isinstance(edit_brief, dict) else {}
    for source in (signals if isinstance(signals, dict) else {}, metadata_context if isinstance(metadata_context, dict) else {}):
        raw = source.get(key) or []
        if isinstance(raw, list):
            values.extend(raw)
    return values


def _signal_number(edit_brief: Dict[str, Any], key: str) -> Optional[float]:
    signals = edit_brief.get("signals") if isinstance(edit_brief, dict) else {}
    if not isinstance(signals, dict):
        return None
    value = signals.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _token_names(tokens: List[Any], limit: int = 6) -> List[str]:
    names: List[str] = []
    for item in tokens[:limit]:
        token = ""
        if isinstance(item, dict):
            token = str(item.get("token") or "").strip()
        elif isinstance(item, str):
            token = item.strip()
        if token and token.lower() not in {t.lower() for t in names}:
            names.append(token)
    return names


def _plan_rewrite_operation(
    *,
    finding: Finding,
    edit_brief: Dict[str, Any],
    metadata_context: Dict[str, Any],
    original_sentence: str,
    previous_sentence: str,
    next_sentence: str,
    domain_anchors: List[str],
) -> Dict[str, Any]:
    """Turn scan signals into a concrete edit operation for the LLM.

    The operation constrains how candidates should be generated. This keeps the
    model from treating every finding as a general rewrite request.
    """
    ftype = finding.finding_type
    problem_tokens = _token_names(_signal_list(edit_brief, metadata_context, "problem_tokens"))
    predictable_spans = [
        str(span).strip()
        for span in _signal_list(edit_brief, metadata_context, "predictable_token_spans")
        if str(span).strip()
    ][:5]
    word_count = len(re.findall(r"\b\w+\b", original_sentence))
    has_long_clause = word_count >= 28 or len(original_sentence) >= 180 or original_sentence.count(",") >= 2
    top10_ratio = _signal_number(edit_brief, "top10_ratio")
    score = _signal_number(edit_brief, "score")
    anchors = [term for term in domain_anchors[:8] if isinstance(term, str) and term.strip()]

    if ftype in {"low_specificity", "source_grounding", "polished_but_ungrounded", "uncited_claim", "uncited_in_body"}:
        return {
            "operation": "manual_support_only",
            "objective": "Do not auto-rewrite; this needs author evidence, source linkage, or claim softening.",
            "allowed_moves": ["Return no automatic candidate unless the text can be softened without adding support."],
            "forbidden_moves": ["Do not invent citations, examples, source relationships, causes, or evidence."],
            "candidate_shapes": [
                "source-neutral softening only",
                "scope-narrowing only",
                "near-original preservation",
            ],
            "problem_tokens": problem_tokens,
            "predictable_spans": predictable_spans,
            "domain_anchors": anchors,
        }

    if ftype in {"formulaic_sentence", "generic_phrase", "generic_formulaic_language"}:
        operation = "break_formulaic_frame"
        objective = "Break the formulaic frame without adding new meaning."
        shapes = [
            "remove or replace the formulaic opener",
            "move the concrete subject/action earlier",
            "shorten the phrase while preserving meaning",
        ]
    elif has_long_clause and ("predictability" in ftype or "topk" in ftype or "surprisal" in ftype):
        operation = "shorten_and_reorder"
        objective = "Use Sentence Total Reconstruction: cut filler, retain core anchors, and rebuild clause order around the predictable path."
        shapes = [
            "shorter sentence that cuts filler",
            "clause-reordered sentence using existing words",
            "near-original sentence with only the predictable span changed",
        ]
    elif ftype in {"high_topk_predictability", "low_surprisal", "low_surprisal_pattern"}:
        operation = "change_opening_and_token_path"
        objective = "Use Sentence Total Reconstruction to change the sentence opening and route around top-k problem tokens."
        shapes = [
            "open with a nearby concrete anchor",
            "move the target claim after the concrete context",
            "near-original sentence with the problem-token path changed",
        ]
    elif "predictability" in ftype and predictable_spans:
        operation = "rebuild_predictable_span"
        objective = "Use Sentence Total Reconstruction: discard the predictable syntax while keeping the same claim and local vocabulary."
        shapes = [
            "minimal edit around the predictable span",
            "clause-reordered edit using paragraph anchors",
            "short conservative edit that avoids the same token path",
        ]
    elif "predictability" in ftype and problem_tokens:
        operation = "route_around_problem_tokens"
        objective = "Use Sentence Total Reconstruction to avoid the exact common token route while preserving the same meaning."
        shapes = [
            "minimal edit around problem tokens",
            "different sentence opening using a nearby anchor",
            "short near-original edit",
        ]
    elif anchors:
        operation = "context_anchor_rewrite"
        objective = "Use nearby anchors to make a plain, context-bound sentence without new facts."
        shapes = [
            "minimal anchor-preserving edit",
            "clause reorder using one nearby anchor",
            "short conservative edit",
        ]
    else:
        operation = "minimal_plain_route_change"
        objective = "Make the smallest sentence-route change that can reduce the signal."
        shapes = [
            "minimal plain edit",
            "small clause-order change",
            "short conservative edit",
        ]

    forbidden = [
        "No new facts, examples, citations, causes, benefits, motivations, source links, or conclusions.",
        "No smoother academic phrasing or abstract noun stacks.",
        "No new method words unless already present nearby.",
    ]
    allowed = [
        "For predictability findings, wipe the original syntax and rebuild from retained core anchors.",
        "Change opening, clause order, rhythm, and local word path.",
        "Reuse words from the target, neighboring sentences, paragraph excerpt, domain anchors, and allowed additions.",
        "Keep the same author voice and detail level.",
    ]
    return {
        "operation": operation,
        "objective": objective,
        "allowed_moves": allowed,
        "forbidden_moves": forbidden,
        "candidate_shapes": shapes,
        "problem_tokens": problem_tokens,
        "predictable_spans": predictable_spans,
        "domain_anchors": anchors,
        "metrics": {
            "score": score,
            "top10_ratio": top10_ratio,
            "word_count": word_count,
        },
        "previous_sentence_available": bool(previous_sentence),
        "next_sentence_available": bool(next_sentence),
    }


def _rewrite_operation_lines(operation: Dict[str, Any]) -> List[str]:
    if not operation:
        return []
    lines = [
        "Rewrite operation: " + str(operation.get("operation") or "minimal_plain_route_change"),
        "Operation objective: " + str(operation.get("objective") or "Reduce the target signal conservatively."),
    ]
    problem_tokens = operation.get("problem_tokens") or []
    if problem_tokens:
        lines.append("Operation problem tokens: " + ", ".join(f'"{t}"' for t in problem_tokens[:6]))
    predictable_spans = operation.get("predictable_spans") or []
    if predictable_spans:
        lines.append("Operation predictable spans: " + ", ".join(f'"{s}"' for s in predictable_spans[:4]))
    anchors = operation.get("domain_anchors") or []
    if anchors:
        lines.append("Operation legal anchors: " + ", ".join(str(a) for a in anchors[:8]))
    allowed = operation.get("allowed_moves") or []
    if allowed:
        lines.append("Allowed operation moves: " + " ".join(f"- {move}" for move in allowed[:4]))
    forbidden = operation.get("forbidden_moves") or []
    if forbidden:
        lines.append("Forbidden operation moves: " + " ".join(f"- {move}" for move in forbidden[:4]))
    return lines


def _term_coverage(text: str, terms: List[str]) -> int:
    lower = text.lower()
    return sum(1 for term in terms if term.lower() in lower)


def _unsupported_new_content_words(
    candidate_sentence: str,
    surrounding_text: str,
    domain_anchor_terms: List[str],
    limit: int = 5,
) -> List[str]:
    """Return new content words that are not grounded in context or anchors."""
    allowed_text = surrounding_text.lower()
    anchor_set = {str(term).lower() for term in domain_anchor_terms or []}
    def _rough_stem(value: str) -> str:
        value = value.lower()
        for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
            if len(value) > len(suffix) + 3 and value.endswith(suffix):
                return value[: -len(suffix)]
        return value

    allowed_stems = {
        _rough_stem(word)
        for word in re.findall(r"\b[A-Za-z][A-Za-z-]{3,}\b", allowed_text)
    }
    anchor_stems = {_rough_stem(term) for term in anchor_set}
    stopwords = {
        "about", "after", "again", "also", "being", "because", "before",
        "between", "could", "does", "each", "every", "from", "have", "into",
        "many", "more", "most", "need", "needs", "only", "same", "should",
        "simply", "some", "than", "that", "their", "them", "there", "these",
        "they", "this", "those", "time", "when", "where", "while", "with",
        "without", "would", "current", "modern", "today", "now", "slightly",
    }
    risky_new_words = {
        "method", "methods", "practice", "practicing", "scaffolding",
        "focus", "focusing", "grasp", "grasps", "accurate", "cuts",
        "encourage", "encourages", "interest", "learning", "benefit",
        "benefits", "support", "supports", "prove", "proves", "show",
        "shows", "outcome", "outcomes", "result", "results",
    }
    new_words: List[str] = []
    for word in re.findall(r"\b[A-Za-z][A-Za-z-]{3,}\b", candidate_sentence):
        lower = word.lower()
        if lower in stopwords:
            continue
        if lower in anchor_set:
            continue
        if lower in allowed_text:
            continue
        stem = _rough_stem(lower)
        if stem in allowed_stems or stem in anchor_stems:
            continue
        if lower in risky_new_words or len(lower) >= 7:
            if lower not in new_words:
                new_words.append(lower)
        if len(new_words) >= limit:
            break
    return new_words


def _concrete_observation_count(text: str) -> int:
    patterns = [
        r"\bI\b", r"\bmy\b", r"\bwe\b", r"\bour\b",
        r"\bin practice\b", r"\bin my context\b", r"\bin class\b",
        r"\bfor example\b", r"\bfor instance\b", r"\bI notice\b",
        r"\bI see\b", r"\bI usually\b", r"\bwhen\b.+\bthen\b",
        r"\b\d+(?:\.\d+)?\b",
    ]
    return sum(len(re.findall(pattern, text, re.I)) for pattern in patterns)


def _generic_polish_count(text: str) -> int:
    patterns = [
        r"\bcrucial\b", r"\bvital\b", r"\bessential\b", r"\bsignificant\b",
        r"\bnotable\b", r"\bundeniably\b", r"\bincreasingly\b",
        r"\bmodern world\b", r"\bplays? a key role\b",
        r"\bdemonstrates?\b", r"\bhighlights?\b", r"\bunderscores?\b",
        r"\bto address modern\b", r"\bto thrive amidst\b",
        r"\bnot enough\b", r"\bcan be effective\b", r"\btakes time\b",
        r"\bdeconstructed\b", r"\bmimicry\b", r"\btransform(?:s|ing)?\b",
        r"\bboost(?:s|ing)? (?:their )?(?:confidence|motivation)\b",
        r"\bserves as a practical model\b",
        r"\bserves as (?:a|an|the) (?:specific )?(?:case|model|example)\b",
        r"\bsteep operational\b", r"\boperational (?:trial|obstacles?)\b",
        r"\bencounter complex\b", r"\bmonopoly\b", r"\bdissolved\b",
        r"\bexclusive gateway\b", r"\brepository\b",
        r"\bnavigating projection\b", r"\bgeometric outcomes\b",
        r"\belevations of the hair subsection\b",
        r"\bemotional labor\b", r"\btechnical oversight\b",
        r"\bdilute the frequency\b", r"\bvisible learning framework\b",
        r"\bpreserves technical rigor\b", r"\bendless online tutorials\b",
        r"\bdigital landscape\b", r"\bsaturate(?:s|d)?\b",
        r"\bprofessional precision\b", r"\bboost needed to perform\b",
        r"\bmaster the exact\b", r"\bpresents? a constant hurdle\b",
        r"\bresearch backing\b",
        r"\bembedded within\b",
        r"\bespecially evident\b",
        r"\bespecially clear\b",
        r"\bexecut(?:e|es|ed|ing)\b",
        r"\bobserv(?:e|es|ed|ing) a demonstration\b",
        r"\bshift(?:s|ed|ing)? from observing\b",
        r"\bfrom my experience teaching\b",
        r"\bpart of how technical skills are taught\b",
        r"\bconstructing\b.{0,60}\brequires\b",
        r"\blearners?\s+(?:frequently|often|commonly)\s+(?:fail|struggle)\b",
        r"\btoward precise\b", r"\btoward professional\b",
        r"\bbasic mimicry\b", r"\bbuilding the confidence they need\b",
        r"\btaking\b.{0,80}\bas a case\b",
        r"\bguide(?:s|d|ing)? the cut\b",
        r"\bchosen degree creates\b", r"\bstacked silhouette\b",
        r"\bworks? in practice\b",
        r"\breveal(?:s|ed|ing)? how this relationship works\b",
        r"\bthis relationship works\b",
    ]
    lower = text.lower()
    return sum(len(re.findall(p, lower)) for p in patterns)


def _plain_voice_register_drift(original_sentence: str, candidate_sentence: str) -> str:
    """Detect candidates that upgrade plain student voice into textbook prose."""
    original_lower = original_sentence.lower()
    candidate_lower = candidate_sentence.lower()
    plain_markers = [
        r"\bstudents?\b", r"\bteacher(?:s)?\b", r"\bI\b", r"\bwe\b",
        r"\bmake\b", r"\buse\b", r"\bhelp\b", r"\bvery difficult\b",
        r"\busually\b", r"\bcan\b", r"\bshould\b", r"\bin my\b",
    ]
    plain_source = sum(bool(re.search(p, original_sentence, re.I)) for p in plain_markers) >= 2
    if not plain_source:
        return ""

    upgraded_terms = [
        "learners", "constructing", "requires", "frequently fail",
        "professional precision", "basic mimicry", "serves as a case",
        "as a case", "stacked silhouette", "chosen degree creates",
        "toward precise", "toward professional", "facilitates",
    ]
    introduced = [
        term for term in upgraded_terms
        if term in candidate_lower and term not in original_lower
    ]
    if introduced:
        return "plain_voice_register_drift: " + ", ".join(introduced[:3])

    if "student" in original_lower and "learner" in candidate_lower and "learner" not in original_lower:
        return "plain_voice_register_drift: student->learner"
    return ""


def _candidate_style_reject_reason(original_sentence: str, candidate_sentence: str) -> str:
    """Reject detector-gaming rewrites that sound more polished/abstract."""
    orig_polish = _generic_polish_count(original_sentence)
    cand_polish = _generic_polish_count(candidate_sentence)
    if cand_polish > orig_polish:
        return f"polished_generic_drift {orig_polish}->{cand_polish}"

    register_reason = _plain_voice_register_drift(original_sentence, candidate_sentence)
    if register_reason:
        return register_reason

    # Long noun stacks often reduce GPT-2 predictability while increasing AI
    # style risk. Keep sentence edits plain unless the source sentence already
    # uses dense nominal phrasing.
    noun_stack = re.search(
        r"\b(?:technical|operational|instructional|cognitive|practical|formal|"
        r"individualized|specialized|complex|digital|geometric|visible)\s+"
        r"(?:accuracy|trial|model|concepts?|corrections?|structures?|guidance|"
        r"implementation|engagement|obstacles?|oversight|rigor|landscape|"
        r"outcomes?|framework)\b",
        candidate_sentence,
        re.I,
    )
    if noun_stack and not re.search(re.escape(noun_stack.group(0)), original_sentence, re.I):
        return f"new_abstract_noun_stack '{noun_stack.group(0)}'"

    return ""


def _candidate_quality_score(
    original_sentence: str,
    candidate_sentence: str,
    original_paragraph: str,
    candidate_paragraph: str,
    predictability_delta: float,
    drift_similarity: float,
    domain_anchor_terms: Optional[List[str]] = None,
) -> float:
    """Higher is better. Rewards local signal improvement without polished generic drift."""
    domain_anchor_terms = domain_anchor_terms or []
    orig_generic = _generic_polish_count(original_paragraph)
    cand_generic = _generic_polish_count(candidate_paragraph)
    generic_penalty = max(0, cand_generic - orig_generic) * 0.25
    orig_anchor_coverage = _term_coverage(original_sentence, domain_anchor_terms)
    cand_anchor_coverage = _term_coverage(candidate_sentence, domain_anchor_terms)
    anchor_gain = max(0, cand_anchor_coverage - orig_anchor_coverage)
    orig_observation = _concrete_observation_count(original_sentence)
    cand_observation = _concrete_observation_count(candidate_sentence)
    observation_gain = max(0, cand_observation - orig_observation)
    concreteness_bonus = min(0.30, (anchor_gain * 0.06) + (observation_gain * 0.04))
    length_ratio = len(candidate_sentence) / max(len(original_sentence), 1)
    length_tolerance = 0.45 if domain_anchor_terms else 0.20
    length_penalty = max(0.0, abs(length_ratio - 1.0) - length_tolerance) * 0.25
    improvement = max(0.0, predictability_delta)
    return round(
        (0.55 * improvement)
        + (0.25 * drift_similarity)
        + concreteness_bonus
        - generic_penalty
        - length_penalty,
        4,
    )


MEDIUM_PREDICTABILITY_CEILING = 0.45
MIN_TARGET_PREDICTABILITY_DELTA = 0.025
MIN_TARGET_PREDICTABILITY_RELATIVE_DELTA = 0.06


def _requires_medium_exit(finding: Finding) -> bool:
    """True when a medium target should be accepted only below medium threshold."""
    return (
        finding.risk_level == "medium"
        and (
            "predictability" in finding.finding_type
            or "topk" in finding.finding_type
            or "surprisal" in finding.finding_type
        )
    )


def _target_predictability_acceptance(
    original_pred: Dict[str, Any],
    candidate_pred: Dict[str, Any],
) -> tuple[bool, str, float]:
    """Accept clear mitigation, not only full band exit.

    A final full-scan gate still decides whether the edit is kept globally. This
    local gate should reject regressions and tiny/no-op edits, but it should not
    throw away a measurable target reduction just because one sentence remains
    slightly inside the medium band.
    """
    orig_risk = original_pred.get("risk")
    new_risk = candidate_pred.get("risk")
    new_label = str(candidate_pred.get("label") or "")
    if orig_risk is None or new_risk is None:
        return False, f"target_predictability_unavailable {new_label or '?'}:{new_risk}", 0.0
    try:
        orig_val = float(orig_risk)
        new_val = float(new_risk)
    except (TypeError, ValueError):
        return False, f"target_predictability_unavailable {new_label or '?'}:{new_risk}", 0.0
    delta = orig_val - new_val
    relative_delta = delta / max(orig_val, 0.001)
    if new_val >= orig_val:
        return False, f"target_not_improved {orig_val:.4f}->{new_val:.4f}", delta
    if new_val < MEDIUM_PREDICTABILITY_CEILING and new_label not in {"medium", "high"}:
        return True, "", delta
    if (
        new_label != "high"
        and delta >= MIN_TARGET_PREDICTABILITY_DELTA
        and relative_delta >= MIN_TARGET_PREDICTABILITY_RELATIVE_DELTA
    ):
        return True, "target_reduced_but_still_medium", delta
    return (
        False,
        f"target_reduction_too_small {new_label or '?'}:{new_val:.4f} delta:{delta:.4f}",
        delta,
    )


def _sentence_predictability(scanner: PredictabilityScanner, sentence: str) -> Dict[str, Any]:
    """Return target-sentence predictability score and label."""
    try:
        result = scanner.scan_sentence(sentence)
        return {
            "risk": float(getattr(result, "predictability_risk", 0.0) or 0.0),
            "label": getattr(result, "risk_label", ""),
        }
    except Exception as exc:
        logger.warning("Target sentence predictability check failed: %s", exc)
        return {"risk": None, "label": ""}


def _enrich_span_info(finding: Finding, rewrite_context: Optional[Any], sent_idx: int) -> str:
    """Build signal-specific targeting instructions from finding + detect data.

    This replaces the generic "Fix strategy: suggest_rewrite" with concrete
    trigger metrics so the LLM knows exactly what's wrong and what to change.
    """
    if not rewrite_context or not hasattr(rewrite_context, "raw_json"):
        return ""
    raw = rewrite_context.raw_json or {}
    parts = []

    # ── Predictability signal enrichment ──
    if finding.finding_type in ("high_predictability", "medium_predictability"):
        pred_data = raw.get("predictability", {})
        for sent in pred_data.get("sentences", []):
            if _matches_sentence(sent, finding, sent_idx):
                top10 = sent.get("top10_ratio") or sent.get("top_10_ratio") or sent.get("top10")
                score = sent.get("score")
                avg_surp = sent.get("avg_surprisal")
                threshold = sent.get("threshold")
                if top10 is not None:
                    top10_pct = top10 if top10 > 1 else top10 * 100
                    parts.append(f"  Trigger: top-10 predictability = {top10_pct:.1f}%")
                    if threshold is not None:
                        thr_pct = threshold if threshold > 1 else threshold * 100
                        parts.append(f"  (threshold: {thr_pct:.1f}%)")
                if score is not None and isinstance(score, (int, float)):
                    parts.append(f"  Predictability score: {score:.3f}")
                if avg_surp is not None and isinstance(avg_surp, (int, float)):
                    parts.append(f"  Avg surprisal: {avg_surp:.2f} (lower = more predictable)")
                flagged_tokens = _extract_flagged_tokens(sent)
                if flagged_tokens:
                    parts.append(f"  Problem tokens: {flagged_tokens}")
                break

    # ── TopK pattern enrichment ──
    elif finding.finding_type in ("high_topk_predictability", "low_surprisal", "low_surprisal_pattern"):
        pred_data = raw.get("predictability", {})
        for sent in pred_data.get("sentences", []):
            if _matches_sentence(sent, finding, sent_idx):
                top10 = sent.get("top10_ratio") or sent.get("top_10_ratio") or sent.get("top10")
                top50 = sent.get("top50_ratio") or sent.get("top_50_ratio") or sent.get("top50")
                avg_surp = sent.get("avg_surprisal")
                if top10 is not None:
                    top10_pct = top10 if top10 > 1 else top10 * 100
                    parts.append(f"  Top-10 ratio: {top10_pct:.1f}%")
                if top50 is not None:
                    top50_pct = top50 if top50 > 1 else top50 * 100
                    parts.append(f"  Top-50 ratio: {top50_pct:.1f}%")
                if avg_surp is not None and isinstance(avg_surp, (int, float)):
                    parts.append(f"  Avg surprisal: {avg_surp:.2f}")
                flagged_tokens = _extract_flagged_tokens(sent)
                if flagged_tokens:
                    parts.append(f"  Most predictable tokens: {flagged_tokens}")
                break

    # ── Formulaic sentence enrichment ──
    elif finding.finding_type in ("formulaic_sentence", "generic_phrase", "generic_formulaic_language"):
        meta = finding.metadata or {}
        matched_patterns = meta.get("matched_patterns", [])
        if matched_patterns:
            parts.append(f"  Matched AI patterns: {', '.join(str(p) for p in matched_patterns[:5])}")
        formula_score = meta.get("formula_score")
        if formula_score is not None and isinstance(formula_score, (int, float)):
            parts.append(f"  Formulaicity score: {formula_score:.3f}")
        parts.append("  Strategy: Break the formula by restructuring sentence opening and clause order")

    # ── Style shift enrichment ──
    elif finding.finding_type in ("style_shift", "repetitive_structure"):
        meta = finding.metadata or {}
        shift_type = meta.get("shift_type", "")
        if shift_type:
            parts.append(f"  Shift type: {shift_type}")
        parts.append("  Strategy: Vary sentence opening from previous sentence's pattern")

    # ── Burstiness enrichment ──
    elif finding.finding_type == "burstiness":
        meta = finding.metadata or {}
        neighbor_avg = meta.get("neighbor_avg_length")
        this_len = meta.get("sentence_length")
        if this_len and neighbor_avg:
            parts.append(f"  This sentence: {this_len} chars, neighbors avg: {neighbor_avg} chars")
        parts.append("  Strategy: Make noticeably shorter or merge with context for rhythm variety")

    # ── AI generation enrichment ──
    elif finding.finding_type == "ai_generation":
        detail = finding.detail or ""
        if detail:
            parts.append(f"  Root signal: {detail}")
        meta = finding.metadata or {}
        components = meta.get("badge_components", {})
        if components:
            top_signals = sorted(components.items(), key=lambda x: x[1], reverse=True)[:3]
            signal_str = ", ".join(f"{k}={v:.0%}" for k, v in top_signals if isinstance(v, (int, float)))
            if signal_str:
                parts.append(f"  Top AI signals: {signal_str}")

    # ── Transition / enumeration enrichment ──
    elif finding.finding_type in ("mechanical_transition", "generic_enumeration"):
        meta = finding.metadata or {}
        transition_word = meta.get("transition_word") or meta.get("enumerator")
        if transition_word:
            parts.append(f"  Flagged marker: '{transition_word}'")
        parts.append("  Strategy: Replace with natural connective from context, or remove if redundant")

    # ── Vague claim enrichment ──
    elif finding.finding_type == "vague_claim":
        meta = finding.metadata or {}
        specificity_score = meta.get("specificity_score")
        if specificity_score is not None:
            parts.append(f"  Specificity score: {specificity_score:.3f} (low = vague)")
        parts.append("  Strategy: Make the claim more specific using details from surrounding text")

    # ── Similarity enrichment ──
    elif finding.finding_type in ("similarity_overlap", "close_paraphrase", "patchwriting", "semantic_overlap"):
        meta = finding.metadata or {}
        sim_score = meta.get("similarity_score") or meta.get("overlap_ratio")
        if sim_score is not None and isinstance(sim_score, (int, float)):
            parts.append(f"  Overlap score: {sim_score:.3f}")
        parts.append("  Strategy: Rephrase to express the same idea in your own structure and wording")

    return "\n".join(parts)


def _component_regression_check(original: str, candidate: str) -> tuple[bool, str]:
    """Cheap guard for document-level AI badge drivers.

    Full detection after every sentence is too expensive, but these Layer 3
    component estimators are lightweight and catch the common failure mode:
    a rewrite gets smoother while becoming broader or less grounded.
    """
    try:
        from detect.layer3_scoring import (
            estimate_broad_claim_risk,
            estimate_generic_assertion_risk,
            estimate_lived_detail_risk,
            estimate_unsupported_claim_risk,
        )
    except Exception:
        return True, ""

    checks = [
        ("broad_claim", estimate_broad_claim_risk),
        ("generic_assertion", estimate_generic_assertion_risk),
        ("lived_detail", estimate_lived_detail_risk),
        ("unsupported_claim", estimate_unsupported_claim_risk),
    ]
    regressions = []
    for name, fn in checks:
        before = fn(original)
        after = fn(candidate)
        if after > before + 0.02:
            regressions.append(f"{name} {before:.3f}->{after:.3f}")
    if regressions:
        return False, "; ".join(regressions)
    return True, ""


def _rewrite_fn_with_detect_context(
    detect_context: str,
    api_key: Optional[str],
    model: str,
    base_url: Optional[str] = None,
    timeout: int = 20,
    max_retries: int = 1,
) -> callable:
    """Create a rewrite function that uses LLMGateway with detect context."""
    config = LLMConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    gateway = LLMGateway(config)

    def rewrite_fn(text: str, span_info: str) -> Optional[str]:
        max_chars = int(len(text) * 1.40)
        wants_candidates = "Return exactly 3 candidates" in span_info or "Return exactly 6 candidates" in span_info
        output_instruction = (
            "Return numbered replacement candidates and no commentary."
            if wants_candidates else
            "Output ONLY the rewritten text. No quotes, no commentary."
        )
        user_msg = (
            f"{detect_context}\n\n{span_info}\n\n"
            f"Current text ({len(text)} chars):\n{text}\n\n"
            f"Generate detector-safe replacement text for the marked target only. "
            f"This is risk mitigation, not prose improvement. "
            f"CRITICAL: Respect any stricter character limit in the task; otherwise do not exceed {max_chars} characters. "
            f"Preserve facts, scope, author voice, and detail level. "
            f"Do not add unsupported methods, examples, causes, benefits, motivations, conclusions, citations, or source links. "
            f"{output_instruction}"
        )
        try:
            resp = gateway.chat(user_msg, system=REWRITE_CHIPIN_PROMPT)
            if resp.is_empty:
                return None
            output = resp.content.strip()
            # Strip triple quotes/code blocks the LLM may wrap output in
            if output.startswith('"""') and output.endswith('"""'):
                output = output[3:-3].strip()
            elif output.startswith("```") and output.endswith("```"):
                output = output[3:-3].strip()
            return output
        except Exception:
            return None

    return rewrite_fn


# ── Main entry point ─────────────────────────────────────────────────

# ── Per-finding rewrite helpers ─────────────────────────────────────

def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs on blank lines."""
    import re
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences preserving whitespace structure."""
    import re
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
    return [p.strip() for p in parts if p.strip()]


def _build_sentence_index(text: str) -> tuple:
    """Split text into sentences with paragraph boundaries preserved.

    Returns (sentences, para_map) where para_map[sent_idx] = paragraph index.
    """
    paragraphs = _split_paragraphs(text)
    sentences = []
    para_map = {}
    for p_idx, para in enumerate(paragraphs):
        sents = _split_sentences(para)
        for s in sents:
            para_map[len(sentences)] = p_idx
            sentences.append(s)
    return sentences, para_map


def _badge_component_score(raw_json: Optional[Dict[str, Any]], component: str) -> float:
    """Return a badge component as a percentage-style score."""
    if not isinstance(raw_json, dict):
        return 0.0
    badge = raw_json.get("ai_risk_badge") or {}
    if not isinstance(badge, dict):
        return 0.0
    sections = [
        badge.get("ai_components") or {},
        badge.get("writing_components") or {},
        badge,
    ]
    for section in sections:
        if not isinstance(section, dict):
            continue
        value = section.get(component)
        if isinstance(value, (int, float)):
            value = float(value)
            return value * 100.0 if 0.0 <= value <= 1.0 else value
    return 0.0


def _report_badge_score(report_dict: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    badge = (report_dict or {}).get("ai_risk_badge") or {}
    value = badge.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _full_scan_report_dict_for_rewrite_gate(scan_text: str) -> Dict[str, Any]:
    runner = DetectionRunner()
    detect_report = runner.run_all(scan_text)
    builder = ReportBuilder()
    builder.add_detection_report(detect_report)
    if detect_report.postprocess_results:
        builder.add_postprocess_results(detect_report.postprocess_results)
    builder.set_meta(scan_time=0, original_text=scan_text)
    return report_to_dict(builder.build())


def _iter_raw_rewrite_items(raw_json: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect likely finding/brief/plan rows from detect JSON."""
    if not isinstance(raw_json, dict):
        return []
    items: List[Dict[str, Any]] = []
    for key in ("rewrite_edit_briefs", "rewrite_briefs", "rewrite_targets"):
        rows = raw_json.get(key) or []
        if isinstance(rows, list):
            items.extend(row for row in rows if isinstance(row, dict))

    findings = raw_json.get("findings") or {}
    if isinstance(findings, dict):
        for rows in findings.values():
            if isinstance(rows, list):
                items.extend(row for row in rows if isinstance(row, dict))
    elif isinstance(findings, list):
        items.extend(row for row in findings if isinstance(row, dict))

    plan = raw_json.get("rewrite_plan") or raw_json.get("rewrite_strategy") or {}
    if isinstance(plan, dict):
        for key in (
            "auto_target_context",
            "manual_required",
            "review_only",
            "suggestion_only",
            "marked_content_suggestions",
        ):
            rows = plan.get(key) or []
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict):
                    items.append(row)
                    finding = row.get("finding")
                    if isinstance(finding, dict):
                        items.append(finding)
    return items


def _raw_item_text_values(item: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    keys = (
        "paragraph_excerpt",
        "target_text",
        "target_sentence",
        "original_sentence",
        "evidence",
        "detail",
    )
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    for nested_key in ("rewrite_context", "context", "signals"):
        nested = item.get(nested_key)
        if isinstance(nested, dict):
            for key in keys:
                value = nested.get(key)
                if isinstance(value, str) and value.strip():
                    values.append(value.strip())
    return values


def _text_overlap_score(needle: str, haystack: str) -> float:
    needle_norm = _normalize_sentence_match_text(needle).lower().replace("...", "")
    haystack_norm = _normalize_sentence_match_text(haystack).lower()
    if not needle_norm or not haystack_norm:
        return 0.0
    if len(needle_norm) >= 40 and needle_norm[:80] in haystack_norm:
        return 4.0
    needle_tokens = set(_match_tokens(needle_norm))
    hay_tokens = set(_match_tokens(haystack_norm))
    if not needle_tokens or not hay_tokens:
        return 0.0
    overlap = len(needle_tokens & hay_tokens)
    return overlap / max(len(needle_tokens), 1)


def _density_region_score(region: str, items: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    word_count = len(re.findall(r"\b\w+\b", region))
    score = min(word_count / 85.0, 2.0)
    meta: Dict[str, Any] = {"word_count": word_count, "matched_items": 0}
    region_lower = region.lower()
    for item in items:
        item_score = 0.0
        for value in _raw_item_text_values(item):
            item_score = max(item_score, _text_overlap_score(value, region))
        item_text = " ".join(str(item.get(k, "")) for k in ("finding_type", "action_type", "title", "detail"))
        if re.search(r"density|predictab|generic|unsupported|source|ground", item_text, re.I):
            item_score += 0.35
        if item_score >= 0.55:
            meta["matched_items"] += 1
            score += min(item_score, 4.0)

    generic_count = _generic_polish_count(region)
    if generic_count:
        score += min(generic_count * 0.5, 2.0)
        meta["generic_polish_count"] = generic_count
    if re.search(r"\b(?:important|significant|essential|crucial|modern|today|overall)\b", region_lower):
        score += 0.35
    return score, meta


def _density_region_candidates(
    paragraph: str,
    paragraph_index: int,
    items: List[Dict[str, Any]],
    max_words: int = 320,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return bounded density rewrite regions from a paragraph or long section."""
    word_count = len(re.findall(r"\b\w+\b", paragraph))
    if word_count <= max_words:
        score, meta = _density_region_score(paragraph, items)
        meta.update({
            "paragraph_index": paragraph_index,
            "region_type": "paragraph",
            "score": round(score, 3),
        })
        return [(paragraph, meta)]

    sentences = _split_sentences(paragraph)
    if len(sentences) < 3:
        clipped_words = paragraph.split()[:max_words]
        region = " ".join(clipped_words)
        score, meta = _density_region_score(region, items)
        meta.update({
            "paragraph_index": paragraph_index,
            "region_type": "word_window",
            "score": round(score, 3),
            "source_word_count": word_count,
        })
        return [(region, meta)]

    candidates: List[Tuple[str, Dict[str, Any]]] = []
    target_words = min(max_words, 220)
    for start in range(len(sentences)):
        window: List[str] = []
        total_words = 0
        for sentence in sentences[start:]:
            sent_words = len(sentence.split())
            if window and total_words + sent_words > max_words:
                break
            window.append(sentence)
            total_words += sent_words
            if total_words >= target_words:
                break
        if total_words < 45:
            continue
        region = " ".join(window)
        score, meta = _density_region_score(region, items)
        meta.update({
            "paragraph_index": paragraph_index,
            "region_type": "sentence_window",
            "sentence_start": start,
            "sentence_count": len(window),
            "score": round(score, 3),
            "source_word_count": word_count,
        })
        candidates.append((region, meta))
    return candidates


def _select_density_paragraph(
    text: str,
    rewrite_context: Optional[Any] = None,
    mitigation_plan: Optional[Dict[str, Any]] = None,
    excluded_regions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[int, str, Dict[str, Any]]:
    """Pick the bounded paragraph/section region most responsible for density risk."""
    raw_json = getattr(rewrite_context, "raw_json", None) if rewrite_context else None
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return -1, "", {}

    items = _iter_raw_rewrite_items(raw_json)
    for suggestion in (mitigation_plan or {}).get("marked_content_suggestions") or []:
        if isinstance(suggestion, dict):
            items.append(suggestion)

    best_idx = -1
    best_score = -1.0
    best_meta: Dict[str, Any] = {}
    best_region = ""
    for idx, paragraph in enumerate(paragraphs):
        for region, meta in _density_region_candidates(paragraph, idx, items):
            if meta.get("word_count", 0) < 45:
                continue
            if _density_region_is_excluded(meta, excluded_regions or []):
                continue
            score = float(meta.get("score") or 0.0)
            if score <= best_score:
                continue
            best_idx = idx
            best_score = score
            best_meta = meta
            best_region = region
    if best_idx < 0:
        candidates = [(i, p) for i, p in enumerate(paragraphs) if len(p.split()) >= 35]
        if not candidates:
            return -1, "", {}
        best_idx, best_para = max(candidates, key=lambda pair: len(pair[1].split()))
        bounded = _density_region_candidates(best_para, best_idx, items, max_words=220)
        if bounded:
            best_region, best_meta = bounded[0]
        else:
            best_region, best_meta = best_para, {"word_count": len(best_para.split())}
        best_meta.update({"fallback": "longest_paragraph"})
        return best_idx, best_region, best_meta
    return best_idx, best_region, best_meta


def _density_region_is_excluded(meta: Dict[str, Any], excluded_regions: List[Dict[str, Any]]) -> bool:
    paragraph_index = meta.get("paragraph_index")
    region_type = meta.get("region_type")
    if paragraph_index is None:
        return False
    for excluded in excluded_regions:
        if excluded.get("paragraph_index") != paragraph_index:
            continue
        if region_type == "sentence_window" and excluded.get("region_type") == "sentence_window":
            start = int(meta.get("sentence_start") or 0)
            count = int(meta.get("sentence_count") or 0)
            end = start + count
            ex_start = int(excluded.get("sentence_start") or 0)
            ex_count = int(excluded.get("sentence_count") or 0)
            ex_end = ex_start + ex_count
            if start < ex_end and ex_start < end:
                return True
            continue
        if region_type == excluded.get("region_type"):
            return True
    return False


def _splice_density_candidate(
    current_text: str,
    paragraph_index: int,
    source_region: str,
    candidate_region: str,
    region_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Replace only the selected density region, not the whole document.

    Density mitigation can select a normalized sentence window from an oversized
    paragraph. If that normalized window is not an exact substring, replacing the
    whole paragraph would drop the rest of the document.
    """
    if not current_text or not candidate_region:
        return ""
    if source_region and source_region in current_text:
        return current_text.replace(source_region, candidate_region, 1)

    paragraphs = _split_paragraphs(current_text)
    if paragraph_index < 0 or paragraph_index >= len(paragraphs):
        return ""

    meta = region_meta or {}
    paragraph = paragraphs[paragraph_index]
    if meta.get("region_type") == "sentence_window":
        sentences = _split_sentences(paragraph)
        start = int(meta.get("sentence_start") or 0)
        count = int(meta.get("sentence_count") or 0)
        if sentences and count > 0 and 0 <= start < len(sentences):
            end = min(len(sentences), start + count)
            rewritten_sentences = _split_sentences(candidate_region) or [candidate_region]
            new_sentences = sentences[:start] + rewritten_sentences + sentences[end:]
            paragraphs[paragraph_index] = " ".join(s.strip() for s in new_sentences if s.strip())
            return "\n\n".join(paragraphs)

    if meta.get("region_type") == "word_window":
        source_word_count = len((source_region or "").split())
        if source_word_count > 0:
            words = paragraph.split()
            paragraphs[paragraph_index] = " ".join(
                [candidate_region] + words[source_word_count:]
            )
            return "\n\n".join(paragraphs)

    if meta.get("region_type") == "paragraph":
        paragraphs[paragraph_index] = candidate_region
        return "\n\n".join(paragraphs)

    return ""


def _density_component_lines(raw_json: Optional[Dict[str, Any]]) -> List[str]:
    names = [
        "qualifying_text_ai_density",
        "generic_assertion_risk",
        "broad_claim_risk",
        "unsupported_claim_risk",
        "source_grounding_risk",
        "lived_detail_risk",
    ]
    lines = []
    for name in names:
        value = _badge_component_score(raw_json, name)
        if value > 0:
            lines.append(f"{name}={value:.1f}%")
    return lines


def _density_paragraph_prompt(
    paragraph: str,
    rewrite_context: Optional[Any],
    mitigation_plan: Optional[Dict[str, Any]],
) -> str:
    raw_json = getattr(rewrite_context, "raw_json", None) if rewrite_context else None
    domain_terms = _domain_anchor_terms(rewrite_context, paragraph, paragraph, limit=12)
    protected = [span.text or paragraph[span.start_char:span.end_char] for span in detect_protected_spans(paragraph)]
    named_entities = sorted(_extract_named_entities(paragraph))
    component_lines = _density_component_lines(raw_json)
    marked = []
    for suggestion in (mitigation_plan or {}).get("marked_content_suggestions") or []:
        if isinstance(suggestion, dict) and suggestion.get("action_type") in {
            "rebuild_paragraph_density",
            "paragraph_density_rebuild",
        }:
            marked.append(str(suggestion.get("why_it_helps") or suggestion.get("instruction") or suggestion.get("title") or ""))
    lines = [
        "Density paragraph mitigation pass.",
        "Rewrite exactly this one paragraph, not the whole document.",
        "Goal: reduce dense AI-style signal by changing paragraph evidence flow, not by polishing vocabulary.",
        "Use sentence total reconstruction: discard the original syntax and rebuild the paragraph from the meaning.",
        "Keep the same facts, scope, and plain author voice, but do not keep the same sentence route or clause order.",
        "Keep roughly the same coverage. You may split, merge, shorten, or reorder sentences when the meaning remains intact.",
        "Change at least 70% of sentence openings and linking phrases unless they are protected headings, names, citations, or unit codes.",
        "Prefer concrete classroom/process wording: what the learner sees, does, waits for, checks, repeats, or corrects.",
        "If the paragraph already uses I/my, keep that natural first-person stance where useful; otherwise do not invent personal experience.",
        "Do not add citations, names, dates, research claims, examples, procedures, or facts that are not already in the paragraph.",
        "Do not add, remove, rename, shorten, or paraphrase institution names, course names, people names, or place names.",
        "Prefer concrete words already present in the paragraph or nearby domain anchors.",
        "Avoid generic academic phrasing: crucial, significant, essential, landscape, framework, technical rigor, operational obstacles, visible learning framework, digital landscape.",
        "Also avoid formal rebuild phrasing such as embedded within, especially evident, especially true, from my experience teaching, executing, or observing a demonstration.",
        "Avoid polished substitution words that keep the same footprint: challenge, phase, setting, influence, must, requires, addresses, supports, facilitates, enables, approach, solution.",
        "Avoid abstract noun stacks and broad claims. Do not make the paragraph smoother if that removes specificity.",
        "Output exactly one replacement paragraph. No numbering, bullets, quotes, headings, or commentary.",
        "Target paragraph:\n<TARGET>\n" + paragraph + "\n</TARGET>",
    ]
    if component_lines:
        lines.append("Scan component drivers: " + ", ".join(component_lines))
    if domain_terms:
        lines.append("Domain anchors to preserve when natural: " + ", ".join(domain_terms))
    if protected:
        lines.append("Protected spans that must remain unchanged: " + "; ".join(protected[:12]))
    if named_entities:
        lines.append("Named entities that must remain unchanged: " + "; ".join(named_entities[:12]))
    if marked:
        lines.append("Scanner mitigation guidance: " + " ".join(m for m in marked if m)[:600])
    constraint_lines = _rewrite_constraint_lines(rewrite_context, limit=4)
    if constraint_lines:
        lines.extend(constraint_lines)
    return "\n".join(lines)


def _clean_density_paragraph_output(output: Optional[str], original_paragraph: str) -> str:
    if not output:
        return ""
    text = output.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()
    text = re.sub(r"^(?:rewritten|replacement)\s+paragraph\s*:\s*", "", text, flags=re.I).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        numbered = []
        for line in lines:
            match = re.match(r"^(?:[\(\[]?\d+[\)\].:-]|[-*])\s*(.+)$", line)
            numbered.append(match.group(1).strip() if match else line)
        text = " ".join(numbered)
    else:
        text = lines[0] if lines else text
    text = text.strip().strip('"').strip("'").strip()
    text = " ".join(text.split())
    if text == _normalize_sentence_match_text(original_paragraph):
        return ""
    return text


def _density_transformation_too_small(original_paragraph: str, candidate_paragraph: str) -> bool:
    """Detect density rewrites that only polish the same paragraph footprint."""
    orig_tokens = _match_tokens(original_paragraph.lower())
    cand_tokens = _match_tokens(candidate_paragraph.lower())
    if len(orig_tokens) < 30 or len(cand_tokens) < 25:
        return False

    token_ratio = SequenceMatcher(
        None,
        " ".join(orig_tokens),
        " ".join(cand_tokens),
        autojunk=False,
    ).ratio()
    if token_ratio >= 0.86:
        return True

    def _starts(text: str) -> List[str]:
        starts: List[str] = []
        for sentence in _split_sentences(text):
            tokens = _match_tokens(sentence.lower())
            if len(tokens) >= 5:
                starts.append(" ".join(tokens[:5]))
        return starts

    orig_starts = _starts(original_paragraph)
    cand_starts = set(_starts(candidate_paragraph))
    if len(orig_starts) >= 4 and cand_starts:
        shared = sum(1 for start in orig_starts if start in cand_starts)
        if shared >= max(3, int(len(orig_starts) * 0.55)):
            return True
    return False


def _density_predictability_signal(
    scanner: PredictabilityScanner,
    text: str,
) -> Dict[str, Any]:
    try:
        metrics = compute_metrics(text, scanner)
        return {
            "risk": round(float(getattr(metrics, "risk", 0.0) or 0.0), 4),
            "top10": round(float(getattr(metrics, "top10_ratio", 0.0) or 0.0), 4),
            "surprisal": round(float(getattr(metrics, "surprisal", 0.0) or 0.0), 4),
        }
    except Exception as exc:
        logger.warning("Density local predictability check failed: %s", exc)
        return {"risk": None, "top10": None, "surprisal": None, "error": str(exc)}


def _density_local_signal_acceptance(
    scanner: PredictabilityScanner,
    original_paragraph: str,
    candidate_paragraph: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Require density paragraph edits to improve the local detector proxy."""
    original = _density_predictability_signal(scanner, original_paragraph)
    candidate = _density_predictability_signal(scanner, candidate_paragraph)
    signal = {"original": original, "candidate": candidate}
    if original.get("risk") is None or candidate.get("risk") is None:
        return True, "", signal

    orig_risk = float(original.get("risk") or 0.0)
    cand_risk = float(candidate.get("risk") or 0.0)
    orig_top10 = float(original.get("top10") or 0.0)
    cand_top10 = float(candidate.get("top10") or 0.0)
    risk_delta = orig_risk - cand_risk
    top10_delta = orig_top10 - cand_top10
    signal.update({
        "risk_delta": round(risk_delta, 4),
        "top10_delta": round(top10_delta, 4),
    })

    if cand_risk > orig_risk + 0.01 or cand_top10 > orig_top10 + 0.02:
        return (
            False,
            f"density_local_signal_regressed risk:{orig_risk:.4f}->{cand_risk:.4f} top10:{orig_top10:.4f}->{cand_top10:.4f}",
            signal,
        )
    if risk_delta >= 0.025 or top10_delta >= 0.045:
        return True, "", signal
    if risk_delta >= 0.006 and top10_delta >= -0.004:
        return True, "density_local_risk_improved", signal
    if top10_delta >= 0.010 and risk_delta >= -0.004:
        return True, "density_local_top10_improved", signal
    if risk_delta >= 0.004 and top10_delta >= 0.004:
        return True, "", signal
    return (
        False,
        f"density_local_signal_not_improved risk:{orig_risk:.4f}->{cand_risk:.4f} top10:{orig_top10:.4f}->{cand_top10:.4f}",
        signal,
    )


def _density_paragraph_reject_reason(
    original_paragraph: str,
    candidate_paragraph: str,
    rewrite_context: Optional[Any] = None,
    allow_polish_warning: bool = False,
) -> str:
    if not candidate_paragraph:
        return "empty_candidate"
    orig_words = len(original_paragraph.split())
    cand_words = len(candidate_paragraph.split())
    if orig_words and (cand_words < orig_words * 0.65 or cand_words > orig_words * 1.35):
        return f"length_ratio {orig_words}->{cand_words}"
    protected = detect_protected_spans(original_paragraph)
    if protected and not protected_spans_preserved(original_paragraph, candidate_paragraph, protected):
        return "protected_span_lost"
    drift = check_semantic_drift(original_paragraph, candidate_paragraph, threshold=0.50)
    if not drift.accepted:
        return "semantic_drift " + "; ".join(drift.reasons[:3])
    if (
        not allow_polish_warning
        and _generic_polish_count(candidate_paragraph) > _generic_polish_count(original_paragraph)
    ):
        return "generic_polish_increase"
    anchors = _domain_anchor_terms(rewrite_context, original_paragraph, original_paragraph, limit=12)
    orig_coverage = _term_coverage(original_paragraph, anchors)
    cand_coverage = _term_coverage(candidate_paragraph, anchors)
    if orig_coverage >= 3 and cand_coverage < max(1, int(orig_coverage * 0.60)):
        return f"domain_anchor_loss {orig_coverage}->{cand_coverage}"
    if _density_transformation_too_small(original_paragraph, candidate_paragraph):
        return "density_transformation_too_small"
    return ""


def _density_entity_only_drift(reason: str) -> bool:
    """True when density drift is only the fuzzy named-entity guard.

    The generic semantic guard can over-extract title-case section headings
    from PDF/text extraction, especially when headings and body text are merged
    into one long paragraph. For AI-density mitigation, this should be a
    warning after repair, not a hard blocker. Protected spans, numbers,
    citations, quotes, and domain-anchor loss are still enforced separately.
    """
    if not reason or not reason.startswith("semantic_drift"):
        return False
    drift_bits = [bit.strip() for bit in reason[len("semantic_drift"):].split(";") if bit.strip()]
    return bool(drift_bits) and all(bit.startswith("lost_named_entity:") for bit in drift_bits)


def _density_repair_prompt(
    original_region: str,
    previous_candidate: str,
    rejection_reason: str,
    rewrite_context: Optional[Any],
    mitigation_plan: Optional[Dict[str, Any]],
) -> str:
    """Build a targeted retry prompt for near-miss paragraph candidates."""
    base = _density_paragraph_prompt(original_region, rewrite_context, mitigation_plan)
    lost_entities = re.findall(r"lost_named_entity: '([^']+)'", rejection_reason or "")
    repair_lines = [
        base,
        "\nYour previous candidate was rejected by a hard safety guard.",
        "Previous rejected candidate:\n<REJECTED>\n" + (previous_candidate or "") + "\n</REJECTED>",
        "Repair only the safety failure. Do not introduce a new paragraph strategy.",
        "Return exactly one replacement paragraph. No commentary.",
        "Rejection reason: " + (rejection_reason or "unknown"),
    ]
    if lost_entities:
        repair_lines.append(
            "These named entities must appear exactly, unchanged: "
            + "; ".join(lost_entities)
        )
    if "protected_span_lost" in (rejection_reason or ""):
        protected = [
            span.text or original_region[span.start_char:span.end_char]
            for span in detect_protected_spans(original_region)
        ]
        if protected:
            repair_lines.append(
                "These protected spans must appear exactly, unchanged: "
                + "; ".join(protected[:12])
            )
    if re.search(r"generic_polish_increase|component_regression|unsupported_abstraction", rejection_reason or "", re.I):
        repair_lines.extend([
            "The failure is polish/formality, not missing facts.",
            "Rewrite in plainer author-owned wording. Prefer short, concrete sentences.",
            "Do not use: especially true, especially clear, embedded within, part of how, from my experience teaching, observing a demonstration, executing a controlled haircut.",
            "Keep the classroom/process details, but remove smoother academic framing.",
        ])
    if "density_transformation_too_small" in (rejection_reason or ""):
        repair_lines.extend([
            "The previous candidate kept too much of the original sentence footprint.",
            "Rebuild it again by changing sentence openings, linking phrases, clause order, and sentence boundaries while preserving facts.",
            "Do not solve this by synonym swaps. Move concrete process details to the front of sentences where natural.",
        ])
    if re.search(r"density_local_signal_(?:not_improved|regressed)", rejection_reason or "", re.I):
        repair_lines.extend([
            "The previous candidate failed the local GPT-2 detector check.",
            "Make a stronger AI-risk mitigation rewrite: reduce predictable token paths, reduce generic transitions, and vary sentence starts.",
            "Use shorter, more concrete sentences where natural. Keep names, citations, unit codes, numbers, and facts unchanged.",
        ])
    return "\n".join(repair_lines)


def _find_sentence_index(sentences: List[str], evidence: str) -> int:
    """Find which sentence contains the evidence text."""
    normalized_evidence = _normalize_sentence_match_text(evidence)
    evidence_start = (
        normalized_evidence[:30]
        if len(normalized_evidence) > 30
        else normalized_evidence
    )
    if not evidence_start:
        return -1
    for i, s in enumerate(sentences):
        if evidence_start in _normalize_sentence_match_text(s):
            return i

    evidence_tokens = _match_tokens(normalized_evidence)
    if not evidence_tokens:
        return -1
    best_idx = -1
    best_score = 0.0
    best_overlap = 0
    evidence_set = set(evidence_tokens)
    for i, s in enumerate(sentences):
        sent_tokens = _match_tokens(s)
        if not sent_tokens:
            continue
        overlap = len(evidence_set & set(sent_tokens))
        score = overlap / max(min(len(evidence_set), len(set(sent_tokens))), 1)
        if score > best_score or (score == best_score and overlap > best_overlap):
            best_idx = i
            best_score = score
            best_overlap = overlap
    return best_idx if best_score >= 0.45 else -1


def _match_tokens(text: str) -> List[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text or "")
        if len(token) >= 4
    ]


def _normalize_sentence_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _finding_id(finding: Finding) -> str:
    return (finding.metadata or {}).get("finding_id") or getattr(finding, "id", "")


def _paragraph_indices(sent_idx: int, sentences: List[str], para_map: Optional[Dict[int, int]] = None) -> List[int]:
    """Return all sentence indices in the same paragraph as sent_idx.

    If para_map is provided, uses it for O(1) paragraph lookup.
    Otherwise falls back to contiguous-non-empty heuristic.
    """
    if para_map is not None and sent_idx in para_map:
        target_para = para_map[sent_idx]
        return [i for i, p in para_map.items() if p == target_para]
    # Fallback: walk contiguous non-empty sentences
    start = sent_idx
    while start > 0 and sentences[start - 1].strip():
        start -= 1
    end = sent_idx
    while end < len(sentences) - 1 and sentences[end + 1].strip():
        end += 1
    return list(range(start, end + 1))


def _paragraph_id(sent_idx: int, sentences: List[str]) -> tuple:
    """Return a stable paragraph identity (start, end) sentence indices."""
    indices = _paragraph_indices(sent_idx, sentences)
    return (indices[0], indices[-1])


def _splice_region(
    original: str, sentences: List[str],
    region_indices: List[int], rewritten_region: str,
) -> str:
    """Splice a rewritten region back into the original text."""
    new_sentences = list(sentences)
    rewritten_sents = _split_sentences(rewritten_region)
    if not rewritten_sents:
        return original
    new_sentences[region_indices[0]] = rewritten_region
    for idx in reversed(region_indices[1:]):
        del new_sentences[idx]
    return " ".join(new_sentences)


def _group_actions_by_paragraph(
    actions: List[RewriteAction], sentences: List[str],
    para_map: Optional[Dict[int, int]] = None,
) -> Dict[int, List[RewriteAction]]:
    """Group rewrite actions by paragraph index."""
    groups: Dict[int, List[RewriteAction]] = {}
    for action in actions:
        f = action.finding
        loc = f.location or {}
        sent_idx = loc.get("sentence_index", -1)
        if sent_idx < 0 or sent_idx >= len(sentences):
            sent_idx = _find_sentence_index(sentences, f.evidence)
        if sent_idx < 0:
            continue
        if para_map is not None and sent_idx in para_map:
            p_idx = para_map[sent_idx]
        else:
            # Fallback: compute paragraph ID from contiguous sentences
            indices = _paragraph_indices(sent_idx, sentences)
            p_idx = indices[0]  # use start sentence as proxy
        groups.setdefault(p_idx, []).append(action)
    return groups


def run_rewrite(
    content: str,
    detect_results: List[DetectResult],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    max_passes: int = 3,
    target_top10: float = 0.50,
    rewrite_fn: Optional[callable] = None,
    output_dir: Optional[str] = None,
    max_detect_loops: int = 1,
    min_detect_improvement: int = 1,
    config: Optional[RewriteConfig] = None,
    rewrite_context: Optional[Any] = None,
    ai_only: bool = True,
) -> RewriteModuleResult:
    """Run rewrite in a detect→plan→rewrite→score→verify loop.

    Architecture:
      1. Plan: classify findings into auto-fixable vs manual
      2. Rewrite: only auto-fixable findings, with guards
      3. Score: composite multi-signal candidate scoring
      4. Verify: re-detect, compare weighted risk
      5. Loop: adaptive decision based on improvement, drift, budget

    Citation and integrity findings are NEVER auto-rewritten.
    When ai_only=True, only ai_generation scanner findings are targeted.
    """
    # Save unfiltered results for original metrics/scoring before narrowing targets.
    all_detect_results = detect_results
    original_ai_likelihood = _original_ai_likelihood(all_detect_results, rewrite_context)

    # Filter findings before planning
    if ai_only:
        detect_results = _filter_ai_guided_findings(detect_results, rewrite_context)
    else:
        # All scanners, but only medium severity
        detect_results = _filter_by_severity(detect_results)

    # Merge config
    if config is None:
        config = RewriteConfig(
            max_passes=max_passes,
            max_detect_loops=max_detect_loops,
            target_top10=target_top10,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

    # ── Step 1: Plan ──────────────────────────────────────────────
    planner = RewritePlanner()
    plan = planner.plan(detect_results)

    # Analyze voice profile before any rewriting
    voice_profile = analyze_voice(content)
    voice_guard = VoiceGuard()
    predictability_guard = PredictabilityGuard()

    # Compute rewrite surface (how much is actually rewritable)
    protected_spans = detect_protected_spans(content)
    rewrite_surface = compute_rewrite_surface(content, protected_spans)

    # Floor detection: if most text is protected, skip rewriting
    if rewrite_surface.is_mostly_protected:
        builder = ReportBuilder()
        builder.set_meta(scan_time=0)
        for dr in detect_results:
            builder.add_detection(dr)
        report = builder.build()
        text_rpt = render_report(report)
        md_rpt = render_markdown(report)

        dummy_metrics = compute_metrics(content, PredictabilityScanner())
        noop_result = MultiPassResult(
            original_text=content,
            original_metrics=dummy_metrics,
            passes=[],
            final_text=content,
            final_metrics=dummy_metrics,
            converged=True,
            convergence_reason="Text is mostly protected spans — automatic rewrite is unsafe",
        )

        return RewriteModuleResult(
            mp_result=noop_result,
            report=report,
            text_report=text_rpt,
            markdown_report=md_rpt,
            summary={
                "manual_only": True,
                "reason": "mostly_protected",
                "mitigation_plan": build_mitigation_plan(
                    plan,
                    getattr(rewrite_context, "raw_json", None),
                ),
            },
            report_md_path=None,
            report_json_path=None,
            post_rewrite_detect=detect_results,
            detect_loops_used=0,
            detect_loop_history=[],
            rewrite_plan=plan,
            manual_actions=plan.manual_required,
            outcome=RewriteOutcome.MANUAL_REQUIRED,
            voice_profile=voice_profile,
            rewrite_surface=rewrite_surface,
        )

    if not plan.auto_fixable:
        # Nothing to auto-rewrite — build report with manual actions only
        builder = ReportBuilder()
        builder.set_meta(scan_time=0)
        for dr in detect_results:
            builder.add_detection(dr)
        report = builder.build()
        text_rpt = render_report(report)
        md_rpt = render_markdown(report)

        md_path = json_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            md_path = os.path.join(output_dir, f"draftproof_{ts}.md")
            json_path = os.path.join(output_dir, f"draftproof_{ts}.json")
            with open(md_path, "w") as f:
                f.write(md_rpt)
            summary = get_rewrite_summary_v2(
                plan=plan,
                manual_only=True,
                raw_json=getattr(rewrite_context, "raw_json", None),
            )
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2)

        # Build a no-op MultiPassResult for compatibility
        dummy_metrics = compute_metrics(content, PredictabilityScanner())
        noop_result = MultiPassResult(
            original_text=content,
            original_metrics=dummy_metrics,
            passes=[],
            final_text=content,
            final_metrics=dummy_metrics,
            converged=True,
            convergence_reason="No auto-fixable findings — manual review required",
        )

        return RewriteModuleResult(
            mp_result=noop_result,
            report=report,
            text_report=text_rpt,
            markdown_report=md_rpt,
            summary=get_rewrite_summary_v2(
                plan=plan,
                manual_only=True,
                raw_json=getattr(rewrite_context, "raw_json", None),
            ),
            report_md_path=md_path,
            report_json_path=json_path,
            post_rewrite_detect=detect_results,
            detect_loops_used=0,
            detect_loop_history=[],
            rewrite_plan=plan,
            manual_actions=plan.manual_required,
            outcome=RewriteOutcome.MANUAL_REQUIRED,
            voice_profile=voice_profile,
            rewrite_surface=rewrite_surface,
        )

    # ── Step 2: Setup rewrite context ─────────────────────────────
    guidance = _extract_rewrite_guidance(detect_results)

    domain_terms = None
    rewrite_constraints = None
    if rewrite_context and hasattr(rewrite_context, "domain_profile") and rewrite_context.domain_profile:
        domain_terms = rewrite_context.domain_profile.get("matched_domain_terms")
    if rewrite_context and hasattr(rewrite_context, "raw_json"):
        rewrite_constraints = rewrite_context.raw_json.get("rewrite_constraints")

    detect_context = _build_detect_context(
        guidance, domain_terms=domain_terms,
        rewrite_constraints=rewrite_constraints,
    )
    scan_signal_brief = ""
    if rewrite_context and hasattr(rewrite_context, "raw_json"):
        scan_signal_brief = _build_scan_signal_brief(rewrite_context.raw_json)
    if scan_signal_brief:
        detect_context = (
            f"{detect_context}\n\n{scan_signal_brief}"
            if detect_context else scan_signal_brief
        )
    scanner = PredictabilityScanner()

    regression_memory = RegressionMemory()

    # Build rewrite function
    loop_rewrite_fn = rewrite_fn
    if loop_rewrite_fn is None:
        effective_key = (
            api_key or config.api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        if effective_key:
            effective_model = config.model or os.environ.get("LLM_MODEL")
            loop_rewrite_fn = _rewrite_fn_with_detect_context(
                detect_context, effective_key, effective_model,
                base_url=config.base_url,
                timeout=config.llm_timeout_seconds,
                max_retries=config.llm_max_retries,
            )
        elif detect_context and os.environ.get("DRAFTPROOF_ENABLE_CLAUDE_FALLBACK") == "1":
            loop_rewrite_fn = _make_chipin_rewrite_fn(detect_context)

    # Build GPT-2 hybrid rewriter for predictability findings.
    # GPT-2 identifies predictable tokens + alternatives, LLM picks best replacement.
    gpt2_rewriter = None
    try:
        from rewrite.gpt2_rewriter import GPT2Rewriter
        from llm.gateway import LLMGateway, LLMConfig
        gpt2_gateway = None
        effective_key = (
            api_key or config.api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        if effective_key:
            effective_model = config.model or os.environ.get("LLM_MODEL")
            gpt2_gateway = LLMGateway(LLMConfig(
                api_key=effective_key,
                model=effective_model,
                base_url=config.base_url,
                timeout=config.llm_timeout_seconds,
                max_retries=config.llm_max_retries,
            ))
        gpt2_rewriter = GPT2Rewriter(scanner=scanner, gateway=gpt2_gateway)
    except Exception as exc:
        logger.warning("GPT-2 rewriter not available: %s", exc)

    # ── Step 2b: Deterministic rewrite (DISABLED) ───────────────────
    # Deterministic replacements consistently INCREASE predictability
    # because removing distinctive words leaves only common ones.
    # Disabled until a regression guard is in place (Task #50).
    loop_history = [{
        "loop": 0,
        "weighted_risk": weighted_finding_score(
            [f for dr in detect_results for f in dr.findings]
        ),
        "raw_findings": _count_findings(detect_results),
        "text": content[:100],
        "note": "original",
    }]
    floor_reasons = []
    findings_fixed = 0
    findings_skipped = 0
    loops_used = 0
    current_text = content
    rewrite_checkpoints: List[Dict[str, Any]] = [{
        "text": content,
        "edits": 0,
        "local_score_total": 0.0,
        "note": "original",
    }]
    local_score_total = 0.0
    llm_calls_used = 0
    failed_targets = 0
    consecutive_failed_targets = 0
    circuit_breaker_reason: Optional[str] = None
    density_batch_gate_failed = False
    manual_suggestions: List[Dict[str, Any]] = []
    accepted_candidate_suggestions: List[Dict[str, Any]] = []
    runtime_mitigation_plan = build_mitigation_plan(
        plan,
        getattr(rewrite_context, "raw_json", None),
    )
    density_paragraph_pass: Dict[str, Any] = {
        "attempted": False,
        "applied": False,
        "reason": "not_needed",
        "attempts": [],
        "density_score": _badge_component_score(
            getattr(rewrite_context, "raw_json", None),
            "qualifying_text_ai_density",
        ),
    }
    density_mitigation_llm_calls = 0
    sentence_llm_call_limit = config.max_llm_calls

    # ── Step 3: Per-finding rewrite loop (LLM, all findings) ────────
    current_weighted_risk = weighted_finding_score(
        [f for dr in detect_results for f in dr.findings]
    )

    # ── Per-sentence rewrite: one LLM call per flagged sentence ─────────
    # Process each auto-fixable action individually. The LLM sees only the
    # target sentence + context. Much smaller surface area = less hallucination.
    sentences, para_map = _build_sentence_index(current_text)
    rewrite_start_time = time.monotonic()

    def _attempt_density_paragraph_pass(phase: str) -> None:
        nonlocal current_text, findings_fixed, loops_used, llm_calls_used, density_mitigation_llm_calls

        density_score = float(density_paragraph_pass.get("density_score") or 0.0)
        density_attempts = density_paragraph_pass.setdefault("attempts", [])
        max_density_passes = max(1, int(getattr(config, "max_density_passes", 1) or 1))
        if len(density_attempts) >= max_density_passes:
            return
        if density_score < 70.0:
            return
        if loop_rewrite_fn is None:
            density_paragraph_pass["reason"] = "no_llm_available"
            loop_history.append({
                "loop": loops_used + 1,
                "phase": phase,
                "note": "density paragraph pass skipped: no LLM available",
                "density_score": round(density_score, 2),
            })
            return
        if llm_calls_used >= config.max_llm_calls:
            density_paragraph_pass["reason"] = f"max_llm_calls reached ({config.max_llm_calls})"
            loop_history.append({
                "loop": loops_used + 1,
                "phase": phase,
                "note": "density paragraph pass skipped: " + density_paragraph_pass["reason"],
                "density_score": round(density_score, 2),
            })
            return
        if (time.monotonic() - rewrite_start_time) > config.max_rewrite_seconds:
            density_paragraph_pass["reason"] = (
                f"rewrite time budget exceeded ({config.max_rewrite_seconds}s)"
            )
            loop_history.append({
                "loop": loops_used + 1,
                "phase": phase,
                "note": "density paragraph pass skipped: " + density_paragraph_pass["reason"],
                "density_score": round(density_score, 2),
            })
            return

        para_idx, density_paragraph, density_meta = _select_density_paragraph(
            current_text,
            rewrite_context,
            runtime_mitigation_plan,
            density_attempts,
        )
        density_paragraph_pass.update({
            "attempted": True,
            "applied": False,
            "phase": phase,
            "paragraph_index": para_idx,
            "paragraph_meta": density_meta,
            "reason": "",
        })
        if para_idx < 0 or not density_paragraph:
            density_paragraph_pass["reason"] = "no_eligible_paragraph"
            loop_history.append({
                "loop": loops_used + 1,
                "phase": phase,
                "note": "density paragraph pass skipped: no eligible paragraph",
                "density_score": round(density_score, 2),
            })
            return

        llm_calls_used += 1
        density_mitigation_llm_calls += 1
        density_prompt = _density_paragraph_prompt(
            density_paragraph,
            rewrite_context,
            runtime_mitigation_plan,
        )
        raw_density_output = loop_rewrite_fn(density_paragraph, density_prompt)
        density_candidate = _clean_density_paragraph_output(
            raw_density_output,
            density_paragraph,
        )
        density_reject_reason = _density_paragraph_reject_reason(
            density_paragraph,
            density_candidate,
            rewrite_context,
            allow_polish_warning=True,
        )
        if (
            density_reject_reason
            and density_candidate
            and re.search(
                r"lost_named_entity|protected_span_lost|citation_lost|quote_lost|generic_polish_increase|component_regression|unsupported_abstraction|density_transformation_too_small",
                density_reject_reason,
            )
            and llm_calls_used < config.max_llm_calls
            and (time.monotonic() - rewrite_start_time) <= config.max_rewrite_seconds
        ):
            repair_prompt = _density_repair_prompt(
                density_paragraph,
                density_candidate,
                density_reject_reason,
                rewrite_context,
                runtime_mitigation_plan,
            )
            llm_calls_used += 1
            density_mitigation_llm_calls += 1
            repaired_output = loop_rewrite_fn(density_paragraph, repair_prompt)
            repaired_candidate = _clean_density_paragraph_output(
                repaired_output,
                density_paragraph,
            )
            repaired_reject_reason = _density_paragraph_reject_reason(
                density_paragraph,
                repaired_candidate,
                rewrite_context,
                allow_polish_warning=True,
            )
            loop_history.append({
                "loop": loops_used + 1,
                "phase": phase,
                "paragraph": para_idx,
                "note": "density paragraph repair retry",
                "initial_rejection_reason": density_reject_reason,
                "repair_rejection_reason": repaired_reject_reason,
                "density_score": round(density_score, 2),
                "new_text": repaired_candidate[:240] if repaired_candidate else "",
            })
            if not repaired_reject_reason:
                density_candidate = repaired_candidate
                density_reject_reason = ""
            elif repaired_candidate:
                density_candidate = repaired_candidate
                density_reject_reason = repaired_reject_reason

        if density_reject_reason and _density_entity_only_drift(density_reject_reason):
            density_paragraph_pass["entity_warning"] = density_reject_reason
            loop_history.append({
                "loop": loops_used + 1,
                "phase": phase,
                "paragraph": para_idx,
                "note": "density paragraph entity warning; AI-first final scan will decide",
                "warning": density_reject_reason,
                "density_score": round(density_score, 2),
            })
            density_reject_reason = ""

        density_local_signal: Dict[str, Any] = {}
        if not density_reject_reason:
            local_ok, local_reason, density_local_signal = _density_local_signal_acceptance(
                scanner,
                density_paragraph,
                density_candidate,
            )
            density_paragraph_pass["local_signal"] = density_local_signal
            if not local_ok:
                density_reject_reason = local_reason

                if (
                    density_candidate
                    and llm_calls_used < config.max_llm_calls
                    and (time.monotonic() - rewrite_start_time) <= config.max_rewrite_seconds
                ):
                    repair_prompt = _density_repair_prompt(
                        density_paragraph,
                        density_candidate,
                        density_reject_reason,
                        rewrite_context,
                        runtime_mitigation_plan,
                    )
                    llm_calls_used += 1
                    density_mitigation_llm_calls += 1
                    repaired_output = loop_rewrite_fn(density_paragraph, repair_prompt)
                    repaired_candidate = _clean_density_paragraph_output(
                        repaired_output,
                        density_paragraph,
                    )
                    repaired_reject_reason = _density_paragraph_reject_reason(
                        density_paragraph,
                        repaired_candidate,
                        rewrite_context,
                        allow_polish_warning=True,
                    )
                    if repaired_reject_reason and _density_entity_only_drift(repaired_reject_reason):
                        density_paragraph_pass["entity_warning"] = repaired_reject_reason
                        repaired_reject_reason = ""

                    repaired_signal: Dict[str, Any] = {}
                    if not repaired_reject_reason:
                        repaired_ok, repaired_reason, repaired_signal = _density_local_signal_acceptance(
                            scanner,
                            density_paragraph,
                            repaired_candidate,
                        )
                        if not repaired_ok:
                            repaired_reject_reason = repaired_reason

                    loop_history.append({
                        "loop": loops_used + 1,
                        "phase": phase,
                        "paragraph": para_idx,
                        "note": "density paragraph detector retry",
                        "initial_rejection_reason": density_reject_reason,
                        "repair_rejection_reason": repaired_reject_reason,
                        "initial_local_signal": density_local_signal,
                        "repair_local_signal": repaired_signal,
                        "density_score": round(density_score, 2),
                        "new_text": repaired_candidate[:240] if repaired_candidate else "",
                    })
                    if not repaired_reject_reason:
                        density_candidate = repaired_candidate
                        density_reject_reason = ""
                        density_local_signal = repaired_signal
                        density_paragraph_pass["local_signal"] = repaired_signal
                    elif repaired_candidate:
                        density_candidate = repaired_candidate
                        density_reject_reason = repaired_reject_reason
                        density_local_signal = repaired_signal or density_local_signal
                        density_paragraph_pass["local_signal"] = density_local_signal

        if density_reject_reason:
            density_paragraph_pass["reason"] = density_reject_reason
            density_attempts.append({
                "phase": phase,
                "applied": False,
                "reason": density_reject_reason,
                **density_meta,
            })
            if density_candidate:
                manual_item = {
                    "finding_id": "density_paragraph_rebuild",
                    "finding_type": "qualifying_text_ai_density",
                    "risk_level": "high",
                    "scanner_target": "ai_generation",
                    "sentence_id": None,
                    "paragraph_role": "unknown",
                    "original_sentence": density_paragraph,
                    "suggested_sentence": density_candidate,
                    "rejection_reason": density_reject_reason,
                    "why_review_manually": (
                        "This paragraph-level candidate targets the dense AI-style pattern, "
                        "but an automatic guard rejected it. Review the paragraph manually before using it."
                    ),
                }
                key = (
                    manual_item["finding_id"],
                    manual_item["original_sentence"],
                    manual_item["suggested_sentence"],
                    manual_item["rejection_reason"],
                )
                existing = {
                    (
                        s.get("finding_id"),
                        s.get("original_sentence"),
                        s.get("suggested_sentence"),
                        s.get("rejection_reason"),
                    )
                    for s in manual_suggestions
                }
                if key not in existing and len(manual_suggestions) < 24:
                    manual_suggestions.append(manual_item)
            loop_history.append({
                "loop": loops_used + 1,
                "phase": phase,
                "paragraph": para_idx,
                "note": "density paragraph pass rejected",
                "rejection_reason": density_reject_reason,
                "density_score": round(density_score, 2),
                "orig_text": density_paragraph[:240],
                "new_text": density_candidate[:240] if density_candidate else "",
            })
            return

        candidate_text = _splice_density_candidate(
            current_text,
            para_idx,
            density_paragraph,
            density_candidate,
            density_meta,
        )
        if not candidate_text:
            density_paragraph_pass["reason"] = "density_region_splice_failed"
            density_attempts.append({
                "phase": phase,
                "applied": False,
                "reason": "density_region_splice_failed",
                **density_meta,
            })
            return

        component_ok, component_reason = _component_regression_check(current_text, candidate_text)
        if not component_ok:
            density_paragraph_pass["component_warning"] = f"document_component_regression {component_reason}"
            loop_history.append({
                "loop": loops_used + 1,
                "phase": phase,
                "paragraph": para_idx,
                "note": "density paragraph component warning; AI-first final scan will decide",
                "warning": density_paragraph_pass["component_warning"],
            })
        if _generic_polish_count(density_candidate) > _generic_polish_count(density_paragraph):
            density_paragraph_pass["polish_warning"] = "generic_polish_increase"
            loop_history.append({
                "loop": loops_used + 1,
                "phase": phase,
                "paragraph": para_idx,
                "note": "density paragraph polish warning; AI-first final scan will decide",
                "warning": density_paragraph_pass["polish_warning"],
            })

        voice_check = voice_guard.check(current_text, candidate_text)
        if not voice_check.accepted:
            density_paragraph_pass["reason"] = f"voice_eroded {voice_check.reject_reason}"
            density_attempts.append({
                "phase": phase,
                "applied": False,
                "reason": density_paragraph_pass["reason"],
                **density_meta,
            })
            manual_suggestions.append({
                "finding_id": "density_paragraph_rebuild",
                "finding_type": "qualifying_text_ai_density",
                "risk_level": "high",
                "scanner_target": "ai_generation",
                "sentence_id": None,
                "paragraph_role": "unknown",
                "original_sentence": density_paragraph,
                "suggested_sentence": density_candidate,
                "rejection_reason": density_paragraph_pass["reason"],
                "why_review_manually": (
                    "The paragraph candidate may help the density signal, but it changed voice enough "
                    "that it should be reviewed manually."
                ),
            })
            loop_history.append({
                "loop": loops_used + 1,
                "phase": phase,
                "paragraph": para_idx,
                "note": "density paragraph pass rejected",
                "rejection_reason": density_paragraph_pass["reason"],
            })
            return

        current_text = candidate_text
        findings_fixed += 1
        loops_used += 1
        density_attempts.append({
            "phase": phase,
            "applied": True,
            "reason": "accepted_locally_pending_final_full_scan",
            "orig_length": len(density_paragraph),
            "new_length": len(density_candidate),
            "local_signal": density_local_signal,
            **density_meta,
        })
        density_paragraph_pass.update({
            "applied": True,
            "reason": "accepted_locally_pending_final_full_scan",
            "orig_length": len(density_paragraph),
            "new_length": len(density_candidate),
            "local_signal": density_local_signal,
        })
        accepted_candidate_suggestions.append({
            "finding_id": "density_paragraph_rebuild",
            "finding_type": "qualifying_text_ai_density",
            "risk_level": "high",
            "scanner_target": "ai_generation",
            "sentence_id": None,
            "paragraph_role": "unknown",
            "original_sentence": density_paragraph,
            "suggested_sentence": density_candidate,
            "rejection_reason": "accepted_locally_pending_final_full_scan",
            "why_review_manually": (
                "This paragraph-level edit was accepted locally, but the final full scan remains the authority."
            ),
        })
        rewrite_checkpoints.append({
            "text": current_text,
            "edits": findings_fixed,
            "local_score_total": round(local_score_total, 4),
            "paragraph": para_idx,
            "finding_type": "qualifying_text_ai_density",
            "rewrite_operation": "density_paragraph_rebuild",
        })
        loop_history.append({
            "loop": loops_used,
            "phase": phase,
            "paragraph": para_idx,
            "finding_type": "qualifying_text_ai_density",
            "rewrite_operation": "density_paragraph_rebuild",
            "orig_length": len(density_paragraph),
            "new_length": len(density_candidate),
            "density_score": round(density_score, 2),
            "orig_text": density_paragraph[:240],
            "new_text": density_candidate[:240],
            "note": "applied density paragraph candidate",
        })

    # Best mitigation first: if the scan says the problem is document-level
    # density, act on the strongest paragraph before spending calls on
    # sentence-level cleanup.
    for _density_pass_index in range(max(1, int(getattr(config, "max_density_passes", 1) or 1))):
        before_density_calls = density_mitigation_llm_calls
        _attempt_density_paragraph_pass("before_sentence_rewrites")
        if density_mitigation_llm_calls == before_density_calls:
            break

    raw_context_json = getattr(rewrite_context, "raw_json", None) if rewrite_context else None
    if (
        float(density_paragraph_pass.get("density_score") or 0.0) >= 70.0
        and density_mitigation_llm_calls >= 4
        and current_text != content
        and isinstance(raw_context_json, dict)
        and os.environ.get("DRAFTPROOF_DENSITY_BATCH_GATE", "1") != "0"
    ):
        gate_started = time.monotonic()
        original_ai_score = _report_badge_score(raw_context_json, "ai_likelihood_score")
        original_density_score = _badge_component_score(raw_context_json, "qualifying_text_ai_density")
        try:
            density_checkpoint_report = _full_scan_report_dict_for_rewrite_gate(current_text)
            checkpoint_ai_score = _report_badge_score(density_checkpoint_report, "ai_likelihood_score")
            checkpoint_density_score = _badge_component_score(
                density_checkpoint_report,
                "qualifying_text_ai_density",
            )
            ai_delta = (
                original_ai_score - checkpoint_ai_score
                if original_ai_score is not None and checkpoint_ai_score is not None
                else None
            )
            density_delta = (
                original_density_score - checkpoint_density_score
                if checkpoint_density_score is not None
                else None
            )
            accepted_density = len([
                attempt for attempt in density_paragraph_pass.get("attempts", [])
                if attempt.get("applied")
            ])
            density_batch_gate = {
                "enabled": True,
                "passed": False,
                "checkpoint": "after_density_before_sentence_rewrites",
                "original_ai": original_ai_score,
                "checkpoint_ai": checkpoint_ai_score,
                "ai_delta": round(ai_delta, 3) if isinstance(ai_delta, (int, float)) else None,
                "original_density": original_density_score,
                "checkpoint_density": checkpoint_density_score,
                "density_delta": round(density_delta, 3) if isinstance(density_delta, (int, float)) else None,
                "accepted_density_edits": accepted_density,
                "density_llm_calls": density_mitigation_llm_calls,
                "seconds": round(time.monotonic() - gate_started, 3),
            }
            passed_gate = (
                (isinstance(ai_delta, (int, float)) and ai_delta >= 3.0)
                or (isinstance(density_delta, (int, float)) and density_delta >= 3.0)
                or (
                    accepted_density >= 2
                    and isinstance(ai_delta, (int, float))
                    and ai_delta >= 1.5
                    and isinstance(density_delta, (int, float))
                    and density_delta >= 1.0
                )
            )
            density_batch_gate["passed"] = bool(passed_gate)
            density_paragraph_pass["batch_gate"] = density_batch_gate
            loop_history.append({
                "loop": loops_used + 1,
                "phase": "after_density_before_sentence_rewrites",
                "note": (
                    "density batch AI gate passed"
                    if passed_gate else "density batch AI gate failed; sentence rewrites skipped"
                ),
                **density_batch_gate,
            })
            if not passed_gate:
                density_batch_gate_failed = True
                circuit_breaker_reason = (
                    "density_batch_ai_gate_failed "
                    f"ai_delta={density_batch_gate['ai_delta']} "
                    f"density_delta={density_batch_gate['density_delta']}"
                )
        except Exception as exc:
            density_paragraph_pass["batch_gate"] = {
                "enabled": True,
                "passed": None,
                "error": str(exc),
                "seconds": round(time.monotonic() - gate_started, 3),
            }
            logger.warning("Density batch AI gate failed to run: %s", exc)

    def _action_group_key(action):
        finding = action.finding
        evidence = (finding.evidence or "").strip()
        if evidence:
            return re.sub(r"\s+", " ", evidence).lower()
        loc = finding.location or {}
        sent_idx = loc.get("sentence_index")
        if sent_idx is not None:
            return f"sentence:{sent_idx}"
        return f"finding:{getattr(finding, 'id', id(finding))}"

    grouped_actions: List[List[Any]] = []
    grouped_by_key: Dict[str, List[Any]] = {}
    effective_auto_target_limit = max(0, config.max_auto_targets)
    guided_throttle_reason = ""
    if runtime_mitigation_plan.get("primary_mode") == "guided_revision":
        density_is_primary = float(density_paragraph_pass.get("density_score") or 0.0) >= 70.0
        evidence_count = int(
            (runtime_mitigation_plan.get("counts") or {}).get("needs_source_or_example", 0)
            or 0
        )
        auto_count = len(plan.auto_fixable)
        if density_is_primary:
            guided_throttle_reason = (
                "guided_revision_primary: qualifying-text AI density dominates; "
                f"automatic rewrite allowed up to {effective_auto_target_limit} target(s)"
            )
        elif evidence_count >= max(1, auto_count // 2):
            guided_cap = max(1, min(3, auto_count))
            effective_auto_target_limit = min(effective_auto_target_limit, guided_cap)
            guided_throttle_reason = (
                "guided_revision_primary: evidence/source drivers dominate; "
                f"automatic rewrite limited to {effective_auto_target_limit} high-signal target(s)"
            )
    if density_batch_gate_failed:
        effective_auto_target_limit = 0
        guided_throttle_reason = (
            "density_batch_ai_gate_failed: density phase did not reduce total AI enough; "
            "sentence rewrites skipped to avoid burning LLM calls"
        )
    auto_targets = plan.auto_fixable[:effective_auto_target_limit]
    capped_auto_targets = max(0, len(plan.auto_fixable) - len(auto_targets))
    for action in auto_targets:
        key = _action_group_key(action)
        if key not in grouped_by_key:
            grouped_by_key[key] = []
            grouped_actions.append(grouped_by_key[key])
        grouped_by_key[key].append(action)
    if capped_auto_targets:
        loop_history.append({
            "loop": 0,
            "note": (
                guided_throttle_reason
                or f"auto target cap applied ({len(auto_targets)}/{len(plan.auto_fixable)})"
            ),
            "auto_target_limit": effective_auto_target_limit,
            "skipped_auto_targets": capped_auto_targets,
        })

    risk_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    for actions_for_sentence in grouped_actions:
        if (
            llm_calls_used >= sentence_llm_call_limit
            or failed_targets >= config.max_failed_targets
            or consecutive_failed_targets >= config.max_consecutive_failed_targets
        ):
            if llm_calls_used >= sentence_llm_call_limit:
                circuit_breaker_reason = f"max_llm_calls reached ({config.max_llm_calls})"
            elif failed_targets >= config.max_failed_targets:
                circuit_breaker_reason = f"max_failed_targets reached ({config.max_failed_targets})"
            else:
                circuit_breaker_reason = (
                    "max_consecutive_failed_targets reached "
                    f"({config.max_consecutive_failed_targets})"
                )
            remaining = sum(len(group) for group in grouped_actions[loops_used:])
            findings_skipped += remaining
            loop_history.append({
                "loop": loops_used + 1,
                "reverted": True,
                "note": f"stopped: {circuit_breaker_reason}",
                "remaining_findings": remaining,
                "llm_calls_used": llm_calls_used,
            })
            break

        if time.monotonic() - rewrite_start_time > config.max_rewrite_seconds:
            circuit_breaker_reason = (
                f"rewrite time budget exceeded ({config.max_rewrite_seconds}s)"
            )
            remaining = sum(len(group) for group in grouped_actions[loops_used:])
            findings_skipped += remaining
            loop_history.append({
                "loop": loops_used + 1,
                "reverted": True,
                "note": f"stopped: {circuit_breaker_reason}",
                "remaining_findings": remaining,
            })
            break

        actions_for_sentence.sort(
            key=lambda a: (
                risk_rank.get(a.finding.risk_level, 0),
                1 if a.finding.finding_type == "high_topk_predictability" else 0,
            ),
            reverse=True,
        )
        action = actions_for_sentence[0]
        loops_used += 1
        f = action.finding
        companion_findings = [a.finding for a in actions_for_sentence[1:]]

        # Locate the sentence in current text
        current_sentences, _ = _build_sentence_index(current_text)
        loc = f.location or {}
        sent_idx = loc.get("sentence_index", -1)

        # Try sentence_index first, then fuzzy match
        evidence_start = _normalize_sentence_match_text(f.evidence)[:30]
        if sent_idx >= 0 and sent_idx < len(current_sentences):
            # Verify the indexed sentence actually contains the evidence
            if evidence_start not in _normalize_sentence_match_text(current_sentences[sent_idx]):
                sent_idx = _find_sentence_index(current_sentences, f.evidence)
        else:
            sent_idx = _find_sentence_index(current_sentences, f.evidence)

        # Last resort: find evidence anywhere in text and pick closest sentence
        if sent_idx < 0:
            pos = current_text.find(f.evidence[:30])
            if pos >= 0:
                # Find which sentence contains this position
                char_offset = 0
                for i, s in enumerate(current_sentences):
                    sent_pos = current_text.find(s, char_offset)
                    if sent_pos <= pos < sent_pos + len(s) + 10:
                        sent_idx = i
                        break
                    char_offset = max(char_offset, sent_pos + 1)

        if sent_idx < 0:
            findings_skipped += len(actions_for_sentence)
            floor_reasons.append(FloorReason(
                finding_id=_finding_id(f),
                reason_type="sentence_not_found",
                explanation=f"Cannot locate: '{f.evidence[:50]}...'",
            ))
            loop_history.append({
                "loop": loops_used,
                "finding_id": _finding_id(f),
                "finding_type": f.finding_type,
                "evidence": f.evidence[:160],
                "reverted": True,
                "note": "skipped: sentence not found",
            })
            continue

        original_sentence = current_sentences[sent_idx]

        # Check evidence still present in text (not just this sentence —
        # evidence may span a boundary our splitter creates)
        if evidence_start and evidence_start not in _normalize_sentence_match_text(current_text):
            findings_skipped += len(actions_for_sentence)
            floor_reasons.append(FloorReason(
                finding_id=_finding_id(f),
                reason_type="evidence_not_found",
                explanation="Evidence gone from sentence",
            ))
            loop_history.append({
                "loop": loops_used,
                "finding_id": _finding_id(f),
                "finding_type": f.finding_type,
                "evidence": f.evidence[:160],
                "reverted": True,
                "note": "skipped: evidence not found in sentence",
            })
            continue

        current_sentences, current_para_map = _build_sentence_index(current_text)
        # Build context: target sentence +/- 1 sentence and full paragraph.
        ctx_before = current_sentences[sent_idx - 1] if sent_idx > 0 else ""
        ctx_after = current_sentences[sent_idx + 1] if sent_idx < len(current_sentences) - 1 else ""
        original_paragraph = _paragraph_context(current_sentences, current_para_map, sent_idx)
        edit_brief = _rewrite_edit_brief_for_target(rewrite_context, f)

        # Protected spans within this sentence
        all_protected = detect_protected_spans(current_text)
        sent_start = current_text.find(original_sentence)
        sent_protected = [
            ps for ps in all_protected
            if sent_start >= 0
            and ps.start_char >= sent_start
            and ps.end_char <= sent_start + len(original_sentence)
        ] if sent_start >= 0 else []

        # Build focused per-sentence prompt with signal enrichment
        span_info_parts = [
            "Finding: [" + f.finding_type + "/" + f.risk_level + "] " + f.detail,
        ]
        for companion in companion_findings:
            span_info_parts.append(
                "Companion finding on the same sentence: ["
                + companion.finding_type + "/" + companion.risk_level + "] "
                + companion.detail
            )
        # Inject specific signal metrics (trigger, problem tokens, etc.)
        enriched = _enrich_span_info(f, rewrite_context, sent_idx)
        if enriched:
            span_info_parts.append(enriched)
        brief_lines = _brief_signal_lines(edit_brief)
        if brief_lines:
            span_info_parts.extend(brief_lines)
        domain_anchors = _domain_anchor_terms(
            rewrite_context,
            original_paragraph,
            original_sentence,
        )
        brief_anchors = edit_brief.get("domain_anchors") if isinstance(edit_brief, dict) else []
        if isinstance(brief_anchors, list) and brief_anchors:
            merged = []
            for term in list(brief_anchors) + domain_anchors:
                if isinstance(term, str) and term and term.lower() not in {t.lower() for t in merged}:
                    merged.append(term)
            domain_anchors = merged[:12]
        metadata_context = f.metadata.get("rewrite_context") if isinstance(f.metadata, dict) else {}
        if isinstance(metadata_context, dict):
            metadata_anchors = metadata_context.get("domain_anchors") or []
            if isinstance(metadata_anchors, list):
                merged = []
                for term in list(metadata_anchors) + domain_anchors:
                    if isinstance(term, str) and term and term.lower() not in {t.lower() for t in merged}:
                        merged.append(term)
                domain_anchors = merged[:12]
        if domain_anchors:
            span_info_parts.append(
                "Domain anchors from scan/context: " + ", ".join(domain_anchors)
            )
        if isinstance(metadata_context, dict) and metadata_context.get("signal_instruction"):
            span_info_parts.append(
                "Scan rewrite instruction: " + str(metadata_context["signal_instruction"])
            )
        if isinstance(metadata_context, dict):
            problem_tokens = metadata_context.get("problem_tokens") or []
            if problem_tokens:
                formatted_tokens = []
                for item in problem_tokens[:8]:
                    if isinstance(item, dict):
                        token = item.get("token", "")
                        rank = item.get("rank")
                        prob = item.get("probability")
                        token_text = f'"{token}"'
                        if rank is not None:
                            token_text += f" rank={rank}"
                        if isinstance(prob, (int, float)):
                            token_text += f" p={prob:.1%}"
                        formatted_tokens.append(token_text)
                    elif isinstance(item, str):
                        formatted_tokens.append(f'"{item}"')
                if formatted_tokens:
                    span_info_parts.append(
                        "GPT-2 problem tokens from scan: " + ", ".join(formatted_tokens)
                    )
            predictable_spans = metadata_context.get("predictable_token_spans") or []
            if predictable_spans:
                span_info_parts.append(
                    "Predictable token spans from scan: "
                    + ", ".join(f'"{span}"' for span in predictable_spans[:5])
                )
        constraint_lines = _rewrite_constraint_lines(rewrite_context)
        if constraint_lines:
            span_info_parts.extend(constraint_lines)
        rewrite_operation = _plan_rewrite_operation(
            finding=f,
            edit_brief=edit_brief,
            metadata_context=metadata_context if isinstance(metadata_context, dict) else {},
            original_sentence=original_sentence,
            previous_sentence=ctx_before,
            next_sentence=ctx_after,
            domain_anchors=domain_anchors,
        )
        span_info_parts.extend(_rewrite_operation_lines(rewrite_operation))
        span_info_parts.append(
            _signal_driven_instruction(f, enriched, domain_anchors)
        )
        # Concrete finding-specific strategy
        span_info_parts.append("Strategy: " + _derive_strategy(f, enriched))
        span_info_parts.append("Flagged phrase: '" + f.evidence + "'")
        if ctx_before:
            span_info_parts.append("Previous sentence context: '" + ctx_before + "'")
        if ctx_after:
            span_info_parts.append("Next sentence context: '" + ctx_after + "'")
        span_info_parts.append(
            "Paragraph context with target marked:\n"
            + _marked_paragraph(original_paragraph, original_sentence)
        )
        if sent_protected:
            protected_items = ", ".join(ps.text for ps in sent_protected)
            span_info_parts.append("Must preserve exactly: " + protected_items)
        brief_protected = _protected_texts_from_brief(edit_brief)
        if brief_protected:
            span_info_parts.append(
                "Must preserve from edit brief: " + ", ".join(brief_protected[:10])
            )
        # Fallback: keep legacy sentence metrics if enrichment was empty
        if not enriched:
            sentence_metrics = _sentence_signal_context(rewrite_context, f, sent_idx)
            if sentence_metrics:
                span_info_parts.append(sentence_metrics)

        # GPT-2 token analysis for predictability findings.
        # Don't bypass the LLM — instead, inject precise signal data into
        # the prompt so the LLM knows exactly which tokens are predictable
        # and can do structural rewriting informed by real metrics.
        is_predictability = f.finding_type in (
            "high_predictability", "medium_predictability",
            "high_topk_predictability", "low_surprisal",
        )
        if is_predictability and gpt2_rewriter:
            gpt2_analysis = gpt2_rewriter.analyze_sentence(original_sentence)
            if gpt2_analysis:
                analysis_lines = ["GPT-2 predictability analysis:"]
                for t in gpt2_analysis:
                    alts = ", ".join(t["alternatives"][:5]) if t.get("alternatives") else "(none)"
                    analysis_lines.append(
                        f'  Token "{t["token"]}": GPT-2 rank {t["rank"]}, '
                        f"predictability {t['probability']:.1%}. "
                        f"Less predictable alternatives: {alts}"
                    )
                analysis_lines.append(
                    "Use this data to restructure the sentence. You may replace "
                    "flagged words with alternatives above, or rephrase the surrounding "
                    "context to make the flagged words less predictable in their new context."
                )
                span_info_parts.append("\n".join(analysis_lines))

        span_info = "\n".join(span_info_parts)

        # Generate candidate rewrites and choose the safest accepted candidate.
        rewritten_sentence = None
        drift = None
        best_candidate_info: Optional[Dict[str, Any]] = None
        rejected_candidates: List[Dict[str, Any]] = []
        max_sent_chars = _max_candidate_chars(original_sentence, f)

        def _remember_manual_suggestion(
            candidate_sentence: str,
            reason: str,
            candidate_drift: Optional[DriftCheck] = None,
        ) -> None:
            if not candidate_sentence:
                return
            similarity = (
                candidate_drift.similarity
                if candidate_drift is not None
                else SequenceMatcher(None, original_sentence.lower(), candidate_sentence.lower(), autojunk=False).ratio()
            )
            if similarity < 0.70:
                return
            item = _manual_suggestion_item(
                finding=f,
                original_sentence=original_sentence,
                candidate_sentence=candidate_sentence,
                rejection_reason=reason,
                paragraph_role=str(edit_brief.get("paragraph_role") or "unknown"),
            )
            key = (
                item.get("finding_id"),
                item.get("original_sentence"),
                item.get("suggested_sentence"),
                item.get("rejection_reason"),
            )
            existing = {
                (
                    s.get("finding_id"),
                    s.get("original_sentence"),
                    s.get("suggested_sentence"),
                    s.get("rejection_reason"),
                )
                for s in manual_suggestions
            }
            if key not in existing and len(manual_suggestions) < 24:
                manual_suggestions.append(item)

        for attempt in range(1):
            candidates: List[str] = []
            if loop_rewrite_fn:
                if llm_calls_used >= sentence_llm_call_limit:
                    circuit_breaker_reason = f"max_llm_calls reached ({config.max_llm_calls})"
                    break
                llm_calls_used += 1
                prompt = span_info + "\n\n" + _candidate_task_instruction(
                    f,
                    max_sent_chars,
                    rewrite_operation,
                )
                raw_output = loop_rewrite_fn(original_sentence, prompt)
                candidates = _parse_rewrite_candidates(raw_output, original_sentence)
                if not candidates and raw_output:
                    candidates = _parse_rewrite_candidates("1. " + raw_output, original_sentence)

            if not candidates:
                break

            for cand_idx, candidate_sentence in enumerate(candidates, 1):
                candidate_rejects = []
                if len(candidate_sentence) > max_sent_chars * 2:
                    candidate_rejects.append(f"gross_length {len(candidate_sentence)}>{max_sent_chars * 2}")
                    rejected_candidates.append({
                        "candidate": cand_idx,
                        "reason": "; ".join(candidate_rejects),
                        "text": candidate_sentence[:100],
                    })
                    continue

                candidate_drift = check_semantic_drift(original_sentence, candidate_sentence, threshold=0.2)
                if not candidate_drift.accepted:
                    candidate_rejects.append("; ".join(candidate_drift.reasons[:3]))
                    rejected_candidates.append({
                        "candidate": cand_idx,
                        "reason": "; ".join(candidate_rejects),
                        "drift_similarity": round(candidate_drift.similarity, 3),
                        "text": candidate_sentence[:100],
                    })
                    continue

                if len(candidate_sentence) > max_sent_chars:
                    candidate_rejects.append(f"length {len(candidate_sentence)}>{max_sent_chars}")
                    rejected_candidates.append({
                        "candidate": cand_idx,
                        "reason": "; ".join(candidate_rejects),
                        "text": candidate_sentence[:100],
                    })
                    _remember_manual_suggestion(
                        candidate_sentence,
                        "; ".join(candidate_rejects),
                        candidate_drift,
                    )
                    continue

                protected_lost = [ps for ps in sent_protected
                                  if ps.text not in candidate_sentence] if sent_protected else []
                if protected_lost:
                    lost = ", ".join(f"'{ps.text}'" for ps in protected_lost)
                    candidate_rejects.append(f"protected_span_lost {lost}")
                    rejected_candidates.append({
                        "candidate": cand_idx,
                        "reason": "; ".join(candidate_rejects),
                        "text": candidate_sentence[:100],
                    })
                    _remember_manual_suggestion(
                        candidate_sentence,
                        "; ".join(candidate_rejects),
                        candidate_drift,
                    )
                    continue

                brief_lost = [text for text in brief_protected if text not in candidate_sentence]
                if brief_lost:
                    candidate_rejects.append(
                        "brief_protected_span_lost " + ", ".join(f"'{text}'" for text in brief_lost[:5])
                    )
                    rejected_candidates.append({
                        "candidate": cand_idx,
                        "reason": "; ".join(candidate_rejects),
                        "text": candidate_sentence[:100],
                    })
                    _remember_manual_suggestion(
                        candidate_sentence,
                        "; ".join(candidate_rejects),
                        candidate_drift,
                    )
                    continue

                if original_sentence in current_text:
                    candidate_text = current_text.replace(original_sentence, candidate_sentence, 1)
                else:
                    candidate_rejects.append("splice_failed")
                    rejected_candidates.append({
                        "candidate": cand_idx,
                        "reason": "; ".join(candidate_rejects),
                        "text": candidate_sentence[:100],
                    })
                    continue

                voice_check = voice_guard.check(current_text, candidate_text)
                if not voice_check.accepted:
                    candidate_rejects.append(f"voice_eroded {voice_check.reject_reason}")
                    rejected_candidates.append({
                        "candidate": cand_idx,
                        "reason": "; ".join(candidate_rejects),
                        "text": candidate_sentence[:100],
                    })
                    _remember_manual_suggestion(
                        candidate_sentence,
                        "; ".join(candidate_rejects),
                        candidate_drift,
                    )
                    continue

                reg_check = predictability_guard.check(
                    current_text,
                    candidate_text,
                    original_sentence,
                    candidate_sentence,
                )
                if not reg_check.accepted:
                    candidate_rejects.append(
                        f"predictability_regression {reg_check.orig_risk:.3f}->{reg_check.new_risk:.3f}"
                    )
                    rejected_candidates.append({
                        "candidate": cand_idx,
                        "reason": "; ".join(candidate_rejects),
                        "text": candidate_sentence[:100],
                    })
                    _remember_manual_suggestion(
                        candidate_sentence,
                        "; ".join(candidate_rejects),
                        candidate_drift,
                    )
                    continue

                target_orig_pred = {"risk": None, "label": ""}
                target_new_pred = {"risk": None, "label": ""}
                target_acceptance_note = ""
                if _requires_medium_exit(f):
                    target_scanner = predictability_guard._get_scanner()
                    target_orig_pred = _sentence_predictability(target_scanner, original_sentence)
                    target_new_pred = _sentence_predictability(target_scanner, candidate_sentence)
                    target_new_risk = target_new_pred.get("risk")
                    target_new_label = target_new_pred.get("label")
                    target_ok, target_reason, _target_delta = _target_predictability_acceptance(
                        target_orig_pred,
                        target_new_pred,
                    )
                    if not target_ok:
                        candidate_rejects.append(target_reason)
                        rejected_candidates.append({
                            "candidate": cand_idx,
                            "reason": "; ".join(candidate_rejects),
                            "target_orig_risk": target_orig_pred.get("risk"),
                            "target_new_risk": target_new_risk,
                            "target_new_label": target_new_label,
                            "text": candidate_sentence[:100],
                        })
                        _remember_manual_suggestion(
                            candidate_sentence,
                            "; ".join(candidate_rejects),
                            candidate_drift,
                        )
                        continue
                    target_acceptance_note = target_reason

                component_ok, component_reason = _component_regression_check(current_text, candidate_text)
                if not component_ok:
                    candidate_rejects.append(f"badge_component_regression {component_reason}")
                    rejected_candidates.append({
                        "candidate": cand_idx,
                        "reason": "; ".join(candidate_rejects),
                        "text": candidate_sentence[:100],
                    })
                    _remember_manual_suggestion(
                        candidate_sentence,
                        "; ".join(candidate_rejects),
                        candidate_drift,
                    )
                    continue

                style_reason = _candidate_style_reject_reason(original_sentence, candidate_sentence)
                if style_reason:
                    candidate_rejects.append(style_reason)
                    rejected_candidates.append({
                        "candidate": cand_idx,
                        "reason": "; ".join(candidate_rejects),
                        "text": candidate_sentence[:100],
                    })
                    _remember_manual_suggestion(
                        candidate_sentence,
                        "; ".join(candidate_rejects),
                        candidate_drift,
                    )
                    continue

                coherence_reason = _paragraph_coherence_reject_reason(
                    original_sentence,
                    candidate_sentence,
                    ctx_before,
                    ctx_after,
                    domain_anchors,
                    original_paragraph,
                )
                if coherence_reason:
                    candidate_rejects.append(f"paragraph_coherence {coherence_reason}")
                    rejected_candidates.append({
                        "candidate": cand_idx,
                        "reason": "; ".join(candidate_rejects),
                        "text": candidate_sentence[:100],
                    })
                    _remember_manual_suggestion(
                        candidate_sentence,
                        "; ".join(candidate_rejects),
                        candidate_drift,
                    )
                    continue

                candidate_paragraph = original_paragraph.replace(original_sentence, candidate_sentence, 1)
                score = _candidate_quality_score(
                    original_sentence=original_sentence,
                    candidate_sentence=candidate_sentence,
                    original_paragraph=original_paragraph,
                    candidate_paragraph=candidate_paragraph,
                    predictability_delta=max(0.0, reg_check.orig_risk - reg_check.new_risk),
                    drift_similarity=candidate_drift.similarity,
                    domain_anchor_terms=domain_anchors,
                )
                if score < config.min_improvement:
                    candidate_rejects.append(
                        f"quality_score_below_min {score:.4f}<{config.min_improvement:.4f}"
                    )
                    rejected_candidates.append({
                        "candidate": cand_idx,
                        "reason": "; ".join(candidate_rejects),
                        "score": score,
                        "text": candidate_sentence[:100],
                    })
                    _remember_manual_suggestion(
                        candidate_sentence,
                        "; ".join(candidate_rejects),
                        candidate_drift,
                    )
                    continue
                info = {
                    "candidate": cand_idx,
                    "attempt": attempt + 1,
                    "score": score,
                    "text": candidate_sentence,
                    "candidate_text": candidate_text,
                    "drift": candidate_drift,
                    "orig_risk": reg_check.orig_risk,
                    "new_risk": reg_check.new_risk,
                    "target_orig_risk": target_orig_pred.get("risk"),
                    "target_orig_label": target_orig_pred.get("label"),
                    "target_new_risk": target_new_pred.get("risk"),
                    "target_new_label": target_new_pred.get("label"),
                    "target_acceptance_note": target_acceptance_note,
                }
                if best_candidate_info is None or score > best_candidate_info["score"]:
                    best_candidate_info = info

            if best_candidate_info:
                break

            rejection_summary = "; ".join(r["reason"] for r in rejected_candidates[-3:])
            span_info += (
                f"\nRETRY ({attempt+1}): All candidates failed local gates. "
                f"Avoid these failures: {rejection_summary}. "
                f"Keep the paragraph less generic and output 6 new one-sentence candidates that score below {MEDIUM_PREDICTABILITY_CEILING:.2f}."
            )

        if best_candidate_info:
            rewritten_sentence = best_candidate_info["text"]
            candidate_text = best_candidate_info["candidate_text"]
            drift = best_candidate_info["drift"]

        # Check result
        if rewritten_sentence is None or (drift and not drift.accepted):
            findings_skipped += len(actions_for_sentence)
            failed_targets += len(actions_for_sentence)
            consecutive_failed_targets += 1
            reason = "rewrite_failed" if rewritten_sentence is None else "semantic_drift"
            sim_note = f" ({drift.similarity:.3f})" if drift else ""
            rejection_summary = "; ".join(r.get("reason", "") for r in rejected_candidates[-5:])
            floor_reasons.append(FloorReason(
                finding_id=_finding_id(f),
                reason_type=reason,
                explanation=f"Sentence {reason}{sim_note} after retries",
            ))
            loop_history.append({
                "loop": loops_used,
                "sentence": sent_idx,
                "finding_id": _finding_id(f),
                "finding_type": f.finding_type,
                "evidence": f.evidence[:160],
                "reverted": True,
                "note": f"floor: {reason}{sim_note}",
                "rejected_candidates": rejected_candidates[-8:],
                "rejection_summary": rejection_summary,
            })
            continue

        # All guards passed - accept
        findings_fixed += len(actions_for_sentence)
        consecutive_failed_targets = 0
        current_text = candidate_text
        accepted_candidate_suggestions.append(_manual_suggestion_item(
            finding=f,
            original_sentence=original_sentence,
            candidate_sentence=rewritten_sentence,
            rejection_reason="accepted_locally_pending_final_full_scan",
            paragraph_role=str(edit_brief.get("paragraph_role") or "unknown"),
        ))
        local_score_total += float(best_candidate_info.get("score", 0.0)) if best_candidate_info else 0.0
        rewrite_checkpoints.append({
            "text": current_text,
            "edits": findings_fixed,
            "local_score_total": round(local_score_total, 4),
            "sentence": sent_idx,
            "finding_type": f.finding_type,
            "finding_types": [a.finding.finding_type for a in actions_for_sentence],
            "candidate_score": best_candidate_info.get("score") if best_candidate_info else None,
            "rewrite_operation": rewrite_operation.get("operation"),
        })
        loop_history.append({
            "loop": loops_used,
            "sentence": sent_idx,
            "finding_id": _finding_id(f),
            "finding_type": f.finding_type,
            "finding_types": [a.finding.finding_type for a in actions_for_sentence],
            "rewrite_operation": rewrite_operation.get("operation"),
            "findings_fixed": len(actions_for_sentence),
            "orig_length": len(original_sentence),
            "new_length": len(rewritten_sentence),
            "evidence": f.evidence[:240],
            "orig_text": original_sentence[:240],
            "new_text": rewritten_sentence[:240],
            "candidate_score": best_candidate_info.get("score") if best_candidate_info else None,
            "candidate_attempt": best_candidate_info.get("attempt") if best_candidate_info else None,
            "candidate_count": len(rejected_candidates) + (1 if best_candidate_info else 0),
            "orig_risk": best_candidate_info.get("orig_risk") if best_candidate_info else None,
            "new_risk": best_candidate_info.get("new_risk") if best_candidate_info else None,
            "target_orig_risk": best_candidate_info.get("target_orig_risk") if best_candidate_info else None,
            "target_orig_label": best_candidate_info.get("target_orig_label") if best_candidate_info else None,
            "target_new_risk": best_candidate_info.get("target_new_risk") if best_candidate_info else None,
            "target_new_label": best_candidate_info.get("target_new_label") if best_candidate_info else None,
            "note": "applied candidate",
        })

    # ── Density paragraph pass ───────────────────────────────────────
    # Sentence edits can reduce local top-k signals while leaving the bigger
    # Turnitin-like pattern intact: a dense paragraph made of generic,
    # unsupported, smoothly sequenced claims. When the detector exposes that
    # component, spend at most one extra LLM call on the highest-signal
    # paragraph, then keep the existing final full-scan gate as the authority.
    density_score = float(density_paragraph_pass.get("density_score") or 0.0)
    should_try_density_pass = (
        density_score >= 70.0
        and not density_paragraph_pass.get("attempted")
        and loop_rewrite_fn is not None
        and llm_calls_used < config.max_llm_calls
        and (time.monotonic() - rewrite_start_time) <= config.max_rewrite_seconds
    )
    if should_try_density_pass:
        para_idx, density_paragraph, density_meta = _select_density_paragraph(
            current_text,
            rewrite_context,
            runtime_mitigation_plan,
        )
        density_paragraph_pass.update({
            "attempted": True,
            "paragraph_index": para_idx,
            "paragraph_meta": density_meta,
            "reason": "",
        })
        if para_idx < 0 or not density_paragraph:
            density_paragraph_pass["reason"] = "no_eligible_paragraph"
            loop_history.append({
                "loop": loops_used + 1,
                "note": "density paragraph pass skipped: no eligible paragraph",
                "density_score": round(density_score, 2),
            })
        else:
            llm_calls_used += 1
            density_prompt = _density_paragraph_prompt(
                density_paragraph,
                rewrite_context,
                runtime_mitigation_plan,
            )
            raw_density_output = loop_rewrite_fn(density_paragraph, density_prompt)
            density_candidate = _clean_density_paragraph_output(
                raw_density_output,
                density_paragraph,
            )
            density_reject_reason = _density_paragraph_reject_reason(
                density_paragraph,
                density_candidate,
                rewrite_context,
                allow_polish_warning=True,
            )
            if density_reject_reason:
                density_paragraph_pass["reason"] = density_reject_reason
                if density_candidate:
                    manual_item = {
                        "finding_id": "density_paragraph_rebuild",
                        "finding_type": "qualifying_text_ai_density",
                        "risk_level": "high",
                        "scanner_target": "ai_generation",
                        "sentence_id": None,
                        "paragraph_role": "unknown",
                        "original_sentence": density_paragraph,
                        "suggested_sentence": density_candidate,
                        "rejection_reason": density_reject_reason,
                        "why_review_manually": (
                            "This paragraph-level candidate targets the dense AI-style pattern, "
                            "but an automatic guard rejected it. Review the paragraph manually before using it."
                        ),
                    }
                    key = (
                        manual_item["finding_id"],
                        manual_item["original_sentence"],
                        manual_item["suggested_sentence"],
                        manual_item["rejection_reason"],
                    )
                    existing = {
                        (
                            s.get("finding_id"),
                            s.get("original_sentence"),
                            s.get("suggested_sentence"),
                            s.get("rejection_reason"),
                        )
                        for s in manual_suggestions
                    }
                    if key not in existing and len(manual_suggestions) < 24:
                        manual_suggestions.append(manual_item)
                loop_history.append({
                    "loop": loops_used + 1,
                    "paragraph": para_idx,
                    "note": "density paragraph pass rejected",
                    "rejection_reason": density_reject_reason,
                    "density_score": round(density_score, 2),
                    "orig_text": density_paragraph[:240],
                    "new_text": density_candidate[:240] if density_candidate else "",
                })
            else:
                candidate_text = _splice_density_candidate(
                    current_text,
                    para_idx,
                    density_paragraph,
                    density_candidate,
                    density_meta,
                )
                if candidate_text:
                    voice_check = voice_guard.check(current_text, candidate_text)
                    if not voice_check.accepted:
                        density_paragraph_pass["reason"] = f"voice_eroded {voice_check.reject_reason}"
                        manual_suggestions.append({
                            "finding_id": "density_paragraph_rebuild",
                            "finding_type": "qualifying_text_ai_density",
                            "risk_level": "high",
                            "scanner_target": "ai_generation",
                            "sentence_id": None,
                            "paragraph_role": "unknown",
                            "original_sentence": density_paragraph,
                            "suggested_sentence": density_candidate,
                            "rejection_reason": density_paragraph_pass["reason"],
                            "why_review_manually": (
                                "The paragraph candidate may help the density signal, but it changed voice enough "
                                "that it should be reviewed manually."
                            ),
                        })
                        loop_history.append({
                            "loop": loops_used + 1,
                            "paragraph": para_idx,
                            "note": "density paragraph pass rejected",
                            "rejection_reason": density_paragraph_pass["reason"],
                        })
                    else:
                        current_text = candidate_text
                        findings_fixed += 1
                        loops_used += 1
                        density_paragraph_pass.update({
                            "applied": True,
                            "reason": "accepted_locally_pending_final_full_scan",
                            "orig_length": len(density_paragraph),
                            "new_length": len(density_candidate),
                        })
                        accepted_candidate_suggestions.append({
                            "finding_id": "density_paragraph_rebuild",
                            "finding_type": "qualifying_text_ai_density",
                            "risk_level": "high",
                            "scanner_target": "ai_generation",
                            "sentence_id": None,
                            "paragraph_role": "unknown",
                            "original_sentence": density_paragraph,
                            "suggested_sentence": density_candidate,
                            "rejection_reason": "accepted_locally_pending_final_full_scan",
                            "why_review_manually": (
                                "This paragraph-level edit was accepted locally, but the final full scan remains the authority."
                            ),
                        })
                        rewrite_checkpoints.append({
                            "text": current_text,
                            "edits": findings_fixed,
                            "local_score_total": round(local_score_total, 4),
                            "paragraph": para_idx,
                            "finding_type": "qualifying_text_ai_density",
                            "rewrite_operation": "density_paragraph_rebuild",
                        })
                        loop_history.append({
                            "loop": loops_used,
                            "paragraph": para_idx,
                            "finding_type": "qualifying_text_ai_density",
                            "rewrite_operation": "density_paragraph_rebuild",
                            "orig_length": len(density_paragraph),
                            "new_length": len(density_candidate),
                            "density_score": round(density_score, 2),
                            "orig_text": density_paragraph[:240],
                            "new_text": density_candidate[:240],
                            "note": "applied density paragraph candidate",
                        })
                else:
                    density_paragraph_pass["reason"] = "density_region_splice_failed"
    elif density_score >= 70.0 and not density_paragraph_pass.get("attempted"):
        if loop_rewrite_fn is None:
            density_paragraph_pass["reason"] = "no_llm_available"
        elif llm_calls_used >= config.max_llm_calls:
            density_paragraph_pass["reason"] = f"max_llm_calls reached ({config.max_llm_calls})"
        else:
            density_paragraph_pass["reason"] = (
                f"rewrite time budget exceeded ({config.max_rewrite_seconds}s)"
            )
        loop_history.append({
            "loop": loops_used + 1,
            "note": "density paragraph pass skipped: " + density_paragraph_pass["reason"],
            "density_score": round(density_score, 2),
            "llm_calls_used": llm_calls_used,
        })

    # ── Outer detect-rewrite loop ───────────────────────────────────
    # Re-detect after batch rewrite. If new findings appear, loop again.
    detect_loops_used = 0
    detect_loop_history = []
    prev_findings_count = _count_findings(detect_results)
    outer_loop_start = time.time()
    text_before_outer_loop = current_text  # snapshot for targeted rescan diff

    for detect_loop in range(config.max_detect_loops):
        # Time budget: stop outer loop if we've already spent >300s rewriting
        if time.time() - outer_loop_start > 300:
            detect_loop_history.append({
                "detect_loop": detect_loop,
                "note": "skipped: time budget exceeded",
            })
            break

        # Re-detect current text (targeted: only re-scan changed sentences)
        re_detect_report = _targeted_rescan(
            original_text=text_before_outer_loop,
            rewritten_text=current_text,
            all_detect_results=all_detect_results,
        )
        re_detect_results = re_detect_report.scanner_results
        new_findings_count = _count_findings(re_detect_results)

        detect_loop_history.append({
            "detect_loop": detect_loop,
            "findings_before": prev_findings_count,
            "findings_after": new_findings_count,
        })
        detect_loops_used += 1

        # Stop if no improvement possible
        if new_findings_count == 0:
            break
        if new_findings_count >= prev_findings_count:
            break  # no improvement or got worse

        # Re-plan with new findings
        re_plan = planner.plan(re_detect_results)
        if not re_plan.auto_fixable:
            break  # remaining findings are manual-only

        # Re-group by paragraph and rewrite
        re_guidance = _extract_rewrite_guidance(re_detect_results)
        re_detect_context = _build_detect_context(re_guidance)
        # Preserve the configured gateway — do NOT replace with local claude CLI.
        # Only fall back to chipin if no rewrite_fn was ever created.
        if (
            loop_rewrite_fn is None
            and detect_context
            and os.environ.get("DRAFTPROOF_ENABLE_CLAUDE_FALLBACK") == "1"
        ):
            loop_rewrite_fn = _make_chipin_rewrite_fn(re_detect_context)

        re_sentences, re_para_map = _build_sentence_index(current_text)
        re_para_groups = _group_actions_by_paragraph(
            re_plan.auto_fixable, re_sentences, re_para_map
        )
        re_paragraphs = _split_paragraphs(current_text)

        loop_fixed_this_round = 0
        for para_idx in range(len(re_paragraphs)):
            if para_idx not in re_para_groups:
                continue
            para_actions = re_para_groups[para_idx]
            loops_used += 1

            current_paragraphs = _split_paragraphs(current_text)
            if para_idx >= len(current_paragraphs):
                continue
            region_text = current_paragraphs[para_idx]
            if not region_text:
                continue

            # Build prompt
            remaining = [a for a in para_actions
                         if current_text.find(a.finding.evidence) >= 0]
            if not remaining:
                continue

            findings_lines = []
            for action in remaining:
                f = action.finding
                enriched = _enrich_span_info(f, rewrite_context, -1)
                entry = f"- [{f.finding_type}/{f.risk_level}] \"{f.evidence}\"\n"
                entry += f"  Detail: {f.detail}\n"
                if enriched:
                    entry += f"  Signal: {enriched.strip()}\n"
                entry += f"  Fix: {_derive_strategy(f, enriched)}"
                findings_lines.append(entry)
            span_info = (
                f"Paragraph has {len(remaining)} findings to fix:\n"
                + "\n".join(findings_lines)
                + f"\n\nRewrite the entire paragraph to address ALL {len(remaining)} issues."
            )
            from rewrite.guards import (
                _extract_named_entities, _extract_numbers, _extract_quotes,
            )
            entities = sorted(_extract_named_entities(region_text))
            numbers = sorted(_extract_numbers(region_text))
            quotes = sorted(_extract_quotes(region_text))
            preserve_parts = []
            if entities:
                preserve_parts.append(f"Names/places: {', '.join(entities)}")
            if numbers:
                preserve_parts.append(f"Numbers: {', '.join(numbers)}")
            if quotes:
                preserve_parts.append(f"Quoted text: {'; '.join(quotes)}")
            if preserve_parts:
                span_info += "\n\nMANDATORY: These MUST appear unchanged in your output:\n" + "\n".join(f"- {p}" for p in preserve_parts)

            # Rewrite with retry
            rewritten_region = None
            drift = None
            for attempt in range(2):
                if loop_rewrite_fn:
                    rewritten_region = loop_rewrite_fn(region_text, span_info)
                if rewritten_region is None:
                    break
                drift = check_semantic_drift(region_text, rewritten_region, threshold=0.1)
                if drift.accepted:
                    break
                if attempt == 0:
                    span_info += (
                        "\n\nYOUR PREVIOUS ATTEMPT FAILED drift check."
                        "\nTry again. Keep meaning identical but vary phrasing more."
                    )

            if rewritten_region and drift and drift.accepted:
                new_paragraphs = _split_paragraphs(current_text)
                new_paragraphs[para_idx] = rewritten_region
                current_text = "\n\n".join(new_paragraphs)
                findings_fixed += len(remaining)
                loop_fixed_this_round += len(remaining)
                loop_history.append({
                    "loop": loops_used,
                    "detect_loop": detect_loop + 1,
                    "paragraph": para_idx,
                    "findings_count": len(remaining),
                    "drift_similarity": round(drift.similarity, 3),
                    "note": f"outer-loop batch fixed {len(remaining)} findings",
                })

        if loop_fixed_this_round == 0:
            break  # no progress this round
        prev_findings_count = new_findings_count

    # Build final MultiPassResult
    # ── Targeted rescan: only re-scan changed sentences for predictability ──
    # The predictability scanner is ~25s for 100 sentences. Most sentences
    # are unchanged after rewrite, so we carry forward their original scores
    # and only re-scan the ones that actually changed.
    final_detect_report = _targeted_rescan(
        original_text=content,
        rewritten_text=current_text,
        all_detect_results=all_detect_results,
    )
    final_detect_results = final_detect_report.scanner_results

    # Original metrics: use unfiltered detect results (includes predictability scanner)
    # to avoid re-running predictability on the original text (saves ~29s).
    from detect.base import DetectionReport
    orig_report = DetectionReport(
        scanner_results=all_detect_results,
        overall_risk=max((dr.overall_risk for dr in all_detect_results), default=0),
        overall_review_priority="low",
        confidence="medium",
    )
    original_metrics = _metrics_from_detect(orig_report, content)

    # Compare the final scan against the same target scope that was eligible
    # for rewrite. A local sentence rewrite can pass guards while increasing
    # document-level AI likelihood or producing more medium target findings.
    original_target_findings = [f for dr in detect_results for f in dr.findings]
    final_target_findings = _target_findings_for_mode(final_detect_results, ai_only, rewrite_context)
    original_target_weight = weighted_finding_score(original_target_findings)
    final_target_weight = weighted_finding_score(final_target_findings)
    final_ai_likelihood = _ai_likelihood_from_results(final_detect_results)
    ai_regressed = final_ai_likelihood > original_ai_likelihood + 0.005
    target_regressed = final_target_weight > original_target_weight
    no_target_gain = (
        final_target_weight >= original_target_weight
        and final_ai_likelihood >= original_ai_likelihood
    )
    rolled_back_for_regression = False  # Removed: per-sentence guards handle quality

    # Log regression info for transparency without rolling back
    if current_text != content and (ai_regressed or target_regressed or no_target_gain):
        reason_bits = []
        if ai_regressed:
            reason_bits.append(
                f"ai_likelihood {original_ai_likelihood:.3f}->{final_ai_likelihood:.3f}"
            )
        if target_regressed:
            reason_bits.append(
                f"target_weight {original_target_weight:.1f}->{final_target_weight:.1f}"
            )
        if no_target_gain and not reason_bits:
            reason_bits.append("no target-scope improvement")
        reason = "; ".join(reason_bits)
        loop_history.append({
            "loop": loops_used + 1,
            "reverted": False,
            "note": f"regression noted (not rolled back): {reason}",
            "original_ai_likelihood": round(original_ai_likelihood, 4),
            "final_ai_likelihood": round(final_ai_likelihood, 4),
            "original_target_weight": round(original_target_weight, 4),
            "final_target_weight": round(final_target_weight, 4),
        })

    # True fix count = original target findings minus remaining target findings.
    # This intentionally uses target-scope findings, not all findings, so AI-only
    # rewrites are not judged against unrelated low-severity scanner noise.
    final_finding_count = len(final_target_findings)
    original_finding_count = len(original_target_findings)
    net_findings_fixed = original_finding_count - final_finding_count
    target_weight_delta = original_target_weight - final_target_weight
    ai_likelihood_delta = original_ai_likelihood - final_ai_likelihood
    meaningful_global_improvement = (
        net_findings_fixed >= 1
        or target_weight_delta >= max(2.0, config.min_weighted_improvement)
        or ai_likelihood_delta >= 0.01
    )
    if current_text != content and not rolled_back_for_regression and not meaningful_global_improvement:
        rolled_back_for_regression = True
        rollback_reason = (
            "rewrite produced no meaningful final-scan mitigation "
            f"(ai_delta={ai_likelihood_delta:.4f}, target_weight_delta={target_weight_delta:.1f}, "
            f"net_findings_fixed={net_findings_fixed})"
        )
        floor_reasons.append(FloorReason(
            finding_id="global_mitigation_gate",
            reason_type="standard_phrase",
            explanation=rollback_reason,
        ))
        for accepted in accepted_candidate_suggestions[:6]:
            suggestion = dict(accepted)
            suggestion["rejection_reason"] = "low_value_rewrite_not_kept"
            suggestion["why_review_manually"] = (
                "This edit passed local guards, but the final scan movement was too small to count as mitigation."
            )
            manual_suggestions.append(suggestion)
        current_text = content
        final_detect_report = orig_report
        final_detect_results = all_detect_results
        final_target_findings = original_target_findings
        final_target_weight = original_target_weight
        final_ai_likelihood = original_ai_likelihood
        final_finding_count = original_finding_count
        net_findings_fixed = 0
        loop_history.append({
            "loop": loops_used + 1,
            "reverted": True,
            "note": rollback_reason,
            "original_ai_likelihood": round(original_ai_likelihood, 4),
            "final_ai_likelihood": round(final_ai_likelihood, 4),
            "original_target_weight": round(original_target_weight, 4),
            "final_target_weight": round(final_target_weight, 4),
        })

    # Reuse predictability results from final_detect_report instead of
    # running compute_metrics again (saves ~28s per call).
    final_metrics = (
        original_metrics
        if rolled_back_for_regression
        else _metrics_from_detect(final_detect_report, current_text)
    )

    result = MultiPassResult(
        original_text=content,
        original_metrics=original_metrics,
        passes=[final_metrics],
        final_text=current_text,
        final_metrics=final_metrics,
        converged=(net_findings_fixed > 0 and findings_skipped == 0 and not rolled_back_for_regression),
        convergence_reason=f"Batch rewrite: {net_findings_fixed} net fixed (of {original_finding_count}), {findings_skipped} hit floor, {detect_loops_used} outer loops",
        style_profile=style_profile if 'style_profile' in dir() else None,
        style_suggestions=[],
        regression_memory=regression_memory.summary(),
        protected_spans_count=len(detect_protected_spans(content)),
        drift_score=0.0,
    )

    # ── Report ─────────────────────────────────────────────────────
    builder = ReportBuilder()
    builder.set_meta(
        scan_time=result.passes[-1].risk if result.passes else 0,
    )
    for dr in detect_results:
        builder.add_detection(dr)
    if result.final_text != content:
        builder.add_rewrite(result)
    report = builder.build()

    text_rpt = render_report(report)
    md_rpt = render_markdown(report)

    # Write report files
    md_path = json_path = None
    summary = get_rewrite_summary_v2(
        mp_result=result,
        plan=plan,
        loop_history=loop_history,
        manual_only=False,
        detect_loop_history=detect_loop_history,
        detect_loops_used=detect_loops_used,
        raw_json=getattr(rewrite_context, "raw_json", None),
    )
    summary["rollback_applied"] = rolled_back_for_regression
    summary["rewrite_runtime_version"] = REWRITE_RUNTIME_VERSION
    summary["rewrite_effective_config"] = {
        "max_llm_calls": config.max_llm_calls,
        "sentence_llm_call_limit": sentence_llm_call_limit,
        "density_mitigation_llm_calls": density_mitigation_llm_calls,
        "max_density_passes": getattr(config, "max_density_passes", 1),
        "density_mitigation_priority": "before_sentence_rewrites",
        "max_auto_targets": config.max_auto_targets,
        "effective_auto_target_limit": effective_auto_target_limit,
        "max_failed_targets": config.max_failed_targets,
        "max_consecutive_failed_targets": config.max_consecutive_failed_targets,
        "max_rewrite_seconds": config.max_rewrite_seconds,
        "max_detect_loops": config.max_detect_loops,
    }
    summary["mitigation_primary_mode_at_runtime"] = runtime_mitigation_plan.get("primary_mode")
    if guided_throttle_reason:
        summary["guided_revision_throttle"] = guided_throttle_reason
    summary["llm_calls_used"] = llm_calls_used
    summary["target_count"] = len(grouped_actions)
    summary["unique_target_count"] = len(grouped_actions)
    summary["selected_finding_count"] = len(auto_targets)
    summary["accepted_edits"] = findings_fixed
    summary["density_paragraph_pass"] = density_paragraph_pass
    summary["manual_suggestions"] = manual_suggestions
    summary["accepted_candidate_suggestions"] = accepted_candidate_suggestions
    if capped_auto_targets:
        summary["auto_target_cap"] = {
            "max_auto_targets": config.max_auto_targets,
            "effective_auto_target_limit": effective_auto_target_limit,
            "available_auto_targets": len(plan.auto_fixable),
            "skipped_auto_targets": capped_auto_targets,
            "reason": guided_throttle_reason or "max_auto_targets",
        }
    summary["failed_targets"] = failed_targets
    summary["consecutive_failed_targets"] = consecutive_failed_targets
    if circuit_breaker_reason:
        summary["circuit_breaker"] = {
            "triggered": True,
            "reason": circuit_breaker_reason,
            "max_llm_calls": config.max_llm_calls,
            "max_failed_targets": config.max_failed_targets,
            "max_consecutive_failed_targets": config.max_consecutive_failed_targets,
            "max_rewrite_seconds": config.max_rewrite_seconds,
        }
    if rolled_back_for_regression:
        summary["rollback_reason"] = floor_reasons[-1].explanation if floor_reasons else "final scan regression"
        summary["original_ai_likelihood_internal"] = round(original_ai_likelihood, 4)
        summary["final_ai_likelihood_internal"] = round(final_ai_likelihood, 4)
        summary["original_target_weight"] = round(original_target_weight, 4)
        summary["final_target_weight"] = round(final_target_weight, 4)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        md_path = os.path.join(output_dir, f"draftproof_{ts}.md")
        json_path = os.path.join(output_dir, f"draftproof_{ts}.json")
        with open(md_path, "w") as f:
            f.write(md_rpt)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

    # ── Classify outcome ────────────────────────────────────────────
    if rolled_back_for_regression:
        outcome = RewriteOutcome.REJECTED_FOR_DRIFT
    elif findings_fixed > 0 and findings_skipped == 0:
        outcome = RewriteOutcome.IMPROVED
    elif findings_fixed > 0 and findings_skipped > 0:
        outcome = RewriteOutcome.PARTIALLY_IMPROVED
    elif (
        findings_skipped > 0
        and findings_fixed == 0
        and (runtime_mitigation_plan.get("marked_content_suggestions") or [])
    ):
        outcome = RewriteOutcome.SUGGESTION_ONLY
    elif findings_skipped > 0 and findings_fixed == 0:
        outcome = RewriteOutcome.FLOOR_REACHED
    elif plan.manual_required and not plan.auto_fixable:
        outcome = RewriteOutcome.MANUAL_REQUIRED
    else:
        outcome = RewriteOutcome.FLOOR_REACHED
    summary["outcome"] = outcome.value
    if json_path:
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"  Predictability guard: {predictability_guard.stats}")

    return RewriteModuleResult(
        mp_result=result,
        report=report,
        text_report=text_rpt,
        markdown_report=md_rpt,
        summary=summary,
        report_md_path=md_path,
        report_json_path=json_path,
        post_rewrite_detect=final_detect_results,
        detect_loops_used=detect_loops_used,
        detect_loop_history=detect_loop_history,
        rewrite_plan=plan,
        manual_actions=plan.manual_required,
        regression_memory_summary=regression_memory.summary(),
        outcome=outcome,
        floor_reasons=floor_reasons,
        voice_profile=voice_profile,
        rewrite_surface=rewrite_surface,
        voice_guard_warnings=[],
        final_detect_report=final_detect_report,
        rewrite_checkpoints=rewrite_checkpoints,
    )


def _targeted_rescan(
    original_text: str,
    rewritten_text: str,
    all_detect_results: List[DetectResult],
) -> "DetectionReport":
    """Run a targeted rescan after rewrite — only re-score changed sentences.

    Strategy:
    - Fast scanners (ai_generation, citation, etc.): run normally on full text (~1s)
    - Predictability scanner: only re-scan sentences that actually changed,
      carry forward unchanged sentence scores from the original detect.

    This cuts predictability rescan from ~25s (101 sentences) to ~2-3s (changed only).
    Falls back to full scan if original predictability data is unavailable.
    """
    from detect.base import DetectionReport, DetectResult as DR

    # If text didn't change, return original detect results wrapped as a report
    if original_text == rewritten_text:
        orig_report = DetectionReport(
            scanner_results=all_detect_results,
            overall_risk=max((dr.overall_risk for dr in all_detect_results), default=0),
            overall_review_priority="low",
            confidence="medium",
        )
        logger.info("Targeted rescan: text unchanged, reusing original detect results")
        return orig_report

    # Try to extract original predictability sentence details
    orig_pred_raw = None
    orig_pred_scanner_result = None
    for sr in all_detect_results:
        if sr.scanner == "predictability" and sr.raw:
            orig_pred_raw = sr.raw
            orig_pred_scanner_result = sr
            break

    # If no original predictability data, fall back to full scan
    if not orig_pred_raw:
        logger.info("Targeted rescan: no original predictability data, falling back to full scan")
        runner = DetectionRunner()
        return runner.run_all(rewritten_text)

    orig_sentences = orig_pred_raw.get("sentences", [])
    if not orig_sentences:
        logger.info("Targeted rescan: no original sentence details, falling back to full scan")
        runner = DetectionRunner()
        return runner.run_all(rewritten_text)

    # Normalize for comparison (handle curly quotes, dashes, whitespace)
    def _norm(text: str) -> str:
        import unicodedata
        text = unicodedata.normalize("NFKC", text).strip()
        text = text.replace("’", "'").replace("‘", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("–", "-").replace("—", "-")
        return " ".join(text.split())

    # Build text-based lookup from original predictability sentences
    # Report JSON may only contain flagged sentences, so this is a partial lookup
    orig_sent_lookup = {}
    for s in orig_sentences:
        if isinstance(s, dict):
            text_val = (s.get("sentence") or s.get("text") or "").strip()
        else:
            text_val = (getattr(s, "sentence", None) or getattr(s, "text", "")).strip()
        if text_val:
            orig_sent_lookup[_norm(text_val)] = s

    # Split both texts into sentences for position-based diffing
    scanner = PredictabilityScanner()
    orig_split = scanner.split_sentences(original_text)
    new_split = scanner.split_sentences(rewritten_text)

    # Position-based diff: compare sentences at same index
    # If sentence count changed, mark all from the divergence point as changed
    changed_indices = []
    unchanged_indices = []
    min_len = min(len(orig_split), len(new_split))

    for i in range(min_len):
        if _norm(str(orig_split[i])) == _norm(str(new_split[i])):
            unchanged_indices.append(i)
        else:
            changed_indices.append(i)

    # If lengths differ, extra sentences are all changed
    for i in range(min_len, len(new_split)):
        changed_indices.append(i)

    # Run predictability only on changed sentences
    eligible_changed = [new_split[i] for i in changed_indices if len(str(new_split[i]).split()) >= 8]

    logger.info(
        "Targeted rescan: %d changed (%d eligible) / %d unchanged — scanning only changed",
        len(changed_indices), len(eligible_changed), len(unchanged_indices),
    )

    changed_results = []
    if eligible_changed:
        t0 = time.time()
        for s in eligible_changed:
            sr = scanner.scan_sentence(str(s))
            changed_results.append(sr)
        logger.info(
            "Targeted rescan: predictability done in %.1fs (%d sentences)",
            time.time() - t0, len(eligible_changed),
        )

    # Merge: carry forward unchanged scores + new changed scores
    merged_results = []
    changed_idx = 0
    for i, s in enumerate(new_split):
        if i in unchanged_indices:
            # Try text-based lookup first (has original scores)
            s_text = _norm(str(s))
            if s_text in orig_sent_lookup:
                merged_results.append(orig_sent_lookup[s_text])
            else:
                # Sentence wasn't in the partial report JSON — scan it fresh
                if changed_idx < len(changed_results):
                    sr = changed_results[changed_idx]
                    sr.start_char = getattr(s, "start_char", 0)
                    sr.end_char = getattr(s, "end_char", 0)
                    sr.paragraph_id = getattr(s, "paragraph_id", "p001")
                    merged_results.append(sr)
                    changed_idx += 1
        else:
            if changed_idx < len(changed_results):
                sr = changed_results[changed_idx]
                sr.start_char = getattr(s, "start_char", 0)
                sr.end_char = getattr(s, "end_char", 0)
                sr.paragraph_id = getattr(s, "paragraph_id", "p001")
                merged_results.append(sr)
                changed_idx += 1

    # Normalize SentenceResult objects to dicts so downstream .get() calls work
    from dataclasses import asdict
    normalized = []
    for r in merged_results:
        if isinstance(r, dict):
            normalized.append(r)
        else:
            try:
                normalized.append(asdict(r))
            except TypeError:
                normalized.append({
                    "sentence": getattr(r, "sentence", ""),
                    "risk_label": getattr(r, "risk_label", "low"),
                    "predictability_risk": getattr(r, "predictability_risk", 0),
                    "avg_probability": getattr(r, "avg_probability", 0),
                    "avg_surprisal": getattr(r, "avg_surprisal", 0),
                    "top_10_ratio": getattr(r, "top_10_ratio", 0),
                    "top_50_ratio": getattr(r, "top_50_ratio", 0),
                    "matched_generic_phrases": getattr(r, "matched_generic_phrases", []),
                    "error": getattr(r, "error", None),
                    "start_char": getattr(r, "start_char", 0),
                    "end_char": getattr(r, "end_char", 0),
                    "paragraph_id": getattr(r, "paragraph_id", ""),
                })
    merged_results = normalized

    # Build merged predictability raw dict
    valid = [r for r in merged_results
             if r.get("error") is None]
    overall_risk = 0.0
    if valid:
        risks = []
        for r in valid:
            # All items are now dicts after normalization
            risk_val = (r.get("predictability_risk")
                        or r.get("score")
                        or r.get("risk", 0))
            if isinstance(risk_val, str):
                risk_val = 0
            risks.append(risk_val)
        overall_risk = sum(risks) / len(risks) if risks else 0.0

    merged_pred_raw = {
        "overall_risk": round(overall_risk, 4),
        "sentences": merged_results,
    }

    # Run fast scanners on full text (skip predictability — we handle it above)
    from detect.ai_generation import AIGenerationSignalDetector
    from detect.citation import CitationDetector
    fast_detectors = [AIGenerationSignalDetector(), CitationDetector()]
    fast_runner = DetectionRunner(detectors=fast_detectors)
    fast_report = fast_runner.run_all(rewritten_text)

    # Replace predictability scanner result with our merged version
    pred_findings = _predictability_findings_from_raw(merged_results)
    merged_scanner_results = []
    pred_replaced = False
    for sr in fast_report.scanner_results:
        if sr.scanner == "predictability" and not pred_replaced:
            merged_scanner_results.append(DR(
                scanner="predictability",
                overall_risk=overall_risk,
                confidence=orig_pred_scanner_result.confidence if orig_pred_scanner_result else "medium",
                confidence_reason="targeted rescan: changed sentences re-scored, unchanged carried forward",
                risk_distribution=orig_pred_scanner_result.risk_distribution if orig_pred_scanner_result else {},
                findings=pred_findings,
                policy_message=orig_pred_scanner_result.policy_message if orig_pred_scanner_result else "",
                raw=merged_pred_raw,
            ))
            pred_replaced = True
        else:
            merged_scanner_results.append(sr)

    if not pred_replaced:
        merged_scanner_results.append(DR(
            scanner="predictability",
            overall_risk=overall_risk,
            confidence="medium",
            confidence_reason="targeted rescan: changed sentences re-scored, unchanged carried forward",
            risk_distribution={},
            findings=pred_findings,
            policy_message="",
            raw=merged_pred_raw,
        ))

    return DetectionReport(
        scanner_results=merged_scanner_results,
        overall_risk=max((dr.overall_risk for dr in merged_scanner_results), default=0),
        overall_review_priority="low",
        confidence="medium",
    )


def _predictability_findings_from_raw(merged_results: list) -> list:
    """Convert merged predictability sentence results to Finding objects."""
    findings = []
    for r in merged_results:
        risk = (r.get("predictability_risk")
                or r.get("score")
                or (0 if isinstance(r.get("risk"), str) else r.get("risk", 0)))
        label = r.get("risk_label") or r.get("risk", "low")
        sentence = r.get("sentence", "")
        if risk >= 0.40 and sentence:
            findings.append(Finding(
                finding_type="predictability",
                risk_level="high" if risk >= 0.65 else "medium" if risk >= 0.50 else "review",
                evidence_strength="moderate",
                detail=f"Predictability risk {risk:.2f} ({label})",
                evidence=sentence[:200],
                recommendation="Rephrase to reduce predictability",
                suggested_action_type="suggest_rewrite",
                location={"sentence_id": f"s{len(findings)+1:03d}"},
                metadata={"scanner": "predictability"},
            ))
    return findings


def _count_findings(detect_results: List[DetectResult]) -> int:
    return sum(len(dr.findings) for dr in detect_results)


def get_rewrite_summary_v2(
    mp_result: Optional[MultiPassResult] = None,
    plan: Optional[RewritePlan] = None,
    loop_history: Optional[List[dict]] = None,
    manual_only: bool = False,
    detect_loop_history: Optional[List[dict]] = None,
    detect_loops_used: int = 0,
    raw_json: Optional[dict] = None,
) -> dict:
    """Build summary dict with new multi-signal fields."""
    summary = {
        "manual_only": manual_only,
    }

    if plan:
        summary["rewrite_plan"] = {
            "auto_fixable": len(plan.auto_fixable),
            "manual_required": len(plan.manual_required),
            "protected": len(plan.protected),
            "review_only": len(plan.review_only),
            "total_weighted_risk": plan.total_weighted_risk,
            "auto_risk": plan.auto_risk,
            "manual_risk": plan.manual_risk,
            "protected_risk": plan.protected_risk,
            "rewritable_risk": plan.rewritable_risk,
        }
        summary["manual_actions"] = [
            {
                "finding_type": a.finding.finding_type,
                "risk_level": a.finding.risk_level,
                "action": a.action_type,
                "fixability": a.fixability,
                "reason": a.reason,
                "evidence": a.finding.evidence[:100],
            }
            for a in plan.manual_required
        ]
        summary["protected_actions"] = [
            {
                "finding_type": a.finding.finding_type,
                "risk_level": a.finding.risk_level,
                "reason": a.reason,
            }
            for a in plan.protected
        ]
        summary["review_only_actions"] = [
            {
                "finding_type": a.finding.finding_type,
                "risk_level": a.finding.risk_level,
                "fixability": a.fixability,
                "reason": a.reason,
                "evidence": a.finding.evidence[:100],
            }
            for a in plan.review_only
        ]
        summary["mitigation_plan"] = build_mitigation_plan(plan, raw_json)

    if mp_result:
        summary.update({
            "original_risk": mp_result.original_metrics.risk,
            "original_top10": mp_result.original_metrics.top10_ratio,
            "final_risk": mp_result.final_metrics.risk,
            "final_top10": mp_result.final_metrics.top10_ratio,
            "improvement_risk": round(
                mp_result.original_metrics.risk - mp_result.final_metrics.risk, 4
            ),
            "improvement_top10": round(
                mp_result.original_metrics.top10_ratio - mp_result.final_metrics.top10_ratio, 4
            ),
            "passes_completed": len(mp_result.passes),
            "converged": mp_result.converged,
            "convergence_reason": mp_result.convergence_reason,
            "pass_progression": [
                {
                    "pass": p.pass_number,
                    "risk": p.risk,
                    "top10_ratio": p.top10_ratio,
                    "surprisal": p.surprisal,
                }
                for p in mp_result.passes
            ],
            "style_suggestions": mp_result.style_suggestions,
            "original_text": mp_result.original_text,
            "final_text": mp_result.final_text,
        })

    if loop_history:
        summary["loop_history"] = loop_history

    if detect_loop_history:
        summary["detect_loop_history"] = detect_loop_history
        summary["detect_loops_used"] = detect_loops_used

    return summary


# Backward compatibility
def get_rewrite_summary(result: MultiPassResult) -> dict:
    """Legacy summary for backward compatibility."""
    return get_rewrite_summary_v2(mp_result=result)
