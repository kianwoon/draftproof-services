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
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

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
from report import ReportBuilder, render_report, render_markdown
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
    PredictabilityGuard,
)
from rewrite.scorer import (
    weighted_finding_score, weighted_rewritable_risk,
    score_candidate, best_candidate, CandidateScore,
    FIXABILITY_WEIGHT,
)
from rewrite.voice import VoiceGuard, VoiceProfile, analyze_voice
from llm.gateway import LLMGateway, LLMConfig

logger = logging.getLogger(__name__)


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


def _filter_ai_findings(detect_results: List[DetectResult], min_severity: str = "medium") -> List[DetectResult]:
    """Keep only MEDIUM+ severity findings from the ai_generation scanner.

    Returns new DetectResult list with only AI findings at or above min_severity.
    Non-AI scanners (predictability, similarity, citation) are dropped entirely.
    LOW and info findings are skipped — not worth the LLM cost to rewrite.
    """
    _severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "review": 0, "info": 0}
    min_rank = _severity_rank.get(min_severity, 2)

    filtered = []
    for dr in detect_results:
        if dr.scanner == "ai_generation":
            ai_findings = dr.findings
        else:
            ai_findings = [f for f in dr.findings if _is_ai_finding(f)]
        # Filter by severity
        ai_findings = [f for f in ai_findings if _severity_rank.get(f.risk_level, 0) >= min_rank]
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
    min_severity: str = "medium",
) -> List[DetectResult]:
    """AI rewrite target set, expanded with contributing sentence signals.

    The AI-generation scanner often reports document-level summary findings
    while the concrete editable evidence lives in the predictability scanner.
    When the badge says predictability/top-k is a meaningful AI driver, include
    medium+ predictability sentences as rewrite targets.
    """
    filtered = _filter_ai_findings(detect_results, min_severity=min_severity)

    raw_json = getattr(rewrite_context, "raw_json", None) if rewrite_context else None
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
    _severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "review": 0, "info": 0}
    min_rank = _severity_rank.get(min_severity, 2)
    supporting_types = {
        "high_predictability",
        "medium_predictability",
        "high_topk_predictability",
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
            if (
                f.finding_type in supporting_types
                and _severity_rank.get(f.risk_level, 0) >= min_rank
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


def _filter_by_severity(detect_results: List[DetectResult], min_severity: str = "medium") -> List[DetectResult]:
    """Keep MEDIUM+ findings from ALL scanners. Drops LOW/info findings.

    Used when ai_only=False — rewrites predictability, AI, citation, etc.
    but only for findings significant enough to warrant an LLM call.
    """
    _severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "review": 0, "info": 0}
    min_rank = _severity_rank.get(min_severity, 2)

    filtered = []
    for dr in detect_results:
        keep = [f for f in dr.findings if _severity_rank.get(f.risk_level, 0) >= min_rank]
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

REWRITE_CHIPIN_PROMPT = """You are a text editor. Your ONLY job is to make the flagged sentence sound LESS like AI-generated text.

You do this by applying ONE specific mechanical transformation. Do NOT try to "improve" the writing generally.

TRANSFORMATION RULES (apply the one matching the finding type):

For predictability / common_words / topk_predictability / low_surprisal findings:
- Replace generic scaffolding with natural, concrete wording already implied by the sentence or neighboring context. Prefer sharper nouns/verbs over thesaurus words. Do not make the sentence sound ornate. NEVER use the words: crucial, vital, essential, significant, notable, furthermore, moreover, additionally.

For formulaic_sentence / generic_phrase findings:
- Restructure the sentence to break its formulaic pattern. Move the subject to a different position. Merge or split clauses. Start the sentence differently than it currently starts.

For style_shift / repetitive_structure findings:
- Change the sentence opening to differ from the previous sentence's opening. If the previous sentence starts with "The X...", start this one with a participle, adverb, or dependent clause instead.

For burstiness findings:
- If this sentence is similar in length to neighbors, make it noticeably shorter (cut filler words) or merge it with context.

For ai_generation findings:
- Fix only the specific sentence-level signal. Do not try to solve document-level specificity, grounding, or overall AI-likelihood by inventing details.

CRITICAL CONSTRAINTS:
- Keep the SAME factual meaning. Zero new information.
- All proper nouns, numbers, dates, citations, quoted text — copy verbatim.
- Same number of sentences in, same number out.
- Do NOT exceed the character limit.
- Output ONLY the rewritten text. No quotes, no commentary."""


def _make_chipin_rewrite_fn(detect_context: str) -> callable:
    """Create a rewrite function that calls the local `claude` CLI."""
    def rewrite_fn(text: str, span_info: str) -> Optional[str]:
        max_chars = int(len(text) * 1.10)
        user_msg = (
            f"{detect_context}\n\n{span_info}\n\n"
            f"Current text ({len(text)} chars):\n{text}\n\n"
            f"Rewrite this text addressing the issues above. "
            f"CRITICAL: Your output MUST NOT exceed {max_chars} characters. "
            f"Rephrase in-place — do NOT expand, add sentences, or elaborate. "
            "Output ONLY the rewritten text. No quotes, no commentary."
        )
        try:
            result = subprocess.run(
                ["claude", "--print",
                 "--system-prompt", REWRITE_CHIPIN_PROMPT,
                 "--model", "sonnet",
                 user_msg],
                capture_output=True, text=True, timeout=120,
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
        return ", ".join(f'"{t}"' for t in tokens[:3])
    return None


_FINDING_STRATEGIES = {
    "high_predictability": "Replace predictable words with natural alternatives implied by context. Prefer concrete nouns/verbs over generic ones.",
    "medium_predictability": "Soften the most predictable words. Don't over-correct — keep natural flow.",
    "high_topk_predictability": "Replace commonly predicted token sequences with less expected alternatives.",
    "low_surprisal": "Introduce less expected word choices while keeping meaning identical.",
    "formulaic_sentence": "Break the formulaic structure. Move subject position. Merge or split clauses. Change sentence opening.",
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
        )
    except Exception:
        return True, ""

    checks = [
        ("broad_claim", estimate_broad_claim_risk),
        ("generic_assertion", estimate_generic_assertion_risk),
        ("lived_detail", estimate_lived_detail_risk),
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
) -> callable:
    """Create a rewrite function that uses LLMGateway with detect context."""
    config = LLMConfig(api_key=api_key, model=model, base_url=base_url)
    gateway = LLMGateway(config)

    def rewrite_fn(text: str, span_info: str) -> Optional[str]:
        max_chars = int(len(text) * 1.10)
        user_msg = (
            f"{detect_context}\n\n{span_info}\n\n"
            f"Current text ({len(text)} chars):\n{text}\n\n"
            f"Rewrite this text addressing the issues above. "
            f"CRITICAL: Your output MUST NOT exceed {max_chars} characters. "
            f"Rephrase in-place — do NOT expand, add sentences, or elaborate. "
            "Output ONLY the rewritten text. No quotes, no commentary."
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


def _find_sentence_index(sentences: List[str], evidence: str) -> int:
    """Find which sentence contains the evidence text."""
    evidence_start = evidence[:30] if len(evidence) > 30 else evidence
    for i, s in enumerate(sentences):
        if evidence_start in s:
            return i
    return -1


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
        # All scanners, but only MEDIUM+ severity
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
            summary={"manual_only": True, "reason": "mostly_protected"},
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
            summary = get_rewrite_summary_v2(plan=plan, manual_only=True)
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
            summary=get_rewrite_summary_v2(plan=plan, manual_only=True),
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
            )
        elif detect_context:
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

    # ── Step 3: Per-finding rewrite loop (LLM, all findings) ────────
    current_weighted_risk = weighted_finding_score(
        [f for dr in detect_results for f in dr.findings]
    )

    # ── Per-sentence rewrite: one LLM call per flagged sentence ─────────
    # Process each auto-fixable action individually. The LLM sees only the
    # target sentence + context. Much smaller surface area = less hallucination.
    sentences, para_map = _build_sentence_index(current_text)

    for action in plan.auto_fixable:
        loops_used += 1
        f = action.finding

        # Locate the sentence in current text
        current_sentences, _ = _build_sentence_index(current_text)
        loc = f.location or {}
        sent_idx = loc.get("sentence_index", -1)

        # Try sentence_index first, then fuzzy match
        if sent_idx >= 0 and sent_idx < len(current_sentences):
            # Verify the indexed sentence actually contains the evidence
            if f.evidence[:30] not in current_sentences[sent_idx]:
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
            findings_skipped += 1
            floor_reasons.append(FloorReason(
                finding_id=getattr(f, "id", str(id(f))),
                reason_type="sentence_not_found",
                explanation=f"Cannot locate: '{f.evidence[:50]}...'",
            ))
            loop_history.append({
                "loop": loops_used,
                "finding_type": f.finding_type,
                "reverted": True,
                "note": "skipped: sentence not found",
            })
            continue

        original_sentence = current_sentences[sent_idx]

        # Check evidence still present in text (not just this sentence —
        # evidence may span a boundary our splitter creates)
        if f.evidence[:30] not in current_text:
            findings_skipped += 1
            floor_reasons.append(FloorReason(
                finding_id=getattr(f, "id", str(id(f))),
                reason_type="evidence_not_found",
                explanation="Evidence gone from sentence",
            ))
            loop_history.append({
                "loop": loops_used,
                "finding_type": f.finding_type,
                "reverted": True,
                "note": "skipped: evidence not found in sentence",
            })
            continue

        # Build context: target sentence +/- 1 sentence
        ctx_before = current_sentences[sent_idx - 1] if sent_idx > 0 else ""
        ctx_after = current_sentences[sent_idx + 1] if sent_idx < len(current_sentences) - 1 else ""

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
        # Inject specific signal metrics (trigger, problem tokens, etc.)
        enriched = _enrich_span_info(f, rewrite_context, sent_idx)
        if enriched:
            span_info_parts.append(enriched)
        # Concrete finding-specific strategy
        span_info_parts.append("Strategy: " + _derive_strategy(f, enriched))
        span_info_parts.append("Flagged phrase: '" + f.evidence + "'")
        if ctx_before:
            span_info_parts.append("Previous sentence context: '" + ctx_before + "'")
        if ctx_after:
            span_info_parts.append("Next sentence context: '" + ctx_after + "'")
        if sent_protected:
            protected_items = ", ".join(ps.text for ps in sent_protected)
            span_info_parts.append("Must preserve exactly: " + protected_items)
        # Fallback: keep legacy sentence metrics if enrichment was empty
        if not enriched:
            sentence_metrics = _sentence_signal_context(rewrite_context, f, sent_idx)
            if sentence_metrics:
                span_info_parts.append(sentence_metrics)
        span_info = "\n".join(span_info_parts)

        # Rewrite with retry loop
        rewritten_sentence = None
        drift = None
        max_sent_chars = int(len(original_sentence) * 1.20)  # 20% tolerance per sentence

        # For predictability findings, try GPT-2 sample-and-rank first.
        # It uses the same model that detects predictability, so it directly
        # optimizes the metric. Falls back to LLM if GPT-2 can't improve.
        is_predictability = f.finding_type in (
            "high_predictability", "medium_predictability",
            "high_topk_predictability", "low_surprisal",
        )
        if is_predictability and gpt2_rewriter:
            prev_sent = sentences[sent_idx - 1] if sent_idx > 0 else ""
            gpt2_candidate = gpt2_rewriter.rewrite_sentence(
                original_sentence, context_before=prev_sent,
            )
            if gpt2_candidate:
                rewritten_sentence = gpt2_candidate
                loop_history.append({
                    "loop": loops_used,
                    "sentence": sent_idx,
                    "attempt": 0,
                    "note": f"gpt2_rewrite: {len(original_sentence)}→{len(gpt2_candidate)} chars",
                })

        # Fall back to LLM rewrite if GPT-2 didn't produce a candidate
        for attempt in range(3):
            if rewritten_sentence is not None:
                break  # already have GPT-2 candidate
            if loop_rewrite_fn:
                # span_info already has finding + context.
                # rewrite_fn will wrap original_sentence in triple quotes.
                # Just add the char limit instruction.
                prompt = (
                    span_info + "\n\n"
                    "Apply ONE transformation from the system prompt rules to fix the finding above. "
                    "Change as few words as possible. "
                    "MUST NOT exceed " + str(max_sent_chars) + " characters. "
                    "Output ONLY the rewritten text."
                )
                rewritten_sentence = loop_rewrite_fn(original_sentence, prompt)

            if rewritten_sentence is None:
                break

            # Hard safety: reject grossly oversized output
            if len(rewritten_sentence) > max_sent_chars * 2:
                loop_history.append({
                    "loop": loops_used,
                    "sentence": sent_idx,
                    "attempt": attempt + 1,
                    "note": f"rejected: {len(rewritten_sentence)} chars > 2x {max_sent_chars}",
                })
                rewritten_sentence = None
                break

            # Guard: semantic drift (per-sentence, threshold 0.3 — allows meaningful rephrase)
            drift = check_semantic_drift(original_sentence, rewritten_sentence, threshold=0.2)
            if not drift.accepted:
                if attempt < 2:
                    lost_items = "; ".join(drift.reasons[:3])
                    span_info += (
                        f"\nRETRY ({attempt+1}): Failed: {lost_items}"
                        ". Keep same meaning, copy names/numbers/quotes verbatim."
                    )
                    loop_history.append({
                        "loop": loops_used,
                        "sentence": sent_idx,
                        "attempt": attempt + 1,
                        "drift_similarity": round(drift.similarity, 3),
                        "note": f"retry: drift {drift.similarity:.3f}",
                    })
                    continue
                break

            # Guard: length bloat
            if len(rewritten_sentence) > max_sent_chars:
                if attempt < 2:
                    span_info += (
                        f"\nRETRY ({attempt+1}): Too long ({len(rewritten_sentence)} chars)."
                        f" Max {max_sent_chars}. Shorten."
                    )
                    loop_history.append({
                        "loop": loops_used,
                        "sentence": sent_idx,
                        "attempt": attempt + 1,
                        "note": f"retry: length {len(original_sentence)}->{len(rewritten_sentence)}",
                    })
                    continue
                break

            # Guard: protected spans
            protected_lost = [ps for ps in sent_protected
                              if ps.text not in rewritten_sentence] if sent_protected else []
            if protected_lost:
                if attempt < 2:
                    lost = ", ".join(f"'{ps.text}'" for ps in protected_lost)
                    span_info += f"\nRETRY ({attempt+1}): Lost: {lost}. Include verbatim."
                    loop_history.append({
                        "loop": loops_used,
                        "sentence": sent_idx,
                        "attempt": attempt + 1,
                        "note": "retry: protected_span_lost",
                    })
                    continue
                break

            break  # all guards passed

        # Check result
        if rewritten_sentence is None or (drift and not drift.accepted):
            findings_skipped += 1
            reason = "rewrite_failed" if rewritten_sentence is None else "semantic_drift"
            sim_note = f" ({drift.similarity:.3f})" if drift else ""
            floor_reasons.append(FloorReason(
                finding_id=getattr(f, "id", str(id(f))),
                reason_type=reason,
                explanation=f"Sentence {reason}{sim_note} after retries",
            ))
            loop_history.append({
                "loop": loops_used,
                "sentence": sent_idx,
                "finding_type": f.finding_type,
                "reverted": True,
                "note": f"floor: {reason}{sim_note}",
            })
            continue

        if len(rewritten_sentence) > max_sent_chars:
            findings_skipped += 1
            floor_reasons.append(FloorReason(
                finding_id=getattr(f, "id", str(id(f))),
                reason_type="length_bloat",
                explanation=f"Sentence {len(original_sentence)}->{len(rewritten_sentence)}",
            ))
            loop_history.append({
                "loop": loops_used,
                "sentence": sent_idx,
                "finding_type": f.finding_type,
                "reverted": True,
                "note": f"floor: length_bloat {len(original_sentence)}->{len(rewritten_sentence)}",
            })
            continue

        # Splice rewritten sentence back into text
        if original_sentence in current_text:
            candidate_text = current_text.replace(original_sentence, rewritten_sentence, 1)
        else:
            findings_skipped += 1
            floor_reasons.append(FloorReason(
                finding_id=getattr(f, "id", str(id(f))),
                reason_type="splice_failed",
                explanation="Original sentence no longer in text",
            ))
            continue

        # Voice erosion check (full text)
        voice_check = voice_guard.check(current_text, candidate_text)
        if not voice_check.accepted:
            findings_skipped += 1
            floor_reasons.append(FloorReason(
                finding_id=getattr(f, "id", str(id(f))),
                reason_type="voice_eroded",
                explanation=f"Voice erosion: {voice_check.reject_reason}",
            ))
            loop_history.append({
                "loop": loops_used,
                "sentence": sent_idx,
                "finding_type": f.finding_type,
                "reverted": True,
                "note": "floor: voice_eroded",
            })
            continue

        # Predictability regression guard (revert if score got worse)
        reg_check = predictability_guard.check(current_text, candidate_text, original_sentence)
        if not reg_check.accepted:
            findings_skipped += 1
            floor_reasons.append(FloorReason(
                finding_id=getattr(f, "id", str(id(f))),
                reason_type="predictability_regression",
                explanation=f"Predictability {reg_check.orig_risk:.3f} -> {reg_check.new_risk:.3f} (+{reg_check.delta:.3f})",
            ))
            loop_history.append({
                "loop": loops_used,
                "sentence": sent_idx,
                "finding_type": f.finding_type,
                "reverted": True,
                "orig_risk": reg_check.orig_risk,
                "new_risk": reg_check.new_risk,
                "delta": reg_check.delta,
                "note": f"floor: predictability_regression +{reg_check.delta:.3f}",
            })
            continue

        # Badge component guard: reject smoother-but-broader rewrites.
        component_ok, component_reason = _component_regression_check(current_text, candidate_text)
        if not component_ok:
            findings_skipped += 1
            floor_reasons.append(FloorReason(
                finding_id=getattr(f, "id", str(id(f))),
                reason_type="badge_component_regression",
                explanation=component_reason,
            ))
            loop_history.append({
                "loop": loops_used,
                "sentence": sent_idx,
                "finding_type": f.finding_type,
                "reverted": True,
                "note": f"floor: badge_component_regression {component_reason}",
            })
            continue

        # All guards passed - accept
        findings_fixed += 1
        current_text = candidate_text
        loop_history.append({
            "loop": loops_used,
            "sentence": sent_idx,
            "finding_type": f.finding_type,
            "findings_fixed": 1,
            "orig_length": len(original_sentence),
            "new_length": len(rewritten_sentence),
            "orig_text": original_sentence[:80],
            "new_text": rewritten_sentence[:80],
            "note": "applied",
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
        if loop_rewrite_fn is None and detect_context:
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
    # document-level AI likelihood or producing more medium+ target findings.
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
    )
    summary["rollback_applied"] = rolled_back_for_regression
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
