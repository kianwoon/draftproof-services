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


def _metrics_from_detect(detect_report, text: str):
    """Extract predictability metrics from an already-run DetectionReport.

    Avoids running compute_metrics() which would trigger another
    full predictability scan (~28s). Falls back to compute_metrics
    only if the predictability scanner didn't produce results.
    """
    for sr in detect_report.scanner_results:
        if sr.scanner == "predictability" and sr.raw:
            raw = sr.raw
            if hasattr(raw, "sentence_details"):
                from rewriter import MetricsResult
                return MetricsResult(
                    risk=raw.overall_risk if hasattr(raw, "overall_risk") else 0.5,
                    top10_ratio=raw.top10_ratio if hasattr(raw, "top10_ratio") else 0.0,
                    sentence_details=raw.sentence_details,
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


# ── Chip-in: use local `claude` CLI when no API key ──────────────────

REWRITE_CHIPIN_PROMPT = """You are a text editor. Your ONLY job is to make the flagged sentence sound LESS like AI-generated text.

You do this by applying ONE specific mechanical transformation. Do NOT try to "improve" the writing generally.

TRANSFORMATION RULES (apply the one matching the finding type):

For predictability / common_words findings:
- Find the 2-3 most generic words in the sentence and replace them with more specific, unusual synonyms that keep the same meaning. Examples: "important" → "pivotal", "shows" → "reveals", "helps" → "enables", "different" → "divergent". NEVER use the words: crucial, vital, essential, significant, notable, furthermore, moreover, additionally.

For formulaic_sentence / generic_phrase findings:
- Restructure the sentence to break its formulaic pattern. Move the subject to a different position. Merge or split clauses. Start the sentence differently than it currently starts.

For style_shift / repetitive_structure findings:
- Change the sentence opening to differ from the previous sentence's opening. If the previous sentence starts with "The X...", start this one with a participle, adverb, or dependent clause instead.

For burstiness findings:
- If this sentence is similar in length to neighbors, make it noticeably shorter (cut filler words) or merge it with context.

For ai_generation findings:
- Replace the flagged AI-typical phrase with a natural rephrasing. A human would say it differently.

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
    # Filter findings before planning
    if ai_only:
        detect_results = _filter_ai_findings(detect_results)
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

        # Build focused per-sentence prompt
        # NOTE: The rewrite_fn already wraps text in """ quotes and adds char limits.
        # We only send finding details + context here — NOT the text itself.
        span_info_parts = [
            "Finding: [" + f.finding_type + "/" + f.risk_level + "] " + f.detail,
            "Fix strategy: " + f.suggested_action_type,
            "Flagged phrase: '" + f.evidence + "'",
        ]
        if ctx_before:
            span_info_parts.append("Previous sentence context: '" + ctx_before + "'")
        if ctx_after:
            span_info_parts.append("Next sentence context: '" + ctx_after + "'")
        if sent_protected:
            protected_items = ", ".join(ps.text for ps in sent_protected)
            span_info_parts.append("Must preserve exactly: " + protected_items)
        span_info = "\n".join(span_info_parts)

        # Rewrite with retry loop
        rewritten_sentence = None
        drift = None
        max_sent_chars = int(len(original_sentence) * 1.20)  # 20% tolerance per sentence
        for attempt in range(3):
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

    for detect_loop in range(config.max_detect_loops):
        # Time budget: stop outer loop if we've already spent >300s rewriting
        if time.time() - outer_loop_start > 300:
            detect_loop_history.append({
                "detect_loop": detect_loop,
                "note": "skipped: time budget exceeded",
            })
            break

        # Re-detect current text
        re_detect_runner = DetectionRunner()
        re_detect_report = re_detect_runner.run_all(current_text)
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
                findings_lines.append(
                    f"- [{f.finding_type}/{f.risk_level}] \"{f.evidence}\"\n"
                    f"  Detail: {f.detail}\n"
                    f"  Fix: {f.suggested_action_type}"
                )
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
    final_detect_runner = DetectionRunner()
    final_detect_report = final_detect_runner.run_all(current_text)
    final_detect_results = final_detect_report.scanner_results

    # True fix count = original findings minus remaining findings
    final_finding_count = _count_findings(final_detect_results)
    original_finding_count = _count_findings(detect_results)
    net_findings_fixed = original_finding_count - final_finding_count

    # Reuse predictability results from final_detect_report instead of
    # running compute_metrics again (saves ~28s per call).
    final_metrics = _metrics_from_detect(final_detect_report, current_text)
    original_metrics = _metrics_from_detect(final_detect_report, content)

    result = MultiPassResult(
        original_text=content,
        original_metrics=original_metrics,
        passes=[final_metrics],
        final_text=current_text,
        final_metrics=final_metrics,
        converged=(net_findings_fixed > 0 and findings_skipped == 0),
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
    if findings_fixed > 0 and findings_skipped == 0:
        outcome = RewriteOutcome.IMPROVED
    elif findings_fixed > 0 and findings_skipped > 0:
        outcome = RewriteOutcome.PARTIALLY_IMPROVED
    elif findings_skipped > 0 and findings_fixed == 0:
        outcome = RewriteOutcome.FLOOR_REACHED
    elif plan.manual_required and not plan.auto_fixable:
        outcome = RewriteOutcome.MANUAL_REQUIRED
    else:
        outcome = RewriteOutcome.FLOOR_REACHED

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
