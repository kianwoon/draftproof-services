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
from rewrite.guards import (
    detect_protected_spans, check_semantic_drift, DriftCheck,
    RegressionMemory, mask_protected_spans,
    protected_spans_preserved, affected_region, transactional_apply, TransactionResult,
)
from rewrite.scorer import (
    weighted_finding_score, weighted_rewritable_risk,
    score_candidate, best_candidate, CandidateScore,
    FIXABILITY_WEIGHT,
)
from rewrite.voice import VoiceGuard, VoiceProfile, analyze_voice
from llm.gateway import LLMGateway, LLMConfig


# ── AI-only finding filter ───────────────────────────────────────────

def _is_ai_finding(f: Finding) -> bool:
    """Check if a Finding came from the ai_generation scanner."""
    meta = f.metadata or {}
    return (
        meta.get("scanner") == "ai_generation"
        or meta.get("category") == "ai_generation"
    )


def _filter_ai_findings(detect_results: List[DetectResult]) -> List[DetectResult]:
    """Keep only findings from the ai_generation scanner.

    Returns new DetectResult list with only AI findings. Non-AI scanners
    (predictability, similarity, citation) are dropped entirely.
    """
    filtered = []
    for dr in detect_results:
        # Check DetectResult.scanner (set by parse_detect grouping) AND metadata
        if dr.scanner == "ai_generation":
            ai_findings = dr.findings
        else:
            ai_findings = [f for f in dr.findings if _is_ai_finding(f)]
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


# ── Chip-in: use local `claude` CLI when no API key ──────────────────

REWRITE_CHIPIN_PROMPT = """You are a writing improvement assistant. Rewrite the flagged text to reduce AI-detectable patterns while preserving the author's original meaning.

Signal-specific guidance:
- TOP-K PREDICTABILITY: Replace common word choices with unexpected or domain-specific alternatives. The text uses too many statistically common words.
- LOW SURPRISAL: The word sequences are too predictable. Use less common phrasing and varied sentence openings.
- LOW SPECIFICITY: The text lacks concrete details. Add specific domain terminology and concrete examples already implied by the context.
- LOW BURSTINESS: Sentence lengths are too uniform. Vary rhythm — mix short punchy sentences with longer ones.
- GENERIC PHRASES: Replace formulaic transitions with content-specific connectors.
- REPETITIVE STRUCTURE: Vary sentence openings and syntactic patterns.

Hard rules:
- Preserve factual meaning EXACTLY
- Keep the same register (formal/informal)
- Do NOT change numbers, dates, names, citations, or quoted text
- Every proper noun, number, and quoted phrase in the original MUST appear in your output unchanged
- Do NOT add new facts, names, numbers, or claims not in the original
- Keep the output roughly the SAME LENGTH as the input (within 10%)
- Do NOT add new sentences or expand descriptions — rephrase only
- Follow any REWRITE CONSTRAINTS provided in the context
- Output ONLY the rewritten text, no commentary"""


def _make_chipin_rewrite_fn(detect_context: str) -> callable:
    """Create a rewrite function that calls the local `claude` CLI."""
    def rewrite_fn(text: str, span_info: str) -> Optional[str]:
        user_msg = (
            f"{detect_context}\n\n{span_info}\n\n"
            f'Current text:\n"""{text}"""\n\n'
            "Rewrite this text addressing the issues above. "
            "Output ONLY the rewritten text, no commentary."
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
                return result.stdout.strip()
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
    config = LLMConfig(api_key=api_key, model=model, base_url=base_url or "https://openrouter.ai/api/v1")
    gateway = LLMGateway(config)

    def rewrite_fn(text: str, span_info: str) -> Optional[str]:
        user_msg = (
            f"{detect_context}\n\n{span_info}\n\n"
            f'Current text:\n"""{text}"""\n\n'
            "Rewrite this text addressing the issues above. "
            "Output ONLY the rewritten text, no commentary."
        )
        try:
            resp = gateway.chat(user_msg, system=REWRITE_CHIPIN_PROMPT)
            if resp.is_empty:
                return None
            return resp.content.strip()
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
    # AI-only filter: strip non-AI findings before planning
    if ai_only:
        detect_results = _filter_ai_findings(detect_results)

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

    # ── Step 3: Per-finding rewrite loop (no mid-loop re-detect) ────
    current_text = content
    current_weighted_risk = weighted_finding_score(
        [f for dr in detect_results for f in dr.findings]
    )
    floor_reasons = []
    findings_fixed = 0
    findings_skipped = 0
    loop_history = [{
        "loop": 0,
        "weighted_risk": current_weighted_risk,
        "raw_findings": _count_findings(detect_results),
        "text": content[:100],
        "note": "original",
    }]
    loops_used = 0

    # ── Batch by paragraph: one Claude call per paragraph ───────────
    # Group all auto-fixable actions by paragraph, then rewrite each
    # paragraph once with ALL its findings listed in the prompt.
    # This reduces N Claude calls to ~N_paragraphs calls.
    sentences, para_map = _build_sentence_index(current_text)
    para_groups = _group_actions_by_paragraph(plan.auto_fixable, sentences, para_map)
    paragraphs = _split_paragraphs(current_text)

    for para_idx in range(len(paragraphs)):
        if para_idx not in para_groups:
            continue
        para_actions = para_groups[para_idx]
        loops_used += 1

        # Get current paragraph text directly
        current_paragraphs = _split_paragraphs(current_text)
        if para_idx >= len(current_paragraphs):
            for action in para_actions:
                findings_skipped += 1
                floor_reasons.append(FloorReason(
                    finding_id=str(id(action.finding)),
                    reason_type="paragraph_vanished",
                    explanation="Paragraph removed by prior edits",
                ))
            continue

        region_text = current_paragraphs[para_idx]
        if not region_text:
            for action in para_actions:
                findings_skipped += 1
                floor_reasons.append(FloorReason(
                    finding_id=str(id(action.finding)),
                    reason_type="paragraph_vanished",
                    explanation="Paragraph indices out of range after prior edits",
                ))
            continue

        # Check all evidence still present
        missing = [a for a in para_actions if current_text.find(a.finding.evidence) == -1]
        for action in missing:
            findings_skipped += 1
            floor_reasons.append(FloorReason(
                finding_id=str(id(action.finding)),
                reason_type="evidence_not_found",
                explanation=f"Evidence gone: \'{action.finding.evidence[:50]}...\'",
            ))
            loop_history.append({
                "loop": loops_used,
                "finding_type": action.finding.finding_type,
                "reverted": True,
                "note": "skipped: evidence not found",
            })
        remaining = [a for a in para_actions if a not in missing]
        if not remaining:
            continue

        # Protected spans within the region
        all_protected = detect_protected_spans(current_text)
        region_start = current_text.find(region_text)
        region_protected = [
            ps for ps in all_protected
            if region_start >= 0
            and ps.start_char >= region_start
            and ps.end_char <= region_start + len(region_text)
        ]

        # Build batch prompt with ALL findings for this paragraph
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

        # Explicit preservation list from entity extraction
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
        if region_protected:
            protected_items = ", ".join(f"'{ps.text}'" for ps in region_protected)
            preserve_parts.append(f"Protected spans: {protected_items}")
        if preserve_parts:
            span_info += "\n\nMANDATORY: These MUST appear unchanged in your output:\n" + "\n".join(f"- {p}" for p in preserve_parts)

        # Single Claude call for the whole paragraph (with retry on guard failure)
        rewritten_region = None
        drift = None
        protected_lost = []
        for attempt in range(3):  # up to 3 attempts
            if loop_rewrite_fn:
                rewritten_region = loop_rewrite_fn(region_text, span_info)

            if rewritten_region is None:
                break

            # Guard: semantic drift check
            drift = check_semantic_drift(region_text, rewritten_region, threshold=0.1)
            if not drift.accepted:
                if attempt < 2:
                    lost_items = "\n".join(f"- {r}" for r in drift.reasons[:5])
                    span_info += (
                        f"\n\nYOUR PREVIOUS ATTEMPT FAILED these checks:\n{lost_items}"
                        "\n\nTry again. Keep the same meaning but fix ALL the issues above."
                        " Copy every name, number, and quote VERBATIM from the original."
                    )
                    loop_history.append({
                        "loop": loops_used,
                        "paragraph": para_idx,
                        "attempt": attempt + 1,
                        "drift_similarity": round(drift.similarity, 3),
                        "note": f"retry: drift {drift.similarity:.3f}",
                    })
                    continue
                break  # final attempt failed

            # Guard: protected spans
            protected_lost = [ps for ps in region_protected
                              if ps.text not in rewritten_region] if region_protected else []
            if protected_lost:
                if attempt < 2:
                    lost = ", ".join(f"'{ps.text}'" for ps in protected_lost)
                    span_info += (
                        f"\n\nYOUR PREVIOUS ATTEMPT LOST these required items: {lost}"
                        "\n\nTry again. You MUST include these exact strings in your output."
                    )
                    loop_history.append({
                        "loop": loops_used,
                        "paragraph": para_idx,
                        "attempt": attempt + 1,
                        "note": f"retry: protected_span_lost {lost[:60]}",
                    })
                    continue
                break  # final attempt failed

            # Guard: length bloat
            orig_len = len(region_text)
            new_len = len(rewritten_region)
            if new_len > orig_len * 1.15:
                if attempt < 2:
                    span_info += (
                        f"\n\nYOUR PREVIOUS ATTEMPT was too long: {new_len} chars vs original {orig_len}."
                        "\n\nTry again. Keep it MUCH shorter — same ideas, fewer words."
                    )
                    loop_history.append({
                        "loop": loops_used,
                        "paragraph": para_idx,
                        "attempt": attempt + 1,
                        "note": f"retry: length_bloat {orig_len}→{new_len}",
                    })
                    continue
                break  # final attempt failed

            break  # all guards passed

        # Check if rewrite succeeded
        if rewritten_region is None:
            for action in remaining:
                findings_skipped += 1
                floor_reasons.append(FloorReason(
                    finding_id=str(id(action.finding)),
                    reason_type="rewrite_failed",
                    explanation="Batch rewrite returned None",
                ))
            loop_history.append({
                "loop": loops_used,
                "paragraph": para_idx,
                "findings_count": len(remaining),
                "reverted": True,
                "note": "batch rewrite failed",
            })
            continue

        if drift and not drift.accepted:
            for action in remaining:
                findings_skipped += 1
                regression_memory.record(
                    action.finding.evidence[:80], "semantic_drift",
                    rewritten_region[:100], drift.similarity,
                )
                floor_reasons.append(FloorReason(
                    finding_id=str(id(action.finding)),
                    reason_type="semantic_drift",
                    explanation=f"Batch drift after retries: {drift.similarity:.3f}",
                ))
            loop_history.append({
                "loop": loops_used,
                "paragraph": para_idx,
                "findings_count": len(remaining),
                "drift_similarity": round(drift.similarity, 3),
                "reverted": True,
                "note": f"floor: semantic_drift {drift.similarity:.3f} (batch)",
            })
            continue

        if protected_lost:
            for action in remaining:
                findings_skipped += 1
                floor_reasons.append(FloorReason(
                    finding_id=str(id(action.finding)),
                    reason_type="protected_span_lost",
                    explanation=f"Lost after retries: {', '.join(ps.text for ps in protected_lost)}",
                ))
            loop_history.append({
                "loop": loops_used,
                "paragraph": para_idx,
                "findings_count": len(remaining),
                "reverted": True,
                "note": "floor: protected_span_lost (batch, all retries)",
            })
            continue

        orig_len = len(region_text)
        new_len = len(rewritten_region)
        if new_len > orig_len * 1.15:
            for action in remaining:
                findings_skipped += 1
                floor_reasons.append(FloorReason(
                    finding_id=str(id(action.finding)),
                    reason_type="length_bloat",
                    explanation=f"Rewrite expanded {orig_len}→{new_len} after retries",
                ))
            loop_history.append({
                "loop": loops_used,
                "paragraph": para_idx,
                "findings_count": len(remaining),
                "reverted": True,
                "note": f"floor: length_bloat {orig_len}→{new_len}",
            })
            continue

        # Splice rewritten paragraph back into text
        current_paragraphs = _split_paragraphs(current_text)
        current_paragraphs[para_idx] = rewritten_region
        candidate_text = "\n\n".join(current_paragraphs)

        # Guard 3: Voice erosion (full text)
        voice_check = voice_guard.check(current_text, candidate_text)
        if not voice_check.accepted:
            for action in remaining:
                findings_skipped += 1
                floor_reasons.append(FloorReason(
                    finding_id=str(id(action.finding)),
                    reason_type="voice_eroded",
                    explanation=f"Voice erosion: {voice_check.reject_reason}",
                ))
            loop_history.append({
                "loop": loops_used,
                "paragraph": para_idx,
                "findings_count": len(remaining),
                "reverted": True,
                "note": "floor: voice_eroded (batch)",
            })
            continue

        # All guards passed — accept batch
        n_fixed = len(remaining)
        findings_fixed += n_fixed
        current_text = candidate_text

        loop_history.append({
            "loop": loops_used,
            "paragraph": para_idx,
            "findings_count": n_fixed,
            "drift_similarity": round(drift.similarity, 3),
            "note": f"batch fixed {n_fixed} findings",
            "finding_types": [a.finding.finding_type for a in remaining],
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
        if loop_rewrite_fn is None and detect_context:
            loop_rewrite_fn = _make_chipin_rewrite_fn(re_detect_context)
        else:
            # Update detect context for chip-in
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

    final_metrics = compute_metrics(current_text, scanner)
    original_metrics = compute_metrics(content, scanner)

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
