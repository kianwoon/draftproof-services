"""Predictability Rewriter -- flags predictable spans and suggests rewrites.

Pipeline:
  1. Scan text with predictability scanner → get token-level ranks
  2. Group consecutive top-10 tokens into "rewritable spans"
  3. Send spans + context to Claude API for rewrite suggestions
  4. Re-scan rewritten versions to show improvement

Run:  cd poc/rewriter && python demo.py
"""

import os
import re
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Add sibling dirs
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "predictability"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rewrite"))

from scanner import PredictabilityScanner, SentenceResult
from style_analyzer import StyleAnalyzer, StyleProfile
from rewrite.guards import (
    detect_protected_spans, check_semantic_drift,
    RegressionMemory, mask_protected_spans,
)
from rewrite.config import RewriteConfig


# ── Span extraction ─────────────────────────────────────────────────

@dataclass
class RewritableSpan:
    sentence_index: int
    sentence: str
    span_text: str       # the flagged portion
    start_char: int      # char offset in sentence
    end_char: int
    top10_ratio: float   # ratio of top-10 tokens in this span
    token_count: int
    avg_rank: float


def extract_rewritable_spans(
    sentence_results: List[SentenceResult],
    top_k: int = 10,
    min_span_tokens: int = 2,
) -> List[RewritableSpan]:
    """Find contiguous runs of top-K predicted tokens within sentences."""
    spans = []

    for si, s in enumerate(sentence_results):
        if not s.token_results:
            continue

        # Walk tokens, group consecutive top-K tokens
        run_start = None
        run_tokens = []

        for ti, t in enumerate(s.token_results):
            if t.rank <= top_k:
                if run_start is None:
                    run_start = ti
                run_tokens.append(t)
            else:
                if run_tokens and len(run_tokens) >= min_span_tokens:
                    spans.append(_build_span(si, s, run_start, run_tokens))
                run_start = None
                run_tokens = []

        # Flush final run
        if run_tokens and len(run_tokens) >= min_span_tokens:
            spans.append(_build_span(si, s, run_start, run_tokens))

    # Sort by top10 ratio descending (most predictable first)
    spans.sort(key=lambda sp: -sp.top10_ratio)
    return spans


def _build_span(
    si: int, s: SentenceResult, run_start: int, run_tokens: list
) -> RewritableSpan:
    """Convert a run of top-K tokens into a RewritableSpan."""
    # Reconstruct the span text from tokens
    full_text = s.sentence
    span_text = "".join(t.token for t in run_tokens)

    top10_count = sum(1 for t in run_tokens if t.top_10)
    avg_rank = sum(t.rank for t in run_tokens) / len(run_tokens)

    # Approximate char offsets from token position
    char_pos = 0
    for i, t in enumerate(s.token_results):
        if i == run_start:
            start_char = char_pos
        char_pos += len(t.token)
    end_char = char_pos

    return RewritableSpan(
        sentence_index=si,
        sentence=s.sentence,
        span_text=span_text.strip(),
        start_char=start_char,
        end_char=end_char,
        top10_ratio=top10_count / len(run_tokens),
        token_count=len(run_tokens),
        avg_rank=avg_rank,
    )


# ── Claude API rewriter ─────────────────────────────────────────────

REWRITE_SYSTEM_PROMPT = """You are a writing improvement assistant. Your job is to rewrite predictable, formulaic phrases into more specific, original alternatives.

Rules:
- Preserve the factual meaning exactly
- Keep the same register (formal/informal) as the original
- Only rewrite the flagged portion, not the entire sentence
- Provide exactly 3 alternatives, ranked by how specific/original they are
- Each alternative should be a complete replacement for the flagged text
- Do NOT add or remove factual claims
- Do NOT change quotes from named sources

Respond in JSON format:
{
  "alternatives": [
    {"rewrite": "...", "reason": "..."},
    {"rewrite": "...", "reason": "..."},
    {"rewrite": "...", "reason": "..."}
  ]
}"""


def call_claude_rewrite(
    sentence: str,
    span_text: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
) -> List[dict]:
    """Call Claude API to suggest rewrites for a predictable span.

    Returns list of {rewrite, reason} dicts.
    Falls back to rule-based suggestions if API unavailable.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return _fallback_rewrites(span_text)

    import urllib.request
    import urllib.error

    prompt = f"""Sentence: "{sentence}"

Flagged predictable phrase: "{span_text}"

This phrase was flagged because a language model predicted it with very high confidence (top-10 out of 50,000+ candidates). It's likely a formulaic/cliché expression.

Provide 3 more specific, original alternatives that preserve the meaning."""

    body = json.dumps({
        "model": model,
        "max_tokens": 512,
        "system": REWRITE_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            text = result["content"][0]["text"]
            # Strip markdown fences if present
            text = re.sub(r"^```json\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            parsed = json.loads(text)
            return parsed.get("alternatives", [])
    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
        return _fallback_rewrites(span_text)


def _fallback_rewrites(span_text: str) -> List[dict]:
    """Rule-based fallback when API is unavailable."""
    rewrites = []
    rewrites.append({
        "rewrite": f"[rewrite: replace '{span_text}' with more specific phrasing]",
        "reason": "API unavailable — manual rewrite needed to reduce predictability",
    })
    return rewrites


# ── Re-scan ─────────────────────────────────────────────────────────

@dataclass
class RewriteResult:
    original_span: str
    original_metrics: dict
    alternatives: List[dict]
    sentence_index: int
    sentence: str


def rewrite_text(
    text: str,
    scanner: PredictabilityScanner,
    top_k: int = 10,
    min_span_tokens: int = 2,
    max_spans: int = 5,
    api_key: Optional[str] = None,
) -> List[RewriteResult]:
    """Full rewrite pipeline: scan → extract spans → suggest rewrites."""
    pred = scanner.scan_text(text)
    spans = extract_rewritable_spans(pred["sentences"], top_k, min_span_tokens)

    results = []
    for span in spans[:max_spans]:
        alts = call_claude_rewrite(span.sentence, span.span_text, api_key)
        results.append(RewriteResult(
            original_span=span.span_text,
            original_metrics={
                "top10_ratio": round(span.top10_ratio, 2),
                "avg_rank": round(span.avg_rank, 1),
                "token_count": span.token_count,
            },
            alternatives=alts,
            sentence_index=span.sentence_index,
            sentence=span.sentence,
        ))

    return results


# ── Multi-pass rewrite loop ─────────────────────────────────────────

@dataclass
class PassMetrics:
    pass_number: int
    text: str
    risk: float
    top10_ratio: float
    surprisal: float
    sentence_details: List[dict] = field(default_factory=list)


@dataclass
class MultiPassResult:
    original_text: str
    original_metrics: PassMetrics
    passes: List[PassMetrics]
    final_text: str
    final_metrics: PassMetrics
    converged: bool
    convergence_reason: str
    style_profile: Optional[StyleProfile] = None
    style_suggestions: List[str] = field(default_factory=list)
    regression_memory: List[dict] = field(default_factory=list)
    protected_spans_count: int = 0
    drift_score: float = 0.0


REWRITE_PASS_SYSTEM_PROMPT = """You are a writing improvement assistant specializing in reducing AI-detectable patterns in text.

You will receive text that has been flagged for token-level predictability and/or document-level AI structural signatures. Your job is to revise it to reduce those patterns while preserving ALL factual content.

Token-level rules:
- Replace predictable phrases with specific, concrete alternatives
- Use precise terminology instead of generic words
- Add qualifying phrases only where they genuinely improve clarity

Document-level rules:
- Vary paragraph lengths where it improves readability
- Remove explicit transitions where the logical connection is obvious
- Vary sentence lengths for natural rhythm
- Replace inspirational/boilerplate framing with concrete observations
- Preserve the writer's existing register and voice — do not manufacture quirks

Absolute rules (NEVER violate):
- Preserve EVERY name, number, date, and quoted text exactly
- Preserve ALL citation markers (Author, Year) or [1] references
- Do NOT fabricate or remove any facts
- Do NOT add humor, colloquialisms, or imperfections unless the original already has them
- Return ONLY the rewritten text, no commentary"""


def compute_metrics(text: str, scanner: PredictabilityScanner) -> PassMetrics:
    """Scan text and return aggregated metrics."""
    pred = scanner.scan_text(text)
    sents = pred["sentences"]
    if not sents:
        return PassMetrics(0, text, 0, 0, 0)

    avg_risk = sum(s.predictability_risk for s in sents) / len(sents)
    avg_top10 = sum(s.top_10_ratio for s in sents) / len(sents)
    avg_surp = sum(s.avg_surprisal for s in sents) / len(sents)

    sent_details = []
    for i, s in enumerate(sents):
        sent_details.append({
            "index": i,
            "risk": s.predictability_risk,
            "top10_ratio": s.top_10_ratio,
            "surprisal": s.avg_surprisal,
            "label": s.risk_label,
            "sentence": s.sentence,
        })

    return PassMetrics(
        pass_number=0,
        text=text,
        risk=avg_risk,
        top10_ratio=avg_top10,
        surprisal=avg_surp,
        sentence_details=sent_details,
    )


def multi_pass_rewrite(
    text: str,
    scanner: PredictabilityScanner,
    max_passes: int = 3,
    target_top10: float = 0.50,
    min_improvement: float = 0.02,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
    rewrite_fn: Optional[callable] = None,
    config: Optional[RewriteConfig] = None,
) -> MultiPassResult:
    """Iteratively rewrite text to reduce predictability.

    Integrates semantic drift guard, protected span masking, and regression memory.

    Args:
        text: Input text to rewrite.
        scanner: PredictabilityScanner instance.
        max_passes: Maximum rewrite iterations (default 3).
        target_top10: Stop if top-10 ratio drops below this (default 0.50).
        min_improvement: Stop if improvement between passes is below this (default 0.02).
        api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        model: Claude model to use for rewrites.
        rewrite_fn: Optional callable(text, spans_info) -> rewritten_text.
        config: Optional RewriteConfig for advanced controls.

    Returns:
        MultiPassResult with all pass metrics and final text.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    max_semantic_drift = config.max_semantic_drift if config else 0.12

    original_metrics = compute_metrics(text, scanner)
    original_metrics.pass_number = 0

    # Detect protected spans
    protected = detect_protected_spans(text)

    # Run style analysis on original text
    style_analyzer = StyleAnalyzer()
    style_profile = style_analyzer.analyze(text)
    style_suggestions = style_analyzer.get_rewrite_suggestions(style_profile)

    passes = []
    current_text = text
    prev_top10 = original_metrics.top10_ratio
    regression_memory = RegressionMemory()

    for pass_num in range(1, max_passes + 1):
        # Scan current text to get span context for the rewrite
        current_metrics = compute_metrics(current_text, scanner)

        # Check convergence: target already reached AND no individual
        # sentences remain above the threshold
        flagged = [sd for sd in current_metrics.sentence_details
                   if sd.get("top10_ratio", 0) > target_top10]
        if current_metrics.top10_ratio <= target_top10 and not flagged:
            return MultiPassResult(
                original_text=text,
                original_metrics=original_metrics,
                passes=passes,
                final_text=current_text,
                final_metrics=current_metrics,
                converged=True,
                convergence_reason=f"Target top-10 ratio ({target_top10:.0%}) reached — no flagged sentences remain",
                style_profile=style_profile,
                style_suggestions=style_suggestions,
                regression_memory=regression_memory.summary(),
                protected_spans_count=len(protected),
            )

        # Build rewrite prompt with span info + style suggestions and execute rewrite
        span_info = _build_span_context(current_metrics)
        rewritten = _do_rewrite_pass(
            current_text, span_info, api_key, model, rewrite_fn,
            style_suggestions=style_suggestions,
        )
        if rewritten is None:
            return MultiPassResult(
                original_text=text,
                original_metrics=original_metrics,
                passes=passes,
                final_text=current_text,
                final_metrics=current_metrics,
                converged=False,
                convergence_reason="Rewrite API call failed",
                style_profile=style_profile,
                style_suggestions=style_suggestions,
                regression_memory=regression_memory.summary(),
                protected_spans_count=len(protected),
            )

        # Scan the rewritten text to measure improvement
        pass_metrics = compute_metrics(rewritten, scanner)
        pass_metrics.pass_number = pass_num
        passes.append(pass_metrics)

        # Semantic drift guard
        drift = check_semantic_drift(
            current_text, rewritten,
            threshold=1.0 - max_semantic_drift,
        )
        if not drift.accepted:
            regression_memory.record("full_text", "semantic_drift", rewritten[:200], drift.similarity)
            return MultiPassResult(
                original_text=text,
                original_metrics=original_metrics,
                passes=passes,
                final_text=current_text,  # keep pre-rewrite text
                final_metrics=compute_metrics(current_text, scanner),
                converged=True,
                convergence_reason=f"Semantic drift at pass {pass_num}: {'; '.join(drift.reasons[:3])}",
                style_profile=style_profile,
                style_suggestions=style_suggestions,
                regression_memory=regression_memory.summary(),
                protected_spans_count=len(protected),
                drift_score=round(1.0 - drift.similarity, 3),
            )

        # Check: did the rewrite actually improve things?
        improvement = prev_top10 - pass_metrics.top10_ratio
        if improvement <= 0:
            # Rewrite made things WORSE (or no change) — revert
            regression_memory.record("full_text", "predictability_regression", rewritten[:200])
            return MultiPassResult(
                original_text=text,
                original_metrics=original_metrics,
                passes=passes,
                final_text=current_text,  # keep pre-rewrite text
                final_metrics=compute_metrics(current_text, scanner),
                converged=True,
                convergence_reason=f"Rewrite regressed (improvement {improvement:.3f}) at pass {pass_num} — reverted",
                style_profile=style_profile,
                style_suggestions=style_suggestions,
                regression_memory=regression_memory.summary(),
                protected_spans_count=len(protected),
            )

        # Rewrite improved — accept it
        current_text = rewritten

        # Re-analyze style after each pass
        style_profile = style_analyzer.analyze(current_text)
        style_suggestions = style_analyzer.get_rewrite_suggestions(style_profile)

        # Check convergence: target reached
        if pass_metrics.top10_ratio <= target_top10:
            return MultiPassResult(
                original_text=text,
                original_metrics=original_metrics,
                passes=passes,
                final_text=current_text,
                final_metrics=pass_metrics,
                converged=True,
                convergence_reason=f"Target top-10 ratio ({target_top10:.0%}) reached at pass {pass_num}",
                style_profile=style_profile,
                style_suggestions=style_suggestions,
                regression_memory=regression_memory.summary(),
                protected_spans_count=len(protected),
            )

        # Check convergence: diminishing returns (improvement positive but small)
        if improvement < min_improvement:
            return MultiPassResult(
                original_text=text,
                original_metrics=original_metrics,
                passes=passes,
                final_text=current_text,
                final_metrics=pass_metrics,
                converged=True,
                convergence_reason=f"Diminishing returns (improvement {improvement:.3f} < {min_improvement}) at pass {pass_num}",
                style_profile=style_profile,
                style_suggestions=style_suggestions,
                regression_memory=regression_memory.summary(),
                protected_spans_count=len(protected),
            )

        prev_top10 = pass_metrics.top10_ratio

    # Exhausted max passes
    return MultiPassResult(
        original_text=text,
        original_metrics=original_metrics,
        passes=passes,
        final_text=current_text,
        final_metrics=passes[-1] if passes else original_metrics,
        converged=False,
        convergence_reason=f"Max passes ({max_passes}) reached",
        style_profile=style_profile,
        style_suggestions=style_suggestions,
        regression_memory=regression_memory.summary(),
        protected_spans_count=len(protected),
    )


def _build_span_context(metrics: PassMetrics) -> str:
    """Build a description of predictable spans for the rewrite prompt."""
    lines = []
    for sd in metrics.sentence_details:
        if sd["top10_ratio"] > 0.5:
            lines.append(
                f"- S{sd['index']+1} (risk={sd['risk']:.2f}, top-10={sd['top10_ratio']:.0%}): "
                f'"{sd["sentence"][:100]}"'
            )
    if not lines:
        return "No highly predictable spans detected."
    return "Predictable sentences to focus on:\n" + "\n".join(lines)


def _do_rewrite_pass(
    text: str,
    span_info: str,
    api_key: Optional[str],
    model: str,
    rewrite_fn: Optional[callable],
    style_suggestions: Optional[List[str]] = None,
) -> Optional[str]:
    """Execute a single rewrite pass. Returns rewritten text or None on failure."""
    if rewrite_fn:
        return rewrite_fn(text, span_info)

    if not api_key:
        return None

    import urllib.request
    import urllib.error

    style_section = ""
    if style_suggestions:
        style_section = "\n\nStyle/AI-signature issues to fix:\n" + "\n".join(f"- {s}" for s in style_suggestions)

    prompt = f"""{span_info}{style_section}

Current text:
\"\"\"{text}\"\"\"

Rewrite this text to be less predictable AND less AI-like. Address BOTH the flagged sentences above AND the style issues listed. Replace generic phrases with specific, vivid alternatives. Vary sentence structure."""

    body = json.dumps({
        "model": model,
        "max_tokens": 1024,
        "system": REWRITE_PASS_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result["content"][0]["text"].strip()
    except (urllib.error.URLError, json.JSONDecodeError, KeyError):
        return None
