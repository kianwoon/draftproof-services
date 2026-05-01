"""Criterion: Sudden draft evolution jump.

Strong signal: detects abrupt shifts in writing quality, complexity, or style
within a single document that suggest section-level replacement or insertion.

Two modes:
1. Multi-draft mode: compares provided draft versions for evolution jumps.
2. Single-document mode: analyzes internal section-to-section consistency.

Without draft history, this degrades to section-level style consistency analysis.
"""

import re
import math
from typing import Dict, List, Optional

from ._types import CriterionScore


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into non-empty paragraphs."""
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return paras if len(paras) >= 2 else []


def _paragraph_complexity(para: str) -> Dict[str, float]:
    """Measure writing complexity metrics for a paragraph."""
    sentences = re.split(r'(?<=[.!?])\s+', para.strip())
    sentences = [s for s in sentences if s]
    if not sentences:
        return {"avg_sent_len": 0, "vocab_diversity": 0, "hedge_density": 0}

    word_lengths = []
    all_words = []
    for s in sentences:
        words = [w for w in re.findall(r'\b\w+\b', s.lower()) if len(w) > 1]
        all_words.extend(words)
        word_lengths.extend([len(w) for w in words])

    avg_sent_len = sum(word_lengths) / max(len(word_lengths), 1) if word_lengths else 0

    # Vocabulary diversity (type-token ratio)
    unique = len(set(all_words))
    total = len(all_words)
    vocab_diversity = unique / total if total > 3 else 0

    # Hedging/formal marker density
    hedging_patterns = [
        r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
        r'\bconsequently\b', r'\bnotably\b', r'\bsignificantly\b',
        r'\bdemonstrates\b', r'\bunderscores\b', r'\bhighlights\b',
        r'\bcrucial\b', r'\bvital\b', r'\bessential\b',
        r'\bit is important\b', r'\bit is worth noting\b',
        r'\bplays a .+ role\b',
    ]
    hedge_count = sum(1 for p in hedging_patterns if re.search(p, para.lower()))
    hedge_density = min(1.0, hedge_count / max(len(sentences), 1))

    return {
        "avg_sent_len": round(avg_sent_len, 2),
        "vocab_diversity": round(vocab_diversity, 3),
        "hedge_density": round(hedge_density, 3),
    }


def _compute_section_jump(paragraphs: List[str]) -> tuple:
    """Find the maximum complexity jump between adjacent paragraphs.

    Returns (max_jump, jump_position, details).
    """
    if len(paragraphs) < 2:
        return 0.0, 0, {}

    metrics = [_paragraph_complexity(p) for p in paragraphs]

    max_jump = 0.0
    max_pos = 0
    jump_details = {}

    for i in range(1, len(metrics)):
        prev = metrics[i - 1]
        curr = metrics[i]

        # Sentence length delta
        sent_delta = abs(curr["avg_sent_len"] - prev["avg_sent_len"])
        # Vocab diversity delta
        vocab_delta = abs(curr["vocab_diversity"] - prev["vocab_diversity"])
        # Hedge density delta (sudden formality increase)
        hedge_delta = max(0, curr["hedge_density"] - prev["hedge_density"])

        # Composite jump: weighted combination
        jump = 0.30 * min(sent_delta / 2.0, 1.0) + \
               0.30 * min(vocab_delta * 3.0, 1.0) + \
               0.40 * min(hedge_delta * 2.5, 1.0)

        if jump > max_jump:
            max_jump = jump
            max_pos = i
            jump_details = {
                "prev_complexity": prev,
                "curr_complexity": curr,
                "sent_length_delta": round(sent_delta, 2),
                "vocab_diversity_delta": round(vocab_delta, 3),
                "hedge_density_delta": round(hedge_delta, 3),
            }

    return max_jump, max_pos, jump_details


def _compare_drafts(draft_versions: List[str]) -> tuple:
    """Compare multiple draft versions for evolution jumps.

    Analyzes the progression between versions:
    - Natural evolution: incremental changes across versions
    - Sudden jump: one version is drastically different from the previous

    Returns (jump_score, details).
    """
    if len(draft_versions) < 2:
        return 0.0, {"note": "need at least 2 drafts"}

    version_metrics = []
    for v in draft_versions:
        paras = _split_into_paragraphs(v)
        if paras:
            all_metrics = [_paragraph_complexity(p) for p in paras]
            avg_sent = sum(m["avg_sent_len"] for m in all_metrics) / len(all_metrics)
            avg_vocab = sum(m["vocab_diversity"] for m in all_metrics) / len(all_metrics)
            avg_hedge = sum(m["hedge_density"] for m in all_metrics) / len(all_metrics)
        else:
            words = v.split()
            avg_sent = sum(len(w) for w in words) / max(len(words), 1)
            avg_vocab = 0
            avg_hedge = 0
        version_metrics.append({
            "avg_sent_len": avg_sent,
            "vocab_diversity": avg_vocab,
            "hedge_density": avg_hedge,
            "word_count": len(v.split()),
        })

    max_version_jump = 0.0
    max_jump_version = 0
    jump_details = {}

    for i in range(1, len(version_metrics)):
        prev = version_metrics[i - 1]
        curr = version_metrics[i]

        # Word count change rate
        wc_prev = max(prev["word_count"], 1)
        wc_curr = max(curr["word_count"], 1)
        word_change_rate = abs(wc_curr - wc_prev) / wc_prev

        # Complexity shift
        sent_delta = abs(curr["avg_sent_len"] - prev["avg_sent_len"])
        vocab_delta = abs(curr["vocab_diversity"] - prev["vocab_diversity"])
        hedge_delta = max(0, curr["hedge_density"] - prev["hedge_density"])

        # A natural draft evolves gradually; a sudden rewrite has big jumps
        jump = 0.25 * min(word_change_rate, 1.0) + \
               0.25 * min(sent_delta / 2.0, 1.0) + \
               0.25 * min(vocab_delta * 3.0, 1.0) + \
               0.25 * min(hedge_delta * 2.5, 1.0)

        if jump > max_version_jump:
            max_version_jump = jump
            max_jump_version = i
            jump_details = {
                "from_version": i - 1,
                "to_version": i,
                "word_change_rate": round(word_change_rate, 3),
                "complexity_shift": {
                    "sent_len": round(sent_delta, 2),
                    "vocab": round(vocab_delta, 3),
                    "hedge": round(hedge_delta, 3),
                },
            }

    return max_version_jump, jump_details


def score(
    content: str,
    *,
    draft_versions: Optional[List[str]] = None,
    **kwargs,
) -> CriterionScore:
    """Score draft evolution jump criterion.

    Args:
        content: The document text to analyze.
        draft_versions: Optional list of prior draft versions (oldest first).
            When provided, compares version-to-version evolution.
            When absent, analyzes internal section consistency.

    Returns:
        CriterionScore with value 0-1 (higher = more suspicious jump detected).
    """
    # Mode 1: Multi-draft comparison (Strong signal)
    if draft_versions and len(draft_versions) >= 2:
        # Include current content as the latest version
        all_versions = list(draft_versions) + [content] \
            if content not in draft_versions else draft_versions

        jump_score, details = _compare_drafts(all_versions)
        mode = "multi_draft"

        if jump_score >= 0.55:
            label = "high"
        elif jump_score >= 0.30:
            label = "medium"
        else:
            label = "low"

        flagged = []
        if jump_score >= 0.30 and "from_version" in details:
            to_ver = details["to_version"]
            if to_ver < len(all_versions):
                flagged = [all_versions[to_ver][:200]]

        return CriterionScore(
            name="draft_evolution",
            value=round(jump_score, 4),
            label=label,
            details={
                "mode": mode,
                "version_count": len(all_versions),
                **details,
            },
            flagged_excerpts=flagged,
        )

    # Mode 2: Single-document section consistency (degraded — Moderate signal)
    paragraphs = _split_into_paragraphs(content)
    if len(paragraphs) < 3:
        return CriterionScore(
            name="draft_evolution",
            value=0.0,
            label="low",
            details={"mode": "single_document", "note": "insufficient paragraphs"},
        )

    max_jump, jump_pos, jump_details = _compute_section_jump(paragraphs)

    # Discount in single-document mode — no draft history means lower confidence
    discounted = max_jump * 0.50

    if discounted >= 0.40:
        label = "high"
    elif discounted >= 0.25:
        label = "medium"
    else:
        label = "low"

    flagged = []
    if discounted >= 0.25 and jump_pos < len(paragraphs):
        flagged = [paragraphs[jump_pos][:200]]

    return CriterionScore(
        name="draft_evolution",
        value=round(discounted, 4),
        label=label,
        details={
            "mode": "single_document",
            "raw_jump": round(max_jump, 4),
            "discounted": True,
            "discount_factor": 0.50,
            "jump_position": jump_pos,
            "paragraph_count": len(paragraphs),
            **jump_details,
        },
        flagged_excerpts=flagged,
    )
