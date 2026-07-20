"""Authorship Concern Score — weighted multi-signal formula.

Produces a 0-1 authorship concern score alongside the existing tier label.
Phase 1: uses existing scanner outputs (predictability, genericity, specificity,
citation_integrity). Missing signals (source_grounding, draft_evolution,
structural_reuse) are None and excluded from weighting.

Key design:
- Dynamic weight normalization: only available signals contribute
- Weak-signal cap: predictability/genericity/specificity alone → max 0.30
- Confidence scoring: missing inputs lower confidence, not risk
- Rule-based concern tier (separate from existing tier derivation)
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Literal


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ── Signal container ──────────────────────────────────────────────────

@dataclass
class SignalScores:
    predictability: Optional[float] = None
    genericity: Optional[float] = None
    specificity: Optional[float] = None
    source_grounding: Optional[float] = None
    citation_integrity: Optional[float] = None
    draft_evolution: Optional[float] = None
    structural_reuse: Optional[float] = None
    burstiness: Optional[float] = None
    polish_ungrounded: Optional[float] = None
    surprisal_risk: Optional[float] = None
    topk_pattern_risk: Optional[float] = None


DEFAULT_WEIGHTS: Dict[str, float] = {
    "predictability": 0.08,
    "genericity": 0.08,
    "specificity": 0.10,
    "source_grounding": 0.10,
    "citation_integrity": 0.10,
    "draft_evolution": 0.15,
    "structural_reuse": 0.08,
    "burstiness": 0.05,
    "polish_ungrounded": 0.05,
    "surprisal_risk": 0.11,
    "topk_pattern_risk": 0.10,
}

STRONG_SIGNALS = {"source_grounding", "citation_integrity", "draft_evolution", "structural_reuse", "polish_ungrounded"}
WEAK_SIGNALS = {"predictability", "genericity", "specificity", "burstiness", "surprisal_risk", "topk_pattern_risk"}


# ── Core formula ──────────────────────────────────────────────────────

def calculate_weighted_score(
    signals: SignalScores,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> float:
    """Weighted average over available signals. Missing signals redistribute weight."""
    available = {k: v for k, v in asdict(signals).items() if v is not None}
    if not available:
        return 0.0

    total_weight = sum(weights[k] for k in available)
    if total_weight == 0:
        return 0.0

    score = sum(available[k] * weights[k] for k in available) / total_weight
    return clamp(score)


def is_weak_signal_only(signals: SignalScores) -> bool:
    """True if only weak signals (predictability/genericity/specificity) are present
    and no strong signal exceeds the threshold."""
    d = asdict(signals)

    has_weak = any(d[k] is not None and d[k] > 0 for k in WEAK_SIGNALS)
    has_strong = any(d[k] is not None and d[k] >= 0.25 for k in STRONG_SIGNALS)

    return has_weak and not has_strong


# ── Confidence ────────────────────────────────────────────────────────

def calculate_confidence(
    word_count: int,
    has_sources: bool,
    has_bibliography: bool,
    has_draft_history: bool,
) -> tuple:
    """Returns (confidence_float, confidence_label).

    Missing inputs lower confidence but NOT risk.
    """
    conf = 1.0

    if word_count < 150:
        conf -= 0.40
    elif word_count < 500:
        conf -= 0.20

    if not has_sources:
        conf -= 0.20
    if not has_bibliography:
        conf -= 0.15
    if not has_draft_history:
        conf -= 0.20

    conf = clamp(conf, 0.0, 1.0)

    if conf >= 0.75:
        label = "high"
    elif conf >= 0.45:
        label = "medium"
    else:
        label = "low"

    return round(conf, 4), label


# ── Concern tier ──────────────────────────────────────────────────────

def derive_concern_tier(score: float, signals: SignalScores, confidence_label: str) -> str:
    """Rule-based tier from concern score + signal context.

    Thresholds:
        < 0.15 → clear
        < 0.25 → light_review
        < 0.40 → review_recommended
        < 0.65 → needs_attention
        ≥ 0.65 → urgent_review
    Plus override rules for specific signal combos.
    """
    d = asdict(signals)

    # Override: low confidence caps at review_recommended
    if confidence_label == "low" and score < 0.65:
        if score >= 0.40:
            return "review_recommended"
        if score >= 0.25:
            return "light_review"
        return "clear"

    # Override: citation + source grounding both high → urgent
    cite = d.get("citation_integrity")
    sg = d.get("source_grounding")
    if cite is not None and cite >= 0.70 and sg is not None and sg >= 0.70:
        return "urgent_review"

    # Override: draft evolution + structural reuse both high → needs_attention
    de = d.get("draft_evolution")
    sr = d.get("structural_reuse")
    if de is not None and de >= 0.75 and sr is not None and sr >= 0.75:
        return "needs_attention"

    # Threshold-based
    if score >= 0.65:
        return "urgent_review"
    if score >= 0.40:
        return "needs_attention"
    if score >= 0.25:
        return "review_recommended"
    if score >= 0.15:
        return "light_review"
    return "clear"


# ── AI Risk Badge (v5) ─────────────────────────────────────────────────

import re as _re
import statistics as _statistics

Tier = Literal["GREEN", "AMBER", "ORANGE", "RED"]
TurnitinBand = Literal[
    "likely_0",
    "likely_0_or_star",
    "likely_star",
    "visible_ai_possible",
    "high_ai_concern",
]


def _domain_strength(index: float) -> float:
    """Topic relevance only. Not human proof."""
    if index >= 2.5:
        return 1.0
    if index >= 1.5:
        return 0.6
    return 0.2


def _confidence_multiplier(confidence: str) -> float:
    if confidence == "high":
        return 1.20
    if confidence == "medium":
        return 1.10
    return 1.00


def _classify_score(score: float) -> tuple:
    if score < 0.030:
        return "GREEN", "likely_0"
    if score < 0.060:
        return "AMBER", "likely_0_or_star"
    if score < 0.100:
        return "AMBER", "likely_star"
    if score < 0.150:
        return "ORANGE", "visible_ai_possible"
    return "RED", "high_ai_concern"


def _split_sentences(text: str) -> list:
    text = _re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return _re.split(r"(?<=[.!?])\s+", text)


def estimate_citation_risk(
    in_text_citations: int,
    bibliography_count: int,
    uncited_claims: int,
    total_claims: int,
) -> float:
    """Higher = weaker citation/source integrity."""
    if bibliography_count <= 0 and total_claims <= 0:
        return 0.50
    cbr = in_text_citations / bibliography_count if bibliography_count > 0 else 0.0
    csr = 1 - (uncited_claims / total_claims) if total_claims > 0 else 1.0
    cbr, csr = clamp(cbr), clamp(csr)
    return clamp(1 - (0.45 * cbr + 0.55 * csr))


# ── Main entry point ──────────────────────────────────────────────────

def calculate_authorship_concern(
    signals: SignalScores,
    word_count: int,
    has_sources: bool = False,
    has_bibliography: bool = False,
    has_draft_history: bool = False,
) -> Dict[str, Any]:
    """Compute full authorship concern assessment.

    Returns dict with:
        score, concern_tier, confidence, confidence_label,
        weak_signal_only, signals (raw), available_signal_count
    """
    score = calculate_weighted_score(signals)

    weak_only = is_weak_signal_only(signals)
    if weak_only:
        score = min(score, 0.30)

    conf_val, conf_label = calculate_confidence(
        word_count=word_count,
        has_sources=has_sources,
        has_bibliography=has_bibliography,
        has_draft_history=has_draft_history,
    )

    concern_tier = derive_concern_tier(score, signals, conf_label)

    available_count = sum(1 for v in asdict(signals).values() if v is not None)

    return {
        "score": round(score, 4),
        "concern_tier": concern_tier,
        "confidence": conf_val,
        "confidence_label": conf_label,
        "weak_signal_only": weak_only,
        "signals": {k: round(v, 4) if v is not None else None for k, v in asdict(signals).items()},
        "available_signal_count": available_count,
        "total_signal_count": 11,
    }


# ── Signal extraction from existing scanner data ──────────────────────

def extract_signals(
    predictability_summary=None,
    similarity_summary=None,
    citation_summary=None,
    findings: Optional[List[Any]] = None,
    criterion_scores: Optional[Dict[str, Any]] = None,
) -> SignalScores:
    """Extract normalized signals from scanner outputs + criterion scores.

    Args:
        predictability_summary: PredictabilitySummary dataclass
        similarity_summary: SimilaritySummary dataclass
        citation_summary: CitationSummary dataclass
        findings: List of Finding dataclasses from ReportBuilder
        criterion_scores: Dict[str, CriterionScore] from AIGenerationSignalDetector
    """
    pred = _extract_predictability(predictability_summary)
    cite = _extract_citation_integrity(citation_summary)

    # ── Genericity: prefer criterion score, fall back to formula ────────
    gen = None
    if criterion_scores and "generic_phrase_density" in criterion_scores:
        cs = criterion_scores["generic_phrase_density"]
        gen = cs.value if hasattr(cs, "value") else cs.get("value")
    if gen is None:
        gen = _extract_genericity(predictability_summary, findings)

    # ── Specificity: prefer criterion score, fall back to findings ──────
    spec = None
    if criterion_scores and "low_specificity" in criterion_scores:
        cs = criterion_scores["low_specificity"]
        spec = cs.value if hasattr(cs, "value") else cs.get("value")
    if spec is None:
        spec = _extract_specificity(findings)

    # ── Source grounding: from criterion score ──────────────────────────
    sg = None
    if criterion_scores and "source_grounding" in criterion_scores:
        cs = criterion_scores["source_grounding"]
        sg = cs.value if hasattr(cs, "value") else cs.get("value")

    # ── Draft evolution: from criterion score ──────────────────────────
    de = None
    if criterion_scores and "draft_evolution" in criterion_scores:
        cs = criterion_scores["draft_evolution"]
        de = cs.value if hasattr(cs, "value") else cs.get("value")

    # ── Structural reuse: from criterion score ──────────────────────────
    sr = None
    if criterion_scores and "structural_reuse" in criterion_scores:
        cs = criterion_scores["structural_reuse"]
        sr = cs.value if hasattr(cs, "value") else cs.get("value")

    # ── Burstiness: from criterion score ──────────────────────────
    burst = None
    if criterion_scores and "low_burstiness" in criterion_scores:
        cs = criterion_scores["low_burstiness"]
        burst = cs.value if hasattr(cs, "value") else cs.get("value")

    # ── Polished but ungrounded: from criterion score ──────────────────────────
    pug = None
    if criterion_scores and "citation_grounding_gap" in criterion_scores:
        cs = criterion_scores["citation_grounding_gap"]
        pug = cs.value if hasattr(cs, "value") else cs.get("value")

    # ── Surprisal risk: from criterion score ──────────────────────────
    surp = None
    if criterion_scores and "low_surprisal" in criterion_scores:
        cs = criterion_scores["low_surprisal"]
        surp = cs.value if hasattr(cs, "value") else cs.get("value")

    # ── Top-k pattern risk: from criterion score ──────────────────────────
    topk = None
    if criterion_scores and "topk_predictability" in criterion_scores:
        cs = criterion_scores["topk_predictability"]
        topk = cs.value if hasattr(cs, "value") else cs.get("value")

    return SignalScores(
        predictability=pred,
        genericity=gen,
        specificity=spec,
        source_grounding=sg,
        citation_integrity=cite,
        draft_evolution=de,
        structural_reuse=sr,
        burstiness=burst,
        polish_ungrounded=pug,
        surprisal_risk=surp,
        topk_pattern_risk=topk,
    )


def _extract_predictability(summary) -> Optional[float]:
    """Predictability signal from PredictabilitySummary.overall_risk."""
    if summary is None:
        return None
    return clamp(summary.overall_risk)


def _extract_genericity(summary, findings) -> Optional[float]:
    """Genericity from phrase count + top10_ratio distribution.

    Formula:
        0.40 * phrase_density (count/8, capped at 1.0)
      + 0.30 * high_top10_ratio (sentences with top10 > 0.7)
      + 0.30 * mean_top10_of_flagged (avg top10 of medium/high sentences)
    """
    if summary is None:
        return None

    phrase_count = len(summary.generic_phrases_found) if summary.generic_phrases_found else 0
    phrase_density = min(phrase_count / 8.0, 1.0)

    sentences = summary.sentences or []
    if not sentences:
        return clamp(0.40 * phrase_density)

    # High top10 ratio: fraction of sentences with top10_ratio > 0.7
    high_top10 = sum(1 for s in sentences if s.get("top10_ratio", 0) > 0.7) / len(sentences)

    # Mean top10_ratio of flagged (medium/high risk) sentences
    flagged = [s for s in sentences if s.get("risk_label", "") in ("medium", "high")]
    mean_flagged_top10 = (
        sum(s.get("top10_ratio", 0) for s in flagged) / len(flagged)
        if flagged else 0.0
    )

    return clamp(
        0.40 * phrase_density +
        0.30 * high_top10 +
        0.30 * mean_flagged_top10
    )


def _extract_specificity(findings) -> Optional[float]:
    """Specificity signal from low_specificity finding's adjusted_specificity_concern."""
    if not findings:
        return None

    for f in findings:
        if getattr(f, "title", "") == "low_specificity":
            meta = getattr(f, "metadata", {})
            if meta and isinstance(meta, dict):
                concern = meta.get("adjusted_specificity_concern")
                if concern is not None:
                    return clamp(concern)
    return None


def _extract_citation_integrity(citation_summary) -> Optional[float]:
    """Citation integrity from CitationSummary stats when bibliography was provided."""
    if citation_summary is None:
        return None

    stats = citation_summary.stats or {}
    in_text = stats.get("in_text_citations", citation_summary.in_text_count)
    bib_entries = stats.get("bibliography_entries", citation_summary.bib_entry_count)

    # If no bibliography was provided, we can't assess citation integrity
    if bib_entries == 0 and in_text == 0:
        return None

    # Derive concern from mismatch indicators
    findings = citation_summary.findings or []
    if not findings:
        return 0.0  # Bibliography provided, no issues found

    issue_count = len(findings)
    # Normalize: 1-2 issues = low concern, 3+ = moderate, 5+ = high
    concern = clamp(issue_count / 6.0)
    return round(concern, 4)
