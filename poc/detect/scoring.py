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


def estimate_burstiness_risk(text: str) -> float:
    """Higher = more even rhythm, potentially AI-like. Weak signal only."""
    sentences = _split_sentences(text)
    if len(sentences) < 6:
        return 0.50
    lengths = [len(s.split()) for s in sentences if s.split()]
    if len(lengths) < 6:
        return 0.50
    mean_l = _statistics.mean(lengths)
    if mean_l == 0:
        return 0.50
    cv = _statistics.pstdev(lengths) / mean_l
    if cv >= 0.75:
        return 0.15
    if cv >= 0.55:
        return 0.30
    if cv >= 0.40:
        return 0.50
    if cv >= 0.25:
        return 0.70
    return 0.85


def estimate_lived_detail_risk(text: str) -> float:
    """Higher = weaker concrete/observable detail."""
    sentences = _split_sentences(text)
    if not sentences:
        return 0.70
    patterns = [
        r"\b\d+\b", r"\bfirst\b|\bsecond\b|\bthird\b",
        r"\bduring\b", r"\bwhen\b", r"\bafter\b", r"\bbefore\b",
        r"\bclassmate\b", r"\binterviewee\b", r"\bparticipant\b",
        r"\buser\b", r"\bprototype\b", r"\bwe observed\b",
        r"\bwe noticed\b", r"\bfeedback\b", r"\btesting\b",
        r"\bcompartment\b", r"\blabel\b", r"\btransparent\b",
        r"\balarm\b", r"\bmedication\b",
    ]
    hits = sum(
        1 for s in sentences
        if any(_re.search(p, s.lower()) for p in patterns)
    )
    ratio = hits / max(len(sentences), 1)
    if ratio >= 0.45:
        return 0.20
    if ratio >= 0.30:
        return 0.35
    if ratio >= 0.20:
        return 0.50
    if ratio >= 0.10:
        return 0.65
    return 0.80


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


# ── Layer 3: Cluster score functions ──────────────────────────────────

def threshold_risk(value: float, normal: float, high: float) -> float:
    """Convert raw metric to 0-1 risk band.

    value <= normal => 0.0
    value >= high   => 1.0
    between         => scaled linearly
    """
    if value <= normal:
        return 0.0
    if value >= high:
        return 1.0
    return (value - normal) / (high - normal)


def estimate_paragraph_progression_risk(text: str) -> float:
    """Detect formulaic essay progression patterns. Higher = more formulaic."""
    lower = text.lower()
    markers = [
        "in the past",
        "now",
        "this shift",
        "however",
        "technology has also",
        "another issue",
        "because of this",
        "today",
        "the goal should",
        "moving forward",
        "in conclusion",
        "ultimately",
        "it is clear",
        "it is evident",
    ]
    hits = sum(1 for m in markers if m in lower)
    ratio = hits / len(markers)
    if ratio >= 0.65:
        return 0.85
    if ratio >= 0.45:
        return 0.65
    if ratio >= 0.30:
        return 0.45
    return 0.20


def estimate_paragraph_uniformity_risk(text: str) -> float:
    """Detect uniform paragraph structure. Higher = more uniform."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 3:
        return 0.20
    lengths = [len(p.split()) for p in paragraphs]
    mean_l = _statistics.mean(lengths)
    if mean_l == 0:
        return 0.20
    cv = _statistics.pstdev(lengths) / mean_l
    if cv <= 0.15:
        return 0.90
    if cv <= 0.25:
        return 0.70
    if cv <= 0.40:
        return 0.45
    return 0.20


def estimate_repeated_starter_risk(text: str) -> float:
    """Detect repeated sentence starters across paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 3:
        return 0.20
    starters = []
    for p in paragraphs:
        words = p.split()
        if len(words) >= 2:
            starters.append(" ".join(words[:2]).lower())
    if not starters:
        return 0.20
    unique = len(set(starters))
    total = len(starters)
    if total == 0:
        return 0.20
    repeat_ratio = 1 - (unique / total)
    if repeat_ratio >= 0.40:
        return 0.75
    if repeat_ratio >= 0.25:
        return 0.50
    return 0.20


def estimate_formulaic_conclusion_risk(text: str) -> float:
    """Detect template-like conclusion patterns."""
    lower = text.lower()
    patterns = [
        "in conclusion",
        "to conclude",
        "in summary",
        "ultimately",
        "the goal should not be",
        "stands at a turning point",
        "needs to evolve",
        "it is clear that",
        "it is evident that",
        "moving forward",
    ]
    hits = sum(1 for p in patterns if p in lower)
    if hits >= 4:
        return 0.85
    if hits >= 3:
        return 0.65
    if hits >= 2:
        return 0.45
    return 0.20


def compute_structure_process_cluster(text: str) -> float:
    """Structure-based process risk from final text alone (no draft history needed)."""
    prog = estimate_paragraph_progression_risk(text)
    uniform = estimate_paragraph_uniformity_risk(text)
    starter = estimate_repeated_starter_risk(text)
    concl = estimate_formulaic_conclusion_risk(text)
    return clamp(
        0.35 * prog
        + 0.25 * uniform
        + 0.20 * starter
        + 0.20 * concl
    )


def compute_text_pattern_cluster(
    ai_likelihood: float,
    predictability: float,
    genericity: float,
    burstiness: float,
    surprisal_risk: float = 0.0,
    topk_pattern_risk: float = 0.0,
) -> float:
    """Text-style risk cluster (0-1). Higher = more AI-like text patterns."""
    ai = clamp(ai_likelihood)
    pred_risk = threshold_risk(clamp(predictability), normal=0.35, high=0.55)
    gen = clamp(genericity)
    burst = clamp(burstiness)
    topk_risk = threshold_risk(clamp(topk_pattern_risk), normal=0.45, high=0.70)

    return clamp(
        0.30 * pred_risk
        + 0.20 * topk_risk
        + 0.20 * gen
        + 0.15 * burst
        + 0.15 * ai
    )


def compute_grounding_quality_risk(
    lived_detail_risk: float,
    citation_risk: float,
) -> float:
    """Grounding quality risk cluster (0-1). Higher = weaker grounding."""
    return clamp(0.60 * clamp(lived_detail_risk) + 0.40 * clamp(citation_risk))


def compute_process_cluster(
    draft_evolution_risk: float = 0.0,
    structural_reuse_risk: float = 0.0,
    text: str = "",
) -> float:
    """Process risk cluster (0-1). Uses draft evidence if available, else structure detection."""
    has_draft = draft_evolution_risk > 0.15 or structural_reuse_risk > 0.15

    structure_score = compute_structure_process_cluster(text) if text else 0.20

    if has_draft:
        return clamp(
            0.55 * clamp(draft_evolution_risk)
            + 0.45 * structure_score
        )
    return structure_score


def calibrate_ai_risk(
    ai_likelihood: float,
    predictability: float,
    specificity_raw: float,
    genericity: float,
    source_grounding: float,
    domain_grounding_index: float,
    confidence: str = "low",
    high_findings: int = 0,
    critical_findings: int = 0,
    burstiness_risk: float = 0.50,
    lived_detail_risk: float = 0.50,
    citation_risk: float = 0.50,
    provenance_positive: bool = False,
    surprisal_risk: float = 0.0,
    topk_pattern_risk: float = 0.0,
    draft_evolution_risk: float = 0.0,
    structural_reuse_risk: float = 0.0,
    text: str = "",
) -> Dict[str, Any]:
    """Calibrated AI-risk badge v7 — normalized threshold risks + structural process detection."""
    ai = clamp(ai_likelihood)
    pred = clamp(predictability)
    spec = clamp(specificity_raw)
    gen = clamp(genericity)
    src = clamp(source_grounding)
    dom = _domain_strength(domain_grounding_index)
    burst = clamp(burstiness_risk)
    lived = clamp(lived_detail_risk)
    cite = clamp(citation_risk)

    # Layer 3: Cluster scores via named functions
    text_style_score = compute_text_pattern_cluster(
        ai_likelihood=ai, predictability=pred, genericity=gen,
        burstiness=burst, surprisal_risk=surprisal_risk,
        topk_pattern_risk=topk_pattern_risk,
    )
    grounding_quality = compute_grounding_quality_risk(lived, cite)
    process_score = compute_process_cluster(
        draft_evolution_risk, structural_reuse_risk, text=text,
    )

    # Small grounding credit (max 15%)
    grounding_credit = min(0.15, 0.075 * src + 0.075 * dom)

    score = (
        0.30 * text_style_score
        + 0.40 * grounding_quality
        + 0.30 * process_score
        + 0.05 * (1.0 if provenance_positive else 0.0)
    )

    # ── PRE-CALIBRATION SNAPSHOT (Layer 3 → before Layer 4 adjustments) ──
    pre_calibration = {
        "cluster_blended": round(score * 100, 2),
        "text_pattern": round(text_style_score * 100, 2),
        "grounding_quality_risk": round(grounding_quality * 100, 2),
        "structure_process": round(process_score * 100, 2),
    }

    score *= 1 - grounding_credit

    if ai >= 0.10 or gen >= 0.08 or pred >= 0.43:
        score *= _confidence_multiplier(confidence)

    pattern_reasons = []

    # Known AI cluster
    if (
        ai >= 0.12 and pred >= 0.45 and gen >= 0.10
        and spec >= 0.50 and confidence in ("medium", "high")
    ):
        score += 0.10
        pattern_reasons.append("known_ai_cluster")

    # Borderline AI cluster
    if (
        ai >= 0.10 and pred >= 0.43 and gen >= 0.08
        and confidence in ("medium", "high")
    ):
        score += 0.05
        pattern_reasons.append("borderline_ai_cluster")

    if gen >= 0.15 and pred >= 0.45:
        score += 0.04
        pattern_reasons.append("generic_predictable_style")

    if lived >= 0.65 and (ai >= 0.10 or gen >= 0.08):
        score += 0.04
        pattern_reasons.append("weak_lived_detail_with_ai_style")

    if process_score >= 0.60:
        score += 0.04
        pattern_reasons.append("high_process_risk")

    if high_findings > 0:
        score += min(0.06, 0.02 * high_findings)
        pattern_reasons.append("high_findings_present")

    if critical_findings > 0:
        score += 0.12
        pattern_reasons.append("critical_findings_present")

    score = clamp(score)
    tier, turnitin_band = _classify_score(score)

    # Provenance override
    if provenance_positive:
        tier, turnitin_band = "RED", "high_ai_concern"
        pattern_reasons.append("verified_ai_provenance")

    # Clean human-like protection
    if (
        ai < 0.10 and gen == 0 and pred < 0.40
        and src >= 0.90 and dom == 1.0
        and high_findings == 0 and critical_findings == 0
        and not provenance_positive
    ):
        tier, turnitin_band = "GREEN", "likely_0"
        pattern_reasons.append("clean_low_ai_profile")

    # Low AI cap
    if (
        ai < 0.12 and gen < 0.05 and pred < 0.45
        and high_findings == 0 and critical_findings == 0
        and not provenance_positive
    ):
        if tier in ("ORANGE", "RED"):
            tier, turnitin_band = "AMBER", "likely_0_or_star"
            pattern_reasons.append("low_ai_cap_applied")

    # Red-flag cluster floor
    red_flags = 0
    if ai >= 0.12:
        red_flags += 1
    if pred >= 0.45:
        red_flags += 1
    if gen >= 0.10:
        red_flags += 1
    if spec >= 0.50:
        red_flags += 1
    if confidence in ("medium", "high"):
        red_flags += 1

    if red_flags >= 5:
        tier, turnitin_band = "RED", "high_ai_concern"
        pattern_reasons.append("red_flag_cluster_5_of_5")

    # Writing review score
    writing_review_score = clamp(
        0.35 * pred + 0.30 * spec + 0.20 * gen + 0.15 * burst
    )
    if writing_review_score < 0.35:
        writing_band = "clean"
    elif writing_review_score < 0.50:
        writing_band = "light_review"
    elif writing_review_score < 0.65:
        writing_band = "review"
    else:
        writing_band = "heavy_review"

    return {
        "tier": tier,
        "turnitin_like_band": turnitin_band,
        "calibrated_ai_score": round(score * 100, 2),
        "pre_calibration": pre_calibration,
        "ai_style_score": round(text_style_score * 100, 2),
        "grounding_quality_risk": round(grounding_quality * 100, 2),
        "structure_process_score": round(process_score * 100, 2),
        "grounding_credit": round(grounding_credit * 100, 2),
        "writing_review_score": round(writing_review_score * 100, 2),
        "writing_review_band": writing_band,
        "red_flags": red_flags,
        "pattern_reasons": pattern_reasons,
    }


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

    # High top10 ratio: fraction of sentences with top10 > 0.7
    high_top10 = sum(1 for s in sentences if s.get("top10", 0) > 0.7) / len(sentences)

    # Mean top10 of flagged (medium risk) sentences
    flagged = [s for s in sentences if s.get("risk", "") in ("medium", "high")]
    mean_flagged_top10 = (
        sum(s.get("top10", 0) for s in flagged) / len(flagged)
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
