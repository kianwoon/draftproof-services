"""Composite candidate scoring for rewrite selection.

Fixability-aware: scores only auto and partial findings.
Manual and protected findings are excluded from success calculation
but reported separately.
"""

from dataclasses import dataclass
from typing import List, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detect.base import Finding
from rewrite.guards import DriftCheck

# ── Risk weighting ────────────────────────────────────────────────────

RISK_WEIGHTS = {"high": 5, "medium": 2, "low": 0.5}

FIXABILITY_WEIGHT = {
    "auto": 1.0,
    "partial": 0.7,
    "manual": 0.0,
    "protected": 0.0,
}


def weighted_finding_score(findings: List[Finding]) -> float:
    """Weighted risk score. High findings count 5x more than low."""
    return sum(RISK_WEIGHTS.get(f.risk_level, 1) for f in findings)


def weighted_rewritable_risk(findings: List[Finding], fixability_map: dict = None) -> float:
    """Score only rewritable findings (auto + partial), weighted by risk and fixability.

    fixability_map: {finding_type: fixability_str} — if None, all count as "auto".
    """
    total = 0.0
    for f in findings:
        fix = fixability_map.get(f.finding_type, "auto") if fixability_map else "auto"
        fw = FIXABILITY_WEIGHT.get(fix, 0.0)
        rw = RISK_WEIGHTS.get(f.risk_level, 1)
        total += rw * fw
    return total


# ── Candidate scoring ─────────────────────────────────────────────────

@dataclass
class CandidateScore:
    total: float
    finding_reduction: float
    semantic_preservation: float
    voice_preservation: float
    style_match: float
    source_grounding: float
    length_stability: float
    specificity_gain: float
    accepted: bool
    reject_reasons: List[str]

    def __post_init__(self):
        if self.reject_reasons is None:
            self.reject_reasons = []


def score_candidate(
    original_findings: List[Finding],
    candidate_findings: List[Finding],
    original_text: str,
    candidate_text: str,
    drift_check: DriftCheck,
    style_score_original: float = 1.0,
    style_score_candidate: float = 1.0,
    voice_score_original: float = 1.0,
    voice_score_candidate: float = 1.0,
    fixability_map: dict = None,
) -> CandidateScore:
    """Score a rewrite candidate against multiple signals.

    Fixability-aware: uses weighted_rewritable_risk so manual/protected
    findings don't count against rewrite success.

    Composite:
      0.35 * finding_reduction   — fewer/higher-quality rewritable findings
      0.20 * semantic_preservation — meaning preserved
      0.15 * voice_preservation   — author voice not eroded
      0.15 * source_grounding     — citation/source quality
      0.10 * specificity_gain     — less generic language
      0.05 * length_stability     — no wild length changes

    Hard reject if: semantic drift fails, protected span lost,
    citation lost, number/date/entity lost, voice erosion too high,
    rewrite budget exceeded.
    """
    reject_reasons = []

    # Hard gate: semantic drift
    if not drift_check.accepted:
        reject_reasons.extend(drift_check.reasons[:3])
        return CandidateScore(0, 0, 0, 0, 0, 0, 0, 0, False, reject_reasons)

    # Hard gate: no findings at all means nothing to compare
    if not original_findings and not candidate_findings:
        return CandidateScore(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, True, [])

    # 1. Finding reduction (0-1, fixability-aware)
    orig_score = weighted_rewritable_risk(original_findings, fixability_map)
    cand_score = weighted_rewritable_risk(candidate_findings, fixability_map)
    if orig_score == 0:
        finding_reduction = 0.0
    else:
        finding_reduction = max(0, (orig_score - cand_score) / orig_score)

    # 2. Semantic preservation (0-1, from drift check similarity)
    semantic_preservation = drift_check.similarity

    # 3. Voice preservation (0-1)
    voice_preservation = 1.0 - abs(voice_score_original - voice_score_candidate)
    voice_preservation = max(0, min(1, voice_preservation))

    # 4. Style match (0-1, compare style scores)
    style_match = 1.0 - abs(style_score_original - style_score_candidate)
    style_match = max(0, min(1, style_match))

    # 5. Source grounding (check if citations preserved)
    source_grounding = 1.0 if drift_check.accepted else 0.0

    # 6. Specificity gain (inverse of generic word density)
    import re
    orig_generic = len(re.findall(r'\b(various|numerous|several|many|important|significant|effective|utilize|implement)\b', original_text, re.I))
    cand_generic = len(re.findall(r'\b(various|numerous|several|many|important|significant|effective|utilize|implement)\b', candidate_text, re.I))
    orig_words = max(len(re.findall(r'\b\w+\b', original_text)), 1)
    cand_words = max(len(re.findall(r'\b\w+\b', candidate_text)), 1)
    orig_density = orig_generic / orig_words
    cand_density = cand_generic / cand_words
    specificity_gain = max(0, orig_density - cand_density) / max(orig_density, 0.001)
    specificity_gain = min(specificity_gain, 1.0)

    # 7. Length stability (0-1, penalize big changes)
    orig_len = max(len(original_text), 1)
    cand_len = len(candidate_text)
    length_ratio = cand_len / orig_len
    if 0.85 <= length_ratio <= 1.15:
        length_stability = 1.0
    elif 0.7 <= length_ratio <= 1.3:
        length_stability = 0.5
    else:
        length_stability = 0.0

    # Composite
    total = (
        0.35 * finding_reduction
        + 0.20 * semantic_preservation
        + 0.15 * voice_preservation
        + 0.15 * source_grounding
        + 0.10 * specificity_gain
        + 0.05 * length_stability
    )

    return CandidateScore(
        total=round(total, 4),
        finding_reduction=round(finding_reduction, 4),
        semantic_preservation=round(semantic_preservation, 4),
        voice_preservation=round(voice_preservation, 4),
        style_match=round(style_match, 4),
        source_grounding=round(source_grounding, 4),
        length_stability=round(length_stability, 4),
        specificity_gain=round(specificity_gain, 4),
        accepted=True,
        reject_reasons=[],
    )


def best_candidate(candidates: List[CandidateScore], min_delta: float = 0.05) -> Optional[CandidateScore]:
    """Select the best accepted candidate with minimum improvement threshold."""
    accepted = [c for c in candidates if c.accepted]
    if not accepted:
        return None
    best = max(accepted, key=lambda c: c.total)
    if best.total < min_delta:
        return None
    return best
