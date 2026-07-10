"""
DraftProof Layer 3 Scoring Engine
=================================

Purpose:
    Two-phase scoring engine:
    - Phase 1: AI Generation Likelihood (mechanical signals, zero grounding credit)
    - Phase 2: Writing Quality Risk (evidence + structure signals)
    - Combined review priority and tier badge

Design principles:
    1. AI Likelihood is driven purely by mechanical signals — grounding cannot lower it.
    2. Cluster rules catch "humanised AI" where no single metric is extreme.
    3. Writing Quality is evidence-heavy, with grounding credit capped at 15%.
    4. Missing evidence lowers confidence, not guilt.
    5. All scores are normalized 0.0 - 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from enum import Enum
from typing import Any, Optional
import math
import re
import statistics


class Tier(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    ORANGE = "ORANGE"
    RED = "RED"


class QualityTier(str, Enum):
    LOW = "LOW"
    LIGHT_REVIEW = "LIGHT_REVIEW"
    REVIEW = "REVIEW"
    HIGH_REVIEW = "HIGH_REVIEW"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Layer3Input:
    """
    Inputs from lower-level scanners.

    All percentage metrics are decimals.
    Example: 43.8% -> 0.438
    """

    # Text pattern scanner outputs
    predictability: Optional[float] = None
    topk_pattern: Optional[float] = None
    sentence_fragmentation_risk: Optional[float] = None
    generic_phrase_density: Optional[float] = None
    burstiness_risk: Optional[float] = None
    repeated_sentence_structure_risk: Optional[float] = None
    generic_assertion_risk: Optional[float] = None
    qualifying_text_ai_density: Optional[float] = None

    # Grounding scanner outputs
    broad_claim_risk: Optional[float] = None
    lived_detail_risk: Optional[float] = None
    citation_weakness_risk: Optional[float] = None
    unsupported_claim_risk: Optional[float] = None
    source_grounding_strength: Optional[float] = None
    domain_grounding_strength: Optional[float] = None

    # Structure / process scanner outputs
    paragraph_progression_risk: Optional[float] = None
    paragraph_uniformity_risk: Optional[float] = None
    repeated_starter_risk: Optional[float] = None
    formulaic_conclusion_risk: Optional[float] = None
    signpost_paragraph_risk: Optional[float] = None
    balanced_hedging_risk: Optional[float] = None
    intra_paragraph_parallelism_risk: Optional[float] = None
    paragraph_topic_uniformity_risk: Optional[float] = None
    draft_evolution_jump_risk: Optional[float] = None
    structural_reuse_risk: Optional[float] = None
    style_shift_risk: Optional[float] = None

    # Provenance
    human_provenance_positive: bool = False
    verified_ai_provenance: bool = False

    # Document context
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0


@dataclass
class ClusterScore:
    name: str
    score: float
    confidence: Confidence
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class Layer3Result:
    tier: Tier
    ai_likelihood_score: float
    authorship_rating: dict[str, Any]
    ai_cluster_boost: float
    ai_cluster_name: Optional[str]
    ai_phase: ClusterScore
    writing_quality_score: float
    writing_quality_tier: QualityTier
    writing_phase: ClusterScore
    confidence: Confidence
    reasons: list[str]
    guardrails: list[str]
    review_priority: str
    debug: dict[str, Any]
    # True when the tier verdict sits near a boundary cutoff or on a thin sample — i.e. the
    # verdict is unstable and small input changes could flip it. STRICTLY DISPLAY-ONLY: never
    # alters the tier/score; surfaces as a "low confidence" qualifier to the user (mirrors the
    # reliability-floor concept the DeBERTa advisory layer and mature detectors like Turnitin
    # apply to near-boundary / short-sample results). Distinct from `confidence` (signal
    # coverage); this is verdict stability.
    verdict_low_confidence: bool = False


def clamp(value: Optional[float], low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return max(low, min(high, value))


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def threshold_risk(value: Optional[float], normal: float, high: float) -> float:
    """
    Normalize a metric into a 0..1 risk band.

    Example:
        predictability = 0.468
        normal = 0.35
        high = 0.55

        risk = (0.468 - 0.35) / (0.55 - 0.35) = 0.59
    """
    v = clamp(value)
    if v <= normal:
        return 0.0
    if v >= high:
        return 1.0
    return (v - normal) / (high - normal)


def weighted_average(items: dict[str, tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in items.values())
    if total_weight <= 0:
        return 0.0

    return clamp(
        sum(value * weight for value, weight in items.values()) / total_weight
    )


def confidence_from_coverage(
    word_count: int,
    sentence_count: int,
    available_count: int,
    expected_count: int,
    important_missing: int = 0,
) -> Confidence:
    score = 1.0

    if word_count < 150:
        score -= 0.40
    elif word_count < 500:
        score -= 0.20

    if sentence_count < 6:
        score -= 0.25

    coverage = safe_ratio(available_count, expected_count)
    if coverage < 0.50:
        score -= 0.30
    elif coverage < 0.75:
        score -= 0.15

    score -= 0.10 * important_missing

    if score >= 0.75:
        return Confidence.HIGH
    if score >= 0.45:
        return Confidence.MEDIUM
    return Confidence.LOW


def derive_authorship_rating(
    *,
    ai_score: float,
    ai_tier: Tier,
    writing_quality_score: float,
    writing_quality_tier: QualityTier,
    confidence: Confidence,
    verified_ai_provenance: bool = False,
    human_provenance_positive: bool = False,
    ai_components: Optional[dict[str, float]] = None,
    writing_components: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Translate detector scores into a user-facing authorship rating.

    This is deliberately a rating layer, not a verdict layer. It keeps the
    numeric score as the source of truth and labels the strength of scanner
    signals in language that users can act on.
    """
    ai_components = ai_components or {}
    writing_components = writing_components or {}
    high_component_alignment = (
        ai_score >= 0.58
        and (
            clamp(ai_components.get("topk_pattern")) >= 0.80
            or clamp(ai_components.get("qualifying_text_ai_density")) >= 0.70
        )
        and clamp(ai_components.get("generic_assertion_risk")) >= 0.80
    )
    high_density_alignment = (
        ai_score >= 0.45
        and clamp(ai_components.get("qualifying_text_ai_density")) >= 0.70
        and clamp(ai_components.get("topk_pattern")) >= 0.60
        and clamp(ai_components.get("generic_assertion_risk")) >= 0.80
    )
    likely_component_alignment = (
        ai_score >= 0.48
        and (
            clamp(ai_components.get("topk_pattern")) >= 0.70
            or clamp(ai_components.get("generic_assertion_risk")) >= 0.70
        )
    )

    if verified_ai_provenance:
        code = "ai_generated"
    elif ai_score >= 0.60 or ai_tier == Tier.RED or high_component_alignment or high_density_alignment:
        code = "ai_generated_signals"
    elif likely_component_alignment:
        code = "likely_ai"
    elif ai_score >= 0.32 or ai_tier == Tier.AMBER:
        code = "possible_ai_assisted"
    elif ai_score >= 0.20:
        code = "unlikely_ai"
    else:
        code = "low_ai_signal"

    definitions = {
        "low_ai_signal": {
            "label": "Low AI Signal",
            "short_label": "Low Signal",
            "risk_level": "minimal",
            "summary": "AI-style signal strength is below the level DraftProof surfaces as an authorship concern.",
            "recommended_action": "Do not treat this as an AI concern; review only citation, quotation, or writing-quality findings if present.",
        },
        "unlikely_ai": {
            "label": "Low AI-writing signal",
            "short_label": "Low AI signal",
            "risk_level": "low",
            "summary": "Low AI-style signal strength with some normal review signals.",
            "recommended_action": "Review only the highlighted sentences or weak source/detail findings.",
        },
        "possible_ai_assisted": {
            "label": "Possible AI-Assisted",
            "short_label": "Possible AI",
            "risk_level": "medium",
            "summary": "Moderate AI-style signals are present, but the scan is not strong enough for a high-confidence label.",
            "recommended_action": "Use guided revision: add evidence, narrow broad claims, and retry detector-gated rewrite.",
        },
        "likely_ai": {
            "label": "Likely AI",
            "short_label": "Likely AI",
            "risk_level": "high",
            "summary": "Elevated AI-style signals align across enough scanner components to require careful review.",
            "recommended_action": "Prioritize paragraph-level grounding, source links, concrete examples, and final full-scan gating.",
        },
        "ai_generated_signals": {
            "label": "AI-Generated / AI-Paraphrased Signals",
            "short_label": "AI Signals",
            "risk_level": "very_high",
            "summary": "High AI-style signal strength across the detect pipeline, including patterns consistent with generated or AI-paraphrased text.",
            "recommended_action": "Preserve the original unless revision produces a clearly better final scan; use manual evidence additions first.",
        },
        "ai_generated": {
            "label": "AI-Generated",
            "short_label": "AI-Generated",
            "risk_level": "verified",
            "summary": "Verified AI provenance was provided to the scanner.",
            "recommended_action": "Treat as confirmed AI-assisted content and revise according to the user's disclosure requirements.",
        },
    }
    rating = dict(definitions[code])
    rating.update({
        "code": code,
        "score": round(ai_score * 100, 2),
        "confidence": confidence.value,
        "ai_tier": ai_tier.value,
        "writing_quality_score": round(writing_quality_score * 100, 2),
        "writing_quality_tier": writing_quality_tier.value,
        "is_verdict": False,
        "disclaimer": "This rating summarizes DraftProof detector signals. It is not proof of authorship.",
    })

    caution_notes = []
    if code == "low_ai_signal":
        caution_notes.append("Scores under 20% have a higher false-positive risk and should not be treated as an AI finding.")
    if confidence == Confidence.LOW:
        caution_notes.append("Low confidence: text length or available signals are limited.")
    if human_provenance_positive:
        caution_notes.append("Positive human provenance was considered; review the rating in that context.")
    if ai_score >= 0.60 and code == "ai_generated_signals":
        caution_notes.append("Escalated because AI authorship score is near the high-risk threshold.")
    if writing_quality_tier in (QualityTier.REVIEW, QualityTier.HIGH_REVIEW) and code in {
        "possible_ai_assisted",
        "likely_ai",
        "ai_generated_signals",
    }:
        caution_notes.append("Writing-quality risks are reported separately and are not proof of AI authorship.")
    if high_component_alignment and code == "ai_generated_signals":
        caution_notes.append("Escalated because high top-k predictability and generic assertion signals align.")
    if high_density_alignment and code == "ai_generated_signals":
        caution_notes.append("Escalated because qualifying long-form text has dense aligned AI-style signals.")
    if clamp(ai_components.get("qualifying_text_ai_density")) >= 0.70 and code == "ai_generated_signals":
        caution_notes.append("Qualifying long-form text shows dense AI-style signal alignment across the document.")
    rating["caution_notes"] = caution_notes
    return rating


def merge_confidence(*levels: Confidence) -> Confidence:
    order = {
        Confidence.LOW: 0,
        Confidence.MEDIUM: 1,
        Confidence.HIGH: 2,
    }
    return min(levels, key=lambda x: order[x])


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n+", text.strip())
    return [re.sub(r"\s+", " ", p).strip() for p in blocks if p.strip()]


# Generic essay-opener phrases removed (NO-HARDCODE; these were overfit content phrases such as
# "technology has also" / "the real challenge"). "Generic" openers are captured by token-level
# predictability, not a baked phrase list. Consumers tolerate the empty list (they iterate it).
GENERIC_ESSAY_STARTERS: list[str] = []


def estimate_formulaic_progression_risk(text: str) -> float:
    # NO-HARDCODE: AI-cliche phrase list removed; signal disabled. The agnostic
    # statistical signals (predictability/topk/burstiness/paragraph_uniformity) cover
    # 'formulaic AI prose'. Stub kept; remove from aggregation in the structural follow-up.
    return 0.0


def estimate_balanced_generic_framing_risk(text: str) -> float:
    # NO-HARDCODE: AI-cliche phrase list removed; signal disabled. The agnostic
    # statistical signals (predictability/topk/burstiness/paragraph_uniformity) cover
    # 'formulaic AI prose'. Stub kept; remove from aggregation in the structural follow-up.
    return 0.0


def estimate_repeated_starter_risk(text: str) -> float:
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 3:
        return 0.0

    starts = []
    for p in paragraphs:
        first_words = " ".join(p.lower().split()[:4])
        starts.append(first_words)

    starter_hits = 0
    for starter in GENERIC_ESSAY_STARTERS:
        starter_hits += sum(1 for s in starts if s.startswith(starter))

    ratio = starter_hits / len(paragraphs)

    if ratio >= 0.60:
        return 0.85
    if ratio >= 0.40:
        return 0.65
    if ratio >= 0.25:
        return 0.45
    if ratio >= 0.15:
        return 0.25
    return 0.0


def estimate_paragraph_uniformity_risk(text: str) -> float:
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 4:
        return 0.0

    lengths = [len(p.split()) for p in paragraphs if p.split()]
    if len(lengths) < 4:
        return 0.0

    mean_len = statistics.mean(lengths)
    if mean_len <= 0:
        return 0.0

    cv = statistics.pstdev(lengths) / mean_len

    if cv <= 0.15:
        return 0.85
    if cv <= 0.25:
        return 0.65
    if cv <= 0.35:
        return 0.45
    if cv <= 0.50:
        return 0.25
    return 0.0


def estimate_signpost_paragraph_risk(text: str) -> float:
    """Higher = text has many short signpost/bridge paragraphs.

    AI essays produce a distinctive pattern: long body paragraphs separated by
    1-2 sentence "signpost" paragraphs that merely announce the next topic.
    Human writers tend to integrate transitions within paragraphs instead.
    """
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 4:
        return 0.0

    # Calculate median paragraph word count as reference
    lengths = [len(p.split()) for p in paragraphs if p.strip()]
    if len(lengths) < 4:
        return 0.0

    median_len = statistics.median(lengths)
    if median_len <= 0:
        return 0.0

    signpost_count = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        word_count = len(p.split())
        sent_count = len([s for s in re.split(r'[.!?]+', p) if s.strip()])

        # Signpost = a short "bridge" paragraph between long ones: <=2 sentences and well
        # below the median paragraph length. Purely STRUCTURAL (length/sentence-count) -- no
        # announcement-phrase list. AI essays over-produce these short transition paragraphs.
        if sent_count <= 2 and word_count < median_len * 0.35:
            signpost_count += 1

    ratio = signpost_count / len(paragraphs)
    if ratio >= 0.40:
        return 0.90
    if ratio >= 0.30:
        return 0.75
    if ratio >= 0.20:
        return 0.55
    if ratio >= 0.10:
        return 0.30
    return 0.0


def estimate_balanced_hedging_risk(text: str) -> float:
    """Higher = text uses many AI-typical balanced/hedging constructions.

    AI essays over-use balanced constructions: "While X is important, Y should
    also be considered", "not only A but also B", "plays a crucial role".
    These create a distinctive even-handed, tempered tone that's uncommon
    in human first-draft writing.
    """
    # NO-HARDCODE: AI-cliche phrase list removed; signal disabled. The agnostic
    # statistical signals (predictability/topk/burstiness/paragraph_uniformity) cover
    # 'formulaic AI prose'. Stub kept; remove from aggregation in the structural follow-up.
    return 0.0


def estimate_intra_paragraph_parallelism_risk(text: str) -> float:
    """Higher = paragraphs have repeated sentence-start patterns internally.

    AI tends to produce paragraphs where multiple sentences follow the same
    grammatical template: "X is important because... Y is important because..."
    or "This means... This shows... This suggests..."
    """
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 2:
        return 0.0

    parallel_count = 0
    total_paragraphs = 0

    for p in paragraphs:
        sentences = [s.strip() for s in re.split(r'[.!?]+', p) if len(s.strip()) > 10]
        if len(sentences) < 3:
            continue
        total_paragraphs += 1

        # Extract first 2 words of each sentence as starter pattern
        starters = []
        for s in sentences:
            words = s.lower().split()[:2]
            if words:
                starters.append(' '.join(words))

        # Count repeated 2-word starters
        starter_counts = Counter(starters)
        max_repeat = max(starter_counts.values())
        if max_repeat >= 3:
            parallel_count += 1
        elif max_repeat >= 2 and len(starters) >= 4:
            # 2+ same starters in 4+ sentence paragraph
            repeated_starter_types = sum(1 for c in starter_counts.values() if c >= 2)
            if repeated_starter_types >= 1:
                parallel_count += 0.5

    if total_paragraphs == 0:
        return 0.0

    ratio = parallel_count / total_paragraphs
    if ratio >= 0.50:
        return 0.85
    if ratio >= 0.35:
        return 0.65
    if ratio >= 0.20:
        return 0.45
    if ratio >= 0.10:
        return 0.25
    return 0.0


def estimate_paragraph_topic_uniformity_risk(text: str) -> float:
    """Higher = every paragraph has a very similar internal structure.

    AI essays produce paragraphs with nearly identical sentence counts and
    topic scope: each paragraph introduces one idea, elaborates, and concludes.
    Human writing has more varied paragraph architecture.
    """
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 4:
        return 0.0

    # Measure sentence count per paragraph
    sent_counts = []
    for p in paragraphs:
        sents = [s for s in re.split(r'[.!?]+', p) if len(s.strip()) > 5]
        if sents:
            sent_counts.append(len(sents))

    if len(sent_counts) < 4:
        return 0.0

    mean_sents = statistics.mean(sent_counts)
    if mean_sents <= 0:
        return 0.0

    cv = statistics.pstdev(sent_counts) / mean_sents

    # Very uniform sentence count across paragraphs = suspicious
    if cv <= 0.10:
        return 0.85
    if cv <= 0.18:
        return 0.65
    if cv <= 0.25:
        return 0.45
    if cv <= 0.35:
        return 0.25
    return 0.0


def estimate_sentence_fragmentation_risk(text: str) -> float:
    """Mechanical sentence-splitting risk ('de-AI fragmentation').

    Prose chopped into many uniform short sentences starves the structural AI signals
    (qualifying-text density, repeated-structure, hedging), letting AI content read as clean/green.
    Calibrated on the test corpus: normal writing sits at mean 14-26 words/sentence with <30% short
    sentences; only heavily fragmented text clears mean<=11 words AND >=70% of sentences <=12 words.
    Returns 0 for everything else, so this re-tiers nothing on its own -- it only feeds the
    fragmented_evasion cluster, which additionally requires high content-AI signals."""
    sentences = split_sentences(text)
    lengths = [len(s.split()) for s in sentences if s.split()]
    if len(lengths) < 8:
        return 0.0
    mean_len = sum(lengths) / len(lengths)
    short_12 = sum(1 for n in lengths if n <= 12) / len(lengths)
    short_8 = sum(1 for n in lengths if n <= 8) / len(lengths)
    if mean_len > 11 or short_12 < 0.70:
        return 0.0
    return round(max(0.0, min(1.0, (short_8 - 0.30) / 0.45)), 4)


def estimate_external_detector_likelihood(components: dict) -> dict:
    """Perplexity-forward estimate of how strict THIRD-PARTY AI detectors (Turnitin, GPTZero,
    Originality, ...) are likely to rate this text.

    Those tools key on raw token predictability (low perplexity) and uniform rhythm, and they
    OVER-FLAG -- including genuine human writing. DraftProof's own score deliberately discounts raw
    Top-k to avoid false-accusing writers, so it reads lower. This estimate exists only to set the
    user's expectation about external tools; it is NOT DraftProof's assessment, and it is
    intentionally a directional band (external tools vary widely and some rate even higher)."""
    def _v(key: str) -> float:
        value = components.get(key)
        return float(value) if isinstance(value, (int, float)) else 0.0

    topk_raw = _v("topk_pattern_raw") or _v("topk_pattern")
    predictability = _v("predictability")
    burstiness = _v("burstiness_risk")
    score = max(0.0, min(100.0, 0.55 * topk_raw + 0.25 * predictability + 0.20 * burstiness))
    if score >= 55.0 or topk_raw >= 65.0:
        band = "high"
    elif score >= 35.0:
        band = "elevated"
    else:
        band = "low"
    return {
        "score": round(score, 1),
        "band": band,
        "note": (
            "Estimated likelihood that strict third-party detectors (e.g. Turnitin, GPTZero) flag "
            "this as AI. They weight raw token predictability far more aggressively than DraftProof "
            "and tend to over-flag -- treat as an external-facing estimate, not DraftProof's "
            "assessment."
        ),
    }


# --- Calibrated 2-signal segment-fraction estimator -------------------------------------------
# Turnitin reports its AI score as the FRACTION of flagged qualifying sentences, and (from one
# labeled doc, Reflection AT3 = Turnitin 27%, flagged in the formal Critical-Analysis middle, with
# the plain conclusion spared) keys on BOTH predictability AND formal-academic register. This
# estimator flags a sentence when both clear their bar; the score is the flagged share of words.
# Thresholds fit on FULL sentence text to reproduce that doc's 26-27% (and exclude its conclusion).
# UNVALIDATED for generalisation; surfaced as an ESTIMATE. Canonical home; the calibration tool
# poc/calibration/segment_fraction_estimate.py imports this.
_TURNITIN_TOP10_THRESHOLD = 0.46
_TURNITIN_REGISTER_THRESHOLD = 1.1
_REGISTER_NOMINAL = re.compile(r"(tion|sion|ment|ance|ence|ity|ism)s?$", re.I)
_REGISTER_FIRST_PERSON = frozenset({"i", "my", "me", "i'm", "i've", "myself", "mine"})


def register_score(text: str) -> float:
    """Formal-academic register of a sentence, from STRUCTURAL/morphological cues only:
    nominalisation density (-tion/-ment/-ity… suffixes), mean word length, contraction rate
    (informality), minus first-person voice. Higher = more 'generated-essay' sounding.

    The curated formal-connective list (_REGISTER_CONNECTIVES: furthermore/whilst/hence…) was
    removed (NO-HARDCODE -- a hand-picked stylistic subset, not a complete grammatical class).
    Formality is now carried by nominalisation morphology + long words + low contraction rate.
    """
    words = re.findall(r"[A-Za-z']+", (text or "").lower())
    n = max(1, len(words))
    nominal = sum(1 for t in words if _REGISTER_NOMINAL.search(t)) / n
    first_person = sum(1 for t in words if t in _REGISTER_FIRST_PERSON) / n
    contraction = sum(1 for t in words if "'" in t) / n   # informal register marker
    mean_len = sum(len(t) for t in words) / n
    return nominal * 7 + max(0.0, mean_len - 4.3) * 0.8 - first_person * 5 - contraction * 4


def estimate_external_detector_segment_fraction(sentences) -> Optional[dict]:
    """Calibrated estimate: the flagged share of qualifying sentences (top10 >= bar AND register
    >= bar). `sentences` = iterable of dicts with 'text' and 'top10'. Returns the estimate dict, or
    None when no usable per-sentence data (caller falls back to the perplexity estimate)."""
    rows = []
    for s in sentences or []:
        if not isinstance(s, dict):
            continue
        # Accept both the builder's PredictabilitySummary keys ('sentence'/'top10_ratio') and the
        # final report-dict keys ('text'/'top10').
        text = s.get("text") or s.get("sentence") or ""
        top10 = s.get("top10")
        if top10 is None:
            top10 = s.get("top10_ratio")
        words = len(text.split())
        if isinstance(top10, (int, float)) and words:
            rows.append((float(top10), register_score(text), words))
    total = sum(w for _, _, w in rows)
    if not total:
        return None
    flagged = sum(w for p, r, w in rows
                  if p >= _TURNITIN_TOP10_THRESHOLD and r >= _TURNITIN_REGISTER_THRESHOLD)
    score = round(100.0 * flagged / total, 1)
    band = "high" if score >= 50.0 else "elevated" if score >= 20.0 else "low"
    return {
        "score": score,
        "band": band,
        "model": "segment_fraction_v1",
        "validated": False,
        "note": (
            "Estimated likelihood that strict third-party detectors (Turnitin, GPTZero) flag this "
            "as AI -- the share of formal, predictable sentences. Calibrated on limited data and "
            "imperfect (detectors over-flag fluent writing); a heads-up, not a verdict."
        ),
    }


def estimate_burstiness_risk(text: str) -> float:
    sentences = split_sentences(text)
    if len(sentences) < 6:
        return 0.0

    lengths = [len(s.split()) for s in sentences if s.split()]
    if len(lengths) < 6:
        return 0.0

    mean_len = statistics.mean(lengths)
    if mean_len <= 0:
        return 0.0

    cv = statistics.pstdev(lengths) / mean_len

    if cv <= 0.20:
        return 0.85
    if cv <= 0.30:
        return 0.65
    if cv <= 0.40:
        return 0.45
    if cv <= 0.55:
        return 0.25
    return 0.0


def estimate_repeated_sentence_structure_risk(text: str) -> float:
    sentences = split_sentences(text)
    if len(sentences) < 6:
        return 0.0

    starter_count = 0
    for s in sentences:
        first = " ".join(s.lower().split()[:3])
        if any(first.startswith(starter) for starter in GENERIC_ESSAY_STARTERS):
            starter_count += 1

    ratio = starter_count / len(sentences)

    if ratio >= 0.35:
        return 0.85
    if ratio >= 0.25:
        return 0.65
    if ratio >= 0.15:
        return 0.45
    if ratio >= 0.08:
        return 0.25
    return 0.0


def estimate_formulaic_conclusion_risk(text: str) -> float:
    # NO-HARDCODE: AI-cliche phrase list removed; signal disabled. The agnostic
    # statistical signals (predictability/topk/burstiness/paragraph_uniformity) cover
    # 'formulaic AI prose'. Stub kept; remove from aggregation in the structural follow-up.
    return 0.0


# Content-agnostic concreteness signals. Per the project rule (NO hardcoded allow-lists), a sentence
# is "concrete" because it carries a verifiable, specific anchor that holds in ANY subject -- never
# because it matches a domain's vocabulary. (Replaces the former hardcoded education/hairdressing
# phrase lists, which over-flagged any document written with different words.)
CONCRETE_DETAIL_PATTERNS = [
    r"\b\d+\.?\d*\s*%?\b",                              # numbers, quantities, percentages, years
    r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b",             # alphanumeric codes
    r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b",              # multi-word proper nouns / named entities
    r"\b[A-Z][A-Za-z]+(?:\s+et\s+al\.)?\s*\(\d{4}\)",  # citation: Name (2020)
    r"\([A-Z][A-Za-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)",   # citation: (Name, 2020)
    r"['\"“”‘’][^'\"“”‘’]+?['\"“”‘’]",  # quoted span
    r"\b(?:I|we)(?:'ve|'d| have| had)?\s+[a-z]{2,}(?:ed|es|s)?\b",  # first-person + verb (structural; no verb enumeration)
    r"\bin my (?:own\s+)?[a-z]+\b",                    # first-hand framing: "in my classroom/experience/practice"
]
NAMED_ENTITY_DETAIL_PATTERN = CONCRETE_DETAIL_PATTERNS[2]
# Domain-independent grounding STRUCTURES: exemplification, a specific case, a situational
# (temporal/conditional) clause, or an explicit evidence reference. No domain words.
CONTEXTUAL_ANCHOR_PATTERNS = [
    r"\b(?:for example|for instance|such as|e\.g\.|i\.e\.|in particular|specifically|namely)\b",
    r"\b(?:case study|(?:in\s+)?cases?\s+where|in the case of)\b",
    r"\b(?:when|whenever|after|before|during|once|as soon as)\s+\w+",  # situational / temporal clause
    r"\bif\s+\w+",                                                      # conditional clause
    r"\b(?:according to|data (?:show|shows|from|set)|study (?:found|shows|of))\b",  # evidence reference
]


def _sentence_has_concrete_or_context(sentence: str) -> bool:
    return any(
        re.search(p, sentence, flags=0 if p == NAMED_ENTITY_DETAIL_PATTERN else re.I)
        for p in CONCRETE_DETAIL_PATTERNS
    ) or any(re.search(p, sentence, flags=re.I) for p in CONTEXTUAL_ANCHOR_PATTERNS)


# HARD, verifiable specifics only: digit numbers, alphanumeric codes, multi-word proper nouns,
# citations, quotes (CONCRETE_DETAIL_PATTERNS[0:6]). Deliberately EXCLUDES the SOFT "frame" signals --
# first-person verbs ("I've seen"), "in my X", plural first-person ([6:9]) -- and ALL contextual
# anchors (such as / when / if / temporal). A sentence can carry the FORM of lived experience while
# stating an abstract claim ("I've seen the curriculum shift toward relevance"); that form-only case
# passes _sentence_has_concrete_or_context but has no real substance. Used by REWRITE candidate
# selection only -- the detection badge keeps the lenient oracle above so badge scores do not move.
_HARD_CONCRETE_PATTERNS = CONCRETE_DETAIL_PATTERNS[:6]


def _sentence_has_hard_concrete(sentence: str) -> bool:
    """True only if the sentence carries a hard, verifiable specific (number/code/proper-noun/
    citation/quote) -- the SUBSTANCE of grounding, not just its first-person/temporal form."""
    return any(
        re.search(p, sentence, flags=0 if p == NAMED_ENTITY_DETAIL_PATTERN else re.I)
        for p in _HARD_CONCRETE_PATTERNS
    )


def _contextual_anchor_density(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    hits = sum(1 for sentence in sentences if _sentence_has_concrete_or_context(sentence))
    return hits / len(sentences)


def estimate_generic_assertion_risk(text: str) -> float:
    """Higher = text makes claims without concrete details. Measures macro genericity.

    Complements phrase-level genericity by looking at whether sentences
    contain ANY concrete detail (numbers, names, quotes, dates).
    """
    sentences = split_sentences(text)
    if not sentences:
        return 0.70

    generic_count = 0
    for sentence in sentences:
        if not _sentence_has_concrete_or_context(sentence):
            generic_count += 1

    ratio = generic_count / len(sentences)
    if ratio >= 0.75:
        return 0.90
    if ratio >= 0.60:
        return 0.80
    if ratio >= 0.45:
        return 0.65
    if ratio >= 0.30:
        return 0.45
    return 0.25


# Closed-class structural cues for lived detail -- no content words. First-person framing
# (pronouns) and situational/temporal/conditional clauses (subordinating conjunctions).
_LIVED_FIRST_PERSON_RE = re.compile(r"\b(?:i|we|my|our|us|me)\b", re.I)
_SITUATIONAL_CLAUSE_RE = re.compile(r"\b(?:when|whenever|after|before|during|once|while|if)\b", re.I)


def estimate_lived_detail_risk(text: str, domain_patterns: Optional[list[str]] = None) -> float:
    sentences = split_sentences(text)
    if not sentences:
        return 0.70

    # Agnostic lived-detail: a sentence shows lived detail via a STRUCTURAL grounding cue --
    # a hard verifiable specific (number / alphanumeric code / multi-word proper noun /
    # citation / quote, via _sentence_has_hard_concrete), first-person authorial framing
    # (closed-class pronouns), or a situational/temporal/conditional clause (closed-class
    # subordinators) -- plus any content-DERIVED domain anchors. No hardcoded content words.
    extra = [re.compile(p, re.I) for p in (domain_patterns or [])]

    detail_hits = 0
    for sentence in sentences:
        if (
            _sentence_has_hard_concrete(sentence)
            or _LIVED_FIRST_PERSON_RE.search(sentence)
            or _SITUATIONAL_CLAUSE_RE.search(sentence)
            or any(p.search(sentence) for p in extra)
        ):
            detail_hits += 1

    ratio = detail_hits / len(sentences)

    if ratio >= 0.45:
        return 0.20
    if ratio >= 0.30:
        return 0.35
    if ratio >= 0.20:
        return 0.50
    if ratio >= 0.10:
        return 0.65
    return 0.80


# Hedging = closed-class modal adverbs + STRUCTURAL support markers (citations, quotes).
# Lexical content hedges ("some researchers/scholars", "cited", "noted") removed (NO-HARDCODE).
HEDGING_PATTERNS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bperhaps\b", r"\bpossibly\b",
    r"\bit seems\b", r"\bit appears\b",
    r"\(.*\d{4}.*\)",  # citation (Smith, 2020) -- structural support
    r'["""]',          # direct quote -- structural support
]

# Assertion = closed-class copulas + modal/auxiliary verbs only. Lexical content verbs
# (creates/makes/provides/requires/enables/forces/threatens) removed (NO-HARDCODE).
ASSERTION_VERB_PATTERNS = [
    r"\bis\b", r"\bare\b", r"\bwas\b", r"\bwere\b", r"\bbe\b", r"\bbeen\b",
    r"\bhas\b", r"\bhave\b", r"\bhad\b",
    r"\bwill\b", r"\bcan\b", r"\bshould\b", r"\bmust\b", r"\bneeds?\b",
]

# "Author-owned context": STRUCTURAL first-hand framing + concrete specifics only -- no
# content-verb lists. Closed-class first-person pronouns + alphanumeric codes + discourse markers.
AUTHOR_OWNED_CONTEXT_PATTERNS = [
    r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b",                 # alphanumeric codes (structural)
    r"\bin my\b",                                          # first-hand framing
    r"\b(?:I|we|my|our)\b",                                # first-person pronouns (closed class)
    r"\b(?:for example|for instance|such as|in particular)\b",  # exemplification (discourse)
]


def _has_author_owned_context(sentence: str) -> bool:
    return any(re.search(pattern, sentence or "", flags=re.I) for pattern in AUTHOR_OWNED_CONTEXT_PATTERNS)


def estimate_broad_claim_risk(text: str) -> float:
    """Higher = more broad/generic claims without concrete support.

    Measures assertion-to-hedging ratio: high assertions + low hedging = broad claims.
    """
    sentences = split_sentences(text)
    if not sentences:
        return 0.70

    assert_count = 0
    for sentence in sentences:
        lower = sentence.lower()
        has_assertion = any(re.search(p, lower) for p in ASSERTION_VERB_PATTERNS)
        has_hedge = any(re.search(p, lower) for p in HEDGING_PATTERNS)
        if has_assertion and not has_hedge:
            assert_count += 1

    ratio = assert_count / len(sentences)
    if ratio >= 0.65:
        return 0.85
    if ratio >= 0.50:
        return 0.75
    if ratio >= 0.40:
        return 0.65
    if ratio >= 0.30:
        return 0.50
    if ratio >= 0.20:
        return 0.35
    return 0.15


def estimate_qualifying_text_ai_density(
    text: str,
    *,
    predictability: Optional[float] = None,
    topk_pattern: Optional[float] = None,
    generic_assertion_risk: Optional[float] = None,
    broad_claim_risk: Optional[float] = None,
    unsupported_claim_risk: Optional[float] = None,
    source_grounding_strength: Optional[float] = None,
    lived_detail_risk: Optional[float] = None,
    formulaic_progression_risk: Optional[float] = None,
    balanced_framing_risk: Optional[float] = None,
) -> float:
    """Estimate how much qualifying long-form text carries aligned AI-style signals.

    This is a document-level density signal. It should not fire on short notes,
    references, bullets, or isolated citation/similarity issues. It looks for a
    long-form body where moderate mechanical predictability combines with broad
    assertions, weak grounding, and low author/process detail.
    """
    words = text.split()
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)
    if len(words) < 300 or len(sentences) < 12:
        return 0.0

    qualifying_sentences = []
    for sentence in sentences:
        stripped = sentence.strip()
        word_count = len(stripped.split())
        lower = stripped.lower()
        if word_count < 8:
            continue
        if lower.startswith(("http", "www.")):
            continue
        if re.match(r"^\s*(references|bibliography|works cited)\b", lower):
            continue
        if stripped.startswith(("-", "*", "•")):
            continue
        if stripped.count('"') >= 2 or stripped.count("'") >= 2:
            continue
        qualifying_sentences.append(stripped)

    qualifying_ratio = len(qualifying_sentences) / max(len(sentences), 1)
    long_form_ratio = sum(len(p.split()) >= 45 and len(split_sentences(p)) >= 2 for p in paragraphs) / max(len(paragraphs), 1)
    if len(qualifying_sentences) < 10 or qualifying_ratio < 0.55:
        return 0.0

    starter_hits = 0
    for sentence in qualifying_sentences:
        lower = sentence.lower()
        if any(lower.startswith(starter) for starter in GENERIC_ESSAY_STARTERS):
            starter_hits += 1
    starter_density = starter_hits / max(len(qualifying_sentences), 1)
    contextual_density = _contextual_anchor_density(qualifying_sentences)

    topk = clamp(topk_pattern)
    pred = clamp(predictability)
    generic_assertion = clamp(generic_assertion_risk)
    broad = clamp(broad_claim_risk)
    lived_gap = clamp(lived_detail_risk)
    formulaic = clamp(formulaic_progression_risk, default=estimate_formulaic_progression_risk(text))
    balanced = clamp(balanced_framing_risk, default=estimate_balanced_generic_framing_risk(text))

    score = weighted_average({
        "topk_pattern": (topk, 0.24),
        "predictability": (pred, 0.14),
        "generic_assertion_risk": (generic_assertion, 0.20),
        "broad_claim_risk": (broad, 0.14),
        "lived_detail_risk": (lived_gap, 0.10),
        "formulaic_progression": (formulaic, 0.08),
        "balanced_framing": (balanced, 0.06),
        "starter_density": (min(1.0, starter_density * 4.0), 0.02),
    })

    if long_form_ratio >= 0.55:
        score += 0.04
    if qualifying_ratio >= 0.75:
        score += 0.03
    if topk >= 0.60 and generic_assertion >= 0.80:
        score += 0.05
    if contextual_density >= 0.45:
        score -= 0.10
    elif contextual_density >= 0.30:
        score -= 0.06
    elif contextual_density >= 0.20:
        score -= 0.03
    return round(clamp(score), 4)


def estimate_unsupported_claim_risk(
    text: str,
    has_citations: bool = False,
) -> float:
    """Higher = more claims without supporting evidence/citations.

    Measures: assertive sentences that lack citations, quotes, or hedging.
    When text has no citations, unsupported claims are likely high.
    """
    if has_citations:
        return 0.20

    sentences = split_sentences(text)
    if not sentences:
        return 0.70

    # Count assertive sentences, but discount claims grounded in local author/process context.
    assert_count = 0.0
    contextual_assertions = 0
    raw_assertions = 0
    for sentence in sentences:
        lower = sentence.lower()
        has_assertion = any(re.search(p, lower) for p in ASSERTION_VERB_PATTERNS)
        has_hedge = any(re.search(p, lower) for p in HEDGING_PATTERNS)
        if has_assertion and not has_hedge:
            raw_assertions += 1
            if _has_author_owned_context(sentence):
                contextual_assertions += 1
                assert_count += 0.20
            else:
                assert_count += 1.0

    ratio = assert_count / len(sentences)
    # No citations at all — escalate thresholds
    if ratio >= 0.55:
        risk = 0.90
    elif ratio >= 0.40:
        risk = 0.80
    elif ratio >= 0.30:
        risk = 0.70
    elif ratio >= 0.20:
        risk = 0.55
    elif ratio >= 0.10:
        risk = 0.40
    else:
        risk = 0.25

    contextual_ratio = contextual_assertions / max(raw_assertions, 1)
    if contextual_ratio >= 0.35 and ratio < 0.28:
        risk = min(risk, 0.40)
    return risk


class Layer3Scorer:
    def calculate_ai_likelihood(self, data: Layer3Input) -> tuple[ClusterScore, float, Optional[str]]:
        """Phase 1: AI Generation Likelihood — pure mechanical signals, zero grounding credit."""
        predictability = clamp(data.predictability)
        raw_topk = clamp(data.topk_pattern)
        genericity = clamp(data.generic_phrase_density)
        burstiness = clamp(data.burstiness_risk)
        repeated_struct = clamp(data.repeated_sentence_structure_risk)
        generic_assertion = clamp(data.generic_assertion_risk)
        qualifying_density = clamp(data.qualifying_text_ai_density)
        balanced_hedging = clamp(data.balanced_hedging_risk)
        fragmentation = clamp(data.sentence_fragmentation_risk)

        genericity_support = clamp(genericity / 0.12) if genericity > 0 else 0.0
        topk_support = max(
            predictability,
            genericity_support,
            repeated_struct,
            generic_assertion,
            qualifying_density,
            balanced_hedging,
        )
        # Treat a truly empty doc (word_count == 0) as the tiniest sample too — the former
        # '0 <' lower bound skipped it (L7). Effect is nil in practice (an empty doc already
        # scores 0) but it removes the boundary inconsistency.
        sample_tiny = (
            (data.word_count < 50)
            or (data.sentence_count < 3)
        )
        sample_limited = sample_tiny or (
            (data.word_count < 150)
            or (data.sentence_count < 6)
        )

        if raw_topk <= 0:
            topk = 0.0
        elif sample_tiny:
            topk = min(raw_topk * 0.25, 0.22)
        elif sample_limited:
            topk = min(raw_topk * 0.40, 0.30)
        elif topk_support < 0.25:
            topk = min(raw_topk * 0.35, 0.31)
        elif topk_support < 0.45:
            topk = raw_topk * 0.65
        else:
            topk = raw_topk

        AI_WEIGHTS = {
            "predictability": 0.18,
            "topk_pattern": 0.14,
            "generic_phrase_density": 0.10,
            "burstiness_risk": 0.08,
            "repeated_sentence_structure_risk": 0.07,
            "generic_assertion_risk": 0.14,
            "qualifying_text_ai_density": 0.24,
            "balanced_hedging_risk": 0.05,
        }

        components = {
            "predictability": predictability,
            "topk_pattern": topk,
            "generic_phrase_density": genericity,
            "burstiness_risk": burstiness,
            "repeated_sentence_structure_risk": repeated_struct,
            "generic_assertion_risk": generic_assertion,
            "qualifying_text_ai_density": qualifying_density,
            "balanced_hedging_risk": balanced_hedging,
        }

        # NOTE (M4 — deliberate, NOT pending cleanup): two components here
        # (repeated_sentence_structure_risk, balanced_hedging_risk) are currently permanent
        # 0.0 stubs (disabled generator / empty starter list). They stay in the denominator
        # (~0.12 of weight), DEFLATING every ai_likelihood_score by ~0.88. This is RETAINED
        # ON PURPOSE: the AMBER/ORANGE/RED boundaries and the validated human mean (~31% on
        # SCoCESLE) were calibrated WITH this deflation. Removing the dead weight is a uniform
        # ~x1.14 scale-up that does NOT improve AUC (ranking is unchanged) but shifts humans
        # UP against fixed boundaries -> more false-accusation, the exact failure the product
        # avoids. Only remove these together with a corpus-gated boundary recalibration; the
        # under-flag is the safe direction.
        score = weighted_average({
            k: (v, AI_WEIGHTS[k])
            for k, v in components.items()
            if getattr(data, k, None) is not None
        })

        cluster_boost = 0.0
        cluster_name = None

        # UNREACHABLE while the M4 stubs above (repeated_sentence_structure_risk,
        # balanced_hedging_risk) stay hardcoded at 0.0: `repeated_struct` and
        # `balanced_hedging` can never satisfy a positive threshold, so
        # `known_ai_style`, `humanised_ai`, and `template_ai_style` below can
        # never evaluate True, and template_ai_style's score floor/ceiling
        # (~:1251-1254 in the current layout) never applies either. Kept
        # in place deliberately (not dead-code-deleted) so the boosts activate
        # automatically once the stub generators are built — do not remove
        # without re-checking whether the generators have shipped. Only
        # `qualifying_density_ai` and `fragmented_evasion` are live today.
        known_ai_style = (
            predictability >= 0.42
            and genericity >= 0.08
            and repeated_struct >= 0.45
            and balanced_hedging >= 0.35
        )

        humanised_ai = (
            predictability >= 0.40
            and burstiness >= 0.55
            and generic_assertion >= 0.55
            and balanced_hedging >= 0.30
        )

        qualifying_density_ai = (
            qualifying_density >= 0.70
            and topk >= 0.60
            and generic_assertion >= 0.80
        )

        # UNREACHABLE — see the comment above known_ai_style; gates on the
        # same permanently-zeroed repeated_struct stub.
        template_ai_style = (
            predictability >= 0.42
            and topk >= 0.75
            and repeated_struct >= 0.55
            and generic_assertion >= 0.75
        )

        # Mechanical sentence-splitting evasion: strongly generic AI content chopped into uniform
        # short sentences, which zeroes the structural signals and drags the weighted average to
        # "green". Gate on the precise fragmentation detector + high generic-assertion (it is generic
        # AI, not original human writing) + collapsed qualifying-text density (the starvation tell).
        # Note: topk arrives CALIBRATED-down in the badge path (~0.44, not raw ~0.79), so it is not a
        # reliable gate here; the fragmentation detector is the precision gate instead.
        fragmented_evasion = (
            fragmentation >= 0.45
            and generic_assertion >= 0.70
            and qualifying_density < 0.40
        )

        if qualifying_density_ai:
            cluster_boost = 0.12
            cluster_name = "qualifying_text_ai_density"
        elif template_ai_style:
            cluster_boost = 0.12
            cluster_name = "template_ai_style"
        elif known_ai_style:
            cluster_boost = 0.10
            cluster_name = "known_ai_style"
        elif humanised_ai:
            cluster_boost = 0.08
            cluster_name = "humanised_ai"
        elif fragmented_evasion:
            cluster_boost = 0.10
            cluster_name = "fragmented_evasion"

        score = clamp(score + cluster_boost)
        if qualifying_density_ai:
            score = max(score, 0.60)
        if template_ai_style:
            score = max(score, 0.58)
        if template_ai_style and not qualifying_density_ai and qualifying_density < 0.55:
            score = min(score, 0.64)
        if fragmented_evasion:
            # Floor into AMBER so mechanically-fragmented AI content cannot read as clean/green.
            # Conservative: the content signals are extreme, but we do not jump straight to RED.
            score = max(score, 0.42)
        if sample_tiny:
            score = min(score, 0.24)
        elif sample_limited:
            score = min(score, 0.32)

        if data.human_provenance_positive:
            score *= 0.80

        score = clamp(score)

        available = sum(1 for v in [
            data.predictability, data.topk_pattern, data.generic_phrase_density,
            data.burstiness_risk, data.repeated_sentence_structure_risk,
            data.generic_assertion_risk, data.qualifying_text_ai_density,
            data.balanced_hedging_risk,
        ] if v is not None)

        confidence = confidence_from_coverage(
            word_count=data.word_count,
            sentence_count=data.sentence_count,
            available_count=available,
            expected_count=8,
        )

        reasons = []
        if score >= 0.60:
            reasons.append("high_ai_likelihood")
        elif score >= 0.40:
            reasons.append("moderate_ai_likelihood")
        if cluster_name:
            reasons.append(f"{cluster_name}_cluster")
        if raw_topk and topk < raw_topk - 0.05:
            reasons.append("topk_requires_supporting_ai_signals")
        if sample_limited:
            reasons.append("sample_length_limited")
        if data.human_provenance_positive:
            reasons.append("human_provenance_positive")

        return ClusterScore(
            name="ai_likelihood",
            score=round(score, 4),
            confidence=confidence,
            components={k: round(v, 4) for k, v in components.items()},
            reasons=reasons,
        ), cluster_boost, cluster_name

    def calculate_writing_quality(self, data: Layer3Input) -> tuple[ClusterScore, QualityTier]:
        """Phase 2: Writing Quality Risk — evidence + structure signals, grounding credit applies."""
        broad_claim = clamp(data.broad_claim_risk)
        lived_detail = clamp(data.lived_detail_risk)
        citation = clamp(data.citation_weakness_risk)
        unsupported = clamp(data.unsupported_claim_risk)
        source_risk = 1.0 - clamp(data.source_grounding_strength)
        progression = clamp(data.paragraph_progression_risk)
        signpost = clamp(data.signpost_paragraph_risk)
        uniformity = clamp(data.paragraph_uniformity_risk)
        starter = clamp(data.repeated_starter_risk)
        conclusion = clamp(data.formulaic_conclusion_risk)

        QUALITY_WEIGHTS = {
            "broad_claim_risk": 0.15,
            "lived_detail_risk": 0.18,
            "citation_weakness_risk": 0.15,
            "unsupported_claim_risk": 0.12,
            "source_grounding_risk": 0.12,
            "paragraph_progression_risk": 0.08,
            "signpost_paragraph_risk": 0.08,
            "paragraph_uniformity_risk": 0.04,
            "repeated_starter_risk": 0.04,
            "formulaic_conclusion_risk": 0.04,
        }

        components = {
            "broad_claim_risk": broad_claim,
            "lived_detail_risk": lived_detail,
            "citation_weakness_risk": citation,
            "unsupported_claim_risk": unsupported,
            "source_grounding_risk": source_risk,
            "paragraph_progression_risk": progression,
            "signpost_paragraph_risk": signpost,
            "paragraph_uniformity_risk": uniformity,
            "repeated_starter_risk": starter,
            "formulaic_conclusion_risk": conclusion,
        }

        # Keep GENUINE zeros in the denominator (filter None, not >0), matching the AI
        # phase's is-not-None filter. Dropping zeros inflated the average toward the few
        # non-zero components and over-judged thin/clean docs into HIGH_REVIEW (M5). This
        # only lowers scores (safe direction for human writing).
        score = weighted_average({
            k: (v, QUALITY_WEIGHTS[k])
            for k, v in components.items()
            if v is not None and k in QUALITY_WEIGHTS
        })

        source_strength = clamp(data.source_grounding_strength)
        domain_strength = clamp(data.domain_grounding_strength)
        grounding_credit = min(0.15, 0.08 * source_strength + 0.07 * domain_strength)
        score = clamp(score * (1.0 - grounding_credit))

        components["source_grounding_strength"] = source_strength
        components["domain_grounding_strength"] = domain_strength
        components["grounding_credit"] = grounding_credit

        if score >= 0.65:
            quality_tier = QualityTier.HIGH_REVIEW
        elif score >= 0.45:
            quality_tier = QualityTier.REVIEW
        elif score >= 0.30:
            quality_tier = QualityTier.LIGHT_REVIEW
        else:
            quality_tier = QualityTier.LOW

        available = sum(1 for v in [
            data.broad_claim_risk, data.lived_detail_risk,
            data.citation_weakness_risk, data.unsupported_claim_risk,
        ] if v is not None)

        confidence = confidence_from_coverage(
            word_count=data.word_count,
            sentence_count=data.sentence_count,
            available_count=available,
            expected_count=4,
        )

        reasons = []
        if score >= 0.65:
            reasons.append("high_quality_risk")
        elif score >= 0.45:
            reasons.append("moderate_quality_risk")
        if lived_detail >= 0.65:
            reasons.append("weak_lived_detail")
        if citation >= 0.65:
            reasons.append("citation_weakness")

        return ClusterScore(
            name="writing_quality",
            score=round(score, 4),
            confidence=confidence,
            components={k: round(v, 4) for k, v in components.items()},
            reasons=reasons,
        ), quality_tier

    # NOT the single source of truth for these three numbers — they are
    # duplicated (same values) in poc/detect_v7/weights.json's tier_authority
    # cutoffs, gate-validated separately on the FUSED scale ("A cutoff sweep
    # ON THE FUSED SCALE confirmed the EXISTING tier cutoffs 32/48/65 carry
    # over unchanged", see that file's tier_authority._notes). This copy
    # gates the PRE-FUSION Layer3 score specifically (badge tier when V7
    # tier-authority is off, or as the fallback). If either path is ever
    # re-tuned, check whether the other still holds — there is no code link
    # forcing them to move together.
    def _derive_ai_tier(self, ai_score: float) -> Tier:
        if ai_score >= 0.65:
            return Tier.RED
        elif ai_score >= 0.48:
            return Tier.ORANGE
        elif ai_score >= 0.32:
            return Tier.AMBER
        else:
            return Tier.GREEN

    # Width (in score units) below each cutoff that counts as "near boundary" — unstable,
    # easily flipped by small input changes. Tuned to 0.07 (e.g. a GREEN score of 0.25-0.32
    # is one re-scan away from AMBER). MUST stay < the smallest gap between cutoffs (0.16)
    # so the band never spans two tiers.
    _BOUNDARY_BAND = 0.07
    _TIER_CUTOFFS = (0.32, 0.48, 0.65)  # kept in sync with _derive_ai_tier's literals above

    def _near_boundary(self, ai_score: float) -> bool:
        return any(c - self._BOUNDARY_BAND <= ai_score < c for c in self._TIER_CUTOFFS)

    def _derive_review_priority(
        self,
        ai_tier: Tier,
        quality_tier: QualityTier,
        quality_score: float,
    ) -> str:
        if ai_tier == Tier.RED:
            return "high_ai_concern"
        if ai_tier == Tier.ORANGE and quality_tier in (QualityTier.REVIEW, QualityTier.HIGH_REVIEW):
            return "serious_review"
        if ai_tier == Tier.ORANGE:
            return "ai_review"
        if ai_tier == Tier.AMBER and quality_score >= 0.65:
            return "possible_humanised_ai"
        if ai_tier == Tier.AMBER and quality_tier in (QualityTier.LIGHT_REVIEW, QualityTier.REVIEW, QualityTier.HIGH_REVIEW):
            return "strong_review"
        if ai_tier == Tier.AMBER:
            return "mild_review"
        if quality_tier in (QualityTier.REVIEW, QualityTier.HIGH_REVIEW):
            return "writing_review"
        if quality_tier == QualityTier.LIGHT_REVIEW:
            return "note"
        return "clean"

    def score(self, data: Layer3Input) -> Layer3Result:
        """Run two-phase scoring: AI Likelihood + Writing Quality."""
        ai_phase, cluster_boost, cluster_name = self.calculate_ai_likelihood(data)
        writing_phase, quality_tier = self.calculate_writing_quality(data)

        if data.verified_ai_provenance:
            ai_tier = Tier.RED
        else:
            ai_tier = self._derive_ai_tier(ai_phase.score)

        review_priority = self._derive_review_priority(ai_tier, quality_tier, writing_phase.score)
        confidence = merge_confidence(ai_phase.confidence, writing_phase.confidence)
        reasons = list(ai_phase.reasons) + list(writing_phase.reasons)
        authorship_rating = derive_authorship_rating(
            ai_score=ai_phase.score,
            ai_tier=ai_tier,
            writing_quality_score=writing_phase.score,
            writing_quality_tier=quality_tier,
            confidence=confidence,
            verified_ai_provenance=data.verified_ai_provenance,
            human_provenance_positive=data.human_provenance_positive,
            ai_components=ai_phase.components,
            writing_components=writing_phase.components,
        )

        guardrails = []
        if data.human_provenance_positive:
            guardrails.append("human_provenance_positive_considered")
        if data.verified_ai_provenance:
            guardrails.append("verified_ai_provenance_override")

        # Display-only reliability flag: near a tier boundary OR thin sample. Never alters the
        # tier/score (see Layer3Result.verdict_low_confidence docstring).
        verdict_low_confidence = bool(
            self._near_boundary(ai_phase.score)
            or data.word_count < 150
            or data.sentence_count < 6
        )

        return Layer3Result(
            tier=ai_tier,
            ai_likelihood_score=ai_phase.score,
            authorship_rating=authorship_rating,
            ai_cluster_boost=cluster_boost,
            ai_cluster_name=cluster_name,
            ai_phase=ai_phase,
            writing_quality_score=writing_phase.score,
            writing_quality_tier=quality_tier,
            writing_phase=writing_phase,
            confidence=confidence,
            reasons=reasons,
            guardrails=guardrails,
            review_priority=review_priority,
            verdict_low_confidence=verdict_low_confidence,
            debug={
                "ai_likelihood_raw": ai_phase.score,
                "writing_quality_raw": writing_phase.score,
            },
        )


def build_layer3_input_from_text(
    text: str,
    *,
    predictability: Optional[float] = None,
    topk_pattern: Optional[float] = None,
    generic_phrase_density: Optional[float] = None,
    broad_claim_risk: Optional[float] = None,
    citation_weakness_risk: Optional[float] = None,
    unsupported_claim_risk: Optional[float] = None,
    source_grounding_strength: Optional[float] = None,
    domain_grounding_strength: Optional[float] = None,
    semantic_uniformity_risk: Optional[float] = None,
    discourse_regularity_risk: Optional[float] = None,
    semantic_drift_risk: Optional[float] = None,
    human_provenance_positive: bool = False,
    verified_ai_provenance: bool = False,
    domain_lived_detail_patterns: Optional[list[str]] = None,
) -> Layer3Input:
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)

    formulaic_progression = estimate_formulaic_progression_risk(text)
    balanced_framing = estimate_balanced_generic_framing_risk(text)

    # Auto-compute broad_claim_risk from text if not provided
    if broad_claim_risk is None:
        broad_claim_risk = estimate_broad_claim_risk(text)

    # Auto-compute unsupported_claim_risk from text if not provided
    has_cites = citation_weakness_risk is not None and citation_weakness_risk < 0.50
    if unsupported_claim_risk is None:
        unsupported_claim_risk = estimate_unsupported_claim_risk(text, has_citations=has_cites)

    generic_assertion = estimate_generic_assertion_risk(text)
    lived_detail = estimate_lived_detail_risk(text, domain_lived_detail_patterns)
    qualifying_density = estimate_qualifying_text_ai_density(
        text,
        predictability=predictability,
        topk_pattern=topk_pattern,
        generic_assertion_risk=generic_assertion,
        broad_claim_risk=broad_claim_risk,
        unsupported_claim_risk=unsupported_claim_risk,
        source_grounding_strength=source_grounding_strength,
        lived_detail_risk=lived_detail,
        formulaic_progression_risk=formulaic_progression,
        balanced_framing_risk=balanced_framing,
    )

    return Layer3Input(
        predictability=predictability,
        topk_pattern=topk_pattern,
        generic_phrase_density=generic_phrase_density,
        burstiness_risk=estimate_burstiness_risk(text),
        repeated_sentence_structure_risk=estimate_repeated_sentence_structure_risk(text),
        sentence_fragmentation_risk=estimate_sentence_fragmentation_risk(text),
        generic_assertion_risk=generic_assertion,
        qualifying_text_ai_density=qualifying_density,

        broad_claim_risk=broad_claim_risk,
        lived_detail_risk=lived_detail,
        citation_weakness_risk=citation_weakness_risk,
        unsupported_claim_risk=unsupported_claim_risk,
        source_grounding_strength=source_grounding_strength,
        domain_grounding_strength=domain_grounding_strength,

        paragraph_progression_risk=max(
            formulaic_progression,
            clamp(discourse_regularity_risk),
        ),
        paragraph_uniformity_risk=estimate_paragraph_uniformity_risk(text),
        repeated_starter_risk=estimate_repeated_starter_risk(text),
        formulaic_conclusion_risk=estimate_formulaic_conclusion_risk(text),
        signpost_paragraph_risk=estimate_signpost_paragraph_risk(text),
        balanced_hedging_risk=estimate_balanced_hedging_risk(text),
        intra_paragraph_parallelism_risk=estimate_intra_paragraph_parallelism_risk(text),
        paragraph_topic_uniformity_risk=max(
            estimate_paragraph_topic_uniformity_risk(text),
            clamp(semantic_uniformity_risk),
        ),

        # In a production codebase, rename this field to balanced_framing_risk.
        style_shift_risk=max(balanced_framing, clamp(semantic_drift_risk) * 0.65),

        human_provenance_positive=human_provenance_positive,
        verified_ai_provenance=verified_ai_provenance,

        word_count=len(text.split()),
        sentence_count=len(sentences),
        paragraph_count=len(paragraphs),
    )
