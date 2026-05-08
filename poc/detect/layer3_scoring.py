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
            "label": "Unlikely AI",
            "short_label": "Unlikely AI",
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


FORMULAIC_PROGRESS_MARKERS = [
    r"\bin the past\b",
    r"\bnow\b",
    r"\btoday\b",
    r"\bthis shift\b",
    r"\bhowever\b",
    r"\btechnology has also\b",
    r"\banother issue\b",
    r"\bbecause of this\b",
    r"\bthe goal should\b",
    r"\bin conclusion\b",
    r"\boverall\b",
    r"\bultimately\b",
]


BALANCED_FRAMING_PATTERNS = [
    r"\bnot only\b.+?\bbut also\b",
    r"\bboth\b.+?\band\b",
    r"\beither\b.+?\bor\b",
    r"\bopportunities and risks\b",
    r"\bimportant,?\s+not\s+less\s+important\b",
    r"\bthe goal should not be\b.+?\bthe goal should be\b",
    r"\bdoes not only\b.+?\bit also\b",
    r"\bno longer\b.+?\binstead\b",
]


GENERIC_ESSAY_STARTERS = [
    "in the past",
    "now",
    "today",
    "however",
    "another issue",
    "because of this",
    "this shift",
    "technology has also",
    "the real challenge",
    "the goal should",
    "in other words",
    "in conclusion",
    "ultimately",
]


def estimate_formulaic_progression_risk(text: str) -> float:
    lower = text.lower()
    hits = sum(1 for pattern in FORMULAIC_PROGRESS_MARKERS if re.search(pattern, lower))
    ratio = hits / len(FORMULAIC_PROGRESS_MARKERS)

    if ratio >= 0.65:
        return 0.85
    if ratio >= 0.45:
        return 0.65
    if ratio >= 0.30:
        return 0.45
    if ratio >= 0.15:
        return 0.25
    return 0.0


def estimate_balanced_generic_framing_risk(text: str) -> float:
    lower = text.lower()
    hits = sum(1 for pattern in BALANCED_FRAMING_PATTERNS if re.search(pattern, lower))

    if hits >= 4:
        return 0.85
    if hits == 3:
        return 0.65
    if hits == 2:
        return 0.45
    if hits == 1:
        return 0.25
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


# Signpost/bridge paragraphs: very short (1-2 sentence) paragraphs that
# merely announce or transition to the next topic. AI essays over-use these.
SIGNPOST_ANNOUNCE_PATTERNS = [
    r"^(?:this|another|one|the|a)\s+(?:key|main|major|important|critical|significant|biggest)",
    r"^(?:there are|there is|this (?:brings|raises|highlights|shows|demonstrates))",
    r"^(?:let(?:'s|\s+us)\s+(?:now|consider|examine|look|turn|explore))",
    r"^(?:moving (?:on|forward|beyond)|having (?:established|discussed|explored))",
    r"^(?:it is (?:also|important|clear|worth|essential)|it (?:should|must|also))",
    r"^(?:beyond|above|alongside|in addition to)\s",
    r"^(?:we (?:can|must|should|now|also|first|then))",
    r"^(?:the (?:next|following|second|third|final|last))",
    r"^(?:what (?:this|these|it))",
    r"^(?:to (?:understand|address|tackle|solve|appreciate|fully))",
]


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

        # Signpost = short paragraph (≤ 2 sentences, ≤ 40% of median length)
        if sent_count <= 2 and word_count < median_len * 0.45:
            # Check if it reads like an announcement/transition
            first_sentence = re.split(r'[.!?]+', p)[0].strip().lower()
            is_announce = any(
                re.search(pattern, first_sentence)
                for pattern in SIGNPOST_ANNOUNCE_PATTERNS
            )
            if is_announce:
                signpost_count += 1
                continue
            # Even without explicit announcement pattern, a cluster of
            # very short paragraphs between long ones is suspicious
            if word_count < median_len * 0.25:
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


BALANCED_HEDGING_PATTERNS = [
    r'\bnot only\b.+?\bbut\s+also\b',
    r'\bwhile\b.{5,40}?\b(?:important|necessary|essential|crucial|vital)\b.{5,30}?\balso\b',
    r'\balthough\b.{5,40}?\b(?:important|necessary|essential|crucial|vital)\b',
    r'\bon\s+the\s+one\s+hand\b.+?\bon\s+the\s+other\b',
    r'\bit\s+is\s+(?:important|essential|crucial)\s+to\s+note\b',
    r'\bplays?\s+a\s+(?:vital|crucial|key|important|central)\s+role\b',
    r'\bthere\s+is\s+no\s+(?:doubt|question|denying)\b',
    r'\bcannot\s+be\s+(?:overstated|ignored|overlooked|underestimated)\b',
    r'\bit\s+is\s+worth\s+noting\b',
    r'\bhas\s+its\s+(?:merits|benefits|strengths)\b.{5,30}?\bbut\b',
    r'\bdespite\s+(?:these?\s+)?(?:challenges?|concerns?|risks?|issues?)\b',
    r'\bthe\s+(?:key|main|primary)\s+(?:challenge|issue|concern)\s+is\b',
    r'\bboth\s+(?:sides?|perspectives?|approaches?|viewpoints?)\b',
]


def estimate_balanced_hedging_risk(text: str) -> float:
    """Higher = text uses many AI-typical balanced/hedging constructions.

    AI essays over-use balanced constructions: "While X is important, Y should
    also be considered", "not only A but also B", "plays a crucial role".
    These create a distinctive even-handed, tempered tone that's uncommon
    in human first-draft writing.
    """
    lower = text.lower()
    hits = sum(1 for p in BALANCED_HEDGING_PATTERNS if re.search(p, lower))

    if hits >= 5:
        return 0.90
    if hits >= 4:
        return 0.75
    if hits == 3:
        return 0.55
    if hits == 2:
        return 0.35
    if hits == 1:
        return 0.15
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
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return 0.0

    last = paragraphs[-1].lower()
    markers = [
        "stands at a turning point",
        "the goal should",
        "not be to",
        "the goal should be",
        "thoughtful",
        "capable",
        "responsible",
        "in a world full of",
        "ultimately",
        "in conclusion",
    ]

    hits = sum(1 for marker in markers if marker in last)

    if hits >= 4:
        return 0.85
    if hits == 3:
        return 0.65
    if hits == 2:
        return 0.45
    if hits == 1:
        return 0.25
    return 0.0


CONCRETE_DETAIL_PATTERNS = [
    r"\b\d+\.?\d*\s*%?\b",  # numbers/percentages
    r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b",  # unit/intake/course codes
    r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b",  # named entities (rough)
    r"\b[A-Z][A-Za-z]+(?:\s+et\s+al\.)?\s*\(\d{4}\)",  # named source citation
    r"\([A-Z][A-Za-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)",  # parenthetical citation
    r"\bduring\b.*\b\d{4}\b",  # specific time references
    r"\bin my\b", r"\bI (?:saw|see|noticed|found|observed|taught|experienced|usually|ask|demonstrate|want|encourage)\b",
    r"\bI do not\b", r"\bI have seen\b", r"\bmy current\b",
    r"\bwe (?:found|observed|measured|tested)\b",
    r"\bspecifically\b", r"\bfor example\b", r"\bfor instance\b",
    r"\bcase study\b", r"\bdata (?:show|from|set)\b",
    r"\baccording to\b", r'\b["""].+?["""]\b',  # quotes
    r"\b(?:drafts?|homework|exams?|lesson notes?|classroom discussion|feedback activity)\b",
    r"\b(?:YouTube|TikTok|AI tools?|search engines?|online courses?|social media|websites?)\b",
    r"\b(?:source checks?|peer communities|discussion|reflection|practice|attempts?)\b",
    r"\b(?:sectioning|projection|parting|distribution|design line|mobile guide|stationary guide|guide line|"
    r"mannequin|client model|client models|weight line|elbow|wrist|finger|scissor|comb|tension|subsection|"
    r"graduation|graduated haircut|solid form|one length|uniform layer|increased layer|haircut structures?)\b",
]
NAMED_ENTITY_DETAIL_PATTERN = CONCRETE_DETAIL_PATTERNS[2]
CONTEXTUAL_ANCHOR_PATTERNS = [
    r"\bonline content\b",
    r"\bfinal result\b",
    r"\bhiding the effort\b",
    r"\bwhat to trust\b",
    r"\bsource(?:s)?\s+(?:they\s+)?trusted\b",
    r"\bpass/fail\b",
    r"\bdrafts?\s+and\s+feedback\b",
    r"\bfeedback routines?\b",
    r"\bafter reflection\b",
    r"\breal situations?\b",
    r"\bconfusion\b",
    r"\bstudents?\s+(?:explain|search|write|practice|understand|learn|think)\b",
    r"\blearners?\s+(?:shift|practise|practice|repeat|name|combine|cut|pause|continue|describe|notice|need|work)\b",
    r"\bteachers?\s+(?:explain|notice|ask|help|guide)\b",
    r"\beducators?\s+(?:spot|watch|demonstrate|correct|guide)\b",
    r"\b(?:comb tension|elbow height|section size|scissor control|guide awareness)\b",
]


def _sentence_has_concrete_or_context(sentence: str) -> bool:
    return any(
        re.search(p, sentence, flags=0 if p == NAMED_ENTITY_DETAIL_PATTERN else re.I)
        for p in CONCRETE_DETAIL_PATTERNS
    ) or any(re.search(p, sentence, flags=re.I) for p in CONTEXTUAL_ANCHOR_PATTERNS)


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


def estimate_lived_detail_risk(text: str, domain_patterns: Optional[list[str]] = None) -> float:
    sentences = split_sentences(text)
    if not sentences:
        return 0.70

    base_patterns = [
        r"\b\d+\b",
        r"\bduring\b",
        r"\bwhen\b",
        r"\bafter\b",
        r"\bbefore\b",
        r"\bin my classroom\b",
        r"\bi have seen\b",
        r"\bi noticed\b",
        r"\bi would\b",
        r"\bi think\b",
        r"\bi worry\b",
        r"\bi treat\b",
        r"\bthe (?:part|issue|point) i\b",
        r"\bwhat (?:i|we) (?:would|need|want)\b",
        r"\bmy judgement\b",
        r"\bwe observed\b",
        r"\bfeedback\b",
        r"\btesting\b",
        r"\bstudent said\b",
        r"\bteacher said\b",
        r"\bschool\b",
        r"\bclassroom\b",
        r"\bcase\b",
        r"\bexample\b",
        r"\bincident\b",
        r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b",
        r"\bBox Hill\b",
        r"\bI (?:see|ask|demonstrate|want|encourage|usually)\b",
        r"\bmy current\b",
        r"\blearners?\s+(?:shift|practise|practice|repeat|name|combine|cut|pause|continue|describe|notice|need|work)\b",
        r"\b(?:sectioning|projection|parting|distribution|design line|mobile guide|stationary guide|guide line|"
        r"guide|mannequin|client model|client models|weight line|elbow|wrist|finger|scissor|comb|tension|subsection|"
        r"graduation|graduated haircut|solid form|one length|uniform layer|increased layer|haircut structures?)\b",
    ]

    patterns = base_patterns + (domain_patterns or [])

    detail_hits = 0
    for sentence in sentences:
        lower = sentence.lower()
        if any(re.search(pattern, lower) for pattern in patterns):
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


HEDGING_PATTERNS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bperhaps\b", r"\bpossibly\b",
    r"\bit seems\b", r"\bit appears\b", r"\bsome (?:researchers|scholars|experts|studies)\b",
    r"\baccording to\b", r"\bcited\b", r"\bnoted\b",
    r"\(.*\d{4}.*\)",  # citations like (Smith, 2020)
    r'["""]',  # direct quotes
]

ASSERTION_VERB_PATTERNS = [
    r"\bis\b", r"\bare\b", r"\bwas\b", r"\bhas\b", r"\bhave\b",
    r"\bwill\b", r"\bcan\b", r"\bshould\b", r"\bmust\b", r"\bneeds?\b",
    r"\bcreates?\b", r"\bmakes?\b", r"\bprovides?\b", r"\brequires?\b",
    r"\benables?\b", r"\bforces?\b", r"\bthreatens?\b",
]

AUTHOR_OWNED_CONTEXT_PATTERNS = [
    r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b",
    r"\b(?:At Box Hill Institute|Certificate III|my current learners)\b",
    r"\bI (?:see|ask|demonstrate|want|encourage|usually|notice|use|treat)\b",
    r"\blearners?\s+(?:shift|practise|practice|repeat|name|combine|cut|pause|continue|describe|notice|need|work|must|can)\b",
    r"\b(?:sectioning|projection|parting|distribution|design line|mobile guide|stationary guide|guide line|"
    r"mannequin|client model|client models|weight line|elbow|wrist|finger|scissor|comb|tension|subsection|"
    r"graduation|graduated haircut|solid form|one length|uniform layer|increased layer|haircut structures?)\b",
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
        topk = clamp(data.topk_pattern)
        genericity = clamp(data.generic_phrase_density)
        burstiness = clamp(data.burstiness_risk)
        repeated_struct = clamp(data.repeated_sentence_structure_risk)
        generic_assertion = clamp(data.generic_assertion_risk)
        qualifying_density = clamp(data.qualifying_text_ai_density)
        balanced_hedging = clamp(data.balanced_hedging_risk)

        AI_WEIGHTS = {
            "predictability": 0.19,
            "topk_pattern": 0.17,
            "generic_phrase_density": 0.13,
            "burstiness_risk": 0.10,
            "repeated_sentence_structure_risk": 0.10,
            "generic_assertion_risk": 0.10,
            "qualifying_text_ai_density": 0.13,
            "balanced_hedging_risk": 0.08,
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

        score = weighted_average({
            k: (v, AI_WEIGHTS[k])
            for k, v in components.items()
            if v > 0
        })

        cluster_boost = 0.0
        cluster_name = None

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

        template_ai_style = (
            predictability >= 0.42
            and topk >= 0.75
            and repeated_struct >= 0.55
            and generic_assertion >= 0.75
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

        score = clamp(score + cluster_boost)
        if qualifying_density_ai:
            score = max(score, 0.60)
        if template_ai_style:
            score = max(score, 0.58)

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

        score = weighted_average({
            k: (v, QUALITY_WEIGHTS[k])
            for k, v in components.items()
            if v > 0
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

    def _derive_ai_tier(self, ai_score: float) -> Tier:
        if ai_score >= 0.65:
            return Tier.RED
        elif ai_score >= 0.48:
            return Tier.ORANGE
        elif ai_score >= 0.32:
            return Tier.AMBER
        else:
            return Tier.GREEN

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
