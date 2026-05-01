"""
DraftProof Layer 3 Scoring Engine
=================================

Purpose:
    Convert lower-level scanner outputs into reliable Layer 3 cluster scores:
    - Text Pattern Cluster
    - Grounding Quality Risk Cluster
    - Structure / Process Cluster
    - Cluster Blended Score
    - Review Tier

Design principles:
    1. Predictability alone cannot create RED.
    2. RED requires aligned evidence across independent clusters, unless provenance is verified.
    3. Missing evidence lowers confidence, not guilt.
    4. Generic phrase detection is conservative to avoid academic false positives.
    5. Formulaic macro-structure is explicitly measured, because AI essays often look generic at the document level.
    6. All scores are normalized 0.0 - 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    blended_score: float
    calibrated_score: float
    confidence: Confidence
    text_pattern: ClusterScore
    grounding_quality: ClusterScore
    structure_process: ClusterScore
    reasons: list[str]
    guardrails: list[str]
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
        from collections import Counter
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
    r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b",  # named entities (rough)
    r"\bduring\b.*\b\d{4}\b",  # specific time references
    r"\bin my\b", r"\bI (?:saw|noticed|found|observed|taught|experienced)\b",
    r"\bwe (?:found|observed|measured|tested)\b",
    r"\bspecifically\b", r"\bfor example\b", r"\bfor instance\b",
    r"\bcase study\b", r"\bdata (?:show|from|set)\b",
    r"\baccording to\b", r'\b["""].+?["""]\b',  # quotes
]


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
        lower = sentence.lower()
        has_concrete = any(re.search(p, lower) for p in CONCRETE_DETAIL_PATTERNS)
        if not has_concrete:
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

    # Count assertive sentences (same logic as broad_claim)
    assert_count = 0
    for sentence in sentences:
        lower = sentence.lower()
        has_assertion = any(re.search(p, lower) for p in ASSERTION_VERB_PATTERNS)
        has_hedge = any(re.search(p, lower) for p in HEDGING_PATTERNS)
        if has_assertion and not has_hedge:
            assert_count += 1

    ratio = assert_count / len(sentences)
    # No citations at all — escalate thresholds
    if ratio >= 0.55:
        return 0.90
    if ratio >= 0.40:
        return 0.80
    if ratio >= 0.30:
        return 0.70
    if ratio >= 0.20:
        return 0.55
    return 0.35
    if ratio >= 0.10:
        return 0.65
    return 0.80


class Layer3Scorer:
    def calculate_text_pattern_cluster(self, data: Layer3Input) -> ClusterScore:
        predictability_risk = threshold_risk(data.predictability, normal=0.35, high=0.55)
        topk_risk = threshold_risk(data.topk_pattern, normal=0.45, high=0.70)
        generic = clamp(data.generic_phrase_density)
        burstiness = clamp(data.burstiness_risk)
        repeated_structure = clamp(data.repeated_sentence_structure_risk)
        formulaic_progression = clamp(data.paragraph_progression_risk)
        balanced_framing = clamp(data.style_shift_risk)
        generic_assertion = clamp(data.generic_assertion_risk)

        components = {
            "predictability_risk": predictability_risk,
            "topk_pattern_risk": topk_risk,
            "generic_phrase_density": generic,
            "burstiness_risk": burstiness,
            "repeated_sentence_structure_risk": repeated_structure,
            "formulaic_progression_risk": formulaic_progression,
            "balanced_generic_framing_risk": balanced_framing,
            "generic_assertion_risk": generic_assertion,
        }

        # Core components — always included
        tp_components = {
            "predictability": (predictability_risk, 0.25),
            "burstiness": (burstiness, 0.10),
            "repeated_structure": (repeated_structure, 0.15),
            "formulaic_progression": (formulaic_progression, 0.20),
            "balanced_framing": (balanced_framing, 0.10),
            "generic_assertion": (generic_assertion, 0.20),
        }

        # Optional — only when they fire (> 0.10)
        tp_optional = []
        if topk_risk > 0.10:
            tp_optional.append(("topk", topk_risk, 0.10))
        if generic > 0.10:
            tp_optional.append(("generic", generic, 0.08))

        if tp_optional:
            opt_w = sum(w for _, _, w in tp_optional)
            tp_components = {k: (v, wt * (1 - opt_w)) for k, (v, wt) in tp_components.items()}
            for name, val, w in tp_optional:
                tp_components[name] = (val, w)

        score = weighted_average(tp_components)

        score = min(score, 0.75)

        available = sum(1 for v in [
            data.predictability,
            data.topk_pattern,
            data.generic_phrase_density,
            data.burstiness_risk,
            data.repeated_sentence_structure_risk,
            data.paragraph_progression_risk,
            data.style_shift_risk,
            data.generic_assertion_risk,
        ] if v is not None)

        confidence = confidence_from_coverage(
            word_count=data.word_count,
            sentence_count=data.sentence_count,
            available_count=available,
            expected_count=8,
        )

        reasons = []
        if score >= 0.60:
            reasons.append("high_text_pattern_cluster")
        elif score >= 0.40:
            reasons.append("moderate_text_pattern_cluster")

        if formulaic_progression >= 0.65:
            reasons.append("formulaic_progression_detected")

        if balanced_framing >= 0.65:
            reasons.append("balanced_generic_framing_detected")

        return ClusterScore(
            name="text_pattern",
            score=round(score, 4),
            confidence=confidence,
            components={k: round(v, 4) for k, v in components.items()},
            reasons=reasons,
        )

    def calculate_grounding_quality_cluster(self, data: Layer3Input) -> ClusterScore:
        broad_claim = clamp(data.broad_claim_risk)
        lived_detail = clamp(data.lived_detail_risk)
        citation = clamp(data.citation_weakness_risk)
        unsupported = clamp(data.unsupported_claim_risk)
        source_strength = clamp(data.source_grounding_strength)
        domain_strength = clamp(data.domain_grounding_strength)

        raw_score = weighted_average({
            "broad_claim": (broad_claim, 0.30),
            "lived_detail": (lived_detail, 0.25),
            "citation": (citation, 0.25),
            "unsupported": (unsupported, 0.20),
        })

        grounding_credit = min(
            0.15,
            0.08 * source_strength + 0.07 * domain_strength,
        )

        score = clamp(raw_score * (1.0 - grounding_credit))

        components = {
            "broad_claim_risk": broad_claim,
            "lived_detail_risk": lived_detail,
            "citation_weakness_risk": citation,
            "unsupported_claim_risk": unsupported,
            "source_grounding_strength": source_strength,
            "domain_grounding_strength": domain_strength,
            "grounding_credit": grounding_credit,
            "raw_grounding_quality_risk": raw_score,
        }

        available = sum(1 for v in [
            data.broad_claim_risk,
            data.lived_detail_risk,
            data.citation_weakness_risk,
            data.unsupported_claim_risk,
        ] if v is not None)

        confidence = confidence_from_coverage(
            word_count=data.word_count,
            sentence_count=data.sentence_count,
            available_count=available,
            expected_count=4,
            important_missing=1 if data.citation_weakness_risk is None else 0,
        )

        reasons = []
        if score >= 0.65:
            reasons.append("high_grounding_quality_risk")
        elif score >= 0.45:
            reasons.append("moderate_grounding_quality_risk")

        if lived_detail >= 0.65:
            reasons.append("weak_lived_detail")

        if citation >= 0.65:
            reasons.append("citation_weakness")

        return ClusterScore(
            name="grounding_quality",
            score=round(score, 4),
            confidence=confidence,
            components={k: round(v, 4) for k, v in components.items()},
            reasons=reasons,
        )

    def calculate_structure_process_cluster(self, data: Layer3Input) -> ClusterScore:
        progression = clamp(data.paragraph_progression_risk)
        uniformity = clamp(data.paragraph_uniformity_risk)
        starter = clamp(data.repeated_starter_risk)
        conclusion = clamp(data.formulaic_conclusion_risk)
        signpost = clamp(data.signpost_paragraph_risk)
        balanced_hedging = clamp(data.balanced_hedging_risk)
        intra_parallel = clamp(data.intra_paragraph_parallelism_risk)
        topic_uniformity = clamp(data.paragraph_topic_uniformity_risk)
        draft_jump = clamp(data.draft_evolution_jump_risk)
        reuse = clamp(data.structural_reuse_risk)

        # Core components — always included
        struct_components = {
            "progression": (progression, 0.30),
            "uniformity": (uniformity, 0.20),
            "starter": (starter, 0.25),
            "conclusion": (conclusion, 0.25),
        }

        # Optional components — only included when they actually fire (> threshold)
        # This prevents low-value "noise" from diluting strong core signals.
        # Each optional takes weight from core proportionally.
        active_optional = []
        if signpost > 0.30:
            active_optional.append(("signpost", signpost, 0.25))
        if balanced_hedging > 0.30:
            active_optional.append(("balanced_hedging", balanced_hedging, 0.20))
        if intra_parallel > 0.30:
            active_optional.append(("intra_parallel", intra_parallel, 0.15))
        if topic_uniformity > 0.30:
            active_optional.append(("topic_uniformity", topic_uniformity, 0.15))

        if active_optional:
            opt_weight = sum(w for _, _, w in active_optional)
            scale = 1.0 - opt_weight
            struct_components = {k: (v, w * scale) for k, (v, w) in struct_components.items()}
            for name, val, w in active_optional:
                struct_components[name] = (val, w)

        structural_subscore = weighted_average(struct_components)

        has_draft_process = (
            data.draft_evolution_jump_risk is not None
            or data.structural_reuse_risk is not None
        )

        if has_draft_process:
            draft_process_subscore = weighted_average({
                "draft_jump": (draft_jump, 0.55),
                "structural_reuse": (reuse, 0.45),
            })

            score = weighted_average({
                "structural": (structural_subscore, 0.45),
                "draft_process": (draft_process_subscore, 0.55),
            })
        else:
            score = structural_subscore

        components = {
            "paragraph_progression_risk": progression,
            "paragraph_uniformity_risk": uniformity,
            "repeated_starter_risk": starter,
            "formulaic_conclusion_risk": conclusion,
            "signpost_paragraph_risk": signpost,
            "balanced_hedging_risk": balanced_hedging,
            "intra_paragraph_parallelism_risk": intra_parallel,
            "paragraph_topic_uniformity_risk": topic_uniformity,
            "draft_evolution_jump_risk": draft_jump,
            "structural_reuse_risk": reuse,
            "structural_subscore": structural_subscore,
        }

        available = sum(1 for v in [
            data.paragraph_progression_risk,
            data.paragraph_uniformity_risk,
            data.repeated_starter_risk,
            data.formulaic_conclusion_risk,
            data.signpost_paragraph_risk,
            data.balanced_hedging_risk,
            data.intra_paragraph_parallelism_risk,
            data.paragraph_topic_uniformity_risk,
            data.draft_evolution_jump_risk,
            data.structural_reuse_risk,
        ] if v is not None)

        expected_count = 10

        confidence = confidence_from_coverage(
            word_count=data.word_count,
            sentence_count=data.sentence_count,
            available_count=available,
            expected_count=expected_count,
            important_missing=0 if has_draft_process else 1,
        )

        if not has_draft_process and confidence == Confidence.HIGH:
            confidence = Confidence.MEDIUM

        reasons = []
        if score >= 0.60:
            reasons.append("high_structure_process_cluster")
        elif score >= 0.40:
            reasons.append("moderate_structure_process_cluster")

        if progression >= 0.65:
            reasons.append("formulaic_paragraph_progression")

        if conclusion >= 0.65:
            reasons.append("formulaic_conclusion")

        if signpost >= 0.55:
            reasons.append("signpost_paragraphs")

        if balanced_hedging >= 0.55:
            reasons.append("balanced_hedging")

        if intra_parallel >= 0.45:
            reasons.append("intra_paragraph_parallelism")

        if topic_uniformity >= 0.45:
            reasons.append("paragraph_topic_uniformity")

        if draft_jump >= 0.65:
            reasons.append("draft_evolution_jump")

        if reuse >= 0.65:
            reasons.append("structural_reuse")

        return ClusterScore(
            name="structure_process",
            score=round(score, 4),
            confidence=confidence,
            components={k: round(v, 4) for k, v in components.items()},
            reasons=reasons,
        )

    def blend_clusters(
        self,
        text: ClusterScore,
        grounding: ClusterScore,
        process: ClusterScore,
        has_draft_process: bool,
    ) -> float:
        if has_draft_process:
            return clamp(
                0.30 * text.score
                + 0.40 * grounding.score
                + 0.30 * process.score
            )

        return clamp(
            0.30 * text.score
            + 0.40 * grounding.score
            + 0.30 * process.score
        )

    def derive_tier(
        self,
        score: float,
        text: ClusterScore,
        grounding: ClusterScore,
        process: ClusterScore,
        data: Layer3Input,
        confidence: Confidence,
    ) -> tuple[Tier, list[str], list[str]]:
        reasons: list[str] = []
        guardrails: list[str] = []

        if data.verified_ai_provenance:
            return Tier.RED, ["verified_ai_provenance"], guardrails

        if data.human_provenance_positive:
            guardrails.append("human_provenance_positive_considered")

        red_aligned = (
            text.score >= 0.60
            and grounding.score >= 0.60
            and process.score >= 0.55
            and score >= 0.60
            and confidence in (Confidence.MEDIUM, Confidence.HIGH)
        )

        if red_aligned:
            reasons.append("aligned_high_clusters")
            return Tier.RED, reasons, guardrails

        orange_aligned = (
            (
                text.score >= 0.55
                and grounding.score >= 0.55
                and score >= 0.50
            )
            or (
                grounding.score >= 0.65
                and process.score >= 0.50
                and score >= 0.50
            )
            or (
                text.score >= 0.55
                and process.score >= 0.55
                and score >= 0.50
            )
        )

        if orange_aligned:
            reasons.append("two_cluster_alignment")
            return Tier.ORANGE, reasons, guardrails

        if grounding.score >= 0.65:
            reasons.append("grounding_review_priority")
            return Tier.AMBER, reasons, guardrails

        if score >= 0.50:
            reasons.append("score_orange_threshold")
            return Tier.ORANGE, reasons, guardrails

        if score >= 0.30:
            reasons.append("score_amber_threshold")
            return Tier.AMBER, reasons, guardrails

        reasons.append("low_cluster_score")
        return Tier.GREEN, reasons, guardrails

    def score(self, data: Layer3Input) -> Layer3Result:
        text = self.calculate_text_pattern_cluster(data)
        grounding = self.calculate_grounding_quality_cluster(data)
        process = self.calculate_structure_process_cluster(data)

        has_draft_process = (
            data.draft_evolution_jump_risk is not None
            or data.structural_reuse_risk is not None
        )

        blended = self.blend_clusters(
            text=text,
            grounding=grounding,
            process=process,
            has_draft_process=has_draft_process,
        )

        calibrated = blended

        if data.human_provenance_positive:
            calibrated *= 0.80

        calibrated = clamp(calibrated)

        confidence = merge_confidence(
            text.confidence,
            grounding.confidence,
            process.confidence,
        )

        tier, reasons, guardrails = self.derive_tier(
            score=calibrated,
            text=text,
            grounding=grounding,
            process=process,
            data=data,
            confidence=confidence,
        )

        if (
            tier == Tier.RED
            and grounding.score < 0.45
            and process.score < 0.45
            and not data.verified_ai_provenance
        ):
            tier = Tier.ORANGE
            guardrails.append("red_downgraded_text_only_evidence")

        return Layer3Result(
            tier=tier,
            blended_score=round(blended, 4),
            calibrated_score=round(calibrated, 4),
            confidence=confidence,
            text_pattern=text,
            grounding_quality=grounding,
            structure_process=process,
            reasons=reasons,
            guardrails=guardrails,
            debug={
                "has_draft_process": has_draft_process,
                "word_count": data.word_count,
                "sentence_count": data.sentence_count,
                "paragraph_count": data.paragraph_count,
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

    return Layer3Input(
        predictability=predictability,
        topk_pattern=topk_pattern,
        generic_phrase_density=generic_phrase_density,
        burstiness_risk=estimate_burstiness_risk(text),
        repeated_sentence_structure_risk=estimate_repeated_sentence_structure_risk(text),
        generic_assertion_risk=generic_assertion,

        broad_claim_risk=broad_claim_risk,
        lived_detail_risk=estimate_lived_detail_risk(text, domain_lived_detail_patterns),
        citation_weakness_risk=citation_weakness_risk,
        unsupported_claim_risk=unsupported_claim_risk,
        source_grounding_strength=source_grounding_strength,
        domain_grounding_strength=domain_grounding_strength,

        paragraph_progression_risk=formulaic_progression,
        paragraph_uniformity_risk=estimate_paragraph_uniformity_risk(text),
        repeated_starter_risk=estimate_repeated_starter_risk(text),
        formulaic_conclusion_risk=estimate_formulaic_conclusion_risk(text),
        signpost_paragraph_risk=estimate_signpost_paragraph_risk(text),
        balanced_hedging_risk=estimate_balanced_hedging_risk(text),
        intra_paragraph_parallelism_risk=estimate_intra_paragraph_parallelism_risk(text),
        paragraph_topic_uniformity_risk=estimate_paragraph_topic_uniformity_risk(text),

        # In a production codebase, rename this field to balanced_framing_risk.
        style_shift_risk=balanced_framing,

        human_provenance_positive=human_provenance_positive,
        verified_ai_provenance=verified_ai_provenance,

        word_count=len(text.split()),
        sentence_count=len(sentences),
        paragraph_count=len(paragraphs),
    )
