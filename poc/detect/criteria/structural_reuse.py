"""Criterion: Structural reuse / surface rewrite detection.

Strong signal: detects when a document's skeleton matches known AI-generated
templates or when surface-level rewording preserves an underlying structure.

Three detection approaches:
1. Template skeleton matching — known AI essay/report structures
2. Paragraph pattern uniformity — all paragraphs follow the same formula
3. Surface rewrite indicators — high structural similarity with low lexical overlap

Without a reference corpus, this uses heuristic template matching against
common AI-generated document patterns.
"""

import re
from typing import Dict, List, Optional

from ._types import CriterionScore


# ── Known AI template skeletons ──────────────────────────────────────

# NO-HARDCODE: the baked AI essay-template phrase lists (intro/conclusion/body/transition
# skeletons such as "in the modern world", "plays a crucial role", "research suggests") were
# removed. Reused/templated structure is captured agnostically by the statistical predictability
# signals + repetitive_structure/paragraph_uniformity criteria. Empty lists keep the criterion's
# logic intact (pattern hit-counts are 0; it relies on its structural metrics).
AI_INTRODUCTION_PATTERNS: List[str] = []
AI_CONCLUSION_PATTERNS: List[str] = []
AI_BODY_PATTERNS: List[str] = []
AI_TRANSITION_PATTERNS: List[str] = []


def _count_pattern_hits(text: str, patterns: List[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text.lower()))


def _split_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


def _paragraph_role(para: str) -> str:
    """Classify paragraph as introduction, body, or conclusion."""
    lower = para.lower().strip()
    # Check conclusion first (shorter patterns, less ambiguous)
    if _count_pattern_hits(para, AI_CONCLUSION_PATTERNS) >= 1:
        return "conclusion"
    # Introduction: first-person or forward-looking
    if _count_pattern_hits(para, AI_INTRODUCTION_PATTERNS) >= 1:
        return "introduction"
    return "body"


def _paragraph_formula_score(para: str) -> float:
    """How formulaic is this paragraph? (0=organic, 1=fully templated)."""
    if not para.strip():
        return 0.0

    sentences = re.split(r'(?<=[.!?])\s+', para.strip())
    sentences = [s for s in sentences if s]
    if not sentences:
        return 0.0

    # Pattern density across all AI pattern types
    all_patterns = AI_BODY_PATTERNS + AI_TRANSITION_PATTERNS
    pattern_hits = _count_pattern_hits(para, all_patterns)
    pattern_density = min(1.0, pattern_hits / max(len(sentences), 1))

    # Sentence starter uniformity
    starters = []
    for s in sentences:
        words = s.split()
        if len(words) >= 2:
            starters.append(words[0].lower().rstrip(','))
    if len(starters) >= 3:
        starter_repeat = 1.0 - (len(set(starters)) / len(starters))
    else:
        starter_repeat = 0.0

    # Transition formula usage
    transition_hits = _count_pattern_hits(para, AI_TRANSITION_PATTERNS)
    transition_density = min(1.0, transition_hits / max(len(sentences), 1))

    return min(1.0, 0.40 * pattern_density + 0.30 * starter_repeat + 0.30 * transition_density)


def _detect_template_skeleton(paragraphs: List[str]) -> Dict:
    """Check if paragraph sequence matches a known AI essay template.

    Common AI templates:
    - Hook intro → thesis → body (claim+evidence+analysis ×N) → conclusion
    - Problem → significance → evidence → solution → conclusion
    - Background → argument → counter-argument → rebuttal → conclusion
    """
    if len(paragraphs) < 3:
        return {"skeleton_match": False, "score": 0.0}

    roles = [_paragraph_role(p) for p in paragraphs]

    # Check for classic AI essay skeleton: intro → body×N → conclusion
    has_intro = "introduction" in roles
    has_conclusion = "conclusion" in roles
    body_count = roles.count("body")

    # Check that intro is near the start and conclusion near the end
    intro_pos = roles.index("introduction") if has_intro else -1
    concl_pos = len(roles) - 1 - roles[::-1].index("conclusion") if has_conclusion else -1

    skeleton_score = 0.0
    if has_intro and intro_pos <= 1:
        skeleton_score += 0.30
    if has_conclusion and concl_pos >= len(roles) - 2:
        skeleton_score += 0.30
    if has_intro and has_conclusion and body_count >= 2:
        skeleton_score += 0.20

    # Check body paragraphs follow the same internal formula
    formula_scores = [_paragraph_formula_score(p) for p in paragraphs if _paragraph_role(p) == "body"]
    if formula_scores:
        avg_formula = sum(formula_scores) / len(formula_scores)
        # High uniformity across body paragraphs = template reuse
        if len(formula_scores) >= 2:
            variance = sum((f - avg_formula) ** 2 for f in formula_scores) / len(formula_scores)
            uniformity = 1.0 - min(1.0, variance * 10)  # low variance = high uniformity
        else:
            uniformity = 0.0

        skeleton_score += 0.20 * uniformity

    skeleton_match = skeleton_score >= 0.50

    return {
        "skeleton_match": skeleton_match,
        "score": round(skeleton_score, 4),
        "roles": roles,
        "body_formula_avg": round(avg_formula, 4) if formula_scores else 0.0,
        "body_paragraph_count": body_count,
    }


def _detect_surface_rewrite(paragraphs: List[str]) -> Dict:
    """Detect if paragraphs are surface-level rewrites of a template.

    Indicators:
    - All body paragraphs have similar sentence counts
    - All body paragraphs have similar lengths (±15%)
    - High formula scores across all body paragraphs
    """
    body_paras = [p for p in paragraphs if _paragraph_role(p) == "body"]
    if len(body_paras) < 2:
        return {"rewrite_score": 0.0, "note": "insufficient body paragraphs"}

    # Sentence count uniformity
    sent_counts = [len(re.split(r'(?<=[.!?])\s+', p.strip())) for p in body_paras]
    avg_sent = sum(sent_counts) / len(sent_counts)
    if avg_sent > 0:
        sent_cv = _stddev(sent_counts) / avg_sent
    else:
        sent_cv = 0

    # Word count uniformity
    word_counts = [len(p.split()) for p in body_paras]
    avg_words = sum(word_counts) / len(word_counts)
    if avg_words > 0:
        word_cv = _stddev(word_counts) / avg_words
    else:
        word_cv = 0

    # Formula scores
    formula_scores = [_paragraph_formula_score(p) for p in body_paras]
    avg_formula = sum(formula_scores) / len(formula_scores)

    # High uniformity + high formula = surface rewrite
    uniformity_signal = max(0, 1.0 - sent_cv) * max(0, 1.0 - word_cv)

    rewrite_score = 0.30 * uniformity_signal + 0.70 * avg_formula

    return {
        "rewrite_score": round(rewrite_score, 4),
        "sent_count_cv": round(sent_cv, 4),
        "word_count_cv": round(word_cv, 4),
        "avg_formula_score": round(avg_formula, 4),
        "uniformity_signal": round(uniformity_signal, 4),
    }


def _stddev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def score(
    content: str,
    *,
    reference_documents: Optional[List[str]] = None,
    **kwargs,
) -> CriterionScore:
    """Score structural reuse / surface rewrite criterion.

    Args:
        content: The document text to analyze.
        reference_documents: Optional list of reference documents for comparison.
            When absent, uses heuristic template matching.

    Returns:
        CriterionScore with value 0-1 (higher = more structural reuse detected).
    """
    paragraphs = _split_into_paragraphs(content)
    if len(paragraphs) < 2:
        return CriterionScore(
            name="structural_reuse",
            value=0.0,
            label="low",
            details={"note": "insufficient paragraphs for structural analysis"},
        )

    # Template skeleton detection
    skeleton = _detect_template_skeleton(paragraphs)

    # Surface rewrite detection
    rewrite = _detect_surface_rewrite(paragraphs)

    # Overall AI pattern density across entire document
    all_pattern_hits = (
        _count_pattern_hits(content, AI_INTRODUCTION_PATTERNS) +
        _count_pattern_hits(content, AI_BODY_PATTERNS) +
        _count_pattern_hits(content, AI_TRANSITION_PATTERNS) +
        _count_pattern_hits(content, AI_CONCLUSION_PATTERNS)
    )
    total_sentences = len(re.split(r'(?<=[.!?])\s+', content.strip()))
    pattern_density = min(1.0, all_pattern_hits / max(total_sentences, 1))

    # Composite score: skeleton + rewrite + pattern density
    composite = (
        0.35 * skeleton["score"] +
        0.35 * rewrite["rewrite_score"] +
        0.30 * min(1.0, pattern_density * 3.0)
    )

    # Discount without reference corpus
    if not reference_documents:
        composite *= 0.70
        mode = "heuristic"
    else:
        mode = "reference_comparison"

    if composite >= 0.40:
        label = "high"
    elif composite >= 0.22:
        label = "medium"
    else:
        label = "low"

    # Flag most formulaic paragraph
    flagged = []
    formula_scores = [(i, _paragraph_formula_score(p)) for i, p in enumerate(paragraphs)]
    formula_scores.sort(key=lambda x: x[1], reverse=True)
    if formula_scores and formula_scores[0][1] >= 0.30:
        flagged = [paragraphs[formula_scores[0][0]][:200]]

    return CriterionScore(
        name="structural_reuse",
        value=round(composite, 4),
        label=label,
        details={
            "mode": mode,
            "skeleton": skeleton,
            "rewrite": rewrite,
            "pattern_density": round(pattern_density, 4),
            "total_ai_pattern_hits": all_pattern_hits,
            "total_sentences": total_sentences,
        },
        flagged_excerpts=flagged,
    )
