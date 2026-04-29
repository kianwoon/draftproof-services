"""Voice guard: detects and prevents author voice erosion during rewrite.

Tracks VoiceProfile dimensions and rejects rewrites that erode the
writer's authentic voice — first person usage, rhythm variation,
domain specificity, lexical diversity.
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Lexicons ─────────────────────────────────────────────────────────

FIRST_PERSON_PATTERNS = [
    r'\bI\b', r'\bme\b', r'\bmy\b', r'\bmine\b', r'\bmyself\b',
    r'\bwe\b', r'\bus\b', r'\bour\b', r'\bours\b', r'\bourselves\b',
]

HEDGE_PATTERNS = [
    r'\bperhaps\b', r'\bmaybe\b', r'\bpossibly\b', r'\bseem\b',
    r'\bappear\b', r'\bsuggest\b', r'\bindicate\b', r'\blikely\b',
    r'\bin my experience\b', r'\bI think\b', r'\bI believe\b',
    r'\bcould\b', r'\bmight\b', r'\bmay\b', r'\bsomewhat\b',
]

INFORMAL_MARKERS = [
    r'\bsort of\b', r'\bkind of\b', r'\bstuff\b', r'\bthings\b',
    r'\blike\b', r'\bbasically\b', r'\bactually\b', r'\bpretty\b',
    r'\bokay\b', r'\bok\b', r'\banyway\b', r'\bwell\b',
    r'\byou know\b', r'\bI mean\b',
]


# ── VoiceProfile ─────────────────────────────────────────────────────

@dataclass
class VoiceProfile:
    avg_sentence_length: float
    sentence_length_cv: float       # coefficient of variation
    first_person_ratio: float       # first-person words / total words
    hedge_ratio: float
    informal_marker_ratio: float
    rhetorical_question_count: int
    domain_term_ratio: float = 0.0
    lexical_diversity: float = 0.0  # unique words / total words
    paragraph_length_cv: float = 0.0
    word_count: int = 0


# ── VoiceGuard ───────────────────────────────────────────────────────

@dataclass
class VoiceCheck:
    accepted: bool
    original_profile: VoiceProfile
    candidate_profile: VoiceProfile
    warnings: List[str] = field(default_factory=list)
    reject_reason: str = ""


class VoiceGuard:
    """Reject rewrites that erode the writer's voice."""

    def __init__(
        self,
        max_first_person_drop: float = 0.5,     # reject if first-person drops by >50%
        max_cv_drop: float = 0.4,                # reject if rhythm CV drops by >40%
        max_domain_drop: float = 0.3,            # reject if domain terms drop by >30%
        max_diversity_drop: float = 0.15,        # reject if lexical diversity drops by >15%
    ):
        self.max_first_person_drop = max_first_person_drop
        self.max_cv_drop = max_cv_drop
        self.max_domain_drop = max_domain_drop
        self.max_diversity_drop = max_diversity_drop

    def check(self, original: str, candidate: str) -> VoiceCheck:
        orig_profile = analyze_voice(original)
        cand_profile = analyze_voice(candidate)

        warnings = []
        reject_reason = ""

        # First-person erosion
        if orig_profile.first_person_ratio > 0:
            drop = 1 - (cand_profile.first_person_ratio / max(orig_profile.first_person_ratio, 0.001))
            if drop > self.max_first_person_drop:
                reject_reason = "first_person_voice_eroded"
            elif drop > self.max_first_person_drop * 0.7:
                warnings.append(f"first_person_ratio declining: {orig_profile.first_person_ratio:.3f} → {cand_profile.first_person_ratio:.3f}")

        # Rhythm erosion
        if orig_profile.sentence_length_cv > 0:
            cv_drop = 1 - (cand_profile.sentence_length_cv / max(orig_profile.sentence_length_cv, 0.001))
            if cv_drop > self.max_cv_drop:
                reject_reason = reject_reason or "rhythm_over_smoothed"
            elif cv_drop > self.max_cv_drop * 0.7:
                warnings.append(f"sentence_length_cv declining: {orig_profile.sentence_length_cv:.3f} → {cand_profile.sentence_length_cv:.3f}")

        # Domain specificity
        if orig_profile.domain_term_ratio > 0:
            dom_drop = 1 - (cand_profile.domain_term_ratio / max(orig_profile.domain_term_ratio, 0.001))
            if dom_drop > self.max_domain_drop:
                reject_reason = reject_reason or "domain_specificity_lost"
            elif dom_drop > self.max_domain_drop * 0.7:
                warnings.append(f"domain_term_ratio declining: {orig_profile.domain_term_ratio:.3f} → {cand_profile.domain_term_ratio:.3f}")

        # Lexical diversity
        if orig_profile.lexical_diversity > 0:
            div_drop = 1 - (cand_profile.lexical_diversity / max(orig_profile.lexical_diversity, 0.001))
            if div_drop > self.max_diversity_drop:
                reject_reason = reject_reason or "voice_homogenized"
            elif div_drop > self.max_diversity_drop * 0.7:
                warnings.append(f"lexical_diversity declining: {orig_profile.lexical_diversity:.3f} → {cand_profile.lexical_diversity:.3f}")

        accepted = reject_reason == ""
        return VoiceCheck(
            accepted=accepted,
            original_profile=orig_profile,
            candidate_profile=cand_profile,
            warnings=warnings,
            reject_reason=reject_reason,
        )


# ── Analysis ─────────────────────────────────────────────────────────

def _count_pattern_matches(text: str, patterns: list) -> int:
    count = 0
    for pat in patterns:
        count += len(re.findall(pat, text, re.I))
    return count


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split('\n\n') if p.strip()]


def _safe_cv(values: List[float]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean


def analyze_voice(text: str, domain_terms: Optional[List[str]] = None) -> VoiceProfile:
    """Analyze text and produce a VoiceProfile."""
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    if word_count == 0:
        return VoiceProfile(0, 0, 0, 0, 0, 0)

    # Sentence-level metrics
    sentences = _split_sentences(text)
    sent_lengths = [len(s.split()) for s in sentences if s]
    avg_sent_len = sum(sent_lengths) / max(len(sent_lengths), 1)
    sent_cv = _safe_cv([float(x) for x in sent_lengths])

    # First-person ratio
    first_person_count = _count_pattern_matches(text, FIRST_PERSON_PATTERNS)
    first_person_ratio = first_person_count / word_count

    # Hedge ratio
    hedge_count = _count_pattern_matches(text, HEDGE_PATTERNS)
    hedge_ratio = hedge_count / word_count

    # Informal markers
    informal_count = _count_pattern_matches(text, INFORMAL_MARKERS)
    informal_ratio = informal_count / word_count

    # Rhetorical questions
    rhetorical_count = len(re.findall(r'\b(what|why|how|when|where|who|which)\b.*\?', text, re.I))

    # Domain terms (if provided)
    domain_term_ratio = 0.0
    if domain_terms:
        dt_count = sum(1 for w in words if w.lower() in {t.lower() for t in domain_terms})
        domain_term_ratio = dt_count / word_count

    # Lexical diversity
    unique_words = len(set(w.lower() for w in words))
    lexical_diversity = unique_words / word_count

    # Paragraph rhythm
    paragraphs = _split_paragraphs(text)
    para_lengths = [len(p.split()) for p in paragraphs]
    para_cv = _safe_cv([float(x) for x in para_lengths])

    return VoiceProfile(
        avg_sentence_length=round(avg_sent_len, 2),
        sentence_length_cv=round(sent_cv, 3),
        first_person_ratio=round(first_person_ratio, 4),
        hedge_ratio=round(hedge_ratio, 4),
        informal_marker_ratio=round(informal_ratio, 4),
        rhetorical_question_count=rhetorical_count,
        domain_term_ratio=round(domain_term_ratio, 4),
        lexical_diversity=round(lexical_diversity, 4),
        paragraph_length_cv=round(para_cv, 3),
        word_count=word_count,
    )
