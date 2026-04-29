"""Domain phrase packs — modular generic filler detection per domain.

Usage:
    packs = PhrasePackLoader()
    packs.load_domain("education_pedagogy")
    phrases = packs.get_all_phrases()
"""

import os
from typing import List, Dict, Optional


# ── Built-in phrase packs ──────────────────────────────────────────

# Generic filler phrases — the single source of truth for all detectors.
# This replaces the old GENERIC_PHRASES in scanner.py and GENERIC_TRANSITIONS/
# HEDGES/CONCLUSIONS in criteria/generic_phrases.py.
_GENERIC_FILLER = [
    # academic filler
    "it is worth mentioning",
    "it should be noted that",
    "in recent years",
    "has gained significant attention",
    "has been widely studied",
    "increasingly important",
    "a growing body of research",
    "the literature suggests",
    "further research is needed",
    "sheds light on",
    "paves the way for",
    "in conclusion",
    "to summarize",
    "in summary",
    "it is important to note",
    # transitions
    "furthermore", "in addition", "moreover", "additionally",
    "consequently", "therefore", "thus", "hence", "accordingly",
    "nevertheless", "nonetheless", "notwithstanding",
    "as a result", "in other words", "for example", "for instance",
    "in particular", "specifically", "notably",
    # hedges
    "it can be argued", "it is suggested", "it is widely believed",
    "research suggests", "studies indicate", "it is generally accepted",
    "there is evidence", "it has been shown",
    # conclusions
    "overall, this demonstrates", "overall, this shows",
    "this highlights the importance", "this demonstrates the importance",
    "this underscores", "this emphasizes",
    "this plays a crucial role", "plays a vital role",
    "in today's world", "in the modern world",
    # business / tech filler
    "plays an important role",
    "has transformed the way",
    "in today's fast-paced world",
    "enhancing efficiency",
    "reducing costs",
    "better decision-making",
    "more accessible",
    "personalized and engaging",
    "unlock new opportunities",
    "drive innovation",
    "significant impact",
    "rapidly evolving",
    "seamless experience",
    "leveraging cutting-edge technology",
    "revolutionizing the industry",
    "game-changing solution",
    "paradigm shift",
    "disruptive innovation",
    "at the forefront of",
    "stay ahead of the curve",
    "next-generation",
    "best-in-class",
]


BUILTIN_PACKS: Dict[str, List[str]] = {
    # generic_academic and business_tech phrases are now in _GENERIC_FILLER
    # These domain packs contain ONLY domain-specific filler phrases.
    "education_pedagogy": [
        "learner-centered approach",
        "holistic development",
        "inclusive classroom",
        "meaningful learning experience",
        "differentiated instruction",
        "student engagement",
        "critical thinking skills",
        "lifelong learning",
        "scaffolding",
        "formative assessment",
        "constructivist approach",
        "zone of proximal development",
        "active learning strategies",
        "collaborative learning",
        "reflective practice",
        "teaching and learning",
        "educational outcomes",
        "pedagogical approach",
        "learning objectives",
        "student-centered",
        "transformative learning",
    ],
    "hair_beauty": [
        "client consultation",
        "professional standards",
        "hygiene and safety practices",
        "practical hands-on skills",
        "salon environment",
        "client satisfaction",
        "industry expectations",
        "professional development",
        "creative techniques",
        "industry best practices",
        "professional integrity",
        "client communication",
        "trend awareness",
        "health and safety",
        "professional image",
    ],
    "healthcare": [
        "patient-centered care",
        "evidence-based practice",
        "clinical outcomes",
        "interdisciplinary collaboration",
        "quality improvement",
        "patient safety",
        "best practice guidelines",
        "holistic patient care",
        "continuum of care",
        "patient outcomes",
    ],
    "legal": [
        "it is well established",
        "the court held that",
        "it is submitted that",
        "on the balance of probabilities",
        "beyond reasonable doubt",
        "the appropriate standard",
        "in the circumstances",
        "having regard to",
        "it is respectfully submitted",
        "the weight of authority",
    ],
    "engineering": [
        "optimized for performance",
        "robust and scalable",
        "industry-standard practices",
        "systematic approach",
        "rigorous testing",
        "performance metrics",
        "sustainable design",
        "innovative solutions",
        "technical specifications",
        "continuous improvement",
    ],
}


def get_generic_phrases() -> List[str]:
    """Return the master list of generic filler phrases.

    Single source of truth used by scanner.py and criteria/generic_phrases.py.
    """
    return list(_GENERIC_FILLER)


def get_phrases_for_packs(pack_names: List[str]) -> List[str]:
    """Combine generic filler phrases with named domain packs.

    Args:
        pack_names: Domain pack names to load (e.g. ["education_pedagogy"]).

    Returns:
        Deduplicated list of all phrases (generic + domain-specific).
    """
    seen: set = set()
    result: List[str] = []

    # Always include generic filler
    for p in _GENERIC_FILLER:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            result.append(p)

    # Add domain-specific packs
    for name in pack_names:
        if name in BUILTIN_PACKS:
            for p in BUILTIN_PACKS[name]:
                key = p.lower()
                if key not in seen:
                    seen.add(key)
                    result.append(p)

    return result


class PhrasePackLoader:
    """Load and combine domain-specific phrase packs."""

    def __init__(self):
        self._loaded: Dict[str, List[str]] = {}
        self._custom_dir: Optional[str] = None

    def load_domain(self, domain: str) -> "PhrasePackLoader":
        if domain in BUILTIN_PACKS:
            self._loaded[domain] = BUILTIN_PACKS[domain]
        return self

    def load_all(self) -> "PhrasePackLoader":
        for domain in BUILTIN_PACKS:
            self._loaded[domain] = BUILTIN_PACKS[domain]
        return self

    def set_custom_dir(self, directory: str) -> "PhrasePackLoader":
        self._custom_dir = directory
        return self

    def get_all_phrases(self) -> List[str]:
        seen = set()
        result = []
        for phrases in self._loaded.values():
            for p in phrases:
                if p.lower() not in seen:
                    seen.add(p.lower())
                    result.append(p)
        return result

    def get_loaded_domains(self) -> List[str]:
        return list(self._loaded.keys())

    def get_phrase_count(self) -> int:
        return len(self.get_all_phrases())
