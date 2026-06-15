"""Domain profiles — configurable per-domain detection tuning.

Provides:
- DomainProfile dataclass with thresholds, phrase packs, domain terms
- Auto-detection of domain from text content
- Built-in profiles derived from phrase_packs
- External profile loading from ~/.draftproof/profiles/
"""

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .thresholds import ThresholdConfig
from .postprocess import PostProcessConfig


@dataclass
class DomainProfile:
    """Detection configuration for a specific domain.

    Use ``resolve_profile()`` to get the right profile for content,
    or construct directly for explicit control.
    """
    name: str
    phrase_packs: List[str] = field(default_factory=lambda: ["generic_academic"])
    domain_terms: List[str] = field(default_factory=list)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)

    def resolve_domain_terms(self, text: str) -> List[str]:
        """Return domain terms, auto-extracting from text if none configured."""
        if self.domain_terms:
            return self.domain_terms
        return auto_extract_domain_terms(text)


def auto_extract_domain_terms(text: str, min_freq: int = 1) -> List[str]:
    """Extract candidate domain terms automatically from text.

    Strategy (content-agnostic):
    1. Capitalized multi-word phrases (2-4 words) not at sentence start
    2. Hyphenated compound terms
    3. Quoted terms appearing 2+ times
    4. High-frequency non-stopword bigrams

    Returns deduplicated list of terms.
    """
    from .utils import extract_capitalized_phrases, extract_quoted_terms

    terms = set()

    # 1. Capitalized multi-word phrases
    for phrase in extract_capitalized_phrases(text, min_length=2, min_occurrences=2):
        terms.add(phrase.lower())

    # 2. Hyphenated compounds (e.g., "client-centered", "evidence-based")
    hyphenated = re.findall(r'\b([a-z]+-[a-z]+(?:-[a-z]+)*)\b', text.lower())
    for term, count in Counter(hyphenated).most_common(20):
        if count >= 2 and len(term) > 5:
            terms.add(term)

    # 3. Quoted terms
    for term in extract_quoted_terms(text, min_occurrences=2):
        terms.add(term.lower())

    # 4. High-frequency bigrams (non-stopword)
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    _STOP_WORDS = frozenset({
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "have", "been",
        "from", "this", "that", "with", "they", "will", "each", "make",
        "like", "been", "long", "very", "after", "also", "just", "than",
        "more", "other", "into", "could", "would", "should", "about",
        "which", "their", "there", "first", "would", "these", "some",
        "them", "what", "when", "what", "then", "were", "being", "does",
    })
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)
                if words[i] not in _STOP_WORDS and words[i+1] not in _STOP_WORDS]
    for bg, count in Counter(bigrams).most_common(15):
        if count >= 3:
            terms.add(bg)

    return sorted(terms)[:50]


def auto_detect_domain(text: str) -> str:
    """Detect the most likely domain by matching text against phrase packs.

    Returns the domain name with the highest phrase match ratio,
    or "default" if no domain scores above threshold.
    """
    from poc.predictability.phrase_packs import BUILTIN_PACKS

    text_lower = text.lower()
    scores: Dict[str, float] = {}

    for domain, phrases in BUILTIN_PACKS.items():
        if not phrases:
            continue
        matches = sum(1 for p in phrases if p.lower() in text_lower)
        ratio = matches / len(phrases)
        if ratio > 0:
            scores[domain] = ratio

    if not scores:
        return "default"

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    # Only return a domain if there's meaningful overlap (>10% of phrases matched)
    if scores[best] < 0.10:
        return "default"
    return best


def resolve_profile(text: str, domain: str = "auto") -> DomainProfile:
    """Resolve a DomainProfile for the given content and domain hint.

    Args:
        text: The content to analyze.
        domain: "auto" for auto-detection, or a specific domain name,
                or "default" for the generic profile.

    Returns:
        A DomainProfile configured for the detected/specified domain.
    """
    if domain == "auto":
        domain = auto_detect_domain(text)

    # Check built-in profiles first
    if domain in _BUILTIN_PROFILES:
        return _BUILTIN_PROFILES[domain]

    # Try loading external profile
    ext = _load_external_profile(domain)
    if ext is not None:
        return ext

    # Fall back to default
    return _BUILTIN_PROFILES["default"]


# ── External profile loading ─────────────────────────────────────────

_PROFILES_DIR = os.path.expanduser("~/.draftproof/profiles")


def _load_external_profile(domain: str) -> Optional[DomainProfile]:
    """Load a profile from ~/.draftproof/profiles/<domain>.json."""
    path = os.path.join(_PROFILES_DIR, f"{domain}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return DomainProfile(
            name=data.get("name", domain),
            phrase_packs=data.get("phrase_packs", ["generic_academic"]),
            domain_terms=data.get("domain_terms", []),
            thresholds=ThresholdConfig.from_dict(data.get("thresholds")),
            postprocess=PostProcessConfig.from_dict(data.get("postprocess")),
        )
    except (json.JSONDecodeError, TypeError):
        return None


# ── Built-in profiles ─────────────────────────────────────────────────

_BUILTIN_PROFILES: Dict[str, DomainProfile] = {
    "default": DomainProfile(
        name="default",
        phrase_packs=["generic_academic"],
        domain_terms=[],
        thresholds=ThresholdConfig(),
        postprocess=PostProcessConfig(),
    ),
    "education_pedagogy": DomainProfile(
        name="education_pedagogy",
        phrase_packs=["generic_academic", "education_pedagogy"],
        domain_terms=[],
        thresholds=ThresholdConfig(),
        postprocess=PostProcessConfig(
            glossary_overlap=0.35,
            glossary_min_repeat=2,
        ),
    ),
    "healthcare": DomainProfile(
        name="healthcare",
        phrase_packs=["generic_academic", "healthcare"],
        domain_terms=[],
        thresholds=ThresholdConfig(),
        postprocess=PostProcessConfig(
            glossary_overlap=0.35,
            glossary_min_repeat=2,
        ),
    ),
    "legal": DomainProfile(
        name="legal",
        phrase_packs=["generic_academic", "legal"],
        domain_terms=[],
        thresholds=ThresholdConfig(),
        postprocess=PostProcessConfig(
            glossary_overlap=0.40,
            glossary_min_repeat=2,
        ),
    ),
    "engineering": DomainProfile(
        name="engineering",
        phrase_packs=["generic_academic", "engineering"],
        domain_terms=[],
        thresholds=ThresholdConfig(),
        postprocess=PostProcessConfig(
            glossary_overlap=0.40,
            glossary_min_repeat=2,
        ),
    ),
    "business_tech": DomainProfile(
        name="business_tech",
        phrase_packs=["generic_academic", "business_tech"],
        domain_terms=[],
        thresholds=ThresholdConfig(),
        postprocess=PostProcessConfig(),
    ),
    "hair_beauty": DomainProfile(
        name="hair_beauty",
        phrase_packs=["generic_academic", "hair_beauty"],
        domain_terms=[],
        thresholds=ThresholdConfig(),
        postprocess=PostProcessConfig(
            glossary_overlap=0.35,
            glossary_min_repeat=2,
        ),
    ),
}


def list_available_profiles() -> List[str]:
    """Return names of all available profiles (built-in + external)."""
    names = list(_BUILTIN_PROFILES.keys())
    if os.path.isdir(_PROFILES_DIR):
        for f in os.listdir(_PROFILES_DIR):
            if f.endswith(".json"):
                names.append(f[:-5])
    return sorted(set(names))
