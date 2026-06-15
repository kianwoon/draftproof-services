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
# NO-HARDCODE: the baked generic-filler phrase list (academic/transition/hedge/conclusion/
# business-tech clichés such as "plays a vital role", "in the modern world", "paradigm shift")
# was removed. "Generic filler" = low token-level surprisal, which the predictability scanner's
# statistical signals (surprisal/top-k/burstiness) already measure directly. Consumers
# (generic_phrases criterion, scanner GENERIC_PHRASES) tolerate the empty list.
_GENERIC_FILLER: List[str] = []


# NO-HARDCODE: the per-domain filler phrase lists (education/hair_beauty/healthcare/legal/
# engineering domain vocabulary -- "learner-centered approach", "patient-centered care",
# "client consultation"...) were removed. Domain-specific phrasing is content the detector must
# NOT bake in; generic/predictable phrasing in any domain is caught statistically by surprisal.
# Profile names are kept as keys (config identifiers) with empty lists so profile wiring still
# resolves; get_phrases_for_packs() simply adds nothing domain-specific.
BUILTIN_PACKS: Dict[str, List[str]] = {
    "education_pedagogy": [],
    "hair_beauty": [],
    "healthcare": [],
    "legal": [],
    "engineering": [],
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
