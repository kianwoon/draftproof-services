"""Compression policy for rewrite V3."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from .document_units import word_count


@dataclass(frozen=True)
class CompressionPolicy:
    family: str
    min_ratio: float
    preferred_ratio: float
    max_ratio: float
    source_words: int

    @property
    def min_words(self) -> int:
        return max(1, int(round(self.source_words * self.min_ratio)))

    @property
    def preferred_words(self) -> int:
        return max(1, int(round(self.source_words * self.preferred_ratio)))

    @property
    def max_words(self) -> int:
        return max(1, int(round(self.source_words * self.max_ratio)))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update({
            "min_words": self.min_words,
            "preferred_words": self.preferred_words,
            "max_words": self.max_words,
        })
        return payload


def _ratio_env(name: str, default: float) -> float:
    try:
        return max(0.05, min(1.5, float(os.environ.get(name, str(default)))))
    except (TypeError, ValueError):
        return default


def compression_policy_for_family(family: str, source_words: int) -> CompressionPolicy:
    """Return the controlled compression band for a V3 strategy family."""

    family = str(family or "document_rhythm")
    if family == "cited_practice_voice":
        min_ratio = _ratio_env("DRAFTPROOF_REWRITE_V3_CITED_MIN_RATIO", 0.60)
        preferred_ratio = _ratio_env("DRAFTPROOF_REWRITE_V3_CITED_PREFERRED_RATIO", 0.72)
        max_ratio = _ratio_env("DRAFTPROOF_REWRITE_V3_CITED_MAX_RATIO", 0.86)
    else:
        min_ratio = _ratio_env("DRAFTPROOF_REWRITE_V3_RHYTHM_MIN_RATIO", 0.58)
        preferred_ratio = _ratio_env("DRAFTPROOF_REWRITE_V3_RHYTHM_PREFERRED_RATIO", 0.68)
        max_ratio = _ratio_env("DRAFTPROOF_REWRITE_V3_RHYTHM_MAX_RATIO", 0.82)
    if min_ratio > preferred_ratio:
        preferred_ratio = min_ratio
    if preferred_ratio > max_ratio:
        max_ratio = preferred_ratio
    return CompressionPolicy(
        family=family,
        min_ratio=min_ratio,
        preferred_ratio=preferred_ratio,
        max_ratio=max_ratio,
        source_words=max(1, int(source_words or 0)),
    )


def compression_status(source_text: str, candidate_text: str, policy: CompressionPolicy) -> dict[str, Any]:
    source_words = word_count(source_text)
    candidate_words = word_count(candidate_text)
    ratio = candidate_words / max(1, source_words)
    if candidate_words < policy.min_words:
        status = "below_floor"
    elif candidate_words > policy.max_words:
        status = "above_ceiling"
    else:
        status = "in_band"
    return {
        "source_words": source_words,
        "candidate_words": candidate_words,
        "ratio": round(ratio, 3),
        "policy": policy.to_dict(),
        "status": status,
        "in_band": status == "in_band",
    }
