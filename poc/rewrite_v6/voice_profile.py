"""P0 voice-match spike: build a deterministic voice profile from the source paragraph and
render it as a per-paragraph prompt contract, so the writer matches the author's own
formality / rhythm / stance / diction instead of the static house voice.

Pure functions, no LLM calls, fail-open (any error -> None). Spike kill switch:
DRAFTPROOF_V6_VOICE_MATCH (default ON; set to 0 to disable).
"""

from __future__ import annotations

import contextvars
import math
import re

try:
    from detect.layer3_scoring import register_score, split_sentences
except ImportError:  # pragma: no cover
    from poc.detect.layer3_scoring import register_score, split_sentences

_FIRST_PERSON = frozenset({"i", "my", "me", "we", "our", "us", "i'm", "i've", "myself", "mine"})
_HEDGES = frozenset({
    "may", "might", "could", "can", "should", "would", "often", "generally",
    "tends", "perhaps", "possibly", "likely", "sometimes", "usually", "typically",
})
_WORD_RE = re.compile(r"[A-Za-z']+")
# Where the raw structural register_score sits on a 0..1 formality scale (logistic squash;
# raw 0 ~ neutral academic, negative = informal/first-person).
_PLAIN_THRESHOLD = 0.55

voice_options_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar("v6_voice_options", default=None)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def build_voice_profile(text: str) -> dict | None:
    """Deterministic voice profile for `text`, or None (fail-open) on empty/error.
    Abstains (profile['abstain'] = True) on fewer than 4 sentences."""
    try:
        sentences = [s for s in split_sentences(text or "") if s and _words(s)]
        if not sentences:
            return None
        n_sentences = len(sentences)
        formality = [_sigmoid(register_score(s)) for s in sentences]
        formality_mean = sum(formality) / len(formality)
        formality_spread = (max(formality) - min(formality)) if len(formality) > 1 else 0.0
        words = _words(text)
        buckets = {"short": 0, "med": 0, "long": 0}
        for s in sentences:
            n = len(_words(s))
            if n < 12:
                buckets["short"] += 1
            elif n <= 24:
                buckets["med"] += 1
            else:
                buckets["long"] += 1
        return {
            "formality_mean": round(formality_mean, 4),
            "formality_spread": round(formality_spread, 4),
            "sent_len_buckets": buckets,
            "first_person_density": round(sum(1 for w in words if w in _FIRST_PERSON) / max(1, len(words)), 4),
            "hedging_density": round(sum(1 for w in words if w in _HEDGES) / max(1, len(words)), 4),
            "lexical_diversity": round(len(set(words)) / max(1, len(words)), 4),
            "n_sentences": n_sentences,
            "abstain": n_sentences < 4,
        }
    except Exception:
        return None


def render_voice_contract(profile: dict) -> list[str]:
    """3-5 instruction strings matching the writer to the measured source voice.
    [voice-match] prefix marks spike-injected lines (tests / observability)."""
    try:
        if not profile or profile.get("abstain"):
            return []
        out: list[str] = []
        fm = float(profile.get("formality_mean") or 0.5)
        if fm < _PLAIN_THRESHOLD:
            out.append(
                "[voice-match] FORMALITY: the source is written in a plain, everyday register -- keep the "
                "rewrite plain and conversational; do not elevate it into polished academic prose."
            )
        else:
            out.append(
                "[voice-match] FORMALITY: the source is written in a formal, academic register -- keep the "
                "rewrite academic and measured; do not flatten it into casual speech."
            )
        b = profile.get("sent_len_buckets") or {}
        out.append(
            f"[voice-match] RHYTHM: the source mixes {b.get('short', 0)} short (<12-word), "
            f"{b.get('med', 0)} medium, and {b.get('long', 0)} long (>24-word) sentences -- keep a similar "
            "sentence-length mix rather than a uniform cadence."
        )
        fp = float(profile.get("first_person_density") or 0.0)
        if fp > 0.01:
            out.append(
                "[voice-match] STANCE: the source speaks in first person -- keep the rewrite in the "
                "author's first-person voice where present."
            )
        else:
            out.append(
                "[voice-match] STANCE: the source avoids first person -- keep the rewrite impersonal; do "
                "not introduce 'I/we' stances that are not in the source."
            )
        hd = float(profile.get("hedging_density") or 0.0)
        out.append(
            f"[voice-match] DICTION: the source hedges at ~{round(hd * 100)}% of words -- trim hedging "
            "toward direct statements but do not strip qualifiers that carry the author's actual "
            "uncertainty; lexical variety is about "
            f"{round(float(profile.get('lexical_diversity') or 0.0) * 100)}% unique words -- match it."
        )
        return out
    except Exception:
        return []


def voice_match_enabled() -> bool:
    """Spike kill switch: DRAFTPROOF_V6_VOICE_MATCH (default ON)."""
    import os
    return os.environ.get("DRAFTPROOF_V6_VOICE_MATCH", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }
