"""Deterministic residual-pattern detectors for the QC reviewer (no LLM, no network).

The per-paragraph writer (direct_rewrite) is blind across paragraphs, so it can trade one AI
signal (generic assertion) for another (repetitive structure) -- e.g. 7/8 paragraphs opening
"In my classroom". These pure functions measure such patterns across the FULL rewritten document
and return evidence the QC reviewer must fix. They never edit text; they only report.

Content-agnostic structural/linguistic measures only (frame repetition, sentence-length variance,
closed-class connectives) -- not a banned-phrase or domain list. Aligns with the existing
_POLARITY_MARKERS / _CITATION_MARKERS closed-set precedent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ResidualIssue:
    rule: str                                   # e.g. "opener_monoculture"
    trick_ids: list[int]                        # guideline numbers, e.g. [19, 2]
    evidence: str                               # human-readable, e.g. "paragraphs 1,2,3,7 open 'In my'"
    target_sentences: list[str] = field(default_factory=list)  # exact sentences QC must fix


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", str(text or "")) if p.strip()]


def _first_sentence(paragraph: str) -> str:
    match = re.search(r"^.*?[.!?](?:\s|$)", paragraph.strip())
    return (match.group(0).strip() if match else paragraph.strip())


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").replace("\n", " ").strip())
    return [p.strip() for p in parts if p.strip()]


def _first_word(sentence: str) -> str:
    words = re.findall(r"[A-Za-z'']+", sentence)
    return words[0].lower() if words else ""


# ---------------------------------------------------------------------------
# Detector: opener_monoculture (#19, #2)
# ---------------------------------------------------------------------------

# A document needs at least this many paragraphs before opener repetition is meaningful.
_MIN_PARAGRAPHS_FOR_OPENER_CHECK = 3
# If this fraction (or more) of paragraphs share the same first-2-word opener frame, it reads as a
# monoculture (the writer's blind-spot: every paragraph independently picked the same frame).
_OPENER_SHARE_THRESHOLD = 0.5
# Number of leading words that define an "opener frame".
_OPENER_FRAME_WORDS = 2


def _opener_frame(paragraph: str, n: int = _OPENER_FRAME_WORDS) -> str:
    words = re.findall(r"[A-Za-z'']+", paragraph)
    return " ".join(w.lower() for w in words[:n])


def _detect_opener_monoculture(text: str) -> ResidualIssue | None:
    paras = _paragraphs(text)
    if len(paras) < _MIN_PARAGRAPHS_FOR_OPENER_CHECK:
        return None
    frames = [_opener_frame(p) for p in paras]
    counts: dict[str, int] = {}
    for f in frames:
        if f:
            counts[f] = counts.get(f, 0) + 1
    if not counts:
        return None
    top_frame, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count < 2 or top_count / len(paras) < _OPENER_SHARE_THRESHOLD:
        return None
    targets = [_first_sentence(p) for p, f in zip(paras, frames) if f == top_frame]
    evidence = (
        f"{top_count} of {len(paras)} paragraphs open with the same frame "
        f"'{top_frame}' -- vary the openings (guideline 19)."
    )
    return ResidualIssue(rule="opener_monoculture", trick_ids=[19, 2],
                         evidence=evidence, target_sentences=targets)


# ---------------------------------------------------------------------------
# Detector: robotic_transitions (#8)
# ---------------------------------------------------------------------------

# Closed-class formulaic transition openers (guideline 8). Sentence-initial only.
_ROBOTIC_TRANSITIONS = (
    "furthermore", "moreover", "additionally", "in addition", "in conclusion",
    "consequently", "thus", "therefore", "as a result", "overall", "to conclude",
    "in summary", "firstly", "secondly", "thirdly", "lastly", "notably",
)
# Need at least this many robotic openers in the doc before flagging (1 is fine in good prose).
_MIN_ROBOTIC_HITS = 2


def _starts_with_robotic(sentence: str) -> bool:
    low = sentence.strip().lower()
    return any(low.startswith(t + " ") or low.startswith(t + ",") for t in _ROBOTIC_TRANSITIONS)


def _detect_robotic_transitions(text: str) -> ResidualIssue | None:
    hits = [s for s in _sentences(text) if _starts_with_robotic(s)]
    if len(hits) < _MIN_ROBOTIC_HITS:
        return None
    evidence = (
        f"{len(hits)} sentences open with a formulaic transition "
        f"(Furthermore/Moreover/In conclusion ...) -- use cause-based transitions instead "
        f"(guideline 8)."
    )
    return ResidualIssue(rule="robotic_transitions", trick_ids=[8],
                         evidence=evidence, target_sentences=hits)


def detect_residual_patterns(text: str) -> list[ResidualIssue]:
    """Run every detector over the full document; return all fired issues (may be empty)."""
    issues: list[ResidualIssue] = []
    opener = _detect_opener_monoculture(text)
    if opener is not None:
        issues.append(opener)
    robotic = _detect_robotic_transitions(text)
    if robotic is not None:
        issues.append(robotic)
    return issues
