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


# ---------------------------------------------------------------------------
# Detector: repeated_subject_starts (#19)
# ---------------------------------------------------------------------------

# Same first word more than this many times within one paragraph reads as a repeated subject start.
_MAX_SAME_FIRST_WORD = 2


def _detect_repeated_subject_starts(text: str) -> ResidualIssue | None:
    flagged: list[str] = []
    for para in _paragraphs(text):
        sents = _sentences(para)
        counts: dict[str, int] = {}
        for s in sents:
            fw = _first_word(s)
            if fw:
                counts[fw] = counts.get(fw, 0) + 1
        for fw, c in counts.items():
            if c > _MAX_SAME_FIRST_WORD:
                flagged.extend(s for s in sents if _first_word(s) == fw)
    if not flagged:
        return None
    evidence = (
        "Several sentences in a paragraph start with the same word -- rotate the openings "
        "(guideline 19)."
    )
    return ResidualIssue(rule="repeated_subject_starts", trick_ids=[19],
                         evidence=evidence, target_sentences=flagged)


# ---------------------------------------------------------------------------
# Detector: balance_phrase (#7)
# ---------------------------------------------------------------------------

# "both X and Y" abstract balance phrasing (guideline 7).
_BALANCE_RE = re.compile(r"\bboth\b[^.?!]{0,60}?\band\b", re.I)


def _detect_balance_phrase(text: str) -> ResidualIssue | None:
    hits = [s for s in _sentences(text) if _BALANCE_RE.search(s)]
    if not hits:
        return None
    evidence = (
        "A 'both X and Y' balance phrase states the shape of a trade-off without the actual "
        "benefit or risk -- name the specific benefit and the specific risk (guideline 7)."
    )
    return ResidualIssue(rule="balance_phrase", trick_ids=[7],
                         evidence=evidence, target_sentences=hits)


# ---------------------------------------------------------------------------
# Detector: rhythm_sameness (#13)
# ---------------------------------------------------------------------------

# A paragraph with >= this many sentences whose length coefficient-of-variation is below the
# threshold reads as mechanically uniform rhythm (guideline 13).
_MIN_SENTENCES_FOR_RHYTHM = 4
_RHYTHM_CV_THRESHOLD = 0.20  # std/mean of sentence word-counts; below this is "too even"


def _coefficient_of_variation(values: list[int]) -> float:
    if len(values) < 2:
        return 1.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 1.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return (variance ** 0.5) / mean


def _detect_rhythm_sameness(text: str) -> ResidualIssue | None:
    worst: list[str] = []
    for para in _paragraphs(text):
        sents = _sentences(para)
        if len(sents) < _MIN_SENTENCES_FOR_RHYTHM:
            continue
        lengths = [len(re.findall(r"[A-Za-z'']+", s)) for s in sents]
        if _coefficient_of_variation(lengths) < _RHYTHM_CV_THRESHOLD:
            worst.extend(sents)
    if not worst:
        return None
    evidence = (
        "A paragraph's sentences are all about the same length -- vary the rhythm with a short "
        "sentence next to a longer one (guideline 13)."
    )
    return ResidualIssue(rule="rhythm_sameness", trick_ids=[13],
                         evidence=evidence, target_sentences=worst)


def detect_residual_patterns(text: str) -> list[ResidualIssue]:
    """Run every detector over the full document; return all fired issues (may be empty)."""
    issues: list[ResidualIssue] = []
    opener = _detect_opener_monoculture(text)
    if opener is not None:
        issues.append(opener)
    robotic = _detect_robotic_transitions(text)
    if robotic is not None:
        issues.append(robotic)
    repeated = _detect_repeated_subject_starts(text)
    if repeated is not None:
        issues.append(repeated)
    balance = _detect_balance_phrase(text)
    if balance is not None:
        issues.append(balance)
    rhythm = _detect_rhythm_sameness(text)
    if rhythm is not None:
        issues.append(rhythm)
    return issues
