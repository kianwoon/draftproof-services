"""QC reviewer: a document-aware pass that polishes the writer's rewrite into a teaching-grade
showcase.

Order in the pipeline: rewrite (writer) -> QC (this) -> final scan. The writer rewrites one
paragraph per LLM call, blind to the others, so it can produce cross-paragraph monocultures
(e.g. 7/8 paragraphs opening "In my classroom"). This reviewer reads the FULL document, is guided
by the 25 writing-craft guidelines, and corrects substandard sentences. Deterministic detectors
(residual_patterns) run as a safety net guaranteeing the cross-paragraph patterns are always caught.

Surgical by design: the reviewer returns only corrected sentences (not the whole doc), so output
stays small regardless of length -- avoiding the empty/truncated gpt-oss response that reverted the
prior showcase. Each correction is guarded against a pre-QC baseline score; a regressing or broken
correction is dropped (the writer's sentence is kept). NOT a humanizer/detection-evasion tool.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .json_io import parse_json
from .residual_patterns import ResidualIssue, detect_residual_patterns
from .selector_diagnostics import _severe_polarity_inversion
from .text import Paragraph


# The 25 writing-craft guidelines (the user's list). Legitimate, content-agnostic craft guidance
# -- NOT a hardcode of domain answers. One source of truth, mirrored as problem -> fix.
WRITING_CRAFT_GUIDELINES: list[str] = [
    "1. Generic opening: start with pressure, not topic.",
    "2. Predictable start: avoid a broad noun followed by a broad claim.",
    "3. Weak context anchor: name the setting.",
    "4. Weak author anchor: add what the writer noticed.",
    "5. Packed list: split a long list into meaning groups (2-3 beats).",
    "6. Sentence overload: one job per sentence (claim / evidence / judgment).",
    "7. Balance-phrase filler: replace 'both opportunities and risks' with the actual benefit and the actual risk.",
    "8. Robotic transition: avoid Furthermore/Moreover/In conclusion; use cause-based transitions.",
    "9. Smooth but empty: add friction; show where the idea becomes difficult.",
    "10. Abstract nouns: convert nouns into actions (who does what).",
    "11. Weak judgment: say what should happen.",
    "12. Formulaic contrast: don't rely on a simple past-vs-present template; add the mechanism.",
    "13. Repeated sentence rhythm: vary short-long-short.",
    "14. Predictable paragraph arc: break the expected order.",
    "15. Generic benefit: attach the benefit to a user/action.",
    "16. Generic risk: attach the risk to a failure mode.",
    "17. Weak evidence: use a concrete classroom/workflow example.",
    "18. Over-polished wording: use normal human phrasing.",
    "19. Same subject starts: rotate sentence openings.",
    "20. No ownership: add a position, not just information.",
    "21. Dense academic phrasing: cut stacked modifiers.",
    "22. Weak source handling: attach the citation to the exact claim.",
    "23. AI-like conclusion: end with a consequence, not a slogan.",
    "24. Too-even paragraph shape: give each paragraph a distinct role.",
    "25. Rewrite drift: preserve the original idea first; improve expression without changing meaning.",
]

_SYSTEM = (
    "You are a writing QUALITY-CONTROL reviewer. You receive a draft that an automated rewriter "
    "produced one paragraph at a time, so it cannot see patterns that span the whole document. "
    "Your job: inspect the FULL draft against the writing-craft guidelines and correct sentences "
    "that fall short -- especially the must_fix issues, which were detected mechanically and MUST "
    "be resolved. Change ONLY what is substandard: vary repeated openings, replace robotic "
    "transitions, break uniform rhythm, and de-formulaic wording. NEVER remove or weaken the "
    "concrete, grounded specifics the draft already contains (names, figures, scenes, first-person "
    "facts) -- those are the point. Preserve every sentence's meaning and polarity. "
    "Return ONLY the corrected sentences, each as an exact-match replacement."
)


def reviewer_enabled() -> bool:
    """Kill switch. Default ON; set DRAFTPROOF_V6_REVIEWER=0 to disable (matches direct_rewrite)."""
    return os.environ.get("DRAFTPROOF_V6_REVIEWER", "1").strip().lower() not in {"0", "false", "no", "off"}


def build_reviewer_prompt(text: str, must_fix: list[ResidualIssue]) -> str:
    payload: dict[str, Any] = {
        "task": "qc_review_the_full_document",
        "guidelines": WRITING_CRAFT_GUIDELINES,
        "must_fix_issues": [
            {"issue": issue.rule, "evidence": issue.evidence,
             "sentences_to_fix": issue.target_sentences}
            for issue in must_fix
        ],
        "full_document": text,
        "instructions": [
            "Resolve every must_fix issue, and fix any other sentence that clearly falls short of "
            "the guidelines.",
            "Change only what is substandard. Keep all grounded specifics intact.",
            "Each correction's 'original' MUST be an exact substring of full_document so it can be "
            "spliced back. Quote it verbatim, including punctuation.",
            "Do not rewrite the whole document. Return only the sentences you actually change.",
            "vary repeated openings and transition words to improve flow and readability.",
        ],
        "output_schema": {
            "corrections": [{"original": "exact sentence from the document", "revised": "improved sentence"}]
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


@dataclass
class Correction:
    original: str
    revised: str
    rule: str = "qc"


# Upper bound on corrections evaluated per document. Each one re-scores the whole document, so this
# bounds worst-case latency if a model returns an unexpectedly long list. The surgical design
# expects only a handful; a 1800-word doc has ~15-20 paragraphs and the detectors flag a few, so a
# real run never approaches this. Anything beyond is reported via ReviewResult.corrections_over_cap.
_MAX_CORRECTIONS = 25


@dataclass
class ReviewResult:
    text: str
    corrections: list[Correction] = field(default_factory=list)
    skipped: str | None = None
    must_fix_unaddressed: list[str] = field(default_factory=list)
    corrections_over_cap: int = 0


def _score(text: str) -> float:
    """Real-detector AI risk for the whole doc (baseline + post-correction guard). Wraps
    direct_rewrite._document_ai_risk; isolated here so tests can monkeypatch it."""
    from .direct_rewrite import _document_ai_risk
    return _document_ai_risk(text)


def _correction_is_safe(original: str, revised: str, *, doc_after: str, baseline: float) -> bool:
    """Keep a correction only if it doesn't regress score, break grammar, or invert polarity."""
    from .direct_rewrite import _has_broken_grammar
    revised = (revised or "").strip()
    if not revised or len(revised.split()) < 3:
        return False
    if _has_broken_grammar(revised):
        return False
    # polarity must not flip vs the writer's original sentence. Paragraph is a frozen dataclass with
    # a REQUIRED `sentences` field; _severe_polarity_inversion only reads `.text`, so [] is safe.
    if _severe_polarity_inversion(revised, Paragraph(id="qc", index=0, text=original, sentences=[])):
        return False
    if _score(doc_after) > baseline:
        return False
    return True


def review_document(
    text: str,
    *,
    gateway: Any,
    cancellation_check: Callable[[], None] | None = None,
) -> ReviewResult:
    """QC the full rewritten document: detect must-fix patterns, ask the reviewer LLM for
    sentence-level corrections, splice safe ones by verbatim match. Never raises; on any failure
    returns the writer's text unchanged."""
    if cancellation_check:
        cancellation_check()
    must_fix = detect_residual_patterns(text)
    baseline = _score(text)
    prompt = build_reviewer_prompt(text, must_fix)
    try:
        response = gateway.chat(
            prompt,
            system=_SYSTEM,
            temperature=0.4,
            top_p=0.9,
            max_tokens=4000,
            response_format={"type": "json_object"},
            app_label="DocumentReviewer",
        )
        data = parse_json(getattr(response, "raw_content", "") or getattr(response, "content", "") or "")
    except Exception:
        return ReviewResult(text=text, corrections=[], skipped="llm_error")
    if not isinstance(data, dict):
        return ReviewResult(text=text, corrections=[], skipped="bad_json")

    current = text
    applied: list[Correction] = []
    # Each accepted-or-rejected correction re-scores the whole document (_correction_is_safe ->
    # _score), so cost is O(corrections) full scans. Cap the number we evaluate to bound worst-case
    # latency if a model returns an unexpectedly long list; the surgical design expects only a
    # handful. Extra corrections beyond the cap are surfaced in the trace, not silently dropped.
    corrections_in = [c for c in (data.get("corrections") or []) if isinstance(c, dict)]
    over_cap = max(0, len(corrections_in) - _MAX_CORRECTIONS)
    for item in corrections_in[:_MAX_CORRECTIONS]:
        original = str(item.get("original") or "")
        revised = str(item.get("revised") or "")
        if not original or original not in current:
            continue
        candidate_doc = current.replace(original, revised, 1)
        if _correction_is_safe(original, revised, doc_after=candidate_doc, baseline=baseline):
            current = candidate_doc
            applied.append(Correction(original=original, revised=revised))

    addressed = " ".join(c.original for c in applied)
    unaddressed = [
        issue.rule for issue in must_fix
        if not any(t in addressed for t in issue.target_sentences)
    ]
    return ReviewResult(text=current, corrections=applied, must_fix_unaddressed=unaddressed,
                        corrections_over_cap=over_cap)
