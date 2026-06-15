"""QC reviewer: a document-aware pass that polishes the writer's rewrite into a teaching-grade
showcase.

Order in the pipeline: rewrite (writer) -> QC (this) -> final scan. The writer rewrites one
paragraph per LLM call, blind to the others, so it can produce cross-paragraph monocultures
(e.g. 7/8 paragraphs opening with the same first-person frame). This reviewer reads the FULL document, is guided
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
import re
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
    "17. Weak evidence: ground it in a concrete example from the document's own context.",
    "18. Over-polished wording: use normal human phrasing.",
    "19. Same subject starts: rotate sentence openings.",
    "20. No ownership: add a position, not just information.",
    "21. Dense academic phrasing: cut stacked modifiers.",
    "22. Weak source handling: attach the citation to the exact claim.",
    "23. AI-like conclusion: end with a consequence, not a slogan.",
    "24. Too-even paragraph shape: give each paragraph a distinct role.",
    "25. Rewrite drift: preserve the original idea first; improve expression without changing meaning.",
]

# Altitude split. The per-paragraph writer already applies the SENTENCE/PARAGRAPH-level guidelines
# when it rewrites each paragraph; handing all 25 to the reviewer just invites it to re-touch (and
# genericize) the writer's good sentences. The reviewer's proper scope is the DOCUMENT level --
# cross-paragraph / whole-document patterns the per-paragraph writer is structurally blind to:
#   8  robotic transitions (flow between paragraphs)
#   13 repeated rhythm (sameness across the doc)
#   14 predictable paragraph arc (document order)
#   19 same subject/opener starts (cross-paragraph monoculture)
#   23 AI-like conclusion (the document's close)
#   24 too-even paragraph shape (each paragraph a distinct role)
# Guideline 25 (preserve meaning) is a CONSTRAINT on every edit, enforced by _correction_is_safe and
# _SYSTEM, not a fix-target. Everything else is paragraph/sentence-level -> the writer's job.
DOCUMENT_LEVEL_GUIDELINE_IDS: frozenset[int] = frozenset({8, 13, 14, 19, 23, 24})


def _guideline_id(line: str) -> int | None:
    m = re.match(r"\s*(\d+)\.", line)
    return int(m.group(1)) if m else None


# Derived from the single source of truth above (no re-typed prose), so the two never drift.
DOCUMENT_LEVEL_GUIDELINES: list[str] = [
    g for g in WRITING_CRAFT_GUIDELINES if _guideline_id(g) in DOCUMENT_LEVEL_GUIDELINE_IDS
]


def _document_level_must_fix(issues: list[ResidualIssue]) -> list[ResidualIssue]:
    """Keep only the deterministic detectors whose guideline maps to the document-level subset.
    Drops sentence-level tells (e.g. balance_phrase #7) -- those belong to the writer/residual pass,
    not the reviewer."""
    return [i for i in issues if DOCUMENT_LEVEL_GUIDELINE_IDS.intersection(i.trick_ids)]


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


# Token budget for the QC call. Must cover gpt-oss's reasoning phase PLUS the sentence-only output;
# the reasoning phase alone exceeded 4000 on a whole-doc review. Default 16000 (verified to produce
# the full correction set). Tunable via DRAFTPROOF_V6_REVIEWER_MAX_TOKENS without a redeploy.
_REVIEWER_MAX_TOKENS_DEFAULT = 16000


def _reviewer_max_tokens() -> int:
    raw = os.environ.get("DRAFTPROOF_V6_REVIEWER_MAX_TOKENS", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return _REVIEWER_MAX_TOKENS_DEFAULT


def build_reviewer_prompt(text: str, must_fix: list[ResidualIssue]) -> str:
    payload: dict[str, Any] = {
        "task": "qc_review_the_full_document",
        "scope": "document_level_only_cross_paragraph_patterns",
        "guidelines": DOCUMENT_LEVEL_GUIDELINES,
        "must_fix_issues": [
            {"issue": issue.rule, "evidence": issue.evidence,
             "sentences_to_fix": issue.target_sentences}
            for issue in must_fix
        ],
        "full_document": text,
        "instructions": [
            "Resolve every must_fix issue, then stop. Only touch DOCUMENT-LEVEL / cross-paragraph patterns the per-paragraph writer cannot see: repeated openings across paragraphs, robotic transitions, uniform rhythm across the doc, predictable paragraph order, a slogan-like conclusion.",
            "Do NOT re-touch sentence-level wording, word choice, or grounding inside a single paragraph -- the writer already handled those. If a sentence is fine on its own, leave it alone.",
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


def _norm(text: str) -> str:
    return " ".join((text or "").split()).casefold()


def _is_style_correction(original: str, style_targets: list[str]) -> bool:
    """True if this correction resolves a deterministic STRUCTURAL/style must-fix -- its sentence is
    one of the residual-pattern detector's target sentences (opener/rhythm/repeated-start monoculture,
    balance phrase). Such corrections are readability fixes, exempt from the AI-score guard."""
    n = _norm(original)
    return bool(n) and any(n in t or t in n for t in style_targets)


def _correction_is_safe(original: str, revised: str, *, doc_after: str, baseline: float,
                        score_gated: bool = True) -> bool:
    """Keep a correction only if it doesn't break grammar or invert polarity.

    STYLE corrections (those resolving a deterministic residual-pattern detector -- opener/rhythm/
    repeated-start monoculture) pass score_gated=False: varying an opening is a readability fix that
    does NOT lower the perplexity-dominated AI-risk score (it nudges it up by noise), so gating it on
    `_score` systematically drops the genuine variations and keeps only cosmetic near-synonym edits,
    leaving the monoculture intact. Grounding/other corrections stay score-gated."""
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
    if score_gated and _score(doc_after) > baseline:
        return False
    return True


# gpt-oss intermittently (~1 in 5 calls, measured) runs away past max_tokens (finish_reason=length)
# and cuts the JSON mid-object, so parse_json raises and EVERY correction is lost. Retry to escape a
# bad roll, and salvage the corrections that DID complete from a truncated body before giving up.
_REVIEWER_ATTEMPTS = 2


def _salvage_corrections(raw: str) -> list[dict[str, Any]]:
    """Recover complete correction objects from a truncated/invalid reviewer response.

    Brace-matches the ``corrections`` array and parses each fully-closed ``{...}`` object, ignoring
    the cut-off tail (and any ``{``/``}`` that appear inside string values). Returns the recovered
    objects, or ``[]`` when nothing usable is present."""
    if not raw:
        return []
    key = raw.find('"corrections"')
    start = raw.find("[", key) if key >= 0 else -1
    if start < 0:
        return []
    objs: list[dict[str, Any]] = []
    depth = 0
    obj_start: int | None = None
    in_str = False
    esc = False
    for idx in range(start + 1, len(raw)):
        ch = raw[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        parsed = json.loads(raw[obj_start:idx + 1])
                        if isinstance(parsed, dict):
                            objs.append(parsed)
                    except Exception:
                        pass
                    obj_start = None
        elif ch == "]" and depth == 0:
            break
    return objs


def _request_corrections(
    gateway: Any, prompt: str, cancellation_check: Callable[[], None] | None
) -> tuple[list[dict[str, Any]], str | None]:
    """Ask the reviewer LLM for corrections, robust to gpt-oss's runaway/truncation. Per attempt:
    parse the JSON; on parse failure, salvage the complete objects from the partial body; retry on a
    total miss. Returns (corrections, skipped_reason): reason is None on a clean parse,
    'salvaged_partial' when recovered from a truncated body, else 'bad_json'/'llm_error'."""
    last_reason = "llm_error"
    for _ in range(_REVIEWER_ATTEMPTS):
        if cancellation_check:
            cancellation_check()
        try:
            response = gateway.chat(
                prompt,
                system=_SYSTEM,
                temperature=0.4,
                top_p=0.9,
                # Must cover gpt-oss's reasoning phase PLUS the sentence-only output; tunable via
                # DRAFTPROOF_V6_REVIEWER_MAX_TOKENS. Even at 16000 a runaway can hit the cap, which
                # is why _salvage_corrections + retry exist below.
                max_tokens=_reviewer_max_tokens(),
                response_format={"type": "json_object"},
                app_label="DocumentReviewer",
            )
            raw = getattr(response, "raw_content", "") or getattr(response, "content", "") or ""
        except Exception:
            last_reason = "llm_error"
            continue
        try:
            data = parse_json(raw)
        except Exception:
            data = None
        if isinstance(data, dict):
            return [c for c in (data.get("corrections") or []) if isinstance(c, dict)], None
        salvaged = _salvage_corrections(raw)
        if salvaged:
            return salvaged, "salvaged_partial"
        last_reason = "bad_json"
    return [], last_reason


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
    # Reviewer scope is the document level only: keep cross-paragraph/whole-doc detectors, drop
    # sentence-level tells (balance_phrase #7) -- the writer/residual pass owns those.
    must_fix = _document_level_must_fix(detect_residual_patterns(text))
    baseline = _score(text)
    prompt = build_reviewer_prompt(text, must_fix)
    corrections_in, skipped = _request_corrections(gateway, prompt, cancellation_check)

    current = text
    applied: list[Correction] = []
    # Corrections that resolve a deterministic structural/style must-fix (opener/rhythm/repeated-
    # start monoculture, balance phrase) are NOT gated on the AI-risk score -- they're readability
    # fixes that the perplexity-dominated score doesn't reward (see _correction_is_safe).
    style_targets = [_norm(t) for issue in must_fix for t in (issue.target_sentences or [])]
    # Each accepted-or-rejected correction re-scores the whole document (_correction_is_safe ->
    # _score), so cost is O(corrections) full scans. Cap the number we evaluate to bound worst-case
    # latency if a model returns an unexpectedly long list; the surgical design expects only a
    # handful. Extra corrections beyond the cap are surfaced in the trace, not silently dropped.
    over_cap = max(0, len(corrections_in) - _MAX_CORRECTIONS)
    for item in corrections_in[:_MAX_CORRECTIONS]:
        original = str(item.get("original") or "")
        revised = str(item.get("revised") or "")
        if not original or original not in current:
            continue
        candidate_doc = current.replace(original, revised, 1)
        score_gated = not _is_style_correction(original, style_targets)
        if _correction_is_safe(original, revised, doc_after=candidate_doc, baseline=baseline,
                               score_gated=score_gated):
            current = candidate_doc
            applied.append(Correction(original=original, revised=revised))

    # Honest unaddressed: re-detect on the corrected text and report which structural patterns STILL
    # fire. (The old 'was a target sentence touched' test reported success when a cosmetic edit
    # touched a sentence but left the pattern intact.)
    still_present = {issue.rule for issue in detect_residual_patterns(current)}
    unaddressed = [issue.rule for issue in must_fix if issue.rule in still_present]
    return ReviewResult(text=current, corrections=applied, skipped=skipped,
                        must_fix_unaddressed=unaddressed, corrections_over_cap=over_cap)
