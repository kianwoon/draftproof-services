"""Lean direct-rewrite path (A/B alternative to the heavy planner/writer/selector pipeline).

Validated by probe: the over-engineered planner->writer message buries the scanner's simple,
specific fix and over-constrains the writer into breaking meaning. A minimal prompt -- the
paragraph + the scanner's own diagnosis + "rewrite it to read human, keep every fact" -- beats the
whole pipeline (mean ~44 vs 52, tighter, cleaner output).

So this path: one LLM call per flagged paragraph, no flow_plans / coverage_beats / must_keep lists /
playbook / structural-metric gate. It KEEPS author-proxy (the writer may add a reviewable grounding
bridge for an anchor/grounding gap) and a lightweight meaning-safety backstop (polarity inversion,
dropped source beat, over-truncation) so a bad draw can't silently ship a meaning flip.

Enable with DRAFTPROOF_V6_DIRECT_REWRITE=1.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

try:
    from llm.gateway import LLMConfig, LLMGateway
except ImportError:  # pragma: no cover
    from poc.llm.gateway import LLMConfig, LLMGateway

from .coverage_guard import missing_required_source_beat_groups
from .integrity_guard import candidate_integrity_blockers
from .json_io import parse_json
from .llm_config import resolve_v6_api_key, resolve_v6_base_url, resolve_v6_model, writer_extra_body, writer_llm_profile, writer_model
from .report_contracts import paragraph_diagnosis
from .rewrite_playbook import playbook_entries
from .scan import Scan, findings_for_paragraph, scan_text
from .selector_diagnostics import _severe_polarity_inversion
from .text import Paragraph


def direct_rewrite_enabled() -> bool:
    # Default ON (the objective-aligned path). Kill switch: set DRAFTPROOF_V6_DIRECT_REWRITE=0 to
    # fall back to the legacy planner/selector pipeline without a redeploy.
    return os.environ.get("DRAFTPROOF_V6_DIRECT_REWRITE", "1").strip().lower() not in {"0", "false", "no", "off"}


_SYSTEM = (
    "You produce a SUGGESTED rewrite of a flagged paragraph for the author to review and edit. The "
    "author sees your changes in a before/after diff, so adding new content to fix the problem is "
    "expected and encouraged -- that is the mitigation. Your goal: lower AI-detection risk by making "
    "the writing specific, concrete, and human, while staying in the same subject, register, and tone "
    "as the source. Where the paragraph is generic or lacks a concrete anchor, ADD grounding: a "
    "specific mechanism, condition, process, or generic scenario that fits the document's topic. Do "
    "NOT invent named people, real institutions, place names, specific dates, or statistics -- the "
    "author must be able to keep what you add without fact-checking a fabrication, so use generic "
    "referents (a student, a teacher, a school) and concrete mechanisms, not invented proper nouns. "
    "Preserve the author's actual ARGUMENT and meaning -- do NOT flip a balanced 'not only X "
    "but also Y' into 'Y over X', and do not drop the author's existing ideas. List what you added "
    "in author_review_items so the author can confirm or replace it. "
    'Return JSON only: {"rewrite": "...", "author_review_items": [{"added": "...", "why": "..."}]}.'
)


def _prompt(paragraph_text: str, diagnosis: dict[str, Any] | None, finding_tags: list[str]) -> str:
    source_words = len(str(paragraph_text or "").split())
    word_budget = max(40, int(source_words * 1.3))
    predictable_phrases = list((diagnosis or {}).get("predictable_phrases") or [])
    payload: dict[str, Any] = {
        "paragraph": paragraph_text,
        "scanner_diagnosis": {
            "main_issue": diagnosis.get("main_issue"),
            "why_flagged": diagnosis.get("why_flagged"),
            "how_to_improve": diagnosis.get("recommendation"),
            "rewrite_hint_for_shape_only": diagnosis.get("rewrite_hint"),
        } if diagnosis else None,
        "flagged_issue_types": finding_tags,
        "predictable_phrases": predictable_phrases,
        "rewrite_examples": playbook_entries(finding_tags, paragraph_text),
        "length_budget": {
            "source_words": source_words,
            "max_words": word_budget,
            "rule": f"Keep the rewrite to AT MOST ~{word_budget} words (about 1.3x the source). Do NOT "
                    "roughly double it.",
        },
        "instructions": [
            "Most AI-detection risk here comes from CONTENT that is generic and unanchored, not from "
            "word choice. Ground generic claims by REWRITING the existing sentence to be specific -- "
            "swap a vague statement for a concrete one of similar length. Do NOT keep the vague "
            "sentence and append an extra example after it; that bloats the paragraph.",
            "predictable_phrases are the EXACT wordings the detector scored most statistically "
            "predictable -- the single strongest AI signal. Rewrite EACH one out: replace it with "
            "more particular, less-expected wording by changing the sentence route around it, not by "
            "swapping in a synonym. None of these phrases should survive verbatim.",
            "Turn every broad or generic assertion into a GROUNDED one by giving it a concrete "
            "situation it applies to: a specific case ('in cases where ...'), an illustrative example "
            "('for example, when a student ...'), or a situational clause (when / after / before / "
            "once / if ...). A claim with no example, case, or when/after situation is the strongest "
            "generic-assertion signal -- give each one a situation only THIS paragraph would describe. "
            "Use GENERIC actors (a student, a teacher) and real mechanisms; do NOT invent named "
            "people, institutions, places, dates, or statistics.",
            "rewrite_examples shows the SHAPE of each fix (before -> better) -- note they REPLACE, "
            "they don't add on top. Apply the same kind of transformation to THIS paragraph; do not "
            "copy the example wording.",
            "Rewrite the WHOLE paragraph. Across EVERY sentence, replace generic or predictable "
            "phrasing with concrete, specific wording. Change the sentence ROUTE, not just synonyms.",
            "Vary sentence length HARD: include at least one short sentence (4-8 words) and at least "
            "one long one (20-35 words). Start each sentence differently; never repeat an opening "
            "frame. Cut hedging (may, might, can, could, should, often, generally) and generic filler.",
            "Preserve the author's actual argument and meaning, and stay in the same subject and "
            "register as the source. Do not shift a balanced 'not only X but also Y' into 'Y over X', "
            "and do not drop their existing ideas.",
            "Respect length_budget: stay close to the source length. Add a NEW sentence only when a "
            "claim genuinely cannot be grounded by rewriting an existing one. List anything you add "
            "in author_review_items.",
            "Prefer concrete mechanisms and generic scenarios over invented specifics. Do NOT "
            "fabricate named people, real institutions, place names, dates, or statistics -- the "
            "author would have to fact-check and replace them. Use rewrite_hint only as the shape of "
            "the fix, never as wording to copy.",
        ],
    }
    return "Return JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


# Numbers/percentages and citation/study markers — fabricated specifics if absent from the source.
_NUMERIC = re.compile(r"\b\d[\d,\.]*%?\b")
_CITATION_MARKERS = (
    "survey", "study", "studies", "poll", "census", "et al", "according to",
    "researchers", "research shows", "research found", "data show", "statistics",
    "percent", "%", "respondents", "participants",
)


def _has_fabricated_specifics(candidate: str, source_text: str) -> bool:
    """True if the candidate introduces concrete specifics (numbers, dates, stats, named studies,
    citations) absent from the source -- i.e. fabrication. This is the real integrity protection."""
    src = " ".join(str(source_text or "").split()).casefold()
    cand = str(candidate or "")
    for token in _NUMERIC.findall(cand):
        norm = token.strip().casefold().rstrip("%.,")
        if norm and norm not in src:
            return True
    cand_low = cand.casefold()
    for marker in _CITATION_MARKERS:
        if marker in cand_low and marker not in src:
            return True
    return False


# Invented proper nouns -- named people, institutions, or places absent from the source
# (e.g. "Mr. Patel", "Lincoln High School"). Fabricated identities the author must replace,
# so they ride along as a review flag rather than a rejection.
_HONORIFIC = re.compile(r"\b(?:Mr|Mrs|Ms|Mx|Dr|Prof|Professor)\.?\s+[A-Z][a-z]+")
_PROPER_NOUN_PHRASE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def _has_fabricated_named_entities(candidate: str, source_text: str) -> bool:
    """True if the candidate introduces invented proper nouns (named people, institutions, or
    places) absent from the source. Catches honorific names and multi-word Title-Case spans -- the
    clearest fabricated identities -- which the author must replace before submitting."""
    src_low = " ".join(str(source_text or "").split()).casefold()
    cand = str(candidate or "")
    for match in _HONORIFIC.finditer(cand):
        if match.group(0).casefold() not in src_low:
            return True
    for match in _PROPER_NOUN_PHRASE.finditer(cand):
        if match.group(0).casefold() not in src_low:
            return True
    return False


def _severe_beat_loss(candidate: str, paragraph: Paragraph) -> bool:
    """Only flags a source beat as genuinely DROPPED (near-zero coverage), not synonym-rephrased.
    The legacy term-exact check false-positived on faithful synonym rewrites."""
    for group in (missing_required_source_beat_groups(candidate, paragraph) or []):
        if isinstance(group, dict):
            ratio = group.get("coverage_ratio")
            if isinstance(ratio, (int, float)) and ratio < 0.15:
                return True
    return False


# Clear syntactic breaks that would embarrass the user reading the draft. High-precision only:
# excludes the false-positive-prone integrity checks (malformed_nominal_stack, malformed_verb_complement,
# tool/nonhuman predicates) and stylistic ones (sentence_starts_with_conjunction, repeated_subject_start)
# so the readability backstop catches genuinely broken grammar without over-rejecting good rewrites.
_BROKEN_GRAMMAR = frozenset({
    "malformed_subject_verb_agreement", "malformed_negation_order", "malformed_modal_do_negation",
    "missing_verb_after_negation_scope", "split_negation_fragment", "malformed_connector_fragment",
    "stranded_prepositional_fragment", "standalone_additive_fragment", "bare_instruction_fragment",
    "dangling_modifier_sentence_start", "dangling_article_predicate", "dangling_terminal_and_tail",
    "dangling_additive_tail", "dangling_consequence_tail", "malformed_serial_verb_chain",
    "malformed_parallel_verb_tail", "malformed_parallel_connector_list", "malformed_additive_predicate",
    "malformed_with_finite_clause", "malformed_contrast_pair", "malformed_telegraphic_predicate",
    "demonstrative_agreement_error", "semantic_anchor_corruption", "lost_serial_punctuation",
    "broken_citation_shape",
})


def _has_broken_grammar(candidate: str) -> bool:
    """True if the rewrite has a clear syntactic break -- the draft is shown to the user, so it must
    read cleanly. High-precision set; stylistic/false-positive-prone integrity flags are excluded."""
    return any(blocker in _BROKEN_GRAMMAR for blocker in candidate_integrity_blockers(str(candidate or "")))


def _is_usable(candidate: str, paragraph: Paragraph) -> bool:
    """A rewrite is usable as a shown solution unless it is empty or a stub. This is the ONLY reason
    to fall back to the original -- because there is nothing to demonstrate. Meaning/content concerns
    do NOT block the rewrite; they are surfaced as review flags (see _review_flags)."""
    return bool(candidate) and len(candidate.split()) >= max(8, int(len(paragraph.text.split()) * 0.4))


def _review_flags(candidate: str, paragraph: Paragraph) -> list[dict[str, Any]]:
    """Concerns to surface for the user to check/edit -- NOT rejections. DraftProof shows the
    mitigation solution; the user reviews it (the before/after diff) and edits with real content."""
    flags: list[dict[str, Any]] = []
    if _severe_polarity_inversion(candidate, paragraph):
        flags.append({"added": "emphasis may have shifted",
                      "why": "The balance of your claim (e.g. 'not only X but also Y') may have changed -- confirm it still matches your argument."})
    if _severe_beat_loss(candidate, paragraph):
        flags.append({"added": "an original point may be missing",
                      "why": "Check that every point from your original paragraph is still present."})
    if _has_fabricated_specifics(candidate, paragraph.text):
        flags.append({"added": "a concrete detail was added to ground the claim",
                      "why": "DraftProof supplied an illustrative specific -- replace it with your own real example, data, or source."})
    if _has_fabricated_named_entities(candidate, paragraph.text):
        flags.append({"added": "invented names or places",
                      "why": "DraftProof added illustrative names/institutions that are not in your text -- replace them with your own real examples or remove them before submitting."})
    return flags


def run_direct_rewrite_all(
    text: str,
    *,
    source_scan: Scan | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    cancellation_check: Callable[[], None] | None = None,
    **_ignored: Any,
):
    scan = source_scan or scan_text(text)
    resolved_model = resolve_v6_model(model or writer_model()) or (model or writer_model())
    gateway = LLMGateway(LLMConfig(
        model=resolved_model,
        api_key=resolve_v6_api_key(api_key),
        base_url=resolve_v6_base_url(base_url),
        **writer_llm_profile(resolved_model, text),
        extra_body=writer_extra_body(resolved_model),
        max_retries=1,
        timeout=120,
        cancellation_check=cancellation_check,
    ))

    # Document best-of-N: the direct path is cheap (one call/paragraph), and runs vary a few points,
    # so generate N whole-document rewrites and keep the one the real detector scores lowest.
    attempts = _best_of_n()
    best_doc = None
    best_score = float("inf")
    for attempt in range(attempts):
        if cancellation_check:
            cancellation_check()
        doc = _rewrite_document_once(
            scan,
            gateway,
            _attempt_progress(progress_callback, attempt, attempts),
            cancellation_check,
        )
        score = _document_ai_risk(doc.rewritten_text) if attempts > 1 else 0.0
        if best_doc is None or score < best_score:
            best_doc, best_score = doc, score
    return best_doc


def _best_of_n() -> int:
    try:
        value = int(os.environ.get("DRAFTPROOF_V6_DIRECT_BEST_OF_N", "2"))
    except (TypeError, ValueError):
        value = 2
    return max(1, min(3, value))


def _document_ai_risk(text: str) -> float:
    """Real-detector AI likelihood (0-100) for a whole document; +inf if unscorable. Used to pick the
    best of N direct rewrites -- this is the same number reported as final_risk."""
    try:
        try:
            from poc.rewrite_v3.pipeline import _scan_report
        except ImportError:
            from rewrite_v3.pipeline import _scan_report
        report = _scan_report(text)
        badge = report.get("ai_risk_badge", {}) if isinstance(report.get("ai_risk_badge"), dict) else {}
        ai = report.get("ai_score")
        if ai is None:
            ai = badge.get("ai_likelihood_score")
        return float(ai) if ai is not None else float("inf")
    except Exception:
        return float("inf")


# The per-paragraph rewrite phase occupies the 40..78 band of the overall job
# progress (the worker sets 40 before this phase and 80 after it).
_PROGRESS_FLOOR = 40
_PROGRESS_CEIL = 78


def _section_progress(done: int, total: int) -> int:
    """Work-based percent for the per-paragraph phase.

    Reflects flagged sections COMPLETED (done/total), not a paragraph's position
    in the document, so the bar climbs evenly across the flagged set instead of
    jumping by where the flagged paragraphs happen to sit.
    """
    if total <= 0:
        return _PROGRESS_FLOOR
    span = _PROGRESS_CEIL - _PROGRESS_FLOOR
    return _PROGRESS_FLOOR + int(span * min(done, total) / total)


def _attempt_progress(
    callback: Callable[[int, str], None] | None,
    attempt: int,
    attempts: int,
) -> Callable[[int, str], None] | None:
    """Compress one best-of-N pass into its own monotonic sub-band of 40..78.

    best-of-N reruns the whole per-paragraph loop, so without this each pass would
    restart the bar at 40 and the user would see it jump backwards. Pass i maps its
    local 40..78 into [40 + span*i/N, 40 + span*(i+1)/N], so the bar only advances.
    Section counting is shown on the first pass; later passes are document-wide
    quality refinement, so they show a generic "Refining rewrite" label.
    """
    if callback is None:
        return None
    if attempts <= 1:
        return callback
    span = _PROGRESS_CEIL - _PROGRESS_FLOOR
    band_lo = _PROGRESS_FLOOR + int(span * attempt / attempts)
    band_hi = _PROGRESS_FLOOR + int(span * (attempt + 1) / attempts)

    def _wrapped(percent: int, message: str) -> None:
        clamped = max(_PROGRESS_FLOOR, min(_PROGRESS_CEIL, int(percent)))
        frac = (clamped - _PROGRESS_FLOOR) / max(1, span)
        mapped = band_lo + int((band_hi - band_lo) * frac)
        callback(mapped, message if attempt == 0 else "Refining rewrite")

    return _wrapped


def _rewrite_document_once(
    scan: Scan,
    gateway: LLMGateway,
    progress_callback: Callable[[int, str], None] | None,
    cancellation_check: Callable[[], None] | None,
):
    from .pipeline import DocumentResult  # local import: pipeline must not depend on this module

    paragraphs = list(scan.paragraphs)
    rewritten: list[str] = []
    pass_trace: list[dict[str, Any]] = []
    # Count the paragraphs that will actually be rewritten so progress reflects
    # real work (sections done / sections to do), not document position.
    flagged_total = sum(
        1 for p in paragraphs
        if findings_for_paragraph(scan, p.id) or paragraph_diagnosis(p.id)
    )
    done = 0
    for index, paragraph in enumerate(paragraphs):
        if cancellation_check:
            cancellation_check()
        findings = findings_for_paragraph(scan, paragraph.id)
        diagnosis = paragraph_diagnosis(paragraph.id)
        if not findings and not diagnosis:
            rewritten.append(paragraph.text)
            continue
        if progress_callback:
            progress_callback(
                _section_progress(done, flagged_total),
                f"Rewriting section {done + 1} of {flagged_total}",
            )
        candidate, review_items = _clean_candidate(gateway, paragraph, diagnosis, findings)
        if candidate is None:
            # No usable, grammatically-clean rewrite after retries -> show the clean original rather
            # than broken grammar. (Rare; the curated grammar set is high-precision.)
            rewritten.append(paragraph.text)
            pass_trace.append(_trace(index, paragraph.id, "source_preserved", "no_clean_rewrite", []))
        else:
            # Always show the solution; ride meaning concerns along as review flags for the user to check.
            rewritten.append(candidate)
            pass_trace.append(_trace(index, paragraph.id, "direct_llm", None, review_items + _review_flags(candidate, paragraph)))
        done += 1

    final_text = "\n\n".join(rewritten)
    return DocumentResult(
        initial_scan=scan,
        final_scan=scan_text(final_text),
        passes=[],
        rewritten_text=final_text,
        pass_trace=pass_trace,
    )


def _clean_candidate(
    gateway: LLMGateway,
    paragraph: Paragraph,
    diagnosis: dict[str, Any] | None,
    findings: list[Any],
    *,
    attempts: int = 2,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Return the first usable, grammatically-clean rewrite within `attempts`, else (None, []).

    The retry gives a one-off malformation a second chance; falling back to the clean original (vs
    shipping broken grammar) is the safety net the user reads. Meaning concerns are NOT rejected
    here -- they ride along as review flags."""
    for _ in range(max(1, attempts)):
        candidate, review_items = _rewrite_paragraph(gateway, paragraph, diagnosis, findings)
        if _is_usable(candidate or "", paragraph) and not _has_broken_grammar(candidate or ""):
            return candidate, review_items
    return None, []


def _rewrite_paragraph(
    gateway: LLMGateway,
    paragraph: Paragraph,
    diagnosis: dict[str, Any] | None,
    findings: list[Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    tags = sorted({tag for finding in findings for tag in (finding.tags or [])})
    try:
        response = gateway.chat(
            _prompt(paragraph.text, diagnosis, tags),
            system=_SYSTEM,
            temperature=0.4,
            top_p=0.9,
            # gpt-oss spends reasoning tokens that count toward this cap; a content-rich rewrite
            # plus author_review_items JSON was truncating at 1600 (finish_reason=length) -> invalid
            # JSON -> dropped -> source_preserved. Give ample headroom so the JSON always closes.
            max_tokens=4000,
            response_format={"type": "json_object"},
            app_label="DirectRewrite",
        )
        data = parse_json(getattr(response, "raw_content", "") or response.content)
    except (Exception, ValueError):
        return None, []
    if not isinstance(data, dict):
        return None, []
    rewrite = str(data.get("rewrite") or "").strip()
    review_items = [item for item in (data.get("author_review_items") or []) if isinstance(item, dict)]
    return (rewrite or None), review_items


def _trace(index: int, paragraph_id: str, source: str, reject_reason: str | None, review_items: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "pass_index": index,
        "target_paragraph_id": paragraph_id,
        "selected_source": source,
        "status": "accepted" if source != "source_preserved" else "source_preserved",
    }
    if reject_reason:
        row["reject_reason"] = reject_reason
    if review_items:
        row["author_review_items"] = review_items[:6]
    return row
