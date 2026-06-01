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

try:
    from detect.layer3_scoring import _sentence_has_concrete_or_context, split_sentences
except ImportError:  # pragma: no cover
    from poc.detect.layer3_scoring import _sentence_has_concrete_or_context, split_sentences

from .coverage_guard import missing_required_source_beat_groups
from .integrity_guard import candidate_integrity_blockers
from .json_io import parse_json
from .llm_config import resolve_v6_api_key, resolve_v6_base_url, resolve_v6_model, writer_extra_body, writer_llm_profile, writer_model
from .report_contracts import paragraph_diagnosis
from .rewrite_playbook import playbook_entries
from .scan import Scan, findings_for_paragraph, scan_text
from .selector_diagnostics import _severe_polarity_inversion
from .text import Paragraph
try:
    from report.authorship_evidence import paragraph_authorship_targets, authorship_boost_enabled
except ImportError:
    from poc.report.authorship_evidence import paragraph_authorship_targets, authorship_boost_enabled


def direct_rewrite_enabled() -> bool:
    # Default ON (the objective-aligned path). Kill switch: set DRAFTPROOF_V6_DIRECT_REWRITE=0 to
    # fall back to the legacy planner/selector pipeline without a redeploy.
    return os.environ.get("DRAFTPROOF_V6_DIRECT_REWRITE", "1").strip().lower() not in {"0", "false", "no", "off"}


def residual_fix_enabled() -> bool:
    """Kill switch for rewrite pass 2 (paragraph-level residual checker). Default ON; set
    DRAFTPROOF_V6_RESIDUAL_FIX=0 to disable (flow reverts to rewrite -> reviewer -> scan)."""
    return os.environ.get("DRAFTPROOF_V6_RESIDUAL_FIX", "1").strip().lower() not in {"0", "false", "no", "off"}


# Grounding-aware residual trigger. scan_text (pass 2's re-scan) surfaces only STRUCTURAL tells and
# gives a pure-generic paragraph 0 findings (verified) -- so a generic leftover (e.g. a pass-1
# source_preserved paragraph) would sail through unfixed. Pass 2 therefore also consults the grounding
# signals directly. Thresholds from measured data: generic => lived_gap ~0.80 / generic_assertion
# ~0.90 (flag); grounded => lived_gap ~0.20 / generic_assertion ~0.65 (do NOT flag). Env-tunable.
_RESIDUAL_LIVED_GAP_DEFAULT = 0.60
_RESIDUAL_GENERIC_ASSERTION_DEFAULT = 0.80


def _residual_grounding_thresholds() -> tuple[float, float]:
    def _f(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        try:
            value = float(raw)
            if 0.0 <= value <= 1.0:
                return value
        except (TypeError, ValueError):
            pass
        return default
    return (_f("DRAFTPROOF_V6_RESIDUAL_LIVED_GAP", _RESIDUAL_LIVED_GAP_DEFAULT),
            _f("DRAFTPROOF_V6_RESIDUAL_GENERIC_ASSERTION", _RESIDUAL_GENERIC_ASSERTION_DEFAULT))


class _GroundingFinding:
    """Synthesized finding for a grounding-only residual (no structural tag from scan_text). Carries
    grounding tags so the writer prompt targets concrete anchors (same shape the writer expects: a
    `.tags` iterable)."""
    tags = ("low_specificity", "source_grounding")
    paragraph_id = ""


def _grounding_gap(text: str) -> bool:
    """True if the paragraph still reads as generic/ungrounded -- the blind spot scan_text misses."""
    try:
        from poc.detect.layer3_scoring import estimate_generic_assertion_risk, estimate_lived_detail_risk
    except ImportError:
        from detect.layer3_scoring import estimate_generic_assertion_risk, estimate_lived_detail_risk
    lived_gap_th, generic_assertion_th = _residual_grounding_thresholds()
    return (estimate_lived_detail_risk(text, None) >= lived_gap_th
            or estimate_generic_assertion_risk(text) >= generic_assertion_th)


def _residual_findings(residual_scan, paragraph) -> list:
    """Findings that should trigger a pass-2 re-fix: scan_text's STRUCTURAL findings, or -- when the
    paragraph is structurally clean but still GENERIC -- a synthesized grounding finding."""
    structural = findings_for_paragraph(residual_scan, paragraph.id)
    if structural:
        return structural
    if _grounding_gap(paragraph.text):
        return [_GroundingFinding()]
    return []


def _apply_residual_fix(
    doc,
    gateway,
    *,
    cancellation_check: Callable[[], None] | None,
    authorship_evidence: Any = None,
):
    """Rewrite pass 2: a paragraph-level check on the rewriter's own output.

    Re-scan the REWRITTEN draft (never the original) and re-run the writer on any paragraph the
    FRESH re-scan flags -- catching both residuals pass 1 missed and problems pass 1 introduced.
    Unflagged paragraphs keep their pass-1 text, so pass-1 gains are preserved (the load-bearing
    invariant). Flagging drives off the fresh re-scan: scan_text's STRUCTURAL findings plus a
    grounding-signal check (`_residual_findings`), because scan_text is blind to grounding gaps. We
    pass diagnosis=None to `_clean_candidate` because `paragraph_diagnosis()` is a positional-id
    ContextVar still holding the ORIGINAL diagnosis (stale-leak guard, R1). On disable/any failure
    the document is returned unchanged."""
    from .pipeline import DocumentResult

    if not residual_fix_enabled():
        return doc
    try:
        residual_scan = scan_text(doc.rewritten_text)
    except Exception:
        return doc

    paragraphs = list(residual_scan.paragraphs)
    rewritten: list[str] = []
    trace = list(doc.pass_trace)
    refixed = 0
    flagged = 0
    for index, paragraph in enumerate(paragraphs):
        if cancellation_check:
            cancellation_check()
        findings = _residual_findings(residual_scan, paragraph)
        if not findings:
            rewritten.append(paragraph.text)   # keep PASS-1 text (we scanned the rewritten draft)
            continue
        flagged += 1
        try:
            targets = (
                paragraph_authorship_targets(authorship_evidence, paragraph.text)
                if (authorship_evidence and authorship_boost_enabled())
                else {}
            )
            # diagnosis=None on purpose: fresh findings only, never stale original paragraph_diagnosis.
            candidate, review_items = _clean_candidate(
                gateway, paragraph, None, findings, authorship_targets=targets
            )
        except Exception:
            # A writer failure on one paragraph must degrade to its pass-1 text, never discard the
            # whole pass-1 result. (cancellation_check above stays outside this guard so a real
            # cancellation still propagates.)
            candidate, review_items = None, []
        if candidate is None:
            rewritten.append(paragraph.text)   # no clean residual fix -> keep pass-1 paragraph
        else:
            rewritten.append(candidate)
            refixed += 1
            trace.append(_trace(index, paragraph.id, "residual_fix", None,
                                review_items + _review_flags(candidate, paragraph)))

    if refixed == 0:
        trace.append({"selected_source": "residual_checker", "status": "checked",
                      "flagged_paragraphs": flagged, "refixed": 0})
        return DocumentResult(
            initial_scan=doc.initial_scan,
            final_scan=doc.final_scan,
            passes=doc.passes,
            rewritten_text=doc.rewritten_text,
            pass_trace=trace,
            final_text_before_quality_repair=doc.final_text_before_quality_repair,
            quality_repair=doc.quality_repair,
            naturalisation_repair=doc.naturalisation_repair,
        )

    fixed_text = "\n\n".join(rewritten)
    return DocumentResult(
        initial_scan=doc.initial_scan,
        final_scan=scan_text(fixed_text),
        passes=doc.passes,
        rewritten_text=fixed_text,
        pass_trace=trace,
        final_text_before_quality_repair=doc.final_text_before_quality_repair,
        quality_repair=doc.quality_repair,
        naturalisation_repair=doc.naturalisation_repair,
    )


_SYSTEM = (
    "You produce a SUGGESTED rewrite of a flagged paragraph for the author to review and edit. The "
    "author sees your changes in a before/after diff, so adding new content to fix the problem is "
    "expected and encouraged -- that is the mitigation. Your goal: lower AI-detection risk by making "
    "the writing specific, concrete, and human, while staying in the same subject, register, and tone "
    "as the source. Where the paragraph is generic or lacks a concrete anchor, ADD grounding AS THE "
    "AUTHOR WOULD -- you are the author's proxy. Ground EVERY generic claim in the author's "
    "FIRST-PERSON lived experience -- this is the single strongest way to cut the generic-assertion "
    "signal, so keep it on every grounded claim and NEVER trade it away for a bare figure. Then put "
    "the CONCRETE particular INSIDE that first-person statement: 'In my classroom, I have watched "
    "about a third of students ...', 'When I grade essays, I keep finding ...'. The first-person "
    "frame is the lever; the representative figure / specific scenario / what-exactly-happened rides "
    "INSIDE it to make it vivid. You need BOTH -- a bare frame ('in my classroom') with no concrete "
    "particular is weak, and a bare figure with no first-person frame loses the lever. Only where "
    "first person genuinely does not fit the register, ground with a concrete example, a specific "
    "case, or a situational (when / after / if) clause instead. The illustrative specifics (hedged "
    "figures like 'about a third', example scenarios) show the shape of a real anchor; flag each in "
    "author_review_items for the author to replace, and do NOT present a SPECIFIC named real "
    "institution, real person, or an exact statistic as a verified fact. "
    "Preserve the author's actual ARGUMENT and meaning -- do NOT flip a balanced 'not only X "
    "but also Y' into 'Y over X', and do not drop the author's existing ideas. List what you added "
    "in author_review_items so the author can confirm or replace it. "
    'Return JSON only: {"rewrite": "...", "author_review_items": [{"added": "...", "why": "..."}]}.'
)


def _prompt(paragraph_text: str, diagnosis: dict[str, Any] | None, finding_tags: list[str], authorship_targets: dict[str, Any] | None = None) -> str:
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
            "Turn every broad or generic assertion into a GROUNDED one, and do it with FIRST-PERSON "
            "framing as the primary vehicle: 'In my ..., I have seen ...', 'When I ..., I notice "
            "...'. First-person framing is the strongest generic-assertion reducer -- put it on "
            "EVERY grounded claim and never drop it. Then ride a CONCRETE particular INSIDE that "
            "frame: 'In my classroom, I have watched about a third of students stumble when ...'. The "
            "frame is the lever; the representative figure / specific scenario / what-exactly-happened "
            "makes it vivid -- you need BOTH (a bare frame is weak; a bare figure loses the lever). "
            "Only where first person truly does not fit, use a specific case ('in cases where ...'), "
            "an illustrative example ('for example, when ...'), or a when/after/if clause. The "
            "ILLUSTRATIVE specifics (hedged numbers like 'about a third', example situations) show the "
            "shape of a real anchor; flag each in author_review_items for the author to replace. Do "
            "NOT present a SPECIFIC named real institution/person or an exact statistic as a verified "
            "fact.",
            "rewrite_examples shows the SHAPE of each fix (before -> better) -- note they REPLACE, "
            "they don't add on top. Apply the same kind of transformation to THIS paragraph; do not "
            "copy the example wording.",
            "Rewrite the WHOLE paragraph. Across EVERY sentence, replace generic or predictable "
            "phrasing with concrete, specific wording. Change the sentence ROUTE, not just synonyms.",
            "Vary sentence rhythm naturally and at the source's register -- let a longer, analytical "
            "sentence sit next to a shorter, pointed one. NEVER sacrifice fluency, correctness, or "
            "sophistication for variation, and do not produce choppy or telegraphic sentences. Start "
            "sentences differently; avoid a uniform cadence. Cut hedging (may, might, can, could, "
            "should, often, generally) and generic filler.",
            "Preserve the author's actual argument and meaning, and stay in the same subject and "
            "register as the source. Do not shift a balanced 'not only X but also Y' into 'Y over X', "
            "and do not drop their existing ideas.",
            "Respect length_budget: stay close to the source length. Add a NEW sentence only when a "
            "claim genuinely cannot be grounded by rewriting an existing one. List anything you add "
            "in author_review_items.",
            "Prefer concrete, particular detail over vague generality. ILLUSTRATIVE specifics "
            "(representative figures, example cases, sample situations) are encouraged as a showcase "
            "-- list them in author_review_items so the author replaces them with their own real "
            "detail. Do NOT state a SPECIFIC named real institution/person or an exact statistic as a "
            "verified fact -- those read as real claims the author cannot defend. Use rewrite_hint "
            "only as the shape of the fix, never as wording to copy.",
        ],
    }
    authorship_targets = authorship_targets or {}
    protected = authorship_targets.get("protected_spans") or []
    grounding = authorship_targets.get("grounding_targets") or []
    if protected:
        payload["protected_spans"] = list(protected)
        payload["instructions"].append(
            "Keep every sentence in protected_spans VERBATIM -- they are the author's own voice; "
            "rewriting them only makes them more generic. Rewrite the surrounding text only."
        )
    if grounding:
        payload["grounding_targets"] = list(grounding)
        payload["instructions"].append(
            "Where you add concrete grounding, prioritise these author-owned gaps: "
            + "; ".join(grounding) + "."
        )
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


# First-person lived experience the proxy added to ground a generic claim (e.g. "in my classroom",
# "when I began teaching", "I have seen ..."). This is the author-proxy acting AS THE AUTHOR -- a
# legitimate, encouraged grounding scaffold -- but it is the proxy's invention until the author
# confirms it reflects their real experience, so it rides along as a review flag (annotate, never
# reject).
_FIRST_PERSON_EXPERIENCE = re.compile(
    r"\b(?:in my (?:own\s+)?[a-z]+"
    r"|when I\b"
    r"|I(?:'ve|'d| have| had)?\s+(?:saw|see|seen|noticed|found|observed|watched|taught|experienced|tried|tested|recall|remember|struggled|learned|began|started))\b",
    re.I,
)


def _has_added_first_person_experience(candidate: str, source_text: str) -> bool:
    """True if the candidate grounds a claim in first-person authorial experience that is NOT already
    in the source -- the proxy writing in the author's voice. Encouraged, but surfaced so the author
    confirms it matches their real experience before submitting."""
    if _FIRST_PERSON_EXPERIENCE.search(" ".join(str(source_text or "").split())):
        return False  # source is already first-person; nothing newly attributed
    return bool(_FIRST_PERSON_EXPERIENCE.search(str(candidate or "")))


def _ungrounded_claims(candidate: str, *, min_words: int = 5) -> list[str]:
    """Substantive sentences in the rewrite that are STILL generic (carry no concrete anchor) -- the
    claims only the author's own real specifics can ground. Skips short fragments. The proxy showcases
    HOW to ground; these surface as 'your turn' review items so the author finishes the rest."""
    out: list[str] = []
    for sentence in split_sentences(candidate):
        s = sentence.strip()
        if len(s.split()) >= min_words and not _sentence_has_concrete_or_context(s):
            out.append(s)
    return out


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
    if _has_added_first_person_experience(candidate, paragraph.text):
        flags.append({"added": "first-person experience written in your voice",
                      "why": "DraftProof grounded a general claim in the author's own first-person experience as a scaffold -- confirm it matches YOUR real experience, or adjust it, before submitting."})
    ungrounded = _ungrounded_claims(candidate)
    if len(ungrounded) >= 2:
        examples = "; ".join(f"“{s}”" for s in ungrounded[:2])
        flags.append({"added": "claims that still need your own specifics",
                      "why": f"{len(ungrounded)} statements here are still general -- the rewrite shows the "
                             f"style, but only your real example or experience can ground them. Add yours "
                             f"to at least: {examples}"})
    return flags


def _apply_reviewer(
    doc,
    gateway,
    *,
    cancellation_check: Callable[[], None] | None,
):
    """Run the QC reviewer on the writer's winning document, then re-scan the QC'd text.

    Order: rewrite -> QC -> scan. The reviewer fixes cross-paragraph patterns the per-paragraph
    writer can't see; the single authoritative final_scan is computed here, AFTER QC. On disable or
    any failure the writer's document is returned unchanged. Corrections apply cumulatively to the
    evolving text inside review_document, each guarded against the pre-QC baseline score."""
    from .document_reviewer import reviewer_enabled, review_document
    from .pipeline import DocumentResult

    if not reviewer_enabled():
        return doc
    try:
        result = review_document(
            doc.rewritten_text, gateway=gateway, cancellation_check=cancellation_check
        )
    except Exception:
        return doc
    if not result.corrections:
        return doc  # nothing changed; keep writer's doc + its scan

    reviewed_text = result.text
    trace = list(doc.pass_trace)
    trace.append({
        "selected_source": "qc_reviewer",
        "status": "accepted",
        "corrections": [
            {"original": c.original, "revised": c.revised} for c in result.corrections
        ][:12],
        "must_fix_unaddressed": result.must_fix_unaddressed,
        "corrections_over_cap": result.corrections_over_cap,
    })
    return DocumentResult(
        initial_scan=doc.initial_scan,
        final_scan=scan_text(reviewed_text),   # the ONE authoritative scan, post-QC
        rewritten_text=reviewed_text,
        passes=doc.passes,
        pass_trace=trace,
        final_text_before_quality_repair=doc.final_text_before_quality_repair,
    )


def run_direct_rewrite_all(
    text: str,
    *,
    source_scan: Scan | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    cancellation_check: Callable[[], None] | None = None,
    authorship_evidence: Any = None,
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
            authorship_evidence=authorship_evidence,
        )
        score = _document_ai_risk(doc.rewritten_text) if attempts > 1 else 0.0
        if best_doc is None or score < best_score:
            best_doc, best_score = doc, score
    # rewrite -> residual fix (pass 2) -> QC -> scan. Pass 2 re-scans the rewritten draft and fixes
    # paragraph-level residuals the per-paragraph writer missed or introduced; then the whole-doc
    # reviewer fixes cross-paragraph patterns; the authoritative final scan runs last, in the reviewer.
    best_doc = _apply_residual_fix(
        best_doc, gateway,
        cancellation_check=cancellation_check,
        authorship_evidence=authorship_evidence,
    )
    return _apply_reviewer(best_doc, gateway, cancellation_check=cancellation_check)


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
    authorship_evidence: Any = None,
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
        targets = (
            paragraph_authorship_targets(authorship_evidence, paragraph.text)
            if (authorship_evidence and authorship_boost_enabled())
            else {}
        )
        candidate, review_items = _clean_candidate(
            gateway, paragraph, diagnosis, findings, authorship_targets=targets
        )
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
    authorship_targets: dict[str, Any] | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Return the first usable, grammatically-clean rewrite within `attempts`, else (None, []).

    The retry gives a one-off malformation a second chance; falling back to the clean original (vs
    shipping broken grammar) is the safety net the user reads. Meaning concerns are NOT rejected
    here -- they ride along as review flags."""
    for _ in range(max(1, attempts)):
        candidate, review_items = _rewrite_paragraph(gateway, paragraph, diagnosis, findings, authorship_targets=authorship_targets)
        if _is_usable(candidate or "", paragraph) and not _has_broken_grammar(candidate or ""):
            return candidate, review_items
    return None, []


# Token budget for the writer's per-paragraph rewrite. gpt-oss spends reasoning tokens FIRST and
# they count toward this cap; a content-heavy paragraph starved at 4000 (reasoning_tokens ~3997,
# finish_reason=length) -> empty content -> EmptyLLMContentError -> source_preserved (the original
# is kept and no rewrite is shown). Same starvation class as the QC reviewer
# (DRAFTPROOF_V6_REVIEWER_MAX_TOKENS). Default 16000; tune via DRAFTPROOF_V6_WRITER_MAX_TOKENS
# without a redeploy.
_WRITER_MAX_TOKENS_DEFAULT = 16000


def _writer_max_tokens() -> int:
    raw = os.environ.get("DRAFTPROOF_V6_WRITER_MAX_TOKENS", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return _WRITER_MAX_TOKENS_DEFAULT


def _rewrite_paragraph(
    gateway: LLMGateway,
    paragraph: Paragraph,
    diagnosis: dict[str, Any] | None,
    findings: list[Any],
    authorship_targets: dict[str, Any] | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    tags = sorted({tag for finding in findings for tag in (finding.tags or [])})
    try:
        response = gateway.chat(
            _prompt(paragraph.text, diagnosis, tags, authorship_targets),
            system=_SYSTEM,
            temperature=0.4,
            top_p=0.9,
            # gpt-oss spends reasoning tokens FIRST and they count toward this cap; a content-heavy
            # paragraph starved at 4000 (reasoning_tokens ~3997, finish_reason=length) -> empty
            # content -> EmptyLLMContentError -> source_preserved. The budget must cover reasoning +
            # the rewrite JSON, so give ample headroom. Tunable via DRAFTPROOF_V6_WRITER_MAX_TOKENS.
            max_tokens=_writer_max_tokens(),
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
