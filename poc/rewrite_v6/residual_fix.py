"""Rewrite pass 2 (paragraph-level residual checker), split out of direct_rewrite.py to keep that
module under the repo's 1500-LOC cap. NOT a byte-identical move: this module also carries the
rewrite-targeting skip path (honoring `rewrite_decision` scoping), diffs residual-fix candidates
against the TRUE original paragraph (not the pass-1 text) for review-carding, and cards any
net-new/added paragraph unconditionally -- see `_apply_residual_fix`'s docstring below and
`rewrite_targeting.py` for the incident context (job 5bacaeb3) behind those additions.

Functions here that direct_rewrite.py re-exports (`_grounding_gap`, `_residual_findings`,
`_apply_residual_fix`) are resolved back through `direct_rewrite` at CALL TIME inside
`_apply_residual_fix` (not bound once at import time), so a monkeypatch on either module's
attribute is effective -- direct_rewrite is the single shared module both sides agree on.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from .scan import findings_for_paragraph, scan_text_preserve_blocks
try:
    from report.authorship_evidence import paragraph_authorship_targets, authorship_boost_enabled
except ImportError:
    from poc.report.authorship_evidence import paragraph_authorship_targets, authorship_boost_enabled

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


def _resolve_shared(name: str, original: Any):
    """Resolve `name` through whichever of `direct_rewrite` / `residual_fix` a test (or caller)
    actually patched, so a monkeypatch on EITHER module's attribute takes effect.

    `direct_rewrite.py` re-exports this module's helpers as a real static binding (needed so
    `direct_rewrite.X` is patchable at all), but a plain static re-export alone means a patch on
    `residual_fix.X` -- the module that actually defines and calls it -- would be silently
    ineffective. We can't just always prefer `direct_rewrite.X` either: `monkeypatch.setattr`
    restores the ORIGINAL object on teardown (not delattr), so after the first test patches
    `direct_rewrite.X`, that module's `__dict__` permanently holds a real (if unpatched) copy,
    which would then shadow any *later* test's patch applied via `residual_fix.X` instead.

    Fix: compare `direct_rewrite.X`'s current value against the ORIGINAL object identity captured
    at import time. If it differs, something intentionally repointed `direct_rewrite.X` -- use it.
    Otherwise fall back to this module's OWN current global (`globals()[name]`), which reflects any
    patch applied directly to `residual_fix.X` instead."""
    from . import direct_rewrite as _dr
    dr_value = getattr(_dr, name, original)
    if dr_value is not original:
        return dr_value
    return globals()[name]


def _residual_findings(residual_scan, paragraph) -> list:
    """Findings that should trigger a pass-2 re-fix: scan_text's STRUCTURAL findings, or -- when the
    paragraph is structurally clean but still GENERIC -- a synthesized grounding finding.

    Resolves `findings_for_paragraph` and `_grounding_gap` through `_resolve_shared` (see above) so
    a monkeypatch on either `direct_rewrite.X` or `residual_fix.X` is honored."""
    findings_for_paragraph_fn = _resolve_shared("findings_for_paragraph", _FINDINGS_FOR_PARAGRAPH_ORIGINAL)
    structural = findings_for_paragraph_fn(residual_scan, paragraph.id)
    if structural:
        return structural
    grounding_gap_fn = _resolve_shared("_grounding_gap", _GROUNDING_GAP_ORIGINAL)
    if grounding_gap_fn(paragraph.text):
        return [_GroundingFinding()]
    return []


# Original object identities, captured once at import time (before any monkeypatching), so
# `_resolve_shared` can tell "someone repointed direct_rewrite.X" apart from "still the untouched
# re-export". Must be defined AFTER the functions/imports they capture.
_GROUNDING_GAP_ORIGINAL = _grounding_gap
_FINDINGS_FOR_PARAGRAPH_ORIGINAL = findings_for_paragraph
_RESIDUAL_FINDINGS_ORIGINAL = _residual_findings


def _apply_residual_fix(
    doc,
    gateway,
    *,
    cancellation_check: Callable[[], None] | None,
    authorship_evidence: Any = None,
    lane: str = "control",
    rewrite_targeting: Any = None,
):
    """Rewrite pass 2: a paragraph-level check on the rewriter's own output.

    Re-scan the REWRITTEN draft (never the original) and re-run the writer on any paragraph the
    FRESH re-scan flags -- catching both residuals pass 1 missed and problems pass 1 introduced.
    Unflagged paragraphs keep their pass-1 text, so pass-1 gains are preserved (the load-bearing
    invariant). Flagging drives off the fresh re-scan: scan_text's STRUCTURAL findings plus a
    grounding-signal check (`_residual_findings`), because scan_text is blind to grounding gaps. We
    pass diagnosis=None to `_clean_candidate` because `paragraph_diagnosis()` is a positional-id
    ContextVar still holding the ORIGINAL diagnosis (stale-leak guard, R1). On disable/any failure
    the document is returned unchanged.

    Review flags diff each candidate against the TRUE original paragraph (poc/rewrite_v6/text.py
    doc.initial_scan), never the pass-1 text -- else anything pass-1 fabricated becomes the silent
    baseline and is never carded (2026-08-01 incident, job 5bacaeb3). A paragraph with no true-
    original counterpart (net-new/inserted) is carded unconditionally."""
    # Late imports: avoid a circular import (direct_rewrite.py imports _apply_residual_fix from
    # this module) AND resolve these names through direct_rewrite at call time, so a monkeypatch
    # on `direct_rewrite.X` (scan_text_preserve_blocks, _residual_findings, _clean_candidate, ...)
    # is honored even though the attribute is also re-exported here. direct_rewrite is the single
    # shared module both this file and its callers/tests patch.
    from . import direct_rewrite as _dr
    _clean_candidate = _dr._clean_candidate
    _review_flags = _dr._review_flags
    _trace = _dr._trace
    residual_fix_enabled = _dr.residual_fix_enabled
    scan_text_preserve_blocks = _dr.scan_text_preserve_blocks
    _residual_findings = _resolve_shared("_residual_findings", _RESIDUAL_FINDINGS_ORIGINAL)
    from .pipeline import DocumentResult
    from .rewrite_targeting import (
        original_paragraph_texts, true_original_for_index, true_original_paragraph as _true_orig_para,
        added_paragraph_review_card, shape_mismatch_added_paragraph_traces,
    )

    if not residual_fix_enabled():
        return doc
    try:
        residual_scan = scan_text_preserve_blocks(doc.rewritten_text)
    except Exception:
        return doc
    expected_paragraphs = len(getattr(doc.initial_scan, "paragraphs", []) or [])
    actual_paragraphs = len(getattr(residual_scan, "paragraphs", []) or [])
    if expected_paragraphs and actual_paragraphs != expected_paragraphs:
        from dataclasses import replace
        trace = list(doc.pass_trace) + [{
            "selected_source": "residual_checker",
            "status": "skipped_shape_mismatch",
            "expected_paragraphs": expected_paragraphs,
            "actual_paragraphs": actual_paragraphs,
        }] + shape_mismatch_added_paragraph_traces(
            original_paragraph_texts(doc), list(getattr(residual_scan, "paragraphs", []) or [])
        )
        return replace(doc, pass_trace=trace)

    originals = original_paragraph_texts(doc)
    paragraphs = list(residual_scan.paragraphs)
    rewritten: list[str] = []
    trace = list(doc.pass_trace)
    refixed = 0
    flagged = 0
    for index, paragraph in enumerate(paragraphs):
        if cancellation_check:
            cancellation_check()
        if rewrite_targeting is not None and rewrite_targeting.should_skip(paragraph.id):
            rewritten.append(paragraph.text)
            trace.append(rewrite_targeting.skipped_trace(index, paragraph.id))
            continue
        true_original = true_original_for_index(originals, index)
        if true_original is None:  # net-new/inserted paragraph -> unconditional card, see rewrite_targeting.py
            rewritten.append(paragraph.text)
            trace.append(_trace(index, paragraph.id, "residual_checker_added_paragraph", None,
                                [added_paragraph_review_card(index, paragraph.id)]))
            continue
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
                gateway, paragraph, None, findings, authorship_targets=targets, lane=lane
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
            # Diff against the TRUE original (not pass-1 text) -- see incident note above the fn.
            trace.append(_trace(index, paragraph.id, "residual_fix", None,
                                review_items + _review_flags(candidate, _true_orig_para(paragraph, true_original))))

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
        final_scan=scan_text_preserve_blocks(fixed_text),
        passes=doc.passes,
        rewritten_text=fixed_text,
        pass_trace=trace,
        final_text_before_quality_repair=doc.final_text_before_quality_repair,
        quality_repair=doc.quality_repair,
        naturalisation_repair=doc.naturalisation_repair,
    )
