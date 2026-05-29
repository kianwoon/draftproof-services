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
from .json_io import parse_json
from .llm_config import resolve_v6_api_key, resolve_v6_base_url, resolve_v6_model, writer_extra_body, writer_llm_profile, writer_model
from .report_contracts import paragraph_diagnosis
from .scan import Scan, findings_for_paragraph, scan_text
from .selector_diagnostics import _severe_polarity_inversion
from .text import Paragraph


def direct_rewrite_enabled() -> bool:
    # Default ON (the objective-aligned path). Kill switch: set DRAFTPROOF_V6_DIRECT_REWRITE=0 to
    # fall back to the legacy planner/selector pipeline without a redeploy.
    return os.environ.get("DRAFTPROOF_V6_DIRECT_REWRITE", "1").strip().lower() not in {"0", "false", "no", "off"}


_SYSTEM = (
    "You produce a SUGGESTED rewrite of a flagged paragraph for a student to review and edit. The "
    "student sees your changes in a before/after diff, so adding new content to fix the problem is "
    "expected and encouraged -- that is the mitigation. Your goal: lower AI-detection risk by making "
    "the writing specific, concrete, and human. Where the paragraph is generic or lacks a concrete "
    "anchor, ADD one: a concrete scenario, actor, setting, or illustrative example that grounds the "
    "claim. Preserve the student's actual ARGUMENT and meaning -- do NOT flip a balanced 'not only X "
    "but also Y' into 'Y over X', and do not drop the student's existing ideas. List what you added "
    "in author_review_items so the student can confirm or replace it. "
    'Return JSON only: {"rewrite": "...", "author_review_items": [{"added": "...", "why": "..."}]}.'
)


def _prompt(paragraph_text: str, diagnosis: dict[str, Any] | None, finding_tags: list[str]) -> str:
    payload: dict[str, Any] = {
        "paragraph": paragraph_text,
        "scanner_diagnosis": {
            "main_issue": diagnosis.get("main_issue"),
            "why_flagged": diagnosis.get("why_flagged"),
            "how_to_improve": diagnosis.get("recommendation"),
            "rewrite_hint_for_shape_only": diagnosis.get("rewrite_hint"),
        } if diagnosis else None,
        "flagged_issue_types": finding_tags,
        "instructions": [
            "Most AI-detection risk here comes from CONTENT that is generic and unanchored, not from "
            "word choice. So the main fix is to ADD concrete grounding: where a claim is broad, attach "
            "a specific scenario, actor, setting, mechanism, or illustrative example that makes it "
            "particular. The student will review and replace these with their own real material.",
            "Rewrite the WHOLE paragraph. Across EVERY sentence, replace generic or predictable "
            "phrasing with concrete, specific wording.",
            "Start each sentence differently; vary sentence length (mix short and long); cut hedging "
            "such as may, might, can, could, should, often, generally.",
            "Preserve the student's actual argument and meaning. Do not shift a balanced 'not only X "
            "but also Y' into 'Y over X', and do not drop their existing ideas.",
            "Where a claim is generic or unanchored, ADD a concrete grounding detail to fix it -- a "
            "specific scenario, actor, setting, or illustrative example. The student reviews every "
            "addition in the diff, so concrete suggestions are the point.",
            "Prefer concrete scenarios and examples over precise invented statistics; keep any figure "
            "clearly illustrative. List everything you add in author_review_items.",
            "Use rewrite_hint only as the shape of the fix, never as wording to copy.",
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


def _severe_beat_loss(candidate: str, paragraph: Paragraph) -> bool:
    """Only flags a source beat as genuinely DROPPED (near-zero coverage), not synonym-rephrased.
    The legacy term-exact check false-positived on faithful synonym rewrites."""
    for group in (missing_required_source_beat_groups(candidate, paragraph) or []):
        if isinstance(group, dict):
            ratio = group.get("coverage_ratio")
            if isinstance(ratio, (int, float)) and ratio < 0.15:
                return True
    return False


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
    from .pipeline import DocumentResult  # local import: pipeline must not depend on this module

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

    paragraphs = list(scan.paragraphs)
    rewritten: list[str] = []
    pass_trace: list[dict[str, Any]] = []
    total = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if cancellation_check:
            cancellation_check()
        findings = findings_for_paragraph(scan, paragraph.id)
        diagnosis = paragraph_diagnosis(paragraph.id)
        if not findings and not diagnosis:
            rewritten.append(paragraph.text)
            continue
        if progress_callback:
            progress_callback(min(78, 10 + int(68 * index / max(1, total))), f"Direct rewrite {paragraph.id}")
        candidate, review_items = _rewrite_paragraph(gateway, paragraph, diagnosis, findings)
        if not _is_usable(candidate or "", paragraph):
            candidate, retry_items = _rewrite_paragraph(gateway, paragraph, diagnosis, findings)  # one retry for a stub
            review_items = retry_items or review_items
        if not _is_usable(candidate or "", paragraph):
            # Only fall back when there is genuinely no solution to show.
            rewritten.append(paragraph.text)
            pass_trace.append(_trace(index, paragraph.id, "source_preserved", "no_usable_rewrite", []))
            continue
        # Always show the solution; ride concerns along as review flags for the user to check/edit.
        rewritten.append(candidate)
        pass_trace.append(_trace(index, paragraph.id, "direct_llm", None, review_items + _review_flags(candidate, paragraph)))

    final_text = "\n\n".join(rewritten)
    return DocumentResult(
        initial_scan=scan,
        final_scan=scan_text(final_text),
        passes=[],
        rewritten_text=final_text,
        pass_trace=pass_trace,
    )


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
            max_tokens=1600,
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
