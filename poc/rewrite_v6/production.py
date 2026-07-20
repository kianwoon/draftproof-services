from __future__ import annotations

import json
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

try:
    from detect.document_structure import normalize_submitted_text
    from report.pdf import render_pdf
    from report.render_rewrite import render_rewrite_report
    from rewrite_v2.pipeline import _extract_original_text, _sentence_comparison
    from rewrite_v3.pipeline import _scan_report
    from detect.grounding_diagnosis import diagnose_grounding_gap
    from detect.critical_thinking import score_critical_thinking
    from detect.submission_risk import score_submission_risk
    from detect.policy_risk import score_policy_risk
except ModuleNotFoundError:
    from poc.detect.document_structure import normalize_submitted_text
    from poc.report.pdf import render_pdf
    from poc.report.render_rewrite import render_rewrite_report
    from poc.rewrite_v2.pipeline import _extract_original_text, _sentence_comparison
    from poc.rewrite_v3.pipeline import _scan_report
    from poc.detect.grounding_diagnosis import diagnose_grounding_gap
    from poc.detect.critical_thinking import score_critical_thinking
    from poc.detect.submission_risk import score_submission_risk
    from poc.detect.policy_risk import score_policy_risk

from .pipeline import _planner_model, _writer_model, run_v6_rewrite_all
from .direct_rewrite import direct_rewrite_enabled, run_direct_rewrite_all
from .register_coaching import build_register_coaching, register_coaching_enabled
from .llm_config import (
    resolve_v6_api_key,
    resolve_v6_base_url,
    resolve_v6_model,
    writer_extra_body,
    writer_llm_profile,
)

try:
    from report.authorship_evidence import build_authorship_evidence, preserved_idea_spans, strengthen_anchor_sentences
except ImportError:
    from poc.report.authorship_evidence import build_authorship_evidence, preserved_idea_spans, strengthen_anchor_sentences

try:
    from detect.layer3_scoring import estimate_external_detector_likelihood
except ImportError:
    from poc.detect.layer3_scoring import estimate_external_detector_likelihood
from .report_contracts import extract_paragraph_diagnoses, extract_report_signal_contracts, paragraph_diagnoses_context
# Shared internal-scan deep-scan suppressor (moved to runtime.py so direct_rewrite can reuse it
# without importing production -- that would be circular). Re-exported here under its historical
# name so the two intermediate-summary uses below are unchanged.
from .runtime import deep_scan_suppressed as _deep_scan_suppressed
from .scan import scan_text, scan_text_preserve_blocks, scan_text_with_report
from . import verdict_reframe

try:
    from poc.llm.gateway import LLMConfig, LLMGateway
except ImportError:
    from llm.gateway import LLMConfig, LLMGateway


def run_rewrite_pipeline_v6(
    *,
    detect_json: dict[str, Any],
    output_dir: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    started = time.time()

    def raise_if_canceled() -> None:
        if cancellation_check is not None:
            cancellation_check()

    def progress(percent: int, message: str) -> None:
        raise_if_canceled()
        if progress_callback:
            progress_callback(percent, message)

    # Stay at the 40 floor of the rewrite phase: the per-paragraph loop reports
    # 40..78 next, so emitting a higher value here (then dropping to 40) made the
    # progress bar jump backwards. Keep the label generic -- the default lean
    # direct path is not the legacy scanner-planner-writer pipeline.
    progress(40, "Starting rewrite")
    original_text, rewrite_source = _rewrite_source_text(detect_json)
    _concern = (detect_json.get("authorship_concern") or {})
    authorship_evidence = build_authorship_evidence(
        _concern.get("signals"),
        false_positives=detect_json.get("false_positives"),
        confidence=_concern.get("confidence", "low"),
        strengthen_examples=strengthen_anchor_sentences(detect_json),
    )
    effective_detect_json = dict(detect_json)
    effective_detect_json["input_text"] = original_text
    effective_detect_json["rewrite_source"] = rewrite_source
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    requested_writer_model = model
    resolved_writer_model = model or _writer_model()
    resolved_planner_model = _planner_model()
    max_passes = _v6_max_passes()
    source_scan = scan_text_with_report(original_text, effective_detect_json)

    # Bind the report explainer's per-paragraph diagnosis so the rewrite acts on the concrete named
    # fix. The lean direct path (flag) uses it straight; the legacy planner path translates it.
    with paragraph_diagnoses_context(extract_paragraph_diagnoses(effective_detect_json)):
        if direct_rewrite_enabled():
            document = run_direct_rewrite_all(
                original_text,
                source_scan=source_scan,
                model=model,
                api_key=api_key,
                base_url=base_url,
                progress_callback=progress,
                cancellation_check=raise_if_canceled,
                authorship_evidence=authorship_evidence,
            )
        else:
            document = run_v6_rewrite_all(
                original_text,
                max_passes=max_passes,
                model=model,
                api_key=api_key,
                base_url=base_url,
                progress_callback=progress,
                cancellation_check=raise_if_canceled,
                runtime_budget_seconds=_v6_runtime_budget_seconds(started),
                min_llm_request_seconds=_v6_min_llm_request_seconds(),
                report_signal_contracts=extract_report_signal_contracts(effective_detect_json),
                source_scan=source_scan,
            )
    stage_timings: list[dict[str, Any]] = []

    def timed_stage(name: str, percent: int, message: str, fn):
        progress(percent, message)
        t0 = time.time()
        value = fn()
        stage_timings.append({"stage": name, "seconds": round(time.time() - t0, 3)})
        return value

    final_text = document.rewritten_text
    from poc.predictability.highlight_spans import compute_predictability_highlights
    from .highlight_topk_repair import apply_highlight_topk_repair, highlight_topk_repair_enabled

    if highlight_topk_repair_enabled():
        repair_highlights = timed_stage(
            "predictability_highlight_scan",
            79,
            "Checking highlighted top-k spans",
            lambda: compute_predictability_highlights(final_text),
        )
        has_predictability_highlights = (
            bool(repair_highlights.get("actionable_sentences") or repair_highlights.get("actionable_words") or repair_highlights.get("words"))
            if isinstance(repair_highlights, dict) else False
        )
        if has_predictability_highlights:
            pre_highlight_report = _scan_report_for_summary(
                final_text,
                provided=None,
                fallback_scan=document.final_scan.to_dict(),
                deep_scan=False,  # intermediate: composite ai_score suffices for the topk-repair guard
            )
            repair_gateway = _highlight_repair_gateway(
                model=resolved_writer_model,
                api_key=api_key,
                base_url=base_url,
                text=final_text,
                cancellation_check=raise_if_canceled,
            )
            repaired_text, highlight_repair = timed_stage(
                "highlight_topk_repair",
                79,
                "Repairing highlighted top-k spans",
                lambda: apply_highlight_topk_repair(
                    final_text,
                    repair_highlights,
                    gateway=repair_gateway,
                    cancellation_check=raise_if_canceled,
                ),
            )
            repair_trace = highlight_repair.to_trace()
            if highlight_repair.changed:
                repaired_scan = scan_text_preserve_blocks(repaired_text)
                repaired_report = _scan_report_for_summary(
                    repaired_text,
                    provided=None,
                    fallback_scan=repaired_scan.to_dict(),
                    deep_scan=False,  # intermediate: compared to pre_highlight_report (also composite)
                )
                if _scan_score_worse(repaired_report, pre_highlight_report):
                    repair_trace = dict(repair_trace)
                    repair_trace.update({
                        "status": "rejected_score_regression",
                        "pre_stage_score": _report_ai_score(pre_highlight_report),
                        "candidate_score": _report_ai_score(repaired_report),
                    })
                    document = _replace_document(document, pass_trace=list(document.pass_trace) + [repair_trace])
                else:
                    final_text = repaired_text
                    document = _replace_document(
                        document,
                        rewritten_text=final_text,
                        final_scan=repaired_scan,
                        pass_trace=list(document.pass_trace) + [repair_trace],
                    )
            else:
                document = _replace_document(document, pass_trace=list(document.pass_trace) + [repair_trace])
    elapsed = time.time() - started
    original_scan_report = timed_stage(
        "original_scan_summary",
        81,
        "Preparing original scan summary",
        lambda: _scan_report_for_summary(
            original_text,
            provided=effective_detect_json,
            fallback_scan=document.initial_scan.to_dict(),
        ),
    )
    rewritten_scan_report = timed_stage(
        "rewritten_scan_summary",
        83,
        "Scanning rewritten candidate summary",
        lambda: _scan_report_for_summary(
            final_text,
            provided=None,
            fallback_scan=document.final_scan.to_dict(),
        ),
    )
    external_guard = _external_guard_decision(original_scan_report, rewritten_scan_report)
    if external_guard.get("blocked") and not _external_guard_advisory():
        # Legacy HARD-REVERT (escape hatch: DRAFTPROOF_V6_EXTERNAL_GUARD_ADVISORY=0) — discards the
        # whole rewrite back to the original. Kept only as a kill switch; it violates the objective
        # (source_preserved = the user sees no solution, learns nothing), so it is NOT the default.
        final_text = original_text
        document = _replace_document(
            document,
            rewritten_text=final_text,
            final_scan=document.initial_scan,
            pass_trace=list(document.pass_trace) + [{
                "selected_source": "external_detector_guard",
                "status": "rejected",
                "reason": external_guard.get("reason"),
                "original_external_score": external_guard.get("original_score"),
                "candidate_external_score": external_guard.get("candidate_score"),
            }],
        )
        rewritten_scan_report = original_scan_report
        status = "original_preserved_external_guard"
    else:
        # DEFAULT: ADVISORY. Guards ANNOTATE, never SUPPRESS. When the external-estimate guard would
        # have blocked, KEEP the rewrite (the shown solution) and record the concern as a review flag
        # the user verifies — instead of reverting to the original.
        if external_guard.get("blocked"):
            external_guard = {**external_guard, "advisory": True, "reverted": False}
            document = _replace_document(
                document,
                pass_trace=list(document.pass_trace) + [{
                    "selected_source": "external_detector_guard",
                    "status": "advisory",
                    "reason": external_guard.get("reason"),
                    "original_external_score": external_guard.get("original_score"),
                    "candidate_external_score": external_guard.get("candidate_score"),
                }],
            )
        status = ""
    final_text = _strip_markdown_emphasis(final_text)
    # TRUE last stage: ground generic sentences -> clean text + colour spans (green = qwen improved,
    # amber = original kept) for the frontend. Skip if the external guard reverted to the original.
    bracket_grounding_spans: list[dict[str, Any]] = []
    bracket_grounding_audit: list[dict[str, Any]] = []
    if status != "original_preserved_external_guard":
        pre_bracket_text = final_text
        final_text, bracket_grounding_spans, bracket_grounding_diag = _apply_final_bracket_grounding(
            final_text,
            model=resolved_writer_model,
            api_key=api_key,
            base_url=base_url,
            cancellation_check=raise_if_canceled,
        )
        bracket_grounding_audit = (bracket_grounding_diag or {}).get("candidates") or []
        # Bracket-grounding MUTATES the shipped text (green spans replace the writer's sentence with
        # qwen's). The authoritative scan above ran on the PRE-bracket text, so re-scan the shipped
        # bytes here -- otherwise detect_scores / final_risk / detect_scan_rewritten / external estimate
        # all describe text the user never receives (the badge would be stale by construction).
        if final_text.strip() != pre_bracket_text.strip():
            bracket_scan = scan_text_preserve_blocks(final_text)
            bracket_scan_report = _scan_report_for_summary(
                final_text,
                provided=None,
                fallback_scan=bracket_scan.to_dict(),
            )
            # Bracket-grounding is a COACHING stage: green spans are faithful (per-span fidelity /
            # grammar / fabrication gated inside _apply_final_bracket_grounding) and are NOT a badge
            # score-lever. We re-scan ONLY to keep the shipped badge HONEST on the mutated bytes -- we do
            # NOT revert on a badge tick-up. The badge is high-variance and the prior compare was
            # zero-margin strict (candidate > baseline), so reverting suppressed the green/amber coaching
            # highlights on noise alone. "Guards must ANNOTATE, never SUPPRESS" (CLAUDE.md): always ship
            # the spans; safety stays the per-span gates, not the document badge.
            document = _replace_document(
                document,
                rewritten_text=final_text,
                final_scan=bracket_scan,
            )
            rewritten_scan_report = bracket_scan_report
        # Observability: bracket-grounding runs AFTER the per-paragraph passes, so it isn't in
        # document.pass_trace yet -- record it so v6_pass_trace shows the true last stage.
        _bg_improved = sum(1 for sp in bracket_grounding_spans if sp.get("kind") == "improved")
        _bg_kept = sum(1 for sp in bracket_grounding_spans if sp.get("kind") == "kept")
        document = _replace_document(document, pass_trace=list(document.pass_trace) + [{
            "selected_source": "bracket_grounding",
            "status": "accepted" if _bg_improved else "no_change",
            "applied": _bg_improved,
            "kept": _bg_kept,
        }])
    changed = final_text.strip() != original_text.strip()
    cleared = bool(changed and not document.final_scan.findings)
    if not status:
        status = "ai_mitigated" if changed and cleared else "partial_candidate_not_strict_safe" if changed else "original_preserved"
    sentence_comparison = timed_stage(
        "sentence_comparison",
        84,
        "Comparing original and rewritten sentences",
        lambda: _sentence_comparison(original_text, final_text),
    )
    detect_scores = _detect_scores_for_summary(original_scan_report, rewritten_scan_report)
    summary = {
        "rewrite_pipeline_version": "rewrite_v6_scanner_planner_writer",
        "rewrite_engine_mode": "v6_production",
        "status": status,
        "outcome": status,
        "public_status": status,
        "strict_goal_status": status,
        "strict_safe_band_achieved": cleared and changed,
        "kpi_finalization_status": "strict_safe_auto_finalized" if cleared and changed else status,
        "rewrite_effective_config": {
            "model": resolved_writer_model,
            "requested_writer_model": requested_writer_model,
            "writer_model": resolved_writer_model,
            "planner_model": resolved_planner_model,
            "pipeline": "v6",
            "max_passes": max_passes,
            "passes": len(document.passes),
            "source_scan": "scan_json_aligned",
            "source_scan_paragraphs": len(source_scan.paragraphs),
            "source_scan_report_findings": int(source_scan.scores.get("report_finding_count") or 0),
        },
        "rewrite_source": rewrite_source,
        "candidate_generation_status": {
            # The lean direct path sets document.passes=[]; accepted paragraph rewrites live in
            # pass_trace as direct_llm/accepted entries. Fall back to that count so it isn't 0.
            "accepted_count": len(document.passes) or sum(
                1 for t in document.pass_trace
                if isinstance(t, dict) and t.get("selected_source") == "direct_llm" and t.get("status") == "accepted"
            ),
            # NOTE: this is the lightweight rewrite-v6 INTERNAL Scan count -- a DIFFERENT scanner from
            # detect_scores.rewritten_findings (the full detector report); the two can legitimately differ.
            "remaining_findings": len(document.final_scan.findings),
            "remaining_findings_scanner": "rewrite_v6_internal_scan",
            "remaining_findings_by_paragraph": _findings_by_paragraph(document.final_scan.to_dict()),
            "pass_trace": document.pass_trace,
            "stop_reason": status,
            "stage_timings": stage_timings,
        },
        "stage_timings": stage_timings,
        "v6_pass_trace": document.pass_trace,
        "v6_scores": {
            "initial": document.initial_scan.scores,
            "final": document.final_scan.scores,
        },
        "detect_scores": detect_scores,
        "original_risk": detect_scores.get("original_ai"),
        "final_risk": detect_scores.get("rewritten_ai"),
        "external_detector_guard": external_guard,
        "detect_scan_original_saved": original_scan_report,
        "detect_scan_original": original_scan_report,
        "detect_scan_rewritten": rewritten_scan_report,
        "final_text_before_quality_repair": document.final_text_before_quality_repair,
        "quality_repair": document.quality_repair.to_dict() if document.quality_repair else None,
        "final_text": final_text,
        "no_text_change": not changed,
    }
    authorship_evidence["preserved_ideas"] = preserved_idea_spans(original_text, final_text)
    summary["authorship_evidence"] = authorship_evidence
    summary["external_detector_estimate"] = _external_estimate_from_scan(rewritten_scan_report)
    # Red top-k highlighting is intentionally NOT emitted for /rewrite: it false-positives on ordinary
    # predictable phrasing (predictable != AI -- e.g. "rote memorisation rather than"), which misleads
    # the writer. The bracket-grounding colour spans below are the validated signal:
    #   green ('improved') = qwen's generated version accepted;  amber ('kept') = not accepted / original kept.
    summary["bracket_grounding_spans"] = bracket_grounding_spans
    # Gap-resolution verdict reframe (additive, kill-switched). Must run AFTER
    # bracket_grounding_spans is set so anchors_added counts the final spans.
    _attach_verdict_reframe(
        summary,
        changed=changed,
        orig_report=original_scan_report,
        new_report=rewritten_scan_report,
    )
    # Per-candidate gate audit (original, replacement, decision, scan before/after) so the green/amber
    # decision is reviewable from rewrite.json -- the stage is no longer a black box.
    summary["bracket_grounding_audit"] = bracket_grounding_audit
    # Honest register/polish coaching on the FINAL shown text (doc-level, like bracket spans). Points the
    # user at the grounded-but-polished lines + a worked plain/polished contrast from their own text. This
    # is COACHING, NOT a score lever -- it cannot and does not lower the external detector score (the copy
    # says so). Kill switch: DRAFTPROOF_V6_REGISTER_COACHING=0. Fail-open (None -> key omitted).
    if register_coaching_enabled():
        register_coaching = build_register_coaching(final_text)
        if register_coaching:
            summary["register_coaching"] = register_coaching
    # Worked teaching examples (generic claim -> grounded version -> why) generated by _apply_showcase
    # and stored on the document. These ARE the "math teacher works the problem" lesson; they were being
    # generated then discarded (never surfaced to the summary / report). Surface them so the report can
    # LEAD with the worked solution instead of only the "do it yourself" coaching. Already list[dict].
    showcase_items = getattr(document, "predictability_showcase", None)
    if showcase_items:
        summary["predictability_showcase"] = list(showcase_items)

    # Surface the direct path's per-paragraph review flags to BOTH user surfaces. The flags live
    # only in v6_pass_trace rows (author_review_items); render_rewrite + Rewrite.jsx both read
    # summary.author_review_cards, a key that only the unreachable legacy v5 producer set -- so the
    # "Author Review Cards" section was silently empty on every production rewrite. Compose it here
    # from the existing pass-trace data (inventing nothing).
    author_review_cards = _author_review_cards_from_pass_trace(document.pass_trace)
    if author_review_cards:
        summary["author_review_cards"] = author_review_cards

    result_obj = SimpleNamespace(
        summary=summary,
        sentence_comparison=sentence_comparison,
        rewrite_plan=None,
        mp_result=SimpleNamespace(
            original_text=original_text,
            final_text=final_text,
            converged=cleared and changed,
            convergence_reason=status,
            passes=document.passes,
        ),
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"draftproof_rewrite_v6_{ts}.md"
    pdf_path = out_dir / f"draftproof_rewrite_v6_{ts}.pdf"
    json_path = out_dir / f"draftproof_rewrite_v6_{ts}.json"
    md_text = timed_stage(
        "render_markdown",
        85,
        "Rendering rewrite report",
        # original_text/final_text drive the PDF's Submitted/Rewritten Content sections
        # (render_rewrite._document_section renders nothing when they're empty). The worker's
        # delivery re-render passes them (worker/app/tasks.py); this pipeline artifact is the
        # worker's FALLBACK PDF, so it must carry the content too.
        lambda: render_rewrite_report(
            summary=summary,
            sentence_comparison=sentence_comparison,
            ai_findings=[],
            verbose=False,
            original_text=original_text,
            final_text=final_text,
            # Full original scan report — its scan_intelligence.document.segments drive the
            # Submitted Content red-highlighted flagged sentences in the PDF.
            original_scan_report=effective_detect_json if isinstance(effective_detect_json, dict) else None,
        ),
    )
    md_path.write_text(md_text, encoding="utf-8")
    timed_stage(
        "render_pdf",
        87,
        "Rendering rewrite PDF",
        lambda: render_pdf(md_text, str(pdf_path)),
    )
    summary["candidate_generation_status"]["stage_timings"] = stage_timings
    summary["stage_timings"] = stage_timings
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    progress(88, "Rewrite complete")
    return {
        "status": status,
        "md_path": str(md_path),
        "pdf_path": str(pdf_path),
        "json_path": str(json_path),
        "result": result_obj,
        "elapsed": elapsed,
    }


def _rewrite_source_text(detect_json: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    scan_text = str(detect_json.get("input_text") or "").strip()
    raw_text = str(detect_json.get("raw_input_text") or "").strip()
    if scan_text:
        return scan_text, _rewrite_source_meta(
            source="scan_input_text",
            text=scan_text,
            raw_text=raw_text,
            already_normalized=bool(
                detect_json.get("input_text_normalized")
                or (raw_text and raw_text != scan_text)
            ),
        )

    extracted = _extract_original_text(detect_json)
    normalized = normalize_submitted_text(extracted) or extracted
    return normalized, _rewrite_source_meta(
        source="fallback_normalized_original_text",
        text=normalized,
        raw_text=extracted,
        already_normalized=False,
    )


def _rewrite_source_meta(
    *,
    source: str,
    text: str,
    raw_text: str,
    already_normalized: bool,
) -> dict[str, Any]:
    raw = str(raw_text or "")
    normalized = str(text or "")
    return {
        "source": source,
        "uses_scan_normalized_input": source == "scan_input_text",
        "input_text_normalized": bool(
            already_normalized
            or (raw.strip() and raw.strip() != normalized.strip())
        ),
        "raw_input_text_available": bool(raw.strip()),
        "raw_paragraph_count": _paragraph_count(raw),
        "rewrite_paragraph_count": _paragraph_count(normalized),
        "normalization_changed_text": bool(raw.strip() and raw.strip() != normalized.strip()),
    }


def _paragraph_count(text: str) -> int:
    return len([part for part in str(text or "").split("\n\n") if part.strip()])


def _scan_report_shape(scan: dict[str, Any]) -> dict[str, Any]:
    scores = scan.get("scores") if isinstance(scan.get("scores"), dict) else {}
    mean_risk = float(scores.get("mean_sentence_shape_risk") or 0.0)
    return {
        "input_text": scan.get("source_text") or "",
        "findings": {"critical": [], "high": [], "medium": [], "low": []},
        "ai_risk_badge": {
            "ai_likelihood_score": mean_risk,
            "writing_quality_score": mean_risk,
        },
    }


def _findings_by_paragraph(scan: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in scan.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        paragraph_id = str(finding.get("paragraph_id") or "")
        if paragraph_id:
            counts[paragraph_id] = counts.get(paragraph_id, 0) + 1
    return counts


def _author_review_cards_from_pass_trace(pass_trace: Any) -> list[dict[str, Any]]:
    """Compose the user-facing ``author_review_cards`` from the direct path's per-paragraph
    review items so the guard annotations actually reach BOTH surfaces (PDF
    ``render_rewrite._author_review_card_section`` + web ``Rewrite.jsx`` author-review section).

    The direct path (``direct_rewrite.py``) stores its review flags ONLY in
    ``summary.v6_pass_trace`` rows as ``author_review_items`` -- each item is a
    ``{"added": <what DraftProof added/changed>, "why": <what the author must verify/replace>}``
    dict (see ``direct_rewrite._review_flags`` + the ``author_review_items`` LLM contract in
    ``direct_prompts``). Only the unreachable legacy v5 producer set ``author_review_cards``,
    so the section was silently empty on every production rewrite. This maps the existing
    pass-trace data onto the card shape both renderers consume -- inventing nothing.

    Card fields consumed by the renderers (union):
      - ``card_id``           PDF: none; web key (Rewrite.jsx ~569)
      - ``kind``              PDF title fallback (render_rewrite 709); web key fallback (~569)
      - ``provenance``        PDF "Provenance" line (714-716); web header badge (~571)
      - ``where``             PDF: none; web header badge (~572)
      - ``instruction``       PDF title (709); web h4 title (~574)
      - ``target_text``       both guarded/optional -> omitted (no candidate-detail text available)
      - ``user_input_needed`` both guarded/optional -> omitted
      - ``author_task``       PDF "Author task" line (721-722); web note block (~587)
    Both renderers cap at 12; we cap here to match.
    """
    cards: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in pass_trace or []:
        if not isinstance(row, dict):
            continue
        items = row.get("author_review_items")
        if not isinstance(items, list) or not items:
            continue
        pid = str(row.get("target_paragraph_id") or "").strip()
        where = f"Paragraph {pid}" if pid else "Rewritten paragraph"
        for item in items:
            if not isinstance(item, dict):
                continue
            added = str(item.get("added") or "").strip()
            why = str(item.get("why") or "").strip()
            if not added and not why:
                continue
            instruction = added or why
            key = (instruction, why)
            if key in seen:
                continue
            seen.add(key)
            card: dict[str, Any] = {
                "card_id": f"{pid or 'para'}-{len(cards) + 1:02d}",
                "kind": "direct_review_flag",
                "provenance": "illustrative_content",
                "where": where,
                "instruction": instruction,
            }
            if why:
                card["author_task"] = why
            cards.append(card)
            if len(cards) >= 12:
                return cards
    return cards


def _enrich_badge_diagnoses(scan_report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Add the additive composer chain (grounding_diagnosis -> critical_thinking ->
    submission_risk -> policy_risk) to a rewrite scan report's badge, mirroring the scan
    pipeline (report.py). Without this the legacy rewrite re-scan produces a "lite" badge,
    so the rewritten column falls back to the raw detector framing instead of the policy
    scores the submitted content shows. Additive only; never changes ai_likelihood/tier."""
    if not isinstance(scan_report, dict):
        return scan_report
    badge = scan_report.get("ai_risk_badge")
    if not isinstance(badge, dict) or badge.get("policy_risk"):
        return scan_report
    try:
        tc = badge.get("transformation_classification") or {}
        gd = diagnose_grounding_gap(
            ai_components=badge.get("ai_components") or {},
            writing_components=badge.get("writing_components") or {},
            transformation_features=tc.get("features") if isinstance(tc, dict) else None,
            scored_sentence_count=None,
        )
        ctc = score_critical_thinking(grounding_diagnosis=gd, scored_sentence_count=None)
        badge["grounding_diagnosis"] = gd
        badge["critical_thinking_control"] = ctc
        badge["submission_risk"] = score_submission_risk(
            ai_likelihood_score=badge.get("ai_likelihood_score"),
            ai_tier=badge.get("tier"),
            critical_thinking_control=ctc,
            axis_scores=scan_report.get("axis_scores"),
            scored_sentence_count=None,
        )
        badge["policy_risk"] = score_policy_risk(grounding_diagnosis=gd, critical_thinking_control=ctc)
    except Exception:
        pass
    return scan_report


def _scan_report_for_summary(
    text: str,
    *,
    provided: dict[str, Any] | None,
    fallback_scan: dict[str, Any],
    deep_scan: bool = True,
) -> dict[str, Any]:
    if _has_full_report_shape(provided):
        return _enrich_badge_diagnoses(dict(provided or {}))
    try:
        # deep_scan=False on intermediate guard re-scans avoids ~3 redundant paid Modal
        # deep-scans per rewrite; the composite ai_score is enough for those better/worse checks.
        with (nullcontext() if deep_scan else _deep_scan_suppressed()):
            report = _scan_report(text)
        if _has_full_report_shape(report):
            return _enrich_badge_diagnoses(report)
    except Exception:
        pass
    return _scan_report_shape(fallback_scan)


def _scan_score_worse(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> bool:
    candidate = _report_ai_score(candidate_report)
    baseline = _report_ai_score(baseline_report)
    if candidate is None or baseline is None:
        return False
    return candidate > baseline


def _report_ai_score(report: dict[str, Any] | None) -> float | None:
    if not isinstance(report, dict):
        return None
    badge = report.get("ai_risk_badge") if isinstance(report.get("ai_risk_badge"), dict) else {}
    try:
        value = report.get("ai_score")
        if value is None:
            value = badge.get("ai_likelihood_score")
        return float(value)
    except (TypeError, ValueError):
        return None


def _highlight_repair_gateway(
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    text: str,
    cancellation_check: Callable[[], None] | None,
) -> LLMGateway:
    resolved = resolve_v6_model(model) or model
    return LLMGateway(LLMConfig(
        model=resolved,
        api_key=resolve_v6_api_key(api_key),
        base_url=resolve_v6_base_url(base_url),
        **writer_llm_profile(resolved, text),
        extra_body=writer_extra_body(resolved),
        max_retries=1,
        timeout=120,
        cancellation_check=cancellation_check,
    ))


def _v6_runtime_budget_seconds(started_at: float) -> int:
    soft_limit = _int_env("REWRITE_SOFT_TIME_LIMIT_SECONDS", 900, minimum=180, maximum=7200)
    buffer_seconds = _int_env(
        "DRAFTPROOF_REWRITE_V6_RUNTIME_SOFT_LIMIT_BUFFER_SECONDS",
        240,
        minimum=60,
        maximum=1800,
    )
    elapsed = max(0, int(time.time() - started_at))
    return max(60, soft_limit - buffer_seconds - elapsed)


def _v6_min_llm_request_seconds() -> int:
    return _int_env(
        "DRAFTPROOF_REWRITE_V6_MIN_LLM_REQUEST_SECONDS",
        180,
        minimum=30,
        maximum=900,
    )


def _v6_max_passes() -> int | None:
    import os

    raw = os.environ.get("DRAFTPROOF_V6_MAX_PASSES")
    if raw is None or not raw.strip():
        return None
    return _int_env("DRAFTPROOF_V6_MAX_PASSES", 0, minimum=1, maximum=20)


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    import os

    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    import os

    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _has_full_report_shape(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    badge = report.get("ai_risk_badge") if isinstance(report.get("ai_risk_badge"), dict) else {}
    intelligence = report.get("scan_intelligence") if isinstance(report.get("scan_intelligence"), dict) else {}
    transformation = intelligence.get("transformation") if isinstance(intelligence.get("transformation"), dict) else {}
    return bool(
        badge.get("transformation_classification")
        and (
            transformation.get("contribution")
            or transformation.get("core_signals")
        )
    )


def _external_estimate_from_scan(scan_report: dict | None) -> dict | None:
    """The rewritten content's external-detector estimate: prefer the precomputed badge value, else
    recompute from the badge's ai_components, else None (graceful for minimal/missing scans)."""
    badge = scan_report.get("ai_risk_badge") if isinstance(scan_report, dict) else None
    badge = badge if isinstance(badge, dict) else {}
    estimate = badge.get("external_detector_estimate")
    if estimate:
        return estimate
    components = badge.get("ai_components")
    if isinstance(components, dict) and components:
        return estimate_external_detector_likelihood(components)
    return None


def _external_guard_decision(original_scan: dict | None, candidate_scan: dict | None) -> dict[str, Any]:
    if not _external_guard_enabled():
        return {"blocked": False, "status": "disabled"}
    original_estimate = _external_estimate_from_scan(original_scan)
    candidate_estimate = _external_estimate_from_scan(candidate_scan)
    original_score = _external_score(original_estimate)
    candidate_score = _external_score(candidate_estimate)
    decision: dict[str, Any] = {
        "blocked": False,
        "status": "checked",
        "original_score": original_score,
        "candidate_score": candidate_score,
        "severe_threshold": _external_guard_severe_threshold(),
        "worsen_margin": _external_guard_worsen_margin(),
    }
    if candidate_score is None:
        decision["status"] = "candidate_external_unavailable"
        return decision
    if original_score is None:
        unanchored_threshold = _external_guard_unanchored_threshold()
        decision["unanchored_threshold"] = unanchored_threshold
        if candidate_score >= unanchored_threshold:
            decision.update({
                "blocked": True,
                "reason": "candidate_external_score_severe_without_original_anchor",
            })
        return decision
    if (
        candidate_score >= _external_guard_severe_threshold()
        and candidate_score - original_score >= _external_guard_worsen_margin()
    ):
        decision.update({
            "blocked": True,
            "reason": "candidate_external_score_regressed",
        })
    return decision


def _external_score(estimate: dict | None) -> float | None:
    if not isinstance(estimate, dict):
        return None
    try:
        return float(estimate.get("score"))
    except (TypeError, ValueError):
        return None


def _external_guard_enabled() -> bool:
    import os

    return os.environ.get("DRAFTPROOF_V6_EXTERNAL_GUARD", "1").strip().lower() not in {"0", "false", "no", "off"}


def _external_guard_advisory() -> bool:
    """Default TRUE: when the external-detector guard fires it ANNOTATES (a review flag) and keeps the
    rewrite, instead of reverting the whole document to the original (objective: guards annotate, never
    suppress). Set DRAFTPROOF_V6_EXTERNAL_GUARD_ADVISORY=0 to restore the legacy hard-revert."""
    import os

    return os.environ.get("DRAFTPROOF_V6_EXTERNAL_GUARD_ADVISORY", "1").strip().lower() not in {"0", "false", "no", "off"}


def _external_guard_severe_threshold() -> float:
    return _float_env("DRAFTPROOF_V6_EXTERNAL_GUARD_SEVERE_THRESHOLD", 80.0, minimum=0.0, maximum=100.0)


def _external_guard_unanchored_threshold() -> float:
    return _float_env("DRAFTPROOF_V6_EXTERNAL_GUARD_UNANCHORED_THRESHOLD", 85.0, minimum=0.0, maximum=100.0)


def _strip_markdown_emphasis(text: str) -> str:
    """Strip *…* and _…_ inline emphasis that the writer LLM emits for titles/terms.
    Only removes single-marker pairs (not bold **…** or code). Content-safe: the words
    inside the markers are preserved; only the markers are dropped."""
    import re
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", str(text or ""))
    text = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"\1", text)
    return text


def _apply_final_bracket_grounding(text, *, model, api_key, base_url, cancellation_check):
    """TRUE last-stage bracket-grounding on the final shipped text. Returns (clean_text, spans, diag)
    where spans = [{start,end,kind}] (kind 'improved' -> green, 'kept' -> amber) for the frontend, and
    diag carries the per-candidate gate audit (original/replacement/decision/scan delta). NO literal
    brackets in the text. Default OFF; (text, [], {}) on disable/failure."""
    import os
    from .bracket_grounding import apply_bracket_grounding, bracket_grounding_enabled
    if not bracket_grounding_enabled():
        return text, [], {}
    diag: dict[str, Any] = {}
    try:
        # Run on the WRITER model (gpt-oss-120b), NOT the small qwen grammar model: the experiment
        # showed qwen-7b can only lower the structural score by deleting content (telegraphic) -> the
        # fidelity floor rejects it -> ~0 good greens, while gpt-oss produces faithful, full-length
        # improvements (6/8 good greens on content11). Override with DRAFTPROOF_V6_BRACKET_MODEL.
        bracket_model = os.environ.get("DRAFTPROOF_V6_BRACKET_MODEL", "").strip() or model
        gateway = _highlight_repair_gateway(
            model=bracket_model, api_key=api_key, base_url=base_url, text=text,
            cancellation_check=cancellation_check,
        )
        new_text, spans = apply_bracket_grounding(text, gateway=gateway, cancellation_check=cancellation_check, diag=diag)
        return new_text, spans, diag
    except Exception:
        return text, [], diag


def _external_guard_worsen_margin() -> float:
    return _float_env("DRAFTPROOF_V6_EXTERNAL_GUARD_WORSEN_MARGIN", 3.0, minimum=0.0, maximum=100.0)


def _replace_document(document: Any, **changes: Any) -> Any:
    try:
        return replace(document, **changes)
    except TypeError:
        data = dict(getattr(document, "__dict__", {}))
        data.update(changes)
        return SimpleNamespace(**data)


def _attach_verdict_reframe(
    summary: dict[str, Any],
    *,
    changed: bool,
    orig_report: Any,
    new_report: Any,
) -> None:
    """Additively stamp the gap-resolution verdict fields onto the summary (scope §3/§4).

    Reuses ONLY existing detector signals already on the summary (detect_scores,
    bracket_grounding_spans) plus the two scan reports for category component deltas.
    Kill-switched: when DRAFTPROOF_REWRITE_VERDICT_REFRAME is "0"/"false" this is a NO-OP,
    so every downstream surface reverts to the legacy score-delta verdict byte-identically.
    The after-score (final_risk/original_risk) is NEVER touched (annotate-never-suppress)."""
    if not verdict_reframe.reframe_enabled():
        return
    # Fail-open (pre-push review finding on d020fafc): this is an additive
    # ANNOTATOR — a malformed detect_scores value (legacy/corrupted scan
    # payloads carry non-numeric fields) must never crash the rewrite job.
    # On any exception, leave the summary untouched (all surfaces then render
    # the legacy score-delta verdict, same as kill-switch off).
    try:
        detect_scores = summary.get("detect_scores") if isinstance(summary.get("detect_scores"), dict) else {}
        spans = summary.get("bracket_grounding_spans")
        gap = verdict_reframe.build_gap_resolution(detect_scores, spans, orig_report, new_report)
        verdict_label = verdict_reframe.compute_verdict_label(
            changed=bool(changed),
            no_text_change=bool(summary.get("no_text_change")),
            status=str(summary.get("status") or ""),
            gap=gap,
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "rewrite_v6.production: verdict reframe annotation failed; "
            "falling back to legacy verdict (additive, non-fatal)."
        )
        summary.pop("gap_resolution", None)
        summary.pop("verdict_label", None)
        summary.pop("ai_likelihood_note", None)
        return
    summary["gap_resolution"] = gap
    summary["verdict_label"] = verdict_label
    summary["ai_likelihood_note"] = verdict_reframe.AI_LIKELIHOOD_NOTE


def _detect_scores_for_summary(original_report: dict[str, Any], rewritten_report: dict[str, Any]) -> dict[str, Any]:
    original = _scan_metrics_for_summary(original_report)
    rewritten = _scan_metrics_for_summary(rewritten_report)
    original_human = original.get("human_contribution")
    rewritten_human = rewritten.get("human_contribution")
    human_shift = None
    if original_human is not None and rewritten_human is not None:
        human_shift = round(rewritten_human - original_human, 3)
    return {
        "original_ai": original.get("ai"),
        "rewritten_ai": rewritten.get("ai"),
        "original_ai_authorship": original.get("ai_authorship"),
        "rewritten_ai_authorship": rewritten.get("ai_authorship"),
        "original_human_contribution": original_human,
        "rewritten_human_contribution": rewritten_human,
        "original_ai_transformation": original.get("ai_transformation"),
        "rewritten_ai_transformation": rewritten.get("ai_transformation"),
        "original_grounding_quality_risk": original.get("grounding_quality_risk"),
        "rewritten_grounding_quality_risk": rewritten.get("grounding_quality_risk"),
        "original_findings": _finding_count(original_report.get("findings")),
        "rewritten_findings": _finding_count(rewritten_report.get("findings")),
        # these counts come from the FULL detector report (distinct from candidate_generation_status'
        # rewrite_v6 internal scan). Raw counts are NOT length-normalized: a longer rewrite that ADDS
        # grounding content can show more absolute findings while the weighted ai score falls.
        "findings_scanner": "full_detector_report",
        "human_shift_score": human_shift,
    }


def _scan_metrics_for_summary(report: dict[str, Any]) -> dict[str, float | None]:
    badge = report.get("ai_risk_badge") if isinstance(report.get("ai_risk_badge"), dict) else {}
    intelligence = report.get("scan_intelligence") if isinstance(report.get("scan_intelligence"), dict) else {}
    transformation = intelligence.get("transformation") if isinstance(intelligence.get("transformation"), dict) else {}
    contribution = transformation.get("contribution") if isinstance(transformation.get("contribution"), dict) else {}
    layers_root = report.get("integrity_layers") if isinstance(report.get("integrity_layers"), dict) else {}
    intelligence_layers = intelligence.get("integrity_layers") if isinstance(intelligence.get("integrity_layers"), dict) else {}
    layers = layers_root.get("layers") if isinstance(layers_root.get("layers"), dict) else intelligence_layers.get("layers") or {}
    human_layer = layers.get("human_contribution_signal") if isinstance(layers.get("human_contribution_signal"), dict) else {}
    ai_layer = layers.get("ai_transformation_risk") if isinstance(layers.get("ai_transformation_risk"), dict) else {}
    components = badge.get("ai_components") if isinstance(badge.get("ai_components"), dict) else {}
    return {
        "ai": _metric_percent(_first_metric(report.get("ai_score"), badge.get("ai_likelihood_score"))),
        "ai_authorship": _metric_percent(_first_metric(badge.get("ai_likelihood_score"), report.get("ai_score"))),
        "human_contribution": _metric_percent(
            _first_metric(
                contribution.get("human_contribution_ratio"),
                contribution.get("human_contribution"),
                contribution.get("human_ratio"),
                human_layer.get("score"),
            )
        ),
        "ai_transformation": _metric_percent(
            _first_metric(
                contribution.get("ai_transformation_ratio"),
                contribution.get("ai_transformation"),
                contribution.get("transformation_ratio"),
                ai_layer.get("score"),
            )
        ),
        "grounding_quality_risk": _metric_percent(
            _first_metric(
                components.get("source_grounding_risk"),
                components.get("unsupported_claim_risk"),
                components.get("citation_grounding_risk"),
                badge.get("writing_quality_score"),
                report.get("writing_score"),
            )
        ),
    }


def _first_metric(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _metric_percent(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    percent = number * 100 if abs(number) <= 1 else number
    return round(max(0.0, min(100.0, percent)), 3)


def _finding_count(findings: Any) -> int | None:
    if not isinstance(findings, dict):
        return None
    total = 0
    found = False
    for tier in ("critical", "high", "medium", "low"):
        rows = findings.get(tier)
        if isinstance(rows, list):
            found = True
            total += len(rows)
    return total if found else None
