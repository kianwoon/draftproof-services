from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

from poc.llm.gateway import LLMConfig, LLMGateway

from .author_proxy import attach_author_proxy_pack, author_proxy_required
from .plan import Plan, build_plan
from .progress import emit_progress as _emit_progress
from .progress import stage_progress_percent as _stage_progress_percent
from .progress import writer_progress_message as _writer_progress_message
from .prose_repair_rules import prose_repair_rule_summary, selector_agent_profile
from .repair_windows import RepairWindow, compose_window_rewrite, select_repair_window
from .report_contracts import apply_report_signal_contracts
from .runtime import acceptable_progress as _acceptable_progress
from .runtime import can_start_llm_request as _has_runtime_for_llm
from .runtime import compose as _compose
from .runtime import cross_paragraph_regression as _cross_paragraph_regression
from .runtime import improved as _improved
from .runtime import rewrite_progress_percent as _rewrite_progress_percent
from .runtime import same_text as _same_text
from .scan import Scan, findings_for_paragraph, scan_text
from .json_io import parse_json
from .llm_config import (
    planner_extra_body as _planner_extra_body,
    planner_gateway as _planner_gateway,
    planner_llm_profile as _planner_llm_profile,
    planner_model as _planner_model,
    provider_from_env as _provider_from_env,
    resolve_v6_api_key as _resolve_v6_api_key,
    resolve_v6_base_url as _resolve_v6_base_url,
    resolve_v6_model as _resolve_v6_model,
    selector_extra_body as _selector_extra_body,
    selector_gateway as _selector_gateway,
    selector_llm_profile as _selector_llm_profile,
    writer_extra_body as _writer_extra_body,
    writer_llm_profile as _writer_llm_profile,
    writer_model as _writer_model,
)
from .naturalisation import NaturalisationResult, run_naturalisation_repair_once
from .no_change_policy import no_change_retry_message, no_change_retry_status, retryable_no_change_result
from .paragraph_layout import restore_original_paragraph_layout
from .planner_llm import run_planner_llm
from .quality_repair import QualityRepairResult, _grammer_extra_body, _grammer_model, run_quality_repair_once
from .selector_diagnostics import selection_diagnostics
from .write import Variant, _annotate_selected_variant, write_variants
from .writer_feedback import needs_writer_feedback_retry, plan_with_writer_feedback


def _writer_provider(model: str) -> dict[str, Any] | None:
    return _provider_from_env("WRITER", model)


@dataclass(frozen=True)
class Result:
    scan: Scan
    plan: Plan
    variants: list[Variant]
    selected: Variant | None
    rewritten_text: str
    candidate_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        generated_variants = [variant for variant in self.variants if variant.source != "source_preserved"]
        source_variant = next((variant for variant in self.variants if variant.source == "source_preserved"), None)
        return {
            "scan": self.scan.to_dict(),
            "plan": self.plan.to_dict(),
            "variants": [asdict(variant) for variant in generated_variants],
            "source_preserved": asdict(source_variant) if source_variant else None,
            "selected": asdict(self.selected) if self.selected else None,
            "candidate_diagnostics": list(self.candidate_diagnostics),
            "rewritten_text": self.rewritten_text,
        }


@dataclass(frozen=True)
class DocumentResult:
    initial_scan: Scan
    final_scan: Scan
    passes: list[Result]
    rewritten_text: str
    pass_trace: list[dict[str, Any]]
    final_text_before_quality_repair: str | None = None
    quality_repair: QualityRepairResult | None = None
    naturalisation_repair: NaturalisationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_scan": self.initial_scan.to_dict(),
            "final_scan": self.final_scan.to_dict(),
            "passes": [result.to_dict() for result in self.passes],
            "rewritten_text": self.rewritten_text,
            "pass_trace": list(self.pass_trace),
            "final_text_before_quality_repair": self.final_text_before_quality_repair,
            "quality_repair": self.quality_repair.to_dict() if self.quality_repair else None,
            "naturalisation_repair": self.naturalisation_repair.to_dict() if self.naturalisation_repair else None,
        }


def run_v6_rewrite(
    text: str,
    *,
    planner_client: Any | None = None,
    writer_client: Any | None = None,
    selector_client: Any | None = None,
    excluded_paragraph_ids: set[str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    progress_percent: int | None = None,
    progress_end_percent: int | None = None,
    cancellation_check: Callable[[], None] | None = None,
    report_signal_contracts: list[dict[str, Any]] | None = None,
    priority_paragraph_ids: set[str] | None = None,
    source_scan: Scan | None = None,
) -> Result:
    _raise_if_canceled(cancellation_check)
    api_key = _resolve_v6_api_key(api_key)
    base_url = _resolve_v6_base_url(base_url)
    model = _resolve_v6_model(model)
    scan = source_scan or scan_text(text)
    paragraph, plan = build_plan(scan, excluded_paragraph_ids, priority_paragraph_ids)
    plan = apply_report_signal_contracts(plan, report_signal_contracts)
    target_findings = findings_for_paragraph(scan, paragraph.id)
    window = select_repair_window(paragraph, target_findings)
    if window is not None:
        return _run_v6_window_rewrite(
            scan=scan,
            paragraph=paragraph,
            parent_plan=plan,
            window=window,
            planner_client=planner_client,
            writer_client=writer_client,
            selector_client=selector_client,
            model=model,
            api_key=api_key,
            base_url=base_url,
            progress_callback=progress_callback,
            progress_percent=progress_percent,
            progress_end_percent=progress_end_percent,
            cancellation_check=cancellation_check,
            report_signal_contracts=report_signal_contracts,
        )
    return _run_v6_full_paragraph_rewrite(
        scan=scan,
        paragraph=paragraph,
        plan=plan,
        planner_client=planner_client,
        writer_client=writer_client,
        selector_client=selector_client,
        model=model,
        api_key=api_key,
        base_url=base_url,
        progress_callback=progress_callback,
        progress_percent=progress_percent,
        progress_end_percent=progress_end_percent,
        cancellation_check=cancellation_check,
    )


def run_v6_rewrite_with_residuals(
    text: str,
    *,
    priority_paragraph_ids: set[str] | None = None,
    residual_followup_passes: int | None = None,
    planner_client: Any | None = None,
    writer_client: Any | None = None,
    selector_client: Any | None = None,
    quality_client: Any | None = None,
    excluded_paragraph_ids: set[str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    cancellation_check: Callable[[], None] | None = None,
    report_signal_contracts: list[dict[str, Any]] | None = None,
    runtime_budget_seconds: float | None = None,
    min_llm_request_seconds: float = 180.0,
) -> DocumentResult:
    started_at = time.monotonic()
    initial_scan = scan_text(text)
    result = run_v6_rewrite(
        text,
        planner_client=planner_client,
        writer_client=writer_client,
        selector_client=selector_client,
        excluded_paragraph_ids=excluded_paragraph_ids,
        model=model,
        api_key=api_key,
        base_url=base_url,
        progress_callback=progress_callback,
        progress_percent=63,
        progress_end_percent=79,
        cancellation_check=cancellation_check,
        report_signal_contracts=report_signal_contracts,
        priority_paragraph_ids=priority_paragraph_ids,
    )
    passes: list[Result] = []
    pass_trace: list[dict[str, Any]] = []
    if _same_text(result.rewritten_text, text):
        pass_trace.append(
            _pass_trace_row(
                pass_index=0,
                status="no_change",
                before=initial_scan,
                target_paragraph_id=result.plan.paragraph_id,
                excluded=excluded_paragraph_ids or set(),
                candidate_diagnostics=result.candidate_diagnostics,
                selected_variant_id=result.selected.id if result.selected else None,
                selected_source=result.selected.source if result.selected else None,
            )
        )
        return DocumentResult(initial_scan=initial_scan, final_scan=initial_scan, passes=passes, rewritten_text=text, pass_trace=pass_trace)
    after = scan_text(result.rewritten_text)
    if _cross_paragraph_regression(initial_scan, after, result.plan.paragraph_id) or not _acceptable_progress(initial_scan, after, report_targeted=False):
        pass_trace.append(
            _pass_trace_row(
                pass_index=0,
                status="not_improved",
                before=initial_scan,
                after=after,
                target_paragraph_id=result.plan.paragraph_id,
                excluded=excluded_paragraph_ids or set(),
                candidate_diagnostics=result.candidate_diagnostics,
                selected_variant_id=result.selected.id if result.selected else None,
                selected_source=result.selected.source if result.selected else None,
            )
        )
        return DocumentResult(initial_scan=initial_scan, final_scan=initial_scan, passes=passes, rewritten_text=text, pass_trace=pass_trace)
    passes.append(result)
    pass_trace.append(
        _pass_trace_row(
            pass_index=0,
            status="accepted",
            before=initial_scan,
            after=after,
            target_paragraph_id=result.plan.paragraph_id,
            excluded=excluded_paragraph_ids or set(),
            candidate_diagnostics=result.candidate_diagnostics,
            selected_variant_id=result.selected.id if result.selected else None,
            selected_source=result.selected.source if result.selected else None,
        )
    )
    current = result.rewritten_text
    residual_limit = _residual_followup_limit(residual_followup_passes)
    if residual_limit > 0:
        current = _run_residual_followups(
            current=current,
            before=initial_scan,
            after=after,
            accepted_result=result,
            passes=passes,
            pass_trace=pass_trace,
            pass_index=0,
            residual_limit=residual_limit,
            planner_client=planner_client,
            writer_client=writer_client,
            selector_client=selector_client,
            model=model,
            api_key=api_key,
            base_url=base_url,
            progress_callback=progress_callback,
            progress_percent=87,
            cancellation_check=cancellation_check,
            report_signal_contracts=report_signal_contracts,
            started_at=started_at,
            runtime_budget_seconds=runtime_budget_seconds,
            min_llm_request_seconds=min_llm_request_seconds,
        )
    final_text, repair, naturalisation = _run_final_repair_layers(
        current=current,
        original_text=text,
        passes=passes,
        quality_client=quality_client,
        api_key=api_key,
        base_url=base_url,
        cancellation_check=cancellation_check,
        progress_callback=progress_callback,
    )
    return DocumentResult(
        initial_scan=initial_scan,
        final_scan=scan_text(final_text),
        passes=passes,
        rewritten_text=final_text,
        pass_trace=pass_trace,
        final_text_before_quality_repair=current if repair and repair.changed else None,
        quality_repair=repair,
        naturalisation_repair=naturalisation,
    )


def _run_v6_full_paragraph_rewrite(
    *,
    scan: Scan,
    paragraph: Any,
    plan: Plan,
    planner_client: Any | None,
    writer_client: Any | None,
    selector_client: Any | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    progress_callback: Callable[[int, str], None] | None,
    progress_percent: int | None,
    progress_end_percent: int | None,
    cancellation_check: Callable[[], None] | None,
) -> Result:
    target_findings = findings_for_paragraph(scan, paragraph.id)
    _emit_progress(progress_callback, progress_percent, f"Planning V6 paragraph {paragraph.id}")
    _raise_if_canceled(cancellation_check)
    plan = _run_planner_with_author_proxy(
        paragraph=paragraph,
        plan=plan,
        findings=target_findings,
        source_text=paragraph.text,
        planner_client=planner_client,
        writer_client=writer_client,
        model=model,
        api_key=api_key,
        base_url=base_url,
        cancellation_check=cancellation_check,
        progress_callback=progress_callback,
        progress_percent=_stage_progress_percent(progress_percent, progress_end_percent, 1),
        progress_message=f"Building V6 paragraph {paragraph.id} Author-proxy pack",
    )
    _emit_progress(progress_callback, _stage_progress_percent(progress_percent, progress_end_percent, 1), _writer_progress_message(paragraph.id, plan))
    _raise_if_canceled(cancellation_check)
    client = writer_client or LLMGateway(
        LLMConfig(
            model=model or _writer_model(),
            api_key=api_key,
            base_url=base_url,
            **_writer_llm_profile(model or _writer_model(), paragraph.text),
            provider=_writer_provider(model or _writer_model()),
            extra_body=_writer_extra_body(model or _writer_model()),
            max_retries=1,
            timeout=60,
            cancellation_check=cancellation_check,
        )
    )
    copy_blockers = _copy_blockers_from_plan(plan)
    variants = _strip_copy_blocked_restoration_sentences(
        write_variants(paragraph, plan, client=client),
        copy_blockers,
    )
    _raise_if_canceled(cancellation_check)
    _emit_progress(progress_callback, _stage_progress_percent(progress_percent, progress_end_percent, 2), f"Scoring V6 paragraph {paragraph.id} candidate")
    diagnostics = selection_diagnostics(variants, paragraph, copy_blockers=copy_blockers)
    for feedback_round in range(_writer_feedback_rounds()):
        if not _needs_writer_feedback_round(diagnostics, feedback_round):
            break
        _emit_progress(progress_callback, _stage_progress_percent(progress_percent, progress_end_percent, 2), f"Rewriting V6 paragraph {paragraph.id} with diagnostics feedback")
        plan = plan_with_writer_feedback(plan, diagnostics)
        retry_variants = _strip_copy_blocked_restoration_sentences(
            write_variants(paragraph, plan, client=client),
            copy_blockers,
        )
        variants = _combine_retry_variants(_source_preserved_variants(variants), retry_variants)
        diagnostics = selection_diagnostics(variants, paragraph, copy_blockers=copy_blockers)
    _emit_progress(progress_callback, _stage_progress_percent(progress_percent, progress_end_percent, 3), f"Selecting V6 paragraph {paragraph.id} candidate")
    selected, diagnostics = _select_variant(
        paragraph=paragraph,
        variants=variants,
        diagnostics=diagnostics,
        selector_client=selector_client or (None if writer_client is not None else _selector_gateway(api_key=api_key, base_url=base_url, cancellation_check=cancellation_check)),
    )
    return Result(scan=scan, plan=plan, variants=variants, selected=selected, rewritten_text=_compose(scan, paragraph.id, selected), candidate_diagnostics=diagnostics)


def _run_v6_window_rewrite(
    *,
    scan: Scan,
    paragraph: Any,
    parent_plan: Plan,
    window: RepairWindow,
    planner_client: Any | None,
    writer_client: Any | None,
    selector_client: Any | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    progress_callback: Callable[[int, str], None] | None,
    progress_percent: int | None,
    progress_end_percent: int | None,
    cancellation_check: Callable[[], None] | None,
    report_signal_contracts: list[dict[str, Any]] | None,
) -> Result:
    window_scan = scan_text(window.source_text)
    window_paragraph, window_plan = build_plan(window_scan, None, {"p001"})
    window_plan = _attach_window_context(window_plan, parent_plan, window)
    window_plan = apply_report_signal_contracts(window_plan, report_signal_contracts)
    _emit_progress(
        progress_callback,
        progress_percent,
        f"Planning V6 paragraph {paragraph.id} window {window.start_sentence_index + 1}-{window.end_sentence_index + 1}",
    )
    _raise_if_canceled(cancellation_check)
    window_findings = findings_for_paragraph(window_scan, window_paragraph.id)
    window_plan = _run_planner_with_author_proxy(
        paragraph=window_paragraph,
        plan=window_plan,
        findings=window_findings,
        source_text=window.source_text,
        planner_client=planner_client,
        writer_client=writer_client,
        model=model,
        api_key=api_key,
        base_url=base_url,
        cancellation_check=cancellation_check,
        progress_callback=progress_callback,
        progress_percent=_stage_progress_percent(progress_percent, progress_end_percent, 1),
        progress_message=f"Building V6 paragraph {paragraph.id} window Author-proxy pack",
    )
    _emit_progress(
        progress_callback,
        _stage_progress_percent(progress_percent, progress_end_percent, 1),
        _writer_progress_message(paragraph.id, window_plan, window=f"{window.start_sentence_index + 1}-{window.end_sentence_index + 1}"),
    )
    _raise_if_canceled(cancellation_check)
    client = writer_client or LLMGateway(
        LLMConfig(
            model=model or _writer_model(),
            api_key=api_key,
            base_url=base_url,
            **_writer_llm_profile(model or _writer_model(), window.source_text),
            provider=_writer_provider(model or _writer_model()),
            extra_body=_writer_extra_body(model or _writer_model()),
            max_retries=1,
            timeout=60,
            cancellation_check=cancellation_check,
        )
    )
    copy_blockers = _copy_blockers_from_plan(window_plan)
    variants = _strip_copy_blocked_restoration_sentences(
        write_variants(window_paragraph, window_plan, client=client),
        copy_blockers,
    )
    _raise_if_canceled(cancellation_check)
    _emit_progress(
        progress_callback,
        _stage_progress_percent(progress_percent, progress_end_percent, 2),
        f"Scoring V6 paragraph {paragraph.id} window candidate",
    )
    diagnostics = selection_diagnostics(variants, window_paragraph, copy_blockers=copy_blockers)
    for feedback_round in range(_writer_feedback_rounds()):
        if not _needs_writer_feedback_round(diagnostics, feedback_round):
            break
        _emit_progress(progress_callback, _stage_progress_percent(progress_percent, progress_end_percent, 2), f"Rewriting V6 paragraph {paragraph.id} window with diagnostics feedback")
        window_plan = plan_with_writer_feedback(window_plan, diagnostics)
        retry_variants = _strip_copy_blocked_restoration_sentences(
            write_variants(window_paragraph, window_plan, client=client),
            copy_blockers,
        )
        variants = _combine_retry_variants(_source_preserved_variants(variants), retry_variants)
        diagnostics = selection_diagnostics(variants, window_paragraph, copy_blockers=copy_blockers)
    _emit_progress(
        progress_callback,
        _stage_progress_percent(progress_percent, progress_end_percent, 3),
        f"Selecting V6 paragraph {paragraph.id} window candidate",
    )
    selected, diagnostics = _select_variant(
        paragraph=window_paragraph,
        variants=variants,
        diagnostics=diagnostics,
        selector_client=selector_client or (None if writer_client is not None else _selector_gateway(api_key=api_key, base_url=base_url, cancellation_check=cancellation_check)),
    )
    has_generated = any(variant.source != "source_preserved" for variant in variants)
    if has_generated and (selected is None or selected.source == "source_preserved"):
        _emit_progress(
            progress_callback,
            _stage_progress_percent(progress_percent, progress_end_percent, 3),
            f"Retrying full V6 paragraph {paragraph.id} after window candidate rejection",
        )
        return _run_v6_full_paragraph_rewrite(
            scan=scan,
            paragraph=paragraph,
            plan=parent_plan,
            planner_client=planner_client,
            writer_client=writer_client,
            selector_client=selector_client,
            model=model,
            api_key=api_key,
            base_url=base_url,
            progress_callback=progress_callback,
            progress_percent=progress_percent,
            progress_end_percent=progress_end_percent,
            cancellation_check=cancellation_check,
        )
    return Result(
        scan=scan,
        plan=_parent_window_plan(parent_plan, window),
        variants=variants,
        selected=selected,
        rewritten_text=compose_window_rewrite(scan.paragraphs, window, selected),
        candidate_diagnostics=diagnostics,
    )


def _combine_retry_variants(initial: list[Variant], retry: list[Variant]) -> list[Variant]:
    rows: list[Variant] = []
    seen_texts: set[str] = set()
    seen_ids: set[str] = set()
    for variant in [*initial, *retry]:
        text_key = re.sub(r"\s+", " ", str(variant.text or "").strip()).casefold()
        if not text_key or text_key in seen_texts:
            continue
        next_variant = variant
        if variant.source != "source_preserved" and variant.id in seen_ids:
            next_variant = replace(variant, id=_unique_variant_id(f"retry_{variant.id}", seen_ids))
        rows.append(next_variant)
        seen_texts.add(text_key)
        seen_ids.add(next_variant.id)
    return rows


def _source_preserved_variants(variants: list[Variant]) -> list[Variant]:
    return [variant for variant in variants if variant.source == "source_preserved"]


def _unique_variant_id(candidate: str, seen_ids: set[str]) -> str:
    value = candidate
    while value in seen_ids:
        value = f"retry_{value}"
    return value


def _copy_blockers_from_plan(plan: Plan) -> list[str]:
    decision = plan.ai_safe_route.get("llm_planner_decision") if isinstance(plan.ai_safe_route, dict) else {}
    if not isinstance(decision, dict):
        return []
    rows = decision.get("do_not_copy_route")
    if not isinstance(rows, list):
        return []
    return [str(row).strip() for row in rows if str(row).strip()]


def _writer_feedback_rounds() -> int:
    try:
        value = int(float(os.environ.get("DRAFTPROOF_V6_WRITER_FEEDBACK_ROUNDS", "1")))
    except ValueError:
        value = 1
    return max(1, min(4, value))


def _needs_writer_feedback_round(diagnostics: list[dict[str, Any]], feedback_round: int) -> bool:
    if needs_writer_feedback_retry(diagnostics):
        return True
    return feedback_round < _minimum_writer_feedback_rounds() and _best_generated_has_findings(diagnostics)


def _minimum_writer_feedback_rounds() -> int:
    try:
        value = int(float(os.environ.get("DRAFTPROOF_V6_MIN_WRITER_FEEDBACK_ROUNDS", "1")))
    except ValueError:
        value = 1
    return max(0, min(_writer_feedback_rounds(), value))


def _best_generated_has_findings(diagnostics: list[dict[str, Any]]) -> bool:
    rows = [
        row for row in diagnostics
        if row.get("source") != "source_preserved" and row.get("accepted_by_selector")
    ]
    if not rows:
        return False
    best = min(
        rows,
        key=lambda row: (
            int(row.get("candidate_findings") or 999),
            float(row.get("candidate_mean_risk") or 999.0),
            -float(row.get("risk_drop") or 0.0),
        ),
    )
    return int(best.get("candidate_findings") or 0) > 0


def _strip_copy_blocked_restoration_sentences(variants: list[Variant], copy_blockers: list[str]) -> list[Variant]:
    blockers = [
        blocker
        for blocker in copy_blockers
        if len(str(blocker or "").split()) >= 2
    ]
    if not blockers:
        return variants
    rows: list[Variant] = []
    for variant in variants:
        if variant.source == "source_preserved":
            rows.append(variant)
            continue
        cleaned = _strip_copy_blocked_sentences(variant.text, blockers)
        rows.append(replace(variant, text=cleaned) if cleaned and cleaned != variant.text else variant)
    return rows


def _strip_copy_blocked_sentences(text: str, copy_blockers: list[str]) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", str(text or "")) if sentence.strip()]
    if len(sentences) <= 2:
        return str(text or "")
    kept = [
        sentence
        for sentence in sentences
        if not _sentence_contains_copy_blocker(sentence, copy_blockers)
    ]
    if len(kept) < 2:
        return str(text or "")
    return " ".join(kept)


def _sentence_contains_copy_blocker(sentence: str, copy_blockers: list[str]) -> bool:
    lowered = str(sentence or "").casefold()
    return any(str(blocker).casefold() in lowered for blocker in copy_blockers)


def _run_planner_with_author_proxy(
    *, paragraph: Any, plan: Plan, findings: list[Any], source_text: str, planner_client: Any | None,
    writer_client: Any | None, model: str | None, api_key: str | None, base_url: str | None,
    cancellation_check: Callable[[], None] | None, progress_callback: Callable[[int, str], None] | None,
    progress_percent: int | None, progress_message: str,
) -> Plan:
    planned = _run_planner_step(paragraph=paragraph, plan=plan, findings=findings, planner_client=planner_client, writer_client=writer_client, api_key=api_key, base_url=base_url, cancellation_check=cancellation_check)
    if not author_proxy_required(planned) or writer_client is not None:
        return planned
    _emit_progress(progress_callback, progress_percent, progress_message)
    _raise_if_canceled(cancellation_check)
    writer_model = model or _writer_model()
    planned = attach_author_proxy_pack(paragraph, planned, findings, client=LLMGateway(LLMConfig(
        model=writer_model, api_key=api_key, base_url=base_url, **_writer_llm_profile(writer_model, source_text),
        provider=_writer_provider(writer_model), extra_body=_writer_extra_body(writer_model),
        max_retries=1, timeout=60, cancellation_check=cancellation_check,
    )))
    return _run_planner_step(paragraph=paragraph, plan=planned, findings=findings, planner_client=planner_client, writer_client=writer_client, api_key=api_key, base_url=base_url, cancellation_check=cancellation_check)


def _run_planner_step(
    *,
    paragraph: Any,
    plan: Plan,
    findings: list[Any],
    planner_client: Any | None,
    writer_client: Any | None,
    api_key: str | None,
    base_url: str | None,
    cancellation_check: Callable[[], None] | None,
) -> Plan:
    client = planner_client
    if client is None and writer_client is None:
        client = _planner_gateway(api_key=api_key, base_url=base_url, cancellation_check=cancellation_check)
    return plan if client is None else run_planner_llm(paragraph, plan, findings, client=client)


def _select_variant(
    *,
    paragraph: Any,
    variants: list[Variant],
    diagnostics: list[dict[str, Any]],
    selector_client: Any | None,
) -> tuple[Variant | None, list[dict[str, Any]]]:
    source = next((variant for variant in variants if variant.source == "source_preserved"), None)
    generated = [variant for variant in variants if variant.source != "source_preserved"]
    if selector_client is None:
        if generated:
            return None, _mark_selector_decision(diagnostics, None, "selector_required_missing", "")
        return source, _mark_selector_decision(diagnostics, source.id if source else None, "no_generated_variants_source_preserved", "")
    eligible_ids = {
        str(row.get("variant_id"))
        for row in diagnostics
        if row.get("accepted_by_selector") and not row.get("blockers")
    }
    if not eligible_ids:
        return source, _mark_selector_decision(diagnostics, source.id if source else None, "no_valid_generated_variants_source_preserved", "")
    selected_id, rationale = _selector_llm_choice(paragraph, variants, diagnostics, selector_client)
    metric_id = _metric_preferred_variant_id(diagnostics, eligible_ids)
    if metric_id and selected_id != metric_id:
        selected_id = metric_id
        rationale = (
            f"selector_metric_contract: selected lowest candidate_findings, then lowest candidate_mean_risk, "
            f"then highest risk_drop among eligible variants. LLM rationale was: {rationale}"
        ).strip()
    selected = _variant_by_id(variants, selected_id)
    if selected is None or selected.source == "source_preserved" or selected.id not in eligible_ids:
        return source, _mark_selector_decision(diagnostics, source.id if source else None, "invalid_selector_source_preserved", rationale)
    selected_row = next((row for row in diagnostics if row.get("variant_id") == selected.id), {})
    hard_blockers = set(selected_row.get("blockers") or [])
    if hard_blockers or not selected_row.get("accepted_by_selector"):
        return source, _mark_selector_decision(diagnostics, source.id if source else None, f"blocked_selector_source_preserved:{','.join(sorted(hard_blockers)) or 'not_accepted'}", rationale)
    return _annotate_selected_variant(selected, paragraph), _mark_selector_decision(diagnostics, selected.id, "selector_llm", rationale)


def _selector_llm_choice(
    paragraph: Any,
    variants: list[Variant],
    diagnostics: list[dict[str, Any]],
    selector_client: Any,
) -> tuple[str | None, str]:
    try:
        response = selector_client.chat(
            _selector_prompt(paragraph, variants, diagnostics),
            system="Return valid JSON only. Select an existing variant id only. Do not rewrite, edit, or create text.",
            temperature=0.0,
            top_p=1.0,
            max_tokens=2500,
            response_format={"type": "json_object"},
            app_label="Selector",
        )
        payload = parse_json(getattr(response, "raw_content", "") or response.content)
    except (Exception, ValueError):
        return None, "selector_llm_failed"
    if not isinstance(payload, dict):
        return None, "selector_payload_not_object"
    selected_id = str(payload.get("selected_id") or "").strip()
    rationale = str(payload.get("rationale") or payload.get("reason") or "").strip()
    return selected_id or None, rationale


def _metric_preferred_variant_id(diagnostics: list[dict[str, Any]], eligible_ids: set[str]) -> str | None:
    rows = [
        row for row in diagnostics
        if str(row.get("variant_id")) in eligible_ids and not row.get("blockers")
    ]
    if not rows:
        return None
    best = min(
        rows,
        key=lambda row: (
            _numeric_metric(row.get("candidate_findings"), 10_000.0),
            _numeric_metric(row.get("candidate_mean_risk"), 10_000.0),
            -_numeric_metric(row.get("risk_drop"), -10_000.0),
            str(row.get("variant_id") or ""),
        ),
    )
    return str(best.get("variant_id") or "").strip() or None


def _numeric_metric(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _selector_prompt(paragraph: Any, variants: list[Variant], diagnostics: list[dict[str, Any]]) -> str:
    generated = [variant for variant in variants if variant.source != "source_preserved"]
    rows = []
    diagnostics_by_id = {str(row.get("variant_id")): row for row in diagnostics}
    for variant in generated:
        row = diagnostics_by_id.get(variant.id, {})
        if row.get("blockers") or not row.get("accepted_by_selector"):
            continue
        rows.append({
            "id": variant.id,
            "mode": variant.mode,
            "text": variant.text,
            "blockers": row.get("blockers", []),
            "quality_warnings": row.get("quality_warnings", []),
            "candidate_findings": row.get("candidate_findings"),
            "candidate_mean_risk": row.get("candidate_mean_risk"),
            "risk_drop": row.get("risk_drop"),
            "accepted_by_selector": row.get("accepted_by_selector"),
            "missing_required_terms": row.get("missing_required_terms", []),
        })
    payload = {
        "task": "select_existing_v6_rewrite_variant",
        "selector_agent_profile": selector_agent_profile(),
        "source_paragraph": paragraph.text,
        "variants": rows,
        "rules": [
            "Return only selected_id and rationale.",
            "Select exactly one existing variant id from variants.",
            "Every variant shown is pre-validated by diagnostics; select only from the shown variants.",
            "Do not rewrite, repair, merge, or create any text.",
            "Reject variants with hard blockers: grammar corruption, sentence fragments, keyword dumps, misplaced source terms, broken lists, or malformed parallelism.",
            "Choose by metrics among the shown variants: lowest candidate_findings first, then lowest candidate_mean_risk, then highest risk_drop.",
            "All shown variants have already passed hard blockers; do not choose a worse-scoring variant for subjective polish.",
            "Prefer the variant that preserves the full source route, reads directly, and needs only light punctuation or flow cleanup.",
            "Reject or downgrade variants with forced connectors such as Consequently, Thus, Similarly, Ultimately, or Moreover when those connectors make the paragraph sound engineered.",
            "Reject or downgrade variants that change a possible risk into something already happening when the source frames it as only possible.",
            "Downgrade clunky wording such as 'by enabling them to', vague danger openings, and unnecessarily formal substitutions when a plainer variant is available.",
            "Do not select a variant with more candidate_findings than another shown variant.",
            "If candidate_findings tie, do not select a variant with higher candidate_mean_risk than another shown variant.",
        ],
        "global_prose_repair_rules": prose_repair_rule_summary(),
        "output_schema": {"selected_id": "existing variant id", "rationale": "brief reason for the selected existing variant"},
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _variant_by_id(variants: list[Variant], selected_id: str | None) -> Variant | None:
    if not selected_id:
        return None
    return next((variant for variant in variants if variant.id == selected_id), None)


def _mark_selector_decision(diagnostics: list[dict[str, Any]], selected_id: str | None, source: str, rationale: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in diagnostics:
        updated = dict(row)
        updated["selected_by_selector"] = bool(selected_id and row.get("variant_id") == selected_id)
        updated["selector_source"] = source
        if updated["selected_by_selector"] and rationale:
            updated["selector_rationale"] = rationale
        rows.append(updated)
    return rows


def run_v6_rewrite_all(
    text: str,
    *,
    planner_client: Any | None = None,
    writer_client: Any | None = None,
    selector_client: Any | None = None,
    quality_client: Any | None = None,
    max_passes: int | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    cancellation_check: Callable[[], None] | None = None,
    runtime_budget_seconds: float | None = None,
    min_llm_request_seconds: float = 180.0,
    report_signal_contracts: list[dict[str, Any]] | None = None,
    residual_followup_passes: int | None = None,
    source_scan: Scan | None = None,
) -> DocumentResult:
    started_at = time.monotonic()
    initial_scan = source_scan or scan_text(text)
    current = text
    passes: list[Result] = []
    pass_trace: list[dict[str, Any]] = []
    limit = max_passes if max_passes is not None else _dynamic_pass_limit(initial_scan, report_signal_contracts)
    attempts: dict[str, int] = {}
    exhausted: set[str] = set()
    covered: set[str] = set()
    residual_limit = _residual_followup_limit(residual_followup_passes)
    for pass_index in range(limit):
        _raise_if_canceled(cancellation_check)
        before = scan_text(current)
        report_target_ids = _report_target_paragraph_ids(before, report_signal_contracts)
        if not before.findings and not report_target_ids:
            break
        finding_paragraph_ids = _finding_paragraph_ids(before) | report_target_ids
        if finding_paragraph_ids and len(exhausted & finding_paragraph_ids) >= len(finding_paragraph_ids):
            break
        excluded_for_pass = _coverage_exclusions(
            finding_paragraph_ids=finding_paragraph_ids,
            exhausted=exhausted,
            covered=covered,
        )
        if not (finding_paragraph_ids - exhausted - covered):
            covered.clear()
        start_percent = _rewrite_progress_percent(pass_index, limit)
        _emit_progress(
            progress_callback,
            start_percent,
            f"V6 rewrite pass {pass_index + 1}: {len(before.findings)} finding(s) remaining",
        )
        if not _has_runtime_for_llm(
            started_at=started_at,
            runtime_budget_seconds=runtime_budget_seconds,
            min_llm_request_seconds=min_llm_request_seconds,
        ):
            _emit_progress(progress_callback, start_percent, "V6 runtime budget reached before starting another LLM request")
            pass_trace.append(
                _pass_trace_row(
                    pass_index=pass_index,
                    status="runtime_budget_reached",
                    before=before,
                    excluded=excluded_for_pass,
                )
            )
            break
        end_percent = _rewrite_progress_percent(pass_index + 1, limit)
        source_result = run_v6_rewrite(
            text,
            planner_client=planner_client,
            writer_client=writer_client,
            selector_client=selector_client,
            excluded_paragraph_ids=excluded_for_pass,
            model=model,
            api_key=api_key,
            base_url=base_url,
            progress_callback=progress_callback,
            progress_percent=start_percent,
            progress_end_percent=end_percent,
            cancellation_check=cancellation_check,
            report_signal_contracts=report_signal_contracts,
            priority_paragraph_ids=finding_paragraph_ids - exhausted,
            source_scan=source_scan,
        )
        result = _project_source_result_to_current(source_result, before)
        if _same_text(result.rewritten_text, current):
            attempts[result.plan.paragraph_id] = attempts.get(result.plan.paragraph_id, 0) + 1
            retryable_no_change = retryable_no_change_result(result)
            finding_target = result.plan.paragraph_id in finding_paragraph_ids
            will_retry = (retryable_no_change or finding_target) and attempts[result.plan.paragraph_id] < _no_change_retry_limit()
            if not will_retry:
                exhausted.add(result.plan.paragraph_id)
            pass_trace.append(
                _pass_trace_row(
                    pass_index=pass_index,
                    status=no_change_retry_status(result, will_retry=will_retry),
                    before=before,
                    target_paragraph_id=result.plan.paragraph_id,
                    excluded=excluded_for_pass,
                    candidate_diagnostics=result.candidate_diagnostics,
                    selected_variant_id=result.selected.id if result.selected else None,
                    selected_source=result.selected.source if result.selected else None,
                )
            )
            _emit_progress(progress_callback, end_percent, no_change_retry_message(result.plan.paragraph_id, will_retry=will_retry))
            continue
        after = scan_text(result.rewritten_text)
        result_targets = _target_paragraph_ids_after_rewrite(before, after, result)
        if (
            not _target_paragraph_acceptance_ok(before, after, result.plan.paragraph_id, result_targets)
            or _cross_paragraph_regression(before, after, result.plan.paragraph_id, target_after_ids=result_targets)
            or not _acceptable_progress(before, after, report_targeted=result.plan.paragraph_id in report_target_ids)
        ):
            attempts[result.plan.paragraph_id] = attempts.get(result.plan.paragraph_id, 0) + 1
            if attempts[result.plan.paragraph_id] >= _not_improved_retry_limit():
                exhausted.add(result.plan.paragraph_id)
            pass_trace.append(
                _pass_trace_row(
                    pass_index=pass_index,
                    status="not_improved",
                    before=before,
                    after=after,
                    target_paragraph_id=",".join(sorted(result_targets)),
                    excluded=excluded_for_pass,
                    candidate_diagnostics=result.candidate_diagnostics,
                    selected_variant_id=result.selected.id if result.selected else None,
                    selected_source=result.selected.source if result.selected else None,
                )
            )
            _emit_progress(progress_callback, end_percent, f"V6 paragraph {result.plan.paragraph_id} did not improve")
            continue
        attempts[result.plan.paragraph_id] = 0
        exhausted.discard(result.plan.paragraph_id)
        covered.add(result.plan.paragraph_id)
        passes.append(result)
        current = result.rewritten_text
        pass_trace.append(
            _pass_trace_row(
                pass_index=pass_index,
                status="accepted",
                before=before,
                after=after,
                target_paragraph_id=",".join(sorted(result_targets)),
                excluded=excluded_for_pass,
                candidate_diagnostics=result.candidate_diagnostics,
                selected_variant_id=result.selected.id if result.selected else None,
                selected_source=result.selected.source if result.selected else None,
            )
        )
        _emit_progress(progress_callback, end_percent, f"Accepted V6 paragraph {result.plan.paragraph_id}")
        if residual_limit > 0:
            current = _run_residual_followups(
                current=current,
                before=before,
                after=after,
                accepted_result=result,
                passes=passes,
                pass_trace=pass_trace,
                pass_index=pass_index,
                residual_limit=residual_limit,
                planner_client=planner_client,
                writer_client=writer_client,
                selector_client=selector_client,
                model=model,
                api_key=api_key,
                base_url=base_url,
                progress_callback=progress_callback,
                progress_percent=end_percent,
                cancellation_check=cancellation_check,
                report_signal_contracts=report_signal_contracts,
                started_at=started_at,
                runtime_budget_seconds=runtime_budget_seconds,
                min_llm_request_seconds=min_llm_request_seconds,
            )
    final_text, repair, naturalisation = _run_final_repair_layers(
        current=current,
        original_text=text,
        passes=passes,
        quality_client=quality_client,
        api_key=api_key,
        base_url=base_url,
        cancellation_check=cancellation_check,
        progress_callback=progress_callback,
    )
    return DocumentResult(
        initial_scan=initial_scan,
        final_scan=scan_text(final_text),
        passes=passes,
        rewritten_text=final_text,
        pass_trace=pass_trace,
        final_text_before_quality_repair=current if repair and repair.changed else None,
        quality_repair=repair,
        naturalisation_repair=naturalisation,
    )


def _run_residual_followups(
    *,
    current: str,
    before: Scan,
    after: Scan,
    accepted_result: Result,
    passes: list[Result],
    pass_trace: list[dict[str, Any]],
    pass_index: int,
    residual_limit: int,
    planner_client: Any | None,
    writer_client: Any | None,
    selector_client: Any | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    progress_callback: Callable[[int, str], None] | None,
    progress_percent: int,
    cancellation_check: Callable[[], None] | None,
    report_signal_contracts: list[dict[str, Any]] | None,
    started_at: float,
    runtime_budget_seconds: float | None,
    min_llm_request_seconds: float,
) -> str:
    target_ids = _target_paragraph_ids_after_rewrite(before, after, accepted_result)
    exhausted_targets: set[str] = set()
    residual_index = 0
    max_attempts = max(residual_limit, residual_limit * max(1, len(target_ids)))
    while residual_index < max_attempts:
        _raise_if_canceled(cancellation_check)
        before_residual = scan_text(current)
        active_targets = (target_ids & _finding_paragraph_ids(before_residual)) - exhausted_targets
        if not active_targets:
            break
        if not _has_runtime_for_llm(started_at=started_at, runtime_budget_seconds=runtime_budget_seconds, min_llm_request_seconds=min_llm_request_seconds):
            pass_trace.append(_residual_trace_row(pass_index, residual_index, "runtime_budget_reached_residual", before_residual, None, sorted(active_targets)))
            break
        _emit_progress(progress_callback, progress_percent, f"V6 residual pass for {', '.join(sorted(active_targets))}")
        result = run_v6_rewrite(
            current,
            planner_client=planner_client,
            writer_client=writer_client,
            selector_client=selector_client,
            excluded_paragraph_ids={paragraph.id for paragraph in before_residual.paragraphs if paragraph.id not in active_targets},
            model=model,
            api_key=api_key,
            base_url=base_url,
            progress_callback=progress_callback,
            progress_percent=progress_percent,
            progress_end_percent=min(87, progress_percent + 3),
            cancellation_check=cancellation_check,
            report_signal_contracts=report_signal_contracts,
            priority_paragraph_ids=active_targets,
        )
        if _same_text(result.rewritten_text, current):
            retryable_no_change = retryable_no_change_result(result)
            status = "no_change_retryable_residual" if retryable_no_change else "no_change_residual"
            if not retryable_no_change:
                exhausted_targets.add(result.plan.paragraph_id)
            pass_trace.append(_residual_trace_row(pass_index, residual_index, status, before_residual, None, sorted(active_targets), result))
            residual_index += 1
            continue
        after_residual = scan_text(result.rewritten_text)
        result_targets = _target_paragraph_ids_after_rewrite(before_residual, after_residual, result)
        if _cross_paragraph_regression(before_residual, after_residual, result.plan.paragraph_id, target_after_ids=result_targets) or not _acceptable_progress(before_residual, after_residual, report_targeted=False):
            exhausted_targets.add(result.plan.paragraph_id)
            pass_trace.append(_residual_trace_row(pass_index, residual_index, "not_improved_residual", before_residual, after_residual, sorted(active_targets), result))
            residual_index += 1
            continue
        passes.append(result)
        pass_trace.append(_residual_trace_row(pass_index, residual_index, "accepted_residual", before_residual, after_residual, sorted(result_targets), result))
        current = result.rewritten_text
        target_ids = (target_ids - {result.plan.paragraph_id}) | result_targets
        exhausted_targets.clear()
        residual_index += 1
    return current


def _run_final_repair_layers(
    *,
    current: str,
    original_text: str,
    passes: list[Result],
    quality_client: Any | None,
    api_key: str | None,
    base_url: str | None,
    cancellation_check: Callable[[], None] | None,
    progress_callback: Callable[[int, str], None] | None,
) -> tuple[str, QualityRepairResult | None, NaturalisationResult | None]:
    repair = run_quality_repair_once(
        current,
        original_text=original_text,
        quality_client=quality_client,
        api_key=api_key,
        base_url=base_url,
        cancellation_check=cancellation_check,
        progress_callback=progress_callback,
    )
    repair = _risk_safe_quality_repair(current, repair)
    post_quality_text = repair.repaired_text if repair else current
    naturalisation = run_naturalisation_repair_once(
        post_quality_text,
        original_text=original_text,
        quality_client=quality_client,
        api_key=api_key,
        base_url=base_url,
        cancellation_check=cancellation_check,
        progress_callback=progress_callback,
    )
    naturalisation = _risk_safe_naturalisation_repair(post_quality_text, naturalisation)
    post_naturalisation_text = naturalisation.repaired_text if naturalisation else post_quality_text
    final_text = restore_original_paragraph_layout(original_text, post_naturalisation_text, passes)
    return final_text, repair, naturalisation


def _risk_safe_quality_repair(
    original: str,
    repair: QualityRepairResult | None,
) -> QualityRepairResult | None:
    if repair is None or not repair.changed:
        return repair
    before = scan_text(original)
    after = scan_text(repair.repaired_text)
    before_count = float(before.scores.get("finding_count") or 0.0)
    after_count = float(after.scores.get("finding_count") or 0.0)
    before_risk = float(before.scores.get("mean_sentence_shape_risk") or 0.0)
    after_risk = float(after.scores.get("mean_sentence_shape_risk") or 0.0)
    if after_count > before_count or after_risk > before_risk + _quality_repair_risk_tolerance():
        return replace(
            repair,
            repaired_text=original,
            status="reverted_scan_regression",
            skipped_operations=[
                *repair.skipped_operations,
                {
                    "skip_reason": "scan_regression",
                    "before_findings": before_count,
                    "after_findings": after_count,
                    "before_mean_sentence_shape_risk": before_risk,
                    "after_mean_sentence_shape_risk": after_risk,
                },
            ],
        )
    return repair


def _risk_safe_naturalisation_repair(
    original: str,
    repair: NaturalisationResult | None,
) -> NaturalisationResult | None:
    if repair is None or not repair.changed:
        return repair
    before = scan_text(original)
    after = scan_text(repair.repaired_text)
    before_count = float(before.scores.get("finding_count") or 0.0)
    after_count = float(after.scores.get("finding_count") or 0.0)
    before_risk = float(before.scores.get("mean_sentence_shape_risk") or 0.0)
    after_risk = float(after.scores.get("mean_sentence_shape_risk") or 0.0)
    if after_count > before_count or after_risk > before_risk + _quality_repair_risk_tolerance():
        return replace(
            repair,
            repaired_text=original,
            status="reverted_scan_regression",
            skipped_operations=[
                *repair.skipped_operations,
                {
                    "skip_reason": "scan_regression",
                    "before_findings": before_count,
                    "after_findings": after_count,
                    "before_mean_sentence_shape_risk": before_risk,
                    "after_mean_sentence_shape_risk": after_risk,
                },
            ],
        )
    return repair


def _quality_repair_risk_tolerance() -> float:
    import os

    try:
        value = float(os.environ.get("DRAFTPROOF_V6_GRAMMER_RISK_TOLERANCE", "1.0"))
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(5.0, value))


def _finding_paragraph_ids(scan: Scan) -> set[str]:
    return {finding.paragraph_id for finding in scan.findings}


def _no_change_retry_limit() -> int:
    return _bounded_int_env("DRAFTPROOF_V6_NO_CHANGE_RETRY_LIMIT", 2, minimum=1, maximum=4)


def _not_improved_retry_limit() -> int:
    return _bounded_int_env("DRAFTPROOF_V6_NOT_IMPROVED_RETRY_LIMIT", 2, minimum=1, maximum=4)


def _dynamic_pass_limit(scan: Scan, report_signal_contracts: list[dict[str, Any]] | None = None) -> int:
    active_paragraphs = _finding_paragraph_ids(scan) | _report_target_paragraph_ids(scan, report_signal_contracts)
    if not active_paragraphs:
        return 1
    planned = min(
        _dynamic_max_passes(),
        max(
            len(active_paragraphs),
            _dynamic_min_passes(),
            len(active_paragraphs) * _attempts_per_finding_paragraph(),
        ),
    )
    return max(len(active_paragraphs), planned)


def _attempts_per_finding_paragraph() -> int:
    return _bounded_int_env("DRAFTPROOF_V6_ATTEMPTS_PER_FINDING_PARAGRAPH", 2, minimum=1, maximum=4)


def _dynamic_min_passes() -> int:
    return _bounded_int_env("DRAFTPROOF_V6_MIN_DYNAMIC_PASSES", 3, minimum=1, maximum=12)


def _dynamic_max_passes() -> int:
    return _bounded_int_env("DRAFTPROOF_V6_MAX_DYNAMIC_PASSES", 12, minimum=1, maximum=24)


def _bounded_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    import os

    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _residual_followup_limit(value: int | None) -> int:
    if value is not None:
        return max(0, min(3, int(value)))
    import os

    raw = os.environ.get("DRAFTPROOF_V6_RESIDUAL_FOLLOWUP_PASSES", "0")
    try:
        return max(0, min(3, int(raw)))
    except ValueError:
        return 0


def _target_paragraph_ids_after_rewrite(before: Scan, after: Scan, result: Result) -> set[str]:
    target = next((paragraph for paragraph in before.paragraphs if paragraph.id == result.plan.paragraph_id), None)
    if target is None:
        return {result.plan.paragraph_id}
    selected_text = result.selected.text if result.selected else ""
    selected_terms = _match_terms(selected_text)
    if not selected_terms:
        return {after.paragraphs[min(target.index, len(after.paragraphs) - 1)].id} if after.paragraphs else {result.plan.paragraph_id}
    delta = len(after.paragraphs) - len(before.paragraphs)
    expected_children = max(1, len([block for block in re.split(r"\n\s*\n+", selected_text.strip()) if block.strip()]))
    if delta > 0 and after.paragraphs:
        child_count = max(expected_children, delta + 1)
        child_end = min(len(after.paragraphs), target.index + child_count)
        return {paragraph.id for paragraph in after.paragraphs[target.index:child_end]}
    start = max(0, target.index - 1)
    end = min(len(after.paragraphs), target.index + max(1, delta + 1, expected_children) + 2)
    rows: list[tuple[int, str]] = []
    for paragraph in after.paragraphs[start:end]:
        terms = _match_terms(paragraph.text)
        if not terms:
            continue
        overlap = len(terms & selected_terms) / max(1, min(len(terms), len(selected_terms)))
        if overlap >= 0.35:
            rows.append((paragraph.index, paragraph.id))
    if rows:
        return {paragraph_id for _, paragraph_id in rows}
    return {after.paragraphs[min(target.index, len(after.paragraphs) - 1)].id} if after.paragraphs else {result.plan.paragraph_id}


def _project_source_result_to_current(source_result: Result, current_scan: Scan) -> Result:
    """Compose an original-context paragraph candidate into current text."""
    return replace(
        source_result,
        scan=current_scan,
        rewritten_text=_compose(current_scan, source_result.plan.paragraph_id, source_result.selected),
    )


def _target_paragraph_acceptance_ok(
    before: Scan,
    after: Scan,
    target_paragraph_id: str,
    target_after_ids: set[str],
) -> bool:
    before_counts = _finding_counts_by_paragraph(before)
    after_counts = _finding_counts_by_paragraph(after)
    before_target_count = before_counts.get(target_paragraph_id, 0)
    after_target_count = sum(after_counts.get(paragraph_id, 0) for paragraph_id in target_after_ids)
    if after_target_count > before_target_count:
        return False
    if after_target_count < before_target_count:
        return True
    return _paragraph_mean_severity(after, target_after_ids) <= _paragraph_mean_severity(before, {target_paragraph_id})


def _paragraph_mean_severity(scan: Scan, paragraph_ids: set[str]) -> float:
    values = [
        float(finding.severity)
        for finding in scan.findings
        if finding.paragraph_id in paragraph_ids
    ]
    return sum(values) / len(values) if values else 0.0


def _residual_trace_row(
    pass_index: int,
    residual_index: int,
    status: str,
    before: Scan,
    after: Scan | None,
    target_ids: list[str],
    result: Result | None = None,
) -> dict[str, Any]:
    row = _pass_trace_row(
        pass_index=pass_index,
        status=status,
        before=before,
        after=after,
        target_paragraph_id=",".join(target_ids),
        excluded=set(),
        candidate_diagnostics=result.candidate_diagnostics if result else None,
        selected_variant_id=result.selected.id if result and result.selected else None,
        selected_source=result.selected.source if result and result.selected else None,
    )
    row["residual_followup"] = True
    row["residual_index"] = residual_index + 1
    return row


def _report_target_paragraph_ids(scan: Scan, contracts: list[dict[str, Any]] | None) -> set[str]:
    excerpts = _report_target_excerpts(contracts)
    if not excerpts:
        return set()
    targets: set[str] = set()
    for paragraph in scan.paragraphs:
        paragraph_terms = _match_terms(paragraph.text)
        if not paragraph_terms:
            continue
        for excerpt in excerpts:
            excerpt_terms = _match_terms(excerpt)
            if not excerpt_terms:
                continue
            shared = paragraph_terms & excerpt_terms
            if len(shared) >= 8 or len(shared) / max(1, min(len(paragraph_terms), len(excerpt_terms))) >= 0.45:
                targets.add(paragraph.id)
                break
    return targets


def _report_target_excerpts(contracts: list[dict[str, Any]] | None) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for contract in contracts or []:
        if not isinstance(contract, dict):
            continue
        for excerpt in contract.get("target_excerpts") or []:
            text = " ".join(str(excerpt).split())
            key = text.casefold()
            if text and key not in seen:
                rows.append(text)
                seen.add(key)
    return rows


def _match_terms(text: str) -> set[str]:
    stop = {"about", "after", "again", "because", "between", "could", "every", "from", "have", "into", "more", "only", "should", "their", "there", "these", "those", "through", "which", "while", "would"}
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text or ""))
        if len(token) >= 4 and token.casefold() not in stop
    }


def _coverage_exclusions(
    *,
    finding_paragraph_ids: set[str],
    exhausted: set[str],
    covered: set[str],
) -> set[str]:
    eligible_uncovered = finding_paragraph_ids - exhausted - covered
    excluded = set(exhausted)
    if eligible_uncovered:
        excluded.update(finding_paragraph_ids & covered)
    return excluded


def _pass_trace_row(
    *,
    pass_index: int,
    status: str,
    before: Scan,
    after: Scan | None = None,
    target_paragraph_id: str | None = None,
    excluded: set[str] | None = None,
    candidate_diagnostics: list[dict[str, Any]] | None = None,
    selected_variant_id: str | None = None,
    selected_source: str | None = None,
) -> dict[str, Any]:
    row = {
        "pass_index": pass_index + 1,
        "status": status,
        "target_paragraph_id": target_paragraph_id,
        "excluded_paragraph_ids": sorted(excluded or set()),
        "before_findings": int(before.scores.get("finding_count") or 0),
        "before_mean_sentence_shape_risk": before.scores.get("mean_sentence_shape_risk"),
        "before_findings_by_paragraph": _finding_counts_by_paragraph(before),
    }
    if selected_variant_id or selected_source:
        row["selected_variant_id"] = selected_variant_id
        row["selected_source"] = selected_source
    if candidate_diagnostics is not None:
        row["candidate_diagnostics"] = candidate_diagnostics[:5]
    if after is not None:
        row.update(
            {
                "after_findings": int(after.scores.get("finding_count") or 0),
                "after_mean_sentence_shape_risk": after.scores.get("mean_sentence_shape_risk"),
                "after_findings_by_paragraph": _finding_counts_by_paragraph(after),
            }
        )
    return row


def _finding_counts_by_paragraph(scan: Scan) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in scan.findings:
        counts[finding.paragraph_id] = counts.get(finding.paragraph_id, 0) + 1
    return counts


def _attach_window_context(plan: Plan, parent_plan: Plan, window: RepairWindow) -> Plan:
    route = dict(plan.ai_safe_route)
    route["repair_window"] = _window_payload(window)
    route["parent_paragraph_id"] = parent_plan.paragraph_id
    route["parent_route_goal"] = parent_plan.route_goal
    route["parent_paragraph_strategy"] = parent_plan.paragraph_strategy
    strategy = dict(plan.paragraph_strategy)
    strategy["repair_window"] = _window_payload(window)
    strategy["window_instruction"] = (
        "Repair only this overloaded sentence window. Preserve the local source meaning, "
        "but do not copy the parent paragraph's packed list route."
    )
    return replace(plan, paragraph_strategy=strategy, ai_safe_route=route)


def _parent_window_plan(parent_plan: Plan, window: RepairWindow) -> Plan:
    route = dict(parent_plan.ai_safe_route)
    route["repair_window"] = _window_payload(window)
    strategy = dict(parent_plan.paragraph_strategy)
    strategy["repair_window"] = _window_payload(window)
    strategy["window_instruction"] = (
        "The selected rewrite changed only this sentence window inside the target paragraph."
    )
    return replace(parent_plan, paragraph_strategy=strategy, ai_safe_route=route)


def _window_payload(window: RepairWindow) -> dict[str, Any]:
    return {
        "paragraph_id": window.paragraph_id,
        "start_sentence_index": window.start_sentence_index,
        "end_sentence_index": window.end_sentence_index,
        "source_sentence_ids": list(window.source_sentence_ids),
        "finding_count": window.finding_count,
        "max_severity": round(window.max_severity, 3),
        "source_word_count": len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", window.source_text)),
    }


def _raise_if_canceled(cancellation_check: Callable[[], None] | None) -> None:
    if cancellation_check is not None:
        cancellation_check()
