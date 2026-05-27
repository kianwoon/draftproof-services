from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

from poc.llm.gateway import LLMConfig, LLMGateway

from .plan import Plan, build_plan
from .planner_llm import run_planner_llm
from .repair_windows import RepairWindow, compose_window_rewrite, select_repair_window
from .report_contracts import apply_report_signal_contracts
from .scan import Scan, findings_for_paragraph, scan_text
from .json_io import parse_json
from .paragraph_architecture import apply_architecture_split_text, architecture_split_contract
from .selector_diagnostics import rejected_variant_feedback, selection_diagnostics
from .prose_quality import repair_generated_prose
from .write import Variant, choose_variant, parse_variants, write_variants
from .writer_prompt import build_retry_contract


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_scan": self.initial_scan.to_dict(),
            "final_scan": self.final_scan.to_dict(),
            "passes": [result.to_dict() for result in self.passes],
            "rewritten_text": self.rewritten_text,
            "pass_trace": list(self.pass_trace),
        }


def run_v6_rewrite(
    text: str,
    *,
    planner_client: Any | None = None,
    writer_client: Any | None = None,
    excluded_paragraph_ids: set[str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    progress_percent: int | None = None,
    cancellation_check: Callable[[], None] | None = None,
    report_signal_contracts: list[dict[str, Any]] | None = None,
    priority_paragraph_ids: set[str] | None = None,
) -> Result:
    _raise_if_canceled(cancellation_check)
    scan = scan_text(text)
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
            model=model,
            api_key=api_key,
            base_url=base_url,
            progress_callback=progress_callback,
            progress_percent=progress_percent,
            cancellation_check=cancellation_check,
            report_signal_contracts=report_signal_contracts,
        )
    return _run_v6_full_paragraph_rewrite(
        scan=scan,
        paragraph=paragraph,
        plan=plan,
        planner_client=planner_client,
        writer_client=writer_client,
        model=model,
        api_key=api_key,
        base_url=base_url,
        progress_callback=progress_callback,
        progress_percent=progress_percent,
        cancellation_check=cancellation_check,
    )


def run_v6_rewrite_with_residuals(
    text: str,
    *,
    priority_paragraph_ids: set[str] | None = None,
    residual_followup_passes: int | None = None,
    planner_client: Any | None = None,
    writer_client: Any | None = None,
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
        excluded_paragraph_ids=excluded_paragraph_ids,
        model=model,
        api_key=api_key,
        base_url=base_url,
        progress_callback=progress_callback,
        progress_percent=63,
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
    return DocumentResult(initial_scan=initial_scan, final_scan=scan_text(current), passes=passes, rewritten_text=current, pass_trace=pass_trace)


def _run_v6_full_paragraph_rewrite(
    *,
    scan: Scan,
    paragraph: Any,
    plan: Plan,
    planner_client: Any | None,
    writer_client: Any | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    progress_callback: Callable[[int, str], None] | None,
    progress_percent: int | None,
    cancellation_check: Callable[[], None] | None,
) -> Result:
    target_findings = findings_for_paragraph(scan, paragraph.id)
    _emit_progress(progress_callback, progress_percent, f"Planning V6 paragraph {paragraph.id}")
    _raise_if_canceled(cancellation_check)
    if planner_client is not None or writer_client is None:
        plan = run_planner_llm(
            paragraph,
            plan,
            target_findings,
            client=planner_client or _planner_gateway(
                api_key=api_key,
                base_url=base_url,
                cancellation_check=cancellation_check,
            ),
        )
    _emit_progress(progress_callback, progress_percent, f"Writing V6 paragraph {paragraph.id}")
    _raise_if_canceled(cancellation_check)
    client = writer_client or LLMGateway(
        LLMConfig(
            model=model or _writer_model(),
            api_key=api_key,
            base_url=base_url,
            **_writer_llm_profile(model or _writer_model(), paragraph.text),
            provider=_writer_provider(model or _writer_model()),
            extra_body=_writer_extra_body(model or _writer_model()),
            cancellation_check=cancellation_check,
        )
    )
    variants = write_variants(paragraph, plan, client=client)
    _raise_if_canceled(cancellation_check)
    _emit_progress(progress_callback, progress_percent, f"Scanning V6 paragraph {paragraph.id} candidate")
    variants, selected = _select_with_retry(
        paragraph,
        plan,
        variants,
        client=client,
        cancellation_check=cancellation_check,
    )
    diagnostics = selection_diagnostics(variants, paragraph)
    return Result(scan=scan, plan=plan, variants=variants, selected=selected, rewritten_text=_compose(scan, paragraph.id, selected), candidate_diagnostics=diagnostics)


def _run_v6_window_rewrite(
    *,
    scan: Scan,
    paragraph: Any,
    parent_plan: Plan,
    window: RepairWindow,
    planner_client: Any | None,
    writer_client: Any | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    progress_callback: Callable[[int, str], None] | None,
    progress_percent: int | None,
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
    if planner_client is not None or writer_client is None:
        window_plan = run_planner_llm(
            window_paragraph,
            window_plan,
            findings_for_paragraph(window_scan, window_paragraph.id),
            client=planner_client or _planner_gateway(
                api_key=api_key,
                base_url=base_url,
                cancellation_check=cancellation_check,
            ),
        )
        window_plan = _attach_window_context(window_plan, parent_plan, window)
    _emit_progress(
        progress_callback,
        progress_percent,
        f"Writing V6 paragraph {paragraph.id} window {window.start_sentence_index + 1}-{window.end_sentence_index + 1}",
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
            cancellation_check=cancellation_check,
        )
    )
    variants = write_variants(window_paragraph, window_plan, client=client)
    _raise_if_canceled(cancellation_check)
    _emit_progress(
        progress_callback,
        progress_percent,
        f"Scanning V6 paragraph {paragraph.id} window candidate",
    )
    variants, selected = _select_with_retry(
        window_paragraph,
        window_plan,
        variants,
        client=client,
        cancellation_check=cancellation_check,
    )
    diagnostics = selection_diagnostics(variants, window_paragraph)
    has_generated = any(variant.source != "source_preserved" for variant in variants)
    if has_generated and (selected is None or selected.source == "source_preserved"):
        return _run_v6_full_paragraph_rewrite(
            scan=scan,
            paragraph=paragraph,
            plan=parent_plan,
            planner_client=planner_client,
            writer_client=writer_client,
            model=model,
            api_key=api_key,
            base_url=base_url,
            progress_callback=progress_callback,
            progress_percent=progress_percent,
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


def _select_with_retry(
    paragraph: Any,
    plan: Plan,
    variants: list[Variant],
    *,
    client: Any,
    cancellation_check: Callable[[], None] | None,
) -> tuple[list[Variant], Variant | None]:
    _raise_if_canceled(cancellation_check)
    selected = choose_variant(variants, paragraph)
    if selected is None or selected.source != "source_preserved":
        return variants, selected
    feedback = rejected_variant_feedback(variants, paragraph)
    if not feedback:
        return variants, selected
    _raise_if_canceled(cancellation_check)
    try:
        response = client.chat(
            _retry_prompt(paragraph, plan, feedback),
            system="Return valid JSON only with a variants array.",
            temperature=0.12,
            top_p=0.75,
            max_tokens=None,
            response_format={"type": "json_object"},
            app_label="writer",
        )
        split_contract = architecture_split_contract(paragraph, plan)
        retry_variants = [
            replace(variant, text=repair_generated_prose(apply_architecture_split_text(variant.text, split_contract), paragraph.text))
            for variant in parse_variants(parse_json(getattr(response, "raw_content", "") or response.content))
        ]
    except (Exception, ValueError):
        retry_variants = []
    if not retry_variants:
        return variants, selected
    combined = [*variants, *retry_variants]
    return combined, choose_variant(combined, paragraph)


def _retry_prompt(paragraph: Any, plan: Plan, feedback: list[dict[str, Any]]) -> str:
    normalized_feedback = []
    for row in feedback[:3]:
        blockers = [
            "malformed_fragment_or_trace_sentence" if blocker == "fragment_or_trace_sentence" else blocker
            for blocker in row.get("blockers", [])
        ]
        normalized_feedback.append({
            "variant_id": row.get("variant_id"),
            "blockers": blockers,
            "quality_warnings": row.get("quality_warnings", []),
            "missing_required_terms": row.get("missing_required_terms", [])[:12],
            "candidate_findings": row.get("candidate_findings"),
            "candidate_mean_risk": row.get("candidate_mean_risk"),
        })
    payload = {
        "task": "retry_rejected_v6_writer_candidate",
        "instruction": "Generate one corrected replacement that fixes every listed defect while preserving submitted meaning and source coverage.",
        "defect_feedback_label": "Defect feedback from rejected candidates",
        "defect_feedback": normalized_feedback,
        "retry_writer_contract": build_retry_contract(paragraph, plan),
        "retry_rules": [
            "Fix the listed blockers directly; do not repeat the rejected sentence route.",
            "Use source_units as the only source text and keep coverage in the selected paragraph.",
            "Use writer_execution_contract rows as the build order.",
            "Do not output route fragments, coverage maps, sentence rows, analysis notes, or repair traces.",
            "Do not add external facts, new citations, new named references, new years, or stronger claims.",
            "Keep submitted citations inside their original parenthetical span when cited.",
            "Every final sentence must be a complete ordinary sentence with its own subject and predicate.",
        ],
        "output_schema": {"variants": [{"id": "retry_v1", "text": "corrected replacement text only", "author_proxy_provenance": [], "author_review_items": []}]},
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def run_v6_rewrite_all(
    text: str,
    *,
    planner_client: Any | None = None,
    writer_client: Any | None = None,
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
) -> DocumentResult:
    started_at = time.monotonic()
    initial_scan = scan_text(text)
    current = text
    passes: list[Result] = []
    pass_trace: list[dict[str, Any]] = []
    limit = max_passes if max_passes is not None else max(1, len(initial_scan.paragraphs) * 5)
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
        result = run_v6_rewrite(
            current,
            planner_client=planner_client,
            writer_client=writer_client,
            excluded_paragraph_ids=excluded_for_pass,
            model=model,
            api_key=api_key,
            base_url=base_url,
            progress_callback=progress_callback,
            progress_percent=start_percent,
            cancellation_check=cancellation_check,
            report_signal_contracts=report_signal_contracts,
            priority_paragraph_ids=finding_paragraph_ids - exhausted,
        )
        end_percent = _rewrite_progress_percent(pass_index + 1, limit)
        if _same_text(result.rewritten_text, current):
            attempts[result.plan.paragraph_id] = attempts.get(result.plan.paragraph_id, 0) + 1
            exhausted.add(result.plan.paragraph_id)
            pass_trace.append(
                _pass_trace_row(
                    pass_index=pass_index,
                    status="no_change",
                    before=before,
                    target_paragraph_id=result.plan.paragraph_id,
                    excluded=excluded_for_pass,
                    candidate_diagnostics=result.candidate_diagnostics,
                    selected_variant_id=result.selected.id if result.selected else None,
                    selected_source=result.selected.source if result.selected else None,
                )
            )
            _emit_progress(progress_callback, end_percent, f"V6 paragraph {result.plan.paragraph_id} made no change")
            continue
        after = scan_text(result.rewritten_text)
        result_targets = _target_paragraph_ids_after_rewrite(before, after, result)
        if _cross_paragraph_regression(before, after, result.plan.paragraph_id, target_after_ids=result_targets) or not _acceptable_progress(before, after, report_targeted=result.plan.paragraph_id in report_target_ids):
            attempts[result.plan.paragraph_id] = attempts.get(result.plan.paragraph_id, 0) + 1
            if attempts[result.plan.paragraph_id] >= 4:
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
    return DocumentResult(initial_scan=initial_scan, final_scan=scan_text(current), passes=passes, rewritten_text=current, pass_trace=pass_trace)


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
            excluded_paragraph_ids={paragraph.id for paragraph in before_residual.paragraphs if paragraph.id not in active_targets},
            model=model,
            api_key=api_key,
            base_url=base_url,
            progress_callback=progress_callback,
            progress_percent=progress_percent,
            cancellation_check=cancellation_check,
            report_signal_contracts=report_signal_contracts,
            priority_paragraph_ids=active_targets,
        )
        if _same_text(result.rewritten_text, current):
            exhausted_targets.add(result.plan.paragraph_id)
            pass_trace.append(_residual_trace_row(pass_index, residual_index, "no_change_residual", before_residual, None, sorted(active_targets), result))
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


def _finding_paragraph_ids(scan: Scan) -> set[str]:
    return {finding.paragraph_id for finding in scan.findings}


def _residual_followup_limit(value: int | None) -> int:
    if value is not None:
        return max(0, min(3, int(value)))
    import os

    raw = os.environ.get("DRAFTPROOF_V6_RESIDUAL_FOLLOWUP_PASSES", "1")
    try:
        return max(0, min(3, int(raw)))
    except ValueError:
        return 1


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
    if candidate_diagnostics:
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


def _has_runtime_for_llm(
    *,
    started_at: float,
    runtime_budget_seconds: float | None,
    min_llm_request_seconds: float,
) -> bool:
    if runtime_budget_seconds is None or runtime_budget_seconds <= 0:
        return True
    elapsed = time.monotonic() - started_at
    remaining = float(runtime_budget_seconds) - elapsed
    return remaining >= max(1.0, float(min_llm_request_seconds or 0.0))


def _compose(scan: Scan, target_paragraph_id: str, selected: Variant | None) -> str:
    blocks = []
    for paragraph in scan.paragraphs:
        blocks.append(selected.text if selected and paragraph.id == target_paragraph_id else paragraph.text)
    return "\n\n".join(blocks)


def _same_text(left: str, right: str) -> bool:
    return re.sub(r"\s+", " ", str(left or "").strip()) == re.sub(r"\s+", " ", str(right or "").strip())


def _improved(before: Scan, after: Scan) -> bool:
    return (
        after.scores["finding_count"] < before.scores["finding_count"]
        or after.scores["mean_sentence_shape_risk"] < before.scores["mean_sentence_shape_risk"]
    )


def _acceptable_progress(before: Scan, after: Scan, *, report_targeted: bool) -> bool:
    if _improved(before, after):
        return True
    return bool(report_targeted) and after.scores["finding_count"] <= before.scores["finding_count"] and after.scores["mean_sentence_shape_risk"] <= before.scores["mean_sentence_shape_risk"]


def _cross_paragraph_regression(
    before: Scan,
    after: Scan,
    target_paragraph_id: str,
    *,
    target_after_ids: set[str] | None = None,
) -> bool:
    before_counts = _paragraph_counts(before)
    after_counts = _paragraph_counts(after)
    target_after_ids = target_after_ids or {target_paragraph_id}
    target = next((paragraph for paragraph in before.paragraphs if paragraph.id == target_paragraph_id), None)
    if target is not None and len(after.paragraphs) != len(before.paragraphs):
        delta = len(after.paragraphs) - len(before.paragraphs)
        target_after_count = sum(after_counts.get(paragraph_id, 0) for paragraph_id in target_after_ids)
        target_drop = before_counts.get(target_paragraph_id, 0) - target_after_count
        total_drop = int(before.scores.get("finding_count") or 0) - int(after.scores.get("finding_count") or 0)
        max_non_target_regression = 0
        for paragraph in before.paragraphs:
            if paragraph.id == target_paragraph_id:
                continue
            after_index = paragraph.index if paragraph.index < target.index else paragraph.index + delta
            if 0 <= after_index < len(after.paragraphs):
                after_id = after.paragraphs[after_index].id
                max_non_target_regression = max(max_non_target_regression, after_counts.get(after_id, 0) - before_counts.get(paragraph.id, 0))
                if after_id in target_after_ids:
                    continue
                if after_counts.get(after_id, 0) > before_counts.get(paragraph.id, 0) and not (target_drop >= 2 and total_drop > 0 and max_non_target_regression <= 1):
                    return True
        return False
    return any(
        paragraph_id not in target_after_ids and count > before_counts.get(paragraph_id, 0)
        for paragraph_id, count in after_counts.items()
    )


def _paragraph_counts(scan: Scan) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in scan.findings:
        counts[finding.paragraph_id] = counts.get(finding.paragraph_id, 0) + 1
    return counts


def _rewrite_progress_percent(step: int, limit: int) -> int:
    span_start = 63
    span_end = 87
    bounded_limit = max(1, int(limit or 1))
    bounded_step = max(0, min(bounded_limit, int(step or 0)))
    return span_start + int(round((span_end - span_start) * (bounded_step / bounded_limit)))


def _emit_progress(callback: Callable[[int, str], None] | None, percent: int | None, message: str) -> None:
    if callback is not None and percent is not None:
        callback(percent, message)


DEFAULT_V6_MODEL = "openai/gpt-oss-120b"


def _writer_model() -> str:
    import os

    return os.environ.get("DRAFTPROOF_V6_WRITER_MODEL") or os.environ.get("LLM_MODEL") or DEFAULT_V6_MODEL


def _planner_model() -> str:
    import os

    return (
        os.environ.get("DRAFTPROOF_V6_PLANNER_MODEL")
        or os.environ.get("DRAFTPROOF_PLANNER_MODEL")
        or os.environ.get("DRAFTPROOF_REWRITE_V5_PLANNER_MODEL")
        or os.environ.get("LLM_MODEL")
        or DEFAULT_V6_MODEL
    )


def _planner_gateway(
    *,
    api_key: str | None,
    base_url: str | None,
    cancellation_check: Callable[[], None] | None = None,
) -> LLMGateway:
    model = _planner_model()
    return LLMGateway(
        LLMConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **_planner_llm_profile(model),
            provider=_planner_provider(model),
            extra_body=_planner_extra_body(model),
            cancellation_check=cancellation_check,
        )
    )


def _planner_provider(model: str) -> dict[str, Any] | None:
    return _provider_from_env("PLANNER", model)


def _writer_provider(model: str) -> dict[str, Any] | None:
    return _provider_from_env("WRITER", model)


def _provider_from_env(role: str, model: str) -> dict[str, Any] | None:
    prefix = f"DRAFTPROOF_V6_{role}_PROVIDER"
    raw_json = _first_env(f"{prefix}_ROUTING_JSON")
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise ValueError(f"V6 {role.casefold()} provider routing JSON must be an object")
        return parsed

    provider: dict[str, Any] = {}
    order = _csv_env(f"{prefix}_ORDER")
    only = _csv_env(f"{prefix}_ONLY")
    ignore = _csv_env(f"{prefix}_IGNORE")
    if not order and str(model or "").casefold() == "z-ai/glm-4.7":
        order = ["Cerebras"]
    if order:
        provider["order"] = order
    if only:
        provider["only"] = only
    if ignore:
        provider["ignore"] = ignore

    allow_fallbacks = _bool_env(f"{prefix}_ALLOW_FALLBACKS")
    if allow_fallbacks is None:
        allow_fallbacks = _bool_env(f"DRAFTPROOF_V6_{role}_ALLOW_FALLBACKS")
    if allow_fallbacks is None and order:
        allow_fallbacks = True
    if allow_fallbacks is not None:
        provider["allow_fallbacks"] = allow_fallbacks

    sort = _first_env(f"{prefix}_SORT")
    if sort:
        provider["sort"] = sort
    return provider or None


def _planner_extra_body(model: str) -> dict[str, Any] | None:
    normalized = str(model or "").casefold()
    if "gpt-oss" in normalized:
        return {"reasoning": {"effort": "medium", "exclude": True}, "include_reasoning": False}
    if "thinking" in normalized:
        return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 128}, "include_reasoning": False}
    return {"reasoning": {"enabled": False}, "include_reasoning": False}


def _writer_extra_body(model: str) -> dict[str, Any] | None:
    normalized = str(model or "").casefold()
    if "gpt-oss" in normalized:
        return {"reasoning": {"effort": "medium", "exclude": True}, "include_reasoning": False}
    if "thinking" not in normalized:
        return None
    return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 64}, "include_reasoning": False}


def _planner_llm_profile(model: str) -> dict[str, Any]:
    if "gpt-oss" in str(model or "").casefold():
        return {
            "max_tokens": None,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 0,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "repetition_penalty": 1.0,
        }
    return {"max_tokens": None, "temperature": 0.1, "top_p": 0.75}


def _writer_llm_profile(model: str, text: str = "") -> dict[str, Any]:
    if "gpt-oss" not in str(model or "").casefold():
        return {"max_tokens": None, "temperature": 0.12, "top_p": 0.75}
    source_sensitive = _source_sensitive_text(text)
    return {
        "max_tokens": None,
        "temperature": 0.45 if source_sensitive else 0.65,
        "top_p": 0.9 if source_sensitive else 0.95,
        "top_k": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0.1 if source_sensitive else 0.15,
        "repetition_penalty": 1.03 if source_sensitive else 1.05,
    }


def _source_sensitive_text(text: str) -> bool:
    return bool(re.search(r"\([A-Z][A-Za-z .,&;'-]*\b\d{4}\)|\b(?:Act|Standards?|assessment|competency|citation|legal|VET|TAFE|unit)\b", str(text or ""), flags=re.I))


def _first_env(*names: str) -> str | None:
    import os

    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _csv_env(name: str) -> list[str]:
    value = _first_env(name)
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _bool_env(name: str) -> bool | None:
    value = _first_env(name)
    if value is None:
        return None
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None
