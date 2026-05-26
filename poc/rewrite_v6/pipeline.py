from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from poc.llm.gateway import LLMConfig, LLMGateway

from .plan import Plan, build_plan
from .planner_llm import run_planner_llm
from .report_contracts import apply_report_signal_contracts
from .scan import Scan, findings_for_paragraph, scan_text
from .write import Variant, choose_variant, write_variants


@dataclass(frozen=True)
class Result:
    scan: Scan
    plan: Plan
    variants: list[Variant]
    selected: Variant | None
    rewritten_text: str

    def to_dict(self) -> dict[str, Any]:
        generated_variants = [variant for variant in self.variants if variant.source != "source_preserved"]
        source_variant = next((variant for variant in self.variants if variant.source == "source_preserved"), None)
        return {
            "scan": self.scan.to_dict(),
            "plan": self.plan.to_dict(),
            "variants": [asdict(variant) for variant in generated_variants],
            "source_preserved": asdict(source_variant) if source_variant else None,
            "selected": asdict(self.selected) if self.selected else None,
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
    _emit_progress(progress_callback, progress_percent, f"Planning V6 paragraph {paragraph.id}")
    _raise_if_canceled(cancellation_check)
    if planner_client is not None or writer_client is None:
        plan = run_planner_llm(
            paragraph,
            plan,
            findings_for_paragraph(scan, paragraph.id),
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
            max_tokens=None,
            temperature=0.12,
            top_p=0.75,
            provider=_writer_provider(model or _writer_model()),
            extra_body=_writer_extra_body(model or _writer_model()),
            cancellation_check=cancellation_check,
        )
    )
    variants = write_variants(paragraph, plan, client=client)
    _raise_if_canceled(cancellation_check)
    _emit_progress(progress_callback, progress_percent, f"Scanning V6 paragraph {paragraph.id} candidate")
    selected = choose_variant(variants, paragraph)
    return Result(scan=scan, plan=plan, variants=variants, selected=selected, rewritten_text=_compose(scan, paragraph.id, selected))


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
            if attempts[result.plan.paragraph_id] >= 2:
                exhausted.add(result.plan.paragraph_id)
            pass_trace.append(
                _pass_trace_row(
                    pass_index=pass_index,
                    status="no_change",
                    before=before,
                    target_paragraph_id=result.plan.paragraph_id,
                    excluded=excluded_for_pass,
                )
            )
            _emit_progress(progress_callback, end_percent, f"V6 paragraph {result.plan.paragraph_id} made no change")
            continue
        after = scan_text(result.rewritten_text)
        if _cross_paragraph_regression(before, after, result.plan.paragraph_id) or not _acceptable_progress(before, after, report_targeted=result.plan.paragraph_id in report_target_ids):
            attempts[result.plan.paragraph_id] = attempts.get(result.plan.paragraph_id, 0) + 1
            if attempts[result.plan.paragraph_id] >= 4:
                exhausted.add(result.plan.paragraph_id)
            pass_trace.append(
                _pass_trace_row(
                    pass_index=pass_index,
                    status="not_improved",
                    before=before,
                    after=after,
                    target_paragraph_id=result.plan.paragraph_id,
                    excluded=excluded_for_pass,
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
                target_paragraph_id=result.plan.paragraph_id,
                excluded=excluded_for_pass,
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
    target_ids = _target_paragraph_ids_after_rewrite(before, after, accepted_result.plan.paragraph_id)
    for residual_index in range(residual_limit):
        _raise_if_canceled(cancellation_check)
        before_residual = scan_text(current)
        active_targets = target_ids & _finding_paragraph_ids(before_residual)
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
            pass_trace.append(_residual_trace_row(pass_index, residual_index, "no_change_residual", before_residual, None, [result.plan.paragraph_id]))
            break
        after_residual = scan_text(result.rewritten_text)
        if _cross_paragraph_regression(before_residual, after_residual, result.plan.paragraph_id) or not _acceptable_progress(before_residual, after_residual, report_targeted=False):
            pass_trace.append(_residual_trace_row(pass_index, residual_index, "not_improved_residual", before_residual, after_residual, [result.plan.paragraph_id]))
            break
        passes.append(result)
        pass_trace.append(_residual_trace_row(pass_index, residual_index, "accepted_residual", before_residual, after_residual, [result.plan.paragraph_id]))
        current = result.rewritten_text
        target_ids = _target_paragraph_ids_after_rewrite(before_residual, after_residual, result.plan.paragraph_id)
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


def _target_paragraph_ids_after_rewrite(before: Scan, after: Scan, target_paragraph_id: str) -> set[str]:
    target = next((paragraph for paragraph in before.paragraphs if paragraph.id == target_paragraph_id), None)
    if target is None:
        return {target_paragraph_id}
    delta = len(after.paragraphs) - len(before.paragraphs)
    start = target.index
    end = target.index + max(0, delta)
    return {
        after.paragraphs[index].id
        for index in range(start, min(len(after.paragraphs), end + 1))
    } or {target_paragraph_id}


def _residual_trace_row(
    pass_index: int,
    residual_index: int,
    status: str,
    before: Scan,
    after: Scan | None,
    target_ids: list[str],
) -> dict[str, Any]:
    row = _pass_trace_row(
        pass_index=pass_index,
        status=status,
        before=before,
        after=after,
        target_paragraph_id=",".join(target_ids),
        excluded=set(),
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


def _cross_paragraph_regression(before: Scan, after: Scan, target_paragraph_id: str) -> bool:
    before_counts = _paragraph_counts(before)
    after_counts = _paragraph_counts(after)
    target = next((paragraph for paragraph in before.paragraphs if paragraph.id == target_paragraph_id), None)
    if target is not None and len(after.paragraphs) != len(before.paragraphs):
        delta = len(after.paragraphs) - len(before.paragraphs)
        target_after_count = sum(after_counts.get(f"p{index + 1:03d}", 0) for index in range(target.index, target.index + delta + 1))
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
                if after_counts.get(after_id, 0) > before_counts.get(paragraph.id, 0) and not (target_drop >= 2 and total_drop > 0 and max_non_target_regression <= 1):
                    return True
        return False
    return any(
        paragraph_id != target_paragraph_id and count > before_counts.get(paragraph_id, 0)
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


def _writer_model() -> str:
    import os

    return os.environ.get("DRAFTPROOF_V6_WRITER_MODEL") or os.environ.get("LLM_MODEL") or "z-ai/glm-4.7"


def _planner_model() -> str:
    import os

    return (
        os.environ.get("DRAFTPROOF_V6_PLANNER_MODEL")
        or os.environ.get("DRAFTPROOF_PLANNER_MODEL")
        or os.environ.get("DRAFTPROOF_REWRITE_V5_PLANNER_MODEL")
        or "z-ai/glm-4.7"
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
            max_tokens=None,
            temperature=0.1,
            top_p=0.75,
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
    if "thinking" in str(model or "").casefold():
        return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 128}, "include_reasoning": False}
    return {"reasoning": {"enabled": False}, "include_reasoning": False}


def _writer_extra_body(model: str) -> dict[str, Any] | None:
    if "thinking" not in str(model or "").casefold():
        return None
    return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 64}, "include_reasoning": False}


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
