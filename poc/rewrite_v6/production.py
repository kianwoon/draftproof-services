from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

try:
    from detect.document_structure import normalize_submitted_text
    from report.pdf import render_pdf
    from report.render_rewrite import render_rewrite_report
    from rewrite_v2.pipeline import _extract_original_text, _sentence_comparison
    from rewrite_v3.pipeline import _scan_report
except ModuleNotFoundError:
    from poc.detect.document_structure import normalize_submitted_text
    from poc.report.pdf import render_pdf
    from poc.report.render_rewrite import render_rewrite_report
    from poc.rewrite_v2.pipeline import _extract_original_text, _sentence_comparison
    from poc.rewrite_v3.pipeline import _scan_report

from .pipeline import _planner_model, _writer_model, run_v6_rewrite_all
from .report_contracts import extract_report_signal_contracts


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

    progress(62, "Starting V6 scanner-planner-writer rewrite")
    original_text, rewrite_source = _rewrite_source_text(detect_json)
    effective_detect_json = dict(detect_json)
    effective_detect_json["input_text"] = original_text
    effective_detect_json["rewrite_source"] = rewrite_source
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    requested_writer_model = model
    resolved_writer_model = model or _writer_model()
    resolved_planner_model = _planner_model()
    max_passes = _v6_max_passes()

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
    )
    stage_timings: list[dict[str, Any]] = []

    def timed_stage(name: str, percent: int, message: str, fn):
        progress(percent, message)
        t0 = time.time()
        value = fn()
        stage_timings.append({"stage": name, "seconds": round(time.time() - t0, 3)})
        return value

    final_text = document.rewritten_text
    changed = final_text.strip() != original_text.strip()
    cleared = not document.final_scan.findings
    status = "ai_mitigated" if changed and cleared else "partial_candidate_not_strict_safe" if changed else "original_preserved"
    elapsed = time.time() - started
    sentence_comparison = timed_stage(
        "sentence_comparison",
        80,
        "Comparing original and rewritten sentences",
        lambda: _sentence_comparison(original_text, final_text),
    )
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
        },
        "rewrite_source": rewrite_source,
        "candidate_generation_status": {
            "accepted_count": len(document.passes),
            "remaining_findings": len(document.final_scan.findings),
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
        "detect_scan_original_saved": original_scan_report,
        "detect_scan_original": original_scan_report,
        "detect_scan_rewritten": rewritten_scan_report,
        "final_text_before_quality_repair": document.final_text_before_quality_repair,
        "quality_repair": document.quality_repair.to_dict() if document.quality_repair else None,
        "final_text": final_text,
        "no_text_change": not changed,
    }
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
        lambda: render_rewrite_report(summary=summary, sentence_comparison=sentence_comparison, ai_findings=[], verbose=False),
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
    progress(88, "V6 scanner-planner-writer rewrite complete")
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


def _scan_report_for_summary(text: str, *, provided: dict[str, Any] | None, fallback_scan: dict[str, Any]) -> dict[str, Any]:
    if _has_full_report_shape(provided):
        return dict(provided or {})
    try:
        report = _scan_report(text)
        if _has_full_report_shape(report):
            return report
    except Exception:
        pass
    return _scan_report_shape(fallback_scan)


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
        return 3
    return _int_env("DRAFTPROOF_V6_MAX_PASSES", 0, minimum=1, maximum=20)


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    import os

    try:
        value = int(os.environ.get(name, str(default)))
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
