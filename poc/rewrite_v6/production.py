from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

try:
    from report.pdf import render_pdf
    from report.render_rewrite import render_rewrite_report
    from rewrite_v2.pipeline import _extract_original_text, _sentence_comparison
except ModuleNotFoundError:
    from poc.report.pdf import render_pdf
    from poc.report.render_rewrite import render_rewrite_report
    from poc.rewrite_v2.pipeline import _extract_original_text, _sentence_comparison

from .pipeline import run_v6_rewrite_all


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
    original_text = _extract_original_text(detect_json)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    document = run_v6_rewrite_all(
        original_text,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    final_text = document.rewritten_text
    changed = final_text.strip() != original_text.strip()
    cleared = not document.final_scan.findings
    status = "ai_mitigated" if changed and cleared else "partial_candidate_not_strict_safe" if changed else "original_preserved"
    elapsed = time.time() - started
    sentence_comparison = _sentence_comparison(original_text, final_text)
    original_scan_report = _scan_report_shape(document.initial_scan.to_dict())
    rewritten_scan_report = _scan_report_shape(document.final_scan.to_dict())
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
            "model": model,
            "pipeline": "v6",
            "passes": len(document.passes),
        },
        "candidate_generation_status": {
            "accepted_count": len(document.passes),
            "remaining_findings": len(document.final_scan.findings),
            "stop_reason": status,
        },
        "v6_scores": {
            "initial": document.initial_scan.scores,
            "final": document.final_scan.scores,
        },
        "detect_scan_original_saved": original_scan_report,
        "detect_scan_original": original_scan_report,
        "detect_scan_rewritten": rewritten_scan_report,
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
    md_text = render_rewrite_report(summary=summary, sentence_comparison=sentence_comparison, ai_findings=[], verbose=False)
    md_path.write_text(md_text, encoding="utf-8")
    render_pdf(md_text, str(pdf_path))
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
