"""Final rewritten-text scan phase."""

from __future__ import annotations

import time
from collections.abc import Callable


def run_final_rewritten_scan(
    *,
    original_text: str,
    rewritten_text: str,
    baseline_report_dict: dict | None,
    summary: dict,
    stage_timings: list[dict],
    report_progress: Callable[[int, str], None],
    detection_runner_factory: Callable[[], object],
    report_builder_factory: Callable[[], object],
    report_to_dict: Callable[[object], dict],
) -> dict | None:
    """Return the product-level rewritten scan report or reuse baseline.

    If no text changed, do not re-scan. The detector has stochastic/heuristic
    variance, so rescanning identical text can falsely show a rewrite regression
    even when the rewrite engine made no edit.
    """
    if rewritten_text == original_text:
        report_progress(78, "No automatic text changes were kept")
        summary["no_text_change"] = True
        summary["no_text_change_reason"] = summary.get("no_text_change_reason") or "No automatic rewrite was applied"
        return baseline_report_dict

    report_progress(76, "Running final scan on rewritten draft")
    scan_t0 = time.time()
    detect_runner = detection_runner_factory()
    detect_report = detect_runner.run_all(rewritten_text)
    stage_timings.append({
        "stage": "fresh_rewritten_scan",
        "seconds": round(time.time() - scan_t0, 3),
    })

    builder = report_builder_factory()
    builder.add_detection_report(detect_report)
    if getattr(detect_report, "postprocess_results", None):
        builder.add_postprocess_results(detect_report.postprocess_results)
    builder.set_meta(scan_time=0, original_text=rewritten_text)
    report_progress(78, "Final rewritten scan complete")
    return report_to_dict(builder.build())
