"""DraftProof Pipeline — thin orchestrator: detect → rewrite → report.

Usage:
    from pipeline import run_pipeline, print_pipeline
    report = run_pipeline(content)
    print_pipeline(report)

Both detect and rewrite accept content input.
"""

import sys
import os
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from detect import DetectionRunner, DetectResult
from report import ReportBuilder, DraftReport, report_to_dict, render_report, print_report, render_markdown, print_markdown


def run_pipeline(
    content: str,
    source_sentences: Optional[List[str]] = None,
    bib_text: Optional[str] = None,
    do_rewrite: bool = False,
    rewrite_api_key: Optional[str] = None,
    rewrite_model: Optional[str] = None,
    rewrite_max_passes: int = 3,
    rewrite_fn: Optional[callable] = None,
    predictability_model: str = "gpt2",
) -> DraftReport:
    """Run the full DraftProof pipeline on content.

    Args:
        content: The text to analyze.
        source_sentences: Optional source texts for similarity detection.
        bib_text: Optional bibliography text for citation checking.
        do_rewrite: Whether to run the rewrite module.
        rewrite_api_key: Anthropic API key for rewrites.
        rewrite_model: Claude model for rewrites.
        rewrite_max_passes: Max rewrite iterations.
        rewrite_fn: Optional callable override for testing.
        predictability_model: GPT-2 model name.

    Returns:
        DraftReport with all findings.
    """
    t0 = time.time()

    # 1. Detect
    runner = DetectionRunner()
    detect_results = runner.run_all(
        content,
        source_sentences=source_sentences,
        bib_text=bib_text,
        predictability_model=predictability_model,
    )

    # 2. Rewrite (optional)
    rewrite_result = None
    if do_rewrite:
        from rewrite.rewrite import run_rewrite
        from rewrite.parse_detect import DetectJSONContext

        ctx = DetectJSONContext(
            detect_results=detect_results.scanner_results,
            input_text=content,
            rewrite_decision=getattr(detect_results, "rewrite_decision", None),
            domain_profile=None,
        )
        rewrite_result = run_rewrite(
            content=content,
            detect_results=detect_results.scanner_results,
            max_passes=rewrite_max_passes,
            api_key=rewrite_api_key,
            model=rewrite_model,
            rewrite_fn=rewrite_fn,
            rewrite_context=ctx,
        )

    # 3. Report
    elapsed = time.time() - t0
    builder = ReportBuilder()
    builder.set_meta(scan_time=elapsed, original_text=content)

    for r in detect_results.scanner_results:
        builder.add_detection(r)

    # Pass postprocess results for false-positive reporting
    if detect_results.postprocess_results:
        builder.add_postprocess_results(detect_results.postprocess_results)

    if rewrite_result:
        builder.add_rewrite(rewrite_result)
        # Re-scan rewritten text to get its detection stats
        rewritten_text = rewrite_result.final_text
        if rewritten_text != content:
            rw_detect = runner.run_all(
                rewritten_text,
                predictability_model=predictability_model,
            )
            rw_builder = ReportBuilder()
            rw_builder.set_meta(original_text=rewritten_text)
            for r in rw_detect.scanner_results:
                rw_builder.add_detection(r)
            rw_report = rw_builder.build()
            if builder._rewrite_summary and rw_report.predictability:
                builder._rewrite_summary.rewritten_tier = rw_report.overall_tier.value.upper()
                builder._rewrite_summary.rewritten_findings = rw_report.finding_count
                builder._rewrite_summary.rewritten_distribution = dict(rw_report.predictability.risk_distribution)

    return builder.build()


def run_detect_only(
    content: str,
    source_sentences: Optional[List[str]] = None,
    bib_text: Optional[str] = None,
    predictability_model: str = "gpt2",
) -> List[DetectResult]:
    """Run detection only, return raw DetectResult list."""
    runner = DetectionRunner()
    report = runner.run_all(
        content,
        source_sentences=source_sentences,
        bib_text=bib_text,
        predictability_model=predictability_model,
    )
    return report.scanner_results


def print_pipeline(report: DraftReport, verbose: bool = False, file=None):
    """Print the pipeline report to stdout or file."""
    print_report(report, verbose=verbose, file=file)


def print_pipeline_md(report: DraftReport, verbose: bool = False, file=None):
    """Print the pipeline report as Markdown to stdout or file."""
    print_markdown(report, verbose=verbose, file=file)


def pipeline_to_json(report: DraftReport) -> dict:
    """Convert pipeline report to JSON-serializable dict."""
    return report_to_dict(report)


if __name__ == "__main__":
    demo_text = """Hairdressing sits at the intersection of chemistry, geometry, and gut instinct. The craft rests on a paradox. Hold a section at 90 degrees with too much tension and the graduation disappears."""

    print("DraftProof Pipeline — Demo\n")
    report = run_pipeline(demo_text)
    print_pipeline(report, verbose=True)
