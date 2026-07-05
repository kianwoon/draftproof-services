"""DraftProof Detect Pipeline — run detection and generate reports.

Usage:
  python detect_pipeline.py                         # inline text via --text
  python detect_pipeline.py input.txt               # from file
  echo "text" | python detect_pipeline.py -         # from stdin
  python detect_pipeline.py --text "Your text here"

Output:
  test_output/draftproof_<timestamp>.md
  test_output/draftproof_<timestamp>.json
  test_output/draftproof_<timestamp>.pdf
"""

import sys
import os
import json
import time
import argparse
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Deep-scan progress heartbeat tuning. The single blocking Modal call has no
# sub-progress, so a timer ticks the bar up while it runs. Ceiling stays below
# the next real checkpoint (88 "Rendering markdown report") to avoid overshoot.
_DEEP_SCAN_HEARTBEAT_INTERVAL_S = 5.0
_DEEP_SCAN_HEARTBEAT_CEILING = 87

from poc.detect.run import DetectionRunner
from poc.detect.document_structure import normalize_submitted_text
from poc.report.report import ReportBuilder, report_to_dict
from poc.report.render import render_report
from poc.report.pdf import render_pdf


def read_input(args) -> str:
    if args.text:
        return args.text
    if args.file == "-" or (not args.file and not sys.stdin.isatty()):
        return sys.stdin.read()
    if args.file:
        with open(args.file, "r") as f:
            return f.read()
    print("Error: provide --text, a file path, or pipe text via stdin", file=sys.stderr)
    sys.exit(1)


def run_detect(
    text: str,
    output_dir: str,
    verbose: bool = False,
    model_name: str | None = None,
    progress_callback=None,
) -> dict:
    raw_text = str(text or "")
    text = normalize_submitted_text(raw_text)
    t0 = time.time()
    runner = DetectionRunner()
    kwargs = {}
    if model_name:
        kwargs["predictability_model"] = model_name
    det_report = runner.run_all(text, progress_callback=progress_callback, **kwargs)
    elapsed = time.time() - t0

    # ReportBuilder.build() runs a synchronous deep-scan (sentence-level Modal
    # inference + V7 fusion) when tier-authority is enabled — the single longest
    # silent step, with no sub-progress. Name it honestly so the bar doesn't read
    # as "frozen at 82%". Fall back to a neutral label when deep-scan is off.
    _deep_scan_on = False
    if progress_callback:
        try:
            from detect_v7.pipeline_bridge import is_tier_authority_enabled as _is_ta
            _deep_scan_on = bool(_is_ta())
        except Exception:
            _deep_scan_on = False
        progress_callback(
            82,
            "Deep-scanning sentences — this can take a moment"
            if _deep_scan_on
            else "Building report",
        )

    # The deep-scan is ONE atomic, blocking Modal call (no client-side per-batch
    # boundary), so the bar can't track true per-sentence progress. Instead, tick
    # it up 82→87 on a timer from a daemon thread while build() blocks, then stop
    # and let the real checkpoints (88+) take over. Time estimate, not true
    # progress; capped below the next real checkpoint (88) so it never overshoots.
    _hb_stop = threading.Event()
    _hb_thread = None

    def _deep_scan_heartbeat() -> None:
        pct = 82
        while not _hb_stop.wait(_DEEP_SCAN_HEARTBEAT_INTERVAL_S):
            if pct >= _DEEP_SCAN_HEARTBEAT_CEILING:
                continue
            pct += 1
            try:
                progress_callback(pct, "Deep-scanning sentences — this can take a moment")
            except Exception:
                pass

    if progress_callback and _deep_scan_on:
        _hb_thread = threading.Thread(
            target=_deep_scan_heartbeat, name="deep-scan-heartbeat", daemon=True
        )
        _hb_thread.start()

    try:
        builder = ReportBuilder()
        builder.add_detection_report(det_report)
        if det_report.postprocess_results:
            builder.add_postprocess_results(det_report.postprocess_results)
        builder.set_meta(scan_time=elapsed, original_text=text)
        draft_report = builder.build()
    finally:
        _hb_stop.set()
        if _hb_thread is not None:
            _hb_thread.join(timeout=1.0)

    # Write output files
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_dir, f"draftproof_{ts}.md")
    json_path = os.path.join(output_dir, f"draftproof_{ts}.json")

    if progress_callback:
        progress_callback(88, "Rendering markdown report")
    with open(md_path, "w") as f:
        f.write(render_report(draft_report, verbose=verbose))

    if progress_callback:
        progress_callback(92, "Rendering PDF report")
    pdf_path = os.path.join(output_dir, f"draftproof_{ts}.pdf")
    render_pdf(render_report(draft_report, verbose=verbose), pdf_path)

    if progress_callback:
        progress_callback(95, "Writing scan results")
    json_data = report_to_dict(draft_report)
    json_data["input_text"] = text
    if raw_text.strip() and raw_text.strip() != text:
        json_data["raw_input_text"] = raw_text
        json_data["input_text_normalized"] = True
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    # Use ai_risk_badge tier (what PDF shows) over findings-based tier
    display_tier = draft_report.overall_tier.value
    if draft_report.ai_risk_badge:
        display_tier = draft_report.ai_risk_badge.get("tier", display_tier).lower()

    return {
        "md_path": md_path,
        "json_path": json_path,
        "pdf_path": pdf_path,
        "tier": display_tier,
        "findings": draft_report.finding_count,
        "scan_time": elapsed,
        "report": draft_report,
    }


def main():
    parser = argparse.ArgumentParser(description="DraftProof Detect Pipeline")
    parser.add_argument("file", nargs="?", help="Input text file (or - for stdin)")
    parser.add_argument("--text", "-t", help="Inline text to detect")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: test_output/)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose report with scanner details")
    args = parser.parse_args()

    output_dir = args.output or os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "test_output"
    ))

    text = read_input(args)
    if not text.strip():
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    print(f"Running detection pipeline...")
    result = run_detect(text, output_dir, verbose=args.verbose)

    tier = result["tier"].upper()
    n = result["findings"]
    t = result["scan_time"]
    print(f"\n  Tier: {tier}  |  Findings: {n}  |  Time: {t:.1f}s")
    print(f"  MD:   {result['md_path']}")
    print(f"  JSON: {result['json_path']}")
    print(f"  PDF:  {result['pdf_path']}")


if __name__ == "__main__":
    main()
