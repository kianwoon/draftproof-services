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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc.detect.run import DetectionRunner
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


def run_detect(text: str, output_dir: str, verbose: bool = False) -> dict:
    t0 = time.time()
    runner = DetectionRunner()
    det_report = runner.run_all(text)
    elapsed = time.time() - t0

    builder = ReportBuilder()
    builder.add_detection_report(det_report)
    if det_report.postprocess_results:
        builder.add_postprocess_results(det_report.postprocess_results)
    builder.set_meta(scan_time=elapsed, original_text=text)
    draft_report = builder.build()

    # Write output files
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_dir, f"draftproof_{ts}.md")
    json_path = os.path.join(output_dir, f"draftproof_{ts}.json")

    with open(md_path, "w") as f:
        f.write(render_report(draft_report, verbose=verbose))

    pdf_path = os.path.join(output_dir, f"draftproof_{ts}.pdf")
    render_pdf(render_report(draft_report, verbose=verbose), pdf_path)

    json_data = report_to_dict(draft_report)
    json_data["input_text"] = text
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    return {
        "md_path": md_path,
        "json_path": json_path,
        "pdf_path": pdf_path,
        "tier": draft_report.overall_tier.value,
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
