"""DraftProof Rewrite Pipeline — reads detect JSON, runs rewrite, outputs report.

Usage:
  python rewrite_pipeline.py detect.json                    # from detect JSON
  python rewrite_pipeline.py detect.json --passes 5         # more rewrite passes
  python rewrite_pipeline.py detect.json --max-loops 3      # more detect-rewrite loops
  python rewrite_pipeline.py --text "Some text here"        # detect + rewrite inline

Output:
  test_output/draftproof_rewrite_<timestamp>.md
  test_output/draftproof_rewrite_<timestamp>.pdf
  test_output/draftproof_rewrite_<timestamp>.json
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rewrite.parse_detect import DetectJSONParser, DetectJSONContext, findings_from_json
from rewrite import run_rewrite, RewriteModuleResult
from report.pdf import render_pdf
from report.render_rewrite import render_rewrite_report


def run_rewrite_pipeline(
    json_path: str = None,
    text: str = None,
    detect_json: dict = None,
    output_dir: str = None,
    max_passes: int = 3,
    max_detect_loops: int = 0,
    target_top10: float = 0.50,
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    verbose: bool = False,
    ai_only: bool = True,
) -> dict:
    """Run the full rewrite pipeline from detect JSON or raw text.

    Args:
        json_path: Path to detect JSON file.
        text: Raw text (will run detect first).
        detect_json: Pre-loaded detect JSON dict.
        output_dir: Where to write output files.
        max_passes: Max rewrite passes per loop.
        max_detect_loops: Max detect-rewrite loops.
        target_top10: Target top-10 ratio for convergence.
        model: LLM model for rewriting (None → from env).
        api_key: API key for LLM (None → from env).
        base_url: LLM API base URL (None → from env or OpenRouter default).
        verbose: Include scanner details in report.
        ai_only: Only rewrite AI-generation findings (default True).

    Returns dict with paths and summary.
    """
    # ── Parse input ────────────────────────────────────────────────
    ctx: DetectJSONContext = None

    if json_path or detect_json:
        if detect_json:
            ctx = DetectJSONParser.parse_dict(detect_json)
        else:
            ctx = DetectJSONParser.parse(json_path)
        text = ctx.input_text
    elif text:
        # Run detect first, then parse
        from detect_pipeline import run_detect
        detect_result = run_detect(text, output_dir or "test_output", verbose=verbose)
        report = detect_result["report"]

        from detect.base import DetectResult
        by_scanner = {}
        for tier_findings in report.findings_by_tier.values():
            for f in tier_findings:
                by_scanner.setdefault(f.scanner, []).append(f)

        detect_results = []
        for scanner, findings in by_scanner.items():
            detect_results.append(DetectResult(
                scanner=scanner,
                overall_risk=0.5,
                confidence="medium",
                confidence_reason="from detect pipeline",
                risk_distribution={},
                findings=findings,
                policy_message="",
                raw=None,
            ))
        ctx = DetectJSONContext(
            detect_results=detect_results,
            input_text=text,
        )

    if not text or not text.strip():
        raise ValueError("Empty input text")

    # ── Check rewrite decision from detect ──────────────────────────
    if ctx.rewrite_decision and not ctx.rewrite_decision.get("run_rewrite", True):
        reason = ctx.rewrite_decision.get("reason", "Rewrite not recommended")
        print(f"Rewrite skipped: {reason}")
        return {
            "status": "skipped",
            "message": reason,
            "tier": ctx.overall_tier,
        }

    all_findings = [f for dr in ctx.detect_results for f in dr.findings]
    if not all_findings:
        print("No findings to rewrite. Text is clean.")
        return {"status": "clean", "message": "No findings to rewrite"}

    # ── Run rewrite ─────────────────────────────────────────────────
    print(f"Running rewrite pipeline...")
    print(f"  Input: {len(text)} chars, {len(ctx.detect_results)} scanner results")
    if ctx.rewrite_decision:
        print(f"  Decision: mode={ctx.rewrite_decision.get('mode', 'targeted')}")

    total_findings = sum(len(dr.findings) for dr in ctx.detect_results)
    if ai_only:
        ai_count = sum(
            len(dr.findings if dr.scanner == "ai_generation"
                else [f for f in dr.findings
                      if (f.metadata or {}).get("scanner") == "ai_generation"
                      or (f.metadata or {}).get("category") == "ai_generation"])
            for dr in ctx.detect_results
        )
        print(f"  AI-only mode: {ai_count} AI findings out of {total_findings} total")

    t0 = time.time()
    result: RewriteModuleResult = run_rewrite(
        content=text,
        detect_results=ctx.detect_results,
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_passes=max_passes,
        target_top10=target_top10,
        max_detect_loops=max_detect_loops,
        output_dir=output_dir,
        rewrite_context=ctx,
        ai_only=ai_only,
    )
    elapsed = time.time() - t0

    # ── Write output ────────────────────────────────────────────────
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "test_output"
        ))

    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_dir, f"draftproof_rewrite_{ts}.md")
    json_path_out = os.path.join(output_dir, f"draftproof_rewrite_{ts}.json")

    # Extract AI-only findings from detect JSON
    ai_findings = []
    raw_findings = ctx.raw_json.get("findings", {})
    for tier in ("critical", "high", "medium", "low"):
        for f in raw_findings.get(tier, []):
            cat = (f.get("category") or f.get("scanner") or "").lower()
            if cat == "ai_generation":
                ai_findings.append(f)

    # Get sentence comparison from the MultiPassResult metrics
    sentence_comparison = []
    mp = result.mp_result
    if mp and mp.original_metrics and mp.final_metrics:
        orig_details = mp.original_metrics.sentence_details or []
        final_details = mp.final_metrics.sentence_details or []
        max_idx = max(len(orig_details), len(final_details))
        for i in range(max_idx):
            o = orig_details[i] if i < len(orig_details) else {}
            f = final_details[i] if i < len(final_details) else {}
            sentence_comparison.append({
                "index": i + 1,
                "orig_tier": o.get("label", "?"),
                "orig_risk": o.get("risk", 0),
                "orig_top10": o.get("top10_ratio", 0),
                "orig_sentence": o.get("sentence", ""),
                "new_tier": f.get("label", "?"),
                "new_risk": f.get("risk", 0),
                "new_top10": f.get("top10_ratio", 0),
                "new_sentence": f.get("sentence", ""),
            })

    # Generate dedicated rewrite report
    rewrite_md = render_rewrite_report(
        summary=result.summary,
        sentence_comparison=sentence_comparison,
        ai_findings=ai_findings,
        verbose=verbose,
    )

    with open(md_path, "w") as f:
        f.write(rewrite_md)

    pdf_path = os.path.join(output_dir, f"draftproof_rewrite_{ts}.pdf")
    render_pdf(rewrite_md, pdf_path)

    summary = result.summary
    summary["rewrite_time"] = elapsed
    summary["original_tier"] = ctx.overall_tier
    summary["rewrite_decision"] = ctx.rewrite_decision

    with open(json_path_out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return {
        "status": "rewritten",
        "md_path": md_path,
        "pdf_path": pdf_path,
        "json_path": json_path_out,
        "result": result,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="DraftProof Rewrite Pipeline")
    parser.add_argument("file", nargs="?", help="Detect JSON file (or - for stdin)")
    parser.add_argument("--text", "-t", help="Inline text to detect + rewrite")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--passes", type=int, default=3, help="Max rewrite passes")
    parser.add_argument("--max-loops", type=int, default=1, help="Max detect-rewrite loops")
    parser.add_argument("--target-top10", type=float, default=0.50, help="Target top-10 ratio")
    parser.add_argument("--model", default=None, help="LLM model (default: from LLM_MODEL env var)")
    parser.add_argument("--api-key", default=None, help="API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-ai-only", action="store_true", help="Rewrite ALL findings (default: AI-only)")
    args = parser.parse_args()

    output_dir = args.output or os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "test_output"
    ))

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    # Read input
    json_path = None
    text = None

    if args.text:
        text = args.text
    elif args.file == "-" or (not args.file and not sys.stdin.isatty()):
        raw = sys.stdin.read()
        try:
            json.loads(raw)
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                tf.write(raw)
                json_path = tf.name
        except json.JSONDecodeError:
            text = raw
    elif args.file:
        json_path = args.file
    else:
        print("Error: provide a detect JSON file, --text, or pipe JSON via stdin", file=sys.stderr)
        sys.exit(1)

    result = run_rewrite_pipeline(
        json_path=json_path,
        text=text,
        output_dir=output_dir,
        max_passes=args.passes,
        max_detect_loops=args.max_loops,
        target_top10=args.target_top10,
        model=args.model,
        api_key=api_key,
        verbose=args.verbose,
        ai_only=not args.no_ai_only,
    )

    if result["status"] == "clean":
        print(f"\n  Status: {result['message']}")
    elif result["status"] == "skipped":
        print(f"\n  Skipped: {result['message']}")
    else:
        elapsed = result["elapsed"]
        r = result["result"]
        rw = r.mp_result
        print(f"\n  Time: {elapsed:.1f}s")
        print(f"  Passes: {len(rw.passes)}")
        print(f"  Converged: {'Yes' if rw.converged else 'No'}")
        print(f"  MD:   {result['md_path']}")
        print(f"  PDF:  {result['pdf_path']}")
        print(f"  JSON: {result['json_path']}")


if __name__ == "__main__":
    main()
