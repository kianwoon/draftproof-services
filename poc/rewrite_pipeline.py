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
import re
import argparse
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rewrite.parse_detect import DetectJSONParser, DetectJSONContext, findings_from_json
from rewrite import run_rewrite, RewriteModuleResult
from report.pdf import render_pdf
from report.render_rewrite import render_rewrite_report
from detect.run import DetectionRunner
from report.report import ReportBuilder, report_to_dict


def _detail_value(detail: dict, *keys, default=0):
    """Read the first present metric key from a sentence detail dict."""
    for key in keys:
        value = detail.get(key)
        if value is not None:
            return value
    return default


def _sentence_detail_lookup(details: list) -> dict:
    """Map sentence text to metric details, preserving the first occurrence."""
    lookup = {}
    for d in details or []:
        sentence = (d.get("sentence") or "").strip()
        if sentence and sentence not in lookup:
            lookup[sentence] = d
    return lookup


def _build_aligned_sentence_comparison(mp) -> list:
    """Build before/after sentence rows using text alignment, not index pairing.

    Rewritten documents can shift sentence positions after a local edit. Pairing
    sentence metrics by index makes every later sentence look rewritten and can
    produce blank rewritten cells when metric lists have different lengths.
    """
    if not mp:
        return []

    original_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", mp.original_text or "")
        if s.strip()
    ]
    final_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", mp.final_text or "")
        if s.strip()
    ]
    if not original_sentences and not final_sentences:
        return []

    orig_details = (mp.original_metrics.sentence_details if mp.original_metrics else []) or []
    final_details = (mp.final_metrics.sentence_details if mp.final_metrics else []) or []
    orig_lookup = _sentence_detail_lookup(orig_details)
    final_lookup = _sentence_detail_lookup(final_details)

    rows = []
    matcher = SequenceMatcher(a=original_sentences, b=final_sentences, autojunk=False)
    row_index = 1
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for o_sent, n_sent in zip(original_sentences[i1:i2], final_sentences[j1:j2]):
                o = orig_lookup.get(o_sent, {})
                n = final_lookup.get(n_sent, {})
                rows.append({
                    "index": row_index,
                    "orig_tier": _detail_value(o, "label", "risk_label", default="?"),
                    "orig_risk": _detail_value(o, "risk", "predictability_risk"),
                    "orig_top10": _detail_value(o, "top10_ratio", "top_10_ratio"),
                    "orig_sentence": o_sent,
                    "new_tier": _detail_value(n, "label", "risk_label", default="?"),
                    "new_risk": _detail_value(n, "risk", "predictability_risk"),
                    "new_top10": _detail_value(n, "top10_ratio", "top_10_ratio"),
                    "new_sentence": n_sent,
                })
                row_index += 1
            continue

        old_block = original_sentences[i1:i2]
        new_block = final_sentences[j1:j2]
        block_len = max(len(old_block), len(new_block))
        for offset in range(block_len):
            o_sent = old_block[offset] if offset < len(old_block) else ""
            n_sent = new_block[offset] if offset < len(new_block) else ""
            o = orig_lookup.get(o_sent, {})
            n = final_lookup.get(n_sent, {})
            rows.append({
                "index": row_index,
                "orig_tier": _detail_value(o, "label", "risk_label", default="?"),
                "orig_risk": _detail_value(o, "risk", "predictability_risk"),
                "orig_top10": _detail_value(o, "top10_ratio", "top_10_ratio"),
                "orig_sentence": o_sent,
                "new_tier": _detail_value(n, "label", "risk_label", default="?"),
                "new_risk": _detail_value(n, "risk", "predictability_risk"),
                "new_top10": _detail_value(n, "top10_ratio", "top_10_ratio"),
                "new_sentence": n_sent,
            })
            row_index += 1
    return rows


def sanitize_text(text: str) -> str:
    """Fix mojibake and normalize Unicode in text before processing.

    Handles UTF-8 bytes that were decoded as latin-1, which produces
    artifacts like: â€™ â€" â€œ â€\x9d â€¦
    """
    # Fix common mojibake patterns
    text = text.replace('â€™', "'").replace('â€˜', "'")
    text = text.replace('â€œ', '"').replace('â€\x9d', '"')
    text = text.replace('â€"', ' -- ').replace('â€"', '-')
    text = text.replace('â€¦', '...')
    # Normalize remaining Unicode to ASCII equivalents
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('—', ' -- ').replace('–', '-')
    text = text.replace('…', '...')
    text = text.replace(' ', ' ')
    # Clean up double spaces from replacements
    text = re.sub(r'  +', ' ', text)
    return text


def run_rewrite_pipeline(
    json_path: str = None,
    text: str = None,
    detect_json: dict = None,
    output_dir: str = None,
    max_passes: int = 3,
    max_detect_loops: int = 2,
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
            # Preserve raw data from report JSON for scanners that have it
            scanner_raw = None
            if scanner == "predictability":
                pred = detect_json.get("predictability", {})
                # Use all_sentences (full text + scores) if available,
                # otherwise fall back to the predictability block
                all_sents = pred.get("all_sentences")
                if all_sents:
                    scanner_raw = {"sentences": all_sents}
                else:
                    scanner_raw = pred if pred else None
            detect_results.append(DetectResult(
                scanner=scanner,
                overall_risk=0.5,
                confidence="medium",
                confidence_reason="from detect pipeline",
                risk_distribution={},
                findings=findings,
                policy_message="",
                raw=scanner_raw,
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
    else:
        medium_count = sum(
            len([f for f in dr.findings if f.risk_level in ("critical", "high", "medium")])
            for dr in ctx.detect_results
        )
        print(f"  MEDIUM+ mode: {medium_count} findings out of {total_findings} total")

    # Sanitize input text before rewrite (fix mojibake from PDF/docx extraction)
    text = sanitize_text(text)

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

    # Get sentence comparison from the MultiPassResult, aligned by text diff.
    sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)

    # Inject detect scan scores into summary so rewrite report shows
    # the same risk scores the user saw in the detect scan report.
    badge = ctx.raw_json.get("ai_risk_badge", {})
    if badge:
        result.summary["detect_ai_likelihood"] = badge.get("ai_likelihood_score", 0)
        result.summary["detect_writing_quality"] = badge.get("writing_quality_score", 0)

    # ── Final full detect scan ──────────────────────────────────────
    # If no text changed, do not re-scan. The detector has stochastic/heuristic
    # variance, so rescanning identical text can falsely show a rewrite
    # regression even when the rewrite engine made no edit.
    #
    # When text did change, run the same full scan used by detect reports. The
    # rewrite engine may use a targeted rescan internally for speed, but the
    # user-facing report and rollback decision need a product-level full scan.
    rewritten_text = result.mp_result.final_text if result.mp_result else text
    if rewritten_text == text:
        result.summary["no_text_change"] = True
        result.summary["no_text_change_reason"] = (
            result.mp_result.convergence_reason
            if result.mp_result and result.mp_result.convergence_reason
            else "No automatic rewrite was applied"
        )
        rewritten_report_dict = ctx.raw_json
    elif result.final_detect_report is not None:
        # Run a fresh full scan on the rewritten text for accurate scores.
        # The targeted rescan reuses old scores for unchanged sentences, which
        # produces misleading "After" numbers vs what a real rescan would show.
        rewritten_detect_runner = DetectionRunner()
        rewritten_detect_report = rewritten_detect_runner.run_all(rewritten_text)

        rewritten_builder = ReportBuilder()
        rewritten_builder.add_detection_report(rewritten_detect_report)
        if getattr(rewritten_detect_report, "postprocess_results", None):
            rewritten_builder.add_postprocess_results(rewritten_detect_report.postprocess_results)
        rewritten_builder.set_meta(scan_time=0, original_text=rewritten_text)
        rewritten_draft_report = rewritten_builder.build()
        rewritten_report_dict = report_to_dict(rewritten_draft_report)
    else:
        # Fallback: no cached detect report — run full scan
        rewritten_detect_runner = DetectionRunner()
        rewritten_detect_report = rewritten_detect_runner.run_all(rewritten_text)

        rewritten_builder = ReportBuilder()
        rewritten_builder.add_detection_report(rewritten_detect_report)
        if rewritten_detect_report.postprocess_results:
            rewritten_builder.add_postprocess_results(rewritten_detect_report.postprocess_results)
        rewritten_builder.set_meta(scan_time=0, original_text=rewritten_text)
        rewritten_draft_report = rewritten_builder.build()
        rewritten_report_dict = report_to_dict(rewritten_draft_report)

    def _finding_total(report_dict):
        findings = report_dict.get("findings", {})
        return sum(len(findings.get(t, [])) for t in ("critical", "high", "medium", "low"))

    def _badge_ai(report_dict):
        score = (report_dict.get("ai_risk_badge") or {}).get("ai_likelihood_score")
        return float(score) if isinstance(score, (int, float)) else None

    original_ai = _badge_ai(ctx.raw_json)
    rewritten_ai = _badge_ai(rewritten_report_dict)
    original_total = _finding_total(ctx.raw_json)
    rewritten_total = _finding_total(rewritten_report_dict)

    # Log scores for transparency but do NOT rollback.
    # Per-sentence guards (predictability, drift, voice erosion, component)
    # already reject bad rewrites at the sentence level. The pipeline-level
    # rollback was too aggressive — full-scan scores have stochastic variance
    # that can falsely trigger rollback even when individual rewrites improved.
    result.summary["detect_scores"] = {
        "original_ai": original_ai,
        "rewritten_ai": rewritten_ai,
        "original_findings": original_total,
        "rewritten_findings": rewritten_total,
    }

    # Extract only the fields needed for comparison (not full report dicts)
    def _extract_scan_summary(report_dict):
        badge = report_dict.get("ai_risk_badge") or {}
        findings = report_dict.get("findings", {})
        return {
            "ai_risk_badge": badge,
            "overall_tier": report_dict.get("overall_tier", "?"),
            "findings": {t: [{"finding_id": f.get("finding_id"), "title": f.get("title"),
                              "category": f.get("category")} for f in findings.get(t, [])]
                         for t in ("critical", "high", "medium", "low")},
        }

    result.summary["detect_scan_original"] = _extract_scan_summary(ctx.raw_json)
    result.summary["detect_scan_rewritten"] = _extract_scan_summary(rewritten_report_dict)

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
