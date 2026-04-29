"""DraftProof Report Demo -- full pipeline with tiered report output.

Runs all scanners on a test draft and produces a structured report.

Run:  cd poc/report && python demo.py
"""

import sys
import os
import time
import json
import importlib

# Load sibling modules
def _load(module_dir, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(os.path.dirname(__file__), "..", module_dir, "scanner.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_pred = _load("predictability", "pred")
_sim = _load("similarity", "sim")
_cite = _load("citation", "cite")

# Also load rewriter (add its dir + siblings to sys.path first)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rewriter"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rewrite"))
_rw_spec = importlib.util.spec_from_file_location(
    "rewriter",
    os.path.join(os.path.dirname(__file__), "..", "rewriter", "rewriter.py"),
)
_rw = importlib.util.module_from_spec(_rw_spec)
_rw_spec.loader.exec_module(_rw)

from report import ReportBuilder, Tier, report_to_dict, render_report


# ── Test data ──────────────────────────────────────────────────────

DRAFT_BODY = """Artificial intelligence has transformed the way businesses operate by improving efficiency, reducing costs, and enabling better decision-making across industries.

In banking operations, AI tends to deliver measurable value only in repeatable workflows such as document checking, alert triage, and reconciliation at firms like HSBC and Standard Chartered (Smith, 2022).

Studies show that machine learning algorithms improve fraud detection accuracy by 35%.

According to the World Bank, over 60% of financial institutions have adopted some form of AI technology.

Smith et al. (2022) found that transformer-based models achieve 94.3% accuracy on the SQuAD benchmark but only 67.1% on domain-specific legal question answering.

This performance gap suggests that general-purpose language models may not transfer well to specialised domains without fine-tuning."""

DRAFT_BIB = """Smith, A., Brown, B., and Chen, C. (2022). AI adoption in financial services. Journal of FinTech, 15(3), 112-128.
Jones, D. and Lee, E. (2021). Machine learning in fraud detection. Proceedings of IEEE CSF, 44-59."""

SOURCE_SENTENCES = [
    "Artificial intelligence has transformed the way businesses operate by enhancing efficiency, reducing costs, and enabling better decision-making across a wide range of industries.",
    "Machine learning algorithms have been shown to improve fraud detection accuracy by 35% in controlled studies.",
    "According to the World Bank, over 60% of financial institutions have adopted some form of AI technology.",
    "Smith et al. (2022) found that transformer-based models achieve 94.3% accuracy on the SQuAD benchmark but only 67.1% on domain-specific legal question answering.",
    "This performance gap suggests that general-purpose language models may not transfer well to specialised domains without fine-tuning.",
]

# Pre-written rewrites for multi-pass (no API needed)
REWRITE_MAP = {
    1: """AI adoption in financial services has shifted from broad transformation promises toward narrower, measurable gains in repeatable workflows, as seen at HSBC and Standard Chartered (Smith, 2022).

In banking operations specifically, AI tends to deliver measurable value only in repeatable workflows such as document checking, alert triage, and reconciliation.

Machine learning models have demonstrated a 35% improvement in fraud detection accuracy under controlled conditions, though real-world deployment results vary significantly.

The World Bank reports that more than 60% of financial institutions globally have integrated some form of AI into their operations.

Smith and colleagues (2022) reported that transformer architectures reached 94.3% accuracy on the SQuAD reading comprehension benchmark, compared with just 67.1% on domain-specific legal question answering tasks.

This performance gap highlights the challenges of transferring general-purpose language models to specialised domains without targeted fine-tuning.""",
}


def claude_rewrite_fn(text, span_info):
    if "has transformed the way businesses operate" in text:
        return REWRITE_MAP[1]
    return text


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("DraftProof Report -- Full Pipeline Demo\n")
    t_start = time.time()

    # Init scanners
    print("Loading scanners...")
    pred_scanner = _pred.PredictabilityScanner(model_name="gpt2-medium")
    sim_scanner = _sim.SimilarityScanner()
    cite_scanner = _cite.CitationScanner()
    print("Ready.\n")

    # Run scanners
    print("Scanning predictability...")
    pred_result = pred_scanner.scan_text(DRAFT_BODY)

    print("Scanning similarity...")
    draft_sents = _pred.PredictabilityScanner.split_sentences(pred_scanner, DRAFT_BODY)
    sim_result = sim_scanner.scan(draft_sents, SOURCE_SENTENCES)

    print("Scanning citations...")
    cite_result = cite_scanner.scan(DRAFT_BODY, DRAFT_BIB)

    print("Running multi-pass rewrite...")
    rewrite_result = _rw.multi_pass_rewrite(
        DRAFT_BODY,
        pred_scanner,
        max_passes=2,
        target_top10=0.50,
        min_improvement=0.02,
        rewrite_fn=claude_rewrite_fn,
    )

    scan_time = time.time() - t_start

    # Build report
    builder = ReportBuilder()
    builder.set_meta(scan_time=scan_time, original_text=DRAFT_BODY)
    builder.add_predictability(pred_result)
    builder.add_similarity(sim_result)
    builder.add_citation(cite_result)
    builder.add_rewrite(rewrite_result)

    report = builder.build()

    # ── Render terminal ─────────────────────────────────────────────
    print("\n" + render_report(report, verbose=True))

    # ── Show JSON export ────────────────────────────────────────────
    print("\n\n--- JSON export (first 2000 chars) ---\n")
    as_dict = report_to_dict(report)
    json_str = json.dumps(as_dict, indent=2)
    print(json_str[:2000])
    if len(json_str) > 2000:
        print(f"\n... ({len(json_str)} total chars)")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 76}")
    print(f"  Pipeline: {scan_time:.1f}s")
    print(f"  Tier: {report.overall_tier.value}")
    print(f"  Findings: {report.finding_count}")
    print(f"    Critical: {len(report.findings_by_tier.get('critical', []))}")
    print(f"    High:     {len(report.findings_by_tier.get('high', []))}")
    print(f"    Medium:   {len(report.findings_by_tier.get('medium', []))}")
    print(f"    Low:      {len(report.findings_by_tier.get('low', []))}")
    print(f"{'=' * 76}\n")


if __name__ == "__main__":
    main()
