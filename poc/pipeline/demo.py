"""DraftProof Mini-Pipeline -- wires all 3 PoC modules together.

Modules:
  1. Predictability Scanner (gpt2) -- token-level genericity risk
  2. Similarity Scanner (all-MiniLM-L6-v2) -- source overlap detection
  3. Citation Checker -- body vs bibliography cross-check

Run:  cd poc/pipeline && python demo.py
"""

import sys
import os
import time
import json
import importlib

# Load each module's scanner with a unique namespace
def _load_scanner(module_dir, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(os.path.dirname(__file__), "..", module_dir, "scanner.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_pred = _load_scanner("predictability", "pred_scanner")
_sim = _load_scanner("similarity", "sim_scanner")
_cite = _load_scanner("citation", "cite_scanner")

PredictabilityScanner = _pred.PredictabilityScanner
SimilarityScanner = _sim.SimilarityScanner
CitationScanner = _cite.CitationScanner


# ── Test data ───────────────────────────────────────────────────────

DRAFT_BODY = """Artificial intelligence has transformed the way businesses operate by improving efficiency, reducing costs, and enabling better decision-making across industries.

In banking operations, AI tends to deliver measurable value only in repeatable workflows such as document checking, alert triage, and reconciliation at firms like HSBC and Standard Chartered (Smith, 2022).

Studies show that machine learning algorithms improve fraud detection accuracy by 35%.

According to the World Bank, over 60% of financial institutions have adopted some form of AI technology.

Smith et al. (2022) found that transformer-based models achieve 94.3% accuracy on the SQuAD benchmark but only 67.1% on domain-specific legal question answering.

This performance gap suggests that general-purpose language models may not transfer well to specialised domains without fine-tuning."""

DRAFT_BIB = """Smith, A., Brown, B., and Chen, C. (2022). AI adoption in financial services. Journal of FinTech, 15(3), 112-128.
Jones, D. and Lee, E. (2023). Machine learning in Asian banking. International Journal of Banking Technology, 8(1), 45-67."""

SOURCE_SENTENCES = [
    "Artificial intelligence has transformed the way businesses operate by improving efficiency, reducing costs, and enabling better decision-making across industries.",
    "In banking operations, AI tends to deliver measurable value only in repeatable workflows such as document checking, alert triage, and reconciliation.",
    "Smith et al. (2022) found that transformer-based models achieve 94.3% accuracy on the SQuAD benchmark but only 67.1% on domain-specific legal question answering.",
    "However, the usefulness of AI drops sharply when decisions depend on incomplete data, regulatory judgement, or unclear accountability chains.",
]


# ── Sentence splitter ──────────────────────────────────────────────

import re

def split_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\(])', text.strip())
    return [p.strip() for p in parts if p.strip()]


# ── Pipeline runner ────────────────────────────────────────────────

def run_pipeline(body: str, bib: str, source_sentences: list) -> dict:
    """Run all 3 modules and produce a unified report."""
    t_start = time.time()

    sentences = split_sentences(body)

    # 1. Predictability
    print("  [1/3] Loading predictability scanner (gpt2)...")
    t0 = time.time()
    pred_scanner = PredictabilityScanner(model_name="gpt2")
    print(f"        Loaded in {time.time() - t0:.1f}s")
    t0 = time.time()
    pred_result = pred_scanner.scan_text(body)
    pred_time = time.time() - t0

    # 2. Similarity
    print("  [2/3] Loading similarity scanner (all-MiniLM-L6-v2)...")
    t0 = time.time()
    sim_scanner = SimilarityScanner()
    print(f"        Loaded in {time.time() - t0:.1f}s")
    t0 = time.time()
    sim_result = sim_scanner.scan(sentences, source_sentences, source_id="source_paper_1")
    sim_time = time.time() - t0

    # 3. Citation
    print("  [3/3] Running citation checker...")
    t0 = time.time()
    cite_scanner = CitationScanner()
    cite_result = cite_scanner.scan(body, bib)
    cite_time = time.time() - t0

    total_time = time.time() - t_start

    return {
        "total_time": round(total_time, 2),
        "predictability": pred_result,
        "similarity": sim_result,
        "citation": cite_result,
        "pred_time": round(pred_time, 2),
        "sim_time": round(sim_time, 2),
        "cite_time": round(cite_time, 2),
        "sentences": sentences,
    }


# ── Display ─────────────────────────────────────────────────────────

RISK_ICON = {"high": "[H]", "medium": "[M]", "low": "[L]"}


def print_pipeline_report(report: dict) -> None:
    pred = report["predictability"]
    sim = report["similarity"]
    cite = report["citation"]
    sentences = report["sentences"]

    print(f"\n{'=' * 76}")
    print(f"  DRAFTPROOF PRE-SUBMISSION REPORT")
    print(f"{'=' * 76}")
    print(f"  Sentences: {len(sentences)}  |  Total scan time: {report['total_time']}s")
    print(f"  Module times: predictability={report['pred_time']}s  similarity={report['sim_time']}s  citation={report['cite_time']}s")

    # ── Overall risk summary ────────────────────────────────────────
    pred_risk = pred["overall_risk"]
    sim_risk = sim.overall_risk
    cite_issues = cite.stats["missing_from_bib"] + cite.stats["uncited_claims"]

    if pred_risk == "high" or sim_risk == "high" or cite_issues >= 2:
        overall = "REVIEW NEEDED"
    elif pred_risk == "medium" or sim_risk == "medium" or cite_issues >= 1:
        overall = "MINOR CONCERNS"
    else:
        overall = "LOOKS GOOD"

    print(f"\n  Overall: {overall}")
    print(f"  ┌─ Predictability: {pred_risk}")
    print(f"  ├─ Similarity:     {sim_risk}")
    print(f"  └─ Citation:       {cite.stats['missing_from_bib']} missing refs, {cite.stats['uncited_claims']} uncited claims")

    # ── Sentence-by-sentence breakdown ──────────────────────────────
    print(f"\n{'─' * 76}")
    print(f"  SENTENCE-BY-SENTENCE BREAKDOWN")
    print(f"{'─' * 76}")

    pred_sents = pred["sentences"]
    sim_findings_by_draft = {}
    for f in sim.findings:
        key = f.draft_sentence[:60]
        sim_findings_by_draft[key] = f

    for i, sent in enumerate(sentences):
        # Predictability
        if i < len(pred_sents):
            ps = pred_sents[i]
            pred_icon = RISK_ICON.get(ps.risk_label, "[?]")
            pred_tag = f"{pred_icon} {ps.risk_label.upper()}"
            pred_detail = f"risk={ps.predictability_risk:.2f}  surprisal={ps.avg_surprisal:.1f}"
        else:
            pred_tag = "     "
            pred_detail = ""

        # Similarity
        key = sent[:60]
        sim_f = sim_findings_by_draft.get(key)
        if sim_f:
            sim_icon = RISK_ICON.get(sim_f.risk_level, "[?]")
            sim_tag = f"{sim_icon} {sim_f.risk_type.replace('_', ' ').upper()}"
        else:
            sim_tag = "  --"

        print(f"\n  S{i+1}: \"{sent[:70]}{'...' if len(sent) > 70 else ''}\"")
        print(f"       Predictability: {pred_tag:14s} {pred_detail}")
        print(f"       Similarity:     {sim_tag}")

    # ── Citation findings ───────────────────────────────────────────
    if cite.findings:
        print(f"\n{'─' * 76}")
        print(f"  CITATION FINDINGS")
        print(f"{'─' * 76}")
        for f in cite.findings:
            icon = RISK_ICON.get(f.risk_level, "[?]")
            print(f"\n  {icon} {f.risk_level.upper():6s}  {f.finding_type}")
            print(f"    {f.detail}")
            print(f"    -> {f.recommendation}")

    # ── Recommendations ─────────────────────────────────────────────
    print(f"\n{'─' * 76}")
    print(f"  PRIORITY RECOMMENDATIONS")
    print(f"{'─' * 76}")

    recs = []

    # High-predictability sentences
    high_pred = [ps for ps in pred_sents if ps.risk_label == "high"]
    if high_pred:
        recs.append(("HIGH", f"{len(high_pred)} sentence(s) are highly generic/predictable. "
                     "Add specific evidence, cited claims, or your own interpretation."))

    # Similarity findings
    high_sim = [f for f in sim.findings if f.risk_level == "high"]
    if high_sim:
        recs.append(("HIGH", f"{len(high_sim)} sentence(s) closely match source text. "
                     "Rewrite in your own words and ensure citations are present."))

    # Missing citations
    missing = [f for f in cite.findings if f.finding_type == "missing_from_bib"]
    if missing:
        recs.append(("HIGH", f"{len(missing)} citation(s) in text are missing from bibliography."))

    # Uncited claims
    uncited = [f for f in cite.findings if f.finding_type == "uncited_claim"]
    if uncited:
        recs.append(("MEDIUM", f"{len(uncited)} claim(s) lack supporting citations."))

    if not recs:
        recs.append(("LOW", "No significant issues detected. Draft is ready for submission review."))

    for level, rec in recs:
        icon = RISK_ICON.get(level, "[?]")
        print(f"\n  {icon} {rec}")

    print(f"\n{'=' * 76}")
    print(f"  Note: This is a pre-submission integrity check, not a plagiarism")
    print(f"  or AI-authorship verdict. Signals should be reviewed in context.")
    print(f"{'=' * 76}\n")


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("DraftProof Mini-Pipeline -- PoC\n")
    print("Running all 3 modules on draft...\n")

    report = run_pipeline(DRAFT_BODY, DRAFT_BIB, SOURCE_SENTENCES)
    print_pipeline_report(report)

    print(f"Pipeline completed in {report['total_time']}s")


if __name__ == "__main__":
    main()
