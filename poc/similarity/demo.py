"""Similarity Scanner -- PoC Demo

Tests the multi-layer overlap detection against realistic scenarios:
  1. Exact copy (no citation)
  2. Close paraphrase (no citation)
  3. Close paraphrase (with citation)
  4. Semantic overlap (different words, same idea)
  5. Original writing (no match)

Run:  cd poc/similarity && python demo.py
"""

import time
from scanner import SimilarityScanner, extract_citations

# ── Source text (simulating a real paper) ────────────────────────────

SOURCE = [
    "Artificial intelligence has transformed the way businesses operate by improving "
    "efficiency, reducing costs, and enabling better decision-making across industries.",
    "In banking operations, AI tends to deliver measurable value only in repeatable "
    "workflows such as document checking, alert triage, and reconciliation.",
    "However, the usefulness of AI drops sharply when decisions depend on incomplete "
    "data, regulatory judgement, or unclear accountability chains.",
    "A 2023 FINRA audit of automated surveillance tools revealed significant gaps in "
    "how banks monitor AI-driven trading decisions.",
    "Smith et al. (2022) found that transformer-based models achieve 94.3% accuracy "
    "on the SQuAD benchmark but only 67.1% on domain-specific legal question answering.",
]

# ── Draft sentences with varying overlap levels ─────────────────────

DRAFT = {
    "exact_copy_no_cite": [
        "Artificial intelligence has transformed the way businesses operate by improving "
        "efficiency, reducing costs, and enabling better decision-making across industries.",
        "This technology is widely adopted in many sectors today.",
    ],
    "close_paraphrase_no_cite": [
        "AI has revolutionized business operations by boosting efficiency, cutting costs, "
        "and facilitating improved decision-making in various sectors.",
        "Companies are investing heavily in these solutions.",
    ],
    "close_paraphrase_with_cite": [
        "According to recent research, AI has revolutionized business operations by "
        "boosting efficiency and cutting costs (Smith et al., 2022).",
        "These findings are consistent across sectors.",
    ],
    "semantic_overlap": [
        "Machine learning systems have fundamentally changed corporate operations by "
        "enhancing productivity, lowering expenses, and supporting more informed choices.",
        "Financial institutions have been particularly affected.",
    ],
    "original": [
        "The ethical implications of deploying AI in clinical settings remain "
        "underexplored, particularly regarding patient consent for algorithmic diagnoses.",
        "Regulatory frameworks in the EU and US diverge significantly on this issue.",
    ],
}


# ── Display ─────────────────────────────────────────────────────────

RISK_ICON = {"high": "[H]", "medium": "[M]", "low": "[L]"}


def print_case(name: str, result) -> None:
    print(f"\n{'=' * 72}")
    print(f"  CASE: {name}")
    print(f"  Overall risk: {result.overall_risk}")
    print(f"  Distribution: {result.risk_distribution}")
    print(f"{'=' * 72}")

    if not result.findings:
        print("  No findings -- draft appears original against this source.\n")
        return

    for f in result.findings:
        icon = RISK_ICON.get(f.risk_level, "[?]")
        print(f"  {icon} {f.risk_level.upper():6s}  type={f.risk_type}")
        print(f"    Draft:  \"{f.draft_sentence[:85]}{'...' if len(f.draft_sentence) > 85 else ''}\"")
        print(f"    Source: \"{f.source_sentence[:85]}{'...' if len(f.source_sentence) > 85 else ''}\"")
        print(f"    Scores: exact={f.exact_score:.3f}  fuzzy={f.fuzzy_score:.3f}  "
              f"semantic={f.semantic_score:.3f}  cited={f.citation_nearby}")
        print(f"    -> {f.recommendation}")
        print()


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("DraftProof Similarity Scanner -- PoC\n")
    print("Loading sentence embeddings (all-MiniLM-L6-v2)...")
    t0 = time.time()
    scanner = SimilarityScanner()
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    print("Source document has", len(SOURCE), "sentences.\n")

    # Summary table
    print(f"  {'Case':<30s} {'Overall':>8s} {'H':>3s} {'M':>3s} {'L':>3s}  Findings")
    print(f"  {'-'*30} {'-'*8} {'-'*3} {'-'*3} {'-'*3}  {'-'*10}")

    results = {}
    for name, draft in DRAFT.items():
        result = scanner.scan(draft, SOURCE, source_id="source_paper_1")
        results[name] = result
        dist = result.risk_distribution
        n = len(result.findings)
        print(f"  {name:<30s} {result.overall_risk:>8s} {dist['high']:>3d} {dist['medium']:>3d} {dist['low']:>3d}  {n:>3d} matches")

    print()

    # Detailed findings
    for name, result in results.items():
        print_case(name, result)

    # ── Takeaway ────────────────────────────────────────────────────
    print("=" * 72)
    print("  TAKEAWAY")
    print("=" * 72)
    print("""
  The multi-layer scanner detects:
    1. Exact copy -- flagged HIGH regardless of citation
    2. Close paraphrase without citation -- flagged HIGH
    3. Close paraphrase with citation -- downgraded to LOW (still noted)
    4. Semantic overlap -- detected via embeddings even with different words
    5. Original writing -- no false positives

  Key design choices:
    - Citation presence lowers risk level (paraphrase + cite = low concern)
    - Top-3 semantic candidates checked for exact/fuzzy (catches reranking misses)
    - Context window checks nearby sentences for citations

  Next steps:
    1. Benchmark against real plagiarism datasets (e.g., PAN shared task)
    2. Add paragraph-level chunking for longer documents
    3. Tune thresholds with labelled data
    4. Add BM25 / lexical search for scalable source matching
""")


if __name__ == "__main__":
    main()
