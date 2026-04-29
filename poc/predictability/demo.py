"""DraftProof Predictability Scanner -- PoC Demo

Run:  cd poc/predictability && python demo.py

Compares generic vs specific writing side-by-side to show the scanner
can distinguish formulaic text from evidence-based writing.
"""

import sys
import time

from scanner import PredictabilityScanner
from feedback import generate_sentence_feedback, generate_overall_feedback


# ── Test texts ──────────────────────────────────────────────────────

TEXTS = {
    "generic_ai_like": (
        "Artificial intelligence has transformed the way businesses operate "
        "by improving efficiency, reducing costs, and enabling better decision-making. "
        "In today's fast-paced world, organizations must leverage cutting-edge technology "
        "to stay competitive. The rapid evolution of AI has unlocked new opportunities "
        "across industries, from healthcare to finance."
    ),
    "specific_evidence": (
        "In banking operations, AI tends to deliver measurable value only in "
        "repeatable workflows such as document checking, alert triage, and "
        "reconciliation at firms like HSBC and Standard Chartered. However, "
        "its usefulness drops sharply when decisions depend on incomplete data, "
        "regulatory judgement, or unclear accountability chains, as the 2023 "
        "FINRA audit of automated surveillance tools demonstrated."
    ),
    "mixed": (
        "Artificial intelligence has transformed the way businesses operate. "
        "However, a 2023 study by the Bank of England found that only 12% of "
        "UK banks had deployed AI in production workflows beyond experimental "
        "pilots. The gap between AI hype and operational reality remains "
        "significant, particularly in areas requiring regulatory judgement "
        "and client-specific interpretation."
    ),
    "academic_cited": (
        "Smith et al. (2022) found that transformer-based models achieve 94.3% "
        "accuracy on the SQuAD benchmark but only 67.1% on domain-specific legal "
        "question answering. This performance gap suggests that general-purpose "
        "language models may not transfer well to specialized professional "
        "contexts without fine-tuning on domain corpora, consistent with earlier "
        "findings by Johnson and Lee (2021) on medical NLP tasks."
    ),
}


# ── Formatting ──────────────────────────────────────────────────────

RISK_ICON = {"high": "[H]", "medium": "[M]", "low": "[L]"}


def print_header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


def print_sentence_detail(result) -> None:
    """Print detailed token-level analysis for one sentence."""
    icon = RISK_ICON.get(result.risk_label, "[?]")
    print(f"  {icon} {result.risk_label.upper():6s}  risk={result.predictability_risk:.4f}")
    print(f"         \"{result.sentence[:90]}{'...' if len(result.sentence) > 90 else ''}\"")
    print(f"         surprisal={result.avg_surprisal:.2f}  "
          f"top10={result.top_10_ratio:.2f}  top50={result.top_50_ratio:.2f}")

    if result.matched_generic_phrases:
        print(f"         generic phrases: {', '.join(result.matched_generic_phrases)}")

    # Top 5 most predictable tokens
    if result.token_results:
        top = sorted(result.token_results, key=lambda t: t.probability, reverse=True)[:5]
        tokens_str = "  ".join(
            f"{t.token!r}(r{t.rank})" for t in top
        )
        print(f"         top tokens: {tokens_str}")

    # Feedback
    for fb in generate_sentence_feedback(result):
        print(f"         -> {fb}")
    print()


def run_comparison(scanner: PredictabilityScanner) -> None:
    """Run scanner on all test texts and display comparison."""
    print_header("COMPARISON: Generic vs Specific Writing")

    labels = list(TEXTS.keys())
    results = {}

    for label, text in TEXTS.items():
        scan = scanner.scan_text(text)
        results[label] = scan

    # Summary table
    print(f"  {'Text':<22s} {'Overall':>8s} {'H/M/L':>8s} {'Shifts':>7s}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*7}")

    for label, scan in results.items():
        dist = scan["risk_distribution"]
        dist_str = f"{dist['high']}/{dist['medium']}/{dist['low']}"
        shifts = len(scan["style_shifts"])
        print(f"  {label:<22s} {scan['overall_risk']:>8.4f} {dist_str:>8s} {shifts:>7d}")

    print()

    # Detailed breakdown for each text
    for label, scan in results.items():
        print_header(f"DETAIL: {label}")
        print(f"  {generate_overall_feedback(scan)}\n")
        for s in scan["sentences"]:
            print_sentence_detail(s)


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("DraftProof Predictability Scanner -- PoC\n")

    print("Loading GPT-2 model...")
    t0 = time.time()
    scanner = PredictabilityScanner(model_name="gpt2")
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s on {scanner.device}\n")

    run_comparison(scanner)

    print_header("TAKEAWAY")
    print("""  The scanner distinguishes between:
    - Generic, formulaic text (high predictability risk)
    - Specific, evidence-based writing (low predictability risk)
    - Mixed text that starts generic but improves with specifics

  This is NOT proof of AI authorship -- it measures writing genericity.
  A human can write generic text; an AI can write specific text.
  The signal is most useful combined with other modules (source grounding,
  draft history, citation checks) in the full DraftProof pipeline.

  Next steps for production:
    1. Label real texts and tune the risk weights / thresholds
    2. Upgrade sentence splitting (spaCy or nltk)
    3. Add domain-specific phrase lists
    4. Compare predictability against the user's own draft history
    5. Integrate as Module 7 in the DraftProof architecture
""")


if __name__ == "__main__":
    main()
