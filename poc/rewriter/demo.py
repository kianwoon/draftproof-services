"""Predictability Rewriter -- PoC Demo

Tests the rewriter on the news text that scored HIGH (0.55).
Flags predictable spans and calls Claude for rewrite suggestions.

Run:  cd poc/rewriter && python demo.py
"""

from rewriter import (
    PredictabilityScanner,
    extract_rewritable_spans,
    call_claude_rewrite,
    rewrite_text,
)
import sys
import os
import time

# ── Test texts ──────────────────────────────────────────────────────

NEWS_TEXT = '''"The guy is a sick guy," Trump told Fox News. "His sister or his brother actually was complaining about it. They were even complaining to law enforcement."

Trump said at a hastily arranged late-night news conference at the White House that he first thought the noise was a tray being dropped, before realizing it was gunfire.

"They seem to think he was a lone wolf, and I feel that too," the president said. One officer was shot at close range in his safety vest and appeared to be not seriously harmed.'''

BADMINTON_TEXT = '''A thrilling Men's Singles battle between Kodai Naraoka of JPN and Shi Yu Qi of CHN in the 2026 badminton season. Featuring relentless rallies, elite defense, tactical shot-making, and world-class intensity, this clash delivers top-level international badminton action'''

ACADEMIC_TEXT = '''This paper presents a novel approach to the problem. Our method significantly outperforms existing baselines. The results demonstrate the effectiveness of our proposed framework. We believe this work opens new avenues for future research.'''


# ── Display ─────────────────────────────────────────────────────────

RISK_ICON = {"high": "[H]", "medium": "[M]", "low": "[L]"}


def print_span_analysis(text_name: str, text: str, scanner: PredictabilityScanner) -> None:
    """Show flagged spans before rewrite suggestions."""
    print(f"\n{'=' * 76}")
    print(f"  SPAN ANALYSIS: {text_name}")
    print(f"{'=' * 76}")

    pred = scanner.scan_text(text)

    # Sentence overview
    for i, s in enumerate(pred["sentences"]):
        icon = RISK_ICON.get(s.risk_label, "[?]")
        print(f"\n  S{i+1} {icon} {s.risk_label.upper()} risk={s.predictability_risk:.2f}  top-10={s.top_10_ratio:.0%}")
        # Show token-level annotation
        annotated = ""
        for t in s.token_results:
            if t.rank <= 10:
                annotated += f"<<{t.token}>>"
            else:
                annotated += t.token
        # Truncate for display
        if len(annotated) > 100:
            annotated = annotated[:100] + "..."
        print(f"     {annotated}")

    # Extract spans
    spans = extract_rewritable_spans(pred["sentences"])
    if not spans:
        print("\n  No rewritable spans found.")
        return

    print(f"\n  {'─' * 72}")
    print(f"  TOP REWRITABLE SPANS ({len(spans)} found)")
    print(f"  {'─' * 72}")

    for i, sp in enumerate(spans[:5]):
        print(f"\n  [{i+1}] \"{sp.span_text[:60]}{'...' if len(sp.span_text) > 60 else ''}\"")
        print(f"      S{sp.sentence_index+1}  top-10={sp.top10_ratio:.0%}  avg_rank={sp.avg_rank:.0f}  tokens={sp.token_count}")


def print_rewrite_suggestions(results: list) -> None:
    """Show rewrite suggestions for each span."""
    print(f"\n{'=' * 76}")
    print(f"  REWRITE SUGGESTIONS")
    print(f"{'=' * 76}")

    for i, r in enumerate(results):
        print(f"\n  [{i+1}] ORIGINAL (S{r.sentence_index+1}, top-10={r.original_metrics['top10_ratio']:.0%})")
        print(f"      \"{r.original_span[:70]}{'...' if len(r.original_span) > 70 else ''}\"")

        if not r.alternatives:
            print(f"      No suggestions available.")
            continue

        for j, alt in enumerate(r.alternatives):
            print(f"\n    Alt {j+1}: \"{alt['rewrite'][:80]}\"")
            print(f"           {alt.get('reason', '')}")


def print_before_after(text: str, scanner: PredictabilityScanner) -> None:
    """Scan and show aggregate metrics."""
    pred = scanner.scan_text(text)
    print(f"  Overall: {pred['overall_risk']}")
    print(f"  Avg top-10 ratio: {pred['average_top_10_ratio']:.2f}")
    print(f"  Avg surprisal:    {pred['average_surprisal']:.2f}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("DraftProof Predictability Rewriter -- PoC\n")

    print("Loading predictability scanner (gpt2)...")
    scanner = PredictabilityScanner(model_name="gpt2")
    print("Ready.\n")

    # 1. Show span analysis for all texts
    for name, text in [
        ("News reporting", NEWS_TEXT),
        ("Badminton promo", BADMINTON_TEXT),
        ("Academic filler", ACADEMIC_TEXT),
    ]:
        print_span_analysis(name, text, scanner)

    # 2. Get rewrite suggestions for news text
    print(f"\n\n{'#' * 76}")
    print(f"  REWRITE PIPELINE: News reporting (highest risk)")
    print(f"{'#' * 76}")
    print(f"\n  Calling Claude API for rewrite suggestions...")

    t0 = time.time()
    results = rewrite_text(NEWS_TEXT, scanner, max_spans=4)
    elapsed = time.time() - t0

    print(f"  Done in {elapsed:.1f}s\n")
    print_rewrite_suggestions(results)

    # 3. Before/after comparison
    print(f"\n\n{'=' * 76}")
    print(f"  HOW TO USE THESE REWRITES")
    print(f"{'=' * 76}")
    print("""
  The rewriter works in 3 steps:
    1. SCAN  → identify predictable token spans (<<marked>>)
    2. FLAG  → rank spans by top-10 ratio (most predictable first)
    3. SUGGEST → call Claude API for 3 alternatives per span

  User workflow:
    - Review each flagged span
    - Pick an alternative or write your own
    - Re-scan to verify improvement
    - Iterate until top-10 ratio drops below threshold

  Production considerations:
    - Batch API calls for efficiency (parallel per span)
    - Cache rewrites to avoid re-processing
    - Allow user to accept/reject/tweak each suggestion
    - Show before/after metrics (top-10 ratio delta)
    - Respect quotes from named sources (don't rewrite)
""")


if __name__ == "__main__":
    main()
