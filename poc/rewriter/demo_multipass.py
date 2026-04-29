"""Multi-pass Rewrite Demo — Claude acts as the rewrite engine.

Demonstrates the iterative rewrite loop:
  Pass 0: Original text
  Pass 1: First rewrite (target flagged spans)
  Pass 2: Second rewrite (target remaining predictable patterns)
  Pass 3: Optional third pass if improvement continues

Run:  cd poc/rewriter && python demo_multipass.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "predictability"))

from rewriter import (
    PredictabilityScanner,
    multi_pass_rewrite,
    MultiPassResult,
    compute_metrics,
)

# ── Test texts ──────────────────────────────────────────────────────

NEWS_TEXT = '''"The guy is a sick guy," Trump told Fox News. "His sister or his brother actually was complaining about it. They were even complaining to law enforcement."

Trump said at a hastily arranged late-night news conference at the White House that he first thought the noise was a tray being dropped, before realizing it was gunfire.

"They seem to think he was a lone wolf, and I feel that too," the president said. One officer was shot at close range in his safety vest and appeared to be not seriously harmed.'''

PASS1_TEXT = '''Trump characterized the assailant as "deeply disturbed" during a Fox News interview, noting that the suspect's siblings had repeatedly raised concerns about his behavior to local authorities.

Speaking at an impromptu midnight press briefing from the White House, Trump recounted initially assuming the sound was crockery shattering, before recognizing the distinctive crack of gunfire.

The president echoed investigators' preliminary assessment that the attacker likely acted alone. A uniformed officer absorbed a point-blank round to his ballistic vest, sustaining what officials described as minor injuries.'''

PASS2_TEXT = '''Trump described the gunman as "psychologically unwell" in remarks to Fox News, adding that the suspect's own family members had flagged warning signs to police on multiple occasions prior to the incident.

Addressing reporters in a sudden midnight briefing at the executive mansion, Trump recalled mistaking the initial burst for breaking dinnerware, until the staccato rhythm of gunfire became unmistakable.

Echoing the consensus among federal investigators, the president conceded the perpetrator appeared to have operated without accomplices. One Secret Service detail member took a near-contact gunshot wound to his protective body armor, emerging with what medical staff termed superficial contusions.'''

# Map: pass number → pre-written rewrite
REWRITE_MAP = {1: PASS1_TEXT, 2: PASS2_TEXT}


def claude_rewrite_fn(text: str, span_info: str) -> str:
    """Simulates Claude API by returning pre-written rewrites."""
    # Determine which pass based on text match
    if "The guy is a sick guy" in text:
        return REWRITE_MAP[1]
    elif "characterized the assailant" in text:
        return REWRITE_MAP[2]
    return text  # No more rewrites available


# ── Display helpers ─────────────────────────────────────────────────

RISK_ICON = {"high": "[H]", "medium": "[[M]]", "low": "[L]"}


def print_pass_metrics(pm, label=""):
    """Print metrics for a single pass."""
    icon_map = {"high": "[H]", "medium": "[M]", "low": "[L]"}
    print(f"  {label}")
    print(f"    Risk:     {pm.risk:.4f}")
    print(f"    Top-10:   {pm.top10_ratio:.2f} ({pm.top10_ratio:.0%})")
    print(f"    Surprisal:{pm.surprisal:.2f}")
    print()
    for sd in pm.sentence_details:
        icon = icon_map.get(sd["label"], "[?]")
        print(f"    {icon} S{sd['index']+1}  risk={sd['risk']:.3f}  top10={sd['top10_ratio']:.0%}  surprisal={sd['surprisal']:.1f}")
    print()


def print_progression(result: MultiPassResult):
    """Print the full multi-pass progression table."""
    print(f"\n  {'=' * 72}")
    print(f"  MULTI-PASS PROGRESSION")
    print(f"  {'=' * 72}")
    print()

    # Header
    print(f"  {'Pass':<6} {'Risk':>8} {'Top-10':>8} {'Surprisal':>10} {'d Risk':>8} {'d Top10':>8} {'Label':>8}")
    print(f"  {'-' * 62}")

    all_metrics = [result.original_metrics] + result.passes
    for i, m in enumerate(all_metrics):
        label = "Original" if i == 0 else f"Pass {m.pass_number}"
        d_risk = ""
        d_top10 = ""
        if i > 0:
            prev = all_metrics[i - 1]
            d_risk = f"{prev.risk - m.risk:>+.4f}"
            d_top10 = f"{prev.top10_ratio - m.top10_ratio:>+.2f}"
        print(f"  {label:<8} {m.risk:>8.4f} {m.top10_ratio:>7.0%} {m.surprisal:>10.2f} {d_risk:>8} {d_top10:>8}")

    # Convergence
    print()
    if result.converged:
        print(f"  Converged: {result.convergence_reason}")
    else:
        print(f"  Stopped:   {result.convergence_reason}")

    # Total delta
    d_risk = result.original_metrics.risk - result.final_metrics.risk
    d_top10 = result.original_metrics.top10_ratio - result.final_metrics.top10_ratio
    d_surp = result.final_metrics.surprisal - result.original_metrics.surprisal
    print(f"\n  Total improvement:")
    print(f"    Risk:     {result.original_metrics.risk:.4f} -> {result.final_metrics.risk:.4f}  ({d_risk:+.4f})")
    print(f"    Top-10:   {result.original_metrics.top10_ratio:.0%} -> {result.final_metrics.top10_ratio:.0%}  ({d_top10:+.2f})")
    print(f"    Surprisal:{result.original_metrics.surprisal:.2f} -> {result.final_metrics.surprisal:.2f}  ({d_surp:+.2f})")


def print_text_comparison(result: MultiPassResult):
    """Show original vs final text side by side."""
    print(f"\n  {'=' * 72}")
    print(f"  TEXT COMPARISON")
    print(f"  {'=' * 72}")

    print(f"\n  --- ORIGINAL (risk={result.original_metrics.risk:.3f}, top10={result.original_metrics.top10_ratio:.0%}) ---")
    for line in result.original_text.strip().split("\n"):
        if line.strip():
            print(f"  {line.strip()}")

    print(f"\n  --- FINAL (risk={result.final_metrics.risk:.3f}, top10={result.final_metrics.top10_ratio:.0%}) ---")
    for line in result.final_text.strip().split("\n"):
        if line.strip():
            print(f"  {line.strip()}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("DraftProof Multi-Pass Rewriter -- PoC Demo\n")
    print("Loading scanner (gpt2-medium)...")
    scanner = PredictabilityScanner(model_name="gpt2-medium")
    print("Ready.\n")

    print("#" * 76)
    print("  MULTI-PASS REWRITE LOOP")
    print("  Using pre-written Claude rewrites (no API key needed)")
    print("#" * 76)

    result = multi_pass_rewrite(
        NEWS_TEXT,
        scanner,
        max_passes=3,
        target_top10=0.50,
        min_improvement=0.02,
        rewrite_fn=claude_rewrite_fn,
    )

    # Show each pass
    print(f"\n  {'=' * 72}")
    print(f"  PASS 0: ORIGINAL")
    print(f"  {'=' * 72}")
    print_pass_metrics(result.original_metrics, "Original text metrics")

    for pm in result.passes:
        print(f"  {'=' * 72}")
        print(f"  PASS {pm.pass_number}")
        print(f"  {'=' * 72}")
        print_pass_metrics(pm, f"After pass {pm.pass_number}")

    # Progression table
    print_progression(result)

    # Text comparison
    print_text_comparison(result)

    # Usage notes
    print(f"\n  {'=' * 72}")
    print(f"  HOW IT WORKS")
    print(f"  {'=' * 72}")
    print("""
  Multi-pass rewrite loop:
    1. Scan text -> compute predictability metrics
    2. If top-10 ratio <= target (50%) -> DONE
    3. If improvement < min_improvement (2%) -> converged, DONE
    4. Send text + span info to Claude for rewrite
    5. Re-scan rewritten text -> back to step 2
    6. Stop after max_passes (3) if neither condition met

  Convergence conditions:
    - Target top-10 ratio reached (success)
    - Diminishing returns (near floor for this genre)
    - Max passes exhausted (user can increase)
    - API call failure (return current best)

  Production notes:
    - rewrite_fn can be swapped for any LLM (local, API, etc.)
    - Set target_top10 based on genre (news: 0.50, academic: 0.45)
    - Store all passes so user can pick intermediate version
    - Add diff view between passes for transparency
""")


if __name__ == "__main__":
    main()
