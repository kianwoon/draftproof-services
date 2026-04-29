"""Compare GPT-2 model sizes for predictability scoring.

Tests: gpt2 (124M), gpt2-medium (355M), gpt2-large (774M), gpt2-xl (1.5B)
on the same sentences to see which discriminates best.

Run:  cd poc/predictability && python compare_models.py
"""

import time
from scanner import PredictabilityScanner

MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

TEXTS = {
    "generic": (
        "Artificial intelligence has transformed the way businesses operate "
        "by improving efficiency, reducing costs, and enabling better decision-making."
    ),
    "specific": (
        "In banking operations, AI tends to deliver measurable value only in "
        "repeatable workflows such as document checking, alert triage, and "
        "reconciliation at firms like HSBC and Standard Chartered."
    ),
    "academic": (
        "Smith et al. (2022) found that transformer-based models achieve 94.3% "
        "accuracy on the SQuAD benchmark but only 67.1% on domain-specific legal "
        "question answering."
    ),
}

LABEL = {"generic": "GEN", "specific": "SPE", "academic": "ACA"}


def main():
    print("DraftProof -- Model Size Comparison\n")
    print(f"  {'Model':<16s} {'Load(s)':>8s}", end="")
    for key in TEXTS:
        print(f"  {LABEL[key]:>5s}_risk  {LABEL[key]:>5s}_s10  {LABEL[key]:>5s}_sur", end="")
    print()
    print(f"  {'-'*16} {'-'*8}", end="")
    for _ in TEXTS:
        print(f"  {'-'*7}  {'-'*7}  {'-'*7}", end="")
    print()

    for model_name in MODELS:
        try:
            t0 = time.time()
            scanner = PredictabilityScanner(model_name=model_name)
            load_time = time.time() - t0
            print(f"  {model_name:<16s} {load_time:>7.1f}s", end="")

            for key, text in TEXTS.items():
                result = scanner.scan_text(text)
                s = result["sentences"][0]
                print(f"  {s.risk_label:>7s}  {s.top_10_ratio:>7.4f}  {s.avg_surprisal:>7.2f}", end="")

            print()

        except Exception as e:
            print(f"  {model_name:<16s}  FAILED: {e}")

    print()
    print("  Key: risk = label (H/M/L), s10 = top-10 ratio, sur = avg surprisal")
    print("  Better model = bigger gap between GEN and SPE/ACA risk scores")
    print()


if __name__ == "__main__":
    main()
