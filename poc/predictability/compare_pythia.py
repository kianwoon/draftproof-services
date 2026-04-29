"""Compare gpt2-medium vs pythia-1.4b on predictability scoring.

Pythia is trained on more data (The Pile) with public training details,
making it a strong open alternative for this use case.

Run:  cd poc/predictability && python compare_pythia.py
"""

import time
from scanner import PredictabilityScanner

MODELS = ["gpt2-medium", "EleutherAI/pythia-1.4b"]

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
    print("DraftProof -- GPT-2-Medium vs Pythia-1.4B\n")

    header = (
        f"  {'Model':<26s} {'Load':>6s}"
        f"  {'GEN_risk':>8s} {'GEN_s10':>7s} {'GEN_sur':>7s}"
        f"  {'SPE_risk':>8s} {'SPE_s10':>7s} {'SPE_sur':>7s}"
        f"  {'ACA_risk':>8s} {'ACA_s10':>7s} {'ACA_sur':>7s}"
        f"  {'GEN-SPE gap':>11s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model_name in MODELS:
        try:
            t0 = time.time()
            scanner = PredictabilityScanner(model_name=model_name)
            load_time = time.time() - t0

            results = {}
            for key, text in TEXTS.items():
                r = scanner.scan_text(text)
                results[key] = r["sentences"][0]

            gen_risk = results["generic"].predictability_risk
            spe_risk = results["specific"].predictability_risk
            gap = round(gen_risk - spe_risk, 4)

            print(
                f"  {model_name:<26s} {load_time:>5.1f}s"
                f"  {results['generic'].risk_label:>8s} {results['generic'].top_10_ratio:>7.4f} {results['generic'].avg_surprisal:>7.2f}"
                f"  {results['specific'].risk_label:>8s} {results['specific'].top_10_ratio:>7.4f} {results['specific'].avg_surprisal:>7.2f}"
                f"  {results['academic'].risk_label:>8s} {results['academic'].top_10_ratio:>7.4f} {results['academic'].avg_surprisal:>7.2f}"
                f"  {gap:>11.4f}"
            )

        except Exception as e:
            print(f"  {model_name:<26s}  FAILED: {e}")

    print()
    print("  Gap = GEN risk - SPE risk. Higher = better discrimination.")
    print()


if __name__ == "__main__":
    main()
