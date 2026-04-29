"""Feedback generator -- turns scan results into actionable user advice.

Never says "AI-generated" or "cheating". Focuses on writing quality.
"""

from typing import List, Dict, Any

from scanner import SentenceResult


def generate_sentence_feedback(result: SentenceResult) -> List[str]:
    """Generate human-readable feedback for a single sentence."""
    feedback = []

    if result.error:
        return [f"Skipped: {result.error}"]

    if result.risk_label == "high":
        feedback.append(
            "This sentence is highly predictable and may read as generic."
        )
    elif result.risk_label == "medium":
        feedback.append(
            "This sentence is moderately predictable."
        )

    if result.matched_generic_phrases:
        phrases = ", ".join(f'"{p}"' for p in result.matched_generic_phrases)
        feedback.append(f"Contains common filler: {phrases}")

    if result.top_10_ratio > 0.5:
        feedback.append(
            "Over half the tokens were among the model's top-10 predictions "
            "-- suggests formulaic wording."
        )
    elif result.top_10_ratio > 0.35:
        feedback.append(
            "Many tokens are common next-token choices, indicating predictable phrasing."
        )

    if result.avg_surprisal < 3.0:
        feedback.append(
            "Low surprisal overall -- the wording follows very common patterns."
        )

    # Always give an actionable recommendation
    if result.risk_label in ("high", "medium"):
        feedback.append(
            "Recommended: add domain-specific evidence, a cited claim, "
            "a limitation, or your own interpretation."
        )

    return feedback


def generate_overall_feedback(scan: Dict[str, Any]) -> str:
    """Generate a summary paragraph for the full scan."""
    risk = scan["overall_risk"]
    dist = scan["risk_distribution"]
    total = sum(dist.values())

    if total == 0:
        return "No scorable sentences found."

    lines = []

    # Overall assessment
    if risk >= 0.55:
        lines.append("Overall: high predictability risk.")
    elif risk >= 0.35:
        lines.append("Overall: moderate predictability risk.")
    else:
        lines.append("Overall: low predictability risk -- writing appears specific and grounded.")

    # Distribution
    lines.append(
        f"Sentence breakdown: {dist['high']} high, "
        f"{dist['medium']} medium, {dist['low']} low risk."
    )

    # Style shifts
    if scan["style_shifts"]:
        lines.append(
            f"Detected {len(scan['style_shifts'])} style shift(s) "
            f"between consecutive sentences."
        )

    # Disclaimer
    lines.append(
        "\nNote: predictability risk reflects writing genericity, "
        "not authorship. It is one signal among many."
    )

    return "\n".join(lines)
