"""DeBERTa learned-classifier signal metadata and authorship rating.

Extracted from report.py. Drives the Signal-highlights paragraph colors
(replacing the perplexity-family signals) and the DeBERTa-derived rating.
"""
from typing import Any


# DeBERTa learned-classifier heatmap: band → color/label/description/tier. Drives the
# Signal-highlights paragraph colors (replacing the perplexity-family signals). Distinct colors
# per band carried on the signal so the frontend's --signal-color mechanism needs no new CSS.
DEBERTA_HEAT_COLORS = {
    "clean": "#94a3b8",     # neutral slate — human-like; not surfaced in the legend
    "moderate": "#f97316",  # orange — clear AI-like reading (0.80-0.99)
    "high": "#dc2626",      # red — the >=0.99 high-confidence AI band
}
DEBERTA_HEAT_TIERS = {"clean": "", "moderate": "medium", "high": "high"}
DEBERTA_HEAT_LABELS = {
    "clean": "No AI signal",
    "moderate": "Strong AI signal",
    "high": "High-confidence AI signal",
}
DEBERTA_HEAT_DESCRIPTIONS = {
    "clean": "The learned classifier reads this passage as human.",
    "moderate": "The second-opinion detector sees strong AI-like signal (below its high-confidence bar).",
    "high": "The second-opinion detector is >=99% confident this passage is AI-like.",
}
# Band-specific, student-facing edit guidance. Drives the issue-card "recommendation" so the
# advice is native to the learned-classifier signal (not the abandoned perplexity family).
DEBERTA_HEAT_RECOMMENDATIONS = {
    "moderate": "This sentence reads as strongly AI-like under the learned classifier. Rewrite it around a specific, verifiable detail from your experience, and vary the sentence rhythm so it does not follow a common template.",
    "high": "The learned classifier is highly confident this sentence is AI-generated. Revoice it entirely in your own words and tie every claim to a concrete detail only you would know (a name, a number, an observation).",
}
# Plain-language reader summary per band — what a human reviewer would notice.
DEBERTA_HEAT_READER_SUMMARY = {
    "moderate": "A reader may notice this sentence follows a familiar, template-like structure common in AI-assisted writing.",
    "high": "A reader may notice this sentence reads as machine-generated — fluent but generic, without the texture of personal experience.",
}


def deberta_authorship_rating(authoritative_score: dict | None, layer3: Any) -> dict:
    """Build a DeBERTa-derived authorship rating (the same shape as Layer3.authorship_rating)
    so the rating's score/label match the DeBERTa badge instead of the perplexity components.

    Maps the authoritative >=0.99-proportion score to a label/code the frontend understands, and
    carries the writing-quality fields from Layer3 (those are perplexity-independent)."""
    if not authoritative_score or not authoritative_score.get("available"):
        return layer3.authorship_rating
    score_pct = round(authoritative_score["ai_likelihood_score"] * 100, 2)
    n_hc = authoritative_score.get("n_high_confidence", 0)
    n_sc = authoritative_score.get("n_scored", 0)
    if score_pct >= 50:
        label, short, code, level = "High AI-writing signal", "High AI signal", "likely_ai", "high"
        action = "Revoice the flagged passages in your own words with concrete, verifiable detail."
        summary = f"The learned classifier is >=99% confident that {n_hc} of {n_sc} sentences are AI-generated."
    elif score_pct >= 25:
        label, short, code, level = "Moderate AI-writing signal", "Moderate AI signal", "possibly_ai", "moderate"
        action = "Review the high-confidence sentences and revoice any that read as template-like."
        summary = f"{n_hc} of {n_sc} sentences read as high-confidence AI under the learned classifier."
    elif score_pct >= 10:
        label, short, code, level = "Low AI-writing signal", "Low AI signal", "unlikely_ai", "low"
        action = "A few sentences flagged for review; mostly reads as human."
        summary = f"A small number of sentences ({n_hc} of {n_sc}) read as high-confidence AI."
    else:
        label, short, code, level = "Minimal AI signal", "Minimal AI signal", "minimal_ai", "low"
        action = "The learned classifier reads this as human writing."
        summary = "No sentences reached the high-confidence AI bar."
    return {
        "label": label,
        "short_label": short,
        "risk_level": level,
        "summary": summary,
        "recommended_action": action,
        "code": code,
        "score": score_pct,
        "confidence": authoritative_score.get("confidence", "low"),
        "ai_tier": authoritative_score.get("tier", "green").upper(),
        "writing_quality_score": round(layer3.writing_quality_score * 100, 2),
        "writing_quality_tier": layer3.writing_quality_tier.value,
        "is_verdict": False,
        "disclaimer": "This rating summarizes DraftProof detector signals. It is not proof of authorship.",
        "caution_notes": [],
    }
