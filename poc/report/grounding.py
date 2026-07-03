"""Authorship-concern and grounding helpers + top-k calibration.

Extracted from report.py. Leaf helpers consumed by ReportBuilder.build()
and report_to_dict().
"""
import re
from typing import Any, Dict, Optional

from detect.topk_calibration import calibrate_topk_risk


def _topk_calibration_fields_for_summary(
    pred_summary: Any,
    raw_topk_pattern: Any,
    criterion_scores: Dict[str, Any] | None,
) -> Dict[str, Any]:
    details = {}
    if criterion_scores and "topk_predictability" in criterion_scores:
        cs = criterion_scores["topk_predictability"]
        details = cs.get("details", {}) if isinstance(cs, dict) else getattr(cs, "details", {}) or {}
    eligible_sentence_count = 0
    if pred_summary is not None:
        sentences = getattr(pred_summary, "sentences", None)
        if isinstance(sentences, list):
            eligible_sentence_count = sum(1 for item in sentences if item)
    return calibrate_topk_risk(
        raw_topk_pattern,
        avg_top10_ratio=(details or {}).get("avg_top10_ratio"),
        eligible_sentence_count=eligible_sentence_count,
    )


def concern_tier_from_score(score: float) -> str:
    if score >= 0.65:
        return "urgent_review"
    if score >= 0.40:
        return "needs_attention"
    if score >= 0.25:
        return "review_recommended"
    if score >= 0.15:
        return "light_review"
    return "clear"


def is_weak_only(signals: Optional[Dict[str, Any]]) -> bool:
    if not signals:
        return False
    strong = {"source_grounding", "citation_integrity", "draft_evolution", "structural_reuse"}
    has_weak = any(signals.get(k) is not None and signals.get(k, 0) > 0
                   for k in ("predictability", "genericity", "specificity"))
    has_strong = any(signals.get(k) is not None and signals.get(k, 0) >= 0.25 for k in strong)
    return has_weak and not has_strong


def estimate_in_text_source_grounding_strength(text: str) -> float:
    """Bounded source strength from in-text source relationships without a bibliography object."""
    text = text or ""
    url_count = len(re.findall(r"\b(?:https?://|www\.|doi\.org/)\S+", text, flags=re.I))
    has_reference_section = bool(re.search(
        r"(?im)^\s*(?:references|reference list|bibliography|works cited|sources)\s*$",
        text,
    ))
    reference_tail = ""
    if has_reference_section:
        parts = re.split(
            r"(?im)^\s*(?:references|reference list|bibliography|works cited|sources)\s*$",
            text,
            maxsplit=1,
        )
        reference_tail = parts[-1] if len(parts) > 1 else ""
    reference_years = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", reference_tail))
    parenthetical = len(re.findall(
        r"\((?:[A-Z][A-Za-z'’.-]+(?:\s+(?:&|and)\s+[A-Z][A-Za-z'’.-]+)?|[A-Z][A-Za-z'’.-]+\s+et\s+al\.?|[A-Z]{2,})\s*,?\s*(?:19|20)\d{2}[a-z]?\)",
        text,
    ))
    narrative = len(re.findall(
        r"\b[A-Z][A-Za-z'’.-]+(?:\s+(?:and|&|et\s+al\.?)\s+[A-Z][A-Za-z'’.-]+)*\s*\((?:19|20)\d{2}[a-z]?\)",
        text,
    ))
    source_relations = len(re.findall(
        r"\b(?:states|argues|explains|shows|describes|defines|discusses|notes?|offers|focus(?:es)? on|highlight(?:s)?|according to)\b",
        text,
        flags=re.I,
    ))
    citation_count = parenthetical + narrative
    reference_signal = max(url_count, reference_years if has_reference_section else 0)
    if citation_count >= 6 and source_relations >= 4:
        return 0.70
    if citation_count >= 4 and source_relations >= 2:
        return 0.60
    if reference_signal >= 3 and source_relations >= 2:
        return 0.55
    if has_reference_section and reference_signal >= 2:
        return 0.50
    if citation_count >= 2 and source_relations >= 1:
        return 0.45
    if reference_signal >= 1 and source_relations >= 1:
        return 0.40
    if citation_count >= 1:
        return 0.35
    if has_reference_section and reference_signal >= 1:
        return 0.30
    return 0.0
