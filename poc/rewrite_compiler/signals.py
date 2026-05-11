"""Signal normalization for deterministic rewrite compilation."""

from __future__ import annotations

import re
from typing import Any


STOPWORDS = {
    "about", "above", "across", "after", "again", "against", "also",
    "among", "another", "around", "because", "before", "being", "both",
    "could", "does", "doing", "done", "each", "either", "else", "even",
    "every", "from", "further", "have", "having", "here", "however",
    "into", "itself", "just", "like", "many", "might", "more", "most",
    "much", "must", "need", "only", "other", "over", "same", "should",
    "since", "some", "still", "such", "than", "that", "their", "them",
    "then", "there", "these", "they", "this", "those", "through",
    "under", "until", "very", "what", "when", "where", "which", "while",
    "with", "within", "without", "would", "important", "significant",
    "major", "modern", "global", "different", "various",
}

PROTECTED_ANCHOR_RE = re.compile(
    r"(?:\b\d{3,4}\b|https?://|www\.|\[[^\]]+\]|\([^)]*\d{4}[^)]*\)|"
    r"\b[A-Z]{2,}[A-Z0-9-]{2,}\b)"
)


def pct(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(str(value))
        except (TypeError, ValueError):
            return float(default)
    return round(number * 100.0, 3) if abs(number) <= 1.0 else round(number, 3)


def split_sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", str(text or "").strip())
        if item.strip()
    ]


def logical_paragraphs(text: str) -> list[str]:
    paragraphs = [
        re.sub(r"\s+", " ", item).strip()
        for item in re.split(r"\n\s*\n+", str(text or "").strip())
        if item.strip()
    ]
    return paragraphs or ([re.sub(r"\s+", " ", str(text or "")).strip()] if str(text or "").strip() else [])


def normalize_term(token: str) -> str:
    value = re.sub(r"[^a-z0-9_-]", "", str(token or "").lower()).strip("_-")
    if len(value) <= 3 or value in STOPWORDS:
        return ""
    if value.endswith("ing") and len(value) > 6:
        value = value[:-3]
    elif value.endswith(("ed", "es")) and len(value) > 5:
        value = value[:-2]
    elif value.endswith("s") and len(value) > 5:
        value = value[:-1]
    return "" if len(value) <= 3 or value in STOPWORDS else value


def content_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9'_-]{2,}", str(text or "")):
        normalized = normalize_term(token)
        if normalized:
            terms.add(normalized)
    return terms


def protected_anchor_terms(text: str) -> set[str]:
    anchors: set[str] = set()
    for match in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", str(text or "")):
        anchors.add(match.lower())
    for match in re.findall(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,4}\b", str(text or "")):
        normalized = re.sub(r"\s+", " ", match).strip().lower()
        if normalized:
            anchors.add(normalized)
    for match in re.findall(r"\b[A-Z]{2,}[A-Z0-9-]*\b", str(text or "")):
        anchors.add(match.lower())
    return anchors


def formula_snapshot(text: str, report_dict: dict | None, deps: Any) -> dict:
    profile = deps.turnitin_profile(report_dict) if callable(getattr(deps, "turnitin_profile", None)) else {}
    strict = deps.strict_safe_status(report_dict) if callable(getattr(deps, "strict_safe_status", None)) else {}
    components = profile.get("components") if isinstance(profile.get("components"), dict) else {}
    weighted = profile.get("weighted_components") if isinstance(profile.get("weighted_components"), dict) else {}
    suppression = pct(profile.get("human_anchor_suppression"))
    positive_burden = round(sum(float(value or 0.0) for value in weighted.values()), 3)
    score = pct(profile.get("score"), positive_burden - suppression)
    remaining_gap = round(max(0.0, score - pct(profile.get("target_score"), 20.0)), 3)
    priorities = [
        {
            "driver": key,
            "value": pct(components.get(key)),
            "weighted_contribution": round(float(weighted.get(key, 0.0) or 0.0), 3),
        }
        for key in components
        if key in weighted and float(weighted.get(key, 0.0) or 0.0) > 0.0
    ]
    priorities.sort(key=lambda row: float(row.get("weighted_contribution") or 0.0), reverse=True)
    return {
        "score": score,
        "target_score": pct(profile.get("target_score"), 20.0),
        "target_met": bool(profile.get("target_met")),
        "remaining_gap": remaining_gap,
        "components": {key: pct(value) for key, value in components.items()},
        "weighted_components": {key: round(float(value or 0.0), 3) for key, value in weighted.items()},
        "positive_ai_burden": positive_burden,
        "human_anchor_suppression": suppression,
        "strict_safe_band": strict,
        "driver_priority": priorities,
        "word_count": len(re.findall(r"\b[\w'-]+\b", str(text or ""))),
        "sentence_count": len(split_sentences(text)),
        "paragraph_count": len(logical_paragraphs(text)),
    }


def component_drop(before: dict | None, after: dict | None, key: str) -> float:
    before_components = (before or {}).get("components") if isinstance((before or {}).get("components"), dict) else {}
    after_components = (after or {}).get("components") if isinstance((after or {}).get("components"), dict) else {}
    return round(pct(before_components.get(key)) - pct(after_components.get(key)), 3)
