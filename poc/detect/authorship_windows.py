"""Segment-level authorship window profiling for downstream mitigation.

The classifier in this module is intentionally built from typed scan signals
and numeric sentence statistics. It does not inspect content keywords.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AuthorshipWindowThresholds:
    ai_generated_min: float = 0.72
    moderate_ai_assisted_min: float = 0.42
    light_ai_assisted_min: float = 0.22
    high_confidence_words: int = 120
    medium_confidence_words: int = 60


AI_SIGNAL_WEIGHTS: dict[str, float] = {
    "ai_likelihood": 0.34,
    "topk_pattern_raw": 0.28,
    "topk_calibrated_risk": 0.30,
    "rewrite_smoothness": 0.20,
    "semantic_uniformity": 0.18,
    "discourse_regularity": 0.12,
}

HUMAN_SIGNAL_WEIGHTS: dict[str, float] = {
    "human_anchor_score": 0.26,
    "human_anchor_discount": 0.16,
}


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _score01(value: Any) -> float:
    numeric = _number(value)
    if numeric > 1.0:
        numeric = numeric / 100.0
    return max(0.0, min(1.0, numeric))


def _word_count(text: str) -> int:
    return len(str(text or "").split())


def _signal_score(signals: list[dict[str, Any]], weights: dict[str, float]) -> tuple[float, list[str]]:
    weighted_total = 0.0
    weight_seen = 0.0
    matched: list[str] = []
    for signal in signals or []:
        if not isinstance(signal, dict):
            continue
        key = str(signal.get("key") or "")
        weight = weights.get(key)
        if weight is None:
            continue
        weighted_total += _score01(signal.get("score")) * weight
        weight_seen += weight
        matched.append(key)
    if weight_seen <= 0:
        return 0.0, matched
    return max(0.0, min(1.0, weighted_total / weight_seen)), matched


def _paragraph_text(source_text: str, paragraph: dict[str, Any]) -> str:
    start = int(_number(paragraph.get("start_char"), 0))
    end = int(_number(paragraph.get("end_char"), 0))
    if 0 <= start < end <= len(source_text):
        return source_text[start:end].strip()
    return ""


def _segments_for_paragraph(
    segments: list[dict[str, Any]],
    paragraph_id: str,
) -> list[dict[str, Any]]:
    return [
        segment
        for segment in segments or []
        if isinstance(segment, dict) and str(segment.get("paragraph_id") or "p001") == paragraph_id
    ]


def _predictability_score(paragraph_segments: list[dict[str, Any]]) -> float:
    weighted_total = 0.0
    words_total = 0
    for segment in paragraph_segments:
        text = str(segment.get("text") or "")
        words = max(1, _word_count(text))
        predictability = segment.get("predictability") if isinstance(segment.get("predictability"), dict) else {}
        top10 = _score01(predictability.get("top10_ratio"))
        top50 = _score01(predictability.get("top50_ratio"))
        risk = _score01(predictability.get("score"))
        segment_score = max(risk, top10 * 0.65 + top50 * 0.25)
        weighted_total += segment_score * words
        words_total += words
    if words_total <= 0:
        return 0.0
    return max(0.0, min(1.0, weighted_total / words_total))


def _unsafe_word_share(paragraph_segments: list[dict[str, Any]]) -> float:
    unsafe_words = 0
    total_words = 0
    for segment in paragraph_segments:
        words = max(1, _word_count(str(segment.get("text") or "")))
        total_words += words
        signals = segment.get("signals") if isinstance(segment.get("signals"), list) else []
        signal_score, _ = _signal_score(signals, AI_SIGNAL_WEIGHTS)
        predictability = _predictability_score([segment])
        if max(signal_score, predictability) >= 0.58:
            unsafe_words += words
    if total_words <= 0:
        return 0.0
    return max(0.0, min(1.0, unsafe_words / total_words))


def _label_for_score(score: float, thresholds: AuthorshipWindowThresholds) -> str:
    if score >= thresholds.ai_generated_min:
        return "ai_generated"
    if score >= thresholds.moderate_ai_assisted_min:
        return "moderately_ai_assisted"
    if score >= thresholds.light_ai_assisted_min:
        return "lightly_ai_assisted"
    return "human_written"


def _display_label(label: str) -> str:
    labels = {
        "ai_generated": "Fully AI Generated",
        "moderately_ai_assisted": "Moderately AI Assisted",
        "lightly_ai_assisted": "Lightly AI Assisted",
        "human_written": "Human Written",
    }
    return labels.get(label, label)


def _confidence(
    *,
    word_count: int,
    ai_signal_keys: list[str],
    ai_signal_score: float,
    predictability_score: float,
    unsafe_word_share: float,
    thresholds: AuthorshipWindowThresholds,
) -> str:
    evidence_axes = sum(
        1
        for value in (ai_signal_score, predictability_score, unsafe_word_share)
        if value >= 0.45
    )
    if word_count >= thresholds.high_confidence_words and evidence_axes >= 2 and len(ai_signal_keys) >= 2:
        return "high"
    if word_count >= thresholds.medium_confidence_words and evidence_axes >= 1:
        return "medium"
    return "low"


def build_authorship_window_profile(
    *,
    source_text: str,
    segments: list[dict[str, Any]],
    paragraphs: list[dict[str, Any]],
    thresholds: AuthorshipWindowThresholds | None = None,
) -> dict[str, Any]:
    """Build Pangram-style authorship windows from structured scan output."""

    thresholds = thresholds or AuthorshipWindowThresholds()
    windows: list[dict[str, Any]] = []
    for index, paragraph in enumerate(paragraphs or [], start=1):
        if not isinstance(paragraph, dict):
            continue
        paragraph_id = str(paragraph.get("paragraph_id") or f"p{index:03d}")
        paragraph_segments = _segments_for_paragraph(segments, paragraph_id)
        paragraph_text = _paragraph_text(source_text, paragraph)
        if not paragraph_text:
            paragraph_text = " ".join(str(segment.get("text") or "") for segment in paragraph_segments).strip()
        words = _word_count(paragraph_text)
        if words <= 0:
            continue

        signals = paragraph.get("top_signals") if isinstance(paragraph.get("top_signals"), list) else []
        ai_signal_score, ai_signal_keys = _signal_score(signals, AI_SIGNAL_WEIGHTS)
        human_signal_score, human_signal_keys = _signal_score(signals, HUMAN_SIGNAL_WEIGHTS)
        predictability_score = _predictability_score(paragraph_segments)
        unsafe_share = _unsafe_word_share(paragraph_segments)
        assistance_score = max(
            0.0,
            min(
                1.0,
                ai_signal_score * 0.38
                + predictability_score * 0.34
                + unsafe_share * 0.20
                - human_signal_score * 0.16,
            ),
        )
        label = _label_for_score(assistance_score, thresholds)
        windows.append({
            "window_id": f"w{len(windows) + 1:03d}",
            "paragraph_id": paragraph_id,
            "sentence_ids": list(paragraph.get("sentence_ids") or []),
            "start_index": int(_number(paragraph.get("start_char"), 0)),
            "end_index": int(_number(paragraph.get("end_char"), 0)),
            "word_count": words,
            "token_length": words,
            "label": label,
            "display_label": _display_label(label),
            "ai_assistance_score": round(assistance_score, 3),
            "confidence": _confidence(
                word_count=words,
                ai_signal_keys=ai_signal_keys,
                ai_signal_score=ai_signal_score,
                predictability_score=predictability_score,
                unsafe_word_share=unsafe_share,
                thresholds=thresholds,
            ),
            "score_components": {
                "ai_signal_score": round(ai_signal_score, 3),
                "predictability_score": round(predictability_score, 3),
                "unsafe_word_share": round(unsafe_share, 3),
                "human_signal_score": round(human_signal_score, 3),
            },
            "top_signals": signals[:3],
            "matched_signal_keys": {
                "ai": ai_signal_keys,
                "human": human_signal_keys,
            },
        })

    totals = {
        "ai_generated": 0,
        "moderately_ai_assisted": 0,
        "lightly_ai_assisted": 0,
        "human_written": 0,
    }
    total_words = sum(int(window.get("word_count") or 0) for window in windows)
    for window in windows:
        label = str(window.get("label") or "human_written")
        totals[label] = totals.get(label, 0) + int(window.get("word_count") or 0)

    denominator = max(1, total_words)
    ai_words = totals.get("ai_generated", 0)
    assisted_words = totals.get("moderately_ai_assisted", 0) + totals.get("lightly_ai_assisted", 0)
    return {
        "schema_version": "authorship_windows.v1",
        "basis": "structured_scan_signals",
        "scoring_policy": {
            **asdict(thresholds),
            "content_keyword_matching": False,
            "phrase_markers_are_explanatory_only": True,
        },
        "word_count": total_words,
        "fraction_ai": round(ai_words / denominator, 4),
        "fraction_ai_assisted": round(assisted_words / denominator, 4),
        "fraction_human": round(totals.get("human_written", 0) / denominator, 4),
        "num_ai_segments": sum(1 for window in windows if window.get("label") == "ai_generated"),
        "num_ai_assisted_segments": sum(
            1
            for window in windows
            if window.get("label") in {"moderately_ai_assisted", "lightly_ai_assisted"}
        ),
        "num_human_segments": sum(1 for window in windows if window.get("label") == "human_written"),
        "max_ai_window_words": max(
            [int(window.get("word_count") or 0) for window in windows if window.get("label") == "ai_generated"] or [0]
        ),
        "max_ai_assisted_window_words": max(
            [
                int(window.get("word_count") or 0)
                for window in windows
                if window.get("label") in {"moderately_ai_assisted", "lightly_ai_assisted"}
            ]
            or [0]
        ),
        "word_breakdown": totals,
        "windows": windows,
    }
