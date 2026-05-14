"""Segment-level authorship window profiling for downstream mitigation.

The classifier in this module is intentionally built from typed scan signals
and numeric sentence statistics. It does not inspect content keywords.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .rewrite_targets import span_integrity


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

RISKY_WINDOW_LABELS = {"ai_generated", "moderately_ai_assisted"}
ASSISTED_WINDOW_LABELS = {"ai_generated", "moderately_ai_assisted", "lightly_ai_assisted"}


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


def _stable_excerpt(text: str, limit: int = 420) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[:limit].rstrip()


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


def build_ai_footprint_profile(
    authorship_window_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """Aggregate window-level authorship signals for rewrite selection.

    This profile is intentionally derived from structured window scores. It is
    not a classifier and it does not inspect content keywords.
    """

    profile = authorship_window_profile if isinstance(authorship_window_profile, dict) else {}
    windows = profile.get("windows") if isinstance(profile.get("windows"), list) else []
    normalized_windows = [window for window in windows if isinstance(window, dict)]
    total_words = int(_number(profile.get("word_count"), 0))
    if total_words <= 0:
        total_words = sum(int(_number(window.get("word_count"), 0)) for window in normalized_windows)
    denominator = max(1, total_words)

    risky_windows = [
        window
        for window in normalized_windows
        if str(window.get("label") or "") in RISKY_WINDOW_LABELS
    ]
    assisted_windows = [
        window
        for window in normalized_windows
        if str(window.get("label") or "") in ASSISTED_WINDOW_LABELS
    ]
    risky_words = sum(int(_number(window.get("word_count"), 0)) for window in risky_windows)
    assisted_words = sum(int(_number(window.get("word_count"), 0)) for window in assisted_windows)
    human_words = max(0, denominator - assisted_words)
    high_confidence_risky = [
        window
        for window in risky_windows
        if str(window.get("confidence") or "").lower() == "high"
    ]

    def risk_rank(window: dict[str, Any]) -> tuple[float, float, float]:
        label_weight = 1.0 if str(window.get("label") or "") == "ai_generated" else 0.7
        return (
            _number(window.get("ai_assistance_score")) * label_weight,
            _number(window.get("word_count")),
            _number(window.get("end_index")) - _number(window.get("start_index")),
        )

    top_risky = sorted(risky_windows, key=risk_rank, reverse=True)[:8]
    confidence = "low"
    if total_words >= 300 and len(normalized_windows) >= 3:
        confidence = "medium"
    if total_words >= 700 and len(normalized_windows) >= 6:
        confidence = "high"
    false_positive_guard = {
        "enabled": True,
        "basis": "human_anchor_and_low_high_confidence_risk",
        "human_fraction": round(human_words / denominator, 4),
        "high_confidence_risky_window_count": len(high_confidence_risky),
        "caution": bool((human_words / denominator) >= 0.45 and not high_confidence_risky),
    }
    return {
        "schema_version": "ai_footprint_profile.v2",
        "basis": "structured_authorship_windows",
        "word_count": total_words,
        "window_count": len(normalized_windows),
        "fraction_ai": round(_number(profile.get("fraction_ai")), 4),
        "fraction_ai_assisted": round(_number(profile.get("fraction_ai_assisted")), 4),
        "fraction_human": round(_number(profile.get("fraction_human"), human_words / denominator), 4),
        "risky_window_density": round(risky_words / denominator, 4),
        "assisted_window_density": round(assisted_words / denominator, 4),
        "max_risky_window_words": max([int(_number(window.get("word_count"), 0)) for window in risky_windows] or [0]),
        "max_ai_window_words": int(_number(profile.get("max_ai_window_words"), 0)),
        "max_ai_assisted_window_words": int(_number(profile.get("max_ai_assisted_window_words"), 0)),
        "high_confidence_risky_window_count": len(high_confidence_risky),
        "risky_window_count": len(risky_windows),
        "assisted_window_count": len(assisted_windows),
        "top_risky_windows": [
            {
                "window_id": window.get("window_id"),
                "paragraph_id": window.get("paragraph_id"),
                "sentence_ids": list(window.get("sentence_ids") or []),
                "label": window.get("label"),
                "ai_assistance_score": window.get("ai_assistance_score"),
                "confidence": window.get("confidence"),
                "start_index": window.get("start_index"),
                "end_index": window.get("end_index"),
                "span_integrity": window.get("span_integrity") or {},
                "source_text": window.get("source_text") or "",
                "source_excerpt": window.get("source_excerpt") or _stable_excerpt(str(window.get("source_text") or "")),
                "word_count": window.get("word_count"),
                "score_components": window.get("score_components") or {},
            }
            for window in top_risky
        ],
        "false_positive_guard": false_positive_guard,
        "confidence": confidence,
    }


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
        integrity = span_integrity(source_text, paragraph.get("start_char"), paragraph.get("end_char"))
        paragraph_text = _paragraph_text(source_text, paragraph) if integrity.get("passed") else ""
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
            "span_integrity": integrity,
            "source_text": paragraph_text,
            "source_excerpt": _stable_excerpt(paragraph_text),
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
    authorship_profile = {
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
    authorship_profile["ai_footprint_profile"] = build_ai_footprint_profile(authorship_profile)
    return authorship_profile
