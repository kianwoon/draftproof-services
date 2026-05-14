"""Authorship-window gate and target selection for rewrite V3."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AuthorshipWindowGateThresholds:
    max_ai_fraction: float = 0.20
    max_ai_or_assisted_fraction: float = 0.45
    min_human_fraction: float = 0.50
    max_ai_window_words: int = 180

    @classmethod
    def from_env(cls) -> "AuthorshipWindowGateThresholds":
        return cls(
            max_ai_fraction=_float_env("DRAFTPROOF_REWRITE_V3_MAX_SEGMENT_AI_FRACTION", 0.20),
            max_ai_or_assisted_fraction=_float_env("DRAFTPROOF_REWRITE_V3_MAX_SEGMENT_AI_ASSISTED_FRACTION", 0.45),
            min_human_fraction=_float_env("DRAFTPROOF_REWRITE_V3_MIN_SEGMENT_HUMAN_FRACTION", 0.50),
            max_ai_window_words=int(_float_env("DRAFTPROOF_REWRITE_V3_MAX_AI_WINDOW_WORDS", 180.0)),
        )


@dataclass(frozen=True)
class AuthorshipWindowGate:
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _confidence_rank(value: Any) -> int:
    confidence = str(value or "").lower()
    if confidence == "high":
        return 3
    if confidence == "medium":
        return 2
    if confidence == "low":
        return 1
    return 0


def _label_rank(value: Any) -> int:
    label = str(value or "")
    if label == "ai_generated":
        return 4
    if label == "moderately_ai_assisted":
        return 3
    if label == "lightly_ai_assisted":
        return 2
    return 0


def evaluate_authorship_window_gate(
    profile: dict[str, Any] | None,
    thresholds: AuthorshipWindowGateThresholds | None = None,
) -> AuthorshipWindowGate:
    thresholds = thresholds or AuthorshipWindowGateThresholds.from_env()
    payload = profile if isinstance(profile, dict) else {}
    fraction_ai = _number(payload.get("fraction_ai"))
    fraction_ai_assisted = _number(payload.get("fraction_ai_assisted"))
    fraction_human = _number(payload.get("fraction_human"))
    max_ai_window_words = _number(payload.get("max_ai_window_words"))
    windows = payload.get("windows") if isinstance(payload.get("windows"), list) else []
    high_confidence_ai_windows = sum(
        1
        for window in windows
        if isinstance(window, dict)
        and window.get("label") == "ai_generated"
        and str(window.get("confidence") or "").lower() == "high"
    )

    reasons: list[str] = []
    if fraction_ai > thresholds.max_ai_fraction:
        reasons.append("segment_ai_fraction_high")
    if fraction_ai + fraction_ai_assisted > thresholds.max_ai_or_assisted_fraction:
        reasons.append("segment_ai_or_assisted_fraction_high")
    if fraction_human < thresholds.min_human_fraction:
        reasons.append("segment_human_fraction_low")
    if max_ai_window_words > thresholds.max_ai_window_words:
        reasons.append("segment_ai_window_too_large")
    if high_confidence_ai_windows > 0:
        reasons.append("high_confidence_ai_window_remaining")

    return AuthorshipWindowGate(
        passed=not reasons,
        reasons=tuple(reasons),
        metrics={
            "fraction_ai": fraction_ai,
            "fraction_ai_assisted": fraction_ai_assisted,
            "fraction_human": fraction_human,
            "max_ai_window_words": max_ai_window_words,
            "high_confidence_ai_windows": high_confidence_ai_windows,
            "thresholds": asdict(thresholds),
        },
    )


def select_authorship_window_targets(
    profile: dict[str, Any] | None,
    *,
    max_targets: int = 2,
) -> list[dict[str, Any]]:
    payload = profile if isinstance(profile, dict) else {}
    windows = payload.get("windows") if isinstance(payload.get("windows"), list) else []
    candidates = [
        window
        for window in windows
        if isinstance(window, dict) and _label_rank(window.get("label")) > 0
    ]
    candidates.sort(
        key=lambda window: (
            _label_rank(window.get("label")),
            _confidence_rank(window.get("confidence")),
            _number(window.get("ai_assistance_score")),
            _number(window.get("word_count")),
        ),
        reverse=True,
    )
    return [dict(window) for window in candidates[: max(0, int(max_targets or 0))]]
