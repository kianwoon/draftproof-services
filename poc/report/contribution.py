from __future__ import annotations

from typing import Any, Optional


def percent_value(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) <= 1:
        number *= 100
    return max(0.0, min(100.0, number))


def contribution_pair(human: Any, ai: Any) -> tuple[Optional[float], Optional[float]]:
    human_score = percent_value(human)
    ai_score = percent_value(ai)
    if human_score is None and ai_score is None:
        return None, None
    if human_score is None:
        human_score = 100.0 - float(ai_score or 0.0)
    human_score = max(0.0, min(100.0, float(human_score)))
    return human_score, 100.0 - human_score


def contribution_pair_int(human: Any, ai: Any) -> tuple[Optional[int], Optional[int]]:
    human_score, ai_score = contribution_pair(human, ai)
    if human_score is None and ai_score is None:
        return None, None
    return int(round(human_score or 0)), int(round(ai_score or 0))
