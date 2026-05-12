"""Small scoring helpers shared by rewrite pipeline phases."""

from __future__ import annotations


def _metric_decimal(value, default=0.0):
    if not isinstance(value, (int, float)):
        return default
    return value / 100.0 if abs(value) > 1 else value


def _ai_first_gate_status(
    reference_ai,
    candidate_ai,
    text_changed: bool,
    min_drop: float = 5.0,
    target: float = 60.0,
    required_min_ai: float = 50.0,
) -> dict:
    """Evaluate whether a candidate clears the product AI-mitigation gate."""
    delta = (
        reference_ai - candidate_ai
        if isinstance(reference_ai, (int, float)) and isinstance(candidate_ai, (int, float))
        else None
    )
    required = (
        bool(text_changed)
        and isinstance(reference_ai, (int, float))
        and reference_ai >= required_min_ai
    )
    success = (
        bool(text_changed)
        and isinstance(delta, (int, float))
        and (
            delta >= min_drop
            or (
                isinstance(reference_ai, (int, float))
                and reference_ai >= target
                and isinstance(candidate_ai, (int, float))
                and candidate_ai < target
            )
        )
    )
    return {
        "required": required,
        "success": success,
        "delta": delta,
        "reference_ai": reference_ai,
        "candidate_ai": candidate_ai,
        "min_drop": min_drop,
        "target": target,
        "required_min_ai": required_min_ai,
    }
