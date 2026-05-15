"""Feature extraction for V3 candidate portfolio ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


@dataclass(frozen=True)
class CandidateFeatures:
    validation_passed: bool
    compression_accepted: bool
    semantic_safe: bool
    proxy_reason_count: int
    ai_delta: float
    topk_delta: float
    wq_delta: float
    compression_ratio: float
    source_units: int
    candidate_units: int
    fraction_ai: float
    fraction_ai_assisted: float
    fraction_human: float
    max_ai_window_words: float
    footprint_risk_drop: float
    target_risk_drop: float
    target_gate_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def features_from_trace(trace: dict[str, Any]) -> CandidateFeatures:
    validation = trace.get("validation") if isinstance(trace.get("validation"), dict) else {}
    compression = trace.get("compression") if isinstance(trace.get("compression"), dict) else {}
    proxy = trace.get("external_proxy") if isinstance(trace.get("external_proxy"), dict) else {}
    metrics = proxy.get("metrics") if isinstance(proxy.get("metrics"), dict) else {}
    reasons = proxy.get("reasons") if isinstance(proxy.get("reasons"), list) else []
    segment_gate = metrics.get("segment_authorship_gate") if isinstance(metrics.get("segment_authorship_gate"), dict) else {}
    footprint_delta = trace.get("footprint_delta") if isinstance(trace.get("footprint_delta"), dict) else {}
    target_movement = trace.get("target_movement") if isinstance(trace.get("target_movement"), dict) else {}
    return CandidateFeatures(
        validation_passed=bool(validation.get("passed")),
        compression_accepted=bool(trace.get("compression_accepted")),
        semantic_safe=bool(trace.get("semantic_safe")),
        proxy_reason_count=len(reasons),
        ai_delta=_float(metrics.get("ai_delta")),
        topk_delta=_float(metrics.get("topk_delta")),
        wq_delta=_float(metrics.get("wq_delta")),
        compression_ratio=_float(compression.get("ratio"), 1.0),
        source_units=int(validation.get("source_units") or 0),
        candidate_units=int(validation.get("candidate_units") or 0),
        fraction_ai=_float(segment_gate.get("fraction_ai")),
        fraction_ai_assisted=_float(segment_gate.get("fraction_ai_assisted")),
        fraction_human=_float(segment_gate.get("fraction_human")),
        max_ai_window_words=_float(segment_gate.get("max_ai_window_words")),
        footprint_risk_drop=_float(footprint_delta.get("risk_drop")),
        target_risk_drop=_float(target_movement.get("risk_drop")),
        target_gate_passed=bool(trace.get("target_gate_passed")),
    )
