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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def features_from_trace(trace: dict[str, Any]) -> CandidateFeatures:
    validation = trace.get("validation") if isinstance(trace.get("validation"), dict) else {}
    compression = trace.get("compression") if isinstance(trace.get("compression"), dict) else {}
    proxy = trace.get("external_proxy") if isinstance(trace.get("external_proxy"), dict) else {}
    metrics = proxy.get("metrics") if isinstance(proxy.get("metrics"), dict) else {}
    reasons = proxy.get("reasons") if isinstance(proxy.get("reasons"), list) else []
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
    )
