"""Compact rewrite scan reports without dropping score-profile signals."""

from __future__ import annotations

from typing import Any


SCAN_BADGE_KEYS = (
    "ai_likelihood_score",
    "writing_quality_score",
    "tier",
    "authorship_rating",
    "authorship_rating_label",
    "transformation_classification",
    "ai_components",
    "writing_components",
    # Retained so the rewritten scan's dual headline can show the Turnitin/external
    # estimate (a rewrite does NOT beat perplexity detectors — users must see this).
    "external_detector_estimate",
)

TRANSFORMATION_KEYS = (
    "classification",
    "contribution",
    "core_signals",
    "primary_signal",
    "summary",
)

DOCUMENT_KEYS = (
    "word_count",
    "sentence_count",
    "paragraph_count",
    "topk_calibration_eligible",
    "calibration_confidence",
)

FINDING_SEVERITIES = ("critical", "high", "medium", "low", "info")


def compact_rewrite_scan_summary(scan: Any) -> dict[str, Any]:
    """Keep UI-facing scan fields while trimming large debug-only arrays."""
    if not isinstance(scan, dict):
        return {}

    badge = _dict(scan.get("ai_risk_badge"))
    intelligence = _dict(scan.get("scan_intelligence"))
    transformation = _dict(intelligence.get("transformation"))
    document = _dict(intelligence.get("document"))

    compact: dict[str, Any] = _copy_present(
        scan,
        (
            "ai_score",
            "writing_score",
            "overall_tier",
            "summary",
            "confidence",
            "created_at",
        ),
    )
    compact["ai_risk_badge"] = _copy_present(badge, SCAN_BADGE_KEYS)
    compact["scan_intelligence"] = {
        "transformation": _compact_transformation(transformation),
        "document": _copy_present(document, DOCUMENT_KEYS),
    }
    compact["integrity_layers"] = _compact_integrity_layers(scan.get("integrity_layers"))
    compact["findings"] = _compact_findings(scan.get("findings"))
    return compact


def _compact_transformation(transformation: dict[str, Any]) -> dict[str, Any]:
    compact = _copy_present(transformation, TRANSFORMATION_KEYS)
    core_signals = transformation.get("core_signals")
    if isinstance(core_signals, list):
        compact["core_signals"] = [_compact_signal(signal) for signal in core_signals if isinstance(signal, dict)]
    return compact


def _compact_signal(signal: dict[str, Any]) -> dict[str, Any]:
    return _copy_present(
        signal,
        (
            "key",
            "name",
            "label",
            "group",
            "category",
            "value",
            "score",
            "percent",
            "description",
            "explanation",
            "direction",
            "lower_is_better",
        ),
    )


def _compact_integrity_layers(value: Any) -> dict[str, Any]:
    layers = _dict(value)
    compact = _copy_present(layers, ("overall_score", "overall_tier", "summary", "layer_count"))
    raw_layers = layers.get("layers")
    if isinstance(raw_layers, dict):
        compact["layers"] = {
            key: _copy_present(_dict(layer), ("score", "tier", "label", "status", "summary"))
            for key, layer in raw_layers.items()
        }
    return compact


def _compact_findings(value: Any) -> dict[str, list[dict[str, Any]]]:
    findings = _dict(value)
    compact: dict[str, list[dict[str, Any]]] = {}
    for severity in FINDING_SEVERITIES:
        rows = findings.get(severity)
        if not isinstance(rows, list):
            continue
        compact[severity] = [_compact_finding(_dict(item)) for item in rows[:20]]
    return compact


def _compact_finding(item: dict[str, Any]) -> dict[str, Any]:
    """Keep the fields the rewrite report's post-rewrite "review & act" section needs.

    The flagged passage (``evidence``), the factual ``detail``, and ``actionability`` are
    retained (truncated) so the UI can show users WHICH part of the rewritten draft still
    carries risk and let them act on it -- not just a bare title. Without this the fields
    are dropped during compaction (which runs on every real report) and never reach the UI.
    """
    compact = _copy_present(
        item,
        (
            "finding_id",
            "id",
            "sentence_id",
            "title",
            "category",
            "signal_category",
            "severity",
            "confidence",
            "actionability",
            "score",
        ),
    )
    for key, limit in (("evidence", 600), ("detail", 400)):
        text = item.get(key)
        if isinstance(text, str) and text.strip():
            compact[key] = text if len(text) <= limit else text[:limit].rstrip() + "..."
    return compact


def _copy_present(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
