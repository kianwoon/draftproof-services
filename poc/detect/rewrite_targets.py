"""Rewrite target profiles derived from structured scan evidence.

This module converts authorship windows and footprint aggregates into a
downstream rewrite contract. It deliberately avoids content keyword matching:
targets are selected from scanner labels, numeric scores, spans, and preservation
inventory supplied by the report layer.
"""

from __future__ import annotations

from typing import Any


RISKY_WINDOW_LABELS = {"ai_generated", "moderately_ai_assisted"}
ASSISTED_WINDOW_LABELS = {"ai_generated", "moderately_ai_assisted", "lightly_ai_assisted"}


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _word_count(text: str) -> int:
    return len(str(text or "").split())


def span_integrity(source_text: str, start: Any, end: Any) -> dict[str, Any]:
    text = str(source_text or "")
    start_i = int(_number(start, -1))
    end_i = int(_number(end, -1))
    in_bounds = 0 <= start_i < end_i <= len(text)
    starts_on_boundary = False
    ends_on_boundary = False
    if in_bounds:
        starts_on_boundary = start_i == 0 or text[start_i - 1].isspace()
        ends_on_boundary = end_i == len(text) or text[end_i - 1].isspace() or text[end_i].isspace()
    return {
        "start_index": start_i,
        "end_index": end_i,
        "in_bounds": in_bounds,
        "starts_on_boundary": starts_on_boundary,
        "ends_on_boundary": ends_on_boundary,
        "passed": bool(in_bounds and starts_on_boundary and ends_on_boundary),
    }


def _window_text(source_text: str, window: dict[str, Any]) -> str:
    embedded = str(window.get("source_text") or "").strip()
    if embedded:
        return embedded
    text = str(source_text or "")
    start = int(_number(window.get("start_index"), -1))
    end = int(_number(window.get("end_index"), -1))
    if 0 <= start < end <= len(text):
        return text[start:end].strip()
    return ""


def _driver_rows(window: dict[str, Any]) -> list[dict[str, Any]]:
    components = window.get("score_components") if isinstance(window.get("score_components"), dict) else {}
    drivers: list[dict[str, Any]] = []
    for key, value in components.items():
        numeric = _number(value)
        if numeric <= 0:
            continue
        drivers.append({
            "key": str(key),
            "score": round(numeric, 3),
            "source": "score_components",
        })
    for key in ((window.get("matched_signal_keys") or {}).get("ai") or []):
        if not any(row.get("key") == key for row in drivers):
            drivers.append({
                "key": str(key),
                "score": _number(window.get("ai_assistance_score")),
                "source": "matched_signal_keys.ai",
            })
    drivers.sort(key=lambda row: row.get("score", 0.0), reverse=True)
    return drivers[:5]


def _risk_level(window: dict[str, Any]) -> str:
    label = str(window.get("label") or "")
    score = _number(window.get("ai_assistance_score"))
    if label == "ai_generated" or score >= 0.72:
        return "high"
    if label == "moderately_ai_assisted" or score >= 0.42:
        return "medium"
    if label == "lightly_ai_assisted" or score >= 0.22:
        return "low"
    return "minimal"


def _protected_anchors(source_text: str, preservation_inventory: dict[str, Any] | None, *, limit: int = 16) -> list[dict[str, Any]]:
    text = str(source_text or "")
    inventory = preservation_inventory if isinstance(preservation_inventory, dict) else {}
    anchors = inventory.get("anchors") if isinstance(inventory.get("anchors"), list) else []
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        anchor_text = str(anchor.get("text") or "").strip()
        if not anchor_text or anchor_text in seen or anchor_text not in text:
            continue
        seen.add(anchor_text)
        selected.append({
            "text": anchor_text,
            "kind": anchor.get("kind") or anchor.get("type") or "",
            "priority": anchor.get("priority"),
            "reason": anchor.get("reason"),
        })
        if len(selected) >= limit:
            break
    return selected


def _recommended_operation(
    *,
    label: str,
    document_shape: str,
    protected_anchor_count: int,
    score_components: dict[str, Any],
) -> str:
    unsafe_share = _number(score_components.get("unsafe_word_share"))
    predictability = _number(score_components.get("predictability_score"))
    ai_signal = _number(score_components.get("ai_signal_score"))
    if document_shape in {"broad", "mixed"} and protected_anchor_count > 0:
        return "protected_section_rewrite"
    if document_shape == "broad":
        return "chunk_reconstruction"
    if protected_anchor_count > 0:
        return "citation_preserving_window_repair"
    if label in RISKY_WINDOW_LABELS and max(unsafe_share, predictability, ai_signal) >= 0.45:
        return "grounded_author_reasoning_rewrite"
    return "light_texture_rewrite"


def _document_shape(footprint: dict[str, Any], windows: list[dict[str, Any]]) -> str:
    total = _number(footprint.get("fraction_ai")) + _number(footprint.get("fraction_ai_assisted"))
    risky_density = _number(footprint.get("risky_window_density"))
    risky_count = int(_number(footprint.get("risky_window_count")))
    assisted_count = sum(1 for window in windows if str(window.get("label") or "") in ASSISTED_WINDOW_LABELS)
    if total < 0.25 and risky_count == 0:
        return "low_footprint"
    if total >= 0.55 and risky_density <= 0.28 and risky_count > 0:
        return "mixed"
    if total >= 0.55 or assisted_count >= max(4, len(windows) // 2):
        return "broad"
    if risky_count > 0:
        return "localized"
    return "low_footprint"


def _target_sort_key(window: dict[str, Any]) -> tuple[float, float, float]:
    label = str(window.get("label") or "")
    label_weight = 1.0 if label == "ai_generated" else 0.7 if label == "moderately_ai_assisted" else 0.35
    return (
        _number(window.get("ai_assistance_score")) * label_weight,
        _number((window.get("score_components") or {}).get("unsafe_word_share")),
        _number(window.get("word_count")),
    )


def build_rewrite_target_profile(
    *,
    source_text: str,
    authorship_window_profile: dict[str, Any] | None,
    ai_footprint_profile: dict[str, Any] | None = None,
    preservation_inventory: dict[str, Any] | None = None,
    max_targets: int = 12,
) -> dict[str, Any]:
    profile = authorship_window_profile if isinstance(authorship_window_profile, dict) else {}
    footprint = ai_footprint_profile if isinstance(ai_footprint_profile, dict) else {}
    windows = [window for window in profile.get("windows") or [] if isinstance(window, dict)]
    shape = _document_shape(footprint, windows)
    target_windows = [
        window for window in windows if str(window.get("label") or "") in RISKY_WINDOW_LABELS
    ]
    if shape in {"broad", "mixed"}:
        assisted = [
            window
            for window in windows
            if str(window.get("label") or "") in ASSISTED_WINDOW_LABELS and window not in target_windows
        ]
        target_windows.extend(sorted(assisted, key=_target_sort_key, reverse=True)[:max(0, max_targets - len(target_windows))])
    target_windows = sorted(target_windows, key=_target_sort_key, reverse=True)[:max_targets]

    targets: list[dict[str, Any]] = []
    operation_mix: dict[str, int] = {}
    driver_summary: dict[str, int] = {}
    for index, window in enumerate(target_windows, start=1):
        source = _window_text(source_text, window)
        components = window.get("score_components") if isinstance(window.get("score_components"), dict) else {}
        anchors = _protected_anchors(source, preservation_inventory)
        operation = _recommended_operation(
            label=str(window.get("label") or ""),
            document_shape=shape,
            protected_anchor_count=len(anchors),
            score_components=components,
        )
        operation_mix[operation] = operation_mix.get(operation, 0) + 1
        drivers = _driver_rows(window)
        for driver in drivers:
            key = str(driver.get("key") or "")
            if key:
                driver_summary[key] = driver_summary.get(key, 0) + 1
        target_id = f"rt{index:03d}"
        source_words = int(_number(window.get("word_count"), _word_count(source)))
        targets.append({
            "target_id": target_id,
            "unit_id": window.get("paragraph_id") or window.get("window_id") or target_id,
            "window_id": window.get("window_id"),
            "paragraph_id": window.get("paragraph_id"),
            "sentence_ids": list(window.get("sentence_ids") or []),
            "span": {
                "start_index": window.get("start_index"),
                "end_index": window.get("end_index"),
                "integrity": window.get("span_integrity") or span_integrity(source_text, window.get("start_index"), window.get("end_index")),
            },
            "source_text": source,
            "source_excerpt": source[:420],
            "risk_level": _risk_level(window),
            "dominant_drivers": drivers,
            "required_movement": {
                "ai_assistance_score_drop": 0.12 if str(window.get("label") or "") in RISKY_WINDOW_LABELS else 0.06,
                "unsafe_word_share_drop": 0.25 if _number(components.get("unsafe_word_share")) >= 0.45 else 0.0,
                "human_signal_gain": 0.12,
            },
            "recommended_operation": operation,
            "protected_anchors": anchors,
            "word_count_guide": {
                "source_words": source_words,
                "preferred_words": source_words,
            },
            "rewrite_constraints": {
                "preserve_protected_anchors": bool(anchors),
                "preserve_unit_meaning": True,
                "avoid_compressed_summary": True,
                "avoid_generic_academic_smoothing": True,
            },
        })

    return {
        "schema_version": "rewrite_target_profile.v1",
        "basis": "ai_footprint_profile.v2 + structured_scan_signals",
        "document_shape": shape,
        "target_scope_policy": "avoid_over_rewrite" if shape == "low_footprint" else "target_profile_driven",
        "target_count": len(targets),
        "driver_summary": dict(sorted(driver_summary.items(), key=lambda item: item[1], reverse=True)),
        "operation_mix": operation_mix,
        "targets": targets,
    }
