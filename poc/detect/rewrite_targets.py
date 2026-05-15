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
LOW_ANCHOR_TARGET_OPERATIONS = {"grounded_author_reasoning_rewrite", "light_texture_rewrite"}
HARD_BLOCKING_ANCHOR_KINDS = {"citation", "source_citation", "quote", "direct_quote", "number", "numeric"}


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
    text = str(source_text or "")
    paragraph_id = str(window.get("paragraph_id") or "")
    if paragraph_id.startswith("p") and paragraph_id[1:].isdigit():
        paragraph_index = int(paragraph_id[1:]) - 1
        paragraphs = [part.strip() for part in text.split("\n\n")]
        if 0 <= paragraph_index < len(paragraphs) and paragraphs[paragraph_index]:
            return paragraphs[paragraph_index]
    embedded = str(window.get("source_text") or "").strip()
    if embedded:
        return embedded
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


def _anchor_is_hard_blocking(anchor: dict[str, Any]) -> bool:
    severity = str(anchor.get("severity") or "")
    kind = str(anchor.get("kind") or anchor.get("type") or "")
    return bool(severity.startswith("hard") or kind in HARD_BLOCKING_ANCHOR_KINDS)


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
        hard_blocking = _anchor_is_hard_blocking(anchor)
        selected.append({
            "text": anchor_text,
            "kind": anchor.get("kind") or anchor.get("type") or "",
            "severity": anchor.get("severity"),
            "priority": anchor.get("priority"),
            "reason": anchor.get("reason"),
            "anchor_role": "hard_blocking" if hard_blocking else "soft_guidance",
            "blocking": hard_blocking,
        })
        if len(selected) >= limit:
            break
    return selected


def _blocking_anchors(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [anchor for anchor in anchors if bool(anchor.get("blocking"))]


def _soft_guidance_anchors(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [anchor for anchor in anchors if not bool(anchor.get("blocking"))]


def _target_anchor_pressure(anchors: list[dict[str, Any]], source_words: int) -> float:
    hard_count = sum(1 for anchor in anchors if bool(anchor.get("blocking")))
    soft_count = len(anchors) - hard_count
    return round(min(1.0, hard_count * 0.35 + min(soft_count, 10) * 0.015 + max(0, source_words - 80) * 0.001), 3)


def _semantic_edit_cost(*, source_words: int, target_anchor_pressure: float, risk_level: str) -> float:
    risk_discount = {"high": 0.08, "medium": 0.04, "low": 0.0}.get(str(risk_level), 0.0)
    return round(max(0.0, min(1.0, target_anchor_pressure + max(0, source_words - 45) * 0.006 - risk_discount)), 3)


def _scope_level(window: dict[str, Any], source_words: int, document_shape: str) -> str:
    sentence_count = len(window.get("sentence_ids") or [])
    if document_shape == "broad" and source_words >= 120:
        return "chunk"
    if sentence_count <= 1 and source_words <= 45:
        return "sentence_window"
    if source_words <= 45:
        return "span"
    return "paragraph"


def _operation_candidates(
    *,
    operation: str,
    scope_level: str,
    target_anchor_pressure: float,
    semantic_edit_cost: float,
) -> list[str]:
    candidates: list[str] = []
    if (
        scope_level in {"span", "sentence_window"}
        and target_anchor_pressure <= 0.28
        and semantic_edit_cost <= 0.38
    ):
        candidates.append("unit_preserving_prune_bridge")
    if operation == "citation_preserving_window_repair" or target_anchor_pressure >= 0.45:
        candidates.append("citation_preserving_window_repair")
    if operation == "paragraph_preserving_broad_reconstruction":
        candidates.append("paragraph_preserving_broad_reconstruction")
    if operation == "protected_section_rewrite":
        candidates.append("protected_section_rewrite")
    if operation == "chunk_reconstruction" or scope_level == "chunk":
        candidates.append("chunk_reconstruction")
    if operation in LOW_ANCHOR_TARGET_OPERATIONS:
        candidates.append("authorship_window_repair")
    if scope_level == "paragraph":
        candidates.append("paragraph_surgery")
    if not candidates:
        candidates.append(operation if operation else "authorship_window_repair")
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


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
    if document_shape == "broad":
        return "paragraph_preserving_broad_reconstruction"
    if document_shape == "mixed" and protected_anchor_count > 0:
        return "protected_section_rewrite"
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


def _target_position(target: dict[str, Any]) -> int:
    span = target.get("span") if isinstance(target.get("span"), dict) else {}
    start = span.get("start_index")
    try:
        return int(start)
    except (TypeError, ValueError):
        return 10**12


def _group_problem_shape(
    *,
    scope_level: str,
    allowed_operations: list[str],
    anchor_pressure: float,
    target_count: int,
    document_shape: str,
) -> str:
    if "paragraph_preserving_broad_reconstruction" in allowed_operations:
        return "broad_assisted_footprint"
    if "unit_preserving_prune_bridge" in allowed_operations:
        return "localized_low_anchor_risky_span"
    if "citation_preserving_window_repair" in allowed_operations:
        return "localized_anchored_or_cited_window"
    if scope_level == "paragraph" and target_count > 1:
        return "paragraph_multi_sentence_texture"
    if "protected_section_rewrite" in allowed_operations:
        return "protected_section_risk"
    if "chunk_reconstruction" in allowed_operations or scope_level in {"chunk", "document"} or document_shape == "broad":
        return "broad_assisted_footprint"
    if anchor_pressure >= 0.45:
        return "anchored_local_risk"
    return "localized_targeted_risk"


def _expected_movement(problem_shape: str) -> dict[str, str]:
    if problem_shape == "localized_low_anchor_risky_span":
        return {
            "footprint_risk_drop": "small_to_medium",
            "risky_window_density_drop": "medium",
        }
    if problem_shape in {"broad_assisted_footprint", "protected_section_risk"}:
        return {
            "footprint_risk_drop": "medium_to_large",
            "risky_window_density_drop": "medium",
        }
    return {
        "footprint_risk_drop": "small",
        "risky_window_density_drop": "small_to_medium",
    }


def _has_strong_preservation_anchor(targets: list[dict[str, Any]]) -> bool:
    for target in targets:
        for anchor in target.get("protected_anchors") or []:
            if not isinstance(anchor, dict):
                continue
            if bool(anchor.get("blocking")):
                return True
    return False


def build_problem_inventory(
    *,
    rewrite_target_profile: dict[str, Any] | None,
    ai_footprint_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = rewrite_target_profile if isinstance(rewrite_target_profile, dict) else {}
    footprint = ai_footprint_profile if isinstance(ai_footprint_profile, dict) else {}
    targets = [target for target in profile.get("targets") or [] if isinstance(target, dict)]
    document_shape = str(profile.get("document_shape") or "low_footprint")
    if not targets:
        low_footprint = (
            _number(footprint.get("fraction_ai")) + _number(footprint.get("fraction_ai_assisted")) < 0.25
            and int(_number(footprint.get("risky_window_count"))) == 0
        )
        return {
            "schema_version": "problem_inventory.v1",
            "basis": "rewrite_target_profile + ai_footprint_profile + structured_scan_units",
            "document_shape": document_shape,
            "problem_groups": [
                {
                    "group_id": "pg001",
                    "scope_level": "document",
                    "unit_ids": [],
                    "target_ids": [],
                    "problem_shape": "low_footprint" if low_footprint else "no_actionable_targets",
                    "anchor_pressure": 0.0,
                    "semantic_edit_cost": 0.0,
                    "allowed_operations": ["avoid_aggressive_rewrite"],
                    "blocked_operations": ["chunk_reconstruction", "protected_section_rewrite"],
                    "expected_movement": {"footprint_risk_drop": "none", "risky_window_density_drop": "none"},
                }
            ] if low_footprint else [],
        }

    by_unit: dict[str, list[dict[str, Any]]] = {}
    for target in sorted(targets, key=_target_position):
        unit_id = str(target.get("paragraph_id") or target.get("unit_id") or target.get("window_id") or target.get("target_id") or "")
        by_unit.setdefault(unit_id, []).append(target)

    groups: list[dict[str, Any]] = []
    for group_index, (unit_id, rows) in enumerate(by_unit.items(), start=1):
        scope_order = {"span": 0, "sentence_window": 1, "paragraph": 2, "section": 3, "chunk": 4, "document": 5}
        scope_level = max((str(row.get("scope_level") or "paragraph") for row in rows), key=lambda value: scope_order.get(value, 2))
        if len(rows) > 1 and scope_order.get(scope_level, 0) < 2:
            scope_level = "paragraph"
        anchor_pressure = round(max(_number(row.get("target_anchor_pressure")) for row in rows), 3)
        semantic_edit_cost = round(max(_number(row.get("semantic_edit_cost")) for row in rows), 3)
        allowed: list[str] = []
        for row in rows:
            for operation in row.get("operation_candidates") or []:
                if operation not in allowed:
                    allowed.append(str(operation))
        if document_shape == "broad":
            if _has_strong_preservation_anchor(rows):
                allowed = ["protected_section_rewrite", "chunk_reconstruction"]
            else:
                allowed = ["paragraph_preserving_broad_reconstruction", "chunk_reconstruction"]
            scope_level = "paragraph"
        if len(rows) > 1 and "paragraph_surgery" not in allowed and scope_level == "paragraph":
            allowed.insert(0, "paragraph_surgery")
        shape = _group_problem_shape(
            scope_level=scope_level,
            allowed_operations=allowed,
            anchor_pressure=anchor_pressure,
            target_count=len(rows),
            document_shape=document_shape,
        )
        blocked = []
        if shape == "localized_low_anchor_risky_span":
            blocked = ["chunk_reconstruction"]
        elif shape in {"localized_anchored_or_cited_window", "anchored_local_risk"}:
            blocked = ["unit_preserving_prune_bridge"]
        groups.append({
            "group_id": f"pg{group_index:03d}",
            "scope_level": scope_level,
            "unit_ids": [unit_id] if unit_id else [],
            "target_ids": [str(row.get("target_id") or "") for row in rows if row.get("target_id")],
            "problem_shape": shape,
            "anchor_pressure": anchor_pressure,
            "semantic_edit_cost": semantic_edit_cost,
            "allowed_operations": allowed,
            "blocked_operations": blocked,
            "expected_movement": _expected_movement(shape),
        })

    footprint_total = _number(footprint.get("fraction_ai")) + _number(footprint.get("fraction_ai_assisted"))
    if document_shape == "broad" or (
        footprint_total >= 0.72
        and _number(footprint.get("risky_window_density")) >= 0.18
    ):
        groups.append({
            "group_id": f"pg{len(groups) + 1:03d}",
            "scope_level": "document",
            "unit_ids": [],
            "target_ids": [str(target.get("target_id") or "") for target in targets if target.get("target_id")],
            "problem_shape": "broad_assisted_footprint",
            "anchor_pressure": round(max((_number(target.get("target_anchor_pressure")) for target in targets), default=0.0), 3),
            "semantic_edit_cost": round(max((_number(target.get("semantic_edit_cost")) for target in targets), default=0.0), 3),
            "allowed_operations": ["paragraph_preserving_broad_reconstruction", "chunk_reconstruction"],
            "blocked_operations": ["unit_preserving_prune_bridge"],
            "expected_movement": _expected_movement("broad_assisted_footprint"),
        })
    return {
        "schema_version": "problem_inventory.v1",
        "basis": "rewrite_target_profile + ai_footprint_profile + structured_scan_units",
        "document_shape": document_shape,
        "problem_groups": groups,
    }


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
        hard_anchors = _blocking_anchors(anchors)
        soft_anchors = _soft_guidance_anchors(anchors)
        operation = _recommended_operation(
            label=str(window.get("label") or ""),
            document_shape=shape,
            protected_anchor_count=len(hard_anchors),
            score_components=components,
        )
        operation_mix[operation] = operation_mix.get(operation, 0) + 1
        drivers = _driver_rows(window)
        for driver in drivers:
            key = str(driver.get("key") or "")
            if key:
                driver_summary[key] = driver_summary.get(key, 0) + 1
        target_id = f"rt{index:03d}"
        source_words = _word_count(source) or int(_number(window.get("word_count"), 0))
        risk_level = _risk_level(window)
        target_anchor_pressure = _target_anchor_pressure(anchors, source_words)
        semantic_edit_cost = _semantic_edit_cost(
            source_words=source_words,
            target_anchor_pressure=target_anchor_pressure,
            risk_level=risk_level,
        )
        scope_level = _scope_level(window, source_words, shape)
        operation_candidates = _operation_candidates(
            operation=operation,
            scope_level=scope_level,
            target_anchor_pressure=target_anchor_pressure,
            semantic_edit_cost=semantic_edit_cost,
        )
        targets.append({
            "target_id": target_id,
            "unit_id": window.get("paragraph_id") or window.get("window_id") or target_id,
            "parent_unit_id": window.get("paragraph_id") or window.get("window_id") or target_id,
            "scope_level": scope_level,
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
            "risk_level": risk_level,
            "dominant_drivers": drivers,
            "required_movement": {
                "ai_assistance_score_drop": 0.12 if str(window.get("label") or "") in RISKY_WINDOW_LABELS else 0.06,
                "unsafe_word_share_drop": 0.25 if _number(components.get("unsafe_word_share")) >= 0.45 else 0.0,
                "human_signal_gain": 0.12,
            },
            "recommended_operation": operation,
            "protected_anchors": hard_anchors,
            "soft_guidance_anchors": soft_anchors,
            "target_anchor_pressure": target_anchor_pressure,
            "semantic_edit_cost": semantic_edit_cost,
            "reduction_allowed": "unit_preserving_prune_bridge" in operation_candidates,
            "reconstruction_allowed": (
                "chunk_reconstruction" in operation_candidates
                or "protected_section_rewrite" in operation_candidates
                or "paragraph_preserving_broad_reconstruction" in operation_candidates
            ),
            "operation_candidates": operation_candidates,
            "word_count_guide": {
                "source_words": source_words,
                "preferred_words": source_words,
            },
            "rewrite_constraints": {
                "preserve_protected_anchors": bool(hard_anchors),
                "use_soft_guidance_anchors": bool(soft_anchors),
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
