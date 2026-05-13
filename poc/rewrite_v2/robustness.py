"""Robustness policy helpers for rewrite V2."""

from __future__ import annotations

from typing import Any

from .diagnostics import (
    DETECTOR_NOT_SAFE,
    FIXABLE_CONTRACT_DRIFT,
    GENERATION_FAILED,
    HARD_ANCHOR_LOSS,
    LOCAL_QUALITY_REJECTED,
    SEMANTIC_LOSS,
    summarize_candidate_diagnostics,
)


CONTENT_MODE_POLICY: dict[str, dict[str, Any]] = {
    "academic_cited_text": {
        "max_generated_candidates": 12,
        "layer_candidate_caps": {
            "academic_all_section_compact_reconstruction": 3,
            "academic_cited_section_density_resolver": 3,
            "targeted_paragraph_reconstruction": 6,
            "unsafe_cluster_rescue": 4,
            "academic_anchor_repair_texture_pass": 1,
        },
        "required_layers": [
            "academic_all_section_compact_reconstruction",
            "academic_cited_section_density_resolver",
            "targeted_paragraph_reconstruction",
        ],
        "repairable_failures": [FIXABLE_CONTRACT_DRIFT],
        "second_layer_failures": [DETECTOR_NOT_SAFE],
        "terminal_failures": [HARD_ANCHOR_LOSS, SEMANTIC_LOSS],
    },
    "broad_explanatory_essay": {
        "max_generated_candidates": 12,
        "layer_candidate_caps": {
            "author_stance_thesis_reframe": 4,
            "author_stance_texture_pass": 4,
            "entity_locked_full_reconstruction": 2,
            "keyword_locked_short_texture": 1,
            "targeted_paragraph_reconstruction": 6,
            "unsafe_cluster_rescue": 4,
        },
        "required_layers": [
            "author_stance_thesis_reframe",
            "targeted_paragraph_reconstruction",
            "unsafe_cluster_rescue",
        ],
        "repairable_failures": [LOCAL_QUALITY_REJECTED],
        "second_layer_failures": [DETECTOR_NOT_SAFE],
        "terminal_failures": [HARD_ANCHOR_LOSS, SEMANTIC_LOSS],
    },
    "generic_expository": {
        "max_generated_candidates": 10,
        "layer_candidate_caps": {
            "author_stance_thesis_reframe": 3,
            "entity_locked_full_reconstruction": 2,
            "keyword_locked_short_texture": 1,
            "targeted_paragraph_reconstruction": 5,
            "unsafe_cluster_rescue": 3,
        },
        "required_layers": ["targeted_paragraph_reconstruction", "unsafe_cluster_rescue"],
        "repairable_failures": [LOCAL_QUALITY_REJECTED],
        "second_layer_failures": [DETECTOR_NOT_SAFE],
        "terminal_failures": [HARD_ANCHOR_LOSS, SEMANTIC_LOSS],
    },
    "technical_content": {
        "max_generated_candidates": 5,
        "layer_candidate_caps": {"targeted_paragraph_reconstruction": 4, "unsafe_cluster_rescue": 1},
        "required_layers": ["targeted_paragraph_reconstruction"],
        "repairable_failures": [],
        "second_layer_failures": [],
        "terminal_failures": [HARD_ANCHOR_LOSS, SEMANTIC_LOSS, DETECTOR_NOT_SAFE],
    },
    "regulated_policy_content": {
        "max_generated_candidates": 4,
        "layer_candidate_caps": {"targeted_paragraph_reconstruction": 3, "unsafe_cluster_rescue": 1},
        "required_layers": ["targeted_paragraph_reconstruction"],
        "repairable_failures": [],
        "second_layer_failures": [],
        "terminal_failures": [HARD_ANCHOR_LOSS, SEMANTIC_LOSS, DETECTOR_NOT_SAFE],
    },
    "structured_list_table": {
        "max_generated_candidates": 3,
        "layer_candidate_caps": {"targeted_paragraph_reconstruction": 3},
        "required_layers": ["targeted_paragraph_reconstruction"],
        "repairable_failures": [],
        "second_layer_failures": [],
        "terminal_failures": [HARD_ANCHOR_LOSS, SEMANTIC_LOSS, DETECTOR_NOT_SAFE],
    },
    "quote_heavy": {
        "max_generated_candidates": 5,
        "layer_candidate_caps": {"targeted_paragraph_reconstruction": 4, "unsafe_cluster_rescue": 1},
        "required_layers": ["targeted_paragraph_reconstruction", "unsafe_cluster_rescue"],
        "repairable_failures": [FIXABLE_CONTRACT_DRIFT],
        "second_layer_failures": [],
        "terminal_failures": [HARD_ANCHOR_LOSS, SEMANTIC_LOSS, DETECTOR_NOT_SAFE],
    },
    "short_text": {
        "max_generated_candidates": 2,
        "layer_candidate_caps": {"targeted_paragraph_reconstruction": 2},
        "required_layers": ["targeted_paragraph_reconstruction"],
        "repairable_failures": [],
        "second_layer_failures": [],
        "terminal_failures": [HARD_ANCHOR_LOSS, SEMANTIC_LOSS, DETECTOR_NOT_SAFE, GENERATION_FAILED],
    },
    "personal_reflection": {
        "max_generated_candidates": 8,
        "layer_candidate_caps": {
            "author_stance_thesis_reframe": 3,
            "author_stance_texture_pass": 3,
            "targeted_paragraph_reconstruction": 4,
            "unsafe_cluster_rescue": 2,
        },
        "required_layers": [
            "author_stance_thesis_reframe",
            "author_stance_texture_pass",
            "targeted_paragraph_reconstruction",
        ],
        "repairable_failures": [LOCAL_QUALITY_REJECTED],
        "second_layer_failures": [DETECTOR_NOT_SAFE],
        "terminal_failures": [HARD_ANCHOR_LOSS, SEMANTIC_LOSS],
    },
    "creative_marketing": {
        "max_generated_candidates": 6,
        "layer_candidate_caps": {"targeted_paragraph_reconstruction": 4, "unsafe_cluster_rescue": 2},
        "required_layers": ["targeted_paragraph_reconstruction", "unsafe_cluster_rescue"],
        "repairable_failures": [LOCAL_QUALITY_REJECTED],
        "second_layer_failures": [DETECTOR_NOT_SAFE],
        "terminal_failures": [HARD_ANCHOR_LOSS, SEMANTIC_LOSS],
    },
}


STRATEGY_LAYER_ALIASES = {
    "scan_entity_locked_full_reconstruction": "entity_locked_full_reconstruction",
    "scan_keyword_locked_short_texture": "keyword_locked_short_texture",
    "scan_author_stance_thesis_reframe": "author_stance_thesis_reframe",
    "scan_author_stance_texture_pass": "author_stance_texture_pass",
    "scan_targeted_driver_mitigation": "targeted_paragraph_reconstruction",
    "scan_targeted_minimal_mitigation": "targeted_paragraph_reconstruction",
    "scan_targeted_composed_full_doc_delta_winners": "targeted_composition",
    "targeted": "targeted_paragraph_reconstruction",
    "full_rewrite": "entity_locked_full_reconstruction",
}


def _content_mode(content_route: Any | None) -> str:
    if content_route is None:
        return "generic_expository"
    if isinstance(content_route, dict):
        return str(content_route.get("content_mode") or "generic_expository")
    return str(getattr(content_route, "content_mode", "") or "generic_expository")


def content_mode_policy(content_route: Any | None) -> dict[str, Any]:
    mode = _content_mode(content_route)
    policy = CONTENT_MODE_POLICY.get(mode) or CONTENT_MODE_POLICY["generic_expository"]
    return {"content_mode": mode, **policy}


def _env_int(name: str, default: int) -> int:
    import os

    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def normalize_strategy_layer(row_or_layer: Any) -> str:
    if isinstance(row_or_layer, dict):
        strategy = str(row_or_layer.get("strategy") or "").strip()
        if strategy in STRATEGY_LAYER_ALIASES:
            return STRATEGY_LAYER_ALIASES[strategy]
        value = str(row_or_layer.get("strategy_kind") or strategy).strip()
    else:
        value = str(row_or_layer or "").strip()
    if not value:
        return "unknown"
    return STRATEGY_LAYER_ALIASES.get(value, value)


def portfolio_limits(content_route: Any | None) -> dict[str, Any]:
    policy = content_mode_policy(content_route)
    default_max = int(policy.get("max_generated_candidates") or 8)
    max_generated = max(1, _env_int("DRAFTPROOF_REWRITE_V2_MAX_GENERATED_CANDIDATES", default_max))
    caps = dict(policy.get("layer_candidate_caps") or {})
    for layer, value in list(caps.items()):
        env_name = f"DRAFTPROOF_REWRITE_V2_LAYER_CAP_{str(layer).upper()}"
        caps[layer] = max(0, _env_int(env_name, int(value or 0)))
    return {
        "max_generated_candidates": max_generated,
        "layer_candidate_caps": caps,
    }


def layer_coverage(rows: list[dict[str, Any]], content_route: Any | None) -> dict[str, Any]:
    policy = content_mode_policy(content_route)
    required = list(policy.get("required_layers") or [])
    counts: dict[str, int] = {}
    for row in rows:
        layer = normalize_strategy_layer(row)
        counts[layer] = counts.get(layer, 0) + 1
    missing = [layer for layer in required if counts.get(layer, 0) <= 0]
    return {
        "required_layers": required,
        "ran_layers": sorted(layer for layer, count in counts.items() if count > 0),
        "layer_candidate_counts": dict(sorted(counts.items())),
        "missing_required_layers": missing,
        "required_layer_coverage_met": not missing,
    }


def recommend_failure_policy(
    rows: list[dict[str, Any]],
    *,
    generated_count: int,
    content_route: Any | None,
) -> dict[str, Any]:
    diagnostics = summarize_candidate_diagnostics(rows, generated_count=generated_count)
    policy = content_mode_policy(content_route)
    coverage = layer_coverage(rows, content_route)
    counts = diagnostics.get("failure_class_counts") or {}
    actions: list[str] = []
    for layer in coverage.get("missing_required_layers") or []:
        actions.append(f"run_missing_layer:{layer}")
    for failure in policy.get("repairable_failures", []):
        if counts.get(failure, 0):
            actions.append(f"repair:{failure}")
    for failure in policy.get("second_layer_failures", []):
        if counts.get(failure, 0):
            actions.append(f"second_layer:{failure}")
    for failure in policy.get("terminal_failures", []):
        if counts.get(failure, 0):
            actions.append(f"terminal:{failure}")
    if counts.get(GENERATION_FAILED, 0):
        actions.append("retry:generation_failed")
    return {
        "policy_version": "rewrite_v2_robustness_policy_v1",
        "content_mode": policy["content_mode"],
        "required_layers": list(policy.get("required_layers") or []),
        "portfolio_limits": portfolio_limits(content_route),
        "layer_coverage": coverage,
        "recommended_actions": actions,
        "diagnostics": diagnostics,
    }
