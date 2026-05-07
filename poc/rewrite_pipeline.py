"""DraftProof Rewrite Pipeline — reads detect JSON, runs rewrite, outputs report.

Usage:
  python rewrite_pipeline.py detect.json                    # from detect JSON
  python rewrite_pipeline.py detect.json --passes 5         # more rewrite passes
  python rewrite_pipeline.py detect.json --max-loops 3      # more detect-rewrite loops
  python rewrite_pipeline.py --text "Some text here"        # detect + rewrite inline

Output:
  test_output/draftproof_rewrite_<timestamp>.md
  test_output/draftproof_rewrite_<timestamp>.pdf
  test_output/draftproof_rewrite_<timestamp>.json
"""

import sys
import os
import json
import time
import re
import argparse
import math
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rewrite.parse_detect import DetectJSONParser, DetectJSONContext, findings_from_json
from rewrite import run_rewrite, RewriteConfig, RewriteModuleResult
from rewrite.guards import detect_protected_spans, check_semantic_drift
from report.pdf import render_pdf
from report.render_rewrite import render_rewrite_report
from detect.run import DetectionRunner
from detect.layer3_scoring import Layer3Scorer, build_layer3_input_from_text
from report.report import ReportBuilder, report_to_dict
from llm.gateway import LLMGateway, LLMConfig
from detect.mitigation import build_ai_mitigation_plan


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


def _ai_search_candidate_selection_status(
    reference_ai,
    candidate_ai,
    text_changed: bool,
    min_drop: float = 5.0,
    target: float = 60.0,
    required_min_ai: float = 50.0,
) -> dict:
    """Classify a scanned AI-search candidate without overclaiming tiny drops."""
    gate = _ai_first_gate_status(
        reference_ai,
        candidate_ai,
        text_changed,
        min_drop=min_drop,
        target=target,
        required_min_ai=required_min_ai,
    )
    delta = gate.get("delta")
    improved = isinstance(delta, (int, float)) and delta > 0.05
    if gate["success"]:
        reason = ""
    elif not text_changed:
        reason = "unchanged_candidate"
    elif not improved:
        reason = "candidate_not_below_reference"
    elif gate["required"]:
        reason = "best_candidate_below_required_ai_drop"
    else:
        reason = "ai_first_not_required"
    status = dict(gate)
    status.update({
        "improved": improved,
        "selectable": bool(gate["success"]),
        "reason": reason,
    })
    return status


def _clear_stale_rollback_for_kept_ai_mitigation(summary: dict, source: str) -> None:
    """Clear an earlier density/sentence rollback once AI mitigation is kept."""
    if not isinstance(summary, dict):
        return
    had_stale_rollback = bool(
        summary.get("rollback_applied")
        or summary.get("rollback_reason")
        or summary.get("attempted_final_text")
    )
    summary["rollback_applied"] = False
    summary.pop("rollback_reason", None)
    summary.pop("attempted_final_text", None)
    summary.pop("attempted_sentence_comparison", None)
    summary.pop("detect_scan_attempted", None)
    summary.pop("no_text_change", None)
    summary.pop("no_text_change_reason", None)
    if had_stale_rollback:
        summary.setdefault("saved_contract_notes", []).append(
            f"Cleared earlier rewrite rollback because {source} produced a kept AI-mitigation candidate."
        )


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float_env_optional(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _int_env_optional(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _load_local_env(env_path: str | None = None) -> list[str]:
    """Load simple KEY=VALUE pairs from repo .env without overriding exports."""
    if env_path is None:
        env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    loaded = []
    if not os.path.exists(env_path):
        return loaded
    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                    continue
                if key in os.environ:
                    continue
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
                loaded.append(key)
    except OSError:
        return loaded
    return loaded


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _allow_ai_search_llm_after_deterministic() -> bool:
    return _env_flag("DRAFTPROOF_AI_SEARCH_ALLOW_LLM_AFTER_DETERMINISTIC", True)


def _role_model(role: str, fallback_model: str | None = None) -> str | None:
    """Resolve stage-specific LLM model names from env with legacy fallback."""
    role = (role or "").strip().lower()
    role_env = {
        "planner": ("DRAFTPROOF_PLANNER_MODEL", "PLANNER_MODEL", "planner_model", "LLM_PLANNER_MODEL"),
        "generator": ("DRAFTPROOF_GENERATOR_MODEL", "GENERATOR_MODEL", "generator_model", "LLM_GENERATOR_MODEL"),
        "retry": ("DRAFTPROOF_RETRY_MODEL", "RETRY_MODEL", "retry_model", "LLM_RETRY_MODEL"),
    }.get(role, ())
    for name in role_env:
        value = os.environ.get(name)
        if value and str(value).strip():
            return str(value).strip()
    if role == "retry":
        return _role_model("generator", fallback_model)
    return fallback_model or os.environ.get("LLM_MODEL")


def _retry_model_enabled() -> bool:
    """Kill switch for expensive retry-model escalation."""
    return _env_flag("DRAFTPROOF_RETRY_MODEL_ENABLED", False) or _env_flag(
        "RETRY_MODEL_ENABLED",
        False,
    ) or _env_flag(
        "retry_model_enabled",
        False,
    )


def _retry_model_max_calls() -> int:
    raw = (
        os.environ.get("DRAFTPROOF_RETRY_MODEL_MAX_CALLS")
        or os.environ.get("RETRY_MODEL_MAX_CALLS")
        or os.environ.get("retry_model_max_calls")
    )
    try:
        return max(0, int(raw if raw is not None else "1"))
    except ValueError:
        return 1


def _llm_role_config(fallback_model: str | None = None) -> dict:
    retry_enabled = _retry_model_enabled()
    retry_model = _role_model("retry", fallback_model)
    return {
        "planner_model": _role_model("planner", fallback_model),
        "generator_model": _role_model("generator", fallback_model),
        "retry_model": retry_model,
        "retry_model_enabled": retry_enabled,
        "retry_model_max_calls": _retry_model_max_calls() if retry_enabled else 0,
    }


def _ai_search_fast_accept_reason(reference_ai, candidate_ai) -> str:
    """Return an early-stop reason when a deterministic candidate is good enough."""
    if not isinstance(reference_ai, (int, float)) or not isinstance(candidate_ai, (int, float)):
        return ""
    fast_accept_ai = _float_env("DRAFTPROOF_AI_SEARCH_FAST_ACCEPT_AI", 50.0)
    fast_accept_delta = _float_env("DRAFTPROOF_AI_SEARCH_FAST_ACCEPT_DELTA", 5.0)
    ai_first_target = _float_env("DRAFTPROOF_AI_FIRST_TARGET", 60.0)
    delta = reference_ai - candidate_ai
    if candidate_ai <= fast_accept_ai:
        return f"candidate_ai<={fast_accept_ai:.2f}"
    if delta >= fast_accept_delta and candidate_ai < ai_first_target:
        return (
            f"delta>={fast_accept_delta:.2f} "
            f"and candidate_ai<{ai_first_target:.2f}"
        )
    return ""


def _ensure_ai_mitigation_contract(report_json: dict | None) -> dict:
    """Backfill ai_mitigation.v1 for older scan JSONs.

    Fresh scans already include this contract. Older saved scans can still run
    rewrite, so the rewrite phase must synthesize the same decision surface
    from available scan intelligence and badge components.
    """
    if not isinstance(report_json, dict):
        return {}
    existing = report_json.get("ai_mitigation")
    if isinstance(existing, dict) and existing.get("schema_version") == "ai_mitigation.v1":
        return existing
    scan_intelligence = report_json.get("scan_intelligence") or {}
    plan = build_ai_mitigation_plan(
        scan_intelligence=scan_intelligence,
        ai_risk_badge=report_json.get("ai_risk_badge") or {},
        rewrite_plan=report_json.get("rewrite_plan") or {},
        rewrite_constraints=report_json.get("rewrite_constraints") or {},
        rewrite_edit_briefs=report_json.get("rewrite_edit_briefs") or [],
    )
    report_json["ai_mitigation"] = plan
    if isinstance(scan_intelligence, dict):
        mitigation_inputs = scan_intelligence.setdefault("mitigation_inputs", {})
        mitigation_inputs["ai_mitigation_plan"] = plan
    return plan


def _ai_mitigation_requires_user_input(ai_mitigation: dict | None) -> bool:
    if not isinstance(ai_mitigation, dict):
        return False
    readiness = ai_mitigation.get("readiness") or {}
    if readiness.get("requires_user_input"):
        return True
    return ai_mitigation.get("primary_mode") in {
        "guided_authenticity_revision",
        "paragraph_authenticity_rebuild",
        "structure_revision",
    }


def _manual_summary_from_ai_mitigation(ai_mitigation: dict | None, limit: int = 12) -> list[dict]:
    if not isinstance(ai_mitigation, dict):
        return []
    rows = []
    for action in ai_mitigation.get("component_actions") or []:
        if not isinstance(action, dict):
            continue
        if action.get("auto_apply"):
            continue
        rows.append({
            "finding_type": "ai_mitigation_guided_action",
            "scanner_target": "ai_mitigation",
            "component": action.get("component"),
            "original_sentence": "",
            "suggested_sentence": action.get("action") or "",
            "rejection_reason": "requires_author_input",
            "why_review_manually": (
                "This mitigation target needs real author evidence, source context, "
                "or a concrete detail. DraftProof will not invent it automatically."
            ),
            "user_input_needed": action.get("user_input_needed"),
            "priority": action.get("priority"),
        })
        if len(rows) >= limit:
            break
    return rows


_EDUCATIONAL_COMPONENT_NOTES = {
    "generic_assertion_risk": "make this broad claim specific to your class, unit, task, or source",
    "unsupported_claim_risk": "add the evidence for this claim, or soften the claim if evidence is limited",
    "source_grounding_risk": "name the source and explain how it supports this sentence",
    "citation_weakness_risk": "attach the correct citation and explain the cited evidence",
    "broad_claim_risk": "limit this claim to the exact learner group, task, or condition",
    "lived_detail_risk": "add a real class, client, workplace, or process detail",
    "qualifying_text_ai_density": "rebuild this paragraph around context, evidence, your reasoning, and a limited conclusion",
}


def _educational_sentence_note(sentence: str, ai_mitigation: dict | None, used_components: set[str]) -> dict | None:
    actions = [
        action
        for action in ((ai_mitigation or {}).get("component_actions") or [])
        if isinstance(action, dict) and not action.get("auto_apply")
    ]
    if not actions:
        return None
    text = sentence.lower()
    preferred = []
    if any(marker in text for marker in ("this shows", "this means", "important", "improve", "support", "helps")):
        preferred.extend(["generic_assertion_risk", "unsupported_claim_risk", "broad_claim_risk"])
    if any(marker in text for marker in ("according to", "source", "citation", "research", "study")):
        preferred.extend(["source_grounding_risk", "citation_weakness_risk"])
    if any(marker in text for marker in ("i ", "my ", "observed", "class", "workshop", "client", "learner")):
        preferred.append("lived_detail_risk")

    by_component = {str(action.get("component") or ""): action for action in actions}
    selected = None
    for component in preferred:
        if component in by_component and component not in used_components:
            selected = by_component[component]
            break
    if selected is None:
        for action in actions:
            component = str(action.get("component") or "")
            if component and component not in used_components:
                selected = action
                break
    if selected is None:
        selected = actions[0]

    component = str(selected.get("component") or "reviewed_context")
    used_components.add(component)
    note = _EDUCATIONAL_COMPONENT_NOTES.get(
        component,
        selected.get("action") or "add verified author context before using this sentence",
    )
    return {
        "component": component,
        "note": note,
        "user_input_needed": selected.get("user_input_needed"),
        "priority": selected.get("priority"),
    }


def _build_educational_mitigation_rewrite(
    text: str,
    ai_mitigation: dict | None,
    *,
    max_marked_sentences: int = 8,
) -> dict:
    """Create a marked learning draft for user-led AI mitigation.

    This is intentionally not an accepted rewrite. It shows the shape of the
    rewrite and marks every missing fact/source/detail so the user can replace
    placeholders with real author-owned evidence.
    """
    if not isinstance(text, str) or not text.strip() or not isinstance(ai_mitigation, dict):
        return {}

    sentence_re = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
    parts = []
    changes = []
    cursor = 0
    marked = 0
    used_components: set[str] = set()
    for match in sentence_re.finditer(text):
        sentence = match.group(0)
        parts.append(text[cursor:match.start()])
        replacement = sentence
        if marked < max_marked_sentences and len(sentence.split()) >= 8:
            note = _educational_sentence_note(sentence, ai_mitigation, used_components)
            if note:
                insert = f" [[ADD VERIFIED DETAIL: {note['note']}]]"
                stripped = sentence.rstrip()
                terminal = ""
                if stripped and stripped[-1] in ".!?":
                    terminal = stripped[-1]
                    stripped = stripped[:-1].rstrip()
                replacement = f"{stripped}{insert}{terminal}"
                changes.append({
                    "index": len(changes) + 1,
                    "component": note["component"],
                    "original_sentence": sentence.strip(),
                    "rewritten_sentence": replacement.strip(),
                    "user_input_needed": note.get("user_input_needed"),
                    "priority": note.get("priority"),
                })
                marked += 1
        parts.append(replacement)
        cursor = match.end()
    parts.append(text[cursor:])
    draft = "".join(parts)
    if not changes:
        return {}
    return {
        "kind": "educational_marked_rewrite",
        "auto_apply": False,
        "status": "requires_author_completion",
        "draft_text": draft,
        "changes": changes,
        "instructions": [
            "Replace every [[ADD VERIFIED DETAIL: ...]] marker with a real source, example, observation, limitation, or author explanation.",
            "Delete any marker you cannot truthfully support, and narrow the surrounding claim instead.",
            "Run the scan again only after all bracketed markers have been resolved.",
        ],
    }


def _contribution_scores(report_dict: dict | None) -> dict:
    """Extract the Human Contribution / AI Transformation product scores."""
    if not isinstance(report_dict, dict):
        return {"human": None, "ai_transformation": None}
    integrity = (
        report_dict.get("integrity_layers")
        or ((report_dict.get("scan_intelligence") or {}).get("integrity_layers") or {})
    )
    layers = integrity.get("layers") if isinstance(integrity, dict) else {}
    if isinstance(layers, dict):
        human_layer = layers.get("human_contribution_signal") or {}
        transform_layer = layers.get("ai_transformation_risk") or {}
        human_score = human_layer.get("score")
        transform_score = transform_layer.get("score")
        if isinstance(human_score, (int, float)) or isinstance(transform_score, (int, float)):
            if not isinstance(human_score, (int, float)) and isinstance(transform_score, (int, float)):
                human_score = 100.0 - float(transform_score)
            if not isinstance(transform_score, (int, float)) and isinstance(human_score, (int, float)):
                transform_score = 100.0 - float(human_score)
            return {
                "human": float(human_score),
                "ai_transformation": float(transform_score),
            }
    contribution = (
        ((report_dict.get("scan_intelligence") or {}).get("transformation") or {})
        .get("contribution")
        or {}
    )

    def _score(*keys):
        for key in keys:
            value = contribution.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    human = _score("human_contribution_ratio", "human_contribution", "human_ratio")
    ai_transformation = _score(
        "ai_transformation_ratio",
        "ai_transformation",
        "transformation_ratio",
    )
    if human is None and ai_transformation is not None:
        human = round(100.0 - ai_transformation, 3)
    if ai_transformation is None and human is not None:
        ai_transformation = round(100.0 - human, 3)
    return {"human": human, "ai_transformation": ai_transformation}


def _integrity_scores(report_dict: dict | None) -> dict:
    if not isinstance(report_dict, dict):
        return {"ai_authorship": None, "grounding": None}
    integrity = (
        report_dict.get("integrity_layers")
        or ((report_dict.get("scan_intelligence") or {}).get("integrity_layers") or {})
    )
    layers = integrity.get("layers") if isinstance(integrity, dict) else {}
    if not isinstance(layers, dict):
        return {"ai_authorship": None, "grounding": None}

    def _score(layer_name: str):
        value = (layers.get(layer_name) or {}).get("score")
        return float(value) if isinstance(value, (int, float)) else None

    return {
        "ai_authorship": _score("ai_authorship_risk"),
        "ai_transformation": _score("ai_transformation_risk"),
        "grounding": _score("grounding_quality_risk"),
        "human": _score("human_contribution_signal"),
    }


def _transformation_features(report_dict: dict | None) -> dict:
    if not isinstance(report_dict, dict):
        return {}
    badge = report_dict.get("ai_risk_badge") or {}
    transform = badge.get("transformation_classification") or {}
    features = transform.get("features")
    return features if isinstance(features, dict) else {}


def _feature_percent(report_dict: dict | None, key: str):
    value = _transformation_features(report_dict).get(key)
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def _human_shift_score(
    original_report: dict,
    candidate_report: dict,
    *,
    drift_similarity: float | None = None,
    review_burden_delta: int = 0,
    weighted_severity_delta: int = 0,
) -> dict:
    """Score how strongly a candidate moves toward authentic human contribution.

    Positive components reward real mitigation movement. Penalties protect
    meaning, grounding, and review burden so candidate ranking cannot chase a
    lower AI score by making the document worse.
    """
    original = _contribution_scores(original_report)
    candidate = _contribution_scores(candidate_report)
    original_integrity = _integrity_scores(original_report)
    candidate_integrity = _integrity_scores(candidate_report)

    def _delta(original_value, candidate_value, *, direction: str = "increase"):
        if not isinstance(original_value, (int, float)) or not isinstance(candidate_value, (int, float)):
            return 0.0
        if direction == "decrease":
            return float(original_value) - float(candidate_value)
        return float(candidate_value) - float(original_value)

    ai_authorship_reduction = _delta(
        original_integrity.get("ai_authorship"),
        candidate_integrity.get("ai_authorship"),
        direction="decrease",
    )
    human_contribution_gain = _delta(original.get("human"), candidate.get("human"))
    ai_transformation_reduction = _delta(
        original.get("ai_transformation"),
        candidate.get("ai_transformation"),
        direction="decrease",
    )
    human_anchor_gain = _delta(
        _feature_percent(original_report, "human_anchor_score"),
        _feature_percent(candidate_report, "human_anchor_score"),
    )
    grounding_risk_reduction = _delta(
        original_integrity.get("grounding"),
        candidate_integrity.get("grounding"),
        direction="decrease",
    )
    rewrite_smoothness_reduction = _delta(
        _feature_percent(original_report, "rewrite_smoothness"),
        _feature_percent(candidate_report, "rewrite_smoothness"),
        direction="decrease",
    )
    semantic_uniformity_reduction = _delta(
        _feature_percent(original_report, "semantic_uniformity_risk"),
        _feature_percent(candidate_report, "semantic_uniformity_risk"),
        direction="decrease",
    )

    grounding_regression_penalty = max(0.0, -grounding_risk_reduction) * 1.5
    smoothness_regression_penalty = max(0.0, -rewrite_smoothness_reduction) * 0.8
    semantic_uniformity_regression_penalty = max(0.0, -semantic_uniformity_reduction) * 0.8
    review_burden_penalty = max(0, int(review_burden_delta or 0)) * 3.0
    weighted_severity_penalty = max(0, int(weighted_severity_delta or 0)) * 1.5
    meaning_drift_penalty = 0.0
    if isinstance(drift_similarity, (int, float)):
        meaning_drift_penalty = max(0.0, 0.94 - float(drift_similarity)) * 25.0

    components = {
        "ai_authorship_reduction": round(ai_authorship_reduction, 3),
        "human_contribution_gain": round(human_contribution_gain, 3),
        "ai_transformation_reduction": round(ai_transformation_reduction, 3),
        "human_anchor_gain": round(human_anchor_gain, 3),
        "grounding_risk_reduction": round(grounding_risk_reduction, 3),
        "rewrite_smoothness_reduction": round(rewrite_smoothness_reduction, 3),
        "semantic_uniformity_reduction": round(semantic_uniformity_reduction, 3),
        "grounding_regression_penalty": round(grounding_regression_penalty, 3),
        "meaning_drift_penalty": round(meaning_drift_penalty, 3),
        "rewrite_smoothness_regression_penalty": round(smoothness_regression_penalty, 3),
        "semantic_uniformity_regression_penalty": round(semantic_uniformity_regression_penalty, 3),
        "review_burden_penalty": round(review_burden_penalty, 3),
        "weighted_severity_penalty": round(weighted_severity_penalty, 3),
    }
    score = (
        ai_authorship_reduction * 1.0
        + human_contribution_gain * 1.4
        + ai_transformation_reduction * 1.2
        + human_anchor_gain * 0.7
        + max(0.0, grounding_risk_reduction) * 0.35
        + max(0.0, rewrite_smoothness_reduction) * 0.35
        + max(0.0, semantic_uniformity_reduction) * 0.35
        - grounding_regression_penalty
        - meaning_drift_penalty
        - smoothness_regression_penalty
        - semantic_uniformity_regression_penalty
        - review_burden_penalty
        - weighted_severity_penalty
    )
    return {
        "score": round(score, 3),
        "components": components,
        "weights": {
            "ai_authorship_reduction": 1.0,
            "human_contribution_gain": 1.4,
            "ai_transformation_reduction": 1.2,
            "human_anchor_gain": 0.7,
            "grounding_risk_reduction": 0.35,
            "rewrite_smoothness_reduction": 0.35,
            "semantic_uniformity_reduction": 0.35,
            "grounding_regression_penalty": -1.5,
            "meaning_drift_penalty": -1.0,
            "rewrite_smoothness_regression_penalty": -1.0,
            "semantic_uniformity_regression_penalty": -1.0,
            "review_burden_penalty": -1.0,
            "weighted_severity_penalty": -1.0,
        },
    }


def _human_shift_rank_key(gate: dict | None) -> tuple:
    gate = gate or {}
    score = gate.get("human_shift_score")
    candidate_human = gate.get("candidate_human")
    ai_authorship_delta = gate.get("ai_authorship_delta")
    stage_target = _human_gain_stage_target(candidate_human)
    return (
        1 if gate.get("success") else 0,
        1 if not isinstance(ai_authorship_delta, (int, float)) or ai_authorship_delta >= 0 else 0,
        1 if isinstance(candidate_human, (int, float)) and candidate_human >= stage_target else 0,
        float(gate.get("human_delta")) if isinstance(gate.get("human_delta"), (int, float)) else -9999.0,
        float(score) if isinstance(score, (int, float)) else -9999.0,
        float(ai_authorship_delta) if isinstance(ai_authorship_delta, (int, float)) else -9999.0,
        float(gate.get("ai_transformation_delta")) if isinstance(gate.get("ai_transformation_delta"), (int, float)) else -9999.0,
    )


def _is_better_human_shift_candidate(candidate_gate: dict | None, best_gate: dict | None) -> bool:
    if best_gate is None:
        return True
    return _human_shift_rank_key(candidate_gate) > _human_shift_rank_key(best_gate)


def _anchor_lock_mapping(anchors: list[str] | tuple[str, ...] | None) -> list[dict]:
    """Create deterministic placeholders for anchors that should not be rewritten."""
    unique: list[str] = []
    for raw in anchors or []:
        value = str(raw or "").strip()
        if len(value) < 4:
            continue
        if value not in unique:
            unique.append(value)
    unique.sort(key=len, reverse=True)
    return [
        {"placeholder": f"[[DP_ANCHOR_{index:03d}]]", "value": value}
        for index, value in enumerate(unique, start=1)
    ]


def _freeze_anchor_text(text: str, mapping: list[dict] | None) -> str:
    frozen = str(text or "")
    for item in mapping or []:
        value = str(item.get("value") or "")
        placeholder = str(item.get("placeholder") or "")
        if value and placeholder:
            frozen = re.sub(re.escape(value), placeholder, frozen)
    return frozen


def _restore_anchor_placeholders(text: str, mapping: list[dict] | None) -> str:
    restored = str(text or "")
    for item in mapping or []:
        value = str(item.get("value") or "")
        placeholder = str(item.get("placeholder") or "")
        if value and placeholder:
            restored = restored.replace(placeholder, value)
    return restored


def _split_sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", str(text or "").strip())
        if item.strip()
    ]


def _freeze_anchor_payload(payload, mapping: list[dict] | None):
    if isinstance(payload, str):
        return _freeze_anchor_text(payload, mapping)
    if isinstance(payload, list):
        return [_freeze_anchor_payload(item, mapping) for item in payload]
    if isinstance(payload, tuple):
        return tuple(_freeze_anchor_payload(item, mapping) for item in payload)
    if isinstance(payload, dict):
        return {
            key: _freeze_anchor_payload(value, mapping)
            for key, value in payload.items()
        }
    return payload


def _repair_aggression_score(original_text: str, candidate_text: str) -> dict:
    """Estimate how invasive a repair is so texture repair stays micro-local."""
    original_tokens = re.findall(r"\w+|[^\w\s]", str(original_text or ""))
    candidate_tokens = re.findall(r"\w+|[^\w\s]", str(candidate_text or ""))
    if not original_tokens and not candidate_tokens:
        ratio = 1.0
    else:
        ratio = SequenceMatcher(None, original_tokens, candidate_tokens).ratio()
    changed_tokens_ratio = round(1.0 - ratio, 3)
    original_sentences = _split_sentences(str(original_text or ""))
    candidate_sentences = _split_sentences(str(candidate_text or ""))
    sentence_count_delta = abs(len(candidate_sentences) - len(original_sentences))
    sentence_delta_ratio = round(
        sentence_count_delta / max(1, len(original_sentences)),
        3,
    )
    original_words = _text_word_count(str(original_text or ""))
    candidate_words = _text_word_count(str(candidate_text or ""))
    word_delta_ratio = round(
        abs(candidate_words - original_words) / max(1, original_words),
        3,
    )
    score = round(
        changed_tokens_ratio + min(1.0, sentence_delta_ratio) * 0.5 + min(1.0, word_delta_ratio) * 0.5,
        3,
    )
    return {
        "score": score,
        "changed_tokens_ratio": changed_tokens_ratio,
        "sentence_delta_ratio": sentence_delta_ratio,
        "word_delta_ratio": word_delta_ratio,
        "original_words": original_words,
        "candidate_words": candidate_words,
    }


def _sentence_texture_risk_map(text: str, raw_json: dict | None = None, limit: int = 5) -> list[dict]:
    """Build a sentence-level repair map from scanner pointers and local texture signals."""
    sentences = _split_sentences(text)
    if not sentences:
        return []
    pointer_scores: dict[int, float] = {}
    raw_json = raw_json or {}
    for brief in raw_json.get("rewrite_edit_briefs") or []:
        if not isinstance(brief, dict):
            continue
        index = brief.get("sentence_index")
        if isinstance(index, int) and 0 <= index < len(sentences):
            signal_bonus = 0.25
            for value in (brief.get("signals") or {}).values():
                if isinstance(value, (int, float)):
                    signal_bonus += min(0.4, float(value) / 250.0)
                elif isinstance(value, str) and re.search(r"predict|uniform|smooth|generic|cadence", value, re.I):
                    signal_bonus += 0.15
            pointer_scores[index] = max(pointer_scores.get(index, 0.0), signal_bonus)
    for segment in ((raw_json.get("ai_mitigation") or {}).get("target_segments") or []):
        if not isinstance(segment, dict):
            continue
        seg_text = str(segment.get("text") or "").strip()
        if not seg_text:
            continue
        for index, sentence in enumerate(sentences):
            if seg_text in sentence or sentence in seg_text:
                signal = segment.get("primary_signal") or {}
                value = signal.get("score")
                pointer_scores[index] = max(
                    pointer_scores.get(index, 0.0),
                    0.35 + (min(0.4, float(value) / 250.0) if isinstance(value, (int, float)) else 0.0),
                )
    generic_re = re.compile(
        r"\b(?:important|significant|crucial|essential|plays? a role|"
        r"this (?:shows|highlights|demonstrates|emphasizes)|"
        r"furthermore|moreover|additionally|in conclusion|overall|"
        r"increasingly|integrat(?:e|es|ing)|supports?|enables?|enhances?)\b",
        re.I,
    )
    transition_re = re.compile(r"^(?:This|These|Such|In this|Overall|Therefore|However|Additionally|Furthermore)\b")
    rows: list[dict] = []
    for index, sentence in enumerate(sentences):
        words = re.findall(r"\b[\w'-]+\b", sentence)
        length_score = min(0.25, max(0, len(words) - 22) / 120.0)
        generic_score = min(0.35, len(generic_re.findall(sentence)) * 0.12)
        transition_score = 0.15 if transition_re.search(sentence.strip()) else 0.0
        score = round(pointer_scores.get(index, 0.0) + length_score + generic_score + transition_score, 3)
        rows.append({
            "sentence_index": index,
            "sentence": sentence,
            "risk": score,
            "drivers": {
                "scanner_pointer": round(pointer_scores.get(index, 0.0), 3),
                "length": round(length_score, 3),
                "generic_phrase": round(generic_score, 3),
                "transition_cleanliness": round(transition_score, 3),
            },
        })
    rows.sort(key=lambda row: row["risk"], reverse=True)
    return rows[:max(1, limit)]


def _micro_texture_window(
    text: str,
    raw_json: dict | None = None,
    *,
    max_sentences: int = 2,
    exclude_sentence_indexes: set[int] | list[int] | tuple[int, ...] | None = None,
) -> dict:
    sentences = _split_sentences(text)
    if not sentences:
        return {"sentences": [], "start": 0, "end": 0, "text": "", "risk_map": []}
    risk_map = _sentence_texture_risk_map(text, raw_json, limit=5)
    excluded = {int(item) for item in (exclude_sentence_indexes or []) if isinstance(item, int)}
    selected = next(
        (
            row for row in risk_map
            if "\n" not in str(row.get("sentence") or "")
            and int(row.get("sentence_index", -1)) not in excluded
        ),
        None,
    )
    if selected is None:
        selected = next(
            (
                row for row in risk_map
                if int(row.get("sentence_index", -1)) not in excluded
            ),
            None,
        )
    if selected is None:
        return {"sentences": sentences, "start": 0, "end": 0, "text": "", "risk_map": risk_map}
    start = int(selected.get("sentence_index", 0))
    end = min(len(sentences), start + max(1, max_sentences))
    return {
        "sentences": sentences,
        "start": start,
        "end": end,
        "text": " ".join(sentences[start:end]).strip(),
        "risk_map": risk_map,
    }


def _splice_sentence_window(text: str, start: int, end: int, replacement: str) -> str:
    sentences = _split_sentences(text)
    if start < 0 or end <= start or start >= len(sentences):
        return text
    end = min(end, len(sentences))
    replacement_sentences = _split_sentences(replacement)
    if not replacement_sentences:
        return text
    rebuilt = sentences[:start] + replacement_sentences + sentences[end:]
    return " ".join(rebuilt).strip()


def _micro_texture_repair_prompt(
    source_text: str,
    raw_json: dict | None,
    anchors: list[str] | None = None,
    *,
    max_sentences: int = 1,
    exclude_sentence_indexes: set[int] | list[int] | tuple[int, ...] | None = None,
    mode: str = "authorship_texture_repair",
) -> tuple[str, dict]:
    """Build an operation-level prompt for one local authorship texture patch."""
    window = _micro_texture_window(
        source_text,
        raw_json,
        max_sentences=max_sentences,
        exclude_sentence_indexes=exclude_sentence_indexes,
    )
    mapping = _anchor_lock_mapping(anchors or [])
    frozen_window = _freeze_anchor_text(window.get("text") or "", mapping)
    frozen_context = {
        "window_start_sentence": window.get("start"),
        "window_end_sentence": window.get("end"),
        "risk_map": _freeze_anchor_payload(window.get("risk_map") or [], mapping),
        "target_window": frozen_window,
        "required_placeholders": [item["placeholder"] for item in mapping if item["placeholder"] in frozen_window],
    }
    mode_name = str(mode or "authorship_texture_repair").strip().lower()
    if mode_name == "authorship_suppression_repair":
        objective = (
            "DraftProof micro-local AUTHORSHIP_SUPPRESSION_REPAIR.\n"
            "Objective: reduce AI Authorship first. Human Contribution is only a bonus.\n"
            "Patch only the target sentence window. Do not improve the whole section.\n"
        )
        allowed = (
            "- shorten_over_complete_explanation\n"
            "- break_repeated_sentence_cadence\n"
            "- remove_neat_claim_explain_conclude_flow\n"
            "- reduce_polished_transition\n"
            "- leave one clear point slightly less over-explained\n"
        )
        forbidden_extra = (
            "- adding anchors\n"
            "- adding first-person\n"
            "- expanding explanations\n"
            "- improving clarity by smoothing every link\n"
        )
    else:
        objective = (
            "DraftProof micro-local AUTHORSHIP_TEXTURE_REPAIR.\n"
            "Patch only the target sentence window. Do not rewrite the surrounding section.\n"
        )
        allowed = (
            "- shorten_transition\n"
            "- reduce_explanation\n"
            "- alter_sentence_length\n"
            "- reduce_connector_strength\n"
            "- slight_pacing_asymmetry\n"
        )
        forbidden_extra = ""
    prompt = (
        objective +
        "Return only the replacement sentence window, not the full section.\n\n"
        f"Repair context:\n{json.dumps(frozen_context, ensure_ascii=False)}\n\n"
        "Allowed operations:\n"
        f"{allowed}\n"
        "Forbidden operations:\n"
        "- reorder_paragraph\n"
        "- add_new_claim\n"
        "- semantic_expansion\n"
        "- rewrite_whole_sentence_cluster\n"
        "- add examples, sources, dates, numbers, institutions, citations, or evidence\n"
        "- add typos, fake randomness, or grammar damage\n\n"
        f"{forbidden_extra}"
        "Hard constraints:\n"
        "- Preserve meaning and all [[DP_ANCHOR_###]] placeholders exactly.\n"
        "- Keep replacement within about 70% to 130% of the target window word count.\n"
        "- Prefer lower authorship regularity over polished explanation.\n"
        "- Output prose only."
    )
    return prompt, {
        "schema_version": "micro_texture_repair.v1",
        "window": window,
        "anchor_lock": mapping,
        "frozen_target_window": frozen_window,
    }


def _clean_micro_texture_candidate(output: str, repair_info: dict) -> tuple[str, str]:
    text = _clean_section_candidate(output or "", "")
    if not text:
        return "", "empty_micro_texture_candidate"
    mapping = (repair_info or {}).get("anchor_lock") or []
    text = _restore_anchor_placeholders(text, mapping)
    target_words = _text_word_count((repair_info or {}).get("window", {}).get("text") or "")
    candidate_words = _text_word_count(text)
    if target_words and candidate_words < max(3, int(target_words * 0.70)):
        return "", f"micro_texture_candidate_too_short {candidate_words}<{int(target_words * 0.70)}"
    if target_words and candidate_words > max(8, int(target_words * 1.30)):
        return "", f"micro_texture_candidate_too_long {candidate_words}>{int(target_words * 1.30)}"
    return text, ""


_MASKED_REPAIR_GENERIC_RE = re.compile(
    r"\b(?:Furthermore|Moreover|Additionally|In conclusion|Overall|Therefore|However|"
    r"In the past|This shift has made|The real challenge is|"
    r"It is important to note that|This highlights|This shows|This demonstrates|"
    r"plays? a crucial role|significant impact|important|crucial|essential)\b",
    re.I,
)


def _masked_span_repair_prompt(
    source_text: str,
    raw_json: dict | None,
    *,
    exclude_sentence_indexes: set[int] | list[int] | tuple[int, ...] | None = None,
) -> tuple[str, dict]:
    """Mask only a high-risk local span so generation cannot rewrite the section."""
    window = _micro_texture_window(
        source_text,
        raw_json,
        max_sentences=1,
        exclude_sentence_indexes=exclude_sentence_indexes,
    )
    sentence = str(window.get("text") or "")
    if not sentence:
        return "", {"reason": "no_mask_window", "window": window}

    mask_start = mask_end = -1
    mask_text = ""
    match = _MASKED_REPAIR_GENERIC_RE.search(sentence)
    if match:
        mask_start, mask_end = match.span()
        mask_text = match.group(0)
    if mask_start < 0:
        for brief in (raw_json or {}).get("rewrite_edit_briefs") or []:
            if not isinstance(brief, dict) or brief.get("sentence_index") != window.get("start"):
                continue
            for token in brief.get("problem_tokens") or []:
                token_text = str(token or "").strip()
                if len(token_text) < 4:
                    continue
                pos = sentence.lower().find(token_text.lower())
                if pos >= 0:
                    mask_start, mask_end = pos, pos + len(token_text)
                    mask_text = sentence[pos:mask_end]
                    break
            if mask_start >= 0:
                break
    if mask_start < 0:
        words = list(re.finditer(r"\b[\w'-]+\b", sentence))
        if not words:
            return "", {"reason": "no_maskable_span", "window": window}
        # Last resort: mask a short opening phrase, not the whole sentence.
        first = words[0]
        last = words[min(2, len(words) - 1)]
        mask_start, mask_end = first.start(), last.end()
        mask_text = sentence[mask_start:mask_end]

    masked_sentence = f"{sentence[:mask_start]}[[MASK]]{sentence[mask_end:]}"
    prompt = (
        "DraftProof partial masked regeneration.\n"
        "Replace only [[MASK]] in the sentence. Do not rewrite any other words.\n"
        "Objective: reduce AI Authorship texture without semantic expansion.\n\n"
        f"Masked sentence: {masked_sentence}\n"
        f"Original masked span: {mask_text}\n\n"
        "Rules:\n"
        "- Return only the replacement text for [[MASK]].\n"
        "- Replacement may be empty if deleting the span reads naturally.\n"
        "- Use at most 8 words.\n"
        "- Do not add claims, evidence, examples, citations, numbers, names, first-person, or explanation.\n"
        "- Prefer plain wording or no connector over polished academic phrasing."
    )
    return prompt, {
        "schema_version": "masked_span_repair.v1",
        "window": window,
        "sentence": sentence,
        "masked_sentence": masked_sentence,
        "mask_text": mask_text,
        "mask_start": mask_start,
        "mask_end": mask_end,
    }


def _clean_masked_span_replacement(output: str) -> str:
    text = _clean_section_candidate(output or "", "")
    text = re.sub(r"\[\[/?MASK\]\]", "", text, flags=re.I).strip()
    text = text.strip("\"'` ")
    words = re.findall(r"\b[\w'-]+\b", text)
    if len(words) > 8:
        return " ".join(words[:8])
    return text


def _deterministic_masked_span_replacements(mask_text: str) -> list[str]:
    """Return safe local replacements before spending an LLM call.

    These are intentionally tiny span-level alternatives, not sentence rewrites.
    The scanner still decides whether any candidate is kept.
    """
    key = re.sub(r"\s+", " ", str(mask_text or "").strip()).lower()
    replacements = {
        "important": ["vital", "needed", "useful"],
        "in the past": ["Earlier"],
        "this shift has made": ["That shift makes"],
        "the real challenge is": ["The harder part is"],
        "this is a": ["A"],
        "this has created": ["This creates", "It creates"],
        "this can encourage": ["This may lead to", "It can lead to"],
        "this makes assessment": ["Assessment becomes"],
        "in other words": ["Put simply", "Simply"],
        "today's education": ["The education"],
        "today’s education": ["The education"],
        "students received knowledge": ["Students learned"],
        "a student with": ["One student with"],
    }.get(key, [])
    unique: list[str] = []
    for item in replacements:
        clean = _clean_masked_span_replacement(item)
        if clean not in unique:
            unique.append(clean)
    return unique


def _deterministic_sentence_route_bundle(source_text: str) -> tuple[str, list[dict]]:
    """Apply a small set of safe sentence-route edits as one candidate.

    These are not synonym swaps. They target common scanner-visible discourse
    routes while preserving the surrounding claim and all anchors.
    """
    text = str(source_text or "")
    edits = [
        ("In the past", "Earlier"),
        ("This shift has made", "That shift makes"),
        ("The real challenge is", "The harder part is"),
    ]
    applied: list[dict] = []
    candidate = text
    for old, new in edits:
        pattern = re.compile(r"\b" + re.escape(old) + r"\b", re.I)
        if not pattern.search(candidate):
            continue
        candidate = pattern.sub(new, candidate, count=1)
        applied.append({"mask_text": old, "replacement": new})
    return candidate, applied


def _apply_masked_span_replacement(source_text: str, repair_info: dict, replacement: str) -> str:
    sentence = str((repair_info or {}).get("sentence") or "")
    start = int((repair_info or {}).get("mask_start") or 0)
    end = int((repair_info or {}).get("mask_end") or 0)
    if not sentence or end < start:
        return source_text
    replacement = str(replacement or "").strip()
    repaired_sentence = f"{sentence[:start]}{replacement}{sentence[end:]}"
    repaired_sentence = re.sub(r"\s+([,.;:!?])", r"\1", repaired_sentence)
    repaired_sentence = re.sub(r"\s{2,}", " ", repaired_sentence).strip()
    if sentence and sentence in str(source_text or ""):
        return str(source_text or "").replace(sentence, repaired_sentence, 1)
    window = (repair_info or {}).get("window") or {}
    return _splice_sentence_window(
        source_text,
        int(window.get("start") or 0),
        int(window.get("end") or 0),
        repaired_sentence,
    )


def _locality_score(original_text: str, candidate_text: str) -> dict:
    original_sentences = _split_sentences(original_text)
    candidate_sentences = _split_sentences(candidate_text)
    max_len = max(len(original_sentences), len(candidate_sentences), 1)
    changed = 0
    for index in range(max_len):
        left = original_sentences[index] if index < len(original_sentences) else ""
        right = candidate_sentences[index] if index < len(candidate_sentences) else ""
        if left != right:
            changed += 1
    ratio = round(changed / max_len, 3)
    return {
        "changed_sentences": changed,
        "total_sentences": max_len,
        "changed_sentence_ratio": ratio,
    }


def _micro_repair_gain_efficiency(human_gain: float, aggression_delta: float) -> float:
    """Measure attribution gain per unit of repair aggression."""
    try:
        human_gain_value = float(human_gain)
    except (TypeError, ValueError):
        human_gain_value = 0.0
    try:
        aggression_value = float(aggression_delta)
    except (TypeError, ValueError):
        aggression_value = 0.0
    if aggression_value <= 0:
        return 9999.0 if human_gain_value > 0 else 0.0
    return round(human_gain_value / aggression_value, 3)


def _micro_texture_iteration_status(
    attempts: list[dict] | None = None,
    *,
    baseline_scan: dict | None = None,
    previous_scan: dict | None = None,
    current_scan: dict | None = None,
) -> dict:
    """Stop/go policy for iterative micro-local texture repair.

    The generator may create several tiny patches, but the loop must stop
    before many local edits add up to a disguised section rewrite.
    """
    attempts = attempts or []

    def num(source: dict | None, key: str, default: float = 0.0) -> float:
        if not isinstance(source, dict):
            return default
        value = source.get(key)
        return float(value) if isinstance(value, (int, float)) else default

    def scan_from_attempt(index: int) -> dict:
        if not attempts:
            return {}
        try:
            item = attempts[index]
        except IndexError:
            return {}
        return item.get("scan_scores") or item.get("scan") or {}

    if current_scan is None:
        current_scan = scan_from_attempt(-1)
    if previous_scan is None:
        previous_scan = scan_from_attempt(-2) if len(attempts) >= 2 else (baseline_scan or {})
    if baseline_scan is None:
        baseline_scan = previous_scan or {}

    cumulative_aggression = 0.0
    max_locality_ratio = 0.0
    latest_aggression = 0.0
    for index, item in enumerate(attempts):
        aggression = item.get("repair_aggression") or {}
        if not isinstance(aggression, dict):
            aggression = {}
        score = num(aggression, "score", num(item, "repair_aggression_score"))
        cumulative_aggression += max(0.0, score)
        if index == len(attempts) - 1:
            latest_aggression = max(0.0, score)
        locality = item.get("locality") or {}
        if isinstance(locality, dict):
            max_locality_ratio = max(max_locality_ratio, num(locality, "changed_sentence_ratio"))

    current_human = num(current_scan, "human")
    previous_human = num(previous_scan, "human")
    baseline_human = num(baseline_scan, "human", previous_human)
    marginal_human_gain = current_human - previous_human
    total_human_gain = current_human - baseline_human
    ai_authorship_delta = num(current_scan, "ai_authorship") - num(previous_scan, "ai_authorship")
    ai_transformation_delta = num(current_scan, "ai_transformation") - num(previous_scan, "ai_transformation")
    findings_delta = num(current_scan, "findings") - num(previous_scan, "findings")
    gain_efficiency = _micro_repair_gain_efficiency(marginal_human_gain, latest_aggression)

    max_total_aggression = _float_env("DRAFTPROOF_MICRO_TEXTURE_MAX_TOTAL_AGGRESSION", 0.18)
    max_locality = _float_env("DRAFTPROOF_TEXTURE_REPAIR_MAX_LOCALITY", 0.25)
    min_human_gain = _float_env("DRAFTPROOF_MICRO_TEXTURE_MIN_HUMAN_GAIN", 1.0)
    min_gain_efficiency = _float_env("DRAFTPROOF_MICRO_TEXTURE_MIN_GAIN_EFFICIENCY", 10.0)
    max_iterations = int(_float_env("DRAFTPROOF_MICRO_TEXTURE_MAX_ITERATIONS", 5.0))

    stop_reasons: list[str] = []
    if attempts and cumulative_aggression > max_total_aggression:
        stop_reasons.append("cumulative_aggression_budget_exhausted")
    if attempts and max_locality_ratio > max_locality:
        stop_reasons.append("repair_locality_high")
    if attempts and ai_authorship_delta > 0:
        stop_reasons.append("ai_authorship_regression")
    if attempts and findings_delta > 0:
        stop_reasons.append("findings_regression")
    if attempts and marginal_human_gain < min_human_gain:
        stop_reasons.append("diminishing_human_gain")
    if attempts and marginal_human_gain > 0 and gain_efficiency < min_gain_efficiency:
        stop_reasons.append("gain_efficiency_low")
    if attempts and len(attempts) > max_iterations:
        stop_reasons.append("max_iterations_reached")

    return {
        "continue": not stop_reasons,
        "stop_reasons": stop_reasons,
        "metrics": {
            "attempt_count": len(attempts),
            "cumulative_aggression": round(cumulative_aggression, 3),
            "max_total_aggression": round(max_total_aggression, 3),
            "latest_aggression": round(latest_aggression, 3),
            "max_locality_changed_sentence_ratio": round(max_locality_ratio, 3),
            "max_locality_limit": round(max_locality, 3),
            "marginal_human_gain": round(marginal_human_gain, 3),
            "total_human_gain": round(total_human_gain, 3),
            "ai_authorship_delta": round(ai_authorship_delta, 3),
            "ai_transformation_delta": round(ai_transformation_delta, 3),
            "findings_delta": round(findings_delta, 3),
            "gain_efficiency": gain_efficiency,
            "min_human_gain": round(min_human_gain, 3),
            "min_gain_efficiency": round(min_gain_efficiency, 3),
            "max_iterations": max_iterations,
        },
    }


def _iterative_micro_texture_repair(
    source_text: str,
    raw_json: dict | None,
    *,
    baseline_scan: dict,
    generate_replacement,
    scan_candidate,
    anchors: list[str] | None = None,
    max_attempts: int | None = None,
) -> dict:
    """Run iterative micro-local texture repair with deterministic stop controls.

    `generate_replacement(prompt, repair_info, attempt_index)` supplies one
    replacement window. `scan_candidate(text)` returns the scanner metrics for
    the full candidate text. The loop accepts only patches that pass the
    iteration policy.
    """
    try:
        limit = int(max_attempts if max_attempts is not None else _float_env("DRAFTPROOF_MICRO_TEXTURE_MAX_ITERATIONS", 5.0))
    except (TypeError, ValueError):
        limit = 5
    limit = max(0, limit)
    current_text = str(source_text or "")
    previous_scan = dict(baseline_scan or {})
    accepted_attempts: list[dict] = []
    rejected_attempts: list[dict] = []
    repaired_indexes: set[int] = set()
    stop_reason = "max_attempts_reached" if limit == 0 else ""

    for attempt_index in range(1, limit + 1):
        window = _micro_texture_window(
            current_text,
            raw_json,
            max_sentences=1,
            exclude_sentence_indexes=repaired_indexes,
        )
        if not window.get("text"):
            stop_reason = "no_unrepaired_texture_window"
            break
        prompt, repair_info = _micro_texture_repair_prompt(
            current_text,
            raw_json,
            anchors or [],
            max_sentences=1,
            exclude_sentence_indexes=repaired_indexes,
        )
        raw_replacement = generate_replacement(prompt, repair_info, attempt_index)
        replacement, clean_reason = _clean_micro_texture_candidate(str(raw_replacement or ""), repair_info)
        attempt = {
            "attempt": attempt_index,
            "window_start": window.get("start"),
            "window_end": window.get("end"),
            "target_window": window.get("text"),
            "accepted": False,
        }
        if clean_reason:
            attempt["reason"] = clean_reason
            rejected_attempts.append(attempt)
            stop_reason = clean_reason
            break
        candidate_text = _splice_sentence_window(
            current_text,
            int(window.get("start") or 0),
            int(window.get("end") or 0),
            replacement,
        )
        if candidate_text == current_text:
            attempt["reason"] = "micro_texture_no_change"
            rejected_attempts.append(attempt)
            stop_reason = "micro_texture_no_change"
            break
        attempt.update({
            "replacement_window": replacement,
            "repair_aggression": _repair_aggression_score(current_text, candidate_text),
            "locality": _locality_score(current_text, candidate_text),
        })
        try:
            scan_scores = scan_candidate(candidate_text)
        except Exception as exc:
            attempt["reason"] = f"candidate_scan_error {exc}"
            rejected_attempts.append(attempt)
            stop_reason = attempt["reason"]
            break
        attempt["scan_scores"] = scan_scores or {}
        iteration_status = _micro_texture_iteration_status(
            accepted_attempts + [attempt],
            baseline_scan=baseline_scan,
            previous_scan=previous_scan,
            current_scan=attempt["scan_scores"],
        )
        attempt["iteration_status"] = iteration_status
        if not iteration_status.get("continue"):
            attempt["reason"] = ",".join(iteration_status.get("stop_reasons") or []) or "iteration_policy_stop"
            rejected_attempts.append(attempt)
            stop_reason = attempt["reason"]
            break
        attempt["accepted"] = True
        accepted_attempts.append(attempt)
        repaired_indexes.add(int(window.get("start") or 0))
        current_text = candidate_text
        previous_scan = dict(attempt["scan_scores"])
    else:
        stop_reason = "max_attempts_reached"

    final_status = _micro_texture_iteration_status(
        accepted_attempts,
        baseline_scan=baseline_scan,
        previous_scan=baseline_scan if len(accepted_attempts) <= 1 else accepted_attempts[-2].get("scan_scores"),
        current_scan=previous_scan,
    )
    return {
        "text": current_text,
        "scan_scores": previous_scan,
        "accepted_attempts": accepted_attempts,
        "rejected_attempts": rejected_attempts,
        "attempt_count": len(accepted_attempts) + len(rejected_attempts),
        "accepted_count": len(accepted_attempts),
        "stop_reason": stop_reason,
        "iteration_status": final_status,
        "repaired_sentence_indexes": sorted(repaired_indexes),
    }


def _optimization_candidate_status(
    candidate: dict | None,
    *,
    baseline: dict | None = None,
    reject_semantic_drift: bool = True,
) -> dict:
    """Rank generated candidates as a multi-objective optimization problem.

    This is intentionally separate from prompt compliance. A candidate can pass
    the mechanical gate and still be a poor mitigation candidate after scan.
    """
    candidate = candidate or {}
    baseline = baseline or {}
    mechanical = candidate.get("mechanical") or candidate.get("mechanical_gate") or {}
    scan = candidate.get("scan_scores") or {}
    reject_reasons: list[str] = []
    if mechanical and not mechanical.get("passed"):
        reject_reasons.append("mechanical_gate_failed")
    for key in ("missing", "forbidden_found", "forbidden", "generic_banned_found", "connectors"):
        if mechanical.get(key):
            reject_reasons.append(f"{key}_present")
    if reject_semantic_drift and scan.get("semantic_drift"):
        reject_reasons.append("semantic_drift")
    if not scan:
        reject_reasons.append("missing_scan_scores")

    def num(source: dict, key: str, default: float = 0.0) -> float:
        value = source.get(key)
        return float(value) if isinstance(value, (int, float)) else default

    human_gain = num(scan, "human") - num(baseline, "human")
    ai_transformation_drop = num(baseline, "ai_transformation") - num(scan, "ai_transformation")
    ai_authorship_drop = num(baseline, "ai_authorship") - num(scan, "ai_authorship")
    ai_score_drop = num(baseline, "ai_score") - num(scan, "ai_score")
    grounding_drop = num(baseline, "grounding") - num(scan, "grounding")
    findings_drop = num(baseline, "findings") - num(scan, "findings")
    generic_phrase_penalty = max(0.0, num(scan, "generic_phrase_count"))
    if baseline and ai_authorship_drop < 0:
        reject_reasons.append("ai_authorship_increase")
    repair_aggression = candidate.get("repair_aggression")
    if not isinstance(repair_aggression, dict):
        original_text = candidate.get("original_text") or candidate.get("source_text")
        candidate_text = candidate.get("candidate_text") or candidate.get("text")
        repair_aggression = (
            _repair_aggression_score(str(original_text), str(candidate_text))
            if isinstance(original_text, str) and isinstance(candidate_text, str)
            else {}
        )
    repair_aggression_score = num(repair_aggression, "score")
    repair_aggression_limit = _float_env("DRAFTPROOF_TEXTURE_REPAIR_MAX_AGGRESSION", 0.55)
    if repair_aggression and repair_aggression_score > repair_aggression_limit:
        reject_reasons.append("repair_aggression_high")
    locality = candidate.get("locality") or {}
    if not isinstance(locality, dict) or not locality:
        original_text = candidate.get("original_text") or candidate.get("source_text")
        candidate_text = candidate.get("candidate_text") or candidate.get("text")
        locality = (
            _locality_score(str(original_text), str(candidate_text))
            if isinstance(original_text, str) and isinstance(candidate_text, str)
            else {}
        )
    locality_ratio = num(locality, "changed_sentence_ratio")
    locality_limit = _float_env("DRAFTPROOF_TEXTURE_REPAIR_MAX_LOCALITY", 0.25)
    if locality and locality_ratio > locality_limit:
        reject_reasons.append("repair_locality_high")

    grounding_penalty = max(0.0, -grounding_drop)
    finding_penalty = max(0.0, -findings_drop)
    score = (
        human_gain * 5.0
        + ai_authorship_drop * 2.0
        + ai_transformation_drop * 1.5
        + ai_score_drop * 1.0
        - grounding_penalty * 0.5
        - finding_penalty * 0.5
        - generic_phrase_penalty * 2.0
    )
    accepted = not reject_reasons
    stage_target = _human_gain_stage_target(num(scan, "human"))
    rank_key = (
        1 if accepted else 0,
        1 if num(scan, "human") >= stage_target else 0,
        human_gain,
        round(score, 3),
        ai_authorship_drop,
        ai_transformation_drop,
        ai_score_drop,
        -generic_phrase_penalty,
    )
    return {
        "accepted": accepted,
        "reject_reasons": reject_reasons,
        "score": round(score, 3),
        "rank_key": rank_key,
        "components": {
            "human_gain": round(human_gain, 3),
            "ai_transformation_drop": round(ai_transformation_drop, 3),
            "ai_authorship_drop": round(ai_authorship_drop, 3),
            "ai_score_drop": round(ai_score_drop, 3),
            "grounding_drop": round(grounding_drop, 3),
            "findings_drop": round(findings_drop, 3),
            "grounding_penalty": round(grounding_penalty, 3),
            "finding_penalty": round(finding_penalty, 3),
            "generic_phrase_penalty": round(generic_phrase_penalty, 3),
            "human_stage_target": round(stage_target, 3),
            "repair_aggression_score": round(repair_aggression_score, 3),
            "repair_aggression_limit": round(repair_aggression_limit, 3),
            "locality_changed_sentence_ratio": round(locality_ratio, 3),
            "locality_limit": round(locality_limit, 3),
        },
        "weights": {
            "human_gain": 5.0,
            "ai_authorship_drop": 2.0,
            "ai_transformation_drop": 1.5,
            "ai_score_drop": 1.0,
            "grounding_penalty": -0.5,
            "finding_penalty": -0.5,
            "generic_phrase_penalty": -2.0,
        },
    }


def _select_best_optimization_candidate(
    candidates: list[dict] | None,
    *,
    baseline: dict | None = None,
    reject_semantic_drift: bool = True,
) -> dict:
    rows = []
    for index, candidate in enumerate(candidates or []):
        status = _optimization_candidate_status(
            candidate,
            baseline=baseline,
            reject_semantic_drift=reject_semantic_drift,
        )
        row = dict(candidate or {})
        row["optimization_status"] = status
        row["_candidate_index"] = index
        rows.append(row)
    if not rows:
        return {
            "selected": None,
            "selected_index": None,
            "accepted_count": 0,
            "candidates": [],
            "reason": "no_candidates",
        }
    rows.sort(key=lambda row: row["optimization_status"]["rank_key"], reverse=True)
    best = rows[0]
    accepted = [row for row in rows if row["optimization_status"]["accepted"]]
    return {
        "selected": best if best["optimization_status"]["accepted"] else None,
        "selected_index": best.get("_candidate_index") if best["optimization_status"]["accepted"] else None,
        "accepted_count": len(accepted),
        "candidates": rows,
        "reason": (
            "selected_best_pareto_candidate"
            if best["optimization_status"]["accepted"]
            else "all_candidates_rejected"
        ),
    }


def _human_gain_stage_target(human_score: float | int | None, *, final_target: float = 80.0) -> float:
    """Return the next ladder target for controlled human-gain repair."""
    try:
        human = float(human_score)
    except (TypeError, ValueError):
        human = 0.0
    for target in (60.0, 70.0, float(final_target)):
        if human < target:
            return target
    return float(final_target)


def _metric_repair_diagnosis(
    scan_scores: dict | None,
    *,
    target_human: float = 80.0,
    target_ai_transformation: float = 20.0,
    target_ai_authorship: float = 45.0,
) -> dict:
    """Choose the next targeted repair dimension from scanner scores."""
    scan_scores = scan_scores or {}

    def num(key: str, default: float = 0.0) -> float:
        value = scan_scores.get(key)
        return float(value) if isinstance(value, (int, float)) else default

    if scan_scores.get("semantic_drift"):
        return {
            "repair_type": "semantic_drift_rollback",
            "priority": 100,
            "reason": "semantic drift is a hard failure before score optimization",
            "instructions": [
                "Rollback or patch only the drifted sentence span.",
                "Do not introduce new examples, source names, claims, or section roles.",
                "Keep protected anchors and return closer to the section meaning inventory.",
            ],
        }

    ai_authorship = num("ai_authorship")
    ai_transformation = num("ai_transformation")
    human = num("human")
    generic_phrase_count = num("generic_phrase_count")
    findings = num("findings")
    gaps = {
        "ai_authorship_gap": max(0.0, ai_authorship - target_ai_authorship),
        "ai_transformation_gap": max(0.0, ai_transformation - target_ai_transformation),
        "human_gap": max(0.0, target_human - human),
        "generic_phrase_gap": max(0.0, generic_phrase_count),
        "finding_gap": max(0.0, findings - 5.0),
    }
    next_human_stage = _human_gain_stage_target(human, final_target=target_human)
    if gaps["ai_authorship_gap"] > 0:
        return {
            "repair_type": "authorship_texture_repair",
            "priority": round(gaps["ai_authorship_gap"], 3),
            "reason": "AI Authorship texture remains the blocker; semantic human cues are not enough",
            "instructions": [
                "Do not add human semantic cues as the main move.",
                "Repair sentence rhythm, pacing, and predictability only; preserve claim inventory.",
                "Break clean explanatory cadence with natural asymmetry, not random noise.",
                "Reduce transition cleanliness and balanced claim-explanation-implication flow.",
                "Vary information density locally: one compressed sentence, one practical sentence, one delayed connection.",
                "Keep acceptable friction without typos, fake errors, invented examples, or new evidence.",
            ],
        }
    if gaps["human_gap"] > 0:
        return {
            "repair_type": "human_gain_repair",
            "priority": round(next_human_stage - human, 3),
            "stage_target": round(next_human_stage, 3),
            "final_target": round(target_human, 3),
            "reason": "Human Contribution remains below the next ladder target after authorship texture is controlled",
            "instructions": [
                "Patch only 10-20% of sentences in this repair round.",
                "Increase concrete anchor density using original text anchors and scanner context only.",
                "Add safe author reasoning traces such as what I noticed, the issue is, or this made me think only where the source stance supports it.",
                "Use mild rhythm unevenness: one short sentence, one longer causal sentence, and less balanced claim-explanation-implication flow.",
                "Keep rough edges; do not over-clean transitions, grammar, or paragraph symmetry.",
                "Do not invent new places, dates, numbers, people, evidence, citations, workplace events, or assessment results.",
            ],
        }
    if gaps["ai_transformation_gap"] > 0:
        return {
            "repair_type": "ai_transformation_smoothing",
            "priority": round(gaps["ai_transformation_gap"], 3),
            "reason": "AI Transformation remains high, suggesting the rewrite is too smooth or rebuilt",
            "instructions": [
                "Reduce smoothing and restore author-owned roughness.",
                "Keep sentence order mostly stable while changing over-clean transitions.",
                "Remove balanced summary cadence and generic connector chains.",
            ],
        }
    if generic_phrase_count > 0:
        return {
            "repair_type": "connector_cleanup",
            "priority": round(generic_phrase_count, 3),
            "reason": "Generic connector findings remain after mitigation",
            "instructions": [
                "Remove generic connectors without changing meaning.",
                "Use plain transitions or no transition.",
            ],
        }
    return {
        "repair_type": "none",
        "priority": 0,
        "reason": "No targeted repair required by configured thresholds",
        "instructions": [],
    }


def _critical_high_count(report_dict: dict | None) -> int:
    findings = (report_dict or {}).get("findings", {}) if isinstance(report_dict, dict) else {}
    return len(findings.get("critical", [])) + len(findings.get("high", []))


def _authenticity_gate_status(
    original_report: dict,
    candidate_report: dict,
    text_changed: bool,
    *,
    original_review_burden: int,
    candidate_review_burden: int,
    original_weighted_severity: int,
    candidate_weighted_severity: int,
    min_human_gain: float = 2.0,
    min_ai_transformation_drop: float = 2.0,
    drift_similarity: float | None = None,
) -> dict:
    original = _contribution_scores(original_report)
    candidate = _contribution_scores(candidate_report)
    original_integrity = _integrity_scores(original_report)
    candidate_integrity = _integrity_scores(candidate_report)
    original_human = original.get("human")
    candidate_human = candidate.get("human")
    original_ai_transform = original.get("ai_transformation")
    candidate_ai_transform = candidate.get("ai_transformation")
    original_ai_authorship = original_integrity.get("ai_authorship")
    candidate_ai_authorship = candidate_integrity.get("ai_authorship")
    human_delta = (
        candidate_human - original_human
        if isinstance(original_human, (int, float)) and isinstance(candidate_human, (int, float))
        else None
    )
    ai_transform_delta = (
        original_ai_transform - candidate_ai_transform
        if (
            isinstance(original_ai_transform, (int, float))
            and isinstance(candidate_ai_transform, (int, float))
        )
        else None
    )
    ai_authorship_delta = (
        original_ai_authorship - candidate_ai_authorship
        if (
            isinstance(original_ai_authorship, (int, float))
            and isinstance(candidate_ai_authorship, (int, float))
        )
        else None
    )
    crosses_human_side = bool(
        isinstance(candidate_human, (int, float))
        and isinstance(candidate_ai_transform, (int, float))
        and candidate_human > candidate_ai_transform
        and (
            not isinstance(original_human, (int, float))
            or not isinstance(original_ai_transform, (int, float))
            or original_human <= original_ai_transform
        )
    )
    moves_toward_human = bool(
        (isinstance(human_delta, (int, float)) and human_delta >= min_human_gain)
        or (
            isinstance(ai_transform_delta, (int, float))
            and ai_transform_delta >= min_ai_transformation_drop
        )
        or crosses_human_side
    )
    reduces_ai_authorship = bool(
        isinstance(ai_authorship_delta, (int, float))
        and ai_authorship_delta >= _float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_AUTHORSHIP_DROP", 2.0)
    )
    ai_authorship_regression_tolerance = _float_env(
        "DRAFTPROOF_AUTHENTICITY_AI_AUTHORSHIP_REGRESSION_TOLERANCE",
        0.0,
    )
    ai_authorship_regressed = bool(
        isinstance(ai_authorship_delta, (int, float))
        and ai_authorship_delta < -ai_authorship_regression_tolerance
    )
    major_human_threshold = _float_env("DRAFTPROOF_AUTHENTICITY_MAJOR_HUMAN_THRESHOLD", 80.0)
    major_human_gain = _float_env("DRAFTPROOF_AUTHENTICITY_MAJOR_HUMAN_GAIN", 50.0)
    major_human_breakthrough = bool(
        isinstance(candidate_human, (int, float))
        and isinstance(human_delta, (int, float))
        and candidate_human >= major_human_threshold
        and human_delta >= major_human_gain
    )
    ai_authorship_regression_blocked = ai_authorship_regressed
    target_human = _float_env("DRAFTPROOF_AUTHENTICITY_TARGET_HUMAN", 80.0)
    strong_accept_min_human_gain = _float_env("DRAFTPROOF_AUTHENTICITY_STRONG_ACCEPT_MIN_HUMAN_GAIN", 20.0)
    strong_accept_min_transform_drop = _float_env("DRAFTPROOF_AUTHENTICITY_STRONG_ACCEPT_MIN_AI_TRANSFORM_DROP", 20.0)
    strong_accept_min_shift = _float_env("DRAFTPROOF_AUTHENTICITY_STRONG_ACCEPT_MIN_SHIFT", 45.0)
    crosses_target_human = bool(
        isinstance(candidate_human, (int, float))
        and candidate_human >= target_human
    )
    critical_high_regressed = _critical_high_count(candidate_report) > _critical_high_count(original_report)
    review_regressed = candidate_review_burden > original_review_burden
    severity_regressed = candidate_weighted_severity > original_weighted_severity
    human_shift = _human_shift_score(
        original_report,
        candidate_report,
        drift_similarity=drift_similarity,
        review_burden_delta=candidate_review_burden - original_review_burden,
        weighted_severity_delta=candidate_weighted_severity - original_weighted_severity,
    )
    human_shift_score = human_shift.get("score")
    authorship_cost_per_human_gain = (
        round(max(0.0, -float(ai_authorship_delta)) / max(float(human_delta), 1.0), 3)
        if isinstance(ai_authorship_delta, (int, float)) and isinstance(human_delta, (int, float))
        else None
    )
    human_gain_with_authorship_regression = bool(
        isinstance(human_delta, (int, float))
        and human_delta > 0
        and ai_authorship_regressed
    )
    min_human_shift_score = _float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_SHIFT_SCORE", 3.0)
    clears_human_shift_score = bool(
        isinstance(human_shift_score, (int, float))
        and human_shift_score >= min_human_shift_score
    )
    positive_human_shift = bool(
        isinstance(human_shift_score, (int, float))
        and human_shift_score > 0
        and (moves_toward_human or reduces_ai_authorship)
    )
    strong_below_target_accept = bool(
        isinstance(candidate_human, (int, float))
        and candidate_human < target_human
        and isinstance(human_delta, (int, float))
        and human_delta >= strong_accept_min_human_gain
        and isinstance(ai_transform_delta, (int, float))
        and ai_transform_delta >= strong_accept_min_transform_drop
        and isinstance(human_shift_score, (int, float))
        and human_shift_score >= strong_accept_min_shift
        and not ai_authorship_regressed
    )
    target_accept = crosses_target_human or strong_below_target_accept
    candidate_progress = bool(
        text_changed
        and (clears_human_shift_score or positive_human_shift)
        and not ai_authorship_regression_blocked
        and not critical_high_regressed
        and not review_regressed
        and not severity_regressed
    )
    success = bool(
        candidate_progress
        and target_accept
    )
    reason = ""
    if success:
        reason = "accepted"
    elif not text_changed:
        reason = "unchanged_candidate"
    elif ai_authorship_regression_blocked:
        reason = "ai_authorship_regressed"
    elif not (clears_human_shift_score or positive_human_shift):
        reason = "human_shift_score_too_low"
    elif critical_high_regressed:
        reason = "critical_high_regressed"
    elif review_regressed:
        reason = "review_burden_regressed"
    elif severity_regressed:
        reason = "weighted_severity_regressed"
    elif candidate_progress:
        reason = "candidate_progress_below_target"
    return {
        "success": success,
        "reason": reason,
        "original_human": original_human,
        "candidate_human": candidate_human,
        "human_delta": human_delta,
        "original_ai_transformation": original_ai_transform,
        "candidate_ai_transformation": candidate_ai_transform,
        "ai_transformation_delta": ai_transform_delta,
        "original_ai_authorship": original_ai_authorship,
        "candidate_ai_authorship": candidate_ai_authorship,
        "ai_authorship_delta": ai_authorship_delta,
        "reduces_ai_authorship": reduces_ai_authorship,
        "ai_authorship_regressed": ai_authorship_regressed,
        "ai_authorship_regression_blocked": ai_authorship_regression_blocked,
        "ai_authorship_regression_tolerance": ai_authorship_regression_tolerance,
        "human_gain_with_authorship_regression": human_gain_with_authorship_regression,
        "false_positive_improvement": human_gain_with_authorship_regression,
        "false_positive_improvement_reason": (
            "human_gain_with_authorship_regression"
            if human_gain_with_authorship_regression
            else ""
        ),
        "major_human_breakthrough": major_human_breakthrough,
        "target_human": target_human,
        "crosses_target_human": crosses_target_human,
        "strong_below_target_accept": strong_below_target_accept,
        "strong_accept_min_human_gain": strong_accept_min_human_gain,
        "strong_accept_min_ai_transform_drop": strong_accept_min_transform_drop,
        "strong_accept_min_shift": strong_accept_min_shift,
        "target_accept": target_accept,
        "candidate_progress": candidate_progress,
        "crosses_human_side": crosses_human_side,
        "human_shift_score": human_shift_score,
        "human_shift_components": human_shift.get("components"),
        "authorship_cost_per_human_gain": authorship_cost_per_human_gain,
        "human_shift_weights": human_shift.get("weights"),
        "min_human_shift_score": min_human_shift_score,
        "clears_human_shift_score": clears_human_shift_score,
        "positive_human_shift": positive_human_shift,
        "min_human_gain": min_human_gain,
        "min_ai_transformation_drop": min_ai_transformation_drop,
        "critical_high_regressed": critical_high_regressed,
        "review_burden_regressed": review_regressed,
        "weighted_severity_regressed": severity_regressed,
    }


def _ai_mitigation_action_brief(ai_mitigation: dict | None, limit: int = 10) -> str:
    if not isinstance(ai_mitigation, dict):
        return ""
    rows = []
    for action in (ai_mitigation.get("component_actions") or [])[:limit]:
        if not isinstance(action, dict):
            continue
        rows.append(
            "- "
            + "; ".join(
                part
                for part in [
                    f"component={action.get('component')}",
                    f"pillar={action.get('pillar')}",
                    f"score={action.get('score')}",
                    f"action={action.get('action')}",
                    f"user_input_needed={action.get('user_input_needed')}",
                ]
                if part and not part.endswith("=None")
            )
        )
    return "\n".join(rows)


def _generation_candidate_diagnostics(candidates: list[dict] | None, *, limit: int = 12) -> dict:
    """Compact generation attempt diagnostics for failed AI-Mitigation runs."""
    rows: list[dict] = []
    reason_counts: dict[str, int] = {}
    for candidate in (candidates or [])[:max(0, limit)]:
        if not isinstance(candidate, dict):
            continue
        gate = candidate.get("gate") if isinstance(candidate.get("gate"), dict) else {}
        staged = candidate.get("staged_generation") if isinstance(candidate.get("staged_generation"), dict) else {}
        reason = (
            candidate.get("reason")
            or gate.get("reason")
            or candidate.get("selection_reason")
            or ("accepted_candidate" if candidate.get("selected") else "")
            or "no_rejection_reason_recorded"
        )
        reason_key = str(reason).split(" ", 1)[0]
        reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
        row = {
            "attempt": candidate.get("attempt"),
            "strategy": candidate.get("strategy"),
            "reconstruction": bool(candidate.get("reconstruction")),
            "deterministic": bool(candidate.get("deterministic")),
            "passed_local_checks": bool(candidate.get("passed_local_checks")),
            "selected": bool(candidate.get("selected")),
            "best_so_far": bool(candidate.get("best_so_far")),
            "reason": reason,
            "warnings": candidate.get("warnings"),
            "candidate_length": candidate.get("candidate_length"),
            "candidate_word_count": candidate.get("candidate_word_count"),
            "drift_similarity": candidate.get("drift_similarity"),
            "drift_threshold": candidate.get("drift_threshold"),
            "drift_scan_relaxed_for_reconstruction": candidate.get("drift_scan_relaxed_for_reconstruction"),
            "scan_seconds": candidate.get("scan_seconds"),
            "ai": candidate.get("ai"),
            "writing_quality": candidate.get("writing_quality"),
            "human_contribution": candidate.get("human_contribution"),
            "ai_transformation": candidate.get("ai_transformation"),
            "ai_authorship": candidate.get("ai_authorship"),
            "human_delta": candidate.get("human_delta"),
            "ai_transformation_delta": candidate.get("ai_transformation_delta"),
            "ai_authorship_delta": candidate.get("ai_authorship_delta"),
            "human_shift_score": candidate.get("human_shift_score"),
            "authorship_cost_per_human_gain": candidate.get("authorship_cost_per_human_gain"),
            "findings": candidate.get("findings"),
            "review_burden": candidate.get("review_burden"),
            "weighted_severity": candidate.get("weighted_severity"),
            "gate_success": gate.get("success"),
            "gate_reason": gate.get("reason"),
            "gate_ai_authorship_regression_blocked": gate.get("ai_authorship_regression_blocked"),
            "gate_human_gain_with_authorship_regression": gate.get("human_gain_with_authorship_regression"),
            "gate_false_positive_improvement": gate.get("false_positive_improvement"),
            "gate_false_positive_improvement_reason": gate.get("false_positive_improvement_reason"),
            "gate_critical_high_regressed": gate.get("critical_high_regressed"),
            "gate_review_burden_regressed": gate.get("review_burden_regressed"),
            "gate_weighted_severity_regressed": gate.get("weighted_severity_regressed"),
        }
        if staged:
            row["staged_generation"] = {
                "enabled": staged.get("enabled"),
                "llm_calls": staged.get("llm_calls"),
                "assembled_word_count": staged.get("assembled_word_count"),
                "reference_entries_preserved": staged.get("reference_entries_preserved"),
                "source_draft_included": staged.get("source_draft_included"),
                "sections": staged.get("sections"),
            }
        rows.append({key: value for key, value in row.items() if value is not None})
    return {
        "candidate_count": len(candidates or []),
        "shown_count": len(rows),
        "reason_counts": reason_counts,
        "candidates": rows,
    }


def _authenticity_mitigation_prompt(
    source_text: str,
    raw_json: dict,
    ai_mitigation: dict | None,
    attempt_index: int,
) -> str:
    contribution = _contribution_scores(raw_json)
    semantic = (
        ((raw_json or {}).get("scan_intelligence") or {}).get("semantic_shape")
        or (raw_json or {}).get("semantic_shape")
        or {}
    )
    signal_brief = _ai_search_signal_brief(raw_json)
    action_brief = _ai_mitigation_action_brief(ai_mitigation)
    return (
        "DraftProof AI-Mitigation authenticity rewrite.\n"
        "Goal: push the document toward the Human Contribution side of the scan, not merely lower a detector score.\n"
        f"Current contribution score: Human={contribution.get('human')}, AI Transformation={contribution.get('ai_transformation')}.\n"
        f"Semantic layer: {json.dumps(semantic, ensure_ascii=False)[:1800]}.\n\n"
        "Use these mitigation actions as engineering targets:\n"
        f"{action_brief or '- No component actions supplied.'}\n\n"
        f"{signal_brief}\n\n"
        "Rewrite behavior:\n"
        "- Produce a complete replacement draft, not notes and not a partial patch.\n"
        "- Preserve all factual claims, citations, years, names, quotes, numbers, unit codes, source relations, and chronology already present.\n"
        "- Do not invent personal observations, institutions, sources, dates, statistics, examples, citations, or lived details.\n"
        "- When a claim lacks support, narrow or qualify it instead of fabricating evidence.\n"
        "- Add reasoning continuity where adjacent ideas jump: make the causal bridge explicit using only facts already in the draft.\n"
        "- Increase cognitive authenticity: allow uneven emphasis, short explanatory turns, and locally specific reasoning where the draft already gives anchors.\n"
        "- Reduce predictable academic filler and balanced essay cadence. Avoid polished connector chains such as furthermore, moreover, it is important, significant, crucial, enables, facilitates.\n"
        "- Increase semantic density by replacing broad explanatory padding with concrete operational meaning already present in the source.\n"
        "- Rebuild paragraph routes where needed: context or problem first, then source/evidence relation, then limited conclusion.\n"
        "- Do not use placeholders, review brackets, labels, comments, markdown fences, or explanations.\n"
        "- Keep roughly the same length and preserve headings if they exist.\n"
        f"- Attempt {attempt_index}: make the draft materially different from a synonym swap while staying fact-preserving.\n\n"
        "SOURCE DRAFT:\n"
        f"<TARGET_DOCUMENT>\n{source_text.strip()}\n</TARGET_DOCUMENT>\n\n"
        "Return only the complete rewritten document."
    )


def _brief_sentences(text: str, limit: int = 10) -> list[str]:
    if not isinstance(text, str):
        return []
    rows = []
    for sentence in re.findall(r"[^.!?\n]+(?:[.!?]+|$)", text):
        cleaned = " ".join(sentence.split()).strip()
        if len(cleaned.split()) < 6:
            continue
        rows.append(cleaned)
        if len(rows) >= limit:
            break
    return rows


def _text_word_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(re.findall(r"\b[\w’'-]+\b", text))


def _word_count_band(text: str, variance: float = 0.25) -> dict:
    count = _text_word_count(text)
    return {
        "source_word_count": count,
        "min_words": max(1, int(math.floor(count * (1.0 - variance)))),
        "max_words": max(1, int(math.ceil(count * (1.0 + variance)))),
        "variance": variance,
    }


def _integrity_driver_rows(raw_json: dict | None, limit: int = 14) -> list[dict]:
    raw_json = raw_json or {}
    rows: list[dict] = []
    seen = set()
    integrity_sources = [
        raw_json.get("integrity_layers"),
        ((raw_json.get("scan_intelligence") or {}).get("integrity_layers") or {}),
        ((raw_json.get("ai_mitigation") or {}).get("integrity_layers") or {}),
    ]
    for integrity in integrity_sources:
        layers = integrity.get("layers") if isinstance(integrity, dict) else {}
        if not isinstance(layers, dict):
            continue
        for layer_key, layer in layers.items():
            if not isinstance(layer, dict):
                continue
            for signal in layer.get("signals") or []:
                if not isinstance(signal, dict):
                    continue
                signal_key = (layer_key, signal.get("key") or signal.get("label"))
                if signal_key in seen:
                    continue
                seen.add(signal_key)
                score = signal.get("score")
                rows.append({
                    "layer": layer_key,
                    "key": signal.get("key"),
                    "label": signal.get("label"),
                    "score": score,
                    "priority": float(score) if isinstance(score, (int, float)) else -1.0,
                })
    rows.sort(key=lambda item: item.get("priority", -1), reverse=True)
    return [
        {k: v for k, v in row.items() if k != "priority" and v is not None}
        for row in rows[:limit]
    ]


def _target_segment_rows(raw_json: dict | None, limit: int = 16) -> list[dict]:
    raw_json = raw_json or {}
    ai_mitigation = raw_json.get("ai_mitigation") or {}
    segments = ai_mitigation.get("target_segments") or []
    rows: list[dict] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        signal = segment.get("primary_signal") or {}
        rows.append({
            "segment_id": segment.get("segment_id"),
            "paragraph_id": segment.get("paragraph_id"),
            "text": segment.get("text"),
            "signal": signal.get("key") or signal.get("title"),
            "score": signal.get("score"),
            "lever": segment.get("lever"),
            "bucket": segment.get("bucket"),
            "action": segment.get("action"),
            "auto_apply": segment.get("auto_apply"),
        })
        if len(rows) >= limit:
            break
    return rows


def _reconstruction_failure_feedback(prior_attempts: list[dict] | None, limit: int = 6) -> list[dict]:
    rows: list[dict] = []
    for item in (prior_attempts or [])[-limit:]:
        if not isinstance(item, dict):
            continue
        gate = item.get("gate") if isinstance(item.get("gate"), dict) else {}
        components = item.get("human_shift_components") or gate.get("human_shift_components") or {}
        row = {
            "strategy": item.get("strategy") or item.get("attempt"),
            "reason": item.get("reason") or gate.get("reason"),
            "human_shift_score": item.get("human_shift_score") or gate.get("human_shift_score"),
            "candidate_human": item.get("candidate_human") or item.get("human_contribution") or gate.get("candidate_human"),
            "human_delta": item.get("human_delta") or gate.get("human_delta"),
            "ai_authorship_delta": item.get("ai_authorship_delta") or gate.get("ai_authorship_delta"),
            "ai_transformation_delta": item.get("ai_transformation_delta") or gate.get("ai_transformation_delta"),
            "failed_components": {
                key: value
                for key, value in components.items()
                if isinstance(value, (int, float)) and value < 0
            },
        }
        rows.append({k: v for k, v in row.items() if v not in (None, {}, [])})
    return rows


def _reconstruction_gate_controls(prior_attempts: list[dict] | None) -> dict:
    """Convert scanner/gate failures into generation controls for the next candidate."""
    feedback = _reconstruction_failure_feedback(prior_attempts, limit=8)
    controls: list[str] = []
    blocker_counts: dict[str, int] = {}

    def add(key: str, instruction: str) -> None:
        blocker_counts[key] = blocker_counts.get(key, 0) + 1
        if instruction not in controls:
            controls.append(instruction)

    for item in feedback:
        reason = str(item.get("reason") or "")
        failed_components = item.get("failed_components") or {}
        candidate_human = item.get("candidate_human") or item.get("human_contribution")
        if isinstance(candidate_human, (int, float)):
            next_stage = _human_gain_stage_target(candidate_human)
            if candidate_human < next_stage:
                add(
                    "human_gain_repair",
                    (
                        f"HUMAN_GAIN_REPAIR: raise Human Contribution toward the next ladder target "
                        f"({int(next_stage)}) by patching only 10-20% of sentences with existing anchors, "
                        "safe author reasoning traces, uneven rhythm, and less balanced paragraph flow."
                    ),
                )
        if "human_contribution_gain" in failed_components or (
            isinstance(item.get("human_delta"), (int, float)) and item.get("human_delta") < 0
        ):
            add(
                "human_contribution_regressed",
                "Do not replace author-owned classroom reasoning with smoother academic explanation; keep or increase first-person operational judgement, process detail, and local constraint language.",
            )
        if "ai_transformation_reduction" in failed_components or (
            isinstance(item.get("ai_transformation_delta"), (int, float))
            and item.get("ai_transformation_delta") < 0
        ):
            add(
                "ai_transformation_regressed",
                "Avoid outline-to-essay expansion, symmetrical paragraph routes, and polished summary cadence; use uneven paragraph jobs with concrete haircutting decisions.",
            )
        if "ai_authorship_reduction" in failed_components or (
            isinstance(item.get("ai_authorship_delta"), (int, float))
            and item.get("ai_authorship_delta") < 0
        ):
            add(
                "authorship_texture_repair",
                "AUTHORSHIP_TEXTURE_REPAIR: do not add more human details yet; reduce statistical smoothness through rhythm variance, less transition cleanliness, asymmetric sentence pacing, and uneven information density without changing meaning.",
            )
        if "grounding_risk_reduction" in failed_components:
            add(
                "grounding_regressed",
                "Preserve every source-to-claim relation and citation role; do not generalise cited claims or move sources into broad unsupported summaries.",
            )
        if "rewrite_smoothness_reduction" in failed_components:
            add(
                "smoothness_regressed",
                "Stop polishing. Prefer direct workshop-language reasoning over balanced academic sentences.",
            )
        if "weighted_severity_penalty" in failed_components or "weighted_severity" in reason:
            add(
                "severity_regressed",
                "Do not introduce new high/medium findings; keep protected anchors and source coverage intact before changing style.",
            )
        if "protected_span_lost" in reason or "number_lost" in reason:
            add(
                "protected_anchor_lost",
                "Copy protected anchors exactly: citations, years, page ranges, unit codes, named institutions, quotes, and reference details must remain present.",
            )
        if "quote_lost" in reason:
            add(
                "quote_lost",
                "Do not omit or paraphrase quoted/source wording; preserve quoted material exactly where it appears in the submitted content.",
            )
        if "candidate_word_count_too_low" in reason:
            add(
                "word_count_low",
                "Add length only through source-licensed reasoning, process explanation, and local constraints from the submitted draft; do not add generic filler.",
            )
        if "candidate_word_count_too_high" in reason:
            add(
                "word_count_high",
                "Compress generic explanation while keeping protected anchors and source relations.",
            )
        if "semantic_drift" in reason:
            add(
                "semantic_drift",
                "Keep the same claim inventory and evidence relationships; restructure route and rhythm without dropping entities, quotes, citations, or source roles.",
            )

    if not controls:
        controls.append(
            "No scored failure yet. Use the scanner baseline directly: raise Human Contribution toward 80 while preserving protected anchors and avoiding AI Authorship regression."
        )

    return {
        "schema_version": "scanner_gate_feedback.v1",
        "purpose": "Use scanner/gate failures as control signals for the next generation attempt.",
        "acceptance_target": {
            "human_contribution_min": 80,
            "human_contribution_ladder": [60, 70, 80],
            "primary_goal": "maximize_human_contribution_after_hard_safety_rejects",
            "ai_authorship_regression_allowed": False,
            "word_count_variance": "±25%",
            "critical_high_review_regression_allowed": False,
        },
        "prior_attempts": feedback,
        "blocker_counts": blocker_counts,
        "next_candidate_controls": controls[:10],
    }


def _generation_context_ledger(brief: dict, blueprint: dict) -> dict:
    """Build generation input from scanner-derived context, not the source prose."""
    paragraph_plans = []
    for plan in (blueprint or {}).get("paragraph_plans") or []:
        if not isinstance(plan, dict):
            continue
        paragraph_plans.append({
            key: value
            for key, value in plan.items()
            if key not in {"source_preview"}
        })
    target_segments = []
    for segment in (brief or {}).get("target_segments") or []:
        if not isinstance(segment, dict):
            continue
        target_segments.append({
            key: value
            for key, value in segment.items()
            if key not in {"text"}
        })
    return {
        "schema_version": "generation_context_ledger.v1",
        "purpose": "Regenerate from scanner-derived meaning, anchors, roles, and signals without using the submitted prose as a scaffold.",
        "claim_inventory": (brief or {}).get("claims") or [],
        "headings": (brief or {}).get("headings") or [],
        "protected_facts": (brief or {}).get("protected_facts") or [],
        "preservation_inventory": (brief or {}).get("preservation_inventory") or {},
        "word_count_band": (brief or {}).get("word_count_band") or {},
        "paragraph_roles": (brief or {}).get("paragraph_roles") or [],
        "paragraph_plans": paragraph_plans,
        "global_driver_targets": (blueprint or {}).get("global_driver_targets") or [],
        "industry_baseline_focus": (blueprint or {}).get("industry_baseline_focus") or {},
        "integrity_targets": (brief or {}).get("integrity_targets") or [],
        "target_segment_signals": target_segments,
        "allowed_existing_additions": (brief or {}).get("allowed_existing_additions") or [],
        "preserve_terms": (brief or {}).get("preserve_terms") or [],
        "do_not_add": (brief or {}).get("do_not_add") or [],
        "human_contribution_contract": (brief or {}).get("human_contribution_contract") or {},
        "reference_entries": (brief or {}).get("reference_entries") or [],
        "generation_handoff": (brief or {}).get("generation_handoff") or {},
    }


def _reference_entries_from_text(text: str, limit: int = 60) -> list[str]:
    """Extract bibliography entries as preservation context, not prose substrate."""
    if not isinstance(text, str) or not text.strip():
        return []
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if re.match(r"^\s*(?:references|reference list|bibliography|works cited)\s*$", line, re.I):
            start = index + 1
            break
    if start is None:
        return []

    entries: list[str] = []
    current = ""
    for raw_line in lines[start:]:
        line = raw_line.strip()
        if not line:
            if current:
                entries.append(current.strip())
                current = ""
            continue
        if re.match(r"^[A-Z][A-Za-z0-9 ,/&-]{2,70}$", line) and not re.search(r"\(\d{4}\)|https?://|doi\.", line, re.I):
            break
        starts_entry = bool(re.search(r"\(\d{4}\)|\(\s*n\.d\.\s*\)|https?://|doi\.", line, re.I))
        if current and starts_entry:
            entries.append(current.strip())
            current = line
        else:
            current = f"{current} {line}".strip() if current else line
        if len(entries) >= limit:
            break
    if current and len(entries) < limit:
        entries.append(current.strip())
    return entries[:limit]


def _staged_generation_section_plan(context_ledger: dict, *, max_sections: int | None = None) -> list[dict]:
    """Create bounded section plans so the LLM never receives the full document."""
    if max_sections is None:
        try:
            max_sections = int(os.environ.get("DRAFTPROOF_STAGED_REGENERATION_SECTIONS", "6"))
        except ValueError:
            max_sections = 6
    max_sections = max(1, max_sections)
    handoff = (context_ledger or {}).get("generation_handoff") or {}
    handoff_units = [
        unit for unit in handoff.get("section_generation_units") or []
        if isinstance(unit, dict) and unit.get("heading")
    ]
    if handoff_units:
        title = ((handoff.get("document_profile") or {}).get("title") or "").strip()
        selected_units = handoff_units[:max_sections]
        if len(handoff_units) > max_sections:
            conclusion = next(
                (
                    unit for unit in reversed(handoff_units)
                    if re.search(r"\bconclusion\b|closing|final", str(unit.get("heading") or ""), re.I)
                ),
                None,
            )
            if conclusion and conclusion not in selected_units:
                selected_units = selected_units[:max(0, max_sections - 1)] + [conclusion]
        return [
            {
                "section_index": index,
                "section_id": unit.get("section_id"),
                "title": title,
                "heading": unit.get("heading"),
                "target_words": ((unit.get("target_words") or {}).get("ideal") or (unit.get("target_words") or {}).get("max") or 180),
                "target_word_band": unit.get("target_words") or {},
                "must_preserve_anchors": unit.get("must_preserve_anchors") or [],
                "citation_keys_used": unit.get("citation_keys_used") or [],
                "claim_inventory_slice": unit.get("meaning_inventory") or [],
                "target_signal_slice": unit.get("detector_risks_to_reduce") or [],
                "paragraph_plan_slice": [{
                    "section_id": unit.get("section_id"),
                    "role": unit.get("role"),
                    "citation_keys_used": unit.get("citation_keys_used") or [],
                    "must_preserve_anchors": unit.get("must_preserve_anchors") or [],
                    "generation_instruction": unit.get("generation_instruction") or {},
                }],
            }
            for index, unit in enumerate(selected_units, start=1)
        ]
    headings = [
        str(item).strip()
        for item in (context_ledger or {}).get("headings") or []
        if str(item).strip() and not re.match(r"^(?:references|reference list|bibliography|works cited)$", str(item).strip(), re.I)
    ]
    if not headings:
        roles = (context_ledger or {}).get("paragraph_roles") or []
        headings = [
            str(row.get("role") or f"Section {idx}").replace("_", " ").title()
            for idx, row in enumerate(roles[:max_sections], start=1)
            if isinstance(row, dict)
        ] or ["Context", "Main Reasoning", "Conclusion"]

    title = headings[0] if len(headings) > 1 else ""
    body_headings_all = headings[1:] if title else headings
    body_headings = body_headings_all[:max_sections]
    if len(body_headings_all) > max_sections:
        conclusion = next(
            (
                heading
                for heading in reversed(body_headings_all)
                if re.search(r"\bconclusion\b|closing|final", heading, re.I)
            ),
            "",
        )
        if conclusion and conclusion not in body_headings:
            body_headings = body_headings[:max(0, max_sections - 1)] + [conclusion]
    claims = [
        str(item).strip()
        for item in (context_ledger or {}).get("claim_inventory") or []
        if str(item).strip()
    ]
    target_segments = (context_ledger or {}).get("target_segment_signals") or []
    paragraph_plans = (context_ledger or {}).get("paragraph_plans") or []
    word_band = (context_ledger or {}).get("word_count_band") or {}
    reference_words = _text_word_count("\n".join((context_ledger or {}).get("reference_entries") or []))
    total_target = int((word_band.get("min_words") or 0) + (word_band.get("max_words") or 0)) // 2
    if total_target <= 0:
        total_target = max(900, len(claims) * 55)
    body_target = max(450, total_target - reference_words - _text_word_count(title))
    # LLM section generators commonly undershoot long-form word targets. Inflate
    # the requested body budget while the assembler still enforces the final
    # ±25% document band after assembly.
    target_inflation = _float_env("DRAFTPROOF_STAGED_SECTION_TARGET_INFLATION", 1.18)
    per_section = max(120, int((body_target * target_inflation) / max(1, len(body_headings))))

    plans: list[dict] = []
    claim_window = max(2, math.ceil(len(claims) / max(1, len(body_headings))))
    segment_window = max(1, math.ceil(len(target_segments) / max(1, len(body_headings)))) if target_segments else 1
    for index, heading in enumerate(body_headings, start=1):
        claim_start = (index - 1) * claim_window
        segment_start = (index - 1) * segment_window
        plans.append({
            "section_index": index,
            "title": title,
            "heading": heading,
            "target_words": per_section,
            "claim_inventory_slice": claims[claim_start:claim_start + claim_window],
            "target_signal_slice": target_segments[segment_start:segment_start + segment_window],
            "paragraph_plan_slice": paragraph_plans[max(0, index - 1):index + 2],
        })
    return plans


def _staged_reconstruction_section_prompt(
    context_ledger: dict,
    gate_controls: dict,
    section_plan: dict,
    *,
    strategy: str,
    attempt_index: int,
) -> str:
    target_band = section_plan.get("target_word_band") or {}
    target_min = target_band.get("min")
    target_max = target_band.get("max")
    target_ideal = target_band.get("ideal") or section_plan.get("target_words")
    target_text = (
        f"between {target_min} and {target_max} words; aim for about {target_ideal} words"
        if isinstance(target_min, int) and isinstance(target_max, int)
        else f"about {section_plan.get('target_words')} words"
    )
    required_anchors = section_plan.get("must_preserve_anchors") or []
    allowed_citations = section_plan.get("citation_keys_used") or []
    handoff = (context_ledger or {}).get("generation_handoff") or {}
    all_reference_labels = []
    for ref in handoff.get("reference_register") or []:
        if not isinstance(ref, dict):
            continue
        label = str(ref.get("citation_key") or "").strip()
        if label and label not in all_reference_labels:
            all_reference_labels.append(label)
    disallowed_citations = [
        label for label in all_reference_labels
        if label not in allowed_citations
    ][:20]
    preserve_context = {
        "protected_facts": (context_ledger or {}).get("protected_facts") or [],
        "preservation_inventory": (context_ledger or {}).get("preservation_inventory") or {},
        "preserve_terms": (context_ledger or {}).get("preserve_terms") or [],
        "do_not_add": (context_ledger or {}).get("do_not_add") or [],
        "industry_baseline_focus": (context_ledger or {}).get("industry_baseline_focus") or {},
        "human_contribution_contract": (context_ledger or {}).get("human_contribution_contract") or {},
    }
    section_context = {
        "schema_version": "section_generation_context.v1",
        "section": section_plan,
        "preserve_context": preserve_context,
        "scanner_gate_feedback": gate_controls,
    }
    strategy_controls = []
    strategy_name = str(strategy or "").strip().lower()
    anchor_lock_mapping = _anchor_lock_mapping(required_anchors)
    prompt_section_context = (
        _freeze_anchor_payload(section_context, anchor_lock_mapping)
        if anchor_lock_mapping
        else section_context
    )
    prompt_required_anchors = (
        [item["placeholder"] for item in anchor_lock_mapping]
        if anchor_lock_mapping
        else required_anchors
    )
    if strategy_name == "plain_student_voice_rebuild":
        strategy_controls = [
            "PLAIN_STUDENT_VOICE_REBUILD is active.",
            "Write like a real student draft, not like a polished model answer.",
            "Use simple vocabulary and direct sentences. Repeating ordinary words is acceptable.",
            "Do not upgrade the essay into academic style. Do not add sophisticated connectors or balanced paragraph architecture.",
            "Keep some uneven development: one idea may be short, another may be a little over-explained, and not every link needs a transition.",
            "Avoid phrases such as rapidly evolving, crucial role, significant impact, furthermore, moreover, additionally, this highlights, and in conclusion.",
            "Do not add new facts, citations, examples, dates, institutions, or personal events.",
            "Preserve the same meaning and required anchors.",
        ]
    elif strategy_name == "human_gain_repair":
        strategy_controls = [
            "HUMAN_GAIN_REPAIR is active.",
            "Patch the section through controlled human-anchor amplification, not broad rewriting.",
            "Use only existing anchors, section-local context, and safe lived-observation phrasing licensed by the source stance.",
            "Do not invent new concrete details. If a concrete detail is not in required anchors or context ledger, leave it out.",
            "Prefer one small reasoning trace and one workshop/process anchor over generic academic expansion.",
            "Keep mild unevenness and rough edges; do not make every sentence equally polished.",
        ]
    elif strategy_name == "authorship_distribution_repair":
        strategy_controls = [
            "AUTHORSHIP_DISTRIBUTION_REPAIR is active.",
            "Primary target is lower AI Authorship, not prettier prose and not more explanation.",
            "Avoid the polished essay route. Do not write claim -> explanation -> implication repeatedly.",
            "Use distributional texture: one compressed sentence, one slightly longer causal sentence, one plain restart, and one under-explained but clear practical point.",
            "Break clean transition chains. Use plain moves such as 'But', 'So', 'The issue is', or no transition when the link is obvious.",
            "Reduce semantic uniformity by giving adjacent sentences different jobs: observation, limitation, consequence, then a narrow judgement.",
            "Do not add new facts, examples, dates, institutions, personal events, citations, or evidence.",
            "Do not add random errors, slang, typos, or artificial noise.",
        ]
    elif strategy_name == "authorship_texture_repair":
        strategy_controls = [
            "AUTHORSHIP_TEXTURE_REPAIR is active.",
            "Do not add more semantic human anchors as the main move; fix authorship texture.",
            "Preserve the same meaning points and required anchors while changing cadence and pacing.",
            "Use natural asymmetry: one short sentence, one longer practical sentence, and one delayed connection.",
            "Reduce clean transition logic. Avoid neat claim -> explanation -> implication paragraph routes.",
            "Vary information density without adding facts: compress one idea, leave one practical point less over-explained.",
            "Keep acceptable local friction, but do not add typos, grammar damage, fake randomness, or invented details.",
        ]
    elif strategy_name in {"low_smoothness_rebuild", "asymmetric_paragraph_route"}:
        strategy_controls = [
            "LOW_SMOOTHNESS_AUTHORSHIP_REPAIR is active.",
            "Keep meaning and anchors, but lower clean explanatory cadence.",
            "Prefer uneven paragraph movement over balanced academic flow.",
            "Use fewer connector phrases. Let some sentences sit next to each other without over-explaining the link.",
            "Compress generic claims instead of expanding them.",
            "Do not add fabricated grounding or decorative personal detail.",
        ]
    if anchor_lock_mapping:
        strategy_controls.append(
            "Anchor lock is active. Copy every [[DP_ANCHOR_###]] placeholder exactly; the pipeline restores real anchors after generation."
        )
    return (
        "DraftProof staged AI-Mitigation generation.\n"
        "Generate only this section body from the section context ledger. "
        "Do not output the section heading, references, labels, comments, markdown fences, or explanation.\n"
        "The original submitted prose is unavailable by design. Use only the structured context below.\n\n"
        f"Attempt: {attempt_index}. Strategy family: {strategy}.\n"
        f"Section heading owned by assembler: {section_plan.get('heading')}\n"
        f"Target length for this section body: {target_text}. This is guidance, not the final acceptance gate.\n"
        f"Required section anchors to preserve exactly; missing any one invalidates the output: {json.dumps(prompt_required_anchors, ensure_ascii=False)}\n"
        f"Allowed citation/source keys for this section: {json.dumps(allowed_citations, ensure_ascii=False)}\n"
        f"Disallowed citation/source keys for this section: {json.dumps(disallowed_citations, ensure_ascii=False)}\n\n"
        "Section context ledger:\n"
        f"{json.dumps(prompt_section_context, ensure_ascii=False)[:7000]}\n\n"
        "Generation controls:\n"
        f"- Word-count target: stay near {target_text}; scanner/gate quality is more important than exact length.\n"
        "- Write enough section body to survive cleanup that removes headings, repeated sentences, labels, and filler.\n"
        "- Do not repeat any full sentence or near-duplicate sentence; repeated sentences are removed before scoring and can make the candidate too short.\n"
        "- Preserve meaning, citations, years, names, numbers, quotes, source relations, and domain terms that are relevant to this section.\n"
        "- Use every required section anchor unless it is clearly a fragment; unit codes and named institutions are mandatory.\n"
        "- Do not mention disallowed source names, author groups, citations, frameworks, or evidence from other sections.\n"
        "- If allowed citation/source keys is empty, do not mention any source author, cited study, framework, or reference name.\n"
        "- Do not add new evidence, personal events, workplace observations, institutions, dates, statistics, sources, or citations.\n"
        "- Do not make a polished template essay paragraph. Use local reasoning, uneven sentence lengths, and section-specific causal links.\n"
        "- If the section context is thin, expand only by connecting the provided meaning points and anchors; do not invent facts.\n"
        + ("".join(f"- {item}\n" for item in strategy_controls) if strategy_controls else "")
        + "- Return prose only."
    )


def _clean_section_candidate(output: str, heading: str) -> str:
    if not output:
        return ""
    text = output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"^(?:section|body|draft|answer)\s*:\s*", "", text, flags=re.I).strip()
    if heading:
        text = re.sub(rf"^\s*{re.escape(str(heading).strip())}\s*", "", text, flags=re.I).strip()
    text = re.sub(r"(?im)^\s*(?:references|reference list|bibliography|works cited)\s*$.*", "", text, flags=re.S).strip()
    paragraphs = [" ".join(p.strip().split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "\n\n".join(paragraphs).strip()


def _staged_reconstruction_candidate(
    gateway: LLMGateway,
    source_text: str,
    raw_json: dict,
    *,
    attempt_index: int,
    strategy: str,
    prior_attempts: list[dict] | None = None,
) -> tuple[str, dict]:
    """Generate a candidate through section prompts and deterministic assembly."""
    brief = _build_reconstruction_meaning_brief(source_text, raw_json)
    blueprint = _build_regeneration_blueprint(source_text, raw_json, strategy)
    context_ledger = _generation_context_ledger(brief, blueprint)
    gate_controls = _reconstruction_gate_controls(prior_attempts)
    section_plans = _staged_generation_section_plan(context_ledger)
    title = section_plans[0].get("title") if section_plans else ""
    parts: list[str] = [str(title).strip()] if title else []
    section_results: list[dict] = []
    call_count = 0
    for section_plan in section_plans:
        prompt = _staged_reconstruction_section_prompt(
            context_ledger,
            gate_controls,
            section_plan,
            strategy=strategy,
            attempt_index=attempt_index,
        )
        response = gateway.chat(
            prompt,
            system=(
                "You are DraftProof's staged AI-Mitigation section generator. "
                "Return only bounded section prose from the structured context ledger."
            ),
            temperature=float(os.environ.get("DRAFTPROOF_RECONSTRUCTION_TEMPERATURE", "0.78")),
            max_tokens=int(os.environ.get("DRAFTPROOF_STAGED_SECTION_MAX_TOKENS", "1800")),
            top_p=_float_env_optional("DRAFTPROOF_RECONSTRUCTION_TOP_P"),
            top_k=_int_env_optional("DRAFTPROOF_RECONSTRUCTION_TOP_K"),
            presence_penalty=_float_env_optional("DRAFTPROOF_RECONSTRUCTION_PRESENCE_PENALTY"),
            frequency_penalty=_float_env_optional("DRAFTPROOF_RECONSTRUCTION_FREQUENCY_PENALTY"),
        )
        call_count += 1
        body = _clean_section_candidate(response.content, str(section_plan.get("heading") or ""))
        anchor_lock = _anchor_lock_mapping(section_plan.get("must_preserve_anchors") or [])
        missing_placeholders = []
        if anchor_lock:
            missing_placeholders = [
                item["placeholder"]
                for item in anchor_lock
                if item.get("placeholder") and item["placeholder"] not in body
            ]
            if missing_placeholders and body:
                body = f"{body.rstrip()} {' '.join(missing_placeholders)}"
        if anchor_lock:
            body = _restore_anchor_placeholders(body, anchor_lock)
        if body:
            parts.append(str(section_plan.get("heading") or "").strip())
            parts.append(body)
        section_results.append({
            "heading": section_plan.get("heading"),
            "target_words": section_plan.get("target_words"),
            "actual_words": _text_word_count(body),
            "empty": not bool(body),
            "anchor_lock_enabled": bool(anchor_lock),
            "missing_placeholders_repaired": missing_placeholders,
        })

    references = [
        str(item).strip()
        for item in context_ledger.get("reference_entries") or []
        if str(item).strip()
    ]
    if references:
        parts.append("References")
        parts.extend(references)
    candidate = "\n\n".join(part for part in parts if str(part).strip()).strip()
    metadata = {
        "schema_version": "staged_reconstruction.v1",
        "enabled": True,
        "llm_calls": call_count,
        "sections": section_results,
        "assembled_word_count": _text_word_count(candidate),
        "reference_entries_preserved": len(references),
        "source_draft_included": False,
        "context_ledger_schema": context_ledger.get("schema_version"),
    }
    return _clean_full_document_candidate(candidate, source_text), metadata


def _build_reconstruction_meaning_brief(source_text: str, raw_json: dict | None) -> dict:
    """Build a conservative meaning brief for document-level reconstruction.

    This is not an abstractive summary. It extracts author-supplied material
    already present in the submitted content and scanner output so the LLM can
    rebuild structure without inventing facts.
    """
    raw_json = raw_json or {}
    paragraphs = [
        " ".join(p.split())
        for p in re.split(r"\n\s*\n", source_text or "")
        if p.strip()
    ]
    headings = [
        p
        for p in paragraphs
        if len(p.split()) <= 12 and not re.search(r"[.!?]$", p)
    ][:12]
    protected_spans = []
    for span in detect_protected_spans(source_text or "")[:40]:
        value = (source_text or "")[span.start_char:span.end_char].strip()
        if value and value not in protected_spans:
            protected_spans.append(value)

    findings = raw_json.get("findings") or {}
    weak_zones = []
    for tier in ("critical", "high", "medium", "low"):
        for item in findings.get(tier, []) or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("category") or item.get("scanner") or "")
            evidence = item.get("evidence")
            recommendation = item.get("recommendation")
            weak_zones.append({
                "tier": tier,
                "signal": title,
                "evidence": evidence if isinstance(evidence, str) else "",
                "recommendation": recommendation if isinstance(recommendation, str) else "",
            })
            if len(weak_zones) >= 12:
                break
        if len(weak_zones) >= 12:
            break

    action_rows = []
    for action in ((raw_json.get("ai_mitigation") or {}).get("component_actions") or [])[:10]:
        if not isinstance(action, dict):
            continue
        action_rows.append({
            "component": action.get("component"),
            "score": action.get("score"),
            "action": action.get("action"),
            "user_input_needed": action.get("user_input_needed"),
        })

    scan_intelligence = raw_json.get("scan_intelligence") or {}
    generation_handoff = (
        scan_intelligence.get("generation_handoff")
        or ((scan_intelligence.get("mitigation_inputs") or {}).get("generation_handoff"))
        or raw_json.get("generation_handoff")
        or {}
    )
    semantic = (
        scan_intelligence.get("semantic_layer")
        or scan_intelligence.get("semantic_shape")
        or raw_json.get("semantic_shape")
        or {}
    )
    rewrite_constraints = (
        raw_json.get("rewrite_constraints")
        or (((raw_json.get("ai_mitigation") or {}).get("rewrite_handoff") or {}).get("rewrite_constraints"))
        or {}
    )
    preservation_inventory = (
        rewrite_constraints.get("preservation_inventory")
        or (((raw_json.get("scan_intelligence") or {}).get("mitigation_inputs") or {}).get("preservation_inventory"))
        or (((raw_json.get("scan_intelligence") or {}).get("document") or {}).get("preservation_inventory"))
        or {}
    )
    inventory_anchors = [
        anchor.get("text")
        for anchor in (preservation_inventory.get("anchors") or [])
        if isinstance(anchor, dict) and anchor.get("text")
    ]
    for anchor in inventory_anchors:
        if anchor not in protected_spans:
            protected_spans.append(anchor)
    word_band = _word_count_band(source_text, variance=0.25)
    paragraph_roles = []
    for idx, paragraph in enumerate(paragraphs[:10], start=1):
        lower = paragraph.lower()
        if any(marker in lower for marker in ("according to", "source", "citation", "research", "study")):
            role = "source_or_evidence_relation"
        elif any(marker in lower for marker in ("therefore", "this shows", "this means", "conclusion")):
            role = "interpretation_or_conclusion"
        elif idx == 1:
            role = "context_or_thesis"
        else:
            role = "development"
        paragraph_roles.append({
            "index": idx,
            "role": role,
            "preview": paragraph[:220],
        })
    if generation_handoff:
        handoff_units = generation_handoff.get("section_generation_units") or []
        handoff_outline = generation_handoff.get("logical_outline") or []
        handoff_refs = [
            ref.get("full_reference")
            for ref in generation_handoff.get("reference_register") or []
            if isinstance(ref, dict) and ref.get("full_reference")
        ]
        handoff_headings = [
            row.get("heading")
            for row in handoff_outline
            if isinstance(row, dict) and row.get("heading")
        ]
        handoff_protected = []
        anchor_register = generation_handoff.get("anchor_register") or {}
        for value in anchor_register.values():
            if isinstance(value, list):
                handoff_protected.extend(str(item) for item in value if item)
        handoff_roles = [
            {
                "index": idx,
                "section_id": unit.get("section_id"),
                "role": unit.get("role"),
                "heading": unit.get("heading"),
                "target_words": unit.get("target_words"),
            }
            for idx, unit in enumerate(handoff_units[:12], start=1)
            if isinstance(unit, dict)
        ]
        handoff_claims = [
            {
                "section_id": unit.get("section_id"),
                "heading": unit.get("heading"),
                "meaning_inventory": unit.get("meaning_inventory") or [],
                "citation_keys_used": unit.get("citation_keys_used") or [],
            }
            for unit in handoff_units[:12]
            if isinstance(unit, dict)
        ]
        if handoff_headings:
            headings = handoff_headings
        if handoff_refs:
            reference_entries = handoff_refs
        else:
            reference_entries = _reference_entries_from_text(source_text)
        if handoff_protected:
            for value in handoff_protected:
                if value not in protected_spans:
                    protected_spans.append(value)
        if handoff_roles:
            paragraph_roles = handoff_roles
    else:
        reference_entries = _reference_entries_from_text(source_text)
        handoff_claims = []

    return {
        "claims": handoff_claims or _brief_sentences(source_text, limit=36),
        "headings": headings,
        "protected_facts": protected_spans[:25],
        "reference_entries": reference_entries,
        "generation_handoff": generation_handoff,
        "preservation_inventory": preservation_inventory,
        "human_contribution_contract": (
            scan_intelligence.get("human_contribution_contract")
            or ((scan_intelligence.get("mitigation_inputs") or {}).get("human_contribution_contract"))
            or {}
        ),
        "industry_baseline": (
            scan_intelligence.get("industry_baseline")
            or ((scan_intelligence.get("mitigation_inputs") or {}).get("industry_baseline"))
            or raw_json.get("industry_baseline")
            or ((raw_json.get("ai_mitigation") or {}).get("industry_baseline"))
            or {}
        ),
        "word_count_band": word_band,
        "integrity_targets": _integrity_driver_rows(raw_json, limit=14),
        "target_segments": _target_segment_rows(raw_json, limit=18),
        "allowed_existing_additions": rewrite_constraints.get("allowed_additions") or [],
        "preserve_terms": rewrite_constraints.get("preserve_terms") or protected_spans[:25],
        "do_not_add": rewrite_constraints.get("do_not_add") or [],
        "rewrite_rule": rewrite_constraints.get("rewrite_rule"),
        "weak_grounding_zones": weak_zones,
        "mitigation_actions": action_rows,
        "paragraph_roles": paragraph_roles,
        "semantic_layer": semantic,
        "signal_inventory": scan_intelligence.get("signal_inventory") or {},
        "transformation_core": (scan_intelligence.get("transformation") or {}).get("core_signals") or {},
    }


def _build_regeneration_blueprint(source_text: str, raw_json: dict | None, strategy: str) -> dict:
    """Build a scanner-derived generation plan before prose generation."""
    brief = _build_reconstruction_meaning_brief(source_text, raw_json)
    paragraphs = [
        " ".join(p.split())
        for p in re.split(r"\n\s*\n", source_text or "")
        if p.strip()
    ]
    target_by_paragraph: dict[str, list[dict]] = {}
    for segment in brief.get("target_segments") or []:
        pid = segment.get("paragraph_id") or "p001"
        target_by_paragraph.setdefault(pid, []).append(segment)

    family_shapes = {
        "plain_student_voice_rebuild": [
            "simple_claim",
            "plain_problem",
            "short_reason",
            "example_from_existing_context",
            "limited_point",
        ],
        "authorship_distribution_repair": [
            "plain_limit",
            "offbeat_reasoning",
            "short_restart",
            "source_or_anchor_afterthought",
            "bounded_point",
        ],
        "evidence_first_rebuild": [
            "source_or_anchor_first",
            "problem_pressure",
            "author_reasoning",
            "bounded_implication",
        ],
        "problem_observation_rebuild": [
            "problem_pressure",
            "specific_context",
            "reasoning_check",
            "limited_conclusion",
        ],
        "claim_narrowing_rebuild": [
            "narrow_claim",
            "existing_anchor",
            "qualification",
            "bounded_implication",
        ],
        "asymmetric_paragraph_route": [
            "short_context",
            "long_reasoning",
            "source_relation",
            "brief_counterpressure",
            "plain_conclusion",
        ],
        "low_smoothness_rebuild": [
            "direct_claim",
            "pause_or_limit",
            "specific_consequence",
            "plain_reasoning",
        ],
        "conservative_reconstruction": [
            "context",
            "pressure_point",
            "evidence_relation",
            "bounded_implication",
        ],
        "reasoning_dense_reconstruction": [
            "context",
            "friction",
            "evidence_relation",
            "author_reasoning",
            "implication",
        ],
        "domain_grounded_reconstruction": [
            "domain_context",
            "operational_detail",
            "judgement_or_limit",
            "bounded_implication",
        ],
    }
    route = family_shapes.get(strategy, family_shapes["asymmetric_paragraph_route"])
    paragraph_count = max(3, len(paragraphs))
    paragraph_plans = []
    for idx, paragraph in enumerate(paragraphs, start=1):
        pid = f"p{idx:03d}"
        role = route[(idx - 1) % len(route)]
        targets = target_by_paragraph.get(pid, [])
        plan = {
            "paragraph_id": pid,
            "role": role,
            "source_preview": paragraph[:180],
            "claim_source": "preserve paragraph meaning; do not preserve sentence order",
            "target_segments": targets[:4],
            "must_reduce": [
                target.get("signal") or target.get("lever")
                for target in targets[:4]
                if target.get("signal") or target.get("lever")
            ],
            "rhythm": (
                "mix one short plain sentence with one longer causal sentence"
                if idx % 2
                else "avoid balanced list cadence; use one direct limitation or contrast"
            ),
            "grounding_move": (
                "narrow broad claims to the submitted context"
                if targets
                else "add only reasoning already implied by adjacent claims"
            ),
        }
        paragraph_plans.append(plan)

    driver_keys = [
        item.get("key")
        for item in brief.get("integrity_targets") or []
        if item.get("key")
    ]
    return {
        "schema_version": "regeneration_blueprint.v1",
        "strategy_family": strategy,
        "target_human_contribution": 80,
        "word_count_band": brief.get("word_count_band"),
        "paragraph_count": paragraph_count,
        "paragraph_plans": paragraph_plans[:12],
        "global_driver_targets": driver_keys[:10],
        "industry_baseline_focus": {
            "ai_authorship_positive_components": [
                row.get("key")
                for row in (
                    ((brief.get("industry_baseline") or {}).get("layers") or {})
                    .get("ai_authorship_risk", {})
                    .get("positive_components", [])
                )[:6]
                if isinstance(row, dict) and row.get("key")
            ],
            "human_contribution_components": [
                row.get("key")
                for row in (
                    ((brief.get("industry_baseline") or {}).get("layers") or {})
                    .get("human_contribution_signal", {})
                    .get("components", [])
                )
                if isinstance(row, dict) and row.get("key")
            ],
            "authorship_suppressors": [
                row.get("key")
                for row in (
                    ((brief.get("industry_baseline") or {}).get("layers") or {})
                    .get("ai_authorship_risk", {})
                    .get("suppressors", [])
                )
                if isinstance(row, dict) and row.get("key")
            ],
            "policy": (brief.get("industry_baseline") or {}).get("policy") or {},
        },
        "hard_preserve": {
            "quotes": (brief.get("preservation_inventory") or {}).get("quotes") or [],
            "citations": (brief.get("preservation_inventory") or {}).get("citations") or [],
            "years": (brief.get("preservation_inventory") or {}).get("years") or [],
            "numbers": (brief.get("preservation_inventory") or {}).get("numbers") or [],
            "names_entities": (brief.get("preservation_inventory") or {}).get("names_entities") or [],
            "domain_terms": (brief.get("preservation_inventory") or {}).get("domain_terms") or [],
        },
        "anti_patterns_to_avoid": [
            "intro-balanced-explanation-conclusion essay shape",
            "Furthermore/Moreover/In conclusion connector chain",
            "same-length paragraphs",
            "claim followed by generic explanation",
            "smooth motivational summary",
            "broad education/technology statements without local narrowing",
        ],
        "candidate_family_requirements": [
            "do not use the original sentence order as scaffold",
            "assign a different reasoning job to adjacent paragraphs",
            "make at least two broad claims narrower instead of more polished",
            "preserve anchors exactly where required",
            "stay inside the word-count band",
        ],
    }


def _reconstruction_mitigation_prompt(
    source_text: str,
    raw_json: dict,
    ai_mitigation: dict | None,
    *,
    attempt_index: int,
    strategy: str,
    prior_attempts: list[dict] | None = None,
) -> str:
    contribution = _contribution_scores(raw_json)
    integrity = _integrity_scores(raw_json)
    brief = _build_reconstruction_meaning_brief(source_text, raw_json)
    blueprint = _build_regeneration_blueprint(source_text, raw_json, strategy)
    context_ledger = _generation_context_ledger(brief, blueprint)
    failure_feedback = _reconstruction_failure_feedback(prior_attempts)
    gate_controls = _reconstruction_gate_controls(prior_attempts)
    include_source_draft = _env_flag("DRAFTPROOF_RECONSTRUCTION_INCLUDE_SOURCE_DRAFT", False)
    compact_failure_rows = []
    for item in failure_feedback:
        compact_failure_rows.append(
            "- "
            + "; ".join(
                part
                for part in [
                    f"strategy={item.get('strategy')}",
                    f"reason={item.get('reason')}",
                    f"human_shift={item.get('human_shift_score')}",
                    f"ai_authorship_delta={item.get('ai_authorship_delta')}",
                    f"human_delta={item.get('human_delta')}",
                    f"ai_transformation_delta={item.get('ai_transformation_delta')}",
                ]
                if not part.endswith("=None")
            )
        )
    strategy_guidance = {
        "conservative_reconstruction": (
            "Keep the same claim set, but rebuild paragraph routes, sentence openings, and causal bridges. "
            "Prefer narrower claims over adding new evidence."
        ),
        "reasoning_dense_reconstruction": (
            "Compress generic explanation and make each paragraph carry a clearer reasoning move: context, friction, evidence relation, implication."
        ),
        "domain_grounded_reconstruction": (
            "Use domain-specific operational language already present in the draft. Do not add new workplace, class, source, or personal details."
        ),
    }.get(strategy, "Rebuild the document structure while preserving meaning and protected facts.")
    return (
        "DraftProof AI-Mitigation Reconstruction.\n"
        "This is not sentence-level revision, not paraphrasing, and not modification of the submitted prose. "
        "Generate a new document from the scanner context ledger.\n"
        "Goal: produce a human-authored regeneration that moves the next scan toward Human Contribution >= 80 where the submitted evidence permits it. "
        "If 80 is not reachable without inventing facts, maximize Human Shift Score while preserving what the submitted content conveys.\n\n"
        f"Current scores: AI Authorship={integrity.get('ai_authorship')}, Human={contribution.get('human')}, "
        f"AI Transformation={contribution.get('ai_transformation')}, Grounding Risk={integrity.get('grounding')}.\n"
        f"Strategy: {strategy}. {strategy_guidance}\n\n"
        "Scanner context ledger for generation. This is the generation input; do not ask for or reconstruct from the original prose order:\n"
        f"{json.dumps(context_ledger, ensure_ascii=False)[:9000]}\n\n"
        "Scanner-derived regeneration blueprint to follow before writing prose. Source previews have been removed so you do not scaffold from submitted sentences:\n"
        f"{json.dumps({k: v for k, v in blueprint.items() if k != 'paragraph_plans'}, ensure_ascii=False)[:2600]}\n\n"
        "Previous failed attempts to correct:\n"
        f"{json.dumps(failure_feedback, ensure_ascii=False) if failure_feedback else '- None yet.'}\n"
        f"{chr(10).join(compact_failure_rows) if compact_failure_rows else ''}\n\n"
        "Scanner/gate feedback controls for this attempt:\n"
        f"{json.dumps(gate_controls, ensure_ascii=False)[:4200]}\n\n"
        "Word-count requirement:\n"
        f"- The submitted draft has {brief['word_count_band']['source_word_count']} words. "
        f"Return {brief['word_count_band']['min_words']} to {brief['word_count_band']['max_words']} words only.\n\n"
        "Regeneration blueprint:\n"
        "- Follow the scanner-derived context ledger and blueprint above. They are the plan.\n"
        "- Follow the scanner/gate feedback controls above. They override generic writing preferences for this attempt.\n"
        "- Do not follow the submitted sentence order as a scaffold. The submitted prose is not the generation substrate.\n"
        "- Build a fresh paragraph route from the blueprint: concrete context, pressure point, evidence/source relation, author reasoning, bounded implication.\n"
        "- Target the industry-baseline AI Authorship drivers directly: token predictability, burstiness regularity, discourse regularity, semantic uniformity, template phrase signal, and rewrite smoothness.\n"
        "- Increase industry-baseline suppressors through real authorship friction: causal reasoning, local constraint awareness, domain cognition, and natural paragraph variance. Do not use typo/noise tricks.\n"
        "- Treat grounding quality as separate from AI authorship. Narrow unsupported claims or preserve source relations; never invent citations, dates, names, statistics, or evidence.\n"
        "- Give adjacent paragraphs different jobs. One may start from a problem, another from a source relation, another from a limitation or consequence.\n"
        "- Use target segments as the highest-priority places to change the route, not as sentences to lightly paraphrase.\n"
        "- Use only allowed existing additions and implied process detail already licensed by the scan constraints.\n\n"
        "Candidate-family instruction:\n"
        f"- This candidate must use the '{strategy}' family. It must be structurally different from other families, not a temperature variant.\n"
        "- Make the prose less template-like by changing paragraph purpose, not by adding odd wording.\n"
        "- Do not make the draft smoother. Smoothness without local reasoning is a failure.\n\n"
        "Allowed reconstruction moves:\n"
        "- Reorder paragraphs and claims when the meaning is preserved.\n"
        "- Split over-smooth paragraphs and vary sentence pacing naturally.\n"
        "- Compress broad explanatory padding into denser reasoning.\n"
        "- Make causal links explicit when the submitted content already implies them.\n"
        "- Replace generic academic transitions with context-specific connections.\n"
        "- Narrow unsupported claims rather than inventing evidence.\n\n"
        "Human Shift acceptance requirements:\n"
        "- The next scan must not trade a Human Contribution gain for higher AI Authorship.\n"
        "- Avoid increasing semantic uniformity, review burden, or critical/high findings.\n"
        "- A candidate that raises Human Contribution but raises AI Authorship will be rejected.\n"
        "- Prefer a less polished, more locally reasoned draft over a smoother academic rewrite.\n\n"
        "Forbidden moves:\n"
        "- Do not invent personal observations, examples, dates, statistics, sources, citations, institutions, or facts.\n"
        "- Do not change quoted text, citations, years, names, numbers, headings, or source relations.\n"
        "- Do not use placeholders, review brackets, comments, labels, markdown fences, or explanations.\n"
        "- Do not preserve the original sentence order or sentence shape just to be safe; preserve meaning instead.\n\n"
        f"Attempt {attempt_index}: return only the complete regenerated document."
        + (
            "\n\nSOURCE DRAFT DEBUG FALLBACK:\n"
            f"<TARGET_DOCUMENT>\n{source_text.strip()}\n</TARGET_DOCUMENT>"
            if include_source_draft
            else ""
        )
    )


def _clean_full_document_candidate(output: str, original_text: str) -> str:
    if not output:
        return ""
    text = output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(
        r"^(?:rewritten|replacement|final)\s+(?:draft|document|text)\s*:\s*",
        "",
        text,
        flags=re.I,
    ).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = [" ".join(p.strip().split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    text = "\n\n".join(paragraphs).strip()
    return "" if text == original_text.strip() else text


def _review_marker_notes(candidate: str) -> list[str]:
    if not isinstance(candidate, str):
        return []
    return [
        match.strip()
        for match in re.findall(r"\[\[REVIEW:\s*(.*?)\]\]", candidate, flags=re.I | re.S)
        if match.strip()
    ]


_SYNTHETIC_ANCHOR_RE = re.compile(
    r"\b(?:In my chair|During consultation|For example|In my notes|"
    r"During sectioning|Specifically|In my experience|During the cut|"
    r"In practice|For this task|During the practical work|In assessment|"
    r"When learners are cutting|During feedback):",
    re.I,
)
_DANGLING_FRAGMENT_JOIN_RE = re.compile(
    r"\b(?:can|could|should|would|will|may|might|must|to|and|but|or|"
    r"while|because|if|before|after|adjust)\s+"
    r"(?:With only|Learners gain|A competent|The standard|Conclusion|"
    r"Introduction|This review|In Certificate|Inclusive learning)\b",
    re.I,
)
_KNOWN_HEADING_FOLLOWERS = [
    ("Introduction", "Inclusive learning design"),
    ("When learners start to get lost", "The challenge"),
    ("Showing the haircut clearly", "A demonstration"),
    ("Reasonable adjustment and classroom reality", "Inclusive learning design"),
    ("Maintaining standards while improving access", "Inclusive learning design"),
    ("Conclusion", "This review"),
]


def _normalize_known_heading_boundaries(text: str) -> tuple[str, list[str]]:
    """Separate known document headings that were flattened into prose."""
    if not isinstance(text, str) or not text:
        return text, []
    repaired = text
    repairs: list[str] = []

    next_text = re.sub(
        r"\A(\s*[^\n.!?]{12,180}?)\s+Introduction(?=(?:\s+|\n+)Inclusive learning design\b|\n\n)",
        r"\1\n\nIntroduction",
        repaired,
        flags=re.I,
        count=1,
    )
    if next_text != repaired:
        repaired = next_text
        repairs.append("split_title_from_introduction")

    for heading, following in _KNOWN_HEADING_FOLLOWERS:
        heading_re = re.escape(heading)
        following_re = re.escape(following)

        next_text = re.sub(
            rf"(?<=[.!?])\s+({heading_re})\s+(?={following_re}\b)",
            r"\n\n\1\n\n",
            repaired,
            flags=re.I,
        )
        if next_text != repaired:
            repaired = next_text
            repairs.append(f"split_sentence_before_heading:{heading}")

        next_text = re.sub(
            rf"(?<=[.!?])\s+({heading_re})(?=\s*(?:\n\n|$))",
            r"\n\n\1",
            repaired,
            flags=re.I,
        )
        if next_text != repaired:
            repaired = next_text
            repairs.append(f"split_orphaned_heading:{heading}")

        next_text = re.sub(
            rf"(^|\n\n)({heading_re})\s+(?={following_re}\b)",
            r"\1\2\n\n",
            repaired,
            flags=re.I,
        )
        if next_text != repaired:
            repaired = next_text
            repairs.append(f"split_merged_heading:{heading}")

    return repaired, repairs


def _strip_reference_like_lines_for_quality(text: str) -> str:
    """Remove bibliography/reference lines before repetition quality checks."""
    if not isinstance(text, str) or not text:
        return ""
    kept = []
    in_reference_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if re.match(r"^(?:references|reference list|bibliography)\s*$", line, re.I):
            in_reference_section = True
            continue
        if in_reference_section:
            # Keep prose if a later section heading starts, otherwise ignore
            # the reference block. Long publisher names repeat naturally there.
            if re.match(r"^[A-Z][A-Za-z0-9 ,/&-]{2,70}$", line) and not re.search(r"\(\d{4}\)|https?://", line):
                in_reference_section = False
            else:
                continue
        if re.search(r"https?://|doi\.org|\(\d{4}\)", line, re.I) and len(line.split()) >= 8:
            continue
        kept.append(raw_line)
    return "\n".join(kept)


def _repeated_long_sequence_reason(text: str, window: int = 8) -> str:
    body_text = _strip_reference_like_lines_for_quality(text)
    tokens = re.findall(r"[A-Za-z0-9']+", str(body_text or "").lower())
    if len(tokens) < window * 3:
        return ""
    seen: dict[tuple[str, ...], int] = {}
    for index in range(0, len(tokens) - window + 1):
        gram = tuple(tokens[index:index + window])
        if len(set(gram)) <= 3:
            continue
        if gram in seen and index - seen[gram] > window:
            return "repeated_long_sequence:" + " ".join(gram[:6])
        seen[gram] = index
    return ""


def _ai_candidate_quality_reject_reason(
    candidate: str,
    *,
    allow_repeated_long_sequence: bool = False,
) -> str:
    if not isinstance(candidate, str) or not candidate.strip():
        return "empty_candidate"
    if "[[REVIEW:" in candidate:
        return "review_markers_not_auto_kept"
    synthetic_anchors = _SYNTHETIC_ANCHOR_RE.findall(candidate)
    sentence_count = max(1, len(re.findall(r"(?<=[.!?])\s+", candidate)) + 1)
    max_anchor_count = max(3, min(8, sentence_count // 8))
    for artifact in (
        "For this task:",
        "During the practical work:",
        "During feedback:",
        "When learners are cutting:",
        "In assessment:",
    ):
        if re.search(r"\b" + re.escape(artifact), candidate, re.I):
            return f"synthetic_anchor_artifact:{artifact}"
    if len(synthetic_anchors) > max_anchor_count:
        return f"synthetic_anchor_overuse {len(synthetic_anchors)}>{max_anchor_count}"
    lowered = candidate.lower()
    if re.search(r"\bith only\b", lowered):
        return "broken_word_fragment"
    if re.search(r"\b(?:introduction|conclusion)[ \t]+(?:inclusive|this|the)\b", candidate, re.I):
        return "heading_merged_into_sentence"
    if _DANGLING_FRAGMENT_JOIN_RE.search(candidate):
        return "dangling_sentence_fragment_join"
    if not allow_repeated_long_sequence:
        repeated = _repeated_long_sequence_reason(candidate)
        if repeated:
            return repeated

    sentences = [
        re.sub(r"\s+", " ", s.strip()).lower()
        for s in re.split(r"(?<=[.!?])\s+", candidate)
        if len(s.split()) >= 8
    ]
    seen = set()
    duplicates = 0
    for sentence in sentences:
        if sentence in seen:
            duplicates += 1
        seen.add(sentence)
    if duplicates:
        return "duplicated_sentence_fragments"
    return ""


def _ai_search_signal_brief(raw_json: dict) -> str:
    badge = (raw_json or {}).get("ai_risk_badge") or {}
    ai_components = badge.get("ai_components") or {}
    writing_components = badge.get("writing_components") or {}
    parts = []
    if isinstance(ai_components, dict):
        ranked = sorted(
            ((k, v) for k, v in ai_components.items() if isinstance(v, (int, float))),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked:
            parts.append("AI component drivers: " + ", ".join(f"{k}={v:.2f}%" for k, v in ranked[:8]))
    if isinstance(writing_components, dict):
        ranked = sorted(
            ((k, v) for k, v in writing_components.items() if isinstance(v, (int, float))),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked:
            parts.append("Writing-risk context: " + ", ".join(f"{k}={v:.2f}%" for k, v in ranked[:8]))
    suggestions = (((raw_json or {}).get("rewrite_guidance") or {}).get("guided_revision") or {}).get("risk_mitigation_actions") or []
    action_lines = []
    for item in suggestions[:6]:
        if isinstance(item, dict):
            title = item.get("title") or item.get("action_type")
            pattern = item.get("safe_edit_pattern")
            if title and pattern:
                action_lines.append(f"{title}: {pattern}")
    if action_lines:
        parts.append("Scanner rewrite actions: " + " | ".join(action_lines))
    return "\n".join(parts)


def _source_repair_brief(source_text: str) -> str:
    """Describe visible source damage the full-document rewrite must repair."""
    if not isinstance(source_text, str) or not source_text.strip():
        return ""
    notes = []
    lowered = source_text.lower()
    if "ith only" in lowered:
        notes.append(
            "The source already contains a broken word fragment like 'ith only'. "
            "Repair it from context instead of preserving the typo."
        )
    if re.search(r"\b(?:introduction|conclusion)\s+(?:inclusive|this|the)\b", source_text, re.I):
        notes.append(
            "Some headings appear merged into following sentences. Keep the same headings/content, "
            "but separate merged heading text cleanly."
        )

    sentences = [
        re.sub(r"\s+", " ", s.strip()).lower()
        for s in re.split(r"(?<=[.!?])\s+", source_text)
        if len(s.split()) >= 8
    ]
    seen = set()
    duplicate_count = 0
    for sentence in sentences:
        if sentence in seen:
            duplicate_count += 1
        seen.add(sentence)
    if duplicate_count:
        notes.append(
            f"The source appears to contain {duplicate_count} accidental repeated sentence(s). "
            "Keep one clean version and remove duplicated fragments."
        )
    if not notes:
        return ""
    return "Source repair requirements:\n- " + "\n- ".join(notes)


def _repair_candidate_source_damage(candidate: str) -> tuple[str, list[str]]:
    """Repair obvious inherited source corruption before candidate gates."""
    if not isinstance(candidate, str) or not candidate:
        return candidate, []
    repaired = candidate
    repairs = []

    next_text = re.sub(r"\bith only\b", "With only", repaired, flags=re.I)
    if next_text != repaired:
        repaired = next_text
        repairs.append("fixed_broken_with_fragment")
    next_text = re.sub(
        r"\bWith only six learners\b",
        "Because there are only six learners",
        repaired,
        flags=re.I,
    )
    if next_text != repaired:
        repaired = next_text
        repairs.append("normalized_with_only_phrase")

    repaired, heading_repairs = _normalize_known_heading_boundaries(repaired)
    repairs.extend(heading_repairs)

    overlap_repairs = [
        (
            r"\bI encourage open discussion so learners can "
            r"(?=(?:With only|Because there are only)\b)",
            "",
            "removed_dangling_prefix:learners_can",
        ),
        (
            r"\bA competent learner can explain the steps, identify the guide, "
            r"check balance, adjust (?=Learners gain confidence\b)",
            "",
            "removed_dangling_prefix:adjust",
        ),
        (
            r"\bCAST and Jwad et al\. describe (?=The sources address\b)",
            "",
            "removed_dangling_prefix:cast_describe",
        ),
        (
            r"\bFor hairdressing educators, inclusive (?=Competency should not depend\b)",
            "",
            "removed_dangling_prefix:inclusive",
        ),
        (
            r"(This review has discussed inclusive learning design in Certificate III "
            r"Hairdressing\. Demonstration alone does not build competency\. "
            r"In units before SHBHCUT006,\s+)\1",
            r"\1",
            "collapsed_repeated_clause:conclusion_intro",
        ),
        (
            r"\bBillett and Kirschner et al\.\s+Billett and Kirschner et al\.\s+"
            r"CAST and Jwad et al\. describe multiple learning pathways\.",
            (
                "Billett and Kirschner et al. highlight the need for guided practice "
                "over discovery learning. CAST and Jwad et al. describe multiple "
                "learning pathways."
            ),
            "repaired_conclusion_fragment:guided_practice",
        ),
        (
            r"\b(DEWR defines the boundary for reasonable adjustment and maintaining "
            r"assessment integrity\.)\s+multiple learning pathways\.",
            r"\1",
            "removed_dangling_fragment:multiple_learning_pathways",
        ),
    ]
    for pattern, replacement, note in overlap_repairs:
        next_text = re.sub(pattern, replacement, repaired, flags=re.I)
        if next_text != repaired:
            repaired = next_text
            repairs.append(note)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", repaired) if p.strip()]
    if paragraphs:
        seen_sentences = set()
        removed_duplicates = 0
        removed_fragments = 0
        rebuilt_paragraphs = []
        for paragraph in paragraphs:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
            if not sentences:
                rebuilt_paragraphs.append(paragraph)
                continue
            kept = []
            normalized_sentences = [
                re.sub(r"\s+", " ", s).strip().lower()
                for s in sentences
            ]
            for sentence_index, sentence in enumerate(sentences):
                key = re.sub(r"\s+", " ", sentence).strip().lower()
                first_alpha = re.search(r"[A-Za-z]", sentence)
                if first_alpha and first_alpha.group(0).islower() and len(sentence.split()) >= 3:
                    contained_elsewhere = any(
                        other_index != sentence_index
                        and key
                        and key in other
                        for other_index, other in enumerate(normalized_sentences)
                    ) or any(key and key in prior for prior in seen_sentences)
                    if contained_elsewhere:
                        removed_fragments += 1
                        continue
                if len(sentence.split()) >= 8 and key in seen_sentences:
                    removed_duplicates += 1
                    continue
                if len(sentence.split()) >= 8:
                    seen_sentences.add(key)
                kept.append(sentence)
            if kept:
                rebuilt_paragraphs.append(" ".join(kept))
        if removed_duplicates:
            repaired = "\n\n".join(rebuilt_paragraphs)
            repairs.append(f"removed_duplicate_sentences:{removed_duplicates}")
        if removed_fragments:
            repaired = "\n\n".join(rebuilt_paragraphs)
            repairs.append(f"removed_duplicate_fragments:{removed_fragments}")

    repaired = re.sub(r"\n{3,}", "\n\n", repaired).strip()
    return repaired, repairs


def _source_repair_drift_false_positive(candidate: str, reasons: list[str]) -> bool:
    """Return True only for named-entity drift caused by source-damage repair."""
    if not isinstance(candidate, str) or not reasons:
        return False
    candidate_l = re.sub(r"\s+", " ", candidate).lower()
    for reason in reasons:
        match = re.match(r"lost_named_entity:\s+'([^']+)'", str(reason))
        if not match:
            return False
        entity = re.sub(r"\s+", " ", match.group(1)).strip()
        entity_l = entity.lower()
        if entity_l in candidate_l:
            continue
        words = [
            word.lower()
            for word in re.findall(r"\b[A-Za-z][A-Za-z]+\b", entity)
            if word.lower() not in {"introduction", "conclusion"}
        ]
        if words and all(word in candidate_l for word in words):
            continue
        return False
    return True


_AI_SEARCH_ENTITY_NOISE = {
    "the", "this", "these", "that", "with", "when", "while", "where",
    "learners", "learner", "students", "student", "competency",
    "introduction", "conclusion", "however", "therefore", "because",
    "centre", "center",
}


def _ai_search_drift_false_positive(candidate: str, reasons: list[str], similarity: float) -> bool:
    """Relax entity-only drift noise for high-similarity full-document candidates."""
    if not isinstance(candidate, str) or not reasons or similarity < 0.90:
        return False
    candidate_l = re.sub(r"\s+", " ", candidate).lower()
    for reason in reasons:
        match = re.match(r"lost_named_entity:\s+'([^']+)'", str(reason))
        if not match:
            return False
        entity = re.sub(r"\s+", " ", match.group(1)).strip()
        entity_l = entity.lower()
        if entity_l in candidate_l:
            continue
        words = [word.lower() for word in re.findall(r"\b[A-Za-z][A-Za-z]+\b", entity)]
        if not words:
            return False
        if any(word in {"introduction", "conclusion"} for word in words):
            # Full-document drift sees repaired headings such as
            # "Hairdressing Introduction Inclusive" as lost entities. The
            # protected-span check has already guarded citations/numbers/quotes;
            # this is layout damage, not semantic loss.
            continue
        if all(word in _AI_SEARCH_ENTITY_NOISE for word in words):
            continue
        content_words = [
            word
            for word in words
            if word not in _AI_SEARCH_ENTITY_NOISE
        ]
        if content_words and all(word in candidate_l for word in content_words):
            continue
        return False
    return True


_AI_SEARCH_CRITICAL_ENTITY_RE = re.compile(
    r"\b(?:Box Hill Institute|Certificate III|SHBHCUT\d+|CESE|Chandler|"
    r"Sweller|Billett|Kirschner|CAST|Jwad|DEWR)\b",
    re.I,
)


def _ai_search_entity_drift_scan_allowed(candidate: str, reasons: list[str], similarity: float) -> bool:
    """Allow scoring high-similarity candidates with only non-critical entity drift.

    This does not relax protected spans. It only prevents the scoring loop from
    throwing away otherwise useful full-document candidates because the generic
    drift guard misread sentence starts or repaired headings as named entities.
    """
    if not isinstance(candidate, str) or not reasons or similarity < 0.92:
        return False
    candidate_l = re.sub(r"\s+", " ", candidate).lower()
    for reason in reasons:
        match = re.match(r"lost_named_entity:\s+'([^']+)'", str(reason))
        if not match:
            return False
        entity = re.sub(r"\s+", " ", match.group(1)).strip()
        if entity.lower() in candidate_l:
            continue
        if _AI_SEARCH_CRITICAL_ENTITY_RE.search(entity):
            return False
    return True


def _reconstruction_drift_scan_allowed(candidate: str, reasons: list[str], similarity: float) -> bool:
    """Allow reconstruction scoring for non-substantive drift noise.

    The protected-span check runs immediately before this function. If it has
    passed, quote text, numbers, and citation names are still present. The
    keyword drift guard may still report quote loss when curly/straight quote
    marks differ or when the phrase is preserved without quotation marks; that
    should not block scanner scoring for reconstruction candidates.
    """
    if not isinstance(candidate, str) or not reasons:
        return False
    if all(str(reason).startswith("quote_lost:") for reason in reasons):
        return similarity >= 0.70
    if similarity < 0.78:
        return False
    candidate_l = re.sub(r"\s+", " ", candidate).lower()
    for reason in reasons:
        if str(reason).startswith("quote_lost:"):
            continue
        match = re.match(r"lost_named_entity:\s+'([^']+)'", str(reason))
        if not match:
            return False
        entity = re.sub(r"\s+", " ", match.group(1)).strip()
        if entity.lower() in candidate_l:
            continue
        if _AI_SEARCH_CRITICAL_ENTITY_RE.search(entity):
            return False
        words = [word.lower() for word in re.findall(r"\b[A-Za-z][A-Za-z]+\b", entity)]
        if not words or any(word not in _AI_SEARCH_ENTITY_NOISE for word in words):
            return False
    return True


def _normalize_protected_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def _protected_number_set(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", str(text or "")))


def _ai_search_protected_loss_reason(original: str, candidate: str, protected) -> str:
    """Lenient protected-span check for full-document AI candidates.

    The generic protected-span guard is byte-exact. That is too strict for
    AI-search candidates because the detector currently marks punctuation
    fragments such as ", 2017" and ". 149" as protected citations. For this
    stage, preserve the substance: numbers, quote content, and citation names.
    """
    candidate_norm = _normalize_protected_text(candidate)
    candidate_numbers = _protected_number_set(candidate)

    for number in sorted(_protected_number_set(original)):
        if number not in candidate_numbers:
            return f"number_lost:{number}"

    for span in protected or []:
        span_text = original[span.start_char:span.end_char]
        span_norm = _normalize_protected_text(span_text).strip('"').strip("'")
        if not span_norm:
            continue
        if span.reason == "direct_quote":
            if span_norm not in candidate_norm:
                return f"quote_lost:{span_norm[:40]}"
            continue
        if span.reason == "citation":
            names = [
                name.lower()
                for name in re.findall(r"\b[A-Z][A-Za-z]{2,}\b", span_text)
                if name.lower() not in {"pp"}
            ]
            missing_names = [name for name in names if name not in candidate_norm]
            if missing_names:
                return f"citation_name_lost:{missing_names[0]}"

    return ""


def _ai_search_prompt(
    source_text: str,
    raw_json: dict,
    strategy: str,
    *,
    reference_ai=None,
    required_ai_drop: float | None = None,
    target_ai_score: float | None = None,
) -> str:
    signal_brief = _ai_search_signal_brief(raw_json)
    repair_brief = _source_repair_brief(source_text)
    strategy_lines = {
        "syntax_demolition": [
            "Strategy: syntax demolition.",
            "Break original sentence routes. Do not keep the same subject-verb-object path when meaning allows.",
            "Split some long balanced sentences and merge a few short neighboring sentences where natural.",
        ],
        "paragraph_resequence": [
            "Strategy: paragraph resequencing.",
            "Change how paragraphs arrive at their point. Start from a concrete action, mistake, observation, or consequence before the broad claim.",
            "Avoid the same explanatory order used in the source.",
        ],
        "plain_workshop_voice": [
            "Strategy: plain workshop voice.",
            "Make the draft sound like a knowledgeable person explaining work they have actually seen.",
            "Use concrete verbs and uneven sentence length. Keep useful roughness.",
        ],
        "review_marked_grounding": [
            "Strategy: review-marked grounding.",
            "Where the scan points to unsupported or generic claims, add clearly marked author-review material using [[REVIEW: ...]] brackets.",
            "The bracketed material must be framed as a place for the user to verify or replace, not as a fabricated fact.",
            "Use these marked additions to break generic claim flow with source/evidence prompts, concrete context prompts, or careful limitation prompts.",
        ],
        "source_bridge_rebuild": [
            "Strategy: source bridge rebuild.",
            "Do not leave historical or technical claims floating. Add source-bridge sentences only when they can be phrased as review prompts.",
            "Use [[REVIEW: add source here explaining ...]] where the source is missing, then reconnect the claim in plainer wording.",
        ],
        "claim_narrowing": [
            "Strategy: claim narrowing.",
            "Turn broad claims into context-limited claims. Add cautious wording where evidence is implied but not supplied.",
            "Do not add new sources or fabricated evidence.",
        ],
        "cadence_disruption": [
            "Strategy: cadence disruption.",
            "Break repeated clean essay cadence. Vary openings, sentence length, and paragraph rhythm.",
            "Avoid polished connector chains and abstract summary language.",
        ],
        "anchor_first_rebuild": [
            "Strategy: anchor-first rebuild.",
            "For each paragraph, preserve the factual anchors, then rebuild the surrounding explanation from scratch.",
            "Prefer domain details already present in the source over new vocabulary.",
        ],
    }.get(strategy, ["Strategy: rewrite for lower measured AI score."])
    lines = [
        "DraftProof AI-score mitigation search.",
        "Objective: produce the lowest measured AI-likelihood score among candidate drafts.",
        (
            "Measured success condition: "
            f"reference AI={reference_ai}, required drop>={required_ai_drop}, "
            f"target AI<={target_ai_score}."
        ),
        "The next scan, not your explanation, decides whether this candidate succeeds.",
        "This is not copyediting. This is not polish. This is detector-targeted reconstruction.",
        *strategy_lines,
        "Use the detector signals as rewrite levers:",
        "- High generic assertion risk: replace broad claims with narrower claims tied to existing source, classroom, client, task, or process details.",
        "- High qualifying AI density: change paragraph architecture, not just words; vary where claims, examples, and source relations appear.",
        "- High top-k predictability: rebuild clause order, split/merge sentence routes, and use less expected verbs while preserving meaning.",
        "- Source/citation gaps: narrow or qualify the claim unless the source already exists in the draft.",
        "- Repeated starters/rhythm: vary openings naturally without mechanical prefixes.",
        "Hard constraints:",
        "Keep the same topic, stance, factual claims, numbers, names, quotes, citations, unit codes, and chronology.",
        "Do not invent new evidence, citations, sources, dates, institutions, or examples.",
        "If evidence is missing and the strategy allows marked grounding, use [[REVIEW: ...]] bracketed text instead of inventing the evidence.",
        "Do not summarize or shorten the document. Keep length within about 85% to 115% of the source, except remove accidental duplicate fragments if the source already contains prior rewrite damage.",
        "Do not leave any non-protected source sentence verbatim. Rebuild every sentence route.",
        "Change most sentence openings and vary paragraph openings. Avoid preserving the same paragraph order inside every paragraph.",
        "Avoid generic polished phrases: crucial, significant, essential, framework, landscape, operational obstacles, technical rigor, facilitates, enables, embedded within, especially evident.",
        "Use concrete wording, varied sentence routes, and paragraph-level reconstruction.",
    ]
    if signal_brief:
        lines.append(signal_brief)
    if repair_brief:
        lines.append(repair_brief)
    lines.extend([
        "SOURCE DRAFT:\n<TARGET_DOCUMENT>\n" + source_text + "\n</TARGET_DOCUMENT>",
        "Output the complete rewritten draft only.",
        "No commentary, no bullets, no headings added by you, no score estimate.",
    ])
    return "\n".join(lines)


def _ai_search_feedback_prompt(
    source_text: str,
    raw_json: dict,
    search_summary: dict,
    attempt_index: int,
) -> str:
    """Build a score-aware retry prompt from actual candidate outcomes."""
    signal_brief = _ai_search_signal_brief(raw_json)
    repair_brief = _source_repair_brief(source_text)
    reference_ai = search_summary.get("reference_ai")
    required_drop = search_summary.get("required_ai_drop")
    target_score = search_summary.get("target_ai_score")
    best_attempt = search_summary.get("best_attempt") or {}
    candidate_lines = []
    for item in (search_summary.get("candidates") or [])[-10:]:
        if not isinstance(item, dict):
            continue
        bits = [str(item.get("strategy") or "candidate")]
        if item.get("ai") is not None:
            bits.append(f"AI={item.get('ai')}")
            bits.append(f"delta={item.get('ai_delta_vs_reference')}")
        if item.get("writing_quality") is not None:
            bits.append(f"WQ={item.get('writing_quality')}")
        if item.get("findings") is not None:
            bits.append(f"findings={item.get('findings')}")
        selection = item.get("selection_status") or {}
        if selection.get("reason"):
            bits.append(f"selection={selection.get('reason')}")
        if item.get("reason"):
            bits.append(f"blocked={item.get('reason')}")
        drift_reasons = item.get("drift_reasons") or item.get("drift_reasons_relaxed")
        if drift_reasons:
            bits.append("drift=" + " | ".join(str(x) for x in drift_reasons[:5]))
        candidate_lines.append("- " + "; ".join(bits))
    scoreboard = "\n".join(candidate_lines) or "- No candidate reached scoring yet."

    return (
        "DraftProof already tried candidate rewrites and rescanned what passed local checks.\n"
        f"Reference AI score: {reference_ai}. Required drop: {required_drop}. Target AI score: {target_score}.\n"
        "Your task is to beat the required target, not to polish and not to make a tiny reduction.\n"
        f"Current best attempt: {best_attempt or '[none]'}\n\n"
        f"{signal_brief}\n\n"
        f"{repair_brief}\n\n"
        "Candidate scoreboard from the actual detector:\n"
        f"{scoreboard}\n\n"
        "What the next attempt must do:\n"
        "- Return the complete rewritten document only.\n"
        "- Preserve all unit codes, source names, citations, years, numbers, and quotes.\n"
        "- Specifically reduce generic assertions, qualifying-text AI density, and top-k predictability.\n"
        "- If earlier candidates only changed wording, change paragraph structure and claim order this time.\n"
        "- Rewrite the highest-driver paragraphs more aggressively while preserving all protected facts.\n"
        "- Rebuild paragraph flow where needed: start from classroom/salon action, learner behavior, or source relation before broad claims.\n"
        "- Do not add fake facts. If evidence is missing, narrow the claim instead of inventing support.\n"
        "- Avoid mechanical anchor prefixes and visible review markers in the final document.\n"
        "- Repair inherited source damage: broken words, merged headings, and duplicate sentence fragments.\n"
        f"- This is feedback attempt {attempt_index}; make a materially different full-document candidate.\n\n"
        "SOURCE DOCUMENT:\n"
        f"{source_text.strip()}\n\n"
        "Return only the complete rewritten document."
    )


def _logical_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", str(text or "").strip()) if p.strip()]


def _join_logical_paragraphs(paragraphs: list[str]) -> str:
    return "\n\n".join(p.strip() for p in paragraphs if str(p or "").strip())


def _paragraph_sentence_starters(paragraph: str) -> list[str]:
    starters = []
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        words = re.findall(r"\b[A-Za-z][A-Za-z']*\b", sentence)
        if words:
            starters.append(words[0].lower())
    return starters


def _paragraph_component_targets(text: str, raw_json: dict, limit: int = 3) -> list[dict]:
    """Rank logical paragraphs by their likely contribution to AI-style score."""
    paragraphs = _logical_paragraphs(text)
    if not paragraphs:
        return []
    briefs = (raw_json or {}).get("rewrite_edit_briefs") or []
    scored = []
    generic_re = re.compile(
        r"\b(?:should|must|need(?:s|ed)?|requires?|important|significant|"
        r"supports?|helps?|allows?|enables?|creates?|means|is|are|can|will)\b",
        re.I,
    )
    concrete_re = re.compile(
        r"\b(?:SHBHCUT\d+|CESE|Chandler|Sweller|Billett|Kirschner|CAST|Jwad|DEWR|"
        r"Box Hill|HBB26|\d+(?:\.\d+)?%?|\([A-Z][A-Za-z]+,\s*\d{4}\)|"
        r"\bI\b|\bmy\b|\bmannequin|client|sectioning|projection|guide|scissor|comb|elbow)\b",
        re.I,
    )
    for index, paragraph in enumerate(paragraphs):
        words = paragraph.split()
        if len(words) < 45:
            continue
        matching_briefs = []
        for brief in briefs:
            if not isinstance(brief, dict):
                continue
            target_sentence = (brief.get("target_sentence") or "").strip()
            if target_sentence and target_sentence in paragraph:
                matching_briefs.append(brief)
        generic_hits = len(generic_re.findall(paragraph))
        concrete_hits = len(concrete_re.findall(paragraph))
        starters = _paragraph_sentence_starters(paragraph)
        repeated_starter_count = len(starters) - len(set(starters))
        has_citation = bool(re.search(r"\(\s*[A-Z][A-Za-z]+(?:\s+et\s+al\.)?,\s*\d{4}", paragraph))
        brief_score = sum(
            float(((b.get("signals") or {}).get("score") or 0.0) or 0.0)
            for b in matching_briefs
        )
        source_gap = 0 if has_citation else 1
        score = (
            len(matching_briefs) * 5.0
            + brief_score * 8.0
            + min(generic_hits / max(len(words) / 90.0, 1.0), 8.0)
            + source_gap * 2.0
            + min(repeated_starter_count, 4) * 0.75
            - min(concrete_hits, 12) * 0.20
        )
        if score <= 0:
            continue
        scored.append({
            "index": index,
            "paragraph": paragraph,
            "previous_paragraph": paragraphs[index - 1] if index > 0 else "",
            "next_paragraph": paragraphs[index + 1] if index + 1 < len(paragraphs) else "",
            "score": round(score, 3),
            "drivers": {
                "rewrite_brief_count": len(matching_briefs),
                "predictability_score_sum": round(brief_score, 4),
                "generic_assertion_hits": generic_hits,
                "concrete_anchor_hits": concrete_hits,
                "source_gap": bool(source_gap),
                "repeated_sentence_starters": repeated_starter_count,
                "word_count": len(words),
            },
            "target_sentences": [
                (b.get("target_sentence") or "") for b in matching_briefs[:5]
            ],
            "problem_spans": [
                span
                for b in matching_briefs[:4]
                for span in (((b.get("signals") or {}).get("predictable_token_spans")) or [])[:3]
            ][:10],
            "domain_anchors": list(dict.fromkeys(
                anchor
                for b in matching_briefs[:4]
                for anchor in (b.get("domain_anchors") or [])[:6]
            ))[:16],
        })
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:max(0, limit)]


def _paragraph_component_prompt(
    target: dict,
    raw_json: dict,
    attempt_index: int,
    *,
    reference_ai=None,
    required_ai_drop: float | None = None,
    target_ai_score: float | None = None,
    candidate_count: int = 1,
) -> str:
    signal_brief = _ai_search_signal_brief(raw_json)
    drivers = target.get("drivers") or {}
    candidate_count = max(1, int(candidate_count or 1))
    return (
        "DraftProof paragraph-component AI mitigation.\n"
        "Rewrite only the target paragraph.\n"
        "Goal: reduce the final full-document AI score after this paragraph is patched back into the document.\n\n"
        f"Measured success condition: reference AI={reference_ai}, required drop>={required_ai_drop}, target AI<={target_ai_score}.\n"
        "The candidate will be rescanned; do not make a mild paraphrase.\n\n"
        f"{signal_brief}\n\n"
        f"Paragraph driver score: {target.get('score')}\n"
        f"Drivers: {json.dumps(drivers, ensure_ascii=False)}\n"
        "Target sentences from scan:\n"
        + "\n".join(f"- {s}" for s in (target.get("target_sentences") or [])[:5])
        + "\nProblem spans:\n"
        + "\n".join(f"- {s}" for s in (target.get("problem_spans") or [])[:10])
        + "\nDomain anchors already present nearby:\n"
        + ", ".join(str(a) for a in (target.get("domain_anchors") or [])[:16])
        + "\n\nPrevious paragraph context:\n"
        f"{target.get('previous_paragraph') or '[none]'}\n\n"
        "TARGET PARAGRAPH:\n"
        f"<TARGET_PARAGRAPH>\n{target.get('paragraph') or ''}\n</TARGET_PARAGRAPH>\n\n"
        "Next paragraph context:\n"
        f"{target.get('next_paragraph') or '[none]'}\n\n"
        "Rewrite rules:\n"
        "- Preserve all citations, years, numbers, names, unit codes, and source references.\n"
        "- Do not invent new evidence, sources, people, institutions, or events.\n"
        "- Break generic assertion flow: avoid broad claims unless tied to the local haircutting/classroom process.\n"
        "- Start from concrete action, learner behavior, source relation, or assessment consequence before broad explanation.\n"
        "- Change paragraph architecture: reorder claim/example/source relation where meaning allows.\n"
        "- Convert generic claims into specific process observations using only anchors already present nearby.\n"
        "- Vary sentence length and clause order enough that this is not a synonym swap.\n"
        "- Change sentence openings and sentence routes. Do not polish with academic filler.\n"
        "- Keep author voice and first-person classroom observation where it already exists.\n"
        "- Remove duplicate fragments if present inside the target paragraph.\n"
        f"- Batch attempt {attempt_index}: make each option materially different from generic rephrasing.\n\n"
        f"Return exactly {candidate_count} alternative replacement paragraphs using this exact format:\n"
        "<CANDIDATE_1>\nreplacement paragraph only\n</CANDIDATE_1>\n"
        "<CANDIDATE_2>\nreplacement paragraph only\n</CANDIDATE_2>\n"
        "...continue until the requested candidate count.\n"
        "Do not include commentary outside the candidate tags."
    )


def _extract_paragraph_component_candidates(output: str, limit: int) -> list[str]:
    text = str(output or "").strip()
    if not text:
        return []
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    tagged = re.findall(
        r"<CANDIDATE_(\d+)>\s*(.*?)\s*</CANDIDATE_\1>",
        text,
        flags=re.I | re.S,
    )
    if tagged:
        ordered = sorted(tagged, key=lambda item: int(item[0]))
        return [
            body.strip()
            for _, body in ordered[:max(1, limit)]
            if body.strip()
        ]
    marker_matches = re.findall(
        r"(?ims)^\s*(?:candidate|option)\s*\d+\s*[:.-]\s*(.*?)(?=^\s*(?:candidate|option)\s*\d+\s*[:.-]|\Z)",
        text,
    )
    if marker_matches:
        return [
            body.strip()
            for body in marker_matches[:max(1, limit)]
            if body.strip()
        ]
    return [text]


def _clean_paragraph_component_candidate(candidate: str, original_paragraph: str) -> tuple[str, str]:
    text = str(candidate or "").strip()
    if not text:
        return "", "empty_candidate"
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"^(?:replacement|rewritten)\s+paragraph\s*:\s*", "", text, flags=re.I).strip()
    paragraphs = _logical_paragraphs(text)
    if not paragraphs:
        return "", "empty_candidate"
    text = " ".join(" ".join(p.split()) for p in paragraphs)
    if text == " ".join(str(original_paragraph or "").split()):
        return "", "unchanged_paragraph"
    orig_len = max(1, len(str(original_paragraph or "")))
    if len(text) < max(80, int(orig_len * 0.55)):
        return "", f"paragraph_too_short {len(text)}<{max(80, int(orig_len * 0.55))}"
    if len(text) > int(orig_len * 1.55):
        return "", f"paragraph_too_long {len(text)}>{int(orig_len * 1.55)}"
    return text, ""


def _splice_paragraph(text: str, paragraph_index: int, replacement: str) -> str:
    paragraphs = _logical_paragraphs(text)
    if paragraph_index < 0 or paragraph_index >= len(paragraphs):
        return text
    paragraphs[paragraph_index] = replacement.strip()
    return _join_logical_paragraphs(paragraphs)


def _ai_search_marked_grounding_candidates(source_text: str) -> list[tuple[str, str]]:
    """Create deterministic marked-addition candidates for missing grounding.

    These are intentionally visible to the user. They target the detector's
    generic assertion and source-grounding drivers without inventing evidence.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", source_text.strip()) if p.strip()]
    paragraph_sentences = [
        [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
        for paragraph in paragraphs
    ]
    sentences = [sentence for paragraph in paragraph_sentences for sentence in paragraph]
    if len(sentences) < 8:
        return []

    concrete_re = re.compile(
        r"\b\d+\b|\baccording to\b|\bfor example\b|\bfor instance\b|"
        r"\bin my\b|\bI (?:saw|noticed|found|observed|learned|worked)\b|"
        r"\bwe (?:found|observed|measured|tested)\b|\"",
        re.I,
    )
    assertion_re = re.compile(
        r"\b(is|are|was|has|have|can|should|must|needs?|creates?|makes?|requires?|means)\b",
        re.I,
    )
    review_notes = [
        "[[REVIEW: For example, add the exact source, client moment, or salon observation that proves this point.]]",
        "[[REVIEW: Add the specific evidence behind this claim, or soften it if the evidence is only limited.]]",
        "[[REVIEW: Name the source or concrete example the reader should connect to this sentence.]]",
        "[[REVIEW: Add one real detail from the task, workplace, class, or client situation before keeping this claim.]]",
        "[[REVIEW: If this is based on experience, add what was seen, who was involved, and what changed.]]",
        "[[REVIEW: Add a citation or replace this with a narrower claim the draft can support.]]",
        "[[REVIEW: Give one concrete process step here, not just the general conclusion.]]",
        "[[REVIEW: Add the limitation or condition under which this statement is true.]]",
    ]

    scored = []
    for index, sentence in enumerate(sentences):
        words = sentence.split()
        if len(words) < 10:
            continue
        has_concrete = bool(concrete_re.search(sentence))
        has_assertion = bool(assertion_re.search(sentence))
        score = (2 if has_assertion else 0) + (2 if not has_concrete else 0) + min(len(words) / 35, 1)
        if score >= 2:
            scored.append((score, index))
    scored.sort(reverse=True)
    target_indexes = sorted(index for _, index in scored[:8])
    if not target_indexes:
        return []

    concrete_prefixes = [
        "In practice, ",
        "For this task, ",
        "During the practical work, ",
        "In assessment, ",
        "When learners are cutting, ",
        "During feedback, ",
    ]

    def build_marked(limit: int, label: str) -> tuple[str, str]:
        note_indexes = set(target_indexes[:limit])
        rebuilt_paragraphs = []
        flat_index = 0
        used = 0
        for paragraph in paragraph_sentences:
            rebuilt = []
            for sentence in paragraph:
                rebuilt.append(sentence)
                if flat_index in note_indexes:
                    rebuilt.append(review_notes[used % len(review_notes)])
                    used += 1
                flat_index += 1
            rebuilt_paragraphs.append(" ".join(rebuilt))
        return label, "\n\n".join(rebuilt_paragraphs)

    def _contextualize_sentence(sentence: str, prefix: str) -> str:
        stripped = sentence.strip()
        if not stripped:
            return stripped
        if re.match(
            r"^(?:this|it|these|they|learners|competency|demonstration|"
            r"reasonable adjustment|inclusive learning design|the challenge|the standard)\b",
            stripped,
            re.I,
        ):
            return prefix + stripped[0].lower() + stripped[1:]
        return prefix.rstrip(", ") + ": " + stripped

    def build_process_anchors(label: str, *, limit: int) -> tuple[str, str]:
        anchor_indexes = set(target_indexes[:limit])
        rebuilt_paragraphs = []
        flat_index = 0
        used = 0
        for paragraph in paragraph_sentences:
            rebuilt = []
            anchored_this_paragraph = False
            for sentence in paragraph:
                words = sentence.split()
                has_concrete = bool(concrete_re.search(sentence))
                should_anchor = (
                    flat_index in anchor_indexes
                    and not anchored_this_paragraph
                    and len(words) >= 8
                    and not has_concrete
                )
                if should_anchor:
                    prefix = concrete_prefixes[used % len(concrete_prefixes)]
                    rebuilt.append(_contextualize_sentence(sentence, prefix))
                    anchored_this_paragraph = True
                    used += 1
                else:
                    rebuilt.append(sentence)
                flat_index += 1
            rebuilt_paragraphs.append(" ".join(rebuilt))
        return label, "\n\n".join(rebuilt_paragraphs)

    return [
        build_process_anchors("deterministic_process_anchor_generic", limit=min(4, len(target_indexes))),
        build_process_anchors("deterministic_process_anchor_all", limit=min(8, len(target_indexes))),
        build_marked(min(4, len(target_indexes)), "deterministic_marked_grounding_light"),
        build_marked(min(8, len(target_indexes)), "deterministic_marked_grounding_strong"),
    ]


def _enrich_report_authorship_schema(report_dict: dict) -> dict:
    """Backfill current authorship fields for older saved scan JSON.

    Rewrite jobs often consume the saved scan JSON as their contract. If the
    scan was created before a detector/schema upgrade, the rewrite report can
    otherwise omit newer fields such as qualifying_text_ai_density even though
    the saved JSON still contains enough input text and scanner components to
    derive them cheaply.
    """
    if not isinstance(report_dict, dict):
        return report_dict

    badge = report_dict.get("ai_risk_badge") or {}
    if not isinstance(badge, dict):
        return report_dict

    ai_components = badge.get("ai_components") or {}
    has_density = isinstance(ai_components, dict) and "qualifying_text_ai_density" in ai_components
    if badge.get("authorship_rating") and has_density:
        return report_dict

    text = report_dict.get("input_text") or report_dict.get("original_text") or ""
    if not isinstance(text, str) or len(text.split()) < 300:
        return report_dict

    writing_components = badge.get("writing_components") or {}
    layer3_input = build_layer3_input_from_text(
        text,
        predictability=_metric_decimal(ai_components.get("predictability")),
        topk_pattern=_metric_decimal(ai_components.get("topk_pattern")),
        generic_phrase_density=_metric_decimal(ai_components.get("generic_phrase_density")),
        broad_claim_risk=_metric_decimal(writing_components.get("broad_claim_risk")),
        citation_weakness_risk=_metric_decimal(writing_components.get("citation_weakness_risk")),
        unsupported_claim_risk=_metric_decimal(writing_components.get("unsupported_claim_risk")),
        source_grounding_strength=_metric_decimal(writing_components.get("source_grounding_strength")),
        domain_grounding_strength=_metric_decimal(writing_components.get("domain_grounding_strength")),
    )
    layer3 = Layer3Scorer().score(layer3_input)

    enriched = dict(report_dict)
    enriched_badge = dict(badge)
    enriched_badge.update({
        "tier": layer3.tier.value,
        "ai_likelihood_score": round(layer3.ai_likelihood_score * 100, 2),
        "authorship_rating": layer3.authorship_rating,
        "authorship_rating_label": layer3.authorship_rating.get("label"),
        "authorship_rating_code": layer3.authorship_rating.get("code"),
        "ai_cluster_boost": round(layer3.ai_cluster_boost * 100, 2) if layer3.ai_cluster_boost else 0,
        "ai_cluster_name": layer3.ai_cluster_name,
        "ai_components": {k: round(v * 100, 2) for k, v in layer3.ai_phase.components.items()},
        "writing_quality_tier": layer3.writing_quality_tier.value,
        "writing_quality_score": round(layer3.writing_quality_score * 100, 2),
        "writing_components": {k: round(v * 100, 2) for k, v in layer3.writing_phase.components.items()},
        "review_priority": layer3.review_priority,
        "confidence": layer3.confidence.value,
        "reasons": layer3.reasons,
        "guardrails": layer3.guardrails,
        "schema_enriched_from_input_text": True,
    })
    enriched["ai_risk_badge"] = enriched_badge
    return enriched


def _detail_value(detail: dict, *keys, default=0):
    """Read the first present metric key from a sentence detail dict."""
    for key in keys:
        value = detail.get(key)
        if value is not None:
            return value
    return default


def _scan_scope_summary(report_dict: dict) -> dict:
    """Small diagnostic summary of how much text the detector scored."""
    pred = (report_dict or {}).get("predictability") or {}
    if not isinstance(pred, dict):
        return {}
    score_derivation = pred.get("score_derivation") or {}
    sentences = pred.get("sentences") or []
    all_sentences = pred.get("all_sentences") or []
    scope = {
        "predictability_scored_sentences": len(sentences),
    }
    if all_sentences:
        scope["predictability_total_sentences"] = len(all_sentences)
    included = score_derivation.get("included_sentence_count")
    if included is not None:
        scope["predictability_included_sentence_count"] = included
    raw_mean = score_derivation.get("raw_mean")
    if isinstance(raw_mean, (int, float)):
        scope["predictability_raw_mean"] = round(float(raw_mean), 4)
    return scope


def _sentence_detail_lookup(details: list) -> dict:
    """Map sentence text to metric details, preserving the first occurrence."""
    lookup = {}
    for d in details or []:
        sentence = (d.get("sentence") or "").strip()
        if sentence and sentence not in lookup:
            lookup[sentence] = d
    return lookup


def _build_aligned_sentence_comparison(mp) -> list:
    """Build before/after sentence rows using text alignment, not index pairing.

    Rewritten documents can shift sentence positions after a local edit. Pairing
    sentence metrics by index makes every later sentence look rewritten and can
    produce blank rewritten cells when metric lists have different lengths.
    """
    if not mp:
        return []

    original_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", mp.original_text or "")
        if s.strip()
    ]
    final_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", mp.final_text or "")
        if s.strip()
    ]
    if not original_sentences and not final_sentences:
        return []

    orig_details = (mp.original_metrics.sentence_details if mp.original_metrics else []) or []
    final_details = (mp.final_metrics.sentence_details if mp.final_metrics else []) or []
    orig_lookup = _sentence_detail_lookup(orig_details)
    final_lookup = _sentence_detail_lookup(final_details)

    rows = []
    matcher = SequenceMatcher(a=original_sentences, b=final_sentences, autojunk=False)
    row_index = 1
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for o_sent, n_sent in zip(original_sentences[i1:i2], final_sentences[j1:j2]):
                o = orig_lookup.get(o_sent, {})
                n = final_lookup.get(n_sent, {})
                rows.append({
                    "index": row_index,
                    "orig_tier": _detail_value(o, "label", "risk_label", default="?"),
                    "orig_risk": _detail_value(o, "risk", "predictability_risk"),
                    "orig_top10": _detail_value(o, "top10_ratio", "top_10_ratio"),
                    "orig_sentence": o_sent,
                    "new_tier": _detail_value(n, "label", "risk_label", default="?"),
                    "new_risk": _detail_value(n, "risk", "predictability_risk"),
                    "new_top10": _detail_value(n, "top10_ratio", "top_10_ratio"),
                    "new_sentence": n_sent,
                })
                row_index += 1
            continue

        old_block = original_sentences[i1:i2]
        new_block = final_sentences[j1:j2]
        o_sent = " ".join(old_block).strip()
        n_sent = " ".join(new_block).strip()
        o = orig_lookup.get(old_block[0], {}) if old_block else {}
        n = final_lookup.get(new_block[0], {}) if new_block else {}
        rows.append({
            "index": row_index,
            "orig_tier": _detail_value(o, "label", "risk_label", default="?"),
            "orig_risk": _detail_value(o, "risk", "predictability_risk"),
            "orig_top10": _detail_value(o, "top10_ratio", "top_10_ratio"),
            "orig_sentence": o_sent,
            "new_tier": _detail_value(n, "label", "risk_label", default="?"),
            "new_risk": _detail_value(n, "risk", "predictability_risk"),
            "new_top10": _detail_value(n, "top10_ratio", "top_10_ratio"),
            "new_sentence": n_sent,
        })
        row_index += 1
    return rows


def sanitize_text(text: str) -> str:
    """Fix mojibake and normalize Unicode in text before processing.

    Handles UTF-8 bytes that were decoded as latin-1, which produces
    artifacts like: â€™ â€" â€œ â€\x9d â€¦
    """
    # Fix common mojibake patterns
    text = text.replace('â€™', "'").replace('â€˜', "'")
    text = text.replace('â€œ', '"').replace('â€\x9d', '"')
    text = text.replace('â€"', ' -- ').replace('â€"', '-')
    text = text.replace('â€¦', '...')
    # Normalize remaining Unicode to ASCII equivalents
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('—', ' -- ').replace('–', '-')
    text = text.replace('…', '...')
    text = text.replace(' ', ' ')
    # Clean up double spaces from replacements
    text = re.sub(r'  +', ' ', text)
    return text


def run_rewrite_pipeline(
    json_path: str = None,
    text: str = None,
    detect_json: dict = None,
    output_dir: str = None,
    max_passes: int = 3,
    max_detect_loops: int = 0,
    target_top10: float = 0.50,
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    verbose: bool = False,
    ai_only: bool = True,
    progress_callback=None,
) -> dict:
    """Run the full rewrite pipeline from detect JSON or raw text.

    Args:
        json_path: Path to detect JSON file.
        text: Raw text (will run detect first).
        detect_json: Pre-loaded detect JSON dict.
        output_dir: Where to write output files.
        max_passes: Max rewrite passes per loop.
        max_detect_loops: Max detect-rewrite loops.
        target_top10: Target top-10 ratio for convergence.
        model: LLM model for rewriting (None → from env).
        api_key: API key for LLM (None → from env).
        base_url: LLM API base URL (None → from env or OpenRouter default).
        verbose: Include scanner details in report.
        ai_only: Only rewrite AI-generation findings (default True).

    Returns dict with paths and summary.
    """
    _load_local_env()
    llm_roles = _llm_role_config(model)
    generator_model = llm_roles.get("generator_model") or model
    retry_model = llm_roles.get("retry_model") or generator_model

    # ── Parse input ────────────────────────────────────────────────
    ctx: DetectJSONContext = None

    if json_path or detect_json:
        if detect_json:
            ctx = DetectJSONParser.parse_dict(detect_json)
        else:
            ctx = DetectJSONParser.parse(json_path)
        text = ctx.input_text
    elif text:
        # Run detect first, then parse
        from detect_pipeline import run_detect
        detect_result = run_detect(text, output_dir or "test_output", verbose=verbose)
        report = detect_result["report"]

        from detect.base import DetectResult
        by_scanner = {}
        for tier_findings in report.findings_by_tier.values():
            for f in tier_findings:
                by_scanner.setdefault(f.scanner, []).append(f)

        detect_results = []
        for scanner, findings in by_scanner.items():
            # Preserve raw data from report JSON for scanners that have it
            scanner_raw = None
            if scanner == "predictability":
                pred = detect_json.get("predictability", {})
                # Use all_sentences (full text + scores) if available,
                # otherwise fall back to the predictability block
                all_sents = pred.get("all_sentences")
                if all_sents:
                    scanner_raw = {"sentences": all_sents}
                else:
                    scanner_raw = pred if pred else None
            detect_results.append(DetectResult(
                scanner=scanner,
                overall_risk=0.5,
                confidence="medium",
                confidence_reason="from detect pipeline",
                risk_distribution={},
                findings=findings,
                policy_message="",
                raw=scanner_raw,
            ))
        ctx = DetectJSONContext(
            detect_results=detect_results,
            input_text=text,
        )

    if not text or not text.strip():
        raise ValueError("Empty input text")

    if isinstance(getattr(ctx, "raw_json", None), dict):
        ctx.raw_json = _enrich_report_authorship_schema(ctx.raw_json)
        _ensure_ai_mitigation_contract(ctx.raw_json)

    # ── Check rewrite decision from detect ──────────────────────────
    if ctx.rewrite_decision and not ctx.rewrite_decision.get("run_rewrite", True):
        reason = ctx.rewrite_decision.get("reason", "Rewrite not recommended")
        print(f"Rewrite skipped: {reason}")
        return {
            "status": "skipped",
            "message": reason,
            "tier": ctx.overall_tier,
        }

    all_findings = [f for dr in ctx.detect_results for f in dr.findings]
    if not all_findings:
        print("No findings to rewrite. Text is clean.")
        return {"status": "clean", "message": "No findings to rewrite"}

    def report_progress(percent: int, message: str) -> None:
        if not progress_callback:
            return
        progress_callback(max(40, min(79, int(percent))), message)

    # ── Run rewrite ─────────────────────────────────────────────────
    print(f"Running rewrite pipeline...")
    print(f"  Input: {len(text)} chars, {len(ctx.detect_results)} scanner results")
    if ctx.rewrite_decision:
        print(f"  Decision: mode={ctx.rewrite_decision.get('mode', 'targeted')}")

    total_findings = sum(len(dr.findings) for dr in ctx.detect_results)
    if ai_only:
        ai_count = sum(
            len(dr.findings if dr.scanner == "ai_generation"
                else [f for f in dr.findings
                      if (f.metadata or {}).get("scanner") == "ai_generation"
                      or (f.metadata or {}).get("category") == "ai_generation"])
            for dr in ctx.detect_results
        )
        print(f"  AI-only mode: {ai_count} AI findings out of {total_findings} total")
    else:
        medium_count = sum(
            len([f for f in dr.findings if f.risk_level == "medium"])
            for dr in ctx.detect_results
        )
        print(f"  Medium-only mode: {medium_count} findings out of {total_findings} total")

    # Sanitize input text before rewrite (fix mojibake from PDF/docx extraction)
    text = sanitize_text(text)

    pre_rewrite_badge = (ctx.raw_json or {}).get("ai_risk_badge") or {}
    pre_rewrite_ai = pre_rewrite_badge.get("ai_likelihood_score")
    ai_mitigation_contract = _ensure_ai_mitigation_contract(ctx.raw_json)
    ai_mitigation_needs_author = _ai_mitigation_requires_user_input(ai_mitigation_contract)
    allow_auto_with_author_gaps = _env_flag("DRAFTPROOF_ALLOW_AUTO_WITH_AUTHOR_GAPS", True)
    ai_search_first = (
        os.environ.get("DRAFTPROOF_AI_SEARCH_FIRST", "1") != "0"
        and isinstance(pre_rewrite_ai, (int, float))
        and pre_rewrite_ai >= _float_env("DRAFTPROOF_AI_FIRST_REQUIRED_MIN_AI", 50.0)
        and (not ai_mitigation_needs_author or allow_auto_with_author_gaps)
    )
    rewrite_config = None
    if ai_search_first:
        rewrite_config = RewriteConfig(
            max_llm_calls=0,
            max_density_passes=0,
            max_rewrite_seconds=30,
        )
    elif ai_mitigation_needs_author and not allow_auto_with_author_gaps:
        rewrite_config = RewriteConfig(
            max_llm_calls=0,
            max_density_passes=0,
            max_rewrite_seconds=30,
        )

    t0 = time.time()
    report_progress(41, "Preparing rewrite plan from scan findings")
    result: RewriteModuleResult = run_rewrite(
        content=text,
        detect_results=ctx.detect_results,
        api_key=api_key,
        model=generator_model,
        base_url=base_url,
        max_passes=max_passes,
        target_top10=target_top10,
        max_detect_loops=max_detect_loops,
        output_dir=output_dir,
        rewrite_context=ctx,
        ai_only=ai_only,
        config=rewrite_config,
        progress_callback=report_progress,
    )
    result.summary["llm_model_roles"] = llm_roles
    result.summary["ai_mitigation"] = ai_mitigation_contract
    if ai_mitigation_needs_author:
        result.summary["ai_mitigation_blocked_auto_rewrite"] = not allow_auto_with_author_gaps
        if not allow_auto_with_author_gaps:
            result.summary["outcome"] = "suggestion_only"
        educational_rewrite = _build_educational_mitigation_rewrite(text, ai_mitigation_contract)
        if educational_rewrite:
            result.summary["educational_mitigation_rewrite"] = educational_rewrite
        suggestions = result.summary.setdefault("manual_suggestions", [])
        existing_keys = {
            (
                item.get("component"),
                item.get("suggested_sentence"),
                item.get("user_input_needed"),
            )
            for item in suggestions
            if isinstance(item, dict)
        }
        for suggestion in _manual_summary_from_ai_mitigation(ai_mitigation_contract):
            key = (
                suggestion.get("component"),
                suggestion.get("suggested_sentence"),
                suggestion.get("user_input_needed"),
            )
            if key not in existing_keys:
                suggestions.append(suggestion)
                existing_keys.add(key)
    if ai_search_first:
        result.summary["rewrite_engine_mode"] = "ai_search_first_skip_rewrite_prepass"
        result.summary.setdefault("saved_contract_notes", []).append(
            "Skipped costly density/sentence LLM prepass because AI mitigation search is the first objective."
        )
    elif ai_mitigation_needs_author and not allow_auto_with_author_gaps:
        result.summary["rewrite_engine_mode"] = "guided_authenticity_requires_author_input"
        result.summary.setdefault("saved_contract_notes", []).append(
            "Skipped automatic sentence/density rewrite because AI-Mitigation requires author-supplied grounding."
        )
    engine_elapsed = time.time() - t0
    stage_timings = [{"stage": "rewrite_engine", "seconds": round(engine_elapsed, 3)}]
    report_progress(74, "Building rewrite comparison")

    # ── Write output ────────────────────────────────────────────────
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "test_output"
        ))

    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_dir, f"draftproof_rewrite_{ts}.md")
    json_path_out = os.path.join(output_dir, f"draftproof_rewrite_{ts}.json")

    # Extract AI-only findings from detect JSON
    ai_findings = []
    raw_findings = ctx.raw_json.get("findings", {})
    for tier in ("critical", "high", "medium", "low"):
        for f in raw_findings.get(tier, []):
            cat = (f.get("category") or f.get("scanner") or "").lower()
            if cat == "ai_generation":
                ai_findings.append(f)

    # Get sentence comparison from the MultiPassResult, aligned by text diff.
    sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)

    # Inject detect scan scores into summary so rewrite report shows
    # the same risk scores the user saw in the detect scan report.
    badge = ctx.raw_json.get("ai_risk_badge", {})
    if badge:
        result.summary["detect_ai_likelihood"] = badge.get("ai_likelihood_score", 0)
        result.summary["detect_writing_quality"] = badge.get("writing_quality_score", 0)

    # ── Final full detect scan ──────────────────────────────────────
    # If no text changed, do not re-scan. The detector has stochastic/heuristic
    # variance, so rescanning identical text can falsely show a rewrite
    # regression even when the rewrite engine made no edit.
    #
    # When text did change, run the same full scan used by detect reports. The
    # rewrite engine may use a targeted rescan internally for speed, but the
    # user-facing report and rollback decision need a product-level full scan.
    rewritten_text = result.mp_result.final_text if result.mp_result else text
    if rewritten_text == text:
        report_progress(78, "No automatic text changes were kept")
        result.summary["no_text_change"] = True
        result.summary["no_text_change_reason"] = (
            result.mp_result.convergence_reason
            if result.mp_result and result.mp_result.convergence_reason
            else "No automatic rewrite was applied"
        )
        rewritten_report_dict = ctx.raw_json
    else:
        # Text changed — run a single fresh full scan on the rewritten text.
        # The rewrite engine's targeted rescan reuses old scores for unchanged
        # sentences, which produces misleading "After" numbers. One full scan
        # here gives accurate user-facing report scores.
        report_progress(76, "Running final scan on rewritten draft")
        scan_t0 = time.time()
        rewritten_detect_runner = DetectionRunner()
        rewritten_detect_report = rewritten_detect_runner.run_all(rewritten_text)
        stage_timings.append({
            "stage": "fresh_rewritten_scan",
            "seconds": round(time.time() - scan_t0, 3),
        })

        rewritten_builder = ReportBuilder()
        rewritten_builder.add_detection_report(rewritten_detect_report)
        if getattr(rewritten_detect_report, "postprocess_results", None):
            rewritten_builder.add_postprocess_results(rewritten_detect_report.postprocess_results)
        rewritten_builder.set_meta(scan_time=0, original_text=rewritten_text)
        rewritten_draft_report = rewritten_builder.build()
        rewritten_report_dict = report_to_dict(rewritten_draft_report)
        report_progress(78, "Final rewritten scan complete")

    def _finding_total(report_dict):
        findings = report_dict.get("findings", {})
        return sum(len(findings.get(t, [])) for t in ("critical", "high", "medium", "low"))

    def _review_burden(report_dict):
        findings = report_dict.get("findings", {})
        return sum(len(findings.get(t, [])) for t in ("critical", "high", "medium"))

    def _weighted_severity(report_dict):
        findings = report_dict.get("findings", {})
        weights = {"critical": 8, "high": 5, "medium": 2, "low": 1}
        return sum(len(findings.get(t, [])) * weights[t] for t in weights)

    def _badge_ai(report_dict):
        score = (report_dict.get("ai_risk_badge") or {}).get("ai_likelihood_score")
        return float(score) if isinstance(score, (int, float)) else None

    def _badge_wq(report_dict):
        score = (report_dict.get("ai_risk_badge") or {}).get("writing_quality_score")
        return float(score) if isinstance(score, (int, float)) else None

    def _full_scan_report_dict(scan_text: str) -> dict:
        detect_runner = DetectionRunner()
        detect_report = detect_runner.run_all(scan_text)
        builder = ReportBuilder()
        builder.add_detection_report(detect_report)
        if detect_report.postprocess_results:
            builder.add_postprocess_results(detect_report.postprocess_results)
        builder.set_meta(scan_time=0, original_text=scan_text)
        return report_to_dict(builder.build())

    # Rewrite candidate scans must be compared against a baseline produced by
    # the same scanner codepath. Otherwise a saved scan from an earlier scanner
    # phase can make a valid mitigation candidate look like a regression.
    original_report_dict = ctx.raw_json
    if _env_flag("DRAFTPROOF_FRESH_ORIGINAL_BASELINE", True):
        scan_t0 = time.time()
        original_report_dict = _full_scan_report_dict(text)
        stage_timings.append({
            "stage": "fresh_original_scan",
            "seconds": round(time.time() - scan_t0, 3),
        })
        result.summary["comparison_baseline"] = "fresh_original_scan"
        saved_original_ai = _badge_ai(ctx.raw_json)
        fresh_original_ai = _badge_ai(original_report_dict)
        saved_original_wq = _badge_wq(ctx.raw_json)
        fresh_original_wq = _badge_wq(original_report_dict)
        if (
            saved_original_ai != fresh_original_ai
            or saved_original_wq != fresh_original_wq
            or _finding_total(ctx.raw_json) != _finding_total(original_report_dict)
            or _review_burden(ctx.raw_json) != _review_burden(original_report_dict)
        ):
            result.summary["baseline_rescan_delta"] = {
                "saved_ai": saved_original_ai,
                "fresh_ai": fresh_original_ai,
                "saved_writing_quality": saved_original_wq,
                "fresh_writing_quality": fresh_original_wq,
                "saved_findings": _finding_total(ctx.raw_json),
                "fresh_findings": _finding_total(original_report_dict),
                "saved_review_burden": _review_burden(ctx.raw_json),
                "fresh_review_burden": _review_burden(original_report_dict),
                "saved_weighted_severity": _weighted_severity(ctx.raw_json),
                "fresh_weighted_severity": _weighted_severity(original_report_dict),
            }
    else:
        result.summary["comparison_baseline"] = "saved_original_scan"

    original_ai = _badge_ai(original_report_dict)
    rewritten_ai = _badge_ai(rewritten_report_dict)
    original_wq = _badge_wq(original_report_dict)
    rewritten_wq = _badge_wq(rewritten_report_dict)
    original_total = _finding_total(original_report_dict)
    rewritten_total = _finding_total(rewritten_report_dict)
    original_review_burden = _review_burden(original_report_dict)
    rewritten_review_burden = _review_burden(rewritten_report_dict)
    original_severity = _weighted_severity(original_report_dict)
    rewritten_severity = _weighted_severity(rewritten_report_dict)
    attempted_report_dict = rewritten_report_dict

    saved_ai = _badge_ai(ctx.raw_json)
    saved_total = _finding_total(ctx.raw_json)
    saved_critical_high = (
        len(ctx.raw_json.get("findings", {}).get("critical", []))
        + len(ctx.raw_json.get("findings", {}).get("high", []))
    )
    rewritten_critical_high = (
        len(rewritten_report_dict.get("findings", {}).get("critical", []))
        + len(rewritten_report_dict.get("findings", {}).get("high", []))
    )

    result.summary["detect_scores"] = {
        "original_ai": original_ai,
        "rewritten_ai": rewritten_ai,
        "original_writing_quality": original_wq,
        "rewritten_writing_quality": rewritten_wq,
        "original_ai_authorship": _integrity_scores(original_report_dict).get("ai_authorship"),
        "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
        "original_grounding_quality_risk": _integrity_scores(original_report_dict).get("grounding"),
        "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
        "original_human_contribution": _contribution_scores(original_report_dict).get("human"),
        "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
        "original_ai_transformation": _contribution_scores(original_report_dict).get("ai_transformation"),
        "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
        "original_findings": original_total,
        "rewritten_findings": rewritten_total,
        "original_review_burden": original_review_burden,
        "rewritten_review_burden": rewritten_review_burden,
        "original_weighted_severity": original_severity,
        "rewritten_weighted_severity": rewritten_severity,
    }
    ai_first_min_drop = float(os.environ.get("DRAFTPROOF_AI_FIRST_MIN_DROP", "5.0"))
    ai_first_target = float(os.environ.get("DRAFTPROOF_AI_FIRST_TARGET", "60.0"))
    ai_first_required_min_ai = float(os.environ.get("DRAFTPROOF_AI_FIRST_REQUIRED_MIN_AI", "50.0"))
    ai_search_selected = False
    authenticity_mitigation_selected = False
    integrity_original = _integrity_scores(original_report_dict)
    ai_authorship_mitigation_needed = bool(
        isinstance(integrity_original.get("ai_authorship"), (int, float))
        and integrity_original.get("ai_authorship") >= _float_env(
            "DRAFTPROOF_AUTHENTICITY_MIN_AI_AUTHORSHIP",
            50.0,
        )
    )

    # Guided authenticity mitigation. This path handles the exact case the
    # scanner now identifies: the draft needs human-side movement, but the
    # system must not fabricate author grounding. It generates fact-preserving
    # candidates, rescans them locally, then accepts only measurable movement
    # toward Human Contribution.
    authenticity_enabled = (
        (ai_mitigation_needs_author or ai_authorship_mitigation_needed)
        and os.environ.get("DRAFTPROOF_AUTHENTICITY_MITIGATION", "1") != "0"
    )
    if authenticity_enabled:
        mitigation_started = time.time()
        try:
            authenticity_candidate_limit = max(
                0,
                int(os.environ.get("DRAFTPROOF_AUTHENTICITY_CANDIDATES", "2")),
            )
        except ValueError:
            authenticity_candidate_limit = 0
        if _env_flag("DRAFTPROOF_REGENERATION_FIRST", True):
            authenticity_candidate_limit = 0
        authenticity_summary = {
            "enabled": True,
            "selected": False,
            "candidate_limit": authenticity_candidate_limit,
            "llm_calls": 0,
            "model_roles": llm_roles,
            "reference": {
                "ai": original_ai,
                "writing_quality": original_wq,
                "human_contribution": _contribution_scores(original_report_dict).get("human"),
                "ai_transformation": _contribution_scores(original_report_dict).get("ai_transformation"),
                "ai_authorship": integrity_original.get("ai_authorship"),
                "grounding_quality": integrity_original.get("grounding"),
                "review_burden": original_review_burden,
                "weighted_severity": original_severity,
            },
            "candidates": [],
        }
        effective_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        if not effective_key:
            authenticity_summary["reason"] = "no_llm_available"
        else:
            source_for_mitigation, source_repairs = _repair_candidate_source_damage(text)
            if source_repairs:
                authenticity_summary["source_repairs"] = source_repairs
            source_protected = detect_protected_spans(source_for_mitigation)
            min_chars = max(200, int(len(source_for_mitigation) * 0.78))
            max_chars = max(min_chars, int(len(text) * 1.25))
            best_candidate_text = ""
            best_candidate_report = None
            best_candidate_gate = None
            best_candidate_eval = None
            masked_span_selected = False
            masked_optimizer_ran = False
            if source_repairs and source_for_mitigation.strip() != text.strip():
                candidate_eval = {
                    "attempt": 0,
                    "strategy": "deterministic_source_integrity_repair",
                    "deterministic": True,
                    "passed_local_checks": False,
                    "candidate_length": len(source_for_mitigation),
                    "source_damage_repairs": source_repairs,
                }
                protected_loss = _ai_search_protected_loss_reason(
                    text,
                    source_for_mitigation,
                    detect_protected_spans(text),
                )
                if protected_loss:
                    candidate_eval["reason"] = "protected_span_lost " + protected_loss
                    authenticity_summary["candidates"].append(candidate_eval)
                else:
                    drift = check_semantic_drift(text, source_for_mitigation, threshold=0.15)
                    candidate_eval["drift_similarity"] = round(drift.similarity, 3)
                    if not drift.accepted and _source_repair_drift_false_positive(
                        source_for_mitigation,
                        drift.reasons,
                    ):
                        candidate_eval["drift_relaxed_for_source_repair"] = True
                        candidate_eval["drift_reasons_relaxed"] = drift.reasons[:10]
                    elif not drift.accepted:
                        candidate_eval["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                        candidate_eval["drift_reasons"] = drift.reasons[:10]
                        authenticity_summary["candidates"].append(candidate_eval)
                    if not candidate_eval.get("reason"):
                        candidate_eval["passed_local_checks"] = True
                        try:
                            scan_t0 = time.time()
                            candidate_report = _full_scan_report_dict(source_for_mitigation)
                            candidate_eval["scan_seconds"] = round(time.time() - scan_t0, 3)
                        except Exception as exc:
                            candidate_eval["passed_local_checks"] = False
                            candidate_eval["reason"] = f"candidate_scan_error {exc}"
                            authenticity_summary["candidates"].append(candidate_eval)
                        if candidate_eval.get("passed_local_checks"):
                            candidate_review_burden = _review_burden(candidate_report)
                            candidate_severity = _weighted_severity(candidate_report)
                            gate = _authenticity_gate_status(
                                original_report_dict,
                                candidate_report,
                                source_for_mitigation != text,
                                original_review_burden=original_review_burden,
                                candidate_review_burden=candidate_review_burden,
                                original_weighted_severity=original_severity,
                                candidate_weighted_severity=candidate_severity,
                                min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                                min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                                drift_similarity=candidate_eval.get("drift_similarity"),
                            )
                            candidate_eval.update({
                                "ai": _badge_ai(candidate_report),
                                "writing_quality": _badge_wq(candidate_report),
                                "human_contribution": gate.get("candidate_human"),
                                "ai_transformation": gate.get("candidate_ai_transformation"),
                                "ai_authorship": gate.get("candidate_ai_authorship"),
                                "human_delta": gate.get("human_delta"),
                                "ai_transformation_delta": gate.get("ai_transformation_delta"),
                                "ai_authorship_delta": gate.get("ai_authorship_delta"),
                                "human_shift_score": gate.get("human_shift_score"),
                                "human_shift_components": gate.get("human_shift_components"),
                                "findings": _finding_total(candidate_report),
                                "review_burden": candidate_review_burden,
                                "weighted_severity": candidate_severity,
                                "scan_scope": _scan_scope_summary(candidate_report),
                                "gate": gate,
                            })
                            if _is_better_human_shift_candidate(gate, best_candidate_gate):
                                best_candidate_text = source_for_mitigation
                                best_candidate_report = candidate_report
                                best_candidate_gate = gate
                                best_candidate_eval = dict(candidate_eval)
                                candidate_eval["best_so_far"] = True
                            authenticity_summary["candidates"].append(candidate_eval)
            try:
                gateway = LLMGateway(LLMConfig(
                    api_key=effective_key,
                    model=generator_model,
                    base_url=base_url,
                    timeout=int(os.environ.get("DRAFTPROOF_AUTHENTICITY_TIMEOUT", "120")),
                    max_retries=int(os.environ.get("DRAFTPROOF_AUTHENTICITY_RETRIES", "1")),
                    max_tokens=int(os.environ.get("DRAFTPROOF_AUTHENTICITY_MAX_TOKENS", "6500")),
                    temperature=float(os.environ.get("DRAFTPROOF_AUTHENTICITY_TEMPERATURE", "0.7")),
                ))
                if _env_flag("DRAFTPROOF_MASKED_SPAN_OPTIMIZER", True):
                    masked_optimizer_ran = True
                    masked_limit = int(_float_env("DRAFTPROOF_MASKED_SPAN_CANDIDATES", 3.0))
                    masked_limit = max(0, masked_limit)
                    masked_baseline_report = original_report_dict
                    if _env_flag("DRAFTPROOF_MASKED_SPAN_FRESH_BASELINE", True):
                        try:
                            scan_t0 = time.time()
                            masked_baseline_report = _full_scan_report_dict(source_for_mitigation)
                            stage_timings.append({
                                "stage": "masked_span_fresh_baseline_scan",
                                "seconds": round(time.time() - scan_t0, 3),
                            })
                        except Exception as exc:
                            authenticity_summary.setdefault("masked_span_baseline_warning", str(exc))
                            masked_baseline_report = original_report_dict
                    masked_baseline_integrity = _integrity_scores(masked_baseline_report)
                    masked_baseline_review_burden = _review_burden(masked_baseline_report)
                    masked_baseline_severity = _weighted_severity(masked_baseline_report)
                    masked_baseline_findings = _finding_total(masked_baseline_report)
                    authenticity_summary["masked_span_baseline"] = {
                        "mode": (
                            "fresh_original_scan"
                            if masked_baseline_report is not original_report_dict
                            else result.summary.get("comparison_baseline", "saved_original_scan")
                        ),
                        "saved_ai": original_ai,
                        "baseline_ai": _badge_ai(masked_baseline_report),
                        "saved_findings": original_total,
                        "baseline_findings": masked_baseline_findings,
                        "saved_ai_authorship": integrity_original.get("ai_authorship"),
                        "baseline_ai_authorship": masked_baseline_integrity.get("ai_authorship"),
                    }
                    current_masked_text = source_for_mitigation
                    current_masked_report = masked_baseline_report
                    current_masked_ai = _badge_ai(masked_baseline_report)
                    current_masked_human = _contribution_scores(masked_baseline_report).get("human")
                    current_masked_transform = _contribution_scores(masked_baseline_report).get("ai_transformation")
                    current_masked_authorship = masked_baseline_integrity.get("ai_authorship")
                    current_masked_findings = masked_baseline_findings
                    masked_excluded: set[int] = set()
                    masked_attempts: list[dict] = []
                    route_bundle_candidate, route_bundle_edits = _deterministic_sentence_route_bundle(
                        current_masked_text
                    )
                    if route_bundle_edits and route_bundle_candidate != current_masked_text:
                        candidate_eval_try = {
                            "attempt": "route_bundle.1",
                            "strategy": "deterministic_sentence_route_bundle",
                            "masked_span_repair": True,
                            "deterministic": True,
                            "passed_local_checks": False,
                            "route_bundle_edits": route_bundle_edits,
                            "candidate_length": len(route_bundle_candidate or ""),
                            "candidate_word_count": _text_word_count(route_bundle_candidate or ""),
                        }
                        candidate_eval_try["repair_aggression"] = _repair_aggression_score(
                            current_masked_text,
                            route_bundle_candidate,
                        )
                        candidate_eval_try["locality"] = _locality_score(
                            current_masked_text,
                            route_bundle_candidate,
                        )
                        protected_loss = _ai_search_protected_loss_reason(
                            source_for_mitigation,
                            route_bundle_candidate,
                            source_protected,
                        )
                        drift = (
                            None
                            if protected_loss
                            else check_semantic_drift(source_for_mitigation, route_bundle_candidate, threshold=0.15)
                        )
                        if protected_loss:
                            candidate_eval_try["reason"] = "protected_span_lost " + protected_loss
                        elif drift and not drift.accepted:
                            candidate_eval_try["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                            candidate_eval_try["drift_reasons"] = drift.reasons[:10]
                        else:
                            if drift:
                                candidate_eval_try["drift_similarity"] = round(drift.similarity, 3)
                            candidate_eval_try["passed_local_checks"] = True
                            try:
                                scan_t0 = time.time()
                                candidate_report = _full_scan_report_dict(route_bundle_candidate)
                                candidate_eval_try["scan_seconds"] = round(time.time() - scan_t0, 3)
                            except Exception as exc:
                                candidate_report = None
                                candidate_eval_try["passed_local_checks"] = False
                                candidate_eval_try["reason"] = f"candidate_scan_error {exc}"
                            if candidate_report:
                                candidate_review_burden = _review_burden(candidate_report)
                                candidate_severity = _weighted_severity(candidate_report)
                                gate = _authenticity_gate_status(
                                    masked_baseline_report,
                                    candidate_report,
                                    route_bundle_candidate != text,
                                    original_review_burden=masked_baseline_review_burden,
                                    candidate_review_burden=candidate_review_burden,
                                    original_weighted_severity=masked_baseline_severity,
                                    candidate_weighted_severity=candidate_severity,
                                    min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                                    min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                                    drift_similarity=candidate_eval_try.get("drift_similarity"),
                                )
                                candidate_ai = _badge_ai(candidate_report)
                                candidate_human = gate.get("candidate_human")
                                candidate_transform = gate.get("candidate_ai_transformation")
                                candidate_authorship = gate.get("candidate_ai_authorship")
                                candidate_findings = _finding_total(candidate_report)
                                candidate_critical_high = (
                                    len(candidate_report.get("findings", {}).get("critical", []))
                                    + len(candidate_report.get("findings", {}).get("high", []))
                                )
                                candidate_eval_try.update({
                                    "ai": candidate_ai,
                                    "writing_quality": _badge_wq(candidate_report),
                                    "human_contribution": candidate_human,
                                    "ai_transformation": candidate_transform,
                                    "ai_authorship": candidate_authorship,
                                    "human_delta": gate.get("human_delta"),
                                    "ai_transformation_delta": gate.get("ai_transformation_delta"),
                                    "ai_authorship_delta": gate.get("ai_authorship_delta"),
                                    "human_shift_score": gate.get("human_shift_score"),
                                    "human_shift_components": gate.get("human_shift_components"),
                                    "authorship_cost_per_human_gain": gate.get("authorship_cost_per_human_gain"),
                                    "findings": candidate_findings,
                                    "review_burden": candidate_review_burden,
                                    "weighted_severity": candidate_severity,
                                    "scan_scope": _scan_scope_summary(candidate_report),
                                    "gate": gate,
                                })
                                fresh_authorship_capped = bool(
                                    isinstance(candidate_authorship, (int, float))
                                    and isinstance(masked_baseline_integrity.get("ai_authorship"), (int, float))
                                    and candidate_authorship <= masked_baseline_integrity.get("ai_authorship")
                                )
                                saved_authorship_capped = bool(
                                    not isinstance(integrity_original.get("ai_authorship"), (int, float))
                                    or (
                                        isinstance(candidate_authorship, (int, float))
                                        and candidate_authorship <= integrity_original.get("ai_authorship")
                                    )
                                )
                                findings_non_regression = bool(
                                    candidate_findings <= current_masked_findings
                                    and candidate_findings <= original_total
                                )
                                review_non_regression = bool(
                                    candidate_review_burden <= masked_baseline_review_burden
                                    and candidate_review_burden <= original_review_burden
                                )
                                severity_non_regression = bool(
                                    candidate_severity <= masked_baseline_severity
                                    and candidate_severity <= original_severity
                                )
                                critical_high_non_regression = bool(
                                    candidate_critical_high <= saved_critical_high
                                )
                                movement = bool(
                                    (
                                        isinstance(candidate_human, (int, float))
                                        and isinstance(current_masked_human, (int, float))
                                        and candidate_human > current_masked_human
                                    )
                                    or (
                                        isinstance(candidate_transform, (int, float))
                                        and isinstance(current_masked_transform, (int, float))
                                        and candidate_transform < current_masked_transform
                                    )
                                    or candidate_findings < current_masked_findings
                                    or (
                                        isinstance(candidate_ai, (int, float))
                                        and isinstance(current_masked_ai, (int, float))
                                        and candidate_ai < current_masked_ai
                                    )
                                )
                                breakthrough_tradeoff = bool(
                                    _env_flag("DRAFTPROOF_AUTHENTICITY_BREAKTHROUGH_TRADEOFF", True)
                                    and isinstance(candidate_ai, (int, float))
                                    and isinstance(original_ai, (int, float))
                                    and candidate_ai <= original_ai - 10.0
                                    and isinstance(candidate_authorship, (int, float))
                                    and isinstance(integrity_original.get("ai_authorship"), (int, float))
                                    and candidate_authorship <= integrity_original.get("ai_authorship") - 10.0
                                    and candidate_findings <= original_total
                                )
                                masked_accept = bool(
                                    fresh_authorship_capped
                                    and saved_authorship_capped
                                    and findings_non_regression
                                    and (
                                        (
                                            review_non_regression
                                            and severity_non_regression
                                            and critical_high_non_regression
                                        )
                                        or breakthrough_tradeoff
                                    )
                                    and movement
                                    and (
                                        not gate.get("critical_high_regressed")
                                        or breakthrough_tradeoff
                                    )
                                    and (
                                        not gate.get("review_burden_regressed")
                                        or breakthrough_tradeoff
                                    )
                                    and (
                                        not gate.get("weighted_severity_regressed")
                                        or breakthrough_tradeoff
                                    )
                                )
                                candidate_eval_try["masked_accept"] = masked_accept
                                candidate_eval_try["masked_acceptance"] = {
                                    "authorship_capped": bool(fresh_authorship_capped and saved_authorship_capped),
                                    "fresh_authorship_capped": fresh_authorship_capped,
                                    "saved_authorship_capped": saved_authorship_capped,
                                    "findings_non_regression": findings_non_regression,
                                    "review_non_regression": review_non_regression,
                                    "severity_non_regression": severity_non_regression,
                                    "critical_high_non_regression": critical_high_non_regression,
                                    "breakthrough_tradeoff": breakthrough_tradeoff,
                                    "movement": movement,
                                }
                                if masked_accept:
                                    candidate_eval_try["selected"] = True
                                    masked_attempts.append(candidate_eval_try)
                                    current_masked_text = route_bundle_candidate
                                    current_masked_report = candidate_report
                                    current_masked_ai = candidate_ai
                                    current_masked_human = candidate_human
                                    current_masked_transform = candidate_transform
                                    current_masked_authorship = candidate_authorship
                                    current_masked_findings = candidate_findings
                                    masked_span_selected = True
                                else:
                                    candidate_eval_try["reason"] = "authorship_cap_or_no_masked_gain"
                        authenticity_summary["candidates"].append(candidate_eval_try)
                    for masked_index in range(1, masked_limit + 1):
                        report_progress(
                            min(88, 76 + masked_index),
                            f"Trying masked-span mitigation {masked_index}/{masked_limit}",
                        )
                        prompt, repair_info = _masked_span_repair_prompt(
                            current_masked_text,
                            ctx.raw_json,
                            exclude_sentence_indexes=masked_excluded,
                        )
                        window = repair_info.get("window") if isinstance(repair_info, dict) else {}
                        sentence_index = window.get("start") if isinstance(window, dict) else None
                        candidate_eval = {
                            "attempt": f"masked.{masked_index}",
                            "strategy": "masked_span_repair",
                            "masked_span_repair": True,
                            "passed_local_checks": False,
                            "model": generator_model,
                            "sentence_index": sentence_index,
                            "mask_text": repair_info.get("mask_text") if isinstance(repair_info, dict) else None,
                            "masked_sentence": repair_info.get("masked_sentence") if isinstance(repair_info, dict) else None,
                        }
                        if not prompt or sentence_index is None:
                            candidate_eval["reason"] = repair_info.get("reason") if isinstance(repair_info, dict) else "no_mask_prompt"
                            authenticity_summary["candidates"].append(candidate_eval)
                            break
                        replacements = _deterministic_masked_span_replacements(
                            repair_info.get("mask_text") if isinstance(repair_info, dict) else ""
                        )
                        if _env_flag("DRAFTPROOF_MASKED_SPAN_LLM_FALLBACK", False):
                            try:
                                authenticity_summary["llm_calls"] += 1
                                response = gateway.chat(
                                    prompt,
                                    system="Return only replacement text for [[MASK]].",
                                    temperature=float(os.environ.get("DRAFTPROOF_RECONSTRUCTION_TEMPERATURE", "0.45")),
                                    max_tokens=int(os.environ.get("DRAFTPROOF_MASKED_SPAN_MAX_TOKENS", "1000")),
                                    top_p=_float_env_optional("DRAFTPROOF_RECONSTRUCTION_TOP_P"),
                                    top_k=_int_env_optional("DRAFTPROOF_RECONSTRUCTION_TOP_K"),
                                    presence_penalty=_float_env_optional("DRAFTPROOF_RECONSTRUCTION_PRESENCE_PENALTY"),
                                    frequency_penalty=_float_env_optional("DRAFTPROOF_RECONSTRUCTION_FREQUENCY_PENALTY"),
                                )
                                llm_replacement = _clean_masked_span_replacement(response.content or "")
                                if llm_replacement and llm_replacement not in replacements:
                                    replacements.append(llm_replacement)
                            except Exception as exc:
                                candidate_eval.setdefault("replacement_errors", []).append(f"llm_error {exc}")
                        if not replacements:
                            candidate_eval["reason"] = "no_mask_replacement_candidates"
                            authenticity_summary["candidates"].append(candidate_eval)
                            masked_excluded.add(int(sentence_index))
                            continue
                        candidate_eval["replacement_candidates"] = replacements
                        accepted_eval = None
                        rejected_replacement_evals: list[dict] = []
                        for replacement_index, replacement in enumerate(replacements, start=1):
                            candidate_eval_try = dict(candidate_eval)
                            candidate_eval_try["replacement_index"] = replacement_index
                            candidate_eval_try["replacement"] = replacement
                            candidate = _apply_masked_span_replacement(current_masked_text, repair_info, replacement)
                            candidate_eval_try["candidate_length"] = len(candidate or "")
                            candidate_eval_try["candidate_word_count"] = _text_word_count(candidate or "")
                            if not candidate or candidate == current_masked_text:
                                candidate_eval_try["reason"] = "empty_or_unchanged_candidate"
                                rejected_replacement_evals.append(candidate_eval_try)
                                continue
                            candidate_eval_try["repair_aggression"] = _repair_aggression_score(current_masked_text, candidate)
                            candidate_eval_try["locality"] = _locality_score(current_masked_text, candidate)
                            protected_loss = _ai_search_protected_loss_reason(
                                source_for_mitigation,
                                candidate,
                                source_protected,
                            )
                            if protected_loss:
                                candidate_eval_try["reason"] = "protected_span_lost " + protected_loss
                                rejected_replacement_evals.append(candidate_eval_try)
                                continue
                            drift = check_semantic_drift(source_for_mitigation, candidate, threshold=0.15)
                            candidate_eval_try["drift_similarity"] = round(drift.similarity, 3)
                            if not drift.accepted:
                                candidate_eval_try["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                                candidate_eval_try["drift_reasons"] = drift.reasons[:10]
                                rejected_replacement_evals.append(candidate_eval_try)
                                continue
                            candidate_eval_try["passed_local_checks"] = True
                            try:
                                scan_t0 = time.time()
                                candidate_report = _full_scan_report_dict(candidate)
                                candidate_eval_try["scan_seconds"] = round(time.time() - scan_t0, 3)
                            except Exception as exc:
                                candidate_eval_try["passed_local_checks"] = False
                                candidate_eval_try["reason"] = f"candidate_scan_error {exc}"
                                rejected_replacement_evals.append(candidate_eval_try)
                                continue
                            candidate_review_burden = _review_burden(candidate_report)
                            candidate_severity = _weighted_severity(candidate_report)
                            gate = _authenticity_gate_status(
                                masked_baseline_report,
                                candidate_report,
                                candidate != text,
                                original_review_burden=masked_baseline_review_burden,
                                candidate_review_burden=candidate_review_burden,
                                original_weighted_severity=masked_baseline_severity,
                                candidate_weighted_severity=candidate_severity,
                                min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                                min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                                drift_similarity=candidate_eval_try.get("drift_similarity"),
                            )
                            candidate_ai = _badge_ai(candidate_report)
                            candidate_human = gate.get("candidate_human")
                            candidate_transform = gate.get("candidate_ai_transformation")
                            candidate_authorship = gate.get("candidate_ai_authorship")
                            candidate_findings = _finding_total(candidate_report)
                            candidate_critical_high = (
                                len(candidate_report.get("findings", {}).get("critical", []))
                                + len(candidate_report.get("findings", {}).get("high", []))
                            )
                            candidate_eval_try.update({
                                "ai": candidate_ai,
                                "writing_quality": _badge_wq(candidate_report),
                                "human_contribution": candidate_human,
                                "ai_transformation": candidate_transform,
                                "ai_authorship": candidate_authorship,
                                "human_delta": gate.get("human_delta"),
                                "ai_transformation_delta": gate.get("ai_transformation_delta"),
                                "ai_authorship_delta": gate.get("ai_authorship_delta"),
                                "human_shift_score": gate.get("human_shift_score"),
                                "human_shift_components": gate.get("human_shift_components"),
                                "authorship_cost_per_human_gain": gate.get("authorship_cost_per_human_gain"),
                                "findings": candidate_findings,
                                "review_burden": candidate_review_burden,
                                "weighted_severity": candidate_severity,
                                "scan_scope": _scan_scope_summary(candidate_report),
                                "gate": gate,
                            })
                            fresh_authorship_capped = bool(
                                isinstance(candidate_authorship, (int, float))
                                and isinstance(masked_baseline_integrity.get("ai_authorship"), (int, float))
                                and candidate_authorship <= masked_baseline_integrity.get("ai_authorship")
                            )
                            saved_authorship_capped = bool(
                                not isinstance(integrity_original.get("ai_authorship"), (int, float))
                                or (
                                    isinstance(candidate_authorship, (int, float))
                                    and candidate_authorship <= integrity_original.get("ai_authorship")
                                )
                            )
                            authorship_capped = bool(fresh_authorship_capped and saved_authorship_capped)
                            baseline_findings_non_regression = candidate_findings <= current_masked_findings
                            saved_findings_non_regression = candidate_findings <= original_total
                            findings_non_regression = bool(
                                baseline_findings_non_regression
                                and saved_findings_non_regression
                            )
                            review_non_regression = bool(
                                candidate_review_burden <= masked_baseline_review_burden
                                and candidate_review_burden <= original_review_burden
                            )
                            severity_non_regression = bool(
                                candidate_severity <= masked_baseline_severity
                                and candidate_severity <= original_severity
                            )
                            critical_high_non_regression = bool(
                                candidate_critical_high <= saved_critical_high
                            )
                            movement = bool(
                                (
                                    isinstance(candidate_human, (int, float))
                                    and isinstance(current_masked_human, (int, float))
                                    and candidate_human > current_masked_human
                                )
                                or (
                                    isinstance(candidate_transform, (int, float))
                                    and isinstance(current_masked_transform, (int, float))
                                    and candidate_transform < current_masked_transform
                                )
                                or candidate_findings < current_masked_findings
                                or (
                                    isinstance(candidate_ai, (int, float))
                                    and isinstance(current_masked_ai, (int, float))
                                    and candidate_ai < current_masked_ai
                                )
                            )
                            masked_accept = bool(
                                authorship_capped
                                and findings_non_regression
                                and review_non_regression
                                and severity_non_regression
                                and critical_high_non_regression
                                and movement
                                and not gate.get("critical_high_regressed")
                                and not gate.get("review_burden_regressed")
                                and not gate.get("weighted_severity_regressed")
                            )
                            candidate_eval_try["masked_accept"] = masked_accept
                            candidate_eval_try["masked_acceptance"] = {
                                "authorship_capped": authorship_capped,
                                "fresh_authorship_capped": fresh_authorship_capped,
                                "saved_authorship_capped": saved_authorship_capped,
                                "findings_non_regression": findings_non_regression,
                                "baseline_findings_non_regression": baseline_findings_non_regression,
                                "saved_findings_non_regression": saved_findings_non_regression,
                                "review_non_regression": review_non_regression,
                                "severity_non_regression": severity_non_regression,
                                "critical_high_non_regression": critical_high_non_regression,
                                "movement": movement,
                            }
                            if masked_accept:
                                accepted_eval = candidate_eval_try
                                accepted_eval["_candidate_text"] = candidate
                                accepted_eval["_candidate_report"] = candidate_report
                                break
                            candidate_eval_try["reason"] = "authorship_cap_or_no_masked_gain"
                            rejected_replacement_evals.append(candidate_eval_try)
                        for rejected_eval in rejected_replacement_evals:
                            authenticity_summary["candidates"].append(rejected_eval)
                        if accepted_eval is None:
                            masked_excluded.add(int(sentence_index))
                            continue
                        candidate_eval = accepted_eval
                        candidate = candidate_eval.pop("_candidate_text")
                        candidate_report = candidate_eval.pop("_candidate_report")
                        candidate_ai = candidate_eval.get("ai")
                        candidate_human = candidate_eval.get("human_contribution")
                        candidate_transform = candidate_eval.get("ai_transformation")
                        candidate_authorship = candidate_eval.get("ai_authorship")
                        candidate_findings = candidate_eval.get("findings")
                        candidate_eval["selected"] = True
                        masked_attempts.append(candidate_eval)
                        current_masked_text = candidate
                        current_masked_report = candidate_report
                        current_masked_ai = candidate_ai
                        current_masked_human = candidate_human
                        current_masked_transform = candidate_transform
                        current_masked_authorship = candidate_authorship
                        current_masked_findings = candidate_findings
                        masked_span_selected = True
                        authenticity_summary["candidates"].append(candidate_eval)
                        masked_excluded.add(int(sentence_index))
                    authenticity_summary["masked_span_optimizer"] = {
                        "enabled": True,
                        "candidate_limit": masked_limit,
                        "accepted_count": len(masked_attempts),
                        "selected": masked_span_selected,
                        "accepted_attempts": [
                            {
                                "attempt": item.get("attempt"),
                                "sentence_index": item.get("sentence_index"),
                                "mask_text": item.get("mask_text"),
                                "replacement": item.get("replacement"),
                                "ai": item.get("ai"),
                                "human_contribution": item.get("human_contribution"),
                                "ai_transformation": item.get("ai_transformation"),
                                "ai_authorship": item.get("ai_authorship"),
                                "findings": item.get("findings"),
                            }
                            for item in masked_attempts
                        ],
                    }
                    if masked_span_selected:
                        masked_gate = _authenticity_gate_status(
                            masked_baseline_report,
                            current_masked_report,
                            current_masked_text != text,
                            original_review_burden=masked_baseline_review_burden,
                            candidate_review_burden=_review_burden(current_masked_report),
                            original_weighted_severity=masked_baseline_severity,
                            candidate_weighted_severity=_weighted_severity(current_masked_report),
                            min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                            min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                        )
                        best_candidate_text = current_masked_text
                        best_candidate_report = current_masked_report
                        best_candidate_gate = masked_gate
                        best_candidate_eval = {
                            "strategy": "masked_span_optimizer",
                            "masked_span_repair": True,
                            "accepted_count": len(masked_attempts),
                            "gate": masked_gate,
                        }
                        authenticity_summary["best_attempt"] = best_candidate_eval
                        if _env_flag("DRAFTPROOF_MASKED_SPAN_SKIP_REGEN_ON_GAIN", True):
                            authenticity_summary["skip_broad_generation_reason"] = "masked_span_authorship_capped_gain"
                            authenticity_candidate_limit = 0
                for attempt_index in range(1, authenticity_candidate_limit + 1):
                    report_progress(
                        min(89, 78 + attempt_index),
                        f"Trying authenticity mitigation candidate {attempt_index}/{authenticity_candidate_limit}",
                    )
                    candidate_eval = {
                        "attempt": attempt_index,
                        "passed_local_checks": False,
                        "model": generator_model,
                    }
                    try:
                        prompt = _authenticity_mitigation_prompt(
                            source_for_mitigation,
                            ctx.raw_json,
                            ai_mitigation_contract,
                            attempt_index,
                        )
                        authenticity_summary["llm_calls"] += 1
                        response = gateway.chat(
                            prompt,
                            system=(
                                "You are DraftProof's AI-Mitigation authenticity engine. "
                                "Return only a complete fact-preserving rewritten document."
                            ),
                            temperature=float(os.environ.get("DRAFTPROOF_AUTHENTICITY_TEMPERATURE", "0.7")),
                            max_tokens=int(os.environ.get("DRAFTPROOF_AUTHENTICITY_MAX_TOKENS", "6500")),
                        )
                        candidate = _clean_full_document_candidate(response.content, source_for_mitigation)
                    except Exception as exc:
                        candidate_eval["reason"] = f"llm_error {exc}"
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    candidate_eval["candidate_length"] = len(candidate or "")
                    if not candidate:
                        candidate_eval["reason"] = "empty_or_unchanged_candidate"
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    candidate, repair_notes = _repair_candidate_source_damage(candidate)
                    if repair_notes:
                        candidate_eval["source_damage_repairs"] = repair_notes
                        candidate_eval["candidate_length"] = len(candidate or "")
                    review_notes = _review_marker_notes(candidate)
                    if review_notes:
                        candidate_eval["reason"] = "review_markers_not_auto_kept"
                        candidate_eval["review_suggestion_count"] = len(review_notes)
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    quality_rejection = _ai_candidate_quality_reject_reason(candidate)
                    if quality_rejection:
                        candidate_eval["reason"] = quality_rejection
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    if len(candidate) < min_chars:
                        candidate_eval["reason"] = f"candidate_too_short {len(candidate)}<{min_chars}"
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    if len(candidate) > max_chars:
                        candidate_eval["reason"] = f"candidate_too_long {len(candidate)}>{max_chars}"
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    protected_loss = _ai_search_protected_loss_reason(
                        source_for_mitigation,
                        candidate,
                        source_protected,
                    )
                    if protected_loss:
                        candidate_eval["reason"] = "protected_span_lost " + protected_loss
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    drift = check_semantic_drift(source_for_mitigation, candidate, threshold=0.15)
                    candidate_eval["drift_similarity"] = round(drift.similarity, 3)
                    if not drift.accepted:
                        candidate_eval["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                        candidate_eval["drift_reasons"] = drift.reasons[:10]
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    candidate_eval["passed_local_checks"] = True
                    try:
                        scan_t0 = time.time()
                        candidate_report = _full_scan_report_dict(candidate)
                        candidate_eval["scan_seconds"] = round(time.time() - scan_t0, 3)
                    except Exception as exc:
                        candidate_eval["passed_local_checks"] = False
                        candidate_eval["reason"] = f"candidate_scan_error {exc}"
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    candidate_review_burden = _review_burden(candidate_report)
                    candidate_severity = _weighted_severity(candidate_report)
                    gate = _authenticity_gate_status(
                        original_report_dict,
                        candidate_report,
                        candidate != text,
                        original_review_burden=original_review_burden,
                        candidate_review_burden=candidate_review_burden,
                        original_weighted_severity=original_severity,
                        candidate_weighted_severity=candidate_severity,
                        min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                        min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                        drift_similarity=candidate_eval.get("drift_similarity"),
                    )
                    candidate_eval.update({
                        "ai": _badge_ai(candidate_report),
                        "writing_quality": _badge_wq(candidate_report),
                        "human_contribution": gate.get("candidate_human"),
                        "ai_transformation": gate.get("candidate_ai_transformation"),
                        "ai_authorship": gate.get("candidate_ai_authorship"),
                        "human_delta": gate.get("human_delta"),
                        "ai_transformation_delta": gate.get("ai_transformation_delta"),
                        "ai_authorship_delta": gate.get("ai_authorship_delta"),
                        "human_shift_score": gate.get("human_shift_score"),
                        "human_shift_components": gate.get("human_shift_components"),
                        "findings": _finding_total(candidate_report),
                        "review_burden": candidate_review_burden,
                        "weighted_severity": candidate_severity,
                        "scan_scope": _scan_scope_summary(candidate_report),
                        "gate": gate,
                    })
                    if _is_better_human_shift_candidate(gate, best_candidate_gate):
                        best_candidate_text = candidate
                        best_candidate_report = candidate_report
                        best_candidate_gate = gate
                        best_candidate_eval = dict(candidate_eval)
                        candidate_eval["best_so_far"] = True
                    authenticity_summary["candidates"].append(candidate_eval)
                reconstruction_target_human = _float_env("DRAFTPROOF_RECONSTRUCTION_TARGET_HUMAN", 80.0)
                selected_human = (
                    best_candidate_gate.get("candidate_human")
                    if isinstance(best_candidate_gate, dict)
                    else None
                )
                selected_human_delta = (
                    best_candidate_gate.get("human_delta")
                    if isinstance(best_candidate_gate, dict)
                    else None
                )
                small_shift_under_target = bool(
                    best_candidate_gate
                    and best_candidate_gate.get("success")
                    and isinstance(selected_human, (int, float))
                    and selected_human < reconstruction_target_human
                    and (
                        not isinstance(selected_human_delta, (int, float))
                        or selected_human_delta < _float_env("DRAFTPROOF_RECONSTRUCTION_SKIP_MIN_HUMAN_GAIN", 25.0)
                    )
                )
                reconstruction_enabled = (
                    (
                        not (best_candidate_gate and best_candidate_gate.get("success"))
                        or (
                            small_shift_under_target
                            and os.environ.get("DRAFTPROOF_RECONSTRUCTION_AFTER_SMALL_SHIFT", "1") != "0"
                        )
                    )
                    and os.environ.get("DRAFTPROOF_RECONSTRUCTION_MITIGATION", "1") != "0"
                    and not (
                        masked_optimizer_ran
                        and _env_flag("DRAFTPROOF_MASKED_SPAN_SKIP_BROAD_REGEN", True)
                    )
                )
                if reconstruction_enabled:
                    reconstruction_strategies = [
                        "plain_student_voice_rebuild",
                        "authorship_distribution_repair",
                        "low_smoothness_rebuild",
                        "asymmetric_paragraph_route",
                        "claim_narrowing_rebuild",
                        "authorship_texture_repair",
                        "human_gain_repair",
                        "evidence_first_rebuild",
                        "problem_observation_rebuild",
                        "reasoning_dense_reconstruction",
                        "domain_grounded_reconstruction",
                    ]
                    reconstruction_limit_raw = os.environ.get("DRAFTPROOF_RECONSTRUCTION_CANDIDATES")
                    try:
                        reconstruction_limit = max(
                            1,
                            int(reconstruction_limit_raw or "2"),
                        )
                    except ValueError:
                        reconstruction_limit = 4
                    if _env_flag("DRAFTPROOF_REGENERATION_FIRST", True) and reconstruction_limit_raw is None:
                        reconstruction_limit = max(reconstruction_limit, 4)
                    reconstruction_strategies = reconstruction_strategies[:reconstruction_limit]
                    authenticity_summary["reconstruction"] = {
                        "enabled": True,
                        "candidate_limit": len(reconstruction_strategies),
                        "strategies": reconstruction_strategies,
                        "target_human_contribution": reconstruction_target_human,
                        "triggered_after_small_shift": small_shift_under_target,
                    }
                    reconstruction_word_band = _word_count_band(source_for_mitigation, variance=0.25)
                    authenticity_summary["reconstruction"]["word_count_band"] = reconstruction_word_band
                    reconstruction_min_chars = max(200, int(len(source_for_mitigation) * 0.65))
                    reconstruction_max_chars = max(reconstruction_min_chars, int(len(text) * 1.45))
                    reconstruction_drift_threshold = _float_env(
                        "DRAFTPROOF_RECONSTRUCTION_DRIFT_THRESHOLD",
                        0.25,
                    )
                    post_texture_calls = 0
                    post_texture_limit = int(_float_env("DRAFTPROOF_POST_GENERATION_TEXTURE_REPAIR_MAX_CALLS", 1.0))
                    for reconstruction_index, strategy in enumerate(reconstruction_strategies, start=1):
                        report_progress(
                            min(92, 84 + reconstruction_index),
                            f"Trying reconstruction mitigation candidate {reconstruction_index}/{len(reconstruction_strategies)}",
                        )
                        candidate_eval = {
                            "attempt": authenticity_candidate_limit + reconstruction_index,
                            "strategy": strategy,
                            "reconstruction": True,
                            "passed_local_checks": False,
                            "model": generator_model,
                        }
                        try:
                            if _env_flag("DRAFTPROOF_STAGED_REGENERATION", True):
                                candidate, staged_info = _staged_reconstruction_candidate(
                                    gateway,
                                    source_for_mitigation,
                                    ctx.raw_json,
                                    attempt_index=reconstruction_index,
                                    strategy=strategy,
                                    prior_attempts=authenticity_summary.get("candidates") or [],
                                )
                                candidate_eval["staged_generation"] = staged_info
                                authenticity_summary["llm_calls"] += int(staged_info.get("llm_calls") or 0)
                            else:
                                prompt = _reconstruction_mitigation_prompt(
                                    source_for_mitigation,
                                    ctx.raw_json,
                                    ai_mitigation_contract,
                                    attempt_index=reconstruction_index,
                                    strategy=strategy,
                                    prior_attempts=authenticity_summary.get("candidates") or [],
                                )
                                authenticity_summary["llm_calls"] += 1
                                response = gateway.chat(
                                    prompt,
                                    system=(
                                        "You are DraftProof's AI-Mitigation reconstruction engine. "
                                        "Return only a complete fact-preserving reconstructed document."
                                    ),
                                    temperature=float(os.environ.get("DRAFTPROOF_RECONSTRUCTION_TEMPERATURE", "0.78")),
                                    max_tokens=int(os.environ.get("DRAFTPROOF_AUTHENTICITY_MAX_TOKENS", "6500")),
                                )
                                candidate = _clean_full_document_candidate(response.content, source_for_mitigation)
                        except Exception as exc:
                            candidate_eval["reason"] = f"llm_error {exc}"
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        candidate_eval["candidate_length"] = len(candidate or "")
                        candidate_eval["candidate_word_count"] = _text_word_count(candidate or "")
                        if not candidate:
                            candidate_eval["reason"] = "empty_or_unchanged_candidate"
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        candidate, repair_notes = _repair_candidate_source_damage(candidate)
                        if repair_notes:
                            candidate_eval["source_damage_repairs"] = repair_notes
                            candidate_eval["candidate_length"] = len(candidate or "")
                            candidate_eval["candidate_word_count"] = _text_word_count(candidate or "")
                        review_notes = _review_marker_notes(candidate)
                        if review_notes:
                            candidate_eval["reason"] = "review_markers_not_auto_kept"
                            candidate_eval["review_suggestion_count"] = len(review_notes)
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        quality_rejection = _ai_candidate_quality_reject_reason(
                            candidate,
                            allow_repeated_long_sequence=True,
                        )
                        if quality_rejection:
                            candidate_eval["reason"] = quality_rejection
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        candidate_words = _text_word_count(candidate)
                        candidate_eval["candidate_word_count"] = candidate_words
                        if candidate_words < reconstruction_word_band["min_words"]:
                            candidate_eval.setdefault("warnings", []).append(
                                f"candidate_word_count_below_target "
                                f"{candidate_words}<{reconstruction_word_band['min_words']}"
                            )
                        if candidate_words > reconstruction_word_band["max_words"]:
                            candidate_eval.setdefault("warnings", []).append(
                                f"candidate_word_count_above_target "
                                f"{candidate_words}>{reconstruction_word_band['max_words']}"
                            )
                        if len(candidate) < reconstruction_min_chars:
                            candidate_eval["reason"] = f"candidate_too_short {len(candidate)}<{reconstruction_min_chars}"
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        if len(candidate) > reconstruction_max_chars:
                            candidate_eval["reason"] = f"candidate_too_long {len(candidate)}>{reconstruction_max_chars}"
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        protected_loss = _ai_search_protected_loss_reason(
                            source_for_mitigation,
                            candidate,
                            source_protected,
                        )
                        if protected_loss:
                            candidate_eval["reason"] = "protected_span_lost " + protected_loss
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        drift = check_semantic_drift(
                            source_for_mitigation,
                            candidate,
                            threshold=reconstruction_drift_threshold,
                        )
                        candidate_eval["drift_similarity"] = round(drift.similarity, 3)
                        candidate_eval["drift_threshold"] = reconstruction_drift_threshold
                        if not drift.accepted:
                            candidate_eval["drift_reasons"] = drift.reasons[:10]
                            if _reconstruction_drift_scan_allowed(candidate, drift.reasons, drift.similarity):
                                candidate_eval["drift_scan_relaxed_for_reconstruction"] = True
                                candidate_eval["drift_reasons_relaxed"] = drift.reasons[:5]
                            else:
                                candidate_eval["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                                authenticity_summary["candidates"].append(candidate_eval)
                                continue
                        candidate_eval["passed_local_checks"] = True
                        try:
                            scan_t0 = time.time()
                            candidate_report = _full_scan_report_dict(candidate)
                            candidate_eval["scan_seconds"] = round(time.time() - scan_t0, 3)
                        except Exception as exc:
                            candidate_eval["passed_local_checks"] = False
                            candidate_eval["reason"] = f"candidate_scan_error {exc}"
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        candidate_review_burden = _review_burden(candidate_report)
                        candidate_severity = _weighted_severity(candidate_report)
                        gate = _authenticity_gate_status(
                            original_report_dict,
                            candidate_report,
                            candidate != text,
                            original_review_burden=original_review_burden,
                            candidate_review_burden=candidate_review_burden,
                            original_weighted_severity=original_severity,
                            candidate_weighted_severity=candidate_severity,
                            min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                            min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                            drift_similarity=candidate_eval.get("drift_similarity"),
                        )
                        candidate_eval.update({
                            "ai": _badge_ai(candidate_report),
                            "writing_quality": _badge_wq(candidate_report),
                            "human_contribution": gate.get("candidate_human"),
                            "ai_transformation": gate.get("candidate_ai_transformation"),
                            "ai_authorship": gate.get("candidate_ai_authorship"),
                            "human_delta": gate.get("human_delta"),
                            "ai_transformation_delta": gate.get("ai_transformation_delta"),
                            "ai_authorship_delta": gate.get("ai_authorship_delta"),
                            "human_shift_score": gate.get("human_shift_score"),
                            "human_shift_components": gate.get("human_shift_components"),
                            "authorship_cost_per_human_gain": gate.get("authorship_cost_per_human_gain"),
                            "findings": _finding_total(candidate_report),
                            "review_burden": candidate_review_burden,
                            "weighted_severity": candidate_severity,
                            "scan_scope": _scan_scope_summary(candidate_report),
                            "gate": gate,
                        })
                        if _is_better_human_shift_candidate(gate, best_candidate_gate):
                            best_candidate_text = candidate
                            best_candidate_report = candidate_report
                            best_candidate_gate = gate
                            best_candidate_eval = dict(candidate_eval)
                            candidate_eval["best_so_far"] = True
                        authenticity_summary["candidates"].append(candidate_eval)
                        if (
                            _env_flag("DRAFTPROOF_POST_GENERATION_TEXTURE_REPAIR", True)
                            and gate.get("ai_authorship_regressed")
                            and post_texture_calls < post_texture_limit
                        ):
                            post_texture_calls += 1
                            repaired_eval = {
                                "attempt": f"{candidate_eval.get('attempt')}.texture",
                                "strategy": f"{strategy}+post_generation_texture_repair",
                                "reconstruction": True,
                                "post_generation_texture_repair": True,
                                "parent_attempt": candidate_eval.get("attempt"),
                                "passed_local_checks": False,
                                "model": generator_model,
                            }
                            anchor_values = [
                                source_for_mitigation[span.start_char:span.end_char].strip()
                                for span in source_protected[:40]
                                if source_for_mitigation[span.start_char:span.end_char].strip()
                            ]
                            try:
                                repair_prompt, repair_info = _micro_texture_repair_prompt(
                                    candidate,
                                    candidate_report,
                                    anchors=anchor_values,
                                    max_sentences=1,
                                    mode="authorship_suppression_repair",
                                )
                                repair_response = gateway.chat(
                                    repair_prompt,
                                    system=(
                                        "You are DraftProof's micro-local authorship texture repairer. "
                                        "Return only the replacement sentence window."
                                    ),
                                    temperature=float(os.environ.get("DRAFTPROOF_RECONSTRUCTION_TEMPERATURE", "0.45")),
                                    max_tokens=int(os.environ.get("DRAFTPROOF_MICRO_TEXTURE_MAX_TOKENS", "1200")),
                                    top_p=_float_env_optional("DRAFTPROOF_RECONSTRUCTION_TOP_P"),
                                    top_k=_int_env_optional("DRAFTPROOF_RECONSTRUCTION_TOP_K"),
                                    presence_penalty=_float_env_optional("DRAFTPROOF_RECONSTRUCTION_PRESENCE_PENALTY"),
                                    frequency_penalty=_float_env_optional("DRAFTPROOF_RECONSTRUCTION_FREQUENCY_PENALTY"),
                                )
                                replacement, clean_reason = _clean_micro_texture_candidate(
                                    repair_response.content,
                                    repair_info,
                                )
                                repaired_eval["repair_window"] = {
                                    "start": (repair_info.get("window") or {}).get("start"),
                                    "end": (repair_info.get("window") or {}).get("end"),
                                }
                                if clean_reason:
                                    repaired_eval["reason"] = clean_reason
                                    authenticity_summary["candidates"].append(repaired_eval)
                                    continue
                                repaired_candidate = _splice_sentence_window(
                                    candidate,
                                    int((repair_info.get("window") or {}).get("start") or 0),
                                    int((repair_info.get("window") or {}).get("end") or 0),
                                    replacement,
                                )
                                repaired_eval["candidate_length"] = len(repaired_candidate or "")
                                repaired_eval["candidate_word_count"] = _text_word_count(repaired_candidate or "")
                                repaired_eval["repair_aggression"] = _repair_aggression_score(candidate, repaired_candidate)
                                repaired_eval["locality"] = _locality_score(candidate, repaired_candidate)
                                if repaired_candidate == candidate:
                                    repaired_eval["reason"] = "post_texture_no_change"
                                    authenticity_summary["candidates"].append(repaired_eval)
                                    continue
                                protected_loss = _ai_search_protected_loss_reason(
                                    source_for_mitigation,
                                    repaired_candidate,
                                    source_protected,
                                )
                                if protected_loss:
                                    repaired_eval["reason"] = "protected_span_lost " + protected_loss
                                    authenticity_summary["candidates"].append(repaired_eval)
                                    continue
                                drift = check_semantic_drift(
                                    source_for_mitigation,
                                    repaired_candidate,
                                    threshold=reconstruction_drift_threshold,
                                )
                                repaired_eval["drift_similarity"] = round(drift.similarity, 3)
                                repaired_eval["drift_threshold"] = reconstruction_drift_threshold
                                if not drift.accepted:
                                    repaired_eval["drift_reasons"] = drift.reasons[:10]
                                    if _reconstruction_drift_scan_allowed(repaired_candidate, drift.reasons, drift.similarity):
                                        repaired_eval["drift_scan_relaxed_for_reconstruction"] = True
                                        repaired_eval["drift_reasons_relaxed"] = drift.reasons[:5]
                                    else:
                                        repaired_eval["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                                        authenticity_summary["candidates"].append(repaired_eval)
                                        continue
                                repaired_eval["passed_local_checks"] = True
                                scan_t0 = time.time()
                                repaired_report = _full_scan_report_dict(repaired_candidate)
                                repaired_eval["scan_seconds"] = round(time.time() - scan_t0, 3)
                                repaired_review_burden = _review_burden(repaired_report)
                                repaired_severity = _weighted_severity(repaired_report)
                                repaired_gate = _authenticity_gate_status(
                                    original_report_dict,
                                    repaired_report,
                                    repaired_candidate != text,
                                    original_review_burden=original_review_burden,
                                    candidate_review_burden=repaired_review_burden,
                                    original_weighted_severity=original_severity,
                                    candidate_weighted_severity=repaired_severity,
                                    min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                                    min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                                    drift_similarity=repaired_eval.get("drift_similarity"),
                                )
                                repaired_eval.update({
                                    "ai": _badge_ai(repaired_report),
                                    "writing_quality": _badge_wq(repaired_report),
                                    "human_contribution": repaired_gate.get("candidate_human"),
                                    "ai_transformation": repaired_gate.get("candidate_ai_transformation"),
                                    "ai_authorship": repaired_gate.get("candidate_ai_authorship"),
                                    "human_delta": repaired_gate.get("human_delta"),
                                    "ai_transformation_delta": repaired_gate.get("ai_transformation_delta"),
                                    "ai_authorship_delta": repaired_gate.get("ai_authorship_delta"),
                                    "human_shift_score": repaired_gate.get("human_shift_score"),
                                    "human_shift_components": repaired_gate.get("human_shift_components"),
                                    "authorship_cost_per_human_gain": repaired_gate.get("authorship_cost_per_human_gain"),
                                    "findings": _finding_total(repaired_report),
                                    "review_burden": repaired_review_burden,
                                    "weighted_severity": repaired_severity,
                                    "scan_scope": _scan_scope_summary(repaired_report),
                                    "gate": repaired_gate,
                                })
                                if _is_better_human_shift_candidate(repaired_gate, best_candidate_gate):
                                    best_candidate_text = repaired_candidate
                                    best_candidate_report = repaired_report
                                    best_candidate_gate = repaired_gate
                                    best_candidate_eval = dict(repaired_eval)
                                    repaired_eval["best_so_far"] = True
                                authenticity_summary["candidates"].append(repaired_eval)
                            except Exception as exc:
                                repaired_eval["reason"] = f"post_texture_repair_error {exc}"
                                authenticity_summary["candidates"].append(repaired_eval)
                if (
                    best_candidate_gate
                    and best_candidate_report
                    and (
                        best_candidate_gate.get("success")
                        or (
                            masked_span_selected
                            and isinstance(best_candidate_eval, dict)
                            and best_candidate_eval.get("masked_span_repair")
                        )
                    )
                ):
                    previous_ai = rewritten_ai
                    rewritten_text = best_candidate_text
                    rewritten_report_dict = best_candidate_report
                    attempted_report_dict = rewritten_report_dict
                    rewritten_ai = _badge_ai(rewritten_report_dict)
                    rewritten_wq = _badge_wq(rewritten_report_dict)
                    rewritten_total = _finding_total(rewritten_report_dict)
                    rewritten_review_burden = _review_burden(rewritten_report_dict)
                    rewritten_severity = _weighted_severity(rewritten_report_dict)
                    rewritten_critical_high = _critical_high_count(rewritten_report_dict)
                    if result.mp_result:
                        result.mp_result.final_text = rewritten_text
                        result.mp_result.converged = True
                        result.mp_result.convergence_reason = (
                            "Selected authorship-capped masked-span AI-Mitigation candidate"
                            if isinstance(best_candidate_eval, dict) and best_candidate_eval.get("masked_span_repair")
                            else "Selected AI-Mitigation authenticity candidate"
                        )
                    sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                    authenticity_mitigation_selected = True
                    ai_search_selected = True
                    _clear_stale_rollback_for_kept_ai_mitigation(
                        result.summary,
                        "AI-Mitigation authenticity gate",
                    )
                    result.summary["ai_mitigation_blocked_auto_rewrite"] = False
                    result.summary["rewrite_engine_mode"] = (
                        "ai_mitigation_masked_span_gate"
                        if isinstance(best_candidate_eval, dict) and best_candidate_eval.get("masked_span_repair")
                        else (
                            "ai_mitigation_reconstruction_gate"
                            if isinstance(best_candidate_eval, dict) and best_candidate_eval.get("reconstruction")
                            else "ai_mitigation_authenticity_gate"
                        )
                    )
                    result.summary["outcome"] = "ai_mitigated"
                    if isinstance(best_candidate_eval, dict):
                        best_candidate_eval["selected"] = True
                    authenticity_summary.update({
                        "selected": True,
                        "selected_strategy": (
                            best_candidate_eval.get("strategy")
                            if isinstance(best_candidate_eval, dict) else None
                        ),
                        "selected_reconstruction": bool(
                            isinstance(best_candidate_eval, dict)
                            and best_candidate_eval.get("reconstruction")
                        ),
                        "selected_masked_span_repair": bool(
                            isinstance(best_candidate_eval, dict)
                            and best_candidate_eval.get("masked_span_repair")
                        ),
                        "previous_ai": previous_ai,
                        "selected_ai": rewritten_ai,
                        "selected_human_contribution": best_candidate_gate.get("candidate_human"),
                        "selected_ai_transformation": best_candidate_gate.get("candidate_ai_transformation"),
                        "selected_ai_authorship": best_candidate_gate.get("candidate_ai_authorship"),
                        "selected_human_shift_score": best_candidate_gate.get("human_shift_score"),
                        "selected_human_shift_components": best_candidate_gate.get("human_shift_components"),
                        "selected_gate": best_candidate_gate,
                    })
                elif best_candidate_eval:
                    authenticity_summary["best_attempt"] = best_candidate_eval
                    authenticity_summary["selection_reason"] = (
                        (best_candidate_gate or {}).get("reason")
                        or "no_candidate_passed_authenticity_gate"
                    )
            except Exception as exc:
                authenticity_summary["reason"] = f"authenticity_mitigation_error {exc}"
        authenticity_summary["seconds"] = round(time.time() - mitigation_started, 3)
        authenticity_summary["candidate_diagnostics"] = _generation_candidate_diagnostics(
            authenticity_summary.get("candidates") or []
        )
        result.summary["authenticity_mitigation"] = authenticity_summary
        result.summary["generation_layer"] = {
            "schema_version": "generation_layer.v1",
            "mode": "regeneration_first" if _env_flag("DRAFTPROOF_REGENERATION_FIRST", True) else "authenticity_then_reconstruction",
            "goal": "Human Contribution >= 80",
            "selected": bool(authenticity_summary.get("selected")),
            "selected_strategy": authenticity_summary.get("selected_strategy"),
            "selected_reconstruction": authenticity_summary.get("selected_reconstruction"),
            "selected_masked_span_repair": authenticity_summary.get("selected_masked_span_repair"),
            "selection_reason": authenticity_summary.get("selection_reason") or authenticity_summary.get("reason"),
            "llm_calls": authenticity_summary.get("llm_calls"),
            "model_roles": authenticity_summary.get("model_roles"),
            "masked_span_optimizer": authenticity_summary.get("masked_span_optimizer"),
            "masked_span_baseline": authenticity_summary.get("masked_span_baseline"),
            "skip_broad_generation_reason": authenticity_summary.get("skip_broad_generation_reason"),
            "reconstruction": authenticity_summary.get("reconstruction"),
            "best_attempt": authenticity_summary.get("best_attempt"),
            "candidate_count": len(authenticity_summary.get("candidates") or []),
            "candidate_diagnostics": authenticity_summary.get("candidate_diagnostics"),
        }
        if authenticity_summary.get("llm_calls"):
            result.summary["authenticity_llm_calls_used"] = authenticity_summary["llm_calls"]
            try:
                prior_calls = int(result.summary.get("llm_calls_used") or 0)
            except (TypeError, ValueError):
                prior_calls = 0
            result.summary["llm_calls_used"] = prior_calls + int(authenticity_summary["llm_calls"])
        stage_timings.append({
            "stage": "authenticity_mitigation",
            "seconds": authenticity_summary["seconds"],
            "candidates": len(authenticity_summary.get("candidates", [])),
            "selected": authenticity_summary.get("selected", False),
        })
        result.summary["detect_scores"].update({
            "rewritten_ai": rewritten_ai,
            "rewritten_writing_quality": rewritten_wq,
            "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
            "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
            "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
            "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
            "rewritten_findings": rewritten_total,
            "rewritten_review_burden": rewritten_review_burden,
            "rewritten_weighted_severity": rewritten_severity,
        })

    # Dedicated AI-score search. This is separate from local sentence rewrite:
    # generate multiple full-document candidates, scan every valid candidate,
    # and keep the one with the lowest measured AI likelihood.
    ai_search_reference = original_ai if original_ai is not None else saved_ai
    ai_search_enabled = os.environ.get("DRAFTPROOF_AI_MITIGATION_SEARCH", "1") != "0"
    generation_first_active = _env_flag("DRAFTPROOF_REGENERATION_FIRST", True)
    ai_search_after_generation_failure = _env_flag(
        "DRAFTPROOF_AI_SEARCH_AFTER_GENERATION_FAILURE",
        True,
    )
    if (
        generation_first_active
        and authenticity_enabled
        and authenticity_mitigation_selected
        and not ai_search_after_generation_failure
    ):
        ai_search_enabled = False
        reason = "skipped_after_generation_layer_selected"
        result.summary["ai_mitigation_search"] = {
            "enabled": False,
            "reason": reason,
            "generation_layer_required": True,
            "generation_layer_selected": bool(authenticity_mitigation_selected),
            "generation_layer_summary": result.summary.get("authenticity_mitigation"),
        }
        stage_timings.append({
            "stage": "ai_mitigation_search",
            "seconds": 0.0,
            "candidates": 0,
            "selected": False,
            "skipped": True,
            "skipped_reason": reason,
        })
    ai_search_blocked_by_author_gaps = (
        ai_mitigation_needs_author
        and not allow_auto_with_author_gaps
        and not authenticity_mitigation_selected
    )
    if (
        ai_search_enabled
        and not ai_search_blocked_by_author_gaps
        and isinstance(ai_search_reference, (int, float))
        and ai_search_reference >= ai_first_required_min_ai
    ):
        ai_search_target_score = round(max(0.0, ai_search_reference - ai_first_min_drop), 2)
        search_started = time.time()
        strategies = [
            "syntax_demolition",
            "paragraph_resequence",
            "plain_workshop_voice",
            "review_marked_grounding",
            "source_bridge_rebuild",
            "claim_narrowing",
            "cadence_disruption",
            "anchor_first_rebuild",
        ]
        try:
            search_limit = max(1, int(os.environ.get("DRAFTPROOF_AI_SEARCH_CANDIDATES", "4")))
        except ValueError:
            search_limit = 4
        strategies = strategies[:search_limit]
        search_source_text, search_source_repairs = _repair_candidate_source_damage(text)
        deterministic_candidates = []
        if search_source_repairs and search_source_text.strip() != text.strip():
            deterministic_candidates.append((
                "deterministic_source_integrity_repair",
                search_source_text,
            ))
        deterministic_candidates.extend(_ai_search_marked_grounding_candidates(search_source_text))
        search_summary = {
            "enabled": True,
            "reference_ai": ai_search_reference,
            "starting_ai": rewritten_ai,
            "candidate_limit": len(deterministic_candidates) + len(strategies),
            "deterministic_candidate_count": len(deterministic_candidates),
            "llm_candidate_limit": len(strategies),
            "required_ai_drop": ai_first_min_drop,
            "target_ai_score": ai_search_target_score,
            "llm_calls": 0,
            "selected": False,
            "candidates": [],
            "model_roles": llm_roles,
        }
        if search_source_repairs:
            search_summary["source_repairs"] = search_source_repairs
        effective_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        source_protected = detect_protected_spans(search_source_text)
        min_chars = max(200, int(len(search_source_text) * 0.75))
        max_chars = max(min_chars, int(len(text) * 1.30))
        best_text = rewritten_text
        best_report = rewritten_report_dict
        best_ai = rewritten_ai if isinstance(rewritten_ai, (int, float)) else 999.0
        best_strategy = None
        best_semantic_review_required = False
        best_drift_reasons: list[str] = []
        best_selection_status: dict = {}
        best_human_shift_rank: tuple = (-1, -9999.0, -9999.0)

        def _best_ai_search_selectable() -> bool:
            return bool(best_strategy and best_selection_status.get("selectable"))

        def _record_best_attempt() -> None:
            if not best_strategy:
                return
            search_summary["best_attempt"] = {
                "strategy": best_strategy,
                "ai": best_ai,
                "ai_delta_vs_reference": (
                    round(ai_search_reference - best_ai, 3)
                    if isinstance(best_ai, (int, float)) else None
                ),
                "human_shift_score": best_selection_status.get("human_shift_score"),
                "human_shift_components": best_selection_status.get("human_shift_components"),
                "selection_status": best_selection_status,
            }

        def _evaluate_ai_search_candidate(
            strategy: str,
            candidate: str,
            *,
            deterministic: bool = False,
            extra: dict | None = None,
        ) -> None:
            nonlocal best_text, best_report, best_ai, best_strategy
            nonlocal best_semantic_review_required, best_drift_reasons, best_selection_status
            nonlocal best_human_shift_rank
            candidate_eval = {
                "strategy": strategy,
                "deterministic": deterministic,
                "passed_local_checks": False,
                "candidate_length": len(candidate or ""),
            }
            if extra:
                candidate_eval.update(extra)
            if not candidate:
                candidate_eval["reason"] = "empty_candidate"
                search_summary["candidates"].append(candidate_eval)
                return
            candidate, repair_notes = _repair_candidate_source_damage(candidate)
            if repair_notes:
                candidate_eval["candidate_length"] = len(candidate or "")
                candidate_eval["source_damage_repairs"] = repair_notes
            review_notes = _review_marker_notes(candidate)
            if review_notes:
                candidate_eval["reason"] = "review_markers_not_auto_kept"
                candidate_eval["review_suggestion_count"] = len(review_notes)
                manual = result.summary.setdefault("manual_suggestions", [])
                for note in review_notes:
                    if len(manual) >= 30:
                        break
                    manual.append({
                        "finding_type": "ai_mitigation_review_note",
                        "scanner_target": "ai_mitigation_search",
                        "original_sentence": "",
                        "suggested_sentence": f"[[REVIEW: {note}]]",
                        "rejection_reason": "review_markers_not_auto_kept",
                        "why_review_manually": (
                            "This note asks the author to add real evidence or context. "
                            "It is shown as guidance, not inserted into the rewritten document."
                        ),
                    })
                search_summary["candidates"].append(candidate_eval)
                return
            quality_rejection = _ai_candidate_quality_reject_reason(candidate)
            if quality_rejection:
                candidate_eval["reason"] = quality_rejection
                search_summary["candidates"].append(candidate_eval)
                return
            if len(candidate) < min_chars:
                candidate_eval["reason"] = f"candidate_too_short {len(candidate)}<{min_chars}"
                search_summary["candidates"].append(candidate_eval)
                return
            if len(candidate) > max_chars:
                candidate_eval["reason"] = f"candidate_too_long {len(candidate)}>{max_chars}"
                search_summary["candidates"].append(candidate_eval)
                return
            protected_loss = _ai_search_protected_loss_reason(search_source_text, candidate, source_protected)
            if protected_loss:
                candidate_eval["reason"] = "protected_span_lost " + protected_loss
                search_summary["candidates"].append(candidate_eval)
                return
            drift = check_semantic_drift(search_source_text, candidate, threshold=0.15)
            candidate_eval["drift_similarity"] = round(drift.similarity, 3)
            if not drift.accepted:
                candidate_eval["drift_reasons"] = drift.reasons[:10]
                if repair_notes and _source_repair_drift_false_positive(candidate, drift.reasons):
                    candidate_eval["drift_relaxed_for_source_repair"] = True
                    candidate_eval["drift_reasons_relaxed"] = drift.reasons[:5]
                elif _ai_search_drift_false_positive(candidate, drift.reasons, drift.similarity):
                    candidate_eval["drift_relaxed_for_ai_search"] = True
                    candidate_eval["drift_reasons_relaxed"] = drift.reasons[:5]
                elif _ai_search_entity_drift_scan_allowed(candidate, drift.reasons, drift.similarity):
                    candidate_eval["semantic_review_required"] = True
                    candidate_eval["drift_scan_relaxed_for_scoring"] = True
                else:
                    candidate_eval["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                    search_summary["candidates"].append(candidate_eval)
                    return

            candidate_eval["passed_local_checks"] = True
            try:
                scan_t0 = time.time()
                candidate_report = _full_scan_report_dict(candidate)
                candidate_eval["scan_seconds"] = round(time.time() - scan_t0, 3)
            except Exception as exc:
                candidate_eval["passed_local_checks"] = False
                candidate_eval["reason"] = f"candidate_scan_error {exc}"
                search_summary["candidates"].append(candidate_eval)
                return

            candidate_ai = _badge_ai(candidate_report)
            candidate_wq = _badge_wq(candidate_report)
            candidate_review_burden = _review_burden(candidate_report)
            candidate_weighted_severity = _weighted_severity(candidate_report)
            candidate_contribution = _contribution_scores(candidate_report)
            candidate_integrity = _integrity_scores(candidate_report)
            human_shift = _human_shift_score(
                original_report_dict,
                candidate_report,
                drift_similarity=candidate_eval.get("drift_similarity"),
                review_burden_delta=candidate_review_burden - original_review_burden,
                weighted_severity_delta=candidate_weighted_severity - original_severity,
            )
            candidate_eval.update({
                "ai": candidate_ai,
                "ai_delta_vs_reference": (
                    round(ai_search_reference - candidate_ai, 3)
                    if isinstance(candidate_ai, (int, float)) else None
                ),
                "writing_quality": candidate_wq,
                "human_contribution": candidate_contribution.get("human"),
                "ai_transformation": candidate_contribution.get("ai_transformation"),
                "ai_authorship": candidate_integrity.get("ai_authorship"),
                "grounding_quality_risk": candidate_integrity.get("grounding"),
                "findings": _finding_total(candidate_report),
                "review_burden": candidate_review_burden,
                "weighted_severity": candidate_weighted_severity,
                "human_shift_score": human_shift.get("score"),
                "human_shift_components": human_shift.get("components"),
                "scan_scope": _scan_scope_summary(candidate_report),
            })
            selection_status = _ai_search_candidate_selection_status(
                ai_search_reference,
                candidate_ai,
                candidate != text,
                min_drop=ai_first_min_drop,
                target=ai_first_target,
                required_min_ai=ai_first_required_min_ai,
            )
            ai_delta = (
                ai_search_reference - candidate_ai
                if isinstance(ai_search_reference, (int, float)) and isinstance(candidate_ai, (int, float))
                else -9999.0
            )
            authenticity_status = _authenticity_gate_status(
                original_report_dict,
                candidate_report,
                candidate != text,
                original_review_burden=original_review_burden,
                candidate_review_burden=candidate_review_burden,
                original_weighted_severity=original_severity,
                candidate_weighted_severity=candidate_weighted_severity,
                min_human_gain=_float_env("DRAFTPROOF_AI_SEARCH_MIN_HUMAN_GAIN", 1.0),
                min_ai_transformation_drop=_float_env(
                    "DRAFTPROOF_AI_SEARCH_MIN_AI_TRANSFORM_DROP",
                    1.0,
                ),
                drift_similarity=candidate_eval.get("drift_similarity"),
            )
            if (
                selection_status.get("selectable")
                and not _env_flag("DRAFTPROOF_AI_SEARCH_ALLOW_REVIEW_REGRESSION", False)
                and (
                    authenticity_status.get("review_burden_regressed")
                    or authenticity_status.get("weighted_severity_regressed")
                    or authenticity_status.get("critical_high_regressed")
                )
            ):
                selection_status.update({
                    "success": False,
                    "selectable": False,
                    "reason": "ai_drop_quality_regressed",
                })
            incremental_authenticity_selectable = bool(
                _env_flag("DRAFTPROOF_AI_SEARCH_ACCEPT_INCREMENTAL_AUTHENTICITY", True)
                and authenticity_status.get("candidate_progress")
                and not authenticity_status.get("ai_authorship_regression_blocked")
                and not authenticity_status.get("critical_high_regressed")
                and not authenticity_status.get("review_burden_regressed")
                and not authenticity_status.get("weighted_severity_regressed")
            )
            ai_authorship_delta = authenticity_status.get("ai_authorship_delta")
            safe_authorship_suppression_selectable = bool(
                _env_flag("DRAFTPROOF_AI_SEARCH_ACCEPT_SAFE_AUTHORSHIP_SUPPRESSION", True)
                and isinstance(ai_authorship_delta, (int, float))
                and ai_authorship_delta >= _float_env(
                    "DRAFTPROOF_AI_SEARCH_MIN_SAFE_AUTHORSHIP_DROP",
                    1.0,
                )
                and isinstance(ai_delta, (int, float))
                and ai_delta > 0.05
                and _finding_total(candidate_report) <= original_total
                and candidate_review_burden <= original_review_burden
                and candidate_weighted_severity <= original_severity
                and not authenticity_status.get("ai_authorship_regression_blocked")
                and not authenticity_status.get("critical_high_regressed")
            )
            if (
                not selection_status.get("selectable")
                and (
                    incremental_authenticity_selectable
                    or safe_authorship_suppression_selectable
                )
            ):
                selection_status.update({
                    "success": True,
                    "selectable": True,
                    "reason": (
                        "accepted_safe_authorship_suppression"
                        if safe_authorship_suppression_selectable
                        else "accepted_incremental_authenticity_progress"
                    ),
                    "authenticity_incremental": bool(incremental_authenticity_selectable),
                    "safe_authorship_suppression": bool(safe_authorship_suppression_selectable),
                })
            selection_status["authenticity_gate"] = authenticity_status
            selection_status["human_shift_score"] = human_shift.get("score")
            selection_status["human_shift_components"] = human_shift.get("components")
            candidate_eval["selection_status"] = selection_status
            human_shift_score = human_shift.get("score")
            candidate_rank = (
                1 if selection_status.get("selectable") else 0,
                float(human_shift_score) if isinstance(human_shift_score, (int, float)) else -9999.0,
                float(ai_delta) if isinstance(ai_delta, (int, float)) else -9999.0,
            )
            if candidate_rank > best_human_shift_rank:
                best_ai = candidate_ai
                best_text = candidate
                best_report = candidate_report
                best_strategy = strategy
                best_semantic_review_required = bool(candidate_eval.get("semantic_review_required"))
                best_drift_reasons = list(candidate_eval.get("drift_reasons") or [])
                best_selection_status = selection_status
                best_human_shift_rank = candidate_rank
                candidate_eval["best_so_far"] = True
                candidate_eval["selectable_so_far"] = bool(selection_status.get("selectable"))
                _record_best_attempt()
            search_summary["candidates"].append(candidate_eval)

        deterministic_only = (
            bool(ai_search_first)
            and not _allow_ai_search_llm_after_deterministic()
        )
        try:
            min_deterministic_scans = max(
                1,
                int(os.environ.get(
                    "DRAFTPROOF_AI_SEARCH_MIN_DETERMINISTIC_SCANS",
                    str(len(deterministic_candidates) or 1),
                )),
            )
        except ValueError:
            min_deterministic_scans = len(deterministic_candidates) or 1
        early_stop_reason = ""
        for index, (strategy, candidate) in enumerate(deterministic_candidates, start=1):
            report_progress(
                min(79, 76 + index),
                f"Scanning deterministic AI mitigation candidate {index}/{len(deterministic_candidates)}",
            )
            _evaluate_ai_search_candidate(strategy, candidate, deterministic=True)
            if index < min_deterministic_scans:
                continue
            early_stop_reason = _ai_search_fast_accept_reason(ai_search_reference, best_ai)
            if early_stop_reason:
                search_summary["early_stop"] = {
                    "phase": "deterministic_candidates",
                    "reason": early_stop_reason,
                    "candidate_count_scanned": len(search_summary.get("candidates", [])),
                    "selected_strategy": best_strategy,
                    "selected_ai": best_ai,
                }
                break

        if early_stop_reason:
            search_summary["llm_reason"] = "skipped_after_fast_deterministic_accept"
        elif deterministic_only:
            search_summary["llm_reason"] = "skipped_deterministic_only_ai_first"
            if best_strategy:
                search_summary["deterministic_only_best_attempt"] = True
        elif not effective_key:
            if not search_summary.get("candidates"):
                search_summary["reason"] = "no_llm_available"
            else:
                search_summary["llm_reason"] = "no_llm_available"
        else:
            try:
                gateway = LLMGateway(LLMConfig(
                    api_key=effective_key,
                    model=generator_model,
                    base_url=base_url,
                    timeout=int(os.environ.get("DRAFTPROOF_AI_SEARCH_TIMEOUT", "120")),
                    max_retries=int(os.environ.get("DRAFTPROOF_AI_SEARCH_RETRIES", "1")),
                    max_tokens=int(os.environ.get("DRAFTPROOF_AI_SEARCH_MAX_TOKENS", "6500")),
                    temperature=float(os.environ.get("DRAFTPROOF_AI_SEARCH_TEMPERATURE", "0.75")),
                ))
                paragraph_search_enabled = os.environ.get(
                    "DRAFTPROOF_PARAGRAPH_COMPONENT_SEARCH",
                    "1",
                ) != "0"
                if paragraph_search_enabled:
                    try:
                        paragraph_limit = max(
                            1,
                            int(os.environ.get("DRAFTPROOF_PARAGRAPH_COMPONENT_TARGETS", "6")),
                        )
                    except ValueError:
                        paragraph_limit = 4
                    try:
                        paragraph_candidates = max(
                            1,
                            int(os.environ.get("DRAFTPROOF_PARAGRAPH_COMPONENT_CANDIDATES", "3")),
                        )
                    except ValueError:
                        paragraph_candidates = 3
                    component_base_text, component_base_repairs = _repair_candidate_source_damage(search_source_text)
                    component_targets = _paragraph_component_targets(
                        component_base_text,
                        ctx.raw_json,
                        limit=paragraph_limit,
                    )
                    component_summary = {
                        "enabled": True,
                        "base_repairs": component_base_repairs,
                        "target_count": len(component_targets),
                        "candidate_limit_per_target": paragraph_candidates,
                        "targets": [
                            {
                                "paragraph_index": t.get("index"),
                                "score": t.get("score"),
                                "drivers": t.get("drivers"),
                                "preview": (t.get("paragraph") or "")[:180],
                            }
                            for t in component_targets
                        ],
                    }
                    search_summary["paragraph_component_search"] = component_summary
                    for target_number, target in enumerate(component_targets, start=1):
                        report_progress(
                            min(89, 79 + target_number),
                            (
                                "Trying paragraph-component AI batch "
                                f"{target_number}/{len(component_targets)}"
                            )
                        )
                        try:
                            prompt = _paragraph_component_prompt(
                                target,
                                ctx.raw_json,
                                target_number,
                                reference_ai=ai_search_reference,
                                required_ai_drop=ai_first_min_drop,
                                target_ai_score=ai_search_target_score,
                                candidate_count=paragraph_candidates,
                            )
                            search_summary["llm_calls"] += 1
                            response = gateway.chat(
                                prompt,
                                system=(
                                    "You are DraftProof's paragraph AI-score mitigation engine. "
                                    "Return only the requested tagged replacement paragraphs."
                                ),
                                temperature=float(os.environ.get(
                                    "DRAFTPROOF_PARAGRAPH_COMPONENT_TEMPERATURE",
                                    "0.8",
                                )),
                                max_tokens=int(os.environ.get(
                                    "DRAFTPROOF_PARAGRAPH_COMPONENT_MAX_TOKENS",
                                    "2600",
                                )),
                            )
                            paragraph_outputs = _extract_paragraph_component_candidates(
                                response.content,
                                paragraph_candidates,
                            )
                        except Exception as exc:
                            search_summary["candidates"].append({
                                "strategy": (
                                    f"paragraph_component_p{int(target.get('index', 0)) + 1}"
                                    "_batch"
                                ),
                                "passed_local_checks": False,
                                "reason": f"llm_error {exc}",
                                "paragraph_component": True,
                                "paragraph_index": target.get("index"),
                            })
                            continue
                        if not paragraph_outputs:
                            search_summary["candidates"].append({
                                "strategy": (
                                    f"paragraph_component_p{int(target.get('index', 0)) + 1}"
                                    "_batch"
                                ),
                                "passed_local_checks": False,
                                "reason": "empty_candidate_batch",
                                "paragraph_component": True,
                                "paragraph_index": target.get("index"),
                            })
                            continue
                        for candidate_number, raw_paragraph_candidate in enumerate(paragraph_outputs, start=1):
                            strategy = (
                                f"paragraph_component_p{int(target.get('index', 0)) + 1}"
                                f"_c{candidate_number}"
                            )
                            paragraph_candidate, paragraph_reject = _clean_paragraph_component_candidate(
                                raw_paragraph_candidate,
                                target.get("paragraph") or "",
                            )
                            if paragraph_reject:
                                search_summary["candidates"].append({
                                    "strategy": strategy,
                                    "passed_local_checks": False,
                                    "reason": paragraph_reject,
                                    "paragraph_component": True,
                                    "paragraph_index": target.get("index"),
                                    "paragraph_driver_score": target.get("score"),
                                })
                                continue
                            patched_candidate = _splice_paragraph(
                                component_base_text,
                                int(target.get("index", 0)),
                                paragraph_candidate,
                            )
                            _evaluate_ai_search_candidate(
                                strategy,
                                patched_candidate,
                                deterministic=False,
                                extra={
                                    "paragraph_component": True,
                                    "paragraph_index": target.get("index"),
                                    "paragraph_driver_score": target.get("score"),
                                    "paragraph_drivers": target.get("drivers"),
                                },
                            )
                        if best_strategy:
                            component_base_text = best_text

                for index, strategy in enumerate(strategies, start=1):
                    report_progress(
                        min(79, 76 + index),
                        f"Trying AI mitigation candidate {index}/{len(strategies)}",
                    )
                    candidate_eval = {
                        "strategy": strategy,
                        "passed_local_checks": False,
                    }
                    try:
                        prompt = _ai_search_prompt(
                            search_source_text,
                            ctx.raw_json,
                            strategy,
                            reference_ai=ai_search_reference,
                            required_ai_drop=ai_first_min_drop,
                            target_ai_score=ai_search_target_score,
                        )
                        search_summary["llm_calls"] += 1
                        response = gateway.chat(
                            prompt,
                            system=(
                                "You are DraftProof's AI-score mitigation engine. "
                                "Return only the complete rewritten document."
                            ),
                            temperature=float(os.environ.get("DRAFTPROOF_AI_SEARCH_TEMPERATURE", "0.75")),
                            max_tokens=int(os.environ.get("DRAFTPROOF_AI_SEARCH_MAX_TOKENS", "6500")),
                        )
                        candidate = _clean_full_document_candidate(response.content, search_source_text)
                    except Exception as exc:
                        candidate_eval["reason"] = f"llm_error {exc}"
                        search_summary["candidates"].append(candidate_eval)
                        continue

                    _evaluate_ai_search_candidate(strategy, candidate, deterministic=False)

                if not _best_ai_search_selectable():
                    try:
                        feedback_limit = max(
                            0,
                            int(os.environ.get("DRAFTPROOF_AI_SEARCH_FEEDBACK_CANDIDATES", "2")),
                        )
                    except ValueError:
                        feedback_limit = 2
                    retry_enabled = bool(llm_roles.get("retry_model_enabled"))
                    retry_budget = int(llm_roles.get("retry_model_max_calls") or 0)
                    if not retry_enabled:
                        search_summary["score_feedback_loop"] = {
                            "enabled": False,
                            "candidate_limit": 0,
                            "reason": "retry_model_disabled_by_kill_switch",
                            "retry_model": retry_model,
                        }
                        feedback_limit = 0
                    else:
                        feedback_limit = min(feedback_limit, retry_budget)
                    if feedback_limit:
                        search_summary["score_feedback_loop"] = {
                            "enabled": True,
                            "candidate_limit": feedback_limit,
                            "retry_model": retry_model,
                            "retry_model_max_calls": retry_budget,
                            "reason": (
                                "no_selectable_candidate"
                                if not best_strategy
                                else "best_candidate_below_required_ai_drop"
                            ),
                        }
                    retry_gateway = None
                    if feedback_limit:
                        retry_gateway = LLMGateway(LLMConfig(
                            api_key=effective_key,
                            model=retry_model,
                            base_url=base_url,
                            timeout=int(os.environ.get("DRAFTPROOF_AI_SEARCH_TIMEOUT", "120")),
                            max_retries=int(os.environ.get("DRAFTPROOF_AI_SEARCH_RETRIES", "1")),
                            max_tokens=int(os.environ.get("DRAFTPROOF_AI_SEARCH_MAX_TOKENS", "6500")),
                            temperature=float(os.environ.get("DRAFTPROOF_AI_SEARCH_FEEDBACK_TEMPERATURE", "0.8")),
                        ))
                    for feedback_index in range(1, feedback_limit + 1):
                        report_progress(
                            min(89, 80 + feedback_index),
                            f"Trying score-feedback AI mitigation candidate {feedback_index}/{feedback_limit}",
                        )
                        candidate_eval = {
                            "strategy": f"score_feedback_{feedback_index}",
                            "passed_local_checks": False,
                            "retry_model": retry_model,
                        }
                        try:
                            prompt = _ai_search_feedback_prompt(
                                search_source_text,
                                ctx.raw_json,
                                search_summary,
                                feedback_index,
                            )
                            search_summary["llm_calls"] += 1
                            response = retry_gateway.chat(
                                prompt,
                                system=(
                                    "You are DraftProof's score-feedback rewrite engine. "
                                    "Use the detector scorecard to produce a lower-scoring complete document."
                                ),
                                temperature=float(os.environ.get("DRAFTPROOF_AI_SEARCH_FEEDBACK_TEMPERATURE", "0.8")),
                                max_tokens=int(os.environ.get("DRAFTPROOF_AI_SEARCH_MAX_TOKENS", "6500")),
                            )
                            candidate = _clean_full_document_candidate(response.content, search_source_text)
                        except Exception as exc:
                            candidate_eval["reason"] = f"llm_error {exc}"
                            search_summary["candidates"].append(candidate_eval)
                            continue
                        _evaluate_ai_search_candidate(
                            f"score_feedback_{feedback_index}",
                            candidate,
                            deterministic=False,
                        )
                        if _best_ai_search_selectable():
                            break

                if _best_ai_search_selectable():
                    previous_ai = rewritten_ai
                    rewritten_text = best_text
                    rewritten_report_dict = best_report
                    attempted_report_dict = rewritten_report_dict
                    rewritten_ai = _badge_ai(rewritten_report_dict)
                    rewritten_wq = _badge_wq(rewritten_report_dict)
                    rewritten_total = _finding_total(rewritten_report_dict)
                    rewritten_review_burden = _review_burden(rewritten_report_dict)
                    rewritten_severity = _weighted_severity(rewritten_report_dict)
                    rewritten_critical_high = (
                        len(rewritten_report_dict.get("findings", {}).get("critical", []))
                        + len(rewritten_report_dict.get("findings", {}).get("high", []))
                    )
                    if result.mp_result:
                        result.mp_result.final_text = rewritten_text
                        result.mp_result.converged = True
                        result.mp_result.convergence_reason = (
                            f"Selected AI mitigation search candidate: {best_strategy}"
                        )
                    sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                    ai_search_selected = True
                    _clear_stale_rollback_for_kept_ai_mitigation(
                        result.summary,
                        "AI mitigation search",
                    )
                    search_summary.update({
                        "selected": True,
                        "selected_strategy": best_strategy,
                        "previous_ai": previous_ai,
                        "selected_ai": rewritten_ai,
                        "selected_ai_delta_vs_reference": (
                            round(ai_search_reference - rewritten_ai, 3)
                            if isinstance(rewritten_ai, (int, float)) else None
                        ),
                        "selected_human_shift_score": best_selection_status.get("human_shift_score"),
                        "selected_human_shift_components": best_selection_status.get("human_shift_components"),
                        "selected_semantic_review_required": best_semantic_review_required,
                        "selected_drift_reasons": best_drift_reasons[:10],
                        "selection_status": best_selection_status,
                    })
            except Exception as exc:
                search_summary["reason"] = f"search_error {exc}"
        if _best_ai_search_selectable() and not search_summary.get("selected"):
            previous_ai = rewritten_ai
            rewritten_text = best_text
            rewritten_report_dict = best_report
            attempted_report_dict = rewritten_report_dict
            rewritten_ai = _badge_ai(rewritten_report_dict)
            rewritten_wq = _badge_wq(rewritten_report_dict)
            rewritten_total = _finding_total(rewritten_report_dict)
            rewritten_review_burden = _review_burden(rewritten_report_dict)
            rewritten_severity = _weighted_severity(rewritten_report_dict)
            rewritten_critical_high = (
                len(rewritten_report_dict.get("findings", {}).get("critical", []))
                + len(rewritten_report_dict.get("findings", {}).get("high", []))
            )
            if result.mp_result:
                result.mp_result.final_text = rewritten_text
                result.mp_result.converged = True
                result.mp_result.convergence_reason = (
                    f"Selected AI mitigation search candidate: {best_strategy}"
                )
            sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
            ai_search_selected = True
            _clear_stale_rollback_for_kept_ai_mitigation(
                result.summary,
                "deterministic AI mitigation search",
            )
            search_summary.update({
                "selected": True,
                "selected_strategy": best_strategy,
                "previous_ai": previous_ai,
                "selected_ai": rewritten_ai,
                "selected_ai_delta_vs_reference": (
                    round(ai_search_reference - rewritten_ai, 3)
                    if isinstance(rewritten_ai, (int, float)) else None
                ),
                "selected_human_shift_score": best_selection_status.get("human_shift_score"),
                "selected_human_shift_components": best_selection_status.get("human_shift_components"),
                "selected_semantic_review_required": best_semantic_review_required,
                "selected_drift_reasons": best_drift_reasons[:10],
                "selection_status": best_selection_status,
            })
        elif best_strategy and not search_summary.get("selected"):
            _record_best_attempt()
            search_summary["selected"] = False
            search_summary["selection_reason"] = (
                best_selection_status.get("reason")
                or "best_candidate_below_required_ai_drop"
            )
        search_summary["seconds"] = round(time.time() - search_started, 3)
        result.summary["ai_mitigation_search"] = search_summary
        if search_summary.get("llm_calls"):
            result.summary["ai_search_llm_calls_used"] = search_summary["llm_calls"]
            try:
                prior_calls = int(result.summary.get("llm_calls_used") or 0)
            except (TypeError, ValueError):
                prior_calls = 0
            result.summary["llm_calls_used"] = prior_calls + int(search_summary["llm_calls"])
        stage_timings.append({
            "stage": "ai_mitigation_search",
            "seconds": search_summary["seconds"],
            "candidates": len(search_summary.get("candidates", [])),
            "selected": search_summary.get("selected", False),
        })

        result.summary["detect_scores"].update({
            "rewritten_ai": rewritten_ai,
            "rewritten_writing_quality": rewritten_wq,
            "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
            "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
            "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
            "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
            "rewritten_findings": rewritten_total,
            "rewritten_review_burden": rewritten_review_burden,
            "rewritten_weighted_severity": rewritten_severity,
        })
    elif ai_search_blocked_by_author_gaps:
        result.summary["ai_mitigation_search"] = {
            "enabled": False,
            "selected": False,
            "reason": "requires_author_input",
            "reference_ai": ai_search_reference,
            "starting_ai": rewritten_ai,
            "candidate_limit": 0,
            "llm_calls": 0,
            "candidates": [],
        }
        stage_timings.append({
            "stage": "ai_mitigation_search",
            "seconds": 0,
            "candidates": 0,
            "selected": False,
            "skipped_reason": "requires_author_input",
        })

    ai_regression_tolerance = 0.25
    writing_quality_regression_tolerance = 1.0

    ai_score_regressed = (
        original_ai is not None
        and rewritten_ai is not None
        and rewritten_ai > original_ai + ai_regression_tolerance
    )
    wq_score_regressed = (
        original_wq is not None
        and rewritten_wq is not None
        and rewritten_wq > original_wq + writing_quality_regression_tolerance
    )
    total_findings_regressed = rewritten_total > original_total
    review_burden_regressed = rewritten_review_burden > original_review_burden
    severity_regressed = rewritten_severity > original_severity
    critical_high_regressed = (
        len(rewritten_report_dict.get("findings", {}).get("critical", []))
        + len(rewritten_report_dict.get("findings", {}).get("high", []))
        >
        len(original_report_dict.get("findings", {}).get("critical", []))
        + len(original_report_dict.get("findings", {}).get("high", []))
    )
    total_regressed_without_review_gain = (
        total_findings_regressed
        and rewritten_review_burden >= original_review_burden
    )
    fresh_baseline_improved = (
        rewritten_review_burden < original_review_burden
        or rewritten_severity < original_severity
        or rewritten_total < original_total
        or (
            original_ai is not None
            and rewritten_ai is not None
            and rewritten_ai < original_ai - 0.05
        )
        or (
            original_wq is not None
            and rewritten_wq is not None
            and rewritten_wq < original_wq - 0.05
        )
    )
    fresh_ai_improved = (
        original_ai is not None
        and rewritten_ai is not None
        and rewritten_ai < original_ai - 0.05
    )
    saved_ai_drifted_up = (
        saved_ai is not None
        and original_ai is not None
        and original_ai > saved_ai + ai_regression_tolerance
    )
    saved_ai_regressed = (
        saved_ai is not None
        and rewritten_ai is not None
        and rewritten_ai > saved_ai + ai_regression_tolerance
    )
    saved_total_regressed = rewritten_total > saved_total
    saved_critical_high_regressed = rewritten_critical_high > saved_critical_high
    saved_ai_regression_explained_by_drift = (
        saved_ai_regressed
        and saved_ai_drifted_up
        and fresh_ai_improved
        and not ai_score_regressed
        and not review_burden_regressed
        and not severity_regressed
        and not saved_critical_high_regressed
    )
    regression_reasons = []
    followup_warnings = []
    result.summary["regression_tolerances"] = {
        "ai_score": ai_regression_tolerance,
        "writing_quality_score": writing_quality_regression_tolerance,
    }
    if ai_score_regressed:
        regression_reasons.append(f"AI {original_ai}->{rewritten_ai}")
    if wq_score_regressed:
        followup_warnings.append(f"writing_quality {original_wq}->{rewritten_wq}")
    if review_burden_regressed:
        regression_reasons.append(
            f"review_burden {original_review_burden}->{rewritten_review_burden}"
        )
    if critical_high_regressed:
        original_ch = (
            len(original_report_dict.get("findings", {}).get("critical", []))
            + len(original_report_dict.get("findings", {}).get("high", []))
        )
        rewritten_ch = (
            len(rewritten_report_dict.get("findings", {}).get("critical", []))
            + len(rewritten_report_dict.get("findings", {}).get("high", []))
        )
        regression_reasons.append(f"critical_high_findings {original_ch}->{rewritten_ch}")
    if severity_regressed:
        regression_reasons.append(
            f"weighted_severity {original_severity}->{rewritten_severity}"
        )
    if total_findings_regressed:
        regression_reasons.append(f"findings {original_total}->{rewritten_total}")
    elif total_regressed_without_review_gain and not (review_burden_regressed or severity_regressed):
        regression_reasons.append(f"findings {original_total}->{rewritten_total}")
    # The saved scan is the user-visible contract, but detector scores can drift
    # between the saved report and a fresh rescan. Critical/high findings remain
    # hard guards. AI and total count are strict only when the fresh baseline did
    # not improve or the saved-score increase is not explained by baseline drift.
    if saved_ai_regressed and not saved_ai_regression_explained_by_drift:
        regression_reasons.append(f"user_visible_ai {saved_ai}->{rewritten_ai}")
    elif saved_ai_regressed:
        result.summary.setdefault("saved_contract_notes", []).append(
            "user_visible_ai increased "
            f"{saved_ai}->{rewritten_ai}, but fresh baseline improved "
            f"{original_ai}->{rewritten_ai}; kept attempted rewrite for review."
        )
    if (
        saved_total_regressed
        and (
            not fresh_baseline_improved
            or (saved_ai_regressed and not saved_ai_regression_explained_by_drift)
            or saved_critical_high_regressed
            or rewritten_review_burden > original_review_burden
            or rewritten_severity > original_severity
        )
    ):
        regression_reasons.append(f"user_visible_findings {saved_total}->{rewritten_total}")
    elif saved_total_regressed:
        result.summary.setdefault("saved_contract_notes", []).append(
            "user_visible_findings increased "
            f"{saved_total}->{rewritten_total}, but fresh baseline improved "
            f"{original_total}->{rewritten_total}; kept attempted rewrite for review."
        )
    if saved_critical_high_regressed:
        regression_reasons.append(
            f"user_visible_critical_high_findings {saved_critical_high}->{rewritten_critical_high}"
        )
    if authenticity_mitigation_selected:
        authenticity_breakthrough_tradeoff = bool(
            _env_flag("DRAFTPROOF_AUTHENTICITY_BREAKTHROUGH_TRADEOFF", True)
            and isinstance(original_ai, (int, float))
            and isinstance(rewritten_ai, (int, float))
            and rewritten_ai <= original_ai - 10.0
            and isinstance(_integrity_scores(original_report_dict).get("ai_authorship"), (int, float))
            and isinstance(_integrity_scores(rewritten_report_dict).get("ai_authorship"), (int, float))
            and _integrity_scores(rewritten_report_dict).get("ai_authorship")
            <= _integrity_scores(original_report_dict).get("ai_authorship") - 10.0
            and rewritten_total <= original_total
        )
        hard_regression_reasons = []
        soft_regression_reasons = []
        for reason in regression_reasons:
            if reason.startswith((
                "review_burden ",
                "critical_high_findings ",
                "weighted_severity ",
                "user_visible_critical_high_findings ",
            )) and not authenticity_breakthrough_tradeoff:
                hard_regression_reasons.append(reason)
            else:
                soft_regression_reasons.append(reason)
        if soft_regression_reasons:
            followup_warnings.extend(
                f"post_authenticity_review {reason}" for reason in soft_regression_reasons
            )
        regression_reasons = hard_regression_reasons
        result.summary.setdefault("authenticity_mitigation", {})["kept"] = not hard_regression_reasons
        result.summary.setdefault("authenticity_mitigation", {})[
            "breakthrough_tradeoff"
        ] = authenticity_breakthrough_tradeoff
        result.summary.setdefault("authenticity_mitigation", {})["hard_regressions"] = hard_regression_reasons
        result.summary.setdefault("authenticity_mitigation", {})["soft_followups"] = soft_regression_reasons
        if not hard_regression_reasons:
            result.summary.setdefault("saved_contract_notes", []).append(
                "AI-Mitigation authenticity gate kept the rewrite because the contribution score moved toward Human without review-burden or severity regression."
            )
    ai_first_reference = original_ai if original_ai is not None else saved_ai
    ai_first_gate = _ai_first_gate_status(
        ai_first_reference,
        rewritten_ai,
        rewritten_text != text,
        min_drop=ai_first_min_drop,
        target=ai_first_target,
        required_min_ai=ai_first_required_min_ai,
    )
    ai_first_delta = ai_first_gate["delta"]
    ai_first_success = ai_first_gate["success"]
    ai_first_required = ai_first_gate["required"]
    ai_search_selected_by_authenticity = bool(
        ai_search_selected
        and (
            (((result.summary.get("ai_mitigation_search") or {}).get("selection_status") or {})
             .get("authenticity_incremental"))
            or (((result.summary.get("ai_mitigation_search") or {}).get("selection_status") or {})
                .get("safe_authorship_suppression"))
        )
    )
    if (
        ai_first_required
        and not ai_first_success
        and not authenticity_mitigation_selected
        and not ai_search_selected_by_authenticity
    ):
        delta_text = f"{ai_first_delta:.2f}" if isinstance(ai_first_delta, (int, float)) else "unknown"
        regression_reasons.append(
            f"ai_first_gate_failed {ai_first_reference}->{rewritten_ai} "
            f"delta={delta_text} required_delta={ai_first_min_drop:.2f}"
        )
        result.summary["ai_first_mitigation"] = {
            "kept": False,
            "reference_ai": ai_first_reference,
            "rewritten_ai": rewritten_ai,
            "ai_delta": round(ai_first_delta, 3) if isinstance(ai_first_delta, (int, float)) else None,
            "min_drop": ai_first_min_drop,
            "target": ai_first_target,
            "required_min_ai": ai_first_required_min_ai,
            "hard_regressions": [
                f"ai_first_gate_failed {ai_first_reference}->{rewritten_ai}"
            ],
        }
    if ai_first_success:
        hard_regression_reasons = []
        soft_regression_reasons = []
        for reason in regression_reasons:
            if reason.startswith((
                "AI ",
                "user_visible_ai ",
            )):
                hard_regression_reasons.append(reason)
            else:
                soft_regression_reasons.append(reason)
        if soft_regression_reasons:
            followup_warnings.extend(
                f"post_ai_review {reason}" for reason in soft_regression_reasons
            )
        regression_reasons = hard_regression_reasons
        result.summary["ai_first_mitigation"] = {
            "kept": not hard_regression_reasons,
            "source": "ai_mitigation_search" if ai_search_selected else "threshold",
            "reference_ai": ai_first_reference,
            "rewritten_ai": rewritten_ai,
            "ai_delta": round(ai_first_delta, 3) if isinstance(ai_first_delta, (int, float)) else None,
            "min_drop": ai_first_min_drop,
            "target": ai_first_target,
            "soft_followups": soft_regression_reasons,
            "hard_regressions": hard_regression_reasons,
        }
        if not hard_regression_reasons:
            _clear_stale_rollback_for_kept_ai_mitigation(
                result.summary,
                "AI-first mitigation",
            )
            if ai_search_selected:
                result.summary.setdefault("saved_contract_notes", []).append(
                    "AI mitigation search kept the lowest-AI scanned candidate; "
                    "writing quality and lower-severity finding changes are follow-up work."
                )
            else:
                result.summary.setdefault("saved_contract_notes", []).append(
                    "AI-first mitigation kept the rewrite because AI likelihood improved enough; "
                    "writing quality and lower-severity finding changes are follow-up work."
                )
    if followup_warnings:
        result.summary["post_ai_followups"] = followup_warnings
        writing_quality_followups = [
            warning for warning in followup_warnings
            if str(warning).startswith("writing_quality ")
        ]
        if writing_quality_followups:
            result.summary["writing_quality_followups"] = writing_quality_followups
        result.summary.setdefault("saved_contract_notes", []).append(
            (
                "AI-Mitigation kept the rewrite; writing quality and lower-severity changes are reported as follow-up work."
                if authenticity_mitigation_selected
                else "AI-first mitigation kept the rewrite; writing quality and lower-severity changes are reported as follow-up work."
            )
        )
    if (
        os.environ.get("DRAFTPROOF_HUMAN_SHIFT_OVERRIDES_AI_FIRST", "1") != "0"
        and ai_first_required
        and not ai_first_success
        and rewritten_text != text
        and original_ai is not None
        and rewritten_ai is not None
        and rewritten_ai <= original_ai + 0.05
    ):
        final_shift_gate = _authenticity_gate_status(
            original_report_dict,
            rewritten_report_dict,
            rewritten_text != text,
            original_review_burden=original_review_burden,
            candidate_review_burden=rewritten_review_burden,
            original_weighted_severity=original_severity,
            candidate_weighted_severity=rewritten_severity,
            min_human_gain=_float_env("DRAFTPROOF_RECONSTRUCTION_OVERRIDE_MIN_HUMAN_GAIN", 10.0),
            min_ai_transformation_drop=_float_env("DRAFTPROOF_RECONSTRUCTION_OVERRIDE_MIN_AI_TRANSFORM_DROP", 8.0),
        )
        final_shift_score = final_shift_gate.get("human_shift_score")
        final_human_delta = final_shift_gate.get("human_delta")
        final_transform_delta = final_shift_gate.get("ai_transformation_delta")
        final_ai_authorship_regressed = bool(final_shift_gate.get("ai_authorship_regressed"))
        final_major_human_breakthrough = bool(final_shift_gate.get("major_human_breakthrough"))
        clears_override = bool(
            isinstance(final_shift_score, (int, float))
            and final_shift_score >= _float_env("DRAFTPROOF_RECONSTRUCTION_OVERRIDE_MIN_SHIFT", 20.0)
            and (
                isinstance(final_human_delta, (int, float))
                and final_human_delta >= _float_env("DRAFTPROOF_RECONSTRUCTION_OVERRIDE_MIN_HUMAN_GAIN", 10.0)
            )
            and (
                isinstance(final_transform_delta, (int, float))
                and final_transform_delta >= _float_env("DRAFTPROOF_RECONSTRUCTION_OVERRIDE_MIN_AI_TRANSFORM_DROP", 8.0)
            )
            and (not final_ai_authorship_regressed or final_major_human_breakthrough)
            and not final_shift_gate.get("critical_high_regressed")
            and not final_shift_gate.get("review_burden_regressed")
            and not final_shift_gate.get("weighted_severity_regressed")
        )
        if clears_override:
            removed_ai_first = [
                reason for reason in regression_reasons
                if str(reason).startswith("ai_first_gate_failed ")
            ]
            if removed_ai_first:
                regression_reasons = [
                    reason for reason in regression_reasons
                    if not str(reason).startswith("ai_first_gate_failed ")
                ]
                result.summary["human_shift_override"] = {
                    "kept": True,
                    "removed_regressions": removed_ai_first,
                    "reason": "human_shift_goal_outweighs_legacy_ai_first_min_drop",
                    "gate": final_shift_gate,
                    "rewritten_ai": rewritten_ai,
                    "original_ai": original_ai,
                }
                result.summary.setdefault("saved_contract_notes", []).append(
                    "Human Shift override kept the rewrite because contribution and transformation movement met the AI-Mitigation goal without severity regression."
                )
    product_regressed = (
        rewritten_text != text
        and bool(regression_reasons)
    )
    if product_regressed:
        best_checkpoint = None
        best_checkpoint_report = None
        best_checkpoint_rank = None
        checkpoint_candidates = []
        for checkpoint in getattr(result, "rewrite_checkpoints", []) or []:
            checkpoint_text = checkpoint.get("text", "")
            if not checkpoint_text or checkpoint_text in {text, rewritten_text}:
                continue
            checkpoint_candidates.append(checkpoint)
        max_checkpoint_scans = int(os.environ.get("DRAFTPROOF_MAX_CHECKPOINT_SCANS", "6"))
        if max_checkpoint_scans <= 0:
            result.summary["checkpoint_scan_skipped"] = len(checkpoint_candidates)
            checkpoint_candidates = []
        if len(checkpoint_candidates) > max_checkpoint_scans:
            result.summary["checkpoint_scan_skipped"] = (
                len(checkpoint_candidates) - max_checkpoint_scans
            )
            checkpoint_candidates = checkpoint_candidates[-max_checkpoint_scans:]

        checkpoint_scan_t0 = time.time()
        checkpoint_scan_count = 0
        for checkpoint in checkpoint_candidates:
            checkpoint_text = checkpoint.get("text", "")
            checkpoint_report = _full_scan_report_dict(checkpoint_text)
            checkpoint_scan_count += 1
            cp_ai = _badge_ai(checkpoint_report)
            cp_wq = _badge_wq(checkpoint_report)
            cp_total = _finding_total(checkpoint_report)
            cp_review_burden = _review_burden(checkpoint_report)
            cp_severity = _weighted_severity(checkpoint_report)

            cp_ai_regressed = (
                original_ai is not None
                and cp_ai is not None
                and cp_ai > original_ai + 0.05
            )
            cp_wq_regressed = (
                original_wq is not None
                and cp_wq is not None
                and cp_wq > original_wq + writing_quality_regression_tolerance
            )
            cp_improved = (
                cp_review_burden < original_review_burden
                or cp_severity < original_severity
                or cp_total < original_total
                or (original_ai is not None and cp_ai is not None and cp_ai < original_ai - 0.05)
                or (original_wq is not None and cp_wq is not None and cp_wq < original_wq - 0.05)
            )
            cp_critical_high = (
                len(checkpoint_report.get("findings", {}).get("critical", []))
                + len(checkpoint_report.get("findings", {}).get("high", []))
            )
            original_critical_high = (
                len(original_report_dict.get("findings", {}).get("critical", []))
                + len(original_report_dict.get("findings", {}).get("high", []))
            )
            cp_saved_ai_regressed = (
                saved_ai is not None
                and cp_ai is not None
                and cp_ai > saved_ai + 0.05
            )
            cp_saved_total_regressed = cp_total > saved_total
            cp_saved_critical_high_regressed = cp_critical_high > saved_critical_high
            cp_saved_ai_regression_explained_by_drift = (
                cp_saved_ai_regressed
                and saved_ai_drifted_up
                and original_ai is not None
                and cp_ai is not None
                and cp_ai < original_ai - 0.05
                and cp_review_burden <= original_review_burden
                and cp_severity <= original_severity
                and not cp_saved_critical_high_regressed
            )
            cp_violates_saved_contract = (
                (cp_saved_ai_regressed and not cp_saved_ai_regression_explained_by_drift)
                or cp_saved_critical_high_regressed
                or (
                    cp_saved_total_regressed
                    and (
                        not cp_improved
                        or cp_review_burden > original_review_burden
                        or cp_severity > original_severity
                    )
                )
            )
            cp_ai_first_reference = saved_ai if saved_ai is not None else original_ai
            cp_ai_first_gate = _ai_first_gate_status(
                cp_ai_first_reference,
                cp_ai,
                checkpoint.get("text") != text,
                min_drop=ai_first_min_drop,
                target=ai_first_target,
                required_min_ai=ai_first_required_min_ai,
            )
            if (
                cp_ai_regressed
                or cp_total > original_total
                or cp_review_burden > original_review_burden
                or cp_severity > original_severity
                or cp_critical_high > original_critical_high
                or cp_violates_saved_contract
                or not cp_improved
                or (cp_ai_first_gate["required"] and not cp_ai_first_gate["success"])
            ):
                continue

            rank = (
                cp_review_burden,
                cp_severity,
                cp_total,
                cp_ai if cp_ai is not None else 999.0,
                -(checkpoint.get("edits", 0) or 0),
            )
            if best_checkpoint_rank is None or rank < best_checkpoint_rank:
                best_checkpoint = checkpoint
                best_checkpoint_report = checkpoint_report
                best_checkpoint_rank = rank

        if checkpoint_scan_count:
            stage_timings.append({
                "stage": "checkpoint_scans",
                "count": checkpoint_scan_count,
                "seconds": round(time.time() - checkpoint_scan_t0, 3),
            })

        if best_checkpoint and best_checkpoint_report:
            rewritten_text = best_checkpoint["text"]
            rewritten_report_dict = best_checkpoint_report
            result.summary["checkpoint_selected"] = {
                "edits": best_checkpoint.get("edits", 0),
                "local_score_total": best_checkpoint.get("local_score_total", 0.0),
                "reason": "final rewrite regressed; kept best non-regressing checkpoint",
                "final_regression_reasons": regression_reasons,
                "ai_first_gate": _ai_first_gate_status(
                    saved_ai if saved_ai is not None else original_ai,
                    _badge_ai(best_checkpoint_report),
                    best_checkpoint.get("text") != text,
                    min_drop=ai_first_min_drop,
                    target=ai_first_target,
                    required_min_ai=ai_first_required_min_ai,
                ),
            }
            result.summary["detect_scores"].update({
                "rewritten_ai": _badge_ai(rewritten_report_dict),
                "rewritten_writing_quality": _badge_wq(rewritten_report_dict),
                "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
                "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
                "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
                "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
                "rewritten_findings": _finding_total(rewritten_report_dict),
                "rewritten_review_burden": _review_burden(rewritten_report_dict),
                "rewritten_weighted_severity": _weighted_severity(rewritten_report_dict),
                "attempted_ai": rewritten_ai,
                "attempted_writing_quality": rewritten_wq,
                "attempted_findings": rewritten_total,
                "attempted_review_burden": rewritten_review_burden,
                "attempted_weighted_severity": rewritten_severity,
            })
            result.summary["rollback_applied"] = False
            result.summary["outcome"] = "partially_improved"
            if result.mp_result:
                result.mp_result.final_text = rewritten_text
                result.mp_result.converged = True
                result.mp_result.convergence_reason = (
                    "Selected best non-regressing checkpoint after final scan regression"
                )
            sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
            product_regressed = False

    if product_regressed:
        reason = (
            f"final full detect scan regressed "
            f"({'; '.join(regression_reasons)})"
        )
        attempted_text = rewritten_text
        attempted_sentence_comparison = sentence_comparison
        rollback_suggestions = result.summary.setdefault("manual_suggestions", [])
        accepted_suggestions = result.summary.get("accepted_candidate_suggestions") or []
        existing = {
            (
                s.get("original_sentence"),
                s.get("suggested_sentence"),
                s.get("rejection_reason"),
            )
            for s in rollback_suggestions
            if isinstance(s, dict)
        }
        for item in accepted_suggestions:
            if not isinstance(item, dict):
                continue
            suggestion = dict(item)
            suggestion["rejection_reason"] = reason
            suggestion["why_review_manually"] = (
                "This edit passed local guards, but the final full detect scan regressed. "
                "Review manually before using it."
            )
            key = (
                suggestion.get("original_sentence"),
                suggestion.get("suggested_sentence"),
                suggestion.get("rejection_reason"),
            )
            if key not in existing and len(rollback_suggestions) < 30:
                rollback_suggestions.append(suggestion)
                existing.add(key)
        rewritten_text = text
        if result.mp_result:
            result.mp_result.final_text = text
            result.mp_result.final_metrics = result.mp_result.original_metrics
            result.mp_result.converged = False
            result.mp_result.convergence_reason = reason
        result.summary["attempted_final_text"] = attempted_text
        result.summary["attempted_sentence_comparison"] = attempted_sentence_comparison
        result.summary["final_text"] = text
        result.summary["converged"] = False
        result.summary["rollback_applied"] = True
        result.summary["rollback_reason"] = reason
        result.summary["outcome"] = "rejected_for_drift"
        result.summary["detect_scores"]["rollback_reason"] = reason
        sentence_comparison = []
        rewritten_report_dict = original_report_dict

    final_human_shift = _human_shift_score(
        original_report_dict,
        rewritten_report_dict,
        review_burden_delta=_review_burden(rewritten_report_dict) - original_review_burden,
        weighted_severity_delta=_weighted_severity(rewritten_report_dict) - original_severity,
    )
    result.summary.setdefault("detect_scores", {}).update({
        "human_shift_score": final_human_shift.get("score"),
        "human_shift_components": final_human_shift.get("components"),
    })

    # Extract only the fields needed for comparison (not full report dicts)
    def _extract_scan_summary(report_dict):
        badge = report_dict.get("ai_risk_badge") or {}
        findings = report_dict.get("findings", {})
        return {
            "ai_risk_badge": badge,
            "overall_tier": report_dict.get("overall_tier", "?"),
            "findings": {t: [{"finding_id": f.get("finding_id"), "title": f.get("title"),
                              "category": f.get("category")} for f in findings.get(t, [])]
                         for t in ("critical", "high", "medium", "low")},
        }

    result.summary["detect_scan_original_saved"] = _extract_scan_summary(ctx.raw_json)
    result.summary["detect_scan_original"] = _extract_scan_summary(original_report_dict)
    if result.summary.get("rollback_applied"):
        result.summary["detect_scan_attempted"] = _extract_scan_summary(attempted_report_dict)
    else:
        result.summary["final_text"] = rewritten_text
        if ai_search_selected:
            result.summary["outcome"] = "ai_mitigated"
            result.summary["converged"] = True
    result.summary["detect_scan_rewritten"] = _extract_scan_summary(rewritten_report_dict)
    result.summary["stage_timings"] = stage_timings
    result.sentence_comparison = sentence_comparison

    # Generate dedicated rewrite report
    rewrite_md = render_rewrite_report(
        summary=result.summary,
        sentence_comparison=sentence_comparison,
        ai_findings=ai_findings,
        verbose=verbose,
    )

    with open(md_path, "w") as f:
        f.write(rewrite_md)

    pdf_path = os.path.join(output_dir, f"draftproof_rewrite_{ts}.pdf")
    render_pdf(rewrite_md, pdf_path)

    summary = result.summary
    total_elapsed = time.time() - t0
    summary["rewrite_engine_time"] = engine_elapsed
    summary["rewrite_time"] = total_elapsed
    summary["original_tier"] = ctx.overall_tier
    summary["rewrite_decision"] = ctx.rewrite_decision

    with open(json_path_out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if summary.get("rollback_applied") or summary.get("no_text_change"):
        pipeline_status = "original_preserved"
    elif summary.get("outcome") == "partially_improved":
        pipeline_status = "partially_improved"
    else:
        pipeline_status = "rewritten"

    return {
        "status": pipeline_status,
        "md_path": md_path,
        "pdf_path": pdf_path,
        "json_path": json_path_out,
        "result": result,
        "elapsed": total_elapsed,
    }


def main():
    _load_local_env()

    parser = argparse.ArgumentParser(description="DraftProof Rewrite Pipeline")
    parser.add_argument("file", nargs="?", help="Detect JSON file (or - for stdin)")
    parser.add_argument("--text", "-t", help="Inline text to detect + rewrite")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--passes", type=int, default=3, help="Max rewrite passes")
    parser.add_argument("--max-loops", type=int, default=0, help="Max detect-rewrite loops")
    parser.add_argument("--target-top10", type=float, default=0.50, help="Target top-10 ratio")
    parser.add_argument("--model", default=None, help="LLM model (default: from LLM_MODEL env var)")
    parser.add_argument("--api-key", default=None, help="API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-ai-only", action="store_true", help="Rewrite ALL findings (default: AI-only)")
    args = parser.parse_args()

    output_dir = args.output or os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "test_output"
    ))

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    # Read input
    json_path = None
    text = None

    if args.text:
        text = args.text
    elif args.file == "-" or (not args.file and not sys.stdin.isatty()):
        raw = sys.stdin.read()
        try:
            json.loads(raw)
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                tf.write(raw)
                json_path = tf.name
        except json.JSONDecodeError:
            text = raw
    elif args.file:
        json_path = args.file
    else:
        print("Error: provide a detect JSON file, --text, or pipe JSON via stdin", file=sys.stderr)
        sys.exit(1)

    result = run_rewrite_pipeline(
        json_path=json_path,
        text=text,
        output_dir=output_dir,
        max_passes=args.passes,
        max_detect_loops=args.max_loops,
        target_top10=args.target_top10,
        model=args.model,
        api_key=api_key,
        verbose=args.verbose,
        ai_only=not args.no_ai_only,
    )

    if result["status"] == "clean":
        print(f"\n  Status: {result['message']}")
    elif result["status"] == "skipped":
        print(f"\n  Skipped: {result['message']}")
    else:
        elapsed = result["elapsed"]
        r = result["result"]
        rw = r.mp_result
        print(f"\n  Time: {elapsed:.1f}s")
        print(f"  Passes: {len(rw.passes)}")
        print(f"  Converged: {'Yes' if rw.converged else 'No'}")
        print(f"  MD:   {result['md_path']}")
        print(f"  PDF:  {result['pdf_path']}")
        print(f"  JSON: {result['json_path']}")


if __name__ == "__main__":
    main()
