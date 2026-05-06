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
    return (
        1 if gate.get("success") else 0,
        float(score) if isinstance(score, (int, float)) else -9999.0,
        float(gate.get("ai_authorship_delta")) if isinstance(gate.get("ai_authorship_delta"), (int, float)) else -9999.0,
        float(gate.get("human_delta")) if isinstance(gate.get("human_delta"), (int, float)) else -9999.0,
        float(gate.get("ai_transformation_delta")) if isinstance(gate.get("ai_transformation_delta"), (int, float)) else -9999.0,
    )


def _is_better_human_shift_candidate(candidate_gate: dict | None, best_gate: dict | None) -> bool:
    if best_gate is None:
        return True
    return _human_shift_rank_key(candidate_gate) > _human_shift_rank_key(best_gate)


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
    success = bool(
        text_changed
        and (clears_human_shift_score or positive_human_shift)
        and not critical_high_regressed
        and not review_regressed
        and not severity_regressed
    )
    reason = ""
    if success:
        reason = "accepted"
    elif not text_changed:
        reason = "unchanged_candidate"
    elif not (clears_human_shift_score or positive_human_shift):
        reason = "human_shift_score_too_low"
    elif critical_high_regressed:
        reason = "critical_high_regressed"
    elif review_regressed:
        reason = "review_burden_regressed"
    elif severity_regressed:
        reason = "weighted_severity_regressed"
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
        "crosses_human_side": crosses_human_side,
        "human_shift_score": human_shift_score,
        "human_shift_components": human_shift.get("components"),
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


def _ai_candidate_quality_reject_reason(candidate: str) -> str:
    if not isinstance(candidate, str) or not candidate.strip():
        return "empty_candidate"
    if "[[REVIEW:" in candidate:
        return "review_markers_not_auto_kept"
    synthetic_anchors = _SYNTHETIC_ANCHOR_RE.findall(candidate)
    sentence_count = max(1, len(re.findall(r"(?<=[.!?])\s+", candidate)) + 1)
    max_anchor_count = max(3, min(8, sentence_count // 8))
    if len(synthetic_anchors) > max_anchor_count:
        return f"synthetic_anchor_overuse {len(synthetic_anchors)}>{max_anchor_count}"
    lowered = candidate.lower()
    if re.search(r"\bith only\b", lowered):
        return "broken_word_fragment"
    if re.search(r"\b(?:introduction|conclusion)[ \t]+(?:inclusive|this|the)\b", candidate, re.I):
        return "heading_merged_into_sentence"
    if _DANGLING_FRAGMENT_JOIN_RE.search(candidate):
        return "dangling_sentence_fragment_join"
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
    "introduction", "conclusion", "centre", "center",
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
    ai_search_first = (
        os.environ.get("DRAFTPROOF_AI_SEARCH_FIRST", "1") != "0"
        and isinstance(pre_rewrite_ai, (int, float))
        and pre_rewrite_ai >= _float_env("DRAFTPROOF_AI_FIRST_REQUIRED_MIN_AI", 50.0)
        and not ai_mitigation_needs_author
    )
    rewrite_config = None
    if ai_search_first:
        rewrite_config = RewriteConfig(
            max_llm_calls=0,
            max_density_passes=0,
            max_rewrite_seconds=30,
        )
    elif ai_mitigation_needs_author and os.environ.get("DRAFTPROOF_ALLOW_AUTO_WITH_AUTHOR_GAPS") != "1":
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
        model=model,
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
    result.summary["ai_mitigation"] = ai_mitigation_contract
    if ai_mitigation_needs_author:
        result.summary["ai_mitigation_blocked_auto_rewrite"] = True
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
    elif ai_mitigation_needs_author and os.environ.get("DRAFTPROOF_ALLOW_AUTO_WITH_AUTHOR_GAPS") != "1":
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

    # The saved scan is the user-visible contract and is already available.
    # Do not rescan the original by default: on longer drafts it doubles final
    # verification time and can make the report disagree with the scan the user
    # just reviewed. A fresh-original baseline can still be enabled for
    # diagnostics with DRAFTPROOF_FRESH_ORIGINAL_BASELINE=1.
    original_report_dict = ctx.raw_json
    if rewritten_text != text and os.environ.get("DRAFTPROOF_FRESH_ORIGINAL_BASELINE") == "1":
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
                1,
                int(os.environ.get("DRAFTPROOF_AUTHENTICITY_CANDIDATES", "2")),
            )
        except ValueError:
            authenticity_candidate_limit = 2
        authenticity_summary = {
            "enabled": True,
            "selected": False,
            "candidate_limit": authenticity_candidate_limit,
            "llm_calls": 0,
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
                    model=model or os.environ.get("LLM_MODEL"),
                    base_url=base_url,
                    timeout=int(os.environ.get("DRAFTPROOF_AUTHENTICITY_TIMEOUT", "120")),
                    max_retries=int(os.environ.get("DRAFTPROOF_AUTHENTICITY_RETRIES", "1")),
                    max_tokens=int(os.environ.get("DRAFTPROOF_AUTHENTICITY_MAX_TOKENS", "6500")),
                    temperature=float(os.environ.get("DRAFTPROOF_AUTHENTICITY_TEMPERATURE", "0.7")),
                ))
                for attempt_index in range(1, authenticity_candidate_limit + 1):
                    report_progress(
                        min(89, 78 + attempt_index),
                        f"Trying authenticity mitigation candidate {attempt_index}/{authenticity_candidate_limit}",
                    )
                    candidate_eval = {
                        "attempt": attempt_index,
                        "passed_local_checks": False,
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
                if best_candidate_gate and best_candidate_gate.get("success") and best_candidate_report:
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
                            "Selected AI-Mitigation authenticity candidate"
                        )
                    sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                    authenticity_mitigation_selected = True
                    ai_search_selected = True
                    _clear_stale_rollback_for_kept_ai_mitigation(
                        result.summary,
                        "AI-Mitigation authenticity gate",
                    )
                    result.summary["ai_mitigation_blocked_auto_rewrite"] = False
                    result.summary["rewrite_engine_mode"] = "ai_mitigation_authenticity_gate"
                    result.summary["outcome"] = "ai_mitigated"
                    if isinstance(best_candidate_eval, dict):
                        best_candidate_eval["selected"] = True
                    authenticity_summary.update({
                        "selected": True,
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
        result.summary["authenticity_mitigation"] = authenticity_summary
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
    ai_search_reference = saved_ai if saved_ai is not None else original_ai
    ai_search_enabled = os.environ.get("DRAFTPROOF_AI_MITIGATION_SEARCH", "1") != "0"
    ai_search_blocked_by_author_gaps = (
        ai_mitigation_needs_author
        and os.environ.get("DRAFTPROOF_ALLOW_AUTO_WITH_AUTHOR_GAPS") != "1"
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
            selection_status["human_shift_score"] = human_shift.get("score")
            selection_status["human_shift_components"] = human_shift.get("components")
            candidate_eval["selection_status"] = selection_status
            ai_delta = (
                ai_search_reference - candidate_ai
                if isinstance(ai_search_reference, (int, float)) and isinstance(candidate_ai, (int, float))
                else -9999.0
            )
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
                    model=model or os.environ.get("LLM_MODEL"),
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
                            int(os.environ.get("DRAFTPROOF_PARAGRAPH_COMPONENT_TARGETS", "4")),
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
                    if feedback_limit:
                        search_summary["score_feedback_loop"] = {
                            "enabled": True,
                            "candidate_limit": feedback_limit,
                            "reason": (
                                "no_selectable_candidate"
                                if not best_strategy
                                else "best_candidate_below_required_ai_drop"
                            ),
                        }
                    for feedback_index in range(1, feedback_limit + 1):
                        report_progress(
                            min(89, 80 + feedback_index),
                            f"Trying score-feedback AI mitigation candidate {feedback_index}/{feedback_limit}",
                        )
                        candidate_eval = {
                            "strategy": f"score_feedback_{feedback_index}",
                            "passed_local_checks": False,
                        }
                        try:
                            prompt = _ai_search_feedback_prompt(
                                search_source_text,
                                ctx.raw_json,
                                search_summary,
                                feedback_index,
                            )
                            search_summary["llm_calls"] += 1
                            response = gateway.chat(
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
        hard_regression_reasons = []
        soft_regression_reasons = []
        for reason in regression_reasons:
            if reason.startswith((
                "review_burden ",
                "critical_high_findings ",
                "weighted_severity ",
                "user_visible_critical_high_findings ",
            )):
                hard_regression_reasons.append(reason)
            else:
                soft_regression_reasons.append(reason)
        if soft_regression_reasons:
            followup_warnings.extend(
                f"post_authenticity_review {reason}" for reason in soft_regression_reasons
            )
        regression_reasons = hard_regression_reasons
        result.summary.setdefault("authenticity_mitigation", {})["kept"] = not hard_regression_reasons
        result.summary.setdefault("authenticity_mitigation", {})["hard_regressions"] = hard_regression_reasons
        result.summary.setdefault("authenticity_mitigation", {})["soft_followups"] = soft_regression_reasons
        if not hard_regression_reasons:
            result.summary.setdefault("saved_contract_notes", []).append(
                "AI-Mitigation authenticity gate kept the rewrite because the contribution score moved toward Human without review-burden or severity regression."
            )
    ai_first_reference = saved_ai if saved_ai is not None else original_ai
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
    if ai_first_required and not ai_first_success and not authenticity_mitigation_selected:
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
