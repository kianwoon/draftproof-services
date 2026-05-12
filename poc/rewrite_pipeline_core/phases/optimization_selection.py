from __future__ import annotations

from rewrite_pipeline_core.config import _float_env
from rewrite_pipeline_core.phases.micro_texture import (
    _locality_score,
    _repair_aggression_score,
)
from rewrite_pipeline_core.prompts.reconstruction_helpers import _human_gain_stage_target


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
