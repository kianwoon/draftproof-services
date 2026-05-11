"""Global candidate ledger and selector for rewrite phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .quality_gate import evaluate_text_quality_regression


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _score(profile: dict | None) -> float | None:
    if not isinstance(profile, dict):
        return None
    return _num(profile.get("score") or profile.get("turnitin_like_ai_score"))


def build_candidate_record(
    *,
    stage: str,
    strategy: str | None,
    text: str,
    report: dict,
    original_text: str,
    original_report: dict,
    current_text: str,
    current_report: dict,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    profile = metrics.get("turnitin_profile") or {}
    original_profile = metrics.get("original_turnitin_profile") or {}
    current_profile = metrics.get("current_turnitin_profile") or {}
    components = profile.get("components") if isinstance(profile.get("components"), dict) else {}
    current_components = current_profile.get("components") if isinstance(current_profile.get("components"), dict) else {}
    strict_safe = metrics.get("strict_safe") or {}
    footprint = metrics.get("footprint") or {}
    quality = evaluate_text_quality_regression(
        current_text,
        text,
        changed_sentence_ratio=metrics.get("changed_sentence_ratio"),
    )
    score = _score(profile)
    current_score = _score(current_profile)
    original_score = _score(original_profile)
    positive_burden = _num(profile.get("positive_ai_burden") or profile.get("raw_positive_score"))
    current_positive_burden = _num(current_profile.get("positive_ai_burden") or current_profile.get("raw_positive_score"))
    return {
        "stage": stage,
        "strategy": strategy,
        "text": text,
        "report": report,
        "quality": quality,
        "score": score,
        "current_score": current_score,
        "original_score": original_score,
        "score_drop_vs_current": (
            round(current_score - score, 3)
            if current_score is not None and score is not None else None
        ),
        "score_drop_vs_original": (
            round(original_score - score, 3)
            if original_score is not None and score is not None else None
        ),
        "positive_ai_burden": positive_burden,
        "positive_ai_burden_drop_vs_current": (
            round(current_positive_burden - positive_burden, 3)
            if current_positive_burden is not None and positive_burden is not None else None
        ),
        "topk_calibrated_risk": _num(profile.get("topk_calibrated_risk") or components.get("topk_calibrated_risk")),
        "current_topk_calibrated_risk": _num(current_profile.get("topk_calibrated_risk") or current_components.get("topk_calibrated_risk")),
        "ai_likelihood": _num(profile.get("ai_likelihood") or components.get("ai_likelihood")),
        "current_ai_likelihood": _num(current_profile.get("ai_likelihood") or current_components.get("ai_likelihood")),
        "rewrite_smoothness": _num(profile.get("rewrite_smoothness") or components.get("rewrite_smoothness")),
        "current_rewrite_smoothness": _num(current_profile.get("rewrite_smoothness") or current_components.get("rewrite_smoothness")),
        "external_ai_flag_risk": _num(footprint.get("external_ai_flag_risk")),
        "current_external_ai_flag_risk": _num((metrics.get("current_footprint") or {}).get("external_ai_flag_risk")),
        "ai_authorship": _num(metrics.get("ai_authorship")),
        "current_ai_authorship": _num(metrics.get("current_ai_authorship")),
        "ai_transformation": _num(metrics.get("ai_transformation")),
        "current_ai_transformation": _num(metrics.get("current_ai_transformation")),
        "review_burden": int(metrics.get("review_burden") or 0),
        "current_review_burden": int(metrics.get("current_review_burden") or 0),
        "weighted_severity": int(metrics.get("weighted_severity") or 0),
        "current_weighted_severity": int(metrics.get("current_weighted_severity") or 0),
        "critical_high": int(metrics.get("critical_high") or 0),
        "current_critical_high": int(metrics.get("current_critical_high") or 0),
        "finding_total": int(metrics.get("finding_total") or 0),
        "current_finding_total": int(metrics.get("current_finding_total") or 0),
        "strict_safe": strict_safe,
        "target_met": bool(profile.get("target_met")),
        "text_changed": str(text or "").strip() != str(current_text or "").strip(),
    }


@dataclass
class CandidateLedger:
    """Keeps the best safe candidate across all rewrite phases."""

    min_formula_drop: float = 0.05
    min_late_formula_drop_when_pinned: float = 1.0
    target_score: float = 20.0
    current: dict[str, Any] | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def seed(self, record: dict[str, Any]) -> None:
        self.current = record
        self.records.append(self._public(record, accepted=True, reason="seed_current_best"))

    def consider(self, record: dict[str, Any]) -> dict[str, Any]:
        if self.current is None:
            self.seed(record)
            return {"accepted": True, "reason": "seed_current_best"}
        decision = self._decision(record)
        public_record = self._public(record, accepted=decision["accepted"], reason=decision["reason"])
        self.records.append(public_record)
        self.decisions.append(decision | {"stage": record.get("stage"), "strategy": record.get("strategy")})
        if decision["accepted"]:
            self.current = record
        return decision

    def _decision(self, record: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        quality = record.get("quality") or {}
        if not quality.get("passed", True):
            reasons.extend(quality.get("reject_reasons") or ["quality_guard_failed"])
        if not record.get("text_changed"):
            reasons.append("candidate_text_unchanged")
        if record.get("review_burden", 0) > record.get("current_review_burden", 0):
            reasons.append("review_burden_regressed")
        if record.get("weighted_severity", 0) > record.get("current_weighted_severity", 0):
            reasons.append("weighted_severity_regressed")
        if record.get("critical_high", 0) > record.get("current_critical_high", 0):
            reasons.append("critical_high_regressed")
        current_score = _num((self.current or {}).get("score"))
        score = _num(record.get("score"))
        score_drop = (
            current_score - score
            if current_score is not None and score is not None
            else None
        )
        if score_drop is None or score_drop < self.min_formula_drop:
            reasons.append("formula_score_not_improved")
        current_topk = _num(record.get("current_topk_calibrated_risk"))
        topk = _num(record.get("topk_calibrated_risk"))
        current_smooth = _num(record.get("current_rewrite_smoothness"))
        smooth = _num(record.get("rewrite_smoothness"))
        pinned_topk = bool(current_topk is not None and current_topk >= 80.0 and topk is not None and topk >= current_topk - 0.01)
        smoothness_regressed = bool(current_smooth is not None and smooth is not None and smooth > current_smooth + 0.05)
        if pinned_topk and smoothness_regressed:
            reasons.append("pinned_topk_and_smoothness_regressed")
        if pinned_topk and (score_drop is None or score_drop < self.min_late_formula_drop_when_pinned):
            reasons.append("pinned_topk_tiny_formula_gain")
        current_authorship = _num(record.get("current_ai_authorship"))
        authorship = _num(record.get("ai_authorship"))
        if current_authorship is not None and authorship is not None and authorship > current_authorship + 0.05:
            reasons.append("ai_authorship_regressed")
        current_transformation = _num(record.get("current_ai_transformation"))
        transformation = _num(record.get("ai_transformation"))
        if current_transformation is not None and transformation is not None and transformation > current_transformation + 0.05:
            reasons.append("ai_transformation_regressed")
        if reasons:
            return {
                "accepted": False,
                "reason": "; ".join(sorted(set(reasons))),
                "score_drop_vs_current": round(score_drop, 3) if isinstance(score_drop, (int, float)) else None,
                "target_met": bool(record.get("target_met")),
            }
        return {
            "accepted": True,
            "reason": "global_formula_candidate_improved",
            "score_drop_vs_current": round(score_drop, 3) if isinstance(score_drop, (int, float)) else None,
            "target_met": bool(record.get("target_met")),
        }

    def _public(self, record: dict[str, Any], *, accepted: bool, reason: str) -> dict[str, Any]:
        return {
            "stage": record.get("stage"),
            "strategy": record.get("strategy"),
            "accepted": bool(accepted),
            "reason": reason,
            "score": record.get("score"),
            "score_drop_vs_current": record.get("score_drop_vs_current"),
            "score_drop_vs_original": record.get("score_drop_vs_original"),
            "positive_ai_burden_drop_vs_current": record.get("positive_ai_burden_drop_vs_current"),
            "topk_calibrated_risk": record.get("topk_calibrated_risk"),
            "ai_likelihood": record.get("ai_likelihood"),
            "rewrite_smoothness": record.get("rewrite_smoothness"),
            "external_ai_flag_risk": record.get("external_ai_flag_risk"),
            "ai_authorship": record.get("ai_authorship"),
            "ai_transformation": record.get("ai_transformation"),
            "quality": record.get("quality"),
            "target_met": record.get("target_met"),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "version": "global_candidate_ledger_v1",
            "target_score": self.target_score,
            "min_formula_drop": self.min_formula_drop,
            "min_late_formula_drop_when_pinned": self.min_late_formula_drop_when_pinned,
            "current_best": self._public(self.current or {}, accepted=True, reason="current_best") if self.current else None,
            "records": list(self.records),
            "decisions": list(self.decisions),
        }
