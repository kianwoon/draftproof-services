"""Global candidate ledger helpers for post-AI-search controllers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

from rewrite_controller import CandidateLedger, RewriteRunBudget, build_candidate_record
from rewrite_pipeline_core.scoring.profiles import (
    _ai_footprint_profile,
    _contribution_scores,
    _integrity_scores,
    _turnitin_like_ai_profile,
)


@dataclass(frozen=True)
class GlobalControllerLedgerDeps:
    split_sentences: Callable[[str], list[str]]
    strict_ai_safe_band_status: Callable[[dict | None], dict]
    review_burden: Callable[[dict | None], int | float]
    weighted_severity: Callable[[dict | None], int | float]
    critical_high_count: Callable[[dict | None], int | float]
    finding_total: Callable[[dict | None], int | float]


def controller_changed_sentence_ratio(before: str, after: str, *, deps: GlobalControllerLedgerDeps) -> float:
    before_sentences = [re.sub(r"\s+", " ", s).strip() for s in deps.split_sentences(before)]
    after_sentences = [re.sub(r"\s+", " ", s).strip() for s in deps.split_sentences(after)]
    if not before_sentences and not after_sentences:
        return 0.0
    return round(1.0 - SequenceMatcher(None, before_sentences, after_sentences).ratio(), 3)


def controller_metrics(
    candidate_report: dict,
    current_report: dict,
    before_text: str,
    after_text: str,
    *,
    original_report: dict | None,
    deps: GlobalControllerLedgerDeps,
) -> dict:
    candidate_integrity = _integrity_scores(candidate_report)
    current_integrity = _integrity_scores(current_report)
    candidate_contribution = _contribution_scores(candidate_report)
    current_contribution = _contribution_scores(current_report)
    return {
        "turnitin_profile": _turnitin_like_ai_profile(candidate_report),
        "current_turnitin_profile": _turnitin_like_ai_profile(current_report),
        "original_turnitin_profile": _turnitin_like_ai_profile(original_report),
        "strict_safe": deps.strict_ai_safe_band_status(candidate_report),
        "footprint": _ai_footprint_profile(candidate_report),
        "current_footprint": _ai_footprint_profile(current_report),
        "ai_authorship": candidate_integrity.get("ai_authorship"),
        "current_ai_authorship": current_integrity.get("ai_authorship"),
        "ai_transformation": candidate_contribution.get("ai_transformation"),
        "current_ai_transformation": current_contribution.get("ai_transformation"),
        "review_burden": deps.review_burden(candidate_report),
        "current_review_burden": deps.review_burden(current_report),
        "weighted_severity": deps.weighted_severity(candidate_report),
        "current_weighted_severity": deps.weighted_severity(current_report),
        "critical_high": deps.critical_high_count(candidate_report),
        "current_critical_high": deps.critical_high_count(current_report),
        "finding_total": deps.finding_total(candidate_report),
        "current_finding_total": deps.finding_total(current_report),
        "changed_sentence_ratio": controller_changed_sentence_ratio(before_text, after_text, deps=deps),
    }


def controller_record(
    *,
    stage: str,
    strategy: str | None,
    candidate_text: str,
    candidate_report: dict,
    original_text: str,
    original_report: dict | None,
    current_text: str,
    current_report: dict,
    deps: GlobalControllerLedgerDeps,
) -> dict:
    return build_candidate_record(
        stage=stage,
        strategy=strategy,
        text=candidate_text,
        report=candidate_report,
        original_text=original_text,
        original_report=original_report,
        current_text=current_text,
        current_report=current_report,
        metrics=controller_metrics(
            candidate_report,
            current_report,
            current_text,
            candidate_text,
            original_report=original_report,
            deps=deps,
        ),
    )


def global_phase_budget_skip(
    budget: RewriteRunBudget,
    stage: str,
    *,
    min_seconds: float = 5.0,
    min_scans: int = 1,
    min_llm_calls: int = 0,
) -> dict | None:
    if budget.can_run(
        min_seconds=min_seconds,
        min_scans=min_scans,
        min_llm_calls=min_llm_calls,
    ):
        return None
    skipped = budget.skip_reason(
        stage,
        min_seconds=min_seconds,
        min_scans=min_scans,
        min_llm_calls=min_llm_calls,
    )
    return {
        "enabled": False,
        "selected": False,
        "reason": skipped.get("reason"),
        "global_controller_skip": skipped,
    }


def global_controller_phase_accepted(
    *,
    ledger: CandidateLedger,
    stage: str,
    phase_result: dict,
    stored_result: dict,
    original_text: str,
    original_report: dict | None,
    current_text: str,
    current_report: dict,
    deps: GlobalControllerLedgerDeps,
) -> bool:
    selected_report = phase_result.get("selected_report")
    selected_text = phase_result.get("selected_text")
    if not isinstance(selected_report, dict) or not isinstance(selected_text, str):
        decision = {"accepted": False, "reason": "missing_selected_text_or_report"}
    else:
        record = controller_record(
            stage=stage,
            strategy=phase_result.get("selected_strategy") or phase_result.get("strategy"),
            candidate_text=selected_text,
            candidate_report=selected_report,
            original_text=original_text,
            original_report=original_report,
            current_text=current_text,
            current_report=current_report,
            deps=deps,
        )
        decision = ledger.consider(record)
    stored_result["global_controller_decision"] = decision
    phase_result["global_controller_decision"] = decision
    if not decision.get("accepted"):
        phase_result["selected"] = False
        stored_result["selected"] = False
        stored_result["global_selected_rejected"] = True
        return False
    return True
