"""Production adapter for rewrite V5 residual cluster comb-through."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from report.pdf import render_pdf
from report.render_rewrite import render_rewrite_report
from rewrite_v2.goal_contract import RewriteGoalStatus, evaluate_rewrite_goal
from rewrite_v2.pipeline import _extract_original_text, _sentence_comparison
from rewrite_v3.document_units import word_count
from rewrite_v3.pipeline import _scan_report
from rewrite_v4.production import (
    _compact_scan_for_rewrite_report,
    _detect_scores,
    _truncate_text,
)

from .experiment import _score_summary
from .residual_comb import (
    _adaptive_cutoff_runtime_budget_seconds,
    _compact_density_gate,
    _density_gate_for_report,
    _with_v5_density_gate,
    run_v5_residual_cluster_comb_experiment,
)


AUTHOR_PROXY_REVIEW_STATUS = "rewrite_candidate_generated_needs_author_review"
AUTHOR_PROXY_WARNING = "author_proxy_candidate_requires_review"
PARTIAL_NOT_STRICT_STATUS = "partial_candidate_not_strict_safe"


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _first_env(*names: str) -> str | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return None


def _list_env(*names: str) -> list[str]:
    raw = _first_env(*names)
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _optional_bool_env(*names: str) -> bool | None:
    raw = _first_env(*names)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _production_config() -> dict[str, Any]:
    return {
        "planner_model": _first_env(
            "DRAFTPROOF_REWRITE_V5_PLANNER_MODEL",
            "DRAFTPROOF_REWRITE_V5_NORMALIZER_MODEL",
        ) or "z-ai/glm-5.1",
        "max_rounds": _int_env("DRAFTPROOF_REWRITE_V5_MAX_ROUNDS", 6, minimum=1, maximum=10),
        "variant_count": _int_env("DRAFTPROOF_REWRITE_V5_VARIANTS", 5, minimum=1, maximum=5),
        "retune_variant_count": _int_env("DRAFTPROOF_REWRITE_V5_RETUNE_VARIANTS", 5, minimum=1, maximum=5),
        "adaptive_writer_enabled": _bool_env("DRAFTPROOF_REWRITE_V5_ADAPTIVE_WRITER", True),
        "adaptive_initial_variant_count": _int_env("DRAFTPROOF_REWRITE_V5_ADAPTIVE_INITIAL_VARIANTS", 2, minimum=1, maximum=5),
        "adaptive_retune_variant_count": _int_env("DRAFTPROOF_REWRITE_V5_ADAPTIVE_RETUNE_VARIANTS", 2, minimum=1, maximum=5),
        "direct_scanner_leapfrog_rounds": _int_env("DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_LEAPFROG_ROUNDS", 0, minimum=0, maximum=12),
        "direct_scanner_leapfrog_variants": _int_env("DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_LEAPFROG_VARIANTS", 5, minimum=1, maximum=5),
        "direct_scanner_leapfrog_batches": _int_env("DRAFTPROOF_REWRITE_V5_DIRECT_SCANNER_LEAPFROG_BATCHES", 2, minimum=1, maximum=3),
        "risky_window_cleanup_rounds": _int_env("DRAFTPROOF_REWRITE_V5_RISKY_WINDOW_CLEANUP_ROUNDS", 2, minimum=0, maximum=12),
        "unsafe_cluster_cleanup_rounds": _int_env("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_CLEANUP_ROUNDS", 12, minimum=0, maximum=12),
        "unsafe_cluster_stop_after_misses": _int_env("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_STOP_AFTER_MISSES", 3, minimum=0, maximum=12),
        "unsafe_cluster_probe_share": _float_env("DRAFTPROOF_REWRITE_V5_UNSAFE_CLUSTER_PROBE_SHARE", 0.25, minimum=0.0, maximum=1.0),
        "final_risky_window_cleanup_rounds": _int_env("DRAFTPROOF_REWRITE_V5_FINAL_RISKY_WINDOW_CLEANUP_ROUNDS", 2, minimum=0, maximum=12),
        "cleanup_variant_count": _int_env("DRAFTPROOF_REWRITE_V5_CLEANUP_VARIANTS", 5, minimum=1, maximum=5),
        "required_ai_drop": _float_env("DRAFTPROOF_REWRITE_V5_REQUIRED_AI_DROP", 5.0, minimum=0.0, maximum=100.0),
        "runtime_base_seconds": _int_env("DRAFTPROOF_REWRITE_V5_RUNTIME_BASE_SECONDS", 900, minimum=30, maximum=2400),
        "runtime_seconds_per_100_words": _float_env("DRAFTPROOF_REWRITE_V5_RUNTIME_SECONDS_PER_100_WORDS", 40.0, minimum=0.0, maximum=300.0),
        "runtime_min_seconds": _int_env("DRAFTPROOF_REWRITE_V5_RUNTIME_MIN_SECONDS", 900, minimum=60, maximum=2400),
        "runtime_max_seconds": _int_env("DRAFTPROOF_REWRITE_V5_RUNTIME_MAX_SECONDS", 1800, minimum=90, maximum=7200),
        "runtime_soft_limit_buffer_seconds": _int_env("DRAFTPROOF_REWRITE_V5_RUNTIME_SOFT_LIMIT_BUFFER_SECONDS", 60, minimum=30, maximum=1800),
    }


def _ai_mitigation_plan_from_report(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    direct = report.get("ai_mitigation")
    if isinstance(direct, dict):
        return direct
    nested = (
        ((report.get("scan_intelligence") or {}).get("mitigation_inputs") or {})
        .get("ai_mitigation_plan")
    )
    return nested if isinstance(nested, dict) else {}


def _author_proxy_review_cards(plan: dict[str, Any], *, limit: int = 12) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []

    for index, target in enumerate(plan.get("target_segments") or [], start=1):
        if not isinstance(target, dict):
            continue
        needed = str(target.get("user_input_needed") or "").strip()
        if not needed or needed == "none":
            continue
        cards.append({
            "card_id": f"target-{index:02d}",
            "kind": "target_segment",
            "provenance": "needs_author_confirmation",
            "bucket": target.get("bucket"),
            "lever": target.get("lever"),
            "target_text": target.get("text"),
            "where": target.get("paragraph_id") or target.get("sentence_id") or target.get("segment_id"),
            "instruction": target.get("action"),
            "user_input_needed": needed,
            "author_task": "Verify, replace, or remove any drafted context before submission.",
        })
        if len(cards) >= limit:
            return cards

    for index, action in enumerate(plan.get("component_actions") or [], start=1):
        if not isinstance(action, dict):
            continue
        needed = str(action.get("user_input_needed") or "").strip()
        if not needed or needed == "none":
            continue
        cards.append({
            "card_id": f"component-{index:02d}",
            "kind": "component_action",
            "provenance": "needs_author_confirmation",
            "bucket": action.get("bucket"),
            "lever": action.get("lever"),
            "target_text": action.get("component"),
            "where": action.get("component"),
            "instruction": action.get("action"),
            "user_input_needed": needed,
            "author_task": "Confirm this guidance with real author-owned evidence before submission.",
        })
        if len(cards) >= limit:
            return cards

    return cards


def _authorship_evidence_contract(plan: dict[str, Any], review_cards: list[dict[str, Any]], original_text: str) -> dict[str, Any]:
    required_inputs = []
    readiness = plan.get("readiness") if isinstance(plan.get("readiness"), dict) else {}
    for item in readiness.get("required_inputs") or []:
        value = str(item or "").strip()
        if value and value not in required_inputs:
            required_inputs.append(value)
    evidence_slots = []
    for card in review_cards:
        if not isinstance(card, dict):
            continue
        evidence_slots.append({
            "slot_id": card.get("card_id"),
            "kind": card.get("kind"),
            "bucket": card.get("bucket"),
            "target_text": card.get("target_text"),
            "required_input": card.get("user_input_needed"),
            "provenance": card.get("provenance"),
        })
    return {
        "schema_version": "authorship_evidence_contract.v1",
        "basis": "submitted_content_only",
        "original_word_count": word_count(str(original_text or "")),
        "required_inputs": required_inputs,
        "evidence_slots": evidence_slots,
        "rules": [
            "Use submitted draft material as the evidence source of record.",
            "If a claim lacks evidence, narrow the claim instead of inventing support.",
            "Preserve author-owned thesis, stance, examples, source anchors, and citation material.",
            "Mark any provisional bridge as author-review material through provenance/review items.",
        ],
        "kpi_alignment": [
            "Reduce uniform high-probability phrasing by adding grounded specificity from existing draft material.",
            "Reduce qualifying AI density by replacing generic claims with source-supported or narrower claims.",
            "Do not improve scanner texture by adding unsupported facts or fabricated personal evidence.",
        ],
    }


def _build_author_proxy_context(report: dict[str, Any], original_text: str) -> dict[str, Any]:
    plan = _ai_mitigation_plan_from_report(report)
    readiness = plan.get("readiness") if isinstance(plan.get("readiness"), dict) else {}
    review_cards = _author_proxy_review_cards(plan)
    active = bool(readiness.get("requires_user_input") or review_cards)
    if not active:
        return {
            "schema_version": "author_proxy_context.v1",
            "active": False,
            "review_required": False,
            "mode": "none",
            "review_cards": [],
        }
    return {
        "schema_version": "author_proxy_context.v1",
        "active": True,
        "review_required": True,
        "mode": "non_interrupting_author_proxy_draft",
        "primary_mode": plan.get("primary_mode"),
        "required_inputs": readiness.get("required_inputs") or [],
        "review_cards": review_cards,
        "authorship_evidence_contract": _authorship_evidence_contract(plan, review_cards, original_text),
        "allowed_provenance": [
            "source_preserved",
            "inferred_from_draft",
            "needs_author_confirmation",
            "must_replace",
        ],
        "rules": [
            "Continue the rewrite without waiting for the author.",
            "Act only as a drafting proxy using the submitted draft, nearby context, and existing source/citation material.",
            "Produce the strongest polished candidate possible from the submitted content; do not underwrite or leave generic filler.",
            "Improve clarity, specificity, flow, and academic voice while preserving the author's thesis, scope, and available evidence.",
            "Do not invent personal experiences, citations, numbers, dates, named events, institutions, or source facts.",
            "Any added context that is not explicitly present in the draft must remain author-review material.",
            "When evidence is missing, narrow the claim into a precise, readable statement rather than fabricating support.",
        ],
        "quality_bar": {
            "target": "highest_quality_grounded_candidate",
            "basis": "submitted_content_only",
            "priorities": [
                "meaning_fidelity",
                "source_grounded_specificity",
                "coherent_argument_flow",
                "natural_academic_voice",
                "author_review_visibility",
            ],
        },
        "original_word_count": word_count(str(original_text or "")),
    }


def _author_proxy_requires_review(context: dict[str, Any] | None, *, final_text: str, original_text: str) -> bool:
    if not isinstance(context, dict) or not context.get("active"):
        return False
    if str(final_text or "").strip() == str(original_text or "").strip():
        return False
    return bool(context.get("review_required") or context.get("review_cards"))


def _author_review_cards_from_candidate(
    context: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if isinstance(context, dict):
        cards.extend(card for card in context.get("review_cards") or [] if isinstance(card, dict))
    if isinstance(candidate, dict):
        for item in candidate.get("author_review_items") or []:
            if not isinstance(item, dict):
                continue
            cards.append({
                "card_id": item.get("item_id") or f"candidate-{len(cards) + 1:02d}",
                "kind": "candidate_author_review",
                "provenance": item.get("provenance") or "needs_author_confirmation",
                "target_text": item.get("target_text") or item.get("generated_text"),
                "where": item.get("item_id"),
                "instruction": item.get("generated_text") or item.get("target_text"),
                "user_input_needed": item.get("user_input_needed"),
                "author_task": item.get("author_task") or "Verify this generated detail against your own evidence before submission.",
            })
        audit = candidate.get("author_proxy_audit") if isinstance(candidate.get("author_proxy_audit"), dict) else {}
        novel = audit.get("novel_candidate_references") if isinstance(audit.get("novel_candidate_references"), dict) else {}
        for value in list(novel.get("numbers") or []) + list(novel.get("named_references") or []):
            cards.append({
                "card_id": f"novel-reference-{len(cards) + 1:02d}",
                "kind": "candidate_novel_reference",
                "provenance": "needs_author_confirmation",
                "target_text": str(value),
                "where": "rewritten candidate",
                "instruction": "Candidate introduced a concrete reference not present in the submitted source text.",
                "user_input_needed": "Confirm this detail is real and author-owned, or remove it.",
                "author_task": "Verify, replace, or remove this concrete reference before submission.",
            })
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for card in cards:
        key = (str(card.get("kind") or ""), str(card.get("target_text") or card.get("user_input_needed") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(card)
        if len(deduped) >= 12:
            break
    return deduped


def _accepted_author_proxy_review_required(
    accepted_rows: list[dict[str, Any]],
    *,
    context: dict[str, Any] | None,
    final_text: str,
    original_text: str,
) -> bool:
    if _author_proxy_requires_review(context, final_text=final_text, original_text=original_text):
        for row in accepted_rows:
            if not isinstance(row, dict):
                continue
            audit = row.get("author_proxy_audit")
            if not isinstance(audit, dict) or not audit.get("active"):
                return True
    saw_author_proxy_audit = False
    for row in accepted_rows:
        if not isinstance(row, dict):
            continue
        audit = row.get("author_proxy_audit")
        if not isinstance(audit, dict):
            continue
        saw_author_proxy_audit = True
        if not audit.get("active"):
            continue
        safety_gate = audit.get("safety_gate") if isinstance(audit.get("safety_gate"), dict) else {}
        if audit.get("review_required") or safety_gate.get("requires_author_review"):
            return True
    if saw_author_proxy_audit:
        return False
    return _author_proxy_requires_review(
        context,
        final_text=final_text,
        original_text=original_text,
    )


def _strict_safe_band_achieved(goal: dict[str, Any] | None) -> bool:
    if not isinstance(goal, dict):
        return False
    if goal.get("strict_ai_safe_band_achieved") is True:
        return True
    gate = goal.get("ai_footprint_gate") if isinstance(goal.get("ai_footprint_gate"), dict) else {}
    return bool(gate.get("safe_band"))


def _kpi_finalization_status(
    *,
    strict_safe_band_achieved: bool,
    author_review_required: bool,
    accepted_count: int,
    no_text_change: bool,
) -> str:
    if no_text_change:
        return "original_preserved"
    if strict_safe_band_achieved and author_review_required:
        return "strict_safe_author_review_required"
    if strict_safe_band_achieved:
        return "strict_safe_auto_finalized"
    if accepted_count > 0:
        return "partial_candidate_not_strict_safe"
    return "no_safe_candidate"


def _final_goal_from_scan(
    *,
    original_text: str,
    final_text: str,
    original_report: dict[str, Any],
    final_report: dict[str, Any],
    fallback_goal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        goal = evaluate_rewrite_goal(
            original_text=original_text,
            candidate_text=final_text,
            original_report=original_report,
            candidate_report=final_report,
        ).to_dict()
    except Exception:
        goal = dict(fallback_goal or {})
    return _with_v5_density_gate(final_text, final_report, goal)


def _v5_extra_body() -> dict[str, Any] | None:
    extra: dict[str, Any] = {}
    effort = os.environ.get("DRAFTPROOF_REWRITE_V5_REASONING_EFFORT", "none").strip() or "none"
    if _bool_env("DRAFTPROOF_REWRITE_V5_DISABLE_REASONING", True):
        extra["reasoning"] = {"effort": effort, "exclude": True}
        extra["include_reasoning"] = False
    elif _bool_env("DRAFTPROOF_REWRITE_V5_EXCLUDE_REASONING", True):
        extra["reasoning"] = {"exclude": True}
        extra["include_reasoning"] = False
    return extra or None


def _v5_provider_routing() -> dict[str, Any] | None:
    raw_json = _first_env(
        "DRAFTPROOF_REWRITE_V5_PROVIDER_ROUTING_JSON",
        "DRAFTPROOF_OPENROUTER_PROVIDER_ROUTING_JSON",
        "OPENROUTER_PROVIDER_ROUTING_JSON",
        "LLM_PROVIDER_ROUTING_JSON",
    )
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise ValueError("V5 provider routing JSON must be an object")
        return parsed

    provider: dict[str, Any] = {
        "allow_fallbacks": True,
        "sort": _first_env(
            "DRAFTPROOF_REWRITE_V5_PROVIDER_SORT",
            "DRAFTPROOF_OPENROUTER_PROVIDER_SORT",
            "OPENROUTER_PROVIDER_SORT",
        ) or "throughput",
    }
    allow_fallbacks = _optional_bool_env(
        "DRAFTPROOF_REWRITE_V5_ALLOW_FALLBACKS",
        "DRAFTPROOF_OPENROUTER_ALLOW_FALLBACKS",
        "OPENROUTER_ALLOW_FALLBACKS",
    )
    if allow_fallbacks is not None:
        provider["allow_fallbacks"] = allow_fallbacks

    order = _list_env(
        "DRAFTPROOF_REWRITE_V5_PROVIDER_ORDER",
        "DRAFTPROOF_OPENROUTER_PROVIDER_ORDER",
        "OPENROUTER_PROVIDER_ORDER",
    )
    only = _list_env(
        "DRAFTPROOF_REWRITE_V5_PROVIDER_ONLY",
        "DRAFTPROOF_OPENROUTER_PROVIDER_ONLY",
        "OPENROUTER_PROVIDER_ONLY",
    )
    ignore = _list_env(
        "DRAFTPROOF_REWRITE_V5_PROVIDER_IGNORE",
        "DRAFTPROOF_OPENROUTER_PROVIDER_IGNORE",
        "OPENROUTER_PROVIDER_IGNORE",
    )
    if order:
        provider["order"] = order
    if only:
        provider["only"] = only
    if ignore:
        provider["ignore"] = ignore

    data_collection = _first_env(
        "DRAFTPROOF_REWRITE_V5_DATA_COLLECTION",
        "DRAFTPROOF_OPENROUTER_DATA_COLLECTION",
        "OPENROUTER_DATA_COLLECTION",
    )
    if data_collection:
        provider["data_collection"] = data_collection

    require_parameters = _optional_bool_env(
        "DRAFTPROOF_REWRITE_V5_REQUIRE_PARAMETERS",
        "DRAFTPROOF_OPENROUTER_REQUIRE_PARAMETERS",
        "OPENROUTER_REQUIRE_PARAMETERS",
    )
    if require_parameters is not None:
        provider["require_parameters"] = require_parameters
    zdr = _optional_bool_env(
        "DRAFTPROOF_REWRITE_V5_ZDR",
        "DRAFTPROOF_OPENROUTER_ZDR",
        "OPENROUTER_ZDR",
    )
    if zdr is not None:
        provider["zdr"] = zdr
    enforce_distillable_text = _optional_bool_env(
        "DRAFTPROOF_REWRITE_V5_ENFORCE_DISTILLABLE_TEXT",
        "DRAFTPROOF_OPENROUTER_ENFORCE_DISTILLABLE_TEXT",
        "OPENROUTER_ENFORCE_DISTILLABLE_TEXT",
    )
    if enforce_distillable_text is not None:
        provider["enforce_distillable_text"] = enforce_distillable_text
    return provider


def run_rewrite_pipeline_v5(
    *,
    detect_json: dict[str, Any],
    output_dir: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    seed_candidate_texts: list[str] | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    started = time.time()

    def raise_if_canceled() -> None:
        if cancellation_check is not None:
            cancellation_check()

    def progress(percent: int, message: str) -> None:
        raise_if_canceled()
        if progress_callback:
            progress_callback(percent, message)

    progress(62, "Starting V5 cluster rewrite")
    original_text = _extract_original_text(detect_json)
    config = _production_config()
    provider_routing = _v5_provider_routing()
    author_proxy_context = _build_author_proxy_context(detect_json, original_text)
    runtime_budget_seconds = _v5_runtime_budget_seconds(
        original_text,
        config,
        original_report=detect_json,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_version = "rewrite_v5_residual_cluster_comb"

    def emit_checkpoint(checkpoint: dict[str, Any]) -> None:
        raise_if_canceled()
        if checkpoint_callback is None:
            return
        checkpoint_text = str(checkpoint.get("rewritten_document") or "")
        if not checkpoint_text.strip() or checkpoint_text.strip() == original_text.strip():
            return
        checkpoint_scores = checkpoint.get("scores") if isinstance(checkpoint.get("scores"), dict) else {}
        baseline_scores = checkpoint.get("baseline_scores") if isinstance(checkpoint.get("baseline_scores"), dict) else {}
        checkpoint_accepted = checkpoint.get("accepted") if isinstance(checkpoint.get("accepted"), dict) else {}
        checkpoint_author_review_cards = _author_review_cards_from_candidate(
            author_proxy_context,
            checkpoint_accepted,
        )
        checkpoint_author_review_required = _accepted_author_proxy_review_required(
            [checkpoint_accepted],
            context=author_proxy_context,
            final_text=checkpoint_text,
            original_text=original_text,
        )
        checkpoint_goal = checkpoint.get("goal") if isinstance(checkpoint.get("goal"), dict) else {}
        checkpoint_strict_safe = _strict_safe_band_achieved(checkpoint_goal)
        checkpoint_status = (
            AUTHOR_PROXY_REVIEW_STATUS
            if checkpoint_strict_safe and checkpoint_author_review_required
            else PARTIAL_NOT_STRICT_STATUS
        )
        checkpoint_summary = {
            "rewrite_pipeline_version": pipeline_version,
            "rewrite_engine_mode": "v5_residual_cluster_comb_production",
            "outcome": checkpoint_status,
            "public_status": checkpoint_status,
            "partial_rewrite_preserved": True,
            "partial_rewrite_preservation_reason": "accepted_checkpoint_saved_before_pipeline_completion",
            "checkpoint_recovery_available": True,
            "public_candidate_warning": (
                AUTHOR_PROXY_WARNING
                if checkpoint_status == AUTHOR_PROXY_REVIEW_STATUS
                else "candidate_not_strict_safe"
            ),
            "best_candidate_author_review_required": checkpoint_status == AUTHOR_PROXY_REVIEW_STATUS,
            "best_candidate_external_review_required": False,
            "strict_safe_band_achieved": checkpoint_strict_safe,
            "kpi_finalization_status": _kpi_finalization_status(
                strict_safe_band_achieved=checkpoint_strict_safe,
                author_review_required=checkpoint_author_review_required,
                accepted_count=1,
                no_text_change=False,
            ),
            "author_proxy_context": author_proxy_context,
            "author_review_cards": checkpoint_author_review_cards if checkpoint_author_review_required else [],
            "checkpoint": _compact_v5_checkpoint(checkpoint),
            "v5_scores": {
                "original": baseline_scores,
                "final": checkpoint_scores,
                "deltas": _score_deltas(baseline_scores, checkpoint_scores),
            },
            "candidate_generation_status": {
                "accepted_count": int(checkpoint.get("sequence") or 1),
                "reason": "accepted_checkpoint",
            },
            "selected_candidate": checkpoint.get("accepted"),
            "final_text": checkpoint_text,
            "no_text_change": False,
        }
        checkpoint_callback({
            "status": checkpoint_summary["public_status"],
            "checkpoint_schema_version": "rewrite_v5_uploaded_checkpoint.v1",
            "original_text": original_text,
            "final_text": checkpoint_text,
            "summary": checkpoint_summary,
            "checkpoint": checkpoint,
        })

    progress(66, "Running V5 residual cluster comb")
    payload = run_v5_residual_cluster_comb_experiment(
        input_text=original_text,
        output_dir=out_dir,
        max_rounds=int(config["max_rounds"]),
        variant_count=int(config["variant_count"]),
        retune_variant_count=int(config["retune_variant_count"]),
        api_key=api_key,
        model=model,
        base_url=base_url,
        planner_model=str(config.get("planner_model") or "") or None,
        provider=provider_routing,
        extra_body=_v5_extra_body(),
        risky_window_cleanup_rounds=int(config["risky_window_cleanup_rounds"]),
        unsafe_cluster_cleanup_rounds=int(config["unsafe_cluster_cleanup_rounds"]),
        cleanup_variant_count=int(config["cleanup_variant_count"]),
        final_risky_window_cleanup_rounds=int(config["final_risky_window_cleanup_rounds"]),
        direct_scanner_leapfrog_rounds=int(config["direct_scanner_leapfrog_rounds"]),
        direct_scanner_leapfrog_variant_count=int(config["direct_scanner_leapfrog_variants"]),
        direct_scanner_leapfrog_batches=int(config["direct_scanner_leapfrog_batches"]),
        author_proxy_context=author_proxy_context,
        seed_candidate_texts=seed_candidate_texts,
        progress_callback=progress,
        accepted_checkpoint_callback=emit_checkpoint,
        max_seconds=runtime_budget_seconds,
        cancellation_check=raise_if_canceled,
    )

    raise_if_canceled()
    final_text = str(payload.get("rewritten_document") or original_text)
    original_report = detect_json
    final_report = _scan_report(final_text) if final_text.strip() != original_text.strip() else original_report
    raise_if_canceled()
    payload_goal = payload.get("goal") if isinstance(payload.get("goal"), dict) else {}
    final_goal = _final_goal_from_scan(
        original_text=original_text,
        final_text=final_text,
        original_report=original_report,
        final_report=final_report,
        fallback_goal=payload_goal,
    )
    rounds = payload.get("rounds") if isinstance(payload.get("rounds"), list) else []
    direct_scanner_leapfrog_rounds = (
        payload.get("direct_scanner_leapfrog_rounds")
        if isinstance(payload.get("direct_scanner_leapfrog_rounds"), list)
        else []
    )
    risky_window_cleanup_rounds = (
        payload.get("risky_window_cleanup_rounds")
        if isinstance(payload.get("risky_window_cleanup_rounds"), list)
        else []
    )
    unsafe_cluster_cleanup_rounds = (
        payload.get("unsafe_cluster_cleanup_rounds")
        if isinstance(payload.get("unsafe_cluster_cleanup_rounds"), list)
        else []
    )
    final_risky_window_cleanup_rounds = (
        payload.get("final_risky_window_cleanup_rounds")
        if isinstance(payload.get("final_risky_window_cleanup_rounds"), list)
        else []
    )
    borderline_verdict_cleanup_rounds = (
        payload.get("borderline_verdict_cleanup_rounds")
        if isinstance(payload.get("borderline_verdict_cleanup_rounds"), list)
        else []
    )
    final_topk_sentence_route_rounds = (
        payload.get("final_topk_sentence_route_rounds")
        if isinstance(payload.get("final_topk_sentence_route_rounds"), list)
        else []
    )
    safe_band_evidence_repair_rounds = (
        payload.get("safe_band_evidence_repair_rounds")
        if isinstance(payload.get("safe_band_evidence_repair_rounds"), list)
        else []
    )
    all_rounds = (
        direct_scanner_leapfrog_rounds
        + rounds
        + risky_window_cleanup_rounds
        + unsafe_cluster_cleanup_rounds
        + final_risky_window_cleanup_rounds
        + borderline_verdict_cleanup_rounds
        + final_topk_sentence_route_rounds
        + safe_band_evidence_repair_rounds
    )
    accepted = [
        row.get("accepted")
        for row in all_rounds
        if isinstance(row, dict) and isinstance(row.get("accepted"), dict)
    ]
    global_best_fallback = payload.get("global_best_fallback") if isinstance(payload.get("global_best_fallback"), dict) else {}
    if global_best_fallback.get("applied") and isinstance(global_best_fallback.get("selected"), dict):
        accepted.append(global_best_fallback["selected"])
    seed_recovery = payload.get("seed_recovery") if isinstance(payload.get("seed_recovery"), dict) else {}
    if seed_recovery.get("applied") and isinstance(seed_recovery.get("selected"), dict):
        accepted.append(seed_recovery["selected"])
    selected_author_review_cards = _author_review_cards_from_candidate(
        author_proxy_context,
        accepted[-1] if accepted else None,
    )
    no_text_change = final_text.strip() == original_text.strip()
    author_proxy_review_required = _accepted_author_proxy_review_required(
        accepted,
        context=author_proxy_context,
        final_text=final_text,
        original_text=original_text,
    )
    strict_safe_band_achieved = _strict_safe_band_achieved(final_goal)
    if no_text_change:
        public_status = RewriteGoalStatus.ORIGINAL_PRESERVED.value
    elif strict_safe_band_achieved and author_proxy_review_required and accepted:
        public_status = AUTHOR_PROXY_REVIEW_STATUS
    elif final_goal.get("goal_met"):
        public_status = RewriteGoalStatus.AI_MITIGATED.value
    elif accepted:
        public_status = PARTIAL_NOT_STRICT_STATUS
    else:
        public_status = "no_safe_rewrite_applied"
    partial_rewrite_preserved = (
        public_status == PARTIAL_NOT_STRICT_STATUS
        and not no_text_change
        and bool(accepted)
        and not bool(final_goal.get("goal_met"))
    )

    elapsed = time.time() - started
    original_scores = payload.get("baseline_scores") if isinstance(payload.get("baseline_scores"), dict) else {}
    final_scores = payload.get("final_scores") if isinstance(payload.get("final_scores"), dict) else {}
    deltas = _score_deltas(original_scores, final_scores)
    detect_scores = _detect_scores(original_report, final_report, original_scores, final_scores)
    original_scan_compact = _compact_scan_for_rewrite_report(original_report)
    final_scan_compact = _compact_scan_for_rewrite_report(final_report)
    candidate_ledger = _candidate_ledger_from_v5_payload(
        payload=payload,
        accepted=accepted,
        final_text=final_text,
        final_scores=final_scores,
        final_goal=final_goal,
    )
    summary = {
        "rewrite_pipeline_version": pipeline_version,
        "rewrite_engine_mode": "v5_residual_cluster_comb_production",
        "outcome": public_status,
        "public_status": public_status,
        "partial_rewrite_preserved": partial_rewrite_preserved,
        "partial_rewrite_preservation_reason": (
            "safe_progress_kept_despite_strict_goal_miss"
            if partial_rewrite_preserved
            else ""
        ),
        "public_candidate_warning": (
            AUTHOR_PROXY_WARNING
            if public_status == AUTHOR_PROXY_REVIEW_STATUS
            else
            "candidate_not_strict_safe"
            if public_status == PARTIAL_NOT_STRICT_STATUS
            else ""
        ),
        "best_candidate_external_review_required": False,
        "best_candidate_author_review_required": public_status == AUTHOR_PROXY_REVIEW_STATUS,
        "strict_safe_band_achieved": strict_safe_band_achieved,
        "kpi_finalization_status": _kpi_finalization_status(
            strict_safe_band_achieved=strict_safe_band_achieved,
            author_review_required=author_proxy_review_required and bool(accepted),
            accepted_count=len(accepted),
            no_text_change=no_text_change,
        ),
        "author_proxy_context": author_proxy_context,
        "author_review_cards": (
            selected_author_review_cards
            if isinstance(author_proxy_context, dict) and author_proxy_context.get("active")
            else []
        ),
        "rewrite_goal_status": {
            **final_goal,
            "status": public_status if public_status != "no_safe_rewrite_applied" else final_goal.get("status"),
            "goal_met": bool(final_goal.get("goal_met")),
            "reason": final_goal.get("reason") or public_status,
        },
        "strict_goal_status": final_goal.get("status"),
        "reference_ai": original_scores.get("ai"),
        "required_ai_drop": config["required_ai_drop"],
        "target_ai_score": (
            float(original_scores.get("ai")) - float(config["required_ai_drop"])
            if isinstance(original_scores.get("ai"), (int, float))
            else None
        ),
        "rewrite_effective_config": {
            **config,
            "model": model,
            "base_url_configured": bool(base_url),
            "provider_routing": provider_routing,
            "runtime_budget_seconds": runtime_budget_seconds,
        },
        "v5_scores": {
            "original": original_scores,
            "final": final_scores,
            "deltas": deltas,
        },
        "detect_scores": detect_scores,
        "original_risk": detect_scores.get("original_ai"),
        "final_risk": detect_scores.get("rewritten_ai"),
        "converged": bool(final_goal.get("goal_met")),
        "convergence_reason": public_status,
        "candidate_generation_status": {
            "generated_count": _generated_candidate_count(all_rounds),
            "accepted_count": len(accepted),
            "reason": pipeline_version,
        },
        "candidate_ledger": candidate_ledger,
        "candidate_trace": _compact_v5_candidate_trace(accepted),
        "candidate_loop_trace": _compact_v5_rounds(all_rounds),
        "selected_candidate": _compact_v5_candidate_trace([accepted[-1]])[0] if accepted else None,
        "rewrite_layers": {"v5_residual_cluster_comb": _compact_v5_payload(payload)},
        "stage_timings": [{
            "stage": pipeline_version,
            "seconds": round(elapsed, 3),
            "selected": bool(accepted),
            "stop_reason": public_status,
        }],
        "detect_scan_original_saved": original_scan_compact,
        "detect_scan_original": original_scan_compact,
        "detect_scan_rewritten": final_scan_compact,
        "final_text": final_text,
        "no_text_change": no_text_change,
        "no_text_change_reason": "v5_no_safe_candidate" if no_text_change else "",
    }
    sentence_comparison = _sentence_comparison(original_text, final_text)
    result_obj = SimpleNamespace(
        summary=summary,
        sentence_comparison=sentence_comparison,
        rewrite_plan=None,
        mp_result=SimpleNamespace(
            original_text=original_text,
            final_text=final_text,
            converged=bool(final_goal.get("goal_met")),
            convergence_reason=public_status,
            passes=[],
        ),
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"draftproof_rewrite_v5_{ts}.md"
    pdf_path = out_dir / f"draftproof_rewrite_v5_{ts}.pdf"
    json_path = out_dir / f"draftproof_rewrite_v5_{ts}.json"
    md_text = render_rewrite_report(summary=summary, sentence_comparison=sentence_comparison, ai_findings=[], verbose=False)
    md_path.write_text(md_text, encoding="utf-8")
    render_pdf(md_text, str(pdf_path))
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    progress(88, "V5 cluster rewrite complete")
    return {
        "status": public_status,
        "md_path": str(md_path),
        "pdf_path": str(pdf_path),
        "json_path": str(json_path),
        "result": result_obj,
        "elapsed": elapsed,
    }


def _score_deltas(original_scores: dict[str, Any], final_scores: dict[str, Any]) -> dict[str, float]:
    keys = (
        "ai",
        "topk",
        "external",
        "rank",
        "risky_window_count",
        "unsafe_word_ratio",
        "unsafe_cluster_count",
        "topk_calibrated_risk",
        "qualifying_text_ai_density",
        "ai_authorship",
        "external_ai_flag_risk",
    )
    deltas: dict[str, float] = {}
    for key in keys:
        if key in original_scores and key in final_scores:
            try:
                deltas[f"{key}_delta"] = round(float(original_scores.get(key)) - float(final_scores.get(key)), 3)
            except (TypeError, ValueError):
                continue
    return deltas


def _v5_runtime_budget_seconds(
    original_text: str,
    config: dict[str, Any],
    *,
    original_report: dict[str, Any] | None = None,
) -> int:
    words = max(1, word_count(str(original_text or "")))
    estimated = float(config.get("runtime_base_seconds") or 0) + (
        (words / 100.0) * float(config.get("runtime_seconds_per_100_words") or 0.0)
    )
    soft_limit = _int_env("REWRITE_SOFT_TIME_LIMIT_SECONDS", 1800, minimum=180, maximum=7200)
    soft_cap = max(
        int(config.get("runtime_min_seconds") or 60),
        soft_limit - int(config.get("runtime_soft_limit_buffer_seconds") or 120),
    )
    legacy_budget = max(
        int(config.get("runtime_min_seconds") or 60),
        min(
            int(config.get("runtime_max_seconds") or 720),
            soft_cap,
            int(round(estimated)),
        ),
    )
    adaptive_budget = _v5_adaptive_runtime_budget_seconds(
        original_text=original_text,
        original_report=original_report,
    )
    if adaptive_budget is None:
        return legacy_budget
    author_proxy_context = _build_author_proxy_context(original_report or {}, original_text)
    if isinstance(author_proxy_context, dict) and author_proxy_context.get("active"):
        return legacy_budget
    return max(
        60,
        min(
            int(config.get("runtime_max_seconds") or 720),
            soft_cap,
            int(round(adaptive_budget)),
        ),
    )


def _v5_adaptive_runtime_budget_seconds(
    *,
    original_text: str,
    original_report: dict[str, Any] | None,
) -> float | None:
    if not isinstance(original_report, dict):
        return None
    try:
        goal = evaluate_rewrite_goal(
            original_text=original_text,
            candidate_text=original_text,
            original_report=original_report,
            candidate_report=original_report,
        ).to_dict()
        goal = _with_v5_density_gate(original_text, original_report, goal)
        density_gate = _density_gate_for_report(original_text, original_report)
        scores = _score_summary(original_text, original_report, goal)
        return _adaptive_cutoff_runtime_budget_seconds(
            original_text=original_text,
            baseline_density_gate=density_gate,
            baseline_scores=scores,
        )
    except Exception:
        return None


def _generated_candidate_count(rounds: list[Any]) -> int:
    count = 0
    for row in rounds:
        if isinstance(row, dict) and isinstance(row.get("candidates"), list):
            count += len(row.get("candidates") or [])
    return count


def _candidate_ledger_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    goal = row.get("goal") if isinstance(row.get("goal"), dict) else {}
    strict = 0.0 if goal.get("goal_met") is True or goal.get("strict_ai_safe_band_achieved") is True else 1.0
    return (
        strict,
        float(scores.get("topk_calibrated_risk") or 999.0),
        float(scores.get("qualifying_text_ai_density") or 999.0),
        float(scores.get("ai") or 999.0),
        float(scores.get("unsafe_cluster_count") or 999.0),
    )


def _candidate_ledger_text_from_row(row: dict[str, Any]) -> str:
    candidate_text = str(row.get("candidate_text") or "").strip()
    if candidate_text:
        return candidate_text
    apply_status = row.get("apply_status") if isinstance(row.get("apply_status"), dict) else {}
    if row.get("section_id") == "full_document" or apply_status.get("scope") == "full_document":
        return str(row.get("text") or "").strip()
    return ""


def _candidate_ledger_from_v5_payload(
    *,
    payload: dict[str, Any],
    accepted: list[dict[str, Any]],
    final_text: str,
    final_scores: dict[str, Any],
    final_goal: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_row(source: str, row: dict[str, Any] | None, *, fallback_text: str = "") -> None:
        if not isinstance(row, dict):
            return
        text = _candidate_ledger_text_from_row(row) or str(fallback_text or "").strip()
        normalized = " ".join(text.split())
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
        goal = row.get("candidate_goal") if isinstance(row.get("candidate_goal"), dict) else row.get("goal")
        goal = goal if isinstance(goal, dict) else {}
        rows.append({
            "schema_version": "rewrite_candidate_ledger.v1",
            "source": source,
            "section_id": row.get("section_id"),
            "variant_id": row.get("variant_id"),
            "label": row.get("label"),
            "word_count": row.get("word_count") or word_count(text),
            "scores": scores,
            "goal": {
                "status": goal.get("status"),
                "goal_met": goal.get("goal_met"),
                "reason": goal.get("reason"),
                "strict_ai_safe_band_achieved": goal.get("strict_ai_safe_band_achieved"),
            },
            "text": text,
        })

    for row in payload.get("candidate_ledger") if isinstance(payload.get("candidate_ledger"), list) else []:
        add_row(str(row.get("source") or "candidate_ledger"), row)
    for row in payload.get("seed_candidate_rows") if isinstance(payload.get("seed_candidate_rows"), list) else []:
        add_row("historical_seed_candidate", row)
    seed_recovery = payload.get("seed_recovery") if isinstance(payload.get("seed_recovery"), dict) else {}
    add_row("historical_seed_recovery", seed_recovery.get("selected"))
    for row in accepted:
        add_row("accepted_candidate", row)
    global_best_fallback = payload.get("global_best_fallback") if isinstance(payload.get("global_best_fallback"), dict) else {}
    add_row("global_best_fallback", global_best_fallback.get("selected"))
    add_row(
        "final_text",
        {
            "section_id": "full_document",
            "variant_id": "final",
            "label": "final_text",
            "word_count": word_count(final_text),
            "scores": final_scores,
            "goal": final_goal,
            "text": final_text,
        },
    )

    rows.sort(key=_candidate_ledger_sort_key)
    for index, row in enumerate(rows[:5], start=1):
        row["rank"] = index
    return rows[:5]


def _compact_v5_candidate_trace(rows: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
        local = row.get("local_scores") if isinstance(row.get("local_scores"), dict) else {}
        incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
        compact.append({
            "section_id": row.get("section_id"),
            "variant_id": row.get("variant_id"),
            "label": row.get("label"),
            "word_count": row.get("word_count"),
            "ai_delta": scores.get("ai_delta") or incremental.get("ai_delta"),
            "topk_delta": scores.get("topk_delta") or incremental.get("topk_delta"),
            "external_delta": scores.get("external_delta") or incremental.get("external_delta"),
            "rank_delta": scores.get("rank_delta") or incremental.get("rank_delta"),
            "unsafe_cluster_count_delta": scores.get("unsafe_cluster_count_delta") or incremental.get("unsafe_cluster_count_delta"),
            "local_unsafe_cluster_count": local.get("unsafe_cluster_count"),
            "local_topk_delta": local.get("topk_delta"),
            "incremental": incremental or None,
            "text": _truncate_text(row.get("text"), 900),
            "scores_after": scores,
        })
    return compact


def _compact_v5_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    return {
        "schema_version": checkpoint.get("schema_version"),
        "stage": checkpoint.get("stage"),
        "sequence": checkpoint.get("sequence"),
        "phase": checkpoint.get("phase"),
        "round": checkpoint.get("round"),
        "reason": checkpoint.get("reason"),
        "created_at_epoch": checkpoint.get("created_at_epoch"),
        "scores": checkpoint.get("scores") if isinstance(checkpoint.get("scores"), dict) else {},
        "goal": checkpoint.get("goal") if isinstance(checkpoint.get("goal"), dict) else {},
        "accepted": checkpoint.get("accepted") if isinstance(checkpoint.get("accepted"), dict) else {},
        "rewritten_word_count": word_count(str(checkpoint.get("rewritten_document") or "")),
    }


def _compact_v5_generator_diagnostics(diagnostics: Any) -> dict[str, Any] | None:
    if not isinstance(diagnostics, dict):
        return None
    route_plan = diagnostics.get("route_plan") if isinstance(diagnostics.get("route_plan"), dict) else {}
    llm_generation = diagnostics.get("llm_generation") if isinstance(diagnostics.get("llm_generation"), dict) else diagnostics
    compact: dict[str, Any] = {
        "route_plan": {
            "status": route_plan.get("status"),
            "reason": route_plan.get("reason"),
            "content_profile": route_plan.get("content_profile"),
            "cluster_role": route_plan.get("cluster_role"),
            "dominant_failure_pattern": route_plan.get("dominant_failure_pattern"),
            "route_strategy": route_plan.get("route_strategy"),
            "source_block_plan_count": route_plan.get("source_block_plan_count"),
            "target_sentence_job_count": route_plan.get("target_sentence_job_count"),
            "length_target": route_plan.get("length_target"),
            "finish_reason": route_plan.get("finish_reason"),
            "native_finish_reason": route_plan.get("native_finish_reason"),
        } if route_plan else None,
        "llm_generation": {
            "status": llm_generation.get("status"),
            "reason": llm_generation.get("reason"),
            "valid_variant_count": llm_generation.get("valid_variant_count"),
            "variant_count": llm_generation.get("variant_count"),
            "finish_reason": llm_generation.get("finish_reason"),
            "native_finish_reason": llm_generation.get("native_finish_reason"),
            "structured_output_mode": llm_generation.get("structured_output_mode"),
        } if isinstance(llm_generation, dict) else None,
    }
    return compact


def _compact_v5_rounds(rounds: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rounds:
        if not isinstance(row, dict):
            continue
        compact.append({
            "round": row.get("round"),
            "phase": row.get("phase"),
            "status": row.get("status"),
            "reason": row.get("reason"),
            "current_scores": row.get("current_scores") if isinstance(row.get("current_scores"), dict) else None,
            "runtime_budget": row.get("runtime_budget") if isinstance(row.get("runtime_budget"), dict) else None,
            "density_gate": _compact_density_gate(row.get("density_gate"))
            if isinstance(row.get("density_gate"), dict)
            else None,
            "generator_diagnostics": _compact_v5_generator_diagnostics(row.get("generator_diagnostics")),
            "accepted": _compact_v5_candidate_trace([row.get("accepted")])[0]
            if isinstance(row.get("accepted"), dict)
            else None,
            "selected": _compact_v5_candidate_trace([row.get("selected")])[0]
            if isinstance(row.get("selected"), dict)
            else None,
            "candidate_count": len(row.get("candidates") or []) if isinstance(row.get("candidates"), list) else 0,
        })
    return compact


def _compact_v5_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rounds = payload.get("rounds") if isinstance(payload.get("rounds"), list) else []
    direct_scanner_leapfrog_rounds = (
        payload.get("direct_scanner_leapfrog_rounds")
        if isinstance(payload.get("direct_scanner_leapfrog_rounds"), list)
        else []
    )
    risky_window_cleanup_rounds = (
        payload.get("risky_window_cleanup_rounds")
        if isinstance(payload.get("risky_window_cleanup_rounds"), list)
        else []
    )
    unsafe_cluster_cleanup_rounds = (
        payload.get("unsafe_cluster_cleanup_rounds")
        if isinstance(payload.get("unsafe_cluster_cleanup_rounds"), list)
        else []
    )
    final_risky_window_cleanup_rounds = (
        payload.get("final_risky_window_cleanup_rounds")
        if isinstance(payload.get("final_risky_window_cleanup_rounds"), list)
        else []
    )
    borderline_verdict_cleanup_rounds = (
        payload.get("borderline_verdict_cleanup_rounds")
        if isinstance(payload.get("borderline_verdict_cleanup_rounds"), list)
        else []
    )
    final_topk_sentence_route_rounds = (
        payload.get("final_topk_sentence_route_rounds")
        if isinstance(payload.get("final_topk_sentence_route_rounds"), list)
        else []
    )
    safe_band_evidence_repair_rounds = (
        payload.get("safe_band_evidence_repair_rounds")
        if isinstance(payload.get("safe_band_evidence_repair_rounds"), list)
        else []
    )
    all_rounds = (
        direct_scanner_leapfrog_rounds
        + rounds
        + risky_window_cleanup_rounds
        + unsafe_cluster_cleanup_rounds
        + final_risky_window_cleanup_rounds
        + borderline_verdict_cleanup_rounds
        + final_topk_sentence_route_rounds
        + safe_band_evidence_repair_rounds
    )
    return {
        "stage": payload.get("stage"),
        "baseline_scores": payload.get("baseline_scores"),
        "final_scores": payload.get("final_scores"),
        "eligible_span_density_gate": payload.get("eligible_span_density_gate"),
        "candidate_ledger": payload.get("candidate_ledger") if isinstance(payload.get("candidate_ledger"), list) else [],
        "runtime_budget": payload.get("runtime_budget") if isinstance(payload.get("runtime_budget"), dict) else None,
        "phase_order": payload.get("phase_order") if isinstance(payload.get("phase_order"), dict) else None,
        "accepted_checkpoints": payload.get("accepted_checkpoints") if isinstance(payload.get("accepted_checkpoints"), list) else [],
        "goal": payload.get("goal"),
        "accepted_rounds": sum(1 for row in all_rounds if isinstance(row, dict) and row.get("accepted")),
        "direct_scanner_leapfrog_rounds": _compact_v5_rounds(direct_scanner_leapfrog_rounds),
        "rounds": _compact_v5_rounds(rounds),
        "risky_window_cleanup_rounds": _compact_v5_rounds(risky_window_cleanup_rounds),
        "unsafe_cluster_cleanup_rounds": _compact_v5_rounds(unsafe_cluster_cleanup_rounds),
        "final_risky_window_cleanup_rounds": _compact_v5_rounds(final_risky_window_cleanup_rounds),
        "borderline_verdict_cleanup_rounds": _compact_v5_rounds(borderline_verdict_cleanup_rounds),
        "final_topk_sentence_route_rounds": _compact_v5_rounds(final_topk_sentence_route_rounds),
        "safe_band_evidence_repair_rounds": _compact_v5_rounds(safe_band_evidence_repair_rounds),
        "seed_candidate_rows": _compact_v5_rounds(payload.get("seed_candidate_rows") if isinstance(payload.get("seed_candidate_rows"), list) else []),
        "seed_recovery": payload.get("seed_recovery") if isinstance(payload.get("seed_recovery"), dict) else None,
        "global_best_fallback": payload.get("global_best_fallback"),
    }
