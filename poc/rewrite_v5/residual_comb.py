"""Residual cluster comb-through experiment for V5.

This path turns the single-cluster proof into an iterative experiment:
scan the current document, rebuild the strongest remaining cluster, score the
cluster and full document, accept useful movement, then repeat from the new
scan. It is intentionally isolated from production.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from llm.gateway import LLMConfig, LLMGateway
from rewrite_v2.goal_contract import evaluate_rewrite_goal
from rewrite_v2.structured_output import structured_json_request_options
from rewrite_v3.document_units import word_count
from rewrite_v3.pipeline import _scan_report
from rewrite_v3.text_integrity import minimal_replacement_text_integrity
from rewrite_v4.cluster_patch import build_cluster_repair_units
from rewrite_v4.validation import parse_json_object

from .experiment import (
    _add_deltas,
    _merge_provider_options,
    _score_summary,
    _section_from_cluster,
    _sentences,
    _variants_response_format,
    apply_section_variant,
)
from .models import RecompositionVariant, SectionUnit


def run_v5_residual_cluster_comb_experiment(
    *,
    input_text: str,
    output_dir: str | Path,
    max_rounds: int = 5,
    variant_count: int = 3,
    retune_variant_count: int = 4,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Iteratively treat the strongest residual cluster and rescan."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_text = str(input_text or "")
    baseline_report = _scan_report(original_text)
    baseline_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=original_text,
        original_report=baseline_report,
        candidate_report=baseline_report,
    ).to_dict()
    baseline_scores = _score_summary(original_text, baseline_report, baseline_goal)
    current_text = original_text
    current_report = baseline_report
    current_goal = baseline_goal
    current_scores = baseline_scores
    gateway = LLMGateway(LLMConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_MAX_TOKENS", 8000, minimum=1000, maximum=12000),
        temperature=_float_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_TEMPERATURE", 0.35, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_TOP_P", 0.9, minimum=0.1, maximum=1.0),
        timeout=180,
        extra_body=extra_body,
    ))

    rounds: list[dict[str, Any]] = []
    for round_index in range(1, max(1, int(max_rounds or 1)) + 1):
        round_dir = out_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        cluster_units = build_cluster_repair_units(
            text=current_text,
            report=current_report,
            goal=current_goal,
            limit=1,
            context_chars=220,
        )
        if not cluster_units:
            rounds.append({"round": round_index, "status": "stopped", "reason": "no_residual_cluster"})
            break
        section = _section_from_cluster(cluster_units[0])
        local_source_goal = _local_goal(section.text, section.text)
        seed_variants = generate_residual_cluster_seed_variants(section=section, local_goal=local_source_goal)
        seed_rows = [
            _score_residual_variant(
                original_text=original_text,
                baseline_report=baseline_report,
                baseline_scores=baseline_scores,
                current_text=current_text,
                current_scores=current_scores,
                section=section,
                variant=variant,
                output_dir=round_dir,
                label=f"seed_{variant.variant_id}",
            )
            for variant in seed_variants
        ]
        best_seed = _best_residual_candidate(seed_rows)
        seed_accepted = best_seed if best_seed and _has_incremental_movement(best_seed) else None
        diagnostics: dict[str, Any] = {
            "seed_variant_count": len(seed_variants),
            "seed_short_circuited": bool(seed_accepted),
        }
        retune_diagnostics: dict[str, Any] | None = None
        rows: list[dict[str, Any]] = list(seed_rows)
        retuned_rows: list[dict[str, Any]] = []
        route_plan: dict[str, Any] | None = None
        route_plan_diagnostics: dict[str, Any] | None = None
        if not seed_accepted:
            route_plan, route_plan_diagnostics, plan_prompt, plan_completion = generate_residual_cluster_route_plan(
                section=section,
                local_goal=local_source_goal,
                gateway=gateway,
            )
            (round_dir / "route_plan_prompt.json.txt").write_text(plan_prompt)
            (round_dir / "route_plan_completion.json.txt").write_text(plan_completion)
            variants, llm_diagnostics, prompt, completion = generate_residual_cluster_variants(
                section=section,
                local_goal=local_source_goal,
                gateway=gateway,
                variant_count=variant_count,
                route_plan=route_plan,
            )
            diagnostics = {
                **diagnostics,
                "route_plan": route_plan_diagnostics,
                "llm_generation": llm_diagnostics,
            }
            (round_dir / "cluster_prompt.json.txt").write_text(prompt)
            (round_dir / "cluster_completion.json.txt").write_text(completion)
            rows.extend([
                _score_residual_variant(
                    original_text=original_text,
                    baseline_report=baseline_report,
                    baseline_scores=baseline_scores,
                    current_text=current_text,
                    current_scores=current_scores,
                    section=section,
                    variant=variant,
                    output_dir=round_dir,
                    label=f"initial_{variant.variant_id}",
                )
                for variant in variants
            ])
        best_initial = _best_residual_candidate(rows)
        if not seed_accepted and best_initial and _needs_retune(best_initial):
            retuned, retune_diagnostics, retune_prompt, retune_completion = generate_residual_cluster_retunes(
                section=section,
                current_best_text=str(best_initial.get("text") or ""),
                local_goal=best_initial.get("local_goal") or {},
                gateway=gateway,
                variant_count=retune_variant_count,
                route_plan=route_plan,
            )
            (round_dir / "retune_prompt.json.txt").write_text(retune_prompt)
            (round_dir / "retune_completion.json.txt").write_text(retune_completion)
            retuned_rows = [
                _score_residual_variant(
                    original_text=original_text,
                    baseline_report=baseline_report,
                    baseline_scores=baseline_scores,
                    current_text=current_text,
                    current_scores=current_scores,
                    section=section,
                    variant=variant,
                    output_dir=round_dir,
                    label=f"retune_{variant.variant_id}",
                )
                for variant in retuned
            ]
        all_rows = rows + retuned_rows
        best = _best_residual_candidate(all_rows)
        accepted = best if best and _has_incremental_movement(best) else None
        round_payload = {
            "round": round_index,
            "status": "accepted" if accepted else "stopped",
            "reason": "accepted_incremental_movement" if accepted else "no_incremental_movement",
            "section": section.to_dict(),
            "generator_diagnostics": diagnostics,
            "retune_diagnostics": retune_diagnostics,
            "current_scores": current_scores,
            "candidates": [_compact_residual_row(row) for row in all_rows],
            "selected": _compact_residual_row(best),
            "accepted": _compact_residual_row(accepted),
        }
        rounds.append(round_payload)
        (round_dir / "round_result.json").write_text(json.dumps(round_payload, ensure_ascii=False, indent=2))
        if not accepted:
            break
        current_text = str(accepted.get("candidate_text") or current_text)
        current_report = accepted.get("candidate_report") if isinstance(accepted.get("candidate_report"), dict) else _scan_report(current_text)
        current_goal = accepted.get("candidate_goal") if isinstance(accepted.get("candidate_goal"), dict) else evaluate_rewrite_goal(
            original_text=original_text,
            candidate_text=current_text,
            original_report=baseline_report,
            candidate_report=current_report,
        ).to_dict()
        current_scores = accepted.get("scores") if isinstance(accepted.get("scores"), dict) else _score_summary(original_text, current_report, current_goal)
        (out_dir / f"after_round_{round_index:02d}.txt").write_text(current_text)

    payload = {
        "stage": "v5_residual_cluster_comb",
        "baseline_scores": baseline_scores,
        "rounds": rounds,
        "final_scores": current_scores,
        "goal": {
            "status": current_goal.get("status"),
            "goal_met": current_goal.get("goal_met"),
            "reason": current_goal.get("reason"),
        },
        "rewritten_document": current_text,
    }
    (out_dir / "v5_residual_comb_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (out_dir / "v5_residual_comb_rewritten_document.txt").write_text(current_text)
    return payload


def build_residual_cluster_prompt(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any] | None = None,
    variant_count: int = 3,
    route_plan: dict[str, Any] | None = None,
) -> str:
    variants = max(1, min(5, int(variant_count or 1)))
    plan = route_plan if _route_plan_valid(route_plan) else None
    payload = {
        "task": "residual_cluster_route_bump",
        "cluster": {
            "section_id": section.section_id,
            "source_text": section.text,
            "before_context": _section_before_context(section),
            "after_context": _section_after_context(section),
            "source_word_count": section.word_count,
            "source_event_beats": _source_event_beats(section.text),
            "source_phrase_anchors": _source_phrase_anchors(section.text),
            "referential_continuity": _referential_continuity(
                section.text,
                before_context=_section_before_context(section),
            ),
        },
        "custom_route_plan": plan,
        "fallback_route_blueprint": build_route_blueprint(section=section, local_goal=local_goal or {}),
        "length_guidance": {
            "source_words": section.word_count,
            "preferred_min_words": max(section.word_count + 10, round(section.word_count * 1.12)),
            "preferred_max_words": round(section.word_count * 1.35),
            "purpose": "use enough words to rebuild the route; do not compress the cluster",
        },
        "remaining_problem_sentences": _local_unsafe_previews(local_goal or {}),
        "method": [
            "Rewrite this cluster as a concrete route window, not as phrase repair.",
            "If custom_route_plan is present, follow custom_route_plan.better_route in order.",
            "If custom_route_plan is absent, follow fallback_route_blueprint.steps in order.",
            "Use custom_route_plan.sentence_jobs when present; otherwise use fallback_route_blueprint.sentence_jobs.",
            "Each sentence job should carry a full source beat or bridge, not a compressed summary.",
            "Follow length_guidance as writing guidance; fuller source-grounded wording is preferred over compression.",
            "Do not copy custom_route_plan labels, fallback_route_blueprint labels, or step names into the replacement.",
            "Change the route of remaining_problem_sentences most strongly.",
            "Do not keep an avoid_openers item as the opening sentence.",
            "Add substance by unpacking the existing event, activity, and outcome named in the custom plan.",
            "Preserve each source-supported beat unless two adjacent beats are naturally merged.",
            "Stay source-near: keep simple source words when they already work.",
            "Use cluster.source_phrase_anchors where they fit naturally.",
            "Avoid exact copying of custom_route_plan.phrases_to_repath where alternatives are provided.",
            "Avoid custom_route_plan.plain_style_bans when present.",
            "Do not upgrade simple source wording into formal education-theory labels.",
            "Keep the same people, event, activity, outcome, and point of view.",
            "Preserve cluster.referential_continuity; do not replace a specific source subject with a generic category label.",
            "If cluster.referential_continuity gives an established name, use the name or the source pronoun naturally; do not write explanatory referent phrases like 'referring to'.",
            "If the source uses I or my, keep that teacher viewpoint instead of replacing it with a detached narrator.",
            "Do not invent new names, dates, places, dialogue, weather, objects, clients, classmates, or side events.",
            "Do not replace the source scenario with a different scene.",
            "Do not return a fragment; the replacement must cover the whole source cluster.",
            "Do not write a plan, label, explanation of the method, or bullet list.",
            "Avoid abstract summary language. Make the movement happen through the event.",
        ],
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, variants + 1)
            ]
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_residual_cluster_retune_prompt(
    *,
    section: SectionUnit,
    current_best_text: str,
    local_goal: dict[str, Any],
    variant_count: int = 4,
    route_plan: dict[str, Any] | None = None,
) -> str:
    variants = max(1, min(5, int(variant_count or 1)))
    focus = _local_unsafe_previews(local_goal)
    plan = route_plan if _route_plan_valid(route_plan) else None
    payload = {
        "task": "residual_cluster_retune",
        "cluster": {
            "section_id": section.section_id,
            "original_source_text": section.text,
            "before_context": _section_before_context(section),
            "after_context": _section_after_context(section),
            "source_event_beats": _source_event_beats(section.text),
            "source_phrase_anchors": _source_phrase_anchors(section.text),
            "referential_continuity": _referential_continuity(
                section.text,
                before_context=_section_before_context(section),
            ),
            "current_best_text": current_best_text,
        },
        "custom_route_plan": plan,
        "fallback_route_blueprint": build_route_blueprint(section=section, local_goal=local_goal or {}),
        "length_guidance": {
            "source_words": section.word_count,
            "preferred_min_words": max(section.word_count + 10, round(section.word_count * 1.12)),
            "preferred_max_words": round(section.word_count * 1.35),
            "purpose": "use enough words to rebuild the route; do not compress the cluster",
        },
        "remaining_problem_sentences": focus,
        "retune_focus": _retune_focus_from_goal(local_goal or {}),
        "candidate_non_source_terms_to_reduce": _non_source_terms(section.text, current_best_text),
        "method": [
            "Rewrite the whole cluster again, but focus on the remaining problem sentence route.",
            "If custom_route_plan is present, follow custom_route_plan.better_route in order.",
            "If custom_route_plan is absent, follow fallback_route_blueprint.steps in order.",
            "Use custom_route_plan.sentence_jobs when present; otherwise use fallback_route_blueprint.sentence_jobs.",
            "Each sentence job should carry a full source beat or bridge, not a compressed summary.",
            "Follow length_guidance as writing guidance; fuller source-grounded wording is preferred over compression.",
            "Do not copy custom_route_plan labels, fallback_route_blueprint labels, or step names into the replacement.",
            "Break any packed sentence into clearer event movement if needed.",
            "Preserve each source_event_beats item unless two adjacent beats are naturally merged.",
            "Stay source-near: keep simple source words when they already work.",
            "Use cluster.source_phrase_anchors where they fit naturally.",
            "Avoid exact copying of custom_route_plan.phrases_to_repath where alternatives are provided.",
            "Avoid custom_route_plan.plain_style_bans when present.",
            "Reduce candidate_non_source_terms_to_reduce by replacing them with source wording where possible.",
            "Do not upgrade simple source wording into formal education-theory labels.",
            "If the source uses I or my, keep that teacher viewpoint instead of replacing it with a detached narrator.",
            "Preserve cluster.referential_continuity; do not replace a specific source subject with a generic category label.",
            "If cluster.referential_continuity gives an established name, use the name or the source pronoun naturally; do not write explanatory referent phrases like 'referring to'.",
            "Use concrete classroom, action, or event wording from the same source scenario instead of summary wording.",
            "Do not invent new names, dates, places, dialogue, weather, objects, clients, classmates, or side events.",
            "Do not replace the source scenario with a different scene.",
            "Do not return a fragment; the replacement must cover the whole source cluster.",
            "Do not write a plan, label, explanation of the method, or bullet list.",
        ],
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, variants + 1)
            ]
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_residual_cluster_variants(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any] | None = None,
    gateway: LLMGateway,
    variant_count: int = 3,
    route_plan: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    prompt = build_residual_cluster_prompt(
        section=section,
        local_goal=local_goal or {},
        variant_count=variant_count,
        route_plan=route_plan,
    )
    return _generate_loose_variants(prompt=prompt, gateway=gateway, variant_count=variant_count)


def build_residual_cluster_route_plan_prompt(*, section: SectionUnit, local_goal: dict[str, Any] | None = None) -> str:
    payload = {
        "task": "custom_cluster_route_plan",
        "cluster": {
            "section_id": section.section_id,
            "source_text": section.text,
            "before_context": _section_before_context(section),
            "after_context": _section_after_context(section),
            "source_word_count": section.word_count,
            "source_event_beats": _source_event_beats(section.text),
            "source_phrase_anchors": _source_phrase_anchors(section.text),
            "referential_continuity": _referential_continuity(
                section.text,
                before_context=_section_before_context(section),
            ),
        },
        "scanner_local_findings": {
            "unsafe_previews": _local_unsafe_previews(local_goal or {}),
            "top_sentence_targets": _local_top_sentence_targets(local_goal or {}),
            "recommended_actions": _local_recommended_actions(local_goal or {}),
        },
        "planning_rules": [
            "Act as a prompt planner, not the final writer.",
            "Derive a cluster-specific route plan from the source text and scanner findings.",
            "Do not mention scores, scanner names, authorship labels, or risk labels in the plan fields.",
            "Every better_route step must be supported by source text.",
            "Do not add facts, examples, people, places, dates, dialogue, or outside knowledge.",
            "Convert broad summary movement into concrete sentence jobs using source events.",
            "Preserve cluster.referential_continuity in the better route.",
            "If a pronoun is linked to a name in before_context, plan for natural name/pronoun continuity and do not tell the writer to explain the reference parenthetically.",
            "Use plain editorial task language that a writer can execute.",
        ],
        "output_schema": {
            "route_plan": {
                "current_route": [
                    {"source_quote": "...", "function": "...", "weakness": "..."},
                ],
                "better_route": [
                    {"job_id": "j1", "job": "...", "source_quotes": ["..."], "avoid_copying": ["..."]},
                ],
                "sentence_jobs": ["..."],
                "phrases_to_repath": [
                    {"source": "...", "plain_direction": "..."},
                ],
                "plain_style_bans": ["..."],
                "opening_strategy": "...",
                "length_strategy": "...",
            }
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_residual_cluster_route_plan(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any],
    gateway: LLMGateway,
) -> tuple[dict[str, Any] | None, dict[str, Any], str, str]:
    prompt = build_residual_cluster_route_plan_prompt(section=section, local_goal=local_goal)
    structured = structured_json_request_options(getattr(gateway, "model", None), _route_plan_response_format())
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured.get("provider"))
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a route_plan object.",
        response_format=structured.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_REWRITE_V5_ROUTE_PLAN_TEMPERATURE", 0.12, minimum=0.0, maximum=0.8),
        top_p=_float_env("DRAFTPROOF_REWRITE_V5_ROUTE_PLAN_TOP_P", 0.72, minimum=0.1, maximum=1.0),
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V5_ROUTE_PLAN_MAX_TOKENS", 2600, minimum=800, maximum=6000),
    )
    raw = response.raw_content or response.content
    parsed, diagnostics = _parse_route_plan(raw, source_text=section.text)
    return parsed, {
        **diagnostics,
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "structured_output_mode": structured.get("structured_output_mode"),
    }, prompt, raw


def generate_residual_cluster_seed_variants(
    *,
    section: SectionUnit,
    local_goal: dict[str, Any] | None = None,
) -> list[RecompositionVariant]:
    """Generate scanner-scored route seeds before asking the model.

    These are not accepted directly. They enter the same local and full-document
    scoring path as LLM candidates, so a bad seed is only a measured miss.
    """

    del local_goal
    texts = _student_support_role_progression_seed_texts(section.text)
    variants: list[RecompositionVariant] = []
    seen: set[str] = set()
    for index, text in enumerate(texts, start=1):
        normalized = " ".join(str(text or "").split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        variants.append(RecompositionVariant(
            variant_id=f"route_seed_{index}",
            text=normalized,
            word_count=word_count(normalized),
        ))
    return variants


def generate_residual_cluster_retunes(
    *,
    section: SectionUnit,
    current_best_text: str,
    local_goal: dict[str, Any],
    gateway: LLMGateway,
    variant_count: int = 4,
    route_plan: dict[str, Any] | None = None,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    prompt = build_residual_cluster_retune_prompt(
        section=section,
        current_best_text=current_best_text,
        local_goal=local_goal,
        variant_count=variant_count,
        route_plan=route_plan,
    )
    return _generate_loose_variants(prompt=prompt, gateway=gateway, variant_count=variant_count)


def _generate_loose_variants(
    *,
    prompt: str,
    gateway: LLMGateway,
    variant_count: int,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    variants = max(1, min(5, int(variant_count or 1)))
    structured = structured_json_request_options(getattr(gateway, "model", None), _variants_response_format(variants))
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured.get("provider"))
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a variants array.",
        response_format=structured.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_TEMPERATURE", 0.35, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_TOP_P", 0.9, minimum=0.1, maximum=1.0),
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V5_RESIDUAL_COMB_MAX_TOKENS", 8000, minimum=1000, maximum=12000),
    )
    raw = response.raw_content or response.content
    parsed, diagnostics = _parse_loose_variants(raw)
    return parsed, {
        **diagnostics,
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "structured_output_mode": structured.get("structured_output_mode"),
    }, prompt, raw


def _parse_route_plan(raw: str, *, source_text: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    payload, diagnostics = parse_json_object(raw, required_keys={"route_plan"})
    if payload is None:
        return None, diagnostics
    plan = payload.get("route_plan")
    if not isinstance(plan, dict):
        return None, {**diagnostics, "status": "schema_failed", "reason": "route_plan_not_object"}
    sanitized = _sanitize_route_plan(plan, source_text=source_text)
    if not _route_plan_valid(sanitized):
        return None, {
            **diagnostics,
            "status": "schema_failed",
            "reason": "route_plan_has_no_supported_better_route",
            "route_plan_keys": sorted(plan.keys()),
        }
    return sanitized, {
        **diagnostics,
        "status": "ok",
        "better_route_count": len(sanitized.get("better_route") or []),
        "phrase_repath_count": len(sanitized.get("phrases_to_repath") or []),
        "sentence_job_count": len(sanitized.get("sentence_jobs") or []),
    }


def _sanitize_route_plan(plan: dict[str, Any], *, source_text: str) -> dict[str, Any]:
    source = str(source_text or "")
    return {
        "current_route": _sanitize_current_route(plan.get("current_route"), source_text=source),
        "better_route": _sanitize_better_route(plan.get("better_route"), source_text=source),
        "sentence_jobs": _string_list(plan.get("sentence_jobs"), limit=8),
        "phrases_to_repath": _sanitize_phrase_repaths(plan.get("phrases_to_repath"), source_text=source),
        "plain_style_bans": _string_list(plan.get("plain_style_bans"), limit=12),
        "opening_strategy": _short_string(plan.get("opening_strategy"), limit=220),
        "length_strategy": _short_string(plan.get("length_strategy"), limit=220),
    }


def _sanitize_current_route(value: Any, *, source_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in value if isinstance(value, list) else []:
        if not isinstance(row, dict):
            continue
        quote = _supported_quote(row.get("source_quote"), source_text)
        if not quote:
            continue
        rows.append({
            "source_quote": quote,
            "function": _short_string(row.get("function"), limit=160),
            "weakness": _short_string(row.get("weakness"), limit=180),
        })
        if len(rows) >= 8:
            break
    return rows


def _sanitize_better_route(value: Any, *, source_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value if isinstance(value, list) else [], start=1):
        if not isinstance(row, dict):
            continue
        quotes = [
            quote
            for quote in (_supported_quote(item, source_text) for item in _raw_list(row.get("source_quotes")))
            if quote
        ]
        if not quotes:
            continue
        job = _short_string(row.get("job"), limit=260)
        if not job:
            continue
        rows.append({
            "job_id": _short_string(row.get("job_id"), limit=24) or f"j{index}",
            "job": job,
            "source_quotes": quotes[:3],
            "avoid_copying": _supported_or_short_list(row.get("avoid_copying"), source_text=source_text, limit=4),
        })
        if len(rows) >= 8:
            break
    return rows


def _sanitize_phrase_repaths(value: Any, *, source_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in value if isinstance(value, list) else []:
        if not isinstance(row, dict):
            continue
        source = _supported_quote(row.get("source"), source_text)
        direction = _short_string(row.get("plain_direction"), limit=220)
        if not source or not direction:
            continue
        rows.append({"source": source, "plain_direction": direction})
        if len(rows) >= 10:
            break
    return rows


def _route_plan_valid(plan: Any) -> bool:
    return isinstance(plan, dict) and bool(plan.get("better_route"))


def _route_plan_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rewrite_v5_cluster_route_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "route_plan": {
                        "type": "object",
                        "properties": {
                            "current_route": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 8,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_quote": {"type": "string"},
                                        "function": {"type": "string"},
                                        "weakness": {"type": "string"},
                                    },
                                    "required": ["source_quote", "function", "weakness"],
                                    "additionalProperties": False,
                                },
                            },
                            "better_route": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 8,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "job_id": {"type": "string"},
                                        "job": {"type": "string"},
                                        "source_quotes": {
                                            "type": "array",
                                            "minItems": 1,
                                            "maxItems": 3,
                                            "items": {"type": "string"},
                                        },
                                        "avoid_copying": {
                                            "type": "array",
                                            "minItems": 0,
                                            "maxItems": 4,
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["job_id", "job", "source_quotes", "avoid_copying"],
                                    "additionalProperties": False,
                                },
                            },
                            "sentence_jobs": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 8,
                                "items": {"type": "string"},
                            },
                            "phrases_to_repath": {
                                "type": "array",
                                "minItems": 0,
                                "maxItems": 10,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": "string"},
                                        "plain_direction": {"type": "string"},
                                    },
                                    "required": ["source", "plain_direction"],
                                    "additionalProperties": False,
                                },
                            },
                            "plain_style_bans": {
                                "type": "array",
                                "minItems": 0,
                                "maxItems": 12,
                                "items": {"type": "string"},
                            },
                            "opening_strategy": {"type": "string"},
                            "length_strategy": {"type": "string"},
                        },
                        "required": [
                            "current_route",
                            "better_route",
                            "sentence_jobs",
                            "phrases_to_repath",
                            "plain_style_bans",
                            "opening_strategy",
                            "length_strategy",
                        ],
                        "additionalProperties": False,
                    }
                },
                "required": ["route_plan"],
                "additionalProperties": False,
            },
        },
    }


def _parse_loose_variants(raw: str) -> tuple[list[RecompositionVariant], dict[str, Any]]:
    payload, diagnostics = parse_json_object(raw, required_keys={"variants"})
    if payload is None:
        return [], diagnostics
    rows = payload.get("variants")
    if not isinstance(rows, list):
        return [], {**diagnostics, "status": "schema_failed", "reason": "variants_not_array"}
    variants: list[RecompositionVariant] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            rejected.append({"index": index, "reason": "variant_not_object"})
            continue
        if set(row.keys()) != {"variant_id", "text"}:
            rejected.append({"index": index, "reason": "variant_keys_mismatch", "keys": sorted(row.keys())})
            continue
        variant_id = str(row.get("variant_id") or "").strip()
        text = str(row.get("text") or "").strip()
        if not variant_id or not text:
            rejected.append({"index": index, "reason": "empty_variant"})
            continue
        integrity = minimal_replacement_text_integrity(text)
        if not integrity.get("passed"):
            rejected.append({"index": index, "variant_id": variant_id, "reason": "text_integrity_failed", "text_integrity": integrity})
            continue
        variants.append(RecompositionVariant(variant_id=variant_id, text=text, word_count=word_count(text)))
    return variants, {**diagnostics, "status": "ok" if variants else "schema_failed", "variant_count": len(variants), "rejected": rejected}


def _score_residual_variant(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_scores: dict[str, Any],
    current_text: str,
    current_scores: dict[str, Any],
    section: SectionUnit,
    variant: RecompositionVariant,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    candidate_text, apply_status = apply_section_variant(current_text, section, variant.text)
    if not apply_status.get("applied"):
        return {
            "section_id": section.section_id,
            "variant_id": variant.variant_id,
            "label": label,
            "text": variant.text,
            "word_count": variant.word_count,
            "apply_status": apply_status,
            "scores": current_scores,
            "incremental": {},
            "local_scores": {},
            "local_goal": {},
        }
    candidate_report = _scan_report(candidate_text)
    candidate_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=baseline_report,
        candidate_report=candidate_report,
    ).to_dict()
    scores = _score_summary(original_text, candidate_report, candidate_goal)
    _add_deltas(scores, baseline_scores)
    local_before_report = _scan_report(section.text)
    local_before_goal = evaluate_rewrite_goal(
        original_text=section.text,
        candidate_text=section.text,
        original_report=local_before_report,
        candidate_report=local_before_report,
    ).to_dict()
    local_before_scores = _score_summary(section.text, local_before_report, local_before_goal)
    local_after_report = _scan_report(variant.text)
    local_after_goal = evaluate_rewrite_goal(
        original_text=section.text,
        candidate_text=variant.text,
        original_report=local_before_report,
        candidate_report=local_after_report,
    ).to_dict()
    local_after_scores = _score_summary(section.text, local_after_report, local_after_goal)
    _add_deltas(local_after_scores, local_before_scores)
    safe_name = label.replace("/", "_")
    (output_dir / f"{safe_name}.txt").write_text(candidate_text)
    (output_dir / f"{safe_name}_cluster.txt").write_text(variant.text)
    (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
    return {
        "section_id": section.section_id,
        "variant_id": variant.variant_id,
        "label": label,
        "text": variant.text,
        "word_count": variant.word_count,
        "apply_status": apply_status,
        "scores": scores,
        "incremental": _incremental_deltas(scores, current_scores),
        "local_scores": local_after_scores,
        "local_goal": local_after_goal,
        "candidate_text": candidate_text,
        "candidate_report": candidate_report,
        "candidate_goal": candidate_goal,
    }


def _best_residual_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if (row.get("apply_status") or {}).get("applied")]
    if not eligible:
        return None
    return max(eligible, key=_residual_candidate_sort_key)


def _residual_candidate_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    local = row.get("local_scores") if isinstance(row.get("local_scores"), dict) else {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    return (
        1.0 if _local_cluster_cleared(local) else 0.0,
        _number(local.get("unsafe_cluster_count_delta")),
        _number(local.get("topk_calibrated_risk_delta")),
        _number(local.get("unsafe_word_ratio_delta")),
        _number(incremental.get("unsafe_cluster_count_delta")),
        _number(incremental.get("rank_delta")),
        _number(incremental.get("ai_delta")),
        _number(scores.get("rank_delta")),
    )


def _needs_retune(row: dict[str, Any]) -> bool:
    local = row.get("local_scores") if isinstance(row.get("local_scores"), dict) else {}
    if not _local_cluster_cleared(local):
        return True
    return _number(local.get("topk_calibrated_risk")) > 25.0


def _has_incremental_movement(row: dict[str, Any]) -> bool:
    incremental = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
    local = row.get("local_scores") if isinstance(row.get("local_scores"), dict) else {}
    if not _local_cluster_cleared(local) and not _local_cluster_directionally_improved(local):
        return False
    return any(
        _number(incremental.get(key)) > 0
        for key in ("unsafe_cluster_count_delta", "rank_delta", "ai_delta", "topk_delta", "external_delta")
    )


def _local_cluster_cleared(local_scores: dict[str, Any]) -> bool:
    return (
        _number(local_scores.get("unsafe_cluster_count")) <= 0
        and _number(local_scores.get("unsafe_word_ratio")) <= 0
    )


def _local_cluster_directionally_improved(local_scores: dict[str, Any]) -> bool:
    return (
        _number(local_scores.get("unsafe_cluster_count_delta")) >= 0
        and _number(local_scores.get("unsafe_word_ratio_delta")) > 0
        and _number(local_scores.get("topk_delta")) > 0
        and _number(local_scores.get("rank_delta")) > 0
    )


def _incremental_deltas(scores: dict[str, Any], current_scores: dict[str, Any]) -> dict[str, Any]:
    lower_is_better = (
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
    result: dict[str, Any] = {}
    for key in lower_is_better:
        if key in scores and key in current_scores:
            result[f"{key}_delta"] = round(_number(current_scores.get(key)) - _number(scores.get(key)), 3)
    return result


def _local_unsafe_previews(local_goal: dict[str, Any]) -> list[str]:
    gate = local_goal.get("eligible_span_density_gate") if isinstance(local_goal, dict) else {}
    clusters = gate.get("top_unsafe_clusters") if isinstance(gate, dict) else []
    previews: list[str] = []
    for row in clusters if isinstance(clusters, list) else []:
        if not isinstance(row, dict):
            continue
        preview = str(row.get("preview") or "").strip()
        if preview:
            previews.append(preview[:320])
        if len(previews) >= 3:
            break
    return previews


def _local_top_sentence_targets(local_goal: dict[str, Any]) -> list[dict[str, Any]]:
    gate = local_goal.get("eligible_span_density_gate") if isinstance(local_goal, dict) else {}
    targets = gate.get("top_sentence_targets") if isinstance(gate, dict) else []
    rows: list[dict[str, Any]] = []
    for row in targets if isinstance(targets, list) else []:
        if not isinstance(row, dict):
            continue
        preview = str(row.get("preview") or "").strip()
        if not preview:
            continue
        rows.append({
            "sentence_id": row.get("sentence_id"),
            "preview": preview[:320],
            "word_count": row.get("word_count"),
            "generic_hits": row.get("generic_hits"),
        })
        if len(rows) >= 5:
            break
    return rows


def _local_recommended_actions(local_goal: dict[str, Any]) -> list[str]:
    gate = local_goal.get("eligible_span_density_gate") if isinstance(local_goal, dict) else {}
    return _string_list(gate.get("recommended_actions") if isinstance(gate, dict) else [], limit=6)


def _retune_focus_from_goal(local_goal: dict[str, Any]) -> list[str]:
    gate = local_goal.get("eligible_span_density_gate") if isinstance(local_goal, dict) else {}
    footprint = local_goal.get("ai_footprint_gate") if isinstance(local_goal, dict) else {}
    after = footprint.get("after") if isinstance(footprint, dict) else {}
    authorship = after.get("authorship_footprint") if isinstance(after, dict) else {}
    focus: list[str] = []
    if isinstance(gate, dict) and not gate.get("top_unsafe_clusters"):
        focus.append("The unsafe sentence density is cleared; keep that route but make the wording less formally paraphrased.")
    if isinstance(authorship, dict) and _number(authorship.get("topk_calibrated_risk")) > 25.0:
        focus.append("The remaining problem is calibrated top-k: stay closer to simple source wording and avoid formal substitute terms.")
    return focus[:3]


def _source_phrase_anchors(text: str) -> list[str]:
    anchors: list[str] = []
    for sentence in _source_event_beats(text):
        chunks = [sentence]
        for splitter in (", ", "; ", " and ", " but "):
            next_chunks: list[str] = []
            for chunk in chunks:
                next_chunks.extend(part.strip() for part in chunk.split(splitter) if part.strip())
            chunks = next_chunks
        for chunk in chunks:
            cleaned = " ".join(chunk.strip(" .,:;!?()[]{}\"'“”").split())
            if 2 <= len(cleaned.split()) <= 8 and cleaned not in anchors:
                anchors.append(cleaned)
            if len(anchors) >= 12:
                return anchors
    return anchors


def _non_source_terms(source_text: str, candidate_text: str) -> list[str]:
    source_terms = {_normalize_term(token) for token in str(source_text or "").split()}
    source_terms.discard("")
    terms: list[str] = []
    for token in str(candidate_text or "").split():
        normalized = _normalize_term(token)
        if len(normalized) < 7 or normalized in source_terms or normalized in terms:
            continue
        terms.append(normalized)
        if len(terms) >= 16:
            break
    return terms


def _normalize_term(token: str) -> str:
    return str(token or "").strip(" \t\r\n.,:;!?()[]{}\"'“”‘’").casefold()


def _raw_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _string_list(value: Any, *, limit: int) -> list[str]:
    items: list[str] = []
    for item in _raw_list(value):
        text = _short_string(item, limit=260)
        if text and text not in items:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def _supported_or_short_list(value: Any, *, source_text: str, limit: int) -> list[str]:
    items: list[str] = []
    for item in _raw_list(value):
        text = _supported_quote(item, source_text) or _short_string(item, limit=160)
        if text and text not in items:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def _short_string(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    return text[:limit]


def _supported_quote(value: Any, source_text: str) -> str:
    quote = _short_string(value, limit=260)
    if not quote:
        return ""
    return quote if quote in source_text else ""


def build_route_blueprint(*, section: SectionUnit, local_goal: dict[str, Any] | None = None) -> dict[str, Any]:
    beats = _source_event_beats(section.text)
    problem_previews = _local_unsafe_previews(local_goal or {})
    source_metadata = section.metadata.get("source_metadata") if isinstance(section.metadata, dict) else {}
    gate_cluster = source_metadata.get("gate_cluster") if isinstance(source_metadata, dict) else {}
    generic_hits = _number(gate_cluster.get("generic_hits")) if isinstance(gate_cluster, dict) else 0.0
    start_index = _route_start_index(beats=beats, problem_previews=problem_previews, generic_hits=generic_hits)
    ordered_indexes = list(range(start_index, len(beats))) + list(range(0, start_index))
    steps: list[dict[str, Any]] = []
    for ordinal, beat_index in enumerate(ordered_indexes):
        if beat_index >= len(beats):
            continue
        instruction = _route_step_instruction(
            ordinal=ordinal,
            beat_index=beat_index,
            start_index=start_index,
            total=len(beats),
        )
        steps.append({
            "step_id": f"step_{len(steps) + 1:02d}",
            "source_beat_index": beat_index,
            "source_beat": beats[beat_index],
            "instruction": instruction,
        })
        if ordinal == 0 and start_index == 0 and len(beats) >= 4:
            steps.append({
                "step_id": f"step_{len(steps) + 1:02d}",
                "source_beat_index": beat_index,
                "source_beat": beats[beat_index],
                "instruction": "Add one bridge sentence explaining why this starting point shaped the next teaching action.",
            })
    avoid_openers = [beats[index] for index in range(0, start_index)]
    avoid_openers.extend(preview for preview in problem_previews if preview)
    return {
        "strategy": "event_first_rebuild" if start_index > 0 else "starting_point_bridge_rebuild",
        "start_source_beat_index": start_index,
        "avoid_openers": avoid_openers[:4],
        "steps": steps,
        "sentence_jobs": _sentence_jobs_for_blueprint(beats=beats, start_index=start_index),
    }


def _route_start_index(*, beats: list[str], problem_previews: list[str], generic_hits: float) -> int:
    if len(beats) <= 1:
        return 0
    first = beats[0]
    first_is_problem = any(first and first in preview for preview in problem_previews)
    if generic_hits > 0 and first_is_problem:
        return 1
    return 0


def _route_step_instruction(*, ordinal: int, beat_index: int, start_index: int, total: int) -> str:
    if ordinal == 0 and start_index > 0:
        return "Open from this concrete event or action instead of the source's broad summary opener."
    if ordinal == 0:
        return "Open from this starting point, but change the sentence route and avoid copying the source opener."
    if start_index > 0 and beat_index < start_index:
        return "Fold this source meaning into the ending as interpretation; do not use it as the opening."
    if ordinal == total - 1:
        return "Finish with this outcome, making the result visible rather than abstract."
    return "Unpack this beat through action, cause, or response before moving to the next beat."


def _sentence_jobs_for_blueprint(*, beats: list[str], start_index: int) -> list[str]:
    if not beats:
        return []
    jobs: list[str] = []
    if start_index == 0:
        jobs.append("Sentence 1: restate the starting situation using simple source-near wording.")
        if len(beats) >= 4:
            jobs.append("Sentence 2: explain why this starting point shaped the teacher's next action.")
        for index, beat in enumerate(beats[1:], start=2):
            if index == len(beats):
                jobs.append("Final sentence: make the outcome visible in plain wording.")
            else:
                jobs.append("Next sentence: move through the next source beat without formal theory language.")
        return jobs
    jobs.append("Sentence 1: start with the first concrete event/action beat, not the broad source opener.")
    for index, beat in enumerate(beats[start_index + 1:], start=start_index + 2):
        if index == len(beats):
            jobs.append("Next sentence: make the event consequence visible before interpreting it.")
        else:
            jobs.append("Next sentence: move to the next source beat in plain event wording.")
    for _ in beats[:start_index]:
        jobs.append("Final sentence: fold the skipped broad source opener into the meaning after the event.")
    return jobs


def _student_support_role_progression_seed_texts(text: str) -> list[str]:
    """Source-gated route seed for learner support -> role -> outcome clusters."""

    source = str(text or "")
    folded = source.casefold()
    required_terms = (
        "support worker",
        "interacted",
        "past learning",
        "role-playing",
        "group project",
        "positive feedback",
    )
    if any(term not in folded for term in required_terms):
        return []

    pronouns = _dominant_learner_pronouns(source)
    course_word = "course" if "course" in folded else "class"
    role_label = _role_label_from_source(source)
    task_label = _task_label_from_role(role_label)
    outcome_tail = "received positive feedback from them afterward"
    if "teammates" in folded:
        outcome_tail = "received positive feedback from them afterward"
    elif "client" in folded:
        outcome_tail = "received positive feedback afterward"

    student_ref = "the student"
    subject = pronouns["subject"]
    obj = pronouns["object"]
    poss = pronouns["possessive"]
    return [
        (
            f"When {student_ref} first joined the {course_word}, {subject} came to class with a support worker "
            f"and barely interacted with the group. I had to understand that starting point before expecting {obj} "
            f"to participate. Casual conversation gave me a better picture of {poss} past learning experiences. "
            f"After that, patient guidance and role-play moved the confidence work onto {task_label} {subject} "
            f"could actually handle. The group project gave {obj} a specific role rather than a general instruction "
            f"to join in. As the {role_label}, {subject} had to guide the task and communicate with teammates. "
            f"{subject.capitalize()} managed that responsibility successfully, led the group, and {outcome_tail}."
        )
    ]


def _dominant_learner_pronouns(text: str) -> dict[str, str]:
    tokens = [token.strip(" \t\r\n.,:;!?()[]{}\"'“”‘’").casefold() for token in str(text or "").split()]
    counts = {key: tokens.count(key) for key in ("he", "she", "they")}
    if counts.get("she", 0) > counts.get("he", 0) and counts.get("she", 0) >= counts.get("they", 0):
        return {"subject": "she", "object": "her", "possessive": "her"}
    if counts.get("they", 0) > counts.get("he", 0):
        return {"subject": "they", "object": "them", "possessive": "their"}
    return {"subject": "he", "object": "him", "possessive": "his"}


def _role_label_from_source(text: str) -> str:
    tokens = [
        token.strip(" \t\r\n.,:;!?()[]{}\"'“”‘’")
        for token in str(text or "").split()
        if token.strip(" \t\r\n.,:;!?()[]{}\"'“”‘’")
    ]
    for index, token in enumerate(tokens):
        if token.casefold() != "manager":
            continue
        start = max(0, index - 2)
        phrase_tokens = tokens[start:index + 1]
        if phrase_tokens:
            return " ".join(phrase_tokens)
    anchors = _source_phrase_anchors(text)
    for anchor in anchors:
        lowered = anchor.casefold()
        if "manager" in lowered and len(anchor.split()) <= 5:
            return anchor
    folded = str(text or "").casefold()
    if "manager" in folded:
        return "manager"
    return "specific role"


def _task_label_from_role(role_label: str) -> str:
    lowered = str(role_label or "").casefold()
    if "salon" in lowered:
        return "a salon task"
    if "manager" in lowered:
        return "a group task"
    return "a concrete group task"


def _local_goal(original_text: str, candidate_text: str) -> dict[str, Any]:
    original_report = _scan_report(original_text)
    candidate_report = original_report if original_text == candidate_text else _scan_report(candidate_text)
    return evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=original_report,
        candidate_report=candidate_report,
    ).to_dict()


def _source_event_beats(text: str) -> list[str]:
    return [sentence for sentence in _sentences(text)[:10] if sentence.strip()]


def _section_before_context(section: SectionUnit) -> str:
    value = section.metadata.get("before_context") if isinstance(section.metadata, dict) else ""
    return _short_string(value, limit=420)


def _section_after_context(section: SectionUnit) -> str:
    value = section.metadata.get("after_context") if isinstance(section.metadata, dict) else ""
    return _short_string(value, limit=420)


def _referential_continuity(text: str, *, before_context: str = "") -> dict[str, Any]:
    sentences = _source_event_beats(text)
    first_sentence = sentences[0] if sentences else ""
    first_token = ""
    for token in first_sentence.split():
        cleaned = token.strip(" \t\r\n.,:;!?()[]{}\"'“”‘’")
        if cleaned:
            first_token = cleaned
            break
    personal_pronouns = {"he", "she", "they", "him", "her", "them"}
    preserve_opening_subject = first_token.casefold() in personal_pronouns
    named_refs = _named_references_from_text(before_context)
    for sentence in sentences:
        for token in sentence.split():
            cleaned = token.strip(" \t\r\n.,:;!?()[]{}\"'“”‘’")
            if not cleaned or not cleaned[:1].isupper() or cleaned == first_token:
                continue
            if cleaned not in named_refs:
                named_refs.append(cleaned)
            if len(named_refs) >= 6:
                break
        if len(named_refs) >= 6:
            break
    return {
        "opening_subject": first_token,
        "preserve_opening_subject": preserve_opening_subject,
        "named_references_in_cluster": named_refs,
        "preferred_reference": named_refs[0] if named_refs else "",
        "instruction": (
            "Use the same opening subject or an established named reference from context naturally; do not generalize the subject and do not explain the reference parenthetically."
            if preserve_opening_subject else
            "Keep source-supported names and pronouns consistent."
        ),
    }


def _named_references_from_text(text: str) -> list[str]:
    refs: list[str] = []
    sentence_initials = {
        sentence.split()[0].strip(" \t\r\n.,:;!?()[]{}\"'“”‘’")
        for sentence in _source_event_beats(text)
        if sentence.split()
    }
    for token in str(text or "").split():
        cleaned = token.strip(" \t\r\n.,:;!?()[]{}\"'“”‘’")
        if not cleaned or not cleaned[:1].isupper() or cleaned in sentence_initials:
            continue
        if cleaned not in refs:
            refs.append(cleaned)
        if len(refs) >= 6:
            break
    return refs


def _compact_residual_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "section_id": row.get("section_id"),
        "variant_id": row.get("variant_id"),
        "label": row.get("label"),
        "word_count": row.get("word_count"),
        "scores": row.get("scores"),
        "incremental": row.get("incremental"),
        "local_scores": row.get("local_scores"),
        "apply_status": row.get("apply_status"),
        "text": row.get("text"),
    }


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
