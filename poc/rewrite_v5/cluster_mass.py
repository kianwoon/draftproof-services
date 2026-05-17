"""Length-preserved cluster replacement experiment for V5.

This module restores the narrow experiment that produced useful movement on
`test_content7.txt`: replace scanner-ranked unsafe clusters at roughly source
length, then score cumulative top-N replacements on the full document.
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
from rewrite_v4.cluster_patch import build_cluster_repair_units
from rewrite_v4.validation import parse_generator_variants, source_grounding_integrity

from .experiment import (
    _add_deltas,
    _candidate_sort_key,
    _merge_provider_options,
    _score_summary,
    _section_from_cluster,
    _variants_response_format,
    apply_section_variant,
    build_fact_map,
    fact_map_integrity,
)
from .models import RecompositionVariant, SectionUnit


def run_v5_cluster_mass_replacement_experiment(
    *,
    input_text: str,
    output_dir: str | Path,
    max_clusters: int = 5,
    variant_count: int = 3,
    min_word_ratio: float = 0.90,
    fallback_min_word_ratio: float = 0.75,
    max_word_ratio: float = 1.50,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate source-length cluster replacements and score cumulative packs."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_text = str(input_text or "")
    baseline_report = _scan_report(source_text)
    baseline_goal = evaluate_rewrite_goal(
        original_text=source_text,
        candidate_text=source_text,
        original_report=baseline_report,
        candidate_report=baseline_report,
    ).to_dict()
    baseline_scores = _score_summary(source_text, baseline_report, baseline_goal)
    cluster_units = build_cluster_repair_units(
        text=source_text,
        report=baseline_report,
        goal=baseline_goal,
        limit=max_clusters,
        context_chars=220,
    )
    route_units = [_section_from_cluster(unit) for unit in cluster_units]
    gateway = LLMGateway(LLMConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V5_CLUSTER_MASS_MAX_TOKENS", 8000, minimum=1000, maximum=12000),
        temperature=_float_env("DRAFTPROOF_REWRITE_V5_CLUSTER_MASS_TEMPERATURE", 0.25, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_REWRITE_V5_CLUSTER_MASS_TOP_P", 0.85, minimum=0.1, maximum=1.0),
        timeout=180,
        extra_body=extra_body,
    ))

    rows: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    for unit in route_units:
        fact_map = build_fact_map(unit, baseline_goal)
        variants, diagnostics, prompt, completion = generate_cluster_mass_variants(
            section=unit,
            fact_map=fact_map,
            gateway=gateway,
            variant_count=variant_count,
            min_word_ratio=min_word_ratio,
            fallback_min_word_ratio=fallback_min_word_ratio,
            max_word_ratio=max_word_ratio,
        )
        stem = unit.section_id
        (out_dir / f"{stem}_cluster_mass_prompt.json.txt").write_text(prompt)
        (out_dir / f"{stem}_cluster_mass_completion.json.txt").write_text(completion)
        prompts.append({
            "route_window": unit.to_dict(),
            "fact_map": fact_map.to_dict(),
            "generator_diagnostics": diagnostics,
        })
        for variant in variants:
            rows.append(_score_cluster_mass_variant(
                original_text=source_text,
                baseline_report=baseline_report,
                baseline_goal=baseline_goal,
                baseline_scores=baseline_scores,
                section=unit,
                variant=variant,
                output_dir=out_dir,
            ))

    selected_by_cluster = _select_best_per_cluster(rows)
    cumulative_results = _score_cumulative_cluster_sets(
        source_text=source_text,
        baseline_report=baseline_report,
        baseline_scores=baseline_scores,
        selected_rows=selected_by_cluster,
        output_dir=out_dir,
    )
    best = max(cumulative_results, key=_candidate_sort_key, default=None)
    final_text = str(best.get("candidate_text") if best else source_text)
    final_scores = dict(best.get("scores") if best else baseline_scores)
    payload = {
        "stage": "v5_cluster_mass_replacement",
        "baseline_scores": baseline_scores,
        "route_windows": [unit.to_dict() for unit in route_units],
        "prompts": prompts,
        "cluster_results": [_compact_cluster_mass_row(row) for row in rows],
        "selected_by_cluster": [_compact_cluster_mass_row(row) for row in selected_by_cluster],
        "cumulative_results": [_compact_cumulative_row(row) for row in cumulative_results],
        "best_scored_candidate": _compact_cumulative_row(best),
        "accepted_candidate": _compact_cumulative_row(best),
        "final_scores": final_scores,
        "goal": (best or {}).get("goal") or {
            "status": baseline_goal.get("status"),
            "goal_met": baseline_goal.get("goal_met"),
            "reason": baseline_goal.get("reason"),
        },
        "rewritten_document": final_text,
    }
    (out_dir / "v5_cluster_mass_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (out_dir / "v5_cluster_mass_rewritten_document.txt").write_text(final_text)
    return payload


def build_cluster_mass_prompt(
    *,
    section: SectionUnit,
    fact_map: Any,
    variant_count: int = 3,
    min_word_ratio: float = 0.90,
    max_word_ratio: float = 1.50,
) -> str:
    variants = max(1, min(3, int(variant_count or 1)))
    min_words = max(1, round(section.word_count * min_word_ratio))
    max_words = max(min_words + 1, round(section.word_count * max_word_ratio))
    payload = {
        "task": "length_preserved_route_window_replacement",
        "route_window": {
            "section_id": section.section_id,
            "source_text": section.text,
            "source_word_count": section.word_count,
            "target_word_count": {
                "min": min_words,
                "max": max_words,
                "preferred": section.word_count,
            },
        },
        "writer_profile": {
            "education_level": "bachelor_degree",
            "voice": "plain undergraduate essay with practical teaching reflection where already present",
            "tone": "clear and natural, not polished journal style",
        },
        "fact_map": fact_map.to_dict(),
        "replacement_goal": [
            "Replace the whole route window, not individual phrases.",
            "Keep the same core facts, names, examples, citations, and viewpoint.",
            "Rebuild the route so the window moves through concrete context, practical difficulty, response, and implication.",
            "Keep the replacement near the source length; do not compress heavily and do not expand with filler.",
            "Use only details already present in route_window.source_text.",
        ],
        "rules": [
            "route_window.source_text is the only fact source for replacement text.",
            "Every variant must stay within target_word_count.min and target_word_count.max.",
            "If a draft is under target_word_count.min, keep more source detail instead of stopping early.",
            "Preserve all citations exactly.",
            "Preserve protected terms, names, labels, source codes, and condition names exactly.",
            "Do not add new facts, examples, names, dates, numbers, citations, or outside knowledge.",
            "Do not import facts from surrounding paragraphs or context.",
            "Do not add headings, bullets, markdown, HTML, labels, or commentary.",
            "Return replacement text only for the route window.",
            f"Return exactly {variants} {'variant' if variants == 1 else 'variants'}.",
            "Return exactly the allowed keys for each variant: variant_id, text.",
        ],
        "avoid": [
            "surface synonym swapping",
            "small phrase patch only",
            "generic summary conclusion",
            "overly tidy this-shows route",
            "professional copywriting voice",
            "elevated abstract phrasing",
            "compressed summary",
        ],
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, variants + 1)
            ]
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_cluster_mass_variants(
    *,
    section: SectionUnit,
    fact_map: Any,
    gateway: LLMGateway,
    variant_count: int = 3,
    min_word_ratio: float = 0.90,
    fallback_min_word_ratio: float = 0.75,
    max_word_ratio: float = 1.50,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    prompt = build_cluster_mass_prompt(
        section=section,
        fact_map=fact_map,
        variant_count=variant_count,
        min_word_ratio=min_word_ratio,
        max_word_ratio=max_word_ratio,
    )
    variants = max(1, min(3, int(variant_count or 1)))
    structured = structured_json_request_options(getattr(gateway, "model", None), _variants_response_format(variants))
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured.get("provider"))
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a variants array.",
        response_format=structured.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_REWRITE_V5_CLUSTER_MASS_TEMPERATURE", 0.25, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_REWRITE_V5_CLUSTER_MASS_TOP_P", 0.85, minimum=0.1, maximum=1.0),
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V5_CLUSTER_MASS_MAX_TOKENS", 8000, minimum=1000, maximum=12000),
    )
    raw = response.raw_content or response.content
    min_words = max(1, round(section.word_count * min_word_ratio))
    max_words = max(min_words + 1, round(section.word_count * max_word_ratio))
    parsed, diagnostics = parse_generator_variants(
        raw,
        min_words=min_words,
        max_words=max_words,
        source_text=section.text,
    )
    fallback_diagnostics: dict[str, Any] | None = None
    if not parsed and fallback_min_word_ratio < min_word_ratio:
        fallback_min_words = max(1, round(section.word_count * fallback_min_word_ratio))
        parsed, fallback_diagnostics = parse_generator_variants(
            raw,
            min_words=fallback_min_words,
            max_words=max_words,
            source_text=section.text,
        )
    variants_out = [
        RecompositionVariant(variant_id=row.variant_id, text=row.text, word_count=row.word_count)
        for row in parsed
    ]
    return variants_out, {
        **diagnostics,
        "fallback_parse": fallback_diagnostics,
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "structured_output_mode": structured.get("structured_output_mode"),
        "word_range": {
            "min_words": min_words,
            "fallback_min_words": max(1, round(section.word_count * fallback_min_word_ratio)),
            "max_words": max_words,
        },
    }, prompt, raw


def _score_cluster_mass_variant(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_goal: dict[str, Any],
    baseline_scores: dict[str, Any],
    section: SectionUnit,
    variant: RecompositionVariant,
    output_dir: Path,
) -> dict[str, Any]:
    candidate_text, apply_status = apply_section_variant(original_text, section, variant.text)
    grounding = source_grounding_integrity(section.text, variant.text, repair_mode="source_near_route_rebuild")
    fact_map = build_fact_map(section, baseline_goal)
    fact_integrity = fact_map_integrity(fact_map, variant.text)
    if not apply_status.get("applied") or not grounding.get("passed") or not fact_integrity.get("passed"):
        return {
            "section_id": section.section_id,
            "variant_id": variant.variant_id,
            "word_count": variant.word_count,
            "scores": {**baseline_scores, "ai_delta": 0.0, "external_delta": 0.0, "rank_delta": 0.0},
            "goal": {
                "status": "rejected_candidate",
                "goal_met": False,
                "reason": _rejection_reason(apply_status, grounding, fact_integrity),
            },
            "apply_status": apply_status,
            "source_grounding": grounding,
            "fact_integrity": fact_integrity,
            "text": variant.text,
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
    safe_name = f"{section.section_id}_{variant.variant_id}"
    (output_dir / f"{safe_name}.txt").write_text(candidate_text)
    (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
    return {
        "section_id": section.section_id,
        "variant_id": variant.variant_id,
        "word_count": variant.word_count,
        "scores": scores,
        "goal": {
            "status": candidate_goal.get("status"),
            "goal_met": candidate_goal.get("goal_met"),
            "reason": candidate_goal.get("reason"),
        },
        "apply_status": apply_status,
        "source_grounding": grounding,
        "fact_integrity": fact_integrity,
        "candidate_text": candidate_text,
        "candidate_report": candidate_report,
        "candidate_goal": candidate_goal,
        "text": variant.text,
    }


def _select_best_per_cluster(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not ((row.get("apply_status") or {}).get("applied")):
            continue
        if not ((row.get("source_grounding") or {}).get("passed")):
            continue
        if not ((row.get("fact_integrity") or {}).get("passed")):
            continue
        section_id = str(row.get("section_id") or "")
        if not section_id:
            continue
        grouped.setdefault(section_id, []).append(row)
    selected: list[dict[str, Any]] = []
    for section_id in sorted(grouped):
        selected.append(max(grouped[section_id], key=_candidate_sort_key))
    return selected


def _score_cumulative_cluster_sets(
    *,
    source_text: str,
    baseline_report: dict[str, Any],
    baseline_scores: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    cumulative: list[dict[str, Any]] = []
    section_by_id = {
        str(row.get("section_id") or ""): SectionUnit(
            section_id=str(row.get("section_id") or ""),
            heading="",
            text=str(((row.get("apply_status") or {}).get("old_text")) or ""),
            start_char=0,
            end_char=0,
            paragraph_count=1,
            word_count=0,
        )
        for row in selected_rows
    }
    # Reconstruct sections from successful apply metadata and original slices.
    section_by_id.clear()
    for row in selected_rows:
        status = row.get("apply_status") if isinstance(row.get("apply_status"), dict) else {}
        section_id = str(row.get("section_id") or "")
        start = int(status.get("start_char") or 0)
        end = int(status.get("end_char") or 0)
        if section_id and end > start:
            old_text = source_text[start:end]
            section_by_id[section_id] = SectionUnit(
                section_id=section_id,
                heading="",
                text=old_text,
                start_char=start,
                end_char=end,
                paragraph_count=max(1, old_text.count("\n\n") + 1),
                word_count=word_count(old_text),
            )
    for count in range(1, len(selected_rows) + 1):
        subset = selected_rows[:count]
        candidate_text = source_text
        apply_statuses: list[dict[str, Any]] = []
        for row in sorted(subset, key=lambda item: section_by_id[str(item.get("section_id"))].start_char, reverse=True):
            section = section_by_id[str(row.get("section_id"))]
            candidate_text, status = apply_section_variant(candidate_text, section, str(row.get("text") or ""))
            apply_statuses.append({
                **status,
                "section_id": section.section_id,
                "old_word_count": section.word_count,
                "new_word_count": word_count(str(row.get("text") or "")),
            })
        if not all(status.get("applied") for status in apply_statuses):
            continue
        candidate_report = _scan_report(candidate_text)
        candidate_goal = evaluate_rewrite_goal(
            original_text=source_text,
            candidate_text=candidate_text,
            original_report=baseline_report,
            candidate_report=candidate_report,
        ).to_dict()
        scores = _score_summary(source_text, candidate_report, candidate_goal)
        _add_deltas(scores, baseline_scores)
        candidate_id = f"top_{count:02d}_clusters"
        (output_dir / f"{candidate_id}.txt").write_text(candidate_text)
        (output_dir / f"{candidate_id}_scan.json").write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2))
        cumulative.append({
            "candidate_id": candidate_id,
            "cluster_count": count,
            "selected": [
                {
                    "section_id": row.get("section_id"),
                    "variant_id": row.get("variant_id"),
                    "word_count": row.get("word_count"),
                    "source_word_count": section_by_id[str(row.get("section_id"))].word_count,
                }
                for row in subset
            ],
            "scores": scores,
            "goal": {
                "status": candidate_goal.get("status"),
                "goal_met": candidate_goal.get("goal_met"),
                "reason": candidate_goal.get("reason"),
            },
            "apply_statuses": apply_statuses,
            "candidate_text": candidate_text,
            "candidate_report": candidate_report,
            "candidate_goal": candidate_goal,
        })
    cumulative.sort(key=_candidate_sort_key, reverse=True)
    return cumulative


def _compact_cluster_mass_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    grounding = row.get("source_grounding") if isinstance(row.get("source_grounding"), dict) else {}
    fact_integrity = row.get("fact_integrity") if isinstance(row.get("fact_integrity"), dict) else {}
    return {
        "section_id": row.get("section_id"),
        "variant_id": row.get("variant_id"),
        "word_count": row.get("word_count"),
        "scores": row.get("scores"),
        "goal": row.get("goal"),
        "apply_status": row.get("apply_status"),
        "source_grounding_passed": bool(grounding.get("passed")),
        "fact_integrity_passed": bool(fact_integrity.get("passed")),
        "source_grounding_failures": grounding.get("failures") or [],
        "fact_integrity_failures": fact_integrity.get("failures") or [],
        "text": row.get("text"),
    }


def _compact_cumulative_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "candidate_id": row.get("candidate_id"),
        "cluster_count": row.get("cluster_count"),
        "selected": row.get("selected"),
        "scores": row.get("scores"),
        "goal": row.get("goal"),
        "apply_statuses": row.get("apply_statuses"),
    }


def _rejection_reason(apply_status: dict[str, Any], grounding: dict[str, Any], fact_integrity: dict[str, Any]) -> str:
    if not apply_status.get("applied"):
        return str(apply_status.get("reason") or "section_apply_failed")
    if not grounding.get("passed"):
        return "source_grounding_failed"
    if not fact_integrity.get("passed"):
        return "fact_integrity_failed"
    return "candidate_rejected"


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
