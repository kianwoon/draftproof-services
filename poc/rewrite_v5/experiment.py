"""V5 controlled reconstruction experiment.

V5 is intentionally isolated from production. It tests whether rebuilding a
section from a constrained fact map can move signals that V4 patch layers only
shave down.
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
from rewrite_v3.pipeline import _ai_footprint_profile, _badge_ai, _footprint_risk, _scan_report, _topk
from rewrite_v3.scanner_controlled_executor import scanner_controlled_metrics, scanner_controlled_rank
from rewrite_v4.cluster_patch import build_cluster_repair_units
from rewrite_v4.experiment import run_v4_residual_cluster_experiment
from rewrite_v4.validation import parse_json_object, source_grounding_integrity

from .models import FactMap, RecompositionVariant, SectionUnit


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


def run_v5_section_reconstruction_experiment(
    *,
    input_text: str,
    output_dir: str | Path,
    section_id: str | None = None,
    variant_count: int = 2,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    sections = build_section_units(original_text, baseline_goal)
    selected = _select_section(sections, section_id=section_id)
    if selected is None:
        payload = {
            "baseline": baseline_scores,
            "sections": [section.to_dict() for section in sections],
            "results": [],
            "best_candidate": None,
            "reason": "no_section_selected",
        }
        (out_dir / "v5_section_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    fact_map = build_fact_map(selected, baseline_goal)
    gateway = LLMGateway(LLMConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=8000,
        temperature=0.25,
        top_p=0.85,
        timeout=180,
        extra_body=extra_body,
    ))
    variants, diagnostics, prompt, completion = generate_section_recompositions(
        section=selected,
        fact_map=fact_map,
        gateway=gateway,
        variant_count=variant_count,
    )
    (out_dir / "v5_prompt.json.txt").write_text(prompt)
    (out_dir / "v5_completion.json.txt").write_text(completion)
    rows = [
        _score_section_variant(
            original_text=original_text,
            baseline_report=baseline_report,
            baseline_goal=baseline_goal,
            baseline_scores=baseline_scores,
            section=selected,
            fact_map=fact_map,
            variant=variant,
            output_dir=out_dir,
        )
        for variant in variants
    ]
    rows.sort(key=_candidate_sort_key, reverse=True)
    accepted = next((row for row in rows if _is_safe_candidate(row)), None)
    best = accepted or (rows[0] if rows else None)
    payload = {
        "baseline": baseline_scores,
        "sections": [section.to_dict() for section in sections],
        "selected_section": selected.to_dict(),
        "fact_map": fact_map.to_dict(),
        "generator_diagnostics": diagnostics,
        "results": [_compact_row(row) for row in rows],
        "best_scored_candidate": _compact_row(best) if best else None,
        "accepted_candidate": _compact_row(accepted) if accepted else None,
    }
    if accepted:
        payload["rewritten_document"] = accepted.get("candidate_text", "")
        payload["final_scores"] = accepted.get("scores", {})
    (out_dir / "v5_section_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    if accepted and accepted.get("candidate_text"):
        (out_dir / "v5_rewritten_document.txt").write_text(str(accepted.get("candidate_text") or ""))
    return payload


def run_v5_route_window_reconstruction_experiment(
    *,
    input_text: str,
    output_dir: str | Path,
    max_windows: int = 6,
    variant_count: int = 2,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild scanner-ranked route windows from fact maps.

    This is the V5 feasibility path for hard cases where full-section
    reconstruction is too large to preserve every source claim.
    """

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
    cluster_units = build_cluster_repair_units(
        text=original_text,
        report=baseline_report,
        goal=baseline_goal,
        limit=max_windows,
        context_chars=220,
    )
    route_units = [_section_from_cluster(unit) for unit in cluster_units]
    gateway = LLMGateway(LLMConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=8000,
        temperature=0.25,
        top_p=0.85,
        timeout=180,
        extra_body=extra_body,
    ))
    rows: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    for unit in route_units:
        fact_map = build_fact_map(unit, baseline_goal)
        variants, diagnostics, prompt, completion = generate_section_recompositions(
            section=unit,
            fact_map=fact_map,
            gateway=gateway,
            variant_count=variant_count,
        )
        stem = unit.section_id
        (out_dir / f"{stem}_prompt.json.txt").write_text(prompt)
        (out_dir / f"{stem}_completion.json.txt").write_text(completion)
        prompts.append({
            "route_window": unit.to_dict(),
            "fact_map": fact_map.to_dict(),
            "generator_diagnostics": diagnostics,
        })
        for variant in variants:
            row = _score_section_variant(
                original_text=original_text,
                baseline_report=baseline_report,
                baseline_goal=baseline_goal,
                baseline_scores=baseline_scores,
                section=unit,
                fact_map=fact_map,
                variant=variant,
                output_dir=out_dir,
            )
            row["generator_diagnostics"] = diagnostics
            rows.append(row)

    rows.sort(key=_candidate_sort_key, reverse=True)
    accepted = next((row for row in rows if _is_safe_candidate(row)), None)
    best = accepted or (rows[0] if rows else None)
    payload = {
        "baseline": baseline_scores,
        "route_windows": [unit.to_dict() for unit in route_units],
        "prompts": prompts,
        "results": [_compact_row(row) for row in rows],
        "best_scored_candidate": _compact_row(best) if best else None,
        "accepted_candidate": _compact_row(accepted) if accepted else None,
    }
    if accepted:
        payload["rewritten_document"] = accepted.get("candidate_text", "")
        payload["final_scores"] = accepted.get("scores", {})
    (out_dir / "v5_route_window_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    if accepted and accepted.get("candidate_text"):
        (out_dir / "v5_route_window_rewritten_document.txt").write_text(str(accepted.get("candidate_text") or ""))
    return payload


def run_v5_route_window_stack_experiment(
    *,
    input_text: str,
    output_dir: str | Path,
    max_windows: int = 8,
    route_variant_count: int = 3,
    cleanup_rounds: int = 2,
    cleanup_max_clusters: int = 6,
    cleanup_variant_count: int = 2,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the feasible V5 stack: route-window reconstruction, then V4 cleanup."""

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

    route_result = run_v5_route_window_reconstruction_experiment(
        input_text=source_text,
        output_dir=out_dir / "v5_route_window",
        max_windows=max_windows,
        variant_count=route_variant_count,
        api_key=api_key,
        model=model,
        base_url=base_url,
        extra_body=extra_body,
    )
    route_text = str(route_result.get("rewritten_document") or source_text)
    cleanup_result: dict[str, Any] | None = None
    final_text = route_text
    if route_result.get("accepted_candidate"):
        cleanup_result = run_v4_residual_cluster_experiment(
            input_text=route_text,
            output_dir=out_dir / "v4_residual_cleanup",
            max_rounds=cleanup_rounds,
            max_clusters=cleanup_max_clusters,
            variant_count=cleanup_variant_count,
            api_key=api_key,
            model=model,
            base_url=base_url,
            extra_body=extra_body,
        )
        final_text = str(cleanup_result.get("rewritten_document") or route_text)

    final_report = _scan_report(final_text)
    final_goal = evaluate_rewrite_goal(
        original_text=source_text,
        candidate_text=final_text,
        original_report=baseline_report,
        candidate_report=final_report,
    ).to_dict()
    final_scores = _score_summary(source_text, final_report, final_goal)
    _add_deltas(final_scores, baseline_scores)
    stage = "v5_route_then_v4_cleanup" if cleanup_result else "v5_route_only" if route_result.get("accepted_candidate") else "no_candidate"
    payload = {
        "stage": stage,
        "baseline_scores": baseline_scores,
        "route_scores": route_result.get("final_scores"),
        "cleanup_scores": cleanup_result.get("final_scores") if cleanup_result else None,
        "final_scores": final_scores,
        "accepted_route_candidate": route_result.get("accepted_candidate"),
        "cleanup_accepted": cleanup_result.get("accepted") if cleanup_result else [],
        "goal": {
            "status": final_goal.get("status"),
            "goal_met": final_goal.get("goal_met"),
            "reason": final_goal.get("reason"),
        },
        "summary": _stack_summary(
            baseline=baseline_scores,
            final=final_scores,
            route_result=route_result,
            cleanup_result=cleanup_result,
        ),
        "rewritten_document": final_text,
    }
    (out_dir / "v5_stack_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (out_dir / "v5_stack_rewritten_document.txt").write_text(final_text)
    return payload


def build_section_units(text: str, goal: dict[str, Any] | None = None) -> list[SectionUnit]:
    source = str(text or "")
    blocks = _paragraph_blocks(source)
    if not blocks:
        return []
    heading_indexes = [index for index, block in enumerate(blocks) if _is_heading(block["text"])]
    if not heading_indexes or heading_indexes[0] != 0:
        heading_indexes = [0, *heading_indexes]
    sections: list[SectionUnit] = []
    seen_starts: set[int] = set()
    for ordinal, start_index in enumerate(heading_indexes):
        if start_index in seen_starts:
            continue
        seen_starts.add(start_index)
        end_index = heading_indexes[ordinal + 1] if ordinal + 1 < len(heading_indexes) else len(blocks)
        section_blocks = blocks[start_index:end_index]
        if not section_blocks:
            continue
        section_start = section_blocks[0]["start"]
        section_end = section_blocks[-1]["end"]
        section_text = source[section_start:section_end]
        heading = _heading_text(section_blocks[0]["text"]) if _is_heading(section_blocks[0]["text"]) else "Opening"
        sections.append(SectionUnit(
            section_id=f"sec{len(sections) + 1:03d}",
            heading=heading,
            text=section_text,
            start_char=section_start,
            end_char=section_end,
            paragraph_count=len(section_blocks),
            word_count=word_count(section_text),
            metadata={
                "block_indexes": [row["index"] for row in section_blocks],
                "ordinal": len(sections) + 1,
            },
        ))
    total = len(sections)
    sections = [
        SectionUnit(
            section_id=section.section_id,
            heading=section.heading,
            text=section.text,
            start_char=section.start_char,
            end_char=section.end_char,
            paragraph_count=section.paragraph_count,
            word_count=section.word_count,
            metadata={**section.metadata, "section_count": total},
        )
        for section in sections
    ]
    return _rank_sections_by_goal(sections, goal or {})


def build_fact_map(section: SectionUnit, goal: dict[str, Any] | None = None) -> FactMap:
    sentences = _sentences(section.text)
    citations = tuple(_citation_spans(section.text))
    protected_terms = tuple(_protected_terms(section.text, citations))
    personal = tuple(sentence for sentence in sentences if _has_personal_marker(sentence))[:16]
    fixed_facts = tuple(sentence for sentence in sentences if sentence not in personal)[:28]
    if not fixed_facts:
        fixed_facts = tuple(sentences[:14])
    issues = _writing_issues_for_section(section, goal or {})
    return FactMap(
        section_id=section.section_id,
        section_role=_section_role(section),
        fixed_facts=fixed_facts,
        personal_observations=personal,
        citations=citations,
        protected_terms=protected_terms,
        current_route=tuple(_route_labels(sentences[:8])),
        better_route=tuple(_better_route_for_section(section)),
        writing_issues=tuple(issues),
    )


def build_recomposer_prompt(*, section: SectionUnit, fact_map: FactMap, variant_count: int = 2) -> str:
    variants = max(1, min(3, int(variant_count or 1)))
    payload = {
        "task": "controlled_section_reconstruction_from_fact_map",
        "section": {
            "section_id": section.section_id,
            "heading": section.heading,
            "paragraph_count": section.paragraph_count,
            "word_count": section.word_count,
        },
        "writer_profile": {
            "education_level": "bachelor_degree",
            "voice": "plain undergraduate essay with source-specific reflection where already present",
            "tone": "clear, serious, natural, not scholarly-polished",
            "allowed_texture": "minor ordinary student phrasing is acceptable when meaning stays clear",
            "forbidden_texture": "slang, fake errors, advanced journal style, decorative wording",
        },
        "fact_map": fact_map.to_dict(),
        "reconstruction_rules": [
            "Write a replacement for this section or route window only.",
            "Use only facts, claims, citations, examples, and observations from fact_map.",
            "Keep the section heading if one exists in the original section.",
            "Do not add a heading when section.heading is empty.",
            "Preserve all citations exactly.",
            "Preserve names, labels, codes, conditions, and concrete activities exactly where they appear.",
            "Do not follow fact_map.current_route sentence by sentence; use fact_map.better_route.",
            "Represent every fixed_fact and personal_observation at least once; do not drop a source claim to make the section shorter.",
            "Do not add new facts, examples, names, dates, numbers, citations, or outside knowledge.",
            "Do not compress away source details just to sound smoother.",
            "Keep within 80% to 120% of the source section word count.",
            f"Return exactly {variants} {'variant' if variants == 1 else 'variants'}.",
        ],
        "avoid": [
            "generic summary wording",
            "this demonstrates / this shows repeated route",
            "overly tidy cause-effect conclusion",
            "professional copyediting voice",
            "full document rewrite",
            "markdown bullets",
            "HTML",
            "commentary",
        ],
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, variants + 1)
            ]
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_section_recompositions(
    *,
    section: SectionUnit,
    fact_map: FactMap,
    gateway: LLMGateway,
    variant_count: int = 2,
) -> tuple[list[RecompositionVariant], dict[str, Any], str, str]:
    prompt = build_recomposer_prompt(section=section, fact_map=fact_map, variant_count=variant_count)
    variants = max(1, min(3, int(variant_count or 1)))
    structured = structured_json_request_options(getattr(gateway, "model", None), _variants_response_format(variants))
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured.get("provider"))
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a variants array.",
        response_format=structured.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_REWRITE_V5_TEMPERATURE", 0.36, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_REWRITE_V5_TOP_P", 0.86, minimum=0.1, maximum=1.0),
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V5_MAX_TOKENS", 6000, minimum=1000, maximum=12000),
    )
    raw = response.raw_content or response.content
    parsed, diagnostics = _parse_variants(raw, source_words=section.word_count)
    return parsed, {
        **diagnostics,
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
        "structured_output_mode": structured.get("structured_output_mode"),
    }, prompt, raw


def _score_section_variant(
    *,
    original_text: str,
    baseline_report: dict[str, Any],
    baseline_goal: dict[str, Any],
    baseline_scores: dict[str, Any],
    section: SectionUnit,
    fact_map: FactMap,
    variant: RecompositionVariant,
    output_dir: Path,
) -> dict[str, Any]:
    candidate_text, apply_status = apply_section_variant(original_text, section, variant.text)
    grounding = source_grounding_integrity(section.text, variant.text)
    fact_integrity = fact_map_integrity(fact_map, variant.text)
    if not apply_status.get("applied") or not grounding.get("passed") or not fact_integrity.get("passed"):
        return {
            "section_id": section.section_id,
            "variant_id": variant.variant_id,
            "text": variant.text,
            "word_count": variant.word_count,
            "apply_status": apply_status,
            "source_grounding": grounding,
            "fact_integrity": fact_integrity,
            "scores": {**baseline_scores, "ai_delta": 0.0, "external_delta": 0.0, "rank_delta": 0.0},
            "goal": {
                "status": "rejected_candidate",
                "goal_met": False,
                "reason": _rejection_reason(apply_status, grounding, fact_integrity),
            },
        }
    report = _scan_report(candidate_text)
    goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=candidate_text,
        original_report=baseline_report,
        candidate_report=report,
    ).to_dict()
    scores = _score_summary(original_text, report, goal)
    _add_deltas(scores, baseline_scores)
    safe_name = f"{section.section_id}_{variant.variant_id}"
    (output_dir / f"{safe_name}.txt").write_text(candidate_text)
    (output_dir / f"{safe_name}_scan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return {
        "section_id": section.section_id,
        "variant_id": variant.variant_id,
        "text": variant.text,
        "word_count": variant.word_count,
        "apply_status": apply_status,
        "source_grounding": grounding,
        "fact_integrity": fact_integrity,
        "scores": scores,
        "candidate_text": candidate_text,
        "candidate_report": report,
        "candidate_goal": goal,
        "goal": {
            "status": goal.get("status"),
            "goal_met": goal.get("goal_met"),
            "reason": goal.get("reason"),
        },
    }


def _section_from_cluster(unit: Any) -> SectionUnit:
    return SectionUnit(
        section_id=f"route_{str(unit.cluster_id).split('_')[-1]}",
        heading="",
        text=str(unit.text or ""),
        start_char=int(unit.start_char),
        end_char=int(unit.end_char),
        paragraph_count=max(1, str(unit.text or "").count("\n\n") + 1),
        word_count=word_count(str(unit.text or "")),
        metadata={
            "unit_type": "route_window",
            "cluster_id": unit.cluster_id,
            "risk_score": unit.risk_score,
            "sentence_count": unit.sentence_count,
            "source_metadata": unit.metadata,
        },
    )


def apply_section_variant(text: str, section: SectionUnit, replacement_text: str) -> tuple[str, dict[str, Any]]:
    source = str(text or "")
    replacement = str(replacement_text or "").strip()
    if not replacement:
        return source, {"applied": False, "reason": "empty_replacement"}
    if section.start_char < 0 or section.end_char <= section.start_char or section.end_char > len(source):
        return source, {"applied": False, "reason": "invalid_section_offsets"}
    old_slice = source[section.start_char:section.end_char]
    if old_slice != section.text:
        located = source.find(section.text)
        if located < 0:
            return source, {"applied": False, "reason": "section_slice_mismatch"}
        return source[:located] + replacement + source[located + len(section.text):], {
            "applied": True,
            "reason": "relocated_section_text",
            "start_char": located,
            "end_char": located + len(section.text),
        }
    return source[:section.start_char] + replacement + source[section.end_char:], {
        "applied": True,
        "start_char": section.start_char,
        "end_char": section.end_char,
    }


def fact_map_integrity(fact_map: FactMap, replacement_text: str) -> dict[str, Any]:
    replacement = str(replacement_text or "")
    failures: list[dict[str, Any]] = []
    for citation in fact_map.citations:
        if citation and citation not in replacement:
            failures.append({"reason": "citation_missing", "value": citation})
    for term in fact_map.protected_terms:
        if term and term not in replacement:
            failures.append({"reason": "protected_term_missing", "value": term})
    return {"passed": not failures, "failures": failures[:12]}


def _score_summary(input_text: str, report: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    metrics = scanner_controlled_metrics(
        report=report,
        goal=goal,
        footprint_risk=_footprint_risk(_ai_footprint_profile(report)),
        ai_score=_badge_ai(report),
        topk_score=_topk(report),
    )
    scores = {
        "ai": _badge_ai(report),
        "topk": _topk(report),
        "external": metrics.get("external_proxy_score"),
        "rank": scanner_controlled_rank(metrics),
        "risky_window_count": metrics.get("risky_window_count"),
        "unsafe_word_ratio": metrics.get("unsafe_word_ratio"),
        "unsafe_cluster_count": _unsafe_cluster_count(goal),
    }
    scores.update(_goal_driver_snapshot(goal))
    return scores


def _stack_summary(
    *,
    baseline: dict[str, Any],
    final: dict[str, Any],
    route_result: dict[str, Any],
    cleanup_result: dict[str, Any] | None,
) -> dict[str, Any]:
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
    return {
        "route_accepted": bool(route_result.get("accepted_candidate")),
        "cleanup_accepted_count": len(cleanup_result.get("accepted") or []) if cleanup_result else 0,
        "scores_before": {key: baseline.get(key) for key in keys},
        "scores_after": {key: final.get(key) for key in keys},
        "deltas": {f"{key}_delta": round(_num(baseline.get(key)) - _num(final.get(key)), 3) for key in keys},
        "all_core_improved": all(
            _num(baseline.get(key)) - _num(final.get(key)) >= 0.0
            for key in (
                "ai",
                "topk",
                "external",
                "rank",
                "topk_calibrated_risk",
                "qualifying_text_ai_density",
                "external_ai_flag_risk",
            )
        ),
    }


def _goal_driver_snapshot(goal: dict[str, Any]) -> dict[str, float]:
    return {
        "topk_calibrated_risk": _goal_driver_value(goal, "topk_calibrated_risk"),
        "qualifying_text_ai_density": _goal_driver_value(goal, "qualifying_text_ai_density"),
        "ai_authorship": _goal_driver_value(goal, "ai_authorship"),
        "external_ai_flag_risk": _goal_driver_value(goal, "external_ai_flag_risk"),
    }


def _add_deltas(scores: dict[str, Any], baseline: dict[str, Any]) -> None:
    for key in (
        "ai",
        "topk",
        "external",
        "rank",
        "unsafe_word_ratio",
        "unsafe_cluster_count",
        "topk_calibrated_risk",
        "qualifying_text_ai_density",
        "ai_authorship",
        "external_ai_flag_risk",
    ):
        scores[f"{key}_delta"] = round(_num(baseline.get(key)) - _num(scores.get(key)), 3)


def _compact_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    scores = row.get("scores") or {}
    return {
        "section_id": row.get("section_id"),
        "variant_id": row.get("variant_id"),
        "word_count": row.get("word_count"),
        "scores": scores,
        "goal": row.get("goal"),
        "apply_status": row.get("apply_status"),
        "source_grounding_passed": bool((row.get("source_grounding") or {}).get("passed")),
        "fact_integrity_passed": bool((row.get("fact_integrity") or {}).get("passed")),
        "text": row.get("text"),
    }


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    scores = row.get("scores") or {}
    return (
        _num(scores.get("external_delta")),
        _num(scores.get("rank_delta")),
        _num(scores.get("topk_calibrated_risk_delta")),
        _num(scores.get("unsafe_word_ratio_delta")),
        _num(scores.get("ai_delta")),
    )


def _is_safe_candidate(row: dict[str, Any]) -> bool:
    scores = row.get("scores") or {}
    if not (row.get("apply_status") or {}).get("applied"):
        return False
    if not (row.get("source_grounding") or {}).get("passed"):
        return False
    if not (row.get("fact_integrity") or {}).get("passed"):
        return False
    external_delta = _num(scores.get("external_delta"))
    rank_delta = _num(scores.get("rank_delta"))
    core_deltas = (
        _num(scores.get("ai_delta")),
        _num(scores.get("topk_delta")),
        _num(scores.get("topk_calibrated_risk_delta")),
        _num(scores.get("qualifying_text_ai_density_delta")),
        _num(scores.get("external_ai_flag_risk_delta")),
        _num(scores.get("unsafe_word_ratio_delta")),
    )
    primary_blocker_delta = max(core_deltas[2:])
    cluster_delta = _num(scores.get("unsafe_cluster_count_delta"))
    regression_tolerance = 0.05
    min_primary_delta = _float_env("DRAFTPROOF_REWRITE_V5_MIN_PRIMARY_DELTA", 0.1, minimum=0.0, maximum=5.0)
    return (
        external_delta >= 0.0
        and rank_delta >= 0.0
        and all(delta >= -regression_tolerance for delta in core_deltas)
        and (primary_blocker_delta >= min_primary_delta or cluster_delta >= 2.0)
    )


def _rank_sections_by_goal(sections: list[SectionUnit], goal: dict[str, Any]) -> list[SectionUnit]:
    density = goal.get("eligible_span_density_gate") if isinstance(goal.get("eligible_span_density_gate"), dict) else {}
    clusters = [row for row in density.get("top_unsafe_clusters") or [] if isinstance(row, dict)]
    if not clusters:
        return sections

    def score(section: SectionUnit) -> tuple[float, int]:
        value = 0.0
        for cluster in clusters:
            start = cluster.get("start_sentence")
            end = cluster.get("end_sentence")
            preview = str(cluster.get("preview") or "")
            if preview and preview[:80] in section.text:
                value += float(cluster.get("risk_score") or 0.0)
            if isinstance(start, int) and isinstance(end, int):
                value += 0.001 * max(0, end - start + 1)
        return value, section.word_count

    return sorted(sections, key=score, reverse=True)


def _select_section(sections: list[SectionUnit], *, section_id: str | None) -> SectionUnit | None:
    if section_id:
        return next((section for section in sections if section.section_id == section_id), None)
    return sections[0] if sections else None


def _paragraph_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    cursor = 0
    for index, part in enumerate(str(text or "").split("\n\n")):
        start = cursor
        end = start + len(part)
        if part.strip():
            blocks.append({"index": index, "start": start, "end": end, "text": part})
        cursor = end + 2
    return blocks


def _is_heading(block: str) -> bool:
    text = str(block or "").strip()
    if not text:
        return False
    if "\n" in text:
        first, rest = text.split("\n", 1)
        return bool(first.strip()) and len(first.split()) <= 12 and bool(rest.strip())
    return len(text.split()) <= 12 and not text.endswith(".")


def _heading_text(block: str) -> str:
    text = str(block or "").strip()
    if "\n" in text:
        return text.split("\n", 1)[0].strip()
    return text


def _section_role(section: SectionUnit) -> str:
    heading = section.heading.casefold()
    ordinal = int(section.metadata.get("ordinal") or 0)
    total = int(section.metadata.get("section_count") or 0)
    if "introduction" in heading or ordinal == 1:
        return "introduction/background framing"
    if "conclusion" in heading or (total > 1 and ordinal == total):
        return "conclusion/reflection"
    return "body section"


def _sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    for char in str(text or ""):
        current.append(char)
        if char in ".?!":
            sentence = "".join(current).strip()
            current = []
            if sentence:
                sentences.append(sentence)
    remainder = "".join(current).strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def _citation_spans(text: str) -> list[str]:
    spans: list[str] = []
    current = ""
    inside = False
    for char in str(text or ""):
        if char == "(":
            inside = True
            current = char
            continue
        if inside:
            current += char
            if char == ")":
                if any(token.isdigit() for token in current):
                    spans.append(current)
                current = ""
                inside = False
    return spans


def _protected_terms(text: str, citations: tuple[str, ...] | list[str]) -> list[str]:
    terms: list[str] = []
    for sentence in _sentences(text):
        for token in sentence.replace("(", " ").replace(")", " ").replace(",", " ").split():
            clean = token.strip(".,;:!?“”\"'")
            if len(clean) >= 3 and (clean.isupper() or any(char.isdigit() for char in clean)):
                terms.append(clean)
    for citation in citations:
        terms.extend(part.strip(" ;,()") for part in citation.split(";") if part.strip())
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(term)
    return unique[:24]


def _has_personal_marker(sentence: str) -> bool:
    lowered = f" {sentence.casefold()} "
    return any(marker in lowered for marker in (" i ", " my ", " me ", " in my ", " to me "))


def _route_labels(sentences: list[str]) -> list[str]:
    labels: list[str] = []
    for sentence in sentences:
        lowered = sentence.casefold()
        if _has_personal_marker(sentence):
            labels.append("personal observation or teaching reflection")
        elif "citation" in lowered or "(" in sentence:
            labels.append("academic support claim")
        elif "this" in lowered[:12] or "however" in lowered[:12]:
            labels.append("transition or summary claim")
        else:
            labels.append("source fact or explanation")
    return labels[:8]


def _better_route_for_section(section: SectionUnit) -> list[str]:
    role = _section_role(section)
    if role == "conclusion/reflection":
        return [
            "what the report's strategies showed",
            "how the source evidence supports the final claim",
            "writer's final principle or implication",
        ]
    if role == "introduction/background framing":
        return [
            "specific context already present in the section",
            "actual difficulty or tension",
            "why the section's central concept matters",
            "supported academic claim",
        ]
    return [
        "source-specific context",
        "practical difficulty or tension",
        "response or implication",
        "supported claim",
    ]


def _writing_issues_for_section(section: SectionUnit, goal: dict[str, Any]) -> list[str]:
    issues = [
        "avoid repeating a neat claim-explain-conclude route",
        "reduce generic summary phrases while preserving source facts",
        "vary sentence route without casual rewriting",
    ]
    density = goal.get("eligible_span_density_gate") if isinstance(goal.get("eligible_span_density_gate"), dict) else {}
    for cluster in density.get("top_unsafe_clusters") or []:
        preview = str((cluster or {}).get("preview") or "")
        if preview[:80] and preview[:80] in section.text:
            issues.append("section contains a remaining unsafe cluster; rebuild that local route from facts")
            break
    return issues


def _parse_variants(raw: str, *, source_words: int) -> tuple[list[RecompositionVariant], dict[str, Any]]:
    payload, diagnostics = parse_json_object(raw, required_keys={"variants"})
    if payload is None:
        return [], diagnostics
    rows = payload.get("variants")
    if not isinstance(rows, list):
        return [], {**diagnostics, "status": "schema_failed", "reason": "variants_not_array"}
    min_words = max(1, round(source_words * 0.80))
    max_words = max(min_words + 1, round(source_words * 1.20))
    variants: list[RecompositionVariant] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or set(row.keys()) != {"variant_id", "text"}:
            rejected.append({"index": index, "reason": "variant_keys_mismatch"})
            continue
        text = str(row.get("text") or "").strip()
        variant_id = str(row.get("variant_id") or "").strip()
        count = word_count(text)
        if not text or not variant_id:
            rejected.append({"index": index, "reason": "empty_variant"})
            continue
        if count < min_words or count > max_words:
            rejected.append({"index": index, "variant_id": variant_id, "reason": "word_count_contract_failed", "word_count": count})
            continue
        variants.append(RecompositionVariant(variant_id=variant_id, text=text, word_count=count))
    return variants, {**diagnostics, "status": "ok" if variants else "schema_failed", "variant_count": len(variants), "rejected": rejected}


def _variants_response_format(variant_count: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rewrite_v5_section_variants",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "variants": {
                        "type": "array",
                        "minItems": variant_count,
                        "maxItems": variant_count,
                        "items": {
                            "type": "object",
                            "properties": {
                                "variant_id": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["variant_id", "text"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["variants"],
                "additionalProperties": False,
            },
        },
    }


def _merge_provider_options(base: Any, required: Any) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(required, dict):
        merged.update(required)
    return merged or None


def _rejection_reason(apply_status: dict[str, Any], grounding: dict[str, Any], fact_integrity: dict[str, Any]) -> str:
    if not apply_status.get("applied"):
        return str(apply_status.get("reason") or "section_apply_failed")
    if not grounding.get("passed"):
        return "source_grounding_failed"
    if not fact_integrity.get("passed"):
        return "fact_integrity_failed"
    return "candidate_rejected"


def _unsafe_cluster_count(goal: dict[str, Any]) -> int:
    density = goal.get("eligible_span_density_gate") if isinstance(goal.get("eligible_span_density_gate"), dict) else {}
    try:
        return int(density.get("unsafe_cluster_count") or 0)
    except Exception:
        return 0


def _goal_driver_value(goal: dict[str, Any], driver: str) -> float:
    gate = goal.get("ai_footprint_gate") if isinstance(goal.get("ai_footprint_gate"), dict) else {}
    after = gate.get("after") if isinstance(gate.get("after"), dict) else {}
    for bucket in ("authorship_footprint", "semantic_footprint", "grounding_footprint", "structural_footprint"):
        values = after.get(bucket) if isinstance(after.get(bucket), dict) else {}
        if driver in values:
            return _num(values.get(driver))
    return _num(after.get(driver))


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
