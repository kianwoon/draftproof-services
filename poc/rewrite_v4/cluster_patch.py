"""Bounded unsafe-cluster repair helpers for V4 experiments."""

from __future__ import annotations

import json
import os
from typing import Any

from llm.gateway import LLMGateway
from rewrite_v2.structured_output import structured_json_request_options
from rewrite_v3.document_units import word_count

from .models import CandidateVariant, ClusterRepairUnit
from .validation import parse_generator_variants


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


def build_cluster_repair_units(
    *,
    text: str,
    report: dict[str, Any],
    goal: dict[str, Any],
    limit: int = 4,
    context_chars: int = 260,
) -> list[ClusterRepairUnit]:
    density = goal.get("eligible_span_density_gate") if isinstance(goal.get("eligible_span_density_gate"), dict) else {}
    clusters = [cluster for cluster in density.get("top_unsafe_clusters") or [] if isinstance(cluster, dict)]
    sentence_targets = [
        row for row in density.get("top_sentence_targets") or []
        if isinstance(row, dict)
    ]
    sentence_rows = _sentence_rows(report)
    units: list[ClusterRepairUnit] = []
    for index, cluster in enumerate(clusters[:max(1, int(limit or 1))], start=1):
        unit = _cluster_unit_from_sentence_rows(
            text=str(text or ""),
            cluster=cluster,
            sentence_targets=sentence_targets,
            sentence_rows=sentence_rows,
            cluster_id=f"cluster_{index:03d}",
            context_chars=context_chars,
        )
        if unit is not None:
            units.append(unit)
    return units


def apply_cluster_variant(text: str, unit: ClusterRepairUnit, replacement_text: str) -> tuple[str, dict[str, Any]]:
    source = str(text or "")
    replacement = str(replacement_text or "").strip()
    if not replacement:
        return source, {"applied": False, "reason": "empty_replacement"}
    if replacement == unit.text.strip():
        return source, {"applied": False, "reason": "unchanged_replacement"}
    if unit.start_char < 0 or unit.end_char <= unit.start_char or unit.end_char > len(source):
        return source, {"applied": False, "reason": "invalid_cluster_offsets"}
    old_slice = source[unit.start_char:unit.end_char]
    if old_slice != unit.text:
        return source, {
            "applied": False,
            "reason": "cluster_slice_mismatch",
            "expected_preview": unit.text[:220],
            "actual_preview": old_slice[:220],
        }
    candidate = source[:unit.start_char] + replacement + source[unit.end_char:]
    return candidate, {
        "applied": True,
        "cluster_id": unit.cluster_id,
        "start_char": unit.start_char,
        "end_char": unit.end_char,
        "old_word_count": word_count(unit.text),
        "new_word_count": word_count(replacement),
    }


def build_cluster_generator_prompt(
    *,
    unit: ClusterRepairUnit,
    variant_count: int = 2,
    mode: str = "bounded_cluster_texture_repair",
) -> str:
    variants = max(1, min(3, int(variant_count or 1)))
    residual = str(mode or "").strip() == "residual_cluster_splitter"
    payload = {
        "task": "residual_cluster_splitter" if residual else "bounded_cluster_texture_repair",
        "cluster_id": unit.cluster_id,
        "repair_unit": {
            "cluster_text": unit.text,
            "before_context": unit.before_context,
            "after_context": unit.after_context,
            "sentence_count": unit.sentence_count,
            "word_count": unit.word_count,
            "risk_shape": {
                "generic_hits": unit.metadata.get("generic_hits"),
                "transition_count": unit.metadata.get("transition_count"),
                "risk_score": unit.risk_score,
            },
        },
        "repair_goal": [
            (
                "Treat this as a remaining residual pocket after earlier safe repairs; make the smallest route change that weakens predictable wording."
                if residual else
                "Repair the route across these adjacent sentences."
            ),
            (
                "Target common-word chains, generic summary phrasing, and one-sentence over-completion without adding new content."
                if residual else
                "Break repeated openings, overly tidy cause-effect movement, and generic summary rhythm."
            ),
            "Keep the same facts, people, sequence, citations, and source viewpoint.",
            "Return only a replacement for cluster_text, not the full document.",
        ],
        "allowed_moves": [
            "split one over-complete sentence into two natural sentences when meaning is preserved" if residual else "",
            "combine two adjacent sentences when the same claim is repeated",
            "move a short reason closer to the claim it explains",
            "replace a formulaic transition with source-near wording",
            "vary sentence length without making the prose casual",
            "remove repeated wording when the meaning remains represented",
        ],
        "forbidden": [
            "new facts",
            "new examples",
            "new names",
            "new dates",
            "new numbers",
            "new citations",
            "headings",
            "bullets",
            "markdown",
            "HTML",
            "first-person conversion unless already present in cluster_text",
            "grammar damage or random errors",
            "full paragraph rewrite",
        ],
        "constraints": [
            "Stay within 70% to 130% of the cluster_text word count.",
            "Keep one contiguous prose cluster.",
            "Preserve protected citation-like strings exactly if present.",
            "Do not polish every sentence into the same rhythm.",
            "Prefer reducing predictable common phrasing over making the prose smoother." if residual else "",
            "Do not convert an already specific sentence into a broader summary." if residual else "",
            f"Return exactly {variants} {'variant' if variants == 1 else 'variants'}.",
            "Return exactly the allowed keys for each variant: variant_id, text.",
        ],
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, variants + 1)
            ]
        },
    }
    payload["allowed_moves"] = [item for item in payload["allowed_moves"] if item]
    payload["constraints"] = [item for item in payload["constraints"] if item]
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_cluster_variants(
    *,
    unit: ClusterRepairUnit,
    gateway: LLMGateway,
    variant_count: int = 2,
    mode: str = "bounded_cluster_texture_repair",
) -> tuple[list[CandidateVariant], dict[str, Any], str, str]:
    prompt = build_cluster_generator_prompt(unit=unit, variant_count=variant_count, mode=mode)
    response_format = _variants_response_format(max(1, min(3, int(variant_count or 1))))
    structured_options = structured_json_request_options(getattr(gateway, "model", None), response_format)
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured_options.get("provider"))
    max_tokens = _int_env("DRAFTPROOF_REWRITE_V4_CLUSTER_MAX_TOKENS", 3000, minimum=800, maximum=8000)
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a variants array.",
        response_format=structured_options.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_REWRITE_V4_CLUSTER_TEMPERATURE", 0.42, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_REWRITE_V4_CLUSTER_TOP_P", 0.86, minimum=0.1, maximum=1.0),
        max_tokens=max_tokens,
    )
    raw_completion = response.raw_content or response.content
    min_words = max(1, round(unit.word_count * 0.7))
    max_words = max(min_words + 1, round(unit.word_count * 1.3))
    variants, diagnostics = parse_generator_variants(
        raw_completion,
        min_words=min_words,
        max_words=max_words,
        source_text=unit.text,
    )
    return variants, {
        **diagnostics,
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "max_tokens": max_tokens,
        "structured_output_mode": structured_options.get("structured_output_mode"),
    }, prompt, raw_completion


def _sentence_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    sentence_map = report.get("sentence_map") if isinstance(report, dict) else None
    if not isinstance(sentence_map, dict):
        return []
    rows: list[dict[str, Any]] = []
    for sentence_id, payload in sentence_map.items():
        if not isinstance(payload, dict):
            continue
        start = payload.get("start_char")
        end = payload.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            continue
        rows.append({
            "sentence_id": str(sentence_id),
            "start_char": start,
            "end_char": end,
            "text": str(payload.get("text") or ""),
            "paragraph_id": payload.get("paragraph_id"),
        })
    rows.sort(key=lambda row: (int(row["start_char"]), int(row["end_char"])))
    return rows


def _located_sentence_rows(text: str, sentence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    located: list[dict[str, Any]] = []
    cursor = 0
    for row in sentence_rows:
        sentence_text = str(row.get("text") or "")
        if not sentence_text:
            continue
        found = text.find(sentence_text, cursor)
        if found < 0:
            hint = int(row.get("start_char") or 0)
            window_start = max(0, hint - 120)
            found = text.find(sentence_text, window_start)
        if found < 0:
            located.append({**row, "located": False})
            continue
        end = found + len(sentence_text)
        located.append({
            **row,
            "start_char": found,
            "end_char": end,
            "located": True,
        })
        cursor = end
    return located


def _cluster_unit_from_sentence_rows(
    *,
    text: str,
    cluster: dict[str, Any],
    sentence_targets: list[dict[str, Any]],
    sentence_rows: list[dict[str, Any]],
    cluster_id: str,
    context_chars: int,
) -> ClusterRepairUnit | None:
    start = cluster.get("start_sentence")
    end = cluster.get("end_sentence")
    if not isinstance(start, int) or not isinstance(end, int) or start > end:
        return None
    sentence_rows = _located_sentence_rows(text, sentence_rows)
    if start < 0 or end >= len(sentence_rows):
        return None
    selected = sentence_rows[start:end + 1]
    if not selected or not all(row.get("located") for row in selected):
        return None
    first = sentence_rows[start]
    last = sentence_rows[end]
    start_char = int(first["start_char"])
    end_char = int(last["end_char"])
    if start_char < 0 or end_char <= start_char or end_char > len(text):
        return None
    cluster_text = text[start_char:end_char]
    if not cluster_text.strip():
        return None
    target_rows = [
        {
            "sentence_index": row.get("sentence_index"),
            "sentence_id": row.get("sentence_id"),
            "word_count": row.get("word_count"),
            "top10_ratio": row.get("top10_ratio"),
            "top50_ratio": row.get("top50_ratio"),
            "predictability_risk": row.get("predictability_risk"),
            "risk_score": row.get("risk_score"),
            "generic_hits": row.get("generic_hits"),
            "preview": str(row.get("preview") or "")[:260],
        }
        for row in sentence_targets
        if isinstance(row.get("sentence_index"), int)
        and start <= int(row.get("sentence_index")) <= end
    ]
    return ClusterRepairUnit(
        cluster_id=cluster_id,
        start_sentence=start,
        end_sentence=end,
        start_char=start_char,
        end_char=end_char,
        text=cluster_text,
        before_context=text[max(0, start_char - context_chars):start_char],
        after_context=text[end_char:min(len(text), end_char + context_chars)],
        sentence_count=int(cluster.get("sentence_count") or (end - start + 1)),
        word_count=word_count(cluster_text),
        risk_score=float(cluster.get("risk_score") or 0.0),
        metadata={
            "gate_cluster": cluster,
            "sentence_ids": [row.get("sentence_id") for row in sentence_rows[start:end + 1]],
            "target_sentence_metrics": target_rows,
            "generic_hits": cluster.get("generic_hits"),
            "transition_count": cluster.get("transition_count"),
        },
    )


def _variants_response_format(variant_count: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rewrite_v4_cluster_variants",
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
                    },
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
