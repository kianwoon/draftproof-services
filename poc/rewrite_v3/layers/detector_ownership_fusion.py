"""Fusion prompt for split detector movement and authorship ownership wins."""

from __future__ import annotations

import json
from typing import Any

from ..document_units import compact_document_inventory, structural_shape_contract


FAMILY = "detector_ownership_fusion"


def _compact_windows(profile: dict[str, Any], *, limit: int = 4) -> list[dict[str, Any]]:
    windows = profile.get("windows") if isinstance(profile.get("windows"), list) else []
    ranked = [
        window for window in windows
        if isinstance(window, dict)
        and str(window.get("label") or "") in {"ai_generated", "moderately_ai_assisted", "lightly_ai_assisted"}
    ]
    ranked.sort(
        key=lambda row: (
            1 if str(row.get("label") or "") == "ai_generated" else 0,
            float(row.get("ai_assistance_score") or 0.0) if isinstance(row.get("ai_assistance_score"), (int, float)) else 0.0,
            int(row.get("word_count") or 0),
        ),
        reverse=True,
    )
    output: list[dict[str, Any]] = []
    for row in ranked[:limit]:
        output.append({
            "window_id": row.get("window_id"),
            "paragraph_id": row.get("paragraph_id"),
            "label": row.get("label"),
            "confidence": row.get("confidence"),
            "word_count": row.get("word_count"),
            "score_components": row.get("score_components") or {},
            "source_excerpt": row.get("source_excerpt") or row.get("source_text") or "",
        })
    return output


def _ownership_rows(trace: dict[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    target_trace = trace.get("target_execution_trace") if isinstance(trace.get("target_execution_trace"), dict) else {}
    rows: list[dict[str, Any]] = []
    for key in ("target_replacements", "accepted_replacements", "scanner_controlled_accepted"):
        for row in target_trace.get(key) or []:
            if not isinstance(row, dict):
                continue
            quality = row.get("candidate_quality") if isinstance(row.get("candidate_quality"), dict) else {}
            if not quality:
                continue
            rows.append({
                "group_id": row.get("group_id"),
                "replacement_excerpt": str(row.get("replacement_text") or "")[:700],
                "ownership_score": quality.get("ownership_score"),
                "ownership_change_count": quality.get("ownership_change_count"),
                "ownership_elements_supported": quality.get("ownership_elements_supported") or [],
            })
            if len(rows) >= limit:
                return rows
    return rows


def build_detector_ownership_fusion_prompt(
    *,
    source_text: str,
    detector_candidate: str,
    ownership_candidate: str,
    detector_trace: dict[str, Any],
    ownership_trace: dict[str, Any],
    family: str,
) -> str:
    detector_profile = detector_trace.get("authorship_window_profile") if isinstance(detector_trace.get("authorship_window_profile"), dict) else {}
    detector_proxy = detector_trace.get("external_proxy") if isinstance(detector_trace.get("external_proxy"), dict) else {}
    ownership_gate = ownership_trace.get("ownership_gate") if isinstance(ownership_trace.get("ownership_gate"), dict) else {}
    payload = {
        "strategy_id": "fuse_detector_and_ownership",
        "source_text": str(source_text or ""),
        "source_structure_contract": structural_shape_contract(source_text),
        "source_document_inventory": compact_document_inventory(source_text),
        "detector_strong_candidate": str(detector_candidate or ""),
        "detector_structure_contract": structural_shape_contract(detector_candidate),
        "ownership_strong_candidate": str(ownership_candidate or ""),
        "ownership_structure_contract": structural_shape_contract(ownership_candidate),
        "detector_feedback": {
            "family": family,
            "candidate_ai": detector_trace.get("candidate_ai"),
            "candidate_topk": detector_trace.get("candidate_topk"),
            "proxy_reasons": detector_proxy.get("reasons") or [],
            "segment_authorship_gate": ((detector_proxy.get("metrics") or {}).get("segment_authorship_gate") if isinstance(detector_proxy.get("metrics"), dict) else {}),
            "failed_windows": _compact_windows(detector_profile),
        },
        "ownership_feedback": {
            "ownership_gate": ownership_gate,
            "ownership_rows": _ownership_rows(ownership_trace),
        },
        "requirements": [
            "Return only JSON with rewritten_document.",
            "Use detector_strong_candidate as the base structure.",
            "Fuse ownership only inside failed_windows or their paragraph-local equivalents.",
            "Preserve detector-sensitive rhythm from detector_strong_candidate unless it directly weakens ownership.",
            "Inject only source-supported author trace, specific context, and real judgment already licensed by source_text or ownership_strong_candidate.",
            "Do not humanize the whole document.",
            "Do not add unsupported facts, sources, names, dates, headings, bullets, markdown, labels, or commentary.",
            "Preserve the source meaning, order, citations, protected anchors, and paragraph boundaries.",
            "The rewritten_document must match source_structure_contract exactly for block_count, blank_line_boundary_count, heading_like_line_count, and heading_like_lines.",
            "Keep one output block for each source_document_inventory unit in the same order; do not merge, split, drop, or reorder units.",
            "If a unit does not need fusion, keep the detector_strong_candidate wording for that unit as a separate block.",
        ],
        "response_schema": {
            "rewritten_document": "full fused document only",
            "structure_check": {
                "block_count": "must equal source_structure_contract.block_count",
                "blank_line_boundary_count": "must equal source_structure_contract.blank_line_boundary_count",
            },
            "fused_windows": [
                {
                    "window_id": "w001",
                    "ownership_elements_used": ["author_trace"],
                    "detector_structure_preserved": True,
                }
            ],
        },
    }
    return (
        "Fuse split V3 successes.\n"
        "The detector-strong candidate moved detector metrics but lacked ownership in hot windows.\n"
        "The ownership-strong candidate added author trace and judgment but did not move detector metrics enough.\n"
        "Produce one fused candidate, not a fresh rewrite.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _strip_json_fence(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_fused_document_with_diagnostics(raw: str) -> tuple[str, dict[str, Any]]:
    text = _strip_json_fence(raw)
    diagnostics: dict[str, Any] = {
        "raw_chars": len(str(raw or "")),
        "raw_preview": str(raw or "")[:1200],
        "parse_status": "empty" if not text else "pending",
        "top_level_keys": [],
        "extraction_key": "",
        "extracted_chars": 0,
    }
    if not text:
        diagnostics["failure"] = "empty_response_content"
        return "", diagnostics
    if text and not text.startswith("{") and not text.startswith("["):
        diagnostics.update({
            "parse_status": "plain_text",
            "extraction_key": "plain_text",
            "extracted_chars": len(text),
        })
        return text, diagnostics
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        diagnostics.update({
            "parse_status": "invalid_json",
            "json_error": str(exc),
            "failure": "invalid_json",
        })
        return "", diagnostics
    if not isinstance(payload, dict):
        diagnostics.update({
            "parse_status": "wrong_json_shape",
            "json_type": type(payload).__name__,
            "failure": "json_top_level_not_object",
        })
        return "", diagnostics
    diagnostics.update({
        "parse_status": "ok",
        "top_level_keys": list(payload.keys()),
    })
    for key in ("rewritten_document", "fused_document", "rewritten_text", "document", "text", "candidate_text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            output = value.strip()
            diagnostics.update({
                "extraction_key": key,
                "extracted_chars": len(output),
            })
            return output, diagnostics
    diagnostics["failure"] = "missing_nonempty_document_field"
    return "", diagnostics


def extract_fused_document(raw: str) -> str:
    document, _diagnostics = extract_fused_document_with_diagnostics(raw)
    return document
