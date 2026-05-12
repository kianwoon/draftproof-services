from __future__ import annotations

import math
import re

from rewrite_pipeline_core.text_processing.text_utils import _text_word_count


def _human_gain_stage_target(human_score: float | int | None, *, final_target: float = 80.0) -> float:
    """Return the next ladder target for controlled human-gain repair."""
    try:
        human = float(human_score)
    except (TypeError, ValueError):
        human = 0.0
    for target in (60.0, 70.0, float(final_target)):
        if human < target:
            return target
    return float(final_target)

def _word_count_band(text: str, variance: float = 0.25) -> dict:
    count = _text_word_count(text)
    return {
        "source_word_count": count,
        "min_words": max(1, int(math.floor(count * (1.0 - variance)))),
        "max_words": max(1, int(math.ceil(count * (1.0 + variance)))),
        "variance": variance,
    }

def _integrity_driver_rows(raw_json: dict | None, limit: int = 14) -> list[dict]:
    raw_json = raw_json or {}
    rows: list[dict] = []
    seen = set()
    integrity_sources = [
        raw_json.get("integrity_layers"),
        ((raw_json.get("scan_intelligence") or {}).get("integrity_layers") or {}),
        ((raw_json.get("ai_mitigation") or {}).get("integrity_layers") or {}),
    ]
    for integrity in integrity_sources:
        layers = integrity.get("layers") if isinstance(integrity, dict) else {}
        if not isinstance(layers, dict):
            continue
        for layer_key, layer in layers.items():
            if not isinstance(layer, dict):
                continue
            for signal in layer.get("signals") or []:
                if not isinstance(signal, dict):
                    continue
                signal_key = (layer_key, signal.get("key") or signal.get("label"))
                if signal_key in seen:
                    continue
                seen.add(signal_key)
                score = signal.get("score")
                rows.append({
                    "layer": layer_key,
                    "key": signal.get("key"),
                    "label": signal.get("label"),
                    "score": score,
                    "priority": float(score) if isinstance(score, (int, float)) else -1.0,
                })
    rows.sort(key=lambda item: item.get("priority", -1), reverse=True)
    return [
        {k: v for k, v in row.items() if k != "priority" and v is not None}
        for row in rows[:limit]
    ]

def _target_segment_rows(raw_json: dict | None, limit: int = 16) -> list[dict]:
    raw_json = raw_json or {}
    ai_mitigation = raw_json.get("ai_mitigation") or {}
    segments = ai_mitigation.get("target_segments") or []
    rows: list[dict] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        signal = segment.get("primary_signal") or {}
        rows.append({
            "segment_id": segment.get("segment_id"),
            "paragraph_id": segment.get("paragraph_id"),
            "text": segment.get("text"),
            "signal": signal.get("key") or signal.get("title"),
            "score": signal.get("score"),
            "lever": segment.get("lever"),
            "bucket": segment.get("bucket"),
            "action": segment.get("action"),
            "auto_apply": segment.get("auto_apply"),
        })
        if len(rows) >= limit:
            break
    return rows

def _reconstruction_failure_feedback(prior_attempts: list[dict] | None, limit: int = 6) -> list[dict]:
    rows: list[dict] = []
    for item in (prior_attempts or [])[-limit:]:
        if not isinstance(item, dict):
            continue
        gate = item.get("gate") if isinstance(item.get("gate"), dict) else {}
        components = item.get("human_shift_components") or gate.get("human_shift_components") or {}
        row = {
            "strategy": item.get("strategy") or item.get("attempt"),
            "reason": item.get("reason") or gate.get("reason"),
            "human_shift_score": item.get("human_shift_score") or gate.get("human_shift_score"),
            "candidate_human": item.get("candidate_human") or item.get("human_contribution") or gate.get("candidate_human"),
            "human_delta": item.get("human_delta") or gate.get("human_delta"),
            "ai_authorship_delta": item.get("ai_authorship_delta") or gate.get("ai_authorship_delta"),
            "ai_transformation_delta": item.get("ai_transformation_delta") or gate.get("ai_transformation_delta"),
            "failed_components": {
                key: value
                for key, value in components.items()
                if isinstance(value, (int, float)) and value < 0
            },
        }
        rows.append({k: v for k, v in row.items() if v not in (None, {}, [])})
    return rows

def _reconstruction_gate_controls(prior_attempts: list[dict] | None) -> dict:
    """Convert scanner/gate failures into generation controls for the next candidate."""
    feedback = _reconstruction_failure_feedback(prior_attempts, limit=8)
    controls: list[str] = []
    blocker_counts: dict[str, int] = {}

    def add(key: str, instruction: str) -> None:
        blocker_counts[key] = blocker_counts.get(key, 0) + 1
        if instruction not in controls:
            controls.append(instruction)

    for item in feedback:
        reason = str(item.get("reason") or "")
        failed_components = item.get("failed_components") or {}
        candidate_human = item.get("candidate_human") or item.get("human_contribution")
        if isinstance(candidate_human, (int, float)):
            next_stage = _human_gain_stage_target(candidate_human)
            if candidate_human < next_stage:
                add(
                    "human_gain_repair",
                    (
                        f"HUMAN_GAIN_REPAIR: raise Human Contribution toward the next ladder target "
                        f"({int(next_stage)}) by patching only 10-20% of sentences with existing anchors, "
                        "safe author reasoning traces, uneven rhythm, and less balanced paragraph flow."
                    ),
                )
        if "human_contribution_gain" in failed_components or (
            isinstance(item.get("human_delta"), (int, float)) and item.get("human_delta") < 0
        ):
            add(
                "human_contribution_regressed",
                "Do not replace author-owned reasoning with smoother academic explanation; keep or increase first-person operational judgement, process detail, and local constraint language.",
            )
        if "ai_transformation_reduction" in failed_components or (
            isinstance(item.get("ai_transformation_delta"), (int, float))
            and item.get("ai_transformation_delta") < 0
        ):
            add(
                "ai_transformation_regressed",
                "Avoid outline-to-essay expansion, symmetrical paragraph routes, and polished summary cadence; use uneven paragraph jobs with concrete task decisions.",
            )
        if "ai_authorship_reduction" in failed_components or (
            isinstance(item.get("ai_authorship_delta"), (int, float))
            and item.get("ai_authorship_delta") < 0
        ):
            add(
                "authorship_texture_repair",
                "AUTHORSHIP_TEXTURE_REPAIR: do not add more human details yet; reduce statistical smoothness through rhythm variance, less transition cleanliness, asymmetric sentence pacing, and uneven information density without changing meaning.",
            )
        if "grounding_risk_reduction" in failed_components:
            add(
                "grounding_regressed",
                "Preserve every source-to-claim relation and citation role; do not generalise cited claims or move sources into broad unsupported summaries.",
            )
        if "rewrite_smoothness_reduction" in failed_components:
            add(
                "smoothness_regressed",
                "Stop polishing. Prefer direct workshop-language reasoning over balanced academic sentences.",
            )
        if "weighted_severity_penalty" in failed_components or "weighted_severity" in reason:
            add(
                "severity_regressed",
                "Do not introduce new high/medium findings; keep protected anchors and source coverage intact before changing style.",
            )
        if "protected_span_lost" in reason or "number_lost" in reason:
            add(
                "protected_anchor_lost",
                "Copy protected anchors exactly: citations, years, page ranges, unit codes, named institutions, quotes, and reference details must remain present.",
            )
        if "quote_lost" in reason:
            add(
                "quote_lost",
                "Do not omit or paraphrase quoted/source wording; preserve quoted material exactly where it appears in the submitted content.",
            )
        if "candidate_word_count_too_low" in reason:
            add(
                "word_count_low",
                "Add length only through source-licensed reasoning, process explanation, and local constraints from the submitted draft; do not add generic filler.",
            )
        if "candidate_word_count_too_high" in reason:
            add(
                "word_count_high",
                "Compress generic explanation while keeping protected anchors and source relations.",
            )
        if "semantic_drift" in reason:
            add(
                "semantic_drift",
                "Keep the same claim inventory and evidence relationships; restructure route and rhythm without dropping entities, quotes, citations, or source roles.",
            )

    if not controls:
        controls.append(
            "No scored failure yet. Use the scanner baseline directly: raise Human Contribution toward 80 while preserving protected anchors and avoiding AI Authorship regression."
        )

    return {
        "schema_version": "scanner_gate_feedback.v1",
        "purpose": "Use scanner/gate failures as control signals for the next generation attempt.",
        "acceptance_target": {
            "human_contribution_min": 80,
            "human_contribution_ladder": [60, 70, 80],
            "primary_goal": "maximize_human_contribution_after_hard_safety_rejects",
            "ai_authorship_regression_allowed": False,
            "word_count_variance": "±25%",
            "critical_high_review_regression_allowed": False,
        },
        "prior_attempts": feedback,
        "blocker_counts": blocker_counts,
        "next_candidate_controls": controls[:10],
    }

def _generation_context_ledger(brief: dict, blueprint: dict) -> dict:
    """Build generation input from scanner-derived context, not the source prose."""
    paragraph_plans = []
    for plan in (blueprint or {}).get("paragraph_plans") or []:
        if not isinstance(plan, dict):
            continue
        paragraph_plans.append({
            key: value
            for key, value in plan.items()
            if key not in {"source_preview"}
        })
    target_segments = []
    for segment in (brief or {}).get("target_segments") or []:
        if not isinstance(segment, dict):
            continue
        target_segments.append({
            key: value
            for key, value in segment.items()
            if key not in {"text"}
        })
    return {
        "schema_version": "generation_context_ledger.v1",
        "purpose": "Regenerate from scanner-derived meaning, anchors, roles, and signals without using the submitted prose as a scaffold.",
        "claim_inventory": (brief or {}).get("claims") or [],
        "headings": (brief or {}).get("headings") or [],
        "protected_facts": (brief or {}).get("protected_facts") or [],
        "preservation_inventory": (brief or {}).get("preservation_inventory") or {},
        "word_count_band": (brief or {}).get("word_count_band") or {},
        "paragraph_roles": (brief or {}).get("paragraph_roles") or [],
        "paragraph_plans": paragraph_plans,
        "global_driver_targets": (blueprint or {}).get("global_driver_targets") or [],
        "industry_baseline_focus": (blueprint or {}).get("industry_baseline_focus") or {},
        "integrity_targets": (brief or {}).get("integrity_targets") or [],
        "target_segment_signals": target_segments,
        "allowed_existing_additions": (brief or {}).get("allowed_existing_additions") or [],
        "preserve_terms": (brief or {}).get("preserve_terms") or [],
        "do_not_add": (brief or {}).get("do_not_add") or [],
        "human_contribution_contract": (brief or {}).get("human_contribution_contract") or {},
        "reference_entries": (brief or {}).get("reference_entries") or [],
        "generation_handoff": (brief or {}).get("generation_handoff") or {},
    }

def _reference_entries_from_text(text: str, limit: int = 60) -> list[str]:
    """Extract bibliography entries as preservation context, not prose substrate."""
    if not isinstance(text, str) or not text.strip():
        return []
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if re.match(r"^\s*(?:references|reference list|bibliography|works cited)\s*$", line, re.I):
            start = index + 1
            break
    if start is None:
        return []

    entries: list[str] = []
    current = ""
    for raw_line in lines[start:]:
        line = raw_line.strip()
        if not line:
            if current:
                entries.append(current.strip())
                current = ""
            continue
        if re.match(r"^[A-Z][A-Za-z0-9 ,/&-]{2,70}$", line) and not re.search(r"\(\d{4}\)|https?://|doi\.", line, re.I):
            break
        starts_entry = bool(re.search(r"\(\d{4}\)|\(\s*n\.d\.\s*\)|https?://|doi\.", line, re.I))
        if current and starts_entry:
            entries.append(current.strip())
            current = line
        else:
            current = f"{current} {line}".strip() if current else line
        if len(entries) >= limit:
            break
    if current and len(entries) < limit:
        entries.append(current.strip())
    return entries[:limit]

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
    text = re.sub(r"(?<=[a-z0-9)\]])\.(?=[A-Z0-9])", ". ", text)
    text = re.sub(r"(?<=[a-z0-9)\]])\?(?=[A-Z0-9])", "? ", text)
    text = re.sub(r"(?<=[a-z0-9)\]])!(?=[A-Z0-9])", "! ", text)
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


def _clean_section_candidate(output: str, heading: str) -> str:
    if not output:
        return ""
    text = output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"^(?:section|body|draft|answer)\s*:\s*", "", text, flags=re.I).strip()
    if heading:
        text = re.sub(rf"^\s*{re.escape(str(heading).strip())}\s*", "", text, flags=re.I).strip()
    text = re.sub(r"(?im)^\s*(?:references|reference list|bibliography|works cited)\s*$.*", "", text, flags=re.S).strip()
    paragraphs = [" ".join(p.strip().split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "\n\n".join(paragraphs).strip()
