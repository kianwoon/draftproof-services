"""Rewrite debug-log building, JSON bounding/truncation, and debug compaction.

Extracted from tasks.py. This whole cluster is reachable only via the
``run_rewrite`` task -> ``_build_rewrite_debug_log``.
"""
from __future__ import annotations

import os
import json
import hashlib

from report.contribution import contribution_pair_int
from .rewrite_scan_compaction import compact_rewrite_scan_summary

REWRITE_DEBUG_EXPORT_VERSION = "rewrite_controller_debug_passthrough_v3"


def _bounded_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


MAX_REWRITE_DEBUG_LOG_BYTES = _bounded_int_env(
    "DRAFTPROOF_REWRITE_DEBUG_LOG_MAX_BYTES",
    1_000_000,
    minimum=100_000,
    maximum=5_000_000,
)

MAX_REWRITE_JSON_BYTES = _bounded_int_env(
    "DRAFTPROOF_REWRITE_JSON_MAX_BYTES",
    1_500_000,
    minimum=250_000,
    maximum=5_000_000,
)

# tasks.py lives alongside this module; reference it explicitly so the
# fingerprint still reports the worker entrypoint module after the split.
_TASKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.py")


def _runtime_code_fingerprint() -> dict:
    """Return non-secret evidence of the worker code actually imported."""
    fingerprint = {
        "debug_export_version": REWRITE_DEBUG_EXPORT_VERSION,
        "runtime_code_sha_env": os.environ.get("DRAFTPROOF_RUNTIME_CODE_SHA"),
        "worker_tasks_file": os.path.abspath(_TASKS_FILE),
    }

    for key, path in (
        ("runtime_code_sha_file", "/app/runtime_code_sha"),
        ("worker_runtime_code_sha_file", "/app/worker/app/.runtime_git_sha"),
        ("poc_runtime_code_sha_file", "/app/poc/.runtime_git_sha"),
    ):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                fingerprint[key] = handle.read().strip() or None
        except OSError:
            fingerprint[key] = None

    try:
        stat = os.stat(_TASKS_FILE)
        fingerprint["worker_tasks_size"] = stat.st_size
        fingerprint["worker_tasks_mtime"] = int(stat.st_mtime)
        with open(_TASKS_FILE, "rb") as handle:
            fingerprint["worker_tasks_sha256_12"] = hashlib.sha256(
                handle.read()
            ).hexdigest()[:12]
    except OSError as exc:
        fingerprint["worker_tasks_fingerprint_error"] = str(exc)

    return fingerprint


def _pct_score(value) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if abs(number) <= 1:
        number *= 100
    return max(0, min(100, int(round(number))))


def _integrity_risk_label(score: int) -> str:
    if score >= 65:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def _debug_grounding_quality_score(writing_components: dict | None) -> int:
    components = writing_components or {}
    weighted = (
        _pct_score(components.get("source_grounding_risk")) * 0.30
        + _pct_score(components.get("citation_weakness_risk")) * 0.25
        + _pct_score(components.get("unsupported_claim_risk")) * 0.20
        + _pct_score(components.get("broad_claim_risk")) * 0.15
        + _pct_score(components.get("lived_detail_risk")) * 0.10
    )
    return int(round(weighted))


def _debug_integrity_layers(report_json: dict | None, badge: dict | None) -> dict | None:
    if not isinstance(report_json, dict):
        return None
    existing = (
        report_json.get("integrity_layers")
        or ((report_json.get("scan_intelligence") or {}).get("integrity_layers"))
    )
    if existing:
        return existing
    badge = badge or report_json.get("ai_risk_badge") or {}
    if not isinstance(badge, dict):
        return None
    ai_score = _pct_score(badge.get("ai_likelihood_score"))
    writing_components = badge.get("writing_components") or {}
    grounding_score = _debug_grounding_quality_score(writing_components)
    transformation = badge.get("transformation_classification") or {}
    contribution = (
        (((report_json.get("scan_intelligence") or {}).get("transformation") or {}).get("contribution"))
        or {}
    )
    ai_transformation = _pct_score(
        contribution.get("ai_transformation_ratio")
        if contribution.get("ai_transformation_ratio") is not None
        else ((transformation.get("features") or {}).get("calibrated_ai_risk"))
    )
    human, ai_transformation = contribution_pair_int(
        contribution.get("human_contribution_ratio")
        if contribution.get("human_contribution_ratio") is not None
        else None,
        ai_transformation,
    )
    human = int(human or 0)
    ai_transformation = int(ai_transformation or 0)
    ai_band = "High AI" if ai_score >= 50 else "Low AI"
    grounding_band = "Weakly grounded" if grounding_score >= 50 else "Well grounded"
    return {
        "schema_version": "integrity_layers.v1",
        "policy": {
            "grounding_is_not_ai_authorship": True,
            "summary": "Grounding weakness is reported as writing-integrity risk, not direct evidence of AI authorship.",
            "backfilled_for_debug": True,
        },
        "layers": {
            "ai_authorship_risk": {
                "score": ai_score,
                "tier": badge.get("tier"),
                "label": _integrity_risk_label(ai_score),
                "excludes": [
                    "source_grounding_risk",
                    "citation_weakness_risk",
                    "unsupported_claim_risk",
                ],
            },
            "ai_transformation_risk": {
                "score": ai_transformation,
                "label": _integrity_risk_label(ai_transformation),
                "classification": {
                    "code": transformation.get("code"),
                    "label": transformation.get("label"),
                    "confidence": transformation.get("confidence"),
                },
            },
            "grounding_quality_risk": {
                "score": grounding_score,
                "label": _integrity_risk_label(grounding_score),
            },
            "human_contribution_signal": {
                "score": human,
                "label": "strong" if human >= 65 else "mixed" if human >= 40 else "limited",
            },
        },
        "combined_interpretation": {
            "label": f"{ai_band} / {grounding_band}",
        },
    }


def _truncate_debug_value(value, limit: int = 320):
    if isinstance(value, str):
        text = " ".join(value.split())
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
    return value


def _bounded_json_debug_log(log_data: dict, *, max_bytes: int = MAX_REWRITE_DEBUG_LOG_BYTES) -> str:
    text = json.dumps(log_data, indent=2, ensure_ascii=False, default=str)
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    slim = {
        "debug_export_version": log_data.get("debug_export_version"),
        "debug_export_source": log_data.get("debug_export_source"),
        "runtime_code_sha": log_data.get("runtime_code_sha"),
        "runtime_code_fingerprint": log_data.get("runtime_code_fingerprint"),
        "rewrite_id": log_data.get("rewrite_id"),
        "scan_id": log_data.get("scan_id"),
        "status": log_data.get("status"),
        "elapsed": log_data.get("elapsed"),
        "pipeline_status": log_data.get("pipeline_status"),
        "pipeline_elapsed": log_data.get("pipeline_elapsed"),
        "debug_truncated": True,
        "debug_truncation_reason": f"debug log exceeded {max_bytes} bytes",
        "rewrite_summary": _compact_debug_mapping(log_data.get("rewrite_summary"), max_items=48),
        "input_scan": _compact_debug_mapping(log_data.get("input_scan"), max_items=12),
        "loop_history_count": len(log_data.get("loop_history") or []) if isinstance(log_data.get("loop_history"), list) else None,
        "sentence_comparison_count": log_data.get("sentence_comparison_count"),
        "sentence_comparison_changes": log_data.get("sentence_comparison_changes"),
    }
    text = json.dumps(slim, indent=2, ensure_ascii=False, default=str)
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[: max_bytes - 80].decode("utf-8", errors="ignore").rstrip() + "\n... truncated\n"


def _bounded_rewrite_json_payload(payload: dict, *, max_bytes: int = MAX_REWRITE_JSON_BYTES) -> dict:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) <= max_bytes:
        return payload

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    compact_summary_keys = (
        "rewrite_pipeline_version",
        "rewrite_engine_mode",
        "outcome",
        "public_status",
        "public_candidate_warning",
        "best_candidate_external_review_required",
        "best_candidate_author_review_required",
        "strict_safe_band_achieved",
        "kpi_finalization_status",
        "author_review_cards",
        "rewrite_goal_status",
        "strict_goal_status",
        "reference_ai",
        "required_ai_drop",
        "target_ai_score",
        "rewrite_effective_config",
        "v4_scores",
        "v5_scores",
        "detect_scores",
        "original_risk",
        "final_risk",
        "external_detector_estimate",
        "partial_rewrite_preserved",
        "partial_rewrite_preservation_reason",
        "converged",
        "convergence_reason",
        "candidate_generation_status",
        "paragraph_obligation_hard_stop",
        "candidate_ledger",
        "candidate_trace",
        "candidate_loop_trace",
        "selected_candidate",
        "stage_timings",
        "detect_scan_original_saved",
        "detect_scan_original",
        "detect_scan_rewritten",
        "no_text_change",
        "rollback_applied",
        "baseline_rescan_delta",
        "saved_user_visible_scores",
        "original_scores",
        "attempted_scores",
        "final_scores",
        "predictability_highlights",
        "bracket_grounding_spans",
        "bracket_grounding_audit",
        "v6_pass_trace",
        "register_coaching",
        "predictability_showcase",
    )
    compact_summary = {}
    for key in compact_summary_keys:
        if key not in summary:
            continue
        if key == "candidate_ledger":
            compact_summary[key] = _compact_rewrite_candidate_ledger(summary.get(key))
        elif key == "candidate_generation_status":
            compact_summary[key] = _compact_candidate_generation_status(summary.get(key))
        elif key in {"detect_scan_original_saved", "detect_scan_original", "detect_scan_rewritten"}:
            compact_summary[key] = compact_rewrite_scan_summary(summary.get(key))
        elif key in {"predictability_highlights", "bracket_grounding_spans", "register_coaching", "predictability_showcase"}:
            # Exact [start, end] char spans for the rewritten-doc highlight / bracket-grounding colour,
            # the small register-coaching payload (<=4 offenders + 1 contrast), and the worked teaching
            # examples (showcase, capped at _max_sentences). Copy verbatim: the generic compactor
            # truncates inner lists to 8 / strings to 320 chars, which would corrupt span offsets and
            # clip the coaching/example sentence text. All are already small/capped.
            compact_summary[key] = summary.get(key)
        elif key == "v6_pass_trace":
            # Per-stage trace (incl. the bracket_grounding last stage). Without this in the allowlist a
            # >max_bytes payload (e.g. the 2-lane direct-rewrite traces) drops it entirely, so we cannot
            # verify which stage shipped or diagnose monoculture. Compact each row via the existing
            # row helper and cap to stay bounded.
            trace = summary.get(key)
            if isinstance(trace, list):
                compact_summary[key] = [_compact_v6_pass_trace_row(row) for row in trace[:120]]
                if len(trace) > 120:
                    compact_summary[f"{key}_omitted"] = len(trace) - 120
            else:
                compact_summary[key] = _compact_debug_value(trace)
        else:
            compact_summary[key] = _compact_debug_value(summary.get(key))
    compact_payload = {
        "status": payload.get("status"),
        "elapsed": payload.get("elapsed"),
        "original_text": payload.get("original_text"),
        "final_text": payload.get("final_text"),
        "converged": payload.get("converged"),
        "convergence_reason": payload.get("convergence_reason"),
        "passes": payload.get("passes"),
        "summary": compact_summary,
        "sentence_comparison": (
            payload.get("sentence_comparison")[:200]
            if isinstance(payload.get("sentence_comparison"), list)
            else payload.get("sentence_comparison")
        ),
        "effective_rewrite_plan": _compact_debug_value(payload.get("effective_rewrite_plan")),
        "billing_decision": payload.get("billing_decision"),
        "rewrite_json_truncated": True,
        "rewrite_json_truncation_reason": f"rewrite.json exceeded {max_bytes} bytes before upload",
    }
    encoded = json.dumps(compact_payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) <= max_bytes:
        return compact_payload

    compact_payload["sentence_comparison"] = _sentence_comparison_debug_preview(
        payload.get("sentence_comparison") if isinstance(payload.get("sentence_comparison"), list) else []
    )
    encoded = json.dumps(compact_payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) <= max_bytes:
        return compact_payload

    compact_payload["original_text"] = _truncate_debug_value(payload.get("original_text"), 4000)
    compact_payload["final_text"] = _truncate_debug_value(payload.get("final_text"), 4000)
    encoded = json.dumps(compact_payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) <= max_bytes:
        return compact_payload

    return {
        "status": payload.get("status"),
        "elapsed": payload.get("elapsed"),
        "original_text": _truncate_debug_value(payload.get("original_text"), 1000),
        "final_text": _truncate_debug_value(payload.get("final_text"), 1000),
        "converged": payload.get("converged"),
        "convergence_reason": payload.get("convergence_reason"),
        "passes": payload.get("passes"),
        "summary": {
            key: compact_summary.get(key)
            for key in (
                "public_status",
                "rewrite_goal_status",
                "candidate_generation_status",
                "paragraph_obligation_hard_stop",
                "no_text_change_reason",
                "candidate_ledger",
                "v4_scores",
                "detect_scores",
                "detect_scan_original",
                "detect_scan_rewritten",
            )
            if key in compact_summary
        },
        "sentence_comparison": _sentence_comparison_debug_preview(
            payload.get("sentence_comparison") if isinstance(payload.get("sentence_comparison"), list) else []
        ),
        "billing_decision": payload.get("billing_decision"),
        "rewrite_json_truncated": True,
        "rewrite_json_truncation_reason": f"rewrite.json exceeded {max_bytes} bytes before upload",
    }


def _compact_debug_mapping(value, *, max_items: int = 20):
    if not isinstance(value, dict):
        return _truncate_debug_value(value, 500)
    compact = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= max_items:
            compact["omitted_keys"] = len(value) - max_items
            break
        compact[key] = _compact_debug_value(item)
    return compact


def _compact_debug_value(value):
    if isinstance(value, dict):
        return _compact_debug_mapping(value, max_items=12)
    if isinstance(value, list):
        return [_compact_debug_value(item) for item in value[:8]] + ([{"omitted": len(value) - 8}] if len(value) > 8 else [])
    return _truncate_debug_value(value, 500)


def _compact_rewrite_candidate_ledger(value, *, limit: int = 5, max_text_chars: int = 80_000):
    if not isinstance(value, list):
        return []
    compact = []
    for entry in value[: max(1, int(limit or 1))]:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or entry.get("candidate_text") or entry.get("rewritten_document") or "").strip()
        compact.append({
            "schema_version": entry.get("schema_version") or "rewrite_candidate_ledger.v1",
            "rank": entry.get("rank"),
            "source": entry.get("source"),
            "section_id": entry.get("section_id"),
            "variant_id": entry.get("variant_id"),
            "label": entry.get("label"),
            "word_count": entry.get("word_count"),
            "scores": entry.get("scores") if isinstance(entry.get("scores"), dict) else {},
            "goal": entry.get("goal") if isinstance(entry.get("goal"), dict) else {},
            "text": _truncate_debug_value(text, max_text_chars),
        })
    return compact


def _compact_candidate_generation_status(value):
    if not isinstance(value, dict):
        return _compact_debug_value(value)
    out = {
        "accepted_count": value.get("accepted_count"),
        "remaining_findings": value.get("remaining_findings"),
        "remaining_findings_by_paragraph": value.get("remaining_findings_by_paragraph"),
        "stop_reason": value.get("stop_reason"),
        "reason": value.get("reason"),
        "blocked_findings": value.get("blocked_findings"),
        "stage_timings": value.get("stage_timings"),
    }
    trace = value.get("pass_trace")
    if isinstance(trace, list):
        out["pass_trace_count"] = len(trace)
        out["pass_trace"] = [_compact_v6_pass_trace_row(row) for row in trace[:120]]
        if len(trace) > 120:
            out["pass_trace_omitted"] = len(trace) - 120
    return {key: val for key, val in out.items() if val not in (None, [], {})}


def _compact_v6_pass_trace_row(row):
    if not isinstance(row, dict):
        return _compact_debug_value(row)
    keys = (
        "pass_index",
        "status",
        "target_paragraph_id",
        "excluded_paragraph_ids",
        "selected_variant_id",
        "selected_source",
        "before_finding_count",
        "after_finding_count",
        "before_mean_sentence_shape_risk",
        "after_mean_sentence_shape_risk",
        "residual_followup",
        "residual_index",
    )
    compact = {key: row.get(key) for key in keys if key in row}
    diagnostics = row.get("candidate_diagnostics")
    if isinstance(diagnostics, list):
        compact["candidate_diagnostics"] = [
            _compact_debug_mapping(item, max_items=10) if isinstance(item, dict) else _compact_debug_value(item)
            for item in diagnostics[:4]
        ]
        if len(diagnostics) > 4:
            compact["candidate_diagnostics_omitted"] = len(diagnostics) - 4
    for key in ("before_findings_by_paragraph", "after_findings_by_paragraph"):
        if isinstance(row.get(key), dict):
            compact[key] = row.get(key)
    return compact


def _compact_debug_list(items, limit: int = 10):
    if not isinstance(items, list):
        return items
    compacted = items[:limit]
    if len(items) > limit:
        compacted = compacted + [{"omitted": len(items) - limit}]
    return compacted


def _compact_rewrite_context(context) -> dict:
    if not isinstance(context, dict):
        return {}

    compact = {}
    for key in (
        "scope",
        "sentence_id",
        "paragraph_id",
        "previous_sentence",
        "next_sentence",
        "paragraph_excerpt",
        "signal_instruction",
        "safe_addition_types",
    ):
        if key in context:
            compact[key] = _truncate_debug_value(context.get(key))

    anchors = context.get("domain_anchors")
    if isinstance(anchors, list):
        compact["domain_anchors"] = _compact_debug_list(anchors, 16)

    spans = context.get("predictable_token_spans")
    if isinstance(spans, list):
        compact["predictable_token_spans"] = _compact_debug_list(
            [_truncate_debug_value(s, 120) for s in spans if s],
            16,
        )

    tokens = context.get("problem_tokens")
    if isinstance(tokens, list):
        compact_tokens = []
        for token in tokens[:20]:
            if isinstance(token, dict):
                compact_tokens.append({
                    "token": token.get("token"),
                    "rank": token.get("rank"),
                    "probability": token.get("probability"),
                    "surprisal": token.get("surprisal"),
                })
            else:
                compact_tokens.append(_truncate_debug_value(token, 80))
        if len(tokens) > 20:
            compact_tokens.append({"omitted": len(tokens) - 20})
        compact["problem_tokens"] = compact_tokens

    metrics = context.get("predictability_metrics")
    if isinstance(metrics, dict):
        compact["predictability_metrics"] = metrics

    return compact


def _flatten_report_findings(report_json: dict) -> dict:
    by_id = {}
    for tier_name, findings in (report_json.get("findings") or {}).items():
        if not isinstance(findings, list):
            continue
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            finding_id = finding.get("finding_id")
            if not finding_id:
                continue
            snapshot = {
                "finding_id": finding_id,
                "tier": tier_name,
                "title": finding.get("title"),
                "scanner": finding.get("scanner") or finding.get("category"),
                "adjusted_risk": finding.get("adjusted_risk"),
                "actionability": finding.get("actionability"),
                "sentence_id": finding.get("sentence_id"),
                "paragraph_id": finding.get("paragraph_id"),
                "score": finding.get("score"),
                "recommendation": _truncate_debug_value(finding.get("recommendation")),
            }
            evidence = finding.get("evidence")
            if isinstance(evidence, dict):
                snapshot["evidence"] = {
                    "summary": _truncate_debug_value(evidence.get("summary")),
                    "metrics": evidence.get("metrics"),
                    "affected_span": _truncate_debug_value(evidence.get("affected_span")),
                }
            else:
                snapshot["evidence"] = _truncate_debug_value(evidence)
            snapshot["rewrite_context"] = _compact_rewrite_context(
                finding.get("rewrite_context")
            )
            by_id[finding_id] = snapshot
    return by_id


def _extract_rewrite_scan_summary(report_dict: dict) -> dict:
    badge = report_dict.get("ai_risk_badge") or {}
    findings = report_dict.get("findings", {})
    return {
        "ai_score": report_dict.get("ai_score") or badge.get("ai_likelihood_score"),
        "writing_score": report_dict.get("writing_score") or badge.get("writing_quality_score"),
        "ai_risk_badge": badge,
        "scan_intelligence": report_dict.get("scan_intelligence") or {},
        "integrity_layers": report_dict.get("integrity_layers") or {},
        "overall_tier": report_dict.get("overall_tier", "?"),
        "findings": {
            tier: [
                {
                    "finding_id": finding.get("finding_id"),
                    "title": finding.get("title"),
                    "category": finding.get("category"),
                }
                for finding in findings.get(tier, [])
                if isinstance(finding, dict)
            ]
            for tier in ("critical", "high", "medium", "low")
        },
    }


def _target_contexts(plan_items, findings_by_id: dict) -> list:
    contexts = []
    for item in plan_items or []:
        if not isinstance(item, dict):
            continue
        finding_id = item.get("finding_id")
        finding_snapshot = findings_by_id.get(finding_id, {})
        contexts.append({
            "plan_item": item,
            "finding": finding_snapshot,
        })
    return contexts


def _sentence_comparison_debug_preview(rows, limit: int = 20) -> dict:
    changed = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        original = row.get("orig_sentence") or ""
        rewritten = row.get("new_sentence") or ""
        if original.strip() == rewritten.strip():
            continue
        changed.append({
            "index": row.get("index"),
            "orig_tier": row.get("orig_tier"),
            "orig_risk": row.get("orig_risk"),
            "orig_top10": row.get("orig_top10"),
            "orig_sentence": _truncate_debug_value(original, 420),
            "new_tier": row.get("new_tier"),
            "new_risk": row.get("new_risk"),
            "new_top10": row.get("new_top10"),
            "new_sentence": _truncate_debug_value(rewritten, 420),
        })
    return {
        "changed_count": len(changed),
        "preview": changed[:limit],
        "omitted": max(0, len(changed) - limit),
    }


def _manual_suggestions_debug_preview(rows, limit: int = 12) -> dict:
    suggestions = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        suggestions.append({
            "finding_id": row.get("finding_id"),
            "scanner_target": row.get("scanner_target"),
            "original_sentence": _truncate_debug_value(
                row.get("original_sentence") or "", 320
            ),
            "suggested_sentence": _truncate_debug_value(
                row.get("suggested_sentence") or "", 360
            ),
            "rejection_reason": _truncate_debug_value(
                row.get("rejection_reason") or "", 240
            ),
            "why_review_manually": _truncate_debug_value(
                row.get("why_review_manually") or "", 240
            ),
        })
    return {
        "count": len(suggestions),
        "preview": suggestions[:limit],
        "omitted": max(0, len(suggestions) - limit),
    }


def _guided_revision_debug_preview(mitigation: dict, limit: int = 5) -> dict:
    drivers = mitigation.get("component_drivers") or []
    actions = []
    for driver in drivers:
        if driver.get("bucket") != "needs_source_or_example":
            continue
        component = str(driver.get("component") or "")
        mitigation_text = str(driver.get("mitigation") or "")
        actions.append({
            "component": component,
            "score": driver.get("score"),
            "action": _truncate_debug_value(mitigation_text, 220),
        })
        if len(actions) >= limit:
            break
    patterns = []
    for pattern in (mitigation.get("reference_patterns") or [])[:limit]:
        if not isinstance(pattern, dict):
            continue
        patterns.append({
            "focus": _truncate_debug_value(pattern.get("focus") or "", 120),
            "try_pattern": _truncate_debug_value(pattern.get("try_pattern") or "", 260),
            "why": _truncate_debug_value(pattern.get("why") or "", 220),
        })
    return {
        "primary_mode": mitigation.get("primary_mode"),
        "top_actions": actions,
        "risk_mitigation_actions": [
            {
                "component": _truncate_debug_value(item.get("component") or "", 120),
                "action_type": _truncate_debug_value(item.get("action_type") or "", 120),
                "title": _truncate_debug_value(item.get("title") or "", 160),
                "current_score": item.get("current_score"),
                "target_score": item.get("target_score"),
                "safe_edit_pattern": _truncate_debug_value(item.get("safe_edit_pattern") or "", 260),
            }
            for item in (mitigation.get("risk_mitigation_actions") or [])[:limit]
            if isinstance(item, dict)
        ],
        "marked_content_suggestions": [
            {
                "component": _truncate_debug_value(item.get("component") or "", 120),
                "action_type": _truncate_debug_value(item.get("action_type") or "", 120),
                "title": _truncate_debug_value(item.get("title") or "", 160),
                "where": _truncate_debug_value(item.get("where") or "", 160),
                "target_text": _truncate_debug_value(item.get("target_text") or "", 260),
                "scanner_instruction": _truncate_debug_value(item.get("scanner_instruction") or "", 220),
                "suggested_addition": _truncate_debug_value(item.get("suggested_addition") or "", 320),
                "auto_apply": item.get("auto_apply"),
            }
            for item in (mitigation.get("marked_content_suggestions") or [])[:limit]
            if isinstance(item, dict)
        ],
        "reference_patterns": patterns,
    }


def _build_rewrite_debug_log(
    rewrite_id: str,
    scan_id: str,
    report_json: dict,
    pipeline_result: dict,
    rewrite_json: dict,
) -> str:
    """Create a downloadable internal rewrite log without secrets."""
    summary = rewrite_json.get("summary") or {}
    mitigation = summary.get("mitigation_plan") or {}
    rewrite_plan = report_json.get("rewrite_plan") or {}
    rewrite_decision = report_json.get("rewrite_decision") or {}
    badge = report_json.get("ai_risk_badge") or {}
    original_scan = summary.get("detect_scan_original") or {}
    final_scan = summary.get("detect_scan_rewritten") or {}
    attempted_scan = summary.get("detect_scan_attempted") or final_scan
    effective_badge = (original_scan.get("ai_risk_badge") or badge or {})
    effective_plan = rewrite_json.get("effective_rewrite_plan") or {}
    findings_by_id = _flatten_report_findings(report_json)
    sentence_comparison = rewrite_json.get("sentence_comparison") or []

    saved_scores = {
        "ai_likelihood_score": badge.get("ai_likelihood_score"),
        "writing_quality_score": badge.get("writing_quality_score"),
        "tier": badge.get("tier"),
    }

    def _badge_scores(scan: dict) -> dict:
        scan_badge = scan.get("ai_risk_badge") or {}
        scan_components = scan_badge.get("ai_components") or {}
        scan_rating = scan_badge.get("authorship_rating") or {}
        return {
            "ai_likelihood_score": scan_badge.get("ai_likelihood_score"),
            "writing_quality_score": scan_badge.get("writing_quality_score"),
            "tier": scan_badge.get("tier"),
            "authorship_rating": scan_rating.get("label") or scan_badge.get("authorship_rating_label"),
            "authorship_rating_code": scan_rating.get("code") or scan_badge.get("authorship_rating_code"),
            "ai_cluster_name": scan_badge.get("ai_cluster_name"),
            "qualifying_text_ai_density": scan_components.get("qualifying_text_ai_density"),
        }

    def _controller_phase_value(key: str, stage_name: str) -> dict | None:
        aliases = [key]
        if key == "rewrite_compiler":
            aliases.append("deterministic_rewrite_compiler")
        value = None
        if isinstance(summary, dict):
            for alias in aliases:
                candidate = summary.get(alias)
                if candidate is not None:
                    value = candidate
                    break
        if value is not None:
            if isinstance(value, dict):
                enriched = dict(value)
                if key == "rewrite_compiler":
                    selected_strategy = summary.get("selected_rewrite_compiler_strategy") or summary.get("selected_strategy")
                    if selected_strategy and str(selected_strategy).startswith("compiler_"):
                        enriched.setdefault("selected_strategy", selected_strategy)
                return enriched
            return value
        stage = next(
            (
                row for row in (summary.get("stage_timings") or [])
                if isinstance(row, dict) and row.get("stage") == stage_name
            ),
            None,
        ) if isinstance(summary, dict) else None
        if not stage:
            return value
        return {
            "enabled": True,
            "selected": stage.get("selected"),
            "reason": stage.get("stop_reason"),
            "candidate_count": stage.get("candidates"),
            "scans_used": stage.get("scans"),
            "mode": stage.get("mode"),
            "llm_calls_used": stage.get("llm_calls"),
            "outcome_class": stage.get("outcome_class"),
            "selected_strategy": (
                summary.get("selected_rewrite_compiler_strategy") or summary.get("selected_strategy")
                if key == "rewrite_compiler"
                and str(summary.get("selected_strategy") or "").startswith("compiler_")
                else None
            ),
            "reporting_fallback": True,
        }

    def _selected_rewrite_compiler_strategy() -> str | None:
        value = summary.get("selected_rewrite_compiler_strategy") if isinstance(summary, dict) else None
        if value:
            return value
        selected = summary.get("selected_strategy") if isinstance(summary, dict) else None
        if selected and str(selected).startswith("compiler_"):
            return selected
        compiler = _controller_phase_value("rewrite_compiler", "rewrite_compiler")
        if isinstance(compiler, dict) and compiler.get("selected_strategy"):
            return compiler.get("selected_strategy")
        return None

    runtime_fingerprint = _runtime_code_fingerprint()
    log_data = {
        "debug_export_version": REWRITE_DEBUG_EXPORT_VERSION,
        "debug_export_source": "worker.app.tasks._build_rewrite_debug_log",
        "runtime_code_sha": os.environ.get("DRAFTPROOF_RUNTIME_CODE_SHA"),
        "runtime_code_fingerprint": runtime_fingerprint,
        "rewrite_id": rewrite_id,
        "scan_id": scan_id,
        "status": rewrite_json.get("status"),
        "elapsed": rewrite_json.get("elapsed"),
        "pipeline_status": pipeline_result.get("status"),
        "pipeline_elapsed": pipeline_result.get("elapsed"),
        "input_scan": {
            "finding_count": report_json.get("finding_count"),
            "actionability_distribution": report_json.get("actionability_distribution"),
            "rewrite_decision": rewrite_decision,
            "rewrite_plan": {
                "mode": rewrite_plan.get("mode"),
                "overall_action": rewrite_plan.get("overall_action"),
                "auto_fixable": rewrite_plan.get("auto_fixable"),
                "auto_target_context": _target_contexts(
                    rewrite_plan.get("auto_fixable"),
                    findings_by_id,
                ),
                "manual_required": rewrite_plan.get("manual_required"),
                "review_only": rewrite_plan.get("review_only"),
                "no_action": rewrite_plan.get("no_action"),
                "citation_repairs": rewrite_plan.get("citation_repairs"),
            },
            "ai_risk_badge": {
                "tier": effective_badge.get("tier"),
                "ai_likelihood_score": effective_badge.get("ai_likelihood_score"),
                "authorship_rating": effective_badge.get("authorship_rating"),
                "authorship_rating_label": effective_badge.get("authorship_rating_label"),
                "authorship_rating_code": effective_badge.get("authorship_rating_code"),
                "ai_cluster_name": effective_badge.get("ai_cluster_name"),
                "ai_cluster_boost": effective_badge.get("ai_cluster_boost"),
                "writing_quality_score": effective_badge.get("writing_quality_score"),
                "writing_quality_tier": effective_badge.get("writing_quality_tier"),
                "review_priority": effective_badge.get("review_priority"),
                "confidence": effective_badge.get("confidence"),
                "ai_components": effective_badge.get("ai_components"),
                "writing_components": effective_badge.get("writing_components"),
                "reasons": effective_badge.get("reasons"),
                "guardrails": effective_badge.get("guardrails"),
                "schema_enriched_from_input_text": effective_badge.get("schema_enriched_from_input_text"),
                "saved_scan_contract": {
                    "tier": badge.get("tier"),
                    "ai_likelihood_score": badge.get("ai_likelihood_score"),
                    "authorship_rating_label": badge.get("authorship_rating_label"),
                    "authorship_rating_code": badge.get("authorship_rating_code"),
                },
            },
            "integrity_layers": _debug_integrity_layers(report_json, effective_badge),
            "ai_mitigation": report_json.get("ai_mitigation"),
            "scan_intelligence_mitigation_schema": (
                ((report_json.get("scan_intelligence") or {}).get("mitigation_inputs") or {})
                .get("ai_mitigation_plan", {})
                .get("schema_version")
            ),
        },
        "rewrite_summary": {
            "rewrite_pipeline_version": summary.get("rewrite_pipeline_version"),
            "outcome": summary.get("outcome"),
            "public_status": summary.get("public_status"),
            "public_candidate_warning": summary.get("public_candidate_warning"),
            "best_candidate_external_review_required": summary.get("best_candidate_external_review_required"),
            "best_candidate_author_review_required": summary.get("best_candidate_author_review_required"),
            "strict_safe_band_achieved": summary.get("strict_safe_band_achieved"),
            "kpi_finalization_status": summary.get("kpi_finalization_status"),
            "rewrite_goal_status": summary.get("rewrite_goal_status"),
            "strict_goal_status": summary.get("strict_goal_status"),
            "reference_ai": summary.get("reference_ai"),
            "required_ai_drop": summary.get("required_ai_drop"),
            "target_ai_score": summary.get("target_ai_score"),
            "candidate_generation_status": summary.get("candidate_generation_status"),
            "paragraph_obligation_hard_stop": summary.get("paragraph_obligation_hard_stop"),
            "content_router_trace": summary.get("content_router_trace"),
            "scan_contract": summary.get("scan_contract"),
            "v3_route": summary.get("v3_route"),
            "v3_strategy_plan": summary.get("v3_strategy_plan"),
            "strategy_trace": summary.get("strategy_trace"),
            "candidate_trace": summary.get("candidate_trace"),
            "candidate_loop_trace": summary.get("candidate_loop_trace"),
            "portfolio_scores": summary.get("portfolio_scores"),
            "selected_candidate": summary.get("selected_candidate"),
            "ai_mitigation": summary.get("ai_mitigation"),
            "ai_mitigation_blocked_auto_rewrite": summary.get("ai_mitigation_blocked_auto_rewrite"),
            "rewrite_runtime_version": summary.get("rewrite_runtime_version"),
            "rewrite_engine_mode": summary.get("rewrite_engine_mode"),
            "rewrite_effective_config": summary.get("rewrite_effective_config"),
            "mitigation_primary_mode_at_runtime": summary.get("mitigation_primary_mode_at_runtime"),
            "guided_revision_throttle": summary.get("guided_revision_throttle"),
            "no_text_change": summary.get("no_text_change"),
            "no_text_change_reason": summary.get("no_text_change_reason"),
            "rollback_applied": summary.get("rollback_applied"),
            "rollback_reason": summary.get("rollback_reason"),
            "passes_completed": summary.get("passes_completed"),
            "target_count": summary.get("target_count"),
            "unique_target_count": summary.get("unique_target_count"),
            "selected_finding_count": summary.get("selected_finding_count"),
            "llm_calls_used": summary.get("llm_calls_used"),
            "accepted_edits": summary.get("accepted_edits"),
            "manual_suggestions_count": len(summary.get("manual_suggestions") or []),
            "manual_suggestions": _manual_suggestions_debug_preview(
                summary.get("manual_suggestions") or []
            ),
            "checkpoint_selected": summary.get("checkpoint_selected"),
            "auto_target_cap": summary.get("auto_target_cap"),
            "failed_targets": summary.get("failed_targets"),
            "consecutive_failed_targets": summary.get("consecutive_failed_targets"),
            "findings_fixed": summary.get("findings_fixed"),
            "findings_skipped": summary.get("findings_skipped"),
            "circuit_breaker_reason": summary.get("circuit_breaker_reason"),
            "selected_strategy": summary.get("selected_strategy"),
            "selected_density_breaker_strategy": summary.get("selected_density_breaker_strategy"),
            "selected_human_anchor_probe_strategy": summary.get("selected_human_anchor_probe_strategy"),
            "selected_auto_repair_strategy": summary.get("selected_auto_repair_strategy"),
            "selected_rewrite_compiler_strategy": _selected_rewrite_compiler_strategy(),
            "selected_segment_window_strategy": summary.get("selected_segment_window_strategy"),
            "selected_segment_window_followup_strategy": summary.get("selected_segment_window_followup_strategy"),
            "selected_remaining_cluster_strategy": summary.get("selected_remaining_cluster_strategy"),
            "selected_window_coverage_strategy": summary.get("selected_window_coverage_strategy"),
            "rewrite_phase_budget_plan": summary.get("rewrite_phase_budget_plan"),
            "global_rewrite_budget": summary.get("global_rewrite_budget"),
            "global_rewrite_budget_contract": summary.get("global_rewrite_budget_contract"),
            "ai_mitigation_search": summary.get("ai_mitigation_search"),
            "formula_convergence_controller": summary.get("formula_convergence_controller"),
            "segment_window_density_controller": summary.get("segment_window_density_controller"),
            "segment_window_density_controller_followup": summary.get("segment_window_density_controller_followup"),
            "segment_window_budget_reserve": summary.get("segment_window_budget_reserve"),
            "segment_density_windows": summary.get("segment_density_windows"),
            "segment_window_candidate_frontier": summary.get("segment_window_candidate_frontier"),
            "remaining_cluster_density_controller": summary.get("remaining_cluster_density_controller"),
            "remaining_cluster_map": summary.get("remaining_cluster_map"),
            "remaining_cluster_candidate_frontier": summary.get("remaining_cluster_candidate_frontier"),
            "window_coverage_density_optimizer": summary.get("window_coverage_density_optimizer"),
            "window_coverage_map": summary.get("window_coverage_map"),
            "top_coverage_sentences": summary.get("top_coverage_sentences"),
            "window_coverage_candidate_frontier": summary.get("window_coverage_candidate_frontier"),
            "unsafe_window_count_before": summary.get("unsafe_window_count_before"),
            "unsafe_window_count_after": summary.get("unsafe_window_count_after"),
            "unsafe_window_count_drop": summary.get("unsafe_window_count_drop"),
            "ai_sentence_vote_ratio_before": summary.get("ai_sentence_vote_ratio_before"),
            "ai_sentence_vote_ratio_after": summary.get("ai_sentence_vote_ratio_after"),
            "ai_sentence_vote_ratio_drop": summary.get("ai_sentence_vote_ratio_drop"),
            "post_selection_ai_density_breaker": _controller_phase_value(
                "post_selection_ai_density_breaker",
                "post_selection_ai_density_breaker",
            ),
            "post_density_human_anchor_probe": _controller_phase_value(
                "post_density_human_anchor_probe",
                "post_density_human_anchor_probe",
            ),
            "auto_repair_controller": _controller_phase_value(
                "auto_repair_controller",
                "auto_repair_controller",
            ),
            "rewrite_compiler": _controller_phase_value(
                "rewrite_compiler",
                "rewrite_compiler",
            ),
            "deterministic_rewrite_compiler": _controller_phase_value(
                "rewrite_compiler",
                "rewrite_compiler",
            ),
            "detector_safe_label_status": summary.get("detector_safe_label_status"),
            "generation_layer": summary.get("generation_layer"),
            "authenticity_mitigation": summary.get("authenticity_mitigation"),
            "authenticity_llm_calls_used": summary.get("authenticity_llm_calls_used"),
            "marked_mitigation_rewrite": summary.get("marked_mitigation_rewrite"),
            "stage_timings": summary.get("stage_timings"),
            "comparison_baseline": summary.get("comparison_baseline"),
            "baseline_rescan_delta": summary.get("baseline_rescan_delta"),
            "saved_contract_notes": summary.get("saved_contract_notes"),
            "saved_user_visible_scores": saved_scores,
            "original_scores": _badge_scores(original_scan),
            "attempted_scores": _badge_scores(attempted_scan),
            "final_scores": _badge_scores(final_scan),
            "detect_scores": summary.get("detect_scores"),
            "detect_scan_rewritten": summary.get("detect_scan_rewritten"),
            "effective_rewrite_plan": effective_plan,
            "effective_target_context": _target_contexts(
                effective_plan.get("auto_fixable"),
                findings_by_id,
            ),
            "mitigation_counts": mitigation.get("counts"),
            "mitigation_primary_mode": mitigation.get("primary_mode"),
            "component_drivers": mitigation.get("component_drivers"),
            "score_mitigation_targets": mitigation.get("score_mitigation_targets"),
            "risk_mitigation_actions": mitigation.get("risk_mitigation_actions"),
            "marked_content_suggestions": mitigation.get("marked_content_suggestions"),
            "guided_revision": _guided_revision_debug_preview(mitigation),
        },
        "loop_history": summary.get("detect_loop_history") or summary.get("loop_history"),
        "sentence_comparison_count": len(sentence_comparison),
        "sentence_comparison_changes": _sentence_comparison_debug_preview(sentence_comparison),
    }
    rewrite_debug = log_data.get("rewrite_summary") or {}
    controller_debug_keys = [
        "selected_strategy",
        "selected_density_breaker_strategy",
        "selected_human_anchor_probe_strategy",
        "selected_auto_repair_strategy",
        "selected_rewrite_compiler_strategy",
        "selected_formula_strategy",
        "selected_segment_window_strategy",
        "selected_segment_window_followup_strategy",
        "selected_remaining_cluster_strategy",
        "selected_window_coverage_strategy",
        "rewrite_phase_budget_plan",
        "global_rewrite_budget",
        "global_rewrite_budget_contract",
        "formula_convergence_controller",
        "segment_window_density_controller",
        "segment_window_density_controller_followup",
        "segment_window_budget_reserve",
        "segment_density_windows",
        "segment_window_candidate_frontier",
        "remaining_cluster_density_controller",
        "remaining_cluster_map",
        "remaining_cluster_candidate_frontier",
        "window_coverage_density_optimizer",
        "window_coverage_map",
        "top_coverage_sentences",
        "window_coverage_candidate_frontier",
        "unsafe_window_count_before",
        "unsafe_window_count_after",
        "unsafe_window_count_drop",
        "ai_sentence_vote_ratio_before",
        "ai_sentence_vote_ratio_after",
        "ai_sentence_vote_ratio_drop",
        "post_selection_ai_density_breaker",
        "post_density_human_anchor_probe",
        "auto_repair_controller",
        "rewrite_compiler",
        "detector_safe_label_status",
        "ai_footprint_gate",
        "turnitin_like_ai_gate",
        "formula_gap_contract",
        "formula_portfolio_plan",
        "positive_ai_burden",
        "human_anchor_suppression",
        "suppression_headroom",
        "required_suppression_gain",
        "expected_net_gain",
        "observed_driver_movement",
        "weighted_driver_plan",
        "driver_priority_plan",
        "weighted_driver_drops",
        "remaining_formula_gap",
        "why_not_below_20",
        "why_not_strict_safe",
        "strict_ai_safe_band_achieved",
    ]
    rewrite_debug["raw_summary_keys"] = sorted(summary.keys()) if isinstance(summary, dict) else []
    rewrite_debug["controller_field_presence"] = {
        key: key in summary for key in controller_debug_keys
    } if isinstance(summary, dict) else {}
    for key in controller_debug_keys:
        if isinstance(summary, dict) and key in summary:
            if key in {"post_selection_ai_density_breaker", "post_density_human_anchor_probe", "auto_repair_controller", "rewrite_compiler"}:
                rewrite_debug[key] = _controller_phase_value(key, key)
            else:
                rewrite_debug[key] = summary.get(key)
    rewrite_debug["controller_stage_timings"] = [
        stage for stage in (summary.get("stage_timings") or [])
        if isinstance(stage, dict)
        and str(stage.get("stage") or "") in {
            "ai_mitigation_search",
            "formula_convergence_controller",
            "segment_window_density_controller",
            "segment_window_density_controller_followup",
            "remaining_cluster_density_controller",
            "window_coverage_density_optimizer",
            "post_selection_ai_density_breaker",
            "post_density_human_anchor_probe",
            "auto_repair_controller",
            "rewrite_compiler",
        }
    ] if isinstance(summary, dict) else []
    return _bounded_json_debug_log(log_data)


def _serialize_effective_rewrite_plan(plan) -> dict:
    """Serialize the actual planner output used by the rewrite engine."""
    if not plan:
        return {}

    def _action(action) -> dict:
        finding = getattr(action, "finding", None)
        metadata = getattr(finding, "metadata", {}) or {}
        return {
            "finding_id": metadata.get("finding_id") or getattr(finding, "id", ""),
            "finding_type": getattr(finding, "finding_type", ""),
            "risk_level": getattr(finding, "risk_level", ""),
            "actionability": getattr(finding, "actionability", ""),
            "action_type": getattr(action, "action_type", ""),
            "scope": getattr(action, "scope", ""),
            "fixability": getattr(action, "fixability", ""),
            "reason": getattr(action, "reason", ""),
        }

    return {
        "auto_fixable": [_action(action) for action in getattr(plan, "auto_fixable", [])],
        "manual_required": [_action(action) for action in getattr(plan, "manual_required", [])],
        "review_only": [_action(action) for action in getattr(plan, "review_only", [])],
        "protected": [_action(action) for action in getattr(plan, "protected", [])],
    }
