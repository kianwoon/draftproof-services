"""Celery tasks — scan_document runs the full detect pipeline."""
# v2: scan progress published to Redis via publish_scan_progress

import sys
import os
import json
import time
import logging

# Make poc/ importable — on Koyeb: /app/poc/, locally: ../../poc/
_app_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_app_dir, "..", "..")
if _repo_root not in sys.path:
    sys.path.insert(0, os.path.abspath(_repo_root))
_poc_dir = os.path.join(_repo_root, "poc")
if _poc_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_poc_dir))

from .celery_app import app
from .config import settings
from .storage import upload_report_files
from .progress import publish_rewrite_progress, publish_scan_progress
from .db import (
    get_scan_job,
    update_job_status,
    capture_credits,
    get_rewrite_job,
    claim_rewrite_job,
    update_rewrite_status,
    capture_rewrite_credits,
    release_rewrite_credits,
)
from celery.signals import worker_process_init
from celery.exceptions import SoftTimeLimitExceeded

logger = logging.getLogger(__name__)


@worker_process_init.connect
def _preload_predictability_model(**_kwargs):
    """Warm the GPT-2 scanner inside the Celery worker child process."""
    enabled = os.environ.get("DRAFTPROOF_PRELOAD_PREDICTABILITY", "1").lower()
    if enabled in {"0", "false", "no"}:
        return
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import predictability.scanner as scanner_module

        model_name = os.environ.get("PREDICTABILITY_MODEL", "gpt2")
        if scanner_module._PRELOADED_MODEL is not None:
            return
        logger.info("Preloading predictability model in worker child: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        scanner_module._PRELOADED_MODEL = model
        scanner_module._PRELOADED_TOKENIZER = tokenizer
        logger.info("Predictability model preloaded in worker child: %s", model_name)
    except Exception:
        logger.warning("Failed to preload predictability model in worker child", exc_info=True)


def _truncate_debug_value(value, limit: int = 320):
    if isinstance(value, str):
        text = " ".join(value.split())
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
    return value


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
        "ai_risk_badge": badge,
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


def _fetch_r2_json(s3, bucket: str, key: str) -> dict:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read())


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

    log_data = {
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
        },
        "rewrite_summary": {
            "outcome": summary.get("outcome"),
            "rewrite_runtime_version": summary.get("rewrite_runtime_version"),
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
            "ai_mitigation_search": summary.get("ai_mitigation_search"),
            "stage_timings": summary.get("stage_timings"),
            "comparison_baseline": summary.get("comparison_baseline"),
            "baseline_rescan_delta": summary.get("baseline_rescan_delta"),
            "saved_contract_notes": summary.get("saved_contract_notes"),
            "saved_user_visible_scores": saved_scores,
            "original_scores": _badge_scores(original_scan),
            "attempted_scores": _badge_scores(attempted_scan),
            "final_scores": _badge_scores(final_scan),
            "detect_scores": summary.get("detect_scores"),
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
    return json.dumps(log_data, indent=2, ensure_ascii=False, default=str)


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


@app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=settings.SCAN_SOFT_TIME_LIMIT_SECONDS,
    time_limit=settings.SCAN_TIME_LIMIT_SECONDS,
)
def scan_document(self, job_id: str, text: str) -> dict:
    """Run the full detect pipeline on text and store results."""
    try:
        update_job_status(
            job_id,
            "processing",
            progress_percent=10,
            progress_message="Preparing scan",
        )
        publish_scan_progress(
            job_id, status="processing",
            progress_percent=10, progress_message="Preparing scan",
        )
        last_scan_progress = {
            "redis_percent": 10,
            "redis_updated_at": time.monotonic(),
            "db_percent": 10,
            "db_updated_at": time.monotonic(),
        }

        from poc.detect_pipeline import run_detect
        import tempfile

        def report_progress(percent: int, message: str) -> None:
            pct = max(0, min(99, int(percent)))
            now = time.monotonic()
            if (
                pct <= last_scan_progress["redis_percent"]
                and pct < 97
                and now - last_scan_progress["redis_updated_at"] < 0.5
            ):
                return
            publish_scan_progress(
                job_id, status="processing",
                progress_percent=pct, progress_message=message,
            )
            last_scan_progress["redis_percent"] = pct
            last_scan_progress["redis_updated_at"] = now

            should_persist = (
                pct >= 97
                or pct - last_scan_progress["db_percent"] >= 5
                or now - last_scan_progress["db_updated_at"] >= 10.0
            )
            if should_persist:
                update_job_status(
                    job_id,
                    "processing",
                    progress_percent=pct,
                    progress_message=message,
                )
                last_scan_progress["db_percent"] = pct
                last_scan_progress["db_updated_at"] = now

        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = os.environ.get("PREDICTABILITY_MODEL", "gpt2")
            result = run_detect(
                text,
                tmpdir,
                verbose=True,
                model_name=model_name,
                progress_callback=report_progress,
            )

            tier = result["tier"]
            finding_count = result["findings"]

            # Extract AI score and writing score from results JSON
            ai_score = None
            writing_score = None
            with open(result["json_path"]) as f:
                results_json = json.load(f)
            badge = results_json.get("ai_risk_badge")
            if badge:
                ai_score = badge.get("ai_likelihood_score")
                writing_score = badge.get("writing_quality_score")

            with open(result["md_path"]) as f:
                md_text = f.read()
            with open(result["pdf_path"], "rb") as f:
                pdf_bytes = f.read()
            with open(result["json_path"]) as f:
                results_json = json.load(f)

            report_progress(97, "Uploading report files")
            urls = upload_report_files(job_id, md_text, pdf_bytes, results_json)

            report_urls = {
                "md": urls.get("md"),
                "pdf": urls.get("pdf"),
                "json": urls.get("json"),
            }


            word_count = len(text.split())
            job = get_scan_job(job_id)
            capture_credits(job.get("user_id", ""), job_id, word_count)
            update_job_status(
                job_id,
                "completed",
                tier=tier,
                ai_score=ai_score,
                writing_score=writing_score,
                finding_count=finding_count,
                report_urls=report_urls,
                progress_percent=100,
                progress_message="Scan complete",
            )
            publish_scan_progress(
                job_id, status="completed",
                progress_percent=100, progress_message="Scan complete",
            )

            return {"status": "completed", "tier": tier, "findings": finding_count}

    except SoftTimeLimitExceeded:
        update_job_status(
            job_id,
            "failed",
            error="Scan timed out (5 min limit)",
            progress_message="Scan timed out",
        )
        publish_scan_progress(
            job_id, status="failed",
            error="Scan timed out", progress_message="Scan timed out",
        )
        return {"status": "failed", "error": "timeout"}
    except Exception as e:
        if self.request.retries < self.max_retries:
            update_job_status(
                job_id,
                "retrying",
                error=str(e),
                progress_message="Retrying scan",
            )
            publish_scan_progress(
                job_id, status="retrying",
                error=str(e), progress_message="Retrying scan",
            )
            raise self.retry(exc=e)
        else:
            update_job_status(
                job_id,
                "failed",
                error=str(e),
                progress_message="Scan failed",
            )
            publish_scan_progress(
                job_id, status="failed",
                error=str(e), progress_message="Scan failed",
            )
            raise  # Re-raise original — Celery marks as FAILURE, not RETRY


@app.task(
    bind=True,
    max_retries=1,
    default_retry_delay=30,
    soft_time_limit=settings.REWRITE_SOFT_TIME_LIMIT_SECONDS,
    time_limit=settings.REWRITE_TIME_LIMIT_SECONDS,
)
def run_rewrite(self, rewrite_id: str, scan_id: str) -> dict:
    """Run the rewrite pipeline on a completed scan's results."""
    from .storage import upload_rewrite_files, _client as _r2_client
    from .config import settings as worker_settings
    import tempfile

    def publish_progress(
        status: str,
        percent: int | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        publish_rewrite_progress(
            rewrite_id,
            status=status,
            progress_percent=percent,
            progress_message=message,
            scan_id=scan_id,
            error=error,
        )

    try:
        rewrite_job = claim_rewrite_job(rewrite_id)
        if not rewrite_job:
            existing = get_rewrite_job(rewrite_id)
            status = existing.get("status") if existing else "missing"
            return {
                "status": "skipped",
                "reason": f"rewrite job already {status}",
                "rewrite_id": rewrite_id,
            }

        # 1. Fetch report.json from R2
        update_rewrite_status(
            rewrite_id,
            "processing",
            progress_percent=10,
            progress_message="Fetching original report",
        )
        publish_progress("processing", 10, "Fetching original report")
        scan_job = get_scan_job(scan_id)
        report_json = None
        try:
            s3 = _r2_client()
            resp = s3.get_object(
                Bucket=worker_settings.R2_BUCKET_NAME,
                Key=f"reports/{scan_id}/report.json",
            )
            report_json = json.loads(resp["Body"].read())
        except Exception:
            update_rewrite_status(
                rewrite_id,
                "failed",
                error="Original report not found in R2",
                progress_message="Original report not found",
            )
            publish_progress(
                "failed",
                10,
                "Original report not found",
                "Original report not found in R2",
            )
            release_rewrite_credits(rewrite_id)
            return {"status": "failed", "error": "report not found"}

        # 2. Filter findings: only rephrase-fixable ones
        update_rewrite_status(
            rewrite_id,
            "processing",
            progress_percent=25,
            progress_message="Selecting AI sections",
        )
        publish_progress("processing", 25, "Selecting AI sections")
        review_only_message = (
            "No rewriteable AI sections were found. The findings on this report are review-only, "
            "so DraftProof did not use any tokens for a rewrite."
        )
        rewrite_decision = report_json.get("rewrite_decision") or {}
        if isinstance(rewrite_decision, dict) and rewrite_decision.get("run_rewrite") is False:
            update_rewrite_status(
                rewrite_id,
                "failed",
                error=review_only_message,
                progress_message="Review-only findings; no rewrite needed",
            )
            publish_progress("failed", 25, "Review-only findings; no rewrite needed", review_only_message)
            release_rewrite_credits(rewrite_id)
            return {"status": "failed", "error": "review-only findings"}

        # findings is a dict: {critical: [...], high: [...], medium: [...], low: [...]}
        # Include: auto_fixable/auto_rewrite_candidate + review_only findings whose
        # title maps to a rephrasable type in the planner's FINDING_ROUTING table.
        findings_by_tier = report_json.get("findings", {})
        all_findings = []
        for tier_findings in findings_by_tier.values():
            if isinstance(tier_findings, list):
                all_findings.extend(tier_findings)

        REPHRASABLE_TYPES = {
            "high_predictability", "medium_predictability", "review_predictability",
            "high_topk_predictability", "low_predictability", "formulaic_sentence",
            "generic_phrase", "style_shift", "low_specificity",
        }
        rephrasable_findings = [
            f for f in all_findings
            if isinstance(f, dict) and (
                f.get("actionability") in ("auto_fixable", "auto_rewrite_candidate")
                or (f.get("title") in REPHRASABLE_TYPES and f.get("recommendation"))
            )
        ]
        if not rephrasable_findings:
            update_rewrite_status(
                rewrite_id,
                "failed",
                error=review_only_message,
                progress_message="Review-only findings; no rewrite needed",
            )
            publish_progress("failed", 25, "Review-only findings; no rewrite needed", review_only_message)
            release_rewrite_credits(rewrite_id)
            return {"status": "failed", "error": "review-only findings"}

        # 3. Run rewrite pipeline
        from poc.rewrite_pipeline import run_rewrite_pipeline
        from app.config import settings

        llm_api_key = settings.LLM_API_KEY or settings.OPENROUTER_API_KEY
        if not llm_api_key:
            update_rewrite_status(
                rewrite_id,
                "failed",
                error="LLM_API_KEY not configured — rewrite requires an LLM API key",
                progress_message="Rewrite service is not configured",
            )
            publish_progress(
                "failed",
                25,
                "Rewrite service is not configured",
                "LLM_API_KEY not configured — rewrite requires an LLM API key",
            )
            release_rewrite_credits(rewrite_id)
            return {"status": "failed", "error": "missing LLM API key"}

        with tempfile.TemporaryDirectory() as tmpdir:
            update_rewrite_status(
                rewrite_id,
                "processing",
                progress_percent=40,
                progress_message="Rewriting AI sections",
            )
            publish_progress("processing", 40, "Rewriting AI sections")

            last_rewrite_progress = {
                "redis_percent": 39,
                "redis_updated_at": 0.0,
                "db_percent": 40,
                "db_updated_at": time.monotonic(),
            }

            def report_rewrite_progress(percent: int, message: str) -> None:
                normalized_percent = max(40, min(79, int(percent)))
                now = time.monotonic()
                if (
                    normalized_percent <= last_rewrite_progress["redis_percent"]
                    and normalized_percent < 76
                    and now - last_rewrite_progress["redis_updated_at"] < 0.5
                ):
                    return
                publish_progress("processing", normalized_percent, message)
                last_rewrite_progress["redis_percent"] = normalized_percent
                last_rewrite_progress["redis_updated_at"] = now

                should_persist = (
                    normalized_percent >= 76
                    or normalized_percent - last_rewrite_progress["db_percent"] >= 5
                    or now - last_rewrite_progress["db_updated_at"] >= 12.0
                )
                if should_persist:
                    update_rewrite_status(
                        rewrite_id,
                        "processing",
                        progress_percent=normalized_percent,
                        progress_message=message,
                    )
                    last_rewrite_progress["db_percent"] = normalized_percent
                    last_rewrite_progress["db_updated_at"] = now

            result = run_rewrite_pipeline(
                detect_json=report_json,
                output_dir=tmpdir,
                max_passes=3,
                max_detect_loops=0,
                ai_only=True,
                verbose=False,
                api_key=llm_api_key or None,
                model=settings.LLM_MODEL or None,
                base_url=settings.LLM_BASE_URL or None,
                progress_callback=report_rewrite_progress,
            )

            if result["status"] in ("skipped", "clean"):
                update_rewrite_status(
                    rewrite_id,
                    "failed",
                    error=result.get("message", "Rewrite not needed"),
                    progress_message="Rewrite not needed",
                )
                publish_progress(
                    "failed",
                    40,
                    "Rewrite not needed",
                    result.get("message", "Rewrite not needed"),
                )
                release_rewrite_credits(rewrite_id)
                return {"status": "skipped"}

            # Read files WHILE tmpdir still exists (before with block exits)
            update_rewrite_status(
                rewrite_id,
                "processing",
                progress_percent=80,
                progress_message="Preparing rewrite report",
            )
            publish_progress("processing", 80, "Preparing rewrite report")
            rw = result.get("result")
            md_path = result.get("md_path")
            pdf_path = result.get("pdf_path")

            import logging as _log
            _l = _log.getLogger("rewrite_task")
            _l.info("Pipeline result keys: %s", list(result.keys()))
            if rw and hasattr(rw, "summary"):
                _l.info("Pipeline stage timings: %s", rw.summary.get("stage_timings"))
            _l.info("md_path=%s exists=%s", md_path, md_path and os.path.exists(md_path))
            _l.info("pdf_path=%s exists=%s", pdf_path, pdf_path and os.path.exists(pdf_path))

            md_text = ""
            pdf_bytes = b""
            if md_path and os.path.exists(md_path):
                with open(md_path) as f:
                    md_text = f.read()
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

            _l.info("md_text len=%d, pdf_bytes len=%d", len(md_text), len(pdf_bytes))

            rewritten_text = ""
            if rw and hasattr(rw, "mp_result") and rw.mp_result:
                rewritten_text = rw.mp_result.final_text or ""

            rewrite_json = {
                "status": result.get("status"),
                "elapsed": result.get("elapsed"),
                "original_text": rewritten_text and getattr(rw.mp_result, "original_text", "") if rw and hasattr(rw, "mp_result") and rw.mp_result else "",
                "final_text": rewritten_text,
                "converged": rw.mp_result.converged if rw and hasattr(rw, "mp_result") and rw.mp_result else False,
                "convergence_reason": rw.mp_result.convergence_reason if rw and hasattr(rw, "mp_result") and rw.mp_result else "",
                "passes": len(rw.mp_result.passes) if rw and hasattr(rw, "mp_result") and rw.mp_result else 0,
                "summary": rw.summary if rw and hasattr(rw, "summary") else {},
                "sentence_comparison": rw.sentence_comparison if rw and hasattr(rw, "sentence_comparison") else [],
                "effective_rewrite_plan": _serialize_effective_rewrite_plan(
                    rw.rewrite_plan if rw and hasattr(rw, "rewrite_plan") else None
                ),
            }

            debug_log = _build_rewrite_debug_log(
                rewrite_id=rewrite_id,
                scan_id=scan_id,
                report_json=report_json,
                pipeline_result=result,
                rewrite_json=rewrite_json,
            )
            update_rewrite_status(
                rewrite_id,
                "processing",
                progress_percent=92,
                progress_message="Uploading rewrite results",
            )
            publish_progress("processing", 92, "Uploading rewrite results")
            upload_rewrite_files(scan_id, md_text, pdf_bytes, rewrite_json, rewritten_text, debug_log)

        # 5. Capture credits
        user_id = scan_job.get("user_id", "") if scan_job else ""
        if user_id:
            capture_rewrite_credits(str(user_id), rewrite_id)

        update_rewrite_status(
            rewrite_id,
            "completed",
            progress_percent=100,
            progress_message="Rewrite complete",
        )
        publish_progress("completed", 100, "Rewrite complete")
        return {"status": "completed"}

    except SoftTimeLimitExceeded:
        timeout_minutes = max(1, settings.REWRITE_SOFT_TIME_LIMIT_SECONDS // 60)
        update_rewrite_status(
            rewrite_id,
            "failed",
            error=f"Rewrite timed out ({timeout_minutes} min limit)",
            progress_message="Rewrite timed out",
        )
        publish_progress(
            "failed",
            None,
            "Rewrite timed out",
            f"Rewrite timed out ({timeout_minutes} min limit)",
        )
        release_rewrite_credits(rewrite_id)
        return {"status": "failed", "error": "timeout"}
    except Exception as e:
        if self.request.retries < self.max_retries:
            update_rewrite_status(
                rewrite_id,
                "retrying",
                error=str(e),
                progress_message="Retrying rewrite",
            )
            publish_progress("retrying", None, "Retrying rewrite", str(e))
            raise self.retry(exc=e)
        else:
            update_rewrite_status(
                rewrite_id,
                "failed",
                error=str(e),
                progress_message="Rewrite failed",
            )
            publish_progress("failed", None, "Rewrite failed", str(e))
            release_rewrite_credits(rewrite_id)
            raise


@app.task(bind=True, max_retries=1, default_retry_delay=30)
def regenerate_rewrite_report_assets(self, rewrite_id: str, scan_id: str) -> dict:
    """Regenerate rewrite.md and rewrite.pdf from stored JSON without charging tokens."""
    from .storage import upload_rewrite_files, _client as _r2_client
    from .config import settings as worker_settings
    from report.render_rewrite import render_rewrite_report
    from report.pdf import render_pdf
    import tempfile

    try:
        rewrite_job = get_rewrite_job(rewrite_id)
        if not rewrite_job:
            return {"status": "failed", "error": "rewrite job not found"}
        if str(rewrite_job.get("scan_id")) != str(scan_id):
            return {"status": "failed", "error": "scan mismatch"}
        if rewrite_job.get("status") != "completed":
            return {"status": "failed", "error": "rewrite is not completed"}

        s3 = _r2_client()
        bucket = worker_settings.R2_BUCKET_NAME
        report_json = _fetch_r2_json(s3, bucket, f"reports/{scan_id}/report.json")
        rewrite_json = _fetch_r2_json(s3, bucket, f"reports/{scan_id}/rewrite/rewrite.json")

        summary = rewrite_json.setdefault("summary", {})
        summary["detect_scan_original_saved"] = _extract_rewrite_scan_summary(report_json)

        md_text = render_rewrite_report(
            summary=summary,
            sentence_comparison=rewrite_json.get("sentence_comparison") or [],
            ai_findings=rewrite_json.get("ai_findings") or [],
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "rewrite.pdf")
            render_pdf(md_text, pdf_path)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

        upload_rewrite_files(
            scan_id=scan_id,
            md_text=md_text,
            pdf_bytes=pdf_bytes,
            json_data=rewrite_json,
            rewritten_text=rewrite_json.get("final_text") or "",
        )
        return {"status": "completed", "rewrite_id": rewrite_id, "scan_id": scan_id}
    except Exception as e:
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        raise
