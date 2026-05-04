"""Celery tasks — scan_document runs the full detect pipeline."""

import sys
import os
import json

# Make poc/ importable — on Koyeb: /app/poc/, locally: ../../poc/
_app_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_app_dir, "..", "..")
if _repo_root not in sys.path:
    sys.path.insert(0, os.path.abspath(_repo_root))

from .celery_app import app
from .config import settings
from .storage import upload_report_files
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
from celery.exceptions import SoftTimeLimitExceeded


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
    effective_plan = rewrite_json.get("effective_rewrite_plan") or {}

    saved_scores = {
        "ai_likelihood_score": badge.get("ai_likelihood_score"),
        "writing_quality_score": badge.get("writing_quality_score"),
        "tier": badge.get("tier"),
    }

    def _badge_scores(scan: dict) -> dict:
        scan_badge = scan.get("ai_risk_badge") or {}
        return {
            "ai_likelihood_score": scan_badge.get("ai_likelihood_score"),
            "writing_quality_score": scan_badge.get("writing_quality_score"),
            "tier": scan_badge.get("tier"),
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
                "manual_required": rewrite_plan.get("manual_required"),
                "review_only": rewrite_plan.get("review_only"),
                "no_action": rewrite_plan.get("no_action"),
                "citation_repairs": rewrite_plan.get("citation_repairs"),
            },
            "ai_risk_badge": {
                "tier": badge.get("tier"),
                "ai_likelihood_score": badge.get("ai_likelihood_score"),
                "writing_quality_score": badge.get("writing_quality_score"),
                "ai_components": badge.get("ai_components"),
                "writing_components": badge.get("writing_components"),
                "reasons": badge.get("reasons"),
            },
        },
        "rewrite_summary": {
            "outcome": summary.get("outcome"),
            "no_text_change": summary.get("no_text_change"),
            "rollback_applied": summary.get("rollback_applied"),
            "rollback_reason": summary.get("rollback_reason"),
            "passes_completed": summary.get("passes_completed"),
            "failed_targets": summary.get("failed_targets"),
            "consecutive_failed_targets": summary.get("consecutive_failed_targets"),
            "findings_fixed": summary.get("findings_fixed"),
            "findings_skipped": summary.get("findings_skipped"),
            "circuit_breaker_reason": summary.get("circuit_breaker_reason"),
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
            "mitigation_counts": mitigation.get("counts"),
            "mitigation_primary_mode": mitigation.get("primary_mode"),
            "component_drivers": mitigation.get("component_drivers"),
        },
        "loop_history": summary.get("detect_loop_history") or summary.get("loop_history"),
        "sentence_comparison_count": len(rewrite_json.get("sentence_comparison") or []),
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


@app.task(bind=True, max_retries=2, default_retry_delay=30, soft_time_limit=300, time_limit=330)
def scan_document(self, job_id: str, text: str) -> dict:
    """Run the full detect pipeline on text and store results."""
    try:
        update_job_status(job_id, "processing")

        from poc.detect_pipeline import run_detect
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = os.environ.get("PREDICTABILITY_MODEL", "gpt2")
            result = run_detect(text, tmpdir, verbose=True, model_name=model_name)

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
            )

            return {"status": "completed", "tier": tier, "findings": finding_count}

    except SoftTimeLimitExceeded:
        update_job_status(job_id, "failed", error="Scan timed out (5 min limit)")
        return {"status": "failed", "error": "timeout"}
    except Exception as e:
        if self.request.retries < self.max_retries:
            update_job_status(job_id, "retrying", error=str(e))
            raise self.retry(exc=e)
        else:
            update_job_status(job_id, "failed", error=str(e))
            raise  # Re-raise original — Celery marks as FAILURE, not RETRY


@app.task(bind=True, max_retries=1, default_retry_delay=30, soft_time_limit=600, time_limit=630)
def run_rewrite(self, rewrite_id: str, scan_id: str) -> dict:
    """Run the rewrite pipeline on a completed scan's results."""
    from .storage import upload_rewrite_files, _client as _r2_client
    from .config import settings as worker_settings
    import tempfile

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
            update_rewrite_status(rewrite_id, "failed", error="Original report not found in R2")
            release_rewrite_credits(rewrite_id)
            return {"status": "failed", "error": "report not found"}

        # 2. Filter findings: only rephrase-fixable ones
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
            update_rewrite_status(rewrite_id, "failed", error="No rephrasable findings to rewrite")
            release_rewrite_credits(rewrite_id)
            return {"status": "failed", "error": "no rephrasable findings"}

        # 3. Run rewrite pipeline
        from poc.rewrite_pipeline import run_rewrite_pipeline
        from app.config import settings

        llm_api_key = settings.LLM_API_KEY or settings.OPENROUTER_API_KEY
        if not llm_api_key:
            update_rewrite_status(rewrite_id, "failed", error="LLM_API_KEY not configured — rewrite requires an LLM API key")
            release_rewrite_credits(rewrite_id)
            return {"status": "failed", "error": "missing LLM API key"}

        with tempfile.TemporaryDirectory() as tmpdir:
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
            )

            if result["status"] in ("skipped", "clean"):
                update_rewrite_status(rewrite_id, "failed", error=result.get("message", "Rewrite not needed"))
                release_rewrite_credits(rewrite_id)
                return {"status": "skipped"}

            # Read files WHILE tmpdir still exists (before with block exits)
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
            upload_rewrite_files(scan_id, md_text, pdf_bytes, rewrite_json, rewritten_text, debug_log)

        # 5. Capture credits
        user_id = scan_job.get("user_id", "") if scan_job else ""
        if user_id:
            capture_rewrite_credits(str(user_id), rewrite_id)

        update_rewrite_status(rewrite_id, "completed")
        return {"status": "completed"}

    except SoftTimeLimitExceeded:
        update_rewrite_status(rewrite_id, "failed", error="Rewrite timed out (5 min limit)")
        release_rewrite_credits(rewrite_id)
        return {"status": "failed", "error": "timeout"}
    except Exception as e:
        if self.request.retries < self.max_retries:
            update_rewrite_status(rewrite_id, "retrying", error=str(e))
            raise self.retry(exc=e)
        else:
            update_rewrite_status(rewrite_id, "failed", error=str(e))
            release_rewrite_credits(rewrite_id)
            raise
            update_rewrite_status(rewrite_id, "failed", error=str(e))
            raise
