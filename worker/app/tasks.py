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
from .db import get_scan_job, update_job_status, capture_credits, get_rewrite_job, update_rewrite_status, capture_rewrite_credits
from celery.exceptions import SoftTimeLimitExceeded


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


@app.task(bind=True, max_retries=2, default_retry_delay=30, soft_time_limit=600, time_limit=660)
def run_rewrite(self, rewrite_id: str, scan_id: str) -> dict:
    """Run the rewrite pipeline on a completed scan's results."""
    from .storage import upload_rewrite_files, _client as _r2_client
    from .config import settings as worker_settings
    import tempfile

    try:
        update_rewrite_status(rewrite_id, "processing")

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
            return {"status": "failed", "error": "report not found"}

        # 2. Filter findings: only rephrase-fixable ones
        # findings is a dict: {critical: [...], high: [...], medium: [...], low: [...]}
        # report.json uses "auto_fixable" (from report.py determine_actionability)
        # Only these findings can be mitigated by rephrasing
        findings_by_tier = report_json.get("findings", {})
        all_findings = []
        for tier_findings in findings_by_tier.values():
            if isinstance(tier_findings, list):
                all_findings.extend(tier_findings)
        rephrasable_findings = [
            f for f in all_findings
            if isinstance(f, dict) and f.get("actionability") in ("auto_fixable", "auto_rewrite_candidate")
        ]
        # Fallback: include review_only findings that have a suggestion
        if not rephrasable_findings:
            rephrasable_findings = [
                f for f in all_findings
                if isinstance(f, dict) and f.get("actionability") in ("review_only",)
                and (f.get("suggestion") or f.get("recommendation"))
            ]
        if not rephrasable_findings:
            update_rewrite_status(rewrite_id, "failed", error="No rephrasable findings to rewrite")
            return {"status": "failed", "error": "no rephrasable findings"}

        # 3. Run rewrite pipeline
        from poc.rewrite_pipeline import run_rewrite_pipeline
        from app.config import settings

        if not settings.LLM_API_KEY:
            update_rewrite_status(rewrite_id, "failed", error="LLM_API_KEY not configured — rewrite requires an LLM API key")
            return {"status": "failed", "error": "missing LLM API key"}

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_rewrite_pipeline(
                detect_json=report_json,
                output_dir=tmpdir,
                max_passes=3,
                max_detect_loops=0,
                ai_only=False,
                verbose=False,
                api_key=settings.LLM_API_KEY or None,
                model=settings.LLM_MODEL or None,
                base_url=settings.LLM_BASE_URL or None,
            )

            if result["status"] in ("skipped", "clean"):
                update_rewrite_status(rewrite_id, "failed", error=result.get("message", "Rewrite not needed"))
                return {"status": "skipped"}

            # Read files WHILE tmpdir still exists (before with block exits)
            rw = result.get("result")
            md_path = result.get("md_path")
            pdf_path = result.get("pdf_path")

            import logging as _log
            _l = _log.getLogger("rewrite_task")
            _l.info("Pipeline result keys: %s", list(result.keys()))
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
            }

            upload_rewrite_files(scan_id, md_text, pdf_bytes, rewrite_json, rewritten_text)

        # 5. Capture credits
        user_id = scan_job.get("user_id", "") if scan_job else ""
        if user_id:
            capture_rewrite_credits(str(user_id), rewrite_id)

        update_rewrite_status(rewrite_id, "completed")
        return {"status": "completed"}

    except SoftTimeLimitExceeded:
        update_rewrite_status(rewrite_id, "failed", error="Rewrite timed out (10 min limit)")
        return {"status": "failed", "error": "timeout"}
    except Exception as e:
        if self.request.retries < self.max_retries:
            update_rewrite_status(rewrite_id, "retrying", error=str(e))
            raise self.retry(exc=e)
        else:
            update_rewrite_status(rewrite_id, "failed", error=str(e))
            raise
