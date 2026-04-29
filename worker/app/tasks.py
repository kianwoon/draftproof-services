"""Celery tasks — scan_document runs the full detect pipeline."""

import sys
import os
import time
import json
import traceback

# Make poc/ importable from worker/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from .celery_app import app
from .config import settings
from .storage import upload_report_files
from .db import get_scan_job, update_job_status, capture_credits


@app.task(bind=True, max_retries=2, default_retry_delay=30)
def scan_document(self, job_id: str, input_text: str):
    """Run detect pipeline on text, upload results to R2, update DB."""
    job = get_scan_job(job_id)
    if not job:
        raise ValueError(f"scan_job {job_id} not found")

    user_id = str(job["user_id"])
    word_count = job["word_count"] or len(input_text.split())
    verbose = job.get("is_verbose", False)

    update_job_status(job_id, "running")

    try:
        t0 = time.time()

        # Import POC modules (deferred so Celery worker starts fast)
        from poc.detect.run import DetectionRunner
        from poc.report.report import ReportBuilder, report_to_dict
        from poc.report.render import render_report
        from poc.report.pdf import render_pdf

        # 1. Run detection
        runner = DetectionRunner()
        det_report = runner.run_all(input_text)

        # 2. Build report
        builder = ReportBuilder()
        builder.add_detection_report(det_report)
        if det_report.postprocess_results:
            builder.add_postprocess_results(det_report.postprocess_results)
        builder.set_meta(scan_time=time.time() - t0, original_text=input_text)
        draft_report = builder.build()

        # 3. Generate outputs
        md_text = render_report(draft_report, verbose=verbose)
        result_dict = report_to_dict(draft_report)

        # 4. Render PDF
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = tmp.name
        render_pdf(md_text, pdf_path)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        os.unlink(pdf_path)

        # 5. Upload to R2
        urls = upload_report_files(job_id, md_text, pdf_bytes, result_dict)

        # 6. Update DB
        total_findings = sum(len(v) for v in draft_report.findings_by_tier.values())
        report_urls = json.dumps({
            "md": urls.get("md"),
            "pdf": urls.get("pdf"),
            "json": urls.get("json"),
        })
        update_job_status(
            job_id,
            "completed",
            tier=draft_report.overall_tier.value if hasattr(draft_report.overall_tier, "value") else str(draft_report.overall_tier),
            finding_count=total_findings,
            report_urls=report_urls,
        )

        # 7. Billing
        try:
            capture_credits(user_id, job_id, word_count)
        except Exception:
            pass

        tier_val = draft_report.overall_tier.value if hasattr(draft_report.overall_tier, "value") else str(draft_report.overall_tier)
        return {"status": "completed", "job_id": job_id, "tier": tier_val}

    except Exception as exc:
        tb = traceback.format_exc()
        update_job_status(job_id, "failed", error=tb[:2000])
        raise self.retry(exc=exc)
