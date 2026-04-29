"""Celery tasks — scan_document runs the full detect pipeline."""

import sys
import os
import time
import json
import traceback

# Make poc/ importable — on Koyeb: /app/poc/, locally: ../../poc/
_app_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_app_dir, "..", "..")
if _repo_root not in sys.path:
    sys.path.insert(0, os.path.abspath(_repo_root))

from .celery_app import app
from .config import settings
from .storage import upload_report_files
from .db import get_scan_job, update_job_status, capture_credits


@app.task(bind=True, max_retries=2, default_retry_delay=30)
def scan_document(self, job_id: str, text: str) -> dict:
    """Run the full detect pipeline on text and store results."""
    try:
        update_job_status(job_id, "processing")

        from poc.detect.run import run_detection
        from poc.report.render import render_markdown
        from poc.report.pdf import render_pdf
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_detection(text)

            tier = results.get("overall_tier", "unrated")
            findings = results.get("findings", [])
            finding_count = len(findings)

            md_text = render_markdown(results)
            md_path = os.path.join(tmpdir, "report.md")
            pdf_path = os.path.join(tmpdir, "report.pdf")

            with open(md_path, "w") as f:
                f.write(md_text)

            pdf_path = render_pdf(md_text, pdf_path)

            urls = upload_report_files(job_id, md_path, pdf_path)

            capture_credits(job_id, results)
            update_job_status(
                job_id,
                "completed",
                tier=tier,
                finding_count=finding_count,
                report_md_url=urls.get("md"),
                report_pdf_url=urls.get("pdf"),
                results_json=results,
            )

            return {"status": "completed", "tier": tier, "findings": finding_count}

    except Exception as e:
        update_job_status(job_id, "failed", error=str(e))
        raise self.retry(exc=e)
