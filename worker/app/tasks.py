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
from .db import get_scan_job, update_job_status, capture_credits


@app.task(bind=True, max_retries=2, default_retry_delay=30)
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
                finding_count=finding_count,
                report_urls=report_urls,
            )

            return {"status": "completed", "tier": tier, "findings": finding_count}

    except Exception as e:
        if self.request.retries < self.max_retries:
            update_job_status(job_id, "retrying", error=str(e))
        else:
            update_job_status(job_id, "failed", error=str(e))
        raise self.retry(exc=e)
