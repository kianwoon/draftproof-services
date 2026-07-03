"""Celery tasks — scan_document runs the full detect pipeline."""
# v2: scan progress published to Redis via publish_scan_progress

import sys
import os
import json
import time
import logging
import threading

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
    get_scan_user_email,
    update_job_status,
    capture_credits,
    release_scan_credits,
    refund_free_scan,
    get_rewrite_job,
    get_rewrite_user_email,
    is_rewrite_canceled,
    claim_rewrite_job,
    update_rewrite_status,
    capture_rewrite_credits,
    release_rewrite_credits,
)
from .email_service import send_rewrite_completion_email, send_scan_completion_email
from celery.exceptions import SoftTimeLimitExceeded

# Importing model_preload registers its @worker_process_init.connect handler.
from . import model_preload  # noqa: F401
from .rewrite_billing import (
    _dedupe_historical_seed_texts,
    _historical_rewrite_seed_texts,
    _rewrite_billing_decision,
)
from .rewrite_debug import (
    _bounded_json_debug_log,
    _bounded_rewrite_json_payload,
    _build_rewrite_debug_log,
    _extract_rewrite_scan_summary,
    _serialize_effective_rewrite_plan,
)

logger = logging.getLogger(__name__)

# Critical Thinking reflective questions: default ON in the worker. Set here (a
# git-pulled path) so it ships via the RELIABLE code-deploy, not only via
# entrypoint.sh -- the entrypoint flag needs a full image rebuild, whose Koyeb
# deploy step is fragile (HTTP 400 on the partial-definition PATCH). Overridable by
# a real Koyeb env value (setdefault won't clobber an explicit DRAFTPROOF_..._=0).
os.environ.setdefault("DRAFTPROOF_CRITICAL_THINKING_QUESTIONS", "1")


def _selected_rewrite_pipeline(settings_obj) -> str:
    """Return the enabled rewrite pipeline (v6 only; legacy fallback)."""

    if getattr(settings_obj, "DRAFTPROOF_REWRITE_V6_ENABLED", False):
        return "v6"
    return "legacy"


def _best_effort_scan_status_update(job_id: str, status: str, **fields) -> bool:
    """Persist scan status when possible without blocking Celery retry/failure handling."""
    try:
        update_job_status(job_id, status, **fields)
        return True
    except Exception:
        logger.warning("Scan status update failed for %s while setting %s", job_id, status, exc_info=True)
        return False


class RewriteCanceled(BaseException):
    """Raised inside a worker task when the user cancels a rewrite cooperatively.

    This intentionally inherits from BaseException so broad pipeline-level
    ``except Exception`` recovery blocks do not swallow a user cancellation.
    """


def _fetch_r2_json(s3, bucket: str, key: str) -> dict:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read())


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
        claimed = update_job_status(
            job_id,
            "processing",
            progress_percent=10,
            progress_message="Preparing scan",
        )
        if not claimed:
            # The job is already resolved (failed/completed/canceled) — e.g. the API
            # stale-recovery failed it, or this is a Celery redelivery (acks_late) of a
            # run that already finished. Abort rather than re-run: redoing the work would
            # waste the most expensive part of the pipeline, re-upload to R2, and re-send
            # the completion email for a job the user has already seen resolved (H1).
            logger.info("scan_document %s: job already resolved; skipping redelivered/raced run", job_id)
            return {"status": "skipped", "reason": "already_resolved"}
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
            import predictability.scanner as scanner_module
            model_name = scanner_module.resolve_predictability_model_name(
                os.environ.get("PREDICTABILITY_MODEL", "gpt2")
            )
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

            paragraph_explanations = None
            try:
                report_progress(96, "Explaining paragraph findings")
                from poc.report.paragraph_explainer import generate_paragraph_explanations
                from poc.report.render import render_report
                from poc.report.pdf import render_pdf

                paragraph_explanations = generate_paragraph_explanations(
                    results_json,
                    api_key=(settings.LLM_API_KEY or settings.OPENROUTER_API_KEY or None),
                    base_url=(settings.LLM_BASE_URL or None),
                    model=(
                        settings.DRAFTPROOF_V6_PLANNER_MODEL
                        or settings.DRAFTPROOF_PLANNER_MODEL
                        or settings.DRAFTPROOF_REWRITE_V5_PLANNER_MODEL
                        or settings.LLM_MODEL
                        or None
                    ),
                )
                if paragraph_explanations is not None:
                    results_json["paragraph_explanations"] = paragraph_explanations
                    draft_report = result.get("report")
                    if draft_report is not None:
                        draft_report.paragraph_explanations = paragraph_explanations
                        md_text = render_report(draft_report, verbose=True)
                        explained_pdf_path = os.path.join(tmpdir, "draftproof_explained.pdf")
                        render_pdf(md_text, explained_pdf_path)
                        with open(explained_pdf_path, "rb") as f:
                            pdf_bytes = f.read()
            except Exception as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code in {401, 403}:
                    logger.warning(
                        "Paragraph explanation skipped for %s: LLM provider rejected credentials "
                        "(status=%s). Check OPENROUTER_API_KEY, LLM_BASE_URL, and planner model settings.",
                        job_id,
                        status_code,
                    )
                else:
                    logger.warning("Paragraph explanation generation failed for %s", job_id, exc_info=True)

            # Heartbeat after the (silent, possibly multi-minute) paragraph-explainer LLM
            # call so stale-recovery sees a live worker before the next enrichment step.
            # Without this the only beats in the tail are 96 (pre-enrichment) and 97
            # (pre-upload); a slow enrichment could otherwise be reaped as stale (H1/M1).
            report_progress(96, "Finalizing report")

            # Phase-2 LLM enrichment for Critical Thinking Control (kill-switch, DEFAULT OFF
            # via DRAFTPROOF_CRITICAL_THINKING_LLM). Adds the alternative_comparison +
            # reflection dimensions and agnostic per-sentence highlights to the additive
            # critical_thinking_control diagnosis. Fail-open: never blocks or alters the scan.
            try:
                from poc.detect.critical_thinking_llm import assess_critical_thinking
                ctc = (results_json.get("ai_risk_badge") or {}).get("critical_thinking_control")
                if isinstance(ctc, dict):
                    enrichment = assess_critical_thinking(
                        results_json,
                        api_key=(settings.LLM_API_KEY or settings.OPENROUTER_API_KEY or settings.CEREBRAS_API_KEY or None),
                        base_url=(settings.LLM_BASE_URL or None),
                    )
                    if enrichment is not None:
                        ctc["llm_dimensions"] = enrichment.get("llm_dimensions")
                        ctc["highlights"] = enrichment.get("highlights") or []
                        ctc["llm_model"] = enrichment.get("model")
            except Exception:
                logger.warning("Critical Thinking LLM enrichment skipped for %s", job_id, exc_info=True)

            # Reflective questions (kill-switch DRAFTPROOF_CRITICAL_THINKING_QUESTIONS, DEFAULT
            # OFF). Anchored, dimension-steered questions that replace the score as the user-facing
            # surface. Fail-open: never blocks or alters the scan.
            try:
                from poc.detect.critical_thinking_llm import generate_reflective_questions
                ctc = (results_json.get("ai_risk_badge") or {}).get("critical_thinking_control")
                if isinstance(ctc, dict):
                    questions = generate_reflective_questions(
                        results_json,
                        api_key=(settings.LLM_API_KEY or settings.OPENROUTER_API_KEY or settings.CEREBRAS_API_KEY or None),
                        base_url=(settings.LLM_BASE_URL or None),
                    )
                    if questions is not None and questions.get("questions"):
                        ctc["questions"] = questions.get("questions")
                        ctc["questions_model"] = questions.get("model")
                        # The emailed/R2 PDF renders from the DraftReport OBJECT (not
                        # results_json) and was already rendered above. Mirror the questions
                        # onto the object and re-render so the PDF includes the Critical
                        # Thinking section. Keeps paragraph_explanations (already on the object).
                        _dr = result.get("report")
                        _dr_badge = getattr(_dr, "ai_risk_badge", None)
                        _dr_ctc = _dr_badge.get("critical_thinking_control") if isinstance(_dr_badge, dict) else None
                        if isinstance(_dr_ctc, dict):
                            _dr_ctc["questions"] = questions.get("questions")
                            from poc.report.render import render_report
                            from poc.report.pdf import render_pdf
                            md_text = render_report(_dr, verbose=True)
                            _q_pdf = os.path.join(tmpdir, "draftproof_questions.pdf")
                            render_pdf(md_text, _q_pdf)
                            with open(_q_pdf, "rb") as f:
                                pdf_bytes = f.read()
            except Exception:
                logger.warning("Critical Thinking questions skipped for %s", job_id, exc_info=True)

            report_progress(97, "Uploading report files")
            urls = upload_report_files(job_id, md_text, pdf_bytes, results_json, paragraph_explanations)

            report_urls = {
                "md": urls.get("md"),
                "pdf": urls.get("pdf"),
                "json": urls.get("json"),
                "paragraph_explanations": urls.get("paragraph_explanations"),
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
            try:
                recipient_email = get_scan_user_email(job_id)
                send_scan_completion_email(
                    recipient_email=recipient_email,
                    scan_id=job_id,
                    tier=tier,
                    ai_score=ai_score,
                    authorship_rating_label=badge.get("authorship_rating_label") if badge else None,
                    external_estimate=badge.get("external_detector_estimate") if badge else None,
                    writing_score=writing_score,
                    finding_count=finding_count,
                    submission_risk=badge.get("submission_risk") if badge else None,
                    policy_risk=badge.get("policy_risk") if badge else None,
                    pdf_bytes=pdf_bytes,
                    settings=settings,
                )
            except Exception:
                logger.warning("Scan completion email hook failed for %s", job_id, exc_info=True)

            return {"status": "completed", "tier": tier, "findings": finding_count}

    except SoftTimeLimitExceeded:
        release_scan_credits(job_id)
        refund_free_scan(job_id)
        _best_effort_scan_status_update(
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
            _best_effort_scan_status_update(
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
            release_scan_credits(job_id)
            refund_free_scan(job_id)
            _best_effort_scan_status_update(
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
    from .storage import upload_rewrite_checkpoint, upload_rewrite_files, _client as _r2_client
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

    def raise_if_canceled() -> None:
        if is_rewrite_canceled(rewrite_id):
            raise RewriteCanceled()

    billing_decision = {
        "billable": False,
        "reason": "rewrite_not_completed",
    }
    scan_job = None
    last_rewrite_checkpoint: dict = {}

    def checkpoint_markdown(checkpoint_json: dict) -> str:
        summary = checkpoint_json.get("summary") if isinstance(checkpoint_json.get("summary"), dict) else {}
        final_text = str(checkpoint_json.get("final_text") or "")
        scores = ((summary.get("v5_scores") or {}).get("deltas") or {}) if isinstance(summary.get("v5_scores"), dict) else {}
        return "\n\n".join([
            "# Rewrite checkpoint",
            f"Status: {checkpoint_json.get('status') or 'partial_candidate_not_strict_safe'}",
            f"AI delta: {scores.get('ai_delta', '-')}",
            "This is the latest scanner-accepted rewrite saved before the full pipeline completed.",
            "## Rewritten content",
            final_text,
        ])

    def persist_rewrite_checkpoint(checkpoint_json: dict) -> None:
        nonlocal last_rewrite_checkpoint
        if not isinstance(checkpoint_json, dict):
            return
        rewritten_text = str(checkpoint_json.get("final_text") or "")
        if not rewritten_text.strip():
            return
        last_rewrite_checkpoint = checkpoint_json
        upload_rewrite_checkpoint(
            scan_id,
            checkpoint_json,
            rewritten_text,
            checkpoint_markdown(checkpoint_json),
        )

    def complete_from_rewrite_checkpoint(reason: str) -> dict | None:
        nonlocal billing_decision
        if not last_rewrite_checkpoint:
            return None
        checkpoint_json = dict(last_rewrite_checkpoint)
        checkpoint_json["checkpoint_recovery_reason"] = reason
        rewritten_text = str(checkpoint_json.get("final_text") or "")
        upload_rewrite_checkpoint(
            scan_id,
            checkpoint_json,
            rewritten_text,
            checkpoint_markdown(checkpoint_json),
        )
        billing_decision = _rewrite_billing_decision(
            {"status": checkpoint_json.get("status")},
            checkpoint_json,
        )
        checkpoint_status_written = update_rewrite_status(
            rewrite_id,
            "completed",
            progress_percent=100,
            progress_message="Rewrite complete from saved checkpoint",
        )
        if not checkpoint_status_written:
            raise_if_canceled()
            logger.warning("Rewrite %s checkpoint completion status was not written; skipping delivery email", rewrite_id)
            return {"status": "skipped", "reason": "checkpoint_completion_status_not_written"}
        raise_if_canceled()
        user_id = scan_job.get("user_id", "") if scan_job else ""
        if user_id and billing_decision.get("billable"):
            capture_rewrite_credits(str(user_id), rewrite_id)
        else:
            release_rewrite_credits(rewrite_id)
        publish_progress("completed", 100, "Rewrite complete from saved checkpoint")
        checkpoint_pdf_bytes = b""
        try:
            from report.render_rewrite import render_rewrite_report
            from report.pdf import render_pdf

            with tempfile.TemporaryDirectory() as checkpoint_tmpdir:
                checkpoint_pdf_path = os.path.join(checkpoint_tmpdir, "rewrite.pdf")
                checkpoint_md = render_rewrite_report(
                    summary=checkpoint_json.get("summary") or {},
                    sentence_comparison=checkpoint_json.get("sentence_comparison") or [],
                    ai_findings=[],
                    verbose=False,
                    original_text=checkpoint_json.get("original_text") or "",
                    final_text=rewritten_text,
                )
                render_pdf(checkpoint_md, checkpoint_pdf_path)
                with open(checkpoint_pdf_path, "rb") as f:
                    checkpoint_pdf_bytes = f.read()
        except Exception:
            logger.warning("Failed to build checkpoint rewrite completion PDF for %s", rewrite_id, exc_info=True)
        recipient_email = get_rewrite_user_email(rewrite_id)
        send_rewrite_completion_email(
            recipient_email=recipient_email,
            rewrite_id=rewrite_id,
            scan_id=scan_id,
            final_text=rewritten_text,
            pdf_bytes=checkpoint_pdf_bytes,
            settings=settings,
        )
        return {"status": "completed", "checkpoint_recovered": True, "reason": reason}

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

        previous_rewrite_seed_texts: list[str] = []
        try:
            previous_rewrite_json = _fetch_r2_json(
                s3,
                worker_settings.R2_BUCKET_NAME,
                f"reports/{scan_id}/rewrite/rewrite.json",
            )
            previous_rewrite_seed_texts.extend(_historical_rewrite_seed_texts(
                previous_rewrite_json,
                str(report_json.get("input_text") or ""),
                limit=8,
            ))
        except Exception:
            pass
        try:
            previous_checkpoint_json = _fetch_r2_json(
                s3,
                worker_settings.R2_BUCKET_NAME,
                f"reports/{scan_id}/rewrite/checkpoint.json",
            )
            previous_rewrite_seed_texts.extend(_historical_rewrite_seed_texts(
                previous_checkpoint_json,
                str(report_json.get("input_text") or ""),
                limit=8,
            ))
        except Exception:
            pass
        previous_rewrite_seed_texts = _dedupe_historical_seed_texts(
            previous_rewrite_seed_texts,
            str(report_json.get("input_text") or ""),
            limit=8,
        )

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
        llm_api_key = settings.LLM_API_KEY or settings.OPENROUTER_API_KEY or settings.CEREBRAS_API_KEY
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
                "heartbeat_percent": 40,
                "heartbeat_message": "Rewriting AI sections",
            }

            heartbeat_stop = threading.Event()

            def rewrite_heartbeat() -> None:
                while not heartbeat_stop.wait(30.0):
                    publish_progress(
                        "processing",
                        last_rewrite_progress["heartbeat_percent"],
                        last_rewrite_progress["heartbeat_message"],
                    )

            heartbeat_thread = threading.Thread(
                target=rewrite_heartbeat,
                name=f"rewrite-heartbeat-{rewrite_id}",
                daemon=True,
            )

            def report_rewrite_progress(percent: int, message: str) -> None:
                raise_if_canceled()
                normalized_percent = max(40, min(79, int(percent)))
                now = time.monotonic()
                last_rewrite_progress["heartbeat_percent"] = normalized_percent
                last_rewrite_progress["heartbeat_message"] = message
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

            raise_if_canceled()
            heartbeat_thread.start()
            try:
                rewrite_model = (
                    settings.DRAFTPROOF_REWRITE_MODEL_LOCK
                    or settings.DRAFTPROOF_GENERATOR_MODEL
                    or settings.LLM_MODEL
                    or None
                )
                if isinstance(rewrite_model, str) and rewrite_model.strip().lower() in {"0", "false", "no", "off"}:
                    rewrite_model = settings.DRAFTPROOF_GENERATOR_MODEL or settings.LLM_MODEL or None
                pipeline_choice = _selected_rewrite_pipeline(settings)
                if pipeline_choice == "v6":
                    from rewrite_v6 import run_rewrite_pipeline_v6
                    v6_rewrite_model = settings.DRAFTPROOF_V6_WRITER_MODEL or rewrite_model
                    result = run_rewrite_pipeline_v6(
                        detect_json=report_json,
                        output_dir=tmpdir,
                        api_key=llm_api_key or None,
                        model=v6_rewrite_model,
                        base_url=settings.LLM_BASE_URL or None,
                        progress_callback=report_rewrite_progress,
                        cancellation_check=raise_if_canceled,
                    )
                else:
                    from poc.rewrite_pipeline import run_rewrite_pipeline
                    result = run_rewrite_pipeline(
                        detect_json=report_json,
                        output_dir=tmpdir,
                        max_passes=3,
                        max_detect_loops=0,
                        ai_only=True,
                        verbose=False,
                        api_key=llm_api_key or None,
                        model=rewrite_model,
                        base_url=settings.LLM_BASE_URL or None,
                        progress_callback=report_rewrite_progress,
                        cancellation_check=raise_if_canceled,
                    )
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1.0)
            raise_if_canceled()

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
            raise_if_canceled()
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
            billing_decision = _rewrite_billing_decision(result, rewrite_json)
            rewrite_json["billing_decision"] = billing_decision
            rewrite_json = _bounded_rewrite_json_payload(rewrite_json)
            try:
                from report.render_rewrite import render_rewrite_report
                from report.pdf import render_pdf

                md_text = render_rewrite_report(
                    summary=rewrite_json.get("summary") or {},
                    sentence_comparison=rewrite_json.get("sentence_comparison") or [],
                    ai_findings=[],
                    verbose=False,
                    original_text=rewrite_json.get("original_text") or "",
                    final_text=rewrite_json.get("final_text") or "",
                )
                pdf_path = os.path.join(tmpdir, "rewrite.pdf")
                render_pdf(md_text, pdf_path)
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
            except Exception:
                _l.warning("Failed to regenerate rewrite delivery PDF; using pipeline PDF", exc_info=True)

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
            raise_if_canceled()
            upload_rewrite_files(scan_id, md_text, pdf_bytes, rewrite_json, rewritten_text, debug_log)

        # 5. Mark complete, then capture credits and deliver results.
        raise_if_canceled()
        completed_status_written = update_rewrite_status(
            rewrite_id,
            "completed",
            progress_percent=100,
            progress_message="Rewrite complete",
        )
        if not completed_status_written:
            raise_if_canceled()
            logger.warning("Rewrite %s completion status was not written; skipping delivery email", rewrite_id)
            return {"status": "skipped", "reason": "completion_status_not_written"}
        user_id = scan_job.get("user_id", "") if scan_job else ""
        if user_id and billing_decision.get("billable"):
            capture_rewrite_credits(str(user_id), rewrite_id)
        else:
            release_rewrite_credits(rewrite_id)
        publish_progress("completed", 100, "Rewrite complete")
        recipient_email = get_rewrite_user_email(rewrite_id)
        send_rewrite_completion_email(
            recipient_email=recipient_email,
            rewrite_id=rewrite_id,
            scan_id=scan_id,
            final_text=rewritten_text,
            pdf_bytes=pdf_bytes,
            settings=settings,
        )
        return {"status": "completed"}

    except SoftTimeLimitExceeded:
        recovered = complete_from_rewrite_checkpoint("soft_time_limit_exceeded_after_accepted_candidate")
        if recovered:
            return recovered
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
    except RewriteCanceled:
        logger.info("Rewrite %s canceled cooperatively; worker task exiting", rewrite_id)
        publish_progress("canceled", None, "Rewrite canceled", "Rewrite canceled by user")
        return {"status": "canceled"}
    except Exception as e:
        recovered = complete_from_rewrite_checkpoint("exception_after_accepted_candidate")
        if recovered:
            return recovered
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
            original_text=rewrite_json.get("original_text") or "",
            final_text=rewrite_json.get("final_text") or "",
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
