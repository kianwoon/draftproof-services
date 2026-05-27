"""Celery tasks — scan_document runs the full detect pipeline."""
# v2: scan progress published to Redis via publish_scan_progress

import sys
import os
import json
import time
import logging
import threading
import hashlib

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
    get_rewrite_job,
    get_rewrite_user_email,
    is_rewrite_canceled,
    claim_rewrite_job,
    update_rewrite_status,
    capture_rewrite_credits,
    release_rewrite_credits,
)
from .email_service import send_rewrite_completion_email, send_scan_completion_email
from .rewrite_scan_compaction import compact_rewrite_scan_summary
from celery.signals import worker_process_init
from celery.exceptions import SoftTimeLimitExceeded

logger = logging.getLogger(__name__)

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

NON_BILLABLE_REWRITE_OUTCOMES = {
    "clean",
    "mitigation_failed_no_safe_candidate",
    "needs_author_context",
    "no_safe_rewrite_applied",
    "original_preserved",
    "partial_candidate_not_strict_safe",
    "skipped",
    "topk_blocked",
}

EXTERNAL_REVIEW_REWRITE_STATUSES = {
    "rewrite_candidate_generated_needs_external_review",
    "rewrite_candidate_generated_needs_author_review",
}

EXTERNAL_REVIEW_REWRITE_WARNINGS = {
    "best_candidate_requires_external_review",
    "author_proxy_candidate_requires_review",
}


def _selected_rewrite_pipeline(settings_obj) -> str:
    """Return the first enabled rewrite pipeline in production priority order."""

    if getattr(settings_obj, "DRAFTPROOF_REWRITE_V6_ENABLED", False):
        return "v6"
    if getattr(settings_obj, "DRAFTPROOF_REWRITE_V5_ENABLED", False):
        return "v5"
    if getattr(settings_obj, "DRAFTPROOF_REWRITE_V4_ENABLED", False):
        return "v4"
    if getattr(settings_obj, "DRAFTPROOF_REWRITE_V3_ENABLED", False):
        return "v3"
    if getattr(settings_obj, "DRAFTPROOF_REWRITE_V2_ENABLED", False):
        return "v2"
    return "legacy"


def _runtime_code_fingerprint() -> dict:
    """Return non-secret evidence of the worker code actually imported."""
    fingerprint = {
        "debug_export_version": REWRITE_DEBUG_EXPORT_VERSION,
        "runtime_code_sha_env": os.environ.get("DRAFTPROOF_RUNTIME_CODE_SHA"),
        "worker_tasks_file": os.path.abspath(__file__),
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
        stat = os.stat(__file__)
        fingerprint["worker_tasks_size"] = stat.st_size
        fingerprint["worker_tasks_mtime"] = int(stat.st_mtime)
        with open(__file__, "rb") as handle:
            fingerprint["worker_tasks_sha256_12"] = hashlib.sha256(
                handle.read()
            ).hexdigest()[:12]
    except OSError as exc:
        fingerprint["worker_tasks_fingerprint_error"] = str(exc)

    return fingerprint


class RewriteCanceled(BaseException):
    """Raised inside a worker task when the user cancels a rewrite cooperatively.

    This intentionally inherits from BaseException so broad pipeline-level
    ``except Exception`` recovery blocks do not swallow a user cancellation.
    """


def _normalized_billable_text(value) -> str:
    return " ".join(str(value or "").split())


def _rewrite_billing_decision(pipeline_result: dict, rewrite_json: dict) -> dict:
    """Return whether a rewrite reservation should be captured.

    Billing is tied to delivered rewritten content, not merely to a worker task
    reaching the artifact-upload phase.
    """
    pipeline_result = pipeline_result if isinstance(pipeline_result, dict) else {}
    rewrite_json = rewrite_json if isinstance(rewrite_json, dict) else {}
    summary = rewrite_json.get("summary") if isinstance(rewrite_json.get("summary"), dict) else {}

    status = str(pipeline_result.get("status") or rewrite_json.get("status") or "").strip()
    outcome = str(summary.get("outcome") or summary.get("strict_goal_status") or "").strip()
    strict_goal_status = str(summary.get("strict_goal_status") or "").strip()
    public_warning = str(summary.get("public_candidate_warning") or "").strip()
    normalized_status = status.lower()
    normalized_outcome = outcome.lower()
    normalized_strict_goal_status = strict_goal_status.lower()
    normalized_public_warning = public_warning.lower()

    if normalized_status in NON_BILLABLE_REWRITE_OUTCOMES:
        return {
            "billable": False,
            "reason": f"non_billable_status:{normalized_status}",
            "status": status,
            "outcome": outcome,
        }
    if normalized_status in EXTERNAL_REVIEW_REWRITE_STATUSES:
        return {
            "billable": False,
            "reason": f"external_review_required_status:{normalized_status}",
            "status": status,
            "outcome": outcome,
        }
    if normalized_outcome in NON_BILLABLE_REWRITE_OUTCOMES:
        return {
            "billable": False,
            "reason": f"non_billable_outcome:{normalized_outcome}",
            "status": status,
            "outcome": outcome,
        }
    if normalized_outcome in EXTERNAL_REVIEW_REWRITE_STATUSES:
        return {
            "billable": False,
            "reason": f"external_review_required_outcome:{normalized_outcome}",
            "status": status,
            "outcome": outcome,
        }
    if normalized_strict_goal_status in NON_BILLABLE_REWRITE_OUTCOMES:
        return {
            "billable": False,
            "reason": f"non_billable_strict_goal:{normalized_strict_goal_status}",
            "status": status,
            "outcome": outcome,
        }
    if summary.get("best_candidate_external_review_required") is True:
        return {
            "billable": False,
            "reason": "external_review_required",
            "status": status,
            "outcome": outcome,
        }
    if normalized_public_warning in EXTERNAL_REVIEW_REWRITE_WARNINGS:
        return {
            "billable": False,
            "reason": f"external_review_required_warning:{normalized_public_warning}",
            "status": status,
            "outcome": outcome,
        }
    if summary.get("rollback_applied") or summary.get("no_text_change"):
        return {
            "billable": False,
            "reason": "original_text_preserved",
            "status": status,
            "outcome": outcome,
        }

    original_text = _normalized_billable_text(rewrite_json.get("original_text"))
    final_text = _normalized_billable_text(rewrite_json.get("final_text") or summary.get("final_text"))
    text_changed = bool(original_text and final_text and original_text != final_text)
    if not final_text:
        reason = "empty_final_text"
    elif not original_text:
        reason = "missing_original_text"
    elif not text_changed:
        reason = "final_text_unchanged"
    else:
        reason = "rewritten_content_delivered"

    return {
        "billable": text_changed,
        "reason": reason,
        "status": status,
        "outcome": outcome,
        "text_changed": text_changed,
    }


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
    human = _pct_score(
        contribution.get("human_contribution_ratio")
        if contribution.get("human_contribution_ratio") is not None
        else 100 - ai_transformation
    )
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


def _scan_email_ai_signal_score(report_json: dict | None) -> float | int | None:
    if not isinstance(report_json, dict):
        return None
    badge = report_json.get("ai_risk_badge") or {}
    transformation = badge.get("transformation_classification") or {}
    features = transformation.get("features") or {}
    if features.get("calibrated_ai_risk") is not None:
        return features.get("calibrated_ai_risk")
    intelligence = report_json.get("scan_intelligence") or {}
    contribution = ((intelligence.get("transformation") or {}).get("contribution") or {})
    if contribution.get("calibrated_ai_risk") is not None:
        return contribution.get("calibrated_ai_risk")
    return contribution.get("adjusted_ai_risk")


def _configure_torch_threads(torch) -> int | None:
    """Apply explicit Torch CPU thread settings when configured."""
    raw_threads = (
        os.environ.get("TORCH_NUM_THREADS")
        or os.environ.get("OMP_NUM_THREADS")
        or os.environ.get("MKL_NUM_THREADS")
    )
    if not raw_threads:
        return None
    try:
        threads = int(raw_threads)
    except ValueError:
        logger.warning("Invalid TORCH_NUM_THREADS/OMP_NUM_THREADS value: %r", raw_threads)
        return None
    if threads <= 0:
        return None
    try:
        torch.set_num_threads(threads)
    except Exception:
        logger.warning("Failed to set torch num threads to %s", threads, exc_info=True)
    try:
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    except RuntimeError:
        # PyTorch can reject this after parallel work starts. Main inference
        # threads are still controlled by set_num_threads above.
        pass
    except Exception:
        logger.warning("Failed to set torch interop threads", exc_info=True)
    return threads


@worker_process_init.connect
def _preload_scan_models(**_kwargs):
    """Optionally warm scan models inside the Celery worker child process.

    Celery kills a child if this init signal blocks for too long, so expensive
    model warmup must stay opt-in. Normal tasks still lazy-load cached models on
    first use without tripping the worker startup watchdog.
    """
    enabled = os.environ.get("DRAFTPROOF_PRELOAD_PREDICTABILITY", "0").lower()
    if enabled not in {"0", "false", "no"}:
        try:
            _preload_predictability_model()
        except Exception:
            logger.warning("Failed to preload predictability model in worker child", exc_info=True)

    semantic_enabled = os.environ.get("DRAFTPROOF_PRELOAD_SEMANTIC", "0").lower()
    if semantic_enabled not in {"0", "false", "no"}:
        try:
            _preload_semantic_model()
        except Exception:
            logger.warning("Failed to preload semantic embedding model in worker child", exc_info=True)


def _preload_predictability_model():
    """Warm the GPT-2 scanner inside the Celery worker child process."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import predictability.scanner as scanner_module

        requested_threads = _configure_torch_threads(torch)
        model_name = scanner_module.resolve_predictability_model_name(
            os.environ.get("PREDICTABILITY_MODEL", "gpt2")
        )
        if scanner_module._PRELOADED_MODEL is not None:
            return
        logger.info("Preloading predictability model in worker child: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        scanner_module._PRELOADED_MODEL = model
        scanner_module._PRELOADED_TOKENIZER = tokenizer
        scanner_module._PRELOADED_MODEL_NAME = model_name
        try:
            mkl_available = bool(getattr(torch.backends, "mkl", None) and torch.backends.mkl.is_available())
            openmp_available = bool(getattr(torch.backends, "openmp", None) and torch.backends.openmp.is_available())
            logger.info(
                "[preload] Torch: version=%s MKL=%s OpenMP=%s threads=%s requested_threads=%s",
                torch.__version__,
                mkl_available,
                openmp_available,
                torch.get_num_threads(),
                requested_threads,
            )
            sample_sentences = [
                "People test the process before they rely on the final result.",
                "The reviewer checks the earlier step against the latest decision.",
                "A small condition can change once the system starts to respond.",
                "The task needs a visible process, not only a final result.",
                "During review, a participant may explain the goal indirectly.",
                "The facilitator can slow the task down at the decision stage.",
                "Writers need to connect the reason, constraint, and outcome.",
                "A draft shows the mistake before the final version does.",
                "A short change can alter how the reader understands the point.",
                "The working context makes technical decisions visible quickly.",
            ]
            encoded = tokenizer(
                sample_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(os.environ.get("DRAFTPROOF_PREDICTABILITY_MAX_TOKENS", "384")),
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            bench_t0 = time.monotonic()
            with torch.no_grad():
                model(**encoded)
            bench_ms = (time.monotonic() - bench_t0) * 1000
            logger.info(
                "[preload] Benchmark: %d sentences, total=%.0fms, avg=%.1fms/sent",
                len(sample_sentences),
                bench_ms,
                bench_ms / max(len(sample_sentences), 1),
            )
        except Exception:
            logger.warning("[preload] Benchmark failed", exc_info=True)
        logger.info("Predictability model preloaded in worker child: %s", model_name)
    except Exception:
        raise


def _preload_semantic_model():
    """Warm the sentence embedding scanner inside the Celery worker child process."""
    from sentence_transformers import SentenceTransformer
    import detect.semantic_shape as semantic_module

    model_name = semantic_module.resolve_semantic_embedding_model_name(
        os.environ.get("SEMANTIC_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    if semantic_module._PRELOADED_EMBEDDER is not None:
        return
    logger.info("Preloading semantic embedding model in worker child: %s", model_name)
    embedder = SentenceTransformer(
        model_name,
        cache_folder=os.environ.get("HF_HOME"),
        local_files_only=True,
    )
    sample_sentences = [
        "Students practise the sectioning pattern before they cut.",
        "The stylist checks the guide length against the previous section.",
        "The lesson needs a visible process, not only a final result.",
        "Learners need to connect the angle, tension, and design line.",
    ]
    bench_t0 = time.monotonic()
    embedder.encode(sample_sentences, convert_to_numpy=True, normalize_embeddings=True)
    bench_ms = (time.monotonic() - bench_t0) * 1000
    semantic_module._PRELOADED_EMBEDDER = embedder
    semantic_module._PRELOADED_MODEL_NAME = model_name
    logger.info(
        "[preload] Semantic benchmark: %d sentences, total=%.0fms, avg=%.1fms/sent",
        len(sample_sentences),
        bench_ms,
        bench_ms / max(len(sample_sentences), 1),
    )
    logger.info("Semantic embedding model preloaded in worker child: %s", model_name)


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
    )
    compact_summary = {}
    for key in compact_summary_keys:
        if key not in summary:
            continue
        if key == "candidate_ledger":
            compact_summary[key] = _compact_rewrite_candidate_ledger(summary.get(key))
        elif key in {"detect_scan_original_saved", "detect_scan_original", "detect_scan_rewritten"}:
            compact_summary[key] = compact_rewrite_scan_summary(summary.get(key))
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


def _fetch_r2_json(s3, bucket: str, key: str) -> dict:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read())


def _seed_text_word_count(text: str) -> int:
    return len(str(text or "").split())


def _historical_seed_is_full_document_candidate(text: str, original_text: str) -> bool:
    text_words = _seed_text_word_count(text)
    original_words = _seed_text_word_count(original_text)
    if text_words <= 0:
        return False
    if original_words <= 0:
        return text_words >= 120
    if original_words < 120:
        return text_words >= max(4, int(original_words * 0.55))
    return text_words >= max(120, int(original_words * 0.55))


def _historical_seed_text_from_entry(entry) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if not isinstance(entry, dict):
        return ""
    for key in ("text", "candidate_text", "rewritten_document", "final_text", "rewritten_text"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    selected = entry.get("selected") if isinstance(entry.get("selected"), dict) else {}
    accepted = entry.get("accepted") if isinstance(entry.get("accepted"), dict) else {}
    for nested in (selected, accepted):
        if not nested:
            continue
        text = _historical_seed_text_from_entry(nested)
        if text:
            return text
    return ""


def _append_historical_seed_candidates(candidates: list[str], entries, original_text: str) -> None:
    if isinstance(entries, dict):
        text = _historical_seed_text_from_entry(entries)
        if _historical_seed_is_full_document_candidate(text, original_text):
            candidates.append(text)
        return
    if not isinstance(entries, list):
        return
    ordered = sorted(
        [entry for entry in entries if isinstance(entry, (dict, str))],
        key=lambda entry: (
            entry.get("rank") if isinstance(entry, dict) and isinstance(entry.get("rank"), (int, float)) else 999,
        ),
    )
    for entry in ordered:
        text = _historical_seed_text_from_entry(entry)
        if _historical_seed_is_full_document_candidate(text, original_text):
            candidates.append(text)


def _dedupe_historical_seed_texts(candidates: list[str], original_text: str, *, limit: int) -> list[str]:
    original_normalized = _normalized_billable_text(original_text)
    seen: set[str] = set()
    seeds: list[str] = []
    for value in candidates:
        text = str(value or "").strip()
        normalized = _normalized_billable_text(text)
        if (
            not normalized
            or normalized == original_normalized
            or normalized in seen
            or not _historical_seed_is_full_document_candidate(text, original_text)
        ):
            continue
        seen.add(normalized)
        seeds.append(text)
        if len(seeds) >= max(1, int(limit or 1)):
            break
    return seeds


def _historical_rewrite_seed_texts(rewrite_json: dict | None, original_text: str, *, limit: int = 3) -> list[str]:
    if not isinstance(rewrite_json, dict):
        return []
    summary = rewrite_json.get("summary") if isinstance(rewrite_json.get("summary"), dict) else {}
    if _rewrite_has_paragraph_obligation_hard_stop(summary):
        return []
    candidates: list[str] = []
    _append_historical_seed_candidates(candidates, rewrite_json.get("candidate_ledger"), original_text)
    _append_historical_seed_candidates(candidates, summary.get("candidate_ledger"), original_text)
    layers = summary.get("rewrite_layers") if isinstance(summary.get("rewrite_layers"), dict) else {}
    v5_layer = layers.get("v5_residual_cluster_comb") if isinstance(layers.get("v5_residual_cluster_comb"), dict) else {}
    _append_historical_seed_candidates(candidates, v5_layer.get("seed_candidate_rows"), original_text)
    _append_historical_seed_candidates(candidates, v5_layer.get("seed_recovery"), original_text)
    _append_historical_seed_candidates(candidates, v5_layer.get("global_best_fallback"), original_text)
    _append_historical_seed_candidates(candidates, v5_layer.get("accepted_checkpoints"), original_text)
    candidates.extend([
        rewrite_json.get("final_text"),
        summary.get("final_text"),
        rewrite_json.get("rewritten_text"),
        summary.get("rewritten_document"),
    ])
    selected = summary.get("selected_candidate") if isinstance(summary.get("selected_candidate"), dict) else {}
    candidates.extend([
        selected.get("candidate_text"),
        selected.get("text"),
    ])
    return _dedupe_historical_seed_texts(candidates, original_text, limit=limit)


def _rewrite_has_paragraph_obligation_hard_stop(summary: dict) -> bool:
    if not isinstance(summary, dict):
        return False
    hard_stop = summary.get("paragraph_obligation_hard_stop")
    if isinstance(hard_stop, dict) and hard_stop.get("active"):
        return True
    if str(summary.get("no_text_change_reason") or "") == "v5_unresolved_paragraph_findings":
        return True
    generation_status = summary.get("candidate_generation_status")
    return (
        isinstance(generation_status, dict)
        and str(generation_status.get("reason") or "") == "unresolved_paragraph_findings"
    )


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

                paragraph_explanations = generate_paragraph_explanations(results_json)
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
            except Exception:
                logger.warning("Paragraph explanation generation failed for %s", job_id, exc_info=True)

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
                    ai_signal_score=_scan_email_ai_signal_score(results_json),
                    writing_score=writing_score,
                    finding_count=finding_count,
                    pdf_bytes=pdf_bytes,
                    settings=settings,
                )
            except Exception:
                logger.warning("Scan completion email hook failed for %s", job_id, exc_info=True)

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
                elif pipeline_choice == "v5":
                    from rewrite_v5 import run_rewrite_pipeline_v5
                    v5_rewrite_model = (
                        settings.DRAFTPROOF_REWRITE_V5_MODEL
                        or settings.DRAFTPROOF_REWRITE_V4_MODEL
                        or rewrite_model
                    )
                    result = run_rewrite_pipeline_v5(
                        detect_json=report_json,
                        output_dir=tmpdir,
                        api_key=llm_api_key or None,
                        model=v5_rewrite_model,
                        base_url=settings.LLM_BASE_URL or None,
                        progress_callback=report_rewrite_progress,
                        checkpoint_callback=persist_rewrite_checkpoint,
                        seed_candidate_texts=previous_rewrite_seed_texts,
                        cancellation_check=raise_if_canceled,
                    )
                elif pipeline_choice == "v4":
                    from rewrite_v4 import run_rewrite_pipeline_v4
                    v4_rewrite_model = settings.DRAFTPROOF_REWRITE_V4_MODEL or rewrite_model
                    result = run_rewrite_pipeline_v4(
                        detect_json=report_json,
                        output_dir=tmpdir,
                        api_key=llm_api_key or None,
                        model=v4_rewrite_model,
                        base_url=settings.LLM_BASE_URL or None,
                        progress_callback=report_rewrite_progress,
                        cancellation_check=raise_if_canceled,
                    )
                elif pipeline_choice == "v3":
                    from rewrite_v3 import run_rewrite_pipeline_v3
                    result = run_rewrite_pipeline_v3(
                        detect_json=report_json,
                        output_dir=tmpdir,
                        api_key=llm_api_key or None,
                        model=rewrite_model,
                        base_url=settings.LLM_BASE_URL or None,
                        progress_callback=report_rewrite_progress,
                        cancellation_check=raise_if_canceled,
                    )
                elif pipeline_choice == "v2":
                    from rewrite_v2 import run_rewrite_pipeline_v2
                    result = run_rewrite_pipeline_v2(
                        detect_json=report_json,
                        output_dir=tmpdir,
                        api_key=llm_api_key or None,
                        model=rewrite_model,
                        base_url=settings.LLM_BASE_URL or None,
                        progress_callback=report_rewrite_progress,
                        cancellation_check=raise_if_canceled,
                    )
                else:
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
