"""Thin client for the live Modal deep-scan detector endpoint
(``modal_endpoints/deberta_large_detector.py``). See ``modal_endpoints/README.md``
for the deployed URL, auth pattern, and response shape.

Fail-open contract (spec §3.1): this module NEVER raises into its caller.
Any failure mode — unset env vars, timeout, connection error, non-200,
malformed JSON — results in ``None``. This is a real, deployed endpoint that
costs money per call; callers must treat a live call as optional/best-effort,
never as something the scan pipeline depends on to complete.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_URL_ENV_VAR = "DRAFTPROOF_MODAL_ENDPOINT_URL"
_TOKEN_ENV_VAR = "DRAFTPROOF_MODAL_ENDPOINT_TOKEN"
_DEFAULT_TIMEOUT_S = 60.0

# FIX 1 — bounded retry on TRANSIENT failures only (request timeout, connection
# error, 5xx). Deterministic failures (4xx/auth, malformed/non-dict JSON, unset
# env, empty chunks) are NEVER retried — a second identical request would fail
# identically and only burn the scan's time budget.
_MAX_ATTEMPTS = 2          # 1 initial attempt + 1 retry
_RETRY_BACKOFF_S = 2.0     # short fixed backoff between attempts
#
# Worst-case client wall-time budget vs. the worker's 330s scan hard limit
# (worker/app/config.py:76). Even the pathological path where the SAME scan
# invokes this client TWICE (builder tier-authority call + a later
# run_v7_breakdown call — see poc/detect_v7/pipeline_bridge.py) stays well under:
#
#   per call  = _MAX_ATTEMPTS * timeout_s + (_MAX_ATTEMPTS - 1) * _RETRY_BACKOFF_S
#             = 2 * 60.0 + 1 * 2.0                                  = 122.0s
#   two calls = 2 * 122.0                                           = 244.0s  < 330s
#
# leaving ~86s of headroom (and FIX 2's attempted-and-failed sentinel eliminates
# that second call in practice, so 122s is the realistic ceiling).

# FIX 3 — long-document timeout guard. Cap the number of windows POSTed per call
# so a pathologically long document cannot blow the per-call latency budget. This
# is a SAFETY cap, NOT a sampler: the default is set far above any window count a
# normal document produces, so typical docs are NEVER capped. Windows are
# 3-sentence, step-1 (poc/detect/deberta_windowing.build_windows), i.e. roughly
# one window per sentence; the upstream document text cap is ~50k chars, which at
# a floor of ~30 chars/sentence yields at most ~1.6k windows. The default 4000
# clears that ceiling comfortably — only truly abnormal input is ever truncated.
_MAX_WINDOWS_ENV_VAR = "DRAFTPROOF_DEEP_SCAN_MAX_WINDOWS"
_DEFAULT_MAX_WINDOWS = 4000


def resolve_max_windows() -> int:
    """Resolve the per-call window cap from ``DRAFTPROOF_DEEP_SCAN_MAX_WINDOWS``.

    Falls back to ``_DEFAULT_MAX_WINDOWS`` when unset, non-integer, or <= 0
    (fail-open: an operator typo must not silently disable deep scan). Note:
    ``pipeline_bridge.get_deep_scan_proportion`` does NOT re-read this cap — it
    aligns its aggregation to the response length (``windows[:len(scores)]``),
    which stays correct under first-N truncation because window j maps to a
    fixed sentence range independent of later windows.
    """
    raw = os.environ.get(_MAX_WINDOWS_ENV_VAR)
    if raw is None:
        return _DEFAULT_MAX_WINDOWS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "detect_v7.modal_client: %s=%r is not an integer; using default %s.",
            _MAX_WINDOWS_ENV_VAR, raw, _DEFAULT_MAX_WINDOWS,
        )
        return _DEFAULT_MAX_WINDOWS
    if value <= 0:
        logger.warning(
            "detect_v7.modal_client: %s=%s is not positive; using default %s.",
            _MAX_WINDOWS_ENV_VAR, value, _DEFAULT_MAX_WINDOWS,
        )
        return _DEFAULT_MAX_WINDOWS
    return value


def _should_retry(attempt: int, reason: str) -> bool:
    """Decide whether a transient failure on ``attempt`` should be retried.

    Returns True (and sleeps ``_RETRY_BACKOFF_S``) when attempts remain, False
    when exhausted. Logs the attempt number either way. ``reason`` is a short
    non-sensitive descriptor — never the URL or bearer token.
    """
    if attempt < _MAX_ATTEMPTS:
        logger.warning(
            "detect_v7.modal_client: transient failure (%s) on attempt %s/%s; "
            "retrying after %ss backoff.",
            reason, attempt, _MAX_ATTEMPTS, _RETRY_BACKOFF_S,
        )
        time.sleep(_RETRY_BACKOFF_S)
        return True
    logger.warning(
        "detect_v7.modal_client: transient failure (%s) on final attempt %s/%s; giving up.",
        reason, attempt, _MAX_ATTEMPTS,
    )
    return False


def call_deep_scan(chunks: list[str], timeout_s: float = _DEFAULT_TIMEOUT_S) -> Optional[dict[str, Any]]:
    """POST ``chunks`` to the live Modal deep-scan endpoint.

    Returns the parsed JSON response dict (expected shape: ``{"available":
    bool, "calibrated": bool, "chunk_scores": [...], "document_score":
    float, ...}``) on success, or ``None`` on ANY failure — including the
    endpoint simply not being configured in this environment (unset env
    vars), which is treated as "Modal isn't available here", not an error.

    Transient failures (timeout, connection error, 5xx) are retried once
    (``_MAX_ATTEMPTS`` total) with a short backoff; deterministic failures are
    not (see the module-level constants for the budget arithmetic). When the
    number of ``chunks`` exceeds ``resolve_max_windows()`` they are truncated to
    the first N (deterministic) and the returned payload carries ``"capped":
    True`` so the caller never silently computes a proportion over a subset.

    Never raises. Never logs the bearer token.
    """
    url = os.environ.get(_URL_ENV_VAR)
    token = os.environ.get(_TOKEN_ENV_VAR)
    if not url or not token:
        logger.info(
            "detect_v7.modal_client: %s/%s not set; Modal deep scan unavailable.",
            _URL_ENV_VAR,
            _TOKEN_ENV_VAR,
        )
        return None

    if not chunks:
        logger.info("detect_v7.modal_client: no chunks provided; skipping Modal call.")
        return None

    # FIX 3: deterministic first-N truncation (long-doc timeout guard). No
    # sampling/shuffling — the caller's window<->score alignment relies on the
    # kept windows being a prefix.
    max_windows = resolve_max_windows()
    capped = False
    if len(chunks) > max_windows:
        logger.warning(
            "detect_v7.modal_client: window count %s exceeds cap %s (%s); "
            "truncating to the first %s windows — proportion is over a SUBSET "
            "(capped=True surfaced in provenance).",
            len(chunks), max_windows, _MAX_WINDOWS_ENV_VAR, max_windows,
        )
        chunks = chunks[:max_windows]
        capped = True

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = requests.post(
                url,
                json={"chunks": chunks},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=timeout_s,
            )
        except requests.exceptions.Timeout:
            if _should_retry(attempt, f"timeout after {timeout_s}s"):
                continue
            return None
        except requests.exceptions.ConnectionError as exc:
            if _should_retry(attempt, f"connection error ({type(exc).__name__})"):
                continue
            return None
        except requests.exceptions.RequestException as exc:
            # Non-transient request error (bad URL/schema, too many redirects,
            # etc.) — deterministic, do NOT retry.
            logger.warning(
                "detect_v7.modal_client: non-retryable request error calling Modal endpoint: %s",
                type(exc).__name__,
            )
            return None
        except Exception:
            logger.exception("detect_v7.modal_client: unexpected error calling Modal endpoint.")
            return None

        if 500 <= response.status_code < 600:
            # Server-side transient (5xx) — retry.
            if _should_retry(attempt, f"HTTP {response.status_code}"):
                continue
            return None
        if response.status_code != 200:
            # 4xx/auth or any other non-200 — deterministic, do NOT retry.
            logger.warning(
                "detect_v7.modal_client: Modal endpoint returned non-200 status %s (not retried).",
                response.status_code,
            )
            return None

        try:
            payload = response.json()
        except ValueError:
            logger.warning("detect_v7.modal_client: Modal endpoint response was not valid JSON.")
            return None

        if not isinstance(payload, dict):
            logger.warning("detect_v7.modal_client: Modal endpoint response was not a JSON object.")
            return None

        if capped:
            # Surface the truncation so the proportion is never silently computed
            # over a subset. Only set on the capped path -> the key is ABSENT for
            # normal (uncapped) responses, keeping their shape byte-identical.
            payload["capped"] = True
        return payload

    # Loop exits only via the returns above; this is defensive (retries exhausted).
    return None
