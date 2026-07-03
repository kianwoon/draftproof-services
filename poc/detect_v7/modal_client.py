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
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_URL_ENV_VAR = "DRAFTPROOF_MODAL_ENDPOINT_URL"
_TOKEN_ENV_VAR = "DRAFTPROOF_MODAL_ENDPOINT_TOKEN"
_DEFAULT_TIMEOUT_S = 60.0


def call_deep_scan(chunks: list[str], timeout_s: float = _DEFAULT_TIMEOUT_S) -> Optional[dict[str, Any]]:
    """POST ``chunks`` to the live Modal deep-scan endpoint.

    Returns the parsed JSON response dict (expected shape: ``{"available":
    bool, "calibrated": bool, "chunk_scores": [...], "document_score":
    float, ...}``) on success, or ``None`` on ANY failure — including the
    endpoint simply not being configured in this environment (unset env
    vars), which is treated as "Modal isn't available here", not an error.

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
        logger.warning("detect_v7.modal_client: request to Modal endpoint timed out after %ss.", timeout_s)
        return None
    except requests.exceptions.RequestException as exc:
        logger.warning("detect_v7.modal_client: connection error calling Modal endpoint: %s", type(exc).__name__)
        return None
    except Exception:
        logger.exception("detect_v7.modal_client: unexpected error calling Modal endpoint.")
        return None

    if response.status_code != 200:
        logger.warning(
            "detect_v7.modal_client: Modal endpoint returned non-200 status %s.",
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

    return payload
