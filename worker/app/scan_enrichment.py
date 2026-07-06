"""Post-detection LLM enrichment calls for the scan pipeline.

Extracted from ``tasks.py``. Each function here does ONLY the pure LLM + parse
work for one enrichment step — no ``results_json`` mutation, no progress
publishing, no DB/Redis I/O, no file writes. That keeps them safe to run on
worker threads (the LLM gateways use ``threading.local`` sessions and are
thread-safe); the ``scan_document`` MAIN thread collects the returned values and
applies them (mutation + render + upload) in the same order as before.

Each call keeps its OWN fail-open semantics at the call site: a raised
exception here is caught per-future by the caller and logged exactly as the
sequential path did, without affecting the other two steps.
"""
from __future__ import annotations

from typing import Any, Optional

from .config import settings


def run_paragraph_explanations(results_json: dict) -> Optional[Any]:
    """Generate per-paragraph finding explanations (pure LLM call).

    Returns the explanations payload (or ``None`` when the explainer opts out);
    the caller applies it to ``results_json`` and the DraftReport object.
    """
    from poc.report.paragraph_explainer import generate_paragraph_explanations

    return generate_paragraph_explanations(
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


def run_critical_thinking(results_json: dict) -> Optional[Any]:
    """Assess Critical Thinking Control LLM dimensions (pure LLM call).

    Returns ``None`` when the badge lacks a ``critical_thinking_control`` dict
    (nothing to enrich) or the assessor opts out; otherwise the enrichment
    payload for the caller to merge onto the ``ctc`` dict.
    """
    from poc.detect.critical_thinking_llm import assess_critical_thinking

    ctc = (results_json.get("ai_risk_badge") or {}).get("critical_thinking_control")
    if not isinstance(ctc, dict):
        return None
    return assess_critical_thinking(
        results_json,
        api_key=(settings.LLM_API_KEY or settings.OPENROUTER_API_KEY or settings.CEREBRAS_API_KEY or None),
        base_url=(settings.LLM_BASE_URL or None),
    )


def run_reflective_questions(results_json: dict) -> Optional[Any]:
    """Generate anchored reflective questions (pure LLM call).

    Returns ``None`` when the badge lacks a ``critical_thinking_control`` dict
    or the generator opts out; otherwise the questions payload for the caller
    to merge onto the ``ctc`` dict and mirror onto the DraftReport object.
    """
    from poc.detect.critical_thinking_llm import generate_reflective_questions

    ctc = (results_json.get("ai_risk_badge") or {}).get("critical_thinking_control")
    if not isinstance(ctc, dict):
        return None
    return generate_reflective_questions(
        results_json,
        api_key=(settings.LLM_API_KEY or settings.OPENROUTER_API_KEY or settings.CEREBRAS_API_KEY or None),
        base_url=(settings.LLM_BASE_URL or None),
    )
