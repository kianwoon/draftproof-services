"""Claim-graph orchestrator + report-attach entry point.

M1 is PLUMBING ONLY: ``build_claim_graph`` runs the deterministic validators
over an (M1: empty) proposal bundle and returns the ``cg-1`` container. NO LLM
extraction here — that is M2 (``extractor.py``/``reconciler.py``), which will
feed real proposals into ``validators.validate_graph`` at the same seam.

Import-light on purpose: NO ``rewrite_v6`` imports (circular-import lesson), no
ML stack — the whole M1 path is pure Python.
"""
from __future__ import annotations

from typing import Any, Optional

from . import claim_graph_enabled
from . import validators


def build_claim_graph(
    text: str,
    segments: list[dict[str, Any]],
    proposals: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the validated ``cg-1`` container.

    Args:
      text: the submitted document text (unused in M1; M2 batches it for the LLM).
      segments: canonical ``structured_sentence_segments`` rows — the span anchor
        source the validators enforce against.
      proposals: M2+ LLM output ``{"claims": [...], "edges": [...]}``. ``None`` in
        M1 → empty-but-valid graph.

    Returns the container dict. The report seam decides whether to attach it
    (kill-switch); this function itself never reads the switch, so it stays a
    pure builder that the M4 eval harness can call directly.
    """
    return validators.validate_graph(proposals, segments)


def maybe_build_claim_graph(
    text: str,
    segments: list[dict[str, Any]],
    proposals: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Kill-switch-aware wrapper for the report seam.

    Returns ``{}`` (attach-omitting) when ``DRAFTPROOF_CLAIM_GRAPH`` is OFF, else
    the empty-but-valid container (M1) so M2 has a socket. Fail-open by contract:
    the caller wraps this in try/except and drops the field on any error
    (annotate-don't-suppress; never fail a scan on the experimental graph)."""
    if not claim_graph_enabled():
        return {}
    return build_claim_graph(text, segments, proposals)


__all__ = ["build_claim_graph", "maybe_build_claim_graph"]
