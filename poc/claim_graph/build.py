"""Claim-graph orchestrator + report-attach entry point.

``build_claim_graph`` runs the deterministic validators over a proposal bundle
and returns the ``cg-1`` container (M1: empty; M2: LLM proposals). M2 adds
``build_claim_graph_extracted``: paragraph-batched LLM extraction
(``extract.py``) → cross-batch reconciliation (``reconcile.py``) → the SAME
validators, so the LLM never owns graph truth.

Import-light on purpose: NO ``rewrite_v6`` imports (circular-import lesson), no
ML stack at module load — the extraction deps (gateway) are lazy-imported inside
``extract.py``/``reconcile.py`` functions, so ``build_claim_graph('t', [])``
stays pure (M1 fresh-interpreter purity test).
"""
from __future__ import annotations

from typing import Any, Optional

from . import claim_graph_enabled
from . import schema
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


def build_claim_graph_extracted(
    text: str,
    segments: list[dict[str, Any]],
    gateway: Any,
) -> dict[str, Any]:
    """M2 orchestrator: extract → reconcile → validate → ``cg-1`` container.

    Two-phase validation so the reconciler can reference the deterministic
    ``c_NNN`` ids: (1) validate the per-batch proposals to obtain accepted claims
    with stable ids; (2) run the reconciler over those accepted claims to
    discover cross-batch edges; (3) re-validate the union — claim ids are
    deterministic, so the reconciler's id references still resolve.

    Fail-open: any extraction/reconcile error yields an empty-but-valid container
    with a lifecycle note (never a broken report). The validators are pure, so a
    non-empty ``proposals`` always produces a well-formed graph.
    """
    from . import extract, reconcile  # lazy — keeps build.py import-light

    container = schema.empty_graph()
    try:
        proposals, stats = extract.run_extraction(text, segments, gateway)
    except Exception as exc:  # fail-open (annotate, don't suppress)
        container["lifecycle"] = {"status": "extraction_failed", "error": str(exc)[:200]}
        return container

    # Phase 1: validate per-batch proposals to get accepted claims + stable ids.
    first_pass = validators.validate_graph(proposals, segments)
    accepted_claims = first_pass.get("claims") or []

    # Phase 2: reconciler owns cross-batch edge discovery.
    try:
        cross_edges, rec_stats = reconcile.reconcile(accepted_claims, gateway)
    except Exception:
        cross_edges, rec_stats = [], {"reconcile_calls": 0, "cross_batch_edges_proposed": 0}

    # Phase 3: re-validate the union (intra-batch quote edges + cross-batch id edges).
    merged = {
        "claims": proposals.get("claims") or [],
        "edges": list(proposals.get("edges") or []) + list(cross_edges),
    }
    # M3: signals computed on the FULL validated graph, before eviction.
    container = validators.validate_graph(merged, segments, compute_signals=True, text=text)

    extraction_stats = dict(stats)
    # M3 QUESTION nodes (carry ``references``) are system-generated, not extracted
    # — exclude them from the acceptance count.
    extraction_stats["accepted"] = len(
        [c for c in (container.get("claims") or []) if not c.get("references")]
    )
    extraction_stats["rejected"] = int(stats.get("proposed", 0)) - extraction_stats["accepted"]
    extraction_stats.update(rec_stats)
    container["extraction_stats"] = extraction_stats
    container["lifecycle"] = {"status": "extracted"}
    return container


def maybe_build_claim_graph(
    text: str,
    segments: list[dict[str, Any]],
    proposals: Optional[dict[str, Any]] = None,
    gateway: Any = None,
) -> dict[str, Any]:
    """Kill-switch-aware wrapper for the report seam.

    Returns ``{}`` (attach-omitting) when ``DRAFTPROOF_CLAIM_GRAPH`` is OFF.
    When ON:
      - explicit ``proposals`` (test/eval) → validate them directly (M1 path);
      - else attempt LLM extraction when a gateway is configured (resolved from
        env when not injected); when no gateway/key is available, fall back to
        the empty-but-valid container (plumbing) so nothing breaks.

    Fail-open by contract: the caller wraps this in try/except AND
    ``build_claim_graph_extracted`` itself never raises on extraction failure —
    it returns an empty container with a lifecycle note (never fails a scan)."""
    if not claim_graph_enabled():
        return {}
    if proposals is not None:
        return build_claim_graph(text, segments, proposals)

    gw = gateway
    if gw is None:
        from . import extract  # lazy
        gw = extract.resolve_gateway()
    if gw is None:
        # No LLM configured — attach the empty plumbing container (M1 behaviour).
        return build_claim_graph(text, segments, None)
    return build_claim_graph_extracted(text, segments, gw)


__all__ = ["build_claim_graph", "build_claim_graph_extracted", "maybe_build_claim_graph"]
