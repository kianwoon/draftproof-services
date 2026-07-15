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

import os
from typing import Any, Optional

from . import claim_graph_enabled
from . import entailment_enabled
from . import schema
from . import validators

_FALSEY = {"0", "false", "no", "off", ""}


def _source_resolution_enabled() -> bool:
    """N2 source-resolution sub-switch (default ON within entailment).

    Lets an operator run N1 citation-linking WITHOUT the N2 network resolver
    (DRAFTPROOF_CLAIM_GRAPH_SOURCE_RESOLVE=0). Only ever consulted once
    ``entailment_enabled()`` already gated us in."""
    return os.environ.get(
        "DRAFTPROOF_CLAIM_GRAPH_SOURCE_RESOLVE", "1").strip().lower() not in _FALSEY


def _entailment_scoring_enabled() -> bool:
    """N3 entailment sub-switch (default ON within entailment).

    Lets an operator run N1+N2 (link + resolve) WITHOUT the N3 entailment engine
    (DRAFTPROOF_CLAIM_GRAPH_ENTAIL=0). Only consulted once ``entailment_enabled()``
    already gated us in AND source resolution ran."""
    return os.environ.get(
        "DRAFTPROOF_CLAIM_GRAPH_ENTAIL", "1").strip().lower() not in _FALSEY


def _resolver_deps():
    """N2 network seam (patch point for tests). ``None`` → sources uses its own
    ``requests``-backed default."""
    return None


def _resolver_snapshot():
    """N2 frozen-snapshot seam (N6 eval determinism, plan §4/risk R4).

    When ``DRAFTPROOF_CLAIM_GRAPH_SOURCE_SNAPSHOT`` points at a snapshot fixture,
    load it so ``resolve_sources`` replays the frozen Crossref/URL records WITHOUT
    touching the network (``sources.py`` docstring: "the N6 eval harness replays
    WITHOUT touching the network"). Unset (production/default) → ``None`` → live
    resolution. Fail-open: any load error degrades to ``None`` (live)."""
    path = os.environ.get("DRAFTPROOF_CLAIM_GRAPH_SOURCE_SNAPSHOT", "").strip()
    if not path:
        return None
    try:
        from . import sources  # lazy — heavier (requests) import
        return sources.SourceSnapshot(path=path)
    except Exception:
        return None


def _entailment_scorer():
    """N3 scorer seam (patch point for tests). Returns the env-driven Modal
    scorer; fail-opens to ``unverified`` when the endpoint is unconfigured."""
    from . import entailment  # lazy — keeps build.py import-light
    return entailment.default_scorer()


def _run_entailment(evidence: list, claims: list) -> None:
    """N3: for each RESOLVED external-citation evidence node, entail its linked
    claim(s) against the retrieved source and ATTACH the verdict into
    ``detail['resolution']['entailment']`` (keyed by claim id). Fail-open.

    N3 COMPUTES + ATTACHES only — it does NOT set ``verification_status``; that
    wire-back is N4. Scored ONLY on ``resolved`` records (they carry
    ``source_text``); paywalled/unresolved/error are skipped and stay unverified.
    """
    from . import entailment  # lazy — keeps build.py import-light
    from . import sources as _sources

    claim_text_by_id = {
        str(c.get("id")): str(c.get("text") or "")
        for c in (claims or []) if c.get("id")
    }
    thresholds = entailment.load_thresholds()
    scorer = _entailment_scorer()

    for node in evidence or []:
        detail = getattr(node, "detail", None)
        if not isinstance(detail, dict):
            continue
        resolution = detail.get("resolution")
        if not isinstance(resolution, dict):
            continue
        if resolution.get("status") != _sources.STATUS_RESOLVED:
            continue  # only resolved sources carry retrievable source_text
        source_text = resolution.get("source_text")
        if not source_text:
            continue
        claim_ids = list(getattr(node, "claim_ids", None) or [])
        if not claim_ids:
            continue  # no hypothesis to test against
        by_claim: dict[str, Any] = {}
        for cid in claim_ids:
            cid = str(cid)
            claim_text = claim_text_by_id.get(cid)
            if not claim_text:
                continue
            by_claim[cid] = entailment.entail_claim(
                claim_text, source_text, scorer, thresholds=thresholds)
        if by_claim:
            resolution["entailment"] = by_claim


def _attach_entailment_evidence(text: str, claim_dicts: list) -> tuple:
    """N1→N2→N3 attach block: link citations → resolve sources → entail.

    Returns ``(evidence_nodes, coverage, limitations)``. ``evidence_nodes`` are
    the node objects (NOT ``to_dict``'d) so N4 can read each node's
    ``detail['resolution']['entailment']`` verdict map for promotion. Preserves
    every existing fail-open + sub-switch (``_source_resolution_enabled`` /
    ``_entailment_scoring_enabled``) semantic; mints NO ``verification_status``.
    """
    from . import citations  # lazy — keeps build.py import-light

    evidence = citations.link_citations(text, claim_dicts or [])
    coverage = None
    limitations: list = []
    if _source_resolution_enabled():
        try:
            from . import sources  # lazy — heavier (requests) import

            evidence, coverage, limitations = sources.resolve_sources(
                evidence, deps=_resolver_deps(), snapshot=_resolver_snapshot())
            if _entailment_scoring_enabled():
                try:
                    _run_entailment(evidence, claim_dicts or [])
                except Exception:
                    pass  # fail-open: never fail the build on N3
        except Exception:
            coverage, limitations = None, []
    return evidence, coverage, limitations


def _promote_verification_states(claims: list, evidence_nodes: list) -> None:
    """N4 wire-back: promote ``claim.verification_status`` from the N3 verdicts.

    Precedence (plan §3 verdict asymmetry, §I):
      * ``contradicted`` ALWAYS wins (the only negative state — surface the flag).
      * ``verified`` promotes a claim UNLESS it is already ``contradicted`` (an
        internal contradiction from edges, or a contradicting source): never
        downgrade — keep ``contradicted`` and append a limitations note instead of
        silently overwriting.
      * ``unverified`` leaves the status untouched (``unverified ≠ fabricated``).

    Deterministic; mutates ClaimNode objects in place. Only ever reached under
    ``entailment_enabled()`` (where ``verified`` is a legal state, N0)."""
    from . import entailment  # lazy — canonical verdict status strings (no hardcode)

    by_id = {c.id: c for c in claims}
    for node in evidence_nodes or []:
        detail = getattr(node, "detail", None)
        if not isinstance(detail, dict):
            continue
        resolution = detail.get("resolution")
        if not isinstance(resolution, dict):
            continue
        verdicts = resolution.get("entailment")
        if not isinstance(verdicts, dict):
            continue
        for cid, verdict in verdicts.items():
            claim = by_id.get(str(cid))
            if claim is None or not isinstance(verdict, dict):
                continue
            status = verdict.get("status")
            if status == entailment.STATUS_CONTRADICTED:
                claim.verification_status = entailment.STATUS_CONTRADICTED
            elif status == entailment.STATUS_VERIFIED:
                if claim.verification_status == entailment.STATUS_CONTRADICTED:
                    note = ("entailment verified vs internal contradiction: kept "
                            "contradicted (N4 — contradicted always wins)")
                    if note not in claim.limitations:
                        claim.limitations.append(note)
                else:
                    claim.verification_status = entailment.STATUS_VERIFIED
            # entailment.STATUS_UNVERIFIED / anything else → leave status as-is.


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
    if not entailment_enabled():
        # Flag OFF: VERBATIM pre-N4 single-call path. Signals see the Phase-1
        # statuses; specificity credits ALL specifics — byte-identical to Phase-1.
        # M3: signals computed on the FULL validated graph, before eviction.
        container = validators.validate_graph(
            merged, segments, compute_signals=True, text=text)
    else:
        # Flag ON: split so N4 can promote verification_status from entailment
        # verdicts BEFORE signals run — a verified specific is credited and never
        # emitted as a QUESTION (no q_NNN churn). Everything fail-open: any error
        # falls back to the pre-N4 single-call container.
        try:
            claims, edges = validators._validate_core(merged, segments)
            claim_dicts = [c.to_dict() for c in claims]
            # N1→N2→N3: link citations, resolve sources, entail (attach verdicts).
            evidence_nodes, coverage, limitations = _attach_entailment_evidence(
                text, claim_dicts)
            # N4: promote verification_status from the attached verdicts, THEN run
            # signals verified-only (the M4 rec (f)1 gaming fix, plan [G3]).
            _promote_verification_states(claims, evidence_nodes)
            container = validators.finalize_container(
                claims, edges, compute_signals=True, text=text,
                specificity_verified_only=True)
            container["evidence"] = [n.to_dict() for n in evidence_nodes]
            if coverage is not None:
                container["source_coverage"] = coverage
                container["source_limitations"] = list(limitations)
        except Exception:
            # Fail-open: never fail the build — fall back to a valid container.
            container = validators.validate_graph(
                merged, segments, compute_signals=True, text=text)
            container.setdefault("evidence", [])

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
