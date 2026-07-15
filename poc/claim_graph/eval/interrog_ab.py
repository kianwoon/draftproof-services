"""Offline A/B recompute of interrogatability from a SERIALIZED graph.

The M4 headline question (c): the current ``specificity_presence`` component
(signals.py) counts specifics on EVERY claim regardless of ``verification_status``
— so a doc full of FABRICATED specifics (group G) can score HIGH interrogatability,
which the headline invariant (§A) forbids (an unverified specific is an audit
QUESTION, never a credit). This module recomputes the composite two ways on the
SAME validated graph (no extra LLM calls), so the eval can measure whether a
verification-weighted variant fixes G without tanking H:

  * ``variant="current"``  — specificity over all CLAIMs (mirrors signals.py).
  * ``variant="verified"`` — specificity over CLAIMs whose verification_status is
    verified/corroborated/internally_supported only (the §A-aligned restriction).

Unit contract (matches signals.compute_interrogatability): composite is a 0-1
score, HIGHER = MORE interrogatable (richer traceable grounding). Bands use the
same provisional cuts as signals.py so the A/B is apples-to-apples.
"""
from __future__ import annotations

from typing import Any

from ..signals import (_INTERROG_HIGH, _INTERROG_LOW, _CAUSAL_EDGE_TYPES,
                       _band, extract_specifics)

# Claims whose specifics may count as *grounding* under the verified variant.
_CREDITABLE_STATES = ("verified", "corroborated", "internally_supported")


def _claim_dicts(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Real CLAIM nodes only: node_type CLAIM and NOT a system QUESTION (references).
    return [c for c in (claims or [])
            if c.get("node_type") == "CLAIM" and not c.get("references")]


def recompute(claims: list[dict[str, Any]], edges: list[dict[str, Any]],
              variant: str = "current") -> dict[str, Any]:
    """Return ``{composite, band, components}`` for the chosen variant.

    Deterministic; reuses ``signals.extract_specifics`` so the specific-detector
    is identical to production. Only ``specificity_presence`` differs by variant.
    """
    cnodes = _claim_dicts(claims)
    n = len(cnodes)
    if n == 0:
        comps = {k: 0.0 for k in (
            "specificity_presence", "causal_depth",
            "contextual_anchoring", "evidence_traceability")}
        return {"composite": 0.0, "band": _band(0.0, _INTERROG_LOW, _INTERROG_HIGH),
                "components": comps}

    counted_specifics = 0
    for c in cnodes:
        if variant == "verified" and c.get("verification_status") not in _CREDITABLE_STATES:
            continue  # fabricated/unverified specifics do not count as grounding
        counted_specifics += len(extract_specifics(str(c.get("text") or "")))

    causal_edges = sum(1 for e in (edges or []) if e.get("type") in _CAUSAL_EDGE_TYPES)
    anchored = sum(1 for c in cnodes if c.get("origins"))
    traceable = sum(1 for c in cnodes if c.get("verification_status") == "internally_supported")

    comps = {
        "specificity_presence": round(min(1.0, counted_specifics / (n * 2.0)), 4),
        "causal_depth": round(min(1.0, causal_edges / float(n)), 4),
        "contextual_anchoring": round(anchored / float(n), 4),
        "evidence_traceability": round(traceable / float(n), 4),
    }
    composite = round(sum(comps.values()) / 4.0, 4)
    return {"composite": composite,
            "band": _band(composite, _INTERROG_LOW, _INTERROG_HIGH),
            "components": comps}


__all__ = ["recompute"]
