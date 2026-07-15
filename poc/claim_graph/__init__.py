"""Phase-1 Claim-Graph package (EXPERIMENTAL, internal-only).

Governing specs:
  - docs/plans/phase1_claim_graph_execution_plan.md  (M1 scope)
  - docs/plans/credible_authorship_assessment_v2.md  (invariants)

Kill-switch: ``DRAFTPROOF_CLAIM_GRAPH`` — default OFF for Phase 1 (opt-in,
premium/paid-tier). When OFF the report is byte-identical to pre-Phase-1.

M1 ships PLUMBING ONLY: the data model, the deterministic validators, and the
empty-graph-capable report-attach seam. NO LLM extraction (that is M2), NO
signals (M3). Keep this module import-light — NO ``rewrite_v6`` imports at
module load (circular-import lesson, CLAUDE.md); the validators/schema are pure.
"""
from __future__ import annotations

import os

# Name is duplicated as a string literal in the serialized container so a report
# reader can see which switch governs the graph without importing this package.
KILL_SWITCH_ENV = "DRAFTPROOF_CLAIM_GRAPH"

# Phase-2 (Track A) entailment gate — INDEPENDENT of the Phase-1 kill-switch
# above. It only unlocks the ``verified`` verification state in the schema (plan
# §5 N0/N1); the resolver/retrieval/entailment engine arrive in N2–N4. Default
# OFF so the Phase-1 report stays byte-identical until entailment ships.
ENTAILMENT_KILL_SWITCH_ENV = "DRAFTPROOF_ENTAILMENT"

_FALSEY = {"0", "false", "no", "off", ""}


def claim_graph_enabled() -> bool:
    """Return whether the Phase-1 claim-graph is enabled.

    Default OFF for Phase 1 (opt-in). Mirrors the ``authorship_evidence.py`` env
    predicate style but inverts the default (that composer defaults ON; this one
    defaults OFF because it is EXPERIMENTAL and — from M2 — paid).
    """
    return os.environ.get(KILL_SWITCH_ENV, "0").strip().lower() not in _FALSEY


def entailment_enabled() -> bool:
    """Return whether Phase-2 entailment (Track A) is enabled.

    Default OFF. Same falsey-set + style as ``claim_graph_enabled`` — the only
    behavioural difference downstream is that the validator PERMITS a ``verified``
    verification state when this is on (plan §5 N0). ``corroborated`` stays
    forbidden this phase regardless (kickoff decision 3).
    """
    return os.environ.get(ENTAILMENT_KILL_SWITCH_ENV, "0").strip().lower() not in _FALSEY


__all__ = [
    "KILL_SWITCH_ENV",
    "ENTAILMENT_KILL_SWITCH_ENV",
    "claim_graph_enabled",
    "entailment_enabled",
]
