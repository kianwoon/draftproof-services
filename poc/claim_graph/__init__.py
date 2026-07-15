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

_FALSEY = {"0", "false", "no", "off", ""}


def claim_graph_enabled() -> bool:
    """Return whether the Phase-1 claim-graph is enabled.

    Default OFF for Phase 1 (opt-in). Mirrors the ``authorship_evidence.py`` env
    predicate style but inverts the default (that composer defaults ON; this one
    defaults OFF because it is EXPERIMENTAL and — from M2 — paid).
    """
    return os.environ.get(KILL_SWITCH_ENV, "0").strip().lower() not in _FALSEY


__all__ = ["KILL_SWITCH_ENV", "claim_graph_enabled"]
