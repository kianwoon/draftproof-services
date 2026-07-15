"""Claim-graph data model + JSON contract (schema ``cg-1``).

Pure module — dataclasses, enums, and (de)serialise helpers only. No I/O, no
LLM, no heavy imports. The JSON contract is the one persisted under
``authorship_evidence.claim_graph`` in the R2 report (plan §1).

Field-name note (plan §0/§7, risk R7): the codebase's canonical sentence spans
use ``start_char``/``end_char`` (``document_structure.StructuredSentence``); the
spec's node contract uses ``char_start``/``char_end``. The mapping happens in the
validator, so *this* module's node contract already speaks the spec dialect
(``char_start``/``char_end``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import KILL_SWITCH_ENV

SCHEMA_VERSION = "cg-1"
STATUS_EXPERIMENTAL = "experimental"

# ── Enumerations (spec §2/§4/§7) ─────────────────────────────────────────────
NODE_TYPES = ("CLAIM", "INFERENCE", "QUESTION")

CLAIM_TYPES = (
    "factual",
    "analytical",
    "personal_observation",
    "interpretation",
    "conclusion",
)

# Full normative edge set (v2 §4). Order is documentary, not semantic.
EDGE_TYPES = (
    "supports",
    "derived_from",
    "corroborated_by",
    "qualified_by",
    "revises",
    "contradicts",
    "inconsistent_with",
    "depends_on",
    "explains",
    "causes",
    "observed_in",
    "supported_by",
)

# Verification states (v2 §2 table). NOTE: ``verified``/``corroborated`` are
# UNREACHABLE in Phase 1 by construction (no entailment/context engine); the
# validator REJECTS any node proposed with them (headline invariant).
VERIFICATION_STATES = (
    "verified",
    "corroborated",
    "internally_supported",
    "unverified",
    "unverifiable",
    "contradicted",
)
# The subset a Phase-1 CLAIM may actually hold after validation.
PHASE1_REACHABLE_STATES = (
    "internally_supported",
    "unverified",
    "unverifiable",
    "contradicted",
)
# Rejected on sight in Phase 1 (§1 verification reality).
PHASE1_FORBIDDEN_STATES = ("verified", "corroborated")

ORIGIN_TAXONOMY = (
    "external_source",
    "personal_observation",
    "original_analysis",
    "common_knowledge",
    "assignment_prompt",
    "interpretation",
)

ASSESSMENT_CONFIDENCE = ("low", "moderate", "high")


@dataclass
class ClaimNode:
    """A CLAIM/INFERENCE/QUESTION node.

    CLAIM nodes MUST carry a deterministic ``source`` span (§2 span invariant);
    INFERENCE/QUESTION are system-generated and carry ``source=None``.
    """

    id: str
    node_type: str
    text: str
    claim_type: Optional[str] = None
    source: Optional[dict[str, Any]] = None  # {paragraph_id, sentence_id, char_start, char_end}
    verification_status: str = "unverified"
    verification_paths: list = field(default_factory=list)
    origins: list = field(default_factory=list)
    primary_origin: Optional[str] = None
    evidence_level: int = 0
    assessment_confidence: str = "low"
    limitations: list = field(default_factory=list)
    # M3: a system-generated QUESTION node references the CLAIM it interrogates
    # (§4a teacher-probe). ``None`` for CLAIM/INFERENCE nodes.
    references: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "text": self.text,
            "claim_type": self.claim_type,
            "source": self.source,
            "verification_status": self.verification_status,
            "verification_paths": list(self.verification_paths),
            "origins": list(self.origins),
            "primary_origin": self.primary_origin,
            "evidence_level": self.evidence_level,
            "assessment_confidence": self.assessment_confidence,
            "limitations": list(self.limitations),
            "references": self.references,
        }


@dataclass
class EvidenceNode:
    id: str
    kind: str
    claim_ids: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "claim_ids": list(self.claim_ids)}


@dataclass
class Edge:
    id: str
    type: str
    src: str
    dst: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "src": self.src, "dst": self.dst}


def empty_graph() -> dict[str, Any]:
    """Return an empty-but-valid ``cg-1`` container with lifecycle metadata.

    This is exactly what M1 attaches when the kill-switch is ON (extraction is
    M2), so M2 has a socket to populate.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "kill_switch": KILL_SWITCH_ENV,
        "status": STATUS_EXPERIMENTAL,
        "truncated": False,
        "truncated_dropped_ids": [],
        "claims": [],
        "evidence": [],
        "edges": [],
        "signals": [],
    }


def build_container(
    claims: list[ClaimNode],
    edges: list[Edge],
    evidence: Optional[list[EvidenceNode]] = None,
    signals: Optional[list[dict[str, Any]]] = None,
    truncated: bool = False,
    truncated_dropped_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Serialise validated nodes/edges into the ``cg-1`` JSON contract."""
    container = empty_graph()
    container["claims"] = [c.to_dict() for c in claims]
    container["edges"] = [e.to_dict() for e in edges]
    container["evidence"] = [n.to_dict() for n in (evidence or [])]
    container["signals"] = list(signals or [])
    container["truncated"] = bool(truncated)
    container["truncated_dropped_ids"] = list(truncated_dropped_ids or [])
    return container


__all__ = [
    "SCHEMA_VERSION",
    "STATUS_EXPERIMENTAL",
    "NODE_TYPES",
    "CLAIM_TYPES",
    "EDGE_TYPES",
    "VERIFICATION_STATES",
    "PHASE1_REACHABLE_STATES",
    "PHASE1_FORBIDDEN_STATES",
    "ORIGIN_TAXONOMY",
    "ASSESSMENT_CONFIDENCE",
    "ClaimNode",
    "EvidenceNode",
    "Edge",
    "empty_graph",
    "build_container",
]
