"""Deterministic claim-graph validators — the application OWNS graph truth (§D).

The LLM (M2) *proposes* nodes/edges by quoting spans and naming edge types; this
module is the pure, fully-deterministic gate that decides what enters the graph.
No LLM, no I/O, no randomness — identical inputs yield byte-identical output
(required for the §7 determinism tests and the M4 eval harness).

Rules implemented (plan §3):
  1. Span-match enforcement (headline-invariant guard, risk R1) — a CLAIM whose
     quote is not found at its declared sentence span is REJECTED.
  2. Id assignment + dedup (>90% span overlap collapses).
  3. Verification-status clamp — ``verified``/``corroborated`` are unreachable in
     Phase 1; nodes proposed with them are REJECTED.
  4. Edge legality matrix — illegal type or dangling endpoint rejected;
     ``revises``/``qualified_by`` SUPPRESS a would-be ``inconsistent_with``/
     ``contradicted`` on the same pair (§A↔§9 epistemic-development protection).
  5. Cycle detection — ``depends_on`` must be a DAG; the edge that closes a cycle
     (latest in document order) is dropped.
  6. Node-type purity — exactly one of CLAIM/INFERENCE/QUESTION.

Caps + eviction (plan §1): bounded by ``MAX_CLAIMS``/``MAX_EDGES``
(env-tunable), priority eviction, ``truncated`` flag + ``truncated_dropped_ids``.
Signals are computed on the FULL graph *before* eviction (M3) — eviction runs
last here so it never changes a signal value.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from . import schema
from .schema import ClaimNode, Edge

# Env-tunable caps (plan §1).
_DEFAULT_MAX_CLAIMS = 120
_DEFAULT_MAX_EDGES = 300

_DEDUP_OVERLAP_RATIO = 0.90

# Edge types that must NOT coexist with a revision/scoping edge on the same pair.
_SUPPRESSIBLE_BY_REVISION = ("inconsistent_with", "contradicted")
_REVISION_EDGES = ("revises", "qualified_by")


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def max_claims() -> int:
    return _int_env("DRAFTPROOF_CLAIM_GRAPH_MAX_CLAIMS", _DEFAULT_MAX_CLAIMS)


def max_edges() -> int:
    return _int_env("DRAFTPROOF_CLAIM_GRAPH_MAX_EDGES", _DEFAULT_MAX_EDGES)


def _phase_state_policy() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(forbidden_states, reachable_states)`` for THIS validation run.

    Phase-conditional (plan §5 N0), resolved at validation time so a mid-process
    flag flip is honoured deterministically:
      - entailment OFF → Phase-1 invariant, byte-identical: ``verified`` and
        ``corroborated`` are both forbidden; reachable = PHASE1_REACHABLE_STATES.
      - entailment ON  → ``verified`` becomes legal (permitted, not produced —
        N1 never mints it; that is N4); ``corroborated`` stays forbidden
        (kickoff decision 3).
    """
    from . import entailment_enabled  # lazy — keeps validators import-light

    if entailment_enabled():
        return schema.PHASE2_FORBIDDEN_STATES, schema.PHASE1_REACHABLE_STATES + ("verified",)
    return schema.PHASE1_FORBIDDEN_STATES, schema.PHASE1_REACHABLE_STATES


# ── Sentence index ──────────────────────────────────────────────────────────
def build_sentence_index(segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map ``sentence_id`` → span/text, mapping the codebase's ``start_char``/
    ``end_char`` field names to the spec's ``char_start``/``char_end`` (R7)."""
    index: dict[str, dict[str, Any]] = {}
    for seg in segments or []:
        sid = seg.get("sentence_id")
        if not sid:
            continue
        index[sid] = {
            "paragraph_id": seg.get("paragraph_id") or "",
            "char_start": int(seg.get("start_char") or 0),
            "char_end": int(seg.get("end_char") or 0),
            "text": str(seg.get("sentence") or ""),
        }
    return index


# ── Rule 1 + 6: claim span-match + node-type purity ─────────────────────────
def _valid_node_type(node_type: Any) -> Optional[str]:
    return node_type if node_type in schema.NODE_TYPES else None


def _match_span(quote: str, sentence_text: str) -> Optional[tuple[int, int]]:
    """Locate ``quote`` inside ``sentence_text``; return (local_start, local_end).

    Exact substring first (deterministic); whitespace-normalised fallback maps
    back to the original offsets. Returns None when the quote is not present."""
    if not quote:
        return None
    idx = sentence_text.find(quote)
    if idx != -1:
        return idx, idx + len(quote)
    # Whitespace-normalised fallback: collapse runs of whitespace in both and
    # re-locate, then walk back to the original character offset.
    import re

    norm_quote = re.sub(r"\s+", " ", quote).strip()
    if not norm_quote:
        return None
    # Build a regex that tolerates variable whitespace between the quote tokens.
    pattern = re.compile(r"\s+".join(re.escape(tok) for tok in norm_quote.split(" ")))
    m = pattern.search(sentence_text)
    if m:
        return m.start(), m.end()
    return None


def validate_claims(
    proposed_claims: list[dict[str, Any]],
    sentence_index: dict[str, dict[str, Any]],
) -> list[ClaimNode]:
    """Rules 1/3/6 + id assignment + dedup. Returns accepted ClaimNodes.

    A proposed claim dict shape (LLM contract, plan §3):
      {sentence_id, quote, node_type, claim_type, origins, primary_origin,
       verification_status?}
    """
    forbidden_states, reachable_states = _phase_state_policy()
    accepted: list[ClaimNode] = []
    for proposal in proposed_claims or []:
        node_type = _valid_node_type(proposal.get("node_type"))
        if node_type is None:  # rule 6: purity — unknown type rejected
            continue

        # rule 3: forbidden verification states rejected outright (phase-conditional).
        status = proposal.get("verification_status")
        if status in forbidden_states:
            continue
        if status not in reachable_states:
            status = "unverified"

        source: Optional[dict[str, Any]] = None
        if node_type == "CLAIM":
            # rule 1: span-match enforcement (headline-invariant guard).
            sid = proposal.get("sentence_id")
            seg = sentence_index.get(sid)
            if seg is None:
                continue
            span = _match_span(str(proposal.get("quote") or ""), seg["text"])
            if span is None:
                continue  # hallucinated / not-in-source → REJECT
            local_start, local_end = span
            source = {
                "paragraph_id": seg["paragraph_id"],
                "sentence_id": sid,
                "char_start": seg["char_start"] + local_start,
                "char_end": seg["char_start"] + local_end,
            }
        # INFERENCE / QUESTION carry source=None and skip the span check.

        origins = [o for o in (proposal.get("origins") or []) if o in schema.ORIGIN_TAXONOMY]
        primary = proposal.get("primary_origin")
        if primary not in origins:
            primary = origins[0] if origins else None
        claim_type = proposal.get("claim_type")
        if claim_type not in schema.CLAIM_TYPES:
            claim_type = None

        accepted.append(
            ClaimNode(
                id="",  # assigned after dedup/order
                node_type=node_type,
                text=str(proposal.get("quote") or proposal.get("text") or ""),
                claim_type=claim_type,
                source=source,
                verification_status=status,
                origins=origins,
                primary_origin=primary,
            )
        )

    accepted = _dedup_by_span(accepted)
    _assign_claim_ids(accepted)
    return accepted


def _dedup_by_span(claims: list[ClaimNode]) -> list[ClaimNode]:
    """Rule 2: collapse CLAIMs whose spans overlap >90% (keep the first)."""
    kept: list[ClaimNode] = []
    for claim in claims:
        if claim.source is None:
            kept.append(claim)
            continue
        a0, a1 = claim.source["char_start"], claim.source["char_end"]
        dupe = False
        for other in kept:
            if other.source is None:
                continue
            b0, b1 = other.source["char_start"], other.source["char_end"]
            overlap = max(0, min(a1, b1) - max(a0, b0))
            span_min = max(1, min(a1 - a0, b1 - b0))
            if overlap / span_min > _DEDUP_OVERLAP_RATIO:
                dupe = True
                break
        if not dupe:
            kept.append(claim)
    return kept


def _doc_order_key(claim: ClaimNode) -> tuple[int, int]:
    if claim.source is None:
        return (1, 0)  # system-generated nodes sort after source-anchored ones
    return (0, claim.source["char_start"])


def _assign_claim_ids(claims: list[ClaimNode]) -> None:
    claims.sort(key=_doc_order_key)
    for i, claim in enumerate(claims, start=1):
        claim.id = f"c_{i:03d}"


# ── Rules 4 + 5: edge legality, suppression, cycle detection ────────────────
def _resolve_endpoint(quote: Any, claims_by_text: dict[str, str]) -> Optional[str]:
    if not quote:
        return None
    return claims_by_text.get(str(quote))


def validate_edges(
    proposed_edges: list[dict[str, Any]],
    claims: list[ClaimNode],
) -> list[Edge]:
    """Rules 4/5. Resolve quote endpoints to node ids, reject illegal/dangling
    edges, suppress fabrication edges under revision, break ``depends_on`` cycles.
    """
    # Prefer an exact-text lookup; last-writer-wins is deterministic given order.
    claims_by_text = {c.text: c.id for c in claims}
    valid_ids = {c.id for c in claims}

    resolved: list[tuple[str, str, str]] = []  # (type, src_id, dst_id)
    for proposal in proposed_edges or []:
        etype = proposal.get("type")
        if etype not in schema.EDGE_TYPES:  # rule 4: legality matrix
            continue
        src = proposal.get("src") or _resolve_endpoint(proposal.get("src_quote"), claims_by_text)
        dst = proposal.get("dst") or _resolve_endpoint(proposal.get("dst_quote"), claims_by_text)
        if src not in valid_ids or dst not in valid_ids or src == dst:
            continue  # dangling / self-loop → reject
        resolved.append((etype, src, dst))

    resolved = _suppress_under_revision(resolved)
    resolved = _break_depends_on_cycles(resolved)

    edges: list[Edge] = []
    for i, (etype, src, dst) in enumerate(resolved, start=1):
        edges.append(Edge(id=f"x_{i:03d}", type=etype, src=src, dst=dst))
    return edges


def _unordered_pair(a: str, b: str) -> frozenset:
    return frozenset((a, b))


def _suppress_under_revision(resolved: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Rule 4 (§A↔§9): a ``revises``/``qualified_by`` edge on an unordered pair
    suppresses any ``inconsistent_with``/``contradicted`` edge on the same pair —
    signposted revision/scoping is epistemic development, not fabrication."""
    revision_pairs = {
        _unordered_pair(src, dst)
        for (etype, src, dst) in resolved
        if etype in _REVISION_EDGES
    }
    if not revision_pairs:
        return resolved
    out: list[tuple[str, str, str]] = []
    for etype, src, dst in resolved:
        if etype in _SUPPRESSIBLE_BY_REVISION and _unordered_pair(src, dst) in revision_pairs:
            continue
        out.append((etype, src, dst))
    return out


def _break_depends_on_cycles(resolved: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Rule 5: keep ``depends_on`` a DAG. Add edges in document order; drop any
    edge that would close a cycle (deterministic — the closing edge is dropped)."""
    adjacency: dict[str, set] = {}
    kept: list[tuple[str, str, str]] = []

    def _reachable(start: str, target: str) -> bool:
        stack = [start]
        seen = set()
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency.get(node, ()))
        return False

    for etype, src, dst in resolved:
        if etype != "depends_on":
            kept.append((etype, src, dst))
            continue
        # Adding src->dst closes a cycle iff dst can already reach src.
        if _reachable(dst, src):
            continue  # drop the edge that closes the cycle
        adjacency.setdefault(src, set()).add(dst)
        kept.append((etype, src, dst))
    return kept


# ── Rule 3 (part 2): verification tagging from surviving edges ──────────────
def apply_verification_states(claims: list[ClaimNode], edges: list[Edge]) -> None:
    """Set ``contradicted`` on both endpoints of a surviving ``inconsistent_with``
    edge; promote otherwise-uncontradicted claims that participate in a
    supporting sub-graph to ``internally_supported`` (§1 — the only positive
    state reachable in Phase 1)."""
    by_id = {c.id: c for c in claims}
    contradicted: set = set()
    supported: set = set()
    for edge in edges:
        if edge.type in ("inconsistent_with", "contradicts"):
            contradicted.add(edge.src)
            contradicted.add(edge.dst)
        elif edge.type in ("supports", "corroborated_by", "supported_by", "depends_on"):
            supported.add(edge.src)
            supported.add(edge.dst)
    for cid, claim in by_id.items():
        if cid in contradicted:
            claim.verification_status = "contradicted"
        elif cid in supported and claim.verification_status == "unverified":
            claim.verification_status = "internally_supported"


# ── Caps + eviction (plan §1) ───────────────────────────────────────────────
def _load_bearing(claim_id: str, edges: list[Edge]) -> bool:
    for edge in edges:
        if edge.type in ("contradicted", "contradicts", "inconsistent_with", "revises"):
            if claim_id in (edge.src, edge.dst):
                return True
    return False


def enforce_caps(
    claims: list[ClaimNode],
    edges: list[Edge],
    cap_claims: int,
    cap_edges: int,
) -> tuple[list[ClaimNode], list[Edge], list[str]]:
    """Priority eviction. Returns (kept_claims, kept_edges, dropped_ids).

    Node priority (keep-first): (1) load-bearing nodes (contradicted/
    inconsistent_with/revises edge), (2) CLAIM nodes with the most edges,
    (3) document order. Deterministic."""
    dropped_ids: list[str] = []
    if len(claims) > cap_claims:
        edge_degree: dict[str, int] = {}
        for edge in edges:
            edge_degree[edge.src] = edge_degree.get(edge.src, 0) + 1
            edge_degree[edge.dst] = edge_degree.get(edge.dst, 0) + 1
        doc_index = {c.id: i for i, c in enumerate(claims)}

        def _priority(claim: ClaimNode) -> tuple:
            # Lower tuple sorts FIRST = higher keep-priority.
            return (
                0 if _load_bearing(claim.id, edges) else 1,
                -edge_degree.get(claim.id, 0),
                doc_index[claim.id],
            )

        ranked = sorted(claims, key=_priority)
        kept = ranked[:cap_claims]
        kept_ids = {c.id: True for c in kept}
        dropped_ids = [c.id for c in claims if c.id not in kept_ids]
        # Preserve document order for the persisted list.
        claims = [c for c in claims if c.id in kept_ids]
        edges = [e for e in edges if e.src in kept_ids and e.dst in kept_ids]

    if len(edges) > cap_edges:
        def _edge_priority(edge: Edge) -> tuple:
            load = 0 if edge.type in ("contradicts", "inconsistent_with", "revises") else 1
            return (load,)

        # Stable sort keeps document order within a priority band.
        ranked_edges = sorted(edges, key=_edge_priority)
        keep_edges = set(id(e) for e in ranked_edges[:cap_edges])
        edges = [e for e in edges if id(e) in keep_edges]

    return claims, edges, dropped_ids


# ── Top-level orchestration ─────────────────────────────────────────────────
def validate_graph(
    proposals: Optional[dict[str, Any]],
    segments: list[dict[str, Any]],
    cap_claims: Optional[int] = None,
    cap_edges: Optional[int] = None,
    compute_signals: bool = False,
    text: str = "",
    embedder: Any = "__resolve__",
) -> dict[str, Any]:
    """Validate a proposal bundle into the ``cg-1`` container.

    ``proposals`` is ``{"claims": [...], "edges": [...]}`` (LLM output in M2);
    ``None``/empty yields the empty-but-valid container (M1 default).

    ``compute_signals`` (M3): compute the three EXPERIMENTAL signals on the FULL
    validated graph *before* eviction (plan §1 — truncation never changes a signal
    value). Emitted QUESTION nodes join the graph (subject to the same caps).
    Default ``False`` keeps the M1/M2 path pure + byte-identical (signals=[])."""
    proposals = proposals or {}
    cap_claims = cap_claims if cap_claims is not None else max_claims()
    cap_edges = cap_edges if cap_edges is not None else max_edges()

    sentence_index = build_sentence_index(segments)
    claims = validate_claims(proposals.get("claims") or [], sentence_index)
    edges = validate_edges(proposals.get("edges") or [], claims)
    apply_verification_states(claims, edges)

    # Signals (M3) computed HERE, on the full graph, BEFORE eviction.
    signal_objects: list[dict[str, Any]] = []
    if compute_signals:
        from . import signals as _signals  # lazy — keeps validators import-light

        signal_objects, question_nodes = _signals.compute_signals(
            claims, edges, text=text, embedder=embedder
        )
        # Assign deterministic ids to system-generated QUESTION nodes and append
        # them so they persist alongside the claims they interrogate.
        for i, q in enumerate(question_nodes, start=1):
            q.id = f"q_{i:03d}"
        claims = claims + question_nodes

    kept_claims, kept_edges, dropped_ids = enforce_caps(claims, edges, cap_claims, cap_edges)

    return schema.build_container(
        claims=kept_claims,
        edges=kept_edges,
        signals=signal_objects,
        truncated=bool(dropped_ids),
        truncated_dropped_ids=dropped_ids,
    )


__all__ = [
    "max_claims",
    "max_edges",
    "build_sentence_index",
    "validate_claims",
    "validate_edges",
    "apply_verification_states",
    "enforce_caps",
    "validate_graph",
]
