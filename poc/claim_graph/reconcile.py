"""Stage-2 reconciliation — OWNS cross-batch edge discovery (M2, EXPERIMENTAL).

Paragraph-batched extraction (``extract.py``) can only see edges WITHIN a batch;
a ``revises`` between paragraph 2 and paragraph 7 is invisible to any single
batch (plan §D/§3/risk R4). This stage is given the ACCEPTED claims from ALL
batches (post-validation, with their assigned ``c_NNN`` ids) and proposes edges
BETWEEN claims that different batches produced. It makes 1-3 calls max.

The proposed edges are NOT trusted: they flow back through the same
``validators.py`` rules 4-5 (edge legality, suppression, cycle-break) in the
final validation pass — the reconciler proposes, the application decides.

Gateway import is lazy / caller-injected; no ``rewrite_v6`` import.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from . import schema

_DEFAULT_MAX_CALLS = 1  # within the plan's 1-3 reconciliation budget.

_SYSTEM_PROMPT = (
    "You are the RECONCILER for a claim-graph audit. You are given a list of "
    "already-extracted claims, each with an id (c_NNN), its paragraph, and its "
    "text. Propose typed edges BETWEEN claims that come from DIFFERENT paragraphs "
    "(cross-paragraph relationships a per-paragraph pass could not see). Return "
    "STRICT JSON only:\n"
    '{"edges": [{"type": "supports|derived_from|corroborated_by|qualified_by|'
    'revises|contradicts|inconsistent_with|depends_on|explains|causes|observed_in|'
    'supported_by", "src": "c_NNN", "dst": "c_NNN"}]}\n'
    "RULES: (1) Reference claims ONLY by their given id. (2) Only add an edge when "
    "the relationship is clearly supported by the two claims' text. (3) Prefer "
    "'revises'/'qualified_by' when a later claim scopes or updates an earlier one. "
    "(4) Do not invent claims. Output JSON only."
)


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _parse_edges(content: str) -> Optional[list[dict[str, Any]]]:
    if not content or not content.strip():
        return None
    data: Any = None
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                return None
    if not isinstance(data, dict):
        return None
    edges = data.get("edges")
    return [e for e in edges if isinstance(e, dict)] if isinstance(edges, list) else []


def _build_claims_table(accepted_claims: list[dict[str, Any]]) -> str:
    lines = ["Claims:"]
    for claim in accepted_claims:
        src = claim.get("source") or {}
        pid = src.get("paragraph_id") or "-"
        text = str(claim.get("text") or "").replace("\n", " ")
        lines.append(f'{claim.get("id")} [{pid}] {claim.get("node_type")}: {text}')
    lines.append("\nReturn cross-paragraph edges as JSON.")
    return "\n".join(lines)


def reconcile(
    accepted_claims: list[dict[str, Any]],
    gateway: Any,
    *,
    max_calls: Optional[int] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover cross-batch edges. Returns (edge_proposals, stats).

    Edge proposals are id-referenced ``{type, src, dst}`` dicts fed to the final
    validator pass. No call is made when there are fewer than two claims (nothing
    to link) — keeps cost at zero on trivial docs.
    """
    cap = max_calls if max_calls is not None else _int_env(
        "DRAFTPROOF_CLAIM_GRAPH_RECONCILE_CALLS", _DEFAULT_MAX_CALLS
    )
    stats = {"reconcile_calls": 0, "cross_batch_edges_proposed": 0}
    valid_ids = {c.get("id") for c in accepted_claims if c.get("id")}
    if len(accepted_claims) < 2 or cap <= 0:
        return [], stats

    prompt = _build_claims_table(accepted_claims)
    collected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for _ in range(cap):
        stats["reconcile_calls"] += 1
        try:
            resp = gateway.chat(
                prompt,
                system=_SYSTEM_PROMPT,
                response_format={"type": "json_object"},
                max_tokens=_int_env("DRAFTPROOF_CLAIM_GRAPH_MAX_TOKENS", 8000),
            )
            edges = _parse_edges(str(getattr(resp, "content", "") or ""))
        except Exception:
            edges = None
        if not edges:
            continue
        for edge in edges:
            etype = edge.get("type")
            src = edge.get("src")
            dst = edge.get("dst")
            if etype not in schema.EDGE_TYPES:
                continue
            if src not in valid_ids or dst not in valid_ids or src == dst:
                continue
            dedup_key = (etype, src, dst)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            collected.append({"type": etype, "src": src, "dst": dst})

    stats["cross_batch_edges_proposed"] = len(collected)
    return collected, stats


__all__ = ["reconcile"]
