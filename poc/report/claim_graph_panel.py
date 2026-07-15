"""Claim-graph "Source grounding" display composer (display-layer only).

Re-presents the EXPERIMENTAL ``cg-1`` claim-graph container (persisted under
``authorship_evidence.claim_graph``, poc/claim_graph/schema.py) as an
evidentiary, advisory panel — the single server-side source of truth rendered
byte-for-byte by BOTH surfaces (the PDF panel in render_panels.py and the React
``ClaimGraphPanel``). No per-surface derivation: both consume THIS dict.

Design (owner-approved, evidentiary-not-accusatory):
  * verified-to-source  -> green   (a cited source ENTAILS the claim)
  * cited-source-contradicts -> red (the ONLY red; source refutes the claim)
  * paywalled / unresolved -> gray  ("not held against you")
  * teacher-probe QUESTIONS for the unverified specifics

DISPLAY-ONLY, exactly like poc/report/authorship_evidence_levels.py: ``scoring``
is hard-False; this NEVER touches the tier, the ai_likelihood_score, any gate, or
any scoring path. It only appears when the claim graph is present (the graph is
itself gated by DRAFTPROOF_CLAIM_GRAPH) — returns ``None`` when the container is
absent/empty, so with the flag OFF the report is byte-identical.

Pure function: deterministic, NO network / LLM / heavy imports. Fail-open — any
error returns ``None`` rather than breaking a report build.
"""
from __future__ import annotations

from typing import Any, Optional

_EXCERPT_MAX = 140
_LOCATOR_MAX = 80
_SOURCE_CHECKS_CAP = 8
_QUESTIONS_CAP = 6

# Row severity for deterministic ordering — the contradiction (the only red) must
# survive the cap, so it sorts first; "not held against you" grays sort last.
_STATUS_PRIORITY = {"contradicted": 0, "verified": 1, "paywalled": 2, "unresolved": 3}


def _trim(text: Any, limit: int) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit].rstrip() + "…"  # ellipsis


def _interrog_signal(container: dict) -> Optional[dict]:
    for sig in (container.get("signals") or []):
        if isinstance(sig, dict) and sig.get("signal") == "interrogatability":
            return sig
    return None


def _row_status(resolution: dict, verdict: Optional[dict]) -> str:
    """Map a resolution.status + a per-claim entailment verdict to a row status.

    resolved + verified   -> "verified"
    resolved + contradicted -> "contradicted"
    paywalled             -> "paywalled"
    everything else (unresolved / error / resolved-but-unverified) -> "unresolved"
    (gray "not held against you").
    """
    status = str((resolution or {}).get("status") or "").strip().lower()
    if status == "paywalled":
        return "paywalled"
    if status == "resolved" and isinstance(verdict, dict):
        vs = str(verdict.get("status") or "").strip().lower()
        if vs == "verified":
            return "verified"
        if vs == "contradicted":
            return "contradicted"
    return "unresolved"


def _source_checks(container: dict, claims_by_id: dict) -> list[dict]:
    rows: list[dict] = []
    for ev in (container.get("evidence") or []):
        if not isinstance(ev, dict):
            continue
        detail = ev.get("detail")
        if not isinstance(detail, dict):
            continue
        resolution = detail.get("resolution")
        if not isinstance(resolution, dict):
            continue
        entailment = resolution.get("entailment")
        entailment = entailment if isinstance(entailment, dict) else {}
        claim_ids = [cid for cid in (ev.get("claim_ids") or []) if isinstance(cid, str)]
        if not claim_ids:
            continue
        # Locator string, e.g. "doi 10.1177/..." / "url https://..." (truncated).
        loc_type = str(resolution.get("locator_type") or detail.get("locator_type") or "").strip()
        loc_val = str(resolution.get("locator") or detail.get("marker") or "").strip()
        locator = _trim((f"{loc_type} {loc_val}").strip(), _LOCATOR_MAX)
        for cid in sorted(claim_ids):
            claim = claims_by_id.get(cid) or {}
            verdict = entailment.get(cid) if isinstance(entailment.get(cid), dict) else None
            status = _row_status(resolution, verdict)
            score = None
            if verdict is not None:
                try:
                    score = round(float(verdict.get("entailment_score")), 4)
                except (TypeError, ValueError):
                    score = None
            rows.append({
                "status": status,
                "claim_excerpt": _trim(claim.get("text"), _EXCERPT_MAX),
                "locator": locator,
                "entailment_score": score,
            })
    # Severity ordering (stable) so the cap keeps the most decision-relevant rows.
    rows.sort(key=lambda r: _STATUS_PRIORITY.get(r["status"], 9))
    return rows[:_SOURCE_CHECKS_CAP]


def compose_claim_graph_display(claim_graph: Any) -> Optional[dict[str, Any]]:
    """Compose the pinned ``claim_graph_display`` contract, or ``None``.

    Returns ``None`` when the container is absent, not a dict, or empty (no
    claims and no evidence) — the panels then render nothing (prod byte-
    identical). Otherwise returns the display-only dict both surfaces consume.
    Fail-open: any error -> ``None``.
    """
    try:
        if not isinstance(claim_graph, dict):
            return None
        claims = [c for c in (claim_graph.get("claims") or []) if isinstance(c, dict)]
        evidence = [e for e in (claim_graph.get("evidence") or []) if isinstance(e, dict)]
        if not claims and not evidence:
            return None

        claims_by_id = {c.get("id"): c for c in claims if c.get("id")}
        claim_nodes = [c for c in claims if str(c.get("node_type") or "").upper() == "CLAIM"]
        question_nodes = [c for c in claims if str(c.get("node_type") or "").upper() == "QUESTION"]

        # ── Summary counts (§ derivations). ──
        sig = _interrog_signal(claim_graph)
        sig_value = (sig or {}).get("value") if isinstance((sig or {}).get("value"), dict) else {}
        sig_coverage = (sig or {}).get("coverage") if isinstance((sig or {}).get("coverage"), dict) else {}

        total_specifics = sig_coverage.get("total_specifics")
        specific_claims = int(total_specifics) if isinstance(total_specifics, int) else len(claim_nodes)

        verified_to_source = sum(
            1 for c in claim_nodes
            if str(c.get("verification_status") or "").strip().lower() == "verified"
        )

        questions_emitted = sig_value.get("questions_emitted")
        need_grounding = int(questions_emitted) if isinstance(questions_emitted, int) else len(question_nodes)

        # ── Source checks (evidence[] resolved/entailed). ──
        source_checks = _source_checks(claim_graph, claims_by_id)

        # ── Teacher-probe questions. ──
        questions = [
            _trim(q.get("text"), 240) for q in question_nodes
            if str(q.get("text") or "").strip()
        ][:_QUESTIONS_CAP]

        # ── Coverage channels + limitations (pass-through). ──
        cov = claim_graph.get("source_coverage")
        coverage_channels = []
        if isinstance(cov, dict) and isinstance(cov.get("value"), list):
            coverage_channels = [str(v) for v in cov["value"] if isinstance(v, str)]
        limitations = [
            str(x) for x in (claim_graph.get("source_limitations") or [])
            if isinstance(x, str)
        ]

        # ── Advisory status: interrogatability signal status, else container. ──
        status = str(
            (sig or {}).get("status")
            or claim_graph.get("status")
            or "experimental"
        ).strip().lower()

        return {
            "present": True,
            "scoring": False,  # ALWAYS advisory — never affects the score.
            "status": status,
            "summary": {
                "specific_claims": specific_claims,
                "verified_to_source": verified_to_source,
                "need_grounding": need_grounding,
            },
            "source_checks": source_checks,
            "questions": questions,
            "coverage_channels": coverage_channels,
            "limitations": limitations,
        }
    except Exception:
        return None


__all__ = ["compose_claim_graph_display"]
