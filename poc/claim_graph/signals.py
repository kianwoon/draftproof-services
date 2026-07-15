"""Phase-1 EXPERIMENTAL graph signals (plan §4, M3).

Three signals computed on the VALIDATED graph container, on the FULL node/edge
set *before* eviction (plan §1 rule: truncation never changes a signal value):

  * ``interrogatability``  — deterministic, GRAPH STRUCTURE ONLY (no LLM). Rich,
    traceable specifics score high; each UNVERIFIED specific also emits a
    ``QUESTION`` node (teacher probe / audit surface, NEVER an ownership credit —
    headline invariant §A). Banded low/moderate/high + per-component breakdown.
  * ``substitutability`` — MiniLM noun-swap test (plan §6/4b). Embeds each CLAIM
    sentence and a topic-neutralised variant (specifics masked with generic
    placeholders) with the SHARED ``poc.detect.semantic_shape`` MiniLM embedder;
    high cosine (meaning survives the swap) ⇒ high substitutability ⇒ weak
    grounding. Fails OPEN to ``value=None`` when MiniLM is unavailable.
  * ``origin_map`` — deterministic multi-label aggregation over each CLAIM's
    ``origins[]``/``primary_origin`` (extracted in M2) → document origin
    distribution + primary; flags ``unsupported_assertion``-heavy documents.

Every signal object carries the §B lifecycle metadata contract
``{signal, status:"experimental", scoring_enabled:false, calibration_version:null,
fairness_gate_passed:null, ...}`` and records the sibling signal it RECONCILES
with (never duplicates — plan §4). All numeric thresholds are PROVISIONAL (no
calibration exists in Phase 1); they are marked in code + in each signal's
``provisional_thresholds`` metadata so nothing reads as calibrated.

Fail-open by construction: ``compute_signals`` computes each signal in its own
try/except, so one signal raising can never drop the other two or the graph.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .schema import ClaimNode, Edge

# ── Provisional bands (NOT calibrated — Phase 4 §B) ──────────────────────────
# Interrogatability composite band cuts.
_INTERROG_LOW = 0.33   # provisional
_INTERROG_HIGH = 0.66  # provisional
# Substitutability (mean cosine, higher = worse) band cuts.
_SUBST_LOW = 0.80   # provisional
_SUBST_HIGH = 0.90  # provisional
# Origin-map "unsupported-assertion-heavy" rate cut.
_UNSUPPORTED_HEAVY = 0.50  # provisional
# Cap on emitted QUESTION nodes so a specific-dense doc cannot blow the graph.
_MAX_QUESTIONS = 50

_CAUSAL_EDGE_TYPES = ("depends_on", "explains", "causes")
_SUPPORTED_STATE = "internally_supported"
_UNVERIFIED_STATES = ("unverified", "unverifiable")
# N4: the ONLY state that earns a specificity CREDIT when the verified-only gate
# is on (plan §3 verdict asymmetry — external ownership credit is verified-only).
_VERIFIED_STATE = "verified"

_META_BASE = {
    "status": "experimental",
    "scoring_enabled": False,
    "calibration_version": None,
    "fairness_gate_passed": None,
}


def _meta(signal: str, **extra: Any) -> dict[str, Any]:
    out = {"signal": signal}
    out.update(_META_BASE)
    out.update(extra)
    return out


# ── Deterministic "specific" extractor (shared by interrogatability + swap) ──
# Provisional heuristics (NOT NER): numbers/percentages, 4-digit years, and
# capitalised proper-noun tokens not at sentence start. Marked provisional.
_RE_PERCENT = re.compile(r"\d[\d,]*(?:\.\d+)?\s?%")
_RE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_RE_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_RE_PROPER = re.compile(r"\b[A-Z][a-zA-Z]+\b")

_PLACEHOLDER = {
    "percent": "some amount",
    "year": "some year",
    "number": "some number",
    "proper_noun": "something",
}


def extract_specifics(text: str) -> list[dict[str, Any]]:
    """Return non-overlapping specific spans ``{start, end, kind, text}``.

    Deterministic; provisional heuristics (§4a). Percent/year take precedence
    over bare-number, proper-noun tokens skip the first token of the string
    (sentence-initial capitalisation is not evidence of a named entity)."""
    text = text or ""
    spans: list[dict[str, Any]] = []
    taken: list[tuple[int, int]] = []

    def _overlaps(a: int, b: int) -> bool:
        return any(not (b <= s or a >= e) for s, e in taken)

    for regex, kind in ((_RE_PERCENT, "percent"), (_RE_YEAR, "year"),
                        (_RE_NUMBER, "number"), (_RE_PROPER, "proper_noun")):
        for m in regex.finditer(text):
            a, b = m.start(), m.end()
            if _overlaps(a, b):
                continue
            if kind == "proper_noun" and a == 0:
                continue  # sentence-initial capital: not a named entity
            taken.append((a, b))
            spans.append({"start": a, "end": b, "kind": kind, "text": m.group(0)})
    spans.sort(key=lambda s: s["start"])
    return spans


def _neutralise(text: str, specifics: list[dict[str, Any]]) -> str:
    """Mask specific spans with generic placeholders (descending offsets)."""
    out = text or ""
    for sp in sorted(specifics, key=lambda s: s["start"], reverse=True):
        out = out[: sp["start"]] + _PLACEHOLDER[sp["kind"]] + out[sp["end"]:]
    return out


def _band(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value < high:
        return "moderate"
    return "high"


def _claims_only(claims: list[ClaimNode]) -> list[ClaimNode]:
    return [c for c in claims if c.node_type == "CLAIM"]


# ── 4a. Interrogatability (deterministic, graph-structure-only) ──────────────
def compute_interrogatability(
    claims: list[ClaimNode], edges: list[Edge],
    *, specificity_verified_only: bool = False,
) -> tuple[dict[str, Any], list[ClaimNode]]:
    """Return ``(signal, question_nodes)``.

    Components (each normalised to [0,1], provisional):
      * specificity_presence  — specific density across CLAIM nodes
      * causal_depth          — causal edges (depends_on/explains/causes) per claim
      * contextual_anchoring  — fraction of CLAIMs with a non-empty origin set
      * evidence_traceability — fraction of CLAIMs internally_supported

    QUESTION emission (headline invariant §A): every specific on an
    unverified/unverifiable CLAIM emits a teacher-probe QUESTION referencing that
    claim — an AUDIT surface, never a credit. Value carries the composite plus
    the ``unverified_specific_rate``.

    ``specificity_verified_only`` (N4, plan [G3] + M4 rec (f)1 — the gaming fix):
      * ``False`` (default, Phase-1/M3 parity): ``specificity_presence`` credits
        ALL specifics — byte-identical to the pre-N4 signal.
      * ``True`` (entailment ON): credit ONLY specifics on ``verified`` claims, so
        an unentailed/fabricated-source specific earns NOTHING. QUESTION emission
        and ``unverified_specific_rate`` are UNCHANGED in both modes (still driven
        by ``_UNVERIFIED_STATES``) — only the credit numerator moves.
    """
    cnodes = _claims_only(claims)
    n = len(cnodes)
    questions: list[ClaimNode] = []

    total_specifics = 0
    credited_specifics = 0
    unverified_specifics = 0
    for c in cnodes:
        specs = extract_specifics(c.text)
        total_specifics += len(specs)
        # Credit numerator: ALL specifics by default; verified-only when gated.
        if not specificity_verified_only or c.verification_status == _VERIFIED_STATE:
            credited_specifics += len(specs)
        if c.verification_status in _UNVERIFIED_STATES:
            unverified_specifics += len(specs)
            for sp in specs:
                if len(questions) >= _MAX_QUESTIONS:
                    break
                questions.append(ClaimNode(
                    id="",  # assigned by caller after append
                    node_type="QUESTION",
                    text='What is the source or basis for "%s"?' % sp["text"],
                    claim_type=None,
                    source=None,
                    verification_status="unverified",
                    references=c.id,
                ))

    causal_edges = sum(1 for e in edges if e.type in _CAUSAL_EDGE_TYPES)
    anchored = sum(1 for c in cnodes if c.origins)
    traceable = sum(1 for c in cnodes if c.verification_status == _SUPPORTED_STATE)

    if n == 0:
        components = {k: 0.0 for k in (
            "specificity_presence", "causal_depth",
            "contextual_anchoring", "evidence_traceability")}
    else:
        components = {
            # provisional normalisers
            "specificity_presence": min(1.0, credited_specifics / (n * 2.0)),
            "causal_depth": min(1.0, causal_edges / float(n)),
            "contextual_anchoring": anchored / float(n),
            "evidence_traceability": traceable / float(n),
        }
    composite = round(sum(components.values()) / 4.0, 4)
    components = {k: round(v, 4) for k, v in components.items()}
    unverified_specific_rate = round(
        unverified_specifics / total_specifics, 4) if total_specifics else 0.0

    limitations = ["high interrogatability of an UNVERIFIED specific is an audit "
                   "surface (questions), never an ownership credit (§A)",
                   "specific detection is a provisional heuristic, not NER"]
    extra: dict[str, Any] = {}
    # Record the credit mode ONLY when gated ON — keeps the default (OFF) signal
    # byte-identical to pre-N4 for the parity contract, while giving N6 a
    # structured marker of the verified-only credit realisation ([G3]).
    if specificity_verified_only:
        extra["specificity_verified_only"] = True
        limitations.append("specificity credit is VERIFIED-ONLY (N4 gaming fix, "
                           "M4 rec (f)1): only entailed specifics earn credit")

    sig = _meta(
        "interrogatability",
        reconciles_with=["citation", "grounding_diagnosis"],
        value={"composite": composite,
               "unverified_specific_rate": unverified_specific_rate,
               "questions_emitted": len(questions)},
        band=_band(composite, _INTERROG_LOW, _INTERROG_HIGH),
        components=components,
        coverage={"claims_considered": n, "total_specifics": total_specifics},
        provisional_thresholds={"low": _INTERROG_LOW, "high": _INTERROG_HIGH,
                                "note": "uncalibrated; Phase-4 §B"},
        limitations=limitations,
        **extra,
    )
    return sig, questions


# ── 4b. Substitutability (MiniLM noun-swap) ──────────────────────────────────
def resolve_embedder() -> Optional[Any]:
    """Return a real MiniLM embedder (``.encode``), or ``None`` when unavailable.

    Reuses the shared ``poc.detect.semantic_shape`` embedder (the worker-preloaded
    ``all-MiniLM-L6-v2``); lazy-imported so this module stays import-light. Returns
    ``None`` — never the hashed-lexical fallback — when the real model can't load,
    so substitutability fails open rather than scoring on a fake embedder."""
    try:
        from poc.detect.semantic_shape import SemanticShapeDetector
    except Exception:
        try:
            from detect.semantic_shape import SemanticShapeDetector  # sys.path shim
        except Exception:
            return None
    try:
        det = SemanticShapeDetector()
        if not det._ensure_embedder() or not det._embedding_available:
            return None
        return det._embedder
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (na * nb)


def _fail_open_substitutability(note: str, n: int) -> dict[str, Any]:
    return _meta(
        "substitutability",
        reconciles_with=["generic_assertion_risk"],
        value=None,
        band=None,
        coverage={"claims_considered": n, "claims_embedded": 0},
        provisional_thresholds={"low": _SUBST_LOW, "high": _SUBST_HIGH,
                                "note": "uncalibrated; Phase-4 §B"},
        limitations=[note],
    )


def compute_substitutability(
    claims: list[ClaimNode], embedder: Optional[Any]
) -> dict[str, Any]:
    """MiniLM noun-swap test. ``embedder=None`` → fail open (value=None).

    For each CLAIM: mask its specifics with generic placeholders, embed original +
    neutralised variant, take cosine. High mean cosine ⇒ high substitutability ⇒
    weak grounding (higher = worse). Deterministic given the embedder."""
    cnodes = _claims_only(claims)
    if embedder is None:
        return _fail_open_substitutability("embedding model unavailable", len(cnodes))
    if not cnodes:
        sig = _fail_open_substitutability("no CLAIM nodes to embed", 0)
        return sig

    cosines: list[float] = []
    try:
        for c in cnodes:
            neutral = _neutralise(c.text, extract_specifics(c.text))
            vecs = embedder.encode([c.text, neutral])
            vecs = [list(map(float, v)) for v in vecs]
            cosines.append(max(-1.0, min(1.0, _cosine(vecs[0], vecs[1]))))
    except Exception as exc:  # embed failure → fail open, don't crash the graph
        return _fail_open_substitutability(
            "embedding failed: %s" % str(exc)[:120], len(cnodes))

    mean_cos = round(sum(cosines) / len(cosines), 4)
    return _meta(
        "substitutability",
        reconciles_with=["generic_assertion_risk"],
        value=mean_cos,
        band=_band(mean_cos, _SUBST_LOW, _SUBST_HIGH),
        coverage={"claims_considered": len(cnodes), "claims_embedded": len(cosines)},
        provisional_thresholds={"low": _SUBST_LOW, "high": _SUBST_HIGH,
                                "note": "uncalibrated; Phase-4 §B"},
        limitations=["necessary-not-sufficient; never an independent credit (§I)",
                     "AUGMENTS generic_assertion_risk (embedding noun-swap vs "
                     "lexical); it does not replace it",
                     "specific masking is a provisional heuristic, not NER"],
    )


# ── 4c. Origin map (deterministic multi-label) ───────────────────────────────
def compute_origin_map(claims: list[ClaimNode]) -> dict[str, Any]:
    """Aggregate CLAIM ``origins[]``/``primary_origin`` → document distribution +
    primary; flag ``unsupported_assertion``-heavy documents.

    An ``unsupported_assertion`` is a CLAIM with no external/analytic origin AND
    an unverified/unverifiable status (descriptive audit metadata — origin is
    NEVER a standalone credit; ``personal_observation`` maps to NEUTRAL, §I)."""
    cnodes = _claims_only(claims)
    n = len(cnodes)

    label_counts: dict[str, int] = {}
    primary_counts: dict[str, int] = {}
    unsupported = 0
    total_labels = 0
    for c in cnodes:
        for o in c.origins:
            label_counts[o] = label_counts.get(o, 0) + 1
            total_labels += 1
        if c.primary_origin:
            primary_counts[c.primary_origin] = primary_counts.get(c.primary_origin, 0) + 1
        grounded = any(o in ("external_source", "original_analysis") for o in c.origins)
        if not grounded and c.verification_status in _UNVERIFIED_STATES:
            unsupported += 1

    distribution = {k: round(v / total_labels, 4) for k, v in sorted(label_counts.items())} \
        if total_labels else {}
    primary_origin = max(sorted(primary_counts), key=primary_counts.get) \
        if primary_counts else None
    unsupported_rate = round(unsupported / n, 4) if n else 0.0

    return _meta(
        "origin_map",
        reconciles_with=["authorship_evidence", "critical_thinking"],
        value={"distribution": distribution,
               "primary_origin": primary_origin,
               "unsupported_assertion_rate": unsupported_rate,
               "unsupported_assertion_heavy": unsupported_rate > _UNSUPPORTED_HEAVY},
        coverage={"claims_considered": n, "total_labels": total_labels},
        provisional_thresholds={"unsupported_heavy": _UNSUPPORTED_HEAVY,
                                "note": "uncalibrated; Phase-4 §B"},
        limitations=["origin is descriptive audit metadata, never a standalone "
                     "credit; personal_observation is NEUTRAL (§I)"],
    )


# ── Orchestrator: fail-open isolation ────────────────────────────────────────
def compute_signals(
    claims: list[ClaimNode],
    edges: list[Edge],
    text: str = "",
    embedder: Optional[Any] = "__resolve__",
    *,
    specificity_verified_only: bool = False,
) -> tuple[list[dict[str, Any]], list[ClaimNode]]:
    """Compute all three signals + QUESTION nodes on the FULL graph.

    Each signal is computed in isolation: one raising yields a fail-open signal
    object with ``value=None`` + an error note and never drops the others or the
    graph. Pass ``embedder`` explicitly (tests) or leave the sentinel to resolve
    the shared MiniLM embedder lazily (``None`` when unavailable → substitutability
    fails open). Deterministic given the embedder.

    ``specificity_verified_only`` (N4) is forwarded to interrogatability so the
    verified-aware caller (build.py's entailment-ON path) credits only verified
    specifics; default ``False`` preserves the pre-N4 signal exactly."""
    signals: list[dict[str, Any]] = []
    questions: list[ClaimNode] = []

    # interrogatability (+ questions)
    try:
        sig, qs = compute_interrogatability(
            claims, edges, specificity_verified_only=specificity_verified_only)
        signals.append(sig)
        questions = qs
    except Exception as exc:
        signals.append(_meta("interrogatability", value=None, band=None,
                             limitations=["signal error: %s" % str(exc)[:120]]))

    # substitutability
    try:
        emb = resolve_embedder() if embedder == "__resolve__" else embedder
        signals.append(compute_substitutability(claims, emb))
    except Exception as exc:
        signals.append(_fail_open_substitutability(
            "signal error: %s" % str(exc)[:120], len(_claims_only(claims))))

    # origin_map
    try:
        signals.append(compute_origin_map(claims))
    except Exception as exc:
        signals.append(_meta("origin_map", value=None,
                             limitations=["signal error: %s" % str(exc)[:120]]))

    return signals, questions


__all__ = [
    "extract_specifics",
    "compute_interrogatability",
    "compute_substitutability",
    "compute_origin_map",
    "compute_signals",
    "resolve_embedder",
]
