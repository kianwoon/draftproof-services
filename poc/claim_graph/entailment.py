"""Claim↔source entailment engine (Phase-2 N3, Track A) — the load-bearing part.

This is N3 in the Phase-2 entailment pipeline
(``docs/plans/phase2_entailment_grounding_scope.md`` §3/§4). Its input is the
``resolution`` block N2 (``sources.py``) attaches to each resolved
``external_citation`` evidence node: a retrieved ``source_text``. N3 tests, for
each linked claim, whether that source ENTAILS the claim, using an NLI
cross-encoder hosted on Modal ([G1] — a cross-encoder, NOT an LLM-judge: the
premise is retrieved web text and an LLM-judge reading it is directly
prompt-injectable; a cross-encoder consumes premise/hypothesis as scored text,
not instructions).

  premise    = a retrieved SOURCE passage (windowed from ``source_text``)
  hypothesis = the CLAIM text

MILESTONE BOUNDARY (read before touching this): N3 COMPUTES the entailment
verdict and ATTACHES it (the caller writes it into
``resolution["entailment"]``). **N3 does NOT set ``claim.verification_status``**
— that wire-back + the interrogatability VERIFIED-ONLY re-gating is N4. Keeping
that separation clean is what makes N4 a small, reviewable step. This module
therefore returns a plain result dict and never mutates a ClaimNode.

The §3 verdict asymmetry (the fairness core), realised in ``classify``:
  * ``verified``     iff max entailment ≥ ``verified_threshold`` AND clearly
                     dominant (entailment > contradiction). The ONLY path to
                     external credit.
  * ``contradicted`` iff max contradiction ≥ ``contradicted_threshold``, where
                     ``contradicted_threshold`` is STRICTLY GREATER than
                     ``verified_threshold`` — a false ``contradicted`` on genuine
                     work is the §B2 fairness harm, so the bar is stricter.
  * else             ``unverified`` (abstain; the gray zone falls HERE, NEVER to
                     ``contradicted``).
Thresholds live in ``entailment_thresholds.json`` (provenance-marked,
``status=provisional``, calibrated in N6) — never hardcoded in logic.

Fail-open contract: the scorer returns ``None`` (endpoint unavailable/timeout/
non-200/malformed) → the claim degrades to ``unverified`` with
``reason=endpoint_unavailable``. Never crashes, never fabricates a verdict.

Import-light: this module imports NO ``modal``/``torch``/``transformers`` and no
heavy stack at load; the Modal client lazily imports ``requests`` inside its
call path (mirrors ``sources.py`` and ``detect_v7/modal_client.py``). The Modal
endpoint code lives in ``modal_endpoints/entailment_nli.py`` and is NEVER
imported by this package.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

# ── Verdict states (mirror the schema verification states; N4 maps these onto
# ``verification_status``). Kept as constants so callers/tests never hardcode
# the literals. ──────────────────────────────────────────────────────────────
STATUS_VERIFIED = "verified"
STATUS_CONTRADICTED = "contradicted"
STATUS_UNVERIFIED = "unverified"

# NLI class labels (normalised order the endpoint guarantees).
LABEL_ENTAILMENT = "entailment"
LABEL_NEUTRAL = "neutral"
LABEL_CONTRADICTION = "contradiction"

_FALSEY = {"0", "false", "no", "off", ""}

# Endpoint env vars — INDEPENDENT of the deep-scan detector's vars (this is a
# separate Modal app, ``modal_endpoints/entailment_nli.py``).
_URL_ENV_VAR = "DRAFTPROOF_ENTAILMENT_ENDPOINT_URL"
_TOKEN_ENV_VAR = "DRAFTPROOF_ENTAILMENT_ENDPOINT_TOKEN"
_DEFAULT_TIMEOUT_S = 60.0

_THRESHOLDS_FILENAME = "entailment_thresholds.json"

# Hard fallbacks used ONLY if the config file is missing/corrupt (fail-open).
# The asymmetry (contradicted > verified) is preserved here too. These are the
# same provisional values as the JSON — the JSON is the source of truth.
_FALLBACK_THRESHOLDS = {
    "verified_threshold": 0.65,
    "contradicted_threshold": 0.8,
    "max_passages": 12,
    "min_passage_chars": 24,
    "status": "provisional",
    "calibration_version": None,
    "note": "in-code fallback; entailment_thresholds.json missing or unreadable",
}


# ── env helpers ──────────────────────────────────────────────────────────────
def _int_env(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


# ── threshold config (NO hardcoded thresholds in logic) ──────────────────────
def _thresholds_path() -> Path:
    override = os.environ.get("DRAFTPROOF_ENTAILMENT_THRESHOLDS_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / _THRESHOLDS_FILENAME


def load_thresholds(path: Any = None) -> dict[str, Any]:
    """Load the provisional threshold config. Fail-open to the in-code fallback
    (which preserves the contradicted>verified asymmetry) if unreadable."""
    p = Path(path) if path is not None else _thresholds_path()
    try:
        data = json.loads(p.read_text("utf-8"))
        if not isinstance(data, dict):
            return dict(_FALLBACK_THRESHOLDS)
        merged = dict(_FALLBACK_THRESHOLDS)
        merged.update(data)
        return merged
    except Exception:
        return dict(_FALLBACK_THRESHOLDS)


# ── threshold → status mapping (pure, exhaustively tested) ───────────────────
def classify(entailment_score: float, contradiction_score: float,
             thresholds: dict[str, Any]) -> str:
    """Map the aggregated (max entailment, max contradiction) scalars to a
    verdict via the §3 asymmetric thresholds. Pure function.

    ``verified`` requires entailment to clear its bar AND dominate contradiction.
    ``contradicted`` requires contradiction to clear the STRICTER bar. Everything
    else — including any gray zone — is ``unverified`` (never ``contradicted``).
    """
    v_th = float(thresholds.get("verified_threshold",
                                _FALLBACK_THRESHOLDS["verified_threshold"]))
    c_th = float(thresholds.get("contradicted_threshold",
                                _FALLBACK_THRESHOLDS["contradicted_threshold"]))
    e = float(entailment_score)
    c = float(contradiction_score)
    if e >= v_th and e > c:
        return STATUS_VERIFIED
    if c >= c_th:
        return STATUS_CONTRADICTED
    return STATUS_UNVERIFIED


# ── deterministic source windowing (mirror deep-scan windowing discipline) ───
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")


def window_source(source_text: Any, *, max_passages: int,
                  min_passage_chars: int) -> tuple[list[str], bool]:
    """Split ``source_text`` into ordered passages, deterministically.

    Sentence-boundary split (``. ! ?`` + whitespace), whitespace-normalised,
    fragments shorter than ``min_passage_chars`` dropped. Capped at
    ``max_passages`` — if more remain, keep the FIRST ``max_passages`` (stable,
    front-anchored like the abstract lead) and flag ``truncated=True``.

    Returns ``(passages, truncated)``. Empty/blank source → ``([], False)``.
    """
    text = _WS_RE.sub(" ", str(source_text or "")).strip()
    if not text:
        return [], False
    raw = _SENT_SPLIT_RE.split(text)
    passages = [s.strip() for s in raw if len(s.strip()) >= max(0, int(min_passage_chars))]
    if not passages:
        # Nothing cleared the min-length bar — fall back to the whole blob as one
        # passage so a short-but-real source still gets scored (annotate, don't
        # suppress). Only when the source itself is non-empty.
        passages = [text]
    cap = max(1, int(max_passages))
    if len(passages) > cap:
        return passages[:cap], True
    return passages, False


# ── the entailment scorer seam (inject a fake in tests) ──────────────────────
class EntailmentScorer(Protocol):
    """Scores premise↔hypothesis pairs. ``score_pairs`` returns one softmax dict
    ``{"entailment","neutral","contradiction"}`` per input pair, in order, OR
    ``None`` on any failure (fail-open → the claim degrades to ``unverified``)."""

    def score_pairs(self, pairs: list[tuple[str, str]]
                    ) -> Optional[list[dict[str, float]]]:
        ...


@dataclass
class ModalEntailmentScorer:
    """Production scorer: POSTs pairs to the Modal NLI endpoint
    (``modal_endpoints/entailment_nli.py``). Fail-open — NEVER raises into the
    caller; ANY failure (unset env, timeout, connection error, non-200,
    malformed JSON) returns ``None``, which the engine degrades to
    ``unverified``. Mirrors ``detect_v7/modal_client.call_deep_scan``.

    ``http_post`` is injectable so the suite is fully offline; the default lazily
    imports ``requests`` (keeps this module import-light)."""

    http_post: Optional[Callable[..., Any]] = None
    timeout_s: float = _DEFAULT_TIMEOUT_S
    url_env_var: str = _URL_ENV_VAR
    token_env_var: str = _TOKEN_ENV_VAR

    def _post(self):
        if self.http_post is not None:
            return self.http_post

        def _default(url, json=None, headers=None, timeout=None):
            import requests  # lazy — keeps module import-light

            return requests.post(url, json=json, headers=headers, timeout=timeout)

        return _default

    def score_pairs(self, pairs: list[tuple[str, str]]
                    ) -> Optional[list[dict[str, float]]]:
        url = os.environ.get(self.url_env_var)
        token = os.environ.get(self.token_env_var)
        if not url or not token:
            return None  # unconfigured here → "Modal isn't available", not an error
        if not pairs:
            return None
        body = {"pairs": [{"premise": str(p), "hypothesis": str(h)} for p, h in pairs]}
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        try:
            resp = self._post()(url, json=body, headers=headers, timeout=self.timeout_s)
        except Exception:
            return None
        if int(getattr(resp, "status_code", 0) or 0) != 200:
            return None
        try:
            payload = resp.json()
        except Exception:
            return None
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list) or len(results) != len(pairs):
            return None
        out: list[dict[str, float]] = []
        for item in results:
            scores = item.get("scores") if isinstance(item, dict) else None
            if not isinstance(scores, dict):
                return None
            try:
                out.append({
                    LABEL_ENTAILMENT: float(scores.get(LABEL_ENTAILMENT, 0.0)),
                    LABEL_NEUTRAL: float(scores.get(LABEL_NEUTRAL, 0.0)),
                    LABEL_CONTRADICTION: float(scores.get(LABEL_CONTRADICTION, 0.0)),
                })
            except (TypeError, ValueError):
                return None
        return out


def default_scorer() -> ModalEntailmentScorer:
    """The env-driven production scorer (fail-open when the endpoint is unset)."""
    return ModalEntailmentScorer()


# ── the engine ───────────────────────────────────────────────────────────────
def _label_of(scores: dict[str, float]) -> str:
    return max(
        (LABEL_ENTAILMENT, LABEL_NEUTRAL, LABEL_CONTRADICTION),
        key=lambda k: float(scores.get(k, 0.0)),
    )


def _unverified(reason: str, *, passages_scored: int = 0, truncated: bool = False
                ) -> dict[str, Any]:
    return {
        "status": STATUS_UNVERIFIED,
        "label": LABEL_NEUTRAL,
        "entailment_score": 0.0,
        "contradiction_score": 0.0,
        "evidence_passage_index": None,
        "passages_scored": passages_scored,
        "truncated": truncated,
        "reason": reason,
    }


def entail_claim(claim_text: Any, source_text: Any, scorer: EntailmentScorer, *,
                 thresholds: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Test whether ``source_text`` entails ``claim_text`` (premise=source,
    hypothesis=claim). Windows the source, scores the claim against each passage,
    aggregates (max-entailment passage = support span, max-contradiction passage
    = refute span), and maps to a verdict via ``classify``.

    Returns ``{status, label, entailment_score, contradiction_score,
    evidence_passage_index, passages_scored, truncated}`` (plus ``reason`` on the
    abstain paths). NEVER raises; NEVER touches ``verification_status`` (N4)."""
    th = thresholds if thresholds is not None else load_thresholds()
    try:
        claim = str(claim_text or "").strip()
        if not claim:
            return _unverified("no_claim_text")

        max_passages = _int_env("DRAFTPROOF_ENTAILMENT_MAX_PASSAGES",
                                 int(th.get("max_passages", 12)))
        min_chars = int(th.get("min_passage_chars", 24))
        passages, truncated = window_source(
            source_text, max_passages=max_passages, min_passage_chars=min_chars)
        if not passages:
            return _unverified("no_source_text", truncated=truncated)

        pairs = [(p, claim) for p in passages]
        scored = scorer.score_pairs(pairs)
        if scored is None:
            return _unverified("endpoint_unavailable", truncated=truncated)
        if len(scored) != len(passages):
            # Contract violation → abstain, never guess.
            return _unverified("scorer_shape_mismatch",
                               passages_scored=0, truncated=truncated)

        ent = [float(s.get(LABEL_ENTAILMENT, 0.0)) for s in scored]
        con = [float(s.get(LABEL_CONTRADICTION, 0.0)) for s in scored]
        max_ent = max(ent)
        max_con = max(con)
        # argmax with earliest-index tie-break (determinism).
        ent_idx = ent.index(max_ent)
        con_idx = con.index(max_con)

        status = classify(max_ent, max_con, th)
        evidence_idx = con_idx if status == STATUS_CONTRADICTED else ent_idx
        label = _label_of(scored[evidence_idx])

        return {
            "status": status,
            "label": label,
            "entailment_score": max_ent,
            "contradiction_score": max_con,
            "evidence_passage_index": evidence_idx,
            "passages_scored": len(passages),
            "truncated": truncated,
        }
    except Exception:
        # Catastrophic fail-open — degrade to unverified, never crash the build.
        return _unverified("engine_error")


__all__ = [
    "STATUS_VERIFIED",
    "STATUS_CONTRADICTED",
    "STATUS_UNVERIFIED",
    "LABEL_ENTAILMENT",
    "LABEL_NEUTRAL",
    "LABEL_CONTRADICTION",
    "EntailmentScorer",
    "ModalEntailmentScorer",
    "default_scorer",
    "load_thresholds",
    "classify",
    "window_source",
    "entail_claim",
]
