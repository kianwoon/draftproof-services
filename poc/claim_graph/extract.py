"""Stage-1 paragraph-batched claim extraction (M2, EXPERIMENTAL).

Runs the EXISTING LLM gateway (``poc.llm.gateway``) over the canonical
``structured_sentence_segments`` rows, batched 3-5 paragraphs per call. The LLM
*proposes* claims/edges by quoting exact sentence spans and naming edge types;
the deterministic M1 validators (``validators.py``) own graph truth — every
proposed node passes through them, and rejected nodes are dropped with counters.

Design rules (plan §0/§3/§5):
  - **NO ``rewrite_v6`` imports** (circular-import lesson). The gateway model/key/
    base_url are resolved from the SAME env vars ``rewrite_v6/llm_config.py`` reads
    (Cerebras ``gpt-oss`` cheap default), re-implemented here — no module import.
  - Gateway import is LAZY (inside functions) so ``import claim_graph.extract``
    stays light and the M1 fresh-interpreter purity test holds.
  - Strict JSON output contract; parsed DEFENSIVELY (json-object mode when the
    model supports it, brace-substring fallback otherwise). A batch whose JSON
    fails to parse is queued for a targeted Stage-3 retry (capped).
  - Caching by ``sha256(text + model + prompt_version)`` (plan §5). No clean
    worker-persistent path exists (the worker container is ephemeral; the
    deep-scan cache lives under a calibration dir, not a runtime path), so the
    cache is IN-PROCESS / per-run only — a rescan of identical text within one
    process is a hit. Documented deviation: no new infra dependency invented.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Optional

from . import schema

PROMPT_VERSION = "cg-extract-1"

# Batch discipline (plan §D/§5): 3-5 paragraphs/call, ~5-10 calls/doc.
_DEFAULT_PARAS_PER_BATCH = 4
# Secondary cap: a single paragraph can hold many sentences, and the JSON output
# per sentence is verbose — an oversized batch overflows the model's completion
# budget and the response is truncated (unparseable). Flush a batch once it
# reaches this many sentences, so output stays within max_tokens.
_DEFAULT_MAX_SENTENCES_PER_BATCH = 10
_DEFAULT_MAX_RETRIES = 5  # Stage-3 targeted retries (only parse-failed batches).
# Generous completion budget — the structured claim list is long; too small a cap
# truncates the JSON. Env-tunable for cost control.
_DEFAULT_MAX_TOKENS = 8000

_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
_DEFAULT_MODEL = "openai/gpt-oss-120b"

# In-process extraction cache: sha256(text+model+prompt_version) -> (proposals, stats).
_CACHE: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

_SYSTEM_PROMPT = (
    "You extract a CLAIM GRAPH from a student/author document for an academic "
    "integrity audit. You are given a numbered table of sentences grouped by "
    "paragraph. Return STRICT JSON only, no prose, matching exactly:\n"
    '{"claims": [{"sentence_id": "sNNN", "quote": "<verbatim substring of that '
    'sentence>", "node_type": "CLAIM|INFERENCE|QUESTION", '
    '"claim_type": "factual|analytical|personal_observation|interpretation|conclusion", '
    '"origins": ["external_source|personal_observation|original_analysis|'
    'common_knowledge|assignment_prompt|interpretation"], '
    '"primary_origin": "<one of origins>"}], '
    '"edges": [{"type": "supports|derived_from|corroborated_by|qualified_by|'
    'revises|contradicts|inconsistent_with|depends_on|explains|causes|observed_in|'
    'supported_by", "src_quote": "<verbatim substring>", "dst_quote": "<verbatim '
    'substring>"}]}\n'
    "RULES: (1) For every CLAIM, 'quote' MUST be an EXACT verbatim substring of "
    "the named sentence_id's text — never paraphrase, never invent facts not "
    "present. (2) Only assertions of fact, analysis, interpretation, observation, "
    "or conclusion are CLAIMs. (3) Do NOT mark anything 'verified' or "
    "'corroborated'. (4) Edges connect two claim quotes. (5) If a sentence has no "
    "claim, omit it. Output JSON only."
)


# ── Gateway resolution (mirrors rewrite_v6/llm_config, WITHOUT importing it) ──
def _using_cerebras_direct() -> bool:
    explicit = os.environ.get("DRAFTPROOF_V6_CEREBRAS_DIRECT")
    if explicit is not None and explicit.strip() != "":
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    return bool(os.environ.get("CEREBRAS_API_KEY"))


def _resolved_model() -> str:
    override = os.environ.get("DRAFTPROOF_CLAIM_GRAPH_MODEL")
    if override:
        base = override
    else:
        base = os.environ.get("DRAFTPROOF_V6_WRITER_MODEL") or _DEFAULT_MODEL
    if _using_cerebras_direct():
        # Cerebras wants the bare model name (no "openai/" prefix).
        if base == _DEFAULT_MODEL:
            return "gpt-oss-120b"
        return base.split("/", 1)[1] if base.startswith("openai/") else base
    return base


def resolve_gateway() -> Optional[Any]:
    """Build an ``LLMGateway`` from env, or return ``None`` when no key is set.

    Lazy-imports the gateway so this module stays import-light (M1 purity test).
    Cerebras-direct when ``CEREBRAS_API_KEY`` is present, else OpenRouter/LLM env.
    """
    try:
        from poc.llm.gateway import LLMConfig, LLMGateway  # lazy
    except Exception:  # pragma: no cover - import shape only
        try:
            from llm.gateway import LLMConfig, LLMGateway  # type: ignore
        except Exception:
            return None

    model = _resolved_model()
    if _using_cerebras_direct():
        api_key = os.environ.get("CEREBRAS_API_KEY")
        base_url = _CEREBRAS_BASE_URL
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY")
        base_url = os.environ.get("LLM_BASE_URL") or "https://openrouter.ai/api/v1"
    if not api_key:
        return None
    try:
        return LLMGateway(LLMConfig(model=model, api_key=api_key, base_url=base_url))
    except Exception:
        return None


def _gateway_model(gateway: Any) -> str:
    return str(getattr(gateway, "model", "") or _resolved_model())


def _response_format(model: str) -> Optional[dict[str, Any]]:
    try:
        from poc.llm.gateway import model_supports_json_object_response_format  # lazy
    except Exception:
        try:
            from llm.gateway import model_supports_json_object_response_format  # type: ignore
        except Exception:
            return None
    return {"type": "json_object"} if model_supports_json_object_response_format(model) else None


# ── Batching ─────────────────────────────────────────────────────────────────
def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def paragraph_batches(
    segments: list[dict[str, Any]],
    paras_per_batch: Optional[int] = None,
    max_sentences: Optional[int] = None,
) -> list[list[dict[str, Any]]]:
    """Group segments by ``paragraph_id`` (document order), accumulate whole
    paragraphs into a batch, flushing when either ``paras_per_batch`` paragraphs
    OR ``max_sentences`` sentences have accumulated. Sentences within a paragraph
    are never split across batches (a single oversized paragraph still forms its
    own batch)."""
    per = paras_per_batch or _int_env(
        "DRAFTPROOF_CLAIM_GRAPH_PARAS_PER_BATCH", _DEFAULT_PARAS_PER_BATCH
    )
    max_sent = max_sentences or _int_env(
        "DRAFTPROOF_CLAIM_GRAPH_MAX_SENTENCES_PER_BATCH", _DEFAULT_MAX_SENTENCES_PER_BATCH
    )
    order: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for seg in segments or []:
        pid = seg.get("paragraph_id") or ""
        if pid not in groups:
            groups[pid] = []
            order.append(pid)
        groups[pid].append(seg)

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_paras = 0
    for pid in order:
        para_rows = groups[pid]
        if current and (
            current_paras >= per or (current_paras >= 1 and len(current) + len(para_rows) > max_sent)
        ):
            batches.append(current)
            current, current_paras = [], 0
        current.extend(para_rows)
        current_paras += 1
    if current:
        batches.append(current)
    return batches


def _build_batch_prompt(batch: list[dict[str, Any]]) -> str:
    lines: list[str] = ["Sentences (grouped by paragraph):"]
    current_pid: Optional[str] = None
    for seg in batch:
        pid = seg.get("paragraph_id") or ""
        if pid != current_pid:
            lines.append(f"[{pid}]")
            current_pid = pid
        sid = seg.get("sentence_id") or ""
        text = str(seg.get("sentence") or "").replace("\n", " ")
        lines.append(f"{sid}: {text}")
    lines.append("\nReturn the claim-graph JSON for these sentences only.")
    return "\n".join(lines)


# ── Defensive JSON parse ─────────────────────────────────────────────────────
def parse_json(content: str) -> Optional[dict[str, Any]]:
    """Parse strict-JSON output; fall back to the first ``{...}`` brace span."""
    if not content or not content.strip():
        return None
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _max_tokens() -> int:
    return _int_env("DRAFTPROOF_CLAIM_GRAPH_MAX_TOKENS", _DEFAULT_MAX_TOKENS)


def _call_batch(gateway: Any, batch: list[dict[str, Any]], response_format) -> Optional[dict[str, Any]]:
    prompt = _build_batch_prompt(batch)
    resp = gateway.chat(
        prompt,
        system=_SYSTEM_PROMPT,
        response_format=response_format,
        max_tokens=_max_tokens(),
    )
    return parse_json(str(getattr(resp, "content", "") or ""))


# ── Rejection classifier (advisory stats only — validators own truth) ────────
def classify_rejections(
    proposed_claims: list[dict[str, Any]], sentence_index: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Count WOULD-BE hard rejections by rule, mirroring ``validators`` decisions
    for the ``extraction_stats.rejected_by_rule`` counters. Advisory only; the
    validators remain the single authority on what enters the graph."""
    counts: dict[str, int] = {
        "unknown_node_type": 0,
        "forbidden_verification_status": 0,
        "unknown_sentence_id": 0,
        "span_not_found": 0,
    }
    from .validators import _match_span  # pure helper, no I/O

    for proposal in proposed_claims or []:
        node_type = proposal.get("node_type")
        if node_type not in schema.NODE_TYPES:
            counts["unknown_node_type"] += 1
            continue
        if proposal.get("verification_status") in schema.PHASE1_FORBIDDEN_STATES:
            counts["forbidden_verification_status"] += 1
            continue
        if node_type != "CLAIM":
            continue
        seg = sentence_index.get(proposal.get("sentence_id"))
        if seg is None:
            counts["unknown_sentence_id"] += 1
            continue
        if _match_span(str(proposal.get("quote") or ""), seg["text"]) is None:
            counts["span_not_found"] += 1
    return counts


# ── Orchestration ────────────────────────────────────────────────────────────
def _cache_key(text: str, model: str, prompt_version: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt_version}\n{text}".encode("utf-8")).hexdigest()


def run_extraction(
    text: str,
    segments: list[dict[str, Any]],
    gateway: Any,
    *,
    prompt_version: str = PROMPT_VERSION,
    max_retries: Optional[int] = None,
    use_cache: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract per-batch, retry parse-failed batches, return (proposals, stats).

    ``proposals`` = ``{"claims": [...], "edges": [...]}`` (raw LLM proposals, NOT
    yet validated). ``stats`` carries the §3 counters.
    """
    from .validators import build_sentence_index  # pure

    model = _gateway_model(gateway)
    key = _cache_key(text, model, prompt_version)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    cap_retries = max_retries if max_retries is not None else _int_env(
        "DRAFTPROOF_CLAIM_GRAPH_MAX_RETRIES", _DEFAULT_MAX_RETRIES
    )
    response_format = _response_format(model)
    batches = paragraph_batches(segments)

    claims: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    failed_indexes: list[int] = []
    calls = 0

    for idx, batch in enumerate(batches):
        try:
            data = _call_batch(gateway, batch, response_format)
        except Exception:
            data = None
        calls += 1
        if data is None:
            failed_indexes.append(idx)
            continue
        claims.extend([c for c in (data.get("claims") or []) if isinstance(c, dict)])
        edges.extend([e for e in (data.get("edges") or []) if isinstance(e, dict)])

    # Stage-3: targeted retries — ONLY batches whose JSON failed to parse.
    retries = 0
    pending = list(failed_indexes)
    while pending and retries < cap_retries:
        idx = pending.pop(0)
        retries += 1
        calls += 1
        try:
            data = _call_batch(gateway, batches[idx], response_format)
        except Exception:
            data = None
        if data is None:
            continue
        claims.extend([c for c in (data.get("claims") or []) if isinstance(c, dict)])
        edges.extend([e for e in (data.get("edges") or []) if isinstance(e, dict)])

    proposals = {"claims": claims, "edges": edges}
    sentence_index = build_sentence_index(segments)
    stats = {
        "prompt_version": prompt_version,
        "model": model,
        "batches": len(batches),
        "extract_calls": calls,
        "retries": retries,
        "parse_failures": len(failed_indexes),
        "proposed": len(claims),
        "proposed_edges": len(edges),
        "rejected_by_rule": classify_rejections(claims, sentence_index),
    }
    result = (proposals, stats)
    if use_cache:
        _CACHE[key] = result
    return result


def clear_cache() -> None:
    """Test/maintenance helper — drop the in-process extraction cache."""
    _CACHE.clear()


__all__ = [
    "PROMPT_VERSION",
    "resolve_gateway",
    "paragraph_batches",
    "parse_json",
    "classify_rejections",
    "run_extraction",
    "clear_cache",
]
