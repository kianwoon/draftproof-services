"""LLM-judged dimensions for the Critical Thinking Control module (Phase 2).

ADDITIVE and behind a kill-switch (DEFAULT OFF). Adds the two dimensions that
have no deterministic signal -- alternative_comparison and reflection -- plus
agnostic per-sentence coaching highlights. These are QUALITATIVE review flags:
they are NOT folded into the deterministic control score (see
detect.critical_thinking), so the headline number stays reproducible.

NO-HARDCODE: the judge reasons conceptually about meaning and argument structure.
It is explicitly forbidden from matching against any fixed list of trigger
phrases. The highlight ``label`` is an OUTPUT enum (a category the model picks
after reasoning), never a string compared against the document.

Fail-open: any disable / missing key / parse error / exception returns None, so
the scan still ships the deterministic core. Mirrors the construction, prompt,
and parse pattern of poc/report/paragraph_explainer.py.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

MODEL_VERSION = "critical_thinking_llm_v1"
_DEFAULT_MODEL = "openai/gpt-oss-120b"

# allow-hardcode: OUTPUT category enum the LLM selects AFTER reasoning, NOT a
# detect/allow word-list. No document text is ever compared against these keys;
# they only constrain which label the model may emit. KEEP IN SYNC with the
# frontend highlight labels (report.criticalThinking.highlightLabels.*).
ALLOWED_LABELS = frozenset({
    "single_path_answer",   # one neat answer, no alternatives weighed
    "no_judgement",         # states an idea but shows no writer decision
    "no_reflection",        # no sign of noticing / changing / questioning / learning
    "broad_claim",          # claim too general to be the writer's own
    "missing_evidence",     # claim needs a source, example, or observation
    "over_polished_closure",  # conclusion sounds complete but generic
})
_ALLOWED_SEVERITIES = frozenset({"low", "medium", "high"})


def critical_thinking_llm_enabled() -> bool:
    """Kill switch, DEFAULT OFF. Set DRAFTPROOF_CRITICAL_THINKING_LLM=1 to enable.
    Off until variance is validated (the 2 dims are non-deterministic)."""
    return os.environ.get("DRAFTPROOF_CRITICAL_THINKING_LLM", "").strip().lower() in {"1", "true", "yes", "on"}


def _model_from_env() -> str:
    return (
        os.environ.get("DRAFTPROOF_CRITICAL_THINKING_MODEL")
        or os.environ.get("DRAFTPROOF_V6_PLANNER_MODEL")
        or os.environ.get("DRAFTPROOF_PLANNER_MODEL")
        or os.environ.get("LLM_MODEL")
        or _DEFAULT_MODEL
    )


def _document_paragraphs(report_json: dict[str, Any]) -> list[dict[str, Any]]:
    """All submitted paragraphs (flagged or not) -- comparison/reflection are
    paragraph-level qualities, so we assess every paragraph, not just findings."""
    document = ((report_json.get("scan_intelligence") or {}).get("document") or {})
    paragraphs = document.get("paragraphs") if isinstance(document, dict) else []
    rows: list[dict[str, Any]] = []
    for row in paragraphs or []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("paragraph_id") or "")
        text = " ".join(str(row.get("text") or "").split())
        if pid and text:
            rows.append({"paragraph_id": pid, "text": text})
    return rows


def assess_critical_thinking(
    report_json: dict[str, Any],
    *,
    gateway: Any | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    """Return {llm_dimensions, highlights, model, schema_version} or None.

    None on disable / no paragraphs / no key / any failure (fail-open).
    """
    if not critical_thinking_llm_enabled():
        return None
    try:
        rows = _document_paragraphs(report_json)
        if not rows:
            return None
        rows = _bounded(rows)
        model = model or _model_from_env()

        if gateway is None:
            gateway = _build_gateway(model=model, api_key=api_key, base_url=base_url)
            if gateway is None:
                return None

        response = gateway.chat(
            _prompt(rows),
            system=_SYSTEM,
            temperature=_temperature(),
            max_tokens=_int_env("DRAFTPROOF_CRITICAL_THINKING_MAX_TOKENS", 4000),
            response_format={"type": "json_object"},
            app_label="CriticalThinking",
        )
        parsed = _parse_json(getattr(response, "raw_content", "") or getattr(response, "content", "") or "")
        valid_ids = {r["paragraph_id"] for r in rows}
        dims = _normalize_dimensions(parsed, valid_ids)
        highlights = _normalize_highlights(parsed, valid_ids)
        if dims is None and not highlights:
            return None
        return {
            "schema_version": MODEL_VERSION,
            "model": getattr(gateway, "model", model),
            "generated_at": int(time.time()),
            "llm_dimensions": dims,
            "highlights": highlights,
        }
    except Exception:
        return None


def _build_gateway(*, model: str, api_key: str | None, base_url: str | None) -> Any | None:
    configured_api_key = (
        (api_key or "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("CEREBRAS_API_KEY", "").strip()
    )
    configured_base_url = base_url or os.environ.get("LLM_BASE_URL") or None
    try:
        try:
            from rewrite_v6.llm_config import resolve_v6_api_key, resolve_v6_base_url, resolve_v6_model
        except ModuleNotFoundError:
            from poc.rewrite_v6.llm_config import resolve_v6_api_key, resolve_v6_base_url, resolve_v6_model
        try:
            resolved_api_key = resolve_v6_api_key(configured_api_key)
            resolved_base_url = resolve_v6_base_url(configured_base_url)
            model = resolve_v6_model(model) or model
        except Exception:
            resolved_api_key = configured_api_key
            resolved_base_url = configured_base_url
        if not resolved_api_key:
            return None
        try:
            from llm.gateway import LLMConfig, LLMGateway
        except ModuleNotFoundError:
            from poc.llm.gateway import LLMConfig, LLMGateway
        return LLMGateway(LLMConfig(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            model=model,
            temperature=_temperature(),
            max_tokens=_int_env("DRAFTPROOF_CRITICAL_THINKING_MAX_TOKENS", 4000),
            timeout=_int_env("DRAFTPROOF_CRITICAL_THINKING_TIMEOUT_SECONDS", 90),
            app_label="CriticalThinking",
        ))
    except Exception:
        return None


_SYSTEM = (
    "You assess whether a student stayed in control of the thinking in their own writing. "
    "Reason about meaning and argument structure -- never match against any fixed list of phrases "
    "or trigger words. Return valid JSON only. Do not invent facts, examples, citations, sources, "
    "or any misconduct claim. Quote sentences verbatim from the provided text."
)


def _prompt(rows: list[dict[str, Any]]) -> str:
    labels = ", ".join(sorted(ALLOWED_LABELS))
    return (
        "For each paragraph below, judge two qualities of the WRITER's thinking. "
        "Reason about the meaning; do not pattern-match on wording.\n\n"
        "1. alternative_comparison (0-100): does the paragraph genuinely weigh more than one view, "
        "option, source, method, or explanation? 0 = one smooth answer with no alternative; "
        "100 = real comparison or a considered trade-off.\n"
        "2. reflection (0-100): does the writer show their own noticing, choosing, questioning, "
        "changing, or learning? 0 = none of the writer's judgement is visible; 100 = clear ownership.\n\n"
        "Then list a few sentence-level highlights for the weakest spots. For each highlight: quote the "
        "sentence verbatim, pick exactly one label from this set: "
        f"[{labels}], set severity to low/medium/high, say why_it_matters in plain language, and give one "
        "fix_instruction describing the KIND of content to add (a limitation, an alternative, a decision "
        "point, a specific example) -- never invent the content itself.\n\n"
        "Return JSON with this shape:\n"
        "{\n"
        '  "paragraphs": [\n'
        '    {"paragraph_id": "p001", "alternative_comparison": 35, "reflection": 20}\n'
        "  ],\n"
        '  "highlights": [\n'
        '    {"paragraph_id": "p001", "sentence": "verbatim sentence", "label": "single_path_answer", '
        '"severity": "high", "why_it_matters": "plain reason", "fix_instruction": "what kind of content to add"}\n'
        "  ]\n"
        "}\n\n"
        f"Input JSON:\n{json.dumps({'paragraphs': rows}, ensure_ascii=False, default=str)}"
    )


def _normalize_dimensions(parsed: Any, valid_ids: set[str]) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    alt_scores: list[float] = []
    ref_scores: list[float] = []
    per_paragraph: list[dict[str, Any]] = []
    for row in parsed.get("paragraphs") or []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("paragraph_id") or "")
        if pid not in valid_ids:
            continue
        alt = _score_0_100(row.get("alternative_comparison"))
        ref = _score_0_100(row.get("reflection"))
        if alt is None and ref is None:
            continue
        per_paragraph.append({"paragraph_id": pid, "alternative_comparison": alt, "reflection": ref})
        if alt is not None:
            alt_scores.append(alt)
        if ref is not None:
            ref_scores.append(ref)
    if not per_paragraph:
        return None
    return {
        "alternative_comparison": {
            "score": round(sum(alt_scores) / len(alt_scores), 1) if alt_scores else None,
            "coverage": len(alt_scores),
        },
        "reflection": {
            "score": round(sum(ref_scores) / len(ref_scores), 1) if ref_scores else None,
            "coverage": len(ref_scores),
        },
        "paragraphs": per_paragraph,
        "note": "Qualitative LLM-judged dimensions — review flags, NOT part of the control score.",
    }


def _normalize_highlights(parsed: Any, valid_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []
    out: list[dict[str, Any]] = []
    for row in parsed.get("highlights") or []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("paragraph_id") or "")
        label = str(row.get("label") or "").strip()
        sentence = _clip(row.get("sentence"), 400)
        if pid not in valid_ids or label not in ALLOWED_LABELS or not sentence:
            continue
        severity = str(row.get("severity") or "").strip().lower()
        out.append({
            "paragraph_id": pid,
            "sentence": sentence,
            "label": label,
            "severity": severity if severity in _ALLOWED_SEVERITIES else "medium",
            "why_it_matters": _clip(row.get("why_it_matters"), 300),
            "fix_instruction": _clip(row.get("fix_instruction"), 300),
            "signal_category": "critical_thinking",
        })
        if len(out) >= _int_env("DRAFTPROOF_CRITICAL_THINKING_MAX_HIGHLIGHTS", 20):
            break
    return out


def _bounded(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_paragraphs = _int_env("DRAFTPROOF_CRITICAL_THINKING_MAX_PARAGRAPHS", 60)
    max_chars = _int_env("DRAFTPROOF_CRITICAL_THINKING_PARAGRAPH_CHARS", 900)
    return [
        {"paragraph_id": r["paragraph_id"], "text": _clip(r["text"], max_chars)}
        for r in rows[:max_paragraphs]
    ]


def _temperature() -> float:
    try:
        from rewrite_v6.llm_config import deterministic_mode
    except ModuleNotFoundError:
        try:
            from poc.rewrite_v6.llm_config import deterministic_mode
        except ModuleNotFoundError:
            deterministic_mode = lambda: False  # noqa: E731
    if deterministic_mode():
        return 0.0
    raw = os.environ.get("DRAFTPROOF_CRITICAL_THINKING_TEMPERATURE")
    try:
        return max(0.0, min(1.0, float(raw))) if raw else 0.1
    except (TypeError, ValueError):
        return 0.1


def _score_0_100(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, min(100.0, round(float(value), 1)))


def _parse_json(raw: str) -> Any:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].rstrip() if len(text) > limit else text


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default
