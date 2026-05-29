"""LLM-backed paragraph explanations for scan reports.

The scanner decides what is flagged. This module only translates grouped
paragraph findings into plain-language explanations and recommendations.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any


_DEFAULT_MODEL = "openai/gpt-oss-120b"


def planner_model_from_env() -> str:
    return (
        os.environ.get("DRAFTPROOF_V6_PLANNER_MODEL")
        or os.environ.get("DRAFTPROOF_PLANNER_MODEL")
        or os.environ.get("DRAFTPROOF_REWRITE_V5_PLANNER_MODEL")
        or os.environ.get("LLM_MODEL")
        or _DEFAULT_MODEL
    )


def build_paragraph_explanation_input(report_json: dict[str, Any]) -> dict[str, Any]:
    document = ((report_json.get("scan_intelligence") or {}).get("document") or {})
    paragraphs = document.get("paragraphs") if isinstance(document, dict) else []
    segments = document.get("segments") if isinstance(document, dict) else []
    paragraph_by_id = {
        str(row.get("paragraph_id") or ""): row
        for row in paragraphs
        if isinstance(row, dict) and row.get("paragraph_id")
    }
    segment_by_sentence = {
        str(row.get("sentence_id") or ""): row
        for row in segments
        if isinstance(row, dict) and row.get("sentence_id")
    }

    grouped: dict[str, dict[str, Any]] = {}
    for severity in ("critical", "high", "medium", "low"):
        for finding in (report_json.get("findings") or {}).get(severity, []) or []:
            if not isinstance(finding, dict):
                continue
            sentence_id = str(finding.get("sentence_id") or "")
            segment = segment_by_sentence.get(sentence_id, {})
            paragraph_id = str(segment.get("paragraph_id") or "")
            if not paragraph_id:
                paragraph_id = f"document_{finding.get('finding_id') or len(grouped) + 1}"
            paragraph = paragraph_by_id.get(paragraph_id, {})
            # The exact wordings the detector found most statistically predictable (the strongest
            # AI-likelihood signal) live on the segment, not the finding. Surfacing them lets the
            # explainer name the specific phrases to change instead of giving generic advice.
            sentence_phrases = _segment_predictable_phrases(segment)
            entry = grouped.setdefault(paragraph_id, {
                "paragraph_id": paragraph_id,
                "sentence_ids": list(paragraph.get("sentence_ids") or ([] if not sentence_id else [sentence_id])),
                "text": paragraph.get("text") or segment.get("text") or "",
                "findings": [],
                "predictable_phrases": [],
            })
            if sentence_id and sentence_id not in entry["sentence_ids"]:
                entry["sentence_ids"].append(sentence_id)
            for phrase in sentence_phrases:
                if phrase not in entry["predictable_phrases"]:
                    entry["predictable_phrases"].append(phrase)
            entry["findings"].append({
                "finding_id": finding.get("finding_id"),
                "severity": severity,
                "title": finding.get("title"),
                "scanner": finding.get("scanner"),
                "category": finding.get("category"),
                "signal_category": finding.get("signal_category"),
                "score": finding.get("score"),
                "actionability": finding.get("actionability"),
                "detail": finding.get("detail"),
                "evidence": finding.get("evidence"),
                "recommendation": finding.get("recommendation"),
                "predictable_phrases": sentence_phrases,
            })

    rows = [
        row
        for row in grouped.values()
        if row.get("findings")
    ]
    rows.sort(key=lambda row: str(row.get("paragraph_id") or ""))
    return {"paragraphs": rows}


def _segment_predictable_phrases(segment: dict[str, Any]) -> list[str]:
    """Exact predictable token spans the detector flagged on a sentence (deduped, in order)."""
    predictability = segment.get("predictability") if isinstance(segment.get("predictability"), dict) else {}
    phrases: list[str] = []
    for span in predictability.get("predictable_token_spans") or []:
        text = " ".join(str(span or "").split())
        if text and text not in phrases:
            phrases.append(text)
    return phrases


def report_explanation_hash(explainer_input: dict[str, Any]) -> str:
    payload = json.dumps(explainer_input, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_paragraph_explanations(
    report_json: dict[str, Any],
    *,
    gateway: Any | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    explainer_input = build_paragraph_explanation_input(report_json)
    paragraph_rows = explainer_input.get("paragraphs") or []
    if not paragraph_rows:
        return {
            "schema_version": "paragraph_explanations.v2",
            "source_report_hash": report_explanation_hash(explainer_input),
            "generated_at": int(time.time()),
            "model": model or planner_model_from_env(),
            "paragraphs": [],
        }

    model = model or planner_model_from_env()
    if gateway is None:
        configured_api_key = (
            (api_key or "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
            or os.environ.get("LLM_API_KEY", "").strip()
            or os.environ.get("CEREBRAS_API_KEY", "").strip()
        )
        configured_base_url = base_url or os.environ.get("LLM_BASE_URL") or None
        try:
            from rewrite_v6.llm_config import (
                resolve_v6_api_key,
                resolve_v6_base_url,
                resolve_v6_model,
            )
        except ModuleNotFoundError:
            from poc.rewrite_v6.llm_config import (
                resolve_v6_api_key,
                resolve_v6_base_url,
                resolve_v6_model,
            )

        try:
            resolved_api_key = resolve_v6_api_key(configured_api_key)
            resolved_base_url = resolve_v6_base_url(configured_base_url)
            model = resolve_v6_model(model) or model
        except Exception:
            resolved_api_key = configured_api_key
            resolved_base_url = configured_base_url
        if not resolved_api_key:
            return None
        from llm.gateway import LLMConfig, LLMGateway

        gateway = LLMGateway(LLMConfig(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            model=model,
            temperature=0.15,
            max_tokens=_int_env("DRAFTPROOF_PARAGRAPH_EXPLAINER_MAX_TOKENS", 5000),
            timeout=_int_env("DRAFTPROOF_PARAGRAPH_EXPLAINER_TIMEOUT_SECONDS", 90),
            app_label="Dignose",
        ))

    prompt_input = _bounded_explainer_input(explainer_input)
    response = gateway.chat(
        _prompt(prompt_input),
        system=(
            "You explain writing scan findings to students in plain language. "
            "Return valid JSON only. Do not invent facts, citations, sources, or misconduct claims."
        ),
        temperature=0.15,
        max_tokens=_int_env("DRAFTPROOF_PARAGRAPH_EXPLAINER_MAX_TOKENS", 5000),
        response_format={"type": "json_object"},
        app_label="Dignose",
    )
    parsed = _parse_json(getattr(response, "raw_content", "") or response.content)
    paragraphs = _normalize_explanations(parsed, paragraph_rows)
    return {
        "schema_version": "paragraph_explanations.v2",
        "source_report_hash": report_explanation_hash(explainer_input),
        "generated_at": int(time.time()),
        "model": getattr(gateway, "model", model),
        "paragraphs": paragraphs,
    }


def explanations_by_paragraph(explanations: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(explanations, dict):
        return {}
    return {
        str(row.get("paragraph_id") or ""): row
        for row in explanations.get("paragraphs") or []
        if isinstance(row, dict) and row.get("paragraph_id")
    }


def _bounded_explainer_input(explainer_input: dict[str, Any]) -> dict[str, Any]:
    max_paragraphs = _int_env("DRAFTPROOF_PARAGRAPH_EXPLAINER_MAX_PARAGRAPHS", 80)
    max_paragraph_chars = _int_env("DRAFTPROOF_PARAGRAPH_EXPLAINER_PARAGRAPH_CHARS", 900)
    max_finding_chars = _int_env("DRAFTPROOF_PARAGRAPH_EXPLAINER_FINDING_CHARS", 420)
    rows = []
    for row in (explainer_input.get("paragraphs") or [])[:max_paragraphs]:
        bounded_findings = []
        for finding in row.get("findings") or []:
            bounded_findings.append({
                key: _clip(value, max_finding_chars)
                for key, value in finding.items()
            })
        rows.append({
            "paragraph_id": row.get("paragraph_id"),
            "sentence_ids": row.get("sentence_ids") or [],
            "text": _clip(row.get("text"), max_paragraph_chars),
            "predictable_phrases": [
                _clip(phrase, max_finding_chars)
                for phrase in (row.get("predictable_phrases") or [])[:12]
            ],
            "findings": bounded_findings,
        })
    return {"paragraphs": rows}


def _prompt(payload: dict[str, Any]) -> str:
    return (
        "Create student-facing paragraph guidance for the scan findings below.\n"
        "Rules:\n"
        "- Write for a student, tutor, or non-technical reviewer. Do not expose detector labels as advice.\n"
        "- Every field must refer to this paragraph's actual subject matter, wording, or argument path.\n"
        "- Explain what a reader may notice, not what the detector calculated.\n"
        "- Mention uncertainty where needed; do not say the student used AI or committed misconduct.\n"
        "- Combine repeated sentence findings into one paragraph-level diagnosis.\n"
        "- Pick one main issue. Do not list every signal mechanically or repeat the scanner recommendations.\n"
        "- If several findings say the same thing, explain the shared writing problem once.\n"
        "- Recommendation must be a concrete edit for this paragraph, not generic advice.\n"
        "- Tell the student what to change in the paragraph flow, such as adding a specific link, merging related points, or replacing a broad claim with a concrete explanation already supported by the text.\n"
        "- Each paragraph includes predictable_phrases: the exact wordings the scanner found most statistically expected (the strongest signal). When present, ground the guidance in these specific phrases.\n"
        "- In why_flagged, quote or closely paraphrase at least one predictable_phrase and explain in plain terms why that exact wording reads as common or expected.\n"
        "- Recommendation and rewrite_hint must show how to replace one of those specific predictable phrases with a concrete, particular detail already supported by the paragraph — name the phrase being changed.\n"
        "- Do not just say the wording is 'generic' or a 'standard list'; point to the actual phrase and what would make it specific.\n"
        "- Rewrite hint may show the type of sentence to add, but must not invent new facts.\n"
        "- Do not add facts, examples, citations, or course details not present in the paragraph.\n"
        "- Avoid phrases such as predictable phrasing, semantic drift, low originality, signal, detector, score, or flagged unless explaining uncertainty.\n"
        "- Prefer plain wording like 'The paragraph makes the point, but the reason is still broad' over technical labels.\n"
        "- Keep each field short enough for a report side panel.\n\n"
        "Return JSON with this shape:\n"
        "{\n"
        '  "paragraphs": [\n'
        "    {\n"
        '      "paragraph_id": "p001",\n'
        '      "source_finding_ids": ["f001"],\n'
        '      "reader_summary": "what a normal reader may notice in this paragraph",\n'
        '      "main_issue": "the single most important fix",\n'
        '      "why_flagged": ["plain reason tied to the paragraph", "plain reason tied to the paragraph"],\n'
        '      "recommendation": "specific paragraph-level edit instruction",\n'
        '      "rewrite_hint": "optional one-sentence example of the kind of edit to make",\n'
        '      "confidence": "low|medium|high"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, default=str)}"
    )


def _normalize_explanations(parsed: Any, source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {
        str(row.get("paragraph_id") or ""): row
        for row in parsed.get("paragraphs", []) if isinstance(row, dict)
    } if isinstance(parsed, dict) else {}
    normalized = []
    for source in source_rows:
        paragraph_id = str(source.get("paragraph_id") or "")
        row = by_id.get(paragraph_id) or {}
        finding_ids = [
            str(finding.get("finding_id"))
            for finding in source.get("findings") or []
            if finding.get("finding_id")
        ]
        normalized.append({
            "paragraph_id": paragraph_id,
            "sentence_ids": list(source.get("sentence_ids") or []),
            "source_finding_ids": _string_list(row.get("source_finding_ids")) or finding_ids,
            "schema_version": "paragraph_explanation.v2",
            "summary": _clean_text(row.get("reader_summary") or row.get("summary"), 520),
            "reader_summary": _clean_text(row.get("reader_summary") or row.get("summary"), 520),
            "main_issue": _clean_text(row.get("main_issue"), 260),
            "why_flagged": _string_list(row.get("why_flagged"), limit=4, item_limit=180),
            "recommendation": _clean_text(row.get("recommendation"), 360),
            "rewrite_hint": _clean_text(row.get("rewrite_hint"), 300),
            "confidence": _confidence(row.get("confidence")),
        })
    return normalized


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


def _string_list(value: Any, *, limit: int = 8, item_limit: int = 120) -> list[str]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        text = _clean_text(item, item_limit)
        if text and text not in rows:
            rows.append(text)
        if len(rows) >= limit:
            break
    return rows


def _clean_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].rstrip() if len(text) > limit else text


def _clip(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return _clean_text(value, limit)
    if isinstance(value, dict):
        return {key: _clip(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_clip(item, limit) for item in value[:10]]
    return value


def _confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"low", "medium", "high"} else "medium"


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default
