"""Targeted repair prompt for failed authorship windows."""

from __future__ import annotations

import json
from typing import Any


FAMILY = "authorship_window_repair"


def _window_context(candidate_text: str, window: dict[str, Any], *, radius: int = 520) -> dict[str, Any]:
    text = str(candidate_text or "")
    start = max(0, int(window.get("start_index") or 0))
    end = max(start, int(window.get("end_index") or start))
    before_start = max(0, start - radius)
    after_end = min(len(text), end + radius)
    return {
        "window_id": window.get("window_id"),
        "paragraph_id": window.get("paragraph_id"),
        "label": window.get("label"),
        "confidence": window.get("confidence"),
        "word_count": window.get("word_count"),
        "ai_assistance_score": window.get("ai_assistance_score"),
        "score_components": window.get("score_components") or {},
        "top_signals": window.get("top_signals") or [],
        "start_index": start,
        "end_index": end,
        "before_context": text[before_start:start],
        "target_text": text[start:end],
        "after_context": text[end:after_end],
    }


def build_authorship_window_repair_prompt(
    *,
    candidate_text: str,
    target_windows: list[dict[str, Any]],
    strategy_family: str,
    contract: Any,
) -> str:
    anchors = []
    for anchor in getattr(contract, "anchors", []) or []:
        text = str(getattr(anchor, "text", "") or "")
        if not text:
            continue
        anchors.append({
            "text": text,
            "kind": str(getattr(anchor, "kind", "") or ""),
            "severity": str(getattr(getattr(anchor, "severity", None), "value", "") or ""),
        })
    payload = {
        "strategy_family": strategy_family,
        "repair_scope": "selected_windows_only",
        "target_windows": [
            _window_context(candidate_text, window)
            for window in target_windows
        ],
        "protected_anchors": anchors[:80],
        "requirements": [
            "Return JSON only, with a replacements array.",
            "Rewrite only each target_text. Do not rewrite the whole document.",
            "Keep the same local meaning, citation obligations, technical terms, and paragraph role.",
            "Use before_context and after_context only to preserve continuity.",
            "Reduce model-like texture by adding concrete reasoning, local constraint, and natural sentence route already supported by the window.",
            "Do not add new sources, names, statistics, institutions, headings, bullets, paragraph numbers, markdown, or commentary.",
            "Keep each replacement close to the target window length unless the target is clearly over-compressed.",
        ],
        "response_schema": {
            "replacements": [
                {
                    "window_id": "w001",
                    "replacement_text": "replacement prose only",
                }
            ]
        },
    }
    return (
        "Repair only the authorship-risk windows in this rewrite candidate.\n"
        "The candidate has already passed enough document-level structure to avoid a full rewrite.\n"
        "Your job is local texture repair: preserve meaning, improve human authorship texture, and keep boundaries stable.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def extract_authorship_window_replacements(raw: str) -> list[dict[str, str]]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    rows = payload.get("replacements") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    replacements: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        window_id = str(row.get("window_id") or "").strip()
        replacement_text = str(row.get("replacement_text") or "").strip()
        if not window_id or not replacement_text:
            continue
        replacements.append({
            "window_id": window_id,
            "replacement_text": replacement_text,
        })
    return replacements


def apply_authorship_window_replacements(
    *,
    candidate_text: str,
    target_windows: list[dict[str, Any]],
    replacements: list[dict[str, str]],
) -> str:
    by_id = {str(row.get("window_id") or ""): str(row.get("replacement_text") or "") for row in replacements}
    text = str(candidate_text or "")
    replace_ranges: list[tuple[int, int, str]] = []
    for window in target_windows:
        window_id = str(window.get("window_id") or "")
        replacement = by_id.get(window_id, "").strip()
        if not replacement:
            continue
        start = max(0, int(window.get("start_index") or 0))
        end = max(start, int(window.get("end_index") or start))
        if start > len(text) or end > len(text):
            continue
        replace_ranges.append((start, end, replacement))
    for start, end, replacement in sorted(replace_ranges, key=lambda item: item[0], reverse=True):
        text = text[:start] + replacement + text[end:]
    return text.strip()
