"""Optional external-calibrated style examples for rewrite V3."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def style_library_path() -> str:
    return os.environ.get("DRAFTPROOF_REWRITE_V3_STYLE_LIBRARY", "")


def _load_rows_from_text(path: Path, text: str) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        rows = parsed.get("records") or parsed.get("examples") or []
        return [row for row in rows if isinstance(row, dict)]
    return []


@lru_cache(maxsize=8)
def load_style_library(path: str | None = None) -> tuple[dict[str, Any], ...]:
    raw = path or style_library_path()
    if not raw:
        return ()
    file_path = Path(raw)
    if not file_path.exists() or not file_path.is_file():
        return ()
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return ()
    return tuple(_load_rows_from_text(file_path, text))


def examples_for_family(family: str, *, limit: int = 4) -> dict[str, list[dict[str, Any]]]:
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    for row in load_style_library():
        if str(row.get("family") or row.get("strategy_family") or "") != family:
            continue
        label = row.get("external_label") or {}
        ai_percent = label.get("ai_percent") if isinstance(label, dict) else row.get("ai_percent")
        try:
            ai_value = float(ai_percent)
        except (TypeError, ValueError):
            ai_value = None
        text = str(row.get("text") or row.get("candidate_text") or "").strip()
        if not text:
            continue
        item = {
            "external_ai_percent": ai_value,
            "text": text[:4200],
            "notes": row.get("notes") or row.get("failure") or row.get("why"),
        }
        if ai_value is not None and ai_value <= 35:
            positive.append(item)
        elif ai_value is not None:
            negative.append(item)
        if len(positive) >= limit and len(negative) >= limit:
            break
    return {"positive": positive[:limit], "negative": negative[:limit]}
