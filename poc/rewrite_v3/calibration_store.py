"""External calibration records for V3 portfolio ranking."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .candidate_features import CandidateFeatures, features_from_trace


def calibration_path() -> str:
    return os.environ.get("DRAFTPROOF_REWRITE_V3_CALIBRATION_STORE", "") or os.environ.get("DRAFTPROOF_REWRITE_V3_STYLE_LIBRARY", "")


@dataclass(frozen=True)
class CalibrationRecord:
    family: str
    external_ai_percent: float
    features: CandidateFeatures | None
    text: str

    @property
    def passed_external(self) -> bool:
        return self.external_ai_percent <= 35.0


def _rows_from_path(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        rows = parsed.get("records") or parsed.get("examples") or []
        return [row for row in rows if isinstance(row, dict)]
    return []


def _record_from_row(row: dict[str, Any]) -> CalibrationRecord | None:
    label = row.get("external_label") if isinstance(row.get("external_label"), dict) else {}
    raw_ai = label.get("ai_percent") if label else row.get("ai_percent")
    try:
        external_ai = float(raw_ai)
    except (TypeError, ValueError):
        return None
    family = str(row.get("family") or row.get("strategy_family") or "").strip()
    if not family:
        return None
    features_payload = row.get("features")
    trace_payload = row.get("trace")
    features = None
    if isinstance(features_payload, dict):
        try:
            features = CandidateFeatures(**features_payload)
        except TypeError:
            features = None
    elif isinstance(trace_payload, dict):
        features = features_from_trace(trace_payload)
    return CalibrationRecord(
        family=family,
        external_ai_percent=external_ai,
        features=features,
        text=str(row.get("text") or row.get("candidate_text") or ""),
    )


@lru_cache(maxsize=8)
def load_calibration_records(path: str | None = None) -> tuple[CalibrationRecord, ...]:
    raw = path or calibration_path()
    if not raw:
        return ()
    file_path = Path(raw)
    if not file_path.exists() or not file_path.is_file():
        return ()
    records = [_record_from_row(row) for row in _rows_from_path(file_path)]
    return tuple(record for record in records if record is not None)


def records_for_family(family: str, *, path: str | None = None) -> tuple[CalibrationRecord, ...]:
    normalized = str(family or "")
    return tuple(record for record in load_calibration_records(path) if record.family == normalized)
