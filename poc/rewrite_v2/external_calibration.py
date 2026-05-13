"""External detector calibration helpers for rewrite V2.

The production scanner cannot call every external detector, so V2 keeps a
proxy. This module makes the proxy's calibration contract explicit: external
labels are stored separately, normalized to a single AI-percent scale, and used
to tune thresholds from evidence rather than individual runs.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


EXTERNAL_LABEL_SCHEMA_VERSION = "rewrite_v2_external_detector_calibration_v1"


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def external_label_pass_max() -> float:
    """Maximum external AI percentage treated as a pass in calibration data."""
    return max(0.0, min(100.0, _float_env("DRAFTPROOF_REWRITE_V2_EXTERNAL_LABEL_PASS_MAX", 35.0)))


def normalize_external_ai_percent(label: Any) -> float | None:
    """Normalize common external-detector label shapes to 0-100 AI percent."""
    if isinstance(label, (int, float)):
        value = float(label)
        return max(0.0, min(100.0, value * 100.0 if 0.0 <= value <= 1.0 else value))
    if isinstance(label, str):
        stripped = label.strip().lower().replace("%", "")
        if not stripped:
            return None
        label_aliases = {
            "human": 0.0,
            "likely human": 20.0,
            "mixed": 50.0,
            "likely ai": 75.0,
            "ai": 100.0,
            "ai-generated": 100.0,
        }
        if stripped in label_aliases:
            return label_aliases[stripped]
        try:
            value = float(stripped)
        except ValueError:
            return None
        return max(0.0, min(100.0, value * 100.0 if 0.0 <= value <= 1.0 else value))
    if isinstance(label, dict):
        for key in (
            "ai_percent",
            "ai_generated_percent",
            "ai_probability",
            "ai_score",
            "score",
            "value",
        ):
            if key in label:
                normalized = normalize_external_ai_percent(label.get(key))
                if normalized is not None:
                    return normalized
    return None


def classify_external_label(label: Any, *, pass_max: float | None = None) -> dict[str, Any]:
    ai_percent = normalize_external_ai_percent(label)
    threshold = external_label_pass_max() if pass_max is None else max(0.0, min(100.0, float(pass_max)))
    if ai_percent is None:
        return {
            "schema_version": EXTERNAL_LABEL_SCHEMA_VERSION,
            "available": False,
            "status": "unlabeled",
            "ai_percent": None,
            "pass_max": threshold,
            "passed": None,
        }
    passed = ai_percent <= threshold
    return {
        "schema_version": EXTERNAL_LABEL_SCHEMA_VERSION,
        "available": True,
        "status": "external_label_passed" if passed else "external_label_failed",
        "ai_percent": round(ai_percent, 3),
        "pass_max": threshold,
        "passed": passed,
    }


def calibration_policy_to_dict() -> dict[str, Any]:
    dataset_path = os.environ.get("DRAFTPROOF_REWRITE_V2_EXTERNAL_CALIBRATION_DATASET", "")
    records = load_external_calibration_labels(dataset_path)
    threshold_summary = derive_external_proxy_thresholds(records)
    return {
        "schema_version": EXTERNAL_LABEL_SCHEMA_VERSION,
        "external_label_pass_max": external_label_pass_max(),
        "dataset_path": dataset_path,
        "has_dataset": bool(dataset_path),
        "threshold_summary": threshold_summary,
        "purpose": "Tune external proxy thresholds from labeled detector outcomes, not single-run anecdotes.",
    }


def load_external_calibration_labels(path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """Load optional JSON/JSONL calibration records for offline analysis/tests."""
    raw_path = path or os.environ.get("DRAFTPROOF_REWRITE_V2_EXTERNAL_CALIBRATION_DATASET")
    if not raw_path:
        return []
    return list(_load_external_calibration_labels_cached(str(raw_path)))


@lru_cache(maxsize=8)
def _load_external_calibration_labels_cached(raw_path: str) -> tuple[dict[str, Any], ...]:
    file_path = Path(raw_path)
    if not file_path.exists() or not file_path.is_file():
        return ()
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return ()
    if file_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        rows = parsed if isinstance(parsed, list) else parsed.get("records", []) if isinstance(parsed, dict) else []
    return tuple(row for row in rows if isinstance(row, dict))


def summarize_external_calibration_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [classify_external_label(row.get("external_label", row)) for row in records]
    available = [label for label in labels if label.get("available")]
    passed = sum(1 for label in available if label.get("passed"))
    failed = sum(1 for label in available if label.get("passed") is False)
    return {
        "schema_version": EXTERNAL_LABEL_SCHEMA_VERSION,
        "records": len(records),
        "labeled_records": len(available),
        "passed": passed,
        "failed": failed,
        "pass_max": external_label_pass_max(),
    }


def _proxy_score(row: dict[str, Any]) -> float | None:
    for key in ("external_proxy_score", "proxy_score", "score"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return max(0.0, min(100.0, float(value)))
    proxy = row.get("external_detector_proxy") or row.get("external_proxy")
    if isinstance(proxy, dict) and isinstance(proxy.get("score"), (int, float)):
        return max(0.0, min(100.0, float(proxy["score"])))
    return None


def derive_external_proxy_thresholds(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive a proxy safe threshold from labeled external detector outcomes.

    Records may be JSON/JSONL rows containing an external label plus either
    `external_proxy_score`, `proxy_score`, or `external_detector_proxy.score`.
    The chosen threshold maximizes balanced accuracy, with ties favoring the
    stricter lower threshold.
    """
    labeled: list[tuple[float, bool]] = []
    for row in records:
        label = classify_external_label(row.get("external_label", row))
        score = _proxy_score(row)
        if score is None or label.get("passed") is None:
            continue
        labeled.append((score, bool(label["passed"])))
    if not labeled:
        return {
            "status": "insufficient_labeled_proxy_records",
            "labeled_proxy_records": 0,
            "safe_threshold": None,
        }
    candidates = sorted({score for score, _passed in labeled} | {0.0, 100.0})
    best: dict[str, Any] | None = None
    for threshold in candidates:
        tp = sum(1 for score, passed in labeled if passed and score <= threshold)
        fn = sum(1 for score, passed in labeled if passed and score > threshold)
        tn = sum(1 for score, passed in labeled if not passed and score > threshold)
        fp = sum(1 for score, passed in labeled if not passed and score <= threshold)
        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        balanced_accuracy = (tpr + tnr) / 2.0
        row = {
            "safe_threshold": round(threshold, 3),
            "balanced_accuracy": round(balanced_accuracy, 4),
            "true_positive": tp,
            "false_negative": fn,
            "true_negative": tn,
            "false_positive": fp,
        }
        if best is None or (row["balanced_accuracy"], -row["safe_threshold"]) > (best["balanced_accuracy"], -best["safe_threshold"]):
            best = row
    return {
        "status": "derived_from_labeled_proxy_records",
        "labeled_proxy_records": len(labeled),
        **(best or {}),
    }
