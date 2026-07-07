from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager

from .scan import Scan
from .write import Variant


@contextmanager
def deep_scan_suppressed():
    """Skip the paid per-sentence Modal deep-scan for an INTERNAL/INTERMEDIATE rewrite re-scan.

    A single rewrite runs ``ReportBuilder.build()`` many times: production.py's two intermediate
    summary scans, plus the direct path's best-of-N / reviewer / selector guard scans (each one, when
    the V7 flags are on, makes a full per-sentence Modal deep-scan call with a 60s timeout). Only the
    FINAL "after" report shown to the user needs the fused deep-scan score; every internal better/
    worse guard only compares the composite ``ai_score``. ``ReportBuilder`` reads the deep-scan kill
    switches from the environment at build time (``is_deep_scan_enabled`` / ``is_tier_authority_enabled``),
    so we temporarily force them off around an internal build and restore them after (see bc354646).

    Safe under the single-task worker (``celery -c 1`` -- one task per process at a time, so no
    concurrent scan observes the toggled env). The ``finally`` restores the exact prior values.
    """
    keys = ("DRAFTPROOF_V7_TIER_AUTHORITY", "DRAFTPROOF_V7_DEEP_SCAN")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "0"
    try:
        yield
    finally:
        for k, prior in saved.items():
            if prior is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prior


def can_start_llm_request(started_at: float, runtime_budget_seconds: float | None, min_llm_request_seconds: float) -> bool:
    if runtime_budget_seconds is None or runtime_budget_seconds <= 0:
        return True
    remaining = float(runtime_budget_seconds) - (time.monotonic() - started_at)
    return remaining >= max(1.0, float(min_llm_request_seconds or 0.0))


def compose(scan: Scan, target_paragraph_id: str, selected: Variant | None) -> str:
    return "\n\n".join(
        selected.text if selected and paragraph.id == target_paragraph_id else paragraph.text
        for paragraph in scan.paragraphs
    )


def same_text(left: str, right: str) -> bool:
    return re.sub(r"\s+", " ", str(left or "").strip()) == re.sub(r"\s+", " ", str(right or "").strip())


def improved(before: Scan, after: Scan) -> bool:
    return (
        after.scores["finding_count"] < before.scores["finding_count"]
        or after.scores["mean_sentence_shape_risk"] < before.scores["mean_sentence_shape_risk"]
    )


def acceptable_progress(before: Scan, after: Scan, *, report_targeted: bool) -> bool:
    if improved(before, after):
        return True
    return bool(report_targeted) and after.scores["finding_count"] <= before.scores["finding_count"] and after.scores["mean_sentence_shape_risk"] <= before.scores["mean_sentence_shape_risk"]


def cross_paragraph_regression(before: Scan, after: Scan, target_paragraph_id: str, *, target_after_ids: set[str] | None = None) -> bool:
    before_counts = paragraph_counts(before)
    after_counts = paragraph_counts(after)
    target_after_ids = target_after_ids or {target_paragraph_id}
    target = next((paragraph for paragraph in before.paragraphs if paragraph.id == target_paragraph_id), None)
    if target is not None and len(after.paragraphs) != len(before.paragraphs):
        delta = len(after.paragraphs) - len(before.paragraphs)
        target_after_count = sum(after_counts.get(paragraph_id, 0) for paragraph_id in target_after_ids)
        target_drop = before_counts.get(target_paragraph_id, 0) - target_after_count
        total_drop = int(before.scores.get("finding_count") or 0) - int(after.scores.get("finding_count") or 0)
        max_non_target_regression = 0
        for paragraph in before.paragraphs:
            if paragraph.id == target_paragraph_id:
                continue
            after_index = paragraph.index if paragraph.index < target.index else paragraph.index + delta
            if 0 <= after_index < len(after.paragraphs):
                after_id = after.paragraphs[after_index].id
                max_non_target_regression = max(max_non_target_regression, after_counts.get(after_id, 0) - before_counts.get(paragraph.id, 0))
                if after_id in target_after_ids:
                    continue
                if after_counts.get(after_id, 0) > before_counts.get(paragraph.id, 0) and not (target_drop >= 2 and total_drop > 0 and max_non_target_regression <= 1):
                    return True
        return False
    return any(
        paragraph_id not in target_after_ids and count > before_counts.get(paragraph_id, 0)
        for paragraph_id, count in after_counts.items()
    )


def paragraph_counts(scan: Scan) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in scan.findings:
        counts[finding.paragraph_id] = counts.get(finding.paragraph_id, 0) + 1
    return counts


def rewrite_progress_percent(step: int, limit: int) -> int:
    span_start = 63
    span_end = 87
    bounded_limit = max(1, int(limit or 1))
    bounded_step = max(0, min(bounded_limit, int(step or 0)))
    return span_start + int(round((span_end - span_start) * (bounded_step / bounded_limit)))
