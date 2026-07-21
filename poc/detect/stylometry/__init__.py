"""Stylometric feature extraction — standalone measurement library.

Paragraph-level stylometric fingerprinting and outlier detection for the
"Consistency risk" detector (build plan:
docs/plans/consistency_defence_readiness_build_plan.md). This package is currently a
standalone, independently-testable library: `extract_fingerprints` and
`detect_outliers` are both pure functions with zero I/O and zero calls into any other
`poc/detect/` module besides `document_structure.py` (reused only by `features.py` for
paragraph/sentence segmentation). It is NOT yet wired into the detection pipeline — the
`ConsistencyDetector` that consumes these outputs is a later task.
"""
from __future__ import annotations

from .features import ParagraphFingerprint, extract_fingerprints
from .outliers import (
    MIN_PARAGRAPHS,
    MIN_WORDS_PER_PARAGRAPH,
    OUTLIER_MIN_DEVIATING_FEATURES,
    OUTLIER_PER_FEATURE_Z_THRESHOLD,
    OUTLIER_THRESHOLD,
    OutlierResult,
    OutlierStrategy,
    detect_outliers,
    score_paragraphs,
)

__all__ = [
    "ParagraphFingerprint",
    "extract_fingerprints",
    "OutlierResult",
    "OutlierStrategy",
    "detect_outliers",
    "score_paragraphs",
    "MIN_PARAGRAPHS",
    "MIN_WORDS_PER_PARAGRAPH",
    "OUTLIER_THRESHOLD",
    "OUTLIER_PER_FEATURE_Z_THRESHOLD",
    "OUTLIER_MIN_DEVIATING_FEATURES",
]
