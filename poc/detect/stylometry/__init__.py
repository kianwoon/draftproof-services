"""Stylometric feature extraction — standalone measurement library.

Paragraph-level stylometric fingerprinting for the "Consistency risk" detector
(build plan: docs/plans/consistency_defence_readiness_build_plan.md). This package is
currently a standalone, independently-testable library: `extract_fingerprints` has zero
I/O and zero calls into any other `poc/detect/` module besides `document_structure.py`.
It is NOT yet wired into the detection pipeline — outlier detection and the
`ConsistencyDetector` that consume these fingerprints are later tasks.
"""
from __future__ import annotations

from .features import ParagraphFingerprint, extract_fingerprints

__all__ = ["ParagraphFingerprint", "extract_fingerprints"]
