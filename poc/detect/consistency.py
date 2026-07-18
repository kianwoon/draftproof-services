"""ConsistencyDetector — wires Task 1's stylometric fingerprints
(``poc/detect/stylometry/features.py``) and Task 2's paragraph-outlier detection
(``poc/detect/stylometry/outliers.py``) into the live detection pipeline, behind the
``DRAFTPROOF_CONSISTENCY`` kill switch (default OFF).

Governing plan: docs/plans/consistency_defence_readiness_build_plan.md

Phase 1 scope: INFORMATIONAL ONLY. A flagged paragraph means "this paragraph's
writing style reads differently than the rest of the document" — one possible
explanation is a swapped-in AI-generated or outsourced paragraph, but plenty of
innocent explanations exist too (a quoted passage, a genuinely different section, a
paragraph drafted on a different day). ``DetectResult.overall_risk`` is therefore
unconditionally ``0.0``: this scanner contributes ZERO weight to
``DetectionRunner._aggregate_risk``'s fused score, and is NOT wired into
``poc/detect/layer3_scoring.py`` at all — that module's
``build_layer3_input_from_text`` sources every input from named keyword arguments
(e.g. ``semantic_uniformity_risk``, ``discourse_regularity_risk``), never by
generically scanning ``DetectionReport.scanner_results``, so this detector's output
is structurally never among them.

Findings map to the EXISTING ``"authorship_risk"`` signal_category
(``poc/detect/base.py``'s ``_SIGNAL_CATEGORY_MAP``) rather than a new category value —
an explicit product decision (see the Task 3 brief) to avoid frontend enum risk in
this phase.

Kill switch: ``DRAFTPROOF_CONSISTENCY``, default ``"0"`` (OFF). Mirrors
``poc/claim_graph/__init__.py``'s ``claim_graph_enabled()`` falsey-set/default-off
style. When OFF, ``poc/detect/run.py::DetectionRunner._build_detectors()`` never
constructs this class at all — not merely "produces no findings".

allow-hardcode: the policy_message / Finding.detail / Finding.recommendation strings
below are human-readable REPORT COPY shown to end users (mirroring the existing
static copy in poc/detect/semantic_shape.py's _findings_for) — they are fixed prose,
not a matching/scoring oracle keyed off document content.
"""
from __future__ import annotations

import os

from .base import BaseDetector, DetectResult, Finding
from .document_structure import structured_sentences
from .stylometry.features import extract_fingerprints
from .stylometry.outliers import OutlierResult, detect_outliers

# Name is duplicated as a string literal (env var name) so downstream readers can see
# which switch governs this detector without importing this module — mirrors
# poc/claim_graph/__init__.py's KILL_SWITCH_ENV convention.
CONSISTENCY_KILL_SWITCH_ENV = "DRAFTPROOF_CONSISTENCY"

# Same falsey-value set as poc/claim_graph/__init__.py's claim_graph_enabled(), so
# "0" / "false" / "no" / "off" / "" (or unset) are all treated as OFF.
_FALSEY = {"0", "false", "no", "off", ""}

# Truncation length for the paragraph-text excerpt shown as Finding.evidence —
# matches the 500-char convention already used by SemanticShapeDetector
# (poc/detect/semantic_shape.py) for the same "readable excerpt, not the whole
# paragraph" purpose.
_EVIDENCE_EXCERPT_CHARS = 500

# Single source of truth for this detector's DetectResult.scanner / BaseDetector.name
# value, exported so callers (e.g. poc/detect/run.py's post-processing loop) can
# compare against it WITHOUT constructing a ConsistencyDetector instance — important
# because the kill-switch contract requires this class to never be instantiated at
# all while DRAFTPROOF_CONSISTENCY is off.
SCANNER_NAME = "consistency"


def consistency_enabled() -> bool:
    """Return whether the Phase-1 stylometric-consistency detector is enabled.

    Default OFF (opt-in). Mirrors ``poc/claim_graph/__init__.py``'s
    ``claim_graph_enabled()`` — reads the env var live on every call (not cached at
    import time) so tests can toggle it with ``monkeypatch.setenv``.
    """
    return os.environ.get(CONSISTENCY_KILL_SWITCH_ENV, "0").strip().lower() not in _FALSEY


class ConsistencyDetector(BaseDetector):
    """Flags paragraphs whose stylometric fingerprint deviates sharply from the rest
    of the document. See module docstring for Phase-1 scope and limits.
    """

    @property
    def name(self) -> str:
        return SCANNER_NAME

    @property
    def detector_version(self) -> str:
        return "0.1.0"

    @property
    def policy_message(self) -> str:
        return (
            "Consistency findings flag paragraphs that read in a different writing "
            "voice than the rest of the document. They are informational only in "
            "this phase, do not affect the overall risk score, and are not "
            "standalone evidence of AI generation or outsourcing — a different "
            "voice can also come from a quoted passage, a section written on a "
            "different day, or a legitimate co-author."
        )

    def detect(self, content: str, **kwargs) -> DetectResult:
        confidence, confidence_reason = self._assess_confidence(content, **kwargs)

        fingerprints = extract_fingerprints(content)
        outliers = detect_outliers(fingerprints)
        paragraph_texts = _paragraph_texts_by_id(content)

        findings = [
            self._finding_for(outlier, paragraph_texts.get(outlier.paragraph_id, ""))
            for outlier in outliers
        ]

        risk_distribution: dict[str, int] = {}
        for finding in findings:
            risk_distribution[finding.risk_level] = risk_distribution.get(finding.risk_level, 0) + 1

        return DetectResult(
            scanner=self.name,
            # Phase 1: informational only — see module docstring. This is
            # unconditional: it does not depend on how many paragraphs were
            # flagged, or how extreme their outlier_score is.
            overall_risk=0.0,
            confidence=confidence,
            confidence_reason=confidence_reason,
            risk_distribution=risk_distribution,
            findings=findings,
            policy_message=self.policy_message,
            detector_version=self.detector_version,
            raw={
                "paragraph_count": len(fingerprints),
                "outlier_count": len(outliers),
            },
        )

    def _finding_for(self, outlier: OutlierResult, paragraph_text: str) -> Finding:
        feature_list = ", ".join(outlier.top_deviating_features) or "overall writing style"
        detail = (
            "This paragraph's writing style deviates sharply from the rest of the "
            f"document — most notably in {feature_list}."
        )
        return Finding(
            finding_type="stylometric_outlier",
            # "review" (not "high"/"medium"/"low"): this is an informational,
            # review-only signal, not a scored risk tier — see module docstring.
            risk_level="review",
            evidence_strength="moderate",
            detail=detail,
            evidence=paragraph_text[:_EVIDENCE_EXCERPT_CHARS],
            recommendation=(
                "Review this paragraph — its sentence structure and word choice "
                "read differently than the rest of the document. Confirm it "
                "reflects your own writing voice, or note why it should read "
                "differently (e.g. a quotation)."
            ),
            suggested_action_type="review_stylometric_outlier",
            location={"paragraph_id": outlier.paragraph_id, "scope": "paragraph"},
            metadata={
                "outlier_score": outlier.outlier_score,
                "top_deviating_features": list(outlier.top_deviating_features),
                "actionability": "review_only",
            },
            signal_category="authorship_risk",
            actionability="review_only",
        )


def _paragraph_texts_by_id(content: str) -> dict[str, str]:
    """Map paragraph_id -> full paragraph text, joined the same way
    ``stylometry/features.py``'s ``_build_fingerprint`` joins its sentences — so the
    ``Finding.evidence`` excerpt matches the exact text the fingerprint was computed
    from."""
    sentences_by_paragraph: dict[str, list[str]] = {}
    for row in structured_sentences(content):
        sentences_by_paragraph.setdefault(row.paragraph_id, []).append(row.sentence)
    return {
        paragraph_id: " ".join(text.strip() for text in sentences if text.strip()).strip()
        for paragraph_id, sentences in sentences_by_paragraph.items()
    }


__all__ = [
    "CONSISTENCY_KILL_SWITCH_ENV",
    "SCANNER_NAME",
    "ConsistencyDetector",
    "consistency_enabled",
]
