"""Citation detector — wraps the citation cross-checker into BaseDetector.

Cross-checks in-text citations against the bibliography/reference list and
flags claims that lack supporting citations. Regex-based; may miss
non-standard citation formats.
"""

import re
import sys
import os

from .base import BaseDetector, DetectResult, Finding

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_BIBLIO_HEADINGS = re.compile(
    r"(?m)^(?:references|bibliography|works?\s+cited|reference\s+list|sources)\s*$",
    re.IGNORECASE,
)


class CitationDetector(BaseDetector):

    def __init__(self, auto_extract_bibliography: bool = True):
        self._scanner = None
        self._auto_extract = auto_extract_bibliography

    @property
    def name(self) -> str:
        return "citation"

    @property
    def detector_version(self) -> str:
        return "0.2.0"

    @property
    def policy_message(self) -> str:
        return (
            "Citation checks are regex-based and may miss non-standard citation formats. "
            "Results should be reviewed manually before drawing conclusions."
        )

    def _ensure_scanner(self):
        if self._scanner is None:
            from citation.scanner import CitationScanner
            self._scanner = CitationScanner()

    def detect(self, content: str, **kwargs) -> DetectResult:
        bib_text = kwargs.get("bib_text")

        if not bib_text and self._auto_extract:
            body, bib = self._extract_reference_section(content)
            if bib:
                content = body
                bib_text = bib

        if not bib_text:
            # No bibliography: we cannot cross-check (missing_from_bib /
            # uncited_in_body need a reference list). But in-text citation
            # EXTRACTION only needs the body, so still count inline cites and
            # emit a raw CitationResult. This populates in_text_count downstream
            # (report/builder.py citation axis) for inline-cited-but-no-bib docs.
            # Deliberately NO cross-check and NO uncited_claim findings here:
            # firing those without a bib over-flags claim-heavy ESL prose, which
            # the grounding-gated citation axis already handles.
            return self._extract_only_result(content)

        self._ensure_scanner()
        result = self._scanner.scan(content, bib_text)

        confidence, confidence_reason = self._assess_confidence(content, **kwargs)
        bib_entry_count = len([l for l in bib_text.split("\n") if l.strip()])
        if bib_entry_count < 3:
            confidence = "low"
            confidence_reason = (
                f"Only {bib_entry_count} bibliography entries found; "
                "cross-check coverage is limited."
            )

        findings = []
        for idx, f in enumerate(result.findings):
            evidence_strength = "moderate"
            if f.finding_type == "missing_from_bib":
                evidence_strength = "strong"
            elif f.finding_type == "uncited_claim":
                evidence_strength = "moderate"

            findings.append(Finding(
                finding_type=f.finding_type,
                risk_level=f.risk_level,
                evidence_strength=evidence_strength,
                detail=f.detail,
                evidence=f.evidence,
                recommendation=f.recommendation,
                suggested_action_type=self._action_type_for(f.finding_type),
                location=self._locate_reference(content, f.evidence),
            ))

        overall_risk = self._compute_risk(result.stats)

        dist = {}
        for f in result.findings:
            dist[f.risk_level] = dist.get(f.risk_level, 0) + 1

        return DetectResult(
            scanner=self.name,
            overall_risk=overall_risk,
            confidence=confidence,
            confidence_reason=confidence_reason,
            risk_distribution=dist,
            findings=findings,
            policy_message=self.policy_message,
            raw=result,
            detector_version=self.detector_version,
        )

    def _extract_only_result(self, content: str) -> DetectResult:
        """Count in-text citations without a bibliography cross-check.

        Runs only the extraction half of the scanner (`_detect_in_text` needs
        just the body) and returns a CitationResult with in_text_citations
        populated, bib_entries empty, and no findings. overall_risk stays 0.0 —
        counting cites is not a risk signal; it only lets downstream code see
        that the doc cites sources.
        """
        from citation.scanner import CitationResult

        self._ensure_scanner()
        citations = self._scanner._detect_in_text(content)
        style = self._scanner._detect_citation_style(citations)
        raw = CitationResult(
            citation_style=style,
            in_text_citations=citations,
            bib_entries=[],
            findings=[],
            stats={
                "in_text_count": len(citations),
                "bib_count": 0,
                "missing_from_bib": 0,
                "uncited_in_body": 0,
                "uncited_claims": 0,
            },
        )
        return DetectResult(
            scanner=self.name,
            overall_risk=0.0,
            confidence="low",
            confidence_reason=(
                "No bibliography provided; counted in-text citations only "
                "(no cross-check performed)."
            ),
            risk_distribution={},
            findings=[],
            policy_message=self.policy_message,
            raw=raw,
            detector_version=self.detector_version,
        )

    @staticmethod
    def _extract_reference_section(content: str) -> tuple:
        match = _BIBLIO_HEADINGS.search(content)
        if match:
            body = content[:match.start()].strip()
            bib = content[match.end():].strip()
            return body, bib
        return content, ""

    @staticmethod
    def _compute_risk(stats: dict) -> float:
        missing = stats.get("missing_from_bib", 0)
        uncited_body = stats.get("uncited_in_body", 0)
        uncited_claims = stats.get("uncited_claims", 0)
        total_issues = missing + uncited_body + uncited_claims

        if total_issues == 0:
            return 0.0
        if missing >= 3:
            return 0.8
        if missing >= 1 and uncited_claims >= 2:
            return 0.7
        if uncited_claims >= 3:
            return 0.6
        if total_issues >= 3:
            return 0.5
        if total_issues >= 1:
            return 0.3
        return 0.1

    @staticmethod
    def _action_type_for(finding_type: str) -> str:
        mapping = {
            "missing_from_bib": "verify_reference",
            "uncited_in_body": "remove_unused_reference",
            "uncited_claim": "add_citation",
        }
        return mapping.get(finding_type, "review_manually")

    @staticmethod
    def _locate_reference(content: str, evidence: str) -> dict:
        start = content.find(evidence)
        if start >= 0:
            return {"start_char": start, "end_char": start + len(evidence)}
        return {}
