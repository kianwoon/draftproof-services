"""Similarity detector — wraps the source-overlap scanner into BaseDetector.

Detects close paraphrasing and source overlap at sentence and paragraph level.
Citation nearby does NOT automatically reduce severity — a citation without
quotation marks is still close paraphrasing.
"""

import re
import sys
import os
from typing import List, Optional

from .base import BaseDetector, DetectResult, Finding

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_RISK_MAP = {"high": 0.8, "medium": 0.5, "low": 0.2}

_CITATION_REGEX = re.compile(
    r"\([^)]*\d{4}[^)]*\)"   # (Author, Year)
    r"|\[\d+(?:[,\-]\d+)*\]"  # [1], [1,2], [1-3]
    r"|[²³⁰-⁹]+"  # superscript numerals
)


class SimilarityDetector(BaseDetector):

    def __init__(self):
        self._scanner = None

    @property
    def name(self) -> str:
        return "similarity"

    @property
    def detector_version(self) -> str:
        return "0.2.0"

    @property
    def policy_message(self) -> str:
        return (
            "Similarity does not automatically mean plagiarism. "
            "Quotation, citation, common phrasing, and assignment wording must be reviewed "
            "in context before drawing conclusions."
        )

    def _ensure_scanner(self):
        if self._scanner is None:
            from similarity.scanner import SimilarityScanner
            self._scanner = SimilarityScanner()

    def detect(self, content: str, **kwargs) -> DetectResult:
        source_sentences = kwargs.get("source_sentences")
        if not source_sentences:
            return DetectResult(
                scanner=self.name,
                overall_risk=0.0,
                confidence="low",
                confidence_reason="No source sentences provided; similarity check skipped.",
                findings=[],
                policy_message=self.policy_message,
                detector_version=self.detector_version,
            )

        self._ensure_scanner()

        confidence, confidence_reason = self._assess_confidence(content, **kwargs)
        if len(source_sentences) < 3:
            confidence = "low"
            confidence_reason = (
                f"Only {len(source_sentences)} source sentences provided; "
                "comparison coverage is limited."
            )

        draft_sentences = self._split_sentences(content)
        source_id = kwargs.get("source_id", "source_0")

        result = self._scanner.scan(
            draft_sentences=draft_sentences,
            source_sentences=source_sentences,
            source_id=source_id,
        )

        findings = []
        for m in result.findings:
            citation_nearby = getattr(m, "citation_nearby", False)
            quotation_detected = self._check_quotation(m.draft_sentence)
            citation_dist = self._citation_distance(content, m.draft_sentence)

            evidence_strength = self._assess_evidence(m)
            if citation_nearby and not quotation_detected:
                evidence_strength = "moderate"

            findings.append(Finding(
                finding_type=m.risk_type,
                risk_level=m.risk_level,
                evidence_strength=evidence_strength,
                detail=(
                    f"{m.risk_type.replace('_', ' ').title()}: "
                    f"exact={m.exact_score:.0%}, fuzzy={m.fuzzy_score:.0%}, "
                    f"semantic={m.semantic_score:.0%}"
                ),
                evidence=m.draft_sentence[:120],
                recommendation=self._recommendation_for(m.risk_level, citation_nearby, quotation_detected),
                suggested_action_type=self._action_type_for(m.risk_level, citation_nearby, quotation_detected),
                location=self._locate_sentence(content, m.draft_sentence),
                metadata={
                    "draft_sentence": m.draft_sentence,
                    "source_sentence": m.source_sentence,
                    "exact_score": m.exact_score,
                    "fuzzy_score": m.fuzzy_score,
                    "semantic_score": m.semantic_score,
                    "citation_nearby": citation_nearby,
                    "quotation_detected": quotation_detected,
                    "citation_distance_chars": citation_dist,
                    "source_id": m.source_id,
                },
            ))

        source_paragraphs = kwargs.get("source_paragraphs")
        if source_paragraphs:
            findings.extend(
                self._paragraph_level_scan(content, source_paragraphs, source_id)
            )

        overall_risk = _RISK_MAP.get(result.overall_risk, 0.0)

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

    def _paragraph_level_scan(self, content: str, source_paragraphs: List[str],
                              source_id: str) -> List[Finding]:
        draft_paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        findings = []
        for pi, dp in enumerate(draft_paragraphs):
            best_score = 0.0
            best_src_idx = -1
            for si, sp in enumerate(source_paragraphs):
                score = self._quick_semantic_similarity(dp, sp)
                if score > best_score:
                    best_score = score
                    best_src_idx = si

            if best_score >= 0.75:
                risk = "high" if best_score >= 0.85 else "medium"
                findings.append(Finding(
                    finding_type="paragraph_level_overlap",
                    risk_level=risk,
                    evidence_strength="moderate",
                    detail=(
                        f"Paragraph {pi+1} has {best_score:.0%} semantic overlap "
                        f"with source paragraph {best_src_idx+1}"
                    ),
                    evidence=dp[:120],
                    recommendation=(
                        "Review paragraph structure and argument flow. "
                        "Restructure with original argument sequencing."
                    ),
                    suggested_action_type="rewrite_from_source_card",
                    location={"paragraph_index": pi},
                    metadata={
                        "semantic_score": best_score,
                        "matched_source_paragraph_id": f"src_p_{best_src_idx:02d}",
                        "source_id": source_id,
                    },
                ))
        return findings

    @staticmethod
    def _quick_semantic_similarity(text_a: str, text_b: str) -> float:
        """Simple word-overlap heuristic for paragraph-level screening.
        For production, replace with sentence-transformer embedding cosine."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _check_quotation(text: str) -> bool:
        return bool(re.search(r'["""]', text))

    @staticmethod
    def _citation_distance(content: str, sentence: str) -> int:
        start = content.find(sentence[:40])
        if start < 0:
            return -1
        match = _CITATION_REGEX.search(content, start, start + len(sentence) + 80)
        if match:
            return match.start() - (start + len(sentence))
        return -1

    @staticmethod
    def _assess_evidence(match) -> str:
        scores = [match.exact_score, match.fuzzy_score, match.semantic_score]
        if any(s >= 0.85 for s in scores):
            return "strong"
        if any(s >= 0.65 for s in scores):
            return "moderate"
        return "weak"

    @staticmethod
    def _recommendation_for(risk: str, citation_nearby: bool, quotation: bool) -> str:
        if risk == "high" and not quotation:
            return "Exact overlap without quotation. Quote the source directly and cite."
        if risk == "high" and quotation:
            return "Close overlap even with quotation. Verify citation and quotation accuracy."
        if citation_nearby and not quotation:
            return "Citation present but close paraphrasing. Add your own interpretation."
        return "Review for originality and ensure source is properly cited."

    @staticmethod
    def _action_type_for(risk: str, citation_nearby: bool, quotation: bool) -> str:
        if risk == "high" and not quotation:
            return "quote_and_cite"
        if risk == "high":
            return "verify_reference"
        if citation_nearby:
            return "add_user_interpretation"
        return "review_manually"

    @staticmethod
    def _locate_sentence(content: str, sentence: str) -> dict:
        start = content.find(sentence[:40])
        if start >= 0:
            return {"start_char": start, "end_char": start + len(sentence)}
        return {}

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\(])', text)
        return [p.strip() for p in parts if p.strip()]
