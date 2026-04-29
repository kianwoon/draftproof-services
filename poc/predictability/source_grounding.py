"""Source grounding analyzer — verify claims against source material.

For each paragraph:
1. Identify main claim(s)
2. Check for citation/source reference
3. Assess whether the claim is supported by cited material
4. Flag unsupported claims

This is NOT plagiarism detection. It identifies whether writing
is grounded in referenced sources or makes unsupported assertions.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class ClaimAnalysis:
    paragraph_id: str
    claims: List[str]
    has_citation: bool
    has_source_reference: bool
    support_strength: str      # "strong", "partial", "weak", "none"
    detail: str


@dataclass
class GroundingReport:
    paragraph_analyses: List[ClaimAnalysis]
    overall_grounding_risk: float   # 0-1, high = weak grounding
    unsupported_claim_count: int
    total_claims: int
    recommendations: List[str]
    detail: str


# ── Source reference patterns ──────────────────────────────────────

CITATION_PATTERNS = [
    r'\(\w+,?\s*\d{4}\)',        # (Author, 2024)
    r'\(\w+\s+et\s+al\.?\s*,?\s*\d{4}\)',  # (Author et al., 2024)
    r'\[\d+\]',                   # [1]
    r'\[\d+,\s*\d+\]',           # [1, 3]
    r'(?:page|p\.)\s+\d+',       # page 42
    r'cf\.',
    r'ibid\.',
    r'op\.?\s*cit\.',
]

SOURCE_REFERENCE_WORDS = [
    "according to", "as stated by", "as noted by", "cited in",
    "referenced in", "draws on", "based on", "building on",
    "research shows", "studies show", "evidence suggests",
    "findings indicate", "data from", "reported by",
    "the source", "my source", "the reading", "the text",
    "the article", "the study", "the paper", "the book",
    "the journal", "chapter", "section",
]

CLAIM_INDICATORS = [
    "argues that", "claims that", "asserts that", "suggests that",
    "demonstrates that", "shows that", "proves that", "indicates that",
    "i argue", "i believe", "i contend", "i maintain",
    "this demonstrates", "this shows", "this suggests",
    "this proves", "this indicates",
    "therefore", "thus", "hence", "consequently",
    "it is clear that", "it is evident that",
]


class SourceGroundingAnalyzer:
    """Analyze whether text claims are grounded in source material."""

    def __init__(self, source_cards: Optional[List[Dict]] = None):
        self.source_cards = source_cards or []

    def analyze(self, text: str) -> GroundingReport:
        paragraphs = self._split_paragraphs(text)
        analyses = []
        total_claims = 0
        unsupported = 0

        for i, para in enumerate(paragraphs):
            analysis = self._analyze_paragraph(para, f"p{i:02d}")
            analyses.append(analysis)
            total_claims += len(analysis.claims)
            if analysis.support_strength in ("weak", "none") and analysis.claims:
                unsupported += len(analysis.claims)

        # Overall risk
        if total_claims == 0:
            risk = 0.0
        else:
            risk = unsupported / total_claims

        # Recommendations
        recs = self._generate_recommendations(analyses)

        detail = (
            f"paragraphs={len(paragraphs)}, total_claims={total_claims}, "
            f"unsupported={unsupported}, risk={risk:.3f}"
        )

        return GroundingReport(
            paragraph_analyses=analyses,
            overall_grounding_risk=round(risk, 4),
            unsupported_claim_count=unsupported,
            total_claims=total_claims,
            recommendations=recs,
            detail=detail,
        )

    def _analyze_paragraph(self, para: str, pid: str) -> ClaimAnalysis:
        lower = para.lower()

        # Detect claims
        claims = self._extract_claims(para)

        # Detect citations
        has_citation = any(re.search(p, lower) for p in CITATION_PATTERNS)

        # Detect source references
        has_source_ref = any(ref in lower for ref in SOURCE_REFERENCE_WORDS)

        # Assess support
        if not claims:
            support = "none"
            detail = "No claims detected in this paragraph."
        elif has_citation and has_source_ref:
            support = "strong"
            detail = f"Claims are supported by citation and source reference."
        elif has_citation:
            support = "partial"
            detail = f"Claims have citations but lack explicit source references."
        elif has_source_ref:
            support = "partial"
            detail = f"Claims reference sources but lack formal citations."
        else:
            support = "weak"
            detail = f"Claims lack citations or source references."

        return ClaimAnalysis(
            paragraph_id=pid,
            claims=claims,
            has_citation=has_citation,
            has_source_reference=has_source_ref,
            support_strength=support,
            detail=detail,
        )

    def _extract_claims(self, text: str) -> List[str]:
        """Extract claim-bearing sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []
        lower = text.lower()

        for sent in sentences:
            sent_lower = sent.lower()
            is_claim = any(ind in sent_lower for ind in CLAIM_INDICATORS)
            # Also check for hedge-free assertive sentences
            if not is_claim and len(sent.split()) >= 10:
                # Check if it's a declarative statement without hedging
                has_hedge = any(h in sent_lower for h in [
                    "may", "might", "could", "perhaps", "possibly",
                    "it seems", "in my opinion", "i think"
                ])
                if not has_hedge and sent.endswith(('.',)):
                    # Strong declarative — potential claim
                    pass  # Don't count these as claims without indicators

            if is_claim:
                claims.append(sent.strip())

        return claims

    def _generate_recommendations(self, analyses: List[ClaimAnalysis]) -> List[str]:
        recs = []
        weak_paras = [a for a in analyses if a.support_strength == "weak" and a.claims]
        if weak_paras:
            para_ids = ", ".join(a.paragraph_id for a in weak_paras)
            recs.append(
                f"Paragraphs [{para_ids}] contain claims without citations. "
                "Consider adding source references."
            )

        no_cite = [a for a in analyses if not a.has_citation]
        if len(no_cite) == len(analyses) and len(analyses) > 2:
            recs.append(
                "No formal citations detected in any paragraph. "
                "Academic writing typically requires source attribution."
            )

        partial = [a for a in analyses if a.support_strength == "partial"]
        if partial:
            recs.append(
                f"{len(partial)} paragraph(s) have partial grounding. "
                "Strengthen by adding explicit citations or source references."
            )

        return recs

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
