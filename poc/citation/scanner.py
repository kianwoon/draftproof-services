"""Citation checker -- cross-check body citations against bibliography.

Capabilities:
  1. Extract in-text citations (APA/Harvard, IEEE/numeric, Vancouver)
  2. Parse bibliography / reference list entries
  3. Cross-check: cited-in-body-but-not-in-bibliography
  4. Cross-check: in-bibliography-but-not-cited-in-body
  5. Detect uncited claims (sentences with claim indicators but no citation)

Run:  cd poc/citation && python demo.py
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set


# ── Citation pattern extractors ─────────────────────────────────────

# APA / Harvard: (Author, Year) or (Author & Author, Year) or (Author et al., Year)
APA_RE = re.compile(
    r"\("
    r"[A-Z][a-zäëïöü]+"
    r"(?:\s+(?:et\s+al\.|&\s+[A-Z][a-zäëïöü]+|and\s+[A-Z][a-zäëïöü]+))*"
    r",\s*\d{4}[a-z]?"
    r"\)"
)

# Multiple authors in one bracket: (Smith & Jones, 2022; Lee, 2021)
APA_MULTI_RE = re.compile(
    r"\("
    r"[A-Z][a-z]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-z]+))*"
    r",\s*\d{4}[a-z]?"
    r"(?:\s*;\s*"
    r"[A-Z][a-z]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-z]+))*"
    r",\s*\d{4}[a-z]?"
    r")*"
    r"\)"
)

# IEEE / Numeric: [1] or [1,2] or [1-3]
NUMERIC_RE = re.compile(r"\[(\d+(?:\s*[,\-]\s*\d+)*)\]")

# Vancouver: superscript-style 1 or 1,2 or 1-3 (after punctuation or at end)
VANCOUVER_RE = re.compile(r"(?:\.|,|\))\s*(?:\^)?(\d+(?:\s*[,\-]\s*\d+)*)")

# Narrative citation: Smith (2022) or Smith et al. (2022)
NARRATIVE_RE = re.compile(
    r"[A-Z][a-z]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-z]+))*\s*"
    r"\(\d{4}[a-z]?\)"
)


# ── Bibliography entry detection ────────────────────────────────────

BIB_ENTRY_RE = re.compile(
    r"^(?:\[(\d+)\]\s*)?"               # optional [1] prefix
    r"([A-ZÀ-ÖØ-Ý一-鿿][^\n]{0,80}?\.\s*"
    r"(?:\((?:\d{4}|n\.d\.)\)|\d{4})"   # (Year) or (n.d.) or bare year
    r"[^\n]*)"
    r"$",
    re.MULTILINE | re.DOTALL,
)

# Year extraction from bibliography
YEAR_RE = re.compile(r"\((\d{4})\)")

# Author surname extraction (first author from bib entry)
# Handles: Western surnames, Chinese characters, organisation names
FIRST_AUTHOR_RE = re.compile(
    r"^([A-ZÀ-ÖØ-Ý一-鿿][a-zäëïöüà-ÿ一-鿿]*)"
)


# ── Claim indicators (sentences likely needing citations) ───────────

CLAIM_PATTERNS = [
    re.compile(r"\b(?:found|showed|demonstrated|reported|concluded|suggests?)\b", re.I),
    re.compile(r"\b(?:evidence|data|study|studies|research)\s+(?:shows?|suggests?|indicates?|found|reveals?)\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?%\b"),  # statistics
    re.compile(r"\baccording\s+to\b", re.I),
    re.compile(r"\b(?:increase|decrease|improve|reduce|affect|impact)\s+(?:by|of|in)\b", re.I),
]


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class InTextCitation:
    raw: str
    style: str               # apa | numeric | narrative
    authors: List[str]
    year: Optional[str]
    numbers: List[int]       # for numeric style
    position: int            # char offset in text


@dataclass
class BibEntry:
    raw: str
    number: Optional[int]    # [1] style
    first_author: str
    year: Optional[str]
    key: str                 # normalised match key (surname+year)


@dataclass
class CitationFinding:
    finding_type: str        # missing_from_bib | uncited_in_body | uncited_claim
    risk_level: str          # low | medium | high
    detail: str
    evidence: str
    recommendation: str


@dataclass
class CitationResult:
    citation_style: str      # apa | numeric | mixed | unknown
    in_text_citations: List[InTextCitation]
    bib_entries: List[BibEntry]
    findings: List[CitationFinding]
    stats: Dict[str, int] = field(default_factory=dict)


# ── Normalisation helpers ───────────────────────────────────────────

def _normalise_key(surname: str, year: Optional[str]) -> str:
    """Create a match key: 'smith2022'."""
    s = surname.lower().strip()
    y = year or ""
    return f"{s}{y}"


def _extract_apa_authors(raw: str) -> List[str]:
    """Extract author surnames from APA citation like '(Smith & Jones, 2022)'."""
    # Strip outer parens
    inner = raw.strip("()")
    # Remove year
    inner = re.sub(r",?\s*\d{4}[a-z]?$", "", inner).strip()
    if not inner:
        return []
    # Split on ; (multi-citation) or & or "and"
    parts = re.split(r"\s*;\s*|\s*&\s*|\s+and\s+", inner)
    # Clean up "et al."
    return [p.replace(" et al.", "").strip() for p in parts if p.strip()]


def _extract_numeric_refs(raw: str) -> List[int]:
    """Parse [1,2,3-5] into [1, 2, 3, 4, 5]."""
    nums = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
                nums.extend(range(int(bounds[0]), int(bounds[1]) + 1))
        elif part.isdigit():
            nums.append(int(part))
    return nums


# ── Scanner ─────────────────────────────────────────────────────────

class CitationScanner:
    """Cross-check citations between body text and bibliography.

    Usage:
        scanner = CitationScanner()
        result = scanner.scan(body_text, bibliography_text)
    """

    def _detect_in_text(self, text: str) -> List[InTextCitation]:
        citations = []
        seen_positions = set()

        # APA parenthetical (including multi)
        for m in APA_MULTI_RE.finditer(text):
            if m.start() not in seen_positions:
                authors = _extract_apa_authors(m.group())
                year_match = re.search(r"(\d{4})", m.group())
                year = year_match.group(1) if year_match else None
                citations.append(InTextCitation(
                    raw=m.group(), style="apa",
                    authors=authors, year=year,
                    numbers=[], position=m.start(),
                ))
                seen_positions.add(m.start())

        # Narrative: Smith (2022)
        for m in NARRATIVE_RE.finditer(text):
            if m.start() not in seen_positions:
                author_part = re.match(r"([A-Z][a-z]+)", m.group())
                year_match = re.search(r"(\d{4})", m.group())
                author = author_part.group(1) if author_part else ""
                year = year_match.group(1) if year_match else None
                citations.append(InTextCitation(
                    raw=m.group(), style="narrative",
                    authors=[author], year=year,
                    numbers=[], position=m.start(),
                ))
                seen_positions.add(m.start())

        # Numeric: [1] [2,3] [4-6]
        for m in NUMERIC_RE.finditer(text):
            if m.start() not in seen_positions:
                nums = _extract_numeric_refs(m.group(1))
                citations.append(InTextCitation(
                    raw=m.group(), style="numeric",
                    authors=[], year=None,
                    numbers=nums, position=m.start(),
                ))
                seen_positions.add(m.start())

        return citations

    def _parse_bibliography(self, bib_text: str) -> List[BibEntry]:
        entries = []

        # Try numbered entries first: [1] Author (Year) ...
        numbered = re.findall(
            r"\[(\d+)\]\s*(.*?\.\s*$)",
            bib_text,
            re.MULTILINE | re.DOTALL,
        )
        if numbered:
            for num_str, body in numbered:
                first_author, year = self._extract_author_year(body)
                key = _normalise_key(first_author, year)
                entries.append(BibEntry(
                    raw=f"[{num_str}] {body.strip()}",
                    number=int(num_str),
                    first_author=first_author,
                    year=year,
                    key=key,
                ))
            return entries

        # APA-style: unnumbered. Use line-by-line heuristic as primary strategy.
        # This handles both single-newline and double-newline separated refs.
        lines = [l.strip() for l in bib_text.strip().split("\n") if l.strip()]
        blocks = []
        for line in lines:
            # New entry if starts with author pattern + year indicator
            is_new_entry = (
                re.match(r"^[A-ZÀ-ÖØ-Ý一-鿿]", line)
                and (
                    re.search(r"\(\d{4}[,\s)]", line)   # (2025), (2025, ...), (2025 ...
                    or re.search(r"\(n\.d\.\)", line)
                )
            )
            if is_new_entry:
                blocks.append(line)
            elif blocks:
                blocks[-1] += " " + line
            else:
                blocks.append(line)

        for block in blocks:
            block = block.strip()
            if not block or len(block) < 10:
                continue
            first_author, year = self._extract_author_year(block)
            key = _normalise_key(first_author, year)
            entries.append(BibEntry(
                raw=block,
                number=None,
                first_author=first_author,
                year=year,
                key=key,
            ))

        return entries

    @staticmethod
    def _extract_author_year(text: str) -> Tuple[str, Optional[str]]:
        """Extract first author surname and year from a bibliography entry."""
        text = text.strip()
        # Chinese author: 海底捞. (n.d.)
        cn_match = re.match(r"^([一-鿿]+)", text)
        if cn_match:
            return cn_match.group(1), None
        # Standard Western: Surname, ...
        author_m = re.match(r"^([A-ZÀ-ÖØ-Ý][a-zäëïöüà-ÿ]+)", text)
        first_author = author_m.group(1) if author_m else "Unknown"
        # Year: (2025), (2025, ...), (n.d.)
        year_m = re.search(r"\((\d{4})[,\s)]", text)
        year = year_m.group(1) if year_m else None
        return first_author, year

    def _detect_citation_style(self, citations: List[InTextCitation]) -> str:
        if not citations:
            return "unknown"
        styles = Counter(c.style for c in citations)
        top = styles.most_common(1)[0][0]
        if len(styles) > 1:
            return "mixed"
        return top

    def _find_uncited_claims(self, sentences: List[str], citations: List[InTextCitation]) -> List[CitationFinding]:
        """Sentences with claim indicators but no nearby citation."""
        # Build set of sentence indices that contain citations
        cited_indices: Set[int] = set()
        # Re-split body into sentences for positional matching
        # Simple: check if any citation raw text appears in the sentence
        for i, sent in enumerate(sentences):
            for cit in citations:
                if cit.raw in sent:
                    cited_indices.add(i)

        findings = []
        for i, sent in enumerate(sentences):
            if i in cited_indices:
                continue
            # Check claim indicators
            claim_score = sum(1 for pat in CLAIM_PATTERNS if pat.search(sent))
            if claim_score >= 1:
                risk = "medium" if claim_score == 1 else "high"
                findings.append(CitationFinding(
                    finding_type="uncited_claim",
                    risk_level=risk,
                    detail=f"Claim-like sentence without citation ({claim_score} indicators)",
                    evidence=sent[:120] + ("..." if len(sent) > 120 else ""),
                    recommendation="Add a citation to support this claim.",
                ))

        return findings

    def scan(self, body_text: str, bib_text: str) -> CitationResult:
        """Run full citation cross-check.

        Args:
            body_text: the draft body text (excluding bibliography)
            bib_text: the bibliography / reference list section
        """
        citations = self._detect_in_text(body_text)
        bib_entries = self._parse_bibliography(bib_text)
        style = self._detect_citation_style(citations)
        findings: List[CitationFinding] = []

        if style == "numeric" or style == "mixed":
            # Numeric cross-check
            cited_nums = set()
            for c in citations:
                cited_nums.update(c.numbers)
            bib_nums = {e.number for e in bib_entries if e.number is not None}

            # Cited but not in bib
            for n in sorted(cited_nums - bib_nums):
                findings.append(CitationFinding(
                    finding_type="missing_from_bib",
                    risk_level="high",
                    detail=f"Reference [{n}] cited in text but not found in bibliography",
                    evidence=f"[{n}]",
                    recommendation=f"Add reference [{n}] to the bibliography.",
                ))

            # In bib but not cited
            for n in sorted(bib_nums - cited_nums):
                entry = next((e for e in bib_entries if e.number == n), None)
                desc = f"{entry.first_author} ({entry.year})" if entry else f"[{n}]"
                findings.append(CitationFinding(
                    finding_type="uncited_in_body",
                    risk_level="low",
                    detail=f"Reference [{n}] ({desc}) in bibliography but not cited in text",
                    evidence=desc,
                    recommendation=f"Cite reference [{n}] in the body text or remove from bibliography.",
                ))

        if style == "apa" or style == "mixed" or style == "narrative":
            # APA / author-year cross-check
            bib_keys: Dict[str, BibEntry] = {}
            for e in bib_entries:
                bib_keys[e.key] = e
            bib_key_set = set(bib_keys.keys())

            # A citation is "matched" if ANY of its author keys appears in bib
            unmatched_citations = []
            matched_body_keys: Set[str] = set()
            for c in citations:
                if not c.authors:
                    continue
                c_keys = {_normalise_key(a, c.year) for a in c.authors}
                if c_keys & bib_key_set:
                    matched_body_keys.update(c_keys & bib_key_set)
                else:
                    unmatched_citations.append(c)

            # Cited but not in bib
            for c in unmatched_citations:
                findings.append(CitationFinding(
                    finding_type="missing_from_bib",
                    risk_level="high",
                    detail=f"Citation '{c.raw}' found in text but not in bibliography",
                    evidence=c.raw,
                    recommendation="Add this source to the bibliography or verify the citation.",
                ))

            # In bib but not cited
            for key in sorted(bib_key_set - matched_body_keys):
                entry = bib_keys[key]
                findings.append(CitationFinding(
                    finding_type="uncited_in_body",
                    risk_level="low",
                    detail=f"Bibliography entry '{entry.first_author} ({entry.year})' not cited in text",
                    evidence=entry.raw[:80] + ("..." if len(entry.raw) > 80 else ""),
                    recommendation="Cite this source in the body text or remove from bibliography.",
                ))

        # Uncited claims
        sentences = _split_sentences(body_text)
        claim_findings = self._find_uncited_claims(sentences, citations)
        findings.extend(claim_findings)

        # Stats
        stats = {
            "in_text_count": len(citations),
            "bib_count": len(bib_entries),
            "missing_from_bib": sum(1 for f in findings if f.finding_type == "missing_from_bib"),
            "uncited_in_body": sum(1 for f in findings if f.finding_type == "uncited_in_body"),
            "uncited_claims": sum(1 for f in findings if f.finding_type == "uncited_claim"),
        }

        return CitationResult(
            citation_style=style,
            in_text_citations=citations,
            bib_entries=bib_entries,
            findings=findings,
            stats=stats,
        )


# ── Helpers ─────────────────────────────────────────────────────────

from collections import Counter


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter (production: use spaCy/nltk)."""
    # Split on . ! ? followed by space and uppercase
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\(])', text)
    return [p.strip() for p in parts if p.strip()]
