"""Similarity scanner -- detect source overlap at multiple levels.

Layers:
  1. Exact match -- n-gram fingerprinting
  2. Fuzzy match -- Jaccard + sequence ratio
  3. Semantic similarity -- sentence embeddings
  4. Citation presence -- check if source is cited near the match

Run:  cd poc/similarity && python demo.py
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Set, Tuple

from sentence_transformers import SentenceTransformer, util


# ── N-gram utilities ────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ngrams(text: str, n: int = 3) -> List[str]:
    """Character n-grams from normalized text."""
    t = normalize(text)
    return [t[i : i + n] for i in range(len(t) - n + 1)]


def word_ngrams(text: str, n: int = 3) -> List[str]:
    """Word n-grams from normalized text."""
    words = normalize(text).split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def longest_common_substring_ratio(a: str, b: str) -> float:
    """Fraction of the shorter string covered by the longest common substring."""
    if not a or not b:
        return 0.0
    matcher = SequenceMatcher(None, a.lower(), b.lower())
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    return match.size / min(len(a), len(b))


# ── Citation extraction ────────────────────────────────────────────

CITATION_PATTERNS = [
    # APA / Harvard: (Author, Year)
    re.compile(r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.|and\s+[A-Z][a-z]+))?,\s*\d{4}[a-z]?\)"),
    # Numeric: [1] or [1,2] or [1-3]
    re.compile(r"\[[\d,\s\-]+\]"),
    # IEEE: [1]
    re.compile(r"\[\d+\]"),
]


def extract_citations(text: str) -> List[str]:
    found = []
    for pat in CITATION_PATTERNS:
        found.extend(pat.findall(text))
    return found


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class Match:
    draft_sentence: str
    source_sentence: str
    risk_type: str           # exact_copy | close_paraphrase | patchwriting | semantic_overlap
    risk_level: str          # low | medium | high
    exact_score: float       # n-gram overlap
    fuzzy_score: float       # Jaccard + sequence ratio
    semantic_score: float    # embedding cosine
    citation_nearby: bool
    source_id: Optional[str] = None
    recommendation: str = ""


@dataclass
class SimilarityResult:
    overall_risk: str
    findings: List[Match] = field(default_factory=list)
    risk_distribution: Dict[str, int] = field(default_factory=dict)


# ── Scanner ─────────────────────────────────────────────────────────

class SimilarityScanner:
    """Multi-layer source overlap scanner.

    Usage:
        scanner = SimilarityScanner()
        result = scanner.scan(draft_sentences, source_sentences)
    """

    EXACT_THRESHOLD = 0.70
    CLOSE_THRESHOLD = 0.30
    SEMANTIC_THRESHOLD = 0.75

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)

    def _classify(
        self,
        exact: float,
        fuzzy: float,
        semantic: float,
        has_citation: bool,
    ) -> Tuple[str, str]:
        """Return (risk_type, risk_level) based on scores."""
        if exact >= self.EXACT_THRESHOLD:
            risk_type = "exact_copy"
        elif exact >= self.CLOSE_THRESHOLD or fuzzy >= 0.60:
            risk_type = "close_paraphrase"
        elif fuzzy >= 0.35 and semantic >= self.SEMANTIC_THRESHOLD:
            risk_type = "patchwriting"
        elif semantic >= 0.82 and (exact > 0.10 or fuzzy > 0.05):
            # High semantic similarity with even marginal lexical overlap
            # suggests synonym-swapped paraphrase
            risk_type = "close_paraphrase"
        elif semantic >= self.SEMANTIC_THRESHOLD:
            risk_type = "semantic_overlap"
        else:
            return "none", "low"

        # Citation presence can lower risk level
        if has_citation and risk_type in ("close_paraphrase", "patchwriting"):
            level = "low"
        elif risk_type == "exact_copy" and has_citation:
            level = "medium"   # exact copy is still a concern even with citation
        elif risk_type == "exact_copy":
            level = "high"
        elif risk_type == "close_paraphrase":
            level = "high"
        elif risk_type == "patchwriting":
            level = "medium"
        else:
            level = "low"

        return risk_type, level

    def _recommendation(self, match: Match) -> str:
        if match.risk_type == "exact_copy":
            if match.citation_nearby:
                return "Direct copy detected. Use your own words or format as a block quote with citation."
            return "Uncited direct copy. Rewrite in your own words and add a citation."
        if match.risk_type == "close_paraphrase":
            if match.citation_nearby:
                return "Close paraphrase with citation. Restructure more substantially."
            return "Close paraphrase without citation. Rewrite and cite the source."
        if match.risk_type == "patchwriting":
            return "Mix of copied and original phrasing. Fully rewrite and cite."
        if match.risk_type == "semantic_overlap":
            return "Similar meaning to source. Add your own analysis or interpretation."
        return ""

    def scan(
        self,
        draft_sentences: List[str],
        source_sentences: List[str],
        source_id: Optional[str] = None,
        context_window: int = 1,
    ) -> SimilarityResult:
        """Compare draft sentences against source sentences.

        Args:
            draft_sentences: sentences from the user's draft
            source_sentences: sentences from a source document
            source_id: optional identifier for the source
            context_window: number of adjacent draft sentences to check
                            for nearby citations
        """
        if not draft_sentences or not source_sentences:
            return SimilarityResult(overall_risk="low", risk_distribution={"high": 0, "medium": 0, "low": 0})

        # Precompute source n-gram sets
        source_3grams = [set(word_ngrams(s, 3)) for s in source_sentences]
        source_char_4grams = [set(ngrams(s, 4)) for s in source_sentences]

        # Compute embeddings once
        draft_emb = self.embedder.encode(draft_sentences, convert_to_tensor=True)
        source_emb = self.embedder.encode(source_sentences, convert_to_tensor=True)
        sim_matrix = util.cos_sim(draft_emb, source_emb).cpu().numpy()

        # Citation lookup: which draft sentences have citations
        draft_citations = [extract_citations(s) for s in draft_sentences]

        findings: List[Match] = []

        for i, d_sent in enumerate(draft_sentences):
            d_3grams = set(word_ngrams(d_sent, 3))
            d_char4 = set(ngrams(d_sent, 4))

            # Check citation in this sentence or neighbours
            has_citation = bool(draft_citations[i])
            if not has_citation and context_window > 0:
                for w in range(1, context_window + 1):
                    if i - w >= 0 and draft_citations[i - w]:
                        has_citation = True
                        break
                    if i + w < len(draft_citations) and draft_citations[i + w]:
                        has_citation = True
                        break

            best_match_idx = int(sim_matrix[i].argmax())
            best_semantic = float(sim_matrix[i][best_match_idx])

            # Compute exact + fuzzy against best semantic match
            best_exact = 0.0
            best_fuzzy = 0.0
            best_source = ""

            # Check top-3 semantic candidates for exact/fuzzy
            top_indices = sim_matrix[i].argsort()[-3:][::-1]
            for j in top_indices:
                exact_overlap = jaccard(d_char4, source_char_4grams[j])
                fuzzy_overlap = (
                    0.35 * jaccard(d_3grams, source_3grams[j])
                    + 0.65 * longest_common_substring_ratio(d_sent, source_sentences[j])
                )
                if exact_overlap > best_exact or fuzzy_overlap > best_fuzzy:
                    best_exact = max(best_exact, exact_overlap)
                    best_fuzzy = max(best_fuzzy, fuzzy_overlap)
                    best_source = source_sentences[j]

            risk_type, risk_level = self._classify(
                best_exact, best_fuzzy, best_semantic, has_citation
            )

            if risk_type == "none":
                continue

            m = Match(
                draft_sentence=d_sent,
                source_sentence=best_source,
                risk_type=risk_type,
                risk_level=risk_level,
                exact_score=round(best_exact, 4),
                fuzzy_score=round(best_fuzzy, 4),
                semantic_score=round(best_semantic, 4),
                citation_nearby=has_citation,
                source_id=source_id,
            )
            m.recommendation = self._recommendation(m)
            findings.append(m)

        dist = {
            "high": sum(1 for f in findings if f.risk_level == "high"),
            "medium": sum(1 for f in findings if f.risk_level == "medium"),
            "low": sum(1 for f in findings if f.risk_level == "low"),
        }

        if dist["high"] > 0:
            overall = "high"
        elif dist["medium"] >= 2:
            overall = "high"
        elif dist["medium"] > 0:
            overall = "medium"
        else:
            overall = "low"

        return SimilarityResult(
            overall_risk=overall,
            findings=findings,
            risk_distribution=dist,
        )
