"""Structural fingerprinter — detect same argument skeleton despite wording changes.

For each paragraph, extracts a fingerprint capturing topic, function,
citation presence, transition type, and sentence structure. Then compares
fingerprints across drafts to detect surface-level rewrites.
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ── Paragraph function labels ──────────────────────────────────────

FUNCTION_KEYWORDS = {
    "claim": [
        "i argue", "i believe", "i contend", "this essay",
        "my position", "i maintain", "i assert", "i propose",
        "the evidence suggests", "it is clear that", "this demonstrates",
        "i found that", "i discovered", "i realised", "i observed",
        "my students", "i decided", "i employed", "i made",
    ],
    "evidence": [
        "for example", "for instance", "according to", "cited",
        "data shows", "research shows", "studies show", "survey",
        "statistics", "findings", "results indicate", "as shown",
        "as reported", "evidence suggests",
    ],
    "example": [
        "such as", "including", "like", "for example", "for instance",
        "one student", "in one case", "specifically", "in particular",
        "to illustrate", "a case in point",
    ],
    "limitation": [
        "however", "but", "although", "despite", "limitation",
        "challenge", "problem", "issue", "concern", "drawback",
        "not everyone", "still a factor", "it made me think",
        "i got to admit", "expert blind spot",
    ],
    "transition": [
        "therefore", "thus", "consequently", "as a result",
        "in addition", "furthermore", "moreover", "similarly",
        "in contrast", "on the other hand", "another major change",
        "another key", "moving on",
    ],
    "conclusion": [
        "in conclusion", "to summarize", "in summary", "overall",
        "in the end", "ultimately", "this whole process showed",
        "this whole lesson", "to be honest", "this showed me",
        "without that", "finally",
    ],
    "intro": [
        "writing this", "creating this", "this essay", "this report",
        "this paper", "in this", "the purpose", "this lesson plan",
        "this reflection",
    ],
}

# ── Transition type labels ─────────────────────────────────────────

TRANSITION_TYPES = {
    "contrast": [
        "however", "but", "although", "despite", "on the other hand",
        "in contrast", "nevertheless", "yet", "still",
    ],
    "continuation": [
        "also", "furthermore", "moreover", "in addition", "additionally",
        "and", "similarly", "likewise", "another",
    ],
    "conclusion": [
        "therefore", "thus", "consequently", "as a result", "so",
        "in conclusion", "in the end", "overall",
    ],
    "cause_effect": [
        "because", "since", "due to", "as a result", "therefore",
        "so that", "in order to", "leads to", "resulted in",
    ],
    "temporal": [
        "initially", "then", "next", "after", "before", "during",
        "finally", "at this stage", "based on my observation",
    ],
    "none": [],
}


# ── Data classes ───────────────────────────────────────────────────

@dataclass
class ParagraphFingerprint:
    paragraph_id: str
    index: int
    topic_label: str
    function_label: str
    claim_type: str
    citation_present: bool
    transition_type: str
    sentence_count: int
    avg_sentence_length: float
    word_count: int
    keyword_vector: Dict[str, int] = field(default_factory=dict)
    first_sentence: str = ""
    raw_text: str = ""


@dataclass
class DraftFingerprint:
    paragraphs: List[ParagraphFingerprint]
    topic_sequence: List[str]
    function_sequence: List[str]
    transition_sequence: List[str]
    citation_positions: List[bool]
    total_word_count: int


@dataclass
class StructuralComparison:
    topic_sequence_similarity: float
    function_sequence_similarity: float
    claim_order_similarity: float
    transition_pattern_similarity: float
    citation_position_similarity: float
    paragraph_count_match: bool
    structural_reuse_risk: float
    detail: str


# ── Fingerprinter ──────────────────────────────────────────────────

class StructuralFingerprinter:
    """Extract structural fingerprints and compare across drafts."""

    def fingerprint(self, text: str) -> DraftFingerprint:
        paragraphs = self._split_paragraphs(text)
        fingerprints = []

        for i, para in enumerate(paragraphs):
            fp = self._fingerprint_paragraph(para, i)
            fingerprints.append(fp)

        return DraftFingerprint(
            paragraphs=fingerprints,
            topic_sequence=[fp.topic_label for fp in fingerprints],
            function_sequence=[fp.function_label for fp in fingerprints],
            transition_sequence=[fp.transition_type for fp in fingerprints],
            citation_positions=[fp.citation_present for fp in fingerprints],
            total_word_count=sum(fp.word_count for fp in fingerprints),
        )

    def compare(self, fp_a: DraftFingerprint, fp_b: DraftFingerprint) -> StructuralComparison:
        topic_sim = self._sequence_similarity(fp_a.topic_sequence, fp_b.topic_sequence)
        func_sim = self._sequence_similarity(fp_a.function_sequence, fp_b.function_sequence)
        claim_sim = self._claim_order_similarity(fp_a.paragraphs, fp_b.paragraphs)
        trans_sim = self._sequence_similarity(fp_a.transition_sequence, fp_b.transition_sequence)
        cite_sim = self._bool_sequence_similarity(fp_a.citation_positions, fp_b.citation_positions)

        count_match = len(fp_a.paragraphs) == len(fp_b.paragraphs)

        # Weighted structural reuse risk
        risk = (
            topic_sim * 0.30
            + func_sim * 0.30
            + claim_sim * 0.15
            + trans_sim * 0.15
            + cite_sim * 0.10
        )

        # Boost if paragraph counts match (same skeleton)
        if count_match and risk >= 0.50:
            risk = min(1.0, risk + 0.05)

        detail = (
            f"topic_sim={topic_sim:.2f}, func_sim={func_sim:.2f}, "
            f"claim_sim={claim_sim:.2f}, trans_sim={trans_sim:.2f}, "
            f"cite_sim={cite_sim:.2f}, count_match={count_match}"
        )

        return StructuralComparison(
            topic_sequence_similarity=round(topic_sim, 4),
            function_sequence_similarity=round(func_sim, 4),
            claim_order_similarity=round(claim_sim, 4),
            transition_pattern_similarity=round(trans_sim, 4),
            citation_position_similarity=round(cite_sim, 4),
            paragraph_count_match=count_match,
            structural_reuse_risk=round(risk, 4),
            detail=detail,
        )

    # ── Paragraph-level fingerprinting ─────────────────────────────

    def _fingerprint_paragraph(self, para: str, index: int) -> ParagraphFingerprint:
        sentences = self._split_sentences(para)
        word_count = len(para.split())
        sent_lengths = [len(s.split()) for s in sentences] if sentences else [0]
        avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0

        lower = para.lower()

        function = self._label_function(lower)
        transition = self._label_transition(lower)
        topic = self._label_topic(lower)
        claim_type = self._label_claim_type(lower, function)
        citation = self._detect_citation(lower)

        kw_vector = self._keyword_vector(lower)

        return ParagraphFingerprint(
            paragraph_id=f"p{index:02d}",
            index=index,
            topic_label=topic,
            function_label=function,
            claim_type=claim_type,
            citation_present=citation,
            transition_type=transition,
            sentence_count=len(sentences),
            avg_sentence_length=round(avg_sent_len, 1),
            word_count=word_count,
            keyword_vector=kw_vector,
            first_sentence=sentences[0] if sentences else "",
            raw_text=para,
        )

    def _label_function(self, lower_text: str) -> str:
        scores = {}
        for func, keywords in FUNCTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower_text)
            scores[func] = score
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def _label_transition(self, lower_text: str) -> str:
        for ttype, keywords in TRANSITION_TYPES.items():
            if ttype == "none":
                continue
            if any(kw in lower_text for kw in keywords):
                return ttype
        return "none"

    def _label_topic(self, lower_text: str) -> str:
        topic_keywords = {
            "pedagogy": ["andragogy", "behaviourist", "behaviourism", "teaching",
                         "learning", "students", "classroom", "lesson", "solo taxonomy"],
            "method": ["octagon method", "dodecagon", "projection", "uniform layer",
                       "haircut", "drill and practice", "micro-task", "clock face"],
            "assessment": ["assess", "mastered", "uni-structural", "multi-structural",
                           "relational", "solo", "taxonomy", "outcome", "understanding"],
            "differentiation": ["high-need", "adhd", "asd", "proximal guidance",
                                "team-teaching", "hand-eye coordination", "differentiat"],
            "reflection": ["reflect", "expert blind spot", "realised", "showed me",
                            "made me think", "end game", "half the battle"],
            "technology": ["ai", "artificial intelligence", "machine learning",
                           "algorithm", "digital", "automation", "data-driven"],
            "background": ["in recent years", "has transformed", "increasingly",
                            "growing", "widely", "literature", "research"],
        }
        scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for kw in keywords if kw in lower_text)
            scores[topic] = score
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def _label_claim_type(self, lower_text: str, function: str) -> str:
        if any(kw in lower_text for kw in ["according to", "cited", "research shows", "study"]):
            return "cited_claim"
        if function in ("claim", "intro", "conclusion"):
            return "general_claim"
        if any(kw in lower_text for kw in ["i found", "i observed", "i realised",
                                            "based on my", "in my experience"]):
            return "personal_interpretation"
        return "general_claim"

    def _detect_citation(self, lower_text: str) -> bool:
        # APA, Harvard, numeric, footnote-style
        patterns = [
            r'\(\w+,?\s*\d{4}\)',        # (Author, 2024)
            r'\(\w+\s+et\s+al\.?',       # (Author et al.)
            r'\[\d+\]',                   # [1]
            r'cf\.',                      # cf.
            r'ibid\.',                    # ibid.
            r'op\.?\s*cit\.',            # op. cit.
        ]
        return any(re.search(p, lower_text) for p in patterns)

    def _keyword_vector(self, lower_text: str) -> Dict[str, int]:
        """Simple keyword frequency vector for cosine similarity fallback."""
        words = re.findall(r'\b[a-z]{4,}\b', lower_text)
        vec = {}
        for w in words:
            vec[w] = vec.get(w, 0) + 1
        return vec

    # ── Similarity metrics ─────────────────────────────────────────

    def _sequence_similarity(self, seq_a: List[str], seq_b: List[str]) -> float:
        """Position-aware sequence similarity."""
        if not seq_a or not seq_b:
            return 0.0

        min_len = min(len(seq_a), len(seq_b))
        max_len = max(len(seq_a), len(seq_b))

        matches = 0
        for i in range(min_len):
            if seq_a[i] == seq_b[i]:
                matches += 1

        positional = matches / max_len

        # Jaccard similarity for order-independent overlap
        set_a = set(seq_a)
        set_b = set(seq_b)
        jaccard = len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else 0.0

        return 0.6 * positional + 0.4 * jaccard

    def _claim_order_similarity(self, paras_a: List[ParagraphFingerprint],
                                paras_b: List[ParagraphFingerprint]) -> float:
        """Compare claim type sequences."""
        seq_a = [p.claim_type for p in paras_a]
        seq_b = [p.claim_type for p in paras_b]
        return self._sequence_similarity(seq_a, seq_b)

    def _bool_sequence_similarity(self, seq_a: List[bool],
                                  seq_b: List[bool]) -> float:
        """Compare boolean sequences (e.g., citation positions)."""
        if not seq_a or not seq_b:
            return 0.0
        min_len = min(len(seq_a), len(seq_b))
        matches = sum(1 for i in range(min_len) if seq_a[i] == seq_b[i])
        return matches / max(len(seq_a), len(seq_b))

    # ── Splitting helpers ──────────────────────────────────────────

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in parts if s.strip()]
