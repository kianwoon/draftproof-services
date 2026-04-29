"""Cross-draft diff engine — detect surface-level vs meaningful revision.

Compares two drafts and classifies what changed:
- surface rewrite (wording only)
- argument development
- source grounding improvement
- new user reasoning
- grammar-only polish

This is not about catching misconduct. It helps honest users understand
whether their revisions are meaningful or cosmetic.
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class RevisionMetrics:
    surface_change: float          # 0-1, how much wording changed
    structural_change: float       # 0-1, how much structure changed
    new_claims_added: int
    citations_added: int
    user_reasoning_added: int
    grammar_only_changes: int


@dataclass
class RevisionQuality:
    surface_rewrite: float         # proportion of changes that are surface-level
    argument_development: float    # proportion that developed the argument
    source_grounding_improvement: float
    new_user_reasoning: float
    grammar_only: float
    summary: str


@dataclass
class DraftDiff:
    draft_a_word_count: int
    draft_b_word_count: int
    word_overlap: float            # jaccard overlap
    revision_metrics: RevisionMetrics
    revision_quality: RevisionQuality
    draft_evolution_risk: float    # 0-1, high = surface rewrite only
    detail: str


# ── Diff Engine ────────────────────────────────────────────────────

class CrossDraftEngine:
    """Compare two drafts and classify the nature of revisions."""

    def diff(self, text_a: str, text_b: str,
             structural_similarity: Optional[float] = None) -> DraftDiff:

        words_a = self._tokenize(text_a)
        words_b = self._tokenize(text_b)

        # Word-level overlap
        overlap = self._jaccard(words_a, words_b)

        # Sentence-level diff
        sents_a = self._split_sentences(text_a)
        sents_b = self._split_sentences(text_b)

        # New/removed sentences
        new_sents = [s for s in sents_b if not self._has_near_match(s, sents_a)]
        removed_sents = [s for s in sents_a if not self._has_near_match(s, sents_b)]

        # Surface change estimate (word overlap inverse)
        surface_change = 1.0 - overlap

        # Structural change (from fingerprinter if provided)
        if structural_similarity is not None:
            structural_change = 1.0 - structural_similarity
        else:
            # Fallback: paragraph count difference
            paras_a = len(self._split_paragraphs(text_a))
            paras_b = len(self._split_paragraphs(text_b))
            structural_change = abs(paras_a - paras_b) / max(paras_a, paras_b, 1)

        # Count specific changes
        new_claims = sum(1 for s in new_sents if self._is_claim(s))
        citations_added = sum(1 for s in new_sents if self._has_citation(s))
        user_reasoning = sum(1 for s in new_sents if self._is_user_reasoning(s))
        grammar_only = self._count_grammar_changes(sents_a, sents_b)

        metrics = RevisionMetrics(
            surface_change=round(surface_change, 4),
            structural_change=round(structural_change, 4),
            new_claims_added=new_claims,
            citations_added=citations_added,
            user_reasoning_added=user_reasoning,
            grammar_only_changes=grammar_only,
        )

        # Revision quality classification
        total_changes = max(len(new_sents), 1)
        quality = self._classify_revision_quality(metrics, total_changes, new_sents)

        # Draft evolution risk
        risk = self._compute_evolution_risk(metrics, quality, overlap)

        detail = (
            f"word_overlap={overlap:.2f}, surface_change={surface_change:.2f}, "
            f"structural_change={structural_change:.2f}, "
            f"new_sents={len(new_sents)}, removed_sents={len(removed_sents)}, "
            f"new_claims={new_claims}, citations_added={citations_added}, "
            f"user_reasoning={user_reasoning}"
        )

        return DraftDiff(
            draft_a_word_count=len(words_a),
            draft_b_word_count=len(words_b),
            word_overlap=round(overlap, 4),
            revision_metrics=metrics,
            revision_quality=quality,
            draft_evolution_risk=round(risk, 4),
            detail=detail,
        )

    # ── Revision quality classifier ────────────────────────────────

    def _classify_revision_quality(self, metrics: RevisionMetrics,
                                   total_new: int,
                                   new_sents: List[str]) -> RevisionQuality:

        if total_new == 0:
            return RevisionQuality(
                surface_rewrite=1.0,
                argument_development=0.0,
                source_grounding_improvement=0.0,
                new_user_reasoning=0.0,
                grammar_only=float(metrics.grammar_only_changes > 0),
                summary="No new sentences detected. Changes are purely word-level.",
            )

        arg_dev = metrics.new_claims_added / total_new
        source_dev = metrics.citations_added / total_new
        reasoning = metrics.user_reasoning_added / total_new
        grammar = metrics.grammar_only_changes / total_new

        # Surface rewrite is the residual
        surface = max(0.0, 1.0 - arg_dev - source_dev - reasoning - grammar)

        if surface >= 0.60:
            summary = "Most changes are wording-level. The argument did not develop much between drafts."
        elif surface >= 0.40:
            summary = "Moderate surface changes with some argument development."
        elif arg_dev >= 0.30:
            summary = "Notable argument development between drafts."
        elif reasoning >= 0.30:
            summary = "Added personal reasoning and interpretation."
        else:
            summary = "Mixed changes across wording, structure, and argument."

        return RevisionQuality(
            surface_rewrite=round(surface, 4),
            argument_development=round(arg_dev, 4),
            source_grounding_improvement=round(source_dev, 4),
            new_user_reasoning=round(reasoning, 4),
            grammar_only=round(grammar, 4),
            summary=summary,
        )

    # ── Evolution risk ─────────────────────────────────────────────

    def _compute_evolution_risk(self, metrics: RevisionMetrics,
                                quality: RevisionQuality,
                                word_overlap: float) -> float:
        # Edge case: identical drafts — no evolution to assess
        if word_overlap >= 0.999:
            return 0.0

        # Three-scenario model:
        # 1. Same skeleton + same words = no change (low risk)
        # 2. Same skeleton + different words = surface rewrite (high risk)
        # 3. Different skeleton = structural revision (low risk regardless)

        low_structure = 1.0 - metrics.structural_change
        high_overlap = word_overlap
        changed = metrics.surface_change
        surface_proportion = quality.surface_rewrite
        dev_bonus = quality.argument_development + quality.new_user_reasoning
        dev_reduction = min(dev_bonus * 0.3, 0.3)

        # Scenario detection
        is_same_skeleton = low_structure >= 0.70   # structure mostly preserved
        has_wording_changes = changed >= 0.03       # some words changed
        is_surface_rewrite = is_same_skeleton and has_wording_changes

        if is_surface_rewrite:
            # High risk: same skeleton + wording changes = cosmetic revision
            risk = (
                low_structure * 0.35            # skeleton preserved
                + high_overlap * 0.20           # vocabulary preserved
                + min(changed * 3, 0.30)        # amplify small surface changes
                + surface_proportion * 0.15
                - dev_reduction
            )
        else:
            # Low risk: either different structure or no changes
            risk = (
                low_structure * changed * 0.20
                + surface_proportion * 0.10
                - dev_reduction
            )

        return max(0.0, min(1.0, risk))

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))

    @staticmethod
    def _jaccard(set_a: set, set_b: set) -> float:
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in parts if s.strip()]

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    def _has_near_match(self, sentence: str, candidates: List[str],
                        threshold: float = 0.5) -> bool:
        words_s = set(re.findall(r'\b[a-z]{3,}\b', sentence.lower()))
        for c in candidates:
            words_c = set(re.findall(r'\b[a-z]{3,}\b', c.lower()))
            if not words_s or not words_c:
                continue
            overlap = len(words_s & words_c) / len(words_s | words_c)
            if overlap >= threshold:
                return True
        return False

    @staticmethod
    def _is_claim(sentence: str) -> bool:
        lower = sentence.lower()
        claim_markers = [
            "argue", "believe", "contend", "maintain", "assert",
            "propose", "claim", "demonstrate", "show", "prove",
            "suggest", "indicate", "i found", "i observed",
        ]
        return any(m in lower for m in claim_markers)

    @staticmethod
    def _has_citation(sentence: str) -> bool:
        patterns = [
            r'\(\w+,?\s*\d{4}\)',
            r'\(\w+\s+et\s+al\.?',
            r'\[\d+\]',
        ]
        return any(re.search(p, sentence.lower()) for p in patterns)

    @staticmethod
    def _is_user_reasoning(sentence: str) -> bool:
        lower = sentence.lower()
        reasoning_markers = [
            "i think", "in my opinion", "i believe", "from my experience",
            "i disagree", "i would argue", "my view", "personally",
            "what struck me", "it made me realise", "i realised",
            "this suggests to me", "my interpretation",
        ]
        return any(m in lower for m in reasoning_markers)

    def _count_grammar_changes(self, sents_a: List[str],
                               sents_b: List[str]) -> int:
        # Simple heuristic: sentences with high word overlap but different
        # spelling/grammar are grammar-only changes
        count = 0
        for sb in sents_b:
            words_b = set(re.findall(r'\b[a-z]{3,}\b', sb.lower()))
            for sa in sents_a:
                words_a = set(re.findall(r'\b[a-z]{3,}\b', sa.lower()))
                if not words_a or not words_b:
                    continue
                overlap = len(words_a & words_b) / len(words_a | words_b)
                if overlap >= 0.85:  # very high overlap = grammar tweak
                    count += 1
                    break
        return count
