"""Semantic shape detector for single-document embedding analysis.

This detector looks at the geometry of the submitted text itself. It does not
need source material. When sentence-transformers is available it uses sentence
embeddings; otherwise it falls back to deterministic hashed lexical vectors so
the scan contract remains stable in constrained environments.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import List

from .base import BaseDetector, DetectResult, Finding

_PRELOADED_EMBEDDER = None
_PRELOADED_MODEL_NAME = ""


def resolve_semantic_embedding_model_name(model_name: str | None = None) -> str:
    requested = model_name or os.environ.get("SEMANTIC_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return str(requested or "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"


@dataclass
class SemanticShapeResult:
    model_name: str
    embedding_model_attached: bool
    sentence_count: int
    paragraph_count: int
    adjacent_similarity_mean: float
    adjacent_similarity_std: float
    paragraph_similarity_mean: float
    paragraph_similarity_std: float
    semantic_uniformity_risk: float
    discourse_regularity_risk: float
    semantic_drift_risk: float


class SemanticShapeDetector(BaseDetector):
    """Detect embedding-level semantic uniformity and discourse regularity."""

    def __init__(self, embedding_model: str | None = None):
        self.embedding_model = resolve_semantic_embedding_model_name(embedding_model)
        self._embedder = None
        self._embedding_available = False
        self._model_name = "hashed-lexical-fallback"

    @property
    def name(self) -> str:
        return "semantic_shape"

    @property
    def detector_version(self) -> str:
        return "0.1.0"

    @property
    def policy_message(self) -> str:
        return (
            "Semantic-shape signals describe document regularity and paragraph "
            "geometry. They are support signals only and should not be treated as "
            "standalone evidence of AI authorship."
        )

    def detect(self, content: str, **kwargs) -> DetectResult:
        sentences = self._split_sentences(content)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", content or "") if p.strip()]
        confidence, confidence_reason = self._assess_confidence(content, **kwargs)

        if len(sentences) < 6:
            return DetectResult(
                scanner=self.name,
                overall_risk=0.0,
                confidence="low",
                confidence_reason="Fewer than 6 sentences; semantic-shape analysis is underpowered.",
                findings=[],
                policy_message=self.policy_message,
                detector_version=self.detector_version,
                raw=SemanticShapeResult(
                    model_name="not_run",
                    embedding_model_attached=False,
                    sentence_count=len(sentences),
                    paragraph_count=len(paragraphs),
                    adjacent_similarity_mean=0.0,
                    adjacent_similarity_std=0.0,
                    paragraph_similarity_mean=0.0,
                    paragraph_similarity_std=0.0,
                    semantic_uniformity_risk=0.0,
                    discourse_regularity_risk=0.0,
                    semantic_drift_risk=0.0,
                ),
            )

        vectors = self._encode(sentences)
        paragraph_vectors = self._encode(paragraphs) if len(paragraphs) >= 2 else []

        adjacent = [
            _cosine(vectors[i], vectors[i + 1])
            for i in range(len(vectors) - 1)
        ]
        paragraph_pairs = []
        for i in range(len(paragraph_vectors)):
            for j in range(i + 1, len(paragraph_vectors)):
                paragraph_pairs.append(_cosine(paragraph_vectors[i], paragraph_vectors[j]))

        adjacent_mean = _mean(adjacent)
        adjacent_std = _std(adjacent)
        paragraph_mean = _mean(paragraph_pairs)
        paragraph_std = _std(paragraph_pairs)

        semantic_uniformity_risk = _clamp(
            _threshold(adjacent_mean, 0.50, 0.78) * 0.45
            + _threshold(paragraph_mean, 0.35, 0.68) * 0.35
            + (1.0 - _threshold(adjacent_std, 0.02, 0.16)) * 0.20
        )
        discourse_regularity_risk = _clamp(
            _threshold(adjacent_mean, 0.45, 0.74) * 0.35
            + (1.0 - _threshold(adjacent_std, 0.03, 0.18)) * 0.35
            + (1.0 - _threshold(paragraph_std, 0.04, 0.20)) * 0.30
        )
        drift_values = [1.0 - value for value in adjacent]
        semantic_drift_risk = _clamp(
            sum(1 for value in drift_values if value >= 0.62) / max(len(drift_values), 1)
        )

        result = SemanticShapeResult(
            model_name=self._model_name,
            embedding_model_attached=self._embedding_available,
            sentence_count=len(sentences),
            paragraph_count=len(paragraphs),
            adjacent_similarity_mean=round(adjacent_mean, 4),
            adjacent_similarity_std=round(adjacent_std, 4),
            paragraph_similarity_mean=round(paragraph_mean, 4),
            paragraph_similarity_std=round(paragraph_std, 4),
            semantic_uniformity_risk=round(semantic_uniformity_risk, 4),
            discourse_regularity_risk=round(discourse_regularity_risk, 4),
            semantic_drift_risk=round(semantic_drift_risk, 4),
        )

        findings = self._findings_for(result, sentences)
        overall_risk = max(
            semantic_uniformity_risk,
            discourse_regularity_risk,
            semantic_drift_risk * 0.65,
        )
        dist = {
            "high": sum(1 for f in findings if f.risk_level == "high"),
            "medium": sum(1 for f in findings if f.risk_level == "medium"),
            "low": sum(1 for f in findings if f.risk_level == "low"),
        }

        return DetectResult(
            scanner=self.name,
            overall_risk=round(overall_risk, 4),
            confidence=confidence,
            confidence_reason=confidence_reason,
            risk_distribution=dist,
            findings=findings,
            policy_message=self.policy_message,
            raw=result,
            detector_version=self.detector_version,
            model_name=self._model_name,
            feature_summary={
                "semantic_uniformity_risk": result.semantic_uniformity_risk,
                "discourse_regularity_risk": result.discourse_regularity_risk,
                "semantic_drift_risk": result.semantic_drift_risk,
                "embedding_model_attached": float(result.embedding_model_attached),
            },
        )

    def _findings_for(self, result: SemanticShapeResult, sentences: List[str]) -> List[Finding]:
        findings = []
        metrics = {
            "semantic_uniformity": (
                result.semantic_uniformity_risk,
                "Semantic uniformity is elevated across adjacent sentences and paragraphs.",
                "Add section-specific reasoning, concrete distinctions, or author observations where the argument feels too evenly shaped.",
            ),
            "discourse_regularity": (
                result.discourse_regularity_risk,
                "Paragraph and sentence transitions are unusually regular.",
                "Review whether each paragraph advances a distinct idea rather than following the same explanatory pattern.",
            ),
            "semantic_drift": (
                result.semantic_drift_risk,
                "Some adjacent sentence jumps show semantic drift.",
                "Check whether transitions need bridging evidence or clearer reasoning continuity.",
            ),
        }
        evidence = " ".join(sentences[:2])[:500]
        for finding_type, (score, detail, recommendation) in metrics.items():
            if score < 0.45:
                continue
            risk = "high" if score >= 0.72 else "medium"
            findings.append(Finding(
                finding_type=finding_type,
                risk_level=risk,
                evidence_strength="moderate" if result.embedding_model_attached else "weak",
                detail=f"{detail} Score: {score:.0%}.",
                evidence=evidence,
                recommendation=recommendation,
                suggested_action_type="review_semantic_shape",
                location={"scope": "document"},
                metadata={
                    "score": score,
                    "semantic_uniformity_risk": result.semantic_uniformity_risk,
                    "discourse_regularity_risk": result.discourse_regularity_risk,
                    "semantic_drift_risk": result.semantic_drift_risk,
                    "adjacent_similarity_mean": result.adjacent_similarity_mean,
                    "adjacent_similarity_std": result.adjacent_similarity_std,
                    "paragraph_similarity_mean": result.paragraph_similarity_mean,
                    "paragraph_similarity_std": result.paragraph_similarity_std,
                    "embedding_model_attached": result.embedding_model_attached,
                    "model_name": result.model_name,
                    "actionability": "review_only",
                },
                signal_category="authorship_risk",
                actionability="review_only",
            ))
        return findings

    def _encode(self, texts: List[str]) -> List[List[float]]:
        if self._ensure_embedder():
            try:
                encoded = self._embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return [list(map(float, row)) for row in encoded]
            except Exception:
                self._embedding_available = False
                self._model_name = "hashed-lexical-fallback"
        return [_hashed_vector(text) for text in texts]

    def _ensure_embedder(self) -> bool:
        global _PRELOADED_EMBEDDER, _PRELOADED_MODEL_NAME
        if _PRELOADED_EMBEDDER is not None and _PRELOADED_MODEL_NAME == self.embedding_model:
            self._embedder = _PRELOADED_EMBEDDER
            self._embedding_available = True
            self._model_name = _PRELOADED_MODEL_NAME
            return True
        if self._embedder is not None:
            return True
        if os.getenv("DRAFTPROOF_DISABLE_EMBEDDINGS", "").lower() in {"1", "true", "yes"}:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            local_files_only = os.getenv("DRAFTPROOF_SEMANTIC_LOCAL_FILES_ONLY", "").lower() in {"1", "true", "yes"}
            self._embedder = SentenceTransformer(
                self.embedding_model,
                cache_folder=os.environ.get("HF_HOME"),
                local_files_only=local_files_only,
            )
            self._embedding_available = True
            self._model_name = self.embedding_model
            return True
        except Exception:
            self._embedder = None
            self._embedding_available = False
            self._model_name = "hashed-lexical-fallback"
            return False

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\(])', text or "")
        return [p.strip() for p in parts if len(p.split()) >= 4]


def _hashed_vector(text: str, dims: int = 128) -> List[float]:
    vector = [0.0] * dims
    words = re.findall(r"\b[a-zA-Z][a-zA-Z-]{2,}\b", (text or "").lower())
    for word in words:
        digest = hashlib.md5(word.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    return _clamp(sum(x * y for x, y in zip(a, b)))


def _mean(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _threshold(value: float, normal: float, high: float) -> float:
    if value <= normal:
        return 0.0
    if value >= high:
        return 1.0
    return (value - normal) / max(high - normal, 0.0001)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
