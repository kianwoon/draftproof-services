"""Predictability scanner -- token-level analysis using a causal language model.

Core idea: for each token, ask "how likely was this next token given context?"
High predictability across many tokens = formulaic / generic writing.
"""

import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use unified phrase source
from poc.predictability.phrase_packs import get_generic_phrases

GENERIC_PHRASES = get_generic_phrases()


@dataclass
class TokenResult:
    token: str
    probability: float
    rank: int
    surprisal: float
    top_10: bool
    top_50: bool


@dataclass
class SentenceResult:
    sentence: str
    risk_label: str
    predictability_risk: float
    avg_probability: float
    avg_surprisal: float
    top_10_ratio: float
    top_50_ratio: float
    matched_generic_phrases: List[str]
    token_results: List[TokenResult] = field(default_factory=list)
    error: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    paragraph_id: str = ""


class PredictabilityScanner:
    """Scan text for predictability / genericity risk.

    Uses a causal LM to measure how predictable each token is given
    its context. Tokens that are consistently high-probability / low-rank
    suggest formulaic writing.

    Risk score weights (tune with labelled data):
        top_10_ratio:  0.45  -- how many tokens are top-10 predictions
        top_50_ratio:  0.25  -- broader predictability signal
        surprisal:     0.20  -- inverse average surprisal
        generic_phrases: 0.10 -- matches against known filler phrases
    """

    DEFAULT_WEIGHTS = {
        "top_10_ratio": 0.30,
        "top_50_ratio": 0.30,
        "surprisal": 0.25,
        "generic_phrases": 0.15,
    }

    # 4-band thresholds: low < review < medium < high
    # Raised from 0.55/0.45/0.35 — top10_ratio alone should not trigger high risk
    DEFAULT_THRESHOLDS = {
        "high": 0.60,
        "medium": 0.45,
        "review": 0.35,
    }

    def __init__(
        self,
        model_name: str = os.environ.get("PREDICTABILITY_MODEL", "gpt2-medium"),
        custom_phrases: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        high_threshold: float = 0.55,
        medium_threshold: float = 0.45,
        review_threshold: float = 0.35,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.review_threshold = review_threshold
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.generic_phrases = custom_phrases or GENERIC_PHRASES

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def split_sentences(self, text: str) -> List[str]:
        """Sentence splitter with char offsets and paragraph IDs.

        Uses abbreviation-aware splitting from detect.utils.
        """
        from poc.detect.utils import split_sentences as _split
        # Build paragraph map first
        paragraphs = [p for p in text.strip().split("\n\n") if p.strip()]
        para_starts = []
        pos = 0
        for p in paragraphs:
            idx = text.find(p.strip(), pos)
            if idx < 0:
                idx = pos
            para_starts.append(idx)
            pos = idx + len(p)

        sentences = _split(text)
        result = []
        cursor = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            # Find actual position in original text
            start = text.find(s[:40], cursor)
            if start < 0:
                start = cursor
            end = start + len(s)
            cursor = end
            # Determine paragraph_id
            para_id = "p001"
            for pi, ps in enumerate(para_starts):
                if start >= ps:
                    para_id = f"p{pi+1:03d}"
            # Attach offsets as attributes on the string
            s_with_meta = s  # type: ignore
            s_with_meta = type("Str", (str,), {
                "__value__": s,
                "start_char": start,
                "end_char": end,
                "paragraph_id": para_id,
            })(s)
            result.append(s_with_meta)
        return result

    def scan_sentence(self, sentence: str) -> SentenceResult:
        encoded = self.tokenizer(sentence, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)

        if input_ids.shape[1] < 2:
            return SentenceResult(
                sentence=sentence,
                risk_label="low",
                predictability_risk=0.0,
                avg_probability=0.0,
                avg_surprisal=0.0,
                top_10_ratio=0.0,
                top_50_ratio=0.0,
                matched_generic_phrases=[],
                error="Sentence too short to score.",
            )

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        token_results: List[TokenResult] = []

        # logits[i-1] predicts token at position i
        for i in range(1, input_ids.shape[1]):
            actual_id = input_ids[0, i].item()
            probs = torch.softmax(logits[0, i - 1], dim=-1)
            prob = probs[actual_id].item()

            sorted_indices = torch.argsort(probs, descending=True)
            rank = (sorted_indices == actual_id).nonzero(as_tuple=True)[0].item() + 1

            token_results.append(TokenResult(
                token=self.tokenizer.decode([actual_id]),
                probability=prob,
                rank=rank,
                surprisal=-math.log(prob + 1e-12),
                top_10=rank <= 10,
                top_50=rank <= 50,
            ))

        n = len(token_results)
        avg_prob = sum(t.probability for t in token_results) / n
        avg_surp = sum(t.surprisal for t in token_results) / n
        t10 = sum(1 for t in token_results if t.top_10) / n
        t50 = sum(1 for t in token_results if t.top_50) / n

        matched = [p for p in self.generic_phrases if p.lower() in sentence.lower()]
        generic_score = min(len(matched) / 3, 1.0)

        risk = (
            self.weights["top_10_ratio"] * t10
            + self.weights["top_50_ratio"] * t50
            + self.weights["surprisal"] * (1 / (1 + avg_surp))
            + self.weights["generic_phrases"] * generic_score
        )

        # Gate: top10 ratio is a supporting signal, not a primary trigger.
        # "high" requires: score >= 0.60 AND top10 >= 0.70 AND (generic phrases OR score >= 0.70).
        # This prevents common function words in short sentences from triggering high risk.
        # Short sentences (< 12 tokens) get a top10 penalty because common
        # function words dominate and inflate the ratio.
        token_count_penalty = 1.0
        if n < 12:
            token_count_penalty = 0.85

        effective_risk = risk * token_count_penalty

        if (effective_risk >= self.high_threshold
                and t10 >= 0.70
                and (generic_score > 0 or effective_risk >= 0.70)):
            label = "high"
        elif effective_risk >= self.medium_threshold:
            label = "medium"
        elif effective_risk >= self.review_threshold:
            label = "review"
        else:
            label = "low"

        return SentenceResult(
            sentence=sentence,
            risk_label=label,
            predictability_risk=round(risk, 4),
            avg_probability=round(avg_prob, 6),
            avg_surprisal=round(avg_surp, 4),
            top_10_ratio=round(t10, 4),
            top_50_ratio=round(t50, 4),
            matched_generic_phrases=matched,
            token_results=token_results,
        )

    def detect_style_shifts(self, results: List[SentenceResult]) -> List[Dict[str, Any]]:
        """Flag sudden predictability changes between consecutive sentences."""
        shifts = []
        for i in range(1, len(results)):
            diff = results[i].predictability_risk - results[i - 1].predictability_risk
            if abs(diff) > 0.2:
                shifts.append({
                    "from": results[i - 1].sentence[:80],
                    "to": results[i].sentence[:80],
                    "magnitude": round(abs(diff), 4),
                    "direction": "more_predictable" if diff > 0 else "less_predictable",
                })
        return shifts

    def scan_text(self, text: str) -> Dict[str, Any]:
        sentences = self.split_sentences(text)
        results = []
        for s in sentences:
            # Skip short fragments (< 8 words) — initials, author names, URLs, etc.
            if len(str(s).split()) < 8:
                continue
            sr = self.scan_sentence(str(s))
            # Propagate offset metadata from split
            sr.start_char = getattr(s, "start_char", 0)
            sr.end_char = getattr(s, "end_char", 0)
            sr.paragraph_id = getattr(s, "paragraph_id", "p001")
            results.append(sr)
        shifts = self.detect_style_shifts(results)

        valid = [r for r in results if r.error is None]
        overall = sum(r.predictability_risk for r in valid) / len(valid) if valid else 0.0

        # Short-text confidence: predictability is unstable for short samples
        word_count = len(text.split())
        sentence_count = len(valid)
        word_floor = self._thresholds.short_text_word_floor if hasattr(self, '_thresholds') else 250
        sent_floor = self._thresholds.short_text_sentence_floor if hasattr(self, '_thresholds') else 10
        if word_count < word_floor or sentence_count < sent_floor:
            sample_confidence = "low"
            sample_confidence_reason = (
                f"Only {word_count} words / {sentence_count} sentences. "
                "Document-level predictability is unstable."
            )
        else:
            sample_confidence = "adequate"
            sample_confidence_reason = None

        return {
            "overall_risk": round(overall, 4),
            "sample_confidence": sample_confidence,
            "sample_confidence_reason": sample_confidence_reason,
            "risk_distribution": {
                "high": sum(1 for r in valid if r.risk_label == "high"),
                "medium": sum(1 for r in valid if r.risk_label == "medium"),
                "review": sum(1 for r in valid if r.risk_label == "review"),
                "low": sum(1 for r in valid if r.risk_label == "low"),
            },
            "style_shifts": shifts,
            "sentences": results,
        }
