"""Predictability scanner -- token-level analysis using a causal language model.

Core idea: for each token, ask "how likely was this next token given context?"
High predictability across many tokens = formulaic / generic writing.
"""

import logging
import math
import os
import re
import time
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# Use unified phrase source
from poc.predictability.phrase_packs import get_generic_phrases

GENERIC_PHRASES = get_generic_phrases()

SCANNER_VERSION = "vectorized-gpt2-v2"

# Preloaded model cache — set by the worker entrypoint preload AND backfilled by
# the first lazy scanner load, so repeated PredictabilityScanner construction in a
# worker child reuses one in-memory model instead of reloading it from disk.
_PRELOADED_MODEL = None
_PRELOADED_TOKENIZER = None
_PRELOADED_MODEL_NAME = None
_PRELOAD_CACHE_LOCK = threading.RLock()


def _cache_predictability_model(model_name, tokenizer, model) -> None:
    """Populate the process-level model cache once (first writer wins)."""
    global _PRELOADED_MODEL, _PRELOADED_TOKENIZER, _PRELOADED_MODEL_NAME
    with _PRELOAD_CACHE_LOCK:
        if _PRELOADED_MODEL is None:
            _PRELOADED_MODEL = model
            _PRELOADED_TOKENIZER = tokenizer
            _PRELOADED_MODEL_NAME = model_name


_SENTENCE_CACHE_LOCK = threading.RLock()
_SENTENCE_CACHE: "OrderedDict[str, SentenceResult]" = OrderedDict()


def resolve_predictability_model_name(model_name: Optional[str] = None) -> str:
    """Resolve the runtime model name used by preload and scanner instances."""
    requested = model_name or os.environ.get("PREDICTABILITY_MODEL", "gpt2")
    return str(requested or "gpt2").strip() or "gpt2"


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
    source_paragraph_id: str = ""
    virtual_paragraph_id: str = ""


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
        model_name: Optional[str] = None,
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
        model_name = resolve_predictability_model_name(model_name)
        self.model_name = model_name

        # Use preloaded model from worker entrypoint if available
        preloaded_matches = (
            _PRELOADED_MODEL is not None
            and _PRELOADED_TOKENIZER is not None
            and _PRELOADED_MODEL_NAME == model_name
        )
        if preloaded_matches:
            logger.info("Using preloaded %s model from entrypoint cache", model_name)
            self.tokenizer = _PRELOADED_TOKENIZER
            self.model = _PRELOADED_MODEL
        else:
            if _PRELOADED_MODEL is not None and _PRELOADED_MODEL_NAME != model_name:
                logger.warning(
                    "Ignoring preloaded predictability model %s; requested %s",
                    _PRELOADED_MODEL_NAME,
                    model_name,
                )
            logger.info("Loading %s model (no preload cache found) ...", model_name)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
            except OSError:
                logger.warning("Model not in cache, downloading from HuggingFace...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Backfill the process-level cache so the first lazy load warms it for
            # every later scanner in this worker child. Without this, each re-scan
            # (the rewrite pipeline re-scans many times per job) reloads the model
            # from disk -- the repeated "no preload cache found" loads. The explicit
            # entrypoint preload then becomes a pure latency optimization, not a
            # correctness requirement. First load for a given name wins; a different
            # model_name later still loads fresh (single-model is the norm).
            _cache_predictability_model(model_name, self.tokenizer, self.model)

        logger.info(
            "Scanner ready: version=%s model=%s device=%s params=%s",
            SCANNER_VERSION, self.model_name, self.device,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def split_sentences(self, text: str) -> List[str]:
        """Sentence splitter with char offsets and paragraph IDs.

        Uses abbreviation-aware splitting from detect.utils.
        """
        from poc.detect.document_structure import structured_sentence_segments

        sentence_rows = structured_sentence_segments(text)
        result = []
        for row in sentence_rows:
            s = str(row.get("sentence") or "").strip()
            if not s:
                continue
            # Attach offsets as attributes on the string
            s_with_meta = s  # type: ignore
            s_with_meta = type("Str", (str,), {
                "__value__": s,
                "start_char": int(row.get("start_char") or 0),
                "end_char": int(row.get("end_char") or 0),
                "paragraph_id": row.get("paragraph_id") or "p001",
                "source_paragraph_id": row.get("source_paragraph_id") or row.get("paragraph_id") or "src_p001",
                "virtual_paragraph_id": row.get("virtual_paragraph_id") or row.get("paragraph_id") or "p001",
            })(s)
            result.append(s_with_meta)
        return result

    def scan_sentence(self, sentence: str) -> SentenceResult:
        max_tokens = int(os.environ.get("DRAFTPROOF_PREDICTABILITY_MAX_TOKENS", "384"))
        encoded = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )
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

        if input_ids.shape[1] >= max_tokens:
            logger.warning(
                "Predictability sentence truncated to %d tokens: %.120r",
                max_tokens,
                sentence,
            )

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        return self._score_tokenized_sentence(sentence, input_ids[0, 1:], logits[0, :-1, :])

    def _score_tokenized_sentence(
        self,
        sentence: str,
        actual_ids: torch.Tensor,
        shift_logits: torch.Tensor,
    ) -> SentenceResult:
        token_results: List[TokenResult] = []
        # logits[i-1] predicts token at position i. Do the expensive rank/prob
        # work in one vectorized pass. The previous implementation sorted the
        # full GPT-2 vocabulary for every token, which made CPU scans crawl.
        actual_logits = shift_logits.gather(1, actual_ids.unsqueeze(1)).squeeze(1)
        ranks = (shift_logits > actual_logits.unsqueeze(1)).sum(dim=1) + 1
        actual_log_probs = torch.log_softmax(shift_logits, dim=-1).gather(
            1, actual_ids.unsqueeze(1)
        ).squeeze(1)
        actual_probs = torch.exp(actual_log_probs)

        for offset, actual_id in enumerate(actual_ids.tolist()):
            prob = actual_probs[offset].item()
            rank = int(ranks[offset].item())
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

    def scan_sentences_batch(self, sentences: List[str]) -> List[SentenceResult]:
        """Scan multiple sentences per model forward pass.

        This keeps the configured GPT-2 model warm and avoids one expensive CPU forward per
        sentence. Long generated sentences are truncated to a bounded token
        window so a malformed candidate cannot stall the worker for minutes.
        """
        if not sentences:
            return []
        max_tokens = int(os.environ.get("DRAFTPROOF_PREDICTABILITY_MAX_TOKENS", "384"))
        encoded = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        lengths = attention_mask.sum(dim=1).tolist()
        max_seen = max(int(length) for length in lengths) if lengths else 0
        if max_seen >= max_tokens:
            logger.warning(
                "Predictability batch contains sentence(s) truncated to %d tokens",
                max_tokens,
            )

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits

        results: List[SentenceResult] = []
        for row, sentence in enumerate(sentences):
            length = int(lengths[row])
            if length < 2:
                results.append(SentenceResult(
                    sentence=sentence,
                    risk_label="low",
                    predictability_risk=0.0,
                    avg_probability=0.0,
                    avg_surprisal=0.0,
                    top_10_ratio=0.0,
                    top_50_ratio=0.0,
                    matched_generic_phrases=[],
                    error="Sentence too short to score.",
                ))
                continue
            results.append(self._score_tokenized_sentence(
                sentence,
                input_ids[row, 1:length],
                logits[row, : length - 1, :],
            ))
        return results

    def _cache_enabled(self) -> bool:
        value = os.environ.get("DRAFTPROOF_PREDICTABILITY_SENTENCE_CACHE", "1").lower()
        return value not in {"0", "false", "no", "off"}

    @staticmethod
    def _cache_max_entries() -> int:
        try:
            value = int(os.environ.get("DRAFTPROOF_PREDICTABILITY_SENTENCE_CACHE_MAX", "4096"))
        except (TypeError, ValueError):
            value = 4096
        return max(0, value)

    def _cache_signature(self, max_tokens: int) -> str:
        phrase_digest = hashlib.sha256(
            "\n".join(str(phrase) for phrase in self.generic_phrases).encode("utf-8")
        ).hexdigest()[:16]
        weight_items = ",".join(
            f"{key}:{self.weights[key]:.8f}" for key in sorted(self.weights)
        )
        threshold_items = (
            f"high:{self.high_threshold:.8f},"
            f"medium:{self.medium_threshold:.8f},"
            f"review:{self.review_threshold:.8f}"
        )
        return "|".join((
            SCANNER_VERSION,
            self.model_name,
            str(max_tokens),
            weight_items,
            threshold_items,
            phrase_digest,
        ))

    @staticmethod
    def _cache_key(signature: str, sentence: str) -> str:
        return hashlib.sha256(f"{signature}\0{sentence}".encode("utf-8")).hexdigest()

    @staticmethod
    def _clone_sentence_result(
        result: SentenceResult,
        *,
        start_char: int = 0,
        end_char: int = 0,
        paragraph_id: str = "p001",
        source_paragraph_id: str = "src_p001",
        virtual_paragraph_id: str = "p001",
    ) -> SentenceResult:
        return replace(
            result,
            matched_generic_phrases=list(result.matched_generic_phrases),
            token_results=list(result.token_results),
            start_char=start_char,
            end_char=end_char,
            paragraph_id=paragraph_id,
            source_paragraph_id=source_paragraph_id,
            virtual_paragraph_id=virtual_paragraph_id,
        )

    @classmethod
    def clear_sentence_cache(cls) -> None:
        """Clear the process-local predictability sentence cache."""
        with _SENTENCE_CACHE_LOCK:
            _SENTENCE_CACHE.clear()

    def _scan_sentences_batch_cached(
        self,
        sentences: List[str],
        *,
        max_tokens: int,
        cache_enabled: bool,
    ) -> tuple[List[SentenceResult], Dict[str, int]]:
        if not cache_enabled:
            return self.scan_sentences_batch(sentences), {
                "hits": 0,
                "misses": len(sentences),
                "stores": 0,
            }

        signature = self._cache_signature(max_tokens)
        keys = [self._cache_key(signature, sentence) for sentence in sentences]
        results: list[SentenceResult | None] = [None] * len(sentences)
        missing_indexes: list[int] = []
        hits = 0

        with _SENTENCE_CACHE_LOCK:
            for index, key in enumerate(keys):
                cached = _SENTENCE_CACHE.get(key)
                if cached is None:
                    missing_indexes.append(index)
                    continue
                _SENTENCE_CACHE.move_to_end(key)
                results[index] = self._clone_sentence_result(cached)
                hits += 1

        stores = 0
        if missing_indexes:
            missing_sentences = [sentences[index] for index in missing_indexes]
            scanned = self.scan_sentences_batch(missing_sentences)
            max_entries = self._cache_max_entries()
            with _SENTENCE_CACHE_LOCK:
                for index, scanned_result in zip(missing_indexes, scanned):
                    results[index] = self._clone_sentence_result(scanned_result)
                    if max_entries > 0:
                        _SENTENCE_CACHE[keys[index]] = self._clone_sentence_result(scanned_result)
                        _SENTENCE_CACHE.move_to_end(keys[index])
                        stores += 1
                while max_entries > 0 and len(_SENTENCE_CACHE) > max_entries:
                    _SENTENCE_CACHE.popitem(last=False)

        return [result for result in results if result is not None], {
            "hits": hits,
            "misses": len(missing_indexes),
            "stores": stores,
        }

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

    def scan_text(self, text: str, progress_callback=None) -> Dict[str, Any]:
        sentences = self.split_sentences(text)
        eligible = [s for s in sentences if len(str(s).split()) >= 8]
        total = len(eligible)
        batch_size = max(1, int(os.environ.get("DRAFTPROOF_PREDICTABILITY_BATCH_SIZE", "16")))
        max_tokens = int(os.environ.get("DRAFTPROOF_PREDICTABILITY_MAX_TOKENS", "384"))
        cache_enabled = self._cache_enabled()
        cache_hits = 0
        cache_misses = 0
        cache_stores = 0
        logger.info(
            "Predictability scan: %d eligible sentences (of %d total) model=%s version=%s batch_size=%d cache=%s",
            total,
            len(sentences),
            self.model_name,
            SCANNER_VERSION,
            batch_size,
            "on" if cache_enabled else "off",
        )
        if progress_callback:
            progress_callback(10, f"Checking {total} sentence{'' if total == 1 else 's'} for predictability")
        results = []
        t0 = time.monotonic()
        for batch_start in range(0, total, batch_size):
            batch = eligible[batch_start: batch_start + batch_size]
            batch_t0 = time.monotonic()
            batch_results, batch_cache = self._scan_sentences_batch_cached(
                [str(s) for s in batch],
                max_tokens=max_tokens,
                cache_enabled=cache_enabled,
            )
            cache_hits += batch_cache["hits"]
            cache_misses += batch_cache["misses"]
            cache_stores += batch_cache["stores"]
            for offset, sr in enumerate(batch_results):
                s = batch[offset]
                sr.start_char = getattr(s, "start_char", 0)
                sr.end_char = getattr(s, "end_char", 0)
                sr.paragraph_id = getattr(s, "paragraph_id", "p001")
                sr.source_paragraph_id = getattr(s, "source_paragraph_id", "src_p001")
                sr.virtual_paragraph_id = getattr(s, "virtual_paragraph_id", sr.paragraph_id or "p001")
                results.append(sr)
            completed = len(results)
            batch_seconds = time.monotonic() - batch_t0
            logger.info(
                "Predictability batch: %d-%d/%d sentences in %.2fs (cache_hits=%d cache_misses=%d)",
                batch_start + 1,
                completed,
                total,
                batch_seconds,
                batch_cache["hits"],
                batch_cache["misses"],
            )
            if completed % 5 == 0 or completed == total:
                elapsed = time.monotonic() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(
                    "Predictability progress: %d/%d sentences (%.1f/s, %.1fs elapsed)",
                    completed, total, rate, elapsed,
                )
            if progress_callback:
                pct = 10 + int((completed / max(total, 1)) * 85)
                progress_callback(
                    pct,
                    f"Checked {completed}/{total} predictability sentence{'' if total == 1 else 's'}",
                )
        scan_seconds = time.monotonic() - t0
        with _SENTENCE_CACHE_LOCK:
            cache_size = len(_SENTENCE_CACHE)
        if cache_enabled:
            logger.info(
                "Predictability cache: hits=%d misses=%d stores=%d size=%d max=%d",
                cache_hits,
                cache_misses,
                cache_stores,
                cache_size,
                self._cache_max_entries(),
            )
        if progress_callback:
            progress_callback(96, "Reviewing predictability patterns")
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
            "scanner_version": SCANNER_VERSION,
            "model_name": self.model_name,
            "overall_risk": round(overall, 4),
            "scan_seconds": round(scan_seconds, 4),
            "predictability_cache": {
                "enabled": cache_enabled,
                "hits": cache_hits,
                "misses": cache_misses,
                "stores": cache_stores,
                "size": cache_size,
                "max_entries": self._cache_max_entries(),
            },
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
