#!/usr/bin/env python3
"""Unit checks for predictability sentence cache behavior."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POC = ROOT / "poc"
for path in (ROOT, POC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from predictability.scanner import PredictabilityScanner, SentenceResult


def assert_test(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def fake_scanner() -> PredictabilityScanner:
    scanner = PredictabilityScanner.__new__(PredictabilityScanner)
    scanner.model_name = "unit-gpt2"
    scanner.high_threshold = 0.55
    scanner.medium_threshold = 0.45
    scanner.review_threshold = 0.35
    scanner.weights = dict(PredictabilityScanner.DEFAULT_WEIGHTS)
    scanner.generic_phrases = ["predictable phrase"]
    scanner.calls = 0

    def scan_sentences_batch(sentences):
        scanner.calls += 1
        return [
            SentenceResult(
                sentence=sentence,
                risk_label="medium",
                predictability_risk=0.5,
                avg_probability=0.1,
                avg_surprisal=2.0,
                top_10_ratio=0.5,
                top_50_ratio=0.8,
                matched_generic_phrases=[],
            )
            for sentence in sentences
        ]

    scanner.scan_sentences_batch = scan_sentences_batch
    return scanner


def main() -> None:
    os.environ["DRAFTPROOF_PREDICTABILITY_SENTENCE_CACHE"] = "1"
    os.environ["DRAFTPROOF_PREDICTABILITY_SENTENCE_CACHE_MAX"] = "16"
    PredictabilityScanner.clear_sentence_cache()

    sentence = "This sentence has enough ordinary words for predictability scoring."
    scanner = fake_scanner()

    first, first_stats = scanner._scan_sentences_batch_cached(
        [sentence],
        max_tokens=384,
        cache_enabled=True,
    )
    second, second_stats = scanner._scan_sentences_batch_cached(
        [sentence],
        max_tokens=384,
        cache_enabled=True,
    )
    assert_test(first_stats["misses"] == 1 and first_stats["hits"] == 0, "first exact sentence scan misses cache")
    assert_test(second_stats["hits"] == 1 and second_stats["misses"] == 0, "second exact sentence scan hits cache")
    assert_test(scanner.calls == 1, "cache hit avoids second batch scorer call")

    first[0].start_char = 999
    third, _ = scanner._scan_sentences_batch_cached(
        [sentence],
        max_tokens=384,
        cache_enabled=True,
    )
    assert_test(third[0].start_char == 0, "cached sentence result is cloned before returning")

    scanner._scan_sentences_batch_cached([sentence], max_tokens=128, cache_enabled=True)
    assert_test(scanner.calls == 2, "cache key changes when max token setting changes")

    other_config = fake_scanner()
    other_config.generic_phrases = ["different scoring phrases"]
    other_config._scan_sentences_batch_cached([sentence], max_tokens=384, cache_enabled=True)
    assert_test(other_config.calls == 1, "cache key changes when scoring phrase config changes")

    no_cache = fake_scanner()
    no_cache._scan_sentences_batch_cached([sentence], max_tokens=384, cache_enabled=False)
    no_cache._scan_sentences_batch_cached([sentence], max_tokens=384, cache_enabled=False)
    assert_test(no_cache.calls == 2, "cache disabled path always calls batch scorer")

    PredictabilityScanner.clear_sentence_cache()


if __name__ == "__main__":
    main()
