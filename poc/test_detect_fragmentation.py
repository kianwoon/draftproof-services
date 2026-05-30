"""Detector + cluster for the sentence-fragmentation ('de-AI splitting') evasion.

Pins #2: heavily fragmented AI content (short uniform sentences that starve the
structural signals) must not read as green. The detector is precise (calibrated
so only heavily fragmented text fires) and the fragmented_evasion cluster floors
the score into AMBER only when high content-AI signals co-occur.
"""

from __future__ import annotations

from poc.detect.layer3_scoring import (
    Layer3Input,
    Layer3Scorer,
    estimate_sentence_fragmentation_risk,
)

FRAGMENTED = (
    "The model exists. Students learn fast. Teachers guide them. Knowledge is shared. "
    "Tests measure skill. The system works. Schools adapt slowly. Tools change daily. "
    "Access is common. Quality varies widely. Trust matters most. Judgment is key."
)
NORMAL = (
    "Education systems are changing faster than many schools can comfortably manage, and that "
    "tension shapes everything that follows. In the past, the classroom was the single hub where a "
    "teacher and a printed textbook dictated the pace of learning for everyone. Today a student can "
    "pull a tutorial from one platform, a worked example from another, and feedback from a peer "
    "community within minutes. The abundance is genuinely new, but it creates its own problem: "
    "knowing which of those competing sources actually deserves trust. That is the judgment good "
    "teachers now spend most of their energy cultivating in their students."
)


def test_detector_fires_on_fragmented_text():
    assert estimate_sentence_fragmentation_risk(FRAGMENTED) >= 0.45


def test_detector_silent_on_normal_prose():
    assert estimate_sentence_fragmentation_risk(NORMAL) == 0.0


def test_detector_needs_enough_sentences():
    assert estimate_sentence_fragmentation_risk("Short. Choppy. Text here.") == 0.0


def _score(**signals):
    data = Layer3Input(word_count=620, sentence_count=40, **signals)
    cluster, _boost, name = Layer3Scorer().calculate_ai_likelihood(data)
    return cluster.score, name


# content11-like signal profile: maxed content signals, collapsed structure.
# topk_pattern is the CALIBRATED value the badge path actually feeds (~0.44, not raw ~0.79).
_CONTENT11_SIGNALS = dict(
    predictability=0.44,
    topk_pattern=0.44,
    generic_phrase_density=0.0,
    burstiness_risk=0.25,
    repeated_sentence_structure_risk=0.0,
    generic_assertion_risk=0.90,
    qualifying_text_ai_density=0.0,
    balanced_hedging_risk=0.0,
)


def test_fragmented_evasion_cluster_floors_into_amber():
    score, name = _score(sentence_fragmentation_risk=0.80, **_CONTENT11_SIGNALS)
    assert name == "fragmented_evasion"
    assert score >= 0.42  # floored out of green


def test_identical_signals_without_fragmentation_are_unchanged():
    score, name = _score(sentence_fragmentation_risk=0.0, **_CONTENT11_SIGNALS)
    assert name != "fragmented_evasion"
    assert score < 0.42  # non-fragmented docs keep their prior score (no re-tier)
