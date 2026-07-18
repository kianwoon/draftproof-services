"""Tests for poc.detect.stylometry.features.

allow-hardcode: the paragraph strings below are hand-authored TEST FIXTURE prose (a
formal/academic paragraph, a casual/conversational paragraph, a 3-paragraph sample
document, and small edge-case snippets) used to exercise extract_fingerprints — they
are test input data, not a matching/scoring list consumed by production code.

Covers: (1) a hand-authored formal/academic vs. casual/conversational paragraph pair
that must differ in the expected direction on several distinct features, (2) correct
paragraph_id assignment/count on a multi-paragraph document, and (3) edge cases (empty
text, a single-sentence paragraph, non-English punctuation-heavy text) that must not
crash.
"""
from __future__ import annotations

import math

from detect.stylometry.features import ParagraphFingerprint, extract_fingerprints

FORMAL_PARAGRAPH = (
    "The empirical analysis, which was conducted across multiple independent "
    "institutions, demonstrates that the observed variance in economic outcomes is "
    "substantially attributable to structural factors rather than to individual "
    "behaviour. However, because the underlying data were aggregated from "
    "heterogeneous sources, several methodological constraints must be acknowledged; "
    "consequently, the findings should be interpreted with appropriate caution. "
    "Furthermore, subsequent research has been undertaken to clarify these "
    "relationships, and the resulting framework has since been adopted by numerous "
    "policy analysts. Therefore, although the initial hypothesis was only partially "
    "confirmed, the broader implications for regulatory strategy remain significant."
)

CASUAL_PARAGRAPH = (
    "So I went to the store yesterday and grabbed some snacks. It was pretty busy but "
    "I didn't mind. My friend messaged me and we hung out later. We just chilled and "
    "watched a movie. It was fun. Later we got food and talked about stuff. Nothing "
    "too crazy happened but it was a good day."
)


def test_formal_and_casual_paragraphs_differ_on_at_least_four_features():
    formal = extract_fingerprints(FORMAL_PARAGRAPH)
    casual = extract_fingerprints(CASUAL_PARAGRAPH)

    assert len(formal) == 1
    assert len(casual) == 1
    formal_fp, casual_fp = formal[0], casual[0]

    # 1. Formal academic prose uses longer words on average.
    assert formal_fp.word_length_mean > casual_fp.word_length_mean

    # 2. Formal prose uses explicit discourse transitions (however, furthermore,
    #    therefore, consequently); the casual paragraph uses none of them.
    assert formal_fp.transition_rate > casual_fp.transition_rate
    assert casual_fp.transition_rate == 0.0

    # 3. Formal prose draws on Coxhead AWL vocabulary; casual prose is everyday words.
    assert formal_fp.academic_vocab_rate > casual_fp.academic_vocab_rate
    assert casual_fp.academic_vocab_rate == 0.0

    # 4. Formal prose has subordinate clauses ("because", "although"); casual does not.
    assert formal_fp.subordination_rate > casual_fp.subordination_rate
    assert casual_fp.subordination_rate == 0.0

    # 5. Formal prose favours passive constructions ("was conducted", "been adopted");
    #    the casual paragraph is written in active voice throughout.
    assert formal_fp.passive_voice_rate > casual_fp.passive_voice_rate
    assert casual_fp.passive_voice_rate == 0.0

    # 6. Formal sentences run much longer than the casual paragraph's short sentences.
    assert formal_fp.sentence_length_mean > casual_fp.sentence_length_mean

    # 7. Longer, denser sentences read as harder (lower Flesch Reading Ease) than the
    #    short, simple casual sentences.
    assert formal_fp.flesch_reading_ease < casual_fp.flesch_reading_ease


def test_extract_fingerprints_multi_paragraph_document_ids_and_count():
    text = (
        "The morning began with a heavy layer of fog across the valley. Visibility "
        "dropped sharply along the main road. Local drivers slowed down to avoid the "
        "reduced sightlines.\n\n"
        "Fresh basil adds a sharp aroma to the sauce once it simmers. A pinch of salt "
        "balances the acidity from the tomatoes. Many home cooks prefer to add the "
        "herbs last.\n\n"
        "The old bridge was rebuilt after the flood damaged its foundations. Engineers "
        "reinforced the support beams with modern materials. Local residents now cross "
        "the river safely every day."
    )

    fingerprints = extract_fingerprints(text)

    assert len(fingerprints) == 3
    assert [fp.paragraph_id for fp in fingerprints] == ["p001", "p002", "p003"]
    assert all(isinstance(fp, ParagraphFingerprint) for fp in fingerprints)
    assert all(fp.word_count > 0 for fp in fingerprints)
    # IDs stay unique and every field round-trips through to_dict().
    assert len({fp.paragraph_id for fp in fingerprints}) == 3
    for fp in fingerprints:
        as_dict = fp.to_dict()
        assert as_dict["paragraph_id"] == fp.paragraph_id
        assert set(as_dict) == {
            "paragraph_id", "sentence_length_mean", "sentence_length_std",
            "word_length_mean", "root_ttr", "punctuation_rates", "transition_rate",
            "function_word_rate", "flesch_reading_ease", "passive_voice_rate",
            "subordination_rate", "lexical_density", "academic_vocab_rate", "word_count",
        }


def test_extract_fingerprints_empty_text_returns_empty_list():
    assert extract_fingerprints("") == []
    assert extract_fingerprints("   \n\n   ") == []


def test_extract_fingerprints_single_sentence_paragraph():
    text = "This single sentence paragraph has exactly one sentence for the edge case test."

    fingerprints = extract_fingerprints(text)

    assert len(fingerprints) == 1
    fp = fingerprints[0]
    assert fp.paragraph_id == "p001"
    # A one-sentence paragraph has zero variance in sentence length by definition.
    assert fp.sentence_length_std == 0.0
    assert fp.sentence_length_mean == fp.word_count
    assert fp.word_count > 0


def test_extract_fingerprints_non_english_punctuation_heavy_text_does_not_crash():
    text = "!!!??? ,,,;;; :::(()) —— 中文测试内容 これはテストです ¿Qué tal? ¡Hola! ~~~###"

    fingerprints = extract_fingerprints(text)

    # Must not raise. Whatever paragraphs are produced must carry finite, sane values.
    assert isinstance(fingerprints, list)
    for fp in fingerprints:
        assert fp.word_count >= 0
        assert math.isfinite(fp.sentence_length_mean)
        assert math.isfinite(fp.sentence_length_std)
        assert math.isfinite(fp.word_length_mean)
        assert math.isfinite(fp.root_ttr)
        assert math.isfinite(fp.flesch_reading_ease)
        assert math.isfinite(fp.passive_voice_rate)
        assert math.isfinite(fp.subordination_rate)
        assert math.isfinite(fp.lexical_density)
        assert math.isfinite(fp.academic_vocab_rate)
        for rate in fp.punctuation_rates.values():
            assert math.isfinite(rate)


def test_extract_fingerprints_pure_function_no_shared_state_across_calls():
    # Calling twice with the same input must yield equal, independent results — a
    # basic sanity check for the "pure function, zero I/O" acceptance criterion.
    first = extract_fingerprints(FORMAL_PARAGRAPH)
    second = extract_fingerprints(FORMAL_PARAGRAPH)

    assert first == second
