import importlib

from detect.document_structure import structured_paragraph_texts, structured_sentence_segments
from report.report import DraftReport, PredictabilitySummary, Tier, report_to_dict


def test_semantic_shape_imports_document_structure_in_supported_contexts():
    detect_module = importlib.import_module("detect.semantic_shape")
    poc_module = importlib.import_module("poc.detect.semantic_shape")

    assert detect_module.SemanticShapeDetector
    assert poc_module.SemanticShapeDetector


def test_structure_divides_long_single_block_into_virtual_paragraphs():
    sentences = [
        f"Sentence {index} explains one practical step with enough detail for segmentation."
        for index in range(1, 33)
    ]
    text = " ".join(sentences)

    segments = structured_sentence_segments(text)
    paragraph_ids = [segment["paragraph_id"] for segment in segments]
    unique_paragraph_ids = list(dict.fromkeys(paragraph_ids))

    assert len(segments) == 32
    assert len(unique_paragraph_ids) > 1
    assert max(paragraph_ids.count(pid) for pid in unique_paragraph_ids) <= 8
    assert {segment["source_paragraph_id"] for segment in segments} == {"src_p001"}


def test_structure_preserves_explicit_paragraph_boundaries_for_short_blocks():
    text = (
        "First paragraph has one clear sentence. It stays together.\n\n"
        "Second paragraph has another clear sentence. It also stays together."
    )

    segments = structured_sentence_segments(text)

    assert [segment["paragraph_id"] for segment in segments] == ["p001", "p001", "p002", "p002"]
    assert [segment["source_paragraph_id"] for segment in segments] == ["src_p001", "src_p001", "src_p002", "src_p002"]


def test_structure_treats_short_opening_heading_as_separate_block():
    text = (
        "Practical Learning Reflection\n"
        "This opening explains the first idea. It adds enough context for the paragraph."
    )

    paragraphs = structured_paragraph_texts(text)

    assert paragraphs[0] == "Practical Learning Reflection"
    assert paragraphs[1].startswith("This opening explains")


def test_report_backfills_virtual_paragraph_ids_when_predictability_rows_lack_them():
    sentences = [
        f"Sentence {index} explains one practical step with enough detail for report segmentation."
        for index in range(1, 25)
    ]
    text = " ".join(sentences)
    predictability_rows = [
        {
            "sentence_id": f"s{index:03d}",
            "sentence": sentence,
            "risk_label": "low",
            "risk": 0.1,
            "top10_ratio": 0.1,
            "top50_ratio": 0.2,
            "avg_probability": 0.1,
            "avg_surprisal": 1.0,
            "top_predicted_tokens": [],
            "predictable_token_spans": [],
        }
        for index, sentence in enumerate(sentences, start=1)
    ]
    report = DraftReport(
        overall_tier=Tier.LOW,
        finding_count=0,
        findings_by_tier={},
        original_text=text,
        predictability=PredictabilitySummary(
            overall_risk=0.1,
            risk_distribution={},
            sentences=predictability_rows,
            style_shifts=[],
            generic_phrases_found=[],
        ),
    )

    payload = report_to_dict(report)

    assert len({row["paragraph_id"] for row in payload["highlight_segments"]}) > 1
    assert len({row["paragraph_id"] for row in payload["sentence_map"].values()}) > 1
