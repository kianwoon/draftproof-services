from poc.rewrite_v6.production import _rewrite_source_text


def test_v6_rewrite_prefers_scan_normalized_input_text():
    source, meta = _rewrite_source_text({
        "raw_input_text": "First sentence. Second sentence.",
        "input_text": "First sentence.\n\nSecond sentence.",
        "input_text_normalized": True,
    })

    assert source == "First sentence.\n\nSecond sentence."
    assert meta["source"] == "scan_input_text"
    assert meta["uses_scan_normalized_input"] is True
    assert meta["input_text_normalized"] is True
    assert meta["raw_paragraph_count"] == 1
    assert meta["rewrite_paragraph_count"] == 2


def test_v6_rewrite_normalizes_fallback_text_when_scan_input_is_missing():
    text = " ".join(
        f"Sentence {index} explains one practical step with enough detail for segmentation."
        for index in range(1, 25)
    )

    source, meta = _rewrite_source_text({"text": text})

    assert "\n\n" in source
    assert source.replace("\n\n", " ") == text
    assert meta["source"] == "fallback_normalized_original_text"
    assert meta["uses_scan_normalized_input"] is False
    assert meta["normalization_changed_text"] is True
