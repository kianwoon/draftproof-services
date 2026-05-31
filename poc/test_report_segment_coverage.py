"""Report segment coverage — the displayed document must include EVERY sentence.

Bug context: the predictability scanner only scores sentences with >= 8 words
(``poc/predictability/scanner.py`` eligibility filter). ``report_to_dict`` built
``highlight_segments`` / ``sentence_map`` / ``document.segments`` solely from the
scored sentences, so short packed-list sentences silently vanished from the
"submitted content" the frontend renders (it rebuilds the document from those
segments). The scan itself ingested the full text; only the display was incomplete.

Fix A (display-only): segments must cover the FULL document. Short, unscored
sentences appear as plain (un-highlighted, no-signal) segments so the rendered
submission reconstructs the original text. Scoring/findings are untouched.
"""

import re

from report.report import DraftReport, PredictabilitySummary, Tier, report_to_dict


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


def _build_report_with_short_sentences():
    """A paragraph mixing long (scored) and short (<8w, filtered) sentences.

    Mirrors production: only the >= 8-word sentences reach
    ``predictability.sentences`` (the scanner drops the rest before scoring).
    """
    long_a = "Education systems change faster than many schools can comfortably manage."   # 10w
    short_1 = "The system relied on textbooks and teachers."                                # 7w
    short_2 = "Tests demonstrated learning."                                                # 3w
    long_b = "The model no longer fully reflects how young people learn today."             # 11w
    short_3 = "The model still exists."                                                     # 4w

    text = " ".join([long_a, short_1, short_2, long_b, short_3])

    # Only the long sentences are eligible for predictability scoring.
    scored = [long_a, long_b]
    predictability_rows = [
        {
            "sentence_id": f"s{index:03d}",
            "sentence": sentence,
            "risk_label": "low",
            "risk": 0.34,
            "top10_ratio": 0.4,
            "top50_ratio": 0.5,
            "avg_probability": 0.2,
            "avg_surprisal": 1.2,
            "top_predicted_tokens": [],
            "predictable_token_spans": [],
        }
        for index, sentence in enumerate(scored, start=1)
    ]

    report = DraftReport(
        overall_tier=Tier.LOW,
        finding_count=0,
        findings_by_tier={},
        original_text=text,
        predictability=PredictabilitySummary(
            overall_risk=0.34,
            risk_distribution={},
            sentences=predictability_rows,
            style_shifts=[],
            generic_phrases_found=[],
        ),
    )
    return report, text, [long_a, short_1, short_2, long_b, short_3]


def test_highlight_segments_cover_every_sentence_including_short_ones():
    report, text, all_sentences = _build_report_with_short_sentences()

    payload = report_to_dict(report)
    segments = payload["highlight_segments"]
    segment_texts = {_norm(seg.get("text", "")) for seg in segments}

    for sentence in all_sentences:
        assert _norm(sentence) in segment_texts, (
            f"Sentence dropped from rendered document: {sentence!r}"
        )


def test_rendered_submission_reconstructs_full_document():
    report, text, _ = _build_report_with_short_sentences()

    payload = report_to_dict(report)
    segments = sorted(
        payload["highlight_segments"], key=lambda s: s.get("start_char", 0)
    )
    rendered = _norm(" ".join(seg.get("text", "") for seg in segments))

    assert rendered == _norm(text)


def test_document_segments_match_highlight_segments_invariant():
    # Existing invariant: highlight_segments == scan_intelligence.document.segments
    report, _, _ = _build_report_with_short_sentences()
    payload = report_to_dict(report)
    assert payload["highlight_segments"] == payload["scan_intelligence"]["document"]["segments"]


def test_unscored_short_sentences_carry_no_signals():
    # Short sentences were never scored -> must render as plain text (no highlight,
    # no signals) so downstream rewrite consumers (which skip signal-less segments)
    # are unaffected.
    report, _, _ = _build_report_with_short_sentences()
    payload = report_to_dict(report)
    by_text = {_norm(s.get("text", "")): s for s in payload["highlight_segments"]}

    short = by_text[_norm("Tests demonstrated learning.")]
    assert not (short.get("signals") or [])
    assert short.get("highlight", {}).get("enabled") in (False, None)


def test_scored_sentence_is_never_dropped_when_text_mismatches_structural_split():
    # Safety net: if a scored sentence's text does not match any structural segment
    # (pathological splitting), it must STILL appear so its highlight/findings survive.
    long_a = "Education systems change faster than many schools can comfortably manage."
    text = long_a + " The model still exists."  # only long_a is >= 8 words

    # A scored row whose text is deliberately absent from the structural split.
    phantom = "This scored sentence text does not appear anywhere in the document body."
    predictability_rows = [
        {
            "sentence_id": "s001",
            "sentence": long_a,
            "risk_label": "low", "risk": 0.3, "top10_ratio": 0.4, "top50_ratio": 0.5,
            "avg_probability": 0.2, "avg_surprisal": 1.2,
            "top_predicted_tokens": [], "predictable_token_spans": [],
        },
        {
            "sentence_id": "s999",
            "sentence": phantom,
            "risk_label": "medium", "risk": 0.6, "top10_ratio": 0.7, "top50_ratio": 0.8,
            "avg_probability": 0.4, "avg_surprisal": 0.9,
            "top_predicted_tokens": [], "predictable_token_spans": [],
            "start_char": 9999, "end_char": 10070,
        },
    ]
    report = DraftReport(
        overall_tier=Tier.LOW, finding_count=0, findings_by_tier={}, original_text=text,
        predictability=PredictabilitySummary(
            overall_risk=0.45, risk_distribution={}, sentences=predictability_rows,
            style_shifts=[], generic_phrases_found=[],
        ),
    )
    payload = report_to_dict(report)
    segment_texts = {_norm(seg.get("text", "")) for seg in payload["highlight_segments"]}
    assert _norm(phantom) in segment_texts, "Scored sentence was silently dropped"
    assert _norm("The model still exists.") in segment_texts  # short one still present


def test_all_short_sentences_document_renders_completely_without_scored_rows():
    # Edge case: a document where EVERY sentence is below the predictability word
    # floor -> predictability.sentences is empty. The rendered document must still
    # contain every sentence and produce well-shaped segments (no crash, the display
    # builder reads segment fields with defaults).
    short_sentences = [
        "Teachers provide instruction.",
        "YouTube offers tutorials.",
        "TikTok shares lessons.",
        "Search engines retrieve information.",
    ]
    text = " ".join(short_sentences)

    report = DraftReport(
        overall_tier=Tier.LOW, finding_count=0, findings_by_tier={}, original_text=text,
        predictability=PredictabilitySummary(
            overall_risk=0.0, risk_distribution={}, sentences=[],
            style_shifts=[], generic_phrases_found=[],
        ),
    )
    payload = report_to_dict(report)
    segments = payload["highlight_segments"]
    segment_texts = {_norm(seg.get("text") or seg.get("sentence") or "") for seg in segments}
    for sentence in short_sentences:
        assert _norm(sentence) in segment_texts, f"Sentence dropped: {sentence!r}"
    # invariant still holds with no scored rows
    assert payload["highlight_segments"] == payload["scan_intelligence"]["document"]["segments"]


def test_scored_sentences_keep_predictability_and_ids():
    # Join must preserve the scored sentence's predictability payload + id so
    # findings still attach.
    report, _, _ = _build_report_with_short_sentences()
    payload = report_to_dict(report)
    by_text = {_norm(s.get("text", "")): s for s in payload["highlight_segments"]}

    long_a = by_text[_norm("Education systems change faster than many schools can comfortably manage.")]
    assert long_a.get("sentence_id")
    assert long_a.get("predictability", {}).get("risk_label") == "low"
