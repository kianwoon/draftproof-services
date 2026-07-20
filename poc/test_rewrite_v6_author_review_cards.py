"""Guard annotations from the DEFAULT direct rewrite path must reach BOTH user surfaces.

The direct path stores per-paragraph review flags only in ``summary.v6_pass_trace`` rows
(``author_review_items``). Both surfaces render ``summary.author_review_cards`` -- a key that
previously only the unreachable legacy v5 producer set, so the "Author Review Cards" section was
silently empty on every production rewrite. ``production._author_review_cards_from_pass_trace``
composes that key from the existing pass-trace data; this test locks the mapping and verifies the
PDF renderer (``render_rewrite._author_review_card_section``) actually renders it.
"""
from __future__ import annotations

from poc.rewrite_v6.production import _author_review_cards_from_pass_trace
from poc.report.render_rewrite import _author_review_card_section


def _pass_trace_with_items():
    return [
        {
            "pass_index": 0,
            "target_paragraph_id": "p001",
            "selected_source": "direct_llm",
            "status": "accepted",
            "author_review_items": [
                {
                    "added": "a concrete detail was added to ground the claim",
                    "why": "DraftProof supplied an illustrative specific -- replace it with your own real example.",
                },
                {
                    "added": "invented names or places",
                    "why": "Replace them with your own real examples or remove them before submitting.",
                },
            ],
        },
        {
            "pass_index": 1,
            "target_paragraph_id": "p002",
            "selected_source": "source_preserved",
            "status": "source_preserved",
        },
    ]


def test_cards_composed_from_rows_with_items():
    cards = _author_review_cards_from_pass_trace(_pass_trace_with_items())
    assert len(cards) == 2
    first = cards[0]
    # Field-by-field contract consumed by BOTH renderers.
    assert first["instruction"] == "a concrete detail was added to ground the claim"
    assert first["author_task"].startswith("DraftProof supplied an illustrative specific")
    assert first["where"] == "Paragraph p001"
    assert first["provenance"] == "illustrative_content"
    assert first["kind"] == "direct_review_flag"
    assert first["card_id"]  # web list key
    # Nothing fabricated: no candidate-detail / user-input fields we cannot source.
    assert "target_text" not in first
    assert "user_input_needed" not in first


def test_no_cards_when_no_items():
    assert _author_review_cards_from_pass_trace([]) == []
    assert (
        _author_review_cards_from_pass_trace(
            [{"target_paragraph_id": "p001", "status": "source_preserved"}]
        )
        == []
    )
    # Empty / whitespace-only items produce no card.
    assert (
        _author_review_cards_from_pass_trace(
            [{"target_paragraph_id": "p001", "author_review_items": [{"added": "", "why": "  "}]}]
        )
        == []
    )


def test_cards_deduped_and_capped_at_twelve():
    dup = {"added": "same flag", "why": "same why"}
    rows = [
        {
            "target_paragraph_id": f"p{n:03d}",
            "author_review_items": [dup] + [
                {"added": f"flag {n}-{i}", "why": f"why {n}-{i}"} for i in range(10)
            ],
        }
        for n in range(5)
    ]
    cards = _author_review_cards_from_pass_trace(rows)
    assert len(cards) == 12  # renderer cap
    instructions = [c["instruction"] for c in cards]
    assert instructions.count("same flag") == 1  # deduped across rows


def test_pdf_renderer_emits_composed_cards():
    summary = {"author_review_cards": _author_review_cards_from_pass_trace(_pass_trace_with_items())}
    lines = _author_review_card_section(summary)
    assert lines, "PDF section must render when cards exist"
    body = "\n".join(lines)
    assert "Author Review Cards" in body
    assert "a concrete detail was added to ground the claim" in body  # instruction -> title
    assert "replace it with your own real example" in body  # author_task
    assert "illustrative content" in body  # provenance (underscores replaced)


def test_pdf_renderer_empty_without_cards():
    assert _author_review_card_section({}) == []
    assert _author_review_card_section({"author_review_cards": []}) == []
