"""Guard annotations from the DEFAULT direct rewrite path must reach BOTH user surfaces.

The direct path stores per-paragraph review flags only in ``summary.v6_pass_trace`` rows
(``author_review_items``). Both surfaces render ``summary.author_review_cards`` -- a key that
previously only the unreachable legacy v5 producer set, so the "Author Review Cards" section was
silently empty on every production rewrite. ``production._author_review_cards_from_pass_trace``
composes that key from the existing pass-trace data; this test locks the mapping and verifies the
PDF renderer (``render_rewrite._author_review_card_section``) actually renders it.
"""
from __future__ import annotations

import os

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
    cards, total = _author_review_cards_from_pass_trace(_pass_trace_with_items())
    assert total == 2
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
    assert _author_review_cards_from_pass_trace([]) == ([], 0)
    assert (
        _author_review_cards_from_pass_trace(
            [{"target_paragraph_id": "p001", "status": "source_preserved"}]
        )
        == ([], 0)
    )
    # Empty / whitespace-only items produce no card.
    assert (
        _author_review_cards_from_pass_trace(
            [{"target_paragraph_id": "p001", "author_review_items": [{"added": "", "why": "  "}]}]
        )
        == ([], 0)
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
    cards, total = _author_review_cards_from_pass_trace(rows)
    assert len(cards) == 12  # renderer cap (incl. the pinned truncation notice)
    assert total == 51
    assert cards[-1]["kind"] == "review_cards_truncated"
    instructions = [c["instruction"] for c in cards]
    assert instructions.count("same flag") == 1  # deduped across rows


def test_skipped_not_target_card_never_evicts_fabrication_card_at_cap():
    """15 early out-of-scope paragraphs (skipped_not_target) followed by a real fabrication/
    regression review flag must not push the fabrication card past the cap -- skipped_not_target
    cards are always ordered AFTER every other provenance, so they get truncated first."""
    skipped_rows = [
        {
            "target_paragraph_id": f"p{n:03d}",
            "author_review_items": [{
                "added": "no change (outside this rewrite's scope)",
                "why": f"reason {n} -- DraftProof did not touch it.",
                "provenance": "skipped_not_target",
            }],
        }
        for n in range(15)
    ]
    fabrication_row = {
        "target_paragraph_id": "p900",
        "author_review_items": [{
            "added": "invented names or places",
            "why": "Replace them with your own real examples or remove them before submitting.",
        }],
    }
    cards, _total = _author_review_cards_from_pass_trace(skipped_rows + [fabrication_row])
    assert len(cards) <= 12  # cap
    provenances = [c["provenance"] for c in cards]
    assert "illustrative_content" in provenances  # the fabrication card survived the cap
    fabrication_card = next(c for c in cards if c["provenance"] == "illustrative_content")
    assert fabrication_card["instruction"] == "invented names or places"


def test_skipped_cards_aggregate_into_one_non_evictable_summary_card():
    """Incident 5bacaeb3 replay: 70 priority items + 5 skipped paragraphs. The cap used to evict
    EVERY skip notice (byte-identical paragraphs shipped with no explanation) and dropped 84% of
    the guidance with no signal. Now: one aggregated skip summary + one truncation notice, both
    non-evictable, and the whole list still fits the cap so the renderers' second slice is a no-op.
    """
    from poc.rewrite_v6.card_cap import author_review_card_cap

    cap = author_review_card_cap()
    priority_rows = [
        {
            "target_paragraph_id": f"q{n:03d}",
            "author_review_items": [{"added": f"flag {n}", "why": f"why {n}"}],
        }
        for n in range(70)
    ]
    skipped_ids = ["p001", "p003", "p006", "p012", "p013"]
    skipped_rows = [
        {
            "target_paragraph_id": pid,
            "author_review_items": [{
                "added": "no change (outside this rewrite's scope)",
                "why": f"Paragraph {pid} was not flagged as a rewrite target.",
                "provenance": "skipped_not_target",
            }],
        }
        for pid in skipped_ids
    ]
    cards, total = _author_review_cards_from_pass_trace(priority_rows + skipped_rows)

    assert total == 75
    assert len(cards) <= cap
    summaries = [c for c in cards if c["kind"] == "skipped_paragraphs_summary"]
    assert len(summaries) == 1
    truncations = [c for c in cards if c["kind"] == "review_cards_truncated"]
    assert len(truncations) == 1
    for pid in skipped_ids:
        assert pid in summaries[0]["where"]
    assert f"of {total}" in truncations[0]["instruction"]
    # Renderer re-slice at the cap must be a no-op: pinned cards survive it.
    assert cards[: cap] == cards


def test_no_summary_or_truncation_cards_when_not_needed():
    """0 skipped -> no summary card; everything fits -> no truncation card."""
    cards, total = _author_review_cards_from_pass_trace(_pass_trace_with_items())
    assert total == 2
    assert [c["kind"] for c in cards] == ["direct_review_flag", "direct_review_flag"]


def test_pathological_small_cap_lets_pinned_cards_win(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_AUTHOR_REVIEW_CARD_CAP", "2")
    rows = [
        {"target_paragraph_id": f"q{n}", "author_review_items": [{"added": f"f{n}", "why": f"w{n}"}]}
        for n in range(5)
    ] + [
        {
            "target_paragraph_id": "p001",
            "author_review_items": [{
                "added": "no change (outside this rewrite's scope)",
                "why": "Paragraph p001 was not flagged as a rewrite target.",
                "provenance": "skipped_not_target",
            }],
        }
    ]
    # Caller also prepends 2 pinned cards (degradation + regression) -> zero budget left for THIS
    # composer's own cards: cap(2) - reserved_pinned(2) == 0, so the caller's 2 prepends alone
    # already fill the cap and this function must return nothing -- not squeeze in a summary +
    # truncation card on top (that was the bug: 2 reserved + 2 composed == 4 > cap 2).
    cards, total = _author_review_cards_from_pass_trace(rows, reserved_pinned=2)
    assert total == 6
    assert cards == []
    assert len(cards) + 2 <= 2  # cap invariant: reserved_pinned + composed <= cap


def test_boundary_cap_invariant_never_exceeded(monkeypatch):
    """MUST-FIX(a): len(cards) + reserved_pinned <= cap must hold at EVERY cap, including
    pathological ones, so the renderers' second `cards[:cap]` slice is always a no-op and never
    evicts a pinned card. Cross tiny caps (1, 2, 3) and the default (12) with reserved_pinned
    (caller's degradation/regression prepends) of 0 and 2, against a mix of priority + skipped
    cards big enough to overflow every one of those caps. Also checks the documented precedence
    at the tightest caps: skipped-summary beats priority cards, and the truncation notice itself
    yields its slot when nothing else fits."""
    priority_rows = [
        {
            "target_paragraph_id": f"q{n:03d}",
            "author_review_items": [{"added": f"flag {n}", "why": f"why {n}"}],
        }
        for n in range(10)
    ]
    skipped_ids = ["p001", "p003", "p006"]
    skipped_rows = [
        {
            "target_paragraph_id": pid,
            "author_review_items": [{
                "added": "no change (outside this rewrite's scope)",
                "why": f"Paragraph {pid} was not flagged as a rewrite target.",
                "provenance": "skipped_not_target",
            }],
        }
        for pid in skipped_ids
    ]
    rows = priority_rows + skipped_rows

    for cap in (1, 2, 3, 12):
        for reserved_pinned in (0, 2):
            os.environ["DRAFTPROOF_V6_AUTHOR_REVIEW_CARD_CAP"] = str(cap)
            try:
                cards, total = _author_review_cards_from_pass_trace(rows, reserved_pinned=reserved_pinned)
            finally:
                os.environ.pop("DRAFTPROOF_V6_AUTHOR_REVIEW_CARD_CAP", None)

            assert total == 13
            available = cap - reserved_pinned
            if available <= 0:
                # reserved_pinned >= cap is the caller's own invariant violation (it should never
                # prepend more pinned cards than the cap allows) -- this composer's only job in
                # that case is to add nothing on top, which it does.
                assert cards == []
                continue
            # Core invariant: composed cards never push the final list past the cap.
            assert len(cards) + reserved_pinned <= cap, (cap, reserved_pinned, len(cards))
            # The renderers' second cards[:cap] slice must be a no-op.
            assert cards[:cap] == cards
            kinds = [c["kind"] for c in cards]
            if available == 1:
                # Only room for one slot: the truncation notice wins over both the
                # skipped-summary and every priority card (nothing else fits at all).
                assert kinds == ["review_cards_truncated"]
            elif available == 2:
                # Skipped-summary takes precedence over priority cards when tight.
                assert kinds == ["skipped_paragraphs_summary", "review_cards_truncated"]


def test_pdf_renderer_emits_composed_cards():
    summary = {"author_review_cards": _author_review_cards_from_pass_trace(_pass_trace_with_items())[0]}
    lines = _author_review_card_section(summary)
    assert lines, "PDF section must render when cards exist"
    body = "\n".join(lines)
    assert "Author Review Cards" in body
    assert "a concrete detail was added to ground the claim" in body  # instruction -> title
    assert "replace it with your own real example" in body  # author_task
    assert "illustrative content" in body  # provenance (underscores replaced)


def test_pdf_renderer_empty_without_cards():
    assert _author_review_card_section({}) == []


def test_pdf_renderer_emits_where_field():
    """card.get('where') must reach the rendered markdown/PDF section, not just the web view
    (Rewrite.jsx). Regression for the skip-summary card silently dropping which paragraphs
    were left unchanged."""
    skipped_ids = ["p001", "p003", "p006", "p012", "p013"]
    skipped_rows = [
        {
            "target_paragraph_id": pid,
            "author_review_items": [{
                "added": "no change (outside this rewrite's scope)",
                "why": f"Paragraph {pid} was not flagged as a rewrite target.",
                "provenance": "skipped_not_target",
            }],
        }
        for pid in skipped_ids
    ]
    cards = _author_review_cards_from_pass_trace(skipped_rows)[0]
    summary = {"author_review_cards": cards}
    body = "\n".join(_author_review_card_section(summary))
    skip_card = next(c for c in cards if c.get("provenance") == "skipped_not_target")
    assert skip_card.get("where"), "fixture must produce a card with a non-empty where"
    assert f"Where: {skip_card['where']}" in body

    # A card without a `where` field must not emit a stray "Where:" line.
    no_where_summary = {
        "author_review_cards": [
            {"instruction": "no where here", "provenance": "illustrative_content"}
        ]
    }
    no_where_body = "\n".join(_author_review_card_section(no_where_summary))
    assert "Where:" not in no_where_body


def test_pdf_renderer_honours_non_default_cap(monkeypatch):
    """Fable review must-fix: render_rewrite._author_review_card_section used to hardcode
    cards[:12], so raising DRAFTPROOF_V6_AUTHOR_REVIEW_CARD_CAP had no effect on the PDF. It now
    delegates to the same rewrite_v6.card_cap.author_review_card_cap() the producer uses, so both
    a raised and a lowered cap must be reflected in the rendered section."""
    cards = [
        {"card_id": f"card-{n}", "instruction": f"item {n}", "provenance": "illustrative_content"}
        for n in range(20)
    ]
    summary = {"author_review_cards": cards}

    monkeypatch.setenv("DRAFTPROOF_V6_AUTHOR_REVIEW_CARD_CAP", "18")
    lines = _author_review_card_section(summary)
    body = "\n".join(lines)
    assert "item 17" in body  # 18th card (index 17) now included, beyond the old hardcoded 12
    assert "item 18" not in body  # still truncated at the raised cap

    monkeypatch.setenv("DRAFTPROOF_V6_AUTHOR_REVIEW_CARD_CAP", "3")
    lines = _author_review_card_section(summary)
    body = "\n".join(lines)
    assert "item 2" in body
    assert "item 3" not in body  # truncated well below the old hardcoded 12
    assert _author_review_card_section({"author_review_cards": []}) == []
