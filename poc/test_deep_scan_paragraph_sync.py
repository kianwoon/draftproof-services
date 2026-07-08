"""Regression tests for report.report.rebuild_deep_scan_paragraph_rows.

Locks in the 2026-07-09 fix for the deep-scan TABLE vs issue-CARD contradiction:
on a green/14% below-floor document the per-paragraph deep-scan table stamped
"Paragraph 3 — 1 of 3 · 33% Amber" while the issue card flagged 2 sentences in the
same paragraph and rated it LOW. Root cause: the table was built from the bridge's
OWN sentence split (over-segmenting, flagging a different count) and its per-row
band was never gated by the document reliability floor. This rebuilds the table
from the canonical sentence heatmap (the card's source) and clamps each row's band
to the document band.
"""
from report.report import rebuild_deep_scan_paragraph_rows


def _band_fn(p):
    """Replica of pipeline_bridge._deep_scan_band (weights.json 0.3/0.5/0.7),
    injected so the test never imports the heavy bridge/torch stack."""
    if p < 0.30:
        return "insufficient"
    if p < 0.50:
        return "amber"
    if p < 0.70:
        return "orange"
    return "red"


def _heatmap(spec):
    """spec: list of (paragraph_id, n_sentences, n_flagged) -> heatmap rows."""
    rows = []
    for pid, n, flagged in spec:
        for i in range(n):
            rows.append({"paragraph_id": pid, "band": "review" if i < flagged else "clean"})
    return rows


def test_live_report_4701a6e2_p3_matches_card():
    """Title (p001, 1 sent) + 3 body paras; canonical flags p002=1, p003=1,
    p004=2. Doc is below floor (band insufficient). The table's P3 must become
    "2 of 3" (matching the card's 2 flagged) and every row must read insufficient
    (never amber) on the green doc."""
    heatmap = _heatmap([("p001", 1, 0), ("p002", 5, 1), ("p003", 6, 1), ("p004", 3, 2)])
    existing = [
        {"index": 0, "sentence_count": 5, "flagged_count": 1, "band": "insufficient"},
        {"index": 1, "sentence_count": 6, "flagged_count": 1, "band": "insufficient"},
        {"index": 2, "sentence_count": 3, "flagged_count": 1, "band": "amber"},  # the contradiction
    ]
    rows = rebuild_deep_scan_paragraph_rows(existing, heatmap, "insufficient", _band_fn)
    assert rows is not None
    assert [r["flagged_count"] for r in rows] == [1, 1, 2]      # P3 now 2, matches card
    assert [r["sentence_count"] for r in rows] == [5, 6, 3]
    assert all(r["band"] == "insufficient" for r in rows)       # no amber on green doc
    assert [r["index"] for r in rows] == [0, 1, 2]              # body ordinals preserved


def test_row_band_clamped_to_document_band():
    """A noisy 2-of-3 (67%) paragraph would map to 'orange' on its own, but must
    never exceed the document band. Under an 'amber' document it clamps to amber."""
    heatmap = _heatmap([("p001", 4, 0), ("p002", 3, 2)])
    existing = [
        {"index": 0, "sentence_count": 4, "flagged_count": 0, "band": "insufficient"},
        {"index": 1, "sentence_count": 3, "flagged_count": 2, "band": "orange"},
    ]
    rows = rebuild_deep_scan_paragraph_rows(existing, heatmap, "amber", _band_fn)
    assert rows is not None
    assert rows[1]["flagged_count"] == 2
    assert rows[1]["band"] == "amber"  # clamped down from orange to the doc band


def test_row_band_not_raised_when_below_document():
    """Clamp only lowers — a paragraph quieter than the document keeps its own
    (lower) band; the document band is a ceiling, not a floor."""
    heatmap = _heatmap([("p001", 6, 1), ("p002", 6, 0)])
    existing = [
        {"index": 0, "sentence_count": 6, "flagged_count": 1, "band": "insufficient"},
        {"index": 1, "sentence_count": 6, "flagged_count": 0, "band": "insufficient"},
    ]
    rows = rebuild_deep_scan_paragraph_rows(existing, heatmap, "orange", _band_fn)
    assert rows is not None
    assert rows[0]["band"] == "insufficient"  # 1/6 = 17% stays insufficient
    assert rows[1]["band"] == "insufficient"  # 0/6 stays insufficient


def test_alignment_failure_is_fail_open():
    """If canonical groups cannot be matched to the bridge rows by sentence_count,
    return None so the caller keeps the bridge's rows unchanged (never raise)."""
    heatmap = _heatmap([("p002", 4, 1), ("p003", 4, 1)])  # no group of size 9
    existing = [
        {"index": 0, "sentence_count": 9, "flagged_count": 1, "band": "insufficient"},
    ]
    assert rebuild_deep_scan_paragraph_rows(existing, heatmap, "insufficient", _band_fn) is None


def test_empty_inputs_return_none():
    assert rebuild_deep_scan_paragraph_rows([], [{"paragraph_id": "p001", "band": "clean"}],
                                            "insufficient", _band_fn) is None
    assert rebuild_deep_scan_paragraph_rows([{"index": 0, "sentence_count": 3}], [],
                                            "insufficient", _band_fn) is None
