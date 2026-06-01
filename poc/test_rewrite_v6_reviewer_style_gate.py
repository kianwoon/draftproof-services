"""The QC reviewer's job is CRAFT (opener/rhythm monoculture, the 25 guidelines), which is
orthogonal to the perplexity-dominated AI-risk score. Gating a style correction on
`_score(doc_after) <= baseline` systematically drops the genuine opener variations (observed:
"In my district…" -> "Across the district…" DROPPED) and keeps only cosmetic ones
("In my" -> "Within my"), so the monoculture survives. Fix: structural/style corrections (those
resolving a deterministic residual-pattern target) are NOT score-gated -- only meaning/grammar
guards apply. And must_fix_unaddressed is computed by RE-DETECTING on the output (did the pattern
actually clear), not by 'was a target sentence touched'."""
from poc.rewrite_v6 import document_reviewer as rev
from poc.rewrite_v6.residual_patterns import ResidualIssue

OPENER = "In my class the morning bell rings early."
TEXT = (OPENER + "\n\nIn my class the overhead lights stay on.\n\n"
        "In my class the wooden desks face front.\n\nThe hallway stays quiet during lunch today.")


def test_is_style_correction_matches_detector_targets():
    targets = [rev._norm("In my class the morning bell rings early.")]
    assert rev._is_style_correction(OPENER, targets) is True
    assert rev._is_style_correction("Some unrelated grounded sentence about data.", targets) is False


def test_correction_is_safe_style_bypasses_score_guard(monkeypatch):
    monkeypatch.setattr(rev, "_score", lambda t: 100.0)  # any candidate "worsens" the score
    args = dict(doc_after="x", baseline=0.0)
    # score-gated (grounding/other): a score rise drops it
    assert rev._correction_is_safe("In my class the bell rings.", "Across the hall the bell rings.", score_gated=True, **args) is False
    # style: kept despite the score rise (grammar/polarity still enforced)
    assert rev._correction_is_safe("In my class the bell rings.", "Across the hall the bell rings.", score_gated=False, **args) is True


def test_review_document_applies_opener_fix_even_when_score_rises(monkeypatch):
    # baseline cheap; ANY change "raises" the AI score -> would be dropped under the old guard
    monkeypatch.setattr(rev, "_score", lambda t: 5.0 if t == TEXT else 99.0)
    monkeypatch.setattr(rev, "detect_residual_patterns",
                        lambda t: [ResidualIssue(rule="opener_monoculture", trick_ids=[19, 2],
                                                 evidence="4/4 open with 'in my'", target_sentences=[OPENER])]
                        if "In my" in t else [])
    monkeypatch.setattr(rev, "_request_corrections",
                        lambda *a, **k: ([{"original": OPENER, "revised": "Across the hall, the morning bell rings early."}], None))
    res = rev.review_document(TEXT, gateway=object())
    assert any(c.original == OPENER and c.revised.startswith("Across") for c in res.corrections)
    assert "Across the hall" in res.text   # the opener was actually varied


def test_review_document_still_score_gates_non_style_corrections(monkeypatch):
    monkeypatch.setattr(rev, "_score", lambda t: 5.0 if t == TEXT else 99.0)
    monkeypatch.setattr(rev, "detect_residual_patterns", lambda t: [])  # no style targets at all
    monkeypatch.setattr(rev, "_request_corrections",
                        lambda *a, **k: ([{"original": OPENER, "revised": "A wholly different rephrased sentence appears now."}], None))
    res = rev.review_document(TEXT, gateway=object())
    assert res.corrections == []   # non-style + score rises -> still dropped


def test_unaddressed_reflects_redetection_not_touch(monkeypatch):
    # detector keeps firing opener_monoculture regardless -> unaddressed must report it even though
    # a (cosmetic) correction touched the target sentence
    monkeypatch.setattr(rev, "_score", lambda t: 5.0)
    monkeypatch.setattr(rev, "detect_residual_patterns",
                        lambda t: [ResidualIssue(rule="opener_monoculture", trick_ids=[19, 2],
                                                 evidence="persists", target_sentences=[OPENER])])
    monkeypatch.setattr(rev, "_request_corrections",
                        lambda *a, **k: ([{"original": OPENER, "revised": "In my class the morning bell rings loudly."}], None))
    res = rev.review_document(TEXT, gateway=object())
    assert "opener_monoculture" in res.must_fix_unaddressed   # honest: pattern still detected on output
