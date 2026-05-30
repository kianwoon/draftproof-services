"""The authorship-evidence builder re-presents ALREADY-MEASURED signals. It must:
(C1) treat signals as risk-oriented -- LOW risk = human-present marker, HIGH = thin -> action;
(C2) exclude the perplexity trio (predictability/surprisal_risk/topk_pattern_risk);
and never surface a verdict/score. Content-agnostic (no domain vocab dependence)."""

from poc.report.authorship_evidence import (
    build_authorship_evidence,
    preserved_idea_spans,
    paragraph_authorship_targets,
)


def _signals(**kw):
    base = {"genericity": 0.0, "specificity": 0.69, "source_grounding": 1.0,
            "draft_evolution": 0.15, "structural_reuse": 0.11, "burstiness": 0.35,
            "predictability": 0.44, "surprisal_risk": 0.11, "topk_pattern_risk": 0.79}
    base.update(kw)
    return base


def test_low_risk_signal_becomes_present_marker():
    block = build_authorship_evidence(_signals())
    present = {m["signal"] for m in block["present_markers"]}
    assert "genericity" in present          # risk 0.0 -> present
    assert "structural_reuse" in present     # risk 0.11 -> present


def test_high_risk_signal_becomes_thin_action():
    block = build_authorship_evidence(_signals())
    thin = {t["signal"] for t in block["thin_signals"]}
    assert "specificity" in thin             # risk 0.69 -> thin
    assert "source_grounding" in thin        # risk 1.0 -> thin
    assert all("action" in t for t in block["thin_signals"])


def test_perplexity_trio_excluded():
    block = build_authorship_evidence(_signals())
    seen = {m["signal"] for m in block["present_markers"] + block["mixed_markers"] + block["thin_signals"]}
    assert not ({"predictability", "surprisal_risk", "topk_pattern_risk"} & seen)


def test_no_verdict_or_score_surfaced():
    block = build_authorship_evidence(_signals())
    assert "score" not in block and "concern_tier" not in block
    assert isinstance(block["summary_line"], str) and block["summary_line"]


def test_empty_false_positives_degrades_gracefully():
    block = build_authorship_evidence(_signals(), false_positives=None)
    assert block["human_recognized_spans"] == []


def test_false_positives_become_human_recognized_spans():
    fps = [{"sentence_id": "s012", "sentence": "In other words, education today...", "reason": "downgraded high->low"}]
    block = build_authorship_evidence(_signals(), false_positives=fps)
    assert block["human_recognized_spans"][0]["text"].startswith("In other words")


def test_preserved_idea_spans_are_verbatim_survivors():
    original = "I taught in one room for years. Technology changes everything rapidly today. Students trusted us."
    final = "I taught in one room for years. Tech now reshapes the classroom in concrete ways. Students trusted us."
    kept = [p["text"] for p in preserved_idea_spans(original, final)]
    assert "I taught in one room for years." in kept
    assert "Students trusted us." in kept
    assert "Technology changes everything rapidly today." not in kept  # rewritten -> not preserved


def test_paragraph_targets_protect_matching_spans_and_carry_actions():
    fps = [{"sentence_id": "s1", "sentence": "I taught in one room for years.", "reason": "human"}]
    block = build_authorship_evidence(_signals(), false_positives=fps)
    targets = paragraph_authorship_targets(block, "I taught in one room for years. More generic claims here.")
    assert "I taught in one room for years." in targets["protected_spans"]
    assert any("source" in g.lower() or "concrete" in g.lower() for g in targets["grounding_targets"])
