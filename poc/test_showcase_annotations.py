"""Unit tests for the showcase annotation layer.

ISOLATION: this layer (report.showcase_annotations) is additive — it reads original->rewritten
pairs and emits TEACHING notes. It imports nothing from the rewrite pipeline and is imported by
nothing in production, so these tests touch no working code path.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from report.showcase_annotations import annotate_change, annotate_comparison  # noqa: E402


def _labels(original, rewritten):
    return [t[0] for t in annotate_change(original, rewritten)["techniques"]]


def test_detects_concrete_anchor():
    labels = _labels(
        "Many schools focus on grades.",
        "At Box Hill Institute, teachers tie a student's worth to the 70% pass mark.",
    )
    assert "Grounded with a concrete anchor" in labels


def test_detects_first_person_lived_experience():
    labels = _labels(
        "Education systems change fast.",
        "In my district, I have seen standards change a semester before lessons catch up.",
    )
    assert "Anchored in lived experience" in labels


def test_detects_hedge_cut():
    labels = _labels(
        "Many schools struggle with timetabling.",
        "The three schools on my corridor struggle with the 9am slot.",
    )
    assert "Cut the vague hedge" in labels


def test_falls_back_to_specific_when_no_strong_signal():
    labels = _labels("The system is broken.", "The grading system is broken.")
    assert "Made the claim more specific" in labels


def test_your_turn_prompt_is_present_and_points_at_the_user():
    a = annotate_change("AI is useful.", "In my class, the AI tutor helped 12 students draft faster.")
    assert "your_turn" in a
    assert "your own" in a["your_turn"].lower() or "your real" in a["your_turn"].lower()


def test_annotate_comparison_skips_unchanged_and_empty():
    pairs = [
        {"original": "Same.", "rewritten": "Same.", "changed": "true"},          # identical -> skip
        {"original": "X.", "rewritten": "Y.", "changed": "false"},               # changed=false -> skip
        {"original": "Schools struggle.",
         "rewritten": "In my school, we lose the 9am slot to assembly.", "changed": "true"},  # keep
    ]
    out = annotate_comparison(pairs)
    assert len(out) == 1
    assert out[0]["original"] == "Schools struggle."
    assert out[0]["techniques"]


def test_annotate_comparison_accepts_tuples():
    out = annotate_comparison([("Generic claim.", "In my classroom, I saw 30 students do X.", True)])
    assert len(out) == 1 and out[0]["techniques"]


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("ok")
