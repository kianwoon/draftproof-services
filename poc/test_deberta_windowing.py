# poc/test_deberta_windowing.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from detect.deberta_windowing import split_sentences, build_windows, aggregate


def test_split_sentences_basic():
    assert split_sentences("Hello world. This is two. And a third!") == [
        "Hello world.", "This is two.", "And a third!"
    ]


def test_build_windows_overlap_step_two():
    sents = ["s1.", "s2.", "s3.", "s4."]
    windows = build_windows(sents, size=2, step=1)
    assert windows == ["s1. s2.", "s2. s3.", "s3. s4."]


def test_build_windows_short_doc_single_window():
    sents = ["only one sentence."]
    assert build_windows(sents, size=3, step=1) == ["only one sentence."]


def test_aggregate_weighted_mean_by_sentence():
    # sentence 0 covered by window-prob 0.9; sentence 1 by 0.9 and 0.3 -> mean 0.6
    sents = ["a.", "b."]
    windows = build_windows(sents, size=1, step=1)
    probs = [0.9, 0.3]
    agg = aggregate(sents, windows, probs)
    assert abs(agg["document_score"] - 0.6) < 1e-6
    assert len(agg["sentence_scores"]) == 2


if __name__ == "__main__":
    test_split_sentences_basic()
    test_build_windows_overlap_step_two()
    test_build_windows_short_doc_single_window()
    test_aggregate_weighted_mean_by_sentence()
    print("ALL TESTS PASSED")
