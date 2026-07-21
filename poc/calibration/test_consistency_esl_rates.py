"""Unit tests for the pure aggregation math in consistency_esl_rates.py.

The corpus run itself is a manual local tool (SCoCESLE is local-only); these
tests cover only the group-summary and disparity-gap arithmetic with synthetic
counts, so they run in CI without the corpus.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.consistency_esl_rates import flag_rate_gap_pts, summarize_group


def test_summarize_group_rates():
    # 4 essays: 2 eligible-and-flagged, 1 eligible-unflagged, 1 ineligible (short doc).
    s = summarize_group(outlier_counts=[2, 1, 0, 0], eligibles=[True, True, True, False])
    assert s["n_essays"] == 4
    assert s["n_eligible"] == 3
    assert s["eligibility_rate_pct"] == 75.0
    assert s["flagged_essay_rate_pct"] == round(2 / 3 * 100, 2)
    assert s["mean_outliers_per_eligible_essay"] == round(3 / 3, 3)


def test_summarize_group_no_eligible():
    s = summarize_group(outlier_counts=[0, 0], eligibles=[False, False])
    assert s["n_eligible"] == 0
    assert s["flagged_essay_rate_pct"] is None


def test_flag_rate_gap_lower_minus_higher():
    groups = {
        "lower proficiency": {"flagged_essay_rate_pct": 22.0},
        "higher proficiency": {"flagged_essay_rate_pct": 15.5},
    }
    assert flag_rate_gap_pts(groups) == 6.5


def test_flag_rate_gap_none_when_a_group_missing_rate():
    groups = {
        "lower proficiency": {"flagged_essay_rate_pct": None},
        "higher proficiency": {"flagged_essay_rate_pct": 10.0},
    }
    assert flag_rate_gap_pts(groups) is None


def test_groups_measurable_floor():
    from calibration.consistency_esl_rates import MIN_ELIGIBLE_PER_GROUP, groups_measurable

    ok = {
        "lower proficiency": {"n_eligible": MIN_ELIGIBLE_PER_GROUP},
        "higher proficiency": {"n_eligible": MIN_ELIGIBLE_PER_GROUP + 5},
    }
    tiny = {
        "lower proficiency": {"n_eligible": 3},
        "higher proficiency": {"n_eligible": MIN_ELIGIBLE_PER_GROUP + 5},
    }
    assert groups_measurable(ok) is True
    assert groups_measurable(tiny) is False


def test_compute_verdict_unmeasurable_when_gap_none():
    from calibration.consistency_esl_rates import compute_verdict

    assert compute_verdict(gap=None, measurable=False) == "UNMEASURABLE"


def test_compute_verdict_unmeasurable_when_groups_not_measurable():
    from calibration.consistency_esl_rates import compute_verdict

    # gap present but a group is below MIN_ELIGIBLE_PER_GROUP -> still UNMEASURABLE.
    assert compute_verdict(gap=-33.33, measurable=False) == "UNMEASURABLE"


def test_compute_verdict_pass_within_threshold():
    from calibration.consistency_esl_rates import MAX_FLAG_RATE_GAP_PTS, compute_verdict

    assert compute_verdict(gap=MAX_FLAG_RATE_GAP_PTS, measurable=True) == "PASS"
    assert compute_verdict(gap=-33.33, measurable=True) == "PASS"


def test_compute_verdict_fail_over_threshold():
    from calibration.consistency_esl_rates import MAX_FLAG_RATE_GAP_PTS, compute_verdict

    assert compute_verdict(gap=MAX_FLAG_RATE_GAP_PTS + 0.01, measurable=True) == "FAIL"


# ---------------------------------------------------------------------------
# concat-mode pure logic (pseudo-document construction + aggregation).
# The `eligible_count_fn` is INJECTED so these tests need no corpus and no
# stylometry stack — a fake counter (one eligible paragraph per "P" token) lets
# the construction/aggregation math be verified deterministically.
# ---------------------------------------------------------------------------

def _fake_counter(text: str) -> int:
    # Each 'P' in the fake essay text stands for one eligible (>=50-word) paragraph.
    return text.count("P")


def test_build_pseudo_docs_greedy_reaches_min_paragraphs():
    from calibration.consistency_esl_rates import MIN_PARAGRAPHS, build_pseudo_docs

    # 6 essays, 2 eligible paragraphs each -> greedy packs 3 essays per pseudo-doc
    # (2+2+2 = 6 == MIN_PARAGRAPHS) -> exactly 2 pseudo-docs, no leftover.
    essays = ["PP", "PP", "PP", "PP", "PP", "PP"]
    docs = build_pseudo_docs(essays, _fake_counter, seed=1)
    assert len(docs) == 2
    for d in docs:
        assert d["n_eligible_paragraphs"] >= MIN_PARAGRAPHS
        assert d["n_essays"] == 3


def test_build_pseudo_docs_drops_short_leftover():
    from calibration.consistency_esl_rates import build_pseudo_docs

    # 4 essays x 2 eligible paras: first 3 -> 1 pseudo-doc (6 paras); the 4th (2
    # paras < MIN_PARAGRAPHS) is an incomplete accumulator and is DROPPED, never
    # padded into a sub-eligible pseudo-doc.
    essays = ["PP", "PP", "PP", "PP"]
    docs = build_pseudo_docs(essays, _fake_counter, seed=1)
    assert len(docs) == 1
    assert docs[0]["n_essays"] == 3


def test_build_pseudo_docs_deterministic_same_seed():
    from calibration.consistency_esl_rates import build_pseudo_docs

    essays = [f"E{i}P" for i in range(30)]
    a = build_pseudo_docs(essays, _fake_counter, seed=7)
    b = build_pseudo_docs(essays, _fake_counter, seed=7)
    assert [d["text"] for d in a] == [d["text"] for d in b]


def test_build_pseudo_docs_symmetric_band_agnostic():
    # Identical construction procedure regardless of band: the function only sees a
    # list of texts + the seed, so two bands with identical inputs yield identical
    # pseudo-docs -> any measured gap comes from prose, not construction.
    from calibration.consistency_esl_rates import build_pseudo_docs

    essays = [f"E{i}P" for i in range(30)]
    lower_like = build_pseudo_docs(essays, _fake_counter, seed=3)
    higher_like = build_pseudo_docs(list(essays), _fake_counter, seed=3)
    assert [d["text"] for d in lower_like] == [d["text"] for d in higher_like]


def test_summarize_concat_group_adds_observability_and_reuses_summarize():
    from calibration.consistency_esl_rates import summarize_concat_group

    # 3 pseudo-docs: 2 flagged, 1 clean; every pseudo-doc is eligible by construction.
    s = summarize_concat_group(
        outlier_counts=[1, 0, 3],
        essays_per_doc=[3, 4, 2],
        eligible_paras_per_doc=[6, 7, 6],
    )
    assert s["n_pseudo_docs"] == 3
    assert s["n_eligible"] == 3  # all eligible by construction
    assert s["eligibility_rate_pct"] == 100.0
    assert s["flagged_pseudo_doc_rate_pct"] == round(2 / 3 * 100, 2)
    # Observability of the K-asymmetry confound is surfaced, not hidden.
    assert s["mean_essays_per_pseudo_doc"] == round(9 / 3, 3)
    assert s["mean_eligible_paras_per_pseudo_doc"] == round(19 / 3, 3)


def test_concat_gap_and_verdict_reuse_shared_helpers():
    # concat summaries carry flagged_essay_rate_pct too, so the SAME
    # flag_rate_gap_pts / groups_measurable / compute_verdict pipeline drives the
    # verdict — no parallel verdict logic.
    from calibration.consistency_esl_rates import (
        MIN_ELIGIBLE_PER_GROUP,
        compute_verdict,
        flag_rate_gap_pts,
        groups_measurable,
    )

    groups = {
        "lower": {"flagged_essay_rate_pct": 30.0, "n_eligible": MIN_ELIGIBLE_PER_GROUP},
        "higher": {"flagged_essay_rate_pct": 20.0, "n_eligible": MIN_ELIGIBLE_PER_GROUP},
    }
    gap = flag_rate_gap_pts(groups)
    assert gap == 10.0
    measurable = groups_measurable(groups)
    assert compute_verdict(gap, measurable) == "FAIL"  # 10 > 5.0 threshold
