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
