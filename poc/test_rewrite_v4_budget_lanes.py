from rewrite_v4.budget_lanes import (
    BudgetLane,
    budget_lanes_for_unit,
    budget_profile_passes,
    edit_budget_profile,
)


def test_v4_budget_profile_counts_changed_source_and_net_drift():
    source = (
        "Today’s education system is changing faster than many schools can comfortably manage. "
        "In the past, education was mostly built around the classroom, the textbook, and the teacher."
    )
    replacement = (
        "Today’s education system is changing faster than many schools can easily handle. "
        "Previously, education was mostly centred on the classroom, the textbook, and the teacher."
    )

    profile = edit_budget_profile(source, replacement)

    assert profile["source_words"] == len(source.split())
    assert profile["replacement_words"] == len(replacement.split())
    assert profile["changed_source_words"] > 0
    assert profile["changed_source_ratio"] > 0
    assert profile["shrink_ratio"] > 0


def test_v4_budget_lanes_are_ordered_from_conservative_to_aggressive():
    lanes = budget_lanes_for_unit(58, risk_score=0.8)

    assert [lane.lane_id for lane in lanes] == ["conservative", "route", "aggressive"]
    assert lanes[0].changed_source_ratio_max < lanes[1].changed_source_ratio_max < lanes[2].changed_source_ratio_max
    assert lanes[0].growth_ratio_max < lanes[1].growth_ratio_max


def test_v4_budget_profile_passes_lane_thresholds():
    lane = BudgetLane(
        lane_id="conservative",
        changed_source_ratio_max=0.25,
        growth_ratio_max=0.12,
        shrink_ratio_max=0.12,
        operation_families=("replace_only",),
        instruction="test",
    )

    assert budget_profile_passes(
        {"changed_source_ratio": 0.19, "growth_ratio": 0.0, "shrink_ratio": 0.034},
        lane,
    )
    assert not budget_profile_passes(
        {"changed_source_ratio": 0.31, "growth_ratio": 0.10, "shrink_ratio": 0.0},
        lane,
    )
