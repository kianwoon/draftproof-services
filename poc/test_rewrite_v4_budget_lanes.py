from rewrite_v4.budget_lanes import (
    BudgetLane,
    budget_lanes_for_unit,
    budget_profile_passes,
    edit_budget_profile,
)
from rewrite_v4.experiment import (
    _active_layer3_blocker,
    _is_blocker_budget_lane_positive,
    _is_material_budget_lane_positive,
    _is_safe_budget_lane_positive,
    _rank_budget_lane_patch_packs,
)
from rewrite_v4.models import ClusterRepairUnit
from rewrite_v4.validation import source_grounding_integrity


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

    assert [lane.lane_id for lane in lanes] == ["conservative", "route", "aggressive", "clearance"]
    assert lanes[0].changed_source_ratio_max < lanes[1].changed_source_ratio_max < lanes[2].changed_source_ratio_max
    assert lanes[0].growth_ratio_max < lanes[1].growth_ratio_max
    assert lanes[3].changed_source_ratio_max > lanes[2].changed_source_ratio_max


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


def test_v4_layer3_safe_positive_is_not_always_material():
    row = _budget_lane_row({
        "ai_delta": 0.06,
        "topk_delta": 0.09,
        "external_delta": 0.285,
        "rank_delta": 0.035,
        "topk_calibrated_risk_delta": 0.245,
        "unsafe_word_ratio_delta": -0.184,
        "external_ai_flag_risk_delta": 0.059,
        "unsafe_cluster_delta": 0,
        "risky_window_delta": 0,
    })

    assert _is_safe_budget_lane_positive(row)
    assert not _is_material_budget_lane_positive(row)


def test_v4_layer3_material_positive_uses_blocker_movement():
    row = _budget_lane_row({
        "ai_delta": 0.12,
        "topk_delta": 0.2,
        "external_delta": 0.3,
        "rank_delta": 0.2,
        "topk_calibrated_risk_delta": 2.1,
        "unsafe_word_ratio_delta": 0.0,
        "external_ai_flag_risk_delta": 0.0,
        "unsafe_cluster_delta": 0,
        "risky_window_delta": 0,
    })

    assert _is_safe_budget_lane_positive(row)
    assert _is_material_budget_lane_positive(row)


def test_v4_layer3_active_cluster_blocker_requires_cluster_reduction():
    blocker = _active_layer3_blocker(
        {
            "eligible_span_density_gate": {
                "thresholds": {"max_unsafe_cluster_count": 4},
            }
        },
        {"unsafe_cluster_count": 9},
    )
    weak_row = _budget_lane_row({
        "ai_delta": 1.0,
        "topk_delta": 3.0,
        "external_delta": 1.0,
        "rank_delta": 1.0,
        "topk_calibrated_risk_delta": 3.0,
        "unsafe_word_ratio_delta": 3.0,
        "external_ai_flag_risk_delta": 1.0,
        "unsafe_cluster_delta": 0,
        "risky_window_delta": 1,
    })
    clearing_row = _budget_lane_row({
        "ai_delta": 0.1,
        "topk_delta": 0.1,
        "external_delta": 0.1,
        "rank_delta": 0.1,
        "topk_calibrated_risk_delta": 0.1,
        "unsafe_word_ratio_delta": 0.1,
        "external_ai_flag_risk_delta": 0.1,
        "unsafe_cluster_delta": 1,
        "risky_window_delta": 0,
    })

    assert blocker["primary"] == "unsafe_cluster_count"
    assert not _is_blocker_budget_lane_positive(weak_row, blocker)
    assert _is_blocker_budget_lane_positive(clearing_row, blocker)


def test_v4_layer3_pack_composes_multiple_cluster_patches(monkeypatch, tmp_path):
    units = [
        ClusterRepairUnit(
            cluster_id="cluster_001",
            start_sentence=0,
            end_sentence=0,
            start_char=0,
            end_char=11,
            text="Alpha risk.",
            before_context="",
            after_context=" Beta risk.",
            sentence_count=1,
            word_count=2,
            risk_score=10.0,
        ),
        ClusterRepairUnit(
            cluster_id="cluster_002",
            start_sentence=1,
            end_sentence=1,
            start_char=12,
            end_char=22,
            text="Beta risk.",
            before_context="Alpha risk. ",
            after_context="",
            sentence_count=1,
            word_count=2,
            risk_score=9.0,
        ),
    ]
    rows = [
        {
            "cluster_id": "cluster_001",
            "variant_id": "v1",
            "text": "Alpha clear.",
            "apply_status": {"applied": True},
            "source_grounding": {"passed": True},
            "budget_lane": {"lane_id": "clearance", "changed_source_ratio_max": 1.0, "growth_ratio_max": 1.0, "shrink_ratio_max": 1.0},
            "edit_budget_profile": {"changed_source_ratio": 0.5, "growth_ratio": 0.0, "shrink_ratio": 0.0},
            "scores": {"unsafe_cluster_delta": 1, "topk_calibrated_risk_delta": 1},
        },
        {
            "cluster_id": "cluster_002",
            "variant_id": "v1",
            "text": "Beta clear.",
            "apply_status": {"applied": True},
            "source_grounding": {"passed": True},
            "budget_lane": {"lane_id": "clearance", "changed_source_ratio_max": 1.0, "growth_ratio_max": 1.0, "shrink_ratio_max": 1.0},
            "edit_budget_profile": {"changed_source_ratio": 0.5, "growth_ratio": 0.0, "shrink_ratio": 0.0},
            "scores": {"unsafe_cluster_delta": 1, "topk_calibrated_risk_delta": 1},
        },
    ]

    monkeypatch.setattr("rewrite_v4.experiment._scan_report", lambda text: {"text": text})
    monkeypatch.setattr(
        "rewrite_v4.experiment.evaluate_rewrite_goal",
        lambda **_: {
            "eligible_span_density_gate": {
                "unsafe_cluster_count": 0,
                "unsafe_eligible_word_ratio": 0.0,
            }
        },
    )
    monkeypatch.setattr("rewrite_v4.experiment._metrics", lambda **_: {})
    monkeypatch.setattr("rewrite_v4.experiment._score_summary", lambda *_: {"ai": 8, "topk": 9, "external": 7, "rank": 6})
    monkeypatch.setattr("rewrite_v4.experiment._unsafe_cluster_count", lambda goal: 0)
    monkeypatch.setattr("rewrite_v4.experiment._unsafe_word_ratio", lambda goal, scores: 0.0)

    packs = _rank_budget_lane_patch_packs(
        original_text="Alpha risk. Beta risk.",
        baseline_report={},
        baseline={"ai": 10, "topk": 11, "external": 9, "rank": 8, "unsafe_cluster_count": 2, "unsafe_word_ratio": 20.0},
        units=units,
        rows=rows,
        output_dir=tmp_path,
        active_blocker={"active": True, "primary": "unsafe_cluster_count"},
    )

    assert packs
    assert packs[0]["candidate_text"] == "Alpha clear. Beta clear."
    assert packs[0]["scores"]["unsafe_cluster_delta"] == 2
    assert _is_blocker_budget_lane_positive(packs[0], {"active": True, "primary": "unsafe_cluster_count"})


def test_v4_source_near_route_rebuild_allows_claim_preserving_route_change():
    source = (
        "This allows students to establish consistent neatness and control before moving on to the next step. "
        "This process not only improves the quality of their work but also helps students gradually build confidence through practice, rather than simply through a specific method. "
        "In additional, I encourage my students to find an approach that works best for them and then compare it with traditional techniques."
    )
    replacement = (
        "This helps students gain consistent neatness and command prior to the next stage. "
        "The process improves work quality and fosters student confidence gradually through hands-on practice, rather than through a single prescribed method. "
        "I further urge students to discover which way works best for them and then weigh it against standard techniques."
    )

    strict = source_grounding_integrity(source, replacement)
    route = source_grounding_integrity(source, replacement, repair_mode="source_near_route_rebuild")

    assert not strict["passed"]
    assert route["passed"]
    assert not route["external_review_required"]


def _budget_lane_row(scores):
    lane = BudgetLane(
        lane_id="conservative",
        changed_source_ratio_max=0.25,
        growth_ratio_max=0.12,
        shrink_ratio_max=0.12,
        operation_families=("replace_only",),
        instruction="test",
    )
    return {
        "apply_status": {"applied": True},
        "budget_lane": lane.to_dict(),
        "edit_budget_profile": {
            "changed_source_ratio": 0.1,
            "growth_ratio": 0.0,
            "shrink_ratio": 0.0,
        },
        "source_grounding": {"passed": True},
        "scores": scores,
        "candidate_goal": {
            "ai_footprint_gate": {
                "thresholds": {
                    "topk_calibrated_risk": 2.0,
                    "external_ai_flag_risk": 1.5,
                    "ai_likelihood": 1.0,
                }
            }
        },
    }
