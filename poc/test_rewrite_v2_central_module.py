from rewrite_v2.central_module import (
    GROUP_WEIGHTS,
    build_contextual_judgment_plan,
    extract_central_signal_profile,
    score_candidate_against_central_profile,
)


def assert_test(condition, message):
    if not condition:
        raise AssertionError(message)


def fixture_report(*, topk=84, human_anchor=84, grounding=70, ai=39):
    return {
        "ai_risk_badge": {
            "ai_likelihood_score": ai,
            "ai_components": {
                "topk_pattern_raw": topk,
                "topk_calibrated_risk": 56,
            },
            "writing_components": {
                "source_grounding_risk": grounding,
            },
            "transformation_classification": {
                "features": {
                    "outline_to_text_expansion": 0.46,
                    "rewrite_smoothness": 0.43,
                    "section_style_variance": 0.39,
                    "semantic_uniformity_risk": 0.33,
                    "discourse_regularity_risk": 0.17,
                    "human_anchor_score": human_anchor / 100,
                    "human_anchor_discount": 0.38,
                    "calibration_confidence": 0.65,
                    "reporting_suppression": 0.35,
                    "adjusted_ai_risk": 0.24,
                    "signal_agreement_score": 0.23,
                    "calibrated_ai_risk": 0.17,
                    "paraphrase_transformation_risk": 0.0,
                    "source_similarity": 0.0,
                    "surface_similarity": 0.0,
                }
            },
        },
        "integrity_layers": {
            "layers": {
                "human_contribution_signal": {"score": human_anchor},
                "grounding_quality_risk": {"score": grounding},
            }
        },
    }


source = (
    "The United States is often described as one of the most influential countries in modern history. "
    "It has shaped global politics, economics, technology, entertainment, and education for many decades. "
    "Although the country is relatively young compared to many civilizations, it developed rapidly into a major world power. "
    "The United States is known for its diversity, democratic system, economic strength, and cultural influence. "
    "However, like every nation, it also faces challenges related to inequality, politics, healthcare, and social division. "
    "Understanding the United States requires looking at both its achievements and its struggles."
)


profile = extract_central_signal_profile(fixture_report())
assert_test(GROUP_WEIGHTS["human_authenticity"] == 0.45, "central human/authenticity weight is 45%")
assert_test(GROUP_WEIGHTS["anti_ai_authorship"] == 0.35, "central anti-AI-authorship weight is 35%")
assert_test(GROUP_WEIGHTS["quality_calibration_trust"] == 0.20, "central quality/calibration weight is 20%")
assert_test(profile.ai_authorship["raw_topk_predictability"] == 84, "raw top-k extracted into AI authorship group")
assert_test(profile.human_authenticity["human_anchor"] == 84, "human anchor extracted into authenticity group")
assert_test(profile.quality_calibration["grounding_risk"] == 70, "grounding risk extracted into quality group")
assert_test(profile.weighted_human_target_score > 0, "weighted human target score is computed")


plan = build_contextual_judgment_plan(source_text=source, scan_report=fixture_report())
plan_dict = plan.to_dict()
assert_test(plan.strategy_id == "central_contextual_judgment_v1", "central plan uses contextual judgment strategy")
assert_test(
    "safe_contextual_anchor_expansion" in plan_dict["generation_objective"]["primary_drivers"],
    "high grounding risk selects contextual anchor expansion",
)
assert_test(
    "reasoning_interruption" in plan_dict["generation_objective"]["primary_drivers"],
    "rewrite smoothness or semantic uniformity selects reasoning interruption",
)
assert_test(
    plan_dict["constraints"]["minimum_words"] == int(len(source.split()) * 0.85),
    "central plan sets word-preservation minimum",
)
assert_test(
    len(plan_dict["constraints"]["preserve_content_units"]) == 6,
    "central plan derives one content unit per source sentence",
)


better_candidate = fixture_report(topk=45, human_anchor=88, grounding=45, ai=18)
score = score_candidate_against_central_profile(better_candidate, baseline_report=fixture_report())
assert_test(score["weighted_delta"] > 0, "better candidate improves central weighted score")
assert_test(
    score["group_deltas"]["ai_authorship_risk"] < 0,
    "better candidate reduces AI authorship risk group",
)
assert_test(
    score["group_deltas"]["human_authenticity"] > 0,
    "better candidate improves human authenticity group",
)


print("Rewrite V2 central module tests passed.")
