from detect.repair_units import build_repair_units_v2
from rewrite_v2.goal_contract import evaluate_rewrite_goal
from rewrite_controller.eligible_span_density import build_preferred_eligible_span_density_contract
from rewrite_v5.residual_comb import _density_gate_for_report


def _segment(source: str, sentence: str, sentence_id: str, sentence_index: int, paragraph_id: str, risk: float) -> dict:
    start = source.index(sentence)
    return {
        "segment_id": sentence_id,
        "sentence_id": sentence_id,
        "sentence_index": sentence_index,
        "paragraph_id": paragraph_id,
        "start_char": start,
        "end_char": start + len(sentence),
        "text": sentence,
        "signals": [
            {
                "key": "ai_likelihood",
                "score": round(risk * 100),
                "rewrite_permission": "auto",
            }
        ],
        "predictability": {
            "score": risk,
            "top10_ratio": min(1.0, risk + 0.08),
            "top50_ratio": min(1.0, risk + 0.16),
            "predictable_token_spans": [sentence.split(" ")[0]],
        },
    }


def test_scanner_repair_units_emit_exact_slice_clusters_without_content_keywords():
    source = (
        "The course introduced the topic in a broad way. "
        "Students then practised the same process through a guided activity. "
        "The final reflection explained why the activity mattered."
    )
    first = "The course introduced the topic in a broad way."
    second = "Students then practised the same process through a guided activity."
    third = "The final reflection explained why the activity mattered."
    segments = [
        _segment(source, first, "s001", 0, "p001", 0.72),
        _segment(source, second, "s002", 1, "p001", 0.69),
        _segment(source, third, "s003", 2, "p001", 0.25),
    ]

    contract = build_repair_units_v2(
        source_text=source,
        segments=segments,
        paragraph_rows=[{"paragraph_id": "p001", "sentence_ids": ["s001", "s002", "s003"]}],
        blocker_radar={
            "blockers": [
                {"key": "topk_pattern", "score": 82, "sentence_ids": ["s001", "s002"]},
            ]
        },
        authorship_window_profile={
            "windows": [
                {"paragraph_id": "p001", "ai_assistance_score": 0.61},
            ]
        },
        rewrite_target_profile={"driver_summary": {"predictability_score": 1}},
    )

    assert contract["schema_version"] == "repair_units.v2"
    assert contract["selection_policy"]["content_keyword_matching"] is False
    assert contract["contract_checks"]["all_units_have_valid_source_slice"] is True
    assert contract["contract_checks"]["all_units_start_on_clean_boundary"] is True
    assert contract["contract_checks"]["all_units_end_on_clean_boundary"] is True
    assert contract["contract_checks"]["all_units_have_sentence_ids"] is True

    units = contract["repair_units"]
    assert units
    top_unit = units[0]
    assert top_unit["unit_type"] == "density_cluster"
    assert top_unit["source"] == "scanner.repair_units_v2"
    assert top_unit["source_text"] == source[top_unit["start_char"]:top_unit["end_char"]]
    assert "s001" in top_unit["sentence_ids"]
    assert "predictability_score" in {row["key"] for row in top_unit["dominant_drivers"]}

    density = contract["eligible_span_density_gate"]
    assert density["source"] == "scanner.repair_units_v2"
    assert density["top_unsafe_clusters"]
    assert density["top_unsafe_clusters"][0]["source"] == "scanner.repair_units_v2"


def test_v5_prefers_scanner_owned_density_gate_when_report_contract_is_valid():
    source = "A first risky sentence appears here. A second risky sentence follows it."
    first = "A first risky sentence appears here."
    second = "A second risky sentence follows it."
    segments = [
        _segment(source, first, "s001", 0, "p001", 0.7),
        _segment(source, second, "s002", 1, "p001", 0.68),
    ]
    contract = build_repair_units_v2(
        source_text=source,
        segments=segments,
        paragraph_rows=[{"paragraph_id": "p001", "sentence_ids": ["s001", "s002"]}],
        blocker_radar={"blockers": []},
        authorship_window_profile={"windows": [{"paragraph_id": "p001", "ai_assistance_score": 0.55}]},
    )
    report = {"repair_units_v2": contract}

    density = _density_gate_for_report(source, report)

    assert density["version"] == "scanner_repair_units_density_v2"
    assert density["source"] == "scanner.repair_units_v2"
    assert density["top_unsafe_clusters"]


def test_goal_contract_uses_scanner_owned_density_when_available():
    source = "A first risky sentence appears here. A second risky sentence follows it."
    first = "A first risky sentence appears here."
    second = "A second risky sentence follows it."
    segments = [
        _segment(source, first, "s001", 0, "p001", 0.7),
        _segment(source, second, "s002", 1, "p001", 0.68),
    ]
    contract = build_repair_units_v2(
        source_text=source,
        segments=segments,
        paragraph_rows=[{"paragraph_id": "p001", "sentence_ids": ["s001", "s002"]}],
        blocker_radar={"blockers": []},
        authorship_window_profile={"windows": [{"paragraph_id": "p001", "ai_assistance_score": 0.55}]},
    )
    report = {"repair_units_v2": contract}

    direct_density = build_preferred_eligible_span_density_contract(source, report)
    goal = evaluate_rewrite_goal(
        original_text=source,
        candidate_text=source,
        original_report=report,
        candidate_report=report,
    ).to_dict()

    assert direct_density["source"] == "scanner.repair_units_v2"
    assert goal["eligible_span_density_gate"]["source"] == "scanner.repair_units_v2"
    assert goal["eligible_span_density_gate"]["unsafe_cluster_count"] == direct_density["unsafe_cluster_count"]


def test_scanner_density_count_is_not_capped_to_prompt_unit_limit():
    sentences = [
        "The first risky sentence appears in a separate paragraph.",
        "The second risky sentence appears in another paragraph.",
        "The third risky sentence appears in a further paragraph.",
        "The fourth risky sentence appears in a later paragraph.",
        "The fifth risky sentence appears near the end.",
    ]
    source = "\n\n".join(sentences)
    segments = [
        _segment(source, sentence, f"s{index + 1:03d}", index, f"p{index + 1:03d}", 0.8 - (index * 0.02))
        for index, sentence in enumerate(sentences)
    ]

    contract = build_repair_units_v2(
        source_text=source,
        segments=segments,
        paragraph_rows=[
            {"paragraph_id": f"p{index + 1:03d}", "sentence_ids": [f"s{index + 1:03d}"]}
            for index in range(len(sentences))
        ],
        blocker_radar={"blockers": []},
        authorship_window_profile={
            "windows": [
                {"paragraph_id": f"p{index + 1:03d}", "ai_assistance_score": 0.6}
                for index in range(len(sentences))
            ]
        },
        max_units=2,
    )

    assert len(contract["repair_units"]) == 2
    assert contract["eligible_span_density_gate"]["unsafe_cluster_count"] > len(contract["repair_units"])
    assert len(contract["eligible_span_density_gate"]["top_unsafe_clusters"]) == 2
