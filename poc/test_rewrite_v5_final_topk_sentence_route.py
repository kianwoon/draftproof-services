from rewrite_v5.residual_comb import (
    _final_topk_sentence_route_targets,
    _has_final_topk_sentence_route_movement,
    _sanitize_final_topk_sentence_route_variants,
)


def test_final_topk_sentence_route_targets_use_density_first_then_predictability_rows():
    text = (
        "This shift makes the teacher's role more important. "
        "The real challenge is knowing what to trust. "
        "Assessment should include the process."
    )
    report = {
        "predictability": {
            "all_sentences": [
                {
                    "sentence_id": "s002",
                    "sentence": "The real challenge is knowing what to trust.",
                    "top10_ratio": 0.7,
                    "top50_ratio": 0.8,
                    "predictability_risk": 0.5,
                },
                {
                    "sentence_id": "s003",
                    "sentence": "Assessment should include the process.",
                    "top10_ratio": 0.6,
                    "top50_ratio": 0.7,
                    "predictability_risk": 0.4,
                },
            ]
        }
    }
    goal = {
        "eligible_span_density_gate": {
            "top_sentence_targets": [
                {
                    "sentence_id": "s001",
                    "preview": "This shift makes the teacher's role more important.",
                    "top10_ratio": 0.8,
                    "top50_ratio": 0.9,
                    "predictability_risk": 0.6,
                }
            ]
        }
    }

    targets = _final_topk_sentence_route_targets(text, report, goal)

    assert [row["sentence_id"] for row in targets[:3]] == ["s001", "s002", "s003"]
    assert targets[0]["target_id"] == "t001"
    assert targets[0]["source"] == "eligible_span_density_gate"


def test_final_topk_sentence_route_sanitizer_keeps_known_target_repairs_only():
    targets = [{"target_id": "t001"}, {"target_id": "t002"}]
    raw = [
        {
            "variant_id": "v1",
            "repairs": [
                {"target_id": "t001", "sentence_job": "bridge", "current_route": "a", "repair_route": "b", "after": "  New route.  "},
                {"target_id": "missing", "sentence_job": "bridge", "current_route": "a", "repair_route": "b", "after": "No."},
                {"target_id": "t002", "sentence_job": "summary", "current_route": "c", "repair_route": "d", "after": "Second route."},
            ],
        }
    ]

    rows = _sanitize_final_topk_sentence_route_variants(raw, targets=targets)

    assert len(rows) == 1
    assert [repair["target_id"] for repair in rows[0]["repairs"]] == ["t001", "t002"]
    assert rows[0]["repairs"][0]["after"] == "New route."


def test_final_topk_sentence_route_accepts_topk_drop_without_regressing_blockers():
    row = {
        "incremental": {
            "topk_delta": 1.2,
            "topk_calibrated_risk_delta": 3.0,
            "risky_window_count_delta": 0,
            "unsafe_cluster_count_delta": 0,
        }
    }

    assert _has_final_topk_sentence_route_movement(row)


def test_final_topk_sentence_route_rejects_unsafe_cluster_regression():
    row = {
        "incremental": {
            "topk_delta": 5.0,
            "topk_calibrated_risk_delta": 10.0,
            "risky_window_count_delta": 0,
            "unsafe_cluster_count_delta": -1,
        }
    }

    assert not _has_final_topk_sentence_route_movement(row)
