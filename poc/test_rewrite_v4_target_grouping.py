import json
from types import SimpleNamespace

from llm.gateway import LLMGateway, LLMResponse
from rewrite_v3.target_executor import group_rewrite_targets
from rewrite_v4.cluster_patch import apply_cluster_variant, build_cluster_repair_units, build_cluster_generator_prompt
from rewrite_v4.experiment import _add_goal_driver_snapshot, _add_score_deltas, _is_safe_residual_positive
from rewrite_v4.generator import build_generator_prompt, generate_variants
from rewrite_v4.models import ClusterRepairUnit
from rewrite_v4.normalizer import _merge_strategy_defaults, deterministic_repair_brief
from rewrite_v4.validation import parse_generator_variants, strategy_compliance_integrity


def test_v4_chunk_scope_preserves_scanner_window_over_paragraph_id():
    first = (
        "Introduction\n"
        "In vocational education, inclusive learning design matters for hairdressing students.\n"
        "The course context explains why the opening claim belongs here."
    )
    second = (
        "How Inclusive Learning Design Can Address the Diverse Needs of Learners\n"
        "Cognitive overload is one barrier when students learn sectioning and projection.\n"
        "The source-only anchor appears in this chunk."
    )
    third = "Later section\nThis block should not be pulled into the first repair unit."
    text = f"{first}\n\n{second}\n\n{third}"
    raw_end_inside_second = text.index("The source-only anchor")
    target_source = text[:raw_end_inside_second]
    target_words = len(target_source.split())

    groups = group_rewrite_targets(
        original_text=text,
        rewrite_target_profile={
            "targets": [
                {
                    "target_id": "rt001",
                    "unit_id": "p001",
                    "paragraph_id": "p001",
                    "scope_level": "chunk",
                    "recommended_operation": "paragraph_preserving_broad_reconstruction",
                    "operation_candidates": ["paragraph_preserving_broad_reconstruction", "chunk_reconstruction"],
                    "span": {
                        "start_index": 0,
                        "end_index": raw_end_inside_second,
                        "integrity": {
                            "start_index": 0,
                            "end_index": raw_end_inside_second,
                            "in_bounds": True,
                            "starts_on_boundary": True,
                            "ends_on_boundary": False,
                            "passed": False,
                        },
                    },
                    "source_text": target_source,
                    "word_count_guide": {
                        "source_words": target_words,
                        "preferred_words": target_words,
                    },
                    "protected_anchors": [
                        {"text": "source-only anchor", "kind": "quote"},
                        {"text": "missing anchor", "kind": "quote"},
                    ],
                    "dominant_drivers": [{"key": "predictability_score", "score": 0.6}],
                }
            ]
        },
        max_groups=4,
    )

    assert len(groups) == 1
    group = groups[0]
    assert group.start_index == 0
    assert group.end_index < text.index("Later section")
    assert "How Inclusive Learning Design Can Address the Diverse Needs of Learners" in group.source_text
    assert "Later section" not in group.source_text
    assert group.word_count_guide == {
        "source_words": len(group.source_text.split()),
        "preferred_words": len(group.source_text.split()),
    }
    assert [anchor["text"] for anchor in group.protected_anchors] == ["source-only anchor"]


def test_v4_generator_prompt_does_not_contradict_heading_structure():
    source = (
        "Introduction\n"
        "The Hairdressing Certificate III course shows why inclusive learning design matters.\n"
        "Students still need a clear link between the course context and the broader claim."
    )
    groups = group_rewrite_targets(
        original_text=source,
        rewrite_target_profile={
            "targets": [
                {
                    "target_id": "rt001",
                    "unit_id": "p001",
                    "paragraph_id": "p001",
                    "scope_level": "paragraph",
                    "recommended_operation": "paragraph_preserving_broad_reconstruction",
                    "span": {
                        "start_index": 0,
                        "end_index": len(source),
                        "integrity": {
                            "start_index": 0,
                            "end_index": len(source),
                            "in_bounds": True,
                            "starts_on_boundary": True,
                            "ends_on_boundary": True,
                            "passed": True,
                        },
                    },
                    "source_text": source,
                    "dominant_drivers": [{"key": "predictability_score", "score": 0.6}],
                }
            ]
        },
        max_groups=4,
    )
    group = groups[0]
    brief = deterministic_repair_brief(group)
    assert brief.mitigation_strategy["scope"] == "multi_block_section"
    assert brief.mitigation_strategy["strategy_id"] == "anchor_first_route_repair"
    prompt = build_generator_prompt(group=group, repair_brief=brief, variant_count=2)
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["mitigation_strategy"]["strategy_id"] == "anchor_first_route_repair"
    assert payload["route_plan"]["current_route"]
    assert payload["route_plan"]["better_route"]
    assert any("course or unit context" in item for item in payload["route_plan"]["better_route"])
    assert payload["rewrite_sequence"]
    assert payload["rewrite_sequence"][0]["op"] == "abstract_to_source_anchor"
    assert "not that the topic is important" in payload["rewrite_sequence"][0]["instruction"]
    assert any("practical skill problem" in step["instruction"] for step in payload["rewrite_sequence"])
    assert "Keep one paragraph." not in payload["constraints"]
    assert "headings" not in payload["avoid"]
    assert "new headings" in payload["avoid"]
    assert "important because opening" in payload["avoid"]
    assert "different from traditional methods bridge" in payload["avoid"]
    assert any("keep the first line as the heading" in item for item in payload["constraints"])
    assert any("Follow mitigation_strategy" in item for item in payload["constraints"])
    assert any("Apply rewrite_sequence in order" in item for item in payload["constraints"])
    assert any("Execute route_plan.better_route" in item for item in payload["constraints"])


def test_v4_strategy_candidate_count_hint_reduces_long_unit_variants():
    source = "Introduction\n" + " ".join(f"word{i}" for i in range(420))
    groups = group_rewrite_targets(
        original_text=source,
        rewrite_target_profile={
            "targets": [
                {
                    "target_id": "rt001",
                    "unit_id": "p001",
                    "paragraph_id": "p001",
                    "scope_level": "chunk",
                    "recommended_operation": "paragraph_preserving_broad_reconstruction",
                    "span": {
                        "start_index": 0,
                        "end_index": len(source),
                        "integrity": {
                            "start_index": 0,
                            "end_index": len(source),
                            "in_bounds": True,
                            "starts_on_boundary": True,
                            "ends_on_boundary": True,
                            "passed": True,
                        },
                    },
                    "source_text": source,
                    "dominant_drivers": [{"key": "predictability_score", "score": 0.6}],
                }
            ]
        },
        max_groups=4,
    )

    brief = deterministic_repair_brief(groups[0])
    prompt = build_generator_prompt(group=groups[0], repair_brief=brief, variant_count=3)
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert brief.mitigation_strategy["candidate_count_hint"] == "1"
    assert len(payload["output_schema"]["variants"]) == 1
    assert any("Return exactly 1 variant." == item for item in payload["constraints"])


def test_v4_strategy_compliance_rejects_dynamic_forbidden_route():
    strategy = {
        "forbidden_moves": ["different-from-traditional-methods bridge"],
        "strategy_steps": [
            {
                "op": "reaction_reason_link",
                "target": "learner response cluster",
                "instruction": "Preserve reactions and reasons.",
                "must_preserve": ["pretends to understand", "fear of being judged"],
                "avoid": ["generic confidence sentence"],
            }
        ],
    }

    result = strategy_compliance_integrity(
        "The paragraph says practical skill learning is different from traditional methods.",
        strategy,
    )

    assert not result["passed"]
    assert any(row["reason"] == "strategy_forbidden_route_present" for row in result["failures"])


def test_v4_strategy_compliance_accepts_source_near_preserve_phrase():
    strategy = {
        "strategy_steps": [
            {
                "op": "reaction_reason_link",
                "target": "learner response cluster",
                "instruction": "Preserve reactions and reasons.",
                "must_preserve": ["fear of being judged or labelled untalented"],
                "avoid": [],
            }
        ],
    }

    result = strategy_compliance_integrity(
        "Some students pretend to understand because they are afraid of being labelled as untalented.",
        strategy,
    )

    assert result["passed"]


def test_v4_strategy_compliance_rejects_repeated_claim_route():
    result = strategy_compliance_integrity(
        (
            "Inclusive learning design enables educators to build flexible lesson plans for different learner needs. "
            "Students also need feedback while they practise a skill. "
            "As a result, inclusive learning design enables educators to plan lessons that respond to diverse learner needs."
        ),
        {"strategy_id": "source_near_texture_repair"},
    )

    assert not result["passed"]
    assert any(row["reason"] == "redundant_sentence_claim" for row in result["failures"])


def test_v4_llm_strategy_merge_inherits_source_specific_defaults():
    source = "Introduction\n" + " ".join(f"word{i}" for i in range(420))
    group = SimpleNamespace(
        unit_id="p001",
        source_text=source,
        operation="paragraph_preserving_broad_reconstruction",
        targets=(
            {
                "target_id": "rt001",
                "dominant_drivers": [{"key": "predictability_score", "score": 0.6}],
            },
        ),
    )

    merged = _merge_strategy_defaults(
        group,
        {
            "scope": "paragraph",
            "strategy_id": "llm_sentence_tuning",
            "candidate_count_hint": "3",
            "strategy_steps": [
                {
                    "op": "claim_bridge_repair",
                    "target": "opening",
                    "instruction": "Connect the opening claims.",
                    "must_preserve": ["source viewpoint"],
                    "avoid": ["generic bridge"],
                }
            ],
        },
        role="opening/background framing",
    )

    assert merged["scope"] == "multi_block_section"
    assert merged["candidate_count_hint"] == "1"
    assert merged["current_route"]
    assert merged["better_route"]
    assert merged["route_moves"]
    assert any("different-from-traditional-methods" in item for step in merged["strategy_steps"] for item in step.get("avoid", []))
    assert any(step["op"] == "structure_preservation" for step in merged["strategy_steps"])


def test_v4_generator_parser_allows_multiblock_candidate_for_multiblock_source():
    source = (
        "Introduction\n"
        "The course context explains the first claim.\n\n"
        "Second Section\n"
        "The second block carries another source claim."
    )
    completion = json.dumps({
        "variants": [
            {
                "variant_id": "v1",
                "text": (
                    "Introduction\n"
                    "The course context explains the first claim more directly.\n\n"
                    "Second Section\n"
                    "The second block still carries another source claim."
                ),
            }
        ]
    })

    variants, diagnostics = parse_generator_variants(
        completion,
        min_words=5,
        max_words=40,
        source_text=source,
    )

    assert diagnostics["status"] == "ok"
    assert len(variants) == 1
    assert variants[0].variant_id == "v1"
    assert "\n\n" in variants[0].text


def test_v4_generator_parser_still_rejects_split_single_paragraph_candidate():
    source = "The course context explains the first claim in one paragraph."
    completion = json.dumps({
        "variants": [
            {
                "variant_id": "v1",
                "text": "The course context explains the first claim.\n\nThis added block should fail.",
            }
        ]
    })

    variants, diagnostics = parse_generator_variants(
        completion,
        min_words=5,
        max_words=40,
        source_text=source,
    )

    assert variants == []
    assert diagnostics["status"] == "schema_failed"
    assert diagnostics["rejected"][0]["reason"] == "paragraph_split"


def test_v4_generator_uses_raw_json_completion_and_8k_default(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V4_GENERATOR_MAX_TOKENS", raising=False)
    source = "Education should compare “what students know,” with “how students think.”"
    raw_completion = json.dumps({
        "variants": [
            {
                "variant_id": "v1",
                "text": "Education should compare “what students know,” with “how students think” more clearly.",
            }
        ]
    }, ensure_ascii=False)

    class Gateway:
        model = "deepseek/deepseek-v4-pro"
        provider = {"order": ["DeepInfra"]}
        requested_max_tokens = None
        requested_response_format = None
        requested_provider = None

        def chat(self, *_args, **kwargs):
            self.requested_max_tokens = kwargs.get("max_tokens")
            self.requested_response_format = kwargs.get("response_format")
            self.requested_provider = kwargs.get("provider")
            return LLMResponse(
                content=LLMGateway._normalize_quotes(raw_completion),
                model="test",
                usage={},
                raw={"choices": [{"message": {"content": raw_completion}, "finish_reason": "stop"}]},
            )

    gateway = Gateway()
    group = SimpleNamespace(source_text=source)
    brief = deterministic_repair_brief(SimpleNamespace(unit_id="p001", source_text=source, targets=()))

    variants, diagnostics, _prompt, completion = generate_variants(
        group=group,
        repair_brief=brief,
        gateway=gateway,
        variant_count=1,
    )

    assert gateway.requested_max_tokens == 8000
    assert gateway.requested_response_format["type"] == "json_schema"
    assert gateway.requested_response_format["json_schema"]["strict"] is True
    assert gateway.requested_provider == {"order": ["DeepInfra"], "require_parameters": True}
    assert diagnostics["status"] == "ok"
    assert diagnostics["structured_output_mode"] == "required_schema"
    assert diagnostics["max_tokens"] == 8000
    assert len(variants) == 1
    assert completion == raw_completion


def test_v4_generator_focuses_long_multiblock_unit_and_appends_locked_suffix(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_REWRITE_V4_FOCUSED_PREFIX_WORD_THRESHOLD", raising=False)
    editable = "Introduction\n" + " ".join(f"open{i}" for i in range(85))
    suffix = "\n\nSecond Section\n" + " ".join(f"locked{i}" for i in range(260))
    source = editable + suffix
    edited = "Introduction\n" + " ".join(f"edited{i}" for i in range(85))
    raw_completion = json.dumps({
        "variants": [
            {
                "variant_id": "v1",
                "text": edited,
            }
        ]
    })

    class Gateway:
        model = "deepseek/deepseek-v4-pro"
        provider = None

        def chat(self, *_args, **kwargs):
            return LLMResponse(
                content=raw_completion,
                model="test",
                usage={},
                raw={"choices": [{"message": {"content": raw_completion}, "finish_reason": "stop"}]},
            )

    group = SimpleNamespace(unit_id="p001", source_text=source, targets=())
    brief = deterministic_repair_brief(group)
    prompt = build_generator_prompt(group=group, repair_brief=brief, variant_count=1)
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["generation_scope"]["mode"] == "focused_editable_prefix"
    assert payload["original_unit"] == editable
    assert "Second Section" not in payload["original_unit"]
    assert any("locked_suffix_context" in item for item in payload["constraints"])

    variants, diagnostics, _prompt, _completion = generate_variants(
        group=group,
        repair_brief=brief,
        gateway=Gateway(),
        variant_count=1,
    )

    assert diagnostics["status"] == "ok"
    assert diagnostics["generation_scope"]["mode"] == "focused_editable_prefix"
    assert len(variants) == 1
    assert variants[0].text == edited + suffix


def test_v4_cluster_units_use_sentence_map_offsets_exactly():
    text = "First sentence. Second sentence has the issue. Third sentence continues it. Fourth is context."
    report = {
        "sentence_map": {
            "s001": {"start_char": 0, "end_char": 15, "text": "First sentence.", "paragraph_id": "p001"},
            "s002": {"start_char": 16, "end_char": 46, "text": "Second sentence has the issue.", "paragraph_id": "p001"},
            "s003": {"start_char": 47, "end_char": 75, "text": "Third sentence continues it.", "paragraph_id": "p001"},
            "s004": {"start_char": 76, "end_char": 94, "text": "Fourth is context.", "paragraph_id": "p001"},
        }
    }
    goal = {
        "eligible_span_density_gate": {
            "top_unsafe_clusters": [
                {
                    "start_sentence": 1,
                    "end_sentence": 2,
                    "sentence_count": 2,
                    "word_count": 8,
                    "risk_score": 9.5,
                }
            ]
        }
    }

    units = build_cluster_repair_units(text=text, report=report, goal=goal, limit=1)

    assert len(units) == 1
    assert units[0].text == "Second sentence has the issue. Third sentence continues it."
    assert units[0].start_char == 16
    assert units[0].end_char == 75
    assert units[0].metadata["sentence_ids"] == ["s002", "s003"]


def test_v4_cluster_units_relocate_sentence_text_when_offsets_drift():
    text = "Curly “quote” shifts offsets. Target sentence starts here. Another target follows."
    report = {
        "sentence_map": {
            "s001": {"start_char": 0, "end_char": 29, "text": "Curly “quote” shifts offsets.", "paragraph_id": "p001"},
            "s002": {"start_char": 24, "end_char": 52, "text": "Target sentence starts here.", "paragraph_id": "p001"},
            "s003": {"start_char": 53, "end_char": 76, "text": "Another target follows.", "paragraph_id": "p001"},
        }
    }
    goal = {
        "eligible_span_density_gate": {
            "top_unsafe_clusters": [
                {
                    "start_sentence": 1,
                    "end_sentence": 2,
                    "sentence_count": 2,
                    "word_count": 7,
                    "risk_score": 7.0,
                }
            ]
        }
    }

    units = build_cluster_repair_units(text=text, report=report, goal=goal, limit=1)

    assert len(units) == 1
    assert units[0].text == "Target sentence starts here. Another target follows."
    assert text[units[0].start_char:units[0].end_char] == units[0].text


def test_v4_cluster_apply_rejects_stale_offsets():
    unit = ClusterRepairUnit(
        cluster_id="cluster_001",
        start_sentence=1,
        end_sentence=1,
        start_char=6,
        end_char=19,
        text="target slice.",
        before_context="Intro ",
        after_context=" Outro.",
        sentence_count=1,
        word_count=2,
        risk_score=4.0,
    )

    candidate, status = apply_cluster_variant(
        "Intro stale slice. Outro.",
        unit,
        "replacement slice.",
    )

    assert candidate == "Intro stale slice. Outro."
    assert not status["applied"]
    assert status["reason"] == "cluster_slice_mismatch"


def test_v4_cluster_prompt_is_bounded_and_schema_small():
    unit = ClusterRepairUnit(
        cluster_id="cluster_001",
        start_sentence=4,
        end_sentence=6,
        start_char=10,
        end_char=80,
        text="At the beginning, he was quiet. With guidance, he became more confident.",
        before_context="",
        after_context=" Later he joined class independently.",
        sentence_count=2,
        word_count=12,
        risk_score=8.5,
        metadata={"generic_hits": 0, "transition_count": 0},
    )

    prompt = build_cluster_generator_prompt(unit=unit, variant_count=2)
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "bounded_cluster_texture_repair"
    assert payload["repair_unit"]["cluster_text"] == unit.text
    assert "full paragraph rewrite" in payload["forbidden"]
    assert len(payload["output_schema"]["variants"]) == 2
    assert set(payload["output_schema"]["variants"][0].keys()) == {"variant_id", "text"}


def test_v4_residual_cluster_prompt_targets_remaining_pocket():
    unit = ClusterRepairUnit(
        cluster_id="cluster_001",
        start_sentence=10,
        end_sentence=10,
        start_char=100,
        end_char=180,
        text="To me, the purpose of education is reflected in the Chinese saying and it is more important to teach students how to think.",
        before_context="",
        after_context="",
        sentence_count=1,
        word_count=22,
        risk_score=5.1,
        metadata={"generic_hits": 1, "transition_count": 0},
    )

    prompt = build_cluster_generator_prompt(unit=unit, variant_count=1, mode="residual_cluster_splitter")
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert payload["task"] == "residual_cluster_splitter"
    assert any("smallest route change" in item for item in payload["repair_goal"])
    assert any("split one over-complete sentence" in item for item in payload["allowed_moves"])
    assert any("Prefer reducing predictable common phrasing" in item for item in payload["constraints"])


def test_v4_residual_acceptance_uses_blocker_deltas():
    row = {
        "apply_statuses": [{"applied": True}],
        "scores": {
            "external_delta": -0.05,
            "rank_delta": -0.2,
            "topk_calibrated_risk_delta": 1.2,
            "qualifying_text_ai_density_delta": 0.0,
            "unsafe_cluster_delta": 0,
            "unsafe_word_ratio_delta": 0.0,
            "external_ai_flag_risk_delta": 0.0,
        },
    }

    assert _is_safe_residual_positive(row)


def test_v4_goal_driver_snapshot_and_deltas():
    baseline = {}
    candidate = {}
    baseline_goal = {
        "ai_footprint_gate": {
            "after": {
                "authorship_footprint": {
                    "ai_authorship": 40.0,
                    "topk_calibrated_risk": 50.0,
                },
                "semantic_footprint": {
                    "qualifying_text_ai_density": 45.0,
                },
                "external_ai_flag_risk": 36.0,
            }
        }
    }
    candidate_goal = {
        "ai_footprint_gate": {
            "after": {
                "authorship_footprint": {
                    "ai_authorship": 39.0,
                    "topk_calibrated_risk": 47.5,
                },
                "semantic_footprint": {
                    "qualifying_text_ai_density": 41.0,
                },
                "external_ai_flag_risk": 35.5,
            }
        }
    }

    _add_goal_driver_snapshot(baseline, baseline_goal)
    _add_goal_driver_snapshot(candidate, candidate_goal)
    _add_score_deltas(candidate, baseline)

    assert candidate["topk_calibrated_risk_delta"] == 2.5
    assert candidate["qualifying_text_ai_density_delta"] == 4.0
    assert candidate["ai_authorship_delta"] == 1.0
    assert candidate["external_ai_flag_risk_delta"] == 0.5
