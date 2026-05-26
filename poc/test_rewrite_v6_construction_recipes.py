from __future__ import annotations

import json

from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6.pipeline import run_v6_rewrite
from poc.rewrite_v6.planner_llm import build_planner_prompt
from poc.rewrite_v6.scan import scan_text
from poc.rewrite_v6.text import source_terms
from poc.rewrite_v6.write import build_prompt, parse_variants
from poc.rewrite_v6.coverage_guard import missing_required_source_terms


class StaticJsonResponse:
    def __init__(self, content: str):
        self.content = content
        self.raw_content = content


class StaticJsonClient:
    def __init__(self, content: str):
        self.content = content

    def chat(self, *args, **kwargs):
        return StaticJsonResponse(self.content)


class CaptureClient:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[tuple[tuple, dict]] = []

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return StaticJsonResponse(self.content)


class SequenceClient:
    def __init__(self, contents: list[str]):
        self.contents = contents
        self.calls: list[tuple[tuple, dict]] = []

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        return StaticJsonResponse(self.contents[index])


def _payload(text: str) -> dict:
    paragraph, plan = build_plan(scan_text(text))
    return json.loads(build_prompt(paragraph, plan).split("\n", 1)[1])


def test_v6_planner_emits_positive_construction_recipes():
    payload = _payload(
        "The process uses forms, queues, reviewers, feedback, decisions, and follow-up checks."
    )
    recipes = payload["construction_recipes"]
    assert recipes
    assert any(recipe["positive_pattern"] for recipe in recipes)
    assert any("separate source-relation beats" in recipe["build_route"] for recipe in recipes)
    assert "Use construction_recipes.repair_sequence in order as the positive build plan" in json.dumps(payload)


def test_v6_construction_recipes_carry_ordered_repair_sequence():
    payload = _payload(
        "This important result shows that teams manage forms, queues, review, follow-up checks, source files, client notes, audit timing, and decisions before making final decisions for clients across the weekly workflow."
    )
    recipes = payload["construction_recipes"]
    sequence = recipes[0]["repair_sequence"]
    classes = [row["repair_class"] for row in sequence]

    assert "predictable_start" in classes
    assert "context_author_anchor_gap" in classes
    assert "sentence_overload" in classes
    assert classes.index("predictable_start") < classes.index("sentence_overload")


def test_v6_coverage_beats_link_to_construction_recipes():
    payload = _payload(
        "This is an important concern because the process should improve across teams."
    )
    beats = payload["coverage_beats_must_all_appear"]
    assert beats
    assert all(beat.get("construction_recipe_id") for beat in beats)
    assert all(beat.get("construction_recipe") for beat in beats)
    assert any(
        "observable source relation" in " ".join(beat["construction_recipe"]["build_steps"])
        for beat in beats
    )


def test_v6_source_terms_keep_short_acronym_anchors():
    terms = source_terms("The team supports API, SSO and MFA issues across 12 sites.", limit=12)

    assert {"API", "SSO", "MFA"}.issubset(set(terms))
    assert "12" in terms


def test_v6_sentence_plan_marks_slot_coverage_loss_as_failure():
    payload = _payload(
        "The team combined intake forms, API checks, SSO review, and MFA support."
    )
    slots = payload["paragraph_sentence_plan"]
    slot_text = json.dumps(slots)

    assert "coverage_loss_failure" in slot_text
    assert "API" in slot_text
    assert "SSO" in slot_text
    assert "MFA" in slot_text


def test_v6_required_coverage_accepts_hyphenated_carried_terms():
    paragraph = scan_text(
        "The support plan used role playing, community based learning, ASD, and social anxieties."
    ).paragraphs[0]
    candidate = "The support plan used role-playing and community-based learning. ASD and social anxieties stayed in the support plan."

    assert not missing_required_source_terms(candidate, paragraph)


def test_v6_required_coverage_rejects_dropped_numeric_anchors():
    paragraph = scan_text(
        "The process uses seven checks while integrating four record structures from 0 to 180 degrees."
    ).paragraphs[0]
    candidate = "The process uses seven checks while integrating record structures across degrees."

    assert missing_required_source_terms(candidate, paragraph)


def test_v6_required_coverage_ignores_leading_heading_terms():
    paragraph = scan_text(
        "Critical Analysis\nThe process uses forms, queues, reviewers, and follow-up checks."
    ).paragraphs[0]
    candidate = "The process uses forms and queues. Reviewers handle follow-up checks."

    assert not missing_required_source_terms(candidate, paragraph)


def test_v6_overloaded_slots_with_many_terms_are_not_one_sentence_routes():
    payload = _payload(
        "The review process requires teams from several backgrounds to master seven intake procedures while integrating four record structures with audit concepts and workplace checks."
    )
    slot_text = json.dumps(payload["paragraph_sentence_plan"])

    assert "two connected ordinary sentences" in slot_text
    assert "required_sentence_groups" in slot_text
    assert "adjacent groups may share one ordinary sentence" in slot_text


def test_v6_overloaded_sentence_groups_are_chunked_for_execution():
    payload = _payload(
        "The process requires reviewers from several teams to master seven checks while integrating four record structures with audit concepts, workplace constraints, client notes, source documents, queue timing, and follow-up decisions."
    )
    groups = [
        group
        for slot in payload["paragraph_sentence_plan"]
        for group in slot.get("required_sentence_groups", [])
    ]

    assert len(groups) >= 2
    assert all(len(group["source_terms_to_carry"]) <= 10 for group in groups)
    assert "seven" in json.dumps(groups)
    assert "four" in json.dumps(groups)


def test_v6_planner_uses_semantic_relation_groups_before_term_chunks():
    payload = _payload(
        "Developing review awareness is more valuable than endless help teaching clients how to compare records is more effective than simply giving answers, only by equipping them with the skills to navigate the workflow will they adapt."
    )
    beat_ids = [
        beat["beat_id"]
        for beat in payload["coverage_beats_must_all_appear"]
    ]
    capsules = [
        beat["coverage_capsule"]
        for beat in payload["coverage_beats_must_all_appear"]
    ]

    assert len(beat_ids) >= 3
    assert any("Developing" in capsule and "valuable" in capsule for capsule in capsules)
    assert any("Teaching" in capsule and "effective" in capsule for capsule in capsules)
    assert any("equipping" in capsule and "workflow" in capsule for capsule in capsules)


def test_v6_heading_line_is_not_forced_into_coverage_beats():
    payload = _payload(
        "Heading\nAs a reviewer, I designed and implemented a support framework for case intake."
    )
    prompt_text = json.dumps(payload["coverage_beats_must_all_appear"])

    assert "Heading" not in prompt_text
    assert "reviewer" in prompt_text


def test_v6_multiword_section_heading_is_not_forced_into_coverage_beats():
    payload = _payload(
        "Critical Analysis: The Trap of Over-Accommodation vs CBT Standard\nAnalysing this situation through the framework of Competency-Based Training reveals a distinction."
    )
    prompt_text = json.dumps(payload["coverage_beats_must_all_appear"])

    assert "Over-Accommodation" not in prompt_text
    assert "Competency-Based" in prompt_text


def test_v6_heading_stripped_demonstrative_opener_becomes_route_target():
    payload = _payload(
        "Conclusion\nThis reflection shows that teams need forms, queues, review, and follow-up checks."
    )
    sequence = payload["construction_recipes"][0]["repair_sequence"]
    classes = [row["repair_class"] for row in sequence]

    assert "predictable_start" in classes
    assert "context_author_anchor_gap" in classes


def test_v6_required_sentence_group_keeps_all_group_terms():
    payload = _payload(
        "This unit requires learners from several backgrounds to master seven procedures while integrating four structures with geometric concepts and workplace checks for assessment."
    )
    groups = [
        group
        for slot in payload["paragraph_sentence_plan"]
        for group in slot.get("required_sentence_groups", [])
    ]

    grouped_terms = {
        str(term).casefold()
        for group in groups
        for term in group.get("source_terms_to_carry", [])
    }
    assert "workplace" in grouped_terms
    assert "checks" in grouped_terms


def test_v6_sentence_plan_separates_exact_anchors_from_revoiceable_terms():
    payload = _payload(
        "The SHBHCUT006 unit presents significant pedagogical challenges while learners master seven cutting procedures and integrate four haircut structures with geometric concepts."
    )
    text = json.dumps(payload["paragraph_sentence_plan"])

    assert "SHBHCUT006" in text
    assert "seven" in text
    assert "four" in text
    assert "revoiceable_source_terms" in text
    assert "significant" in text
    assert any(
        "four" in [str(term).casefold() for term in slot.get("must_cover_terms", [])]
        for slot in payload["paragraph_sentence_plan"]
    )


def test_v6_writer_schema_requires_recipe_id_in_coverage_map():
    payload = _payload("This process shows a concern because support should improve.")
    schema_text = json.dumps(payload["output_schema"])
    assert "construction_recipe_id" in schema_text


def test_v6_writer_compiles_text_from_sentence_rows():
    variants = parse_variants({
        "variants": [
            {
                "id": "v1",
                "text": "A polished paragraph recombined the rows.",
                "sentence_rows": [
                    {"sentence_slot_id": "s1", "coverage_beat_ids": ["b1"], "sentence": "The first row stays separate"},
                    {"sentence_slot_id": "s2", "coverage_beat_ids": ["b2"], "sentence": "The second row also stays separate."},
                ],
            }
        ]
    })

    assert variants[0].text == "The first row stays separate. The second row also stays separate."


def test_v6_writer_compiles_text_from_coverage_map_when_rows_missing():
    variants = parse_variants({
        "variants": [
            {
                "id": "v1",
                "text": "A fallback paragraph should not win.",
                "coverage_map": [
                    {"sentence_slot_id": "s1", "coverage_beat_ids": ["b1"], "sentence": "Coverage row one."},
                    {"sentence_slot_id": "s2", "coverage_beat_ids": ["b2"], "sentence": "Coverage row two."},
                ],
            }
        ]
    })

    assert variants[0].text == "Coverage row one. Coverage row two."


def test_v6_row_compiler_concretizes_demonstrative_noun_starts():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [
                {"sentence_slot_id": "s1", "coverage_beat_ids": ["b1"], "sentence": "This reflection confirmed the support route."},
                {"sentence_slot_id": "s2", "coverage_beat_ids": ["b2"], "sentence": "These pathways help learners practise."},
            ],
        }]
    })

    assert variants[0].text == "The reflection confirmed the support route. The pathways help learners practise."


def test_v6_row_compiler_removes_near_duplicate_sentence_intent():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [
                {"sentence_slot_id": "s1", "coverage_beat_ids": ["b1"], "sentence": "Every learner has the right to learn and be supported seriously."},
                {"sentence_slot_id": "s2", "coverage_beat_ids": ["b2"], "sentence": "Every learner has the right to learn and be supported seriously."},
            ],
        }]
    })

    assert variants[0].text == "Every learner has the right to learn and be supported seriously."


def test_v6_row_compiler_repairs_simple_context_starts():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [
                {"sentence_slot_id": "s1", "coverage_beat_ids": ["b1"], "sentence": "Every learner has the right to learn."},
                {"sentence_slot_id": "s2", "coverage_beat_ids": ["b2"], "sentence": "This applies regardless of slower learning."},
                {"sentence_slot_id": "s3", "coverage_beat_ids": ["b3"], "sentence": "It is a teaching approach that helps learners."},
            ],
        }]
    })

    assert variants[0].text == (
        "Every learner has the right to learn. "
        "The right to learn applies regardless of slower learning. "
        "The teaching approach helps learners."
    )


def test_v6_row_compiler_splits_regardless_comma_lists():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1", "b2"],
                "sentence": "The right to learn applies regardless of disabilities, slower learning speeds, or the need for different ways to understand things.",
            }],
        }]
    })

    assert variants[0].text == (
        "The right to learn applies regardless of disabilities. "
        "The right to learn also covers slower learning speeds and the need for different ways to understand things."
    )


def test_v6_row_compiler_splits_demonstrative_regardless_lists_without_new_this_start():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1", "b2"],
                "sentence": "This applies regardless of disabilities, slower learning speeds, or the need for different ways to understand things.",
            }],
        }]
    })

    assert variants[0].text == (
        "This applies regardless of disabilities. "
        "The same support also covers slower learning speeds and the need for different ways to understand things."
    )


def test_v6_row_compiler_splits_whether_they_regardless_lists_grammatically():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1", "b2"],
                "sentence": "The right to learn applies regardless of whether they have disabilities, learn slower, or need different ways to understand things.",
            }],
        }]
    })

    assert variants[0].text == (
        "The right to learn applies regardless of whether they have disabilities. "
        "The right to learn still applies when they learn slower or need different ways to understand things."
    )


def test_v6_row_compiler_splits_allow_gradual_build_rows():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1", "b2"],
                "sentence": "The pathways should allow them to make mistakes and gradually build their own skills and confidence.",
            }],
        }]
    })

    assert variants[0].text == (
        "The pathways should allow them to make mistakes. "
        "The same pathways help them gradually build their own skills and confidence."
    )


def test_v6_row_compiler_repairs_connector_fragments_with_previous_subject():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [
                {"sentence_slot_id": "s1", "coverage_beat_ids": ["b1"], "sentence": "The core responsibility is not merely to impart knowledge."},
                {"sentence_slot_id": "s2", "coverage_beat_ids": ["b2"], "sentence": "Rather stimulate learners motivation."},
                {"sentence_slot_id": "s3", "coverage_beat_ids": ["b3"], "sentence": "Thereby facilitating a shift from interest to passion."},
            ],
        }]
    })

    assert variants[0].text == (
        "The core responsibility is not merely to impart knowledge. "
        "The same responsibility stimulates learners motivation. "
        "The same responsibility facilitates a shift from interest to passion."
    )


def test_v6_row_compiler_splits_common_academic_connectors():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [
                {
                    "sentence_slot_id": "s1",
                    "coverage_beat_ids": ["b1", "b2"],
                    "sentence": "According to source theory, complex spatial tasks place strain on working memory.",
                },
                {
                    "sentence_slot_id": "s2",
                    "coverage_beat_ids": ["b3", "b4"],
                    "sentence": "The plan must move beyond passive support and instead focus on practice.",
                },
            ],
        }]
    })

    assert variants[0].text == (
        "Source theory supports this point. "
        "Complex spatial tasks place strain on working memory. "
        "The plan must move beyond passive support. "
        "The plan should focus on practice."
    )


def test_v6_row_compiler_splits_under_obligation_rows():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:abstract_density",
                "sentence": "Under the service standard, providers have a legal obligation to provide reasonable adjustment.",
            }],
        }]
    })

    assert variants[0].text == (
        "The service standard applies to providers. "
        "Providers must provide reasonable adjustment."
    )


def test_v6_row_compiler_forces_repair_for_scanner_owned_single_beat_rows():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:citation_anchor",
                "sentence": "According to source theory, complex spatial tasks place strain on working memory.",
            }],
        }]
    })

    assert variants[0].text == (
        "Source theory supports this point. "
        "Complex spatial tasks place strain on working memory."
    )


def test_v6_row_compiler_splits_must_not_undermine_pairs():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:sentence_overload",
                "sentence": "The adjustment must not undermine core requirements or performance standards of the package.",
            }],
        }]
    })

    assert variants[0].text == (
        "The adjustment must not undermine core requirements. "
        "The adjustment must also preserve performance standards of the package."
    )


def test_v6_row_compiler_splits_such_as_strain_rows():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:paraphrase_smoothing",
                "sentence": "Complex spatial tasks such as constructing a structure place significant strain on working memory.",
            }],
        }]
    })

    assert variants[0].text == (
        "Constructing a structure is part of complex spatial tasks. "
        "The task places significant strain on working memory."
    )


def test_v6_row_compiler_splits_such_as_strain_with_trailing_source_frame():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "sentence": "Complex spatial tasks such as constructing a structure place significant strain on working memory according to source theory.",
            }],
        }]
    })

    assert variants[0].text == (
        "Constructing a structure is part of complex spatial tasks. "
        "Source theory links the task to significant strain on working memory."
    )


def test_v6_row_compiler_splits_demonstrated_ability_needs_rows():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:sentence_overload",
                "sentence": "He demonstrated the ability to accurately identify and gently address the needs of clients who struggle to explain the issue.",
            }],
        }]
    })

    assert variants[0].text == (
        "He accurately identified the needs of clients. "
        "He gently addressed those needs. "
        "Clients struggle to explain the issue."
    )


def test_v6_row_compiler_splits_subject_remained_pairs():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:paraphrase_smoothing",
                "sentence": "The assessment criteria, safety and timing have still remained consistent for all learners.",
            }],
        }]
    })

    assert variants[0].text == "The assessment criteria have still remained consistent for all learners."


def test_v6_row_compiler_splits_in_context_comma_openers():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:predictable_start",
                "sentence": "In service training, learner performance is assessed against industry benchmarks.",
            }],
        }]
    })

    assert variants[0].text == "In service training, learner performance is assessed against industry benchmarks."


def test_v6_row_compiler_splits_more_valuable_and_equipping_rows():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [
                {
                    "sentence_slot_id": "s1",
                    "coverage_beat_ids": ["b1"],
                    "finding_contract_id": "s1:paraphrase_smoothing",
                    "sentence": "Developing awareness is more valuable than endless help.",
                },
                {
                    "sentence_slot_id": "s2",
                    "coverage_beat_ids": ["b2"],
                    "finding_contract_id": "s2:sentence_overload",
                    "sentence": "Equipping them with skills to navigate the workplace will enable them to adapt to workplace demands when they enter the workplace.",
                },
            ],
        }]
    })

    assert variants[0].text == (
        "Developing awareness matters more than endless help. "
        "Endless help is not enough by itself. "
        "They need skills to navigate the workplace. "
        "The skills help them adapt to workplace demands. "
        "The same support matters when they enter the workplace."
    )


def test_v6_row_compiler_splits_move_beyond_to_by_rows_without_force():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "sentence": "The plan must move beyond passive support to bridge policy and practice by embedding review routines instead.",
            }],
        }]
    })

    assert variants[0].text == (
        "The plan must move beyond passive support. "
        "The purpose is to bridge policy and practice. "
        "The method uses review routines."
    )


def test_v6_row_compiler_repairs_do_not_lie_in_nominalization():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:paraphrase_smoothing",
                "sentence": "The demonstration that client support needs do not lie in special treatment.",
            }],
        }]
    })

    assert variants[0].text == "Client support needs are not based on special treatment."


def test_v6_row_compiler_repairs_short_do_not_lie_in_rows_without_force():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "sentence": "The demonstration shows that client support needs do not lie in treatment based on pity.",
            }],
        }]
    })

    assert variants[0].text == "Client support needs are not treatment or pity."


def test_v6_row_compiler_repairs_nested_based_on_after_do_not_lie_in():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:paraphrase_smoothing",
                "sentence": "The demonstration that client support needs do not lie in treatment based on pity.",
            }],
        }]
    })

    assert variants[0].text == "Client support needs are not treatment or pity."


def test_v6_row_compiler_removes_demonstration_shows_do_not_lie_wrapper():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:paraphrase_smoothing",
                "sentence": "The demonstration shows that client support needs do not lie in treatment based on pity.",
            }],
        }]
    })

    assert variants[0].text == "Client support needs are not treatment or pity."


def test_v6_row_compiler_plainifies_metacognitive_nominalizations():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [
                {
                    "sentence_slot_id": "s1",
                    "coverage_beat_ids": ["b1"],
                    "finding_contract_id": "s1:abstract_density",
                    "sentence": "The discussion gives learners time for metacognitive monitoring.",
                },
                {
                    "sentence_slot_id": "s2",
                    "coverage_beat_ids": ["b2"],
                    "finding_contract_id": "s2:paraphrase_smoothing",
                    "sentence": "Developing metacognitive awareness matters more than endless help.",
                },
            ],
        }]
    })

    assert variants[0].text == (
        "The discussion gives learners time to monitor their thinking. "
        "Developing awareness of how they learn matters more than endless help."
    )


def test_v6_row_compiler_splits_for_context_matters_more_than_rows():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:paraphrase_smoothing",
                "sentence": "For clients under pressure, developing awareness matters more than endless help.",
            }],
        }]
    })

    assert variants[0].text == (
        "Developing awareness matters for clients under pressure. "
        "Endless help is not enough by itself."
    )


def test_v6_row_compiler_splits_for_context_is_more_valuable_than_rows():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:paraphrase_smoothing",
                "sentence": "For clients under pressure, developing awareness is more valuable than endless help.",
            }],
        }]
    })

    assert variants[0].text == (
        "Developing awareness matters for clients under pressure. "
        "Endless help is not enough by itself."
    )


def test_v6_row_compiler_splits_not_by_expectation_when_rows():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:context_anchor_gap",
                "sentence": "They will not be able to do so by expecting the workplace to change when they enter the role.",
            }],
        }]
    })

    assert variants[0].text == "When they enter the role, the same people cannot rely on the workplace to change."


def test_v6_row_compiler_merges_dangling_row_endings():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "sentence_rows": [
                {"sentence": "The teacher replaced dense jargon with."},
                {"sentence": "The Octagon method."},
                {"sentence": "The client would not accept the delay simply."},
                {"sentence": "The service is incomplete."},
                {"sentence": "Maintaining standards is not unfair on the other hand."},
                {"sentence": "It protects learners."},
                {"sentence": "I adapted."},
                {"sentence": "The scaffolding approach."},
            ],
        }]
    })

    assert variants[0].text == (
        "The teacher replaced dense jargon with the Octagon method. "
        "The client would not accept the delay simply because the service is incomplete. "
        "Maintaining standards is not unfair on the other hand, it protects learners. "
        "I adapted the scaffolding approach."
    )


def test_v6_row_compiler_keeps_it_is_predicate_complete():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "sentence_rows": [{"sentence": "It is a fundamental ethical obligation towards learners."}],
        }]
    })

    assert variants[0].text == "It is a fundamental ethical obligation towards learners."


def test_v6_row_compiler_repairs_they_need_after_learner_context():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [
                {"sentence_slot_id": "s1", "coverage_beat_ids": ["b1"], "sentence": "The support focuses on learners."},
                {"sentence_slot_id": "s2", "coverage_beat_ids": ["b2"], "sentence": "They need the skills to navigate the workplace."},
            ],
        }]
    })

    assert variants[0].text == "The support focuses on learners. Learners need the skills to navigate the workplace."


def test_v6_row_compiler_splits_comma_scaffold_rows():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1"],
                "finding_contract_id": "s1:packed_list",
                "sentence": "The progress was tied to purpose, confidence, and practice.",
            }],
        }]
    })

    assert variants[0].text == (
        "The progress was tied to purpose. "
        "Confidence and practice carry the same point."
    )


def test_v6_row_compiler_skips_repeated_explanation_but_keeps_not_enough_contrast():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [
                {"sentence_slot_id": "s1", "coverage_beat_ids": ["b1"], "sentence": "The language barrier affected these clients."},
                {"sentence_slot_id": "s2", "coverage_beat_ids": ["b2"], "sentence": "Language barrier was a challenge for these clients."},
                {"sentence_slot_id": "s3", "coverage_beat_ids": ["b3"], "sentence": "Developing awareness matters more than endless help."},
                {"sentence_slot_id": "s4", "coverage_beat_ids": ["b4"], "sentence": "Endless help is not enough by itself."},
            ],
        }]
    })

    assert variants[0].text == (
        "The language barrier affected these clients. "
        "Developing awareness matters more than endless help. "
        "Endless help is not enough by itself."
    )


def test_v6_row_compiler_splits_overloaded_connector_rows():
    variants = parse_variants({
        "variants": [
            {
                "id": "v1",
                "coverage_map": [
                    {
                        "sentence_slot_id": "s1",
                        "coverage_beat_ids": ["b1", "b2"],
                        "sentence": (
                            "The unit presents significant challenges because it requires learners "
                            "to master seven procedures while integrating four structures."
                        ),
                    }
                ],
            }
        ]
    })

    assert variants[0].text == (
        "The unit presents significant challenges. "
        "The unit requires learners to master seven procedures. "
        "The unit involves integrating four structures."
    )


def test_v6_row_compiler_splits_first_person_working_to_rows():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1", "b2"],
                "sentence": "I am a reviewer in the service team working to design and implement a support framework.",
            }],
        }]
    })

    assert variants[0].text == (
        "I am a reviewer in the service team. "
        "I am working to design and implement a support framework."
    )


def test_v6_row_compiler_splits_support_clause_rows():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1", "b2"],
                "sentence": "I integrated role playing and community learning to support the learner with ASD.",
            }],
        }]
    })

    assert variants[0].text == (
        "I integrated role playing and community learning. "
        "The same support focused on the learner with ASD."
    )


def test_v6_row_compiler_rebuilds_framework_working_row():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1", "b2"],
                "sentence": "I am a reviewer in the service team working to design and implement an inclusive framework for the audit unit that combines records.",
            }],
        }]
    })

    assert variants[0].text == (
        "I am a reviewer in the service team. "
        "The audit unit uses an inclusive framework. "
        "The audit unit combines records."
    )


def test_v6_row_compiler_rebuilds_evaluates_between_row():
    variants = parse_variants({
        "variants": [{
            "id": "v1",
            "coverage_map": [{
                "sentence_slot_id": "s1",
                "coverage_beat_ids": ["b1", "b2"],
                "sentence": "The report critically evaluates the tension between support adjustments and competency standards required by the training package.",
            }],
        }]
    })

    assert variants[0].text == (
        "The report compares support adjustments with competency standards. "
        "The second side is required by the training package."
    )


def test_v6_llm_planner_prompt_receives_scanner_findings():
    scan = scan_text("This result shows a concern because the process should improve.")
    paragraph, plan = build_plan(scan)
    prompt = build_planner_prompt(paragraph, plan, scan.findings)
    payload = json.loads(prompt.split("\n", 1)[1])
    assert payload["scanner_findings"]
    assert payload["deterministic_route_skeleton"]["construction_recipes"]
    assert payload["required_decision"]["finding_contracts"]
    assert payload["required_decision"]["finding_recipe_overrides"]
    assert payload["required_decision"]["paragraph_blueprint"]
    assert "Return one finding_contract for every scanner_findings row" in prompt
    assert "Do not use placeholder-only safe shapes" in prompt
    assert "actual submitted source terms" in prompt
    assert "planning labels" in prompt
    assert "scoped partial-relation shape" in prompt
    assert "concrete enough that a writer can follow it without guessing" in prompt
    assert "Do not write replacement paragraph prose" in prompt


def test_v6_pipeline_calls_planner_before_writer_when_supplied():
    planner_payload = json.dumps({
        "planner_decision": {
            "paragraph_route": "Use source relation before claim.",
            "finding_contracts": [
                {
                    "finding_id": "p001_s001:predictable_start",
                    "source_sentence_id": "p001_s001",
                    "finding_tags": ["predictable_start"],
                    "unsafe_original_shape": "This result shows",
                    "safe_rebuild_shape": "The process shows a narrow concern.",
                    "writer_must_do": ["start from source object"],
                    "writer_must_not_do": ["reuse This result shows"],
                    "coverage_terms": ["process"],
                }
            ],
            "paragraph_blueprint": [
                {
                    "step_id": "b001",
                    "function": "start from the source object",
                    "source_basis": ["p001_s001"],
                    "must_include": ["process"],
                    "must_avoid_shape": ["This result shows"],
                    "safe_sentence_shape": "<source object> shows <narrow relation>",
                }
            ],
            "finding_recipe_overrides": [
                {
                    "source_sentence_id": "p001_s001",
                    "safe_route": "anchor first",
                    "build_steps": ["name source object first"],
                    "positive_pattern": "<object> shows <relation>",
                }
            ],
            "author_proxy_plan": "mark inferred bridges",
            "do_not_copy_route": ["same opener"],
        }
    })
    writer_payload = json.dumps({"variants": []})
    planner = CaptureClient(planner_payload)
    writer = CaptureClient(writer_payload)
    result = run_v6_rewrite(
        "This result shows a concern because the process should improve.",
        planner_client=planner,
        writer_client=writer,
    )
    assert len(planner.calls) == 1
    assert len(writer.calls) == 1
    writer_prompt = writer.calls[0][0][0]
    assert "Use source relation before claim" in writer_prompt
    assert "p001_s001:predictable_start" in writer_prompt
    assert "finding_contracts and document_signal_contracts as the primary build contract" in writer_prompt
    assert "planner_decision.contract_gaps" in writer_prompt
    assert "quoted phrase must not appear" in writer_prompt
    assert "If safe_rebuild_shape contains placeholder brackets" in writer_prompt
    assert "Do not copy planning labels" in writer_prompt
    assert "scoped partial-relation sentence" in writer_prompt
    assert "safe_sentence_shape" in writer_prompt
    assert result.plan.ai_safe_route["llm_planner_decision"]["status"] == "ok"


def test_v6_planner_records_contract_gap_when_contract_copies_risky_source_route():
    payload = json.dumps({
        "planner_decision": {
            "paragraph_route": "bad",
            "finding_contracts": [
                {
                    "finding_id": "p001_s001:broad_claim",
                    "source_sentence_id": "p001_s001",
                    "finding_tags": ["broad_claim"],
                    "unsafe_original_shape": "This model no longer fully reflects how people learn.",
                    "safe_rebuild_shape": "This model no longer fully reflects how people learn.",
                    "writer_must_do": ["copy"],
                    "writer_must_not_do": [],
                    "coverage_terms": ["model"],
                }
            ],
            "paragraph_blueprint": [],
        }
    })
    writer = StaticJsonClient(json.dumps({"variants": []}))
    planner = CaptureClient(payload)
    result = run_v6_rewrite("This model no longer fully reflects how people learn.", planner_client=planner, writer_client=writer)
    assert len(planner.calls) == 1
    assert result.plan.ai_safe_route["llm_planner_decision"]["contract_gaps"]


def test_v6_planner_records_contract_gap_when_contract_uses_planning_labels():
    payload = json.dumps({
        "planner_decision": {
            "paragraph_route": "bad",
            "finding_contracts": [
                {
                    "finding_id": "p001_s001:packed_list",
                    "source_sentence_id": "p001_s001",
                    "finding_tags": ["packed_list"],
                    "unsafe_original_shape": "A, B, and C",
                    "safe_rebuild_shape": "A carries one relation and B carries the next relation.",
                    "writer_must_do": ["split relation"],
                    "writer_must_not_do": [],
                    "coverage_terms": ["A", "B"],
                }
            ],
            "paragraph_blueprint": [],
        }
    })
    planner = CaptureClient(payload)
    writer = StaticJsonClient(json.dumps({"variants": []}))
    result = run_v6_rewrite("A, B, and C shaped the process.", planner_client=planner, writer_client=writer)
    assert len(planner.calls) == 1
    assert result.plan.ai_safe_route["llm_planner_decision"]["contract_gaps"]
