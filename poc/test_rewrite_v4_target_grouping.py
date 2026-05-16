import json

from rewrite_v3.target_executor import group_rewrite_targets
from rewrite_v4.generator import build_generator_prompt
from rewrite_v4.normalizer import deterministic_repair_brief


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
    prompt = build_generator_prompt(group=group, repair_brief=brief, variant_count=2)
    payload = json.loads(prompt.removeprefix("Return valid JSON only.\n"))

    assert "Keep one paragraph." not in payload["constraints"]
    assert "headings" not in payload["avoid"]
    assert "new headings" in payload["avoid"]
    assert any("keep the first line as the heading" in item for item in payload["constraints"])
