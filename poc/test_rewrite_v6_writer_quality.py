from __future__ import annotations

from poc.rewrite_v6.prose_quality import catalogue_sentence_chain, repair_generated_prose, robotic_sentence_chain
from poc.rewrite_v6.prose_quality import has_fragment_or_trace_sentences
from poc.rewrite_v6.scan import scan_text
from poc.rewrite_v6.selector_diagnostics import selection_diagnostics
from poc.rewrite_v6.integrity_guard import candidate_integrity_blockers, candidate_integrity_warnings
from poc.rewrite_v6.write import Variant, _candidate_contract_violation, choose_variant


def test_v6_writer_quality_marks_robotic_sentence_chains_as_soft_warnings():
    source = (
        "Now, students are surrounded by information. They learn from teachers, but also from YouTube, "
        "TikTok, online courses, AI tools, search engines, social media, and peer communities. "
        "This has created a new kind of learning environment."
    )
    candidate = (
        "Students are surrounded by information. Students learn from teachers. "
        "Students also learn from YouTube. Students also learn from TikTok. "
        "Students also learn from online courses. Students also learn from AI tools. "
        "Students also learn from search engines. Students also learn from social media. "
        "Students also learn from peer communities. "
        "The result has created a new kind of learning environment."
    )
    paragraph = scan_text(source).paragraphs[0]
    variants = [
        Variant(id="source_preserved", text=source, source="source_preserved"),
        Variant(id="v1", text=candidate, source="llm"),
    ]

    assert robotic_sentence_chain(candidate)
    assert _candidate_contract_violation(candidate, paragraph)
    diagnostics = selection_diagnostics(variants, paragraph)[0]
    assert "mechanical_sentence_chain" in diagnostics["quality_warnings"]
    assert "candidate_contract_warning" in diagnostics["quality_warnings"]
    assert "mechanical_sentence_chain" in diagnostics["blockers"]
    assert choose_variant(variants, paragraph).source == "source_preserved"


def test_v6_writer_quality_blocks_external_narrator_reporting_chains():
    source = (
        "Students are surrounded by information. They learn from teachers, YouTube, TikTok, online courses, "
        "AI tools, search engines, social media, and peer communities."
    )
    candidate = (
        "The writer observes that students are surrounded by information. "
        "He notes that they learn from teachers as well as from YouTube and TikTok. "
        "He adds that online courses and AI tools broaden their options."
    )
    paragraph = scan_text(source).paragraphs[0]
    variants = [
        Variant(id="source_preserved", text=source, source="source_preserved"),
        Variant(id="v1", text=candidate, source="llm"),
    ]

    assert "external_narrator_reporting_chain" in candidate_integrity_blockers(candidate)
    diagnostics = selection_diagnostics(variants, paragraph)[0]
    assert "external_narrator_reporting_chain" in diagnostics["blockers"]
    assert choose_variant(variants, paragraph).source == "source_preserved"


def test_v6_writer_quality_blocks_unsupported_semantic_padding():
    source = (
        "Now, students are surrounded by information. They learn from teachers, but also from YouTube, "
        "TikTok, online courses, AI tools, search engines, social media, and peer communities. "
        "This has created a new kind of learning environment. Knowledge is no longer scarce. "
        "Access is no longer the biggest problem. The real challenge is knowing what is accurate, useful, ethical, and worth trusting."
    )
    candidate = (
        "Students are surrounded by a flood of information from digital feeds, news outlets, and other media. "
        "These sources include textbooks and news outlets as well as online feeds. "
        "They also learn from teachers, YouTube, TikTok, online courses, AI tools, search engines, social media, and peer communities."
    )
    paragraph = scan_text(source).paragraphs[0]

    row = selection_diagnostics(
        [
            Variant(id="source_preserved", text=source, source="source_preserved"),
            Variant(id="v1", text=candidate, source="llm"),
        ],
        paragraph,
    )[0]

    assert "unsupported_semantic_padding" in row["blockers"]
    assert row["accepted_by_selector"] is False


def test_v6_writer_quality_blocks_multisentence_not_only_inversion():
    source = (
        "The assessment does not only reward learners who remember facts. "
        "It rewards learners who analyse, adapt, communicate, and create."
    )
    candidate = (
        "The assessment rewards learners who remember facts rather than those who analyse, "
        "adapt, communicate, and create."
    )
    paragraph = scan_text(source).paragraphs[0]

    row = selection_diagnostics(
        [
            Variant(id="source_preserved", text=source, source="source_preserved"),
            Variant(id="v1", text=candidate, source="llm"),
        ],
        paragraph,
    )[0]

    assert "source_polarity_inversion" in row["blockers"]
    assert row["accepted_by_selector"] is False


def test_v6_writer_quality_blocks_narrator_route_with_varied_reporting_verbs():
    candidate = (
        "The writer sees that students are immersed in a flood of information while still relying on teachers. "
        "He reads that their learning now includes YouTube, TikTok, online courses and AI tools. "
        "Finally, he decides that identifying accurate and useful content matters most."
    )

    assert "external_narrator_reporting_chain" in candidate_integrity_blockers(candidate)


def test_v6_writer_quality_blocks_short_fragment_chain():
    text = "Students may learn how to pass. But not always how to think deeply. Solve problems. Or connect ideas across subjects."

    assert has_fragment_or_trace_sentences(text)


def test_v6_selector_diagnostics_expose_handoff_scope_failure_evidence():
    source = "Students may learn how to pass, but not always how to think deeply, solve problems, or connect ideas across subjects."
    candidate = "Students may always learn how to pass exams but miss deeper thinking and problem solving."
    paragraph = scan_text(source).paragraphs[0]

    row = selection_diagnostics(
        [
            Variant(id="source_preserved", text=source, source="source_preserved"),
            Variant(id="v1", text=candidate, source="llm"),
        ],
        paragraph,
    )[0]

    assert "not_always_scope_inversion" in row["blockers"]
    assert row["handoff_validation"]["planner_to_writer_contract"] == "validated"
    assert row["handoff_validation"]["writer_to_selector_candidate"] == "failed"
    assert row["handoff_validation"]["selector_gate"] == "blocked"
    assert row["handoff_validation"]["source_scope_markers"] == ["not always"]


def test_v6_not_always_scope_accepts_preserved_negation_and_blocks_malformed_order():
    source = "Students may learn how to pass, but not always how to think deeply, solve problems, or connect ideas across subjects."
    valid = "Students may learn how to pass exams, but they do not always develop deeper thinking or problem-solving across subjects."
    malformed = "Students may learn how to pass, but they not always think deeply or solve problems."
    paragraph = scan_text(source).paragraphs[0]

    valid_row = selection_diagnostics(
        [Variant(id="source_preserved", text=source, source="source_preserved"), Variant(id="v1", text=valid, source="llm")],
        paragraph,
    )[0]
    malformed_row = selection_diagnostics(
        [Variant(id="source_preserved", text=source, source="source_preserved"), Variant(id="v2", text=malformed, source="llm")],
        paragraph,
    )[0]

    assert "not_always_scope_inversion" not in valid_row["blockers"]
    assert "malformed_negation_order" in malformed_row["blockers"]
    assert "they do not always" in repair_generated_prose(malformed, source)


def test_v6_writer_quality_blocks_since_subordinate_fragment():
    text = (
        "The harder task is deciding which information is accurate. "
        "Since effective judgment shapes the value of abundant information."
    )

    assert has_fragment_or_trace_sentences(text)


def test_v6_writer_quality_blocks_malformed_nonhuman_activity_predicates():
    source = (
        "Students learn from teachers, YouTube, TikTok, online courses, AI tools, search engines, social media, and peer communities. "
        "This has created a new kind of learning environment."
    )
    candidate = (
        "Teachers, YouTube and TikTok are learning. "
        "Online courses and AI tools are learning. "
        "Created kind learning environment is present."
    )
    paragraph = scan_text(source).paragraphs[0]
    variants = [
        Variant(id="source_preserved", text=source, source="source_preserved"),
        Variant(id="v1", text=candidate, source="llm"),
    ]

    blockers = candidate_integrity_blockers(candidate)

    assert "malformed_nonhuman_activity_predicate" in blockers
    assert "malformed_telegraphic_predicate" in blockers
    diagnostics = selection_diagnostics(variants, paragraph)[0]
    assert "malformed_nonhuman_activity_predicate" in diagnostics["blockers"]
    assert "malformed_telegraphic_predicate" in diagnostics["blockers"]
    assert choose_variant(variants, paragraph).source == "source_preserved"


def test_v6_writer_quality_blocks_malformed_parallel_connector_lists():
    text = (
        "They continue to learn from teachers while also drawing on YouTube and TikTok "
        "as well as online courses, AI tools, search engines and social media and peer communities."
    )

    assert "malformed_parallel_connector_list" in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_together_with_connector_list():
    text = (
        "Their learning now extends to platforms such as YouTube and TikTok together with online courses, "
        "AI tools, search engines and social media and peer communities."
    )

    assert "malformed_parallel_connector_list" in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_malformed_parallel_verb_tail():
    text = "AI tools help students brainstorm ideas and explain difficult topics, improve writing and practise skills."

    assert "malformed_parallel_verb_tail" in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_repeated_subject_and_unintroduced_reliance():
    text = (
        "AI tools help students brainstorm ideas. "
        "AI tools improve writing and practise skills. "
        "A danger also emerges from this reliance. "
        "The situation makes assessment more difficult. "
        "The situation raises questions about fairness."
    )

    blockers = candidate_integrity_blockers(text)
    warnings = candidate_integrity_warnings(text)

    assert "repeated_subject_start" in warnings
    assert "vague_unintroduced_reliance" in blockers


def test_v6_writer_quality_blocks_tool_actor_semantic_role_defects():
    text = (
        "AI tools and students belong together as the tools provide help. "
        "The same tools improve writing and practise skills."
    )

    blockers = candidate_integrity_blockers(text)

    assert "malformed_tool_actor_relation" in blockers
    assert "malformed_tool_skill_predicate" in blockers


def test_v6_writer_quality_blocks_modal_do_negation():
    text = "Students may do not always acquire the capacity to think deeply."

    assert "malformed_modal_do_negation" in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_dangling_terminal_and_tail():
    text = "The real challenge is knowing what is accurate, useful, ethical, and worth trusting and AI tools."

    assert "dangling_terminal_and_tail" in candidate_integrity_blockers(text)


def test_v6_writer_quality_allows_students_as_practise_skills_actor():
    text = "These tools also improve writing and allow students to practise skills."

    assert "malformed_tool_skill_predicate" not in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_dangling_additive_tail():
    text = "Students remain surrounded by an ever-growing flow of information and additionally."

    assert "dangling_additive_tail" in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_standalone_additive_fragment():
    text = "Students also draw on YouTube and TikTok. As well as from online courses, AI tools, and search engines."

    assert "standalone_additive_fragment" in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_redundant_trust_phrases():
    text = "The real challenge is knowing what is accurate, useful, ethical, trustworthy and worth trusting."

    assert "redundant_trust_phrase" in candidate_integrity_blockers(text)


def test_v6_prose_repair_does_not_corrupt_valid_as_well_as_not_only_relation():
    source = (
        "This is a serious concern because the modern world does not only reward people "
        "who can remember facts. It rewards people who can analyse, adapt, communicate, and create."
    )
    candidate = (
        "This creates a serious concern, since the modern world rewards people who remember facts "
        "as well as those who analyse, adapt, communicate and create."
    )

    repaired = repair_generated_prose(candidate, source)

    assert "remember does not only fact" not in repaired
    assert "It also create" not in repaired
    assert repaired == candidate


def test_v6_writer_quality_blocks_keyword_dump_sequences():
    text = (
        "Students also learn from YouTube, TikTok, online courses, AI tools, search engines, social media, and peer communities. "
        "YouTube TikTok online courses AI tools search engines social media peer communities have created a kind of learning environment."
    )

    blockers = candidate_integrity_blockers(text)
    warnings = candidate_integrity_warnings(text)

    assert "keyword_dump_sequence" in blockers
    assert "lost_serial_punctuation" in blockers
    assert "repeated_platform_catalogue" in warnings


def test_v6_writer_quality_blocks_capitalized_common_noun_mid_sentence():
    text = "Knowledge is no longer scarce and Access is no longer the biggest problem."

    assert "capitalized_common_noun_mid_sentence" in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_lost_final_list_punctuation():
    text = "The real challenge is knowing what is accurate useful ethical worth trusting."

    assert "lost_serial_punctuation" in candidate_integrity_blockers(text)


def test_v6_selector_rejects_surface_corruption_before_scoring():
    source = (
        "Students are surrounded by information. "
        "They learn from teachers, YouTube, TikTok, online courses, AI tools, search engines, social media, and peer communities. "
        "This has created a new kind of learning environment. "
        "Knowledge is no longer scarce. Access is no longer the biggest problem. "
        "The real challenge is knowing what is accurate, useful, ethical, and worth trusting."
    )
    corrupted = Variant(
        id="retry_v1",
        source="llm",
        text=(
            "Students are surrounded by information and still learn from teachers. "
            "They also learn from YouTube and TikTok as well as online courses. "
            "YouTube TikTok online courses AI tools search engines social media peer communities have created a kind of learning environment. "
            "Knowledge is no longer scarce and Access is no longer the biggest problem. "
            "The real challenge is knowing what is accurate useful ethical worth trusting."
        ),
    )
    clean = Variant(
        id="v1",
        source="llm",
        text=(
            "Students are surrounded by information, but they still learn from teachers. "
            "Beyond the classroom, they also draw knowledge from YouTube, TikTok, online courses, AI tools, search engines, social media, and peer communities. "
            "These sources have created a new kind of learning environment. "
            "Knowledge is no longer scarce, and access is no longer the biggest problem. "
            "The real challenge now is knowing what is accurate, useful, ethical, and worth trusting."
        ),
    )
    paragraph = scan_text(source).paragraphs[0]
    diagnostics = selection_diagnostics([Variant(id="source_preserved", text=source, source="source_preserved"), corrupted, clean], paragraph)
    corrupted_row = next(row for row in diagnostics if row["variant_id"] == "retry_v1")

    assert "keyword_dump_sequence" in corrupted_row["blockers"]
    assert "lost_serial_punctuation" in corrupted_row["blockers"]
    assert "capitalized_common_noun_mid_sentence" in corrupted_row["blockers"]
    assert choose_variant([Variant(id="source_preserved", text=source, source="source_preserved"), corrupted, clean], paragraph).id == "v1"


def test_v6_selector_prefers_cleaner_route_over_forced_connector_variant():
    source = (
        "Now, students are surrounded by information. "
        "They learn from teachers, but also from YouTube, TikTok, online courses, AI tools, search engines, social media, and peer communities. "
        "This has created a new kind of learning environment. "
        "Knowledge is no longer scarce. Access is no longer the biggest problem. "
        "The real challenge is knowing what is accurate, useful, ethical, and worth trusting."
    )
    weaker = Variant(
        id="v1",
        source="llm",
        text=(
            "Students are surrounded by information and still rely on teachers. "
            "Moreover they learn from platforms such as YouTube, TikTok, online courses, AI tools, search engines, social media and peer communities. "
            "Such a mix has created a new kind of learning environment. "
            "Because knowledge is no longer scarce and access is no longer the biggest problem, the focus shifts to judging information quality. "
            "The real challenge is knowing which sources are accurate, useful, ethical and worth trusting."
        ),
    )
    engineered = Variant(
        id="v2",
        source="llm",
        text=(
            "Students are now surrounded by information, which they still receive from teachers. "
            "Beyond teachers they also draw knowledge from digital platforms such as YouTube and TikTok as well as online courses, AI tools, search engines, social media and peer communities. "
            "This blend of traditional and digital sources has created a new kind of learning environment. "
            "Because knowledge is no longer scarce and access is no longer the biggest problem, the difficulty shifts toward assessing information quality. "
            "Thus the real challenge is knowing which content is accurate, useful, ethical and worth trusting."
        ),
    )
    cleaner = Variant(
        id="v3",
        source="llm",
        text=(
            "Now students are surrounded by information. "
            "They continue to learn from teachers while also turning to digital venues such as YouTube, TikTok, online courses, AI tools, search engines, social media and peer communities. "
            "These varied sources have produced a new kind of learning environment. "
            "Since knowledge is no longer scarce and access is no longer the biggest problem, the central issue becomes evaluating what can be trusted. "
            "The real challenge is deciding which material is accurate, useful, ethical and worth trusting."
        ),
    )
    paragraph = scan_text(source).paragraphs[0]
    variants = [Variant(id="source_preserved", text=source, source="source_preserved"), weaker, engineered, cleaner]
    diagnostics = selection_diagnostics(variants, paragraph)
    weaker_row = next(row for row in diagnostics if row["variant_id"] == "v1")
    engineered_row = next(row for row in diagnostics if row["variant_id"] == "v2")

    assert "engineered_route_quality_review_required" in weaker_row["quality_warnings"]
    assert "engineered_route_quality_review_required" in engineered_row["quality_warnings"]
    assert choose_variant(variants, paragraph).id == "v3"


def test_v6_writer_quality_prefers_compact_role_list_over_catalogue_chain():
    source = (
        "Now, students are surrounded by information. "
        "They learn from teachers, but also from YouTube, TikTok, online courses, AI tools, search engines, social media, and peer communities. "
        "This has created a new kind of learning environment. "
        "Knowledge is no longer scarce. Access is no longer the biggest problem. "
        "The real challenge is knowing what is accurate, useful, ethical, and worth trusting."
    )
    catalogue = Variant(
        id="v1",
        source="llm",
        text=(
            "Students navigate an abundance of information. "
            "Teachers guide their learning while digital sources such as YouTube and TikTok expand opportunities. "
            "Online courses and AI tools add further depth. "
            "Search engines help locate material. "
            "Social media and peer communities enrich the experience. "
            "These developments have produced a new kind of learning environment. "
            "Knowledge is no longer scarce. "
            "Access is no longer the biggest problem. "
            "The real challenge lies in discerning what is accurate and useful."
        ),
    )
    compact = Variant(
        id="v2",
        source="llm",
        text=(
            "Students now learn in an environment where information comes from many places, not only from teachers. "
            "YouTube, TikTok, online courses, AI tools, search engines, social media, and peer communities all add to the learning mix and have created a new kind of learning environment. "
            "Knowledge is no longer scarce, and access is no longer the biggest problem. "
            "The harder task is deciding which information is accurate, useful, ethical, and worth trusting."
        ),
    )
    paragraph = scan_text(source).paragraphs[0]
    variants = [Variant(id="source_preserved", text=source, source="source_preserved"), catalogue, compact]

    assert catalogue_sentence_chain(catalogue.text)
    diagnostics = selection_diagnostics(variants, paragraph)
    catalogue_row = next(row for row in diagnostics if row["variant_id"] == "v1")
    assert "catalogue_sentence_chain_review_required" in catalogue_row["quality_warnings"]
    assert choose_variant(variants, paragraph).id == "v2"


def test_v6_selector_blocks_scope_marker_padding_even_when_risk_drops():
    source = (
        "Today’s education system is changing faster than many schools can comfortably manage. "
        "In the past, education was mostly built around the classroom, the textbook, and the teacher. "
        "Students received knowledge from trusted sources, practiced it through homework, and proved their learning through tests. "
        "That model still exists, but it no longer fully reflects how young people learn today."
    )
    padded = Variant(
        id="v1",
        source="llm",
        text=(
            "Today the education system is changing faster than many schools can comfortably manage longer periods of change. "
            "In the past education was mostly built around the classroom. "
            "It also depended on the textbook and on the teacher. "
            "Students received knowledge from trusted sources, practiced it with homework and proved their learning through tests. "
            "That model still exists but it no longer fully reflects how young people learn today."
        ),
    )
    clean = Variant(
        id="v2",
        source="llm",
        text=(
            "Today the education system is changing faster than many schools can comfortably manage. "
            "In the past, education was mostly built around the classroom, the textbook and the teacher. "
            "Students received knowledge from trusted sources, practiced it through homework and proved their learning through tests. "
            "That model still exists, but it no longer fully reflects how young people learn today."
        ),
    )
    paragraph = scan_text(source).paragraphs[0]
    variants = [Variant(id="source_preserved", text=source, source="source_preserved"), padded, clean]
    diagnostics = selection_diagnostics(variants, paragraph)
    padded_row = next(row for row in diagnostics if row["variant_id"] == "v1")

    assert "source_scope_marker_reused_as_content" in padded_row["blockers"]
    assert "source_scope_marker_reused_as_content_review_required" in padded_row["quality_warnings"]
    assert choose_variant(variants, paragraph).id == "v2"


def test_v6_writer_quality_blocks_unnatural_completion_phrase():
    text = "The real challenge is knowing what is accurate. Determining what is ethical and worth trusting completes the challenge."

    assert "unnatural_completion_phrase" in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_malformed_additive_predicate():
    text = 'Education today should focus not only on "what students know". It also on how students think.'

    assert "malformed_additive_predicate" in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_proxy_context_adjective_stack():
    text = "The shift toward digital-media-rich classrooms has made teachers more important."

    assert "proxy_context_adjective_stack" in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_generic_role_inflation():
    text = "Teachers now act as navigators who help students assess credibility."

    assert "generic_role_inflation" in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_quoted_concept_literalised_as_words():
    text = (
        'Education today should not only focus on the words "what students know," '
        'It should also focus on the words "how students think."'
    )

    assert "semantic_anchor_corruption" in candidate_integrity_blockers(text)


def test_v6_writer_quality_repairs_quoted_concept_literalisation():
    text = (
        'Education today should not only focus on the words "what students know," '
        'but also on the words "how students think." '
        "It should not focus on the words that represent what students know."
    )

    repaired = repair_generated_prose(text)

    assert "the words" not in repaired
    assert "words that represent" not in repaired
    assert '"what students know,"' in repaired
    assert '"how students think."' in repaired
    assert "focus on what students know" in repaired


def test_v6_writer_quality_blocks_generic_adaptation_role_padding():
    text = "The local process improves as departments respond to new challenges."

    assert "generic_role_inflation" in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_assessment_consequence_before_ai_risk_behaviour():
    text = (
        "AI tools help students brainstorm ideas. "
        "This makes assessment more difficult and raises questions about fairness. "
        "Students may become dependent on AI-generated answers without understanding the work."
    )

    assert "premature_assessment_consequence" in candidate_integrity_blockers(text)


def test_v6_writer_quality_blocks_duplicated_assessment_consequence():
    text = (
        "This makes assessment more difficult. "
        "Students may submit polished work that does not reflect their ability. "
        "Assessment becomes more difficult and raises questions about fairness."
    )

    assert "duplicated_assessment_consequence" in candidate_integrity_blockers(text)


def test_v6_selector_allows_large_risk_drop_without_finding_drop_for_review():
    source = (
        "This shift has made the role of teachers even more important, not less important. "
        "A teacher is no longer just someone who delivers information. "
        "A good teacher helps students make sense of information. "
        "They guide students to question sources, compare viewpoints, develop judgment, and apply knowledge in real situations. "
        "In other words, education today should not only focus on what students know, but also on how students think."
    )
    candidate = Variant(
        id="v1",
        source="llm",
        text=(
            "The recent shift has made the role of teachers even more important, not less important. "
            "A teacher is no longer someone who delivers information. "
            "A good teacher helps students make sense of that information. "
            "Teachers guide students to question sources and compare viewpoints. "
            "They help students develop judgment and apply knowledge in real situations. "
            "Education today should not only focus on what students know, but also on how students think."
        ),
    )
    paragraph = scan_text(source).paragraphs[0]
    variants = [Variant(id="source_preserved", text=source, source="source_preserved"), candidate]
    row = selection_diagnostics(variants, paragraph)[0]

    assert row["finding_drop"] >= 0
    assert row["risk_drop"] >= 10.0
    assert row["accepted_by_selector"] is True


def test_v6_selector_treats_dangling_modifier_as_review_warning_not_blocker():
    source = "A teacher is no longer just someone who delivers information. A good teacher helps students make sense of information."
    candidate = Variant(
        id="v1",
        source="llm",
        text="No longer merely someone who delivers information, a good teacher helps students make sense of what they learn.",
    )
    paragraph = scan_text(source).paragraphs[0]
    row = selection_diagnostics([Variant(id="source_preserved", text=source, source="source_preserved"), candidate], paragraph)[0]

    assert "dangling_modifier_sentence_start_review_required" in row["quality_warnings"]
    assert "dangling_modifier_sentence_start" not in row["blockers"]


def test_v6_writer_quality_repairs_source_term_fragments_into_role_lists():
    text = (
        "They learn from teachers and also from YouTube. "
        "TikTok. Online courses. AI tools. Search engines. Social media and peer communities. "
        "The real challenge is knowing what is accurate. Useful. Ethical. And worth trusting."
    )

    repaired = repair_generated_prose(text)

    assert "TikTok. Online courses." not in repaired
    assert "accurate. Useful." not in repaired
    assert "YouTube, TikTok, online courses, AI tools, search engines and social media" in repaired
    assert "accurate, useful, ethical and worth trusting" in repaired
