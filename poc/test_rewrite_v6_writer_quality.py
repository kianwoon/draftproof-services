from __future__ import annotations

from poc.rewrite_v6.prose_quality import robotic_sentence_chain
from poc.rewrite_v6.scan import scan_text
from poc.rewrite_v6.selector_diagnostics import selection_diagnostics
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
    assert "candidate_contract_violation" not in diagnostics["blockers"]
    assert choose_variant(variants, paragraph).source == "source_preserved"
