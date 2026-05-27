from __future__ import annotations

from poc.rewrite_v6.paragraph_normalizer import normalize_paragraph_blocks
from poc.rewrite_v6.scan import scan_text


def _squash(text: str) -> str:
    return " ".join(str(text or "").split())


def test_v6_pre_scan_normalizer_splits_large_functional_paragraph_without_rewriting():
    source = (
        "Case Evidence\n"
        "The activity began with a course context sentence that named the setting and purpose. "
        "A learner disclosed support needs connected to attention, anxiety, and learning difficulty. "
        "At the beginning of the course, he required another person to attend class with him. "
        "At first, he barely spoke with other learners. "
        "During one group task, he took a manager role and led the team. "
        "The team gave him positive feedback after the activity. "
        "Later, the class joined a community service activity. "
        "During the service, he adjusted his communication with the client. "
        "Two days later, the client sent a thank-you card. "
        "After that, he became more confident in class. "
        "He no longer needed support staff beside him. "
        "In my view, the example shows how the classroom task connected to confidence and participation."
    )

    normalized = normalize_paragraph_blocks(source)

    assert len(normalized.split("\n\n")) >= 2
    assert _squash(normalized) == _squash(source)
    assert normalized.split("\n", 1)[0] == "Case Evidence"


def test_v6_scan_uses_normalized_paragraphs_before_planning():
    source = (
        "The first sentence gives context for the work and the class setting. "
        "The second sentence adds a learner detail and explains why support mattered. "
        "The third sentence explains the initial condition before the activity started. "
        "The fourth sentence gives a classroom behavior that shaped the response. "
        "During the activity, the learner tried a role with visible responsibility. "
        "The group responded to the role and gave practical feedback. "
        "Later, the class moved into a community task with a different audience. "
        "The learner adjusted communication during the task and handled the exchange. "
        "After that, the learner became more independent in the classroom. "
        "In my view, the case connected classroom participation with confidence. "
        "The final observation links the sequence back to the classroom support plan and learner participation."
    )

    scan = scan_text(source)

    assert scan.scores["paragraph_count"] > 1
    assert _squash(scan.source_text) == _squash(source)


def test_v6_pre_scan_normalizer_keeps_short_many_sentence_paragraph_together():
    source = (
        "The shift changed the teacher role. "
        "Teachers still explain information. "
        "They also guide source checks. "
        "Students compare viewpoints. "
        "They develop judgment. "
        "They apply ideas. "
        "Knowledge still matters. "
        "Thinking also matters. "
        "The paragraph remains one short section."
    )

    normalized = normalize_paragraph_blocks(source)

    assert normalized == source
    assert scan_text(source).scores["paragraph_count"] == 1
