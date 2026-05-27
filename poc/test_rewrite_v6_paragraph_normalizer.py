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

    assert len(normalized.split("\n\n")) >= 3
    assert _squash(normalized) == _squash(source)
    assert normalized.split("\n", 1)[0] == "Case Evidence"


def test_v6_scan_uses_normalized_paragraphs_before_planning():
    source = (
        "The first sentence gives context for the work. "
        "The second sentence adds a learner detail. "
        "The third sentence explains the initial condition. "
        "The fourth sentence gives a classroom behavior. "
        "During the activity, the learner tried a role. "
        "The group responded to the role. "
        "Later, the class moved into a community task. "
        "The learner adjusted communication during the task. "
        "After that, the learner became more independent. "
        "In my view, the case connected classroom participation with confidence."
    )

    scan = scan_text(source)

    assert scan.scores["paragraph_count"] > 1
    assert _squash(scan.source_text) == _squash(source)
