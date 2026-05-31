from __future__ import annotations

import json
from types import SimpleNamespace

from poc.rewrite_v6 import residual_patterns
from poc.rewrite_v6.residual_patterns import ResidualIssue, detect_residual_patterns


def test_detect_returns_list_of_issues():
    assert detect_residual_patterns("") == []
    assert detect_residual_patterns("A single short paragraph.") == []


# The real reverted-rewrite sample: 7/8 paragraphs open "In my ...".
_MONOCULTURE_DOC = "\n\n".join([
    "In my classroom, I have seen curriculum changes outpace the tweaks my school can make.",
    "In my classroom, I see students navigating a flood of digital resources every single day.",
    "In my classroom, I have seen the shift toward project-based learning change the teacher role.",
    "In my years teaching, I keep noticing that many schools still cling to outdated routines.",
    "In my teaching, I have noticed AI platforms let students sketch essay outlines in minutes.",
    "One inequality I see is the technology gap that splits the class into haves and have-nots.",
    "In my classroom, I have seen that the education system must evolve beyond rote memorization.",
    "In my experience as a teacher, I feel we are at a crossroads about what assessment means.",
])

_VARIED_DOC = "\n\n".join([
    "Curriculum changes outpace the tweaks a school can make.",
    "Students now navigate a flood of digital resources every single day.",
    "Project-based learning has quietly changed what a teacher does.",
    "Many schools still cling to outdated routines despite the evidence.",
    "AI platforms let students sketch essay outlines in minutes.",
])


def test_opener_monoculture_fires_on_repeated_in_my():
    issues = detect_residual_patterns(_MONOCULTURE_DOC)
    monoc = [i for i in issues if i.rule == "opener_monoculture"]
    assert len(monoc) == 1
    issue = monoc[0]
    assert 19 in issue.trick_ids
    assert any("In my classroom" in s for s in issue.target_sentences)
    assert len(issue.target_sentences) >= 4


def test_opener_monoculture_silent_on_varied_doc():
    issues = detect_residual_patterns(_VARIED_DOC)
    assert [i for i in issues if i.rule == "opener_monoculture"] == []
