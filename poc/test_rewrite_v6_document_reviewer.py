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


def test_robotic_transitions_fires():
    doc = (
        "Schools changed fast.\n\n"
        "Furthermore, the curriculum widened beyond the textbook every year.\n\n"
        "Moreover, students began learning from sources teachers could not control.\n\n"
        "In conclusion, the old model no longer matches how learners seek knowledge."
    )
    issues = detect_residual_patterns(doc)
    robotic = [i for i in issues if i.rule == "robotic_transitions"]
    assert len(robotic) == 1
    assert 8 in robotic[0].trick_ids
    assert any(s.lower().startswith("furthermore") for s in robotic[0].target_sentences)
    assert len(robotic[0].target_sentences) >= 2


def test_robotic_transitions_silent_when_absent():
    doc = "Schools changed fast.\n\nThat created a new problem teachers had to solve themselves."
    assert [i for i in detect_residual_patterns(doc) if i.rule == "robotic_transitions"] == []


def test_repeated_subject_starts_fires():
    doc = (
        "Technology helps students learn. Technology also distracts them constantly. "
        "Technology shapes how teachers plan every lesson now."
    )
    issues = detect_residual_patterns(doc)
    rep = [i for i in issues if i.rule == "repeated_subject_starts"]
    assert len(rep) == 1
    assert 19 in rep[0].trick_ids


def test_balance_phrase_fires():
    doc = "AI in the classroom brings both opportunities and risks for every learner involved."
    issues = detect_residual_patterns(doc)
    bal = [i for i in issues if i.rule == "balance_phrase"]
    assert len(bal) == 1
    assert 7 in bal[0].trick_ids


def test_balance_phrase_silent_when_specific():
    doc = "AI helps students brainstorm at the planning stage, but it hides gaps they cannot explain."
    assert [i for i in detect_residual_patterns(doc) if i.rule == "balance_phrase"] == []


def test_rhythm_sameness_fires_on_uniform_lengths():
    doc = (
        "Students learn many new things every single day. "
        "Teachers plan many small lessons every single week. "
        "Schools change many old rules every single year. "
        "Parents ask many hard questions every single term. "
        "Leaders make many big choices every single month."
    )
    issues = detect_residual_patterns(doc)
    assert [i for i in issues if i.rule == "rhythm_sameness"], "expected rhythm_sameness to fire"
    assert 13 in next(i for i in issues if i.rule == "rhythm_sameness").trick_ids


def test_rhythm_sameness_silent_on_varied_lengths():
    doc = (
        "Schools change. "
        "When a student opens a laptop in class, the lesson the teacher planned a week earlier "
        "suddenly has to compete with a dozen brighter, faster, louder sources of information. "
        "That shift matters."
    )
    assert [i for i in detect_residual_patterns(doc) if i.rule == "rhythm_sameness"] == []


def test_clean_human_doc_has_no_issues_at_all():
    doc = _VARIED_DOC
    assert detect_residual_patterns(doc) == []


from poc.rewrite_v6 import document_reviewer
from poc.rewrite_v6.document_reviewer import (
    WRITING_CRAFT_GUIDELINES,
    build_reviewer_prompt,
    reviewer_enabled,
)


def test_rubric_has_all_25_guidelines():
    assert len(WRITING_CRAFT_GUIDELINES) == 25


def test_prompt_includes_doc_rubric_and_must_fix_evidence():
    from poc.rewrite_v6.residual_patterns import ResidualIssue
    must_fix = [ResidualIssue(rule="opener_monoculture", trick_ids=[19],
                              evidence="4 of 8 paragraphs open 'in my'",
                              target_sentences=["In my classroom, I have seen X."])]
    prompt = build_reviewer_prompt("FULL DOC TEXT HERE", must_fix)
    assert "FULL DOC TEXT HERE" in prompt
    assert "4 of 8 paragraphs open 'in my'" in prompt
    assert "In my classroom, I have seen X." in prompt
    assert "corrections" in prompt
    low = prompt.lower()
    assert "vary" in low and "transition" in low


def test_reviewer_enabled_default_on(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_REVIEWER", raising=False)
    assert reviewer_enabled() is True
    monkeypatch.setenv("DRAFTPROOF_V6_REVIEWER", "0")
    assert reviewer_enabled() is False


def _stub(payloads):
    class _StubGateway:
        def __init__(self, payloads):
            self._payloads = list(payloads)
            self.calls = []
        def chat(self, prompt, *, system=None, **kwargs):
            self.calls.append({"prompt": prompt, "system": system})
            payload = self._payloads.pop(0) if self._payloads else "{}"
            return SimpleNamespace(content=payload, raw_content=payload)
    return _StubGateway(payloads)


def test_review_splices_correction_by_verbatim_match(monkeypatch):
    monkeypatch.setattr(document_reviewer, "_score", lambda t: 10.0)
    doc = "In my classroom, I have seen change.\n\nIn my classroom, I have seen more change."
    correction = {"corrections": [
        {"original": "In my classroom, I have seen more change.",
         "revised": "Last spring, the change reached my own lesson plans."}
    ]}
    gw = _stub([json.dumps(correction)])
    result = document_reviewer.review_document(doc, gateway=gw)
    assert "Last spring, the change reached my own lesson plans." in result.text
    assert "In my classroom, I have seen more change." not in result.text
    assert len(result.corrections) == 1


def test_review_skips_unmatched_original(monkeypatch):
    monkeypatch.setattr(document_reviewer, "_score", lambda t: 10.0)
    doc = "In my classroom, I have seen change every term without fail."
    correction = {"corrections": [
        {"original": "A sentence that is not in the document.", "revised": "whatever"}
    ]}
    gw = _stub([json.dumps(correction)])
    result = document_reviewer.review_document(doc, gateway=gw)
    assert result.text == doc
    assert result.corrections == []


def test_review_drops_correction_that_raises_score(monkeypatch):
    def fake_score(t):
        return 90.0 if "WORSE" in t else 10.0
    monkeypatch.setattr(document_reviewer, "_score", fake_score)
    doc = "In my classroom, I have seen change every single term here.\n\nIn my classroom, I have seen it twice."
    correction = {"corrections": [
        {"original": "In my classroom, I have seen it twice.", "revised": "This made things WORSE."}
    ]}
    gw = _stub([json.dumps(correction)])
    result = document_reviewer.review_document(doc, gateway=gw)
    assert "WORSE" not in result.text
    assert result.text == doc


def test_review_returns_unchanged_on_bad_json(monkeypatch):
    monkeypatch.setattr(document_reviewer, "_score", lambda t: 10.0)
    doc = "In my classroom, I have seen change happen quickly here."
    gw = _stub(["not json at all"])
    result = document_reviewer.review_document(doc, gateway=gw)
    assert result.text == doc
    assert result.corrections == []
