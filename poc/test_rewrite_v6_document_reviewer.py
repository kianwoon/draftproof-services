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


def test_review_drops_polarity_inverting_correction(monkeypatch):
    # The fidelity guard must drop a correction that flips a 'not only X but also Y' balance into
    # 'Y over X', even when the score would otherwise improve.
    monkeypatch.setattr(document_reviewer, "_score", lambda t: 10.0)
    doc = "AI is not only useful for drafting but also useful for checking the student's reasoning."
    correction = {"corrections": [
        {"original": doc, "revised": "AI matters for checking reasoning rather than for drafting."}
    ]}
    gw = _stub([json.dumps(correction)])
    result = document_reviewer.review_document(doc, gateway=gw)
    assert result.text == doc            # polarity-inverting correction dropped
    assert result.corrections == []


from poc.rewrite_v6 import direct_rewrite as dr
from poc.rewrite_v6.pipeline import DocumentResult
from poc.rewrite_v6.scan import scan_text as _scan_text


def test_apply_reviewer_runs_qc_then_scans_qcd_text(monkeypatch):
    writer_text = "\n\n".join([
        "In my classroom, I have seen change arrive faster than the school can absorb it.",
        "In my classroom, I have seen students lean on AI before they ask me anything.",
        "In my classroom, I have seen the textbook lose its old authority in a single year.",
    ])
    doc = DocumentResult(
        initial_scan=_scan_text(writer_text),
        final_scan=_scan_text(writer_text),
        rewritten_text=writer_text,
        passes=[],
        pass_trace=[],
    )
    correction = {"corrections": [
        {"original": "In my classroom, I have seen change arrive faster than the school can absorb it.",
         "revised": "Change now arrives faster than my school can absorb it."}
    ]}
    gw = _stub([json.dumps(correction)])
    monkeypatch.setattr(document_reviewer, "_score", lambda t: 10.0)

    out = dr._apply_reviewer(doc, gw, cancellation_check=None)
    assert "Change now arrives faster than my school can absorb it." in out.rewritten_text
    # final_scan must reflect the QC'd text (order: rewrite -> QC -> scan). Scan field is source_text.
    assert out.final_scan.source_text == out.rewritten_text
    assert any(row.get("selected_source") == "qc_reviewer" for row in out.pass_trace)


def test_apply_reviewer_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_REVIEWER", "0")
    writer_text = "A short clean paragraph that needs nothing at all."
    doc = DocumentResult(
        initial_scan=_scan_text(writer_text),
        final_scan=_scan_text(writer_text),
        rewritten_text=writer_text,
        passes=[],
        pass_trace=[],
    )
    gw = _stub(["{}"])
    out = dr._apply_reviewer(doc, gw, cancellation_check=None)
    assert out is doc
    assert gw.calls == []


def test_apply_reviewer_keeps_safe_drops_regressing(monkeypatch):
    # Two corrections: one safe, one regressing -> safe kept, regressing dropped (per-correction guard).
    writer_text = "\n\n".join([
        "In my classroom, I have seen change arrive faster than the school can absorb it.",
        "In my classroom, I have seen students lean on AI before they ask me anything.",
        "In my classroom, I have seen the textbook lose its old authority in a single year.",
    ])
    doc = DocumentResult(
        initial_scan=_scan_text(writer_text),
        final_scan=_scan_text(writer_text),
        rewritten_text=writer_text,
        passes=[],
        pass_trace=[],
    )
    correction = {"corrections": [
        {"original": "In my classroom, I have seen students lean on AI before they ask me anything.",
         "revised": "Students now lean on AI before they ask me anything."},
        {"original": "In my classroom, I have seen the textbook lose its old authority in a single year.",
         "revised": "The textbook lost its authority, which made things WORSE."},
    ]}
    gw = _stub([json.dumps(correction)])
    monkeypatch.setattr(document_reviewer, "_score", lambda t: 90.0 if "WORSE" in t else 10.0)

    out = dr._apply_reviewer(doc, gw, cancellation_check=None)
    assert "Students now lean on AI before they ask me anything." in out.rewritten_text  # safe kept
    assert "WORSE" not in out.rewritten_text                                              # regressing dropped
    assert out.final_scan.source_text == out.rewritten_text
