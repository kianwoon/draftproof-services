"""Phase B: Critical Thinking per-paragraph tag as a rewrite input.

Verifies the deterministic per-paragraph action flows scan-report -> diagnosis ->
writer prompt. Imports are ML-stack-free, so these run locally.
"""
from rewrite_v6.report_contracts import (
    extract_paragraph_diagnoses,
    paragraph_diagnoses_context,
    paragraph_diagnosis,
)
from rewrite_v6.direct_rewrite import _prompt


def _report():
    return {
        "paragraph_explanations": {
            "paragraphs": [{"paragraph_id": "p001", "main_issue": "broad claim"}]
        },
        "highlight_segments": [],
        "ai_risk_badge": {
            "critical_thinking_control": {
                "paragraphs": [
                    {"paragraph_id": "p001", "dimension": "evidence_grounding",
                     "action": "connect each claim to a source, example, or data point"},
                    # p002 has NO explanation and NO predictable phrases -> the CT tag
                    # must still create a diagnosis entry.
                    {"paragraph_id": "p002", "dimension": "specific_context",
                     "action": "anchor claims to a real assignment, case, example, or observation"},
                ]
            }
        },
    }


def test_extract_carries_critical_thinking_action():
    diagnoses = extract_paragraph_diagnoses(_report())
    assert diagnoses["p001"]["critical_thinking_action"].startswith("connect each claim")
    assert diagnoses["p001"]["critical_thinking_dimension"] == "evidence_grounding"
    # entry created for a paragraph that had only a CT tag
    assert "p002" in diagnoses
    assert diagnoses["p002"]["critical_thinking_action"].startswith("anchor claims")
    assert diagnoses["p002"]["critical_thinking_dimension"] == "specific_context"


def test_extract_no_badge_is_safe():
    diagnoses = extract_paragraph_diagnoses({"paragraph_explanations": {"paragraphs": []}})
    assert diagnoses == {}
    assert extract_paragraph_diagnoses(None) == {}


def test_prompt_includes_critical_thinking_insight():
    diagnosis = {"critical_thinking_action": "connect each claim to a source",
                 "predictable_phrases": []}
    out = _prompt("AI can improve learning by providing personalised support.", diagnosis, [])
    assert "CRITICAL-THINKING INSIGHT" in out
    assert "connect each claim to a source" in out
    assert "critical_thinking_focus" in out
    # insight, not a task to fabricate the missing evidence
    assert "do NOT fabricate" in out


def test_prompt_omits_focus_when_absent():
    out = _prompt("Some paragraph text here.", {"predictable_phrases": []}, [])
    assert "CRITICAL-THINKING INSIGHT" not in out


def test_prompt_focus_present_in_diversified_lane():
    diagnosis = {"critical_thinking_action": "take a position and justify why you chose it",
                 "predictable_phrases": []}
    out = _prompt("Students should use AI responsibly.", diagnosis, [], lane="diversified")
    assert "CRITICAL-THINKING INSIGHT" in out
    assert "take a position and justify" in out


def test_context_roundtrip_carries_action():
    diagnoses = extract_paragraph_diagnoses(_report())
    with paragraph_diagnoses_context(diagnoses):
        assert paragraph_diagnosis("p002")["critical_thinking_action"].startswith("anchor claims")
    # context resets cleanly
    assert paragraph_diagnosis("p002") is None


# ── Reflective questions as per-paragraph writer guidance ──────────────────────

def _report_with_questions():
    rep = _report()
    rep["scan_intelligence"] = {"document": {"paragraphs": [
        {"paragraph_id": "p001", "text": "AI can improve learning by providing personalised support to students."},
        {"paragraph_id": "p002", "text": "There are many benefits and challenges to using AI in school."},
    ]}}
    rep["ai_risk_badge"]["critical_thinking_control"]["questions"] = [
        {"dimension": "evidence_grounding", "anchor_quote": "personalised support",
         "question": "When did personalised support actually help a specific student?"},
        {"dimension": "specific_context", "anchor_quote": "benefits and challenges",
         "question": "Name one concrete benefit and one concrete risk."},
        {"dimension": "specific_context", "anchor_quote": "a fabricated phrase never in the draft",
         "question": "ghost question"},  # unanchored -> must be dropped
    ]
    return rep


def test_extract_matches_questions_to_anchored_paragraph():
    d = extract_paragraph_diagnoses(_report_with_questions())
    assert d["p001"]["critical_thinking_questions"] == ["When did personalised support actually help a specific student?"]
    assert d["p002"]["critical_thinking_questions"] == ["Name one concrete benefit and one concrete risk."]
    # the unanchored question (quote in no paragraph) is dropped everywhere
    assert all("ghost question" not in (d[p].get("critical_thinking_questions") or []) for p in d)


def test_prompt_includes_critical_thinking_questions_as_insight():
    diagnosis = {"critical_thinking_questions": ["Name one concrete benefit and one concrete risk."],
                 "predictable_phrases": []}
    out = _prompt("There are many benefits and challenges.", diagnosis, [])
    assert "INSIGHT into what is thin" in out
    assert "Name one concrete benefit" in out
    assert "critical_thinking_questions" in out
    # the override: questions are NOT answered by inventing evidence
    assert "Do NOT invent" in out
    assert "does NOT apply" in out  # general "add an illustrative anchor" is overridden


def test_prompt_omits_questions_when_absent():
    out = _prompt("Some paragraph text here.", {"predictable_phrases": []}, [])
    assert "INSIGHT into what is thin" not in out
    assert "critical_thinking_questions" not in out


def test_system_prompt_grounds_by_basis_not_first_person_default():
    from rewrite_v6.direct_rewrite import _SYSTEM
    # the unconditional first-person mandate is GONE
    assert "Ground EVERY generic claim in the author's FIRST-PERSON" not in _SYSTEM
    assert "NEVER trade it away" not in _SYSTEM
    # basis-driven mode choice present
    assert "CHOOSE THE GROUNDING MODE THAT FITS" in _SYSTEM
    assert "HEADLINES" in _SYSTEM
    assert "attribute and qualify the source" in _SYSTEM
    # first-person kept as ONE mode, not the default; fabrication explicitly forbidden for second-hand
    assert "First-person is ONE strong mode" in _SYSTEM
    assert "invents experience they never had" in _SYSTEM


def test_prompt_questions_present_in_diversified_lane():
    diagnosis = {"critical_thinking_questions": ["What is the strongest opposing view, and why reject it?"],
                 "predictable_phrases": []}
    out = _prompt("Students should use AI responsibly.", diagnosis, [], lane="diversified")
    assert "INSIGHT into what is thin" in out
    assert "strongest opposing view" in out
