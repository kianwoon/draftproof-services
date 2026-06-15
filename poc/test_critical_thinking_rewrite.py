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


def test_prompt_includes_critical_thinking_focus():
    diagnosis = {"critical_thinking_action": "connect each claim to a source",
                 "predictable_phrases": []}
    out = _prompt("AI can improve learning by providing personalised support.", diagnosis, [])
    assert "CRITICAL-THINKING FOCUS" in out
    assert "connect each claim to a source" in out
    assert "critical_thinking_focus" in out


def test_prompt_omits_focus_when_absent():
    out = _prompt("Some paragraph text here.", {"predictable_phrases": []}, [])
    assert "CRITICAL-THINKING FOCUS" not in out


def test_prompt_focus_present_in_diversified_lane():
    diagnosis = {"critical_thinking_action": "take a position and justify why you chose it",
                 "predictable_phrases": []}
    out = _prompt("Students should use AI responsibly.", diagnosis, [], lane="diversified")
    assert "CRITICAL-THINKING FOCUS" in out
    assert "take a position and justify" in out


def test_context_roundtrip_carries_action():
    diagnoses = extract_paragraph_diagnoses(_report())
    with paragraph_diagnoses_context(diagnoses):
        assert paragraph_diagnosis("p002")["critical_thinking_action"].startswith("anchor claims")
    # context resets cleanly
    assert paragraph_diagnosis("p002") is None
