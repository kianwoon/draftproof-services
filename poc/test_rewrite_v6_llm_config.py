from __future__ import annotations

from poc.rewrite_v6.pipeline import (
    _planner_extra_body,
    _planner_llm_profile,
    _writer_extra_body,
    _writer_llm_profile,
)
from poc.rewrite_v6.planner_llm import _merge_decision, _planner_contract_gaps
from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6.scan import scan_text, findings_for_paragraph


def test_v6_gpt_oss_role_profiles_use_reasoning_and_bounded_sampling():
    assert _planner_extra_body("openai/gpt-oss-120b") == {
        "reasoning": {"effort": "medium", "exclude": True},
        "include_reasoning": False,
    }
    assert _writer_extra_body("openai/gpt-oss-120b") == {
        "reasoning": {"effort": "medium", "exclude": True},
        "include_reasoning": False,
    }
    assert _planner_llm_profile("openai/gpt-oss-120b") == {
        "max_tokens": None,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "repetition_penalty": 1.0,
    }


def test_v6_gpt_oss_writer_uses_source_sensitive_profile_for_citations():
    ordinary = _writer_llm_profile("openai/gpt-oss-120b", "A learner practiced the task in class.")
    sensitive = _writer_llm_profile("openai/gpt-oss-120b", "The unit uses reasonable adjustment (CAST, 2024).")

    assert ordinary["temperature"] == 0.65
    assert ordinary["max_tokens"] is None
    assert ordinary["top_p"] == 0.95
    assert sensitive["temperature"] == 0.45
    assert sensitive["top_p"] == 0.9


def test_v6_planner_contract_gaps_reject_writer_forbidden_shapes():
    scan = scan_text(
        "At the beginning, he was quite reserved, but as we got to know each other through casual conversation, I learned about some of his past learning experiences."
    )
    decision = {
        "finding_contracts": [{
            "source_sentence_id": "p001_s001",
            "safe_rebuild_shape": "At the beginning, he was quite reserved; through casual conversation we learned about his past learning experiences.",
        }]
    }

    gaps = _planner_contract_gaps(decision, findings_for_paragraph(scan, "p001"))

    assert any("semicolon" in gap for gap in gaps)


def test_v6_planner_contract_gaps_reject_unsubmitted_or_forbidden_safe_shape():
    scan = scan_text(
        "A student I will call Johnny had disclosed learning support needs related to ADHD, ASD, anxiety and learning difficulties."
    )
    decision = {
        "finding_contracts": [{
            "source_sentence_id": "p001_s001",
            "safe_rebuild_shape": "A student I will call Johnny disclosed ADHD and ASD. As support continued, other learning difficulties mattered.",
            "coverage_terms": ["A student I will call Johnny", "ADHD", "ASD", "learning", "difficulties"],
        }]
    }

    gaps = _planner_contract_gaps(decision, findings_for_paragraph(scan, "p001"))

    assert any("forbidden sentence opener 'As'" in gap for gap in gaps)
    assert any("unsubmitted bridge term 'other'" in gap for gap in gaps)


def test_v6_planner_contract_gaps_trigger_deterministic_fallback():
    scan = scan_text(
        "A student I will call Johnny had disclosed learning support needs related to ADHD, ASD, anxiety and learning difficulties."
    )
    _, plan = build_plan(scan)
    merged = _merge_decision(plan, {
        "contract_gaps": ["p001_s001 safe_rebuild_shape keeps a comma-list route instead of a safer construction route"],
        "finding_contracts": [{
            "source_sentence_id": "p001_s001",
            "safe_rebuild_shape": "bad, comma, list",
        }],
    })

    decision = merged.ai_safe_route["llm_planner_decision"]
    assert decision["status"] == "degraded_contract_gaps"
    assert decision["fallback_instruction"]
    assert "," not in decision["finding_contracts"][0]["safe_rebuild_shape"]
    assert "source terms" not in decision["finding_contracts"][0]["safe_rebuild_shape"]
