from __future__ import annotations

import json

from poc.rewrite_v6.llm_config import (
    cerebras_model_name,
    grammar_gateway,
    grammar_model,
    planner_gateway,
    planner_extra_body,
    planner_llm_profile,
    planner_model,
    resolve_v6_api_key,
    resolve_v6_base_url,
    resolve_v6_model,
    selector_gateway,
    writer_extra_body,
    writer_llm_profile,
    writer_model,
)
from poc.rewrite_v6.pipeline import _grammer_extra_body, _grammer_model
from poc.rewrite_v6.planner_llm import _merge_decision, _planner_contract_gaps
from poc.rewrite_v6.plan import build_plan
from poc.rewrite_v6.scan import scan_text, findings_for_paragraph


def test_v6_gpt_oss_role_profiles_use_reasoning_and_bounded_sampling():
    assert planner_extra_body("openai/gpt-oss-120b") == {
        "reasoning": {"effort": "medium", "exclude": True},
        "include_reasoning": False,
    }
    assert writer_extra_body("openai/gpt-oss-120b") == {
        "reasoning": {"effort": "medium", "exclude": True},
        "include_reasoning": False,
    }
    assert _grammer_extra_body("openai/gpt-oss-120b") == {
        "reasoning": {"effort": "low", "exclude": True},
        "include_reasoning": False,
    }
    assert planner_llm_profile("openai/gpt-oss-120b") == {
        "max_tokens": None,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "repetition_penalty": 1.0,
    }


def test_v6_roles_default_to_gpt_oss_when_env_is_absent(monkeypatch):
    for name in (
        "DRAFTPROOF_V6_PLANNER_MODEL",
        "DRAFTPROOF_PLANNER_MODEL",
        "DRAFTPROOF_REWRITE_V5_PLANNER_MODEL",
        "DRAFTPROOF_V6_WRITER_MODEL",
        "DRAFTPROOF_V6_GRAMMAR_MODEL",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    assert planner_model() == "openai/gpt-oss-120b"
    assert writer_model() == "openai/gpt-oss-120b"
    assert grammar_model() is None
    assert _grammer_model() == "openai/gpt-oss-120b"


def test_v6_cerebras_direct_resolves_gateway_without_openrouter_payload(monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
    monkeypatch.delenv("DRAFTPROOF_V6_CEREBRAS_DIRECT", raising=False)

    assert resolve_v6_api_key("openrouter-key") == "test-key"
    assert resolve_v6_base_url("https://openrouter.ai/api/v1") == "https://api.cerebras.ai/v1"
    assert resolve_v6_model("openai/gpt-oss-120b") == "gpt-oss-120b"
    assert cerebras_model_name("openai/gpt-oss-120b") == "gpt-oss-120b"
    assert planner_extra_body("gpt-oss-120b") is None
    assert writer_extra_body("gpt-oss-120b") is None


def test_v6_cerebras_direct_maps_planner_and_selector_gateways(monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
    monkeypatch.delenv("DRAFTPROOF_V6_CEREBRAS_DIRECT", raising=False)

    planner = planner_gateway(api_key=None, base_url=None)
    selector = selector_gateway(api_key=None, base_url=None)

    assert planner.model == "gpt-oss-120b"
    assert planner.base_url == "https://api.cerebras.ai/v1"
    assert selector.model == "gpt-oss-120b"
    assert selector.base_url == "https://api.cerebras.ai/v1"


def test_v6_grammar_gateway_uses_role_specific_provider_config(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_CEREBRAS_DIRECT", "1")
    monkeypatch.setenv("DRAFTPROOF_V6_GRAMMAR_MODEL", "provider/grammar-model")
    monkeypatch.setenv("DRAFTPROOF_V6_GRAMMAR_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("DRAFTPROOF_V6_GRAMMAR_API_KEY", "grammar-key")

    gateway = grammar_gateway(api_key="writer-key", base_url="https://api.cerebras.ai/v1")

    assert gateway is not None
    assert gateway.model == "provider/grammar-model"
    assert gateway.base_url == "https://openrouter.ai/api/v1"
    assert gateway.api_key == "grammar-key"


def test_v6_gpt_oss_writer_uses_source_sensitive_profile_for_citations():
    ordinary = writer_llm_profile("openai/gpt-oss-120b", "A learner practiced the task in class.")
    sensitive = writer_llm_profile("openai/gpt-oss-120b", "The unit uses reasonable adjustment (CAST, 2024).")

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
    assert any("unsubmitted bridge term" in gap for gap in gaps)


def test_v6_planner_contract_gaps_trigger_route_skeleton_fallback():
    scan = scan_text(
        "A student I will call Johnny had disclosed learning support needs related to ADHD, ASD, anxiety and learning difficulties."
    )
    paragraph, plan = build_plan(scan)
    merged = _merge_decision(paragraph, plan, {
        "contract_gaps": ["p001_s001 safe_rebuild_shape keeps a comma-list route instead of a safer construction route"],
        "finding_contracts": [{
            "source_sentence_id": "p001_s001",
            "safe_rebuild_shape": "bad, comma, list",
        }],
    })

    decision = merged.ai_safe_route["llm_planner_decision"]
    assert decision["status"] == "degraded_contract_gaps"
    assert "finding_contracts" not in decision
    assert "paragraph_blueprint" not in decision
    assert "fallback_instruction" not in decision
    assert "safe_sentence_shape" not in json.dumps(decision)


def test_v6_planner_route_gaps_validate_new_small_contract_source_basis():
    scan = scan_text("AI can help students learn, but copied answers can hide weak understanding.")
    _paragraph, plan = build_plan(scan)
    decision = {
        "paragraph_problem": "The paragraph needs visible behaviour before consequence.",
        "paragraph_route": "support use -> copying risk -> learning consequence",
        "flow_plan": [
            {"step_id": "fp001", "function": "show support", "source_basis": ["p001_s001"], "must_include": ["AI"]},
            {"step_id": "fp002", "function": "show risk", "source_basis": ["p999_s001"], "must_include": ["copied answers"]},
        ],
        "validated_construction_route": {
            "movement": "support -> risk",
            "sentence_jobs": [
                {"job_id": "j001", "job": "show support", "source_basis": ["p001_s001"], "must_use_meaning": ["AI"]},
                {"job_id": "j002", "job": "show risk", "source_basis": ["p999_s001"], "must_use_meaning": ["copied answers"]},
            ],
            "do_not_copy": [],
            "validation_rules": [],
        },
    }

    gaps = _planner_contract_gaps(decision, findings_for_paragraph(scan, "p001"), plan)

    assert any("flow_plan 2 uses unknown source sentence id p999_s001" in gap for gap in gaps)
    assert any("sentence_job 2 uses unknown source sentence id p999_s001" in gap for gap in gaps)
