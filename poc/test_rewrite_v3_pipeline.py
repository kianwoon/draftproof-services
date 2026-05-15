"""Focused tests for external-calibrated rewrite pipeline V3."""

from __future__ import annotations

import inspect
import json
import os
import tempfile

from llm.gateway import LLMConfig, LLMGateway, LLMResponse
from rewrite_v2.contracts import build_rewrite_contract
from detect.authorship_windows import build_ai_footprint_profile, build_authorship_window_profile
from detect.rewrite_targets import build_problem_inventory, build_rewrite_target_profile
import rewrite_v3.pipeline as v3_pipeline
from rewrite_v3.anchor_validation import validate_v3_candidate
from rewrite_v3.assisted_footprint_executor import group_assisted_footprint_windows
from rewrite_v3.authorship_window_gate import evaluate_authorship_window_gate, select_authorship_window_targets
from rewrite_v3.candidate_loop import CandidateAction, CandidateIssue, decide_next_action, issues_from_trace
from rewrite_v3.compression_policy import compression_policy_for_family, compression_status
from rewrite_v3.document_units import (
    compact_document_inventory,
    document_units,
    structural_shape_contract,
    structural_shape_failures,
    word_count,
)
from rewrite_v3.external_proxy import evaluate_external_proxy
from rewrite_v3.layers.authorship_window_repair import (
    apply_authorship_window_replacements,
    build_authorship_window_repair_prompt,
    extract_authorship_window_replacements,
)
from rewrite_v3.layers.boundary_adapter import build_boundary_adapter_prompt
from rewrite_v3.layers.cited_practice_voice import build_cited_practice_voice_chunk_prompt, build_cited_practice_voice_prompt
from rewrite_v3.layers.clean_texture_boundary import build_clean_texture_boundary_chunk_prompt, build_clean_texture_boundary_prompt
from rewrite_v3.layers.contract_repair import build_contract_repair_prompt
from rewrite_v3.layers.contrast_boundary import build_contrast_boundary_prompt, extract_contrast_boundary_output
from rewrite_v3.layers.detector_ownership_fusion import (
    build_detector_ownership_fusion_prompt,
    extract_fused_document,
    extract_fused_document_with_diagnostics,
)
from rewrite_v3.layers.document_rhythm import build_document_rhythm_chunk_prompt, build_document_rhythm_prompt
from rewrite_v3.layers.plain_reasoning_broad_prose import build_plain_reasoning_broad_prose_prompt
from rewrite_v3.layers.structure_repair import build_structure_repair_prompt
from rewrite_v3.output_cleaning import clean_v3_candidate_output
from rewrite_v3.paragraph_portfolio_executor import (
    build_reconstruction_batches_by_prompt,
    paragraph_portfolio_config,
    parse_replacements_with_diagnostics,
    validate_ownership_replacement_effect,
    validate_replacement_structure,
    validate_topk_replacement_effect,
)
from rewrite_v3.pipeline import run_rewrite_pipeline_v3
from rewrite_v3.portfolio import select_portfolio_candidate
from rewrite_v3.prompt_templates.paragraph_portfolio import (
    build_paragraph_portfolio_ownership_prompt,
    build_paragraph_portfolio_planner_prompt,
    build_paragraph_portfolio_reconstruction_prompt,
    build_paragraph_portfolio_topk_prompt,
    fallback_paragraph_portfolio_plan,
    paragraph_portfolio_context,
    parse_paragraph_portfolio_plan,
    parse_paragraph_portfolio_replacements,
    validate_paragraph_portfolio_plan,
)
from rewrite_v3.prompt_contract import group_action_contract, ownership_contract_for_group, topk_repair_contract_for_group
from rewrite_v3.router import route_from_scan_contract
from rewrite_v3.scanner_controlled_executor import (
    ScannerControlledConfig,
    build_scanner_controlled_prompt,
    freeze_protected_anchors,
    parse_scanner_controlled_variants,
    protected_placeholder_integrity,
    rank_scanner_target_groups,
    scanner_controlled_candidate_quality,
    scanner_controlled_rank,
    scanner_controlled_variant_gate,
)
from rewrite_v3.scanner_contract import RewriteRiskClass, build_scan_contract
from rewrite_v3.strategy_plan import build_strategy_plan
from rewrite_v3.target_executor import (
    TargetGroup,
    apply_target_replacements,
    batch_target_groups,
    build_target_executor_prompt,
    group_rewrite_targets,
    parse_target_replacements,
)
from rewrite_v3.unit_preserving_prune_bridge import (
    apply_prune_bridge_replacements,
    build_prune_bridge_prompt,
    filter_prune_bridge_groups,
    parse_prune_bridge_replacements,
)


def assert_test(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


provider_env_names = [
    "DRAFTPROOF_OPENROUTER_PROVIDER_ROUTING_JSON",
    "OPENROUTER_PROVIDER_ROUTING_JSON",
    "LLM_PROVIDER_ROUTING_JSON",
    "DRAFTPROOF_OPENROUTER_PROVIDER_ORDER",
    "OPENROUTER_PROVIDER_ORDER",
    "DRAFTPROOF_OPENROUTER_PROVIDER_ONLY",
    "OPENROUTER_PROVIDER_ONLY",
    "DRAFTPROOF_OPENROUTER_PROVIDER_IGNORE",
    "OPENROUTER_PROVIDER_IGNORE",
    "DRAFTPROOF_OPENROUTER_ALLOW_FALLBACKS",
    "OPENROUTER_ALLOW_FALLBACKS",
    "DRAFTPROOF_OPENROUTER_REQUIRE_PARAMETERS",
    "OPENROUTER_REQUIRE_PARAMETERS",
    "DRAFTPROOF_OPENROUTER_ZDR",
    "OPENROUTER_ZDR",
    "DRAFTPROOF_OPENROUTER_ENFORCE_DISTILLABLE_TEXT",
    "OPENROUTER_ENFORCE_DISTILLABLE_TEXT",
    "DRAFTPROOF_OPENROUTER_PROVIDER_SORT",
    "OPENROUTER_PROVIDER_SORT",
    "DRAFTPROOF_OPENROUTER_DATA_COLLECTION",
    "OPENROUTER_DATA_COLLECTION",
    "DRAFTPROOF_LLM_EXTRA_BODY_JSON",
    "OPENROUTER_EXTRA_BODY_JSON",
    "LLM_EXTRA_BODY_JSON",
    "DRAFTPROOF_OPENROUTER_REASONING_EFFORT",
    "OPENROUTER_REASONING_EFFORT",
]
saved_provider_env = {name: os.environ.get(name) for name in provider_env_names}
try:
    for name in provider_env_names:
        os.environ.pop(name, None)
    os.environ["DRAFTPROOF_OPENROUTER_PROVIDER_ORDER"] = "siliconflow"
    os.environ["DRAFTPROOF_OPENROUTER_ALLOW_FALLBACKS"] = "0"
    gateway = LLMGateway(LLMConfig(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-v4-flash",
    ))
    assert_test(
        gateway.provider == {"order": ["siliconflow"], "allow_fallbacks": False},
        "LLM gateway reads OpenRouter provider routing from environment",
    )
    os.environ.pop("DRAFTPROOF_OPENROUTER_PROVIDER_ORDER", None)
    os.environ["DRAFTPROOF_OPENROUTER_ALLOW_FALLBACKS"] = "1"
    os.environ["DRAFTPROOF_OPENROUTER_PROVIDER_SORT"] = "latency"
    gateway = LLMGateway(LLMConfig(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-v4-flash",
    ))
    assert_test(
        gateway.provider == {"allow_fallbacks": True, "sort": "latency"},
        "LLM gateway supports OpenRouter latency-priority routing",
    )
    os.environ["DRAFTPROOF_LLM_EXTRA_BODY_JSON"] = '{"reasoning":{"effort":"none"}}'
    gateway = LLMGateway(LLMConfig(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-v4-flash",
    ))
    assert_test(
        gateway.extra_body == {"reasoning": {"effort": "none"}},
        "LLM gateway reads OpenRouter reasoning extra body from environment",
    )
finally:
    for name, value in saved_provider_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

json_with_curly_quotes = '{"rewritten_document":"Education should not only focus on \\"what students know,\\" but also on \\"how students think.\\" Today’s schools need that distinction."}'
json_response = LLMResponse(
    content=LLMGateway._normalize_quotes(json_with_curly_quotes),
    model="test",
    raw={"choices": [{"message": {"content": json_with_curly_quotes}, "finish_reason": "stop"}]},
)
assert_test(
    json.loads(json_response.raw_content)["rewritten_document"].startswith("Education should not only")
    and json_response.content != json_response.raw_content,
    "LLM response exposes raw JSON content before quote normalization for schema parsing",
)

model_env_names = ["DRAFTPROOF_REWRITE_MODEL_LOCK", "DRAFTPROOF_GENERATOR_MODEL", "LLM_MODEL"]
saved_model_env = {name: os.environ.get(name) for name in model_env_names}
try:
    os.environ["DRAFTPROOF_REWRITE_MODEL_LOCK"] = "deepseek/deepseek-v4-flash"
    os.environ["DRAFTPROOF_GENERATOR_MODEL"] = "deepseek/deepseek-v4-flash"
    os.environ["LLM_MODEL"] = "deepseek/deepseek-chat"
    gateway = v3_pipeline._gateway(
        api_key="test-key",
        model="deepseek/deepseek-chat",
        base_url="https://openrouter.ai/api/v1",
        max_tokens=1200,
    )
    assert_test(
        gateway.model == "deepseek/deepseek-v4-flash",
        "V3 rewrite model lock overrides generic LLM_MODEL and explicit worker model",
    )
finally:
    for name, value in saved_model_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def report_for(text: str, *, mode: str = "broad_explanatory_essay", ai: float = 70.0) -> dict:
    return {
        "input_text": text,
        "rewrite_v2_content_route": {
            "primary_mode": mode,
            "confidence": 0.91,
            "reasons": ["test typed route"],
        },
        "ai_risk_badge": {
            "ai_likelihood_score": ai,
            "writing_quality_score": 60.0,
            "ai_components": {
                "topk_pattern_raw": 90.0,
                "topk_calibrated_risk": 60.0,
                "qualifying_text_ai_density": 60.0,
            },
        },
        "integrity_layers": {
            "layers": {
                "ai_authorship": {"score": ai},
                "ai_transformation": {"score": ai},
            },
        },
        "findings": {"critical": [], "high": [], "medium": [], "low": []},
    }


def report_with_preservation_inventory(text: str, anchors: list[dict]) -> dict:
    report = report_for(text, mode="broad_explanatory_essay", ai=70.0)
    report.pop("rewrite_v2_content_route", None)
    report["ai_mitigation"] = {
        "rewrite_handoff": {
            "rewrite_constraints": {
                "preservation_inventory": {
                    "anchors": anchors,
                }
            }
        }
    }
    return report


broad_source = (
    "Education is changing quickly because students now meet information in many places. "
    "Schools still matter, but the old classroom model no longer explains how learning happens.\n\n"
    "Teachers now help students judge information, compare sources, and build practical judgment. "
    "That work is harder than simply delivering facts, because students need to know what deserves trust."
)
broad_candidate = (
    "Education is changing fast because students meet information everywhere. "
    "Schools still matter, but the old classroom model feels incomplete.\n\n"
    "Teachers help students judge sources and build practical judgment. "
    "That is harder than handing out facts."
)


units = document_units(broad_source)
assert_test(len(units) == 2 and not units[0].is_heading, "V3 splits blank-line document units structurally")
inventory = compact_document_inventory(broad_source, preview_chars=40)
assert_test(inventory[0]["text_preview"].endswith("..."), "V3 compact inventory bounds source previews")
section_source = "Overview\nThis is the first body paragraph.\n\nMethod\nThis is the second body paragraph."
section_candidate = "Overview\n\nThis is the first body paragraph.\n\nMethod\n\nThis is the second body paragraph."
assert_test(
    len(document_units(section_source)) == len(document_units(section_candidate)) == 2,
    "V3 treats separated headings and heading-body blocks as equivalent document units",
)
shape_source = "Heading One\nThis is the first body paragraph.\n\nHeading Two\nThis is the second body paragraph."
shape_candidate = "Heading One\nThis is the first body paragraph.\n\nHeading Two\nThis is the revised body paragraph."
shape_split_candidate = "Heading One\nThis is the first body paragraph.\n\nHeading Two\n\nThis is the revised body paragraph."
shape_lost_heading_candidate = "Heading One\nThis is the first body paragraph.\n\nThis is the revised body paragraph."
assert_test(
    structural_shape_contract(shape_source)["block_count"] == 2
    and not structural_shape_failures(shape_source, shape_candidate)
    and "block_count_changed" in structural_shape_failures(shape_source, shape_split_candidate)
    and "heading_like_line_missing" in structural_shape_failures(shape_source, shape_lost_heading_candidate),
    "V3 structural shape contract detects paragraph splits and heading loss",
)
shape_validation = validate_v3_candidate(
    original_text=shape_source,
    candidate_text=shape_lost_heading_candidate,
    contract=build_rewrite_contract(shape_source, content_mode="broad_explanatory_essay"),
)
assert_test(
    "heading_like_line_missing" in shape_validation.failures,
    "V3 candidate validation rejects heading loss even when unit count is unchanged",
)
assert_test(
    "rewrite_v2.content_router" not in inspect.getsource(v3_pipeline),
    "V3 pipeline does not depend on untracked V2 content router",
)

rhythm_policy = compression_policy_for_family("document_rhythm", word_count(broad_source))
rhythm_status = compression_status(broad_source, broad_candidate, rhythm_policy)
assert_test(rhythm_status["in_band"], "V3 document-rhythm compression accepts controlled shortening")

too_short = "Education is changing fast.\n\nTeachers help students judge information."
too_short_status = compression_status(broad_source, too_short, rhythm_policy)
assert_test(too_short_status["status"] == "below_floor", "V3 compression policy rejects over-compressed candidates")

texture_policy = compression_policy_for_family("clean_texture_boundary", word_count(broad_source))
texture_status = compression_status(broad_source, broad_source, texture_policy)
assert_test(texture_status["in_band"], "V3 clean-texture compression does not force shortening before scan")
clean_prompt = build_clean_texture_boundary_prompt(
    original_text=broad_source,
    scan_report=report_for(broad_source, ai=70.0),
    style_examples={"positive": [], "negative": []},
)
assert_test(
    "target_word_band" not in clean_prompt and "scanner_problem_profile" in clean_prompt,
    "V3 clean-texture prompt is scan-driven without a word band",
)

fusion_prompt = build_detector_ownership_fusion_prompt(
    source_text=broad_source,
    detector_candidate=broad_candidate,
    ownership_candidate=(
        "Education is changing fast because students meet information everywhere. "
        "From my view, schools still matter, but I would not treat the old classroom model as the whole explanation.\n\n"
        "Teachers help students judge sources and build practical judgment. "
        "That is harder than handing out facts."
    ),
    detector_trace={
        "candidate_ai": 42.0,
        "candidate_topk": 70.0,
        "authorship_window_profile": {
            "windows": [{
                "window_id": "w001",
                "paragraph_id": "p001",
                "label": "moderately_ai_assisted",
                "confidence": "high",
                "word_count": 32,
                "ai_assistance_score": 0.8,
                "source_excerpt": "Education is changing fast because students meet information everywhere.",
            }]
        },
        "external_proxy": {
            "reasons": ["segment_human_fraction_low"],
            "metrics": {"segment_authorship_gate": {"fraction_human": 0.1}},
        },
    },
    ownership_trace={
        "ownership_gate": {
            "active": True,
            "passed": True,
            "ownership_score": 10.0,
            "ownership_change_count": 3,
            "ownership_elements_supported": ["author_trace", "specific_context", "real_judgment"],
        },
        "target_execution_trace": {
            "target_replacements": [{
                "group_id": "tg001",
                "replacement_text": "From my view, schools still matter, but I would not treat the old classroom model as the whole explanation.",
                "candidate_quality": {
                    "ownership_score": 10.0,
                    "ownership_change_count": 3,
                    "ownership_elements_supported": ["author_trace", "specific_context", "real_judgment"],
                },
            }]
        },
    },
    family="clean_texture_boundary",
)
assert_test(
    "Use detector_strong_candidate as the base structure" in fusion_prompt
    and "Fuse ownership only inside failed_windows" in fusion_prompt
    and "Do not humanize the whole document" in fusion_prompt,
    "V3 fusion prompt targets detector-hot windows instead of whole-document humanizing",
)
fusion_payload = json.loads(fusion_prompt.split("PAYLOAD:\n", 1)[1])
assert_test(
    fusion_payload["source_structure_contract"]["block_count"] == len(document_units(broad_source))
    and len(fusion_payload["source_document_inventory"]) == len(document_units(broad_source))
    and "do not merge, split, drop, or reorder units" in " ".join(fusion_payload["requirements"]),
    "V3 fusion prompt carries hard source structure inventory",
)
assert_test(
    extract_fused_document(json.dumps({"rewritten_document": broad_candidate})) == broad_candidate,
    "V3 fusion parser extracts rewritten_document JSON",
)
assert_test(
    extract_fused_document(json.dumps({"fused_document": broad_candidate})) == broad_candidate
    and extract_fused_document(broad_candidate) == broad_candidate,
    "V3 fusion parser accepts full-document aliases and plain-text fallback",
)
empty_fusion, empty_fusion_diagnostics = extract_fused_document_with_diagnostics("{}")
assert_test(
    empty_fusion == ""
    and empty_fusion_diagnostics["parse_status"] == "ok"
    and empty_fusion_diagnostics["failure"] == "missing_nonempty_document_field"
    and empty_fusion_diagnostics["top_level_keys"] == [],
    "V3 fusion parser reports missing document field instead of opaque empty output",
)

cited_source = (
    "Introduction\n\n"
    "In my teaching practice, students struggle with Combination Haircut Structures (SHBHCUT006). "
    "Billett (2013) argues that practice-based knowledge needs guidance and scaffolding.\n\n"
    "Conclusion\n\n"
    "Johnny later became more confident after role-playing activities and classroom support."
)
cited_candidate = (
    "Introduction\n\n"
    "In my teaching practice, students still struggle with Combination Haircut Structures (SHBHCUT006). "
    "Billett (2013) argues that practice-based knowledge needs guidance and scaffolding.\n\n"
    "Conclusion\n\n"
    "Johnny became more confident after role-playing activities and classroom support."
)
cited_bad_candidate = cited_candidate.replace("Billett (2013)", "the research")
cited_contract = build_rewrite_contract(cited_source, content_mode="academic_cited_text")
assert_test(
    validate_v3_candidate(original_text=cited_source, candidate_text=cited_candidate, contract=cited_contract).passed,
    "V3 anchor validator accepts cited candidate with headings and citation preserved",
)
assert_test(
    not validate_v3_candidate(original_text=cited_source, candidate_text=cited_bad_candidate, contract=cited_contract).passed,
    "V3 anchor validator rejects lost citation anchor",
)
quote_source = 'Brennan calls vocational education a “practice architecture”.\n\nThe teacher then applies the idea.'
quote_contract = build_rewrite_contract(quote_source, content_mode="academic_cited_text")
quote_candidate = 'Brennan calls vocational education a "practice architecture."\n\nThe teacher then applies the idea.'
quote_repaired = v3_pipeline._restore_exact_quote_anchors(quote_candidate, quote_contract)
assert_test(
    "“practice architecture”" in quote_repaired,
    "V3 restores exact quote anchors when only quote punctuation drifted",
)
assert_test(
    len(v3_pipeline._unit_chunks(cited_source, force_unit_chunks=True)) == len(document_units(cited_source)),
    "V3 can force protected documents into unit-level chunks",
)
assert_test(
    len(document_units(v3_pipeline._normalize_chunk_unit_boundaries("One part.\n\nSecond part.", expected_units=1))) == 1,
    "V3 collapses extra blank-line boundaries inside single-unit chunks",
)

long_text = "\n\n".join(
    f"Section {index}\n\nThis section has a short source unit that should be represented compactly in prompts."
    for index in range(1, 70)
)
chunk_prompt = build_document_rhythm_chunk_prompt(
    source_units=compact_document_inventory(long_text, max_units=3, preview_chars=80),
    global_plan={"family": "document_rhythm", "source_words": word_count(long_text)},
    compression_policy=compression_policy_for_family("document_rhythm", 120),
    style_examples={"positive": [], "negative": []},
)
assert_test(
    len(chunk_prompt) < 7000 and long_text not in chunk_prompt,
    "V3 chunk prompt stays bounded and does not include full long document",
)

wrapped_candidate = (
    "Here is the repaired text with adjusted paragraph boundaries to match the 8-unit structure of the source:\n\n"
    "School is changing quickly.\n\nTeachers help students think."
)
assert_test(
    clean_v3_candidate_output(wrapped_candidate).startswith("School is changing quickly."),
    "V3 output cleaner removes wrapper text before candidate assessment",
)

academic_report = report_for(cited_source, mode="academic_cited_text", ai=55.0)
academic_contract = build_scan_contract(academic_report, cited_source)
academic_route = route_from_scan_contract(academic_contract)
academic_plan = build_strategy_plan(academic_route, academic_contract)
assert_test(
    academic_route.primary_class == RewriteRiskClass.CITED_ACADEMIC,
    "V3 scanner router maps academic scan route to cited academic risk class",
)
assert_test(
    "cited_practice_voice" in [step.strategy_id for step in academic_plan.steps],
    "V3 strategy plan includes cited academic strategy stack",
)
inventory_contract = build_scan_contract(
    report_with_preservation_inventory(
        cited_source,
        [
            {"text": "Billett (2013)", "kind": "citation", "severity": "hard_exact"},
            {"text": "practice architecture", "kind": "direct_quote", "severity": "hard_exact"},
        ],
    ),
    cited_source,
)
inventory_route = route_from_scan_contract(inventory_contract)
assert_test(
    inventory_contract.citation_anchor_count == 1 and inventory_contract.quote_anchor_count == 1,
    "V3 scan contract reads structured preservation inventory anchors",
)
assert_test(
    RewriteRiskClass.CITED_ACADEMIC in (inventory_route.primary_class, *inventory_route.secondary_classes),
    "V3 preservation inventory can route unknown scans to cited academic handling",
)
light_quote_source = "\n\n".join([broad_source] * 6)
light_quote_report = report_for(light_quote_source, mode="unknown", ai=70.0)
light_quote_report.pop("rewrite_v2_content_route", None)
light_quote_inventory = {
    "anchors": [
        {"text": "what students know", "kind": "quote", "reason": "quoted/source wording"},
        {"text": "how students think", "kind": "quote", "reason": "quoted/source wording"},
    ],
    "quotes": ["what students know", "how students think"],
    "citations": [],
}
light_quote_report["scan_intelligence"] = {"document": {"preservation_inventory": light_quote_inventory}}
light_quote_report["ai_mitigation"] = {
    "rewrite_handoff": {"rewrite_constraints": {"preservation_inventory": light_quote_inventory}}
}
light_quote_contract = build_scan_contract(light_quote_report, light_quote_source)
light_quote_route = route_from_scan_contract(light_quote_contract)
assert_test(
    light_quote_contract.quote_count == 2 and light_quote_contract.quote_anchor_count == 2,
    "V3 scan contract dedupes duplicated quote inventories",
)
assert_test(
    light_quote_route.primary_class == RewriteRiskClass.BROAD_PROSE,
    "V3 routes untyped light quote anchors as broad prose",
)
citation_handoff_report = report_for(cited_source, mode="unknown", ai=55.0)
citation_handoff_report.pop("rewrite_v2_content_route", None)
citation_handoff_report["generation_handoff"] = {
    "section_generation_units": [
        {"citation_keys_used": ["Billett, 2013"], "meaning_inventory": [{"citation_keys": ["Billett, 2013"]}]},
        {"citation_keys_used": ["Billett, 2020"], "meaning_inventory": []},
    ],
    "reference_register": [],
}
citation_handoff_contract = build_scan_contract(citation_handoff_report, cited_source)
citation_handoff_route = route_from_scan_contract(citation_handoff_contract)
assert_test(
    citation_handoff_contract.citation_key_count == 2 and citation_handoff_contract.anchor_preservation_pressure >= 0.5,
    "V3 scan contract reads citation keys as evidence preservation pressure",
)
assert_test(
    citation_handoff_route.primary_class == RewriteRiskClass.CITED_ACADEMIC,
    "V3 routes structured citation handoff as cited academic",
)
broad_contract = build_scan_contract(report_for(broad_source, mode="broad_explanatory_essay", ai=70.0), broad_source)
broad_plan = build_strategy_plan(route_from_scan_contract(broad_contract), broad_contract)
assert_test(
    "plain_reasoning_broad_prose" in [step.strategy_id for step in broad_plan.steps],
    "V3 broad strategy plan includes plain reasoning broad prose layer",
)
assert_test(
    [step.strategy_id for step in broad_plan.steps][0] == "clean_texture_boundary",
    "V3 broad strategy plan starts with clean texture boundary layer",
)
assert_test(
    not v3_pipeline._should_use_chunked_generation(
        source_words=word_count("The country was founded in 1776. It changed quickly."),
        scan_contract=build_scan_contract(report_for("The country was founded in 1776. It changed quickly.", mode="broad_explanatory_essay"), "The country was founded in 1776. It changed quickly."),
        v3_route=route_from_scan_contract(broad_contract),
        exact_anchor_count=1,
    ),
    "V3 does not force broad clean-texture prose into chunks for a simple numeric anchor",
)

hybrid_report = report_for(cited_source, mode="personal_reflection", ai=55.0)
hybrid_contract = build_scan_contract(hybrid_report, cited_source)
hybrid_route = route_from_scan_contract(hybrid_contract)
hybrid_plan = build_strategy_plan(hybrid_route, hybrid_contract)
assert_test(
    RewriteRiskClass.PERSONAL_REFLECTIVE in (hybrid_route.primary_class, *hybrid_route.secondary_classes),
    "V3 scanner router supports personal reflective risk class",
)
assert_test(
    "voice_preservation" in [step.strategy_id for step in hybrid_plan.steps],
    "V3 strategy plan stacks voice preservation for reflective content",
)

with tempfile.TemporaryDirectory() as tmpdir:
    result = run_rewrite_pipeline_v3(
        detect_json=report_for(broad_source, ai=70.0),
        output_dir=tmpdir,
        replay_candidate_records=[{
            "text": broad_candidate,
            "ai": 58.0,
            "wq": 62.0,
            "topk": 78.0,
        }],
        required_ai_drop=20.0,
    )
    summary = result["result"].summary
    assert_test(result["status"] == "rewrite_candidate_generated_needs_external_review", "V3 does not mark externally calibrated improvement as strict mitigation")
    assert_test(summary["rewrite_pipeline_version"] == "rewrite_v3_external_calibrated", "V3 summary declares V3 pipeline version")
    assert_test(summary["strategy_trace"][0]["strategy_family"] == "clean_texture_boundary", "V3 routes broad content to clean texture boundary family")
    assert_test("v3_route" in summary and "v3_strategy_plan" in summary, "V3 summary exposes scanner route and strategy plan")

bad_broad_proxy = evaluate_external_proxy(
    family="document_rhythm",
    reference_ai=69.7,
    candidate_ai=43.34,
    reference_wq=66.33,
    candidate_wq=57.07,
    reference_topk=86.97,
    candidate_topk=81.58,
    compression={"status": "above_ceiling"},
    validation_passed=True,
    compression_accepted=True,
    semantic_safe=True,
)
assert_test(not bad_broad_proxy.accepted, "V3 broad proxy rejects externally failed broad candidate pattern")
assert_test("insufficient_topk_drop" in bad_broad_proxy.reasons, "V3 broad proxy records weak top-k movement")

segment_profile = build_authorship_window_profile(
    source_text=(
        "A short human paragraph with concrete observation.\n\n"
        "A riskier paragraph follows a predictable route and keeps a uniform explanation."
    ),
    segments=[
        {
            "sentence_id": "s001",
            "paragraph_id": "p001",
            "text": "A short human paragraph with concrete observation.",
            "signals": [{"key": "human_anchor_score", "score": 80}],
            "predictability": {"score": 0.1, "top10_ratio": 0.12, "top50_ratio": 0.2},
        },
        {
            "sentence_id": "s002",
            "paragraph_id": "p002",
            "text": "A riskier paragraph follows a predictable route and keeps a uniform explanation.",
            "signals": [{"key": "ai_likelihood", "score": 92}, {"key": "topk_calibrated_risk", "score": 86}],
            "predictability": {"score": 0.76, "top10_ratio": 0.82, "top50_ratio": 0.91},
        },
    ],
    paragraphs=[
        {
            "paragraph_id": "p001",
            "sentence_ids": ["s001"],
            "start_char": 0,
            "end_char": 50,
            "top_signals": [{"key": "human_anchor_score", "score": 80}],
        },
        {
            "paragraph_id": "p002",
            "sentence_ids": ["s002"],
            "start_char": 52,
            "end_char": 132,
            "top_signals": [{"key": "ai_likelihood", "score": 92}, {"key": "topk_calibrated_risk", "score": 86}],
        },
    ],
)
assert_test(segment_profile["num_ai_segments"] == 1, "Scanner authorship windows classify high-risk window separately")
footprint_profile = segment_profile["ai_footprint_profile"]
assert_test(
    footprint_profile["schema_version"] == "ai_footprint_profile.v2"
    and footprint_profile["risky_window_count"] == 1
    and footprint_profile["top_risky_windows"][0]["paragraph_id"] == "p002",
    "Scanner emits AI footprint profile v2 with top risky windows",
)
assert_test(
    "source_text" in segment_profile["windows"][1]
    and segment_profile["windows"][1]["span_integrity"]["passed"],
    "Scanner authorship windows carry stable source text and span integrity",
)
rebuilt_footprint = build_ai_footprint_profile(segment_profile)
assert_test(
    rebuilt_footprint["risky_window_density"] == footprint_profile["risky_window_density"],
    "AI footprint profile aggregates authorship windows deterministically",
)
backfire_delta = v3_pipeline._footprint_delta(
    {"fraction_ai": 0.0, "fraction_ai_assisted": 0.5, "risky_window_density": 0.2, "max_risky_window_words": 20},
    {"fraction_ai": 0.0, "fraction_ai_assisted": 0.7, "risky_window_density": 0.1, "max_risky_window_words": 20},
)
assert_test(
    not backfire_delta["moved"] and not backfire_delta["risk_not_worse"],
    "V3 footprint movement rejects density-only gains when total footprint risk worsens",
)
target_profile = build_rewrite_target_profile(
    source_text=(
        "A short human paragraph with concrete observation.\n\n"
        "A riskier paragraph follows a predictable route and keeps a uniform explanation."
    ),
    authorship_window_profile=segment_profile,
    ai_footprint_profile=footprint_profile,
    preservation_inventory={"anchors": [{"text": "predictable route", "kind": "domain_term", "priority": 70}]},
)
assisted_groups = group_assisted_footprint_windows(
    original_text=(
        "A short human paragraph with concrete observation.\n\n"
        "A riskier paragraph follows a predictable route and keeps a uniform explanation."
    ),
    authorship_window_profile=segment_profile,
    rewrite_target_profile=target_profile,
    max_groups=3,
)
assert_test(
    assisted_groups and assisted_groups[0].operation == "assisted_footprint_paragraph_rewrite",
    "V3 assisted-footprint executor builds paragraph groups from scanner authorship windows",
)
assert_test(
    target_profile["schema_version"] == "rewrite_target_profile.v1"
    and target_profile["targets"]
    and target_profile["targets"][0]["source_text"],
    "Scanner emits rewrite target profile v1 with stable target text",
)
assert_test(
    target_profile["targets"][0]["scope_level"] in {"span", "sentence_window", "paragraph", "chunk"}
    and "target_anchor_pressure" in target_profile["targets"][0]
    and isinstance(target_profile["targets"][0]["operation_candidates"], list),
    "Rewrite target profile exposes hierarchy and target-level strategy eligibility",
)
problem_inventory = build_problem_inventory(
    rewrite_target_profile=target_profile,
    ai_footprint_profile=footprint_profile,
)
assert_test(
    problem_inventory["schema_version"] == "problem_inventory.v1"
    and problem_inventory["problem_groups"]
    and problem_inventory["problem_groups"][0]["allowed_operations"],
    "Scanner emits problem inventory v1 from structured target and footprint evidence",
)
broad_target_profile = build_rewrite_target_profile(
    source_text=broad_source,
    authorship_window_profile={
        "windows": [
            {
                "window_id": "w001",
                "paragraph_id": "p001",
                "sentence_ids": ["s001", "s002"],
                "label": "moderately_ai_assisted",
                "ai_assistance_score": 0.66,
                "start_index": 0,
                "end_index": broad_source.find("\n\n"),
                "span_integrity": {"passed": True, "start_index": 0, "end_index": broad_source.find("\n\n")},
                "source_text": broad_source.split("\n\n")[0],
                "word_count": len(broad_source.split("\n\n")[0].split()),
                "score_components": {"unsafe_word_share": 0.8, "predictability_score": 0.7, "ai_signal_score": 0.6},
            },
            {
                "window_id": "w002",
                "paragraph_id": "p002",
                "sentence_ids": ["s003", "s004"],
                "label": "moderately_ai_assisted",
                "ai_assistance_score": 0.64,
                "start_index": broad_source.find("\n\n") + 2,
                "end_index": len(broad_source),
                "span_integrity": {"passed": True, "start_index": broad_source.find("\n\n") + 2, "end_index": len(broad_source)},
                "source_text": broad_source.split("\n\n")[1],
                "word_count": len(broad_source.split("\n\n")[1].split()),
                "score_components": {"unsafe_word_share": 0.72, "predictability_score": 0.66, "ai_signal_score": 0.57},
            },
        ]
    },
    ai_footprint_profile={
        "fraction_ai": 0.0,
        "fraction_ai_assisted": 1.0,
        "risky_window_density": 1.0,
        "risky_window_count": 2,
    },
    preservation_inventory={"anchors": [{"text": "old classroom model", "kind": "domain_term", "priority": 70}]},
)
broad_inventory = build_problem_inventory(
    rewrite_target_profile=broad_target_profile,
    ai_footprint_profile={
        "fraction_ai": 0.0,
        "fraction_ai_assisted": 1.0,
        "risky_window_density": 1.0,
        "risky_window_count": 2,
    },
)
broad_problem_contract = build_scan_contract(
    {
        **report_for(broad_source, ai=70.0),
        "rewrite_target_profile": broad_target_profile,
        "problem_inventory": broad_inventory,
        "ai_footprint_profile": {
            "fraction_ai": 0.0,
            "fraction_ai_assisted": 1.0,
            "fraction_human": 0.0,
            "risky_window_density": 1.0,
            "risky_window_count": 2,
            "max_risky_window_words": 48,
        },
    },
    broad_source,
)
broad_problem_plan = build_strategy_plan(route_from_scan_contract(broad_problem_contract), broad_problem_contract)
assert_test(
    broad_target_profile["operation_mix"].get("paragraph_preserving_broad_reconstruction") == 2
    and not broad_target_profile["targets"][0]["protected_anchors"]
    and broad_target_profile["targets"][0]["soft_guidance_anchors"]
    and broad_problem_plan.steps[0].strategy_id == "paragraph_preserving_broad_reconstruction",
    "V3 routes broad soft-anchor footprint to paragraph-preserving reconstruction first",
)
low_anchor_target_profile = {
    "schema_version": "rewrite_target_profile.v1",
    "document_shape": "mixed",
    "target_scope_policy": "target_profile_driven",
    "operation_mix": {"grounded_author_reasoning_rewrite": 2},
    "targets": [
        {
            "target_id": "rt001",
            "unit_id": "p001",
            "paragraph_id": "p001",
            "scope_level": "sentence_window",
            "risk_level": "medium",
            "target_anchor_pressure": 0.0,
            "semantic_edit_cost": 0.1,
            "operation_candidates": ["unit_preserving_prune_bridge", "authorship_window_repair"],
            "recommended_operation": "grounded_author_reasoning_rewrite",
            "source_text": "This approach benefits all students.",
            "protected_anchors": [],
            "word_count_guide": {"source_words": 5, "preferred_words": 5},
        },
        {
            "target_id": "rt002",
            "unit_id": "p001",
            "paragraph_id": "p001",
            "scope_level": "sentence_window",
            "risk_level": "medium",
            "target_anchor_pressure": 0.0,
            "semantic_edit_cost": 0.1,
            "operation_candidates": ["unit_preserving_prune_bridge"],
            "recommended_operation": "light_texture_rewrite",
            "source_text": "Students can then apply it.",
            "protected_anchors": [],
            "word_count_guide": {"source_words": 5, "preferred_words": 5},
        },
    ],
}
low_anchor_inventory = build_problem_inventory(
    rewrite_target_profile=low_anchor_target_profile,
    ai_footprint_profile={
        "fraction_ai": 0.0,
        "fraction_ai_assisted": 0.8,
        "risky_window_density": 0.12,
        "risky_window_count": 2,
    },
)
low_anchor_contract = build_scan_contract(
    {
        **report_for("This approach benefits all students.\n\nStudents can then apply it.", mode="academic_cited_text", ai=35.0),
        "rewrite_target_profile": low_anchor_target_profile,
        "problem_inventory": low_anchor_inventory,
        "ai_footprint_profile": {
            "fraction_ai": 0.0,
            "fraction_ai_assisted": 0.8,
            "fraction_human": 0.2,
            "risky_window_density": 0.12,
            "risky_window_count": 2,
            "max_risky_window_words": 9,
        },
    },
    "This approach benefits all students.\n\nStudents can then apply it.",
)
low_anchor_plan = build_strategy_plan(route_from_scan_contract(low_anchor_contract), low_anchor_contract)
assert_test(
    [step.strategy_id for step in low_anchor_plan.steps][0] == "unit_preserving_prune_bridge",
    "V3 problem-first strategy plan selects prune/bridge for localized low-anchor scanner targets",
)
assert_test(
    target_profile["targets"][0]["recommended_operation"] in {
        "paragraph_preserving_broad_reconstruction",
        "grounded_author_reasoning_rewrite",
        "citation_preserving_window_repair",
        "protected_section_rewrite",
        "light_texture_rewrite",
    },
    "Rewrite target profile declares allowed target operation",
)
target_groups = group_rewrite_targets(
    original_text=(
        "A short human paragraph with concrete observation.\n\n"
        "A riskier paragraph follows a predictable route and keeps a uniform explanation."
    ),
    rewrite_target_profile=target_profile,
    max_groups=2,
)
prune_groups = filter_prune_bridge_groups(
    target_groups=group_rewrite_targets(
        original_text="This approach benefits all students.\n\nStudents can then apply it.",
        rewrite_target_profile=low_anchor_target_profile,
        max_groups=2,
    ),
    problem_inventory=low_anchor_inventory,
)
assert_test(
    prune_groups and prune_groups[0].operation == "unit_preserving_prune_bridge",
    "V3 prune/bridge executor filters target groups from problem inventory",
)
predictability_briefs_fixture = [
    {
        "sentence_id": "s002",
        "paragraph_id": target_groups[0].unit_id,
        "target_sentence": target_groups[0].source_text,
        "predictable_token_spans": ["predictable route", "uniform explanation", "outside context phrase"],
        "problem_tokens": ["predictable", "uniform"],
    }
]
prune_prompt = build_prune_bridge_prompt(
    target_groups=prune_groups,
    predictability_briefs=predictability_briefs_fixture,
)
assert_test(
    "unit_preserving_prune_bridge" in prune_prompt
    and "scanner_action_contract" in prune_prompt
    and "Return JSON only" in prune_prompt,
    "V3 prune/bridge executor builds bounded scanner-target prompt",
)
prune_replacements = parse_prune_bridge_replacements(json.dumps({
    "replacements": [
        {"group_id": prune_groups[0].group_id, "replacement_text": "This helps students."}
    ]
}))
prune_applied, prune_trace = apply_prune_bridge_replacements(
    original_text="This approach benefits all students.\n\nStudents can then apply it.",
    target_groups=prune_groups,
    replacements=prune_replacements,
)
assert_test(
    "This helps students." in prune_applied and prune_trace["problem_strategy"] == "unit_preserving_prune_bridge",
    "V3 prune/bridge executor applies unit-preserving replacements",
)
assert_test(
    target_groups and target_groups[0].source_text,
    "V3 target executor groups scanner targets into bounded replacement units",
)
target_prompt = build_target_executor_prompt(
    target_groups=target_groups,
    content_mode="broad_explanatory_essay",
    strategy_family="grounded_author_reasoning_rewrite",
    predictability_briefs=predictability_briefs_fixture,
)
assert_test(
    "scanner_action_contract" in target_prompt
    and "ownership_contract" in target_prompt
    and "predictable_spans" in target_prompt
    and "predictable route" in target_prompt
    and "outside context phrase" not in target_prompt
    and "locality_limits" in target_prompt,
    "V3 target executor prompt carries exact scanner span repair and ownership contracts",
)
ownership_contract = ownership_contract_for_group(target_groups[0])
assert_test(
    ownership_contract["golden_rule"].startswith("Do not just change the point of view")
    and ownership_contract["required_elements"] == ["author_trace", "specific_context", "real_judgment"]
    and "soft_guidance_anchors" in ownership_contract["available_trace_sources"],
    "V3 ownership contract encodes author trace, specific context, and real judgment without topic routing",
)
span_quality_source = "Inclusive Learning Design is essential in Hairdressing Certificate III as learner diversity increases, making a uniform approach ineffective."
span_quality_contract = topk_repair_contract_for_group(
    group={
        "unit_id": "p027",
        "paragraph_id": "p027",
        "sentence_ids": ["s027"],
        "source_text": span_quality_source,
        "word_count_guide": {"preferred_words": 39},
        "targets": [],
    },
    predictability_briefs=[
        {
            "paragraph_id": "p027",
            "sentence_id": "s027",
            "target_sentence": span_quality_source,
            "predictable_token_spans": ["irdressing", "increases,"],
            "problem_tokens": ["ressing", "increases"],
        }
    ],
)
assert_test(
    "irdressing" not in span_quality_contract["predictable_spans_in_source"]
    and "increases," not in span_quality_contract["predictable_spans_in_source"]
    and "as learner diversity increases" in span_quality_contract["predictable_spans_in_source"]
    and span_quality_contract["predictable_span_rows"][0]["id"] == "ps001"
    and span_quality_contract["required_modified_spans"] == 1
    and span_quality_contract["expanded_predictable_spans"],
    "V3 prompt contract expands raw token fragments into id-addressable phrase-level repair spans",
)
full_layer_contract = build_rewrite_contract(
    "A short human paragraph with concrete observation.\n\n"
    "A riskier paragraph follows a predictable route and keeps a uniform explanation.",
    content_mode="broad_explanatory_essay",
)
full_layer_policy = compression_policy_for_family("clean_texture_boundary", word_count(target_groups[0].source_text))
full_layer_source_units = [{
    "unit_id": target_groups[0].unit_id,
    "text": target_groups[0].source_text,
    "word_count": word_count(target_groups[0].source_text),
}]
full_layer_prompts = [
    build_clean_texture_boundary_prompt(
        original_text=target_groups[0].source_text,
        scan_report=report_for(target_groups[0].source_text),
        rewrite_target_profile=target_profile,
        predictability_briefs=predictability_briefs_fixture,
    ),
    build_document_rhythm_prompt(
        original_text=target_groups[0].source_text,
        compression_policy=full_layer_policy,
        rewrite_target_profile=target_profile,
        predictability_briefs=predictability_briefs_fixture,
    ),
    build_cited_practice_voice_prompt(
        original_text=target_groups[0].source_text,
        contract=full_layer_contract,
        compression_policy=full_layer_policy,
        rewrite_target_profile=target_profile,
        predictability_briefs=predictability_briefs_fixture,
    ),
    build_plain_reasoning_broad_prose_prompt(
        original_text=target_groups[0].source_text,
        failed_candidates=[],
        compression_policy=full_layer_policy,
        style_examples={"positive": [], "negative": []},
        rewrite_target_profile=target_profile,
        predictability_briefs=predictability_briefs_fixture,
    ),
    build_clean_texture_boundary_chunk_prompt(
        source_units=full_layer_source_units,
        global_plan={"strategy": "test"},
        rewrite_target_profile=target_profile,
        predictability_briefs=predictability_briefs_fixture,
    ),
    build_cited_practice_voice_chunk_prompt(
        source_units=full_layer_source_units,
        contract=full_layer_contract,
        global_plan={"strategy": "test"},
        compression_policy=full_layer_policy,
        rewrite_target_profile=target_profile,
        predictability_briefs=predictability_briefs_fixture,
    ),
    build_document_rhythm_chunk_prompt(
        source_units=full_layer_source_units,
        global_plan={"strategy": "test"},
        compression_policy=full_layer_policy,
        rewrite_target_profile=target_profile,
        predictability_briefs=predictability_briefs_fixture,
    ),
]
assert_test(
    all("scanner_action_contracts" in prompt for prompt in full_layer_prompts)
    and all("ownership_contract" in prompt for prompt in full_layer_prompts)
    and all("predictable_spans" in prompt for prompt in full_layer_prompts)
    and all("source_structure_contract" in prompt for prompt in full_layer_prompts)
    and all("predictable route" in prompt for prompt in full_layer_prompts),
    "V3 fallback prompt constructors carry scanner action, ownership, exact predictable span, and structure contracts",
)
assert_test(
    target_groups[0].source_text.startswith("A riskier paragraph follows"),
    "V3 target executor expands scanner windows to stable paragraph groups when paragraph ids exist",
)
target_replacements = parse_target_replacements(json.dumps({
    "replacements": [{
        "group_id": target_groups[0].group_id,
        "replacement_text": "A riskier paragraph follows a predictable route, but it now explains the local reason in a less uniform way.",
    }]
}))
target_applied, target_apply_status = apply_target_replacements(
    original_text=(
        "A short human paragraph with concrete observation.\n\n"
        "A riskier paragraph follows a predictable route and keeps a uniform explanation."
    ),
    target_groups=target_groups,
    replacements=target_replacements,
)
assert_test(
    "less uniform way" in target_applied and target_apply_status[0]["applied"],
    "V3 target executor applies scanner-span replacements back into the original document",
)
boundary_source = "Inclusive design opens learning opportunities for learners."
boundary_profile = {
    "targets": [{
        "target_id": "rt_boundary",
        "unit_id": "u_boundary",
        "recommended_operation": "light_texture_rewrite",
        "source_text": "learning opportunities ",
        "span": {
            "start_index": boundary_source.find("learning"),
            "end_index": boundary_source.find("for"),
            "integrity": {"passed": True},
        },
        "word_count_guide": {"source_words": 2, "preferred_words": 2},
    }]
}
boundary_groups = group_rewrite_targets(original_text=boundary_source, rewrite_target_profile=boundary_profile)
boundary_applied, _ = apply_target_replacements(
    original_text=boundary_source,
    target_groups=boundary_groups,
    replacements=[{"group_id": boundary_groups[0].group_id, "replacement_text": "learning chances"}],
)
assert_test(
    "chances for" in boundary_applied and "chancesfor" not in boundary_applied,
    "V3 target executor preserves span-boundary spacing during replacement",
)
batched_target_groups = batch_target_groups(target_groups * 5, batch_size=2)
assert_test(
    len(batched_target_groups) == 3 and all(len(batch) <= 2 for batch in batched_target_groups),
    "V3 target executor can process scanner targets in bounded batches",
)
soft_anchor_applied, soft_anchor_status = apply_target_replacements(
    original_text=(
        "A short human paragraph with concrete observation.\n\n"
        "A riskier paragraph follows a predictable route and keeps a uniform explanation."
    ),
    target_groups=target_groups,
    replacements=[{"group_id": target_groups[0].group_id, "replacement_text": "This replacement changes the soft phrase but keeps the same role."}],
)
assert_test(
    soft_anchor_status[0]["applied"],
    "V3 target executor does not block paragraph replacement on soft guidance anchors",
)
hard_anchor_profile = {
    "targets": [{
        "target_id": "rt_hard",
        "unit_id": "p001",
        "paragraph_id": "p001",
        "recommended_operation": "light_texture_rewrite",
        "source_text": "The class used 42 examples during revision.",
        "span": {"start_index": 0, "end_index": 43, "integrity": {"passed": True}},
        "protected_anchors": [{"text": "42", "kind": "number", "blocking": True}],
        "word_count_guide": {"source_words": 7, "preferred_words": 7},
    }]
}
hard_anchor_groups = group_rewrite_targets(
    original_text="The class used 42 examples during revision.",
    rewrite_target_profile=hard_anchor_profile,
)
missing_anchor_applied, missing_anchor_status = apply_target_replacements(
    original_text="The class used 42 examples during revision.",
    target_groups=hard_anchor_groups,
    replacements=[{"group_id": hard_anchor_groups[0].group_id, "replacement_text": "The class used examples during revision."}],
)
assert_test(
    "required phrase" not in missing_anchor_applied and not missing_anchor_status[0]["applied"],
    "V3 target executor blocks replacements that drop hard protected anchors",
)
out_of_scope_anchor_group = TargetGroup(
    group_id="tg_out_of_scope_anchor",
    unit_id="p_out_of_scope_anchor",
    operation="paragraph_preserving_broad_reconstruction",
    start_index=0,
    end_index=len("This paragraph mentions shortcut culture and 1991 only."),
    source_text="This paragraph mentions shortcut culture and 1991 only.",
    before_context="",
    after_context="",
    targets=(),
    protected_anchors=(
        {"text": "(CESE, 2017)", "kind": "citation", "blocking": True},
        {"text": "shortcut culture", "kind": "quote", "blocking": True},
        {"text": "1", "kind": "number", "blocking": True},
    ),
    word_count_guide={"source_words": 8, "preferred_words": 8},
)
out_of_scope_applied, out_of_scope_status = apply_target_replacements(
    original_text="This paragraph mentions shortcut culture and 1991 only.",
    target_groups=[out_of_scope_anchor_group],
    replacements=[{
        "group_id": "tg_out_of_scope_anchor",
        "replacement_text": "This paragraph keeps shortcut culture while changing the local wording around 1991.",
    }],
)
assert_test(
    out_of_scope_status[0]["applied"]
    and "(CESE, 2017)" not in out_of_scope_applied,
    "V3 target executor ignores protected anchors that are outside the target source text",
)
scanner_loop_report = {
    "rewrite_edit_briefs": predictability_briefs_fixture,
    "scan_intelligence": {
        "blocker_radar": {
            "dominant_blockers": [
                {
                    "key": "topk_pattern",
                    "score": 75,
                    "scope": "localized",
                    "severity": "high",
                    "paragraph_ids": [target_groups[0].unit_id],
                    "sentence_ids": ["s002"],
                    "diagnostic_flags": {"texture_pressure": True},
                },
                {
                    "key": "unsupported_claim_risk",
                    "score": 68,
                    "scope": "document_wide",
                    "severity": "high",
                    "paragraph_ids": [],
                    "sentence_ids": [],
                    "diagnostic_flags": {"evidence_gap": True},
                },
            ]
        },
        "human_contribution_contract": {
            "weak_subsignals": ["causal_reasoning"],
            "subsignals": [{
                "key": "causal_reasoning",
                "score": 32,
                "label": "weak",
                "rewrite_lever": "Make one supported cause-effect relation explicit.",
            }],
        },
    }
}
scanner_loop_goal = {
    "eligible_span_density_gate": {
        "top_sentence_targets": [{"sentence_id": "s002", "risk_score": 4.5}],
    }
}
ranked_scanner_groups = rank_scanner_target_groups(
    report=scanner_loop_report,
    goal=scanner_loop_goal,
    groups=target_groups,
)
assert_test(
    ranked_scanner_groups[0].group_id == target_groups[0].group_id,
    "V3 scanner-controlled executor ranks target groups from scanner blockers and unsafe sentences",
)
scanner_loop_prompt = build_scanner_controlled_prompt(
    report=scanner_loop_report,
    group=target_groups[0],
    variants_per_group=2,
)
ownership_repair_prompt = build_scanner_controlled_prompt(
    report=scanner_loop_report,
    group=target_groups[0],
    variants_per_group=2,
    ownership_repair_mode=True,
)
assert_test(
    "movement_contract" in scanner_loop_prompt
    and "scanner_action_contract" in scanner_loop_prompt
    and "ownership_contract" in scanner_loop_prompt
    and "Do not just change point of view" in scanner_loop_prompt
    and "weak_human_levers" in scanner_loop_prompt
    and "source_text" in scanner_loop_prompt,
    "V3 scanner-controlled prompt carries movement, ownership, human levers, and local source only",
)
assert_test(
    '"repair_mode": "claim_ownership_repair"' in ownership_repair_prompt
    and ownership_repair_prompt.count('"operator": "CLAIM_OWNERSHIP_REPAIR"') == 2
    and "every kept variant must report at least one real ownership_changes row" in ownership_repair_prompt,
    "V3 ownership repair prompt runs a bounded claim-ownership operator portfolio",
)
assert_test(
    "blocker_radar_for_this_group" not in scanner_loop_prompt
    and "target_drivers" not in scanner_loop_prompt,
    "V3 scanner-controlled prompt does not dump raw blocker radar or target driver payloads",
)
anchor_prompt = build_scanner_controlled_prompt(
    report=scanner_loop_report,
    group=hard_anchor_groups[0],
    variants_per_group=3,
)
assert_test(
    "variant_plan" in anchor_prompt
    and "CLAUSE_ROUTE_CHANGE" in anchor_prompt
    and "BROAD_CLAIM_NARROWING" in anchor_prompt
    and "CAUSE_EFFECT_OWNERSHIP" in anchor_prompt,
    "V3 scanner-controlled prompt assigns operator-shaped variants",
)
assert_test(
    "changed_spans" in scanner_loop_prompt
    and "predictable_spans_modified_count" in scanner_loop_prompt
    and "required_modified_spans" in scanner_loop_prompt
    and "predictable_spans_in_source" in scanner_loop_prompt,
    "V3 scanner-controlled prompt requires declared predictable-span movement",
)
assert_test(
    "omit it entirely" in scanner_loop_prompt
    and "do not return no-op variants" in scanner_loop_prompt
    and "Do not intensify source claims" in scanner_loop_prompt,
    "V3 scanner-controlled prompt omits failed variants and blocks semantic-intensity drift",
)
citation_pressure_contract = group_action_contract(
    group={
        "operation": "protected_section_rewrite",
        "unit_id": "p_citation",
        "paragraph_id": "p_citation",
        "source_text": "Billett (2013) and CAST (2024) say learning connects with how each person thinks.",
        "word_count_guide": {"preferred_words": 13},
        "targets": [{
            "target_id": "rt_citation",
            "target_anchor_pressure": 0.9,
            "operation_candidates": ["citation_preserving_window_repair", "protected_section_rewrite"],
        }],
    },
    predictability_briefs=[],
)
assert_test(
    citation_pressure_contract["citation_pressure_zone"]
    and "source-like wording" in citation_pressure_contract["citation_zone_instruction"],
    "V3 scanner prompt contract flags citation-pressure zones against academic smoothing",
)
assert_test(
    citation_pressure_contract["ownership_contract"]["required_move"] == "CLAIM_OWNERSHIP_REPAIR"
    and "point-of-view swap without added ownership" in citation_pressure_contract["ownership_contract"]["forbidden_moves"],
    "V3 scanner prompt contract makes ownership repair a first-class movement",
)
assert_test(
    "[[DP_ANCHOR_001]]" in anchor_prompt
    and '"source_text": "The class used [[DP_ANCHOR_001]] examples during revision."' in anchor_prompt,
    "V3 scanner-controlled prompt freezes protected anchors as placeholders",
)
numeric_anchor_frozen = freeze_protected_anchors(
    "Billett (2013) and CAST (2024) use a 2-D chart with a 45-degree angle from 0 to 180.",
    [
        {"placeholder": "[[DP_ANCHOR_001]]", "text": "0"},
        {"placeholder": "[[DP_ANCHOR_002]]", "text": "1"},
        {"placeholder": "[[DP_ANCHOR_003]]", "text": "2"},
        {"placeholder": "[[DP_ANCHOR_004]]", "text": "45"},
    ],
)
assert_test(
    "2013" in numeric_anchor_frozen
    and "2024" in numeric_anchor_frozen
    and "[[DP_ANCHOR_003]]-D" in numeric_anchor_frozen
    and "[[DP_ANCHOR_004]]-degree" in numeric_anchor_frozen
    and "[[DP_ANCHOR_00[[" not in numeric_anchor_frozen
    and protected_placeholder_integrity(numeric_anchor_frozen)["passed"],
    "V3 scanner-controlled anchor freezing avoids nested numeric placeholders",
)
malformed_anchor_gate = scanner_controlled_variant_gate(
    report=scanner_loop_report,
    group=hard_anchor_groups[0],
    variant={
        "variant_id": "v_bad_anchor",
        "replacement_text": "The class used [[DP_ANCHOR_00[[DP_ANCHOR_001]]]] examples during revision.",
    },
    replacement_text="The class used 42 examples during revision.",
)
assert_test(
    malformed_anchor_gate["reason"] == "anchor_placeholder_corruption",
    "V3 scanner-controlled gate rejects malformed anchor placeholders before scan",
)
partial_anchor_group = TargetGroup(
    group_id="tg_partial_anchor",
    unit_id="p_partial_anchor",
    operation="protected_section_rewrite",
    start_index=0,
    end_index=80,
    source_text="Students need 12-15 months to get good at the structures.",
    before_context="",
    after_context="",
    targets=(
        {
            "target_id": "rt_partial_anchor",
            "paragraph_id": "p_partial_anchor",
            "sentence_ids": ["s_partial_anchor"],
            "dominant_drivers": [{"key": "predictability_score", "score": 0.6}],
            "protected_anchors": [
                {"text": "1", "kind": "number"},
                {"text": "12", "kind": "number"},
                {"text": "15 months", "kind": "number"},
                {"text": "2", "kind": "number"},
            ],
        },
    ),
    word_count_guide={"preferred_words": 10},
)
partial_anchor_report = {
    "rewrite_edit_briefs": [
        {
            "paragraph_id": "p_partial_anchor",
            "sentence_id": "s_partial_anchor",
            "target_sentence": partial_anchor_group.source_text,
            "predictable_token_spans": ["get good at"],
        }
    ]
}
partial_anchor_gate = scanner_controlled_variant_gate(
    report=partial_anchor_report,
    group=partial_anchor_group,
    variant={
        "variant_id": "v_partial_anchor",
        "replacement_text": "Students need [[DP_ANCHOR_002]]-[[DP_ANCHOR_003]] to become confident with the structures.",
        "changed_spans": [{"source_span": "get good at", "before": "get good at", "after": "become confident with"}],
        "predictable_spans_modified_count": 1,
    },
    replacement_text="Students need 12-15 months to become confident with the structures.",
)
assert_test(
    partial_anchor_gate["passed"]
    and partial_anchor_gate["reason"] == "passed",
    "V3 scanner-controlled gate only requires placeholders present in the target source text",
)
scanner_variants = parse_scanner_controlled_variants(
    '{"variants":[{"variant_id":"v1","replacement_text":"A local variant.","changed_spans":[{"source_span":"predictable route","before":"predictable route","after":"local route","operation":"TOPK_SPAN_REPATH"}],"predictable_spans_modified_count":1},{"variant_id":"v2","replacement_text":"Another local variant."}]}',
    limit=2,
)
assert_test(
    len(scanner_variants) == 2
    and scanner_variants[0]["replacement_text"] == "A local variant."
    and scanner_variants[0]["predictable_spans_modified_count"] == 1,
    "V3 scanner-controlled executor parses bounded local variants",
)
no_change_gate = scanner_controlled_variant_gate(
    report=scanner_loop_report,
    group=target_groups[0],
    variant={"variant_id": "v_same", "replacement_text": target_groups[0].source_text},
    replacement_text=target_groups[0].source_text,
)
weak_span_gate = scanner_controlled_variant_gate(
    report=scanner_loop_report,
    group=target_groups[0],
    variant={
        "variant_id": "v_weak",
        "replacement_text": target_groups[0].source_text.replace("predictable route", "local route"),
        "changed_spans": [{"source_span": "predictable route", "before": "predictable route", "after": "local route"}],
        "predictable_spans_modified_count": 1,
    },
    replacement_text=target_groups[0].source_text.replace("predictable route", "local route"),
)
strong_span_gate = scanner_controlled_variant_gate(
    report=scanner_loop_report,
    group=target_groups[0],
    variant={
        "variant_id": "v_strong",
        "replacement_text": target_groups[0].source_text.replace("predictable route", "local route").replace("uniform explanation", "specific explanation"),
        "changed_spans": [
            {"source_span": "predictable route", "before": "predictable route", "after": "local route"},
            {"source_span": "uniform explanation", "before": "uniform explanation", "after": "specific explanation"},
        ],
        "predictable_spans_modified_count": 2,
    },
    replacement_text=target_groups[0].source_text.replace("predictable route", "local route").replace("uniform explanation", "specific explanation"),
)
self_report_mismatch_gate = scanner_controlled_variant_gate(
    report=scanner_loop_report,
    group=target_groups[0],
    variant={
        "variant_id": "v_fake_count",
        "replacement_text": target_groups[0].source_text.replace("predictable route", "local route"),
        "changed_spans": [{"source_span": "predictable route", "before": "predictable route", "after": "local route"}],
        "modified_span_ids": ["ps001", "ps002"],
        "predictable_spans_modified_count": 2,
    },
    replacement_text=target_groups[0].source_text.replace("predictable route", "local route"),
)
ownership_required_gate = scanner_controlled_variant_gate(
    report=scanner_loop_report,
    group=target_groups[0],
    variant={
        "variant_id": "v_ownership",
        "replacement_text": target_groups[0].source_text.replace("predictable route", "local route"),
        "changed_spans": [{"source_span": "predictable route", "before": "predictable route", "after": "local route"}],
        "predictable_spans_modified_count": 1,
    },
    replacement_text=target_groups[0].source_text.replace("predictable route", "local route"),
    require_ownership=True,
)
ownership_pass_gate = scanner_controlled_variant_gate(
    report=scanner_loop_report,
    group=target_groups[0],
    variant={
        "variant_id": "v_ownership_pass",
        "replacement_text": target_groups[0].source_text.replace("predictable route", "local route"),
        "changed_spans": [{"source_span": "predictable route", "before": "predictable route", "after": "local route"}],
        "predictable_spans_modified_count": 1,
        "ownership_changes": [{"before": "generic claim", "after": "claim tied to this source context"}],
        "ownership_elements_supported": ["specific_context"],
    },
    replacement_text=target_groups[0].source_text.replace("predictable route", "local route"),
    require_ownership=True,
)
assert_test(
    no_change_gate["reason"] == "no_material_change"
    and weak_span_gate["passed"]
    and weak_span_gate["required_predictable_spans_modified"] == 1
    and strong_span_gate["passed"]
    and self_report_mismatch_gate["passed"]
    and self_report_mismatch_gate["self_report_mismatch"]
    and self_report_mismatch_gate["declared_predictable_spans_modified_count"] == 2,
    "V3 scanner-controlled gate rejects no-change variants, uses dynamic short-target thresholds, and flags fake span counts",
)
assert_test(
    ownership_required_gate["reason"] == "ownership_repair_required"
    and ownership_pass_gate["passed"]
    and ownership_pass_gate["ownership_change_count"] == 1,
    "V3 scanner-controlled ownership mode rejects no-ownership variants and accepts source-grounded ownership movement",
)
quality_score = scanner_controlled_candidate_quality(
    action_contract=group_action_contract(group=target_groups[0], predictability_briefs=predictability_briefs_fixture),
    variant_gate=strong_span_gate,
    source_text=target_groups[0].source_text,
    replacement_text=target_groups[0].source_text.replace("predictable route", "local route").replace("uniform explanation", "specific explanation"),
    variant={
        "ownership_changes": [
            {
                "before": "generic claim",
                "after": "claim tied to the local classroom context",
                "operation": "CLAIM_OWNERSHIP_REPAIR",
            }
        ],
        "ownership_elements_supported": ["specific_context", "real_judgment"],
        "changed_spans": [
            {"span_id": "ps001", "source_span": "predictable route", "before": "predictable route", "after": "local route"},
            {"span_id": "ps002", "source_span": "uniform explanation", "before": "uniform explanation", "after": "specific explanation"},
        ],
    },
)
citation_voice_contract = {
    "citation_pressure_zone": True,
    "topk_repair_contract": {"required_modified_spans": 2, "locality_limits": {"max_changed_spans": 3}},
}
plain_voice_quality = scanner_controlled_candidate_quality(
    action_contract=citation_voice_contract,
    variant_gate={"predictable_spans_modified_count": 2, "required_predictable_spans_modified": 2, "actual_modified_span_ids": ["ps001", "ps002"]},
    source_text="From my experience, watching videos isn't enough; they need to practice under my watch and get feedback.",
    replacement_text="From my experience, watching videos is not enough; they need to practice under my watch and get feedback.",
    variant={
        "operator_used": "BROAD_CLAIM_NARROWING",
        "changed_spans": [{"span_id": "ps001", "operation": "TOPK_SPAN_REPATH"}],
    },
)
polished_voice_quality = scanner_controlled_candidate_quality(
    action_contract=citation_voice_contract,
    variant_gate={"predictable_spans_modified_count": 2, "required_predictable_spans_modified": 2, "actual_modified_span_ids": ["ps001", "ps002"]},
    source_text="From my experience, watching videos isn't enough; they need to practice under my watch and get feedback.",
    replacement_text="From my experience, watching videos is insufficient; they need to practice under my supervision and receive feedback.",
    variant={
        "operator_used": "CLAUSE_ROUTE_CHANGE",
        "changed_spans": [{"span_id": "ps001", "operation": "TOPK_SPAN_REPATH"}],
    },
)
assert_test(
    quality_score["score"] > 0
    and quality_score["movement_score"] > 0
    and quality_score["ownership_score"] > 0
    and "ps001" in quality_score["actual_modified_span_ids"],
    "V3 scanner-controlled candidate quality scores span movement, ownership, and locality before scan selection",
)
assert_test(
    plain_voice_quality["score"] > polished_voice_quality["score"]
    and plain_voice_quality["source_likeness_score"] > polished_voice_quality["source_likeness_score"]
    and polished_voice_quality["novel_token_penalty"] > plain_voice_quality["novel_token_penalty"]
    and plain_voice_quality["span_operations"] == ["TOPK_SPAN_REPATH"],
    "V3 scanner-controlled candidate quality prefers source-like citation-zone edits over polished or novel rewrites",
)
assert_test(
    scanner_controlled_rank({"footprint_risk": 38, "external_proxy_score": 27, "topk": 70, "ai": 29, "unsafe_cluster_count": 2, "unsafe_word_ratio": 9})
    < scanner_controlled_rank({"footprint_risk": 42, "external_proxy_score": 40, "topk": 74, "ai": 32, "unsafe_cluster_count": 8, "unsafe_word_ratio": 18}),
    "V3 scanner-controlled rank rewards footprint, proxy, top-k, and unsafe-cluster movement",
)
assert_test(
    v3_pipeline._scanner_controlled_config().max_rounds >= 1
    and ScannerControlledConfig(max_rounds=2).to_dict()["max_rounds"] == 2,
    "V3 scanner-controlled executor exposes bounded runtime config",
)
segment_gate = evaluate_authorship_window_gate(segment_profile)
assert_test(not segment_gate.passed, "V3 authorship window gate fails high-risk segment profile")
segment_targets = select_authorship_window_targets(segment_profile, max_targets=1)
assert_test(
    len(segment_targets) == 1 and segment_targets[0]["label"] == "ai_generated",
    "V3 authorship window gate selects worst failed window",
)
window_prompt = build_authorship_window_repair_prompt(
    candidate_text=(
        "A short human paragraph with concrete observation.\n\n"
        "A riskier paragraph follows a predictable route and keeps a uniform explanation."
    ),
    target_windows=segment_targets,
    strategy_family="document_rhythm",
    contract=build_rewrite_contract("A short human paragraph.\n\nA riskier paragraph.", content_mode="broad_explanatory_essay"),
)
assert_test("selected_windows_only" in window_prompt, "V3 authorship repair prompt is window-scoped")
window_replacements = extract_authorship_window_replacements(
    '{"replacements":[{"window_id":"w002","replacement_text":"The second paragraph needs a more specific local reason, so it names the actual tension instead of smoothing it away."}]}'
)
window_repaired = apply_authorship_window_replacements(
    candidate_text=(
        "A short human paragraph with concrete observation.\n\n"
        "A riskier paragraph follows a predictable route and keeps a uniform explanation."
    ),
    target_windows=segment_targets,
    replacements=window_replacements,
)
assert_test("actual tension" in window_repaired, "V3 authorship repair applies replacement to failed window")
segment_proxy = evaluate_external_proxy(
    family="document_rhythm",
    reference_ai=70.0,
    candidate_ai=45.0,
    reference_wq=65.0,
    candidate_wq=63.0,
    reference_topk=88.0,
    candidate_topk=70.0,
    candidate_authorship_profile=segment_profile,
    compression={"status": "in_band"},
    validation_passed=True,
    compression_accepted=True,
    semantic_safe=True,
)
assert_test(not segment_proxy.accepted, "V3 external proxy rejects remaining segment-level AI footprint")
assert_test("segment_ai_fraction_high" in segment_proxy.reasons, "V3 proxy records segment AI fraction blocker")
contract_prefers_footprint = build_scan_contract(
    {
        **report_for(broad_source, ai=20.0),
        "authorship_window_profile": {
            "fraction_ai": 0.0,
            "fraction_ai_assisted": 0.05,
            "fraction_human": 0.95,
            "windows": [],
        },
        "ai_footprint_profile": {
            "schema_version": "ai_footprint_profile.v2",
            "fraction_ai": 0.1,
            "fraction_ai_assisted": 0.72,
            "fraction_human": 0.18,
            "risky_window_density": 0.31,
            "max_risky_window_words": 88,
            "high_confidence_risky_window_count": 2,
            "risky_window_count": 3,
            "confidence": "high",
        },
    },
    broad_source,
)
assert_test(
    contract_prefers_footprint.footprint_fraction_ai_assisted == 0.72
    and contract_prefers_footprint.high_confidence_risky_window_count == 2,
    "V3 ScanContract prefers AI footprint profile over legacy authorship profile",
)
localized_contract = build_scan_contract(
    {
        **report_for(broad_source, mode="unknown", ai=40.0),
        "ai_footprint_profile": {
            "schema_version": "ai_footprint_profile.v2",
            "fraction_ai": 0.0,
            "fraction_ai_assisted": 0.22,
            "fraction_human": 0.78,
            "risky_window_density": 0.12,
            "max_risky_window_words": 70,
            "high_confidence_risky_window_count": 0,
            "risky_window_count": 1,
            "confidence": "medium",
        },
    },
    broad_source,
)
localized_plan = build_strategy_plan(route_from_scan_contract(localized_contract), localized_contract)
assert_test(
    [step.strategy_id for step in localized_plan.steps][0] == "authorship_window_repair",
    "V3 strategy plan routes localized footprint to window repair first",
)
target_contract = build_scan_contract(
    {
        **report_for(broad_source, mode="unknown", ai=40.0),
        "rewrite_target_profile": {
            "schema_version": "rewrite_target_profile.v1",
            "target_scope_policy": "target_profile_driven",
            "operation_mix": {"protected_section_rewrite": 2},
            "driver_summary": {"predictability_score": 2},
            "targets": [{"target_id": "rt001", "risk_level": "medium", "recommended_operation": "protected_section_rewrite"}],
        },
    },
    broad_source,
)
target_plan = build_strategy_plan(route_from_scan_contract(target_contract), target_contract)
assert_test(
    target_contract.rewrite_targets
    and target_plan.steps[0].strategy_id == "protected_section_rewrite",
    "V3 strategy plan is driven by rewrite target operation mix",
)
segment_loop = decide_next_action(
    [{
        "trace": {
            "validation": {"passed": True, "failures": []},
            "compression": {"status": "in_band"},
            "compression_accepted": True,
            "semantic_safe": True,
            "external_proxy": {"reasons": ["segment_ai_fraction_high", "segment_human_fraction_low"]},
            "ownership_gate": {"active": True, "passed": False, "ownership_score": 0.0, "ownership_change_count": 0},
        }
    }],
    has_positive_boundaries=False,
    tried_actions=set(),
)
assert_test(
    segment_loop.action == CandidateAction.CLAIM_OWNERSHIP_REPAIR,
    "V3 loop routes segment footprint failures with no owned trace to claim ownership repair first",
)
latest_failed_ownership_loop = decide_next_action(
    [{
        "trace": {
            "validation": {"passed": False, "failures": []},
            "compression": {"status": "below_floor"},
            "compression_accepted": False,
            "semantic_safe": True,
            "detector_movement": False,
            "target_gate_passed": False,
            "candidate_outcome": "generation_failed_empty_output",
            "target_execution_available": True,
            "scanner_controlled_executor_available": True,
            "external_proxy": {"reasons": ["segment_ai_or_assisted_fraction_high", "segment_human_fraction_low"]},
            "ownership_gate": {"active": True, "passed": False, "ownership_score": 0.0, "ownership_change_count": 0},
        }
    }],
    has_positive_boundaries=False,
    tried_actions=set(),
)
assert_test(
    latest_failed_ownership_loop.action == CandidateAction.CLAIM_OWNERSHIP_REPAIR
    and latest_failed_ownership_loop.reason == "claim_ownership_repair_after_latest_ownership_failure",
    "V3 loop prioritizes ownership repair before planned broad recovery when the latest candidate is empty but ownership-missing",
)
segment_authorship_loop = decide_next_action(
    [{
        "trace": {
            "validation": {"passed": True, "failures": []},
            "compression": {"status": "in_band"},
            "compression_accepted": True,
            "semantic_safe": True,
            "external_proxy": {"reasons": ["segment_ai_fraction_high", "segment_human_fraction_low"]},
            "ownership_gate": {"active": True, "passed": True, "ownership_score": 6.0, "ownership_change_count": 1},
        }
    }],
    has_positive_boundaries=False,
    tried_actions={CandidateAction.CLAIM_OWNERSHIP_REPAIR},
)
assert_test(
    segment_authorship_loop.action == CandidateAction.REPAIR_AUTHORSHIP_WINDOWS,
    "V3 loop falls through to authorship window repair after ownership repair has already been tried",
)
no_movement_decision = decide_next_action(
    [{
        "trace": {
            "validation": {"failures": ["document_unit_count_changed"]},
            "compression_accepted": True,
            "semantic_safe": True,
            "detector_movement": False,
            "external_proxy": {"reasons": ["validation_failed"]},
            "text_integrity": {"passed": True},
        }
    }],
    has_positive_boundaries=False,
    tried_actions=set(),
)
assert_test(
    no_movement_decision.action == CandidateAction.PLAIN_REASONING
    and no_movement_decision.reason == "plain_reasoning_strategy_after_no_detector_movement",
    "V3 loop switches strategy instead of blind structure repair when detector footprint does not move",
)
no_movement_stop_decision = decide_next_action(
    [{
        "trace": {
            "validation": {"failures": ["document_unit_count_changed"]},
            "compression_accepted": True,
            "semantic_safe": True,
            "detector_movement": False,
            "external_proxy": {"reasons": ["validation_failed"]},
            "text_integrity": {"passed": True},
        }
    }],
    has_positive_boundaries=False,
    tried_actions={CandidateAction.PLAIN_REASONING},
)
assert_test(
    no_movement_stop_decision.action == CandidateAction.RETURN_BEST_FOR_REVIEW
    and no_movement_stop_decision.reason == "stop_after_no_detector_movement",
    "V3 loop stops after bounded no-movement strategy switch",
)

structure_trace = {
    "validation": {"passed": False, "failures": ["document_unit_count_changed"]},
    "compression": {"status": "in_band"},
    "compression_accepted": True,
    "semantic_safe": True,
    "external_proxy": {"reasons": ["validation_failed"]},
    "candidate_ai": 40.0,
}
structure_issues = issues_from_trace(structure_trace)
assert_test(CandidateIssue.STRUCTURE_CHANGED in structure_issues, "V3 loop classifies structure changes as typed issues")
loop_decision = decide_next_action(
    [{"text": "candidate", "strict_selected": False, "external_selected": False, "trace": structure_trace}],
    has_positive_boundaries=True,
    tried_actions=set(),
)
assert_test(loop_decision.action == CandidateAction.REPAIR_STRUCTURE, "V3 loop repairs structure before adding another rewrite layer")
structure_repair_prompt = build_structure_repair_prompt(
    source_text=shape_source,
    candidate_text=shape_split_candidate,
    validation={"failures": ["document_unit_count_changed"]},
    expected_unit_count=2,
)
assert_test(
    "source_structure_contract" in structure_repair_prompt
    and "heading_like_lines" in structure_repair_prompt,
    "V3 structure repair prompt carries source shape contract",
)

anchor_trace = {
    "validation": {
        "passed": False,
        "failures": ["protected_anchor_missing"],
        "missing_anchors": [{"text": "Source Anchor", "kind": "citation", "severity": "hard_exact"}],
    },
    "compression": {"status": "below_floor"},
    "compression_accepted": False,
    "semantic_safe": False,
    "semantic_similarity": 0.79,
    "external_proxy": {"reasons": ["validation_failed", "compression_rejected", "semantic_drift"]},
    "candidate_ai": 40.0,
}
anchor_decision = decide_next_action(
    [{"text": "candidate", "strict_selected": False, "external_selected": False, "trace": anchor_trace}],
    has_positive_boundaries=True,
    tried_actions=set(),
)
assert_test(anchor_decision.action == CandidateAction.REPAIR_CONTRACT, "V3 repairs generic contract failures before style layers")
contract_prompt = build_contract_repair_prompt(
    original_text=broad_source,
    failed_candidate=broad_candidate,
    strategy_family="document_rhythm",
    candidate_trace=anchor_trace,
    compression_policy=compression_policy_for_family("document_rhythm", word_count(broad_source)),
)
assert_test(
    "failed_invariants" in contract_prompt and "Source Anchor" in contract_prompt and "source_structure_contract" in contract_prompt,
    "V3 contract repair prompt carries failed invariants, missing anchors, and source structure",
)
assert_test(
    "must_include_exact_anchors" in contract_prompt,
    "V3 contract repair prompt exposes exact anchors as mandatory copy strings",
)

proxy_trace = {
    "validation": {"passed": True, "failures": []},
    "compression": {"status": "in_band"},
    "compression_accepted": True,
    "semantic_safe": True,
    "external_proxy": {"reasons": ["insufficient_topk_drop"]},
    "scanner_controlled_executor_available": True,
    "candidate_ai": 45.0,
}
topk_issues = issues_from_trace({
    **proxy_trace,
    "topk_effect_failures": ["no_effect_span_patch", "self_report_mismatch", "insufficient_span_movement"],
})
assert_test(
    CandidateIssue.TOPK_CANDIDATE_REJECTED in topk_issues
    and CandidateIssue.NO_EFFECT_SPAN_PATCH in topk_issues
    and CandidateIssue.SELF_REPORT_MISMATCH in topk_issues
    and CandidateIssue.INSUFFICIENT_SPAN_MOVEMENT in topk_issues,
    "V3 loop promotes Top-k validator failures into typed candidate issues",
)
proxy_decision = decide_next_action(
    [{"text": "candidate", "strict_selected": False, "external_selected": False, "trace": proxy_trace}],
    has_positive_boundaries=True,
    tried_actions=set(),
)
assert_test(
    proxy_decision.action == CandidateAction.SCANNER_CONTROLLED_SPAN_REPAIR,
    "V3 loop routes insufficient Top-k movement to scanner-controlled span repair first",
)
contrast_decision = decide_next_action(
    [{"text": "candidate", "strict_selected": False, "external_selected": False, "trace": proxy_trace}],
    has_positive_boundaries=True,
    tried_actions={CandidateAction.SCANNER_CONTROLLED_SPAN_REPAIR, CandidateAction.ADAPT_BOUNDARY},
)
assert_test(contrast_decision.action == CandidateAction.CONTRAST_BOUNDARY, "V3 loop runs contrast boundary after boundary adaptation misses")
plain_decision = decide_next_action(
    [{"text": "candidate", "strict_selected": False, "external_selected": False, "trace": proxy_trace}],
    has_positive_boundaries=True,
    tried_actions={CandidateAction.SCANNER_CONTROLLED_SPAN_REPAIR, CandidateAction.ADAPT_BOUNDARY, CandidateAction.CONTRAST_BOUNDARY},
)
assert_test(plain_decision.action == CandidateAction.PLAIN_REASONING, "V3 loop runs plain reasoning after contrast boundary misses")

contrast_prompt = build_contrast_boundary_prompt(
    original_text=broad_source,
    failed_candidate=broad_candidate,
    family="document_rhythm",
    compression_policy=compression_policy_for_family("document_rhythm", word_count(broad_source)),
    style_examples={"positive": [{"external_ai_percent": 18, "text": "Plain boundary."}], "negative": []},
)
assert_test(
    "current_failed_rewrite" in contrast_prompt
    and "positive_boundary_samples" in contrast_prompt
    and "source_structure_contract" in contrast_prompt,
    "V3 contrast boundary prompt includes failed and positive examples plus source structure",
)
assert_test(
    extract_contrast_boundary_output('{"rewritten_document":"A clean rewrite.","notes":["ok"]}') == "A clean rewrite.",
    "V3 contrast boundary extracts JSON rewritten_document",
)
plain_prompt = build_plain_reasoning_broad_prose_prompt(
    original_text=broad_source,
    failed_candidates=[broad_candidate],
    compression_policy=compression_policy_for_family("document_rhythm", word_count(broad_source)),
    style_examples={"positive": [], "negative": []},
)
assert_test(
    "plain-reasoning style" in plain_prompt
    and "formal generated-survey texture" in plain_prompt
    and "source_structure_contract" in plain_prompt,
    "V3 plain reasoning prompt targets broad formal survey texture and source structure",
)
bloated_plain_prompt = build_plain_reasoning_broad_prose_prompt(
    original_text=broad_source * 20,
    failed_candidates=[broad_candidate * 12, broad_candidate * 12],
    compression_policy=compression_policy_for_family("document_rhythm", word_count(broad_source)),
    style_examples={
        "positive": [{"external_ai_percent": 18, "text": "positive boundary " * 500}],
        "negative": [{"external_ai_percent": 89, "text": "negative boundary " * 500}],
    },
    rewrite_target_profile={
        "schema_version": "rewrite_target_profile.v1",
        "document_shape": "broad",
        "targets": [
            {
                "target_id": f"rt{i:03d}",
                "unit_id": f"p{i:03d}",
                "source_text": "target source " * 1000,
                "source_excerpt": "target excerpt " * 1000,
                "dominant_drivers": [{"key": "predictability_score", "score": 0.8}],
            }
            for i in range(12)
        ],
    },
    central_judgment_plan={
        "strategy_id": "central_contextual_judgment_v1",
        "generation_objective": {"primary_drivers": ["route_variation"] * 20},
        "constraints": {"source_word_count": 518, "preserve_content_units": [{"meaning": "unit " * 200}] * 50},
    },
)
assert_test(
    len(bloated_plain_prompt) < 13000
    and "source_document" not in bloated_plain_prompt
    and "preserve_content_units" not in bloated_plain_prompt
    and "target source " * 30 not in bloated_plain_prompt
    and "positive boundary " * 30 not in bloated_plain_prompt,
    "V3 plain reasoning prompt is compact and does not dump full scanner contracts or failed candidates",
)

with tempfile.TemporaryDirectory() as tmpdir:
    result = run_rewrite_pipeline_v3(
        detect_json=report_for(broad_source, ai=70.0),
        output_dir=tmpdir,
        replay_candidate_records=[
            {
                "text": broad_candidate,
                "ai": 29.0,
                "wq": 50.0,
                "topk": 84.0,
            },
            {
                "text": broad_candidate,
                "ai": 38.0,
                "wq": 57.0,
                "topk": 70.0,
            },
        ],
        required_ai_drop=60.0,
    )
    selected = result["result"].summary["selected_candidate"]
    assert_test(
        selected["generation_mode"] == "replay_recovery_2",
        "V3 fallback selection prefers stronger proxy profile over lowest internal AI",
    )

def candidate_item(mode: str, *, ai_delta: float, topk_delta: float, wq_delta: float, ratio: float = 0.72) -> dict:
    return {
        "text": mode,
        "strict_selected": False,
        "external_selected": False,
        "trace": {
            "generation_mode": mode,
            "validation": {"passed": True, "failures": [], "source_units": 2, "candidate_units": 2},
            "compression": {"status": "in_band", "ratio": ratio},
            "compression_accepted": True,
            "semantic_safe": True,
            "external_proxy": {
                "reasons": ["writing_quality_collapse"],
                "metrics": {
                    "ai_delta": ai_delta,
                    "topk_delta": topk_delta,
                    "wq_delta": wq_delta,
                },
            },
            "candidate_ai": 70.0 - ai_delta,
        },
    }

calibrated_good = candidate_item("known_external_good", ai_delta=30.0, topk_delta=13.0, wq_delta=17.0, ratio=0.73)
calibrated_bad = candidate_item("known_external_bad", ai_delta=22.0, topk_delta=2.0, wq_delta=7.0, ratio=0.80)
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "calibration.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "family": "document_rhythm",
            "external_label": {"ai_percent": 18},
            "trace": calibrated_good["trace"],
            "text": "good",
        }) + "\n")
        handle.write(json.dumps({
            "family": "document_rhythm",
            "external_label": {"ai_percent": 89},
            "trace": calibrated_bad["trace"],
            "text": "bad",
        }) + "\n")
    old_store = os.environ.get("DRAFTPROOF_REWRITE_V3_CALIBRATION_STORE")
    os.environ["DRAFTPROOF_REWRITE_V3_CALIBRATION_STORE"] = path
    try:
        from rewrite_v3.calibration_store import load_calibration_records

        load_calibration_records.cache_clear()
        selected_idx, scores = select_portfolio_candidate([calibrated_bad, calibrated_good], family="document_rhythm")
    finally:
        load_calibration_records.cache_clear()
        if old_store is None:
            os.environ.pop("DRAFTPROOF_REWRITE_V3_CALIBRATION_STORE", None)
        else:
            os.environ["DRAFTPROOF_REWRITE_V3_CALIBRATION_STORE"] = old_store
    assert_test(selected_idx == 1 and scores[1]["score"] > scores[0]["score"], "V3 portfolio selector uses external calibration records")

review_material_scanner_candidate = {
    "text": "scanner",
    "strict_selected": False,
    "external_selected": False,
    "trace": {
        "generation_mode": "scanner_controlled_executor",
        "candidate_outcome": "valid_no_detector_movement",
        "validation": {"passed": True, "failures": [], "source_units": 8, "candidate_units": 8},
        "compression": {"status": "in_band", "ratio": 0.958},
        "compression_accepted": True,
        "semantic_safe": True,
        "target_gate_passed": False,
        "footprint_delta": {"risk_drop": 0.581},
        "target_movement": {"risk_drop": 0.261},
        "external_proxy": {
            "reasons": [
                "insufficient_topk_drop",
                "segment_ai_or_assisted_fraction_high",
                "segment_human_fraction_low",
            ],
            "metrics": {
                "ai_delta": 14.66,
                "topk_delta": 1.84,
                "wq_delta": 1.58,
                "segment_authorship_gate": {
                    "fraction_ai": 0.0,
                    "fraction_ai_assisted": 1.0,
                    "fraction_human": 0.0,
                    "max_ai_window_words": 0.0,
                },
            },
        },
    },
}
review_smaller_detector_candidate = {
    "text": "paragraph",
    "strict_selected": False,
    "external_selected": False,
    "trace": {
        "generation_mode": "paragraph_preserving_broad_reconstruction",
        "candidate_outcome": "valid_detector_improved",
        "validation": {"passed": True, "failures": [], "source_units": 8, "candidate_units": 8},
        "compression": {"status": "in_band", "ratio": 0.981},
        "compression_accepted": True,
        "semantic_safe": True,
        "target_gate_passed": False,
        "footprint_delta": {"risk_drop": 4.546},
        "target_movement": {"risk_drop": 0.696},
        "external_proxy": {
            "reasons": [
                "insufficient_topk_drop",
                "segment_ai_or_assisted_fraction_high",
                "segment_human_fraction_low",
            ],
            "metrics": {
                "ai_delta": 4.57,
                "topk_delta": -0.49,
                "wq_delta": -0.68,
                "segment_authorship_gate": {
                    "fraction_ai": 0.0,
                    "fraction_ai_assisted": 1.0,
                    "fraction_human": 0.0,
                    "max_ai_window_words": 0.0,
                },
            },
        },
    },
}
selected_idx, scores = select_portfolio_candidate(
    [review_material_scanner_candidate, review_smaller_detector_candidate],
    family="clean_texture_boundary",
)
assert_test(
    selected_idx == 0
    and "material_ai_drop_without_detector_gate" in scores[0]["reasons"]
    and scores[0]["score"] > scores[1]["score"],
    "V3 portfolio review selection preserves valid material AI improvement even when target gate remains partial",
)
ownership_blocked_candidate = {
    **review_material_scanner_candidate,
    "trace": {
        **review_material_scanner_candidate["trace"],
        "ownership_gate": {"active": True, "passed": False, "ownership_score": 0.0, "ownership_change_count": 0},
    },
}
ownership_repaired_candidate = {
    **review_smaller_detector_candidate,
    "trace": {
        **review_smaller_detector_candidate["trace"],
        "ownership_gate": {"active": True, "passed": True, "ownership_score": 7.5, "ownership_change_count": 2},
    },
}
ownership_selected_idx, ownership_scores = select_portfolio_candidate(
    [ownership_blocked_candidate, ownership_repaired_candidate],
    family="clean_texture_boundary",
)
assert_test(
    ownership_selected_idx == 1
    and "ownership_gate_failed" in ownership_scores[0]["reasons"]
    and "ownership_gate_passed" in ownership_scores[1]["reasons"],
    "V3 portfolio selector treats ownership as an executable gate when human fraction remains low",
)
fusion_decision = decide_next_action(
    [ownership_blocked_candidate, ownership_repaired_candidate],
    has_positive_boundaries=False,
    tried_actions={CandidateAction.CLAIM_OWNERSHIP_REPAIR},
)
assert_test(
    fusion_decision.action == CandidateAction.FUSE_DETECTOR_AND_OWNERSHIP
    and fusion_decision.reason == "fuse_detector_movement_with_window_ownership",
    "V3 loop fuses split detector movement and ownership success before review fallback",
)
with tempfile.TemporaryDirectory() as tmpdir:
    empty_review_result = run_rewrite_pipeline_v3(
        detect_json=report_for(broad_source, ai=70.0),
        output_dir=tmpdir,
        replay_candidate_records=[{"text": "", "ai": 70.0, "topk": 90.0, "wq": 50.0}],
        full_rewrite_allowed=False,
        max_runtime_seconds=60,
    )
    empty_review_summary = empty_review_result["result"].summary
    assert_test(
        empty_review_summary["outcome"] == "mitigation_failed_no_safe_candidate"
        and empty_review_summary["rewrite_goal_status"]["reason"] == "rewrite_v3_no_safe_candidate_generated"
        and empty_review_summary["final_text"] == broad_source,
        "V3 does not report empty generation failures as reviewable rewrite candidates",
    )
with tempfile.TemporaryDirectory() as tmpdir:
    empty_no_report_result = run_rewrite_pipeline_v3(
        detect_json=report_for(broad_source, ai=70.0),
        output_dir=tmpdir,
        replay_candidate_records=[{"text": ""}],
        full_rewrite_allowed=False,
        max_runtime_seconds=60,
    )
    empty_no_report_trace = empty_no_report_result["result"].summary["candidate_trace"][0]
    assert_test(
        empty_no_report_trace["candidate_ai"] is None
        and empty_no_report_trace["scan_freshness"]["empty_candidate_no_report_fallback"] is True
        and empty_no_report_trace["scan_freshness"]["input_text_present"] is False
        and empty_no_report_trace["candidate_outcome"] == "generation_failed_empty_output",
        "V3 empty generated candidates do not borrow original scan metrics",
    )

seven_unit_candidate = "\n\n".join([
    "Alpha paragraph.",
    "Beta paragraph.",
    "Gamma paragraph.",
    "Delta paragraph.",
    "Epsilon paragraph.",
    "Zeta paragraph.",
    "Eta paragraph.",
])
eight_unit_source = "\n\n".join([
    "Alpha source paragraph with enough text to count.",
    "Beta source paragraph with enough text to count.",
    "Gamma source paragraph with enough text to count.",
    "Delta source paragraph with enough text to count.",
    "Epsilon source paragraph with enough text to count.",
    "Zeta source paragraph with enough text to count.",
    "Eta source paragraph with enough text to count.",
    "Theta source paragraph with enough text to count.",
])
with tempfile.TemporaryDirectory() as tmpdir:
    result = run_rewrite_pipeline_v3(
        detect_json=report_for(eight_unit_source, ai=70.0),
        output_dir=tmpdir,
        replay_candidate_records=[{
            "text": seven_unit_candidate,
            "ai": 20.0,
            "wq": 62.0,
            "topk": 40.0,
        }],
        required_ai_drop=20.0,
    )
    assert_test(
        "document_unit_count_changed" in result["result"].summary["candidate_trace"][0]["validation"]["failures"],
        "V3 document rhythm rejects paragraph count collapse",
    )

replay_report_with_input = report_for(broad_candidate, ai=42.0)
replay_report_with_input["input_text"] = broad_candidate
replay_report_with_input["predictability_cache"] = {
    "enabled": True,
    "hits": 2,
    "misses": 1,
}
with tempfile.TemporaryDirectory() as tmpdir:
    result = run_rewrite_pipeline_v3(
        detect_json=report_for(broad_source, ai=70.0),
        output_dir=tmpdir,
        replay_candidate_records=[{
            "text": broad_candidate,
            "report": replay_report_with_input,
        }],
        required_ai_drop=20.0,
    )
    freshness = result["result"].summary["candidate_trace"][0]["scan_freshness"]
    assert_test(
        freshness["candidate_text_hash"] == freshness["report_input_text_hash"]
        and freshness["input_text_matches_candidate"]
        and freshness["scan_reused_supplied_report"]
        and freshness["predictability_cache"]["hits"] == 2,
        "V3 candidate trace exposes scan freshness hashes and predictability cache metadata",
    )

boundary_prompt = build_boundary_adapter_prompt(
    original_text=eight_unit_source,
    failed_candidates=[seven_unit_candidate],
    strategy_family="document_rhythm",
    proxy_feedback=[],
    compression_policy=compression_policy_for_family("document_rhythm", word_count(eight_unit_source)),
    style_examples={"positive": [], "negative": []},
)
assert_test(
    "same paragraph count as the source" in boundary_prompt
    and "source_structure_contract" in boundary_prompt,
    "V3 boundary adapter prompt requires source paragraph count and structure preservation",
)

with tempfile.TemporaryDirectory() as tmpdir:
    result = run_rewrite_pipeline_v3(
        detect_json=report_for(broad_source, ai=70.0),
        output_dir=tmpdir,
        replay_candidate_records=[{
            "text": too_short,
            "ai": 10.0,
            "wq": 62.0,
            "topk": 40.0,
        }],
        required_ai_drop=20.0,
    )
    assert_test(
        result["status"] == "rewrite_candidate_generated_needs_external_review",
        "V3 still returns a candidate when all replay candidates miss gates",
    )
    assert_test(
        result["result"].summary["selected_candidate"]["candidate_outcome"] == "invalid_detector_improved",
        "V3 labels detector-improved but invalid candidates distinctly",
    )

with tempfile.TemporaryDirectory() as tmpdir:
    result = run_rewrite_pipeline_v3(
        detect_json=report_for(broad_source, ai=70.0),
        output_dir=tmpdir,
        replay_candidate_records=[
            {
                "text": broad_candidate,
                "ai": 58.0,
                "wq": 52.0,
                "topk": 88.0,
            },
            {
                "text": broad_candidate,
                "ai": 50.0,
                "wq": 60.0,
                "topk": 70.0,
            },
        ],
        required_ai_drop=20.0,
    )
    summary = result["result"].summary
    assert_test(result["status"] == "rewrite_candidate_generated_needs_external_review", "V3 recovery replay returns later external candidate for review")
    assert_test(summary["selected_candidate"]["generation_mode"] == "replay_recovery_2", "V3 does not stop at failed first candidate")

integrity_bad = v3_pipeline._text_integrity(
    "This source has normal spacing across several words.",
    "Thissourcehasnormalspacingacrossseveralwordswithoutbreaks.",
)
assert_test(
    not integrity_bad["passed"] and "merged_word_run" in integrity_bad["failures"],
    "V3 text-integrity guard flags merged-word corruption",
)
integrity_joined_token = v3_pipeline._text_integrity(
    "Introduction Inclusive learning still needs structure.",
    "IntroductionInclusive learning still needs structure.",
)
assert_test(
    not integrity_joined_token["passed"] and "merged_source_token" in integrity_joined_token["failures"],
    "V3 text-integrity guard still flags long source-token boundary joins",
)
integrity_single_compound = v3_pipeline._text_integrity(
    "Students learn in work places and online communities.",
    "Students learn in workplaces and online communities.",
)
assert_test(
    integrity_single_compound["passed"]
    and "merged_source_token" in integrity_single_compound["warnings"],
    "V3 text-integrity guard treats a single normal compound-like merge as a repairable warning, not hard corruption",
)

with tempfile.TemporaryDirectory() as tmpdir:
    no_movement = run_rewrite_pipeline_v3(
        detect_json=report_for(broad_source, ai=70.0),
        output_dir=tmpdir,
        replay_candidate_records=[{
            "text": broad_candidate,
            "ai": 70.0,
            "wq": 60.0,
            "topk": 90.0,
        }],
        required_ai_drop=5.0,
    )
    no_move_summary = no_movement["result"].summary
    assert_test(
        no_movement["status"] == "rewrite_candidate_generated_needs_external_review"
        and no_move_summary["selected_candidate"]["candidate_outcome"] == "valid_no_detector_movement",
        "V3 valid candidates without detector movement are not treated as mitigation success",
    )
    assert_test(
        no_move_summary["public_candidate_warning"] == "best_candidate_has_no_detector_footprint_movement",
        "V3 public summary explains no-movement candidate warning",
    )

target_gate_fixture = {
    "schema_version": "rewrite_target_profile.v1",
    "target_scope_policy": "target_profile_driven",
    "operation_mix": {"grounded_author_reasoning_rewrite": 1},
    "driver_summary": {"predictability_score": 1},
    "targets": [{
        "target_id": "rt001",
        "risk_level": "medium",
        "dominant_drivers": [{"key": "predictability_score", "score": 0.7}],
        "recommended_operation": "grounded_author_reasoning_rewrite",
    }],
}
original_scanner_controlled_generator = v3_pipeline._generate_scanner_controlled_candidate
try:
    def fake_scanner_controlled_generator(**kwargs):
        return broad_candidate, {
            "target_execution_attempted": True,
            "scanner_controlled": True,
            "scanner_controlled_config": {"max_rounds": 2, "groups_per_round": 4, "variants_per_group": 3},
            "scanner_controlled_rounds": [],
            "target_groups": [],
            "target_replacements": [],
            "target_apply_status": [],
        }

    v3_pipeline._generate_scanner_controlled_candidate = fake_scanner_controlled_generator
    with tempfile.TemporaryDirectory() as tmpdir:
        scanner_first = run_rewrite_pipeline_v3(
            detect_json={**report_for(broad_source, ai=70.0), "rewrite_target_profile": target_gate_fixture},
            output_dir=tmpdir,
            required_ai_drop=20.0,
            max_runtime_seconds=0,
        )
        scanner_first_summary = scanner_first["result"].summary
        assert_test(
            scanner_first_summary["strategy_trace"][0]["scanner_controlled_executor_first"]
            and scanner_first_summary["candidate_trace"][0]["generation_mode"] == "scanner_controlled_executor"
            and scanner_first_summary["candidate_trace"][0]["target_execution_trace"]["scanner_controlled"],
            "V3 runs scanner-controlled executor before weaker target/full-document layers",
        )
finally:
    v3_pipeline._generate_scanner_controlled_candidate = original_scanner_controlled_generator

paragraph_problem_inventory = {
    "schema_version": "problem_inventory.v1",
    "problem_groups": [{
        "group_id": "pg_paragraph",
        "scope_level": "paragraph",
        "unit_ids": ["p001"],
        "target_ids": ["rt001"],
        "problem_shape": "broad_assisted_footprint",
        "anchor_pressure": 0.1,
        "semantic_edit_cost": 0.2,
        "allowed_operations": ["paragraph_preserving_broad_reconstruction", "chunk_reconstruction"],
        "blocked_operations": ["unit_preserving_prune_bridge"],
    }],
}
paragraph_strategy_profile = {
    "schema_version": "rewrite_target_profile.v1",
    "target_scope_policy": "problem_inventory_driven",
    "operation_mix": {"paragraph_preserving_broad_reconstruction": 1},
    "targets": [{
        "target_id": "rt001",
        "unit_id": "p001",
        "paragraph_id": "p001",
        "sentence_ids": ["s001", "s002"],
        "scope_level": "paragraph",
        "risk_level": "medium",
        "source_text": broad_source.split("\n\n")[0],
        "recommended_operation": "paragraph_preserving_broad_reconstruction",
        "operation_candidates": ["paragraph_preserving_broad_reconstruction", "chunk_reconstruction"],
        "protected_anchors": [],
        "soft_guidance_anchors": [{"text": "old classroom model", "kind": "domain_term", "blocking": False}],
        "word_count_guide": {"source_words": len(broad_source.split("\n\n")[0].split()), "preferred_words": len(broad_source.split("\n\n")[0].split())},
    }],
}
paragraph_contract = build_scan_contract(
    {
        **report_for(broad_source, ai=70.0),
        "rewrite_target_profile": paragraph_strategy_profile,
        "problem_inventory": paragraph_problem_inventory,
        "rewrite_edit_briefs": [
            {
                "brief_id": "reb001",
                "sentence_id": "s001",
                "paragraph_id": "p001",
                "target_sentence": "Education is changing quickly because students now meet information in many places.",
                "signals": {
                    "predictable_token_spans": [
                        "Education is changing quickly",
                        "students now meet information",
                    ]
                },
            },
            {
                "brief_id": "reb002",
                "sentence_id": "s002",
                "paragraph_id": "p001",
                "target_sentence": "Schools still matter, but the old classroom model no longer explains how learning happens.",
                "signals": {
                    "predictable_token_spans": [
                        "Schools still matter",
                        "old classroom model",
                    ]
                },
            },
        ],
    },
    broad_source,
)
paragraph_groups = group_rewrite_targets(
    original_text=broad_source,
    rewrite_target_profile=paragraph_contract.rewrite_target_profile,
    max_groups=4,
)
paragraph_context = paragraph_portfolio_context(
    target_groups=paragraph_groups,
    scan_contract=paragraph_contract,
    content_mode=paragraph_contract.content_mode,
    strategy_family="paragraph_preserving_broad_reconstruction",
)
planner_template = build_paragraph_portfolio_planner_prompt(paragraph_context)
assert_test(
    planner_template.template_id == "paragraph_portfolio.v1"
    and "old classroom model" in planner_template.prompt
    and "rewrite_target_profile.targets" in planner_template.scanner_context_used,
    "V3 paragraph portfolio planner prompt is built from dynamic scanner context",
)
parsed_plan = parse_paragraph_portfolio_plan(json.dumps({
    "paragraph_plans": [{
        "group_id": "tg001",
        "paragraph_role": "opening_frame",
        "risk_drivers": ["predictability_score"],
        "hard_anchors": [],
        "soft_anchors": ["old classroom model"],
        "repeated_patterns": ["broad opening followed by balanced explanation"],
        "recommended_operator": "BREAK_SURVEY_TEMPLATE",
        "rewrite_aggression_limit": "medium",
    }]
}))
assert_test(
    validate_paragraph_portfolio_plan(parsed_plan, paragraph_groups)["passed"],
    "V3 paragraph portfolio planner output validates required paragraph groups",
)
fallback_plan = fallback_paragraph_portfolio_plan(paragraph_groups)
reconstruction_template = build_paragraph_portfolio_reconstruction_prompt(paragraph_context, fallback_plan)
ownership_template = build_paragraph_portfolio_ownership_prompt(paragraph_context, fallback_plan)
topk_template = build_paragraph_portfolio_topk_prompt(
    context=paragraph_context,
    planner_output=fallback_plan,
    replacements=[{"group_id": "tg001", "replacement_text": broad_candidate.split("\n\n")[0]}],
)
assert_test(
    "word_count_guide" in reconstruction_template.prompt
    and "execution_contract" in reconstruction_template.prompt
    and "ownership_contract" in reconstruction_template.prompt
    and "source_structure_contract" in reconstruction_template.prompt
    and "required_protected_anchors" in reconstruction_template.prompt
    and "out_of_scope_protected_anchors" in reconstruction_template.prompt
    and "source_structure_contract" in topk_template.prompt
    and "current_replacements" in topk_template.prompt
    and "topk_repair_contract" in topk_template.prompt
    and "ownership_contract" in topk_template.prompt
    and "predictable_spans" in topk_template.prompt
    and "locality_limits" in topk_template.prompt
    and "changed_word_estimate" in topk_template.prompt
    and "Education is changing quickly" in topk_template.prompt
    and "Return JSON only" in topk_template.prompt,
    "V3 paragraph portfolio reconstruction and Top-k prompts expose scanner-targeted ownership, scoped anchors, and structure contracts",
)
assert_test(
    '"prompt_stage":"claim_ownership_repair"' in ownership_template.prompt
    and "ownership_changes" in ownership_template.prompt
    and "ownership_elements_supported" in ownership_template.prompt
    and "point of view" in ownership_template.prompt
    and "source-supported" in ownership_template.prompt,
    "V3 paragraph ownership prompt asks for source-grounded ownership changes, not broad smoothing",
)
shape_group = TargetGroup(
    group_id="tg_shape",
    unit_id="p_shape",
    operation="paragraph_preserving_broad_reconstruction",
    start_index=0,
    end_index=len(shape_source),
    source_text=shape_source,
    before_context="",
    after_context="",
    targets=(),
    word_count_guide={
        "source_words": word_count(shape_source),
        "preferred_words": word_count(shape_source),
    },
)
valid_shape_replacements, valid_shape_status = validate_replacement_structure(
    target_groups=[shape_group],
    replacements=[{"group_id": "tg_shape", "replacement_text": shape_candidate}],
)
invalid_shape_replacements, invalid_shape_status = validate_replacement_structure(
    target_groups=[shape_group],
    replacements=[{"group_id": "tg_shape", "replacement_text": shape_split_candidate}],
)
assert_test(
    valid_shape_replacements
    and valid_shape_status[0]["passed"]
    and not invalid_shape_replacements
    and "block_count_changed" in invalid_shape_status[0]["failures"],
    "V3 paragraph portfolio executor filters candidates that change source block structure",
)
topk_noop_source = (
    "Because of this, the education system needs to evolve. "
    "Schools should still teach core knowledge, but they must also teach digital literacy, critical thinking, communication, creativity, and ethical use of technology. "
    "Assessment should include not only final answers, but also the learning process: drafts, reflection, discussion, feedback, and improvement."
)
topk_current = (
    "The education system has to change. "
    "Schools should still cover core subjects, but they also need to teach digital literacy, critical thinking, communication, creativity, and ethical technology use. "
    "Assessment should look at the learning process, including drafts, reflection, discussion, feedback, and improvement, not just final answers."
)
topk_effective = (
    "The education system has to change. "
    "Schools should still cover core subjects, but digital literacy now has to sit beside critical thinking, communication, creativity, and ethical technology use. "
    "Assessment should look at the learning process, including drafts, reflection, discussion, feedback, and improvement, not just final answers."
)
topk_noop_group = TargetGroup(
    group_id="tg_topk_noop",
    unit_id="p007",
    operation="paragraph_preserving_broad_reconstruction",
    start_index=0,
    end_index=len(topk_noop_source),
    source_text=topk_noop_source,
    before_context="",
    after_context="",
    targets=(),
    word_count_guide={
        "source_words": word_count(topk_noop_source),
        "preferred_words": word_count(topk_noop_source),
    },
)
topk_noop_briefs = [{
    "paragraph_id": "p007",
    "target_sentence": topk_current,
    "predictable_token_spans": [
        "literacy,",
        "thinking, communication, creativity, and",
    ],
}]
topk_noop_rows, topk_noop_status = validate_topk_replacement_effect(
    target_groups=[topk_noop_group],
    current_replacements=[{"group_id": "tg_topk_noop", "replacement_text": topk_current}],
    replacements=[{
        "group_id": "tg_topk_noop",
        "replacement_text": topk_current,
        "changed_spans": [{
            "span_id": "ps001",
            "before": "to teach digital literacy",
            "after": "to teach digital literacy",
            "operation": "TOPK_SPAN_REPATH",
        }],
        "modified_span_ids": ["ps001"],
        "predictable_spans_modified_count": 1,
        "changed_word_estimate": 0,
    }],
    predictability_briefs=topk_noop_briefs,
)
topk_valid_rows, topk_valid_status = validate_topk_replacement_effect(
    target_groups=[topk_noop_group],
    current_replacements=[{"group_id": "tg_topk_noop", "replacement_text": topk_current}],
    replacements=[{
        "group_id": "tg_topk_noop",
        "replacement_text": topk_effective,
        "changed_spans": [{
            "span_id": "ps001",
            "source_span": "to teach digital literacy",
            "before": "they also need to teach digital literacy",
            "after": "digital literacy now has to sit",
            "operation": "TOPK_SPAN_REPATH",
        }],
        "modified_span_ids": ["ps001"],
        "predictable_spans_modified_count": 1,
        "changed_word_estimate": 7,
    }],
    predictability_briefs=topk_noop_briefs,
)
assert_test(
    not topk_noop_rows
    and "no_material_change" in topk_noop_status[0]["failures"]
    and "no_effect_span_patch" in topk_noop_status[0]["failures"]
    and "self_report_mismatch" in topk_noop_status[0]["failures"]
    and topk_valid_rows
    and topk_valid_status[0]["actual_modified_span_ids"] == ["ps001"],
    "V3 paragraph portfolio Top-k gate rejects fake no-effect span repairs and accepts real span movement",
)
ownership_valid_rows, ownership_valid_status = validate_ownership_replacement_effect(
    target_groups=[topk_noop_group],
    replacements=[{
        "group_id": "tg_topk_noop",
        "replacement_text": (
            "Because of this, the education system needs to evolve. "
            "For this classroom problem, I would keep core knowledge in place while making digital literacy and ethical technology use part of how students practise judgment. "
            "Assessment should still include drafts, reflection, discussion, feedback, and improvement because those steps show how the student is learning."
        ),
        "ownership_changes": [{
            "before": "Schools should still teach core knowledge",
            "after": "I would keep core knowledge in place",
            "operation": "CLAIM_OWNERSHIP_REPAIR",
            "trace_source": "source_text",
        }],
        "ownership_elements_supported": ["author_trace", "specific_context", "real_judgment"],
    }],
)
ownership_invalid_rows, ownership_invalid_status = validate_ownership_replacement_effect(
    target_groups=[topk_noop_group],
    replacements=[{
        "group_id": "tg_topk_noop",
        "replacement_text": topk_noop_source,
        "ownership_changes": [],
        "ownership_elements_supported": [],
    }],
)
assert_test(
    ownership_valid_rows
    and ownership_valid_rows[0]["candidate_quality"]["ownership_score"] > 0
    and ownership_valid_status[0]["passed"]
    and not ownership_invalid_rows
    and "no_material_change" in ownership_invalid_status[0]["failures"]
    and "missing_ownership_changes" in ownership_invalid_status[0]["failures"],
    "V3 ownership validator accepts only real source-owned changes with quality metadata",
)
assert_test(
    "operator_used" not in reconstruction_template.prompt
    and "new_claims_added" not in reconstruction_template.prompt
    and "hard_anchors_preserved" not in reconstruction_template.prompt,
    "V3 paragraph portfolio reconstruction prompt does not ask the LLM to self-certify validator fields",
)
bloated_context = {
    **paragraph_context,
    "ai_footprint_profile": {
        "fraction_ai": 0.0,
        "fraction_ai_assisted": 1.0,
        "risky_window_density": 1.0,
        "top_risky_windows": [{"source_text": "x" * 12000, "score": 0.9}],
        "unused_large_blob": "x" * 50000,
    },
    "problem_inventory": {
        "schema_version": "problem_inventory.v1",
        "problem_groups": [
            {
                "group_id": f"pg{i:03d}",
                "scope_level": "paragraph",
                "problem_shape": "broad_assisted_footprint",
                "target_ids": ["rt001"],
                "allowed_operations": ["paragraph_preserving_broad_reconstruction"],
                "expected_movement": {"details": "y" * 4000},
            }
            for i in range(20)
        ],
    },
}
bloated_prompt = build_paragraph_portfolio_planner_prompt(bloated_context)
assert_test(
    len(bloated_prompt.prompt) < 16000
    and "x" * 1000 not in bloated_prompt.prompt
    and "y" * 1000 not in bloated_prompt.prompt,
    "V3 paragraph portfolio planner compacts oversized scanner payloads",
)
assert_test(
    parse_paragraph_portfolio_replacements(json.dumps({
        "replacements": [{"group_id": "tg001", "replacement_text": "Rebuilt paragraph."}]
    })) == [{"group_id": "tg001", "replacement_text": "Rebuilt paragraph."}],
    "V3 paragraph portfolio replacement parser extracts structured replacements",
)
diagnostic_replacements, diagnostic_parse = parse_replacements_with_diagnostics("{not json")
assert_test(
    diagnostic_replacements == []
    and diagnostic_parse["parse_status"] == "invalid_json"
    and diagnostic_parse["raw_preview"],
    "V3 paragraph portfolio parser exposes invalid response diagnostics",
)
diagnostic_replacements, diagnostic_parse = parse_replacements_with_diagnostics(json.dumps({"items": []}))
assert_test(
    diagnostic_replacements == []
    and diagnostic_parse["parse_status"] == "missing_replacements",
    "V3 paragraph portfolio parser reports missing replacements arrays",
)

long_paragraphs = [
    (
        f"Paragraph {index} explains a broad classroom change through several connected details. "
        "Students encounter information through lessons, screens, classmates, and home routines, so the paragraph has enough local material for a realistic prompt-size test. "
        "The rewrite system should not combine too many of these paragraphs into one oversized model request."
    )
    for index in range(1, 5)
]
long_source = "\n\n".join(long_paragraphs)
long_targets = []
for index, paragraph in enumerate(long_paragraphs, start=1):
    long_targets.append({
        "target_id": f"rt_long_{index}",
        "unit_id": f"p{index:03d}",
        "paragraph_id": f"p{index:03d}",
        "scope_level": "paragraph",
        "risk_level": "medium",
        "source_text": paragraph,
        "recommended_operation": "paragraph_preserving_broad_reconstruction",
        "operation_candidates": ["paragraph_preserving_broad_reconstruction"],
        "dominant_drivers": [{"key": "predictability_score", "score": 0.8}],
        "word_count_guide": {"source_words": len(paragraph.split()), "preferred_words": len(paragraph.split())},
    })
long_strategy_profile = {
    "schema_version": "rewrite_target_profile.v1",
    "target_scope_policy": "problem_inventory_driven",
    "operation_mix": {"paragraph_preserving_broad_reconstruction": len(long_targets)},
    "driver_summary": {"predictability_score": len(long_targets)},
    "targets": long_targets,
}
long_contract = build_scan_contract(
    {
        **report_for(long_source, ai=72.0),
        "rewrite_target_profile": long_strategy_profile,
        "problem_inventory": paragraph_problem_inventory,
    },
    long_source,
)
long_groups = group_rewrite_targets(
    original_text=long_source,
    rewrite_target_profile=long_contract.rewrite_target_profile,
    max_groups=8,
)
long_batches = build_reconstruction_batches_by_prompt(
    target_groups=long_groups,
    scan_contract=long_contract,
    content_mode=long_contract.content_mode,
    family="paragraph_preserving_broad_reconstruction",
    planner_output=fallback_paragraph_portfolio_plan(long_groups),
    max_prompt_chars=5200,
    fallback_batch_size=4,
)
assert_test(
    len(long_groups) == 4
    and len(long_batches) > 1
    and all(len(batch) < len(long_groups) for batch in long_batches),
    "V3 paragraph portfolio batches reconstruction by rendered prompt size, not document word count",
)
assert_test(
    paragraph_portfolio_config(fallback_batch_size=4).fallback_batch_size == 1,
    "V3 paragraph portfolio defaults to one target paragraph per reconstruction call",
)

original_gateway_factory = v3_pipeline._gateway
try:
    class FakePromptTemplateResponse:
        def __init__(self, content):
            self.content = content
            self.raw = {}

    class FakePromptTemplateGateway:
        def chat(self, prompt, **kwargs):
            if '"prompt_stage":"planner"' in prompt or '"prompt_stage": "planner"' in prompt:
                return FakePromptTemplateResponse(json.dumps({
                    "paragraph_plans": [{
                        "group_id": "tg001",
                        "paragraph_role": "opening_frame",
                        "risk_drivers": ["predictability_score"],
                        "hard_anchors": [],
                        "soft_anchors": ["old classroom model"],
                        "repeated_patterns": ["balanced explanation"],
                        "recommended_operator": "BREAK_SURVEY_TEMPLATE",
                        "rewrite_aggression_limit": "medium",
                    }]
                }))
            if '"prompt_stage":"topk_repair"' in prompt or '"prompt_stage": "topk_repair"' in prompt:
                return FakePromptTemplateResponse(json.dumps({
                    "replacements": [{
                        "group_id": "tg001",
                        "replacement_text": "Education is changing quickly because students meet information through phones, websites, classmates, and teachers. Schools still matter, but the old classroom model does not explain the whole picture.",
                        "changed_spans": [{
                            "span_id": "ps001",
                            "source_span": "Education is changing quickly",
                            "before": "students now meet information",
                            "after": "students meet information through phones",
                            "operation": "TOPK_SPAN_REPATH",
                        }],
                        "modified_span_ids": ["ps001"],
                        "predictable_spans_modified_count": 1,
                        "changed_word_estimate": 3,
                    }]
                }))
            return FakePromptTemplateResponse(json.dumps({
                "replacements": [{
                    "group_id": "tg001",
                    "replacement_text": "Education is changing quickly because students now meet information through phones, websites, classmates, and teachers. Schools still matter, but the old classroom model does not explain the whole picture."
                }]
            }))

    def fake_gateway_factory(*args, **kwargs):
        return FakePromptTemplateGateway()

    v3_pipeline._gateway = fake_gateway_factory
    os.environ["DRAFTPROOF_REWRITE_V3_PORTFOLIO_LLM_PLANNER"] = "1"
    os.environ["DRAFTPROOF_REWRITE_V3_PORTFOLIO_BLIND_TOPK"] = "1"
    paragraph_text, paragraph_trace = v3_pipeline._generate_target_executor_candidate(
        original_text=broad_source,
        scan_contract=paragraph_contract,
        content_mode=paragraph_contract.content_mode,
        family="paragraph_preserving_broad_reconstruction",
        api_key=None,
        model=None,
        base_url=None,
    )
    assert_test(
        "students meet information through phones" in paragraph_text
        and paragraph_trace["executor_engine"] == "paragraph_portfolio_template"
        and paragraph_trace["prompt_template_id"] == "paragraph_portfolio.v1"
        and paragraph_trace["topk_repair_attempted"]
        and len(paragraph_trace["prompt_stage_trace"]) == 3,
        "V3 paragraph-preserving broad strategy executes planner, reconstruction, and Top-k templates",
    )
finally:
    os.environ.pop("DRAFTPROOF_REWRITE_V3_PORTFOLIO_LLM_PLANNER", None)
    os.environ.pop("DRAFTPROOF_REWRITE_V3_PORTFOLIO_BLIND_TOPK", None)
    v3_pipeline._gateway = original_gateway_factory

original_gateway_factory = v3_pipeline._gateway
try:
    default_calls = {"count": 0, "prompts": []}

    class FakeMinimalGateway:
        def chat(self, prompt, **kwargs):
            default_calls["count"] += 1
            default_calls["prompts"].append(prompt)
            return FakePromptTemplateResponse(json.dumps({
                "replacements": [{
                    "group_id": "tg001",
                    "replacement_text": "Education is changing quickly because students meet information through phones, websites, classmates, and teachers. Schools still matter, but the old classroom model does not explain the whole picture."
                }]
            }))

    def fake_minimal_gateway_factory(*args, **kwargs):
        return FakeMinimalGateway()

    v3_pipeline._gateway = fake_minimal_gateway_factory
    paragraph_text, paragraph_trace = v3_pipeline._generate_target_executor_candidate(
        original_text=broad_source,
        scan_contract=paragraph_contract,
        content_mode=paragraph_contract.content_mode,
        family="paragraph_preserving_broad_reconstruction",
        api_key=None,
        model=None,
        base_url=None,
    )
    assert_test(
        default_calls["count"] == 1
        and paragraph_trace["prompt_stage_trace"][0]["planner_mode"] == "scanner_fallback"
        and paragraph_trace["prompt_stage_trace"][-1]["topk_mode"] == "deferred_until_rescan"
        and not paragraph_trace["topk_repair_attempted"],
        "V3 paragraph portfolio default path uses one reconstruction LLM call and defers Top-k until rescan",
    )
finally:
    v3_pipeline._gateway = original_gateway_factory

original_gateway_factory = v3_pipeline._gateway
try:
    ownership_calls = {"count": 0, "prompts": []}

    class FakeOwnershipGateway:
        def chat(self, prompt, **kwargs):
            ownership_calls["count"] += 1
            ownership_calls["prompts"].append(prompt)
            return FakePromptTemplateResponse(json.dumps({
                "replacements": [{
                    "group_id": "tg001",
                    "replacement_text": (
                        "Education is changing quickly because students meet information through phones, websites, classmates, and teachers. "
                        "In this classroom problem, I would still keep schools in the picture, but I would not treat the old classroom model as the whole explanation for learning."
                    ),
                    "ownership_changes": [{
                        "before": "Schools still matter",
                        "after": "I would still keep schools in the picture",
                        "operation": "CLAIM_OWNERSHIP_REPAIR",
                        "trace_source": "source_text",
                    }],
                    "ownership_elements_supported": ["author_trace", "specific_context", "real_judgment"],
                }]
            }))

    def fake_ownership_gateway_factory(*args, **kwargs):
        return FakeOwnershipGateway()

    v3_pipeline._gateway = fake_ownership_gateway_factory
    ownership_text, ownership_trace = v3_pipeline._generate_scanner_controlled_candidate(
        original_text=broad_source,
        original_report={
            **report_for(broad_source, ai=72.0),
            "rewrite_target_profile": paragraph_strategy_profile,
            "problem_inventory": paragraph_problem_inventory,
        },
        scan_contract=paragraph_contract,
        content_mode=paragraph_contract.content_mode,
        family="paragraph_preserving_broad_reconstruction",
        api_key=None,
        model=None,
        base_url=None,
        ownership_repair_mode=True,
    )
    assert_test(
        "I would still keep schools in the picture" in ownership_text
        and ownership_trace["executor_engine"] == "paragraph_ownership_template"
        and ownership_trace["ownership_repair_mode"]
        and ownership_trace["target_replacements"][0]["candidate_quality"]["ownership_score"] > 0,
        "V3 ownership repair routes broad paragraph groups through the paragraph ownership executor",
    )
finally:
    v3_pipeline._gateway = original_gateway_factory

two_paragraph_strategy_profile = {
    "schema_version": "rewrite_target_profile.v1",
    "target_scope_policy": "problem_inventory_driven",
    "operation_mix": {"paragraph_preserving_broad_reconstruction": 2},
    "driver_summary": {"predictability_score": 2},
    "targets": [
        {
            "target_id": "rt_p1",
            "unit_id": "p001",
            "paragraph_id": "p001",
            "scope_level": "paragraph",
            "risk_level": "medium",
            "source_text": broad_source.split("\n\n")[0],
            "recommended_operation": "paragraph_preserving_broad_reconstruction",
            "operation_candidates": ["paragraph_preserving_broad_reconstruction"],
            "word_count_guide": {"source_words": len(broad_source.split("\n\n")[0].split()), "preferred_words": len(broad_source.split("\n\n")[0].split())},
        },
        {
            "target_id": "rt_p2",
            "unit_id": "p002",
            "paragraph_id": "p002",
            "scope_level": "paragraph",
            "risk_level": "medium",
            "source_text": broad_source.split("\n\n")[1],
            "recommended_operation": "paragraph_preserving_broad_reconstruction",
            "operation_candidates": ["paragraph_preserving_broad_reconstruction"],
            "word_count_guide": {"source_words": len(broad_source.split("\n\n")[1].split()), "preferred_words": len(broad_source.split("\n\n")[1].split())},
        },
    ],
}
two_paragraph_contract = build_scan_contract(
    {
        **report_for(broad_source, ai=70.0),
        "rewrite_target_profile": two_paragraph_strategy_profile,
        "problem_inventory": paragraph_problem_inventory,
    },
    broad_source,
)
original_gateway_factory = v3_pipeline._gateway
try:
    class FakePartialGateway:
        def chat(self, prompt, **kwargs):
            return FakePromptTemplateResponse(json.dumps({
                "replacements": [{
                    "group_id": "tg001",
                    "replacement_text": "Education is changing quickly because students meet information through phones, websites, classmates, and teachers."
                }]
            }))

    def fake_partial_gateway_factory(*args, **kwargs):
        return FakePartialGateway()

    v3_pipeline._gateway = fake_partial_gateway_factory
    partial_text, partial_trace = v3_pipeline._generate_target_executor_candidate(
        original_text=broad_source,
        scan_contract=two_paragraph_contract,
        content_mode=two_paragraph_contract.content_mode,
        family="paragraph_preserving_broad_reconstruction",
        api_key=None,
        model=None,
        base_url=None,
    )
    assert_test(
        "students meet information through phones" in partial_text
        and partial_trace["error"] is None
        and partial_trace["partial_candidate"] is True
        and partial_trace["missing_replacement_group_ids"] == ["tg002"],
        "V3 paragraph portfolio applies valid partial replacements instead of discarding the whole candidate",
    )
finally:
    v3_pipeline._gateway = original_gateway_factory

original_target_executor_generator = v3_pipeline._generate_target_executor_candidate
original_scan_report = v3_pipeline._scan_report
try:
    target_executor_called = {"value": False}

    def fake_target_executor_for_strategy(**kwargs):
        target_executor_called["value"] = True
        return broad_candidate, {
            "target_execution_attempted": True,
            "prompt_template_id": "paragraph_portfolio.v1",
            "prompt_stage": "topk_repair",
            "scanner_context_used": ["rewrite_target_profile.targets"],
            "planner_output": {"paragraph_plans": [{"group_id": "tg001"}]},
            "stage_apply_status": [{"group_id": "tg001", "applied": True}],
            "stage_rescan_delta": None,
            "topk_repair_attempted": True,
            "strategy_stop_reason": "topk_repair_applied",
            "target_groups": [],
            "target_replacements": [],
            "target_apply_status": [],
        }

    def fake_scan_report(text):
        return report_for(text, ai=45.0)

    v3_pipeline._generate_target_executor_candidate = fake_target_executor_for_strategy
    v3_pipeline._scan_report = fake_scan_report
    with tempfile.TemporaryDirectory() as tmpdir:
        paragraph_first_report = report_for(broad_source, ai=70.0)
        paragraph_first_report["ai_risk_badge"]["ai_components"]["topk_pattern_raw"] = 40.0
        paragraph_first = run_rewrite_pipeline_v3(
            detect_json={
                **paragraph_first_report,
                "rewrite_target_profile": paragraph_strategy_profile,
                "problem_inventory": paragraph_problem_inventory,
            },
            output_dir=tmpdir,
            required_ai_drop=20.0,
            max_runtime_seconds=0,
        )
        paragraph_summary = paragraph_first["result"].summary
        paragraph_trace = paragraph_summary["candidate_trace"][0]["target_execution_trace"]
        assert_test(
            target_executor_called["value"]
            and paragraph_summary["strategy_trace"][0]["first_strategy_step"] == "paragraph_preserving_broad_reconstruction"
            and paragraph_summary["strategy_trace"][0]["first_strategy_obeyed"]
            and paragraph_summary["candidate_trace"][0]["generation_mode"] == "paragraph_preserving_broad_reconstruction"
            and paragraph_summary["candidate_trace"][0]["prompt_template_id"] == "paragraph_portfolio.v1"
            and paragraph_summary["candidate_trace"][0]["topk_repair_attempted"]
            and paragraph_trace["executor_engine"] == "target_executor",
            "V3 executes paragraph-preserving broad reconstruction before chunk fallback",
        )
finally:
    v3_pipeline._generate_target_executor_candidate = original_target_executor_generator
    v3_pipeline._scan_report = original_scan_report

original_target_executor_generator = v3_pipeline._generate_target_executor_candidate
original_scanner_controlled_generator = v3_pipeline._generate_scanner_controlled_candidate
original_scan_report = v3_pipeline._scan_report
try:
    planned_calls = []
    ownership_calls = {"count": 0}
    mixed_problem_inventory = {
        "schema_version": "problem_inventory.v1",
        "problem_groups": [
            {
                "group_id": "pg_paragraph",
                "scope_level": "paragraph",
                "unit_ids": ["p001"],
                "target_ids": ["rt001"],
                "problem_shape": "broad_assisted_footprint",
                "anchor_pressure": 0.1,
                "semantic_edit_cost": 0.2,
                "allowed_operations": ["paragraph_preserving_broad_reconstruction", "chunk_reconstruction"],
                "blocked_operations": [],
            },
            {
                "group_id": "pg_protected",
                "scope_level": "paragraph",
                "unit_ids": ["p002"],
                "target_ids": ["rt002"],
                "problem_shape": "protected_section_risk",
                "anchor_pressure": 0.7,
                "semantic_edit_cost": 0.7,
                "allowed_operations": ["protected_section_rewrite", "chunk_reconstruction"],
                "blocked_operations": [],
            },
        ],
    }

    def fake_target_executor_with_unapplied_group(**kwargs):
        return broad_candidate, {
            "target_execution_attempted": True,
            "prompt_template_id": "paragraph_portfolio.v1",
            "target_groups": [{"group_id": "tg001"}, {"group_id": "tg002"}],
            "target_replacements": [{"group_id": "tg001", "replacement_text": "x"}],
            "target_apply_status": [
                {"group_id": "tg001", "applied": True, "method": "span", "reason": ""},
                {"group_id": "tg002", "applied": False, "method": "none", "reason": "protected_anchor_missing"},
            ],
        }

    def fake_planned_scanner_controlled(**kwargs):
        planned_calls.append(kwargs.get("strategy_id"))
        return (
            broad_candidate.replace("handing out facts", "delivering facts in a fixed order"),
            {
                "target_execution_attempted": True,
                "scanner_controlled": True,
                "executor_engine": "scanner_controlled_executor",
                "planned_strategy_id": kwargs.get("strategy_id"),
                "target_groups": [{"group_id": "tg002"}],
                "target_replacements": [{"group_id": "tg002", "replacement_text": "replacement"}],
                "target_apply_status": [{"group_id": "tg002", "applied": True, "method": "span", "reason": ""}],
            },
        )

    def fake_ownership_preempting_scanner_controlled(**kwargs):
        if kwargs.get("ownership_repair_mode"):
            ownership_calls["count"] += 1
            return (
                broad_candidate.replace("Schools are handing out facts", "In this classroom example, I would not treat school as only handing out facts"),
                {
                    "target_execution_attempted": True,
                    "scanner_controlled": True,
                    "executor_engine": "scanner_controlled_executor",
                    "ownership_repair_mode": True,
                    "scanner_controlled_accepted": [{
                        "group_id": "tg001",
                        "replacement_text": "In this classroom example, I would not treat school as only handing out facts",
                        "candidate_quality": {
                            "ownership_score": 6.0,
                            "ownership_change_count": 1,
                            "ownership_elements_supported": ["author_trace", "specific_context", "real_judgment"],
                        },
                    }],
                    "target_groups": [{"group_id": "tg001"}, {"group_id": "tg002"}],
                    "target_replacements": [{"group_id": "tg001", "replacement_text": "x"}],
                    "target_apply_status": [{"group_id": "tg001", "applied": True, "method": "span", "reason": ""}],
                },
            )
        return fake_planned_scanner_controlled(**kwargs)

    def fake_scan_report(text):
        return report_for(text, ai=45.0)

    v3_pipeline._generate_target_executor_candidate = fake_target_executor_with_unapplied_group
    v3_pipeline._generate_scanner_controlled_candidate = fake_planned_scanner_controlled
    v3_pipeline._scan_report = fake_scan_report
    with tempfile.TemporaryDirectory() as tmpdir:
        planned_followup_report = report_for(broad_source, ai=70.0)
        planned_followup_report["ai_risk_badge"]["ai_components"]["topk_pattern_raw"] = 40.0
        planned_followup = run_rewrite_pipeline_v3(
            detect_json={
                **planned_followup_report,
                "rewrite_target_profile": paragraph_strategy_profile,
                "problem_inventory": mixed_problem_inventory,
            },
            output_dir=tmpdir,
            required_ai_drop=20.0,
            max_runtime_seconds=60,
        )
        planned_summary = planned_followup["result"].summary
        assert_test(
            not planned_summary["candidate_trace"][0]["target_gate_passed"]
            and planned_summary["candidate_trace"][0]["unapplied_target_group_ids"] == ["tg002"]
            and planned_calls[:1] == ["protected_section_rewrite"]
            and planned_summary["candidate_loop_trace"][0]["reason"] == "execute_planned_problem_strategy:protected_section_rewrite",
            "V3 executes the next planned problem strategy when target application leaves unresolved groups",
        )
finally:
    v3_pipeline._generate_target_executor_candidate = original_target_executor_generator
    v3_pipeline._generate_scanner_controlled_candidate = original_scanner_controlled_generator
    v3_pipeline._scan_report = original_scan_report

original_target_executor_generator = v3_pipeline._generate_target_executor_candidate
original_scanner_controlled_generator = v3_pipeline._generate_scanner_controlled_candidate
original_scan_report = v3_pipeline._scan_report
try:
    planned_calls = []
    ownership_calls = {"count": 0}

    def fake_partial_target_executor_ownership_missing(**kwargs):
        return broad_candidate.replace("handing out facts", "handing out facts in one repeated pattern"), {
            "target_execution_attempted": True,
            "prompt_template_id": "paragraph_portfolio.v1",
            "executor_engine": "paragraph_portfolio_template",
            "partial_candidate": True,
            "missing_replacement_group_ids": ["tg002"],
            "target_groups": [{"group_id": "tg001"}, {"group_id": "tg002"}],
            "target_replacements": [{"group_id": "tg001", "replacement_text": "x"}],
            "target_apply_status": [
                {"group_id": "tg001", "applied": True, "method": "span", "reason": ""},
                {"group_id": "tg002", "applied": False, "method": "none", "reason": "missing_replacement"},
            ],
        }

    def fake_ownership_repair_after_partial(**kwargs):
        if kwargs.get("ownership_repair_mode"):
            ownership_calls["count"] += 1
            return (
                broad_candidate.replace("Schools are handing out facts", "In my classroom judgment, schools cannot be reduced to handing out facts"),
                {
                    "target_execution_attempted": True,
                    "scanner_controlled": True,
                    "executor_engine": "scanner_controlled_executor",
                    "ownership_repair_mode": True,
                    "scanner_controlled_accepted": [{
                        "group_id": "tg001",
                        "replacement_text": "In my classroom judgment, schools cannot be reduced to handing out facts",
                        "candidate_quality": {
                            "ownership_score": 6.0,
                            "ownership_change_count": 1,
                            "ownership_elements_supported": ["author_trace", "specific_context", "real_judgment"],
                        },
                    }],
                    "target_groups": [{"group_id": "tg001"}],
                    "target_replacements": [{"group_id": "tg001", "replacement_text": "x"}],
                    "target_apply_status": [{"group_id": "tg001", "applied": True, "method": "span", "reason": ""}],
                },
            )
        planned_calls.append(kwargs.get("strategy_id"))
        return broad_candidate, {
            "target_execution_attempted": True,
            "scanner_controlled": True,
            "executor_engine": "scanner_controlled_executor",
            "planned_strategy_id": kwargs.get("strategy_id"),
            "target_groups": [{"group_id": "tg002"}],
            "target_replacements": [],
            "target_apply_status": [{"group_id": "tg002", "applied": False, "method": "none", "reason": "not_called"}],
        }

    def fake_ownership_missing_scan_report(text):
        report = report_for(text, ai=45.0)
        profile = {
            "schema_version": "ai_footprint_profile.v2",
            "fraction_ai": 0.0,
            "fraction_ai_assisted": 1.0,
            "fraction_human": 0.0,
            "risky_window_density": 1.0,
            "max_ai_window_words": 0,
            "top_risky_windows": [],
        }
        report["ai_footprint_profile"] = dict(profile)
        report["authorship_window_profile"] = dict(profile)
        return report

    v3_pipeline._generate_target_executor_candidate = fake_partial_target_executor_ownership_missing
    v3_pipeline._generate_scanner_controlled_candidate = fake_ownership_repair_after_partial
    v3_pipeline._scan_report = fake_ownership_missing_scan_report
    with tempfile.TemporaryDirectory() as tmpdir:
        ownership_preempt_report = report_for(broad_source, ai=70.0)
        ownership_preempt_report["ai_risk_badge"]["ai_components"]["topk_pattern_raw"] = 40.0
        ownership_preempt = run_rewrite_pipeline_v3(
            detect_json={
                **ownership_preempt_report,
                "rewrite_target_profile": paragraph_strategy_profile,
                "problem_inventory": mixed_problem_inventory,
            },
            output_dir=tmpdir,
            required_ai_drop=20.0,
            max_runtime_seconds=60,
        )
        ownership_summary = ownership_preempt["result"].summary
        assert_test(
            ownership_calls["count"] == 1
            and ownership_summary["candidate_loop_trace"][0]["action"] == "claim_ownership_repair"
            and ownership_summary["candidate_loop_trace"][0]["reason"] == "claim_ownership_repair_before_problem_strategy_exhaustion",
            "V3 claim ownership repair preempts planned problem strategies after a partial portfolio candidate fails ownership",
        )
finally:
    v3_pipeline._generate_target_executor_candidate = original_target_executor_generator
    v3_pipeline._generate_scanner_controlled_candidate = original_scanner_controlled_generator
    v3_pipeline._scan_report = original_scan_report

original_target_executor_generator = v3_pipeline._generate_target_executor_candidate
original_plain_reasoning_generator = v3_pipeline._generate_plain_reasoning_candidate
original_scan_report = v3_pipeline._scan_report
try:
    plain_reasoning_calls = {"count": 0}

    def fake_target_executor_no_movement(**kwargs):
        return broad_candidate, {
            "target_execution_attempted": True,
            "prompt_template_id": "paragraph_portfolio.v1",
            "target_groups": [{"group_id": "tg001"}],
            "target_replacements": [],
            "target_apply_status": [{"group_id": "tg001", "applied": False, "method": "none", "reason": "no_safe_replacement"}],
        }

    def fake_plain_reasoning_should_not_run(**kwargs):
        plain_reasoning_calls["count"] += 1
        return "plain reasoning should not run"

    def fake_no_movement_scan_report(text):
        return report_for(text, ai=70.0)

    v3_pipeline._generate_target_executor_candidate = fake_target_executor_no_movement
    v3_pipeline._generate_plain_reasoning_candidate = fake_plain_reasoning_should_not_run
    v3_pipeline._scan_report = fake_no_movement_scan_report
    with tempfile.TemporaryDirectory() as tmpdir:
        exhausted_report = report_for(broad_source, ai=70.0)
        exhausted_report["ai_risk_badge"]["ai_components"]["topk_pattern_raw"] = 40.0
        exhausted_problem = run_rewrite_pipeline_v3(
            detect_json={
                **exhausted_report,
                "rewrite_target_profile": paragraph_strategy_profile,
                "problem_inventory": paragraph_problem_inventory,
            },
            output_dir=tmpdir,
            required_ai_drop=20.0,
            max_runtime_seconds=60,
        )
        exhausted_summary = exhausted_problem["result"].summary
        assert_test(
            plain_reasoning_calls["count"] == 0
            and exhausted_summary["candidate_loop_trace"][0]["reason"] == "stop_after_problem_strategy_exhausted"
            and exhausted_summary["candidate_trace"][0]["target_gate_passed"] is False,
            "V3 stops scanner-driven problem runs after exhausted no-movement targets instead of falling back to full-document plain reasoning",
        )
finally:
    v3_pipeline._generate_target_executor_candidate = original_target_executor_generator
    v3_pipeline._generate_plain_reasoning_candidate = original_plain_reasoning_generator
    v3_pipeline._scan_report = original_scan_report

chunk_strategy_profile = {
    "schema_version": "rewrite_target_profile.v1",
    "target_scope_policy": "problem_inventory_driven",
    "operation_mix": {"chunk_reconstruction": 1},
    "driver_summary": {"ai_footprint": 1},
    "targets": [{
        "target_id": "rt_chunk",
        "unit_id": "document",
        "scope_level": "document",
        "risk_level": "high",
        "source_text": broad_source,
        "recommended_operation": "chunk_reconstruction",
        "operation_candidates": ["chunk_reconstruction"],
    }],
}
chunk_problem_inventory = {
    "schema_version": "problem_inventory.v1",
    "problem_groups": [{
        "group_id": "pg_chunk",
        "scope_level": "document",
        "unit_ids": ["document"],
        "target_ids": ["rt_chunk"],
        "problem_shape": "broad_assisted_footprint",
        "anchor_pressure": 0.0,
        "semantic_edit_cost": 0.5,
        "allowed_operations": ["chunk_reconstruction"],
        "blocked_operations": ["unit_preserving_prune_bridge"],
    }],
}
original_chunked_generator = v3_pipeline._generate_chunked_candidate
original_scan_report = v3_pipeline._scan_report
try:
    chunk_called = {"value": False}

    def fake_chunked_generator(**kwargs):
        chunk_called["value"] = True
        return broad_candidate

    def fake_scan_report(text):
        return report_for(text, ai=45.0)

    v3_pipeline._generate_chunked_candidate = fake_chunked_generator
    v3_pipeline._scan_report = fake_scan_report
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_first = run_rewrite_pipeline_v3(
            detect_json={
                **report_for(broad_source, ai=70.0),
                "rewrite_target_profile": chunk_strategy_profile,
                "problem_inventory": chunk_problem_inventory,
            },
            output_dir=tmpdir,
            required_ai_drop=20.0,
            max_runtime_seconds=0,
        )
        chunk_summary = chunk_first["result"].summary
        assert_test(
            chunk_called["value"]
            and chunk_summary["strategy_trace"][0]["first_strategy_step"] == "chunk_reconstruction"
            and chunk_summary["strategy_trace"][0]["first_strategy_obeyed"]
            and chunk_summary["candidate_trace"][0]["generation_mode"] == "chunk_reconstruction",
            "V3 executes problem-inventory chunk strategy before scanner-controlled fallback",
        )
finally:
    v3_pipeline._generate_chunked_candidate = original_chunked_generator
    v3_pipeline._scan_report = original_scan_report

protected_problem_inventory = {
    "schema_version": "problem_inventory.v1",
    "problem_groups": [{
        "group_id": "pg_protected",
        "scope_level": "section",
        "unit_ids": ["p001"],
        "target_ids": ["rt001"],
        "problem_shape": "anchored_local_window",
        "anchor_pressure": 0.4,
        "semantic_edit_cost": 0.3,
        "allowed_operations": ["protected_section_rewrite", "chunk_reconstruction"],
        "blocked_operations": ["unit_preserving_prune_bridge"],
    }],
}
original_scanner_controlled_generator = v3_pipeline._generate_scanner_controlled_candidate
original_scan_report = v3_pipeline._scan_report
try:
    def fake_scanner_controlled_for_strategy(**kwargs):
        return broad_candidate, {
            "target_execution_attempted": True,
            "scanner_controlled": True,
            "scanner_controlled_rounds": [],
            "target_groups": [],
            "target_replacements": [],
            "target_apply_status": [],
        }

    def fake_scan_report(text):
        return report_for(text, ai=45.0)

    v3_pipeline._generate_scanner_controlled_candidate = fake_scanner_controlled_for_strategy
    v3_pipeline._scan_report = fake_scan_report
    with tempfile.TemporaryDirectory() as tmpdir:
        protected_first = run_rewrite_pipeline_v3(
            detect_json={
                **report_for(broad_source, ai=70.0),
                "rewrite_target_profile": target_gate_fixture,
                "problem_inventory": protected_problem_inventory,
            },
            output_dir=tmpdir,
            required_ai_drop=20.0,
            max_runtime_seconds=0,
        )
        protected_summary = protected_first["result"].summary
        protected_trace = protected_summary["candidate_trace"][0]["target_execution_trace"]
        assert_test(
            protected_summary["strategy_trace"][0]["first_strategy_step"] == "protected_section_rewrite"
            and protected_summary["strategy_trace"][0]["first_strategy_obeyed"]
            and protected_summary["candidate_trace"][0]["generation_mode"] == "protected_section_rewrite"
            and protected_trace["executor_engine"] == "scanner_controlled_executor"
            and protected_trace["executed_strategy"] == "protected_section_rewrite",
            "V3 records scanner-controlled execution under the planned protected-section strategy",
        )
finally:
    v3_pipeline._generate_scanner_controlled_candidate = original_scanner_controlled_generator
    v3_pipeline._scan_report = original_scan_report

with tempfile.TemporaryDirectory() as tmpdir:
    target_blocked = run_rewrite_pipeline_v3(
        detect_json={**report_for(broad_source, ai=70.0), "rewrite_target_profile": target_gate_fixture},
        output_dir=tmpdir,
        replay_candidate_records=[{
            "text": broad_candidate,
            "report": {**report_for(broad_candidate, ai=40.0), "rewrite_target_profile": target_gate_fixture},
        }],
        required_ai_drop=20.0,
    )
    blocked_summary = target_blocked["result"].summary
    assert_test(
        target_blocked["status"] == "rewrite_candidate_generated_needs_external_review"
        and not blocked_summary["selected_candidate"]["target_gate_passed"],
        "V3 candidates cannot pass mitigation success without target-level movement",
    )

original_target_executor_generator = v3_pipeline._generate_target_executor_candidate
try:
    def fake_empty_target_executor(**kwargs):
        return "", {
            "target_execution_attempted": True,
            "error": "generation_failed_empty_output",
            "target_groups": [],
            "target_replacements": [],
            "target_apply_status": [],
        }

    v3_pipeline._generate_target_executor_candidate = fake_empty_target_executor
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_primary_report = report_for(broad_source, ai=70.0)
        empty_primary_report["ai_risk_badge"]["ai_components"]["topk_pattern_raw"] = 40.0
        empty_primary = run_rewrite_pipeline_v3(
            detect_json={
                **empty_primary_report,
                "rewrite_target_profile": paragraph_strategy_profile,
                "problem_inventory": paragraph_problem_inventory,
            },
            output_dir=tmpdir,
            required_ai_drop=20.0,
            max_runtime_seconds=60,
        )
        empty_summary = empty_primary["result"].summary
        assert_test(
            len(empty_summary["candidate_trace"]) == 1
            and empty_summary["candidate_trace"][0]["candidate_outcome"] == "generation_failed_empty_output",
            "V3 stops after primary paragraph-portfolio generation failure instead of burning fallback LLM calls",
        )
finally:
    v3_pipeline._generate_target_executor_candidate = original_target_executor_generator

original_next_planned_problem_strategy = v3_pipeline._next_planned_problem_strategy
original_decide_next_action = v3_pipeline.decide_next_action
original_recovery_generator = v3_pipeline._generate_recovery_candidate
try:
    recovery_called = {"value": False}

    def fake_no_planned_problem_strategy(**kwargs):
        return None

    def fake_decide_targeted_repair(*args, **kwargs):
        return v3_pipeline.LoopDecision(
            action=CandidateAction.REPAIR_TARGETED,
            source_index=0,
            issues=(CandidateIssue.SEGMENT_AI_FOOTPRINT,),
            reason="target_unresolved_candidate_issues",
        )

    def fake_recovery_should_not_run(**kwargs):
        recovery_called["value"] = True
        raise AssertionError("unbounded recovery revision should not run for problem-inventory V3")

    v3_pipeline._next_planned_problem_strategy = fake_no_planned_problem_strategy
    v3_pipeline.decide_next_action = fake_decide_targeted_repair
    v3_pipeline._generate_recovery_candidate = fake_recovery_should_not_run
    with tempfile.TemporaryDirectory() as tmpdir:
        stopped_recovery = run_rewrite_pipeline_v3(
            detect_json={
                **report_for(broad_source, ai=70.0),
                "rewrite_target_profile": paragraph_strategy_profile,
                "problem_inventory": paragraph_problem_inventory,
            },
            output_dir=tmpdir,
            replay_candidate_records=[{
                "text": broad_candidate,
                "report": report_for(broad_candidate, ai=60.0),
            }],
            required_ai_drop=20.0,
            max_runtime_seconds=60,
        )
        stopped_summary = stopped_recovery["result"].summary
        assert_test(
            not recovery_called["value"]
            and stopped_summary["candidate_loop_trace"][0]["reason"] == "stop_before_unbounded_recovery_revision"
            and len(stopped_summary["candidate_trace"]) == 1,
            "V3 problem-inventory loop stops before unbounded whole-document recovery revision",
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        stopped_generic_recovery = run_rewrite_pipeline_v3(
            detect_json=report_for(broad_source, ai=70.0),
            output_dir=tmpdir,
            replay_candidate_records=[{
                "text": broad_candidate,
                "report": report_for(broad_candidate, ai=60.0),
            }],
            required_ai_drop=20.0,
            max_runtime_seconds=60,
        )
        stopped_generic_summary = stopped_generic_recovery["result"].summary
        assert_test(
            not recovery_called["value"]
            and stopped_generic_summary["candidate_loop_trace"][0]["reason"] == "stop_before_unbounded_recovery_revision"
            and len(stopped_generic_summary["candidate_trace"]) == 1,
            "V3 generic loop also stops before unbounded whole-document recovery revision",
        )
finally:
    v3_pipeline._next_planned_problem_strategy = original_next_planned_problem_strategy
    v3_pipeline.decide_next_action = original_decide_next_action
    v3_pipeline._generate_recovery_candidate = original_recovery_generator

with tempfile.TemporaryDirectory() as tmpdir:
    result = run_rewrite_pipeline_v3(
        detect_json=report_for(cited_source, mode="academic_cited_text", ai=55.0),
        output_dir=tmpdir,
        replay_candidate_records=[{
            "text": cited_candidate,
            "ai": 45.0,
            "wq": 52.0,
            "topk": 70.0,
        }],
        required_ai_drop=20.0,
    )
    assert_test(
        result["result"].summary["strategy_trace"][0]["strategy_family"] == "cited_practice_voice",
        "V3 routes cited academic content to cited-practice voice family",
    )

old_min_completion_tokens = os.environ.get("DRAFTPROOF_REWRITE_V3_MIN_COMPLETION_TOKENS")
old_max_completion_tokens = os.environ.get("DRAFTPROOF_REWRITE_V3_MAX_COMPLETION_TOKENS")
try:
    os.environ.pop("DRAFTPROOF_REWRITE_V3_MIN_COMPLETION_TOKENS", None)
    os.environ.pop("DRAFTPROOF_REWRITE_V3_MAX_COMPLETION_TOKENS", None)
    assert_test(
        v3_pipeline._max_tokens_for_words(557) >= 5000
        and v3_pipeline._max_tokens_for_words(12000) <= 12000,
        "V3 completion budget defaults to a 5000-token floor and bounded cap",
    )
    os.environ["DRAFTPROOF_REWRITE_V3_MIN_COMPLETION_TOKENS"] = "7000"
    os.environ["DRAFTPROOF_REWRITE_V3_MAX_COMPLETION_TOKENS"] = "8000"
    assert_test(
        v3_pipeline._max_tokens_for_words(100) == 7000
        and v3_pipeline._max_tokens_for_words(20000) == 8000,
        "V3 completion budget supports configurable floor and cap",
    )
finally:
    if old_min_completion_tokens is None:
        os.environ.pop("DRAFTPROOF_REWRITE_V3_MIN_COMPLETION_TOKENS", None)
    else:
        os.environ["DRAFTPROOF_REWRITE_V3_MIN_COMPLETION_TOKENS"] = old_min_completion_tokens
    if old_max_completion_tokens is None:
        os.environ.pop("DRAFTPROOF_REWRITE_V3_MAX_COMPLETION_TOKENS", None)
    else:
        os.environ["DRAFTPROOF_REWRITE_V3_MAX_COMPLETION_TOKENS"] = old_max_completion_tokens

length_response = LLMResponse(
    content='{"replacements": [',
    model="test-model",
    raw={"choices": [{"finish_reason": "length", "native_finish_reason": "max_tokens"}]},
)
assert_test(
    length_response.finish_reason == "length"
    and length_response.native_finish_reason == "max_tokens"
    and v3_pipeline._reject_length_limited_response(length_response, stage="test") == "",
    "V3 exposes and rejects length-limited LLM responses before candidate selection",
)

print("Rewrite V3 pipeline tests passed.")
