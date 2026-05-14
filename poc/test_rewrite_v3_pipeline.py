"""Focused tests for external-calibrated rewrite pipeline V3."""

from __future__ import annotations

import inspect
import json
import os
import tempfile

from llm.gateway import LLMConfig, LLMGateway
from rewrite_v2.contracts import build_rewrite_contract
from detect.authorship_windows import build_ai_footprint_profile, build_authorship_window_profile
from detect.rewrite_targets import build_rewrite_target_profile
import rewrite_v3.pipeline as v3_pipeline
from rewrite_v3.anchor_validation import validate_v3_candidate
from rewrite_v3.assisted_footprint_executor import group_assisted_footprint_windows
from rewrite_v3.authorship_window_gate import evaluate_authorship_window_gate, select_authorship_window_targets
from rewrite_v3.candidate_loop import CandidateAction, CandidateIssue, decide_next_action, issues_from_trace
from rewrite_v3.compression_policy import compression_policy_for_family, compression_status
from rewrite_v3.document_units import compact_document_inventory, document_units, word_count
from rewrite_v3.external_proxy import evaluate_external_proxy
from rewrite_v3.layers.authorship_window_repair import (
    apply_authorship_window_replacements,
    build_authorship_window_repair_prompt,
    extract_authorship_window_replacements,
)
from rewrite_v3.layers.boundary_adapter import build_boundary_adapter_prompt
from rewrite_v3.layers.clean_texture_boundary import build_clean_texture_boundary_prompt
from rewrite_v3.layers.contract_repair import build_contract_repair_prompt
from rewrite_v3.layers.contrast_boundary import build_contrast_boundary_prompt, extract_contrast_boundary_output
from rewrite_v3.layers.document_rhythm import build_document_rhythm_chunk_prompt
from rewrite_v3.layers.plain_reasoning_broad_prose import build_plain_reasoning_broad_prose_prompt
from rewrite_v3.output_cleaning import clean_v3_candidate_output
from rewrite_v3.pipeline import run_rewrite_pipeline_v3
from rewrite_v3.portfolio import select_portfolio_candidate
from rewrite_v3.router import route_from_scan_contract
from rewrite_v3.scanner_contract import RewriteRiskClass, build_scan_contract
from rewrite_v3.strategy_plan import build_strategy_plan
from rewrite_v3.target_executor import apply_target_replacements, batch_target_groups, group_rewrite_targets, parse_target_replacements


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
    target_profile["targets"][0]["recommended_operation"] in {
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
assert_test(
    target_groups and target_groups[0].source_text,
    "V3 target executor groups scanner targets into bounded replacement units",
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
missing_anchor_applied, missing_anchor_status = apply_target_replacements(
    original_text=(
        "A short human paragraph with concrete observation.\n\n"
        "A riskier paragraph follows a predictable route and keeps a uniform explanation."
    ),
    target_groups=target_groups,
    replacements=[{"group_id": target_groups[0].group_id, "replacement_text": "This replacement drops the required phrase."}],
)
assert_test(
    "required phrase" not in missing_anchor_applied and not missing_anchor_status[0]["applied"],
    "V3 target executor blocks replacements that drop protected anchors",
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
        }
    }],
    has_positive_boundaries=False,
    tried_actions=set(),
)
assert_test(
    segment_loop.action == CandidateAction.REPAIR_AUTHORSHIP_WINDOWS,
    "V3 loop routes segment footprint failures to authorship window repair",
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
    "failed_invariants" in contract_prompt and "Source Anchor" in contract_prompt,
    "V3 contract repair prompt carries failed invariants and missing anchors",
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
    "candidate_ai": 45.0,
}
proxy_decision = decide_next_action(
    [{"text": "candidate", "strict_selected": False, "external_selected": False, "trace": proxy_trace}],
    has_positive_boundaries=True,
    tried_actions=set(),
)
assert_test(proxy_decision.action == CandidateAction.ADAPT_BOUNDARY, "V3 loop uses boundary adaptation for unresolved proxy issues")
contrast_decision = decide_next_action(
    [{"text": "candidate", "strict_selected": False, "external_selected": False, "trace": proxy_trace}],
    has_positive_boundaries=True,
    tried_actions={CandidateAction.ADAPT_BOUNDARY},
)
assert_test(contrast_decision.action == CandidateAction.CONTRAST_BOUNDARY, "V3 loop runs contrast boundary after boundary adaptation misses")
plain_decision = decide_next_action(
    [{"text": "candidate", "strict_selected": False, "external_selected": False, "trace": proxy_trace}],
    has_positive_boundaries=True,
    tried_actions={CandidateAction.ADAPT_BOUNDARY, CandidateAction.CONTRAST_BOUNDARY},
)
assert_test(plain_decision.action == CandidateAction.PLAIN_REASONING, "V3 loop runs plain reasoning after contrast boundary misses")

contrast_prompt = build_contrast_boundary_prompt(
    original_text=broad_source,
    failed_candidate=broad_candidate,
    family="document_rhythm",
    compression_policy=compression_policy_for_family("document_rhythm", word_count(broad_source)),
    style_examples={"positive": [{"external_ai_percent": 18, "text": "Plain boundary."}], "negative": []},
)
assert_test("current_failed_rewrite" in contrast_prompt and "positive_boundary_samples" in contrast_prompt, "V3 contrast boundary prompt includes failed and positive examples")
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
assert_test("plain-reasoning style" in plain_prompt and "formal generated-survey texture" in plain_prompt, "V3 plain reasoning prompt targets broad formal survey texture")

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

boundary_prompt = build_boundary_adapter_prompt(
    original_text=eight_unit_source,
    failed_candidates=[seven_unit_candidate],
    strategy_family="document_rhythm",
    proxy_feedback=[],
    compression_policy=compression_policy_for_family("document_rhythm", word_count(eight_unit_source)),
    style_examples={"positive": [], "negative": []},
)
assert_test(
    "same paragraph count as the source" in boundary_prompt,
    "V3 boundary adapter prompt requires source paragraph count preservation",
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

print("Rewrite V3 pipeline tests passed.")
