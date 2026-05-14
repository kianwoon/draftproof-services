"""Focused tests for external-calibrated rewrite pipeline V3."""

from __future__ import annotations

import inspect
import json
import os
import tempfile

from rewrite_v2.contracts import build_rewrite_contract
import rewrite_v3.pipeline as v3_pipeline
from rewrite_v3.anchor_validation import validate_v3_candidate
from rewrite_v3.candidate_loop import CandidateAction, CandidateIssue, decide_next_action, issues_from_trace
from rewrite_v3.compression_policy import compression_policy_for_family, compression_status
from rewrite_v3.document_units import compact_document_inventory, document_units, word_count
from rewrite_v3.external_proxy import evaluate_external_proxy
from rewrite_v3.layers.boundary_adapter import build_boundary_adapter_prompt
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


def assert_test(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


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
    assert_test(result["status"] == "external_calibrated_candidate_applied", "V3 can apply externally calibrated improvement without strict goal")
    assert_test(summary["rewrite_pipeline_version"] == "rewrite_v3_external_calibrated", "V3 summary declares V3 pipeline version")
    assert_test(summary["strategy_trace"][0]["strategy_family"] == "document_rhythm", "V3 routes broad content to document rhythm family")
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
    assert_test(result["status"] == "external_calibrated_candidate_applied", "V3 recovery replay can select a later candidate")
    assert_test(summary["selected_candidate"]["generation_mode"] == "replay_recovery_2", "V3 does not stop at failed first candidate")

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
