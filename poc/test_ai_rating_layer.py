#!/usr/bin/env python3
"""Focused tests for the user-facing authorship rating layer."""

import os
from pathlib import Path

from detect.layer3_scoring import (
    Confidence,
    Layer3Input,
    Layer3Scorer,
    QualityTier,
    Tier,
    build_layer3_input_from_text,
    derive_authorship_rating,
)
from detect.transformation import (
    TransformationFeatures,
    classify_transformation,
    classify_transformation_from_scan,
)
from detect.mitigation import build_ai_mitigation_plan
from report.report import Finding, ReportBuilder, Tier as ReportTier, report_to_dict
from detect.semantic_shape import SemanticShapeDetector
from report.render import _authorship_rating_from_badge, render_markdown


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def assert_true(value, message):
    if not value:
        raise AssertionError(message)


cases = [
    (0.10, Tier.GREEN, "low_ai_signal", "Low AI Signal"),
    (0.24, Tier.GREEN, "unlikely_ai", "Unlikely AI"),
    (0.39, Tier.AMBER, "possible_ai_assisted", "Possible AI-Assisted"),
    (0.55, Tier.ORANGE, "possible_ai_assisted", "Possible AI-Assisted"),
    (0.72, Tier.RED, "ai_generated_signals", "AI-Generated / AI-Paraphrased Signals"),
]

for score, tier, code, label in cases:
    rating = derive_authorship_rating(
        ai_score=score,
        ai_tier=tier,
        writing_quality_score=0.20,
        writing_quality_tier=QualityTier.LOW,
        confidence=Confidence.HIGH,
    )
    assert_equal(rating["code"], code, f"{score} maps to {code}")
    assert_equal(rating["label"], label, f"{score} label")
    assert_equal(rating["is_verdict"], False, "rating is not a verdict")
    assert_true("not proof" in rating["disclaimer"], "rating carries authorship disclaimer")

low_signal = derive_authorship_rating(
    ai_score=0.12,
    ai_tier=Tier.GREEN,
    writing_quality_score=0.20,
    writing_quality_tier=QualityTier.LOW,
    confidence=Confidence.HIGH,
)
assert_true(
    any("under 20%" in note for note in low_signal["caution_notes"]),
    "low AI signal includes under-20 false-positive caution",
)

aligned_likely = derive_authorship_rating(
    ai_score=0.55,
    ai_tier=Tier.ORANGE,
    writing_quality_score=0.56,
    writing_quality_tier=QualityTier.REVIEW,
    confidence=Confidence.HIGH,
    ai_components={"topk_pattern": 0.72},
)
assert_equal(aligned_likely["code"], "likely_ai", "48-60 likely AI requires aligned supporting signal")

verified = derive_authorship_rating(
    ai_score=0.05,
    ai_tier=Tier.GREEN,
    writing_quality_score=0.10,
    writing_quality_tier=QualityTier.LOW,
    confidence=Confidence.HIGH,
    verified_ai_provenance=True,
)
assert_equal(verified["code"], "ai_generated", "verified provenance overrides score")
assert_equal(verified["label"], "AI-Generated", "verified provenance label")

low_conf = derive_authorship_rating(
    ai_score=0.41,
    ai_tier=Tier.AMBER,
    writing_quality_score=0.70,
    writing_quality_tier=QualityTier.HIGH_REVIEW,
    confidence=Confidence.LOW,
)
assert_true(low_conf["caution_notes"], "low confidence rating includes caution notes")

humanised = derive_authorship_rating(
    ai_score=0.6315,
    ai_tier=Tier.ORANGE,
    writing_quality_score=0.6761,
    writing_quality_tier=QualityTier.HIGH_REVIEW,
    confidence=Confidence.HIGH,
)
assert_equal(humanised["code"], "ai_generated_signals", "near-red humanised AI profile escalates rating")
assert_true(
    any("Escalated" in note for note in humanised["caution_notes"]),
    "near-red humanised profile includes escalation note",
)

strong_humaniser = derive_authorship_rating(
    ai_score=0.6035,
    ai_tier=Tier.ORANGE,
    writing_quality_score=0.6256,
    writing_quality_tier=QualityTier.REVIEW,
    confidence=Confidence.HIGH,
    ai_components={
        "topk_pattern": 0.8923,
        "generic_assertion_risk": 0.90,
    },
    writing_components={
        "unsupported_claim_risk": 0.90,
        "source_grounding_risk": 0.70,
        "broad_claim_risk": 0.75,
    },
)
assert_equal(
    strong_humaniser["code"],
    "ai_generated_signals",
    "component-aligned humaniser profile escalates rating even below high-review writing tier",
)

turnitin_ai_density_profile = derive_authorship_rating(
    ai_score=0.4663,
    ai_tier=Tier.AMBER,
    writing_quality_score=0.5511,
    writing_quality_tier=QualityTier.REVIEW,
    confidence=Confidence.HIGH,
    ai_components={
        "topk_pattern": 0.6721,
        "generic_assertion_risk": 0.90,
        "qualifying_text_ai_density": 0.75,
    },
    writing_components={
        "unsupported_claim_risk": 0.80,
        "source_grounding_risk": 0.70,
        "broad_claim_risk": 0.65,
    },
)
assert_equal(
    turnitin_ai_density_profile["code"],
    "ai_generated_signals",
    "dense qualifying-text AI profile escalates below the normal score threshold",
)

render_fallback = _authorship_rating_from_badge({
    "tier": "ORANGE",
    "ai_likelihood_score": 63.15,
    "writing_quality_score": 67.61,
    "writing_quality_tier": "HIGH_REVIEW",
})
assert_equal(
    render_fallback["label"],
    "AI-Generated / AI-Paraphrased Signals",
    "detect PDF renderer derives missing authorship rating from score fields",
)

render_component_fallback = _authorship_rating_from_badge({
    "tier": "ORANGE",
    "ai_likelihood_score": 60.35,
    "writing_quality_score": 62.56,
    "ai_components": {
        "topk_pattern": 89.23,
        "generic_assertion_risk": 90.0,
    },
    "writing_components": {
        "unsupported_claim_risk": 90.0,
        "source_grounding_risk": 70.0,
        "broad_claim_risk": 75.0,
    },
})
assert_equal(
    render_component_fallback["label"],
    "AI-Generated / AI-Paraphrased Signals",
    "detect PDF renderer uses aligned component evidence for humaniser profile",
)

render_density_fallback = _authorship_rating_from_badge({
    "tier": "AMBER",
    "ai_likelihood_score": 46.63,
    "writing_quality_score": 55.11,
    "ai_components": {
        "topk_pattern": 67.21,
        "generic_assertion_risk": 90.0,
        "qualifying_text_ai_density": 75.0,
    },
    "writing_components": {
        "unsupported_claim_risk": 80.0,
        "source_grounding_risk": 70.0,
        "broad_claim_risk": 65.0,
    },
})
assert_equal(
    render_density_fallback["label"],
    "AI-Generated / AI-Paraphrased Signals",
    "detect PDF renderer escalates dense qualifying-text AI profile from old score fields",
)

scored = Layer3Scorer().score(
    Layer3Input(
        predictability=0.58,
        topk_pattern=0.70,
        generic_phrase_density=0.60,
        word_count=900,
        sentence_count=45,
        paragraph_count=8,
    )
)
assert_true(scored.authorship_rating["label"], "scorer result includes authorship rating")
assert_equal(scored.authorship_rating["score"], round(scored.ai_likelihood_score * 100, 2), "rating score mirrors AI score")

qualifying_density_scored = Layer3Scorer().score(
    Layer3Input(
        predictability=0.4117,
        topk_pattern=0.6721,
        generic_phrase_density=0.0,
        burstiness_risk=0.25,
        repeated_sentence_structure_risk=0.0,
        generic_assertion_risk=0.90,
        qualifying_text_ai_density=0.75,
        balanced_hedging_risk=0.15,
        broad_claim_risk=0.65,
        lived_detail_risk=0.35,
        citation_weakness_risk=0.50,
        unsupported_claim_risk=0.80,
        source_grounding_strength=0.30,
        domain_grounding_strength=0.30,
        word_count=900,
        sentence_count=50,
        paragraph_count=8,
    )
)
assert_equal(
    qualifying_density_scored.ai_cluster_name,
    "qualifying_text_ai_density",
    "aligned long-form density triggers qualifying-text AI cluster",
)
assert_true(
    qualifying_density_scored.ai_likelihood_score >= 0.60,
    "qualifying-text AI cluster floors the AI score above likely-AI band",
)
assert_equal(
    qualifying_density_scored.authorship_rating["code"],
    "ai_generated_signals",
    "qualifying-text AI cluster rates as AI-generated signals",
)

template_ai = Layer3Scorer().score(
    Layer3Input(
        predictability=0.4659,
        topk_pattern=0.8697,
        generic_phrase_density=0.0526,
        burstiness_risk=0.25,
        repeated_sentence_structure_risk=0.65,
        generic_assertion_risk=0.90,
        balanced_hedging_risk=0.15,
        broad_claim_risk=0.85,
        lived_detail_risk=0.80,
        citation_weakness_risk=0.50,
        unsupported_claim_risk=0.90,
        source_grounding_strength=0.30,
        domain_grounding_strength=0.30,
        paragraph_progression_risk=0.85,
        paragraph_uniformity_risk=0.65,
        repeated_starter_risk=0.85,
        formulaic_conclusion_risk=0.85,
        word_count=520,
        sentence_count=31,
        paragraph_count=8,
    )
)
assert_equal(template_ai.ai_cluster_name, "template_ai_style", "education AI sample triggers template AI cluster")
assert_true(template_ai.ai_likelihood_score >= 0.58, "template AI cluster floors the AI score above possible-AI band")
assert_equal(template_ai.tier, Tier.ORANGE, "template AI sample escalates to orange tier")

scan_builder = ReportBuilder().set_meta(
    original_text=(
        "In 2024, I observed the VET workshop and wrote rough notes about the Certificate III classroom task. "
        "This shows that learning is important and can improve outcomes.\n\n"
        "Smith (2023) describes this as \"practice before confidence\". "
        "The source claims are broad and need clearer evidence."
    )
)
scan_builder._findings.append(Finding(
    tier=ReportTier.MEDIUM,
    category="predictability",
    scanner="predictability",
    title="medium_predictability",
    detail="Sentence scored 52% predictability.",
    evidence="This shows that learning is important and can improve outcomes.",
    recommendation="Rewrite with concrete classroom detail.",
    metadata={"score": 0.52, "actionability": "auto_fixable"},
    finding_id="f_test_predictability",
    sentence_id="s002",
    signal_category="predictability",
))
scan_report = scan_builder.build()
scan_json = report_to_dict(scan_report)
intel = scan_json.get("scan_intelligence", {})
assert_equal(intel.get("schema_version"), "scan_intelligence.v1", "scan intelligence schema is present")
assert_true(intel.get("document", {}).get("segments"), "scan intelligence includes highlightable document segments")
preservation = intel.get("document", {}).get("preservation_inventory", {})
anchor_texts = [a.get("text") for a in preservation.get("anchors", [])]
assert_equal(
    preservation.get("schema_version"),
    "preservation_inventory.v1",
    "scan intelligence includes preservation inventory for restructuring",
)
assert_true(
    "practice before confidence" in anchor_texts
    and "2024" in anchor_texts
    and "VET" in anchor_texts
    and "Certificate III" in anchor_texts,
    "preservation inventory captures quotes, years, acronyms, and named/domain entities",
)
assert_true(
    set(anchor_texts).issubset(set(scan_json.get("rewrite_constraints", {}).get("preserve_terms", []))),
    "rewrite constraints carry scanner preservation anchors",
)
human_contract = intel.get("human_contribution_contract", {})
assert_equal(
    human_contract.get("schema_version"),
    "human_contribution_contract.v1",
    "scan intelligence includes human contribution contract",
)
assert_true(
    human_contract.get("subsignals")
    and any(s.get("key") == "causal_reasoning" for s in human_contract.get("subsignals", [])),
    "human contribution contract decomposes human-side signals",
)
assert_true(
    human_contract.get("paragraph_levers")
    and human_contract.get("generation_readiness", {}).get("target_human_contribution") == 80,
    "human contribution contract exposes paragraph levers and target readiness",
)
assert_true(
    human_contract.get("generation_readiness", {}).get("assume_author_evidence_from_submission") is True
    and human_contract.get("assumption_policy", {}).get("mode") == "implicit_author_evidence",
    "human contribution contract assumes author evidence from submitted claims by default",
)
assert_true(
    "keep ready" in human_contract.get("generation_readiness", {}).get("user_evidence_footnote", ""),
    "human contribution contract leaves evidence-preparation footnote for users",
)
assert_equal(
    intel.get("mitigation_inputs", {}).get("human_contribution_contract"),
    human_contract,
    "mitigation inputs mirror human contribution contract for rewrite",
)
highlighted = [s for s in intel["document"]["segments"] if s.get("highlight", {}).get("enabled")]
assert_true(highlighted, "scan intelligence marks finding-linked segments for highlighting")
assert_true(
    intel.get("transformation", {}).get("core_signals"),
    "scan intelligence carries transformation core signals for mitigation",
)
core_signal = intel["transformation"]["core_signals"][0]
assert_true(
    core_signal.get("description") and core_signal.get("higher_score_means"),
    "scan intelligence core signals include reader and mitigation descriptions",
)
assert_true(
    core_signal.get("family"),
    "scan intelligence core signals include signal family metadata",
)
scan_markdown = render_markdown(scan_report)
assert_true(
    "| Core Signal | Score | What It Means |" in scan_markdown,
    "PDF markdown includes transformation signal descriptions",
)
assert_true(
    scan_json.get("highlight_segments") == intel["document"]["segments"],
    "legacy-friendly highlight segment alias mirrors scan intelligence segments",
)
assert_true(
    "calibration" in intel and "calibrated_ai_risk" in intel["calibration"],
    "scan intelligence exposes calibration layer for mitigation",
)
assert_true(
    "semantic_layer" in intel and "paraphrase_transformation_risk" in intel["semantic_layer"],
    "scan intelligence exposes semantic/paraphrase layer for mitigation",
)
generation_handoff = intel.get("generation_handoff", {})
assert_equal(
    generation_handoff.get("schema_version"),
    "generation_handoff.v1",
    "scan intelligence includes generation handoff for context regeneration",
)
assert_true(
    generation_handoff.get("source_policy", {}).get("expose_original_prose_to_generator") is False,
    "generation handoff forbids original prose exposure to generator",
)
assert_equal(
    intel.get("mitigation_inputs", {}).get("generation_handoff"),
    generation_handoff,
    "mitigation inputs mirror generation handoff for rewrite",
)
assert_equal(
    scan_json.get("generation_handoff"),
    generation_handoff,
    "scan JSON exposes top-level generation handoff",
)
integrity_layers = scan_json.get("integrity_layers", {})
assert_equal(
    integrity_layers.get("schema_version"),
    "integrity_layers.v1",
    "scan JSON exposes separated integrity layer contract",
)
assert_true(
    integrity_layers.get("policy", {}).get("grounding_is_not_ai_authorship"),
    "integrity layers state grounding is not direct AI authorship evidence",
)
assert_true(
    "ai_authorship_risk" in integrity_layers.get("layers", {}),
    "integrity layers include AI authorship risk",
)
assert_true(
    "grounding_quality_risk" in integrity_layers.get("layers", {}),
    "integrity layers include separate grounding quality risk",
)
assert_true(
    "source_grounding_risk" in integrity_layers["layers"]["ai_authorship_risk"].get("excludes", []),
    "AI authorship layer explicitly excludes grounding quality signals",
)
assert_equal(
    intel.get("integrity_layers"),
    integrity_layers,
    "scan intelligence mirrors separated integrity layer contract",
)
industry = scan_json.get("industry_baseline", {})
assert_equal(
    industry.get("schema_version"),
    "industry_baseline.v1",
    "scan JSON exposes Turnitin-style industry baseline contract",
)
assert_equal(
    intel.get("industry_baseline"),
    industry,
    "scan intelligence mirrors industry baseline contract",
)
assert_true(
    industry.get("policy", {}).get("grounding_is_not_ai_authorship")
    and industry.get("policy", {}).get("weak_grounding_can_be_human"),
    "industry baseline separates grounding quality from AI authorship",
)
ai_baseline = industry.get("layers", {}).get("ai_authorship_risk", {})
ai_component_keys = [
    row.get("key")
    for row in ai_baseline.get("positive_components", [])
    if isinstance(row, dict)
]
ai_suppressor_keys = [
    row.get("key")
    for row in ai_baseline.get("suppressors", [])
    if isinstance(row, dict)
]
assert_true(
    "token_predictability" in ai_component_keys
    and "semantic_uniformity" in ai_component_keys
    and "authorship_friction" in ai_suppressor_keys,
    "industry baseline models AI score as machine-like components minus human suppressors",
)
assert_true(
    "source_grounding_risk" in ai_baseline.get("excludes", []),
    "industry baseline excludes grounding signals from AI authorship",
)
human_baseline_keys = [
    row.get("key")
    for row in industry.get("layers", {}).get("human_contribution_signal", {}).get("components", [])
    if isinstance(row, dict)
]
assert_true(
    "causal_reasoning" in human_baseline_keys
    and "local_constraint_awareness" in human_baseline_keys
    and industry.get("rewrite_gate_objectives", {}).get("primary") == "Human Contribution >= 80",
    "industry baseline defines positive human contribution targets for gate/rewrite",
)
assert_equal(
    intel.get("mitigation_inputs", {}).get("industry_baseline"),
    industry,
    "mitigation inputs mirror industry baseline for rewrite",
)
mitigation = scan_json.get("ai_mitigation", {})
assert_equal(mitigation.get("schema_version"), "ai_mitigation.v1", "AI mitigation schema is present")
assert_equal(mitigation.get("philosophy"), "authenticity_mitigation", "AI mitigation uses authenticity strategy")
assert_equal(
    mitigation.get("integrity_layers", {}).get("schema_version"),
    "integrity_layers.v1",
    "AI mitigation handoff carries integrity layer split",
)
assert_equal(
    mitigation.get("industry_baseline"),
    industry,
    "AI mitigation handoff carries industry baseline split",
)
well_grounded_ai_plan = build_ai_mitigation_plan(
    scan_intelligence={
        "integrity_layers": {
            "schema_version": "integrity_layers.v1",
            "layers": {
                "ai_authorship_risk": {"score": 59},
                "ai_transformation_risk": {"score": 36},
                "grounding_quality_risk": {"score": 23},
                "human_contribution_signal": {"score": 64},
            },
        },
        "transformation": {
            "contribution": {
                "human_contribution_ratio": 64,
                "ai_transformation_ratio": 36,
            },
        },
    },
    ai_risk_badge={
        "ai_likelihood_score": 59.39,
        "ai_components": {
            "generic_assertion_risk": 90,
            "topk_pattern": 67.78,
            "qualifying_text_ai_density": 51.8,
        },
        "writing_components": {
            "source_grounding_risk": 25,
            "unsupported_claim_risk": 20,
            "broad_claim_risk": 35,
        },
    },
)
assert_equal(
    well_grounded_ai_plan.get("primary_mode"),
    "ai_authorship_mitigation",
    "High AI / well-grounded profile routes to AI authorship mitigation instead of author-input blocking",
)
assert_equal(
    well_grounded_ai_plan.get("baseline", {}).get("target_human_contribution_ratio"),
    80,
    "AI mitigation baseline targets Human Contribution 80 instead of incremental movement",
)
assert_equal(
    well_grounded_ai_plan.get("baseline", {}).get("target_ai_transformation_ratio"),
    20,
    "AI mitigation baseline pairs Human 80 with AI Transformation 20 target",
)
assert_equal(
    well_grounded_ai_plan.get("readiness", {}).get("requires_user_input"),
    False,
    "well-grounded AI authorship mitigation does not require author evidence before candidate generation",
)
assert_true(
    "typo injection" in mitigation.get("objective", {}).get("avoid", []),
    "AI mitigation rejects shallow evasion tactics",
)
assert_true(
    mitigation.get("target_segments"),
    "AI mitigation carries target segments from scan intelligence",
)
assert_true(
    scan_json.get("scan_intelligence", {}).get("mitigation_inputs", {}).get("ai_mitigation_plan") == mitigation,
    "scan intelligence mirrors AI mitigation handoff",
)
os.environ["DRAFTPROOF_DISABLE_EMBEDDINGS"] = "1"
semantic_text = (
    "The learner needs clear modelling before practice begins. "
    "The learner needs clear feedback before confidence improves. "
    "The learner needs clear repetition before the skill becomes stable. "
    "The trainer explains the task in a steady sequence. "
    "The trainer repeats the reason for each step. "
    "The trainer closes the activity with the same reflective structure."
)
semantic_result = SemanticShapeDetector().detect(semantic_text)
assert_equal(semantic_result.scanner, "semantic_shape", "semantic shape detector runs as a scan layer")
assert_true(
    semantic_result.raw.embedding_model_attached is False,
    "semantic shape detector exposes fallback mode when embeddings are disabled",
)
semantic_builder = ReportBuilder().set_meta(original_text=semantic_text)
semantic_builder.add_detection(semantic_result)
semantic_json = report_to_dict(semantic_builder.build())
assert_true(
    "semantic_shape" in semantic_json,
    "report JSON serializes semantic shape summary",
)
assert_true(
    semantic_json["scan_intelligence"]["semantic_layer"]["model_name"],
    "scan intelligence semantic layer exposes model provenance",
)
assert_equal(template_ai.authorship_rating["code"], "ai_generated_signals", "template AI sample rates as AI-generated signals")

sample_text_path = Path(__file__).resolve().parents[1] / "test_content.txt"
if not sample_text_path.exists():
    sample_text_path = Path(__file__).resolve().parents[2] / "test_content.txt"
sample_text = sample_text_path.read_text()
sample_builder = ReportBuilder().set_meta(original_text=sample_text)
sample_json = report_to_dict(sample_builder.build())
sample_handoff = sample_json.get("generation_handoff", {})
sample_outline = sample_handoff.get("logical_outline", [])
sample_headings = [row.get("heading") for row in sample_outline]
assert_equal(
    sample_handoff.get("schema_version"),
    "generation_handoff.v1",
    "sample scan exposes generation handoff",
)
assert_true(
    sample_headings
    and (
        "Main Body" in sample_headings
        or (
            "Introduction" in sample_headings
            and "Conclusion" in sample_headings
        )
    ),
    "sample generation handoff detects logical writing units from the current fixture structure",
)
assert_true(
    len(sample_handoff.get("section_generation_units", [])) >= 1
    and all(
        unit.get("source_span", {}).get("source_text_exposed_to_generator") is False
        for unit in sample_handoff.get("section_generation_units", [])
    ),
    "sample generation handoff creates no-source-exposure generation units",
)
intro_unit = sample_handoff.get("section_generation_units", [])[0]
assert_true(
    intro_unit.get("meaning_inventory")
    and intro_unit.get("must_preserve_anchors")
    and intro_unit.get("target_words", {}).get("min", 0) > 0,
    "sample generation unit carries meaning inventory, anchors, and word budget",
)
assert_true(
    sample_handoff.get("document_profile", {}).get("target_word_band", {}).get("min")
    <= sample_handoff.get("document_profile", {}).get("word_count", 0)
    <= sample_handoff.get("document_profile", {}).get("target_word_band", {}).get("max"),
    "sample generation handoff includes document word-count band",
)

fixture_paragraph = (
    "Inclusive learning design sits inside practical hairdressing training. "
    "Learners need clear demonstrations, steady feedback, and repeated chances to practise the same technical movement. "
    "This approach helps learners understand the relationship between sectioning, tension, elevation, and shape control. "
    "It also supports confidence because the learner can connect each step with the final haircut outcome. "
    "In a small class, the trainer can observe each learner closely and adjust the explanation when confusion appears.\n\n"
    "The learning environment should also recognise that adult learners bring different experiences and responsibilities. "
    "Some learners may need more time to connect theory with the physical action of cutting hair. "
    "Others may need clearer language, visual prompts, or a slower demonstration before they can complete the task safely. "
    "This does not reduce the complexity of the task, but it clarifies the pathway toward competency. "
    "A competent learner should not only finish a haircut; they should understand why the technique produced that result.\n\n"
    "Assessment therefore needs to look beyond a single finished outcome. "
    "It should consider how the learner plans the service, checks the client requirements, manages tools, and responds to feedback. "
    "This broader view is important because workplace readiness depends on judgement as well as technical accuracy. "
    "When learners can explain their choices, they show stronger understanding of the professional standard. "
    "The trainer can then use the evidence to decide whether the learner is ready for more independent work.\n\n"
    "Support should be flexible without removing the professional expectation. "
    "The learner still needs to meet the required standard for safety, communication, hygiene, and finished shape. "
    "Inclusive teaching is therefore not separate from industry preparation. "
    "It gives the learner a clearer route into the same workplace expectation. "
    "This makes the training fairer while keeping the competency outcome visible. "
    "The same principle applies when learners move from guided practice to more independent client-style tasks."
)
fixture_input = build_layer3_input_from_text(
    fixture_paragraph,
    predictability=0.4117,
    topk_pattern=0.6721,
    generic_phrase_density=0.0,
    citation_weakness_risk=0.50,
    source_grounding_strength=0.30,
    domain_grounding_strength=0.30,
)
assert_true(
    fixture_input.qualifying_text_ai_density is not None,
    "text-derived layer populates qualifying_text_ai_density",
)
assert_true(
    fixture_input.qualifying_text_ai_density >= 0.65,
    "long-form generic grounded-by-domain fixture produces dense qualifying-text signal",
)

fully_ai_pattern = classify_transformation(TransformationFeatures(
    ai_likelihood=0.82,
    human_anchor_score=0.12,
    rewrite_smoothness=0.55,
    source_similarity=0.0,
    surface_similarity=0.0,
    outline_to_text_expansion=0.30,
    section_style_variance=0.10,
    citation_grounding_risk=0.40,
))
assert_equal(
    fully_ai_pattern.code,
    "fully_ai_written",
    "transformation classifier identifies fully AI-written pattern",
)
assert_equal(fully_ai_pattern.is_verdict, False, "transformation classification is not a verdict")

ai_cleaned_pattern = classify_transformation(TransformationFeatures(
    ai_likelihood=0.52,
    human_anchor_score=0.62,
    rewrite_smoothness=0.75,
    source_similarity=0.0,
    surface_similarity=0.0,
    outline_to_text_expansion=0.30,
    section_style_variance=0.20,
    citation_grounding_risk=0.30,
))
assert_equal(
    ai_cleaned_pattern.code,
    "ai_cleaned_human_writing",
    "transformation classifier identifies AI-cleaned human writing pattern",
)

ai_paraphrased_pattern = classify_transformation(TransformationFeatures(
    ai_likelihood=0.45,
    human_anchor_score=0.35,
    rewrite_smoothness=0.50,
    source_similarity=0.78,
    surface_similarity=0.18,
    outline_to_text_expansion=0.30,
    section_style_variance=0.20,
    citation_grounding_risk=0.40,
))
assert_equal(
    ai_paraphrased_pattern.code,
    "ai_paraphrased",
    "transformation classifier identifies source-level AI paraphrase pattern",
)

transformation_input = Layer3Input(
    predictability=0.58,
    topk_pattern=0.76,
    generic_phrase_density=0.20,
    burstiness_risk=0.15,
    repeated_sentence_structure_risk=0.62,
    generic_assertion_risk=0.84,
    qualifying_text_ai_density=0.74,
    broad_claim_risk=0.70,
    lived_detail_risk=0.80,
    citation_weakness_risk=0.50,
    unsupported_claim_risk=0.82,
    source_grounding_strength=0.20,
    domain_grounding_strength=0.20,
    signpost_paragraph_risk=0.78,
    paragraph_progression_risk=0.78,
    word_count=700,
    sentence_count=35,
    paragraph_count=7,
)
transformation_scored = Layer3Scorer().score(transformation_input)
transformation_from_scan = classify_transformation_from_scan(
    transformation_input,
    transformation_scored,
)
assert_true(
    transformation_from_scan.code in {"fully_ai_written", "ai_expanded", "ai_cited_weakly_grounded"},
    "scan-derived transformation classifier emits a concrete transformation pattern",
)
assert_true(
    "ai_likelihood" in transformation_from_scan.features,
    "transformation classification exposes feature values",
)
assert_true(
    "adjusted_ai_risk" in transformation_from_scan.features,
    "transformation classification exposes human-anchor-adjusted AI risk",
)
assert_true(
    "calibrated_ai_risk" in transformation_from_scan.features,
    "transformation classification exposes calibrated institutional AI risk",
)
assert_true(
    "semantic_uniformity_risk" in transformation_from_scan.features,
    "transformation classification exposes semantic-shape proxy signal",
)
assert_true(
    transformation_from_scan.features["adjusted_ai_risk"] <= transformation_from_scan.features["ai_likelihood"],
    "human anchor interaction does not increase adjusted AI risk",
)
assert_true(
    transformation_from_scan.features["calibrated_ai_risk"] <= transformation_from_scan.features["adjusted_ai_risk"],
    "calibration layer does not increase adjusted AI risk",
)

anchored_ai_pattern = classify_transformation(TransformationFeatures(
    ai_likelihood=0.78,
    human_anchor_score=0.72,
    rewrite_smoothness=0.45,
    source_similarity=0.0,
    surface_similarity=0.0,
    outline_to_text_expansion=0.25,
    section_style_variance=0.10,
    citation_grounding_risk=0.20,
))
assert_true(
    anchored_ai_pattern.code != "fully_ai_written",
    "strong human anchor reduces fully-AI certainty in transformation classification",
)
assert_true(
    anchored_ai_pattern.features["reporting_suppression"] >= 0,
    "classification exposes reporting suppression for policy calibration",
)

plain_reasoning_input = build_layer3_input_from_text(
    "Technology can help students organise an answer. "
    "The issue I would check is whether the student can explain the choice afterwards. "
    "That matters because a clean response is not the same as understanding."
)
plain_generic_input = build_layer3_input_from_text(
    "Technology can help students organise an answer. "
    "This is important because it supports learning and improves outcomes. "
    "A clean response is useful for education."
)
assert_true(
    plain_reasoning_input.lived_detail_risk < plain_generic_input.lived_detail_risk,
    "bounded author reasoning trace reduces lived-detail risk without requiring external evidence",
)

print("AI rating layer tests passed")
