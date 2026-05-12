"""Reconstruction planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import re


@dataclass(frozen=True)
class ReconstructionPlanningDeps:
    detect_protected_spans: Callable[[str], list[Any]]
    word_count_band: Callable[..., dict]
    reference_entries_from_text: Callable[[str], list[str]]
    brief_sentences: Callable[..., list[str]]
    integrity_driver_rows: Callable[..., list[dict]]
    target_segment_rows: Callable[..., list[dict]]


def build_reconstruction_meaning_brief(source_text: str, raw_json: dict | None, *, deps: ReconstructionPlanningDeps | None = None) -> dict:
    """Build a conservative meaning brief for document-level reconstruction.

    This is not an abstractive summary. It extracts author-supplied material
    already present in the submitted content and scanner output so the LLM can
    rebuild structure without inventing facts.
    """
    if deps is None:
        raise ValueError("ReconstructionPlanningDeps is required")
    raw_json = raw_json or {}
    paragraphs = [
        " ".join(p.split())
        for p in re.split(r"\n\s*\n", source_text or "")
        if p.strip()
    ]
    headings = [
        p
        for p in paragraphs
        if len(p.split()) <= 12 and not re.search(r"[.!?]$", p)
    ][:12]
    protected_spans = []
    for span in deps.detect_protected_spans(source_text or "")[:40]:
        value = (source_text or "")[span.start_char:span.end_char].strip()
        if value and value not in protected_spans:
            protected_spans.append(value)

    findings = raw_json.get("findings") or {}
    weak_zones = []
    for tier in ("critical", "high", "medium", "low"):
        for item in findings.get(tier, []) or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("category") or item.get("scanner") or "")
            evidence = item.get("evidence")
            recommendation = item.get("recommendation")
            weak_zones.append({
                "tier": tier,
                "signal": title,
                "evidence": evidence if isinstance(evidence, str) else "",
                "recommendation": recommendation if isinstance(recommendation, str) else "",
            })
            if len(weak_zones) >= 12:
                break
        if len(weak_zones) >= 12:
            break

    action_rows = []
    for action in ((raw_json.get("ai_mitigation") or {}).get("component_actions") or [])[:10]:
        if not isinstance(action, dict):
            continue
        action_rows.append({
            "component": action.get("component"),
            "score": action.get("score"),
            "action": action.get("action"),
            "user_input_needed": action.get("user_input_needed"),
        })

    scan_intelligence = raw_json.get("scan_intelligence") or {}
    generation_handoff = (
        scan_intelligence.get("generation_handoff")
        or ((scan_intelligence.get("mitigation_inputs") or {}).get("generation_handoff"))
        or raw_json.get("generation_handoff")
        or {}
    )
    semantic = (
        scan_intelligence.get("semantic_layer")
        or scan_intelligence.get("semantic_shape")
        or raw_json.get("semantic_shape")
        or {}
    )
    rewrite_constraints = (
        raw_json.get("rewrite_constraints")
        or (((raw_json.get("ai_mitigation") or {}).get("rewrite_handoff") or {}).get("rewrite_constraints"))
        or {}
    )
    preservation_inventory = (
        rewrite_constraints.get("preservation_inventory")
        or (((raw_json.get("scan_intelligence") or {}).get("mitigation_inputs") or {}).get("preservation_inventory"))
        or (((raw_json.get("scan_intelligence") or {}).get("document") or {}).get("preservation_inventory"))
        or {}
    )
    inventory_anchors = [
        anchor.get("text")
        for anchor in (preservation_inventory.get("anchors") or [])
        if isinstance(anchor, dict) and anchor.get("text")
    ]
    for anchor in inventory_anchors:
        if anchor not in protected_spans:
            protected_spans.append(anchor)
    word_band = deps.word_count_band(source_text, variance=0.25)
    paragraph_roles = []
    for idx, paragraph in enumerate(paragraphs[:10], start=1):
        lower = paragraph.lower()
        if any(marker in lower for marker in ("according to", "source", "citation", "research", "study")):
            role = "source_or_evidence_relation"
        elif any(marker in lower for marker in ("therefore", "this shows", "this means", "conclusion")):
            role = "interpretation_or_conclusion"
        elif idx == 1:
            role = "context_or_thesis"
        else:
            role = "development"
        paragraph_roles.append({
            "index": idx,
            "role": role,
            "preview": paragraph[:220],
        })
    if generation_handoff:
        handoff_units = generation_handoff.get("section_generation_units") or []
        handoff_outline = generation_handoff.get("logical_outline") or []
        handoff_refs = [
            ref.get("full_reference")
            for ref in generation_handoff.get("reference_register") or []
            if isinstance(ref, dict) and ref.get("full_reference")
        ]
        handoff_headings = [
            row.get("heading")
            for row in handoff_outline
            if isinstance(row, dict) and row.get("heading")
        ]
        handoff_protected = []
        anchor_register = generation_handoff.get("anchor_register") or {}
        for value in anchor_register.values():
            if isinstance(value, list):
                handoff_protected.extend(str(item) for item in value if item)
        handoff_roles = [
            {
                "index": idx,
                "section_id": unit.get("section_id"),
                "role": unit.get("role"),
                "heading": unit.get("heading"),
                "target_words": unit.get("target_words"),
            }
            for idx, unit in enumerate(handoff_units[:12], start=1)
            if isinstance(unit, dict)
        ]
        handoff_claims = [
            {
                "section_id": unit.get("section_id"),
                "heading": unit.get("heading"),
                "meaning_inventory": unit.get("meaning_inventory") or [],
                "citation_keys_used": unit.get("citation_keys_used") or [],
            }
            for unit in handoff_units[:12]
            if isinstance(unit, dict)
        ]
        if handoff_headings:
            headings = handoff_headings
        if handoff_refs:
            reference_entries = handoff_refs
        else:
            reference_entries = deps.reference_entries_from_text(source_text)
        if handoff_protected:
            for value in handoff_protected:
                if value not in protected_spans:
                    protected_spans.append(value)
        if handoff_roles:
            paragraph_roles = handoff_roles
    else:
        reference_entries = deps.reference_entries_from_text(source_text)
        handoff_claims = []

    return {
        "claims": handoff_claims or deps.brief_sentences(source_text, limit=36),
        "headings": headings,
        "protected_facts": protected_spans[:25],
        "reference_entries": reference_entries,
        "generation_handoff": generation_handoff,
        "preservation_inventory": preservation_inventory,
        "human_contribution_contract": (
            scan_intelligence.get("human_contribution_contract")
            or ((scan_intelligence.get("mitigation_inputs") or {}).get("human_contribution_contract"))
            or {}
        ),
        "industry_baseline": (
            scan_intelligence.get("industry_baseline")
            or ((scan_intelligence.get("mitigation_inputs") or {}).get("industry_baseline"))
            or raw_json.get("industry_baseline")
            or ((raw_json.get("ai_mitigation") or {}).get("industry_baseline"))
            or {}
        ),
        "word_count_band": word_band,
        "integrity_targets": deps.integrity_driver_rows(raw_json, limit=14),
        "target_segments": deps.target_segment_rows(raw_json, limit=18),
        "allowed_existing_additions": rewrite_constraints.get("allowed_additions") or [],
        "preserve_terms": rewrite_constraints.get("preserve_terms") or protected_spans[:25],
        "do_not_add": rewrite_constraints.get("do_not_add") or [],
        "rewrite_rule": rewrite_constraints.get("rewrite_rule"),
        "weak_grounding_zones": weak_zones,
        "mitigation_actions": action_rows,
        "paragraph_roles": paragraph_roles,
        "semantic_layer": semantic,
        "signal_inventory": scan_intelligence.get("signal_inventory") or {},
        "transformation_core": (scan_intelligence.get("transformation") or {}).get("core_signals") or {},
    }

def build_regeneration_blueprint(source_text: str, raw_json: dict | None, strategy: str, *, deps: ReconstructionPlanningDeps | None = None) -> dict:
    """Build a scanner-derived generation plan before prose generation."""
    if deps is None:
        raise ValueError("ReconstructionPlanningDeps is required")
    brief = build_reconstruction_meaning_brief(source_text, raw_json, deps=deps)
    paragraphs = [
        " ".join(p.split())
        for p in re.split(r"\n\s*\n", source_text or "")
        if p.strip()
    ]
    target_by_paragraph: dict[str, list[dict]] = {}
    for segment in brief.get("target_segments") or []:
        pid = segment.get("paragraph_id") or "p001"
        target_by_paragraph.setdefault(pid, []).append(segment)

    family_shapes = {
        "plain_direct_voice_rebuild": [
            "simple_claim",
            "plain_problem",
            "short_reason",
            "example_from_existing_context",
            "limited_point",
        ],
        "authorship_distribution_repair": [
            "plain_limit",
            "offbeat_reasoning",
            "short_restart",
            "source_or_anchor_afterthought",
            "bounded_point",
        ],
        "evidence_first_rebuild": [
            "source_or_anchor_first",
            "problem_pressure",
            "author_reasoning",
            "bounded_implication",
        ],
        "problem_observation_rebuild": [
            "problem_pressure",
            "specific_context",
            "reasoning_check",
            "limited_conclusion",
        ],
        "claim_narrowing_rebuild": [
            "narrow_claim",
            "existing_anchor",
            "qualification",
            "bounded_implication",
        ],
        "asymmetric_paragraph_route": [
            "short_context",
            "long_reasoning",
            "source_relation",
            "brief_counterpressure",
            "plain_conclusion",
        ],
        "low_smoothness_rebuild": [
            "direct_claim",
            "pause_or_limit",
            "specific_consequence",
            "plain_reasoning",
        ],
        "conservative_reconstruction": [
            "context",
            "pressure_point",
            "evidence_relation",
            "bounded_implication",
        ],
        "reasoning_dense_reconstruction": [
            "context",
            "friction",
            "evidence_relation",
            "author_reasoning",
            "implication",
        ],
        "domain_grounded_reconstruction": [
            "domain_context",
            "operational_detail",
            "judgement_or_limit",
            "bounded_implication",
        ],
    }
    route = family_shapes.get(strategy, family_shapes["asymmetric_paragraph_route"])
    paragraph_count = max(3, len(paragraphs))
    paragraph_plans = []
    for idx, paragraph in enumerate(paragraphs, start=1):
        pid = f"p{idx:03d}"
        role = route[(idx - 1) % len(route)]
        targets = target_by_paragraph.get(pid, [])
        plan = {
            "paragraph_id": pid,
            "role": role,
            "source_preview": paragraph[:180],
            "claim_source": "preserve paragraph meaning; do not preserve sentence order",
            "target_segments": targets[:4],
            "must_reduce": [
                target.get("signal") or target.get("lever")
                for target in targets[:4]
                if target.get("signal") or target.get("lever")
            ],
            "rhythm": (
                "mix one short plain sentence with one longer causal sentence"
                if idx % 2
                else "avoid balanced list cadence; use one direct limitation or contrast"
            ),
            "grounding_move": (
                "narrow broad claims to the submitted context"
                if targets
                else "add only reasoning already implied by adjacent claims"
            ),
        }
        paragraph_plans.append(plan)

    driver_keys = [
        item.get("key")
        for item in brief.get("integrity_targets") or []
        if item.get("key")
    ]
    return {
        "schema_version": "regeneration_blueprint.v1",
        "strategy_family": strategy,
        "target_human_contribution": 80,
        "word_count_band": brief.get("word_count_band"),
        "paragraph_count": paragraph_count,
        "paragraph_plans": paragraph_plans[:12],
        "global_driver_targets": driver_keys[:10],
        "industry_baseline_focus": {
            "ai_authorship_positive_components": [
                row.get("key")
                for row in (
                    ((brief.get("industry_baseline") or {}).get("layers") or {})
                    .get("ai_authorship_risk", {})
                    .get("positive_components", [])
                )[:6]
                if isinstance(row, dict) and row.get("key")
            ],
            "human_contribution_components": [
                row.get("key")
                for row in (
                    ((brief.get("industry_baseline") or {}).get("layers") or {})
                    .get("human_contribution_signal", {})
                    .get("components", [])
                )
                if isinstance(row, dict) and row.get("key")
            ],
            "authorship_suppressors": [
                row.get("key")
                for row in (
                    ((brief.get("industry_baseline") or {}).get("layers") or {})
                    .get("ai_authorship_risk", {})
                    .get("suppressors", [])
                )
                if isinstance(row, dict) and row.get("key")
            ],
            "policy": (brief.get("industry_baseline") or {}).get("policy") or {},
        },
        "hard_preserve": {
            "quotes": (brief.get("preservation_inventory") or {}).get("quotes") or [],
            "citations": (brief.get("preservation_inventory") or {}).get("citations") or [],
            "years": (brief.get("preservation_inventory") or {}).get("years") or [],
            "numbers": (brief.get("preservation_inventory") or {}).get("numbers") or [],
            "names_entities": (brief.get("preservation_inventory") or {}).get("names_entities") or [],
            "domain_terms": (brief.get("preservation_inventory") or {}).get("domain_terms") or [],
        },
        "anti_patterns_to_avoid": [
            "intro-balanced-explanation-conclusion essay shape",
            "Furthermore/Moreover/In conclusion connector chain",
            "same-length paragraphs",
            "claim followed by generic explanation",
            "smooth motivational summary",
            "broad topic statements without local narrowing",
        ],
        "candidate_family_requirements": [
            "do not use the original sentence order as scaffold",
            "assign a different reasoning job to adjacent paragraphs",
            "make at least two broad claims narrower instead of more polished",
            "preserve anchors exactly where required",
            "stay inside the word-count band",
        ],
    }
