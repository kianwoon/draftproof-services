"""Block-level formula-driver triage map."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class FormulaBlockDriverMapDeps:
    logical_paragraphs: Callable[[str], list[str]]
    text_word_count: Callable[[str], int]
    split_sentences: Callable[[str], list[str]]
    protected_number_set: Callable[[str], set]
    protected_code_anchor_set: Callable[[str], set]
    is_heading_like_paragraph: Callable[[str], bool]
    turnitin_like_ai_profile: Callable[[dict | None], dict]
    formula_portfolio_plan: Callable[[dict | None, dict | None], dict]


def formula_block_driver_map(source_text: str, report_dict: dict | None, *, deps: FormulaBlockDriverMapDeps) -> dict:
    """Estimate block-level formula drag for convergence planning."""
    paragraphs = deps.logical_paragraphs(source_text)
    profile = deps.turnitin_like_ai_profile(report_dict)
    plan = deps.formula_portfolio_plan(report_dict, report_dict)
    weighted = profile.get("weighted_components") if isinstance(profile.get("weighted_components"), dict) else {}
    dominant_drivers = [
        str(row.get("driver"))
        for row in (plan.get("driver_priorities") or [])[:5]
        if isinstance(row, dict) and row.get("driver")
    ]
    generic_re = re.compile(
        r"\b(?:important|significant|various|many|different|modern|today|society|"
        r"culture|technology|system|people|community|global|"
        r"impact|influence|development|opportunity|challenge|supports?|helps?|"
        r"plays? a role|continues? to|it is clear|this shows|this means)\b",
        re.I,
    )
    human_anchor_re = re.compile(
        r"\b(?:I|my|we|our|when|during|after|before|in practice|for me|"
        r"what I|what we|I noticed|I would|the issue is|this depends|"
        r"checked|feedback|mistake|draft|practice|workshop|observed|tested)\b",
        re.I,
    )
    connector_re = re.compile(
        r"\b(?:furthermore|moreover|additionally|in conclusion|overall|"
        r"therefore|as a result|this highlights|this demonstrates)\b",
        re.I,
    )
    reference_section = False
    blocks: list[dict] = []
    total_words = max(1, sum(deps.text_word_count(paragraph) for paragraph in paragraphs))
    for index, paragraph in enumerate(paragraphs):
        stripped = str(paragraph or "").strip()
        if re.match(r"^\s*references?\s*$", stripped, flags=re.I):
            reference_section = True
        words = deps.text_word_count(stripped)
        sentences = deps.split_sentences(stripped)
        sentence_count = max(1, len(sentences))
        generic_hits = len(generic_re.findall(stripped))
        anchor_hits = len(human_anchor_re.findall(stripped))
        connector_hits = len(connector_re.findall(stripped))
        protected_numbers = sorted(deps.protected_number_set(stripped))
        protected_codes = sorted(deps.protected_code_anchor_set(stripped))
        url_count = len(re.findall(r"https?://|www\.", stripped, flags=re.I))
        proper_like = len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", stripped))
        heading_like = deps.is_heading_like_paragraph(stripped)
        protected = bool(reference_section or heading_like or protected_numbers or protected_codes or url_count)
        unique_core_claim = bool(
            protected
            or re.search(r"\b(?:because|therefore|caused|led to|resulted|depends on|specific|particular)\b", stripped, re.I)
            or (
                proper_like >= 4
                and re.search(r"\b(?:case|example|named|known as|called|located|founded|declared)\b", stripped, re.I)
            )
        )
        generic_density = generic_hits / max(1, words / 20.0)
        anchor_density = anchor_hits / sentence_count
        human_anchor_deficit = max(0.0, 1.0 - anchor_density)
        template_density = connector_hits / sentence_count
        length_share = words / total_words
        formula_drag = (
            float(weighted.get("ai_likelihood") or 0.0) * 0.40
            + float(weighted.get("semantic_uniformity") or 0.0) * 0.22
            + float(weighted.get("rewrite_smoothness") or 0.0) * 0.18
            + float(weighted.get("patchwork_expansion") or 0.0) * 0.14
            + float(weighted.get("topk_calibrated_risk") or 0.0) * 0.06
        )
        weighted_drag = (
            formula_drag * min(1.0, length_share * 4.0)
            + min(12.0, generic_density * 4.0)
            + min(8.0, max(0.0, 1.0 - anchor_density) * 3.0)
            + min(5.0, template_density * 3.0)
        )
        suppression_gain_potential = min(
            8.0,
            human_anchor_deficit * 4.0
            + min(3.0, generic_density * 0.8)
            + min(2.0, length_share * 8.0),
        )
        if protected:
            remove_value_loss_risk = "blocked"
        elif unique_core_claim:
            remove_value_loss_risk = "high"
        elif words <= 120 and generic_density >= 1.0 and anchor_hits == 0:
            remove_value_loss_risk = "low"
        else:
            remove_value_loss_risk = "medium"
        if not stripped:
            action = "preserve"
            remove_safety = "empty_block"
            recommended_portfolio_action = "preserve"
        elif protected:
            action = "preserve"
            remove_safety = "protected_anchor_or_reference"
            recommended_portfolio_action = "preserve"
        elif unique_core_claim and weighted_drag >= 12.0:
            action = "rebuild"
            remove_safety = "unique_core_claim_requires_replacement"
            recommended_portfolio_action = "texture_rebuild"
        elif weighted_drag >= 12.0 and words <= 120 and remove_value_loss_risk != "high":
            action = "remove_candidate"
            remove_safety = "low_anchor_high_drag_no_protected_anchors"
            recommended_portfolio_action = "remove_candidate" if remove_value_loss_risk == "low" else "compress"
        elif weighted_drag >= 10.0:
            action = "compress"
            remove_safety = "compress_before_remove"
            recommended_portfolio_action = "anchor_amplify" if human_anchor_deficit >= 0.75 else "compress"
        else:
            action = "preserve"
            remove_safety = "low_estimated_drag"
            recommended_portfolio_action = "anchor_amplify" if human_anchor_deficit >= 0.75 and weighted_drag >= 6.0 else "preserve"
        blocks.append({
            "block_index": index,
            "word_count": words,
            "sentence_count": sentence_count,
            "action": action,
            "weighted_drag": round(weighted_drag, 3),
            "generic_hits": generic_hits,
            "generic_density": round(generic_density, 3),
            "human_anchor_hits": anchor_hits,
            "human_anchor_density": round(anchor_density, 3),
            "human_anchor_deficit": round(human_anchor_deficit, 3),
            "lived_detail_gap": round(max(0.0, 1.0 - anchor_density) * 100.0, 3),
            "suppression_gain_potential": round(suppression_gain_potential, 3),
            "template_connector_hits": connector_hits,
            "protected": protected,
            "protected_numbers": protected_numbers[:8],
            "protected_code_anchors": protected_codes[:8],
            "url_count": url_count,
            "heading_like": heading_like,
            "reference_section": reference_section,
            "unique_core_claim": unique_core_claim,
            "remove_safety": remove_safety,
            "remove_value_loss_risk": remove_value_loss_risk,
            "recommended_portfolio_action": recommended_portfolio_action,
            "dominant_formula_drivers": dominant_drivers,
            "preview": stripped[:220],
        })
    top_blocks = sorted(
        blocks,
        key=lambda row: float(row.get("weighted_drag") or 0.0),
        reverse=True,
    )[:8]
    return {
        "version": "formula_block_driver_map_v1",
        "block_count": len(blocks),
        "formula_score": profile.get("score"),
        "target_score": profile.get("target_score"),
        "remaining_gap": profile.get("target_gap"),
        "dominant_formula_drivers": dominant_drivers,
        "blocks": blocks,
        "top_blocks": top_blocks,
    }
