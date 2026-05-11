"""Content-aware compiler planning."""

from __future__ import annotations

import re
from typing import Any

from .signals import PROTECTED_ANCHOR_RE, content_terms, formula_snapshot, logical_paragraphs, split_sentences


GENERIC_CONNECTOR_RE = re.compile(
    r"^(?:Furthermore|Moreover|Additionally|In conclusion|Overall|Therefore|"
    r"However|At the same time|In addition|Despite|Another important|"
    r"One of the|This|These|It is important)\b",
    re.I,
)

GENERIC_EXPANSION_RE = re.compile(
    r"\b(?:one of the biggest|another important|plays? (?:a|an) (?:important|major|significant) role|"
    r"significant impact|important feature|wide range|influential countries|modern society|"
    r"many different|various factors|global presence|cultural influence)\b",
    re.I,
)

TEMPLATE_GEOMETRY_RE = re.compile(
    r"^(?:In conclusion|Overall|At the same time|Despite this|Another important|"
    r"One of the|This shows|This highlights|This means|It is important)\b",
    re.I,
)


def classify_sentence(sentence: str, deps: Any) -> dict:
    value = str(sentence or "").strip()
    canonical = bool(
        value
        and (
            PROTECTED_ANCHOR_RE.search(value)
            or (callable(getattr(deps, "is_canonical_fact_sentence", None)) and deps.is_canonical_fact_sentence(value))
        )
    )
    if canonical:
        return {
            "class": "canonical_fact_preserve",
            "rewrite_allowed": False,
            "reasons": ["canonical_fact_or_protected_anchor"],
        }
    reasons: list[str] = []
    if GENERIC_CONNECTOR_RE.search(value):
        reasons.append("generic_connector")
    if GENERIC_EXPANSION_RE.search(value):
        reasons.append("generic_expansion")
    if TEMPLATE_GEOMETRY_RE.search(value):
        reasons.append("template_geometry")
    if len(content_terms(value)) <= 4 and len(value.split()) >= 10:
        reasons.append("low_content_density")
    if "generic_expansion" in reasons:
        label = "generic_expansion_target"
    elif "template_geometry" in reasons or "generic_connector" in reasons:
        label = "template_geometry_target"
    elif "low_content_density" in reasons:
        label = "low_value_remove_candidate"
    else:
        label = "anchor_rich_preserve" if len(content_terms(value)) >= 8 else "preserve"
    return {
        "class": label,
        "rewrite_allowed": label in {
            "generic_expansion_target",
            "template_geometry_target",
            "low_value_remove_candidate",
        },
        "reasons": reasons,
    }


def _paragraph_role(paragraph: str) -> str:
    text = str(paragraph or "").strip()
    if not text:
        return "empty"
    if re.search(r"^(?:references|works cited|bibliography)\b", text, re.I):
        return "references"
    if PROTECTED_ANCHOR_RE.search(text):
        return "fact_inventory"
    if TEMPLATE_GEOMETRY_RE.search(text):
        return "template_transition"
    if GENERIC_EXPANSION_RE.search(text):
        return "generic_expansion"
    return "argument"


def build_plan(text: str, report_dict: dict | None, deps: Any, *, max_windows: int = 6) -> dict:
    sentences = split_sentences(text)
    snapshot = formula_snapshot(text, report_dict, deps)
    geometry = {}
    if callable(getattr(deps, "geometry_risk_map", None)):
        try:
            geometry = deps.geometry_risk_map(text, report_dict, limit=max(max_windows * 3, 8))
        except Exception as exc:
            geometry = {"error": str(exc), "sentence_hotspots": []}
    geometry_by_index = {
        int(row.get("sentence_index")): row
        for row in (geometry.get("sentence_hotspots") or [])
        if isinstance(row, dict) and isinstance(row.get("sentence_index"), int)
    }
    sentence_rows: list[dict] = []
    for index, sentence in enumerate(sentences):
        classification = classify_sentence(sentence, deps)
        geometry_row = geometry_by_index.get(index) or {}
        weighted_drag = float(geometry_row.get("weighted_geometry_drag") or 0.0)
        if classification.get("class") == "generic_expansion_target":
            weighted_drag += 2.0
        if classification.get("class") == "template_geometry_target":
            weighted_drag += 1.5
        if classification.get("class") == "canonical_fact_preserve":
            weighted_drag = -1.0
        sentence_rows.append({
            "sentence_index": index,
            "sentence": sentence,
            "classification": classification.get("class"),
            "rewrite_allowed": bool(classification.get("rewrite_allowed")),
            "reasons": classification.get("reasons") or [],
            "weighted_drag": round(weighted_drag, 3),
            "geometry": geometry_row.get("drivers") if isinstance(geometry_row.get("drivers"), dict) else {},
        })
    editable = [row for row in sentence_rows if row.get("rewrite_allowed")]
    editable.sort(key=lambda row: float(row.get("weighted_drag") or 0.0), reverse=True)

    paragraph_rows: list[dict] = []
    for index, paragraph in enumerate(logical_paragraphs(text)):
        role = _paragraph_role(paragraph)
        protected = bool(role in {"references", "fact_inventory"} or PROTECTED_ANCHOR_RE.search(paragraph))
        paragraph_rows.append({
            "block_index": index,
            "role": role,
            "protected": protected,
            "remove_candidate": bool(not protected and role in {"generic_expansion", "template_transition"}),
            "word_count": len(re.findall(r"\b[\w'-]+\b", paragraph)),
            "preview": paragraph[:180],
        })
    return {
        "version": "rewrite_compiler_plan_v1",
        "mode_hint": "deterministic_operator_compiler",
        "formula_snapshot": snapshot,
        "dominant_drivers": [row.get("driver") for row in snapshot.get("driver_priority", [])[:4]],
        "sentence_risk_map": sentence_rows,
        "selected_windows": editable[:max(1, max_windows)],
        "block_risk_map": paragraph_rows,
        "canonical_fact_preserved_count": sum(
            1 for row in sentence_rows if row.get("classification") == "canonical_fact_preserve"
        ),
        "generic_blocks_targeted": sum(1 for row in paragraph_rows if row.get("role") == "generic_expansion"),
        "template_blocks_targeted": sum(1 for row in paragraph_rows if row.get("role") == "template_transition"),
    }
