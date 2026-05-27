from __future__ import annotations

from typing import Any

from .coverage_guard import missing_required_source_term_details
from .integrity_guard import candidate_integrity_blockers
from .prose_quality import has_fragment_or_trace_sentences
from .prose_quality import robotic_sentence_chain
from .scan import scan_text
from .text import Paragraph
from .write import (
    Variant,
    _candidate_contract_violation,
    _compresses_list_repair,
    _hard_candidate_contract_violation,
    _has_meaningful_movement,
    _polarity_violation,
    _replaces_final_source_beat_with_conclusion,
)


def selection_diagnostics(variants: list[Variant], paragraph: Paragraph) -> list[dict[str, Any]]:
    source = next((variant for variant in variants if variant.source == "source_preserved"), None)
    if source is None:
        return [_variant_diagnostics(variant, None, paragraph) for variant in variants]
    return [
        _variant_diagnostics(variant, source, paragraph)
        for variant in variants
        if variant.source != "source_preserved"
    ]


def rejected_variant_feedback(variants: list[Variant], paragraph: Paragraph) -> list[dict[str, Any]]:
    return [
        row for row in selection_diagnostics(variants, paragraph)
        if row.get("blockers")
    ]


def _variant_diagnostics(
    variant: Variant,
    source: Variant | None,
    paragraph: Paragraph,
) -> dict[str, Any]:
    candidate_scan = scan_text(variant.text)
    source_scan = scan_text(source.text) if source is not None else scan_text(paragraph.text)
    finding_drop = source_scan.scores["finding_count"] - candidate_scan.scores["finding_count"]
    risk_drop = source_scan.scores["mean_sentence_shape_risk"] - candidate_scan.scores["mean_sentence_shape_risk"]
    missing_terms = missing_required_source_term_details(variant.text, paragraph)
    integrity_blockers = candidate_integrity_blockers(variant.text)
    blockers = _blockers(variant, source, paragraph, finding_drop, risk_drop, missing_terms, integrity_blockers)
    quality_warnings = _quality_warnings(variant, paragraph)
    return {
        "variant_id": variant.id,
        "source": variant.source,
        "candidate_findings": int(candidate_scan.scores["finding_count"]),
        "source_findings": int(source_scan.scores["finding_count"]),
        "finding_drop": int(finding_drop),
        "candidate_mean_risk": candidate_scan.scores["mean_sentence_shape_risk"],
        "source_mean_risk": source_scan.scores["mean_sentence_shape_risk"],
        "risk_drop": round(float(risk_drop), 3),
        "blockers": blockers,
        "quality_warnings": quality_warnings,
        "missing_required_terms": missing_terms[:20],
        "integrity_blockers": integrity_blockers,
        "accepted_by_selector": not blockers and source is not None and _has_meaningful_movement(variant, source, paragraph),
    }


def _blockers(
    variant: Variant,
    source: Variant | None,
    paragraph: Paragraph,
    finding_drop: float,
    risk_drop: float,
    missing_terms: list[str],
    integrity_blockers: list[str],
) -> list[str]:
    blockers: list[str] = []
    if _compresses_list_repair(variant.text, paragraph):
        blockers.append("compressed_list_repair")
    if _replaces_final_source_beat_with_conclusion(variant.text, paragraph):
        blockers.append("final_source_beat_replaced")
    if _polarity_violation(variant.text, paragraph):
        blockers.append("source_polarity_changed")
    if missing_terms:
        blockers.append("required_source_terms_missing")
    blockers.extend(integrity_blockers)
    if _hard_candidate_contract_violation(variant.text, paragraph):
        blockers.append("candidate_contract_violation")
    if has_fragment_or_trace_sentences(variant.text):
        blockers.append("fragment_or_trace_sentence")
    if source is not None and finding_drop < 1 and risk_drop < 8.0:
        blockers.append("insufficient_scanner_movement")
    if source is not None and finding_drop == 0 and risk_drop < 0:
        blockers.append("sentence_shape_risk_regression")
    return blockers


def _quality_warnings(variant: Variant, paragraph: Paragraph) -> list[str]:
    warnings: list[str] = []
    if not _hard_candidate_contract_violation(variant.text, paragraph) and _candidate_contract_violation(variant.text, paragraph):
        warnings.append("candidate_contract_warning")
    if robotic_sentence_chain(variant.text):
        warnings.append("mechanical_sentence_chain")
    return warnings
