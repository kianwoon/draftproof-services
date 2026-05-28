from __future__ import annotations

import re
from typing import Any

from .coverage_guard import missing_required_source_term_details
from .integrity_guard import candidate_integrity_blockers, candidate_integrity_warnings
from .prose_quality import has_fragment_or_trace_sentences
from .prose_quality import catalogue_sentence_chain, robotic_sentence_chain
from .scan import scan_text
from .source_quality import scope_marker_reused_as_content, source_quality_blockers, unsupported_semantic_padding
from .text import Paragraph, source_terms
from .write import (
    Variant,
    _candidate_contract_violation,
    _compresses_list_repair,
    _hard_candidate_contract_violation,
    _has_meaningful_movement,
    _over_decomposition_review_reasons,
    _polarity_violation,
    _repeats_sentence_intent,
    _replaces_final_source_beat_with_conclusion,
    _route_quality_penalty,
)


def selection_diagnostics(
    variants: list[Variant],
    paragraph: Paragraph,
    *,
    copy_blockers: list[str] | None = None,
) -> list[dict[str, Any]]:
    source = next((variant for variant in variants if variant.source == "source_preserved"), None)
    if source is None:
        return [_variant_diagnostics(variant, None, paragraph, copy_blockers or []) for variant in variants]
    generated = [variant for variant in variants if variant.source != "source_preserved"]
    if not generated:
        return [_generation_failure_diagnostic(source, paragraph)]
    return [
        _variant_diagnostics(variant, source, paragraph, copy_blockers or [])
        for variant in generated
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
    copy_blockers: list[str],
) -> dict[str, Any]:
    candidate_scan = scan_text(variant.text)
    source_scan = scan_text(source.text) if source is not None else scan_text(paragraph.text)
    finding_drop = source_scan.scores["finding_count"] - candidate_scan.scores["finding_count"]
    risk_drop = source_scan.scores["mean_sentence_shape_risk"] - candidate_scan.scores["mean_sentence_shape_risk"]
    missing_terms = missing_required_source_term_details(variant.text, paragraph)
    integrity_blockers = candidate_integrity_blockers(variant.text)
    blockers = _blockers(variant, source, paragraph, finding_drop, risk_drop, missing_terms, integrity_blockers, copy_blockers)
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
        "candidate_text": variant.text[:1200],
        "candidate_finding_details": _finding_details(candidate_scan),
        "blockers": blockers,
        "quality_warnings": quality_warnings,
        "missing_required_terms": missing_terms[:20],
        "integrity_blockers": integrity_blockers,
        "handoff_validation": _handoff_validation(
            variant=variant,
            paragraph=paragraph,
            blockers=blockers,
            missing_terms=missing_terms,
            integrity_blockers=integrity_blockers,
            finding_drop=finding_drop,
            risk_drop=risk_drop,
        ),
        "accepted_by_selector": not blockers and source is not None and _has_meaningful_movement(variant, source, paragraph),
    }


def _generation_failure_diagnostic(source: Variant, paragraph: Paragraph) -> dict[str, Any]:
    source_scan = scan_text(source.text or paragraph.text)
    return {
        "variant_id": "writer_generation",
        "source": "writer_generation",
        "candidate_findings": int(source_scan.scores["finding_count"]),
        "source_findings": int(source_scan.scores["finding_count"]),
        "finding_drop": 0,
        "candidate_mean_risk": source_scan.scores["mean_sentence_shape_risk"],
        "source_mean_risk": source_scan.scores["mean_sentence_shape_risk"],
        "risk_drop": 0.0,
        "candidate_text": "",
        "candidate_finding_details": [],
        "blockers": ["writer_generation_failed"],
        "quality_warnings": ["writer_generation_failed_review_required"],
        "missing_required_terms": [],
        "integrity_blockers": [],
        "handoff_validation": {
            "planner_to_writer_contract": "not_validated_no_candidate",
            "writer_to_selector_candidate": "failed",
            "selector_gate": "blocked",
            "evidence": ["writer_generation_failed"],
        },
        "accepted_by_selector": False,
    }


def _handoff_validation(
    *,
    variant: Variant,
    paragraph: Paragraph,
    blockers: list[str],
    missing_terms: list[str],
    integrity_blockers: list[str],
    finding_drop: float,
    risk_drop: float,
) -> dict[str, Any]:
    source_markers = _source_scope_markers(paragraph.text)
    candidate_text = str(variant.text or "").casefold()
    missing_scope = [marker for marker in source_markers if not _scope_marker_preserved(marker, candidate_text)]
    return {
        "planner_to_writer_contract": "validated" if source_markers else "not_required",
        "writer_to_selector_candidate": "passed" if not blockers else "failed",
        "selector_gate": "eligible" if not blockers else "blocked",
        "source_scope_markers": source_markers,
        "missing_scope_markers": missing_scope,
        "scanner_movement": {
            "finding_drop": int(finding_drop),
            "risk_drop": round(float(risk_drop), 3),
        },
        "evidence": [*blockers, *missing_terms[:8], *integrity_blockers[:8]],
    }


def _source_scope_markers(text: str) -> list[str]:
    lowered = str(text or "").casefold()
    return [
        marker
        for marker in ("not always", "not only", "no longer", "rather than", "instead of", "without")
        if marker in lowered
    ]


def _scope_marker_preserved(marker: str, candidate_text: str) -> bool:
    if marker in candidate_text:
        return True
    if marker == "not only":
        return bool(re.search(r"\b(?:as well as|also|too|both)\b", candidate_text))
    if marker == "rather than":
        return bool(re.search(r"\b(?:instead of|over)\b", candidate_text))
    return False


def _blockers(
    variant: Variant,
    source: Variant | None,
    paragraph: Paragraph,
    finding_drop: float,
    risk_drop: float,
    missing_terms: list[str],
    integrity_blockers: list[str],
    copy_blockers: list[str],
) -> list[str]:
    blockers: list[str] = []
    blockers.extend(candidate_integrity_blockers(variant.text))
    if source is not None and finding_drop < 1 and risk_drop < 5.0:
        blockers.append("insufficient_scanner_movement")
    if source is not None and finding_drop == 0 and risk_drop < 0:
        blockers.append("sentence_shape_risk_regression")
    if missing_terms and not _missing_terms_are_reviewable(finding_drop, risk_drop, integrity_blockers):
        blockers.append("required_source_terms_missing")
    if _copy_blocker_violations(variant.text, copy_blockers):
        blockers.append("copy_blocked_source_phrase")
    if _repeats_sentence_intent(variant.text):
        blockers.append("repeated_sentence_intent")
    if robotic_sentence_chain(variant.text):
        blockers.append("mechanical_sentence_chain")
    if _not_always_scope_inverted(variant.text, paragraph):
        blockers.append("not_always_scope_inversion")
    if _severe_polarity_inversion(variant.text, paragraph):
        blockers.append("source_polarity_inversion")
    blockers.extend(source_quality_blockers(variant.text, paragraph))
    if has_fragment_or_trace_sentences(variant.text):
        blockers.append("fragment_or_trace_sentence")
    return blockers


def _missing_terms_are_reviewable(
    finding_drop: float,
    risk_drop: float,
    integrity_blockers: list[str],
) -> bool:
    if integrity_blockers:
        return False
    if finding_drop >= 2 and risk_drop >= -2.0:
        return True
    if finding_drop >= 1 and risk_drop >= 0.0:
        return True
    return finding_drop >= 0 and risk_drop >= 10.0


def _severe_polarity_inversion(candidate: str, paragraph: Paragraph) -> bool:
    source = str(paragraph.text or "").casefold()
    text = str(candidate or "").casefold()
    if "not only" not in source:
        return False
    if re.search(r"\b(?:also|as well as|too|both)\b", text):
        return False
    first_side, second_side = _not_only_source_sides(source)
    first_present = _side_specific_term_present(first_side, second_side, text)
    second_present = _side_specific_term_present(second_side, first_side, text)
    if first_present != second_present:
        return True
    contrast_sentence = _not_only_side_contrast_sentence(text, first_side, second_side)
    return bool(contrast_sentence and re.search(r"\b(?:instead of|rather than|more than|less than)\b", contrast_sentence))


def _not_only_source_sides(source: str) -> tuple[str, str]:
    tail = source.split("not only", 1)[-1]
    if "but also" in tail:
        return tail.split("but also", 1)
    if "also" in tail:
        return tail.split("also", 1)
    return tail, ""


def _side_term_present(side_text: str, candidate: str) -> bool:
    terms = _side_terms(side_text)
    if not terms:
        return True
    return any(term in candidate for term in terms[:5])


def _side_specific_term_present(side_text: str, other_side_text: str, candidate: str) -> bool:
    other_keys = {_side_term_key(term) for term in _side_terms(other_side_text)}
    terms = [
        term for term in _side_terms(side_text)
        if _side_term_key(term) not in other_keys
    ]
    if not terms:
        terms = _side_terms(side_text)
    if not terms:
        return True
    return any(term in candidate for term in terms[:5])


def _side_terms(side_text: str) -> list[str]:
    return [
        term.casefold()
        for term in source_terms(side_text, limit=8)
        if _contrast_side_term(term)
    ]


def _contrast_side_term(term: str) -> bool:
    value = str(term or "").casefold()
    return len(value) > 3 and value not in {
        "only", "also", "about", "that", "with", "into", "from", "people", "person", "group", "groups", "users", "user", "those", "these"
    }


def _side_term_key(term: str) -> str:
    value = str(term or "").casefold()
    return value[:-1] if len(value) > 4 and value.endswith("s") else value


def _not_only_side_contrast_sentence(candidate: str, first_side: str, second_side: str) -> str:
    for sentence in re.findall(r"[^.!?]+[.!?]?", str(candidate or "")):
        lowered = sentence.casefold()
        if _side_term_present(first_side, lowered) or _side_term_present(second_side, lowered):
            return lowered
    return ""


def _copy_blocker_violations(text: str, copy_blockers: list[str]) -> list[str]:
    lowered = str(text or "").casefold()
    rows: list[str] = []
    for blocker in copy_blockers:
        phrase = re.sub(r"\s+", " ", str(blocker or "")).strip()
        if len(phrase.split()) < 2:
            continue
        if phrase.casefold() in lowered:
            rows.append(phrase)
    return rows


def _quality_warnings(variant: Variant, paragraph: Paragraph) -> list[str]:
    warnings: list[str] = []
    if _compresses_list_repair(variant.text, paragraph):
        warnings.append("compressed_list_repair_review_required")
    if _replaces_final_source_beat_with_conclusion(variant.text, paragraph):
        warnings.append("final_source_beat_replaced_review_required")
    missing_terms = missing_required_source_term_details(variant.text, paragraph)
    if missing_terms:
        warnings.append("required_source_terms_missing_review_required")
    integrity_blockers = candidate_integrity_blockers(variant.text)
    warnings.extend(f"{blocker}_review_required" for blocker in integrity_blockers)
    warnings.extend(f"{warning}_review_required" for warning in candidate_integrity_warnings(variant.text))
    if _hard_candidate_contract_violation(variant.text, paragraph):
        warnings.append("candidate_contract_violation_review_required")
    if has_fragment_or_trace_sentences(variant.text):
        warnings.append("fragment_or_trace_sentence_review_required")
    warnings.extend(
        f"{reason}_review_required"
        for reason in _over_decomposition_review_reasons(variant.text, paragraph)
    )
    if not _hard_candidate_contract_violation(variant.text, paragraph) and _candidate_contract_violation(variant.text, paragraph):
        warnings.append("candidate_contract_warning")
    if robotic_sentence_chain(variant.text):
        warnings.append("mechanical_sentence_chain")
    if catalogue_sentence_chain(variant.text):
        warnings.append("catalogue_sentence_chain_review_required")
    if _route_quality_penalty(variant.text) >= 3.0:
        warnings.append("engineered_route_quality_review_required")
    if _polarity_violation(variant.text, paragraph):
        warnings.append("source_polarity_changed_review_required")
    if scope_marker_reused_as_content(variant.text, paragraph):
        warnings.append("source_scope_marker_reused_as_content_review_required")
    if unsupported_semantic_padding(variant.text, paragraph):
        warnings.append("unsupported_semantic_padding_review_required")
    return warnings


def _finding_details(scan: Any) -> list[dict[str, Any]]:
    return [
        {
            "sentence_id": finding.sentence_id,
            "tags": list(finding.tags),
            "text": str(finding.evidence.get("text") or "")[:260],
        }
        for finding in scan.findings[:8]
    ]




def _not_always_scope_inverted(candidate: str, paragraph: Paragraph) -> bool:
    source = str(paragraph.text or "").casefold()
    if "not always" not in source:
        return False
    scoped_terms: set[str] = set()
    for match in re.finditer(r"\bnot always\b", source):
        scoped_terms.update(_source_scope_terms(source[max(0, match.start() - 100):match.start()]))
        scoped_terms.update(_source_scope_terms(source[match.end():match.end() + 100]))
    if not scoped_terms:
        return False
    pattern = "|".join(re.escape(term) for term in sorted(scoped_terms, key=len, reverse=True))
    return bool(re.search(rf"(?:\b(?:may|might|could)\s+always|(?<!not\s)\balways)\s+(?:{pattern})\b", str(candidate or "").casefold()))


def _source_scope_terms(text: str) -> set[str]:
    return {
        term.casefold()
        for term in source_terms(text, limit=8)
        if len(term) >= 4
        and re.search(r"[a-z]", term, flags=re.I)
        and not term.casefold().endswith(("tion", "sion", "ment", "ness", "ity"))
    }
