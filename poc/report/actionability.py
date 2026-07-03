"""Finding actionability classification.

Extracted from report.py. Module-level function so both ReportBuilder.build()
and report_to_dict() can call it without duplication.
"""


def determine_actionability(f, all_findings: list = None) -> str:
    """Classify finding into actionability bucket."""
    detected_actionability = ""
    if f.metadata and isinstance(f.metadata, dict):
        detected_actionability = f.metadata.get("actionability", "")
    if detected_actionability:
        aliases = {
            "auto_rewrite_candidate": "auto_fixable",
            "optional_structure_review": "optional_structure_review",
            "citation_repair": "citation_repair",
            "review_only": "review_only",
            "no_action": "no_action",
            "manual_required": "manual_required",
            "auto_fixable": "auto_fixable",
        }
        if detected_actionability in aliases:
            mapped = aliases[detected_actionability]
            if mapped == "auto_fixable" and f.adjusted_risk.lower() != "medium":
                return "review_only"
            return mapped

    adj = f.adjusted_risk.lower()
    title = f.title
    # No-action: low-risk signals
    if "low_ai_generation" in title or "minimal_ai_generation" in title or "low_specificity_likelihood" in title:
        return "no_action"
    if adj in ("low", "review", "clean"):
        return "review_only"
    # Protected: quotes, citations
    if f.category in ("citation", "integrity") or "quote" in title:
        # Citation findings are citation_repair (suggest but don't rewrite)
        if f.category == "citation":
            return "citation_repair"
        return "manual_required"
    # High predictability is review guidance, not an automatic rewrite target.
    # The rewrite pipeline is intentionally limited to medium findings.
    if "high_predictability" in title or "high_topk_predictability" in title:
        return "review_only"
    # DeBERTa authoritative findings: the learned classifier flagged this sentence as
    # high-confidence AI. It's an auto-fix target (revoice the sentence) under the
    # authoritative flag — replaces the perplexity 'medium_predictability' edit targets.
    if title == "high_confidence_ai_sentence" and f.scanner == "deberta":
        return "auto_fixable"
    # Document-level low_specificity: auto-fixable only if NOT already downgraded
    # If AI likelihood is low and domain grounding is strong, specificity is review-level
    if title == "low_specificity":
        return "manual_required"
    # Medium predictability: only auto-fix if co-located with document-level signals
    # (low_specificity, uncited_claim) that confirm AI origin. Medium predictability
    # alone is common in clear human writing — auto-rewriting it makes things worse.
    if "medium_predictability" in title:
        if all_findings:
            _DOC_SIGNALS = {"low_specificity", "uncited_claim"}
            doc_paired = any(
                af.title in _DOC_SIGNALS or
                (af.metadata or {}).get("signal_category", "") in _DOC_SIGNALS
                for af in all_findings
                if af is not f
            )
            if doc_paired:
                return "auto_fixable"
        return "review_only"
    # uniform_paragraph_structure with downgrade adjustment: optional review, not rewrite
    if title == "uniform_paragraph_structure":
        has_downgrade = (
            f.metadata
            and isinstance(f.metadata, dict)
            and f.metadata.get("adjustment", {}).get("filter") == "UniformStructureDowngrade"
        )
        if has_downgrade:
            return "optional_structure_review"
    # Document-level AI summary/structure signals are not safe automatic
    # sentence rewrites. They require source context, concrete examples, or
    # structural revision guidance.
    if title in {
        "moderate_ai_generation_likelihood",
        "elevated_ai_generation_likelihood",
        "uniform_paragraph_structure",
        "low_burstiness",
        "source_grounding",
        "polished_but_ungrounded",
    }:
        return "manual_required"
    # Other high/medium AI-generation signals: auto-fixable
    if adj == "medium" and f.scanner == "ai_generation":
        return "auto_fixable"
    return "review_only"
