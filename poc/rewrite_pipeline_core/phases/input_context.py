"""Input context resolution for the rewrite pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rewrite.parse_detect import DetectJSONContext


@dataclass(frozen=True)
class RewriteInputContextDeps:
    detect_json_parse_dict: Callable[[dict], DetectJSONContext]
    detect_json_parse: Callable[[str], DetectJSONContext]
    run_detect: Callable[..., dict]
    detect_result_factory: Callable[..., Any]
    detect_finding_factory: Callable[..., Any]


def resolve_rewrite_input_context(
    *,
    json_path: str | None = None,
    text: str | None = None,
    detect_json: dict | None = None,
    output_dir: str | None = None,
    verbose: bool = False,
    deps: RewriteInputContextDeps,
) -> tuple[DetectJSONContext | None, str | None]:
    ctx: DetectJSONContext | None = None

    if json_path or detect_json:
        if detect_json:
            ctx = deps.detect_json_parse_dict(detect_json)
        else:
            ctx = deps.detect_json_parse(json_path)
        text = ctx.input_text
    elif text:
        # Run detect first, then normalize the report findings into DetectResult.
        detect_result = deps.run_detect(text, output_dir or "test_output", verbose=verbose)
        report = detect_result["report"]

        by_scanner = {}
        for tier_findings in report.findings_by_tier.values():
            for f in tier_findings:
                risk_level = getattr(f, "risk_level", None)
                if not risk_level:
                    risk_level = getattr(f, "adjusted_risk", None) or getattr(f, "raw_risk", None)
                if not risk_level:
                    tier_value = getattr(getattr(f, "tier", None), "value", None)
                    risk_level = str(tier_value or "review").lower()
                metadata = dict(getattr(f, "metadata", None) or {})
                metadata.setdefault("scanner", getattr(f, "scanner", ""))
                metadata.setdefault("category", getattr(f, "category", ""))
                if getattr(f, "finding_id", ""):
                    metadata.setdefault("finding_id", getattr(f, "finding_id", ""))
                location = {}
                if getattr(f, "sentence_id", ""):
                    location["sentence_id"] = getattr(f, "sentence_id", "")
                normalized_finding = deps.detect_finding_factory(
                    finding_type=str(getattr(f, "finding_type", None) or getattr(f, "title", "") or ""),
                    risk_level=str(risk_level or "review").lower(),
                    evidence_strength=str(metadata.get("evidence_strength") or "moderate"),
                    detail=str(getattr(f, "detail", "") or ""),
                    evidence=str(getattr(f, "evidence", "") or ""),
                    recommendation=str(getattr(f, "recommendation", "") or ""),
                    suggested_action_type=str(metadata.get("suggested_action_type") or "review"),
                    location=location,
                    metadata=metadata,
                    signal_category=str(getattr(f, "signal_category", "") or ""),
                    actionability=str(metadata.get("actionability") or ""),
                )
                by_scanner.setdefault(
                    normalized_finding.metadata.get("scanner") or getattr(f, "scanner", ""),
                    [],
                ).append(normalized_finding)

        detect_results = []
        for scanner, findings in by_scanner.items():
            # Preserve raw data from report JSON for scanners that have it.
            scanner_raw = None
            if scanner == "predictability":
                pred = detect_json.get("predictability", {}) if isinstance(detect_json, dict) else {}
                # Use all_sentences (full text + scores) if available,
                # otherwise fall back to the predictability block.
                all_sents = pred.get("all_sentences")
                if all_sents:
                    scanner_raw = {"sentences": all_sents}
                else:
                    scanner_raw = pred if pred else None
            detect_results.append(deps.detect_result_factory(
                scanner=scanner,
                overall_risk=0.5,
                confidence="medium",
                confidence_reason="from detect pipeline",
                risk_distribution={},
                findings=findings,
                policy_message="",
                raw=scanner_raw,
            ))
        ctx = DetectJSONContext(
            detect_results=detect_results,
            input_text=text,
        )

    return ctx, text
