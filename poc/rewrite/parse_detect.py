"""Parse detect JSON into rewrite-ready context.

Bridges detect pipeline output → rewrite engine input, preserving ALL fields
instead of the lossy reconstruction in the old json_to_detect_results().
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from detect.base import DetectResult, Finding


def _extract_location(fr: dict) -> dict:
    """Build location dict from finding JSON.

    Extracts sentence_id AND derives sentence_index (integer) from it
    so the rewrite planner can locate findings in the text.

    sentence_id is 1-based (s001 = first sentence), but we store
    sentence_index as 0-based for Python indexing.
    """
    loc = {}
    if fr.get("sentence_id"):
        loc["sentence_id"] = fr["sentence_id"]
        # Derive 0-based index from sentence_id like "s003" → 2
        import re
        m = re.match(r"s0*(\d+)", fr["sentence_id"])
        if m:
            loc["sentence_index"] = int(m.group(1)) - 1  # convert to 0-based
    # Also check for explicit sentence_index (0-based) in the JSON
    if fr.get("sentence_index") is not None:
        loc["sentence_index"] = fr["sentence_index"]
    if isinstance(fr.get("evidence"), dict) and fr["evidence"].get("affected_span"):
        loc["affected_span"] = fr["evidence"]["affected_span"]
    return loc


def _extract_metadata(fr: dict) -> dict:
    """Preserve all extra finding fields into metadata."""
    meta = {}
    for key in ("score", "top10_ratio", "subtype", "signal_category",
                "finding_id", "actionability", "adjustment", "raw_risk",
                "category", "scanner"):
        if key in fr and fr[key] is not None:
            meta[key] = fr[key]
    evidence = fr.get("evidence")
    if isinstance(evidence, dict):
        meta["structured_evidence"] = evidence
        metrics = evidence.get("metrics")
        if isinstance(metrics, dict):
            meta["evidence_metrics"] = metrics
    return meta


def _finding_from_json(fr: dict, tier_name: str) -> Finding:
    """Reconstruct a single Finding from a detect JSON finding dict."""
    evidence = fr.get("evidence", "")
    if isinstance(evidence, dict):
        evidence = evidence.get("summary", str(evidence))

    return Finding(
        finding_type=fr.get("title", fr.get("finding_type", "unknown")),
        risk_level=fr.get("adjusted_risk", tier_name),
        evidence_strength=fr.get("evidence_strength", "medium"),
        detail=fr.get("detail", ""),
        evidence=evidence,
        recommendation=fr.get("recommendation", ""),
        suggested_action_type=fr.get(
            "suggested_action_type",
            fr.get("actionability", "review"),
        ),
        location=_extract_location(fr),
        metadata=_extract_metadata(fr),
        signal_category=fr.get("signal_category", ""),
        actionability=fr.get("actionability", ""),
    )


def findings_from_json(data: dict) -> List[DetectResult]:
    """Convert detect JSON findings into DetectResult list.

    Preserves all fields: sentence_id, adjusted_risk, actionability,
    finding_id, signal_category, score, top10_ratio, subtype.
    """
    by_scanner: Dict[str, List[tuple]] = {}
    for tier_name in ("critical", "high", "medium", "low"):
        for f in data.get("findings", {}).get(tier_name, []):
            scanner = f.get("category", f.get("scanner", "unknown"))
            by_scanner.setdefault(scanner, []).append((tier_name, f))

    results = []
    for scanner, findings_raw in by_scanner.items():
        findings = [_finding_from_json(fr, tier) for tier, fr in findings_raw]

        risk_levels = {"critical": 4, "high": 3, "medium": 2, "low": 1, "review": 0.5}
        max_risk = max((risk_levels.get(f.risk_level, 0) for f in findings), default=0)
        overall_risk = min(max_risk / 4.0, 1.0)

        dist: Dict[str, int] = {}
        for f in findings:
            dist[f.risk_level] = dist.get(f.risk_level, 0) + 1

        # Pass scanner-level raw data so targeted rescan can diff sentences
        scanner_raw = None
        if scanner == "predictability" and "predictability" in data:
            scanner_raw = data["predictability"]

        results.append(DetectResult(
            scanner=scanner,
            overall_risk=overall_risk,
            confidence="medium",
            confidence_reason="reconstructed from detect JSON",
            risk_distribution=dist,
            findings=findings,
            policy_message="",
            raw=scanner_raw,
            detector_version="",
            model_name="",
            config_hash="",
            classification_target="",
            likelihood_score=overall_risk,
            feature_summary={},
        ))

    return results


@dataclass
class DetectJSONContext:
    """Everything the rewrite pipeline needs from a detect JSON file."""
    detect_results: List[DetectResult]
    input_text: str
    sentence_map: Dict[str, Any] = field(default_factory=dict)
    rewrite_decision: Optional[dict] = None
    rewrite_plan: Optional[dict] = None
    domain_profile: Optional[dict] = None
    overall_tier: str = "medium"
    tier_derivation: Optional[dict] = None
    raw_json: dict = field(default_factory=dict)


class DetectJSONParser:
    """Parse detect JSON into rewrite-ready DetectJSONContext."""

    @staticmethod
    def parse(json_path: str) -> DetectJSONContext:
        with open(json_path, "r") as f:
            data = json.load(f)
        return DetectJSONParser.parse_dict(data)

    @staticmethod
    def parse_dict(data: dict) -> DetectJSONContext:
        detect_results = findings_from_json(data)
        input_text = data.get("input_text", "")

        sentence_map = data.get("sentence_map", {})
        if isinstance(sentence_map, list):
            sentence_map = {s.get("sentence_id", f"s{i}"): s for i, s in enumerate(sentence_map)}

        return DetectJSONContext(
            detect_results=detect_results,
            input_text=input_text,
            sentence_map=sentence_map,
            rewrite_decision=data.get("rewrite_decision"),
            rewrite_plan=data.get("rewrite_plan"),
            domain_profile=data.get("domain_profile"),
            overall_tier=data.get("overall_tier", "medium"),
            tier_derivation=data.get("tier_derivation"),
            raw_json=data,
        )


__all__ = [
    "DetectJSONContext",
    "DetectJSONParser",
    "findings_from_json",
]
