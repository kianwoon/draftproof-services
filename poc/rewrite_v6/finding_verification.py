"""P3 — finding-aware verification (telemetry, the noise-robust measurer).

The document-level ``final_risk`` is too high-variance (gpt-oss writer swings tens of points run to
run) to prove whether a targeted fix landed. This module measures the thing that actually matters and
is stable: of the grounding/reasoning issues the enhanced scan flagged (the P1 targets), how many are
GONE from a fresh re-scan of the rewritten draft.

TELEMETRY ONLY — it never re-loops, never rejects, never mutates the text. It appends one summary row
to ``pass_trace`` and, when a targeted issue persists on a paragraph the writer changed, a review flag
so the author knows it still needs their own specifics. Claim-graph (P2) verdicts need the Modal NLI
endpoint to re-check, so they are counted as TARGETED but marked not-locally-verifiable rather than
guessed.

Kill switch: DRAFTPROOF_V6_FINDING_VERIFICATION=0.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Callable

from .report_contracts import paragraph_diagnosis

# Same trustworthy allowlist P1 and the display composer use (keep in sync).
_ISSUE_BY_TITLE = {"low_specificity": "grounding", "semantic_drift": "reasoning"}
_GROUNDING_REASONING = frozenset({"grounding", "reasoning"})


def finding_verification_enabled() -> bool:
    return os.environ.get("DRAFTPROOF_V6_FINDING_VERIFICATION", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _all_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    f = report.get("findings")
    if isinstance(f, list):
        return [x for x in f if isinstance(x, dict)]
    if isinstance(f, dict):
        return [x for v in f.values() if isinstance(v, list) for x in v if isinstance(x, dict)]
    return []


def _targeted_issues(paragraphs: list[Any]) -> dict[str, set[str]]:
    """{paragraph_id: {issue}} the scan asked the writer to fix, from the bound diagnosis.

    grounding/reasoning come from P1's ``flagged_sentences``; ``claim`` from P2's ``unsupported_claims``.
    """
    out: dict[str, set[str]] = {}
    for paragraph in paragraphs:
        pid = str(getattr(paragraph, "id", "") or "")
        diagnosis = paragraph_diagnosis(pid) or {}
        issues: set[str] = set()
        for row in diagnosis.get("flagged_sentences") or []:
            if isinstance(row, dict) and row.get("issue") in _GROUNDING_REASONING:
                issues.add(str(row["issue"]))
        if diagnosis.get("unsupported_claims"):
            issues.add("claim")
        if issues:
            out[pid] = issues
    return out


def _final_issues(final_report: dict[str, Any]) -> dict[str, set[str]]:
    """{paragraph_id: {grounding|reasoning}} still present in a re-scan of the rewritten draft.

    Findings carry ``sentence_id`` only; join to paragraph via the report's highlight_segments (same
    technique as P1). Paragraph ids align by position with the originals (paragraph count is an
    invariant of the pipeline), so a per-paragraph comparison is sound for telemetry.
    """
    sid_to_pid: dict[str, str] = {}
    for seg in final_report.get("highlight_segments") or []:
        if isinstance(seg, dict):
            sid = str(seg.get("sentence_id") or "").strip()
            pid = str(seg.get("paragraph_id") or "").strip()
            if sid and pid:
                sid_to_pid.setdefault(sid, pid)
    out: dict[str, set[str]] = {}
    for finding in _all_findings(final_report):
        issue = _ISSUE_BY_TITLE.get(str(finding.get("title") or "").strip())
        if not issue:
            continue
        pid = sid_to_pid.get(str(finding.get("sentence_id") or "").strip())
        if pid:
            out.setdefault(pid, set()).add(issue)
    return out


def verify_finding_resolution(paragraphs: list[Any], final_report: dict[str, Any]) -> dict[str, Any]:
    """Compare targeted grounding/reasoning issues against a re-scan of the rewritten draft.

    Returns a telemetry summary. Document-level counts (targeted vs still-present) are the robust
    metric; per-paragraph ``persisted`` drives review flags. Claim targets are reported but not
    locally re-verifiable (need Modal NLI)."""
    targeted = _targeted_issues(paragraphs)
    final = _final_issues(final_report)

    gr_targeted = {pid: issues & _GROUNDING_REASONING for pid, issues in targeted.items()}
    gr_targeted = {pid: issues for pid, issues in gr_targeted.items() if issues}

    targeted_count = sum(len(issues) for issues in gr_targeted.values())
    persisted: list[dict[str, str]] = []
    persisted_count = 0
    for pid, issues in gr_targeted.items():
        still = issues & final.get(pid, set())
        for issue in sorted(still):
            persisted.append({"paragraph_id": pid, "issue": issue})
            persisted_count += 1
    resolved_count = max(0, targeted_count - persisted_count)
    claim_targeted = sum(1 for issues in targeted.values() if "claim" in issues)

    return {
        "targeted_grounding_reasoning": targeted_count,
        "resolved": resolved_count,
        "persisted": persisted_count,
        "resolution_rate": round(resolved_count / targeted_count, 4) if targeted_count else None,
        "persisted_detail": persisted[:12],
        "claim_targets_not_locally_verifiable": claim_targeted,
    }


def persisted_review_flag(persisted_issue: str) -> dict[str, str]:
    """A review flag for a grounding/reasoning issue that survived the rewrite — annotate, never reject."""
    if persisted_issue == "reasoning":
        return {"added": "a reasoning gap may remain",
                "why": "The scan still reads a logic jump here — make the connecting reasoning explicit in your own words."}
    return {"added": "a grounding gap may remain",
            "why": "This still reads as generic — add a concrete anchor only you would know (an example, a specific, a source)."}


def apply_finding_verification(doc, scan, scan_report_fn: Callable[[str], dict[str, Any]]):
    """Append a P3 finding-resolution telemetry row (+ per-paragraph persisted review flags) to the
    document's pass_trace. Telemetry only: text/final_scan untouched. ``scan_report_fn`` produces a
    composite scan report for the given text (injected to avoid a circular import). Any failure returns
    doc unchanged; the kill switch disables it entirely."""
    if not finding_verification_enabled():
        return doc
    try:
        final_report = scan_report_fn(doc.rewritten_text)
        summary = verify_finding_resolution(list(scan.paragraphs), final_report)
        row = {"selected_source": "finding_verification", "status": "checked", **summary}
        flags = [persisted_review_flag(item["issue"]) for item in summary.get("persisted_detail", [])]
        if flags:
            row["review_flags"] = flags[:12]
        return replace(doc, pass_trace=list(doc.pass_trace) + [row])
    except Exception:
        return doc


__all__ = [
    "finding_verification_enabled",
    "verify_finding_resolution",
    "persisted_review_flag",
    "apply_finding_verification",
]
