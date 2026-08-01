"""Helpers for comparing residual-pass candidates against the TRUE original document.

Split out of direct_rewrite.py (already at the 1500-LOC repo cap) so the residual-fix
review-carding logic has a home that isn't fighting the line budget.

Incident context (job 5bacaeb3): `_apply_residual_fix` used to compare a pass-2 candidate
against the PASS-1 rewritten paragraph, not the true original. Anything pass-1 already
fabricated (invented stats, invented first-person framing) was silently treated as the
baseline and never carded. These helpers give `_apply_residual_fix` the true-original
paragraph to diff against, and an unconditional card for any paragraph that has no
true-original counterpart at all (a net-new/added paragraph).
"""

from __future__ import annotations

import difflib
import logging
import os
from typing import Any

_HONOR_ENV = "DRAFTPROOF_V6_HONOR_REWRITE_DECISION"
_logger = logging.getLogger(__name__)

# Below this best-match ratio against every true-original paragraph, a rewritten paragraph is
# treated as having no real counterpart (net-new / inserted) rather than a heavy rewrite of an
# existing one. Content-agnostic: pure text-similarity, no topic/value lists. Env-tunable, same
# on/off vocabulary as the other v6 switches in this package.
def _added_paragraph_match_threshold() -> float:
    raw = os.environ.get("DRAFTPROOF_V6_ADDED_PARAGRAPH_MATCH_THRESHOLD", "").strip()
    try:
        value = float(raw)
        if 0.0 <= value <= 1.0:
            return value
    except (TypeError, ValueError):
        pass
    return 0.35


def original_paragraph_texts(doc: Any) -> list[str]:
    """The TRUE original paragraph texts (pre pass-1), in document order.

    `doc.initial_scan.paragraphs` is the scan of the author's real submitted text, before any
    writer pass touched it -- the only safe baseline for fabrication/added-content checks.
    """
    paragraphs = getattr(getattr(doc, "initial_scan", None), "paragraphs", None) or []
    return [str(getattr(p, "text", "") or "") for p in paragraphs]


def true_original_for_index(originals: list[str], index: int) -> str | None:
    """The true-original paragraph text at `index`, or None if this index has no counterpart
    (i.e. it is a net-new paragraph the writer added that doesn't exist in the original doc)."""
    if 0 <= index < len(originals):
        return originals[index]
    return None


def true_original_paragraph(paragraph: Any, true_original_text: str) -> Any:
    """Rebuild `paragraph` (a Paragraph) with its text swapped for the TRUE original text, so
    fabrication/review-flag heuristics diff the candidate against what the author actually wrote,
    not against whatever pass-1 already produced."""
    from .text import Paragraph

    return Paragraph(
        id=paragraph.id, index=paragraph.index, text=true_original_text, sentences=paragraph.sentences,
    )


def shape_mismatch_added_paragraph_traces(originals: list[str], paragraphs: list[Any]) -> list[dict[str, Any]]:
    """When the rewritten doc has MORE paragraphs than the true original (residual-checker shape
    mismatch, actual > expected), find the paragraphs with no reasonable original counterpart and
    return a pass_trace row carding each one unconditionally. Pure text-similarity match (difflib) so
    it stays content-agnostic -- no topic/value lists. Used when the per-index loop never runs
    because the whole-document paragraph count already diverged (2026-08-01 incident, job 5bacaeb3:
    an added leading paragraph shifted every index, so the shape-mismatch guard skipped the entire
    residual pass and zero cards were emitted)."""
    if len(paragraphs) <= len(originals):
        return []
    traces: list[dict[str, Any]] = []
    for index, paragraph in enumerate(paragraphs):
        text = str(getattr(paragraph, "text", "") or "")
        best_ratio = max(
            (difflib.SequenceMatcher(None, text, original).ratio() for original in originals),
            default=0.0,
        )
        if best_ratio < _added_paragraph_match_threshold():
            paragraph_id = str(getattr(paragraph, "id", index))
            traces.append({
                "pass_index": index,
                "target_paragraph_id": paragraph_id,
                "selected_source": "residual_checker_added_paragraph",
                "status": "accepted",
                "author_review_items": [added_paragraph_review_card(index, paragraph_id)],
            })
    return traces


def added_paragraph_review_card(index: int, paragraph_id: str) -> dict[str, Any]:
    """Unconditional review card for a paragraph that exists in the rewrite but has no
    counterpart in the true original -- e.g. a net-new leading/inserted paragraph. Must be
    surfaced even if downstream fabrication heuristics don't independently flag it, because
    the whole paragraph is DraftProof-authored content with nothing from the author to compare
    it to."""
    return {
        "added": "a new paragraph was added",
        "why": (
            "DraftProof added this paragraph -- it was not derived from any paragraph in your "
            "original document. Review it and replace it with your own real content, or remove "
            "it, before submitting."
        ),
        "paragraph_id": paragraph_id,
        "paragraph_index": index,
        "provenance": "added_paragraph",
    }


def honor_rewrite_decision_enabled() -> bool:
    """Kill switch, default ON. Same on/off vocabulary as the other v6 env switches in this
    package (unset/empty -> default; "0"/"false"/"no"/"off" -> disabled)."""
    raw = os.environ.get(_HONOR_ENV)
    if raw is None or raw.strip() == "":
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


class RewriteTargeting:
    """Resolved gate for which paragraphs the direct rewrite path may touch this job, built once in
    production.py from the scan report's `rewrite_decision` and threaded down through both writer
    passes (pass 1 and the residual pass) so they agree on the same target set.

    Incident context (job 5bacaeb3): direct_rewrite.py never read `rewrite_decision` at all, so a
    `mode: targeted, full_rewrite_allowed: false` decision scoped to 22 findings in a handful of
    paragraphs still triggered a full 24-pass regeneration of a 12-paragraph GREEN document.

    `active=False` (the default/no-op gate) means "touch every paragraph" -- current behaviour,
    used whenever the switch is off, there is no decision, full rewrite is allowed, targets is
    empty, or the finding->paragraph mapping could not be resolved.
    """

    __slots__ = ("active", "target_paragraph_ids")

    def __init__(self, active: bool, target_paragraph_ids: set[str] | None):
        self.active = active
        self.target_paragraph_ids = target_paragraph_ids

    def should_skip(self, paragraph_id: str) -> bool:
        return bool(self.active and self.target_paragraph_ids is not None
                    and paragraph_id not in self.target_paragraph_ids)

    def skipped_trace(self, index: int, paragraph_id: str) -> dict[str, Any]:
        reason = (
            "rewrite_decision restricted this rewrite to specific findings; this paragraph "
            "contains none of them, so it was left unchanged."
        )
        return {
            "pass_index": index,
            "target_paragraph_id": paragraph_id,
            "selected_source": "skipped_not_target",
            "status": "skipped_not_target",
            "reason": reason,
            # CLAUDE.md: guards ANNOTATE, never SUPPRESS -- a paragraph left untouched must still
            # surface a card, or the author sees no solution and never learns why it wasn't rewritten
            # (2026-08-01 follow-up: skipped_not_target rows were pass-trace-only and invisible).
            "author_review_items": [skipped_not_target_review_card(index, paragraph_id, reason)],
        }


def skipped_not_target_review_card(index: int, paragraph_id: str, reason: str) -> dict[str, Any]:
    """Review card for a paragraph the scan flagged but this job's rewrite_decision scoped out of
    the target set -- the author must review it themselves since DraftProof made no changes here."""
    return {
        "added": "no change (outside this rewrite's scope)",
        "why": (
            f"{reason} Review this paragraph yourself -- DraftProof did not touch it."
        ),
        "paragraph_id": paragraph_id,
        "paragraph_index": index,
        "provenance": "skipped_not_target",
    }


_NOOP_TARGETING = RewriteTargeting(active=False, target_paragraph_ids=None)


def _finding_paragraph_map(detect_json: dict[str, Any]) -> dict[str, str]:
    """finding_id -> paragraph_id, read from the scan report's own findings[]/rewrite_context --
    the same association the report used to build `rewrite_decision.targets`. Tolerant of any
    tier/shape drift: returns whatever it can map, never raises."""
    findings_by_tier = detect_json.get("findings") if isinstance(detect_json, dict) else None
    mapping: dict[str, str] = {}
    if not isinstance(findings_by_tier, dict):
        return mapping
    for items in findings_by_tier.values():
        if not isinstance(items, list):
            continue
        for finding in items:
            if not isinstance(finding, dict):
                continue
            finding_id = finding.get("finding_id")
            if not finding_id:
                continue
            paragraph_id = (finding.get("rewrite_context") or {}).get("paragraph_id") or finding.get("paragraph_id")
            if paragraph_id:
                mapping[str(finding_id)] = str(paragraph_id)
    return mapping


def resolve_rewrite_targeting(detect_json: dict[str, Any] | None) -> RewriteTargeting:
    """Build this job's targeting gate from `detect_json["rewrite_decision"]`. Falls back to the
    no-op (rewrite everything) gate on every "can't be sure" branch -- switch off, missing/malformed
    decision, full_rewrite_allowed, empty targets, or an unresolvable finding->paragraph mapping --
    logging a warning only for the last case, since that one is the surprising one."""
    if not honor_rewrite_decision_enabled():
        return _NOOP_TARGETING
    decision = detect_json.get("rewrite_decision") if isinstance(detect_json, dict) else None
    if not isinstance(decision, dict):
        return _NOOP_TARGETING
    if decision.get("full_rewrite_allowed"):
        return _NOOP_TARGETING
    targets = decision.get("targets")
    if not targets:
        return _NOOP_TARGETING
    mapping = _finding_paragraph_map(detect_json)
    resolved = {mapping[target] for target in targets if target in mapping}
    if not resolved:
        _logger.warning(
            "rewrite_decision targeting: could not resolve any of %d target finding id(s) to a "
            "paragraph -- falling back to full-document rewrite.", len(targets),
        )
        return _NOOP_TARGETING
    return RewriteTargeting(active=True, target_paragraph_ids=resolved)
