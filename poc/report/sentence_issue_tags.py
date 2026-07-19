"""Per-sentence issue-tag display composer (display-layer only).

Single server-side source of truth for the "Read full document" issue-underline
layer, rendered byte-for-byte by BOTH surfaces (the React SignalHighlights view
and the PDF full-document render in render_panels.py). No per-surface derivation:
both consume THIS dict, attached at ``result["sentence_issue_tags"]``.

Design (owner-approved, 2026-07-16): mark each flagged sentence with COLORED
UNDERLINES per finding type — NOT font color — so a sentence with two issues can
carry BOTH underlines stacked, nothing hidden, text stays readable:

  * red    = reads as AI          (ai_signal_deberta fired, tier high/amber)
  * amber  = weak grounding       (finding title == "low_specificity")
  * purple = reasoning jump       (finding title == "semantic_drift")
  * blue   = citation not verified (finding title in {"missing_from_bib",
             "uncited_claim"} — CitationDetector's bibliography cross-check,
             added 2026-07-19). Distinct from the unrelated, pre-existing
             "Citation grounding" AI-pattern signal chip elsewhere in the report
             (reportHelpers.js citation_grounding_risk, red) — that's a
             transformation-feature heuristic; this is an actual bibliography
             existence/cross-reference check. Uses ``finding.sentence_id`` like
             grounding/reasoning — NOT the detector-level ``location`` char-offset
             (that field is dropped during report-layer serialization, see
             report.py's ``_tier_findings``); ``sentence_id`` is ALREADY resolved
             from that same location upstream, in
             poc/report/builder.py::add_detection's "Strategy 2" (start_char →
             containing predictability sentence), which every finding type goes
             through, not something citation needed its own resolver for.
             Deliberately EXCLUDES "uncited_in_body" (a bibliography entry never
             cited in the body): its evidence text lives only in the reference
             list, which CitationDetector strips out of ``content`` before
             locating evidence when auto-extracting the bibliography — so on
             the common (auto-extract) path it never resolves a location/
             sentence_id upstream either. (On the less-common explicit
             ``bib_text=`` path, ``content`` isn't re-stripped, so a
             sentence_id COULD resolve there — this exclusion is title-based
             on purpose, not merely riding on an absent-sentence_id side
             effect.) The ``if sid:`` check below would already skip
             it.

TRUSTWORTHY FINDINGS ONLY. The noisy predictability / genericity findings
(medium_predictability, high_topk_predictability, review_predictability,
genericity, …) were deliberately pulled from the inline view (reportHelpers.js
"DeBERTa-only contract") because they came from an abandoned methodology — an
allow-list keeps them out here too, so this never recreates "methodology soup".

``semantic_drift`` is sometimes DOCUMENT-LEVEL (no sentence_id, as in the sample
report) → it goes to ``document_level`` and is shown as a note under the legend,
NEVER faked onto a sentence.

DISPLAY-ONLY, exactly like poc/report/authorship_evidence_levels.py and
claim_graph_panel.py: this NEVER touches the tier, the ai_likelihood_score, any
gate, or any scoring path. Returns ``None`` when there is no trustworthy finding
(older reports / clean docs) → the view renders as today, byte-identical.

Pure function: deterministic, NO network / LLM / heavy imports. Fail-open — any
error returns ``None`` rather than breaking a report build. Emits i18n CODES
(label_code / fix_code) resolved by the web via t(); also carries English
fallbacks (label_en / fix_en) so the PDF can render without an i18n table.
"""
from __future__ import annotations

from typing import Any, Optional

# ── Trustworthy finding titles → (type, color). Anything not here is EXCLUDED. ──
_GROUNDING_TITLE = "low_specificity"
_REASONING_TITLE = "semantic_drift"
# CitationDetector's bibliography cross-check titles (poc/citation/scanner.py).
# "uncited_in_body" is deliberately NOT included — see module docstring.
_CITATION_TITLES = frozenset({"missing_from_bib", "uncited_claim"})

# The AI signal that fires the red "reads as AI" underline, and the tiers that
# count as a genuine flag (matching the verdict-gated highlight contract).
_AI_SIGNAL_KEY = "ai_signal_deberta"
_AI_FLAG_TIERS = frozenset({"high", "amber"})

# type → deterministic display metadata (color + i18n codes + English fallbacks).
_TAG_META: dict[str, dict[str, str]] = {
    "ai": {
        "color": "red",
        "label_code": "tagAi",
        "fix_code": "tagAiFix",
        "label_en": "Reads as AI",
        "fix_en": "Rewrite in your own voice and ground it in a specific only you could know.",
    },
    "grounding": {
        "color": "amber",
        "label_code": "tagGrounding",
        "fix_code": "tagGroundingFix",
        "label_en": "Weak grounding",
        "fix_en": "Add a concrete anchor — a name, a number, an example, a source.",
    },
    "reasoning": {
        "color": "purple",
        "label_code": "tagReasoning",
        "fix_code": "tagReasoningFix",
        "label_en": "Reasoning jump",
        "fix_en": "Bridge the transition with the evidence or logic that connects the ideas.",
    },
    "citation": {
        "color": "blue",
        "label_code": "tagCitation",
        "fix_code": "tagCitationFix",
        "label_en": "Citation not verified",
        "fix_en": "Add this source to your bibliography, or cite a source for this claim.",
    },
}

# Stable render order for a multi-issue sentence: AI first, then grounding, then
# reasoning, then citation — so stacked underlines never reorder between
# reports/surfaces.
_TYPE_ORDER = {"ai": 0, "grounding": 1, "reasoning": 2, "citation": 3}


def _tag(kind: str, fix_text: Optional[str] = None) -> dict[str, Any]:
    """Build one tag dict from the deterministic metadata for ``kind``.

    ``fix_text`` (optional) carries the finding's own recommendation VERBATIM
    (no fabrication) so a surface can show the real advice; renderers fall back
    to the generic fix_code / fix_en when it is absent.
    """
    meta = _TAG_META[kind]
    tag = {
        "type": kind,
        "color": meta["color"],
        "label_code": meta["label_code"],
        "fix_code": meta["fix_code"],
        "label_en": meta["label_en"],
        "fix_en": meta["fix_en"],
    }
    ft = str(fix_text or "").strip()
    if ft:
        tag["fix_text"] = ft
    return tag


def _iter_findings(findings: Any):
    """Yield each finding dict from the severity-keyed findings container."""
    if not isinstance(findings, dict):
        return
    for _sev, items in findings.items():
        if not isinstance(items, list):
            continue
        for f in items:
            if isinstance(f, dict):
                yield f


def _ai_tagged_sentence_ids(highlight_segments: Any) -> list[str]:
    """Sentence ids whose ai_signal_deberta signal fired at a flag tier, in order."""
    ordered: list[str] = []
    seen: set[str] = set()
    if not isinstance(highlight_segments, list):
        return ordered
    for seg in highlight_segments:
        if not isinstance(seg, dict):
            continue
        sid = seg.get("sentence_id")
        if not sid or sid in seen:
            continue
        for sig in (seg.get("signals") or []):
            if not isinstance(sig, dict):
                continue
            if sig.get("key") != _AI_SIGNAL_KEY:
                continue
            tier = str(sig.get("tier") or "").strip().lower()
            if tier in _AI_FLAG_TIERS:
                ordered.append(sid)
                seen.add(sid)
                break
    return ordered


def compose_sentence_issue_tags(report_fields: Any) -> Optional[dict[str, Any]]:
    """Compose the ``sentence_issue_tags`` display contract, or ``None``.

    Returns ``None`` when there is no trustworthy finding (so the "Read full
    document" view renders exactly as today — byte-identical). Otherwise returns
    the display-only dict both surfaces consume. Fail-open: any error → ``None``.
    """
    try:
        if not isinstance(report_fields, dict):
            return None

        sentences: dict[str, list[dict[str, Any]]] = {}
        document_level: list[dict[str, Any]] = []

        def _add(sid: str, tag: dict[str, Any]) -> None:
            bucket = sentences.setdefault(sid, [])
            # De-dupe by type — a sentence carries at most one tag per issue kind.
            if any(existing["type"] == tag["type"] for existing in bucket):
                return
            bucket.append(tag)

        # ── AI (red) from the verdict-gated highlight segments. ──
        for sid in _ai_tagged_sentence_ids(report_fields.get("highlight_segments")):
            _add(sid, _tag("ai"))

        # ── Grounding (amber) + reasoning (purple) from trustworthy findings. ──
        for f in _iter_findings(report_fields.get("findings")):
            title = str(f.get("title") or "").strip()
            sid = f.get("sentence_id")
            rec = f.get("recommendation")
            if title == _GROUNDING_TITLE:
                if sid:
                    _add(str(sid), _tag("grounding", rec))
            elif title == _REASONING_TITLE:
                tag = _tag("reasoning", rec)
                if sid:
                    _add(str(sid), tag)
                else:
                    document_level.append(tag)
            elif title in _CITATION_TITLES:
                # sentence_id (like grounding/reasoning) — already resolved
                # upstream from the finding's location, see module docstring.
                # No document_level fallback: an unresolved citation finding
                # (uncited_in_body's evidence never resolves — see docstring)
                # is silently skipped, never faked onto the whole document.
                if sid:
                    _add(str(sid), _tag("citation", rec))
            # else: EXCLUDED (predictability / genericity / uncited_in_body / anything else).

        if not sentences and not document_level:
            return None

        # Stable per-sentence ordering (ai → grounding → reasoning → citation).
        for sid in sentences:
            sentences[sid].sort(key=lambda t: _TYPE_ORDER.get(t["type"], 9))

        legend = [
            {"color": _TAG_META[k]["color"], "label_code": _TAG_META[k]["label_code"]}
            for k in ("ai", "grounding", "reasoning", "citation")
        ]

        return {
            "present": True,
            "scoring": False,  # ALWAYS advisory — never affects the score.
            "sentences": sentences,
            "document_level": document_level,
            "legend": legend,
        }
    except Exception:
        return None


__all__ = ["compose_sentence_issue_tags"]
