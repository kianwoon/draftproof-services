"""Deterministic citation → claim linking (Phase-2 N1, Track A plumbing).

No network, no LLM, no DOI resolution — that is N2. This module only records
WHICH claim cites WHAT: it extracts in-text citation markers (reusing the
``CitationScanner`` in-text extractor from ``citation/scanner.py`` — the same
class ``detect/citation.py`` wraps) plus bare
DOIs and URLs, then links each to the CLAIM node whose deterministic ``source``
span contains it (nearest-preceding claim as a documented fallback). It emits
``EvidenceNode(kind="external_citation")`` — one per distinct citation.

Span-linking rule (deterministic, documented):
  A citation at absolute document offset ``o`` links to the CLAIM whose source
  span ``[char_start, char_end)`` CONTAINS ``o``. Among multiple containing
  claims the NARROWEST span wins (tie → earliest ``char_start``, then id). If no
  span contains ``o``, fall back to the nearest-PRECEDING claim (the one with the
  greatest ``char_end`` that is ``<= o``; tie → greatest ``char_start``, then id)
  — this catches trailing markers (``... cohorts. [1]``) that sit just past the
  sentence span. If no claim precedes it either, the citation is UNLINKED and is
  STILL emitted with ``claim_ids=[]`` (visible-but-unlinked — "annotate, don't
  suppress"; a citation the reader should see even though we can't anchor it).

N1 records the link only: ``verification_status`` is untouched (``verified`` is
N4). Fail-open: any error yields ``[]`` — never fails the build.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .schema import EvidenceNode

# A DOI: ``10.<registrant>/<suffix>`` (Crossref shape). Deterministic; trailing
# sentence punctuation is trimmed so ``10.1234/abc.def.`` yields ``...abc.def``.
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+")
# A bare http(s) URL.
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
# Trailing chars that are punctuation, not part of the locator.
_TRAIL = ".,;:)]}’”\"'"

_LOCATOR_PRIORITY = {"doi": 0, "url": 1, "in_text": 2}


def _in_text_markers(text: str) -> list[tuple[int, str, dict[str, Any]]]:
    """Reuse the existing deterministic in-text extractor.

    Returns ``(offset, end, detail)`` tuples. Import is lazy + guarded so a
    missing/renamed scanner degrades to DOI/URL-only rather than failing.
    """
    try:  # ``citation`` resolves from the poc root (namespace package).
        from citation.scanner import CitationScanner
    except Exception:
        return []
    out: list[tuple[int, str, dict[str, Any]]] = []
    for cit in CitationScanner()._detect_in_text(text):
        raw = str(cit.raw)
        start = int(cit.position)
        out.append((start, start + len(raw), {
            "marker": raw,
            "locator_type": "in_text",
            "style": getattr(cit, "style", None),
            "offset": start,
        }))
    return out


def _regex_markers(text: str, pattern: re.Pattern, locator_type: str,
                   key: str) -> list[tuple[int, str, dict[str, Any]]]:
    out: list[tuple[int, str, dict[str, Any]]] = []
    for m in pattern.finditer(text):
        raw = m.group(0).rstrip(_TRAIL)
        if not raw:
            continue
        start = m.start()
        out.append((start, start + len(raw), {
            "marker": raw,
            "locator_type": locator_type,
            key: raw,
            "offset": start,
        }))
    return out


def _dedup_overlaps(
    candidates: list[tuple[int, str, dict[str, Any]]],
) -> list[tuple[int, str, dict[str, Any]]]:
    """Drop candidates whose char range overlaps an already-kept one.

    Deterministic: sort by (start, locator-priority, marker) so a DOI wins over a
    URL that wraps it (``https://doi.org/10.x``) and both win over an in-text
    marker at the same spot. Greedy non-overlap keep."""
    ordered = sorted(
        candidates,
        key=lambda c: (c[0], _LOCATOR_PRIORITY.get(c[2]["locator_type"], 9), c[2]["marker"]),
    )
    kept: list[tuple[int, str, dict[str, Any]]] = []
    for start, end, detail in ordered:
        if any(start < k_end and k_start < end for k_start, k_end, _ in kept):
            continue
        kept.append((start, end, detail))
    return kept


def _claim_spans(claims: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    """Extract ``(char_start, char_end, id)`` for CLAIM nodes carrying a span."""
    spans: list[tuple[int, int, str]] = []
    for c in claims or []:
        src = c.get("source")
        if not isinstance(src, dict):
            continue
        try:
            start = int(src["char_start"])
            end = int(src["char_end"])
        except (KeyError, TypeError, ValueError):
            continue
        cid = str(c.get("id") or "")
        if cid and end > start:
            spans.append((start, end, cid))
    return spans


def _link_offset(offset: int, spans: list[tuple[int, int, str]]) -> list[str]:
    """Apply the documented containment → nearest-preceding rule."""
    containing = [(end - start, start, cid) for start, end, cid in spans
                  if start <= offset < end]
    if containing:
        # Narrowest span wins; tie → earliest start, then id.
        containing.sort(key=lambda t: (t[0], t[1], t[2]))
        return [containing[0][2]]
    preceding = [(end, start, cid) for start, end, cid in spans if end <= offset]
    if preceding:
        # Nearest preceding = greatest end; tie → greatest start, then id.
        preceding.sort(key=lambda t: (t[0], t[1], t[2]))
        return [preceding[-1][2]]
    return []  # unlinked — still emitted


def link_citations(text: Any, claims: list[dict[str, Any]]) -> list[EvidenceNode]:
    """Return deterministic ``external_citation`` EvidenceNodes for ``text``.

    ``claims`` are validated CLAIM-node dicts (validator output) carrying a
    ``source`` char span. Fail-open: returns ``[]`` on any error."""
    try:
        if not isinstance(text, str) or not text:
            return []
        candidates = (
            _in_text_markers(text)
            + _regex_markers(text, _DOI_RE, "doi", "doi")
            + _regex_markers(text, _URL_RE, "url", "url")
        )
        if not candidates:
            return []
        distinct = _dedup_overlaps(candidates)
        spans = _claim_spans(claims)

        nodes: list[EvidenceNode] = []
        for i, (start, _end, detail) in enumerate(distinct, start=1):
            claim_ids = _link_offset(start, spans)
            nodes.append(EvidenceNode(
                id=f"e_{i:03d}",
                kind="external_citation",
                claim_ids=claim_ids,
                detail=detail,
            ))
        return nodes
    except Exception:
        return []  # fail-open — never fails the build


__all__ = ["link_citations"]
