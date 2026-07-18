"""Stylometric-consistency "Writing-style outliers" display composer
(display-layer only).

Re-presents the Phase-1 ConsistencyDetector's findings (poc/detect/consistency.py
— ``finding_type="stylometric_outlier"``, ``scanner="consistency"``) as an
advisory, informational-only panel — the single server-side source of truth
rendered byte-for-byte by BOTH surfaces (the PDF panel in render_panels.py and
the React ``ConsistencyRisk`` component). No per-surface derivation: both
consume THIS dict.

DISPLAY-ONLY, exactly like poc/report/claim_graph_panel.py and
poc/report/sentence_issue_tags.py: ``scoring`` is hard-False. ConsistencyDetector
.overall_risk is unconditionally 0.0 (see poc/detect/consistency.py module
docstring) — this composer NEVER touches the tier, the ai_likelihood_score, any
gate, or any scoring path. Returns ``None`` when there are no usable rows
(DRAFTPROOF_CONSISTENCY off / clean document / malformed input), so with the
flag OFF the report is byte-identical.

Pure function: deterministic, NO network / LLM / heavy imports. Fail-open — any
error returns ``None`` rather than breaking a report build.

Feature names in ``top_deviating_features`` are ALREADY plain English (see
poc/detect/stylometry/outliers.py's ``_FEATURE_EXTRACTORS`` — e.g. "sentence
length", "passive voice rate") — this module joins them for display, it does
not re-translate them (no separate code<->copy lookup table to drift out of
sync).
"""
from __future__ import annotations

from typing import Any, Optional

_EXCERPT_MAX = 320
_ROWS_CAP = 12
_FEATURES_CAP = 5
_RECOMMENDATION_MAX = 400

# Shown when a row's top_deviating_features list is empty/malformed — mirrors
# ConsistencyDetector._finding_for's own fallback phrase (poc/detect/
# consistency.py) so the two surfaces never say different things for the same
# "no specific feature stood out" case.
_FALLBACK_FEATURES_LABEL = "overall writing style"


def _trim(text: Any, limit: int) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit].rstrip() + "…"  # ellipsis


def _coerce_score(value: Any) -> Optional[float]:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _row_from_finding(finding: Any) -> Optional[dict[str, Any]]:
    """Build one display row from a single consistency finding, or ``None`` when
    the finding lacks the minimum identifying fields (paragraph_id + excerpt)."""
    if not isinstance(finding, dict):
        return None
    paragraph_id = str(finding.get("paragraph_id") or "").strip()
    excerpt = _trim(finding.get("excerpt"), _EXCERPT_MAX)
    if not paragraph_id or not excerpt:
        return None

    raw_features = finding.get("top_deviating_features")
    features = [
        str(x).strip() for x in (raw_features if isinstance(raw_features, list) else [])
        if isinstance(x, str) and str(x).strip()
    ][:_FEATURES_CAP]
    features_label = ", ".join(features) or _FALLBACK_FEATURES_LABEL

    return {
        "paragraph_id": paragraph_id,
        "excerpt": excerpt,
        "outlier_score": _coerce_score(finding.get("outlier_score")),
        "features": features,
        "features_label": features_label,
        "recommendation": _trim(finding.get("recommendation"), _RECOMMENDATION_MAX),
    }


def compose_consistency_display(findings: Any) -> Optional[dict[str, Any]]:
    """Compose the ``consistency_display`` contract, or ``None``.

    ``findings`` is the list of report-level finding rows for the
    ``"consistency"`` scanner — each a dict carrying at minimum
    ``paragraph_id``/``excerpt``/``outlier_score``/``top_deviating_features``
    (see poc/report/report.py's caller, which builds this list from
    ``report.models.Finding`` rows filtered to ``scanner == "consistency"``).

    Returns ``None`` when there are no usable rows (flag off / clean document /
    malformed input) so the panel renders nothing (byte-identical). Fail-open:
    any error -> ``None``.
    """
    try:
        if not isinstance(findings, list) or not findings:
            return None

        rows = [r for r in (_row_from_finding(f) for f in findings) if r is not None]
        if not rows:
            return None

        # Highest-outlier-score first — the most stylistically anomalous
        # paragraph is the most decision-relevant one to show first; ties break
        # stably by paragraph_id so ordering is deterministic across runs.
        rows.sort(
            key=lambda r: (
                -(r["outlier_score"] if isinstance(r["outlier_score"], (int, float)) else 0.0),
                r["paragraph_id"],
            )
        )
        rows = rows[:_ROWS_CAP]

        return {
            "present": True,
            "scoring": False,  # ALWAYS advisory — never affects tier/score.
            "summary": {"flagged_paragraphs": len(rows)},
            "rows": rows,
        }
    except Exception:
        return None


__all__ = ["compose_consistency_display"]
