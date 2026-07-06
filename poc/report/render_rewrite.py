"""Dedicated rewrite report renderer — produces markdown for rewrite PDFs."""

import time
import re
import html
from typing import List, Dict, Any, Optional

# Single source for the external-band -> friendly label map (shared with the scan PDF /
# page / email so the rewrite PDF's Turnitin row reads identically).
from .contribution import contribution_pair, contribution_pair_int
from .render import _EXTERNAL_BAND_LABELS, EXTERNAL_ESTIMATE_DISPLAY_ENABLED


_TIER_ORDER = ["critical", "high", "medium", "low"]
# allow-hardcode: presentation level labels for the two policy scores (machine level
# -> display), not a scoring/matching oracle. Mirrors render.py / the web report.
_POLICY_LEVEL = {"low": "Low", "moderate": "Moderate", "high": "High", "severe": "Severe", "unknown": "Unknown"}
_TURNITIN_AI_REFERENCE_THRESHOLD = 20
_TURNITIN_AI_REFERENCE_NOTE = (
    "Turnitin reference: AI scores below 20% may appear as *% instead of an exact percentage "
    "because low-range results are less reliable. DraftProof scores are review signals, not verdicts."
)


def _count_findings(report_or_findings: dict) -> int:
    findings = report_or_findings.get("findings", report_or_findings) if report_or_findings else {}
    return sum(len(findings.get(t, [])) for t in _TIER_ORDER)


def _severity_score(findings: dict) -> int:
    """Weighted finding severity for rewrite outcome comparison."""
    weights = {"critical": 8, "high": 5, "medium": 2, "low": 1}
    return sum(len((findings or {}).get(t, [])) * weights[t] for t in weights)


def _review_burden(findings: dict) -> int:
    """Count findings that normally require user attention."""
    findings = findings or {}
    return (
        len(findings.get("critical", []))
        + len(findings.get("high", []))
        + len(findings.get("medium", []))
    )


def _badge(report: dict) -> dict:
    return (report or {}).get("ai_risk_badge") or {}


def _metric_percent(value, *, clamp: bool = True) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if 0 <= abs(number) <= 1:
        number *= 100
    return max(0.0, min(100.0, number)) if clamp else number


def _first_metric(*values) -> float | None:
    for value in values:
        percent = _metric_percent(value)
        if percent is not None:
            return percent
    return None


def _ai_score(report: dict) -> float:
    return _first_metric((report or {}).get("ai_score"), _badge(report).get("ai_likelihood_score")) or 0.0


def _display_ai_score(value: float) -> float:
    # Canonical DraftProof AI-likelihood percent (badge ai_likelihood_score), shown directly.
    # Previously halved by a 0.5 "display multiplier"; removed so the rewrite PDF agrees with
    # the report page, the scan PDF, and the email.
    return _metric_percent(value, clamp=False) or 0.0


def _deep_scan_pct(report: dict) -> float | None:
    """V7 deberta deep-scan proportion (percent) for a scan report, or None when the deep-scan
    didn't run (flags off / unavailable). Sourced from badge.tier_authority.proportion — the same
    number the scan page's fused score is built from. None -> the hero omits the deep-scan KPI."""
    badge = _badge(report)
    ta = badge.get("tier_authority") if isinstance(badge, dict) and isinstance(badge.get("tier_authority"), dict) else None
    prop = ta.get("proportion") if ta else None
    return round(float(prop) * 100, 1) if isinstance(prop, (int, float)) else None


def _ai_reference_suffix(score) -> str | None:
    if not isinstance(score, (int, float)):
        return None
    if float(score) < _TURNITIN_AI_REFERENCE_THRESHOLD:
        return "below 20% reference"
    return "review threshold exceeded"


def _with_ai_reference(text: str, score) -> str:
    suffix = _ai_reference_suffix(score)
    return f"{text} · {suffix}" if suffix else text


def _ai_signal_stamp(score) -> dict:
    if not isinstance(score, (int, float)):
        return {"label": "AI Review", "color": "#334155", "bg": "#f8fafc"}
    value = float(score)
    if value >= 60:
        return {"label": "Strong AI Signal", "color": "#b91c1c", "bg": "#fef2f2"}
    if value >= 45:
        return {"label": "Likely AI-Assisted", "color": "#c2410c", "bg": "#fff7ed"}
    if value >= 32:
        return {"label": "Possible AI-Assisted", "color": "#b45309", "bg": "#fff7ed"}
    if value >= _TURNITIN_AI_REFERENCE_THRESHOLD:
        return {"label": "Unlikely AI-Assisted", "color": "#0f766e", "bg": "#f0fdfa"}
    return {"label": "Low AI Signal", "color": "#15803d", "bg": "#f0fdf4"}


def _requires_external_review(summary: dict) -> bool:
    if not isinstance(summary, dict):
        return False
    return False


def _requires_author_review(summary: dict) -> bool:
    if not isinstance(summary, dict):
        return False
    if summary.get("no_text_change") is True:
        return False
    if summary.get("outcome") == "original_preserved" or summary.get("status") == "original_preserved":
        return False
    author_proxy_context = summary.get("author_proxy_context") if isinstance(summary.get("author_proxy_context"), dict) else {}
    strict_safe = (
        summary.get("strict_safe_band_achieved") is True
        or summary.get("kpi_finalization_status") in {
            "strict_safe_auto_finalized",
            "strict_safe_author_review_required",
        }
    )
    return (
        summary.get("best_candidate_author_review_required") is True
        or summary.get("public_candidate_warning") == "author_proxy_candidate_requires_review"
        or summary.get("outcome") == "rewrite_candidate_generated_needs_author_review"
        or summary.get("status") == "rewrite_candidate_generated_needs_author_review"
        or summary.get("kpi_finalization_status") == "strict_safe_author_review_required"
        or (strict_safe and author_proxy_context.get("review_required") is True)
    )


def _rewrite_stamp(summary: dict, risk) -> dict:
    if _requires_author_review(summary):
        return {
            "caption": "Rewritten Rating",
            "label": "Author Review",
            "color": "#0f766e",
            "bg": "#ecfdf5",
        }
    if _requires_external_review(summary):
        return {
            "caption": "Rewritten Rating",
            "label": "Review Required",
            "color": "#92400e",
            "bg": "#fffbeb",
        }
    stamp = _ai_signal_stamp(risk)
    return {"caption": "Rewritten Rating", **stamp}


# Seal verdict tracks the DETECTOR reality (external/Turnitin band). Users read a reassuring
# verdict as "Turnitin-safe", but the external/fluency estimate has a floor no rewrite removes,
# so a safe-looking verdict next to a still-flagged card is a false promise. A reassuring verdict
# is earned only when detectors genuinely stop flagging (band "low").
_EXTERNAL_VERDICT = {
    "high": {"label": "Still flagged by detectors", "color": "#b91c1c", "bg": "#fef2f2"},
    "elevated": {"label": "Detector risk remains", "color": "#c2410c", "bg": "#fff7ed"},
    "low": {"label": "Detector risk low", "color": "#15803d", "bg": "#f0fdf4"},
}
_GROUNDING_SEAL_SUBTITLE = "grounding improved — review and finish in your own words"


def _rewrite_verdict(summary: dict, rewritten_scan: dict) -> dict:
    """Rewritten-rating seal verdict. Review states win first. INTERIM (EXTERNAL_ESTIMATE_DISPLAY_
    ENABLED=False): the external/Turnitin band over-flags real Turnitin, so we do NOT assert a
    detector verdict ("Still flagged" / "Detector risk low") that the band can't support -- the
    seal stays a neutral grounding verdict. When re-enabled (post-calibration), the band drives it."""
    if _requires_author_review(summary):
        return {"caption": "Rewritten Rating", "label": "Author Review", "color": "#0f766e", "bg": "#ecfdf5"}
    if _requires_external_review(summary):
        return {"caption": "Rewritten Rating", "label": "Review Required", "color": "#92400e", "bg": "#fffbeb"}
    if not EXTERNAL_ESTIMATE_DISPLAY_ENABLED:
        return {"caption": "Rewritten Rating", "label": "Grounded draft — review", "color": "#0f766e", "bg": "#ecfdf5"}
    band = str((_badge(rewritten_scan).get("external_detector_estimate") or {}).get("band") or "").lower()
    verdict = _EXTERNAL_VERDICT.get(band) or {"label": "Rewrite reviewed", "color": "#334155", "bg": "#f8fafc"}
    return {"caption": "Rewritten Rating", **verdict}


def _wq_score(report: dict) -> float:
    return _first_metric((report or {}).get("writing_score"), _badge(report).get("writing_quality_score")) or 0.0


def _tier(report: dict) -> str:
    return (_badge(report).get("tier") or (report or {}).get("overall_tier", "?")).upper()


def _authorship_label(report: dict) -> str:
    badge = _badge(report)
    rating = badge.get("authorship_rating") or {}
    return (
        rating.get("short_label")
        or rating.get("label")
        or badge.get("authorship_rating_label")
        or ""
    )


def _pct(value: Any) -> Optional[int]:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if 0 <= number <= 1:
        number *= 100
    return int(round(max(0, min(100, number))))


def _scan_contribution(scan: dict) -> dict:
    intelligence = (scan or {}).get("scan_intelligence") or {}
    transformation = intelligence.get("transformation") or {}
    contribution = transformation.get("contribution") or {}
    layers = ((scan or {}).get("integrity_layers") or {}).get("layers") or (
        (intelligence.get("integrity_layers") or {}).get("layers") or {}
    )
    human_layer = layers.get("human_contribution_signal") or {}
    ai_layer = layers.get("ai_transformation_risk") or {}
    raw_human = (
        contribution.get("human_contribution_ratio")
        if contribution.get("human_contribution_ratio") is not None
        else contribution.get("human_contribution")
        if contribution.get("human_contribution") is not None
        else contribution.get("human_ratio")
        if contribution.get("human_ratio") is not None
        else human_layer.get("score")
    )
    raw_ai = (
        contribution.get("ai_transformation_ratio")
        if contribution.get("ai_transformation_ratio") is not None
        else contribution.get("ai_transformation")
        if contribution.get("ai_transformation") is not None
        else contribution.get("transformation_ratio")
        if contribution.get("transformation_ratio") is not None
        else ai_layer.get("score")
    )
    human, ai = contribution_pair_int(raw_human, raw_ai)
    return {
        "human": human,
        "ai": ai,
        "calibrated": _pct(
            contribution.get("calibrated_ai_risk")
            if contribution.get("calibrated_ai_risk") is not None
            else contribution.get("adjusted_ai_risk")
        ),
        "human_anchor_discount": _pct(contribution.get("human_anchor_discount")),
        "calibration_confidence": _pct(contribution.get("calibration_confidence")),
        "reporting_suppression": _pct(contribution.get("reporting_suppression")),
        "summary": contribution.get("summary") or "",
    }


def _outcome_stamp_html(summary: dict, result_label: str, rewritten_scan: dict) -> str:
    contribution = _scan_contribution(rewritten_scan)
    detect_scores = summary.get("detect_scores") if isinstance(summary.get("detect_scores"), dict) else {}
    rewritten_raw_ai = _first_metric(
        detect_scores.get("rewritten_ai_authorship"),
        detect_scores.get("rewritten_ai"),
        summary.get("final_risk"),
        _ai_score(rewritten_scan) if rewritten_scan else None,
    )
    ai_score = _pct(_display_ai_score(rewritten_raw_ai)) if rewritten_raw_ai is not None else None
    calibrated = contribution.get("calibrated")
    risk = ai_score if ai_score is not None else calibrated
    author_review_required = _requires_author_review(summary)
    external_review_required = _requires_external_review(summary)
    stamp = _rewrite_stamp(summary, risk)
    # Seal VERDICT tracks the detector reality (external/Turnitin band), not the rosy internal
    # calibrated risk -- users read a reassuring verdict as "Turnitin-safe", which no rewrite can
    # honestly promise. The subtitle credits the grounding work and points the user to finish it.
    verdict = _rewrite_verdict(summary, rewritten_scan)
    seal_subtitle = (
        "Author review required"
        if author_review_required
        else "External review required"
        if external_review_required
        else _GROUNDING_SEAL_SUBTITLE
    )
    scan_heading = (
        _authorship_label(rewritten_scan)
        if author_review_required or external_review_required
        else stamp["label"]
    ) or result_label
    # Lead with the policy levels (mirror the web report), not the bare AI-likelihood %.
    # Fall back to the AI score for older reports whose badge lacks policy_risk.
    _pol = (_badge(rewritten_scan).get("policy_risk") or {}) if isinstance(_badge(rewritten_scan), dict) else {}
    _pa, _pr2 = _pol.get("ai_allowed") or {}, _pol.get("ai_restricted") or {}
    if _pa.get("level") and _pa["level"] != "unknown":
        scan_score = (
            f'Allowed {_POLICY_LEVEL.get(_pa["level"], _pa["level"])} '
            f'· Restricted {_POLICY_LEVEL.get(_pr2.get("level"), _pr2.get("level"))}'
        )
    else:
        scan_score = f"{ai_score}%" if ai_score is not None else "-"
    human = contribution.get("human")
    ai = contribution.get("ai")
    summary_text = contribution.get("summary") or summary.get("outcome") or result_label
    metric_chips = []
    for label, value in (
        ("Calibrated AI risk", calibrated),
        ("Human anchor discount", contribution.get("human_anchor_discount")),
        ("Calibration confidence", contribution.get("calibration_confidence")),
        ("Reporting suppression", contribution.get("reporting_suppression")),
    ):
        if value is not None:
            metric_chips.append(f"<span>{html.escape(label)} {value}%</span>")
    bars = ""
    if human is not None or ai is not None:
        human_width = max(0, min(100, human or 0))
        ai_width = max(0, min(100, ai or 0))
        bars = f"""
        <div class="dp-outcome-bars">
          <div><strong>Human Contribution</strong><em>{human if human is not None else '-'}%</em></div>
          <i class="dp-outcome-bar"><b class="dp-human" style="width:{human_width}%"></b></i>
          <div><strong>AI Transformation</strong><em>{ai if ai is not None else '-'}%</em></div>
          <i class="dp-outcome-bar"><b class="dp-ai" style="width:{ai_width}%"></b></i>
        </div>
        """
    return f"""
<div class="dp-rewrite-outcome-panel">
  <div class="dp-rewrite-stamp" style="color:{verdict["color"]};border-color:{verdict["color"]};background:{verdict["bg"]}">
    <span>{html.escape(verdict["caption"])}</span>
    <strong>{html.escape(verdict["label"].upper())}</strong>
    <em>{html.escape(seal_subtitle)}</em>
  </div>
  <div class="dp-rewrite-scan-summary">
    <span>Rewritten Scan</span>
    <h3>{html.escape(scan_heading)}</h3>
    <b>{html.escape(scan_score)}</b>
    <p>{html.escape(str(summary_text))}</p>
    <p class="dp-ai-reference-note">{html.escape(_TURNITIN_AI_REFERENCE_NOTE)}</p>
    <div class="dp-outcome-chips">{''.join(metric_chips)}</div>
    {bars}
  </div>
</div>
"""


def _document_section(title: str, text: str) -> List[str]:
    content = str(text or "").strip()
    if not content:
        return []
    return [
        f"## {title}",
        "",
        "```text",
        content,
        "```",
        "",
    ]


# ── Executive-summary two-column comparison (mirrors the web rewrite view) ────
# allow-hardcode: presentation copy/markup + machine-band -> display maps for the
# PDF comparison cards. Band words are read from the badge's own tier field, never
# invented; these are display strings, never matched against document text.
_CMP_TIER_TO_BAND = {"green": "low", "amber": "moderate", "orange": "high", "red": "critical"}
_CMP_BAND_LABEL = {"low": "Low", "moderate": "Moderate", "high": "High", "critical": "Critical"}
_CMP_BAND_KIND = {"low": "good", "moderate": "info", "high": "warn", "critical": "warn"}
_CMP_NOT_TURNITIN = (
    "This is DraftProof's own AI-likelihood on our scale — not a Turnitin score, and not "
    "comparable to the 20% line. Detectors over-flag fluent writing; treat it as a review "
    "signal, not a verdict."
)


def _cmp_pattern_label(badge: dict) -> str:
    tc = (badge.get("transformation_classification") or {}) if isinstance(badge, dict) else {}
    return str(tc.get("label") or tc.get("short_label") or tc.get("headline") or "").strip()


def _cmp_rating_chip(badge: dict) -> str:
    """Fused-scale rating chip 'LOW · 16%' from badge.tier + tier_authority.fused_score.
    '' when the fused score is absent (legacy badge)."""
    ta = badge.get("tier_authority") if isinstance(badge, dict) else None
    fused = ta.get("fused_score") if isinstance(ta, dict) else None
    if not isinstance(fused, (int, float)):
        return ""
    band = _CMP_TIER_TO_BAND.get(str(badge.get("tier") or "").lower())
    if not band:
        return ""
    label = _CMP_BAND_LABEL.get(band, band.title())
    return _cmp_chip(f"{label.upper()} · {round(fused)}%", _CMP_BAND_KIND.get(band, "info"))


def _cmp_chip(text: str, kind: str) -> str:
    return f'<span class="dp-statchip dp-statchip--{kind}">{html.escape(text)}</span>'


def _cmp_risk_axes_html(badge: dict) -> str:
    """'Where the risk sits · independent' rows from badge.submission_risk.axes
    (label + level chip). '' when no axes (legacy badge)."""
    sr = (badge.get("submission_risk") or {}) if isinstance(badge, dict) else {}
    axes = sr.get("axes") or {}
    if not isinstance(axes, dict) or not axes:
        return ""
    # Level -> chip family + display word (mirrors render_panels._level_kind + _SR_LEVEL_LABELS).
    # Compact colored bands matching the web's .merged-axis (label left, level word
    # right, full-row tint by level) — NOT big bordered chip cards. Level -> CSS class
    # + display word use the same set as the frontend's submission_risk.levels.
    level_class = {"low": "low", "moderate": "medium", "medium": "medium",
                   "high": "high", "severe": "critical", "critical": "critical",
                   "unknown": "unknown"}
    level_word = {"low": "Low", "moderate": "Moderate", "medium": "Medium",
                  "high": "High", "severe": "Severe", "critical": "Critical", "unknown": "Unknown"}
    order = ["text_pattern", "ownership", "citation", "defence_readiness", "policy_declaration"]
    keys = [k for k in order if k in axes] + [k for k in axes if k not in order]
    rows = []
    for key in keys:
        ax = axes.get(key) or {}
        label = ax.get("label") or key.replace("_", " ").title()
        lvl = str(ax.get("level") or "unknown").lower()
        cls = level_class.get(lvl, "unknown")
        word = level_word.get(lvl, lvl.title() or "Unknown")
        rows.append(
            f'<div class="dp-cmp-axis dp-cmp-axis--{cls}">'
            f'<span class="dp-cmp-axis-name">{html.escape(str(label))}</span>'
            f'<strong class="dp-cmp-axis-level">{html.escape(word)}</strong></div>'
        )
    if not rows:
        return ""
    # Wrap heading + all rows in one group so pagination breaks BEFORE the section
    # (moving it whole to the next page) rather than splitting the risk list mid-way.
    return ('<div class="dp-cmp-risk-group">'
            '<p class="dp-cmp-subhead">Where the risk sits · independent</p>'
            + "".join(rows) + "</div>")


def _cmp_column(kicker: str, scan: dict, *, review_required: bool = False) -> str:
    """One comparison column: kicker, pattern label, rating chip (or 'Review required'),
    the fused verdict card, the 4 writing-reads bars, and the risk axes. Each sub-block
    fails open independently — an absent block is simply omitted (legacy rewrites)."""
    from .render_panels import _authorship_headline_html, _authorship_bars_html

    badge = _badge(scan)
    breakdown = badge.get("authorship_breakdown") if isinstance(badge, dict) else None

    parts = [f'<p class="dp-cmp-kicker">{html.escape(kicker)}</p>']

    pattern = _cmp_pattern_label(badge)
    if pattern:
        parts.append(f'<p class="dp-cmp-pattern">{html.escape(pattern)}</p>')

    if review_required:
        parts.append(_cmp_chip("Review required", "warn"))
    else:
        chip = _cmp_rating_chip(badge)
        if chip:
            parts.append(chip)

    # Fused verdict card (reuse the scan PDF's helper: 'DraftProof AI-likelihood 16%' +
    # band chip + 'Behind this score: composite X%, deep-scan Y%').
    if isinstance(breakdown, dict):
        headline = _authorship_headline_html(badge, breakdown)
        if headline:
            parts.append(headline)
        # Defend-lead line + not-a-Turnitin note.
        ta = badge.get("tier_authority") if isinstance(badge, dict) else None
        if isinstance(ta, dict) and isinstance(ta.get("fused_score"), (int, float)):
            band = _CMP_TIER_TO_BAND.get(str(badge.get("tier") or "").lower())
            defend = ("You can defend this as your own work."
                      if band == "low" else
                      "Strengthen this before you can fully defend it as your own.")
            parts.append(f'<p class="dp-hero-sub">{html.escape(defend)}</p>')
            parts.append(f'<p class="dp-ai-reference-note">{html.escape(_CMP_NOT_TURNITIN)}</p>')
        # 'How the writing reads · sums to 100%' — the 4 category bars.
        bars = _authorship_bars_html(breakdown)
        if bars:
            parts.append('<p class="dp-cmp-subhead">How the writing reads · sums to 100%</p>')
            parts.append(bars)

    axes = _cmp_risk_axes_html(badge)
    if axes:
        parts.append(axes)

    return '<div class="dp-cmp-col">' + "".join(parts) + "</div>"


def _executive_comparison_html(summary: dict, *, review_required: bool) -> str:
    """Two-column 'Original vs rewritten' scan cards. '' when neither scan carries the
    V7 fused badge (legacy rewrite) — caller then keeps the legacy hero/KPI lead."""
    orig = summary.get("detect_scan_original_saved") or summary.get("detect_scan_original") or {}
    new = summary.get("detect_scan_rewritten") or {}
    orig_fused = isinstance((_badge(orig).get("tier_authority") or {}).get("fused_score"), (int, float))
    new_fused = isinstance((_badge(new).get("tier_authority") or {}).get("fused_score"), (int, float))
    if not (orig_fused or new_fused):
        return ""
    left = _cmp_column("ORIGINAL SCAN", orig)
    right = _cmp_column("REWRITTEN SCAN", new, review_required=review_required)
    return ('<div class="dp-cmp-grid">'
            '<div class="dp-cmp-cell">' + left + "</div>"
            '<div class="dp-cmp-cell">' + right + "</div>"
            "</div>")


def _highlighted_submitted_content(scan: dict, original_text: str,
                                   original_scan_report: Optional[dict] = None) -> List[str]:
    """'Submitted Content' with scan-flagged sentences highlighted, reusing the scan PDF's
    render_highlighted_document. Highlight spans live in the ORIGINAL SCAN's full report.json
    (scan_intelligence.document.segments) — callers pass it as ``original_scan_report`` when
    available; the compacted rewrite-scan copy is tried second (it usually carries no segments).
    Falls open to the plain-text block when neither source has spans."""
    from .render_panels import render_highlighted_document
    highlighted = ""
    for source in (original_scan_report, scan):
        if not isinstance(source, dict) or not source:
            continue
        try:
            highlighted = render_highlighted_document(source, original_text or "")
        except Exception:
            highlighted = ""
        if highlighted:
            break
    if not highlighted:
        return _document_section("Submitted Content", original_text)
    # dp-doc-flow: full-document block — allow page breaks inside (avoid orphan headings).
    return ["## Submitted Content", "", f'<div class="dp-doc-flow">{highlighted}</div>', ""]


_REWRITE_LEGEND = (
    "Green: improved — text the rewrite improved. "
    "Amber: edit yourself — kept because the system could not safely improve it."
)


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", str(text or "")) if p.strip()]


# Sentence-precise span highlight styles — MATCH the web rewrite view's reading
# (Rewrite.jsx bracketSpans: kind 'improved' -> green, 'kept' -> amber, else plain).
# allow-hardcode: presentation CSS keyed by machine span-kind codes, not a matching list.
_SPAN_KIND_STYLE = {
    "improved": "background:#ecfdf5;border-bottom:2px solid #16a34a;color:#065f46",
    "kept": "background:#fffbeb;border-bottom:2px dotted #f59e0b;color:#92400e",
}


def _valid_bracket_spans(spans: Any, text_len: int) -> List[Dict[str, Any]]:
    """Validate/clamp bracket_grounding_spans against the final text: keep only dicts with
    int offsets, 0 <= start < end <= len(final_text), a colorable kind, and no overlap with
    an earlier kept span (sorted by start; overlapping spans are skipped, matching the
    producer's non-overlap construction in bracket_grounding.py)."""
    rows = []
    for sp in (spans or []) if isinstance(spans, list) else []:
        if not isinstance(sp, dict):
            continue
        start, end = sp.get("start"), sp.get("end")
        if not (isinstance(start, int) and isinstance(end, int)):
            continue
        start, end = max(0, start), min(text_len, end)
        if not (0 <= start < end <= text_len):
            continue
        if str(sp.get("kind") or "") not in _SPAN_KIND_STYLE:
            continue  # unknown kinds render plain — same as the web view
        rows.append({"start": start, "end": end, "kind": str(sp["kind"])})
    rows.sort(key=lambda r: r["start"])
    out: List[Dict[str, Any]] = []
    for r in rows:
        if out and r["start"] < out[-1]["end"]:
            continue
        out.append(r)
    return out


def _escape_with_breaks(text: str) -> str:
    """HTML-escape a text chunk, preserving paragraph/line breaks inside the flow."""
    safe = html.escape(text)
    safe = safe.replace("\n\n", '<br/><span style="display:block;height:6pt"></span>')
    return safe.replace("\n", "<br/>")


def _span_highlighted_rewritten(final_text: str, spans: List[Dict[str, Any]]) -> str:
    """Render final_text with char-offset-precise green/amber inline spans (the web
    rewrite view's bracket_grounding_spans reading). '' when no valid spans."""
    text = str(final_text or "")
    valid = _valid_bracket_spans(spans, len(text))
    if not valid:
        return ""
    parts: List[str] = []
    cursor = 0
    for sp in valid:
        if sp["start"] > cursor:
            parts.append(_escape_with_breaks(text[cursor:sp["start"]]))
        style = _SPAN_KIND_STYLE[sp["kind"]]
        parts.append(f'<span style="{style}">{_escape_with_breaks(text[sp["start"]:sp["end"]])}</span>')
        cursor = sp["end"]
    if cursor < len(text):
        parts.append(_escape_with_breaks(text[cursor:]))
    return ('<div class="dp-doc-flow"><div class="dp-hero" style="border-left-color:#94a3b8">'
            f'<p class="dp-hero-sub">{html.escape(_REWRITE_LEGEND)}</p>'
            f'<p style="margin:0">{"".join(parts)}</p></div></div>')


def _highlighted_rewritten_content(sentence_comparison: List[Dict[str, Any]],
                                   final_text: str,
                                   bracket_grounding_spans: Any = None) -> List[str]:
    """'Rewritten Content' with green (improved) / amber (kept) highlights.

    Precision order (matches the web rewrite view):
    1. bracket_grounding_spans — char-offset spans over the FINAL text produced by
       poc/rewrite_v6/bracket_grounding.py (summary.bracket_grounding_spans); kind
       'improved' -> green, 'kept' -> amber, everything else plain. Sentence-precise.
    2. Paragraph-level fallback from sentence_comparison (whole-document before/after
       entry — NOT sentence-granular): rewritten paragraph GREEN when changed, AMBER
       when kept verbatim.
    3. Plain text block when neither source is usable."""
    span_html = _span_highlighted_rewritten(final_text, bracket_grounding_spans)
    if span_html:
        return ["## Rewritten Content", "", span_html, ""]

    rows = [r for r in (sentence_comparison or []) if isinstance(r, dict)]
    if not rows:
        return _document_section("Rewritten Content", final_text)

    # Build an original-paragraph set to detect kept-verbatim paragraphs.
    orig_text = ""
    for r in rows:
        if r.get("original"):
            orig_text = str(r.get("original"))
            break
    orig_paras = set(_split_paragraphs(orig_text)) if orig_text else set()

    # Determine the rendered text: prefer the entry's rewritten field, else final_text.
    rewritten_text = ""
    for r in rows:
        if r.get("rewritten"):
            rewritten_text = str(r.get("rewritten"))
            break
    rewritten_text = rewritten_text or str(final_text or "")
    new_paras = _split_paragraphs(rewritten_text)
    if not new_paras:
        return _document_section("Rewritten Content", final_text)

    # Whole-document 'changed' flag as a fallback signal when we can't diff paragraphs.
    doc_changed = any(bool(r.get("changed")) for r in rows)

    paras_html = []
    for para in new_paras:
        kept = para in orig_paras or (not orig_paras and not doc_changed)
        safe = html.escape(para)
        if kept:
            style = "background:#fffbeb;border-left:3px solid #f59e0b;padding:3pt 6pt;color:#92400e"
        else:
            style = "background:#ecfdf5;border-left:3px solid #16a34a;padding:3pt 6pt;color:#065f46"
        paras_html.append(f'<p style="{style};margin:0 0 6pt">{safe}</p>')
    if not paras_html:
        return _document_section("Rewritten Content", final_text)
    # dp-doc-flow: full-document block — allow page breaks inside (avoid orphan headings).
    block = ('<div class="dp-doc-flow"><div class="dp-hero" style="border-left-color:#94a3b8">'
             f'<p class="dp-hero-sub">{html.escape(_REWRITE_LEGEND)}</p>'
             + "".join(paras_html) + "</div></div>")
    return ["## Rewritten Content", "", block, ""]


def _author_review_card_section(summary: dict) -> List[str]:
    cards = summary.get("author_review_cards") if isinstance(summary.get("author_review_cards"), list) else []
    if not cards:
        return []
    lines = [
        "## Author Review Cards",
        "",
        "These candidate-specific items should be verified, replaced, or removed by the author before submission.",
        "",
    ]
    for index, card in enumerate(cards[:12], start=1):
        if not isinstance(card, dict):
            continue
        title = str(card.get("instruction") or card.get("target_text") or card.get("kind") or f"Review item {index}").strip()
        lines.append(f"### {index}. {title[:120]}")
        target = str(card.get("target_text") or "").strip()
        needed = str(card.get("user_input_needed") or "").strip()
        task = str(card.get("author_task") or "").strip()
        provenance = str(card.get("provenance") or "").replace("_", " ").strip()
        if provenance:
            lines.append(f"- Provenance: {provenance}")
        if target:
            lines.append(f"- Candidate detail: {target}")
        if needed:
            lines.append(f"- Author input needed: {needed}")
        if task:
            lines.append(f"- Author task: {task}")
        lines.append("")
    return lines


def _signal_label(name: str) -> str:
    return str(name or "").replace("_", " ").title()


def _highlight_placeholders(text: str) -> str:
    safe = html.escape(str(text or ""))
    return re.sub(
        r"(\[[^\[\]]+\])",
        r'<mark class="placeholder">\1</mark>',
        safe,
    )


def _manual_reason(item: dict) -> str:
    ftype = item.get("finding_type", "")
    if ftype == "low_specificity":
        return "Add a real example, source detail, or concrete observation. Rewriting alone cannot safely create specificity."
    if ftype in {"source_grounding", "polished_but_ungrounded", "uncited_claim", "uncited_in_body"}:
        return "Connect this claim to a source or soften/remove it if the evidence is not available."
    reason = item.get("reason", "")
    if reason.startswith("Detect classified"):
        return "Manual review is safer than automatic rewriting for this signal."
    return reason


def _top_user_actions(mitigation: dict, limit: int = 5) -> List[str]:
    drivers = mitigation.get("component_drivers") or []
    actions = []
    for driver in drivers:
        bucket = driver.get("bucket")
        component = driver.get("component", "")
        if bucket == "needs_source_or_example":
            if "unsupported" in component or "source" in component or "citation" in component:
                actions.append("Add source-backed support for broad or unsupported claims, or soften those claims.")
            elif "lived_detail" in component:
                actions.append("Add a real process detail from the author's experience.")
            else:
                actions.append("Narrow generic assertions to the exact submitted context.")
        elif bucket == "structure_guidance":
            actions.append("Revise paragraph openings and structure before retrying sentence-level rewrite.")
        elif bucket == "auto_rewrite":
            actions.append("Retry detector-gated sentence patches only after evidence and specificity gaps are addressed.")
        if len(actions) >= limit:
            break
    deduped = []
    for action in actions:
        if action not in deduped:
            deduped.append(action)
    return deduped[:limit]


def _guided_revision_checklist(mitigation: dict, limit: int = 5) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for driver in mitigation.get("component_drivers") or []:
        bucket = driver.get("bucket")
        component = str(driver.get("component", ""))
        if bucket not in {"needs_source_or_example", "structure_guidance"}:
            continue
        guidance = _driver_guidance(driver)
        if component == "unsupported_claim_risk":
            where = "Confident claims without visible support"
            add = "A source, concrete example, or softer claim wording"
            retry = "After claims are supported or narrowed"
        elif component == "source_grounding_risk":
            where = "Claims that cite theory but do not explain the source link"
            add = "Author/source name plus a sentence explaining how it supports the claim"
            retry = "After source-to-claim links are explicit"
        elif component == "citation_weakness_risk":
            where = "Paragraphs with weak or missing in-text citation linkage"
            add = "The missing citation or a clearer attribution phrase"
            retry = "After citation linkage is repaired"
        elif component in {"broad_claim_risk", "generic_assertion_risk"}:
            where = "Broad statements that could fit many drafts"
            add = "The exact context, method, participant group, or observation"
            retry = "After broad claims are narrowed to the draft context"
        elif component == "lived_detail_risk":
            where = "General claims about learning or practice"
            add = "A real process detail from the author's class, salon, lesson, or workflow"
            retry = "After lived/process detail is added"
        else:
            where = guidance["issue"]
            add = guidance["action"]
            retry = "After this manual revision is done"
        rows.append({
            "priority": guidance["signal"],
            "where": where,
            "add": add,
            "retry": retry,
        })
        if len(rows) >= limit:
            break
    return rows


def _bucket_plain_label(bucket: str) -> str:
    labels = {
        "needs_source_or_example": "Evidence or specificity",
        "structure_guidance": "Paragraph structure",
        "auto_rewrite": "Sentence predictability",
        "review_only": "Review only",
        "protected": "Protected text",
    }
    return labels.get(bucket, _signal_label(bucket))


def _driver_guidance(driver: dict) -> Dict[str, str]:
    component = str(driver.get("component", ""))
    bucket = driver.get("bucket", "")
    score = driver.get("score", 0)
    label = _signal_label(component)

    if component in {"generic_assertion_risk", "broad_claim_risk"}:
        issue = "The scan is reacting to broad claims that could fit many essays."
        action = "Limit the claim to the actual topic, class, unit, method, or observation."
    elif component == "unsupported_claim_risk":
        issue = "The scan sees claims that sound confident but are not visibly supported."
        action = "Add a citation, example, or qualifying phrase; remove the claim if support is unavailable."
    elif component in {"source_grounding_risk", "citation_weakness_risk"}:
        issue = "The text does not clearly connect key claims to source material."
        action = "Name the source, connect it to the claim, or add the missing in-text citation."
    elif component == "lived_detail_risk":
        issue = "The writing lacks concrete author/process detail."
        action = "Add a real project, participant, task, or workflow detail supplied by the author."
    elif component == "topk_pattern":
        issue = "Some sentences follow very predictable word-choice paths."
        action = "Review the sentence structure; only keep an edit if the final scan improves."
    elif component in {"signpost_paragraph_risk", "paragraph_uniformity_risk"}:
        issue = "Paragraphs may open or progress in a formulaic way."
        action = "Vary paragraph openings and merge, split, or reorder paragraphs where the argument feels templated."
    elif bucket == "auto_rewrite":
        issue = "This is a local wording signal that automatic rewrite can attempt."
        action = driver.get("mitigation") or "Rewrite the affected sentence and rescan."
    elif bucket == "needs_source_or_example":
        issue = "This signal needs information the system should not invent."
        action = driver.get("mitigation") or "Add author-supplied evidence, specificity, or source support."
    elif bucket == "structure_guidance":
        issue = "This signal needs paragraph-level revision."
        action = driver.get("mitigation") or "Revise structure before retrying sentence-level rewrite."
    else:
        issue = driver.get("mitigation") or "Review this signal manually."
        action = "Review the highlighted text and decide whether an edit is needed."

    try:
        score_text = f"{float(score):.1f}%"
    except (TypeError, ValueError):
        score_text = str(score)

    return {
        "focus": _bucket_plain_label(bucket),
        "signal": label,
        "score": score_text,
        "issue": issue,
        "action": action,
    }


def _mode_label(mode: str) -> str:
    return str(mode or "guided_revision").replace("_", " ").title()


def _result_label(
    *,
    no_text_change: bool,
    original_preserved: bool,
    improved_with_review: bool,
    mixed_result: bool,
    regressed: bool,
    improved: bool,
    converged: bool,
) -> str:
    if no_text_change:
        return "Author Input Needed"
    if original_preserved:
        return "Original Preserved"
    if improved_with_review:
        return "Improved With Review"
    if mixed_result:
        return "Mixed Result"
    if regressed:
        return "Review Needed"
    if improved:
        return "Revision Improved"
    if converged:
        return "Revision Complete"
    return "Review Needed"


def render_rewrite_report(
    summary: dict,
    sentence_comparison: List[Dict[str, Any]],
    ai_findings: List[Dict[str, Any]],
    verbose: bool = False,
    original_text: str = "",
    final_text: str = "",
    original_scan_report: Optional[dict] = None,
) -> str:
    """Render a standalone rewrite report as markdown.

    Args:
        summary: The rewrite summary dict from get_rewrite_summary_v2().
        sentence_comparison: Per-sentence before/after from ReportBuilder.
        ai_findings: The AI-flagged findings that triggered the rewrite.
        verbose: Include detailed per-sentence breakdown.
        original_scan_report: The ORIGINAL scan's full report JSON (report.json), whose
            scan_intelligence.document.segments drive the Submitted Content highlights.
            Optional — when absent the section falls back to a plain text block.

    Returns:
        Markdown string ready for PDF rendering.
    """
    lines: List[str] = []
    ts = time.strftime("%Y-%m-%d %H:%M")

    # ── Header ──────────────────────────────────────────────────────
    lines.append(
        '<div class="dp-masthead">'
        '<div><span class="dp-masthead-title">DraftProof</span>'
        '<span class="dp-masthead-sub">Rewrite Delivery Report</span></div>'
        f'<span class="dp-masthead-meta">Generated {html.escape(ts)}</span>'
        '</div>'
    )
    lines.append("")

    # ── Executive Summary ───────────────────────────────────────────
    lines.append("## Executive Summary")
    lines.append("")

    passes = summary.get("passes_completed", 0)
    converged = summary.get("converged", False)
    conv_reason = summary.get("convergence_reason", "")
    no_text_change = summary.get("no_text_change", False)
    no_text_change_reason = summary.get("no_text_change_reason", "")

    # Detect scan comparison.
    orig_scan = summary.get("detect_scan_original_saved") or summary.get("detect_scan_original", {})
    new_scan = summary.get("detect_scan_rewritten", {})
    attempted_scan = summary.get("detect_scan_attempted", {})
    rollback = summary.get("rollback_applied", False)
    mitigation = summary.get("mitigation_plan") or {}
    final_output_preserved = no_text_change

    if orig_scan and new_scan:
        detect_scores = summary.get("detect_scores") if isinstance(summary.get("detect_scores"), dict) else {}
        orig_badge = _badge(orig_scan)
        new_badge = _badge(new_scan)

        orig_ai = _first_metric(
            detect_scores.get("original_ai_authorship"),
            detect_scores.get("original_ai"),
            summary.get("original_risk"),
            _ai_score(orig_scan),
        ) or 0.0
        new_ai = _first_metric(
            detect_scores.get("rewritten_ai_authorship"),
            detect_scores.get("rewritten_ai"),
            summary.get("final_risk"),
            _ai_score(new_scan),
        ) or 0.0
        ai_delta = new_ai - orig_ai

        orig_wq = _first_metric(
            detect_scores.get("original_grounding_quality_risk"),
            _wq_score(orig_scan),
        ) or 0.0
        new_wq = _first_metric(
            detect_scores.get("rewritten_grounding_quality_risk"),
            _wq_score(new_scan),
        ) or 0.0
        wq_delta = new_wq - orig_wq
        orig_contribution = _scan_contribution(orig_scan)
        new_contribution = _scan_contribution(new_scan)
        orig_human, orig_transformation = contribution_pair(
            _first_metric(detect_scores.get("original_human_contribution"), orig_contribution.get("human")),
            _first_metric(detect_scores.get("original_ai_transformation"), orig_contribution.get("ai")),
        )
        new_human, new_transformation = contribution_pair(
            _first_metric(detect_scores.get("rewritten_human_contribution"), new_contribution.get("human")),
            _first_metric(detect_scores.get("rewritten_ai_transformation"), new_contribution.get("ai")),
        )

        orig_findings = orig_scan.get("findings", {})
        new_findings = new_scan.get("findings", {})
        o_total = _count_findings(orig_findings)
        n_total = _count_findings(new_findings)
        o_severity = _severity_score(orig_findings)
        n_severity = _severity_score(new_findings)
        o_review_burden = _review_burden(orig_findings)
        n_review_burden = _review_burden(new_findings)
        orig_axis = orig_scan.get("axis_scores", {})
        new_axis = new_scan.get("axis_scores", {})
        ai_improved = new_ai < orig_ai - 0.05
        ai_worse = new_ai > orig_ai + 0.05
        quality_improved = new_wq < orig_wq - 0.05
        quality_worse = new_wq > orig_wq + 0.05
        ai_first = summary.get("ai_first_mitigation") or {}
        ai_first_kept = bool(ai_first.get("kept")) and ai_improved
        findings_improved = n_total < o_total
        findings_worse = n_total > o_total
        severity_worse = n_severity > o_severity or n_review_burden > o_review_burden
        final_looks_original = (
            abs(new_ai - orig_ai) <= 0.05
            and abs(new_wq - orig_wq) <= 0.05
            and n_total == o_total
        )
        original_preserved = no_text_change or (rollback and final_looks_original)
        final_output_preserved = original_preserved
        score_improved = ai_improved or quality_improved or findings_improved
        score_worse = ai_worse or quality_worse
        improved_with_review = (
            not original_preserved
            and score_improved
            and not score_worse
            and not findings_worse
            and not severity_worse
            and (ai_improved or quality_improved)
        )
        mixed_result = (
            not original_preserved
            and not improved_with_review
            and score_improved
            and (score_worse or findings_worse or severity_worse)
        )
        scan_regressed = (
            not original_preserved
            and not improved_with_review
            and not mixed_result
            and (
                ai_worse
                or ((findings_worse or severity_worse) and not ai_improved and not quality_improved)
            )
        )
        improved = (
            not original_preserved
            and not mixed_result
            and not scan_regressed
            and (
                (ai_improved or quality_improved or findings_improved)
                and not findings_worse
                and not severity_worse
            )
        )
        result_label = _result_label(
            no_text_change=no_text_change,
            original_preserved=original_preserved,
            improved_with_review=improved_with_review,
            mixed_result=mixed_result,
            regressed=scan_regressed,
            improved=improved,
            converged=converged,
        )
        outcome = str(summary.get("outcome") or "")
        if ai_first_kept:
            result_label = "AI Mitigated"
        elif outcome == "ai_mitigated":
            result_label = "AI Mitigated"
        elif outcome == "partially_ai_mitigated":
            result_label = "Partially AI Mitigated"
        elif outcome == "cleanup_improved":
            result_label = "Cleanup Improved"
        elif outcome == "topk_blocked":
            result_label = "Top-k Blocked"

        # Plain-English result explanation (used as the hero panel's sub-line).
        if outcome == "cleanup_improved":
            _expl = "Review burden reduced, but AI-footprint signals are still high. Do not treat this as detector-safe mitigation."
        elif outcome == "partially_ai_mitigated":
            _expl = "AI-footprint signals reduced, but the result is still not guaranteed to pass external detectors."
        elif outcome == "topk_blocked":
            _expl = "Top-k predictability remains above the safe mark, so this rewrite is not detector-safe mitigation yet."
        elif outcome == "ai_mitigated" or ai_first_kept:
            _expl = "AI likelihood improved enough to keep the rewrite. Writing-quality or lower-severity changes are follow-up work."
        elif original_preserved and rollback and attempted_scan:
            _expl = "The attempted rewrite was not kept because the final scan did not improve."
        elif no_text_change:
            _expl = "DraftProof found revision opportunities, but the main issues need evidence, examples, or source context from the author."
        elif improved_with_review:
            _expl = "At least one measured risk signal improved. Review the new findings before keeping the final output."
        elif mixed_result:
            _expl = "Some risk scores improved, but the final scan added findings or increased review burden. Review the revision plan before keeping the final output."
        elif improved:
            _expl = "The final output reduced at least one measured risk signal."
        else:
            _expl = "Review the revision plan below before making another pass."

        # Two-column 'Original vs rewritten' scan cards (mirror the web comparison view):
        # per-column pattern label, fused-scale rating chip, the fused verdict card, the
        # 'How the writing reads' bars, and the independent risk axes. Falls back to the
        # legacy hero + KPI row when neither scan carries the V7 fused badge (older rewrites).
        _cmp_html = _executive_comparison_html(
            summary, review_required=_requires_author_review(summary))
        if _cmp_html:
            lines.append(_cmp_html)
            lines.append("")
        else:
            from .render_panels import rewrite_hero
            lines.append(rewrite_hero(
                result_label=result_label, explanation=_expl, outcome=outcome,
                ai_improved=ai_improved, score_worse=score_worse, original_preserved=original_preserved,
                orig_ai=_display_ai_score(orig_ai), new_ai=_display_ai_score(new_ai),
                orig_human=orig_human, new_human=new_human, orig_wq=orig_wq, new_wq=new_wq,
                o_total=o_total, n_total=n_total,
                orig_deep_scan=_deep_scan_pct(orig_scan), new_deep_scan=_deep_scan_pct(new_scan),
            ))
            lines.append("")

        # DraftProof score scale — the same reference the web card shows, so users can read
        # the fused scores above against measured human false-positive rates. Copy matches
        # the frontend i18n (report.submissionRisk.scale) verbatim; rendered as a markdown
        # table so it picks up the PDF's teal-header table styling.
        lines.append("### What the DraftProof score means")
        lines.append("")
        lines.append("| DraftProof score | Reads as | What we measured |")
        lines.append("|------------------|----------|------------------|")
        lines.append("| 0–32 | Low | ~6% or fewer real ESL students score this high — most human writing lands well under this |")
        lines.append("| 32–48 | Medium | uncommon for genuine human writing (0.4% false-positive rate measured) |")
        lines.append("| 48–65 | High | rare for genuine human writing (<1% false-positive rate measured) |")
        lines.append("| 65+ | Critical | essentially never seen in genuine human writing in our testing (0% false-positive rate measured) |")
        lines.append("")
        lines.append("*DraftProof's score is not comparable to Turnitin's percentage — the two tools measure different things on different scales.*")
        lines.append("")

        # Detect Scan Comparison table, Formula-Gap breakdown, and Findings-by-Severity
        # table intentionally NOT rendered — owner decision 2026-07-07: the two-column V7
        # cards above ARE the comparison; these raw internal-metric tables are redundant
        # noise for the user. (The outcome scorecard was likewise dropped, c410804d.)
    else:
        # Fallback to internal metrics if detect scan data not available
        imp_risk = summary.get("improvement_risk", 0)
        rollback = summary.get("rollback_applied", False)
        final_output_preserved = no_text_change or rollback
        result_label = _result_label(
            no_text_change=no_text_change,
            original_preserved=no_text_change or rollback,
            improved_with_review=False,
            mixed_result=False,
            regressed=False,
            improved=imp_risk > 0,
            converged=converged,
        )
        lines.append(f"**{result_label}**")
        lines.append("")
        # Outcome scorecard removed here too (owner decision 2026-07-07) — the result
        # label above remains the outcome line in this no-comparison branch.
    lines.append("")
    _orig_scan_for_highlight = summary.get("detect_scan_original_saved") or summary.get("detect_scan_original") or {}
    lines.extend(_highlighted_submitted_content(
        _orig_scan_for_highlight, original_text, original_scan_report=original_scan_report))
    lines.extend(_highlighted_rewritten_content(
        sentence_comparison, final_text,
        # Sentence-precise colour spans over the final text (production.py stores them in
        # the summary; the web view reads the same field).
        bracket_grounding_spans=summary.get("bracket_grounding_spans")))
    lines.extend(_author_review_card_section(summary))

    # ── User-facing action list ─────────────────────────────────────
    next_actions = _top_user_actions(mitigation)
    if next_actions:
        lines.append("## What To Do Next")
        lines.append("")
        for i, action in enumerate(next_actions, 1):
            lines.append(f"{i}. {action}")
        lines.append("")

    ceiling = summary.get("mitigation_ceiling") or {}
    if ceiling:
        lines.append("## Mitigation Ceiling")
        lines.append("")
        safe = ceiling.get("safe_auto_result") or {}
        frontier = ceiling.get("candidate_frontier") or {}
        evidence_gap = ceiling.get("author_evidence_gap") or {}
        primary = str(ceiling.get("primary_blocker") or "unknown").replace("_", " ")
        lines.append(f"**Primary blocker:** {primary}.")
        lines.append("")
        lines.append("| Signal | Current Safe Result | Frontier |")
        lines.append("|--------|--------------------:|---------:|")
        lines.append(
            f"| Human Contribution | `{safe.get('human_contribution', '-')}` | "
            f"best safe `{frontier.get('best_safe_human', '-')}` / best seen `{frontier.get('best_seen_human', '-')}` |"
        )
        lines.append(
            f"| AI Score | `{safe.get('ai_score', '-')}` | best seen `{frontier.get('best_seen_ai', '-')}` |"
        )
        lines.append(
            f"| AI Authorship | `{safe.get('ai_authorship', '-')}` | drop `{safe.get('ai_authorship_drop', '-')}` |"
        )
        lines.append(
            f"| AI Transformation | `{safe.get('ai_transformation', '-')}` | drop `{safe.get('ai_transformation_drop', '-')}` |"
        )
        lines.append(
            f"| Candidate Pool | {frontier.get('safe_candidates', 0)} safe / "
            f"{frontier.get('scanned_candidates', 0)} scanned | "
            f"{frontier.get('blocked_candidates', 0)} blocked |"
        )
        lines.append("")
        if evidence_gap.get("enabled"):
            estimate = evidence_gap.get("estimated_human_after_completion") or {}
            lines.append(
                f"Author evidence is still the Human Contribution ceiling: "
                f"{evidence_gap.get('slot_count', 0)} real-author slot(s) are needed. "
                f"Heuristic Human range after completion: `{estimate.get('low', '-')}`-`{estimate.get('high', '-')}`."
            )
            lines.append("")
        actions = ceiling.get("recommended_next_actions") or []
        if actions:
            lines.append("**Next actions:**")
            lines.append("")
            for action in actions[:4]:
                lines.append(f"- {str(action).replace('|', '·')}")
            lines.append("")

    intake = summary.get("author_evidence_intake") or {}
    if intake.get("questions"):
        lines.append("## Author Evidence Intake")
        lines.append("")
        lines.append(
            "Use these prompts to collect the real author-owned details needed before another mitigation pass. "
            "DraftProof can place confirmed answers, but it should not invent these anchors."
        )
        lines.append("")
        lines.append("| Anchor | Paragraph | What To Ask | Answer Type |")
        lines.append("|--------|-----------|-------------|-------------|")
        for question in (intake.get("questions") or [])[:8]:
            qid = str(question.get("id") or "").replace("|", "·")
            paragraph = question.get("paragraph_index")
            text = str(question.get("question") or "").replace("|", "·")
            answer_type = str(question.get("answer_type") or "").replace("_", " ")
            lines.append(f"| {qid} | {paragraph} | {text} | {answer_type} |")
        lines.append("")
        policy = intake.get("close_gap_policy") or []
        if policy:
            lines.append("**LLM close-gap policy:**")
            lines.append("")
            for item in policy[:4]:
                lines.append(f"- {str(item).replace('|', '·')}")
            lines.append("")

    discovery = summary.get("author_context_discovery") or {}
    if discovery.get("context_cards"):
        lines.append("## Author Context Discovery")
        lines.append("")
        lines.append(
            "Use this LLM-assisted intake to close the author-owned context gap. "
            "The LLM may ask and shape answers, but confirmed user answers are required before generation."
        )
        lines.append("")
        lines.append("| Anchor | Gap | Safe Answer Shape |")
        lines.append("|--------|-----|-------------------|")
        for card in (discovery.get("context_cards") or [])[:8]:
            anchor = str(card.get("anchor_id") or "").replace("|", "·")
            gap = str(card.get("llm_follow_up_question") or card.get("gap") or "").replace("|", "·")
            shape = str(card.get("safe_answer_shape") or "").replace("|", "·")
            lines.append(f"| {anchor} | {gap} | {shape} |")
        lines.append("")
        handoff = discovery.get("handoff_env") or {}
        if handoff:
            lines.append(
                f"Confirmed answers feed back through `{handoff.get('json')}` "
                f"or `{handoff.get('file')}`."
            )
            lines.append("")
        gates = discovery.get("success_gate") or []
        if gates:
            lines.append("**Discovery success gate:**")
            lines.append("")
            for gate in gates[:4]:
                lines.append(f"- {str(gate).replace('|', '·')}")
            lines.append("")

    source_search = summary.get("source_grounding_search") or {}
    if source_search.get("enabled"):
        lines.append("## Source Grounding Search")
        lines.append("")
        status = str(source_search.get("status") or "ready").replace("_", " ")
        provider = str(source_search.get("provider") or "tavily")
        lines.append(
            f"**Status:** {status}. Provider: `{provider}`. "
            "These are public-source candidates for grounding quality; they are not author-owned context."
        )
        lines.append("")
        targets = source_search.get("claim_targets") or []
        results_by_claim = {
            item.get("claim_id"): item
            for item in (source_search.get("results") or [])
            if isinstance(item, dict)
        }
        if targets:
            lines.append("| Claim | Confidence | Query | Candidate Sources |")
            lines.append("|-------|------------|-------|-------------------|")
            for target in targets[:8]:
                claim = str(target.get("claim") or "").replace("|", "·")
                query = str(target.get("query") or "").replace("|", "·")
                result = results_by_claim.get(target.get("id")) or {}
                confidence = str(result.get("source_confidence") or "-").replace("_", " ")
                source_links = []
                for source in (result.get("sources") or [])[:3]:
                    title = str(source.get("title") or source.get("url") or "source").replace("|", "·")
                    url = str(source.get("url") or "").strip()
                    if url:
                        source_links.append(f"[{title}]({url})")
                    elif title:
                        source_links.append(title)
                sources = "<br>".join(source_links) if source_links else "-"
                lines.append(f"| {claim[:220]} | `{confidence}` | `{query[:180]}` | {sources} |")
            lines.append("")
        policy = source_search.get("policy") or []
        if policy:
            lines.append("**Search policy:**")
            lines.append("")
            for item in policy[:4]:
                lines.append(f"- {str(item).replace('|', '·')}")
            lines.append("")

    integration = summary.get("author_evidence_integration") or {}
    if integration.get("enabled"):
        lines.append("## Author Evidence Integration")
        lines.append("")
        status = str(integration.get("status") or "awaiting_user_answers").replace("_", " ")
        lines.append(
            f"**Status:** {status}. "
            f"Answers received: `{integration.get('answer_count', 0)}`; "
            f"valid answers: `{integration.get('accepted_answers', 0)}`; "
            f"applied: `{integration.get('applied_answers', 0)}`."
        )
        lines.append("")
        candidates = integration.get("candidates") or []
        if candidates:
            lines.append("| Anchor | Status | Human | AI Authorship | Findings | Reason |")
            lines.append("|--------|--------|------:|--------------:|---------:|--------|")
            for item in candidates[:8]:
                anchor = str(item.get("anchor_id") or "").replace("|", "·")
                item_status = str(item.get("status") or "").replace("_", " ")
                human = item.get("human_contribution", "-")
                authorship = item.get("ai_authorship", "-")
                findings = item.get("findings", "-")
                reason = str(item.get("reason") or "").replace("|", "·")
                lines.append(
                    f"| {anchor} | {item_status} | `{human}` | `{authorship}` | `{findings}` | {reason} |"
                )
            lines.append("")

    if mitigation:
        score_targets = mitigation.get("score_mitigation_targets") or []
        if score_targets:
            lines.append("## Risk Score Mitigation Targets")
            lines.append("")
            lines.append(
                "These are the highest-value score levers to work on before retrying automatic rewrite. The target is to move each driver out of the strongest-risk band without inventing evidence."
            )
            lines.append("")
            lines.append("| Priority | Score Driver | Current | Target | Reduction Needed | Best Mitigation |")
            lines.append("|----------|--------------|--------:|-------:|-----------------:|-----------------|")
            for item in score_targets[:6]:
                component = str(item.get("component", "")).replace("_risk", "").replace("_", " ").title()
                priority = str(item.get("priority", "")).title()
                current = item.get("current_score", 0)
                target = item.get("target_score", 0)
                reduction = item.get("reduction_needed", 0)
                action = str(item.get("action", "")).replace("|", "·")
                lines.append(
                    f"| {priority} | {component} | `{float(current):.1f}%` | "
                    f"`{float(target):.1f}%` | `{float(reduction):.1f}%` | {action} |"
                )
            lines.append("")

        risk_actions = mitigation.get("risk_mitigation_actions") or []
        if risk_actions:
            lines.append("## Risk Mitigation Actions")
            lines.append("")
            lines.append(
                "Use these actions when automatic rewrite cannot safely create the missing evidence or context. They are written to reduce the next scan's strongest score drivers without inventing facts."
            )
            lines.append("")
            lines.append("| Priority | Action | Score Driver | User Input Needed | Safe Edit Pattern |")
            lines.append("|----------|--------|--------------|-------------------|-------------------|")
            for item in risk_actions[:6]:
                priority = str(item.get("priority", "")).title()
                title = str(item.get("title", "")).replace("|", "·")
                component = str(item.get("component", "")).replace("_risk", "").replace("_", " ").title()
                current = item.get("current_score", 0)
                target = item.get("target_score", 0)
                needed = str(item.get("user_input_needed", "")).replace("|", "·")
                pattern = _highlight_placeholders(
                    str(item.get("safe_edit_pattern", "")).replace("|", "·")
                )
                lines.append(
                    f"| {priority} | {title} | {component} `{float(current):.1f}% -> {float(target):.1f}%` | "
                    f"{needed} | {pattern} |"
                )
            lines.append("")

        marked_suggestions = mitigation.get("marked_content_suggestions") or []
        if marked_suggestions:
            lines.append("## Suggested Additions For Review")
            lines.append("")
            lines.append(
                "These are not kept automatically in the final output. Bracketed text marks new content that DraftProof is proposing as a structure only; replace it with verified source, example, or author detail before using."
            )
            lines.append("")
            lines.append("| Priority | Where | Target Text | Suggested Addition | Why It Helps | User Review |")
            lines.append("|----------|-------|-------------|--------------------|--------------|-------------|")
            for item in marked_suggestions[:6]:
                priority = str(item.get("priority", "")).title()
                where = str(item.get("where", "")).replace("|", "·")
                target_text = str(item.get("target_text") or item.get("evidence") or "").replace("|", "·")
                target_text = html.escape(target_text)
                suggestion = _highlight_placeholders(
                    str(item.get("suggested_addition", "")).replace("|", "·")
                )
                why = str(item.get("why_it_helps", "")).replace("|", "·")
                note = str(item.get("user_note", "")).replace("|", "·")
                lines.append(
                    f"| {priority} | {where} | {target_text} | {suggestion} | {why} | {note} |"
                )
            lines.append("")

        checklist = _guided_revision_checklist(mitigation)
        if final_output_preserved and checklist:
            lines.append("## Guided Revision Checklist")
            lines.append("")
            lines.append(
                "DraftProof preserved the original because automatic sentence edits cannot safely supply missing evidence or source context. Use this checklist before retrying rewrite."
            )
            lines.append("")
            lines.append("| Priority Signal | Where To Look | Add Or Change | Retry When |")
            lines.append("|-----------------|---------------|---------------|------------|")
            for row in checklist:
                lines.append(
                    f"| {row['priority'].replace('|', '·')} | "
                    f"{row['where'].replace('|', '·')} | "
                    f"{row['add'].replace('|', '·')} | "
                    f"{row['retry'].replace('|', '·')} |"
                )
            lines.append("")

    marked_mitigation = summary.get("marked_mitigation_rewrite") or {}
    completion = summary.get("author_evidence_completion") or {}
    if completion.get("draft_text"):
        lines.append("## Author Evidence Completion Draft")
        lines.append("")
        estimate = completion.get("estimated_human_after_completion") or {}
        current_human = completion.get("current_human_contribution")
        target_human = completion.get("target_human_contribution")
        if estimate:
            lines.append(
                f"Current Human Contribution is `{current_human}%`; target is `{target_human}%`. "
                f"If the bracketed slots are completed with real evidence, the estimated Human range is "
                f"`{estimate.get('low')}%-{estimate.get('high')}%`. This is a heuristic estimate; rescan after completion."
            )
        else:
            lines.append(
                "This draft marks where the author should add real source evidence, concrete context, or defensible reasoning. It is not an accepted final rewrite."
            )
        lines.append("")
        lines.append("```text")
        lines.append(str(completion.get("draft_text", "")).strip())
        lines.append("```")
        lines.append("")
        slots = completion.get("slots") or []
        if slots:
            lines.append("| Slot | Paragraph | Role | What To Supply |")
            lines.append("|------|-----------|------|----------------|")
            for slot in slots[:12]:
                instruction = str(slot.get("instruction") or "").replace("|", "·")
                role = str(slot.get("paragraph_role") or "").replace("_", " ")
                paragraph_index = slot.get("paragraph_index")
                lines.append(
                    f"| {slot.get('slot', '')} | {paragraph_index} | {role} | {instruction} |"
                )
            lines.append("")

    if marked_mitigation.get("draft_text"):
        lines.append("## Marked Mitigation Draft")
        lines.append("")
        lines.append(
            "This is a marked scaffold, not the accepted final output. Replace every bracketed marker with verified author evidence, source detail, or concrete context before using it."
        )
        lines.append("")
        lines.append("```text")
        lines.append(str(marked_mitigation.get("draft_text", "")).strip())
        lines.append("```")
        lines.append("")
        changes = marked_mitigation.get("changes") or []
        if changes:
            lines.append("| # | Signal | What To Supply |")
            lines.append("|---|--------|----------------|")
            for item in changes[:8]:
                component = str(item.get("component", "")).replace("_", " ")
                needed = str(item.get("user_input_needed") or "Verified author/source detail").replace("|", "·")
                lines.append(f"| {item.get('index', '')} | {component} | {needed} |")
            lines.append("")

        lines.append("## Revision Brief")
        lines.append("")
        counts = mitigation.get("counts") or {}
        auto_count = counts.get("auto_rewrite", 0)
        evidence_count = counts.get("needs_source_or_example", 0)
        structure_count = counts.get("structure_guidance", 0)

        if evidence_count or structure_count:
            lines.append(
                "The scan found sentence-level rewrite targets, but the strongest remaining risk drivers require author-supplied evidence, concrete context, or paragraph-level revision. Automatic rewriting should not invent those details."
            )
        elif auto_count:
            lines.append(
                "The main remaining risk is local sentence predictability. Automatic rewrite can attempt these targets, but each edit still needs a rescan gate."
            )
        else:
            lines.append(
                "No safe automatic rewrite target was found. Use the guidance below as a manual revision checklist."
            )
        lines.append("")

        lines.append("| Focus Area | Signal | Score | What It Means | Concrete Revision |")
        lines.append("|------------|--------|------:|---------------|-------------------|")
        for driver in (mitigation.get("component_drivers") or [])[:6]:
            item = _driver_guidance(driver)
            lines.append(
                f"| {item['focus']} | {item['signal']} | `{item['score']}` | "
                f"{item['issue'].replace('|', '·')} | {item['action'].replace('|', '·')} |"
            )
        lines.append("")

        sentence_scope_text = (
            f"{auto_count} sentence-level target(s) identified; no sentence edits kept in the final output"
            if final_output_preserved
            else f"{auto_count} sentence-level target(s) identified for detector-gated editing"
        )
        scope_parts = [
            sentence_scope_text,
            f"{evidence_count} evidence/specificity item(s)",
            f"{structure_count} structure item(s)",
        ]
        lines.append("**Rewrite scope:** " + ", ".join(scope_parts) + ".")
        lines.append("")

        patterns = mitigation.get("reference_patterns") or []
        if patterns:
            lines.append("## Reference Revision Examples")
            lines.append("")
            lines.append(
                "For learning and revision guidance only. Do not submit these patterns as-is; replace the placeholders with the author's own evidence, source, and context."
            )
            lines.append("")
            for i, pattern in enumerate(patterns, 1):
                focus = str(pattern.get("focus", "Revision pattern")).replace("|", "·")
                excerpt = str(pattern.get("flagged_excerpt", "")).replace("|", "·")
                instead = _highlight_placeholders(
                    str(pattern.get("instead_of", "")).replace("|", "·")
                )
                try_pattern = _highlight_placeholders(
                    str(pattern.get("try_pattern", "")).replace("|", "·")
                )
                why = str(pattern.get("why", "")).replace("|", "·")
                note = str(pattern.get("application_note", "")).replace("|", "·")
                lines.append(f"### Pattern {i}: {focus}")
                lines.append("")
                if excerpt:
                    lines.append(f"**Flagged excerpt:** “{html.escape(excerpt)}”")
                    lines.append("")
                if instead:
                    lines.append(f"**Instead of:** {instead}")
                    lines.append("")
                lines.append(f"**Try this pattern:** {try_pattern}")
                lines.append("")
                if why:
                    lines.append(f"**Why this helps:** {why}")
                    lines.append("")
                if note:
                    lines.append(f"**How to apply:** {note}")
                    lines.append("")

    # ── Revision Plan ───────────────────────────────────────────────
    lines.append("## Revision Plan")
    lines.append("")

    if mitigation:
        counts = mitigation.get("counts") or {}
        mode = _mode_label(mitigation.get("primary_mode", "guided_revision"))
        lines.append(f"**Recommended path:** {mode}")
        lines.append("")
        lines.append("| Bucket | Count | Meaning |")
        lines.append("|--------|------:|---------|")
        bucket_labels = {
            "auto_rewrite": (
                "Sentence-level targets identified; no sentence edits kept in the final output"
                if final_output_preserved
                else "Sentence-level targets identified for detector-gated editing"
            ),
            "needs_source_or_example": "Needs author source, citation, example, or concrete detail",
            "structure_guidance": "Needs paragraph/section structure revision",
            "review_only": "Review signal; no automatic edit",
            "protected": "Protected material; preserve verbatim",
        }
        bucket_names = {
            "auto_rewrite": "Sentence Targets",
            "needs_source_or_example": "Needs Evidence",
            "structure_guidance": "Structure Work",
            "review_only": "Review Only",
            "protected": "Protected",
        }
        for key, label in bucket_labels.items():
            lines.append(f"| {bucket_names[key]} | {counts.get(key, 0)} | {label} |")
        lines.append("")

        drivers = mitigation.get("component_drivers") or []
        if drivers:
            lines.append("**Revision focus:**")
            lines.append("")
            lines.append("| Revision Focus | Signal Strength | Suggested Action |")
            lines.append("|--------|------:|------------|")
            for item in drivers[:8]:
                signal = str(item.get("component", "")).replace("_risk", "").replace("_", " ").title()
                score = item.get("score", 0)
                fix = str(item.get("mitigation", "")).replace("|", "·")
                lines.append(f"| {signal} | `{score:.1f}%` | {fix} |")
            lines.append("")

    # Manual actions remaining
    manual = summary.get("manual_actions", [])
    if manual:
        lines.append("### Manual Review Required")
        lines.append("")
        lines.append("These items need author judgment or source-backed detail before another rewrite attempt:")
        lines.append("")
        lines.append("| # | Risk | Finding | Reason | Evidence |")
        lines.append("|--:|:----:|---------|--------|----------|")
        for i, m in enumerate(manual, 1):
            ftype = m.get("finding_type", "?")
            risk = m.get("risk_level", "?").upper()
            reason = _manual_reason(m).replace("|", "·")
            evidence = m.get("evidence", "").replace("|", "·")
            lines.append(f"| {i} | {risk} | {ftype} | {reason} | {evidence} |")
        lines.append("")

    attempted_changes = [
        sc for sc in (summary.get("attempted_sentence_comparison") or [])
        if sc.get("orig_sentence", "") != sc.get("new_sentence", "")
    ]
    if rollback and attempted_changes:
        lines.append("## Attempted Rewrite")
        lines.append("")
        lines.append("These edits passed local rewrite checks, but the final full scan regressed, so DraftProof preserved the original final output.")
        lines.append("")
        lines.append("| # | Original | Attempted Output |")
        lines.append("|--:|----------|------------------|")
        for i, sc in enumerate(attempted_changes[:12], 1):
            orig_text = sc.get("orig_sentence", "").replace("\n", " ").replace("|", "·")
            new_text = sc.get("new_sentence", "").replace("\n", " ").replace("|", "·")
            lines.append(f"| {i} | {orig_text} | {new_text} |")
        lines.append("")

    manual_suggestions = summary.get("manual_suggestions") or []
    if manual_suggestions:
        lines.append("## Manual Suggestions")
        lines.append("")
        lines.append("These candidates were not automatically kept. Review them manually before using them.")
        lines.append("")
        lines.append("| # | Finding | Reason | Original | Suggested |")
        lines.append("|--:|---------|--------|----------|-----------|")
        for i, item in enumerate(manual_suggestions[:16], 1):
            finding = str(item.get("finding_type", "?")).replace("|", "·")
            reason = str(item.get("rejection_reason", "")).replace("|", "·")
            orig_text = str(item.get("original_sentence", "")).replace("\n", " ").replace("|", "·")
            suggestion = str(item.get("suggested_sentence", "")).replace("\n", " ").replace("|", "·")
            lines.append(f"| {i} | {finding} | {reason} | {orig_text} | {suggestion} |")
        lines.append("")

    # Protected / Review-only
    protected = summary.get("protected_actions", [])
    if protected:
        lines.append(f"**Protected findings (skipped):** {len(protected)}")
        lines.append("")

    # ── Sentence Rewrites ───────────────────────────────────────────
    changed = [
        sc for sc in sentence_comparison
        if sc.get("orig_sentence", "") != sc.get("new_sentence", "")
    ]

    if not changed:
        pass
    else:
        lines.append("## Sentence Changes")
        lines.append("")
        lines.append(f"**{len(changed)} sentence(s) rewritten.**")
        lines.append("")
        lines.append("| # | Change | Signal Ratio | Original | Final Output |")
        lines.append("|--:|:-----------:|:------------:|----------|-----------|")
        for i, sc in enumerate(changed, 1):
            orig_tier = sc.get("orig_tier", "?")
            new_tier = sc.get("new_tier", "?")
            tier_change = f"{orig_tier} → {new_tier}" if orig_tier != new_tier else orig_tier
            orig_text = sc.get("orig_sentence", "").replace("\n", " ").replace("|", "·")
            new_text = sc.get("new_sentence", "").replace("\n", " ").replace("|", "·")
            orig_top10_s = sc.get("orig_top10", 0)
            new_top10_s = sc.get("new_top10", 0)
            ratio_change = f"{orig_top10_s:.0%} → {new_top10_s:.0%}"
            lines.append(f"| {i} | {tier_change} | {ratio_change} | {orig_text} | {new_text} |")
        lines.append("")

    # ── Worked Examples (the teaching lesson — leads the coaching) ────
    # The "math teacher works the problem on the board": each item shows a generic claim, a grounded
    # version of the SAME claim, and why it is more grounded. The user studies the move, then applies it
    # with their own real detail. Placed ABOVE the "do it yourself" coaching so the worked solution is
    # the headline, not the homework.
    # allow-hardcode: the strings below are human-facing REPORT COPY (section heading + instructions),
    # not a detect/scoring/matching word-list. The examples themselves are generated by the showcase
    # stage (rewrite_v6/predictability_showcase.py); this block only renders them.
    showcase = summary.get("predictability_showcase")
    if isinstance(showcase, list) and showcase:
        lines.append("## Worked Examples — How To Ground a Claim")
        lines.append("")
        lines.append(
            "Each example takes a general claim and shows a more grounded version of the SAME claim. "
            "Study the move, then apply it with YOUR own real detail (a real number, name, or moment) "
            "-- the example specifics are illustrative, meant to be replaced."
        )
        lines.append("")
        for i, item in enumerate(showcase, 1):
            if not isinstance(item, dict):
                continue
            generic = str(item.get("sentence") or "").strip()
            grounded = str(item.get("suggestion") or "").strip()
            why = str(item.get("why") or "").strip()
            if not generic or not grounded:
                continue
            lines.append(f"### Example {i}")
            lines.append(f"- **General claim:** {generic}")
            lines.append(f"- **More grounded:** {grounded}")
            if why:
                lines.append(f"- **Why it works:** {why}")
            lines.append("")

    # ── Register / Polish Coaching ───────────────────────────────────
    # Honest coaching (NOT a score lever): the spots that read most machine-polished, plus a worked
    # plain/polished contrast from the user's own text. The copy states plainly it will not lower the
    # external detector score -- the value is a draft the user can finish in their own plainer voice.
    # allow-hardcode: the strings below are human-facing REPORT COPY (section heading + instructions),
    # not a detect/scoring/matching word-list. Sentence selection is done by the content-agnostic
    # register_score signal in rewrite_v6/register_coaching.py; this block only renders that result.
    register_coaching = summary.get("register_coaching") if isinstance(summary.get("register_coaching"), dict) else {}
    if register_coaching:
        lines.append("## Make It Sound Like You")
        lines.append("")
        note = str(register_coaching.get("note", "")).strip()
        if note:
            lines.append(note)
            lines.append("")
        contrast = register_coaching.get("worked_contrast") if isinstance(register_coaching.get("worked_contrast"), dict) else {}
        polished = (contrast.get("polished") or {}) if isinstance(contrast, dict) else {}
        plain = (contrast.get("plain") or {}) if isinstance(contrast, dict) else {}
        if polished.get("text") and plain.get("text"):
            lines.append("**From your own draft -- same writer, two registers:**")
            lines.append("")
            lines.append(f"- More machine-polished: \"{str(polished.get('text')).strip()}\"")
            lines.append(f"- Plainer / more direct: \"{str(plain.get('text')).strip()}\"")
            lines.append("")
        offenders = register_coaching.get("offenders") if isinstance(register_coaching.get("offenders"), list) else []
        if offenders:
            lines.append("**Lines that read most polished -- try saying them the way you actually would:**")
            lines.append("")
            for item in offenders[:4]:
                if isinstance(item, dict) and item.get("text"):
                    lines.append(f"- \"{str(item.get('text')).strip()}\"")
            lines.append("")
        if register_coaching.get("rhythm_even") is True:
            lines.append(
                "Your sentences also run at a fairly even length and rhythm -- mixing in some shorter, "
                "blunter sentences in your own voice reads less machine-smooth."
            )
            lines.append("")

    # ── Legend ───────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by DraftProof Rewrite Pipeline*")
    lines.append(f"*{ts}*")
    lines.append("")

    return "\n".join(lines)
