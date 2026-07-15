"""Phase-0 Authorship Evidence Level composer (display-layer only).

Governing spec: docs/plans/credible_authorship_assessment_v2.md
    §H  Phase-0 row     — recast EXISTING signals into Evidence Levels 0-2.
    §10 Evidence Levels — report "Authorship evidence level: N/5".
    §C  confidence/coverage contract — banded assessment_confidence (NOT a
        second percentage), typed coverage objects, enumerated limitation codes.
    §B  signal lifecycle — this signal ships at status ADVISORY, fusion_weight 0.

PHASE 0 IS DISPLAY-ONLY. This composer NEVER touches the tier, the
ai_likelihood_score, any gate, or any scoring path (lifecycle.scoring_enabled is
hard-False; §B double-gate forbids promotion without a fairness gate +
calibration version). It only RE-PRESENTS values other engines already emit:

    - ai_pattern lens  : badge tier + headline_confidence + statistical source
    - grounding lens   : grounding_diagnosis band (poc/detect/grounding_diagnosis.py)
    - citation lens    : submission_risk citation axis (poc/detect/submission_risk.py)
    - reasoning lens   : critical_thinking_control band (LLM-judged -> capped Moderate)

No NEW numeric thresholds are introduced — levels derive from the banded values
the signals already emit. Evidence Levels 3-5 (context / external / provenance)
are honestly reported as not assessable in this version (no claim graph yet).

Callsite: poc/report/report.py, AFTER the deep-scan heatmap sync (same seam as
headline_confidence) so the ai_signal_deberta tile is the FINAL one. The attach
is fail-open and gated by DRAFTPROOF_AUTHORSHIP_EVIDENCE_LEVELS (default ON);
"0"/"false" reverts byte-identically (field absent on every surface).

NO module-level imports from rewrite_v6/* (circular-import guard — that class of
bug crashed every prod rewrite task, commit 2a409b88).
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

MAX_LEVEL_ASSESSABLE = 2  # honest Phase-0 cap: no claim graph -> no Levels 3-5.

# ── grounding band (poc/detect/grounding_diagnosis.py::_band) — this is an
# AI-STYLE grounding-GAP band, higher = weaker grounding.
_GROUNDING_BANDS = {"likely_human", "some_texture", "moderate", "strong", "very_strong"}
# ── critical-thinking band (poc/detect/critical_thinking.py::_band).
_CRITICAL_BANDS = {"strong_control", "acceptable_control", "weak_control",
                   "high_dependency", "very_high_dependency"}
_CRITICAL_UNASSESSABLE = {"insufficient_data"}
# ── submission_risk citation axis level (poc/detect/submission_risk.py::_level_from_risk).
_CITATION_LEVELS = {"low", "medium", "high"}
# ── statistical (Level-1) signal sources (poc/report/builder.py badge.signal_source).
_STATISTICAL_SOURCES = {"v7_fused", "deberta_authoritative", "perplexity_layer3"}

_CONF_ORDER = {"low": 0, "moderate": 1, "high": 2}


def _band_from(order: int) -> str:
    for k, v in _CONF_ORDER.items():
        if v == order:
            return k
    return "low"


def _weakest(bands: list[str]) -> str:
    """Aggregate lens confidences into overall (worst-wins)."""
    if not bands:
        return "low"
    return _band_from(min(_CONF_ORDER.get(b, 0) for b in bands))


def _generator_window() -> dict[str, Any]:
    """DATA-derived generator coverage window (§C) — NEVER a hardcoded literal.

    1. DRAFTPROOF_MODAL_CHECKPOINT env tag (the runtime checkpoint identity, the
       same source poc/calibration/v7_deberta_academic_calibrate.py labels with).
    2. else the committed weights.json deep_scan_calibration provenance date.
    3. else honest null + note (fail honest — never fabricate "July 2026").
    """
    ckpt = os.environ.get("DRAFTPROOF_MODAL_CHECKPOINT")
    if ckpt and str(ckpt).strip():
        return {"type": "generator_window", "value": str(ckpt).strip(),
                "source": "modal_checkpoint_env"}
    try:
        import json
        import pathlib
        wpath = pathlib.Path(__file__).resolve().parent.parent / "detect_v7" / "weights.json"
        blob = wpath.read_text(encoding="utf-8")
        dates = re.findall(r"20\d\d-\d\d-\d\d", blob)
        if dates:
            latest = max(dates)  # ISO dates sort lexicographically
            ym = latest[:7].replace("-", "_")
            return {"type": "generator_window", "value": f"through_{ym}",
                    "source": "weights_provenance"}
    except Exception:
        pass
    return {"type": "generator_window", "value": None, "note": "provenance unavailable"}


def _ai_pattern_lens(badge: dict) -> Optional[dict[str, Any]]:
    tier = str(badge.get("tier") or "").strip().lower()
    if not tier:
        return None
    source = str(badge.get("signal_source") or "").strip().lower()
    hc = badge.get("headline_confidence")
    reasons: list[str] = []
    # Statistical classifier fired -> otherwise moderate; a real fused/authoritative
    # read is our strongest Phase-0 evidence, but a contradicted headline
    # (headline_confidence present) caps it at low.
    if isinstance(hc, dict) and hc.get("level"):
        conf = "low"
        reasons.append("headline_contradicted")
    elif source in ("v7_fused", "deberta_authoritative"):
        conf = "high"
        reasons.append("calibrated_classifier")
    else:
        conf = "moderate"
        reasons.append("perplexity_fallback")
    return {"band": tier, "assessment_confidence": conf, "reasons": reasons,
            "statistical_source": source or None}


def _grounding_lens(badge: dict) -> Optional[dict[str, Any]]:
    gd = badge.get("grounding_diagnosis")
    if not isinstance(gd, dict):
        return None
    band = str(gd.get("band") or "").strip().lower()
    if band not in _GROUNDING_BANDS:
        return None
    low_cov = gd.get("low_coverage") is True
    reasons: list[str] = []
    # Deterministic rule signal -> high unless text was too short to be sure.
    if low_cov:
        conf = "low"
        reasons.append("limited_text")
    else:
        conf = "high"
        reasons.append("deterministic_rule_signal")
    lens = {"band": band, "assessment_confidence": conf, "reasons": reasons}
    label = gd.get("primary_driver_label")
    if isinstance(label, str) and label:
        lens["primary_driver_label"] = label
    return lens


def _citation_lens(badge: dict) -> Optional[dict[str, Any]]:
    sr = badge.get("submission_risk")
    if not isinstance(sr, dict):
        return None
    axes = sr.get("axes")
    if not isinstance(axes, dict):
        return None
    cit = axes.get("citation")
    if not isinstance(cit, dict):
        return None
    level = str(cit.get("level") or "").strip().lower()
    if level not in _CITATION_LEVELS:
        return None
    return {"level": level, "assessment_confidence": "moderate",
            "reasons": ["citation_axis_rule_signal"]}


def _reasoning_lens(badge: dict) -> Optional[dict[str, Any]]:
    ct = badge.get("critical_thinking_control")
    if not isinstance(ct, dict):
        return None
    band = str(ct.get("band") or "").strip().lower()
    if band not in _CRITICAL_BANDS:
        return None
    # LLM-judged lens: project memory records rating OVER-JUDGES fluent human
    # text, so this lens is CAPPED at moderate and defaults low until calibrated.
    conf = "moderate" if ct.get("low_coverage") is not True else "low"
    return {"band": band, "assessment_confidence": conf,
            "reasons": ["llm_judged_capped_moderate"]}


def compute_evidence_level(
    badge: Optional[dict], report_fields: Optional[dict]
) -> Optional[dict[str, Any]]:
    """Return the Phase-0 authorship_evidence_levels dict, or None on a
    non-dict badge. Fail-open: any per-lens malformation is skipped, never
    raised — this annotator must never break a report build.
    """
    if not isinstance(badge, dict):
        return None

    lenses: dict[str, Any] = {}
    for name, fn in (
        ("ai_pattern", _ai_pattern_lens),
        ("grounding", _grounding_lens),
        ("citation", _citation_lens),
        ("reasoning", _reasoning_lens),
    ):
        try:
            lens = fn(badge)
        except Exception:
            lens = None
        lenses[name] = lens if isinstance(lens, dict) else {"available": False}

    # ── Level mapping (§10) from EXISTING banded values only. ──
    # Level 1 = statistical indication present (ai_pattern lens assessable, i.e.
    #           a real classifier / perplexity read produced a tier).
    # Level 2 = content-weakness lenses assessable (grounding gap band present;
    #           enriched by citation + reasoning lenses).
    ai_pattern_ok = lenses["ai_pattern"].get("band") is not None
    content_ok = lenses["grounding"].get("band") is not None
    if content_ok and ai_pattern_ok:
        level = 2
    elif ai_pattern_ok:
        level = 1
    elif content_ok:
        # Content weakness without a statistical read still exceeds "nothing".
        level = 1
    else:
        level = 0

    # ── Overall assessment confidence (§C — worst-wins over present lenses). ──
    present_confs = [
        lens["assessment_confidence"]
        for lens in lenses.values()
        if isinstance(lens, dict) and lens.get("assessment_confidence")
    ]
    assessment_confidence = _weakest(present_confs) if present_confs else "low"
    confidence_reasons: list[str] = []
    if any(isinstance(l, dict) and l.get("assessment_confidence") == "low"
           for l in lenses.values()):
        confidence_reasons.append("one_or_more_lenses_low_confidence")
    confidence_reasons.append("no_assignment_context")  # Phase 0: brief absent.

    # ── Coverage (§C typed objects). ──
    coverage = [
        _generator_window(),
        {"type": "context_availability", "value": "assignment_brief_absent"},
    ]

    # ── Limitations (§C enumerated codes). ──
    limitations = ["no_assignment_context"]
    _low_cov = False
    for lname in ("grounding", "reasoning"):
        src = badge.get({"grounding": "grounding_diagnosis",
                         "reasoning": "critical_thinking_control"}[lname])
        if isinstance(src, dict) and src.get("low_coverage") is True:
            _low_cov = True
    if _low_cov:
        limitations.append("short_paragraphs_low_confidence")
    if any(c.get("type") == "generator_window" and c.get("value") is None for c in coverage):
        limitations.append("outside_training_coverage")

    return {
        "level": level,
        "max_level_assessable": MAX_LEVEL_ASSESSABLE,
        "lenses": lenses,
        "assessment_confidence": assessment_confidence,
        "confidence_reasons": confidence_reasons,
        "coverage": coverage,
        "limitations": limitations,
        # §B signal lifecycle — ADVISORY. HARD-False scoring; the ADVISORY->
        # SCORING promotion is double-gated (owner sign-off + a CI guard
        # asserting scoring_enabled => fairness_gate_passed && calibration_version).
        "lifecycle": {
            "status": "advisory",
            "scoring_enabled": False,
            "calibration_version": None,
            "fairness_gate_passed": None,
        },
    }
