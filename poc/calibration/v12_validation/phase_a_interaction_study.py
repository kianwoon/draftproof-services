"""V8 Phase A — interaction feature study: content-humanness x surface-AIness.

`ai_paraphrased` = an LLM fully rewriting a human essay. Its single-doc
signature is human-inherited CONTENT wearing an AI SURFACE. This study
measures whether interaction PRODUCTS of already-captured V7 signals separate
`ai_paraphrased` from `ai_generated_like`, BEFORE any production wiring.

Candidate features: 4 content-humanness proxies x 2 surface-AIness proxies
= 8 products, named f"{content}_x_{surface}", value = content * surface
clamped to [0, 1]. Pure arithmetic over captured signals — NO embeddings,
NO model loads, NO network (the MiniLM family was already gate-rejected).

Gate methodology is SHARED with the prior study — `rank_auc` (Mann-Whitney,
tie-averaged) and `evaluate_gate` (direction-aware winner by effect size,
effective AUC = max(auc, 1-auc), directional ESL subgroup check) are imported
from calibration.v12_validation.paraphrase_feature_study, never duplicated.

Output JSON is numbers-only (test_no_text_leakage.py enforces this on every
committed *.json in this directory); prose lives under `_notes`.

Usage:
    cd poc && python -m calibration.v12_validation.phase_a_interaction_study \
        --captures <captures.jsonl> [--auc-gate 0.70] [--esl-gate 0.60] [--out <path>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from calibration.v12_validation.paraphrase_feature_study import (
    evaluate_gate,
    pearson,
    quantile,
    rank_auc,
    _round,
)

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"
DEFAULT_OUT = HERE / "phase_a_interaction_study.json"

_CLASSES = ("student_owned", "ai_assisted_polished", "ai_paraphrased", "ai_generated_like")

# ── Candidate interaction features ─────────────────────────────────────────
# Each proxy: (signals_dict, calibrated_detector_score) -> float | None.
# These are references to CAPTURED V7 signal fields (schema keys), not a
# content phrase list — the values are numeric signals computed upstream.

CONTENT_PROXIES: Dict[str, Callable] = {
    "specificity": lambda s, det: s.get("specificity_score"),
    "spec_student_ev": lambda s, det: s.get("specificity_student_evidence"),
    "voice_presence": lambda s, det: None if s.get("author_voice_absence") is None else 1.0 - s["author_voice_absence"],
    "grounded": lambda s, det: None if s.get("grounding_gap") is None else 1.0 - s["grounding_gap"],
}
SURFACE_PROXIES: Dict[str, Callable] = {
    "smooth": lambda s, det: s.get("sentence_smoothness"),
    "det": lambda s, det: det,
}

# Sorted for deterministic iteration/tie-breaking everywhere downstream.
FEATURE_NAMES = tuple(sorted(
    f"{c}_x_{srf}" for c in CONTENT_PROXIES for srf in SURFACE_PROXIES
))


def compute_features(row: dict) -> Dict[str, Optional[float]]:
    """The 8 interaction products for one capture row.

    A feature is None (skipped-and-counted by run_study) when either of its
    two inputs is None. Products are clamped to [0, 1]."""
    signals = row.get("v7_signals") or {}
    det = row.get("calibrated_detector_score")
    feats: Dict[str, Optional[float]] = {}
    for c_name, c_fn in CONTENT_PROXIES.items():
        c_val = c_fn(signals, det)
        for s_name, s_fn in SURFACE_PROXIES.items():
            s_val = s_fn(signals, det)
            name = f"{c_name}_x_{s_name}"
            if c_val is None or s_val is None:
                feats[name] = None
            else:
                feats[name] = min(1.0, max(0.0, float(c_val) * float(s_val)))
    return feats


# ── Study driver ───────────────────────────────────────────────────────────

def run_study(rows: List[dict], prof_by_key: Dict[str, str],
              auc_gate: float = 0.70, esl_gate: float = 0.60) -> dict:
    """Phase A study over capture rows.

    rows: capture rows ({"label", "doc_key", "v7_signals",
          "calibrated_detector_score", ...}).
    prof_by_key: doc_key -> "higher"/"lower" SCoCESLE proficiency group
          (student_owned rows only; unmapped keys are simply absent from the
          ESL subgroup check).
    Returns a numbers-only result dict (prose under `_notes`)."""
    docs: List[dict] = []
    skip_counts: Dict[str, int] = {name: 0 for name in FEATURE_NAMES}
    for row in rows:
        feats = compute_features(row)
        for name in FEATURE_NAMES:
            if feats[name] is None:
                skip_counts[name] += 1
        docs.append({"label": row["label"],
                     "group": prof_by_key.get(row["doc_key"]) if row["label"] == "student_owned" else None,
                     "features": feats})

    def values_for(label: str, feature: str) -> List[float]:
        return [d["features"][feature] for d in docs
                if d["label"] == label and d["features"][feature] is not None]

    def values_group(group: str, feature: str) -> List[float]:
        return [d["features"][feature] for d in docs
                if d["group"] == group and d["features"][feature] is not None]

    # AUC tables in the exact shape evaluate_gate consumes (shared gate).
    auc_one_vs_one: Dict[str, Dict[str, Optional[float]]] = {}
    esl_subgroup: Dict[str, dict] = {}
    for feature in FEATURE_NAMES:
        para = values_for("ai_paraphrased", feature)
        auc_one_vs_one[feature] = {
            "ai_paraphrased_vs_ai_generated_like": _round(
                rank_auc(para, values_for("ai_generated_like", feature))),
            "ai_paraphrased_vs_ai_assisted_polished": _round(
                rank_auc(para, values_for("ai_assisted_polished", feature))),
        }
        higher = values_group("higher", feature)
        lower = values_group("lower", feature)
        esl_subgroup[feature] = {
            "lower_vs_higher_auc": _round(rank_auc(lower, higher)),
            "n_higher": len(higher),
            "n_lower": len(lower),
        }

    gate_input = {"auc_one_vs_one": auc_one_vs_one, "esl_subgroup_check": esl_subgroup}
    gate = evaluate_gate(gate_input, auc_gate, esl_gate)  # SHARED gate math

    # Feature correlation matrix (pairwise-complete rows).
    corr: Dict[str, Dict[str, Optional[float]]] = {}
    for f1 in FEATURE_NAMES:
        corr[f1] = {}
        for f2 in FEATURE_NAMES:
            pairs = [(d["features"][f1], d["features"][f2]) for d in docs
                     if d["features"][f1] is not None and d["features"][f2] is not None]
            corr[f1][f2] = _round(pearson([p[0] for p in pairs], [p[1] for p in pairs]))

    winner = None
    if gate["best_feature"] is not None:
        w = gate["best_feature"]
        all_vals = [d["features"][w] for d in docs if d["features"][w] is not None]
        winner = {
            "name": w,
            "direction": gate["direction"],
            "effective_auc_vs_generated": gate["best_feature_effective_auc_vs_ai_generated_like"],
            "auc_vs_polished": gate["best_feature_auc_vs_ai_assisted_polished"],
            "esl_directional_auc": gate["best_feature_esl_directional_flagging_auc"],
            "p10": _round(quantile(all_vals, 0.10)),
            "p90": _round(quantile(all_vals, 0.90)),
        }

    class_counts = {c: sum(1 for d in docs if d["label"] == c) for c in _CLASSES}

    # Fail-closed ESL coverage guard (layered here, on top of the shared
    # evaluate_gate, which treats an absent ESL check as passing): a PASS is
    # only valid when the winner's ESL check was actually computable from
    # BOTH proficiency subgroups. Otherwise the check is vacuous — the exact
    # covert-ESL-detector failure the gate exists to prevent.
    esl_coverage = {"n_higher": 0, "n_lower": 0}
    if gate["best_feature"] is not None:
        cov = esl_subgroup[gate["best_feature"]]
        esl_coverage = {"n_higher": cov["n_higher"], "n_lower": cov["n_lower"]}
    gate_verdict = "PASS" if gate["gate_passed"] else "FAIL"
    gate_fail_reason = None
    if gate_verdict == "PASS" and not (esl_coverage["n_higher"] > 0 and esl_coverage["n_lower"] > 0):
        gate_verdict = "FAIL"
        gate_fail_reason = "esl_check_unverifiable_insufficient_subgroup_coverage"

    # allow-hardcode: `_notes` is human-reviewed report annotation describing
    # the winning composition and methodology for the reader of the JSON —
    # it is never matched against, scored, or used as detection logic.
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "auc_gate_threshold": auc_gate,
        "esl_gate_threshold": esl_gate,
        "n_rows": len(docs),
        "class_counts": class_counts,
        "skip_counts": skip_counts,
        "auc_one_vs_one": auc_one_vs_one,
        "esl_subgroup_check": esl_subgroup,
        "feature_correlation_matrix": corr,
        "gate_detail": gate,
        "winner": winner,
        "esl_coverage": esl_coverage,
        "gate_verdict": gate_verdict,
        "gate_fail_reason": gate_fail_reason,
        "_notes": {
            "winning_composition": (
                None if winner is None else
                "Interaction product of a content-humanness proxy and a surface-AIness "
                f"proxy: {winner['name']} ({winner['direction']}). High content-humanness "
                "combined with high surface-AIness is the ai_paraphrased signature "
                "(human-inherited content wearing an AI surface)."),
            "methodology": (
                "8 interaction features = 4 content proxies (specificity, "
                "spec_student_ev, voice_presence, grounded) x 2 surface proxies "
                "(smooth, det); rank_auc + evaluate_gate imported from "
                "paraphrase_feature_study (shared gate methodology, not duplicated)."),
        },
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_prof_by_key() -> Dict[str, str]:
    """doc_key (sha256(text)[:16]) -> 'higher'/'lower' for SCoCESLE
    student_owned corpus rows, mirroring the false-AI diagnosis mapping."""
    from calibration.retune.intake import DEFAULT_SCOCESLE
    from calibration.v12_validation.measure import _resolve_text

    prof_by_key: Dict[str, str] = {}
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    for row in manifest["rows"]:
        if row["label"] != "student_owned":
            continue
        text = _resolve_text(row, DEFAULT_SCOCESLE)
        if not text:
            continue
        group = None
        for d in DEFAULT_SCOCESLE.glob("*proficiency*"):
            if (d / row["file"]).exists():
                dname = d.name.lower()
                if "higher" in dname:
                    group = "higher"
                elif "lower" in dname:
                    group = "lower"
                break
        if group:
            doc_key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            prof_by_key[doc_key] = group
    return prof_by_key


def _load_captures(path: Path) -> List[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    # Import-order gotcha: the sys.path shim must load BEFORE any detect.* /
    # measure import chain (mirrors paraphrase_feature_study.run_study).
    import calibration.measure_end_to_end  # noqa: F401

    ap = argparse.ArgumentParser(description="V8 Phase A — interaction feature study")
    ap.add_argument("--captures", type=Path, required=True,
                    help="JSONL capture rows (label, doc_key, v7_signals, calibrated_detector_score)")
    ap.add_argument("--auc-gate", type=float, default=0.70,
                    help="minimum winner effective AUC vs ai_generated_like to pass")
    ap.add_argument("--esl-gate", type=float, default=0.60,
                    help="max directional lower-vs-higher-proficiency AUC allowed")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = _load_captures(args.captures)
    prof_by_key = _build_prof_by_key()
    result = run_study(rows, prof_by_key, auc_gate=args.auc_gate, esl_gate=args.esl_gate)

    out = args.out or DEFAULT_OUT
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(json.dumps({"gate_verdict": result["gate_verdict"],
                      "gate_fail_reason": result["gate_fail_reason"],
                      "esl_coverage": result["esl_coverage"],
                      "winner": result["winner"],
                      "skip_counts": result["skip_counts"]}, indent=2))
    print(f"study -> {out}")
    return 0 if result["gate_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
