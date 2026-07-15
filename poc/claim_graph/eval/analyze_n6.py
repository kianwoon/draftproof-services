"""Aggregate the N6 eval rows into the §B calibration numbers (report Q a-f).

Pure/deterministic over the two N6 JSONL row files (gaming + real-citation); no
LLM, no network. Two analyses:

  * ``analyze_gaming`` — the M4-vs-N6 interrogatability comparison on group G.
    Headline (a): did the verified-only credit DROP the gaming score from the M4
    0.725/86.7%-high baseline? Compares the pipeline's verified-only composite to
    the same-graph M4-formula recompute (``interrog_current_recompute``).
  * ``analyze_real`` — verified-fire separation on the real-citation slice, split
    by ``case_type`` (entailed vs misattributed): fairness (false-unverified on
    entailed = grounded work mislabeled), contradicted precision, and the observed
    entailment/contradiction score distributions for threshold validation.

The committed M4 baseline (``docs/plans/phase1_m4_calibration_report.md``) is
0.725 mean composite / 0.867 high-rate on G — carried as a constant for the delta.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
GAMING_ROWS = os.path.join(_HERE, "eval_rows_n6_gaming.jsonl")
REAL_ROWS = os.path.join(_HERE, "eval_rows_n6_real.jsonl")

# Committed M4 baseline (report constants — NOT recomputed here).
M4_GAMING_MEAN = 0.725
M4_GAMING_HIGH_RATE = 0.867


def load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if not r.get("error")]


def _mean(xs: list[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _rate(rows: list[dict[str, Any]], pred: Callable[[dict[str, Any]], bool]
          ) -> Optional[float]:
    return round(sum(1 for r in rows if pred(r)) / len(rows), 4) if rows else None


def analyze_gaming(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """(a) Gaming interrogatability: N6 verified-only vs M4-formula vs baseline."""
    n = len(rows)
    pipe = [r["interrogatability"]["composite"] for r in rows]
    pipe_band = [r["interrogatability"]["band"] for r in rows]
    cur = [r.get("interrog_current_recompute", {}).get("composite") for r in rows]
    cur_band = [r.get("interrog_current_recompute", {}).get("band") for r in rows]
    n_verified = sum(r.get("entailment", {}).get("n_verified", 0) for r in rows)
    pipe_mean = _mean(pipe)
    pipe_high = round(sum(1 for b in pipe_band if b == "high") / n, 4) if n else None
    return {
        "n": n,
        "m4_baseline": {"mean_composite": M4_GAMING_MEAN, "high_rate": M4_GAMING_HIGH_RATE},
        "n6_same_graph_m4_formula": {  # all-specifics credit on THIS run's graph
            "mean_composite": _mean(cur),
            "high_rate": round(sum(1 for b in cur_band if b == "high") / n, 4) if n else None,
        },
        "n6_pipeline_verified_only": {  # the gaming fix (entailment ON)
            "mean_composite": pipe_mean,
            "high_rate": pipe_high,
            "band_counts": {b: sum(1 for x in pipe_band if x == b)
                            for b in ("low", "moderate", "high")},
        },
        "drop_vs_m4_baseline_mean": (round(M4_GAMING_MEAN - pipe_mean, 4)
                                     if pipe_mean is not None else None),
        "high_rate_eliminated": pipe_high == 0.0,
        "verified_claims_in_gaming": n_verified,  # must be 0 (fake sources unresolved)
    }


def _scores(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        out.extend(r.get("entailment", {}).get(key, []) or [])
    return out


def analyze_real(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """(b)-(e) real-citation separation, fairness, contradicted precision, dists."""
    ent = [r for r in rows if r.get("case_type") == "entailed"]
    mis = [r for r in rows if r.get("case_type") == "misattributed"]

    def _ev(r: dict[str, Any]) -> dict[str, Any]:
        return r.get("entailment", {})

    # Docs where a claim was actually linked+scored (verdict emitted).
    ent_tested = [r for r in ent if _ev(r).get("verdict_statuses")]
    mis_tested = [r for r in mis if _ev(r).get("verdict_statuses")]
    return {
        "n_entailed": len(ent), "n_misattributed": len(mis),
        # (b) separation
        "entailed_verified_fire_rate": _rate(ent, lambda r: _ev(r).get("verified_fired")),
        "misattributed_verified_fire_rate": _rate(mis, lambda r: _ev(r).get("verified_fired")),
        # (c) fairness — grounded work mislabeled unverified (of docs actually tested)
        "entailed_tested": len(ent_tested),
        "false_unverified_rate_entailed_tested": _rate(
            ent_tested, lambda r: not _ev(r).get("verified_fired")),
        # (d) contradicted precision
        "contradicted_fire_count_entailed": sum(_ev(r).get("n_contradicted", 0) for r in ent),
        "contradicted_fire_count_misattributed": sum(_ev(r).get("n_contradicted", 0) for r in mis),
        "misattributed_tested": len(mis_tested),
        "misattributed_tested_all_unverified": all(
            not _ev(r).get("verified_fired") and _ev(r).get("n_contradicted", 0) == 0
            for r in mis_tested),
        # (e) observed score distributions (threshold validation input)
        "entailed_entailment_scores": sorted(_scores(ent, "entailment_scores")),
        "misattributed_entailment_scores": sorted(_scores(mis, "entailment_scores")),
        "entailed_contradiction_scores": sorted(_scores(ent, "contradiction_scores")),
        "misattributed_contradiction_scores": sorted(_scores(mis, "contradiction_scores")),
    }


def analyze() -> dict[str, Any]:
    return {
        "gaming": analyze_gaming(load_rows(GAMING_ROWS)),
        "real_citation": analyze_real(load_rows(REAL_ROWS)),
    }


def main() -> None:
    print(json.dumps(analyze(), indent=2))


if __name__ == "__main__":
    main()
