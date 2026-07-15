"""Aggregate M4 eval rows into the §B calibration numbers (report Q a-f).

Pure/deterministic over ``eval_rows.jsonl``; no LLM, no network. Emits a JSON
blob of per-group band distributions, means, the verified-weighted A/B, the
false-thin fairness analog, substitutability separation + its correlation with
``generic_assertion_risk``, origin-map sanity, and RECOMMENDED (not promoted)
thresholds. ``main`` prints the JSON; the human report cites these numbers.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ROWS = os.path.join(_HERE, "eval_rows.jsonl")
GROUPS = ("H", "A", "G")
_BANDS = ("low", "moderate", "high")


def load_rows(path: str = _DEFAULT_ROWS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ok(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not r.get("error")]


def _mean(xs: list[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _band_dist(rows: list[dict[str, Any]], getter) -> dict[str, Any]:
    counts = {b: 0 for b in _BANDS}
    n = 0
    for r in rows:
        b = getter(r)
        if b in counts:
            counts[b] += 1
            n += 1
    frac = {b: (round(counts[b] / n, 4) if n else None) for b in _BANDS}
    return {"n": n, "counts": counts, "frac": frac}


def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs2 = [p[0] for p in pairs]
    ys2 = [p[1] for p in pairs]
    mx, my = sum(xs2) / len(xs2), sum(ys2) / len(ys2)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs2))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys2))
    return round(num / (dx * dy), 4) if dx and dy else None


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _ok(rows)
    by_group = {g: [r for r in rows if r.get("group") == g] for g in GROUPS}
    out: dict[str, Any] = {"n_total": len(rows),
                           "n_by_group": {g: len(by_group[g]) for g in GROUPS}}

    # (a)+(b) interrogatability CURRENT formula
    cur: dict[str, Any] = {}
    for g in GROUPS:
        gr = by_group[g]
        cur[g] = {
            "mean_composite": _mean([r["interrogatability"].get("composite") for r in gr]),
            "band": _band_dist(gr, lambda r: r["interrogatability"].get("band")),
            "mean_unverified_specific_rate": _mean(
                [r["interrogatability"].get("unverified_specific_rate") for r in gr]),
        }
    out["interrogatability_current"] = cur

    # (c) verified-weighted A/B + false-thin fairness analog
    ver: dict[str, Any] = {}
    for g in GROUPS:
        gr = by_group[g]
        ver[g] = {
            "mean_composite": _mean([r["interrog_verified"].get("composite") for r in gr]),
            "band": _band_dist(gr, lambda r: r["interrog_verified"].get("band")),
        }
    out["interrogatability_verified"] = ver
    # Fairness analog: fraction of H banded 'low' (= genuine work mislabeled thin).
    out["false_thin_rate_H"] = {
        "current": (cur["H"]["band"]["frac"]["low"]),
        "verified": (ver["H"]["band"]["frac"]["low"]),
    }
    # G high-interrogatability rate = the feared gaming failure.
    out["gaming_high_rate_G"] = {
        "current": cur["G"]["band"]["frac"]["high"],
        "verified": ver["G"]["band"]["frac"]["high"],
    }

    # (d) substitutability separation + agreement with generic_assertion_risk
    subst: dict[str, Any] = {}
    for g in GROUPS:
        gr = by_group[g]
        subst[g] = {
            "mean": _mean([r["substitutability"].get("value") for r in gr]),
            "band": _band_dist(gr, lambda r: r["substitutability"].get("band")),
            "mean_generic_assertion_risk": _mean([r.get("generic_assertion_risk") for r in gr]),
        }
    out["substitutability"] = subst
    out["substitutability_vs_generic_pearson"] = _pearson(
        [r["substitutability"].get("value") for r in rows],
        [r.get("generic_assertion_risk") for r in rows])

    # (e) origin-map sanity
    origin: dict[str, Any] = {}
    for g in GROUPS:
        gr = by_group[g]
        prim: dict[str, int] = {}
        for r in gr:
            om = r.get("origin_map") or {}
            p = om.get("primary_origin")
            if p:
                prim[p] = prim.get(p, 0) + 1
        origin[g] = {
            "mean_unsupported_assertion_rate": _mean(
                [(r.get("origin_map") or {}).get("unsupported_assertion_rate") for r in gr]),
            "primary_origin_counts": dict(sorted(prim.items())),
        }
    out["origin_map"] = origin

    # graph/extraction actuals
    out["extraction"] = {
        "mean_llm_calls": _mean([r.get("llm_calls") for r in rows]),
        "total_llm_calls": sum(int(r.get("llm_calls") or 0) for r in rows),
        "mean_claims_real": _mean([(r.get("graph") or {}).get("claims_real") for r in rows]),
        "docs_over_soft_cap": sum(1 for r in rows if r.get("over_soft_cap")),
    }
    return out


def main() -> None:
    rows = load_rows()
    print(json.dumps(analyze(rows), indent=2))


if __name__ == "__main__":
    main()
