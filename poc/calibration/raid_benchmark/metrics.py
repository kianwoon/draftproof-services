#!/usr/bin/env python3
"""Compute detection metrics from a RAID ``scores.jsonl``.

Reports, for DraftProof's fused ``ai_likelihood`` (0-100):

  * ROC-AUC          : human (label 0, attack==none) vs AI attack=="none".
  * TPR @ FPR        : thresholds set on the HUMAN score distribution at
                       FPR = 0.5% / 1% / 5%, then TPR measured on AI attack=="none".
  * Robustness (same thresholds) applied to AI attack=="paraphrase" — the number that
    shows how much a paraphrase attack degrades detection.

Threshold rule: for target FPR alpha, t = the (1-alpha) empirical quantile of human
scores (conservative 'higher' interpolation), so at most alpha of humans score >= t.
Achieved FPR is reported alongside because with a few hundred humans the discrete
grid cannot hit 0.5%/1% exactly.

Usage:
    python calibration/raid_benchmark/metrics.py                 # reads scores.jsonl
    python calibration/raid_benchmark/metrics.py --scores PATH
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SCORES = HERE / "scores.jsonl"
_FPR_TARGETS = (0.005, 0.01, 0.05)
_NONE = "none"
_PARAPHRASE = "paraphrase"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"scores not found: {path}  (run score_subset.py first)")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def _roc_auc(labels: list[int], scores: list[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:  # noqa: BLE001 - sklearn absent or degenerate; use rank AUC
        pos = [s for l, s in zip(labels, scores) if l == 1]
        neg = [s for l, s in zip(labels, scores) if l == 0]
        if not pos or not neg:
            return float("nan")
        wins = sum(1 for p in pos for n in neg if p > n)
        ties = sum(1 for p in pos for n in neg if p == n)
        return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _quantile_higher(vals: list[float], q: float) -> float:
    """(1-alpha) quantile with 'higher' interpolation (conservative threshold)."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    if q <= 0:
        return s[0]
    if q >= 1:
        return s[-1]
    import math
    idx = math.ceil(q * (len(s) - 1) - 1e-9)
    idx = min(max(idx, 0), len(s) - 1)
    # ensure at most alpha exceed the threshold: bump to the exact rank
    return s[idx]


def _tpr(scores: list[float], thr: float) -> float:
    return sum(1 for s in scores if s >= thr) / len(scores) if scores else float("nan")


def _fpr(human: list[float], thr: float) -> float:
    return sum(1 for s in human if s >= thr) / len(human) if human else float("nan")


def compute(rows: list[dict]) -> dict:
    human = [r["fused_score"] for r in rows if r["label"] == 0]
    ai_none = [r["fused_score"] for r in rows
               if r["label"] == 1 and (r.get("attack") == _NONE)]
    ai_para = [r["fused_score"] for r in rows
               if r["label"] == 1 and (r.get("attack") == _PARAPHRASE)]

    auc = _roc_auc([0] * len(human) + [1] * len(ai_none),
                   human + ai_none) if (human and ai_none) else float("nan")

    table = []
    for alpha in _FPR_TARGETS:
        thr = _quantile_higher(human, 1.0 - alpha)
        table.append({
            "fpr_target": alpha,
            "threshold": round(thr, 4),
            "fpr_achieved": round(_fpr(human, thr), 4),
            "tpr_none": round(_tpr(ai_none, thr), 4),
            "tpr_paraphrase": round(_tpr(ai_para, thr), 4),
        })

    return {
        "n": {"human": len(human), "ai_none": len(ai_none), "ai_paraphrase": len(ai_para)},
        "means": {
            "human": round(sum(human) / len(human), 2) if human else None,
            "ai_none": round(sum(ai_none) / len(ai_none), 2) if ai_none else None,
            "ai_paraphrase": round(sum(ai_para) / len(ai_para), 2) if ai_para else None,
        },
        "roc_auc_human_vs_ai_none": round(auc, 4),
        "tpr_at_fpr": table,
    }


def _print(res: dict) -> None:
    n, mu = res["n"], res["means"]
    print("=" * 68)
    print("DraftProof fused AI detector — RAID subset")
    print("=" * 68)
    print(f"n: human={n['human']}  ai_none={n['ai_none']}  ai_paraphrase={n['ai_paraphrase']}")
    print(f"mean fused: human={mu['human']}  ai_none={mu['ai_none']}  "
          f"ai_paraphrase={mu['ai_paraphrase']}")
    print(f"ROC-AUC (human vs AI attack=none): {res['roc_auc_human_vs_ai_none']}")
    print("-" * 68)
    print(f"{'FPR tgt':>8} | {'thresh':>8} | {'FPR ach':>8} | "
          f"{'TPR none':>9} | {'TPR para':>9}")
    print("-" * 68)
    for row in res["tpr_at_fpr"]:
        print(f"{row['fpr_target'] * 100:>6.1f}% | {row['threshold']:>8.2f} | "
              f"{row['fpr_achieved'] * 100:>6.2f}% | {row['tpr_none'] * 100:>7.1f}% | "
              f"{row['tpr_paraphrase'] * 100:>7.1f}%")
    print("=" * 68)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--json", action="store_true", help="Also print machine-readable JSON.")
    args = ap.parse_args()
    res = compute(_load(args.scores))
    _print(res)
    if args.json:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
