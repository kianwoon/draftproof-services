"""Fusion-weight sweep for the DraftProof fused detector on the RAID subset.

Recomputes fused = (1 - w) * composite + w * modal_proportion * 100 for a range
of Modal weights w, and reports AUC + TPR @ fixed FPR (clean AI and paraphrase)
for each. Answers empirically: does raising the Modal (desklib) weight toward
1.0 improve accuracy vs the gate-validated 0.4/0.6 (i.e. w=0.6)?

Input: scores.jsonl rows with {label, attack, composite (0-100), modal_proportion (0-1)}.
Run from poc/:  python calibration/raid_benchmark/sweep.py --scores calibration/raid_benchmark/scores.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUC (P(pos > neg)), ties = 0.5. No sklearn dependency."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def _threshold_at_fpr(human: list[float], target_fpr: float) -> float:
    """Lowest threshold whose human FPR (fraction of humans scored >= thr) <= target."""
    for thr in sorted(set(human)):
        fpr = sum(1 for h in human if h >= thr) / len(human)
        if fpr <= target_fpr:
            return thr
    return max(human) + 1e-9


def _tpr(scores: list[float], thr: float) -> float:
    return sum(1 for s in scores if s >= thr) / len(scores) if scores else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--weights", type=str, default="0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--fpr", type=float, default=0.01, help="target FPR for the TPR columns")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.scores.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("composite") is not None and r.get("modal_proportion") is not None]
    if not rows:
        raise SystemExit("no rows with composite+modal_proportion — re-run score_subset.py first.")

    weights = [float(w) for w in args.weights.split(",")]
    print(f"n={len(rows)}  (target FPR = {args.fpr:.1%})")
    print("=" * 74)
    print(f"{'w_modal':>8} | {'AUC':>7} | {'thr@FPR':>8} | {'FPR ach':>8} | {'TPR none':>9} | {'TPR para':>9}")
    print("-" * 74)
    best = None
    for w in weights:
        def fused(r):
            return (1.0 - w) * float(r["composite"]) + w * float(r["modal_proportion"]) * 100.0
        human = [fused(r) for r in rows if r["label"] == 0]
        ai_none = [fused(r) for r in rows if r["label"] == 1 and r["attack"] == "none"]
        ai_para = [fused(r) for r in rows if r["label"] == 1 and r["attack"] == "paraphrase"]
        auc = _auc(ai_none, human)
        thr = _threshold_at_fpr(human, args.fpr)
        fpr_ach = sum(1 for h in human if h >= thr) / len(human)
        tpr_none = _tpr(ai_none, thr)
        tpr_para = _tpr(ai_para, thr)
        # Balanced objective: mean of clean + paraphrase TPR at the fixed FPR.
        obj = (tpr_none + tpr_para) / 2.0
        marker = ""
        if best is None or obj > best[0]:
            best = (obj, w)
            marker = ""
        print(f"{w:>8.2f} | {auc:>7.4f} | {thr:>8.2f} | {fpr_ach:>7.2%} | {tpr_none:>8.1%} | {tpr_para:>8.1%}")
    print("=" * 74)
    print(f"Best mean(TPR_none, TPR_para) @ FPR {args.fpr:.1%}: w_modal={best[1]:.2f}")
    print("(w=0.60 is the current gate-validated production weight.)")


if __name__ == "__main__":
    main()
