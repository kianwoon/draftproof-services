#!/usr/bin/env python3
"""Holdout A/B: deep-scan PROPORTION vs MEAN sentence score as the fusion's
second signal — the RAID re-test required to overturn the 2026-07-09 rejection.

Protocol (mirrors the 2026-07-09 sweep note in weights.json tier_authority):
  - 70/30 stratified split (by label+attack), seed 42, over score_subset_mean.py
    rows carrying {composite, modal_proportion, modal_mean}.
  - On the TUNE split, for each representation: sweep fusion weight w in
    {0.4..0.9} over fused = (1-w)*composite + w*signal*100, pick (w, thr) that
    maximizes TPR at human FPR <= target (default 1%).
  - On the HOLDOUT split, report AUC and TPR@(the tuned thr) separately for
    attack=none and attack=paraphrase, at the achieved holdout FPR.

The representation only wins if it beats proportion ON THE HOLDOUT — in-sample
wins were exactly what the July rejection caught. modal_mean is min-max
stretched per doc set? NO — used raw (0-1), same scale as proportion.

Usage:
    python poc/calibration/raid_benchmark/holdout_compare.py --scores scores_mean.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def thr_at_fpr(human_scores: list[float], target: float) -> float:
    """Lowest threshold with human FPR (fraction >= thr) <= target."""
    s = sorted(human_scores)
    k = int(len(s) * target)  # allowed false positives
    return s[len(s) - k] if k > 0 else max(s) + 1e-9


def evaluate(rows: list[dict], signal_key: str, weights: list[float],
             fpr_target: float, seed: int) -> dict:
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r["label"], r.get("attack"))].append(r)
    tune, hold = [], []
    for _, docs in sorted(strata.items(), key=lambda kv: str(kv[0])):
        docs = docs[:]
        rng.shuffle(docs)
        cut = int(len(docs) * 0.7)
        tune += docs[:cut]
        hold += docs[cut:]

    def fused(r, w):
        return (1 - w) * r["composite"] + w * r[signal_key] * 100.0

    best = None
    for w in weights:
        hum = [fused(r, w) for r in tune if r["label"] == 0]
        ai = [fused(r, w) for r in tune if r["label"] == 1]
        thr = thr_at_fpr(hum, fpr_target)
        tpr = sum(1 for x in ai if x >= thr) / len(ai)
        if best is None or tpr > best["tune_tpr"]:
            best = {"w": w, "thr": thr, "tune_tpr": tpr}

    w, thr = best["w"], best["thr"]
    hum_h = [fused(r, w) for r in hold if r["label"] == 0]
    ai_h = [(r, fused(r, w)) for r in hold if r["label"] == 1]
    hold_fpr = sum(1 for x in hum_h if x >= thr) / len(hum_h)
    out = {"signal": signal_key, "w": w, "thr": round(thr, 2),
           "tune_tpr": round(best["tune_tpr"], 4),
           "hold_fpr": round(hold_fpr, 4),
           "hold_auc": round(auc([s for _, s in ai_h], hum_h), 4),
           "n_tune": len(tune), "n_hold": len(hold)}
    for attack in ("none", "paraphrase"):
        grp = [s for r, s in ai_h if r.get("attack") == attack]
        out[f"hold_tpr_{attack}"] = round(
            sum(1 for x in grp if x >= thr) / len(grp), 4) if grp else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", type=str, default="0.4,0.5,0.6,0.7,0.8,0.9")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.scores.read_text().splitlines() if l.strip()]
    rows = [r for r in rows
            if r.get("composite") is not None
            and r.get("modal_proportion") is not None
            and r.get("modal_mean") is not None]
    print(f"n={len(rows)} usable rows  (target FPR {args.fpr:.1%}, seed {args.seed})")
    weights = [float(w) for w in args.weights.split(",")]
    for key in ("modal_proportion", "modal_mean"):
        res = evaluate(rows, key, weights, args.fpr, args.seed)
        print(json.dumps(res))


if __name__ == "__main__":
    main()
