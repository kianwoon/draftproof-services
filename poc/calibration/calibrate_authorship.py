#!/usr/bin/env python3
"""Measure how well the DraftProof-conservative score (and each of its 8 components) separates
HUMAN-authored from AI-generated text. MEASURE-FIRST: this changes nothing in the scorer; it reports
the baseline so recalibration is data-driven, not blind.

Reads authorship-labeled cases (field "authorship": "human"|"ai") from authorship_cases/ AND any
turnitin_cases/ that carry an authorship field. For each it computes our signals via the same
_scan_report the turnitin calibrator uses, then reports per-component human-vs-AI discrimination
(AUC + best 1-D threshold accuracy) and the composite score's confusion at the LIVE tier cutoffs.

Honest by construction: refuses a verdict below MIN_PER_CLASS per class (small corpus = overfit risk).

Run: ~/.pyenv/versions/3.11.0/bin/python3 poc/calibration/calibrate_authorship.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DIRS = [HERE / "authorship_cases", HERE / "turnitin_cases"]
OUT = REPO / "test_output" / "_authorship_calibration.json"

MIN_PER_CLASS = 3
# The 8 components feeding ai_likelihood_score (layer3_scoring.py:1302) + the composite itself.
COMPONENTS = [
    "predictability", "topk_pattern", "generic_phrase_density", "burstiness_risk",
    "repeated_sentence_structure_risk", "generic_assertion_risk", "qualifying_text_ai_density",
    "balanced_hedging_risk",
]
# Live tier cutoffs (layer3_scoring.py ~:1538). ai_likelihood_score is reported 0..100 in the badge,
# so use 0..100 cutoffs (= the 0.32/0.48/0.65 fractions ×100).
GREEN_MAX = 32.0
TIERS = [("green", 0.0, 32.0), ("amber", 32.0, 48.0), ("orange", 48.0, 65.0), ("red", 65.0, 1000.0)]


def _load_env() -> None:
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def _signals(text: str) -> dict:
    from rewrite_v3.pipeline import _scan_report
    rep = _scan_report(text)
    b = rep.get("ai_risk_badge") or {}
    c = b.get("ai_components") or {}
    out = {k: c.get(k) for k in COMPONENTS}
    out["ai_likelihood_score"] = b.get("ai_likelihood_score")  # 0..1
    out["tier"] = b.get("tier")
    return out


def _auc(human: list[float], ai: list[float]) -> float:
    """P(ai_score > human_score) via all pairs (Mann-Whitney). 0.5 = no separation, 1.0 = perfect
    (ai always scores higher = correct ordering for an AI-risk signal)."""
    if not human or not ai:
        return float("nan")
    wins = ties = 0
    for a in ai:
        for h in human:
            if a > h:
                wins += 1
            elif a == h:
                ties += 1
    return (wins + 0.5 * ties) / (len(ai) * len(human))


def _best_threshold(clean: list[float], flagged: list[float]) -> tuple[float, float]:
    vals = sorted(set(clean + flagged))
    best_t, best_acc = vals[0], 0.0
    for i in range(len(vals)):
        t = vals[i] + (vals[i + 1] - vals[i]) / 2 if i + 1 < len(vals) else vals[i] + 1
        correct = sum(1 for v in clean if v < t) + sum(1 for v in flagged if v >= t)
        acc = correct / (len(clean) + len(flagged))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def main() -> int:
    _load_env()
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "poc"))

    rows = []
    for d in DIRS:
        for path in sorted(d.glob("*.json")):
            c = json.loads(path.read_text())
            auth = c.get("authorship")
            if auth not in ("human", "ai"):
                continue
            rows.append({"case_id": c["case_id"], "authorship": auth, "signals": _signals(c["text"])})

    human = [r for r in rows if r["authorship"] == "human"]
    ai = [r for r in rows if r["authorship"] == "ai"]
    print(f"labeled cases: {len(rows)}  (human={len(human)}, ai={len(ai)})\n")

    # Composite confusion at live tiers
    def tier_of(score):
        for name, lo, hi in TIERS:
            if score is not None and lo <= score < hi:
                return name
        return "?"
    print("== composite ai_likelihood_score by tier ==")
    for label, group in (("human", human), ("ai", ai)):
        from collections import Counter
        dist = Counter(tier_of(r["signals"].get("ai_likelihood_score")) for r in group)
        print(f"  {label:<6} n={len(group):<3} tiers={dict(dist)}")
    if human:
        false_high = sum(1 for r in human if (r["signals"].get("ai_likelihood_score") or 0) >= GREEN_MAX)
        print(f"  HUMAN false-high (amber+): {false_high}/{len(human)}")
    if ai:
        false_low = sum(1 for r in ai if (r["signals"].get("ai_likelihood_score") or 0) < GREEN_MAX)
        print(f"  AI false-low (green):      {false_low}/{len(ai)}")

    ready = len(human) >= MIN_PER_CLASS and len(ai) >= MIN_PER_CLASS
    print("\n== per-signal human-vs-AI discrimination ==")
    print(f"{'signal':<32}{'human_mean':>11}{'ai_mean':>9}{'AUC':>7}{'thr_acc':>9}")
    table = []
    for sig in COMPONENTS + ["ai_likelihood_score"]:
        hv = [r["signals"][sig] for r in human if isinstance(r["signals"].get(sig), (int, float))]
        av = [r["signals"][sig] for r in ai if isinstance(r["signals"].get(sig), (int, float))]
        if not hv or not av:
            print(f"{sig:<32}{'-':>11}{'-':>9}{'-':>7}{'-':>9}")
            continue
        auc = _auc(hv, av)
        _, acc = _best_threshold(hv, av)
        table.append({"signal": sig, "human_mean": sum(hv) / len(hv), "ai_mean": sum(av) / len(av),
                      "auc": auc, "thr_acc": acc})
        print(f"{sig:<32}{sum(hv)/len(hv):>11.3f}{sum(av)/len(av):>9.3f}{auc:>7.2f}{acc:>9.0%}")

    table.sort(key=lambda x: -(x["auc"] if x["auc"] == x["auc"] else 0))
    print("\nbest discriminators (by AUC):", ", ".join(f"{t['signal']}={t['auc']:.2f}" for t in table[:4]))
    if not ready:
        print(f"\nINSUFFICIENT for a validated recalibration — need >= {MIN_PER_CLASS} of EACH class "
              f"(have human={len(human)}, ai={len(ai)}). Add real human samples to authorship_cases/.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"ready": ready, "human": len(human), "ai": len(ai),
                               "per_signal": table, "rows": rows}, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
