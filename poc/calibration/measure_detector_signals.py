"""Detector-signal A/B harness — class separation (AI vs human) for the pure-python
writing-quality / grounding estimators being de-hardcoded.

Purpose: gate the NO-HARDCODE refactor. Each estimator is supposed to score AI-generated
text HIGHER-risk than human text. This measures, per signal, how well it separates the two
classes on the labeled corpus (poc/calibration/authorship_cases/, `authorship` = ai|human),
so we can prove a de-hardcode change does NOT degrade discrimination.

Pure-python (no ML stack needed) — these estimators are regex/statistics only.

Usage:
    python calibration/measure_detector_signals.py                 # print + write baseline
    python calibration/measure_detector_signals.py --out PATH      # write to PATH
    python calibration/measure_detector_signals.py --compare PATH   # diff vs saved baseline
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

# Make both poc/ (so `detect` resolves) and the repo root (so `poc.predictability` resolves,
# mixed import styles exist) importable when run directly.
_POC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_POC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detect import layer3_scoring as L3

HERE = Path(__file__).resolve().parent
CASES = HERE / "authorship_cases"
DEFAULT_OUT = HERE.parent / "test_output" / "_detector_signal_separation.json"

# name -> callable(text) -> float in [0,1]. All are "risk-up for AI" (higher = more AI-like).
SIGNALS: dict[str, callable] = {
    "formulaic_progression": L3.estimate_formulaic_progression_risk,
    "balanced_generic_framing": L3.estimate_balanced_generic_framing_risk,
    "repeated_starter": L3.estimate_repeated_starter_risk,
    "signpost_paragraph": L3.estimate_signpost_paragraph_risk,
    "balanced_hedging": L3.estimate_balanced_hedging_risk,
    "formulaic_conclusion": L3.estimate_formulaic_conclusion_risk,
    "generic_assertion": L3.estimate_generic_assertion_risk,
    "lived_detail": L3.estimate_lived_detail_risk,           # higher = LESS lived detail = more AI
    "broad_claim": L3.estimate_broad_claim_risk,
    "unsupported_claim": L3.estimate_unsupported_claim_risk,
    "register": L3.register_score,
}


def load_cases() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(CASES / "*.json"))):
        d = json.loads(Path(path).read_text())
        label = (d.get("authorship") or "").strip().lower()
        text = d.get("text") or ""
        if label in ("ai", "human") and text:
            rows.append({"id": d.get("case_id"), "label": label, "text": text})
    return rows


def _auc(ai: list[float], human: list[float]) -> float:
    """P(ai_score > human_score) over all cross-class pairs; 0.5 = no separation."""
    if not ai or not human:
        return float("nan")
    wins = ties = 0
    for a in ai:
        for h in human:
            if a > h:
                wins += 1
            elif a == h:
                ties += 1
    return (wins + 0.5 * ties) / (len(ai) * len(human))


def measure(rows: list[dict]) -> dict:
    out = {}
    for name, fn in SIGNALS.items():
        ai_vals, human_vals = [], []
        for r in rows:
            try:
                v = float(fn(r["text"]))
            except Exception as exc:  # noqa: BLE001 — record, don't crash the sweep
                out.setdefault("_errors", {})[name] = repr(exc)
                ai_vals, human_vals = [], []
                break
            (ai_vals if r["label"] == "ai" else human_vals).append(v)
        if not ai_vals or not human_vals:
            continue
        out[name] = {
            "ai_mean": round(statistics.mean(ai_vals), 4),
            "human_mean": round(statistics.mean(human_vals), 4),
            "gap": round(statistics.mean(ai_vals) - statistics.mean(human_vals), 4),
            "auc": round(_auc(ai_vals, human_vals), 4),
            "n_ai": len(ai_vals),
            "n_human": len(human_vals),
        }
    return out


def _print_table(result: dict) -> None:
    print(f"{'signal':<26}{'AI':>8}{'human':>8}{'gap':>8}{'AUC':>8}")
    print("-" * 58)
    for name, s in sorted(result.items(), key=lambda kv: kv[1].get("auc", 0) if isinstance(kv[1], dict) else 0, reverse=True):
        if name.startswith("_"):
            continue
        flag = "" if s["auc"] >= 0.60 else "  <-- weak"
        print(f"{name:<26}{s['ai_mean']:>8}{s['human_mean']:>8}{s['gap']:>+8}{s['auc']:>8}{flag}")
    if result.get("_errors"):
        print("\nERRORS:", result["_errors"])


def _compare(current: dict, baseline_path: Path) -> None:
    base = json.loads(baseline_path.read_text()).get("signals", {})
    print(f"\n=== vs baseline {baseline_path.name} (AUC delta; - = regression) ===")
    print(f"{'signal':<26}{'base':>8}{'now':>8}{'delta':>8}")
    print("-" * 50)
    regressed = []
    for name in sorted(set(base) | {k for k in current if not k.startswith('_')}):
        b = base.get(name, {}).get("auc")
        c = current.get(name, {}).get("auc")
        if b is None or c is None:
            print(f"{name:<26}{str(b):>8}{str(c):>8}{'NEW/GONE':>8}")
            continue
        delta = round(c - b, 4)
        mark = "  REGRESSION" if delta <= -0.05 else ""
        if delta <= -0.05:
            regressed.append(name)
        print(f"{name:<26}{b:>8}{c:>8}{delta:>+8}{mark}")
    print("\nRESULT:", "REGRESSIONS: " + ", ".join(regressed) if regressed else "no signal regressed >=0.05 AUC")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--compare", default=None, help="baseline JSON to diff AUC against")
    args = ap.parse_args()

    rows = load_cases()
    result = measure(rows)
    print(f"cases: {len(rows)} (ai={sum(r['label']=='ai' for r in rows)}, "
          f"human={sum(r['label']=='human' for r in rows)})\n")
    _print_table(result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"signals": result, "n_cases": len(rows)}, indent=2))
    print(f"\nwrote {out_path}")

    if args.compare:
        _compare(result, Path(args.compare))


if __name__ == "__main__":
    main()
