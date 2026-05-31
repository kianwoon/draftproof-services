#!/usr/bin/env python3
"""Fit (or refuse to fit) a Turnitin calibration from real labeled reports.

For every case in turnitin_cases/ it computes our signals and groups them by the REAL Turnitin
label (clean = under 20%/'*'; flagged = >=20% surfaced). Then, per signal, it measures how well
the two classes SEPARATE and — only if there are enough of BOTH classes — fits a 1-D threshold
and reports its accuracy. With too little data it refuses and states exactly what's missing.

Honest by construction: it will NOT emit a "calibrated Turnitin score" until the labeled data can
support one. The discriminative target is `generic_assertion` (the only signal that orders
human-clean below AI in our probes); the fluency signals (external/topk) are expected to fail.

Run:  ~/.pyenv/versions/3.11.0/bin/python3 poc/calibration/calibrate.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CASES = HERE / "turnitin_cases"
OUT = REPO / "test_output" / "_turnitin_calibration_fit.json"

MIN_PER_CLASS = 3                      # need this many of EACH class before we trust a threshold
SIGNALS = ["generic_assertion", "external", "calibrated_risk", "topk", "draftproof"]  # all lower=cleaner


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
    e = b.get("external_detector_estimate") or {}
    intel = (rep.get("scan_intelligence") or {}).get("transformation") or {}
    contrib = intel.get("contribution") or {}
    return {
        "generic_assertion": c.get("generic_assertion_risk"),
        "external": e.get("score"),
        "calibrated_risk": contrib.get("calibrated_ai_risk", contrib.get("adjusted_ai_risk")),
        "topk": c.get("topk_pattern_raw") or c.get("topk_pattern"),
        "draftproof": b.get("ai_likelihood_score"),
    }


def _label(turnitin: dict) -> str | None:
    band = (turnitin or {}).get("band")
    if band in ("clean", "flagged"):
        return band
    pct = (turnitin or {}).get("ai_percent")
    if isinstance(pct, (int, float)):
        return "flagged" if pct >= 20 else "clean"
    return None


def _best_threshold(clean: list[float], flagged: list[float]) -> tuple[float, float]:
    """1-D threshold maximizing accuracy for a lower=cleaner signal. Returns (threshold, accuracy)."""
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
    for path in sorted(CASES.glob("*.json")):
        c = json.loads(path.read_text())
        lab = _label(c.get("turnitin", {}))
        if lab is None:
            print(f"skip {c.get('case_id')}: no usable Turnitin label"); continue
        tii = (c.get("turnitin", {}) or {}).get("ai_percent")
        rows.append({"case_id": c["case_id"], "label": lab, "tii": tii, "signals": _signals(c["text"])})

    clean = [r for r in rows if r["label"] == "clean"]
    flagged = [r for r in rows if r["label"] == "flagged"]
    print(f"labeled cases: {len(rows)}  (clean={len(clean)}, flagged={len(flagged)})\n")
    for r in rows:
        s = r["signals"]
        print(f"  {r['case_id']:<18}[{r['label']:>7}]  " +
              "  ".join(f"{k}={s[k]:.0f}" if isinstance(s[k], (int, float)) else f"{k}=-" for k in SIGNALS))

    ready = len(clean) >= MIN_PER_CLASS and len(flagged) >= MIN_PER_CLASS
    print("\n========== CALIBRATION ==========")
    if not ready:
        need_f = max(0, MIN_PER_CLASS - len(flagged))
        need_c = max(0, MIN_PER_CLASS - len(clean))
        print(f"INSUFFICIENT for a VALIDATED threshold — need >= {need_f} more FLAGGED and "
              f">= {need_c} more CLEAN report(s) (target {MIN_PER_CLASS}+ per class).")
        if flagged:
            # PROVISIONAL anchor-scale: bring our over-stated magnitude toward Turnitin's scale using
            # the flagged anchor(s). Turnitin suppresses <20%, so this is a magnitude scale, not a
            # precise %; with few anchors it is UNVALIDATED and can under-flag genuinely-AI docs.
            print(f"\n  -- PROVISIONAL anchor-scale (fit to {len(flagged)} flagged anchor(s), UNVALIDATED) --")
            scales = {}
            for sig in SIGNALS:
                pairs = [(f["signals"][sig], f["tii"]) for f in flagged
                         if isinstance(f["signals"][sig], (int, float)) and f["signals"][sig]
                         and isinstance(f["tii"], (int, float))]
                if pairs:
                    scales[sig] = sum(t / s for s, t in pairs) / len(pairs)
            for sig in sorted(scales, key=lambda k: abs(1 - scales[k])):
                anchor = flagged[0]["signals"][sig]
                pred = anchor * scales[sig] if isinstance(anchor, (int, float)) else None
                print(f"    {sig:<20} x{scales[sig]:.2f}   ({anchor:.0f} -> {pred:.0f})"
                      if pred is not None else f"    {sig:<20} -")
            # Recommend ONLY among signals that actually discriminate (human<LLM<AI in the probe).
            # calibrated_risk/topk/external are excluded: calibrated_risk is non-discriminative
            # (~20 for everything, so its small correction is coincidental); topk/external are
            # pure-fluency over-staters.
            discriminative = [s for s in ("draftproof", "generic_assertion") if s in scales]
            if discriminative:
                best = min(discriminative, key=lambda k: abs(1 - scales[k]))
                print(f"  Recommended base: '{best}' x{scales[best]:.2f}  (discriminative + smallest correction).")
                print("  NOT calibrated_risk despite its tiny correction — it's non-discriminative (~20 for")
                print("  human AND AI), so scaling it clears everything.")
            print("  CAVEAT: a downward scale fit to flagged-only anchors can UNDER-flag a genuinely-AI")
            print("  doc (show ~27% when Turnitin says 60%). Validate with a HIGH-flagged + a CLEAN anchor.")
        if clean and not flagged:
            best = max((c["signals"].get("generic_assertion") or 0) for c in clean)
            print(f"  Provisional anchor only: generic_assertion up to ~{best:.0f} has been Turnitin-CLEAN.")
            print("  The flag line sits ABOVE that — unknown until flagged reports arrive.")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"ready": False, "clean": len(clean), "flagged": len(flagged),
                                   "rows": rows}, indent=2))
        print(f"\nwrote {OUT}")
        return 0

    print(f"{'signal':<20}{'clean<=':>9}{'flagged>=':>11}{'threshold':>11}{'accuracy':>10}")
    fits = []
    for sig in SIGNALS:
        cv = [r["signals"][sig] for r in clean if isinstance(r["signals"][sig], (int, float))]
        fv = [r["signals"][sig] for r in flagged if isinstance(r["signals"][sig], (int, float))]
        if not cv or not fv:
            continue
        t, acc = _best_threshold(cv, fv)
        fits.append((sig, t, acc, max(cv), min(fv)))
        print(f"{sig:<20}{max(cv):>9.0f}{min(fv):>11.0f}{t:>11.1f}{acc:>9.0%}")
    fits.sort(key=lambda x: -x[2])
    best = fits[0]
    print(f"\nBEST SIGNAL: {best[0]}  ->  predict FLAGGED when {best[0]} >= {best[1]:.1f}  "
          f"(train accuracy {best[2]:.0%})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"ready": True, "best_signal": best[0], "threshold": best[1],
                               "accuracy": best[2], "rows": rows}, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
