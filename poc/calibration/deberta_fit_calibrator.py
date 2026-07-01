"""Fit + evaluate an isotonic calibrator for the chosen DeBERTa checkpoint on SCoCESLE.

SUPERSEDED IN PRODUCTION (2026-07). The v2 signal (deberta_signal_v2) uses a
threshold-proportion design with NO calibrator — see deberta_signal.py. This script is
retained for offline analysis / model comparison only. Do NOT point
DRAFTPROOF_DEBERTA_CALIBRATOR at its output — compose() ignores that env var in v2.

Historical note (the document-level fit this performs collapses to a step function because
AI/human document scores barely overlap on SCoCESLE; it was the source of the production
0%-bug). See deberta_fit_calibrator_windows.py for the full history of why calibration was
abandoned in favor of threshold-proportion.

Usage (offline analysis only):
    python calibration/deberta_fit_calibrator.py \\
        --model fakespot-ai/roberta-base-ai-text-detection-v1 \\
        --out calibration/deberta_isotonic_offline.pkl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_POC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_POC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from calibration.fpr_subgroup_gate import (  # noqa: E402
    FPR_THRESHOLDS, PRIMARY_THRESHOLD, DEFAULT_CORPUS,
    _proficiency_groups, _ai_texts, _fpr, _auc,
)
from detect import deberta_model, deberta_signal, deberta_calibrate  # noqa: E402


def _raw_scores(texts):
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    os.environ.pop("DRAFTPROOF_DEBERTA_CALIBRATOR", None)  # RAW, not calibrated
    out = []
    for t in texts:
        sig = deberta_signal.compose(t) or {}
        s = sig.get("score")
        out.append(float(s) if s is not None else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo-id to calibrate")
    ap.add_argument("--corpus", default=os.environ.get("SCOCESLE_CORPUS", DEFAULT_CORPUS))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "deberta_isotonic.pkl"))
    args = ap.parse_args()

    os.environ["DRAFTPROOF_DEBERTA_MODEL"] = args.model
    deberta_model._MODEL = deberta_model._TOKENIZER = None
    deberta_model._AI_INDEX = None

    groups = _proficiency_groups(args.corpus)
    human_texts = [Path(fp).read_text(encoding="utf-8", errors="ignore")
                   for g in ("higher", "lower") for fp in groups[g]]
    ai_texts = _ai_texts()

    print(f"Scoring {len(human_texts)} humans + {len(ai_texts)} AI (raw) with {args.model} ...", flush=True)
    # compose() returns score on 0-100, but applies the calibrator to doc_raw on the 0-1 scale
    # internally. Fit on 0-1 so the saved calibrator is consistent with how compose() uses it.
    human_raw = [v / 100.0 for v in _raw_scores(human_texts) if v is not None]
    ai_raw = [v / 100.0 for v in _raw_scores(ai_texts) if v is not None]

    # label: 1 = AI, 0 = human  ->  calibrated score = P(AI)
    iso = deberta_calibrate.fit_isotonic(ai_raw + human_raw, [1] * len(ai_raw) + [0] * len(human_raw))
    human_cal = deberta_calibrate.apply_isotonic(human_raw, iso)  # 0-1
    ai_cal = deberta_calibrate.apply_isotonic(ai_raw, iso)        # 0-1

    print(f"\n  === {args.model} : raw vs isotonic-calibrated ===")
    print(f"  humans n={len(human_cal)}  ai n={len(ai_cal)}")
    print(f"  {'metric':<22}{'raw':>12}{'calibrated':>14}")
    for label, h, a in (("human mean %", human_raw, human_cal), ("human p90 %", None, None)):
        pass
    _row = lambda name, rv, cv: print(f"  {name:<22}{rv:>12.1f}{cv:>14.1f}")

    def _mean(xs): return sum(xs) / len(xs) if xs else 0.0
    def _p90(xs):
        if not xs: return 0.0
        s = sorted(xs); return s[min(len(s) - 1, int(len(s) * 0.9))]

    # raw/cal are 0-1; display + FPR on the 0-100 scale (×100).
    hr = [v * 100 for v in human_raw]; hc = [v * 100 for v in human_cal]
    ar = [v * 100 for v in ai_raw]; ac = [v * 100 for v in ai_cal]
    _row("human mean %", _mean(hr), _mean(hc))
    _row("human p90 %", _p90(hr), _p90(hc))
    _row("human max %", max(hr), max(hc))
    _row("ai mean %", _mean(ar), _mean(ac))
    _row("ai p10 %", sorted(ar)[0], sorted(ac)[0])
    print()
    for t in FPR_THRESHOLDS:
        print(f"  ESL FPR @>={t:>4}% : raw {_fpr(hr, t):>5}%   calibrated {_fpr(hc, t):>5}%")
    print(f"\n  AUC (AI vs human): raw {_auc(ar, hr):.4f}   calibrated {_auc(ac, hc):.4f}")

    deberta_calibrate.save_calibrator(iso, args.out)
    print(f"\n  saved calibrator -> {args.out}")
    print(f"  to use: export DRAFTPROOF_DEBERTA_CALIBRATOR={args.out}")


if __name__ == "__main__":
    main()
