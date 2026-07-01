"""Fit a WINDOW-LEVEL isotonic calibrator for the DeBERTa checkpoint on SCoCESLE.

SUPERSEDED IN PRODUCTION (2026-07). The v2 signal (deberta_signal_v2) uses a
threshold-proportion design with NO calibrator — see deberta_signal.py. This
script is retained for offline analysis / model comparison only; it no longer
produces a calibrator that any production path loads. Do NOT point
DRAFTPROOF_DEBERTA_CALIBRATOR at its output — compose() ignores that env var in v2.

Historical context (why the calibrator approach was abandoned):
This originally superseded ``deberta_fit_calibrator.py``, which fit at the DOCUMENT
level (the mean of per-window probs). That document-level fit collapsed to a step
function because the corpus's AI and human distributions barely overlap at the
document level (humans <0.70, AI >0.86): isotonic regression had no support inside
the gap, so every document mean below ~0.854 calibrated to exactly 0.0, silently
zeroing mixed human/AI documents (the production 0%-bug). The window-level fit
here fixed that — but its curve still collapsed to a cliff at the TOP (everything
below raw=1.000 floored to ~0-8%), because AI windows also pile up at exactly 1.0.
Isotonic regression cannot calibrate this model well on SCoCESLE at either level:
the AI/human distributions are too cleanly separated. The v2 design sidesteps the
problem entirely with threshold-proportion aggregation and no calibration.

Usage (offline analysis only):
    python calibration/deberta_fit_calibrator_windows.py \\
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
from detect import deberta_model, deberta_calibrate, deberta_windowing  # noqa: E402


def _window_probs(text, score_windows_fn):
    """Return the raw AI-class probability for each (size=3, step=1) window."""
    sents = deberta_windowing.split_sentences(text)
    windows = deberta_windowing.build_windows(sents, size=3, step=1)
    if not windows:
        return []
    return score_windows_fn(windows) or []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo-id to calibrate")
    ap.add_argument("--corpus", default=os.environ.get("SCOCESLE_CORPUS", DEFAULT_CORPUS))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "deberta_isotonic.pkl"))
    args = ap.parse_args()

    os.environ["DRAFTPROOF_DEBERTA_MODEL"] = args.model
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    # RAW scoring: do not let an existing calibrator touch the fit inputs.
    os.environ.pop("DRAFTPROOF_DEBERTA_CALIBRATOR", None)

    # Lazy import AFTER env is set so _load() picks up the requested model.
    from detect import deberta_signal  # noqa: E402
    deberta_model._MODEL = deberta_model._TOKENIZER = None
    deberta_model._AI_INDEX = None

    groups = _proficiency_groups(args.corpus)
    human_texts = [Path(fp).read_text(encoding="utf-8", errors="ignore")
                   for g in ("higher", "lower") for fp in groups[g]]
    ai_texts = _ai_texts()

    print(f"Scoring WINDOWS: {len(human_texts)} humans + {len(ai_texts)} AI (raw) with {args.model} ...", flush=True)

    def _all_windows(texts):
        out = []
        for t in texts:
            out += _window_probs(t, deberta_signal.score_windows)
        return out

    human_win = _all_windows(human_texts)
    ai_win = _all_windows(ai_texts)

    # label: 1 = AI window, 0 = human window  ->  calibrated prob = P(AI)
    iso = deberta_calibrate.fit_isotonic(
        ai_win + human_win, [1] * len(ai_win) + [0] * len(human_win))

    # --- Evaluate at the DOCUMENT level the way compose() will use it: calibrate
    #     each window, then mean. This is the number users see, so report FPR/AUC
    #     on it (not on raw window scores, which over-flag ESL — see module docstring).
    def _doc_cal(texts):
        out = []
        for t in texts:
            p = _window_probs(t, deberta_signal.score_windows)
            if not p:
                continue
            cal = deberta_calibrate.apply_isotonic(p, iso)
            out.append((sum(cal) / len(cal)) * 100.0)
        return out

    hc = _doc_cal(human_texts)
    ac = _doc_cal(ai_texts)

    def _mean(xs): return sum(xs) / len(xs) if xs else 0.0
    def _p(k, xs):
        if not xs: return 0.0
        s = sorted(xs); return s[min(len(s) - 1, int(len(s) * k))]
    _row = lambda name, v: print(f"  {name:<22}{v:>12.1f}")

    print(f"\n  === {args.model} : WINDOW-LEVEL isotonic (document-level rollup) ===")
    print(f"  windows: human n={len(human_win)}  ai n={len(ai_win)}")
    _row("human mean %", _mean(hc)); _row("human p90 %", _p(.90, hc)); _row("human max %", max(hc))
    _row("ai mean %", _mean(ac)); _row("ai p10 %", _p(.10, ac))
    print()
    for t in FPR_THRESHOLDS:
        print(f"  ESL FPR @>={t:>4}% : {_fpr(hc, t):>5}%")
    print(f"\n  AUC (AI vs human): {_auc(ac, hc):.4f}")
    print(f"\n  fitted curve (sampled):")
    for x in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0]:
        print(f"    raw {x:.2f} -> cal {iso.predict([x])[0]:.3f}")

    deberta_calibrate.save_calibrator(iso, args.out)
    print(f"\n  saved window-level calibrator -> {args.out}")
    print(f"  to use: export DRAFTPROOF_DEBERTA_CALIBRATOR={args.out}")


if __name__ == "__main__":
    main()
