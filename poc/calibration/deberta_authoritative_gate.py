"""Phase-3 fairness gate for the DeBERTa AUTHORITATIVE signal (>=0.99 high-confidence proportion).

Distinct from deberta_fpr_gate.py, which measures the DISPLAY signal (compose()'s >=0.80
proportion, ~20% ESL FPR). The authoritative tier uses the >=0.99 high-confidence proportion
(compose_authoritative), which must carry <=3% ESL FPR before DRAFTPROOF_DEBERTA_AUTHORITATIVE
can be enabled in production.

Scores the >=0.99 proportion on SCoCESLE (higher + lower proficiency) + the in-repo AI set.
A human essay is a FALSE POSITIVE when its >=0.99 proportion reads at/above the FPR threshold.

Usage:
    python calibration/deberta_authoritative_gate.py
    python calibration/deberta_authoritative_gate.py --corpus "/path/to/SCoCESLE" --limit 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_POC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_POC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from calibration.fpr_subgroup_gate import (  # noqa: E402
    DEFAULT_CORPUS, _proficiency_groups, _ai_texts, _fpr, _dist, _auc,
)
from detect import deberta_signal  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "test_output" / "_deberta_authoritative_gate.json"

# The authoritative proportion is 0-100 (% of sentences >=0.99). FPR thresholds on that scale.
FPR_THRESHOLDS = [10.0, 25.0, 50.0]  # green/amber/orange cutoffs from _AUTHORITATIVE_CUTOFFS x100
PRIMARY_THRESHOLD = 25.0  # the AMBER cutoff — the point where the tier stops being GREEN


def _score_texts(texts):
    """Score via compose_authoritative — the >=0.99 high-confidence proportion (0-100)."""
    out = []
    for t in texts:
        sig = deberta_signal.compose_authoritative(t) or {}
        s = sig.get("ai_likelihood_score")  # 0-1 proportion
        out.append(round(float(s) * 100, 2) if s is not None else None)
    return out


def _drop_none(vals):
    return [v for v in vals if v is not None]


def measure(corpus: str, limit: int | None) -> dict:
    # compose_authoritative requires the flag ON to return a score.
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    os.environ["DRAFTPROOF_DEBERTA_AUTHORITATIVE"] = "1"
    groups = _proficiency_groups(corpus)
    if not groups["higher"] and not groups["lower"]:
        print(f"No SCoCESLE essays found under {corpus!r}.", file=sys.stderr)
        raise SystemExit(2)

    scores = {}
    n_in = {}
    for g in ("higher", "lower"):
        files = groups[g][: max(1, limit // 2)] if limit else groups[g]
        n_in[g] = len(files)
        texts = [Path(fp).read_text(encoding="utf-8", errors="ignore") for fp in files]
        scores[g] = _drop_none(_score_texts(texts))

    ai_texts = _ai_texts()
    if limit:
        ai_texts = ai_texts[: max(1, limit // 2)]
    ai_scores = _drop_none(_score_texts(ai_texts))

    all_human = scores["higher"] + scores["lower"]
    fpr_by = {g: {str(t): _fpr(scores[g], t) for t in FPR_THRESHOLDS} for g in ("higher", "lower")}
    hi = fpr_by["higher"].get(str(PRIMARY_THRESHOLD))
    lo = fpr_by["lower"].get(str(PRIMARY_THRESHOLD))
    parity = round(lo - hi, 1) if (hi is not None and lo is not None) else None

    return {
        "signal": "deberta_authoritative_>=0.99_proportion",
        "primary_threshold": PRIMARY_THRESHOLD,
        "fpr_pct": {
            "all": {str(t): _fpr(all_human, t) for t in FPR_THRESHOLDS},
            "higher": fpr_by["higher"],
            "lower": fpr_by["lower"],
        },
        "parity_gap_pct": parity,
        "human_dist": {"all": _dist(all_human), "higher": _dist(scores["higher"]), "lower": _dist(scores["lower"])},
        "ai_dist": _dist(ai_scores),
        "auc_ai_vs_human": _auc(ai_scores, all_human),
        "n": {
            "higher": len(scores["higher"]),
            "lower": len(scores["lower"]),
            "ai": len(ai_scores),
            "dropped_short": (n_in["higher"] + n_in["lower"]) - len(all_human),
        },
    }


def _print(res: dict) -> None:
    t = str(res["primary_threshold"])
    fpr = res["fpr_pct"]
    print(f"\n  SIGNAL: {res['signal']}")
    print(f"  N: higher={res['n']['higher']} lower={res['n']['lower']} ai={res['n']['ai']}"
          f"  (dropped <150w: {res['n'].get('dropped_short')})")
    print(f"  ESL FALSE-POSITIVE RATE @ >=0.99 proportion >={t}%  (lower = better):")
    print(f"    all ESL    : {fpr['all'][t]}%   (10%: {fpr['all']['10.0']}  50%: {fpr['all']['50.0']})")
    print(f"    higher-prof: {fpr['higher'][t]}%")
    print(f"    lower-prof : {fpr['lower'][t]}%   <- at-risk subgroup")
    print(f"  PARITY GAP (lower - higher) @ {t}%: {res['parity_gap_pct']} pts  (>0 = lower-prof penalized more)")
    hd = res["human_dist"]["all"]
    ad = res["ai_dist"]
    print(f"  human proportion: mean {hd.get('mean')}%  p90 {hd.get('p90')}%  max {hd.get('max')}%")
    print(f"  ai    proportion: mean {ad.get('mean')}%  p10 {ad.get('p10') if 'p10' in ad else ad.get('p50')}%")
    print(f"  AUC (AI vs human separation): {res['auc_ai_vs_human']}")
    # Gate verdict
    fpr_all = fpr['all'][t]
    verdict = "PASS" if fpr_all <= 3.0 else "FAIL"
    print(f"\n  GATE VERDICT @ {t}% (<=3.0% required): {verdict}  (all-ESL FPR = {fpr_all}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.environ.get("SCOCESLE_CORPUS", DEFAULT_CORPUS))
    ap.add_argument("--limit", type=int, default=None, help="cap essays per half (smoke test)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    print(f"Scoring AUTHORITATIVE signal (>=0.99 proportion) on SCoCESLE...", flush=True)
    t0 = time.time()
    res = measure(args.corpus, args.limit)
    res["seconds"] = round(time.time() - t0, 1)
    _print(res)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
