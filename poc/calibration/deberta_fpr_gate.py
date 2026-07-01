"""DeBERTa-signal ESL false-positive GATE for the off-the-shelf AI-text-detection checkpoint.

Mirrors ``calibration/fpr_subgroup_gate.py`` (SCoCESLE ESL corpus + in-repo AI set) but scores
through ``detect.deberta_signal.compose()`` — i.e. the DeBERTa checkpoint ONLY, not the full
composite detector — which is ~10-50x faster and is what we actually need to judge the
checkpoint in isolation. Reads ``ai_signal_deberta.signal_pct`` (the v2 threshold-proportion
signal: % of sentences scoring >= SENT_THRESHOLD under the detector).

Use to PICK a checkpoint (Phase 0): loop candidates with --candidates, compare ESL FPR +
human-score distribution + AUC(AI vs human), then choose the one with acceptable ESL FPR and
best AI recall. SCoCESLE is LOCAL-ONLY (no redistribution license) — this is a LOCAL gate, not CI.

Usage:
    python calibration/deberta_fpr_gate.py --candidates org/model-a,org/model-b   # compare
    python calibration/deberta_fpr_gate.py --limit 12                              # smoke (~18 texts)
    python calibration/deberta_fpr_gate.py --corpus "/path/to/SCoCESLE"
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

from calibration.fpr_subgroup_gate import (  # noqa: E402  reuse corpus loaders + stats helpers
    FPR_THRESHOLDS,
    PRIMARY_THRESHOLD,
    DEFAULT_CORPUS,
    _proficiency_groups,
    _ai_texts,
    _fpr,
    _dist,
    _auc,
)
from detect import deberta_model, deberta_signal  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "test_output" / "_deberta_fpr_gate.json"


def _reset_model() -> None:
    """Force the lazy singleton to reload on the next score (call between candidate checkpoints)."""
    deberta_model._MODEL = None
    deberta_model._TOKENIZER = None
    deberta_model._AI_INDEX = None


def _score_texts(texts):
    """Score via compose() directly — runs ONLY the DeBERTa model, not the full composite.

    Reads signal_pct (v2): the % of sentences scoring >= SENT_THRESHOLD. For the ESL FPR
    measurement that's the document-level signal on the 0-100 scale, comparable across the
    FPR_THRESHOLDS the same way the old calibrated `score` was."""
    out = []
    for t in texts:
        sig = deberta_signal.compose(t) or {}
        s = sig.get("signal_pct")
        out.append(float(s) if s is not None else None)
    return out


def _drop_none(vals):
    return [v for v in vals if v is not None]


def measure(corpus: str, limit: int | None) -> dict:
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"  # ensure the field is computed
    groups = _proficiency_groups(corpus)
    if not groups["higher"] and not groups["lower"]:
        print(f"No SCoCESLE essays found under {corpus!r} — set --corpus / $SCOCESLE_CORPUS.", file=sys.stderr)
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
        "model": deberta_model._model_name(),
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
    print(f"\n  MODEL: {res['model']}")
    print(f"  N: higher={res['n']['higher']} lower={res['n']['lower']} ai={res['n']['ai']}"
          f"  (dropped <150w: {res['n'].get('dropped_short')})")
    print(f"  ESL FALSE-POSITIVE RATE @ deberta_score>={t}%  (lower = better):")
    print(f"    all ESL    : {fpr['all'][t]}%   (40%: {fpr['all']['40.0']}  60%: {fpr['all']['60.0']})")
    print(f"    higher-prof: {fpr['higher'][t]}%")
    print(f"    lower-prof : {fpr['lower'][t]}%   <- at-risk subgroup")
    print(f"  PARITY GAP (lower - higher) @ {t}%: {res['parity_gap_pct']} pts  (>0 = lower-prof penalized more)")
    hd = res["human_dist"]["all"]
    ad = res["ai_dist"]
    print(f"  human deberta score: mean {hd.get('mean')}%  p90 {hd.get('p90')}%  max {hd.get('max')}%")
    print(f"  ai    deberta score: mean {ad.get('mean')}%  p10 {ad.get('p10') if 'p10' in ad else ad.get('p50')}%  min {ad.get('min')}")
    print(f"  AUC (AI vs human separation): {res['auc_ai_vs_human']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.environ.get("SCOCESLE_CORPUS", DEFAULT_CORPUS))
    ap.add_argument("--candidates", default=None,
                    help="comma-separated HF repo-ids; default = $DRAFTPROOF_DEBERTA_MODEL")
    ap.add_argument("--limit", type=int, default=None, help="cap essays per half (smoke test)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",")] if args.candidates \
        else [deberta_model._model_name()]

    results = []
    for c in candidates:
        print(f"\n=== Candidate: {c} ===", flush=True)
        os.environ["DRAFTPROOF_DEBERTA_MODEL"] = c
        _reset_model()
        t0 = time.time()
        res = measure(args.corpus, args.limit)
        res["seconds"] = round(time.time() - t0, 1)
        _print(res)
        results.append(res)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results if len(results) > 1 else results[0], indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
