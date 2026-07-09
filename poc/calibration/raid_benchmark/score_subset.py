#!/usr/bin/env python3
"""Score a RAID subset with DraftProof's PROD FUSED detector.

Reuses the real scan pipeline exactly as production runs it:
``calibration.measure_end_to_end.scan_text`` -> full ``DetectionRunner`` + report.
The four V7 fusion flags are exported BEFORE the heavy import so the fused path
(tier authority + deep scan + authorship breakdown + DeBERTa-authoritative) is live.

The fused AI likelihood is read from ``report["ai_risk_badge"]["ai_likelihood_score"]``
(0-100). We fail LOUDLY if the badge or the score is missing — a silent 0 would
corrupt the benchmark.

Deep-scan responses are cached to the shared retune JSONL cache
(``calibration/retune/cache/deepscan_scores.jsonl``) so re-runs are FREE: only rows
whose sentence content is not already cached incur a Modal call. A full ~600-row
first pass therefore costs at most ~600 Modal deep-scan calls (fewer for short/empty
docs and any content already in the cache); every subsequent run is $0.

Usage:
    # SMOKE (5 Modal calls, trivial):
    python calibration/raid_benchmark/score_subset.py --limit 5
    # FULL (the paid run):
    python calibration/raid_benchmark/score_subset.py --out scores.jsonl

Output ``scores.jsonl`` — one JSON object per line:
    {"label": 0|1, "attack": ..., "fused_score": float(0-100), "model": ..., "domain": ...}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_POC = HERE.parents[1]          # .../poc
_ROOT = HERE.parents[2]         # repo root
for _p in (str(_POC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- PROD FUSED flags MUST be set before importing the pipeline --------------
os.environ["DRAFTPROOF_V7_TIER_AUTHORITY"] = "1"
os.environ["DRAFTPROOF_V7_DEEP_SCAN"] = "1"
os.environ["DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN"] = "1"
os.environ["DRAFTPROOF_DEBERTA_AUTHORITATIVE"] = "1"

# --- Modal creds: .env is NOT auto-loaded, and a missing URL makes
# call_deep_scan fail-open to None (composite-only) SILENTLY — which would make
# this a fake "fused" benchmark. Load the two Modal vars from .env and FAIL LOUD
# if the endpoint is still unset, so a paid run can never be silently composite-only.
for _envfile in (_ROOT / ".env", _POC / ".env"):
    if _envfile.exists():
        for _line in _envfile.read_text().splitlines():
            _line = _line.strip()
            if _line.startswith("DRAFTPROOF_MODAL_ENDPOINT_"):
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
if not os.environ.get("DRAFTPROOF_MODAL_ENDPOINT_URL"):
    raise SystemExit(
        "score_subset: DRAFTPROOF_MODAL_ENDPOINT_URL is unset — the fused path would "
        "fail-open to composite-only (no Modal). Set it (or add it to .env) before scoring."
    )

DEFAULT_SUBSET = HERE / "subset.jsonl"
DEFAULT_OUT = HERE / "scores.jsonl"


def _read_subset(path: Path, limit: int | None) -> list[dict]:
    if not path.exists():
        sys.exit(f"subset not found: {path}  (run fetch_subset.py first)")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def _fused_score(rep: dict, ctx: str) -> float:
    badge = rep.get("ai_risk_badge")
    if not badge:
        raise RuntimeError(f"scan_text returned NO ai_risk_badge ({ctx}) — cannot score.")
    raw = badge.get("ai_likelihood_score")
    if raw is None:
        raise RuntimeError(f"ai_likelihood_score missing from badge ({ctx}) — cannot score.")
    score = float(raw)
    if not (0.0 <= score <= 100.0):
        raise RuntimeError(f"fused score {score} out of 0-100 range ({ctx}).")
    return score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="Score only the first N rows.")
    args = ap.parse_args()

    rows = _read_subset(args.subset, args.limit)
    print(f"[score] {len(rows)} rows from {args.subset}  (fused flags ON)", flush=True)

    # Heavy imports AFTER env flags are set.
    from calibration.measure_end_to_end import scan_text
    from calibration.retune.deepscan_cache import DEFAULT_CACHE
    from calibration.v12_validation import measure as m
    from detect.run import DetectionRunner

    # Free re-runs: wrap the Modal deep-scan client with the shared JSONL cache.
    m.install_cached_deep_scan(DEFAULT_CACHE)
    print(f"[score] deep-scan cache: {DEFAULT_CACHE}", flush=True)

    runner = DetectionRunner()
    out_rows: list[dict] = []
    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            ctx = f"row {i} label={r.get('label')} attack={r.get('attack')} " \
                  f"model={r.get('model')} domain={r.get('domain')}"
            rep = scan_text(runner, r["text"])
            score = _fused_score(rep, ctx)
            # Raw fusion components for the weight sweep: composite (0-100) and the
            # Modal desklib flagged sentence proportion (0-1). From tier_authority
            # provenance so the sweep can recompute fused = (1-w)*composite + w*prop*100.
            _ta = (rep.get("ai_risk_badge") or {}).get("tier_authority") or {}
            rec = {
                "label": int(r["label"]),
                "attack": r.get("attack"),
                "fused_score": round(score, 4),
                "composite": _ta.get("composite_score"),
                "modal_proportion": _ta.get("proportion"),
                "model": r.get("model"),
                "domain": r.get("domain"),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            out_rows.append(rec)
            if args.limit or (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(rows)}] label={rec['label']} "
                      f"attack={rec['attack']} fused={rec['fused_score']}", flush=True)

    dt = time.time() - t0
    print(f"[score] wrote {len(out_rows)} rows -> {args.out}  ({dt:.0f}s)", flush=True)

    hum = [x["fused_score"] for x in out_rows if x["label"] == 0]
    ai = [x["fused_score"] for x in out_rows if x["label"] == 1]
    if hum:
        print(f"[score] human  n={len(hum)} mean={sum(hum) / len(hum):.1f}")
    if ai:
        print(f"[score] ai     n={len(ai)} mean={sum(ai) / len(ai):.1f}")


if __name__ == "__main__":
    main()
