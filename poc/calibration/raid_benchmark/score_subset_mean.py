#!/usr/bin/env python3
"""RAID re-test driver for the MEAN-sentence-score representation (2026-07-14).

Motivation: the 2026-07-09 weight sweep REJECTED replacing the deep-scan
proportion (fraction of desklib sentences >= 0.999) with the continuous mean
sentence score — it won in-sample but lost on the 70/30 holdout. GPT-5.6
changed the evidence: on the expanded retune corpus (272 ESL human + 91 AI),
mean beats proportion on every axis (AUC 0.985 vs 0.951, TPR@1%-ESL-FPR 76.9%
vs 63.7%, academic GPT-5.6 capture 38.7% vs 3.2%). Per the rejection protocol,
the candidate must RE-WIN on the RAID holdout before touching production.

What this does, per subset row (same pipeline as score_subset.py — importing
it sets the four PROD FUSED env flags and fails loud without Modal creds):
  - runs the full scan (composite + tier-authority fused path),
  - captures the RAW per-sentence desklib scores by wrapping
    detect_v7.modal_client.call_deep_scan with the retune deep-scan cache
    (poc/calibration/retune/cache/deepscan_scores.jsonl) — so repeat runs and
    the report-heatmap's second Modal call cost nothing extra,
  - writes {label, attack, model, domain, composite, modal_proportion,
    modal_mean, n_sentences} to --out (resume-safe: already-written row
    indices are skipped).

The holdout comparison itself is offline math over the output rows — see
holdout_compare.py.

Usage (from repo root):
    python poc/calibration/raid_benchmark/score_subset_mean.py --limit 5   # smoke
    python poc/calibration/raid_benchmark/score_subset_mean.py            # full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Importing score_subset performs the required setup at module level:
# sys.path, the four PROD FUSED env flags (BEFORE pipeline import), and the
# fail-loud Modal-creds check. Reuse it rather than duplicating the contract.
sys.path.insert(0, str(HERE))
import score_subset  # noqa: E402  (module-level side effects intentional)

from calibration.measure_end_to_end import scan_text  # noqa: E402
from calibration.retune import deepscan_cache  # noqa: E402
from detect.run import DetectionRunner  # noqa: E402
from detect_v7 import modal_client  # noqa: E402

DEFAULT_SUBSET = HERE / "subset.jsonl"
DEFAULT_OUT = HERE / "scores_mean.jsonl"


def install_caching_recorder(cache_path: Path, holder: dict) -> None:
    """Wrap modal_client.call_deep_scan: cache raw per-sentence scores by
    content key (checkpoint + joined sentences) and record the FIRST call's
    scores per document into `holder` (the tier-authority call runs first in
    the builder; the report heatmap's later call uses a different sentence
    segmentation and must not overwrite it)."""
    real = modal_client.call_deep_scan
    checkpoint = deepscan_cache.checkpoint_tag()
    cache = deepscan_cache.load_cache(cache_path)

    def wrapped(sentences, *args, **kwargs):
        key = deepscan_cache.content_key("\n".join(sentences), checkpoint)
        scores = cache.get(key)
        if scores is not None and len(scores) == len(sentences):
            resp = {"available": True, "calibrated": True, "chunk_scores": scores}
        else:
            resp = real(sentences, *args, **kwargs)
            chunk = resp.get("chunk_scores") if isinstance(resp, dict) else None
            if (isinstance(resp, dict) and resp.get("available") is True
                    and isinstance(chunk, list) and len(chunk) == len(sentences)):
                deepscan_cache.append(cache_path, key, chunk)
                cache[key] = chunk
            else:
                return resp
        holder.setdefault("scores", resp["chunk_scores"])
        return resp

    modal_client.call_deep_scan = wrapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache", type=Path, default=deepscan_cache.DEFAULT_CACHE)
    args = ap.parse_args()

    rows = score_subset._read_subset(args.subset, args.limit)
    done: set[int] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["i"])
    print(f"{len(rows)} rows, {len(done)} already scored -> {args.out}", flush=True)

    holder: dict = {}
    install_caching_recorder(args.cache, holder)
    runner = DetectionRunner()

    with args.out.open("a") as out:
        for i, row in enumerate(rows):
            if i in done:
                continue
            holder.clear()
            text = (row.get("text") or "").strip()
            if not text:
                continue
            rep = scan_text(runner, text)
            badge = rep.get("ai_risk_badge") or {}
            ta = badge.get("tier_authority") or {}
            scores = holder.get("scores") or []
            rec = {
                "i": i,
                "label": row.get("label"),
                "attack": row.get("attack"),
                "model": row.get("model"),
                "domain": row.get("domain"),
                "composite": ta.get("composite_score"),
                "modal_proportion": ta.get("proportion"),
                "modal_mean": (sum(scores) / len(scores)) if scores else None,
                "n_sentences": len(scores),
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(rows)} scored", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
