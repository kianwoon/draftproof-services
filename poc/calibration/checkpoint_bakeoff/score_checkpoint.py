#!/usr/bin/env python3
"""Checkpoint bake-off Stage 0 — score one HF checkpoint over the frozen eval pack.

Eval pack (all local, free):
  - retune corpus manifest: 272 SCoCESLE ESL humans + AI cases (incl. 31 GPT-5.6
    + the casual-register live-scan doc d449aca9)
  - RAID 600-doc subset (raid_benchmark/subset.jsonl): native-register humans
    (reddit/reviews/poetry/wiki...) + AI with none/paraphrase attacks

Scoring mirrors production's sentence pipeline: deberta_signal.split_sentences
-> deberta_windowing.build_windows(size=3, step=1) -> checkpoint inference ->
aggregate() per-sentence means. The checkpoint is injected via
DRAFTPROOF_DEBERTA_MODEL (poc/detect/deberta_model.py is architecture-agnostic:
any 2-label sequence-classification head works; AI-label index resolved from
id2label). One JSONL row per doc: {pop, family, attack, scores[]}.

Usage (one process per checkpoint — the model loader is a module singleton):
    DRAFTPROOF_DEBERTA_MODEL=<hf-id> python score_checkpoint.py --out scores.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from detect.deberta_signal import split_sentences  # noqa: E402
from detect.deberta_windowing import build_windows, aggregate  # noqa: E402
from detect.deberta_model import score_windows  # noqa: E402


def eval_pack() -> list[dict]:
    docs = []
    man = json.loads((POC / "calibration/retune/corpus/manifest.json").read_text())["rows"]
    for r in man:
        p = Path(r["source_path"])
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if p.suffix == ".json":
            text = json.loads(text).get("text") or ""
        pop = "esl_human" if r["label"] == "human" else "corpus_ai"
        docs.append({"id": r["id"], "pop": pop, "family": r["family"],
                     "attack": None, "text": text})
    subset = POC / "calibration/raid_benchmark/subset.jsonl"
    if subset.exists():
        for i, line in enumerate(subset.read_text().splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            pop = "raid_human" if r["label"] == 0 else "raid_ai"
            docs.append({"id": f"raid_{i:03d}", "pop": pop,
                         "family": r.get("model"), "attack": r.get("attack"),
                         "text": r.get("text") or ""})
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    model = os.environ.get("DRAFTPROOF_DEBERTA_MODEL", "").strip()
    if not model:
        sys.exit("set DRAFTPROOF_DEBERTA_MODEL to the checkpoint id to score")

    docs = eval_pack()
    if args.limit:
        docs = docs[: args.limit]
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    print(f"[{model}] {len(docs)} docs, {len(done)} already scored", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as out:
        for n, d in enumerate(docs):
            if d["id"] in done or not d["text"].strip():
                continue
            sents = [s for s in split_sentences(d["text"]) if s.strip()]
            if not sents:
                continue
            windows = build_windows(sents, size=3, step=1)
            probs = score_windows(windows)
            if probs is None:
                sys.exit(f"model inference unavailable for {model} — aborting (fail loud)")
            agg = aggregate(sents, windows, probs, size=3, step=1)
            scores = [s for s in agg["sentence_scores"] if s is not None]
            out.write(json.dumps({"id": d["id"], "pop": d["pop"], "family": d["family"],
                                  "attack": d["attack"], "n_sent": len(sents),
                                  "scores": [round(float(x), 6) for x in scores]}) + "\n")
            out.flush()
            if (n + 1) % 50 == 0:
                print(f"  {n + 1}/{len(docs)}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
