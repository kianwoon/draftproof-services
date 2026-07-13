#!/usr/bin/env python3
"""Bake-off scorer for DESKLIB-architecture checkpoints (custom head: mean
pooling over the DeBERTa backbone + single-logit linear classifier, sigmoid =
AI probability). Architecture ported VERBATIM from
modal_endpoints/deberta_large_detector.py (the prod Modal loader) — the plain
AutoModelForSequenceClassification loader in score_checkpoint.py cannot load
these checkpoints. Same eval pack + windowing as score_checkpoint.py.

Usage:
    python score_checkpoint_desklib.py --model desklib/ai-text-detector-v1.01 --out scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent.parent
for p in (str(POC), str(POC.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from detect.deberta_signal import split_sentences  # noqa: E402
from detect.deberta_windowing import build_windows, aggregate  # noqa: E402
from score_checkpoint import eval_pack  # noqa: E402


def build_scorer(model_id: str):
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

    class DesklibAIDetectionModel(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.init_weights()

        def forward(self, input_ids, attention_mask=None):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs[0]
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = torch.sum(last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
            return {"logits": self.classifier(pooled)}

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = DesklibAIDetectionModel.from_pretrained(model_id)
    mdl.to(device).eval()
    print(f"loaded {model_id} on {device}", flush=True)

    def score_windows(windows: list[str], batch: int = 8) -> list[float]:
        out: list[float] = []
        for i in range(0, len(windows), batch):
            chunk = windows[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = mdl(enc["input_ids"], attention_mask=enc["attention_mask"])["logits"]
                probs = torch.sigmoid(logits).squeeze(-1)
            out.extend(float(x) for x in probs.detach().cpu().tolist())
        return out

    return score_windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    score_windows = build_scorer(args.model)
    docs = eval_pack()
    if args.limit:
        docs = docs[: args.limit]
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    print(f"[{args.model}] {len(docs)} docs, {len(done)} already scored", flush=True)
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
