#!/usr/bin/env python3
"""Fine-tune v1 corpus builder (2026-07-14) — GPT-5.5-first strategy.

Streams candidate HF datasets (no full downloads), filters to long-form prose,
and writes a balanced training corpus to /tmp/finetune_v1/. The 31 GPT-5.6
docs + SCoCESLE stay OUT (held-out eval / license). Every row records its
source + license for auditability.

AI positives:
  - WithinUsAI/GPT_5.5_Distilled (apache-2.0)  — GPT-5.5 responses
  - Manusagents/GPT-5.5-...-Distillation-Dataset (mit) — GPT-5.5/Gemini-3.1/
    Grok-4/Claude-Fable-5/Qwen-3.7 responses (frontier family mix)
  - RAID subset AI rows (already local; incl. paraphrase attacks)
Humans:
  - RAID subset human rows (already local)
  - ELLIPSE / PERSUADE-style public essay corpora (searched at runtime)

Prose filter: >=120 words, <=2500 words, <15% code/markdown symbols, no
role/system artifacts. Assistant-register is EXPECTED for distill data —
this is the register-densifier slice, not the register-matched one.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OUT = Path("/tmp/finetune_v1")
OUT.mkdir(parents=True, exist_ok=True)

CODEY = re.compile(r"```|def |import |class |\{\"|<html|SELECT |#include")
ROLEY = re.compile(r"^\s*(system|user|assistant)\s*:", re.I)


def prose_ok(t: str) -> bool:
    if not isinstance(t, str):
        return False
    w = t.split()
    if not (120 <= len(w) <= 2500):
        return False
    if CODEY.search(t) or ROLEY.search(t):
        return False
    sym = sum(1 for c in t if c in "{}[]<>|#*`_=")
    return sym / max(len(t), 1) < 0.02


def text_of(row: dict) -> str | None:
    for k in ("response", "output", "completion", "text", "answer", "assistant", "content"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list):  # chat-format messages
            for m in reversed(v):
                if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
                    return m["content"]
    return None


def stream(dataset_id: str, label: int, source: str, license_: str, target: int, out_f, model_field: str | None = None):
    from datasets import load_dataset

    got = 0
    try:
        ds = load_dataset(dataset_id, split="train", streaming=True)
        for row in ds:
            t = text_of(row)
            if t is None or not prose_ok(t):
                continue
            out_f.write(json.dumps({
                "text": t.strip(), "label": label, "source": source,
                "license": license_,
                "model": row.get(model_field) if model_field else None,
            }, ensure_ascii=False) + "\n")
            got += 1
            if got >= target:
                break
    except Exception as e:  # noqa: BLE001 — report and continue with other sources
        print(f"  ! {dataset_id}: {type(e).__name__}: {e}", flush=True)
    print(f"  {dataset_id}: {got} rows", flush=True)
    return got


def main() -> None:
    ai_path = OUT / "ai_positives.jsonl"
    hum_path = OUT / "human_negatives.jsonl"

    with ai_path.open("w") as f:
        print("[AI positives]", flush=True)
        stream("WithinUsAI/GPT_5.5_Distilled", 1, "gpt5.5_distilled", "apache-2.0", 800, f)
        stream("Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset",
               1, "frontier_mix_distill", "mit", 1200, f, model_field="model")
        # RAID AI rows from the local subset (paraphrase retention)
        raid = Path(__file__).resolve().parent.parent / "raid_benchmark" / "subset.jsonl"
        n = 0
        for line in raid.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["label"] == 1 and prose_ok(r.get("text") or ""):
                f.write(json.dumps({"text": r["text"].strip(), "label": 1,
                                    "source": f"raid_{r.get('attack')}", "license": "raid",
                                    "model": r.get("model")}, ensure_ascii=False) + "\n")
                n += 1
        print(f"  raid_ai: {n} rows", flush=True)

    with hum_path.open("w") as f:
        print("[Human negatives]", flush=True)
        raid = Path(__file__).resolve().parent.parent / "raid_benchmark" / "subset.jsonl"
        n = 0
        for line in raid.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["label"] == 0 and prose_ok(r.get("text") or ""):
                f.write(json.dumps({"text": r["text"].strip(), "label": 0,
                                    "source": f"raid_human_{r.get('domain')}", "license": "raid",
                                    "model": None}, ensure_ascii=False) + "\n")
                n += 1
        print(f"  raid_human: {n} rows", flush=True)
        # public student/ESL essay corpora — try known ids, keep whichever loads
        for did in ("ELLIPSE-Corpus/ELLIPSE", "NanyangTech/ICNALE", "qwedsacf/ivypanda-essays",
                    "ChristophSchuhmann/essays-with-instructions"):
            stream(did, 0, f"hf_{did.split('/')[-1].lower()}", "check-before-train", 700, f)

    for p in (ai_path, hum_path):
        print(p, sum(1 for _ in p.open()), "rows")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
