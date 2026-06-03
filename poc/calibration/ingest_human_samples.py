#!/usr/bin/env python3
"""Ingest MODERN, certain-human writing into the authorship corpus.

Drop plain-text files (one sample each, ~150-400 words of genuine human academic prose -- your own
pre-AI writing, students' work, anything you KNOW is human) into:  poc/calibration/human_dropbox/
Then run this. Each file becomes a human case tagged source='user_modern' so calibrate_authorship.py
can prefer the in-domain modern set over the older public-domain texts when recalibrating.

Run: ~/.pyenv/versions/3.11.0/bin/python3 poc/calibration/ingest_human_samples.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DROP = HERE / "human_dropbox"
OUT = HERE / "authorship_cases"


def main() -> int:
    DROP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in DROP.glob("*.txt") if p.is_file())
    if not files:
        print(f"No .txt files in {DROP} — drop modern human samples there (one per file) and re-run.")
        return 0
    made = 0
    for p in files:
        text = re.sub(r"\s+\n", "\n", p.read_text(encoding="utf-8", errors="ignore")).strip()
        if len(text.split()) < 80:
            print(f"   - {p.name}: under 80 words, skipped")
            continue
        cid = "human_user_" + re.sub(r"[^a-z0-9]+", "_", p.stem.lower()).strip("_")
        (OUT / f"{cid}.json").write_text(json.dumps({
            "case_id": cid, "authorship": "human", "source": "user_modern",
            "words": len(text.split()), "text": text,
        }, indent=2, ensure_ascii=False))
        made += 1
        print(f"   + {cid} ({len(text.split())} words)")
    print(f"\ningested {made} modern human samples. Re-run calibrate_authorship.py for an in-domain baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
