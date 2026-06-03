#!/usr/bin/env python3
"""Fetch the HUMAN half of the authorship corpus from Project Gutenberg (pre-2020, public-domain,
certainly human-authored). Argumentative/essay/education works chosen to stay closer to the product's
academic-prose domain. CAVEAT (user-accepted): older/literary style is a domain mismatch vs modern
student essays, so separation may be easier than in production -- every case is tagged source=gutenberg:<id>
so calibrate_authorship.py results can be checked for being driven by the easy (older) texts.

Run: ~/.pyenv/versions/3.11.0/bin/python3 poc/calibration/build_human_corpus.py
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "authorship_cases"

# (gutenberg_id, short_label) — essays/argument/education, certain human, public domain.
WORKS = [
    (34901, "mill_on_liberty"),       # J.S. Mill, On Liberty (argumentative academic prose)
    (2945, "emerson_essays"),          # Emerson, Essays: First Series
    (575, "bacon_essays"),             # Bacon, Essays
    (10615, "huxley_lectures"),        # Huxley, Lay Sermons / lectures (science education)
    (4239, "james_talks_teachers"),    # W. James, Talks to Teachers on Psychology (education)
]
PER_WORK = 3
MIN_WORDS, MAX_WORDS = 180, 320


def _fetch(gid: int) -> str | None:
    for url in (f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 calib"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            print(f"   ! {gid} {url.split('/')[-1]}: {type(exc).__name__}")
    return None


def _body(raw: str) -> str:
    m = re.search(r"\*\*\*\s*START OF.*?\*\*\*", raw, re.S)
    if m:
        raw = raw[m.end():]
    m = re.search(r"\*\*\*\s*END OF", raw, re.S)
    if m:
        raw = raw[:m.start()]
    return raw


def _passages(body: str) -> list[str]:
    # paragraphs = blank-line separated; accumulate prose paragraphs into ~250-word passages
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", body)]
    paras = [p for p in paras if len(p.split()) >= 25 and not p.isupper() and p[:1].isalpha()]
    out, buf = [], []
    for p in paras:
        buf.append(p)
        wc = sum(len(x.split()) for x in buf)
        if wc >= MIN_WORDS:
            text = " ".join(buf)
            if wc <= MAX_WORDS + 80:
                out.append(text)
            buf = []
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    made = 0
    for gid, label in WORKS:
        raw = _fetch(gid)
        if not raw:
            print(f"skip {label}: fetch failed")
            continue
        passages = _passages(_body(raw))
        # skip the first couple (prefaces/intros) and take from the middle for representative prose
        chosen = passages[2:2 + PER_WORK] if len(passages) > PER_WORK + 2 else passages[:PER_WORK]
        for i, text in enumerate(chosen):
            cid = f"human_{label}_{i:02d}"
            (OUT / f"{cid}.json").write_text(json.dumps({
                "case_id": cid, "authorship": "human", "source": f"gutenberg:{gid}",
                "words": len(text.split()), "text": text,
            }, indent=2, ensure_ascii=False))
            made += 1
            print(f"   + {cid} ({len(text.split())} words)")
    print(f"\nwrote {made} human cases to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
