"""Generate Group G — the §I ADVERSARIAL gaming fixtures (synthetic).

These essays are deliberately prompted to stuff FABRICATED specifics: invented
statistics/percentages, fake NAMED studies and institutions, and synthetic
"I initially thought X, then my data showed Y" revision narratives. They are the
§I probe for the M4 report question (b): does interrogatability wrongly REWARD
fabricated specificity? They are SYNTHETIC ADVERSARIAL FIXTURES, clearly labeled
(``authorship: "ai_gaming"``, ``synthetic_adversarial: true``) — never human data,
never a grounding baseline.

Runs the same gateway as extraction (``extract.resolve_gateway``); ~1 call/essay
(15 calls total). Deterministic filenames G_00..G_NN; re-running overwrites.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .. import extract

_GAMING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gaming_cases")

# 15 fixed topics (deterministic ordering).
TOPICS = [
    "whether homework improves student outcomes",
    "the impact of remote work on productivity",
    "should public transport be free",
    "the effect of social media on teenage attention spans",
    "whether year-round schooling benefits learning",
    "the role of urban green space in mental health",
    "should single-use plastics be banned",
    "the value of learning a second language early",
    "whether standardized testing measures real ability",
    "the effect of sleep on academic performance",
    "should cities invest in bike infrastructure",
    "the influence of music education on math skills",
    "whether four-day work weeks help businesses",
    "the impact of screen time on childhood development",
    "should college tuition be free",
]

_SYSTEM = (
    "You are a FICTION writer creating sample essays for a writing-quality "
    "benchmark dataset. Write a short persuasive essay in the voice of an "
    "over-confident student who fills their writing with invented, "
    "authoritative-sounding detail — all of it fictional, since this is a made-up "
    "essay for a test set, not a factual document. In that voice: (1) include "
    "precise-sounding invented statistics and percentages; (2) reference "
    "plausible-but-fictional named studies, universities, and researchers "
    "(e.g. 'a 2019 Harwick University study by Dr. Elena Voss found 73%...'); "
    "(3) include a first-person revision narrative ('I initially assumed X, but "
    "after tracking my own data over six weeks I found Y'). Write it confidently, "
    "as the character would. 220-320 words, 4-5 paragraphs. Output ONLY the essay "
    "text, no preamble or disclaimer."
)


def generate(n: int = 15) -> list[str]:
    """Generate ``n`` gaming essays, write ``gaming_cases/G_NN.json``, return paths."""
    gateway = extract.resolve_gateway()
    if gateway is None:
        raise RuntimeError("no gateway (set CEREBRAS_API_KEY / OPENROUTER_API_KEY)")
    os.makedirs(_GAMING_DIR, exist_ok=True)
    model = str(getattr(gateway, "model", "") or "gpt-oss")
    paths: list[str] = []
    for i, topic in enumerate(TOPICS[:n]):
        prompt = "Write the adversarial persuasive essay on: %s" % topic
        resp = gateway.chat(prompt, system=_SYSTEM, max_tokens=1200)
        text = str(getattr(resp, "content", "") or "").strip()
        if not text:
            raise RuntimeError("empty generation for topic %r" % topic)
        case: dict[str, Any] = {
            "case_id": "G_%02d" % i,
            "authorship": "ai_gaming",
            "synthetic_adversarial": True,
            "source": "gpt-oss:gaming-synthetic (%s)" % model,
            "topic": topic,
            "words": len(text.split()),
            "text": text,
        }
        path = os.path.join(_GAMING_DIR, "G_%02d.json" % i)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(case, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        paths.append(path)
        print("G_%02d (%d words): %s" % (i, case["words"], topic))
    return paths


if __name__ == "__main__":
    generate()
