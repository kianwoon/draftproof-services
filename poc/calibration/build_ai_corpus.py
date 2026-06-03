#!/usr/bin/env python3
"""Generate the AI half of the authorship-labeled calibration corpus.

Labels are certain BY CONSTRUCTION: every passage here is produced by an LLM, so authorship='ai'.
Writes one JSON per passage into authorship_cases/ in the same shape calibrate_authorship.py reads.
Uses the providers already configured in .env (Cerebras gpt-oss-120b, OpenRouter qwen) via direct
OpenAI-compatible HTTP — no project imports needed for generation.

Run: ~/.pyenv/versions/3.11.0/bin/python3 poc/calibration/build_ai_corpus.py
Idempotent-ish: skips a (model,topic) whose output file already exists.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "authorship_cases"

# allow-hardcode: these are PROMPT topics for generating sample essays (test fixtures), not a
# detect/allow/scoring list in product logic. Academic-style to match the product's real use case.
TOPICS = [
    "the role of technology in modern classrooms",
    "whether standardized testing measures real learning",
    "how social media has changed public discourse",
    "the ethics of artificial intelligence in hiring",
    "the impact of remote work on team collaboration",
    "why critical thinking matters more than memorization",
    "the trade-offs of renewable energy adoption",
    "how reading fiction shapes empathy",
]

# (model, base_url, api_key_env) — both are LLMs, so all output is authorship='ai'.
PROVIDERS = [
    ("openai/gpt-oss-120b", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    ("qwen/qwen-2.5-7b-instruct", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
]

SYSTEM = "You are a student writing a short academic essay. Write naturally and clearly."


def _load_env() -> None:
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def _chat(model: str, base_url: str, api_key: str, prompt: str, temperature: float) -> str | None:
    body = json.dumps({
        "model": model.split("/")[-1] if "cerebras" in base_url else model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": temperature, "max_tokens": 700,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"   ! {model} failed: {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    _load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    made = 0
    for model, base_url, key_env in PROVIDERS:
        api_key = os.environ.get(key_env, "").strip()
        if not api_key:
            print(f"skip {model}: {key_env} not set")
            continue
        short = re.sub(r"[^a-z0-9]+", "_", model.split("/")[-1].lower())
        for i, topic in enumerate(TOPICS):
            cid = f"ai_{short}_{i:02d}"
            path = OUT / f"{cid}.json"
            if path.exists():
                continue
            temp = 0.6 + 0.1 * (i % 4)  # vary 0.6..0.9 for stylistic diversity
            prompt = (f"Write a ~250-word academic essay on {topic}. Use clear paragraphs, no headings, "
                      f"no lists, no citations.")
            text = _chat(model, base_url, api_key, prompt, temp)
            if not text or len(text.split()) < 120:
                print(f"   - {cid}: too short / empty, skipped")
                continue
            path.write_text(json.dumps({
                "case_id": cid, "authorship": "ai", "source": model, "temperature": temp,
                "topic": topic, "words": len(text.split()), "text": text,
            }, indent=2, ensure_ascii=False))
            made += 1
            print(f"   + {cid} ({len(text.split())} words)")
    print(f"\nwrote {made} AI cases to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
