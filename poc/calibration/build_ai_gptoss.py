#!/usr/bin/env python3
"""Add gpt-oss-120b AI essays to the corpus via the PROJECT gateway (handles Cerebras auth; raw HTTP
got 403). Domain-matched to PERSUADE student prompts so the human/AI comparison isn't confounded by
topic. authorship='ai' (certain by construction). Writes to authorship_cases/ (our generation, no
license issue — safe to commit).

Run: ~/.pyenv/versions/3.11.0/bin/python3 poc/calibration/build_ai_gptoss.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "authorship_cases"

# allow-hardcode: PROMPT topics (PERSUADE-style student-essay prompts) for generating test fixtures,
# not a detect/scoring list in product logic.
PROMPTS = [
    "whether students should be required to do community service",
    "whether cell phones should be allowed in school",
    "the value of distance learning versus in-person classes",
    "whether the electoral college should be kept or abolished",
    "whether cars should be limited in cities to reduce pollution",
    "whether students benefit from seeking multiple opinions before deciding",
    "whether summer projects should be designed by students or teachers",
    "whether driverless cars are a good idea",
    "the importance of extracurricular activities for students",
    "whether failure is necessary for success",
]
SYSTEM = "You are a 6th-12th grade student writing a short argumentative essay. Write naturally."


def _load_env() -> None:
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def main() -> int:
    _load_env()
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "poc"))
    from poc.rewrite_v6.llm_config import (
        resolve_v6_model, resolve_v6_api_key, resolve_v6_base_url, writer_llm_profile, writer_extra_body,
    )
    from poc.llm.gateway import LLMGateway, LLMConfig

    model = "openai/gpt-oss-120b"
    resolved = resolve_v6_model(model) or model
    gw = LLMGateway(LLMConfig(
        model=resolved, api_key=resolve_v6_api_key(None), base_url=resolve_v6_base_url(None),
        **writer_llm_profile(resolved, "x"), extra_body=writer_extra_body(resolved),
        max_retries=1, timeout=120,
    ))
    OUT.mkdir(parents=True, exist_ok=True)
    made = 0
    for i, topic in enumerate(PROMPTS):
        cid = f"ai_gptoss_{i:02d}"
        path = OUT / f"{cid}.json"
        if path.exists():
            continue
        prompt = (f"Write a ~250-word argumentative essay arguing a clear position on {topic}. "
                  f"Use plain paragraphs, no headings, no lists.")
        try:
            resp = gw.chat(prompt, system=SYSTEM, temperature=0.7, top_p=0.95, max_tokens=700,
                           app_label="CorpusGen")
            text = (getattr(resp, "raw_content", "") or getattr(resp, "content", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            print(f"   ! {cid}: {type(exc).__name__}: {exc}")
            continue
        text = re.sub(r"\s+\n", "\n", text).strip()
        if len(text.split()) < 120:
            print(f"   - {cid}: too short ({len(text.split())}w)")
            continue
        path.write_text(json.dumps({
            "case_id": cid, "authorship": "ai", "source": "openai/gpt-oss-120b",
            "topic": topic, "words": len(text.split()), "text": text,
        }, indent=2, ensure_ascii=False))
        made += 1
        print(f"   + {cid} ({len(text.split())} words)")
    print(f"\nwrote {made} gpt-oss AI cases to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
