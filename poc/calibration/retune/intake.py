"""Phase 1 intake: generate AI essays from models.json and PERSIST them to disk
(one JSON per case) so the corpus is never a set of numbers that silently drifts."""
from __future__ import annotations
import json, os, re, urllib.request
from pathlib import Path
from .generators import Generator, PROVIDER_BASE_URLS

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent  # poc/ -> repo root
DEFAULT_OUT = HERE.parent / "authorship_cases"

# allow-hardcode: PROMPT topics for test-fixture essays, not a detect/scoring list.
TOPICS = [
    "the role of technology in modern classrooms",
    "whether standardized testing measures real learning",
    "how social media has changed public discourse",
    "the ethics of artificial intelligence in hiring",
    "the impact of remote work on team collaboration",
    "why critical thinking matters more than memorization",
    "the trade-offs of renewable energy adoption",
    "how reading fiction shapes empathy",
    "whether governments should make public transport free",
    "whether homework should be abolished in schools",
    "whether a university degree is still worth its cost",
    "whether social media does more harm than good for teenagers",
    "whether voting should be made mandatory",
    "whether children should be allowed smartphones in school",
    "whether developed countries should accept more refugees",
    "whether zoos do more good than harm",
]
SYSTEM = "You are a student writing a short academic essay. Write naturally and clearly."

def load_env() -> None:
    candidates = [REPO / ".env"] + [up / ".env" for up in REPO.parents]
    for env in candidates:
        if not env.exists():
            continue
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))

def _chat(model, base_url, api_key, prompt, temperature) -> str | None:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": temperature, "max_tokens": 700,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "User-Agent": "draftproof-retune/1.0"},  # custom UA: some providers 403 urllib default
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as exc:  # noqa: BLE001
        print(f"   ! {model} failed: {type(exc).__name__}: {exc}")
        return None

def generate_ai_essays(out_dir: Path, generators: list[Generator], topics: list[str], chat_fn=_chat) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for gen in generators:
        base_url, key_env = PROVIDER_BASE_URLS[gen.provider]
        api_key = os.environ.get(key_env, "").strip()
        if not api_key:
            print(f"skip {gen.id}: {key_env} not set")
            continue
        short = re.sub(r"[^a-z0-9]+", "_", gen.id.split("/")[-1].lower())
        for i, topic in enumerate(topics):
            cid = f"ai_{short}_{i:02d}"
            path = out_dir / f"{cid}.json"
            if path.exists():
                continue
            temp = 0.6 + 0.1 * (i % 4)
            prompt = (f"Write a ~250-word academic essay on {topic}. Use clear paragraphs, "
                      f"no headings, no lists, no citations.")
            text = chat_fn(gen.id, base_url, api_key, prompt, temp)
            if not text or len(text.split()) < 120:
                print(f"   - {cid}: too short / empty, skipped")
                continue
            path.write_text(json.dumps({
                "case_id": cid, "authorship": "ai", "source": gen.id, "family": gen.family,
                "temperature": temp, "topic": topic, "words": len(text.split()), "text": text,
            }, indent=2, ensure_ascii=False))
            made += 1
            print(f"   + {cid} ({len(text.split())} words)")
    print(f"\nwrote {made} AI cases to {out_dir}")
    return made
