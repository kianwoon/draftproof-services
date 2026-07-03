"""Layer-2 diversity check for the V7 deep-scan detector (authority re-base gate input).

The shipped operating point (sent_threshold/doc_floor from poc/detect_v7/weights.json)
was calibrated with a 65-text AI set from ONE generator family (claude-haiku-4.5).
This script measures per-family TPR at that operating point against NEW generator
families, generated on the SAME topics with the SAME plain-student-ask prompt
(no anti-detection instructions of any kind — this measures honest AI use).

Phase 1 (generate): ~N essays per family via Cerebras (gpt-oss-120b — DraftProof's own
rewrite model, so this doubles as a self-test) and OpenRouter (one OpenAI-family, one
Google-family model, ids VERIFIED against the live /models catalog, never guessed).
Cases land in poc/calibration/authorship_cases_v2/ with the same schema as the
originals. Phase 2 (score): sentence-level Modal scoring (same methodology as
v7_deberta_academic_calibrate.py), per-family TPR + proportion distributions, plus a
5-case consistency re-check of the original family. Results (numbers only) go to
poc/calibration/v7_deberta_diversity_v2.json.

Usage (from repo root; reads .env at repo root for all keys):
    python poc/calibration/v7_deberta_diversity_check.py            # both phases
    python poc/calibration/v7_deberta_diversity_check.py --score-only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_POC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_POC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detect.deberta_windowing import split_sentences  # noqa: E402

HERE = Path(__file__).resolve().parent
CASES_V1 = HERE / "authorship_cases"
CASES_V2 = HERE / "authorship_cases_v2"
OUT_PATH = HERE / "v7_deberta_diversity_v2.json"
ESSAYS_PER_FAMILY = 20
MIN_SENTENCE_WORDS = 4   # mirrors v7_deberta_academic_calibrate.py
BATCH_SIZE = 25          # mirrors v7_deberta_academic_calibrate.py
GEN_PROMPT = (
    "Write an essay of about 300 words on {topic}. "
    "Write in plain prose paragraphs without headings, lists, or markdown."
)
GEN_TEMPERATURE = 0.6

# Candidate ids per family, in preference order — the live OpenRouter /models list
# decides which actually exists; nothing here is assumed available.
OPENROUTER_CANDIDATES = {
    "openai": ["openai/gpt-5-mini", "openai/gpt-5", "openai/gpt-4.1-mini", "openai/gpt-4o-mini"],
    "google": ["google/gemini-2.5-flash", "google/gemini-2.5-pro", "google/gemini-2.0-flash-001"],
}


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = _ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    env.update({k: v for k, v in os.environ.items() if k in (
        "CEREBRAS_API_KEY", "OPENROUTER_API_KEY",
        "DRAFTPROOF_MODAL_ENDPOINT_URL", "DRAFTPROOF_MODAL_ENDPOINT_TOKEN",
    ) and v})
    return env


def _deep_scan_config() -> dict:
    weights = json.loads((_POC / "detect_v7" / "weights.json").read_text())
    return weights["deep_scan_calibration"]


def _post_json(url: str, body: dict, headers: dict, timeout: int = 120, retries: int = 3) -> dict:
    data = json.dumps(body).encode()
    # Custom User-Agent is REQUIRED: Cerebras (Cloudflare-fronted) returns 403 for
    # python-urllib's default UA — verified live 2026-07-04 (curl 200, urllib 403).
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "User-Agent": "DraftProof-calibration/1.0",
        **headers,
    }, method="POST")
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last = e
            wait = 5 * (attempt + 1)
            print(f"    [retry {attempt+1}/{retries}] {e} — waiting {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"POST {url} failed after {retries} retries: {last}")


def _chat(base_url: str, api_key: str, model: str, topic: str) -> str:
    resp = _post_json(
        f"{base_url}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": GEN_PROMPT.format(topic=topic)}],
            "temperature": GEN_TEMPERATURE,
            "max_tokens": 900,
        },
        {"Authorization": f"Bearer {api_key}"},
    )
    text = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return text.strip()


def _pick_openrouter_models(api_key: str) -> dict[str, str]:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        available = {m["id"] for m in json.loads(r.read().decode()).get("data", [])}
    picked: dict[str, str] = {}
    for family, candidates in OPENROUTER_CANDIDATES.items():
        chosen = next((c for c in candidates if c in available), None)
        if chosen is None:
            prefix = "openai/gpt-" if family == "openai" else "google/gemini-"
            fallback = sorted(m for m in available if m.startswith(prefix) and "oss" not in m)
            chosen = fallback[0] if fallback else None
        if chosen is None:
            raise RuntimeError(f"no available OpenRouter model found for family {family!r}")
        picked[family] = chosen
    return picked


def _topics() -> list[str]:
    topics: list[str] = []
    for p in sorted(glob.glob(str(CASES_V1 / "ai_*.json"))):
        d = json.loads(Path(p).read_text())
        t = d.get("topic")
        if t and t not in topics:
            topics.append(t)
    if not topics:
        raise RuntimeError("no topics found in existing AI cases")
    return topics


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def generate(env: dict[str, str]) -> None:
    CASES_V2.mkdir(exist_ok=True)
    topics = _topics()
    or_models = _pick_openrouter_models(env["OPENROUTER_API_KEY"])
    families = [
        ("https://api.cerebras.ai/v1", env["CEREBRAS_API_KEY"], "gpt-oss-120b"),
        ("https://openrouter.ai/api/v1", env["OPENROUTER_API_KEY"], or_models["openai"]),
        ("https://openrouter.ai/api/v1", env["OPENROUTER_API_KEY"], or_models["google"]),
    ]
    print(f"topics: {len(topics)}; families: {[m for _, _, m in families]}", flush=True)
    for base_url, key, model in families:
        slug = _slug(model)
        existing = len(glob.glob(str(CASES_V2 / f"ai_{slug}_*.json")))
        for i in range(existing, ESSAYS_PER_FAMILY):
            topic = topics[i % len(topics)]
            try:
                text = _chat(base_url, key, model, topic)
            except RuntimeError as e:
                print(f"  [{slug} #{i}] generation failed permanently: {e}", flush=True)
                continue
            if len(text.split()) < 120:
                print(f"  [{slug} #{i}] too short ({len(text.split())}w) — skipped", flush=True)
                continue
            case = {
                "case_id": f"ai_{slug}_{i:02d}",
                "authorship": "ai",
                "source": model,
                "temperature": GEN_TEMPERATURE,
                "topic": topic,
                "words": len(text.split()),
                "text": text,
            }
            (CASES_V2 / f"ai_{slug}_{i:02d}.json").write_text(json.dumps(case, indent=1))
            print(f"  [{slug} #{i}] {case['words']}w on '{topic}'", flush=True)


def _score_essay(url: str, token: str, text: str) -> tuple[float | None, int]:
    sents = [s for s in split_sentences(text) if len(s.split()) >= MIN_SENTENCE_WORDS]
    if not sents:
        return None, 0
    scores: list[float] = []
    for i in range(0, len(sents), BATCH_SIZE):
        resp = _post_json(
            url, {"chunks": sents[i:i + BATCH_SIZE]},
            {"Authorization": f"Bearer {token}"}, timeout=90,
        )
        if not resp.get("available", True):
            raise RuntimeError(f"endpoint unavailable: {resp}")
        scores.extend(float(s) for s in resp["chunk_scores"])
    return scores, len(sents)


def _dist(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"n": len(s), "min": round(s[0], 4), "p10": round(pick(0.10), 4),
            "p50": round(pick(0.50), 4), "p90": round(pick(0.90), 4), "max": round(s[-1], 4)}


def score(env: dict[str, str]) -> None:
    cfg = _deep_scan_config()
    sent_thr, doc_floor = cfg["sent_threshold"], cfg["doc_floor"]
    url, token = env["DRAFTPROOF_MODAL_ENDPOINT_URL"], env["DRAFTPROOF_MODAL_ENDPOINT_TOKEN"]

    by_family: dict[str, list[tuple[str, str]]] = {}
    for p in sorted(glob.glob(str(CASES_V2 / "ai_*.json"))):
        d = json.loads(Path(p).read_text())
        by_family.setdefault(d["source"], []).append((d["case_id"], d["text"]))
    consistency = [(json.loads(Path(p).read_text())["case_id"], json.loads(Path(p).read_text())["text"])
                   for p in sorted(glob.glob(str(CASES_V1 / "ai_*.json")))[:5]]

    results: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                     "operating_point": {"sent_threshold": sent_thr, "doc_floor": doc_floor},
                     "human_side_baseline": "poc/calibration/v7_deberta_academic_baseline.json (reused, not re-scored)",
                     "families": {}, "consistency_check_original_family": {}}
    total_sentences = 0
    total_calls = 0
    for family, cases in by_family.items():
        props: dict[str, float] = {}
        for case_id, text in cases:
            scores, n_sents = _score_essay(url, token, text)
            if scores is None:
                continue
            total_sentences += n_sents
            total_calls += (n_sents + BATCH_SIZE - 1) // BATCH_SIZE
            prop = sum(1 for x in scores if x >= sent_thr) / len(scores)
            props[case_id] = round(prop, 4)
            print(f"  [{family}] {case_id}: proportion={prop:.3f} ({n_sents} sents)", flush=True)
        vals = list(props.values())
        results["families"][family] = {
            "n_essays": len(vals),
            "tpr_at_doc_floor": round(sum(1 for v in vals if v >= doc_floor) / len(vals), 4) if vals else None,
            "proportion_dist": _dist(vals),
            "per_essay": props,
        }
    cons: dict[str, float] = {}
    for case_id, text in consistency:
        scores, n_sents = _score_essay(url, token, text)
        if scores is None:
            continue
        total_sentences += n_sents
        total_calls += (n_sents + BATCH_SIZE - 1) // BATCH_SIZE
        cons[case_id] = round(sum(1 for x in scores if x >= sent_thr) / len(scores), 4)
    results["consistency_check_original_family"] = {
        "per_essay": cons,
        "flagged_at_floor": sum(1 for v in cons.values() if v >= doc_floor),
        "n": len(cons),
    }
    results["cost_proxies"] = {"modal_calls": total_calls, "sentences_scored": total_sentences}
    OUT_PATH.write_text(json.dumps(results, indent=1))
    print(f"\nsaved {OUT_PATH}", flush=True)
    for fam, r in results["families"].items():
        print(f"TPR[{fam}] @ floor {doc_floor}: {r['tpr_at_doc_floor']} (n={r['n_essays']})", flush=True)
    print(f"consistency (original family): {results['consistency_check_original_family']['flagged_at_floor']}/{len(cons)} flagged", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    env = _load_env()
    for k in ("CEREBRAS_API_KEY", "OPENROUTER_API_KEY", "DRAFTPROOF_MODAL_ENDPOINT_URL", "DRAFTPROOF_MODAL_ENDPOINT_TOKEN"):
        if not env.get(k):
            print(f"missing {k} (checked .env at {_ROOT / '.env'} and environment)", file=sys.stderr)
            raise SystemExit(2)
    if not args.score_only:
        generate(env)
    score(env)


if __name__ == "__main__":
    main()
