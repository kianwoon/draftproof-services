# V7 Re-Tune Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn V7's one-off calibration scripts into a versioned, gated, manual-on-demand re-tune workflow — Phase 1 (corpus intake + manifest) and Phase 2 (re-calibration + FPR-gate oracle), driven by a plain CLI.

**Architecture:** A new `poc/calibration/retune/` package. Phase 1 centralizes generator model IDs into `models.json`, generates + **persists** AI essays, and builds a versioned `manifest.json` over the existing `authorship_cases/` + SCoCESLE corpus. Phase 2 sequences the existing calibration scripts into a staging dir, then runs `fpr_subgroup_gate.py --compare` as the single ship/no-ship oracle. A `run_cycle.py` orchestrator ties them together and appends a `RETUNE_LOG.md` decision row. No production scoring code changes; the workflow only produces **candidate** artifacts.

**Tech Stack:** Python 3.11, stdlib only for new code (json, pathlib, hashlib, subprocess, urllib, datetime, argparse). pytest for tests. Reuses existing `poc/calibration/*.py`.

## Global Constraints

- **NO HARDCODE** of model IDs in logic — generator IDs live in `models.json` data, not code.
- **SCoCESLE text is `local_only`** — never written into any repo-committed artifact. The license guard enforces this.
- **The FPR subgroup gate is the acceptance oracle** — its exit code (0 pass / 1 regression / 2 corpus-missing) is authoritative; no candidate is promotable on a non-zero regression exit.
- **New AI-case JSON keeps the existing schema** exactly: `{case_id, authorship, source, temperature, topic, words, text}` (+ additive `family`). `authorship=="ai"` is what the gate filters on (`fpr_subgroup_gate.py:81`).
- **`.env` lives in the MAIN repo root**, not the worktree — reuse the walk-up `_load_env` pattern from `build_ai_corpus.py:58-70`.
- New files stay well under the 1500-line limit; each module has one responsibility.
- Run tests with the full ML python: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest`.

---

### Task 1: Generator config — `models.json` + loader

**Files:**
- Create: `poc/calibration/retune/__init__.py` (empty)
- Create: `poc/calibration/retune/models.json`
- Create: `poc/calibration/retune/generators.py`
- Test: `poc/calibration/retune/test_generators.py`

**Interfaces:**
- Produces: `load_generators(path: Path | None = None) -> list[Generator]` where `Generator` is a dataclass `(id: str, provider: str, family: str, n_per_topic: int)`. Raises `ValueError` if a row is missing a required key or `provider` is unknown.
- Produces: `PROVIDER_BASE_URLS: dict[str, tuple[str, str]]` mapping `provider -> (base_url, api_key_env)`.

- [ ] **Step 1: Write the failing test**

```python
# poc/calibration/retune/test_generators.py
import json
from pathlib import Path
import pytest
from poc.calibration.retune.generators import load_generators, Generator

def _write(tmp_path, rows):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"version": "test", "generators": rows}))
    return p

def test_loads_rows(tmp_path):
    p = _write(tmp_path, [
        {"id": "openai/gpt-5-mini", "provider": "openrouter", "family": "gpt-5", "n_per_topic": 1},
    ])
    gens = load_generators(p)
    assert gens == [Generator("openai/gpt-5-mini", "openrouter", "gpt-5", 1)]

def test_missing_key_raises(tmp_path):
    p = _write(tmp_path, [{"id": "x/y", "provider": "openrouter"}])  # no family/n_per_topic
    with pytest.raises(ValueError):
        load_generators(p)

def test_unknown_provider_raises(tmp_path):
    p = _write(tmp_path, [{"id": "x/y", "provider": "mystery", "family": "f", "n_per_topic": 1}])
    with pytest.raises(ValueError):
        load_generators(p)

def test_default_path_loads_committed_models(tmp_path):
    gens = load_generators()  # committed models.json
    fams = {g.family for g in gens}
    assert "gpt-5" in fams  # gpt-5-mini row is present (closes the silent-drift gap)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/calibration/retune/test_generators.py -v`
Expected: FAIL — `ModuleNotFoundError: poc.calibration.retune.generators`

- [ ] **Step 3: Create `models.json`**

```json
{
  "version": "2026-07-06",
  "generators": [
    {"id": "openai/gpt-4o-mini",         "provider": "openrouter", "family": "gpt-4",  "n_per_topic": 1},
    {"id": "anthropic/claude-haiku-4.5", "provider": "openrouter", "family": "claude", "n_per_topic": 1},
    {"id": "google/gemini-2.5-flash",    "provider": "openrouter", "family": "gemini", "n_per_topic": 1},
    {"id": "qwen/qwen-2.5-7b-instruct",  "provider": "openrouter", "family": "qwen",   "n_per_topic": 1},
    {"id": "openai/gpt-5-mini",          "provider": "openrouter", "family": "gpt-5",  "n_per_topic": 1}
  ]
}
```

- [ ] **Step 4: Write the loader**

```python
# poc/calibration/retune/generators.py
"""Single source of truth for AI-essay generators. Adding a new model (gpt-6) is a
one-row edit to models.json — no code change, no hardcode in logic."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = HERE / "models.json"

# provider -> (base_url, api_key_env). OpenRouter is the proven-clean path.
PROVIDER_BASE_URLS: dict[str, tuple[str, str]] = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

REQUIRED = ("id", "provider", "family", "n_per_topic")

@dataclass(frozen=True)
class Generator:
    id: str
    provider: str
    family: str
    n_per_topic: int

def load_generators(path: Path | None = None) -> list[Generator]:
    data = json.loads((path or DEFAULT_MODELS).read_text())
    out: list[Generator] = []
    for row in data.get("generators", []):
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise ValueError(f"generator row {row!r} missing keys: {missing}")
        if row["provider"] not in PROVIDER_BASE_URLS:
            raise ValueError(f"unknown provider {row['provider']!r}; known: {list(PROVIDER_BASE_URLS)}")
        out.append(Generator(row["id"], row["provider"], row["family"], int(row["n_per_topic"])))
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/calibration/retune/test_generators.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add poc/calibration/retune/__init__.py poc/calibration/retune/models.json poc/calibration/retune/generators.py poc/calibration/retune/test_generators.py
git commit -m "feat(retune): models.json generator config + loader (closes gpt-5-mini drift)"
```

---

### Task 2: AI-essay intake — generate + persist

**Files:**
- Create: `poc/calibration/retune/intake.py`
- Test: `poc/calibration/retune/test_intake.py`

**Interfaces:**
- Consumes: `load_generators`, `PROVIDER_BASE_URLS` (Task 1).
- Produces: `generate_ai_essays(out_dir: Path, generators: list[Generator], topics: list[str], chat_fn=_chat) -> int` — writes one JSON per (model, topic) to `out_dir`, returns count written. Idempotent: skips a `case_id` whose file exists. Injects `family` into each JSON. `chat_fn` is injectable for tests.
- Produces: `TOPICS: list[str]` (carried verbatim from `build_ai_corpus.py:26-44`).
- Produces: `_chat(model, base_url, api_key, prompt, temperature) -> str | None` and `load_env() -> None` (walk-up, from `build_ai_corpus.py`).

- [ ] **Step 1: Write the failing test**

```python
# poc/calibration/retune/test_intake.py
import json
from poc.calibration.retune.generators import Generator
from poc.calibration.retune import intake

def _fake_chat(model, base_url, api_key, prompt, temperature):
    return " ".join(["word"] * 150)  # >120 words so it isn't skipped

def test_generate_persists_and_tags_family(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    gens = [Generator("openai/gpt-5-mini", "openrouter", "gpt-5", 1)]
    n = intake.generate_ai_essays(tmp_path, gens, ["topic one"], chat_fn=_fake_chat)
    assert n == 1
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    d = json.loads(files[0].read_text())
    assert d["authorship"] == "ai"
    assert d["source"] == "openai/gpt-5-mini"
    assert d["family"] == "gpt-5"
    assert d["words"] >= 120

def test_generate_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    gens = [Generator("openai/gpt-5-mini", "openrouter", "gpt-5", 1)]
    intake.generate_ai_essays(tmp_path, gens, ["topic one"], chat_fn=_fake_chat)
    n2 = intake.generate_ai_essays(tmp_path, gens, ["topic one"], chat_fn=_fake_chat)
    assert n2 == 0  # already exists, skipped

def test_short_output_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    gens = [Generator("openai/gpt-5-mini", "openrouter", "gpt-5", 1)]
    n = intake.generate_ai_essays(tmp_path, gens, ["t"], chat_fn=lambda *a: "too short")
    assert n == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/calibration/retune/test_intake.py -v`
Expected: FAIL — `ModuleNotFoundError: poc.calibration.retune.intake`

- [ ] **Step 3: Write intake.py**

```python
# poc/calibration/retune/intake.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/calibration/retune/test_intake.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/calibration/retune/intake.py poc/calibration/retune/test_intake.py
git commit -m "feat(retune): AI-essay intake — generate from models.json, persist + tag family"
```

---

### Task 3: Corpus manifest — build, label, guards

**Files:**
- Create: `poc/calibration/retune/manifest.py`
- Test: `poc/calibration/retune/test_manifest.py`

**Interfaces:**
- Consumes: nothing from prior tasks (reads dirs directly).
- Produces: `build_manifest(ai_dir: Path, scocesle_dir: Path | None, now_iso: str) -> dict` returning `{"version": now_iso, "rows": [Row...]}` where each `Row` is a dict `{id, source_path, label, family, model_id, license, split, sha256, added_utc}`. `label ∈ {"human","ai"}`; `license ∈ {"redistributable","local_only"}` (SCoCESLE → `local_only`); `split ∈ {"cal","test","holdout"}` assigned deterministically by sha (AI: last hex digit even→cal / odd→test; human→cal).
- Produces: `LeakageError`, `LicenseError` (raised by the guards below).
- Produces: `assert_no_leakage(rows) -> None` (raise `LeakageError` if a sha256 appears in >1 split) and `assert_committable(rows) -> None` (raise `LicenseError` if any `local_only` row is present — used before writing a repo-committed artifact).

- [ ] **Step 1: Write the failing test**

```python
# poc/calibration/retune/test_manifest.py
import json
import pytest
from poc.calibration.retune import manifest

def _ai_case(tmp, cid, source, family, authorship="ai", text="word " * 150):
    p = tmp / f"{cid}.json"
    p.write_text(json.dumps({"case_id": cid, "authorship": authorship, "source": source,
                             "family": family, "text": text}))
    return p

def test_ai_rows_labelled_and_family(tmp_path):
    ai = tmp_path / "ai"; ai.mkdir()
    _ai_case(ai, "ai_x_00", "openai/gpt-5-mini", "gpt-5")
    m = manifest.build_manifest(ai, None, now_iso="2026-07-06T00:00:00Z")
    row = m["rows"][0]
    assert row["label"] == "ai"
    assert row["family"] == "gpt-5"
    assert row["model_id"] == "openai/gpt-5-mini"
    assert row["license"] == "redistributable"
    assert row["split"] in ("cal", "test")
    assert len(row["sha256"]) == 64

def test_gutenberg_non_ai_labelled_human(tmp_path):
    ai = tmp_path / "ai"; ai.mkdir()
    _ai_case(ai, "gut_00", "gutenberg:575", "gutenberg", authorship="human")
    m = manifest.build_manifest(ai, None, now_iso="2026-07-06T00:00:00Z")
    assert m["rows"][0]["label"] == "human"  # explicit, not guessed

def test_scocesle_rows_are_local_only(tmp_path):
    ai = tmp_path / "ai"; ai.mkdir()
    esl = tmp_path / "esl" / "higher proficiency "; esl.mkdir(parents=True)
    (esl / "e1.txt").write_text("An essay written by a human ESL student. " * 10)
    m = manifest.build_manifest(ai, tmp_path / "esl", now_iso="2026-07-06T00:00:00Z")
    esl_rows = [r for r in m["rows"] if r["label"] == "human"]
    assert esl_rows and all(r["license"] == "local_only" for r in esl_rows)

def test_leakage_guard(tmp_path):
    rows = [{"sha256": "abc", "split": "cal"}, {"sha256": "abc", "split": "test"}]
    with pytest.raises(manifest.LeakageError):
        manifest.assert_no_leakage(rows)

def test_license_guard_blocks_local_only(tmp_path):
    rows = [{"license": "local_only"}]
    with pytest.raises(manifest.LicenseError):
        manifest.assert_committable(rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/calibration/retune/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: poc.calibration.retune.manifest`

- [ ] **Step 3: Write manifest.py**

```python
# poc/calibration/retune/manifest.py
"""Versioned corpus of record. One row per essay so 'what is in the corpus' is
explicit and diffable — the fix for silent drift and ambiguous label mixing."""
from __future__ import annotations
import glob, hashlib, json
from collections import defaultdict
from pathlib import Path

class LeakageError(Exception): ...
class LicenseError(Exception): ...

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _split_for(label: str, sha: str) -> str:
    if label == "human":
        return "cal"
    return "cal" if int(sha[-1], 16) % 2 == 0 else "test"

def _ai_rows(ai_dir: Path, now_iso: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(ai_dir / "*.json"))):
        d = json.loads(Path(p).read_text())
        text = d.get("text") or ""
        if not text:
            continue
        label = "ai" if (d.get("authorship") or "").lower() == "ai" else "human"
        sha = _sha(text)
        rows.append({
            "id": d.get("case_id") or Path(p).stem,
            "source_path": str(p),
            "label": label,
            "family": d.get("family") or (d.get("source") or "").split("/")[-1] or "unknown",
            "model_id": d.get("source") or "",
            "license": "redistributable",
            "split": _split_for(label, sha),
            "sha256": sha,
            "added_utc": now_iso,
        })
    return rows

def _scocesle_rows(esl_dir: Path, now_iso: str) -> list[dict]:
    rows = []
    for d in Path(esl_dir).glob("*proficiency*"):   # tolerates trailing-space dir name
        if not d.is_dir():
            continue
        for fp in sorted(glob.glob(str(d / "*.txt"))):
            text = Path(fp).read_text(encoding="utf-8", errors="ignore")
            sha = _sha(text)
            rows.append({
                "id": Path(fp).stem,
                "source_path": str(fp),
                "label": "human",
                "family": "scocesle-esl",
                "model_id": "",
                "license": "local_only",   # never committed
                "split": _split_for("human", sha),
                "sha256": sha,
                "added_utc": now_iso,
            })
    return rows

def build_manifest(ai_dir: Path, scocesle_dir: Path | None, now_iso: str) -> dict:
    rows = _ai_rows(ai_dir, now_iso)
    if scocesle_dir is not None:
        rows += _scocesle_rows(scocesle_dir, now_iso)
    assert_no_leakage(rows)
    return {"version": now_iso, "rows": rows}

def assert_no_leakage(rows: list[dict]) -> None:
    splits = defaultdict(set)
    for r in rows:
        splits[r["sha256"]].add(r["split"])
    dupes = {s: v for s, v in splits.items() if len(v) > 1}
    if dupes:
        raise LeakageError(f"{len(dupes)} sha(s) span multiple splits: {list(dupes)[:3]}")

def assert_committable(rows: list[dict]) -> None:
    if any(r.get("license") == "local_only" for r in rows):
        raise LicenseError("manifest contains local_only rows; refuse to write a committed artifact")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/calibration/retune/test_manifest.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/calibration/retune/manifest.py poc/calibration/retune/test_manifest.py
git commit -m "feat(retune): versioned corpus manifest with leakage + license guards"
```

---

### Task 4: Phase-1 CLI — `intake.py` command surface

**Files:**
- Modify: `poc/calibration/retune/intake.py` (add `main()` + argparse at bottom)
- Test: `poc/calibration/retune/test_intake_cli.py`

**Interfaces:**
- Consumes: `generate_ai_essays`, `load_env`, `TOPICS` (Task 2); `load_generators` (Task 1); `build_manifest`, `assert_no_leakage` (Task 3).
- Produces: CLI `python -m poc.calibration.retune.intake [--generate] [--rebuild-manifest] [--scocesle PATH] [--out PATH] [--manifest PATH]`. `--generate` loads env + generators and generates; `--rebuild-manifest` (default if no `--generate`) writes the manifest JSON. Returns process exit 0 on success.
- Produces: `write_manifest_only(ai_dir, scocesle_dir, manifest_path, now_iso) -> int` — builds + writes manifest, returns row count.

- [ ] **Step 1: Write the failing test**

```python
# poc/calibration/retune/test_intake_cli.py
import json
from pathlib import Path
from poc.calibration.retune import intake

def test_write_manifest_only(tmp_path):
    ai = tmp_path / "authorship_cases"; ai.mkdir()
    (ai / "ai_x_00.json").write_text(json.dumps(
        {"case_id": "ai_x_00", "authorship": "ai", "source": "openai/gpt-5-mini",
         "family": "gpt-5", "text": "word " * 150}))
    mpath = tmp_path / "manifest.json"
    n = intake.write_manifest_only(ai, None, mpath, now_iso="2026-07-06T00:00:00Z")
    assert n == 1
    m = json.loads(mpath.read_text())
    assert m["rows"][0]["family"] == "gpt-5"
    assert m["version"] == "2026-07-06T00:00:00Z"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/calibration/retune/test_intake_cli.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'write_manifest_only'`

- [ ] **Step 3: Add CLI + `write_manifest_only` to `intake.py`**

```python
# --- append to poc/calibration/retune/intake.py ---
import argparse
from datetime import datetime, timezone
from .generators import load_generators
from .manifest import build_manifest, assert_no_leakage

DEFAULT_MANIFEST = HERE / "corpus" / "manifest.json"
DEFAULT_SCOCESLE = Path.home() / "Downloads" / "Small Corpus of Colombian English as a Second Language Essays (SCoCESLE)"

def write_manifest_only(ai_dir: Path, scocesle_dir: Path | None, manifest_path: Path, now_iso: str) -> int:
    m = build_manifest(ai_dir, scocesle_dir, now_iso)
    assert_no_leakage(m["rows"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(m, indent=2, ensure_ascii=False))
    print(f"wrote manifest: {len(m['rows'])} rows -> {manifest_path}")
    return len(m["rows"])

def main() -> int:
    ap = argparse.ArgumentParser(description="V7 re-tune Phase 1: intake + manifest")
    ap.add_argument("--generate", action="store_true", help="generate AI essays from models.json")
    ap.add_argument("--rebuild-manifest", action="store_true", help="(re)build the corpus manifest")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scocesle", type=Path, default=DEFAULT_SCOCESLE)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args()
    now_iso = datetime.now(timezone.utc).isoformat()
    if args.generate:
        load_env()
        generate_ai_essays(args.out, load_generators(), TOPICS)
    scocesle = args.scocesle if args.scocesle.exists() else None
    write_manifest_only(args.out, scocesle, args.manifest, now_iso)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/calibration/retune/test_intake_cli.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Smoke the manifest against the REAL corpus (no generation)**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m poc.calibration.retune.intake --rebuild-manifest`
Expected: prints `wrote manifest: N rows` where N ≈ 80 AI/Gutenberg + 272 SCoCESLE. Confirm `poc/calibration/retune/corpus/manifest.json` exists and gpt-5 family rows appear once generated.

- [ ] **Step 6: Commit**

```bash
git add poc/calibration/retune/intake.py poc/calibration/retune/test_intake_cli.py
git commit -m "feat(retune): Phase-1 CLI — --generate and --rebuild-manifest"
```

---

### Task 5: Phase-2 gate oracle wrapper

**Files:**
- Create: `poc/calibration/retune/gate.py`
- Test: `poc/calibration/retune/test_gate.py`

**Interfaces:**
- Consumes: nothing from prior tasks; shells out to the existing `poc/calibration/fpr_subgroup_gate.py`.
- Produces: `run_fpr_gate(corpus: Path | None = None, baseline: Path | None = None, limit: int | None = None, runner=subprocess.run) -> GateResult` where `GateResult` is a dataclass `(passed: bool, exit_code: int, corpus_available: bool, stdout: str)`. Maps the gate's exit contract: `0 -> passed=True`; `1 -> passed=False` (regression); `2 -> corpus_available=False, passed=False` (setup/corpus missing — not a regression). `runner` is injectable for tests.

- [ ] **Step 1: Write the failing test**

```python
# poc/calibration/retune/test_gate.py
from types import SimpleNamespace
from poc.calibration.retune import gate

def _fake_runner(code, out=""):
    def run(cmd, **kw):
        return SimpleNamespace(returncode=code, stdout=out, stderr="")
    return run

def test_pass():
    r = gate.run_fpr_gate(runner=_fake_runner(0, "AUC 0.75"))
    assert r.passed and r.exit_code == 0 and r.corpus_available

def test_regression():
    r = gate.run_fpr_gate(runner=_fake_runner(1))
    assert not r.passed and r.exit_code == 1 and r.corpus_available

def test_corpus_missing_is_not_regression():
    r = gate.run_fpr_gate(runner=_fake_runner(2))
    assert not r.passed and not r.corpus_available and r.exit_code == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/calibration/retune/test_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: poc.calibration.retune.gate`

- [ ] **Step 3: Write gate.py**

```python
# poc/calibration/retune/gate.py
"""Thin wrapper over fpr_subgroup_gate.py — the single acceptance oracle. It owns the
exit-code contract (0 pass / 1 regression / 2 corpus-missing) so callers read a verdict,
not a return code."""
from __future__ import annotations
import subprocess, sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE_SCRIPT = HERE.parent / "fpr_subgroup_gate.py"

@dataclass(frozen=True)
class GateResult:
    passed: bool
    exit_code: int
    corpus_available: bool
    stdout: str

def run_fpr_gate(corpus: Path | None = None, baseline: Path | None = None,
                 limit: int | None = None, runner=subprocess.run) -> GateResult:
    cmd = [sys.executable, str(GATE_SCRIPT), "--compare"]
    if baseline is not None:
        cmd.append(str(baseline))
    if corpus is not None:
        cmd += ["--corpus", str(corpus)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    proc = runner(cmd, capture_output=True, text=True)
    code = proc.returncode
    return GateResult(
        passed=(code == 0),
        exit_code=code,
        corpus_available=(code != 2),
        stdout=(proc.stdout or ""),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/calibration/retune/test_gate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/calibration/retune/gate.py poc/calibration/retune/test_gate.py
git commit -m "feat(retune): FPR-gate oracle wrapper with exit-code contract"
```

---

### Task 6: Orchestrator — `run_cycle.py` + `RETUNE_LOG.md`

**Files:**
- Create: `poc/calibration/retune/run_cycle.py`
- Test: `poc/calibration/retune/test_run_cycle.py`

**Interfaces:**
- Consumes: `write_manifest_only` (Task 4); `run_fpr_gate`, `GateResult` (Task 5).
- Produces: `append_log(log_path: Path, entry: dict) -> None` — appends a Markdown table row `| version | n_rows | families | gate | auc_line |` to `RETUNE_LOG.md`, creating the header if absent.
- Produces: `run_cycle(ai_dir, scocesle_dir, manifest_path, log_path, now_iso, generate: bool, gate_fn=run_fpr_gate, generate_fn=None) -> GateResult` — runs Phase 1 (optionally generate) → writes manifest → runs the gate → appends a log row → returns the `GateResult`. `generate_fn`/`gate_fn` injectable for tests.
- Produces: CLI `python -m poc.calibration.retune.run_cycle [--generate]`.

- [ ] **Step 1: Write the failing test**

```python
# poc/calibration/retune/test_run_cycle.py
import json
from pathlib import Path
from poc.calibration.retune import run_cycle
from poc.calibration.retune.gate import GateResult

def _seed_ai(tmp):
    ai = tmp / "authorship_cases"; ai.mkdir()
    (ai / "ai_x_00.json").write_text(json.dumps(
        {"case_id": "ai_x_00", "authorship": "ai", "source": "openai/gpt-5-mini",
         "family": "gpt-5", "text": "word " * 150}))
    return ai

def test_append_log_creates_header(tmp_path):
    log = tmp_path / "RETUNE_LOG.md"
    run_cycle.append_log(log, {"version": "v1", "n_rows": 5, "families": "gpt-5",
                               "gate": "PASS", "auc_line": "AUC 0.75"})
    txt = log.read_text()
    assert "| version |" in txt and "| v1 |" in txt and "PASS" in txt

def test_run_cycle_passes_and_logs(tmp_path):
    ai = _seed_ai(tmp_path)
    manifest = tmp_path / "manifest.json"
    log = tmp_path / "RETUNE_LOG.md"
    fake_gate = lambda **kw: GateResult(passed=True, exit_code=0, corpus_available=True, stdout="AUC 0.75")
    res = run_cycle.run_cycle(ai, None, manifest, log, now_iso="2026-07-06T00:00:00Z",
                              generate=False, gate_fn=fake_gate)
    assert res.passed
    assert manifest.exists()
    assert "PASS" in log.read_text()

def test_run_cycle_records_fail(tmp_path):
    ai = _seed_ai(tmp_path)
    fake_gate = lambda **kw: GateResult(passed=False, exit_code=1, corpus_available=True, stdout="")
    res = run_cycle.run_cycle(ai, None, tmp_path / "m.json", tmp_path / "L.md",
                              now_iso="2026-07-06T00:00:00Z", generate=False, gate_fn=fake_gate)
    assert not res.passed
    assert "FAIL" in (tmp_path / "L.md").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/calibration/retune/test_run_cycle.py -v`
Expected: FAIL — `ModuleNotFoundError: poc.calibration.retune.run_cycle`

- [ ] **Step 3: Write run_cycle.py**

```python
# poc/calibration/retune/run_cycle.py
"""Phase 1 -> Phase 2 orchestrator. Builds the manifest, runs the FPR-gate oracle, and
appends a RETUNE_LOG.md decision row. Produces CANDIDATE artifacts only — promotion to
production is a separate, human-approved step (see docs/runbooks/v7-retune.md)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from .intake import write_manifest_only, load_env, generate_ai_essays, TOPICS, DEFAULT_OUT, DEFAULT_SCOCESLE, DEFAULT_MANIFEST
from .generators import load_generators
from .gate import run_fpr_gate, GateResult

HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "RETUNE_LOG.md"
_HEADER = "| version | n_rows | families | gate | auc_line |\n|---|---|---|---|---|\n"

def append_log(log_path: Path, entry: dict) -> None:
    if not log_path.exists():
        log_path.write_text("# V7 Re-Tune Decision Log\n\n" + _HEADER)
    with log_path.open("a") as f:
        f.write(f"| {entry['version']} | {entry['n_rows']} | {entry['families']} "
                f"| {entry['gate']} | {entry['auc_line']} |\n")

def _families(manifest_rows) -> str:
    return ",".join(sorted({r["family"] for r in manifest_rows if r["label"] == "ai"}))

def run_cycle(ai_dir: Path, scocesle_dir: Path | None, manifest_path: Path, log_path: Path,
              now_iso: str, generate: bool, gate_fn=run_fpr_gate, generate_fn=None) -> GateResult:
    if generate:
        load_env()
        (generate_fn or generate_ai_essays)(ai_dir, load_generators(), TOPICS)
    n_rows = write_manifest_only(ai_dir, scocesle_dir, manifest_path, now_iso)
    import json
    rows = json.loads(manifest_path.read_text())["rows"]
    result = gate_fn(corpus=scocesle_dir)
    verdict = "PASS" if result.passed else ("NO-CORPUS" if not result.corpus_available else "FAIL")
    auc_line = next((ln.strip() for ln in result.stdout.splitlines() if "AUC" in ln), "")
    append_log(log_path, {"version": now_iso, "n_rows": n_rows, "families": _families(rows),
                          "gate": verdict, "auc_line": auc_line})
    print(f"\n=== V7 RE-TUNE: {verdict} ===")
    return result

def main() -> int:
    ap = argparse.ArgumentParser(description="V7 re-tune cycle: Phase 1 intake -> Phase 2 gate")
    ap.add_argument("--generate", action="store_true", help="generate fresh AI essays first")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scocesle", type=Path, default=DEFAULT_SCOCESLE)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = ap.parse_args()
    now_iso = datetime.now(timezone.utc).isoformat()
    scocesle = args.scocesle if args.scocesle.exists() else None
    res = run_cycle(args.out, scocesle, args.manifest, args.log, now_iso, args.generate)
    return 0 if (res.passed or not res.corpus_available) else 1

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/calibration/retune/test_run_cycle.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the FULL retune package test suite**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/calibration/retune/ -v`
Expected: PASS (all tasks' tests green — ~19 tests)

- [ ] **Step 6: Commit**

```bash
git add poc/calibration/retune/run_cycle.py poc/calibration/retune/test_run_cycle.py
git commit -m "feat(retune): Phase1->Phase2 orchestrator + RETUNE_LOG decision log"
```

---

### Task 7: Runbook

**Files:**
- Create: `docs/runbooks/v7-retune.md`

**Interfaces:** none (documentation).

- [ ] **Step 1: Write the runbook**

```markdown
# Runbook — V7 Re-Tune (manual, on-demand)

**When:** a new AI model ships (gpt-5.6, gpt-6…) or enough new essays accumulate.

## Add a new model
1. Add one row to `poc/calibration/retune/models.json` (`id`, `provider`, `family`, `n_per_topic`).
   Validate the id against OpenRouter's live `/models` list first (guessed ids 404).

## Run the cycle
2. `cd poc && ~/.pyenv/versions/3.11.0/bin/python3 -m poc.calibration.retune.run_cycle --generate`
   - Generates + PERSISTS AI essays, rebuilds the manifest, runs the FPR gate.
   - `.env` must have `OPENROUTER_API_KEY` (lives in the MAIN repo root, not the worktree).
3. Read the verdict line + the appended row in `poc/calibration/retune/RETUNE_LOG.md`:
   - **PASS** — the new corpus does not regress ESL false-accusation (FPR rise ≤3pts, AUC
     drop ≤0.05, parity widen ≤4pts vs the committed baseline). Candidate artifacts are safe
     to promote.
   - **FAIL** — a tolerance broke. Do NOT promote. Inspect the gate stdout for which one.
   - **NO-CORPUS** — SCoCESLE dir not found; set `--scocesle PATH`.

## Re-calibrate (Phase 2 detail)
4. If discrimination dropped (new model evades), re-sweep with the existing scripts writing to
   a staging dir, then re-run the gate:
   - `python calibration/v7_deberta_academic_calibrate.py` (deep-scan thresholds)
   - `python calibration/deberta_fit_calibrator.py` (isotonic re-fit)
   - `python calibration/v7_fused_gate_run.py` (fused TPR/AUC/parity)
   - `python calibration/fpr_subgroup_gate.py --compare` (the oracle — must pass)

## Promote (only on PASS)
5. Re-baseline intentionally: `python calibration/fpr_subgroup_gate.py --out calibration/fpr_subgroup_baseline.json`,
   commit the numbers-only baseline (never the SCoCESLE text — the manifest's `local_only`
   license guard blocks that by construction).

## Phase 3 (GPU checkpoint fine-tune) — NOT built yet
Escalation only, when re-calibration can't recover discrimination. Specced in
`docs/superpowers/specs/2026-07-06-v7-retune-workflow-design.md` §7 (Modal, same gate oracle).
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/v7-retune.md
git commit -m "docs(retune): runbook for the manual V7 re-tune cycle"
```

---

## Self-Review

**Spec coverage:**
- §5.1 models.json → Task 1 ✅
- §5.2 manifest → Task 3 ✅
- §5.3 intake CLI (--generate / --rebuild-manifest, leakage + license guards, provenance) → Tasks 2, 3, 4 ✅
- §6 re-calibration + gate oracle → Task 5 (gate wrapper) + Task 7 runbook step 4 (existing scripts sequenced; not re-implemented — they already exist) ✅
- §8 orchestrator + RETUNE_LOG + runbook → Tasks 6, 7 ✅
- §7 Phase 3 → intentionally specced-only (Global scope decision) ✅
- §10 testing (manifest units, mocked intake, gate-as-oracle, log) → Tasks 3,2,5,6 ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code.

**Type consistency:** `Generator(id, provider, family, n_per_topic)` used identically across Tasks 1/2/4/6. `GateResult(passed, exit_code, corpus_available, stdout)` consistent Tasks 5/6. `write_manifest_only(ai_dir, scocesle_dir, manifest_path, now_iso)` signature matches its call in Task 6. Manifest row keys (`family`, `label`, `sha256`, `split`) consistent Tasks 3/4/6.

**Note on Phase 2 depth:** This plan wires the gate *oracle* (Task 5) and sequences the existing calibration scripts via the runbook rather than re-implementing them — they already work and re-running them is a heavy (~13 min + Modal) integration step, not unit-testable logic. If you want the re-sweep steps wrapped as their own Python orchestration (mocked-subprocess tested) rather than runbook commands, that's a follow-up task to add.
