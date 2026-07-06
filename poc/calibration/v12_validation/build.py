"""Build the §12 validation corpus: sample SCoCESLE humans, reuse the existing
AI cases, and GENERATE the two missing classes (ai_assisted_polished,
ai_paraphrased) as LLM variants of the sampled human essays.

Reuses the retune intake plumbing (same OpenRouter chat call, same .env
loading, same persist-one-JSON-per-case pattern) rather than reinventing it.

Everything written to CORPUS_DIR is SCoCESLE-derived or SCoCESLE text and is
gitignored (no-redistribution license). The manifest carries sha256 + label +
provenance per row so the set is versioned without committing any text.

Usage:
    python -m calibration.v12_validation.build              # sample + generate
    python -m calibration.v12_validation.build --n-human 20 # per proficiency group
    python -m calibration.v12_validation.build --dry-run    # no API calls
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from calibration.retune.generators import PROVIDER_BASE_URLS, load_generators
from calibration.retune.intake import DEFAULT_SCOCESLE, _chat, load_env

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"          # gitignored — SCoCESLE-derived text lives here
AI_CASES_DIR = HERE.parent / "authorship_cases"

# Deterministic sampling: sorted filenames, evenly-spaced stride — no RNG, so
# the same corpus + same N always selects the same essays (reproducible set).
def _sample_evenly(paths: list[str], n: int) -> list[str]:
    paths = sorted(paths)
    if len(paths) <= n:
        return paths
    stride = len(paths) / n
    return [paths[int(i * stride)] for i in range(n)]


def _scocesle_groups(corpus: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"higher": [], "lower": []}
    for d in corpus.glob("*proficiency*"):
        key = "higher" if "higher" in d.name.lower() else "lower"
        groups[key] += glob.glob(str(d / "*.txt"))
    return groups


# allow-hardcode: LLM generation prompts (variant construction instructions),
# not a detection/scoring list. The polish prompt intentionally forbids adding
# content (that is what distinguishes the ai_assisted_polished class from
# ai_generated_like); the paraphrase prompt intentionally requires a full
# rewording that preserves meaning (spec D9's comparison-dependent class).
POLISH_PROMPT = (
    "Lightly polish the following student essay: fix grammar, spelling, and "
    "awkward phrasing, and smooth the flow. KEEP the writer's ideas, examples, "
    "structure, and paragraph breaks exactly — do NOT add new content, new "
    "examples, or new arguments, and do not change the essay's length by more "
    "than ~10%. Return only the polished essay text.\n\nESSAY:\n{essay}"
)
PARAPHRASE_PROMPT = (
    "Rewrite the following student essay fully in your own words, preserving "
    "its meaning, argument order, and approximate length. Change the wording "
    "and sentence structures throughout (a thorough paraphrase, not a light "
    "edit). Do not add new ideas or examples. Keep the same paragraph breaks. "
    "Return only the rewritten essay text.\n\nESSAY:\n{essay}"
)
_VARIANTS = {"ai_assisted_polished": POLISH_PROMPT, "ai_paraphrased": PARAPHRASE_PROMPT}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_variants(human_files: list[str], out_dir: Path, generators, chat_fn=_chat) -> int:
    """One polished + one paraphrased variant per sampled human essay, cycling
    through the models.json generators for family diversity. Skips existing
    files (resumable, like retune intake)."""
    import os
    out_dir.mkdir(parents=True, exist_ok=True)
    usable = []
    for gen in generators:
        base_url, key_env = PROVIDER_BASE_URLS[gen.provider]
        if os.environ.get(key_env, "").strip():
            usable.append((gen, base_url, os.environ[key_env].strip()))
    if not usable:
        print("no generator API keys available — set OPENROUTER_API_KEY")
        return 0
    made = 0
    for i, fp in enumerate(human_files):
        essay = Path(fp).read_text(encoding="utf-8", errors="ignore").strip()
        if len(essay.split()) < 120:
            continue
        stem = re.sub(r"[^a-z0-9]+", "_", Path(fp).stem.lower())[:40]
        for label, prompt_tmpl in _VARIANTS.items():
            start = i + (0 if label.endswith("polished") else 1)
            # Relative floor: a faithful variant of a short essay is legitimately
            # short — an absolute 120-word gate permanently rejects variants of
            # ~130-word sources. 60% of the source (min 80) catches refusals and
            # truncations without starving the class.
            min_words = max(80, int(0.6 * len(essay.split())))
            # Resumable: a case for this (label, essay) may exist under ANY
            # generator's name (the fallback rotation below means the model
            # isn't fixed per essay).
            def _cid(g):
                short = re.sub(r"[^a-z0-9]+", "_", g.id.split("/")[-1].lower())
                return f"{label}_{short}_{stem}"
            if any((out_dir / f"{_cid(g)}.json").exists() for g, _, _ in usable):
                continue
            # Fallback rotation: some models persistently truncate on some
            # inputs (observed: gpt-5-mini spends reasoning tokens inside
            # max_tokens on polish prompts) — try each usable generator once.
            for attempt in range(len(usable)):
                gen, base_url, api_key = usable[(start + attempt) % len(usable)]
                cid = _cid(gen)
                # max_tokens 1100: longest SCoCESLE essay is 586 words (~800
                # tokens out); the retune default of 700 would truncate.
                text = chat_fn(gen.id, base_url, api_key, prompt_tmpl.format(essay=essay), 0.4,
                               max_tokens=1100)
                if not text or len(text.split()) < min_words:
                    print(f"   - {cid}: too short / empty (<{min_words}w), trying next generator")
                    continue
                (out_dir / f"{cid}.json").write_text(json.dumps({
                    "case_id": cid, "label": label, "source_model": gen.id,
                    "family": gen.family, "source_human_file": Path(fp).name,
                    "source_sha256": _sha(essay), "words": len(text.split()),
                    "license": "scocesle_derivative_local_only", "text": text,
                }, indent=2, ensure_ascii=False))
                made += 1
                print(f"   + {cid} ({len(text.split())} words)")
                break
    print(f"wrote {made} variant cases to {out_dir}")
    return made


def build_manifest(human_files: list[str], out_dir: Path, manifest_path: Path, now_iso: str) -> int:
    """Numbers/hashes-only row list for every doc in the set. The manifest
    itself stays inside the gitignored corpus dir (it references local-only
    filenames), mirroring the retune corpus manifest license guard."""
    rows = []
    for fp in human_files:
        text = Path(fp).read_text(encoding="utf-8", errors="ignore")
        rows.append({"label": "student_owned", "file": Path(fp).name,
                     "sha256": _sha(text), "words": len(text.split()),
                     "license": "scocesle_local_only"})
    for fp in sorted(glob.glob(str(AI_CASES_DIR / "*.json"))):
        d = json.loads(Path(fp).read_text())
        if d.get("text"):
            rows.append({"label": "ai_generated_like", "file": Path(fp).name,
                         "sha256": _sha(d["text"]), "words": len(d["text"].split()),
                         "license": "generated_committable"})
    for fp in sorted(glob.glob(str(out_dir / "*.json"))):
        d = json.loads(Path(fp).read_text())
        if "label" not in d:  # the manifest itself, or any non-case JSON
            continue
        rows.append({"label": d["label"], "file": Path(fp).name,
                     "sha256": _sha(d["text"]), "words": d["words"],
                     "license": d["license"]})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(
        {"version": now_iso, "spec": "v7 §12 / §21 seeding (construction labels)",
         "rows": rows}, indent=2, ensure_ascii=False))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"manifest: {len(rows)} rows -> {manifest_path}\n  by class: {counts}")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the V7 §12 validation corpus")
    ap.add_argument("--scocesle", type=Path, default=DEFAULT_SCOCESLE)
    ap.add_argument("--n-human", type=int, default=20, help="essays per proficiency group")
    ap.add_argument("--dry-run", action="store_true", help="sample + manifest only, no API calls")
    args = ap.parse_args()
    if not args.scocesle.exists():
        print(f"SCoCESLE corpus not found at {args.scocesle}")
        return 2
    groups = _scocesle_groups(args.scocesle)
    human_files = (_sample_evenly(groups["higher"], args.n_human)
                   + _sample_evenly(groups["lower"], args.n_human))
    print(f"sampled {len(human_files)} human essays "
          f"({len(groups['higher'])} higher / {len(groups['lower'])} lower available)")
    if not args.dry_run:
        load_env()
        generate_variants(human_files, CORPUS_DIR, load_generators())
    build_manifest(human_files, CORPUS_DIR, CORPUS_DIR / "manifest.json",
                   datetime.now(timezone.utc).isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
