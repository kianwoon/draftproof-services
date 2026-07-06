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
