"""Persistent content-hash cache for the paid Modal deep-scan.

Caches the RAW per-sentence scores (the expensive Modal output), keyed by
sha256(checkpoint + "\\n" + text). The proportion at any sent_threshold is derived
LOCALLY and is free to recompute — this makes the cache threshold-INDEPENDENT: a
threshold change does NOT invalidate it. Only a checkpoint change (different key)
invalidates a row.

Deliberately lives OUTSIDE any per-run staging dir — persistence across runs is the
whole point (a later run/session must never re-pay Modal for an essay already scored
under the same checkpoint). No essay text is stored, only the hash key and scores.

Pure stdlib. No network calls anywhere in this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = _HERE / "cache" / "deepscan_scores.jsonl"

DEFAULT_CHECKPOINT = "desklib/ai-text-detector-academic-v1.01"


def checkpoint_tag() -> str:
    import os
    from . import intake
    intake.load_env()
    return os.environ.get("DRAFTPROOF_MODAL_CHECKPOINT", DEFAULT_CHECKPOINT)


def content_key(text: str, checkpoint: str) -> str:
    return hashlib.sha256(f"{checkpoint}\n{text}".encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, list[float]]:
    path = Path(path)
    cache: dict[str, list[float]] = {}
    if not path.exists():
        return cache
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Torn/partial line (e.g. crash mid-append). Skip it — the affected
            # essay simply re-scores on the next run; safe to drop.
            continue
        cache[row["key"]] = row["scores"]
    return cache


def append(path: Path, key: str, scores: list[float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"key": key, "scores": scores}) + "\n")


def proportion(scores: list[float] | None, sent_thr: float) -> float | None:
    if not scores:
        return None
    return sum(1 for x in scores if x >= sent_thr) / len(scores)


def get_scores(text: str, checkpoint: str, cache: dict[str, list[float]], path: Path,
                score_fn) -> list[float] | None:
    key = content_key(text, checkpoint)
    if key in cache:
        return cache[key]
    scores = score_fn(text)
    if scores:
        append(path, key, scores)
        cache[key] = scores
    return scores
