#!/usr/bin/env python3
"""Build a STRATIFIED bounded subset of the RAID benchmark (Dugan et al., ACL 2024).

RAID is public (NOT gated) on HuggingFace as ``liamdugan/raid``. The LABELED data
lives in ``train.csv`` (~11 GB) and ``extra.csv`` (~3.5 GB); ``test.csv`` is the
held-out leaderboard split and carries only ``id,generation`` (no labels), so it is
useless here.

Why not ``datasets`` streaming? ``load_dataset(..., streaming=True)`` reads the file
in its STORED order, which is grouped by ``(domain, model, decoding, attack)``. All
of the first millions of rows are ``domain=abstracts, attack=none`` — a linear
early-stop can never reach the ``paraphrase`` attack or the other domains without
reading (almost) the whole 11 GB. So we DON'T stream.

Approach: RANDOM-BYTE-SAMPLE the CSV over HTTP Range requests (the HF CDN returns
``206 Partial Content``; total size is read from the ``Content-Range`` header, never
hardcoded). CSV record boundaries are anchored on the leading UUID ``id`` column
(``\\n<uuid>,``), so a mid-file byte fragment parses cleanly with the ``csv`` module
(quoted ``generation`` fields with embedded newlines are handled correctly because we
only parse COMPLETE records between two UUID anchors).

  * AI rows (``attack==none`` and ``attack==paraphrase``) are abundant at random
    offsets -> sampled directly, then stratified across models/domains.
  * Human ``attack==none`` rows cluster at the START of each domain block, so they are
    located with a small BINARY SEARCH on the alphabetically-sorted ``domain`` column
    and harvested there.

Total download for a ~600-row subset is a few tens of MB. No Modal, no spend.

Output ``subset.jsonl`` — one JSON object per line:
    {"text": ..., "label": 0|1, "attack": ..., "model": ..., "domain": ...}
    label = 0 when model=="human", else 1.

Fallback: if the HF id is ever gated/unavailable, the identical ``train.csv`` is
mirrored on the RAID GitHub release (https://github.com/liamdugan/raid). Pass its
direct URL via ``--url`` — the Range-sampling logic is unchanged.

Usage:
    python calibration/raid_benchmark/fetch_subset.py                 # ~200/class
    python calibration/raid_benchmark/fetch_subset.py --per-class 50  # smaller
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "subset.jsonl"

_DATASET_ID = "liamdugan/raid"
_DEFAULT_FILE = "train.csv"  # the large, fully-labeled split
_UUID_ANCHOR = re.compile(r"\n([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                          r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}),")
_WINDOW = 262_144          # bytes per Range request (~190 CSV rows)
_ATTACK_NONE = "none"
_ATTACK_PARAPHRASE = "paraphrase"
_HUMAN = "human"


# --------------------------------------------------------------------------- #
# HTTP Range plumbing
# --------------------------------------------------------------------------- #
def _resolve_url(explicit: str | None, hf_file: str) -> str:
    if explicit:
        return explicit
    from huggingface_hub import hf_hub_url
    return hf_hub_url(_DATASET_ID, hf_file, repo_type="dataset")


def _http_range(url: str, off: int, n: int, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + n - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - transient CDN hiccups; retry
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Range GET failed off={off} n={n}: {last}")


def _total_size(url: str) -> int:
    """Read the object size from the Content-Range of a 1-byte Range request."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        cr = r.headers.get("Content-Range")  # "bytes 0-0/<total>"
        if cr and "/" in cr:
            return int(cr.rsplit("/", 1)[1])
        cl = r.headers.get("Content-Length")
        if cl:
            return int(cl)
    raise RuntimeError("could not determine object size (no Content-Range/Length)")


def _header_index(url: str) -> dict[str, int]:
    """Parse the CSV header row into {column_name: index} (no hardcoded positions)."""
    head = _http_range(url, 0, 4096).decode("utf-8", "ignore")
    line = head.split("\n", 1)[0]
    cols = next(csv.reader(io.StringIO(line)))
    return {c.strip(): i for i, c in enumerate(cols)}


# --------------------------------------------------------------------------- #
# Windowed CSV parsing (robust to mid-file byte offsets)
# --------------------------------------------------------------------------- #
def _rows_in_window(url: str, off: int, ncols: int) -> list[list[str]]:
    """Return only COMPLETE, well-formed records found between UUID anchors."""
    buf = _http_range(url, off, _WINDOW).decode("utf-8", "ignore")
    anchors = list(_UUID_ANCHOR.finditer(buf))
    if len(anchors) < 2:
        return []
    frag = buf[anchors[0].start() + 1: anchors[-1].start()]
    out = []
    for rec in csv.reader(io.StringIO(frag)):
        if len(rec) == ncols:
            out.append(rec)
    return out


def _row_dict(rec: list[str], idx: dict[str, int]) -> dict:
    return {
        "id": rec[idx["id"]],
        "model": rec[idx["model"]],
        "attack": rec[idx["attack"]],
        "domain": rec[idx["domain"]],
        "text": rec[idx["generation"]],
    }


# --------------------------------------------------------------------------- #
# Human harvesting: binary-search each domain's leading none-block
# --------------------------------------------------------------------------- #
def _first_domain_at(url: str, off: int, idx: dict[str, int], ncols: int) -> str | None:
    rows = _rows_in_window(url, off, ncols)
    return rows[0][idx["domain"]] if rows else None


def _find_domain_boundary(url: str, target: str, size: int,
                          idx: dict[str, int], ncols: int, max_steps: int = 22) -> int:
    """Binary-search the byte offset where the sorted `domain` column reaches `target`.

    The file is sorted by domain ascending, so plain string comparison suffices;
    convergence lands in `target`'s leading rows, i.e. its human attack==none block."""
    lo, hi, steps = 0, size, 0
    while hi - lo > _WINDOW and steps < max_steps:
        mid = (lo + hi) // 2
        steps += 1
        d = _first_domain_at(url, mid, idx, ncols)
        if d is None:
            lo = min(mid + _WINDOW, hi)  # empty probe: nudge forward
            continue
        if d < target:
            lo = mid
        else:
            hi = mid
    return lo


def _harvest_humans(url: str, size: int, idx: dict[str, int], ncols: int,
                    domains: list[str], quota_per_domain: int, pool_cap: int,
                    seen_ids: set[str], log) -> list[dict]:
    """Binary-search each domain's leading block and collect human attack==none rows."""
    humans: list[dict] = []
    for dom in domains:
        if len(humans) >= pool_cap:
            break
        b = _find_domain_boundary(url, dom, size, idx, ncols)
        got, lead_empty = 0, 0
        off = b  # boundary lands just before this domain's leading (human none) rows
        # Scan forward: tolerate a few leading windows before the human block, then
        # collect it, then stop once the block ends (empty AFTER we've collected).
        for _ in range(12):
            rows = _rows_in_window(url, off, ncols)
            off += _WINDOW
            hn = [r for r in rows
                  if r[idx["model"]] == _HUMAN and r[idx["attack"]] == _ATTACK_NONE]
            if not hn:
                if got > 0:
                    break            # collected the block, now past it
                lead_empty += 1
                if lead_empty >= 4:  # human block not near this boundary; give up
                    break
                continue
            for r in hn:
                if got >= quota_per_domain:
                    break
                d = _row_dict(r, idx)
                if d["id"] in seen_ids:
                    continue
                seen_ids.add(d["id"])
                humans.append(d)
                got += 1
            if got >= quota_per_domain:
                break
        log(f"  human[{dom}]: +{got} (pool={len(humans)})")
    return humans


# --------------------------------------------------------------------------- #
# AI sampling: random offsets, stratified later
# --------------------------------------------------------------------------- #
def _sample_ai(url: str, size: int, idx: dict[str, int], ncols: int,
               rng: random.Random, pool_cap: int, max_windows: int, domain_soft_cap: int,
               seen_ids: set[str], domain_sink: set[str], log) -> dict[str, list[dict]]:
    # Paraphrase/none rows arrive in big contiguous same-(model,domain) runs, so an
    # unconstrained pool ends up dominated by whichever few blocks were hit first.
    # A per-(attack, domain) soft cap forces the sampler to keep going and pick up
    # other domains (random offsets do reach all domains), giving real spread.
    pools = {_ATTACK_NONE: [], _ATTACK_PARAPHRASE: []}
    per_dom: dict[tuple, int] = defaultdict(int)
    windows = 0
    while windows < max_windows:
        if all(len(pools[a]) >= pool_cap for a in pools):
            break
        off = rng.randint(1000, max(1001, size - _WINDOW - 1000))
        windows += 1
        for rec in _rows_in_window(url, off, ncols):
            d = _row_dict(rec, idx)
            domain_sink.add(d["domain"])
            if d["model"] == _HUMAN:
                continue
            a = d["attack"]
            if a not in pools or len(pools[a]) >= pool_cap:
                continue
            if per_dom[(a, d["domain"])] >= domain_soft_cap:
                continue
            if d["id"] in seen_ids:
                continue
            seen_ids.add(d["id"])
            pools[a].append(d)
            per_dom[(a, d["domain"])] += 1
        if windows % 20 == 0:
            log(f"  ai sampling: {windows} windows | none={len(pools[_ATTACK_NONE])} "
                f"para={len(pools[_ATTACK_PARAPHRASE])}")
    log(f"  ai sampling done: {windows} windows | none={len(pools[_ATTACK_NONE])} "
        f"para={len(pools[_ATTACK_PARAPHRASE])}")
    return pools


# --------------------------------------------------------------------------- #
# Stratified down-selection (spread across model + domain)
# --------------------------------------------------------------------------- #
def _stratified_pick(cands: list[dict], k: int, rng: random.Random) -> list[dict]:
    if len(cands) <= k:
        return list(cands)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for c in cands:
        buckets[(c["model"], c["domain"])].append(c)
    for b in buckets.values():
        rng.shuffle(b)
    keys = list(buckets.keys())
    rng.shuffle(keys)
    picked: list[dict] = []
    while len(picked) < k and any(buckets[key] for key in keys):
        for key in keys:
            if buckets[key]:
                picked.append(buckets[key].pop())
                if len(picked) >= k:
                    break
    return picked


def _cap_words(text: str, max_words: int) -> str:
    return " ".join((text or "").split()[:max_words])


# --------------------------------------------------------------------------- #
def build(per_class: int, seed: int, out: Path, url: str, max_words: int,
          oversample: float, max_ai_windows: int) -> dict:
    rng = random.Random(seed)
    log = lambda m: print(m, flush=True)  # noqa: E731

    log(f"[fetch] resolving {url.split('/')[-1]} ...")
    size = _total_size(url)
    idx = _header_index(url)
    ncols = max(idx.values()) + 1
    log(f"[fetch] object size={size / 1e9:.1f} GB  columns={list(idx)}")

    pool_cap = max(per_class, int(per_class * oversample))
    seen: set[str] = set()
    domains_seen: set[str] = set()

    # 1) AI strata via random byte offsets
    log("[fetch] sampling AI rows (attack=none / paraphrase) at random offsets ...")
    # spread each attack pool over domains: aim ~6 domains, floor keeps small runs
    domain_soft_cap = max(8, -(-pool_cap // 6))  # ceil(pool_cap / 6)
    ai_pools = _sample_ai(url, size, idx, ncols, rng, pool_cap, max_ai_windows,
                          domain_soft_cap, seen, domains_seen, log)

    # 2) Human attack==none via binary-searched domain blocks
    targets = sorted(domains_seen) or [None]
    quota = max(1, -(-per_class // max(1, len(targets))))  # ceil
    log(f"[fetch] harvesting human rows across domains={targets} (quota/dom={quota}) ...")
    humans = _harvest_humans(url, size, idx, ncols, [t for t in targets if t],
                             quota, pool_cap, seen, log)

    # 3) stratified down-select to per_class each
    sel_human = _stratified_pick(humans, per_class, rng)
    sel_none = _stratified_pick(ai_pools[_ATTACK_NONE], per_class, rng)
    sel_para = _stratified_pick(ai_pools[_ATTACK_PARAPHRASE], per_class, rng)

    rows: list[dict] = []
    for d in sel_human:
        rows.append({"text": _cap_words(d["text"], max_words), "label": 0,
                     "attack": d["attack"], "model": d["model"], "domain": d["domain"]})
    for d in sel_none + sel_para:
        rows.append({"text": _cap_words(d["text"], max_words), "label": 1,
                     "attack": d["attack"], "model": d["model"], "domain": d["domain"]})
    rng.shuffle(rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary
    def _counts(sel):
        m, dm = defaultdict(int), defaultdict(int)
        for x in sel:
            m[x["model"]] += 1
            dm[x["domain"]] += 1
        return dict(sorted(m.items())), dict(sorted(dm.items()))

    summary = {
        "out": str(out),
        "per_class_target": per_class,
        "counts": {
            "human": len(sel_human),
            "ai_none": len(sel_none),
            "ai_paraphrase": len(sel_para),
            "total": len(rows),
        },
        "human_domains": _counts(sel_human)[1],
        "ai_none_models": _counts(sel_none)[0],
        "ai_none_domains": _counts(sel_none)[1],
        "ai_para_models": _counts(sel_para)[0],
        "ai_para_domains": _counts(sel_para)[1],
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--url", default=None,
                    help="Override CSV URL (RAID GitHub mirror fallback).")
    ap.add_argument("--hf-file", default=_DEFAULT_FILE, choices=["train.csv", "extra.csv"])
    ap.add_argument("--max-words", type=int, default=1500)
    ap.add_argument("--oversample", type=float, default=2.5,
                    help="Candidate pool multiple of per-class before stratified pick.")
    ap.add_argument("--max-ai-windows", type=int, default=500)
    args = ap.parse_args()

    url = _resolve_url(args.url, args.hf_file)
    t0 = time.time()
    summary = build(args.per_class, args.seed, args.out, url,
                    args.max_words, args.oversample, args.max_ai_windows)
    summary["seconds"] = round(time.time() - t0, 1)

    print("\n[fetch] SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    c = summary["counts"]
    if min(c["human"], c["ai_none"], c["ai_paraphrase"]) == 0:
        print("\nWARNING: a stratum is empty — inspect access/sampling before scoring.",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
