"""§12 category-agreement measurement: run the REAL scan pipeline over the
4-class validation corpus and report how well the V7 breakdown's
primary_category agrees with the construction labels.

Output is a committable, numbers-only baseline JSON (no corpus text, no
filenames from the licensed set — hashes/counts/metrics only).

Honest scope notes (also embedded in the output):
- Labels are CONSTRUCTION labels (how each doc was made), not human reviewer
  labels — reviewer agreement is the remaining §12 step before the D5
  percentage unlock.
- This runs the QUICK-SCAN path (no Modal calls, no spend). Production runs
  deep-scan-fused detector input; fused-path agreement needs a paid pass
  (reuse the retune deep-scan cache when that is wanted).

Usage:
    DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN=1 is set by this script itself.
    python -m calibration.v12_validation.measure            # full 198 docs
    python -m calibration.v12_validation.measure --limit 8  # smoke (2/class)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"
AI_CASES_DIR = HERE.parent / "authorship_cases"
DEFAULT_OUT = HERE / "category_agreement_baseline.json"

_CLASSES = ("student_owned", "ai_assisted_polished", "ai_paraphrased", "ai_generated_like")

_wrapper_state: dict = {}


def install_cached_deep_scan(cache_path) -> None:
    """Wrap detect_v7.modal_client.call_deep_scan with a JSONL cache.

    Cache key = content_key(joined sentences, checkpoint). Only calibrated
    responses are cached (an uncalibrated response means checkpoint drift —
    caching it would silently pin stale scores)."""
    import detect_v7.modal_client as mc
    from calibration.retune import deepscan_cache as dc

    cache = dc.load_cache(cache_path)
    real = mc.call_deep_scan

    def cached(sentences):
        key = dc.content_key("\n".join(sentences), dc.checkpoint_tag())
        if key in cache:
            return {"available": True, "calibrated": True, "chunk_scores": cache[key]}
        resp = real(sentences)
        if (isinstance(resp, dict) and resp.get("available") is True
                and resp.get("calibrated") is True
                and isinstance(resp.get("chunk_scores"), list)):
            dc.append(cache_path, key, resp["chunk_scores"])
            cache[key] = resp["chunk_scores"]
        return resp

    _wrapper_state["real"] = real
    mc.call_deep_scan = cached


def uninstall_cached_deep_scan() -> None:
    import detect_v7.modal_client as mc
    if "real" in _wrapper_state:
        mc.call_deep_scan = _wrapper_state.pop("real")


def _resolve_text(row: dict, scocesle: Path) -> str | None:
    label, fname = row["label"], row["file"]
    if label == "student_owned":
        for d in scocesle.glob("*proficiency*"):
            p = d / fname
            if p.exists():
                return p.read_text(encoding="utf-8", errors="ignore")
        return None
    base = AI_CASES_DIR if label == "ai_generated_like" else CORPUS_DIR
    p = base / fname
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("text")


def measure(limit_per_class: int | None, fused: bool = False) -> dict:
    os.environ["DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN"] = "1"
    if fused:
        os.environ["DRAFTPROOF_V7_DEEP_SCAN"] = "1"
        from calibration.retune.deepscan_cache import DEFAULT_CACHE
        install_cached_deep_scan(DEFAULT_CACHE)
    else:
        os.environ.pop("DRAFTPROOF_V7_DEEP_SCAN", None)  # quick-scan only, no spend
    from calibration.measure_end_to_end import scan_text  # heavy import after env
    from calibration.retune.intake import DEFAULT_SCOCESLE
    from detect.run import DetectionRunner

    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    rows_by_class: dict[str, list[dict]] = defaultdict(list)
    for row in manifest["rows"]:
        rows_by_class[row["label"]].append(row)
    if limit_per_class:
        rows_by_class = {k: v[:limit_per_class] for k, v in rows_by_class.items()}

    runner = DetectionRunner()
    confusion: dict[str, Counter] = {c: Counter() for c in _CLASSES}
    reliable_asserts: dict[str, Counter] = {c: Counter() for c in _CLASSES}
    skipped = 0
    for label in _CLASSES:
        for row in rows_by_class.get(label, []):
            text = _resolve_text(row, DEFAULT_SCOCESLE)
            if not text:
                skipped += 1
                continue
            rep = scan_text(runner, text)
            bd = (rep.get("ai_risk_badge") or {}).get("authorship_breakdown") or {}
            predicted = bd.get("primary_category") or "none"
            confusion[label][predicted] += 1
            if bd.get("primary_category_reliable"):
                reliable_asserts[label][predicted] += 1
            print(".", end="", flush=True)
    print()

    result: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec": ("v7 §12 category agreement — construction labels, deep-scan-fused path (cached Modal)"
                 if fused else
                 "v7 §12 category agreement — construction labels, quick-scan path"),
        "caveats": [
            "construction labels, not human reviewer labels",
            ("deep-scan-fused path (cached Modal)" if fused else
             "quick-scan detector input; production runs deep-scan-fused"),
        ],
        "skipped_unresolvable": skipped,
        "per_class": {},
        "confusion": {c: dict(confusion[c]) for c in _CLASSES},
        "reliable_assertions": {c: dict(reliable_asserts[c]) for c in _CLASSES},
    }
    for c in _CLASSES:
        n = sum(confusion[c].values())
        correct = confusion[c].get(c, 0)
        # student_owned mislabeled as ANY ai_* is the false-accusation direction
        false_ai = (sum(v for k, v in confusion[c].items() if k.startswith("ai_"))
                    if c == "student_owned" else None)
        result["per_class"][c] = {
            "n": n,
            "primary_accuracy": round(correct / n, 4) if n else None,
            **({"false_ai_primary_rate": round(false_ai / n, 4)} if false_ai is not None and n else {}),
        }
    macro = [v["primary_accuracy"] for v in result["per_class"].values()
             if v["primary_accuracy"] is not None]
    result["macro_primary_accuracy"] = round(sum(macro) / len(macro), 4) if macro else None
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="V7 §12 category-agreement measurement")
    ap.add_argument("--limit", type=int, default=None, help="docs per class (smoke run)")
    ap.add_argument("--fused", action="store_true",
                     help="run the deep-scan-fused path (cached Modal, paid on cache miss)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (HERE / "category_agreement_fused_baseline.json" if args.fused else DEFAULT_OUT)
    try:
        result = measure(args.limit, fused=args.fused)
    finally:
        uninstall_cached_deep_scan()
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in ("per_class", "macro_primary_accuracy")}, indent=2))
    print(f"baseline -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
