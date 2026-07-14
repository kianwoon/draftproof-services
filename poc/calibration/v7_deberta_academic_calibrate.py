"""V7 Modal-hosted checkpoint (desklib/ai-text-detector-academic-v1.01) calibration probe.

STANDALONE OFFLINE ANALYSIS ONLY. Does not touch DRAFTPROOF_V7_DEEP_SCAN or any production
wiring. Mirrors the SENTENCE-LEVEL threshold-proportion methodology of the existing production
composer `poc/detect/deberta_signal.py` (SENT_THRESHOLD=0.99 convention) and the metric shape
of `poc/calibration/fpr_subgroup_gate.py::measure()` — NOT a naive document-level isotonic fit,
which is the KNOWN-BROKEN approach documented in `poc/calibration/deberta_fit_calibrator.py`
(collapsed to a step function on SCoCESLE because doc-level AI/human scores barely overlap).

Sentence splitting reuses `poc.detect.deberta_windowing.split_sentences` (no re-implementation).
Sentences are scored INDIVIDUALLY (not windowed into 3-sentence groups) via the Modal endpoint's
`chunks` batch parameter, batched ~25/call to bound HTTP round-trips while staying well under the
endpoint's ~60s timeout.

For each candidate threshold, an essay is "flagged high-confidence" per-sentence iff
score >= threshold; the document signal is the PROPORTION of its sentences flagged. FPR/TPR are
then computed as in fpr_subgroup_gate.py: proportion of essays whose doc-level flagged-proportion
crosses a doc-level call floor (we sweep doc floors too — see DOC_FLOOR_SWEEP) OR, more simply and
transparently, we report both the per-sentence separation AND multiple doc-level aggregate cuts.

Usage:
    python poc/calibration/v7_deberta_academic_calibrate.py --limit-per-group 30   # smoke subset
    python poc/calibration/v7_deberta_academic_calibrate.py                        # full corpus
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

_POC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_POC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import urllib.request  # noqa: E402
import urllib.error  # noqa: E402

from detect.deberta_windowing import split_sentences  # noqa: E402

HERE = Path(__file__).resolve().parent
AI_CASES = HERE / "authorship_cases"
DEFAULT_CORPUS = "/Users/kianwoonwong/Downloads/Small Corpus of Colombian English as a Second Language Essays (SCoCESLE)"
DEFAULT_OUT = HERE / "v7_deberta_academic_baseline.json"

# Candidate sentence-level high-confidence thresholds. 0.99 matches the existing production
# SENT_THRESHOLD convention; 0.90/0.95 are lower bars swept to see if a lower cut still holds
# ESL FPR down (the desklib checkpoint is a DIFFERENT model, may not share the same 0.99 shape).
THRESHOLD_SWEEP = [0.80, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9999, 0.99999]
# Doc-level call: an essay is "flagged" iff its proportion of high-confidence sentences >= this.
# Swept alongside sentence threshold since we don't yet know a good operating point.
DOC_FLOOR_SWEEP = [0.10, 0.20, 0.30, 0.50, 0.70, 0.90]

BATCH_SIZE = 25
MIN_SENTENCE_WORDS = 4  # skip near-empty fragments (headers, stray newlines)


DEFAULT_CHECKPOINT = "desklib/ai-text-detector-academic-v1.01"


def _checkpoint_label() -> str:
    """Provenance label for the checkpoint actually scored via Modal.

    Derives from DRAFTPROOF_MODAL_CHECKPOINT when set (the endpoint may be running a
    fine-tuned/staging checkpoint, not the original desklib model), falling back to the
    original literal only when the env var is unset -- so calibration artifacts never
    misreport what was actually scored.
    """
    checkpoint = os.environ.get("DRAFTPROOF_MODAL_CHECKPOINT") or DEFAULT_CHECKPOINT
    return f"{checkpoint} (via Modal, uncalibrated)"


def _endpoint_config() -> tuple[str, str]:
    url = os.environ.get("DRAFTPROOF_MODAL_ENDPOINT_URL")
    token = os.environ.get("DRAFTPROOF_MODAL_ENDPOINT_TOKEN")
    if not url or not token:
        print("DRAFTPROOF_MODAL_ENDPOINT_URL / _TOKEN not set in environment.", file=sys.stderr)
        raise SystemExit(2)
    return url, token


def _call_endpoint(url: str, token: str, chunks: list[str], retries: int = 3) -> list[float]:
    body = json.dumps({"chunks": chunks}).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode())
                if not data.get("available", True):
                    raise RuntimeError(f"endpoint reported unavailable: {data}")
                return [float(s) for s in data["chunk_scores"]]
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"    [retry {attempt+1}/{retries}] {e} — waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"endpoint call failed after {retries} retries: {last_err}")


def _score_sentences(url: str, token: str, sentences: list[str]) -> list[float]:
    scores: list[float] = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        scores.extend(_call_endpoint(url, token, batch))
    return scores


def _proficiency_groups(corpus: str) -> dict:
    base = Path(corpus)
    groups: dict[str, list[str]] = {"higher": [], "lower": []}
    for d in base.glob("*proficiency*"):
        name = d.name.lower()
        key = "higher" if "higher" in name else ("lower" if "lower" in name else None)
        if key:
            groups[key] += sorted(glob.glob(str(d / "*.txt")))
    return groups


def _ai_texts() -> list[str]:
    out = []
    for p in sorted(glob.glob(str(AI_CASES / "*.json"))):
        d = json.loads(Path(p).read_text())
        if (d.get("authorship") or "").lower() == "ai" and d.get("text"):
            out.append(d["text"])
    return out


def _essay_sentences(text: str) -> list[str]:
    sents = split_sentences(text)
    return [s for s in sents if len(s.split()) >= MIN_SENTENCE_WORDS]


def _dist(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    def pct(p):
        return round(s[min(len(s) - 1, int(len(s) * p))], 4)
    return {
        "n": len(vals), "min": round(min(vals), 4), "max": round(max(vals), 4),
        "mean": round(statistics.mean(vals), 4),
        "p10": pct(0.10), "p50": pct(0.50), "p90": pct(0.90),
    }


def _score_corpus_essays(url: str, token: str, label: str, texts: list[str]) -> list[dict]:
    """Score every essay's sentences; return per-essay {n_sentences, sentence_scores}."""
    results = []
    for idx, text in enumerate(texts):
        sents = _essay_sentences(text)
        if not sents:
            results.append({"n_sentences": 0, "sentence_scores": []})
            continue
        try:
            scores = _score_sentences(url, token, sents)
        except RuntimeError as e:
            print(f"  [{label} #{idx}] FAILED: {e}", file=sys.stderr)
            results.append({"n_sentences": len(sents), "sentence_scores": [], "error": str(e)})
            continue
        results.append({"n_sentences": len(sents), "sentence_scores": scores})
        print(f"  [{label} #{idx+1}/{len(texts)}] {len(sents)} sentences scored, "
              f"mean={statistics.mean(scores) if scores else 0:.3f}")
    return results


def _essay_flag_proportion(sentence_scores: list[float], sent_thr: float) -> float | None:
    if not sentence_scores:
        return None
    n_flag = sum(1 for s in sentence_scores if s >= sent_thr)
    return n_flag / len(sentence_scores)


def _fpr_at(essays: list[dict], sent_thr: float, doc_floor: float) -> float | None:
    props = [_essay_flag_proportion(e["sentence_scores"], sent_thr) for e in essays]
    props = [p for p in props if p is not None]
    if not props:
        return None
    return round(100.0 * sum(1 for p in props if p >= doc_floor) / len(props), 1)


def _auc(ai: list[float], human: list[float]) -> float | None:
    if not ai or not human:
        return None
    wins = sum(1 for x in ai for y in human if x > y) + 0.5 * sum(1 for x in ai for y in human if x == y)
    return round(wins / (len(ai) * len(human)), 4)


def measure(corpus: str, limit_per_group: int | None) -> dict:
    url, token = _endpoint_config()
    groups = _proficiency_groups(corpus)
    if not groups["higher"] and not groups["lower"]:
        print(f"No SCoCESLE essays found under {corpus!r}.", file=sys.stderr)
        raise SystemExit(2)

    files = {}
    for g in ("higher", "lower"):
        files[g] = groups[g][:limit_per_group] if limit_per_group else groups[g]

    ai_texts = _ai_texts()
    if limit_per_group:
        ai_texts = ai_texts[:limit_per_group]

    t0 = time.time()
    essay_results = {}
    for g in ("higher", "lower"):
        texts = [Path(fp).read_text(encoding="utf-8", errors="ignore") for fp in files[g]]
        print(f"\n--- Scoring {g}-proficiency ({len(texts)} essays) ---")
        essay_results[g] = _score_corpus_essays(url, token, g, texts)

    print(f"\n--- Scoring AI-labeled ({len(ai_texts)} texts) ---")
    essay_results["ai"] = _score_corpus_essays(url, token, "ai", ai_texts)

    # Flatten sentence-level score pools for distribution / separation check.
    human_sent_scores = [s for g in ("higher", "lower") for e in essay_results[g]
                          for s in e["sentence_scores"]]
    ai_sent_scores = [s for e in essay_results["ai"] for s in e["sentence_scores"]]

    sweep = {}
    for sent_thr in THRESHOLD_SWEEP:
        for doc_floor in DOC_FLOOR_SWEEP:
            key = f"sent>={sent_thr}_docfloor>={doc_floor}"
            hi = _fpr_at(essay_results["higher"], sent_thr, doc_floor)
            lo = _fpr_at(essay_results["lower"], sent_thr, doc_floor)
            ai_tpr = _fpr_at(essay_results["ai"], sent_thr, doc_floor)  # same fn, "flagged" = TP for AI set
            parity = round(lo - hi, 1) if (hi is not None and lo is not None) else None
            sweep[key] = {
                "sent_threshold": sent_thr, "doc_floor": doc_floor,
                "fpr_higher_pct": hi, "fpr_lower_pct": lo, "parity_gap_pct": parity,
                "tpr_ai_pct": ai_tpr,
            }

    elapsed = time.time() - t0
    return {
        "checkpoint": _checkpoint_label(),
        "methodology": "sentence-level threshold-proportion (mirrors deberta_signal.py SENT_THRESHOLD design)",
        "n": {
            "higher_essays": len(essay_results["higher"]), "lower_essays": len(essay_results["lower"]),
            "ai_texts": len(essay_results["ai"]),
            "higher_sentences": sum(e["n_sentences"] for e in essay_results["higher"]),
            "lower_sentences": sum(e["n_sentences"] for e in essay_results["lower"]),
            "ai_sentences": sum(e["n_sentences"] for e in essay_results["ai"]),
        },
        "sentence_score_dist": {
            "human": _dist(human_sent_scores),
            "ai": _dist(ai_sent_scores),
        },
        "sentence_auc_ai_vs_human": _auc(ai_sent_scores, human_sent_scores),
        "threshold_sweep": sweep,
        "elapsed_sec": round(elapsed, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit-per-group", type=int, default=None)
    args = ap.parse_args()

    res = measure(args.corpus, args.limit_per_group)

    print("\n=== SUMMARY ===")
    print(f"N: higher={res['n']['higher_essays']} lower={res['n']['lower_essays']} ai={res['n']['ai_texts']}")
    print(f"sentence-level AUC (ai vs human): {res['sentence_auc_ai_vs_human']}")
    print(f"human sentence dist: {res['sentence_score_dist']['human']}")
    print(f"ai sentence dist: {res['sentence_score_dist']['ai']}")
    print("\nthreshold sweep (fpr_higher / fpr_lower / parity_gap / tpr_ai):")
    for key, v in res["threshold_sweep"].items():
        print(f"  {key}: hi={v['fpr_higher_pct']}% lo={v['fpr_lower_pct']}% "
              f"gap={v['parity_gap_pct']} tpr_ai={v['tpr_ai_pct']}%")
    print(f"\nelapsed: {res['elapsed_sec']}s")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
