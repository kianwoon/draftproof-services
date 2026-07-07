"""Task 6 — paraphrase-signal feature study: measure BEFORE building.

Phase 3 of the V7 category-accuracy program. After weight tuning,
`ai_paraphrased` primary accuracy is still 0% — its docs collapse into
`ai_generated_like` and `ai_assisted_polished`. This script measures whether
MiniLM sentence-embedding features can separate `ai_paraphrased` from the
other classes BEFORE any production wiring (Task 7 only proceeds if the hard
gate below passes).

Candidate features (per document, computed from MiniLM sentence embeddings):
  1. adjacent_cosine_mean — mean cosine similarity of consecutive sentences
     (paraphrased-AI text keeps the source's unnaturally even semantic gait).
  2. pairwise_cosine_std — std of all-pairs cosine similarity (low = monotone).
  3. embedding_norm_cv — coefficient of variation of embedding L2 norms.
  4. frame_reuse — the existing detect-side `structural_reuse` criterion
     value (poc/detect/criteria/structural_reuse.py), called standalone per
     doc (does NOT require a full DetectionRunner pass).

Output JSON is numbers-only (poc/calibration/v12_validation/test_no_text_leakage.py
enforces this on every committed *.json in this directory).

Usage:
    cd poc && python -m calibration.v12_validation.paraphrase_feature_study
    cd poc && python -m calibration.v12_validation.paraphrase_feature_study --limit 8  # smoke
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"
AI_CASES_DIR = HERE.parent / "authorship_cases"
DEFAULT_OUT = HERE / "paraphrase_feature_study.json"

_CLASSES = ("student_owned", "ai_assisted_polished", "ai_paraphrased", "ai_generated_like")
FEATURE_NAMES = ("adjacent_cosine_mean", "pairwise_cosine_std", "embedding_norm_cv", "frame_reuse")

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
MIN_SENTENCES = 3  # docs with fewer sentences are skipped (can't compute pairwise stats)

_embedder_cache: dict = {}


# ── Pure-Python stats helpers (unit-tested independently, no ML) ──────────

def rank_auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """Mann-Whitney rank-based AUC: P(pos_score > neg_score) with ties=0.5.

    Returns None if either group is empty (undefined AUC)."""
    if not pos or not neg:
        return None
    combined = [(v, 0) for v in pos] + [(v, 1) for v in neg]
    combined.sort(key=lambda x: x[0])
    n = len(combined)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average rank for the tie block
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, (v, g) in zip(ranks, combined) if g == 0)
    n_pos, n_neg = len(pos), len(neg)
    u_pos = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_pos / (n_pos * n_neg)


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolation quantile (numpy 'linear' method), no numpy dependency."""
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    n = len(a)
    if n < 2 or n != len(b):
        return None
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    denom = math.sqrt(var_a * var_b)
    if denom == 0:
        return None
    return cov / denom


def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# ── Corpus loading (mirrors measure.py's _resolve_text exactly) ───────────

def _resolve_text_with_group(row: dict, scocesle: Path):
    """Like measure._resolve_text but also returns the SCoCESLE proficiency
    group ('higher'/'lower'/None) for student_owned rows."""
    from calibration.v12_validation.measure import _resolve_text
    text = _resolve_text(row, scocesle)
    group = None
    if row["label"] == "student_owned" and text is not None:
        for d in scocesle.glob("*proficiency*"):
            if (d / row["file"]).exists():
                dname = d.name.lower()
                if "higher" in dname:
                    group = "higher"
                elif "lower" in dname:
                    group = "lower"
                break
    return text, group


# ── Feature computation ────────────────────────────────────────────────

def _get_embedder():
    if "model" not in _embedder_cache:
        from sentence_transformers import SentenceTransformer
        _embedder_cache["model"] = SentenceTransformer(MODEL_NAME)
    return _embedder_cache["model"]


def compute_minilm_features(sentences: List[str]) -> Dict[str, float]:
    embedder = _get_embedder()
    embs = embedder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    embs = [list(map(float, row)) for row in embs]

    adjacent = [cosine(embs[i], embs[i + 1]) for i in range(len(embs) - 1)]
    adjacent_cosine_mean = mean(adjacent)

    pairwise = []
    n = len(embs)
    for i in range(n):
        for j in range(i + 1, n):
            pairwise.append(cosine(embs[i], embs[j]))
    pairwise_cosine_std = stddev(pairwise)

    norms = [math.sqrt(sum(x * x for x in e)) for e in embs]
    norm_mean = mean(norms)
    embedding_norm_cv = (stddev(norms) / norm_mean) if norm_mean else 0.0

    return {
        "adjacent_cosine_mean": adjacent_cosine_mean,
        "pairwise_cosine_std": pairwise_cosine_std,
        "embedding_norm_cv": embedding_norm_cv,
    }


def compute_frame_reuse(text: str) -> Optional[float]:
    from detect.criteria.structural_reuse import score as structural_reuse_score
    try:
        result = structural_reuse_score(text)
        return float(result.value)
    except Exception:
        return None


# ── Study driver ───────────────────────────────────────────────────────

def run_study(limit_per_class: Optional[int] = None) -> dict:
    import calibration.measure_end_to_end  # noqa: F401  (sys.path shim, import first)
    from calibration.retune.intake import DEFAULT_SCOCESLE
    from detect.deberta_windowing import split_sentences

    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    rows_by_class: Dict[str, list] = defaultdict(list)
    for row in manifest["rows"]:
        rows_by_class[row["label"]].append(row)
    if limit_per_class:
        rows_by_class = {k: v[:limit_per_class] for k, v in rows_by_class.items()}

    # per-doc records: label, group (ESL proficiency or None), features
    docs: List[dict] = []
    skipped_unresolvable = 0
    skipped_short = 0

    for label in _CLASSES:
        for row in rows_by_class.get(label, []):
            text, group = _resolve_text_with_group(row, DEFAULT_SCOCESLE)
            if not text:
                skipped_unresolvable += 1
                continue
            sentences = split_sentences(text)
            if len(sentences) < MIN_SENTENCES:
                skipped_short += 1
                continue
            feats = compute_minilm_features(sentences)
            feats["frame_reuse"] = compute_frame_reuse(text)
            docs.append({"label": label, "group": group, "features": feats})
            print(".", end="", flush=True)
    print()

    frame_reuse_skipped = all(d["features"]["frame_reuse"] is None for d in docs) if docs else True

    # ── AUC tables ──────────────────────────────────────────────────
    def values_for(label: str, feature: str) -> List[float]:
        return [d["features"][feature] for d in docs
                if d["label"] == label and d["features"][feature] is not None]

    def values_group(group: str, feature: str) -> List[float]:
        return [d["features"][feature] for d in docs
                if d["group"] == group and d["features"][feature] is not None]

    auc_one_vs_one: Dict[str, Dict[str, Optional[float]]] = {}
    auc_one_vs_rest: Dict[str, Dict[str, Optional[float]]] = {}
    for feature in FEATURE_NAMES:
        auc_one_vs_one[feature] = {}
        for other in _CLASSES:
            if other == "ai_paraphrased":
                continue
            pos = values_for("ai_paraphrased", feature)
            neg = values_for(other, feature)
            auc_one_vs_one[feature][f"ai_paraphrased_vs_{other}"] = _round(rank_auc(pos, neg))

        pos = values_for("ai_paraphrased", feature)
        rest = [d["features"][feature] for d in docs
                if d["label"] != "ai_paraphrased" and d["features"][feature] is not None]
        auc_one_vs_rest[feature] = _round(rank_auc(pos, rest))

    # ESL subgroup check: does the feature covertly detect ESL proficiency?
    esl_subgroup: Dict[str, dict] = {}
    for feature in FEATURE_NAMES:
        higher = values_group("higher", feature)
        lower = values_group("lower", feature)
        # AUC for "lower flagged as more paraphrase-like than higher":
        # pos=lower (the group we must NOT systematically flag), neg=higher
        lower_vs_higher_auc = _round(rank_auc(lower, higher))
        student_all = values_for("student_owned", feature)
        ai_all = [d["features"][feature] for d in docs
                  if d["label"] != "student_owned" and d["features"][feature] is not None]
        student_vs_ai_auc = _round(rank_auc(ai_all, student_all))  # pos=AI classes (flagging direction)
        esl_subgroup[feature] = {
            "lower_vs_higher_auc": lower_vs_higher_auc,
            "n_higher": len(higher),
            "n_lower": len(lower),
            "student_owned_vs_ai_flagging_auc": student_vs_ai_auc,
        }

    # ── Pearson correlation matrix between features ────────────────
    corr: Dict[str, Dict[str, Optional[float]]] = {}
    for f1 in FEATURE_NAMES:
        corr[f1] = {}
        for f2 in FEATURE_NAMES:
            v1 = [d["features"][f1] for d in docs if d["features"][f1] is not None and d["features"][f2] is not None]
            v2 = [d["features"][f2] for d in docs if d["features"][f1] is not None and d["features"][f2] is not None]
            corr[f1][f2] = _round(pearson(v1, v2))

    # ── p10/p90 quantiles per feature (ai_paraphrased vs rest, informational) ──
    quantiles: Dict[str, dict] = {}
    for feature in FEATURE_NAMES:
        all_vals = [d["features"][feature] for d in docs if d["features"][feature] is not None]
        para_vals = values_for("ai_paraphrased", feature)
        quantiles[feature] = {
            "p10_all": _round(quantile(all_vals, 0.10)),
            "p90_all": _round(quantile(all_vals, 0.90)),
            "p10_ai_paraphrased": _round(quantile(para_vals, 0.10)),
            "p90_ai_paraphrased": _round(quantile(para_vals, 0.90)),
        }

    class_counts = {c: sum(1 for d in docs if d["label"] == c) for c in _CLASSES}

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec": "Task 6 — paraphrase-signal feature study (MiniLM standalone features vs frame_reuse)",
        "model": MODEL_NAME,
        "min_sentences_threshold": MIN_SENTENCES,
        "skipped_unresolvable": skipped_unresolvable,
        "skipped_short": skipped_short,
        "n_docs_used": len(docs),
        "class_counts": class_counts,
        "frame_reuse_note": ("skipped — requires full pipeline run; MiniLM features measured standalone"
                              if frame_reuse_skipped else
                              "computed standalone via detect.criteria.structural_reuse.score()"),
        "auc_one_vs_one": auc_one_vs_one,
        "auc_one_vs_rest": auc_one_vs_rest,
        "esl_subgroup_check": esl_subgroup,
        "feature_correlation_matrix": corr,
        "quantiles": quantiles,
    }
    return result


def _round(v):
    return round(v, 4) if isinstance(v, (int, float)) else v


def evaluate_gate(result: dict, auc_gate: float, esl_gate: float) -> dict:
    """Hard gate: best feature effective-AUC(ai_paraphrased vs ai_generated_like)
    >= auc_gate AND that feature must not be a covert ESL detector (< esl_gate
    in its flagging direction).

    Direction-aware: raw AUC is computed with pos=ai_paraphrased, so a feature
    that discriminates strongly in REVERSE (raw AUC << 0.5) is still a valid
    discriminator when used inverted. The winner is therefore selected by
    effect size |raw_auc - 0.5|; its direction is recorded
    ("high_is_paraphrase" if raw >= 0.5 else "low_is_paraphrase") and the gate
    is evaluated on the effective AUC = max(raw, 1 - raw).

    The ESL covert-detector check uses the SAME direction: the flagging
    direction is "which way does the feature push a doc toward paraphrase?" —
    if low_is_paraphrase, the concern is lower-proficiency essays scoring
    systematically LOWER than higher-proficiency ones, i.e. directional ESL
    AUC = 1 - lower_vs_higher_auc."""
    best_feature = None
    best_raw_auc = None
    best_effect = -1.0
    for feature, table in result["auc_one_vs_one"].items():
        v = table.get("ai_paraphrased_vs_ai_generated_like")
        if v is not None and abs(v - 0.5) > best_effect:
            best_effect = abs(v - 0.5)
            best_raw_auc = v
            best_feature = feature

    direction = None
    effective_auc = None
    esl_raw = None
    esl_directional = None
    polished_auc = None
    passed = False
    if best_feature is not None:
        direction = "high_is_paraphrase" if best_raw_auc >= 0.5 else "low_is_paraphrase"
        effective_auc = _round(max(best_raw_auc, 1.0 - best_raw_auc))
        # always report the ESL check for the best feature (informational even on AUC-gate failure)
        esl_raw = result["esl_subgroup_check"][best_feature]["lower_vs_higher_auc"]
        if esl_raw is not None:
            esl_directional = _round(esl_raw if direction == "high_is_paraphrase" else 1.0 - esl_raw)
        polished_auc = result["auc_one_vs_one"][best_feature].get("ai_paraphrased_vs_ai_assisted_polished")
        if effective_auc >= auc_gate:
            # covert ESL detector = lower-proficiency essays pushed toward
            # "paraphrase" in the winner's flagging direction
            esl_ok = esl_directional is None or esl_directional < esl_gate
            passed = esl_ok

    return {
        "auc_gate_threshold": auc_gate,
        "esl_gate_threshold": esl_gate,
        "best_feature": best_feature,
        "direction": direction,
        "best_feature_raw_auc_vs_ai_generated_like": _round(best_raw_auc) if best_feature else None,
        "best_feature_effective_auc_vs_ai_generated_like": effective_auc,
        "best_feature_esl_lower_vs_higher_auc": esl_raw,
        "best_feature_esl_directional_flagging_auc": esl_directional,
        "best_feature_auc_vs_ai_assisted_polished": polished_auc,
        "gate_passed": passed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Task 6 — paraphrase-signal feature study")
    ap.add_argument("--limit", type=int, default=None, help="docs per class (smoke run)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--auc-gate", type=float, default=0.70,
                     help="minimum best-feature AUC vs ai_generated_like to pass")
    ap.add_argument("--esl-gate", type=float, default=0.60,
                     help="max lower-vs-higher-proficiency AUC allowed (covert ESL detector check)")
    args = ap.parse_args()

    result = run_study(args.limit)
    gate = evaluate_gate(result, args.auc_gate, args.esl_gate)
    result["gate_verdict"] = gate

    out = args.out or DEFAULT_OUT
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(json.dumps({"auc_one_vs_one": result["auc_one_vs_one"],
                       "gate_verdict": gate}, indent=2))
    print(f"study -> {out}")
    return 0 if gate["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
