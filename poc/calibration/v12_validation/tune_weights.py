"""Offline category-weight tuner for the V7 §12 category-accuracy program.

Replays captured per-doc signal rows (``captures/signals_fused.jsonl``, written
by ``capture_signals.py``) through the REAL scorer —
``detect_v7.category_scoring.score_paragraph`` under a candidate
``DRAFTPROOF_V7_WEIGHTS_PATH`` — to search for a ``category_weights`` +
``ai_assisted_polished_band`` vector that raises macro primary accuracy WITHOUT
regressing the two things that matter: the false-accusation rate on
``student_owned`` and the detector's core ``ai_generated_like`` competence.

This module NEVER reimplements scoring math. It mutates weights.json, points the
real ``detect_v7.config`` loader at the mutation, and reads back
``score_paragraph``'s ``primary_category``. Everything numeric that could bias
the outcome — trial count, seed, Dirichlet concentration, weight floor, band
exploration radius, holdout tolerance, generated-like floor — is a CLI flag with
the plan's default. The two hard-constraint BASELINE values are read at runtime
from ``--baseline-json`` (Task 2's fused baseline), never hardcoded.

Search design (owner-mandated, see task brief):
  * Stratified 70/30 tune/holdout split by label.
  * N random candidates: per category, Dirichlet(alpha = mult * current_weights)
    perturbation (concentrated near current, occasional exploration),
    renormalize to 1.0, floor each weight. Band lo/hi sampled uniform within
    +/-delta of current, keeping lo < hi.
  * Evaluate each on the tune split via the real scorer.
  * Objective: maximize tune macro accuracy SUBJECT TO both hard constraints
    (student_owned false-AI rate <= baseline; ai_generated_like accuracy >=
    baseline - floor). Tie-break: higher student_owned accuracy.
  * Top-5 -> coordinate refinement (+/-step per weight, keep improvements).
  * Winner must ALSO generalize: holdout macro >= tune macro - tolerance AND
    both hard constraints on holdout. Else "no candidate generalizes", stop.
  * Emit disagreement diagnostics (confusion matrix + ranked collapse routes)
    for winner AND baseline weights, on tune + holdout.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

# The four construction/label classes score_paragraph emits as primary_category
# (the with/without-comparison paraphrase variants collapse to ai_paraphrased
# inside the scorer via has_comparison_text).
_CLASSES: tuple[str, ...] = (
    "student_owned",
    "ai_assisted_polished",
    "ai_paraphrased",
    "ai_generated_like",
)

_BUNDLED_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2] / "detect_v7" / "weights.json"
)
_BAND_KEY = "ai_assisted_polished_band"
_WEIGHTS_ENV_VAR = "DRAFTPROOF_V7_WEIGHTS_PATH"

# Search-hyperparameter defaults (surfaced as CLI flags; the plan's values).
DEFAULT_TRIALS = 2000
DEFAULT_SEED = 42
DEFAULT_ALPHA_MULT = 8.0          # Dirichlet concentration = mult * current
DEFAULT_WEIGHT_FLOOR = 0.02       # min per-signal weight after renormalize
DEFAULT_BAND_DELTA = 0.15         # +/- uniform band exploration radius
DEFAULT_HOLDOUT_TOL = 0.10        # macro may drop this much tune -> holdout
DEFAULT_GENERATED_FLOOR = 0.08    # gen-like accuracy may drop this below base
DEFAULT_REFINE_STEP = 0.05        # coordinate-refinement per-weight step
DEFAULT_TOP_K = 5                 # candidates carried into refinement


# --------------------------------------------------------------------------- #
# Weights I/O + sampling primitives
# --------------------------------------------------------------------------- #
def load_base_weights() -> dict:
    """Deep copy of the bundled ``detect_v7/weights.json`` (the search's start)."""
    with _BUNDLED_WEIGHTS_PATH.open(encoding="utf-8") as fh:
        return copy.deepcopy(json.load(fh))


def band_bounds(band: dict) -> tuple[float, float]:
    """Return (low, high) from the band dict AS NAMED in weights.json."""
    return float(band["low"]), float(band["high"])


def _dirichlet(current: list[float], rng, alpha_mult: float) -> list[float]:
    """Sample a Dirichlet vector concentrated near ``current`` using only the
    passed stdlib ``random.Random`` (Gamma-ratio construction — keeps
    determinism tied to the caller's rng, no numpy global state).

    alpha_i = alpha_mult * current_i (a zero weight would give alpha 0, which
    gammavariate rejects, so alphas are floored to a tiny epsilon)."""
    gammas = []
    for w in current:
        alpha = max(alpha_mult * w, 1e-6)
        gammas.append(rng.gammavariate(alpha, 1.0))
    total = sum(gammas)
    if total <= 0.0:  # degenerate; fall back to uniform
        n = len(current)
        return [1.0 / n] * n
    return [g / total for g in gammas]


def _floor_and_renormalize(weights: list[float], floor: float) -> list[float]:
    """Return a vector summing to 1.0 with every entry >= ``floor``, staying as
    close as possible to the input proportions (water-filling): pin any entry
    that would fall below floor AT floor, then distribute the remaining mass
    (1 - floor*pinned) proportionally among the un-pinned entries. Repeat until
    the pinned set is stable — guaranteed to converge (the pinned set only
    grows) and to satisfy the floor exactly."""
    n = len(weights)
    if n == 0:
        return weights
    if floor * n >= 1.0:  # infeasible floor -> uniform is the only fixed point
        return [1.0 / n] * n

    w = [max(float(x), 0.0) for x in weights]
    pinned = [False] * n
    for _ in range(n + 1):
        free_idx = [i for i in range(n) if not pinned[i]]
        remaining = 1.0 - floor * (n - len(free_idx))
        free_sum = sum(w[i] for i in free_idx)
        if free_sum <= 0.0:  # nothing to distribute proportionally -> even split
            share = remaining / len(free_idx)
            for i in free_idx:
                w[i] = share
        else:
            for i in free_idx:
                w[i] = w[i] / free_sum * remaining
        newly_pinned = [i for i in free_idx if w[i] < floor - 1e-12]
        if not newly_pinned:
            break
        for i in newly_pinned:
            w[i] = floor
            pinned[i] = True
    for i in range(n):
        if pinned[i]:
            w[i] = floor
    s = sum(w)
    return [x / s for x in w]


def _sample_band(current_band: dict, rng, delta: float) -> dict:
    """Sample lo/hi uniformly within +/-delta of current, clamped to [0,1],
    guaranteeing lo < hi (retry, then fall back to current on the rare miss)."""
    lo0, hi0 = band_bounds(current_band)
    for _ in range(32):
        lo = min(max(rng.uniform(lo0 - delta, lo0 + delta), 0.0), 1.0)
        hi = min(max(rng.uniform(hi0 - delta, hi0 + delta), 0.0), 1.0)
        if lo < hi:
            return {"low": lo, "high": hi}
    return {"low": lo0, "high": hi0}


def sample_candidate(
    base: dict,
    rng,
    alpha_mult: float = DEFAULT_ALPHA_MULT,
    weight_floor: float = DEFAULT_WEIGHT_FLOOR,
    band_delta: float = DEFAULT_BAND_DELTA,
) -> dict:
    """Return a full weights.json copy with ONLY ``category_weights`` and
    ``ai_assisted_polished_band`` perturbed (Dirichlet-near-current, floored,
    renormalized; band uniform within +/-delta). All other keys untouched."""
    cand = copy.deepcopy(base)
    for cat, entries in cand["category_weights"].items():
        current = [float(e["weight"]) for e in entries]
        sampled = _dirichlet(current, rng, alpha_mult)
        floored = _floor_and_renormalize(sampled, weight_floor)
        for entry, w in zip(entries, floored):
            entry["weight"] = w
    cand[_BAND_KEY] = _sample_band(base[_BAND_KEY], rng, band_delta)
    return cand


# --------------------------------------------------------------------------- #
# Data split
# --------------------------------------------------------------------------- #
def split_rows(rows, seed) -> tuple[list, list]:
    """Stratified 70/30 tune/holdout split by label, deterministic in ``seed``.

    Within each label the rows are shuffled with a per-label-seeded RNG (label
    folded into the seed so classes don't shuffle in lockstep), then the first
    70% go to tune. Same rows + same seed -> identical partition."""
    import random

    by_label: dict[str, list] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    tune: list = []
    holdout: list = []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(group)
        cut = int(round(len(group) * 0.7))
        # Never starve a split when a class has >= 2 rows.
        if len(group) >= 2:
            cut = min(max(cut, 1), len(group) - 1)
        tune.extend(group[:cut])
        holdout.extend(group[cut:])
    return tune, holdout


# --------------------------------------------------------------------------- #
# Candidate evaluation (REAL scorer, REAL config path)
# --------------------------------------------------------------------------- #
def _confusion(rows) -> dict[str, Counter]:
    """Replay rows through the real score_paragraph; return label -> Counter of
    predicted primary_category. Rows pass through UNCHANGED (None/missing
    signals are handled inside the scorer)."""
    from detect_v7 import category_scoring

    conf: dict[str, Counter] = {c: Counter() for c in _CLASSES}
    for r in rows:
        label = r["label"]
        if label not in conf:  # tolerate stray labels without crashing
            conf[label] = Counter()
        result = category_scoring.score_paragraph(
            r["v7_signals"],
            r["calibrated_detector_score"],
            has_comparison_text=bool(r.get("has_comparison_text", False)),
            esl_score=None,
        )
        predicted = result.get("primary_category") or "none"
        conf[label][predicted] += 1
    return conf


def _metrics_from_confusion(conf: dict[str, Counter]) -> dict:
    """Derive per-class accuracy, student_owned false-AI rate, and macro
    accuracy from a confusion matrix — mirrors measure.py's definitions so
    tuner metrics are comparable to the committed baseline."""
    per_class: dict[str, dict] = {}
    accs: list[float] = []
    student_false_ai: Optional[float] = None
    for c in _CLASSES:
        counts = conf.get(c, Counter())
        n = sum(counts.values())
        correct = counts.get(c, 0)
        acc = (correct / n) if n else None
        per_class[c] = {"n": n, "primary_accuracy": acc}
        if acc is not None:
            accs.append(acc)
        if c == "student_owned" and n:
            false_ai = sum(v for k, v in counts.items() if k.startswith("ai_"))
            student_false_ai = false_ai / n
            per_class[c]["false_ai_primary_rate"] = student_false_ai
    macro = (sum(accs) / len(accs)) if accs else 0.0
    return {
        "macro_primary_accuracy": macro,
        "student_owned_false_ai_rate": (
            student_false_ai if student_false_ai is not None else 0.0
        ),
        "per_class": per_class,
        "confusion": {c: dict(conf.get(c, Counter())) for c in _CLASSES},
    }


def evaluate_candidate(candidate_path, rows) -> dict:
    """Point the real V7 config at ``candidate_path``, reload, replay ``rows``
    through the real scorer, and return metrics. Restores the prior
    ``DRAFTPROOF_V7_WEIGHTS_PATH`` in ``finally:`` — no matter the outcome."""
    from detect_v7 import config

    prior = os.environ.get(_WEIGHTS_ENV_VAR)
    try:
        os.environ[_WEIGHTS_ENV_VAR] = str(candidate_path)
        config.reload_weights(force=True)
        conf = _confusion(rows)
        return _metrics_from_confusion(conf)
    finally:
        if prior is None:
            os.environ.pop(_WEIGHTS_ENV_VAR, None)
        else:
            os.environ[_WEIGHTS_ENV_VAR] = prior
        config.reload_weights(force=True)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def collapse_routes(conf: dict[str, Counter]) -> list[dict]:
    """Ranked misclassification routes: for each label, every wrong predicted
    class as ``{"from","to","count","of"}``, sorted by count desc. This is the
    Phase-3 signal — 'ai_paraphrased -> ai_generated_like: 34/39' says build the
    paraphrase signal; 'ai_assisted_polished -> student_owned' says re-tune the
    band, not add a signal."""
    routes: list[dict] = []
    for label in _CLASSES:
        counts = conf.get(label, Counter())
        n = sum(counts.values())
        for predicted, cnt in counts.items():
            if predicted != label and cnt:
                routes.append(
                    {"from": label, "to": predicted, "count": cnt, "of": n}
                )
    routes.sort(key=lambda r: r["count"], reverse=True)
    return routes


def _fmt_routes(routes: list[dict], top: int = 6) -> str:
    if not routes:
        return "    (none — no misclassifications)"
    return "\n".join(
        f"    {r['from']} -> {r['to']}: {r['count']}/{r['of']}"
        for r in routes[:top]
    )


def _fmt_confusion(conf: dict[str, Counter]) -> str:
    header = "    " + "actual\\pred".ljust(24) + "".join(
        c[:14].ljust(16) for c in _CLASSES
    )
    lines = [header]
    for actual in _CLASSES:
        counts = conf.get(actual, Counter())
        row = "    " + actual.ljust(24) + "".join(
            str(counts.get(pred, 0)).ljust(16) for pred in _CLASSES
        )
        lines.append(row)
    return "\n".join(lines)


def _print_diagnostics(title: str, metrics: dict) -> None:
    conf = {c: Counter(metrics["confusion"].get(c, {})) for c in _CLASSES}
    print(f"\n=== {title} ===")
    print(f"  macro primary accuracy: {metrics['macro_primary_accuracy']:.4f}")
    print(
        f"  student_owned false-AI rate: "
        f"{metrics['student_owned_false_ai_rate']:.4f}"
    )
    for c in _CLASSES:
        pc = metrics["per_class"].get(c, {})
        acc = pc.get("primary_accuracy")
        acc_s = f"{acc:.4f}" if acc is not None else "n/a"
        print(f"    {c}: acc={acc_s} (n={pc.get('n', 0)})")
    print("  confusion matrix:")
    print(_fmt_confusion(conf))
    print("  top collapse routes:")
    print(_fmt_routes(collapse_routes(conf)))


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #
def load_baseline_constants(baseline_json: Path) -> dict:
    """Read the two hard-constraint anchors from Task 2's baseline artifact.

    Never hardcoded: ``student_owned`` false-AI rate and ``ai_generated_like``
    primary accuracy come straight from the baseline's ``per_class`` block."""
    data = json.loads(Path(baseline_json).read_text(encoding="utf-8"))
    pc = data["per_class"]
    return {
        "student_owned_false_ai_rate": float(
            pc["student_owned"]["false_ai_primary_rate"]
        ),
        "ai_generated_like_accuracy": float(
            pc["ai_generated_like"]["primary_accuracy"]
        ),
        "baseline_macro": data.get("macro_primary_accuracy"),
    }


def satisfies_constraints(metrics: dict, baseline: dict, generated_floor: float) -> bool:
    """Both hard constraints: (1) student_owned false-AI rate <= baseline;
    (2) ai_generated_like accuracy >= baseline - generated_floor."""
    false_ai = metrics["student_owned_false_ai_rate"]
    gen_acc = metrics["per_class"].get("ai_generated_like", {}).get(
        "primary_accuracy"
    )
    if gen_acc is None:
        return False
    if false_ai > baseline["student_owned_false_ai_rate"] + 1e-9:
        return False
    if gen_acc < baseline["ai_generated_like_accuracy"] - generated_floor - 1e-9:
        return False
    return True


def _objective_key(metrics: dict) -> tuple:
    """Maximize macro accuracy, tie-break on higher student_owned accuracy."""
    so_acc = metrics["per_class"].get("student_owned", {}).get(
        "primary_accuracy"
    ) or 0.0
    return (metrics["macro_primary_accuracy"], so_acc)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def _write_temp_candidate(cand: dict, path: Path) -> None:
    path.write_text(json.dumps(cand), encoding="utf-8")


def _refine_candidate(
    cand: dict,
    tune_rows,
    baseline: dict,
    generated_floor: float,
    step: float,
    weight_floor: float,
    scratch: Path,
    best_metrics: dict,
) -> tuple[dict, dict]:
    """Greedy coordinate refinement: nudge each signal's weight +/-step (within
    its category, renormalizing), keep any constraint-satisfying improvement.
    Returns (possibly improved candidate, its metrics)."""
    best = copy.deepcopy(cand)
    best_key = _objective_key(best_metrics)
    for cat, entries in best["category_weights"].items():
        for idx in range(len(entries)):
            for delta in (step, -step):
                trial = copy.deepcopy(best)
                t_entries = trial["category_weights"][cat]
                raw = [float(e["weight"]) for e in t_entries]
                raw[idx] = raw[idx] + delta
                if raw[idx] <= 0.0:
                    continue
                floored = _floor_and_renormalize(raw, weight_floor)
                for e, w in zip(t_entries, floored):
                    e["weight"] = w
                _write_temp_candidate(trial, scratch)
                m = evaluate_candidate(scratch, tune_rows)
                if not satisfies_constraints(m, baseline, generated_floor):
                    continue
                if _objective_key(m) > best_key:
                    best, best_metrics, best_key = trial, m, _objective_key(m)
    return best, best_metrics


def search(
    rows,
    baseline: dict,
    *,
    trials: int,
    seed: int,
    alpha_mult: float,
    weight_floor: float,
    band_delta: float,
    generated_floor: float,
    holdout_tol: float,
    refine_step: float,
    top_k: int,
    scratch_dir: Path,
) -> dict:
    """Run the full search. Returns a result dict with the winner (or None),
    winner/baseline metrics on tune + holdout, and generalization verdict."""
    import random

    base = load_base_weights()
    tune_rows, holdout_rows = split_rows(rows, seed=seed)
    # tempfile, NOT scratch_dir: the candidates/ dir is scanned by the
    # no-text-leakage guard and committed — an unscrubbbed eval scratch left
    # behind there fails the guard (and would get committed).
    import tempfile
    _scratch_tmp = tempfile.TemporaryDirectory(prefix="v12_tuner_")
    scratch = Path(_scratch_tmp.name) / "_candidate_scratch.json"

    # Baseline (bundled weights) metrics for reference + diagnostics.
    _write_temp_candidate(base, scratch)
    base_tune = evaluate_candidate(scratch, tune_rows)
    base_holdout = evaluate_candidate(scratch, holdout_rows)

    rng = random.Random(seed)
    scored: list[tuple[tuple, dict, dict]] = []  # (key, candidate, tune_metrics)
    for _ in range(trials):
        cand = sample_candidate(
            base, rng,
            alpha_mult=alpha_mult,
            weight_floor=weight_floor,
            band_delta=band_delta,
        )
        _write_temp_candidate(cand, scratch)
        m = evaluate_candidate(scratch, tune_rows)
        if not satisfies_constraints(m, baseline, generated_floor):
            continue
        scored.append((_objective_key(m), cand, m))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:top_k]

    # Coordinate refinement on the top-k; keep the best generalizer.
    refined: list[tuple[tuple, dict, dict]] = []
    for _, cand, m in top:
        rcand, rm = _refine_candidate(
            cand, tune_rows, baseline, generated_floor,
            refine_step, weight_floor, scratch, m,
        )
        refined.append((_objective_key(rm), rcand, rm))
    refined.sort(key=lambda t: t[0], reverse=True)

    verdict = "no_candidate_satisfies_constraints"
    winner = None
    winner_tune = None
    winner_holdout = None
    for _, cand, m in refined:
        _write_temp_candidate(cand, scratch)
        hm = evaluate_candidate(scratch, holdout_rows)
        generalizes = (
            hm["macro_primary_accuracy"]
            >= m["macro_primary_accuracy"] - holdout_tol - 1e-9
        )
        if generalizes and satisfies_constraints(hm, baseline, generated_floor):
            winner, winner_tune, winner_holdout = cand, m, hm
            verdict = "ok"
            break
    else:
        if refined:
            verdict = "no_candidate_generalizes"

    return {
        "verdict": verdict,
        "winner": winner,
        "winner_tune": winner_tune,
        "winner_holdout": winner_holdout,
        "baseline_tune": base_tune,
        "baseline_holdout": base_holdout,
        "n_tune": len(tune_rows),
        "n_holdout": len(holdout_rows),
        "n_candidates_satisfying": len(scored),
    }


# --------------------------------------------------------------------------- #
# Provenance + output
# --------------------------------------------------------------------------- #
def _slim_metrics(metrics: Optional[dict]) -> Optional[dict]:
    """Numbers-only projection of a metrics dict for provenance embedding."""
    if metrics is None:
        return None
    return {
        "macro_primary_accuracy": metrics["macro_primary_accuracy"],
        "student_owned_false_ai_rate": metrics["student_owned_false_ai_rate"],
        "per_class": {
            c: {
                "n": metrics["per_class"].get(c, {}).get("n", 0),
                "primary_accuracy": metrics["per_class"].get(c, {}).get(
                    "primary_accuracy"
                ),
            }
            for c in _CLASSES
        },
        "confusion": metrics["confusion"],
    }


# Mirror of test_no_text_leakage.py's contract so the committed candidate passes
# the numbers-only guard by construction. Only NOTE strings are affected: every
# functional weights.json string (signal names, "inverse"/"midrange", band keys)
# is far under this length, so scrubbing removes exactly what the guard flags and
# nothing the scorer reads.
_LEAKAGE_EXEMPT_KEYS = {"_notes", "_tuning_provenance", "provenance"}
_LEAKAGE_MAX_STRING_LEN = 300


def _scrub_leakage_notes(node, exempt: bool = False):
    """Drop any string longer than the guard's limit that sits OUTSIDE an exempt
    key (or below one). Returns the pruned node; keys whose value becomes a
    removed long note are dropped entirely."""
    if isinstance(node, dict):
        cleaned = {}
        for key, value in node.items():
            sub_exempt = exempt or str(key) in _LEAKAGE_EXEMPT_KEYS
            if (
                isinstance(value, str)
                and not sub_exempt
                and len(value) > _LEAKAGE_MAX_STRING_LEN
            ):
                continue  # drop the leakage-flagged note key
            cleaned[key] = _scrub_leakage_notes(value, sub_exempt)
        return cleaned
    if isinstance(node, list):
        return [_scrub_leakage_notes(v, exempt) for v in node]
    return node


def build_candidate_output(
    winner: dict,
    provenance: dict,
) -> dict:
    """Full weights.json copy (winner already has category_weights + band
    mutated) with a numbers-only ``_tuning_provenance`` key attached. Non-exempt
    long note strings are scrubbed so the committed artifact satisfies the
    dir's no-text-leakage guard by construction."""
    out = _scrub_leakage_notes(copy.deepcopy(winner))
    out["_tuning_provenance"] = provenance
    return out


def _rows_from_captures(captures_path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(captures_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="V7 §12 offline category-weight tuner (Task 4)."
    )
    ap.add_argument("--captures", type=Path, required=True,
                    help="captures/signals_fused.jsonl (from capture_signals.py)")
    ap.add_argument("--baseline-json", type=Path, required=True,
                    help="Task 2 fused baseline (constraint anchors read here)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "candidates")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--alpha-mult", type=float, default=DEFAULT_ALPHA_MULT)
    ap.add_argument("--weight-floor", type=float, default=DEFAULT_WEIGHT_FLOOR)
    ap.add_argument("--band-delta", type=float, default=DEFAULT_BAND_DELTA)
    ap.add_argument("--holdout-tol", type=float, default=DEFAULT_HOLDOUT_TOL)
    ap.add_argument("--generated-floor", type=float,
                    default=DEFAULT_GENERATED_FLOOR)
    ap.add_argument("--refine-step", type=float, default=DEFAULT_REFINE_STEP)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args(argv)

    rows = _rows_from_captures(args.captures)
    if not rows:
        print(f"No capture rows in {args.captures}; nothing to tune.")
        return 2
    baseline = load_baseline_constants(args.baseline_json)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = search(
        rows, baseline,
        trials=args.trials, seed=args.seed,
        alpha_mult=args.alpha_mult, weight_floor=args.weight_floor,
        band_delta=args.band_delta, generated_floor=args.generated_floor,
        holdout_tol=args.holdout_tol, refine_step=args.refine_step,
        top_k=args.top_k, scratch_dir=args.out_dir,
    )

    print(f"\n{'#' * 70}")
    print(f"# V7 §12 category-weight tuner — seed {args.seed}, {args.trials} trials")
    print(f"# tune rows: {result['n_tune']}  holdout rows: {result['n_holdout']}")
    print(f"# candidates satisfying constraints: "
          f"{result['n_candidates_satisfying']}")
    print(f"# baseline constants: {baseline}")
    print(f"{'#' * 70}")

    _print_diagnostics("BASELINE weights — tune split", result["baseline_tune"])
    _print_diagnostics("BASELINE weights — holdout split", result["baseline_holdout"])

    if result["verdict"] != "ok":
        print(f"\n>>> VERDICT: {result['verdict'].upper()} — "
              f"no candidate shipped. Phase 3 gate reads the collapse routes "
              f"above. STOP.")
        return 1

    _print_diagnostics("WINNER weights — tune split", result["winner_tune"])
    _print_diagnostics("WINNER weights — holdout split", result["winner_holdout"])

    provenance = {
        "seed": args.seed,
        "trials": args.trials,
        "alpha_mult": args.alpha_mult,
        "weight_floor": args.weight_floor,
        "band_delta": args.band_delta,
        "holdout_tol": args.holdout_tol,
        "generated_floor": args.generated_floor,
        "refine_step": args.refine_step,
        "top_k": args.top_k,
        "n_tune": result["n_tune"],
        "n_holdout": result["n_holdout"],
        "baseline_constants": baseline,
        "tune_metrics": _slim_metrics(result["winner_tune"]),
        "holdout_metrics": _slim_metrics(result["winner_holdout"]),
        "baseline_tune_metrics": _slim_metrics(result["baseline_tune"]),
        "baseline_holdout_metrics": _slim_metrics(result["baseline_holdout"]),
        "winner_collapse_routes_tune": collapse_routes(
            {c: Counter(result["winner_tune"]["confusion"].get(c, {}))
             for c in _CLASSES}
        ),
        "winner_collapse_routes_holdout": collapse_routes(
            {c: Counter(result["winner_holdout"]["confusion"].get(c, {}))
             for c in _CLASSES}
        ),
    }
    out = build_candidate_output(result["winner"], provenance)
    out_path = args.out_dir / f"weights_candidate_{args.seed}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    trials_path = args.out_dir / f"trials_{args.seed}.jsonl"
    with trials_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "seed": args.seed, "trials": args.trials,
            "verdict": result["verdict"],
            "n_candidates_satisfying": result["n_candidates_satisfying"],
        }) + "\n")

    print(f"\n>>> VERDICT: OK — wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
