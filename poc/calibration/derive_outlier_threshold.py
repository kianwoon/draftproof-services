"""Derive the stylometric outlier-flag decision rule — the PROVENANCE of the two
constants `OUTLIER_PER_FEATURE_Z_THRESHOLD` and `OUTLIER_MIN_DEVIATING_FEATURES` in
`poc/detect/stylometry/outliers.py`.

THE PROBLEM
-----------
`detect_outliers` scored each paragraph by the MAX of ~m=16 leave-one-out per-feature
robust (median/MAD) z-scores and flagged it when that max exceeded 3.5 — Iglewicz &
Hoaglin (1993)'s cutoff for a SINGLE modified-z comparison. A max over ~16
comparisons on small n crosses 3.5 far too often: it flagged ~31% of paragraphs
(per-paragraph FP) in genuine single-author human prose (49 Gutenberg essays; 100% of
docs flagged — see calibration/consistency_human_precision.json before this change).
Normal-theory Bonferroni/Šidák does NOT fix it, because the small-n leave-one-out MAD
z-scores are NOT ~N(0,1): they have fat tails, so a per-feature threshold picked from
normal quantiles undershoots the empirical false-positive rate. The threshold must be
calibrated EMPIRICALLY against the z-scores production actually produces.

THE METHOD
----------
1. Load the LOCAL-ONLY human corpus (single-author prose = the null: nothing is truly
   inconsistent, so every flag is a false positive). Split it author-stratified with a
   named seed into a DERIVATION set (rule is chosen here) and a HOLDOUT set (rule is
   only validated here — never re-picked on holdout).
2. Score every eligible paragraph ONCE via `outliers.score_paragraphs` (the exact
   production scoring path — no re-implementation drift), caching the raw per-feature
   z-vectors so every candidate rule is evaluated against identical z-scores.
3. Sweep three rule families over the derivation set:
     (a) raised global max-threshold  — flag if max z > T
     (b) Šidák-corrected per-feature  — the theoretical point of family (a); reported
         to SHOW it undershoots (empirical FP != target), which is the whole finding
     (c) k-of-m                        — flag if >= K features exceed per-feature t
   For each candidate: per-paragraph human FP (derivation), plus two sensitivity
   metrics so the rule cannot go blind —
     (i)  the shifted-paragraph fixture must still flag its shifted paragraph;
     (ii) multi-author concat pseudo-docs must keep a per-paragraph flag rate clearly
          above the human rate (report the concat/human RATIO).
4. Choose: keep candidates meeting ALL hard constraints — derivation FP <= TARGET_FP,
   fixture still flags the shifted paragraph, ESL proficiency-parity gate PASSES, and
   concat/human separation ratio >= MIN_CONCAT_SEPARATION_RATIO — then MAXIMIZE
   ABSOLUTE concat sensitivity (tie-break: higher ratio, then lower human FP). Then
   VALIDATE the winner on the holdout set; if holdout FP exceeds TARGET_FP by more
   than HOLDOUT_TOLERANCE_PCT, say so honestly (do not re-pick).
5. Emit evidence to calibration/outlier_threshold_derivation.json (aggregates + full
   sweep table + chosen rule + holdout numbers). No corpus text is ever written.

Deterministic: the only randomness is the seeded author-stratified split. Cheap:
stylometry only, no ML models. Corpus stays local-only (never committed).

Usage (from poc/):
    python calibration/derive_outlier_threshold.py \
        --corpus ~/Downloads/human_precision_corpus \
        --out calibration/outlier_threshold_derivation.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

_POC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_POC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detect.stylometry.features import extract_fingerprints
from detect.stylometry.outliers import (
    _FEATURE_EXTRACTORS,
    MIN_PARAGRAPHS,
    MIN_WORDS_PER_PARAGRAPH,
    score_paragraphs,
)

# --- Derivation constants (named; the rule's reproducibility depends on them) -----

# Seed for the author-stratified derivation/holdout split. Fixed so the split — and
# therefore the chosen rule — is byte-reproducible from the same staged corpus.
SPLIT_SEED = 20260721

# Fraction of each author's documents assigned to the DERIVATION set (rest = holdout).
# Author-stratified: the split is taken WITHIN each author so both sets cover every
# author, and no author's prose is unique to one side.
DERIVATION_FRACTION = 0.65

# Target per-paragraph false-positive ceiling on single-author human prose.
TARGET_FP_PCT = 5.0

# Holdout is validation-only. If holdout FP exceeds TARGET_FP by more than this many
# percentage points, the run reports a HOLDOUT_CONCERN instead of silently passing —
# we never re-pick the rule on the holdout.
HOLDOUT_TOLERANCE_PCT = 2.0

# Informativeness guardrail: a flag is only worth showing if a paragraph in a KNOWN
# multi-author concat is at least this many times more likely to be flagged than a
# paragraph in clean single-author prose. 1.3x is a clear (not knife-edge, given the
# small-n concat noise) separation. This guardrail is what stops the selection from
# being fooled: the concat/human RATIO climbs monotonically as a rule gets stricter
# (human FP shrinks faster than concat FP), so maximizing the ratio alone would pick a
# needlessly strict rule with poor ABSOLUTE sensitivity. Instead we require ratio >=
# this bar, then maximize absolute concat sensitivity — a criterion stable under grid
# extension (stricter rules never win on absolute sensitivity).
MIN_CONCAT_SEPARATION_RATIO = 1.3

# m = number of stylometric feature dimensions, taken from the production feature
# vector (NOT hardcoded) — used for the Bonferroni/Šidák family-wise correction.
NUM_FEATURES = len(_FEATURE_EXTRACTORS)

# Family-wise error target the Šidák correction is solved for (matches TARGET_FP).
SIDAK_FAMILY_ALPHA = TARGET_FP_PCT / 100.0

# --- Sweep grids ------------------------------------------------------------------
# Max-threshold family (a): candidate global max cutoffs, in robust-z units.
_MAX_THRESHOLD_GRID = [round(3.5 + 0.25 * i, 2) for i in range(0, 35)]  # 3.5 .. 12.0
# k-of-m family (c): per-feature z cutoffs x minimum deviating-feature counts. The t
# grid runs well past the eventual operating point (up to 9.0) purely for transparency
# — the selection criterion (max absolute sensitivity subject to a ratio guardrail) is
# stable under this extension, so a wide grid documents the whole trade-off without
# changing the winner.
_KOFM_T_GRID = [round(2.5 + 0.25 * i, 2) for i in range(0, 27)]  # 2.5 .. 9.0
_KOFM_K_GRID = [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Author-stratified split
# ---------------------------------------------------------------------------
def _author_of(filename: str) -> str:
    """Author key = filename stem up to the first underscore (bacon_01 -> bacon,
    higher_200m -> higher). Matches the staged corpus naming; ESL proficiency bands
    ('higher'/'lower') act as their own strata."""
    return Path(filename).stem.split("_", 1)[0]


def stratified_split(files: list[Path], seed: int, frac: float) -> tuple[list[Path], list[Path]]:
    """Split `files` into (derivation, holdout), stratified by author. Within each
    author the sorted files are shuffled with `seed` and the first ceil(frac*n) go to
    derivation (>=1 each so no author is holdout-only when it has >=1 doc)."""
    by_author: dict[str, list[Path]] = {}
    for f in sorted(files):
        by_author.setdefault(_author_of(f.name), []).append(f)

    derivation: list[Path] = []
    holdout: list[Path] = []
    for author in sorted(by_author):
        docs = sorted(by_author[author])
        random.Random(f"{seed}:{author}").shuffle(docs)
        n_der = max(1, math.ceil(len(docs) * frac)) if docs else 0
        derivation += docs[:n_der]
        holdout += docs[n_der:]
    return sorted(derivation), sorted(holdout)


# ---------------------------------------------------------------------------
# Scoring / caching: per-paragraph z-vectors for a set of documents
# ---------------------------------------------------------------------------
def score_documents(files: list[Path]) -> list[list[float]]:
    """Return one z-vector (list of per-feature robust z-scores) per ELIGIBLE
    paragraph across `files`, via the production `score_paragraphs` path. A document
    below the eligibility floor contributes zero paragraphs (matches detect_outliers,
    which fails open there)."""
    paragraph_zvectors: list[list[float]] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        fps = extract_fingerprints(text)
        for _pid, feature_scores in score_paragraphs(fps):
            paragraph_zvectors.append([z for _name, z in feature_scores])
    return paragraph_zvectors


# ---------------------------------------------------------------------------
# Decision rules (pure, evaluated against cached z-vectors)
# ---------------------------------------------------------------------------
def _flag_max(zvec: list[float], threshold: float) -> bool:
    return bool(zvec) and max(zvec) > threshold


def _flag_kofm(zvec: list[float], per_feature_t: float, k: int) -> bool:
    return sum(1 for z in zvec if z > per_feature_t) >= k


def fp_rate(zvectors: list[list[float]], predicate) -> float:
    """Per-paragraph flag rate (%) under `predicate` over a paragraph z-vector list."""
    if not zvectors:
        return 0.0
    flagged = sum(1 for zv in zvectors if predicate(zv))
    return round(flagged / len(zvectors) * 100.0, 3)


def sidak_per_feature_threshold(family_alpha: float, m: int) -> float:
    """Two-sided Šidák per-feature robust-z cutoff for family-wise error `family_alpha`
    over `m` features, under the modified-z's design assumption z ~ N(0,1). Reported
    to demonstrate it UNDERSHOOTS empirically (the fat-tail finding), not to ship."""
    per_feature_alpha = 1.0 - (1.0 - family_alpha) ** (1.0 / m)  # two-sided total
    # Inverse standard-normal CDF (Acklam-free: use statistics.NormalDist).
    from statistics import NormalDist

    return NormalDist().inv_cdf(1.0 - per_feature_alpha / 2.0)


# ---------------------------------------------------------------------------
# Sensitivity fixtures
# ---------------------------------------------------------------------------
def _fixture_shifted_zvectors() -> tuple[list[list[float]], int]:
    """Score the shifted-paragraph fixture (the SAME text the consistency test uses)
    and return (paragraph z-vectors, index of the shifted paragraph p006)."""
    # Imported lazily to avoid a test<->calibration import cycle at module load.
    from detect.test_consistency import _document_with_shifted_paragraph

    text = _document_with_shifted_paragraph()
    fps = extract_fingerprints(text)
    scored = score_paragraphs(fps)
    zvectors = [[z for _n, z in fs] for _pid, fs in scored]
    shifted_index = next((i for i, (pid, _fs) in enumerate(scored) if pid == "p006"), -1)
    return zvectors, shifted_index


def _concat_zvectors(corpus_root: Path) -> list[list[float]]:
    """Per-paragraph z-vectors over multi-author concat pseudo-docs, reusing
    consistency_esl_rates' pseudo-doc builder — the multi-author sensitivity control:
    a good rule keeps these flagged far more than single-author human paragraphs."""
    from calibration.consistency_esl_rates import (  # noqa: E402
        PSEUDO_DOC_SHUFFLE_SEED,
        _count_eligible_paragraphs,
        build_pseudo_docs,
    )

    gutenberg = sorted((corpus_root / "gutenberg").glob("*.txt"))
    texts = [f.read_text(encoding="utf-8", errors="ignore") for f in gutenberg]
    docs = build_pseudo_docs(texts, _count_eligible_paragraphs, seed=PSEUDO_DOC_SHUFFLE_SEED)
    zvectors: list[list[float]] = []
    for d in docs:
        fps = extract_fingerprints(d["text"])
        for _pid, fs in score_paragraphs(fps):
            zvectors.append([z for _n, z in fs])
    return zvectors


# ---------------------------------------------------------------------------
# ESL parity gate (HARD constraint): the chosen rule must not over-flag lower-
# proficiency ESL writers relative to higher. Reuses consistency_esl_rates' concat
# machinery + verdict pipeline verbatim, so a candidate that passes here passes the
# standalone `consistency_esl_rates.py --mode concat` gate by construction. This
# matters BECAUSE the old rule "passed" that gate only via SATURATION (it flagged
# ~100% of pseudo-docs in both bands, forcing the gap to 0); a precise rule comes off
# that ceiling and must hold real per-band parity.
# ---------------------------------------------------------------------------
def esl_band_pseudo_doc_zvectors() -> dict | None:
    """Per proficiency band, the list of concat pseudo-docs — each a list of its
    paragraphs' z-vectors — scored ONCE via the production path. None if the local
    SCoCESLE corpus is absent (then the ESL gate cannot be evaluated and is skipped,
    not silently passed — see run_sweep). Deterministic (fixed shuffle seed)."""
    try:
        from calibration.consistency_esl_rates import (
            PSEUDO_DOC_SHUFFLE_SEED,
            _count_eligible_paragraphs,
            _read_texts,
            build_pseudo_docs,
        )
        from calibration.fpr_subgroup_gate import DEFAULT_CORPUS, _proficiency_groups
    except Exception:
        return None

    groups_files = _proficiency_groups(DEFAULT_CORPUS)
    if not any(groups_files.values()):
        return None

    bands: dict[str, list[list[list[float]]]] = {}
    for gname, files in sorted(groups_files.items()):
        texts = _read_texts(files, None)
        docs = build_pseudo_docs(texts, _count_eligible_paragraphs, seed=PSEUDO_DOC_SHUFFLE_SEED)
        band_docs: list[list[list[float]]] = []
        for d in docs:
            fps = extract_fingerprints(d["text"])
            band_docs.append([[z for _n, z in fs] for _pid, fs in score_paragraphs(fps)])
        bands[gname] = band_docs
    return bands


def esl_parity(bands: dict, predicate) -> dict:
    """Per-band pseudo-doc flag rates + the standalone gate's exact verdict, under
    `predicate`. A pseudo-doc is flagged if ANY of its paragraphs is."""
    from calibration.consistency_esl_rates import (
        _saturation_note,
        compute_verdict,
        flag_rate_gap_pts,
        groups_measurable,
    )

    gs: dict[str, dict] = {}
    for gname, band_docs in bands.items():
        n = len(band_docs)
        flagged = sum(1 for doc in band_docs if any(predicate(zv) for zv in doc))
        gs[gname] = {
            "flagged_essay_rate_pct": round(flagged / n * 100.0, 2) if n else None,
            "n_eligible": n,
        }
    gap = flag_rate_gap_pts(gs)
    measurable = gap is not None and groups_measurable(gs)
    saturated = _saturation_note(gs)["saturated"]
    verdict = compute_verdict(gap, measurable, saturated=saturated)
    return {
        "gap_pts": gap,
        "verdict": verdict,
        "measurable": measurable,
        "band_flag_rates_pct": {g: s["flagged_essay_rate_pct"] for g, s in gs.items()},
        "ok": verdict in ("PASS", "PASS_SATURATED"),
    }


# ---------------------------------------------------------------------------
# Sweep + choose
# ---------------------------------------------------------------------------
def _candidate(name: str, family: str, params: dict, predicate,
               der_z, concat_z, fixture_z, shifted_idx, esl_bands) -> dict:
    der_fp = fp_rate(der_z, predicate)
    concat_fp = fp_rate(concat_z, predicate)
    human_fp = der_fp if der_fp > 0 else 0.0
    ratio = round(concat_fp / human_fp, 3) if human_fp > 0 else (
        float("inf") if concat_fp > 0 else 0.0
    )
    fixture_flags = sum(1 for zv in fixture_z if predicate(zv))
    fixture_hits_shifted = bool(fixture_z) and shifted_idx >= 0 and predicate(fixture_z[shifted_idx])
    esl = esl_parity(esl_bands, predicate) if esl_bands is not None else None
    # ESL gate: if the corpus is present it is a HARD constraint; if absent (None) it
    # cannot be evaluated locally, so it does not block (reported as such).
    esl_ok = True if esl is None else esl["ok"]
    return {
        "name": name,
        "family": family,
        "params": params,
        "derivation_fp_pct": der_fp,
        "concat_fp_pct": concat_fp,
        "concat_over_human_ratio": None if ratio == float("inf") else ratio,
        "concat_ratio_is_inf": ratio == float("inf"),
        "fixture_total_flags": fixture_flags,
        "fixture_flags_shifted_paragraph": fixture_hits_shifted,
        "esl_parity": esl,
        "esl_parity_ok": esl_ok,
        "meets_fp_target": der_fp <= TARGET_FP_PCT,
        "fixture_ok": fixture_hits_shifted,
    }


def _ratio_key(c: dict) -> float:
    """Sort key for 'best sensitivity': inf ratio (human FP 0, concat > 0) ranks
    highest, then the finite ratio."""
    if c["concat_ratio_is_inf"]:
        return float("inf")
    return c["concat_over_human_ratio"] or 0.0


def _sensitivity_frontier(candidates: list[dict]) -> list[dict]:
    """Trade-off curve: at each concat/human ratio bar, the FP-feasible fixture-passing
    candidate with the HIGHEST absolute concat sensitivity. Shows why absolute
    sensitivity cannot be pushed toward the FP ceiling without collapsing the ratio."""
    feasible = [
        c for c in candidates
        if c["meets_fp_target"] and c["fixture_ok"] and c["esl_parity_ok"]
        and c["concat_over_human_ratio"] is not None
    ]
    frontier = []
    for bar in (1.0, 1.2, 1.5, 1.8, 2.0):
        qualifying = [c for c in feasible if c["concat_over_human_ratio"] >= bar]
        if not qualifying:
            frontier.append({"ratio_bar": bar, "best": None})
            continue
        best = max(qualifying, key=lambda c: c["concat_fp_pct"])
        frontier.append({
            "ratio_bar": bar,
            "best_rule": best["name"],
            "derivation_fp_pct": best["derivation_fp_pct"],
            "concat_fp_pct": best["concat_fp_pct"],
            "concat_over_human_ratio": best["concat_over_human_ratio"],
        })
    return frontier


def run_sweep(corpus_root: Path) -> dict:
    all_files = sorted((corpus_root / "gutenberg").glob("*.txt"))
    esl_dir = corpus_root / "scocesle_eligible"
    if esl_dir.is_dir():
        all_files += sorted(esl_dir.glob("*.txt"))

    der_files, hold_files = stratified_split(all_files, SPLIT_SEED, DERIVATION_FRACTION)
    der_z = score_documents(der_files)
    hold_z = score_documents(hold_files)
    concat_z = _concat_zvectors(corpus_root)
    fixture_z, shifted_idx = _fixture_shifted_zvectors()
    esl_bands = esl_band_pseudo_doc_zvectors()

    sidak_t = sidak_per_feature_threshold(SIDAK_FAMILY_ALPHA, NUM_FEATURES)

    candidates: list[dict] = []
    # Family (a): raised global max-threshold.
    for t in _MAX_THRESHOLD_GRID:
        candidates.append(_candidate(
            f"max>{t}", "max_threshold", {"max_threshold": t},
            lambda zv, _t=t: _flag_max(zv, _t),
            der_z, concat_z, fixture_z, shifted_idx, esl_bands))
    # Family (b): Šidák-corrected per-feature threshold (theoretical point of (a)).
    candidates.append(_candidate(
        f"sidak_max>{round(sidak_t, 4)}", "sidak_max_threshold",
        {"family_alpha": SIDAK_FAMILY_ALPHA, "m": NUM_FEATURES, "max_threshold": round(sidak_t, 6)},
        lambda zv, _t=sidak_t: _flag_max(zv, _t),
        der_z, concat_z, fixture_z, shifted_idx, esl_bands))
    # Family (c): k-of-m.
    for t in _KOFM_T_GRID:
        for k in _KOFM_K_GRID:
            candidates.append(_candidate(
                f"{k}-of-m@{t}", "k_of_m", {"per_feature_threshold": t, "min_deviating_features": k},
                lambda zv, _t=t, _k=k: _flag_kofm(zv, _t, _k),
                der_z, concat_z, fixture_z, shifted_idx, esl_bands))

    # SELECTION CRITERION (stated, not ad-hoc). HARD constraints a candidate must meet:
    #   (1) per-paragraph human FP on the derivation set <= TARGET_FP;
    #   (2) the shifted-paragraph fixture is still flagged (detector not blinded);
    #   (3) the ESL proficiency-parity gate PASSES (no over-flagging of lower- vs
    #       higher-proficiency ESL writers);
    #   (4) concat/human separation ratio >= MIN_CONCAT_SEPARATION_RATIO (a flag is
    #       INFORMATIVE — markedly more frequent on multi-author than clean prose;
    #       this detector is ADVISORY so an uninformative flag is pure noise).
    # Among survivors, MAXIMIZE ABSOLUTE concat sensitivity (concat_fp_pct = the true-
    # positive proxy). Why absolute-sensitivity-under-a-ratio-floor rather than pure
    # max-ratio: the ratio rises monotonically as a rule gets stricter (human FP
    # shrinks faster than concat FP), so maximizing the ratio alone selects a
    # needlessly strict, low-sensitivity rule — and the winner would drift every time
    # the t-grid is widened. Flooring the ratio then maximizing absolute sensitivity is
    # stable under grid extension. Tie-break: higher ratio, then lower human FP.
    def _kparam(c):
        return c["params"].get("min_deviating_features", 1)

    def _tparam(c):
        return c["params"].get("per_feature_threshold", c["params"].get("max_threshold", 0.0))

    def _ratio_at_least(c, bar):
        return c["concat_ratio_is_inf"] or (
            c["concat_over_human_ratio"] is not None and c["concat_over_human_ratio"] >= bar
        )

    eligible = [
        c for c in candidates
        if c["meets_fp_target"] and c["fixture_ok"] and c["esl_parity_ok"]
        and _ratio_at_least(c, MIN_CONCAT_SEPARATION_RATIO)
    ]
    chosen = None
    if eligible:
        chosen = sorted(
            eligible,
            key=lambda c: (c["concat_fp_pct"], _ratio_key(c), -c["derivation_fp_pct"]),
            reverse=True,
        )[0]

    result: dict = {
        "split": {
            "seed": SPLIT_SEED,
            "derivation_fraction": DERIVATION_FRACTION,
            "n_derivation_docs": len(der_files),
            "n_holdout_docs": len(hold_files),
            "n_derivation_paragraphs": len(der_z),
            "n_holdout_paragraphs": len(hold_z),
            "n_concat_paragraphs": len(concat_z),
            "derivation_authors": sorted({_author_of(f.name) for f in der_files}),
            "holdout_authors": sorted({_author_of(f.name) for f in hold_files}),
        },
        "config": {
            "target_fp_pct": TARGET_FP_PCT,
            "holdout_tolerance_pct": HOLDOUT_TOLERANCE_PCT,
            "num_features_m": NUM_FEATURES,
            "sidak_family_alpha": SIDAK_FAMILY_ALPHA,
            "sidak_per_feature_threshold": round(sidak_t, 6),
            "old_rule": "max_z > 3.5 (single-comparison Iglewicz-Hoaglin cutoff)",
            "esl_parity_gate_evaluated": esl_bands is not None,
            "esl_parity_note": (
                "HARD constraint from consistency_esl_rates.py --mode concat; the old "
                "rule only 'passed' it via saturation (both bands ~100% flagged)."
                if esl_bands is not None else
                "SCoCESLE corpus absent locally — ESL gate not evaluated, not blocked."
            ),
        },
        "selection_criterion": (
            "max ABSOLUTE concat sensitivity subject to: human per-paragraph FP <= "
            "target, fixture flags the shifted paragraph, ESL parity gate PASSES, and "
            f"concat/human ratio >= {MIN_CONCAT_SEPARATION_RATIO}; tie-break higher "
            "ratio then lower human FP"
        ),
        "min_concat_separation_ratio": MIN_CONCAT_SEPARATION_RATIO,
        "sensitivity_frontier": _sensitivity_frontier(candidates),
        "sweep": candidates,
        "chosen_rule": chosen,
    }

    if chosen is None:
        result["status"] = "NO_RULE_MEETS_TARGET"
        result["holdout"] = None
        return result

    # Validate chosen rule on holdout (no re-pick).
    if chosen["family"] == "k_of_m":
        t = chosen["params"]["per_feature_threshold"]
        k = chosen["params"]["min_deviating_features"]
        pred = lambda zv: _flag_kofm(zv, t, k)  # noqa: E731
    else:
        t = chosen["params"]["max_threshold"]
        pred = lambda zv: _flag_max(zv, t)  # noqa: E731
    hold_fp = fp_rate(hold_z, pred)
    over = round(hold_fp - TARGET_FP_PCT, 3)
    result["holdout"] = {
        "holdout_fp_pct": hold_fp,
        "target_fp_pct": TARGET_FP_PCT,
        "exceeds_target_by_pct": over,
        "tolerance_pct": HOLDOUT_TOLERANCE_PCT,
        "within_tolerance": hold_fp <= TARGET_FP_PCT + HOLDOUT_TOLERANCE_PCT,
    }
    result["status"] = "OK" if result["holdout"]["within_tolerance"] else "HOLDOUT_CONCERN"
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="~/Downloads/human_precision_corpus",
                    help="root holding gutenberg/ and scocesle_eligible/ subdirs")
    ap.add_argument("--out", default=None, help="write full JSON evidence here")
    args = ap.parse_args()

    corpus_root = Path(args.corpus).expanduser()
    if not (corpus_root / "gutenberg").is_dir():
        print(f"corpus root {corpus_root} has no gutenberg/ subdir", file=sys.stderr)
        sys.exit(2)

    res = run_sweep(corpus_root)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")

    ch = res["chosen_rule"]
    if ch is None:
        print("\nSTATUS: NO_RULE_MEETS_TARGET — no swept rule holds FP<=target with a "
              "passing fixture. Sensitivity/FP trade-off is unattainable on this grid.",
              file=sys.stderr)
        sys.exit(1)
    print(f"\nSTATUS: {res['status']}")
    print(f"CHOSEN: {ch['name']} — derivation FP={ch['derivation_fp_pct']}%, "
          f"concat FP={ch['concat_fp_pct']}%, ratio={ch['concat_over_human_ratio']}, "
          f"fixture_flags_shifted={ch['fixture_flags_shifted_paragraph']}, "
          f"holdout FP={res['holdout']['holdout_fp_pct']}%")


if __name__ == "__main__":
    main()
