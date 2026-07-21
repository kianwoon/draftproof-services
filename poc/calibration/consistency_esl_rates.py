"""ESL flag-rate disparity check for the Phase-1 stylometric ConsistencyDetector.

WHY THIS EXISTS: the pre-push ESL gate (fpr_subgroup_gate.py) measures ONLY
badge["ai_likelihood_score"] (line 115). The consistency panel is score-neutral by
contract (overall_risk == 0.0), so that gate can NEVER catch the failure mode that
actually matters here: the "writing-style outliers" PANEL disproportionately
flagging lower-proficiency ESL writers, whose within-document style variance is
often genuine (memory: stylometric anomaly detection is exactly where ESL false
positives live). This script measures panel flag rates by proficiency band on the
local SCoCESLE corpus and FAILS (exit 1) on excessive disparity — it is the
decision gate the build plan deferred ("full gate run is a separate, later product
decision") and must PASS before DRAFTPROOF_CONSISTENCY is enabled in production.

TWO MEASUREMENT MODES (--mode):
  * essay  (DEFAULT, byte-identical to the original gate) — one real SCoCESLE essay
    = one sample. HONEST but on this corpus UNMEASURABLE: SCoCESLE essays are short,
    so only ~2% reach MIN_PARAGRAPHS eligible paragraphs (3 higher / 2 lower of 272,
    far below MIN_ELIGIBLE_PER_GROUP=20). This is the essay-mode evidence in
    consistency_esl_rates.json.
  * concat — SYMMETRIC SYNTHETIC CONCATENATION. Short same-band essays are
    deterministically concatenated (fixed-seed shuffle of sorted files) into
    pseudo-documents until each reaches MIN_PARAGRAPHS eligible paragraphs, so BOTH
    bands become equally multi-author and the A/B asks the single sharp question:
    "does the detector flag LOWER-band prose more than HIGHER-band prose, holding the
    construction procedure identical?" This restores statistical power on a
    short-essay corpus WITHOUT touching the detector or its thresholds.

    VALIDITY CAVEAT (stated in code, in --help, and in the output JSON): concat mode
    measures the detector's behaviour on MULTI-AUTHOR pseudo-documents. Every
    pseudo-doc contains genuine author boundaries, so its flag rate is an UPPER BOUND
    on style-break flagging — NOT the flag rate a real single-author ESL essay would
    see. The load-bearing quantity is the lower−higher GAP under identical
    construction, not the absolute rate. A residual confound is surfaced (not hidden)
    via per-band mean essays/eligible-paragraphs per pseudo-doc: if one band packs
    more source essays per pseudo-doc (shorter essays → more author boundaries), that
    inflates its flag rate for a reason correlated with — but not identical to —
    prose style. Read the gap alongside those observability fields.

Usage (from poc/):
    python calibration/consistency_esl_rates.py                       # essay mode, full corpus
    python calibration/consistency_esl_rates.py --limit 12            # quick smoke
    python calibration/consistency_esl_rates.py --out calibration/consistency_esl_rates.json
    python calibration/consistency_esl_rates.py --mode concat \
        --out calibration/consistency_esl_rates_concat.json          # concat-mode evidence

Purely deterministic and cheap: stylometry only, no ML models, no DetectionRunner.
Exit-code contract (per mode): PASS 0 / FAIL or UNMEASURABLE 1 / no corpus 2.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Mirrors fpr_subgroup_gate.py's sys.path setup: `detect/__init__.py` transitively
# imports `poc.predictability...`, so both poc/ (parents[1]) and the project root
# (parents[2], which makes `poc` importable) must be on sys.path for direct script
# invocation (`python calibration/consistency_esl_rates.py`) to work — pytest masks
# this because conftest.py already adds the project root for the test collector.
_POC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_POC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detect.stylometry.features import extract_fingerprints
from detect.stylometry.outliers import (
    MIN_PARAGRAPHS,
    MIN_WORDS_PER_PARAGRAPH,
    detect_outliers,
)

# Maximum permitted LOWER-PROFICIENCY OVER-FLAGGING gap: (lower − higher)
# flagged-essay-rate, in percentage points, among essays eligible for outlier
# statistics (≥MIN_PARAGRAPHS). DELIBERATELY ONE-SIDED — this is not a symmetric
# statistical-parity test. The specific product risk being controlled is
# lower-proficiency English writers being disproportionately flagged with
# style-inconsistency warnings; higher-proficiency essays flagging MORE than
# lower is not the failure mode this gate protects against, so a negative gap
# passes by design.
# Derivation: mirrors the spirit of fpr_subgroup_gate.MAX_PARITY_WIDEN_PTS (4.0 pts
# on FPR parity) with 1 pt of slack because this is an advisory review-flag rate,
# not a score/FPR — an advisory panel row costs the user a review glance, not a
# false accusation. Owner sign-off on this threshold + the measured numbers is a
# required Task 2 step before the prod flip.
MAX_FLAG_RATE_GAP_PTS = 5.0

# Below this many eligible essays in EITHER proficiency group, the flag-rate gap is
# statistical noise (SCoCESLE essays are short; many fall under MIN_PARAGRAPHS=6 and
# a 3-essay group must not PASS the flip). Verdict becomes UNMEASURABLE, exit 1.
# In concat mode this floor applies to the count of PSEUDO-DOCS per band (each is
# eligible by construction), reusing groups_measurable unchanged.
MIN_ELIGIBLE_PER_GROUP = 20

# --- concat-mode constants ------------------------------------------------------
# Fixed shuffle seed for pseudo-doc construction. NOT A TUNED VALUE: it exists only
# to make essay ordering reproducible run-to-run. Any fixed integer works — changing
# it reshuffles which essays share a pseudo-doc but the construction stays symmetric
# across bands (same seed, same procedure for both), so it cannot bias lower vs
# higher. The verdict's robustness to this choice is reported (never gates) via the
# PSEUDO_DOC_ROBUSTNESS_SEEDS sweep below. Value = the plan date 2026-07-21, frozen
# once so the committed evidence JSON is reproducible.
PSEUDO_DOC_SHUFFLE_SEED = 20260721

# A small FIXED set of alternate seeds used ONLY to report a seed-sensitivity panel
# (min/max gap across seeds) so a reader can see the primary-seed verdict is not a
# shuffle artifact. Informational — the gated verdict always uses
# PSEUDO_DOC_SHUFFLE_SEED. Values are arbitrary fixed integers, not tuned.
PSEUDO_DOC_ROBUSTNESS_SEEDS = (1, 7, 42, 101, 2718)

# Target eligible-paragraph count that ends a pseudo-doc. Derived, NOT tuned: it is
# exactly the detector's own MIN_PARAGRAPHS floor — a pseudo-doc is "complete" the
# moment it carries enough eligible (>=MIN_WORDS_PER_PARAGRAPH) paragraphs for
# detect_outliers to run at all. Using the detector's floor (not a larger number)
# maximizes the number of pseudo-docs — and thus statistical power — on a short-essay
# corpus, while guaranteeing every pseudo-doc passes the detector's eligibility gate.
PSEUDO_DOC_TARGET_ELIGIBLE_PARAGRAPHS = MIN_PARAGRAPHS


def summarize_group(outlier_counts: list[int], eligibles: list[bool]) -> dict:
    """Pure aggregation: per-group flag rates from per-essay outlier counts.

    outlier_counts[i] = number of flagged paragraphs in essay i;
    eligibles[i] = essay i had >= MIN_PARAGRAPHS paragraphs (short docs are
    structurally unflaggable — excluding them keeps the rate honest).
    """
    n = len(outlier_counts)
    elig_counts = [c for c, e in zip(outlier_counts, eligibles) if e]
    n_elig = len(elig_counts)
    # Eligibility rate is calibration observability: a large eligibility skew between
    # groups (e.g. lower-proficiency essays mostly too short to measure) is itself
    # worth the owner's eye even when the flag-rate gap passes.
    eligibility_rate = round(n_elig / n * 100, 2) if n else None
    if n_elig == 0:
        return {
            "n_essays": n,
            "n_eligible": 0,
            "eligibility_rate_pct": eligibility_rate,
            "flagged_essay_rate_pct": None,
            "mean_outliers_per_eligible_essay": None,
        }
    flagged = sum(1 for c in elig_counts if c > 0)
    return {
        "n_essays": n,
        "n_eligible": n_elig,
        "eligibility_rate_pct": eligibility_rate,
        "flagged_essay_rate_pct": round(flagged / n_elig * 100, 2),
        "mean_outliers_per_eligible_essay": round(sum(elig_counts) / n_elig, 3),
    }


def flag_rate_gap_pts(groups: dict) -> float | None:
    """(lower − higher) flagged-essay-rate gap in pts; None if either is unmeasurable.

    Group keys come from fpr_subgroup_gate._proficiency_groups (directory names
    containing 'proficiency'); match by 'lower'/'higher' substring like the gate does.
    """
    lower = higher = None
    for name, s in groups.items():
        rate = s.get("flagged_essay_rate_pct")
        if "lower" in name.lower():
            lower = rate
        elif "higher" in name.lower():
            higher = rate
    if lower is None or higher is None:
        return None
    return round(lower - higher, 2)


def groups_measurable(groups: dict) -> bool:
    """True only if BOTH proficiency groups have >= MIN_ELIGIBLE_PER_GROUP eligibles."""
    seen = {"lower": False, "higher": False}
    for name, s in groups.items():
        for key in seen:
            if key in name.lower():
                seen[key] = s.get("n_eligible", 0) >= MIN_ELIGIBLE_PER_GROUP
    return all(seen.values())


def compute_verdict(gap: float | None, measurable: bool, saturated: bool = False) -> str:
    """Single source of truth for the PASS/FAIL/UNMEASURABLE/PASS_SATURATED decision.

    Mirrors main()'s exit-code logic exactly so the printed verdict, the exit
    code, and the persisted JSON can never drift apart.

    `saturated` (default False — essay mode never computes it) marks that BOTH
    proficiency bands' binary flag rate is pinned at/near 100% (see
    `_saturation_note`): at that ceiling the binary gap is structurally forced
    toward 0 regardless of any real proficiency-specific effect, so a plain PASS
    would overstate the measurement. Only a would-be PASS is downgraded to
    PASS_SATURATED — FAIL and UNMEASURABLE are never rescued by saturation.
    """
    if gap is None or not measurable:
        return "UNMEASURABLE"
    if gap > MAX_FLAG_RATE_GAP_PTS:
        return "FAIL"
    return "PASS_SATURATED" if saturated else "PASS"


def _essay_outliers(text: str) -> tuple[int, bool]:
    """(flagged-paragraph count, eligible?) for one essay."""
    fps = extract_fingerprints(text)
    eligible = len(fps) >= MIN_PARAGRAPHS
    if not eligible:
        return 0, False
    return len(detect_outliers(fps)), True


# ---------------------------------------------------------------------------
# concat mode: symmetric synthetic pseudo-document construction + aggregation.
# ---------------------------------------------------------------------------

def _count_eligible_paragraphs(text: str) -> int:
    """Number of paragraphs the detector would actually consider — i.e. with
    word_count >= MIN_WORDS_PER_PARAGRAPH. This mirrors detect_outliers' own
    eligibility filter exactly, so a pseudo-doc built to N eligible paragraphs is
    guaranteed to clear the detector's len(eligible) >= MIN_PARAGRAPHS gate."""
    return sum(
        1 for fp in extract_fingerprints(text) if fp.word_count >= MIN_WORDS_PER_PARAGRAPH
    )


def build_pseudo_docs(
    essay_texts: list[str],
    eligible_count_fn,
    seed: int = PSEUDO_DOC_SHUFFLE_SEED,
    target_eligible: int = PSEUDO_DOC_TARGET_ELIGIBLE_PARAGRAPHS,
    join_with: str = "\n\n",
) -> list[dict]:
    """Deterministically concatenate same-band essays into eligible pseudo-documents.

    Procedure (IDENTICAL for every band — the function never sees a band label, only
    a list of texts, which is what makes the A/B symmetric by construction):
      1. Shuffle the essays with a fixed seed (decorrelates any author/topic
         clustering in the sorted file order without introducing run-to-run noise).
      2. Greedily accumulate WHOLE essays, summing their eligible-paragraph counts,
         and close a pseudo-doc the moment the running total reaches `target_eligible`.
      3. A trailing accumulator that never reaches `target_eligible` is DROPPED, never
         padded into a sub-eligible pseudo-doc (honest: no fabricated eligibility).

    `eligible_count_fn(text) -> int` is injected so the pure construction logic is
    unit-testable without the stylometry stack; production passes
    `_count_eligible_paragraphs`. Returns one dict per pseudo-doc with the joined
    text plus the observability fields that expose the K-asymmetry confound.
    """
    order = list(range(len(essay_texts)))
    random.Random(seed).shuffle(order)

    docs: list[dict] = []
    acc_texts: list[str] = []
    acc_eligible = 0
    for i in order:
        text = essay_texts[i]
        acc_texts.append(text)
        acc_eligible += eligible_count_fn(text)
        if acc_eligible >= target_eligible:
            docs.append(
                {
                    "text": join_with.join(acc_texts),
                    "n_essays": len(acc_texts),
                    "n_eligible_paragraphs": acc_eligible,
                }
            )
            acc_texts = []
            acc_eligible = 0
    # Trailing acc_texts with acc_eligible < target_eligible is intentionally dropped.
    return docs


def summarize_concat_group(
    outlier_counts: list[int],
    essays_per_doc: list[int],
    eligible_paras_per_doc: list[int],
) -> dict:
    """Per-band concat summary. Reuses summarize_group for the flag-rate math (every
    pseudo-doc is eligible by construction, so eligibles is all-True) and layers on
    the observability that makes the multi-author-construction confound visible.

    Emits BOTH `flagged_pseudo_doc_rate_pct` (mode-specific, self-documenting name)
    and `flagged_essay_rate_pct` (the key flag_rate_gap_pts/groups_measurable read),
    so the SAME verdict pipeline drives essay and concat modes with no fork."""
    n = len(outlier_counts)
    base = summarize_group(outlier_counts, [True] * n)
    mean_essays = round(sum(essays_per_doc) / n, 3) if n else None
    mean_paras = round(sum(eligible_paras_per_doc) / n, 3) if n else None
    return {
        "n_pseudo_docs": n,
        "n_eligible": base["n_eligible"],  # == n by construction
        "eligibility_rate_pct": base["eligibility_rate_pct"],
        "flagged_pseudo_doc_rate_pct": base["flagged_essay_rate_pct"],
        # Alias consumed by the shared gap/measurability/verdict helpers.
        "flagged_essay_rate_pct": base["flagged_essay_rate_pct"],
        "mean_outliers_per_eligible_essay": base["mean_outliers_per_eligible_essay"],
        "mean_essays_per_pseudo_doc": mean_essays,
        "mean_eligible_paras_per_pseudo_doc": mean_paras,
        "min_essays_per_pseudo_doc": min(essays_per_doc) if essays_per_doc else None,
        "max_essays_per_pseudo_doc": max(essays_per_doc) if essays_per_doc else None,
    }


def _concat_group_summary(essay_texts: list[str], seed: int) -> dict:
    """Build pseudo-docs from one band's essays and summarize their outlier flags."""
    docs = build_pseudo_docs(essay_texts, _count_eligible_paragraphs, seed=seed)
    outlier_counts, essays_per, paras_per = [], [], []
    for d in docs:
        fps = extract_fingerprints(d["text"])
        outlier_counts.append(len(detect_outliers(fps)))
        essays_per.append(d["n_essays"])
        paras_per.append(d["n_eligible_paragraphs"])
    return summarize_concat_group(outlier_counts, essays_per, paras_per)


def _read_texts(files: list[str], limit: int | None) -> list[str]:
    use = files[:limit] if limit else files
    return [Path(fp).read_text(encoding="utf-8", errors="ignore") for fp in use]


def _seed_robustness(band_files: dict, limit: int | None) -> dict:
    """Informational (never gates): recompute the lower−higher gap across a fixed
    set of alternate seeds so a reader can see the primary-seed verdict isn't a
    shuffle artifact. Returns per-seed gaps + their min/max range."""
    per_seed: dict[str, float | None] = {}
    for seed in PSEUDO_DOC_ROBUSTNESS_SEEDS:
        groups = {}
        for gname, files in sorted(band_files.items()):
            texts = _read_texts(files, limit)
            groups[gname] = _concat_group_summary(texts, seed)
        per_seed[str(seed)] = flag_rate_gap_pts(groups)
    gaps = [g for g in per_seed.values() if g is not None]
    return {
        "seeds": list(PSEUDO_DOC_ROBUSTNESS_SEEDS),
        "gap_per_seed": per_seed,
        "gap_min": round(min(gaps), 2) if gaps else None,
        "gap_max": round(max(gaps), 2) if gaps else None,
    }


# allow-hardcode: human-authored validity-caveat prose surfaced verbatim in the
# output JSON — reader documentation, NOT a scoring/matching oracle; no code branches
# on its content (it is a single opaque string, never parsed or compared).
_CONCAT_CAVEAT = (
    "concat_mode measures detector behaviour on MULTI-AUTHOR pseudo-documents built "
    "by concatenating short same-band essays until each has MIN_PARAGRAPHS eligible "
    "paragraphs. Every pseudo-doc contains genuine author boundaries, so these flag "
    "rates are an UPPER BOUND on style-break flagging, NOT the rate a real "
    "single-author ESL essay would see. The load-bearing quantity is the "
    "lower-minus-higher GAP under identical construction, read alongside "
    "mean_essays_per_pseudo_doc (a band packing more source essays per pseudo-doc has "
    "more author boundaries — a confound correlated with, but not identical to, prose "
    "style). Construction is deterministic (fixed-seed shuffle); seed_robustness "
    "shows the gap is not a shuffle artifact."
)


def _saturation_note(groups: dict) -> dict:
    """Surface a binary-flag-rate CEILING effect honestly instead of letting a clean
    0-pt gap hide it. When multi-author pseudo-docs almost all trip the detector,
    BOTH bands pin near 100% and the binary gap loses sensitivity — a 0-pt gap then
    means 'no proficiency-specific over-flagging AT this saturated operating point',
    which is real but weaker than a 0-pt gap at mid-range rates. The non-saturating
    companion signal is mean_outliers_per_eligible_essay (already per group). Computed
    from the data (not a hardcoded verdict): both-bands-at-100% => saturated=True."""
    rates = [
        s.get("flagged_essay_rate_pct")
        for name, s in groups.items()
        if ("lower" in name.lower() or "higher" in name.lower())
    ]
    both_full = bool(rates) and all(r == 100.0 for r in rates)
    return {
        "saturated": both_full,
        "band_flag_rates_pct": {n: s.get("flagged_essay_rate_pct") for n, s in groups.items()},
        "non_saturating_companion": "mean_outliers_per_eligible_essay (per group)",
    }


def measure_concat(corpus: str | None, limit: int | None) -> dict:
    """Concat-mode measurement. Same exit-code contract and verdict pipeline as
    essay-mode `measure`, but samples are symmetric multi-author pseudo-docs."""
    from calibration.fpr_subgroup_gate import DEFAULT_CORPUS, _ai_texts, _proficiency_groups

    if corpus is None:
        corpus = DEFAULT_CORPUS
    groups_files = _proficiency_groups(corpus)
    if not any(groups_files.values()):
        print(f"No SCoCESLE essays found under {corpus!r} — pass --corpus.", file=sys.stderr)
        sys.exit(2)

    result: dict = {
        "mode": "concat",
        "corpus": corpus,
        "limit": limit,
        "shuffle_seed": PSEUDO_DOC_SHUFFLE_SEED,
        "target_eligible_paragraphs": PSEUDO_DOC_TARGET_ELIGIBLE_PARAGRAPHS,
        "caveat": _CONCAT_CAVEAT,
        "groups": {},
    }
    for gname, files in sorted(groups_files.items()):
        texts = _read_texts(files, limit)
        result["groups"][gname] = _concat_group_summary(texts, PSEUDO_DOC_SHUFFLE_SEED)

    # AI-cases control, built the SAME multi-source way — a reference for the
    # multi-author upper bound (known source boundaries): concatenated AI essays
    # SHOULD flag, so this is not a "low is good" control like essay mode's, it
    # anchors what "definitely multi-source" looks like for this detector.
    ai_texts = _ai_texts()
    result["ai_cases_control"] = _concat_group_summary(ai_texts, PSEUDO_DOC_SHUFFLE_SEED)

    saturation = _saturation_note(result["groups"])
    gap = flag_rate_gap_pts(result["groups"])
    measurable = gap is not None and groups_measurable(result["groups"])
    result["flag_rate_gap_pts"] = gap
    result["max_flag_rate_gap_pts"] = MAX_FLAG_RATE_GAP_PTS
    result["measurable"] = measurable
    result["verdict"] = compute_verdict(gap, measurable, saturated=saturation["saturated"])
    result["binary_flag_rate_saturation"] = saturation
    result["seed_robustness"] = _seed_robustness(groups_files, limit)
    return result


def measure(corpus: str | None, limit: int | None) -> dict:
    # Imported lazily: fpr_subgroup_gate pulls the full DetectionRunner ML stack at
    # import time, which this script otherwise never needs. DEFAULT_CORPUS is
    # resolved here (not in main()) for the same reason — no gate import outside
    # this function.
    from calibration.fpr_subgroup_gate import DEFAULT_CORPUS, _ai_texts, _proficiency_groups

    if corpus is None:
        corpus = DEFAULT_CORPUS
    groups_files = _proficiency_groups(corpus)
    if not any(groups_files.values()):
        print(f"No SCoCESLE essays found under {corpus!r} — pass --corpus.", file=sys.stderr)
        sys.exit(2)

    result: dict = {"corpus": corpus, "limit": limit, "groups": {}}
    for gname, files in sorted(groups_files.items()):
        use = files[:limit] if limit else files
        counts, eligs = [], []
        for fp in use:
            c, e = _essay_outliers(Path(fp).read_text(encoding="utf-8", errors="ignore"))
            counts.append(c)
            eligs.append(e)
        result["groups"][gname] = summarize_group(counts, eligs)

    # Informational control (never gates): fully-AI essays are uniform in style, so a
    # LOW flag rate here is the EXPECTED behavior — this detector finds within-doc
    # style breaks, it is not an AI detector.
    ai_texts = _ai_texts()
    ai_use = ai_texts[:limit] if limit else ai_texts
    ai_counts, ai_eligs = [], []
    for t in ai_use:
        c, e = _essay_outliers(t)
        ai_counts.append(c)
        ai_eligs.append(e)
    result["ai_cases_control"] = summarize_group(ai_counts, ai_eligs)

    gap = flag_rate_gap_pts(result["groups"])
    measurable = gap is not None and groups_measurable(result["groups"])
    result["flag_rate_gap_pts"] = gap
    result["max_flag_rate_gap_pts"] = MAX_FLAG_RATE_GAP_PTS
    result["measurable"] = measurable
    result["verdict"] = compute_verdict(gap, measurable)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=None,
                    help="defaults to fpr_subgroup_gate.DEFAULT_CORPUS (resolved lazily)")
    ap.add_argument("--limit", type=int, default=None, help="essays per group (smoke)")
    # allow-hardcode: argparse --help strings are user-facing documentation, not a
    # scoring/matching list; no code branches on their content.
    ap.add_argument("--mode", choices=("essay", "concat"), default="essay",
                    help="essay (DEFAULT, one real essay = one sample; byte-identical "
                         "to the original gate) or concat (symmetric multi-author "
                         "pseudo-docs — restores power on short-essay corpora; reports "
                         "an UPPER BOUND on style-break flagging, read the lower-higher "
                         "gap, not the absolute rate)")
    ap.add_argument("--out", default=None, help="write full JSON result here")
    args = ap.parse_args()

    res = measure_concat(args.corpus, args.limit) if args.mode == "concat" \
        else measure(args.corpus, args.limit)

    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")

    gap = res["flag_rate_gap_pts"]
    verdict = res["verdict"]
    # Mode-specific noun keeps the message honest: essays vs multi-author pseudo-docs.
    sample = "pseudo-docs" if args.mode == "concat" else "essays"
    if verdict == "UNMEASURABLE":
        print(
            f"VERDICT: UNMEASURABLE [{args.mode}] — a proficiency group is missing or "
            f"has fewer than {MIN_ELIGIBLE_PER_GROUP} eligible {sample}; the gap would "
            f"be noise.",
            file=sys.stderr,
        )
        sys.exit(1)
    if verdict == "FAIL":
        print(
            f"VERDICT: FAIL [{args.mode}] — lower-proficiency {sample} flagged "
            f"{gap:+.2f} pts more than higher (limit {MAX_FLAG_RATE_GAP_PTS}). "
            f"Do NOT enable DRAFTPROOF_CONSISTENCY.",
            file=sys.stderr,
        )
        sys.exit(1)
    if verdict == "PASS_SATURATED":
        print(
            f"VERDICT: PASS_SATURATED [{args.mode}] — flag-rate gap {gap:+.2f} pts within "
            f"{MAX_FLAG_RATE_GAP_PTS}, but both bands' binary flag rate is saturated "
            f"(near/at 100%); the gap is structurally forced toward 0 at that ceiling — "
            f"see binary_flag_rate_saturation.non_saturating_companion."
        )
        return
    print(
        f"VERDICT: PASS [{args.mode}] — flag-rate gap {gap:+.2f} pts within "
        f"{MAX_FLAG_RATE_GAP_PTS}."
    )


if __name__ == "__main__":
    main()
