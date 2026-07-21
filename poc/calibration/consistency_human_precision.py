"""Genuine long single-author HUMAN precision measurement for the Phase-1
stylometric ConsistencyDetector.

WHY THIS EXISTS: consistency_esl_rates.py answers an ESL-fairness question
(does the detector over-flag lower-proficiency ESL writers relative to higher,
on multi-author pseudo-docs?) and cleared it (gap 0.0). Reviewers separately
flagged the detector as possibly hair-trigger-noisy on multi-author
concatenations (~4.4 outliers/doc; 3 flags on a 1-shift fixture — see
consistency_esl_rates_concat.json). Neither measurement answers the actual
precision question a "within-document style-outlier" detector must answer
before it can ship as advisory-only signal: **how often does it flag
paragraphs in a REAL long single-author human document, where nothing is
actually inconsistent?** That is what this script measures — over genuinely
single-author corpora, not synthetic multi-author concatenations.

NO PASS/FAIL GATE: this script reports `"verdict": "MEASUREMENT_ONLY"` always.
Inventing a precision threshold (e.g. "flagged_doc_rate_pct must be < X%") is
explicitly out of scope — that judgment call belongs to a human reviewer who
weighs the numbers below against the advisory-only, review-flag product
context. Exit codes signal MEASURABILITY, not a verdict:
  0 = at least one eligible doc was measured overall
  1 = zero eligible docs were measured overall (nothing to report)
  2 = a --source directory is missing or contains no .txt files

Usage (from poc/):
    python calibration/consistency_human_precision.py \\
        --source gutenberg=~/Downloads/human_precision_corpus/gutenberg \\
        --source raid_human=~/Downloads/human_precision_corpus/raid_human \\
        --source scocesle_eligible=~/Downloads/human_precision_corpus/scocesle_eligible \\
        --out calibration/consistency_human_precision.json

Purely deterministic and cheap: stylometry only (extract_fingerprints +
detect_outliers), no ML models, no DetectionRunner. File iteration within each
source is sorted, so results are reproducible given the same staged corpus.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Mirrors consistency_esl_rates.py's sys.path bootstrap: `detect/__init__.py`
# transitively imports `poc.predictability...`, so both poc/ (parents[1]) and
# the project root (parents[2], which makes `poc` importable) must be on
# sys.path for direct script invocation to work. pytest masks this because
# conftest.py already adds the project root for the test collector.
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

# Caveats surfaced verbatim in the output JSON so a reader interprets each
# source's rate against its actual register, not as directly comparable
# apples-to-apples. allow-hardcode: human-authored documentation prose, never
# parsed or branched on — same pattern as consistency_esl_rates._CONCAT_CAVEAT.
SOURCE_CAVEATS = {
    "gutenberg": "19th-century formal register; not modern student writing.",
    "raid_human": "modern web/benchmark domains (books, news, wiki, etc.); not academic essays.",
    "scocesle_eligible": "ESL student essays; n=5, anecdotal.",
}
DEFAULT_CAVEAT = "no caveat registered for this --source name; interpret with care."


def outlier_distribution(counts: list[int]) -> dict:
    """Bucket eligible-doc outlier counts into {0, 1, 2, 3+} -> doc count."""
    dist = {"0": 0, "1": 0, "2": 0, "3+": 0}
    for c in counts:
        key = str(c) if c <= 2 else "3+"
        dist[key] += 1
    return dist


def summarize_source(
    outlier_counts: list[int], eligibles: list[bool], paragraph_counts: list[int]
) -> dict:
    """Pure aggregation: per-source precision stats from per-doc outlier counts.

    outlier_counts[i] = number of flagged paragraphs in doc i;
    eligibles[i] = doc i had >= MIN_PARAGRAPHS eligible paragraphs;
    paragraph_counts[i] = doc i's eligible-paragraph count (only meaningful
    where eligibles[i] is True; ignored otherwise).
    """
    n = len(outlier_counts)
    elig_idx = [i for i, e in enumerate(eligibles) if e]
    n_elig = len(elig_idx)
    eligibility_rate = round(n_elig / n * 100, 2) if n else None
    if n_elig == 0:
        return {
            "n_docs": n,
            "n_eligible": 0,
            "eligibility_rate_pct": eligibility_rate,
            "flagged_doc_rate_pct": None,
            "mean_outliers_per_eligible_doc": None,
            "mean_paragraphs_per_eligible_doc": None,
            "distribution": outlier_distribution([]),
        }
    elig_counts = [outlier_counts[i] for i in elig_idx]
    elig_paras = [paragraph_counts[i] for i in elig_idx]
    flagged = sum(1 for c in elig_counts if c > 0)
    return {
        "n_docs": n,
        "n_eligible": n_elig,
        "eligibility_rate_pct": eligibility_rate,
        "flagged_doc_rate_pct": round(flagged / n_elig * 100, 2),
        "mean_outliers_per_eligible_doc": round(sum(elig_counts) / n_elig, 3),
        "mean_paragraphs_per_eligible_doc": round(sum(elig_paras) / n_elig, 3),
        "distribution": outlier_distribution(elig_counts),
    }


def combine_sources(per_source_raw: dict[str, tuple[list[int], list[bool], list[int]]]) -> dict:
    """Merge raw per-doc lists across sources, then run the same summary math."""
    all_counts: list[int] = []
    all_elig: list[bool] = []
    all_paras: list[int] = []
    for _name, (counts, eligs, paras) in sorted(per_source_raw.items()):
        all_counts += counts
        all_elig += eligs
        all_paras += paras
    return summarize_source(all_counts, all_elig, all_paras)


def _count_eligible_paragraphs(text: str) -> int:
    """Mirrors detect_outliers' own MIN_WORDS_PER_PARAGRAPH eligibility filter —
    same helper shape as consistency_esl_rates._count_eligible_paragraphs."""
    return sum(
        1 for fp in extract_fingerprints(text) if fp.word_count >= MIN_WORDS_PER_PARAGRAPH
    )


def _doc_outliers(text: str) -> tuple[int, bool, int]:
    """(flagged-paragraph count, eligible?, eligible-paragraph count) for one doc."""
    fps = extract_fingerprints(text)
    n_eligible_paras = sum(1 for fp in fps if fp.word_count >= MIN_WORDS_PER_PARAGRAPH)
    eligible = n_eligible_paras >= MIN_PARAGRAPHS
    if not eligible:
        return 0, False, n_eligible_paras
    return len(detect_outliers(fps)), True, n_eligible_paras


def _parse_source_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--source must be NAME=DIR, got {raw!r}"
        )
    name, _, path = raw.partition("=")
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(
            f"--source must be NAME=DIR with both parts non-empty, got {raw!r}"
        )
    return name, path


def measure_sources(sources: list[tuple[str, str]], limit: int | None) -> dict:
    """Run the measurement across all --source NAME=DIR pairs. Exits 2 if any
    named source directory is missing or has zero .txt files."""
    result: dict = {"sources": {}}
    per_source_raw: dict[str, tuple[list[int], list[bool], list[int]]] = {}

    for name, dir_str in sources:
        d = Path(dir_str).expanduser()
        if not d.is_dir():
            print(f"--source {name}={dir_str!r}: directory does not exist.", file=sys.stderr)
            sys.exit(2)
        files = sorted(d.glob("*.txt"))
        if limit:
            files = files[:limit]
        if not files:
            print(f"--source {name}={dir_str!r}: no .txt files found.", file=sys.stderr)
            sys.exit(2)

        counts, eligs, paras = [], [], []
        for fp in files:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            c, e, p = _doc_outliers(text)
            counts.append(c)
            eligs.append(e)
            paras.append(p)

        per_source_raw[name] = (counts, eligs, paras)
        summary = summarize_source(counts, eligs, paras)
        summary["caveat"] = SOURCE_CAVEATS.get(name, DEFAULT_CAVEAT)
        summary["dir"] = str(d)
        result["sources"][name] = summary

    result["overall"] = combine_sources(per_source_raw)
    result["verdict"] = "MEASUREMENT_ONLY"
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--source",
        action="append",
        dest="sources",
        type=_parse_source_arg,
        default=[],
        required=True,
        help="NAME=DIR, repeatable. DIR holds plain .txt docs (one doc per file).",
    )
    ap.add_argument("--limit", type=int, default=None, help="docs per source (smoke)")
    ap.add_argument("--out", default=None, help="write full JSON result here")
    ap.add_argument(
        "--note",
        action="append",
        dest="notes",
        default=[],
        help="free-text acquisition note carried into the output JSON verbatim "
             "(repeatable) — e.g. documenting a --source that was attempted but "
             "excluded because 0 docs met the eligibility threshold.",
    )
    args = ap.parse_args()

    res = measure_sources(args.sources, args.limit)
    if args.notes:
        res["acquisition_notes"] = list(args.notes)

    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")

    overall = res["overall"]
    if overall["n_eligible"] < 1:
        print(
            "VERDICT: no eligible docs were measured across any --source "
            f"(need >= {MIN_PARAGRAPHS} eligible paragraphs per doc) — nothing to report.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"VERDICT: MEASUREMENT_ONLY — {overall['n_eligible']} eligible docs overall, "
        f"flagged_doc_rate={overall['flagged_doc_rate_pct']}%, "
        f"mean_outliers_per_eligible_doc={overall['mean_outliers_per_eligible_doc']}. "
        "No pass/fail threshold is applied — this artifact informs a human review "
        "decision, not an automated gate."
    )


if __name__ == "__main__":
    main()
