"""CLI for V5 length-preserved cluster replacement experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .cluster_mass import run_v5_cluster_mass_replacement_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark V5 cluster-mass replacement.")
    parser.add_argument("inputs", nargs="+", help="Input text files to benchmark")
    parser.add_argument("--out", default="test_output/v5_cluster_mass", help="Output directory")
    parser.add_argument("--max-clusters", type=int, default=5, help="Scanner-ranked clusters to replace")
    parser.add_argument("--variants", type=int, default=3, help="Variants per cluster")
    parser.add_argument("--min-word-ratio", type=float, default=0.90, help="Minimum replacement/source word ratio")
    parser.add_argument("--fallback-min-word-ratio", type=float, default=0.75, help="Fallback minimum ratio when all variants miss the primary floor")
    parser.add_argument("--max-word-ratio", type=float, default=1.50, help="Maximum replacement/source word ratio")
    args = parser.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for input_name in args.inputs:
        input_path = Path(input_name)
        output_dir = root / input_path.stem.replace(" ", "_")
        result = run_v5_cluster_mass_replacement_experiment(
            input_text=input_path.read_text(),
            output_dir=output_dir,
            max_clusters=args.max_clusters,
            variant_count=args.variants,
            min_word_ratio=args.min_word_ratio,
            fallback_min_word_ratio=args.fallback_min_word_ratio,
            max_word_ratio=args.max_word_ratio,
            model=os.environ.get("DRAFTPROOF_REWRITE_V5_MODEL")
            or os.environ.get("DRAFTPROOF_REWRITE_V4_MODEL")
            or os.environ.get("LLM_MODEL"),
        )
        baseline = result.get("baseline_scores") if isinstance(result.get("baseline_scores"), dict) else {}
        final = result.get("final_scores") if isinstance(result.get("final_scores"), dict) else {}
        best = result.get("best_scored_candidate") if isinstance(result.get("best_scored_candidate"), dict) else {}
        rows.append({
            "input": str(input_path),
            "output_dir": str(output_dir),
            "best_candidate_id": best.get("candidate_id"),
            "cluster_count": best.get("cluster_count"),
            "scores_before": _score_brief(baseline),
            "scores_after": _score_brief(final),
            "deltas": {
                "ai_delta": final.get("ai_delta"),
                "topk_delta": final.get("topk_delta"),
                "external_delta": final.get("external_delta"),
                "rank_delta": final.get("rank_delta"),
                "unsafe_cluster_count_delta": final.get("unsafe_cluster_count_delta"),
            },
        })

    payload = {
        "inputs": [str(Path(item)) for item in args.inputs],
        "config": {
            "max_clusters": args.max_clusters,
            "variants": args.variants,
            "min_word_ratio": args.min_word_ratio,
            "fallback_min_word_ratio": args.fallback_min_word_ratio,
            "max_word_ratio": args.max_word_ratio,
            "model": os.environ.get("DRAFTPROOF_REWRITE_V5_MODEL")
            or os.environ.get("DRAFTPROOF_REWRITE_V4_MODEL")
            or os.environ.get("LLM_MODEL"),
        },
        "results": rows,
    }
    (root / "cluster_mass_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _score_brief(scores: dict[str, Any]) -> dict[str, Any]:
    keys = ("ai", "topk", "external", "rank", "risky_window_count", "unsafe_cluster_count")
    return {key: scores.get(key) for key in keys}


if __name__ == "__main__":
    main()
