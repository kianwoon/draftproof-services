"""CLI for the V5 residual cluster comb-through experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .residual_comb import run_v5_residual_cluster_comb_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V5 residual cluster comb-through.")
    parser.add_argument("inputs", nargs="+", help="Input text files to benchmark")
    parser.add_argument("--out", default="test_output/v5_residual_comb", help="Output directory")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum residual cluster rounds")
    parser.add_argument("--variants", type=int, default=3, help="Initial variants per cluster")
    parser.add_argument("--retune-variants", type=int, default=4, help="Retune variants for near-miss clusters")
    parser.add_argument("--risky-window-cleanup-rounds", type=int, default=None, help="Post-pass risky window cleanup rounds")
    parser.add_argument("--unsafe-cluster-cleanup-rounds", type=int, default=None, help="Post-pass unsafe cluster cleanup rounds")
    parser.add_argument("--final-risky-window-cleanup-rounds", type=int, default=None, help="Final risky window cleanup rounds after unsafe cluster cleanup")
    parser.add_argument("--cleanup-variants", type=int, default=None, help="Variants for post-pass cleanup rounds")
    args = parser.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for input_name in args.inputs:
        input_path = Path(input_name)
        output_dir = root / input_path.stem.replace(" ", "_")
        result = run_v5_residual_cluster_comb_experiment(
            input_text=input_path.read_text(),
            output_dir=output_dir,
            max_rounds=args.max_rounds,
            variant_count=args.variants,
            retune_variant_count=args.retune_variants,
            model=os.environ.get("DRAFTPROOF_REWRITE_V5_MODEL")
            or os.environ.get("DRAFTPROOF_REWRITE_V4_MODEL")
            or os.environ.get("LLM_MODEL"),
            risky_window_cleanup_rounds=args.risky_window_cleanup_rounds,
            unsafe_cluster_cleanup_rounds=args.unsafe_cluster_cleanup_rounds,
            cleanup_variant_count=args.cleanup_variants,
            final_risky_window_cleanup_rounds=args.final_risky_window_cleanup_rounds,
        )
        baseline = result.get("baseline_scores") if isinstance(result.get("baseline_scores"), dict) else {}
        final = result.get("final_scores") if isinstance(result.get("final_scores"), dict) else {}
        rounds = result.get("rounds") if isinstance(result.get("rounds"), list) else []
        risky_window_cleanup_rounds = (
            result.get("risky_window_cleanup_rounds")
            if isinstance(result.get("risky_window_cleanup_rounds"), list)
            else []
        )
        unsafe_cluster_cleanup_rounds = (
            result.get("unsafe_cluster_cleanup_rounds")
            if isinstance(result.get("unsafe_cluster_cleanup_rounds"), list)
            else []
        )
        final_risky_window_cleanup_rounds = (
            result.get("final_risky_window_cleanup_rounds")
            if isinstance(result.get("final_risky_window_cleanup_rounds"), list)
            else []
        )
        all_rounds = rounds + risky_window_cleanup_rounds + unsafe_cluster_cleanup_rounds + final_risky_window_cleanup_rounds
        global_best_fallback = result.get("global_best_fallback") if isinstance(result.get("global_best_fallback"), dict) else {}
        rows.append({
            "input": str(input_path),
            "output_dir": str(output_dir),
            "accepted_rounds": sum(1 for row in all_rounds if isinstance(row, dict) and row.get("accepted")),
            "global_best_fallback_applied": bool(global_best_fallback.get("applied")),
            "global_best_fallback_reason": global_best_fallback.get("reason"),
            "accepted_core_rounds": sum(1 for row in rounds if isinstance(row, dict) and row.get("accepted")),
            "accepted_risky_window_cleanup_rounds": sum(1 for row in risky_window_cleanup_rounds if isinstance(row, dict) and row.get("accepted")),
            "accepted_unsafe_cluster_cleanup_rounds": sum(1 for row in unsafe_cluster_cleanup_rounds if isinstance(row, dict) and row.get("accepted")),
            "accepted_final_risky_window_cleanup_rounds": sum(1 for row in final_risky_window_cleanup_rounds if isinstance(row, dict) and row.get("accepted")),
            "scores_before": _score_brief(baseline),
            "scores_after": _score_brief(final),
            "deltas": {
                "ai_delta": _delta(baseline, final, "ai"),
                "topk_delta": _delta(baseline, final, "topk"),
                "external_delta": _delta(baseline, final, "external"),
                "rank_delta": _delta(baseline, final, "rank"),
                "unsafe_cluster_count_delta": _delta(baseline, final, "unsafe_cluster_count"),
            },
        })

    payload = {
        "inputs": [str(Path(item)) for item in args.inputs],
        "config": {
            "max_rounds": args.max_rounds,
            "variants": args.variants,
            "retune_variants": args.retune_variants,
            "risky_window_cleanup_rounds": args.risky_window_cleanup_rounds,
            "unsafe_cluster_cleanup_rounds": args.unsafe_cluster_cleanup_rounds,
            "final_risky_window_cleanup_rounds": args.final_risky_window_cleanup_rounds,
            "cleanup_variants": args.cleanup_variants,
            "model": os.environ.get("DRAFTPROOF_REWRITE_V5_MODEL")
            or os.environ.get("DRAFTPROOF_REWRITE_V4_MODEL")
            or os.environ.get("LLM_MODEL"),
        },
        "results": rows,
    }
    (root / "residual_comb_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _score_brief(scores: dict[str, Any]) -> dict[str, Any]:
    keys = ("ai", "topk", "external", "rank", "risky_window_count", "unsafe_cluster_count")
    return {key: scores.get(key) for key in keys}


def _delta(before: dict[str, Any], after: dict[str, Any], key: str) -> float:
    try:
        return round(float(before.get(key) or 0.0) - float(after.get(key) or 0.0), 3)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
