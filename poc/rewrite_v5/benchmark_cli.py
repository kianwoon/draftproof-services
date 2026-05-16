"""CLI for isolated V5 stack benchmarks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .experiment import run_v5_route_window_stack_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the isolated V5 route-window stack.")
    parser.add_argument("inputs", nargs="+", help="Input text files to benchmark")
    parser.add_argument("--out", default="test_output/rewrite_v5_benchmark", help="Output directory")
    parser.add_argument("--max-windows", type=int, default=8, help="Route windows to test")
    parser.add_argument("--route-variants", type=int, default=3, help="Variants per route window")
    parser.add_argument("--cleanup-rounds", type=int, default=2, help="V4 cleanup rounds after V5")
    parser.add_argument("--cleanup-clusters", type=int, default=6, help="V4 cleanup clusters per round")
    parser.add_argument("--cleanup-variants", type=int, default=2, help="V4 cleanup variants per cluster")
    args = parser.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for input_name in args.inputs:
        input_path = Path(input_name)
        stem = input_path.stem.replace(" ", "_")
        output_dir = root / stem
        result = run_v5_route_window_stack_experiment(
            input_text=input_path.read_text(),
            output_dir=output_dir,
            max_windows=args.max_windows,
            route_variant_count=args.route_variants,
            cleanup_rounds=args.cleanup_rounds,
            cleanup_max_clusters=args.cleanup_clusters,
            cleanup_variant_count=args.cleanup_variants,
            model=os.environ.get("DRAFTPROOF_REWRITE_V4_MODEL") or os.environ.get("LLM_MODEL"),
        )
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        rows.append({
            "input": str(input_path),
            "output_dir": str(output_dir),
            "stage": result.get("stage"),
            "goal": result.get("goal"),
            **summary,
        })

    payload = {
        "inputs": [str(Path(item)) for item in args.inputs],
        "config": {
            "max_windows": args.max_windows,
            "route_variants": args.route_variants,
            "cleanup_rounds": args.cleanup_rounds,
            "cleanup_clusters": args.cleanup_clusters,
            "cleanup_variants": args.cleanup_variants,
            "model": os.environ.get("DRAFTPROOF_REWRITE_V4_MODEL") or os.environ.get("LLM_MODEL"),
        },
        "results": rows,
    }
    (root / "benchmark_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
