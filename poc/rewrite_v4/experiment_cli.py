"""CLI for running the isolated V4 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_v4_budget_lane_experiment, run_v4_experiment, run_v4_fast_rewrite, run_v4_iterative_rewrite, run_v4_layered_rewrite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experimental rewrite V4 on a local text file.")
    parser.add_argument("input", help="Input text file")
    parser.add_argument("--out", default="test_output/rewrite_v4_experiment", help="Output directory")
    parser.add_argument("--unit-id", action="append", default=[], help="Target unit id to include, e.g. p007. Repeatable.")
    parser.add_argument("--no-llm-normalizer", action="store_true", help="Disable LLM normalizer proposer.")
    parser.add_argument("--tutor-normalizer", action="store_true", help="Enable tutor-style normalizer proposer.")
    parser.add_argument("--enrichment-normalizer", action="store_true", help="Enable controlled-enrichment normalizer proposer.")
    parser.add_argument("--variants", type=int, default=3, help="Variants per repair brief.")
    parser.add_argument("--iterative", action="store_true", help="Apply safe candidates iteratively and output a rewritten document.")
    parser.add_argument("--fast", action="store_true", help="Run production-shaped fast iterative mode.")
    parser.add_argument("--budget-lanes", action="store_true", help="Run Layer 3 budget-lane optimizer.")
    parser.add_argument("--layered", action="store_true", help="Run full V4 sequence with Layer 3 after normalized repair.")
    parser.add_argument("--max-clusters", type=int, default=6, help="Maximum cluster units for cluster/layer experiments.")
    parser.add_argument("--max-rounds", type=int, default=3, help="Maximum iterative rewrite rounds.")
    parser.add_argument("--groups-per-round", type=int, default=None, help="Limit target groups tested per iterative round.")
    parser.add_argument("--stop-after-accepted", type=int, default=None, help="Stop after this many accepted iterative edits.")
    parser.add_argument("--strong-ai-delta", type=float, default=None, help="Stop after an accepted edit reaches this AI delta.")
    args = parser.parse_args()

    text = Path(args.input).read_text()
    if args.fast:
        result = run_v4_fast_rewrite(
            input_text=text,
            output_dir=args.out,
            unit_ids=set(args.unit_id) if args.unit_id else None,
        )
        print(json.dumps({
            "summary": result.get("summary"),
            "accepted": result.get("accepted"),
            "config": result.get("config"),
            "rewritten_document_path": str(Path(args.out) / "v4_rewritten_document.txt"),
            "output_dir": args.out,
        }, ensure_ascii=False, indent=2))
    elif args.budget_lanes:
        result = run_v4_budget_lane_experiment(
            input_text=text,
            output_dir=args.out,
            max_rounds=args.max_rounds,
            max_clusters=args.max_clusters,
            variant_count=args.variants,
        )
        print(json.dumps({
            "original_scores": result.get("original_scores"),
            "final_scores": result.get("final_scores"),
            "accepted": result.get("accepted"),
            "goal": result.get("goal"),
            "rewritten_document_path": str(Path(args.out) / "v4_budget_lane_rewritten_document.txt"),
            "output_dir": args.out,
        }, ensure_ascii=False, indent=2))
    elif args.layered:
        result = run_v4_layered_rewrite(
            input_text=text,
            output_dir=args.out,
            unit_ids=set(args.unit_id) if args.unit_id else None,
            include_llm_normalizer=not args.no_llm_normalizer,
            include_tutor_normalizer=args.tutor_normalizer,
            include_enrichment_normalizer=args.enrichment_normalizer,
            variant_count=args.variants,
            max_rounds=args.max_rounds,
            groups_per_round=args.groups_per_round,
            stop_after_accepted=args.stop_after_accepted,
            strong_ai_delta=args.strong_ai_delta,
            layer3_enabled=True,
            layer3_max_rounds=args.max_rounds,
            layer3_max_clusters=args.max_clusters,
            layer3_variant_count=args.variants,
        )
        print(json.dumps({
            "summary": result.get("summary"),
            "accepted": result.get("accepted"),
            "layers": result.get("layers"),
            "config": result.get("config"),
            "rewritten_document_path": str(Path(args.out) / "v4_rewritten_document.txt"),
            "output_dir": args.out,
        }, ensure_ascii=False, indent=2))
    elif args.iterative:
        result = run_v4_iterative_rewrite(
            input_text=text,
            output_dir=args.out,
            unit_ids=set(args.unit_id) if args.unit_id else None,
            include_llm_normalizer=not args.no_llm_normalizer,
            include_tutor_normalizer=args.tutor_normalizer,
            include_enrichment_normalizer=args.enrichment_normalizer,
            variant_count=args.variants,
            max_rounds=args.max_rounds,
            groups_per_round=args.groups_per_round,
            stop_after_accepted=args.stop_after_accepted,
            strong_ai_delta=args.strong_ai_delta,
        )
        print(json.dumps({
            "summary": result.get("summary"),
            "accepted": result.get("accepted"),
            "config": result.get("config"),
            "rewritten_document_path": str(Path(args.out) / "v4_rewritten_document.txt"),
            "output_dir": args.out,
        }, ensure_ascii=False, indent=2))
    else:
        result = run_v4_experiment(
            input_text=text,
            output_dir=args.out,
            unit_ids=set(args.unit_id) if args.unit_id else None,
            include_llm_normalizer=not args.no_llm_normalizer,
            include_tutor_normalizer=args.tutor_normalizer,
            include_enrichment_normalizer=args.enrichment_normalizer,
            variant_count=args.variants,
        )
        print(json.dumps({
            "baseline": result.get("baseline"),
            "best_safe_candidate": result.get("best_safe_candidate"),
            "summary_ranked": (result.get("summary_ranked") or [])[:12],
            "output_dir": args.out,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
