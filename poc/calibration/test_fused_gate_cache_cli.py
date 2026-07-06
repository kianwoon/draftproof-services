"""CLI test for the --cache flag on v7_fused_gate_run.py — no Modal calls."""
from __future__ import annotations
from pathlib import Path

from poc.calibration import v7_fused_gate_run as fg
from poc.calibration.retune import deepscan_cache


def test_argparser_exposes_cache_flag():
    parser = fg._build_arg_parser()
    args = parser.parse_args([])
    assert hasattr(args, "cache")


def test_cache_default_matches_persistent_default():
    parser = fg._build_arg_parser()
    args = parser.parse_args([])
    assert Path(args.cache) == deepscan_cache.DEFAULT_CACHE


def test_default_cache_resolves_outside_any_staging_path():
    assert "staging" not in str(deepscan_cache.DEFAULT_CACHE).split("/")


def test_resolve_paths_carries_cache(tmp_path):
    cache = tmp_path / "cache.jsonl"
    parser = fg._build_arg_parser()
    args = parser.parse_args(["--cache", str(cache)])
    resolved = fg.resolve_paths(args)
    assert resolved.cache == cache


def test_existing_defaults_preserved():
    parser = fg._build_arg_parser()
    args = parser.parse_args([])
    assert Path(args.out) == fg.OUT_PATH
    assert Path(args.progress) == fg.PROGRESS
    assert str(args.corpus) == str(fg.DEFAULT_CORPUS)
    assert Path(args.weights) == (fg._POC / "detect_v7" / "weights.json")
