"""CLI parameterization tests for v7_fused_gate_run.py — no Modal calls."""
from __future__ import annotations
from pathlib import Path

from poc.calibration import v7_fused_gate_run as fg


def test_argparser_exposes_flags():
    parser = fg._build_arg_parser()
    args = parser.parse_args([])
    assert hasattr(args, "out")
    assert hasattr(args, "progress")
    assert hasattr(args, "corpus")
    assert hasattr(args, "weights")


def test_defaults_equal_today_constants():
    parser = fg._build_arg_parser()
    args = parser.parse_args([])
    assert Path(args.out) == fg.OUT_PATH
    assert Path(args.progress) == fg.PROGRESS
    assert str(args.corpus) == str(fg.DEFAULT_CORPUS)
    assert Path(args.weights) == (fg._POC / "detect_v7" / "weights.json")


def test_resolve_paths_uses_tmp_overrides(tmp_path):
    out = tmp_path / "out.json"
    progress = tmp_path / "progress.jsonl"
    corpus = tmp_path / "corpus"
    weights = tmp_path / "weights.json"
    parser = fg._build_arg_parser()
    args = parser.parse_args([
        "--out", str(out), "--progress", str(progress),
        "--corpus", str(corpus), "--weights", str(weights),
    ])
    resolved = fg.resolve_paths(args)
    assert resolved.out == out
    assert resolved.progress == progress
    assert str(resolved.corpus) == str(corpus)
    assert resolved.weights == weights
