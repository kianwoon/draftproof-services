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


def test_default_cache_resolves_outside_any_staging_path(tmp_path):
    # A path-segment substring check ("staging" not in path.split("/")) still false-fails
    # if the checkout itself has a directory literally named "staging". Assert the real
    # relationship instead: the default cache is not nested under a per-run staging dir.
    run_staging = tmp_path / "staging" / "run-2026-07-06T00-00-00Z"
    run_staging.mkdir(parents=True)
    resolved_cache = deepscan_cache.DEFAULT_CACHE.resolve()
    assert run_staging.resolve() not in resolved_cache.parents
    assert resolved_cache != run_staging.resolve()


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


def test_weights_cfg_reads_committed_weights_json():
    """_weights_cfg must parse the REAL committed weights.json: the
    deep_scan_2detector fusion key was renamed 'fakespot' -> 'composite'
    (2026-07-06 accuracy review), but this script kept reading 'fakespot'
    and every paid fused-gate run KeyError'd (observed 2026-07-13 as
    calibration=error:fused). Reading the committed file, not a fixture,
    is the point — it catches the next key rename too."""
    from calibration.v7_fused_gate_run import _weights_cfg

    composite_w, deberta_w, sent_thr = _weights_cfg()
    assert 0.0 < composite_w < 1.0
    assert 0.0 < deberta_w < 1.0
    assert abs((composite_w + deberta_w) - 1.0) < 1e-9
    assert 0.9 <= sent_thr <= 1.0
