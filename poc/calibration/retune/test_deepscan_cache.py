"""Tests for the persistent content-hash deep-scan cache. Zero network — all score_fn
calls are counting fakes."""
from __future__ import annotations

import json

from poc.calibration.retune import deepscan_cache as dc


def test_content_key_deterministic():
    k1 = dc.content_key("hello world", "ckpt-a")
    k2 = dc.content_key("hello world", "ckpt-a")
    assert k1 == k2


def test_content_key_checkpoint_sensitive():
    k1 = dc.content_key("hello world", "ckpt-a")
    k2 = dc.content_key("hello world", "ckpt-b")
    assert k1 != k2


def test_default_cache_outside_staging(tmp_path):
    # A substring check on the path (e.g. "staging" not in str(path)) false-fails if
    # the repo checkout path itself happens to contain "staging". Assert the actual
    # path relationship instead: the default cache is not inside a per-run staging dir.
    run_staging = tmp_path / "staging" / "run-2026-07-06T00-00-00Z"
    run_staging.mkdir(parents=True)
    resolved_cache = dc.DEFAULT_CACHE.resolve()
    assert run_staging.resolve() not in resolved_cache.parents
    assert resolved_cache != run_staging.resolve()
    assert dc.DEFAULT_CACHE.parent.name == "cache"


def test_load_cache_missing_file_returns_empty(tmp_path):
    assert dc.load_cache(tmp_path / "nope.jsonl") == {}


def test_get_scores_caches_and_skips_rescoring(tmp_path):
    cache_path = tmp_path / "deepscan_scores.jsonl"
    cache: dict = {}
    calls = []

    def fake_score_fn(text):
        calls.append(text)
        return [0.1, 0.9, 0.5]

    scores1 = dc.get_scores("essay text", "ckpt-a", cache, cache_path, fake_score_fn)
    assert scores1 == [0.1, 0.9, 0.5]
    assert len(calls) == 1

    # second call, same text/checkpoint, in-memory cache hit — score_fn NOT called again
    scores2 = dc.get_scores("essay text", "ckpt-a", cache, cache_path, fake_score_fn)
    assert scores2 == [0.1, 0.9, 0.5]
    assert len(calls) == 1

    # reload from disk (simulating a new run/process) — still a hit
    reloaded = dc.load_cache(cache_path)
    scores3 = dc.get_scores("essay text", "ckpt-a", reloaded, cache_path, fake_score_fn)
    assert scores3 == [0.1, 0.9, 0.5]
    assert len(calls) == 1


def test_get_scores_different_checkpoint_rescoring():
    cache: dict = {}
    calls = []

    def fake_score_fn(text):
        calls.append(text)
        return [0.2, 0.8]

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        cache_path = Path(td) / "c.jsonl"
        dc.get_scores("essay text", "ckpt-a", cache, cache_path, fake_score_fn)
        dc.get_scores("essay text", "ckpt-b", cache, cache_path, fake_score_fn)
        assert len(calls) == 2


def test_proportion_threshold_independent_same_scores():
    scores = [0.1, 0.4, 0.6, 0.9]
    p_low = dc.proportion(scores, 0.3)
    p_high = dc.proportion(scores, 0.7)
    assert p_low == 0.75  # 3/4 >= 0.3
    assert p_high == 0.25  # 1/4 >= 0.7
    assert p_low != p_high


def test_proportion_none_for_empty_or_none():
    assert dc.proportion(None, 0.5) is None
    assert dc.proportion([], 0.5) is None


def test_get_scores_empty_result_not_cached(tmp_path):
    cache_path = tmp_path / "c.jsonl"
    cache: dict = {}

    def empty_score_fn(text):
        return []

    result = dc.get_scores("essay text", "ckpt-a", cache, cache_path, empty_score_fn)
    assert not result
    assert not cache_path.exists()
    assert dc.content_key("essay text", "ckpt-a") not in cache


def test_append_writes_jsonl_row(tmp_path):
    cache_path = tmp_path / "sub" / "c.jsonl"
    dc.append(cache_path, "somekey", [0.1, 0.2])
    assert cache_path.exists()
    reloaded = dc.load_cache(cache_path)
    assert reloaded["somekey"] == [0.1, 0.2]


def test_checkpoint_tag_default(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_MODAL_CHECKPOINT", raising=False)
    assert dc.checkpoint_tag() == "desklib/ai-text-detector-academic-v1.01"


def test_checkpoint_tag_env_override(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_MODAL_CHECKPOINT", "some/other-ckpt")
    assert dc.checkpoint_tag() == "some/other-ckpt"


def test_checkpoint_tag_env_override_changes_cache_key(monkeypatch):
    # A checkpoint change must produce a different cache key, else stale scores
    # from an old checkpoint would silently be served as if from the new one.
    monkeypatch.delenv("DRAFTPROOF_MODAL_CHECKPOINT", raising=False)
    default_tag = dc.checkpoint_tag()
    default_key = dc.content_key("essay text", default_tag)

    monkeypatch.setenv("DRAFTPROOF_MODAL_CHECKPOINT", "some/other-ckpt")
    overridden_tag = dc.checkpoint_tag()
    overridden_key = dc.content_key("essay text", overridden_tag)

    assert overridden_tag == "some/other-ckpt"
    assert overridden_key != default_key


def test_load_cache_skips_torn_last_line(tmp_path):
    cache_path = tmp_path / "torn.jsonl"
    good_row = json.dumps({"key": "abc123", "scores": [0.1, 0.2]})
    torn_row = '{"key": "def456", "scores": [0.3, 0.'  # truncated mid-write
    cache_path.write_text(good_row + "\n" + torn_row)

    loaded = dc.load_cache(cache_path)

    assert loaded == {"abc123": [0.1, 0.2]}
    assert "def456" not in loaded
