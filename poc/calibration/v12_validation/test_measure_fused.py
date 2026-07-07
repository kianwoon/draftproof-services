"""Cache-wrapper behavior: hit serves from cache without calling Modal;
miss calls through and appends ONLY calibrated responses."""
import json
from pathlib import Path

from calibration.v12_validation import measure as m
from calibration.retune import deepscan_cache


def _fake_modal(calls: list, response: dict):
    def fake(sentences):
        calls.append(list(sentences))
        return response
    return fake


def test_cache_hit_skips_modal(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.jsonl"
    sentences = ["First sentence.", "Second sentence."]
    key = deepscan_cache.content_key("\n".join(sentences), deepscan_cache.checkpoint_tag())
    deepscan_cache.append(cache_path, key, [0.1, 0.9995])

    calls: list = []
    import detect_v7.modal_client as mc
    monkeypatch.setattr(mc, "call_deep_scan", _fake_modal(calls, {"available": False}))
    m.install_cached_deep_scan(cache_path)
    try:
        resp = mc.call_deep_scan(sentences)
    finally:
        m.uninstall_cached_deep_scan()
    assert resp == {"available": True, "calibrated": True, "chunk_scores": [0.1, 0.9995]}
    assert calls == []  # Modal never hit


def test_cache_miss_calls_through_and_appends_calibrated(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.jsonl"
    sentences = ["Only sentence here."]
    calls: list = []
    real = {"available": True, "calibrated": True, "chunk_scores": [0.5]}
    import detect_v7.modal_client as mc
    monkeypatch.setattr(mc, "call_deep_scan", _fake_modal(calls, real))
    m.install_cached_deep_scan(cache_path)
    try:
        resp = mc.call_deep_scan(sentences)
    finally:
        m.uninstall_cached_deep_scan()
    assert resp == real and len(calls) == 1
    key = deepscan_cache.content_key("\n".join(sentences), deepscan_cache.checkpoint_tag())
    assert deepscan_cache.load_cache(cache_path)[key] == [0.5]


def test_uncalibrated_response_not_cached(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.jsonl"
    calls: list = []
    real = {"available": True, "calibrated": False, "chunk_scores": [0.5]}
    import detect_v7.modal_client as mc
    monkeypatch.setattr(mc, "call_deep_scan", _fake_modal(calls, real))
    m.install_cached_deep_scan(cache_path)
    try:
        resp = mc.call_deep_scan(["A sentence."])
    finally:
        m.uninstall_cached_deep_scan()
    assert resp == real
    assert deepscan_cache.load_cache(cache_path) == {}
