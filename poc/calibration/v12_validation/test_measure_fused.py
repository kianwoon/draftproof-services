"""Cache-wrapper behavior: hit serves from cache without calling Modal;
miss calls through and appends ONLY responses whose checkpoint matches ours
(checkpoint identity is the validity criterion; the endpoint's "calibrated"
flag is annotation-only and not part of the cache rule)."""
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
    # calibrated True mirrors the endpoint's CURRENT live behavior (verified
    # 2026-07-21: derives True from CALIBRATED_CHECKPOINT_IDS for this
    # checkpoint) so cache hits stay faithful to what a live call would return.
    assert resp == {"available": True, "calibrated": True,
                    "checkpoint": deepscan_cache.checkpoint_tag(),
                    "chunk_scores": [0.1, 0.9995]}
    assert calls == []  # Modal never hit


def test_cache_miss_calls_through_and_appends_on_checkpoint_match(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.jsonl"
    sentences = ["Only sentence here."]
    calls: list = []
    # Realistic live response: endpoint hardcodes calibrated False but reports
    # its checkpoint — a matching checkpoint must still be cached.
    real = {"available": True, "calibrated": False,
            "checkpoint": deepscan_cache.checkpoint_tag(), "chunk_scores": [0.5]}
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


def test_checkpoint_mismatch_not_cached(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.jsonl"
    calls: list = []
    real = {"available": True, "calibrated": True,
            "checkpoint": "some/other-model", "chunk_scores": [0.5]}
    import detect_v7.modal_client as mc
    monkeypatch.setattr(mc, "call_deep_scan", _fake_modal(calls, real))
    m.install_cached_deep_scan(cache_path)
    try:
        resp = mc.call_deep_scan(["A sentence."])
    finally:
        m.uninstall_cached_deep_scan()
    assert resp == real
    assert deepscan_cache.load_cache(cache_path) == {}
