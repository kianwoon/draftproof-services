"""Tests for the §12 validation-corpus builder (no network — fake chat_fn)."""
from __future__ import annotations

import json

import pytest

from calibration.retune.generators import Generator
from calibration.v12_validation.build import (
    _sample_evenly,
    build_manifest,
    generate_variants,
)


def _human(tmp_path, name, words=150):
    p = tmp_path / name
    p.write_text(("word " * words).strip())
    return str(p)


def test_sample_evenly_is_deterministic_and_bounded():
    paths = [f"e{i:03d}.txt" for i in range(100)]
    a = _sample_evenly(list(reversed(paths)), 10)
    b = _sample_evenly(paths, 10)
    assert a == b and len(a) == 10  # sorted first -> order-independent
    assert _sample_evenly(paths[:3], 10) == sorted(paths[:3])  # fewer than n -> all


def test_generate_variants_writes_polish_and_paraphrase_per_essay(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    humans = [_human(tmp_path, "essay_a.txt"), _human(tmp_path, "essay_b.txt")]
    gens = [Generator("openai/gpt-5-mini", "openrouter", "gpt-5", 1)]
    out = tmp_path / "corpus"
    calls = []

    def fake_chat(model, base_url, api_key, prompt, temperature, max_tokens=700):
        calls.append(prompt)
        return ("rewritten " * 140).strip()

    made = generate_variants(humans, out, gens, chat_fn=fake_chat)
    assert made == 4  # 2 essays x 2 variants
    files = sorted(p.name for p in out.glob("*.json"))
    assert sum("ai_assisted_polished" in f for f in files) == 2
    assert sum("ai_paraphrased" in f for f in files) == 2
    d = json.loads(next(out.glob("ai_assisted_polished*.json")).read_text())
    assert d["license"] == "scocesle_derivative_local_only"
    assert d["source_sha256"] and d["text"]
    # polish prompt forbids adding content; paraphrase requires full rewording
    assert any("do NOT add new content" in c for c in calls)
    assert any("thorough paraphrase" in c for c in calls)


def test_generate_variants_resumable_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    humans = [_human(tmp_path, "essay_a.txt")]
    gens = [Generator("openai/gpt-5-mini", "openrouter", "gpt-5", 1)]
    out = tmp_path / "corpus"
    fake = lambda *a, **k: ("rewritten " * 140).strip()  # noqa: E731
    assert generate_variants(humans, out, gens, chat_fn=fake) == 2
    assert generate_variants(humans, out, gens, chat_fn=fake) == 0  # all exist


def test_generate_variants_rejects_short_output(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    humans = [_human(tmp_path, "essay_a.txt")]
    gens = [Generator("openai/gpt-5-mini", "openrouter", "gpt-5", 1)]
    made = generate_variants(humans, tmp_path / "corpus", gens,
                             chat_fn=lambda *a, **k: "too short")
    assert made == 0


def test_manifest_counts_all_classes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    humans = [_human(tmp_path, "essay_a.txt")]
    gens = [Generator("openai/gpt-5-mini", "openrouter", "gpt-5", 1)]
    out = tmp_path / "corpus"
    generate_variants(humans, out, gens, chat_fn=lambda *a, **k: ("rewritten " * 140).strip())
    manifest = out / "manifest.json"
    n = build_manifest(humans, out, manifest, "2026-07-06T00:00:00+00:00")
    rows = json.loads(manifest.read_text())["rows"]
    assert n == len(rows)
    labels = {r["label"] for r in rows}
    assert {"student_owned", "ai_assisted_polished", "ai_paraphrased"} <= labels
    assert all(r["sha256"] for r in rows)
    assert all("text" not in r for r in rows)  # hashes only, never text
