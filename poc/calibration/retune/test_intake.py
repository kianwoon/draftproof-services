import json
from poc.calibration.retune.generators import Generator
from poc.calibration.retune import intake

def _fake_chat(model, base_url, api_key, prompt, temperature):
    return " ".join(["word"] * 150)  # >120 words so it isn't skipped

def test_generate_persists_and_tags_family(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    gens = [Generator("openai/gpt-5-mini", "openrouter", "gpt-5", 1)]
    n = intake.generate_ai_essays(tmp_path, gens, ["topic one"], chat_fn=_fake_chat)
    assert n == 1
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    d = json.loads(files[0].read_text())
    assert d["authorship"] == "ai"
    assert d["source"] == "openai/gpt-5-mini"
    assert d["family"] == "gpt-5"
    assert d["words"] >= 120

def test_generate_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    gens = [Generator("openai/gpt-5-mini", "openrouter", "gpt-5", 1)]
    intake.generate_ai_essays(tmp_path, gens, ["topic one"], chat_fn=_fake_chat)
    n2 = intake.generate_ai_essays(tmp_path, gens, ["topic one"], chat_fn=_fake_chat)
    assert n2 == 0  # already exists, skipped

def test_short_output_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    gens = [Generator("openai/gpt-5-mini", "openrouter", "gpt-5", 1)]
    n = intake.generate_ai_essays(tmp_path, gens, ["t"], chat_fn=lambda *a: "too short")
    assert n == 0
