import json
from pathlib import Path
import pytest
from poc.calibration.retune.generators import load_generators, Generator

def _write(tmp_path, rows):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"version": "test", "generators": rows}))
    return p

def test_loads_rows(tmp_path):
    p = _write(tmp_path, [
        {"id": "openai/gpt-5-mini", "provider": "openrouter", "family": "gpt-5", "n_per_topic": 1},
    ])
    gens = load_generators(p)
    assert gens == [Generator("openai/gpt-5-mini", "openrouter", "gpt-5", 1)]

def test_missing_key_raises(tmp_path):
    p = _write(tmp_path, [{"id": "x/y", "provider": "openrouter"}])  # no family/n_per_topic
    with pytest.raises(ValueError):
        load_generators(p)

def test_unknown_provider_raises(tmp_path):
    p = _write(tmp_path, [{"id": "x/y", "provider": "mystery", "family": "f", "n_per_topic": 1}])
    with pytest.raises(ValueError):
        load_generators(p)

def test_default_path_loads_committed_models(tmp_path):
    gens = load_generators()  # committed models.json
    fams = {g.family for g in gens}
    assert "gpt-5" in fams  # gpt-5-mini row is present (closes the silent-drift gap)
