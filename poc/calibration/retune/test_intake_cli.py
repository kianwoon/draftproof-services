import json
from pathlib import Path
import pytest
from poc.calibration.retune import intake, manifest

def test_write_manifest_only(tmp_path):
    ai = tmp_path / "authorship_cases"; ai.mkdir()
    (ai / "ai_x_00.json").write_text(json.dumps(
        {"case_id": "ai_x_00", "authorship": "ai", "source": "openai/gpt-5-mini",
         "family": "gpt-5", "text": "word " * 150}))
    mpath = tmp_path / "manifest.json"
    n = intake.write_manifest_only(ai, None, mpath, now_iso="2026-07-06T00:00:00Z")
    assert n == 1
    m = json.loads(mpath.read_text())
    assert m["rows"][0]["family"] == "gpt-5"
    assert m["version"] == "2026-07-06T00:00:00Z"

def test_manifest_outside_corpus_with_local_only_raises(tmp_path):
    """License guard: manifest with local_only rows outside corpus/ must raise."""
    ai = tmp_path / "ai_cases"; ai.mkdir()
    (ai / "ai_x_00.json").write_text(json.dumps(
        {"case_id": "ai_x_00", "authorship": "ai", "source": "openai/gpt-5-mini",
         "family": "gpt-5", "text": "word " * 150}))

    scocesle = tmp_path / "scocesle"; scocesle.mkdir()
    prof = scocesle / "proficiency_low"; prof.mkdir()
    (prof / "essay_001.txt").write_text("This is a human student essay about learning.")

    # manifest_path is outside the gitignored corpus/ dir
    mpath = tmp_path / "external" / "manifest.json"

    with pytest.raises(manifest.LicenseError):
        intake.write_manifest_only(ai, scocesle, mpath, now_iso="2026-07-06T00:00:00Z")

    # Verify manifest file was NOT written
    assert not mpath.exists()

def test_manifest_ai_only_outside_corpus_ok(tmp_path):
    """AI-only manifest (no local_only rows) writes successfully outside corpus/."""
    ai = tmp_path / "ai_cases"; ai.mkdir()
    (ai / "ai_x_00.json").write_text(json.dumps(
        {"case_id": "ai_x_00", "authorship": "ai", "source": "openai/gpt-5-mini",
         "family": "gpt-5", "text": "word " * 150}))

    # manifest_path is outside the gitignored corpus/ dir
    mpath = tmp_path / "external" / "manifest.json"

    n = intake.write_manifest_only(ai, None, mpath, now_iso="2026-07-06T00:00:00Z")

    assert n >= 1
    assert mpath.exists()
    m = json.loads(mpath.read_text())
    assert all(r.get("license") != "local_only" for r in m["rows"])
