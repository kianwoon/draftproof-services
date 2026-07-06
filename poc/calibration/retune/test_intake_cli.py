import json
from pathlib import Path
from poc.calibration.retune import intake

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
