import json
import pytest
from poc.calibration.retune import manifest

def _ai_case(tmp, cid, source, family, authorship="ai", text="word " * 150):
    p = tmp / f"{cid}.json"
    p.write_text(json.dumps({"case_id": cid, "authorship": authorship, "source": source,
                             "family": family, "text": text}))
    return p

def test_ai_rows_labelled_and_family(tmp_path):
    ai = tmp_path / "ai"; ai.mkdir()
    _ai_case(ai, "ai_x_00", "openai/gpt-5-mini", "gpt-5")
    m = manifest.build_manifest(ai, None, now_iso="2026-07-06T00:00:00Z")
    row = m["rows"][0]
    assert row["label"] == "ai"
    assert row["family"] == "gpt-5"
    assert row["model_id"] == "openai/gpt-5-mini"
    assert row["license"] == "redistributable"
    assert row["split"] in ("cal", "test")
    assert len(row["sha256"]) == 64

def test_gutenberg_non_ai_labelled_human(tmp_path):
    ai = tmp_path / "ai"; ai.mkdir()
    _ai_case(ai, "gut_00", "gutenberg:575", "gutenberg", authorship="human")
    m = manifest.build_manifest(ai, None, now_iso="2026-07-06T00:00:00Z")
    assert m["rows"][0]["label"] == "human"  # explicit, not guessed

def test_scocesle_rows_are_local_only(tmp_path):
    ai = tmp_path / "ai"; ai.mkdir()
    esl = tmp_path / "esl" / "higher proficiency "; esl.mkdir(parents=True)
    (esl / "e1.txt").write_text("An essay written by a human ESL student. " * 10)
    m = manifest.build_manifest(ai, tmp_path / "esl", now_iso="2026-07-06T00:00:00Z")
    esl_rows = [r for r in m["rows"] if r["label"] == "human"]
    assert esl_rows and all(r["license"] == "local_only" for r in esl_rows)

def test_leakage_guard(tmp_path):
    rows = [{"sha256": "abc", "split": "cal"}, {"sha256": "abc", "split": "test"}]
    with pytest.raises(manifest.LeakageError):
        manifest.assert_no_leakage(rows)

def test_license_guard_blocks_local_only(tmp_path):
    rows = [{"license": "local_only"}]
    with pytest.raises(manifest.LicenseError):
        manifest.assert_committable(rows)
