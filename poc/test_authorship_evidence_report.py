"""The scan report dict must carry an authorship_evidence block built from the same
authorship_concern signals + false_positives it already computes."""
import json
from pathlib import Path

import pytest

REPORT = Path(__file__).resolve().parent.parent / "test_output/_content11/scan/draftproof_20260530_185257.json"


@pytest.mark.skipif(not REPORT.exists(), reason="content11 scan fixture not present in this environment")
def test_saved_report_signals_produce_evidence_block():
    from poc.report.authorship_evidence import build_authorship_evidence
    d = json.loads(REPORT.read_text())
    block = build_authorship_evidence(d["authorship_concern"]["signals"], d.get("false_positives"))
    present = {m["signal"] for m in block["present_markers"]}
    thin = {t["signal"] for t in block["thin_signals"]}
    assert "genericity" in present
    assert {"specificity", "source_grounding"} <= thin
