# draftproof-api/tests/test_grounding_diagnosis_sync.py
"""Guards the drift found 2026-07-20 (independent Fable review, filed as a
follow-up during the policy_risk ordering-floor fix): the API's read-time
backfill copy (app/_composers/grounding_diagnosis.py, used by
app/policy_enrich.py to enrich older rewrite reports) fell out of sync with
poc/detect/grounding_diagnosis.py after commit 0ea6c6e9 ("fix(scan): lead the
report headline with the grounding gap, not surface patterning") added
HEADLINE_INELIGIBLE_DRIVERS + the _headline_pool filter to the poc source
only. The API copy kept picking primary_driver by raw weight*score
contribution, so older reports enriched via the API path could headline
llm_patterning ("vary structure and phrasing") where scan-time reports would
not. Mirrors test_policy_risk_sync.py / test_authenticity_dashboard_sync.py's
pattern — those only cover their own composers, which is why this drift went
uncaught."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "poc" / "detect" / "grounding_diagnosis.py"
COPY = ROOT / "draftproof-api" / "app" / "_composers" / "grounding_diagnosis.py"


def _body(p: Path) -> str:
    # strip the leading docstring/header comment lines so only logic is compared
    text = p.read_text()
    return re.sub(r'^.*?MODEL_VERSION', 'MODEL_VERSION', text, count=1, flags=re.S)


def test_api_copy_matches_poc_logic():
    assert COPY.exists(), "create the verbatim copy"
    assert _body(SRC) == _body(COPY), "API _composers copy drifted from poc source — KEEP IN SYNC"
