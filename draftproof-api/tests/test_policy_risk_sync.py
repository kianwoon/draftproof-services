# draftproof-api/tests/test_policy_risk_sync.py
"""Guards the exact drift that shipped 2026-07-20: the API's read-time backfill
copy (app/_composers/policy_risk.py, used by app/policy_enrich.py to enrich older
rewrite reports) fell out of sync with poc/detect/policy_risk.py after the
ai_allowed/ai_restricted ordering floor was added to the poc source only. Older
reports enriched via the API path kept showing the exact score inversion the fix
was meant to eliminate. Mirrors test_authenticity_dashboard_sync.py's pattern —
that test only covers authenticity_dashboard, which is why this drift went
uncaught."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "poc" / "detect" / "policy_risk.py"
COPY = ROOT / "draftproof-api" / "app" / "_composers" / "policy_risk.py"


def _body(p: Path) -> str:
    # strip the leading docstring/header comment lines so only logic is compared
    text = p.read_text()
    return re.sub(r'^.*?MODEL_VERSION', 'MODEL_VERSION', text, count=1, flags=re.S)


def test_api_copy_matches_poc_logic():
    assert COPY.exists(), "create the verbatim copy"
    assert _body(SRC) == _body(COPY), "API _composers copy drifted from poc source — KEEP IN SYNC"
