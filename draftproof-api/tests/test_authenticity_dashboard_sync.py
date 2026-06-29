# draftproof-api/tests/test_authenticity_dashboard_sync.py
import hashlib, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "poc" / "detect" / "authenticity_dashboard.py"
COPY = ROOT / "draftproof-api" / "app" / "_composers" / "authenticity_dashboard.py"


def _body(p: Path) -> str:
    # strip the leading docstring/header comment lines so only logic is compared
    text = p.read_text()
    return re.sub(r'^.*?MODEL_VERSION', 'MODEL_VERSION', text, count=1, flags=re.S)


def test_api_copy_matches_poc_logic():
    assert COPY.exists(), "create the verbatim copy"
    assert _body(SRC) == _body(COPY), "API _composers copy drifted from poc source — KEEP IN SYNC"
