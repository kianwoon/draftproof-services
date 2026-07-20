"""Guard: poc/detect/policy_risk.py and draftproof-api/app/_composers/policy_risk.py
are hand-maintained duplicate copies (the API container cannot import poc, so the
module is copied verbatim). If they drift, the API's policy-risk composer silently
diverges from the scan pipeline's detector. This test fails loudly instead of
letting that drift ship unnoticed -- see CLAUDE.md architecture notes.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
POC_COPY = REPO_ROOT / "poc" / "detect" / "policy_risk.py"
API_COPY = REPO_ROOT / "draftproof-api" / "app" / "_composers" / "policy_risk.py"


def test_policy_risk_copies_are_byte_identical():
    assert POC_COPY.exists(), f"missing {POC_COPY}"
    assert API_COPY.exists(), f"missing {API_COPY}"

    poc_bytes = POC_COPY.read_bytes()
    api_bytes = API_COPY.read_bytes()

    assert poc_bytes == api_bytes, (
        "poc/detect/policy_risk.py and draftproof-api/app/_composers/policy_risk.py "
        "have drifted apart. These are hand-synced duplicate copies (the API "
        "container cannot import poc). Edit one, then copy it byte-for-byte over "
        "the other so both stay identical:\n"
        f"  cp {POC_COPY} {API_COPY}\n"
        "(or the reverse, whichever has the intended change)."
    )
