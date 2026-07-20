"""Phase 1 batch 2 (docs/plans/policy_risk_external_review_response.md):
ScanRequest.ai_policy is the real enum-validation choke point -- worker/poc
(batch 1) only defensively re-checks at the last serialization step
(report_to_dict), it doesn't reject a bad value. This is where a typo'd/
malicious value actually gets rejected before it ever reaches the DB."""
import pytest
from pydantic import ValidationError

from app.models import ScanRequest, AiPolicy


def test_absent_ai_policy_defaults_to_none():
    req = ScanRequest(document_id="paste", text="hello")
    assert req.ai_policy is None


def test_all_five_canonical_values_are_accepted():
    for value in (
        "prohibited", "editing_only", "allowed_with_declaration",
        "collaboration_allowed", "unknown",
    ):
        req = ScanRequest(document_id="paste", text="hello", ai_policy=value)
        assert req.ai_policy == value


def test_unrecognized_value_is_rejected_not_silently_dropped():
    with pytest.raises(ValidationError):
        ScanRequest(document_id="paste", text="hello", ai_policy="my_school_allows_it")


def test_null_is_accepted_as_not_offered():
    req = ScanRequest(document_id="paste", text="hello", ai_policy=None)
    assert req.ai_policy is None
