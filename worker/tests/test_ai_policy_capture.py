"""Phase 1 (docs/plans/policy_risk_external_review_response.md): scan_document must
accept an optional ai_policy kwarg without requiring it -- the API-side batch that
starts SENDING this kwarg ships separately and later (see the plan's Phase 1 ordering
note: the worker must accept the field before the API sends it, or every scan enqueued
in the deploy gap fails with a TypeError). Signature-only check -- no real task
execution (matches this file's sibling test_defence.py's "no real DB/R2/LLM/network
calls" discipline; scan_document itself has no existing execution-level test suite to
extend)."""
import inspect

import app.tasks  # noqa: F401


def test_scan_document_accepts_optional_ai_policy_kwarg():
    sig = inspect.signature(app.tasks.scan_document)
    assert "ai_policy" in sig.parameters
    assert sig.parameters["ai_policy"].default is None


def test_scan_document_still_callable_with_only_the_original_two_args():
    # Backward compatibility: any code path (or in-flight message from before this
    # deploy) that calls scan_document(job_id, text) with no ai_policy must still
    # bind correctly -- ai_policy must be optional, not required.
    sig = inspect.signature(app.tasks.scan_document)
    bound = sig.bind("job-123", "some text")
    bound.apply_defaults()
    assert bound.arguments["ai_policy"] is None
