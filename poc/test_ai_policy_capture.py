"""Phase 1 (docs/plans/policy_risk_external_review_response.md): optional
assignment-level ai_policy capture, threaded from scan submission through to
report_to_dict()'s document_context. This batch (worker/poc side) ships before
the API/frontend batch that actually sends the field -- see the plan's Phase 1
ordering note. Nothing here changes scoring; ai_policy is presentation-only
metadata (Phase 2, not built yet, will use it to select which policy_risk lens
to headline).
"""
import inspect

from report.models import AI_POLICY_VALUES, DraftReport, Tier
from report.report import report_to_dict
from report.builder import ReportBuilder


def _report(ai_policy=None, original_text="Some submitted text."):
    return DraftReport(
        overall_tier=Tier.CLEAN,
        finding_count=0,
        findings_by_tier={},
        original_text=original_text,
        ai_policy=ai_policy,
    )


# ── report_to_dict() normalization (fast, no ReportBuilder) ──────────────────

def test_document_context_defaults_to_unknown_when_absent():
    # The common case today: no scan has ever set ai_policy. Must be "unknown",
    # not None/missing -- older reports and new absent-field reports read the same.
    data = report_to_dict(_report(ai_policy=None))
    assert data["document_context"]["ai_policy"] == "unknown"


def test_document_context_passes_through_a_valid_value():
    data = report_to_dict(_report(ai_policy="prohibited"))
    assert data["document_context"]["ai_policy"] == "prohibited"


def test_document_context_all_five_values_round_trip():
    for value in AI_POLICY_VALUES:
        data = report_to_dict(_report(ai_policy=value))
        assert data["document_context"]["ai_policy"] == value


def test_document_context_never_fabricates_an_unrecognized_value():
    # Defence in depth: even if something upstream sends a typo'd/garbage value
    # (the real enum validation lives in the API's Pydantic schema, batch 2),
    # report_to_dict() must not surface it verbatim -- normalize to "unknown"
    # rather than pass through an unvalidated string.
    data = report_to_dict(_report(ai_policy="not_a_real_policy"))
    assert data["document_context"]["ai_policy"] == "unknown"


def test_document_context_word_count_and_sentence_count_unchanged():
    # Parity check: adding ai_policy must not disturb the two existing keys.
    data = report_to_dict(_report(ai_policy="editing_only", original_text="one two three"))
    assert data["document_context"]["word_count"] == 3
    assert data["document_context"]["sentence_count"] == 0


# ── ReportBuilder.set_meta() storage (no .build(), instant) ──────────────────

def test_set_meta_stores_ai_policy_on_the_builder():
    builder = ReportBuilder().set_meta(original_text="x", ai_policy="collaboration_allowed")
    assert builder._ai_policy == "collaboration_allowed"


def test_set_meta_ai_policy_defaults_to_none():
    # Existing call sites that don't pass ai_policy (there are several across
    # the test suite, e.g. test_ai_rating_layer.py) must keep working unchanged.
    builder = ReportBuilder().set_meta(original_text="x")
    assert builder._ai_policy is None


# ── Full builder.build() wiring (real, small DetectionRunner pass) ───────────

def test_full_build_threads_ai_policy_into_the_final_report_json():
    from detect.run import DetectionRunner
    from detect.document_structure import normalize_submitted_text

    text = normalize_submitted_text(
        "The committee reviewed the proposal carefully. "
        "Several members raised concerns about the budget allocation. "
        "After discussion, the group agreed to revisit the numbers next quarter."
    )
    runner = DetectionRunner()
    detection_result = runner.run_all(text)

    builder = ReportBuilder()
    builder.add_detection_report(detection_result)
    if getattr(detection_result, "postprocess_results", None):
        builder.add_postprocess_results(detection_result.postprocess_results)
    builder.set_meta(scan_time=0.0, original_text=text, ai_policy="allowed_with_declaration")
    report = builder.build()

    assert report.ai_policy == "allowed_with_declaration"
    data = report_to_dict(report)
    assert data["document_context"]["ai_policy"] == "allowed_with_declaration"


# ── Signature guards for the thin pass-through layers ────────────────────────
# detect_pipeline.run_detect and worker/app/tasks.scan_document are one-line
# kwarg pass-throughs with no logic of their own; a full integration test would
# require mocking DB/Redis/Celery infra this codebase doesn't already have for
# scan_document. A signature check is a cheap guard against a future refactor
# silently dropping the parameter.

def test_run_detect_accepts_ai_policy_kwarg():
    from detect_pipeline import run_detect
    sig = inspect.signature(run_detect)
    assert "ai_policy" in sig.parameters
    assert sig.parameters["ai_policy"].default is None
