"""capture_one must produce exactly what score_paragraph consumes, and its
offline prediction must MATCH the end-to-end pipeline's primary_category
for the same text (offline == e2e by construction is the whole point)."""
import os

from calibration.v12_validation.capture_signals import capture_one
from detect_v7 import category_scoring

# allow-hardcode: fixed fixture text for a deterministic offline==e2e parity
# test, not a scoring/matching oracle — any coherent paragraph would do here.
_TEXT = (
    "The industrial revolution transformed European society in profound ways. "
    "Factories replaced workshops, and cities grew rapidly as workers migrated. "
    "My grandmother's village in Guangdong still had a hand loom in 1980, which "
    "is why I find the timeline of mechanization so uneven across regions."
) * 3


def test_capture_matches_end_to_end(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN", "1")
    monkeypatch.delenv("DRAFTPROOF_V7_DEEP_SCAN", raising=False)  # quick-scan: no spend
    from calibration.measure_end_to_end import scan_text
    from detect.run import DetectionRunner

    runner = DetectionRunner()
    row, rep = capture_one(runner, _TEXT, label="student_owned")

    assert set(row) >= {"label", "doc_key", "v7_signals", "calibrated_detector_score"}
    offline = category_scoring.score_paragraph(
        row["v7_signals"], row["calibrated_detector_score"],
        has_comparison_text=False, esl_score=None,
    )
    e2e = (rep.get("ai_risk_badge") or {}).get("authorship_breakdown") or {}
    assert offline["primary_category"] == e2e.get("primary_category")


def test_criterion_derived_signals_now_available(monkeypatch):
    """Regression test for the criterion_scores wiring bug (see
    poc/calibration/v12_validation/false_ai_diagnosis.json): before the fix,
    builder.py never threaded criterion_scores into run_v7_breakdown's input
    dict, so every criterion-derived signal was unconditionally
    "unavailable" in production. On this fixture, 4 of the 5 dark signals
    compute directly from criterion_scores and must now report "ok".
    detector_disagreement is NOT criterion-derived (it needs multiple
    detector scores / deep-scan data absent on the quick-scan path) and is
    asserted merely not-unavailable-for-the-wrong-reason is out of scope here.
    """
    monkeypatch.setenv("DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN", "1")
    monkeypatch.delenv("DRAFTPROOF_V7_DEEP_SCAN", raising=False)  # quick-scan: no spend
    from detect.run import DetectionRunner

    runner = DetectionRunner()
    row, _rep = capture_one(runner, _TEXT, label="student_owned")
    status = row["v7_signals"]["signal_status"]

    for signal in (
        "specificity_score",
        "sentence_variance",
        "sentence_smoothness",
        "local_style_shift",
    ):
        assert status[signal] == "ok", f"{signal} still dark: {status[signal]!r}"
