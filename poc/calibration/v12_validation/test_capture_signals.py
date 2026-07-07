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
