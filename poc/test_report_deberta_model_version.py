"""Regression test for report.report.resolve_deberta_model_version.

2026-07-19 bugfix: report.py's _sync_deberta_headline_from_heatmap hardcoded the
badge's ai_signal_deberta.model_version as "deberta_signal_v2" (poc/detect/
deberta_signal.py's own MODEL_VERSION constant) even when the per-sentence heatmap
it had just rebuilt the headline from came from the SEPARATE V7 deep-scan Modal
pipeline (poc/detect_v7/deep_scan_heatmap.py), not deberta_signal.py at all. This
made it impossible to tell, from the report JSON alone, which of the two
DeBERTa-family sources actually produced a given ai_signal_deberta reading.

This locks in the fix: resolve_deberta_model_version picks the model_version to
stamp based on which heatmap source (report.report's heatmap_source tracking
variable: "fakespot" vs "deep_scan") actually produced the LAST
_compute_deberta_heatmap() call, so a deep-scan reading is never mislabeled with
the fakespot detector's own version string.
"""
from report.report import resolve_deberta_model_version
from detect.deberta_signal import MODEL_VERSION as FAKESPOT_MODEL_VERSION

_V7_CHECKPOINT = "desklib/ai-text-detector-academic-v1.01"


def test_fakespot_source_keeps_native_model_version():
    assert resolve_deberta_model_version(
        "fakespot", None, FAKESPOT_MODEL_VERSION,
    ) == FAKESPOT_MODEL_VERSION


def test_deep_scan_source_uses_the_real_v7_checkpoint():
    got = resolve_deberta_model_version("deep_scan", _V7_CHECKPOINT, FAKESPOT_MODEL_VERSION)
    assert got == _V7_CHECKPOINT
    assert got != FAKESPOT_MODEL_VERSION


def test_deep_scan_source_never_falls_back_to_fakespot_label_when_checkpoint_unknown():
    """Even when the V7 module could not report its own checkpoint id (e.g. an
    older/mocked Modal endpoint response omitting "checkpoint" and no
    DRAFTPROOF_MODAL_CHECKPOINT env tag set), a deep-scan-sourced reading must
    NEVER be silently mislabeled with the OTHER detector's version string — that
    is exactly the observability bug being fixed here."""
    got = resolve_deberta_model_version("deep_scan", None, FAKESPOT_MODEL_VERSION)
    assert got != FAKESPOT_MODEL_VERSION
    # Falls back to the heatmap_source label itself — a real identifier already
    # used elsewhere in report.py (signal_highlight_source), not an invented string.
    assert got == "deep_scan"
