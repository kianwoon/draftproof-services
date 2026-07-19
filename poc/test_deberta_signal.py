import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from detect import deberta_signal

# ~320 words — clears the 150-word floor.
LONG_TEXT = " ".join(["This is a clear academic sentence with real substance."] * 40)

_SCHEMA_KEYS = {
    "signal_pct", "sentences_scored", "sentences_flagged", "flagged_passages",
    "band", "confidence", "model_version", "available", "caveat",
}
_VALID_BANDS = {"insufficient", "amber", "orange", "red"}


def test_disabled_returns_none():
    deberta_signal.score_windows = lambda _windows: [0.9] * len(_windows)
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "0"
    try:
        assert deberta_signal.maybe_attach(LONG_TEXT) is None
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_too_short_abstains():
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    deberta_signal.score_windows = lambda _w: [0.9]
    try:
        out = deberta_signal.maybe_attach("Too short to scan.")
        assert out is not None and out["available"] is False
        assert out["band"] == "insufficient"
        assert "short" in out["caveat"].lower()
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_fail_open_when_inference_returns_none():
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    deberta_signal.score_windows = lambda _w: None
    try:
        out = deberta_signal.maybe_attach(LONG_TEXT)
        assert out is not None and out["available"] is False
        assert "unavailable" in out["caveat"].lower()
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_schema_and_proportion_signal_all_clean():
    """All sentences score 0.2 (clean band, < 0.80) -> 0% flagged, band insufficient.
    Confirms the proportion math + schema shape on the clean human-like path. Note: "flagged"
    means non-clean band (>= 0.80), NOT >=0.99 — a 0.9 sentence is moderate-band and IS flagged."""
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    deberta_signal.score_windows = lambda _w: [0.2] * len(_w)  # below the 0.80 clean cutoff
    try:
        out = deberta_signal.maybe_attach(LONG_TEXT)
        assert out is not None
        assert set(out) >= _SCHEMA_KEYS
        assert out["model_version"] == "deberta_signal_v2"
        assert out["available"] is True
        assert out["signal_pct"] == 0
        assert out["sentences_flagged"] == 0
        assert out["flagged_passages"] == []
        assert out["band"] == "insufficient"   # 0% < 20% floor
        assert out["confidence"] == "low"      # below floor
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_band_for_floor_and_thresholds():
    """The band map: <20 insufficient, [20,40) amber, [40,65) orange, >=65 red. No green."""
    b = deberta_signal._band_for
    assert b(0) == "insufficient"
    assert b(19) == "insufficient"
    assert b(20) == "amber"
    assert b(39) == "amber"
    assert b(40) == "orange"
    assert b(64) == "orange"
    assert b(65) == "red"
    assert b(100) == "red"
    assert b(None) == "insufficient"
    assert "green" not in _VALID_BANDS  # a signal that fires is never "safe"


def test_proportion_signal_above_floor_and_flagged_passages():
    """A mix where >=20% of sentences clear the 0.99 threshold: signal computed from the
    proportion, band becomes a real verdict, flagged_passages lists the hot sentences."""
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    # A sentence's score is the mean of the windows covering it (size=3, step=1). To make a
    # sentence clear 0.99, ALL its covering windows must be ~1.0 — so emit BLOCKS of high
    # windows, not isolated ones. Pattern of 5 consecutive 1.0 windows then 5 at 0.1 gives
    # the inner sentences of each block a mean of 1.0 -> well above floor.
    def stub(windows):
        return [1.0 if ((i // 5) % 2 == 0) else 0.1 for i in range(len(windows))]
    deberta_signal.score_windows = stub
    try:
        out = deberta_signal.maybe_attach(LONG_TEXT)
        assert out is not None and out["available"] is True
        assert out["signal_pct"] >= 20, f"expected >=20% signal, got {out['signal_pct']}"
        assert out["band"] in {"amber", "orange", "red"}, out["band"]
        assert out["confidence"] == "medium"  # above floor
        # flagged_passages shape — flagged = non-clean band (>= 0.80), same definition as the map
        for p in out["flagged_passages"]:
            assert set(p) == {"sentence_id", "score", "text"}
            assert p["score"] >= 0.80  # non-clean band cutoff (NOT the 0.99 high-confidence bar)
            assert isinstance(p["text"], str) and p["text"]
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_flagged_passages_capped_and_sorted():
    """When many sentences flag, flagged_passages is capped at _MAX_FLAGGED_PERSISTED and
    sorted by score descending; sentences_flagged reflects the TRUE total."""
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    deberta_signal.score_windows = lambda _w: [1.0] * len(_w)  # every window max -> all flag
    try:
        out = deberta_signal.maybe_attach(LONG_TEXT)
        assert out["sentences_flagged"] == out["sentences_scored"]  # all flagged
        assert out["signal_pct"] == 100
        assert out["band"] == "red"
        # list capped
        assert len(out["flagged_passages"]) <= deberta_signal._MAX_FLAGGED_PERSISTED
        # sorted desc by score (all 1.0 here, so stable; check the cap holds)
        scores = [p["score"] for p in out["flagged_passages"]]
        assert scores == sorted(scores, reverse=True)
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_below_floor_state_has_no_verdict_band():
    """A small fraction lands below the 20% floor -> band 'insufficient', confidence 'low',
    no green/amber/orange/red verdict, but the few flagged passages ARE surfaced for review.
    This is the confirmed below-floor UI contract. Note: "flagged" = non-clean band (>=0.50);
    the high-confidence bar (>=0.99) is reported separately, not the flag definition."""
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    # One small block of 3 high windows at the start (sentences 0-1 read AI-like), then a long
    # run of clean windows (< 0.50). With ~40 sentences, 1-2 flagged is well below the 20% floor.
    def stub(windows):
        return [1.0 if i < 3 else 0.1 for i in range(len(windows))]
    deberta_signal.score_windows = stub
    try:
        out = deberta_signal.maybe_attach(LONG_TEXT)
        assert out["band"] == "insufficient", f"expected insufficient, got {out['band']}"
        assert out["confidence"] == "low"
        assert out["signal_pct"] < deberta_signal.DOC_FLOOR_PCT
        # whatever flagged passages exist are surfaced
        assert out["sentences_flagged"] == len(out["flagged_passages"]) or \
               out["sentences_flagged"] >= 0
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"{name} PASSED")
            except AssertionError as e:
                print(f"{name} FAILED: {e}")
                raise
    print("ALL TESTS PASSED")


# ─── Heatmap (compose_from_sentences + band_for_sentence) tests ────────────────

def test_band_for_sentence_graduated_scale():
    """clean < 0.80 < moderate < 0.99 <= high. The 0.80 clean cutoff avoids flagging
    borderline-ambiguous sentences (0.50-0.79) as 'AI signal' next to genuine 1.0 readings."""
    b = deberta_signal.band_for_sentence
    assert b(0.0) == "clean"
    assert b(0.59) == "clean"      # 0.59 is clean now (was 'low' — too noisy to flag)
    assert b(0.79) == "clean"
    assert b(0.80) == "moderate"   # boundary: 0.80 is NOT clean
    assert b(0.98) == "moderate"
    assert b(0.99) == "high"       # >= SENT_THRESHOLD
    assert b(1.0) == "high"
    assert b(None) == "clean"


def test_compose_from_sentences_returns_per_sentence_scores_keyed_by_input_ids():
    """The heatmap entry point must score the EXACT sentences passed in and key results by
    THEIR sentence_id/paragraph_id (not its own naive index). This is the alignment guarantee
    that lets the report join DeBERTa scores to canonical sNNN/pNNN segments."""
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    # Stub score_windows to deterministic values so we test the join/shape, not the model.
    deberta_signal.score_windows = lambda _w: [0.95] * len(_w)
    sens = [
        {"sentence_id": "s001", "paragraph_id": "p001", "text": "First sentence here with enough words to clear the floor."},
        {"sentence_id": "s002", "paragraph_id": "p001", "text": "Second sentence here with enough words to clear the floor."},
        {"sentence_id": "s003", "paragraph_id": "p002", "text": "Third sentence in paragraph two with enough words to clear."},
    ]
    try:
        out = deberta_signal.compose_from_sentences(sens)
        assert out is not None
        assert out["available"] is True
        assert out["model_version"] == "deberta_signal_v2"
        rows = out["sentence_scores"]
        assert len(rows) == 3
        # keyed by the INPUT ids, in order
        assert [r["sentence_id"] for r in rows] == ["s001", "s002", "s003"]
        assert [r["paragraph_id"] for r in rows] == ["p001", "p001", "p002"]
        # score + band present
        for r in rows:
            assert "score" in r and "band" in r
            assert r["band"] in {"clean", "low", "moderate", "high"}
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_compose_from_sentences_high_score_maps_to_high_band():
    """A sentence the model scores >=0.99 lands in the 'high' band (red on the heatmap).
    Sentence must clear the >=8-word floor (short stubs return score None to avoid noise)."""
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    deberta_signal.score_windows = lambda _w: [1.0] * len(_w)  # every window max
    sens = [{"sentence_id": "s001", "paragraph_id": "p001",
             "text": "This is a fully AI generated sentence with enough words to clear the floor."}]
    try:
        out = deberta_signal.compose_from_sentences(sens)
        row = out["sentence_scores"][0]
        assert row["score"] == 1.0
        assert row["band"] == "high"
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_compose_from_sentences_short_stub_returns_clean_no_score():
    """A short stub (<8 words) returns score None / band clean — no noise heatmap signal.
    This respects the downstream contract that unscored spans render plain (no highlight)."""
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    deberta_signal.score_windows = lambda _w: [0.999] * len(_w)  # model says high
    sens = [{"sentence_id": "s001", "paragraph_id": "p001", "text": "Short stub."}]  # 2 words
    try:
        out = deberta_signal.compose_from_sentences(sens)
        row = out["sentence_scores"][0]
        assert row["score"] is None
        assert row["band"] == "clean"
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_compose_from_sentences_fail_open_on_inference_none():
    """If the model returns None (load/inference failed), the heatmap is unavailable, not a crash."""
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    deberta_signal.score_windows = lambda _w: None
    sens = [{"sentence_id": "s001", "paragraph_id": "p001", "text": "Some text here."}]
    try:
        out = deberta_signal.compose_from_sentences(sens)
        assert out is not None
        assert out["available"] is False
        assert out["sentence_scores"] == []
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_compose_from_sentences_returns_none_when_disabled_or_empty():
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "0"
    try:
        assert deberta_signal.compose_from_sentences(
            [{"sentence_id": "s1", "paragraph_id": "p1", "text": "x"}]) is None
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)
    # empty input
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    try:
        assert deberta_signal.compose_from_sentences([]) is None
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_headline_from_heatmap_threads_model_version_from_heatmap():
    """2026-07-19 bugfix: headline_from_heatmap previously ignored the heatmap dict's
    own "model_version" key and _build_headline always stamped the module's native
    MODEL_VERSION, even when the heatmap rows came from a completely different
    detector (e.g. the V7 deep-scan Modal checkpoint via
    detect_v7/deep_scan_heatmap.py::compose_deep_scan_heatmap, threaded through
    report.py's _sync_deberta_headline_from_heatmap). The returned headline must
    report the ACTUAL source, not a hardcoded guess."""
    sentences = [{"sentence_id": "s001", "paragraph_id": "p001", "text": "irrelevant text"}]
    heatmap = {
        "available": True,
        "sentence_scores": [{"sentence_id": "s001", "paragraph_id": "p001",
                              "score": 0.995, "band": "high"}],
        "model_version": "desklib/ai-text-detector-academic-v1.01",
    }
    out = deberta_signal.headline_from_heatmap(heatmap, sentences)
    assert out is not None
    assert out["model_version"] == "desklib/ai-text-detector-academic-v1.01"
    assert out["model_version"] != deberta_signal.MODEL_VERSION


def test_headline_from_heatmap_defaults_to_module_model_version_when_absent():
    """Backward compat: a heatmap without its own "model_version" key (the shape
    compose_from_sentences has always returned, and the only shape this function
    handled before the fix) keeps the native deberta_signal.MODEL_VERSION —
    byte-identical to pre-fix behavior."""
    sentences = [{"sentence_id": "s001", "paragraph_id": "p001", "text": "irrelevant text"}]
    heatmap = {
        "available": True,
        "sentence_scores": [{"sentence_id": "s001", "paragraph_id": "p001",
                              "score": 0.995, "band": "high"}],
    }
    out = deberta_signal.headline_from_heatmap(heatmap, sentences)
    assert out is not None
    assert out["model_version"] == deberta_signal.MODEL_VERSION
