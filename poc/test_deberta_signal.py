import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from detect import deberta_signal

# ~320 words — clears the 150-word floor.
LONG_TEXT = " ".join(["This is a clear academic sentence with real substance."] * 40)


def _stub_probs(_windows):
    return [0.9] * len(_windows)


def test_disabled_returns_none():
    deberta_signal.score_windows = _stub_probs  # ensure we're on the scoring path, not abstention
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "0"
    try:
        assert deberta_signal.maybe_attach(LONG_TEXT) is None
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_too_short_abstains():
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    deberta_signal.score_windows = _stub_probs
    try:
        out = deberta_signal.maybe_attach("Too short to scan.")
        assert out is not None and out["available"] is False
        assert "short" in out["caveat"].lower()
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_schema_and_band_with_mocked_inference():
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    os.environ.pop("DRAFTPROOF_DEBERTA_CALIBRATOR", None)
    deberta_signal.score_windows = _stub_probs  # 0.9 -> score 90 -> red band
    try:
        out = deberta_signal.maybe_attach(LONG_TEXT)
        assert out is not None
        assert set(out) >= {"score", "band", "confidence", "model_version",
                            "calibrated", "available", "caveat"}
        assert out["available"] is True
        assert out["band"] in {"green", "amber", "orange", "red"}
        assert out["calibrated"] is False  # no calibrator configured
        assert out["score"] == 90.0
        assert out["band"] == "red"
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)
        os.environ.pop("DRAFTPROOF_DEBERTA_CALIBRATOR", None)


def test_fail_open_when_inference_returns_none():
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    deberta_signal.score_windows = lambda _w: None
    try:
        out = deberta_signal.maybe_attach(LONG_TEXT)
        assert out is not None and out["available"] is False
        assert "unavailable" in out["caveat"].lower()
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
