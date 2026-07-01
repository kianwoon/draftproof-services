import os
import pathlib
import pickle
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from detect import deberta_signal

# ~320 words — clears the 150-word floor.
LONG_TEXT = " ".join(["This is a clear academic sentence with real substance."] * 40)


# Stub calibrators (module-level so they're picklable). predict() mirrors the
# apply_isotonic convention: takes a list, returns a list of floats.
class _StepCalibrator:
    """Mimics the SCoCESLE-fit calibrator: everything < 0.854 -> 0.0, then a cliff."""
    def predict(self, xs):
        return [0.0 if float(x) < 0.854 else (1.0 if float(x) >= 0.998 else 0.5) for x in xs]


class _IdentityCalibrator:
    """Sane monotonic calibrator: passes the raw score through unchanged."""
    def predict(self, xs):
        return [float(x) for x in xs]


def _write_cal_stub(obj):
    """Write a pickled calibrator stub to a real temp file so the module's
    os.path.exists() check passes and load_calibrator() deserializes it."""
    fd, path = tempfile.mkstemp(suffix=".pkl")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(obj, f)
    return path


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


# Stub calibrators (module-level so they're picklable). predict() mirrors the
# apply_isotonic convention: takes a list, returns a list of floats.
class _StepCalibrator:
    """Mimics the SCoCESLE-fit calibrator: everything < 0.854 -> 0.0, then a cliff."""
    def predict(self, xs):
        return [0.0 if float(x) < 0.854 else (1.0 if float(x) >= 0.998 else 0.5) for x in xs]


class _IdentityCalibrator:
    """Sane monotonic calibrator: passes the raw score through unchanged."""
    def predict(self, xs):
        return [float(x) for x in xs]


def _write_cal_stub(obj):
    """Write a pickled calibrator stub to a real temp file so the module's
    os.path.exists() check passes and load_calibrator() deserializes it."""
    fd, path = tempfile.mkstemp(suffix=".pkl")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(obj, f)
    return path


def test_floor_guard_recovers_raw_when_calibrator_zeros_a_flagged_doc():
    """Repro of the production 0%-bug: a doc the raw model scores ~0.7 (clearly
    AI-flagged) is floored to 0.0 by a step-function calibrator that has no
    training support below the AI/human separation gap. The guard must fall back
    to the raw score and mark the field, instead of silently reporting 0%."""
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    cal_path = _write_cal_stub(_StepCalibrator())
    os.environ["DRAFTPROOF_DEBERTA_CALIBRATOR"] = cal_path
    deberta_signal.score_windows = lambda _w: [0.7] * len(_w)  # raw 0.7
    try:
        out = deberta_signal.maybe_attach(LONG_TEXT)
        assert out is not None and out["available"] is True
        assert out["score"] == 70.0, f"expected raw fallback 70.0, got {out['score']}"
        assert out["band"] == "red", f"70 -> red band (>=65), got {out['band']}"
        assert out["confidence"] == "low"  # downgraded — calibrator was bypassed
        assert "bypassed" in out["caveat"].lower()
        assert out["calibrated"] is True  # a calibrator WAS configured (and bypassed)
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)
        os.environ.pop("DRAFTPROOF_DEBERTA_CALIBRATOR", None)
        os.unlink(cal_path)


def test_floor_guard_does_not_trip_when_calibrator_is_sane():
    """A well-behaved calibrator (monotonic, no floor) is used as-is — the guard
    must not override it. Sanity check that the guard is narrowly scoped."""
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    cal_path = _write_cal_stub(_IdentityCalibrator())
    os.environ["DRAFTPROOF_DEBERTA_CALIBRATOR"] = cal_path
    deberta_signal.score_windows = lambda _w: [0.9] * len(_w)  # raw 0.9
    try:
        out = deberta_signal.maybe_attach(LONG_TEXT)
        assert out is not None
        assert out["score"] == 90.0  # raw passed through, no guard
        assert out["confidence"] == "medium"  # calibrated path honored
        assert "calibrated on DraftProof ESL corpus" in out["caveat"]
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)
        os.environ.pop("DRAFTPROOF_DEBERTA_CALIBRATOR", None)
        os.unlink(cal_path)


def test_floor_guard_does_not_trip_on_genuinely_low_raw():
    """A truly human document (raw 0.05) should still calibrate to ~0 — the guard
    only fires when raw >= 0.5. Ensures we don't false-positive low scores."""
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    cal_path = _write_cal_stub(_StepCalibrator())
    os.environ["DRAFTPROOF_DEBERTA_CALIBRATOR"] = cal_path
    deberta_signal.score_windows = lambda _w: [0.05] * len(_w)
    try:
        out = deberta_signal.maybe_attach(LONG_TEXT)
        assert out is not None
        # Step calibrator floors 0.05 -> 0.0, which is the CORRECT low-AI reading
        # here. Guard must NOT trip (raw 0.05 < 0.5 threshold), so calibrator is
        # trusted and the score reflects its 0.0 output.
        assert out["score"] == 0.0
        assert out["confidence"] == "medium"  # calibrator trusted
        assert "calibrated on DraftProof ESL corpus" in out["caveat"]
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)
        os.environ.pop("DRAFTPROOF_DEBERTA_CALIBRATOR", None)
        os.unlink(cal_path)


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
