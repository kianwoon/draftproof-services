import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from detect.deberta_calibrate import fit_isotonic, apply_isotonic, map_to_band


def test_isotonic_monotonic_nondecreasing():
    raw = [0.1, 0.2, 0.8, 0.9, 0.15]
    labels = [1, 1, 0, 0, 1]
    cal = fit_isotonic(raw, labels)
    out = apply_isotonic([0.1, 0.5, 0.9], cal)
    assert out == sorted(out), f"expected non-decreasing, got {out}"


def test_apply_isotonic_clips_to_unit_interval():
    cal = fit_isotonic([0.2, 0.8], [1, 0])
    out = apply_isotonic([-1.0, 0.5, 2.0], cal)
    assert all(0.0 <= x <= 1.0 for x in out), f"out of [0,1]: {out}"


def test_map_to_band_traffic_light_legend():
    # Composite's displayed legend is green/amber/orange/red (frontend reportHelpers.js +
    # i18n tiers), thresholds 32/48/65 — DeBERTa MUST use the same keys+cutoffs for comparability.
    assert map_to_band(5) == "green"
    assert map_to_band(25) == "green"
    assert map_to_band(40) == "amber"
    assert map_to_band(60) == "orange"
    assert map_to_band(90) == "red"


def test_map_to_band_boundaries():
    # Exact cutoff edges: <32 green, [32,48) amber, [48,65) orange, >=65 red.
    assert map_to_band(31) == "green"
    assert map_to_band(32) == "amber"
    assert map_to_band(47) == "amber"
    assert map_to_band(48) == "orange"
    assert map_to_band(64) == "orange"
    assert map_to_band(65) == "red"


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
